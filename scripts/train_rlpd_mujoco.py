#!/usr/bin/env python3
"""
RLPD training script for MuJoCo simulation with Human-in-the-Loop (SpaceMouse).

This is a state-based variant of train_rlpd.py designed for MuJoCo environments.
It uses the actor-learner separation via agentlace for distributed training,
and ReplayBufferDataStore (not the memory-efficient pixel variant).

Usage:
    # 1) Start the learner (with GPU):
    python scripts/train_rlpd_mujoco.py --exp_name mujoco_insertion --learner \
        --demo_path demo_data/insertion_demos.pkl --checkpoint_path output/mujoco_insertion

    # 2) Start the actor (with SpaceMouse + MuJoCo visualisation):
    python scripts/train_rlpd_mujoco.py --exp_name mujoco_insertion --actor --ip localhost \
        --checkpoint_path output/mujoco_insertion_actor
"""

from __future__ import annotations

import copy
import csv
import glob
import os
import pickle as pkl
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    parent_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str in sys.path:
        sys.path.remove(repo_root_str)
    sys.path.insert(0, repo_root_str)

    sibling_sim = parent_root / "sim"
    if sibling_sim.exists():
        sibling_sim_str = str(sibling_sim)
        if sibling_sim_str in sys.path:
            sys.path.remove(sibling_sim_str)
        sys.path.insert(1, sibling_sim_str)

# On desktop Ubuntu setups, JAX's default GPU memory preallocation can starve
# GNOME / Xorg / Chrome of VRAM and trigger a black screen when the display
# reconfigures or wakes up. Keep the default conservative unless the user has
# explicitly chosen otherwise in their shell environment.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.data.data_store import ReplayBufferDataStore
from serl_launcher.utils.launcher import (
    make_sac_state_agent,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from scripts.experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path(s) to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_enum(
    "reset_perturbation_preset",
    "default",
    ["default", "yaw_only"],
    "Reset perturbation preset for training, actor, and eval environments. "
    "'default' keeps the current XY+yaw reset distribution; "
    "'yaw_only' zeros reset XY and keeps only reset yaw.",
)
flags.DEFINE_float(
    "reset_yaw_range_override",
    -1.0,
    "Optional reset yaw range override. Negative keeps the preset default.",
)
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean(
    "eval_sample_task_case",
    False,
    "Sample XY+yaw task cases on each eval reset for transfer testing.",
)
flags.DEFINE_enum(
    "eval_task_case_range_preset",
    "aligned",
    ["aligned", "wide", "yaw_only"],
    "Task-case sampling range preset for eval. "
    "'aligned' matches the training reset distribution; "
    "'wide' restores the broader XY+yaw transfer range; "
    "'yaw_only' samples only hole yaw and keeps task-case XY at zero.",
)
flags.DEFINE_float(
    "eval_task_case_yaw_range_override",
    -1.0,
    "Optional eval task-case yaw range override. Negative keeps the preset default.",
)
flags.DEFINE_boolean(
    "eval_zero_reset_perturbation",
    False,
    "Zero reset XY and reset yaw only for eval environments. Useful for pure hole yaw tests.",
)
flags.DEFINE_boolean(
    "eval_disable_nric",
    False,
    "Disable NRIC only for eval environments to isolate control-layer effects.",
)
flags.DEFINE_boolean(
    "eval_zero_nullspace",
    False,
    "Set nullspace stiffness/damping to zero only for eval environments.",
)
flags.DEFINE_boolean(
    "eval_transfer_diagnostics",
    False,
    "Run strict default-only transfer diagnostics: bucketed eval, policy "
    "action consistency probe, and closed-loop invariance probe.",
)
flags.DEFINE_integer(
    "eval_transfer_probe_num_states",
    12,
    "Number of successful default states to probe for transfer diagnostics.",
)
flags.DEFINE_integer(
    "eval_transfer_probe_horizon",
    40,
    "Fixed-action closed-loop probe horizon for transfer diagnostics.",
)
flags.DEFINE_float(
    "eval_realtime_factor",
    0.0,
    "Eval playback speed relative to MuJoCo simulated time. "
    "0 disables sleeps, 1.0 is realtime, >1.0 is slower-than-realtime, "
    "and 0<factor<1.0 is faster-than-realtime with throttling.",
)
flags.DEFINE_string(
    "eval_video_path",
    None,
    "Optional MP4 output path for policy eval rollouts.",
)
flags.DEFINE_float("eval_video_fps", 30.0, "Eval MP4 frame rate.")
flags.DEFINE_integer("eval_video_width", 1280, "Eval MP4 frame width.")
flags.DEFINE_integer("eval_video_height", 720, "Eval MP4 frame height.")
flags.DEFINE_string(
    "eval_video_camera",
    "tool_follow_camera",
    "Optional MuJoCo camera name for eval MP4. Defaults to tool_follow_camera.",
)
flags.DEFINE_boolean("render_actor", True, "Render the MuJoCo actor environment.")
flags.DEFINE_float(
    "actor_render_fps",
    20.0,
    "Actor viewer refresh rate. Use 0 to disable active viewer refresh.",
)
flags.DEFINE_enum(
    "actor_viewer",
    "mujoco",
    ["mujoco", "meshcat", "none"],
    "Actor viewer backend.",
)
flags.DEFINE_integer(
    "actor_stats_timeout_ms",
    10000,
    "Timeout for actor -> learner stats requests.",
)
flags.DEFINE_float(
    "actor_stats_backoff_s",
    30.0,
    "Wall-clock backoff after actor stats request failures.",
)
flags.DEFINE_boolean(
    "actor_run_until_learner_done",
    True,
    "Keep actor sampling past config.max_steps until the learner reports done.",
)
flags.DEFINE_float(
    "learner_done_grace_s",
    60.0,
    "Seconds for the learner request server to stay alive after training ends, "
    "so actors can receive learner_done=True and exit cleanly.",
)
flags.DEFINE_string(
    "metrics_dir",
    None,
    "Optional local metrics output directory. Defaults to checkpoint_path/metrics.",
)
flags.DEFINE_boolean(
    "plot_metrics_on_exit",
    True,
    "Generate local reward/success/error plots when learner exits normally.",
)
flags.DEFINE_boolean("debug", False, "Debug mode (disables wandb logging).")

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def optional_flag_float(value: float) -> Optional[float]:
    """Convert a negative sentinel float flag into None."""
    if value is None or value < 0.0:
        return None
    return float(value)


def load_transition_file(path: str) -> list[Any]:
    """Load a pickle transition list with a clear error for empty demo files."""
    demo_path = Path(path)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo file does not exist: {demo_path}")
    if demo_path.stat().st_size == 0:
        raise ValueError(
            "Demo file is empty: "
            f"{demo_path}. Please record successful demos again before training."
        )

    try:
        with demo_path.open("rb") as file:
            transitions = pkl.load(file)
    except EOFError as exc:
        raise ValueError(
            "Demo file is incomplete or corrupted: "
            f"{demo_path}. Please record successful demos again."
        ) from exc

    if not isinstance(transitions, list):
        raise TypeError(
            f"Demo file must contain a list of transitions, got {type(transitions)!r}: "
            f"{demo_path}"
        )
    return transitions


def _space_matches_value(
    space: gym.Space,
    value: Any,
    *,
    field_path: str,
    source_label: str,
) -> None:
    """Raise a clear error when saved transition data mismatches env schema."""
    if isinstance(space, gym.spaces.Box):
        array = np.asarray(value)
        if array.shape != space.shape:
            raise ValueError(
                "Saved transition schema does not match the current environment. "
                f"{source_label}: expected {field_path} shape {space.shape}, "
                f"got {array.shape}. "
                "This usually means the data was recorded before the observation "
                "change to `pose_error_body + twist_body + wrench_control_body`. "
                "Please re-record demos and use a fresh checkpoint directory "
                "without old `buffer/` or `demo_buffer/` contents."
            )
        return

    if isinstance(space, gym.spaces.Dict):
        if not isinstance(value, dict):
            raise ValueError(
                "Saved transition schema does not match the current environment. "
                f"{source_label}: expected dict-like data at {field_path}, "
                f"got {type(value)!r}. "
                "Please re-record demos and use a fresh checkpoint directory."
            )
        for key, subspace in space.spaces.items():
            if key not in value:
                raise ValueError(
                    "Saved transition schema does not match the current environment. "
                    f"{source_label}: missing key {field_path}/{key}. "
                    "Please re-record demos and use a fresh checkpoint directory."
                )
            _space_matches_value(
                subspace,
                value[key],
                field_path=f"{field_path}/{key}",
                source_label=source_label,
            )
        return


def validate_transition_schema(
    transition: dict[str, Any],
    *,
    observation_space: gym.Space,
    action_space: gym.Space,
    source_label: str,
) -> None:
    """Validate one stored transition against the current env I/O schema."""
    required_keys = ("observations", "next_observations", "actions")
    for key in required_keys:
        if key not in transition:
            raise ValueError(
                "Saved transition schema is incomplete. "
                f"{source_label}: missing top-level key {key!r}."
            )
    _space_matches_value(
        observation_space,
        transition["observations"],
        field_path="observations",
        source_label=source_label,
    )
    _space_matches_value(
        observation_space,
        transition["next_observations"],
        field_path="next_observations",
        source_label=source_label,
    )
    _space_matches_value(
        action_space,
        transition["actions"],
        field_path="actions",
        source_label=source_label,
    )


def insert_validated_transitions(
    *,
    replay_buffer,
    transitions: list[Any],
    observation_space: gym.Space,
    action_space: gym.Space,
    source_label: str,
) -> None:
    """Validate transitions before inserting so schema errors are easy to read."""
    for idx, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            raise ValueError(
                f"{source_label}: transition {idx} is {type(transition)!r}, expected dict."
            )
        validate_transition_schema(
            transition,
            observation_space=observation_space,
            action_space=action_space,
            source_label=f"{source_label} transition {idx}",
        )
        replay_buffer.insert(transition)


def resolve_metrics_dir() -> Optional[Path]:
    """Return the local metrics directory for learner-side CSV/PNG outputs."""
    if FLAGS.metrics_dir:
        return Path(FLAGS.metrics_dir).expanduser()
    if FLAGS.checkpoint_path:
        return Path(FLAGS.checkpoint_path).expanduser() / "metrics"
    return None


def build_learner_ip_candidates(ip: str) -> list[str]:
    """Return a small ordered list of learner endpoints for actor retries.

    Local runs are common in this repo. When the user passes a machine IP
    (for example a VPN / tailscale address) from the same host, that route may
    not loop back reliably. Including localhost as a fallback keeps same-host
    actor/learner runs robust without affecting true remote runs.
    """
    candidates: list[str] = []
    normalized = ip.strip()
    if normalized:
        candidates.append(normalized)

    localhost_aliases = ("localhost", "127.0.0.1", "::1")
    if normalized not in localhost_aliases:
        candidates.extend(localhost_aliases[:2])

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def create_actor_trainer_client(*, data_stores: dict[str, Any]) -> TrainerClient:
    """Connect an actor to the learner with localhost fallback retries."""
    target_ips = build_learner_ip_candidates(FLAGS.ip)
    retry_sleep_s = 2.0
    last_warning_time = 0.0

    while True:
        last_errors: list[str] = []
        for target_ip in target_ips:
            try:
                client = TrainerClient(
                    "actor_env",
                    target_ip,
                    make_trainer_config(),
                    data_stores=data_stores,
                    wait_for_server=False,
                    timeout_ms=FLAGS.actor_stats_timeout_ms,
                )
                if target_ip != FLAGS.ip:
                    print_green(
                        "Actor connected to learner via "
                        f"{target_ip} (requested --ip {FLAGS.ip})."
                    )
                return client
            except Exception as exc:
                last_errors.append(f"{target_ip}: {exc}")

        now = time.monotonic()
        if now - last_warning_time >= retry_sleep_s:
            print(
                "[WARN] actor could not reach learner yet. "
                f"Tried {', '.join(target_ips)}. "
                f"Retrying in {retry_sleep_s:.1f}s. "
                f"Last errors: {' | '.join(last_errors)}",
                flush=True,
            )
            if FLAGS.ip not in ("localhost", "127.0.0.1"):
                print(
                    "[INFO] If actor and learner are on the same machine, "
                    "using --ip localhost is recommended.",
                    flush=True,
                )
            last_warning_time = now
        time.sleep(retry_sleep_s)


class EpisodeMetricsWriter:
    """Append actor episode metrics to a local CSV file."""

    FIELDNAMES = (
        "learner_step",
        "wall_time",
        "episode_return",
        "success",
        "failure",
        "xy_error",
        "z_gap",
        "rot_error",
        "insertion_depth",
        "intervention_count",
        "intervention_steps",
    )

    def __init__(self, metrics_dir: Path) -> None:
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.metrics_dir / "episode_metrics.csv"
        needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        self._file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if needs_header:
            self._writer.writeheader()
            self._file.flush()

    def write(self, *, learner_step: int, payload: dict) -> bool:
        environment = payload.get("environment")
        if not isinstance(environment, dict):
            return False
        episode = environment.get("episode")
        if not isinstance(episode, dict):
            return False

        row = {
            "learner_step": int(learner_step),
            "wall_time": time.time(),
            "episode_return": episode.get("return", ""),
            "success": int(bool(environment.get("success", False))),
            "failure": int(bool(environment.get("failure", False))),
            "xy_error": environment.get("xy_error", ""),
            "z_gap": environment.get("z_gap", ""),
            "rot_error": environment.get("rot_error", ""),
            "insertion_depth": environment.get("insertion_depth", ""),
            "intervention_count": episode.get("intervention_count", ""),
            "intervention_steps": episode.get("intervention_steps", ""),
        }
        self._writer.writerow(row)
        self._file.flush()
        return True

    def close(self) -> None:
        self._file.close()


def iter_env_chain(env):
    """Yield an env and its nested Gymnasium wrappers from outer to inner."""
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = getattr(current, "env", None)


def get_raw_env(env):
    """Return the innermost unwrapped Gymnasium env when available."""
    return getattr(env, "unwrapped", env)


def find_wrapper_by_name(env, class_name: str):
    """Return the first wrapper whose class name matches ``class_name``."""
    for wrapped_env in iter_env_chain(env):
        if wrapped_env.__class__.__name__ == class_name:
            return wrapped_env
    return None


def build_policy_observation_from_raw_env(env, raw_env, raw_obs: Optional[dict] = None):
    """Build the policy-facing observation from a raw env observation."""
    raw_obs = raw_env._get_obs() if raw_obs is None else raw_obs
    obs_wrapper = find_wrapper_by_name(env, "MujocoSERLObsWrapper")
    policy_obs = raw_obs if obs_wrapper is None else obs_wrapper.observation(raw_obs)
    return {key: np.asarray(value)[None] for key, value in policy_obs.items()}


TRANSFER_BUCKET_ORDER = (
    "default",
    "small_xy",
    "medium_xy",
    "large_xy",
    "small_yaw",
    "medium_yaw",
    "large_yaw",
    "combined_xy_yaw",
)

TRANSFER_BUCKET_LABELS = {
    "default": "default",
    "small_xy": "small XY [0, 2mm]",
    "medium_xy": "medium XY (2, 5]mm",
    "large_xy": "large XY (5, 10]mm",
    "small_yaw": "small yaw [0, 5deg]",
    "medium_yaw": "medium yaw (5, 10]deg",
    "large_yaw": "large yaw (10, 15]deg",
    "combined_xy_yaw": "combined XY+yaw",
}


def sample_task_case_for_bucket(
    raw_env,
    bucket: str,
    rng: np.random.Generator,
    *,
    task_case_range_preset: str = "aligned",
    task_case_yaw_range_override: Optional[float] = None,
):
    """Sample a deterministic-style eval case from one transfer bucket."""
    if bucket == "default":
        return None

    xy_bucket_limits = {
        "small_xy": (0.0, 0.002),
        "medium_xy": (0.002, 0.005),
        "large_xy": (0.005, 0.010),
    }
    yaw_bucket_limits = {
        "small_yaw": (0.0, np.deg2rad(5.0)),
        "medium_yaw": (np.deg2rad(5.0), np.deg2rad(10.0)),
        "large_yaw": (np.deg2rad(10.0), np.deg2rad(15.0)),
    }

    xy = np.zeros(2, dtype=np.float64)
    yaw = 0.0

    if bucket in xy_bucket_limits:
        radius_low, radius_high = xy_bucket_limits[bucket]
        radius = rng.uniform(radius_low, radius_high)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        xy = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    elif bucket in yaw_bucket_limits:
        yaw_low, yaw_high = yaw_bucket_limits[bucket]
        yaw_mag = rng.uniform(yaw_low, yaw_high)
        yaw = float(yaw_mag * rng.choice([-1.0, 1.0]))
    elif bucket == "combined_xy_yaw":
        if task_case_range_preset == "yaw_only":
            yaw_high = task_case_yaw_range_override
            if yaw_high is None or yaw_high <= 0.0:
                yaw_high = float(raw_env.insertion_cfg.eval_case_yaw_range)
            yaw = float(rng.uniform(-yaw_high, yaw_high))
        else:
            radius = rng.uniform(0.002, 0.010)
            angle = rng.uniform(0.0, 2.0 * np.pi)
            xy = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            yaw_mag = rng.uniform(np.deg2rad(5.0), np.deg2rad(15.0))
            yaw = float(yaw_mag * rng.choice([-1.0, 1.0]))
    else:
        raise ValueError(f"Unknown transfer bucket: {bucket}")

    return {"task_case": {"xy": xy, "yaw": yaw}}


def capture_transfer_metrics(raw_env) -> dict[str, Any]:
    """Collect the task-relative metrics we care about for diagnostics."""
    raw_obs = raw_env._get_obs()
    info = raw_env._get_info(raw_obs)
    return {
        "pose_error_body": np.asarray(raw_obs["state"]["pose_error_body"], dtype=np.float64),
        "twist_body": np.asarray(raw_obs["state"]["twist_body"], dtype=np.float64),
        "wrench_control_body": np.asarray(
            raw_obs["state"]["wrench_control_body"],
            dtype=np.float64,
        ),
        "xy_error": float(info["xy_error"]),
        "z_gap": float(info["z_gap"]),
        "rot_error": float(info["rot_error"]),
        "success": bool(info["success"]),
        "task_case_x": float(info.get("task_case_x", 0.0)),
        "task_case_y": float(info.get("task_case_y", 0.0)),
        "task_case_yaw": float(info.get("task_case_yaw", 0.0)),
        "reset_case_x": float(info.get("reset_case_x", 0.0)),
        "reset_case_y": float(info.get("reset_case_y", 0.0)),
        "reset_case_z": float(info.get("reset_case_z", 0.0)),
        "reset_case_yaw": float(info.get("reset_case_yaw", 0.0)),
    }


def sample_policy_action(agent, obs, sampling_rng):
    """Sample one deterministic policy action and return the advanced RNG."""
    sampling_rng, key = jax.random.split(sampling_rng)
    action = agent.sample_actions(
        observations=jax.device_put(obs),
        argmax=True,
        seed=key,
    )
    return np.asarray(jax.device_get(action)), sampling_rng


def run_eval_episode(
    *,
    agent,
    env,
    raw_env,
    sampling_rng,
    renderer,
    video_recorder,
    reset_options=None,
    realtime_factor: float = 0.0,
    collect_trace: bool = False,
):
    """Run one eval episode and optionally collect per-step default-case states."""
    obs, _ = env.reset(options=reset_options)
    if video_recorder is not None:
        video_recorder.start_episode(0)
    if renderer is not None:
        renderer.maybe_render(force=True)
    done = False
    episode_steps = 0
    episode_trace = []
    episode_start_wall_time = time.perf_counter()
    episode_start_sim_time = float(raw_env.data.time)

    while not done:
        if collect_trace:
            episode_trace.append(
                {
                    "snapshot": raw_env.capture_diagnostic_snapshot(),
                    "obs": copy.deepcopy(obs),
                }
            )
        actions, sampling_rng = sample_policy_action(agent, obs, sampling_rng)
        if collect_trace:
            episode_trace[-1]["action"] = actions.copy()
        next_obs, reward, terminated, truncated, info = env.step(actions)
        episode_steps += 1
        if realtime_factor > 0.0:
            sim_elapsed = float(raw_env.data.time) - episode_start_sim_time
            wall_elapsed = time.perf_counter() - episode_start_wall_time
            target_wall_elapsed = sim_elapsed * realtime_factor
            sleep_s = target_wall_elapsed - wall_elapsed
            if sleep_s > 0.0:
                time.sleep(sleep_s)
        if renderer is not None:
            renderer.maybe_render()
        if video_recorder is not None:
            video_recorder.maybe_capture()
        obs = next_obs
        done = terminated or truncated

    episode_wall_time = time.perf_counter() - episode_start_wall_time
    episode_sim_time = float(raw_env.data.time) - episode_start_sim_time
    if video_recorder is not None:
        video_recorder.finish_episode()

    result = {
        "success": bool(info.get("success", False)),
        "steps": int(episode_steps),
        "success_sim_time": float(episode_sim_time),
        "success_wall_time": float(episode_wall_time),
        "xy_error": float(info.get("xy_error", np.nan)),
        "z_gap": float(info.get("z_gap", np.nan)),
        "rot_error": float(info.get("rot_error", np.nan)),
        "task_case_x": float(info.get("task_case_x", 0.0)),
        "task_case_y": float(info.get("task_case_y", 0.0)),
        "task_case_yaw": float(info.get("task_case_yaw", 0.0)),
        "reset_case_x": float(info.get("reset_case_x", 0.0)),
        "reset_case_y": float(info.get("reset_case_y", 0.0)),
        "reset_case_z": float(info.get("reset_case_z", 0.0)),
        "reset_case_yaw": float(info.get("reset_case_yaw", 0.0)),
        "trace": episode_trace,
    }
    return result, sampling_rng


def summarize_bucket_results(bucket_results: dict[str, list[dict[str, Any]]]) -> None:
    """Print a compact summary of strict transfer buckets."""
    print("\nTransfer bucket summary:")
    for bucket in TRANSFER_BUCKET_ORDER:
        rows = bucket_results.get(bucket, [])
        label = TRANSFER_BUCKET_LABELS[bucket]
        if not rows:
            print(f"  {label}: no episodes")
            continue
        success_rate = float(np.mean([row["success"] for row in rows]))
        success_steps = [row["steps"] for row in rows if row["success"]]
        mean_steps = float(np.mean(success_steps)) if success_steps else float("nan")
        mean_xy = float(np.mean([row["xy_error"] for row in rows]))
        mean_z = float(np.mean([row["z_gap"] for row in rows]))
        mean_rot = float(np.mean([row["rot_error"] for row in rows]))
        steps_str = f"{mean_steps:.1f}" if success_steps else "n/a"
        print(
            f"  {label}: success_rate={success_rate:.3f}, "
            f"avg_success_steps={steps_str}, "
            f"final_xy={mean_xy:.4f}, final_z_gap={mean_z:.4f}, "
            f"final_rot={mean_rot:.4f}"
        )


def select_probe_samples(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Evenly subsample a list of recorded default-case states."""
    if limit <= 0 or not samples:
        return []
    count = min(limit, len(samples))
    indices = np.linspace(0, len(samples) - 1, num=count, dtype=int)
    return [samples[int(index)] for index in indices]


def run_policy_action_consistency_probe(
    *,
    agent,
    env,
    raw_env,
    sampling_rng,
    default_samples: list[dict[str, Any]],
    num_states: int,
    rng: np.random.Generator,
    task_case_range_preset: str,
    task_case_yaw_range_override: Optional[float],
):
    """Check whether matched task-relative states produce similar policy actions."""
    selected_samples = select_probe_samples(default_samples, num_states)
    results = []
    for idx, sample in enumerate(selected_samples):
        sampled_case = sample_task_case_for_bucket(
            raw_env,
            "combined_xy_yaw",
            rng,
            task_case_range_preset=task_case_range_preset,
            task_case_yaw_range_override=task_case_yaw_range_override,
        )
        if sampled_case is None:
            continue

        raw_env.restore_diagnostic_snapshot(sample["snapshot"])
        default_raw_obs = raw_env._get_obs()
        default_policy_obs = build_policy_observation_from_raw_env(
            env,
            raw_env,
            raw_obs=default_raw_obs,
        )
        default_action, sampling_rng = sample_policy_action(
            agent,
            default_policy_obs,
            sampling_rng,
        )

        raw_env.restore_transformed_diagnostic_snapshot(
            sample["snapshot"],
            xy_offset=sampled_case["task_case"]["xy"],
            yaw_offset=sampled_case["task_case"]["yaw"],
        )
        case_raw_obs = raw_env._get_obs()
        case_policy_obs = build_policy_observation_from_raw_env(
            env,
            raw_env,
            raw_obs=case_raw_obs,
        )
        case_action, sampling_rng = sample_policy_action(
            agent,
            case_policy_obs,
            sampling_rng,
        )

        action_diff = float(np.linalg.norm(default_action - case_action))
        pose_error_diff = float(
            np.linalg.norm(
                np.asarray(default_raw_obs["state"]["pose_error_body"], dtype=np.float64)
                - np.asarray(case_raw_obs["state"]["pose_error_body"], dtype=np.float64)
            )
        )
        twist_diff = float(
            np.linalg.norm(
                np.asarray(default_raw_obs["state"]["twist_body"], dtype=np.float64)
                - np.asarray(case_raw_obs["state"]["twist_body"], dtype=np.float64)
            )
        )
        wrench_diff = float(
            np.linalg.norm(
                np.asarray(
                    default_raw_obs["state"]["wrench_control_body"],
                    dtype=np.float64,
                )
                - np.asarray(
                    case_raw_obs["state"]["wrench_control_body"],
                    dtype=np.float64,
                )
            )
        )
        results.append(
            {
                "index": idx,
                "task_case_xy": sampled_case["task_case"]["xy"].copy(),
                "task_case_yaw": float(sampled_case["task_case"]["yaw"]),
                "action_diff": action_diff,
                "pose_error_diff": pose_error_diff,
                "twist_diff": twist_diff,
                "wrench_diff": wrench_diff,
            }
        )

    if results:
        action_diffs = np.asarray([item["action_diff"] for item in results], dtype=np.float64)
        pose_diffs = np.asarray([item["pose_error_diff"] for item in results], dtype=np.float64)
        twist_diffs = np.asarray([item["twist_diff"] for item in results], dtype=np.float64)
        wrench_diffs = np.asarray([item["wrench_diff"] for item in results], dtype=np.float64)
        print("\nPolicy action consistency probe:")
        print(
            "  sampled states="
            f"{len(results)}, action_diff mean={action_diffs.mean():.4f}, "
            f"median={np.median(action_diffs):.4f}, max={action_diffs.max():.4f}"
        )
        print(
            "  matched observation diffs: "
            f"pose_error_body mean={pose_diffs.mean():.6f}, "
            f"twist_body mean={twist_diffs.mean():.6f}, "
            f"wrench_control_body mean={wrench_diffs.mean():.6f}"
        )
    else:
        print("\nPolicy action consistency probe: no successful default states available.")
    return results, sampling_rng


def rollout_fixed_action_sequence(
    *,
    raw_env,
    snapshot: dict[str, Any],
    action_sequence: list[np.ndarray],
    task_case: Optional[dict[str, Any]] = None,
):
    """Roll out a fixed action sequence from a default or matched transformed state."""
    if task_case is None:
        raw_env.restore_diagnostic_snapshot(snapshot)
    else:
        raw_env.restore_transformed_diagnostic_snapshot(
            snapshot,
            xy_offset=task_case["xy"],
            yaw_offset=task_case["yaw"],
        )

    trajectory = [capture_transfer_metrics(raw_env)]
    for action in action_sequence:
        _, _, terminated, truncated, _ = raw_env.step(np.asarray(action, dtype=np.float64))
        trajectory.append(capture_transfer_metrics(raw_env))
        if terminated or truncated:
            break
    return trajectory


def run_closed_loop_invariance_probe(
    *,
    agent,
    env,
    raw_env,
    sampling_rng,
    default_samples: list[dict[str, Any]],
    num_states: int,
    horizon: int,
    rng: np.random.Generator,
    task_case_range_preset: str,
    task_case_yaw_range_override: Optional[float],
):
    """Compare closed-loop evolution under the same action sequence."""
    selected_samples = select_probe_samples(default_samples, max(1, min(num_states, 5)))
    results = []
    for idx, sample in enumerate(selected_samples):
        sampled_case = sample_task_case_for_bucket(
            raw_env,
            "combined_xy_yaw",
            rng,
            task_case_range_preset=task_case_range_preset,
            task_case_yaw_range_override=task_case_yaw_range_override,
        )
        if sampled_case is None:
            continue

        raw_env.restore_diagnostic_snapshot(sample["snapshot"])
        action_sequence: list[np.ndarray] = []
        for _ in range(max(horizon, 1)):
            policy_obs = build_policy_observation_from_raw_env(env, raw_env)
            action, sampling_rng = sample_policy_action(agent, policy_obs, sampling_rng)
            action_sequence.append(action.copy())
            _, _, terminated, truncated, _ = raw_env.step(action)
            if terminated or truncated:
                break

        default_traj = rollout_fixed_action_sequence(
            raw_env=raw_env,
            snapshot=sample["snapshot"],
            action_sequence=action_sequence,
            task_case=None,
        )
        case_traj = rollout_fixed_action_sequence(
            raw_env=raw_env,
            snapshot=sample["snapshot"],
            action_sequence=action_sequence,
            task_case=sampled_case["task_case"],
        )

        common_length = min(len(default_traj), len(case_traj))
        pose_diffs = np.asarray(
            [
                np.linalg.norm(
                    default_traj[t]["pose_error_body"] - case_traj[t]["pose_error_body"]
                )
                for t in range(common_length)
            ],
            dtype=np.float64,
        )
        twist_diffs = np.asarray(
            [
                np.linalg.norm(
                    default_traj[t]["twist_body"] - case_traj[t]["twist_body"]
                )
                for t in range(common_length)
            ],
            dtype=np.float64,
        )
        wrench_diffs = np.asarray(
            [
                np.linalg.norm(
                    default_traj[t]["wrench_control_body"]
                    - case_traj[t]["wrench_control_body"]
                )
                for t in range(common_length)
            ],
            dtype=np.float64,
        )
        xy_diffs = np.asarray(
            [
                abs(default_traj[t]["xy_error"] - case_traj[t]["xy_error"])
                for t in range(common_length)
            ],
            dtype=np.float64,
        )
        z_gap_diffs = np.asarray(
            [
                abs(default_traj[t]["z_gap"] - case_traj[t]["z_gap"])
                for t in range(common_length)
            ],
            dtype=np.float64,
        )
        rot_diffs = np.asarray(
            [
                abs(default_traj[t]["rot_error"] - case_traj[t]["rot_error"])
                for t in range(common_length)
            ],
            dtype=np.float64,
        )

        results.append(
            {
                "index": idx,
                "task_case_xy": sampled_case["task_case"]["xy"].copy(),
                "task_case_yaw": float(sampled_case["task_case"]["yaw"]),
                "trajectory_length": int(common_length),
                "pose_error_diff_mean": float(pose_diffs.mean()),
                "pose_error_diff_final": float(pose_diffs[-1]),
                "twist_diff_mean": float(twist_diffs.mean()),
                "wrench_diff_mean": float(wrench_diffs.mean()),
                "xy_error_diff_mean": float(xy_diffs.mean()),
                "z_gap_diff_mean": float(z_gap_diffs.mean()),
                "rot_error_diff_mean": float(rot_diffs.mean()),
            }
        )

    if results:
        pose_mean = np.asarray([item["pose_error_diff_mean"] for item in results], dtype=np.float64)
        pose_final = np.asarray([item["pose_error_diff_final"] for item in results], dtype=np.float64)
        wrench_mean = np.asarray([item["wrench_diff_mean"] for item in results], dtype=np.float64)
        print("\nClosed-loop invariance probe:")
        print(
            "  sampled states="
            f"{len(results)}, mean pose_error_body divergence={pose_mean.mean():.4f}, "
            f"mean final divergence={pose_final.mean():.4f}, "
            f"mean wrench divergence={wrench_mean.mean():.4f}"
        )
    else:
        print("\nClosed-loop invariance probe: no successful default states available.")
    return results, sampling_rng


def peek_spacemouse_intervention(env) -> Tuple[Optional[np.ndarray], bool]:
    """Read one cached SpaceMouse action from the intervention wrapper if present."""
    for wrapped_env in iter_env_chain(env):
        peek = getattr(wrapped_env, "peek_intervention", None)
        if callable(peek):
            action, intervened = peek()
            return np.asarray(action, dtype=np.float64), bool(intervened)
    return None, False


class ActorRenderManager:
    """Throttle actor visualization so rendering does not set the control rate."""

    def __init__(
        self,
        env,
        *,
        enabled: bool,
        viewer: str,
        fps: float,
    ) -> None:
        if fps < 0.0:
            raise ValueError("--actor_render_fps must be non-negative.")

        self.env = env
        self.viewer = viewer
        self.interval = 1.0 / fps if fps > 0.0 else np.inf
        self.next_render_time = 0.0
        self.enabled = bool(enabled) and viewer != "none" and fps > 0.0
        self._meshcat = None
        self._meshcat_geometry = None
        self._raw_env = None
        self._meshcat_geom_paths = []
        self._meshcat_site_paths = []

        if self.enabled and self.viewer == "meshcat":
            self._setup_meshcat()

    def maybe_render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        if not force and now < self.next_render_time:
            return

        if self.viewer == "mujoco":
            self.env.render()
        elif self.viewer == "meshcat":
            self._render_meshcat()

        self.next_render_time = now + self.interval

    def _setup_meshcat(self) -> None:
        try:
            from importlib import metadata as importlib_metadata

            importlib_metadata.version("typing")
        except importlib_metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError(
                "--actor_viewer=meshcat cannot run while the Python 2/3 "
                "`typing` backport package is installed in Python 3.11. "
                "Run `python -m pip uninstall -y typing` in the actor environment."
            )

        try:
            import meshcat
            import meshcat.geometry as geometry
        except ImportError as exc:
            raise RuntimeError(
                "--actor_viewer=meshcat requires meshcat. "
                "Install it in the actor environment with `python -m pip install meshcat`."
            ) from exc

        raw_env = get_raw_env(self.env)
        if not hasattr(raw_env, "model") or not hasattr(raw_env, "data"):
            raise RuntimeError(
                "--actor_viewer=meshcat requires a MuJoCo env exposing model/data."
            )

        self._raw_env = raw_env
        self._meshcat_geometry = geometry
        self._meshcat = meshcat.Visualizer()
        self._meshcat.open()
        self._meshcat.delete()
        self._load_mujoco_visual_geoms()

    def _render_meshcat(self) -> None:
        if self._meshcat is None or self._raw_env is None:
            return

        for path, geom_id, correction in self._meshcat_geom_paths:
            self._meshcat[path].set_transform(
                self._geom_transform(geom_id) @ correction
            )
        for path, site_id, correction in self._meshcat_site_paths:
            self._meshcat[path].set_transform(
                self._site_transform(site_id) @ correction
            )

    def _load_mujoco_visual_geoms(self) -> None:
        import mujoco

        model = self._raw_env.model
        geometry = self._meshcat_geometry
        self._meshcat_geom_paths = []
        self._meshcat_site_paths = []

        for geom_id in range(model.ngeom):
            if int(model.geom_group[geom_id]) != 2:
                continue
            if model.geom_rgba[geom_id, 3] <= 0.0:
                continue

            geom_type = int(model.geom_type[geom_id])
            if geom_type in (
                int(mujoco.mjtGeom.mjGEOM_PLANE),
                int(mujoco.mjtGeom.mjGEOM_HFIELD),
                int(mujoco.mjtGeom.mjGEOM_SDF),
            ):
                continue

            meshcat_geom, correction = self._meshcat_geometry_for_mujoco_geom(
                geom_id
            )
            if meshcat_geom is None:
                continue

            material = self._meshcat_material_for_mujoco_geom(geom_id)
            geom_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            path = f"mujoco/geoms/{geom_id:03d}_{geom_name or 'geom'}"
            self._meshcat[path].set_object(meshcat_geom, material)
            self._meshcat[path].set_transform(
                self._geom_transform(geom_id) @ correction
            )
            self._meshcat_geom_paths.append((path, geom_id, correction))

        self._meshcat["world/tool_tip"].set_object(
            geometry.Sphere(0.004),
            geometry.MeshLambertMaterial(color=0x00CC44),
        )
        self._meshcat["world/socket"].set_object(
            geometry.Sphere(0.005),
            geometry.MeshLambertMaterial(color=0xCC2222),
        )
        self._meshcat_site_paths.extend(
            [
                (
                    "world/tool_tip",
                    self._raw_env._ee_site_id,
                    np.eye(4, dtype=np.float64),
                ),
                (
                    "world/socket",
                    self._raw_env._socket_site_id,
                    np.eye(4, dtype=np.float64),
                ),
            ]
        )

        print_green(
            "MeshCat loaded "
            f"{len(self._meshcat_geom_paths)} MuJoCo visual geoms and "
            f"{len(self._meshcat_site_paths)} site markers."
        )

    def _meshcat_geometry_for_mujoco_geom(self, geom_id: int):
        import mujoco

        model = self._raw_env.model
        geometry = self._meshcat_geometry
        geom_type = int(model.geom_type[geom_id])
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
        correction = np.eye(4, dtype=np.float64)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                return None, correction

            vert_adr = int(model.mesh_vertadr[mesh_id])
            vert_num = int(model.mesh_vertnum[mesh_id])
            face_adr = int(model.mesh_faceadr[mesh_id])
            face_num = int(model.mesh_facenum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[vert_adr : vert_adr + vert_num],
                dtype=np.float32,
            ).copy()
            faces = np.asarray(
                model.mesh_face[face_adr : face_adr + face_num],
                dtype=np.uint32,
            ).copy()
            return geometry.TriangularMeshGeometry(vertices, faces), correction

        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            return geometry.Box(2.0 * size[:3]), correction

        if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return geometry.Sphere(float(size[0])), correction

        if geom_type in (
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        ):
            correction[:3, :3] = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            )
            return geometry.Cylinder(2.0 * float(size[1]), float(size[0])), correction

        return None, correction

    def _meshcat_material_for_mujoco_geom(self, geom_id: int):
        model = self._raw_env.model
        material_id = int(model.geom_matid[geom_id])
        if material_id >= 0:
            rgba = model.mat_rgba[material_id]
        else:
            rgba = model.geom_rgba[geom_id]
        rgba = np.clip(rgba, 0.0, 1.0)
        rgb = (255.0 * rgba[:3]).astype(int)
        color = int(rgb[0] << 16 | rgb[1] << 8 | rgb[2])
        return self._meshcat_geometry.MeshLambertMaterial(
            color=color,
            transparent=bool(rgba[3] < 1.0),
            opacity=float(rgba[3]),
        )

    def _geom_transform(self, geom_id: int) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self._raw_env.data.geom_xmat[geom_id].reshape(3, 3)
        transform[:3, 3] = self._raw_env.data.geom_xpos[geom_id]
        return transform

    def _site_transform(self, site_id: int) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self._raw_env.data.site_xmat[site_id].reshape(3, 3)
        transform[:3, 3] = self._raw_env.data.site_xpos[site_id]
        return transform


class EvalVideoRecorder:
    """Record MuJoCo RGB frames during checkpoint eval rollouts."""

    def __init__(
        self,
        env,
        *,
        output_path: Optional[str],
        num_episodes: int,
        fps: float,
        width: int,
        height: int,
        camera: Optional[str],
    ) -> None:
        self.enabled = bool(output_path)
        self._writer = None
        self._renderer = None
        self._imageio = None
        self._active_path: Optional[Path] = None
        self._next_frame_time = 0.0
        self._frame_interval = 0.0
        self._last_frame_time: Optional[float] = None

        if not self.enabled:
            return

        if fps <= 0.0:
            raise ValueError("--eval_video_fps must be positive.")
        if width <= 0 or height <= 0:
            raise ValueError(
                "--eval_video_width and --eval_video_height must be positive."
            )

        raw_env = get_raw_env(env)
        if not hasattr(raw_env, "model") or not hasattr(raw_env, "data"):
            raise RuntimeError(
                "--eval_video_path requires a MuJoCo env exposing model/data."
            )

        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "--eval_video_path requires imageio and imageio-ffmpeg. "
                "Install them with `python -m pip install imageio imageio-ffmpeg`."
            ) from exc

        import mujoco

        base_path = Path(str(output_path)).expanduser()
        if base_path.suffix and base_path.suffix.lower() != ".mp4":
            raise ValueError("--eval_video_path must be an .mp4 file or a directory.")

        self.raw_env = raw_env
        self.base_path = base_path
        self.num_episodes = num_episodes
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.camera = camera or getattr(
            getattr(raw_env, "cfg", None),
            "render_camera",
            None,
        )
        self._frame_interval = 1.0 / self.fps
        self._imageio = imageio
        try:
            self._renderer = mujoco.Renderer(raw_env.model, self.height, self.width)
        except Exception as exc:
            raise RuntimeError(
                "Failed to create the MuJoCo offscreen renderer for eval video. "
                "If this is a headless session, start the command with "
                "`MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`."
            ) from exc

    def start_episode(self, episode_idx: int) -> None:
        if not self.enabled:
            return

        if self._writer is None:
            path = self._output_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._writer = self._imageio.get_writer(
                    str(path),
                    fps=self.fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to create MP4 writer. Make sure imageio-ffmpeg is installed: "
                    "`python -m pip install imageio-ffmpeg`."
                ) from exc
            self._active_path = path

        self._last_frame_time = None
        self._next_frame_time = float(self.raw_env.data.time)
        self.capture(force=True)

    def maybe_capture(self) -> None:
        self.capture(force=False)

    def capture(self, *, force: bool = False) -> None:
        if not self.enabled or self._writer is None:
            return

        sim_time = float(self.raw_env.data.time)
        if not force and sim_time + 1e-12 < self._next_frame_time:
            return
        if force and self._last_frame_time is not None:
            if abs(sim_time - self._last_frame_time) < 1e-12:
                return

        self._renderer.update_scene(self.raw_env.data, camera=self.camera)
        self._writer.append_data(self._renderer.render())
        self._last_frame_time = sim_time

        if force:
            self._next_frame_time = sim_time + self._frame_interval
            return

        while self._next_frame_time <= sim_time + 1e-12:
            self._next_frame_time += self._frame_interval

    def finish_episode(self) -> None:
        if self._writer is None:
            return

        self.capture(force=True)

    def close(self) -> None:
        self.finish_episode()
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None
                if self._active_path is not None:
                    print_green(f"Saved eval video: {self._active_path}")
                self._active_path = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _output_path(self) -> Path:
        if self.base_path.exists() and self.base_path.is_dir():
            return self.base_path / "eval.mp4"
        if not self.base_path.suffix:
            return self.base_path / "eval.mp4"
        return self.base_path


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """Actor loop: steps the environment, optionally records SpaceMouse interventions."""
    renderer = ActorRenderManager(
        env,
        enabled=FLAGS.render_actor,
        viewer=FLAGS.actor_viewer,
        fps=FLAGS.actor_render_fps,
    )

    if FLAGS.eval_checkpoint_step:
        if FLAGS.eval_checkpoint_step < -1:
            raise ValueError(
                "--eval_checkpoint_step must be positive, 0 to disable eval, "
                "or -1 to load the latest checkpoint."
            )
        if FLAGS.eval_n_trajs <= 0:
            raise ValueError("--eval_n_trajs must be positive when running eval.")
        if FLAGS.eval_realtime_factor < 0.0:
            raise ValueError("--eval_realtime_factor must be non-negative.")
        if FLAGS.checkpoint_path is None:
            raise ValueError("--checkpoint_path is required when running eval.")

        raw_env = get_raw_env(env)
        success_counter = 0
        success_step_list = []
        success_sim_time_list = []
        success_wall_time_list = []
        checkpoint_dir = os.path.abspath(FLAGS.checkpoint_path)
        checkpoint_step = (
            None if FLAGS.eval_checkpoint_step == -1 else FLAGS.eval_checkpoint_step
        )
        if checkpoint_step is None:
            latest = checkpoints.latest_checkpoint(checkpoint_dir)
            if latest is None:
                raise FileNotFoundError(
                    f"No checkpoint found in checkpoint_path={checkpoint_dir!r}."
                )
            print_green(f"Loading latest checkpoint: {latest}")
        else:
            print_green(
                f"Loading checkpoint step {checkpoint_step} from {checkpoint_dir}"
            )

        ckpt = checkpoints.restore_checkpoint(
            checkpoint_dir,
            agent.state,
            step=checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        video_recorder = EvalVideoRecorder(
            env,
            output_path=FLAGS.eval_video_path,
            num_episodes=FLAGS.eval_n_trajs,
            fps=FLAGS.eval_video_fps,
            width=FLAGS.eval_video_width,
            height=FLAGS.eval_video_height,
            camera=FLAGS.eval_video_camera,
        )
        eval_reset_options = (
            {"sample_task_case": True} if FLAGS.eval_sample_task_case else None
        )
        reset_mode = (
            "zeroed"
            if FLAGS.eval_zero_reset_perturbation
            else FLAGS.reset_perturbation_preset
        )
        if FLAGS.eval_sample_task_case:
            print(
                "Eval task-case preset: "
                f"{FLAGS.eval_task_case_range_preset} | "
                f"reset={reset_mode} | "
                f"NRIC={'off' if FLAGS.eval_disable_nric else 'on'} | "
                f"nullspace={'zero' if FLAGS.eval_zero_nullspace else 'default'}"
            )
        diagnostics_rng = np.random.default_rng(FLAGS.seed)
        eval_task_case_yaw_range_override = optional_flag_float(
            FLAGS.eval_task_case_yaw_range_override
        )

        try:
            if FLAGS.eval_transfer_diagnostics:
                obs_wrapper = find_wrapper_by_name(env, "MujocoSERLObsWrapper")
                proprio_keys = getattr(obs_wrapper, "proprio_keys", None)
                print("\nTransfer diagnostics:")
                print(f"  policy proprio keys: {proprio_keys}")
                print(
                    "  evaluating default and transfer buckets separately "
                    "under the current reset/task-case presets."
                )

                bucket_results = defaultdict(list)
                successful_default_samples = []

                for bucket in TRANSFER_BUCKET_ORDER:
                    for episode_idx in range(FLAGS.eval_n_trajs):
                        reset_options = sample_task_case_for_bucket(
                            raw_env,
                            bucket,
                            diagnostics_rng,
                            task_case_range_preset=FLAGS.eval_task_case_range_preset,
                            task_case_yaw_range_override=eval_task_case_yaw_range_override,
                        )
                        collect_trace = bucket == "default"
                        result, sampling_rng = run_eval_episode(
                            agent=agent,
                            env=env,
                            raw_env=raw_env,
                            sampling_rng=sampling_rng,
                            renderer=renderer,
                            video_recorder=video_recorder if bucket == "default" else None,
                            reset_options=reset_options,
                            realtime_factor=FLAGS.eval_realtime_factor,
                            collect_trace=collect_trace,
                        )
                        bucket_results[bucket].append(result)
                        if bucket == "default" and result["success"]:
                            successful_default_samples.extend(result["trace"])
                        print(
                            f"[{TRANSFER_BUCKET_LABELS[bucket]}] "
                            f"episode {episode_idx + 1}/{FLAGS.eval_n_trajs}: "
                            f"success={result['success']}, steps={result['steps']}, "
                            f"xy={result['xy_error']:.4f}, z_gap={result['z_gap']:.4f}, "
                            f"rot={result['rot_error']:.4f}, "
                            f"task_case=({result['task_case_x']:.4f}, "
                            f"{result['task_case_y']:.4f}, "
                            f"yaw={result['task_case_yaw']:.3f}), "
                            f"reset_case=({result['reset_case_x']:.4f}, "
                            f"{result['reset_case_y']:.4f}, "
                            f"yaw={result['reset_case_yaw']:.3f})"
                        )

                summarize_bucket_results(bucket_results)
                _, sampling_rng = run_policy_action_consistency_probe(
                    agent=agent,
                    env=env,
                    raw_env=raw_env,
                    sampling_rng=sampling_rng,
                    default_samples=successful_default_samples,
                    num_states=FLAGS.eval_transfer_probe_num_states,
                    rng=diagnostics_rng,
                    task_case_range_preset=FLAGS.eval_task_case_range_preset,
                    task_case_yaw_range_override=eval_task_case_yaw_range_override,
                )
                _, sampling_rng = run_closed_loop_invariance_probe(
                    agent=agent,
                    env=env,
                    raw_env=raw_env,
                    sampling_rng=sampling_rng,
                    default_samples=successful_default_samples,
                    num_states=FLAGS.eval_transfer_probe_num_states,
                    horizon=FLAGS.eval_transfer_probe_horizon,
                    rng=diagnostics_rng,
                    task_case_range_preset=FLAGS.eval_task_case_range_preset,
                    task_case_yaw_range_override=eval_task_case_yaw_range_override,
                )
            else:
                for episode in range(FLAGS.eval_n_trajs):
                    result, sampling_rng = run_eval_episode(
                        agent=agent,
                        env=env,
                        raw_env=raw_env,
                        sampling_rng=sampling_rng,
                        renderer=renderer,
                        video_recorder=video_recorder,
                        reset_options=eval_reset_options,
                        realtime_factor=FLAGS.eval_realtime_factor,
                        collect_trace=False,
                    )
                    if result["success"]:
                        success_counter += 1
                        success_step_list.append(result["steps"])
                        success_sim_time_list.append(result["success_sim_time"])
                        success_wall_time_list.append(result["success_wall_time"])

                    print(
                        "Episode "
                        f"{episode + 1}: success={result['success']}, "
                        f"steps={result['steps']}, "
                        f"success_sim_time={result['success_sim_time']:.2f}s, "
                        f"success_wall_time={result['success_wall_time']:.2f}s, "
                        f"task_case=({result['task_case_x']:.4f}, "
                        f"{result['task_case_y']:.4f}, "
                        f"yaw={result['task_case_yaw']:.3f}), "
                        f"reset_case=({result['reset_case_x']:.4f}, "
                        f"{result['reset_case_y']:.4f}, "
                        f"yaw={result['reset_case_yaw']:.3f})"
                    )
                    print(f"  running success rate: {success_counter}/{episode + 1}")
        finally:
            video_recorder.close()

        if not FLAGS.eval_transfer_diagnostics:
            print(f"Final success rate: {success_counter / FLAGS.eval_n_trajs}")
            if success_sim_time_list:
                print(
                    "Average success steps: "
                    f"{np.mean(success_step_list):.1f}"
                )
                print(
                    "Average simulated success time: "
                    f"{np.mean(success_sim_time_list):.2f}s"
                )
                print(
                    "Average wall-clock rollout time: "
                    f"{np.mean(success_wall_time_list):.2f}s"
                )
        return

    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        buffer_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
        if buffer_files:
            start_step = (
                int(os.path.basename(natsorted(buffer_files)[-1])[12:-4]) + 1
            )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = create_actor_trainer_client(data_stores=datastore_dict)

    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    renderer.maybe_render(force=True)
    done = False
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0
    stats_backoff_until = 0.0
    last_stats_warning_time = 0.0
    learner_done = False

    def safe_send_stats(stats: dict, *, label: str) -> bool:
        nonlocal stats_backoff_until, last_stats_warning_time, learner_done
        now = time.monotonic()
        if now < stats_backoff_until:
            return False
        try:
            response = client.request("send-stats", stats)
            stats_backoff_until = 0.0
            if isinstance(response, dict):
                learner_done = bool(response.get("learner_done", False))
            return True
        except Exception as exc:
            stats_backoff_until = now + max(FLAGS.actor_stats_backoff_s, 0.0)
            if now - last_stats_warning_time >= max(FLAGS.actor_stats_backoff_s, 1.0):
                print(
                    f"[WARN] actor failed to send {label} stats to learner: {exc}. "
                    f"Backing off for {FLAGS.actor_stats_backoff_s:.1f}s.",
                    flush=True,
                )
                last_stats_warning_time = now
            return False

    pbar_total = config.max_steps if not FLAGS.actor_run_until_learner_done else None
    pbar = tqdm.tqdm(
        total=pbar_total,
        initial=start_step if pbar_total is not None else 0,
        dynamic_ncols=True,
    )
    step = start_step
    while True:
        if not FLAGS.actor_run_until_learner_done and step >= config.max_steps:
            break
        if (
            FLAGS.actor_run_until_learner_done
            and step >= config.max_steps
            and learner_done
        ):
            print_green("actor reached max_steps and learner reports done; exiting.")
            break

        timer.tick("total")
        intervention_action, intervention_requested = peek_spacemouse_intervention(env)

        with timer.context("sample_actions"):
            if intervention_requested:
                actions = intervention_action
            elif step < config.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                actions = np.asarray(jax.device_get(actions))

        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            renderer.maybe_render()
            if "left" in info:
                info.pop("left")
            if "right" in info:
                info.pop("right")

            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                info["episode"] = info.get("episode", {})
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                info["episode"]["return"] = running_return
                stats = {"environment": info}
                episode_stats_sent = safe_send_stats(stats, label="episode")
                pbar.set_description(f"return: {running_return:.1f}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                if episode_stats_sent:
                    try:
                        client.update()
                    except Exception as exc:
                        print(
                            f"[WARN] actor failed to update network from learner: {exc}",
                            flush=True,
                        )
                obs, _ = env.reset()
                renderer.maybe_render(force=True)

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            os.makedirs(buffer_path, exist_ok=True)
            os.makedirs(demo_buffer_path, exist_ok=True)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            timer_stats_sent = safe_send_stats(stats, label="timer")
            if (
                FLAGS.actor_run_until_learner_done
                and step >= config.max_steps
                and learner_done
            ):
                print_green("actor received learner_done=True; exiting.")
                break
            if (
                FLAGS.actor_run_until_learner_done
                and step >= config.max_steps
                and not timer_stats_sent
            ):
                print_green(
                    "actor reached max_steps and learner is unavailable; exiting."
                )
                break

        if (
            FLAGS.actor_run_until_learner_done
            and step >= config.max_steps
            and learner_done
        ):
            print_green("actor received learner_done=True; exiting.")
            break

        step += 1
        pbar.update(1)

    pbar.close()


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """Learner loop: trains SAC from replay + demo buffers."""
    start_step = 0
    if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest is not None:
            start_step = int(os.path.basename(latest)[11:]) + 1

    step = start_step
    metrics_dir = resolve_metrics_dir()
    metrics_writer = None
    if metrics_dir is None:
        print_green(
            "Local episode metrics disabled: set --checkpoint_path or --metrics_dir "
            "to save reward curves."
        )
    else:
        metrics_writer = EpisodeMetricsWriter(metrics_dir)
        print_green(f"Writing local episode metrics to {metrics_writer.csv_path}")

    learner_done = False

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        if metrics_writer is not None:
            metrics_writer.write(learner_step=step, payload=payload)
        return {
            "learner_done": learner_done,
            "learner_step": int(step),
            "learner_max_steps": int(config.max_steps),
        }

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # 50/50 RLPD sampling
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": config.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    timer = Timer()
    train_critic_networks = frozenset({"critic"})
    train_all_networks = frozenset({"critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        for _ in range(config.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, _ = agent.update(
                    batch,
                    networks_to_update=train_critic_networks,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_all_networks,
            )

        if step > 0 and step % config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
                step=step,
                keep=20,
            )

    learner_done = True
    grace_s = max(float(FLAGS.learner_done_grace_s), 0.0)
    if grace_s > 0.0:
        print_green(
            f"learner finished; keeping request server alive for {grace_s:.1f}s "
            "so actors can exit cleanly."
        )
        time.sleep(grace_s)

    if metrics_writer is not None:
        metrics_writer.close()
        if FLAGS.plot_metrics_on_exit:
            try:
                from scripts.plot_mujoco_metrics import plot_metrics

                plot_metrics(metrics_dir=metrics_dir, window=20)
            except Exception as exc:
                print(f"Failed to plot local metrics: {exc}")


##############################################################################


def main(_):
    global config
    assert FLAGS.exp_name in CONFIG_MAPPING, (
        f"Unknown experiment: {FLAGS.exp_name}. "
        f"Available: {list(CONFIG_MAPPING.keys())}"
    )
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    setattr(config, "reset_perturbation_preset", FLAGS.reset_perturbation_preset)
    setattr(
        config,
        "reset_yaw_range_override",
        optional_flag_float(FLAGS.reset_yaw_range_override),
    )
    setattr(config, "eval_task_case_range_preset", FLAGS.eval_task_case_range_preset)
    setattr(
        config,
        "eval_task_case_yaw_range_override",
        optional_flag_float(FLAGS.eval_task_case_yaw_range_override),
    )
    setattr(
        config,
        "eval_zero_reset_perturbation",
        FLAGS.eval_zero_reset_perturbation if FLAGS.eval_checkpoint_step else False,
    )
    setattr(config, "eval_disable_nric", FLAGS.eval_disable_nric)
    setattr(config, "eval_zero_nullspace", FLAGS.eval_zero_nullspace)

    assert config.batch_size % num_devices == 0
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=False,
        classifier=config.use_reward_classifier,
        enable_intervention=not bool(FLAGS.eval_checkpoint_step),
    )
    env = RecordEpisodeStatistics(env)

    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()

    agent: SACAgent = make_sac_state_agent(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        discount=config.discount,
    )

    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        latest = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        if latest is not None:
            ckpt = checkpoints.restore_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path),
                agent.state,
            )
            agent = agent.replace(state=ckpt)
            ckpt_number = os.path.basename(latest)[11:]
            print_green(f"Loaded previous checkpoint at step {ckpt_number}.")

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())

        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
        )
        demo_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
        )

        wandb_logger = make_wandb_logger(
            project="hil-serl-mujoco",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )

        if FLAGS.demo_path is not None:
            for path in FLAGS.demo_path:
                transitions = load_transition_file(path)
                insert_validated_transitions(
                    replay_buffer=demo_buffer,
                    transitions=transitions,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    source_label=f"demo file {path}",
                )
            print_green(f"demo buffer size: {len(demo_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in natsorted(
                glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    insert_validated_transitions(
                        replay_buffer=replay_buffer,
                        transitions=transitions,
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        source_label=f"checkpoint replay buffer file {file}",
                    )
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in natsorted(
                glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl"))
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    insert_validated_transitions(
                        replay_buffer=demo_buffer,
                        transitions=transitions,
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        source_label=f"checkpoint demo buffer file {file}",
                    )
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)

        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )
    else:
        raise ValueError("Must specify either --learner or --actor")


if __name__ == "__main__":
    app.run(main)
