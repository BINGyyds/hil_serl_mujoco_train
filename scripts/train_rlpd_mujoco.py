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

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
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
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
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


def resolve_metrics_dir() -> Optional[Path]:
    """Return the local metrics directory for learner-side CSV/PNG outputs."""
    if FLAGS.metrics_dir:
        return Path(FLAGS.metrics_dir).expanduser()
    if FLAGS.checkpoint_path:
        return Path(FLAGS.checkpoint_path).expanduser() / "metrics"
    return None


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

        try:
            for episode in range(FLAGS.eval_n_trajs):
                obs, _ = env.reset()
                video_recorder.start_episode(episode)
                renderer.maybe_render(force=True)
                done = False
                episode_steps = 0
                episode_start_wall_time = time.perf_counter()
                episode_start_sim_time = float(raw_env.data.time)
                while not done:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=True,
                        seed=key,
                    )
                    actions = np.asarray(jax.device_get(actions))
                    next_obs, reward, done, truncated, info = env.step(actions)
                    episode_steps += 1
                    if FLAGS.eval_realtime_factor > 0.0:
                        sim_elapsed = float(raw_env.data.time) - episode_start_sim_time
                        wall_elapsed = time.perf_counter() - episode_start_wall_time
                        target_wall_elapsed = (
                            sim_elapsed * FLAGS.eval_realtime_factor
                        )
                        sleep_s = target_wall_elapsed - wall_elapsed
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)
                    renderer.maybe_render()
                    video_recorder.maybe_capture()
                    obs = next_obs
                    done = done or truncated
                episode_wall_time = time.perf_counter() - episode_start_wall_time
                episode_sim_time = float(raw_env.data.time) - episode_start_sim_time
                if info.get("success", False):
                    success_counter += 1
                    success_step_list.append(episode_steps)
                    success_sim_time_list.append(episode_sim_time)
                    success_wall_time_list.append(episode_wall_time)

                video_recorder.finish_episode()
                print(
                    "Episode "
                    f"{episode + 1}: success={info.get('success', False)}, "
                    f"steps={episode_steps}, "
                    f"success_sim_time={episode_sim_time:.2f}s, "
                    f"success_wall_time={episode_wall_time:.2f}s"
                )
                print(f"  running success rate: {success_counter}/{episode + 1}")
        finally:
            video_recorder.close()

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

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=FLAGS.actor_stats_timeout_ms,
    )

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
                for transition in transitions:
                    demo_buffer.insert(transition)
            print_green(f"demo buffer size: {len(demo_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in natsorted(
                glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
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
                    for transition in transitions:
                        demo_buffer.insert(transition)
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
