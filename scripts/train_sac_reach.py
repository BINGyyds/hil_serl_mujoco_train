#!/usr/bin/env python3
"""
SAC training script for the Franka reach task in MuJoCo.

Two backends are supported:

  1) --backend sb3    (default) — uses Stable-Baselines3 SAC (quick prototyping)
  2) --backend serl   — uses hil-serl's SACAgent (for later HIL integration)

Usage:
    # Quick start with SB3
    python scripts/train_sac_reach.py --backend sb3

    # With hil-serl agent (requires serl_launcher installed)
    python scripts/train_sac_reach.py --backend serl
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    parent_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    for sibling_name in ("sim", "robot_controllers_mujoco"):
        sibling_sim = parent_root / sibling_name
        if sibling_sim.exists():
            sys.path.insert(0, str(sibling_sim))

from robot_controllers_mujoco.configs.reach_config import TrainConfig, make_reach_config
from robot_controllers_mujoco.envs.franka_reach_env import FrankaReachEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

class FlattenStateWrapper(gym.ObservationWrapper):
    """Flatten the nested state dict into a single vector for MLP-based agents."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        sample_obs, _ = env.reset()
        self._state_keys = sorted(sample_obs["state"].keys())
        flat = self._flatten(sample_obs)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=flat.shape, dtype=np.float64
        )

    def observation(self, obs):
        return self._flatten(obs)

    def _flatten(self, obs) -> np.ndarray:
        parts = [obs["state"][k].flatten() for k in self._state_keys]
        return np.concatenate(parts)


def make_env(render_mode=None):
    """Create a FrankaReachEnv with default config, wrapped for flat obs."""
    cfg = make_reach_config()
    env = FrankaReachEnv(config=cfg, render_mode=render_mode)
    env = FlattenStateWrapper(env)
    return env


# ---------------------------------------------------------------------------
# Backend: Stable-Baselines3
# ---------------------------------------------------------------------------

def train_sb3(args):
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        logger.error(
            "stable-baselines3 is not installed. Install with:\n"
            "  pip install stable-baselines3"
        )
        sys.exit(1)

    tc = TrainConfig()
    os.makedirs(tc.output_dir, exist_ok=True)

    logger.info("Creating training environment ...")
    train_env = Monitor(make_env())

    logger.info("Creating evaluation environment ...")
    eval_env = Monitor(make_env())

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=tc.learning_rate,
        buffer_size=tc.buffer_size,
        batch_size=tc.batch_size,
        tau=tc.tau,
        gamma=tc.discount,
        learning_starts=tc.learning_starts,
        train_freq=tc.train_freq,
        gradient_steps=tc.gradient_steps,
        ent_coef="auto",
        verbose=1,
        seed=tc.seed,
        policy_kwargs=dict(net_arch=list(tc.hidden_dims)),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(tc.output_dir, "best_model"),
        log_path=os.path.join(tc.output_dir, "eval_logs"),
        eval_freq=tc.eval_interval,
        n_eval_episodes=tc.eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=tc.save_interval,
        save_path=os.path.join(tc.output_dir, "checkpoints"),
    )

    logger.info("Starting SB3 SAC training for %d steps ...", tc.total_timesteps)
    model.learn(
        total_timesteps=tc.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=tc.log_interval // tc.train_freq,
    )

    final_path = os.path.join(tc.output_dir, "final_model")
    model.save(final_path)
    logger.info("Training complete. Model saved to %s", final_path)

    train_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# Backend: hil-serl SACAgent (state-based)
# ---------------------------------------------------------------------------

def train_serl(args):
    """
    Single-process SAC training using hil-serl's SACAgent.

    This demonstrates how to use the serl_launcher agent with our MuJoCo env.
    For full HIL training, see the actor-learner scripts in hil-serl/examples/.
    """
    try:
        import jax
        import jax.numpy as jnp
        from functools import partial
        from flax import linen as nn

        from serl_launcher.agents.continuous.sac import SACAgent
        from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
        from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
        from serl_launcher.networks.mlp import MLP
    except ImportError as e:
        logger.error("Failed to import serl_launcher dependencies: %s", e)
        logger.error("Make sure JAX/Flax and serl_launcher are installed.")
        sys.exit(1)

    tc = TrainConfig()
    os.makedirs(tc.output_dir, exist_ok=True)

    logger.info("Creating environment ...")
    env = make_env()
    eval_env = make_env()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    logger.info("Observation dim: %d, Action dim: %d", obs_dim, act_dim)

    # --- Create state-based SAC agent ---
    sample_obs = jnp.zeros((1, obs_dim))
    sample_action = jnp.zeros((1, act_dim))

    critic_network_kwargs = {
        "hidden_dims": list(tc.hidden_dims),
        "activations": nn.tanh,
        "use_layer_norm": True,
    }
    policy_network_kwargs = {
        "hidden_dims": list(tc.hidden_dims),
        "activations": nn.tanh,
        "use_layer_norm": True,
    }
    critic_network_kwargs["activate_final"] = True
    policy_network_kwargs["activate_final"] = True

    critic_backbone = partial(MLP, **critic_network_kwargs)
    critic_backbone = ensemblize(critic_backbone, 2)(name="critic_ensemble")
    critic_def = partial(Critic, encoder=None, network=critic_backbone)(name="critic")

    policy_def = Policy(
        encoder=None,
        network=MLP(**policy_network_kwargs),
        action_dim=act_dim,
        tanh_squash_distribution=True,
        std_parameterization="exp",
        std_min=1e-5,
        std_max=5,
        name="actor",
    )

    temperature_def = GeqLagrangeMultiplier(
        init_value=tc.initial_temperature,
        constraint_shape=(),
        constraint_type="geq",
        name="temperature",
    )

    rng = jax.random.PRNGKey(tc.seed)
    agent = SACAgent.create(
        rng,
        sample_obs,
        sample_action,
        actor_def=policy_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        discount=tc.discount,
        soft_target_update_rate=tc.tau,
        critic_ensemble_size=2,
    )

    # --- Simple replay buffer ---
    from collections import deque
    import random

    class ReplayBuffer:
        def __init__(self, capacity):
            self.buf = deque(maxlen=capacity)

        def add(self, obs, action, reward, next_obs, done):
            self.buf.append((obs, action, reward, next_obs, done))

        def sample(self, batch_size):
            batch = random.sample(self.buf, batch_size)
            obs, act, rew, nobs, done = zip(*batch)
            return {
                "observations": jnp.array(np.array(obs)),
                "actions": jnp.array(np.array(act)),
                "rewards": jnp.array(np.array(rew)),
                "next_observations": jnp.array(np.array(nobs)),
                "masks": jnp.array(1.0 - np.array(done, dtype=np.float32)),
                "dones": jnp.array(np.array(done, dtype=np.float32)),
            }

        def __len__(self):
            return len(self.buf)

    replay_buffer = ReplayBuffer(tc.buffer_size)

    # --- Training loop ---
    logger.info("Starting serl SAC training for %d steps ...", tc.total_timesteps)
    obs, _ = env.reset(seed=tc.seed)
    episode_reward = 0.0
    episode_count = 0
    networks_to_update = frozenset({"critic", "actor", "temperature"})

    for step in range(tc.total_timesteps):
        if step < tc.learning_starts:
            action = env.action_space.sample()
        else:
            rng, key = jax.random.split(rng)
            action = agent.sample_actions(
                observations=jax.device_put(obs[None]),
                seed=key,
                argmax=False,
            )
            action = np.asarray(jax.device_get(action))[0]

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        episode_reward += reward

        if done:
            episode_count += 1
            if episode_count % 10 == 0:
                logger.info(
                    "Step %d | Episode %d | Return: %.2f | Buffer: %d",
                    step, episode_count, episode_reward, len(replay_buffer),
                )
            obs, _ = env.reset()
            episode_reward = 0.0

        # Train
        if step >= tc.learning_starts and len(replay_buffer) >= tc.batch_size:
            for _ in range(tc.gradient_steps):
                batch = replay_buffer.sample(tc.batch_size)
                agent, update_info = agent.update(
                    batch, networks_to_update=networks_to_update,
                )

        # Eval
        if step > 0 and step % tc.eval_interval == 0:
            successes = 0
            total_return = 0.0
            for _ in range(tc.eval_episodes):
                eval_obs, _ = eval_env.reset()
                ep_ret = 0.0
                eval_done = False
                while not eval_done:
                    rng, key = jax.random.split(rng)
                    eval_action = agent.sample_actions(
                        observations=jax.device_put(eval_obs[None]),
                        seed=key,
                        argmax=True,
                    )
                    eval_action = np.asarray(jax.device_get(eval_action))[0]
                    eval_obs, r, term, trunc, eval_info = eval_env.step(eval_action)
                    ep_ret += r
                    eval_done = term or trunc
                    if eval_info.get("success", False):
                        successes += 1
                total_return += ep_ret
            avg_return = total_return / tc.eval_episodes
            success_rate = successes / tc.eval_episodes
            logger.info(
                "EVAL @ step %d | Avg return: %.2f | Success rate: %.1f%%",
                step, avg_return, success_rate * 100,
            )

    logger.info("Training complete.")
    env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SAC for Franka reach task")
    parser.add_argument(
        "--backend",
        choices=["sb3", "serl"],
        default="sb3",
        help="RL backend: 'sb3' for Stable-Baselines3, 'serl' for hil-serl SAC",
    )
    args = parser.parse_args()

    if args.backend == "sb3":
        train_sb3(args)
    elif args.backend == "serl":
        train_serl(args)


if __name__ == "__main__":
    main()
