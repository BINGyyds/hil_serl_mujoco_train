"""
Training configuration for the MuJoCo Franka insertion task with SpaceMouse HIL.
"""

from robot_controllers_mujoco.configs.insertion_config import make_insertion_config
from robot_controllers_mujoco.envs.franka_insertion_env import FrankaInsertionEnv
from robot_controllers_mujoco.envs.wrappers import (
    MujocoSERLObsWrapper,
    MujocoSpacemouseIntervention,
)
from scripts.experiments.config import DefaultTrainingConfig
from serl_launcher.wrappers.chunking import ChunkingWrapper


class TrainConfig(DefaultTrainingConfig):
    image_keys = []
    classifier_keys = []
    proprio_keys = [
        "tcp_pose",
        "tcp_vel",
        "q_pos",
        "q_vel",
        "socket_pose",
        "relative_pose",
        "insertion_depth",
        "controller_target_pose",
    ]
    setup_mode = "single-arm-fixed-gripper"
    encoder_type = "none"
    use_reward_classifier = False

    max_traj_length = 1000
    batch_size = 256
    discount = 0.99
    cta_ratio = 2

    max_steps = 5_000_000
    replay_buffer_capacity = 5_000_000
    training_starts = 1000
    random_steps = 1000
    steps_per_update = 10

    log_period = 100
    checkpoint_period = 5000
    buffer_period = 1000

    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        classifier=False,
        enable_intervention=True,
    ):
        cfg = make_insertion_config()
        env = FrankaInsertionEnv(
            config=cfg,
            render_mode="human" if not fake_env else None,
        )
        if enable_intervention and not fake_env:
            env = MujocoSpacemouseIntervention(env)
        env = MujocoSERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        return env

    def process_demos(self, demo):
        return demo
