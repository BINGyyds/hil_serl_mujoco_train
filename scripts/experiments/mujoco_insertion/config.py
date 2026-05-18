"""
Training configuration for the MuJoCo Franka insertion task with SpaceMouse HIL.
"""

import numpy as np

from serl_robot_infra.configs.insertion_config import make_insertion_config
from serl_robot_infra.envs.franka_insertion_env import FrankaInsertionEnv
from serl_robot_infra.envs.wrappers import (
    MujocoSERLObsWrapper,
    MujocoSpacemouseIntervention,
)
from scripts.experiments.config import DefaultTrainingConfig
from serl_launcher.wrappers.chunking import ChunkingWrapper


class TrainConfig(DefaultTrainingConfig):
    WIDE_EVAL_CASE_XY_LOW = np.array([-0.01, -0.01], dtype=np.float64)
    WIDE_EVAL_CASE_XY_HIGH = np.array([0.01, 0.01], dtype=np.float64)
    WIDE_EVAL_CASE_YAW_RANGE = 0.25

    image_keys = []
    classifier_keys = []
    proprio_keys = [
        "pose_error_body",
        "twist_body",
        "wrench_control_body",
    ]
    setup_mode = "single-arm-fixed-gripper"
    encoder_type = "none"
    use_reward_classifier = False

    max_traj_length = 3000
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
        overrides = {}
        if getattr(self, "eval_task_case_range_preset", None) == "wide":
            overrides.update(
                eval_case_xy_low=self.WIDE_EVAL_CASE_XY_LOW.copy(),
                eval_case_xy_high=self.WIDE_EVAL_CASE_XY_HIGH.copy(),
                eval_case_yaw_range=self.WIDE_EVAL_CASE_YAW_RANGE,
            )
        if bool(getattr(self, "eval_disable_nric", False)):
            overrides["enable_nric"] = False
        if bool(getattr(self, "eval_zero_nullspace", False)):
            overrides.update(
                nullspace_stiffness=0.0,
                nullspace_damping=0.0,
            )
        cfg = make_insertion_config(**overrides)
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
