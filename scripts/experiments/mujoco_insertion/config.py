"""
Training configuration for the MuJoCo Franka insertion task with SpaceMouse HIL.
"""

from serl_robot_infra.configs.insertion_config import (
    build_eval_task_case_overrides,
    build_reset_perturbation_overrides,
    get_proprio_keys,
    make_insertion_config,
)
from serl_robot_infra.envs.franka_insertion_env import FrankaInsertionEnv
from serl_robot_infra.envs.wrappers import (
    MujocoSERLObsWrapper,
    MujocoSpacemouseIntervention,
)
from scripts.experiments.config import DefaultTrainingConfig
from serl_launcher.wrappers.chunking import ChunkingWrapper


class TrainConfig(DefaultTrainingConfig):
    image_keys = []
    classifier_keys = []
    proprio_preset = "wrench"
    proprio_keys = get_proprio_keys(proprio_preset)
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

    def set_proprio_preset(self, preset: str) -> None:
        self.proprio_preset = preset
        self.proprio_keys = get_proprio_keys(preset)

    def get_environment(
        self,
        fake_env=False,
        save_video=False,
        classifier=False,
        enable_intervention=True,
    ):
        overrides = {}
        reset_preset = getattr(self, "reset_perturbation_preset", "default")
        reset_yaw_range_override = getattr(self, "reset_yaw_range_override", None)
        overrides.update(
            build_reset_perturbation_overrides(
                preset=reset_preset,
                yaw_range_override=reset_yaw_range_override,
                zero_reset_perturbation=bool(
                    getattr(self, "eval_zero_reset_perturbation", False)
                ),
            )
        )
        overrides.update(
            build_eval_task_case_overrides(
                preset=getattr(self, "eval_task_case_range_preset", "aligned"),
                xy_range_override=getattr(
                    self,
                    "eval_task_case_xy_range_override",
                    None,
                ),
                yaw_range_override=getattr(
                    self,
                    "eval_task_case_yaw_range_override",
                    None,
                ),
            )
        )
        overrides["task_case_moves_reset_anchor"] = bool(
            getattr(self, "eval_task_case_moves_reset_anchor", False)
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
