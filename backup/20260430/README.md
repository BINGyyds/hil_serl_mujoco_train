# 20260430 训练备份

这份备份记录的是一次 `mujoco_insertion` 的代表性训练结果，包含：

- 训练 checkpoint：`checkpoint_4995000`
- 本地 metrics：`episode_metrics.csv`
- 导出的 reward / success / error 曲线

## 基本信息

- 日期：`2026-04-30`
- 分支：`develop`
- 代码版本：`ca03068`
- 任务：`mujoco_insertion`
- 训练范式：state-based SAC / RLPD

## 目录内容

```text
20260430/
├── README.md
├── checkpoint_4995000/
└── metrics/
    ├── episode_metrics.csv
    ├── reward_curve.png
    ├── success_curve.png
    └── error_curve.png
```

## 训练配置

训练配置来自：

- [`train/scripts/experiments/mujoco_insertion/config.py`](/home/lrb/Workspace/hil_serl_mujoco/train/scripts/experiments/mujoco_insertion/config.py)
- [`sim/robot_controllers_mujoco/configs/insertion_config.py`](/home/lrb/Workspace/hil_serl_mujoco/sim/robot_controllers_mujoco/configs/insertion_config.py)

### RL 主配置

- `image_keys = []`
- `encoder_type = "none"`
- `use_reward_classifier = False`
- `max_traj_length = 1000`
- `batch_size = 256`
- `discount = 0.99`
- `cta_ratio = 2`
- `max_steps = 5_000_000`
- `replay_buffer_capacity = 5_000_000`
- `training_starts = 1000`
- `random_steps = 1000`
- `steps_per_update = 10`
- `checkpoint_period = 5000`

### Observation

当前训练默认是 **state-only**，使用的 proprio/state keys 为：

- `tcp_pose`
- `tcp_vel`
- `q_pos`
- `q_vel`
- `socket_pose`
- `relative_pose`
- `insertion_depth`
- `controller_target_pose`

### 仿真与控制配置

- `sim_timestep = 0.001`
- `n_substeps = 10`
- `control_dt = 0.01`
- 控制频率：`100 Hz`
- `max_episode_steps = 1000`
- `actuator_mode = "velocity"`

### 动作尺度

- `pos_action_scale = 0.02`
- `rot_action_scale = 0.08`

### Reward 配置

- `reward_xy_weight = 8.0`
- `reward_z_weight = 1.5`
- `reward_rot_weight = 0.3`
- `reward_depth_weight = 2.0`
- `reward_success_bonus = 2000.0`
- `reward_failure_penalty = 25.0`
- `reward_step_penalty = 0.01`
- `reward_depth_xy_threshold = 0.004`
- `reward_depth_rot_threshold = 0.12`

当前 reward 设计为：

- 位置误差惩罚
- 姿态误差惩罚
- 弱绝对深度奖励
- 较大的成功奖励

### Success 判据

- `success_xy_threshold = 0.002`
- `success_rot_threshold = 0.1`
- `success_z_gap_threshold = 0.002`
- `success_linear_velocity_threshold = 0.01`
- `success_angular_velocity_threshold = 0.05`

也就是说，任务成功要求：

- XY 对齐误差足够小
- 姿态误差足够小
- Z gap 足够小
- 末端线速度足够小
- 末端角速度足够小

## 本次备份结果摘要

基于 `metrics/episode_metrics.csv` 的简单统计：

- 总 episode 数：`106411`
- 全程平均 success rate：约 `99.30%`
- 最后 `50` 条 episode success rate：`100%`
- 最后 `100` 条 episode success rate：`100%`
- 最后 `200` 条 episode success rate：`100%`
- 最后 `100` 条 episode 平均 return：约 `2104.07`

最后 `100` 条 episode 的平均误差大致为：

- `xy_error ≈ 0.000351`
- `z_gap ≈ 0.000468`
- `rot_error ≈ 0.030589`
- `insertion_depth ≈ 0.044100`

## 建议的使用方式

这份备份适合用于：

- 后续 checkpoint eval
- 组会 / 展示材料
- 与后续 reward / success 判据改动做对照

如需做纯策略评估，可使用：

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --checkpoint_path /home/lrb/Workspace/hil_serl_mujoco/train/backup/20260430 \
  --eval_checkpoint_step 4995000 \
  --eval_n_trajs 10
```

如需录制总 MP4，可额外加：

```bash
--actor_viewer none \
--eval_video_path /home/lrb/Workspace/hil_serl_mujoco/train/output/eval/mujoco_insertion_backup_20260430.mp4
```
