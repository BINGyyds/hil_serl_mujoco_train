# backup

这个目录用于保存**值得长期保留的训练结果快照**，例如：

- 训练效果较好的 checkpoint
- 对应的 metrics 曲线
- 训练配置记录
- 后续用于汇报或展示的评估视频


## 20260430 训练备份

优化任务完成奖励，强化学习训练第一次成功

### 基本配置

- `max_steps = 5_000_000`
- `replay_buffer_capacity = 5_000_000`

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


## 20260513 训练备份

仓库结构优化后强化学习训练验证

### 基本配置

- `max_steps = 1_000_000`
- `replay_buffer_capacity = 1_000_000`

其他配置同0430训练

## 20260514 训练备份

Obsservation Action Controller全都放到body-frame下的SE(3)量进行训练，但是控制器依然是速度控制，训练依然是固定pitch和hole的位置，训练结果表明，在hole的xy坐标变化的额情况下效果还行，但是一旦hole的角度变化效果就很差，基本无法成功。

### 基本配置

- `max_steps = 500_000`
- `replay_buffer_capacity = 500_000`

## 20260515 训练备份

在0514训练基础上将控制链路都替换为SE(3)阻抗控制，控制器采用力矩控制
训练得到的结果图成功率等指标收敛的很好，但是评估模型性能的时候表现很差，很奇怪
验证结论是：当前 policy 确实学会了“把 peg 拉回默认 hole”，但没有学会“当任务 frame 自己变了时，也做等价行为”

### 基本配置

- `max_steps = 5_000_000`
- `replay_buffer_capacity = 5_000_000`

### Reward 配置

提高了reward_rot_weight，希望agent能学到对齐姿态的policy，这点在完成任务方面很关键

- `reward_rot_weight = 1.2`

