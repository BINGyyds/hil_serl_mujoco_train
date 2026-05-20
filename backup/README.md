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

## 20260517 训练备份

在0515训练基础上
1. 对Observation进行改造，新增wrench_control_body：6 维局部 wrench，定义在当前 control frame（tool_mount / attachment frame），现在Observation是18维向量
2. 对Reward进行重构，使用 pose_error_body、twist_body、wrench_control_body 构造主要 dense reward，同时保留任务语义最强的 task-frame 项。

误差向量：
* e_p = ||pose_error_body[:3]||
* e_r = ||pose_error_body[3:]||
* v_lin = ||twist_body[:3]||
* v_ang = ||twist_body[3:]||
* f_lat = ||wrench_control_body[:2]||
* tau_lat = ||wrench_control_body[3:5]||

权重：
* w_ep = 1.0
* w_er = 3.0
* w_vlin = 0.15
* w_vang = 0.10
* w_flat = 0.20
* w_tlat = 0.10

Reward：
r =
  - w_ep   * e_p
  - w_er   * e_r
  - w_vlin * v_lin
  - w_vang * v_ang
  - w_flat * f_lat
  - w_tlat * tau_lat
  + depth_reward_if_aligned
  - step_penalty
  + success_bonus
  - failure_penalty

## 训练结果

与0515结果类似，目前是能够对hole的XY做泛化，但是对yaw的泛化不足
![alt text](<Screenshot from 2026-05-18 15-27-46.png>)
![alt text](<Screenshot from 2026-05-18 15-28-11.png>)

## 20260517-2 训练备份

在20260517基础上，训练yaw-only的泛化agent

### 基本配置

- `max_steps = 2_000_000`
- `replay_buffer_capacity = 2_000_000`

训练结果表现比较差

## 20260517-3 训练备份

在20260517-2基础上，新增obeservation：q和q_dot，继续训练yaw-only的泛化agent

### 基本配置

- `max_steps = 5_000_000`
- `replay_buffer_capacity = 5_000_000`

训练结果表现比较差，推测可能是接触惩罚导致后期agent为了减小接触惩罚而刻意不进去

## 20260519 训练备份

去掉接触惩罚进行训练，继续训练yaw-only的泛化agent

训练结果依旧表现很差，现在是学习不到末端yaw角度改变，并完成任务的能力了

目前排查出了一个问题是：goal_tip_rot和socket_rot是有差别的，等待解决！！

## 20260520 排查总结

这轮排查把此前 “hole yaw 泛化失败” 拆成了两个 eval 几何问题。

### 1. task-case yaw 的 goal 姿态定义

旧版 task-case yaw evaluation 中：

```text
R_goal_case = R_goal_default * Rz(yaw)
```

由于 `goal_tip_rot` 和 `socket/task frame` 默认存在接近 180 度的固定工具-孔姿态偏置，二者并不是同一根 yaw 轴。直接在 `goal_tip_rot` 右乘 yaw 会导致：

```text
R_task_case^T * R_goal_case
```

随 yaw 改变。也就是说，旧 eval 不是单纯旋转 hole，而是在改变工具-孔目标姿态关系。

现在修正为：

```text
B0 = R_task_default^T * R_goal_default
R_goal_case = R_task_case * B0
```

这样 hole yaw 后，工具-孔相对姿态偏置保持恒定。

### 2. task-case 不应默认移动 reset/preinsert anchor

继续看视频后发现，旧 task-case eval 会让 `preinsert_tip_transform` 跟着 moved `goal_tip_transform` 走。这样每个 episode 初始时 peg 已经在新 hole 上方并且姿态基本对齐，policy 本质上只是在向下插入。

当前主线 eval 改为：

- hole/socket 位姿按 `task_case` 改变
- `goal_tip` 跟随 socket 正确改变，并保持默认工具-孔姿态偏置
- reset/preinsert 起点固定在默认 preinsert anchor
- 只有显式传 `--eval_task_case_moves_reset_anchor` 时，才恢复旧的 task-frame prealigned eval

日志中应看到：

```text
reset_anchor=fixed
```

### 3. 重新评估结论

使用 `20260517/checkpoint_4995000`，18 维 observation：

```text
pose_error_body + twist_body + wrench_control_body
```

修正后重新评估：

- pure hole yaw，reset 清零：`20/20`
- aligned task-case，reset 清零：`20/20`
- aligned task-case，default reset：`20/20`
- wide task-case，default reset：`20/20`
- wide task-case，fixed reset anchor，reset 清零，视频验证：`20/20`

其中 wide task-case 覆盖约：

- XY：`±10mm`
- yaw：`±0.25rad`

进一步加入 eval-only override 后，可测试更大的边界，例如：

```bash
--eval_task_case_xy_range_override 0.02
--eval_task_case_yaw_range_override 0.4
```

即 XY `±20mm`、yaw `±0.4rad`。

### 4. 当前判断

此前 yaw 泛化失败主要来自 evaluation/task-case 几何定义不一致，而不是 policy 缺乏 yaw 泛化能力。

当前推荐主线回到稳定的 18 维 observation checkpoint。32 维 `q_pos/q_vel` joint-observation 分支训练不稳定，暂时保留为可选诊断 preset：

```bash
--proprio_preset wrench_joint
```

默认主线仍使用：

```bash
--proprio_preset wrench
```

由于最终定位到的是 eval/task-case 几何问题，而不是接触惩罚问题，主线 reward 默认值也恢复到 20260517 配置：

```text
reward_lateral_force_weight = 0.20
reward_lateral_torque_weight = 0.10
```

训练长度也保持为 20260517 稳定配置：

```text
max_steps = 5_000_000
replay_buffer_capacity = 5_000_000
```
