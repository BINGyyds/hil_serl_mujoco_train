# hil_serl_mujoco_train

`train/` 负责 `mujoco_insertion` 的 actor / learner 训练、checkpoint eval、MP4 导出和本地 metrics。

当前主线：

- 算法：state-based SAC / RLPD
- 数据：demo buffer + replay buffer
- 评估：success stats + MP4 + metrics
- 仿真控制：effort scene + `SE(3)` torque control + NRIC

## 安装

本地 actor：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
./setup_env_actor.sh
conda activate hil_serl_actor
```

远端 learner：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
./setup_env_train.sh
conda activate hil_serl_train
```

## 启动 learner

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --learner \
  --demo_path ../sim/demo_data/insertion_demos.pkl \
  --checkpoint_path output/mujoco_insertion
```

## 启动 actor

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --ip localhost \
  --checkpoint_path output/mujoco_insertion_actor \
  --actor_viewer meshcat \
  --actor_render_fps 20
```

无窗口采样：

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --ip localhost \
  --checkpoint_path output/mujoco_insertion_actor \
  --actor_viewer none
```

## checkpoint eval

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --checkpoint_path output/mujoco_insertion \
  --eval_checkpoint_step -1 \
  --eval_n_trajs 10
```

按 1x 实时播放：

```bash
--eval_realtime_factor 1.0
```

## MP4 导出

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --checkpoint_path output/mujoco_insertion_local \
  --eval_checkpoint_step -1 \
  --eval_n_trajs 10 \
  --actor_viewer none \
  --eval_video_path output/eval/mujoco_insertion.mp4
```

当前会将所有 eval episode 拼接到一个总 MP4 中。

## task-case 泛化评估

默认 task-case eval 会固定 reset/preinsert anchor，只移动 hole/socket 和对应的 `goal_tip`，因此可以真实测试 policy 是否能主动对齐新 hole。日志中应看到 `reset_anchor=fixed`。

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --checkpoint_path backup/20260517 \
  --eval_checkpoint_step -1 \
  --eval_n_trajs 20 \
  --eval_sample_task_case \
  --eval_task_case_range_preset wide \
  --eval_zero_reset_perturbation \
  --actor_viewer none
```

边界测试可扩大 task-case 采样范围：

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --checkpoint_path backup/20260517 \
  --eval_checkpoint_step -1 \
  --eval_n_trajs 20 \
  --eval_sample_task_case \
  --eval_task_case_range_preset wide \
  --eval_task_case_xy_range_override 0.02 \
  --eval_task_case_yaw_range_override 0.4 \
  --eval_zero_reset_perturbation \
  --actor_viewer none
```

## 本地训练指标

learner 会持续写入：

```text
output/.../metrics/episode_metrics.csv
```

默认输出：

- `error_curve.png`
- `training_performance.png`

其他图和 PDF 通过绘图脚本的显式参数生成。

手动画图：

```bash
python scripts/plot_mujoco_metrics.py \
  --metrics_dir /home/lrb/Workspace/hil_serl_mujoco/train/output/mujoco_insertion_local/metrics
```

## 视觉扩展

当前默认训练是 state-only。仓库同时保留了：

- camera image 输入路径
- pixel / vision encoder 训练基础设施
- image-key 相关公共编码与数据路径

完整端到端流程见父仓 [README.md](/home/lrb/Workspace/hil_serl_mujoco/README.md)。
