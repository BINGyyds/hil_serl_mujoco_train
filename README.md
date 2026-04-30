# hil_serl_mujoco_train

`train/` 负责 `mujoco_insertion` 的 actor / learner 训练、checkpoint eval、MP4 导出和本地 metrics。

当前主线：

- 算法：state-based SAC / RLPD
- 数据：demo buffer + replay buffer
- 评估：success stats + MP4 + metrics

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
  --demo_path demo_data/insertion_demos.pkl \
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
  --norender_actor
```

## checkpoint eval

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --checkpoint_path output/mujoco_insertion_local \
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

## 本地训练指标

learner 会持续写入：

```text
output/.../metrics/episode_metrics.csv
```

默认输出：

- `reward_curve.png`
- `success_curve.png`
- `error_curve.png`

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
