# hil_serl_mujoco_train

`train/` 是强化学习训练侧仓库，负责：

- `serl_launcher`
- `scripts/train_rlpd_mujoco.py`
- `scripts/experiments/mujoco_insertion`
- actor / learner 分布式训练
- replay buffer、demo buffer、checkpoint

它不再维护第二份 MuJoCo 环境和模型，而是直接依赖旁边 `sim/` 里的 `robot_controllers_mujoco`。

## 当前主线

当前默认主线已经切到：

- `mujoco_insertion`
- 少量 demo 预热
- 在线 HIL-RLPD

## 环境安装

### 本地 actor

```bash
cd train
./setup_env_actor_mac.sh
conda activate hil_serl_actor
```

如果本地 actor 机器是 Ubuntu：

```bash
cd train
./setup_env_actor_ubuntu.sh
conda activate hil_serl_actor
```

### 远端 learner

```bash
cd train
./setup_env_autodl.sh
conda activate hil_serl_train
```

这两个脚本都会优先尝试安装旁边的 `sim/`，所以推荐目录保持：

```text
hil_serl_mujoco/
├── sim/
└── train/
```

## 训练入口

### learner

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --learner \
  --demo_path demo_data/insertion_demos.pkl \
  --checkpoint_path output/mujoco_insertion
```

### actor

```bash
python scripts/train_rlpd_mujoco.py \
  --exp_name mujoco_insertion \
  --actor \
  --ip localhost \
  --checkpoint_path output/mujoco_insertion_actor
```

## 依赖关系

- `sim/`
  - MuJoCo 环境、MJCF、SpaceMouse、demo 录制
- `train/`
  - learner / actor、训练配置、buffer、checkpoint

`train` 仓默认不单独录 demo。插入 demo 统一在 `sim/scripts/record_demos_mujoco.py` 中录制。

## 推荐操作顺序

1. 先在 `sim_env` 中录 `demo_data/insertion_demos.pkl`
2. 把 demo 传到远端 learner
3. 远端启动 learner
4. 本地启动 actor
5. 再做 SpaceMouse intervention

## 完整部署文档

跨本地 `sim/`、本地 actor、远端 learner 的完整部署步骤，统一写在父仓 [`README.md`](/home/lrb/Workspace/hil_serl_mujoco/README.md)。
