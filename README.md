# hil_serl_mujoco_train

`train/` 是本工作区的训练侧仓库，负责：

- `serl_launcher`
- `scripts/train_rlpd_mujoco.py`
- `mujoco_insertion` 实验配置
- actor / learner 分布式训练
- replay buffer、demo buffer、checkpoint

它不维护第二份 MuJoCo 环境，而是直接依赖旁边 `sim/` 中安装的 `robot_controllers_mujoco`。

## 当前训练主线

当前默认训练主线为：

- 任务：`mujoco_insertion`
- 算法：state-based HIL-RLPD
- 数据策略：少量 demo 预热 + 在线 intervention

## 环境安装

### 本地 actor

macOS：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
./setup_env_actor_mac.sh
conda activate hil_serl_actor
```

Ubuntu：

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
./setup_env_actor_ubuntu.sh
conda activate hil_serl_actor
```

### 远端 learner

```bash
cd /home/lrb/Workspace/hil_serl_mujoco/train
./setup_env_autodl.sh
conda activate hil_serl_train
```

## 版本与兼容性策略

当前训练侧版本固定为一组兼容组合：

- Python 3.11
- actor：CPU JAX `0.6.2`
- learner：GPU JAX `0.6.2` + `jax-cuda12-plugin[with-cuda]`
- `tensorflow-probability==0.25.0`
- `distrax==0.1.5`
- `flax==0.10.7`
- `optax==0.2.8`
- `numpy==1.26.4`
- `scipy==1.15.3`

精确约束文件位于：

- `train/constraints/actor_cpu_py311.txt`
- `train/constraints/learner_cuda_py311.txt`

## GPU learner 的 CUDA 说明

`hil_serl_train` 采用 JAX bundled CUDA/cuDNN，而不是依赖 shell 中手工注入的系统 CUDA toolkit。

激活 `hil_serl_train` 时，`activate.d` 会自动：

- 过滤所有 `/usr/local/cuda*`
- 注入 learner 所需的 bundled CUDA 动态库路径
- 保留其余非 CUDA 的业务路径

检查命令：

```bash
conda activate hil_serl_train
echo $LD_LIBRARY_PATH | tr ':' '\n'
```

预期现象：

- 不应再出现 `/usr/local/cuda*`
- 应出现类似：
  - `.../site-packages/nvidia/nvjitlink/lib`
  - `.../site-packages/nvidia/cusparse/lib`
  - `.../site-packages/nvidia/cudnn/lib`

基础 GPU 自检：

```bash
python -c "import jax; print(jax.devices())"
python -c "import jax, jax.numpy as jnp; x=jnp.ones((2048,2048), dtype=jnp.float32); y=(x@x).block_until_ready(); print(jax.devices(), y[0,0])"
```

## 启动命令

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

## W&B 行为

当前默认策略如下：

- 如果当前 shell 提供了 `WANDB_API_KEY`，则按 `online` 模式运行
- 如果未提供 `WANDB_API_KEY`，则自动以 `offline` 模式运行
- 如果传 `--debug`，则完全禁用 wandb

常用控制方式：

- 强制在线：`export WANDB_API_KEY=...` 或 `export WANDB_MODE=online`
- 强制离线：`export WANDB_MODE=offline`
- 完全禁用：启动命令加 `--debug`

## 运行前置条件

在启动训练前，需要先完成：

1. 在 `sim/` 中录制 insertion demos
2. 将 `demo_data/insertion_demos.pkl` 放到 learner 可访问的位置
3. 保证 learner 机器上的 `sim/` 已经可导入

`train/` 不负责录制 demo。demo 统一由 `sim/scripts/record_demos_mujoco.py` 生成。

## 运行时行为说明

- learner 启动后会先加载 `demo_buffer`
- 然后等待 actor 填充 `replay_buffer`
- 只有当 `replay_buffer` 达到 `training_starts` 后，learner 才会开始正式训练

因此，看到类似：

```text
Filling up replay buffer: 0/1000
```

并不表示程序出错，而是 learner 正在等待 actor 的在线交互数据。

## 依赖关系

推荐目录布局：

```text
hil_serl_mujoco/
├── sim/
└── train/
```

`train` 脚本会优先尝试安装或导入旁边的 `sim/`。如果 learner 报 `robot_controllers_mujoco` 找不到，通常说明 `sim/` 未安装或路径不符合预期。

## 进一步说明

更完整的端到端部署流程，包括 demo 录制、文件传输、远端 learner 与本地 actor 的启动顺序，统一写在父仓 [`README.md`](/home/lrb/Workspace/hil_serl_mujoco/README.md)。
