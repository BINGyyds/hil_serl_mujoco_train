# hil_serl_mujoco_train

这个目录是未来独立 GitHub 仓库 `hil_serl_mujoco_train` 的本地形态，负责强化学习训练侧能力。

说明：

- GitHub 仓库名：`hil_serl_mujoco_train`
- 父仓挂载目录名：`train/`
- 当前 Python 包名 / 发行名：`hil_serl_train`

## 包含内容

- `serl_launcher`
- `scripts/train_sac_reach.py`
- `scripts/train_rlpd_mujoco.py`
- `scripts/experiments/mujoco_reach/`

当前主线只保留 `mujoco_reach`。

## 依赖关系

训练仓不再维护第二份 MuJoCo 环境和模型。

它直接依赖：

- `robot_controllers_mujoco`

也就是 `sim/` 对应的独立仓库。

## 本地 actor 安装

```bash
bash setup_env_actor_mac.sh
conda activate hil_serl_actor
```

## AutoDL learner 安装

```bash
bash setup_env_autodl.sh
conda activate hil_serl_train
```

## 为什么有两个 setup 脚本

- `setup_env_actor_mac.sh`
  - 给本地 Mac 的 actor 用
  - 安装 CPU JAX
  - 用于本地启动 `--actor`

- `setup_env_autodl.sh`
  - 给远程 AutoDL learner 用
  - 安装 CUDA JAX
  - 用于远程启动 `--learner`

它们分别服务两台不同机器，所以保留两个脚本是有必要的。

## 训练入口

### learner

```bash
python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --learner \
  --demo_path demo_data/reach_demos.pkl \
  --checkpoint_path output/reach_hil
```

### actor

```bash
python scripts/train_rlpd_mujoco.py --exp_name mujoco_reach --actor --ip localhost
```

## 备注

- Mac actor 环境用 CPU JAX 即可
- AutoDL learner 环境使用 CUDA JAX
- 如果你把 `sim/` 和 `train/` 拆成独立仓库，推荐并排 clone，方便安装脚本自动找到 `robot_controllers_mujoco`

## 推送到 GitHub

```bash
cd train
git init
git add .
git commit -m "Initial commit for hil_serl_mujoco_train"
git branch -M main
git remote add origin git@github.com:<your-name>/hil_serl_mujoco_train.git
git push -u origin main
```
