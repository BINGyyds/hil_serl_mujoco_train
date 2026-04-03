#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-hil_serl_actor}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_REPO_PATH="${ROBOT_CONTROLLERS_MUJOCO_PATH:-}"

info()  { echo -e "\033[92m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[93m[WARN]\033[0m $*"; }
error() { echo -e "\033[91m[ERROR]\033[0m $*"; exit 1; }

if [[ "$(uname)" != "Darwin" ]]; then
    error "setup_env_actor_mac.sh 仅适用于 macOS。"
fi

if command -v conda &>/dev/null; then
    CONDA_BIN="$(command -v conda)"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
else
    error "未找到 Conda。"
fi

eval "$($CONDA_BIN shell.bash hook 2>/dev/null)" || eval "$($CONDA_BIN shell.zsh hook 2>/dev/null)" || true

if [ -z "$SIM_REPO_PATH" ]; then
    if [ -d "$PROJECT_DIR/../sim" ]; then
        SIM_REPO_PATH="$PROJECT_DIR/../sim"
    elif [ -d "$PROJECT_DIR/../robot_controllers_mujoco" ]; then
        SIM_REPO_PATH="$PROJECT_DIR/../robot_controllers_mujoco"
    else
        SIM_REPO_PATH="$PROJECT_DIR/../sim"
    fi
fi

if conda env list | grep -qw "$ENV_NAME"; then
    warn "Conda 环境 '$ENV_NAME' 已存在，跳过创建。"
else
    info "创建 Conda 环境: $ENV_NAME"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

info "升级 pip / setuptools / wheel"
pip install --upgrade pip setuptools wheel

info "安装 JAX CPU"
pip install jax jaxlib

if [ -d "$SIM_REPO_PATH" ]; then
    info "安装 sibling sim 仓: $SIM_REPO_PATH"
    pip install -e "$SIM_REPO_PATH"
else
    warn "未找到 sibling sim 仓，请确保 robot_controllers_mujoco 已提前安装。"
fi

info "安装 hil_serl_train"
pip install -e "$PROJECT_DIR[distributed,sb3]"

python -c "
from robot_controllers_mujoco.configs.insertion_config import make_insertion_config
from robot_controllers_mujoco.envs.franka_insertion_env import FrankaInsertionEnv
from serl_launcher.agents.continuous.sac import SACAgent
env = FrankaInsertionEnv(config=make_insertion_config())
env.reset()
env.close()
print('actor env ok')
print('serl_launcher ok')
"

echo
info "环境已准备完成。"
info "激活环境: conda activate $ENV_NAME"
info "启动 actor: python scripts/train_rlpd_mujoco.py --exp_name mujoco_insertion --actor --ip localhost --checkpoint_path output/mujoco_insertion_actor"
