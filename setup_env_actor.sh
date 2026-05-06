#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-hil_serl_actor}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_REPO_PATH="${SERL_ROBOT_INFRA_SIM_PATH:-}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_UDEV="${SKIP_UDEV:-0}"
APT_LOCK_TIMEOUT="${APT_LOCK_TIMEOUT:-300}"
CONDA_CHANNEL_ARGS=(--override-channels -c conda-forge)
CONSTRAINTS_FILE="${PROJECT_DIR}/constraints/actor_cpu_py311.txt"

info()  { echo -e "\033[92m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[93m[WARN]\033[0m $*"; }
error() { echo -e "\033[91m[ERROR]\033[0m $*"; exit 1; }

if [[ "$(uname)" != "Linux" ]]; then
    error "setup_env_actor.sh 仅适用于 Ubuntu/Linux。"
fi

if ! command -v apt-get &>/dev/null; then
    error "未找到 apt-get。当前脚本面向 Ubuntu/Debian 系统。"
fi

if command -v conda &>/dev/null; then
    CONDA_BIN="$(command -v conda)"
elif [[ -f "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
else
    error "未找到 Conda。"
fi

eval "$($CONDA_BIN shell.bash hook 2>/dev/null)" \
    || eval "$($CONDA_BIN shell.zsh hook 2>/dev/null)" \
    || true

info "Conda 安装将忽略本机 .condarc，只使用 conda-forge，避免 defaults channel 的 Anaconda ToS 阻塞"

if [[ -z "$SIM_REPO_PATH" ]]; then
    if [[ -d "$PROJECT_DIR/../sim" ]]; then
        SIM_REPO_PATH="$PROJECT_DIR/../sim"
    else
        SIM_REPO_PATH="$PROJECT_DIR/../sim"
    fi
fi

if [[ "$SKIP_APT" == "1" ]]; then
    warn "已设置 SKIP_APT=1，跳过 Ubuntu 系统依赖安装。"
else
    info "安装 Ubuntu 系统依赖"
    if ! sudo apt-get -o DPkg::Lock::Timeout="$APT_LOCK_TIMEOUT" update; then
        error "apt-get update 失败。请先修复系统里的失效源；如果这些系统依赖已经装好，也可以使用 SKIP_APT=1 ./setup_env_actor.sh 跳过此步骤。"
    fi

    sudo apt-get -o DPkg::Lock::Timeout="$APT_LOCK_TIMEOUT" install -y \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libhidapi-dev \
        libhidapi-hidraw0 \
        libsm6 \
        libxext6 \
        libxrender1
fi

if [[ "$SKIP_UDEV" == "1" ]]; then
    warn "已设置 SKIP_UDEV=1，跳过 SpaceMouse udev 规则安装。"
elif [[ -f "$SIM_REPO_PATH/99-spacemouse.rules" ]]; then
    info "安装 SpaceMouse udev 规则"
    sudo install -m 0644 "$SIM_REPO_PATH/99-spacemouse.rules" /etc/udev/rules.d/99-spacemouse.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=hidraw || warn "udevadm trigger 失败；请拔插 SpaceMouse 后重试。"
else
    warn "未找到 sim/99-spacemouse.rules，跳过 SpaceMouse udev 规则安装。"
fi

if conda env list | grep -qw "$ENV_NAME"; then
    warn "Conda 环境 '$ENV_NAME' 已存在，跳过创建。"
else
    info "创建 Conda 环境: $ENV_NAME"
    conda create "${CONDA_CHANNEL_ARGS[@]}" -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

info "升级 pip / setuptools / wheel"
python -m pip install --upgrade pip setuptools wheel

if [[ -d "$SIM_REPO_PATH" ]]; then
    info "安装 sibling sim 仓: $SIM_REPO_PATH"
    python -m pip install -e "$SIM_REPO_PATH"
else
    warn "未找到 sibling sim 仓，请确保 serl_robot_infra sim 包已提前安装。"
fi

info "安装 hil_serl_train"
python -m pip install -e "$PROJECT_DIR[distributed]"

info "清理与 actor 兼容路线冲突的旧 GPU/ML 依赖"
python -m pip uninstall -y \
    stable-baselines3 \
    torch \
    tensorflow \
    tf-keras \
    keras \
    cuda-toolkit \
    cuda-bindings \
    triton \
    nvidia-cudnn-cu13 \
    nvidia-cusparselt-cu13 \
    nvidia-nccl-cu13 \
    nvidia-nvshmem-cu13 \
    nvidia-cuda-runtime-cu13 \
    nvidia-cublas-cu13 \
    nvidia-cuda-cupti-cu13 \
    nvidia-cuda-nvrtc-cu13 \
    nvidia-cufft-cu13 \
    nvidia-curand-cu13 \
    nvidia-cusolver-cu13 \
    nvidia-cusparse-cu13 \
    nvidia-nvjitlink-cu13 || true

info "安装 actor 精确约束版本（CPU JAX 0.6.2）"
python -m pip install --upgrade --force-reinstall -r "$CONSTRAINTS_FILE"

info "安装可选轻量网页 viewer 依赖（Pinocchio + MeshCat）"
conda install "${CONDA_CHANNEL_ARGS[@]}" -n "$ENV_NAME" pinocchio -y \
    || warn "Pinocchio 安装失败；--actor_viewer meshcat 将不可用。"
python -m pip install meshcat==0.3.2 \
    || warn "MeshCat 安装失败；--actor_viewer meshcat 将不可用。"
python -m pip uninstall -y typing \
    || true

python -c "
import jax
from serl_robot_infra.configs.insertion_config import make_insertion_config
from serl_robot_infra.envs.franka_insertion_env import FrankaInsertionEnv
from serl_launcher.agents.continuous.sac import SACAgent
print('jax devices:', jax.devices())
env = FrankaInsertionEnv(config=make_insertion_config())
env.reset()
env.close()
print('actor env ok')
print('serl_launcher ok')
print('serl_robot_infra ok')
"

echo
info "环境已准备完成。"
info "激活环境: conda activate $ENV_NAME"
info "启动 actor: python scripts/train_rlpd_mujoco.py --exp_name mujoco_insertion --actor --ip localhost --checkpoint_path output/mujoco_insertion_actor"
warn "如果 SpaceMouse 已经插着但仍无法打开，请拔插一次设备后重试。"
