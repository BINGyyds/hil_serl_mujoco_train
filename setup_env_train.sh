#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-hil_serl_train}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_REPO_PATH="${ROBOT_CONTROLLERS_MUJOCO_PATH:-}"
CONDA_CHANNEL_ARGS=(--override-channels -c conda-forge)
CONSTRAINTS_FILE="${PROJECT_DIR}/constraints/learner_cuda_py311.txt"

info()  { echo -e "\033[92m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[93m[WARN]\033[0m $*"; }
error() { echo -e "\033[91m[ERROR]\033[0m $*"; exit 1; }

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

info "Conda 安装将忽略本机 .condarc，只使用 conda-forge，避免 defaults channel 的 Anaconda ToS 阻塞"

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
    conda create "${CONDA_CHANNEL_ARGS[@]}" -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

info "升级 pip / setuptools / wheel"
pip install --upgrade pip setuptools wheel

if [ -d "$SIM_REPO_PATH" ]; then
    info "安装 sibling sim 仓: $SIM_REPO_PATH"
    pip install -e "$SIM_REPO_PATH"
else
    warn "未找到 sibling sim 仓，请确保 robot_controllers_mujoco 已提前安装。"
fi

info "安装 hil_serl_train"
pip install -e "$PROJECT_DIR[distributed]"

info "清理与 learner 兼容路线冲突的旧 GPU/ML 依赖"
pip uninstall -y \
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

info "安装 learner 精确约束版本（JAX 0.6.2 + bundled CUDA/cuDNN）"
pip install --upgrade --force-reinstall -r "$CONSTRAINTS_FILE"

ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

cat > "$ACTIVATE_DIR/hil_serl_train_cuda_paths.sh" <<'EOF'
#!/usr/bin/env bash
export _HIL_SERL_TRAIN_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

_hil_serl_train_filtered_ld="$(
python - <<'PY'
import os

raw = os.environ.get("LD_LIBRARY_PATH", "")
parts = [p for p in raw.split(":") if p]
filtered = []
seen = set()
for part in parts:
    if part.startswith("/usr/local/cuda"):
        continue
    if part in seen:
        continue
    seen.add(part)
    filtered.append(part)
print(":".join(filtered))
PY
)"

_hil_serl_train_bundled_cuda="$(
python - <<'PY'
import os
import site

base = site.getsitepackages()[0]
ordered = [
    os.path.join(base, "nvidia", "nvjitlink", "lib"),
    os.path.join(base, "nvidia", "cublas", "lib"),
    os.path.join(base, "nvidia", "cuda_cupti", "lib"),
    os.path.join(base, "nvidia", "cuda_runtime", "lib"),
    os.path.join(base, "nvidia", "cudnn", "lib"),
    os.path.join(base, "nvidia", "cusparse", "lib"),
]
existing = [p for p in ordered if os.path.isdir(p)]
print(":".join(existing))
PY
)"

if [[ -n "$_hil_serl_train_bundled_cuda" && -n "$_hil_serl_train_filtered_ld" ]]; then
    export LD_LIBRARY_PATH="${_hil_serl_train_bundled_cuda}:${_hil_serl_train_filtered_ld}"
elif [[ -n "$_hil_serl_train_bundled_cuda" ]]; then
    export LD_LIBRARY_PATH="${_hil_serl_train_bundled_cuda}"
else
    export LD_LIBRARY_PATH="${_hil_serl_train_filtered_ld}"
fi

unset _hil_serl_train_filtered_ld
unset _hil_serl_train_bundled_cuda
EOF

cat > "$DEACTIVATE_DIR/hil_serl_train_cuda_paths.sh" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${_HIL_SERL_TRAIN_OLD_LD_LIBRARY_PATH+x}" ]]; then
    export LD_LIBRARY_PATH="${_HIL_SERL_TRAIN_OLD_LD_LIBRARY_PATH}"
    unset _HIL_SERL_TRAIN_OLD_LD_LIBRARY_PATH
fi
EOF

source "$ACTIVATE_DIR/hil_serl_train_cuda_paths.sh"

if ! python -c "
import jax
from serl_launcher.agents.continuous.sac import SACAgent
from robot_controllers_mujoco.configs.insertion_config import make_insertion_config
from robot_controllers_mujoco.envs.franka_insertion_env import FrankaInsertionEnv
print('jax devices:', jax.devices())
env = FrankaInsertionEnv(config=make_insertion_config())
env.reset()
env.close()
print('serl_launcher ok')
print('robot_controllers_mujoco ok')
"
then
    error "初始自检失败。请优先检查当前 shell 是否通过 .zshrc / ROS / 其他 SDK 全局注入了 /usr/local/cuda* 到 LD_LIBRARY_PATH。"
fi

if ! python -c "
import jax
import jax.numpy as jnp
x = jnp.ones((1024, 1024), dtype=jnp.float32)
y = (x @ x).block_until_ready()
print(jax.devices(), y[0, 0])
"
then
    error "GPU 计算自检失败。请检查激活后的 LD_LIBRARY_PATH 是否仍残留 /usr/local/cuda*，并确认 bundled CUDA 路径已被注入。"
fi

echo
info "环境已准备完成。"
info "激活环境: conda activate $ENV_NAME"
info "推荐 GPU: 当前主线 4090 足够，5090 适合后续更重实验。"
info "启动 learner: python scripts/train_rlpd_mujoco.py --exp_name mujoco_insertion --learner --demo_path demo_data/insertion_demos.pkl --checkpoint_path output/mujoco_insertion"
