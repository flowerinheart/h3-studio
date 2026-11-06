#!/usr/bin/env bash
# Build the existing MiniMax H3 WebUI. Weights and third-party checkouts go to
# $H3_EXTERNAL (default: <repo>/external) and are not part of the git repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="${H3_EXTERNAL:-$ROOT/external}"
SRC="$EXTERNAL/src"
COMFY="$EXTERNAL/ComfyUI"
VENV="$EXTERNAL/venv"
MODELS="$EXTERNAL/models"
LOG="$EXTERNAL/logs"
mkdir -p "$SRC" "$MODELS" "$LOG" "$EXTERNAL/outputs"

ARCH="$(uname -m)"
if [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
fi
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_HOST="${PIP_HOST:-pypi.tuna.tsinghua.edu.cn}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT
export PIP_DISABLE_PIP_VERSION_CHECK=1

COMFY_COMMIT="${COMFY_COMMIT:-e0a32bce84f4603f2b61b16b50f8958e68fd6171}"
REGEN_COMMIT="${REGEN_COMMIT:-9b686c5ffd380e9d1351f9b8beef6d20d703ca3e}"

log() { printf '== %s\n' "$*"; }

pip_install() {
  "$VENV/bin/python" -m pip install --retries 8 --timeout 120 \
    -i "$PIP_INDEX" --trusted-host "$PIP_HOST" "$@"
}

clone_pin() {
  local dest="$1" commit="$2"
  shift 2
  if [[ -d "$dest/.git" ]] && git -C "$dest" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "$dest" checkout --detach --force "$commit"
    return
  fi
  local url
  for url in "$@"; do
    log "clone $url"
    if git clone "$url" "$dest"; then
      git -C "$dest" checkout --detach --force "$commit"
      return
    fi
    # A failed clone must not leave a half directory for the next mirror.
    if [[ -d "$dest" ]]; then
      find "$dest" -mindepth 1 -delete
      rmdir "$dest" 2>/dev/null || true
    fi
  done
  echo "无法克隆 $(basename "$dest")，已尝试的镜像都失败" >&2
  exit 1
}

have_file() {
  local path="$1" size="$2"
  [[ -f "$path" ]] && [[ "$(stat -c%s "$path")" == "$size" ]]
}

download_weight() {
  local rel="$1" size="$2"
  local dest="$MODELS/$rel"
  mkdir -p "$(dirname "$dest")"
  if have_file "$dest" "$size"; then
    log "权重已在 $rel"
    return
  fi
  log "从 $HF_ENDPOINT 下载 $rel"
  "$VENV/bin/huggingface-cli" download Comfy-Org/MiniMax-H3 "$rel" \
    --repo-type model --local-dir "$MODELS"
  if ! have_file "$dest" "$size"; then
    echo "权重大小不符: $dest" >&2
    exit 1
  fi
}

install_torch() {
  if "$VENV/bin/python" - << 'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    log "torch 已可用: $("$VENV/bin/python" -c 'import torch; print(torch.__version__)')"
    return
  fi
  local index
  if [[ "$ARCH" == "aarch64" ]]; then
    # NVIDIA CUDA. Do not install a ROCm wheel on Thor.
    local indexes=(
      "https://mirrors.aliyun.com/pytorch-wheels/cu130"
      "https://mirrors.cloud.tencent.com/nvidia-pytorch/cu130"
      "https://download.pytorch.org/whl/cu130"
      "https://download.pytorch.org/whl/cu128"
    )
  elif [[ "$ARCH" == "x86_64" ]]; then
    # AMD ROCm. Do not install a CUDA wheel on this host.
    local indexes=(
      "https://mirrors.aliyun.com/pytorch-wheels/rocm6.3"
      "https://download.pytorch.org/whl/rocm6.3"
      "https://download.pytorch.org/whl/rocm6.2"
    )
  else
    echo "不支持的架构: $ARCH" >&2
    exit 1
  fi
  local ok=0
  for index in "${indexes[@]}"; do
    log "安装 torch，索引 $index"
    # Do not add a generic PyPI extra index here. It can select a newer torch
    # build that is not the CUDA/ROCm wheel from $index.
    if "$VENV/bin/python" -m pip install --retries 5 --timeout 180 \
        --index-url "$index" \
        --trusted-host "$PIP_HOST" --trusted-host download.pytorch.org \
        --trusted-host mirrors.aliyun.com --trusted-host mirrors.cloud.tencent.com \
        'torch' 'torchvision'; then
      if "$VENV/bin/python" - << 'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
      then
        ok=1
        break
      fi
      log "该索引的 torch 看不到 GPU，换下一个"
    fi
  done
  if [[ "$ok" != 1 ]]; then
    echo "torch 安装后 torch.cuda.is_available() 仍为 false" >&2
    exit 1
  fi
}

log "架构 $ARCH，外部目录 $EXTERNAL"
if ! command -v python3 >/dev/null; then
  echo "需要 python3" >&2
  exit 1
fi
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install -U pip -i "$PIP_INDEX" --trusted-host "$PIP_HOST"
install_torch

# Platform-specific attention packages. HIP/CK is never downloaded on aarch64.
if [[ "$ARCH" == "aarch64" ]]; then
  log "aarch64/CUDA：使用 PyTorch 注意力，不安装 ROCm 或 HIP 算子"
elif [[ "$ARCH" == "x86_64" ]]; then
  log "x86_64/ROCm：安装 comfy-kitchen。gfx1151 的 CK 二进制不在仓库里，缺省走 PyTorch 注意力"
fi

clone_pin "$COMFY" "$COMFY_COMMIT" \
  "https://gitclone.com/github.com/comfyanonymous/ComfyUI.git" \
  "https://gh.llkk.cc/https://github.com/comfyanonymous/ComfyUI.git" \
  "https://github.com/comfyanonymous/ComfyUI.git"
clone_pin "$SRC/ComfyUI-Open-H3-Regenerate" "$REGEN_COMMIT" \
  "https://gitclone.com/github.com/rdh9616/ComfyUI-Open-H3-Regenerate.git" \
  "https://gh.llkk.cc/https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate.git" \
  "https://github.com/rdh9616/ComfyUI-Open-H3-Regenerate.git"

grep -v -E '^(torch|torchvision|torchaudio)([<>=~! ].*)?$' "$COMFY/requirements.txt" > "$LOG/requirements.txt"
pip_install -r "$LOG/requirements.txt" huggingface_hub opencv-python-headless

download_weight diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors 40225724176
download_weight text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors 51506295256
download_weight vae/minimax_h3_video_vae_fp16.safetensors 5207808496
download_weight vae/minimax_h3_audio_vae_fp32.safetensors 605254808

mkdir -p "$COMFY/custom_nodes" "$MODELS/loras"
ln -sfn "$ROOT/runtime/h3-swap-simple-ui" "$COMFY/custom_nodes/h3-swap-simple-ui"
ln -sfn "$ROOT/runtime/h3-platform-attention" "$COMFY/custom_nodes/h3-platform-attention"
ln -sfn "$SRC/ComfyUI-Open-H3-Regenerate" "$COMFY/custom_nodes/ComfyUI-Open-H3-Regenerate"
cat > "$COMFY/extra_model_paths.yaml" << EOF
h3:
  base_path: $MODELS
  is_default: true
  diffusion_models: diffusion_models/
  text_encoders: text_encoders/
  vae: vae/
  loras: loras/
EOF

log "构建完成。启动: $ROOT/deploy/start.sh"
