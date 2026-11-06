# H3 Studio

把现有的 MiniMax H3 ComfyUI 和 simple-swap WebUI 部署到 x86_64（ROCm）或 aarch64（NVIDIA CUDA）。仓库里只有启动脚本和 WebUI。ComfyUI、自定义节点上游、PyTorch 和 BF16 权重都在构建时下载到外部目录，不进 git。

## 依赖怎么选

| 架构 | PyTorch | 注意力 |
| --- | --- | --- |
| `x86_64` | ROCm 轮子 | 检测到 gfx1151 才走现有 HIP/CK。其他情况用 PyTorch |
| `aarch64` | CUDA 轮子 | 只用 PyTorch。不安装 ROCm，不下载 HIP 算子 |

下载顺序：PyPI 用清华源，权重大文件用 `HF_ENDPOINT`（默认 `https://hf-mirror.com`），git 先试 `gitclone.com` 和 `gh.llkk.cc`，失败再用 GitHub。

权重是 Comfy-Org/MiniMax-H3 的 BF16 文件：

- `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors`
- `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`
- `vae/minimax_h3_video_vae_fp16.safetensors`
- `vae/minimax_h3_audio_vae_fp32.safetensors`

已经存在且字节数一致的文件会跳过。

## 部署

需要 Python 3.12、`python3-venv`、git、ffmpeg。Thor 上：

```sh
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-pip ffmpeg git curl build-essential
git clone https://github.com/flowerinheart/h3-studio.git
cd h3-studio
# 可选。不设的话外部文件放在仓库下的 external/，该目录被 git 忽略。
export H3_EXTERNAL=$HOME/h3-external
bash deploy/build.sh
bash deploy/start.sh
```

`start.sh` 监听 `127.0.0.1:8188`。WebUI 在 `http://127.0.0.1:8188/simple-swap/`。

另一台终端提交一条现有的 BF16 底稿（384 短边，约 1 秒，不用加速 LoRA）：

```sh
export H3_EXTERNAL=$HOME/h3-external
"$H3_EXTERNAL/venv/bin/python" deploy/smoke.py
```

成功时打印 `VIDEO_OK` 和 `PROMPT_OK`。成片在 `$H3_EXTERNAL/outputs/h3-webui/videos/`。

本机如果装了 Paseo，成片路径仍要落在 `~/.paseo/streaming-video.json` 的 `allowedRoots` 里。没有这份配置的机器只检查文件能被 ffmpeg 解码。
