#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="${H3_EXTERNAL:-$ROOT/external}"
export H3_WEBUI_OUTPUT="${H3_WEBUI_OUTPUT:-$EXTERNAL/outputs/h3-webui}"
mkdir -p "$H3_WEBUI_OUTPUT"
cd "$EXTERNAL/ComfyUI"
exec "$EXTERNAL/venv/bin/python" main.py \
  --listen "${H3_LISTEN:-0.0.0.0}" --port "${H3_PORT:-8188}" \
  --cache-none --reserve-vram 8 \
  "$@"
