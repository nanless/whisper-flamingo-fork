#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_ID=${GPU_ID:-1}
CONFIG=${CONFIG:-${ROOT}/config/audio/audio_en_large_vc2_clean.local.yaml}
cd "${ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u whisper_ft_muavic.py "${CONFIG}"
