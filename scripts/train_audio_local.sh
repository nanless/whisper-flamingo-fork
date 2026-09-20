#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_ID=${GPU_ID:-1}
CONFIG=${CONFIG:-${ROOT}/config/audio/audio_en_large_vc2_clean.local.yaml}
PYTHON_BIN=${PYTHON_BIN:-/root/miniforge3/envs/whisper-flamingo-repro/bin/python}
[[ -x ${PYTHON_BIN} ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }
cd "${ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" -u whisper_ft_muavic.py "${CONFIG}"
