#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${MODE:-avsr}
GPU_ID=${GPU_ID:-1}
DATA_ROOT=${DATA_ROOT:-${ROOT}}
BEAM_SIZE=${BEAM_SIZE:-15}
SNR=${SNR:-1000}
NOISE_FN=${NOISE_FN:-${ROOT}/noise/babble/lrs3/test.tsv}
WHISPER_PATH=${WHISPER_PATH:-/root/.cache/whisper}
PYTHON_BIN=${PYTHON_BIN:-/root/miniforge3/envs/whisper-flamingo-repro/bin/python}
AVHUBERT_PATH=${AVHUBERT_PATH:-${ROOT}/av_hubert/avhubert}
AVHUBERT_CKPT=${AVHUBERT_CKPT:-${ROOT}/models/large_noise_pt_noise_ft_433h_only_weights.pt}
DECODE_PATH=${DECODE_PATH:-${ROOT}/decode}

[[ -x ${PYTHON_BIN} ]] || { echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1; }

if [[ ${MODE} == avsr ]]; then
  CHECKPOINT=${CHECKPOINT:-${ROOT}/models/whisper-flamingo_en_large_vc2_clean.pt}
  USE_AV_HUBERT=1
  AV_FUSION=separate
elif [[ ${MODE} == asr ]]; then
  CHECKPOINT=${CHECKPOINT:-${ROOT}/models/whisper_en_large_vc2_clean.pt}
  USE_AV_HUBERT=0
  AV_FUSION=None
else
  echo "MODE must be avsr or asr" >&2
  exit 2
fi

for path in "${CHECKPOINT}" "${NOISE_FN}"; do
  [[ -f ${path} ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
if [[ ${MODE} == avsr ]]; then
  [[ -f ${AVHUBERT_CKPT} ]] || { echo "Missing AV-HuBERT checkpoint: ${AVHUBERT_CKPT}" >&2; exit 1; }
fi

cd "${ROOT}"
CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" -u whisper_decode_video.py \
  --lang en \
  --model-type large-v2 \
  --data-root "${DATA_ROOT}" \
  --noise-snr "${SNR}" \
  --noise-fn "${NOISE_FN}" \
  --modalities "${MODE}" \
  --use_av_hubert_encoder "${USE_AV_HUBERT}" \
  --av_fusion "${AV_FUSION}" \
  --checkpoint-path "${CHECKPOINT}" \
  --whisper-path "${WHISPER_PATH}" \
  --av-hubert-path "${AVHUBERT_PATH}" \
  --av-hubert-ckpt "${AVHUBERT_CKPT}" \
  --beam-size "${BEAM_SIZE}" \
  --normalizer fairseq \
  --decode-path "${DECODE_PATH}"
