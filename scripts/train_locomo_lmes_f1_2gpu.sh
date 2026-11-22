#!/usr/bin/env bash
# One Reader GPU (4) + shared frozen Task/Judge GPU (5), with repaired mixed directories.
set -Eeuo pipefail
agmem_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_project"
export CUDA_VISIBLE_DEVICES="${AGMEM_TRAIN_GPUS:-${CUDA_VISIBLE_DEVICES:-4,5}}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
agmem_config=${AGMEM_TRAIN_CONFIG:-configs/train_dapo_locomo_lmes_hierarchical_f1_2gpu.yaml}
python -u -m agmem.prepare_mixed_hierarchical_training --config "$agmem_config"
agmem_args=()
if [[ -n ${AGMEM_TRAIN_OUTPUT_DIR:-} ]]; then agmem_args+=(--output-dir "$AGMEM_TRAIN_OUTPUT_DIR"); fi
exec python -u -m agmem.two_gpu_train --train-config "$agmem_config" "${agmem_args[@]}" "$@"
