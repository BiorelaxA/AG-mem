#!/usr/bin/env bash
set -Eeuo pipefail
agmem_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_project"
export CUDA_VISIBLE_DEVICES="${AGMEM_REPAIR_GPU:-5}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
exec python -u -m agmem.repair_fallbacks_gpu \
    --accept-length-exceptions \
    --config configs/repair_longmemeval_s_fallbacks_len500.yaml \
    --source-root runs/parent_child_directories/qwen27b_adjacent_qwen4b_31535150_incremental_v2 \
    --previous-repairs runs/parent_child_directories/longmemeval_s_31535150_fallback_repair_20261114 \
    --datasets split_longmemeval_s --splits train validation test \
    --workers "${AGMEM_REPAIR_WORKERS:-64}" --max-num-seqs "${AGMEM_REPAIR_MAX_NUM_SEQS:-64}" "$@"
