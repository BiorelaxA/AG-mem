#!/usr/bin/env bash
# GPU0: Qwen3-Embedding-4B; GPU1/2: independent Qwen3.5-27B Writers.
set -Eeuo pipefail
agmem_directory_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_directory_project"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
exec python -u -m agmem.build_adjacent_directories_gpu \
    --config configs/build_adjacent_qwen_locomo_lmes.yaml \
    --workers "${AGMEM_WRITER_WORKERS:-128}" \
    --max-num-seqs "${AGMEM_WRITER_MAX_NUM_SEQS:-64}" "$@"
