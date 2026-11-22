#!/usr/bin/env bash
# Fresh LongMemEval-M directories: DeepSeek API Writer, two Embedding-4B GPUs.
set -Eeuo pipefail
agmem_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_project"
export AGMEM_EMBEDDING_GPUS="${AGMEM_EMBEDDING_GPUS:-${CUDA_VISIBLE_DEVICES:-4,5}}"
export CUDA_VISIBLE_DEVICES="$AGMEM_EMBEDDING_GPUS"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Set DEEPSEEK_API_KEY in the environment; neither commands nor logs print it.
# --dry-run requires no key, network, model loading, or CUDA access.
exec python -u -m agmem.build_deepseek_directories_gpu \
    --config configs/build_longmemeval_m_deepseek.yaml \
    --workers "${AGMEM_WRITER_WORKERS:-64}" \
    --embedding-max-model-len "${AGMEM_EMBEDDING_MAX_MODEL_LEN:-8192}" \
    --embedding-max-num-seqs "${AGMEM_EMBEDDING_MAX_NUM_SEQS:-64}" \
    --embedding-max-num-batched-tokens "${AGMEM_EMBEDDING_MAX_NUM_BATCHED_TOKENS:-32768}" \
    --embedding-gpu-memory-utilization "${AGMEM_EMBEDDING_GPU_MEMORY_UTILIZATION:-0.35}" \
    --datasets split_longmemeval_m --splits train validation test "$@"
