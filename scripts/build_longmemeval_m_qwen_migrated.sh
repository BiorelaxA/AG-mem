#!/usr/bin/env bash
# Full M continuation after transferring the data package to another server.
set -Eeuo pipefail
agmem_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_project"
export AGMEM_BUILD_GPUS="${AGMEM_BUILD_GPUS:-${CUDA_VISIBLE_DEVICES:-4,5}}"
export CUDA_VISIBLE_DEVICES="$AGMEM_BUILD_GPUS"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec python -u -m agmem.build_portable_qwen_directories \
    --config configs/build_longmemeval_m_qwen_continue.yaml \
    --migration-manifest "${AGMEM_MIGRATION_MANIFEST:-migration/longmemeval_m_continuation.json}" \
    --workers "${AGMEM_WRITER_WORKERS:-256}" \
    --max-num-seqs "${AGMEM_WRITER_MAX_NUM_SEQS:-256}" \
    --max-num-batched-tokens "${AGMEM_WRITER_MAX_NUM_BATCHED_TOKENS:-8192}" \
    --gpu-memory-utilization "${AGMEM_WRITER_GPU_MEMORY_UTILIZATION:-0.90}" \
    --embedding-max-model-len "${AGMEM_EMBEDDING_MAX_MODEL_LEN:-8192}" \
    --embedding-max-num-seqs "${AGMEM_EMBEDDING_MAX_NUM_SEQS:-64}" \
    --embedding-max-num-batched-tokens "${AGMEM_EMBEDDING_MAX_NUM_BATCHED_TOKENS:-32768}" \
    --embedding-gpu-memory-utilization "${AGMEM_EMBEDDING_GPU_MEMORY_UTILIZATION:-0.35}" \
    --datasets split_longmemeval_m --splits train validation test "$@"
