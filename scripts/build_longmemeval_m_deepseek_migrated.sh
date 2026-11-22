#!/usr/bin/env bash
# Full M continuation on another host: tokenizer-only reference, two embedding replicas.
set -Eeuo pipefail
agmem_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_project"
export AGMEM_EMBEDDING_GPUS="${AGMEM_EMBEDDING_GPUS:-${CUDA_VISIBLE_DEVICES:-4,5}}"
export CUDA_VISIBLE_DEVICES="$AGMEM_EMBEDDING_GPUS"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Set DEEPSEEK_API_KEY externally. --dry-run and --import-only use no API or GPU.
# The package carries reference tokenizer files; install Embedding-4B separately.
exec python -u -m agmem.build_portable_deepseek_directories \
    --config configs/build_longmemeval_m_deepseek_no_repair.yaml \
    --migration-manifest "${AGMEM_MIGRATION_MANIFEST:-migration/longmemeval_m_continuation.json}" \
    --workers "${AGMEM_WRITER_WORKERS:-256}" \
    --embedding-max-model-len "${AGMEM_EMBEDDING_MAX_MODEL_LEN:-8192}" \
    --embedding-max-num-seqs "${AGMEM_EMBEDDING_MAX_NUM_SEQS:-64}" \
    --embedding-max-num-batched-tokens "${AGMEM_EMBEDDING_MAX_NUM_BATCHED_TOKENS:-32768}" \
    --embedding-gpu-memory-utilization "${AGMEM_EMBEDDING_GPU_MEMORY_UTILIZATION:-0.35}" \
    --datasets split_longmemeval_m --splits train validation test "$@"
