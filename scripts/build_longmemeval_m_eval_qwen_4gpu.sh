#!/usr/bin/env bash
# GPU 0 embeds; GPUs 1,2,3 each host a complete Writer replica.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${AGMEM_BUILD_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec python -u -m agmem.build_adjacent_directories_gpu \
  --m-eval --config configs/build_longmemeval_m_qwen_eval.yaml \
  --datasets split_longmemeval_m --splits validation test \
  --workers "${AGMEM_WRITER_WORKERS:-192}" --max-num-seqs "${AGMEM_WRITER_MAX_NUM_SEQS:-64}" \
  --max-num-batched-tokens 8192 --embedding-gpu-memory-utilization 0.35 "$@"
