#!/usr/bin/env bash
# Keep existing directories; only missing sessions use topic/event v2.
set -Eeuo pipefail
agmem_hierarchy_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_hierarchy_project"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
exec python -u -m agmem.build_adjacent_directories_gpu \
    --topic-event --preserve-existing \
    --config configs/build_adjacent_qwen_longmemeval_811_incremental.yaml \
    --reuse-root runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1 \
    --reuse-mapping data/processed/longmemeval_811_seed42_20261111/row_mapping.jsonl \
    --datasets split_longmemeval_s --splits train validation test \
    --workers "${AGMEM_WRITER_WORKERS:-128}" \
    --max-num-seqs "${AGMEM_WRITER_MAX_NUM_SEQS:-64}" "$@"
