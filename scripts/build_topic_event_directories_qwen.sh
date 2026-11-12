#!/usr/bin/env bash
# Independent topic/event boundary revision; GPU0 embeds, GPU1/2 write.
set -Eeuo pipefail
agmem_topic_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_topic_project"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
exec python -u -m agmem.build_adjacent_directories_gpu \
    --topic-event --config configs/build_adjacent_qwen_topic_event_v2.yaml \
    --workers "${AGMEM_WRITER_WORKERS:-128}" \
    --max-num-seqs "${AGMEM_WRITER_MAX_NUM_SEQS:-64}" "$@"
