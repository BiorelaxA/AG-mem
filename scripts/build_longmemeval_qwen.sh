#!/usr/bin/env bash
# Build the new LoCoMo/LongMemEval parent-child directories in a three-GPU allocation.
set -Eeuo pipefail
agmem_writer_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_writer_project"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
exec python -u -m agmem.build_local_directories_gpu \
    --config configs/build_parent_child_qwen.yaml \
    --workers "${AGMEM_WRITER_WORKERS:-48}" "$@"
