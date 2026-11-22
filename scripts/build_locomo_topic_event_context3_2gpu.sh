#!/usr/bin/env bash
# GPU4 embeds; GPU5 writes child entries with three preceding raw segments.
set -Eeuo pipefail
agmem_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_project"
export CUDA_VISIBLE_DEVICES="${AGMEM_DIRECTORY_GPUS:-4,5}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'USAGE'
Usage: bash scripts/build_locomo_topic_event_context3_2gpu.sh [OPTIONS]
Rebuild LoCoMo train/validation/test with thresholds 0.72/0.50 and the topic/event
boundary prompt. Each child summary receives its complete original segment plus
up to three complete preceding segments from the same session, not their summaries.
Default GPU4: Qwen3-Embedding-4B; GPU5: Qwen3.5-27B Writer.
AGMEM_DIRECTORY_GPUS=4,5; AGMEM_WRITER_WORKERS=64; AGMEM_WRITER_MAX_NUM_SEQS=64.
--dry-run             Check the plan without model requests or GPU deployment.
--output-root PATH    Put the new build/cache on another filesystem.
--job-dir PATH        Put this invocation's logs on another filesystem (fresh path).
--limit-sessions N    Build at most N unfinished sessions for a smoke run.
Other GPU-launcher options are forwarded. Existing jobs are never stopped.
No old directories or segmentation decisions are imported. Repeating this command
resumes only this new build; it does not rebuild completed sessions again.
USAGE
    exit 0
fi
exec python -u -m agmem.build_adjacent_directories_gpu \
    --segment-context --config configs/build_locomo_topic_event_context3.yaml \
    --workers "${AGMEM_WRITER_WORKERS:-64}" \
    --max-num-seqs "${AGMEM_WRITER_MAX_NUM_SEQS:-64}" \
    --datasets locomo --splits train validation test "$@"
