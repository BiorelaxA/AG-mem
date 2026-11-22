#!/usr/bin/env bash
# Current fixed315/35/150 data. GPU4 embeds; GPU5 writes. Reuse all S splits,
# then fill S test gaps and construct M train/validation/test with topic/event v2.
set -Eeuo pipefail
agmem_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_project"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
agmem_mode=all
agmem_dry=false
for agmem_arg in "$@"; do
    case "$agmem_arg" in
        --import-only|--s-test-only|--m-only)
            if [[ "$agmem_mode" != all && "$agmem_mode" != "$agmem_arg" ]]; then
                printf 'Choose only one phase selector.\n' >&2; exit 2
            fi
            agmem_mode=$agmem_arg ;;
        --dry-run) agmem_dry=true ;;
        --help|-h)
            cat <<'USAGE'
Usage: bash scripts/build_longmemeval_semantic_2gpu.sh [PHASE] [--dry-run]
Default: verify/create fixed315/35/150, preserve old directories for all S/M
splits, then build missing S test sessions followed by all M splits.
--import-only  CPU-only preparation/import; no GPU or model requests.
--s-test-only  Only build missing S test sessions after preparation/import.
--m-only       Only build M train/validation/test after preparation/import.
--dry-run      Check/print the plan; no new data or model requests.
CUDA_VISIBLE_DEVICES defaults to 4,5: first Embedding-4B, second Qwen3.5-27B.
AGMEM_WRITER_WORKERS=64; AGMEM_WRITER_MAX_NUM_SEQS=64 (one Writer GPU).
Repeated execution resumes. Existing complete/fallback directories are retained.
USAGE
            exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$agmem_arg" >&2; exit 2 ;;
    esac
done
agmem_config=configs/build_adjacent_qwen_longmemeval_31535150.yaml
agmem_data=data/processed/longmemeval_315_35_150_seed42_20261113
agmem_extra=()
if "$agmem_dry"; then agmem_extra+=(--dry-run); fi

python -u -m agmem.longmemeval_fixed_split "${agmem_extra[@]}"
python -u -m agmem.reuse_topic_event_directories --config "$agmem_config" \
    --reuse-root runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1 \
    --mapping "$agmem_data/row_mapping.jsonl" \
    --datasets split_longmemeval_s split_longmemeval_m --splits train validation test \
    "${agmem_extra[@]}"
if [[ "$agmem_mode" == --import-only ]]; then exit 0; fi

# The complete imported scope was checked above. The launchers only build their
# selected missing sessions. Each invocation releases its owned GPU processes.
agmem_build=(python -u -m agmem.build_adjacent_directories_gpu --topic-event --scope-completion
    --config "$agmem_config" --workers "${AGMEM_WRITER_WORKERS:-64}"
    --max-num-seqs "${AGMEM_WRITER_MAX_NUM_SEQS:-64}" "${agmem_extra[@]}")
if [[ "$agmem_mode" != --m-only ]]; then
    "${agmem_build[@]}" --datasets split_longmemeval_s --splits test
fi
if [[ "$agmem_mode" != --s-test-only ]]; then
    "${agmem_build[@]}" --datasets split_longmemeval_m --splits train validation test
fi
