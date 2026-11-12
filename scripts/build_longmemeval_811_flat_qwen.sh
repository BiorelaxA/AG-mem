#!/usr/bin/env bash
# Build legacy flat test directories, release the three Writers, then evaluate
# old ORM checkpoints 20/32 on S/M with two Reader GPUs and one Task/Judge GPU.
set -Eeuo pipefail
agmem_flat_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_flat_project"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
agmem_flat_mode=all
agmem_flat_dry_run=false
agmem_flat_args=()
for agmem_flat_arg in "$@"; do
    case "$agmem_flat_arg" in
        --build-only|--eval-only)
            if [[ "$agmem_flat_mode" != all && "$agmem_flat_mode" != "$agmem_flat_arg" ]]; then
                printf 'Use only one of --build-only and --eval-only.\n' >&2
                exit 2
            fi
            agmem_flat_mode=$agmem_flat_arg ;;
        --dry-run) agmem_flat_dry_run=true ;;
        --help|-h)
            cat <<'USAGE'
Usage: bash scripts/build_longmemeval_811_flat_qwen.sh [MODE] [arguments]
  (default)     Resume/build S/M flat test directories, then evaluate 20/32.
  --build-only  Build directories without starting evaluation.
  --eval-only   Evaluate already completed directories without starting Writers.
  --dry-run     Check the build recipe and print evaluation command; no GPUs used.
Other arguments go to the directory builder, or evaluate_gpu with --eval-only.
The default evaluation uses three allocated GPUs and makes no external API calls.
An already-running older invocation will not automatically gain this new stage;
after it finishes, use --eval-only in the same GPU allocation.
USAGE
            exit 0 ;;
        *) agmem_flat_args+=("$agmem_flat_arg") ;;
    esac
done

if [[ "$agmem_flat_mode" != --eval-only ]]; then
    agmem_flat_build=(python -u -m agmem.build_local_directories_gpu
        --legacy-flat --config configs/build_longmemeval_811_flat_qwen.yaml
        --datasets split_longmemeval_s split_longmemeval_m --splits test
        --workers "${AGMEM_WRITER_WORKERS:-192}"
        --max-num-seqs "${AGMEM_WRITER_MAX_NUM_SEQS:-64}"
        "${agmem_flat_args[@]}")
    if "$agmem_flat_dry_run"; then agmem_flat_build+=(--dry-run); fi
    # Do not exec: a successful builder returns only after its Writers are closed.
    # set -e prevents evaluation after interrupted/failed construction or cleanup.
    "${agmem_flat_build[@]}"
fi

if [[ "$agmem_flat_mode" != --build-only ]]; then
    agmem_flat_eval=(python -u -m agmem.evaluate_gpu
        --config configs/eval_longmemeval_811_orm_checkpoints.yaml
        --stage all --task-agents local --no-parallel-directories)
    if [[ "$agmem_flat_mode" == --eval-only ]]; then
        agmem_flat_eval+=("${agmem_flat_args[@]}")
    fi
    if "$agmem_flat_dry_run"; then
        printf 'Evaluation after Writer cleanup (not executed):'
        printf ' %q' "${agmem_flat_eval[@]}"
        printf '\n'
    else
        printf 'Starting checkpoint20/32 evaluation on LongMemEval-S/M.\n'
        exec "${agmem_flat_eval[@]}"
    fi
fi
