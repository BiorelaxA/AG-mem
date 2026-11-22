#!/usr/bin/env bash
# Run later on a networked CPU/login node. Reuse local evaluation's exact Reader
# evidence and Task messages; GPT answers/Judge results never overwrite Qwen's.
set -Eeuo pipefail
agmem_lme_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_lme_project"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"
agmem_lme_config=configs/eval_longmemeval_811_orm_checkpoints.yaml
agmem_lme_local=runs/evaluation/longmemeval_811_orm/results
# Stable path: restarting resumes completed requests and Judge results.
agmem_lme_remote=${AGMEM_EVAL_OUTPUT_ROOT:-runs/evaluation/longmemeval_811_orm/gpt}
agmem_lme_workers=${AGMEM_EVAL_WORKERS:-2}

# Fail before any API call if a local evaluation has not exported its bundle.
for agmem_lme_checkpoint in best final; do
    for agmem_lme_dataset in split_longmemeval_s split_longmemeval_m; do
        agmem_lme_bundle="$agmem_lme_local/$agmem_lme_checkpoint/$agmem_lme_dataset/offline/requests.json.gz"
        if [[ ! -f "$agmem_lme_bundle" ]]; then
            printf 'Missing exported Task requests: %s\nComplete local evaluation first.\n' "$agmem_lme_bundle" >&2
            exit 1
        fi
    done
done

printf 'GPT answer/Judge output: %s\n' "$agmem_lme_remote"
for agmem_lme_checkpoint in best final; do
    for agmem_lme_dataset in split_longmemeval_s split_longmemeval_m; do
        python -u agmem/remote_task_runner.py \
            --requests "$agmem_lme_local/$agmem_lme_checkpoint/$agmem_lme_dataset/offline/requests.json.gz" \
            --output "$agmem_lme_remote/responses/$agmem_lme_checkpoint/$agmem_lme_dataset" \
            --workers "$agmem_lme_workers" --timeout 120 --retries 6 \
            --empty-answer-max-tokens 16384
        python -u -m agmem.judge_gpt_responses \
            --config "$agmem_lme_config" --datasets "$agmem_lme_dataset" \
            --checkpoints "$agmem_lme_checkpoint" \
            --responses-root "$agmem_lme_remote/responses" \
            --output-root "$agmem_lme_remote/judge_results" \
            --workers "$agmem_lme_workers" --timeout 120 --retries 3 --network-retries 6
    done
done
printf 'GPT Judge results: %s/judge_results/summary.csv\n' "$agmem_lme_remote"
cat "$agmem_lme_remote/judge_results/summary.csv"
