#!/usr/bin/env bash
# On a networked CPU/login node: export, re-answer, then judge both checkpoints.
# Each invocation starts a fresh answer/judge run unless AGMEM_EVAL_OUTPUT_ROOT
# names an existing run to resume. Reader evidence and prompts stay fixed.
set -Eeuo pipefail
agmem_answer_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_answer_project"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL}"
agmem_answer_config=configs/eval_locomo_answer_tags.yaml
agmem_answer_results=runs/evaluation/format05_best_final/answer_tags/results
agmem_answer_workers=${AGMEM_EVAL_WORKERS:-2}
agmem_answer_run=${AGMEM_EVAL_OUTPUT_ROOT:-runs/evaluation/format05_best_final/answer_tags/retest_gpt_$(date +%Y%m%d_%H%M%S)_$$}
agmem_answer_responses="$agmem_answer_run/responses"
agmem_answer_judged="$agmem_answer_run/gpt_judge_results"

printf 'Evaluation output root: %s\n' "$agmem_answer_run"
printf 'Resume: AGMEM_EVAL_OUTPUT_ROOT=%q bash scripts/eval_locomo_gpt_answer_tags.sh\n' "$agmem_answer_run"

python -u -m agmem.evaluate_checkpoints \
    --config "$agmem_answer_config" --stage export-remote \
    --datasets locomo --checkpoints best final

for agmem_answer_checkpoint in best final; do
    python -u agmem/remote_task_runner.py \
        --requests "$agmem_answer_results/$agmem_answer_checkpoint/locomo/offline/requests.json.gz" \
        --output "$agmem_answer_responses/$agmem_answer_checkpoint/locomo" \
        --workers "$agmem_answer_workers" --timeout 120 --retries 6 \
        --empty-answer-max-tokens 16384
    python -u -m agmem.judge_gpt_responses \
        --config "$agmem_answer_config" --datasets locomo \
        --checkpoints "$agmem_answer_checkpoint" \
        --responses-root "$agmem_answer_responses" \
        --output-root "$agmem_answer_judged" \
        --workers "$agmem_answer_workers" --timeout 120 --retries 3 \
        --network-retries 6
done

printf 'Results: %s/summary.csv\n' "$agmem_answer_judged"
cat "$agmem_answer_judged/summary.csv"
