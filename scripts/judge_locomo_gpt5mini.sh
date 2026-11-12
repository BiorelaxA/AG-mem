#!/usr/bin/env bash
# Run on a networked CPU/login host with OPENAI_API_KEY and OPENAI_BASE_URL set.
set -Eeuo pipefail
agmem_judge_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_judge_project"
exec python -u -m agmem.judge_gpt_responses \
    --datasets locomo --checkpoints best final "$@"
