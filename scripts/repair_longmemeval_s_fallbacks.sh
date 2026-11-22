#!/usr/bin/env bash
# Continue the user-approved 500/500/30 repair, retaining the first 49 results.
set -Eeuo pipefail
agmem_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_project"
exec bash scripts/repair_longmemeval_s_fallbacks_len500.sh "$@"
