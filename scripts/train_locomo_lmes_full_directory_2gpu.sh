#!/usr/bin/env bash
# Full parent/child visibility, up to two leaf IDs per READ, trajectory-only ORM.
set -Eeuo pipefail
agmem_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_project"
export AGMEM_TRAIN_CONFIG="${AGMEM_TRAIN_CONFIG:-configs/train_dapo_locomo_lmes_full_directory_orm.yaml}"
exec bash scripts/train_locomo_lmes_f1_2gpu.sh "$@"
