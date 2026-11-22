#!/usr/bin/env bash
# Run on a networked CPU/login node. S completes before M. Reruns resume.
set -Eeuo pipefail
agmem_writer_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_writer_project"
exec python -u -m agmem.build_remote_directories \
    --config configs/build_longmemeval_gpt5mini.yaml \
    --workers "${AGMEM_WRITER_WORKERS:-8}" "$@"
