#!/usr/bin/env bash
# Run on a network-enabled login node before launching the offline GPU build.
# Optional first argument: destination; AGMEM_EMBEDDING_REVISION pins a commit.
set -Eeuo pipefail
agmem_embedding_project=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$agmem_embedding_project"
agmem_embedding_target=${1:-../model/qwen3-embedding-4b}
if (( $# > 1 )); then
    echo "Usage: bash scripts/download_directory_embedding.sh [local_model_directory]" >&2
    exit 2
fi
# These overrides apply only to this download, never to GPU build processes.
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python - "$agmem_embedding_target" <<'PY'
import json
import os
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

destination = Path(sys.argv[1]).expanduser().resolve()
repo = "Qwen/Qwen3-Embedding-4B"
metadata_path = destination / "agmem_download.json"
previous = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
if previous is not None and (not isinstance(previous, dict) or previous.get("repo_id") != repo
                             or not isinstance(previous.get("revision"), str) or not previous["revision"]):
    raise ValueError("Existing download metadata does not identify this embedding model; use a new destination")
revision = os.environ.get("AGMEM_EMBEDDING_REVISION")
if revision is None:
    revision = previous["revision"] if previous else "main"
if not revision.strip():
    raise ValueError("AGMEM_EMBEDDING_REVISION must identify a nonempty branch, tag or commit")
info = HfApi().model_info(repo, revision=revision)
snapshot_download(repo_id=repo, revision=info.sha, local_dir=str(destination),
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
metadata = {"repo_id": repo, "revision": info.sha}
# Fingerprints include file mtimes; a repeated download of the pinned revision
# must not invalidate an otherwise identical build or its embedding cache.
if previous != metadata:
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(metadata_path)
print(json.dumps({"event": "directory_embedding_downloaded", "path": str(destination), **metadata}))
PY
