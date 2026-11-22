#!/usr/bin/env bash
set -Eeuo pipefail
cd "${AGMEM_PROJECT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}"
export AGMEM_EMBEDDING_GPUS=4 CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
args=()
manifest=${AGMEM_MIGRATION_MANIFEST:-migration/longmemeval_m_continuation.json}
if [[ -f "$manifest" ]]; then args+=(--migration-manifest "$manifest"); fi
exec python -u -m agmem.build_deepseek_directories_gpu --train-range \
    --config configs/build_longmemeval_m_deepseek_train_200_315.yaml \
    --workers 256 --datasets split_longmemeval_m --splits train "${args[@]}" "$@"
