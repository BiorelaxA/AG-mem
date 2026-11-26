#!/usr/bin/env bash
#SBATCH --job-name=lmem-m-fill
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=lmem-m-fill-%j.out
#SBATCH --error=lmem-m-fill-%j.err
set -Eeuo pipefail
cd "${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}"
source "${AGMEM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate agmem
unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY

: "${DEEPSEEK_API_KEY:?请先在提交任务的终端设置 DEEPSEEK_API_KEY}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

embedding=model/qwen3_embedding_4b
if [[ ! -d "$embedding" ]]; then embedding=models/qwen3_embedding_4b; fi
embedding=${AGMEM_EMBEDDING_MODEL_PATH:-$embedding}
args=()
case "${1:-}" in
    --dry-run) args+=(--dry-run) ;;
    "") : "${SLURM_JOB_ID:?Submit with sbatch or run with --dry-run}"
        : "${DEEPSEEK_API_KEY:?Export DEEPSEEK_API_KEY before submitting}"
        # vLLM 0.19.1 parses visible device IDs as integers.
        assigned=${CUDA_VISIBLE_DEVICES:?Slurm must assign one embedding GPU}
        if [[ "$assigned" =~ ^[0-9]+$ ]]; then
            AGMEM_EMBEDDING_GPUS=$assigned
        elif [[ "$assigned" == GPU-* && "$assigned" != *,* ]]; then
            AGMEM_EMBEDDING_GPUS=$(nvidia-smi --id="$assigned" --query-gpu=index --format=csv,noheader,nounits)
            [[ "$AGMEM_EMBEDDING_GPUS" =~ ^[0-9]+$ ]] || { echo "Cannot map Slurm GPU $assigned to an index" >&2; exit 1; }
            actual=$(CUDA_VISIBLE_DEVICES="$AGMEM_EMBEDDING_GPUS" python -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)')
            if [[ "GPU-${actual#GPU-}" != "$assigned"* ]]; then
                echo "Numeric CUDA index does not select the GPU assigned by Slurm" >&2
                exit 1
            fi
        else
            echo "Expected one Slurm GPU, received $assigned" >&2
            exit 1
        fi
        export AGMEM_EMBEDDING_GPUS
        args+=(--job-dir "runs/deepseek_directory_build/slurm/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}")
        ;;
    *) echo "Usage: sbatch $0, or bash $0 --dry-run" >&2; exit 2 ;;
esac

exec python -u -m agmem.build_deepseek_directories_gpu --train-range \
    --config configs/build_longmemeval_m_deepseek_train_missing.yaml \
    --migration-manifest migration/longmemeval_m_continuation.json \
    --datasets split_longmemeval_m --splits train --workers 512 \
    --embedding-model-path "$embedding" --embedding-max-model-len 8192 \
    --embedding-max-num-seqs 32 --embedding-max-num-batched-tokens 16384 \
    --embedding-gpu-memory-utilization 0.60 "${args[@]}"
