#!/usr/bin/env bash
#SBATCH --job-name=lmem-m-test-qwen
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_rtx_a6000:3
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=lmem-m-test-qwen-%j.out
#SBATCH --error=lmem-m-test-qwen-%j.err
set -Eeuo pipefail
cd "${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}"
source "${AGMEM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate agmem
unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
args=(--m-test-missing --config configs/build_longmemeval_m_qwen_test_missing.yaml
      --datasets split_longmemeval_m --splits test --writer-tp 2
      --workers 64 --max-num-seqs 64 --max-num-batched-tokens 8192
      --gpu-memory-utilization 0.90 --embedding-gpu-memory-utilization 0.50
      --embedding-max-num-seqs 32 --embedding-max-num-batched-tokens 16384)
case "${1:-}" in
    --dry-run) export CUDA_VISIBLE_DEVICES=""; args+=(--dry-run) ;;
    "") : "${SLURM_JOB_ID:?Submit with sbatch or run with --dry-run}"
        : "${CUDA_VISIBLE_DEVICES:?Slurm must assign three GPUs}"
        args+=(--job-dir "runs/qwen_m_test_build/slurm/${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}") ;;
    *) echo "Usage: sbatch $0, or bash $0 --dry-run" >&2; exit 2 ;;
esac
exec python -u -m agmem.build_adjacent_directories_gpu "${args[@]}"
