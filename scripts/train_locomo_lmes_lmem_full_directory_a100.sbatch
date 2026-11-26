#!/usr/bin/env bash
#SBATCH --job-name=agmem-orm-lsm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:2
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --output=agmem-orm-lsm-%j.out
#SBATCH --error=agmem-orm-lsm-%j.err
set -euo pipefail
cd "${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}"
source "${AGMEM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate agmem
export AGMEM_TRAIN_GPUS="${CUDA_VISIBLE_DEVICES:?Slurm must allocate three GPUs}"
export AGMEM_TRAIN_CONFIG=configs/train_dapo_locomo_lmes_lmem_full_directory_orm.yaml
exec bash scripts/train_locomo_lmes_full_directory_2gpu.sh "$@"
