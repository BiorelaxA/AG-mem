#!/usr/bin/env bash
#SBATCH --job-name=agmem-lme-dirs
#SBATCH --partition=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100-sxm4-80gb:3
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=agmem-lme-dirs-%j.out
#SBATCH --error=agmem-lme-dirs-%j.err

# Three independent Qwen3.5-27B replicas; intended for 80-GB GPUs.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
export AGMEM_PROJECT_DIR=${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?}}
cd "$AGMEM_PROJECT_DIR"
source "${AGMEM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${AGMEM_CONDA_ENV:-agmem}"
exec bash scripts/build_longmemeval_qwen.sh "$@"
