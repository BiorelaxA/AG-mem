#!/usr/bin/env bash
#SBATCH --job-name=agmem-train-ls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-pcie-40gb:4
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=agmem-train-ls-%j.out
#SBATCH --error=agmem-train-ls-%j.err

# Two GPUs split Reader decoder layers; two GPUs serve the frozen 27B model.
# Build missing LoCoMo/S train/validation directories, train, then prepare/test.
set -Eeuo pipefail
PROJECT_DIR=${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit with sbatch}}
cd "$PROJECT_DIR"
source scripts/lib/a100_job.sh
agmem_a100_job train-eval
