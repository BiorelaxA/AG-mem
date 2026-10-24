#!/usr/bin/env bash
#SBATCH --job-name=agmem-dirs-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-pcie-40gb:2
#SBATCH --mem=120G
#SBATCH --time=96:00:00
#SBATCH --output=agmem-dirs-all-%j.out
#SBATCH --error=agmem-dirs-all-%j.err

# Build LongMemEval-M then the other six sources; LoCoMo/S use the train/eval job.
# Submit from the repository root. This job never loads/trains the 4B Reader.
set -Eeuo pipefail
PROJECT_DIR=${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?Submit with sbatch}}
cd "$PROJECT_DIR"
source scripts/lib/a100_job.sh
agmem_a100_job directories
