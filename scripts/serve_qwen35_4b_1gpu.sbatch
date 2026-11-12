#!/usr/bin/env bash
#SBATCH --job-name=agmem-4b-1gpu
#SBATCH --partition=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=agmem-4b-1gpu-%j.out
#SBATCH --error=agmem-4b-1gpu-%j.err

# Submit from the project root. The single allocated GPU hosts Qwen3.5-4B.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
export AGMEM_PROJECT_DIR=${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?}}
export AGMEM_EXPECTED_GPUS=1
cd "$AGMEM_PROJECT_DIR"
exec bash scripts/serve_qwen35_4b.sh
