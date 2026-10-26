#!/usr/bin/env bash
#SBATCH --job-name=agmem-4b-serve
#SBATCH --partition=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --no-requeue
#SBATCH --output=agmem-4b-serve-%j.out
#SBATCH --error=agmem-4b-serve-%j.err

# Submit from the project root; do not run this on a login node with bash.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
export AGMEM_PROJECT_DIR=${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?}}
cd "$AGMEM_PROJECT_DIR"
exec bash scripts/serve_qwen35_4b.sh
