#!/usr/bin/env bash
#SBATCH --job-name=agmem-gpt-judge
#SBATCH --partition=gpu02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=agmem-gpt-judge-%j.out
#SBATCH --error=agmem-gpt-judge-%j.err

# Submit from the repository root when ready. This script only scores the
# saved GPT answers; the frozen training Judge prompt is loaded by the suite.
set -Eeuo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
export AGMEM_PROJECT_DIR=${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?}}
cd "$AGMEM_PROJECT_DIR"
source "${AGMEM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${AGMEM_CONDA_ENV:-agmem}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

exec python -u -m agmem.evaluate_gpu \
    --config configs/eval_locomo_longmemeval_checkpoints.yaml \
    --stage judge-remote \
    --datasets locomo \
    --checkpoints best final
