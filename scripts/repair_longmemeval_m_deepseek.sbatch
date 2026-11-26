#!/bin/bash
#SBATCH --job-name=lmem-m-repair
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=lmem-m-repair-%j.out
#SBATCH --error=lmem-m-repair-%j.err
set -euo pipefail
cd "${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}}"
source "${AGMEM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate agmem


: "${DEEPSEEK_API_KEY:?请先在提交任务的终端设置 DEEPSEEK_API_KEY}"

unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
export TOKENIZERS_PARALLELISM=false
exec python -u -m agmem.repair_longmemeval_m_deepseek --workers "${AGMEM_REPAIR_WORKERS:-512}" "$@"
