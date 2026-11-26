#!/usr/bin/env bash
#SBATCH --job-name=agmem-orm-lsm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:3
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --output=agmem-orm-lsm-%j.out
#SBATCH --error=agmem-orm-lsm-%j.err

set -euo pipefail
source "${AGMEM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate agmem

model_dir=${AGMEM_MODEL_PATH:-/home/share/models/Qwen3.5-27B}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

port=$(python - <<'PY'
import socket
import torch

count = torch.cuda.device_count()
if count != 3:
    raise SystemExit(f"Expected 3 Slurm-allocated CUDA devices, found {count}")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)
echo "node=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-Slurm} model=$model_dir"
echo "vLLM endpoint on compute node: http://127.0.0.1:$port/v1"

# Split decoder layers across the three GPUs and expose a single API endpoint.
exec python -u -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" --served-model-name agmem-27b \
    --host 127.0.0.1 --port "$port" \
    --pipeline-parallel-size 3 --tensor-parallel-size 1 --distributed-executor-backend mp \
    --dtype bfloat16 --language-model-only --enforce-eager \
    --max-model-len 65536 --enable-chunked-prefill \
    --max-num-batched-tokens 4096 --max-num-seqs 2 --gpu-memory-utilization 0.90
