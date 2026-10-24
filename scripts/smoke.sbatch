#!/usr/bin/env bash
#SBATCH --job-name=agmem-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-pcie-40gb:3
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=agmem-smoke-%j.out
#SBATCH --error=agmem-smoke-%j.err

# Submit from the repository root:
#   sbatch scripts/smoke.sbatch
# If required by the cluster, add --partition=NAME and --account=NAME.
# The GRES type must match sinfo; override --gres on submission if needed.
set -Eeuo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Submit this script with sbatch; it requires a three-GPU allocation." >&2
    exit 1
fi

PROJECT_DIR=${AGMEM_PROJECT_DIR:-${SLURM_SUBMIT_DIR:?}}
cd "$PROJECT_DIR"
PROJECT_DIR=$PWD
CONDA_SH=${AGMEM_CONDA_SH:-/home/hongyshen/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${AGMEM_CONDA_ENV:-agmem}
source "$CONDA_SH"
conda activate "$CONDA_ENV"
unset all_proxy
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

RUN_DIR="$PROJECT_DIR/runs/slurm_smoke_${SLURM_JOB_ID}"
# Refuse to reuse an output directory, including on an automatic job requeue.
mkdir -p "$PROJECT_DIR/runs"
mkdir "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/job.log") 2>&1

server_pid=
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n $server_pid ]]; then
        # vLLM and its workers run in their own process group via setsid.
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for ((attempt=0; attempt<10; attempt++)); do
            if ! kill -0 -- "-$server_pid" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    echo "[$(date -Is)] Job finished with exit code $status; artifacts: $RUN_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[$(date -Is)] job=$SLURM_JOB_ID node=$(hostname) output=$RUN_DIR"
command -v setsid >/dev/null
command -v vllm >/dev/null
nvidia-smi

# Keep Slurm's assigned IDs/UUIDs rather than assuming physical GPUs 0,1,2.
IFS=',' read -r -a assigned_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm must set CUDA_VISIBLE_DEVICES}"
if [[ ${#assigned_gpus[@]} -ne 3 ]]; then
    echo "Expected exactly three allocated GPUs; CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
READER_GPU=${assigned_gpus[0]}
SERVICE_GPUS="${assigned_gpus[1]},${assigned_gpus[2]}"
echo "Reader CUDA_VISIBLE_DEVICES=$READER_GPU; 27B CUDA_VISIBLE_DEVICES=$SERVICE_GPUS"

# Fail before loading the 27B model if CUDA or required dependencies are missing.
python - <<'PY'
from pathlib import Path
import torch
import vllm
from torch.utils.tensorboard import SummaryWriter
from agmem.core import load_config

assert torch.cuda.is_available(), "CUDA-enabled PyTorch and a working GPU driver are required"
assert torch.cuda.device_count() == 3, "PyTorch must see all three allocated GPUs"
config = load_config("configs/smoke.yaml")
for key in ("reader_path", "task_path"):
    path = Path(config["models"][key])
    assert (path / "config.json").is_file(), f"Model config missing: {path}"
assert config["training"]["group_size"] == 8
print(f"PyTorch={torch.__version__}, vLLM={vllm.__version__}")
PY

# Select a free local port so another job's service cannot be reused accidentally.
SERVICE_PORT=${AGMEM_SERVICE_PORT:-$(python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)}
CONFIG_PATH="$RUN_DIR/config.yaml"
python - "$RUN_DIR" "$SERVICE_PORT" <<'PY'
from pathlib import Path
import sys
import yaml
from agmem.core import load_config

run_dir = Path(sys.argv[1])
port = int(sys.argv[2])
assert 1 <= port <= 65535
config = load_config("configs/smoke.yaml")
config["data"]["prepared_root"] = str(run_dir / "prepared")
config["training"]["output_dir"] = str(run_dir / "train")
config["training"]["tensorboard"] = True
config["service"]["base_url"] = f"http://127.0.0.1:{port}/v1"
(run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
PY

# Per-attempt scheduling only: retain the frozen Writer identity and checkpoint config.
SERVICE_MAX_SEQS=$(python -m agmem.execution "$CONFIG_PATH" "$RUN_DIR/execution_config.yaml")
CONFIG_PATH="$RUN_DIR/execution_config.yaml"
echo "Execution config: $CONFIG_PATH; vLLM max-num-seqs=$SERVICE_MAX_SEQS"
TASK_MODEL=$(python - "$CONFIG_PATH" <<'PY'
import sys
from agmem.core import load_config
print(load_config(sys.argv[1])["models"]["task_path"])
PY
)
echo "[$(date -Is)] Starting 27B service on port $SERVICE_PORT"
setsid env CUDA_VISIBLE_DEVICES="$SERVICE_GPUS" vllm serve "$TASK_MODEL" \
    --served-model-name agmem-27b \
    --host 127.0.0.1 \
    --port "$SERVICE_PORT" \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 65536 \
    --max-num-seqs "$SERVICE_MAX_SEQS" \
    --gpu-memory-utilization 0.90 \
    --language-model-only \
    > "$RUN_DIR/vllm.log" 2>&1 &
server_pid=$!
echo "$server_pid" > "$RUN_DIR/vllm.pid"

# Check both the HTTP endpoint and the service process; startup errors fail early.
if ! python - "$SERVICE_PORT" "$server_pid" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request

port, pid = map(int, sys.argv[1:])
deadline = time.monotonic() + 900
next_report = 0
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("vLLM exited before becoming ready; see vllm.log")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            if response.status == 200:
                print("27B service is ready", flush=True)
                break
    except (OSError, urllib.error.URLError):
        pass
    if time.monotonic() >= next_report:
        print("Waiting for 27B service; startup details are in vllm.log", flush=True)
        next_report = time.monotonic() + 30
    time.sleep(2)
else:
    raise SystemExit("27B service did not become ready within 900 seconds")
PY
then
    tail -n 80 "$RUN_DIR/vllm.log"
    exit 1
fi

echo "[$(date -Is)] Preparing two train queries and two validation queries"
python -u -m agmem --config "$CONFIG_PATH" prepare --limit-per-source 2 \
    > "$RUN_DIR/prepare.log" 2>&1

echo "[$(date -Is)] Building session directories with the current Writer prompt"
python -u -m agmem --config "$CONFIG_PATH" build-directories \
    > "$RUN_DIR/directories.log" 2>&1

echo "[$(date -Is)] Training one GRPO group (eight Reader trajectories)"
env CUDA_VISIBLE_DEVICES="$READER_GPU" python -u -m agmem --config "$CONFIG_PATH" \
    train --max-groups 1 > "$RUN_DIR/train.log" 2>&1

echo "[$(date -Is)] Smoke command completed successfully"
echo "Training log: $RUN_DIR/train.log"
echo "TensorBoard events: $RUN_DIR/train/tensorboard"
echo "View with: tensorboard --logdir '$RUN_DIR/train/tensorboard' --host 127.0.0.1 --port 6006"
