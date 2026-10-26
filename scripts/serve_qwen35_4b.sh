#!/usr/bin/env bash
# Run inside a Slurm allocation. Submit serve_qwen35_4b.sbatch from the repo root.
# Touch runs/serve_qwen35_4b_JOBID/STOP_SERVING to release model memory while
# keeping the allocation alive for a subsequent training step.
agmem_serve_qwen35_4b() {
    set -Eeuo pipefail
    if [[ -z ${SLURM_JOB_ID:-} ]]; then
        echo "Run with sbatch scripts/serve_qwen35_4b.sbatch or inside an salloc allocation." >&2
        return 1
    fi

    local project_dir model_dir run_dir conda_sh service_port server_pid=""
    local supervisor_pid=$BASHPID
    project_dir=${AGMEM_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
    cd "$project_dir"
    project_dir=$PWD
    model_dir=${AGMEM_MODEL_PATH:-$project_dir/../model/qwen3.5-4b}
    run_dir="$project_dir/runs/serve_qwen35_4b_${SLURM_JOB_ID}"
    mkdir -p "$project_dir/runs"
    # Reusing an old STOP_SERVING marker after a requeue would stop a new server.
    mkdir "$run_dir"
    exec > >(tee -a "$run_dir/job.log") 2>&1

    stop_server() {
        if [[ -n $server_pid ]]; then
            kill -TERM -- "-$server_pid" 2>/dev/null || true
            for ((attempt=0; attempt<20; attempt++)); do
                kill -0 -- "-$server_pid" 2>/dev/null || break
                sleep 1
            done
            kill -KILL -- "-$server_pid" 2>/dev/null || true
            wait "$server_pid" 2>/dev/null || true
            server_pid=""
        fi
    }
    cleanup() {
        local status=$?
        trap - EXIT INT TERM ERR
        stop_server
        rm -f "$run_dir/READY" "$run_dir/TRAINING_READY" "$run_dir/ALLOCATION_READY"
        echo "[$(date -Is)] Job ending: exit=$status"
        exit "$status"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    hold_allocation() {
        touch "$run_dir/ALLOCATION_READY"
        echo "[$(date -Is)] Allocation retained until scancel or time limit."
        echo "Enter it with: srun --jobid=$SLURM_JOB_ID --overlap --nodes=1 --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-16} --gres=gpu:4 --pty bash"
        while true; do
            sleep 30 &
            wait $!
        done
    }
    service_failure() {
        local status=$1 line=$2
        # A failing command substitution must return to the supervisor, not
        # start an invisible holding loop inside its captured stdout.
        [[ $BASHPID == "$supervisor_pid" ]] || return "$status"
        trap - ERR
        echo "[$(date -Is)] SERVICE_FAILED: status=$status line=$line; see job.log and vllm.log."
        printf 'status=%s line=%s\n' "$status" "$line" > "$run_dir/SERVICE_FAILED"
        stop_server
        rm -f "$run_dir/READY"
        hold_allocation
    }
    trap 'service_failure "$?" "$LINENO"' ERR

    echo "[$(date -Is)] job=$SLURM_JOB_ID node=$(hostname) model=$model_dir"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; logs=$run_dir"
    hostname > "$run_dir/node.txt"
    conda_sh=${AGMEM_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}
    echo "Activating conda: $conda_sh; environment=${AGMEM_CONDA_ENV:-agmem}"
    if [[ ! -r $conda_sh ]]; then
        echo "ERROR: cannot read conda initialization script: $conda_sh"
        service_failure 1 "$LINENO"
    fi
    source "$conda_sh"
    conda activate "${AGMEM_CONDA_ENV:-agmem}"
    export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
    # All model files are local; compute nodes need no Hugging Face connection.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    if ! command -v setsid >/dev/null; then
        echo "ERROR: setsid is missing; cannot manage vLLM worker processes."
        service_failure 127 "$LINENO"
    fi
    if ! command -v vllm >/dev/null; then
        echo "ERROR: vllm is not installed or not on PATH in conda environment ${CONDA_PREFIX:-unknown}."
        echo "Install requirements-serving.txt with a compatible CUDA wheel, then retry deployment inside this allocation."
        service_failure 127 "$LINENO"
    fi
    nvidia-smi

    # Check the actual allocation and local shards before starting workers.
    service_port=$(python - "$model_dir" "${AGMEM_SERVICE_PORT:-0}" <<'PY'
import json
from pathlib import Path
import socket
import sys
import torch

model = Path(sys.argv[1]).resolve()
assert torch.cuda.is_available(), "CUDA is unavailable; check GPU node driver and PyTorch wheel"
assert torch.cuda.device_count() == 4, f"Expected 4 allocated GPUs, got {torch.cuda.device_count()}"
for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"):
    assert (model / name).is_file(), f"Missing model file: {model / name}"
index = json.loads((model / "model.safetensors.index.json").read_text())
for name in set(index["weight_map"].values()):
    assert (model / name).is_file(), f"Missing model shard: {name}"
port = int(sys.argv[2])
assert 0 <= port <= 65535, "Invalid AGMEM_SERVICE_PORT"
with socket.socket() as sock:
    sock.bind(("127.0.0.1", port))
    print(sock.getsockname()[1])
print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}, GPUs=4", file=sys.stderr)
PY
    )
    echo "$service_port" > "$run_dir/port.txt"
    echo "http://127.0.0.1:$service_port/v1" > "$run_dir/base_url.txt"
    echo "Starting TP=4 service: http://127.0.0.1:$service_port/v1 (compute node only)"
    setsid vllm serve "$model_dir" \
        --served-model-name agmem-4b \
        --host 127.0.0.1 --port "$service_port" \
        --dtype bfloat16 --tensor-parallel-size 4 \
        --distributed-executor-backend mp \
        --max-model-len "${AGMEM_MAX_MODEL_LEN:-32768}" \
        --max-num-seqs "${AGMEM_MAX_NUM_SEQS:-8}" \
        --gpu-memory-utilization "${AGMEM_GPU_MEMORY_UTILIZATION:-0.80}" \
        --language-model-only > "$run_dir/vllm.log" 2>&1 &
    server_pid=$!
    echo "$server_pid" > "$run_dir/vllm.pid"

    local started=$SECONDS
    while [[ ! -f "$run_dir/STOP_SERVING" ]]; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            local status=0
            wait "$server_pid" || status=$?
            echo "vLLM exited unexpectedly (status=$status). Last log lines:"
            tail -n 60 "$run_dir/vllm.log"
            service_failure 1 "$LINENO"
        fi
        if [[ ! -f "$run_dir/READY" ]]; then
            if python - "$service_port" <<'PY'
import sys
import urllib.request
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except OSError:
    raise SystemExit(1)
PY
            then
                touch "$run_dir/READY"
                echo "[$(date -Is)] Service ready; model name: agmem-4b"
                echo "To switch to training: touch $run_dir/STOP_SERVING"
            elif (( SECONDS - started >= 1800 )); then
                echo "vLLM startup timed out after 1800 seconds."
                tail -n 60 "$run_dir/vllm.log"
                service_failure 1 "$LINENO"
            fi
        fi
        sleep 5 &
        wait $!
    done

    echo "[$(date -Is)] Stopping vLLM and its worker process group for training."
    stop_server
    rm -f "$run_dir/READY"
    touch "$run_dir/TRAINING_READY"
    echo "[$(date -Is)] TRAINING_READY: service stopped."
    hold_allocation
}

# Parse the entire function before launching long-running work.
agmem_serve_qwen35_4b "$@"
