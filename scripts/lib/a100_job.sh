# Shared runtime for the two independent A100 jobs; parsed before long-running work.
agmem_a100_job() {
set -Eeuo pipefail
MODE=$1
: "${SLURM_JOB_ID:?Submit with sbatch}"
PROJECT_DIR=$PWD
source "${AGMEM_CONDA_SH:-/home/hongyshen/miniconda3/etc/profile.d/conda.sh}"
conda activate "${AGMEM_CONDA_ENV:-agmem}"
unset all_proxy
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
command -v flock >/dev/null
command -v setsid >/dev/null
command -v vllm >/dev/null
if [[ $MODE == directories ]]; then
    BASE_CONFIG=configs/directory_all.yaml
    JOB_ROOT=${AGMEM_JOB_ROOT:-$PROJECT_DIR/runs/directory_all}
    EXPECTED_GPUS=2
else
    BASE_CONFIG=configs/train_eval_locomo_s.yaml
    JOB_ROOT=${AGMEM_JOB_ROOT:-$PROJECT_DIR/runs/train_eval_locomo_s}
    EXPECTED_GPUS=4
fi
mkdir -p "$JOB_ROOT/jobs"
JOB_DIR=$(mktemp -d "$JOB_ROOT/jobs/${SLURM_JOB_ID}.XXXXXX")
JOB_DIR=$(cd "$JOB_DIR" && pwd)
exec > >(tee -a "$JOB_DIR/job.log") 2>&1
exec 8> "$JOB_ROOT/.job.lock"
flock -n 8 || { echo "Another $MODE job uses $JOB_ROOT" >&2; exit 1; }

# Old pilot scripts take this lock exclusively. The two new jobs share it and
# additionally coordinate individual sources in Python. Do not truncate it.
PREPARED_ROOT=${AGMEM_PREPARED_ROOT:-$PROJECT_DIR/runs/locomo_longmemeval/prepared}
mkdir -p "$PREPARED_ROOT"
PREPARED_ROOT=$(cd "$PREPARED_ROOT" && pwd)
exec 9>> "$(dirname "$PREPARED_ROOT")/.job.lock"
flock -s -n 9 || { echo "Legacy pilot still owns prepared data; let it exit before starting these jobs" >&2; exit 1; }
export AGMEM_PREPARED_ROOT="$PREPARED_ROOT"
IFS=',' read -r -a assigned_gpus <<< "${CUDA_VISIBLE_DEVICES:?Slurm must assign GPUs}"
if [[ ${#assigned_gpus[@]} -ne $EXPECTED_GPUS ]]; then
    echo "Expected $EXPECTED_GPUS GPUs; received $CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
if [[ $MODE == directories ]]; then
    SERVICE_GPUS="${assigned_gpus[0]},${assigned_gpus[1]}"
    SERVICE_TP=2
    READER_GPU=""
else
    READER_GPU="${assigned_gpus[0]},${assigned_gpus[1]}"
    SERVICE_GPUS="${assigned_gpus[2]},${assigned_gpus[3]}"
    SERVICE_TP=2
fi
server_pid=
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n $server_pid ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        for ((attempt=0; attempt<10; attempt++)); do
            kill -0 -- "-$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    echo "[$(date -Is)] mode=$MODE exit=$status logs=$JOB_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[$(date -Is)] mode=$MODE job=$SLURM_JOB_ID node=$(hostname) logs=$JOB_DIR"
echo "27B GPUs=$SERVICE_GPUS; TP=$SERVICE_TP; Reader GPUs=${READER_GPU:-not allocated}"
nvidia-smi
CONFIG_PATH="$JOB_DIR/config.yaml"
python - "$BASE_CONFIG" "$JOB_DIR" "$MODE" "$EXPECTED_GPUS" <<'PYCONFIG'
import os
from pathlib import Path
import socket
import sys
import yaml
import torch
import vllm
from agmem.core import load_config
from agmem.execution import make_job_config

base, job, mode, expected = sys.argv[1:]
job = Path(job)
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == int(expected), 'Unexpected visible GPU count'
config = load_config(base)
config['data']['input_root'] = str(Path(config['data']['input_root']).resolve())
config['data']['prepared_root'] = os.environ['AGMEM_PREPARED_ROOT']
root = Path(config['data']['prepared_root']).parent
config['jobs']['test_prepared_root'] = str(root / 'prepared_test')
config['jobs']['legacy_prepared_root'] = str(Path(config['jobs']['legacy_prepared_root']).resolve())
experiment = config['jobs'].get('experiment_name', 'locomo_s')
assert isinstance(experiment, str) and experiment not in ('', '.', '..') and Path(experiment).name == experiment, 'Invalid experiment name'
config['training']['output_dir'] = str(root / 'experiments' / experiment / 'train')
for key in (('task_path',) if mode == 'directories' else ('reader_path', 'task_path')):
    assert (Path(config['models'][key]) / 'config.json').is_file(), f'Missing model: {key}'
if mode == 'train-eval':
    from torch.utils.tensorboard import SummaryWriter
# Preserve the logical URL for existing Writer cache identity. Bind a separate
# deployment port so both independent jobs may run on the same host.
port = int(os.environ.get('AGMEM_SERVICE_PORT', '0'))
assert 0 <= port <= 65535
with socket.socket() as sock:
    sock.bind(('127.0.0.1', port))
    port = sock.getsockname()[1]
config['service']['request_base_url'] = f'http://127.0.0.1:{port}/v1'
(job / 'base_config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
max_seqs = make_job_config(job / 'base_config.yaml', job / 'config.yaml', os.environ)
(job / 'service.env').write_text(f"{port}\n{config['models']['task_path']}\n{max_seqs}\n")
print(f'PyTorch={torch.__version__}; vLLM={vllm.__version__}; sources={config["data"]["sources"]}')
print(f'Writer cache URL={config["service"]["base_url"]}; actual endpoint={config["service"]["request_base_url"]}')
PYCONFIG
mapfile -t service_args < "$JOB_DIR/service.env"
SERVICE_PORT=${service_args[0]}
TASK_MODEL=${service_args[1]}
SERVICE_MAX_SEQS=${service_args[2]}
if [[ $MODE == train-eval ]]; then
    echo "Checking Reader two-GPU generation/backward/checkpoint compatibility with tiny local weights"
    if ! env CUDA_VISIBLE_DEVICES="$READER_GPU" python -m unittest tests.test_model_parallel -v \
        > "$JOB_DIR/reader_parallel_check.log" 2>&1; then
        tail -n 80 "$JOB_DIR/reader_parallel_check.log"
        exit 1
    fi
fi
setsid env CUDA_VISIBLE_DEVICES="$SERVICE_GPUS" vllm serve "$TASK_MODEL" \
    --served-model-name agmem-27b --host 127.0.0.1 --port "$SERVICE_PORT" \
    --dtype bfloat16 --tensor-parallel-size "$SERVICE_TP" --max-model-len 65536 \
    --max-num-seqs "$SERVICE_MAX_SEQS" --gpu-memory-utilization 0.90 \
    --language-model-only > "$JOB_DIR/vllm.log" 2>&1 &
server_pid=$!
echo "$server_pid" > "$JOB_DIR/vllm.pid"
if ! python - "$SERVICE_PORT" "$server_pid" <<'PYREADY'
import os, sys, time, urllib.request
port, pid = map(int, sys.argv[1:])
deadline = time.monotonic() + 900
report = 0
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3) as response:
            if response.status == 200:
                print('27B service ready', flush=True)
                break
    except OSError:
        pass
    if time.monotonic() >= report:
        print('Waiting for 27B; see vllm.log', flush=True)
        report = time.monotonic() + 30
    time.sleep(2)
else:
    raise SystemExit('27B startup timeout')
PYREADY
then
    tail -n 60 "$JOB_DIR/vllm.log"
    exit 1
fi
# The orchestrator holds per-source locks and launches Reader train/evaluate in
# separate processes, releasing its GPU model memory between stages.
READER_PARALLEL_SIZE=1
if [[ $MODE == train-eval ]]; then
    READER_PARALLEL_SIZE=2
fi
env CUDA_VISIBLE_DEVICES="$READER_GPU" AGMEM_READER_MODEL_PARALLEL_SIZE="$READER_PARALLEL_SIZE" \
    python -u -m agmem.jobs "$MODE" \
    --config "$CONFIG_PATH" --job-dir "$JOB_DIR" > "$JOB_DIR/pipeline.log" 2>&1
exit 0
}
