#!/usr/bin/env bash
#SBATCH --job-name=agmem-pilot-a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100-pcie-40gb:3
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --output=agmem-pilot-a6000-%j.out
#SBATCH --error=agmem-pilot-a6000-%j.err

# Submit from the repository root:
#   sbatch scripts/train_pilot_a6000.sbatch
# If required by the cluster, add --partition=NAME and --account=NAME.
# The GRES type must match sinfo; override --gres on submission if needed.
# Parse the complete runtime body before starting any long-running command.
# Keep the final exit inside the function so Bash never resumes reading a source
# file that may have been edited while this invocation was running.
agmem_main() {
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

# Persistent artifacts survive job timeouts; logs are separate for every attempt.
RUN_DIR=${AGMEM_RUN_DIR:-$PROJECT_DIR/runs/locomo_longmemeval}
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
command -v flock >/dev/null
exec 9> "$RUN_DIR/.job.lock"
if ! flock -n 9; then
    echo "Another job is using $RUN_DIR; choose a different AGMEM_RUN_DIR." >&2
    exit 1
fi
mkdir -p "$RUN_DIR/jobs"
JOB_DIR=$(mktemp -d "$RUN_DIR/jobs/${SLURM_JOB_ID}.XXXXXX")
exec > >(tee -a "$JOB_DIR/job.log") 2>&1

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

echo "[$(date -Is)] job=$SLURM_JOB_ID node=$(hostname) output=$RUN_DIR logs=$JOB_DIR"
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

# Freeze the resolved configuration on the first submission. In particular, keep
# the same service URL: it is part of the Writer cache/directory version.
CONFIG_PATH="$RUN_DIR/config.yaml"
python - "$RUN_DIR" <<'PYCONFIG'
from pathlib import Path
import os
import sys
import yaml
from agmem.core import load_config

run_dir = Path(sys.argv[1])
path = run_dir / "config.yaml"
requested_port = os.environ.get("AGMEM_SERVICE_PORT")
if path.exists():
    config = load_config(path)
    if requested_port:
        port = int(requested_port)
        assert 1 <= port <= 65535
        requested_url = f"http://127.0.0.1:{port}/v1"
        if config["service"]["base_url"] != requested_url:
            if any(Path(config[section][key]).exists() for section, key in
                   (("data", "prepared_root"), ("training", "output_dir"))):
                raise SystemExit("Service port differs from saved config and artifacts exist; free the saved port or select a new AGMEM_RUN_DIR")
            config["service"]["base_url"] = requested_url
            temporary = path.with_suffix(".yaml.tmp")
            temporary.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            temporary.replace(path)
            print(f"Updated service port to {port}; no prepared/training artifacts exist")
    print(f"Reusing frozen configuration: {path}")
else:
    port = int(requested_port or 50543)
    assert 1 <= port <= 65535
    config = load_config("configs/pilot.yaml")
    config["data"]["input_root"] = str(Path(config["data"]["input_root"]).resolve())
    config["data"]["prepared_root"] = str(run_dir / "prepared")
    config["training"]["output_dir"] = str(run_dir / "train")
    config["service"]["base_url"] = f"http://127.0.0.1:{port}/v1"
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    temporary.replace(path)
    print(f"Saved full-training configuration: {path}")
assert config["data"]["prepared_root"] == str(run_dir / "prepared"), "Prepared root differs from run directory"
assert config["training"]["output_dir"] == str(run_dir / "train"), "Training output differs from run directory"
assert config["data"].get("layout") == "by_source", "Pilot requires independent dataset stores; choose a fresh AGMEM_RUN_DIR"
assert set(config["data"]["sources"]).issubset({"locomo", "split_longmemeval_s", "split_longmemeval_m"}), "Pilot sources must be LoCoMo/LongMemEval"
PYCONFIG

# Stage selection is independent of the frozen data registry and prepared cache.
STAGE=${AGMEM_STAGE:-all}
case "$STAGE" in
    all|build|train) ;;
    *) echo "AGMEM_STAGE must be all, build, or train" >&2; exit 1 ;;
esac
WORKFLOW_PATH=${AGMEM_WORKFLOW_CONFIG:-configs/pilot_workflow.yaml}
SERVICE_MAX_SEQS=$(python -m agmem.execution "$CONFIG_PATH" "$JOB_DIR/config.yaml")
CONFIG_PATH="$JOB_DIR/config.yaml"
python -m agmem.workflow "$CONFIG_PATH" "$WORKFLOW_PATH" "$JOB_DIR"
BUILD_CONFIG="$JOB_DIR/build_config.yaml"
TRAIN_CONFIG="$JOB_DIR/train_config.yaml"
TRAIN_DIR=$(python - "$TRAIN_CONFIG" <<'PYTRAINPATH'
import sys
from agmem.core import load_config
print(load_config(sys.argv[1])["training"]["output_dir"])
PYTRAINPATH
)
TRAIN_COMPLETE="$TRAIN_DIR/train.complete"
echo "Stage=$STAGE; vLLM max-num-seqs=$SERVICE_MAX_SEQS"
if [[ $STAGE == train && -f $TRAIN_COMPLETE ]]; then
    echo "Selected experiment already completed: $TRAIN_COMPLETE"
    exit 0
fi
if [[ $STAGE == train ]]; then
    # Verify selected sources before allocating model memory. No Writer calls.
    python - "$TRAIN_CONFIG" <<'PYREADY'
import sys
from agmem.core import load_config
from agmem.pipeline import open_prepared, validate_directories
config = load_config(sys.argv[1])
store = open_prepared(config)
try:
    validate_directories(config, store, ["train", "validation"])
finally:
    store.close()
PYREADY
fi

# Fail before loading the 27B model if CUDA or required dependencies are missing.
PREFLIGHT_CONFIG="$TRAIN_CONFIG"
if [[ $STAGE == build ]]; then
    PREFLIGHT_CONFIG="$BUILD_CONFIG"
fi
python - "$PREFLIGHT_CONFIG" <<'PYPREFLIGHT'
from pathlib import Path
import sys
import torch
import vllm
from torch.utils.tensorboard import SummaryWriter
from agmem.core import load_config

assert torch.cuda.is_available(), "CUDA-enabled PyTorch and a working GPU driver are required"
assert torch.cuda.device_count() == 3, "PyTorch must see all three allocated GPUs"
config = load_config(sys.argv[1])
for key in ("reader_path", "task_path"):
    path = Path(config["models"][key])
    assert (path / "config.json").is_file(), f"Model config missing: {path}"
assert config["training"]["group_size"] == 8
assert config["training"].get("tensorboard", True)
assert config["service"]["model"] == "agmem-27b"
assert config["task_agent"]["max_context_tokens"] == 65536
print(f"PyTorch={torch.__version__}, vLLM={vllm.__version__}")
print(f"Sources={config['data']['sources']}; splits={config['data']['splits']}")
print(f"Epochs={config['training']['epochs']}; no sample/group limit")
PYPREFLIGHT

# Before any artifacts exist, a busy default port can be replaced safely.
# Once data/directory/training outputs exist, preserve the URL used in provenance.
SERVICE_PORT=$(python - "$RUN_DIR/config.yaml" <<'PYPORT'
import errno
import os
from pathlib import Path
import socket
import sys
from urllib.parse import urlsplit
import yaml
from agmem.core import load_config

path = Path(sys.argv[1])
config = load_config(path)
url = urlsplit(config["service"]["base_url"])
assert url.hostname == "127.0.0.1" and url.port is not None
port = url.port
try:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
except OSError as exc:
    if exc.errno != errno.EADDRINUSE:
        raise
    artifacts_exist = any(Path(config[section][key]).exists() for section, key in
                          (("data", "prepared_root"), ("training", "output_dir")))
    if artifacts_exist or os.environ.get("AGMEM_SERVICE_PORT"):
        raise SystemExit(f"Service port {port} is occupied. Free that port or use a new AGMEM_RUN_DIR with an available AGMEM_SERVICE_PORT. Existing cache/checkpoint provenance will not be changed.") from None
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config["service"]["base_url"] = f"http://127.0.0.1:{port}/v1"
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    temporary.replace(path)
    print(f"Port {url.port} is occupied; selected port {port} and saved it to {path} (no prepared/training artifacts exist)", file=sys.stderr)
print(port)
PYPORT
)
# Port selection may update the frozen URL before any artifacts exist.
python - "$RUN_DIR/config.yaml" "$CONFIG_PATH" "$BUILD_CONFIG" "$TRAIN_CONFIG" <<'PYSYNCURL'
from pathlib import Path
import sys
import yaml
from agmem.core import load_config
url = load_config(sys.argv[1])["service"]["base_url"]
for name in sys.argv[2:]:
    path = Path(name)
    value = load_config(path)
    if value["service"]["base_url"] != url:
        value["service"]["base_url"] = url
        path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False))
PYSYNCURL
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
    > "$JOB_DIR/vllm.log" 2>&1 &
server_pid=$!
echo "$server_pid" > "$JOB_DIR/vllm.pid"

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
    tail -n 80 "$JOB_DIR/vllm.log"
    exit 1
fi

if [[ $STAGE != train ]]; then
    echo "[$(date -Is)] Preparing selected directory sources"
    python -u -m agmem --config "$BUILD_CONFIG" prepare \
        > "$JOB_DIR/prepare.log" 2>&1
    echo "[$(date -Is)] Building selected directories (reuse completed entries)"
    python -u -m agmem --config "$BUILD_CONFIG" build-directories \
        > "$JOB_DIR/directories.log" 2>&1
fi
if [[ $STAGE == build ]]; then
    echo "[$(date -Is)] Selected directory build finished; no GRPO requested"
    exit 0
fi
if [[ -f $TRAIN_COMPLETE ]]; then
    echo "[$(date -Is)] Directory pass finished; selected experiment was already complete"
    exit 0
fi
CONFIG_PATH="$TRAIN_CONFIG"

# Only atomically completed checkpoints can be resumed. Preserve unfinished
# saves/logs in this attempt's directory, so retries cannot collide with .building.
RESUME_PATH=$(python - "$TRAIN_DIR" "$JOB_DIR" <<'PYRESUME'
from pathlib import Path
import sys

train_dir, job_dir = map(Path, sys.argv[1:])
checkpoints = sorted(path for path in train_dir.glob("checkpoint-*")
                     if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit())
if checkpoints:
    latest = checkpoints[-1]
    if not (latest / "trainer.pt").is_file() or not (latest / "model" / "config.json").is_file():
        raise SystemExit(f"Incomplete committed checkpoint: {latest}; inspect before retrying")
    for partial in train_dir.glob("checkpoint-*.building"):
        partial.rename(job_dir / partial.name)
        print(f"Archived unfinished checkpoint: {partial.name}", file=sys.stderr)
    print(latest)
elif train_dir.exists():
    # No optimizer state was committed: the first update must be started again.
    # Keep all evidence from the previous attempt, without altering prepared data.
    train_dir.rename(job_dir / "train_without_checkpoint")
    print("No completed checkpoint; archived previous training attempt and restarting from initial Reader", file=sys.stderr)
PYRESUME
)
TRAIN_ARGS=()
if [[ -n $RESUME_PATH ]]; then
    TRAIN_ARGS+=(--resume "$RESUME_PATH")
    echo "[$(date -Is)] Resuming full GRPO training from $RESUME_PATH"
else
    echo "[$(date -Is)] Starting full GRPO training (eight trajectories per group)"
fi
env CUDA_VISIBLE_DEVICES="$READER_GPU" python -u -m agmem --config "$CONFIG_PATH" \
    train "${TRAIN_ARGS[@]}" > "$JOB_DIR/train.log" 2>&1

date -Is > "$TRAIN_COMPLETE"
echo "[$(date -Is)] Full training completed successfully"
echo "Training log: $JOB_DIR/train.log"
echo "TensorBoard events: $TRAIN_DIR/tensorboard"
echo "View with: tensorboard --logdir '$TRAIN_DIR/tensorboard' --host 127.0.0.1 --port 6006"

exit 0
}

agmem_main "$@"
