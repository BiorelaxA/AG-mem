"""Build LongMemEval-M directories with a DeepSeek Writer and local embeddings.

One or two selected GPUs serve Qwen3-Embedding-4B replicas. All Writer requests use
the remote API. Only process groups created by this invocation are stopped.
"""

import argparse
from contextlib import ExitStack
from copy import copy
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

from .build_adjacent_directories_gpu import (
    EMBEDDING_NAME, embedding_command, load_recipe, wait_embedding_ready,
)
from .core import atomic_json
from .four_gpu import free_port, stop_group
from .hierarchical.embedding_pool import embedding_endpoints


DEFAULT_CONFIG = "configs/build_longmemeval_m_deepseek.yaml"


def allocated_devices(visible):
    devices = [part.strip() for part in visible.split(",")]
    if (len(devices) not in {1, 2} or len(set(devices)) != len(devices)
            or any(part in {"", "-1", "NoDevFiles"} for part in devices)):
        raise ValueError("Select one or two distinct embedding GPUs using AGMEM_EMBEDDING_GPUS "
                         "or CUDA_VISIBLE_DEVICES (default: 4,5)")
    return devices


def build_command(args, endpoints=None):
    command = [sys.executable, "-u", "-m", "agmem.build_deepseek_directories",
               "--config", args.config, "--workers", str(args.workers)]
    for name in ("datasets", "splits"):
        values = getattr(args, name, None)
        if values:
            command.extend(["--" + name, *values])
    for name in ("embedding_model_path", "output_root", "limit_sessions"):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if endpoints:
        command.extend(["--embedding-base-urls", *endpoints])
    if args.dry_run:
        command.append("--dry-run")
    return command


def local_embedding_model(args, recipe):
    model = Path(args.embedding_model_path or recipe.get("embedding", {}).get(
        "model_path", "models/qwen3-embedding-4b")).expanduser().resolve()
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    if any(not (model / name).is_file() for name in required) or not any(model.glob("*.safetensors")):
        raise ValueError(f"Missing or incomplete local embedding model: {model}")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        import json
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if any(not (model / name).is_file() for name in shards):
            raise ValueError(f"Missing embedding checkpoint shard under {model}")
    return str(model)


def deepseek_endpoint(recipe):
    service = recipe.get("service", {})
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or service.get("base_url", "https://api.deepseek.com")
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except (ValueError, TypeError):
        raise ValueError("Invalid DeepSeek base URL") from None
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname or port == 0
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("DeepSeek base URL must not contain credentials/query/fragment")
    return base_url.rstrip("/")


def probe_deepseek(recipe):
    """Authenticate with the free model-list endpoint before occupying GPUs."""
    import httpx
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("Set DEEPSEEK_API_KEY before starting the build")
    endpoint = deepseek_endpoint(recipe)
    model = recipe.get("service", {}).get("model", "deepseek-flash")
    try:
        with httpx.Client(timeout=30, trust_env=recipe.get("service", {}).get("trust_env", True)) as client:
            response = client.get(endpoint + "/models", headers={"Authorization": "Bearer " + key})
    except (httpx.HTTPError, ValueError) as exc:
        causes = [type(exc).__name__]
        cause = exc.__cause__
        while cause is not None and len(causes) < 5:
            causes.append(type(cause).__name__)
            cause = cause.__cause__
        raise RuntimeError(f"DeepSeek GET /models request failed ({' <- '.join(causes)})") from None
    if response.status_code != 200:
        raise RuntimeError(f"DeepSeek authentication/model probe returned HTTP {response.status_code}")
    try:
        models = response.json()["data"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"DeepSeek GET /models returned HTTP 200 with an invalid model list "
                           f"({type(exc).__name__})") from None
    if not isinstance(models, list):
        raise RuntimeError("DeepSeek GET /models returned HTTP 200 with an invalid data field")
    if model not in {row["id"] for row in models if isinstance(row, dict) and isinstance(row.get("id"), str)}:
        raise RuntimeError("Configured Writer model was absent from DeepSeek /models")
    return endpoint, model


def ensure_gpus_idle(devices, memory_fraction):
    """Read GPU status without opening CUDA or affecting existing processes."""
    try:
        status = subprocess.run([
            "nvidia-smi", "--id=" + ",".join(devices),
            "--query-gpu=uuid,memory.total,memory.free", "--format=csv,noheader,nounits",
        ], check=True, text=True, capture_output=True, timeout=30)
        rows = [line.split(",") for line in status.stdout.splitlines() if line.strip()]
        if len(rows) != len(devices) or any(len(row) != 3 for row in rows):
            raise ValueError("Incomplete GPU status")
        uuids = {row[0].strip() for row in rows}
        for _, total, free in rows:
            if float(total) <= 0 or float(free) / float(total) < memory_fraction:
                raise RuntimeError("Selected embedding GPU has insufficient free memory; "
                                   "existing processes are left running")
        processes = subprocess.run([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
        ], check=True, text=True, capture_output=True, timeout=30)
        for line in processes.stdout.splitlines():
            if line.split(",", 1)[0].strip() in uuids:
                raise RuntimeError("Selected embedding GPU is busy; existing training or services "
                                   "are left running. Select idle embedding GPUs")
    except (OSError, subprocess.SubprocessError, ValueError):
        raise RuntimeError("Could not verify selected embedding GPUs are idle; no services started") from None


def _validate_options(args):
    for name in ("workers", "embedding_max_model_len", "embedding_max_num_seqs",
                 "embedding_max_num_batched_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 < args.embedding_gpu_memory_utilization < 1:
        raise ValueError("embedding_gpu_memory_utilization must lie between zero and one")
    if args.embedding_max_model_len > 32768:
        raise ValueError("Qwen3 embedding deployment is limited to 32768 input tokens")
    if args.embedding_max_num_batched_tokens < args.embedding_max_model_len:
        raise ValueError("Embedding batched token capacity must cover a complete input")
    if args.limit_sessions is not None and args.limit_sessions < 1:
        raise ValueError("limit_sessions must be positive")
    if args.embedding_base_urls:
        embedding_endpoints(args.embedding_base_urls)


def launch(args):
    _validate_options(args)
    # CPU-only validation reads local files but loads no GPU weights, and needs
    # no credentials, CUDA initialization or network access.
    if args.dry_run:
        subprocess.run(build_command(args, args.embedding_base_urls), check=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        return None
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("Set DEEPSEEK_API_KEY before starting the build")
    recipe = load_recipe(args.config)
    check = copy(args)
    check.dry_run = True
    subprocess.run(build_command(check, args.embedding_base_urls), check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if args.embedding_base_urls:
        probe_deepseek(recipe)
        subprocess.run(build_command(args, args.embedding_base_urls), check=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        return None
    devices = allocated_devices(os.environ.get("AGMEM_EMBEDDING_GPUS")
                                or os.environ.get("CUDA_VISIBLE_DEVICES", "4,5"))
    embedding = recipe.get("embedding", {})
    dimensions = embedding.get("expected_dimensions", 2560)
    tokens = embedding.get("max_input_tokens", 4096)
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("embedding.expected_dimensions must be a positive integer")
    if type(tokens) is not int or not 0 < tokens <= args.embedding_max_model_len:
        raise ValueError("embedding.max_input_tokens exceeds the deployed model capacity")
    if embedding.get("model", EMBEDDING_NAME) != EMBEDDING_NAME:
        raise ValueError(f"Managed embeddings require served model name {EMBEDDING_NAME}")
    model = local_embedding_model(args, recipe)
    # All checks here precede any process that can allocate GPU memory.
    ensure_gpus_idle(devices, args.embedding_gpu_memory_utilization)
    writer_endpoint, writer_model = probe_deepseek(recipe)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    job = Path(args.job_dir) if args.job_dir else Path("runs/deepseek_directory_build/jobs") / stamp
    job.mkdir(parents=True, exist_ok=False)
    print(f"Logs: {job.resolve()}; embedding GPUs: {','.join(devices)}; "
          f"remote Writer concurrency: {args.workers}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"}
    processes, previous_signals = [], {}
    complete = False

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signals[signum] = signal.signal(signum, interrupted)
    try:
        with ExitStack() as stack:
            def start(name, command, gpu_ids):
                log = stack.enter_context((job / f"{name}.log").open("w"))
                child_env = {**env, "CUDA_VISIBLE_DEVICES": gpu_ids}
                if gpu_ids:
                    child_env.pop("DEEPSEEK_API_KEY", None)
                process = subprocess.Popen(command, env=child_env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                return process

            try:
                ports = []
                for _ in devices:
                    port = free_port()
                    while port in ports:
                        port = free_port()
                    ports.append(port)
                urls = [f"http://127.0.0.1:{port}/v1" for port in ports]
                atomic_json(job / "deployment.json", {
                    "writer": {"provider": "deepseek", "base_url": writer_endpoint,
                               "model": writer_model, "workers": args.workers},
                    "embedding_model": model, "tensor_parallel_size": 1,
                    "embedding_replicas": [{"gpu": gpu, "base_url": url}
                                           for gpu, url in zip(devices, urls)],
                    "embedding": {"served_model": EMBEDDING_NAME,
                                  "expected_dimensions": dimensions,
                                  "max_model_len": args.embedding_max_model_len,
                                  "max_num_seqs_per_replica": args.embedding_max_num_seqs,
                                  "max_num_batched_tokens": args.embedding_max_num_batched_tokens,
                                  "gpu_memory_utilization": args.embedding_gpu_memory_utilization,
                                  "total_client_concurrency": embedding.get("workers", 4),
                                  "client_batch_size": embedding.get("batch_size", 32)},
                })
                servers = []
                # Load replicas sequentially to limit host RAM peaks.
                for index, (gpu, port) in enumerate(zip(devices, ports), 1):
                    name = f"embedding_{index}"
                    process = start(name, embedding_command(args, model, port), gpu)
                    servers.append((name, process))
                    wait_embedding_ready(process, port, job / f"{name}.log",
                                         expected_dimensions=dimensions)
                worker = start("directories", build_command(args, urls), "")
                print(f"{len(servers)} embedding replica(s) ready; progress: {job.resolve() / 'directories.log'}", flush=True)
                while worker.poll() is None:
                    failed = [name for name, process in servers if process.poll() is not None]
                    if failed:
                        raise RuntimeError(f"Embedding services {failed} exited; inspect {job}")
                    time.sleep(2)
                if worker.returncode:
                    raise RuntimeError(f"Directory builder failed ({worker.returncode}); "
                                       f"inspect {job / 'directories.log'}")
                complete = True
            finally:
                failures = []
                for process in reversed(processes):
                    try:
                        stop_group(process)
                    except Exception as exc:
                        failures.append(exc)
                        print(f"Could not clean up owned process {process.pid}: {type(exc).__name__}",
                              file=sys.stderr, flush=True)
                if failures and complete:
                    raise RuntimeError("Owned process cleanup failed; GPU release is unconfirmed") from None
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    print(f"Directory construction finished; logs: {job.resolve()}", flush=True)
    return job


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", choices=["split_longmemeval_m"], default=["split_longmemeval_m"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"],
                        default=["train", "validation", "test"])
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--output-root")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--embedding-base-urls", nargs="+", help="Reuse already running embedding services")
    parser.add_argument("--embedding-max-model-len", type=int, default=8192)
    parser.add_argument("--embedding-max-num-seqs", type=int, default=64)
    parser.add_argument("--embedding-max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--embedding-gpu-memory-utilization", type=float, default=.35)
    parser.add_argument("--job-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    launch(argument_parser().parse_args())


if __name__ == "__main__":
    main()
