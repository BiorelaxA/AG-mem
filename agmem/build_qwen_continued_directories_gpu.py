"""Continue a verified DeepSeek build with one local Qwen Writer and embedding GPU.

Source validation and import run on CPU before either service starts. Only
process groups created by this invocation are stopped on completion or failure.
"""

import argparse
from contextlib import ExitStack
from copy import copy
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .build_adjacent_directories_gpu import (
    EMBEDDING_NAME, _validate_options, embedding_command, load_recipe,
    local_models, wait_embedding_ready, writer_command,
)
from .core import atomic_json
from .four_gpu import free_port, stop_group, wait_ready


DEFAULT_CONFIG = "configs/build_longmemeval_m_qwen_continue.yaml"


def allocated_devices(visible):
    devices = [part.strip() for part in visible.split(",")]
    if (len(devices) != 2 or len(set(devices)) != 2
            or any(part in {"", "-1", "NoDevFiles"} for part in devices)):
        raise ValueError("Select two distinct GPUs using AGMEM_BUILD_GPUS or CUDA_VISIBLE_DEVICES: "
                         "first embedding, then Qwen Writer (default: 4,5)")
    return devices


def build_command(args, endpoints=None, embedding_endpoint=None):
    command = [sys.executable, "-u", "-m", "agmem.build_qwen_continued_directories",
               "--config", args.config, "--workers", str(args.workers)]
    for name in ("datasets", "splits"):
        values = getattr(args, name, None)
        if values:
            command.extend(["--" + name, *values])
    for name in ("source_root", "embedding_model_path", "output_root", "limit_sessions"):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if endpoints:
        command.extend(["--base-urls", *endpoints])
    if embedding_endpoint:
        command.extend(["--embedding-base-url", embedding_endpoint])
    if args.import_only:
        command.append("--import-only")
    if args.dry_run:
        command.append("--dry-run")
    return command


def ensure_gpus_idle(devices, fractions):
    """Fail closed on occupied GPUs without initializing CUDA or stopping anyone."""
    try:
        result = subprocess.run([
            "nvidia-smi", "--id=" + ",".join(devices),
            "--query-gpu=uuid,memory.total,memory.free", "--format=csv,noheader,nounits",
        ], check=True, text=True, capture_output=True, timeout=30)
        rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
        if len(rows) != len(devices) or any(len(row) != 3 for row in rows):
            raise ValueError("Incomplete GPU status")
        uuids = {row[0].strip() for row in rows}
        if len(uuids) != len(devices):
            raise ValueError("GPU aliases identify the same device")
        for index, ((_, total, free), fraction) in enumerate(zip(rows, fractions)):
            total, free = float(total), float(free)
            if total <= 0 or free / total < fraction:
                raise RuntimeError("Selected GPU has insufficient free memory; existing processes are left running")
            if index == 1 and total < 70 * 1024:
                raise RuntimeError("Qwen Writer requires an 80-GB-class GPU for this configuration")
        processes = subprocess.run([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
        ], check=True, text=True, capture_output=True, timeout=30)
        for line in processes.stdout.splitlines():
            if line.split(",", 1)[0].strip() in uuids:
                raise RuntimeError("Selected GPU is busy; existing processes are left running. Select idle GPUs")
    except (OSError, subprocess.SubprocessError, ValueError):
        raise RuntimeError("Could not verify selected GPUs are idle; no services started") from None


def _models(args, recipe):
    writer, embedding = local_models(args, recipe)
    for model in (writer, embedding):
        index = Path(model) / "model.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if any(not (Path(model) / name).is_file() for name in shards):
                raise ValueError(f"Missing checkpoint shard under {model}")
    return writer, embedding


def launch(args):
    # Normalize the singular compatibility option before the shared validation.
    if args.base_url:
        args.base_urls = [args.base_url]
    _validate_options(args)
    if not args.base_urls and args.workers > args.max_num_seqs:
        raise ValueError("workers exceeds the single Writer's max_num_seqs; "
                         "reduce --workers or raise --max-num-seqs")
    cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    # The worker reads both identities and reports the verified import plan.
    # Dry runs do not create output, deployment directories, or subprocess services.
    if args.dry_run or args.import_only:
        subprocess.run(build_command(args, args.base_urls, args.embedding_base_url),
                       check=True, env=cpu_env)
        return None
    check = copy(args)
    check.dry_run = True
    subprocess.run(build_command(check, args.base_urls, args.embedding_base_url), check=True, env=cpu_env)

    recipe = load_recipe(args.config)
    devices = None
    if not args.base_urls:
        devices = allocated_devices(os.environ.get("AGMEM_BUILD_GPUS")
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
        if recipe.get("service", {}).get("model", "agmem-27b") != "agmem-27b":
            raise ValueError("Managed Writers require served model name agmem-27b")
        writer_model, embedding_model = _models(args, recipe)
        context_tokens = recipe.get("writer", {}).get("context_tokens", 65536)

    imported = copy(args)
    imported.import_only = True
    print("Verifying and importing completed DeepSeek work before local inference...", flush=True)
    subprocess.run(build_command(imported), check=True, env=cpu_env)
    if args.base_urls:
        subprocess.run(build_command(args, args.base_urls, args.embedding_base_url), check=True, env=cpu_env)
        return None

    ensure_gpus_idle(devices, [args.embedding_gpu_memory_utilization, args.gpu_memory_utilization])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    job = (Path(args.job_dir) if args.job_dir else
           Path("runs/qwen_continued_directory_build/jobs") / stamp)
    job.mkdir(parents=True, exist_ok=False)
    print(f"Logs: {job.resolve()}; embedding GPU: {devices[0]}; "
          f"Qwen Writer GPU: {devices[1]}; Writer concurrency: {args.workers}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"}
    env.pop("DEEPSEEK_API_KEY", None)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    processes, previous_signals = [], {}
    complete = False

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_signals[signum] = signal.signal(signum, interrupted)
        with ExitStack() as stack:
            def start(name, command, gpu_ids):
                log = stack.enter_context((job / f"{name}.log").open("w"))
                process = subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": gpu_ids},
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                return process

            try:
                embedding_port, writer_port = free_port(), free_port()
                while writer_port == embedding_port:
                    writer_port = free_port()
                embedding_url = f"http://127.0.0.1:{embedding_port}/v1"
                writer_url = f"http://127.0.0.1:{writer_port}/v1"
                atomic_json(job / "deployment.json", {
                    "source_root": args.source_root or recipe.get("source_root"),
                    "output_root": args.output_root or recipe.get("output_root"),
                    "writer": {"gpu": devices[1], "base_url": writer_url, "model": writer_model,
                               "served_model": "agmem-27b", "workers": args.workers,
                               "context_tokens": context_tokens, "max_num_seqs": args.max_num_seqs,
                               "max_num_batched_tokens": args.max_num_batched_tokens,
                               "gpu_memory_utilization": args.gpu_memory_utilization},
                    "tensor_parallel_size": 1,
                    "embedding": {"gpu": devices[0], "base_url": embedding_url,
                                  "model": embedding_model, "served_model": EMBEDDING_NAME,
                                  "expected_dimensions": dimensions,
                                  "max_model_len": args.embedding_max_model_len,
                                  "max_num_seqs": args.embedding_max_num_seqs,
                                  "max_num_batched_tokens": args.embedding_max_num_batched_tokens,
                                  "gpu_memory_utilization": args.embedding_gpu_memory_utilization,
                                  "client_concurrency": embedding.get("workers", 8),
                                  "client_batch_size": embedding.get("batch_size", 32)},
                })
                embedding_server = start("embedding", embedding_command(args, embedding_model, embedding_port),
                                         devices[0])
                wait_embedding_ready(embedding_server, embedding_port, job / "embedding.log",
                                     expected_dimensions=dimensions)
                writer_server = start("writer", writer_command(args, writer_model, writer_port, context_tokens),
                                      devices[1])
                wait_ready(writer_server, writer_port, job / "writer.log")
                worker = start("directories", build_command(args, [writer_url], embedding_url), "")
                servers = [("embedding", embedding_server), ("writer", writer_server)]
                print(f"Local services ready; progress: {job.resolve() / 'directories.log'}", flush=True)
                while worker.poll() is None:
                    failed = [name for name, process in servers if process.poll() is not None]
                    if failed:
                        raise RuntimeError(f"Services {failed} exited; inspect {job}. Rerun to resume.")
                    time.sleep(2)
                if worker.returncode:
                    raise RuntimeError(f"Directory builder failed ({worker.returncode}); "
                                       f"inspect {job / 'directories.log'}. Rerun to resume.")
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
                    raise RuntimeError("Owned process cleanup failed; GPU release is unconfirmed") from failures[0]
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    print(f"Directory construction finished; logs: {job.resolve()}", flush=True)
    return job


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-root")
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--datasets", nargs="+", choices=["split_longmemeval_m"], default=["split_longmemeval_m"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"],
                        default=["train", "validation", "test"])
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--output-root")
    parser.add_argument("--limit-sessions", type=int)
    endpoints = parser.add_mutually_exclusive_group()
    endpoints.add_argument("--base-url", help="Reuse one Writer together with --embedding-base-url")
    endpoints.add_argument("--base-urls", nargs="+", help="Reuse Writers together with --embedding-base-url")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.90)
    parser.add_argument("--embedding-max-model-len", type=int, default=8192)
    parser.add_argument("--embedding-max-num-seqs", type=int, default=64)
    parser.add_argument("--embedding-max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--embedding-gpu-memory-utilization", type=float, default=.35)
    parser.add_argument("--job-dir")
    parser.add_argument("--import-only", action="store_true", help="Import verified source work on CPU, then exit")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    launch(argument_parser().parse_args())


if __name__ == "__main__":
    main()
