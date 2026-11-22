"""Deploy three Qwen Writers to build parent/child directories over session text."""

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .core import atomic_json
from .four_gpu import free_port, stop_group, wait_ready


def allocated_devices(visible):
    devices = [value.strip() for value in visible.split(",")]
    if (len(devices) != 3 or len(set(devices)) != 3
            or any(value in {"", "-1", "NoDevFiles"} for value in devices)):
        raise ValueError("This launcher requires three allocated GPUs in CUDA_VISIBLE_DEVICES. "
                         "Run inside the three-GPU Slurm allocation, or pass --base-urls "
                         "to reuse already deployed Qwen services.")
    return devices


def build_command(args, endpoints=None):
    module = ("agmem.build_local_directories" if getattr(args, "legacy_flat", False)
              else "agmem.build_parent_child_directories")
    command = [sys.executable, "-u", "-m", module,
               "--config", args.config, "--workers", str(args.workers)]
    for name in ("datasets", "splits"):
        values = getattr(args, name, None)
        if values:
            command.extend(["--" + name, *values])
    if endpoints:
        command.extend(["--base-urls", *endpoints])
    if args.dry_run:
        command.append("--dry-run")
    return command


def server_command(args, model, port):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--served-model-name", "agmem-27b",
            "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
            "--tensor-parallel-size", "1", "--max-model-len", "65536",
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--enable-chunked-prefill", "--language-model-only", "--enforce-eager"]


def model_path(recipe_path):
    import yaml
    recipe = yaml.safe_load(Path(recipe_path).read_text())
    model = recipe.get("model_path")
    if not model:
        saved = json.loads((Path(recipe["training_run"]) / "run.json").read_text())
        model = saved["config"]["models"]["task_path"]
    model = Path(model).resolve()
    if not (model / "config.json").is_file():
        raise ValueError(f"Missing local Qwen model: {model}")
    return str(model)


def launch(args):
    for name in ("workers", "max_num_seqs", "max_num_batched_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must lie between zero and one")
    endpoints = args.base_urls or ([args.base_url] if args.base_url else None)
    # Neither mode owns a model process. A dry run must also work on a CPU node.
    if args.dry_run or endpoints:
        subprocess.run(build_command(args, endpoints), check=True)
        return None
    devices = allocated_devices(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    model = model_path(args.config)
    # Validate input provenance before allocating three model replicas.
    from copy import copy
    check = copy(args)
    check.dry_run = True
    subprocess.run(build_command(check), check=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    job = (Path(args.job_dir) if args.job_dir else Path("runs/longmemeval_qwen_build/jobs") /
           f"{os.environ.get('SLURM_JOB_ID', 'local')}-{stamp}")
    job.mkdir(parents=True, exist_ok=False)
    print(f"Logs: {job.resolve()}; Qwen Writer GPUs: {','.join(devices)}; "
          f"total concurrent requests: {args.workers}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"}
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    processes = []
    previous_signals = {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signals[signum] = signal.signal(signum, interrupted)
    try:
        with ExitStack() as stack:
            def start(name, command, gpu_ids):
                log = stack.enter_context((job / f"{name}.log").open("w"))
                process = subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": gpu_ids},
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                return process

            construction_complete = False
            try:
                preflight_code = (
                    "import json, torch; import vllm._C\n"
                    "assert torch.cuda.is_available() and torch.cuda.device_count()==3, 'Expected three CUDA GPUs'\n"
                    "for i in range(3):\n"
                    " free,total=torch.cuda.mem_get_info(i)\n"
                    " print(json.dumps({'gpu':i,'name':torch.cuda.get_device_name(i),'free_gib':free/2**30,'total_gib':total/2**30}),flush=True)\n"
                    " assert total >= 70*2**30, 'Each Qwen replica requires an 80-GB-class GPU for this configuration'\n"
                    f" assert free/total >= {args.gpu_memory_utilization}, 'GPU is occupied: stop the existing training/service before building directories'\n"
                    " (torch.ones((32,32),device=f'cuda:{i}') @ torch.ones((32,32),device=f'cuda:{i}')).sum().item()\n"
                )
                preflight = start("cuda_preflight", [sys.executable, "-c", preflight_code], ",".join(devices))
                if preflight.wait() != 0:
                    raise RuntimeError(f"CUDA/vLLM preflight failed; inspect {job / 'cuda_preflight.log'}. "
                                       "The launcher does not stop existing training processes.")
                ports = []
                for _ in devices:
                    port = free_port()
                    while port in ports:
                        port = free_port()
                    ports.append(port)
                endpoints = [f"http://127.0.0.1:{port}/v1" for port in ports]
                atomic_json(job / "deployment.json", {
                    "model": model, "tensor_parallel_size": 1, "writer_workers": args.workers,
                    "replicas": [{"gpu": device, "base_url": url} for device, url in zip(devices, endpoints)],
                    "max_num_seqs_per_replica": args.max_num_seqs,
                    "max_num_batched_tokens": args.max_num_batched_tokens,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                })
                # Loading 27B replicas sequentially limits transient host RAM use.
                # All three serve requests concurrently once initialization ends.
                servers = []
                for i, (device, port) in enumerate(zip(devices, ports)):
                    server = start(f"writer_{i}", server_command(args, model, port), device)
                    servers.append(server)
                    wait_ready(server, port, job / f"writer_{i}.log")
                worker = start("directories", build_command(args, endpoints), "")
                print(f"All three Writers are ready; progress: {job.resolve() / 'directories.log'}", flush=True)
                while worker.poll() is None:
                    failed = [i for i, server in enumerate(servers) if server.poll() is not None]
                    if failed:
                        raise RuntimeError(f"Writer replicas {failed} exited; inspect writer logs in {job}. "
                                           "Rerun to reuse completed entries.")
                    time.sleep(2)
                if worker.returncode:
                    raise RuntimeError(f"Directory builder failed ({worker.returncode}); "
                                       f"inspect {job / 'directories.log'}. Rerun to resume.")
                print(f"Directory construction finished; logs: {job.resolve()}", flush=True)
                construction_complete = True
            finally:
                # Only this invocation's process groups are owned here. Never pkill a model name.
                cleanup_failures = []
                for process in reversed(processes):
                    try:
                        stop_group(process)
                    except Exception as exc:
                        cleanup_failures.append((process.pid, exc))
                        print(f"Could not clean up owned process {process.pid}: {type(exc).__name__}",
                              file=sys.stderr, flush=True)
                if cleanup_failures and construction_complete:
                    # A sequential caller must not deploy its next models while
                    # this launcher's Writers may still occupy the GPUs. Keep
                    # any original construction exception when it already failed.
                    pids = ", ".join(str(pid) for pid, _ in cleanup_failures)
                    raise RuntimeError(
                        f"Directory construction completed, but cleanup failed for owned processes {pids}; "
                        "GPU release is unconfirmed. Do not start the next stage until they have stopped."
                    ) from cleanup_failures[0][1]
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/build_parent_child_qwen.yaml")
    parser.add_argument("--legacy-flat", action="store_true", help="Explicitly select the old flat-summary builder with its old recipe")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--datasets", nargs="+", choices=["locomo", "split_longmemeval_s", "split_longmemeval_m"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"])
    endpoint = parser.add_mutually_exclusive_group()
    endpoint.add_argument("--base-url", help="Reuse one running Qwen service; do not deploy GPUs")
    endpoint.add_argument("--base-urls", nargs="+", help="Reuse running Qwen replicas; do not deploy GPUs")
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--job-dir")
    parser.add_argument("--dry-run", action="store_true")
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
