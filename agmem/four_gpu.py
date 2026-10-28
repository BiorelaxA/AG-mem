"""Run two single-GPU 27B services and a two-GPU Reader in an existing allocation.

From the repository root: python -u -m agmem.four_gpu [--check-only | --resume]
No Slurm allocation is requested here. CUDA_VISIBLE_DEVICES remains authoritative.
"""
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from .core import atomic_json, file_digest, load_config
from .data import ADAPTER_VERSION, task_examples
from .jobs import file_lock, latest_checkpoint
from .pipeline import dataset_configs, validate_directories
from .store import Store


def gpu_roles(visible):
    ids = [part.strip() for part in visible.split(",")]
    if len(ids) != 4 or any(not part for part in ids) or len(set(ids)) != 4:
        raise ValueError(
            f"Expected four comma-separated GPU IDs in CUDA_VISIBLE_DEVICES; got {visible!r}. "
            "Run in an srun step of your existing four-GPU job "
            "(--overlap --nodes=1 --ntasks=1 --gres=gpu:4 --pty bash). "
            "nvidia-smi visibility alone does not establish the job's CUDA device mapping."
        )
    return {"writer": ids[0], "reader": ",".join(ids[1:3]), "task": ids[3]}


def configs(train_path="configs/train_dapo_locomo_s_v2.yaml"):
    train = load_config(train_path)
    build = load_config("configs/build_others_4x80gb.yaml")
    for config in (train, build):
        for key in ("input_root", "prepared_root"):
            config["data"][key] = str(Path(config["data"][key]).resolve())
        for key, value in config["models"].items():
            config["models"][key] = str(Path(value).resolve())
        for key in ("test_prepared_root", "legacy_prepared_root"):
            config["jobs"][key] = str(Path(config["jobs"][key]).resolve())
        config["training"]["output_dir"] = str(Path(config["training"]["output_dir"]).resolve())
    if set(train["data"]["sources"]) & set(build["data"]["sources"]):
        raise ValueError("Training sources must not overlap background directory sources")
    return train, build


def check_inputs(train):
    for path in train["models"].values():
        root = Path(path)
        index = json.loads((root / "model.safetensors.index.json").read_text())
        for name in set(index["weight_map"].values()):
            if not (root / name).is_file():
                raise ValueError(f"Missing model shard: {root / name}")
    report = {}
    raw = Path(train["data"]["input_root"])
    manifest = json.loads((raw / "manifest.json").read_text())
    entries = {row["source"]: row for row in manifest["sources"]}
    for source, child in dataset_configs(train):
        root = Path(child["data"]["prepared_root"])
        if not (root / "manifest.json").is_file():
            raise ValueError(f"Missing prepared data: {root}")
        # Read without close()/save_index(): validation must not modify the
        # imported JSONL/index files. Actual training can rebuild offset caches.
        store = Store(root)
        identity = store.get_metadata("prepare_identity", {})
        expected = {"input_manifest_sha256": file_digest(raw / "manifest.json"),
                    "sources": [source], "splits": ["train", "validation"],
                    "limit_per_source_split": None, "adapter_version": ADAPTER_VERSION,
                    "storage_format": Store.FORMAT}
        if not store.get_metadata("prepare_complete") or any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Prepared provenance differs: {root}")
        validate_directories(child, store, ["train", "validation"])
        counts = {}
        for split in ("train", "validation"):
            path = raw / source / f"{split}.parquet"
            if file_digest(path) != entries[source]["output_files"][split]["sha256"]:
                raise ValueError(f"Changed input: {path}")
            counts[split] = len(task_examples(child, store, split))
        report[source] = {**counts, "directory_version": store.get_metadata("directory_version"),
                          "original_writer": store.get_metadata("directory_service_identity")}
    return report


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server_limits(name, config):
    """Reserve workspace for full-vocabulary prompt logprobs on the Task GPU."""
    likelihood_mode = config.get("reward", {}).get("likelihood", {}).get("mode", "off")
    if name == "task" and likelihood_mode in {"diagnostic", "reward"}:
        # vLLM 0.22.1 materializes FP32 log_softmax for every prefill token,
        # even with prompt_logprobs=0. Qwen3.5's 248320-token vocabulary needs
        # 3.79 GiB for 4096 tokens, on top of logits, weights and the KV cache.
        # Chunking at 1024 lowers that tensor to 0.95 GiB; a smaller KV budget
        # and one active request leave headroom without shortening the context.
        return {"max_num_seqs": 1, "max_num_batched_tokens": 1024,
                "gpu_memory_utilization": 0.80}
    return {"max_num_seqs": 2, "max_num_batched_tokens": 4096,
            "gpu_memory_utilization": 0.90}


def stop_group(process):
    # The leader can exit before its workers; always address its process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader so a zombie cannot keep the group alive.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def wait_ready(process, port, log):
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"27B server exited ({process.returncode}); inspect {log}")
        try:
            with client.open(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"27B startup timed out; inspect {log}")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "agmem-27b", "messages": [{"role": "user", "content": "Reply with OK."}],
                         "max_tokens": 32, "temperature": 0,
                         "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with client.open(request, timeout=600) as response:
        result = json.load(response)
    if not result["choices"][0]["message"].get("content", "").strip():
        raise RuntimeError(f"27B inference probe returned no answer; inspect {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check cached training inputs without using GPUs")
    parser.add_argument("--resume", action="store_true", help="Resume the latest committed Reader checkpoint")
    parser.add_argument("--train-config", default="configs/train_dapo_locomo_s_v2.yaml")
    parser.add_argument("--max-updates", type=int, help="Stop after this many optimizer updates and save a checkpoint")
    args = parser.parse_args()
    train, build = configs(args.train_config)
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be positive")
    if args.check_only:
        print(json.dumps(check_inputs(train), ensure_ascii=False, indent=2))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inside your allocated Slurm GPU job")
    roles = gpu_roles(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    processes = []
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with ExitStack() as stack:
        stack.enter_context(file_lock(Path(train["data"]["prepared_root"]).parent / ".job.lock", shared=True))
        stack.enter_context(file_lock(Path(train["training"]["output_dir"]).parent / ".job.lock"))
        for source in sorted(train["data"]["sources"]):
            stack.enter_context(file_lock(Path(train["data"]["prepared_root"]) / ".locks" / f"{source}.lock"))
        report = check_inputs(train)
        destination = Path(train["training"]["output_dir"])
        checkpoint = latest_checkpoint(destination) if args.resume else None
        if args.resume and checkpoint is None:
            raise ValueError("--resume requested but no committed checkpoint exists")
        if not args.resume and destination.exists() and any(destination.iterdir()):
            raise ValueError("Training output already exists; use --resume or a new output_dir")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        job = Path("runs/four_gpu") / f"{os.environ['SLURM_JOB_ID']}-{stamp}"
        job.mkdir(parents=True)
        print(f"Logs: {job.resolve()}; GPU allocation: {roles}", flush=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"}
        # FlashInfer's sampler JIT can pick up a stale CUDA toolkit from another
        # conda environment. vLLM provides a native PyTorch sampler; keep this
        # launcher independent of nvcc for sampling (including startup profiling).
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        def start(name, command, devices, extra=None):
            log = stack.enter_context((job / f"{name}.log").open("w"))
            child_env = {**env, "CUDA_VISIBLE_DEVICES": devices, **(extra or {})}
            process = subprocess.Popen(command, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append(process)
            return process
        try:
            check = start("cuda_preflight", [sys.executable, "-c", "import torch; import vllm._C; "
                          "assert torch.cuda.device_count()==4; "
                          "[(torch.ones((32,32),device=f'cuda:{i}') @ torch.ones((32,32),device=f'cuda:{i}')).sum().item() for i in range(4)]; "
                          "print(torch.__version__, torch.version.cuda, 'CUDA kernels OK')"], os.environ["CUDA_VISIBLE_DEVICES"])
            if check.wait() != 0:
                raise RuntimeError(f"CUDA/vLLM preflight failed: {job / 'cuda_preflight.log'}; fix the environment before training")
            stop_group(check)
            processes.remove(check)
            check = start("reader_parallel_check", [sys.executable, "-m", "unittest", "tests.test_model_parallel", "-v"], roles["reader"])
            if check.wait() != 0:
                raise RuntimeError(f"Two-GPU Reader check failed: {job / 'reader_parallel_check.log'}")
            stop_group(check)
            processes.remove(check)
            import yaml
            servers = {}
            # Load sequentially to avoid two simultaneous 27B host-memory peaks.
            for name, config in (("task", train), ("writer", build)):
                port = free_port()
                limits = server_limits(name, config)
                atomic_json(job / f"{name}_server_limits.json", limits)
                print(f"{name} vLLM limits: {limits}", flush=True)
                config["service"]["request_base_url"] = f"http://127.0.0.1:{port}/v1"
                (job / f"{name}_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
                server = start(name + "_vllm", [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", config["models"]["task_path"], "--served-model-name", "agmem-27b",
                    "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
                    "--tensor-parallel-size", "1", "--max-model-len", "65536",
                    "--max-num-seqs", str(limits["max_num_seqs"]),
                    "--max-num-batched-tokens", str(limits["max_num_batched_tokens"]),
                    "--gpu-memory-utilization", str(limits["gpu_memory_utilization"]),
                    "--enable-chunked-prefill",
                    "--language-model-only", "--enforce-eager"], roles[name])
                servers[name] = server
                wait_ready(server, port, job / f"{name}_vllm.log")
                print(f"{name} 27B ready: GPU {roles[name]}, port {port}", flush=True)
            atomic_json(job / "inputs.json", report)
            atomic_json(job / "gpu_roles.json", roles)
            train_cmd = [sys.executable, "-u", "-m", "agmem", "--config", str(job / "task_config.yaml"), "train"]
            if checkpoint:
                train_cmd += ["--resume", str(checkpoint)]
            if args.max_updates is not None:
                train_cmd += ["--max-groups", str(args.max_updates)]
            workers = {
                "task": start("train", train_cmd, roles["reader"], {"AGMEM_READER_MODEL_PARALLEL_SIZE": "2", "OMP_NUM_THREADS": "8"}),
                "writer": start("directories", [sys.executable, "-u", "-m", "agmem.jobs", "directories",
                    "--config", str(job / "writer_config.yaml"), "--job-dir", str(job)], roles["writer"]),
            }
            print("Training and background directory construction started; validation/checkpoint every 25 steps.", flush=True)
            while workers:
                for name, worker in list(workers.items()):
                    status = worker.poll()
                    if status is not None:
                        if status != 0:
                            raise RuntimeError(f"{name} pipeline failed ({status}); inspect logs in {job}")
                        stop_group(servers[name])
                        processes.remove(servers[name])
                        stop_group(worker)
                        processes.remove(worker)
                        print(f"{name} pipeline completed", flush=True)
                        del workers[name]
                    elif servers[name].poll() is not None:
                        raise RuntimeError(f"{name} server exited unexpectedly; inspect logs in {job}")
                if workers:
                    time.sleep(2)
        finally:
            # Only terminate groups created by this launcher, never the Slurm job.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            for process in reversed(processes):
                stop_group(process)


if __name__ == "__main__":
    main()
