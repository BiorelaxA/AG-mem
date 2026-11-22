"""Run an embedding service and one to three Qwen Writers on allocated GPUs.

The first visible GPU embeds; the remaining GPUs each serve a Writer.
Only processes created by this invocation are stopped on completion or failure.
"""

import argparse
from contextlib import ExitStack
from copy import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from .core import atomic_json
from .four_gpu import free_port, stop_group, wait_ready


DEFAULT_CONFIG = "configs/build_adjacent_qwen_locomo_lmes.yaml"
EMBEDDING_NAME = "agmem-embedding"


def allocated_devices(visible):
    """Keep the selected physical IDs intact for two- to four-GPU builds."""
    devices = [value.strip() for value in visible.split(",")]
    if (len(devices) not in {2, 3, 4} or len(set(devices)) != len(devices)
            or any(value in {"", "-1", "NoDevFiles"} for value in devices)):
        raise ValueError("This launcher requires two to four distinct allocated GPUs in "
                         "CUDA_VISIBLE_DEVICES: first embedding, then one to three Writers. "
                         "For example, CUDA_VISIBLE_DEVICES=4,5 selects only GPUs 4 and 5. "
                         "Provide --base-urls and --embedding-base-url to reuse external services.")
    return devices


def build_command(args, endpoints=None, embedding_endpoint=None):
    module = ("agmem.build_qwen_m_eval_directories" if getattr(args, "m_eval", False)
              else "agmem.build_segment_context_directories" if getattr(args, "segment_context", False)
              else "agmem.build_selected_topic_event_directories" if getattr(args, "scope_completion", False)
              else "agmem.repair_adjacent_fallbacks" if getattr(args, "repair_fallbacks", False)
              else "agmem.build_topic_event_directories" if getattr(args, "topic_event", False)
              else "agmem.build_adjacent_directories")
    command = [sys.executable, "-u", "-m", module,
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
        command.extend(["--base-urls", *endpoints])
    if embedding_endpoint:
        command.extend(["--embedding-base-url", embedding_endpoint])
    if args.dry_run:
        command.append("--dry-run")
    return command


def reuse_command(args):
    """Rebind proven identical histories before starting the inference services."""
    module = ("agmem.reuse_topic_event_directories" if getattr(args, "preserve_existing", False)
              else "agmem.reuse_adjacent_directories")
    command = [sys.executable, "-u", "-m", module,
               "--config", args.config, "--reuse-root", args.reuse_root,
               "--mapping", args.reuse_mapping]
    for name in ("datasets", "splits"):
        values = getattr(args, name, None)
        if values:
            command.extend(["--" + name, *values])
    for name in ("embedding_model_path", "output_root"):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def load_recipe(path):
    import yaml
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Directory build recipe must be a mapping")
    return value


def local_models(args, recipe):
    writer = Path(recipe["model_path"]).expanduser().resolve()
    embedding = Path(args.embedding_model_path or recipe.get("embedding", {}).get(
        "model_path", "../model/qwen3-embedding-4b")).expanduser().resolve()
    for name, path in (("Writer", writer), ("embedding", embedding)):
        required = ("config.json", "tokenizer.json", "tokenizer_config.json")
        missing = [filename for filename in required if not (path / filename).is_file()]
        if missing or not any(path.glob("*.safetensors")):
            suffix = (" Run bash scripts/download_directory_embedding.sh on a network-enabled "
                      "node, or set --embedding-model-path to a complete local checkpoint."
                      if name == "embedding" else " Set model_path in the build recipe.")
            raise ValueError(f"Missing or incomplete local {name} model: {path}.{suffix}")
    return str(writer), str(embedding)


def writer_command(args, model, port, context_tokens=65536):
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--served-model-name", "agmem-27b",
            "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
            "--tensor-parallel-size", "1", "--max-model-len", str(context_tokens),
            "--max-num-seqs", str(args.max_num_seqs),
            "--max-num-batched-tokens", str(args.max_num_batched_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--enable-chunked-prefill", "--language-model-only", "--enforce-eager"]


def embedding_command(args, model, port):
    # Qwen3-Embedding's published config uses Qwen3ForCausalLM. The installed
    # vLLM adapter supports LAST-token normalized embeddings via convert=embed.
    # Pooling chooses its own supported prefill/cache defaults.
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model, "--served-model-name", EMBEDDING_NAME,
            "--runner", "pooling", "--convert", "embed",
            "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
            "--tensor-parallel-size", "1", "--max-model-len", str(args.embedding_max_model_len),
            "--max-num-seqs", str(args.embedding_max_num_seqs),
            "--max-num-batched-tokens", str(args.embedding_max_num_batched_tokens),
            "--gpu-memory-utilization", str(args.embedding_gpu_memory_utilization),
            "--enforce-eager"]


def wait_embedding_ready(process, port, log, *, expected_dimensions=2560):
    if type(expected_dimensions) is not int or expected_dimensions < 1:
        raise ValueError("embedding.expected_dimensions must be a positive integer")
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Embedding server exited ({process.returncode}); inspect {log}")
        try:
            with client.open(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        time.sleep(2)
    else:
        raise RuntimeError(f"Embedding startup timed out; inspect {log}")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/embeddings",
        data=json.dumps({"model": EMBEDDING_NAME, "input": ["A short topic segmentation probe."],
                         "encoding_format": "float"}).encode(),
        headers={"Content-Type": "application/json"})
    with client.open(request, timeout=180) as response:
        result = json.load(response)
    data = result.get("data", [])
    vector = data[0].get("embedding") if len(data) == 1 else None
    if (not isinstance(vector, list) or len(vector) != expected_dimensions or data[0].get("index") != 0
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
            or not any(vector)):
        raise RuntimeError(f"Embedding inference probe returned an invalid {expected_dimensions}-dimensional vector; inspect {log}")


def _validate_writer_capacity(args, writer_replicas):
    if not args.base_urls and args.workers > writer_replicas * args.max_num_seqs:
        raise ValueError(f"workers exceeds {writer_replicas} Writer replica(s)' combined max_num_seqs; "
                         "reduce --workers or raise --max-num-seqs explicitly")


def _validate_options(args):
    for name in ("workers", "max_num_seqs", "max_num_batched_tokens", "embedding_max_model_len",
                 "embedding_max_num_seqs", "embedding_max_num_batched_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("gpu_memory_utilization", "embedding_gpu_memory_utilization"):
        if not 0 < getattr(args, name) < 1:
            raise ValueError(f"{name} must lie between zero and one")
    if args.embedding_max_model_len > 32768:
        raise ValueError("This Qwen3 embedding deployment is limited to 32768 input tokens")
    if args.embedding_max_num_batched_tokens < args.embedding_max_model_len:
        raise ValueError("Embedding max_num_batched_tokens must cover a complete embedding input")
    if args.limit_sessions is not None and args.limit_sessions < 1:
        raise ValueError("limit_sessions must be positive")
    if bool(args.base_urls) != bool(args.embedding_base_url):
        raise ValueError("To reuse services, provide both --base-urls and --embedding-base-url")
    # The actual managed replica count is checked after resolving the allocation.


def launch(args):
    _validate_options(args)
    if getattr(args, "segment_context", False):
        if any(getattr(args, name, False) for name in (
                "topic_event", "repair_fallbacks", "reuse_root", "reuse_mapping",
                "preserve_existing", "scope_completion")):
            raise ValueError("Segment-context construction requires a separate build without "
                             "topic-event mode, legacy reuse, scope completion or fallback-only repair")
        if args.datasets and set(args.datasets) != {"locomo"}:
            raise ValueError("Segment-context construction currently supports LoCoMo only")
    if getattr(args, 'scope_completion', False) and (
            not getattr(args, 'topic_event', False) or getattr(args, 'repair_fallbacks', False)):
        raise ValueError('--scope-completion requires --topic-event without --repair-fallbacks')
    devices = None
    if not args.base_urls:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # CPU-only plans need no devices. If the user supplies an allocation,
        # validate its effective Writer capacity even during a dry run.
        if not args.dry_run or visible.strip() not in {"", "-1", "NoDevFiles"}:
            devices = allocated_devices(visible)
            _validate_writer_capacity(args, len(devices) - 1)
    reuse_root = getattr(args, "reuse_root", None)
    reuse_mapping = getattr(args, "reuse_mapping", None)
    if bool(reuse_root) != bool(reuse_mapping):
        raise ValueError("Provide both --reuse-root and --reuse-mapping")
    if reuse_root and getattr(args, "repair_fallbacks", False):
        raise ValueError("Dataset rebinding and in-place fallback repair are separate operations")
    preserve_existing = getattr(args, "preserve_existing", False)
    if preserve_existing and (not getattr(args, "topic_event", False) or not reuse_root
                              or getattr(args, "repair_fallbacks", False)):
        raise ValueError("--preserve-existing requires --topic-event, --reuse-root and --reuse-mapping; "
                         "it cannot be used for in-place fallback repair")
    if getattr(args, "topic_event", False) and (getattr(args, "repair_fallbacks", False)
                                               or (reuse_root and not preserve_existing)):
        raise ValueError("Topic/event v2 requires a separate build; v1 directories and boundary decisions "
                         "cannot be relabeled as new-prompt results. Use --preserve-existing for an "
                         "explicitly documented mixed-version incremental build")

    def import_existing():
        if not reuse_root:
            return
        print("Checking existing directory provenance and rebinding matching histories...", flush=True)
        subprocess.run(reuse_command(args), check=True)

    # Existing services and dry runs do not own processes or require CUDA.
    if args.dry_run or args.base_urls:
        import_existing()
        subprocess.run(build_command(args, args.base_urls, args.embedding_base_url), check=True)
        return None
    recipe = load_recipe(args.config)
    embedding_config = recipe.get("embedding", {})
    embedding_input = embedding_config.get("max_input_tokens", 4096)
    if type(embedding_input) is not int or not 0 < embedding_input <= args.embedding_max_model_len:
        raise ValueError("embedding.max_input_tokens must be positive and no larger than "
                         "--embedding-max-model-len; refusing to deploy an undersized embedding service")
    embedding_dimensions = embedding_config.get("expected_dimensions", 2560)
    if type(embedding_dimensions) is not int or embedding_dimensions < 1:
        raise ValueError("embedding.expected_dimensions must be a positive integer")
    writer_model, embedding_model = local_models(args, recipe)
    if recipe.get("service", {}).get("model", "agmem-27b") != "agmem-27b":
        raise ValueError("Managed Writers use served model name agmem-27b; update service.model")
    if recipe.get("embedding", {}).get("model", EMBEDDING_NAME) != EMBEDDING_NAME:
        raise ValueError(f"Managed embedding service uses served model name {EMBEDDING_NAME}; update embedding.model")
    context_tokens = recipe.get("writer", {}).get("context_tokens", 65536)
    check = copy(args)
    check.dry_run = True
    subprocess.run(build_command(check), check=True)
    import_existing()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    job = (Path(args.job_dir) if args.job_dir else Path("runs/adjacent_directory_build/jobs") /
           f"{os.environ.get('SLURM_JOB_ID', 'local')}-{stamp}")
    job.mkdir(parents=True, exist_ok=False)
    print(f"Logs: {job.resolve()}; embedding GPU: {devices[0]}; "
          f"Qwen Writer GPUs: {','.join(devices[1:])}; total Writer concurrency: {args.workers}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"}
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    processes, previous_signals = [], {}

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
                gpu_count = len(devices)
                fractions = [args.embedding_gpu_memory_utilization] + [args.gpu_memory_utilization] * (gpu_count - 1)
                preflight_code = (
                    "import json, torch; import vllm._C\n"
                    f"assert torch.cuda.is_available() and torch.cuda.device_count()=={gpu_count}, 'Expected {gpu_count} CUDA GPUs'\n"
                    f"fractions={fractions!r}\n"
                    f"for i in range({gpu_count}):\n"
                    " free,total=torch.cuda.mem_get_info(i)\n"
                    " print(json.dumps({'gpu':i,'role':'embedding' if i==0 else 'writer','name':torch.cuda.get_device_name(i),'free_gib':free/2**30,'total_gib':total/2**30}),flush=True)\n"
                    " assert i==0 or total >= 70*2**30, 'Each Writer requires an 80-GB-class GPU for this configuration'\n"
                    " assert free/total >= fractions[i], 'GPU is occupied: stop the existing service before launching the new build'\n"
                    " (torch.ones((32,32),device=f'cuda:{i}') @ torch.ones((32,32),device=f'cuda:{i}')).sum().item()\n"
                )
                preflight = start("cuda_preflight", [sys.executable, "-c", preflight_code], ",".join(devices))
                if preflight.wait() != 0:
                    raise RuntimeError(f"CUDA/vLLM preflight failed; inspect {job / 'cuda_preflight.log'}. "
                                       "Existing training/build/service processes are never stopped by this launcher.")
                ports = []
                for _ in devices:
                    port = free_port()
                    while port in ports:
                        port = free_port()
                    ports.append(port)
                urls = [f"http://127.0.0.1:{port}/v1" for port in ports]
                atomic_json(job / "deployment.json", {
                    "writer_model": writer_model, "embedding_model": embedding_model,
                    "tensor_parallel_size": 1, "writer_workers": args.workers,
                    "writer_replicas": [{"gpu": device, "base_url": url}
                                        for device, url in zip(devices[1:], urls[1:])],
                    "writer_context_tokens": context_tokens,
                    "max_num_seqs_per_writer": args.max_num_seqs,
                    "writer_max_num_batched_tokens": args.max_num_batched_tokens,
                    "writer_gpu_memory_utilization": args.gpu_memory_utilization,
                    "embedding": {"gpu": devices[0], "base_url": urls[0], "served_model": EMBEDDING_NAME,
                                  "runner": "pooling", "convert": "embed",
                                  "expected_dimensions": embedding_dimensions,
                                  "max_model_len": args.embedding_max_model_len,
                                  "max_num_seqs": args.embedding_max_num_seqs,
                                  "max_num_batched_tokens": args.embedding_max_num_batched_tokens,
                                  "gpu_memory_utilization": args.embedding_gpu_memory_utilization,
                                  "client_concurrency": embedding_config.get("workers"),
                                  "client_batch_size": embedding_config.get("batch_size")},
                })
                servers = []
                embedding_server = start("embedding", embedding_command(args, embedding_model, ports[0]), devices[0])
                servers.append(("embedding", embedding_server))
                wait_embedding_ready(embedding_server, ports[0], job / "embedding.log",
                                     expected_dimensions=embedding_dimensions)
                # Sequential loading limits host RAM peaks; inference is concurrent.
                for index, (device, port) in enumerate(zip(devices[1:], ports[1:]), 1):
                    name = f"writer_{index}"
                    process = start(name, writer_command(args, writer_model, port, context_tokens), device)
                    servers.append((name, process))
                    wait_ready(process, port, job / f"{name}.log")
                worker = start("directories", build_command(args, urls[1:], urls[0]), "")
                print(f"Embedding and {len(devices) - 1} Writer replica(s) are ready; "
                      f"progress: {job.resolve() / 'directories.log'}", flush=True)
                while worker.poll() is None:
                    failed = [name for name, process in servers if process.poll() is not None]
                    if failed:
                        raise RuntimeError(f"Services {failed} exited; inspect logs in {job}. Rerun to resume.")
                    time.sleep(2)
                if worker.returncode:
                    raise RuntimeError(f"Directory builder failed ({worker.returncode}); "
                                       f"inspect {job / 'directories.log'}. Rerun to resume.")
                print(f"Directory construction finished; logs: {job.resolve()}", flush=True)
                construction_complete = True
            finally:
                cleanup_failures = []
                for process in reversed(processes):
                    try:
                        stop_group(process)
                    except Exception as exc:
                        cleanup_failures.append((process.pid, exc))
                        print(f"Could not clean up owned process {process.pid}: {type(exc).__name__}",
                              file=sys.stderr, flush=True)
                if cleanup_failures and construction_complete:
                    raise RuntimeError("Directory construction completed, but owned process cleanup failed; "
                                       "GPU release is unconfirmed. Inspect this job's logs before restarting.") from cleanup_failures[0][1]
    finally:
        for signum, handler in previous_signals.items():
            signal.signal(signum, handler)
    return job


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repair-fallbacks", action="store_true",
                        help="Repair existing fallback sessions only; do not enqueue unfinished sessions")
    version = parser.add_mutually_exclusive_group()
    version.add_argument("--m-eval", action="store_true",
                         help="Build M validation/test using the compact boundary and one-block context recipe")
    version.add_argument("--topic-event", action="store_true",
                         help="Build with the topic/event v2 boundary prompt in an independent output root")
    version.add_argument("--segment-context", action="store_true",
                         help="Rebuild LoCoMo using topic/event v2 boundaries and three preceding raw segments")
    parser.add_argument("--scope-completion", action="store_true",
                        help="Verify selected LongMemEval splits independently of other pending journal splits")
    parser.add_argument("--preserve-existing", action="store_true",
                        help="Preserve verified legacy directories with source provenance; build only missing "
                             "sessions using topic/event v2, without importing old boundary caches")
    parser.add_argument("--reuse-root", help="Existing adjacent build to rebind using an audited resplit mapping")
    parser.add_argument("--reuse-mapping", help="row_mapping.jsonl produced by longmemeval_resplit")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--datasets", nargs="+", choices=["locomo", "split_longmemeval_s", "split_longmemeval_m"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"])
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--output-root")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--base-urls", nargs="+", help="Reuse running Writers together with --embedding-base-url")
    parser.add_argument("--embedding-base-url", help="Reuse an embedding service together with --base-urls")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Per Writer GPU")
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096, help="Per Writer GPU")
    parser.add_argument("--gpu-memory-utilization", type=float, default=.90, help="Per Writer GPU")
    parser.add_argument("--embedding-max-model-len", type=int, default=8192)
    parser.add_argument("--embedding-max-num-seqs", type=int, default=64)
    parser.add_argument("--embedding-max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--embedding-gpu-memory-utilization", type=float, default=.25)
    parser.add_argument("--job-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    launch(argument_parser().parse_args())


if __name__ == "__main__":
    main()
