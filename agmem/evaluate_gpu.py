"""Launch checkpoint evaluation in an existing GPU allocation.

With parallel directory building, position 0 builds the other datasets while
positions 1/2 run the foreground Reader and position 3 serves Task/Judge.
Single-dataset and staged runs leave position 0 available for other work.
Sequential evaluation also supports three GPUs: positions 0/1 run the Reader
and position 2 serves Task/Judge, with no background directory service.
The judge-remote stage also supports a dedicated single-GPU allocation.
Only processes created by this invocation are stopped on exit.
"""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import argparse
import os
import signal
import subprocess
import sys
import time

from .evaluate_checkpoints import checkpoint_model, evaluation_settings, load_suite, validate_selections
from .core import atomic_json
from .four_gpu import free_port, gpu_roles, server_limits, stop_group, wait_ready
from .services import resolve_task_service


def evaluation_command(args, local_base_url=None):
    command = [sys.executable, "-u", "-m", "agmem.evaluate_checkpoints",
               "--config", args.config, "--stage", args.stage]
    for name in ("datasets", "checkpoints", "task_agents"):
        values = getattr(args, name)
        if values:
            command.extend(["--" + name.replace("_", "-"), *values])
    for name in ("remote_model", "limit", "responses_root"):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if local_base_url:
        command.extend(["--local-base-url", local_base_url])
    return command


def parallel_directories_enabled(args, suite, datasets):
    enabled = getattr(args, "parallel_directories", None)
    if enabled is None:
        enabled = suite.get("parallel_directory_build", False)
    if type(enabled) is not bool:
        raise ValueError("parallel_directory_build must be a boolean")
    return enabled and args.stage == "all" and len(datasets) > 1


def evaluation_gpu_roles(stage, visible):
    """Preserve four-card/one-card judging; allow three-card sequential eval."""
    ids = [part.strip() for part in visible.split(",")]
    if stage == "judge-remote":
        if len(ids) == 1 and ids[0] not in {"", "-1", "NoDevFiles"}:
            return {"writer": "", "reader": "", "task": ids[0]}
        if len(ids) != 4 or any(not part for part in ids) or len(set(ids)) != 4:
            raise ValueError("judge-remote requires one or four allocated GPU IDs in "
                             f"CUDA_VISIBLE_DEVICES; got {visible!r}. "
                             "Use scripts/judge_locomo_gpt.sbatch for a single-GPU job, "
                             "or --local-base-url to reuse a running Judge service.")
    elif len(ids) == 3:
        if any(part in {"", "-1", "NoDevFiles"} for part in ids):
            raise ValueError("Expected three distinct allocated GPU IDs in CUDA_VISIBLE_DEVICES; "
                             f"got {visible!r}")
        return {"writer": "", **gpu_roles(visible, train_only=True)}
    return gpu_roles(visible)


def run_parallel_evaluation(args, datasets, roles, job, start, stop_owned, start_qwen):
    """Evaluate LoCoMo now; build the remaining stores on an independent GPU.

    The background worker owns only its datasets. It releases their source
    locks before the next evaluation starts. A background failure is reported
    immediately but does not interrupt the foreground evaluation.
    """
    foreground = "locomo" if "locomo" in datasets else datasets[0]
    background = [source for source in datasets if source != foreground]
    atomic_json(job / "schedule.json", {
        "mode": "parallel_directory_build", "gpu_roles": roles,
        "foreground_dataset": foreground, "background_datasets": background,
        "evaluation_order": [foreground, *background],
    })
    errors, statuses = [], {}

    def status(name, value):
        statuses[name] = value
        atomic_json(job / "progress.json", statuses)

    def failure(message):
        errors.append(message)
        print(message, file=sys.stderr, flush=True)

    def retire(process, name):
        try:
            stop_owned(process)
        except Exception as exc:
            # The launcher still owns the group and retries cleanup on exit.
            # A cleanup failure on one GPU must not cancel independent work.
            failure(f"Cleanup failed for {name} ({type(exc).__name__}); independent work continues "
                    "and cleanup will be retried on exit.")
            return False
        return True

    def evaluate(source, endpoint):
        child = deepcopy(args)
        child.datasets, child.stage = [source], "all"
        worker = start(f"eval_{source}", evaluation_command(child, endpoint), roles["reader"],
                       {"AGMEM_READER_MODEL_PARALLEL_SIZE": "2", "OMP_NUM_THREADS": "8"})
        status(source, "evaluating")
        print(f"Evaluating {source}; progress: {job.resolve() / f'eval_{source}.log'}", flush=True)
        return worker

    task_service, endpoint = None, args.local_base_url
    if endpoint is None:
        task_service, endpoint = start_qwen("qwen_vllm", "task")
    # Start foreground evaluation before waiting for the independent Writer to
    # initialize. Slow directory generation never gates this process.
    active_source = foreground
    evaluator = evaluate(foreground, endpoint)
    builder = writer_service = None
    build_state, task_available = "starting", True
    pending = list(background)
    try:
        writer_service, writer_endpoint = start_qwen("writer_vllm", "writer")
        child = deepcopy(args)
        child.datasets, child.stage = background, "build"
        builder = start("directories", evaluation_command(child, writer_endpoint), "")
        build_state = "building"
        status("directory_build", build_state)
        print(f"Building directories for {', '.join(background)} on GPU {roles['writer']}; "
              f"progress: {job.resolve() / 'directories.log'}", flush=True)
    except Exception as exc:
        build_state, pending = "failed", []
        status("directory_build", build_state)
        failure(f"Background directory startup failed ({type(exc).__name__}); "
                f"{foreground} evaluation continues. Inspect {job / 'writer_vllm.log'}")
        if writer_service is not None:
            retire(writer_service, "directory Writer")
            writer_service = None

    announced_wait = False
    while evaluator is not None or builder is not None or pending:
        if builder is not None:
            result = builder.poll()
            server_exit = writer_service.poll() if result is None else None
            if result is not None or server_exit is not None:
                build_state = "ready" if result == 0 else "failed"
                status("directory_build", build_state)
                if build_state == "failed":
                    failure(f"Background directory build failed (worker={result}, server={server_exit}); "
                            f"the current evaluation continues. Inspect {job / 'directories.log'} "
                            f"and {job / 'writer_vllm.log'}")
                    pending.clear()
                else:
                    print("Background test directories are ready; saved progress is reusable.", flush=True)
                builder_stopped = retire(builder, "directory worker")
                writer_stopped = retire(writer_service, "directory Writer")
                if not builder_stopped or not writer_stopped:
                    # Do not hand a potentially still-owned store to a new worker.
                    build_state = "failed"
                    status("directory_build", "cleanup_failed")
                    pending.clear()
                builder = writer_service = None

        if evaluator is not None:
            result = evaluator.poll()
            if result is not None:
                status(active_source, "complete" if result == 0 else "failed")
                if result:
                    failure(f"Evaluation failed for {active_source} ({result}); "
                            f"inspect {job / f'eval_{active_source}.log'}")
                else:
                    print(f"Evaluation complete: {active_source}", flush=True)
                if not retire(evaluator, f"{active_source} evaluation"):
                    task_available = False
                    pending.clear()
                evaluator = None

        if task_service is not None and task_available and task_service.poll() is not None:
            task_available = False
            failure(f"Task/Judge service exited; directory construction continues. "
                    f"Inspect {job / 'qwen_vllm.log'}")
            if evaluator is not None:
                status(active_source, "failed")
                retire(evaluator, f"{active_source} evaluation")
                evaluator = None
            pending.clear()

        if evaluator is None and pending and build_state == "ready" and task_available:
            active_source = pending.pop(0)
            evaluator = evaluate(active_source, endpoint)
        elif evaluator is None and builder is not None and not announced_wait:
            print("Foreground evaluation has ended; waiting for the background directory build. "
                  "Completed evaluation results are already saved.", flush=True)
            announced_wait = True

        if evaluator is not None or builder is not None or pending:
            time.sleep(2)

    if errors:
        raise RuntimeError("; ".join(errors))
    print(f"Evaluation finished; logs: {job.resolve()}", flush=True)
    return job


def launch(args):
    suite, base, _ = load_suite(args.config)
    settings = evaluation_settings(suite)
    datasets = args.datasets or suite["datasets"]
    checkpoints = args.checkpoints or list(suite["checkpoints"])
    task_agents = args.task_agents or suite["task_agents"]
    validate_selections(suite, datasets, checkpoints, task_agents)
    parallel_directories = parallel_directories_enabled(args, suite, datasets)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.stage in {"all", "score"} and "remote" in task_agents:
        remote = deepcopy(base)
        remote["task_agent"]["service"] = deepcopy(suite["remote_task"])
        if args.remote_model:
            remote["task_agent"]["service"].update(model=args.remote_model, model_revision=args.remote_model)
        resolve_task_service(remote)  # Missing credentials fail before starting any GPU process.

    needs_reader = args.stage in {"all", "retrieve"}
    needs_service = args.stage in {"all", "build", "score", "judge-remote"}
    start_service = needs_service and not args.local_base_url
    roles = evaluation_gpu_roles(args.stage, os.environ.get("CUDA_VISIBLE_DEVICES", "")) if needs_reader or start_service else None
    if parallel_directories and roles is not None and not roles["writer"]:
        raise ValueError("Parallel directory building requires four allocated GPUs. "
                         "Use --no-parallel-directories (or parallel_directory_build: false) "
                         "for sequential evaluation on three GPUs.")
    if "required_prepared_sources" in suite and args.stage in {"all", "retrieve", "score", "judge-remote"}:
        # Strict suites consume existing directories. Validate every selected
        # checkpoint and store before reserving a service or creating job logs.
        # The CPU prepare worker retains the evaluator's source locks and full
        # provenance checks; it does not generate missing directories.
        for label in checkpoints:
            checkpoint_model(suite, label)
        preflight = deepcopy(args)
        preflight.stage = "prepare"
        cpu_environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1",
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        cpu_environment.pop("AGMEM_READER_MODEL_PARALLEL_SIZE", None)
        print("Checking required test directories and checkpoints before GPU startup...", flush=True)
        subprocess.run(evaluation_command(preflight), env=cpu_environment, check=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    job = Path(args.job_dir) if args.job_dir else Path(suite["output_root"]).parent / "jobs" / f"{stamp}-{os.getpid()}"
    job.mkdir(parents=True, exist_ok=False)
    print(f"Evaluation logs: {job.resolve()}", flush=True)
    print(f"Evaluation concurrency: Reader queries={settings['reader_query_concurrency']}, "
          f"Reader batch={settings['reader_batch_size']}, "
          f"Task/Judge workers local={settings['local_scoring_workers']}, "
          f"remote={settings['remote_scoring_workers']}", flush=True)
    if roles:
        background = (f"{'directory Writer GPU' if parallel_directories else 'reserved background GPU'}: "
                      f"{roles['writer']}" if roles["writer"] else "no reserved background GPU")
        print(f"Reader GPUs: {roles['reader'] if needs_reader else 'unused'}; "
              f"Task/Judge Qwen GPU: {roles['task'] if start_service else 'external/unused'}; "
              f"{background}", flush=True)
    if not parallel_directories:
        atomic_json(job / "schedule.json", {"mode": "sequential_evaluation", "gpu_roles": roles,
                    "datasets": datasets, "checkpoints": checkpoints, "task_agents": task_agents})
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"}
    environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    processes = []
    previous_handlers = {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    with ExitStack() as cleanup:
        def start(name, command, devices, extra=None):
            stream = cleanup.enter_context((job / f"{name}.log").open("w"))
            child_environment = {**environment, "CUDA_VISIBLE_DEVICES": devices}
            child_environment.pop("AGMEM_READER_MODEL_PARALLEL_SIZE", None)
            child_environment.update(extra or {})
            process = subprocess.Popen(command, env=child_environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            return process

        def stop_owned(process):
            stop_group(process)
            processes.remove(process)

        def start_qwen(name, role):
            port = free_port()
            limits = server_limits(role, base)
            command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                       "--model", base["models"]["task_path"],
                       "--served-model-name", base["service"]["model"],
                       "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
                       "--tensor-parallel-size", "1", "--max-model-len",
                       str(base["task_agent"].get("max_context_tokens") or 65536),
                       "--max-num-seqs", str(limits["max_num_seqs"]),
                       "--max-num-batched-tokens", str(limits["max_num_batched_tokens"]),
                       "--gpu-memory-utilization", str(limits["gpu_memory_utilization"]),
                       "--enable-chunked-prefill", "--language-model-only", "--enforce-eager"]
            # The readiness probe expects the training service alias.
            if base["service"]["model"] != "agmem-27b":
                raise ValueError("Managed local startup expects the training service alias agmem-27b; "
                                 "use --local-base-url for another deployment")
            server = start(name, command, roles[role])
            try:
                wait_ready(server, port, job / f"{name}.log")
            except BaseException:
                stop_owned(server)
                raise
            endpoint = f"http://127.0.0.1:{port}/v1"
            print(f"{role.capitalize()} Qwen service ready: {endpoint}", flush=True)
            return server, endpoint

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, interrupted)
            if parallel_directories:
                return run_parallel_evaluation(args, datasets, roles, job, start, stop_owned, start_qwen)
            service, endpoint = None, args.local_base_url
            if start_service:
                service, endpoint = start_qwen("qwen_vllm", "task")
            worker = start("eval", evaluation_command(args, endpoint),
                           roles["reader"] if needs_reader else "",
                           {"AGMEM_READER_MODEL_PARALLEL_SIZE": "2", "OMP_NUM_THREADS": "8"}
                           if needs_reader else None)
            print(f"Evaluation started; progress: {job.resolve() / 'eval.log'}", flush=True)
            while True:
                status = worker.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f"Evaluation failed ({status}); inspect {job / 'eval.log'}")
                    print(f"Evaluation finished; logs: {job.resolve()}", flush=True)
                    return job
                if service is not None and service.poll() is not None:
                    raise RuntimeError(f"Shared Qwen service exited; inspect {job / 'qwen_vllm.log'}")
                time.sleep(2)
        finally:
            already_failed = sys.exc_info()[0] is not None
            cleanup_errors = []
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
            try:
                # Even exited leaders may have subprocesses in their owned group.
                for process in reversed(processes):
                    try:
                        stop_group(process)
                    except Exception as exc:
                        cleanup_errors.append(exc)
                        print(f"Owned process cleanup failed ({type(exc).__name__}); "
                              "continuing cleanup of the other owned processes.", file=sys.stderr, flush=True)
            finally:
                for signum, previous in previous_handlers.items():
                    signal.signal(signum, previous)
            if cleanup_errors and not already_failed:
                raise RuntimeError("Evaluation finished but an owned process group could not be cleaned up") from cleanup_errors[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_locomo_longmemeval_checkpoints.yaml")
    parser.add_argument("--stage", choices=["all", "prepare", "build", "retrieve", "score", "summarize",
                                           "export-remote", "judge-remote"], default="all")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--task-agents", nargs="+", choices=["local", "remote"])
    parser.add_argument("--local-base-url", help="Reuse this local Qwen service instead of starting one")
    parser.add_argument("--remote-model")
    parser.add_argument("--responses-root", help="Returned GPT answers under ROOT/checkpoint/dataset; "
                        "defaults to each evaluation's offline/responses directory")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--job-dir", help="New directory for this invocation's process logs")
    parser.add_argument("--parallel-directories", action=argparse.BooleanOptionalAction, default=None,
                        help="Use the first GPU to build other test directories while evaluating LoCoMo first "
                             "(--stage all, multiple datasets); otherwise follow the suite setting")
    args = parser.parse_args()
    launch(args)


if __name__ == "__main__":
    main()
