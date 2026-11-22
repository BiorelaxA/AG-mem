"""CPU-only continuation deployment checks; no model, API, or GPU is opened."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from agmem import build_qwen_continued_directories_gpu as launcher


ROOT = Path(__file__).resolve().parents[1]


class Process:
    next_pid = 90000

    def __init__(self, kind, worker_code=0, service_failure=None):
        self.kind = kind
        self.pid = Process.next_pid
        Process.next_pid += 1
        self.returncode = worker_code if kind == "worker" else service_failure

    def poll(self):
        return self.returncode


class ContinuationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, **changes):
        args = launcher.argument_parser().parse_args([])
        args.job_dir = str(self.root / "job")
        for name, value in changes.items():
            setattr(args, name, value)
        return args

    def run_launch(self, args=None, *, embedding_error=None, writer_error=None,
                   worker_code=0, import_error=None, dry_error=None, busy_error=None,
                   start_error=None, cleanup_error=False, service_failure=None, environment=None):
        calls, stopped, events = [], [], []

        def run(command, **kwargs):
            kind = "dry" if "--dry-run" in command else "import" if "--import-only" in command else "external"
            events.append(kind)
            if kind == "dry" and dry_error:
                raise dry_error
            if kind == "import" and import_error:
                raise import_error
            return SimpleNamespace(returncode=0)

        def start(command, **kwargs):
            kind = ("embedding" if "pooling" in command else "writer"
                    if "vllm.entrypoints.openai.api_server" in command else "worker")
            if kind == start_error:
                raise OSError("spawn failed")
            process = Process(kind, worker_code, 1 if kind == service_failure else None)
            calls.append((kind, command, kwargs, process))
            events.append(kind)
            return process

        def stop(process):
            stopped.append(process)
            if cleanup_error and process.kind == "worker":
                raise RuntimeError("cleanup failed")

        def idle(*args):
            events.append("idle")
            if busy_error:
                raise busy_error

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"DEEPSEEK_API_KEY": "not-used", **(environment or {})}, clear=True))
            load = stack.enter_context(patch.object(launcher, "load_recipe", return_value={
                "source_root": "source", "output_root": "target",
                "service": {"model": "agmem-27b"},
                "writer": {"context_tokens": 65536},
                "embedding": {"workers": 8, "batch_size": 32, "expected_dimensions": 2560},
            }))
            models = stack.enter_context(patch.object(launcher, "_models", return_value=("/writer", "/embedding")))
            cpu = stack.enter_context(patch.object(launcher.subprocess, "run", side_effect=run))
            idle_mock = stack.enter_context(patch.object(launcher, "ensure_gpus_idle", side_effect=idle))
            stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=start))
            stack.enter_context(patch.object(launcher, "free_port", side_effect=[10100, 10101]))
            embed_wait = stack.enter_context(patch.object(launcher, "wait_embedding_ready", side_effect=embedding_error))
            writer_wait = stack.enter_context(patch.object(launcher, "wait_ready", side_effect=writer_error))
            stack.enter_context(patch.object(launcher, "stop_group", side_effect=stop))
            signals = stack.enter_context(patch.object(launcher.signal, "signal"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                result = launcher.launch(args or self.args())
            except BaseException as exc:
                result = exc
        return SimpleNamespace(calls=calls, stopped=stopped, events=events, result=result,
                               cpu=cpu, idle=idle_mock, models=models, load=load,
                               embed_wait=embed_wait, writer_wait=writer_wait, signals=signals)

    def test_import_finishes_before_gpu_startup_and_worker_is_cpu_only(self):
        run = self.run_launch(self.args(source_root="/source build"))
        self.assertIsInstance(run.result, Path)
        self.assertEqual(run.events, ["dry", "import", "idle", "embedding", "writer", "worker"])
        self.assertEqual([(kind, opts["env"]["CUDA_VISIBLE_DEVICES"]) for kind, _, opts, _ in run.calls],
                         [("embedding", "4"), ("writer", "5"), ("worker", "")])
        for call in run.cpu.call_args_list:
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertIn("agmem.build_qwen_continued_directories", call.args[0])
            self.assertEqual(call.args[0][call.args[0].index("--source-root") + 1], "/source build")
        for _, _, options, _ in run.calls:
            self.assertNotIn("DEEPSEEK_API_KEY", options["env"])
            self.assertTrue(options["start_new_session"])
        embedding, writer, worker = [row[1] for row in run.calls]
        for flag, value in {"--max-num-seqs": "256", "--max-num-batched-tokens": "8192",
                            "--gpu-memory-utilization": "0.9", "--max-model-len": "65536"}.items():
            self.assertEqual(writer[writer.index(flag) + 1], value)
        for flag, value in {"--max-num-seqs": "64", "--max-num-batched-tokens": "32768",
                            "--gpu-memory-utilization": "0.35"}.items():
            self.assertEqual(embedding[embedding.index(flag) + 1], value)
        self.assertEqual(worker[worker.index("--workers") + 1], "256")
        self.assertEqual(worker[worker.index("--base-urls") + 1], "http://127.0.0.1:10101/v1")
        self.assertEqual(worker[worker.index("--embedding-base-url") + 1], "http://127.0.0.1:10100/v1")
        self.assertEqual(run.stopped, [row[3] for row in reversed(run.calls)])
        self.assertEqual(run.signals.call_count, 4)
        deployment = json.loads((run.result / "deployment.json").read_text())
        self.assertEqual(deployment["writer"]["max_num_seqs"], 256)
        self.assertEqual(deployment["source_root"], "/source build")
        self.assertEqual(deployment["embedding"]["gpu"], "4")
        self.assertEqual(deployment["writer"]["gpu"], "5")

    def test_explicit_allocation_and_invalid_allocations(self):
        run = self.run_launch(environment={"AGMEM_BUILD_GPUS": "6,7", "CUDA_VISIBLE_DEVICES": "4,5"})
        self.assertEqual([row[2]["env"]["CUDA_VISIBLE_DEVICES"] for row in run.calls[:2]], ["6", "7"])
        for visible in ("", "4", "4,4", "4,,5", "-1,5", "NoDevFiles,5", "0,1,2"):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                launcher.allocated_devices(visible)

    def test_cpu_modes_need_no_models_gpus_credentials_or_job_directory(self):
        for mode in ("dry_run", "import_only"):
            with self.subTest(mode=mode):
                run = self.run_launch(self.args(**{mode: True, "limit_sessions": 2,
                                                  "output_root": "/tmp/new output"}))
                self.assertIsNone(run.result)
                self.assertEqual(run.calls, [])
                run.load.assert_not_called()
                run.models.assert_not_called()
                run.idle.assert_not_called()
                run.cpu.assert_called_once()
                command = run.cpu.call_args.args[0]
                self.assertIn("--" + mode.replace("_", "-"), command)
                self.assertEqual(command[command.index("--limit-sessions") + 1], "2")
                self.assertEqual(command[command.index("--output-root") + 1], "/tmp/new output")
                self.assertFalse((self.root / "job").exists())

    def test_source_dry_run_or_import_failure_prevents_any_gpu_process(self):
        for error in ("dry_error", "import_error"):
            with self.subTest(error=error):
                run = self.run_launch(**{error: RuntimeError("source mismatch")})
                self.assertIsInstance(run.result, RuntimeError)
                self.assertEqual(run.calls, [])
                self.assertEqual(run.stopped, [])
                run.idle.assert_not_called()
                self.assertFalse((self.root / "job").exists())

    def test_existing_endpoints_import_before_cpu_worker_without_managing_services(self):
        run = self.run_launch(self.args(base_url="http://writer/v1", embedding_base_url="http://embedding/v1"))
        self.assertIsNone(run.result)
        self.assertEqual(run.events, ["dry", "import", "external"])
        run.models.assert_not_called()
        run.idle.assert_not_called()
        self.assertEqual(run.calls, [])
        self.assertEqual(run.stopped, [])
        self.assertIn("http://writer/v1", run.cpu.call_args.args[0])
        self.assertEqual(run.cpu.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_capacity_and_endpoint_pairing_fail_before_any_subprocess(self):
        for changes in ({"workers": 257}, {"max_num_seqs": 128}, {"base_urls": ["http://writer/v1"]},
                        {"embedding_base_url": "http://embedding/v1"}):
            with self.subTest(changes=changes):
                run = self.run_launch(self.args(**changes))
                self.assertIsInstance(run.result, ValueError)
                run.cpu.assert_not_called()
                self.assertEqual(run.calls, [])

    def test_busy_gpu_never_starts_or_stops_a_process(self):
        run = self.run_launch(busy_error=RuntimeError("busy"))
        self.assertIsInstance(run.result, RuntimeError)
        self.assertEqual(run.calls, [])
        self.assertEqual(run.stopped, [])

    def test_startup_failures_clean_up_only_successfully_started_owned_processes(self):
        for kwargs, expected in (({"embedding_error": RuntimeError("embedding failed")}, ["embedding"]),
                                 ({"writer_error": RuntimeError("writer failed")}, ["embedding", "writer"]),
                                 ({"start_error": "writer"}, ["embedding"]),
                                 ({"start_error": "worker"}, ["embedding", "writer"])):
            with self.subTest(kwargs=kwargs):
                args = self.args(job_dir=str(self.root / str(len(list(self.root.iterdir())))))
                run = self.run_launch(args, **kwargs)
                self.assertIsInstance(run.result, (RuntimeError, OSError))
                self.assertEqual([row[0] for row in run.calls], expected)
                self.assertEqual(run.stopped, [row[3] for row in reversed(run.calls)])
                self.assertEqual(run.signals.call_count, 4)

    def test_worker_and_service_failures_clean_up_owned_processes(self):
        for kwargs in ({"worker_code": 3}, {"worker_code": None, "service_failure": "writer"}):
            with self.subTest(kwargs=kwargs):
                args = self.args(job_dir=str(self.root / str(len(list(self.root.iterdir())))))
                run = self.run_launch(args, **kwargs)
                self.assertIsInstance(run.result, RuntimeError)
                self.assertEqual(run.stopped, [row[3] for row in reversed(run.calls)])

    def test_interruption_cleans_up_owned_processes_and_restores_signal_handlers(self):
        run = self.run_launch(writer_error=SystemExit(143))
        self.assertIsInstance(run.result, SystemExit)
        self.assertEqual(run.result.code, 143)
        self.assertEqual([row[0] for row in run.calls], ["embedding", "writer"])
        self.assertEqual(run.stopped, [row[3] for row in reversed(run.calls)])
        self.assertEqual(run.signals.call_count, 4)

    def test_cleanup_failure_does_not_skip_other_owned_processes(self):
        with patch("sys.stderr", new=io.StringIO()):
            run = self.run_launch(cleanup_error=True)
        self.assertIsInstance(run.result, RuntimeError)
        self.assertIn("GPU release is unconfirmed", str(run.result))
        self.assertEqual(run.stopped, [row[3] for row in reversed(run.calls)])

    def test_idle_check_rejects_existing_processes_and_does_not_stop_them(self):
        outputs = [SimpleNamespace(stdout="GPU-a, 80000, 79000\nGPU-b, 80000, 79000\n"),
                   SimpleNamespace(stdout="GPU-other, 11\nGPU-b, 12\n")]
        with patch.object(launcher.subprocess, "run", side_effect=outputs), \
                patch.object(launcher, "stop_group") as stop, \
                self.assertRaisesRegex(RuntimeError, "busy.*left running"):
            launcher.ensure_gpus_idle(["4", "5"], [.35, .90])
        stop.assert_not_called()

    def test_idle_check_requires_enough_memory_for_each_role_and_fails_closed(self):
        for text in ("GPU-a, 80000, 10000\nGPU-b, 80000, 79000\n",
                     "GPU-a, 80000, 79000\nGPU-b, 80000, 70000\n",
                     "GPU-a, 80000, 79000\nGPU-b, 40000, 39000\n"):
            with self.subTest(text=text), patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(stdout=text)), \
                    self.assertRaises(RuntimeError):
                launcher.ensure_gpus_idle(["4", "5"], [.35, .90])
        with patch.object(launcher.subprocess, "run", side_effect=FileNotFoundError), \
                self.assertRaisesRegex(RuntimeError, "no services started"):
            launcher.ensure_gpus_idle(["4", "5"], [.35, .90])


class ContinuationRecipeAndScriptTests(unittest.TestCase):
    def test_recipe_keeps_source_semantics_and_local_writer_concurrency(self):
        source = yaml.safe_load((ROOT / "configs/build_longmemeval_m_deepseek.yaml").read_text())
        target = yaml.safe_load((ROOT / launcher.DEFAULT_CONFIG).read_text())
        for key in ("datasets", "embedding", "segmentation", "writer", "model_path", "retry_fallback"):
            self.assertEqual(source[key], target[key], key)
        self.assertEqual(target["protocol"], "adjacent_topic_event_qwen_continuation_v1")
        self.assertEqual(target["source_root"], source["output_root"])
        self.assertNotEqual(target["output_root"], source["output_root"])
        self.assertEqual(target["workers"], 256)
        self.assertFalse(target["preserve_existing_directories"])
        self.assertEqual(target["service"]["model"], "agmem-27b")
        self.assertEqual(target["service"]["timeout_seconds"], 1200)
        self.assertEqual(target["service"]["network_retries"], 3)
        self.assertNotIn("DEEPSEEK_API_KEY", json.dumps(target))

    def test_shell_defaults_overrides_and_argument_forwarding_without_running_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "python"
            executable.write_text(f"#!{sys.executable}\n" + "import json, os, sys\n"
                "print(json.dumps({'args': sys.argv[1:], 'gpus': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n")
            executable.chmod(0o755)
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                   "AGMEM_PROJECT_DIR": str(ROOT)}
            for key in ("CUDA_VISIBLE_DEVICES", "AGMEM_BUILD_GPUS", "AGMEM_WRITER_WORKERS",
                        "AGMEM_WRITER_MAX_NUM_SEQS", "AGMEM_EMBEDDING_MAX_NUM_SEQS"):
                env.pop(key, None)
            for override in (False, True):
                if override:
                    env.update(AGMEM_BUILD_GPUS="6,7", AGMEM_WRITER_WORKERS="192",
                               AGMEM_WRITER_MAX_NUM_SEQS="192", AGMEM_EMBEDDING_MAX_NUM_SEQS="80")
                result = subprocess.run(["bash", str(ROOT / "scripts/build_longmemeval_m_qwen_continue.sh"),
                    "--dry-run", "--source-root", "/tmp/source build", "--output-root", "/tmp/new output"],
                    check=True, env=env, capture_output=True, text=True)
                row = json.loads(result.stdout)
                args = row["args"]
                self.assertEqual(args[2], "agmem.build_qwen_continued_directories_gpu")
                self.assertEqual(row["gpus"], "6,7" if override else "4,5")
                self.assertEqual(args[args.index("--workers") + 1], "192" if override else "256")
                self.assertEqual(args[args.index("--max-num-seqs") + 1], "192" if override else "256")
                self.assertEqual(args[args.index("--embedding-max-num-seqs") + 1], "80" if override else "64")
                self.assertEqual(args[-5:], ["--dry-run", "--source-root", "/tmp/source build",
                                            "--output-root", "/tmp/new output"])


if __name__ == "__main__":
    unittest.main()
