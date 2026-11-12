"""GPU process ownership and deployment routing without loading model weights."""

from argparse import Namespace
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agmem.build_local_directories_gpu import allocated_devices, build_command, launch


class Process:
    counter = 1000

    def __init__(self, kind, code=0):
        Process.counter += 1
        self.pid, self.kind = Process.counter, kind
        self.returncode = None if kind == "server" else code

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


class LocalDirectoryLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, **changes):
        values = dict(config="recipe.yaml", workers=48, datasets=None, splits=None,
                      dry_run=False, base_url=None, base_urls=None, max_num_seqs=16,
                      max_num_batched_tokens=4096, gpu_memory_utilization=.90,
                      job_dir=str(self.root / "job"))
        return Namespace(**{**values, **changes})

    def run_launch(self, args, *, preflight_code=0, worker_code=0, ready_error=None,
                   cleanup_error=None):
        calls, stopped = [], []

        def start(command, **kwargs):
            kind = ("server" if "vllm.entrypoints.openai.api_server" in command else
                    "worker" if any(module in command for module in
                                    ("agmem.build_parent_child_directories", "agmem.build_local_directories"))
                    else "preflight")
            process = Process(kind, preflight_code if kind == "preflight" else worker_code)
            calls.append((kind, command, kwargs, process))
            return process

        def stop(process):
            stopped.append(process)
            if cleanup_error is not None and process.kind == "server":
                raise cleanup_error

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,4,6"}, clear=True))
            stack.enter_context(patch("agmem.build_local_directories_gpu.model_path", return_value="/qwen27b"))
            stack.enter_context(patch("agmem.build_local_directories_gpu.subprocess.run"))
            stack.enter_context(patch("agmem.build_local_directories_gpu.subprocess.Popen", side_effect=start))
            stack.enter_context(patch("agmem.build_local_directories_gpu.free_port", side_effect=[10100, 10101, 10102]))
            stack.enter_context(patch("agmem.build_local_directories_gpu.wait_ready", side_effect=ready_error))
            stack.enter_context(patch("agmem.build_local_directories_gpu.stop_group", side_effect=stop))
            stack.enter_context(patch("agmem.build_local_directories_gpu.signal.signal"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            try:
                result = launch(args)
            except BaseException as exc:
                result = exc
        return calls, stopped, result

    def test_three_independent_replicas_and_single_cpu_coordinator(self):
        calls, stopped, result = self.run_launch(self.args())
        self.assertIsInstance(result, Path)
        servers = [c for c in calls if c[0] == "server"]
        self.assertEqual([c[2]["env"]["CUDA_VISIBLE_DEVICES"] for c in servers], ["2", "4", "6"])
        for _, command, options, _ in servers:
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
            self.assertEqual(command[command.index("--max-num-seqs") + 1], "16")
            self.assertTrue(options["start_new_session"])
            self.assertEqual(options["env"]["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        worker = next(c for c in calls if c[0] == "worker")
        self.assertEqual(worker[2]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(worker[1][-4:], ["--base-urls", "http://127.0.0.1:10100/v1",
                                        "http://127.0.0.1:10101/v1", "http://127.0.0.1:10102/v1"])
        self.assertEqual(stopped, [c[3] for c in reversed(calls)])

    def test_occupied_or_broken_gpu_stops_before_model_start(self):
        calls, stopped, result = self.run_launch(self.args(), preflight_code=1)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("preflight", str(result))
        self.assertEqual([c[0] for c in calls], ["preflight"])
        self.assertEqual(len(stopped), 1)

    def test_flat_811_build_uses_three_writers_and_forwards_test_selection(self):
        args = self.args(legacy_flat=True, workers=192, max_num_seqs=64,
                         config="configs/build_longmemeval_811_flat_qwen.yaml",
                         datasets=["split_longmemeval_s", "split_longmemeval_m"], splits=["test"])
        calls, stopped, result = self.run_launch(args)
        self.assertIsInstance(result, Path)
        servers = [row for row in calls if row[0] == "server"]
        self.assertEqual(len(servers), 3)
        self.assertEqual([row[2]["env"]["CUDA_VISIBLE_DEVICES"] for row in servers], ["2", "4", "6"])
        for _, command, _, _ in servers:
            self.assertEqual(command[command.index("--max-num-seqs") + 1], "64")
        worker = next(row for row in calls if row[0] == "worker")
        self.assertIn("agmem.build_local_directories", worker[1])
        self.assertNotIn("agmem.build_parent_child_directories", worker[1])
        self.assertEqual(worker[1][worker[1].index("--workers") + 1], "192")
        self.assertEqual(worker[1][worker[1].index("--splits") + 1], "test")
        self.assertIn("split_longmemeval_s", worker[1])
        self.assertIn("split_longmemeval_m", worker[1])
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_readiness_failure_cleans_up_every_started_server(self):
        calls, stopped, result = self.run_launch(self.args(), ready_error=[None, RuntimeError("startup failed")])
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual([c[0] for c in calls], ["preflight", "server", "server"])
        self.assertEqual(stopped, [c[3] for c in reversed(calls)])

    def test_builder_failure_keeps_logs_and_cleans_up_owned_servers(self):
        calls, stopped, result = self.run_launch(self.args(), worker_code=1)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("Rerun to resume", str(result))
        self.assertTrue((self.root / "job" / "deployment.json").is_file())
        self.assertEqual(stopped, [c[3] for c in reversed(calls)])

    def test_cleanup_failure_prevents_success_after_completed_build(self):
        failure = RuntimeError("shutdown failed")
        calls, stopped, result = self.run_launch(self.args(), cleanup_error=failure)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("GPU release is unconfirmed", str(result))
        self.assertIs(result.__cause__, failure)
        self.assertEqual(stopped, [c[3] for c in reversed(calls)])
        for kind, _, _, process in calls:
            if kind == "server":
                self.assertIn(str(process.pid), str(result))

    def test_cleanup_failure_preserves_original_builder_failure(self):
        calls, stopped, result = self.run_launch(
            self.args(), worker_code=7, cleanup_error=RuntimeError("shutdown failed"))
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("Directory builder failed (7)", str(result))
        self.assertNotIn("GPU release is unconfirmed", str(result))
        self.assertEqual(stopped, [c[3] for c in reversed(calls)])

    def test_dry_run_and_existing_services_never_touch_cuda_or_deploy(self):
        for args in (self.args(dry_run=True), self.args(base_urls=["http://localhost:1234/v1"])):
            with self.subTest(args=args), patch.dict(os.environ, {}, clear=True), \
                    patch("agmem.build_local_directories_gpu.subprocess.run") as run, \
                    patch("agmem.build_local_directories_gpu.subprocess.Popen") as start, \
                    patch("agmem.build_local_directories_gpu.model_path") as model:
                launch(args)
                run.assert_called_once()
                start.assert_not_called()
                model.assert_not_called()
        self.assertFalse((self.root / "job").exists())

    def test_selections_forwarded_and_allocation_must_be_explicit(self):
        command = build_command(self.args(datasets=["split_longmemeval_s"], splits=["test"], dry_run=True))
        self.assertIn("split_longmemeval_s", command)
        self.assertEqual(command[-3:], ["--splits", "test", "--dry-run"])
        self.assertIn("agmem.build_parent_child_directories", command)
        self.assertIn("agmem.build_local_directories", build_command(self.args(legacy_flat=True)))
        self.assertEqual(allocated_devices("GPU-a,GPU-b,GPU-c"), ["GPU-a", "GPU-b", "GPU-c"])
        for visible in ("", "0", "0,1,2,3", "0,0,1", "0,1,-1"):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                allocated_devices(visible)


if __name__ == "__main__":
    unittest.main()
