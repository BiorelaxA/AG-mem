"""Four-GPU directory/evaluation scheduling without CUDA, models or HTTP."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agmem.evaluate_gpu import launch


class Process:
    def __init__(self, name, statuses, events, pid):
        self.name, self.statuses, self.events = name, iter(statuses), events
        self.pid, self.returncode = pid, None
        self.polls, self.reported_exit = 0, False

    def poll(self):
        self.polls += 1
        if self.polls > 100:
            raise AssertionError(f"Scheduler did not finish: {self.name}")
        self.returncode = next(self.statuses, self.returncode)
        if self.returncode is not None and not self.reported_exit:
            self.events.append(("exit", self.name, self.returncode))
            self.reported_exit = True
        return self.returncode


class ParallelEvaluationLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.suite = {
            "datasets": ["locomo", "split_longmemeval_s"],
            "checkpoints": {"best": "checkpoint-00000020", "final": "checkpoint-00000032"},
            "task_agents": ["local", "remote"],
            "parallel_directory_build": True,
            "output_root": str(self.root / "results"),
            "remote_task": {"model": "gpt-5-mini", "model_revision": "gpt-5-mini",
                            "request_profile": "openai_chat", "base_url_env": "OPENAI_BASE_URL",
                            "api_key_env": "OPENAI_API_KEY"},
        }
        self.base = {"models": {"task_path": "/models/qwen27b"},
                     "service": {"model": "agmem-27b"},
                     "task_agent": {"max_context_tokens": 65536},
                     "reward": {"likelihood": {"mode": "off"}}}

    def args(self, **changes):
        args = dict(config="suite.yaml", stage="all", datasets=None, checkpoints=None,
                    task_agents=["local"], local_base_url=None, remote_model=None,
                    limit=None, job_dir=str(self.root / "job"), parallel_directories=None)
        args.update(changes)
        return Namespace(**args)

    @staticmethod
    def values(command, flag):
        if flag not in command:
            return None
        values = []
        for value in command[command.index(flag) + 1:]:
            if value.startswith("--"):
                break
            values.append(value)
        return values

    def run_launch(self, args, *, plans=None, environment=None, interrupt=False,
                   readiness_failure=None, cleanup_failures_once=()):
        calls, processes, events, stopped = {}, {}, [], []
        failed_cleanups = set()
        defaults = {
            "directories": [None, None, 0],
            "eval_locomo": [None, 0],
            "eval_split_longmemeval_s": [None, 0],
            "eval": [0],
        }
        defaults.update(plans or {})

        def popen(command, **kwargs):
            name = Path(kwargs["stdout"].name).stem
            self.assertNotIn(name, calls, "Each worker owns its own log")
            self.assertTrue(kwargs["start_new_session"])
            calls[name] = (command, kwargs)
            events.append(("start", name))
            process = Process(name, defaults.get(name, [None]), events, 10000 + len(calls))
            processes[name] = process
            return process

        def stop(process):
            self.assertIs(processes[process.name], process)
            stopped.append(process.name)
            events.append(("stop", process.name))
            if process.name in cleanup_failures_once and process.name not in failed_cleanups:
                failed_cleanups.add(process.name)
                events.append(("stop_failed", process.name))
                raise RuntimeError(f"fixture cleanup failed: {process.name}")
            events.append(("stopped", process.name))

        def ready(process, port, path):
            events.append(("ready", process.name))
            if process.name == readiness_failure:
                raise RuntimeError(f"fixture startup failed: {process.name}")

        sleeps = 0

        def sleep(seconds):
            nonlocal sleeps
            sleeps += 1
            if interrupt:
                raise KeyboardInterrupt()
            if sleeps > 100:
                raise AssertionError("Scheduler waited indefinitely")

        with ExitStack() as stack:
            stack.enter_context(patch("agmem.evaluate_gpu.load_suite", return_value=(
                deepcopy(self.suite), deepcopy(self.base), {})))
            stack.enter_context(patch("agmem.evaluate_gpu.subprocess.Popen", side_effect=popen))
            stack.enter_context(patch("agmem.evaluate_gpu.stop_group", side_effect=stop))
            stack.enter_context(patch("agmem.evaluate_gpu.wait_ready", side_effect=ready))
            stack.enter_context(patch("agmem.evaluate_gpu.free_port", side_effect=iter((48123, 48124))))
            stack.enter_context(patch("agmem.evaluate_gpu.signal.signal"))
            stack.enter_context(patch("agmem.evaluate_gpu.time.sleep", side_effect=sleep))
            stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,4,6,7",
                                                       **(environment or {})}, clear=True))
            with redirect_stdout(io.StringIO()):
                try:
                    result = launch(args)
                except BaseException as exc:
                    result = exc
        return calls, processes, events, stopped, result

    def assert_finished_before(self, events, process, target):
        self.assertLess(events.index(("exit", process, 0)), events.index(("start", target)))

    def test_directory_writer_and_locomo_run_together_with_disjoint_gpu_roles(self):
        calls, _, events, stopped, result = self.run_launch(self.args())
        self.assertIsInstance(result, Path, repr(result))
        self.assertEqual(set(calls), {"qwen_vllm", "writer_vllm", "directories",
                                     "eval_locomo", "eval_split_longmemeval_s"})
        self.assertEqual(calls["writer_vllm"][1]["env"]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(calls["qwen_vllm"][1]["env"]["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(calls["directories"][1]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("AGMEM_READER_MODEL_PARALLEL_SIZE", calls["directories"][1]["env"])
        for worker in ("eval_locomo", "eval_split_longmemeval_s"):
            command, options = calls[worker]
            self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "4,6")
            self.assertEqual(options["env"]["AGMEM_READER_MODEL_PARALLEL_SIZE"], "2")
            self.assertEqual(self.values(command, "--stage"), ["all"])
        self.assertEqual(self.values(calls["directories"][0], "--stage"), ["build"])
        self.assertEqual(self.values(calls["directories"][0], "--datasets"), ["split_longmemeval_s"])
        self.assertEqual(self.values(calls["eval_locomo"][0], "--datasets"), ["locomo"])
        self.assertEqual(self.values(calls["eval_split_longmemeval_s"][0], "--datasets"), ["split_longmemeval_s"])

        writer_port = self.values(calls["writer_vllm"][0], "--port")[0]
        task_port = self.values(calls["qwen_vllm"][0], "--port")[0]
        self.assertNotEqual(writer_port, task_port)
        self.assertEqual(self.values(calls["directories"][0], "--local-base-url"),
                         [f"http://127.0.0.1:{writer_port}/v1"])
        for name in ("eval_locomo", "eval_split_longmemeval_s"):
            self.assertEqual(self.values(calls[name][0], "--local-base-url"),
                             [f"http://127.0.0.1:{task_port}/v1"])
        self.assertLess(events.index(("start", "eval_locomo")), events.index(("exit", "directories", 0)))
        self.assertLess(events.index(("start", "directories")), events.index(("exit", "eval_locomo", 0)))
        self.assert_finished_before(events, "directories", "eval_split_longmemeval_s")
        self.assert_finished_before(events, "eval_locomo", "eval_split_longmemeval_s")
        self.assertEqual(set(stopped), set(calls))

    def test_fast_background_completion_keeps_locomo_running_and_reader_exclusive(self):
        calls, _, events, _, result = self.run_launch(self.args(), plans={
            "directories": [0], "eval_locomo": [None, None, None, 0]})
        self.assertIsInstance(result, Path, repr(result))
        end = events.index(("exit", "eval_locomo", 0))
        self.assertNotIn(("stop", "eval_locomo"), events[:end])
        self.assertNotIn(("stop", "qwen_vllm"), events[:end])
        self.assert_finished_before(events, "eval_locomo", "eval_split_longmemeval_s")
        self.assertEqual(len(calls), 5)

    def test_completed_locomo_waits_for_background_before_longmemeval_evaluation(self):
        _, _, events, _, result = self.run_launch(self.args(), plans={
            "directories": [None, None, None, 0], "eval_locomo": [0]})
        self.assertIsInstance(result, Path, repr(result))
        self.assertLess(events.index(("exit", "eval_locomo", 0)), events.index(("exit", "directories", 0)))
        self.assert_finished_before(events, "directories", "eval_split_longmemeval_s")

    def test_background_failure_preserves_locomo_and_skips_unready_evaluation(self):
        calls, _, events, stopped, result = self.run_launch(self.args(), plans={
            "directories": [1], "eval_locomo": [None, None, 0]})
        self.assertIsInstance(result, RuntimeError, repr(result))
        self.assertNotIn("eval_split_longmemeval_s", calls)
        self.assertIn(("exit", "eval_locomo", 0), events)
        self.assertLess(events.index(("exit", "directories", 1)), events.index(("exit", "eval_locomo", 0)))
        end = events.index(("exit", "eval_locomo", 0))
        self.assertNotIn(("stop", "eval_locomo"), events[:end])
        self.assertNotIn(("stop", "qwen_vllm"), events[:end])
        self.assertEqual(set(stopped), set(calls))

    def test_writer_startup_failure_happens_after_locomo_starts_and_preserves_it(self):
        calls, _, events, stopped, result = self.run_launch(
            self.args(), readiness_failure="writer_vllm")
        self.assertIsInstance(result, RuntimeError, repr(result))
        self.assertLess(events.index(("start", "eval_locomo")), events.index(("ready", "writer_vllm")))
        self.assertIn(("exit", "eval_locomo", 0), events)
        self.assertNotIn("directories", calls)
        self.assertNotIn("eval_split_longmemeval_s", calls)
        self.assertEqual(set(stopped), set(calls))

    def test_background_cleanup_failure_preserves_locomo_and_retries_owned_groups_on_exit(self):
        calls, _, events, stopped, result = self.run_launch(self.args(), plans={
            "directories": [0], "eval_locomo": [None, None, 0]},
            cleanup_failures_once=("directories", "writer_vllm"))
        self.assertIsInstance(result, RuntimeError, repr(result))
        self.assertIn("Cleanup failed", str(result))
        self.assertNotIn("eval_split_longmemeval_s", calls)
        foreground_end = events.index(("exit", "eval_locomo", 0))
        for name in ("eval_locomo", "qwen_vllm"):
            self.assertNotIn(("stop", name), events[:foreground_end])
        for name in ("directories", "writer_vllm"):
            self.assertLess(events.index(("stop_failed", name)), foreground_end)
            self.assertGreater(events.index(("stopped", name)), foreground_end)
            self.assertEqual(stopped.count(name), 2)
        self.assertEqual({event[1] for event in events if event[0] == "stopped"}, set(calls))

    def test_task_service_failure_does_not_interrupt_independent_directory_build(self):
        calls, _, events, stopped, result = self.run_launch(self.args(), plans={
            "qwen_vllm": [None, 1], "directories": [None, None, None, 0],
            "eval_locomo": [None]})
        self.assertIsInstance(result, RuntimeError, repr(result))
        self.assertIn(("exit", "directories", 0), events)
        self.assertLess(events.index(("exit", "qwen_vllm", 1)), events.index(("exit", "directories", 0)))
        self.assertNotIn("eval_split_longmemeval_s", calls)
        self.assertEqual(set(stopped), set(calls))

    def test_failed_locomo_does_not_discard_successful_background_evaluation(self):
        calls, _, events, stopped, result = self.run_launch(self.args(), plans={
            "eval_locomo": [1], "directories": [None, 0]})
        self.assertIsInstance(result, RuntimeError, repr(result))
        self.assertIn(("exit", "eval_split_longmemeval_s", 0), events)
        self.assert_finished_before(events, "directories", "eval_split_longmemeval_s")
        self.assertEqual(set(stopped), set(calls))

    def test_checkpoint_task_and_remote_settings_reach_each_evaluation(self):
        args = self.args(checkpoints=["final"], task_agents=["local", "remote"],
                         remote_model="provider-model", limit=2)
        calls, _, _, _, result = self.run_launch(args, environment={
            "OPENAI_API_KEY": "fixture-secret", "OPENAI_BASE_URL": "https://provider.invalid/v1"})
        self.assertIsInstance(result, Path, repr(result))
        for worker in ("eval_locomo", "eval_split_longmemeval_s"):
            command, options = calls[worker]
            self.assertEqual(self.values(command, "--checkpoints"), ["final"])
            self.assertEqual(self.values(command, "--task-agents"), ["local", "remote"])
            self.assertEqual(self.values(command, "--remote-model"), ["provider-model"])
            self.assertEqual(self.values(command, "--limit"), ["2"])
            self.assertEqual(options["env"]["OPENAI_API_KEY"], "fixture-secret")
            self.assertNotIn("fixture-secret", " ".join(command))
        self.assertTrue(all("fixture-secret" not in p.read_text() for p in result.iterdir() if p.is_file()))

    def test_interrupt_closes_every_created_group_without_starting_longmemeval_eval(self):
        calls, _, _, stopped, result = self.run_launch(self.args(), interrupt=True, plans={
            "directories": [None], "eval_locomo": [None]})
        self.assertIsInstance(result, KeyboardInterrupt, repr(result))
        self.assertEqual(set(calls), {"qwen_vllm", "writer_vllm", "directories", "eval_locomo"})
        self.assertEqual(set(stopped), set(calls))

    def test_external_task_service_is_reused_but_writer_has_its_own_gpu_service(self):
        endpoint = "http://127.0.0.1:8111/v1"
        calls, _, _, stopped, result = self.run_launch(self.args(local_base_url=endpoint))
        self.assertIsInstance(result, Path, repr(result))
        self.assertNotIn("qwen_vllm", calls)
        self.assertNotIn("qwen_vllm", stopped)
        self.assertEqual(calls["writer_vllm"][1]["env"]["CUDA_VISIBLE_DEVICES"], "2")
        for worker in ("eval_locomo", "eval_split_longmemeval_s"):
            self.assertEqual(self.values(calls[worker][0], "--local-base-url"), [endpoint])
        self.assertNotEqual(self.values(calls["directories"][0], "--local-base-url"), [endpoint])

    def test_explicit_disable_and_single_dataset_keep_first_gpu_reserved(self):
        cases = ({"parallel_directories": False}, {"datasets": ["locomo"]})
        for index, overrides in enumerate(cases):
            with self.subTest(overrides=overrides):
                args = self.args(job_dir=str(self.root / str(index)), **overrides)
                calls, _, _, stopped, result = self.run_launch(args)
                self.assertIsInstance(result, Path, repr(result))
                self.assertEqual(set(calls), {"qwen_vllm", "eval"})
                self.assertTrue(all(options["env"]["CUDA_VISIBLE_DEVICES"] != "2"
                                    for _, options in calls.values()))
                self.assertEqual(set(stopped), set(calls))


if __name__ == "__main__":
    unittest.main()
