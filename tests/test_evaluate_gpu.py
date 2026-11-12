"""Launcher process ownership, GPU allocation and credential preflight without GPUs."""
from argparse import Namespace
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agmem.evaluate_gpu import launch


class FakeProcess:
    def __init__(self, label, statuses):
        self.label = label
        self.statuses = iter(statuses)
        self.returncode = None

    def poll(self):
        self.returncode = next(self.statuses, self.returncode)
        return self.returncode


class EvaluationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.suite = {
            "datasets": ["locomo", "split_longmemeval_s"],
            "checkpoints": {"best": "checkpoint-00000020", "final": "checkpoint-00000032"},
            "task_agents": ["local", "remote"], "output_root": str(self.root / "results"),
            "remote_task": {"model": "gpt-5-mini", "model_revision": "gpt-5-mini",
                            "request_profile": "openai_chat", "base_url_env": "OPENAI_BASE_URL",
                            "api_key_env": "OPENAI_API_KEY"},
        }
        self.base = {"models": {"task_path": "/model/qwen27b"},
                     "service": {"model": "agmem-27b"},
                     "task_agent": {"max_context_tokens": 65536},
                     "reward": {"likelihood": {"mode": "off"}}}

    def tearDown(self):
        self.temp.cleanup()

    def args(self, **changes):
        value = dict(config="eval.yaml", stage="all", datasets=None, checkpoints=None,
                     task_agents=["local"], local_base_url=None, remote_model=None,
                     limit=None, job_dir=str(self.root / "job"))
        value.update(changes)
        return Namespace(**value)

    def run_launch(self, args, *, visible="2,4,6,7", worker_status=0, readiness_error=None,
                   extra_environment=None, cleanup_failure=None):
        calls, stopped = [], []
        def popen(command, **kwargs):
            server = "vllm.entrypoints.openai.api_server" in command
            label = "qwen" if server else "evaluation"
            calls.append((label, command, kwargs))
            self.assertTrue(kwargs["start_new_session"])
            return FakeProcess(label, [None] if server else [worker_status])
        def stop(process):
            stopped.append(process.label)
            if process.label == cleanup_failure:
                raise RuntimeError("owned process cleanup failed")
        with ExitStack() as stack:
            stack.enter_context(patch("agmem.evaluate_gpu.load_suite", return_value=(deepcopy(self.suite), deepcopy(self.base), {})))
            stack.enter_context(patch("agmem.evaluate_gpu.subprocess.Popen", side_effect=popen))
            stack.enter_context(patch("agmem.evaluate_gpu.stop_group", side_effect=stop))
            ready = stack.enter_context(patch("agmem.evaluate_gpu.wait_ready", side_effect=readiness_error))
            stack.enter_context(patch("agmem.evaluate_gpu.free_port", return_value=48123))
            stack.enter_context(patch("agmem.evaluate_gpu.signal.signal"))
            stack.enter_context(patch("agmem.evaluate_gpu.time.sleep"))
            stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible,
                                                       **(extra_environment or {})}, clear=True))
            with redirect_stdout(io.StringIO()):
                try:
                    result = launch(args)
                except BaseException as exc:
                    return calls, stopped, ready, exc
            return calls, stopped, ready, result

    def test_shared_service_and_reader_preserve_allocation_and_leave_background_alone(self):
        calls, stopped, ready, result = self.run_launch(self.args())
        self.assertIsInstance(result, Path)
        self.assertEqual([label for label, _, _ in calls], ["qwen", "evaluation"])
        server_command, server_options = calls[0][1:]
        reader_command, reader_options = calls[1][1:]
        self.assertEqual(server_options["env"]["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(reader_options["env"]["CUDA_VISIBLE_DEVICES"], "4,6")
        self.assertEqual(reader_options["env"]["AGMEM_READER_MODEL_PARALLEL_SIZE"], "2")
        self.assertNotIn("2", [options["env"]["CUDA_VISIBLE_DEVICES"] for _, _, options in calls])
        self.assertEqual(server_command[server_command.index("--model") + 1], "/model/qwen27b")
        self.assertEqual(server_command[server_command.index("--max-model-len") + 1], "65536")
        self.assertEqual(reader_command[reader_command.index("--local-base-url") + 1], "http://127.0.0.1:48123/v1")
        self.assertEqual(stopped, ["evaluation", "qwen"])
        ready.assert_called_once()

    def test_three_gpu_sequential_evaluation_uses_two_reader_and_one_shared_device(self):
        for index, visible in enumerate(("2,4,6", "GPU-a,GPU-b,GPU-c")):
            with self.subTest(visible=visible):
                args = self.args(job_dir=str(self.root / f"three-{index}"), parallel_directories=False)
                calls, stopped, ready, result = self.run_launch(args, visible=visible)
                self.assertIsInstance(result, Path)
                self.assertEqual([name for name, _, _ in calls], ["qwen", "evaluation"])
                ids = visible.split(",")
                self.assertEqual(calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"], ids[2])
                self.assertEqual(calls[1][2]["env"]["CUDA_VISIBLE_DEVICES"], ",".join(ids[:2]))
                self.assertEqual(calls[1][2]["env"]["AGMEM_READER_MODEL_PARALLEL_SIZE"], "2")
                self.assertEqual(stopped, ["evaluation", "qwen"])
                ready.assert_called_once()
                schedule = json.loads((result / "schedule.json").read_text())
                self.assertEqual(schedule["mode"], "sequential_evaluation")
                self.assertEqual(schedule["gpu_roles"]["writer"], "")

    def test_three_gpu_mapping_also_supports_staged_retrieve_and_score(self):
        for stage in ("retrieve", "score"):
            with self.subTest(stage=stage):
                calls, stopped, ready, result = self.run_launch(
                    self.args(stage=stage, job_dir=str(self.root / stage)), visible="1,3,5")
                self.assertIsInstance(result, Path)
                worker = calls[-1][2]["env"]
                self.assertEqual(worker["CUDA_VISIBLE_DEVICES"], "1,3" if stage == "retrieve" else "")
                if stage == "score":
                    self.assertEqual(calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"], "5")
                    ready.assert_called_once()
                else:
                    ready.assert_not_called()

    def test_three_gpu_parallel_directories_request_fails_before_startup(self):
        args = self.args(parallel_directories=True)
        calls, stopped, ready, result = self.run_launch(args, visible="0,1,2")
        self.assertIsInstance(result, ValueError)
        self.assertIn("requires four allocated GPUs", str(result))
        self.assertEqual(calls, [])
        self.assertFalse(Path(args.job_dir).exists())
        ready.assert_not_called()

    def test_invalid_three_gpu_allocation_fails_before_startup(self):
        for index, visible in enumerate(("0,1,1", "0,,2", "-1,1,2", "NoDevFiles,1,2")):
            with self.subTest(visible=visible):
                args = self.args(job_dir=str(self.root / f"invalid-three-{index}"))
                calls, stopped, ready, result = self.run_launch(args, visible=visible)
                self.assertIsInstance(result, ValueError)
                self.assertEqual(calls, [])
                self.assertFalse(Path(args.job_dir).exists())
                ready.assert_not_called()

    def test_required_stores_and_checkpoints_are_preflighted_on_cpu_before_gpu_startup(self):
        self.suite["required_prepared_sources"] = {name: "/prepared/" + name for name in self.suite["datasets"]}
        for stage in ("all", "retrieve", "score", "judge-remote"):
            with self.subTest(stage=stage):
                args = self.args(stage=stage, datasets=["split_longmemeval_s"], checkpoints=["final"],
                                 job_dir=str(self.root / ("preflight-" + stage)))
                def prepare(command, **kwargs):
                    self.assertFalse(Path(args.job_dir).exists())
                    self.assertEqual(command[command.index("--stage") + 1], "prepare")
                    self.assertEqual(command[command.index("--datasets") + 1], "split_longmemeval_s")
                    self.assertEqual(command[command.index("--checkpoints") + 1], "final")
                    self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                    self.assertNotIn("AGMEM_READER_MODEL_PARALLEL_SIZE", kwargs["env"])
                    self.assertNotIn("--local-base-url", command)
                    self.assertTrue(kwargs["check"])
                with patch("agmem.evaluate_gpu.subprocess.run", side_effect=prepare) as preflight, \
                     patch("agmem.evaluate_gpu.checkpoint_model", return_value=Path("/checkpoint/model")) as checkpoint:
                    calls, stopped, ready, result = self.run_launch(
                        args, extra_environment={"AGMEM_READER_MODEL_PARALLEL_SIZE": "2"})
                self.assertIsInstance(result, Path)
                self.assertTrue(calls)
                preflight.assert_called_once()
                self.assertEqual([call.args[1] for call in checkpoint.call_args_list], ["final"])

    def test_required_missing_or_incomplete_directory_never_starts_gpu_or_creates_job(self):
        self.suite["required_prepared_sources"] = {name: "/prepared/" + name for name in self.suite["datasets"]}
        for index, reason in enumerate(("Missing required prepared source", "Incomplete or changed directory")):
            with self.subTest(reason=reason):
                args = self.args(job_dir=str(self.root / f"bad-source-{index}"))
                error = subprocess.CalledProcessError(1, ["prepare"], stderr=reason)
                with patch("agmem.evaluate_gpu.subprocess.run", side_effect=error) as preflight, \
                     patch("agmem.evaluate_gpu.checkpoint_model", return_value=Path("/checkpoint/model")) as checkpoint:
                    calls, stopped, ready, result = self.run_launch(args)
                self.assertIs(result, error)
                self.assertEqual(calls, [])
                self.assertEqual(stopped, [])
                self.assertFalse(Path(args.job_dir).exists())
                ready.assert_not_called()
                preflight.assert_called_once()
                self.assertEqual([call.args[1] for call in checkpoint.call_args_list], ["best", "final"])

    def test_required_missing_checkpoint_fails_before_directory_or_gpu_process(self):
        self.suite["required_prepared_sources"] = {name: "/prepared/" + name for name in self.suite["datasets"]}
        args = self.args()
        error = ValueError("Missing checkpoint model")
        with patch("agmem.evaluate_gpu.subprocess.run") as preflight, \
             patch("agmem.evaluate_gpu.checkpoint_model", side_effect=error):
            calls, stopped, ready, result = self.run_launch(args)
        self.assertIs(result, error)
        self.assertEqual(calls, [])
        self.assertFalse(Path(args.job_dir).exists())
        preflight.assert_not_called()
        ready.assert_not_called()

    def test_legacy_suite_and_summary_do_not_run_required_store_preflight(self):
        for stage in ("all", "summarize"):
            with self.subTest(stage=stage):
                if stage == "summarize":
                    self.suite["required_prepared_sources"] = {
                        name: "/prepared/" + name for name in self.suite["datasets"]}
                with patch("agmem.evaluate_gpu.subprocess.run") as preflight, \
                     patch("agmem.evaluate_gpu.checkpoint_model") as checkpoint:
                    calls, stopped, ready, result = self.run_launch(
                        self.args(stage=stage, job_dir=str(self.root / ("no-preflight-" + stage))))
                self.assertIsInstance(result, Path)
                preflight.assert_not_called()
                checkpoint.assert_not_called()

    def test_cpu_stages_require_neither_credentials_nor_gpu_visibility(self):
        for stage in ("prepare", "summarize", "export-remote"):
            with self.subTest(stage=stage):
                args = self.args(stage=stage, task_agents=["local", "remote"], job_dir=str(self.root / stage))
                calls, stopped, ready, result = self.run_launch(args, visible="")
                self.assertIsInstance(result, Path)
                self.assertEqual([label for label, _, _ in calls], ["evaluation"])
                self.assertEqual(calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertNotIn("AGMEM_READER_MODEL_PARALLEL_SIZE", calls[0][2]["env"])
                ready.assert_not_called()
                self.assertEqual(stopped, ["evaluation"])

    def test_judge_remote_requires_only_local_judge_and_forwards_returned_root(self):
        args = self.args(stage="judge-remote", task_agents=["remote"],
                         responses_root="/returned", datasets=["locomo"])
        calls, stopped, ready, result = self.run_launch(args)
        self.assertIsInstance(result, Path)
        self.assertEqual([label for label, _, _ in calls], ["qwen", "evaluation"])
        self.assertEqual(calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(calls[1][2]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("AGMEM_READER_MODEL_PARALLEL_SIZE", calls[1][2]["env"])
        command = calls[1][1]
        self.assertEqual(command[command.index("--responses-root") + 1], "/returned")
        self.assertEqual(command[command.index("--stage") + 1], "judge-remote")
        ready.assert_called_once()
        self.assertEqual(stopped, ["evaluation", "qwen"])

    def test_judge_remote_can_reuse_existing_qwen_without_gpu_or_credentials(self):
        args = self.args(stage="judge-remote", task_agents=["remote"],
                         local_base_url="http://127.0.0.1:8111/v1")
        calls, stopped, ready, result = self.run_launch(args, visible="")
        self.assertIsInstance(result, Path)
        self.assertEqual([label for label, _, _ in calls], ["evaluation"])
        self.assertEqual(calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn("http://127.0.0.1:8111/v1", calls[0][1])
        ready.assert_not_called()
        self.assertEqual(stopped, ["evaluation"])

    def test_judge_remote_single_gpu_uses_only_the_allocated_device(self):
        for index, device in enumerate(("5", "GPU-fixture-uuid")):
            with self.subTest(device=device):
                args = self.args(stage="judge-remote", task_agents=["remote"], datasets=["locomo"],
                                 job_dir=str(self.root / f"single-{index}"))
                calls, stopped, ready, result = self.run_launch(args, visible=device)
                self.assertIsInstance(result, Path)
                self.assertEqual([label for label, _, _ in calls], ["qwen", "evaluation"])
                self.assertEqual(calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"], device)
                self.assertEqual(calls[1][2]["env"]["CUDA_VISIBLE_DEVICES"], "")
                self.assertNotIn("AGMEM_READER_MODEL_PARALLEL_SIZE", calls[1][2]["env"])
                command = calls[0][1]
                self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
                self.assertEqual(command[command.index("--model") + 1], "/model/qwen27b")
                self.assertEqual(calls[1][1][calls[1][1].index("--stage") + 1], "judge-remote")
                ready.assert_called_once()
                self.assertEqual(stopped, ["evaluation", "qwen"])

    def test_judge_remote_rejects_missing_or_ambiguous_gpu_mapping(self):
        for index, visible in enumerate(("", "-1", "NoDevFiles", "0,1", "0,1,2", "0,1,2,2")):
            with self.subTest(visible=visible):
                args = self.args(stage="judge-remote", job_dir=str(self.root / f"invalid-{index}"))
                calls, stopped, ready, result = self.run_launch(args, visible=visible)
                self.assertIsInstance(result, ValueError)
                self.assertIn("one or four allocated GPU", str(result))
                self.assertEqual(calls, [])
                self.assertFalse(Path(args.job_dir).exists())
                ready.assert_not_called()

    def test_single_gpu_does_not_relax_reader_allocation_requirement(self):
        calls, stopped, ready, result = self.run_launch(self.args(), visible="5")
        self.assertIsInstance(result, ValueError)
        self.assertIn("Expected four", str(result))
        self.assertEqual(calls, [])
        ready.assert_not_called()

    def test_local_suite_default_finishes_without_remote_credentials(self):
        self.suite["task_agents"] = ["local"]
        self.suite["export_remote_inputs"] = True
        calls, stopped, ready, result = self.run_launch(self.args(task_agents=None))
        self.assertIsInstance(result, Path)
        self.assertEqual([label for label, _, _ in calls], ["qwen", "evaluation"])
        ready.assert_called_once()
        self.assertEqual(stopped, ["evaluation", "qwen"])

    def test_retrieve_uses_reader_only_and_does_not_require_remote_credentials(self):
        calls, stopped, ready, result = self.run_launch(self.args(stage="retrieve", task_agents=["remote"]))
        self.assertIsInstance(result, Path)
        self.assertEqual([label for label, _, _ in calls], ["evaluation"])
        self.assertEqual(calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"], "4,6")
        ready.assert_not_called()
        self.assertEqual(stopped, ["evaluation"])

    def test_external_service_score_stage_uses_no_gpu_or_owned_server(self):
        calls, stopped, ready, result = self.run_launch(
            self.args(stage="score", local_base_url="http://127.0.0.1:8111/v1"), visible="")
        self.assertIsInstance(result, Path)
        self.assertEqual([label for label, _, _ in calls], ["evaluation"])
        self.assertEqual(calls[0][2]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertIn("http://127.0.0.1:8111/v1", calls[0][1])
        ready.assert_not_called()
        self.assertEqual(stopped, ["evaluation"])

    def test_missing_remote_credentials_fail_before_starting_or_writing_logs(self):
        cases = ({}, {"OPENAI_BASE_URL": "https://provider.example/v1"})
        for index, environment in enumerate(cases):
            with self.subTest(environment=environment):
                args = self.args(task_agents=["remote"], job_dir=str(self.root / str(index)))
                calls, stopped, ready, result = self.run_launch(args, extra_environment=environment)
                self.assertIsInstance(result, ValueError)
                self.assertIn("environment variable", str(result))
                self.assertFalse(Path(args.job_dir).exists())
                self.assertEqual(calls, [])
                self.assertEqual(stopped, [])
                ready.assert_not_called()

    def test_remote_override_and_credentials_only_pass_through_environment(self):
        args = self.args(task_agents=["remote"], remote_model="provider-model", limit=2,
                         datasets=["locomo"], checkpoints=["best"])
        calls, stopped, _, result = self.run_launch(args, extra_environment={
            "OPENAI_BASE_URL": "https://provider.example/v1", "OPENAI_API_KEY": "secret-test-token"})
        self.assertIsInstance(result, Path)
        command, options = calls[-1][1:]
        self.assertEqual(command[command.index("--remote-model") + 1], "provider-model")
        self.assertEqual(command[command.index("--limit") + 1], "2")
        self.assertNotIn("secret-test-token", " ".join(command))
        self.assertEqual(options["env"]["OPENAI_API_KEY"], "secret-test-token")
        self.assertTrue(all("secret-test-token" not in p.read_text() for p in result.iterdir()))
        self.assertEqual(stopped, ["evaluation", "qwen"])

    def test_failed_evaluation_cleans_up_only_its_two_process_groups(self):
        calls, stopped, _, result = self.run_launch(self.args(), worker_status=1)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("Evaluation failed", str(result))
        self.assertEqual(stopped, ["evaluation", "qwen"])
        self.assertEqual(len(calls), 2)

    def test_one_cleanup_failure_does_not_leave_other_owned_process_untouched(self):
        calls, stopped, _, result = self.run_launch(self.args(), cleanup_failure="evaluation")
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("could not be cleaned up", str(result))
        self.assertEqual(stopped, ["evaluation", "qwen"])
        self.assertEqual(len(calls), 2)

    def test_invalid_yaml_selections_fail_before_any_process_or_log_directory(self):
        for index, (key, value) in enumerate((
                ("task_agents", []), ("task_agents", ["unknown"]),
                ("task_agents", ["local", "local"]), ("datasets", []), ("checkpoints", {}))):
            with self.subTest(key=key, value=value):
                previous = self.suite[key]
                self.suite[key] = value
                try:
                    args = self.args(task_agents=None, job_dir=str(self.root / str(index)))
                    calls, stopped, ready, result = self.run_launch(args)
                finally:
                    self.suite[key] = previous
                self.assertIsInstance(result, ValueError)
                self.assertEqual(calls, [])
                self.assertEqual(stopped, [])
                self.assertFalse(Path(args.job_dir).exists())
                ready.assert_not_called()

    def test_interrupted_or_failed_startup_cleans_up_only_created_service(self):
        for index, error in enumerate((RuntimeError("startup failed"), KeyboardInterrupt())):
            with self.subTest(error=type(error).__name__):
                args = self.args(job_dir=str(self.root / str(index)))
                calls, stopped, _, result = self.run_launch(args, readiness_error=error)
                self.assertIs(result, error)
                self.assertEqual([label for label, _, _ in calls], ["qwen"])
                self.assertEqual(stopped, ["qwen"])


if __name__ == "__main__":
    unittest.main()
