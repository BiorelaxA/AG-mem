"""Exercise GPU isolation and cleanup without allocating GPUs or starting models."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import io
import os
import tempfile
import unittest

from agmem.four_gpu import configs, gpu_roles, main


class FakeProcess:
    def __init__(self, label, statuses):
        self.label = label
        self.statuses = iter(statuses)
        self.returncode = None

    def poll(self):
        self.returncode = next(self.statuses, self.returncode)
        return self.returncode

    def wait(self):
        self.returncode = 0
        return 0


class FourGPUWorkflowTests(unittest.TestCase):
    def test_bad_visibility_reports_actual_value_instead_of_guessing_devices(self):
        for visible in ("", "0", "0-3", "0,1,2,2", "0,1,2,3,4", "0,1,2,"):
            with self.subTest(visible=visible):
                with self.assertRaises(ValueError) as error:
                    gpu_roles(visible)
                self.assertIn(repr(visible), str(error.exception))
                self.assertIn("srun", str(error.exception))

    def run_workflow(self, writer_failure=False, likelihood_mode="off", process_enabled=False,
                     train_only=False, train_failure=False, extra_args=(), same_parent=False):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            old_cwd = Path.cwd()
            os.chdir(directory)
            stack.callback(os.chdir, old_cwd)
            config = {
                "data": {"sources": ["locomo", "split_longmemeval_s"], "prepared_root": "prepared"},
                "models": {"task_path": "model27b"}, "service": {},
                "reward": {"likelihood": {"mode": likelihood_mode}},
                "training": {"output_dir": "experiment/train", "validation_every": 25, "checkpoint_every": 25,
                             "process_reward": {"enabled": process_enabled}},
            }
            if same_parent:
                config["data"]["prepared_root"] = "experiment/prepared"
            build = deepcopy(config)
            build["data"]["sources"] = ["split_longmemeval_m"]
            calls, stopped = [], []
            def popen(command, **kwargs):
                if "-c" in command:
                    label, states = "preflight", [0]
                elif "unittest" in command:
                    label, states = "parallel", [0]
                elif "vllm.entrypoints.openai.api_server" in command:
                    label, states = "server-" + kwargs["env"]["CUDA_VISIBLE_DEVICES"], [None]
                elif "agmem.jobs" in command:
                    label, states = "writer", [1 if writer_failure else 0]
                else:
                    label, states = "train", [None, 1 if train_failure else 0]
                calls.append((label, command, kwargs))
                self.assertTrue(kwargs["start_new_session"])
                return FakeProcess(label, states)
            for target, options in [
                ("agmem.four_gpu.configs", {"return_value": (config, None if train_only else build)}),
                ("agmem.four_gpu.check_inputs", {"return_value": {"checked": True}}),
                ("agmem.four_gpu.subprocess.Popen", {"side_effect": popen}),
                ("agmem.four_gpu.stop_group", {"side_effect": lambda p: stopped.append(p.label)}),
                ("agmem.four_gpu.wait_ready", {}),
                ("agmem.four_gpu.free_port", {"side_effect": [48001, 48002]}),
                ("agmem.four_gpu.time.sleep", {}),
                ("agmem.four_gpu.signal.signal", {}),
                ("agmem.four_gpu.latest_checkpoint", {"return_value": Path("checkpoint-00000004")}),
                ("sys.argv", {"new": ["four_gpu", *(["--train-only"] if train_only else []), *extra_args]}),
            ]:
                stack.enter_context(patch(target, **options))
            stack.enter_context(patch.dict(os.environ, {"SLURM_JOB_ID": "test",
                                                      "CUDA_VISIBLE_DEVICES": "2,4,7" if train_only else "2,4,6,7"}))
            with redirect_stdout(io.StringIO()):
                if writer_failure:
                    with self.assertRaisesRegex(RuntimeError, "writer pipeline failed"):
                        main()
                elif train_failure:
                    with self.assertRaisesRegex(RuntimeError, "task pipeline failed"):
                        main()
                else:
                    main()
            return calls, stopped

    def test_writer_completion_does_not_stop_training_and_gpu_ids_are_preserved(self):
        calls, stopped = self.run_workflow()
        by_name = {name: kwargs for name, _, kwargs in calls}
        self.assertEqual(by_name["writer"]["env"]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(by_name["train"]["env"]["CUDA_VISIBLE_DEVICES"], "4,6")
        self.assertEqual(by_name["train"]["env"]["AGMEM_READER_MODEL_PARALLEL_SIZE"], "2")
        self.assertIn("server-7", by_name)
        self.assertIn("server-2", by_name)
        self.assertLess(stopped.index("server-2"), stopped.index("train"))
        self.assertCountEqual(stopped, [name for name, _, _ in calls])

    def test_failed_writer_cleans_up_both_servers_and_reader(self):
        calls, stopped = self.run_workflow(writer_failure=True)
        self.assertCountEqual(stopped, [name for name, _, _ in calls])

    def test_likelihood_limits_task_logprob_workspace_without_shortening_context(self):
        for mode in ("off", "diagnostic", "reward"):
            with self.subTest(mode=mode):
                calls, _ = self.run_workflow(likelihood_mode=mode)
                servers = {name: command for name, command, _ in calls if name.startswith("server-")}
                for name, command in servers.items():
                    scoring = name == "server-7" and mode != "off"
                    def value(flag):
                        return command[command.index(flag) + 1]
                    self.assertEqual(int(value("--max-num-batched-tokens")), 1024 if scoring else 4096)
                    self.assertEqual(float(value("--gpu-memory-utilization")), 0.80 if scoring else 0.90)
                    self.assertEqual(int(value("--max-num-seqs")), 1 if scoring else 2)
                    self.assertEqual(int(value("--max-model-len")), 65536)
                    self.assertIn("--enable-chunked-prefill", command)

    def test_process_scoring_reserves_logprob_workspace_even_with_terminal_likelihood_off(self):
        calls, _ = self.run_workflow(likelihood_mode="off", process_enabled=True)
        servers = {name: command for name, command, _ in calls if name.startswith("server-")}
        task, writer = servers["server-7"], servers["server-2"]
        for command, scoring in ((task, True), (writer, False)):
            def value(flag):
                return command[command.index(flag) + 1]
            self.assertEqual(int(value("--max-num-batched-tokens")), 1024 if scoring else 4096)
            self.assertEqual(float(value("--gpu-memory-utilization")), 0.80 if scoring else 0.90)
            self.assertEqual(int(value("--max-num-seqs")), 1 if scoring else 2)
            self.assertEqual(int(value("--max-model-len")), 65536)

    def test_train_only_requires_three_distinct_visible_ids(self):
        self.assertEqual(gpu_roles("2,4,7", train_only=True), {"reader": "2,4", "task": "7"})
        self.assertEqual(gpu_roles("GPU-a,GPU-b,GPU-c", train_only=True),
                         {"reader": "GPU-a,GPU-b", "task": "GPU-c"})
        for visible in ("", "0,1", "0,1,1", "0,1,2,3", "0,1,"):
            with self.subTest(visible=visible), self.assertRaises(ValueError) as error:
                gpu_roles(visible, train_only=True)
            self.assertIn(repr(visible), str(error.exception))
            self.assertIn("--gres=gpu:3", str(error.exception))

    def test_train_only_config_never_loads_background_directory_configuration(self):
        train = {"data": {"input_root": "input", "prepared_root": "prepared", "sources": ["locomo"]},
                 "models": {"reader_path": "model4b", "task_path": "model27b"},
                 "jobs": {"test_prepared_root": "test", "legacy_prepared_root": "legacy"},
                 "training": {"output_dir": "experiment/train"}}
        with patch("agmem.four_gpu.load_config", return_value=deepcopy(train)) as load:
            selected, build = configs("train_prm.yaml", train_only=True)
        load.assert_called_once_with("train_prm.yaml")
        self.assertIsNone(build)
        self.assertTrue(Path(selected["training"]["output_dir"]).is_absolute())

    def test_train_only_runs_one_server_two_reader_gpus_and_no_writer(self):
        calls, stopped = self.run_workflow(train_only=True, process_enabled=True)
        by_name = {name: (command, kwargs) for name, command, kwargs in calls}
        self.assertEqual(set(by_name), {"preflight", "parallel", "server-7", "train"})
        self.assertEqual(by_name["train"][1]["env"]["CUDA_VISIBLE_DEVICES"], "2,4")
        self.assertEqual(by_name["parallel"][1]["env"]["CUDA_VISIBLE_DEVICES"], "2,4")
        self.assertEqual(by_name["train"][1]["env"]["AGMEM_READER_MODEL_PARALLEL_SIZE"], "2")
        code = by_name["preflight"][0][-1]
        self.assertIn("device_count()==3", code)
        self.assertIn("range(3)", code)
        self.assertIn("runs/three_gpu/", str(by_name["train"][1]["stdout"].name))
        command = by_name["server-7"][0]
        def value(flag):
            return command[command.index(flag) + 1]
        self.assertEqual(int(value("--max-num-seqs")), 1)
        self.assertEqual(int(value("--max-num-batched-tokens")), 1024)
        self.assertEqual(float(value("--gpu-memory-utilization")), 0.80)
        self.assertEqual(int(value("--max-model-len")), 65536)
        self.assertCountEqual(stopped, list(by_name))

    def test_train_only_failure_cleans_up_its_reader_and_server(self):
        calls, stopped = self.run_workflow(train_only=True, train_failure=True)
        self.assertCountEqual(stopped, [name for name, _, _ in calls])

    def test_train_only_starts_when_prepared_and_training_are_siblings(self):
        calls, stopped = self.run_workflow(train_only=True, same_parent=True)
        names = [name for name, _, _ in calls]
        self.assertIn("train", names)
        self.assertCountEqual(stopped, names)

    def test_train_only_forwards_resume_and_update_limit(self):
        calls, _ = self.run_workflow(train_only=True, extra_args=("--resume", "--max-updates", "4"))
        command = next(command for name, command, _ in calls if name == "train")
        self.assertEqual(command[command.index("--resume") + 1], "checkpoint-00000004")
        self.assertEqual(command[command.index("--max-groups") + 1], "4")


if __name__ == "__main__":
    unittest.main()
