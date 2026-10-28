"""Exercise GPU isolation and cleanup without allocating GPUs or starting models."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import io
import os
import tempfile
import unittest

from agmem.four_gpu import gpu_roles, main


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

    def run_workflow(self, writer_failure=False, likelihood_mode="off"):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            old_cwd = Path.cwd()
            os.chdir(directory)
            stack.callback(os.chdir, old_cwd)
            config = {
                "data": {"sources": ["locomo", "split_longmemeval_s"], "prepared_root": "prepared"},
                "models": {"task_path": "model27b"}, "service": {},
                "reward": {"likelihood": {"mode": likelihood_mode}},
                "training": {"output_dir": "experiment/train", "validation_every": 25, "checkpoint_every": 25},
            }
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
                    label, states = "train", [None, 0]
                calls.append((label, command, kwargs))
                self.assertTrue(kwargs["start_new_session"])
                return FakeProcess(label, states)
            for target, options in [
                ("agmem.four_gpu.configs", {"return_value": (config, build)}),
                ("agmem.four_gpu.check_inputs", {"return_value": {"checked": True}}),
                ("agmem.four_gpu.subprocess.Popen", {"side_effect": popen}),
                ("agmem.four_gpu.stop_group", {"side_effect": lambda p: stopped.append(p.label)}),
                ("agmem.four_gpu.wait_ready", {}),
                ("agmem.four_gpu.free_port", {"side_effect": [48001, 48002]}),
                ("agmem.four_gpu.time.sleep", {}),
                ("agmem.four_gpu.signal.signal", {}),
                ("sys.argv", {"new": ["four_gpu"]}),
            ]:
                stack.enter_context(patch(target, **options))
            stack.enter_context(patch.dict(os.environ, {"SLURM_JOB_ID": "test", "CUDA_VISIBLE_DEVICES": "2,4,6,7"}))
            with redirect_stdout(io.StringIO()):
                if writer_failure:
                    with self.assertRaisesRegex(RuntimeError, "writer pipeline failed"):
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


if __name__ == "__main__":
    unittest.main()
