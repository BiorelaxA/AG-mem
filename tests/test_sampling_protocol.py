from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from agmem.core import load_config
from agmem.execution import execution_settings, sampling_protocol
from agmem.resume import validate_resume_config


class SamplingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/train_dapo_parallel_v2.yaml")

    def test_legacy_missing_settings_match_explicit_defaults_on_resume(self):
        previous = deepcopy(self.config)
        previous.pop("execution", None)
        current = deepcopy(previous)
        current["execution"] = execution_settings(previous)
        self.assertEqual(sampling_protocol(previous), sampling_protocol(current))
        self.assertEqual(sampling_protocol(previous)["scheduler"], "serial_groups_v1")
        validate_resume_config(previous, current)
        validate_resume_config(current, previous)

    def test_dapo_resume_rejects_changes_to_generation_batching(self):
        for key, value in (("sampling_group_concurrency", 2),
                           ("reader_batch_size", 2),
                           ("reader_batch_max_tokens", 8192)):
            current = deepcopy(self.config)
            current["execution"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "sampling protocol"):
                validate_resume_config(self.config, current)
        serial = deepcopy(self.config)
        serial["execution"]["sampling_group_concurrency"] = 1
        with self.assertRaisesRegex(ValueError, "sampling protocol"):
            validate_resume_config(serial, self.config)

    def test_http_and_writer_workers_can_change_without_changing_sampling(self):
        current = deepcopy(self.config)
        current["execution"]["task_agent_workers"] = 4
        current["execution"]["writer_workers"] = 3
        self.assertEqual(sampling_protocol(current), sampling_protocol(self.config))
        validate_resume_config(self.config, current)

    def test_group_concurrency_must_be_a_positive_integer(self):
        for value in (0, -1, 1.5, True, "4", None):
            config = {"execution": {"sampling_group_concurrency": value}}
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "sampling_group_concurrency"):
                execution_settings(config)

    def test_loading_config_rejects_cross_group_sampling_for_grpo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grpo_parallel.yaml"
            path.write_text(yaml.safe_dump({
                "extends": str(Path("configs/default.yaml").resolve()),
                "training": {"algorithm": "grpo"},
                "execution": {"sampling_group_concurrency": 4},
            }))
            with self.assertRaisesRegex(ValueError, "requires training.algorithm=dapo"):
                load_config(path)

    def test_parallel_recipe_preserves_training_and_uses_separate_output(self):
        serial = load_config("configs/train_dapo_micro2_v2.yaml")
        settings = execution_settings(self.config)
        self.assertEqual(settings["sampling_group_concurrency"], 4)
        self.assertEqual(settings["reader_batch_size"], 4)
        self.assertEqual(settings["reader_batch_max_tokens"], 16384)
        self.assertEqual(settings["task_agent_workers"], 2)
        self.assertEqual(sampling_protocol(self.config)["scheduler"], "ordered_group_windows_v1")
        training = self.config["training"]
        self.assertEqual(training["algorithm"], "dapo")
        self.assertEqual(training["mini_batch_groups"], 1)
        self.assertEqual(training["micro_batch_rollouts"], 2)
        self.assertEqual(training["group_size"], 8)
        self.assertEqual(training["candidate_groups"], 32)
        self.assertEqual(training["target_groups"], 16)
        self.assertNotEqual(training["output_dir"], serial["training"]["output_dir"])
        self.assertEqual({k: v for k, v in training.items() if k != "output_dir"},
                         {k: v for k, v in serial["training"].items() if k != "output_dir"})
        self.assertEqual(self.config["reward"], serial["reward"])
        self.assertEqual(self.config["reader"], serial["reader"])
        self.assertEqual(self.config["task_agent"], serial["task_agent"])


if __name__ == "__main__":
    unittest.main()
