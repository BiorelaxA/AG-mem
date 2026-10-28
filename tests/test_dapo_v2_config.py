"""Validate the runnable recipe and isolate it from existing v1 experiments."""
from pathlib import Path
import tempfile
import unittest

from agmem.core import load_config
from agmem.four_gpu import configs


ROOT = Path(__file__).resolve().parents[1]


class DAPOV2ConfigTests(unittest.TestCase):
    def test_default_launcher_uses_mixed_judge_reward_and_repeat_feedback(self):
        train, _ = configs()
        self.assertEqual(train["reward"]["formula"], "judge_cost_v2")
        self.assertEqual(train["retrieval"]["id_protocol"], "strict_v2")
        self.assertEqual(train["training"]["filter_metric"], "judge")
        self.assertEqual(train["reward"]["likelihood"]["mode"], "off")
        self.assertEqual(train["training"]["advantage_std_floor"], 0)
        self.assertEqual(train["training"]["validation_every"], 25)
        self.assertEqual(train["training"]["checkpoint_every"], 25)
        legacy = load_config(ROOT / "configs/train_dapo_no_d_restart.yaml")
        self.assertEqual(legacy["reward"]["formula"], "judge_gated_v1")
        self.assertEqual(legacy["training"]["filter_metric"], "reward")
        self.assertNotEqual(Path(legacy["training"]["output_dir"]).resolve(), Path(train["training"]["output_dir"]))
        # No change to model selection, task prompts/settings, or Judge rubrics.
        for key in ("reader", "task_agent", "writer"):
            self.assertEqual(train[key], legacy[key])
        self.assertEqual(train["reward"]["judge"], legacy["reward"]["judge"])

    def test_opt_in_recipes_have_separate_outputs_and_valid_batch_sizes(self):
        recipes = {
            "train_dapo_locomo_s_v2.yaml": (1, 1, "off"),
            "train_dapo_micro2_v2.yaml": (1, 2, "off"),
            "train_dapo_mini2_micro2_v2.yaml": (2, 2, "off"),
            "train_dapo_diagnostic_v2.yaml": (1, 1, "diagnostic"),
        }
        outputs = set()
        for name, expected in recipes.items():
            with self.subTest(recipe=name):
                config = load_config(ROOT / "configs" / name)
                training = config["training"]
                self.assertEqual((training["mini_batch_groups"], training["micro_batch_rollouts"],
                                  config["reward"]["likelihood"]["mode"]), expected)
                self.assertEqual(training["max_micro_batch_tokens"], 16384)
                outputs.add(training["output_dir"])
        self.assertEqual(len(outputs), len(recipes))

    def test_new_reward_cannot_accidentally_use_legacy_retrieval(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yaml"
            parent = ROOT / "configs/train_dapo_locomo_s_v2.yaml"
            path.write_text(f"extends: {parent}\nretrieval:\n  id_protocol: strict_v1\n")
            with self.assertRaisesRegex(ValueError, "requires retrieval.id_protocol=strict_v2"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
