"""Validate the runnable recipe and isolate it from existing v1 experiments."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from agmem.core import load_config
from agmem.four_gpu import configs


ROOT = Path(__file__).resolve().parents[1]


class DAPOV2ConfigTests(unittest.TestCase):
    def test_default_launcher_uses_half_point_penalty_and_all_correct_format_filter(self):
        train, _ = configs()
        self.assertEqual(train["data"]["sources"], ["locomo"])
        test = load_config(ROOT / "configs/eval_dapo_locomo_query9010_format05.yaml")
        self.assertEqual(test["data"]["sources"], ["locomo"])
        self.assertEqual(test["data"]["splits"], ["test"])
        self.assertEqual(Path(test["data"]["prepared_root"]).resolve(), Path(train["jobs"]["test_prepared_root"]))
        self.assertEqual(train["reward"]["formula"], "judge_cost_v2")
        self.assertEqual(train["reward"]["format_penalty_mode"], "per_id")
        self.assertEqual(train["reward"]["format_penalty"], 0.5)
        self.assertEqual(test["reward"], train["reward"])
        self.assertEqual(train["reward"]["read_call_cost"], 0.01)
        self.assertEqual(train["retrieval"]["id_protocol"], "strict_v2")
        self.assertEqual(train["retrieval"]["max_ids_per_call"], 1)
        self.assertEqual(train["training"]["filter_metric"], "judge_or_all_correct_format")
        self.assertEqual(train["training"]["group_size"], 8)
        self.assertEqual(train["training"]["filter_min_std"], 1e-6)
        self.assertEqual(train["reward"]["likelihood"]["mode"], "off")
        self.assertEqual(train["training"]["advantage_std_floor"], 0)
        self.assertEqual(train["training"]["validation_every"], 4)
        self.assertEqual(train["training"]["checkpoint_every"], 25)
        self.assertEqual(train["execution"]["sampling_group_concurrency"], 4)
        self.assertEqual(train["execution"]["reader_batch_size"], 4)
        self.assertEqual(train["execution"]["task_agent_workers"], 2)
        self.assertEqual(train["training"]["mini_batch_groups"], 8)
        self.assertEqual(train["training"]["micro_batch_rollouts"], 2)
        self.assertEqual(train["reader"]["prompt_version"], "evidence_loop_v3")
        self.assertTrue(train["data"]["input_root"].endswith("locomo_query9010_seed42_20261102"))
        self.assertTrue(train["data"]["prepared_root"].endswith("prepared_locomo_query9010_seed42_20261102"))
        legacy = load_config(ROOT / "configs/train_dapo_no_d_restart.yaml")
        self.assertEqual(legacy["reward"]["formula"], "judge_gated_v1")
        self.assertEqual(legacy["training"]["filter_metric"], "reward")
        self.assertNotEqual(Path(legacy["training"]["output_dir"]).resolve(), Path(train["training"]["output_dir"]))
        # Reader state/prompt changes; model/generation settings and other roles do not.
        self.assertEqual({k: v for k, v in train["reader"].items() if k != "prompt_version"}, legacy["reader"])
        for key in ("task_agent", "writer"):
            self.assertEqual(train[key], legacy[key])
        self.assertEqual(train["reward"]["judge"], legacy["reward"]["judge"])

    def test_opt_in_recipes_have_separate_outputs_and_valid_batch_sizes(self):
        recipes = {
            "train_dapo_locomo_s_v2.yaml": (1, 1, "off"),
            "train_dapo_micro2_v2.yaml": (1, 2, "off"),
            "train_dapo_mini2_micro2_v2.yaml": (2, 2, "off"),
            "train_dapo_diagnostic_v2.yaml": (1, 1, "diagnostic"),
            "train_dapo_parallel_v2.yaml": (1, 2, "off"),
            "train_dapo_reader_ids_v2.yaml": (1, 2, "off"),
            "train_dapo_reader_ids_format_count_v2.yaml": (1, 2, "off"),
            "train_dapo_max2_v2.yaml": (1, 2, "off"),
            "train_dapo_max1_v2.yaml": (1, 2, "off"),
            "train_dapo_max1_no_call_cost_v2.yaml": (1, 2, "off"),
            "train_dapo_max1_no_call_cost_mixed_v2.yaml": (1, 2, "off"),
            "train_dapo_reader_loop_v3_val1pct.yaml": (1, 2, "off"),
            "train_dapo_reader_loop_v3_val100.yaml": (1, 2, "off"),
            "train_dapo_reader_loop_v3_query9010.yaml": (1, 2, "off"),
            "train_dapo_reader_loop_v3_query9010_id_cost.yaml": (8, 2, "off"),
            "train_dapo_reader_loop_v3_query9010_format05.yaml": (8, 2, "off"),
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

    def test_cumulative_recipe_preserves_old_mode_and_rejects_reward_switch_on_resume(self):
        from agmem.prompts import configured_reader_input_contract
        from agmem.resume import validate_resume_config
        previous = load_config(ROOT / "configs/train_dapo_reader_ids_v2.yaml")
        current = load_config(ROOT / "configs/train_dapo_reader_ids_format_count_v2.yaml")
        self.assertEqual(previous["reward"].get("format_penalty_mode", "per_rollout"), "per_rollout")
        self.assertEqual(current["reward"]["format_penalty_mode"], "per_decision")
        self.assertEqual(configured_reader_input_contract(previous), configured_reader_input_contract(current))
        expected = deepcopy(previous)
        expected["reward"]["format_penalty_mode"] = "per_decision"
        expected["training"]["output_dir"] = current["training"]["output_dir"]
        self.assertEqual(current, expected)
        # Isolate the reward change from output-directory changes.
        switched = deepcopy(previous)
        switched["reward"]["format_penalty_mode"] = "per_decision"
        with self.assertRaisesRegex(ValueError, "Resume configuration changed: reward"):
            validate_resume_config(previous, switched)
        validate_resume_config(current, deepcopy(current))

    def test_no_call_cost_recipe_preserves_old_experiment_and_requires_fresh_training(self):
        from agmem.resume import validate_resume_config
        old = load_config(ROOT / "configs/train_dapo_max1_v2.yaml")
        new = load_config(ROOT / "configs/train_dapo_max1_no_call_cost_v2.yaml")
        self.assertEqual(old["reward"]["read_call_cost"], 0.01)
        self.assertEqual(old["training"]["filter_metric"], "judge")
        expected = deepcopy(old)
        expected["reward"]["read_call_cost"] = 0.0
        expected["training"].update(filter_metric="reward", filter_min_std=0.0,
                                    output_dir=new["training"]["output_dir"])
        self.assertEqual(expected, new)
        self.assertNotEqual(old["training"]["output_dir"], new["training"]["output_dir"])
        for section, key, value in (("reward", "read_call_cost", 0.0),
                                    ("training", "filter_metric", "reward")):
            switched = deepcopy(old)
            switched[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"Resume configuration changed: {section}"):
                validate_resume_config(old, switched)
        validate_resume_config(new, deepcopy(new))

    def test_restored_mixed_recipe_changes_only_filter_and_output(self):
        from agmem.resume import validate_resume_config
        old = load_config(ROOT / "configs/train_dapo_max1_no_call_cost_v2.yaml")
        new = load_config(ROOT / "configs/train_dapo_max1_no_call_cost_mixed_v2.yaml")
        expected = deepcopy(old)
        expected["training"].update(filter_metric="judge", filter_min_std=1e-6,
                                    output_dir=new["training"]["output_dir"])
        self.assertEqual(expected, new)
        self.assertEqual(old["training"]["filter_metric"], "reward")
        self.assertEqual(new["reward"]["read_call_cost"], 0.0)
        self.assertNotEqual(old["training"]["output_dir"], new["training"]["output_dir"])
        switched = deepcopy(old)
        switched["training"]["filter_metric"] = "judge"
        with self.assertRaisesRegex(ValueError, "Resume configuration changed: training"):
            validate_resume_config(old, switched)
        validate_resume_config(new, deepcopy(new))

    def test_half_point_recipe_preserves_old_experiment_and_rejects_incompatible_resume(self):
        from agmem.resume import validate_resume_config
        old = load_config(ROOT / "configs/train_dapo_reader_loop_v3_query9010_id_cost.yaml")
        new = load_config(ROOT / "configs/train_dapo_reader_loop_v3_query9010_format05.yaml")
        self.assertEqual(old["reward"]["format_penalty"], .1)
        self.assertEqual(old["training"]["filter_metric"], "judge")
        expected = deepcopy(old)
        expected["reward"]["format_penalty"] = .5
        expected["training"].update(filter_metric="judge_or_all_correct_format",
                                    output_dir=new["training"]["output_dir"])
        expected["jobs"]["experiment_name"] = new["jobs"]["experiment_name"]
        self.assertEqual(new, expected)
        self.assertNotEqual(old["training"]["output_dir"], new["training"]["output_dir"])
        for section, key, value in (("reward", "format_penalty", .5),
                                    ("training", "filter_metric", "judge_or_all_correct_format")):
            switched = deepcopy(old)
            switched[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"Resume configuration changed: {section}"):
                validate_resume_config(old, switched)
        validate_resume_config(new, deepcopy(new))

    def test_default_filter_accepts_only_correct_format_signal_or_trainable_mixed_judges(self):
        from agmem.dapo_sampling import assess_group
        from tests.test_dapo_sampling import examples, group_for
        config, _ = configs()
        example = examples(1)[0]
        for judge in (0, 1):
            rewards = [2 * judge - 1 - 0.5 * (i % 2) for i in range(8)]
            group = group_for(example, rewards, judges=[judge] * 8)
            for i, row in enumerate(group):
                row.format_reward = -.5 * (i % 2)
                row.format_violation_count = i % 2
                row.format_failure = bool(i % 2)
            result = assess_group(group, config)
            self.assertEqual(result["accepted"], judge == 1)
            self.assertEqual(result["accepted_all_correct_format"], judge == 1)
            self.assertTrue(result["cost_format_only_signal"])
            self.assertTrue(any(row.advantage > 0 for row in group))
            self.assertTrue(any(row.advantage < 0 for row in group))
        mixed = group_for(example, [1.0] + [-1.0] * 7, judges=[1] + [0] * 7)
        self.assertTrue(assess_group(mixed, config)["accepted"])
        self.assertEqual(len(mixed), 8)
        mixed[0].decisions[0].completion.truncated = True
        self.assertFalse(assess_group(mixed, config)["accepted"])
        equal = group_for(example, [-1.1] * 8, judges=[0] * 8)
        self.assertFalse(assess_group(equal, config)["accepted"])
        self.assertTrue(all(row.advantage == 0 for row in equal))

    def test_read_cost_must_be_finite_nonnegative_and_numeric(self):
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cost.yaml"
            parent = ROOT / "configs/train_dapo_max1_no_call_cost_v2.yaml"
            for value in (-0.01, float("inf"), float("nan"), True, "0", None):
                path.write_text(yaml.safe_dump({"extends": str(parent), "reward": {"read_call_cost": value}}))
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "read_call_cost"):
                    load_config(path)

    def test_new_reward_cannot_accidentally_use_legacy_retrieval(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.yaml"
            parent = ROOT / "configs/train_dapo_locomo_s_v2.yaml"
            path.write_text(f"extends: {parent}\nretrieval:\n  id_protocol: strict_v1\n")
            with self.assertRaisesRegex(ValueError, "requires retrieval.id_protocol=strict_v2"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
