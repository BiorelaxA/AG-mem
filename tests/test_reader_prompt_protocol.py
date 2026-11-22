"""Keep Reader input changes explicit across runs, checkpoints and resumes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import yaml

from agmem.core import load_config
from agmem.prompts import configured_reader_input_contract
from agmem.resume import validate_reader_input_contract, validate_resume_config


ROOT = Path(__file__).resolve().parents[1]


class ReaderPromptProtocolTests(unittest.TestCase):
    def setUp(self):
        self.legacy = load_config(ROOT / "configs/train_dapo_parallel_v2.yaml")
        self.current = deepcopy(self.legacy)
        self.current["reader"]["prompt_version"] = "evidence_ids_v2"

    @staticmethod
    def metadata(config):
        return {"config": deepcopy(config),
                "reader_input_contract": configured_reader_input_contract(config)}

    def test_implicit_and_explicit_legacy_resume_are_equivalent_without_mutation(self):
        implicit = deepcopy(self.legacy)
        implicit["reader"].pop("prompt_version", None)
        explicit = deepcopy(implicit)
        explicit["reader"]["prompt_version"] = "legacy_v1"
        before = deepcopy((implicit, explicit))
        for previous, current in ((implicit, explicit), (explicit, implicit)):
            validate_resume_config(previous, current)
            # Older checkpoints have neither an explicit version nor a contract.
            validate_reader_input_contract({"config": previous}, current)
            validate_reader_input_contract(self.metadata(previous), current)
        self.assertEqual((implicit, explicit), before)

    def test_prompt_change_rejects_resume_in_both_directions(self):
        for previous, current in ((self.legacy, self.current), (self.current, self.legacy)):
            with self.subTest(version=current["reader"].get("prompt_version")):
                with self.assertRaisesRegex(ValueError, "(?i)reader"):
                    validate_resume_config(previous, current)
                with self.assertRaisesRegex(ValueError, "(?i)reader.*(?:contract|prompt_version)"):
                    validate_reader_input_contract(self.metadata(previous), current)

    def test_evidence_protocol_requires_matching_recorded_contract(self):
        metadata = self.metadata(self.current)
        validate_reader_input_contract(metadata, self.current)
        with self.assertRaisesRegex(ValueError, "(?i)reader.*contract"):
            validate_reader_input_contract({"config": self.current}, self.current)
        corrupted = deepcopy(metadata)
        hashes = [key for key in corrupted["reader_input_contract"] if "sha256" in key]
        self.assertTrue(hashes, "Reader contract must fingerprint the rendered system prompt")
        corrupted["reader_input_contract"][hashes[0]] = "0" * 64
        with self.assertRaisesRegex(ValueError, "(?i)reader.*contract"):
            validate_reader_input_contract(corrupted, self.current)

    def test_recorded_legacy_contract_detects_prompt_edit_without_version_change(self):
        from agmem import prompts
        metadata = self.metadata(self.legacy)
        with patch.object(prompts, "READER_SYSTEM", prompts.READER_SYSTEM + "\nChanged instruction."):
            with self.assertRaisesRegex(ValueError, "(?i)reader.*contract"):
                validate_reader_input_contract(metadata, self.legacy)

    def test_existing_retrieval_forks_allow_only_their_intended_prompt_difference(self):
        previous = load_config(ROOT / "configs/default.yaml")
        metadata = self.metadata(previous)
        for mode, changes in (("fork_retrieval", {"max_ids_per_call": 2}),
                              ("fork_id_protocol", {"id_protocol": "repair_v2"})):
            current = deepcopy(previous)
            current["retrieval"].update(changes)
            current["training"]["output_dir"] = "new-fork/train"
            with self.subTest(mode=mode):
                self.assertNotEqual(configured_reader_input_contract(previous),
                                    configured_reader_input_contract(current))
                validate_resume_config(previous, current, **{mode: True})
                validate_reader_input_contract(metadata, current, **{mode: True})
                corrupted = deepcopy(metadata)
                corrupted["reader_input_contract"]["unexpected"] = "changed"
                with self.assertRaisesRegex(ValueError, "(?i)reader.*contract"):
                    validate_reader_input_contract(corrupted, current, **{mode: True})
                current["reader"]["prompt_version"] = "evidence_ids_v2"
                with self.assertRaisesRegex(ValueError, "(?i)reader"):
                    validate_resume_config(previous, current, **{mode: True})

    def test_new_recipe_changes_reader_input_validation_and_destination(self):
        from agmem.four_gpu import configs
        recipe = load_config(ROOT / "configs/train_dapo_reader_ids_v2.yaml")
        self.assertEqual(recipe["reader"]["prompt_version"], "evidence_ids_v2")
        self.assertNotEqual(recipe["training"]["output_dir"], self.legacy["training"]["output_dir"])
        expected = deepcopy(self.legacy)
        expected["reader"]["prompt_version"] = "evidence_ids_v2"
        expected["training"]["validation_every"] = 16
        expected["training"]["output_dir"] = recipe["training"]["output_dir"]
        self.assertEqual(recipe, expected)
        launched, _ = configs(ROOT / "configs/train_dapo_reader_ids_v2.yaml")
        self.assertEqual(launched["reader"]["prompt_version"], "evidence_ids_v2")
        self.assertEqual(Path(launched["training"]["output_dir"]),
                         Path(recipe["training"]["output_dir"]).resolve())

    def test_load_config_rejects_invalid_reader_prompt_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.yaml"
            for value in (None, True, 1, "unknown", [], {}):
                path.write_text(yaml.safe_dump({
                    "extends": str(ROOT / "configs/train_dapo_parallel_v2.yaml"),
                    "reader": {"prompt_version": value},
                }))
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "prompt_version"):
                    load_config(path)

    def test_make_environment_passes_prompt_version_to_training_and_evaluation(self):
        from agmem.pipeline import make_environment
        policy = SimpleNamespace(check_capacity=lambda messages: None)
        with patch("agmem.pipeline.validate_directory_version", return_value="directory-v1"), \
                patch("agmem.policy.load_tokenizer", return_value=object()):
            env = make_environment(self.current, object(), policy)
        self.assertEqual(env.prompt_version, "evidence_ids_v2")

    def test_checkpoint_persists_reader_contract_with_its_config(self):
        import torch
        from agmem.pipeline import save_checkpoint

        class SavedFixture:
            def save_pretrained(self, directory, **kwargs):
                Path(directory).mkdir(parents=True, exist_ok=True)
                (Path(directory) / "fixture.json").write_text("{}")

        with tempfile.TemporaryDirectory() as tmp:
            config = deepcopy(self.current)
            config["training"]["output_dir"] = tmp
            policy = SimpleNamespace(model=SavedFixture(), tokenizer=SavedFixture(), policy_version=1)
            optimizer = SimpleNamespace(state_dict=lambda: {})
            destination = save_checkpoint(config, policy, optimizer, Path(tmp), 1, 0, 1)
            state = torch.load(destination / "trainer.pt", map_location="cpu", weights_only=True)
            self.assertEqual(state["reader_input_contract"], configured_reader_input_contract(config))
            validate_reader_input_contract(state, config)


if __name__ == "__main__":
    unittest.main()
