"""Exercise hierarchical ORM configuration, Task parsing and actual DAPO updates."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from agmem.core import Example, load_config
from agmem.dapo import DAPOUpdater
from agmem.pipeline import make_reward_scorer
from agmem.prompts import configured_reader_input_contract, task_input_contract, task_messages
from agmem.services import TaskAgent
from agmem.training_dapo import run_training_loop
from tests.test_training_dapo import collector, make_components


class HierarchicalTrainingIntegrationTests(unittest.TestCase):
    def config(self):
        return load_config("configs/train_dapo_hierarchical_f1.yaml")

    def test_configuration_and_no_judge_service(self):
        config = self.config()
        self.assertEqual(config["retrieval"], {"id_protocol": "strict_v2", "max_ids_per_call": 1,
                          "max_read_calls": 10, "max_open_directories": 2})
        self.assertEqual(config["training"]["target_groups"], 8)
        self.assertEqual(config["training"]["mini_batch_groups"], 8)
        self.assertEqual(config["training"]["epochs"], 3)
        self.assertEqual(config["training"]["validation_every"], 4)
        self.assertEqual(config["training"]["checkpoint_every"], 0)
        self.assertFalse(config["training"]["process_reward"]["enabled"])
        self.assertEqual(config["reward"]["read_call_cost"], .05)
        with patch("agmem.judges.JudgeScorer", side_effect=AssertionError("Judge must not be created")):
            scorer = make_reward_scorer(config, None)
        self.assertIsNone(scorer.judge)
        scorer.close()
        # These protocols must stay separate from existing flat experiments.
        legacy = load_config("configs/train_dapo_reader_loop_v3_query9010_format05.yaml")
        self.assertEqual(legacy["reward"]["formula"], "judge_cost_v2")
        self.assertNotEqual(configured_reader_input_contract(config), configured_reader_input_contract(legacy))

    def test_task_partial_answer_uses_segment_id_summary_and_public_observation(self):
        example = Example("locomo/train/0", "locomo", "train", "m", "Where?",
                          observation="Public timestamp", gold_answers=["private gold"])
        evidence = [{"id": "p0001:c0002", "parent_id": "p0001", "content": "Selected summary",
                     "raw_text": "User: I live in Paris.", "time": "2020-01-01"}]
        class Service:
            def complete(self, messages, max_tokens, *, metadata, answer_policy):
                metadata.update(finish_reason="length", truncated=True, max_tokens=max_tokens)
                return "internal explanation<answer>Paris"
        agent = TaskAgent(Service(), 2048, "hierarchical_evidence_v1")
        metadata = {}
        answer, _ = agent.answer(example, evidence, metadata=metadata)
        self.assertEqual(answer, "Paris")
        prompt = json.dumps(task_messages(example, evidence, "hierarchical_evidence_v1"))
        for value in ("p0001:c0002", "Selected summary", "User: I live in Paris.", "Public timestamp"):
            self.assertIn(value, prompt)
        self.assertNotIn("private gold", prompt)
        self.assertEqual(task_input_contract("hierarchical_evidence_v1")["answer_parser"], "answer_tag_partial_v1")

    def test_two_updates_resample_new_policy_and_record_f1_primary(self):
        from agmem.checkpoints import best_checkpoint
        from agmem.rewards import score_trajectory
        from agmem.monitoring import summary_writer_class
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = self.config()
                config["execution"].update(sampling_group_concurrency=1)
                config["training"].update(output_dir=str(root), group_size=2, mini_batch_groups=1,
                    target_groups=1, candidate_groups=1, max_num_gen_batches=1, validation_every=1,
                    epochs=1, micro_batch_rollouts=1, keep_last_checkpoints=3)
                examples = [Example(f"locomo/train/{i}", "locomo", "train", "m", "Question",
                                    gold_answers=["Paris"]) for i in range(2)]
                policy, optimizer = make_components()
                seen_versions = []
                def collect(config, env, policy, agent, example, candidate_index, scorer=None):
                    seen_versions.append(policy.policy_version)
                    rows = collector(config, env, policy, agent, example, candidate_index)
                    for i, row in enumerate(rows):
                        row.prediction = "Paris" if i else "London"
                        row.n_read_calls = 2
                        score_trajectory(example, row, config["reward"], max_ids=1)
                    return rows
                def validate(config, store, policy, split, output, limit=None):
                    # Reward and F1 disagree: primary must follow F1.
                    f1 = .8 if policy.policy_version == 1 else .6
                    values = {"outcome_f1": f1, "answer_f1": f1, "reward": policy.policy_version / 10}
                    report = {"pooled": values, "source_macro": values,
                              "by_source": {"locomo": values}, "primary_outcome_metric": "outcome_f1"}
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "metrics.json").write_text(json.dumps(report))
                    return report
                writer = summary_writer_class()(log_dir=str(root / "tensorboard"))
                try:
                    with patch("agmem.pipeline.collect_group", side_effect=collect), \
                         patch("agmem.pipeline.evaluate_with_policy", side_effect=validate), \
                         redirect_stdout(StringIO()):
                        run_training_loop(config, None, policy, optimizer,
                            DAPOUpdater(policy, optimizer, config["training"]), SimpleNamespace(version="frozen"),
                            None, None, examples, root, writer, max_groups=2)
                finally:
                    writer.close()
                self.assertEqual(seen_versions, [0, 1])
                self.assertEqual(policy.policy_version, 2)
                self.assertEqual(best_checkpoint(root).name, "checkpoint-00000001")
                reports = [json.loads(line) for line in (root / "train_metrics.jsonl").read_text().splitlines()]
                self.assertTrue(all(row["answer_f1"] == .5 for row in reports))
                self.assertTrue(all("judge_accuracy" not in row for row in reports))
                events = EventAccumulator(str(root / "tensorboard")).Reload()
                self.assertEqual(len(events.Scalars("validation/outcome_f1")), 2)
                self.assertNotIn("train/judge_accuracy", events.Tags()["scalars"])
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
