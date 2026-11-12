from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import gzip
import json
import math
import tempfile
import unittest
from unittest.mock import patch

import torch

from agmem.core import Completion, Decision, Example, InfrastructureError, Trajectory, load_config
from agmem.dapo import DAPOUpdater
from agmem.policy import sequence_logprobs
from agmem.process_rewards import (ProcessAttributionScorer, action_weights, attribution_settings,
                                   evidence_prefixes, process_metrics)
from agmem.training_dapo import run_training_loop
from tests.test_training_dapo import make_components, test_config


class PrefixScorer:
    def __init__(self):
        self.calls = []

    def prefix_provenance(self):
        return {"fixture": "frozen-prefix"}

    def score_prefixes(self, example, prefixes):
        self.calls.append(deepcopy(prefixes))
        return {"prefix_logprobs": [-2.0 + len(prefix) * .5 for prefix in prefixes],
                "reference_token_count": 2, "references": []}

    def close(self):
        pass


def configuration(root="unused"):
    value = test_config(root, candidate_groups=1, target_groups=1)
    value["reader"]["prompt_version"] = "interaction_history_v4"
    value["retrieval"] = {"id_protocol": "strict_v2", "max_ids_per_call": 2, "max_read_calls": 8}
    value["task_agent"] = {"prompt_version": "history_answer_query_v1"}
    value["reward"].update(formula="judge_cost_v2", read_call_cost=.01, format_penalty=.5, format_penalty_mode="per_id")
    value["training"].update(validation_every=0, filter_metric="judge_or_all_correct_format",
                             max_num_gen_batches=0, process_reward={"enabled": True, "normalization": "rms"})
    return value


def trajectory(policy, example, seed=0, judge=1.0, *, repeat=False):
    record = {"id": "A", "raw_text": "historical fact", "time": "2020", "order": 0,
              "source_session_id": "s", "content": "directory summary"}
    row = Trajectory(example.sample_id, seed, example.memory_id, policy_version=policy.policy_version,
                     evidence=[record], n_read_calls=2 if repeat else 1, termination_reason="stop",
                     prediction="answer", outcome=judge, outcome_reward=2 * judge - 1,
                     reward=2 * judge - 1 - .01, judge_score=judge, answer_f1=judge)
    actions = [{"ids": ["A"], "stop": False}]
    if repeat:
        actions.append({"ids": ["A"], "stop": False})
    actions.append({"ids": [], "stop": True})
    for index, action in enumerate(actions):
        prompt, target = [1, 2] + [5] * index, [3 + index, 4]
        with torch.no_grad():
            old = sequence_logprobs(policy.model, prompt, target, policy.temperature, policy.device).tolist()
        result = {"records": [record] if index == 0 else []} if not action["stop"] else None
        row.decisions.append(Decision([], Completion(json.dumps(action), prompt, target, old),
                                      action=action, tool_result=result))
    return row


class ProcessRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.policy, self.optimizer = make_components()
        self.example = Example("fixture/train/0", "fixture", "train", "memory-0", "question", gold_answers=["answer"])

    def test_prefixes_deduplicate_and_never_use_future_records(self):
        row = trajectory(self.policy, self.example, repeat=True)
        prefixes = evidence_prefixes(row)
        self.assertEqual([len(value) for value in prefixes], [0, 1, 1, 1])
        self.assertEqual(prefixes[0], [])
        row.evidence = []
        with self.assertRaisesRegex(ValueError, "final evidence"):
            evidence_prefixes(row)

    def test_real_frozen_reader_scores_exact_next_tokens_under_previous_prompt(self):
        row = trajectory(self.policy, self.example, repeat=True)
        likelihood = PrefixScorer()
        scorer = ProcessAttributionScorer(configuration(), likelihood)
        saved = deepcopy(self.policy.model.state_dict())
        scorer.score_group(self.example, [row], self.policy)
        entries = row.process_attribution["per_decision"]
        for index in (0, 1):
            with torch.no_grad():
                expected = sequence_logprobs(self.policy.model, row.decisions[index].completion.prompt_ids,
                    row.decisions[index + 1].completion.token_ids, self.policy.temperature, self.policy.device).mean().item()
            self.assertEqual(entries[index]["ell_minus"], expected)
            self.assertAlmostEqual(entries[index]["ell_plus"],
                                   sum(row.decisions[index + 1].completion.old_logprobs) / 2)
        self.assertEqual([entry["g"] for entry in entries], [.5, 0, 0])
        self.assertEqual(entries[1]["reward"], 0)
        self.assertEqual(entries[2]["w"], 1)
        self.assertEqual(row.reward, .99)
        self.assertEqual(self.policy.policy_version, 0)
        for name, tensor in self.policy.model.state_dict().items():
            torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in self.policy.model.parameters()))
        with self.assertRaisesRegex(ValueError, "rescore"):
            scorer.score_group(self.example, [row], self.policy)

    def test_stable_clipped_weights_and_missing_transition(self):
        self.assertEqual(action_weights([None, None], .5, 2), [1, 1])
        for actual, expected in zip(action_weights([1000, -1000, None], .5, 2), [2, .5, 1]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(action_weights([math.log(2), 0], .5, 2)[0], 4 / 3)
        self.assertEqual(action_weights([9, None], .5, 2), [1, 1])
        with self.assertRaises(ValueError):
            action_weights([float("nan")], .5, 2)

    def test_failure_does_not_assign_partial_process_or_change_terminal_reward(self):
        rows = [trajectory(self.policy, self.example, index) for index in range(2)]
        scorer = ProcessAttributionScorer(configuration(), PrefixScorer(),
            action_logprob=lambda *_: float("nan"))
        with self.assertRaises(InfrastructureError):
            scorer.score_group(self.example, rows, self.policy)
        self.assertTrue(all(row.process_attribution == {} for row in rows))
        self.assertEqual([row.reward for row in rows], [.99, .99])

    def test_stale_actor_rejected_before_likelihood_requests(self):
        row = trajectory(self.policy, self.example)
        likelihood = PrefixScorer()
        scorer = ProcessAttributionScorer(configuration(), likelihood)
        self.policy.policy_version += 1
        with self.assertRaisesRegex(ValueError, "frozen sampling"):
            scorer.score_group(self.example, [row], self.policy)
        self.assertFalse(likelihood.calls)

    def test_monitoring_counts_wrong_and_bad_actions_with_positive_advantages(self):
        from agmem.process_advantages import assign_process_advantages
        rows = [trajectory(self.policy, self.example, i, float(i)) for i in range(2)]
        rows[0].advantage, rows[1].advantage = -1, 1
        rows[0].decisions[0].format_violations = ["repeated_read"]
        config = configuration()
        config["training"]["process_reward"]["alpha"] = 2
        ProcessAttributionScorer(config, PrefixScorer()).score_group(self.example, rows, self.policy)
        assign_process_advantages(rows, config["training"])
        metrics = process_metrics(rows)
        self.assertEqual(metrics["process_wrong_read_steps"], 1)
        self.assertEqual(metrics["process_wrong_read_positive_advantage_rate"], 1)
        self.assertEqual(metrics["process_bad_action_positive_advantage_rate"], 1)
        rows[0].decisions[0].completion.truncated = True
        self.assertEqual(process_metrics(rows)["process_wrong_read_steps"], 0)

    def test_new_recipe_and_fresh_sampling_validation(self):
        config = load_config("configs/train_dapo_directory_process_v1.yaml")
        self.assertTrue(attribution_settings(config)["enabled"])
        config["training"]["target_groups"] = 16
        with self.assertRaisesRegex(ValueError, "fresh sampling"):
            attribution_settings(config)

    def test_dapo_loop_filters_before_process_then_saves_true_step_advantages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = configuration(root)
            examples = [self.example, Example("fixture/train/1", "fixture", "train", "memory-1", "question", gold_answers=["answer"])]
            likelihood = PrefixScorer()
            scorer = ProcessAttributionScorer(config, likelihood)
            updater = DAPOUpdater(self.policy, self.optimizer, config["training"])
            def collect(config, env, policy, agent, example, candidate_index, scorer=None):
                return [trajectory(policy, example, i, float(i == 1 and candidate_index > 0)) for i in range(2)]
            with patch("agmem.pipeline.collect_group", side_effect=collect), redirect_stdout(StringIO()):
                run_training_loop(config, None, self.policy, self.optimizer, updater,
                    SimpleNamespace(version="fixture-v1"), None, None, examples, root,
                    max_groups=1, process_scorer=scorer)
            self.assertEqual(len(likelihood.calls), 2)  # rejected all-wrong group never process-scored
            self.assertEqual(self.policy.policy_version, 1)
            with gzip.open(root / "groups/group-00000000.json.gz", "rt") as stream:
                group = json.load(stream)["rollouts"]
            for row in group:
                entries = row["process_attribution"]["per_decision"]
                self.assertGreater(entries[0]["advantage"], row["advantage"])
                self.assertEqual(entries[1]["advantage"], row["advantage"])
                self.assertAlmostEqual(row["reward"], row["outcome_reward"] - .01)
            report = json.loads((root / "train_metrics.jsonl").read_text())
            self.assertAlmostEqual(report["reward"], -.01)
            self.assertIn("process_reward_mean", report)
            state = torch.load(root / "checkpoint-00000001/trainer.pt", weights_only=True)
            self.assertEqual(state["task_input_contract"]["task_context_version"], "query_only_v1")
            self.assertEqual(state["reader_input_contract"]["prompt_version"], "interaction_history_v4")
            compact = json.loads((root / "rollouts/group-00000000.json").read_text())
            self.assertIn("process_attribution", compact["rollouts"][0])

    def test_all_correct_terminal_tie_uses_process_signal_in_real_loop(self):
        class DifferentEvidenceScorer(PrefixScorer):
            def score_prefixes(self, example, prefixes):
                result = super().score_prefixes(example, prefixes)
                result["prefix_logprobs"] = [
                    -2.0 + sum(.5 if record["id"] == "A" else .1 for record in prefix)
                    for prefix in prefixes]
                return result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = configuration(root)
            config["training"]["filter_metric"] = "exclude_all_wrong"
            config["training"]["process_reward"]["normalization"] = "group_standardize"
            updater = DAPOUpdater(self.policy, self.optimizer, config["training"])
            scorer = ProcessAttributionScorer(config, DifferentEvidenceScorer())
            saved = deepcopy(self.policy.model.state_dict())

            def collect(config, env, policy, agent, example, candidate_index, scorer=None):
                rows = [trajectory(policy, example, i) for i in range(2)]
                # Both correct, one READ, no format cost difference. Evidence
                # quality varies, so only the new process term can distinguish.
                row = rows[1]
                row.evidence[0]["id"] = "B"
                decision = row.decisions[0]
                decision.action["ids"] = ["B"]
                decision.completion.text = json.dumps(decision.action)
                decision.completion.token_ids = [6, 4]
                with torch.no_grad():
                    decision.completion.old_logprobs = sequence_logprobs(
                        policy.model, decision.completion.prompt_ids, decision.completion.token_ids,
                        policy.temperature, policy.device).tolist()
                return rows

            with patch("agmem.pipeline.collect_group", side_effect=collect), redirect_stdout(StringIO()):
                run_training_loop(config, None, self.policy, self.optimizer, updater,
                    SimpleNamespace(version="fixture-v1"), None, None, [self.example], root,
                    max_groups=1, process_scorer=scorer)
            self.assertEqual(self.policy.policy_version, 1)
            self.assertTrue(any(not torch.equal(value, saved[name])
                                for name, value in self.policy.model.state_dict().items()))
            report = json.loads((root / "train_metrics.jsonl").read_text())
            self.assertEqual(report["process_zero_terminal_groups"], 1)
            self.assertEqual(report["process_zero_combined_advantage_groups"], 0)
            self.assertFalse(report["optimizer_step_skipped"])
            with gzip.open(root / "groups/group-00000000.json.gz", "rt") as stream:
                rows = json.load(stream)["rollouts"]
            self.assertEqual([row["advantage"] for row in rows], [0, 0])
            advantages = [row["process_attribution"]["per_decision"][0]["advantage"] for row in rows]
            self.assertGreater(advantages[0], 0)
            self.assertLess(advantages[1], 0)

    def test_zero_signal_batches_are_audited_without_counting_optimizer_steps(self):
        for followed_by_update in (False, True):
            with self.subTest(followed_by_update=followed_by_update), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = configuration(root)
                config["training"]["filter_metric"] = "exclude_all_wrong"
                config["training"]["process_reward"]["normalization"] = "group_standardize"
                policy, optimizer = make_components()
                updater = DAPOUpdater(policy, optimizer, config["training"])
                scorer = ProcessAttributionScorer(config, PrefixScorer())
                examples = [self.example, Example("fixture/train/1", "fixture", "train", "memory-1",
                                                  "question", gold_answers=["answer"])]
                if followed_by_update:
                    examples.append(Example("fixture/train/2", "fixture", "train", "memory-2",
                                            "question", gold_answers=["answer"]))

                def collect(config, env, policy, agent, example, candidate_index, scorer=None):
                    return [trajectory(policy, example, i,
                        float(i == 1 or candidate_index < 2)) for i in range(2)]

                with patch("agmem.pipeline.collect_group", side_effect=collect), \
                        patch.object(optimizer, "step", wraps=optimizer.step) as stepping, \
                        redirect_stdout(StringIO()):
                    run_training_loop(config, None, policy, optimizer, updater,
                        SimpleNamespace(version="fixture-v1"), None, None, examples, root,
                        max_groups=1, process_scorer=scorer)
                expected_step = int(followed_by_update)
                self.assertEqual(stepping.call_count, expected_step)
                self.assertEqual(policy.policy_version, expected_step)
                skipped = [json.loads(line) for line in (root / "skipped_updates.jsonl").read_text().splitlines()]
                self.assertEqual([row["skip_index"] for row in skipped], [1, 2])
                self.assertTrue(all(row["step"] == 0 and row["training_tokens"] == 0 for row in skipped))
                for row in skipped:
                    with gzip.open(Path(row["trace_root"]) / "groups/group-00000000.json.gz", "rt") as stream:
                        traces = json.load(stream)["rollouts"]
                    self.assertTrue(all(entry["advantage"] == 0 for trace in traces
                                        for entry in trace["process_attribution"]["per_decision"]))
                state = torch.load(root / f"checkpoint-{expected_step:08d}/trainer.pt", weights_only=True)
                counters = state["dapo_state"]["counters"]
                self.assertEqual(counters["skipped_zero_signal_batches"], 2)
                self.assertEqual(counters["zero_signal_groups"], 2)
                self.assertEqual(counters["trained_groups"], expected_step)
                self.assertIsNone(state["dapo_state"]["active"])
                self.assertEqual(state["dapo_state"]["sampler"]["candidate_index"], len(examples))
                if followed_by_update:
                    report = json.loads((root / "train_metrics.jsonl").read_text())
                    self.assertEqual(report["step"], 1)
                    self.assertEqual(report["cumulative_trained_groups"], 1)
                else:
                    self.assertFalse((root / "train_metrics.jsonl").exists())
                    self.assertFalse((root / "groups").exists())


if __name__ == "__main__":
    unittest.main()
