"""Exercise the real DAPO loop, optimizer, checkpoint files and TensorBoard."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import json
import math
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from agmem.core import Completion, Decision, Example, Trajectory
from agmem.dapo import DAPOUpdater
from agmem.grpo import CPUAdamW
from agmem.monitoring import summary_writer_class
from agmem.policy import sequence_logprobs
from agmem.training_dapo import run_training_loop


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 4)
        self.head = torch.nn.Linear(4, 9)

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
        logits = self.head(self.embedding(input_ids).cumsum(1))
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

    def save_pretrained(self, destination, **kwargs):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text('{"fixture":true}')
        torch.save(self.state_dict(), destination / "weights.pt")


class TinyTokenizer:
    def save_pretrained(self, destination):
        (Path(destination) / "tokenizer_config.json").write_text('{"fixture":true}')


def test_config(run_dir, *, candidate_groups=3, target_groups=3):
    return {"seed": 19, "reader": {"rollout_temperature": 1.0},
            "reward": {"formula": "judge_gated_v1", "likelihood": {"mode": "off"}},
            "training": {"algorithm": "dapo", "output_dir": str(run_dir), "group_size": 2,
                         "candidate_groups": candidate_groups, "target_groups": target_groups,
                         "epochs": 1, "max_num_gen_batches": 2, "filter_min_std": 1e-6,
                         "advantage_epsilon": 1e-6, "mini_batch_groups": 1,
                         "micro_batch_rollouts": 1, "ppo_epochs": 1, "kl_coefficient": 0.0,
                         "max_grad_norm": 1.0, "validation_every": 2, "checkpoint_every": 2,
                         "keep_last_checkpoints": 10, "group_infrastructure_retries": 0}}


# Pytest should not mistake the configuration factory for a standalone test.
test_config.__test__ = False


def make_components(model=None, *, version=0):
    if model is None:
        torch.manual_seed(31)
        model = TinyLM()
    policy = SimpleNamespace(model=model, tokenizer=TinyTokenizer(), device=torch.device("cpu"),
                             temperature=1.0, policy_version=version)
    return policy, CPUAdamW(model, learning_rate=0.01)


def collector(config, env, policy, agent, example, candidate_index, scorer=None, *, zero=False):
    rows = []
    for i in range(config["training"]["group_size"]):
        tokens = [3 + i] * (i + 1)
        prompt = [1, 2, 5 + int(example.sample_id.rsplit("/", 1)[-1]) % 3]
        old = sequence_logprobs(policy.model, prompt, tokens, policy.temperature, policy.device).detach().tolist()
        judge = 0.0 if zero else float(i % 2)
        rows.append(Trajectory(example.sample_id, candidate_index * 2 + i, example.memory_id,
                               policy_version=policy.policy_version,
                               decisions=[Decision([], Completion("fixture", prompt, tokens, old))],
                               termination_reason="stop", prediction=f"answer-{i}",
                               judge_score=judge, outcome=judge, answer_f1=judge,
                               outcome_reward=2 * judge - 1, reward=2 * judge - 1,
                               likelihood={"mode": "off", "d": None}))
    return rows


def validation(config, store, policy, split, destination, limit=None):
    score = policy.policy_version / 10
    values = {"reward": score, "judge_accuracy": score, "answer_f1": score}
    report = {"pooled": values, "source_macro": values, "by_source": {"fixture": values}}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(report))
    return report


class DAPOTrainingLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def examples(self, count=6):
        return [Example(f"fixture/train/{i}", "fixture", "train", f"memory-{i}", "question", gold_answers=["answer"])
                for i in range(count)]

    def run_loop(self, root, config, policy, optimizer, *, steps, state=None, collect=collector, example_count=6):
        root.mkdir(parents=True, exist_ok=True)
        updater = DAPOUpdater(policy, optimizer, config["training"])
        writer = summary_writer_class()(log_dir=str(root / "tensorboard"),
                                         purge_step=state["step"] + 1 if state else None)
        try:
            with patch("agmem.pipeline.collect_group", side_effect=collect) as sampling, \
                    patch("agmem.pipeline.evaluate_with_policy", side_effect=validation) as evaluating, \
                    redirect_stdout(StringIO()):
                run_training_loop(config, None, policy, optimizer, updater, SimpleNamespace(version="fixture-v1"),
                                  None, None, self.examples(example_count), root, writer, state=state, max_groups=steps)
        finally:
            writer.close()
        return sampling, evaluating

    def test_real_training_records_entropy_and_counts_optimizer_steps_for_cadence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            policy, optimizer = make_components()
            initial = deepcopy(policy.model.state_dict())
            sampling, evaluating = self.run_loop(root, config, policy, optimizer, steps=4)
            self.assertEqual(policy.policy_version, 4)
            self.assertTrue(any(not torch.equal(initial[key], value) for key, value in policy.model.state_dict().items()))
            # Six candidate groups were sampled; four optimizer steps were run.
            self.assertEqual(sampling.call_count, 6)
            self.assertEqual(evaluating.call_count, 2)
            self.assertEqual([call.args[4].name for call in evaluating.call_args_list], ["step-00000002", "step-00000004"])
            self.assertEqual(sorted(path.name for path in root.glob("checkpoint-*")),
                             ["checkpoint-00000002", "checkpoint-00000004"])
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            entropy = events.Scalars("train/entropy")
            self.assertEqual([point.step for point in entropy], [1, 2, 3, 4])
            self.assertTrue(all(math.isfinite(point.value) and 0 <= point.value <= math.log(9) + 1e-6 for point in entropy))
            self.assertEqual([point.step for point in events.Scalars("validation/judge_accuracy")], [2, 4])
            metrics = [json.loads(line) for line in (root / "train_metrics.jsonl").read_text().splitlines()]
            for point, report in zip(entropy, metrics):
                self.assertAlmostEqual(point.value, report["entropy"], places=6)
            state = torch.load(root / "checkpoint-00000002/trainer.pt", weights_only=True)
            self.assertEqual(state["step"], 2)
            self.assertEqual(state["dapo_state"]["active"]["consumed_groups"], 2)
            self.assertEqual(len(state["dapo_state"]["active"]["groups"]), 3)

    def test_resume_mid_batch_crosses_into_next_batch_identically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            continuous_dir, resumed_dir = root / "continuous", root / "resumed"
            continuous_config, resumed_config = test_config(continuous_dir), test_config(resumed_dir)
            policy, optimizer = make_components()
            self.run_loop(continuous_dir, continuous_config, policy, optimizer, steps=4)
            stopped_policy, stopped_optimizer = make_components()
            self.run_loop(resumed_dir, resumed_config, stopped_policy, stopped_optimizer, steps=2)
            checkpoint = resumed_dir / "checkpoint-00000002"
            state = torch.load(checkpoint / "trainer.pt", weights_only=True)
            resumed_model = TinyLM()
            resumed_model.load_state_dict(torch.load(checkpoint / "model/weights.pt", weights_only=True))
            restored_policy, restored_optimizer = make_components(resumed_model, version=state["policy_version"])
            restored_optimizer.load_state_dict(state["optimizer"])
            sampling, _ = self.run_loop(resumed_dir, resumed_config, restored_policy, restored_optimizer, steps=2, state=state)
            # First resumed group comes from the saved buffer, never re-generated.
            self.assertEqual(sampling.call_count, 3)
            self.assertEqual(restored_policy.policy_version, 4)
            for actual, expected in zip(restored_policy.model.parameters(), policy.model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual_state, expected_state = restored_optimizer.state_dict(), optimizer.state_dict()
            for actual, expected in zip(actual_state["master"], expected_state["master"]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for parameter_id, actual in actual_state["optimizer"]["state"].items():
                for name, value in actual.items():
                    torch.testing.assert_close(value, expected_state["optimizer"]["state"][parameter_id][name], rtol=0, atol=0)
            events = EventAccumulator(str(resumed_dir / "tensorboard")).Reload()
            # EventAccumulator may read same-second writer files in lexical
            # counter order (e.g. .10 before .9), not optimizer-step order.
            # Require exactly one correct scalar per step, independent of order.
            entropy = sorted(events.Scalars("train/entropy"), key=lambda point: point.step)
            self.assertEqual([point.step for point in entropy], [1, 2, 3, 4])
            actual_rows = [json.loads(line) for line in (resumed_dir / "train_metrics.jsonl").read_text().splitlines()]
            expected_rows = [json.loads(line) for line in (continuous_dir / "train_metrics.jsonl").read_text().splitlines()]
            expected_entropy = {row["step"]: row["entropy"] for row in expected_rows}
            for point in entropy:
                self.assertAlmostEqual(point.value, expected_entropy[point.step], places=6)
            for actual, expected in zip(actual_rows, expected_rows):
                for key in ("sample_id", "step", "batch_id", "behavior_policy_version", "entropy", "loss"):
                    self.assertEqual(actual[key], expected[key])

    def test_zero_reward_groups_are_audited_without_any_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            policy, optimizer = make_components()
            before = deepcopy(policy.model.state_dict())
            def zero_collector(*args, **kwargs):
                return collector(*args, **kwargs, zero=True)
            with self.assertRaisesRegex(RuntimeError, "dynamic sampling exhausted|No complete DAPO batch"):
                self.run_loop(root, config, policy, optimizer, steps=1, collect=zero_collector)
            self.assertEqual(policy.policy_version, 0)
            self.assertFalse((root / "train_metrics.jsonl").exists())
            self.assertEqual(list(root.glob("checkpoint-*")), [])
            self.assertEqual(optimizer.optimizer.state_dict()["state"], {})
            for key, value in policy.model.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            candidates = [json.loads(line) for line in (root / "candidate_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(candidates), 6)
            self.assertTrue(all(row["judge_all_wrong"] and row["zero_reward_variance"] and not row["selected"]
                                for row in candidates))
            self.assertEqual(len(list((root / "candidate_rollouts").glob("*.json"))), 6)
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertNotIn("train/entropy", events.Tags()["scalars"])

    def test_two_group_updates_resume_with_exact_counters_and_optimizer_cadence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full, resumed = root / "full", root / "resumed"
            configs = []
            for destination in (full, resumed):
                config = test_config(destination, candidate_groups=4, target_groups=4)
                config["reward"]["formula"] = "judge_cost_v2"
                config["training"].update(mini_batch_groups=2, filter_metric="judge")
                configs.append(config)
            expected_policy, expected_optimizer = make_components()
            self.run_loop(full, configs[0], expected_policy, expected_optimizer, steps=3, example_count=8)
            stopped_policy, stopped_optimizer = make_components()
            self.run_loop(resumed, configs[1], stopped_policy, stopped_optimizer, steps=1, example_count=8)
            checkpoint = resumed / "checkpoint-00000001"
            state = torch.load(checkpoint / "trainer.pt", weights_only=True)
            self.assertEqual(state["dapo_state"]["active"]["consumed_groups"], 2)
            self.assertEqual(state["dapo_state"]["counters"], {
                "trained_groups": 2, "update_rollouts": 4, "training_tokens": 6})
            model = TinyLM()
            model.load_state_dict(torch.load(checkpoint / "model/weights.pt", weights_only=True))
            policy, optimizer = make_components(model, version=state["policy_version"])
            optimizer.load_state_dict(state["optimizer"])
            sampling, evaluating = self.run_loop(resumed, configs[1], policy, optimizer,
                                                  steps=2, state=state, example_count=8)
            self.assertEqual(sampling.call_count, 4)  # Pending mini-batch reused, then a fresh batch.
            self.assertEqual(evaluating.call_count, 1)
            self.assertEqual(evaluating.call_args.args[4].name, "step-00000002")
            self.assertEqual(policy.policy_version, 3)
            for actual, expected in zip(policy.model.parameters(), expected_policy.model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            reports = [json.loads(line) for line in (resumed / "train_metrics.jsonl").read_text().splitlines()]
            for i, report in enumerate(reports, 1):
                self.assertEqual(report["step"], i)
                self.assertEqual(report["cumulative_trained_groups"], 2 * i)
                self.assertEqual(report["cumulative_update_rollouts"], 4 * i)
                self.assertEqual(report["cumulative_training_tokens"], 6 * i)
                self.assertEqual(len(report["sample_ids"]), 2)
                self.assertEqual(report["groups_per_update"], 2)
            trace = json.loads((resumed / "rollouts/group-00000002.json").read_text())
            self.assertEqual(len(trace["rollouts"]), 4)
            self.assertEqual(len({row["sample_id"] for row in trace["rollouts"]}), 2)
            final_state = torch.load(resumed / "checkpoint-00000003/trainer.pt", weights_only=True)
            self.assertEqual(final_state["dapo_state"]["active"]["consumed_groups"], 2)
            self.assertEqual(final_state["dapo_state"]["counters"]["trained_groups"], 6)

    def test_task_truncation_is_trained_logged_and_saved_in_pending_checkpoint_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            policy, optimizer = make_components()
            def truncated_task_collector(*args, **kwargs):
                group = collector(*args, **kwargs)
                group[0].task_completion = {"truncated": True, "finish_reason": "length",
                                            "max_tokens": 512, "usage": {"completion_tokens": 512}}
                return group
            self.run_loop(root, config, policy, optimizer, steps=1, collect=truncated_task_collector)
            self.assertEqual(policy.policy_version, 1)
            report = json.loads((root / "train_metrics.jsonl").read_text())
            self.assertEqual(report["task_truncation_rate"], 0.5)
            self.assertEqual(report["masked_rollouts"], 0)
            self.assertEqual(report["training_tokens"], 3)
            candidate = json.loads((root / "candidate_rollouts/candidate-00000000.json").read_text())
            self.assertTrue(candidate["rollouts"][0]["task_completion"]["truncated"])
            self.assertEqual(candidate["sampling"]["task_truncated_rollout_fraction"], 0.5)
            self.assertEqual(candidate["sampling"]["truncated_rollout_fraction"], 0.0)
            state = torch.load(root / "checkpoint-00000001/trainer.pt", weights_only=True)
            self.assertTrue(state["dapo_state"]["active"]["groups"][1][0]["task_completion"]["truncated"])
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertEqual(events.Scalars("train/task_truncation_rate")[0].value, 0.5)

    def test_scoring_service_failure_exits_training_after_group_retry_without_update(self):
        from agmem.core import InfrastructureError
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            config["training"]["group_infrastructure_retries"] = 1
            policy, optimizer = make_components()
            calls = []
            def failed_collector(*args, **kwargs):
                calls.append(1)
                raise InfrastructureError("judge service failed; no task score assigned")
            with self.assertRaisesRegex(InfrastructureError, "judge service failed"):
                self.run_loop(root, config, policy, optimizer, steps=1, collect=failed_collector)
            self.assertEqual(len(calls), 2)
            self.assertEqual(policy.policy_version, 0)
            self.assertFalse((root / "train_metrics.jsonl").exists())
            self.assertEqual(list(root.glob("checkpoint-*")), [])
            failures = [json.loads(line) for line in (root / "infrastructure_failures.jsonl").read_text().splitlines()]
            self.assertEqual([row["attempt"] for row in failures], [0, 1])


if __name__ == "__main__":
    unittest.main()
