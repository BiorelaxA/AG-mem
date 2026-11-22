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
from agmem.monitoring import replay_tensorboard, summary_writer_class
from agmem.policy import sequence_logprobs
from agmem.training_dapo import run_training_loop


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 4)
        self.head = torch.nn.Linear(4, 9)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, logits_to_keep=0):
        values = self.embedding(input_ids)
        if attention_mask is not None:
            values = values * attention_mask.unsqueeze(-1)
        logits = self.head(values.cumsum(1))
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
        tokens = [3 + i % 6] * (i + 1)
        prompt = [1, 2, 5 + int(example.sample_id.rsplit("/", 1)[-1]) % 3]
        old = sequence_logprobs(policy.model, prompt, tokens, policy.temperature, policy.device).detach().tolist()
        judge = 0.0 if zero else float(i % 2)
        rows.append(Trajectory(example.sample_id, candidate_index * config["training"]["group_size"] + i, example.memory_id,
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

    def run_loop(self, root, config, policy, optimizer, *, steps, state=None, collect=collector, example_count=6,
                 stream=None):
        root.mkdir(parents=True, exist_ok=True)
        updater = DAPOUpdater(policy, optimizer, config["training"])
        writer = summary_writer_class()(log_dir=str(root / "tensorboard"),
                                         purge_step=0 if state else None)
        try:
            if state:
                replay_tensorboard(writer, root)
            with patch("agmem.pipeline.collect_group", side_effect=collect) as sampling, \
                    patch("agmem.rollout_pipeline.collect_candidate_groups", side_effect=stream), \
                    patch("agmem.pipeline.evaluate_with_policy", side_effect=validation) as evaluating, \
                    redirect_stdout(StringIO()):
                run_training_loop(config, None, policy, optimizer, updater, SimpleNamespace(version="fixture-v1"),
                                  None, None, self.examples(example_count), root, writer, state=state, max_groups=steps)
        finally:
            writer.close()
        # Fast test resumes can create files in the same second: TensorBoard's
        # unpadded writer IDs sort .10 before .9, replaying the purge out of order.
        # Normalize only these closed temporary fixtures before reading events.
        for path in (root / "tensorboard").glob("events.out.tfevents.*"):
            prefix, uid = path.name.rsplit(".", 1)
            if uid.isdigit():
                ordered = path.with_name(f"{prefix}.{int(uid):020d}")
                if ordered != path:
                    path.rename(ordered)
        return sampling, evaluating

    def test_cumulative_format_reward_survives_sampling_update_checkpoint_and_metrics(self):
        from agmem.pipeline import metrics
        from agmem.rewards import score_trajectory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            config["reward"].update(formula="judge_cost_v2", read_call_cost=.01,
                                    format_penalty=.1, format_penalty_mode="per_decision")
            config["training"].update(group_size=8, micro_batch_rollouts=2, filter_metric="judge")
            scored_groups = []

            def collect(config, env, policy, agent, example, candidate_index, scorer=None):
                group = collector(config, env, policy, agent, example, candidate_index)
                for index, row in enumerate(group):
                    first = row.decisions[0]
                    first.action = {"ids": ["A"], "stop": False}
                    first.tool_result = {"new_ids": ["A"], "read_call_consumed": True}
                    for _ in range(index % 4):
                        repeated = deepcopy(first)
                        # Multiple labels in one action must still count only once.
                        repeated.format_violations = ["repeated_read", "duplicate_ids"]
                        repeated.action = {"ids": ["A", "A"], "stop": False}
                        repeated.tool_result = {"already_read_ids": ["A"], "new_ids": [],
                                                "no_new_evidence": True, "read_call_consumed": True,
                                                "read_calls_used": len(row.decisions) + 1,
                                                "read_calls_remaining": 8 - (len(row.decisions) + 1)}
                        row.decisions.append(repeated)
                    row.n_read_calls = len(row.decisions)
                    judge = SimpleNamespace(score=lambda *_, value=row.judge_score: value)
                    score_trajectory(example, row, config["reward"], judge=judge)
                scored_groups.append(group)
                return group

            policy, optimizer = make_components()
            self.run_loop(root, config, policy, optimizer, steps=1, collect=collect)
            self.assertEqual(policy.policy_version, 1)
            train = json.loads((root / "train_metrics.jsonl").read_text().splitlines()[0])
            sampling = json.loads((root / "sampling_metrics.jsonl").read_text().splitlines()[0])
            candidate = json.loads((root / "candidate_metrics.jsonl").read_text().splitlines()[0])
            evaluation = metrics(scored_groups[0])
            for report in (train, sampling, candidate, evaluation["pooled"],
                           evaluation["source_macro"], evaluation["by_source"]["fixture"]):
                self.assertAlmostEqual(report["format_violation_count"], 1.5)
                self.assertAlmostEqual(report["format_reward"], -.15)
                self.assertAlmostEqual(report["reward"], -.175)
                self.assertEqual(report["format_failure_rate"], .75)
            compact = json.loads((root / "rollouts/group-00000000.json").read_text())
            candidates = json.loads((root / "candidate_rollouts/candidate-00000000.json").read_text())
            for trace in (compact, candidates):
                self.assertEqual([r["format_violation_count"] for r in trace["rollouts"]], [0, 1, 2, 3] * 2)
                self.assertAlmostEqual(trace["rollouts"][3]["format_reward"], -.3)
            state = torch.load(root / "checkpoint-00000001/trainer.pt", weights_only=True)
            self.assertEqual(state["config"]["reward"]["format_penalty_mode"], "per_decision")
            saved = state["dapo_state"]["active"]["groups"][1]
            self.assertEqual([r["format_violation_count"] for r in saved], [0, 1, 2, 3] * 2)
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            for prefix in ("train", "sampling"):
                self.assertAlmostEqual(events.Scalars(f"{prefix}/format_violation_count")[0].value, 1.5)
                self.assertAlmostEqual(events.Scalars(f"{prefix}/format_reward")[0].value, -.15)

    def test_all_correct_format_groups_reach_optimizer_checkpoint_and_tensorboard(self):
        from agmem.rewards import score_trajectory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root, candidate_groups=2, target_groups=2)
            config["reward"].update(formula="judge_cost_v2", read_call_cost=.01,
                                    format_penalty=.5, format_penalty_mode="per_id")
            config["training"]["filter_metric"] = "judge_or_all_correct_format"

            def collect(config, env, policy, agent, example, candidate_index, scorer=None):
                group = collector(config, env, policy, agent, example, candidate_index)
                for index, row in enumerate(group):
                    first = row.decisions[0]
                    first.action = {"ids": ["A"], "stop": False}
                    first.tool_result = {"records": [{"id": "A"}], "new_ids": ["A"], "read_call_consumed": True}
                    if index:
                        repeated = deepcopy(first)
                        repeated.format_violations = ["repeated_read"]
                        repeated.tool_result = {"already_read_ids": ["A"], "records": [], "new_ids": [],
                                                "no_new_evidence": True, "read_call_consumed": True,
                                                "read_calls_used": 2, "read_calls_remaining": 6}
                        row.decisions.append(repeated)
                    row.n_read_calls = len(row.decisions)
                    score_trajectory(example, row, config["reward"], max_ids=1,
                                     judge=SimpleNamespace(score=lambda *_: 1))
                return group

            policy, optimizer = make_components()
            before = deepcopy(policy.model.state_dict())
            self.run_loop(root, config, policy, optimizer, steps=1, collect=collect)
            self.assertEqual(policy.policy_version, 1)
            self.assertTrue(any(not torch.equal(before[key], value)
                                for key, value in policy.model.state_dict().items()))
            train = json.loads((root / "train_metrics.jsonl").read_text().splitlines()[0])
            self.assertEqual(train["judge_accuracy"], 1)
            self.assertAlmostEqual(train["format_reward"], -.25)
            self.assertAlmostEqual(train["reward"], .735)
            candidates = [json.loads(line) for line in (root / "candidate_metrics.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["selected"] and row["accepted_all_correct_format"] for row in candidates))
            state = torch.load(root / "checkpoint-00000001/trainer.pt", weights_only=True)
            self.assertEqual(state["config"]["training"]["filter_metric"], "judge_or_all_correct_format")
            saved = state["dapo_state"]["active"]["groups"][1]
            self.assertEqual([row["format_reward"] for row in saved], [0, -.5])
            self.assertGreater(saved[0]["advantage"], 0)
            self.assertLess(saved[1]["advantage"], 0)
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertEqual(events.Scalars("sampling/accepted_all_correct_format_rate")[0].value, 1)

    def test_parallel_sampling_drains_before_micro2_updates_and_resumes_saved_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            serial_dir, parallel_dir = root / "serial", root / "parallel"
            configs = []
            for destination in (serial_dir, parallel_dir):
                config = test_config(destination, candidate_groups=4, target_groups=4)
                config["reward"]["formula"] = "judge_cost_v2"
                config["training"].update(group_size=8, mini_batch_groups=1,
                                           micro_batch_rollouts=2, filter_metric="judge")
                configs.append(config)
            configs[1]["execution"] = {"sampling_group_concurrency": 4, "reader_batch_size": 4,
                                       "task_agent_workers": 2}
            expected_policy, expected_optimizer = make_components()
            self.run_loop(serial_dir, configs[0], expected_policy, expected_optimizer, steps=5, example_count=8)
            phases, sampling_active = [], False

            def stream(config, env, policy, agent, scorer, candidates, *, on_failure, on_progress):
                nonlocal sampling_active
                self.assertFalse(sampling_active)
                sampling_active = True
                version = policy.policy_version
                phases.append(version)
                try:
                    count = 0
                    for example, candidate_id in candidates:
                        self.assertEqual(policy.policy_version, version)
                        yield example, candidate_id, collector(config, env, policy, agent, example, candidate_id)
                        count += 1
                    on_progress({"event": "sampling_window_completed", "groups_completed": count,
                                 "peak_groups": 4, "reader_seconds": 0.5, "scoring_wait_seconds": 0.25})
                finally:
                    self.assertEqual(policy.policy_version, version)
                    sampling_active = False

            def run_parallel(policy, optimizer, steps, state=None):
                original_step = optimizer.step
                def checked_step(*args, **kwargs):
                    self.assertFalse(sampling_active, "Optimizer must wait for every scoring request")
                    return original_step(*args, **kwargs)
                with patch.object(optimizer, "step", side_effect=checked_step):
                    sampling, _ = self.run_loop(parallel_dir, configs[1], policy, optimizer, steps=steps,
                                                state=state, example_count=8, stream=stream)
                self.assertEqual(sampling.call_count, 0)

            stopped_policy, stopped_optimizer = make_components()
            run_parallel(stopped_policy, stopped_optimizer, 2)
            checkpoint = parallel_dir / "checkpoint-00000002"
            state = torch.load(checkpoint / "trainer.pt", weights_only=True)
            self.assertEqual(state["dapo_state"]["active"]["consumed_groups"], 2)
            self.assertEqual([len(group) for group in state["dapo_state"]["active"]["groups"]], [8] * 4)
            model = TinyLM()
            model.load_state_dict(torch.load(checkpoint / "model/weights.pt", weights_only=True))
            policy, optimizer = make_components(model, version=state["policy_version"])
            optimizer.load_state_dict(state["optimizer"])
            run_parallel(policy, optimizer, 3, state=state)
            # The remaining two groups use the saved batch; new sampling starts only at version 4.
            self.assertEqual(phases, [0, 4])
            for actual, expected in zip(policy.model.parameters(), expected_policy.model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            reports = [json.loads(line) for line in (parallel_dir / "train_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([report["step"] for report in reports], [1, 2, 3, 4, 5])
            self.assertTrue(all(report["update_rollouts"] == 8 and report["groups_per_update"] == 1
                                and report["micro_batches"] == 4 for report in reports))
            sampling_reports = [json.loads(line) for line in (parallel_dir / "sampling_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([report["pipeline_round_groups_completed"] for report in sampling_reports], [4, 4])
            events = EventAccumulator(str(parallel_dir / "tensorboard")).Reload()
            self.assertEqual(len(events.Scalars("sampling/pipeline_round_reader_seconds")), 2)
            self.assertEqual(len(events.Scalars("train/entropy")), 5)

    def test_parallel_scoring_failure_drains_and_never_updates_partial_batch(self):
        from agmem.core import InfrastructureError
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = test_config(root)
            config["execution"] = {"sampling_group_concurrency": 4}
            policy, optimizer = make_components()
            drained = []
            def stream(config, env, policy, agent, scorer, candidates, *, on_failure, on_progress):
                try:
                    example, candidate_id = next(candidates)
                    yield example, candidate_id, collector(config, env, policy, agent, example, candidate_id)
                    example, candidate_id = next(candidates)
                    error = InfrastructureError("judge service failed")
                    on_failure(example, candidate_id, 0, error)
                    raise error
                finally:
                    drained.append(True)
            with self.assertRaisesRegex(InfrastructureError, "judge service failed"):
                self.run_loop(root, config, policy, optimizer, steps=1, stream=stream)
            self.assertEqual(drained, [True])
            self.assertEqual(policy.policy_version, 0)
            self.assertEqual(optimizer.optimizer.state_dict()["state"], {})
            self.assertFalse((root / "train_metrics.jsonl").exists())
            self.assertEqual(list(root.glob("checkpoint-*")), [])
            failures = [json.loads(line) for line in (root / "infrastructure_failures.jsonl").read_text().splitlines()]
            self.assertEqual([row["candidate_id"] for row in failures], [1])

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
