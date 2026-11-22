"""Full-directory ORM config, sampler, CPU actor, journals and TensorBoard."""
from contextlib import redirect_stdout
from copy import deepcopy
import gzip
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from agmem.core import Example, Trajectory, load_config, trajectory_from_dict
from agmem.dapo import DAPOUpdater, clipped_loss
from agmem.dapo_sampling import DynamicGroupSampler, assess_group, validate_dapo_config
from agmem.full_directory_env import FullDirectoryMemoryEnvironment
from agmem.monitoring import summary_writer_class
from agmem.pipeline import evaluate_with_policy, make_environment, make_reward_scorer, metrics
from agmem.rewards import RewardScorer
from agmem.training_dapo import run_training_loop
from tests.test_training_dapo import collector, make_components


CONFIG = Path("configs/train_dapo_locomo_lmes_full_directory_orm.yaml")
LEGACY = Path("configs/train_dapo_locomo_lmes_hierarchical_batch2first.yaml")
COMPONENTS = ("outcome_reward", "format_reward", "cost_reward", "repetition_reward")
EXPECTED = {
    "answer_f1": 0.4, "outcome_f1": 0.4, "judge_accuracy": 0.5, "outcome_q": 0.9,
    "outcome_reward": 0.8, "format_reward": 0.5, "cost_reward": -0.015,
    "repetition_reward": -0.15, "reward": 1.135, "format_accuracy": 0.5,
    "format_failure_rate": 0.5, "repeated_id_count": 0.5, "invalid_id_count": 1,
    "budget_violation_count": 0, "retrieval_violation_count": 1.5,
}


def config(tmp_path=None):
    cfg = load_config(CONFIG)
    cfg["training"].update(group_size=2, target_groups=2, mini_batch_groups=1,
                           candidate_groups=3, max_num_gen_batches=1, epochs=1,
                           micro_batch_rollouts=1, initial_query_ids=[], validation_every=1,
                           checkpoint_every=0)
    cfg["execution"]["sampling_group_concurrency"] = 1
    if tmp_path is not None:
        cfg["training"]["output_dir"] = str(tmp_path)
    return cfg


def examples(split="train", count=2):
    return [Example(f"{source}/{split}/{i}", source, split, f"m{i}", "What colors?",
                    gold_answers=["red green blue"])
            for i in range(count)
            for source in ["locomo" if i % 2 == 0 else "split_longmemeval_s"]]


def judge():
    return SimpleNamespace(score=lambda _, prediction: float(prediction == "correct synonym"),
                           provenance=lambda: {"fixture": "binary-judge"})


def scoring(cfg):
    return RewardScorer(cfg["reward"], judge=judge(), max_ids=2)


def step_from(template, ids=(), *, stopping=False, records=(), invalid=(), already=(), labels=(), calls=1):
    step = deepcopy(template)
    step.action = {"ids": list(ids), "stop": stopping}
    step.completion.text = json.dumps(step.action)
    step.format_violations = list(labels)
    step.tool_result = {
        "directory_protocol": "hierarchical_full_v1", "action_kind": "stop" if stopping else "segment",
        "records": [{"id": key} for key in records], "invalid_ids": list(invalid),
        "already_read_ids": list(already), "new_ids": list(records),
        "no_new_evidence": not records, "read_call_consumed": not stopping,
        "read_calls_used": calls, "read_calls_remaining": 10 - calls,
    }
    return step


def collected_group(cfg, policy, example, candidate_index=0, scorer=None):
    rows = collector(cfg, None, policy, None, example, candidate_index)
    first, second = rows
    first.prediction, first.n_read_calls = "correct synonym", 2
    template = first.decisions[0]
    first.decisions = [step_from(template, ["a"], records=["a"]),
                       step_from(template, ["a", "unknown"], invalid=["unknown"], already=["a"],
                                 labels=["repeated_read", "unknown_id"], calls=2),
                       step_from(template, stopping=True, calls=2)]
    second.prediction, second.n_read_calls = "red green", 1
    template = second.decisions[0]
    second.decisions = [step_from(template, ["unknown"], invalid=["unknown"], labels=["unknown_id"]),
                        step_from(template, ["unknown"], stopping=True, labels=["stop_with_ids"])]
    active_scorer = scorer or scoring(cfg)
    for row in rows:
        active_scorer.score(example, row)
    return rows


def assert_additive(report, prefix=""):
    assert report[prefix + "reward"] == pytest.approx(sum(report[prefix + key] for key in COMPONENTS))


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_recipe_is_opt_in_preserves_frozen_data_and_all_seventeen_promoted_queries():
    current, old = load_config(CONFIG), load_config(LEGACY)
    assert current["data"] == old["data"]
    assert current["data"]["directory_protocol"] == "hierarchical_v1"
    assert current["reader"]["prompt_version"] == "hierarchical_full_v1"
    assert current["retrieval"]["max_ids_per_call"] == 2
    assert current["reward"]["formula"] == "full_directory_orm_v1"
    assert current["reward"]["read_call_cost"] == 0.01
    assert current["reward"]["repetition_penalty"] == 0.1
    assert current["reward"]["format_penalty"] == 0
    assert current["reward"]["format_penalty_mode"] == "binary_protocol"
    assert current["reward"].get("visible_navigation_penalty") is None
    assert not current["training"].get("visible_navigation_negative_advantage", {}).get("enabled", False)
    assert not current["training"].get("process_reward", {}).get("enabled", False)
    assert current["training"]["filter_metric"] == "judge_nonzero_advantage"
    assert current["training"]["checkpoint_retention"] == "best_and_latest_overwrite"
    assert current["training"]["validation_primary_metric"] == old["training"]["validation_primary_metric"]
    initial = current["training"]["initial_query_ids"]
    assert initial == old["training"]["initial_query_ids"] and len(initial) == 17
    assert current["training"]["output_dir"] != old["training"]["output_dir"]
    source = [Example(key, key.split("/")[0], "train", "m", "?") for key in reversed(initial)]
    sampler = DynamicGroupSampler(current, source)
    assert [sampler._next().sample_id for _ in range(17)] == initial
    # The prior experiment keeps its one-parent / forced-advantage behavior.
    assert old["reader"]["prompt_version"] == "hierarchical_v1"
    assert old["retrieval"]["max_ids_per_call"] == 1
    assert old["reward"]["formula"] == "judge_f1_cost_v1"
    assert old["training"]["visible_navigation_negative_advantage"]["enabled"]


@pytest.mark.parametrize("section,key,value", [
    ("reader", "prompt_version", "hierarchical_v1"),
    ("retrieval", "max_ids_per_call", 1),
    ("retrieval", "max_ids_per_call", 3),
    ("retrieval", "id_protocol", "strict_v1"),
    ("training", "visible_navigation_negative_advantage", {"enabled": True}),
    ("training", "process_reward", {"enabled": True}),
    ("training", "filter_metric", "nonzero_advantage"),
    ("reward", "format_penalty", 0.1),
])
def test_load_config_rejects_incompatible_reader_reward_or_advantage_modes(tmp_path, section, key, value):
    cfg = load_config(CONFIG)
    cfg[section][key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_reward_scorer_dispatches_real_quality_and_versions_the_new_components():
    cfg = config()
    ex = examples()[0]
    scorer = scoring(cfg)
    for prediction, f1, q in (("red green", 0.8, 0.8), ("correct synonym", 0, 1)):
        row = scorer.score(ex, Trajectory(ex.sample_id, 0, ex.memory_id, prediction=prediction))
        assert row.answer_f1 == f1
        assert row.outcome == q
        assert row.outcome_reward == pytest.approx(2 * q - 1)
        assert row.reward == pytest.approx(2 * q)
    provenance = scorer.provenance()
    assert provenance["formula"] == "full_directory_orm_v1"
    assert provenance["outcome_quality_formula"] == "1 if judge==1 else answer_f1"
    assert provenance["reader_tool_protocol"] == "hierarchical_full_v1"
    assert provenance["judge"] == {"fixture": "binary-judge"}
    with patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "frozen-27b"}), \
         patch("agmem.judges.JudgeScorer", return_value=judge()) as factory, \
         patch("agmem.policy.load_tokenizer", side_effect=AssertionError("Likelihood is disabled")):
        made = make_reward_scorer(cfg, None)
    assert made.max_ids == 2
    assert factory.call_args.args[0]["model_revision"] == "frozen-27b"
    assert made.score(ex, Trajectory(ex.sample_id, 0, ex.memory_id,
                                    prediction="correct synonym")).outcome == 1


def test_environment_factory_selects_full_directory_without_parent_window_setting():
    cfg = config()
    tokenizer = SimpleNamespace(encode=lambda text, **_: list(text))
    with patch("agmem.pipeline.validate_directory_version", return_value="frozen-bank"), \
         patch("agmem.policy.load_tokenizer", return_value=tokenizer), \
         patch("agmem.full_directory_env.FullDirectoryMemoryEnvironment", wraps=FullDirectoryMemoryEnvironment) as factory:
        env = make_environment(cfg, SimpleNamespace(), SimpleNamespace(check_capacity=lambda _: None))
    assert isinstance(env, FullDirectoryMemoryEnvironment)
    assert env.version == "frozen-bank" and env.max_ids == 2
    assert "max_open_directories" not in factory.call_args.kwargs
    assert not hasattr(env, "max_open_directories")


def test_evaluation_and_candidate_metrics_keep_retrieval_errors_out_of_format(cpu_threads):
    cfg = config()
    policy, _ = make_components()
    rows = collected_group(cfg, policy, examples()[0])
    assert not rows[0].format_failure
    assert rows[0].format_reward == 1
    assert rows[0].retrieval_scoring["retrieval_violation_count"] == 2
    candidate = assess_group(rows, cfg)
    assert candidate["accepted"]
    report = metrics(rows)
    for scope in (candidate, report["pooled"], report["source_macro"], report["by_source"]["locomo"]):
        assert_additive(scope)
        for key, expected in EXPECTED.items():
            assert scope[key] == pytest.approx(expected), key
    assert candidate["format_decisions_unknown_id"] == 0
    assert candidate["format_decisions_repeated_read"] == 0
    assert candidate["retrieval_decisions_unknown_id"] > 0
    assert candidate["retrieval_decisions_repeated_read"] > 0


def test_all_judge_wrong_group_is_filtered_even_with_positive_and_different_raw_f1(cpu_threads):
    cfg = config()
    policy, _ = make_components()
    ex = examples()[0]
    rows = collected_group(cfg, policy, ex)
    rows[0].prediction = "red green blue"
    scoring(cfg).score(ex, rows[0])
    report = assess_group(rows, cfg)
    assert [row.judge_score for row in rows] == [0, 0]
    assert [row.answer_f1 for row in rows] == [1, 0.8]
    assert report["outcome_q"] == 0.9
    assert report["reward_std"] > 0
    assert not report["accepted"] and report["filtered_judge_all_wrong"]
    assert not report["filtered_f1_all_zero"]


def test_sampling_reports_additive_components_for_candidates_sources_and_selected_groups(cpu_threads):
    cfg = config()
    policy, _ = make_components()
    source = examples(count=3)
    cfg["training"]["initial_query_ids"] = [example.sample_id for example in source]
    sampler = DynamicGroupSampler(cfg, source)
    candidates = []

    def collect(example, index):
        rows = collected_group(cfg, policy, example, index)
        if index == 0:
            rows[0].prediction = "red green blue"
            scoring(cfg).score(example, rows[0])
        return rows

    groups, report = sampler.collect(collect, 0, record_candidate=lambda _, stats: candidates.append(stats))
    assert len(groups) == 2
    assert [row["selected"] for row in candidates] == [False, True, True]
    for scope in [report, *candidates, *report["source_metrics"].values()]:
        assert_additive(scope)
        assert scope["format_accuracy"] == 0.5
        assert scope["retrieval_violation_count"] == 1.5
    for scope in report["source_metrics"].values():
        if scope["selected_groups"]:
            assert_additive(scope, "selected_")
            assert scope["selected_outcome_q"] == 0.9


def test_cpu_actor_uses_same_frozen_row_advantage_for_every_generated_token(cpu_threads):
    cfg = config()
    policy, optimizer = make_components()
    rows = collected_group(cfg, policy, examples()[0])
    assert assess_group(rows, cfg)["accepted"]
    frozen = deepcopy(rows)
    expected = [row.advantage for row in rows for step in row.decisions for _ in step.completion.token_ids]
    observed = []

    def capture(new, old, advantage, *args):
        observed.extend([advantage] * new.numel())
        return clipped_loss(new, old, advantage, *args)

    before = deepcopy(policy.model.state_dict())
    updater = DAPOUpdater(policy, optimizer, cfg["training"])
    updater.begin_batch([rows], "frozen-full-directory")
    with patch("agmem.dapo.clipped_loss", side_effect=capture), redirect_stdout(StringIO()):
        report = updater.update(rows)
    updater.end_batch()
    assert observed == expected
    assert report["training_tokens"] == len(expected)
    assert rows == frozen
    assert rows[0].advantage > 0  # Its repeated/unknown READ still uses this same positive A.
    assert all(not row.action_advantages and not row.process_attribution for row in rows)
    assert policy.policy_version == 1
    assert any(not torch.equal(before[key], value) for key, value in policy.model.state_dict().items())


def test_frozen_retrieval_dictionary_is_bound_to_the_behavior_batch(cpu_threads):
    cfg = config()
    policy, optimizer = make_components()
    rows = collected_group(cfg, policy, examples()[0])
    assess_group(rows, cfg)
    updater = DAPOUpdater(policy, optimizer, cfg["training"])
    updater.begin_batch([rows], "immutable-components")
    rows[0].retrieval_scoring["repetition_reward"] -= 0.1
    before = deepcopy(policy.model.state_dict())
    with pytest.raises(ValueError, match="changed or foreign"):
        updater.update(rows)
    for key, value in policy.model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert policy.policy_version == 0


def test_real_cpu_training_logs_four_components_and_persists_pending_groups(cpu_threads, tmp_path):
    cfg = config(tmp_path)
    source = examples()
    cfg["training"]["initial_query_ids"] = [example.sample_id for example in source]
    active_scorer = scoring(cfg)
    captured = []

    def collect(config, env, policy, agent, example, candidate_index, scorer=None):
        rows = collected_group(config, policy, example, candidate_index, scorer)
        captured.append(rows)
        return rows

    def validate(config, store, policy, split, output, limit=None):
        evaluation_examples = examples("validation")
        rows = deepcopy(captured[0])
        for ex, row in zip(evaluation_examples, rows):
            row.sample_id = ex.sample_id
        fake_agent = SimpleNamespace(service=SimpleNamespace(client=SimpleNamespace(close=lambda: None)))
        with patch("agmem.pipeline.validate_directories"), \
             patch("agmem.pipeline.make_environment", return_value=SimpleNamespace(version="frozen-bank")), \
             patch("agmem.pipeline.make_task_agent", return_value=fake_agent), \
             patch("agmem.pipeline.make_reward_scorer", return_value=active_scorer), \
             patch("agmem.pipeline.task_examples", return_value=evaluation_examples), \
             patch("agmem.pipeline.scored_rollout", side_effect=rows):
            output.mkdir(parents=True, exist_ok=True)
            return evaluate_with_policy(config, store, policy, split, output, limit)

    policy, optimizer = make_components()
    writer = summary_writer_class()(log_dir=str(tmp_path / "tensorboard"))
    try:
        with patch("agmem.pipeline.collect_group", side_effect=collect), \
             patch("agmem.pipeline.evaluate_with_policy", side_effect=validate), redirect_stdout(StringIO()):
            run_training_loop(cfg, None, policy, optimizer, DAPOUpdater(policy, optimizer, cfg["training"]),
                              SimpleNamespace(version="frozen-bank"), None, active_scorer, source,
                              tmp_path, writer, max_groups=1)
    finally:
        writer.close()
    assert policy.policy_version == 1
    for path in ("train_metrics.jsonl", "sampling_metrics.jsonl", "candidate_metrics.jsonl"):
        for report in map(json.loads, (tmp_path / path).read_text().splitlines()):
            assert_additive(report)
            for key, expected in EXPECTED.items():
                assert report[key] == pytest.approx(expected), (path, key)
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    for prefix in ("sampling", "train", "validation", "validation/source_macro"):
        for key, expected in EXPECTED.items():
            assert events.Scalars(f"{prefix}/{key}")[-1].value == pytest.approx(expected), (prefix, key)
    for source_name in ("locomo", "split_longmemeval_s"):
        for prefix in (f"sampling/by_source/{source_name}", f"validation/by_source/{source_name}"):
            actual = {key: events.Scalars(f"{prefix}/{key}")[-1].value for key in (*COMPONENTS, "reward")}
            assert_additive(actual)
    validation = json.loads((tmp_path / "validation/step-00000001/metrics.json").read_text())
    assert validation["primary_outcome_metric"] == cfg["training"]["validation_primary_metric"]
    assert validation["scoring_protocol"]["formula"] == "full_directory_orm_v1"
    with gzip.open(tmp_path / "groups/group-00000000.json.gz", "rt") as stream:
        full = json.load(stream)["rollouts"]
    compact = json.loads((tmp_path / "rollouts/group-00000000.json").read_text())["rollouts"]
    candidates = json.loads((tmp_path / "candidate_rollouts/candidate-00000000.json").read_text())["rollouts"]
    for stored in (full, compact, candidates):
        assert [row["retrieval_scoring"] for row in stored] == [row.retrieval_scoring for row in captured[0]]
        assert all("action_advantages" not in row and "process_attribution" not in row for row in stored)
    assert [trajectory_from_dict(row).to_dict() for row in full] == full
    state = torch.load(tmp_path / "checkpoint-00000001/trainer.pt", weights_only=True)
    assert state["config"]["reward"]["formula"] == "full_directory_orm_v1"
    pending = state["dapo_state"]["active"]["groups"][1]
    assert pending == [row.to_dict() for row in captured[1]]
    assert all(not row.action_advantages for group in captured for row in group)
    assert "retrieval_scoring" not in Trajectory("legacy", 0, "m").to_dict()
