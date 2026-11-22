"""Whole-group selection uses Judge success, while metrics preserve raw F1."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from agmem.dapo_sampling import DynamicGroupSampler, assess_group, validate_dapo_config
from agmem.rewards import score_trajectory
from tests.test_dapo_sampling import examples, group_for, settings
from tests.test_hierarchical_orm_rewards import decision
from tests.test_judge_f1_rewards import reward_config


def config_for(**overrides):
    config = settings(filter_metric="judge_nonzero_advantage", **overrides)
    config["reward"] = reward_config()
    return config


def source(count=1):
    values = examples(count)
    for row in values:
        row.gold_answers = ["gold answer"]
    return values


def joint_group(example, predictions, judges, calls=None, *, truncated=()):
    calls = [2] * len(predictions) if calls is None else calls
    rows = group_for(example, [0] * len(predictions), judges=judges, truncated=truncated)
    for row, prediction, count, judge_score in zip(rows, predictions, calls, judges):
        row.prediction, row.n_read_calls = prediction, count
        score_trajectory(example, row, reward_config(), max_ids=1,
                         judge=Mock(score=lambda *_, value=judge_score: value))
    return rows


def test_all_wrong_group_with_high_f1_and_cost_difference_is_filtered():
    group = joint_group(source()[0], ["gold answer", "gold answer"], [0, 0], [0, 3])
    report = assess_group(group, config_for())
    assert not report["accepted"]
    assert report["filtered_judge_all_wrong"]
    assert not report["filtered_f1_all_zero"]
    assert report["answer_f1"] == report["outcome_f1"] == 1
    assert report["outcome_q"] == report["judge_accuracy"] == 0
    assert report["outcome_reward"] == -1
    assert report["cost_format_only_signal"]


def test_semantic_success_with_zero_word_f1_keeps_the_whole_group():
    ex = source()[0]
    ex.gold_answers = ["pendants"]
    group = joint_group(ex, ["pendant", "unknown"], [1, 0])
    report = assess_group(group, config_for())
    assert report["accepted"]
    assert report["f1_all_zero"] and not report["filtered_f1_all_zero"]
    assert report["answer_f1"] == report["outcome_f1"] == 0
    assert report["outcome_q"] == pytest.approx(0.4)
    assert report["outcome_reward"] == pytest.approx(-0.2)
    assert report["judge_accuracy"] == 0.5
    assert group[0].advantage > 0 > group[1].advantage


def test_all_correct_group_can_train_from_f1_cost_or_format_differences():
    ex = source()[0]
    groups = [joint_group(ex, ["gold answer", "gold answer explanation"], [1, 1]),
              joint_group(ex, ["gold answer"] * 2, [1, 1], [2, 3]),
              joint_group(ex, ["gold answer"] * 2, [1, 1])]
    groups[-1][1].decisions = [decision(["unknown"], invalid=["unknown"], violations=["unknown_id"])]
    score_trajectory(ex, groups[-1][1], reward_config(), max_ids=1, judge=Mock(score=lambda *_: 1.0))
    for index, group in enumerate(groups):
        report = assess_group(group, config_for())
        assert report["accepted_all_correct"]
        assert group[0].advantage > 0 > group[1].advantage
        assert report["cost_format_only_signal"] is (index > 0)


def test_all_correct_exact_ties_remain_filtered():
    group = joint_group(source()[0], ["gold answer"] * 2, [1, 1])
    report = assess_group(group, config_for())
    assert not report["accepted"]
    assert report["zero_reward_variance"] and report["no_unmasked_signal"]


def test_only_correct_trajectory_masked_or_without_training_tokens_does_not_qualify():
    ex = source()[0]
    group = joint_group(ex, ["gold answer", "wrong"], [1, 0], truncated=(0,))
    report = assess_group(group, config_for())
    assert not report["accepted"] and report["no_trainable_correct_signal"]
    assert not report["filtered_judge_all_wrong"]
    assert assess_group(group, config_for(overlong_filter=False))["accepted"]
    group[0].decisions[0].completion.token_ids = []
    assert not assess_group(group, config_for(overlong_filter=False))["accepted"]


def test_sampling_reports_judge_raw_f1_quality_and_reward_by_source():
    values = source(3)
    sampler = DynamicGroupSampler(config_for(target_groups=2), values)
    candidates = []

    def collect(example, index):
        return joint_group(example, ["gold answer"] * 2,
                           [0, 0] if index == 0 else [1, 0], [1, 2])

    groups, report = sampler.collect(collect, 7, record_candidate=lambda _, stats: candidates.append(stats))
    assert len(groups) == 2
    assert [row["selected"] for row in candidates] == [False, True, True]
    assert report["filtered_judge_all_wrong_rate"] == pytest.approx(1 / 3)
    assert report["judge_accuracy"] == report["outcome_q"] == pytest.approx(1 / 3)
    assert report["outcome_reward"] == pytest.approx(-1 / 3)
    assert report["outcome_f1"] == report["answer_f1"] == 1
    for values in report["source_metrics"].values():
        assert "judge_accuracy" in values and "outcome_q" in values and "outcome_reward" in values
        if values["selected_groups"]:
            assert values["selected_judge_accuracy"] == 0.5
            assert values["selected_answer_f1"] == 1
            assert values["selected_outcome_q"] == 0.5


def test_combined_recipe_requires_its_filter_and_rejects_prm_or_likelihood_reward():
    validate_dapo_config(config_for())
    for field, value, message in (("filter_metric", "nonzero_advantage", "requires"),
                                  ("process_reward", {"enabled": True}, "process_reward")):
        config = config_for()
        config["training"][field] = value
        with pytest.raises(ValueError, match=message):
            validate_dapo_config(config)
    config = config_for()
    config["reward"]["likelihood"] = {"mode": "reward"}
    with pytest.raises(ValueError, match="likelihood off/diagnostic"):
        validate_dapo_config(config)
    config = config_for()
    config["reward"]["judge_f1_alpha"] = False
    with pytest.raises(ValueError, match="judge_f1_alpha"):
        validate_dapo_config(config)
    config = config_for()
    config["reward"]["formula"] = "judge_cost_v2"
    with pytest.raises(ValueError, match="judge_nonzero_advantage requires"):
        validate_dapo_config(config)


def test_wrong_rollouts_are_not_dropped_from_an_accepted_group():
    group = joint_group(source()[0], ["wrong", "gold answer"], [0, 1])
    before = deepcopy(group)
    assert assess_group(group, config_for())["accepted"]
    assert len(group) == 2
    assert [row.sample_id for row in group] == [row.sample_id for row in before]
    assert [row.prediction for row in group] == [row.prediction for row in before]
