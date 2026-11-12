from copy import deepcopy

import pytest

from agmem.dapo_sampling import DynamicGroupSampler, assess_group, validate_dapo_config
from agmem.rewards import score_trajectory
from tests.test_dapo_sampling import examples, group_for, settings
from tests.test_hierarchical_orm_rewards import decision, reward_config


def config_for(**overrides):
    config = settings(filter_metric="nonzero_advantage", **overrides)
    config["reward"] = reward_config()
    return config


def f1_group(example, predictions, calls, *, truncated=(), version=7):
    group = group_for(example, rewards=[0] * len(calls), judges=[None] * len(calls),
                      truncated=truncated, version=version)
    for row, prediction, count in zip(group, predictions, calls):
        row.prediction, row.n_read_calls = prediction, count
        score_trajectory(example, row, reward_config(), max_ids=1)
    return group


def source(count=1):
    values = examples(count)
    for example in values:
        example.gold_answers = ["gold answer"]
    return values


def test_zero_f1_group_with_different_cost_is_filtered_without_judge():
    config = config_for()
    validate_dapo_config(config)
    group = f1_group(source()[0], ["wrong", "wrong"], [2, 4])
    result = assess_group(group, config)
    assert not result["accepted"]
    assert result["f1_all_zero"]
    assert result["filtered_f1_all_zero"]
    assert result["cost_format_only_signal"]
    assert group[0].advantage > 0 > group[1].advantage
    assert all(row.reward < 0 and row.judge_score is None for row in group)
    assert not any("judge" in key for key in result)


def test_zero_f1_group_with_format_difference_is_filtered():
    example = source()[0]
    group = f1_group(example, ["wrong", "wrong"], [2, 2])
    group[1].decisions = [decision(["p1:c1"], already=["p1:c1"],
                                  violations=["repeated_read"])]
    score_trajectory(example, group[1], reward_config(), max_ids=1)
    result = assess_group(group, config_for())
    assert group[0].format_reward == 0
    assert group[1].format_reward == -0.5
    assert result["cost_format_only_signal"]
    assert result["filtered_f1_all_zero"]
    assert not result["accepted"]


def test_mixed_f1_group_keeps_all_rollouts_and_its_original_advantages():
    group = f1_group(source()[0], ["wrong", "gold answer"], [2, 2])
    result = assess_group(group, config_for())
    assert result["accepted"]
    assert not result["filtered_f1_all_zero"]
    assert [row.answer_f1 for row in group] == [0, 1]
    assert group[0].advantage < 0 < group[1].advantage


def test_f1_one_group_can_learn_from_cost_and_tied_total_rewards_are_filtered():
    group = f1_group(source()[0], ["gold answer", "gold answer"], [2, 3])
    assert assess_group(group, config_for())["accepted"]
    assert all(row.answer_f1 == 1 for row in group)
    # F1=1 at 10 reads and F1=.75 at zero reads both give total reward .5.
    example = source()[0]
    example.gold_answers = ["one two three four"]
    equal = f1_group(example, ["one two three four", "one two three extra"], [10, 0])
    result = assess_group(equal, config_for())
    assert not result["accepted"]
    assert result["zero_reward_variance"]
    assert [row.advantage for row in equal] == [0, 0]


def test_positive_f1_ties_use_exact_advantage_not_min_std():
    group = f1_group(source()[0], ["gold"] * 2, [1, 2])
    result = assess_group(group, config_for(filter_min_std=10.0))
    assert result["accepted"]
    assert all(0 < row.answer_f1 < 1 for row in group)
    assert result["cost_format_only_signal"]
    group[1].reward = group[0].reward + 1e-9
    assert assess_group(group, config_for(filter_min_std=10.0))["accepted"]


def test_no_trainable_nonzero_signal_remains_filtered():
    group = f1_group(source()[0], ["gold answer"] * 2, [1, 2], truncated=(0, 1))
    assert not assess_group(group, config_for())["accepted"]
    assert assess_group(group, config_for(overlong_filter=False))["accepted"]
    for row in group:
        row.decisions[0].completion.token_ids = []
        row.decisions[0].completion.old_logprobs = []
    assert not assess_group(group, config_for(overlong_filter=False))["accepted"]


def test_sampler_refills_after_zero_f1_group_and_reports_filter_rate():
    sampler = DynamicGroupSampler(config_for(target_groups=2), source(3))
    candidates = []
    groups, report = sampler.collect(
        lambda example, index: f1_group(
            example, ["wrong", "wrong"] if index == 0 else ["wrong", "gold answer"], [1, 2]),
        7, record_candidate=lambda group, result: candidates.append(result))
    assert len(groups) == 2
    assert [row["selected"] for row in candidates] == [False, True, True]
    assert all(any(row.answer_f1 > 0 for row in group) for group in groups)
    assert all(len(group) == 2 for group in groups)
    assert report["answer_f1"] == pytest.approx(1 / 3)
    assert report["f1_all_zero_rate"] == pytest.approx(1 / 3)
    assert report["filtered_f1_all_zero_rate"] == pytest.approx(1 / 3)
    assert report["accepted_rate"] == pytest.approx(2 / 3)
    assert not any("judge" in key for key in report)


def test_constant_shift_does_not_change_group_advantages():
    group = f1_group(source()[0], ["wrong", "gold answer"], [2, 3])
    shifted = deepcopy(group)
    for row in shifted:
        row.reward += 1
    assess_group(group, config_for())
    assess_group(shifted, config_for())
    assert [row.advantage for row in group] == pytest.approx([row.advantage for row in shifted])


def test_protocol_rejects_legacy_judge_filter_and_process_reward():
    config = config_for()
    config["training"]["filter_metric"] = "judge"
    with pytest.raises(ValueError, match="nonzero_advantage"):
        validate_dapo_config(config)
    config = config_for()
    config["training"]["process_reward"] = {"enabled": True}
    with pytest.raises(ValueError, match="process_reward"):
        validate_dapo_config(config)
