"""Visible-parent repeats get a distinct per-position rate, never a second fee."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agmem.core import Completion, Decision, Example, Trajectory
from agmem.rewards import (RewardScorer, score_trajectory, validate_format_penalty_mode,
                           visible_navigation_id_count)
from tests.test_hierarchical_environment import Policy, Store, read, STOP
from tests.test_hierarchical_orm_rewards import decision
from agmem.hierarchical_env import HierarchicalMemoryEnvironment


def reward_config(**overrides):
    return {"formula": "judge_f1_cost_v1", "judge_f1_alpha": 0.8,
            "read_call_cost": 0.05, "format_penalty_mode": "per_id",
            "format_penalty": 0.1, "visible_navigation_penalty": 0.6, **overrides}


def example():
    return Example("locomo/train/q", "locomo", "train", "mem_fixture", "Question?",
                   gold_answers=["answer"])


def score(row, config=None, *, judge_value=1.0):
    row.prediction = "answer"
    return score_trajectory(example(), row, config or reward_config(), max_ids=1,
                            judge=SimpleNamespace(score=lambda *_: judge_value))


def visible_parent(ids=("p1",), *, kind="navigation", invalid=(), violations=()):
    step = decision(ids, kind, invalid=invalid, violations=violations)
    step.tool_result["visible_navigation_ids"] = list(dict.fromkeys(key for key in ids if key.startswith("p")))
    return step


def test_actual_one_open_directory_repeat_costs_point_six_not_point_seven():
    env = HierarchicalMemoryEnvironment(Store(), "frozen_v1", max_open_directories=1,
                                        count_tokens=len)
    row = env.run(example(), Policy([read("p0001"), read("p0001"), STOP]), 42)
    score(row)
    assert row.n_read_calls == 2
    assert row.evidence == []
    assert row.decisions[1].format_violations == ["visible_navigation"]
    assert visible_navigation_id_count(row) == row.format_violation_count == 1
    assert row.format_reward == -0.6
    assert row.read_cost == 0.1
    assert row.reward == pytest.approx(0.3)


def test_evicted_parent_reopen_is_legal_but_original_segment_repeat_still_costs_point_one():
    env = HierarchicalMemoryEnvironment(Store(), "frozen_v1", max_open_directories=1,
                                        count_tokens=len)
    row = env.run(example(), Policy([
        read("p0001"), read("p0001:c0001"), read("p0002"),
        read("p0001"), read("p0001:c0001"), STOP]), 42)
    score(row)
    assert visible_navigation_id_count(row) == 0
    assert row.decisions[3].tool_result["repeated_navigation"] is True
    assert row.decisions[3].tool_result["visible_navigation_ids"] == []
    assert row.format_violation_count == 1
    assert row.format_reward == -0.1
    assert row.read_cost == 0.25
    assert row.reward == pytest.approx(0.65)


def test_mixed_suffix_positions_use_navigation_rate_once_despite_overlapping_labels():
    step = visible_parent(("child", "p1", "p1", "unknown"), kind="segment", invalid=["unknown"],
                          violations=["too_many_ids", "duplicate_ids", "unknown_id", "visible_navigation"])
    row = score(Trajectory("q", 0, "m", decisions=[step], n_read_calls=1))
    assert visible_navigation_id_count(row) == 2
    assert row.format_violation_count == 3  # Two visible parents and one other excess ID.
    assert row.format_reward == pytest.approx(-1.3)
    assert row.reward == pytest.approx(-0.35)


def test_same_position_is_not_charged_again_for_duplicate_excess_or_invalid_labels():
    step = visible_parent(("p1", "p1"), invalid=["p1"],
                          violations=["too_many_ids", "duplicate_ids", "unknown_id", "visible_navigation"])
    row = score(Trajectory("q", 0, "m", decisions=[step], n_read_calls=1))
    assert visible_navigation_id_count(row) == row.format_violation_count == 2
    assert row.format_reward == pytest.approx(-1.2)
    assert row.reward == pytest.approx(-0.25)


def test_stop_with_nonempty_ids_uses_only_ordinary_protocol_penalty():
    stop = Decision([], Completion("stop"), {"ids": ["p1", "p1"], "stop": True},
                    {"action_kind": "stop", "visible_navigation_ids": ["p1"],
                     "read_call_consumed": False}, format_violations=["stop_with_ids"])
    row = score(Trajectory("q", 0, "m", decisions=[stop]))
    assert visible_navigation_id_count(row) == 0
    assert row.format_violation_count == 1
    assert row.format_reward == -0.1
    assert row.read_cost == 0
    assert row.reward == pytest.approx(0.9)


def test_historical_repeat_flag_without_pre_action_visibility_is_not_enough():
    step = decision(["p1"], "navigation")
    step.tool_result.update(repeated_navigation=True, visible_navigation_ids=[])
    row = score(Trajectory("q", 0, "m", decisions=[step], n_read_calls=1))
    assert visible_navigation_id_count(row) == 0
    assert row.format_violation_count == 0
    assert row.format_reward == 0


def test_protocol_error_and_navigation_position_keep_separate_rates():
    nav = visible_parent()
    malformed = Decision([], Completion("not JSON"), None,
                         format_violations=["invalid_json_or_schema"])
    row = score(Trajectory("q", 0, "m", decisions=[nav, malformed], n_read_calls=1,
                           termination_reason="protocol_failure"))
    assert visible_navigation_id_count(row) == 1
    assert row.format_violation_count == 2
    assert row.format_reward == pytest.approx(-0.7)
    assert row.reward == pytest.approx(0.25)


def test_navigation_attempt_after_read_budget_keeps_format_but_adds_no_read_cost():
    step = visible_parent(violations=["visible_navigation", "read_budget_exceeded"])
    step.tool_result["read_call_consumed"] = False
    row = score(Trajectory("q", 0, "m", decisions=[step], n_read_calls=10))
    assert row.format_violation_count == 2
    assert row.format_reward == pytest.approx(-0.7)
    assert row.read_cost == 0.5


def test_wrong_answer_does_not_gate_away_navigation_penalties():
    row = score(Trajectory("q", 0, "m", decisions=[visible_parent()], n_read_calls=1), judge_value=0.0)
    assert row.outcome == 0 and row.outcome_reward == -1
    assert row.format_reward == -0.6
    assert row.reward == pytest.approx(-1.65)


@pytest.mark.parametrize("value", [True, False, -0.1, "0.6", None, float("nan"), float("inf")])
def test_invalid_navigation_rate_rejected_before_judge_request(value):
    cfg = reward_config(visible_navigation_penalty=value)
    judge = Mock()
    with pytest.raises(ValueError, match="visible_navigation_penalty"):
        score_trajectory(example(), Trajectory("q", 0, "m"), cfg, judge=judge)
    with pytest.raises(ValueError, match="visible_navigation_penalty"):
        RewardScorer(cfg, judge=judge)
    judge.score.assert_not_called()


@pytest.mark.parametrize("mode", ["per_rollout", "per_decision"])
def test_navigation_rate_requires_position_counting(mode):
    with pytest.raises(ValueError, match="per_id"):
        validate_format_penalty_mode(reward_config(format_penalty_mode=mode))


def test_explicit_zero_navigation_rate_does_not_remove_other_format_penalties():
    row = Trajectory("q", 0, "m", n_read_calls=1, decisions=[visible_parent(("p1", "unknown"), invalid=["unknown"])])
    score(row, reward_config(visible_navigation_penalty=0))
    assert row.format_violation_count == 2
    assert row.format_reward == -0.1


def test_opt_in_provenance_binds_both_rates_without_changing_legacy_recipes():
    judge = SimpleNamespace(provenance=lambda: {"model": "fixture"})
    cfg = reward_config()
    old = {key: value for key, value in cfg.items() if key != "visible_navigation_penalty"}
    old_scorer = RewardScorer(old, judge=judge)
    old_provenance = old_scorer.provenance()
    assert old_provenance["format_counting_version"] == "hierarchical_reader_id_with_protocol_v4"
    assert "visible_navigation_penalty" not in old_provenance
    assert "base_format_penalty" not in old_provenance
    assert "format_penalty_composition" not in old_provenance
    scorer = RewardScorer(cfg, judge=judge)
    new = scorer.provenance()
    assert new["format_counting_version"] == "hierarchical_reader_id_with_protocol_v5"
    assert new["visible_navigation_penalty"] == 0.6
    assert new["base_format_penalty"] == 0.1
    assert new["format_penalty_composition"] == "visible_navigation_replaces_base_per_id_v1"
    assert RewardScorer(reward_config(visible_navigation_penalty=0.5), judge=judge).provenance() != new
    assert RewardScorer(reward_config(format_penalty=0.2), judge=judge).provenance() != new
    base_row = Trajectory("q", 0, "m", decisions=[visible_parent()], n_read_calls=1)
    old_row, new_row = deepcopy(base_row), deepcopy(base_row)
    score(old_row, old)
    score(new_row, cfg)
    assert old_row.format_reward == -0.1
    assert new_row.format_reward == -0.6
    assert old_row.outcome == new_row.outcome
    assert old_scorer.provenance() == old_provenance


@pytest.mark.parametrize("formula", ["f1_cost_v1", "judge_cost_v2", "judge_f1_cost_v1"])
def test_targeted_penalty_works_for_supported_per_id_formulas(formula):
    row = score(Trajectory("q", 0, "m", decisions=[visible_parent()], n_read_calls=1),
                reward_config(formula=formula))
    assert row.format_reward == -0.6
    assert row.reward == pytest.approx(0.35)
