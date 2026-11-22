"""Semantic correctness and raw F1 remain separate in the combined ORM."""
from unittest.mock import Mock

import pytest

from agmem.core import Completion, Decision, Example, Trajectory
from agmem.rewards import RewardScorer, WORD_F1_PROTOCOL, score_trajectory
from tests.test_hierarchical_orm_rewards import decision


def reward_config(**overrides):
    return {"formula": "judge_f1_cost_v1", "judge_f1_alpha": 0.8,
            "read_call_cost": 0.05, "format_penalty": 0.1,
            "format_penalty_mode": "per_id", **overrides}


def example(gold="pendants"):
    return Example("locomo/train/q", "locomo", "train", "m", "Question?", gold_answers=[gold])


def test_semantically_correct_zero_overlap_retains_floor_and_raw_f1():
    judge = Mock(score=Mock(return_value=1.0))
    row = Trajectory("q", 0, "m", prediction="pendant", n_read_calls=2)
    score_trajectory(example(), row, reward_config(), max_ids=1, judge=judge)
    assert row.judge_score == 1
    assert row.answer_f1 == 0
    assert row.outcome == pytest.approx(0.8)
    assert row.outcome_reward == pytest.approx(0.6)
    assert row.read_cost == row.raw_read_cost == 0.1
    assert row.reward == pytest.approx(0.5)
    judge.score.assert_called_once()


def test_rejected_answer_keeps_diagnostic_f1_but_receives_no_quality_reward():
    row = Trajectory("q", 0, "m", prediction="the dog park I do not know", n_read_calls=2)
    score_trajectory(example("the dog park"), row, reward_config(), max_ids=1,
                     judge=Mock(score=lambda *_: 0.0))
    assert row.answer_f1 == pytest.approx(0.6)
    assert row.judge_score == row.outcome == 0
    assert row.outcome_reward == -1
    assert row.read_cost == 0.1  # Requested penalties remain unconditional.
    assert row.reward == pytest.approx(-1.1)


def test_alpha_interpolates_only_within_semantically_correct_answers():
    row = Trajectory("q", 0, "m", prediction="gold answer explanation", n_read_calls=0)
    ex = example("gold answer")
    judge = Mock(score=lambda *_: 1.0)
    for alpha, expected in ((0, 0.8), (0.8, 0.96), (1, 1.0)):
        score_trajectory(ex, row, reward_config(judge_f1_alpha=alpha), judge=judge)
        assert row.answer_f1 == pytest.approx(0.8)
        assert row.outcome == pytest.approx(expected)
        assert row.outcome_reward == pytest.approx(2 * expected - 1)


def test_id_count_and_nonempty_stop_each_use_point_one_penalty_without_quality_gating():
    read = decision(["p0001", "p0001", "unknown"], "navigation", invalid=["unknown"],
                    violations=["too_many_ids", "duplicate_ids", "unknown_id"])
    stop = Decision([], Completion("bad stop"), {"ids": ["p0001:c0001"], "stop": True},
                    {"directory_protocol": "hierarchical_v1", "action_kind": "stop",
                     "read_call_consumed": False}, format_violations=["stop_with_ids"])
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=1, decisions=[read, stop])
    for judge_score in (1.0, 0.0):
        score_trajectory(example("answer"), row, reward_config(), max_ids=1,
                         judge=Mock(score=lambda *_: judge_score))
        assert row.format_violation_count == 3
        assert row.format_reward == pytest.approx(-0.3)
        assert row.read_cost == 0.05
        assert row.reward == pytest.approx((1 if judge_score else -1) - 0.35)


@pytest.mark.parametrize("alpha", [True, False, -0.1, 1.1, float("nan"), float("inf"), "0.8", None])
def test_alpha_rejected_before_judge_request(alpha):
    judge = Mock()
    config = reward_config(judge_f1_alpha=alpha)
    with pytest.raises(ValueError, match="judge_f1_alpha"):
        score_trajectory(example(), Trajectory("q", 0, "m", prediction="pendant"), config, judge=judge)
    with pytest.raises(ValueError, match="judge_f1_alpha"):
        RewardScorer(config, judge=judge)
    judge.score.assert_not_called()


@pytest.mark.parametrize("value", [None, True, "1", 0.5, float("nan"), float("inf")])
def test_missing_or_nonbinary_judge_never_assigns_a_quality_score(value):
    judge = None if value is None else Mock(score=lambda *_: value)
    with pytest.raises(ValueError, match="Judge"):
        score_trajectory(example(), Trajectory("q", 0, "m", prediction="pendant"),
                         reward_config(), judge=judge)


def test_judge_infrastructure_error_propagates_instead_of_becoming_wrong():
    judge = Mock(score=Mock(side_effect=RuntimeError("service unavailable")))
    with pytest.raises(RuntimeError, match="service unavailable"):
        score_trajectory(example(), Trajectory("q", 0, "m", prediction="pendant"),
                         reward_config(), judge=judge)


def test_provenance_binds_alpha_judge_and_word_f1_without_changing_f1_recipe():
    judge = Mock(provenance=lambda: {"protocols": {"locomo": "locomo_refined"}})
    scorer = RewardScorer(reward_config(), judge=judge)
    provenance = scorer.provenance()
    assert provenance["judge"] == judge.provenance()
    assert provenance["judge_f1_alpha"] == 0.8
    assert provenance["answer_f1_protocol"] == WORD_F1_PROTOCOL
    assert provenance["primary_outcome_metric"] == "outcome_q"
    assert provenance["outcome_reward_transform"] == "2*q-1"
    assert provenance["penalty_composition"] == "unconditional_cost_and_format_v1"
    assert provenance["format_counting_version"] == "hierarchical_reader_id_with_protocol_v4"
    assert RewardScorer(reward_config(judge_f1_alpha=0.9), judge=judge).provenance() != provenance
    previous = RewardScorer({**reward_config(), "formula": "f1_cost_v1"}, judge=judge).provenance()
    assert previous["judge"] is None
    assert "judge_f1_alpha" not in previous


def test_likelihood_reward_is_not_implicitly_added_to_combined_outcome():
    with pytest.raises(ValueError, match="likelihood off/diagnostic"):
        score_trajectory(example(), Trajectory("q", 0, "m"),
                         reward_config(likelihood={"mode": "reward"}), judge=Mock())
