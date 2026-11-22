"""Independent full-directory terminal components and backward compatibility."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agmem.core import Completion, Decision, Example, Trajectory, trajectory_from_dict
from agmem.full_directory_rewards import (
    FORMULA, full_directory_metrics, full_directory_provenance,
    score_full_directory, validate_full_directory_reward,
)
from agmem.rewards import RewardScorer, score_trajectory


CONFIG = {
    "formula": FORMULA, "read_call_cost": 0.01, "repetition_penalty": 0.1,
    "format_penalty_mode": "binary_protocol", "format_penalty": 0,
}
EXAMPLE = Example("locomo/test/0", "locomo", "test", "m", "What colors?",
                  gold_answers=["red green blue"])


def read(*ids, records=(), invalid=(), already=(), labels=(), error=None, ok=True):
    result = {
        "directory_protocol": "hierarchical_full_v1", "ok": ok,
        "records": [{"id": key} for key in records], "invalid_ids": list(invalid),
        "already_read_ids": list(already),
    }
    if error:
        result["error"] = {"code": error}
    return Decision([], Completion(""), {"ids": list(ids), "stop": False}, result,
                    format_violations=list(labels))


def stop(*ids):
    return Decision([], Completion(""), {"ids": list(ids), "stop": True})


def score(decisions=(), *, prediction="red green", judge=0, reads=0, termination="stop", **kwargs):
    row = Trajectory(EXAMPLE.sample_id, 0, "m", decisions=list(decisions),
                     n_read_calls=reads, prediction=prediction, termination_reason=termination)
    return score_full_directory(EXAMPLE, row, CONFIG,
                                judge=SimpleNamespace(score=lambda *_: judge), **kwargs)


def test_wrong_judge_preserves_raw_point_eight_f1_without_beta_or_evidence_gate():
    row = score([stop()])
    assert not row.evidence
    assert row.answer_f1 == 0.8
    assert row.judge_score == 0
    assert row.outcome == 0.8
    assert row.outcome_reward == pytest.approx(0.6)
    assert row.format_reward == 1
    assert row.reward == pytest.approx(1.6)


def test_correct_judge_gets_one_even_when_word_f1_is_zero():
    row = score([stop()], prediction="synonym", judge=1)
    assert row.answer_f1 == 0
    assert row.judge_score == 1
    assert row.outcome == row.outcome_reward == 1
    assert row.reward == 2


def test_raw_word_protocol_keeps_articles_and_uses_best_reference():
    example = Example("x", "locomo", "test", "m", "?",
                      gold_answers=["unrelated", "the red-blue green"])
    row = Trajectory("x", 0, "m", prediction="THE redblue")
    score_full_directory(example, row, CONFIG, judge=SimpleNamespace(score=lambda *_: 0))
    assert row.answer_f1 == row.outcome == 0.8


@pytest.mark.parametrize("judge", [True, False, "1", "0", None, 0.5, 2, -1, float("nan"), float("inf")])
def test_judge_requires_numeric_binary_output_and_leaves_row_unmodified_on_error(judge):
    row = Trajectory("x", 0, "m", prediction="red green", judge_score=1, reward=42)
    before = row.to_dict()
    with pytest.raises(ValueError, match="binary score"):
        score_full_directory(EXAMPLE, row, CONFIG, judge=SimpleNamespace(score=lambda *_: judge))
    assert row.to_dict() == before


def test_judge_failure_propagates_instead_of_becoming_an_incorrect_score():
    def fail(*_):
        raise RuntimeError("Judge service unavailable")
    with pytest.raises(RuntimeError, match="Judge service unavailable"):
        score_full_directory(EXAMPLE, Trajectory("x", 0, "m"), CONFIG,
                             judge=SimpleNamespace(score=fail))
    with pytest.raises(ValueError, match="requires a Judge"):
        score_full_directory(EXAMPLE, Trajectory("x", 0, "m"), CONFIG)


def test_format_reward_is_one_per_trajectory_not_one_per_valid_call():
    short = score([read("a", records=["a"]), stop()], judge=1, reads=1)
    long = score([read("a", records=["a"]), read("b", records=["b"]), stop()], judge=1, reads=2)
    assert short.format_reward == long.format_reward == 1
    assert short.format_violation_count == long.format_violation_count == 0
    assert short.reward == pytest.approx(1.99)
    assert long.reward == pytest.approx(1.98)


@pytest.mark.parametrize("decision", [
    Decision([], Completion("{"), format_violations=["invalid_json_or_schema"]),
    read(), stop("a"), read("a", "b", "c"),
    Decision([], Completion("{}"), {"ids": [], "stop": 1}),
])
def test_each_syntax_or_schema_violation_zeroes_binary_format_only(decision):
    row = score([decision, stop()], judge=1)
    assert row.format_failure
    assert row.format_reward == 0
    assert row.format_violation_count == 1
    assert row.retrieval_scoring["retrieval_violation_count"] == 0
    assert row.reward == 1


def test_multiple_format_failures_do_not_create_a_negative_format_penalty():
    row = score([read(), stop("unknown"), read("a", "b", "c"), stop()], judge=1, reads=2)
    assert row.format_violation_count == 3
    assert row.format_reward == 0
    assert row.reward == pytest.approx(0.98)


def test_semantic_labels_do_not_fail_format_and_excess_positions_are_not_bad_by_themselves():
    row = score([read("a", records=["a"]),
                 read("a", "unknown", invalid=["unknown"], already=["a"],
                      labels=["unknown_id", "repeated_read"]), stop()], judge=1, reads=2)
    assert not row.format_failure
    assert row.format_reward == 1
    assert row.retrieval_scoring["retrieval_violation_count"] == 2
    assert row.reward == pytest.approx(1.78)
    excess = score([read("a", "b", "c", labels=["too_many_ids"]), stop()], judge=1, reads=1)
    assert excess.format_reward == 0
    assert excess.retrieval_scoring["retrieval_violation_count"] == 0
    assert excess.reward == pytest.approx(0.99)


@pytest.mark.parametrize("first,second,invalid,repeated_count,invalid_count,bad", [
    ([], ["new", "new"], [], 1, 0, 1),
    (["known"], ["known", "known"], [], 2, 0, 2),
    ([], ["unknown", "unknown"], ["unknown"], 1, 2, 2),
    (["known"], ["known", "unknown"], ["unknown"], 1, 1, 2),
])
def test_each_requested_position_is_charged_at_most_once(first, second, invalid,
                                                        repeated_count, invalid_count, bad):
    steps = [read(*first, records=first)] if first else []
    row = score([*steps, read(*second, invalid=invalid), stop()], judge=1, reads=len(steps) + 1)
    components = row.retrieval_scoring
    assert components["repeated_id_count"] == repeated_count
    assert components["invalid_id_count"] == invalid_count
    assert components["retrieval_violation_count"] == bad
    assert components["repetition_reward"] == pytest.approx(-0.1 * bad)
    assert row.format_reward == 1


def test_overlapping_repeat_invalid_and_excess_labels_charge_union_of_positions():
    row = score([read("a", records=["a"]),
                 read("a", "a", "unknown", "unknown", "b", invalid=["a", "unknown"],
                      already=["a"], labels=["too_many_ids", "duplicate_ids", "unknown_id", "repeated_read"]),
                 stop()], judge=1, reads=2)
    assert row.format_reward == 0
    assert row.format_violation_count == 1
    assert row.retrieval_scoring["repeated_id_count"] == 3
    assert row.retrieval_scoring["invalid_id_count"] == 4
    assert row.retrieval_scoring["retrieval_violation_count"] == 4
    assert row.reward == pytest.approx(0.58)


def test_unknown_and_parent_ids_are_independent_of_format_and_not_navigation_penalties():
    row = score([read("parent", "missing", invalid=["parent", "missing"],
                      labels=["unknown_id", "visible_navigation"], error="unknown_or_unauthorized_id"),
                 stop()], judge=1, reads=1)
    assert row.format_reward == 1
    assert row.retrieval_scoring["invalid_id_count"] == 2
    assert row.retrieval_scoring["repeated_id_count"] == 0
    assert row.reward == pytest.approx(1.79)


def test_stop_and_invalid_json_never_gain_id_or_budget_penalties_from_tool_metadata():
    malformed = Decision([], Completion("{"), tool_result={
        "invalid_ids": ["x"], "already_read_ids": ["x"], "error": {"code": "max_read_calls"}},
        format_violations=["invalid_json_or_schema", "read_budget_exceeded"])
    stopping = stop("x", "x")
    stopping.tool_result = malformed.tool_result
    stopping.format_violations = ["stop_with_ids", "read_budget_exceeded"]
    row = score([malformed, stopping], judge=1)
    assert row.format_reward == 0
    assert row.format_violation_count == 2
    assert row.retrieval_scoring["retrieval_violation_count"] == 0
    assert row.reward == 1


def test_budget_limit_itself_is_free_but_each_additional_read_attempt_is_charged():
    at_limit = score([read("a", records=["a"]), stop()], judge=1, reads=1,
                     termination="max_read_calls")
    assert at_limit.retrieval_scoring["budget_violation_count"] == 0
    over = score([read("a", records=["a"]),
                  read("a", "a", labels=["read_budget_exceeded"], error="max_read_calls", ok=False),
                  read("missing", invalid=["missing"], error="max_read_calls", ok=False)],
                 judge=1, reads=1, termination="max_read_calls")
    assert over.format_reward == 1
    assert over.read_cost == 0.01
    assert over.retrieval_scoring["repeated_id_count"] == 2
    assert over.retrieval_scoring["invalid_id_count"] == 1
    assert over.retrieval_scoring["budget_violation_count"] == 2
    assert over.retrieval_scoring["retrieval_violation_count"] == 5
    assert over.reward == pytest.approx(1.49)


def test_budget_and_format_are_independent_when_empty_read_is_over_budget():
    row = score([read(error="max_read_calls", labels=["empty_ids", "read_budget_exceeded"]), stop()], judge=1)
    assert row.format_reward == 0
    assert row.format_violation_count == 1
    assert row.retrieval_scoring["budget_violation_count"] == 1
    assert row.reward == pytest.approx(0.9)


@pytest.mark.parametrize("code", ["context_capacity_exceeded", "capacity_error", "infrastructure_error"])
def test_failed_retrieval_does_not_mark_evidence_as_read_or_fail_format(code):
    row = score([read("a", error=code, ok=False),
                 read("a", records=["a"]), stop()], judge=1, reads=2)
    assert not row.format_failure
    assert row.retrieval_scoring["retrieval_violation_count"] == 0
    assert row.reward == pytest.approx(1.98)


def test_partially_successful_request_records_still_enter_read_history():
    row = score([read("a", "unknown", records=["a"], invalid=["unknown"], ok=False),
                 read("a"), stop()], judge=1, reads=2)
    assert row.retrieval_scoring["invalid_id_count"] == 1
    assert row.retrieval_scoring["repeated_id_count"] == 1
    assert row.retrieval_scoring["retrieval_violation_count"] == 2
    assert row.reward == pytest.approx(1.78)


def test_service_failure_without_reader_action_is_not_a_format_error():
    row = score([Decision([], Completion(""), tool_result={"error": {"code": "infrastructure_error"}})],
                termination="infrastructure_error", judge=1)
    assert row.format_reward == 1
    assert row.reward == 2
    old_parse_failure = score([Decision([], Completion("{"))], termination="protocol_failure", judge=1)
    assert old_parse_failure.format_reward == 0


def test_task_confidence_and_task_failure_metadata_do_not_add_reward_penalties():
    row = score([stop()], judge=1)
    row.task_completion = {"confidence": 0, "failed": True, "truncated": True}
    score_full_directory(EXAMPLE, row, CONFIG, judge=SimpleNamespace(score=lambda *_: 1))
    assert row.reward == 2
    assert row.auxiliary_reward == 0


def test_read_cost_is_unconditional_for_correct_incorrect_and_format_failed_rollouts():
    for judge in (0, 1):
        for steps in ([stop()], [read(), stop()]):
            row = score(steps, prediction="wrong", judge=judge, reads=3)
            assert row.raw_read_cost == row.read_cost == 0.03
            assert row.reward == pytest.approx(2 * judge - 1 + row.format_reward - 0.03)


def test_likelihood_diagnostics_do_not_change_reward_and_reward_mode_is_rejected():
    diagnostic = SimpleNamespace(score=lambda *_: {"mode": "diagnostic", "d": 100})
    row = score([stop()], judge=1, likelihood=diagnostic)
    assert row.likelihood == {"mode": "diagnostic", "d": 100}
    assert row.reward == 2
    with pytest.raises(ValueError, match="off/diagnostic"):
        score([stop()], likelihood=SimpleNamespace(score=lambda *_: {"mode": "reward", "d": 100}))


@pytest.mark.parametrize("key,value", [
    ("formula", "judge_f1_cost_v1"), ("read_call_cost", 0.05),
    ("read_call_cost", True), ("read_call_cost", float("nan")),
    ("repetition_penalty", 0.2), ("repetition_penalty", -0.1),
    ("format_penalty", 0.1), ("format_penalty_mode", "per_id"),
    ("visible_navigation_penalty", 0), ("visible_navigation_penalty", 0.1),
    ("likelihood", {"mode": "reward"}),
])
def test_configuration_cannot_silently_reintroduce_old_reward_composition(key, value):
    with pytest.raises(ValueError):
        validate_full_directory_reward({**CONFIG, key: value})


def test_defaults_and_explicit_null_visible_navigation_penalty_are_valid():
    validate_full_directory_reward({"formula": FORMULA})
    validate_full_directory_reward({**CONFIG, "visible_navigation_penalty": None})
    with pytest.raises(ValueError, match="max_ids_per_call=2"):
        validate_full_directory_reward(CONFIG, max_ids=1)


def test_metrics_report_independent_components_and_leave_legacy_rows_unchanged():
    clean = score([stop()], judge=1)
    bad = score([read("u", "u", invalid=["u"], error="max_read_calls"), stop("u")], judge=1)
    report = full_directory_metrics([clean, bad])
    assert report == pytest.approx({
        "format_accuracy": 0.5, "repeated_id_count": 0.5, "invalid_id_count": 1,
        "budget_violation_count": 0.5, "retrieval_violation_count": 1.5,
        "repetition_reward": -0.15, "cost_reward": 0,
    })
    assert full_directory_metrics([]) == {}
    assert full_directory_metrics([Trajectory("x", 0, "m")]) == {}


def test_provenance_versions_all_reward_components():
    provenance = full_directory_provenance(CONFIG)
    assert provenance["formula"] == FORMULA
    assert provenance["reader_tool_protocol"] == "hierarchical_full_v1"
    assert provenance["outcome_quality_formula"] == "1 if judge==1 else answer_f1"
    assert provenance["format_penalty_mode"] == "binary_protocol"
    assert provenance["reward_composition"] == "outcome_reward+format_reward-read_cost+repetition_reward"
    assert provenance["read_call_cost"] == 0.01
    assert provenance["repetition_penalty"] == 0.1


def test_opt_in_components_round_trip_and_do_not_extend_legacy_serialization():
    row = score([read("u", invalid=["u"]), stop()], judge=1, reads=1)
    assert row.to_dict()["retrieval_scoring"] == row.retrieval_scoring
    assert trajectory_from_dict(row.to_dict()).to_dict() == row.to_dict()
    assert "retrieval_scoring" not in Trajectory("x", 0, "m").to_dict()


def test_existing_judge_f1_reward_retains_gate_format_penalty_and_provenance():
    legacy = {"formula": "judge_f1_cost_v1", "judge_f1_alpha": 0.8,
              "read_call_cost": 0.05, "format_penalty": 0.1, "format_penalty_mode": "per_id"}
    judge = SimpleNamespace(score=lambda *_: 0, provenance=lambda: {"fixture": "legacy"})
    row = Trajectory(EXAMPLE.sample_id, 0, "m", prediction="red green", n_read_calls=1,
                     decisions=[read("unknown", invalid=["unknown"], labels=["unknown_id"]), stop()])
    original = deepcopy(legacy)
    score_trajectory(EXAMPLE, row, legacy, max_ids=2, judge=judge)
    assert row.answer_f1 == 0.8
    assert row.outcome == 0
    assert row.outcome_reward == -1
    assert row.format_reward == -0.1
    assert row.reward == pytest.approx(-1.15)
    assert not getattr(row, "retrieval_scoring", {})
    provenance = RewardScorer(legacy, judge=judge, max_ids=2).provenance()
    assert provenance["outcome_quality_formula"] == "judge*(alpha+(1-alpha)*answer_f1)"
    assert "repetition_penalty" not in provenance
    assert legacy == original
