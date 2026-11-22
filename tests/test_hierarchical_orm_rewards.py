"""Opt-in word F1 reward and the distinction between navigation and evidence."""
from unittest.mock import Mock

import pytest

from agmem.core import Completion, Decision, Example, Trajectory
from agmem.rewards import RewardScorer, protocol_metrics, score_trajectory, text_f1, word_f1, word_tokens


def reward_config():
    return {"formula": "f1_cost_v1", "read_call_cost": 0.05,
            "format_penalty": 0.5, "format_penalty_mode": "per_id"}


def example(*gold):
    return Example("locomo/train/q", "locomo", "train", "m", "question", gold_answers=list(gold))


def decision(ids, kind="segment", *, records=(), already=(), invalid=(), violations=(), noop=False):
    return Decision([], Completion("action", [1], [2], [-0.1]),
                    {"ids": list(ids), "stop": False},
                    {"action_kind": kind, "ok": not already and not invalid,
                     "requested_ids": list(ids), "selected_ids": list(ids[:1]),
                     "records": [{"id": key} for key in records], "new_ids": list(records),
                     "already_read_ids": list(already), "invalid_ids": list(invalid),
                     "no_new_evidence": not records, "read_call_consumed": True,
                     "directory_navigation_success": kind == "navigation",
                     "visible_navigation_noop": noop}, format_violations=list(violations))


def test_word_f1_uses_requested_normalization_and_multiset_overlap():
    assert word_tokens("THE cat, cat! — dog's") == ["the", "cat", "cat", "dogs"]
    assert word_f1("THE cat, cat!", "the CAT") == pytest.approx(0.8)
    assert word_f1("the cat", "cat") == pytest.approx(2 / 3)
    assert text_f1("the cat", "cat") == 1.0  # Legacy article removal is unchanged.
    assert word_f1("cat cat cat", "cat") == 0.5
    assert word_f1("", "") == 0.0
    assert word_f1("!!!", "the") == 0.0
    assert word_f1("New-York", "newyork") == 1.0


def test_f1_reward_max_alternative_gold_unconditional_cost_and_no_judge():
    judge = Mock()
    judge.score.side_effect = AssertionError("F1 reward must not call Judge")
    judge.provenance.side_effect = AssertionError("F1 provenance must not call Judge")
    scorer = RewardScorer(reward_config(), judge=judge, max_ids=1)
    row = Trajectory("q", 0, "m", prediction="THE cat", n_read_calls=2, judge_score=0.0)
    scorer.score(example("a dog", "the cat"), row)
    assert row.answer_f1 == row.outcome == 1.0
    assert row.judge_score is None
    assert row.outcome_reward == 1.0
    assert row.read_cost == row.raw_read_cost == 0.1
    assert row.reward == pytest.approx(0.9)
    assert scorer.provenance()["judge"] is None
    judge.score.assert_not_called()
    row.prediction = "unrelated"
    scorer.score(example("the cat"), row)
    assert row.answer_f1 == 0.0
    assert row.reward == pytest.approx(-1.1)  # Cost is not multiplied by F1.


def test_old_navigation_trace_without_new_metadata_keeps_its_original_reward():
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=5, decisions=[
        decision(["p1"], "navigation"),
        decision(["p1.c1"], records=["p1.c1"]),
        decision(["p1"], "navigation", noop=True),
        # Normalize an explicitly classified navigation even if stale logs
        # contain the old repeated_read label.
        decision(["p1"], "navigation", already=["p1"], violations=["repeated_read"], noop=True),
        decision(["p1.c1"], already=["p1.c1"], violations=["repeated_read"]),
    ])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 1
    assert row.format_reward == -0.5
    assert row.read_cost == 0.25
    assert row.reward == 0.25
    metrics = protocol_metrics([row])
    assert metrics["repeated_read_calls"] == 1
    assert metrics["format_decisions_repeated_read"] == 1
    assert metrics["navigation_calls"] == 3
    assert metrics["navigation_repeat_calls"] == 2
    assert metrics["visible_navigation_noop_calls"] == 2
    assert metrics["segment_calls"] == 2
    assert metrics["no_new_evidence_calls"] == 1
    assert metrics["segment_no_new_evidence_call_rate"] == 0.5
    assert metrics["visible_navigation_penalty_calls"] == 0


def visible_parent(parent="p1", *, success=True):
    step = decision([parent], "navigation", violations=["visible_navigation"], noop=True)
    step.tool_result.update(visible_navigation_ids=[parent], directory_navigation_success=success)
    return step


def test_each_marked_visible_parent_request_costs_one_format_penalty():
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=3, decisions=[
        decision(["p1"], "navigation"), visible_parent(), visible_parent(),
    ])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 2
    assert row.format_reward == -1.0
    assert row.read_cost == pytest.approx(0.15)
    assert row.reward == pytest.approx(-0.15)
    metrics = protocol_metrics([row])
    assert metrics["visible_navigation_penalty_calls"] == 2
    assert metrics["format_decisions_visible_navigation"] == 2
    assert metrics["repeated_read_calls"] == 0  # Parent repetition is a separate violation.


@pytest.mark.parametrize("intervening", ["segment", "navigation"])
def test_intervening_action_does_not_exempt_a_parent_that_remains_visible(intervening):
    middle = (decision(["p1.c1"], records=["p1.c1"]) if intervening == "segment"
              else decision(["p2"], "navigation"))
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=3, decisions=[
        decision(["p1"], "navigation"), middle, visible_parent(),
    ])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 1
    assert row.reward == pytest.approx(0.35)
    metrics = protocol_metrics([row])
    assert metrics["navigation_repeat_calls"] == 1
    assert metrics["visible_navigation_penalty_calls"] == 1


def test_evicted_parent_reopen_has_no_visible_navigation_penalty():
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=4, decisions=[
        decision(["p1"], "navigation"), decision(["p2"], "navigation"),
        decision(["p3"], "navigation"), decision(["p1"], "navigation"),
    ])
    for step in row.decisions:
        step.tool_result["visible_navigation_ids"] = []
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 0
    assert row.reward == pytest.approx(0.8)
    metrics = protocol_metrics([row])
    assert metrics["navigation_repeat_calls"] == 1
    assert metrics["visible_navigation_penalty_calls"] == 0


def test_failed_initial_navigation_does_not_make_retry_visible():
    failed = decision(["p1"], "navigation")
    failed.tool_result.update(directory_navigation_success=False, visible_navigation_ids=[],
                              error={"code": "context_capacity_exceeded"})
    retry = decision(["p1"], "navigation")
    retry.tool_result["visible_navigation_ids"] = []
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=2, decisions=[failed, retry])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 0
    assert row.reward == pytest.approx(0.9)
    assert protocol_metrics([row])["visible_navigation_penalty_calls"] == 0


def test_child_repeat_and_visible_parent_penalties_accumulate_separately():
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=4, decisions=[
        decision(["p1"], "navigation"), visible_parent(),
        decision(["p1.c1"], records=["p1.c1"]),
        decision(["p1.c1"], already=["p1.c1"], violations=["repeated_read"]),
    ])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 2
    assert row.reward == pytest.approx(-0.2)
    metrics = protocol_metrics([row])
    assert metrics["repeated_read_calls"] == 1
    assert metrics["visible_navigation_penalty_calls"] == 1


def test_visible_navigation_overlapping_id_labels_do_not_double_charge_positions():
    step = decision(["p1", "p1"], "navigation", invalid=["p1"],
                    violations=["visible_navigation", "unknown_id", "duplicate_ids", "too_many_ids"])
    step.tool_result["visible_navigation_ids"] = ["p1"]
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=2, decisions=[
        decision(["p1"], "navigation"), step,
    ])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 2  # Two positions, regardless of labels per position.
    assert row.reward == pytest.approx(-0.1)


@pytest.mark.parametrize("metadata", ["ids_only", "ids_and_label", "ids_and_error_code"])
def test_visible_navigation_explicit_metadata_forms_are_normalized(metadata):
    step = decision(["p1"], "navigation")
    step.tool_result["visible_navigation_ids"] = ["p1"]
    if metadata == "ids_and_label":
        step.format_violations = ["visible_navigation"]
    elif metadata == "ids_and_error_code":
        step.tool_result["error"] = {"code": "visible_navigation"}
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=1, decisions=[step])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 1
    assert protocol_metrics([row])["format_decisions_visible_navigation"] == 1


@pytest.mark.parametrize("first_kind", ["segment", "invalid"])
def test_visible_parent_suffix_is_counted_even_when_first_action_is_not_navigation(first_kind):
    first_id = "p1.c1" if first_kind == "segment" else "unknown"
    step = decision([first_id, "p1", "p1", "p2"], first_kind,
                    invalid=[first_id] if first_kind == "invalid" else [],
                    records=[first_id] if first_kind == "segment" else [],
                    violations=["too_many_ids", "duplicate_ids", "visible_navigation"])
    step.tool_result["visible_navigation_ids"] = ["p1", "p2"]
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=1, decisions=[step])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == (3 if first_kind == "segment" else 4)
    metrics = protocol_metrics([row])
    assert metrics["visible_navigation_penalty_calls"] == 1
    assert metrics["format_decisions_visible_navigation"] == 1
    assert metrics["visible_navigation_noop_calls"] == 0
    assert metrics["repeated_read_calls"] == 0


def test_visible_navigation_counts_failed_attempts_but_not_uncharged_overbudget_call():
    failed = visible_parent(success=False)
    failed.tool_result["error"] = {"code": "context_capacity_exceeded"}
    overbudget = visible_parent(success=False)
    overbudget.tool_result["read_call_consumed"] = False
    overbudget.format_violations.append("read_budget_exceeded")
    row = Trajectory("q", 0, "m", n_read_calls=1, decisions=[failed, overbudget])
    metrics = protocol_metrics([row])
    assert metrics["visible_navigation_penalty_calls"] == 1
    assert metrics["format_decisions_visible_navigation"] == 2
    assert metrics["navigation_calls"] == 1


def test_navigation_format_protocol_version_changes_only_new_f1_reward():
    assert RewardScorer(reward_config()).provenance()["format_counting_version"] == (
        "hierarchical_reader_id_with_protocol_v4")
    flat = {**reward_config(), "formula": "judge_cost_v2"}
    assert RewardScorer(flat).provenance()["format_counting_version"] == "reader_id_with_protocol_v2"


@pytest.mark.parametrize("mode", ["per_id", "per_decision", "per_rollout"])
@pytest.mark.parametrize("metadata", ["label", "error", "both"])
def test_explicit_nonempty_stop_violation_counts_once_without_id_penalties(mode, metadata):
    step = Decision([], Completion("stop", [1], [2], [-0.1]),
                    {"ids": ["p1:c1", "p1:c1", "unknown"], "stop": True},
                    {"directory_protocol": "hierarchical_v1", "action_kind": "stop",
                     "read_call_consumed": False, "records": [], "new_ids": []})
    if metadata in {"label", "both"}:
        step.format_violations = ["stop_with_ids"]
    if metadata in {"error", "both"}:
        step.tool_result["error"] = {"code": "stop_with_ids"}
    row = Trajectory("q", 0, "m", prediction="answer", decisions=[step], termination_reason="stop")
    score_trajectory(example("answer"), row,
                     {**reward_config(), "format_penalty_mode": mode}, max_ids=1)
    assert row.format_violation_count == 1
    assert row.format_failure
    assert row.format_reward == -0.5
    assert row.read_cost == 0
    assert row.reward == 0.5
    metrics = protocol_metrics([row])
    assert metrics["format_decisions_stop_with_ids"] == 1
    assert metrics["stop_with_ids_rate"] == 1
    assert metrics["format_decisions_too_many_ids"] == 0
    assert metrics["format_decisions_duplicate_ids"] == 0
    assert metrics["format_decisions_unknown_id"] == 0
    assert metrics["navigation_calls"] == metrics["segment_calls"] == 0


@pytest.mark.parametrize("hierarchical", [False, True])
def test_unmarked_historical_nonempty_stop_is_not_reinterpreted(hierarchical):
    result = {"action_kind": "stop", "read_call_consumed": False, "records": [], "new_ids": []}
    if hierarchical:
        result["directory_protocol"] = "hierarchical_v1"
    step = Decision([], Completion("stop", [1], [2], [-0.1]),
                    {"ids": ["unknown", "unknown"], "stop": True}, result)
    row = Trajectory("q", 0, "m", prediction="answer", decisions=[step], termination_reason="stop")
    config = reward_config() if hierarchical else {**reward_config(), "formula": "judge_cost_v2"}
    score_trajectory(example("answer"), row, config, max_ids=1,
                     judge=None if hierarchical else Mock(score=lambda *_: 1.0))
    assert row.reward == 1
    assert not row.format_failure
    assert row.format_violation_count == 0
    assert protocol_metrics([row]).get("format_decisions_stop_with_ids", 0) == 0


def test_nonempty_stop_metrics_are_rollout_weighted_and_reach_source_macro():
    from agmem.pipeline import metrics

    bad = Decision([], Completion("stop", [1], [2], [-0.1]),
                   {"ids": ["p1:c1"], "stop": True},
                   {"directory_protocol": "hierarchical_v1", "action_kind": "stop",
                    "read_call_consumed": False}, format_violations=["stop_with_ids"])
    good = Decision([], Completion("stop", [1], [2], [-0.1]),
                    {"ids": [], "stop": True},
                    {"directory_protocol": "hierarchical_v1", "action_kind": "stop",
                     "read_call_consumed": False})
    rows = [Trajectory("locomo/train/q1", 0, "m", decisions=[bad]),
            Trajectory("locomo/train/q2", 0, "m", decisions=[good])]
    for row in rows:
        score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    report = metrics(rows)
    for scope in (report["pooled"], report["by_source"]["locomo"], report["source_macro"]):
        assert scope["format_decisions_stop_with_ids"] == 0.5
        assert scope["stop_with_ids_rate"] == 0.5


@pytest.mark.parametrize("mode", ["per_id", "per_decision", "per_rollout"])
def test_superseded_consecutive_markers_are_explicitly_ignored_by_new_rule(mode):
    step = decision(["p1"], "navigation", violations=["consecutive_navigation"], noop=True)
    step.tool_result.update(consecutive_navigation_ids=["p1"],
                            error={"code": "consecutive_navigation"})
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=1, decisions=[step], reward=-0.55)
    metrics = protocol_metrics([row])
    assert row.reward == -0.55  # Reading diagnostics does not rewrite historical rewards.
    assert metrics["visible_navigation_penalty_calls"] == 0
    assert not any("consecutive_navigation" in key for key in metrics)
    score_trajectory(example("answer"), row, {**reward_config(), "format_penalty_mode": mode}, max_ids=1)
    assert row.format_violation_count == 0
    assert row.format_failure is False
    assert row.reward == pytest.approx(0.95)


def test_navigation_exemption_does_not_waive_excess_duplicate_or_unknown_ids():
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=2, decisions=[
        decision(["p1"], "navigation"),
        decision(["p1", "p1", "unknown"], "navigation", already=["p1"], invalid=["unknown"],
                 violations=["too_many_ids", "duplicate_ids", "unknown_id"]),
    ])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 2  # Each offending position counts once.
    assert row.reward == pytest.approx(-0.1)


def test_excess_already_read_child_is_not_hidden_by_parent_navigation_classification():
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=3, decisions=[
        decision(["p1"], "navigation"),
        decision(["p1.c1"], records=["p1.c1"]),
        decision(["p1", "p1.c1"], "navigation", already=["p1.c1"],
                 violations=["too_many_ids", "repeated_read"]),
    ])
    score_trajectory(example("answer"), row, reward_config(), max_ids=1)
    assert row.format_violation_count == 1  # Excess and repeat at the same position.
    metrics = protocol_metrics([row])
    assert metrics["format_decisions_repeated_read"] == 1
    assert metrics["repeated_read_calls"] == 1
    assert metrics["navigation_repeat_calls"] == 1


def test_legacy_repeat_penalty_does_not_change_without_navigation_metadata():
    repeated = decision(["id1"], already=["id1"], violations=["repeated_read"])
    repeated.tool_result.pop("action_kind")
    row = Trajectory("q", 0, "m", prediction="answer", n_read_calls=1, decisions=[repeated])
    config = {**reward_config(), "formula": "judge_cost_v2"}
    score_trajectory(example("answer"), row, config, max_ids=1, judge=Mock(score=lambda *_: 1.0))
    assert row.judge_score == 1.0
    assert row.format_violation_count == 1
    assert row.reward == pytest.approx(0.45)
    assert "navigation_calls" not in protocol_metrics([row])


def test_unconsumed_budget_exhaustion_is_not_another_navigation_call():
    step = decision(["p1"], "navigation", noop=True)
    step.tool_result["read_call_consumed"] = False
    assert protocol_metrics([Trajectory("q", 0, "m", decisions=[step])])["navigation_calls"] == 0


@pytest.mark.parametrize("action", [{"ids": [], "stop": True}, None])
def test_zero_read_hierarchical_trajectory_still_has_zero_navigation_metrics(action):
    step = Decision([], Completion("stop or malformed", [1], [2], [-0.1]), action,
                    {"directory_protocol": "hierarchical_v1", "read_call_consumed": False,
                     "records": [], "new_ids": []})
    metrics = protocol_metrics([Trajectory("q", 0, "m", decisions=[step])])
    for name in ("navigation_calls", "navigation_repeat_calls", "visible_navigation_noop_calls",
                 "segment_calls", "segment_no_new_evidence_calls", "segment_no_new_evidence_call_rate",
                 "visible_navigation_penalty_calls", "format_decisions_visible_navigation"):
        assert metrics[name] == 0.0
