"""Isolated reward composition for the opt-in full-directory Reader protocol.

The terminal reward is ``(2*q - 1) + format_ok - .01*reads - .1*bad``.
Semantic retrieval mistakes never change the binary protocol-format reward.
ID diagnostics may overlap, but ``bad`` charges each requested position at
most once, then adds one for each READ attempted beyond the call budget.
"""
from __future__ import annotations

import math


FORMULA = "full_directory_orm_v1"
DIRECTORY_PROTOCOL = "hierarchical_full_v1"
FORMAT_VIOLATIONS = frozenset({
    "invalid_json_or_schema", "empty_ids", "stop_with_ids", "too_many_ids",
})
_COUNT_KEYS = (
    "repeated_id_count", "invalid_id_count", "budget_violation_count",
    "retrieval_violation_count",
)


def validate_full_directory_reward(config, max_ids=2):
    """Reject configuration that silently changes this version's reward rule."""
    if config.get("formula", FORMULA) != FORMULA:
        raise ValueError(f"Expected reward.formula={FORMULA}")
    if type(max_ids) is not int or max_ids != 2:
        raise ValueError(f"{FORMULA} requires max_ids_per_call=2")
    for key, expected in (("read_call_cost", 0.01), ("repetition_penalty", 0.1),
                          ("format_penalty", 0.0)):
        value = config.get(key, expected)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value != expected):
            raise ValueError(f"{FORMULA} requires reward.{key}={expected}")
    if config.get("format_penalty_mode", "binary_protocol") != "binary_protocol":
        raise ValueError(f"{FORMULA} requires reward.format_penalty_mode=binary_protocol")
    if config.get("visible_navigation_penalty") is not None:
        raise ValueError(f"{FORMULA} does not support reward.visible_navigation_penalty")
    if config.get("likelihood", {}).get("mode") == "reward":
        raise ValueError(f"{FORMULA} supports likelihood off/diagnostic only")


def _valid_action_schema(action):
    return (isinstance(action, dict) and set(action) == {"ids", "stop"}
            and type(action["stop"]) is bool and isinstance(action["ids"], list)
            and all(isinstance(key, str) for key in action["ids"]))


def _format_reasons(decision, *, max_ids, missing_parse_failure=False):
    """Normalize only syntax/schema rules, without importing semantic labels."""
    reasons = set(decision.format_violations) & FORMAT_VIOLATIONS
    result = decision.tool_result or {}
    code = (result.get("error") or {}).get("code")
    mapped = {"invalid_json_or_schema": "invalid_json_or_schema",
              "invalid_batch_size": "empty_ids", "empty_ids": "empty_ids",
              "too_many_ids": "too_many_ids", "stop_with_ids": "stop_with_ids"}.get(code)
    if mapped:
        reasons.add(mapped)
    action = decision.action
    if action is None:
        if missing_parse_failure:
            reasons.add("invalid_json_or_schema")
    elif not _valid_action_schema(action):
        reasons.add("invalid_json_or_schema")
    elif action["stop"]:
        if action["ids"]:
            reasons.add("stop_with_ids")
    else:
        if not action["ids"]:
            reasons.add("empty_ids")
        if len(action["ids"]) > max_ids:
            reasons.add("too_many_ids")
    return reasons


def _retrieval_components(trajectory, *, max_ids):
    counts = dict.fromkeys(_COUNT_KEYS, 0)
    successful_reads = set()
    format_count = 0
    for index, decision in enumerate(trajectory.decisions):
        reasons = _format_reasons(
            decision, max_ids=max_ids,
            missing_parse_failure=(index == len(trajectory.decisions) - 1
                                   and trajectory.termination_reason == "protocol_failure"),
        )
        format_count += bool(reasons)
        action, result = decision.action, decision.tool_result or {}
        # STOP IDs and unparseable/schema-invalid output are format-only. They
        # cannot be converted into READ requests for extra ID/budget penalties.
        if (not _valid_action_schema(action) or action["stop"]
                or "invalid_json_or_schema" in reasons):
            continue
        invalid = set(result.get("invalid_ids", []))
        already_read = successful_reads | set(result.get("already_read_ids", []))
        batch_seen = set()
        for key in action["ids"]:
            repeated = key in already_read or key in batch_seen
            unknown = key in invalid
            counts["repeated_id_count"] += int(repeated)
            counts["invalid_id_count"] += int(unknown)
            counts["retrieval_violation_count"] += int(repeated or unknown)
            batch_seen.add(key)
        budget_exceeded = ("read_budget_exceeded" in decision.format_violations
                           or (result.get("error") or {}).get("code")
                           in {"max_read_calls", "read_budget_exceeded"})
        counts["budget_violation_count"] += int(budget_exceeded)
        counts["retrieval_violation_count"] += int(budget_exceeded)
        # Only successfully returned evidence enters history. Failed capacity
        # checks, invalid IDs and an unexecuted excess suffix do not mark reads.
        successful_reads.update(record["id"] for record in result.get("records", []))
    return format_count, counts


def score_full_directory(example, trajectory, config, *, max_ids=2, judge=None, likelihood=None):
    """Score J=1 as q=1; otherwise use raw word F1, with no evidence gate."""
    # Lazy import permits rewards.py to dispatch to this isolated module.
    from .rewards import word_outcome

    validate_full_directory_reward(config, max_ids=max_ids)
    if judge is None:
        raise ValueError(f"{FORMULA} requires a Judge scorer")
    judge_score = judge.score(example, trajectory.prediction)
    if (isinstance(judge_score, bool) or not isinstance(judge_score, (int, float))
            or judge_score not in (0, 1)):
        raise ValueError("Judge must return a binary score (numeric 0 or 1)")
    answer_f1 = word_outcome(example, trajectory.prediction)
    diagnostic = likelihood.score(example, trajectory.evidence) if likelihood is not None else None
    if diagnostic is not None and diagnostic["mode"] == "reward":
        raise ValueError(f"{FORMULA} supports likelihood off/diagnostic only")
    format_count, retrieval = _retrieval_components(trajectory, max_ids=max_ids)
    trajectory.answer_f1 = answer_f1
    trajectory.judge_score = float(judge_score)
    trajectory.outcome = 1.0 if judge_score == 1 else answer_f1
    trajectory.outcome_reward = 2 * trajectory.outcome - 1
    trajectory.format_violation_count = format_count
    trajectory.format_failure = bool(format_count)
    trajectory.format_reward = float(not trajectory.format_failure)
    trajectory.raw_read_cost = config.get("read_call_cost", 0.01) * trajectory.n_read_calls
    trajectory.read_cost = trajectory.raw_read_cost
    repetition_reward = -config.get("repetition_penalty", 0.1) * retrieval["retrieval_violation_count"]
    trajectory.retrieval_scoring = {
        "protocol": FORMULA,
        **retrieval,
        "repetition_reward": repetition_reward,
    }
    trajectory.auxiliary_reward = 0.0
    if diagnostic is not None:
        trajectory.likelihood = diagnostic
    trajectory.reward = (trajectory.outcome_reward + trajectory.format_reward
                         - trajectory.read_cost + repetition_reward)
    if not math.isfinite(trajectory.reward):
        raise ValueError("Non-finite composed reward")
    return trajectory


def full_directory_provenance(config):
    """Version each independently reported component for reproducible runs."""
    from .rewards import WORD_F1_PROTOCOL

    validate_full_directory_reward(config)
    return {
        "formula": FORMULA,
        "reader_tool_protocol": DIRECTORY_PROTOCOL,
        "answer_f1_protocol": WORD_F1_PROTOCOL,
        "primary_outcome_metric": "outcome_q",
        "outcome_quality_formula": "1 if judge==1 else answer_f1",
        "outcome_reward_transform": "2*q-1",
        "read_call_cost": config.get("read_call_cost", 0.01),
        "repetition_penalty": config.get("repetition_penalty", 0.1),
        "format_penalty_mode": "binary_protocol",
        "format_counting_version": "full_directory_binary_protocol_v1",
        "format_reward_formula": "1 if all_reader_actions_protocol_valid else 0",
        "retrieval_counting_version": "full_directory_id_position_union_plus_budget_v1",
        "reward_composition": "outcome_reward+format_reward-read_cost+repetition_reward",
    }


def full_directory_metrics(rows):
    """Return means for opt-in rows, leaving all historical reports unchanged.

    Repeated and invalid ID diagnostics can overlap. The retrieval total is
    their per-position union plus budget violations, never their naive sum.
    """
    scored = [row for row in rows
              if getattr(row, "retrieval_scoring", {}).get("protocol") == FORMULA]
    if not scored:
        return {}
    n = len(scored)
    return {
        "format_accuracy": sum(not row.format_failure for row in scored) / n,
        "cost_reward": -sum(row.read_cost for row in scored) / n,
        **{key: sum(row.retrieval_scoring[key] for row in scored) / n for key in _COUNT_KEYS},
        "repetition_reward": sum(row.retrieval_scoring["repetition_reward"] for row in scored) / n,
    }
