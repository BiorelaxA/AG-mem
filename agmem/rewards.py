from collections import Counter
import math
import unicodedata

from .data import option_label


def text_tokens(text: str) -> list[str]:
    import regex
    text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    pattern = r"\p{Han}|[+-]?\d+(?:\.\d+)?(?:%|‰)?|[\p{L}\p{M}--\p{Han}]+(?:['’][\p{L}\p{M}--\p{Han}]+)*|[^\s\p{P}\p{Z}]"
    return [token for token in regex.findall(pattern, text, flags=regex.V1) if token not in {"a", "an", "the"}]


def text_f1(prediction: str, reference: str) -> float:
    left, right = text_tokens(prediction), text_tokens(reference)
    if not left or not right:
        return 0.0
    common = sum((Counter(left) & Counter(right)).values())
    return 2 * common / (len(left) + len(right))


def outcome(example, prediction: str) -> float:
    if not example.gold_answers:
        raise ValueError("No valid reference")
    if example.answer_mode == "choice":
        label = option_label(prediction)
        # Do not apply English article removal to the option label A.
        return float(label in example.options and label in example.gold_answers)
    return max(text_f1(prediction, reference) for reference in example.gold_answers)


def score_trajectory(example, trajectory, config, *, max_ids=4, judge=None, likelihood=None):
    """Apply one terminal reward, including at most one format penalty."""
    formula = config.get("formula", "outcome_cost_v1")
    if formula not in {"outcome_cost_v1", "outcome_gated_v2", "judge_gated_v1", "judge_cost_v2"}:
        raise ValueError(f"Unknown reward formula: {formula}")
    if formula == "judge_cost_v2" and config.get("likelihood", {}).get("mode") == "reward":
        raise ValueError("judge_cost_v2 supports likelihood off/diagnostic only")
    trajectory.outcome = outcome(example, trajectory.prediction)
    trajectory.answer_f1 = trajectory.outcome
    if formula in {"judge_gated_v1", "judge_cost_v2"}:
        if judge is None:
            raise ValueError(f"{formula} requires a Judge scorer")
        trajectory.judge_score = judge.score(example, trajectory.prediction)
        if trajectory.judge_score not in (0.0, 1.0):
            raise ValueError("Judge must return a binary score")
        trajectory.outcome = trajectory.judge_score
    trajectory.raw_read_cost = config["read_call_cost"] * trajectory.n_read_calls
    trajectory.outcome_reward = trajectory.outcome
    trajectory.read_cost = trajectory.raw_read_cost
    if formula in {"outcome_gated_v2", "judge_gated_v1", "judge_cost_v2"}:
        trajectory.outcome_reward = 2 * trajectory.outcome - 1
    if formula in {"outcome_gated_v2", "judge_gated_v1"}:
        trajectory.read_cost *= trajectory.outcome
    # Inspect every generated action, including the last decision after the read
    # budget is exhausted. A later valid stop does not erase an earlier bad batch.
    trajectory.format_failure = trajectory.termination_reason == "protocol_failure" or any(
        step.format_violations for step in trajectory.decisions
    ) or any(
        step.action is not None and not step.action["stop"] and (
            not 1 <= len(step.action["ids"]) <= max_ids
            or len(set(step.action["ids"])) != len(step.action["ids"])
        )
        for step in trajectory.decisions
    )
    if formula == "judge_cost_v2":
        # strict_v2 records normalized violations. Also recognize explicit tool
        # errors when scoring an older trajectory under this reward protocol.
        trajectory.format_failure |= any(_decision_violations(step) for step in trajectory.decisions)
    # Standalone historical run configs lack this key: preserve their old reward.
    trajectory.format_reward = -config.get("format_penalty", 0.0) if trajectory.format_failure else 0.0
    trajectory.reward = trajectory.outcome_reward - trajectory.read_cost + trajectory.format_reward
    trajectory.auxiliary_reward = 0.0
    if likelihood is not None:
        trajectory.likelihood = likelihood.score(example, trajectory.evidence)
        if trajectory.likelihood["mode"] == "reward":
            if formula == "judge_cost_v2":
                raise ValueError("judge_cost_v2 supports likelihood off/diagnostic only")
            coefficient = config["likelihood"]["coefficient"]
            trajectory.auxiliary_reward = coefficient * (1 - trajectory.outcome) * trajectory.likelihood["d"]
            trajectory.reward += trajectory.auxiliary_reward
    if not math.isfinite(trajectory.reward):
        raise ValueError("Non-finite composed reward")
    return trajectory


class RewardScorer:
    """Keep semantic judging, optional likelihood and reward composition separate."""
    def __init__(self, config, *, judge=None, likelihood=None, max_ids=4):
        self.config, self.judge, self.likelihood, self.max_ids = config, judge, likelihood, max_ids

    def score(self, example, trajectory):
        return score_trajectory(example, trajectory, self.config, max_ids=self.max_ids,
                                judge=self.judge, likelihood=self.likelihood)

    def provenance(self):
        from .services import TASK_OUTPUT_POLICY
        return {"formula": self.config.get("formula", "outcome_cost_v1"),
                "task_output_policy": TASK_OUTPUT_POLICY,
                "judge": self.judge.provenance() if self.judge else None,
                "likelihood": self.likelihood.provenance() if self.likelihood else None}

    def close(self):
        for scorer in (self.judge, self.likelihood):
            if scorer is not None and hasattr(scorer, "close"):
                scorer.close()


def advantages(rewards: list[float], epsilon: float = 1e-6) -> list[float]:
    if not rewards or any(not math.isfinite(value) for value in rewards):
        raise ValueError("Empty or non-finite rewards")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    return [(value - mean) / (std + epsilon) for value in rewards]


def _decision_violations(decision):
    """Normalize historical tool errors for diagnostics without changing old rewards."""
    result = decision.tool_result or {}
    reasons = set(decision.format_violations)
    code = result.get("error", {}).get("code")
    mapped = {"invalid_batch_size": "empty_ids", "too_many_ids": "too_many_ids",
              "duplicate_ids_in_batch": "duplicate_ids", "unknown_or_unauthorized_id": "unknown_id",
              "repeated_read": "repeated_read", "max_read_calls": "read_budget_exceeded"}.get(code)
    if mapped:
        reasons.add(mapped)
    if result.get("id_repairs"):
        reasons.add("repaired_id")
    if result.get("invalid_ids"):
        reasons.add("unknown_id")
    if result.get("already_read_ids"):
        reasons.add("repeated_read")
    if result.get("ids_truncated"):
        reasons.add("too_many_ids")
    action = decision.action
    if action is not None and not action["stop"]:
        if not action["ids"]:
            reasons.add("empty_ids")
        if len(set(action["ids"])) != len(action["ids"]):
            reasons.add("duplicate_ids")
    return reasons


def protocol_metrics(rows):
    """Rollout rates and mean decision counts; *_call_rate uses executed reads."""
    n = len(rows) or 1
    reasons = ("invalid_json_or_schema", "empty_ids", "too_many_ids", "duplicate_ids",
               "repaired_id", "unknown_id", "repeated_invalid_action", "repeated_read",
               "read_budget_exceeded")
    counts, rollouts = Counter(), Counter()
    read_calls = repeated_calls = no_new_calls = 0
    for row in rows:
        seen, row_reasons = set(), set()
        for decision in row.decisions:
            violations = _decision_violations(decision)
            result = decision.tool_result or {}
            action = decision.action
            requested = result.get("selected_ids", result.get("requested_ids", (action or {}).get("ids", [])))
            is_read = action is not None and not action["stop"]
            repeated = bool(result.get("already_read_ids") or (is_read and set(requested) & seen)
                            or "repeated_read" in violations)
            if repeated:
                violations.add("repeated_read")
            counts.update(violations)
            row_reasons.update(violations)
            # Legacy protocols omit read_call_consumed, but a populated tool
            # result for a read action denotes an executed call in their logs.
            consumed = result.get("read_call_consumed", bool(result) and is_read)
            if not consumed:
                continue
            returned = {record["id"] for record in result.get("records", [])}
            new_ids = set(result.get("new_ids", returned - seen))
            read_calls += 1
            repeated_calls += repeated
            no_new_calls += bool(result.get("no_new_evidence", not new_ids))
            seen.update(new_ids)
        rollouts.update(row_reasons)
    return {
        "id_repair_rate": rollouts["repaired_id"] / n,
        "unknown_id_rate": rollouts["unknown_id"] / n,
        "repeated_read_rate": rollouts["repeated_read"] / n,
        "repeated_read_calls": repeated_calls / n,
        "repeated_read_call_rate": repeated_calls / read_calls if read_calls else 0.0,
        "no_new_evidence_calls": no_new_calls / n,
        "no_new_evidence_call_rate": no_new_calls / read_calls if read_calls else 0.0,
        "repeated_invalid_action_rate": sum(r.termination_reason == "repeated_invalid_action" for r in rows) / n,
        "empty_evidence_rate": sum(not r.evidence for r in rows) / n,
        **{f"format_decisions_{reason}": counts[reason] / n for reason in reasons},
    }
