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


def score_trajectory(example, trajectory, config, *, max_ids=4):
    """Apply one terminal reward, including at most one format penalty."""
    formula = config.get("formula", "outcome_cost_v1")
    if formula not in {"outcome_cost_v1", "outcome_gated_v2"}:
        raise ValueError(f"Unknown reward formula: {formula}")
    trajectory.outcome = outcome(example, trajectory.prediction)
    trajectory.raw_read_cost = config["read_call_cost"] * trajectory.n_read_calls
    trajectory.outcome_reward = trajectory.outcome
    trajectory.read_cost = trajectory.raw_read_cost
    if formula == "outcome_gated_v2":
        trajectory.outcome_reward = 2 * trajectory.outcome - 1
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
    # Standalone historical run configs lack this key: preserve their old reward.
    trajectory.format_reward = -config.get("format_penalty", 0.0) if trajectory.format_failure else 0.0
    trajectory.reward = trajectory.outcome_reward - trajectory.read_cost + trajectory.format_reward
    return trajectory


def advantages(rewards: list[float], epsilon: float = 1e-6) -> list[float]:
    if not rewards or any(not math.isfinite(value) for value in rewards):
        raise ValueError("Empty or non-finite rewards")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    return [(value - mean) / (std + epsilon) for value in rewards]


def protocol_metrics(rows):
    """Rates are per rollout; decision counts are means per rollout, never rewards."""
    n = len(rows)
    reasons = ("invalid_json_or_schema", "empty_ids", "too_many_ids", "duplicate_ids",
               "repaired_id", "unknown_id", "repeated_invalid_action")
    return {
        "id_repair_rate": sum(any((d.tool_result or {}).get("id_repairs") for d in r.decisions) for r in rows) / n,
        "unknown_id_rate": sum(any((d.tool_result or {}).get("invalid_ids") or
                                   (d.tool_result or {}).get("error", {}).get("code") == "unknown_or_unauthorized_id"
                                   for d in r.decisions) for r in rows) / n,
        "repeated_invalid_action_rate": sum(r.termination_reason == "repeated_invalid_action" for r in rows) / n,
        "empty_evidence_rate": sum(not r.evidence for r in rows) / n,
        **{f"format_decisions_{reason}": sum(reason in d.format_violations for r in rows for d in r.decisions) / n
           for reason in reasons},
    }
