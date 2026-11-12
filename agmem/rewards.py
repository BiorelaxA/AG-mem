from collections import Counter
import math
import string
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


WORD_F1_PROTOCOL = "lowercase_remove_punctuation_word_multiset_v1"


def word_tokens(text: str) -> list[str]:
    """Requested word-overlap protocol: lowercase, remove punctuation, split.

    Keep articles and repeated words. Punctuation is removed, not replaced by
    spaces (``well-known`` becomes ``wellknown``). Unlike the historical F1,
    this does not apply NFKC, character-level Han tokenization or article removal.
    """
    return "".join(char for char in text.lower()
                   if char not in string.punctuation
                   and not unicodedata.category(char).startswith("P")).split()


def word_f1(prediction: str, reference: str) -> float:
    left, right = word_tokens(prediction), word_tokens(reference)
    if not left or not right:
        return 0.0
    common = sum((Counter(left) & Counter(right)).values())
    return 2 * common / (len(left) + len(right))


def word_outcome(example, prediction: str) -> float:
    if not example.gold_answers:
        raise ValueError("No valid reference")
    if example.answer_mode == "choice":
        return outcome(example, prediction)
    # gold_answers are alternative valid references, not required fact fragments.
    return max(word_f1(prediction, reference) for reference in example.gold_answers)


def outcome(example, prediction: str) -> float:
    if not example.gold_answers:
        raise ValueError("No valid reference")
    if example.answer_mode == "choice":
        label = option_label(prediction)
        # Do not apply English article removal to the option label A.
        return float(label in example.options and label in example.gold_answers)
    return max(text_f1(prediction, reference) for reference in example.gold_answers)


def validate_format_penalty_mode(config):
    """Keep historical rewards unchanged unless counting is explicitly enabled."""
    mode = config.get("format_penalty_mode", "per_rollout")
    if not isinstance(mode, str) or mode not in {"per_rollout", "per_decision", "per_id"}:
        raise ValueError("reward.format_penalty_mode must be per_rollout, per_decision or per_id")
    if mode in {"per_decision", "per_id"} and config.get("formula") not in {"judge_cost_v2", "f1_cost_v1"}:
        raise ValueError(f"reward.format_penalty_mode={mode} requires judge_cost_v2 or f1_cost_v1")
    return mode


def format_violation_count(trajectory, *, max_ids=4, exclude_id_violations=False):
    """Count violating Reader decisions, never labels or IDs within a decision.

    When counting IDs separately, exclude their labels to avoid charging
    the old flat penalty on top of the per-ID count. Other protocol violations
    still count once per decision. Tool capacity/service failures are not
    protocol violations. A final malformed
    action in an older trace may lack its explicit invalid_json_or_schema label;
    its protocol_failure termination identifies that one failed decision.
    """
    count = 0
    for index, decision in enumerate(trajectory.decisions):
        reasons = _decision_violations(decision)
        if exclude_id_violations:
            reasons -= {"too_many_ids", "duplicate_ids", "unknown_id", "repeated_read", "visible_navigation"}
        violated = bool(reasons)
        action = decision.action
        if not exclude_id_violations and action is not None and not action["stop"]:
            violated |= len(action["ids"]) > max_ids
        if (index == len(trajectory.decisions) - 1 and action is None
                and trajectory.termination_reason == "protocol_failure"):
            violated = True
        count += int(violated)
    return count


def invalid_id_count(trajectory, *, max_ids=4):
    """Count bad ID occurrences, with at most one penalty per list position.

    Excess positions, duplicate occurrences after the first, unknown IDs,
    IDs already successfully read and explicitly marked currently visible
    parent navigation qualify. Strict-v2 tool results identify
    invalid IDs across the full request, including its unexecuted suffix.
    Malformed actions, empty requests and budget exhaustion add no ID penalty.
    """
    count, successful_reads = 0, set()
    for decision in trajectory.decisions:
        action = decision.action
        result = decision.tool_result or {}
        if action is not None and not action["stop"]:
            invalid = set(result.get("invalid_ids", []))
            already_read = successful_reads | set(result.get("already_read_ids", []))
            already_read.discard(_navigation_id(decision))
            # Visibility is the environment's pre-action state, including all
            # requested IDs. Never infer it from older navigation histories.
            visible_navigation = set(result.get("visible_navigation_ids", []))
            batch_seen = set()
            for index, key in enumerate(action["ids"]):
                count += int(index >= max_ids or key in batch_seen or key in invalid
                             or key in already_read or key in visible_navigation)
                batch_seen.add(key)
            # Failed capacity checks and truncated/excess requests never enter
            # successful read history; only returned records do.
            successful_reads.update(record["id"] for record in result.get("records", []))
    return count


def score_trajectory(example, trajectory, config, *, max_ids=4, judge=None, likelihood=None):
    """Apply terminal reward with the configured rollout/decision/ID format count."""
    formula = config.get("formula", "outcome_cost_v1")
    if formula not in {"outcome_cost_v1", "outcome_gated_v2", "judge_gated_v1", "judge_cost_v2", "f1_cost_v1"}:
        raise ValueError(f"Unknown reward formula: {formula}")
    if formula in {"judge_cost_v2", "f1_cost_v1"} and config.get("likelihood", {}).get("mode") == "reward":
        raise ValueError(f"{formula} supports likelihood off/diagnostic only")
    penalty_mode = validate_format_penalty_mode(config)
    trajectory.outcome = (word_outcome if formula == "f1_cost_v1" else outcome)(example, trajectory.prediction)
    trajectory.answer_f1 = trajectory.outcome
    if formula == "f1_cost_v1":
        # Explicitly clear any stale scorer value when rescoring a saved trace.
        trajectory.judge_score = None
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
    if formula in {"outcome_gated_v2", "judge_gated_v1", "judge_cost_v2", "f1_cost_v1"}:
        trajectory.outcome_reward = 2 * trajectory.outcome - 1
    if formula in {"outcome_gated_v2", "judge_gated_v1"}:
        trajectory.read_cost *= trajectory.outcome
    # Inspect every generated action, including the last decision after the read
    # budget is exhausted. A later valid stop does not erase an earlier bad batch.
    trajectory.format_failure = trajectory.termination_reason == "protocol_failure" or any(
        (_decision_violations(step) if formula == "f1_cost_v1" else step.format_violations)
        for step in trajectory.decisions
    ) or any(
        step.action is not None and not step.action["stop"] and (
            not 1 <= len(step.action["ids"]) <= max_ids
            or len(set(step.action["ids"])) != len(step.action["ids"])
        )
        for step in trajectory.decisions
    )
    if formula in {"judge_cost_v2", "f1_cost_v1"}:
        # strict_v2 records normalized violations. Also recognize explicit tool
        # errors when scoring an older trajectory under this reward protocol.
        trajectory.format_failure |= any(_decision_violations(step) for step in trajectory.decisions)
    if penalty_mode == "per_id":
        trajectory.format_violation_count = (
            invalid_id_count(trajectory, max_ids=max_ids)
            + format_violation_count(trajectory, max_ids=max_ids, exclude_id_violations=True)
        )
    else:
        trajectory.format_violation_count = format_violation_count(trajectory, max_ids=max_ids)
    if penalty_mode in {"per_decision", "per_id"}:
        trajectory.format_failure = trajectory.format_violation_count > 0
        penalty_count = trajectory.format_violation_count
    else:
        penalty_count = int(trajectory.format_failure)
    # Standalone historical run configs lack this key: preserve their old reward.
    trajectory.format_reward = -config.get("format_penalty", 0.0) * penalty_count if penalty_count else 0.0
    trajectory.reward = trajectory.outcome_reward - trajectory.read_cost + trajectory.format_reward
    trajectory.auxiliary_reward = 0.0
    if likelihood is not None:
        trajectory.likelihood = likelihood.score(example, trajectory.evidence)
        if trajectory.likelihood["mode"] == "reward":
            if formula in {"judge_cost_v2", "f1_cost_v1"}:
                raise ValueError(f"{formula} supports likelihood off/diagnostic only")
            coefficient = config["likelihood"]["coefficient"]
            trajectory.auxiliary_reward = coefficient * (1 - trajectory.outcome) * trajectory.likelihood["d"]
            trajectory.reward += trajectory.auxiliary_reward
    if not math.isfinite(trajectory.reward):
        raise ValueError("Non-finite composed reward")
    return trajectory


class RewardScorer:
    """Keep semantic judging, optional likelihood and reward composition separate."""
    def __init__(self, config, *, judge=None, likelihood=None, max_ids=4, task_output_policy=None):
        validate_format_penalty_mode(config)
        self.config, self.judge, self.likelihood, self.max_ids = config, judge, likelihood, max_ids
        self.task_output_policy = task_output_policy

    def score(self, example, trajectory):
        return score_trajectory(example, trajectory, self.config, max_ids=self.max_ids,
                                judge=self.judge, likelihood=self.likelihood)

    def provenance(self):
        from .services import TASK_OUTPUT_POLICY
        result = {"formula": self.config.get("formula", "outcome_cost_v1"),
                  "task_output_policy": self.task_output_policy or TASK_OUTPUT_POLICY,
                  "judge": (self.judge.provenance()
                            if self.judge and self.config.get("formula") != "f1_cost_v1" else None),
                  "likelihood": self.likelihood.provenance() if self.likelihood else None}
        # Existing runs compare this mapping exactly when resuming. Only the new
        # opt-in protocol extends it; unchanged legacy recipes retain their hash.
        mode = validate_format_penalty_mode(self.config)
        if mode == "per_decision":
            result.update(format_penalty_mode="per_decision", format_counting_version="reader_decision_v1")
        elif mode == "per_id":
            result.update(format_penalty_mode="per_id", format_counting_version="reader_id_with_protocol_v2")
        if self.config.get("formula") == "f1_cost_v1":
            result.update(answer_f1_protocol=WORD_F1_PROTOCOL,
                          primary_outcome_metric="answer_f1",
                          format_counting_version="hierarchical_reader_id_with_protocol_v3")
            result["judge"] = None
        return result

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


def _navigation_id(decision):
    """Only the executed first parent is navigation, not excess child requests."""
    if (decision.tool_result or {}).get("action_kind") != "navigation":
        return None
    requested = (decision.action or {}).get("ids", [])
    return requested[0] if requested else None


def _decision_violations(decision):
    """Normalize historical tool errors for diagnostics without changing old rewards."""
    result = decision.tool_result or {}
    reasons = set(decision.format_violations)
    # The superseded consecutive rule is not a generic protocol violation in
    # the visible-directory rule. Existing recorded rewards are not modified;
    # new scoring must not reinterpret an old marker as current visibility.
    reasons.discard("consecutive_navigation")
    code = (result.get("error") or {}).get("code")
    mapped = {"invalid_batch_size": "empty_ids", "too_many_ids": "too_many_ids",
              "duplicate_ids_in_batch": "duplicate_ids", "unknown_or_unauthorized_id": "unknown_id",
              "repeated_read": "repeated_read", "visible_navigation": "visible_navigation",
              "max_read_calls": "read_budget_exceeded"}.get(code)
    if mapped:
        reasons.add(mapped)
    if result.get("id_repairs"):
        reasons.add("repaired_id")
    if result.get("invalid_ids"):
        reasons.add("unknown_id")
    if result.get("already_read_ids"):
        reasons.add("repeated_read")
    if result.get("visible_navigation_ids"):
        reasons.add("visible_navigation")
    if result.get("ids_truncated"):
        reasons.add("too_many_ids")
    action = decision.action
    if action is not None and not action["stop"]:
        if not action["ids"]:
            reasons.add("empty_ids")
        if len(set(action["ids"])) != len(action["ids"]):
            reasons.add("duplicate_ids")
    if (result.get("action_kind") == "navigation"
            and not (set(result.get("already_read_ids", [])) - {_navigation_id(decision)})):
        reasons.discard("repeated_read")
    return reasons


def protocol_metrics(rows):
    """Rollout rates and mean decision counts; *_call_rate uses executed reads."""
    n = len(rows) or 1
    reasons = ("invalid_json_or_schema", "empty_ids", "too_many_ids", "duplicate_ids",
               "repaired_id", "unknown_id", "repeated_invalid_action", "repeated_read",
               "read_budget_exceeded")
    counts, rollouts = Counter(), Counter()
    read_calls = repeated_calls = no_new_calls = 0
    navigation_calls = navigation_repeat_calls = visible_navigation_noop_calls = 0
    visible_navigation_penalty_calls = 0
    segment_calls = segment_no_new_calls = 0
    hierarchical = False
    for row in rows:
        seen, row_reasons, navigated = set(), set(), set()
        for decision in row.decisions:
            violations = _decision_violations(decision)
            result = decision.tool_result or {}
            action = decision.action
            requested = result.get("selected_ids", result.get("requested_ids", (action or {}).get("ids", [])))
            is_read = action is not None and not action["stop"]
            navigation = result.get("action_kind") == "navigation"
            segment = result.get("action_kind") == "segment"
            hierarchical |= (navigation or segment
                             or result.get("directory_protocol") == "hierarchical_v1"
                             or "visible_navigation" in violations)
            repeated_ids = set(result.get("already_read_ids", [])) | (set(requested) & seen if is_read else set())
            repeated_ids.discard(_navigation_id(decision))
            repeated = bool(repeated_ids or "repeated_read" in violations)
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
            # Opening a directory returns no raw evidence by design; it is not
            # an empty segment retrieval and must not inflate this failure rate.
            no_new_calls += not navigation and bool(result.get("no_new_evidence", not new_ids))
            navigation_calls += navigation
            # A visible parent may occur in an unexecuted suffix after a child
            # or invalid first ID; the complete charged request is penalized.
            visible_navigation_penalty_calls += is_read and "visible_navigation" in violations
            if navigation and result.get("directory_navigation_success", result.get("ok", False)):
                navigation_repeat_calls += bool(result.get("repeated_navigation", set(requested) & navigated))
                visible_navigation_noop_calls += bool(result.get("visible_navigation_noop", False))
                navigated.update(requested)
            segment_calls += segment
            segment_no_new_calls += segment and bool(result.get("no_new_evidence", not new_ids))
            if not navigation:
                seen.update(new_ids)
        rollouts.update(row_reasons)
    result = {
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
    if hierarchical:
        result.update(navigation_calls=navigation_calls / n,
                      navigation_repeat_calls=navigation_repeat_calls / n,
                      visible_navigation_noop_calls=visible_navigation_noop_calls / n,
                      visible_navigation_penalty_calls=visible_navigation_penalty_calls / n,
                      format_decisions_visible_navigation=counts["visible_navigation"] / n,
                      segment_calls=segment_calls / n,
                      segment_no_new_evidence_calls=segment_no_new_calls / n,
                      segment_no_new_evidence_call_rate=(segment_no_new_calls / segment_calls
                                                        if segment_calls else 0.0))
    return result
