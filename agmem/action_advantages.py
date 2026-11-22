"""Frozen action supervision for redundant directory/evidence retrieval.

Terminal rewards, group advantages and sampling eligibility stay unchanged.
Only the offending READ's output tokens receive the negative rule advantage.
"""
from __future__ import annotations

import math

from .core import digest
from .process_advantages import decision_advantage

ACTION_ADVANTAGE_PROTOCOL = "reader_visible_navigation_negative_advantage_v1"
REPEATED_RETRIEVAL_PROTOCOL = "reader_repeated_retrieval_negative_advantage_v2"


def action_advantage_protocol(settings):
    return (REPEATED_RETRIEVAL_PROTOCOL if settings.get("include_repeated_segments", False)
            else ACTION_ADVANTAGE_PROTOCOL)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def action_advantage_settings(training):
    supplied = training.get("visible_navigation_negative_advantage", {})
    if not isinstance(supplied, dict):
        raise ValueError("training.visible_navigation_negative_advantage must be a mapping")
    if set(supplied) - {"enabled", "magnitude", "include_repeated_segments"}:
        raise ValueError("Unknown visible_navigation_negative_advantage setting")
    enabled, magnitude = supplied.get("enabled", False), supplied.get("magnitude", 1.0)
    if type(enabled) is not bool:
        raise ValueError("visible_navigation_negative_advantage.enabled must be boolean")
    if not _finite(magnitude) or magnitude <= 0:
        raise ValueError("visible_navigation_negative_advantage.magnitude must be finite and positive")
    segments = supplied.get("include_repeated_segments", False)
    if type(segments) is not bool:
        raise ValueError("visible_navigation_negative_advantage.include_repeated_segments must be boolean")
    if segments and not enabled:
        raise ValueError("Repeated segment supervision requires enabled action supervision")
    if enabled and training.get("process_reward", {}).get("enabled", False):
        raise ValueError("Navigation action supervision cannot be combined with process_reward")
    # Preserve historical v1 settings/provenance exactly when the new rule is off.
    return {"enabled": enabled, "magnitude": float(magnitude),
            **({"include_repeated_segments": True} if segments else {})}


def validate_action_advantage_config(config):
    settings = action_advantage_settings(config["training"])
    if config.get("reward", {}).get("formula") == "full_directory_orm_v1" and settings["enabled"]:
        raise ValueError("Full-directory ORM uses trajectory reward only; forced action advantages must be disabled")
    if settings["enabled"]:
        if (config["training"].get("algorithm", "grpo") != "dapo"
                or config.get("data", {}).get("directory_protocol") != "hierarchical_v1"
                or config["reward"].get("formula") != "judge_f1_cost_v1"):
            raise ValueError("Navigation action supervision requires hierarchical judge_f1 DAPO")
    return settings


def action_source_signature(row, *, include_repeated_segments=None):
    if include_repeated_segments is None:
        include_repeated_segments = row.action_advantages.get("settings", {}).get("include_repeated_segments", False)
    return digest([{"action": step.action,
                    "directory_protocol": (step.tool_result or {}).get("directory_protocol"),
                    "visible_navigation_ids": (step.tool_result or {}).get("visible_navigation_ids", []),
                    "format_violations": step.format_violations,
                    **({"already_read_ids": (step.tool_result or {}).get("already_read_ids", []),
                        "new_ids": (step.tool_result or {}).get("new_ids", []),
                        "returned_ids": [record["id"] for record in (step.tool_result or {}).get("records", [])],
                        "read_call_consumed": (step.tool_result or {}).get("read_call_consumed")}
                       if include_repeated_segments else {})} for step in row.decisions])


def _visible_navigation(step):
    result = step.tool_result or {}
    if result.get("directory_protocol") != "hierarchical_v1":
        raise ValueError("Navigation action supervision requires hierarchical decision metadata")
    action = step.action
    if action is None or action["stop"]:
        return False
    visible = result.get("visible_navigation_ids", [])
    if not isinstance(visible, list) or any(not isinstance(key, str) for key in visible):
        raise ValueError("Invalid frozen visible_navigation_ids")
    repeated = bool(set(action["ids"]) & set(visible))
    if "visible_navigation" in step.format_violations and not repeated:
        raise ValueError("Visible navigation violation lacks its pre-action ID metadata")
    return repeated


def _repeated_segment(step):
    """Use the environment's pre-action evidence, never final trajectory evidence.

The first successful read has no already_read_ids. Failed attempts are not
retained and may be retried; evidence survives parent-directory eviction.
"""
    action = step.action
    if action is None or action["stop"]:
        return False
    already = (step.tool_result or {}).get("already_read_ids", [])
    if (not isinstance(already, list)
            or any(not isinstance(key, str) or ":c" not in key for key in already)):
        raise ValueError("Invalid frozen already_read_ids for hierarchical segments")
    repeated = bool(set(action["ids"]) & set(already))
    if "repeated_read" in step.format_violations and not repeated:
        raise ValueError("Repeated segment violation lacks its pre-action already_read_ids")
    return repeated


def assign_action_advantages(group, training, *, require_saved=False):
    """Freeze per-decision advantages or validate the saved assignment.

    All decisions are recorded before the actor's unchanged overlong masking.
    No second centering/normalization is applied; row.advantage is never replaced.
    """
    settings = action_advantage_settings(training)
    saved = [bool(row.action_advantages) for row in group]
    if not settings["enabled"]:
        if any(saved):
            raise ValueError("Saved action advantages require enabled navigation supervision")
        return
    if not group or len({(row.sample_id, row.memory_id) for row in group}) != 1:
        raise ValueError("Action advantages require one nonempty same-query group")
    if require_saved and not all(saved):
        raise ValueError("Resumed action-supervised batches require saved action advantages")
    if any(saved) and not all(saved):
        raise ValueError("Partially saved action advantages cannot be regenerated")
    planned = []
    include_segments = settings.get("include_repeated_segments", False)
    for row in group:
        if not _finite(row.advantage) or not row.decisions:
            raise ValueError("Action advantages require finite terminal advantages and decisions")
        entries = []
        for index, step in enumerate(row.decisions):
            navigation = _visible_navigation(step)
            segment = _repeated_segment(step) if include_segments else False
            value = min(row.advantage, -settings["magnitude"]) if navigation or segment else row.advantage
            entries.append({"index": index, "visible_navigation": navigation, "advantage": value,
                            **({"repeated_segment": segment} if include_segments else {})})
        planned.append({"protocol": action_advantage_protocol(settings), "settings": dict(settings),
                        "terminal_advantage": row.advantage,
                        "decision_source_sha256": action_source_signature(row, include_repeated_segments=include_segments),
                        "per_decision": entries})
    if any(saved):
        if any(row.action_advantages != expected for row, expected in zip(group, planned)):
            raise ValueError("Saved action advantages differ from frozen decisions/group/config")
        return
    for row, expected in zip(group, planned):
        row.action_advantages = expected


def effective_decision_advantage(row, index, *, process_enabled=False):
    if not row.action_advantages:
        return decision_advantage(row, index, enabled=process_enabled)
    saved = row.action_advantages
    if (process_enabled or saved.get("protocol") not in {ACTION_ADVANTAGE_PROTOCOL, REPEATED_RETRIEVAL_PROTOCOL}
            or saved.get("protocol") != action_advantage_protocol(saved.get("settings", {}))):
        raise ValueError("Incompatible frozen action advantage protocol")
    entries = saved.get("per_decision", [])
    if len(entries) != len(row.decisions):
        raise ValueError("Saved action advantages must align with every decision")
    value = entries[index].get("advantage")
    if not _finite(value):
        raise ValueError("Missing or non-finite frozen action advantage")
    return value


def action_advantage_metrics(rows):
    """Report exactly the non-truncated rows receiving actor gradients."""
    if not rows or not any(row.action_advantages for row in rows):
        return {}
    repeated = [(row, entry) for row in rows for entry in row.action_advantages["per_decision"]
                if entry["visible_navigation"]]
    count = len(repeated)
    positives = sum(entry["advantage"] > 0 for _, entry in repeated)
    by_query = {}
    for row in rows:
        by_query.setdefault((row.sample_id, row.memory_id), []).append(row)
    sole_correct = [correct[0] for group in by_query.values()
                    if len(correct := [row for row in group if row.judge_score == 1]) == 1]
    metrics = {
        "navigation_rule_decisions": count,
        "navigation_rule_overridden_decisions": sum(entry["advantage"] != row.advantage
                                                     for row, entry in repeated),
        "navigation_rule_positive_before": sum(row.advantage > 0 for row, _ in repeated),
        "navigation_rule_positive_after": positives,
        "navigation_rule_positive_advantage_rate": positives / (count or 1),
        "navigation_rule_advantage_mean": sum(entry["advantage"] for _, entry in repeated) / (count or 1),
        "navigation_rule_correct_decisions": sum(row.judge_score == 1 for row, _ in repeated),
        "sole_correct_groups": len(sole_correct),
        "sole_correct_nonpositive_group_advantage": sum(row.advantage <= 0 for row in sole_correct),
        "sole_correct_positive_valid_read_decisions": sum(
            entry["advantage"] > 0 and not entry["visible_navigation"]
            and step.action is not None and not step.action["stop"] and not step.format_violations
            for row in sole_correct for step, entry in zip(row.decisions, row.action_advantages["per_decision"])),
    }
    if any(row.action_advantages.get("settings", {}).get("include_repeated_segments", False) for row in rows):
        segments = [(row, entry) for row in rows for entry in row.action_advantages["per_decision"]
                    if entry.get("repeated_segment", False)]
        count = len(segments)
        positives = sum(entry["advantage"] > 0 for _, entry in segments)
        metrics.update({
            "segment_rule_decisions": count,
            "segment_rule_overridden_decisions": sum(entry["advantage"] != row.advantage
                                                      for row, entry in segments),
            "segment_rule_positive_before": sum(row.advantage > 0 for row, _ in segments),
            "segment_rule_positive_after": positives,
            "segment_rule_positive_advantage_rate": positives / (count or 1),
            "segment_rule_advantage_mean": sum(entry["advantage"] for _, entry in segments) / (count or 1),
            "segment_rule_correct_decisions": sum(row.judge_score == 1 for row, _ in segments),
        })
    return metrics
