"""Frozen, query-group process returns for the existing DAPO actor loss.

This module does no inference and never changes the terminal reward or the
dynamic-sampling filter. The complete same-query group supplies normalization
statistics, just as it does for the terminal advantages; overlong masking is
applied later by the actor loss. STOP is excluded from process normalization.
"""
from __future__ import annotations

import math


ATTRIBUTION_PROTOCOL = "reader_process_attribution_v1"
ADVANTAGE_PROTOCOL = "reader_process_advantage_v1"
_DERIVED = ("normalized_reward", "process_return", "advantage")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def process_reward_settings(training):
    """Validate optimizer-side settings, leaving scorer-specific keys intact."""
    supplied = training.get("process_reward", {})
    if not isinstance(supplied, dict):
        raise ValueError("training.process_reward must be a mapping")
    enabled = supplied.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("training.process_reward.enabled must be boolean")
    result = {"enabled": enabled, "alpha": supplied.get("alpha", 1.0),
              "gamma": supplied.get("gamma", 1.0),
              "normalization": supplied.get("normalization", "group_standardize"),
              "epsilon": supplied.get("epsilon", training.get("advantage_epsilon", 1e-6))}
    for key in ("alpha", "gamma", "epsilon"):
        if not _finite(result[key]):
            raise ValueError(f"training.process_reward.{key} must be finite")
    if result["alpha"] < 0:
        raise ValueError("training.process_reward.alpha must be nonnegative")
    if not 0 <= result["gamma"] <= 1:
        raise ValueError("training.process_reward.gamma must lie in [0, 1]")
    if result["epsilon"] <= 0:
        raise ValueError("training.process_reward.epsilon must be positive")
    if not isinstance(result["normalization"], str) or result["normalization"] not in {"group_standardize", "rms"}:
        raise ValueError("training.process_reward.normalization must be group_standardize or rms")
    return result


def _entries(row):
    attribution = row.process_attribution
    if not isinstance(attribution, dict) or attribution.get("protocol") != ATTRIBUTION_PROTOCOL:
        raise ValueError("Missing or incompatible frozen process attribution")
    entries = attribution.get("per_decision")
    if not isinstance(entries, list) or len(entries) != len(row.decisions):
        raise ValueError("Process attribution must align with every Reader decision")
    for decision, entry in zip(row.decisions, entries):
        if not isinstance(entry, dict) or type(entry.get("is_read")) is not bool:
            raise ValueError("Invalid process decision/read marker")
        is_read = decision.action is not None and not decision.action["stop"]
        if entry["is_read"] != is_read:
            raise ValueError("Process attribution READ marker disagrees with Reader action")
        if not all(_finite(entry.get(key)) for key in ("g", "w", "reward")) or entry["w"] <= 0:
            raise ValueError("Process scores must be finite with a strictly positive weight")
        expected = entry["w"] * entry["g"] if entry["g"] >= 0 else entry["g"] / entry["w"]
        if not math.isfinite(expected) or not math.isclose(entry["reward"], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Frozen process reward disagrees with its g/w attribution")
        if not is_read and (entry["g"] != 0 or entry["reward"] != 0):
            raise ValueError("STOP/non-READ decisions must have zero process reward")
    return entries


def assign_process_advantages(group, training):
    """Compute once, or validate saved advantages without silently replacing them.

    ``row.advantage`` is the terminal group-relative advantage already assigned
    by the unchanged sampler. The process term is added only to decision-level
    advantages. Both raw attribution and the derived numbers survive resume.
    """
    settings = process_reward_settings(training)
    if not settings["enabled"]:
        return
    if not group or len({(row.sample_id, row.memory_id) for row in group}) != 1:
        raise ValueError("Process advantages require one nonempty same-query group")
    if any(not _finite(row.advantage) for row in group):
        raise ValueError("Process advantages require finite terminal advantages")
    per_row = [_entries(row) for row in group]
    rewards = [entry["reward"] for entries in per_row for entry in entries if entry["is_read"]]
    mean = sum(rewards) / len(rewards) if rewards else 0.0
    if settings["normalization"] == "group_standardize":
        # Avoid treating repeated decimal values as nonconstant through roundoff.
        constant = not rewards or all(value == rewards[0] for value in rewards)
        scale = 0.0 if constant else math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    else:
        constant = False
        scale = math.sqrt(sum(value * value for value in rewards) / len(rewards)) if rewards else 0.0
    if not math.isfinite(mean) or not math.isfinite(scale):
        raise ValueError("Non-finite process reward normalization")
    metadata = {"protocol": ADVANTAGE_PROTOCOL, "settings": settings,
                "read_decisions": len(rewards), "reward_mean": mean, "reward_scale": scale,
                "population": "all_group_read_decisions"}
    planned = []
    for row, entries in zip(group, per_row):
        output = []
        for entry in entries:
            if not entry["is_read"] or constant:
                normalized = 0.0
            else:
                numerator = entry["reward"] - mean if settings["normalization"] == "group_standardize" else entry["reward"]
                normalized = numerator / (scale + settings["epsilon"])
            output.append({"normalized_reward": normalized})
        future = 0.0
        for entry, derived in reversed(list(zip(entries, output))):
            future = derived["normalized_reward"] + settings["gamma"] * future
            # STOP is terminal. A malformed terminal decision also receives
            # only the terminal reward, never a fabricated process gain.
            process_return = future if entry["is_read"] else 0.0
            derived.update(process_return=process_return,
                           advantage=row.advantage + settings["alpha"] * process_return)
            if not all(_finite(value) for value in derived.values()):
                raise ValueError("Non-finite process return/advantage")
        planned.append(output)
    saved = any("advantage_normalization" in row.process_attribution for row in group) or any(
        any(key in entry for key in _DERIVED) for entries in per_row for entry in entries)
    if saved:
        for row, entries, output in zip(group, per_row, planned):
            if row.process_attribution.get("advantage_normalization") != metadata:
                raise ValueError("Saved process advantage normalization differs from this group/config")
            for entry, derived in zip(entries, output):
                if any(not _finite(entry.get(key)) or entry[key] != value for key, value in derived.items()):
                    raise ValueError("Saved process advantages differ from frozen attribution/terminal advantages")
        return
    for row, entries, output in zip(group, per_row, planned):
        row.process_attribution["advantage_normalization"] = dict(metadata)
        for entry, derived in zip(entries, output):
            entry.update(derived)


def decision_advantage(row, index, *, enabled):
    """Return the exact scalar shared by all output tokens of one decision."""
    if not enabled:
        return row.advantage
    value = row.process_attribution["per_decision"][index]["advantage"]
    if not _finite(value):
        raise ValueError("Missing or non-finite frozen decision advantage")
    return value
