"""Dynamic whole-group sampling for DAPO; no per-rollout reward selection.

Following the DAPO recipe, collect generation batches until a fixed number of
eligible groups is available. judge_cost_v2 accepts only Judge-mixed groups;
cost or format differences alone never qualify. A non-positive max_num_gen_batches disables
the per-update budget; the finite dataset/epoch cursor still bounds sampling.
"""
from __future__ import annotations

from collections import Counter
import math
import random

from .core import digest
from .dapo import trajectory_truncated, valid_training_tokens
from .rewards import protocol_metrics


def _validate_judge_cost_protocol(config):
    if config["reward"].get("formula") != "judge_cost_v2":
        return
    training = config["training"]
    if training.get("filter_metric", "reward") != "judge":
        raise ValueError("judge_cost_v2 requires filter_metric=judge (mixed-correctness groups only)")
    if training.get("advantage_std_floor", 0.0) != 0:
        raise ValueError("judge_cost_v2 requires advantage_std_floor=0")
    if config["reward"].get("likelihood", {}).get("mode") == "reward":
        raise ValueError("judge_cost_v2 supports likelihood off/diagnostic only")


def validate_dapo_config(config):
    training = config["training"]
    for key, default, minimum in (("group_size", 8, 2), ("candidate_groups", 32, 1),
                                  ("target_groups", 16, 1), ("epochs", 1, 1)):
        value = training.get(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"training.{key} must be an integer >= {minimum}")
    if type(training.get("max_num_gen_batches", 4)) is not int:
        raise ValueError("max_num_gen_batches must be an integer")
    for key, default in (("mini_batch_groups", 1), ("micro_batch_rollouts", 1), ("max_micro_batch_tokens", 16384)):
        if type(training.get(key, default)) is not int or training.get(key, default) < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    mini = training.get("mini_batch_groups", 1)
    if training.get("target_groups", 16) % mini:
        raise ValueError("target_groups must be divisible by mini_batch_groups")
    if training.get("micro_batch_rollouts", 1) > mini * training.get("group_size", 8):
        raise ValueError("micro_batch_rollouts cannot exceed the rollouts in one mini batch")
    if type(training.get("ppo_epochs", 1)) is not int or training.get("ppo_epochs", 1) != 1:
        raise ValueError("This DAPO execution engine requires ppo_epochs=1")
    if training.get("kl_coefficient", 0) != 0:
        raise ValueError("DAPO requires kl_coefficient=0")
    low, high = training.get("clip_ratio_low", 0.2), training.get("clip_ratio_high", 0.28)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (low, high))
            or not 0 < low < 1 or high < low):
        raise ValueError("DAPO requires 0 < clip_ratio_low < 1 and clip_ratio_high >= clip_ratio_low")
    if type(training.get("entropy_chunk_size", 32)) is not int or training.get("entropy_chunk_size", 32) < 1:
        raise ValueError("entropy_chunk_size must be a positive integer")
    if type(training.get("overlong_filter", True)) is not bool:
        raise ValueError("overlong_filter must be boolean")
    if config["reward"].get("formula") not in {"judge_gated_v1", "judge_cost_v2"}:
        raise ValueError("This DAPO experiment requires judge_gated_v1 or judge_cost_v2")
    if training.get("filter_metric", "reward") not in {"reward", "judge"}:
        raise ValueError("filter_metric must be reward or judge")
    for key, default in (("filter_min_std", 1e-6), ("advantage_std_floor", 0.0), ("advantage_epsilon", 1e-6)):
        value = training.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if training.get("advantage_epsilon", 1e-6) <= 0:
        raise ValueError("advantage_epsilon must be positive")
    _validate_judge_cost_protocol(config)
    if config["reader"]["rollout_temperature"] <= 0:
        raise ValueError("DAPO requires stochastic Reader sampling")
    likelihood = config["reward"].get("likelihood", {})
    if likelihood.get("mode", "off") not in {"off", "diagnostic", "reward"}:
        raise ValueError("likelihood.mode must be off, diagnostic or reward")
    if likelihood.get("mode") == "reward":
        for key in ("coefficient", "advantage_std_floor"):
            value = likelihood.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Likelihood reward requires positive {key}")


def population_std(values):
    if not values or any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("Group scores must be nonempty and finite")
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def assess_group(group, config):
    """Log every candidate, calculate fixed advantages, then filter whole groups."""
    _validate_judge_cost_protocol(config)
    training = config["training"]
    rewards = [row.reward for row in group]
    judges = [row.judge_score for row in group]
    reward_std, judge_std = population_std(rewards), population_std(judges)
    if any(score not in (0, 1) for score in judges):
        raise ValueError("Judge must return a binary score for every rollout")
    floor = training.get("advantage_std_floor", 0.0)
    likelihood = config["reward"].get("likelihood", {})
    if likelihood.get("mode") == "reward":
        floor = max(floor, likelihood["advantage_std_floor"])
    mean = sum(rewards) / len(rewards)
    denominator = max(reward_std, floor) + training.get("advantage_epsilon", 1e-6)
    for row in group:
        row.advantage = (row.reward - mean) / denominator
    threshold = training.get("filter_min_std", 1e-6)
    metric_std = judge_std if training.get("filter_metric", "reward") == "judge" else reward_std
    overlong = training.get("overlong_filter", True)
    unmasked = [row for row in group if not (overlong and trajectory_truncated(row))]
    usable = valid_training_tokens(group, overlong) > 0 and any(row.advantage != 0 for row in unmasked)
    # The complete group can be mixed while its only successful (or failing)
    # trajectory is masked. Such a group would train only one Judge outcome.
    trainable_judges = {row.judge_score for row in unmasked if valid_training_tokens([row], False) > 0}
    unmasked_judge_signal = trainable_judges == {0, 1}
    accepted = metric_std > threshold and usable
    if config["reward"].get("formula") == "judge_cost_v2":
        accepted = accepted and unmasked_judge_signal
    base = [row.reward - row.auxiliary_reward for row in group]
    return {**protocol_metrics(group), "accepted": accepted, "reward_std": reward_std, "judge_std": judge_std,
            "judge_accuracy": sum(judges) / len(group), "reward": mean,
            "read_calls": sum(row.n_read_calls for row in group) / len(group),
            "format_failure_rate": sum(row.format_failure for row in group) / len(group),
            "judge_all_correct": all(score == 1 for score in judges),
            "judge_all_wrong": all(score == 0 for score in judges),
            "judge_mixed": judge_std > 0,
            "zero_reward_variance": reward_std <= threshold,
            "cost_format_only_signal": judge_std == 0 and population_std(base) > threshold
                                       and population_std([row.auxiliary_reward for row in group]) == 0,
            "auxiliary_only_signal": population_std(base) <= threshold and reward_std > threshold,
            "no_unmasked_signal": not usable,
            "no_unmasked_judge_signal": not unmasked_judge_signal,
            "unique_evidence_fraction": len({digest(row.evidence) for row in group}) / len(group),
            "unique_prediction_fraction": len({row.prediction for row in group}) / len(group),
            "truncated_rollout_fraction": sum(trajectory_truncated(row) for row in group) / len(group),
            "task_truncated_rollout_fraction": sum(bool(row.task_completion.get("truncated"))
                                                    for row in group) / len(group)}


class DynamicGroupSampler:
    def __init__(self, config, examples, *, epoch=0, next_index=0, candidate_index=0, batch_index=0):
        validate_dapo_config(config)
        if not examples:
            raise ValueError("No DAPO training examples")
        self.config, self.examples = config, examples
        self.epoch, self.next_index = epoch, next_index
        self.candidate_index, self.batch_index = candidate_index, batch_index
        self._ordered_epoch, self._ordered = None, None
        if not (0 <= epoch <= config["training"]["epochs"] and 0 <= next_index <= len(examples)):
            raise ValueError("Invalid DAPO dataset cursor")

    def state_dict(self):
        return {"epoch": self.epoch, "next_index": self.next_index,
                "candidate_index": self.candidate_index, "batch_index": self.batch_index}

    def _next(self):
        while self.epoch < self.config["training"]["epochs"]:
            if self.next_index >= len(self.examples):
                self.epoch += 1
                self.next_index = 0
                continue
            if self._ordered_epoch != self.epoch:
                self._ordered = list(self.examples)
                random.Random(self.config["seed"] + self.epoch).shuffle(self._ordered)
                self._ordered_epoch = self.epoch
            example = self._ordered[self.next_index]
            self.next_index += 1
            return example
        return None

    def collect(self, collect_group, behavior_version, *, record_candidate=None, record_round=None):
        training = self.config["training"]
        target, width = training.get("target_groups", 16), training.get("candidate_groups", 32)
        budget = training.get("max_num_gen_batches", 4)
        groups, stats, source_counts = [], [], Counter()
        batch_id = f"batch-{self.batch_index:08d}"
        exhausted, rounds = False, 0
        while len(groups) < target:
            rounds += 1
            for _ in range(width):
                example = self._next()
                if example is None:
                    exhausted = True
                    break
                candidate_id = self.candidate_index
                group = collect_group(example, candidate_id)
                self.candidate_index += 1
                if len(group) != training["group_size"] or any(row.policy_version != behavior_version for row in group):
                    raise ValueError("DAPO sampling requires complete groups from an unchanged behavior policy")
                if any(row.sample_id != example.sample_id or row.memory_id != example.memory_id for row in group):
                    raise ValueError("DAPO group does not belong to the sampled question/snapshot")
                result = assess_group(group, self.config)
                result.update({"sample_id": example.sample_id, "candidate_id": candidate_id, "batch_id": batch_id,
                               "behavior_policy_version": behavior_version,
                               "selected": result["accepted"] and len(groups) < target})
                stats.append(result)
                source_counts[example.source] += 1
                if record_candidate:
                    record_candidate(group, result)
                if result["selected"]:
                    groups.append(group)
            count = len(stats)
            report = {"batch_id": batch_id, "generation_batches": rounds, "candidate_groups": count,
                      "selected_groups": len(groups), "accepted_groups": sum(s["accepted"] for s in stats),
                      "source_candidates": dict(source_counts), "dataset_exhausted": exhausted}
            keys = ("judge_all_correct", "judge_all_wrong", "judge_mixed", "zero_reward_variance",
                    "cost_format_only_signal", "auxiliary_only_signal", "no_unmasked_signal",
                    "no_unmasked_judge_signal", "accepted")
            report.update({key + "_rate": sum(s[key] for s in stats) / count if count else 0.0 for key in keys})
            for key in ("unique_evidence_fraction", "unique_prediction_fraction", "truncated_rollout_fraction",
                        "task_truncated_rollout_fraction", "judge_accuracy", "reward", "read_calls",
                        "format_failure_rate", "repeated_read_rate", "repeated_read_calls", "no_new_evidence_calls",
                        "unknown_id_rate", "id_repair_rate", "empty_evidence_rate"):
                report[key] = sum(s[key] for s in stats) / count if count else 0.0
            # Group sizes are fixed, so mean counts retain the rollout weighting.
            # Ratios use total calls rather than averaging per-group call rates.
            calls = sum(s["read_calls"] for s in stats)
            for numerator, key in (("repeated_read_calls", "repeated_read_call_rate"),
                                   ("no_new_evidence_calls", "no_new_evidence_call_rate")):
                report[key] = sum(s[numerator] for s in stats) / calls if calls else 0.0
            if record_round:
                record_round(report)
            if len(groups) >= target:
                self.batch_index += 1
                return groups, report
            if exhausted:
                report["dropped_tail_groups"] = len(groups)
                return [], report
            if budget > 0 and rounds >= budget:
                raise RuntimeError(f"DAPO dynamic sampling exhausted {budget} generation batches: "
                                   f"{len(groups)}/{target} usable groups; inspect candidate_metrics.jsonl. "
                                   "max_num_gen_batches<=0 disables this budget.")
        raise AssertionError("Unreachable DAPO sampling state")
