"""Dynamic whole-group sampling for DAPO; no per-rollout reward selection.

Following the DAPO recipe, collect generation batches until a fixed number of
eligible groups is available. The filter metric selects reward differences,
Judge-mixed groups, all-correct format contrasts, or any group with a trainable
correct trajectory when process rewards are enabled.
A non-positive max_num_gen_batches disables
the per-update budget; the finite dataset/epoch cursor still bounds sampling.
"""
from __future__ import annotations

from collections import Counter
import math
import random

from .core import digest
from .dapo import trajectory_truncated, valid_training_tokens
from .rewards import protocol_metrics, validate_format_penalty_mode


def _validate_judge_cost_protocol(config):
    formula = config["reward"].get("formula")
    if formula not in {"judge_cost_v2", "f1_cost_v1"}:
        return
    training = config["training"]
    if training.get("advantage_std_floor", 0.0) != 0:
        raise ValueError(f"{formula} requires advantage_std_floor=0")
    if config["reward"].get("likelihood", {}).get("mode") == "reward":
        raise ValueError(f"{formula} supports likelihood off/diagnostic only")


def _validate_process_filter(training):
    if training.get("filter_metric") == "exclude_all_wrong":
        from .process_advantages import process_reward_settings
        if not process_reward_settings(training)["enabled"]:
            raise ValueError("exclude_all_wrong requires training.process_reward.enabled=true")


def validate_dapo_config(config):
    from .process_rewards import attribution_settings
    if (config["reward"].get("formula") == "f1_cost_v1"
            and config["training"].get("process_reward", {}).get("enabled", False)):
        raise ValueError("f1_cost_v1 is an outcome-only protocol; process_reward must be disabled")
    attribution_settings(config)
    validate_format_penalty_mode(config["reward"])
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
    formula = config["reward"].get("formula")
    if formula not in {"judge_gated_v1", "judge_cost_v2", "f1_cost_v1"}:
        raise ValueError("This DAPO experiment requires judge_gated_v1, judge_cost_v2 or f1_cost_v1")
    if training.get("filter_metric", "reward") not in {
            "reward", "judge", "judge_or_all_correct_format", "exclude_all_wrong", "nonzero_advantage"}:
        raise ValueError("Unknown DAPO filter_metric")
    if formula == "f1_cost_v1" and training.get("filter_metric") != "nonzero_advantage":
        raise ValueError("f1_cost_v1 requires filter_metric=nonzero_advantage")
    if training.get("filter_metric") == "nonzero_advantage" and formula != "f1_cost_v1":
        raise ValueError("nonzero_advantage requires f1_cost_v1")
    _validate_process_filter(training)
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
    # Exact-zero filtering must not accept constant decimal rewards because of
    # rounding in sum(values) / len(values), e.g. eight copies of -1.1.
    if all(value == values[0] for value in values):
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def assess_group(group, config):
    """Log every candidate, calculate fixed advantages, then filter whole groups."""
    _validate_judge_cost_protocol(config)
    training = config["training"]
    _validate_process_filter(training)
    rewards = [row.reward for row in group]
    f1_protocol = config["reward"].get("formula") == "f1_cost_v1"
    judges = [row.judge_score for row in group]
    reward_std = population_std(rewards)
    judge_std = None if f1_protocol else population_std(judges)
    if not f1_protocol and any(score not in (0, 1) for score in judges):
        raise ValueError("Judge must return a binary score for every rollout")
    floor = training.get("advantage_std_floor", 0.0)
    likelihood = config["reward"].get("likelihood", {})
    if likelihood.get("mode") == "reward":
        floor = max(floor, likelihood["advantage_std_floor"])
    mean = sum(rewards) / len(rewards)
    denominator = max(reward_std, floor) + training.get("advantage_epsilon", 1e-6)
    for row in group:
        row.advantage = (row.reward - mean) / denominator if reward_std else 0.0
    threshold = training.get("filter_min_std", 1e-6)
    filter_metric = training.get("filter_metric", "reward")
    metric_std = judge_std if filter_metric == "judge" else reward_std
    overlong = training.get("overlong_filter", True)
    unmasked = [row for row in group if not (overlong and trajectory_truncated(row))]
    trainable = [row for row in unmasked if valid_training_tokens([row], False) > 0]
    usable = any(row.advantage != 0 for row in trainable)
    if f1_protocol:
        if filter_metric != "nonzero_advantage":
            raise ValueError("f1_cost_v1 requires filter_metric=nonzero_advantage")
        f1s = [row.answer_f1 for row in group]
        f1_std = population_std(f1s)
        if any(isinstance(value, bool) or not 0 <= value <= 1 for value in f1s):
            raise ValueError("f1_cost_v1 requires an answer_f1 in [0, 1] for every rollout")
        # All-zero F1 groups have no outcome-quality signal: cost/format-only
        # advantages can reinforce stopping without evidence. Reject the whole
        # group, while retaining the existing trainable/nonzero-advantage guard.
        # Other F1 ties may still learn from cost/format differences.
        f1_all_zero = all(value == 0 for value in f1s)
        accepted = usable and not f1_all_zero
        return {**protocol_metrics(group), "accepted": accepted,
                "reward_std": reward_std, "f1_std": f1_std,
                "answer_f1": sum(f1s) / len(group), "outcome_f1": sum(f1s) / len(group),
                "f1_all_zero": f1_all_zero,
                "filtered_f1_all_zero": f1_all_zero,
                "f1_all_one": all(value == 1 for value in f1s),
                "reward": mean, "read_calls": sum(row.n_read_calls for row in group) / len(group),
                "format_reward": sum(row.format_reward for row in group) / len(group),
                "format_violation_count": sum(row.format_violation_count for row in group) / len(group),
                "format_failure_rate": sum(row.format_failure for row in group) / len(group),
                "zero_reward_variance": reward_std == 0,
                "cost_format_only_signal": f1_std == 0 and reward_std > 0,
                "no_unmasked_signal": not usable,
                "accepted_zero_terminal_advantage": False,
                "unique_evidence_fraction": len({digest(row.evidence) for row in group}) / len(group),
                "unique_prediction_fraction": len({row.prediction for row in group}) / len(group),
                "truncated_rollout_fraction": sum(trajectory_truncated(row) for row in group) / len(group),
                "task_truncated_rollout_fraction": sum(bool(row.task_completion.get("truncated"))
                                                        for row in group) / len(group)}
    # The complete group can be mixed while its only successful (or failing)
    # trajectory is masked. Such a group would train only one Judge outcome.
    trainable_judges = {row.judge_score for row in trainable}
    unmasked_judge_signal = trainable_judges == {0, 1}
    # Require a clean and a penalized *trainable* trajectory. Cost differences
    # alone, or format differences existing only in masked trajectories, do not
    # qualify. Judge correctness is checked on the entire original group so
    # masking its only wrong trajectory cannot turn a mixed group into this case.
    trainable_formats = [row.format_reward for row in trainable]
    all_correct_format_signal = (
        all(score == 1 for score in judges)
        and any(value == 0 for value in trainable_formats)
        and any(value < 0 for value in trainable_formats)
        and population_std(trainable_formats) > threshold
    )
    accepted = metric_std > threshold and usable
    if (config["reward"].get("formula") == "judge_cost_v2"
            and filter_metric == "judge"):
        accepted = accepted and unmasked_judge_signal
    if filter_metric == "judge_or_all_correct_format":
        accepted = usable and ((judge_std > threshold and unmasked_judge_signal)
                               or all_correct_format_signal)
    if filter_metric == "exclude_all_wrong":
        # Process rewards are calculated after selection and can supply a signal
        # even when every terminal advantage is zero. Retain token/overlong
        # safeguards: at least one successful trajectory must actually train.
        accepted = 1 in trainable_judges
    base = [row.reward - row.auxiliary_reward for row in group]
    return {**protocol_metrics(group), "accepted": accepted, "reward_std": reward_std, "judge_std": judge_std,
            "judge_accuracy": sum(judges) / len(group), "reward": mean,
            "read_calls": sum(row.n_read_calls for row in group) / len(group),
            "format_reward": sum(row.format_reward for row in group) / len(group),
            "format_violation_count": sum(row.format_violation_count for row in group) / len(group),
            "format_failure_rate": sum(row.format_failure for row in group) / len(group),
            "judge_all_correct": all(score == 1 for score in judges),
            "judge_all_wrong": all(score == 0 for score in judges),
            "judge_mixed": judge_std > 0,
            "all_correct_format_signal": all_correct_format_signal,
            "accepted_all_correct_format": accepted and all_correct_format_signal,
            "accepted_all_correct": accepted and all(score == 1 for score in judges),
            "accepted_zero_terminal_advantage": accepted and all(row.advantage == 0 for row in group),
            "no_trainable_correct_signal": 1 not in trainable_judges,
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

    def collect(self, collect_group, behavior_version, *, record_candidate=None, record_round=None,
                collect_candidates=None):
        """Collect a frozen-policy batch, draining already admitted candidates.

        An optional concurrent collector consumes a lazy iterator of
        ``(example, candidate_id)`` and yields ``(example, candidate_id, group)``
        in that same order. Closing its result iterator must finish or cancel
        all outstanding work before returning. This lets the execution layer
        bound its own in-flight groups without eagerly advancing this cursor.
        Once the target is reached, the source admits no further candidates.
        The fixed-window collector finishes and audits its current window before
        exhausting the source; no Reader or scoring work overlaps an update.
        """
        training = self.config["training"]
        target, width = training.get("target_groups", 16), training.get("candidate_groups", 32)
        budget = training.get("max_num_gen_batches", 4)
        groups, stats, source_counts = [], [], Counter()
        batch_id = f"batch-{self.batch_index:08d}"
        exhausted, rounds = False, 0
        while len(groups) < target:
            rounds += 1
            round_start, reserved, source_consumed = self.candidate_index, {}, False

            def candidates():
                nonlocal exhausted, source_consumed
                for offset in range(width):
                    # Check before advancing the dataset cursor. Do not break
                    # result iteration: other groups in the current window may
                    # still be running and must be scored and recorded.
                    if len(groups) >= target:
                        source_consumed = True
                        return
                    example = self._next()
                    if example is None:
                        exhausted = source_consumed = True
                        return
                    candidate_id = round_start + offset
                    reserved[candidate_id] = example
                    yield example, candidate_id
                source_consumed = True

            candidate_source = candidates()

            def serial_results():
                for example, candidate_id in candidate_source:
                    yield example, candidate_id, collect_group(example, candidate_id)

            results = None
            try:
                results = iter(collect_candidates(candidate_source) if collect_candidates is not None else serial_results())
                for example, candidate_id, group in results:
                    if (type(candidate_id) is not int or candidate_id != self.candidate_index
                            or candidate_id not in reserved):
                        raise ValueError("DAPO candidate results must follow the reserved candidate order exactly")
                    if example != reserved[candidate_id]:
                        raise ValueError("DAPO candidate result does not match its reserved example")
                    if len(group) != training["group_size"] or any(row.policy_version != behavior_version for row in group):
                        raise ValueError("DAPO sampling requires complete groups from an unchanged behavior policy")
                    if any(row.sample_id != example.sample_id or row.memory_id != example.memory_id for row in group):
                        raise ValueError("DAPO group does not belong to the sampled question/snapshot")
                    result = assess_group(group, self.config)
                    self.candidate_index += 1
                    del reserved[candidate_id]
                    result.update({"sample_id": example.sample_id, "candidate_id": candidate_id, "batch_id": batch_id,
                                   "behavior_policy_version": behavior_version,
                                   "selected": result["accepted"] and len(groups) < target})
                    stats.append(result)
                    source_counts[example.source] += 1
                    if record_candidate:
                        record_candidate(group, result)
                    if result["selected"]:
                        groups.append(group)
                if not source_consumed or reserved:
                    raise ValueError("DAPO candidate collector must exhaust the round and return every reserved group")
            finally:
                try:
                    close = getattr(results, "close", None)
                    if close is not None:
                        close()
                finally:
                    candidate_source.close()
            count = len(stats)
            report = {"batch_id": batch_id, "generation_batches": rounds, "candidate_groups": count,
                      "selected_groups": len(groups), "accepted_groups": sum(s["accepted"] for s in stats),
                      "source_candidates": dict(source_counts), "dataset_exhausted": exhausted}
            keys = ("judge_all_correct", "judge_all_wrong", "judge_mixed", "zero_reward_variance",
                    "cost_format_only_signal", "auxiliary_only_signal", "no_unmasked_signal",
                    "no_unmasked_judge_signal", "all_correct_format_signal",
                    "accepted_all_correct_format", "accepted_all_correct",
                    "accepted_zero_terminal_advantage", "no_trainable_correct_signal", "accepted",
                    "f1_all_zero", "f1_all_one", "filtered_f1_all_zero")
            report.update({key + "_rate": sum(s[key] for s in stats) / count
                           for key in keys if count and key in stats[0]})
            for key in ("unique_evidence_fraction", "unique_prediction_fraction", "truncated_rollout_fraction",
                        "task_truncated_rollout_fraction", "judge_accuracy", "reward", "read_calls",
                        "format_reward", "format_violation_count", "format_failure_rate",
                        "repeated_read_rate", "repeated_read_calls", "no_new_evidence_calls",
                        "unknown_id_rate", "id_repair_rate", "empty_evidence_rate",
                        "answer_f1", "outcome_f1", "navigation_calls", "segment_calls",
                        "segment_no_new_evidence_calls", "navigation_repeat_calls",
                        "visible_navigation_noop_calls", "visible_navigation_penalty_calls",
                        "format_decisions_visible_navigation"):
                if any(key in s for s in stats):
                    report[key] = sum(s.get(key, 0.0) for s in stats) / count
            # Group sizes are fixed, so mean counts retain the rollout weighting.
            # Ratios use total calls rather than averaging per-group call rates.
            calls = sum(s["read_calls"] for s in stats)
            for numerator, key in (("repeated_read_calls", "repeated_read_call_rate"),
                                   ("no_new_evidence_calls", "no_new_evidence_call_rate")):
                report[key] = sum(s[numerator] for s in stats) / calls if calls else 0.0
            if "segment_calls" in report:
                segment_calls = sum(s.get("segment_calls", 0.0) for s in stats)
                report["segment_no_new_evidence_call_rate"] = (
                    sum(s.get("segment_no_new_evidence_calls", 0.0) for s in stats) / segment_calls
                    if segment_calls else 0.0)
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
