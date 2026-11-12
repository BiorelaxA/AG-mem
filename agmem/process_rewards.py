"""Frozen, step-level reward/action attribution for accepted DAPO groups.

Terminal rewards and filtering remain independent. Scoring happens after a
sampling window drains and before any actor update, using the same Reader.
"""
from __future__ import annotations

import math

from .core import InfrastructureError, digest
from .process_advantages import process_reward_settings


PROCESS_PROTOCOL = "reader_process_attribution_v1"


def attribution_settings(config):
    settings = process_reward_settings(config["training"])
    raw = config["training"].get("process_reward", {})
    low, high = raw.get("weight_min", 0.5), raw.get("weight_max", 2.0)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (low, high)) or not 0 < low <= 1 <= high:
        raise ValueError("Process weights require 0 < weight_min <= 1 <= weight_max")
    settings.update(weight_min=low, weight_max=high)
    if settings["enabled"]:
        if config["reader"].get("prompt_version") not in {"interaction_history_v4", "interaction_history_v5"}:
            raise ValueError("Process attribution requires Reader interaction_history_v4 or interaction_history_v5")
        if config["retrieval"].get("id_protocol") != "strict_v2":
            raise ValueError("Process attribution requires strict_v2 retrieval")
        if config["training"].get("target_groups", 16) != config["training"].get("mini_batch_groups", 1):
            raise ValueError("Process attribution requires target_groups=mini_batch_groups for fresh sampling")
    return settings


def evidence_prefixes(trajectory):
    """Reconstruct E_0..E_T from successful returns, never from future evidence."""
    evidence, prefixes = {}, [[]]
    for step in trajectory.decisions:
        for record in (step.tool_result or {}).get("records", []):
            key = record["id"]
            if key in evidence and evidence[key] != record:
                raise ValueError("A retrieved session changed within one rollout")
            evidence[key] = record
        prefixes.append(list(evidence.values()))
    if prefixes[-1] != trajectory.evidence:
        raise ValueError("Process prefixes do not match the saved final evidence")
    return prefixes


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def action_weights(deltas, low, high):
    """Normalize exp(delta) over observed transitions of this trajectory.

    Missing actual next actions receive exactly one and are excluded from the
    normalization denominator. Log-sum-exp avoids numerical overflow.
    """
    observed = [value for value in deltas if value is not None]
    if any(not _finite(value) for value in observed):
        raise ValueError("Nonfinite action attribution")
    if not observed:
        return [1.0] * len(deltas)
    maximum = max(observed)
    log_mean = maximum + math.log(math.fsum(math.exp(x - maximum) for x in observed) / len(observed))
    return [1.0 if value is None else
            math.exp(min(math.log(high), max(math.log(low), value - log_mean))) for value in deltas]


class ProcessAttributionScorer:
    def __init__(self, config, likelihood, *, action_logprob=None):
        self.settings = attribution_settings(config)
        if not self.settings["enabled"]:
            raise ValueError("ProcessAttributionScorer requires enabled process rewards")
        self.likelihood = likelihood
        self.action_logprob = action_logprob
        self.workers = config.get("execution", {}).get("task_agent_workers", 1)

    def provenance(self):
        return {"protocol": PROCESS_PROTOCOL, "settings": self.settings,
                "likelihood": self.likelihood.prefix_provenance(),
                "prefix_scoring": "task_reference_prefix_mean_logprob_v1",
                "action_target": "exact_next_sampled_completion_tokens_including_eos",
                "action_probability": "token_mean_logprob_at_behavior_temperature",
                "counterfactual": "previous_decision_prompt_without_current_interaction",
                "weight_normalization": "observed_transitions_within_each_trajectory",
                "missing_next_action_weight": 1.0,
                "scored_groups": "accepted_groups_only_before_actor_update"}

    def close(self):
        self.likelihood.close()

    def _counterfactual(self, policy, prompt_ids, target_ids):
        if self.action_logprob is not None:
            return self.action_logprob(policy, prompt_ids, target_ids)
        import torch
        from .policy import sequence_logprobs
        was_training = policy.model.training
        policy.model.eval()
        try:
            with torch.no_grad():
                values = sequence_logprobs(policy.model, prompt_ids, target_ids,
                                          policy.temperature, policy.device)
                return values.mean().item()
        finally:
            policy.model.train(was_training)

    def _score_row(self, example, row, likelihood, policy):
        if row.policy_version != policy.policy_version:
            raise ValueError("Process attribution cannot use an updated Reader for an old rollout")
        fs = likelihood["prefix_logprobs"]
        if len(fs) != len(row.decisions) + 1 or any(not _finite(x) for x in fs):
            raise InfrastructureError("Invalid prefix likelihoods; no process score assigned")
        entries, deltas = [], []
        for index, decision in enumerate(row.decisions):
            is_read = decision.action is not None and not decision.action["stop"]
            before, after = fs[index:index + 2]
            gain = after - before if is_read else 0.0
            entry = {"is_read": is_read, "f_before": before, "f_after": after,
                     "g": gain, "ell_plus": None, "ell_minus": None, "log_support_ratio": None}
            delta = None
            if is_read and index + 1 < len(row.decisions):
                next_step = row.decisions[index + 1]
                # A malformed completion is not a READ/STOP action. Its terminal
                # format penalty is retained, but no action support is invented.
                if next_step.action is not None:
                    target = next_step.completion
                    if (not decision.completion.prompt_ids or not target.token_ids
                            or len(target.old_logprobs) != len(target.token_ids)
                            or any(not _finite(x) or x > 1e-6 for x in target.old_logprobs)):
                        raise ValueError("Missing sampled action probabilities for process attribution")
                    plus = math.fsum(target.old_logprobs) / len(target.old_logprobs)
                    minus = self._counterfactual(policy, decision.completion.prompt_ids, target.token_ids)
                    if not _finite(minus) or minus > 1e-6:
                        raise InfrastructureError("Invalid Reader action logprob; no process score assigned")
                    delta = plus - minus
                    entry.update(ell_plus=plus, ell_minus=minus, log_support_ratio=delta,
                                 counterfactual_prompt_sha256=digest(decision.completion.prompt_ids),
                                 target_token_sha256=digest(target.token_ids))
            entries.append(entry)
            deltas.append(delta)
        weights = action_weights(deltas, self.settings["weight_min"], self.settings["weight_max"])
        for entry, weight in zip(entries, weights):
            gain = entry["g"]
            entry.update(w=weight, reward=weight * gain if gain >= 0 else gain / weight)
            if not _finite(entry["reward"]):
                raise InfrastructureError("Nonfinite process reward; no score assigned")
        return {"protocol": PROCESS_PROTOCOL, "behavior_version": policy.policy_version,
                "scoring_sha256": digest(self.provenance()),
                "prefix_logprobs": fs, "reference_token_count": likelihood["reference_token_count"],
                "references": likelihood["references"], "per_decision": entries}

    def score_group(self, example, group, policy, *, on_progress=None):
        from .execution import bounded_results
        if any(row.sample_id != example.sample_id or row.memory_id != example.memory_id for row in group):
            raise ValueError("Process scoring requires the matching question and memory")
        if any(row.policy_version != policy.policy_version for row in group):
            raise ValueError("Process scoring requires the frozen sampling Reader")
        if any(row.process_attribution for row in group):
            raise ValueError("Do not rescore stored process attribution")
        def score_prefix(index):
            return self.likelihood.score_prefixes(example, evidence_prefixes(group[index]))
        scored = {}
        for index, value in bounded_results(score_prefix, range(len(group)), self.workers):
            scored[index] = self._score_row(example, group[index], value, policy)
            if on_progress:
                on_progress({"sample_id": example.sample_id, "completed": len(scored), "total": len(group)})
        # Atomic group assignment: service failures cannot produce a partially
        # scored training group or become a task/format penalty.
        for index, row in enumerate(group):
            row.process_attribution = scored[index]
        return group


def process_metrics(rows, *, overlong_filter=True):
    from .dapo import trajectory_truncated
    rows = [row for row in rows if not (overlong_filter and trajectory_truncated(row))]
    entries = [entry for row in rows for entry in row.process_attribution.get("per_decision", [])]
    reads = [entry for entry in entries if entry["is_read"]]
    if not reads:
        return {}
    result = {"process_negative_gain_fraction": sum(e["g"] < 0 for e in reads) / len(reads),
              "process_zero_gain_fraction": sum(e["g"] == 0 for e in reads) / len(reads)}
    observed = [entry["log_support_ratio"] for entry in reads if entry.get("log_support_ratio") is not None]
    if observed:
        result["process_positive_support_fraction"] = sum(value > 0 for value in observed) / len(observed)
    if all("advantage" in entry for entry in entries):
        wrong_reads = [entry for row in rows if row.judge_score == 0
                       for entry in row.process_attribution.get("per_decision", []) if entry["is_read"]]
        bad_actions = [entry for row in rows for decision, entry in
                       zip(row.decisions, row.process_attribution.get("per_decision", []))
                       if decision.format_violations]
        for name, selected in (("wrong_read", wrong_reads), ("bad_action", bad_actions)):
            result[f"process_{name}_steps"] = len(selected)
            result[f"process_{name}_positive_advantage_rate"] = (
                sum(entry["advantage"] > 0 for entry in selected) / len(selected) if selected else 0.0)
    return result


def make_process_scorer(config, store):
    if not attribution_settings(config)["enabled"]:
        return None
    from .likelihood import ReferenceLikelihoodScorer
    from .policy import load_tokenizer, model_fingerprint
    from .services import resolve_task_service
    service = resolve_task_service(config)
    if service.get("request_profile", "local_vllm") != "local_vllm":
        raise ValueError("Process likelihood requires a local prompt-logprob Task service")
    service = {**service, "model_revision": model_fingerprint(config["models"]["task_path"])["fingerprint"]}
    likelihood = ReferenceLikelihoodScorer(
        service, store, {"mode": "diagnostic", "max_context_tokens": config["task_agent"].get("max_context_tokens", 65536)},
        tokenizer_factory=lambda: load_tokenizer(config["models"]["task_path"]),
        prompt_version=config["task_agent"].get("prompt_version", "legacy_v1"))
    return ProcessAttributionScorer(config, likelihood)
