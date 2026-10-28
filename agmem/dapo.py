"""DAPO actor updates for the existing multi-decision Reader execution engine.

This is AG-mem's implementation of the DAPO objective, not a verl/Ray backend.
The loss follows arXiv:2503.14476v1 equations 8--12: decoupled clipping and a
shared completion-token denominator, with no reference-policy KL. Dynamic
sampling lives in the training loop; truncated trajectories can be loss-masked
as in the paper's overlong filtering ablation. Original prompts stay unchanged.
"""
from __future__ import annotations

import math

from .core import canonical, digest
from .policy import sequence_stats, sequence_stats_batch
from .runtime_memory import activation_offload_context, memory_settings, use_activation_offload


def clipped_loss(new_logprobs, old_logprobs, advantage, clip_low=0.2, clip_high=0.28):
    """Per-token negative DAPO clipped surrogate, including negative advantages."""
    import torch
    ratio = torch.exp(new_logprobs - old_logprobs)
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_low, 1 + clip_high) * advantage)
    if not torch.isfinite(ratio).all() or not torch.isfinite(loss).all():
        raise FloatingPointError("Non-finite DAPO loss/ratio")
    return loss


def trajectory_truncated(row):
    """A token-length truncation is distinct from invalid JSON or a read budget."""
    return any(getattr(step.completion, "truncated", False) for step in row.decisions)


def valid_training_tokens(group, overlong_filter=True):
    return sum(len(step.completion.token_ids)
               for row in group if not (overlong_filter and trajectory_truncated(row))
               for step in row.decisions)


def _group_signature(group):
    # Bind both advantages and actual sampling probabilities. Neither may be
    # recomputed between mini-batches or silently changed after a checkpoint.
    return digest([{
        "sample_id": row.sample_id, "memory_id": row.memory_id, "seed": row.seed,
        "policy_version": row.policy_version, "advantage": row.advantage,
        "decisions": [{"prompt_ids": step.completion.prompt_ids,
                       "token_ids": step.completion.token_ids,
                       "old_logprobs": step.completion.old_logprobs,
                       "truncated": getattr(step.completion, "truncated", False)}
                      for step in row.decisions],
    } for row in group])


def _micro_batches(rows, rollout_limit, token_budget):
    """Plan decision forwards, with at most one pending decision per rollout.

    The budget counts padded input tokens (batch width times sequence count),
    not target tokens. A long decision runs alone even above the budget. Serial
    execution keeps its original rollout/decision order exactly. Batched execution
    picks the shortest pending decisions, retaining each rollout's decision order.
    """
    if rollout_limit == 1:
        return [[(i, j, step.completion)] for i, row in enumerate(rows)
                for j, step in enumerate(row.decisions)]
    pending = [0] * len(rows)
    batches = []
    while True:
        candidates = sorted((len(row.decisions[j].completion.prompt_ids)
                             + len(row.decisions[j].completion.token_ids) - 1, i, j)
                            for i, (row, j) in enumerate(zip(rows, pending)) if j < len(row.decisions))
        if not candidates:
            return batches
        block = []
        for width, i, j in candidates:
            if block and (len(block) >= rollout_limit or width * (len(block) + 1) > token_budget):
                break
            block.append((i, j, rows[i].decisions[j].completion))
            pending[i] += 1
        batches.append(block)


class DAPOUpdater:
    """Consume one frozen behavior batch in ordered, complete-group mini-batches.

    mini_batch_groups controls complete query groups per optimizer step, while
    micro_batch_rollouts bounds simultaneous decision sequences. A rollout may
    need multiple forwards. All micro-batches share one completion-token
    denominator across the entire mini-batch and one optimizer step.
    The caller persists full groups, batch_id, behavior_version and consumed_groups
    alongside optimizer state, then supplies them to begin_batch when resuming.
    """

    def __init__(self, policy, optimizer, config, runtime_memory=None):
        self.policy, self.optimizer, self.config = policy, optimizer, config
        self.runtime_memory = memory_settings() if runtime_memory is None else runtime_memory
        self.group_size = config.get("group_size", 8)
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError("DAPO group_size must be an integer >= 2")
        for name, default in (("mini_batch_groups", 1), ("micro_batch_rollouts", 1),
                              ("max_micro_batch_tokens", 16384)):
            value = config.get(name, default)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        if config.get("kl_coefficient", 0.0) != 0:
            raise ValueError("DAPO does not load a reference model; kl_coefficient must be 0")
        if config.get("ppo_epochs", 1) != 1:
            raise ValueError("This DAPO execution engine uses one epoch per behavior batch")
        self.clip_low, self.clip_high = config.get("clip_ratio_low", 0.2), config.get("clip_ratio_high", 0.28)
        if not (isinstance(self.clip_low, (int, float)) and isinstance(self.clip_high, (int, float))
                and math.isfinite(self.clip_low) and math.isfinite(self.clip_high)
                and 0 < self.clip_low < 1 and self.clip_high >= self.clip_low):
            raise ValueError("DAPO requires 0 < clip_ratio_low < 1 and clip_ratio_high >= clip_ratio_low")
        self.overlong_filter = config.get("overlong_filter", True)
        self._signatures = None
        self._cursor = 0
        self.batch_id = None
        self.behavior_version = None

    @property
    def consumed_groups(self):
        return self._cursor

    def begin_batch(self, groups, batch_id, consumed_groups=0, behavior_version=None):
        if self._signatures is not None and self._cursor < len(self._signatures):
            raise ValueError("Cannot replace an unconsumed DAPO behavior batch")
        if not groups or batch_id is None or not str(batch_id):
            raise ValueError("A nonempty DAPO behavior batch and batch_id are required")
        if type(consumed_groups) is not int or not 0 <= consumed_groups <= len(groups):
            raise ValueError("Invalid consumed_groups checkpoint cursor")
        if len(groups) % self.mini_batch_groups or consumed_groups % self.mini_batch_groups:
            raise ValueError("Behavior batch length and consumed_groups cursor must align with mini_batch_groups")
        if behavior_version is None:
            behavior_version = groups[0][0].policy_version if groups[0] else None
        if type(behavior_version) is not int or behavior_version < 0:
            raise ValueError("Invalid behavior policy version")
        if self.policy.policy_version != behavior_version + consumed_groups // self.mini_batch_groups:
            raise ValueError("Behavior batch/checkpoint cursor does not match current policy version")
        signatures = []
        for group in groups:
            if len(group) != self.group_size or len({row.sample_id for row in group}) != 1 or len({row.memory_id for row in group}) != 1:
                raise ValueError("Each DAPO group must contain complete rollouts of one question/snapshot")
            if any(row.policy_version != behavior_version for row in group):
                raise ValueError("Mixed or stale behavior policy versions in DAPO batch")
            for row in group:
                if not math.isfinite(row.advantage) or not row.decisions:
                    raise ValueError("Missing decisions or non-finite fixed advantages")
                for step in row.decisions:
                    value = step.completion
                    if not value.prompt_ids or not (len(value.token_ids) == len(value.old_logprobs) > 0):
                        raise ValueError("Missing generation token IDs/behavior logprobs")
                    if any(not math.isfinite(p) for p in value.old_logprobs):
                        raise ValueError("Non-finite behavior logprobs")
            if not valid_training_tokens(group, self.overlong_filter):
                raise ValueError("DAPO group has no unmasked training tokens")
            if not any(row.advantage != 0 for row in group
                       if not (self.overlong_filter and trajectory_truncated(row))):
                raise ValueError("DAPO group has no unmasked advantage signal; filter it before updating")
            signatures.append(_group_signature(group))
        self._signatures, self._cursor = signatures, consumed_groups
        self.batch_id, self.behavior_version = str(batch_id), behavior_version

    def end_batch(self):
        if self._signatures is not None and self._cursor != len(self._signatures):
            raise ValueError("Cannot finish an unconsumed DAPO behavior batch")
        self._signatures = None
        self.batch_id = self.behavior_version = None
        self._cursor = 0

    def update(self, group):
        """Backward-compatible single-group update; use update_groups for larger minis."""
        if self.mini_batch_groups != 1:
            raise ValueError("Use update_groups with mini_batch_groups complete groups")
        return self.update_groups([group])

    def update_groups(self, groups):
        """Consume exactly one mini-batch; advantages remain fixed within each query."""
        import torch
        if self._signatures is None or self._cursor >= len(self._signatures):
            raise ValueError("Call begin_batch before consuming a DAPO group")
        if len(groups) != self.mini_batch_groups or self._cursor + len(groups) > len(self._signatures):
            raise ValueError("update_groups requires exactly mini_batch_groups remaining complete groups")
        if self.policy.policy_version != self.behavior_version + self._cursor // self.mini_batch_groups:
            raise ValueError("Policy changed outside the active DAPO behavior batch")
        if any(_group_signature(group) != self._signatures[self._cursor + offset]
               for offset, group in enumerate(groups)):
            raise ValueError("Out-of-order, changed or foreign DAPO group")
        group = [row for query_group in groups for row in query_group]
        rows = [row for row in group if not (self.overlong_filter and trajectory_truncated(row))]
        token_count = valid_training_tokens(rows, False)
        lengths = [len(step.completion.prompt_ids) + len(step.completion.token_ids) - 1
                   for row in rows for step in row.decisions]
        batches = _micro_batches(rows, self.micro_batch_rollouts, self.max_micro_batch_tokens)
        batch_tokens = [len(block) * max(len(value.prompt_ids) + len(value.token_ids) - 1
                                        for _, _, value in block) for block in batches]
        offloaded = sum(len(block) for block, tokens in zip(batches, batch_tokens)
                        if use_activation_offload(tokens, self.policy.device, self.runtime_memory))
        batch_report = {"update_groups": len(groups), "update_rollouts": len(group),
                        "micro_batches": len(batches), "max_micro_batch_tokens": self.max_micro_batch_tokens,
                        "max_micro_batch_input_tokens": max(batch_tokens),
                        "micro_batch_input_tokens": sum(batch_tokens),
                        "micro_batch_padding_tokens": sum(batch_tokens) - sum(lengths),
                        "micro_batch_over_budget_singletons": sum(tokens > self.max_micro_batch_tokens
                                                                    for tokens in batch_tokens)}
        devices = sorted({parameter.device.index for parameter in self.policy.model.parameters()
                          if parameter.device.type == "cuda"})
        for device in devices:
            torch.cuda.reset_peak_memory_stats(device)
        print(canonical({"event": "dapo_update_started", "batch_id": self.batch_id,
                         "behavior_policy_version": self.behavior_version,
                         "policy_version": self.policy.policy_version, "decisions": len(lengths),
                         "training_tokens": token_count, "masked_rollouts": len(group) - len(rows),
                         "max_update_input_tokens": max(lengths), "activation_offload_decisions": offloaded,
                         **batch_report}), flush=True)
        self.optimizer.zero_grad()
        self.policy.model.train()
        total_loss = total_entropy = total_surprisal = total_behavior_surprisal = 0.0
        total_kl = total_clipped = 0.0
        max_logprob_error = 0.0
        try:
            remaining = [len(row.decisions) for row in rows]
            pad_token_id = getattr(getattr(self.policy, "tokenizer", None), "pad_token_id", None)
            for block_index, (block, input_tokens) in enumerate(zip(batches, batch_tokens)):
                try:
                    with activation_offload_context(input_tokens, self.policy.device, self.runtime_memory):
                        if len(block) == 1:
                            value = block[0][2]
                            stats = [sequence_stats(self.policy.model, value.prompt_ids, value.token_ids,
                                                    self.policy.temperature, self.policy.device,
                                                    entropy_chunk_size=self.config.get("entropy_chunk_size", 32))]
                        else:
                            stats = sequence_stats_batch(
                                self.policy.model, [value.prompt_ids for _, _, value in block],
                                [value.token_ids for _, _, value in block], self.policy.temperature,
                                self.policy.device, pad_token_id=pad_token_id if pad_token_id is not None else 0,
                                entropy_chunk_size=self.config.get("entropy_chunk_size", 32))
                        losses, olds = [], []
                        for (i, _, value), (new, _) in zip(block, stats):
                            old = torch.tensor(value.old_logprobs, dtype=new.dtype, device=new.device)
                            losses.append(clipped_loss(new, old, rows[i].advantage,
                                                       self.clip_low, self.clip_high).sum() / token_count)
                            olds.append(old)
                        scaled = losses[0] if len(losses) == 1 else torch.stack(losses).sum()
                    scaled.backward()
                except torch.OutOfMemoryError:
                    print(canonical({"event": "dapo_backward_oom", "batch_id": self.batch_id,
                                     "micro_batch_index": block_index, "input_tokens": input_tokens,
                                     "decisions": [[i, j] for i, j, _ in block]}), flush=True)
                    raise
                total_loss += scaled.detach().item()
                for (i, _, _), (new, entropy), old in zip(block, stats, olds):
                    with torch.no_grad():
                        log_ratio = new.detach() - old
                        ratio = log_ratio.exp()
                        max_logprob_error = max(max_logprob_error, log_ratio.abs().max().item())
                        total_entropy += entropy.sum().item()
                        total_surprisal -= new.detach().sum().item()
                        total_behavior_surprisal -= old.sum().item()
                        total_kl += (ratio - 1 - log_ratio).sum().item()
                        total_clipped += ((ratio < 1 - self.clip_low) | (ratio > 1 + self.clip_high)).sum().item()
                    remaining[i] -= 1
                print(canonical({"event": "dapo_backward_progress", "batch_id": self.batch_id,
                                 "rollouts_completed": sum(n == 0 for n in remaining), "rollouts_total": len(rows),
                                 "micro_batches_completed": block_index + 1, "micro_batches_total": len(batches)}), flush=True)
            norm = torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.config["max_grad_norm"], error_if_nonfinite=True)
            self.optimizer.step()
        except BaseException:
            self.optimizer.zero_grad()
            raise
        finally:
            self.policy.model.eval()
        self.policy.policy_version += 1
        self._cursor += len(groups)
        peaks = {f"gpu_{device}_peak_{kind}_gib": getattr(torch.cuda, f"max_memory_{kind}")(device) / 2**30
                 for device in devices for kind in ("allocated", "reserved")}
        return {"loss": total_loss, "entropy": total_entropy / token_count,
                "sampled_surprisal": total_surprisal / token_count,
                "behavior_sampled_surprisal": total_behavior_surprisal / token_count,
                "behavior_approx_kl": total_kl / token_count, "clip_fraction": total_clipped / token_count,
                "grad_norm": float(norm), "max_old_logprob_error": max_logprob_error,
                "training_tokens": token_count, "masked_rollouts": len(group) - len(rows),
                "activation_offload_decisions": offloaded, "max_update_input_tokens": max(lengths),
                **batch_report, **peaks}
