from __future__ import annotations

import math

from .policy import sequence_logprobs


def clipped_loss(new_logprobs, old_logprobs, advantage, reference_logprobs=None, clip_ratio=0.2, beta=0.01):
    import torch
    ratio = torch.exp(new_logprobs - old_logprobs)
    policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
    kl = torch.zeros_like(policy_loss)
    if reference_logprobs is not None and beta:
        difference = reference_logprobs - new_logprobs
        kl = torch.exp(difference) - difference - 1  # Sampled nonnegative k3 KL estimator.
    result = policy_loss + beta * kl
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite GRPO loss/ratio/KL")
    return result, kl


class CPUAdamW:
    """Full precision CPU master weights and Adam moments; model parameters remain BF16 on GPU.

    This trades CPU RAM/transfers for GPU memory; it is not LoRA, quantized training or ZeRO sharding.
    """
    def __init__(self, model, learning_rate, weight_decay=0.0):
        import torch
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.master = [torch.nn.Parameter(parameter.detach().float().cpu().clone()) for parameter in self.parameters]
        self.optimizer = torch.optim.AdamW(self.master, lr=learning_rate, weight_decay=weight_decay)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        import torch
        with torch.no_grad():
            for parameter, master in zip(self.parameters, self.master):
                master.grad = parameter.grad.detach().float().cpu() if parameter.grad is not None else None
            self.optimizer.step()
            for parameter, master in zip(self.parameters, self.master):
                if master.grad is not None:
                    parameter.copy_(master.to(parameter.device, dtype=parameter.dtype))
        self.zero_grad()

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "master": [item.detach() for item in self.master]}

    def load_state_dict(self, state):
        import torch
        if len(state["master"]) != len(self.master):
            raise ValueError("Optimizer parameter count changed")
        self.optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            for master, saved, parameter in zip(self.master, state["master"], self.parameters):
                master.copy_(saved)
                parameter.copy_(master.to(parameter.device, dtype=parameter.dtype))


class GRPOUpdater:
    def __init__(self, policy, optimizer, config, reference=None):
        self.policy, self.optimizer, self.config, self.reference = policy, optimizer, config, reference

    def update(self, group):
        import torch
        if len(group) != 8 or len({row.sample_id for row in group}) != 1 or len({row.memory_id for row in group}) != 1:
            raise ValueError("A GRPO group must be eight complete rollouts of one question/snapshot")
        if any(row.policy_version != self.policy.policy_version for row in group):
            raise ValueError("Stale rollout policy version")
        for row in group:
            if not row.decisions or not all(len(step.completion.token_ids) == len(step.completion.old_logprobs) > 0 for step in row.decisions):
                raise ValueError("Missing generation token IDs/behavior logprobs")
        reference_values = {}
        if self.reference is not None:
            self.reference.to(self.policy.device).eval()
            with torch.inference_mode():
                for i, row in enumerate(group):
                    for j, step in enumerate(row.decisions):
                        value = step.completion
                        reference_values[i, j] = sequence_logprobs(self.reference, value.prompt_ids, value.token_ids, self.policy.temperature, self.policy.device).detach().cpu()
            self.reference.to("cpu")
            if self.policy.device.type == "cuda":
                torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        self.policy.model.train()
        total_loss, total_kl, max_logprob_error = 0.0, 0.0, 0.0
        for i, row in enumerate(group):
            token_count = sum(len(step.completion.token_ids) for step in row.decisions)
            for j, step in enumerate(row.decisions):
                value = step.completion
                new = sequence_logprobs(self.policy.model, value.prompt_ids, value.token_ids, self.policy.temperature, self.policy.device)
                old = torch.tensor(value.old_logprobs, dtype=new.dtype, device=new.device)
                # Diagnostic only: do not reject a rollout or replace its actual sampling probabilities.
                error = (new.detach() - old).abs().max().item()
                max_logprob_error = max(error, max_logprob_error)
                ref = reference_values[i, j].to(new.device) if (i, j) in reference_values else None
                loss, kl = clipped_loss(new, old, row.advantage, ref, self.config["clip_ratio"], self.config["kl_coefficient"])
                scaled = loss.sum() / (8 * token_count)
                scaled.backward()
                total_loss += scaled.detach().item()
                total_kl += kl.detach().sum().item() / (8 * token_count)
        norm = torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.config["max_grad_norm"], error_if_nonfinite=True)
        self.optimizer.step()
        self.policy.policy_version += 1
        self.policy.model.eval()
        return {"loss": total_loss, "sampled_kl": total_kl, "grad_norm": float(norm), "max_old_logprob_error": max_logprob_error}
