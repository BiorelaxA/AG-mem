from copy import deepcopy
from types import SimpleNamespace
import math
import unittest
from unittest.mock import patch

import torch

from agmem.core import Completion, Decision, Trajectory
from agmem.grpo import CPUAdamW, GRPOUpdater, clipped_loss
from agmem.policy import HFPolicy, sequence_logprobs
from agmem.rewards import advantages


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 6)
        self.head = torch.nn.Linear(6, 9)
    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
        hidden = self.embedding(input_ids).cumsum(1)
        logits = self.head(hidden)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


class GradientTests(unittest.TestCase):
    def test_actual_generation_scores_are_saved_without_consistency_checks(self):
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 8
            def apply_chat_template(self, *args, **kwargs):
                return [1, 2, 3]
            def decode(self, *args, **kwargs):
                return '{"ids":[],"stop":true}'
        class GeneratingLM(TinyLM):
            bias = 0.0
            truncate_scores = False
            config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=128))
            def generate(self, input_ids, **kwargs):
                self.generation_kwargs = kwargs
                seq = input_ids.clone()
                scores = []
                for token in [4, 5]:
                    raw = self.forward(seq).logits[:, -1, :].float().clone()
                    score = raw.clone() / kwargs["temperature"]
                    score[:, token] += self.bias
                    scores.append(score)
                    seq = torch.cat([seq, torch.tensor([[token]])], dim=1)
                return SimpleNamespace(sequences=seq, scores=scores[:-1] if self.truncate_scores else scores)
        model = GeneratingLM()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        config = {"rollout_temperature": 0.8, "max_new_tokens_per_decision": 2, "max_context_tokens": 128}
        policy = HFPolicy(model, Tokenizer(), config)
        # Sampling must not run an extra teacher-forcing pass, even with a large score difference.
        with patch("agmem.policy.sequence_logprobs", side_effect=AssertionError("Unexpected sampling-time recomputation")):
            for bias in (0.0, 1.0):
                with self.subTest(bias=bias):
                    model.bias = bias
                    completion = policy.complete([], 42)
                    expected = bias - math.log(math.exp(bias) + 8)
                    torch.testing.assert_close(torch.tensor(completion.old_logprobs), torch.full((2,), expected))
                    self.assertEqual(completion.token_ids, [4, 5])
                    self.assertTrue(model.generation_kwargs["output_scores"])
                    self.assertFalse(model.generation_kwargs.get("output_logits", False))
        teacher = sequence_logprobs(model, [1, 2, 3], [4, 5], 0.8, "cpu").detach()
        self.assertGreater((torch.tensor(completion.old_logprobs) - teacher).abs().max().item(), 0.18)
        # Token/score alignment is still required to save actual behavior probabilities.
        model.truncate_scores = True
        with self.assertRaisesRegex(ValueError, "scores do not align"):
            policy.complete([], 42)

    def test_completion_alignment_excludes_prompt_targets(self):
        torch.manual_seed(7)
        model = TinyLM()
        prompt, generated = [1, 2, 3], [4, 5]
        values = sequence_logprobs(model, prompt, generated, 0.8, "cpu")
        output = model(torch.tensor([prompt + generated])).logits[0]
        expected = torch.log_softmax(output[2:4] / 0.8, -1).gather(-1, torch.tensor([[4], [5]])).flatten()
        torch.testing.assert_close(values, expected)
        self.assertEqual(values.numel(), len(generated))

    def test_full_group_updates_only_policy_and_preserves_reference(self):
        torch.manual_seed(19)
        model, group = TinyLM(), []
        reference = deepcopy(model).requires_grad_(False)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        policy = SimpleNamespace(model=model, device=torch.device("cpu"), temperature=1.0, policy_version=0)
        for i, advantage in enumerate(advantages([0, 1] * 4)):
            generated = [4 if i % 2 else 5, 6]
            old = sequence_logprobs(model, [1, 2, 3], generated, 1.0, "cpu").detach().tolist()
            completion = Completion("fixture action", [1, 2, 3], generated, old)
            group.append(Trajectory("same-query", i, "same-memory", decisions=[Decision([], completion)], advantage=advantage))
        optimizer = CPUAdamW(model, 0.01)
        config = {"clip_ratio": 0.2, "kl_coefficient": 0.01, "max_grad_norm": 1.0}
        metrics = GRPOUpdater(policy, optimizer, config, reference).update(group)
        self.assertEqual(policy.policy_version, 1)
        self.assertGreater(metrics["grad_norm"], 0)
        self.assertTrue(any(not torch.equal(before[key], value) for key, value in model.state_dict().items()))
        for key, value in reference.state_dict().items():
            torch.testing.assert_close(value, before[key])
        saved = deepcopy(optimizer.state_dict())
        optimizer.load_state_dict(saved)
        for parameter, master in zip(model.parameters(), optimizer.master):
            torch.testing.assert_close(parameter, master)
        with self.assertRaises(ValueError):
            GRPOUpdater(policy, optimizer, config).update(group)

    def test_ratio_clipping_and_zero_advantage(self):
        new = torch.tensor([0.0, 1.0], requires_grad=True)
        old = torch.zeros(2)
        loss, kl = clipped_loss(new, old, 1.0, beta=0)
        torch.testing.assert_close(loss, torch.tensor([-1.0, -1.2]))
        zero, _ = clipped_loss(new, old, 0.0, beta=0)
        torch.testing.assert_close(zero, torch.zeros_like(zero))

    def test_logprob_drift_is_recorded_without_blocking_updates_or_replacing_behavior(self):
        for drift in (0.08, 0.4):
            with self.subTest(drift=drift):
                torch.manual_seed(23)
                model = TinyLM()
                policy = SimpleNamespace(model=model, device=torch.device("cpu"), temperature=1.0, policy_version=0)
                group = []
                for index, advantage in enumerate([-1.0, 1.0] * 4):
                    tokens = [4 if advantage > 0 else 5]
                    teacher = sequence_logprobs(model, [1, 2, 3], tokens, 1.0, "cpu").detach().tolist()
                    actual_behavior = [value + drift * advantage for value in teacher]
                    completion = Completion("fixture", [1, 2, 3], tokens, actual_behavior)
                    group.append(Trajectory("query", index, "memory", decisions=[Decision([], completion)], advantage=advantage))
                config = {"clip_ratio": 0.2, "kl_coefficient": 0.0, "max_grad_norm": 1.0}
                metrics = GRPOUpdater(policy, CPUAdamW(model, 0.01), config).update(group)
                # Replacing behavior probabilities with teacher-forced ones would incorrectly give zero here.
                self.assertAlmostEqual(metrics["loss"], math.sinh(drift), places=5)
                self.assertAlmostEqual(metrics["max_old_logprob_error"], drift, places=5)
                self.assertEqual(policy.policy_version, 1)


if __name__ == "__main__":
    unittest.main()
