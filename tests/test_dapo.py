from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import torch

from agmem.core import Completion, Decision, Trajectory
from agmem.dapo import DAPOUpdater, clipped_loss, valid_training_tokens
from agmem.policy import HFPolicy, sequence_logprobs, sequence_stats


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4)
        self.head = torch.nn.Linear(4, 7)
        self.forward_count = 0

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0, position_ids=None):
        self.forward_count += 1
        embedded = self.embedding(input_ids)
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1)
        logits = self.head(embedded.cumsum(dim=1))
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


def policy_for(model, version=0):
    return SimpleNamespace(model=model, device=torch.device("cpu"), temperature=1.0, policy_version=version)


def make_group(model, sample_id="q", *, version=0, lengths=(1, 3)):
    group = []
    for i, (length, advantage) in enumerate(zip(lengths, [-1.0, 1.0])):
        decisions = []
        # The positive rollout has two decisions with distinct true prompts.
        remaining = length
        while remaining:
            count = min(2, remaining)
            prompt = [1, 2] + ([6] if decisions else [])
            tokens = [3 + i] * count
            old = sequence_logprobs(model, prompt, tokens, 1.0, "cpu").detach().tolist()
            decisions.append(Decision([], Completion("action", prompt, tokens, old)))
            remaining -= count
        group.append(Trajectory(sample_id, i, "memory", policy_version=version,
                                decisions=decisions, advantage=advantage))
    return group


def config(**overrides):
    return {"group_size": 2, "mini_batch_groups": 1, "micro_batch_rollouts": 1,
            "kl_coefficient": 0, "max_grad_norm": 10000.0, **overrides}


class DAPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_decoupled_clip_keeps_negative_advantages(self):
        ratios = torch.tensor([0.5, 1.0, 1.25, 1.5])
        new = ratios.log().requires_grad_()
        old = torch.zeros_like(new)
        positive = clipped_loss(new, old, 1.0)
        negative = clipped_loss(new, old, -1.0)
        torch.testing.assert_close(positive, torch.tensor([-0.5, -1.0, -1.25, -1.28]))
        torch.testing.assert_close(negative, torch.tensor([0.8, 1.0, 1.25, 1.5]))
        positive.sum().backward()
        self.assertAlmostEqual(new.grad[-1].item(), 0.0)
        self.assertNotEqual(new.grad[-2].item(), 0.0)

    def test_entropy_is_full_distribution_aligned_and_detached(self):
        torch.manual_seed(5)
        model = TinyLM()
        prompt, tokens = [1, 2, 6], [3, 4]
        calls = model.forward_count
        logprobs, entropy = sequence_stats(model, prompt, tokens, 0.8, "cpu", entropy_chunk_size=1)
        self.assertEqual(model.forward_count, calls + 1)
        logits = model(torch.tensor([prompt + tokens])).logits[0, 2:4] / 0.8
        p = torch.softmax(logits, dim=-1)
        expected = -(p * torch.log_softmax(logits, dim=-1)).sum(-1)
        torch.testing.assert_close(entropy, expected)
        torch.testing.assert_close(logprobs, torch.log_softmax(logits, -1).gather(1, torch.tensor([[3], [4]])).flatten())
        self.assertTrue(logprobs.requires_grad)
        self.assertFalse(entropy.requires_grad)
        self.assertFalse(torch.allclose(entropy, -logprobs))
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
        _, uniform = sequence_stats(model, prompt, tokens, 1.0, "cpu")
        torch.testing.assert_close(uniform, torch.full((2,), math.log(7)))

    def test_generation_distinguishes_limit_truncation_from_eos_at_limit(self):
        class Tokenizer:
            pad_token_id, eos_token_id = 0, 6

            def decode(self, tokens, **kwargs):
                return str(tokens)

        class GeneratingLM(TinyLM):
            tokens = [3, 4]

            def generate(self, input_ids, **kwargs):
                continuation = torch.tensor([self.tokens], device=input_ids.device)
                return SimpleNamespace(sequences=torch.cat([input_ids, continuation], dim=1),
                                       scores=[torch.zeros(1, 7) for _ in self.tokens])

        model = GeneratingLM()
        policy = HFPolicy(model, Tokenizer(), {"rollout_temperature": 1.0, "max_new_tokens_per_decision": 2})
        for tokens, truncated in (([3, 4], True), ([3, 6], False), ([6], False)):
            with self.subTest(tokens=tokens):
                model.tokens = tokens
                row = policy._generate([[1, 2]], [7])[0]
                self.assertEqual(row.truncated, truncated)
                self.assertEqual(row.token_ids, tokens)

    def test_micro_accumulation_matches_joint_token_mean_not_trajectory_mean(self):
        torch.manual_seed(8)
        model = TinyLM()
        expected_model = deepcopy(model)
        group = make_group(model)
        expected_loss, expected_entropy = 0.0, 0.0
        count = valid_training_tokens(group)
        for row in group:
            for step in row.decisions:
                value = step.completion
                lp, entropy = sequence_stats(expected_model, value.prompt_ids, value.token_ids, 1.0, "cpu")
                expected_loss = expected_loss + clipped_loss(lp, torch.tensor(value.old_logprobs), row.advantage).sum() / count
                expected_entropy += entropy.sum().item() / count
        expected_loss.backward()
        torch.optim.SGD(expected_model.parameters(), lr=0.01).step()
        policy = policy_for(model)
        updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.01), config())
        updater.begin_batch([group], "batch-0")
        report = updater.update(group)
        updater.end_batch()
        self.assertAlmostEqual(report["loss"], -0.5, places=6)
        self.assertAlmostEqual(report["entropy"], expected_entropy, places=6)
        self.assertEqual(report["training_tokens"], 4)
        self.assertEqual(policy.policy_version, 1)
        for actual, expected in zip(model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_frozen_batch_multiple_updates_and_checkpoint_cursor(self):
        torch.manual_seed(10)
        model = TinyLM()
        groups = [make_group(model, "q0"), make_group(model, "q1")]
        behavior = deepcopy(groups)
        policy = policy_for(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        updater = DAPOUpdater(policy, optimizer, config())
        updater.begin_batch(groups, "batch-0")
        updater.update(groups[0])
        restored_model = deepcopy(model)
        restored_policy = policy_for(restored_model, 1)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
        restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        restored = DAPOUpdater(restored_policy, restored_optimizer, config())
        restored.begin_batch(deepcopy(groups), "batch-0", consumed_groups=1, behavior_version=0)
        resumed_report = restored.update(groups[1])
        continuous_report = updater.update(groups[1])
        self.assertEqual(updater.consumed_groups, 2)
        self.assertEqual(restored_policy.policy_version, 2)
        self.assertEqual(groups, behavior)
        for actual, expected in zip(restored_model.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(resumed_report["loss"], continuous_report["loss"], places=6)
        with self.assertRaisesRegex(ValueError, "begin_batch"):
            updater.update(groups[0])
        updater.end_batch()
        with self.assertRaisesRegex(ValueError, "cursor"):
            updater.begin_batch(groups, "old-batch")

    def test_batch_rejects_changed_advantages_foreign_groups_and_outside_updates(self):
        torch.manual_seed(11)
        model = TinyLM()
        groups = [make_group(model, "q0"), make_group(model, "q1")]
        policy = policy_for(model)
        updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.01), config())
        updater.begin_batch(groups, "batch")
        with self.assertRaisesRegex(ValueError, "Out-of-order"):
            updater.update(groups[1])
        changed = deepcopy(groups[0])
        changed[0].advantage += 0.1
        with self.assertRaisesRegex(ValueError, "changed"):
            updater.update(changed)
        policy.policy_version = 1
        with self.assertRaisesRegex(ValueError, "outside"):
            updater.update(groups[0])
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            updater.end_batch()

    def test_truncated_rollout_is_masked_from_loss_entropy_and_denominator(self):
        torch.manual_seed(13)
        model = TinyLM()
        group = make_group(model)
        group[1].decisions[0].completion.truncated = True
        self.assertEqual(valid_training_tokens(group), 1)
        policy = policy_for(model)
        updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.01), config())
        updater.begin_batch([group], "batch")
        report = updater.update(group)
        self.assertEqual(report["masked_rollouts"], 1)
        self.assertEqual(report["training_tokens"], 1)
        self.assertAlmostEqual(report["loss"], 1.0)
        # A read-budget termination by itself must NOT trigger overlong masking.
        group[1].decisions[0].completion.truncated = False
        group[1].termination_reason = "max_read_calls"
        self.assertEqual(valid_training_tokens(group), 4)

    def test_zero_signal_and_all_truncated_are_rejected_before_step(self):
        model = TinyLM()
        group = make_group(model)
        policy = policy_for(model)
        updater = DAPOUpdater(policy, torch.optim.AdamW(model.parameters()), config())
        for row in group:
            row.advantage = 0.0
        with self.assertRaisesRegex(ValueError, "advantage signal"):
            updater.begin_batch([group], "zero")
        for row in group:
            row.decisions[0].completion.truncated = True
        with self.assertRaisesRegex(ValueError, "no unmasked training tokens"):
            updater.begin_batch([group], "truncated")
        self.assertEqual(policy.policy_version, 0)


if __name__ == "__main__":
    unittest.main()
