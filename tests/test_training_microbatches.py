"""CPU-only equivalence tests for padded DAPO forwards and complete-group minis."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from agmem.dapo import DAPOUpdater, _micro_batches, clipped_loss, valid_training_tokens
from agmem.policy import sequence_stats, sequence_stats_batch
from test_dapo import TinyLM, config, make_group, policy_for


class PositionLM(TinyLM):
    """A causal model whose outputs detect incorrect masks and position offsets."""
    def __init__(self):
        super().__init__()
        self.positions = torch.nn.Embedding(128, 4)
        self.inputs = []

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        self.forward_count += 1
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1])[None, :]
        self.inputs.append((input_ids.detach().clone(), attention_mask.detach().clone(),
                            position_ids.detach().clone()))
        embedded = self.embedding(input_ids) + self.positions(position_ids)
        embedded = embedded * attention_mask.unsqueeze(-1)
        logits = self.head(embedded.cumsum(dim=1))
        keep = kwargs.get("logits_to_keep", 0)
        return SimpleNamespace(logits=logits[:, -keep:] if keep else logits)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr=lr)
        self.steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return super().step(*args, **kwargs)


class TrainingMicrobatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_parameters_and_gradients_equal(self, actual, expected, **kwargs):
        for (name, parameter), (expected_name, reference) in zip(actual.named_parameters(), expected.named_parameters()):
            self.assertEqual(name, expected_name)
            torch.testing.assert_close(parameter, reference, **kwargs)
            if parameter.grad is None or reference.grad is None:
                self.assertIs(parameter.grad, reference.grad)
            else:
                torch.testing.assert_close(parameter.grad, reference.grad, **kwargs)

    def test_left_padding_positions_completion_masks_and_gradient_equivalence(self):
        torch.manual_seed(18)
        model = PositionLM()
        serial = deepcopy(model)
        prompts = [[1, 2], [1, 3, 4, 5], [2]]
        completions = [[3], [4, 5], [3, 4, 6]]
        stats = sequence_stats_batch(model, prompts, completions, 0.8, "cpu", pad_token_id=6)
        refs = [sequence_stats(serial, p, c, 0.8, "cpu") for p, c in zip(prompts, completions)]
        self.assertEqual(model.forward_count, 1)
        ids, mask, positions = model.inputs[0]
        torch.testing.assert_close(mask, torch.tensor([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]))
        torch.testing.assert_close(positions, torch.tensor([[0, 0, 0, 0, 1], [0, 1, 2, 3, 4], [0, 0, 0, 1, 2]]))
        self.assertTrue((ids[mask == 0] == 6).all())
        for (lp, entropy), (ref_lp, ref_entropy), tokens in zip(stats, refs, completions):
            self.assertEqual(len(lp), len(tokens))
            self.assertFalse(entropy.requires_grad)
            torch.testing.assert_close(lp, ref_lp)
            torch.testing.assert_close(entropy, ref_entropy)
        sum((i + 1) * lp.sum() for i, (lp, _) in enumerate(stats)).backward()
        sum((i + 1) * lp.sum() for i, (lp, _) in enumerate(refs)).backward()
        self.assert_parameters_and_gradients_equal(model, serial)
        # Altering another sample never changes the first sample's predictions.
        changed = sequence_stats_batch(model, [prompts[0], [6] * 7, prompts[2]], completions, 0.8, "cpu")
        torch.testing.assert_close(changed[0][0], stats[0][0])
        torch.testing.assert_close(changed[2][0], stats[2][0])

    def test_micro_two_matches_serial_gradients_and_uses_fewer_forwards(self):
        torch.manual_seed(19)
        original = PositionLM()
        group = make_group(original, lengths=(3, 5))
        reports, models = [], []
        for micro in (1, 2):
            model = deepcopy(original)
            start = model.forward_count
            optimizer = CountingSGD(model.parameters())
            updater = DAPOUpdater(policy_for(model), optimizer, config(micro_batch_rollouts=micro))
            updater.begin_batch([group], "micro-equivalence")
            report = updater.update(group)
            self.assertEqual(optimizer.steps, 1)
            self.assertEqual(model.forward_count - start, report["micro_batches"])
            reports.append(report)
            models.append(model)
        self.assert_parameters_and_gradients_equal(models[1], models[0])
        self.assertLess(reports[1]["micro_batches"], reports[0]["micro_batches"])
        for metric in ("loss", "entropy", "sampled_surprisal", "grad_norm", "training_tokens"):
            self.assertAlmostEqual(reports[0][metric], reports[1][metric], places=5)

    def test_two_group_mini_is_one_joint_token_objective_and_one_step(self):
        torch.manual_seed(20)
        model = PositionLM()
        expected = deepcopy(model)
        groups = [make_group(model, "q0", lengths=(1, 3)), make_group(model, "q1", lengths=(2, 5))]
        for row in groups[1]:
            row.advantage *= 0.5  # A different fixed scale must remain local to its group.
        frozen = deepcopy(groups)
        count = sum(valid_training_tokens(group) for group in groups)
        objective = 0
        for group in groups:
            for row in group:
                for step in row.decisions:
                    value = step.completion
                    lp, _ = sequence_stats(expected, value.prompt_ids, value.token_ids, 1, "cpu")
                    objective += clipped_loss(lp, torch.tensor(value.old_logprobs), row.advantage).sum() / count
        objective.backward()
        torch.optim.SGD(expected.parameters(), lr=0.01).step()
        policy = policy_for(model)
        optimizer = CountingSGD(model.parameters())
        updater = DAPOUpdater(policy, optimizer, config(mini_batch_groups=2, micro_batch_rollouts=2))
        updater.begin_batch(groups, "mini")
        with patch("torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clip:
            report = updater.update_groups(groups)
            self.assertEqual(clip.call_count, 1)
        self.assertEqual(optimizer.steps, 1)
        self.assertEqual(policy.policy_version, 1)
        self.assertEqual(updater.consumed_groups, 2)
        self.assertEqual(report["update_groups"], 2)
        self.assertEqual(report["update_rollouts"], 4)
        self.assertEqual(report["training_tokens"], 11)
        self.assertEqual(groups, frozen)
        self.assertAlmostEqual(report["loss"], objective.item(), places=6)
        self.assert_parameters_and_gradients_equal(model, expected)
        updater.end_batch()

    def test_token_budget_counts_padding_and_long_singletons_are_preserved(self):
        model = TinyLM()
        rows = make_group(model, lengths=(3, 5))
        rows[0].decisions[1].completion.prompt_ids = [1, 2] * 12
        plan = _micro_batches(rows, 2, 6)
        seen = []
        over_budget = 0
        for block in plan:
            self.assertLessEqual(len(block), 2)
            self.assertEqual(len({i for i, _, _ in block}), len(block))
            allocated = max(len(v.prompt_ids) + len(v.token_ids) - 1 for _, _, v in block) * len(block)
            if allocated > 6:
                self.assertEqual(len(block), 1)
                over_budget += 1
            seen.extend((i, j) for i, j, _ in block)
        self.assertEqual(over_budget, 1)
        self.assertEqual(sorted(seen), [(i, j) for i, row in enumerate(rows) for j in range(len(row.decisions))])
        self.assertEqual(len(rows[0].decisions[1].completion.prompt_ids), 24)

    def test_mini_resume_cursor_counts_groups_and_only_accepts_boundaries(self):
        torch.manual_seed(22)
        model = PositionLM()
        groups = [make_group(model, f"q{i}") for i in range(4)]
        policy = policy_for(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        settings = config(mini_batch_groups=2, micro_batch_rollouts=2)
        updater = DAPOUpdater(policy, optimizer, settings)
        with self.assertRaisesRegex(ValueError, "align"):
            updater.begin_batch(groups[:3], "uneven")
        with self.assertRaisesRegex(ValueError, "align"):
            updater.begin_batch(groups, "unaligned", consumed_groups=1)
        updater.begin_batch(groups, "resume")
        with self.assertRaisesRegex(ValueError, "update_groups"):
            updater.update(groups[0])
        with self.assertRaisesRegex(ValueError, "exactly"):
            updater.update_groups(groups[:1])
        updater.update_groups(groups[:2])
        restored_model = deepcopy(model)
        restored_policy = policy_for(restored_model, version=1)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
        restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        restored = DAPOUpdater(restored_policy, restored_optimizer, settings)
        restored.begin_batch(deepcopy(groups), "resume", consumed_groups=2, behavior_version=0)
        restored.update_groups(groups[2:])
        updater.update_groups(groups[2:])
        self.assertEqual(restored_policy.policy_version, 2)
        self.assertEqual(restored.consumed_groups, 4)
        self.assert_parameters_and_gradients_equal(restored_model, model)

    def test_legacy_micro_one_never_uses_padded_forward(self):
        model = TinyLM()
        group = make_group(model)
        updater = DAPOUpdater(policy_for(model), CountingSGD(model.parameters()), config())
        updater.begin_batch([group], "legacy")
        with patch("agmem.dapo.sequence_stats_batch", side_effect=AssertionError("Legacy path changed")):
            report = updater.update(group)
        self.assertEqual(report["micro_batches"], 3)
        self.assertEqual(report["micro_batch_padding_tokens"], 0)
        self.assertEqual(report["max_micro_batch_tokens"], 16384)

    def test_invalid_batch_sizes_rejected(self):
        model = TinyLM()
        for key in ("micro_batch_rollouts", "mini_batch_groups", "max_micro_batch_tokens"):
            for value in (0, -1, 1.5, True):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    DAPOUpdater(policy_for(model), CountingSGD(model.parameters()), config(**{key: value}))

    def test_tiny_qwen_hybrid_batched_logprobs_and_gradients(self):
        try:
            from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        except ImportError:
            self.skipTest("Installed Transformers lacks Qwen3.5")
        torch.manual_seed(23)
        cfg = Qwen3_5Config(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                             linear_num_value_heads=4, layer_types=["linear_attention", "full_attention"],
                             max_position_embeddings=128, pad_token_id=0, eos_token_id=63),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                               out_hidden_size=32, patch_size=2, temporal_patch_size=2, spatial_merge_size=2),
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        model = Qwen3_5ForConditionalGeneration(cfg).eval()
        serial = deepcopy(model)
        prompts, completions = [[1, 2], [1, 4, 5, 6, 7]], [[3, 4, 5], [8]]
        actual = sequence_stats_batch(model, prompts, completions, 1, "cpu")
        expected = [sequence_stats(serial, p, c, 1, "cpu") for p, c in zip(prompts, completions)]
        for (lp, entropy), (ref_lp, ref_entropy) in zip(actual, expected):
            torch.testing.assert_close(lp, ref_lp, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(entropy, ref_entropy, atol=2e-6, rtol=2e-6)
        sum(lp.sum() for lp, _ in actual).backward()
        sum(lp.sum() for lp, _ in expected).backward()
        self.assert_parameters_and_gradients_equal(model, serial, atol=3e-5, rtol=3e-4)


if __name__ == "__main__":
    unittest.main()
