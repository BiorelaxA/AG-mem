from types import SimpleNamespace
import unittest

import torch

from agmem.core import CapacityError
from agmem.policy import HFPolicy


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 8
    def apply_chat_template(self, messages, **kwargs):
        return list(messages)
    def decode(self, tokens, **kwargs):
        return ' '.join(str(t) for t in tokens if t not in (0, 7, 8))


class BatchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=64))
        self.generation_config = SimpleNamespace(eos_token_id=[7, 8])
        self.calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append((input_ids.clone(), attention_mask.clone(), kwargs))
        generated, scores = [], []
        for row in input_ids:
            # Different completion lengths, EOS alternatives, and no EOS at cap.
            last = row[-1].item()
            generated.append([4, 7, 0] if last == 1 else [5, 6, 8] if last == 2 else [4, 5, 6])
        generated = torch.tensor(generated)
        for index in range(3):
            values = torch.zeros(len(generated), 9)
            for row in range(len(generated)):
                values[row, generated[row, index]] = row + index + .5
            scores.append(values)
        return SimpleNamespace(sequences=torch.cat([input_ids, generated], dim=1), scores=scores)


class BatchPolicyTests(unittest.TestCase):
    def policy(self):
        return HFPolicy(BatchModel(), Tokenizer(), {'rollout_temperature': 1.0,
                        'max_new_tokens_per_decision': 3, 'max_context_tokens': 64})

    def test_left_padding_eos_mask_and_actual_processed_scores(self):
        policy = self.policy()
        prompts = [[1], [3, 4, 2], [3, 3]]
        rng_before = torch.get_rng_state().clone()
        rows = policy.complete_batch(prompts, [11, 12, 13])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        ids, mask, kwargs = policy.model.calls[0]
        self.assertEqual(ids.tolist(), [[0, 0, 1], [3, 4, 2], [0, 3, 3]])
        self.assertEqual(mask.tolist(), [[0, 0, 1], [1, 1, 1], [0, 1, 1]])
        self.assertEqual([r.prompt_ids for r in rows], prompts)
        self.assertEqual([r.token_ids for r in rows], [[4, 7], [5, 6, 8], [4, 5, 6]])
        self.assertEqual([len(r.old_logprobs) for r in rows], [2, 3, 3])
        for i, row in enumerate(rows):
            expected = []
            for j, token in enumerate(row.token_ids):
                logits = torch.zeros(9)
                logits[token] = i + j + .5
                expected.append(torch.log_softmax(logits, -1)[token].item())
            self.assertEqual(row.old_logprobs, expected)
            self.assertEqual(row.sampling_batch_size, 3)
        self.assertEqual(len({r.sampling_seed for r in rows}), 1)
        repeated = policy.complete_batch(prompts, [11, 12, 13])
        self.assertEqual(rows, repeated)
        self.assertEqual(kwargs['top_k'], 0)
        self.assertEqual(kwargs['top_p'], 1.0)
        self.assertEqual(kwargs['temperature'], 1.0)

    def test_token_budget_splits_batches_without_truncating_long_single_input(self):
        policy = self.policy()
        prompts = [[1], [2], [3] * 20, [1]]
        rows = policy.complete_batch(prompts, [11, 12, 13, 14], max_batch_tokens=8)
        self.assertEqual([c[0].shape[0] for c in policy.model.calls], [2, 1, 1])
        self.assertEqual([r.prompt_ids for r in rows], prompts)
        self.assertEqual([r.sampling_batch_size for r in rows], [2, 2, 1, 1])
        self.assertEqual(rows[2].sampling_seed, 13)
        before = len(policy.model.calls)
        with self.assertRaises(CapacityError):
            policy.complete_batch([[1], [2] * 63], [1, 2])
        self.assertEqual(len(policy.model.calls), before)

    def test_single_batch_and_serial_are_equivalent_greedy_has_no_logprobs(self):
        policy = self.policy()
        self.assertEqual(policy.complete([1, 2], 17), policy.complete_batch([[1, 2]], [17])[0])
        self.assertEqual(policy.complete([1, 2], 17).sampling_seed, 17)
        policy.temperature = 0
        rows = policy.complete_batch([[1], [2]], [1, 2])
        self.assertTrue(all(row.old_logprobs == [] for row in rows))
        self.assertFalse(policy.model.calls[-1][2]['do_sample'])
        self.assertFalse(policy.model.calls[-1][2]['output_scores'])
        self.assertEqual(policy.complete_batch([], []), [])


if __name__ == '__main__':
    unittest.main()
