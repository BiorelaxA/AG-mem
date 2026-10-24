"""Tiny local Qwen integration tests; CUDA coverage runs when two GPUs exist."""
import copy
import tempfile
import unittest
from unittest.mock import patch

import torch

from agmem.model_parallel import place_reader_layers, reader_devices
from agmem.policy import HFPolicy, sequence_logprobs
from agmem.grpo import CPUAdamW
from tests.test_qwen_batch import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Tokenizer


class DeviceSelectionTests(unittest.TestCase):
    def test_reference_stays_cpu_and_reader_requires_two_gpus(self):
        env = {"AGMEM_READER_MODEL_PARALLEL_SIZE": "2"}
        self.assertEqual(reader_devices("cpu", env), [torch.device("cpu")])
        with patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(reader_devices("cuda:0", env), [torch.device("cuda:0"), torch.device("cuda:1")])
        with patch("torch.cuda.device_count", return_value=1):
            with self.assertRaises(ValueError):
                reader_devices("cuda:0", env)
        with self.assertRaises(ValueError):
            reader_devices("cuda:0", {"AGMEM_READER_MODEL_PARALLEL_SIZE": "3"})


@unittest.skipIf(Qwen3_5ForConditionalGeneration is None, "Requires the agmem Transformers environment")
class LayerPlacementTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(37)
        config = Qwen3_5Config(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                             linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                             linear_num_value_heads=4, layer_types=['linear_attention', 'full_attention'] * 2,
                             max_position_embeddings=128, pad_token_id=0, eos_token_id=63),
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                               out_hidden_size=32, patch_size=2, temporal_patch_size=2, spatial_merge_size=2),
            image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
        self.model = Qwen3_5ForConditionalGeneration(config).eval()

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def check_placement(self, devices):
        plain = self.model.to(devices[0])
        split = place_reader_layers(copy.deepcopy(plain).cpu(), devices)
        self.assertEqual(list(plain.state_dict()), list(split.state_dict()))
        for index, layer in enumerate(split.model.language_model.layers):
            actual = next(layer.parameters()).device
            expected = torch.device(devices[index // 2])
            self.assertEqual(actual, expected)
        cfg = dict(rollout_temperature=0.0, max_context_tokens=128,
                   max_new_tokens_per_decision=3, rollout_use_cache=True)
        prompts = [[1, 2, 3], [1, 4, 5, 6, 7]]
        policies = [HFPolicy(m, Tokenizer(), cfg) for m in (plain, split)]
        results = [p.complete_batch(prompts, [7, 8]) for p in policies]
        self.assertEqual([r.token_ids for r in results[0]], [r.token_ids for r in results[1]])
        # Check checkpoint recomputation and gradients through both boundaries.
        for model in (plain, split):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            for row in results[0]:
                sequence_logprobs(model, row.prompt_ids, row.token_ids, 1, devices[0]).sum().backward()
        for (name, a), (_, b) in zip(plain.named_parameters(), split.named_parameters()):
            if a.grad is not None:
                self.assertIsNotNone(b.grad, name)
                torch.testing.assert_close(a.grad.cpu(), b.grad.cpu(), rtol=2e-4, atol=2e-5, msg=name)
        # Existing CPU optimizer state can resume into a partitioned model.
        old_optimizer = CPUAdamW(plain, 1e-4)
        old_optimizer.step()
        new_optimizer = CPUAdamW(split, 1e-4)
        new_optimizer.load_state_dict(old_optimizer.state_dict())
        for a, b in zip(plain.parameters(), split.parameters()):
            torch.testing.assert_close(a.cpu(), b.cpu())
        with tempfile.TemporaryDirectory() as root:
            split.save_pretrained(root, safe_serialization=True)
            restored = Qwen3_5ForConditionalGeneration.from_pretrained(root)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, split.state_dict()[name].cpu())

    def test_cpu_hooks_cache_backward_optimizer_and_checkpoint(self):
        self.check_placement(["cpu", "cpu"])

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two allocated CUDA GPUs")
    def test_two_gpu_cache_backward_optimizer_and_checkpoint(self):
        self.check_placement(["cuda:0", "cuda:1"])


if __name__ == "__main__":
    unittest.main()
