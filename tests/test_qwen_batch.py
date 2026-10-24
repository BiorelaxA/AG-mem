"""CPU integration check using tiny random Qwen3.5 weights, without downloads."""
import unittest
import torch
try:
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5Config = Qwen3_5ForConditionalGeneration = None
from agmem.policy import HFPolicy, sequence_logprobs

class Tokenizer:
    pad_token_id = 0
    eos_token_id = 63
    def apply_chat_template(self, messages, **kwargs):
        return list(messages)
    def decode(self, tokens, **kwargs):
        return str(tokens)


@unittest.skipIf(Qwen3_5ForConditionalGeneration is None, "Installed Transformers lacks Qwen3.5; use the agmem environment")
class QwenBatchIntegrationTests(unittest.TestCase):
    def test_hybrid_attention_padding_cache_sampling_and_backward(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            torch.manual_seed(21)
            config = Qwen3_5Config(
                text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                 linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
                                 linear_num_value_heads=4, layer_types=['linear_attention', 'full_attention'],
                                 max_position_embeddings=128, pad_token_id=0, eos_token_id=63),
                vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                                   out_hidden_size=32, patch_size=2, temporal_patch_size=2, spatial_merge_size=2),
                image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
            model = Qwen3_5ForConditionalGeneration(config).eval()
            policy = HFPolicy(model, Tokenizer(), dict(rollout_temperature=1.0, max_context_tokens=128,
                                                     max_new_tokens_per_decision=4, rollout_use_cache=True))
            prompts = [[1, 2, 3], [1, 4, 5, 6, 7], [1, 8]]
            policy.temperature = 0
            serial = [policy.complete(p, 7).token_ids for p in prompts]
            batch = [r.token_ids for r in policy.complete_batch(prompts, [7, 8, 9])]
            assert serial == batch, (serial, batch)
            policy.temperature = 1
            rows = policy.complete_batch(prompts, [7, 8, 9])
            assert rows == policy.complete_batch(prompts, [7, 8, 9])
            for row in rows:
                assert len(row.token_ids) == len(row.old_logprobs) > 0
                teacher = sequence_logprobs(model, row.prompt_ids, row.token_ids, 1, policy.device)
                assert torch.isfinite(teacher).all()
                teacher.sum().backward()
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
