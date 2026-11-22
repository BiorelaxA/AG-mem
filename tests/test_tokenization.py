from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BatchEncoding

from agmem.core import CapacityError
from agmem.pipeline import make_environment
from agmem.policy import HFPolicy, chat_token_ids
from agmem.services import DirectoryBuilder
from agmem.store import Store


class BatchDefaultTokenizer:
    """Reproduce Transformers 5's default, including real tokenizers.Encoding objects."""

    def __init__(self):
        backend = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        encoding = backend.encode("a b c")
        self.encoding = BatchEncoding({"input_ids": encoding.ids, "attention_mask": [1, 1, 1]}, encoding=encoding)

    def apply_chat_template(self, messages, return_dict=True, **kwargs):
        return self.encoding if return_dict else self.encoding["input_ids"]


class ChatTokenizationTests(unittest.TestCase):
    def test_reader_uses_integer_ids_and_counts_tokens_instead_of_dictionary_fields(self):
        tokenizer = BatchDefaultTokenizer()
        self.assertEqual(len(tokenizer.apply_chat_template([])), 2)
        self.assertEqual(type(tokenizer.apply_chat_template([])[0]).__name__, "Encoding")
        model = torch.nn.Linear(1, 1)
        policy = HFPolicy(model, tokenizer, {"rollout_temperature": 1.0, "max_context_tokens": 4,
                                            "max_new_tokens_per_decision": 2})
        self.assertEqual(policy.encode([]), [1, 2, 3])
        self.assertEqual(torch.tensor([policy.encode([])], dtype=torch.long).shape, (1, 3))
        with self.assertRaisesRegex(CapacityError, "3 prompt tokens"):
            policy.check_capacity([])
        policy.config["max_context_tokens"] = 5
        policy.check_capacity([])

    def test_task_agent_capacity_counts_real_token_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary))
            identity = {"model": "fixture", "base_url": "http://fixture/v1", "model_revision": "fixture-v1"}
            config = {"service": identity, "models": {"task_path": "fixture"},
                      "writer": {"lookback_sessions": 3, "content_max_tokens": 300},
                      "retrieval": {"max_ids_per_call": 10, "max_read_calls": 8},
                      "task_agent": {"max_context_tokens": 4, "max_output_tokens": 2}}
            version = DirectoryBuilder(store, SimpleNamespace(identity=identity), config["writer"], None).version
            store.set_metadata("directory_version", version)
            policy = SimpleNamespace(check_capacity=Mock())
            try:
                with patch("agmem.policy.load_tokenizer", return_value=BatchDefaultTokenizer()), \
                        patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "fixture-v1"}):
                    env = make_environment(config, store, policy)
                    with self.assertRaisesRegex(CapacityError, "Task-agent context would overflow"):
                        env.capacity([], [])
                    policy.check_capacity.assert_called_once_with([])
                    config["task_agent"]["max_context_tokens"] = 5
                    env.capacity([], [])
            finally:
                store.close()

    def test_unexpected_tokenization_shape_is_rejected_before_model_execution(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: {"input_ids": [1, 2, 3]})
        with self.assertRaisesRegex(TypeError, "flat list of integer input_ids"):
            chat_token_ids(tokenizer, [])


if __name__ == "__main__":
    unittest.main()
