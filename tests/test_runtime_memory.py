from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from agmem.core import Completion, Decision, Trajectory
from agmem.grpo import CPUAdamW, GRPOUpdater
from agmem.policy import sequence_logprobs
from agmem.runtime_memory import (activation_offload_context, configure_cuda_allocator,
                                  memory_settings, use_activation_offload)
from tests.test_grpo import TinyLM


class MemorySettingsTests(unittest.TestCase):
    def test_allocator_default_and_explicit_overrides(self):
        empty = {}
        self.assertEqual(configure_cuda_allocator(empty), "expandable_segments:True")
        self.assertEqual(empty, {"PYTORCH_ALLOC_CONF": "expandable_segments:True"})
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            env = {key: "expandable_segments:False"}
            self.assertEqual(configure_cuda_allocator(env), "expandable_segments:False")
            self.assertEqual(env, {key: "expandable_segments:False"})

    def test_threshold_disable_and_cpu_bypass(self):
        settings = memory_settings({})
        self.assertFalse(use_activation_offload(16383, "cuda:0", settings))
        self.assertTrue(use_activation_offload(16384, "cuda:0", settings))
        self.assertFalse(use_activation_offload(40000, "cpu", settings))
        disabled = memory_settings({"AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS": "0"})
        self.assertFalse(use_activation_offload(40000, "cuda:0", disabled))
        with patch("torch.autograd.graph.save_on_cpu", side_effect=AssertionError("Unexpected offload")):
            for length, device, config in [(16383, "cuda:0", settings), (40000, "cpu", settings),
                                            (40000, "cuda:0", disabled)]:
                with activation_offload_context(length, device, config):
                    pass
        for value in ("-1", "true", "1.5"):
            with self.assertRaises(ValueError):
                memory_settings({"AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS": value})


class CheckpointLM(TinyLM):
    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
        hidden = self.embedding(input_ids)
        hidden = checkpoint(lambda x: x.cumsum(1).tanh(), hidden, use_reentrant=False)
        logits = self.head(hidden)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


class OffloadUpdateTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous)

    def check_update(self, device):
        torch.manual_seed(17)
        plain = CheckpointLM().to(device)
        offloaded = deepcopy(plain)
        reference = deepcopy(plain).requires_grad_(False)
        group = []
        for i, advantage in enumerate([-1., 1.] * 4):
            decisions = []
            for prompt, tokens in [([1, 2, 3], [4, 5]), ([1, 2, 3, 4, 5], [6])]:
                old = sequence_logprobs(plain, prompt, tokens, 1, device).detach().tolist()
                decisions.append(Decision([], Completion("fixture", prompt, tokens, old)))
            group.append(Trajectory("query", i, "memory", decisions=decisions, advantage=advantage))
        config = {"clip_ratio": .2, "kl_coefficient": .01, "max_grad_norm": 1.0}
        disabled = memory_settings({"AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS": "0"})
        enabled = memory_settings({"AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS": "5"})
        # Resume both optimizers from the same existing-format state.
        optimizers = [CPUAdamW(m, .01) for m in (plain, offloaded)]
        optimizers[1].load_state_dict(deepcopy(optimizers[0].state_dict()))
        policies = [SimpleNamespace(model=m, device=torch.device(device), temperature=1., policy_version=0)
                    for m in (plain, offloaded)]
        baseline = GRPOUpdater(policies[0], optimizers[0], config, deepcopy(reference), disabled).update(group)
        saved_devices = []
        real_context = torch.autograd.graph.save_on_cpu

        class TrackingContext(real_context):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                pack = self.pack_hook
                def capture(tensor):
                    result = pack(tensor)
                    saved_devices.append((tensor.device.type, result[1].device.type))
                    return result
                self.pack_hook = capture

        def cpu_exercise(length, actual_device, settings):
            # CPU test exercises real hooks and real gradients without a GPU;
            # CUDA test below uses the production device/threshold selection.
            return activation_offload_context(length, "cuda:0", settings)

        with patch("torch.autograd.graph.save_on_cpu", TrackingContext):
            if device == "cpu":
                with patch("agmem.grpo.activation_offload_context", side_effect=cpu_exercise):
                    result = GRPOUpdater(policies[1], optimizers[1], config, deepcopy(reference), enabled).update(group)
            else:
                result = GRPOUpdater(policies[1], optimizers[1], config, deepcopy(reference), enabled).update(group)
        self.assertTrue(saved_devices)
        self.assertTrue(all(destination == "cpu" for _, destination in saved_devices))
        if device != "cpu":
            self.assertTrue(any(source == "cuda" for source, _ in saved_devices))
            self.assertEqual(result["activation_offload_decisions"], 8)
        for key in ("loss", "sampled_kl", "grad_norm"):
            self.assertAlmostEqual(result[key], baseline[key], places=5)
        for a, b in zip(plain.parameters(), offloaded.parameters()):
            torch.testing.assert_close(a, b)
        for key in ("master",):
            for a, b in zip(optimizers[0].state_dict()[key], optimizers[1].state_dict()[key]):
                torch.testing.assert_close(a, b)
        left, right = [o.state_dict()["optimizer"]["state"] for o in optimizers]
        for index in left:
            for key in left[index]:
                torch.testing.assert_close(left[index][key], right[index][key])

    def test_checkpointed_group_gradients_optimizer_and_resume_on_cpu(self):
        self.check_update("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "Requires an allocated CUDA GPU")
    def test_real_cuda_saved_tensors_and_update_equivalence(self):
        self.check_update("cuda:0")


if __name__ == "__main__":
    unittest.main()
