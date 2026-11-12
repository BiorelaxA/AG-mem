"""Process credit must remain separate from terminal rewards and DAPO filtering."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import torch

from agmem.core import Completion, Decision, Trajectory, trajectory_from_dict
from agmem.dapo import DAPOUpdater, clipped_loss, valid_training_tokens
from agmem.dapo_sampling import assess_group
from agmem.policy import sequence_stats
from agmem.process_advantages import (ATTRIBUTION_PROTOCOL, assign_process_advantages,
                                      process_reward_settings)
from tests.test_dapo import TinyLM, config, make_group, policy_for
from tests.test_dapo_all_correct_format import config_for, format_group
from tests.test_dapo_sampling import examples


def settings(**values):
    return {"process_reward": {"enabled": True, **values}}


def attribute(row, gains, weights=None):
    weights = [1.0] * len(gains) if weights is None else weights
    entries = []
    for index, decision in enumerate(row.decisions):
        is_read = index < len(gains)
        decision.action = {"stop": not is_read, "ids": [f"id-{index}"] if is_read else []}
        gain = gains[index] if is_read else 0.0
        weight = weights[index] if is_read else 1.0
        entries.append({"is_read": is_read, "g": gain, "w": weight,
                        "reward": weight * gain if gain >= 0 else gain / weight})
    row.process_attribution = {"protocol": ATTRIBUTION_PROTOCOL, "per_decision": entries}
    return row


def trajectory(gains, *, advantage=0.0, sample_id="q", weights=None, stop=True):
    row = Trajectory(sample_id, 0, "memory", advantage=advantage, reward=0.79,
                     decisions=[Decision([], Completion("action", [1], [2], [-1.0]))
                                for _ in range(len(gains) + int(stop))])
    return attribute(row, gains, weights)


class ProcessAdvantageTests(unittest.TestCase):
    def test_standardization_matches_diagram_and_stop_only_gets_terminal_advantage(self):
        group = [trajectory([0.0, 1.0], advantage=-1.0), trajectory([0.0, 1.0], advantage=1.0)]
        terminal = [(row.reward, row.advantage) for row in group]
        assign_process_advantages(group, settings(gamma=0.9, alpha=0.5))
        scale = 0.5 / (0.5 + 1e-6)
        for row in group:
            first, second, stop = row.process_attribution["per_decision"]
            self.assertAlmostEqual(first["normalized_reward"], -scale)
            self.assertAlmostEqual(first["process_return"], -0.1 * scale)
            self.assertAlmostEqual(first["advantage"], row.advantage - 0.05 * scale)
            self.assertAlmostEqual(second["process_return"], scale)
            self.assertEqual(stop["normalized_reward"], 0.0)
            self.assertEqual(stop["process_return"], 0.0)
            self.assertEqual(stop["advantage"], row.advantage)
            self.assertEqual(row.process_attribution["advantage_normalization"]["read_decisions"], 4)
        self.assertEqual(terminal, [(row.reward, row.advantage) for row in group])

    def test_optional_rms_preserves_zero_and_rewards_bridge_through_future_return(self):
        group = [trajectory([0.0, 1.0]), trajectory([0.0, 1.0])]
        assign_process_advantages(group, settings(normalization="rms", gamma=0.9))
        positive = 1.0 / (math.sqrt(0.5) + 1e-6)
        first, second, _ = group[0].process_attribution["per_decision"]
        self.assertEqual(first["normalized_reward"], 0.0)
        self.assertAlmostEqual(first["process_return"], 0.9 * positive)
        self.assertAlmostEqual(second["process_return"], positive)

    def test_zero_gain_receives_no_extra_weight_and_negative_gain_uses_division(self):
        group = [trajectory([0.0, -2.0], weights=[10.0, 2.0]),
                 trajectory([0.0, -2.0], weights=[0.1, 2.0])]
        assign_process_advantages(group, settings(normalization="rms"))
        left, right = [row.process_attribution["per_decision"] for row in group]
        self.assertEqual(left[0]["reward"], 0.0)
        self.assertEqual(left[1]["reward"], -1.0)
        self.assertEqual(left[0]["advantage"], right[0]["advantage"])

    def test_normalization_includes_masked_rows_but_excludes_stop(self):
        group = [trajectory([1.0]), trajectory([3.0])]
        group[1].decisions[0].completion.truncated = True
        assign_process_advantages(group, settings())
        stats = group[0].process_attribution["advantage_normalization"]
        self.assertEqual(stats["read_decisions"], 2)
        self.assertEqual(stats["reward_mean"], 2.0)
        self.assertEqual(stats["reward_scale"], 1.0)
        self.assertLess(group[0].process_attribution["per_decision"][0]["advantage"], 0)

    def test_constant_decimal_and_no_read_groups_have_zero_process_term(self):
        for group in ([trajectory([0.1] * 8), trajectory([0.1] * 8)],
                      [trajectory([]), trajectory([])]):
            assign_process_advantages(group, settings())
            for row in group:
                for entry in row.process_attribution["per_decision"]:
                    self.assertEqual(entry["normalized_reward"], 0.0)
                    self.assertEqual(entry["process_return"], 0.0)

    def test_resume_roundtrip_preserves_values_and_refuses_changed_or_partial_credit(self):
        original = [trajectory([0.0, 1.0], advantage=-1.0), trajectory([-1.0], advantage=1.0)]
        cfg = settings(alpha=0.3)
        assign_process_advantages(original, cfg)
        restored = [trajectory_from_dict(row.to_dict()) for row in original]
        assign_process_advantages(restored, cfg)
        self.assertEqual(restored, original)
        changed = deepcopy(restored)
        changed[0].process_attribution["per_decision"][0]["advantage"] += 0.1
        with self.assertRaisesRegex(ValueError, "Saved process advantages"):
            assign_process_advantages(changed, cfg)
        with self.assertRaisesRegex(ValueError, "normalization"):
            assign_process_advantages(deepcopy(restored), settings(alpha=0.4))
        partial = deepcopy(restored)
        del partial[0].process_attribution["per_decision"][0]["process_return"]
        with self.assertRaisesRegex(ValueError, "Saved process advantages"):
            assign_process_advantages(partial, cfg)

    def test_wrong_data_is_rejected_before_credit_assignment(self):
        cases = []
        missing = trajectory([1.0])
        missing.process_attribution = {}
        cases.append(missing)
        bad_reward = trajectory([1.0])
        bad_reward.process_attribution["per_decision"][0]["reward"] = 2.0
        cases.append(bad_reward)
        wrong_marker = trajectory([1.0])
        wrong_marker.process_attribution["per_decision"][0]["is_read"] = False
        cases.append(wrong_marker)
        bad_stop = trajectory([1.0])
        bad_stop.process_attribution["per_decision"][-1].update(g=1.0, reward=1.0)
        cases.append(bad_stop)
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                assign_process_advantages([row], settings())
        with self.assertRaisesRegex(ValueError, "same-query"):
            assign_process_advantages([trajectory([1.0]), trajectory([1.0], sample_id="different")], settings())

    def test_strict_settings_and_disabled_legacy_path(self):
        for key, value in (("enabled", 1), ("alpha", -1), ("alpha", True), ("gamma", 1.1),
                           ("gamma", float("nan")), ("epsilon", 0), ("normalization", [])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                process_reward_settings(settings(**{key: value}))
        legacy = [Trajectory("q", 0, "m")]
        before = deepcopy(legacy)
        assign_process_advantages(legacy, {})
        self.assertEqual(before, legacy)
        self.assertNotIn("process_attribution", legacy[0].to_dict())
        self.assertEqual(trajectory_from_dict(legacy[0].to_dict()), legacy[0])

    def test_process_scores_do_not_admit_all_wrong_or_clean_all_correct_groups(self):
        example = examples(1)[0]
        cfg = config_for()
        cfg["training"].update(settings())
        for judges, counts, accepted in (((0, 0), (0, 1), False),
                                        ((1, 1), (0, 0), False),
                                        ((1, 1), (0, 1), True),
                                        ((0, 1), (0, 0), True)):
            group = format_group(example, judges=judges, counts=counts)
            for index, row in enumerate(group):
                attribute(row, [100.0 if index else -100.0])
            with self.subTest(judges=judges, counts=counts):
                self.assertEqual(assess_group(group, cfg)["accepted"], accepted)


class ProcessDAPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actor_uses_step_advantages_with_joint_token_mean_in_serial_and_batched_modes(self):
        for width in (1, 2):
            with self.subTest(width=width):
                torch.manual_seed(41)
                model = TinyLM()
                expected_model = deepcopy(model)
                group = make_group(model, lengths=(5, 5))
                gains = [[0.0, 1.0], [-0.5, 0.25]]
                for row, values in zip(group, gains):
                    attribute(row, values)
                cfg = config(micro_batch_rollouts=width,
                             **settings(alpha=0.7, gamma=0.9, normalization="rms"))
                policy = policy_for(model)
                updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.01), cfg)
                updater.begin_batch([group], "process")
                expected_loss = 0.0
                count = valid_training_tokens(group)
                scale = math.sqrt((0.0 + 1.0 + 0.25 + 0.0625) / 4) + 1e-6
                for row, values in zip(group, gains):
                    expected_advantages = [row.advantage + 0.7 * (values[0] + 0.9 * values[1]) / scale,
                                           row.advantage + 0.7 * values[1] / scale,
                                           row.advantage]
                    for decision, advantage in zip(row.decisions, expected_advantages):
                        value = decision.completion
                        lp, _ = sequence_stats(expected_model, value.prompt_ids, value.token_ids, 1.0, "cpu")
                        expected_loss = expected_loss + clipped_loss(lp, torch.tensor(value.old_logprobs), advantage).sum() / count
                expected_loss.backward()
                torch.optim.SGD(expected_model.parameters(), lr=0.01).step()
                report = updater.update(group)
                self.assertAlmostEqual(report["loss"], expected_loss.item(), places=6)
                self.assertEqual(report["process_read_steps"], 4)
                for actual, expected in zip(model.parameters(), expected_model.parameters()):
                    torch.testing.assert_close(actual, expected)

    def test_query_group_statistics_do_not_mix_and_frozen_scores_are_bound(self):
        torch.manual_seed(17)
        model = TinyLM()
        groups = [make_group(model, "q0", lengths=(3, 3)), make_group(model, "q1", lengths=(3, 3))]
        for group, gains in zip(groups, ([1.0, 3.0], [100.0, 300.0])):
            for row, gain in zip(group, gains):
                attribute(row, [gain])
        cfg = config(mini_batch_groups=2, **settings())
        updater = DAPOUpdater(policy_for(model), torch.optim.SGD(model.parameters(), lr=0.01), cfg)
        updater.begin_batch(groups, "process")
        self.assertEqual(groups[0][0].process_attribution["advantage_normalization"]["reward_mean"], 2.0)
        self.assertEqual(groups[1][0].process_attribution["advantage_normalization"]["reward_mean"], 200.0)
        changed = deepcopy(groups)
        changed[0][0].process_attribution["per_decision"][0]["w"] = 2.0
        with self.assertRaisesRegex(ValueError, "changed"):
            updater.update_groups(changed)

    def test_process_resume_keeps_original_behavior_and_computed_advantages(self):
        torch.manual_seed(7)
        model = TinyLM()
        groups = [make_group(model, "q0", lengths=(3, 3)), make_group(model, "q1", lengths=(3, 3))]
        for group in groups:
            for row, gain in zip(group, [0.0, 1.0]):
                row.advantage = 0.0
                attribute(row, [gain])
        cfg = config(**settings(normalization="rms"))
        policy = policy_for(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        updater = DAPOUpdater(policy, optimizer, cfg)
        updater.begin_batch(groups, "process")
        frozen = deepcopy(groups)
        first = updater.update(groups[0])
        self.assertFalse(first["optimizer_step_skipped"])
        self.assertGreater(first["grad_norm"], 0.0)
        self.assertEqual(first["process_zero_terminal_groups"], 1)
        restored_model = deepcopy(model)
        restored_policy = policy_for(restored_model, version=1)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
        restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
        restored_groups = [[trajectory_from_dict(row.to_dict()) for row in group] for group in groups]
        restored = DAPOUpdater(restored_policy, restored_optimizer, cfg)
        restored.begin_batch(restored_groups, "process", consumed_groups=1, behavior_version=0)
        restored.update(restored_groups[1])
        updater.update(groups[1])
        self.assertEqual(restored_groups, frozen)
        for actual, expected in zip(restored_model.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_zero_group_stays_in_mixed_mini_and_process_only_gradient_uses_all_tokens(self):
        torch.manual_seed(19)
        model = TinyLM()
        expected_model = deepcopy(model)
        groups = [make_group(model, "zero", lengths=(3, 3)),
                  make_group(model, "informative", lengths=(3, 3))]
        for group, gains in zip(groups, ([0.25, 0.25], [-1.0, 1.0])):
            for row, gain in zip(group, gains):
                row.advantage = 0.0
                attribute(row, [gain])
        cfg = config(mini_batch_groups=2, **settings())
        policy = policy_for(model)
        updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.01), cfg)
        updater.begin_batch(groups, "mixed-process")
        token_count = sum(valid_training_tokens(group) for group in groups)
        expected_loss = 0.0
        for row in groups[1]:
            for step, entry in zip(row.decisions, row.process_attribution["per_decision"]):
                value = step.completion
                logprobs, _ = sequence_stats(expected_model, value.prompt_ids, value.token_ids, 1.0, "cpu")
                expected_loss = expected_loss + clipped_loss(
                    logprobs, torch.tensor(value.old_logprobs), entry["advantage"]).sum() / token_count
        expected_loss.backward()
        torch.optim.SGD(expected_model.parameters(), lr=0.01).step()
        report = updater.update_groups(groups)
        self.assertFalse(report["optimizer_step_skipped"])
        self.assertGreater(report["grad_norm"], 0.0)
        self.assertEqual(report["training_tokens"], token_count)
        self.assertEqual(report["update_groups"], 2)
        self.assertEqual(report["process_zero_terminal_groups"], 2)
        self.assertEqual(report["process_zero_combined_advantage_groups"], 1)
        self.assertEqual(policy.policy_version, 1)
        self.assertEqual(updater.consumed_groups, 2)
        for actual, expected in zip(model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_zero_process_mini_does_not_apply_adam_momentum_or_weight_decay(self):
        for mask_nonzero in (False, True):
            with self.subTest(mask_nonzero=mask_nonzero):
                torch.manual_seed(23)
                model = TinyLM()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
                # Seed real optimizer momentum: stepping a zero gradient here
                # would still change the parameters and optimizer counters.
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                group = make_group(model, lengths=(3, 3))
                for index, row in enumerate(group):
                    row.advantage = 0.0
                    attribute(row, [1.0 if mask_nonzero and index else 0.0])
                if mask_nonzero:
                    group[1].decisions[0].completion.truncated = True
                cfg = config(**settings(normalization="rms"))
                policy = policy_for(model)
                updater = DAPOUpdater(policy, optimizer, cfg)
                updater.begin_batch([group], "zero-process")
                parameters = [parameter.detach().clone() for parameter in model.parameters()]
                gradients = [parameter.grad.clone() for parameter in model.parameters()]
                state = deepcopy(optimizer.state_dict())
                forwards = model.forward_count
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    report = updater.update(group)
                step.assert_not_called()
                self.assertTrue(report["optimizer_step_skipped"])
                self.assertEqual(report["skip_reason"], "zero_combined_advantage")
                self.assertEqual(report["training_tokens"], 0)
                self.assertEqual(report["eligible_training_tokens"], valid_training_tokens(group))
                self.assertEqual(report["process_zero_combined_advantage_groups"], 1)
                self.assertEqual(policy.policy_version, 0)
                self.assertEqual(updater.consumed_groups, 1)
                self.assertEqual(model.forward_count, forwards)
                for parameter, expected, gradient in zip(model.parameters(), parameters, gradients):
                    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
                    torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
                current = optimizer.state_dict()
                self.assertEqual(current["param_groups"], state["param_groups"])
                for key, values in state["state"].items():
                    for name, value in values.items():
                        torch.testing.assert_close(current["state"][key][name], value, rtol=0, atol=0)
                updater.end_batch()
                self.assertEqual(updater.consumed_groups, 0)

    def test_zero_process_mini_requires_single_mini_behavior_batch_and_legacy_guard_remains(self):
        model = TinyLM()
        groups = [make_group(model, "q0"), make_group(model, "q1")]
        for group in groups:
            for row in group:
                row.advantage = 0.0
                attribute(row, [0.0])
        policy = policy_for(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        updater = DAPOUpdater(policy, optimizer, config(**settings()))
        updater.begin_batch(groups, "wide-zero-batch")
        with self.assertRaisesRegex(ValueError, "entire behavior batch"):
            updater.update(groups[0])
        self.assertEqual(updater.consumed_groups, 0)
        self.assertEqual(policy.policy_version, 0)
        legacy = DAPOUpdater(policy, optimizer, config())
        with self.assertRaisesRegex(ValueError, "no unmasked advantage signal"):
            legacy.begin_batch([groups[0]], "legacy-zero")


if __name__ == "__main__":
    unittest.main()
