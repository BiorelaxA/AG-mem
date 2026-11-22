"""Freeze visible-parent READ penalties at the action level, including actor gradients."""
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from agmem.action_advantages import (action_advantage_metrics, action_advantage_settings,
                                     assign_action_advantages, effective_decision_advantage)
from agmem.core import Completion, Decision, Example, Trajectory, trajectory_from_dict
from agmem.dapo import DAPOUpdater
from agmem.hierarchical_env import HierarchicalMemoryEnvironment
from agmem.monitoring import summary_writer_class, write_train_metrics
from agmem.policy import sequence_logprobs
from agmem.traces import compact_group, save_training_group
from tests.test_dapo import config, policy_for
from tests.test_hierarchical_environment import Policy, STOP, Store, read


def settings(**values):
    return {"visible_navigation_negative_advantage": {"enabled": True, **values}}


def decision(ids=(), *, visible=(), stop=False, kind="navigation", protocol="hierarchical_v1"):
    return Decision(
        [], Completion("action"), {"ids": list(ids), "stop": stop},
        {"directory_protocol": protocol, "action_kind": "stop" if stop else kind,
         "visible_navigation_ids": list(visible), "read_call_consumed": not stop},
        format_violations=["visible_navigation"] if set(ids) & set(visible) and not stop else [],
    )


def trajectory(*steps, advantage=2.0, seed=0):
    return Trajectory("q", seed, "memory", decisions=list(steps), advantage=advantage,
                      reward=0.3, outcome=1.0, judge_score=1.0, outcome_reward=1.0,
                      format_reward=-0.6, read_cost=0.1, raw_read_cost=0.1,
                      format_violation_count=1)


def successful_repeat():
    return trajectory(decision(["p1"]), decision(["p1"], visible=["p1"]), decision(stop=True))


def advantages(row, *, process_enabled=False):
    return [effective_decision_advantage(row, index, process_enabled=process_enabled)
            for index in range(len(row.decisions))]


def test_unique_judge_success_keeps_useful_reads_and_stop_positive_but_repeat_negative():
    winner = successful_repeat()
    group = [winner] + [trajectory(decision(stop=True), advantage=-0.5, seed=index)
                        for index in range(1, 5)]
    for row in group[1:]:
        row.judge_score = row.outcome = 0.0
    before = [row.to_dict() for row in group]

    assign_action_advantages(group, settings())

    assert [row.judge_score for row in group] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert advantages(winner) == [2.0, -1.0, 2.0]
    assert [advantages(row) for row in group[1:]] == [[-0.5]] * 4
    assert [row.advantage for row in group] == [2.0, -0.5, -0.5, -0.5, -0.5]
    # Apart from frozen action metadata, neither reward components nor source
    # decisions may be changed by optimizer-side credit assignment.
    for row, saved in zip(group, before):
        current = row.to_dict()
        current.pop("action_advantages")
        assert current == saved


@pytest.mark.parametrize("terminal,magnitude,expected", [
    (2.0, 1.0, -1.0), (0.0, 1.0, -1.0), (-0.5, 1.0, -1.0),
    (-2.0, 1.0, -2.0), (2.0, 0.25, -0.25), (-2.0, 3.0, -3.0),
])
def test_repeat_clamps_once_without_weakening_an_existing_negative_advantage(terminal, magnitude, expected):
    row = trajectory(decision(["p1", "p1", "p2"], visible=["p1", "p2"]),
                     decision(stop=True), advantage=terminal)
    assign_action_advantages([row], settings(magnitude=magnitude))
    assert advantages(row) == [expected, terminal]
    assert row.advantage == terminal


def test_rule_does_not_standardize_negative_action_advantages_across_the_group():
    group = [trajectory(decision(["p1"], visible=["p1"]), advantage=2.0, seed=index)
             for index in range(5)]
    assign_action_advantages(group, settings())
    assert [advantages(row) for row in group] == [[-1.0]] * 5


def test_actual_single_open_reopen_and_repeated_segments_keep_terminal_credit():
    example = Example("q", "locomo", "train", "mem_fixture", "Question?")
    env = HierarchicalMemoryEnvironment(Store(), "frozen_v1", max_open_directories=1,
                                        count_tokens=len)
    row = env.run(example, Policy([
        read("p0001"), read("p0001:c0001"), read("p0002"),
        read("p0001"), read("p0001:c0001"), read("p0001"), STOP,
    ]), 42)
    row.advantage = 2.0
    assert row.decisions[3].tool_result["repeated_navigation"] is True
    assert row.decisions[3].tool_result["visible_navigation_ids"] == []
    assert row.decisions[4].format_violations == ["repeated_read"]
    assert row.decisions[5].format_violations == ["visible_navigation"]
    before = row.to_dict()

    assign_action_advantages([row], settings())

    assert advantages(row) == [2.0, 2.0, 2.0, 2.0, 2.0, -1.0, 2.0]
    after = row.to_dict()
    after.pop("action_advantages")
    assert after == before


@pytest.mark.parametrize("step", [
    decision(["p1"], visible=["p1"], stop=True),
    decision(["p2"], visible=["p1"]),
    decision(["p1"]),
    decision(["p1:c1"], kind="segment"),
    Decision([], Completion("invalid JSON"), None,
             {"directory_protocol": "hierarchical_v1", "visible_navigation_ids": ["p1"]}),
])
def test_only_non_stop_hierarchical_requests_for_visible_parent_ids_are_clamped(step):
    row = trajectory(deepcopy(step))
    assign_action_advantages([row], settings())
    assert advantages(row) == [2.0]


def test_historical_navigation_repeat_and_noop_flags_do_not_invent_visibility():
    step = decision(["p1"])
    step.tool_result.update(repeated_navigation=True, visible_navigation_noop=True)
    row = trajectory(step)
    assign_action_advantages([row], settings())
    assert advantages(row) == [2.0]


def test_enabled_rule_rejects_flat_protocol_or_inconsistent_violation_metadata():
    flat = decision(["p1"], visible=["p1"], protocol="flat_v1")
    missing_visibility = decision(["p1"])
    missing_visibility.format_violations = ["visible_navigation"]
    for step in (flat, missing_visibility):
        with pytest.raises(ValueError):
            assign_action_advantages([trajectory(step)], settings())


def test_visible_parent_in_a_malformed_segment_request_suffix_still_clamps_whole_action():
    step = decision(["p1:c1", "p2", "p2"], visible=["p2"], kind="segment")
    step.format_violations += ["duplicate_ids", "too_many_ids"]
    row = trajectory(step)
    assign_action_advantages([row], settings())
    assert advantages(row) == [-1.0]


def test_configuration_defaults_and_valid_custom_magnitude():
    assert action_advantage_settings({}) == {"enabled": False, "magnitude": 1.0}
    assert action_advantage_settings(settings()) == {"enabled": True, "magnitude": 1.0}
    assert action_advantage_settings(settings(magnitude=0.25)) == {"enabled": True, "magnitude": 0.25}


@pytest.mark.parametrize("value", [None, [], True, "enabled"])
def test_settings_require_a_mapping(value):
    with pytest.raises(ValueError):
        action_advantage_settings({"visible_navigation_negative_advantage": value})


@pytest.mark.parametrize("value", [0, 1, None, "true", []])
def test_settings_require_a_real_boolean(value):
    with pytest.raises(ValueError):
        action_advantage_settings(settings(enabled=value))


@pytest.mark.parametrize("value", [0, -1, True, False, None, "1", [],
                                  float("nan"), float("inf"), float("-inf")])
def test_settings_require_a_finite_strictly_positive_magnitude(value):
    with pytest.raises(ValueError):
        action_advantage_settings(settings(magnitude=value))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_nonfinite_or_boolean_terminal_advantages_are_rejected(value):
    row = successful_repeat()
    row.advantage = value
    with pytest.raises(ValueError):
        assign_action_advantages([row], settings())


def test_disabled_legacy_rows_and_their_serialized_identity_remain_unchanged():
    row = successful_repeat()
    before = deepcopy(row)
    assert row.action_advantages == {}
    assign_action_advantages([row], {})
    assert row == before
    assert advantages(row) == [2.0, 2.0, 2.0]
    assert "action_advantages" not in row.to_dict()
    assert trajectory_from_dict(row.to_dict()) == row


def test_saved_action_credit_roundtrips_and_reassignment_is_idempotent():
    row = successful_repeat()
    assign_action_advantages([row], settings())
    restored = trajectory_from_dict(row.to_dict())
    assert restored == row
    before = deepcopy(restored)
    assign_action_advantages([restored], settings())
    assert restored == before
    assert advantages(restored) == [2.0, -1.0, 2.0]


@pytest.mark.parametrize("changed_settings", [{}, settings(enabled=False), settings(magnitude=0.5)])
def test_saved_action_credit_cannot_be_silently_disabled_or_reinterpreted(changed_settings):
    row = successful_repeat()
    assign_action_advantages([row], settings())
    before = deepcopy(row)
    with pytest.raises(ValueError):
        assign_action_advantages([row], changed_settings)
    assert row == before


@pytest.mark.parametrize("mutation", ["terminal", "visibility", "request", "stop", "format_flag", "saved_advantage", "missing_entry"])
def test_frozen_credit_rejects_modified_terminal_advantage_action_or_visibility(mutation):
    row = successful_repeat()
    assign_action_advantages([row], settings())
    if mutation == "terminal":
        row.advantage = 3.0
    elif mutation == "visibility":
        row.decisions[1].tool_result["visible_navigation_ids"] = []
    elif mutation == "request":
        row.decisions[1].action["ids"] = ["p2"]
    elif mutation == "stop":
        row.decisions[1].action["stop"] = True
    elif mutation == "format_flag":
        row.decisions[1].format_violations = []
    elif mutation == "saved_advantage":
        row.action_advantages["per_decision"][1]["advantage"] = 0.0
    else:
        row.action_advantages["per_decision"].pop()
    before = deepcopy(row)
    with pytest.raises(ValueError):
        assign_action_advantages([row], settings())
    assert row == before


def test_visibility_marker_is_frozen_even_when_terminal_advantage_already_below_clamp():
    row = successful_repeat()
    row.advantage = -2.0
    assign_action_advantages([row], settings())
    assert advantages(row) == [-2.0, -2.0, -2.0]
    row.decisions[1].tool_result["visible_navigation_ids"] = []
    with pytest.raises(ValueError):
        assign_action_advantages([row], settings())


def test_effective_credit_preserves_process_fallback_and_rejects_combined_protocols():
    row = successful_repeat()
    row.process_attribution = {"per_decision": [{"advantage": value} for value in (5.0, 6.0, 7.0)]}
    assert advantages(row) == [2.0, 2.0, 2.0]
    assert advantages(row, process_enabled=True) == [5.0, 6.0, 7.0]
    assign_action_advantages([row], settings())
    assert advantages(row) == [2.0, -1.0, 2.0]
    with pytest.raises(ValueError):
        advantages(row, process_enabled=True)


class IndependentPromptLM(torch.nn.Module):
    """Separate tiny logit vectors make each decision's gradient sign observable."""

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Embedding(8, 8)
        torch.nn.init.zeros_(self.logits.weight)

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0, position_ids=None):
        logits = self.logits(input_ids)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


def model_group(model):
    winner = successful_repeat()
    group = [winner] + [trajectory(decision(stop=True), advantage=-0.5, seed=index)
                        for index in range(1, 5)]
    for row_index, row in enumerate(group):
        if row_index:
            row.judge_score = row.outcome = 0.0
        for index, step in enumerate(row.decisions):
            prompt = [index + 1 if row_index == 0 else 4]
            tokens = [6]
            old = sequence_logprobs(model, prompt, tokens, 1.0, "cpu").detach().tolist()
            step.completion = Completion("action", prompt, tokens, old)
    return group


def action_probabilities(model, row):
    return [sequence_logprobs(model, step.completion.prompt_ids, step.completion.token_ids,
                             1.0, "cpu").detach().exp().item() for step in row.decisions]


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("enabled", [True, False])
def test_actual_dapo_update_decreases_repeat_probability_and_increases_useful_actions(cpu_threads, width, enabled):
    model = IndependentPromptLM()
    group = model_group(model)
    winner = group[0]
    before_probabilities = action_probabilities(model, winner)
    before_rewards = [(row.reward, row.advantage) for row in group]
    policy = policy_for(model)
    cfg = config(group_size=5, micro_batch_rollouts=width, **settings(enabled=enabled))
    updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.3), cfg)

    updater.begin_batch([group], "navigation-gradient")
    frozen = deepcopy(group)
    report = updater.update(group)
    updater.end_batch()

    after = action_probabilities(model, winner)
    assert after[0] > before_probabilities[0]
    assert after[2] > before_probabilities[2]
    if enabled:
        assert after[1] < before_probabilities[1]
        assert advantages(winner) == [2.0, -1.0, 2.0]
    else:
        assert after[1] > before_probabilities[1]
        assert advantages(winner) == [2.0, 2.0, 2.0]
    assert report["training_tokens"] == 7
    assert report["loss"] == pytest.approx(-1.0 / 7 if enabled else -4.0 / 7)
    assert policy.policy_version == 1
    assert group == frozen
    assert [(row.reward, row.advantage) for row in group] == before_rewards


@pytest.mark.parametrize("mutation", ["settings", "saved_advantage", "visibility"])
def test_actual_dapo_update_rejects_mutations_after_freezing(cpu_threads, mutation):
    model = IndependentPromptLM()
    group = model_group(model)
    cfg = config(group_size=5, **settings())
    policy = policy_for(model)
    updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.3), cfg)
    updater.begin_batch([group], "frozen-navigation")
    if mutation == "settings":
        cfg["visible_navigation_negative_advantage"]["magnitude"] = 0.5
    elif mutation == "saved_advantage":
        group[0].action_advantages["per_decision"][1]["advantage"] = 2.0
    else:
        group[0].decisions[1].tool_result["visible_navigation_ids"] = []
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        updater.update(group)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert policy.policy_version == 0


def test_actual_dapo_checkpoint_resume_keeps_frozen_action_credit_and_identical_updates(cpu_threads):
    model = IndependentPromptLM()
    groups = [model_group(model), model_group(model)]
    for row in groups[1]:
        row.sample_id = "q-resumed"
    cfg = config(group_size=5, **settings())
    policy = policy_for(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    updater = DAPOUpdater(policy, optimizer, cfg)
    updater.begin_batch(groups, "navigation-resume")
    updater.update(groups[0])

    restored_groups = [[trajectory_from_dict(row.to_dict()) for row in group] for group in groups]
    restored_model = deepcopy(model)
    restored_policy = policy_for(restored_model, version=1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    restored = DAPOUpdater(restored_policy, restored_optimizer, cfg)
    restored.begin_batch(restored_groups, "navigation-resume", consumed_groups=1, behavior_version=0)
    resumed_report = restored.update(restored_groups[1])
    continuous_report = updater.update(groups[1])

    assert restored_groups == groups
    assert advantages(restored_groups[1][0]) == [2.0, -1.0, 2.0]
    assert restored_policy.policy_version == policy.policy_version == 2
    assert resumed_report["loss"] == pytest.approx(continuous_report["loss"])
    for actual, expected in zip(restored_model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected)
    updater.end_batch()
    restored.end_batch()


def test_actual_dapo_overlong_mask_removes_all_tokens_of_the_truncated_rollout(cpu_threads):
    model = IndependentPromptLM()
    group = model_group(model)
    winner = group[0]
    winner.decisions[1].completion.truncated = True
    before = action_probabilities(model, winner)
    policy = policy_for(model)
    updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.3),
                          config(group_size=5, **settings()))
    updater.begin_batch([group], "navigation-overlong")
    report = updater.update(group)
    updater.end_batch()

    assert advantages(winner) == [2.0, -1.0, 2.0]
    assert report["training_tokens"] == 4
    assert report["masked_rollouts"] == 1
    assert report["loss"] == pytest.approx(0.5)
    assert action_probabilities(model, winner) == before
    assert report["navigation_rule_decisions"] == 0
    assert report["sole_correct_groups"] == 0


def test_require_saved_rejects_deleted_whole_group_and_keeps_partial_metadata_immutable():
    group = [successful_repeat(), trajectory(decision(stop=True), advantage=-2.0, seed=1)]
    assign_action_advantages(group, settings())
    for deleted_count in (1, len(group)):
        restored = deepcopy(group)
        for row in restored[:deleted_count]:
            row.action_advantages = {}
        before = deepcopy(restored)
        with pytest.raises(ValueError, match="saved action advantages"):
            assign_action_advantages(restored, settings(), require_saved=True)
        assert restored == before


@pytest.mark.parametrize("cursor", [0, 1])
@pytest.mark.parametrize("deleted_group", [0, 1])
def test_actual_dapo_resume_rejects_deleted_whole_group_metadata(cpu_threads, cursor, deleted_group):
    model = IndependentPromptLM()
    groups = [model_group(model), model_group(model)]
    for row in groups[1]:
        row.sample_id = "q-resumed"
    cfg = config(group_size=5, **settings())
    policy = policy_for(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    updater = DAPOUpdater(policy, optimizer, cfg)
    updater.begin_batch(groups, "deleted-navigation-metadata")
    if cursor:
        updater.update(groups[0])
    restored_groups = [[trajectory_from_dict(row.to_dict()) for row in group] for group in groups]
    for row in restored_groups[deleted_group]:
        row.action_advantages = {}
    restored_model = deepcopy(model)
    restored_policy = policy_for(restored_model, version=cursor)
    restored = DAPOUpdater(restored_policy, torch.optim.SGD(restored_model.parameters(), lr=0.3), cfg)
    before = deepcopy(restored_groups)
    # A checkpoint saved before its first optimizer step needs the explicit
    # restoration flag; a nonzero cursor implies restoration automatically.
    options = {"require_saved_action_advantages": True} if cursor == 0 else {}
    with pytest.raises(ValueError, match="saved action advantages"):
        restored.begin_batch(restored_groups, "deleted-navigation-metadata",
                             consumed_groups=cursor, behavior_version=0, **options)
    assert restored_groups == before
    assert restored_policy.policy_version == cursor
    for actual, expected in zip(restored_model.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_action_metrics_distinguish_rule_matches_from_overrides_and_keep_sole_correct_credit():
    winner = successful_repeat()
    loser = trajectory(decision(["p2"], visible=["p2"]), advantage=-2.0, seed=1)
    loser.judge_score = loser.outcome = 0.0
    group = [winner, loser]
    assign_action_advantages(group, settings())

    assert action_advantage_metrics(group) == {
        "navigation_rule_decisions": 2,
        "navigation_rule_overridden_decisions": 1,
        "navigation_rule_positive_before": 1,
        "navigation_rule_positive_after": 0,
        "navigation_rule_positive_advantage_rate": 0.0,
        "navigation_rule_advantage_mean": -1.5,
        "navigation_rule_correct_decisions": 1,
        "sole_correct_groups": 1,
        "sole_correct_nonpositive_group_advantage": 0,
        "sole_correct_positive_valid_read_decisions": 1,
    }
    assert action_advantage_metrics([]) == {}
    assert action_advantage_metrics([successful_repeat()]) == {}


def test_action_metrics_expose_a_sole_correct_rollout_with_nonpositive_terminal_credit():
    winner = successful_repeat()
    winner.advantage = 0.0
    assign_action_advantages([winner], settings())
    metrics = action_advantage_metrics([winner])
    assert metrics["sole_correct_groups"] == 1
    assert metrics["sole_correct_nonpositive_group_advantage"] == 1
    assert metrics["sole_correct_positive_valid_read_decisions"] == 0
    assert metrics["navigation_rule_positive_after"] == 0
    assert metrics["navigation_rule_advantage_mean"] == -1.0


def test_saved_compact_trace_records_actual_training_credit_and_preserves_terminal_credit(tmp_path):
    row = successful_repeat()
    row.decisions[0].messages = [{"role": "user", "content": "PRIVATE_PROMPT"}]
    row.decisions[0].tool_result["records"] = [{"raw_text": "PRIVATE_EVIDENCE"}]
    assign_action_advantages([row], settings())
    save_training_group(tmp_path, 1, "navigation-audit", "frozen", [row])

    text = (tmp_path / "rollouts/group-00000001.json").read_text()
    saved = json.loads(text)["rollouts"][0]
    assert [step["training_advantage"] for step in saved["reader_steps"]] == [2.0, -1.0, 2.0]
    assert [step["group_advantage"] for step in saved["reader_steps"]] == [2.0, 2.0, 2.0]
    assert [step["negative_navigation_rule"] for step in saved["reader_steps"]] == [False, True, False]
    assert saved["action_advantages"] == row.action_advantages
    assert saved["advantage"] == row.advantage == 2.0
    assert saved["reward"] == row.reward
    assert "PRIVATE_" not in text
    with gzip.open(tmp_path / "groups/group-00000001.json.gz", "rt") as stream:
        full = json.load(stream)
    restored = trajectory_from_dict(full["rollouts"][0])
    assign_action_advantages([restored], settings(), require_saved=True)
    assert restored == row


def test_legacy_compact_trace_does_not_fabricate_action_supervision_fields():
    row = successful_repeat()
    group = {"group_id": "legacy", "directory_version": "frozen", "rollouts": [row.to_dict()]}
    saved = compact_group(group)["rollouts"][0]
    assert "action_advantages" not in saved
    for step in saved["reader_steps"]:
        assert "training_advantage" not in step
        assert "group_advantage" not in step
        assert "negative_navigation_rule" not in step


def test_actual_dapo_metrics_write_actual_negative_credit_to_tensorboard(cpu_threads, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    model = IndependentPromptLM()
    group = model_group(model)
    policy = policy_for(model)
    updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.3),
                          config(group_size=5, **settings()))
    updater.begin_batch([group], "navigation-metrics")
    report = updater.update(group)
    updater.end_batch()
    expected = action_advantage_metrics(group)
    assert expected["navigation_rule_advantage_mean"] == -1.0
    assert expected["navigation_rule_positive_before"] == 1
    assert expected["navigation_rule_positive_after"] == 0
    assert expected["sole_correct_positive_valid_read_decisions"] == 1
    assert {key: report[key] for key in expected} == expected

    writer = summary_writer_class()(log_dir=str(tmp_path))
    try:
        write_train_metrics(writer, {"step": 7, **report})
    finally:
        writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    for key, value in expected.items():
        points = events.Scalars(f"train/{key}")
        assert len(points) == 1
        assert points[0].step == 7
        assert points[0].value == pytest.approx(value)
