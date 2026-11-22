"""Freeze opt-in repeated-segment credit, including real CPU actor updates."""
from copy import deepcopy
import json

import pytest
import torch

from agmem.action_advantages import (action_advantage_metrics, action_advantage_settings,
                                     assign_action_advantages, effective_decision_advantage)
from agmem.core import CapacityError, Completion, Decision, Example, trajectory_from_dict
from agmem.dapo import DAPOUpdater
from agmem.hierarchical_env import HierarchicalMemoryEnvironment
from agmem.monitoring import summary_writer_class, write_train_metrics
from agmem.policy import sequence_logprobs
from agmem.traces import save_training_group
from tests.test_dapo import config, policy_for
from tests.test_hierarchical_environment import Policy, STOP, Store, read
from tests.test_navigation_action_advantages import (IndependentPromptLM, action_probabilities,
                                                    advantages, cpu_threads, decision,
                                                    settings as navigation_settings, trajectory)


SEGMENT = "p0001:c0001"
OTHER_SEGMENT = "p0001:c0002"


def settings(**values):
    return navigation_settings(include_repeated_segments=values.get("enabled", True), **values)


def segment_decision(ids=(SEGMENT,), *, already=(), new=(), records=None):
    step = decision(ids, kind="segment")
    step.tool_result.update(
        requested_ids=list(ids), selected_ids=list(ids[:1]),
        already_read_ids=list(already), new_ids=list(new),
        records=[{"id": key, "raw_text": "exact evidence"} for key in new] if records is None else records,
    )
    if set(ids) & set(already):
        step.format_violations.append("repeated_read")
    return step


def successful_segment_repeat(*, advantage=2.0, seed=0):
    row = trajectory(segment_decision(new=(SEGMENT,)),
                     segment_decision(already=(SEGMENT,)), decision(stop=True),
                     advantage=advantage, seed=seed)
    # Both requests appear in final evidence; that is not pre-action state.
    row.evidence = deepcopy(row.decisions[0].tool_result["records"])
    return row


def run_actions(actions, **kwargs):
    environment = HierarchicalMemoryEnvironment(Store(), "frozen_v1", count_tokens=len, **kwargs)
    row = environment.run(Example("q", "locomo", "train", "mem_fixture", "Question?"),
                          Policy(actions), 42)
    row.advantage = 2.0
    row.judge_score = row.outcome = 1.0
    return row


def segment_model_group(model):
    winner = successful_segment_repeat()
    group = [winner] + [trajectory(decision(stop=True), advantage=-0.5, seed=seed)
                        for seed in range(1, 5)]
    for row_index, row in enumerate(group):
        if row_index:
            row.judge_score = row.outcome = 0.0
        for index, step in enumerate(row.decisions):
            prompt = [index + 1 if row_index == 0 else 4]
            tokens = [6]
            old = sequence_logprobs(model, prompt, tokens, 1.0, "cpu").detach().tolist()
            step.completion = Completion("action", prompt, tokens, old)
    return group


def test_segment_supervision_is_opt_in_and_preserves_legacy_settings_shape():
    assert action_advantage_settings({}) == {"enabled": False, "magnitude": 1.0}
    assert action_advantage_settings(navigation_settings()) == {"enabled": True, "magnitude": 1.0}
    assert action_advantage_settings(navigation_settings(include_repeated_segments=False)) == {
        "enabled": True, "magnitude": 1.0,
    }
    assert action_advantage_settings(settings()) == {
        "enabled": True, "magnitude": 1.0, "include_repeated_segments": True,
    }


@pytest.mark.parametrize("value", [0, 1, None, "true", [], {}])
def test_segment_flag_requires_a_real_boolean(value):
    with pytest.raises(ValueError):
        action_advantage_settings(navigation_settings(include_repeated_segments=value))


def test_segment_rule_cannot_be_enabled_under_disabled_action_supervision():
    with pytest.raises(ValueError):
        action_advantage_settings(navigation_settings(enabled=False, include_repeated_segments=True))


def test_v2_clamps_only_the_repeat_and_leaves_terminal_rewards_and_actions_unchanged():
    row = successful_segment_repeat()
    before = row.to_dict()
    assign_action_advantages([row], settings())

    assert advantages(row) == [2.0, -1.0, 2.0]
    assert row.action_advantages["protocol"].endswith("_v2")
    assert [entry["repeated_segment"] for entry in row.action_advantages["per_decision"]] == [False, True, False]
    assert all(not entry["visible_navigation"] for entry in row.action_advantages["per_decision"])
    after = row.to_dict()
    after.pop("action_advantages")
    assert after == before


@pytest.mark.parametrize("terminal,magnitude,expected", [
    (2.0, 1.0, -1.0), (0.0, 1.0, -1.0), (-0.5, 1.0, -1.0),
    (-2.0, 1.0, -2.0), (2.0, 0.25, -0.25), (-2.0, 3.0, -3.0),
])
def test_repeated_segment_clamp_is_not_added_per_id_or_recentred(terminal, magnitude, expected):
    row = trajectory(segment_decision((SEGMENT, SEGMENT, OTHER_SEGMENT),
                                      already=(SEGMENT, OTHER_SEGMENT)),
                     decision(stop=True), advantage=terminal)
    row.decisions[0].format_violations += ["duplicate_ids", "too_many_ids"]
    group = [deepcopy(row) for _ in range(5)]
    assign_action_advantages(group, settings(magnitude=magnitude))
    assert [advantages(item) for item in group] == [[expected, terminal]] * 5
    assert all(item.advantage == terminal for item in group)


def test_real_environment_eviction_does_not_forget_read_child_and_legal_reopen_stays_positive():
    row = run_actions([read("p0001"), read(SEGMENT), read("p0002"), read(SEGMENT),
                       read("p0001"), read(SEGMENT), read("p0001"), STOP],
                      max_open_directories=1)
    assert row.decisions[3].tool_result["open_parent_ids_lru"] == ["p0002"]
    assert row.decisions[3].tool_result["already_read_ids"] == [SEGMENT]
    assert row.decisions[4].tool_result["repeated_navigation"] is True
    assert row.decisions[4].tool_result["visible_navigation_ids"] == []
    before = row.to_dict()

    assign_action_advantages([row], settings())

    assert advantages(row) == [2.0, 2.0, 2.0, -1.0, 2.0, -1.0, -1.0, 2.0]
    assert [entry["repeated_segment"] for entry in row.action_advantages["per_decision"]] == [
        False, False, False, True, False, True, False, False,
    ]
    after = row.to_dict()
    after.pop("action_advantages")
    assert after == before


def test_failed_capacity_attempt_does_not_make_the_first_success_a_repeat():
    rejected = False

    def capacity(reader, task):
        nonlocal rejected
        state = json.loads(reader[-1]["content"])
        if state["retrieved_evidence"] and not rejected:
            rejected = True
            raise CapacityError("reject only the first candidate segment")

    row = run_actions([read("p0001"), read(SEGMENT), read(SEGMENT), read(SEGMENT), STOP],
                      capacity=capacity)
    failed, first, repeated = row.decisions[1:4]
    assert failed.tool_result["error"]["code"] == "context_capacity_exceeded"
    assert failed.tool_result["new_ids"] == failed.tool_result["records"] == []
    assert failed.tool_result["already_read_ids"] == first.tool_result["already_read_ids"] == []
    assert first.tool_result["new_ids"] == [SEGMENT]
    assert repeated.tool_result["already_read_ids"] == [SEGMENT]

    assign_action_advantages([row], settings())

    assert advantages(row) == [2.0, 2.0, 2.0, -1.0, 2.0]


def test_unopened_child_attempt_then_successful_retry_does_not_count_as_repeat():
    row = run_actions([read(SEGMENT), read("p0001"), read(SEGMENT), STOP])
    assert row.decisions[0].tool_result["invalid_ids"] == [SEGMENT]
    assert row.decisions[0].tool_result["new_ids"] == []
    assert row.decisions[2].tool_result["new_ids"] == [SEGMENT]
    assign_action_advantages([row], settings())
    assert advantages(row) == [2.0] * 4


def test_duplicate_id_in_its_first_request_does_not_invent_a_previous_success():
    row = run_actions([read("p0001"), read(SEGMENT, SEGMENT), STOP])
    assert "duplicate_ids" in row.decisions[1].format_violations
    assert row.decisions[1].tool_result["already_read_ids"] == []
    assign_action_advantages([row], settings())
    assert advantages(row) == [2.0, 2.0, 2.0]


def test_already_read_segment_in_excess_suffix_clamps_the_whole_read_once():
    row = run_actions([read("p0001"), read(SEGMENT), read(OTHER_SEGMENT, SEGMENT, SEGMENT), STOP])
    step = row.decisions[2]
    assert step.tool_result["new_ids"] == [OTHER_SEGMENT]
    assert step.tool_result["already_read_ids"] == [SEGMENT]
    assign_action_advantages([row], settings())
    assert advantages(row) == [2.0, 2.0, -1.0, 2.0]


def test_stop_with_already_read_ids_never_receives_read_supervision():
    row = run_actions([read("p0001"), read(SEGMENT), {"ids": [SEGMENT], "stop": True}])
    assert row.decisions[-1].format_violations == ["stop_with_ids"]
    assert row.decisions[-1].tool_result["read_call_consumed"] is False
    assign_action_advantages([row], settings())
    assert advantages(row) == [2.0, 2.0, 2.0]
    assert not row.action_advantages["per_decision"][-1]["repeated_segment"]


def test_wholly_correct_useful_trajectory_keeps_every_action_positive():
    row = run_actions([read("p0001"), read(SEGMENT), read(OTHER_SEGMENT),
                       read("p0002"), read("p0002:c0001"), STOP])
    assert all(not step.format_violations for step in row.decisions)
    assign_action_advantages([row], settings())
    assert advantages(row) == [2.0] * len(row.decisions)
    assert action_advantage_metrics([row])["segment_rule_decisions"] == 0


def test_parsing_failure_and_parent_id_are_not_mistaken_for_a_read_segment():
    invalid = Decision([], Completion("not JSON"), None,
                       {"directory_protocol": "hierarchical_v1", "action_kind": "invalid",
                        "already_read_ids": [SEGMENT], "new_ids": [], "records": []})
    parent = decision(["p0001"])
    parent.tool_result.update(already_read_ids=[SEGMENT], new_ids=[], records=[])
    row = trajectory(invalid, parent, decision(stop=True))
    assign_action_advantages([row], settings())
    assert advantages(row) == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("already", ["p0001:c0001", ["p0001"], [1], None])
def test_segment_rule_rejects_invalid_frozen_pre_action_segment_ids(already):
    row = successful_segment_repeat()
    row.decisions[1].tool_result["already_read_ids"] = already
    with pytest.raises(ValueError):
        assign_action_advantages([row], settings())
    assert row.action_advantages == {}


def test_repeated_read_flag_alone_cannot_invent_successful_segment_history():
    row = trajectory(segment_decision())
    row.decisions[0].format_violations = ["repeated_read"]
    with pytest.raises(ValueError):
        assign_action_advantages([row], settings())
    assert row.action_advantages == {}


@pytest.mark.parametrize("training", [{}, settings(enabled=False)])
def test_disabled_segment_rule_leaves_serialization_and_training_credit_unchanged(training):
    row = successful_segment_repeat()
    before = deepcopy(row)
    assign_action_advantages([row], training)
    assert row == before
    assert "action_advantages" not in row.to_dict()
    assert advantages(row) == [2.0, 2.0, 2.0]


def test_legacy_protocol_still_ignores_segment_repeat_and_roundtrips_unchanged():
    row = successful_segment_repeat()
    assign_action_advantages([row], navigation_settings())
    assert row.action_advantages["protocol"] == "reader_visible_navigation_negative_advantage_v1"
    assert all("repeated_segment" not in entry for entry in row.action_advantages["per_decision"])
    assert advantages(row) == [2.0, 2.0, 2.0]
    restored = trajectory_from_dict(row.to_dict())
    assign_action_advantages([restored], navigation_settings(include_repeated_segments=False), require_saved=True)
    assert restored == row
    assert not any(key.startswith("segment_rule_") for key in action_advantage_metrics([row]))


def test_v2_roundtrip_is_idempotent_and_rejects_reinterpretation_as_v1():
    row = successful_segment_repeat()
    assign_action_advantages([row], settings())
    restored = trajectory_from_dict(row.to_dict())
    assign_action_advantages([restored], settings(), require_saved=True)
    assert restored == row
    assert advantages(restored) == [2.0, -1.0, 2.0]
    for changed in ({}, navigation_settings(), navigation_settings(include_repeated_segments=False)):
        with pytest.raises(ValueError):
            assign_action_advantages([restored], changed)
        assert restored == row
    with pytest.raises(ValueError):
        effective_decision_advantage(restored, 0, process_enabled=True)


@pytest.mark.parametrize("field,value", [
    ("already_read_ids", []), ("new_ids", [OTHER_SEGMENT]),
    ("records", [{"id": OTHER_SEGMENT, "raw_text": "changed source evidence"}]),
    ("read_call_consumed", False),
])
def test_v2_freezes_repeat_source_metadata_even_below_the_negative_clamp(field, value):
    row = successful_segment_repeat(advantage=-2.0)
    assign_action_advantages([row], settings())
    assert advantages(row) == [-2.0, -2.0, -2.0]
    row.decisions[1].tool_result[field] = value
    before = deepcopy(row)
    with pytest.raises(ValueError):
        assign_action_advantages([row], settings(), require_saved=True)
    assert row == before


def test_v2_freezes_success_metadata_of_the_original_read():
    row = successful_segment_repeat()
    assign_action_advantages([row], settings())
    row.decisions[0].tool_result["records"][0]["id"] = OTHER_SEGMENT
    with pytest.raises(ValueError):
        assign_action_advantages([row], settings(), require_saved=True)


@pytest.mark.parametrize("width", [1, 3])
@pytest.mark.parametrize("enabled", [True, False])
def test_actual_cpu_dapo_unique_judge_success_only_decreases_the_repeat(cpu_threads, width, enabled):
    model = IndependentPromptLM()
    group = segment_model_group(model)
    winner = group[0]
    before = action_probabilities(model, winner)
    original_rewards = [(row.reward, row.advantage, row.judge_score) for row in group]
    policy = policy_for(model)
    updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.3),
                          config(group_size=5, micro_batch_rollouts=width, **settings(enabled=enabled)))
    updater.begin_batch([group], "segment-gradient")
    frozen = deepcopy(group)
    report = updater.update(group)
    updater.end_batch()

    after = action_probabilities(model, winner)
    assert after[0] > before[0]
    assert after[2] > before[2]
    if enabled:
        assert after[1] < before[1]
        assert advantages(winner) == [2.0, -1.0, 2.0]
    else:
        assert after[1] > before[1]
        assert advantages(winner) == [2.0, 2.0, 2.0]
    assert report["training_tokens"] == 7
    assert report["loss"] == pytest.approx(-1.0 / 7 if enabled else -4.0 / 7)
    assert group == frozen
    assert policy.policy_version == 1
    assert [(row.reward, row.advantage, row.judge_score) for row in group] == original_rewards


@pytest.mark.parametrize("mutation", ["settings", "already_read_ids", "records", "marker"])
def test_actual_cpu_dapo_rejects_frozen_segment_mutation_before_changing_weights(cpu_threads, mutation):
    model = IndependentPromptLM()
    group = segment_model_group(model)
    cfg = config(group_size=5, **settings())
    policy = policy_for(model)
    updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.3), cfg)
    updater.begin_batch([group], "frozen-segment")
    if mutation == "settings":
        cfg["visible_navigation_negative_advantage"]["include_repeated_segments"] = False
    elif mutation == "marker":
        group[0].action_advantages["per_decision"][1]["repeated_segment"] = False
    elif mutation == "records":
        group[0].decisions[0].tool_result["records"][0]["id"] = OTHER_SEGMENT
    else:
        group[0].decisions[1].tool_result["already_read_ids"] = []
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        updater.update(group)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert policy.policy_version == 0


def test_actual_cpu_dapo_v2_checkpoint_resume_reproduces_the_continuous_update(cpu_threads):
    model = IndependentPromptLM()
    groups = [segment_model_group(model), segment_model_group(model)]
    for row in groups[1]:
        row.sample_id = "q-resumed"
    cfg = config(group_size=5, **settings())
    policy = policy_for(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    updater = DAPOUpdater(policy, optimizer, cfg)
    updater.begin_batch(groups, "segment-resume")
    updater.update(groups[0])

    restored_groups = [[trajectory_from_dict(row.to_dict()) for row in group] for group in groups]
    restored_model = deepcopy(model)
    restored_policy = policy_for(restored_model, version=1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    restored = DAPOUpdater(restored_policy, restored_optimizer, cfg)
    restored.begin_batch(restored_groups, "segment-resume", consumed_groups=1, behavior_version=0)
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


def test_actual_cpu_dapo_overlong_mask_excludes_segment_credit_and_tokens(cpu_threads):
    model = IndependentPromptLM()
    group = segment_model_group(model)
    winner = group[0]
    winner.decisions[1].completion.truncated = True
    before = action_probabilities(model, winner)
    policy = policy_for(model)
    updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.3),
                          config(group_size=5, **settings()))
    updater.begin_batch([group], "segment-overlong")
    report = updater.update(group)
    updater.end_batch()

    assert advantages(winner) == [2.0, -1.0, 2.0]
    assert report["training_tokens"] == 4
    assert report["masked_rollouts"] == 1
    assert report["loss"] == pytest.approx(0.5)
    assert action_probabilities(model, winner) == before
    assert report["segment_rule_decisions"] == 0
    assert report["sole_correct_groups"] == 0


def test_segment_metrics_distinguish_matches_overrides_and_keep_useful_sole_correct_credit():
    winner = successful_segment_repeat()
    loser = successful_segment_repeat(advantage=-2.0, seed=1)
    loser.judge_score = loser.outcome = 0.0
    assign_action_advantages([winner, loser], settings())
    metrics = action_advantage_metrics([winner, loser])
    assert {key: value for key, value in metrics.items() if key.startswith("segment_rule_")} == {
        "segment_rule_decisions": 2,
        "segment_rule_overridden_decisions": 1,
        "segment_rule_positive_before": 1,
        "segment_rule_positive_after": 0,
        "segment_rule_positive_advantage_rate": 0.0,
        "segment_rule_advantage_mean": -1.5,
        "segment_rule_correct_decisions": 1,
    }
    assert metrics["navigation_rule_decisions"] == 0
    assert metrics["sole_correct_groups"] == 1
    assert metrics["sole_correct_nonpositive_group_advantage"] == 0
    assert metrics["sole_correct_positive_valid_read_decisions"] == 1


def test_compact_trace_exposes_v2_action_credit_without_private_evidence(tmp_path):
    row = run_actions([read("p0001"), read(SEGMENT), read(SEGMENT), STOP])
    row.decisions[0].messages = [{"role": "user", "content": "PRIVATE_PROMPT"}]
    row.decisions[1].tool_result["records"][0]["raw_text"] = "PRIVATE_EVIDENCE"
    assign_action_advantages([row], settings())
    save_training_group(tmp_path, 1, "segment-audit", "frozen", [row])

    text = (tmp_path / "rollouts/group-00000001.json").read_text()
    saved = json.loads(text)["rollouts"][0]
    assert [step["training_advantage"] for step in saved["reader_steps"]] == [2.0, 2.0, -1.0, 2.0]
    assert [step["group_advantage"] for step in saved["reader_steps"]] == [2.0] * 4
    assert [step["negative_segment_rule"] for step in saved["reader_steps"]] == [False, False, True, False]
    assert [step["negative_navigation_rule"] for step in saved["reader_steps"]] == [False] * 4
    assert saved["action_advantages"] == row.action_advantages
    assert saved["advantage"] == row.advantage == 2.0
    assert saved["reward"] == row.reward
    assert "PRIVATE_" not in text


def test_actual_cpu_dapo_writes_segment_rule_metrics_to_tensorboard(cpu_threads, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    model = IndependentPromptLM()
    group = segment_model_group(model)
    policy = policy_for(model)
    updater = DAPOUpdater(policy, torch.optim.SGD(model.parameters(), lr=0.3),
                          config(group_size=5, **settings()))
    updater.begin_batch([group], "segment-metrics")
    report = updater.update(group)
    updater.end_batch()
    expected = action_advantage_metrics(group)
    assert expected["segment_rule_advantage_mean"] == -1.0
    assert expected["segment_rule_positive_before"] == 1
    assert expected["segment_rule_positive_after"] == 0
    assert {key: report[key] for key in expected} == expected

    writer = summary_writer_class()(log_dir=str(tmp_path))
    try:
        write_train_metrics(writer, {"step": 9, **report})
    finally:
        writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    for key, value in expected.items():
        points = events.Scalars(f"train/{key}")
        assert len(points) == 1
        assert points[0].step == 9
        assert points[0].value == pytest.approx(value)
