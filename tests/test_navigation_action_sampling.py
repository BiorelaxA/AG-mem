"""Rule supervision must not admit all-wrong or tied outcome groups."""
from copy import deepcopy

import pytest

from agmem.core import load_config
from agmem.dapo_sampling import assess_group, validate_dapo_config
from agmem.resume import validate_resume_config
from tests.test_judge_f1_sampling import config_for, joint_group, source


@pytest.mark.parametrize("judges,calls,accepted", [
    ([0, 0], [0, 3], False),
    ([1, 1], [2, 2], False),
    ([1, 0], [3, 2], True),
    ([1, 1], [2, 3], True),
])
def test_action_rule_keeps_existing_whole_group_filter(judges, calls, accepted):
    cfg = config_for()
    original = joint_group(source()[0], ["gold answer"] * 2, judges, calls)
    expected = assess_group(original, cfg)
    cfg["training"]["visible_navigation_negative_advantage"] = {"enabled": True, "magnitude": 1.0}
    current = deepcopy(original)
    actual = assess_group(current, cfg)
    assert actual == expected
    assert actual["accepted"] is accepted
    assert all(not row.action_advantages for row in current)
    assert [row.advantage for row in current] == [row.advantage for row in original]


def test_current_recipe_validates_but_cannot_silently_resume_old_objective():
    cfg = load_config("configs/train_dapo_locomo_lmes_hierarchical_f1_2gpu.yaml")
    validate_dapo_config(cfg)
    validate_resume_config(cfg, deepcopy(cfg))
    old = deepcopy(cfg)
    old["training"].pop("visible_navigation_negative_advantage")
    with pytest.raises(ValueError, match="training"):
        validate_resume_config(old, cfg)


@pytest.mark.parametrize("section,key,value", [
    ("training", "algorithm", "grpo"),
    ("data", "directory_protocol", "flat_v1"),
    ("reward", "formula", "f1_cost_v1"),
    ("training", "process_reward", {"enabled": True}),
])
def test_rule_cannot_silently_change_other_training_protocols(section, key, value):
    cfg = load_config("configs/train_dapo_locomo_lmes_hierarchical_f1_2gpu.yaml")
    cfg[section][key] = value
    with pytest.raises(ValueError):
        validate_dapo_config(cfg)
