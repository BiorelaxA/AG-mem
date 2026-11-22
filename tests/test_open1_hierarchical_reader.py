"""Single-open navigation state stays explicit and keeps retrieved evidence."""
import json
from types import SimpleNamespace

import pytest

from agmem.core import Example
from agmem.hierarchical_env import HierarchicalMemoryEnvironment
from agmem.hierarchical_prompts import hierarchical_reader_input_contract
from agmem.rewards import score_trajectory, visible_navigation_id_count
from tests.test_hierarchical_environment import Policy, Store, read, STOP


def test_one_open_directory_identity_eviction_and_targeted_penalties():
    example = Example("locomo/train/0", "locomo", "train", "mem_fixture", "Question?",
                      gold_answers=["answer"])
    policy = Policy([read("p0001"), read("p0001:c0001"), read("p0002"), read("p0002"),
                     read("p0001"), read("p0001:c0001"), read("p0002:c0001"), STOP])
    env = HierarchicalMemoryEnvironment(Store(), "frozen_v1", count_tokens=len,
                                       max_open_directories=1)
    row = env.run(example, policy, 100)
    states = [json.loads(m[-1]["content"]) for m in policy.messages]
    assert states[0]["currently_open_parent_id"] is None
    assert states[0]["open_child_directories"] == []
    for state in states[1:]:
        assert len(state["open_child_directories"]) == 1
        directory = state["open_child_directories"][0]
        parent = next(p for p in state["parent_directory"]
                      if p["id"] == state["currently_open_parent_id"])
        assert directory["parent_id"] == parent["id"]
        assert directory["parent_title"] == parent["title"]
        assert state["open_parent_ids_lru"] == [parent["id"]]
    assert states[3]["currently_open_parent_id"] == "p0002"
    assert states[3]["read_ids"] == ["p0001:c0001"]
    assert "Exact original 1.1" in states[3]["retrieved_evidence"][0]["raw_text"]
    assert states[5]["currently_open_parent_id"] == "p0001"
    assert "CHILD_SUMMARY_2_1" not in json.dumps(states[5])
    assert row.decisions[3].format_violations == ["visible_navigation"]
    assert row.decisions[4].format_violations == []  # Reopen after eviction is legal.
    assert row.decisions[5].format_violations == ["repeated_read"]
    assert row.decisions[6].format_violations == ["unknown_id"]
    assert visible_navigation_id_count(row) == 1
    assert row.n_read_calls == 7
    assert [e["id"] for e in row.evidence] == ["p0001:c0001"]
    row.prediction = "answer"
    score_trajectory(example, row, {
        "formula": "judge_f1_cost_v1", "judge_f1_alpha": .8, "read_call_cost": .05,
        "format_penalty": .1, "visible_navigation_penalty": .6, "format_penalty_mode": "per_id",
    }, max_ids=1, judge=SimpleNamespace(score=lambda *_: 1.0))
    assert row.format_violation_count == 3
    assert row.format_reward == pytest.approx(-.8)
    assert row.read_cost == pytest.approx(.35)
    assert row.reward == pytest.approx(-.15)
    system = policy.messages[0][0]["content"]
    assert "currently_open_parent_id" in system
    assert "only a candidate, not a confirmed relevant or target directory" in system
    assert "not just facts visible in directory summaries" in system


def test_single_open_schema_and_prompt_are_distinct_from_two_open_protocol():
    one = hierarchical_reader_input_contract(max_open_directories=1)
    two = hierarchical_reader_input_contract(max_open_directories=2)
    assert one["state_schema"] == "hierarchical_orm_single_open_state_v1"
    assert two["state_schema"] == "hierarchical_orm_state_v1"
    assert one["reader_prompt_sha256"] != two["reader_prompt_sha256"]
