"""CPU-only capacity preflight counts real prompts without mutating snapshots."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agmem.core import CapacityError, Example
from agmem.full_directory_preflight import initial_context_report
from agmem.full_directory_prompts import full_directory_reader_messages
from agmem.hierarchical.reader import DirectoryReader
from agmem.policy import chat_token_ids


class CharacterTokenizer:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True,
                          "enable_thinking": False, "return_dict": False,
                          "return_tensors": None, "padding": False, "truncation": False}
        self.messages.append(deepcopy(messages))
        # Include a synthetic chat-template boundary, rather than counting the
        # user JSON alone. Public question length remains observably different.
        return [1] * (17 + sum(len(message["content"]) for message in messages))


def setup_case(tmp_path, native_limit=100000):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {
        "max_position_embeddings": native_limit}}))
    config = {
        "data": {"directory_protocol": "hierarchical_v1"},
        "models": {"reader_path": str(tmp_path)},
        "reader": {"prompt_version": "hierarchical_full_v1", "max_context_tokens": None,
                   "max_new_tokens_per_decision": 16},
        "retrieval": {"max_ids_per_call": 2, "max_read_calls": 10},
        "reward": {"formula": "full_directory_orm_v1"},
    }
    directories = [{"id": f"p{i:04}", "order": i, "time": "2025-01-01", "participants": [],
                    "title": f"Parent {i}", "content": f"Parent summary {i}",
                    "private_source": "DO_NOT_RENDER_PARENT",
                    "children": [{"id": f"p{i:04}:c0001", "title": f"Child {i}",
                                  "content": f"Child summary {i}", "entities": [],
                                  "source": "DO_NOT_RENDER_CHILD"}]} for i in (2, 1)]
    store = SimpleNamespace(get_metadata=Mock(return_value="frozen-v1"),
                            hierarchical_bank=Mock(return_value=(directories, [{"raw": "PRIVATE_RAW"}])),
                            close=Mock(side_effect=AssertionError("Read-only preflight must not close the store")))
    examples = {
        "train": [Example("locomo/train/a", "locomo", "train", "memory", "short", gold_answers=["PRIVATE_GOLD"]),
                  Example("locomo/train/b", "locomo", "train", "memory", "longer question")],
        "validation": [Example("other/validation/a", "other", "validation", "memory", "check")],
    }
    return config, store, examples


def run_report(config, store, examples, tokenizer=None):
    tokenizer = tokenizer or CharacterTokenizer()
    with patch("agmem.full_directory_preflight.load_tokenizer", return_value=tokenizer) as loader:
        result = initial_context_report(config, store, examples)
    loader.assert_called_once_with(config["models"]["reader_path"])
    return result, tokenizer


def test_counts_each_query_with_chat_template_and_reuses_only_public_memory_view(tmp_path):
    config, store, examples = setup_case(tmp_path)
    before = deepcopy((config, store.hierarchical_bank.return_value, examples))
    report, tokenizer = run_report(config, store, examples)
    assert (config, store.hierarchical_bank.return_value, examples) == before
    store.hierarchical_bank.assert_called_once_with("memory", "frozen-v1")
    store.close.assert_not_called()
    assert len(tokenizer.messages) == 3
    states = [json.loads(messages[1]["content"]) for messages in tokenizer.messages]
    assert states[0]["query"] != states[1]["query"]
    assert all([parent["id"] for parent in state["parent_directory"]] == ["p0001", "p0002"]
               for state in states)
    assert all(len(state["child_directories"]) == 2 and state["retrieved_evidence"] == [] for state in states)
    rendered = json.dumps(tokenizer.messages)
    assert not any(secret in rendered for secret in ("DO_NOT_RENDER_PARENT", "DO_NOT_RENDER_CHILD", "PRIVATE_RAW", "PRIVATE_GOLD"))
    assert report["queries"] == 3 and report["memories"] == 1
    assert report["native_context_limit"] == report["context_limit"] == 100000
    assert report["output_token_reserve"] == 16 and report["overflow_count"] == 0
    assert report["by_source"]["locomo"]["queries"] == 2
    assert report["by_split"]["validation"]["queries"] == 1
    assert report["max_sample_id"] == "locomo/train/b"
    assert report["max"] - report["min"] == len("longer question") - len("short")
    assert report["p95"] == report["max"] and report["gpu_memory_checked"] is False
    directories = sorted(store.hierarchical_bank.return_value[0], key=lambda row: row["order"])
    view = {"parents": [DirectoryReader._parent_view(row) for row in directories],
            "child_directories": [{"parent_id": row["id"],
                                   "children": [DirectoryReader._child_view(child) for child in row["children"]]}
                                  for row in directories]}
    actual = full_directory_reader_messages(examples["train"][1], view, [])
    assert report["max"] == len(chat_token_ids(CharacterTokenizer(), actual))


def test_output_reserve_and_explicit_context_limit_are_checked_at_exact_boundary(tmp_path):
    config, store, examples = setup_case(tmp_path)
    report, _ = run_report(config, store, examples)
    config["reader"]["max_context_tokens"] = report["max"] + 16
    exact, _ = run_report(config, store, examples)
    assert exact["context_limit"] == report["max"] + 16
    assert exact["native_context_limit"] == 100000
    config["reader"]["max_context_tokens"] -= 1
    with patch("agmem.full_directory_preflight.load_tokenizer", return_value=CharacterTokenizer()), \
         pytest.raises(CapacityError, match="No truncation performed") as caught:
        initial_context_report(config, store, examples)
    assert caught.value.report["overflow_sample_ids"] == ["locomo/train/b"]
    assert caught.value.report["overflow_count"] == 1


def test_native_limit_is_used_when_reader_limit_is_null(tmp_path):
    config, store, examples = setup_case(tmp_path, native_limit=32)
    with patch("agmem.full_directory_preflight.load_tokenizer", return_value=CharacterTokenizer()), \
         pytest.raises(CapacityError) as caught:
        initial_context_report(config, store, examples)
    assert caught.value.report["context_limit"] == 32
    assert caught.value.report["overflow_count"] == 3


def test_cached_projection_matches_the_environment_initial_input_exactly(tmp_path):
    from agmem.full_directory_env import FullDirectoryMemoryEnvironment
    from tests.test_hierarchical_environment import Store

    config, _, _ = setup_case(tmp_path)
    store = Store()
    store.get_metadata = Mock(return_value="frozen_v1")
    example = Example("locomo/train/a", "locomo", "train", "mem_fixture", "Question?",
                      observation="Public observation", options={"A": "Choice"},
                      gold_answers=["PRIVATE_GOLD"])
    report, tokenizer = run_report(config, store, {"train": [example]})
    episode = FullDirectoryMemoryEnvironment(store, "frozen_v1", count_tokens=len)._episode(example, 0, 0)
    try:
        messages, _ = next(episode)
    finally:
        episode.close()
    assert tokenizer.messages == [messages]
    assert report["max"] == len(chat_token_ids(CharacterTokenizer(), messages))


@pytest.mark.parametrize("path,value", [("formula", "judge_f1_cost_v1"), ("prompt", "hierarchical_v1"),
                                      ("data", "flat_v1"), ("ids", 1), ("ids", True), ("context", 0)])
def test_only_full_protocol_and_valid_context_settings_are_accepted(tmp_path, path, value):
    config, store, examples = setup_case(tmp_path)
    section, key = {"formula": ("reward", "formula"), "prompt": ("reader", "prompt_version"),
                    "data": ("data", "directory_protocol"), "ids": ("retrieval", "max_ids_per_call"),
                    "context": ("reader", "max_context_tokens")}[path]
    config[section][key] = value
    with patch("agmem.full_directory_preflight.load_tokenizer", side_effect=AssertionError("Must reject before tokenizer")), \
         pytest.raises(ValueError):
        initial_context_report(config, store, examples)
