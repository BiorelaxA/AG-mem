"""Exercise the test supplement with the real local Writer retry loop offline."""
import json
from unittest.mock import Mock, patch

import httpx
import pytest

from agmem.build_qwen_m_test_missing import TestSupplementWriter as SupplementWriter
from agmem.core import InfrastructureError
from agmem.hierarchical.client import WriterPool
from tests.test_adjacent_writer import TOKENIZER, session, settings


def make_writer(monkeypatch, outcomes):
    transport = Mock()
    transport.post.side_effect = outcomes
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: transport)
    pool = WriterPool(["http://localhost:8000/v1"], 1, {"network_retries": 3}, Mock())
    writer = SupplementWriter(TOKENIZER, settings(context_blocks=1, semantic_retries=0),
                              pool.for_session(session()))
    return writer, transport


def success(child=False):
    entry = {"title": "Topic", "content": "Source evidence"}
    if child:
        entry["entities"] = []
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                  "message": {"content": json.dumps(entry)}}]})


@pytest.mark.parametrize("stage", ["parent", "child"])
def test_exhausted_read_timeouts_only_replace_the_affected_entry(monkeypatch, stage):
    timeouts = [httpx.ReadTimeout("timed out") for _ in range(4)]
    outcomes = ([success(True), success(True), *timeouts] if stage == "parent"
                else [*timeouts, success(True), success()])
    writer, transport = make_writer(monkeypatch, outcomes)
    with patch("agmem.hierarchical.client.time.sleep"):
        result = writer.build(session(), [{"start_block": 1, "end_block": 2},
                                          {"start_block": 3, "end_block": 4}])
    assert transport.post.call_count == 6
    assert result["status"] == "fallback"
    assert result["children"][1]["content"] == "Source evidence"
    assert (result["parent"]["content"] == "Source evidence") == (stage == "child")
    assert (result["children"][0]["content"] == "Source evidence") == (stage == "parent")
    assert any(t.get("reason") == "writer_read_timeout_after_retries"
               and t["stage"] == "adjacent_" + stage for t in result["traces"])


def test_connect_failure_still_propagates(monkeypatch):
    writer, _ = make_writer(monkeypatch, [httpx.ConnectError("offline") for _ in range(4)])
    with patch("agmem.hierarchical.client.time.sleep"), pytest.raises(InfrastructureError):
        writer.build(session(), [{"start_block": 1, "end_block": 4}])
