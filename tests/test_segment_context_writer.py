"""Complete prior-segment context is immutable, bounded and source-grounded."""
from copy import deepcopy
import unittest

from agmem.core import CapacityError, canonical
from agmem.hierarchical.adjacent_writer import AdjacentDirectoryWriter
from agmem.hierarchical.segment_context_writer import (
    BOUNDARY_PROMPT, CHILD_PROMPT, SegmentContextDirectoryWriter, context_policy)
from agmem.hierarchical.writer import chat_token_count, validate_build_result
from tests.test_adjacent_writer import TOKENIZER, Transport, session, settings


def new_settings(**changes):
    return settings(context_blocks=0, **changes)


class SegmentContextWriterTests(unittest.TestCase):
    def test_variable_length_previous_three_are_segments_not_blocks(self):
        source = session(10)
        original = deepcopy(source)
        spans = [{"start_block": 1, "end_block": 2}, {"start_block": 3, "end_block": 3},
                 {"start_block": 4, "end_block": 7}, {"start_block": 8, "end_block": 9},
                 {"start_block": 10, "end_block": 10}]
        transport = Transport()
        result = SegmentContextDirectoryWriter(TOKENIZER, new_settings(), transport).build(source, spans)
        children = [call[4] for call in transport.calls if call[3] == "adjacent_child"]
        self.assertEqual([len(x["context_segments"]) for x in children], [0, 1, 2, 3, 3])
        self.assertEqual([b["index"] for b in children[-1]["blocks"]], [10])
        self.assertEqual([[b["index"] for b in x["blocks"]] for x in children[-1]["context_segments"]],
                         [[3], [4, 5, 6, 7], [8, 9]])
        self.assertNotIn("fact1 fact1 fact1", canonical(children[-1]))
        self.assertNotIn("context_blocks", children[-1])
        for i, child in enumerate(children):
            start, end = spans[i]["start_block"], spans[i]["end_block"]
            self.assertEqual([b["index"] for b in child["blocks"]], list(range(start, end + 1)))
            self.assertTrue(all(b["index"] < start for s in child["context_segments"] for b in s["blocks"]))
            self.assertEqual(child["session_metadata"]["participants"], ["user", "assistant"])
            self.assertEqual(child["session_metadata"]["time"]["raw"], "2020-01-01")
            self.assertEqual(child["context_policy"], context_policy())
            self.assertNotIn("Reported source facts", canonical(child))  # No previous generated summaries.
            for private in ("private_bank", "p1:r", "record_id", "source_hash", "call_id"):
                self.assertNotIn(private, canonical(child))
        parent = transport.calls[-1]
        self.assertEqual(parent[3], "adjacent_parent")
        self.assertNotIn("context_segments", parent[4])
        self.assertEqual([b["index"] for b in parent[4]["blocks"]], list(range(1, 11)))
        self.assertEqual([c["end_block"] for c in parent[4]["children"]], [2, 3, 7, 9, 10])
        self.assertEqual([c["end_block"] for c in result["children"]], [2, 3, 7, 9, 10])
        self.assertEqual(source, original)
        validate_build_result(result, source, TOKENIZER, new_settings())

    def test_capacity_preflight_does_not_drop_context_or_split_current_segment(self):
        source = session(7, words=80)
        spans = [{"start_block": 1, "end_block": 3}, {"start_block": 4, "end_block": 7}]
        probe = SegmentContextDirectoryWriter(TOKENIZER, new_settings(), Transport())
        probe.prepare(source)
        probe._fixed_spans = deepcopy(spans)
        full = probe._source_payload(4, 7, "child", [])
        reservation = probe.settings["output_tokens"] + probe.settings["safety_margin"]
        limit = chat_token_count(TOKENIZER, probe._entry_messages(full, "child")) + reservation - 1
        transport = Transport()
        with self.assertRaisesRegex(CapacityError, "child_segment_context_exceeds_capacity"):
            SegmentContextDirectoryWriter(TOKENIZER, new_settings(context_tokens=limit), transport).build(source, spans)
        self.assertFalse(transport.calls)  # Later-child overflow detected before first child generation.

    def test_parent_may_window_while_every_child_keeps_complete_prior_segments(self):
        source = session(9, words=100)
        spans = [{"start_block": i, "end_block": i} for i in range(1, 10)]
        transport = Transport()
        result = SegmentContextDirectoryWriter(
            TOKENIZER, new_settings(context_tokens=1100), transport).build(source, spans)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["mode"], "windowed")
        children = [c for c in transport.calls if c[3] == "adjacent_child"]
        self.assertEqual(len(children), 9)
        self.assertTrue(all(len(c[4]["blocks"]) == 1 for c in children))
        self.assertEqual([len(c[4]["context_segments"]) for c in children], [0, 1, 2, 3, 3, 3, 3, 3, 3])
        self.assertFalse(any(c[3] == "adjacent_child_rollup" for c in transport.calls))
        self.assertTrue(any(c[3] == "adjacent_parent_rollup" for c in transport.calls))
        self.assertEqual([c["end_block"] for c in result["children"]], list(range(1, 10)))

    def test_repair_keeps_full_raw_context_and_boundaries(self):
        def bad_once(transport, data, budget, stage):
            if stage == "adjacent_child" and data["write_range"]["start"] == 3:
                return {"text": "invalid JSON", "finish_reason": "stop"}
        transport = Transport(bad_once)
        result = SegmentContextDirectoryWriter(TOKENIZER, new_settings(), transport).build(
            session(4), [{"start_block": 1, "end_block": 2}, {"start_block": 3, "end_block": 4}])
        original = next(c[4] for c in transport.calls if c[3] == "adjacent_child" and c[4]["write_range"]["start"] == 3)
        repaired = deepcopy(next(c[4] for c in transport.calls if c[3] == "adjacent_child_repair"))
        repaired.pop("repair")
        self.assertEqual(original, repaired)
        self.assertEqual(result["status"], "complete")

    def test_truncated_generation_never_changes_the_source_range_or_context(self):
        def truncate(transport, data, budget, stage):
            if stage.startswith("adjacent_child"):
                return {"text": '{"title":', "finish_reason": "length"}
        transport = Transport(truncate)
        result = SegmentContextDirectoryWriter(TOKENIZER, new_settings(), transport).build(
            session(4), [{"start_block": 1, "end_block": 2}, {"start_block": 3, "end_block": 4}])
        self.assertEqual(result["status"], "fallback")
        self.assertEqual([c["end_block"] for c in result["children"]], [2, 4])
        second = [c[4] for c in transport.calls if c[3] == "adjacent_child" and c[4]["write_range"]["start"] == 3]
        self.assertEqual(len(second), 3)
        self.assertTrue(all(value == second[0] for value in second))
        self.assertEqual([s["start_block"] for s in second[0]["context_segments"]], [1])

    def test_boundary_review_uses_v2_before_segment_context_is_bound(self):
        transport = Transport()
        writer = SegmentContextDirectoryWriter(TOKENIZER, new_settings(), transport)
        source = session(4)
        result = writer.review_boundary({"session": source,
            "left": {"start_block": 1, "end_block": 2}, "right": {"start_block": 3, "end_block": 4}})
        self.assertTrue(result["available"])
        self.assertEqual(transport.calls[0][0][0]["content"], BOUNDARY_PROMPT.read_text())
        writer.build(source, [{"start_block": 1, "end_block": 4}])
        self.assertEqual(transport.calls[1][0][0]["content"], CHILD_PROMPT.read_text())

    def test_session_isolation_and_empty_session(self):
        transport = Transport()
        writer = SegmentContextDirectoryWriter(TOKENIZER, new_settings(), transport)
        result = writer.build(session(0), [])
        self.assertEqual((result["status"], result["children"]), ("empty", []))
        self.assertFalse(transport.calls)
        with self.assertRaisesRegex(ValueError, "one session"):
            writer.build(session(1), [{"start_block": 1, "end_block": 1}])

    def test_invalid_context_unit_and_partial_child_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "context_blocks=0"):
            SegmentContextDirectoryWriter(TOKENIZER, settings(), Transport())
        for number in (True, 0, 2, 4):
            with self.subTest(number=number), self.assertRaisesRegex(ValueError, "previous_segments=3"):
                SegmentContextDirectoryWriter(TOKENIZER, new_settings(), Transport(), previous_segments=number)
        writer = SegmentContextDirectoryWriter(TOKENIZER, new_settings(), Transport())
        writer.build(session(4), [{"start_block": 1, "end_block": 4}])
        with self.assertRaisesRegex(ValueError, "one complete fixed segment"):
            writer._source_payload(1, 2, "child", [])

    def test_legacy_writer_still_uses_two_blocks_and_old_prompt(self):
        old_transport, new_transport = Transport(), Transport()
        spans = [{"start_block": 1, "end_block": 3}, {"start_block": 4, "end_block": 4}]
        AdjacentDirectoryWriter(TOKENIZER, settings(), old_transport).build(session(4), spans)
        SegmentContextDirectoryWriter(TOKENIZER, new_settings(), new_transport).build(session(4), spans)
        self.assertEqual([b["index"] for b in old_transport.calls[1][4]["context_blocks"]], [2, 3])
        self.assertNotIn("context_segments", old_transport.calls[1][4])
        self.assertEqual([b["index"] for b in new_transport.calls[1][4]["context_segments"][0]["blocks"]], [1, 2, 3])
        self.assertNotEqual(old_transport.calls[1][0][0]["content"], new_transport.calls[1][0][0]["content"])


if __name__ == "__main__":
    unittest.main()
