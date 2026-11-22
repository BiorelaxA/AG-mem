"""Boundary-preserving Writer tests; no model calls or GPU required."""

from copy import deepcopy
import json
import unittest

from agmem.core import InfrastructureError, canonical, digest
from agmem.hierarchical.adjacent_writer import AdjacentDirectoryWriter, validate_spans
from agmem.hierarchical.records import make_session
from agmem.hierarchical.writer import chat_token_count, public_session, validate_build_result


class WordTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(canonical(messages)) + [1, 2, 3]


TOKENIZER = WordTokenizer()


def session(n=4, words=3):
    records = [{"record_id": f"p1:r{i}", "seq": i, "kind": "message",
                "speaker": "user" if i % 2 else "assistant", "call_id": None,
                "text": " ".join([f"fact{i}"] * words), "attachments": []}
               for i in range(1, n + 1)]
    return make_session(bank_id="private_bank", parent_id="p1", order=1, records=records,
                        time={"raw": "2020-01-01", "iso": "2020-01-01", "timezone": None},
                        participants=["user", "assistant"],
                        count_tokens=lambda text: len(TOKENIZER.encode(text)))


def settings(**changes):
    return {"context_tokens": 8192, "output_tokens": 32, "max_output_tokens": 128,
            "safety_margin": 16, **changes}


class Transport:
    def __init__(self, behavior=None):
        self.behavior = behavior
        self.calls, self.accepted, self.rejected = [], [], []

    def __call__(self, messages, budget, schema, stage):
        data = json.loads(messages[-1]["content"])
        self.calls.append((deepcopy(messages), budget, deepcopy(schema), stage, data))
        if self.behavior:
            result = self.behavior(self, data, budget, stage)
            if result is not None:
                return result
        if stage.startswith("adjacent_boundary"):
            result = {"decision": "merge", "reason": "A direct clarification of the same request.",
                      "evidence_blocks": [data["left"]["end_block"], data["right"]["start_block"]]}
        else:
            result = {"title": "Source topic", "content": "Reported source facts and choices."}
            if stage.startswith("adjacent_child"):
                result["entities"] = []
        return {"text": canonical(result), "finish_reason": "stop"}

    def accept(self, *args, **kwargs):
        self.accepted.append((args, kwargs))

    def reject(self, *args, **kwargs):
        self.rejected.append((args, kwargs))


class AdjacentWriterTests(unittest.TestCase):
    def test_children_are_generated_before_parent_and_parent_receives_all_source(self):
        source = session(4)
        original = deepcopy(source)
        transport = Transport()
        result = AdjacentDirectoryWriter(TOKENIZER, settings(), transport).build(
            source, [{"start_block": 1, "end_block": 2}, {"start_block": 3, "end_block": 4}])
        self.assertEqual(result["status"], "complete")
        self.assertEqual([c[3] for c in transport.calls], ["adjacent_child", "adjacent_child", "adjacent_parent"])
        first, second, parent = [c[4] for c in transport.calls]
        self.assertEqual([b["index"] for b in first["blocks"]], [1, 2])
        self.assertNotIn("fact3", canonical(first))
        self.assertEqual([b["index"] for b in second["blocks"]], [3, 4])
        self.assertEqual([b["index"] for b in second["context_blocks"]], [1, 2])
        self.assertEqual([b["index"] for b in parent["blocks"]], [1, 2, 3, 4])
        self.assertEqual(parent["context_blocks"], [])
        self.assertEqual([(c["start_block"], c["end_block"]) for c in parent["children"]], [(1, 2), (3, 4)])
        self.assertEqual([c["end_block"] for c in result["children"]], [2, 4])
        self.assertEqual(source, original)
        for _, _, _, _, data in transport.calls:
            for private in ("private_bank", "p1:r", "record_id", "source_hash", "call_id"):
                self.assertNotIn(private, canonical(data))
        validate_build_result(result, source, TOKENIZER, settings())

    def test_invalid_spans_fail_before_any_generation(self):
        bad = [[], [{"start_block": 2, "end_block": 4}],
               [{"start_block": 1, "end_block": 2}, {"start_block": 2, "end_block": 4}],
               [{"start_block": 1, "end_block": 5}], [{"start_block": True, "end_block": 4}]]
        for spans in bad:
            transport = Transport()
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                AdjacentDirectoryWriter(TOKENIZER, settings(), transport).build(session(), spans)
            self.assertFalse(transport.calls)

    def test_child_repair_cannot_change_fixed_boundaries(self):
        def behavior(transport, data, budget, stage):
            if len(transport.calls) == 1:
                return {"text": canonical({"title": "Wrong shape", "content": "Content",
                                           "entities": [], "end_block": 4}), "finish_reason": "stop"}
        transport = Transport(behavior)
        result = AdjacentDirectoryWriter(TOKENIZER, settings(), transport).build(
            session(), [{"start_block": 1, "end_block": 2}, {"start_block": 3, "end_block": 4}])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(transport.calls[1][3], "adjacent_child_repair")
        repaired = deepcopy(transport.calls[1][4]); repaired.pop("repair")
        self.assertEqual(transport.calls[0][4], repaired)
        self.assertEqual([c["end_block"] for c in result["children"]], [2, 4])
        self.assertTrue(all(kw["status"] == "complete" for _, kw in transport.accepted))

    def test_length_expansion_never_accepts_truncated_text(self):
        def behavior(transport, data, budget, stage):
            if stage == "adjacent_child" and budget < 128:
                return {"text": '{"title":', "finish_reason": "length"}
        transport = Transport(behavior)
        result = AdjacentDirectoryWriter(TOKENIZER, settings(), transport).build(
            session(2), [{"start_block": 1, "end_block": 2}])
        self.assertEqual([c[1] for c in transport.calls[:3]], [32, 64, 128])
        self.assertEqual(result["stats"]["output_expansions"], 2)
        self.assertEqual(result["status"], "complete")

    def test_summary_failure_marks_fallback_without_resegmenting(self):
        def behavior(transport, data, budget, stage):
            if stage.startswith("adjacent_child"):
                return {"text": "bad JSON", "finish_reason": "stop"}
        transport = Transport(behavior)
        result = AdjacentDirectoryWriter(TOKENIZER, settings(), transport).build(
            session(17), [{"start_block": 1, "end_block": 17}])
        self.assertEqual(result["status"], "fallback")
        self.assertEqual([c["end_block"] for c in result["children"]], [17])
        self.assertIn("unavailable", result["children"][0]["content"])
        self.assertEqual(len(transport.calls[-1][4]["blocks"]), 17)
        self.assertEqual(result["stats"]["semantic_repairs"], 2)

    def test_parent_uses_complete_source_windows_when_full_session_does_not_fit(self):
        source = session(8, words=180)
        spans = [{"start_block": i, "end_block": i} for i in range(1, 9)]
        transport = Transport()
        result = AdjacentDirectoryWriter(TOKENIZER, settings(context_tokens=900), transport).build(source, spans)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["mode"], "windowed")
        self.assertEqual([c["end_block"] for c in result["children"]], list(range(1, 9)))
        parent_windows = [c for c in transport.calls if c[3] == "adjacent_parent"]
        self.assertGreater(len(parent_windows), 1)
        indices = [b["index"] for c in parent_windows for b in c[4]["blocks"]]
        self.assertEqual(indices, list(range(1, 9)))
        self.assertEqual(transport.calls[-1][3], "adjacent_parent_rollup")
        for messages, maximum, _, _, _ in transport.calls:
            self.assertLessEqual(chat_token_count(TOKENIZER, messages) + maximum + 16, 900)

    def test_large_child_reads_windows_without_altering_its_binding(self):
        source = session(6, words=150)
        transport = Transport()
        result = AdjacentDirectoryWriter(TOKENIZER, settings(context_tokens=800), transport).build(
            source, [{"start_block": 1, "end_block": 6}])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["mode"], "windowed")
        self.assertEqual([c["end_block"] for c in result["children"]], [6])
        source_calls = [c for c in transport.calls if c[3] == "adjacent_child"]
        self.assertEqual([b["index"] for c in source_calls for b in c[4]["blocks"]], list(range(1, 7)))
        self.assertTrue(any(c[3] == "adjacent_child_rollup" for c in transport.calls))

    def test_indivisible_oversized_block_fails_without_truncation(self):
        source = session(1, words=10000); original = deepcopy(source)
        transport = Transport()
        with self.assertRaisesRegex(ValueError, "oversized_block"):
            AdjacentDirectoryWriter(TOKENIZER, settings(), transport).build(
                source, [{"start_block": 1, "end_block": 1}])
        self.assertEqual(source, original)
        self.assertFalse(transport.calls)

    def test_empty_and_infrastructure_failure_remain_distinct(self):
        transport = Transport()
        result = AdjacentDirectoryWriter(TOKENIZER, settings(), transport).build(session(0), [])
        self.assertEqual((result["status"], result["children"]), ("empty", []))
        self.assertFalse(transport.calls)
        def fail(*args):
            raise InfrastructureError("transport failure")
        with self.assertRaises(InfrastructureError):
            AdjacentDirectoryWriter(TOKENIZER, settings(), fail).build(
                session(2), [{"start_block": 1, "end_block": 2}])

    def test_boundary_review_can_precede_build_and_keeps_source_private(self):
        source = session(4)
        transport = Transport()
        writer = AdjacentDirectoryWriter(TOKENIZER, settings(), transport)
        decision = writer.review_boundary({"session": source, "left": {"start_block": 1, "end_block": 2},
                                           "right": {"start_block": 3, "end_block": 4},
                                           "score": .5, "cohesion": .4})
        self.assertEqual(decision["decision"], "merge")
        self.assertEqual(decision["evidence_blocks"], [2, 3])
        self.assertTrue(decision["available"])
        result = writer.build(source, [{"start_block": 1, "end_block": 4}])
        self.assertEqual(result["stats"]["boundary_reviews"], 1)
        self.assertEqual(result["stats"]["calls"], 3)
        data = transport.calls[0][4]
        self.assertNotIn("private_bank", canonical(data))
        self.assertNotIn("score", data)  # The model independently reviews the source.

    def test_boundary_rejects_unsupplied_evidence_and_fails_conservatively(self):
        def bad(transport, data, budget, stage):
            return {"text": canonical({"decision": "merge", "reason": "Something happens elsewhere.",
                                       "evidence_blocks": [100]}), "finish_reason": "stop"}
        transport = Transport(bad)
        writer = AdjacentDirectoryWriter(TOKENIZER, settings(), transport)
        decision = writer.review_boundary({"session": session(4), "left": {"start_block": 1, "end_block": 2},
                                           "right": {"start_block": 3, "end_block": 4}})
        self.assertEqual(decision["decision"], "keep_boundary")
        self.assertFalse(decision["available"])
        self.assertTrue(decision["review_unavailable"])
        self.assertEqual(writer.stats["boundary_review_failures"], 1)
        self.assertEqual(len(transport.calls), 3)

    def test_boundary_context_overflow_is_explicitly_unavailable(self):
        transport = Transport()
        writer = AdjacentDirectoryWriter(TOKENIZER, settings(), transport)
        decision = writer.review_boundary({"session": session(2, words=10000),
                                           "left": {"start_block": 1, "end_block": 1},
                                           "right": {"start_block": 2, "end_block": 2}})
        self.assertEqual(decision["decision"], "keep_boundary")
        self.assertFalse(decision["available"])
        self.assertTrue(decision["review_unavailable"])
        self.assertEqual(writer.stats["boundary_review_failures"], 1)
        self.assertFalse(transport.calls)

    def test_oversized_repair_response_can_be_shortened_but_source_cannot(self):
        def bad(transport, data, budget, stage):
            if len(transport.calls) == 1:
                return {"text": "invalid " * 10000, "finish_reason": "stop"}
        transport = Transport(bad)
        AdjacentDirectoryWriter(TOKENIZER, settings(context_tokens=1200), transport).build(
            session(2, words=100), [{"start_block": 1, "end_block": 2}])
        self.assertEqual(transport.calls[0][4]["blocks"], transport.calls[1][4]["blocks"])
        self.assertLess(len(transport.calls[1][4]["repair"]["previous_response"]), 80000)


if __name__ == "__main__":
    unittest.main()
