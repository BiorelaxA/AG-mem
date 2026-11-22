"""Parent/child Writer contracts with deterministic transports, never model calls."""

from copy import deepcopy
import json
import unittest

from jsonschema import ValidationError

from agmem.core import InfrastructureError, canonical, digest
from agmem.hierarchical.records import make_session
from agmem.hierarchical.writer import (DirectoryWriter, chat_token_count, public_session,
                                       validate_build_result, validate_schema,
                                       validate_writer_output, writer_settings)


class WordTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(canonical(messages)) + [1, 2, 3]


TOKENIZER = WordTokenizer()


def session(n=3, words=3):
    records = [{"record_id": f"p1:r{i}", "seq": i, "kind": "message", "speaker": "user",
                "call_id": None, "text": " ".join([f"fact{i}"] * words), "attachments": []}
               for i in range(1, n + 1)]
    return make_session(bank_id="private_bank", parent_id="p1", order=1, records=records,
                        time={"raw": "2020-01-01", "iso": "2020-01-01", "timezone": None},
                        participants=["user"], count_tokens=lambda text: len(TOKENIZER.encode(text)))


def payload(messages):
    text = messages[-1]["content"]
    found = []
    for index, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict) and ("write_range" in value or "ordered_inputs" in value):
            found.append(value)
    return found[-1]


def output(data, schema=None):
    if "ordered_inputs" in data:
        return {"title": "Session overview", "content": "Ordered source topics."}
    ends = [data["write_range"]["end"]]
    prefix = (schema or {}).get("properties", {}).get("children", {}).get("prefixItems")
    if prefix:
        ends = [item["properties"]["end_block"]["const"] for item in prefix]
    return {"parent": {"title": "Session overview", "content": "Source topics."},
            "children": [{"end_block": end, "title": "Topic", "content": "Reported source fact.", "entities": []}
                         for end in ends]}


class Transport:
    def __init__(self, behavior=None):
        self.calls, self.accepted, self.rejected = [], [], []
        self.behavior = behavior

    def __call__(self, messages, maximum, schema, stage):
        data = payload(messages)
        self.calls.append((deepcopy(messages), maximum, deepcopy(schema), stage, data))
        if self.behavior is not None:
            special = self.behavior(self, messages, maximum, schema, stage, data)
            if special is not None:
                return special
        return {"text": canonical(output(data, schema)), "finish_reason": "stop", "usage": {}}

    def accept(self, *args, **kwargs):
        self.accepted.append((args, kwargs))

    def reject(self, *args, **kwargs):
        self.rejected.append((args, kwargs))


class HierarchicalWriterTests(unittest.TestCase):
    def settings(self, **changes):
        return {"context_tokens": 8192, "output_tokens": 32, "max_output_tokens": 128,
                "safety_margin": 16, **changes}

    def test_single_request_outputs_parent_children_and_no_private_addresses(self):
        transport = Transport()
        original = session()
        result = DirectoryWriter(TOKENIZER, self.settings(), transport).build(original)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["mode"], "single_session")
        self.assertEqual(result["children"][0]["start_reason"], "session_start")
        self.assertEqual(len(transport.calls), 1)
        request = canonical(transport.calls[0][4])
        for private in ("private_bank", "p1:r", "record_id", "source_hash", "call_id"):
            self.assertNotIn(private, request)
        self.assertEqual(transport.calls[0][4]["context_blocks"], [])
        self.assertEqual(transport.accepted[0][1]["status"], "complete")

    def test_corrupt_source_snapshot_is_rejected_before_model_request(self):
        original = session()
        original["records"][0]["text"] = "Changed without refreshing the source hash."
        transport = Transport()
        with self.assertRaisesRegex(ValueError, "source hash"):
            DirectoryWriter(TOKENIZER, self.settings(), transport).build(original)
        self.assertEqual(transport.calls, [])

    def test_long_child_summary_uses_300_token_budget_without_fallback(self):
        def long_summary(transport, messages, maximum, schema, stage, data):
            result = output(data, schema)
            result["children"][0]["content"] = " ".join(["detail"] * 300)
            return {"text": canonical(result), "finish_reason": "stop", "usage": {}}
        transport = Transport(long_summary)
        result = DirectoryWriter(TOKENIZER, self.settings(), transport).build(session())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][4]["limits"]["child_content_tokens"], 300)
        invalid = output(transport.calls[0][4])
        invalid["children"][0]["content"] = " ".join(["detail"] * 301)
        with self.assertRaisesRegex(ValueError, "child.content exceeds 300"):
            validate_writer_output(invalid, 1, 3, TOKENIZER, self.settings())

    def test_repair_keeps_original_blocks_and_rejects_incomplete_coverage(self):
        def behavior(transport, messages, maximum, schema, stage, data):
            if len(transport.calls) == 1:
                bad = output(data);bad["children"][0]["end_block"] = 1
                return {"text": canonical(bad), "finish_reason": "stop"}
        transport = Transport(behavior)
        result = DirectoryWriter(TOKENIZER, self.settings(), transport).build(session(4))
        self.assertEqual(result["status"], "complete")
        self.assertEqual([c[3] for c in transport.calls], ["segment", "repair"])
        self.assertEqual(transport.calls[0][4]["blocks"], transport.calls[1][4]["blocks"])
        self.assertEqual(len(transport.rejected), 1)

    def test_length_expands_output_budget_before_retrying_smaller_windows(self):
        def behavior(transport, messages, maximum, schema, stage, data):
            if maximum < 128:
                return {"text": '{"parent":', "finish_reason": "length"}
        transport = Transport(behavior)
        result = DirectoryWriter(TOKENIZER, self.settings(), transport).build(session())
        self.assertEqual([c[1] for c in transport.calls], [32, 64, 128])
        self.assertEqual(result["stats"]["output_expansions"], 2)
        self.assertEqual(result["stats"]["semantic_repairs"], 0)
        self.assertEqual(len(transport.accepted), 1)

    def test_unrecoverable_output_length_shrinks_windows_and_preserves_coverage(self):
        def behavior(transport, messages, maximum, schema, stage, data):
            if "blocks" in data and len(data["blocks"]) > 1:
                return {"text": "{}", "finish_reason": "length"}
        transport = Transport(behavior)
        result = DirectoryWriter(TOKENIZER, self.settings(max_output_tokens=32), transport).build(session(5))
        self.assertEqual(result["mode"], "windowed")
        self.assertEqual(result["status"], "complete")
        self.assertEqual([c["end_block"] for c in result["children"]], [1, 2, 3, 4, 5])
        self.assertEqual([c["start_reason"] for c in result["children"]],
                         ["session_start"] + ["window_limit"] * 4)
        self.assertEqual(transport.calls[-1][3], "rollup")

    def test_input_windows_use_actual_chat_budget_and_read_only_context(self):
        normalized = session(7, words=90)
        probe = DirectoryWriter(TOKENIZER, self.settings(), Transport())
        probe.public = public_session(normalized)
        one = chat_token_count(TOKENIZER, probe._messages(probe._payload(1, 1)))
        settings = self.settings(context_tokens=one + 32 + 16 + 160)
        transport = Transport()
        result = DirectoryWriter(TOKENIZER, settings, transport).build(normalized)
        self.assertEqual(result["mode"], "windowed")
        coverage = []
        for messages, maximum, _, stage, data in transport.calls:
            self.assertLessEqual(chat_token_count(TOKENIZER, messages) + maximum + 16, settings["context_tokens"])
            if stage == "segment":
                coverage += [b["index"] for b in data["blocks"]]
                self.assertLessEqual(len(data["context_blocks"]), 2)
                self.assertTrue(all(b["index"] < data["write_range"]["start"] for b in data["context_blocks"]))
        self.assertEqual(coverage, list(range(1, 8)))

    def test_single_oversized_block_fails_without_calling_or_truncating(self):
        original = session(1, words=10000);saved = deepcopy(original)
        transport = Transport()
        with self.assertRaisesRegex(ValueError, "oversized_block"):
            DirectoryWriter(TOKENIZER, self.settings(), transport).build(original)
        self.assertEqual(transport.calls, [])
        self.assertEqual(original, saved)

    def test_fixed_boundary_model_fallback_is_separate_from_complete(self):
        def behavior(transport, messages, maximum, schema, stage, data):
            if stage in {"segment", "repair"}:
                return {"text": "not JSON", "finish_reason": "stop"}
        transport = Transport(behavior)
        result = DirectoryWriter(TOKENIZER, self.settings(), transport).build(session(19))
        self.assertEqual(result["status"], "fallback")
        self.assertEqual([c["end_block"] for c in result["children"]], [8, 16, 19])
        self.assertEqual([c[3] for c in transport.calls], ["segment", "repair", "repair", "fixed"])
        self.assertEqual(transport.accepted[0][1]["status"], "fallback")

    def test_failed_fixed_summaries_use_honest_placeholders_with_all_blocks(self):
        transport = Transport(lambda *args: {"text": "not JSON", "finish_reason": "stop"})
        result = DirectoryWriter(TOKENIZER, self.settings(), transport).build(session(17))
        self.assertEqual(result["status"], "fallback")
        self.assertEqual([c["end_block"] for c in result["children"]], [8, 16, 17])
        self.assertTrue(all("unavailable" in c["content"] for c in result["children"]))
        self.assertEqual(len(transport.calls), 6)
        self.assertFalse(transport.accepted)

    def test_empty_session_and_transport_failures_are_distinct(self):
        transport = Transport()
        result = DirectoryWriter(TOKENIZER, self.settings(), transport).build(session(0))
        self.assertEqual((result["status"], result["children"]), ("empty", []))
        self.assertFalse(transport.calls)
        def failure(*args):
            raise InfrastructureError("transport retries exhausted")
        with self.assertRaises(InfrastructureError):
            DirectoryWriter(TOKENIZER, self.settings(), failure).build(session())

    def test_bad_schema_boundaries_and_field_budgets_are_rejected(self):
        normal = output({"write_range": {"start": 1, "end": 3}})
        mutations = [lambda x: x["children"][0].update(end_block=True),
                     lambda x: x["children"][0].update(end_block=3.0),
                     lambda x: x["children"][0].update(end_block=2),
                     lambda x: x["children"][0].update(end_block=4),
                     lambda x: x["children"].append(deepcopy(x["children"][0])),
                     lambda x: x["children"][0].update(entities=["A", "A"]),
                     lambda x: x["children"][0].update(entities=[" "]),
                     lambda x: x["children"][0].update(title="x " * 25),
                     lambda x: x["children"][0].update(id="invented")]
        for mutate in mutations:
            value = deepcopy(normal);mutate(value)
            with self.subTest(value=value), self.assertRaises((ValueError, ValidationError)):
                validate_writer_output(value, 1, 3, TOKENIZER, self.settings())

    def test_source_placeholder_literals_and_caption_policy_survive_projection(self):
        normalized = session(2)
        normalized["records"][0]["text"] = "Literal {{FIXED_ENDS_JSON}} and has_answer are ordinary source text."
        normalized["records"][0]["attachments"] = [{"kind": "image", "ref": "photo.jpg", "caption": "a dog"}]
        normalized["source_hash"] = digest({k: v for k, v in normalized.items() if k != "source_hash"})
        transport = Transport()
        DirectoryWriter(TOKENIZER, self.settings(modality_policy="references_only"), transport).build(normalized)
        source = transport.calls[0][4]["blocks"][0]["records"][0]
        self.assertEqual(source["text"], normalized["records"][0]["text"])
        self.assertIsNone(source["attachments"][0]["caption"])
        self.assertEqual(normalized["records"][0]["attachments"][0]["caption"], "a dog")

    def test_repair_may_shorten_failed_response_but_never_original_source(self):
        def behavior(transport, messages, maximum, schema, stage, data):
            if len(transport.calls) == 1:
                return {"text": "bad " * 10000, "finish_reason": "stop"}
        transport = Transport(behavior)
        settings = self.settings(context_tokens=2000)
        DirectoryWriter(TOKENIZER, settings, transport).build(session(3, words=100))
        self.assertEqual(len(transport.calls), 2)
        self.assertLess(len(transport.calls[1][0][-1]["content"]), 40000)
        self.assertEqual(transport.calls[0][4], transport.calls[1][4])
        for messages, maximum, _, _, _ in transport.calls:
            self.assertLessEqual(chat_token_count(TOKENIZER, messages) + maximum + 16, 2000)

    def test_parent_rollup_recursively_reduces_ordered_ranges(self):
        def behavior(transport, messages, maximum, schema, stage, data):
            if "blocks" in data:
                value = output(data, schema)
                value["children"][0]["content"] = "detail " * 120
                return {"text": canonical(value), "finish_reason": "stop"}
        transport = Transport(behavior)
        settings = self.settings(context_tokens=2000, output_tokens=256, max_output_tokens=256)
        result = DirectoryWriter(TOKENIZER, settings, transport).build(session(30, words=500))
        self.assertEqual(result["status"], "complete")
        self.assertGreaterEqual(result["stats"]["rollup_levels"], 1)
        intermediate = [call[4] for call in transport.calls if call[3] == "rollup"
                        and call[4]["scope"] == "contiguous_range"]
        self.assertTrue(intermediate)
        for call in transport.calls:
            if call[3] == "rollup" and call[4]["scope"] == "contiguous_range":
                self.assertNotIn("ONE completed session", call[0][0]["content"])
                self.assertNotIn("entire session", call[0][0]["content"])
        for group in intermediate:
            ranges = group["ordered_inputs"]
            self.assertEqual(group["start_block"], ranges[0]["start_block"])
            self.assertEqual(group["end_block"], ranges[-1]["end_block"])
            for left, right in zip(ranges, ranges[1:]):
                self.assertEqual(left["end_block"] + 1, right["start_block"])
        self.assertEqual(transport.calls[-1][4]["scope"], "session")
        self.assertEqual(transport.calls[-1][4]["end_block"], 30)

    def test_rollup_failure_retains_all_valid_children_and_marks_fallback(self):
        def behavior(transport, messages, maximum, schema, stage, data):
            if "ordered_inputs" in data:
                return {"text": "bad", "finish_reason": "stop"}
            if len(data["blocks"]) > 1:
                return {"text": "{}", "finish_reason": "length"}
        result = DirectoryWriter(TOKENIZER, self.settings(max_output_tokens=32), Transport(behavior)).build(session(4))
        self.assertEqual(result["status"], "fallback")
        self.assertEqual([c["end_block"] for c in result["children"]], [1, 2, 3, 4])
        self.assertIn("unavailable", result["parent"]["content"])

    def test_cached_build_must_retain_start_reason_and_nonempty_parent(self):
        normalized = session()
        result = DirectoryWriter(TOKENIZER, self.settings(), Transport()).build(normalized)
        result["children"][0]["start_reason"] = "semantic"
        with self.assertRaisesRegex(ValueError, "start_reason"):
            validate_build_result(result, normalized, TOKENIZER, self.settings())
        value = output({"write_range": {"start": 1, "end": 3}})
        value["children"][0]["entities"] = [str(i) for i in range(13)]
        with self.assertRaises(ValidationError):
            validate_schema(value, "writer_output")
        validate_schema(value, "writer_output", self.settings(max_entities=13))


if __name__ == "__main__":
    unittest.main()
