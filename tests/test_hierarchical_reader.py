"""Exact source binding and two-stage Reader visibility without model inference."""

from copy import deepcopy
import unittest

from agmem.core import canonical, digest
from agmem.hierarchical.binding import bind_directory, directory_version, validate_directory
from agmem.hierarchical.reader import DirectoryReader
from agmem.hierarchical.records import make_session, render_records


def session(number=1, *, empty=False, source_kind="dialogue"):
    pid = f"p{number:04d}"
    header = {"record_id": f"{pid}:h0001", "seq": 1, "kind": "task_instruction",
              "speaker": "user", "call_id": None, "text": "Public instruction", "attachments": []}
    body = [{"record_id": f"{pid}:r{i:04d}", "seq": i + 1, "kind": "message",
             "speaker": "Alice" if i % 2 else "Bob", "call_id": None,
             "text": f"Raw message {number}.{i}; keep literal has_answer text.", "attachments": []}
            for i in range(1, 5)] if not empty else []
    return make_session(bank_id="bank_demo", parent_id=pid, order=number,
                        time={"raw": "2022-01-01", "iso": "2022-01-01T00:00:00", "timezone": None},
                        participants=["Alice", "Bob"], header_records=[header], records=body,
                        source_kind=source_kind, count_tokens=len)


def writer_result(*, empty=False):
    return {"parent": {"title": "Session overview", "content": "Two local topics"},
            "children": [] if empty else [
                {"end_block": 2, "title": "First topic", "content": "The first two messages", "entities": ["Alice"],
                 "start_reason": "session_start"},
                {"end_block": 4, "title": "Second topic", "content": "The final two messages", "entities": ["Alice", "Bob"],
                 "start_reason": "semantic"}],
            "status": "empty" if empty else "complete", "mode": "single_session"}


def directory(source, result=None):
    return bind_directory(source, result or writer_result(empty=not source["blocks"]),
                          {"model": "fixture-qwen", "model_revision": "fixture-revision"},
                          "fixture-character-counter", count_tokens=len)


class SourceBindingTests(unittest.TestCase):
    def test_exact_addresses_and_token_counts_are_program_bound(self):
        source = session()
        result = directory(source)
        validate_directory(result, source, count_tokens=len)
        self.assertEqual(result["entities"], ["Alice", "Bob"])
        self.assertEqual(result["source"]["header_record_ids"], ["p0001:h0001"])
        self.assertEqual(result["source"]["body_token_count"], len(render_records(source["records"])))
        child = result["children"][1]
        self.assertEqual(child["id"], "p0001:c0002")
        self.assertEqual(child["source"], {
            "start_block": 3, "end_block": 4, "start_record_id": "p0001:r0003",
            "end_record_id": "p0001:r0004", "block_count": 2, "record_count": 2,
            "body_token_count": len(render_records(source["records"][2:]))})
        self.assertEqual(result["directory_version"], directory_version(result))

    def test_invalid_or_incomplete_model_boundaries_are_rejected(self):
        for ends in ((2, 3), (2, 2), (0, 4), (True, 4), (2, 5)):
            candidate = writer_result()
            for child, end in zip(candidate["children"], ends):
                child["end_block"] = end
            with self.subTest(ends=ends), self.assertRaises(ValueError):
                directory(session(), candidate)

    def test_rehashed_inexact_range_or_tampered_source_is_rejected(self):
        source = session()
        result = directory(source)
        changed = deepcopy(result)
        changed["children"][0]["source"]["start_record_id"] = "p0001:r0002"
        changed["directory_version"] = directory_version(changed)
        with self.assertRaisesRegex(ValueError, "mapping"):
            validate_directory(changed, source)
        source["records"][0]["text"] = "Changed after committing the directory"
        with self.assertRaisesRegex(ValueError, "hash"):
            validate_directory(result, source)

    def test_empty_session_retains_header_but_has_no_body_span(self):
        source = session(empty=True)
        result = directory(source)
        self.assertEqual(result["children"], [])
        self.assertEqual(result["build"]["status"], "empty")
        self.assertEqual(result["source"]["header_record_ids"], ["p0001:h0001"])
        self.assertIsNone(result["source"]["start_record_id"])
        self.assertEqual(result["source"]["body_token_count"], 0)


class HierarchicalReaderTests(unittest.TestCase):
    def setUp(self):
        self.sessions = [session(1), session(2)]
        self.directories = [directory(s) for s in self.sessions]

    def reader(self, retention="cumulative", **kwargs):
        return DirectoryReader(self.directories, self.sessions, count_tokens=len, retention=retention, **kwargs)

    def test_initial_view_has_only_parents_and_public_task_context(self):
        reader = self.reader()
        view = reader.render("Question?", "Public observation", reader.initial_state())
        self.assertEqual(view["query"], "Question?")
        self.assertEqual(view["observation"], "Public observation")
        self.assertEqual([p["id"] for p in view["parents"]], ["p0001", "p0002"])
        self.assertEqual(view["child_directories"], [])
        self.assertEqual(view["retrieved_children"], [])
        self.assertEqual(view["evidence"], [])
        rendered = canonical(view)
        for forbidden in ("Raw message", "p0001:c0001", "source_hash", "snapshot_id", "writer_revision"):
            self.assertNotIn(forbidden, rendered)

    def test_parent_expands_without_raw_text_and_hidden_children_cannot_be_guessed(self):
        reader = self.reader()
        initial = reader.initial_state()
        for ids in (["p0001:c0001"], ["p0001", "p0001:c0001"]):
            result = reader.read(ids, initial, 100000)
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"]["code"], "child_not_visible")
            self.assertEqual(result["state"], initial)
        expanded = reader.read(["p0001"], initial, 100000)
        self.assertTrue(expanded["ok"])
        self.assertEqual(expanded["result"]["records"], [])
        self.assertEqual(expanded["state"]["retrieved_child_ids"], [])
        self.assertNotIn("Raw message", canonical(expanded["result"]))
        view = reader.render("Q", "O", expanded["state"])
        self.assertEqual(len(view["parents"]), 2)
        self.assertEqual([c["id"] for c in view["child_directories"][0]["children"]],
                         ["p0001:c0001", "p0001:c0002"])
        self.assertEqual(initial["expanded_parent_ids"], [])

    def test_child_returns_entry_header_context_and_exact_body(self):
        reader = self.reader()
        state = reader.read(["p0001"], reader.initial_state(), 100000)["state"]
        result = reader.read(["p0001:c0002"], state, 100000)
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["selected_children"][0]["id"], "p0001:c0002")
        records = result["result"]["records"]
        self.assertEqual([r["record_id"] for r in records],
                         ["p0001:h0001", "p0001:r0002", "p0001:r0003", "p0001:r0004"])
        self.assertEqual([r["introduced_by"][0]["role"] for r in records], ["header", "context", "core", "core"])
        self.assertEqual(records[2]["text"], self.sessions[0]["records"][2]["text"])
        self.assertEqual(result["cost"]["returned_tokens"], len(canonical(result["result"])))

    def test_overlapping_children_deduplicate_and_preserve_all_node_attributions(self):
        reader = self.reader()
        state = reader.read(["p0001"], reader.initial_state(), 100000)["state"]
        first = reader.read(["p0001:c0001"], state, 100000)
        second = reader.read(["p0001:c0002"], first["state"], 100000)
        view = reader.render("Q", "O", second["state"])
        self.assertEqual(len(view["evidence"]), 5)
        overlap = next(r for r in view["evidence"] if r["record_id"] == "p0001:r0002")
        self.assertEqual(overlap["introduced_by"], [
            {"node_id": "p0001:c0001", "role": "core"}, {"node_id": "p0001:c0002", "role": "context"}])
        self.assertEqual(second["new_record_ids"], ["p0001:r0003", "p0001:r0004"])
        repeated = reader.read(["p0001:c0002"], second["state"], 100000)
        self.assertEqual(repeated["new_record_ids"], [])
        self.assertEqual(repeated["state"], second["state"])

    def test_latest_list_policy_preserves_all_parents_and_acquired_evidence(self):
        reader = self.reader(retention="latest")
        state = reader.read(["p0001"], reader.initial_state(), 100000)["state"]
        state = reader.read(["p0001:c0001"], state, 100000)["state"]
        state = reader.read(["p0002"], state, 100000)["state"]
        view = reader.render("Q", "O", state)
        self.assertEqual(len(view["parents"]), 2)
        self.assertEqual([r["parent_id"] for r in view["child_directories"]], ["p0002"])
        self.assertEqual([r["id"] for r in view["retrieved_children"]], ["p0001:c0001"])
        self.assertEqual(len(view["evidence"]), 3)
        self.assertEqual(reader.read(["p0001:c0002"], state, 100000)["error"]["code"], "child_not_visible")
        self.assertTrue(reader.read(["p0001:c0001"], state, 100000)["ok"])

    def test_cumulative_policy_and_selection_order_preserve_source_order(self):
        reader = self.reader()
        state = reader.read(["p0002", "p0001"], reader.initial_state(), 100000)["state"]
        result = reader.read(["p0002:c0001", "p0001:c0002"], state, 100000)
        self.assertTrue(result["ok"])
        self.assertEqual(result["state"]["display_parent_ids"], ["p0001", "p0002"])
        self.assertEqual([r["parent_id"] for r in result["result"]["records"]],
                         ["p0001"] * 4 + ["p0002"] * 3)

    def test_budget_rejection_is_atomic_and_does_not_expose_partial_content(self):
        reader = self.reader()
        state = reader.initial_state()
        result = reader.read(["p0001", "p0002"], state, 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["state"], state)
        self.assertGreater(result["error"]["required_tokens"], 1)
        self.assertNotIn("result", result)
        state = reader.read(["p0001"], state, 100000)["state"]
        result = reader.read(["p0001:c0001"], state, 1)
        self.assertEqual(result["state"], state)
        self.assertNotIn("Raw message", canonical(result))

    def test_cross_bank_stale_versions_unknown_ids_and_frontier_are_rejected(self):
        reader = self.reader(visible_frontier=1)
        state = reader.initial_state()
        self.assertEqual(len(reader.render("Q", "O", state)["parents"]), 1)
        for ids, extras, code in [
            (["p0001"], {"bank_id": "different-bank"}, "bank_mismatch"),
            (["p0001"], {"directory_versions": {}}, "directory_version_mismatch"),
            (["p0002"], {}, "beyond_visible_frontier"),
            (["p0002:c0001"], {}, "beyond_visible_frontier"),
            (["unknown"], {}, "unknown_id"),
        ]:
            result = reader.read(ids, state, 100000, **extras)
            self.assertEqual(result["error"]["code"], code)
            self.assertEqual(result["state"], state)

    def test_changed_source_and_directory_fail_integrity_checks(self):
        reader = self.reader()
        reader._sessions["p0001"]["records"][0]["text"] = "uncommitted source"
        result = reader.read(["p0001"], reader.initial_state(), 100000)
        self.assertEqual(result["error"]["code"], "source_or_directory_integrity_error")
        with self.assertRaisesRegex(ValueError, "bank"):
            source = session(2)
            source["bank_id"] = "second-bank"
            source["source_hash"] = digest({k: v for k, v in source.items() if k != "source_hash"})
            DirectoryReader([self.directories[0], directory(source)], [self.sessions[0], source],
                            count_tokens=len, retention="cumulative")

    def test_stop_does_not_expand_or_read_anything(self):
        reader = self.reader()
        initial = reader.initial_state()
        stopped = reader.stop(initial)
        self.assertTrue(stopped["state"]["stopped"])
        self.assertFalse(initial["stopped"])
        self.assertEqual(stopped["state"]["retrieved_child_ids"], [])
        self.assertEqual(reader.read(["p0001"], stopped["state"], 100000)["error"]["code"], "reader_stopped")
        self.assertEqual(reader.render("Q", "O", stopped["state"])["evidence"], [])

    def test_modality_projection_preserves_snapshot_but_can_hide_auxiliary_caption(self):
        source = session()
        source["records"][0]["attachments"] = [
            {"kind": "image", "ref": "https://example.invalid/image.jpg", "caption": "Auxiliary image description"}]
        source = make_session(**{key: source[key] for key in
                                 ("bank_id", "parent_id", "order", "records", "time", "participants", "header_records")},
                              count_tokens=len)
        reader = DirectoryReader([directory(source)], [source], count_tokens=len, retention="cumulative",
                                 modality_policy="references_only")
        state = reader.read(["p0001"], reader.initial_state(), 100000)["state"]
        result = reader.read(["p0001:c0001"], state, 100000)
        record = next(r for r in result["result"]["records"] if r["record_id"] == "p0001:r0001")
        self.assertIsNone(record["attachments"][0]["caption"])
        self.assertEqual(source["records"][0]["attachments"][0]["caption"], "Auxiliary image description")
        self.assertEqual(result["cost"]["returned_tokens"], len(canonical(result["result"])))


if __name__ == "__main__":
    unittest.main()
