"""Opt-in DeepSeek continuation keeps committed work and never repairs an entry."""

from copy import deepcopy
from contextlib import ExitStack
import json
import os
import sqlite3
import unittest
from unittest.mock import Mock, patch

import httpx

from agmem import build_deepseek_no_repair as worker
from agmem import build_deepseek_directories_gpu as launcher
from agmem import import_deepseek_directories as importer
from agmem.build_adjacent_directories import AdjacentBuildStore
from agmem.core import InfrastructureError, canonical, file_digest
from agmem.hierarchical.records import make_session, time_metadata
from tests.test_adjacent_directory_pipeline import Tokenizer
from tests.test_adjacent_directory_pipeline import Encoder
from tests.test_deepseek_directory_pipeline import HTTPPool
from tests import test_import_deepseek_directories as import_fixtures
from tests import test_qwen_continuation_pipeline as continuation_fixtures


class NoReasonHTTPPool(HTTPPool):
    """Keep the shared frozen provider fixture unchanged outside this flow."""

    def respond(self, request):
        response = super().respond(request)
        if response.status_code == 200 and self.stages[-1][0] == "boundary":
            payload = response.json()
            value = json.loads(payload["choices"][0]["message"]["content"])
            del value["reason"]
            payload["choices"][0]["message"]["content"] = canonical(value)
            return httpx.Response(200, json=payload)
        return response


def use_no_reason_boundary(identity):
    identity["boundary_prompt_version"] = worker.BOUNDARY_PROMPT_VERSION
    identity["prompt_hashes"]["boundary.txt"] = file_digest(worker.BOUNDARY_PROMPT)
    identity["boundary_output_fields"] = ["decision", "evidence_blocks"]


class NoRepairWriterTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.session = make_session(
            bank_id="bank_fixture", parent_id="p0001", order=1,
            records=[{"record_id": f"p0001:r{i:04d}", "seq": i, "kind": "message",
                      "speaker": "user", "call_id": None, "text": text, "attachments": []}
                     for i, text in enumerate(("First source fact", "Second source fact"), 1)],
            time=time_metadata("2024-01-01"), participants=["user"],
            count_tokens=lambda value: len(self.tokenizer.encode(value)))
        self.settings = {"context_tokens": 8192, "output_tokens": 64,
                         "max_output_tokens": 256, "semantic_retries": 0, "context_blocks": 1}

    def writer(self, response):
        complete = Mock(return_value=response)
        return worker.NoRepairDeepSeekDirectoryWriter(self.tokenizer, self.settings, complete), complete

    def segment(self, writer):
        encoder = Mock()
        encoder.encode.return_value = [[1, 0], [.65, .5775 ** .5]]
        return worker.deepseek.DeepSeekSegmenter(encoder, settings={"context_weight": 0},
            review_boundary=writer.review_boundary).segment(self.session)

    def test_writer_requires_exactly_one_context_block(self):
        for value in (0, 2, True, 1.0, "1", None):
            with self.subTest(context_blocks=value), self.assertRaises(ValueError):
                worker.NoRepairDeepSeekDirectoryWriter(
                    self.tokenizer, {**self.settings, "context_blocks": value}, Mock())
        missing = {key: value for key, value in self.settings.items() if key != "context_blocks"}
        with self.assertRaises(ValueError):
            worker.NoRepairDeepSeekDirectoryWriter(self.tokenizer, missing, Mock())

    def test_children_receive_only_previous_block_and_parent_keeps_all_source_and_children(self):
        from tests.test_adjacent_writer import TOKENIZER, Transport, session

        source, transport, frozen_transport = session(7), Transport(), Transport()
        original = deepcopy(source)
        spans = [{"start_block": 1, "end_block": 3}, {"start_block": 4, "end_block": 5},
                 {"start_block": 6, "end_block": 7}]
        writer = worker.NoRepairDeepSeekDirectoryWriter(TOKENIZER, self.settings, transport)
        result = writer.build(source, spans)
        worker.deepseek.DeepSeekDirectoryWriter(
            TOKENIZER, {**self.settings, "context_blocks": 2}, frozen_transport).build(source, spans)

        self.assertEqual(result["status"], "complete")
        self.assertEqual([call[3] for call in transport.calls],
                         ["adjacent_child"] * 3 + ["adjacent_parent"])
        children = [call[4] for call in transport.calls[:-1]]
        self.assertEqual([[block["index"] for block in child["blocks"]] for child in children],
                         [[1, 2, 3], [4, 5], [6, 7]])
        self.assertEqual([[block["index"] for block in child["context_blocks"]] for child in children],
                         [[], [3], [5]])
        for child, previous_index in zip(children[1:], (3, 5)):
            self.assertEqual(child["context_blocks"], [writer.public["blocks"][previous_index - 1]])
        for current_call, old_call in zip(transport.calls, frozen_transport.calls):
            current, old = deepcopy(current_call[4]), deepcopy(old_call[4])
            current.pop("context_blocks")
            old.pop("context_blocks")
            self.assertEqual(current, old)
        parent = transport.calls[-1][4]
        self.assertEqual(parent["blocks"], writer.public["blocks"])
        self.assertEqual(parent["context_blocks"], [])
        self.assertEqual(parent["write_range"], {"start": 1, "end": 7})
        self.assertEqual(parent["children"], [
            {**span, **{key: child[key] for key in ("title", "content", "entities")}}
            for span, child in zip(spans, result["children"])])
        self.assertEqual(source, original)

    def test_context_cannot_cross_session_boundary(self):
        from tests.test_adjacent_writer import TOKENIZER, Transport, session

        previous, current = session(3, words=9), session(2, words=2)
        transport = Transport()
        writer = worker.NoRepairDeepSeekDirectoryWriter(TOKENIZER, self.settings, transport)
        writer.build(previous, [{"start_block": 1, "end_block": 3}])
        calls_before = len(transport.calls)
        with self.assertRaisesRegex(ValueError, "belongs to one session"):
            writer.build(current, [{"start_block": 1, "end_block": 2}])
        self.assertEqual(len(transport.calls), calls_before)
        fresh = worker.NoRepairDeepSeekDirectoryWriter(TOKENIZER, self.settings, transport)
        fresh.build(current, [{"start_block": 1, "end_block": 1}, {"start_block": 2, "end_block": 2}])
        first, second, parent = [call[4] for call in transport.calls[calls_before:]]
        self.assertEqual(first["context_blocks"], [])
        self.assertEqual(second["context_blocks"], [fresh.public["blocks"][0]])
        self.assertEqual(parent["context_blocks"], [])
        self.assertNotIn("fact3", canonical(transport.calls[calls_before:]))

    def test_failed_later_child_keeps_one_context_block_without_repair_or_splitting(self):
        from tests.test_adjacent_writer import TOKENIZER, Transport, session

        for response in ({"finish_reason": "stop", "text": "invalid JSON"},
                         {"finish_reason": "length", "text": "truncated"}):
            with self.subTest(response=response):
                transport = Transport(lambda *args: response)
                writer = worker.NoRepairDeepSeekDirectoryWriter(TOKENIZER, self.settings, transport)
                result = writer.build(session(5), [
                    {"start_block": 1, "end_block": 3}, {"start_block": 4, "end_block": 5}])
                self.assertEqual(result["status"], "fallback")
                self.assertEqual([call[3] for call in transport.calls],
                                 ["adjacent_child", "adjacent_child", "adjacent_parent"])
                self.assertEqual([call[1] for call in transport.calls], [64, 64, 64])
                later = transport.calls[1][4]
                self.assertEqual([block["index"] for block in later["blocks"]], [4, 5])
                self.assertEqual([block["index"] for block in later["context_blocks"]], [3])
                self.assertEqual([child["end_block"] for child in result["children"]], [3, 5])
                self.assertEqual(result["stats"]["semantic_repairs"], 0)
                self.assertEqual(result["stats"]["output_expansions"], 0)
                self.assertEqual(result["stats"]["fallback_windows"], 3)
                self.assertEqual(len(transport.rejected), 3)

    def test_no_reason_boundary_decisions_drive_real_segmenter(self):
        for decision in ("merge", "keep_boundary"):
            with self.subTest(decision=decision):
                output = {"decision": decision, "evidence_blocks": [1, 2]}
                writer, complete = self.writer({"finish_reason": "stop", "text": canonical(output)})
                result = self.segment(writer)
                complete.assert_called_once()
                complete.accept.assert_called_once()
                complete.reject.assert_not_called()
                self.assertEqual(len(result["spans"]), 1 if decision == "merge" else 2)
                self.assertEqual(result["stats"]["review_merges"], int(decision == "merge"))
                self.assertEqual(result["stats"]["review_unavailable"], 0)
                self.assertEqual(result["traces"][0]["review"],
                                 {**output, "available": True, "review_unavailable": False})
                self.assertEqual(writer.stats["boundary_review_failures"], 0)
                self.assertEqual(complete.call_args.args[3], "adjacent_boundary")

    def test_invalid_boundary_evidence_and_extra_reason_keep_boundary_without_retry(self):
        invalid = [
            {"decision": "merge", "evidence_blocks": [1, 2], "reason": "unrequested text"},
            {"decision": "maybe", "evidence_blocks": [1, 2]},
            {"decision": "merge"},
            {"evidence_blocks": [1, 2]},
        ]
        invalid.extend({"decision": "merge", "evidence_blocks": evidence} for evidence in
                       ([0], [3], [1, 1], [True], [1.0], ["1"], [None], None))
        for output in invalid:
            with self.subTest(output=output):
                writer, complete = self.writer({"finish_reason": "stop", "text": canonical(output)})
                result = self.segment(writer)
                complete.assert_called_once()
                complete.reject.assert_called_once()
                complete.accept.assert_not_called()
                self.assertEqual(len(result["spans"]), 2)
                self.assertEqual(result["stats"]["review_merges"], 0)
                self.assertEqual(result["stats"]["review_unavailable"], 1)
                self.assertEqual(writer.stats["boundary_review_failures"], 1)
                self.assertEqual(writer.stats["semantic_repairs"], 0)
                self.assertEqual(writer.stats["output_expansions"], 0)

    def test_frozen_writer_keeps_reason_and_summary_contracts_are_unchanged(self):
        writer, complete = self.writer(None)
        frozen = worker.deepseek.DeepSeekDirectoryWriter(self.tokenizer, self.settings, complete)
        schema = frozen._schema("boundary")
        self.assertEqual(set(schema["properties"]), {"decision", "reason", "evidence_blocks"})
        self.assertEqual(set(schema["required"]), {"decision", "reason", "evidence_blocks"})
        self.assertEqual(frozen.adjacent_prompts["boundary"], worker.SOURCE_BOUNDARY_PROMPT.read_text())
        self.assertEqual(writer.adjacent_prompts["boundary"], worker.BOUNDARY_PROMPT.read_text())
        for kind in ("child", "parent"):
            with self.subTest(kind=kind):
                self.assertEqual(writer._schema(kind), frozen._schema(kind))
                self.assertEqual(writer.adjacent_prompts[kind], frozen.adjacent_prompts[kind])

    def test_invalid_json_schema_and_token_lengths_get_exactly_one_request(self):
        invalid = ["not JSON", '{"title":"one","title":"two","content":"fact","entities":[]}',
                   canonical({"title": "Topic", "content": "Fact", "entities": [], "unexpected": True}),
                   canonical({"title": "word " * 25, "content": "Fact", "entities": []}),
                   canonical({"title": "Topic", "content": "word " * 301, "entities": []}),
                   canonical({"title": "Topic", "content": "Fact", "entities": [" "]})]
        for text in invalid:
            with self.subTest(response=text[:80]):
                writer, complete = self.writer({"finish_reason": "stop", "text": text})
                writer.prepare(self.session)
                payload = writer._raw_payload(1, 2, "child", [])
                original = deepcopy(payload)
                self.assertIsNone(writer._request_entry(payload, "child", "adjacent_child"))
                complete.assert_called_once()
                self.assertEqual(complete.call_args.args[1], 64)
                self.assertEqual(complete.call_args.args[3], "adjacent_child")
                self.assertEqual(json.loads(complete.call_args.args[0][-1]["content"]), original)
                self.assertEqual(payload, original)
                self.assertEqual(writer.stats["semantic_repairs"], 0)
                self.assertEqual(writer.stats["output_expansions"], 0)
                complete.reject.assert_called_once()

    def test_length_finish_creates_one_placeholder_per_entry_without_splitting(self):
        writer, complete = self.writer({"finish_reason": "length", "text": "truncated"})
        result = writer.build(self.session, [{"start_block": 1, "end_block": 2}])
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["mode"], "single_session")
        self.assertEqual(len(result["children"]), 1)
        self.assertEqual(result["children"][0]["end_block"], 2)
        self.assertEqual([call.args[3] for call in complete.call_args_list],
                         ["adjacent_child", "adjacent_parent"])
        self.assertEqual([call.args[1] for call in complete.call_args_list], [64, 64])
        for call in complete.call_args_list:
            payload = json.loads(call.args[0][-1]["content"])
            self.assertEqual([block["index"] for block in payload["blocks"]], [1, 2])
        self.assertEqual(result["stats"]["semantic_repairs"], 0)
        self.assertEqual(result["stats"]["output_expansions"], 0)
        self.assertEqual(result["stats"]["windows"], 2)
        self.assertEqual(result["stats"]["fallback_windows"], 2)
        self.assertEqual(complete.reject.call_count, 2)

    def test_invalid_boundary_keeps_original_boundary_after_one_request(self):
        for response in ({"finish_reason": "stop", "text": "invalid"},
                         {"finish_reason": "length", "text": "truncated"}):
            with self.subTest(response=response):
                writer, complete = self.writer(response)
                result = writer.review_boundary({"session": self.session,
                    "left": {"start_block": 1, "end_block": 1},
                    "right": {"start_block": 2, "end_block": 2}})
                complete.assert_called_once()
                self.assertEqual(result["decision"], "keep_boundary")
                self.assertTrue(result["review_unavailable"])
                self.assertEqual(writer.stats["boundary_review_failures"], 1)

    def test_transport_failure_escapes_without_becoming_placeholder(self):
        for kind in ("child", "parent", "boundary"):
            with self.subTest(kind=kind):
                writer, complete = self.writer(None)
                complete.side_effect = InfrastructureError("fixture service unavailable")
                writer.prepare(self.session)
                payload = writer._raw_payload(1, 2, "child", [])
                with self.assertRaisesRegex(InfrastructureError, "fixture service unavailable"):
                    writer._request_entry(payload, kind, "adjacent_" + kind)
                complete.assert_called_once()
                complete.reject.assert_not_called()
                self.assertEqual(writer.stats["fallback_windows"], 0)

    def test_parent_entry_rejects_excess_content_and_non_normal_finish_once(self):
        responses = [{"finish_reason": "stop", "text": canonical({"title": "Topic", "content": "word " * 201})},
                     {"finish_reason": "content_filter", "text": "{}"},
                     {"finish_reason": "stop", "text": ""}]
        for response in responses:
            with self.subTest(response=str(response)[:100]):
                writer, complete = self.writer(response)
                writer.prepare(self.session)
                self.assertIsNone(writer._request_entry(writer._raw_payload(1, 2, "parent", []),
                                                       "parent", "adjacent_parent"))
                complete.assert_called_once()
                complete.reject.assert_called_once()

    def test_input_overflow_still_reads_all_complete_source_windows(self):
        from tests.test_adjacent_writer import TOKENIZER, Transport, session, settings

        source, transport = session(8, words=180), Transport()
        original = deepcopy(source)
        writer = worker.NoRepairDeepSeekDirectoryWriter(
            TOKENIZER, settings(context_tokens=1200, semantic_retries=0, context_blocks=1), transport)
        result = writer.build(source, [{"start_block": 1, "end_block": 8}])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["mode"], "windowed")
        self.assertEqual([child["end_block"] for child in result["children"]], [8])
        for kind in ("child", "parent"):
            windows = [call[4] for call in transport.calls if call[3] == "adjacent_" + kind]
            self.assertGreater(len(windows), 1)
            self.assertEqual([block["index"] for window in windows for block in window["blocks"]],
                             list(range(1, 9)))
        self.assertEqual(result["stats"]["semantic_repairs"], 0)
        self.assertEqual(result["stats"]["output_expansions"], 0)
        self.assertEqual(source, original)


class NoRepairImportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = import_fixtures.ImportDeepSeekDirectoriesTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source_identity = self.fixture.old_identity
        self.source_identity["writer_settings"].update(semantic_retries=2, context_blocks=2)
        self.source_identity["boundary_prompt_version"] = worker.SOURCE_BOUNDARY_PROMPT_VERSION
        self.source_identity["prompt_hashes"]["boundary.txt"] = file_digest(worker.SOURCE_BOUNDARY_PROMPT)
        connection = self.fixture.connect(self.fixture.source)
        with connection:
            connection.execute("UPDATE metadata SET value=? WHERE key='identity'",
                               (canonical(self.source_identity),))
        self.target_identity = deepcopy(self.source_identity)
        self.target_identity["protocol"] = worker.PROTOCOL
        self.target_identity["writer_settings"].update(semantic_retries=0, context_blocks=1)
        use_no_reason_boundary(self.target_identity)

    def test_compatibility_allows_only_context_boundary_contract_and_retry_policy_changes(self):
        original = importer.compatible_identity
        before = deepcopy((self.source_identity, self.target_identity))
        worker.compatible_identity(self.source_identity, self.target_identity)
        self.assertEqual((self.source_identity, self.target_identity), before)
        self.assertIs(importer.compatible_identity, original)
        changes = [("tokenizer_id", "changed"), ("segmentation", {"high_threshold": .99}),
                   ("embedding", {"model_revision": "changed"}),
                   ("writer", {"model": "other", "model_revision": "different"}),
                   ("request_options", {"temperature": 1}),
                   ("prompt_hashes", {"child.txt": "changed"}),
                   ("boundary_prompt_version", "changed"), ("block_policy", "changed"),
                   ("boundary_output_fields", ["decision", "reason", "evidence_blocks"]),
                   ("base_resource_hashes", {"schema": "changed"}),
                   ("protocol", importer.TARGET_PROTOCOL)]
        for key, value in changes:
            with self.subTest(key=key):
                changed = deepcopy(self.target_identity)
                changed[key] = value
                with self.assertRaises(ValueError):
                    worker.compatible_identity(self.source_identity, changed)
        changed = deepcopy(self.target_identity)
        changed["datasets"][0]["files"][0]["sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            worker.compatible_identity(self.source_identity, changed)
        changed_source = deepcopy(self.source_identity)
        changed_source["writer_settings"]["semantic_retries"] = 1
        with self.assertRaises(ValueError):
            worker.compatible_identity(changed_source, self.target_identity)
        for key, value in (("parent_content_tokens", 401), ("child_content_tokens", 301),
                           ("context_tokens", 32768), ("semantic_retries", 1)):
            changed = deepcopy(self.target_identity)
            changed["writer_settings"][key] = value
            with self.subTest(writer_setting=key), self.assertRaises(ValueError):
                worker.compatible_identity(self.source_identity, changed)

    def test_compatibility_requires_explicit_integer_context_two_to_one_migration(self):
        for side, values in (("source", (0, 1, 3, True, 2.0, "2", None)),
                             ("target", (0, 2, 3, True, 1.0, "1", None))):
            for value in values:
                with self.subTest(side=side, context_blocks=value):
                    source, target = deepcopy((self.source_identity, self.target_identity))
                    identity = source if side == "source" else target
                    identity["writer_settings"]["context_blocks"] = value
                    with self.assertRaises(ValueError):
                        worker.compatible_identity(source, target)
            with self.subTest(side=side, context_blocks="missing"):
                source, target = deepcopy((self.source_identity, self.target_identity))
                identity = source if side == "source" else target
                del identity["writer_settings"]["context_blocks"]
                with self.assertRaises(ValueError):
                    worker.compatible_identity(source, target)

    def test_compatibility_rejects_unknown_or_stale_boundary_identities(self):
        for side in ("source", "target"):
            for field in ("boundary_prompt_version", "boundary.txt"):
                with self.subTest(side=side, field=field):
                    source, target = deepcopy((self.source_identity, self.target_identity))
                    identity = source if side == "source" else target
                    if field == "boundary.txt":
                        identity["prompt_hashes"][field] = "unknown-boundary-hash"
                    else:
                        identity[field] = "unknown-boundary-version"
                    with self.assertRaises(ValueError):
                        worker.compatible_identity(source, target)
        stale = deepcopy(self.target_identity)
        stale["boundary_prompt_version"] = self.source_identity["boundary_prompt_version"]
        stale["prompt_hashes"]["boundary.txt"] = self.source_identity["prompt_hashes"]["boundary.txt"]
        with self.assertRaises(ValueError):
            worker.compatible_identity(self.source_identity, stale)
        missing = deepcopy(self.target_identity)
        del missing["boundary_output_fields"]
        with self.assertRaises(ValueError):
            worker.compatible_identity(self.source_identity, missing)

    def test_recipe_records_new_boundary_identity_without_mutating_frozen_recipe(self):
        config = {"source_root": str(self.fixture.source), "output_root": str(self.fixture.target),
                  "writer": {"semantic_retries": 0, "context_blocks": 1}, "retry_fallback": False}
        original = deepcopy(self.source_identity)
        original["protocol"] = worker.PROTOCOL
        original["writer_settings"].update(semantic_retries=0, context_blocks=1)
        with patch.object(worker.deepseek, "load_recipe", return_value=(config, original)), \
                patch.object(importer, "_existing_target") as existing:
            _, identity = worker.load_recipe(worker.DEFAULT_CONFIG)
        self.assertEqual(identity["boundary_prompt_version"], worker.BOUNDARY_PROMPT_VERSION)
        self.assertEqual(identity["prompt_hashes"]["boundary.txt"], file_digest(worker.BOUNDARY_PROMPT))
        self.assertEqual(identity["boundary_output_fields"], ["decision", "evidence_blocks"])
        self.assertEqual(identity["writer_settings"]["context_blocks"], 1)
        self.assertEqual(original["boundary_prompt_version"], worker.SOURCE_BOUNDARY_PROMPT_VERSION)
        self.assertEqual(original["prompt_hashes"]["boundary.txt"], file_digest(worker.SOURCE_BOUNDARY_PROMPT))
        self.assertNotIn("boundary_output_fields", original)
        existing.assert_called_once()

    def test_recipe_rejects_other_context_policies_before_accessing_target(self):
        for value in (0, 2):
            config = {"source_root": str(self.fixture.source), "output_root": str(self.fixture.target),
                      "writer": {"semantic_retries": 0, "context_blocks": value}, "retry_fallback": False}
            with self.subTest(context_blocks=value), \
                    patch.object(worker.deepseek, "load_recipe", return_value=(config, self.target_identity)), \
                    patch.object(importer, "_existing_target") as existing:
                with self.assertRaisesRegex(ValueError, "context_blocks"):
                    worker.load_recipe(worker.DEFAULT_CONFIG)
                existing.assert_not_called()

    def test_import_keeps_exact_committed_directories_and_makes_failures_retryable(self):
        before = self.fixture.files(self.fixture.source)
        original = importer.compatible_identity
        report = worker.import_committed(self.fixture.source, self.fixture.target,
            self.target_identity, count_tokens=self.fixture.count)
        self.assertIs(importer.compatible_identity, original)
        self.assertEqual(self.fixture.files(self.fixture.source), before)
        self.assertEqual(report["source_identity"], self.source_identity)
        self.assertEqual(report["target_identity"], self.target_identity)
        self.assertEqual(report["source_identity"]["writer_settings"]["context_blocks"], 2)
        self.assertEqual(report["target_identity"]["writer_settings"]["context_blocks"], 1)
        source, target = self.fixture.connect(self.fixture.source), self.fixture.connect(self.fixture.target)
        for parent_id in ("p0001", "p0002", "p0003"):
            query = "SELECT snapshot,status,directory,result_stats FROM sessions WHERE parent_id=?"
            self.assertEqual(source.execute(query, (parent_id,)).fetchone(),
                             target.execute(query, (parent_id,)).fetchone())
        for parent_id in ("p0004", "p0005"):
            self.assertEqual(target.execute("SELECT status,directory,error,result_stats FROM sessions WHERE parent_id=?",
                                           (parent_id,)).fetchone(), ("pending", None, None, None))
        for table in ("writer_cache", "segment_cache", "writer_traces"):
            self.assertEqual(target.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 0)
        self.assertFalse((self.fixture.target / "embedding_cache.sqlite3").exists())
        store = AdjacentBuildStore(self.fixture.target, self.target_identity)
        session = self.fixture.sessions[4]
        completed = self.fixture.directory(session, self.target_identity)
        store.commit(session, completed, {"calls": 2})
        store.close()
        worker.import_committed(self.fixture.source, self.fixture.target,
                                self.target_identity, count_tokens=self.fixture.count)
        self.assertEqual(json.loads(target.execute("SELECT directory FROM sessions WHERE parent_id='p0005'").fetchone()[0]),
                         completed)

    def test_dry_import_needs_no_tokenizer_and_creates_no_target(self):
        before = self.fixture.files(self.fixture.source)
        with patch("agmem.policy.load_tokenizer", side_effect=AssertionError("must not load tokenizer")), \
                patch("httpx.Client", side_effect=AssertionError("must not use API")):
            report = worker.import_committed(self.fixture.source, self.fixture.target,
                self.target_identity, dry_run=True, count_tokens=lambda _: self.fail("must not tokenize"))
        self.assertEqual(report["status"], "planned")
        self.assertFalse(self.fixture.target.exists())
        self.assertEqual(self.fixture.files(self.fixture.source), before)

    def test_worker_dry_run_never_loads_tokenizer_or_constructs_services(self):
        before = self.fixture.files(self.fixture.source)
        config = {"source_root": str(self.fixture.source), "output_root": str(self.fixture.target),
                  "model_path": "fixture-reference-tokenizer"}
        pool, embedding, writer = Mock(), Mock(), Mock()
        args = worker.argument_parser(worker_mode=True).parse_args(["--dry-run"])
        with patch.object(worker, "load_recipe", return_value=(config, self.target_identity)), \
                patch.object(worker.engine, "plan", return_value=([], {"protocol": worker.PROTOCOL})), \
                patch.object(worker.engine, "announce"), \
                patch("agmem.policy.load_tokenizer", side_effect=AssertionError("must not load tokenizer")), \
                patch("httpx.Client", side_effect=AssertionError("must not use API")):
            result = worker.run(args, pool_factory=pool, embedding_factory=embedding, writer_factory=writer)
        self.assertEqual(result["protocol"], worker.PROTOCOL)
        self.assertEqual(result["no_repair_policy"]["semantic_retries"], 0)
        for factory in (pool, embedding, writer):
            factory.assert_not_called()
        self.assertFalse(self.fixture.target.exists())
        self.assertEqual(self.fixture.files(self.fixture.source), before)

    def test_worker_import_only_checks_token_counts_on_cpu_without_services(self):
        config = {"source_root": str(self.fixture.source), "output_root": str(self.fixture.target),
                  "model_path": "fixture-reference-tokenizer"}
        pool, embedding = Mock(), Mock()
        args = worker.argument_parser(worker_mode=True).parse_args(["--import-only"])
        with patch.object(worker, "load_recipe", return_value=(config, self.target_identity)), \
                patch.object(worker.engine, "plan", return_value=([], {})), \
                patch.object(worker.engine, "announce"), \
                patch("agmem.policy.load_tokenizer", return_value=Tokenizer()) as tokenizer, \
                patch("httpx.Client", side_effect=AssertionError("must not use API")):
            result = worker.run(args, pool_factory=pool, embedding_factory=embedding)
        tokenizer.assert_called_once_with("fixture-reference-tokenizer")
        self.assertTrue(result["token_counts_rechecked"])
        self.assertEqual(result["status"], "ready")
        pool.assert_not_called()
        embedding.assert_not_called()


class NoRepairLauncherTests(unittest.TestCase):
    def args(self, *arguments):
        return worker.argument_parser().parse_args(list(arguments))

    def test_cpu_import_precedes_legacy_launcher_and_routes_new_worker(self):
        args = self.args("--source-root", "/tmp/frozen source", "--output-root", "/tmp/fresh target")
        original = launcher.build_command
        events = []

        def imported(command, **options):
            events.append(("import", command, options))

        def launched(options):
            command = launcher.build_command(options, ["http://embedding/v1"])
            events.append(("launch", command, {}))
            return "fixture-complete"

        with patch.object(launcher.subprocess, "run", side_effect=imported), \
                patch.object(launcher, "launch", side_effect=launched):
            self.assertEqual(worker.launch(args), "fixture-complete")
        self.assertEqual([event[0] for event in events], ["import", "launch"])
        self.assertIn("--import-only", events[0][1])
        self.assertEqual(events[0][2]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("--import-only", events[1][1])
        for _, command, _ in events:
            self.assertIn("agmem.build_deepseek_no_repair", command)
            index = command.index("agmem.build_deepseek_no_repair")
            self.assertEqual(command[index + 1], "worker")
            self.assertEqual(command[command.index("--source-root") + 1], "/tmp/frozen source")
            self.assertEqual(command[command.index("--output-root") + 1], "/tmp/fresh target")
            self.assertEqual(command[command.index("--workers") + 1], "256")
        self.assertIs(launcher.build_command, original)

    def test_dry_run_and_import_only_never_require_keys_models_or_gpu(self):
        for option in ("--dry-run", "--import-only"):
            with self.subTest(option=option), ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {}, clear=True))
                run = stack.enter_context(patch.object(launcher.subprocess, "run"))
                guards = [stack.enter_context(patch.object(launcher, name)) for name in
                          ("local_embedding_model", "ensure_gpus_idle", "probe_deepseek")]
                start = stack.enter_context(patch.object(launcher.subprocess, "Popen"))
                worker.launch(self.args(option))
                run.assert_called_once()
                command = run.call_args.args[0]
                self.assertIn("agmem.build_deepseek_no_repair", command)
                self.assertIn("worker", command)
                self.assertIn(option, command)
                self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
                for guard in (*guards, start):
                    guard.assert_not_called()

    def test_failed_cpu_import_does_not_reach_launcher_and_restores_command(self):
        original = launcher.build_command
        with patch.object(launcher.subprocess, "run", side_effect=RuntimeError("fixture import failure")), \
                patch.object(launcher, "launch") as launch:
            with self.assertRaisesRegex(RuntimeError, "fixture import failure"):
                worker.launch(self.args())
            launch.assert_not_called()
        self.assertIs(launcher.build_command, original)


class NoRepairPipelineTests(unittest.TestCase):
    def setUp(self):
        self.continuation = continuation_fixtures.QwenContinuationPipelineTests(methodName="runTest")
        fixture_type = continuation_fixtures.fixtures.AdjacentPipelineTests
        item = fixture_type.item("q1", [["ALPHA fact"], [
            "ALPHA first question", "ALPHA second fact", "ALPHA third fact",
            "BETA question", "BETA clarification"]])
        with patch.object(fixture_type, "item", return_value=item):
            self.continuation.setUp()
        self.addCleanup(self.continuation.doCleanups)
        self.fixture = self.continuation.fixture
        self.source, self.target = self.continuation.source, self.continuation.target
        with sqlite3.connect(self.source / "build_journal.sqlite3") as db:
            self.identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        self.config = deepcopy(self.continuation.config)
        self.config.update(protocol=worker.PROTOCOL, retry_fallback=False)
        self.config["writer"].update(semantic_retries=0, context_blocks=1)
        self.config["service"] = {"model": "deepseek-flash", "base_url": "https://api.deepseek.com",
                                  "network_retries": 0, "trust_env": False}
        self.identity.update(protocol=worker.PROTOCOL, no_repair_policy=deepcopy(worker.NO_REPAIR_POLICY))
        self.identity["writer_settings"].update(semantic_retries=0, context_blocks=1)
        use_no_reason_boundary(self.identity)
        self.enterContext(patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-deepseek-key"}))

    def run_worker(self, pool=None, encoder=None):
        pool, encoder = pool or NoReasonHTTPPool(), encoder or Encoder(gray=True)
        args = self.fixture.args(embedding_base_urls=["http://fixture-embedding/v1"])
        with patch.object(worker, "load_recipe", side_effect=lambda *a, **k: deepcopy((self.config, self.identity))), \
                patch.object(worker.engine, "announce"), patch.object(importer, "announce"):
            return worker.run(args, tokenizer=self.fixture.tokenizer, embedding_tokenizer=self.fixture.tokenizer,
                              pool_factory=pool.factory, embedding_factory=encoder.factory,
                              writer_factory=None, segmenter_factory=None)

    def assert_source_and_globals_unchanged(self):
        self.assertEqual(self.continuation.source_files(), self.continuation.saved_source)
        engine = worker.engine
        self.assertEqual((engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results),
                         self.continuation.old_globals)

    def test_real_import_builds_unfinished_session_and_resume_makes_no_http_calls(self):
        pool = NoReasonHTTPPool()
        report = self.run_worker(pool)
        self.assertEqual(report["states"]["complete"], 2)
        self.assertEqual(report["execution"]["sessions_resumed"], 1)
        self.assertEqual([kind for kind, _ in pool.stages], ["boundary", "child", "parent"])
        system = pool.requests[0]["messages"][0]["content"]
        self.assertTrue(system.startswith(worker.BOUNDARY_PROMPT.read_text()))
        schema = json.loads(system.split("JSON Schema:\n")[1])
        self.assertEqual(set(schema["properties"]), {"decision", "evidence_blocks"})
        self.assertEqual(set(schema["required"]), {"decision", "evidence_blocks"})
        self.assertFalse(schema["additionalProperties"])
        rows = [json.loads(line) for line in (self.target / "directories.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0], self.continuation.old_directory)
        self.assertEqual(rows[1]["build"]["writer_model"], "deepseek-flash")
        with sqlite3.connect(self.target / "build_journal.sqlite3") as db:
            manifest = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        self.assertEqual(manifest["no_repair_policy"], worker.NO_REPAIR_POLICY)
        self.assertEqual(manifest["boundary_prompt_version"], worker.BOUNDARY_PROMPT_VERSION)
        self.assertEqual(manifest["prompt_hashes"]["boundary.txt"], file_digest(worker.BOUNDARY_PROMPT))
        self.assertEqual(manifest["boundary_output_fields"], ["decision", "evidence_blocks"])
        self.assertEqual(manifest["writer_settings"]["context_blocks"], 1)
        self.assert_source_and_globals_unchanged()
        saved = (self.target / "directories.jsonl").read_bytes()
        resumed_pool = NoReasonHTTPPool(forbidden=True)
        report = self.run_worker(resumed_pool, Encoder(forbidden=True))
        self.assertEqual(report["execution"]["sessions_resumed"], 2)
        self.assertEqual(resumed_pool.requests, [])
        self.assertEqual((self.target / "directories.jsonl").read_bytes(), saved)
        self.assert_source_and_globals_unchanged()

    def test_import_preserves_context_two_commits_and_builds_pending_session_with_context_one(self):
        pool = NoReasonHTTPPool()
        report = self.run_worker(pool, Encoder())
        self.assertEqual(report["states"]["complete"], 2)
        self.assertEqual(report["execution"]["sessions_resumed"], 1)
        self.assertEqual([kind for kind, _ in pool.stages], ["child", "child", "parent"])
        first, later, parent = [payload for _, payload in pool.stages]
        self.assertEqual([block["index"] for block in first["blocks"]], [1, 2, 3])
        self.assertEqual(first["context_blocks"], [])
        self.assertEqual([block["index"] for block in later["blocks"]], [4, 5])
        self.assertEqual(later["context_blocks"], [first["blocks"][-1]])
        self.assertEqual(parent["blocks"], first["blocks"] + later["blocks"])
        self.assertEqual(parent["context_blocks"], [])
        self.assertEqual([(child["start_block"], child["end_block"]) for child in parent["children"]],
                         [(1, 3), (4, 5)])
        with sqlite3.connect(self.source / "build_journal.sqlite3") as source, \
                sqlite3.connect(self.target / "build_journal.sqlite3") as target:
            query = "SELECT snapshot,status,directory,result_stats FROM sessions WHERE parent_id='p0001'"
            self.assertEqual(source.execute(query).fetchone(), target.execute(query).fetchone())
            query = "SELECT snapshot FROM sessions WHERE parent_id='p0002'"
            self.assertEqual(source.execute(query).fetchone(), target.execute(query).fetchone())
        manifest = json.loads((self.target / "deepseek_import_manifest.json").read_text())
        self.assertEqual(manifest["source_identity"]["writer_settings"]["context_blocks"], 2)
        self.assertEqual(manifest["target_identity"]["writer_settings"]["context_blocks"], 1)
        self.assert_source_and_globals_unchanged()
        saved = (self.target / "directories.jsonl").read_bytes()
        resumed_pool = NoReasonHTTPPool(forbidden=True)
        report = self.run_worker(resumed_pool, Encoder(forbidden=True))
        self.assertEqual(report["execution"]["sessions_resumed"], 2)
        self.assertEqual(resumed_pool.requests, [])
        self.assertEqual((self.target / "directories.jsonl").read_bytes(), saved)
        self.assert_source_and_globals_unchanged()

    def test_boundary_http_failure_is_failed_then_retryable_without_placeholder(self):
        pool = NoReasonHTTPPool(failure=402)
        with self.assertRaises(InfrastructureError):
            self.run_worker(pool)
        self.assertEqual([kind for kind, _ in pool.stages], ["boundary"])
        with sqlite3.connect(self.target / "build_journal.sqlite3") as db:
            self.assertEqual(dict(db.execute("SELECT status,COUNT(*) FROM sessions GROUP BY status")),
                             {"complete": 1, "failed": 1})
            status, directory, error = db.execute(
                "SELECT status,directory,error FROM sessions WHERE parent_id='p0002'").fetchone()
        self.assertEqual(status, "failed")
        self.assertIsNone(directory)
        self.assertEqual(json.loads(error)["error_type"], "InfrastructureError")
        self.assertEqual(self.run_worker()["states"]["complete"], 2)
        self.assert_source_and_globals_unchanged()

    def test_new_summary_fallback_is_committed_and_preserved_on_resume(self):
        class InvalidSummaries(NoReasonHTTPPool):
            def respond(self, request):
                response = super().respond(request)
                if self.stages[-1][0] in {"child", "parent"}:
                    payload = response.json()
                    payload["choices"][0]["message"]["content"] = "invalid JSON"
                    return httpx.Response(200, json=payload)
                return response

        pool = InvalidSummaries()
        report = self.run_worker(pool)
        self.assertEqual(report["states"]["complete"], 1)
        self.assertEqual(report["states"]["fallback"], 1)
        self.assertEqual([kind for kind, _ in pool.stages], ["boundary", "child", "parent"])
        with sqlite3.connect(self.target / "build_journal.sqlite3") as db:
            stats = json.loads(db.execute("SELECT result_stats FROM sessions WHERE parent_id='p0002'").fetchone()[0])
        self.assertEqual(stats["semantic_repairs"], 0)
        self.assertEqual(stats["output_expansions"], 0)
        saved = (self.target / "directories.jsonl").read_bytes()
        resumed_pool = NoReasonHTTPPool(forbidden=True)
        report = self.run_worker(resumed_pool, Encoder(forbidden=True))
        self.assertEqual(report["execution"]["sessions_resumed"], 2)
        self.assertEqual(report["states"]["fallback"], 1)
        self.assertEqual(resumed_pool.requests, [])
        self.assertEqual((self.target / "directories.jsonl").read_bytes(), saved)
        self.assert_source_and_globals_unchanged()


if __name__ == "__main__":
    unittest.main()
