"""Fresh DeepSeek directories, strict budgets, and resumable failures, all offline."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
from threading import Barrier
import unittest
from unittest.mock import Mock, patch

import httpx

from agmem import build_adjacent_directories as legacy
from agmem import build_deepseek_directories as deepseek
from agmem import build_topic_event_directories as topic
from agmem.core import InfrastructureError, canonical, file_digest
from agmem.hierarchical.deepseek_client import DeepSeekWriterPool, schema_messages
from agmem.hierarchical.reader import DirectoryReader
from agmem.hierarchical.writer import chat_token_count
from tests import test_topic_event_directory_pipeline as fixtures
from tests.test_adjacent_directory_pipeline import Encoder, Pool


class HTTPPool:
    """Use the real DeepSeek HTTP adapter with a synthetic provider response."""

    def __init__(self, *, failure=None, forbidden=False):
        self.failure, self.forbidden = failure, forbidden
        self.requests, self.stages, self.instances = [], [], []

    def factory(self, *args, **kwargs):
        pool = DeepSeekWriterPool(*args, **kwargs, transport=httpx.MockTransport(self.respond))
        self.instances.append(pool)
        return pool

    def respond(self, request):
        if self.forbidden:
            raise AssertionError("Resume must not issue any HTTP request")
        body = json.loads(request.content)
        self.requests.append(body)
        schema = json.loads(body["messages"][0]["content"].split("JSON Schema:\n")[1])
        payload = json.loads(body["messages"][-1]["content"])
        if "decision" in schema["properties"]:
            kind = "boundary"
            value = {"decision": "merge", "reason": "The request is clarified.",
                     "evidence_blocks": [payload["left"]["end_block"], payload["right"]["start_block"]]}
        else:
            kind = "child" if "entities" in schema["properties"] else "parent"
            value = {"title": "Source topics", "content": "The reported source facts."}
            if kind == "child":
                value["entities"] = []
        self.stages.append((kind, payload))
        if self.failure is not None:
            return httpx.Response(self.failure, text="private-server-body fixture-deepseek-key")
        return httpx.Response(200, json={
            "model": "deepseek-flash", "choices": [{"message": {"content": canonical(value)},
                                                      "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12},
        })


class DeepSeekDirectoryPipelineTests(unittest.TestCase):
    def setUp(self):
        # Compose the established topic/event fixture without rerunning its tests.
        self.topic_fixture = fixtures.TopicEventPipelineTests(methodName="runTest")
        self.topic_fixture.setUp()
        self.addCleanup(self.topic_fixture.doCleanups)
        self.fixture = self.topic_fixture.fixture
        self.original_globals = (legacy.PROTOCOL, legacy.PROMPT_VERSION, legacy.DATASETS,
                                 legacy.load_recipe, legacy.bounded_results)
        self.enterContext(patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-deepseek-key",
                                                 "DEEPSEEK_BASE_URL": ""}))

    def recipe(self, rows=None, *, workers=3):
        if rows is None:
            config, identity = self.topic_fixture.recipe()
        else:
            config, identity = self.fixture.recipe(rows)
            identity.update(prompt_hashes={"boundary.txt": "old-boundary", "child.txt": "unchanged"},
                            implementation_hashes={"build_adjacent_directories.py": "frozen"})
        config.update(protocol=deepseek.PROTOCOL, workers=workers, preserve_existing_directories=False)
        config["datasets"][0]["name"] = "split_longmemeval_m"
        config["service"] = {"model": "deepseek-flash", "network_retries": 0, "trust_env": False}
        identity["protocol"] = deepseek.PROTOCOL
        return config, identity

    def run_deepseek(self, recipe, pool=None, encoder=None, **args):
        pool, encoder = pool or HTTPPool(), encoder or Encoder()
        arguments = self.fixture.args(embedding_base_urls=["http://127.0.0.1:9001/v1",
                                                           "http://127.0.0.1:9002/v1"], **args)
        with patch.object(deepseek, "_BASE_LOAD_RECIPE", side_effect=lambda *a, **k: deepcopy(recipe)), \
                patch.object(legacy, "announce"):
            return deepseek.run(arguments, tokenizer=self.fixture.tokenizer,
                                embedding_tokenizer=self.fixture.tokenizer,
                                pool_factory=pool.factory, embedding_factory=encoder.factory)

    def assert_globals_restored(self):
        self.assertEqual((legacy.PROTOCOL, legacy.PROMPT_VERSION, legacy.DATASETS,
                          legacy.load_recipe, legacy.bounded_results), self.original_globals)

    def test_real_recipe_loader_identifies_remote_writer_and_reference_tokenizer_separately(self):
        encoder = self.fixture.home / "encoder"
        encoder.mkdir()
        (encoder / "config.json").write_text(json.dumps({"hidden_size": 2560}))
        recipe_path = self.fixture.home / "recipe.json"
        recipe_path.write_text(json.dumps({
            "protocol": deepseek.PROTOCOL, "model_path": "fixture-qwen-reference-tokenizer",
            "output_root": str(self.fixture.output), "embedding": {"model_path": str(encoder)},
            "datasets": [{"name": "split_longmemeval_m"}],
        }))
        with patch.object(legacy, "_model_identity", return_value=(
                {"ready": True, "model_revision": "old-local-weight-hash"}, "reference-tokenizer-hash")), \
                patch.object(legacy, "input_files", return_value=[]), patch("httpx.Client") as http:
            config, identity = deepseek.load_recipe(recipe_path)
        http.assert_not_called()
        self.assertEqual(config["service"]["model"], "deepseek-flash")
        self.assertEqual(config["service"]["api_key_env"], "DEEPSEEK_API_KEY")
        self.assertEqual(identity["protocol"], deepseek.PROTOCOL)
        self.assertEqual(identity["writer"]["provider"], "deepseek")
        self.assertEqual(identity["writer"]["model"], "deepseek-flash")
        self.assertEqual(identity["writer"]["model_revision"], "provider-alias:deepseek-flash")
        self.assertNotIn("old-local-weight-hash", canonical(identity["writer"]))
        self.assertEqual(identity["budget_tokenizer"]["tokenizer_id"], "reference-tokenizer-hash")
        self.assertFalse(identity["budget_tokenizer"]["local_writer_weights_loaded"])
        self.assertIn("not DeepSeek billing tokens", identity["budget_tokenizer"]["purpose"])
        self.assertEqual(identity["request_options"]["thinking"], {"type": "disabled"})
        self.assertEqual(identity["request_options"]["response_format"], {"type": "json_object"})
        self.assertIn("not sent", identity["request_options"]["seed"])
        self.assertEqual(identity["directory_reuse_policy"], "fresh_deepseek_root_only.v1")
        self.assertEqual(identity["prompt_hashes"]["boundary.txt"], file_digest(topic.BOUNDARY_PROMPT))
        self.assertEqual(identity["implementation_hashes"]["deepseek_client.py"],
                         file_digest(Path(deepseek.__file__).with_name("hierarchical") / "deepseek_client.py"))
        self.assert_globals_restored()

    def test_dry_run_requires_no_api_key_or_service_construction_and_writes_no_store(self):
        recipe = self.recipe()
        pool, encoder = Mock(), Mock()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), patch("httpx.Client") as http:
            preview = self.run_deepseek(recipe, pool, encoder, dry_run=True)
        self.assertEqual(preview["protocol"], deepseek.PROTOCOL)
        self.assertEqual(preview["writer"]["model"], "deepseek-flash")
        self.assertFalse(self.fixture.output.exists())
        pool.factory.assert_not_called()
        encoder.factory.assert_not_called()
        http.assert_not_called()
        self.assert_globals_restored()

    def test_rejects_old_qwen_root_before_services_and_preserves_existing_files(self):
        old = self.fixture.recipe([self.fixture.item("old", [["ALPHA source"]])])
        self.fixture.run_pipeline(old)
        saved = {path.name: path.read_bytes() for path in self.fixture.output.iterdir() if path.is_file()}
        pool, encoder = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "different build protocol"):
            self.run_deepseek(self.recipe(), pool, encoder)
        pool.factory.assert_not_called()
        encoder.factory.assert_not_called()
        self.assertEqual({path.name: path.read_bytes() for path in self.fixture.output.iterdir() if path.is_file()}, saved)
        self.assert_globals_restored()

    def test_rejects_nonempty_root_without_manifest_and_explicit_reuse(self):
        self.fixture.output.mkdir()
        (self.fixture.output / "unrelated.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "fresh output root"):
            self.run_deepseek(self.recipe(), Mock(), Mock())
        (self.fixture.output / "unrelated.txt").unlink()
        for changes in ({"preserve_existing_directories": True}, {"reuse_root": "old-qwen-root"}):
            recipe = self.recipe()
            recipe[0].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "reuse is disabled"):
                self.run_deepseek(recipe, Mock(), Mock())
        self.assert_globals_restored()

    def test_rejects_other_dataset_or_model(self):
        for location, value in (("dataset", "split_longmemeval_s"), ("model", "old-qwen-writer")):
            recipe = self.recipe()
            if location == "dataset":
                recipe[0]["datasets"][0]["name"] = value
            else:
                recipe[0]["service"]["model"] = value
            with self.subTest(location=location), self.assertRaises(ValueError):
                self.run_deepseek(recipe, Mock(), Mock())
        self.assert_globals_restored()

    def test_actual_http_adapter_builds_children_then_parent_and_resumes_without_requests(self):
        history = [["ALPHA question", "ALPHA answer", "BETA question", "BETA answer"]]
        recipe = self.recipe([self.fixture.item("q1", history), self.fixture.item("q2", history)])
        pool, encoder = HTTPPool(), Encoder()
        report = self.run_deepseek(recipe, pool, encoder)
        self.assertEqual(report["states"]["complete"], 2)
        self.assertEqual(report["execution"]["cache_hits"], 1)
        self.assertEqual([kind for kind, payload in pool.stages], ["child", "child", "parent"])
        self.assertEqual(len(encoder.calls), 1)
        parent = pool.stages[-1][1]
        self.assertEqual([block["index"] for block in parent["blocks"]], [1, 2, 3, 4])
        self.assertEqual(len(parent["children"]), 2)
        directories = self.fixture.rows("directories.jsonl")
        for directory in directories:
            self.assertEqual([(child["source"]["start_block"], child["source"]["end_block"])
                              for child in directory["children"]], [(1, 2), (3, 4)])
            self.assertIn("deepseek-flash", canonical(directory["build"]))
            self.assertNotIn("fixture", canonical(directory["build"]))
        manifest = json.loads((self.fixture.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["writer"]["model"], "deepseek-flash")
        self.assertEqual(manifest["boundary_prompt_version"], topic.BOUNDARY_PROMPT_VERSION)
        self.assertTrue(encoder.closed)
        self.assertTrue(all(client.is_closed for _, client in pool.instances[0].clients))
        saved = (self.fixture.output / "directories.jsonl").read_bytes()
        resumed_pool = HTTPPool(forbidden=True)
        resumed = self.run_deepseek(recipe, resumed_pool, Encoder(forbidden=True))
        self.assertEqual(resumed["execution"]["sessions_resumed"], 2)
        self.assertFalse(resumed_pool.requests)
        self.assertEqual((self.fixture.output / "directories.jsonl").read_bytes(), saved)
        self.assert_globals_restored()

    def test_semantic_boundary_prompt_survives_transport_schema_and_source_is_reader_accessible(self):
        pool = HTTPPool()
        report = self.run_deepseek(self.recipe(), pool, Encoder(gray=True))
        self.assertEqual(report["states"]["complete"], 1)
        self.assertEqual([kind for kind, payload in pool.stages], ["boundary", "child", "parent"])
        self.assertTrue(pool.requests[0]["messages"][0]["content"].startswith(topic.BOUNDARY_PROMPT.read_text()))
        directory = self.fixture.rows("directories.jsonl")[0]
        with sqlite3.connect(self.fixture.output / "build_journal.sqlite3") as db:
            session = json.loads(db.execute("SELECT snapshot FROM sessions").fetchone()[0])
        reader = DirectoryReader([directory], [session],
                                 count_tokens=lambda value: len(self.fixture.tokenizer.encode(value)),
                                 retention="cumulative")
        state = reader.read(["p0001"], reader.initial_state(), 9999)["state"]
        evidence = reader.read(["p0001:c0001"], state, 9999)
        self.assertTrue(evidence["ok"])
        self.assertEqual([record["text"] for record in evidence["result"]["records"]],
                         ["ALPHA request", "BETA clarification"])
        serialized = canonical(pool.requests) + canonical(self.fixture.rows("writer_traces.jsonl"))
        for hidden in ("PRIVATE QUERY", "PRIVATE GOLD", "answer_hidden", "has_answer", "fixture-deepseek-key"):
            self.assertNotIn(hidden, serialized)

    def test_changed_deepseek_identity_cannot_resume(self):
        recipe = self.recipe()
        self.run_deepseek(recipe)
        changed = deepcopy(recipe)
        changed[1]["segmentation"]["high_threshold"] = .99
        pool, encoder = Mock(), Mock()
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.run_deepseek(changed, pool, encoder)
        pool.factory.assert_not_called()
        encoder.factory.assert_not_called()
        self.assert_globals_restored()

    def test_schema_overhead_participates_in_fit_and_output_reservations(self):
        config, _ = self.recipe()
        writer = deepseek.DeepSeekDirectoryWriter(self.fixture.tokenizer, config["writer"], Mock())
        messages = [{"role": "system", "content": "Retain all source facts."},
                    {"role": "user", "content": "ALPHA " * 60}]
        original = deepcopy(messages)
        unaugmented = chat_token_count(writer.tokenizer, messages)
        sizes = [chat_token_count(writer.tokenizer, schema_messages(messages, writer._schema(kind)))
                 for kind in ("boundary", "child", "parent")]
        writer.settings["context_tokens"] = (writer.settings["safety_margin"]
                                              + writer.settings["output_tokens"] + unaugmented)
        self.assertGreater(max(sizes), unaugmented)
        self.assertFalse(writer._fits(messages))
        self.assertEqual(writer._available(messages), writer.settings["output_tokens"] + unaugmented - max(sizes))
        writer.settings["context_tokens"] += max(sizes) - unaugmented
        self.assertTrue(writer._fits(messages))
        self.assertEqual(writer._available(messages), writer.settings["output_tokens"])
        self.assertEqual(messages, original)
        writer.complete.assert_not_called()

    def test_every_actual_request_fits_the_reference_context_including_transport_schema(self):
        recipe = self.recipe()
        pool = HTTPPool()
        self.run_deepseek(recipe, pool, Encoder(gray=True))
        settings = recipe[0]["writer"]
        for request in pool.requests:
            with self.subTest(model=request["model"]):
                total = (chat_token_count(self.fixture.tokenizer, request["messages"])
                         + request["max_tokens"] + settings["safety_margin"])
                self.assertLessEqual(total, settings["context_tokens"])

    def test_402_commits_failed_session_then_stops_admission_and_can_resume(self):
        bodies = [[f"ALPHA source {index}"] for index in range(12)]
        recipe = self.recipe([self.fixture.item("q", bodies)], workers=1)
        pool, encoder = HTTPPool(failure=402), Encoder()
        with self.assertRaisesRegex(InfrastructureError, "Directory service failed"):
            self.run_deepseek(recipe, pool, encoder)
        self.assertEqual(len(pool.requests), 1)
        # A single-block session needs no embedding to determine its sole span.
        self.assertEqual(len(encoder.calls), 0)
        self.assertTrue(encoder.closed)
        self.assertTrue(all(client.is_closed for _, client in pool.instances[0].clients))
        report = json.loads((self.fixture.output / "build_report.json").read_text())
        self.assertEqual(report["execution"]["run_status"], "interrupted")
        self.assertEqual(report["sessions"], 1)
        self.assertEqual(report["states"]["failed"], 1)
        self.assertEqual(self.fixture.rows("directories.jsonl"), [])
        with sqlite3.connect(self.fixture.output / "build_journal.sqlite3") as db:
            status, error = db.execute("SELECT status,error FROM sessions").fetchone()
        self.assertEqual(status, "failed")
        self.assertEqual(json.loads(error)["error_type"], "InfrastructureError")
        self.assertIn("HTTP 402", error)
        serialized = canonical(self.fixture.rows("writer_traces.jsonl")) + canonical(report) + error
        for secret in ("fixture-deepseek-key", "private-server-body", "PRIVATE QUERY", "PRIVATE GOLD"):
            self.assertNotIn(secret, serialized)
        self.assert_globals_restored()
        resumed = self.run_deepseek(recipe)
        self.assertEqual(resumed["states"]["complete"], 12)
        self.assertEqual(resumed["states"]["failed"], 0)
        self.assertEqual(resumed["execution"]["segment_cache_hits"], 1)
        self.assert_globals_restored()

    def test_boundary_402_stops_after_one_http_and_is_failed_instead_of_fallback(self):
        body = ["ALPHA start", "BETA next", "ALPHA more", "BETA finish"]
        recipe = self.recipe([self.fixture.item("q", [body])], workers=1)
        pool = HTTPPool(failure=402)
        with self.assertRaisesRegex(InfrastructureError, "Directory service failed"):
            self.run_deepseek(recipe, pool, Encoder(gray=True))
        self.assertEqual([kind for kind, payload in pool.stages], ["boundary"])
        self.assertEqual(len(pool.requests), 1)
        report = json.loads((self.fixture.output / "build_report.json").read_text())
        self.assertEqual(report["states"]["failed"], 1)
        self.assertEqual(report["states"]["fallback"], 0)
        self.assertEqual(report["states"]["complete"], 0)
        with sqlite3.connect(self.fixture.output / "build_journal.sqlite3") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM segment_cache").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM writer_cache").fetchone()[0], 0)
        self.assertEqual(self.fixture.rows("directories.jsonl"), [])
        self.assert_globals_restored()

    def test_parallel_failfast_never_admits_beyond_initial_worker_window(self):
        admitted, completed = [], []
        barrier = Barrier(4)

        def items():
            for item in range(1000):
                admitted.append(item)
                yield item

        def fail(item):
            barrier.wait(timeout=5)
            completed.append(item)
            return {"error_type": "InfrastructureError", "reason": "synthetic failure"}

        results = deepseek._fail_fast_results(fail, items(), 4)
        first = next(results)
        self.assertIn(first[0], range(4))
        with self.assertRaises(InfrastructureError):
            next(results)
        self.assertEqual(admitted, [0, 1, 2, 3])
        self.assertEqual(sorted(completed), [0, 1, 2, 3])

    def test_loader_and_runtime_errors_restore_scoped_engine_globals(self):
        with patch.object(deepseek, "_BASE_LOAD_RECIPE", side_effect=ValueError("invalid recipe")):
            with self.assertRaisesRegex(ValueError, "invalid recipe"):
                deepseek.load_recipe("unused")
        self.assert_globals_restored()
        with self.assertRaises(InfrastructureError):
            self.run_deepseek(self.recipe(), Pool(fail_parent=True))
        self.assert_globals_restored()


if __name__ == "__main__":
    unittest.main()
