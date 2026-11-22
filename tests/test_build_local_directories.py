from argparse import Namespace
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from agmem.build_local_directories import (LocalWriterPool, SOURCES, _entry_metadata, _version,
                                           load_recipe, run)
from agmem.core import InfrastructureError, atomic_json, canonical, file_digest
from agmem.data import clean_history
from agmem.pipeline import validate_directories
from agmem.store import Store


class TinyTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()


class LocalDirectoryBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "input"
        manifests = []
        self.row = {"question": "Do not send this query to Writer", "answer": "private gold",
                    "question_date": "2026-01-02", "question_type": "single-session-user",
                    "haystack_sessions": canonical([[{"role": "user", "content": f"Fact {i}"}]
                                                     for i in range(4)]),
                    "haystack_dates": ["2026-01-01"] * 4,
                    "haystack_session_ids": [f"session{i}" for i in range(4)]}
        for source in SOURCES:
            files = {}
            for split in ["train", "validation", "test"]:
                p = self.input / source / f"{split}.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist([self.row]), p)
                files[split] = {"sha256": file_digest(p)}
            manifests.append({"source": source, "output_files": files})
        atomic_json(self.input / "manifest.json", {"sources": manifests})
        self.writer = {"lookback_sessions": 3, "content_max_tokens": 300}
        self.local_identity = {"base_url": "http://127.0.0.1:50543/v1", "model": "agmem-27b",
                               "model_revision": "fixture-local-revision"}
        self.gpt_identity = {"base_url": "https://fixture.invalid/v1", "model": "gpt-5-mini",
                             "model_revision": "fixture-remote", "request_profile": "openai_chat"}
        self.saved = {"models": {"task_path": str(self.root / "model")}, "writer": self.writer,
                      "data": {"frozen_directory_version": "a" * 64}, "execution": {}}
        atomic_json(self.root / "training/run.json", {"config": self.saved})
        self.recipe = {"training_run": str(self.root / "training"), "input_root": str(self.input),
                       "prepared_root": str(self.root / "completed"),
                       "reuse_train_root": str(self.root / "old_train"),
                       "reuse_test_roots": [{"root": str(self.root / "gpt"), "label": "gpt-5-mini"},
                                            {"root": str(self.root / "qwen"), "label": "qwen3.5-27B"}],
                       "writer_workers": 48}
        self.recipe_path = self.root / "recipe.yaml"
        self.recipe_path.write_text(yaml.safe_dump(self.recipe))
        self.fingerprint = patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "fixture-local-revision"})
        self.fingerprint.start()
        self.addCleanup(self.fingerprint.stop)
        self.tokenizer = patch("agmem.policy.load_tokenizer", return_value=TinyTokenizer())
        self.tokenizer.start()
        self.addCleanup(self.tokenizer.stop)

    def args(self, **kwargs):
        defaults = {"config": str(self.recipe_path), "workers": 1, "base_url": None,
                    "base_urls": ["http://localhost:8000/v1"], "datasets": [SOURCES[0]],
                    "splits": ["test"], "dry_run": False}
        return Namespace(**{**defaults, **kwargs})

    def origin(self, label, entries, pending=()):
        identity = self.gpt_identity if label == "gpt" else self.local_identity
        store = Store(self.root / label / SOURCES[0])
        store.set_metadata("prepare_complete", True)
        store.set_metadata("prepare_identity", {"sources": [SOURCES[0]], "splits": ["test"]})
        if label == "gpt":
            atomic_json(store.root.parent / "remote_build_manifest.json",
                        {"writer": identity, "writer_config": self.writer})
        mid = store.add_memory(clean_history(self.row))
        version = _version(identity, self.writer)
        for i in entries:
            record = store._records(mid)[i]
            entry = {**_entry_metadata(record), "content": f"{label} exact summary {i}"}
            if i in pending:
                entry.update(content="", writer_status="empty_pending")
            store.put_directory_entry(mid, version, entry)
        store.close()
        return mid

    def run_with_pool(self, answer, **kwargs):
        class Pool:
            def __init__(self, config, store, local_identity, mixed_identity, urls, workers):
                self.identity = mixed_identity

            def complete(self, messages, *args, **kwargs):
                return answer(messages)

            def close(self):
                pass

        with patch("agmem.build_local_directories.LocalWriterPool", Pool), redirect_stdout(io.StringIO()):
            return run(self.args(**kwargs))

    def test_partial_gpt_and_qwen_are_preserved_and_only_remaining_sessions_generated(self):
        mid = self.origin("gpt", [0])
        self.origin("qwen", [0, 1])
        before = {str(p): p.read_bytes() for root in (self.root / "gpt", self.root / "qwen")
                  for p in root.rglob("*") if p.is_file()}
        calls = []

        def answer(messages):
            calls.append(messages)
            self.assertNotIn("private gold", canonical(messages))
            self.assertNotIn("Do not send this query", canonical(messages))
            return "New local summary"

        result = self.run_with_pool(answer)
        report = result[f"test/{SOURCES[0]}"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["by_writer"], {"gpt-5-mini": 1, "qwen3.5-27B": 1, "qwen3.5-27B (new)": 2})
        output = Path(report["prepared"])
        store = Store(output)
        entries = store.directory(mid, report["directory_version"])
        self.assertEqual(entries[0]["content"], "gpt exact summary 0")
        self.assertEqual(entries[1]["content"], "qwen exact summary 1")
        self.assertEqual(set(entries[0]), set(Store.ENTRY_FIELDS))
        config = json.loads((output / "completed_config.json").read_text())
        validate_directories(config, store, ["test"])
        store.close()
        self.assertEqual(before, {str(p): p.read_bytes() for root in (self.root / "gpt", self.root / "qwen")
                                 for p in root.rglob("*") if p.is_file()})
        # A complete snapshot resumes without a tokenizer or local inference.
        with patch("agmem.policy.load_tokenizer", side_effect=AssertionError("must not load")), \
                patch("agmem.build_local_directories.LocalWriterPool", side_effect=AssertionError("must not infer")), \
                redirect_stdout(io.StringIO()):
            again = run(self.args(workers=48, base_urls=["http://localhost:9001/v1", "http://localhost:9002/v1"]))
        self.assertEqual(again, result)

    def test_failure_keeps_completed_sessions_and_resume_only_retries_missing(self):
        self.origin("gpt", [0])
        self.origin("qwen", [1])
        calls = []

        def interrupted(messages):
            calls.append(messages)
            if len(calls) == 2:
                raise InfrastructureError("replica stopped")
            return "Completed before interruption"

        with self.assertRaises(InfrastructureError):
            self.run_with_pool(interrupted)
        output = self.root / "completed/test" / SOURCES[0]
        self.assertNotIn("directory_version", json.loads((output / "manifest.json").read_text()))
        self.assertEqual(json.loads((output / "local_build_status.json").read_text())["status"], "failed")
        retries = []
        result = self.run_with_pool(lambda messages: retries.append(messages) or "Recovered final summary")
        self.assertEqual(len(retries), 1)
        self.assertEqual(result[f"test/{SOURCES[0]}"]["sessions"], 4)

    def test_empty_pending_is_rebuilt_and_complete_sources_do_not_call_model(self):
        self.origin("gpt", [0, 1], pending=[1])
        self.origin("qwen", [1, 2, 3])
        with patch("agmem.build_local_directories.LocalWriterPool", side_effect=AssertionError("no inference")), \
                patch("agmem.policy.load_tokenizer", side_effect=AssertionError("no tokenizer")), \
                redirect_stdout(io.StringIO()):
            report = run(self.args())[f"test/{SOURCES[0]}"]
        self.assertEqual(report["by_writer"], {"gpt-5-mini": 1, "qwen3.5-27B": 3})

    def test_dry_run_does_not_write_or_call_model_and_validates_source_versions(self):
        self.origin("gpt", [0])
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch("agmem.build_local_directories.LocalWriterPool", side_effect=AssertionError("no requests")), \
                patch("agmem.policy.load_tokenizer", side_effect=AssertionError("no tokenizer")), \
                redirect_stdout(io.StringIO()):
            plan = run(self.args(dry_run=True))
        self.assertEqual(plan["jobs"][0]["source_entries"][0]["entries"], 1)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
        path = self.root / "gpt" / SOURCES[0] / "directories.jsonl"
        row = json.loads(path.read_text())
        row["version"] = "b" * 64
        path.write_text(canonical(row) + "\n")
        with self.assertRaisesRegex(ValueError, "unverified Writer version"), redirect_stdout(io.StringIO()):
            run(self.args(dry_run=True))

    def test_three_replicas_have_sixteen_inflight_slots_each_and_share_one_store(self):
        barrier, lock = Barrier(48), Lock()
        seen, peak, active = Counter(), Counter(), Counter()
        services = []
        shared_store = object()

        class Service:
            def __init__(self, config, store, role):
                self.endpoint = config["request_base_url"]
                self.store = store
                self.client = Mock()
                services.append(self)

            def complete(self, messages):
                with lock:
                    active[self.endpoint] += 1
                    peak[self.endpoint] = max(peak[self.endpoint], active[self.endpoint])
                    seen[self.endpoint] += 1
                barrier.wait(timeout=10)
                with lock:
                    active[self.endpoint] -= 1
                return self.endpoint

        urls = [f"http://localhost:{8000 + i}/v1" for i in range(3)]
        with patch("agmem.build_local_directories.ChatService", Service):
            pool = LocalWriterPool({"service": {}}, shared_store, self.local_identity, {"mixed": True}, urls, 48)
            with ThreadPoolExecutor(max_workers=48) as executor:
                results = list(executor.map(pool.complete, range(48)))
            pool.close()
        self.assertEqual(len(results), 48)
        self.assertEqual(dict(seen), {url: 16 for url in urls})
        self.assertEqual(dict(peak), {url: 16 for url in urls})
        self.assertTrue(all(service.store is shared_store for service in services))


if __name__ == "__main__":
    unittest.main()
