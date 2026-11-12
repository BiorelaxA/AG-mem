"""CPU-only remote Writer entrypoint and resume/provenance checks."""

from argparse import Namespace
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from agmem.build_remote_directories import build_plan, load_recipe, run
from agmem.core import atomic_json


class BuildRemoteDirectoriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input_root = self.root / "input"
        self.prepared_root = self.root / "prepared"
        self.train_root = self.root / "train"
        self.sources = ["split_longmemeval_s", "split_longmemeval_m"]
        self.saved = {"data": {"sources": ["locomo"], "splits": ["train", "validation"],
                               "input_root": "old", "prepared_root": "old",
                               "frozen_directory_version": "a" * 64},
                      "writer": {"lookback_sessions": 3, "content_max_tokens": 300},
                      "models": {"task_path": "original-qwen-tokenizer"},
                      "service": {"model": "qwen-local", "base_url": "http://127.0.0.1:8000/v1"},
                      "execution": {"writer_workers": 2}}
        atomic_json(self.train_root / "run.json", {"config": self.saved})
        atomic_json(self.input_root / "manifest.json", {"fixture": True})
        for source in self.sources:
            path = self.input_root / source / "test.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(pa.Table.from_pylist([{"question": "held out question"}]), path)
        self.recipe = {"training_run": str(self.train_root), "input_root": str(self.input_root),
                       "prepared_root": str(self.prepared_root), "datasets": self.sources,
                       "writer_workers": 8,
                       "writer_service": {"request_profile": "openai_chat", "model": "gpt-5-mini",
                                          "model_revision": "fixture-gpt-v1",
                                          "summary_tokenizer_revision": "fixture-qwen-v1",
                                          "base_url": "https://fixture.invalid/v1",
                                          "api_key_env": "FIXTURE_WRITER_KEY",
                                          "base_url_env": "FIXTURE_WRITER_URL"}}
        self.path = self.root / "recipe.yaml"
        self.write_recipe()
        self.env = patch.dict(os.environ, {"FIXTURE_WRITER_KEY": "never-persist-secret"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def write_recipe(self):
        self.path.write_text(yaml.safe_dump(self.recipe))

    def args(self, **kwargs):
        values = {"config": str(self.path), "workers": None, "datasets": None, "dry_run": False}
        return Namespace(**{**values, **kwargs})

    def prepared(self, config, source, root):
        result_config = deepcopy(config)
        result_config["data"].update(sources=[source], prepared_root=str(root / source))
        return {"source": source, "config": result_config, "status": "needs_build",
                "examples": 1, "histories": 1, "sessions": 4, "directory_version": None}

    def completed(self, prepared):
        result = deepcopy(prepared)
        result.update(status="ready", directory_version="b" * 64)
        result["config"]["data"]["frozen_directory_version"] = "b" * 64
        return result

    def test_inherits_writer_and_local_tokenizer_but_selects_only_test(self):
        _, config = load_recipe(self.path, workers=3)
        self.assertEqual(config["writer"], self.saved["writer"])
        self.assertEqual(config["models"], self.saved["models"])
        self.assertEqual(config["service"], self.saved["service"])
        self.assertEqual(config["data"]["splits"], ["test"])
        self.assertEqual(config["data"]["sources"], self.sources)
        self.assertNotIn("frozen_directory_version", config["data"])
        self.assertEqual(config["execution"]["writer_workers"], 3)
        self.assertEqual(json.loads((self.train_root / "run.json").read_text())["config"], self.saved)

    def test_rejects_changed_window_limit_non_test_source_and_invalid_workers(self):
        for field, value in (("lookback_sessions", 4), ("content_max_tokens", 301)):
            with self.subTest(field=field):
                saved = deepcopy(self.saved)
                saved["writer"][field] = value
                atomic_json(self.train_root / "run.json", {"config": saved})
                with self.assertRaises(ValueError):
                    load_recipe(self.path)
        atomic_json(self.train_root / "run.json", {"config": self.saved})
        for workers in (0, -1, True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                load_recipe(self.path, workers)
        self.recipe["datasets"] = ["locomo"]
        self.write_recipe()
        with self.assertRaisesRegex(ValueError, "LongMemEval"):
            load_recipe(self.path)

    def test_dry_run_without_key_writes_nothing_and_never_loads_weights_or_calls_api(self):
        os.environ.pop("FIXTURE_WRITER_KEY")
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        stream = io.StringIO()
        with patch("agmem.policy.load_tokenizer", side_effect=AssertionError("must not load")), \
                patch("agmem.build_remote_directories.prepare_source", side_effect=AssertionError("must not prepare")), \
                patch("agmem.remote_writer._complete", side_effect=AssertionError("must not call API")), \
                redirect_stdout(stream):
            plan = run(self.args(dry_run=True, datasets=[self.sources[1]]))
        self.assertEqual([d["source"] for d in plan["datasets"]], [self.sources[1]])
        self.assertEqual(plan["datasets"][0]["queries"], 1)
        self.assertFalse(plan["requires_gpu"])
        after = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertNotIn("never-persist-secret", stream.getvalue())

    def test_worker_count_does_not_change_writer_identity_or_prompt(self):
        recipe, first = load_recipe(self.path, 2)
        _, second = load_recipe(self.path, 16)
        left, right = build_plan(recipe, first, self.sources), build_plan(recipe, second, self.sources)
        self.assertEqual(left["writer"], right["writer"])
        self.assertEqual(left["prompt_sha256"], right["prompt_sha256"])
        self.assertNotEqual(left["writer_workers"], right["writer_workers"])

    def test_missing_key_fails_before_preparation_or_writes(self):
        os.environ.pop("FIXTURE_WRITER_KEY")
        with patch("agmem.build_remote_directories.prepare_source") as prepare, \
                self.assertRaisesRegex(ValueError, "FIXTURE_WRITER_KEY"), redirect_stdout(io.StringIO()):
            run(self.args())
        prepare.assert_not_called()
        self.assertFalse(self.prepared_root.exists())

    def test_selected_source_and_resume_preserve_manifest_across_worker_changes(self):
        with patch("agmem.policy.load_tokenizer"), \
                patch("agmem.build_remote_directories._current_writer_version", return_value="b" * 64), \
                patch("agmem.build_remote_directories.prepare_source", side_effect=self.prepared) as prepare, \
                patch("agmem.build_remote_directories.build_source_directories", side_effect=self.completed), \
                redirect_stdout(io.StringIO()):
            first = run(self.args(datasets=[self.sources[0]], workers=2))
            manifest_path = self.prepared_root / "remote_build_manifest.json"
            manifest = manifest_path.read_bytes()
            second = run(self.args(datasets=[self.sources[0]], workers=16))
        self.assertEqual(first, second)
        self.assertEqual(manifest, manifest_path.read_bytes())
        self.assertEqual([call.args[1] for call in prepare.call_args_list], [self.sources[0]] * 2)
        self.assertFalse((self.prepared_root / self.sources[1]).exists())
        for path in self.prepared_root.rglob("*.json"):
            self.assertNotIn("never-persist-secret", path.read_text())

    def test_published_qwen_version_is_rejected_before_build(self):
        def old_directory(config, source, root):
            result = self.prepared(config, source, root)
            result.update(status="ready", directory_version="a" * 64)
            return result
        with patch("agmem.policy.load_tokenizer"), \
                patch("agmem.build_remote_directories._current_writer_version", return_value="b" * 64), \
                patch("agmem.build_remote_directories.prepare_source", side_effect=old_directory), \
                patch("agmem.build_remote_directories.build_source_directories") as build, \
                self.assertRaisesRegex(ValueError, "different Writer"), redirect_stdout(io.StringIO()):
            run(self.args(datasets=[self.sources[0]]))
        build.assert_not_called()
        status = json.loads((self.prepared_root / self.sources[0] / "remote_build_status.json").read_text())
        self.assertEqual(status["status"], "failed")

    def test_protocol_change_is_rejected_before_preparation(self):
        atomic_json(self.prepared_root / "remote_build_manifest.json", {"different": "protocol"})
        with patch("agmem.policy.load_tokenizer"), \
                patch("agmem.build_remote_directories._current_writer_version", return_value="b" * 64), \
                patch("agmem.build_remote_directories.prepare_source") as prepare, \
                self.assertRaisesRegex(ValueError, "protocol changed"), redirect_stdout(io.StringIO()):
            run(self.args())
        prepare.assert_not_called()

    def test_failure_records_safe_status_and_next_invocation_retries(self):
        with patch("agmem.policy.load_tokenizer"), \
                patch("agmem.build_remote_directories._current_writer_version", return_value="b" * 64), \
                patch("agmem.build_remote_directories.prepare_source", side_effect=self.prepared), \
                patch("agmem.build_remote_directories.build_source_directories") as build, \
                redirect_stdout(io.StringIO()):
            build.side_effect = RuntimeError("never-persist-secret")
            with self.assertRaises(RuntimeError):
                run(self.args(datasets=[self.sources[0]]))
            path = self.prepared_root / self.sources[0] / "remote_build_status.json"
            status = json.loads(path.read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["error_type"], "RuntimeError")
            self.assertNotIn("never-persist-secret", path.read_text())
            build.side_effect = self.completed
            run(self.args(datasets=[self.sources[0]]))
            self.assertEqual(json.loads(path.read_text())["status"], "ready")


if __name__ == "__main__":
    unittest.main()
