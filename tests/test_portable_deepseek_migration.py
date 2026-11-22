"""Real recipe/import/resume checks after relocating data and reinstalling models."""

from contextlib import chdir
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agmem import build_portable_deepseek_directories as portable
from agmem.core import file_digest
from agmem.portable_directory_identity import VERSION, model_files
from tests import test_adjacent_directory_pipeline as fixtures
from tests.test_adjacent_directory_pipeline import Encoder, Pool, Tokenizer


class PortableDeepSeekMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        old = self.home / "removed-host"
        self.project = self.home / "transferred-project"
        old_writer, old_embedding = old / "models/writer", old / "models/embedding"
        assets = {"config.json": '{"hidden_size":2560}',
                  "tokenizer.json": '{"version":"1.0"}',
                  "tokenizer_config.json": '{"model_max_length":65536}',
                  "chat_template.jinja": "{{ messages }}", "merges.txt": "a b\n",
                  "model.safetensors": "tiny-embedding-weight-fixture"}
        for root in (old_writer, old_embedding):
            root.mkdir(parents=True)
            for name, value in assets.items():
                path = root / name
                path.write_text(value)
                os.utime(path, ns=(1_600_000_000_000_000_000,) * 2)
        old_data, old_source = old / "data.json", old / "runs/source"
        old_data.write_text(json.dumps([fixtures.AdjacentPipelineTests.item("q1", [
            ["ALPHA original fact"], ["BETA unfinished fact"]])]))
        source_config = {
            "protocol": portable.deepseek.PROTOCOL, "model_path": str(old_writer),
            "output_root": str(old_source), "workers": 1, "retry_fallback": False,
            "embedding": {"model_path": str(old_embedding)},
            "writer": {"semantic_retries": 2, "context_blocks": 2},
            "service": {"model": "deepseek-flash", "base_url": "https://api.deepseek.com"},
            "datasets": [{"name": "split_longmemeval_m", "path": str(old_data), "split": "test"}],
        }
        old_recipe = old / "source.json"
        old_recipe.write_text(json.dumps(source_config))
        self.enterContext(patch.dict(os.environ, {"DEEPSEEK_BASE_URL": ""}))
        self.enterContext(patch.object(portable.engine, "announce"))
        self.enterContext(patch.object(portable.importer, "announce"))
        source_args = portable.deepseek.argument_parser().parse_args([
            "--config", str(old_recipe), "--embedding-base-urls", "http://fixture/v1", "--limit-sessions", "1"])
        portable.deepseek.run(source_args, tokenizer=Tokenizer(), embedding_tokenizer=Tokenizer(),
                              pool_factory=Pool().factory, embedding_factory=Encoder().factory)
        with sqlite3.connect(old_source / "build_journal.sqlite3") as db:
            self.source_identity = json.loads(db.execute(
                "SELECT value FROM metadata WHERE key='identity'").fetchone()[0])

        self.writer = self.project / "models/qwen3.5-27b"
        self.embedding = self.project / "models/embedding"
        self.source, self.target = self.project / "runs/source", self.project / "runs/continued"
        self.writer.mkdir(parents=True)
        for name in model_files(old_writer, tokenizer_only=True):
            shutil.copy2(old_writer / name, self.writer / name)
        shutil.copytree(old_embedding, self.embedding)
        shutil.copytree(old_source, self.source)
        self.data = self.project / "data.json"
        shutil.copy2(old_data, self.data)
        self.manifest = {
            "version": VERSION, "source_root_relative": "runs/source",
            "reference_tokenizer_path": "models/qwen3.5-27b",
            "source_identity": deepcopy(self.source_identity),
            "data_files": [{**self.entry(self.data, self.project), "split": "test"}],
            "models": {
                "writer_tokenizer": {"source_model_path": str(old_writer),
                    "tokenizer_id": self.source_identity["tokenizer_id"],
                    "files": [self.entry(old_writer / name, old_writer)
                              for name in model_files(old_writer, tokenizer_only=True)]},
                "embedding": {"source_model_path": str(old_embedding),
                    "identity": deepcopy(self.source_identity["embedding"]),
                    "files": [self.entry(old_embedding / name, old_embedding)
                              for name in model_files(old_embedding)]},
            },
        }
        for root in (self.writer, self.embedding):
            for path in root.rglob("*"):
                if path.is_file():
                    os.utime(path, ns=(1_700_000_000_000_000_123,) * 2)
        self.config = deepcopy(source_config)
        self.config.update(protocol=portable.worker.PROTOCOL, model_path=str(self.writer),
                           source_root=str(old_source), output_root=str(self.target))
        self.config["embedding"]["model_path"] = str(self.embedding)
        self.config["writer"].update(semantic_retries=0, context_blocks=1)
        self.config["datasets"][0]["path"] = str(self.data)
        self.recipe, self.manifest_path = self.project / "recipe.json", self.project / "migration.json"
        self.write_config()
        self.manifest_path.write_text(json.dumps(self.manifest))
        shutil.rmtree(old)
        self.enterContext(chdir(self.project))
        self.originals = (portable.engine._model_identity, portable.importer._verify_files,
                          portable.deepseek._BASE_LOAD_RECIPE, portable.worker.compatible_identity)

    @staticmethod
    def entry(path, root):
        return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                "sha256": file_digest(path)}

    def write_config(self):
        self.recipe.write_text(json.dumps(self.config))

    def args(self, *extra):
        return portable.argument_parser(worker_mode=True).parse_args([
            "--config", str(self.recipe), "--migration-manifest", str(self.manifest_path), *extra])

    def snapshot(self):
        return {str(path): (file_digest(path), path.stat().st_mtime_ns, path.stat().st_ctime_ns)
                for root in (self.source, self.writer, self.embedding)
                for path in root.rglob("*") if path.is_file()}

    def assert_restored(self):
        self.assertEqual((portable.engine._model_identity, portable.importer._verify_files,
                          portable.deepseek._BASE_LOAD_RECIPE, portable.worker.compatible_identity),
                         self.originals)

    def test_tokenizer_only_import_preserves_source_and_real_identities_and_resumes(self):
        self.assertFalse((self.writer / "config.json").exists())
        self.assertFalse(list(self.writer.glob("*.safetensors")))
        self.assertNotEqual(portable.engine._model_identity(self.embedding)[0]["model_revision"],
                            self.source_identity["embedding"]["model_revision"])
        before = self.snapshot()
        args = self.args("--import-only")
        original_args = deepcopy(vars(args))
        with patch("httpx.Client", side_effect=AssertionError("no API")), \
                patch("agmem.policy.load_tokenizer", side_effect=AssertionError("fixture supplied")):
            report = portable.run(args, tokenizer=Tokenizer())
        self.assertEqual(report["counters"]["imported_complete"], 1)
        self.assertEqual(vars(args), original_args)
        manifest = json.loads((self.target / "deepseek_import_manifest.json").read_text())
        self.assertEqual(manifest["source_identity"], self.source_identity)
        identity = manifest["target_identity"]
        self.assertEqual(identity["writer"], self.source_identity["writer"])
        self.assertEqual(identity["budget_tokenizer"]["path"], str(self.writer))
        self.assertNotEqual(identity["budget_tokenizer"]["path"],
                            self.source_identity["budget_tokenizer"]["path"])
        self.assertEqual(identity["embedding"], self.source_identity["embedding"])
        self.assertEqual(identity["writer_settings"]["context_blocks"], 1)
        self.assertEqual(identity["boundary_output_fields"], ["decision", "evidence_blocks"])
        proof = identity["portable_migration"]["models"]
        self.assertTrue(proof["writer_tokenizer"]["actual_identity"]["model_revision"].startswith(
            "reference-tokenizer@"))
        self.assertIn("build_portable_deepseek_directories.py", identity["implementation_hashes"])
        self.assertIn("portable_directory_identity.py", identity["implementation_hashes"])
        with sqlite3.connect(self.source / "build_journal.sqlite3") as source, \
                sqlite3.connect(self.target / "build_journal.sqlite3") as target:
            query = "SELECT snapshot,directory,result_stats FROM sessions WHERE status='complete'"
            self.assertEqual(source.execute(query).fetchall(), target.execute(query).fetchall())
            for table in ("writer_cache", "segment_cache", "writer_traces"):
                self.assertEqual(target.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)
        saved_manifest = (self.target / "deepseek_import_manifest.json").read_bytes()
        with patch.object(portable.importer, "_import_session", side_effect=AssertionError("must use marker")):
            resumed = portable.run(self.args("--import-only"), tokenizer=Tokenizer())
        self.assertTrue(resumed["import_already_complete"])
        self.assertEqual((self.target / "deepseek_import_manifest.json").read_bytes(), saved_manifest)
        self.assertEqual(self.snapshot(), before)
        self.assert_restored()

    def test_dry_run_needs_no_key_model_loading_service_or_target_writes(self):
        before = self.snapshot()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
                patch("httpx.Client", side_effect=AssertionError("no HTTP")), \
                patch("agmem.policy.load_tokenizer", side_effect=AssertionError("no tokenizer loading")):
            result = portable.run(self.args("--dry-run"))
        self.assertEqual(result["protocol"], portable.worker.PROTOCOL)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.snapshot(), before)
        self.assert_restored()

    def test_inference_reload_after_import_retains_portable_identity_and_resumes(self):
        args = self.args("--embedding-base-urls", "http://fixture/v1")
        pool, encoder = Pool(), Encoder()
        report = portable.run(args, tokenizer=Tokenizer(), embedding_tokenizer=Tokenizer(),
                              pool_factory=pool.factory, embedding_factory=encoder.factory)
        self.assertEqual(report["states"]["complete"], 2)
        self.assertEqual(len(pool.calls), 2)
        resumed = portable.run(args, tokenizer=Tokenizer(), embedding_tokenizer=Tokenizer(),
                               pool_factory=Pool(forbidden=True).factory,
                               embedding_factory=Encoder(forbidden=True).factory)
        self.assertEqual(resumed["execution"]["sessions_resumed"], 2)
        self.assert_restored()

    def test_provider_and_budget_changes_are_rejected_before_target_creation(self):
        original = deepcopy(self.config)
        for section, key, value in (("service", "base_url", "https://different.example"),
                                    ("writer", "child_content_tokens", 299)):
            self.config = deepcopy(original)
            self.config[section][key] = value
            self.write_config()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "semantics"):
                portable.run(self.args("--import-only"), tokenizer=Tokenizer())
            self.assertFalse(self.target.exists())
            self.assert_restored()

    def test_compatibility_normalizes_only_verified_budget_path_without_mutation(self):
        with portable.portable_worker(self.manifest_path, self.recipe):
            _, target = portable.worker.load_recipe(self.recipe)
            before = deepcopy((self.source_identity, target))
            portable.worker.compatible_identity(self.source_identity, target)
            self.assertEqual((self.source_identity, target), before)
            for key, value in (("path", "/unverified/tokenizer"), ("purpose", "different budgets"),
                               ("tokenizer_id", "different-tokenizer")):
                changed = deepcopy(target)
                changed["budget_tokenizer"][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "budget_tokenizer"):
                    portable.worker.compatible_identity(self.source_identity, changed)
            changed = deepcopy(self.source_identity)
            changed["writer"]["provider"] = "other"
            with self.assertRaisesRegex(ValueError, "Source journal identity"):
                portable.worker.compatible_identity(changed, target)
        self.assert_restored()

    def test_tokenizer_embedding_and_data_corruption_rejected_before_target_creation(self):
        for path in (self.writer / "tokenizer.json", self.embedding / "model.safetensors", self.data):
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "SHA256"):
                portable.run(self.args("--import-only"), tokenizer=Tokenizer())
            path.write_bytes(original)
            self.assertFalse(self.target.exists())
            self.assert_restored()

    def test_source_override_and_reference_path_cannot_bypass_manifest(self):
        with self.assertRaisesRegex(ValueError, "source-root"):
            portable.run(self.args("--source-root", str(self.project / "other"), "--dry-run"))
        self.manifest["reference_tokenizer_path"] = "models/other"
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "reference_tokenizer_path"):
            portable.run(self.args("--dry-run"))
        self.assertFalse(self.target.exists())
        self.assert_restored()


class PortableDeepSeekLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.manifest = Path(temporary.name) / "manifest.json"
        self.manifest.write_text("{}")

    def args(self, *extra):
        return portable.argument_parser().parse_args(["--migration-manifest", str(self.manifest), *extra])

    def assert_command(self, command):
        module = command.index("-m")
        self.assertEqual(command[module + 1:module + 3],
                         ["agmem.build_portable_deepseek_directories", "worker"])
        self.assertEqual(command[command.index("--migration-manifest") + 1], str(self.manifest))
        self.assertEqual(command.count("--migration-manifest"), 1)
        self.assertEqual(command[command.index("--workers") + 1], "256")

    def test_real_launcher_routes_import_preflight_and_inference_to_portable_worker(self):
        original = portable.launcher.build_command
        args = self.args("--embedding-base-urls", "http://fixture/v1", "--source-root", "source")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-key"}), \
                patch.object(portable.launcher, "load_recipe", return_value={}), \
                patch.object(portable.launcher, "probe_deepseek") as probe, \
                patch.object(portable.subprocess, "run") as run:
            portable.launch(args)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            self.assert_command(call.args[0])
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertIn("--source-root", call.args[0])
        self.assertIn("--import-only", run.call_args_list[0].args[0])
        self.assertIn("--dry-run", run.call_args_list[1].args[0])
        self.assertNotIn("--import-only", run.call_args_list[2].args[0])
        probe.assert_called_once()
        self.assertIs(portable.launcher.build_command, original)

    def test_dry_run_and_import_only_skip_gpu_launcher_and_restore_on_failure(self):
        original = portable.launcher.build_command
        for mode in ("--dry-run", "--import-only"):
            with self.subTest(mode=mode), patch.object(portable.launcher, "launch") as launch, \
                    patch.object(portable.subprocess, "run") as run:
                portable.launch(self.args(mode))
                self.assert_command(run.call_args.args[0])
                self.assertIn(mode, run.call_args.args[0])
                launch.assert_not_called()
        with patch.object(portable.launcher, "launch") as launch, \
                patch.object(portable.subprocess, "run", side_effect=RuntimeError("CPU import failed")), \
                self.assertRaisesRegex(RuntimeError, "CPU import"):
            portable.launch(self.args())
        launch.assert_not_called()
        self.assertIs(portable.launcher.build_command, original)


if __name__ == "__main__":
    unittest.main()
