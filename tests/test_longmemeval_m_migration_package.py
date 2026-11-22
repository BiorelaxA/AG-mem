"""Portable packaging preserves source bytes and exports only resumable state."""

import fcntl
import importlib.util
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from agmem.build_adjacent_directories import _model_identity
from agmem.core import canonical


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_longmemeval_m_continuation.py"
SPEC = importlib.util.spec_from_file_location("longmemeval_m_packager", SCRIPT)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


class LongMemEvalMigrationPackageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        self.root = self.home / "repo"
        self.root.mkdir()
        self.output = self.home / "package"
        self.source = self.root / packager.DEFAULT_SOURCE
        self.source.mkdir(parents=True)
        self.writer = self.model(packager.DEFAULT_WRITER, "writer")
        self.embedding = self.model(packager.DEFAULT_EMBEDDING, "embedding")
        _, tokenizer = _model_identity(self.writer)
        encoder, encoder_tokenizer = _model_identity(self.embedding)
        self.data = self.root / "data" / "split_longmemeval_m"
        self.data.mkdir(parents=True)
        files = []
        for split in packager.SPLITS:
            path = self.data / f"{split}.parquet"
            path.write_bytes(b"PAR1" + split.encode() + b" fixture PAR1")
            files.append({"split": split, "path": str(path), "sha256": packager.sha256(path)})
        self.identity = {
            "protocol": packager.SOURCE_PROTOCOL,
            "datasets": [{"spec": {"name": "split_longmemeval_m", "format": "longmemeval_parquet",
                                    "input_root": str(self.data.parent), "splits": list(packager.SPLITS)},
                          "files": files}],
            "tokenizer_id": tokenizer,
            "embedding": {**encoder, "tokenizer_id": encoder_tokenizer, "settings": {"instruction": "topic"}},
            "writer": {"model": "deepseek-flash", "model_revision": "provider-alias:deepseek-flash"},
        }
        (self.source / ".build.lock").write_bytes(b"old lock bytes")
        (self.source / "run_manifest.json").write_text(json.dumps(self.identity, indent=2) + "\n")
        (self.source / "build_report.json").write_text('{"sessions": 3}\n')
        (self.source / "embedding_cache.sqlite3").write_bytes(b"cache excluded")
        (self.source / "writer_traces.jsonl").write_text('{"secret": "exclude traces"}')
        with sqlite3.connect(self.source / "build_journal.sqlite3") as db:
            db.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE banks (bank_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE sessions (
                    bank_id TEXT NOT NULL, parent_id TEXT NOT NULL, source_hash TEXT NOT NULL,
                    snapshot TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    directory TEXT, error TEXT, result_stats TEXT, PRIMARY KEY(bank_id,parent_id));
                CREATE TABLE writer_cache (key TEXT PRIMARY KEY, result TEXT NOT NULL);
                CREATE TABLE writer_traces (seq INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL);
                CREATE TABLE segment_cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
            db.executemany("INSERT INTO metadata VALUES (?,?)", [
                ("identity", canonical(self.identity)), ("excluded_metadata", "do not transfer")])
            db.execute("INSERT INTO banks VALUES (?,?,?)", ("bank1", "split_longmemeval_m", '{"bank_id":"bank1"}'))
            for index, status in enumerate(("complete", "failed", "pending")):
                db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)", (
                    "bank1", f"p{index}", f"sha{index}", '{"immutable": true}', status,
                    '{"directory": "exact old bytes"}' if status == "complete" else None,
                    '{"error": "source error"}' if status == "failed" else None,
                    '{"tokens":7}' if status == "complete" else None))
            db.execute("INSERT INTO writer_cache VALUES (?,?)", ("huge", "x" * 100000))
            db.execute("INSERT INTO writer_traces(payload) VALUES (?)", ("private provider trace",))
            db.execute("INSERT INTO segment_cache VALUES (?,?)", ("seg", "{}"))
        self.add_code("agmem/worker.py", "print('code fixture')\n")
        self.add_code("configs/build.yaml", "workers: 4\n")
        self.add_code("scripts/run.sh", "#!/bin/sh\n")
        self.add_code("tests/test_fixture.py", "# source test\n")
        self.add_code("pyproject.toml", "[project]\nname='fixture'\n")
        self.add_code("requirements-test.txt", "pytest\n")
        self.add_code("README.md", "Fixture code bundle\n")
        self.add_code("environment.yml", "name: fixture\n")
        self.add_code("agmem/__pycache__/worker.pyc", "excluded")
        self.add_code("tests/.pytest_cache/state", "excluded")
        self.add_code("scripts/old.pyc", "excluded")
        self.add_code(".env", "SECRET=excluded\n")
        (self.root / "agmem" / "secret-link").symlink_to(self.root / ".env")
        (self.root / "scripts" / "data-link").symlink_to(self.data, target_is_directory=True)
        progress = patch.object(packager, "announce")
        progress.start()
        self.addCleanup(progress.stop)

    def add_code(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def model(self, relative, name):
        root = self.root / relative
        root.mkdir(parents=True)
        for filename, content in {
            "config.json": '{"hidden_size": 4}',
            "tokenizer.json": '{"model": "fixture"}',
            "tokenizer_config.json": '{"name": "fixture"}',
            "vocab.json": '{"hello": 1}',
            "special_tokens_map.json": '{"bos_token": "<s>"}',
            "model.safetensors": name + "-weights",
            "chat_template.jinja": "{{ messages }}",
            "merges.txt": "a b\n",
            "tokenizer.model": "fixture-tokenizer",
            "spiece.model": "fixture-spiece",
            "irrelevant.jinja": "ignored",
            "README.md": "not a model proof input",
        }.items():
            (root / filename).write_text(content)
        (root / "1_Pooling").mkdir()
        (root / "1_Pooling" / "config.json").write_text('{"pooling_mode_mean_tokens": true}')
        (root / ".cache").mkdir()
        (root / ".cache" / "download.json").write_text('{"ignored": true}')
        return root

    def run_package(self, **kwargs):
        return packager.package(output=self.output, repo_root=self.root, **kwargs)

    def source_state(self):
        return {path.relative_to(self.root).as_posix(): (packager.sha256(path), path.stat().st_mtime_ns)
                for base in (self.source, self.writer, self.embedding, self.data)
                for path in base.rglob("*") if path.is_file()}

    def extract(self, filename):
        destination = self.home / (filename + "-extracted")
        destination.mkdir()
        with tarfile.open(self.output / filename) as archive:
            for member in archive.getmembers():
                self.assertTrue(member.isfile())
                self.assertFalse(Path(member.name).is_absolute())
                self.assertNotIn("..", Path(member.name).parts)
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
        return destination

    def test_package_preserves_source_and_exact_rows_without_caches_or_weights(self):
        before = self.source_state()
        result = self.run_package()
        self.assertEqual(self.source_state(), before)
        self.assertEqual(result["journal_rows"], {"metadata": 1, "banks": 1, "sessions": 3})
        self.assertEqual({path.name for path in self.output.iterdir()},
                         {"data.tar.gz", "code.tar.gz", "SHA256SUMS", "COMPLETE.json"})
        extracted = self.extract("data.tar.gz")
        compact = extracted / packager.DEFAULT_SOURCE
        with sqlite3.connect(self.source / "build_journal.sqlite3") as original, \
                sqlite3.connect(compact / "build_journal.sqlite3") as copied:
            tables = {row[0] for row in copied.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(tables, set(packager.TABLES))
            self.assertEqual(copied.execute("SELECT key FROM metadata").fetchall(), [("identity",)])
            for table in packager.TABLES:
                query = "SELECT sql FROM sqlite_master WHERE type='table' AND name=?"
                self.assertEqual(original.execute(query, (table,)).fetchone(), copied.execute(query, (table,)).fetchone())
                rows = f'SELECT * FROM "{table}"'
                if table == "metadata":
                    rows += " WHERE key='identity'"
                self.assertEqual(original.execute(rows).fetchall(), copied.execute(rows).fetchall())
        self.assertLess((compact / "build_journal.sqlite3").stat().st_size,
                        (self.source / "build_journal.sqlite3").stat().st_size)
        self.assertEqual((compact / ".build.lock").stat().st_size, 0)
        self.assertEqual((compact / "run_manifest.json").read_bytes(), (self.source / "run_manifest.json").read_bytes())
        self.assertEqual((compact / "build_report.json").read_bytes(), (self.source / "build_report.json").read_bytes())
        manifest = json.loads((extracted / packager.MANIFEST_PATH).read_text())
        self.assertNotIn("reference_tokenizer_path", manifest)
        self.assertNotIn("reference_tokenizer_path", result)
        self.assertEqual(manifest["source_identity"], self.identity)
        self.assertEqual(manifest["models"]["embedding"]["identity"], self.identity["embedding"])
        self.assertEqual(manifest["models"]["writer_tokenizer"]["tokenizer_id"], self.identity["tokenizer_id"])
        for entry in manifest["data_files"]:
            self.assertEqual((extracted / entry["path"]).read_bytes(), (self.root / entry["path"]).read_bytes())
            self.assertEqual(packager.sha256(extracted / entry["path"]), entry["sha256"])
        for label, model in (("writer_tokenizer", self.writer), ("embedding", self.embedding)):
            files = manifest["models"][label]["files"]
            self.assertTrue(files)
            for entry in files:
                self.assertEqual(packager.sha256(model / entry["path"]), entry["sha256"])
                self.assertEqual((model / entry["path"]).stat().st_size, entry["bytes"])
            self.assertNotIn("README.md", {entry["path"] for entry in files})
            self.assertNotIn("irrelevant.jinja", {entry["path"] for entry in files})
            self.assertNotIn(".cache/download.json", {entry["path"] for entry in files})
        writer_paths = {entry["path"] for entry in manifest["models"]["writer_tokenizer"]["files"]}
        embedding_paths = {entry["path"] for entry in manifest["models"]["embedding"]["files"]}
        self.assertNotIn("model.safetensors", writer_paths)
        self.assertNotIn("1_Pooling/config.json", writer_paths)
        self.assertIn("model.safetensors", embedding_paths)
        self.assertIn("1_Pooling/config.json", embedding_paths)
        self.assertFalse((extracted / "models").exists())
        checksums = (self.output / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(checksums, [f"{item['sha256']}  {item['path']}" for item in result["archives"]])
        for item in result["archives"]:
            self.assertEqual(packager.sha256(self.output / item["path"]), item["sha256"])
        self.assertEqual(json.loads((self.output / "COMPLETE.json").read_text()), result)
        code = self.extract("code.tar.gz")
        self.assertEqual({path.relative_to(code).as_posix() for path in code.rglob("*") if path.is_file()}, {
            "agmem/worker.py", "configs/build.yaml", "scripts/run.sh", "tests/test_fixture.py",
            "pyproject.toml", "requirements-test.txt", "README.md", "environment.yml"})
        with tarfile.open(self.output / "data.tar.gz") as archive:
            for entry in manifest["data_files"]:
                self.assertAlmostEqual(archive.getmember(entry["path"]).mtime,
                                       (self.root / entry["path"]).stat().st_mtime, places=5)

    def test_reference_tokenizer_bundle_contains_only_verified_assets(self):
        before = self.source_state()
        result = self.run_package(include_reference_tokenizer=True)
        self.assertEqual(self.source_state(), before)
        extracted = self.extract("data.tar.gz")
        manifest = json.loads((extracted / packager.MANIFEST_PATH).read_text())
        self.assertEqual(manifest["reference_tokenizer_path"], packager.DEFAULT_WRITER)
        self.assertEqual(result["reference_tokenizer_path"], packager.DEFAULT_WRITER)
        self.assertEqual(json.loads((self.output / "COMPLETE.json").read_text()), result)
        tokenizer = extracted / manifest["reference_tokenizer_path"]
        proof = manifest["models"]["writer_tokenizer"]
        self.assertEqual(proof["source_model_path"], str(self.writer))
        self.assertEqual({path.relative_to(tokenizer).as_posix() for path in tokenizer.rglob("*")},
                         {entry["path"] for entry in proof["files"]})
        for entry in proof["files"]:
            copied = tokenizer / entry["path"]
            self.assertEqual(copied.read_bytes(), (self.writer / entry["path"]).read_bytes())
            self.assertEqual(copied.stat().st_size, entry["bytes"])
            self.assertEqual(packager.sha256(copied), entry["sha256"])
        self.assertFalse((tokenizer / "config.json").exists())
        self.assertFalse((tokenizer / "model.safetensors").exists())
        self.assertFalse((extracted / packager.DEFAULT_EMBEDDING).exists())
        self.assertFalse(list(extracted.rglob("*.safetensors")))
        self.assertFalse((self.extract("code.tar.gz") / "models").exists())

    def test_reference_tokenizer_bundle_preserves_custom_relative_location(self):
        relative = "local_models/reference_tokenizer"
        relocated = self.root / relative
        relocated.parent.mkdir()
        self.writer.rename(relocated)
        self.writer = relocated
        result = self.run_package(writer_model_path=relative, include_reference_tokenizer=True)
        extracted = self.extract("data.tar.gz")
        manifest = json.loads((extracted / packager.MANIFEST_PATH).read_text())
        self.assertEqual(manifest["reference_tokenizer_path"], relative)
        self.assertEqual(result["reference_tokenizer_path"], relative)
        self.assertTrue((extracted / relative / "tokenizer.json").is_file())
        self.assertFalse((extracted / packager.DEFAULT_WRITER).exists())

    def test_external_reference_tokenizer_bundles_at_default_relative_location(self):
        external = self.home / "external_writer"
        self.writer.rename(external)
        self.run_package(writer_model_path=external, include_reference_tokenizer=True)
        extracted = self.extract("data.tar.gz")
        manifest = json.loads((extracted / packager.MANIFEST_PATH).read_text())
        self.assertEqual(manifest["reference_tokenizer_path"], packager.DEFAULT_WRITER)
        self.assertEqual(manifest["models"]["writer_tokenizer"]["source_model_path"], str(external))
        self.assertEqual((extracted / packager.DEFAULT_WRITER / "tokenizer.json").read_bytes(),
                         (external / "tokenizer.json").read_bytes())

    def test_corrupted_reference_tokenizer_copy_is_rejected(self):
        before = self.source_state()
        original_copy = packager.shutil.copy2

        def corrupt_copy(source, target):
            copied = original_copy(source, target)
            if Path(source) == self.writer / "tokenizer.json":
                target.write_bytes(b"X" * target.stat().st_size)
            return copied

        with patch.object(packager.shutil, "copy2", side_effect=corrupt_copy):
            with self.assertRaisesRegex(ValueError, "Copied reference tokenizer SHA256 or size differs"):
                self.run_package(include_reference_tokenizer=True)
        self.assertEqual(self.source_state(), before)
        self.assertFalse((self.output / "COMPLETE.json").exists())
        self.assertFalse((self.output / "data.tar.gz").exists())
        self.assertTrue((self.output / ".packaging-in-progress").exists())

    def test_reference_tokenizer_dry_run_reports_assets_without_copying(self):
        before = self.source_state()
        with patch.object(packager, "_model_proof", side_effect=AssertionError("no full model hashes")), \
                patch.object(packager.shutil, "copy2", side_effect=AssertionError("no copying")):
            result = self.run_package(dry_run=True, include_reference_tokenizer=True)
        self.assertEqual(result["reference_tokenizer_path"], packager.DEFAULT_WRITER)
        self.assertEqual(result["reference_tokenizer_bytes"],
                         sum(path.stat().st_size for path in packager.tokenizer_file_paths(self.writer)))
        self.assertIn("reference_tokenizer_copy", result["dry_run_skips"])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source_state(), before)

    def test_cli_reference_tokenizer_flag_is_opt_in(self):
        for arguments, expected in (([], False), (["--include-reference-tokenizer"], True)):
            with self.subTest(arguments=arguments), patch.object(packager, "package") as package:
                packager.main(["--output", str(self.output), *arguments])
            self.assertIs(package.call_args.kwargs["include_reference_tokenizer"], expected)

    def test_dry_run_does_not_write_or_hash_model_weights(self):
        before = self.source_state()
        with patch.object(packager, "_model_proof", side_effect=AssertionError("no full model hashes")):
            result = self.run_package(dry_run=True)
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["source_states"], {"complete": 1, "failed": 1, "pending": 1})
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source_state(), before)

    def test_nonempty_relative_and_source_outputs_are_rejected(self):
        self.output.mkdir()
        sentinel = self.output / "keep"
        sentinel.write_text("original")
        for path in (self.output, Path("relative-package"), self.source / "package", self.writer, self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                packager.package(output=path, repo_root=self.root)
        self.assertEqual(sentinel.read_text(), "original")

    def test_changed_dataset_is_rejected_before_output_creation(self):
        (self.data / "train.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Dataset SHA256 changed"):
            self.run_package()
        self.assertFalse(self.output.exists())

    def test_changed_model_identity_is_rejected_before_output_creation(self):
        for model, filename, message in ((self.writer, "tokenizer.json", "Writer tokenizer"),
                                         (self.embedding, "1_Pooling/config.json", "embedding identity")):
            path = model / filename
            original = path.read_bytes()
            stat = path.stat()
            path.write_text('{"changed": true}')
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, message):
                self.run_package()
            self.assertFalse(self.output.exists())
            path.write_bytes(original)
            import os
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    def test_source_lock_and_recovery_sidecars_are_respected(self):
        with (self.source / ".build.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "builder is active"):
                self.run_package()
            fcntl.flock(lock, fcntl.LOCK_UN)
        sidecar = self.source / "build_journal.sqlite3-wal"
        sidecar.write_bytes(b"owner unfinished")
        with self.assertRaisesRegex(ValueError, "recovery/WAL"):
            self.run_package()
        self.assertEqual(sidecar.read_bytes(), b"owner unfinished")
        self.assertFalse(self.output.exists())

    def test_failure_retains_only_owned_staging_without_completion(self):
        with patch.object(packager, "_compact_journal", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_package()
        self.assertFalse((self.output / "COMPLETE.json").exists())
        self.assertTrue((self.output / ".packaging-in-progress").is_file())
        self.assertEqual(len(list(self.output.glob(".staging-*"))), 1)
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.run_package()


if __name__ == "__main__":
    unittest.main()
