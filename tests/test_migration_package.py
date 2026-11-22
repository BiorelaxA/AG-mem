"""Archive selection, integrity and failure handling on small local fixtures."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/package_migration_data.py"
SPEC = importlib.util.spec_from_file_location("package_migration_data", SCRIPT)
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


class MigrationPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.output = self.root / "packages"
        for name, content in {
            "data/old/train.parquet": "old rows",
            "data/new/train.parquet": "new rows",
            "runs/old/build_journal.sqlite3": "sqlite fixture",
            "runs/old/trajectory.pt": "non-model tensor data",
            "runs/old/checkpoint-00000001/trainer.pt": "old optimizer",
            "runs/current/checkpoint-00000020/model/model.safetensors": "selected model",
            "runs/current/checkpoint-00000032/trainer.pt": "selected optimizer",
            "runs/current/checkpoint-00000033.building/trainer.pt": "partial optimizer",
            "runs/current/run.json": "config and protocol",
            "reports/audit.json": "audit",
            "root log.out": "log",
            "agmem/source.py": "code on Github",
            "configs/local.yaml": "not in data package",
            ".codex/session.jsonl": "private conversation",
        }.items():
            p = self.project / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

    def package(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return packager.package(self.project, self.output, **kwargs)

    def test_dry_run_preserves_old_data_and_excludes_every_checkpoint(self):
        result = self.package(dry_run=True)
        self.assertFalse(self.output.exists())
        self.assertEqual(result["file_count"], 7)
        self.assertEqual(len(result["excluded_checkpoint_directories"]), 4)
        records, _ = packager.snapshot(self.project)
        selected = {row["path"] for row in records}
        self.assertIn("runs/old/trajectory.pt", selected)
        self.assertIn("runs/old/build_journal.sqlite3", selected)
        self.assertIn("data/old/train.parquet", selected)
        self.assertNotIn("agmem/source.py", selected)
        self.assertNotIn(".codex/session.jsonl", selected)

    @unittest.skipUnless(shutil.which("tar") and shutil.which("zstd"), "tar/zstd required")
    def test_round_trip_split_archive_checksums_and_nanosecond_times(self):
        import os
        file = self.project / "data/old/train.parquet"
        timestamp = 1700000000123456789
        os.utime(file, ns=(timestamp, timestamp))
        before = packager.snapshot(self.project)
        result = self.package(part_bytes=97, threads=1)
        self.assertGreater(len(result["parts"]), 1)
        self.assertEqual(packager.snapshot(self.project), before)
        checks = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=self.output,
                                capture_output=True, text=True)
        self.assertEqual(checks.returncode, 0, checks.stderr)
        marker = json.loads((self.output / "COMPLETE.json").read_text())
        self.assertEqual(marker["sha256sums_sha256"], packager.sha256_file(self.output / "SHA256SUMS"))
        compressed = b"".join((self.output / part["name"]).read_bytes() for part in result["parts"])
        archive = subprocess.run(["zstd", "-dc"], input=compressed, capture_output=True, check=True).stdout
        target = self.root / "restored"
        target.mkdir()
        subprocess.run(["tar", "--no-same-owner", "-xpf", "-", "-C", str(target)],
                       input=archive, capture_output=True, check=True)
        for record in before[0]:
            if record["type"] == "file":
                self.assertEqual((target / record["path"]).read_bytes(), (self.project / record["path"]).read_bytes())
        self.assertEqual((target / "data/old/train.parquet").stat().st_mtime_ns, timestamp)
        self.assertFalse((target / "runs/current/checkpoint-00000020").exists())
        self.assertTrue((self.project / "runs/current/checkpoint-00000020/model/model.safetensors").exists())

    def test_external_symlink_is_rejected(self):
        (self.project / "data/link").symlink_to(self.root / "outside")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.package(dry_run=True)

    def test_unclassified_weights_are_not_silently_archived_or_dropped(self):
        (self.project / "runs/old/model.safetensors").write_text("unknown model")
        with self.assertRaisesRegex(ValueError, "outside a recognized checkpoint"):
            self.package(dry_run=True)

    def test_nonempty_output_and_source_containment_are_rejected(self):
        self.output.mkdir()
        (self.output / "keep").write_text("existing")
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.package(dry_run=True)
        self.assertEqual((self.output / "keep").read_text(), "existing")
        with self.assertRaisesRegex(ValueError, "outside"):
            packager.package(self.project, self.project / "data/archive", dry_run=True)

    def test_compression_failure_never_marks_complete(self):
        with patch.object(packager, "pack_stream", side_effect=RuntimeError("compression failure")):
            with self.assertRaisesRegex(RuntimeError, "compression failure"):
                self.package()
        self.assertFalse((self.output / "COMPLETE.json").exists())

    def test_changes_after_streaming_prevent_completion(self):
        def change(project, output, records, part_bytes, threads):
            (project / "runs/current/run.json").write_text("changed after snapshot")
            return [{"name": "placeholder", "bytes": 1, "sha256": "0" * 64}]
        with patch.object(packager, "pack_stream", side_effect=change):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                self.package()
        self.assertFalse((self.output / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
