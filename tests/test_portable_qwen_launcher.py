"""Relocation proof reaches every subprocess and stays scoped to this worker."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agmem import build_portable_qwen_directories as portable


class PortableQwenLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text("{}")
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"model_path": "writer", "embedding": {"model_path": "embed"}}))

    def test_launcher_passes_same_manifest_to_all_modes_and_restores_dispatch(self):
        args = portable.argument_parser().parse_args(["--migration-manifest", str(self.manifest)])
        old_command = portable.launcher.build_command
        commands = []

        def launch(given):
            for kind in ("dry_run", "import_only", None):
                selected = deepcopy(given)
                if kind:
                    setattr(selected, kind, True)
                commands.append(portable.launcher.build_command(selected, ["http://local/v1"], "http://embed/v1"))
            raise RuntimeError("synthetic launcher interruption")

        with patch.object(portable.launcher, "launch", side_effect=launch), \
                self.assertRaisesRegex(RuntimeError, "interruption"):
            portable.launch(args)
        self.assertIs(portable.launcher.build_command, old_command)
        for command in commands:
            self.assertIn("agmem.build_portable_qwen_directories", command)
            self.assertEqual(command[command.index("agmem.build_portable_qwen_directories") + 1], "worker")
            self.assertEqual(command[command.index("--migration-manifest") + 1], str(self.manifest))
            self.assertEqual(command[command.index("--workers") + 1], "256")
        self.assertIn("--dry-run", commands[0])
        self.assertIn("--import-only", commands[1])

    def verifier(self):
        proof = Mock()
        proof.project_root = self.root
        proof.manifest = {"source_identity": {"protocol": "source"}, "source_root_relative": "source"}
        proof.report = {"models": {"embedding": {"verified": True, "sha256": "content-proof"}}}
        return proof

    def test_worker_scopes_aliases_and_records_proof_without_mutating_source(self):
        args = portable.argument_parser(worker_mode=True).parse_args([
            "--config", str(self.config), "--migration-manifest", str(self.manifest), "--dry-run"])
        proof = self.verifier()
        old_model, old_verify = portable.engine._model_identity, portable.importer._verify_files
        identity = {"implementation_hashes": {}}

        def run(given, **factories):
            self.assertIs(portable.engine._model_identity, proof.model_identity)
            self.assertIs(portable.importer._verify_files, proof.verify_files)
            self.assertEqual(given.source_root, str(self.root / "source"))
            _, built = portable.worker._BASE_LOAD_RECIPE("ignored")
            self.assertEqual(built["portable_migration"]["models"], proof.report["models"])
            self.assertIn("portable_directory_identity.py", built["implementation_hashes"])
            self.assertIn("build_portable_qwen_directories.py", built["implementation_hashes"])
            return built

        with patch.object(portable, "PortableIdentity", return_value=proof), \
                patch.object(portable.worker, "_BASE_LOAD_RECIPE", return_value=({}, identity)) as base, \
                patch.object(portable.worker, "run", side_effect=run), patch.object(portable.engine, "announce"):
            portable.run(args)
            self.assertIs(portable.worker._BASE_LOAD_RECIPE, base)
        proof.verify_models.assert_called_once_with("writer", "embed", old_model)
        self.assertIs(portable.engine._model_identity, old_model)
        self.assertIs(portable.importer._verify_files, old_verify)

    def test_source_override_cannot_bypass_proof_and_context_restores_on_error(self):
        args = portable.argument_parser(worker_mode=True).parse_args([
            "--config", str(self.config), "--migration-manifest", str(self.manifest),
            "--source-root", str(self.root / "wrong"), "--dry-run"])
        old_model, old_verify, old_load = (portable.engine._model_identity, portable.importer._verify_files,
                                          portable.worker._BASE_LOAD_RECIPE)
        with patch.object(portable, "PortableIdentity", return_value=self.verifier()), \
                patch.object(portable.worker, "run") as run, patch.object(portable.engine, "announce"), \
                self.assertRaisesRegex(ValueError, "source-root"):
            portable.run(args)
        run.assert_not_called()
        self.assertEqual((portable.engine._model_identity, portable.importer._verify_files,
                          portable.worker._BASE_LOAD_RECIPE), (old_model, old_verify, old_load))

    def test_failed_content_verification_never_enters_original_worker(self):
        args = portable.argument_parser(worker_mode=True).parse_args([
            "--config", str(self.config), "--migration-manifest", str(self.manifest), "--dry-run"])
        proof = self.verifier()
        proof.verify_models.side_effect = ValueError("wrong embedding weights")
        with patch.object(portable, "PortableIdentity", return_value=proof), \
                patch.object(portable.worker, "run") as run, self.assertRaisesRegex(ValueError, "weights"):
            portable.run(args)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
