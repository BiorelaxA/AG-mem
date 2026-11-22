"""Small real-file proofs for relocated models and frozen source datasets."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from agmem import build_adjacent_directories as engine
from agmem import import_deepseek_directories as importer
from agmem import portable_directory_identity as portable
from agmem.core import file_digest


class PortableIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        old = self.home / "removed-source-host"
        self.project = self.home / "new-host"
        self.writer, self.embedding = self.project / "models/writer", self.project / "models/embedding"
        old_writer, old_embedding = old / "writer", old / "embedding"
        files = {"config.json": '{"hidden_size":2560}', "tokenizer.json": '{"version":"1.0"}',
                 "tokenizer_config.json": '{"model_max_length":4096}',
                 "special_tokens_map.json": '{"eos_token":"<eos>"}',
                 "chat_template.jinja": "{{ messages }}", "merges.txt": "a b\n",
                 "model.safetensors": "tiny-weight-fixture", "1_Pooling/config.json": '{"dimension":2560}'}
        for root in (old_writer, old_embedding):
            for name, value in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
                os.utime(path, ns=(1_600_000_000_000_000_000,) * 2)
        old_writer_identity, writer_tokenizer = engine._model_identity(old_writer)
        old_embedding_identity, embedding_tokenizer = engine._model_identity(old_embedding)
        self.old_writer_identity = old_writer_identity
        self.source = {"protocol": importer.SOURCE_PROTOCOL,
                       "datasets": [{"spec": {"name": "split_longmemeval_m", "input_root": str(old / "data")},
                                     "files": []}],
                       "writer": {"model": "deepseek-flash", "model_revision": "remote"},
                       "tokenizer_id": writer_tokenizer,
                       "embedding": {**old_embedding_identity, "tokenizer_id": embedding_tokenizer,
                                     "settings": {"model_id": "embedding", "instruction": "topic/event"}},
                       "writer_settings": {"child_content_tokens": 300}, "segmentation": {"threshold": .5},
                       "prompt_hashes": {"child.txt": "child", "parent.txt": "parent", "boundary.txt": "boundary"},
                       "prompt_version": "directory.v1", "block_policy": "block.v1",
                       "boundary_prompt_version": "topic_event.v2", "base_resource_hashes": {"schema": "fixed"}}
        data = []
        for split in ("train", "validation", "test"):
            path = self.project / "data" / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"immutable {split} raw data".encode())
            row = self.entry(path, self.project)
            data.append({**row, "split": split})
            self.source["datasets"][0]["files"].append(
                {"path": str(old / "data" / path.name), "split": split, "sha256": row["sha256"]})
        self.manifest = {"version": portable.VERSION, "source_root_relative": "runs/deepseek",
                         "source_identity": deepcopy(self.source), "data_files": data,
                         "models": {
                             "writer_tokenizer": {"source_model_path": str(old_writer), "tokenizer_id": writer_tokenizer,
                                                  "files": self.model_entries(old_writer, tokenizer_only=True)},
                             "embedding": {"source_model_path": str(old_embedding), "identity": deepcopy(self.source["embedding"]),
                                           "files": self.model_entries(old_embedding)}}}
        shutil.copytree(old_writer, self.writer)
        shutil.copytree(old_embedding, self.embedding)
        for root in (self.writer, self.embedding):
            for path in root.rglob("*"):
                if path.is_file():
                    os.utime(path, ns=(1_700_000_000_000_000_123,) * 2)
        shutil.rmtree(old)
        self.target = deepcopy(self.source)
        self.target.update(protocol=importer.TARGET_PROTOCOL, writer={"model": "new-qwen", "model_revision": "local"})
        self.target["datasets"][0]["spec"]["input_root"] = str(self.project / "data")
        for row in self.target["datasets"][0]["files"]:
            row["path"] = str(self.project / "data" / Path(row["path"]).name)
        self.manifest_path = self.project / "migration/longmemeval_m_continuation.json"
        self.manifest_path.parent.mkdir()
        self.write_manifest()

    @staticmethod
    def entry(path, root):
        return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": file_digest(path)}

    def model_entries(self, root, **kwargs):
        return [self.entry(root / name, root) for name in portable.model_files(root, **kwargs)]

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def verifier(self):
        return portable.PortableIdentity(self.manifest_path, self.project)

    def verify(self, verifier=None, base=engine._model_identity):
        verifier = verifier or self.verifier()
        verifier.verify_models(self.writer, self.embedding, base)
        return verifier

    def snapshot(self):
        return {str(path.relative_to(self.project)): (file_digest(path), path.stat().st_size,
                path.stat().st_mtime_ns, path.stat().st_ctime_ns, path.stat().st_mode)
                for path in self.project.rglob("*") if path.is_file()}

    def test_same_bytes_different_paths_and_mtimes_alias_only_proved_identities(self):
        actual_writer, actual_writer_tokenizer = engine._model_identity(self.writer)
        actual_embedding, actual_embedding_tokenizer = engine._model_identity(self.embedding)
        self.assertNotEqual(actual_writer, self.old_writer_identity)
        self.assertNotEqual(actual_writer_tokenizer, self.source["tokenizer_id"])
        self.assertNotEqual(actual_embedding["model_revision"], self.source["embedding"]["model_revision"])
        self.assertNotEqual(actual_embedding_tokenizer, self.source["embedding"]["tokenizer_id"])
        verifier = self.verify()
        self.assertEqual(verifier.model_identity(self.writer), (actual_writer, self.source["tokenizer_id"]))
        expected = ({key: self.source["embedding"][key] for key in ("ready", "model_revision")},
                    self.source["embedding"]["tokenizer_id"])
        self.assertEqual(verifier.model_identity(self.embedding), expected)
        for name, path in (("writer_tokenizer", self.writer), ("embedding", self.embedding)):
            proof = verifier.report["models"][name]
            self.assertEqual(proof["local_model_path"], str(path))
            self.assertEqual(len(proof["content_proof_digest"]), 64)
            self.assertNotEqual(proof["actual_identity"]["tokenizer_id"], proof["effective_identity"]["tokenizer_id"])

    def test_verification_never_changes_files_or_metadata_or_source_identity(self):
        before, source = self.snapshot(), deepcopy(self.source)
        with patch("os.utime", side_effect=AssertionError("must not restore timestamps")):
            verifier = self.verify()
            verifier.verify_files(self.source, self.target)
        self.assertEqual(before, self.snapshot())
        self.assertEqual(self.source, source)

    def test_data_remapping_supports_nonexistent_absolute_old_paths(self):
        self.assertTrue(all(not Path(row["path"]).exists() for row in self.source["datasets"][0]["files"]))
        verifier = self.verifier()
        expected = {row["path"]: row["sha256"] for row in self.target["datasets"][0]["files"]}
        self.assertEqual(verifier.verify_files(self.source, self.target), expected)
        self.assertEqual(len(verifier.report["datasets"]), 3)
        for mapping in verifier.report["datasets"]:
            self.assertFalse(Path(mapping["source_path"]).exists())
            self.assertTrue(Path(mapping["target_path"]).is_file())

    def test_full_content_hashing_and_base_identity_are_cached_per_instance(self):
        base = Mock(side_effect=engine._model_identity)
        verifier = self.verify(base=base)
        self.assertEqual(base.call_count, 2)
        with patch.object(portable, "file_digest", side_effect=AssertionError("must reuse content proof")):
            verifier.verify_models(self.writer, self.embedding, base)
            verifier.model_identity(self.writer)
            verifier.model_identity(self.embedding)
        self.assertEqual(base.call_count, 2)

    def test_changed_embedding_weights_configs_tokenizers_nested_metadata_and_extras_rejected(self):
        changes = [(self.writer, "tokenizer.json"), (self.writer, "chat_template.jinja"),
                   (self.writer, "merges.txt"), (self.embedding, "model.safetensors"),
                   (self.embedding, "config.json"), (self.embedding, "tokenizer_config.json"),
                   (self.embedding, "1_Pooling/config.json"), (self.embedding, "chat_template.jinja")]
        for root, name in changes:
            path = root / name
            original = path.read_bytes()
            path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.subTest(root=root.name, path=name), self.assertRaisesRegex(ValueError, "SHA256"):
                self.verify()
            path.write_bytes(original)

    def test_exact_relevant_file_sets_reject_added_and_missing_files(self):
        changes = [(self.writer, "extra_tokenizer.json"), (self.writer, "tokenizer.model"),
                   (self.embedding, "extra.safetensors"), (self.embedding, "2_Dense/config.json")]
        for root, name in changes:
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("{}")
            with self.subTest(path=name), self.assertRaisesRegex(ValueError, "file set differs"):
                self.verify()
            path.unlink()
        (self.embedding / "merges.txt").unlink()
        with self.assertRaisesRegex(ValueError, "file set differs"):
            self.verify()

    def test_writer_weights_and_config_remain_new_local_provenance(self):
        (self.writer / "model.safetensors").write_bytes(b"different writer weights are allowed")
        (self.writer / "config.json").write_text('{"new_writer_config":true}')
        verifier = self.verify()
        self.assertEqual(verifier.model_identity(self.writer)[0], engine._model_identity(self.writer)[0])
        self.assertEqual(verifier.model_identity(self.writer)[1], self.source["tokenizer_id"])

    def test_download_cache_and_readme_do_not_enter_embedding_proof(self):
        cache = self.embedding / ".cache/huggingface/metadata.json"
        cache.parent.mkdir(parents=True)
        cache.write_text('{"download_time":"new-host"}')
        (self.embedding / "README.md").write_text("New download location")
        self.verify()

    def test_no_alias_before_both_proofs_pass_and_no_unmapped_models(self):
        verifier = self.verifier()
        with self.assertRaisesRegex(ValueError, "No verified"):
            verifier.model_identity(self.writer, allow_missing=True)
        (self.embedding / "model.safetensors").write_text("corrupt")
        with self.assertRaises(ValueError):
            self.verify(verifier)
        with self.assertRaisesRegex(ValueError, "No verified"):
            verifier.model_identity(self.writer)
        with self.assertRaisesRegex(ValueError, "No verified"):
            verifier.model_identity(self.project / "other", allow_missing=True)

    def test_cached_alias_rejects_subsequent_model_mutation(self):
        verifier = self.verify()
        (self.embedding / "model.safetensors").write_bytes(b"changed after proof")
        with self.assertRaisesRegex(ValueError, "changed since"):
            verifier.model_identity(self.embedding)
        with self.assertRaisesRegex(ValueError, "changed since"):
            self.verify(verifier)

    def test_changed_dataset_bytes_rejected_even_after_prior_verification(self):
        verifier = self.verifier()
        verifier.verify_files(self.source, self.target)
        path = self.project / "data/train.parquet"
        original = path.read_bytes()
        path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            verifier.verify_files(self.source, self.target)

    def test_source_journal_identity_must_match_manifest_exactly(self):
        source = deepcopy(self.source)
        source["writer"]["model"] = "different source"
        with self.assertRaisesRegex(ValueError, "Source journal identity"):
            self.verifier().verify_files(source, self.target)

    def test_target_dataset_hashes_splits_and_paths_must_match_manifest(self):
        for mutation in ("sha256", "split", "path"):
            target = deepcopy(self.target)
            target["datasets"][0]["files"][0][mutation] = "f" * 64 if mutation == "sha256" else "unmapped"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.verifier().verify_files(self.source, target)

    def test_portable_alias_keeps_existing_semantic_comparison_active(self):
        verifier = self.verify()
        identity, token = verifier.model_identity(self.embedding)
        target = deepcopy(self.target)
        target["embedding"] = {**identity, "tokenizer_id": token, "settings": self.source["embedding"]["settings"]}
        target["tokenizer_id"] = verifier.model_identity(self.writer)[1]
        importer.compatible_identity(self.source, target)
        target["segmentation"] = {"threshold": .99}
        with self.assertRaisesRegex(ValueError, "segmentation"):
            importer.compatible_identity(self.source, target)

    def test_unsafe_manifest_paths_are_rejected(self):
        for value in ("/absolute", "../escape", "a/../escape", "a//b", "a/./b", "C:/models", "a\\b", ""):
            with self.subTest(path=value), self.assertRaises(ValueError):
                portable.validate_path(value)
        for location in ("source", "data", "model"):
            manifest = deepcopy(self.manifest)
            if location == "source":
                manifest["source_root_relative"] = "../escape"
            elif location == "data":
                manifest["data_files"][0]["path"] = "/absolute"
            else:
                manifest["models"]["embedding"]["files"][0]["path"] = "../escape"
            self.manifest_path.write_text(json.dumps(manifest))
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "safe relative path"):
                self.verifier()

    def test_manifest_aliases_cannot_disagree_with_source_identity(self):
        for model in ("writer_tokenizer", "embedding"):
            manifest = deepcopy(self.manifest)
            if model == "embedding":
                manifest["models"][model]["identity"]["model_revision"] = "unproved"
            else:
                manifest["models"][model]["tokenizer_id"] = "unproved"
            self.manifest_path.write_text(json.dumps(manifest))
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "differs"):
                self.verifier()


if __name__ == "__main__":
    unittest.main()
