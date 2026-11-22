import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from agmem.checkpoints import (
    OVERWRITE_RETENTION_POLICY,
    cleanup_overwrite_temporaries,
    committed_checkpoints,
    prepare_checkpoint_overwrite,
    refresh_checkpoint_aliases,
    retain_checkpoints,
    select_best_checkpoints,
)
from agmem.core import atomic_json


class OverwriteCheckpointRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def committed(self, step, reward=None, outcome=None, *, primary=True):
        checkpoint = self.root / f"checkpoint-{step:08d}"
        atomic_json(checkpoint / "model/config.json", {})
        (checkpoint / "trainer.pt").write_bytes(b"optimizer and trainer state")
        if reward is not None:
            report = {"source_macro": {"reward": reward, "outcome_f1": outcome}}
            if primary:
                report["primary_outcome_metric"] = "outcome_f1"
            atomic_json(checkpoint / "validation_metrics.json", report)
        return checkpoint

    def retain(self):
        return retain_checkpoints(self.root, 0, policy=OVERWRITE_RETENTION_POLICY)

    def names(self):
        return [path.name for path in committed_checkpoints(self.root)]

    def assert_aliases(self, best, latest):
        for name, target in (("best", best), ("latest", latest)):
            alias = self.root / name
            self.assertTrue(alias.is_symlink())
            self.assertEqual(os.readlink(alias), target.name)
            self.assertEqual(alias.resolve(), target.resolve())
            self.assertEqual((alias / "trainer.pt").read_bytes(), b"optimizer and trainer state")

    def test_prepare_releases_previous_latest_before_replacement_exists(self):
        reward_best = self.committed(1, 1, 0.1)
        primary_best = self.committed(2, 0.5, 0.9)
        latest = self.committed(3, 0.7, 0.8)
        refresh_checkpoint_aliases(self.root)
        best = prepare_checkpoint_overwrite(self.root, 4)
        self.assertEqual(self.names(), [primary_best.name])
        self.assertFalse(reward_best.exists())
        self.assertFalse(latest.exists())
        self.assertFalse((self.root / "checkpoint-00000004.building").exists())
        self.assert_aliases(primary_best, primary_best)
        self.assertEqual(best, select_best_checkpoints(self.root))
        self.assertEqual(best, json.loads((self.root / "best_checkpoints.json").read_text()))
        self.assertEqual(best["best_reward"]["checkpoint"], primary_best.name)

    def test_prepare_keeps_only_copy_when_best_already_is_latest(self):
        latest = self.committed(1, 0.8, 0.9)
        self.retain()
        prepare_checkpoint_overwrite(self.root, 2)
        self.assertEqual(self.names(), [latest.name])
        self.assert_aliases(latest, latest)
        self.assertEqual(prepare_checkpoint_overwrite(self.root, 2), select_best_checkpoints(self.root))

    def test_no_validation_history_protects_latest_complete_payload(self):
        first = self.committed(1)
        latest = self.committed(2)
        self.assertEqual(prepare_checkpoint_overwrite(self.root, 3), {})
        self.assertFalse(first.exists())
        self.assertEqual(self.names(), [latest.name])
        self.assert_aliases(latest, latest)

    def test_legacy_validation_report_uses_outcome_instead_of_reward(self):
        best = self.committed(1, 0.1, 0.9, primary=False)
        self.committed(2, 1, 0.8, primary=False)
        manifest = prepare_checkpoint_overwrite(self.root, 3)
        self.assertEqual(manifest["best_outcome"]["checkpoint"], best.name)
        self.assert_aliases(best, best)

    def test_retention_ties_keep_earliest_and_improved_latest_deduplicates(self):
        first = self.committed(1, 0.1, 0.9)
        self.committed(2, 1, 0.9)
        latest = self.committed(3, 0.2, 0.9)
        manifest = self.retain()
        self.assertEqual(self.names(), [first.name, latest.name])
        self.assertEqual(manifest["best_primary"]["checkpoint"], first.name)
        self.assertEqual(manifest["best_reward"]["checkpoint"], latest.name)
        self.assert_aliases(first, latest)
        self.assertEqual(self.retain(), manifest)
        prepare_checkpoint_overwrite(self.root, 4)
        improved = self.committed(4, 0.3, 1)
        self.retain()
        self.assertEqual(self.names(), [improved.name])
        self.assert_aliases(improved, improved)

    def test_invalid_or_nonincreasing_step_does_not_mutate_anything(self):
        best = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.7, 0.8)
        for step in (True, -1, 1.5, "3", None, 0, 1, 2):
            with self.subTest(step=step):
                with self.assertRaisesRegex(ValueError, "checkpoint step|integer step"):
                    prepare_checkpoint_overwrite(self.root, step)
                self.assertEqual(self.names(), [best.name, latest.name])
                self.assertFalse((self.root / "best_checkpoints.json").exists())
                self.assertFalse((self.root / "best").is_symlink())

    def test_incomplete_obsolete_candidate_rejected_before_any_mutation(self):
        incomplete = self.committed(1)
        (incomplete / "trainer.pt").unlink()
        best = self.committed(2, 0.8, 0.9)
        latest = self.committed(3, 0.7, 0.8)
        partial = self.root / "checkpoint-00000004.building"
        partial.mkdir()
        for action in (lambda: prepare_checkpoint_overwrite(self.root, 4), self.retain,
                       lambda: refresh_checkpoint_aliases(self.root),
                       lambda: cleanup_overwrite_temporaries(self.root)):
            with self.subTest(action=action):
                with self.assertRaisesRegex(ValueError, "Incomplete committed checkpoint"):
                    action()
                self.assertEqual(self.names(), [incomplete.name, best.name, latest.name])
                self.assertTrue(partial.is_dir())
                self.assertFalse((self.root / "best_checkpoints.json").exists())
                self.assertFalse((self.root / "best").is_symlink())

    def test_invalid_obsolete_validation_score_rejected_before_deletion(self):
        invalid = self.committed(1, float("nan"), 0.1)
        best = self.committed(2, 0.8, 0.9)
        latest = self.committed(3, 0.7, 0.8)
        for action in (lambda: prepare_checkpoint_overwrite(self.root, 4), self.retain):
            with self.assertRaisesRegex(ValueError, "finite validation"):
                action()
            self.assertEqual(self.names(), [invalid.name, best.name, latest.name])
            self.assertFalse((self.root / "best_checkpoints.json").exists())

    def test_alias_directory_or_file_collision_rejected_before_mutation(self):
        best = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.7, 0.8)
        for name in ("best", "latest"):
            for directory in (False, True):
                with self.subTest(name=name, directory=directory):
                    collision = self.root / name
                    if directory:
                        collision.mkdir()
                    else:
                        collision.write_bytes(b"user data")
                    for action in (lambda: prepare_checkpoint_overwrite(self.root, 3), self.retain):
                        with self.assertRaisesRegex(ValueError, "alias name is already occupied"):
                            action()
                    self.assertEqual(self.names(), [best.name, latest.name])
                    self.assertFalse((self.root / "best_checkpoints.json").exists())
                    collision.rmdir() if directory else collision.unlink()

    def test_unknown_alias_symlink_is_never_replaced(self):
        best = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.7, 0.8)
        targets = (str(best), "../checkpoint-00000001", "./checkpoint-00000001",
                   "checkpoint-00000001/model", "checkpoint-custom", "checkpoint-²")
        for name in ("best", "latest"):
            for target in targets:
                with self.subTest(name=name, target=target):
                    alias = self.root / name
                    alias.symlink_to(target, target_is_directory=True)
                    with self.assertRaisesRegex(ValueError, "unknown checkpoint alias"):
                        prepare_checkpoint_overwrite(self.root, 3)
                    self.assertEqual(os.readlink(alias), target)
                    self.assertEqual(self.names(), [best.name, latest.name])
                    self.assertFalse((self.root / "best_checkpoints.json").exists())
                    alias.unlink()

    def test_committed_payload_symlink_cannot_escape_run_directory(self):
        best = self.committed(1, 0.5, 0.9)
        outside = self.root / "user-checkpoint"
        atomic_json(outside / "model/config.json", {})
        (outside / "trainer.pt").write_bytes(b"preserve user data")
        alias = self.root / "checkpoint-00000002"
        alias.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            prepare_checkpoint_overwrite(self.root, 3)
        self.assertTrue(best.is_dir())
        self.assertTrue(alias.is_symlink())
        self.assertEqual((outside / "trainer.pt").read_bytes(), b"preserve user data")
        self.assertFalse((self.root / "best_checkpoints.json").exists())

    def test_refresh_repairs_owned_dangling_links_without_rewriting_manifest(self):
        best = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.7, 0.8)
        for name in ("best", "latest"):
            (self.root / name).symlink_to("checkpoint-00000000", target_is_directory=True)
        atomic_json(self.root / "best_checkpoints.json", {"fixture": "unchanged"})
        manifest = refresh_checkpoint_aliases(self.root)
        self.assert_aliases(best, latest)
        self.assertEqual(manifest, select_best_checkpoints(self.root))
        self.assertEqual(json.loads((self.root / "best_checkpoints.json").read_text()),
                         {"fixture": "unchanged"})

    def test_cleanup_only_removes_exact_numeric_partial_directories(self):
        best = self.committed(1, 0.5, 0.9)
        removed = [self.root / f"checkpoint-00000002{suffix}" for suffix in (".building", ".pruning")]
        preserved = [self.root / name for name in (
            "checkpoint-manual.building", "checkpoint-².pruning", "checkpoint-.building",
            "checkpoint-00000003.building.backup", "checkpoint-00000003.building.pruning")]
        for path in removed + preserved:
            path.mkdir()
            (path / "data").write_bytes(b"fixture")
        partial_file = self.root / "checkpoint-00000004.building"
        partial_file.write_bytes(b"preserve")
        partial_link = self.root / "checkpoint-00000005.pruning"
        partial_link.symlink_to(preserved[0], target_is_directory=True)
        self.assertEqual(cleanup_overwrite_temporaries(self.root), removed)
        self.assertEqual(cleanup_overwrite_temporaries(self.root), [])
        self.assertEqual(self.names(), [best.name])
        self.assertTrue(partial_link.is_symlink())
        self.assertEqual(partial_file.read_bytes(), b"preserve")
        for path in preserved:
            self.assertEqual((path / "data").read_bytes(), b"fixture")

    def test_partial_prune_failure_preserves_best_and_retry_cleans_all_debris(self):
        best = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.7, 0.8)
        failed_save = self.root / "checkpoint-00000003.building"
        failed_save.mkdir()
        (failed_save / "partial").write_bytes(b"unfinished model")
        pending = latest.with_name(latest.name + ".pruning")
        real_rmtree = shutil.rmtree

        def interrupted_delete(path):
            if path == pending:
                (path / "trainer.pt").unlink()
                raise OSError("interrupted deletion")
            real_rmtree(path)

        with patch("agmem.checkpoints.shutil.rmtree", side_effect=interrupted_delete):
            with self.assertRaisesRegex(OSError, "interrupted deletion"):
                prepare_checkpoint_overwrite(self.root, 3)
        self.assertEqual(self.names(), [best.name])
        self.assert_aliases(best, best)
        self.assertTrue(pending.exists())
        self.assertTrue(failed_save.exists())
        manifest = prepare_checkpoint_overwrite(self.root, 3)
        self.assertEqual(manifest, select_best_checkpoints(self.root))
        self.assertFalse(pending.exists())
        self.assertFalse(failed_save.exists())
        self.assertEqual(prepare_checkpoint_overwrite(self.root, 3), manifest)

    def test_manifest_publish_failure_preserves_all_payloads(self):
        best = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.7, 0.8)
        self.retain()
        previous = (self.root / "best_checkpoints.json").read_bytes()
        with patch("agmem.checkpoints.atomic_json", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                prepare_checkpoint_overwrite(self.root, 3)
        self.assertEqual(self.names(), [best.name, latest.name])
        self.assertEqual((self.root / "best_checkpoints.json").read_bytes(), previous)
        self.assert_aliases(best, latest)

    def test_alias_publish_failure_preserves_all_payloads_and_can_retry(self):
        best = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.7, 0.8)
        self.retain()
        real_replace = Path.replace

        def failed_alias_replace(path, target):
            if Path(target).name == "latest":
                raise OSError("alias replacement failed")
            return real_replace(path, target)

        with patch.object(Path, "replace", failed_alias_replace):
            with self.assertRaisesRegex(OSError, "alias replacement failed"):
                prepare_checkpoint_overwrite(self.root, 3)
        self.assertEqual(self.names(), [best.name, latest.name])
        self.assert_aliases(best, latest)
        self.assertEqual(list(self.root.glob(".latest.*.tmp")), [])
        prepare_checkpoint_overwrite(self.root, 3)
        self.assertEqual(self.names(), [best.name])
        self.assert_aliases(best, best)

    def test_retention_ignores_uncommitted_better_validation(self):
        best = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.7, 0.8)
        partial = self.root / "checkpoint-00000003.building"
        atomic_json(partial / "validation_metrics.json", {
            "primary_outcome_metric": "outcome_f1", "source_macro": {"reward": 1, "outcome_f1": 1}})
        self.retain()
        self.assertEqual(self.names(), [best.name, latest.name])
        self.assert_aliases(best, latest)
        self.assertTrue(partial.exists())


if __name__ == "__main__":
    unittest.main()
