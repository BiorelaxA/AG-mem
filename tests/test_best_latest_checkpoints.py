import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agmem.checkpoints import committed_checkpoints, retain_checkpoints, select_best_checkpoints
from agmem.core import atomic_json
from agmem.dapo_recovery import reconcile_run


class BestLatestCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def committed(self, step, reward=None, outcome=None, *, primary=True):
        checkpoint = self.root / f"checkpoint-{step:08d}"
        atomic_json(checkpoint / "model/config.json", {})
        (checkpoint / "trainer.pt").write_bytes(b"fixture")
        if reward is not None:
            report = {"source_macro": {"reward": reward, "outcome_f1": outcome},
                      "pooled": {"reward": -reward, "outcome_f1": 1 - outcome}}
            if primary:
                report["primary_outcome_metric"] = "outcome_f1"
            atomic_json(checkpoint / "validation_metrics.json", report)
        return checkpoint

    def retain(self, keep_last=2):
        return retain_checkpoints(self.root, keep_last, policy="best_and_latest")

    def names(self):
        return [path.name for path in committed_checkpoints(self.root)]

    def test_retains_primary_best_and_latest_without_extra_reward_best(self):
        reward_best = self.committed(1, 0.99, 0.1)
        primary_best = self.committed(2, 0.5, 0.9)
        latest = self.committed(3, 0.6, 0.8)
        best = self.retain()
        self.assertEqual(self.names(), [primary_best.name, latest.name])
        self.assertFalse(reward_best.exists())
        self.assertEqual(best["best_primary"]["checkpoint"], primary_best.name)
        self.assertEqual(best["best_reward"]["checkpoint"], latest.name)
        self.assertEqual(best, select_best_checkpoints(self.root))
        self.assertEqual(best, json.loads((self.root / "best_checkpoints.json").read_text()))
        for entry in best.values():
            self.assertTrue((self.root / entry["checkpoint"] / "trainer.pt").is_file())

    def test_latest_new_best_merges_to_one_checkpoint(self):
        self.committed(1, 0.9, 0.7)
        self.committed(2, 0.8, 0.8)
        latest = self.committed(3, 0.7, 0.9)
        self.retain()
        self.assertEqual(self.names(), [latest.name])

    def test_primary_ties_preserve_earlier_checkpoint(self):
        first = self.committed(1, 0.1, 0.9)
        self.committed(2, 0.99, 0.9)
        latest = self.committed(3, 0.2, 0.9)
        best = self.retain()
        self.assertEqual(self.names(), [first.name, latest.name])
        self.assertEqual(best["best_primary"]["checkpoint"], first.name)
        self.assertEqual(self.retain(), best)

    def test_legacy_report_falls_back_to_outcome_best(self):
        self.committed(1, 0.99, 0.1, primary=False)
        primary_best = self.committed(2, 0.5, 0.9, primary=False)
        latest = self.committed(3, 0.6, 0.8, primary=False)
        best = self.retain()
        self.assertEqual(self.names(), [primary_best.name, latest.name])
        self.assertEqual(best["best_outcome"]["checkpoint"], primary_best.name)
        self.assertNotIn("best_primary", best)

    def test_no_validation_retains_latest_even_with_unlimited_legacy_count(self):
        self.committed(1)
        self.committed(2)
        latest = self.committed(3)
        self.assertEqual(self.retain(keep_last=0), {})
        self.assertEqual(self.names(), [latest.name])

    def test_nonvalidation_latest_can_resume_alongside_validation_best(self):
        first = self.committed(1, 0.2, 0.8)
        self.committed(2)
        latest = self.committed(3)
        self.retain()
        self.assertEqual(self.names(), [first.name, latest.name])

    def test_uncommitted_building_directory_is_ignored_and_preserved(self):
        first = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.6, 0.8)
        building = self.root / "checkpoint-00000003.building"
        atomic_json(building / "validation_metrics.json",
                    {"source_macro": {"reward": 1, "outcome_f1": 1}})
        self.retain()
        self.assertEqual(self.names(), [first.name, latest.name])
        self.assertTrue(building.is_dir())

    def test_incomplete_latest_aborts_before_manifest_or_deletion(self):
        first = self.committed(1, 0.5, 0.9)
        latest = self.committed(2, 0.6, 0.8)
        self.retain()
        previous = (self.root / "best_checkpoints.json").read_bytes()
        incomplete = self.root / "checkpoint-00000003"
        incomplete.mkdir()
        with self.assertRaisesRegex(ValueError, "Incomplete committed checkpoint"):
            self.retain()
        self.assertTrue(first.is_dir())
        self.assertTrue(latest.is_dir())
        self.assertEqual((self.root / "best_checkpoints.json").read_bytes(), previous)

    def test_manifest_write_failure_preserves_all_existing_checkpoints(self):
        for step, score in ((1, 0.7), (2, 0.9), (3, 0.8)):
            self.committed(step, score, score)
        before = self.names()
        with patch("agmem.checkpoints.atomic_json", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.retain()
        self.assertEqual(self.names(), before)

    def test_partial_delete_failure_is_excluded_from_recovery_and_retried(self):
        obsolete = self.committed(1, 0.99, 0.1)
        primary_best = self.committed(2, 0.5, 0.9)
        latest = self.committed(3, 0.6, 0.8)
        pending = obsolete.with_name(obsolete.name + ".pruning")

        def partial_delete(path):
            self.assertEqual(path, pending)
            self.assertFalse(obsolete.exists())
            (path / "trainer.pt").unlink()
            raise OSError("interrupted deletion")

        with patch("agmem.checkpoints.shutil.rmtree", side_effect=partial_delete):
            with self.assertRaisesRegex(OSError, "interrupted deletion"):
                self.retain()
        self.assertTrue((pending / "validation_metrics.json").is_file())
        self.assertEqual(self.names(), [primary_best.name, latest.name])
        for checkpoint in (primary_best, latest):
            self.assertEqual((checkpoint / "trainer.pt").read_bytes(), b"fixture")
            self.assertTrue((checkpoint / "model/config.json").is_file())
        best = json.loads((self.root / "best_checkpoints.json").read_text())
        self.assertEqual(select_best_checkpoints(self.root), best)
        result = reconcile_run(self.root, {"step": 3, "dapo_state": {"sampler": {}}})
        self.assertIsNone(result["archive"])
        self.assertTrue(pending.is_dir())
        self.assertEqual(self.retain(), best)
        self.assertFalse(pending.exists())
        self.assertEqual(self.names(), [primary_best.name, latest.name])

    def test_cleanup_only_removes_exact_numeric_pruning_directories(self):
        latest = self.committed(3, 0.6, 0.8)
        pending = self.root / "checkpoint-00000001.pruning"
        pending.mkdir()
        (pending / "partial").write_bytes(b"cleanup")
        preserved = [self.root / name for name in (
            "checkpoint-00000002.building", "checkpoint-00000002.building.pruning",
            "checkpoint-old.pruning", "checkpoint-00000002.pruning.backup",
            "checkpoint-.pruning", "checkpoint-².pruning")]
        for path in preserved:
            path.mkdir()
            (path / "keep").write_bytes(b"preserve")
        file = self.root / "checkpoint-00000000.pruning"
        file.write_bytes(b"preserve")
        symlink = self.root / "checkpoint-00000009.pruning"
        symlink.symlink_to(preserved[0], target_is_directory=True)
        self.retain()
        self.assertFalse(pending.exists())
        self.assertEqual(self.names(), [latest.name])
        for path in preserved:
            self.assertEqual((path / "keep").read_bytes(), b"preserve")
        self.assertEqual(file.read_bytes(), b"preserve")
        self.assertTrue(symlink.is_symlink())

    def test_legacy_does_not_clean_pending_pruning_directories(self):
        self.committed(1, 0.5, 0.9)
        pending = self.root / "checkpoint-00000000.pruning"
        pending.mkdir()
        retain_checkpoints(self.root, 2)
        self.assertTrue(pending.is_dir())

    def test_recovery_reconstructs_manifest_from_retained_checkpoints(self):
        self.committed(1, 0.99, 0.1)
        self.committed(2, 0.5, 0.9)
        self.committed(3, 0.6, 0.8)
        best = self.retain()
        atomic_json(self.root / "best_checkpoints.json", {"stale": {"checkpoint": "checkpoint-00000001"}})
        result = reconcile_run(self.root, {"step": 3, "dapo_state": {"sampler": {}}})
        self.assertIsNotNone(result["archive"])
        self.assertEqual(best, json.loads((self.root / "best_checkpoints.json").read_text()))
        self.assertEqual(best, select_best_checkpoints(self.root))

    def test_unknown_policy_fails_before_mutating_checkpoints(self):
        first = self.committed(1, 0.5, 0.9)
        with self.assertRaisesRegex(ValueError, "Unknown checkpoint retention policy"):
            retain_checkpoints(self.root, 2, policy="unknown")
        self.assertTrue(first.is_dir())
        self.assertFalse((self.root / "best_checkpoints.json").exists())


if __name__ == "__main__":
    unittest.main()
