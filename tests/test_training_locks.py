"""Real flock behavior for launcher lock aliasing; no GPU/process operations."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agmem import four_gpu
from agmem.jobs import file_lock


class TrainingLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def config(self, *, prepared=None, output=None, sources=None):
        return {"data": {"prepared_root": str(prepared or self.root / "shared" / "prepared"),
                         "sources": sources or ["locomo"]},
                "training": {"output_dir": str(output or self.root / "experiment" / "train")}}

    def test_same_parent_combines_shared_and_exclusive_guards_once(self):
        config = self.config(prepared=self.root / "experiment" / "prepared")
        path = self.root / "experiment" / ".job.lock"
        calls = []

        def tracked(lock_path, shared=False, **kwargs):
            calls.append((Path(lock_path), shared))
            return file_lock(lock_path, shared=shared, **kwargs)

        with patch.object(four_gpu, "file_lock", side_effect=tracked):
            with four_gpu.training_locks(config):
                with self.assertRaisesRegex(RuntimeError, "Another job"):
                    with file_lock(path, shared=True):
                        self.fail("A combined experiment guard must be exclusive")
        self.assertEqual(calls.count((path.resolve(), False)), 1)
        self.assertEqual(len(calls), 2)
        self.assertNotIn((path.resolve(), True), calls)

    def test_another_invocation_of_same_experiment_is_rejected(self):
        config = self.config(prepared=self.root / "experiment" / "prepared")
        with four_gpu.training_locks(config):
            with self.assertRaisesRegex(RuntimeError, "Another job"):
                with four_gpu.training_locks(config):
                    self.fail("Concurrent mutation of an experiment must be rejected")
        with four_gpu.training_locks(config):
            pass

    def test_shared_prepared_data_allows_distinct_sources_and_experiments(self):
        first = self.config(output=self.root / "first" / "train", sources=["locomo"])
        second = self.config(output=self.root / "second" / "train", sources=["split_longmemeval_s"])
        with four_gpu.training_locks(first), four_gpu.training_locks(second):
            with file_lock(self.root / "shared" / ".job.lock", shared=True):
                pass
            with self.assertRaisesRegex(RuntimeError, "Another job"):
                with file_lock(self.root / "shared" / ".job.lock"):
                    self.fail("A writer requiring the global prepared guard must wait")

    def test_same_source_conflicts_even_with_distinct_output_directories(self):
        first = self.config(output=self.root / "first" / "train")
        second = self.config(output=self.root / "second" / "train")
        with four_gpu.training_locks(first):
            with self.assertRaisesRegex(RuntimeError, r"locomo\.lock"):
                with four_gpu.training_locks(second):
                    self.fail("The selected source remains protected")
            # The rejected invocation must not leak its already acquired output guard.
            with file_lock(self.root / "second" / ".job.lock"):
                pass

    def test_all_guards_release_when_training_body_raises(self):
        config = self.config(sources=["locomo", "split_longmemeval_s"])
        with self.assertRaisesRegex(ValueError, "training failed"):
            with four_gpu.training_locks(config):
                raise ValueError("training failed")
        with four_gpu.training_locks(config):
            pass
        # Lock files persist; deleting them could create unlocked replacement inodes.
        self.assertTrue((self.root / "shared" / ".job.lock").is_file())
        self.assertTrue((self.root / "experiment" / ".job.lock").is_file())

    def test_partial_acquisition_failure_releases_all_earlier_guards(self):
        config = self.config(sources=["a", "z"])
        prepared = Path(config["data"]["prepared_root"])
        with file_lock(prepared / ".locks" / "z.lock"):
            with self.assertRaisesRegex(RuntimeError, r"z\.lock"):
                with four_gpu.training_locks(config):
                    self.fail("Pre-existing source lock must reject startup")
            with file_lock(prepared.parent / ".job.lock"), \
                    file_lock(self.root / "experiment" / ".job.lock"), \
                    file_lock(prepared / ".locks" / "a.lock"):
                pass

    def test_distinct_legacy_guards_retain_order_and_modes(self):
        config = self.config(sources=["split_longmemeval_s", "locomo"])
        prepared = Path(config["data"]["prepared_root"])
        calls = []

        def tracked(path, shared=False, **kwargs):
            calls.append((Path(path), shared))
            return file_lock(path, shared=shared, **kwargs)

        with patch.object(four_gpu, "file_lock", side_effect=tracked):
            with four_gpu.training_locks(config):
                pass
        self.assertEqual(calls, [
            ((prepared.parent / ".job.lock").resolve(), True),
            ((self.root / "experiment" / ".job.lock").resolve(), False),
            ((prepared / ".locks" / "locomo.lock").resolve(), False),
            ((prepared / ".locks" / "split_longmemeval_s.lock").resolve(), False),
        ])

    def test_symbolic_parent_alias_is_merged_before_acquiring_any_guard(self):
        real, alias = self.root / "real", self.root / "alias"
        real.mkdir()
        alias.symlink_to(real, target_is_directory=True)
        config = self.config(prepared=real / "prepared", output=alias / "train")
        with four_gpu.training_locks(config):
            with self.assertRaisesRegex(RuntimeError, "Another job"):
                with file_lock(real / ".job.lock", shared=True):
                    self.fail("Aliased output guard must retain exclusive protection")
        with four_gpu.training_locks(config):
            pass


if __name__ == "__main__":
    unittest.main()
