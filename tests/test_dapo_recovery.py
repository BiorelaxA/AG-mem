from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from agmem.dapo_recovery import reconcile_run


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def populate(root):
    jsonl(root / "train_metrics.jsonl", [{"step": i, "entropy": 0.5} for i in (1, 2, 3)])
    with (root / "train_metrics.jsonl").open("ab") as stream:
        stream.write(b'{"step":')  # A crash halfway through the final write.
    jsonl(root / "sampling_metrics.jsonl", [{"candidate_index": i} for i in (3, 6, 9)])
    jsonl(root / "candidate_metrics.jsonl", [{"candidate_id": i} for i in range(9)])
    for directory, prefix, suffix, indices in (
        ("groups", "group-", ".json.gz", range(3)),
        ("rollouts", "group-", ".json", range(3)),
        ("candidate_rollouts", "candidate-", ".json", range(9)),
    ):
        (root / directory).mkdir()
        for i in indices:
            (root / directory / f"{prefix}{i:08d}{suffix}").write_bytes(f"original-{directory}-{i}".encode())
    for i in (1, 2, 3):
        directory = root / "validation" / f"step-{i:08d}"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({"step": i}))
    for i in (2, 3):
        checkpoint = root / f"checkpoint-{i:08d}"
        (checkpoint / "model").mkdir(parents=True)
        (checkpoint / "model/config.json").write_text("{}")
        (checkpoint / "trainer.pt").write_bytes(f"original-torch-state-{i}".encode())
        (checkpoint / "validation_metrics.json").write_text(json.dumps({
            "source_macro": {"judge_accuracy": i / 10, "reward": i / 10}}))
    building = root / "checkpoint-00000004.building"
    building.mkdir()
    (building / "unfinished").write_text("keep evidence")
    (root / "best_checkpoints.json").write_text(json.dumps({
        "best_outcome": {"checkpoint": "checkpoint-00000003", "step": 3, "value": 0.3},
        "best_reward": {"checkpoint": "checkpoint-00000003", "step": 3, "value": 0.3}}))
    # Sampling has already committed six candidates although only two groups
    # have been updated. Remaining groups exist in the checkpoint's buffer.
    return {"step": 2, "dapo_state": {"sampler": {"candidate_index": 6},
                                      "active": {"consumed_groups": 2, "groups": [[], [], []]}}}


class DAPORecoveryTests(unittest.TestCase):
    def test_archives_future_artifacts_but_retains_committed_sampling_buffer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = populate(root)
            original_state = deepcopy(state)
            logs = {name: (root / name).read_bytes() for name in (
                "train_metrics.jsonl", "sampling_metrics.jsonl", "candidate_metrics.jsonl", "best_checkpoints.json")}
            result = reconcile_run(root, state)
            archive = Path(result["archive"])
            self.assertTrue(archive.is_dir())
            self.assertEqual(state, original_state)
            self.assertEqual([row["step"] for row in map(json.loads, (root / "train_metrics.jsonl").read_text().splitlines())], [1, 2])
            self.assertEqual([row["candidate_index"] for row in map(json.loads, (root / "sampling_metrics.jsonl").read_text().splitlines())], [3, 6])
            self.assertEqual([row["candidate_id"] for row in map(json.loads, (root / "candidate_metrics.jsonl").read_text().splitlines())], list(range(6)))
            for name, content in logs.items():
                self.assertEqual((archive / name).read_bytes(), content)
            for directory, prefix, suffix, kept, moved in (
                ("groups", "group-", ".json.gz", [0, 1], [2]),
                ("rollouts", "group-", ".json", [0, 1], [2]),
                ("candidate_rollouts", "candidate-", ".json", range(6), [6, 7, 8]),
            ):
                for i in kept:
                    self.assertTrue((root / directory / f"{prefix}{i:08d}{suffix}").exists())
                for i in moved:
                    relative = Path(directory) / f"{prefix}{i:08d}{suffix}"
                    self.assertFalse((root / relative).exists())
                    self.assertEqual((archive / relative).read_bytes(), f"original-{directory}-{i}".encode())
            self.assertTrue((root / "validation/step-00000002").is_dir())
            self.assertFalse((root / "validation/step-00000003").exists())
            self.assertTrue((archive / "validation/step-00000003/metrics.json").is_file())
            self.assertEqual((root / "checkpoint-00000002/trainer.pt").read_bytes(), b"original-torch-state-2")
            self.assertFalse((root / "checkpoint-00000003").exists())
            self.assertEqual((archive / "checkpoint-00000003/trainer.pt").read_bytes(), b"original-torch-state-3")
            self.assertFalse((root / "checkpoint-00000004.building").exists())
            self.assertEqual((archive / "checkpoint-00000004.building/unfinished").read_text(), "keep evidence")
            best = json.loads((root / "best_checkpoints.json").read_text())
            self.assertEqual(best["best_outcome"]["checkpoint"], "checkpoint-00000002")
            self.assertEqual(best["best_reward"]["checkpoint"], "checkpoint-00000002")
            self.assertEqual(json.loads((archive / "recovery.json").read_text()), result)

    def test_repeated_recovery_is_idempotent_and_middle_corruption_fails_before_mutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = populate(root)
            first = reconcile_run(root, state)
            retained = (root / "train_metrics.jsonl").read_bytes()
            second = reconcile_run(root, state)
            self.assertIsNone(second["archive"])
            self.assertEqual(second["archived_artifacts"], [])
            self.assertEqual((root / "train_metrics.jsonl").read_bytes(), retained)
            self.assertEqual(len(list((root / "resume_orphans").iterdir())), 1)
            self.assertTrue(Path(first["archive"]).exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = populate(root)
            damaged = b'{"candidate_id":0}\nBROKEN\n{"candidate_id":1}\n'
            (root / "candidate_metrics.jsonl").write_bytes(damaged)
            original_train = (root / "train_metrics.jsonl").read_bytes()
            with self.assertRaisesRegex(ValueError, "before the tail"):
                reconcile_run(root, state)
            self.assertFalse((root / "resume_orphans").exists())
            self.assertEqual((root / "candidate_metrics.jsonl").read_bytes(), damaged)
            self.assertEqual((root / "train_metrics.jsonl").read_bytes(), original_train)
            self.assertTrue((root / "checkpoint-00000003").exists())


if __name__ == "__main__":
    unittest.main()
