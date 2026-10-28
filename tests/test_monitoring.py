from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from agmem.cli import main
from agmem.monitoring import export_tensorboard, summary_writer_class, write_train_metrics, write_validation_metrics, write_sampling_metrics


class MonitoringTests(unittest.TestCase):
    def test_rebuild_restores_committed_sampling_axis_and_entropy_after_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = summary_writer_class()(log_dir=str(root / "tensorboard"))
            write_train_metrics(writer, {"step": 1, "entropy": 0.7})
            write_train_metrics(writer, {"step": 2, "entropy": 0.2})
            write_sampling_metrics(writer, {"step": 0, "candidate_index": 32, "accepted_rate": 0.5})
            write_sampling_metrics(writer, {"step": 1, "candidate_index": 64, "accepted_rate": 0.1})
            writer.close()
            # The checkpoint includes update 1 and the first entire sampled batch.
            (root / "train_metrics.jsonl").write_text('{"step":1,"entropy":0.7}\n')
            (root / "sampling_metrics.jsonl").write_text('{"step":0,"candidate_index":32,"accepted_rate":0.5}\n')
            export_tensorboard(root)
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertEqual([p.step for p in events.Scalars("train/entropy")], [1])
            self.assertEqual([p.step for p in events.Scalars("sampling/accepted_rate")], [32])
            self.assertAlmostEqual(events.Scalars("train/entropy")[0].value, 0.7, places=6)

    def validation_report(self):
        return {"pooled": {"reward": 0.5, "outcome_f1": 0.6},
                "source_macro": {"reward": 0.4}, "by_source": {"fixture": {"reward": 0.5}}}

    def test_real_event_files_contain_training_and_validation_curves(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = summary_writer_class()(log_dir=temporary)
            try:
                write_train_metrics(writer, {"step": 1, "reward": -0.02, "sample_id": "fixture/0", "zero_variance_group": True})
                write_train_metrics(writer, {"step": 2, "reward": 0.7, "protocol_failure_rate": 0.125, "zero_variance_group": False})
                write_validation_metrics(writer, self.validation_report(), 2)
            finally:
                writer.close()
            events = EventAccumulator(temporary).Reload()
            rewards = events.Scalars("train/reward")
            self.assertEqual([point.step for point in rewards], [1, 2])
            self.assertAlmostEqual(rewards[0].value, -0.02, places=6)
            self.assertAlmostEqual(rewards[1].value, 0.7, places=6)
            self.assertEqual([point.value for point in events.Scalars("train/zero_variance_group")], [1.0, 0.0])
            self.assertEqual(events.Scalars("validation/reward")[0].step, 2)
            self.assertAlmostEqual(events.Scalars("validation/source_macro/reward")[0].value, 0.4, places=6)
            self.assertEqual(events.Scalars("validation/by_source/fixture/reward")[0].value, 0.5)
            self.assertNotIn("train/sample_id", events.Tags()["scalars"])

    def test_resuming_purges_only_points_after_the_saved_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = summary_writer_class()(log_dir=temporary)
            write_train_metrics(writer, {"step": 1, "reward": 0.1})
            write_train_metrics(writer, {"step": 2, "reward": 0.2})
            writer.close()
            writer = summary_writer_class()(log_dir=temporary, purge_step=2)
            write_train_metrics(writer, {"step": 2, "reward": 0.9})
            writer.close()
            rewards = EventAccumulator(temporary).Reload().Scalars("train/reward")
            self.assertEqual([point.step for point in rewards], [1, 2])
            self.assertAlmostEqual(rewards[1].value, 0.9, places=6)

    def test_export_completed_run_through_cli_without_config_or_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = '\n'.join(json.dumps(row) for row in ({"step": 1, "reward": 0.0}, {"step": 2, "reward": 0.7})) + '\n'
            (root / "train_metrics.jsonl").write_text(original)
            validation = root / "validation/step-00000002"
            validation.mkdir(parents=True)
            (validation / "metrics.json").write_text(json.dumps(self.validation_report()))
            # A missing config confirms this command bypasses training/config/model setup.
            with patch("sys.argv", ["agmem", "--config", "/missing/config.yaml", "export-tensorboard", temporary]), redirect_stdout(StringIO()) as output:
                main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["train_steps"], 2)
            self.assertEqual(result["validation_points"], 1)
            export_tensorboard(root)  # Repeating an export must not duplicate old scalar points.
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertEqual([point.step for point in events.Scalars("train/reward")], [1, 2])
            self.assertEqual(len(events.Scalars("validation/reward")), 1)
            self.assertEqual((root / "train_metrics.jsonl").read_text(), original)


if __name__ == "__main__":
    unittest.main()
