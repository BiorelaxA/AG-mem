"""Dedicated full LoCoMo rebuild routing without any model or GPU requests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_locomo_topic_event_context3_2gpu.sh"


class LocomoContext3ScriptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "calls.jsonl"
        executable = self.root / "python"
        executable.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
with Path(os.environ['TEST_CALLS']).open('a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'gpus': os.environ.get('CUDA_VISIBLE_DEVICES'),
                            'offline': os.environ.get('HF_HUB_OFFLINE')}) + '\\n')
''')
        executable.chmod(0o755)
        self.env = {**os.environ, "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
                    "AGMEM_PROJECT_DIR": str(ROOT), "TEST_CALLS": str(self.log)}
        for name in ("AGMEM_DIRECTORY_GPUS", "AGMEM_WRITER_WORKERS", "AGMEM_WRITER_MAX_NUM_SEQS"):
            self.env.pop(name, None)

    def invoke(self, *args):
        result = subprocess.run(["bash", str(SCRIPT), *args], env=self.env,
                                text=True, capture_output=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_default_rebuild_is_locomo_only_and_does_not_import_old_directories(self):
        self.env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        result, calls = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        command = calls[0]["args"]
        self.assertEqual(command[:3], ["-u", "-m", "agmem.build_adjacent_directories_gpu"])
        self.assertIn("--segment-context", command)
        self.assertNotIn("--topic-event", command)
        self.assertNotIn("--reuse-root", command)
        self.assertNotIn("--preserve-existing", command)
        self.assertEqual(command[command.index("--workers") + 1], "64")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "64")
        self.assertEqual(command[-6:], ["--datasets", "locomo", "--splits", "train", "validation", "test"])
        self.assertEqual(calls[0]["gpus"], "4,5")
        self.assertEqual(calls[0]["offline"], "1")

    def test_dry_run_smoke_and_other_filesystem_options_are_forwarded(self):
        result, calls = self.invoke("--dry-run", "--output-root", "/tmp/new directory",
                                    "--job-dir", "/tmp/new logs", "--limit-sessions", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[0]["args"][-7:], ["--dry-run", "--output-root", "/tmp/new directory",
                                                "--job-dir", "/tmp/new logs", "--limit-sessions", "2"])

    def test_explicit_gpu_and_concurrency_environment(self):
        self.env.update(AGMEM_DIRECTORY_GPUS="1,3", AGMEM_WRITER_WORKERS="16",
                        AGMEM_WRITER_MAX_NUM_SEQS="32")
        result, calls = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[0]["gpus"], "1,3")
        command = calls[0]["args"]
        self.assertEqual(command[command.index("--workers") + 1], "16")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "32")

    def test_help_does_not_run_python(self):
        result, calls = self.invoke("--help")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls, [])
        self.assertIn("three complete preceding segments", result.stdout)

    def test_recipe_has_new_identity_and_full_raw_segment_context(self):
        config = yaml.safe_load((ROOT / "configs/build_locomo_topic_event_context3.yaml").read_text())
        self.assertEqual(config["protocol"], "adjacent_topic_event_context3_v1")
        self.assertEqual(config["model_path"], "models/qwen3.5-27b")
        self.assertEqual(config["embedding"]["model_path"], "models/qwen3-embedding-4b")
        self.assertEqual(config["child_context_segments"], 3)
        self.assertEqual(config["writer"]["context_blocks"], 0)
        self.assertEqual(config["writer"]["parent_content_tokens"], 400)
        self.assertEqual(config["writer"]["child_content_tokens"], 300)
        self.assertEqual(config["writer"]["title_tokens"], 24)
        self.assertEqual((config["segmentation"]["high_threshold"],
                          config["segmentation"]["low_threshold"]), (0.72, 0.50))
        self.assertTrue(config["retry_fallback"])
        self.assertFalse(config.get("preserve_existing_directories", False))
        self.assertEqual(config["datasets"], [{"name": "locomo", "format": "locomo_parquet",
            "input_root": "data/processed/locomo_query9010_seed42_20261102",
            "splits": ["train", "validation", "test"], "modality_policy": "existing_captions"}])
        self.assertEqual(config["output_root"],
                         "runs/parent_child_directories/qwen27b_locomo_topic_event_context3_v1")


if __name__ == "__main__":
    unittest.main()
