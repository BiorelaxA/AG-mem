"""Exercise shell stage handoff without models, GPUs or API calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_longmemeval_811_flat_qwen.sh"


class FlatBuildEvalScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "calls.jsonl"
        executable = self.root / "python"
        executable.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
with open(os.environ["AGMEM_TEST_CALLS"], "a") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
if "agmem.build_local_directories_gpu" in sys.argv:
    sys.exit(int(os.environ.get("AGMEM_TEST_BUILD_EXIT", "0")))
''')
        executable.chmod(0o755)
        self.env = {**os.environ, "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
                    "AGMEM_PROJECT_DIR": str(ROOT), "AGMEM_TEST_CALLS": str(self.log)}

    def invoke(self, *args, failure=0):
        result = subprocess.run(["bash", str(SCRIPT), *args], env={
            **self.env, "AGMEM_TEST_BUILD_EXIT": str(failure)},
            capture_output=True, text=True)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return result, calls

    def test_success_hands_off_to_local_eval_only_after_build_returns(self):
        result, calls = self.invoke("--workers", "96")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertIn("agmem.build_local_directories_gpu", calls[0])
        self.assertEqual(calls[0][-2:], ["--workers", "96"])
        self.assertIn("agmem.evaluate_gpu", calls[1])
        self.assertIn("configs/eval_longmemeval_811_orm_checkpoints.yaml", calls[1])
        self.assertEqual(calls[1][-3:], ["--task-agents", "local", "--no-parallel-directories"])

    def test_build_failure_or_interruption_never_starts_evaluation(self):
        result, calls = self.invoke(failure=130)
        self.assertEqual(result.returncode, 130)
        self.assertEqual(len(calls), 1)
        self.assertIn("agmem.build_local_directories_gpu", calls[0])

    def test_eval_only_never_starts_writers_and_forwards_selection(self):
        result, calls = self.invoke("--eval-only", "--checkpoints", "final")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIn("agmem.evaluate_gpu", calls[0])
        self.assertEqual(calls[0][-2:], ["--checkpoints", "final"])

    def test_dry_run_does_not_launch_evaluation(self):
        result, calls = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIn("--dry-run", calls[0])
        self.assertIn("not executed", result.stdout)

    def test_eval_only_dry_run_starts_no_process(self):
        result, calls = self.invoke("--eval-only", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_build_only_preserves_directory_only_workflow(self):
        result, calls = self.invoke("--build-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIn("agmem.build_local_directories_gpu", calls[0])

    def test_conflicting_modes_fail_before_work(self):
        result, calls = self.invoke("--eval-only", "--build-only")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])

    def test_remote_helper_requires_all_bundles_before_any_request(self):
        script = ROOT / "scripts/eval_longmemeval_811_gpt.sh"
        result = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                                env={**self.env, "AGMEM_PROJECT_DIR": str(self.root),
                                     "OPENAI_API_KEY": "test-placeholder", "OPENAI_BASE_URL": "https://example.invalid/v1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing exported Task requests", result.stderr)
        self.assertFalse(self.log.exists())

    def test_remote_helper_pairs_each_checkpoint_and_dataset_without_reader(self):
        local = self.root / "runs/evaluation/longmemeval_811_orm/results"
        for label in ("best", "final"):
            for source in ("split_longmemeval_s", "split_longmemeval_m"):
                bundle = local / label / source / "offline/requests.json.gz"
                bundle.parent.mkdir(parents=True)
                bundle.touch()
        remote = self.root / "remote-results"
        (remote / "judge_results").mkdir(parents=True)
        (remote / "judge_results/summary.csv").write_text("checkpoint,dataset\n")
        result = subprocess.run(["bash", str(ROOT / "scripts/eval_longmemeval_811_gpt.sh")],
                                capture_output=True, text=True, env={
                                    **self.env, "AGMEM_PROJECT_DIR": str(self.root),
                                    "AGMEM_EVAL_OUTPUT_ROOT": str(remote),
                                    "OPENAI_API_KEY": "test-placeholder", "OPENAI_BASE_URL": "https://example.invalid/v1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(len(calls), 8)
        expected = [(label, source) for label in ("best", "final")
                    for source in ("split_longmemeval_s", "split_longmemeval_m")]
        for index, (label, source) in enumerate(expected):
            answer, judge = calls[index * 2:index * 2 + 2]
            self.assertIn("agmem/remote_task_runner.py", answer)
            self.assertTrue(answer[answer.index("--requests") + 1].endswith(
                f"/{label}/{source}/offline/requests.json.gz"))
            self.assertIn("agmem.judge_gpt_responses", judge)
            self.assertEqual(judge[judge.index("--checkpoints") + 1], label)
            self.assertEqual(judge[judge.index("--datasets") + 1], source)


if __name__ == "__main__":
    unittest.main()
