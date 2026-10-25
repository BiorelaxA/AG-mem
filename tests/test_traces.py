from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import gzip
import json
import tempfile
import unittest

from agmem.cli import main
from agmem.core import Completion, Decision, Trajectory
from agmem.traces import export_rollouts, save_training_group


class TraceTests(unittest.TestCase):
    def trajectory(self):
        row = Trajectory("fixture/train/0", 42, "memory0")
        row.decisions = [
            Decision(
                [{"role": "user", "content": "PRIVATE_QUERY_AND_HISTORY"}],
                Completion('```json\n{"ids":["s1"],"stop":false}\n```', [999], [888], [-0.25]),
                {"ids": ["s1"], "stop": False},
                {"records": [{"raw_text": "PRIVATE_SESSION_TEXT"}]},
            ),
            Decision([], Completion('{"ids":[],"stop":true}'), {"ids": [], "stop": True}),
            Decision([], Completion("malformed output")),
        ]
        row.evidence = [{"raw_text": "PRIVATE_SESSION_TEXT"}]
        row.prediction = "21 August, 2022"
        row.termination_reason = "protocol_failure"
        row.outcome, row.reward = 1.0, 0.89
        row.read_cost, row.format_failure, row.format_reward = 0.01, True, -0.1
        return row

    def test_training_save_writes_outputs_without_prompts_evidence_or_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self.trajectory()
            save_training_group(root, 0, "group0", "version0", [row])
            with gzip.open(root / "groups/group-00000000.json.gz", "rt") as stream:
                self.assertEqual(json.load(stream)["rollouts"], [row.to_dict()])
            compact_path = root / "rollouts/group-00000000.json"
            text = compact_path.read_text()
            compact = json.loads(text)
            saved = compact["rollouts"][0]
            self.assertEqual(saved["task_agent_output"], row.prediction)
            self.assertEqual(saved["termination_reason"], "protocol_failure")
            self.assertEqual(saved["format_reward"], -0.1)
            self.assertTrue(saved["format_failure"])
            self.assertEqual(saved["read_cost"], 0.01)
            for index, decision in enumerate(row.decisions):
                self.assertEqual(saved["reader_steps"][index], {
                    "step": index + 1, "reader_output": decision.completion.text, "action": decision.action})
            self.assertIsNone(saved["reader_steps"][-1]["action"])
            for forbidden in ("PRIVATE_", "raw_text", "messages", "prompt_ids", "token_ids", "old_logprobs", "tool_result"):
                self.assertNotIn(forbidden, text)

    def test_export_cli_backfills_groups_without_model_or_config_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full_path = root / "groups/group-00000000.json.gz"
            full_path.parent.mkdir()
            rows = [self.trajectory().to_dict() for _ in range(8)]
            with gzip.open(full_path, "wt") as stream:
                json.dump({"group_id": "group0", "directory_version": "version0", "rollouts": rows}, stream)
            original = full_path.read_bytes()
            with patch("sys.argv", ["agmem", "--config", "/missing/config.yaml", "export-rollouts", temporary]), redirect_stdout(StringIO()) as output:
                main()
            self.assertEqual(json.loads(output.getvalue()), {
                "groups": 1, "rollouts": 8, "output_dir": str(root / "rollouts")})
            compact_path = root / "rollouts/group-00000000.json"
            exported = compact_path.read_bytes()
            export_rollouts(root)
            self.assertEqual(compact_path.read_bytes(), exported)
            self.assertEqual(full_path.read_bytes(), original)

    def test_truncated_ids_are_auditable_without_session_text(self):
        from agmem.traces import compact_group
        row = self.trajectory()
        row.decisions[0].action = {'ids': ['s1', 's2', 'ignored'], 'stop': False}
        row.decisions[0].tool_result = {
            'ids_truncated': True, 'selected_ids': ['s1', 's2'],
            'records': [{'id': key, 'raw_text': 'PRIVATE_TEXT'} for key in ['s1', 's2']]}
        group = {'group_id': 'g', 'directory_version': 'v', 'rollouts': [row.to_dict()]}
        compact = compact_group(group)
        step = compact['rollouts'][0]['reader_steps'][0]
        self.assertEqual(step['action']['ids'], ['s1', 's2', 'ignored'])
        self.assertEqual(step['returned_ids'], ['s1', 's2'])
        self.assertTrue(step['ids_truncated'])
        self.assertNotIn('PRIVATE', json.dumps(compact))
        row.decisions[0].tool_result['records'] = []
        group['rollouts'] = [row.to_dict()]
        self.assertEqual(compact_group(group)['rollouts'][0]['reader_steps'][0]['returned_ids'], [])


if __name__ == "__main__":
    unittest.main()
