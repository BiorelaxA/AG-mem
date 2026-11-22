"""Fresh tagged local/remote evaluations share checked Reader evidence."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from agmem.core import Example, Trajectory, atomic_json, digest
from agmem.evaluate_checkpoints import EpisodeJournal
from agmem.evaluation_offline import export_task_requests
from agmem.judge_gpt_responses import load_inputs
from agmem.prompts import task_input_contract, task_messages
from agmem.remote_task_runner import load_bundle, response_manifest


class TaggedPairTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.target = self.root / 'results/best/locomo'
        self.examples = [Example('locomo/test/000000', 'locomo', 'test', 'history', 'What color?', gold_answers=['blue'])]
        self.config = {'task_agent': {'prompt_version': 'history_answer_v1', 'max_output_tokens': 2048}}
        self.remote = {'model': 'gpt-5-mini', 'model_revision': 'gpt-5-mini', 'request_profile': 'openai_chat',
                       'send_seed': False, 'temperature': None, 'max_tokens_parameter': 'max_completion_tokens'}
        contract = task_input_contract('history_answer_v1')
        self.retrieval = EpisodeJournal(self.target / 'retrieval', {
            'sample_ids': [e.sample_id for e in self.examples], 'task_input_contract': contract,
            'examples_sha256': digest([asdict(e) for e in self.examples])})
        self.local = EpisodeJournal(self.target / 'answers/local', {
            'sample_ids': self.retrieval.sample_ids, 'retrieval_manifest_sha256': self.retrieval.signature,
            'task_input_contract': contract, 'task_max_output_tokens': 2048})
        self.row = Trajectory(self.examples[0].sample_id, 42, 'history', evidence=[{'id': 'r1', 'raw_text': 'Blue.'}])
        self.retrieval.write(self.row); self.retrieval.finish()
        self.row.prediction = 'blue'
        self.row.task_input_hash = digest(task_messages(self.examples[0], self.row.evidence, prompt_version='history_answer_v1'))
        self.local.write(self.row); self.local.finish()

    def export(self):
        return export_task_requests(self.config, self.examples, self.retrieval, self.target, self.remote, 'best', 'locomo')

    def test_same_prompt_pair_exports_identical_input_and_tagged_protocol(self):
        path = self.export()
        bundle = load_bundle(path)
        self.assertEqual(bundle['manifest']['protocol'], 'checkpoint_task_requests_v2')
        self.assertEqual(bundle['manifest']['paired_local_task_manifest_sha256'], self.local.signature)
        self.assertEqual(bundle['requests'][0]['task_input_hash'], self.row.task_input_hash)
        original = path.read_bytes()
        self.assertEqual(self.export(), path)
        self.assertEqual(path.read_bytes(), original)
        self.assertNotIn('gold_answers', json.dumps(bundle))

    def test_incomplete_local_or_different_actual_local_input_is_rejected(self):
        (self.local.root / 'complete.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.export()
        self.local.finish()
        self.row.task_input_hash = '0' * 64; self.local.write(self.row)
        with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
            self.export()

    def test_actual_prompt_change_still_requires_separate_output(self):
        other = self.root / 'old/best/locomo'
        old = EpisodeJournal(other / 'retrieval', {
            'sample_ids': self.retrieval.sample_ids, 'task_input_contract': task_input_contract(),
            'examples_sha256': digest([asdict(e) for e in self.examples])})
        old.write(self.row); old.finish()
        with self.assertRaisesRegex(ValueError, 'separate output'):
            export_task_requests(self.config, self.examples, old, other, self.remote, 'best', 'locomo')

    def test_future_gpt_judge_uses_explicit_prepared_path_and_paired_hash(self):
        bundle = load_bundle(self.export())
        manifest = response_manifest(bundle, 'https://provider.invalid/v1')
        responses = self.target / 'offline/responses'
        atomic_json(responses / 'manifest.json', manifest)
        atomic_json(responses / 'episodes/00000000.json', {
            'manifest_sha256': digest(manifest), 'sample_id': self.row.sample_id,
            'task_input_hash': self.row.task_input_hash,
            'completion': {'protocol': 'score_nonempty_truncated_v1', 'text': '<answer>blue</answer>',
                           'finish_reason': 'stop', 'truncated': False, 'usage': None, 'max_tokens': 2048}})
        atomic_json(responses / 'complete.json', {'manifest_sha256': digest(manifest), 'n': 1})
        prepared = self.root / 'flat/test/locomo'; prepared.mkdir(parents=True)
        (prepared / 'test_examples.jsonl').write_text(json.dumps(asdict(self.examples[0])) + '\n')
        suite = {'output_root': str(self.root / 'results'), 'prepared_root': str(self.root / 'unused'),
                 'task_prompt_version': 'history_answer_v1', 'required_prepared_sources': {'locomo': str(prepared)}}
        result = load_inputs(suite, 'best', 'locomo')
        self.assertEqual(result['n'], 1)
        self.assertEqual(result['finish_reasons'], {'stop': 1})
        (self.local.root / 'complete.json').unlink()
        with self.assertRaises(FileNotFoundError):
            load_inputs(suite, 'best', 'locomo')


if __name__ == '__main__':
    unittest.main()
