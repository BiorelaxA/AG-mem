"""Reader ID execution, reward accounting, and unchanged sampled actions."""
from copy import deepcopy
from types import SimpleNamespace
import json
import unittest

from tests import test_core as fixtures

ScriptedPolicy = fixtures.ScriptedPolicy
from agmem.core import CapacityError
from agmem.env import MemoryEnvironment, resolve_ids
from agmem.pipeline import scored_rollout, metrics
from agmem.traces import compact_group


class IDProtocolTests(unittest.TestCase):
    setUp = fixtures.CoreTests.setUp
    tearDown = fixtures.CoreTests.tearDown

    def run_actions(self, actions, *, correct=False, capacity=None, max_calls=8):
        self.env = MemoryEnvironment(self.store, self.builder.version, max_ids=4,
                                     max_calls=max_calls, capacity=capacity, id_protocol='repair_v2')
        self.policy = ScriptedPolicy(actions)
        self.seen = []
        def answer(example, evidence, *, metadata=None):
            self.seen.append(deepcopy(evidence))
            return ('SECRET_GOLD' if correct else 'wrong'), 'hash'
        return scored_rollout({'reward': {'formula': 'outcome_gated_v2', 'read_call_cost': .01,
                                          'format_penalty': .1}}, self.env, self.policy,
                              SimpleNamespace(answer=answer), self.example, 42)

    @staticmethod
    def read(ids):
        return {'ids': ids, 'stop': False}

    stop = {'ids': [], 'stop': True}

    def test_exact_repair_preserves_sampled_action_and_rewards_once(self):
        key = self.directory[2]['id']
        alias = key.replace('r_00002_', 'r_02_')
        action = self.read([alias])
        row = self.run_actions([action, self.stop], correct=True)
        self.assertEqual(row.decisions[0].action, action)
        self.assertEqual(json.loads(row.decisions[0].completion.text), action)
        self.assertEqual(self.seen[0][0]['id'], key)
        self.assertEqual(row.decisions[0].tool_result['id_repairs'],
                         [{'requested_id': alias, 'resolved_id': key}])
        self.assertAlmostEqual(row.reward, .89)
        self.assertEqual(row.decisions[0].format_violations, ['repaired_id'])
        report = metrics([row])['pooled']
        self.assertEqual(report['id_repair_rate'], 1)
        compact = compact_group({'group_id': 'test', 'directory_version': 'v', 'rollouts': [row.to_dict()]})
        step = compact['rollouts'][0]['reader_steps'][0]
        self.assertEqual(step['returned_ids'], [key])
        self.assertNotIn('raw_text', json.dumps(compact))

    def test_repair_never_guesses_other_hash_number_or_ambiguous_directory(self):
        key = 'r_00017_abcdef012345'
        alias = 'r_0017_abcdef012345'
        for wrong in ['r_0018_abcdef012345', 'r_0017_abcdef012346', 'r_000017_abcdef012345',
                      'R_0017_abcdef012345', 'r_0017_ABCDEF012345', 'other']:
            self.assertEqual(resolve_ids([wrong], {key: {}}), ([], [], [wrong]))
        self.assertEqual(resolve_ids([alias], {}), ([], [], [alias]))
        # A malformed directory with two numeric aliases must not be guessed.
        self.assertEqual(resolve_ids([alias], {key: {}, 'r_017_abcdef012345': {}}), ([], [], [alias]))

    def test_unknown_id_keeps_valid_prefix_and_penalty(self):
        key = self.directory[0]['id']
        row = self.run_actions([self.read([key, 'not-in-this-history']), self.stop])
        self.assertEqual([x['id'] for x in row.evidence], [key])
        self.assertEqual(row.decisions[0].tool_result['invalid_ids'], ['not-in-this-history'])
        self.assertEqual(row.reward, -1.1)
        self.assertEqual(len(self.seen[0]), 1)

    def test_identical_invalid_retry_stops_and_still_calls_task_agent(self):
        row = self.run_actions([self.read(['unknown']), self.read(['unknown'])])
        self.assertEqual(row.termination_reason, 'repeated_invalid_action')
        self.assertEqual(row.n_read_calls, 2)
        self.assertEqual(self.seen, [[]])
        self.assertEqual(row.reward, -1.1)
        self.assertIn('repeated_invalid_action', row.decisions[-1].format_violations)
        self.assertEqual(metrics([row])['pooled']['repeated_invalid_action_rate'], 1)

    def test_repeated_invalid_set_with_new_evidence_can_continue(self):
        a, b = [x['id'] for x in self.directory[:2]]
        row = self.run_actions([self.read(['unknown', a]), self.read([b, 'unknown']),
                                self.read(['unknown', b])])
        self.assertEqual(row.n_read_calls, 3)
        self.assertEqual(row.termination_reason, 'repeated_invalid_action')
        self.assertEqual({x['id'] for x in self.seen[0]}, {a, b})
        self.assertEqual(row.format_reward, -.1)

    def test_different_error_or_valid_action_resets_retry(self):
        a = self.directory[0]['id']
        row = self.run_actions([self.read(['x']), self.read(['y']), self.read([a]),
                                self.read(['y']), self.stop])
        self.assertEqual(row.termination_reason, 'stop')
        self.assertEqual(row.n_read_calls, 4)

    def test_corrected_alias_duplicates_are_read_once_and_penalized(self):
        key = self.directory[2]['id']; alias = key.replace('00002', '02')
        row = self.run_actions([self.read([key, alias]), self.stop])
        self.assertEqual(len(row.evidence), 1)
        self.assertIn('duplicate_ids', row.decisions[0].format_violations)
        self.assertEqual(row.format_reward, -.1)

    def test_excess_suffix_not_executed_or_resolved(self):
        keys = [x['id'] for x in self.directory[:4]]
        row = self.run_actions([self.read(keys + ['unknown']), self.stop], correct=True)
        self.assertEqual(row.decisions[0].tool_result['invalid_ids'], [])
        self.assertEqual(row.decisions[0].format_violations, ['too_many_ids'])
        self.assertEqual(len(row.evidence), 4)
        self.assertAlmostEqual(row.reward, .89)

    def test_stop_and_valid_re_reads_are_not_penalized(self):
        key = self.directory[0]['id']
        row = self.run_actions([self.read([key]), self.read([key]),
                                {'ids': ['unknown'] * 8, 'stop': True}], correct=True)
        self.assertFalse(row.format_failure)
        self.assertEqual(row.n_read_calls, 2)
        self.assertAlmostEqual(row.reward, .98)

    def test_capacity_rejection_remains_atomic_and_not_format_failure(self):
        def reject(*args):
            raise CapacityError('fixture')
        key = self.directory[0]['id']
        row = self.run_actions([self.read([key]), self.stop], capacity=reject)
        self.assertEqual(row.evidence, [])
        self.assertFalse(row.format_failure)
        self.assertEqual(row.reward, -1)
        self.assertEqual(row.decisions[0].tool_result['error']['code'], 'context_capacity_exceeded')

    def test_final_nonexecuted_action_is_checked_but_stop_ignored(self):
        key = self.directory[0]['id']
        row = self.run_actions([self.read([key]), self.read(['unknown'])], max_calls=1)
        self.assertEqual(row.n_read_calls, 1)
        self.assertEqual(row.format_reward, -.1)
        self.assertIsNone(row.decisions[-1].tool_result)

    def test_json_empty_duplicate_penalties_are_not_accumulated(self):
        key = self.directory[0]['id']
        row = self.run_actions([self.read([]), self.read([key, key]), 'not json'])
        self.assertEqual(row.format_reward, -.1)
        self.assertEqual(row.termination_reason, 'protocol_failure')
        self.assertEqual(len(self.seen[0]), 1)

    def test_batch_uses_identical_protocol(self):
        from agmem.core import Completion
        class Batch:
            def complete_batch(self, messages, seeds, max_batch_tokens):
                return [Completion('{"ids":["unknown"],"stop":false}') for _ in seeds]
        env = MemoryEnvironment(self.store, self.builder.version, id_protocol='repair_v2')
        rows = env.run_batch(self.example, Batch(), [1, 2], 0)
        self.assertEqual([r.termination_reason for r in rows], ['repeated_invalid_action'] * 2)
        self.assertEqual([r.n_read_calls for r in rows], [2, 2])
