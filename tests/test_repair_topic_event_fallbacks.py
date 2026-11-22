"""Placeholder repairs preserve source bytes, fixed child spans and good text."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agmem import repair_topic_event_fallbacks as repair
from agmem.core import InfrastructureError, canonical, digest
from agmem.hierarchical.binding import directory_version
from tests.test_adjacent_directory_pipeline import Pool
from tests import test_repair_adjacent_fallbacks as fixtures

LongParentPool = fixtures.LongParentPool


class PlaceholderRepairTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.RepairAdjacentFallbackTests(methodName='runTest')
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.initial_build()
        self.output = self.f.home/'repairs'
        self.config = deepcopy(self.f.new_config)
        self.config.update(output_root=str(self.output), preserve_existing_directories=False)
        self.identity = deepcopy(self.f.new_identity)

    def args(self, **overrides):
        args = self.f.args(splits=['test'], **overrides)
        args.source_root = str(self.f.output)
        return args

    def source_bytes(self):
        return {p.name:p.read_bytes() for p in self.f.output.iterdir() if p.is_file()}

    def invoke(self, pool=None, **overrides):
        pool = pool or LongParentPool()
        with patch.object(repair.topic, 'load_recipe', return_value=(self.config,self.identity)), \
                patch.object(repair, 'announce'):
            return repair.run(self.args(**overrides), tokenizer=self.f.tokenizer, pool_factory=pool.factory)

    def test_parent_only_repair_preserves_source_and_children_and_resumes(self):
        before = self.source_bytes()
        pool = LongParentPool()
        report = self.invoke(pool)
        self.assertEqual((report['selected'],report['repaired'],report['remaining']), (1,1,0))
        self.assertEqual([r[2] for r in pool.calls], ['adjacent_parent'])
        patches = [json.loads(line) for line in (self.output/'repair_patches.jsonl').read_text().splitlines()]
        self.assertEqual(len(patches),1)
        patch_row = patches[0]
        with self.f.db() as db:
            original = json.loads(db.execute("select directory from sessions where parent_id='p0002'").fetchone()[0])
        self.assertEqual(patch_row['before_directory_sha256'], digest(original))
        self.assertEqual(patch_row['directory']['children'], original['children'])
        self.assertEqual(patch_row['directory']['build']['status'], 'complete')
        self.assertEqual(before,self.source_bytes())
        resumed = self.invoke(Pool(forbidden=True))
        self.assertEqual(resumed['repaired'],1)
        self.assertEqual(before,self.source_bytes())
        self.assertTrue((self.output/'complete.json').exists())

    def test_failed_attempt_never_replaces_original_and_can_resume(self):
        before = self.source_bytes()
        with self.assertRaisesRegex(InfrastructureError,'remain incomplete'):
            self.invoke(Pool(fail_parent=True))
        self.assertEqual(before,self.source_bytes())
        self.assertEqual((self.output/'repair_patches.jsonl').read_text(), '')
        self.assertFalse((self.output/'complete.json').exists())
        self.assertEqual(self.invoke()['status'],'ready')
        self.assertEqual(before,self.source_bytes())

    def test_dryrun_is_readonly_and_has_no_services(self):
        before = self.source_bytes()
        report = self.invoke(Pool(forbidden=True),dry_run=True)
        self.assertEqual(report['selected'],1)
        self.assertFalse(report['requires_embedding'])
        self.assertFalse(self.output.exists())
        self.assertEqual(before,self.source_bytes())

    def test_source_drift_is_rejected_before_repair_on_resume(self):
        self.invoke()
        with self.f.db() as db:
            row = db.execute("select bank_id,directory from sessions where parent_id='p0002'").fetchone()
            directory = json.loads(row[1]);directory['title']='Changed placeholder'
            directory['directory_version']=directory_version(directory)
            db.execute('update sessions set directory=? where bank_id=? and parent_id=?',
                       (canonical(directory),row[0],'p0002'))
        with self.assertRaisesRegex(ValueError,'selection changed'):
            self.invoke(Pool(forbidden=True))

    def test_changed_repair_identity_is_rejected_in_dryrun(self):
        self.invoke()
        self.identity['writer']['model_revision']='different'
        with self.assertRaisesRegex(ValueError,'identity changed'):
            self.invoke(Pool(forbidden=True),dry_run=True)

    def test_good_child_change_is_not_accepted(self):
        self.invoke()
        patch_row=json.loads((self.output/'repair_patches.jsonl').read_text())
        with self.f.db() as db:
            session,directory=db.execute("select snapshot,directory from sessions where parent_id='p0002'").fetchone()
        replacement=patch_row['directory']
        replacement['children'][0]['content']='Changed good summary'
        replacement['directory_version']=directory_version(replacement)
        with self.assertRaisesRegex(ValueError,'valid child'):
            repair.validate_replacement(json.loads(directory),replacement,json.loads(session))

    def test_placeholder_child_is_regenerated_without_boundary_review(self):
        # Model this second placeholder exactly as generated by the legacy Writer.
        with self.f.db() as db:
            bid,encoded=db.execute("select bank_id,directory from sessions where parent_id='p0002'").fetchone()
            directory=json.loads(encoded)
            directory['children'][0]['content']=repair.PLACEHOLDER
            directory['children'][0]['title']='Original records 1-1'
            directory['children'][0]['entities']=[]
            directory['entities']=[]
            directory['directory_version']=directory_version(directory)
            db.execute('update sessions set directory=? where bank_id=? and parent_id=?',
                       (canonical(directory),bid,'p0002'))
        before=self.source_bytes();pool=LongParentPool()
        report=self.invoke(pool)
        self.assertEqual(report['repaired'],1)
        self.assertEqual([r[2] for r in pool.calls], ['adjacent_child','adjacent_parent'])
        self.assertEqual(before,self.source_bytes())

    def test_non_placeholder_fallback_requires_different_repair(self):
        with self.f.db() as db:
            bid,encoded=db.execute("select bank_id,directory from sessions where parent_id='p0002'").fetchone()
            directory=json.loads(encoded);directory['content']='A normal parent summary.'
            directory['directory_version']=directory_version(directory)
            db.execute('update sessions set directory=? where bank_id=? and parent_id=?',
                       (canonical(directory),bid,'p0002'))
        with self.assertRaisesRegex(ValueError,'no recognized placeholder'):
            self.invoke(Pool(forbidden=True),dry_run=True)

    def test_apply_patch_binds_exact_source_and_returns_independent_copy(self):
        self.invoke()
        patch_row=json.loads((self.output/'repair_patches.jsonl').read_text())
        with self.f.db() as db:
            session,directory=db.execute("select snapshot,directory from sessions where parent_id='p0002'").fetchone()
        session,original=json.loads(session),json.loads(directory)
        value=repair.apply_repair_patch(original,patch_row,session,self.f.old_identity)
        self.assertEqual(value,patch_row['directory'])
        value['title']='caller-owned'
        self.assertNotEqual(value,patch_row['directory'])
        self.assertEqual(original['build']['status'],'fallback')
        for field in ('source_hash','before_directory_sha256','before_directory_version','after_directory_sha256'):
            changed=deepcopy(patch_row);changed[field]='bad'
            with self.subTest(field=field),self.assertRaises(ValueError):
                repair.apply_repair_patch(original,changed,session,self.f.old_identity)
        with self.assertRaisesRegex(ValueError,'another source construction'):
            repair.apply_repair_patch(original,patch_row,session,{'other':'identity'})
        with self.assertRaises(ValueError):
            repair.apply_repair_patch(patch_row['directory'],patch_row,session,self.f.old_identity)


if __name__=='__main__':
    unittest.main()
