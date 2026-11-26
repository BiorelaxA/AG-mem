"""Mixed ORM shares one Task/Judge GPU and preserves isolated resumable saves."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agmem import two_gpu_train as launch
from agmem.core import Example
from agmem.judges import load_judge_templates


class Process:
    next_pid = 6000

    def __init__(self, kind, result):
        Process.next_pid += 1
        self.pid, self.kind, self.returncode = Process.next_pid, kind, result

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


class TwoGpuTrainTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cfg = {
            'data': {'directory_protocol': 'hierarchical_v1', 'sources': ['locomo', 'split_longmemeval_s'],
                     'splits': ['train','validation'], 'prepared_root': str(self.root/'prepared')},
            'models': {'reader_path': str(self.root/'models/reader'), 'task_path': str(self.root/'models/task')},
            'training': {'output_dir': str(self.root/'train'), 'algorithm': 'dapo',
                         'process_reward': {'enabled': False}, 'keep_last_checkpoints': 2},
            'reward': {'formula': 'f1_cost_v1', 'likelihood': {'mode': 'off'}},
            'reader': {'device': 'cuda:0'}, 'service': {'model': 'agmem-27b'},
            'task_agent': {'max_context_tokens': 65536},
        }

    def args(self, **changes):
        result = launch.argument_parser().parse_args([])
        for k,v in changes.items(): setattr(result,k,v)
        return result

    def workflow(self, *, check_only=False, storage=True, train_status=0,
                 probe_status=0, startup_error=None, report_error=None,
                 cleanup_error=False, server_dead=False, args=None, environ=None):
        calls, stopped = [], []
        report = {'storage': {'storage_ready':storage, 'free_bytes':7*launch.GIB,
                             'minimum_required_bytes':80*launch.GIB},
                  'inputs': {'total': {'train':1334,'validation':150}}}

        def popen(command, **kwargs):
            kind = ('probe' if '-c' in command else 'server' if 'vllm.entrypoints.openai.api_server' in command else 'train')
            status = probe_status if kind=='probe' else train_status if kind=='train' else 1 if server_dead else None
            p=Process(kind,status);calls.append((kind,command,kwargs,p));return p

        def stop(p):
            stopped.append(p.kind)
            if cleanup_error and p.kind=='server':raise RuntimeError('owned cleanup failed')

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'4,5',
                'AGMEM_READER_MODEL_PARALLEL_SIZE':'2', **(environ or {})}, clear=True))
            stack.enter_context(patch.object(launch,'training_config',return_value=deepcopy(self.cfg)))
            precheck=stack.enter_context(patch.object(launch,'preflight_report',
                           side_effect=report_error,return_value=report))
            stack.enter_context(patch.object(launch.subprocess,'Popen',side_effect=popen))
            stack.enter_context(patch.object(launch,'stop_group',side_effect=stop))
            stack.enter_context(patch.object(launch,'wait_ready',side_effect=startup_error))
            stack.enter_context(patch.object(launch,'free_port',return_value=49999))
            stack.enter_context(patch.object(launch,'latest_checkpoint',return_value=self.root/'train/checkpoint-00000004'))
            stack.enter_context(patch.object(launch.signal,'signal'))
            stack.enter_context(patch.object(launch.time,'sleep'))
            stack.enter_context(redirect_stdout(StringIO()));stack.enter_context(redirect_stderr(StringIO()))
            try: result=launch.launch(args or self.args(check_only=check_only))
            except BaseException as error:result=error
        return result,calls,stopped,precheck

    def test_two_gpus_without_slurm_and_reader_model_parallel_one(self):
        result,calls,stopped,_=self.workflow()
        self.assertIsInstance(result,Path)
        self.assertEqual([x[0] for x in calls], ['probe','server','train'])
        self.assertEqual([x[2]['env']['CUDA_VISIBLE_DEVICES'] for x in calls], ['4,5','5','4'])
        self.assertEqual(calls[2][2]['env']['AGMEM_READER_MODEL_PARALLEL_SIZE'],'1')
        self.assertNotIn('AGMEM_READER_MODEL_PARALLEL_SIZE',calls[1][2]['env'])
        self.assertEqual(calls[2][2]['env']['AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS'],'8192')
        self.assertTrue(all(x[2]['start_new_session'] for x in calls))
        self.assertEqual(stopped,['probe','train','server'])
        self.assertFalse(any('agmem.jobs' in x[1] or 'tests.test_model_parallel' in x[1] for x in calls))
        self.assertEqual(json.loads((result/'gpu_roles.json').read_text()),{'reader':'4','task':'5'})

    def test_user_offload_threshold_is_preserved(self):
        _,calls,_,_=self.workflow(environ={'AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS':'4096'})
        self.assertEqual(calls[2][2]['env']['AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS'],'4096')

    def test_judge_and_task_share_one_server_and_runtime_endpoint(self):
        self.cfg['reward'].update(formula='judge_f1_cost_v1', judge_f1_alpha=.8,
                                 judge={'protocols': {'locomo': 'locomo_refined'}})
        result,calls,_,_=self.workflow()
        self.assertIsInstance(result,Path)
        self.assertEqual([row[0] for row in calls],['probe','server','train'])
        import yaml
        runtime=yaml.safe_load((result/'train_config.yaml').read_text())
        self.assertEqual(runtime['service']['request_base_url'],'http://127.0.0.1:49999/v1')
        self.assertEqual(runtime['service']['model'],'agmem-27b')
        self.assertEqual(runtime['reward']['formula'],'judge_f1_cost_v1')
        self.assertNotIn('service',runtime['reward']['judge'])
        self.assertEqual(calls[1][2]['env']['CUDA_VISIBLE_DEVICES'],'5')

    def test_check_only_reports_insufficient_storage_without_starting_processes(self):
        result,calls,stopped,_=self.workflow(check_only=True,storage=False)
        self.assertFalse(result['storage']['storage_ready'])
        self.assertEqual(calls,[]);self.assertEqual(stopped,[])
        self.assertFalse((self.root/'jobs').exists())

    def test_insufficient_storage_stops_before_gpu_or_job_creation(self):
        result,calls,_,_=self.workflow(storage=False)
        self.assertIsInstance(result,RuntimeError)
        self.assertIn('Insufficient training output space',str(result))
        self.assertFalse(calls or (self.root/'jobs').exists())

    def test_user_disabled_storage_reserve_does_not_block_low_space(self):
        self.cfg['training']['enforce_storage_reserve']=False
        result,calls,_,_=self.workflow(storage=False)
        self.assertIsInstance(result,Path)
        self.assertEqual([r[0] for r in calls],['probe','server','train'])
        report=json.loads((result/'preflight.json').read_text())
        self.assertFalse(report['storage']['storage_ready'])
        self.assertFalse(report['storage']['enforced'])
        self.assertTrue(report['ready_to_launch'])
        import yaml
        runtime=yaml.safe_load((result/'train_config.yaml').read_text())
        self.assertNotIn('enforce_storage_reserve',runtime['training'])

    def test_disabled_storage_policy_is_explicit_in_check_only(self):
        self.cfg['training']['enforce_storage_reserve']=False
        result,calls,_,_=self.workflow(check_only=True,storage=False)
        self.assertEqual(calls,[])
        self.assertTrue(result['ready_to_launch'])
        self.assertFalse(result['storage']['enforced'])

    def test_invalid_storage_enforcement_type_is_rejected(self):
        self.cfg['training']['enforce_storage_reserve']='false'
        result,calls,_,_=self.workflow(storage=False)
        self.assertIsInstance(result,ValueError)
        self.assertEqual(calls,[])

    def test_invalid_inputs_stop_before_deployment(self):
        error=ValueError('stale hierarchy')
        result,calls,_,_=self.workflow(report_error=error)
        self.assertIs(result,error);self.assertFalse(calls)

    def test_cuda_failure_starts_no_service(self):
        result,calls,stopped,_=self.workflow(probe_status=1)
        self.assertIsInstance(result,RuntimeError)
        self.assertEqual([x[0] for x in calls],['probe']);self.assertEqual(stopped,['probe'])

    def test_service_startup_failure_cleans_owned_process(self):
        error=RuntimeError('service startup')
        result,calls,stopped,_=self.workflow(startup_error=error)
        self.assertIs(result,error)
        self.assertEqual([x[0] for x in calls],['probe','server']);self.assertEqual(stopped,['probe','server'])

    def test_training_failure_preserved_when_cleanup_also_fails(self):
        result,_,stopped,_=self.workflow(train_status=7,cleanup_error=True)
        self.assertIn('Reader training failed (7)',str(result))
        self.assertEqual(stopped,['probe','train','server'])

    def test_cleanup_error_after_success_is_reported(self):
        result,_,stopped,_=self.workflow(cleanup_error=True)
        self.assertIn('owned process group',str(result));self.assertEqual(stopped,['probe','train','server'])

    def test_service_death_stops_active_trainer(self):
        result,_,stopped,_=self.workflow(train_status=None,server_dead=True)
        self.assertIn('Task service exited',str(result));self.assertEqual(stopped,['probe','train','server'])

    def test_resume_and_update_limit_pass_to_train(self):
        _,calls,_,_=self.workflow(args=self.args(resume=True,max_updates=3))
        command=calls[-1][1]
        self.assertEqual(command[command.index('--max-groups')+1],'3')
        self.assertEqual(command[command.index('--resume')+1],str(self.root/'train/checkpoint-00000004'))

    def test_two_reader_gpus_and_one_shared_task_judge_gpu(self):
        self.cfg['reader']['model_parallel_size'] = 2
        result, calls, _, _ = self.workflow(environ={'CUDA_VISIBLE_DEVICES': '3,5,7'})
        self.assertIsInstance(result, Path)
        self.assertEqual([call[2]['env']['CUDA_VISIBLE_DEVICES'] for call in calls], ['3,5,7', '7', '3,5'])
        self.assertEqual(calls[2][2]['env']['AGMEM_READER_MODEL_PARALLEL_SIZE'], '2')
        self.assertNotIn('AGMEM_READER_MODEL_PARALLEL_SIZE', calls[1][2]['env'])
        self.assertIn('torch.cuda.device_count()==3', calls[0][1][-1])

    def test_exactly_two_distinct_devices(self):
        self.assertEqual(launch.gpu_roles('4,5'),{'reader':'4','task':'5'})
        for value in ('','4','4,4','4,5,6','4,','-1,5'):
            with self.subTest(value=value),self.assertRaises(ValueError):launch.gpu_roles(value)

    def test_runtime_output_override_and_model_paths(self):
        with patch.object(launch,'load_config',return_value=deepcopy(self.cfg)):
            config=launch.training_config('fixture',str(self.root/'other/train'))
        self.assertEqual(config['training']['output_dir'],str(self.root/'other/train'))
        self.assertEqual(config['models']['reader_path'],str(self.root/'models/reader'))

    def test_remote_task_override_is_rejected(self):
        config=deepcopy(self.cfg);config['task_agent']['service']={'model':'remote'}
        with patch.object(launch,'load_config',return_value=config),self.assertRaisesRegex(ValueError,'local 27B'):
            launch.training_config('fixture')

    def test_remote_judge_override_is_rejected(self):
        config=deepcopy(self.cfg)
        config['reward'].update(formula='judge_f1_cost_v1',judge={'service': {'model':'remote'}})
        with patch.object(launch,'load_config',return_value=config),self.assertRaisesRegex(ValueError,'Judge must share'):
            launch.training_config('fixture')

    def test_current_config_uses_relaxed_judge_prompts_and_fresh_training_output(self):
        config=launch.training_config(launch.DEFAULT_CONFIG)
        self.assertEqual(config['reward']['formula'],'judge_f1_cost_v1')
        self.assertEqual(config['reward']['judge_f1_alpha'],.8)
        self.assertEqual(config['reward']['format_penalty'],.1)
        self.assertEqual(config['reward']['visible_navigation_penalty'],.1)
        self.assertEqual(config['retrieval']['max_open_directories'],1)
        self.assertEqual(config['retrieval']['max_ids_per_call'],1)
        self.assertEqual(config['reward']['read_call_cost'],.05)
        self.assertEqual(config['training']['filter_metric'],'judge_nonzero_advantage')
        self.assertEqual(config['training']['visible_navigation_negative_advantage'],
                         {'enabled': True, 'magnitude': 1.0, 'include_repeated_segments': True})
        self.assertEqual(config['training']['checkpoint_retention'],'best_and_latest')
        self.assertEqual(config['training']['validation_primary_metric'],'outcome_f1')
        self.assertFalse(config['training']['enforce_storage_reserve'])
        self.assertTrue(config['training']['output_dir'].endswith('/locomo_lmes_context3_open1_repeatneg/train'))
        self.assertEqual(config['jobs']['experiment_name'],'locomo_lmes_context3_open1_repeatneg')
        self.assertTrue(config['data']['prepared_root'].endswith('/locomo_lmes_context3_open1_nav06/prepared'))
        self.assertEqual(config['data']['source_specs'][0]['directory_root'],
                         'runs/parent_child_directories/qwen27b_locomo_topic_event_context3_v1')
        self.assertIsNone(config['reward']['judge']['template_file'])
        templates,_,_=load_judge_templates(config['reward']['judge'])
        self.assertIn('TIME (semantic equivalence',templates['locomo_refined'])
        self.assertIn('Ignore extra non-contradictory items',templates['locomo_refined'])
        self.assertEqual(config['reward']['judge']['protocols']['split_longmemeval_s'],'longmemeval_official')
        # Historical pure-F1 recipe remains explicitly selectable.
        pure=launch.training_config('configs/train_dapo_hierarchical_f1.yaml')
        self.assertEqual(pure['reward']['formula'],'f1_cost_v1')

    def test_preflight_validates_judge_routing_and_hashes_without_requests(self):
        config=launch.training_config(launch.DEFAULT_CONFIG)
        values={split: [
            Example(f'locomo/{split}/1','locomo',split,'m1','Where?',gold_answers=['Paris']),
            Example(f'split_longmemeval_s/{split}/1','split_longmemeval_s',split,'m2','When?',
                    gold_answers=['Unknown'],judge_metadata={'question_id':'fixture_abs',
                                                             'question_type':'single-session-user'}),
        ] for split in ('train','validation')}
        store=SimpleNamespace(get_metadata=lambda _: 'frozen')
        with patch.object(launch,'open_prepared',return_value=store), \
             patch.object(launch,'validate_directories'), \
             patch.object(launch,'task_examples',side_effect=lambda c,s,split: iter(values[split])), \
             patch('agmem.policy.model_fingerprint',return_value={'fingerprint':'fixture-revision'}), \
             patch('agmem.services.ChatService.complete',side_effect=AssertionError('No requests during preflight')):
            result=launch.check_frozen_inputs(config)
            self.assertEqual(result['total'],{'train':2,'validation':2})
            self.assertEqual(result['judge']['model']['model'],'agmem-27b')
            self.assertEqual(result['judge']['model']['model_revision'],'fixture-revision')
            _,_,hashes=load_judge_templates(config['reward']['judge'])
            self.assertEqual(result['judge']['template_sha256'],hashes)
            values['validation'][1].judge_metadata={}
            with self.assertRaisesRegex(ValueError,'original question_id'):
                launch.check_frozen_inputs(config)

    def model(self):
        p=self.root/'model';p.mkdir()
        for name in ('config.json','tokenizer.json','tokenizer_config.json'):(p/name).write_text('{}')
        header=json.dumps({'weight':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}}).encode()
        (p/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'1234')
        return p

    def test_model_size_header_checks_truncated_weights_without_loading_torch(self):
        model=self.model();report=launch.model_files(model)
        self.assertEqual(report['weight_files'],1)
        p=model/'model.safetensors';p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError,'Incomplete or malformed'):launch.model_files(model)

    def test_missing_index_shard_is_rejected(self):
        model=self.model()
        (model/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'missing.safetensors'}}))
        with self.assertRaisesRegex(ValueError,'Missing model shard'):launch.model_files(model)

    def test_storage_includes_optimizer_state_and_uses_override_filesystem_parent(self):
        cfg=deepcopy(self.cfg);cfg['training']['output_dir']=str(self.root/'new-path/run')
        with patch.object(launch.shutil,'disk_usage',return_value=SimpleNamespace(free=7*launch.GIB)):
            report=launch.storage_report(cfg,9*launch.GIB)
        self.assertFalse(report['storage_ready'])
        self.assertEqual(report['filesystem_checked_at'],str(self.root))
        self.assertEqual(report['minimum_required_bytes'],80*launch.GIB)
        self.assertEqual(report['estimated_full_checkpoint_bytes'],63*launch.GIB)
        self.assertGreater(report['recommended_free_bytes'],300*launch.GIB)
        self.assertFalse(Path(cfg['training']['output_dir']).exists())


if __name__=='__main__':unittest.main()
