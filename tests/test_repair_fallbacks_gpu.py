"""Fallback repair deploys only its selected Writer and preserves process ownership."""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agmem import repair_fallbacks_gpu as launcher


class Process:
    next_pid = 12000

    def __init__(self, kind, code):
        Process.next_pid += 1
        self.pid, self.kind, self.returncode = Process.next_pid, kind, code

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


class RepairFallbackGpuTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / 'qwen3.5-27b'
        self.model.mkdir()
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            (self.model / name).write_text('{}')
        (self.model / 'model.safetensors').write_bytes(b'fixture')

    def args(self, **changes):
        args = launcher.argument_parser().parse_args([])
        args.job_dir = str(self.root / 'job')
        for name, value in changes.items():
            setattr(args, name, value)
        return args

    def run_launch(self, args, *, preflight=0, worker=0, readiness_error=None,
                   plan_error=None, cleanup_error=False, server_dead=False):
        calls, stopped, order = [], [], []

        def start(command, **kwargs):
            kind = ('writer' if 'vllm.entrypoints.openai.api_server' in command else
                    'worker' if any(m in command for m in ('agmem.repair_topic_event_fallbacks',
                                                           'agmem.accept_fallback_length_exceptions',
                                                           'agmem.repair_pending_parents')) else 'preflight')
            code = (preflight if kind == 'preflight' else worker if kind == 'worker'
                    else 1 if server_dead else None)
            process = Process(kind, code)
            calls.append((kind, command, kwargs, process))
            order.append(kind)
            return process

        def cpu(command, **kwargs):
            order.append('cpu_plan' if '--dry-run' in command else 'cpu_worker')
            if plan_error:
                raise plan_error

        def stop(process):
            stopped.append(process)
            if cleanup_error and process.kind == 'writer':
                raise RuntimeError('cleanup failed')

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '5'}, clear=True))
            stack.enter_context(patch.object(launcher, 'load_recipe', return_value={
                'model_path': str(self.model), 'writer': {'context_tokens': 65536}}))
            run = stack.enter_context(patch.object(launcher.subprocess, 'run', side_effect=cpu))
            stack.enter_context(patch.object(launcher.subprocess, 'Popen', side_effect=start))
            stack.enter_context(patch.object(launcher, 'free_port', return_value=12345))
            ready = stack.enter_context(patch.object(launcher, 'wait_ready', side_effect=readiness_error))
            stack.enter_context(patch.object(launcher, 'stop_group', side_effect=stop))
            stack.enter_context(patch.object(launcher.signal, 'signal'))
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(redirect_stderr(StringIO()))
            try:
                result = launcher.launch(args)
            except BaseException as error:
                result = error
        return result, calls, stopped, order, run, ready

    def test_one_gpu_writer_no_embedding_and_cpu_plan_before_gpu(self):
        result, calls, stopped, order, run, ready = self.run_launch(self.args())
        self.assertIsInstance(result, Path)
        self.assertEqual(order, ['cpu_plan', 'preflight', 'writer', 'worker'])
        self.assertEqual([(kind, options['env']['CUDA_VISIBLE_DEVICES'])
                          for kind, _, options, _ in calls],
                         [('preflight', '5'), ('writer', '5'), ('worker', '')])
        self.assertTrue(all(options['start_new_session'] for _, _, options, _ in calls))
        command = calls[1][1]
        self.assertEqual(command[command.index('--tensor-parallel-size') + 1], '1')
        self.assertEqual(command[command.index('--max-num-seqs') + 1], '64')
        self.assertNotIn('--runner', command)
        self.assertFalse(any('embedding' in item for item in command))
        self.assertIn('torch.cuda.device_count()==1', calls[0][1][-1])
        self.assertEqual(calls[2][1][-2:], ['--base-url', 'http://127.0.0.1:12345/v1'])
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])
        self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertTrue(ready.called)
        deployment = json.loads((result / 'deployment.json').read_text())
        self.assertFalse(deployment['embedding_service'])
        self.assertEqual(deployment['gpu'], '5')

    def test_approved_length_flow_routes_both_preflight_and_worker_to_recovery(self):
        args=self.args(accept_length_exceptions=True,previous_repairs='/tmp/previous-overlay')
        result,calls,_,_,run,_=self.run_launch(args)
        self.assertIsInstance(result,Path)
        self.assertIn('agmem.accept_fallback_length_exceptions',run.call_args.args[0])
        worker=next(command for kind,command,_,_ in calls if kind=='worker')
        self.assertIn('agmem.accept_fallback_length_exceptions',worker)
        self.assertEqual(worker[worker.index('--previous-repairs')+1],'/tmp/previous-overlay')

    def test_pending_parent_route_is_writer_only(self):
        result,calls,_,_,run,_=self.run_launch(self.args(pending_parents=True))
        self.assertIsInstance(result,Path)
        self.assertIn('agmem.repair_pending_parents',run.call_args.args[0])
        worker=next(command for kind,command,_,_ in calls if kind=='worker')
        self.assertIn('agmem.repair_pending_parents',worker)

    def test_input_validation_failure_starts_no_gpu_or_job(self):
        error = subprocess.CalledProcessError(1, ['worker', '--dry-run'])
        result, calls, stopped, _, _, _ = self.run_launch(self.args(), plan_error=error)
        self.assertIs(result, error)
        self.assertFalse(calls or stopped or (self.root/'job').exists())

    def test_gpu_preflight_failure_starts_no_model(self):
        result, calls, stopped, _, _, _ = self.run_launch(self.args(), preflight=1)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn('cuda_preflight.log', str(result))
        self.assertEqual([row[0] for row in calls], ['preflight'])
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_startup_failure_cleans_only_owned_processes(self):
        result, calls, stopped, _, _, _ = self.run_launch(self.args(), readiness_error=RuntimeError('startup failed'))
        self.assertIn('startup failed', str(result))
        self.assertEqual([row[0] for row in calls], ['preflight', 'writer'])
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_interrupt_during_readiness_also_cleans_owned_writer(self):
        result, calls, stopped, _, _, _ = self.run_launch(self.args(), readiness_error=SystemExit(130))
        self.assertIsInstance(result, SystemExit)
        self.assertEqual(result.code, 130)
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_failed_repair_is_not_hidden_by_cleanup_error(self):
        result, calls, stopped, _, _, _ = self.run_launch(self.args(), worker=1, cleanup_error=True)
        self.assertIn('Fallback repair failed (1)', str(result))
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_success_cannot_hide_owned_service_cleanup_failure(self):
        result, calls, stopped, _, _, _ = self.run_launch(self.args(), cleanup_error=True)
        self.assertIn('GPU release is unconfirmed', str(result))
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_runtime_service_failure_cleans_worker_and_model(self):
        result, calls, stopped, _, _, _ = self.run_launch(self.args(), worker=None, server_dead=True)
        self.assertIn('Qwen Writer exited', str(result))
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_dry_run_and_external_service_use_no_gpu_process(self):
        for changes, expected in (({'dry_run': True}, ['cpu_plan']),
                                  ({'base_url': 'http://127.0.0.1:8000/v1'}, ['cpu_plan','cpu_worker'])):
            with self.subTest(changes=changes):
                result, calls, stopped, order, _, _ = self.run_launch(self.args(**changes))
                self.assertIsNone(result)
                self.assertEqual(order, expected)
                self.assertFalse(calls or stopped)
                self.assertFalse((self.root/'job').exists())

    def test_device_selection_uses_one_allocated_gpu(self):
        self.assertEqual(launcher.selected_gpu(None, '5'), '5')
        self.assertEqual(launcher.selected_gpu('5', '4,5'), '5')
        self.assertEqual(launcher.selected_gpu('5', ''), '5')
        for explicit, visible in ((None,''),(None,'4,5'),(None,'-1'),('6','4,5')):
            with self.subTest(explicit=explicit,visible=visible), self.assertRaises(ValueError):
                launcher.selected_gpu(explicit,visible)

    def test_worker_command_forwards_selection_and_never_embedding(self):
        args = self.args(datasets=['split_longmemeval_s'],splits=['validation'],
                         output_root='/tmp/repair fixture',limit_sessions=1)
        command = launcher.worker_command(args, dry_run=True)
        self.assertIn('agmem.repair_topic_event_fallbacks', command)
        self.assertIn('--source-root', command)
        self.assertIn('/tmp/repair fixture', command)
        self.assertEqual(command[-1], '--dry-run')
        self.assertNotIn('--embedding-base-url', command)

    def test_worker_capacity_fails_before_plan_or_gpu(self):
        with patch.object(launcher.subprocess,'run') as run, self.assertRaisesRegex(ValueError,'single Writer'):
            launcher.launch(self.args(workers=65))
        run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
