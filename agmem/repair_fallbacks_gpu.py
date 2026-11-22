"""Serve one local Qwen Writer to repair fallback directory entries.

The repair worker owns source/overlay validation. This launcher verifies its
CPU plan before reserving one GPU and stops only its own process groups.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .build_adjacent_directories_gpu import load_recipe, writer_command
from .core import atomic_json
from .four_gpu import free_port, stop_group, wait_ready


DEFAULT_CONFIG = 'configs/repair_longmemeval_s_fallbacks.yaml'
DEFAULT_SOURCE = 'runs/parent_child_directories/qwen27b_adjacent_qwen4b_31535150_incremental_v2'


def worker_command(args, endpoint=None, *, dry_run=False):
    if getattr(args,'pending_parents',False) and getattr(args,'accept_length_exceptions',False):
        raise ValueError('Choose pending-parent completion or fallback length acceptance')
    module = ('agmem.repair_pending_parents' if getattr(args,'pending_parents',False)
              else 'agmem.accept_fallback_length_exceptions' if getattr(args,'accept_length_exceptions',False)
              else 'agmem.repair_topic_event_fallbacks')
    command = [sys.executable, '-u', '-m', module,
               '--config', args.config, '--source-root', args.source_root,
               '--workers', str(args.workers)]
    for key in ('datasets', 'splits'):
        values = getattr(args, key, None)
        if values:
            command.extend(['--' + key, *values])
    for key in ('output_root', 'limit_sessions'):
        value = getattr(args, key, None)
        if value is not None:
            command.extend(['--' + key.replace('_', '-'), str(value)])
    if endpoint:
        command.extend(['--base-url', endpoint])
    if getattr(args,'previous_repairs',None):
        if not getattr(args,'accept_length_exceptions',False):
            raise ValueError('--previous-repairs requires --accept-length-exceptions')
        command.extend(['--previous-repairs',args.previous_repairs])
    if dry_run:
        command.append('--dry-run')
    return command


def selected_gpu(explicit=None, visible=None):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '') if visible is None else visible
    value = explicit if explicit is not None else visible
    if not isinstance(value, str) or not value.strip() or ',' in value or value.strip() in {'-1', 'NoDevFiles'}:
        raise ValueError('Select one allocated GPU with --gpu or CUDA_VISIBLE_DEVICES (for example 5)')
    value = value.strip()
    if explicit is not None and visible.strip() and value not in {v.strip() for v in visible.split(',')}:
        raise ValueError('--gpu must be present in CUDA_VISIBLE_DEVICES')
    return value


def launch(args):
    for key in ('workers', 'max_num_seqs', 'max_num_batched_tokens'):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise ValueError(key + ' must be a positive integer')
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError('gpu_memory_utilization must be between zero and one')
    if args.workers > args.max_num_seqs and not args.base_url:
        raise ValueError('workers exceeds the single Writer max_num_seqs')
    if args.limit_sessions is not None and args.limit_sessions < 1:
        raise ValueError('limit_sessions must be positive')
    environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4'}
    environment.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    environment.pop('AGMEM_READER_MODEL_PARALLEL_SIZE', None)
    cpu_environment = {**environment, 'CUDA_VISIBLE_DEVICES': ''}
    # No Torch, CUDA, model process or remote request in the worker dry plan.
    subprocess.run(worker_command(args, dry_run=True), env=cpu_environment, check=True)
    if args.dry_run:
        return None
    if args.base_url:
        subprocess.run(worker_command(args, args.base_url), env=cpu_environment, check=True)
        return None
    gpu = selected_gpu(args.gpu)
    recipe = load_recipe(args.config)
    model = Path(recipe['model_path']).expanduser().resolve()
    required = ('config.json', 'tokenizer.json', 'tokenizer_config.json')
    if any(not (model / name).is_file() for name in required) or not any(model.glob('*.safetensors')):
        raise ValueError(f'Missing local Writer model files: {model}')
    if recipe.get('service', {}).get('model', 'agmem-27b') != 'agmem-27b':
        raise ValueError('Managed Writer requires service.model=agmem-27b')
    context_tokens = recipe.get('writer', {}).get('context_tokens', 65536)
    if type(context_tokens) is not int or context_tokens < 1:
        raise ValueError('writer.context_tokens must be a positive integer')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    job = (Path(args.job_dir) if args.job_dir else Path('runs/fallback_repair/jobs') /
           f'{stamp}-{os.getpid()}')
    job.mkdir(parents=True, exist_ok=False)
    print(f'Fallback repair logs: {job.resolve()}; Qwen Writer GPU: {gpu}; workers: {args.workers}', flush=True)
    processes, handlers = [], {}

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    with ExitStack() as cleanup:
        def start(name, command, devices):
            log = cleanup.enter_context((job / f'{name}.log').open('w'))
            process = subprocess.Popen(command, env={**environment, 'CUDA_VISIBLE_DEVICES': devices},
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            return process

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.signal(signum, interrupted)
            probe = (
                "import json, torch; import vllm._C\n"
                "assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Expected one CUDA GPU'\n"
                "free,total=torch.cuda.mem_get_info(0)\n"
                "print(json.dumps({'gpu':0,'name':torch.cuda.get_device_name(0),'free_gib':free/2**30,'total_gib':total/2**30}),flush=True)\n"
                "assert total >= 70*2**30, 'The Writer requires an 80-GB-class GPU'\n"
                f"assert free/total >= {args.gpu_memory_utilization}, 'Selected GPU is occupied'\n"
                "(torch.ones((32,32),device='cuda:0') @ torch.ones((32,32),device='cuda:0')).sum().item()\n")
            preflight = start('cuda_preflight', [sys.executable, '-c', probe], gpu)
            if preflight.wait():
                raise RuntimeError(f'CUDA/vLLM preflight failed; inspect {job / "cuda_preflight.log"}')
            port = free_port()
            endpoint = f'http://127.0.0.1:{port}/v1'
            atomic_json(job / 'deployment.json', {
                'model_path': str(model), 'gpu': gpu, 'tensor_parallel_size': 1,
                'workers': args.workers, 'max_num_seqs': args.max_num_seqs,
                'base_url': endpoint, 'source_root': str(Path(args.source_root).resolve()),
                'embedding_service': False,
            })
            server = start('writer', writer_command(args, str(model), port, context_tokens), gpu)
            wait_ready(server, port, job / 'writer.log')
            worker = start('repair', worker_command(args, endpoint), '')
            print(f'Writer ready; repair progress: {job.resolve() / "repair.log"}', flush=True)
            while True:
                status = worker.poll()
                if status is not None:
                    if status:
                        raise RuntimeError(f'Fallback repair failed ({status}); inspect {job / "repair.log"}; rerun to resume')
                    print(f'Fallback repair finished; logs: {job.resolve()}', flush=True)
                    return job
                if server.poll() is not None:
                    raise RuntimeError(f'Qwen Writer exited; inspect {job / "writer.log"}')
                time.sleep(2)
        finally:
            already_failed = sys.exc_info()[0] is not None
            failures = []
            for signum in handlers:
                signal.signal(signum, signal.SIG_IGN)
            try:
                for process in reversed(processes):
                    try:
                        stop_group(process)
                    except Exception as exc:
                        failures.append(exc)
                        print(f'Owned process cleanup failed ({type(exc).__name__}); continuing remaining cleanup.',
                              file=sys.stderr, flush=True)
            finally:
                for signum, previous in handlers.items():
                    signal.signal(signum, previous)
            if failures and not already_failed:
                raise RuntimeError('Repair finished but owned process cleanup failed; GPU release is unconfirmed') from failures[0]


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--source-root', default=DEFAULT_SOURCE)
    parser.add_argument('--accept-length-exceptions',action='store_true',
                        help='Reuse successful repairs and saved outputs under the approved 500/500/30 budgets')
    parser.add_argument('--pending-parents',action='store_true',help='Reuse validated children and complete timed-out parents only')
    parser.add_argument('--previous-repairs',help='Previous repair overlay for approved length recovery')
    parser.add_argument('--datasets', nargs='+', default=['split_longmemeval_s'])
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'],
                        default=['train', 'validation', 'test'])
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--max-num-seqs', type=int, default=64)
    parser.add_argument('--max-num-batched-tokens', type=int, default=4096)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.9)
    parser.add_argument('--gpu', help='One allocated physical GPU ID; otherwise use CUDA_VISIBLE_DEVICES')
    parser.add_argument('--base-url', help='Reuse an existing Writer; do not deploy a GPU service')
    parser.add_argument('--output-root')
    parser.add_argument('--limit-sessions', type=int)
    parser.add_argument('--job-dir')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main():
    launch(argument_parser().parse_args())


if __name__ == '__main__':
    main()
