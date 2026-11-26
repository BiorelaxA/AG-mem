"""Train a hierarchical Reader on one or two GPUs with a dedicated Task/Judge GPU.

No Slurm job is required. Select the allocated Reader/Task devices explicitly
with CUDA_VISIBLE_DEVICES, for example 4,5. Checkpoints retain optimizer state.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

from .core import atomic_json, load_config
from .data import task_examples
from .execution import execution_settings
from .four_gpu import free_port, server_limits, stop_group, training_locks, wait_ready
from .jobs import latest_checkpoint
from .pipeline import open_prepared, validate_directories


DEFAULT_CONFIG = 'configs/train_dapo_locomo_lmes_hierarchical_f1_2gpu.yaml'
GIB = 1024 ** 3


def gpu_roles(visible, reader_size=1):
    if type(reader_size) is not int or reader_size not in (1, 2):
        raise ValueError('reader.model_parallel_size must be 1 or 2')
    devices = [s.strip() for s in visible.split(',')]
    count = reader_size + 1
    if (len(devices) != count or len(set(devices)) != count
            or any(s in {'', '-1', 'NoDevFiles'} for s in devices)):
        raise ValueError(f'Select {count} distinct allocated GPUs in CUDA_VISIBLE_DEVICES '
                         f'({reader_size} for the Reader followed by one for Task/Judge)')
    return {'reader': ','.join(devices[:-1]), 'task': devices[-1]}


def training_config(path, output_dir=None):
    config = load_config(path)
    if (config['data'].get('directory_protocol') != 'hierarchical_v1'
            or config['training'].get('algorithm') != 'dapo'
            or config['reward'].get('formula') not in {'f1_cost_v1', 'judge_f1_cost_v1', 'full_directory_orm_v1'}
            or config['training'].get('process_reward', {}).get('enabled', False)
            or config['reward'].get('likelihood', {}).get('mode', 'off') != 'off'):
        raise ValueError('This launcher requires hierarchical F1 or Judge+F1 ORM DAPO without process/likelihood rewards')
    if config['task_agent'].get('service') is not None:
        raise ValueError('Use the managed local 27B Task service, without a Task service override')
    if config['service'].get('request_profile', 'local_vllm') != 'local_vllm':
        raise ValueError('This training workflow uses local Qwen Task generation only')
    config['service']['model'] = 'agmem-27b'
    config['reader']['device'] = 'cuda:0'
    for key in ('reader_path', 'task_path'):
        config['models'][key] = str(Path(config['models'][key]).expanduser().resolve())
    config['data']['prepared_root'] = str(Path(config['data']['prepared_root']).expanduser().resolve())
    config['training']['output_dir'] = str(Path(output_dir or config['training']['output_dir']).expanduser().resolve())
    if config['reward'].get('formula') in {'judge_f1_cost_v1', 'full_directory_orm_v1'}:
        judge = config['reward'].setdefault('judge', {})
        if judge.get('service') is not None:
            raise ValueError('Judge must share the managed local 27B Task service')
        if judge.get('template_file') is not None:
            judge['template_file'] = str(Path(judge['template_file']).expanduser().resolve())
    return config


def model_files(path):
    """Check shard sizes against safetensors headers without loading GPU weights."""
    root = Path(path)
    required = ('config.json', 'tokenizer.json', 'tokenizer_config.json')
    missing = [name for name in required if not (root/name).is_file()]
    if missing:
        raise ValueError(f'Missing model files in {root}: {missing}')
    index = root/'model.safetensors.index.json'
    names = (sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists()
             else sorted(p.name for p in root.glob('*.safetensors')))
    if not names:
        raise ValueError(f'Missing model weights in {root}')
    total = 0
    for name in names:
        if Path(name).name != name:
            raise ValueError('Model shard names must be local filenames')
        shard = root/name
        if not shard.is_file():
            raise ValueError(f'Missing model shard: {shard}')
        size = shard.stat().st_size
        with shard.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f'Incomplete model shard: {shard}')
            length = struct.unpack('<Q', prefix)[0]
            if not 2 <= length <= min(64*1024**2, size-8):
                raise ValueError(f'Invalid safetensors header: {shard}')
            header = json.loads(stream.read(length))
        offsets = [item['data_offsets'] for key, item in header.items() if key != '__metadata__']
        if (not offsets or any(len(v) != 2 or not 0 <= v[0] <= v[1] for v in offsets)
                or 8+length+max(v[1] for v in offsets) != size):
            raise ValueError(f'Incomplete or malformed model weights: {shard}')
        total += size
    return {'path': str(root), 'weight_files': len(names), 'weight_bytes': total}


def check_dependencies(config):
    versions = {}
    names = ['torch', 'vllm', 'transformers', 'pyarrow', 'yaml']
    if config['training'].get('tensorboard', True):
        names.append('tensorboard')
    for name in names:
        if util.find_spec(name) is None:
            raise ValueError(f'Missing Python dependency {name}; run from the agmem environment')
        versions[name] = metadata.version('PyYAML' if name == 'yaml' else name)
    return versions


def check_frozen_inputs(config):
    # Deliberately do not call Store.close(): check-only must not rewrite JSONL
    # indexes while inspecting an immutable source snapshot.
    store = open_prepared(config)
    validate_directories(config, store, ['train', 'validation'])
    examples = {split: list(task_examples(config, store, split)) for split in ('train', 'validation')}
    counts = {split: Counter(e.source for e in values) for split, values in examples.items()}
    if any(not counts[split].get(source) for split in counts for source in config['data']['sources']):
        raise ValueError('Every selected source needs nonempty train and validation data')
    result = {'by_source': {source: {split: counts[split][source] for split in counts}
                           for source in config['data']['sources']},
              'total': {split: sum(values.values()) for split, values in counts.items()},
              'directory_version': store.get_metadata('directory_version'),
              'directory_protocol': 'hierarchical_v1'}
    if config['reward'].get('formula') in {'judge_f1_cost_v1', 'full_directory_orm_v1'}:
        from .judges import JudgeScorer
        from .policy import model_fingerprint
        service = dict(config['service'])
        service['model_revision'] = model_fingerprint(config['models']['task_path'])['fingerprint']
        # Construction and routing/template validation do not send requests.
        # A None store also prevents preflight from touching completion caches.
        judge = JudgeScorer(service, None, config['reward'].get('judge', {}))
        try:
            for values in examples.values():
                for example in values:
                    judge.validate_example(example)
            result['judge'] = judge.provenance()
        finally:
            judge.close()
    if config['reward'].get('formula') == 'full_directory_orm_v1':
        from .full_directory_preflight import initial_context_report
        result['reader_initial_context'] = initial_context_report(config, store, examples)
    return result


def storage_report(config, reader_weight_bytes):
    output = Path(config['training']['output_dir'])
    existing = output
    while not existing.exists():
        if existing.parent == existing:
            raise ValueError(f'No existing ancestor for training output: {output}')
        existing = existing.parent
    if not existing.is_dir() or not os.access(existing, os.W_OK | os.X_OK):
        raise ValueError(f'Training output parent is not a writable directory: {existing}')
    # BF16 model + FP32 master weights and Adam moments: approximately 7x the
    # weight size. Include additional margin for metadata, logs and atomic save.
    estimated = 7 * reader_weight_bytes
    minimum = max(80 * GIB, estimated + 10 * GIB)
    keep_last = max(0, config['training'].get('keep_last_checkpoints', 2))
    retention = config['training'].get('checkpoint_retention', 'legacy')
    overwrite = retention == 'best_and_latest_overwrite'
    retained_copies = 2 if retention in {'best_and_latest', 'best_and_latest_overwrite'} else keep_last + 2
    peak_copies = retained_copies if overwrite else retained_copies + 1
    recommended = estimated * peak_copies + 10 * GIB
    if overwrite:
        minimum = max(minimum, recommended)
    free = shutil.disk_usage(existing).free
    return {'output_dir': str(output), 'filesystem_checked_at': str(existing),
            'free_bytes': free, 'minimum_required_bytes': minimum,
            'estimated_full_checkpoint_bytes': estimated,
            'checkpoint_retention': retention,
            'estimated_peak_checkpoint_copies': peak_copies,
            'recommended_free_bytes': recommended, 'storage_ready': free >= minimum,
            'checkpoint_contents': 'model_and_full_CPUAdamW_state',
            'note': 'Overwrite mode frees the obsolete latest before saving, while protecting the complete best; '
                    'best/latest share one payload when equal. Peak storage is two full checkpoints.'
                    if overwrite else
                    'Old checkpoints are pruned only after a complete atomic save; best_and_latest can temporarily need three full copies.'
                    if retention == 'best_and_latest' else
                    'Minimum permits an initial save; retention and other writers may need more space.'}


def preflight_report(config):
    models = {name: model_files(config['models'][name]) for name in ('reader_path', 'task_path')}
    dependencies = check_dependencies(config)
    inputs = check_frozen_inputs(config)
    storage = storage_report(config, models['reader_path']['weight_bytes'])
    return {'inputs': inputs, 'models': models, 'dependencies': dependencies, 'storage': storage,
            'gpu_checked': False,
            'execution': execution_settings(config)}


def task_command(config, port):
    limits = server_limits('task', config)
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', config['models']['task_path'], '--served-model-name', 'agmem-27b',
            '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
            '--tensor-parallel-size', '1', '--max-model-len', str(config['task_agent']['max_context_tokens']),
            '--max-num-seqs', str(limits['max_num_seqs']),
            '--max-num-batched-tokens', str(limits['max_num_batched_tokens']),
            '--gpu-memory-utilization', str(limits['gpu_memory_utilization']),
            '--enable-chunked-prefill', '--language-model-only', '--enforce-eager']


def launch(args):
    if args.max_updates is not None and args.max_updates < 1:
        raise ValueError('--max-updates must be positive')
    config = training_config(args.train_config, args.output_dir)
    report = preflight_report(config)
    enforce_storage = config['training'].get('enforce_storage_reserve', True)
    if type(enforce_storage) is not bool:
        raise ValueError('training.enforce_storage_reserve must be boolean')
    report['storage']['enforced'] = enforce_storage
    report['ready_to_launch'] = report['storage']['storage_ready'] or not enforce_storage
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report
    if enforce_storage and not report['storage']['storage_ready']:
        space = report['storage']
        raise RuntimeError(f"Insufficient training output space: {space['free_bytes']/GIB:.1f} GiB available, "
                           f"at least {space['minimum_required_bytes']/GIB:.1f} GiB required before GPU startup. "
                           'Free space or pass --output-dir on a filesystem with enough capacity; '
                           'full resumable checkpoints are retained.')
    if not enforce_storage:
        print(f"Storage reserve enforcement disabled by configuration; "
              f"available space: {report['storage']['free_bytes']/GIB:.1f} GiB (informational).", flush=True)
    reader_size = config['reader'].get('model_parallel_size', 1)
    roles = gpu_roles(os.environ.get('CUDA_VISIBLE_DEVICES', ''), reader_size)
    gpu_count = reader_size + 1
    destination = Path(config['training']['output_dir'])
    if (destination == Path(config['data']['prepared_root'])
            or destination in Path(config['data']['prepared_root']).parents
            or Path(config['data']['prepared_root']) in destination.parents):
        raise ValueError('Training output must be separate from the frozen data directory')
    with training_locks(config):
        checkpoint = latest_checkpoint(destination) if args.resume else None
        if args.resume and checkpoint is None:
            raise ValueError('--resume requested but no committed training checkpoint exists')
        if not args.resume and destination.exists() and any(destination.iterdir()):
            raise ValueError('Training output already exists; use --resume or a new --output-dir')
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        job_prefix = 'three-gpu' if reader_size == 2 else 'two-gpu'
        job = destination.parent/'jobs'/f'{job_prefix}-{stamp}-{os.getpid()}'
        job.mkdir(parents=True, exist_ok=False)
        atomic_json(job/'preflight.json', report)
        atomic_json(job/'gpu_roles.json', roles)
        service_role = 'Task/Judge' if config['reward']['formula'] in {'judge_f1_cost_v1', 'full_directory_orm_v1'} else 'Task'
        print(f'Training logs: {job.resolve()}; Reader GPU: {roles["reader"]}; {service_role} GPU: {roles["task"]}', flush=True)
        environment = {**os.environ, 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4'}
        environment.pop('AGMEM_READER_MODEL_PARALLEL_SIZE', None)
        environment.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        environment.setdefault('AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS', '8192')
        processes, handlers = [], {}

        def interrupted(signum, frame):
            raise SystemExit(128 + signum)

        with ExitStack() as logs:
            def start(name, command, devices, extra=None):
                log = logs.enter_context((job/f'{name}.log').open('w'))
                child = {**environment, 'CUDA_VISIBLE_DEVICES': devices, **(extra or {})}
                process = subprocess.Popen(command, env=child, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                processes.append(process)
                return process

            try:
                for signum in (signal.SIGINT, signal.SIGTERM):
                    handlers[signum] = signal.signal(signum, interrupted)
                probe = ('import json, torch; import vllm._C\n'
                         f"assert torch.cuda.is_available() and torch.cuda.device_count()=={gpu_count}, 'Expected {gpu_count} CUDA GPUs'\n"
                         f'for i in range({gpu_count}):\n'
                         ' free,total=torch.cuda.mem_get_info(i)\n'
                         " print(json.dumps({'gpu':i,'name':torch.cuda.get_device_name(i),'free_gib':free/2**30,'total_gib':total/2**30}),flush=True)\n"
                         " assert total>=70*2**30, 'Each selected device must be an 80GB-class GPU'\n"
                         " assert free/total>=0.90, 'Selected GPU is occupied; free it before training'\n"
                         " (torch.ones((32,32),device=f'cuda:{i}')@torch.ones((32,32),device=f'cuda:{i}')).sum().item()\n")
                check = start('cuda_preflight', [sys.executable, '-c', probe], ','.join(roles.values()))
                if check.wait() != 0:
                    raise RuntimeError(f'CUDA/vLLM preflight failed; inspect {job/"cuda_preflight.log"}')
                stop_group(check); processes.remove(check)
                port = free_port()
                config['service']['request_base_url'] = f'http://127.0.0.1:{port}/v1'
                import yaml
                runtime_config = job/'train_config.yaml'
                runtime = deepcopy(config)
                # This temporary launcher policy does not affect updates or
                # the resume cursor; keep it out of frozen training parameters.
                runtime['training'].pop('enforce_storage_reserve', None)
                runtime_config.write_text(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False))
                server = start('task_vllm', task_command(config, port), roles['task'])
                wait_ready(server, port, job/'task_vllm.log')
                command = [sys.executable, '-u', '-m', 'agmem', '--config', str(runtime_config), 'train']
                if checkpoint:
                    command.extend(['--resume', str(checkpoint)])
                if args.max_updates is not None:
                    command.extend(['--max-groups', str(args.max_updates)])
                trainer = start('train', command, roles['reader'],
                                {'AGMEM_READER_MODEL_PARALLEL_SIZE': str(reader_size), 'OMP_NUM_THREADS': '8'})
                recipe = ('Full-directory ORM' if config['reward']['formula'] == 'full_directory_orm_v1'
                          else 'Judge+F1 ORM' if config['reward']['formula'] == 'judge_f1_cost_v1' else 'F1-ORM')
                print(f'{recipe} training started; progress: {job.resolve()/"train.log"}', flush=True)
                while True:
                    status = trainer.poll()
                    if status is not None:
                        if status:
                            raise RuntimeError(f'Reader training failed ({status}); inspect {job/"train.log"}')
                        print(f'Training finished: {destination}', flush=True)
                        return job
                    if server.poll() is not None:
                        raise RuntimeError(f'Task service exited; inspect {job/"task_vllm.log"}')
                    time.sleep(2)
            finally:
                failed = sys.exc_info()[0] is not None
                errors = []
                for signum in handlers:
                    signal.signal(signum, signal.SIG_IGN)
                try:
                    for process in reversed(processes):
                        try:
                            stop_group(process)
                        except Exception as exc:
                            errors.append(exc)
                            print(f'Owned process cleanup failed ({type(exc).__name__}); continuing cleanup.',
                                  file=sys.stderr, flush=True)
                finally:
                    for signum, previous in handlers.items():
                        signal.signal(signum, previous)
                if errors and not failed:
                    raise RuntimeError('Training finished but an owned process group could not be stopped') from errors[0]


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-config', default=DEFAULT_CONFIG)
    parser.add_argument('--check-only', action='store_true', help='CPU-only input/model/storage checks; no model services')
    parser.add_argument('--resume', action='store_true', help='Resume latest complete checkpoint including optimizer state')
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--output-dir', help='Training/checkpoint directory on a filesystem with sufficient free capacity')
    return parser


def main():
    launch(argument_parser().parse_args())


if __name__ == '__main__':
    main()
