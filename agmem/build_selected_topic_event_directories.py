"""Validate completion of the selected build scope without changing generation.

The frozen engine reports incomplete sessions across its whole journal. A job
building S test or M may leave unrelated S train gaps. Keep that global report,
but validate every requested query/session and its exported directory separately.
"""
from collections import Counter
import json
from pathlib import Path
import sqlite3

from . import build_topic_event_directories as topic
from .core import InfrastructureError, atomic_json, digest, file_digest
from .hierarchical.adapters import input_files
from .hierarchical.binding import validate_directory
from .hierarchical.records import validate_session
from .jobs import file_lock
from .pipeline import announce


def verify_selected_scope(args, config, identity):
    import pyarrow.parquet as pq

    root = Path(config['output_root'])
    specs = {s['name']: s for s in config['datasets']}
    datasets = args.datasets or list(specs)
    if not datasets or not set(datasets).issubset(specs):
        raise ValueError('Unknown selected dataset')
    recorded = {(item['spec']['name'], f['split']): f['sha256']
                for item in identity['datasets'] for f in item['files']}
    expected, files, queries = {}, {}, Counter()
    for name in datasets:
        if specs[name].get('format') != 'longmemeval_parquet':
            raise ValueError('Scoped completion currently supports LongMemEval Parquet only')
        selected_splits = args.splits or specs[name].get('splits', ['train'])
        available = dict(input_files(specs[name]))
        if not set(selected_splits).issubset(available):
            raise ValueError('Selected split is absent from the recipe')
        for split in selected_splits:
            path = available[split]
            sha = file_digest(path)
            if sha != recorded.get((name, split)):
                raise ValueError('Selected input differs from the construction identity')
            files[str(path)] = sha
            rows = pq.read_table(path, columns=['question_id', 'haystack_session_ids']).to_pylist()
            for index, row in enumerate(rows):
                ids = row['haystack_session_ids']
                ids = json.loads(ids) if isinstance(ids, str) else ids
                if not isinstance(ids, list) or not ids:
                    raise ValueError('Expected a nonempty history in each selected query')
                bank = 'bank_' + digest([name, split, sha, index])[:24]
                expected[bank] = (name, split, index, row['question_id'], len(ids))
                queries[(name, split)] += 1
    if not expected:
        raise ValueError('Selected scope contains no queries')

    with file_lock(root / '.build.lock'):
        directory_hashes, states = {}, Counter()
        db = sqlite3.connect((root/'build_journal.sqlite3').resolve().as_uri()+'?mode=ro', uri=True)
        try:
            stored = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if stored is None or json.loads(stored[0]) != identity:
                raise ValueError('Journal construction identity changed')
            for bid, (name, split, index, qid, count) in expected.items():
                row = db.execute('SELECT dataset,payload FROM banks WHERE bank_id=?', (bid,)).fetchone()
                if row is None or row[0] != name:
                    raise InfrastructureError(f'Selected query has no matching directory bank: {name}/{split}/{index}')
                bank = json.loads(row[1])
                if not any(m.get('split') == split and m.get('row_index') == index
                           and m.get('original_question_id') == qid for m in bank.get('query_memberships', [])):
                    raise ValueError('Selected query membership differs from its input')
                if any(i.get('status') == 'failed' for i in bank.get('issues', [])):
                    raise InfrastructureError('Selected query has a source adaptation failure')
                rows = db.execute('SELECT parent_id,status,snapshot,directory FROM sessions WHERE bank_id=?', (bid,)).fetchall()
                if {r[0] for r in rows} != {f'p{i:04d}' for i in range(1, count+1)}:
                    raise InfrastructureError(f'Selected query still has missing sessions: {bid}')
                for pid, status, snapshot, encoded in rows:
                    if status not in {'complete', 'fallback', 'empty'} or encoded is None:
                        raise InfrastructureError(f'Selected session is unfinished: {bid}/{pid} ({status})')
                    session, directory = json.loads(snapshot), json.loads(encoded)
                    validate_session(session)
                    validate_directory(directory, session)
                    if (session['bank_id'] != bid or session['parent_id'] != pid
                            or directory['build']['status'] != status):
                        raise ValueError('Selected session/directory address or status differs')
                    directory_hashes[(bid, pid)] = digest(directory)
                    states[status] += 1
        finally:
            db.close()
        seen = set()
        with (root/'directories.jsonl').open() as stream:
            for line in stream:
                item = json.loads(line)
                key = (item['bank_id'], item['id'])
                if key[0] not in expected:
                    continue
                if key in seen or directory_hashes.get(key) != digest(item):
                    raise ValueError('Selected exported directory differs from the journal')
                seen.add(key)
        if seen != set(directory_hashes):
            raise InfrastructureError('Selected directory exports are incomplete')
        for path, sha in files.items():
            if file_digest(path) != sha:
                raise ValueError('Selected input changed during verification')
        result = {'selected_scope_complete': True, 'queries': len(expected), 'sessions': len(seen),
                  'states': dict(states), 'construction_identity_sha256': digest(identity),
                  'scope': [{'dataset': name, 'split': split, 'queries': n}
                            for (name, split), n in sorted(queries.items())],
                  'global_completion_unchanged': True}
        atomic_json(root/f'selected_scope_{digest(result["scope"])[:16]}.json', result)
    return result


def run(args, **injected_factories):
    config, identity = topic.load_recipe(args.config, args.workers,
        embedding_model_path=getattr(args, 'embedding_model_path', None),
        output_root=getattr(args, 'output_root', None), allow_missing_embedding=args.dry_run)
    root = Path(config['output_root'])
    try:
        report = topic.run(args, **injected_factories)
    except InfrastructureError as exc:
        terminal = f'Some sessions failed or remain pending; inspect {root / "build_report.json"}; rerun to resume'
        if args.dry_run or getattr(args, 'limit_sessions', None) is not None or str(exc) != terminal:
            raise
        origin = exc.__traceback__
        while origin is not None and origin.tb_next is not None:
            origin = origin.tb_next
        if origin is None or origin.tb_frame.f_code is not topic.legacy.run.__code__:
            raise
        report = json.loads((root/'build_report.json').read_text())
        execution = report.get('execution', {})
        if execution.get('run_status') != 'incomplete' or execution.get('fatal_error'):
            raise
        if (execution.get('datasets') != (args.datasets or [s['name'] for s in config['datasets']])
                or execution.get('splits') != (args.splits or ['train', 'validation', 'test'])):
            raise
    if args.dry_run or getattr(args, 'limit_sessions', None) is not None:
        return report
    selected = verify_selected_scope(args, config, identity)
    announce('selected_directory_scope_finished', **selected)
    return {**report, 'selected_scope': selected}


def main():
    run(topic.argument_parser().parse_args())


if __name__ == '__main__':
    main()
