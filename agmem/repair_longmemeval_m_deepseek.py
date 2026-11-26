"""Repair LongMemEval-M train or validation placeholders into resumable overlays."""
import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3

import yaml

from .build_deepseek_directories import DeepSeekDirectoryWriter, remote_endpoint
from .build_deepseek_previous_segment import CHILD_PROMPT
from .core import atomic_json, canonical, digest, file_digest
from .execution import bounded_results
from .hierarchical.binding import directory_version, validate_directory
from .hierarchical.deepseek_client import DeepSeekWriterPool
from .hierarchical.writer import TOKENIZER_LOCK, token_count
from .jobs import file_lock
from .pipeline import announce
from .policy import load_tokenizer
from .repair_topic_event_fallbacks import (
    GUIDANCE, VERSION, is_placeholder, validate_replacement,
)
from .reuse_adjacent_directories import source_connection

CONFIG = Path('configs/build_longmemeval_m_deepseek_train_missing.yaml')
OUTPUT = Path('runs/fallback_repair/longmemeval_m_train_deepseek_400_600')
# The previous writer accepted only entries within the same field limits.
SUMMARY_ONLY_IMPLEMENTATION = '531127e6fafc3b57ab90d323af68b4feaa8327101c9e5fb142eecc29d19caf54'
TRAIN_TRUNCATION_IMPLEMENTATION = 'c4d9f5b018bbf6eaaae87ba617bf6665e4754c6cbcc0fa9c436ede70b8ba3dbc'


def truncate_summary(text, tokenizer, limit):
    """Return an exact prefix, dropping a partial UTF-8 character at the cut."""
    with TOKENIZER_LOCK:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= limit:
            return text
        for end in range(limit, 0, -1):
            prefix = tokenizer.decode(ids[:end], skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False).rstrip()
            if (prefix and text.startswith(prefix)
                    and len(tokenizer.encode(prefix, add_special_tokens=False)) <= limit):
                return prefix
    raise ValueError('No nonempty text prefix fits its token budget')


class RepairWriter(DeepSeekDirectoryWriter):
    def __init__(self, tokenizer, settings, complete):
        super().__init__(tokenizer, settings, complete)
        self.adjacent_prompts['child'] = CHILD_PROMPT.read_text()
        for kind in ('child', 'parent'):
            self.adjacent_prompts[kind] += GUIDANCE

    def _raw_payload(self, start, end, kind, children, context_count=0):
        data = super()._raw_payload(start, end, kind, children, context_count)
        if kind == 'child':
            del data['context_blocks']
            data['context_segments'] = []
            if self.previous_span:
                a, b = self.previous_span
                data['context_segments'].append({'start_block': a, 'end_block': b,
                                                  'blocks': self.public['blocks'][a-1:b]})
        return data

    def _validate_entry(self, output, kind, payload):
        # Only normalize complete parsed objects. Malformed JSON goes through
        # the existing Writer's corrective retries with the original source.
        if not isinstance(output, dict) or kind not in ('child', 'parent'):
            return super()._validate_entry(output, kind, payload)
        changes = []
        for field, limit, stat in [('title', self.settings['title_tokens'], 'titles_truncated'),
                                   ('content', self.settings[f'{kind}_content_tokens'], 'summaries_truncated')]:
            text = output.get(field)
            if isinstance(text, str):
                output[field] = truncate_summary(text.strip(), self.tokenizer, limit)
                if output[field] != text.strip():
                    changes.append(f'{kind}_{stat}')
        entities = output.get('entities')
        if kind == 'child' and isinstance(entities, list) and all(isinstance(e, str) for e in entities):
            unique = dict.fromkeys(e.strip() for e in entities if e.strip())
            output['entities'] = list(unique)[:self.settings['max_entities']]
            if output['entities'] != entities:
                changes.append('entity_lists_normalized')
        if isinstance(output.get('content'), str) and is_placeholder(output):
            raise ValueError('Provide a source-grounded summary instead of placeholder text')
        super()._validate_entry(output, kind, payload)
        for key in changes:
            self.stats[key] = self.stats.get(key, 0) + 1

    def repair(self, session, old):
        validate_directory(old, session)
        self.prepare(session)
        self.stats.update(child_summaries_truncated=0, parent_summaries_truncated=0)
        new = deepcopy(old)
        inputs = []
        self.previous_span = None
        for child in new['children']:
            start, end = child['source']['start_block'], child['source']['end_block']
            if is_placeholder(child):
                entry, failed = self._summarize_range(start, end, 'child', [])
                if failed:
                    raise ValueError(f"Child repair exhausted: {child['id']}")
                child.update(entry)
            inputs.append({'start_block': start, 'end_block': end, 'title': child['title'],
                           'content': child['content'], 'entities': child['entities']})
            self.previous_span = start, end
        if is_placeholder(old):
            entry, failed = self._summarize_range(1, len(session['blocks']), 'parent', inputs)
            if failed:
                raise ValueError('Parent repair exhausted')
            new.update(entry)
        new['entities'] = list(dict.fromkeys(e for child in new['children'] for e in child['entities']))
        new['build'].update(status='complete', writer_model='deepseek-flash',
                            writer_revision='provider-alias:deepseek-flash')
        if self._windowed:
            new['build']['mode'] = 'windowed'
        new['directory_version'] = directory_version(new)
        validate_replacement(old, new, session)
        return new, self.stats


class ValidationRepairWriter(RepairWriter):
    """Give the model one chance to condense the summary before clipping it."""

    def _entry_messages(self, payload, kind, previous=None, errors=None):
        self.first_attempt = previous is None
        return super()._entry_messages(payload, kind, previous, errors)

    def _validate_entry(self, output, kind, payload):
        text = output.get('content') if isinstance(output, dict) else None
        if self.first_attempt and kind in ('child', 'parent') and isinstance(text, str):
            limit = self.settings[f'{kind}_content_tokens']
            if token_count(self.tokenizer, text.strip()) > limit:
                raise ValueError(f'Condense {kind}.content to at most {limit} tokens, preserving key facts.')
        super()._validate_entry(output, kind, payload)


class TransportLog:
    def trace(self, event):
        if event['event'] in ('writer_transport_error', 'writer_validation_failed'):
            announce(event['event'].replace('writer_', 'repair_', 1),
                     **{k: v for k, v in event.items() if k != 'event'})


def selection(source, split='train'):
    # Only keep addresses in RAM. Load original transcripts for in-flight work.
    selected = []
    banks = {bid for bid, payload in source.execute(
        "SELECT bank_id,payload FROM banks WHERE dataset='split_longmemeval_m'")
        if any(m['split'] == split for m in json.loads(payload)['query_memberships'])}
    for bid, pid, sha, encoded in source.execute(
            "SELECT bank_id,parent_id,source_hash,directory FROM sessions WHERE status='fallback' "
            "ORDER BY bank_id,parent_id"):
        if bid not in banks:
            continue
        old = json.loads(encoded)
        children = [c['id'] for c in old['children'] if is_placeholder(c)]
        if not children and not is_placeholder(old):
            raise ValueError(f'Fallback without a recognized placeholder: {bid}/{pid}')
        selected.append(dict(bank_id=bid, parent_id=pid, source_hash=sha, before_status='fallback',
                             before_directory_version=old['directory_version'], before_directory_sha256=digest(old),
                             placeholder_children=children, placeholder_parent=is_placeholder(old)))
    return selected


def run_source(source_root, out, args, settings, service, model_path, *, split='train'):
    with closing(source_connection(source_root)) as source:
        source.execute('BEGIN')
        identity = json.loads(source.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
        selected = selection(source, split)
        announce('repair_plan', source=str(source_root), split=split, sessions=len(selected),
                 parents=sum(r['placeholder_parent'] for r in selected),
                 children=sum(len(r['placeholder_children']) for r in selected), workers=args.workers)
        if args.dry_run:
            return
        manifest = dict(protocol=VERSION, source_root=str(source_root.resolve()),
                        source_identity_sha256=digest(identity), selected=selected)
        run_identity = dict(selection_sha256=digest(manifest), settings=settings, model=service['model'],
                            endpoint=service['base_url'], implementation=file_digest(Path(__file__)),
                            tokenizer_sha256=file_digest(Path(model_path)/'tokenizer.json'),
                            child_prompt_sha256=file_digest(CHILD_PROMPT))
        with file_lock(out/'.build.lock'), closing(sqlite3.connect(out/'repair.sqlite3')) as db:
            run_file = out/'run_manifest.json'
            if run_file.exists():
                saved = json.loads(run_file.read_text())
                compatible = [run_identity]
                if split == 'train':
                    compatible.extend(dict(run_identity, implementation=h) for h in
                                      (SUMMARY_ONLY_IMPLEMENTATION, TRAIN_TRUNCATION_IMPLEMENTATION))
                if saved not in compatible:
                    raise ValueError(f'Selection or repair settings changed for {out}')
            atomic_json(run_file, run_identity)
            atomic_json(out/'repair_manifest.json', manifest)
            db.execute('CREATE TABLE IF NOT EXISTS repairs (bank TEXT, parent TEXT, patch TEXT, '
                       'PRIMARY KEY(bank,parent))')
            done = set(db.execute('SELECT bank,parent FROM repairs'))
            pending = [r for r in selected if (r['bank_id'], r['parent_id']) not in done]
            todo = pending[:args.limit_sessions] if args.limit_sessions else pending
            announce('repair_resume', already_saved=len(done), pending=len(pending), this_run=len(todo))
            counts = Counter()
            if todo:
                tokenizer = load_tokenizer(model_path)
                def inputs():
                    for r in todo:
                        snapshot, encoded = source.execute(
                            'SELECT snapshot,directory FROM sessions WHERE bank_id=? AND parent_id=?',
                            (r['bank_id'], r['parent_id'])).fetchone()
                        yield r, json.loads(snapshot), json.loads(encoded)

                with closing(DeepSeekWriterPool([service['base_url']], args.workers, service, TransportLog())) as pool:
                    def work(item):
                        row, session, old = item
                        writer_cls = ValidationRepairWriter if split == 'validation' else RepairWriter
                        writer = writer_cls(tokenizer, settings, pool.for_session(session))
                        try:
                            new, stats = writer.repair(session, old)
                        except ValueError as exc:
                            return None, str(exc)
                        return dict(protocol=VERSION, **row, source_identity_sha256=manifest['source_identity_sha256'],
                                    directory=new, after_directory_sha256=digest(new), repair_stats=stats,
                                    repair_implementation=run_identity['implementation']), None

                    for (row, _, _), (patch, error) in bounded_results(work, inputs(), args.workers):
                        if error:
                            counts['failed'] += 1
                            announce('repair_failed', bank_id=row['bank_id'], parent_id=row['parent_id'], reason=error)
                            continue
                        with db:
                            db.execute('INSERT INTO repairs VALUES (?,?,?)',
                                       (row['bank_id'], row['parent_id'], canonical(patch)))
                        counts['repaired_now'] += 1
                        if counts['repaired_now'] % 100 == 0:
                            announce('repair_progress', complete=len(done)+counts['repaired_now'], total=len(selected))
            total = db.execute('SELECT COUNT(*) FROM repairs').fetchone()[0]
            report = dict(selected=len(selected), repaired=total, remaining=len(selected)-total, **counts)
            atomic_json(out/'repair_report.json', report)
            announce('repair_report', output=str(out), **report)
            if total == len(selected):
                destination = out/'repair_patches.jsonl'
                tmp = destination.with_suffix('.jsonl.tmp')
                with tmp.open('w') as f:
                    for patch, in db.execute('SELECT patch FROM repairs ORDER BY bank,parent'):
                        f.write(patch+'\n')
                tmp.replace(destination)
                atomic_json(out/'complete.json', dict(n=total, selection_sha256=digest(manifest),
                                                      patches_sha256=file_digest(destination)))
            return counts['failed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=512)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit-sessions', type=int, help='Maximum pending sessions per source for this run')
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit_sessions is not None and args.limit_sessions < 1):
        parser.error('workers and limit-sessions must be positive')
    config = yaml.safe_load(CONFIG.read_text())
    settings = dict(config['writer'], child_content_tokens=400, parent_content_tokens=600, semantic_retries=2)
    service = dict(config['service'])
    service['base_url'] = remote_endpoint(os.environ.get('DEEPSEEK_BASE_URL') or service['base_url'])
    if args.split == 'validation':
        eval_config = yaml.safe_load(Path('configs/build_longmemeval_m_qwen_test_missing.yaml').read_text())
        roots = [Path(eval_config['datasets'][0]['exclude_committed_root'])]
        output = OUTPUT.with_name('longmemeval_m_validation_deepseek_400_600')
    else:
        roots = [Path(config['datasets'][0]['exclude_committed_root']), Path(config['output_root'])]
        output = OUTPUT
    failures = 0
    for root in roots:
        failures += run_source(root, (args.output_root or output)/root.name, args, settings, service,
                               config['model_path'], split=args.split) or 0
    if failures:
        raise SystemExit(f'{failures} session repairs remain unsuccessful. Rerun to retry.')


if __name__ == '__main__':
    main()
