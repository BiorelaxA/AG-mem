# 新服务器迁移验收清单

本清单用于文件全部转移后，确认新服务器是否完整保留 AG-mem 的代码、新旧数据、构建状态、实验记录、checkpoint20/32 和当前会话。它是目标端验收文档；传输命令和项目设计交接见 [migration.md](migration.md)。

源端核查基线为 2026-11-12。若打包前又有新构建或评测，以实际数据包里的 `SOURCE_FILES.jsonl`、`ARCHIVE_MANIFEST.json` 和源端最新清单为准，不把本文的旧进度当作迁移后的最新进度。

约定：代码来自 GitHub；全部新旧非checkpoint数据来自 HF；仅 checkpoint20/32 通过 rsync；基础模型和 Python 环境在目标端重建；会话 JSONL 单独传输。其他 checkpoint 缺失是预期行为，不应补传或据此判断失败。

## 1. 先确定路径与检查顺序

以下命令在新服务器执行，替换为实际路径。先完成文件检查，再运行构建、训练或评测，以免程序更新日志、缓存和状态文件，影响与归档的比较。

```bash
export PROJECT='/share/home/wtma/AG-mem'
export PACK_DIR='/你的路径/agmem-migration-download'
export TRAIN_REL='runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train'
cd "$PROJECT"
```

按顺序检查：归档校验 → 解压文件 → 代码/文档 → 数据/目录 → checkpoint → 会话 → 重建环境/模型 → 程序预检查。前六项通过说明文件迁移完整，最后两项通过才说明可以运行；二者不能互相替代。

## 2. 需要首先阅读和保留的文档

| 文件 | 用途与检查重点 |
|---|---|
| `doc/migration_destination_checklist.md` | 本验收清单；应是最后一次代码提交中的版本 |
| `doc/migration.md` | 当前迁移范围、传输命令和“给新会话的项目交接”；底部旧迁移说明只作历史参考 |
| `README.md` | 各构建、训练、评测入口；辨别旧单层与新父子目录流程 |
| `doc/build_directory.md` | 父子目录结构、session/子目录项含义和构建要求 |
| `doc/dapo_judge_training.md` | DAPO/Judge 相关设计记录，与所选配置及保存的 run.json 一起阅读 |
| `method.png` | 过程奖励/动作归因设计图；迁移不改变其定义 |
| `doc/environment_setup.md`、`environment.yml`、`requirements*.txt`、`pyproject.toml` | 安装入口；旧候选依赖版本不等于实际源端环境 |
| `reports/migration_20261112/environment_and_models.json` | 实际环境和五个模型的快速指纹；不包含完整权重内容哈希 |
| `reports/migration_20261112/installed_packages.txt` | 源端包版本清单；不是已验证的跨服务器安装锁文件 |
| `reports/migration_20261112/working_tree.json` | 当时尚未提交文件的记录，防止只 clone 远端旧代码而漏掉新实现 |
| `reports/migration_20261112/payload_sizes.json` | 原项目占用和checkpoint清单；含有明确不迁移的旧checkpoint，不要求全部复现 |

这些报告随 HF 数据包迁移；不能因为它们没有加入 GitHub 就判断源端没有生成。

## 3. 数据包本身是否完整

必须存在以下文件，不能只下载压缩分片：

| 文件 | 正常状态 |
|---|---|
| `COMPLETE.json` | 生产打包成功后生成；没有它表示包未完成 |
| `ARCHIVE_MANIFEST.json` | 记录协议、分片顺序、源文件数量/总大小、代码 commit、排除目录 |
| `SOURCE_FILES.jsonl` | 所有入包文件及目录的相对路径、类型、大小、mtime_ns、权限 |
| `SHA256SUMS` | 覆盖压缩分片及两份清单的 SHA256 |
| `archive.tar.zst.part-00000` 等 | 与 manifest 的分片名称、数量、大小一致 |

```bash
cd "$PACK_DIR"
test -s COMPLETE.json
test -s ARCHIVE_MANIFEST.json
test -s SOURCE_FILES.jsonl
sha256sum -c SHA256SUMS
```

通过标准：全部显示 `OK` 且命令退出码为0。校验失败时先重新下载对应分片，不继续解压或启动程序。

再核对完成标识与分片集合，防止目录中混入另一批分片：

```bash
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
p = Path(os.environ['PACK_DIR'])
m = json.loads((p / 'ARCHIVE_MANIFEST.json').read_text())
c = json.loads((p / 'COMPLETE.json').read_text())
assert m['protocol'] == c['protocol'] == 'agmem_migration_data_v1'
assert c['sha256sums_sha256'] == hashlib.sha256((p / 'SHA256SUMS').read_bytes()).hexdigest()
names = [r['name'] for r in m['parts']]
assert names == sorted(names) and len(names) == len(set(names)) == c['part_count']
assert set(names) == {f.name for f in p.glob('archive.tar.zst.part-*')}
assert all((p / r['name']).stat().st_size == r['bytes'] for r in m['parts'])
print('PASS: package metadata and complete part set')
print('Code commit at packaging:', m['git_commit'])
print('Source working tree was dirty:', m['git_dirty'])
PY
```

分片是一个连续的 tar.zst 流，需要按名称拼接后解压；不是分别可解压的多个独立 tar。代码只读检查所使用的 Git commit 应与包记录对应。`git_dirty=true` 可能来自未提交日志或迁移报告，必须核对实际未提交内容，不能仅凭该标志认定业务代码已完整提交。

## 4. 按清单核对所有已解压数据

不要把“目录存在”或“du体积相近”当成全部文件已恢复。以下命令逐条检查清单中的所有文件和目录，不需要安装训练依赖：

```bash
python3 - <<'PY'
import json, os, stat
from pathlib import Path
root = Path(os.environ['PROJECT']).resolve()
pack = Path(os.environ['PACK_DIR'])
manifest = json.loads((pack / 'ARCHIVE_MANIFEST.json').read_text())
missing, wrong, changed_time = [], [], []
count = total = 0
for line in (pack / 'SOURCE_FILES.jsonl').open():
    if not line.strip():
        continue
    r = json.loads(line)
    rel = Path(r['path'])
    assert not rel.is_absolute() and '..' not in rel.parts
    p = root / rel
    is_file = r['type'] == 'file'
    if is_file:
        count += 1
        total += r['bytes']
    try:
        s = p.lstat()
    except FileNotFoundError:
        missing.append(r['path'])
        continue
    if is_file:
        if not stat.S_ISREG(s.st_mode) or s.st_size != r['bytes']:
            wrong.append(r['path'])
        elif s.st_mtime_ns != r['mtime_ns']:
            changed_time.append(r['path'])
    elif not stat.S_ISDIR(s.st_mode):
        wrong.append(r['path'])
assert count == manifest['file_count'] and total == manifest['source_bytes']
print('Expected files:', count, 'bytes:', total)
print('Missing:', len(missing), 'wrong type/size:', len(wrong), 'file mtime changes:', len(changed_time))
for name, values in [('missing', missing), ('wrong', wrong), ('mtime', changed_time)]:
    for value in values[:20]:
        print(name, value)
assert not missing and not wrong, 'Restore missing/incomplete data before proceeding'
assert not changed_time, 'Investigate file timestamp/content changes before declaring an exact restore'
print('PASS: all archived data files restored')
PY
```

此检查覆盖清单中的所有数据，但大小/时间比较不是逐文件内容哈希。压缩包SHA校验、成功解压和清单检查应一起完成。需要更强的解压后内容复核时，可在尚未运行任何程序的目标目录上对归档执行 GNU tar `--compare`；目录mtime可能因随后复制checkpoint而变化，需区分目录元数据变化和文件内容差异。不要修改时间戳来掩盖失败。

以下额外文件不由这份清单覆盖：GitHub代码、通过rsync迁移的两个checkpoint、单独迁移的会话、重新下载的基础模型。后续分别检查。

## 5. 代码、配置和prompt资源

```bash
cd "$PROJECT"
git rev-parse HEAD
git status --short
git diff -- agmem configs scripts tests doc README.md pyproject.toml
```

核对包记录的代码commit或实际迁移快照commit。若验收文档是在打包之后单独补交，可以说明文档差异，但不能把尚未提交的业务代码也忽略。HF解压可能改变已跟踪的旧LFS数据，`git status`非空不一定意味着代码错；不要运行 `git reset --hard` 或 `git clean` 来使结果看起来干净。

| 文件/目录 | 检查内容 |
|---|---|
| `agmem/`、`configs/`、`scripts/`、`tests/` | 新旧实现全部保留，不能只迁移最近新增模块 |
| `agmem/build_local_directories.py`、`build_local_directories_gpu.py` | 旧ORM单层目录构建与三卡启动 |
| `agmem/build_parent_child_directories.py`、`build_adjacent_directories.py`、`build_adjacent_directories_gpu.py` | 旧父子目录与相邻语义聚类构建 |
| `agmem/build_topic_event_directories.py`、`reuse_topic_event_directories.py` | 新话题/事件灰区规则和保留旧目录的增量导入 |
| `agmem/hierarchical/` | records、adapters、storage、segmentation、embedding、reader、writer等完整模块 |
| `agmem/hierarchical/resources/` | 所有schema与prompt文本递归保留；不仅顶层文件 |
| `agmem/hierarchical/resources/adjacent/topic_event_v2/boundary.txt` | 新灰区合并prompt，尤其容易遗漏的新增子目录资源 |
| `agmem/evaluate_checkpoints.py`、`evaluate_gpu.py`、`evaluation_*.py` | 检索、评分、目录身份校验、三卡评测与离线证据导出 |
| `agmem/remote_task_runner.py`、`remote_judge.py`、`judge_gpt_responses.py` | 联网节点GPT回答和评分 |
| `agmem/process_rewards.py`、`process_advantages.py` | PRM代码仍需保留，不能因当前运行ORM而漏掉 |
| `configs/protocols/format05_training_judge.json` | 旧训练冻结Judge模板，不能用当前默认prompt替代 |
| `scripts/package_migration_data.py` | 本次打包规则和排除范围的可复现实现 |

重点配置与脚本应成对存在：

| 配置 | 入口/用途 |
|---|---|
| `configs/build_longmemeval_811_flat_qwen.yaml` | `scripts/build_longmemeval_811_flat_qwen.sh`，旧单层目录与后续评测 |
| `configs/eval_longmemeval_811_orm_checkpoints.yaml` | checkpoint20/32 × 新S/M测试集，本地Qwen回答与Judge |
| 同上 | `scripts/eval_longmemeval_811_gpt.sh`，复用证据做GPT评测 |
| `configs/build_adjacent_qwen_longmemeval_811_incremental.yaml` | `scripts/build_longmemeval_811_hierarchical_qwen.sh`，S增量父子目录 |
| `configs/build_adjacent_qwen_topic_event_v2.yaml` | `scripts/build_topic_event_directories_qwen.sh`，独立新规则构建 |
| `configs/train_dapo_hierarchical_f1.yaml` | 新父子目录F1 ORM训练；所有`extends`引用的父配置也必须存在 |
| `configs/train_dapo_directory_process_v1.yaml`及历史训练配置 | PRM与旧实验复现记录，保留但不混作当前ORM配置 |

## 6. 所有数据划分与原始内容

`data/processed/` 下必须保留实际入包的全部版本，至少核对源端已有的五个版本：

```text
query_dedup_val10_seed42_20261018/
query_dedup_val100_lmes34_seed42_20261031/
query_dedup_val10_plus1pct_seed42_20261031/
locomo_query9010_seed42_20261102/
longmemeval_811_seed42_20261111/
```

每版的 Parquet 与manifest、行号映射、审计文件须成套。新LoCoMo和新811根都有 `manifest.json`、`row_mapping.jsonl`。旧manifest中的源服务器绝对路径可以作为历史来源保留，不要直接全文替换；执行时使用的路径需要另行适配。

当前关键样本行数（不是去重后的唯一query数）：

| 数据根/source | train | validation | test |
|---|---:|---:|---:|
| `locomo_query9010_seed42_20261102/locomo` | 1019 | 115 | 400 |
| 同根历史 `split_longmemeval_s` | 316 | 34 | 150 |
| `longmemeval_811_seed42_20261111/split_longmemeval_s` | 400 | 50 | 50 |
| 同根 `split_longmemeval_m` | 400 | 50 | 50 |

旧根里的LME-S要保存，但后续LME训练/验证/测试应使用新811根。安装pyarrow后可只读验证关键Parquet行数和必需的清单文件：

```bash
cd "$PROJECT"
python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq
expected = {
    ('locomo_query9010_seed42_20261102', 'locomo'): [1019, 115, 400],
    ('locomo_query9010_seed42_20261102', 'split_longmemeval_s'): [316, 34, 150],
    ('longmemeval_811_seed42_20261111', 'split_longmemeval_s'): [400, 50, 50],
    ('longmemeval_811_seed42_20261111', 'split_longmemeval_m'): [400, 50, 50],
}
for (version, source), counts in expected.items():
    root = Path('data/processed') / version
    for split, count in zip(['train', 'validation', 'test'], counts):
        p = root / source / f'{split}.parquet'
        actual = pq.read_metadata(p).num_rows
        assert actual == count, (str(p), actual, count)
        print('PASS', p, actual)
    assert (root / 'manifest.json').is_file() and (root / 'row_mapping.jsonl').is_file()
print('PASS: key datasets and split rows')
PY
```

这段命令核对行数与必需清单文件，内容完整性以第3/4节归档校验为基础；需要核对数据内容时再将各Parquet的SHA256与对应manifest中的 `output_files` 记录逐项比较。

## 7. 旧单层目录与新旧父子目录

### 7.1 旧ORM单层LME测试目录

两个目录都必须存在：

```text
runs/locomo_longmemeval/prepared_longmemeval_811_flat/test/split_longmemeval_s/
runs/locomo_longmemeval/prepared_longmemeval_811_flat/test/split_longmemeval_m/
```

| 核心文件 | 用途 |
|---|---|
| `manifest.json` | 数据准备身份、目录版本及完成状态 |
| `memory_records.jsonl` | 完整历史原文，不能只有摘要 |
| `test_examples.jsonl` | 测试问题、gold、question_id/question_type等元数据 |
| `directories.jsonl` | 每个session的单层摘要 |
| `directory_manifest.json` | 目录快照完整性记录 |
| `directory_provenance.jsonl` | 每条目录项的Writer来源及内容校验依据 |
| `local_build_manifest.json` | 构建数据、旧prompt和模型身份 |
| `local_build_status.json` | 最终完成报告 |
| `completed_config.json` | 完成后的运行配置记录 |

`preparation_report.json`、`local_build_runtime.json`、`completions.jsonl`及索引也按清单保留。`*.index.json`是可重建偏移索引；缺少它与丢失原文不是同一问题。

完成报告应分别为：S=`ready/50 histories/2403 sessions`；M=`ready/50 histories/23784 sessions`。S有GPT历史摘要复用是原方案允许的来源，不能因此当作迁移混入其他版本。

```bash
python3 - <<'PY'
import json
from pathlib import Path
base = Path('runs/locomo_longmemeval/prepared_longmemeval_811_flat/test')
for source, sessions in [('split_longmemeval_s', 2403), ('split_longmemeval_m', 23784)]:
    p = base / source
    status = json.loads((p / 'local_build_status.json').read_text())
    assert status['status'] == 'ready' and status['histories'] == 50 and status['sessions'] == sessions
    assert sum(bool(line.strip()) for line in (p / 'test_examples.jsonl').open()) == 50
    print('PASS', source, status['histories'], status['sessions'])
PY
```

### 7.2 父子目录构建库

| 根目录 | 源端实际应保存的内容 | 不应误判的情况 |
|---|---|---|
| `runs/parent_child_directories/qwen27b_v1/` | `.build.lock`、`build_journal.sqlite3`、`run_manifest.json` | 源端就没有导出的JSONL，缺JSONL不代表迁移失败 |
| `runs/parent_child_directories/qwen27b_child300_v1/` | 同上 | 同上 |
| `runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1/` | 见下方完整清单 | fallback/failed/pending应原样保留，不要求迁移时变成complete |

相邻聚类根中应保留：

```text
run_manifest.json                 构建版本、输入数据与参数身份
build_journal.sqlite3             每个session的构建状态和断点
embedding_cache.sqlite3           语义向量缓存
bank_manifest.jsonl               历史轨迹/bank清单
blocks.jsonl                      不可拆分的原始block
records.jsonl                     记录与原文映射
source_map.jsonl                  原始数据行与bank映射
directories.jsonl                 父目录和子目录内容
writer_traces.jsonl               Writer调用与修复轨迹
build_report.json                 构建状态汇总
fallback_repair_report.json       后备目录修复记录
retry_failed.json                 失败重试记录
.build.lock                      锁路径；文件存在本身不表示有进程占锁
```

源端本次核查的 `build_report.json`：LoCoMo complete272；LME-S complete19070、fallback50、failed2、pending128，修复状态incomplete。该状态应与打包时保持一致，不因“验收”而重试或重新生成数据。

以下新输出根在本次源端核查时尚未生成；配置存在但输出目录不存在属于正常情况，除非实际包清单后来已经包含它们：

```text
runs/parent_child_directories/qwen27b_adjacent_qwen4b_811_incremental_v2/
runs/parent_child_directories/qwen27b_adjacent_qwen4b_topic_event_v2/
```

SQLite校验应在目标副本和无构建进程的条件下进行。下面使用只读连接，不执行repair/vacuum或删库；`quick_check`通过表示数据库结构可读，不等于所有session都构建成功：

```bash
python3 - <<'PY'
import sqlite3
from pathlib import Path
paths = sorted(Path('runs/parent_child_directories').rglob('*.sqlite3'))
assert paths, 'No migrated directory databases found'
for p in paths:
    db = sqlite3.connect(p.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        rows = db.execute('PRAGMA quick_check').fetchall()
        assert rows == [('ok',)], (str(p), rows)
        print('PASS', p)
    finally:
        db.close()
PY
```

### 7.3 其他prepared和实验记录

还需保留 `runs/locomo_longmemeval/prepared/`、`prepared_locomo_query9010_seed42_20261102/`、`prepared_val100_lmes34_seed42_20261031/`、`prepared_val10_plus1pct_seed42_20261031/`、`runs/locomo_hierarchical_adjacent_f1/prepared/`。它们分别服务于历史实验和父子目录训练，不能被同一个新prepared覆盖。

旧 prepared 原文如果只剩以 `version https://git-lfs.github.com/spec/v1` 开头的小文本，就是Git LFS指针而非真实数据，应从HF归档恢复。第4节的文件大小检查通常能发现该问题。

所有 `runs/*/experiments`、训练目录和 `runs/evaluation/` 的非checkpoint文件都由第4节全量核对；包括 `run.json`、rollout、验证结果、TensorBoard、请求/响应和失败日志。不要只保留最终summary。

## 8. 只验收checkpoint20和checkpoint32

父训练根为 `$PROJECT/$TRAIN_REL`。父级 `run.json`通过HF包迁移，是旧训练配置和Reader/Judge协议的来源，不能只复制两个model目录。

两个checkpoint当前均是单个safetensors文件；没有 `model.safetensors.index.json` 是正常的：

| 相对各checkpoint目录的文件 | 源端基线/作用 |
|---|---|
| `trainer.pt` | 51,803,641,294 bytes；完整训练状态，当前评测也检查它存在 |
| `validation_metrics.json` | 对应验证结果 |
| `model/model.safetensors` | 9,078,624,992 bytes；训练后Reader权重 |
| `model/config.json`、`model/generation_config.json` | 模型及生成配置 |
| `model/tokenizer.json`、`model/tokenizer_config.json` | tokenizer及模板配置 |
| `model/chat_template.jinja` | 实际chat模板文件 |

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
base = Path(os.environ['PROJECT']) / os.environ['TRAIN_REL']
assert isinstance(json.loads((base / 'run.json').read_text()), dict)
for step in (20, 32):
    p = base / f'checkpoint-{step:08d}'
    assert (p / 'trainer.pt').stat().st_size == 51803641294
    assert (p / 'model/model.safetensors').stat().st_size == 9078624992
    for name in ['validation_metrics.json', 'model/config.json', 'model/generation_config.json',
                 'model/tokenizer.json', 'model/tokenizer_config.json', 'model/chat_template.jinja']:
        assert (p / name).is_file() and (p / name).stat().st_size > 0, str(p / name)
    print('PASS', p.name)
PY
```

这只证明文件齐全和大小吻合。源服务器仍可访问时，使用下面的只读rsync内容复核；会读取两端约113GiB数据，但不会传输修改：

```bash
# 在源服务器项目根执行；替换目标SSH及项目路径。
export TRAIN_REL='runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train'
rsync -aHnci -s --modify-window=-1 \
  "$TRAIN_REL/checkpoint-00000020" "$TRAIN_REL/checkpoint-00000032" \
  'USER@NEW_SERVER:/目标路径/AG-mem/runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train/'
```

退出码0且没有文件差异才通过。快速模型指纹另见 `environment_and_models.json` 中 `orm_checkpoint20/orm_checkpoint32`；它包含文件mtime，不是完整权重内容校验。不要用空的 `trainer.pt` 或重新下载的基础模型冒充这两个checkpoint。

## 9. 评测文件哪些必须有，哪些尚未生成

| 路径 | 检查原则 |
|---|---|
| `runs/evaluation/format05_best_final/` | 既有LoCoMo评测、重答、GPT请求/响应等，按归档清单全部保留 |
| `runs/evaluation/longmemeval_gpt5mini/` | 既有远端目录构建结果及来源记录，按清单保留 |
| `runs/evaluation/longmemeval_811_orm/jobs/` | 本次LME评测启动与错误日志 |
| `runs/evaluation/longmemeval_811_orm/results/` | 尚未完成的评测状态也应按原样恢复 |

本次源端核查时四组LME评测均0/50，仅 `best/split_longmemeval_s` 建立了空检索进度，其余组尚未创建；没有完整 `summary.csv`、`metrics.json`、`complete.json` 和 `offline/requests.json.gz` 并不代表迁移丢失。各文件按相应评测阶段生成，是否应存在以实际 `SOURCE_FILES.jsonl` 为准，不创建假完成标识。

## 10. 会话记录与交接内容

在新服务器的Codex用户目录检查：

```text
~/.codex/sessions/2026/11/02/
  rollout-2026-11-02T20-02-12-01a19c23-65d1-73a2-b52b-09c9b363e14e.jsonl
```

- [ ] 会话文件在结束源端对话之后复制，没有停留在更早版本；66MiB只是先前参考体积，不是固定大小。
- [ ] 用两端 `sha256sum` 核对整个会话文件相同，而不是只看文件名。
- [ ] `doc/migration.md` 的项目交接内容也已提交并迁移。
- [ ] 新服务器重新登录Codex；没有把 `auth.json` 当作项目数据上传。

安装兼容版本Codex后，可按迁移文档中的命令恢复会话。会话JSONL不代表所有后台任务、数据库记忆和工具连接已迁移。如果客户端无法恢复，先让新会话读取 `doc/migration.md` 和本清单，再按需查看原会话记录；不要把不能恢复UI直接理解为项目文件丢失。

## 11. 模型、环境和路径的运行检查

这部分在重新安装环境和下载基础模型后执行，不属于“环境目录或基础模型已经迁移”。

| 项目 | 通过标准 |
|---|---|
| Qwen3.5-27B、Qwen3.5-4B、Qwen3-Embedding-4B | 模型配置、tokenizer、完整权重存在；是所选配置需要的型号/版本 |
| Python/torch/transformers/vllm等 | 与实际源端版本记录核对，并在目标GPU验证兼容性 |
| `agmem` 导入路径 | 指向目标项目，而不是另一个旧clone或错误安装 |
| 配置的input/prepared/output/model路径 | 指向目标实际位置；不把新输出写进已冻结的旧库 |
| prompt与Judge模板 | 与选定实验记录一致；不同方案不混用 |

```bash
cd "$PROJECT"
python -m pip check
python - <<'PY'
import agmem
from importlib.metadata import version
print('agmem:', agmem.__file__)
for name in ['torch', 'transformers', 'vllm', 'accelerate', 'pyarrow', 'safetensors']:
    print(name, version(name))
PY
rg -n '/share/home/wtma|\.\./model' configs scripts
```

当前模型指纹包含 `mtime_ns`。重新下载同名模型仍可能导致旧实验身份检查失败；源码里的路径又有 `.resolve()` 绑定，普通软链接未必能兼容旧绝对路径。应先核实模型版本/内容及所需路径映射，再做显式迁移适配，不删除manifest、不伪造指纹，也不为了通过检查重建原目录。

## 12. 最后的程序预检查与验收记录

完成环境、模型和路径适配后，在项目根运行旧ORM评测的CPU预检查；不会启动GPU模型或重新生成摘要，但会使用评测锁文件：

```bash
cd "$PROJECT"
CUDA_VISIBLE_DEVICES='' python -u -m agmem.evaluate_checkpoints \
  --config configs/eval_longmemeval_811_orm_checkpoints.yaml \
  --stage prepare --task-agents local
```

通过标准：出现 `test_scoring_protocol_verified`；S/M `test_data_status` 都为ready且验证目录条数为2403/23784。若失败，先区分路径/重新下载模型的身份差异和缺文件，不直接重建数据。CPU prepare不代替checkpoint内容校验，也不验证GPU显存或推理。

如需继续父子目录构建，只用对应脚本 `--dry-run` 检查输入和规则，不把正常启动命令当作检查命令。正式训练或评测之前再由用户选择目标任务做小规模GPU验证；迁移验收本身不自动开始训练、补建或外部API调用。

最终记录建议保存在目标端独立的验收报告中：

- [ ] GitHub commit/HF revision/数据包SHA已记录。
- [ ] 完成标识、全部分片、解压文件清单检查通过。
- [ ] 所有新旧数据划分、原文、摘要、SQLite和来源清单完整。
- [ ] 只迁移checkpoint20/32，大小及内容复核通过，父级run.json存在。
- [ ] 旧未完成构建/评测状态与源端一致；未把未生成文件误判为丢失。
- [ ] 会话文件校验通过，交接文档可读取。
- [ ] 模型版本/路径适配和CPU预检查通过，或清楚标记“文件迁移完成，运行适配待完成”。
- [ ] 源端文件未删除；目标端尚未覆盖或重建旧实验数据。

给新服务器上的Codex可直接发送：

> 请先阅读 doc/migration.md 和 doc/migration_destination_checklist.md，按数据包清单逐项检查迁移结果并报告缺失/差异。只迁移了checkpoint20和32，旧checkpoint缺失正常；保留所有旧数据与实验记录。先检查文件，再处理路径和重新下载模型的身份兼容，不自动重建目录或启动训练评测。
