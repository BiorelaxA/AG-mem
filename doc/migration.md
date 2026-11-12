# 新服务器迁移方案与会话交接

更新日期：2026-11-12。当前迁移范围是：代码通过 GitHub，所有新旧数据、目录、轨迹和日志通过 Hugging Face；checkpoint20/32 通过 rsync 单独完整传输，其他 checkpoint 不迁移；基础模型在新服务器重新下载，环境重新安装。过时方案的非checkpoint代码与数据同样保留。本文下面的旧迁移说明仅作历史参考，不是本次执行入口。

## 迁移范围与实际体积

逐文件清单位于 `reports/migration_20261112/payload_sizes.json`。整个项目约 1.2TB，其中主要体积是自训练 checkpoint，并不是数据：

| 类别 | 当前统计 | 迁移处理 |
|---|---:|---|
| 原始及处理后数据、所有 prepared、旧单层及父子目录、缓存、轨迹、评测、日志、代码 | 约 28.42GiB | 全部保留，代码进 GitHub，数据包进 HF |
| checkpoint 配置及其他元数据 | 约 0.39GiB | 仅20/32随完整checkpoint传输，旧checkpoint不迁移 |
| 21 个 `trainer.pt` | 约 1013.37GiB | 仅20/32随完整checkpoint传输 |
| 21 个 safetensors 权重文件 | 约 177.56GiB | 仅20/32随完整checkpoint传输 |
| 项目外 `/share/home/wtma/model/` 基础模型 | 不打包 | 新服务器重新下载 |
| Conda 环境目录 | 不打包 | 新服务器重新安装，版本清单保留 |

用户已明确仅迁移checkpoint20/32，两套完整目录约113.44GiB，包含model、trainer.pt、配置及验证指标；旧checkpoint不打包、不传输，也不在源端删除。排除checkpoint后的非模型数据约28GiB，实际包清单以打包时扫描为准；上表其余部分还包括走GitHub的代码和.git。统计是本次只读扫描的快照，不代表后续运行不会新增文件。

打包脚本生产只读预览已通过：14,689个文件、30,256,696,723字节（约28.18GiB），排除21个checkpoint目录。7项小样本测试通过，包括真实分片/解压、SHA校验及纳秒mtime保留；尚未生成生产数据包。

## GitHub 保存代码

现有远端为 `https://github.com/BiorelaxA/AG-mem.git`。需要提交实际工作区中的所有新增和已修改源文件，不能仅在新服务器 clone 当前远端。保留旧代码和 Git 历史，不重写历史、不删除旧方案。

代码快照包含 `agmem/`、全部 `configs/`（本机凭证配置单独处理）、`scripts/`、`tests/`、`doc/`、README、依赖声明、`.gitignore`、`.gitattributes`、`method.png` 等设计文件。上传前检查 `git status --short` 和 `git diff --cached --stat`，确认目录 prompt 的资源文件及新脚本已加入。实验数据、SQLite、权重和完整会话记录不进入代码提交。不要直接使用未经核对的 `git add .` 把日志或本机凭证一起提交。

建议创建独立迁移分支/标签，并在数据包清单中记录最终 commit SHA。新服务器按该 SHA 检出，保证数据包与代码版本对应。当前尚未 commit 或 push 本次迁移内容。

源服务器执行（迁移分支首次创建；若已存在则切换到该分支）：

```bash
cd /share/home/wtma/AG-mem
git switch -c migration-20261112
git add -- agmem configs scripts tests doc README.md pyproject.toml \
  requirements.txt requirements-dev.txt requirements-serving.txt environment.yml \
  .gitignore .gitattributes method.png
git diff --cached --check
git diff --cached --stat
git commit -m "Preserve all directory methods and migration tools"
git push -u origin migration-20261112
git rev-parse HEAD
```

旧仓库已跟踪少量prepared的Git LFS文件；新服务器clone时用`GIT_LFS_SKIP_SMUDGE=1`跳过自动下载，由完整HF包覆盖恢复真实数据。

## Hugging Face 保存完整数据

建议新建 private dataset 仓库，例如 `YOUR_ACCOUNT/agmem-migration-20261112`，保存原始档案，不进行重新划分、去重、清洗或重建目录。打包范围包括：

- `data/` 全部版本，包括早期划分、LoCoMo 9:1、LongMemEval 8:1:1 的 Parquet、manifest、row mapping。
- `runs/` 全部实验：单层目录、父子目录、prepared、构建断点库、embedding 缓存、rollout、validation、TensorBoard、评测答案、离线 GPT 请求/回答、失败运行及日志。
- `reports/` 全部审计与分析结果，以及根目录 `.out`、`.err`、`.log` 等实验记录。
- 数据包排除所有数字编号的 `checkpoint-*` 目录及其 `.building` 临时目录；仅20/32通过rsync完整复制。checkpoint外的 `run.json`、训练轨迹、评测指标和日志仍完整打包。

脚本 `scripts/package_migration_data.py` 保留全部原相对路径，输出结构为：

```text
agmem-migration-20261112/
  ARCHIVE_MANIFEST.json
  SOURCE_FILES.jsonl
  SHA256SUMS
  COMPLETE.json
  archive.tar.zst.part-00000
  archive.tar.zst.part-00001
  ...
```

每个分卷最多4GiB，是同一个tar.zst流的连续分片，需要按文件名顺序拼接后解压。脚本生成源文件清单、分片SHA256和完成标识，不额外保留全量tar副本。它不会按`.pt`后缀笼统过滤数据；发现checkpoint之外未分类的权重/训练状态会报错，防止把模型意外混入HF数据包。`--dry-run`只扫描、不创建输出；实际打包只允许新的输出目录。若打包中断或检测到源文件变化，不生成完成标识，应使用新输出目录重新打包；上传阶段则可重复执行同一上传命令续传。

在最终打包前停止构建、训练和评测写入，保证 SQLite 与 JSONL/manifest 属于同一时刻。父子目录的 `build_journal.sqlite3` 是续建状态，`embedding_cache.sqlite3` 可复用计算；仅保存导出的目录 JSONL 不足以完整恢复。归档使用支持高精度时间的 pax 格式，保留修改时间，源文件不删除。打包目录位于项目外，避免将归档自身再次纳入归档。

GPU 节点不能联网时，在 GPU 节点完成停写/打包，通过共享盘让可联网登录节点上传；若共享盘配额不足，使用足够空间的本地临时盘并通过内网拷到登录节点。`/tmp` 不是跨节点共享位置，也不作为唯一长期副本。

可以在独立的上传工具环境安装 `huggingface_hub`，不改训练环境。以下命令在可联网登录节点执行；使用该节点本地/tmp存包时，上传必须在同一节点完成。先停止项目的构建、训练和评测写入。目录包含新旧数据，包尚未实际生成：

```bash
cd /share/home/wtma/AG-mem
export HF_REPO='YOUR_ACCOUNT/agmem-migration-20261112'
export PACK_DIR="/tmp/agmem-data-${USER}-$(date +%Y%m%d_%H%M%S)"
/share/home/wtma/miniconda3/envs/agmem/bin/python scripts/package_migration_data.py --output "$PACK_DIR" --dry-run
/share/home/wtma/miniconda3/envs/agmem/bin/python scripts/package_migration_data.py --output "$PACK_DIR"

python3 -m venv "$HOME/.venvs/agmem-transfer"
source "$HOME/.venvs/agmem-transfer/bin/activate"
python -m pip install -U huggingface_hub
hf auth login
python - <<'PY'
import os
from huggingface_hub import HfApi
# First creation only. Refuse to silently reuse a pre-existing public repo.
HfApi().create_repo(repo_id=os.environ['HF_REPO'], repo_type='dataset', private=True)
PY
test -f "$PACK_DIR/COMPLETE.json"
hf upload "$HF_REPO" "$PACK_DIR" --repo-type dataset
```

上传完成后记录 HF commit/revision；新服务器按该 revision 下载，校验 `SHA256SUMS` 后再恢复到 GitHub clone 的项目根目录。

```bash
hf auth login
hf download YOUR_ACCOUNT/agmem-migration-20261112 \
  --repo-type dataset --revision HF_COMMIT_SHA --local-dir /path/to/downloaded-packages
```

上传命令以安装版本的 `hf --help` 为准。官方上传说明支持大目录分批上传与重复执行续传；不要更新已经开始上传的快照包。断线续传只需重跑`hf upload`，不重建仓库或数据包。当前 HF 官方列出的免费 private 总额度为100GB，共享账户的其他仓库也占用此额度；本方案checkpoint走rsync，不占用该数据仓库。资料：[上传说明](https://huggingface.co/docs/huggingface_hub/en/guides/upload)、[存储额度](https://huggingface.co/docs/hub/storage-limits)。

新服务器先具备git、Python3、zstd和rsync，然后执行；项目路径尽量与旧服务器相同。`/path/to/downloaded-packages`必须在项目外。解压前检查SHA256与完成标识，连续分片不能单独解压：

```bash
export PROJECT='/share/home/wtma/AG-mem'
export HF_REPO='YOUR_ACCOUNT/agmem-migration-20261112'
export PACK_DIR='/path/to/downloaded-packages'
GIT_LFS_SKIP_SMUDGE=1 git clone --branch migration-20261112 \
  https://github.com/BiorelaxA/AG-mem.git "$PROJECT"
python3 -m venv "$HOME/.venvs/agmem-transfer"
source "$HOME/.venvs/agmem-transfer/bin/activate"
python -m pip install -U huggingface_hub
hf auth login
hf download "$HF_REPO" --repo-type dataset --local-dir "$PACK_DIR"
cd "$PACK_DIR"
test -f COMPLETE.json
sha256sum -c SHA256SUMS
set -o pipefail
cat archive.tar.zst.part-* | zstd -dc | tar --no-same-owner -xpf - -C "$PROJECT"
mkdir -p "$PROJECT/runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train"
mkdir -p "$HOME/.codex/sessions/2026/11/02"
```

实际执行时任何校验/解压报错都应停止后续步骤；在脚本中使用`set -euo pipefail`确保自动停止。数据仓库应为本次固定快照；需要严格固定下载版本时追加`--revision HF_COMMIT_SHA`。

## checkpoint20与32单独传输

先在新服务器完成Git clone和数据恢复，创建相同的训练目录。以下在源项目根目录执行；将目标SSH和项目路径替换为实际值。两个源checkpoint路径末尾不加斜杠，以保留两个目录名。

```bash
cd /share/home/wtma/AG-mem
export TARGET_SSH='USER@NEW_SERVER'
export TARGET_PROJECT='/share/home/wtma/AG-mem'
export TRAIN_REL='runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train'
rsync -aH -s --partial --modify-window=-1 --info=progress2 \
  "$TRAIN_REL/checkpoint-00000020" "$TRAIN_REL/checkpoint-00000032" \
  "$TARGET_SSH:$TARGET_PROJECT/$TRAIN_REL/"
```

两端rsync建议至少3.1.3，以支持纳秒mtime比较。中断后原命令重跑，源文件不会删除；不要加`--delete`。最终可将上述`-aH`改为`-aHnci`进行只读内容复核，无文件差异再继续使用。不会传输其余旧checkpoint。

## 当前对话的迁移

当前 Codex CLI 版本为 0.167.1，会话 ID 为：

```text
01a19c23-65d1-73a2-b52b-09c9b363e14e
```

本地原始记录（约 66.44MiB，后续对话会继续增长）：

```text
/share/home/wtma/.codex/sessions/2026/11/02/rollout-2026-11-02T20-02-12-01a19c23-65d1-73a2-b52b-09c9b363e14e.jsonl
```

结束当前会话后再复制这个文件到自己的电脑或新服务器，保留 `sessions/2026/11/02/` 相对层级。对话原文单独私下传输，不混入代码仓库或数据包，也无需转移 `auth.json`。新服务器安装兼容版本 Codex 并重新登录，在默认 `~/.codex/sessions/2026/11/02/` 放置记录后尝试：

```bash
# 源服务器，目标机上已创建对应的 ~/.codex/sessions/2026/11/02/。
rsync -aP -s \
  /share/home/wtma/.codex/sessions/2026/11/02/rollout-2026-11-02T20-02-12-01a19c23-65d1-73a2-b52b-09c9b363e14e.jsonl \
  "$TARGET_SSH:.codex/sessions/2026/11/02/"

# 新服务器，安装兼容版本Codex并登录后。
codex resume 01a19c23-65d1-73a2-b52b-09c9b363e14e -C /new/path/AG-mem
```

如按 ID 找不到，可使用 `codex resume --all --include-non-interactive` 检查可见性。会话 JSONL 保存对话，但不等同于所有 Codex 数据库状态、后台任务、外部工具连接或认证。不同客户端之间不能保证自动导入；至少可让新会话阅读下面的项目交接内容，并按需读取原记录。迁移整个 Codex home 的数据库时需停止全部使用它的客户端，避免复制正在变化的 SQLite；本方案仅迁移当前会话，不要求覆盖目标机已有 Codex 数据库。

官方资料：[会话文件位置](https://learn.chatgpt.com/docs/reference/troubleshooting)、[恢复命令](https://learn.chatgpt.com/docs/developer-commands?surface=cli)。

## 新服务器恢复与校验顺序

1. 检出代码快照，下载 HF 固定 revision，对全部分卷执行 SHA256 校验。
2. 将数据解压回项目相对路径，比较归档清单的文件数/大小；对 SQLite 副本执行完整性检查。
3. 重新安装环境。已记录实际环境：Python 3.11.17、torch 2.11.0+cu129、Transformers 5.5.3、vLLM 0.22.1+cu129。完整包版本在 `reports/migration_20261112/installed_packages.txt`，它是版本清单而非保证跨机器可直接安装的锁文件。CUDA wheel 和驱动需要在目标 GPU 上验证；根目录旧环境说明不能替代这份实际记录。
4. 重新下载 Qwen3.5-27B、Qwen3.5-4B、Qwen3-Embedding-4B 的对应模型版本，保存 tokenizer 和配置文件。
5. 处理路径与模型身份：父子构建 manifest 含 `.resolve()` 后的绝对数据路径；flat 来源清单及已保存 Reader retrieval 也有绝对路径。普通软链接未必兼容。尽量使用原实际路径或容器内相同路径；若必须换路径，进行显式路径映射和校验，不能批量替换 manifest 后绕过检查。
6. 重新下载模型会改变修改时间，现有快速指纹包含 `mtime_ns`，因此即使同一模型也可能触发旧评测协议不匹配。`environment_and_models.json` 保存了原快速指纹，但没有完整权重 SHA；需要核实精确权重版本/内容，再选择经过校验的身份迁移方案。不要为通过校验盲目篡改模型时间或旧 run.json。
7. 先做 CPU 检查和一题 GPU smoke，再续建或评测。自训练 checkpoint 未迁移时不能运行 checkpoint20/32 的评测。

## 给新会话的项目交接

以下是 2026-11-12 检查时的状态；接手后先看实际文件与最新日志，不将这段状态当作后续进度。

- 项目是目录引导的多步记忆检索。Reader 直接输出目录 ID；Writer 构建目录，Task 根据证据回答。旧单层 ORM、PRM、新父子目录 ORM 是并存方案，不能混用 prompt、目录或奖励配置。
- LongMemEval-S/M 当前统一数据根是 `data/processed/longmemeval_811_seed42_20261111`，各 train400/validation50/test50。原训练数据尽量保留，新测试集为原测试集子集；划分依据在 manifest/row_mapping 中。旧划分仍需归档。
- 旧 ORM 模型位于 `runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train`；best=checkpoint-00000020，final=checkpoint-00000032。Reader 使用保存的 `evidence_loop_v3` 协议，每次1ID、最多8次读取。
- 旧单层 LME 测试目录已全部完成：S50条历史/2403个session摘要，M50条历史/23784个摘要；根为 `runs/locomo_longmemeval/prepared_longmemeval_811_flat/test/`。S复用了1626条GPT摘要、777条Qwen补建；M23784条为Qwen。完整性和来源校验已通过。
- 旧 ORM LME 评测入口 `scripts/build_longmemeval_811_flat_qwen.sh --eval-only`；配置 `configs/eval_longmemeval_811_orm_checkpoints.yaml`。两卡Reader、一卡本地Qwen3.5-27B Task/Judge；Task为history_answer_v1完整answer标签，Judge为冻结的LongMemEval官方题型prompt，包含_abs特殊路由。输出 `runs/evaluation/longmemeval_811_orm/results/`。
- 首次评测在 20261112-041802-650268-1864847 作业退出，4组均0/50。根因：已关闭的 likelihood 仍记录旧Task prompt，导致评分协议比较失败。已在 `agmem/evaluate_checkpoints.py` 添加严格限定的关闭模块prompt元数据兼容及CPU提前检查；Judge/公式/其他协议仍严格检查。相关46测试和真实S/M CPU准备检查通过，未自动重新启动GPU评测。
- `scripts/eval_longmemeval_811_gpt.sh` 用于以后联网节点复用本地检索证据，让GPT-5-mini回答并评分；不再次运行Reader，本地和GPT结果分开存储。
- 父子目录结构：一个历史轨迹有一份父目录，每个session一项并对应自己的子目录；子目录项覆盖相邻连续blocks，dialogue block是完整message，不是句子。目录项有title/summary/entities，读取子项返回对应原文。
- 既有相邻聚类根 `runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1` 保留，不全量重建。旧高/低阈值0.78/0.55；新补充数据采用0.72/0.50和topic_event_v2灰区判断，围绕同一具体话题/事件合并。新补充父摘要400、子摘要300 tokens。现有complete/empty/fallback按原样保留，并记录版本来源。
- 增量LME-S父子目录入口 `scripts/build_longmemeval_811_hierarchical_qwen.sh`，配置 `configs/build_adjacent_qwen_longmemeval_811_incremental.yaml`。三卡：Embedding4B一张，Qwen27B Writer两张，每Writer64并发，总128。配置已实现不代表新库已构建完成，应查看目标库实际进度。
- 父子目录ORM配置 `configs/train_dapo_hierarchical_f1.yaml`：每次1ID、最多10次READ、最多展开2个子目录；reward=2F1-1-0.05×READ次数-0.5×format违规数。读取当前已展示子目录会format罚，读取已被淘汰的导航目录不罚format但照计cost；重复读取segment会罚。DAPO排除全F1=0组，也过滤无有效advantage组；epoch3、group_size8、mini_batch_groups8，验证每4step，主指标F1。
- PRM相关实现保留在 `process_rewards.py`、`process_advantages.py`，方案定义见 `method.png`，不要在迁移时重写或将它混入旧ORM评测。
- 当前迁移的明确要求：新旧代码和非checkpoint数据全部保留；基础模型和环境重新安装；GitHub代码、HF数据；仅checkpoint20/32通过rsync完整传输，旧checkpoint不迁移、不删除源文件。没有执行远端上传、Git push、数据删除或生产数据归档。

可在新会话直接发送：

> 请先阅读 doc/migration.md 中的当前迁移方案和项目交接、reports/migration_20261112/ 的清单，再检查目标服务器文件。保留全部新旧方案与数据，不擅自重建目录或改变reward。首先完成迁移完整性、路径和模型身份检查，再按我要求继续构建或评测。

---

# 历史迁移说明 仅保留作旧方案记录

以下内容写于更早版本，其中“没有有效Git元数据”“不读取SQLite”“默认重建目录”等判断不适用于当前工程；本次迁移按上文执行。

# Git 提交与换机器运行

代码和小型复现清单通过 Git 同步，大数据与本地模型单独同步。新的 clone 不自带 prepared JSONL 原文和目录。当前代码使用完整轨迹 JSONL 存储，旧 SQLite 产物不再作为训练或检索输入。

## 当前忽略规则

| 文件 | 通过 Git 保存 | 原因/后续操作 |
|---|---|---|
| `agmem/`、`tests/`、`doc/`、README | 是 | 实现、验证和运行说明 |
| `configs/default.yaml`、`smoke.yaml`、`local.example.yaml` | 是 | 公共配置与目标机器路径模板 |
| Conda YAML、requirements、pyproject | 是 | 新环境安装入口 |
| `data/processed/*/manifest.json`、`*_manifest.jsonl`、CSV 和文字说明 | 是 | 固定划分成员、来源行号与 SHA-256；其中训练清单约 5MB |
| `data/**/*.parquet` | 否 | 原始与处理后的数据体积大，单独传输 |
| `data/audits/`、`data/prepared/`、旧 SQLite 文件 | 否 | 本地审计产物、JSONL 历史/目录/缓存和旧派生存储 |
| 模型权重、`models/`、`checkpoints/`、`runs/` | 否 | 大型本地资源及实验输出 |
| `.env`、`.aws/`、本地环境/缓存、`configs/local*.yaml` | 否 | 本机运行配置；示例配置例外保留 |

`.gitignore` 对已经跟踪的文件不追溯生效。在实际 Git 仓库中先检查：

```bash
git status --short
git ls-files -- '*.parquet' '*.sqlite' '*.sqlite3' '*.db' '*.safetensors' '*.pt' '*.pth' '*.ckpt'
git ls-files -- data/audits data/prepared runs .env .aws .codex
```

若其中已有应排除的文件，根据输出逐项执行 `git rm --cached -- 文件路径`，再提交移除记录；这会保留工作区文件。对已跟踪目录可使用 `git rm -r --cached -- 目录路径`。不要不经检查移除全部索引；如果大型文件已经存在于较早提交中，后续忽略或普通删除提交也不会移除旧提交里的文件。

本次工作区可见的 `.git/` 没有有效仓库元数据，`git status` 返回 not a git repository，因此这里只验证忽略规则，没有检查既有索引/提交历史，也没有进行 add、commit 或 push。

## 推荐迁移流程

1. 在目标机器 clone 仓库，并按 [环境说明](environment_setup.md) 创建新 Conda 环境。
2. 从当前机器单独同步当前固定版本的处理后数据目录，约 1.9GB。只需该目录，无须为运行训练复制全部原始数据和旧审计目录。

在源机器的项目根目录执行下例，将用户名、主机和目标项目绝对路径替换为实际值：

```bash
rsync -aP -R data/processed/query_dedup_val10_seed42_20261018/ \
  user@target-host:/absolute/path/to/AG-Mem/
```

`-R` 保留相对路径中的 `data/processed/...`。同步完整版本目录，保持 manifest 和 Parquet 成套。目标机器应得到 9 个来源各自的 train、validation、test Parquet；`prepare` 会检查其 SHA-256。当前 `prepare` 是读取既有划分做适配，不是从原始数据重新生成本次 train/validation/test 划分。

3. 单独准备完整的 Qwen3.5-4B / Qwen3.5-27B 本地模型目录，包括配置、tokenizer 和权重文件。
4. 在目标机器项目根目录创建本机配置：

```bash
cp configs/local.example.yaml configs/local.yaml
```

编辑其中的两个模型路径；若 27B 服务在其他地址，同步修改 `service.base_url`。此文件已被忽略，不会覆盖其他机器的配置。

5. 在目标机器重新检查并适配数据：

```bash
python -m agmem --config configs/local.yaml inspect-data
python -m agmem --config configs/local.yaml prepare
```

6. 按根目录 README 使用目标机器实际模型路径启动 27B 服务，再构建目录和运行训练：

```bash
python -m agmem --config configs/local.yaml build-directories
CUDA_VISIBLE_DEVICES=0 python -m agmem --config configs/local.yaml train
```

正式训练前仍应完成 README 中的单组真实模型 smoke 验证；两张 80GB 卡、依赖和实际显存链路尚未实测。

## 若直接迁移现成 JSONL

`data/prepared/session_jsonl_v2/` 可以单独传输，从而跳过 `prepare`；smoke 对应 `data/prepared/smoke_session_jsonl_v2/`。在构建进程停止写入时同步整个目录，保持 `memory_records.jsonl`、各 split 的 examples、`directories.jsonl`、`manifest.json` 和 `directory_manifest.json` 一致。`*.index.json` 为可重建偏移索引，目标机器缺失或失效时会从 JSONL 重建。prepared 目录不通过 Git 传输。

旧 `session_v1/artifacts.sqlite` 与 `smoke_session_v1/artifacts.sqlite` 保留供核查；当前代码不读取它们。切换存储格式后，从已固定的 processed Parquet 重新执行 `prepare`、`build-directories`，不修改原始数据或重新划分 train/validation/test。

默认建议在目标机器重建目录：模型快速指纹包含文件 mtime，服务地址也参与目录版本；模型复制方式或地址改变时，旧 Writer 缓存可能不匹配，训练会要求重新构建。若还要恢复已有训练，`runs/` 中 checkpoint 和优化器状态同样需要另外传输并满足恢复时的版本/配置校验。
