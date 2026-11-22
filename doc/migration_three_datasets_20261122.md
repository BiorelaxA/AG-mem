# AG-mem 跨服务器迁移与目标机验收（2026-11-22）

这次迁移以 gpu27 实际工作区为准。GitHub 保存代码，私有 Hugging Face dataset 保存 LoCoMo、LongMemEval-M 和 LongMemEval-S 的数据与构建状态，SSH 单独传输 Codex 会话。三者都可以独立验收。以下 Bash 命令均在注明的机器执行；将目标机 SSH 地址替换成自己的地址。

## 1. 源端交付清单

| 内容 | 源端位置或标识 |
|---|---|
| 已准备的独立代码仓库 | `/data/home/wtma/AG-mem/runs/code_sync/project-three-20261122` |
| GitHub 目标分支 | `BiorelaxA/AG-mem` 的 `migration-three-datasets-20261122` |
| HF 文件夹 | `/tmp/agmem-three-datasets-20261122`（gpu27 本机 /tmp） |
| 建议的 HF 私有仓库 | `hongyshen/agmem-three-datasets-20261122` |
| 会话快照 | `/tmp/agmem-private-sessions-20261122`，含本项目的 123 个 rollout（含项目子会话） |
| 当前主会话 ID | `01a19c23-65d1-73a2-b52b-09c9b363e14e`，源端 Codex CLI 0.169.0 |

代码快照从工作区收集 `agmem/`、`configs/`、`scripts/`、`tests/`、提示词与 schema、`doc/`、依赖声明、`.wheels/` 和根目录设计图。散落在 `reports/`、`runs/analysis/` 的分析脚本也按原路径放入 Git。快照有独立的提交历史，避免通过旧 Git 历史带入其他数据集的已跟踪文件或 LFS 指针。

HF 主包包含当前及历史划分下三个目标数据集的 Parquet、划分映射、旧单层 prepared、训练的冻结数据、12 个父子目录构建/修复目录、各自的 SQLite、缓存、已有导出和恢复所需的数据库旁文件。还保留项目中 `runs/locomo*` 下的相关实验记录及这三套数据对应的评测目录。没有可见的独立上游原始数据目录时，以本机实际存在的 Parquet 为准。最早的混合划分保留原版 manifest 以维持下游 SHA256 引用；其中仍列出其他数据集名称和统计。样本 Parquet 和该划分的逐行 JSONL 仅选这三个来源。较早混合划分的非必要跨库审计附件不在范围内。

附包 `analysis/` 保存 LoCoMo joint 划分、`runs/analysis` 以及相关历史分析报告。已排除机器迁移盘点报告、历史打包文件、checkpoint 权重、优化器、基础模型和环境，以及用户级凭据。若需要恢复 step 12 **训练**，另见第 6 节。

## 2. 在 gpu27 上传

```bash
set -e
CODE=/data/home/wtma/AG-mem/runs/code_sync/project-three-20261122
git -C "$CODE" show -s --format='%H %s'
git -C "$CODE" push -u origin migration-three-datasets-20261122
```

保存显示的 Git commit。若目的机从头安装，代码分支已经包含检查工具。如要重新创建一个更新的代码快照，可以在项目根目录运行 `bash scripts/prepare_three_dataset_code.sh /新的空目录`，为新快照选择新的远端分支名，不要强推覆盖这次快照。

```bash
(
  set -euo pipefail
  export HF_ENDPOINT=https://huggingface.co
  unset HF_HUB_OFFLINE
  repo=hongyshen/agmem-three-datasets-20261122
  cd /tmp/agmem-three-datasets-20261122
  test -s SHA256SUMS && test -s analysis/SHA256SUMS || exit 1
  sha256sum -c SHA256SUMS --quiet
  (cd analysis && sha256sum -c SHA256SUMS --quiet)
  test -s DIRECTORY_STATES.json
  hf auth whoami || hf auth login
  hf repos create "$repo" --repo-type dataset --private
  hf upload "$repo" . --repo-type dataset
)
```

第一次创建私有仓库。若已创建且确认其为私有，跳过 create 行，直接上传即可。记录上传返回的 commit SHA；有些版本仅打印分支 URL，可在 HF 的 Files and versions → History 查看。本机 /tmp 在确认目标端验收完成前需要保留。

如需重打包，先暂停会写入所选数据、prepared 和目录库的任务，在新的空目录执行下面命令。打包使用只读源文件、构建锁、文件变化检查和 4 GiB 分卷；残留 SQLite 回滚日志与数据库一起保存。包生成完成的标志是 `SHA256SUMS`。请在执行任务的同一台节点读取 /tmp。

```bash
cd /data/home/wtma/AG-mem
export NEW_PACK=/tmp/agmem-three-datasets-new-snapshot
python scripts/migrate_three_datasets.py pack --package "$NEW_PACK" &&
python scripts/migrate_three_datasets.py pack --supplement --package "$NEW_PACK/analysis" &&
conda run -n agmem python scripts/check_three_dataset_directories.py --save "$NEW_PACK/DIRECTORY_STATES.json"
```

## 3. 在目标服务器恢复代码、数据和目录

使用**新目录**，避免覆盖目标机已经构建的目录。下载盘需要同时容纳压缩包和约 116 GiB 解压文件。环境变量在同一终端设置。

```bash
export PROJECT="$HOME/AG-mem-three"
export PACK_DIR="$HOME/agmem-three-download"
export HF_REPO=hongyshen/agmem-three-datasets-20261122
export HF_REV='替换成这次HF上传的commit_SHA'
export EXPECTED_CODE='替换成源端git显示的commit_SHA'
export HF_ENDPOINT=https://huggingface.co
unset HF_HUB_OFFLINE
git clone --single-branch --branch migration-three-datasets-20261122 \
  https://github.com/BiorelaxA/AG-mem.git "$PROJECT"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$EXPECTED_CODE"
hf auth whoami || hf auth login
hf download "$HF_REPO" --repo-type dataset --revision "$HF_REV" --local-dir "$PACK_DIR"
```

在 Bash 运行。分卷属于同一个压缩流，按文件名排序后拼接。两个包分别校验、解压。以下过程若被中断，保留已构建的目标目录，优先在新的空白工程目录重新恢复。

```bash
(
  set -euo pipefail
  export LC_ALL=C
  for bundle in "$PACK_DIR" "$PACK_DIR/analysis"; do
    ( cd "$bundle"
      test -s COMPLETE.json
      sha256sum -c SHA256SUMS --quiet
      mapfile -t parts < <(python3 -c 'import json; print(*json.load(open("COMPLETE.json"))["parts"],sep="\n")')
      cat "${parts[@]}" | zstd -dc | tar --no-same-owner --keep-old-files -xpf - -C "$PROJECT"
    )
    python3 "$PROJECT/scripts/migrate_three_datasets.py" verify --project "$PROJECT" --package "$bundle"
  done
)
```

**先验收再运行训练或构建**。verify 会逐文件比对 SHA256、大小及纳秒 mtime，包括数据库旁文件；两行 PASS 才表示两个包均还原完整。发现有 SQLite 热回滚日志时，先完成字节校验，再在目标独立副本上用 SQLite 正常恢复并做 quick_check，勿手工删日志。具体续跑入口还会校验目录构建协议、绝对路径、模型/分词器指纹。数据通过 SHA256 验收并不代表这些运行时约束已处理；迁移的 DeepSeek 特定工作流可参照项目的 portable 入口与原先专用的迁移清单。

## 4. 验收项目数据和目录

- [ ] `git rev-parse HEAD` 与源端记录相同；`git ls-files agmem configs scripts tests doc` 能列出代码、脚本和提示词。版本化子目录提示词在 `agmem/hierarchical/resources/adjacent/` 下。
- [ ] 主包和 analysis 附包的文件内容校验均通过，`FILES.json` 可按相对路径检索每个目标文件。
- [ ] 最新 LoCoMo 查询划分为 train 1019 / validation 115 / test 400。LongMemEval-S 和 M 各为 315 / 35 / 150。历史 8:1:1 和其他早期划分分别保留在各自目录。
- [ ] `runs/locomo_longmemeval/prepared*` 与 `runs/evaluation/*/prepared_test` 包含旧单层目录、对应原文和索引。混合 ORM 的冻结库位于 `runs/locomo_lmes_context3_open1_nav06/prepared`。
- [ ] `runs/parent_child_directories` 的 12 个构建/修复根目录均存在。LoCoMo 的较新目录是 `qwen27b_locomo_topic_event_context3_v1`；S 的 `qwen27b_adjacent_qwen4b_31535150_incremental_v2` 需连同 fallback 和 pending 修复层恢复。
- [ ] M 的早期 DeepSeek、v3、训练 200–315 分片及本地 Qwen 验证/测试分片各保留独立的 `build_journal.sqlite3`、`run_manifest.json`。未完成任务、fallback 和历史版本会同时出现；不同版本不可直接相加或覆盖。使用下面的比较命令核对源端记录的**每个库**状态。
- [ ] 环境/模型按目标实际情况配置，运行所选任务的 CPU dry-run 或 check-only 后再分配 GPU。需要 API 时请在目标端设置新的环境变量。

```bash
cd "$PROJECT"
conda run -n agmem python scripts/check_three_dataset_directories.py \
  --project "$PROJECT" --compare "$PACK_DIR/DIRECTORY_STATES.json"
```

## 5. Codex 项目会话走私有 SSH

实际项目在旧、新挂载路径下都有会话。导出器按会话的工作目录筛选并保留原 `sessions/YYYY/MM/DD` 或 `archived_sessions` 布局，仅收录完整 JSONL 行。已准备当时的快照；若要包含本轮最终回复或其后的对话，在源机**退出这次会话后**重新导出。

```bash
# gpu27 单独的终端，在项目根目录执行
cd /data/home/wtma/AG-mem
umask 077
export SESSION_PACK=$(mktemp -d /tmp/agmem-project-sessions.XXXXXXXX)
python3 scripts/export_project_codex_sessions.py \
  --also-cwd /share/home/wtma/AG-mem --output "$SESSION_PACK"

export TARGET_SSH='hongyshen@你的目标服务器地址'
ssh "$TARGET_SSH" 'umask 077; mkdir -p "$HOME/agmem-private-session-import"'
rsync -a --partial --info=progress2 "$SESSION_PACK/" "$TARGET_SSH:agmem-private-session-import/"
```

目标机在目标用户下执行（若有自定义 `CODEX_HOME`，命令会使用该目录）。为避免覆盖目标端同 ID 的更新记录，先解压到私有临时目录，再按原层级导入缺失的文件。

```bash
(
  set -euo pipefail
  umask 077
  cd "$HOME/agmem-private-session-import"
  sha256sum -c SHA256SUMS
  staging=$(mktemp -d)
  tar -xzf project-sessions.tar.gz --no-same-owner -C "$staging"
  (cd "$staging" && sha256sum -c SESSION_SHA256SUMS --quiet)
  dest=${CODEX_HOME:-"$HOME/.codex"}
  mkdir -p "$dest"
  rsync -a --ignore-existing --exclude SESSION_SHA256SUMS "$staging/" "$dest/"
  (cd "$dest" && sha256sum -c "$staging/SESSION_SHA256SUMS" --quiet)
)
codex resume 01a19c23-65d1-73a2-b52b-09c9b363e14e -C "$PROJECT"
```

最后一次校验若报同 ID 不同哈希，保留两份并比较新旧记录再决定采用哪份。若当前版本无法发现恢复记录，使用 `codex resume --all` 检查，并可直接在私有 JSONL 中查看历史。不同 Codex 版本的会话索引或登录信息需要目标端自行建立。工具执行记录可能含私密数据，请仅通过自己的 SSH 连接传送，不放入 GitHub/HF。官方说明见 [Codex CLI](https://developers.openai.com/codex/cli/features/)。

## 6. 如果迁移目的还包括接续 ORM 训练

Checkpoint 12 在本机是经过验证的完整恢复点，约 56.72 GiB。本次数据包按数据集和目录打包，未包含该模型和优化器。需要在目标继续训练时，在对应训练已停止的状态下从源机另外传（先使用本次 HF 包恢复其训练记录）：

```bash
cd /data/home/wtma/AG-mem
train=runs/locomo_lmes_full_directory_orm/train
ssh "$TARGET_SSH" 'mkdir -p "$HOME/AG-mem-three/runs/locomo_lmes_full_directory_orm/train"'
rsync -aH --partial --info=progress2 "$train/checkpoint-00000012" \
  "$TARGET_SSH:AG-mem-three/$train/"
```

在首次 resume 前先核实目标端的模型、训练配置、数据目录与协议身份。早期迁移文档中的 step 20/32 在这台机器上未确认有完整可用权重；如需对应推理模型，请从有完整原件的服务器另行复制。
