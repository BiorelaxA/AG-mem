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
