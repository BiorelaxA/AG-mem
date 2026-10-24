# 按数据集构建目录，选定数据集联合训练

首轮使用 `locomo`、`split_longmemeval_s`、`split_longmemeval_m`。入口配置为 `configs/pilot.yaml`，`data.layout: by_source`。各数据集分别清洗、保存历史、构建目录和续跑；训练时逻辑汇总选定来源的 train，验证时逻辑汇总相同来源的 validation。validation 不参与梯度更新，不合并到 train。

## 完整历史共享审计

2026-10-22 对已有有效 prepared 数据做全量核对，范围为全部 9 个来源的 train 和 validation。分别比较程序的完整 memory_id，以及排除原始 session ID、日期后按顺序比较的 session 内容 hash。后者仍保留 session 边界、顺序和清洗后的消息角色、姓名、正文。

- train 内部跨数据集的完整历史共享：0。
- validation 内部跨数据集的完整历史共享：0。
- train 与 validation 跨数据集的完整历史共享：0。
- 同一数据集内部多个 query 可以共享完整历史；同一来源的 train 与 validation 也存在历史共享，不能称为历史隔离验证。全部来源合计共享 35 条历史。
- LoCoMo 的 10 条 validation query 对应 7 条历史，这 7 条都出现在 train；LongMemEval s、m 各自 train/validation 的完整历史不重合。保留之前确定的数据划分，不重抽验证集。

审计结果见 `runs/audits/history_sharing_by_split_20261022.json`。这是对当前数据和上述精确比较口径的结论，不是对任意近似内容、未来数据版本或部分 session 重叠的保证。

| 来源 | train query | validation query | train+validation 独立历史 | session 总数 |
|---|---:|---:|---:|---:|
| locomo | 1124 | 10 | 10 | 272 |
| split_longmemeval_s | 340 | 10 | 350 | 16704 |
| split_longmemeval_m | 340 | 10 | 350 | 166371 |
| 合计 | 1804 | 30 | 710 | 183347 |

LongMemEval m 占本轮目录的大部分，缩小到这三个来源仍不意味着目录构建很快。可以先单独构建 LoCoMo 或 LongMemEval s，以较低成本检查目录设计。

## 独立存储

```text
runs/locomo_longmemeval/
  prepared/
    locomo/
      memory_records.jsonl
      train_examples.jsonl
      validation_examples.jsonl
      directories.jsonl
      completions.jsonl              # 首次 Writer 请求时创建
      manifest.json
      directory_manifest.json
    split_longmemeval_s/
      ...
    split_longmemeval_m/
      ...
    _runtime/                       # 联合训练的 Task Agent 响应缓存
  train/
```

一个完整历史对应一套目录，一个 session 对应一个目录项。共享完整历史的 query（包括同一来源中 train/validation 共享历史）复用该套目录，不是整条历史只有一个目录项。部分 session 重叠的不同完整历史独立建目录。

构建顺序按配置中的数据集顺序；每个数据集内部按 memory_id 排序，每条历史内部按 session 原始顺序。训练样本仍按既定 seed 和 epoch 打乱。调整目录窗口或提示词会产生新版本，各来源独立重建；训练要求所有选定来源使用当前 Writer 版本且所需历史的目录已完整发布。不要求其他未选中数据集完成目录。

## 迁移与运行

已有集中存储可离线迁移，不调用模型，不删除旧文件：

```bash
python -m agmem --config configs/pilot.yaml import-prepared runs/slurm_full_train/prepared
```

只复制配置选定来源的历史、query 和已保存的目录项，保留 memory_id、目录版本和完成快照。部分历史已生成的目录项也会保留。旧 Writer completion 缓存不整体复制，已落盘摘要通过目录项直接复用。重跑导入不会覆盖已经准备完成的目标数据集。源数据和目的数据均不应同时有写入作业。

本配置的服务地址为 `http://127.0.0.1:50543/v1`，与旧运行一致，用于保持 Writer 缓存身份。迁移后重新 build-directories 会核对当前模型、配置、prompt 版本，只复用相同版本的成功项。不要为复用摘要而伪造模型或 prompt 版本。

如果没有旧 prepared 数据，可以分别从原 parquet 准备：

```bash
python -m agmem --config configs/pilot.yaml prepare --sources locomo
python -m agmem --config configs/pilot.yaml prepare --sources split_longmemeval_s
python -m agmem --config configs/pilot.yaml prepare --sources split_longmemeval_m
```

启动配置指定的 27B 服务后，分别构建：

```bash
python -m agmem --config configs/pilot.yaml build-directories --sources locomo
python -m agmem --config configs/pilot.yaml build-directories --sources split_longmemeval_s
python -m agmem --config configs/pilot.yaml build-directories --sources split_longmemeval_m
```

省略 `--sources` 会按配置顺序构建这三个数据集。该参数仅用于 prepare/build-directories，不改变训练范围。`--limit-memories` 在此布局下是每个选定数据集的上限，适用于调试；未完成目录不能直接进入训练。不要并发写入同一个来源目录。

训练使用这三个来源的 1,804 条 train 和 30 条 validation：

```bash
CUDA_VISIBLE_DEVICES=0 python -m agmem --config configs/pilot.yaml train
```

若只想先训练 LoCoMo，将新的实验配置 data.sources 改为 `[locomo]`，沿用同一个 prepared_root，并使用新的 training.output_dir；无需重新复制其他数据。checkpoint 恢复时仍检查来源、split 和 Writer 版本，不能直接用旧实验 checkpoint 改换训练来源。

在 Slurm 上一次执行服务启动、准备、按来源构建和联合训练：

```bash
sbatch scripts/train_pilot_a6000.sbatch
```

新脚本申请 3 张 RTX A6000、120G 主存、48 小时，默认独立运行目录 `runs/locomo_longmemeval`。两张卡部署 27B，一张训练 4B。旧全量脚本保持原用途。首次运行保存冻结配置；后续修改实验范围应使用新的运行目录或显式管理快照，不能仅修改基础 YAML 后假设旧快照自动更新。


## 2026-10-23：本轮只构建和训练 LoCoMo + LongMemEval-S

`configs/pilot.yaml` 的 `data.sources` 是可用数据集注册列表，保留 M 以供以后选择。实际执行计划单独放在 `configs/pilot_workflow.yaml`：

```yaml
directory_sources: [locomo, split_longmemeval_s]
training_sources: [locomo, split_longmemeval_s]
experiment: locomo_s
```

`directory_sources` 决定本次 prepare/build-directories 的输入；`training_sources` 决定 train 与 validation 的输入，两个 split 不混合。两组来源可以不同，例如只补建 M，但训练选择三来源；所选训练来源的目录必须已经完整，训练前会校验。

默认一轮 1,464 个训练样本、20 个验证样本；group size=8、每个训练样本更新一次，因此一个 epoch 为 1,464 steps。LoCoMo 和 S 的目录合计 16,976 项，现有成功项复用。M 的 partial directories 原样保留，本轮不加载、不校验、不等待 M。

```bash
# 默认：补完 LoCoMo/S 的目录后立即进入对应 GRPO。
sbatch scripts/train_pilot_a6000.sbatch

# 两个阶段可拆成独立作业，后者要求 LoCoMo/S 目录已经完成。
AGMEM_STAGE=build sbatch scripts/train_pilot_a6000.sbatch
AGMEM_STAGE=train sbatch scripts/train_pilot_a6000.sbatch

# 以后只续建 M，不训练。
AGMEM_STAGE=build AGMEM_WORKFLOW_CONFIG=configs/pilot_workflow_m.yaml \
sbatch scripts/train_pilot_a6000.sbatch

# M 完成后，单独训练 M。
AGMEM_STAGE=train AGMEM_WORKFLOW_CONFIG=configs/pilot_workflow_m.yaml \
sbatch scripts/train_pilot_a6000.sbatch

# 以后补完三来源目录，并在新实验中训练三来源。
AGMEM_WORKFLOW_CONFIG=configs/pilot_workflow_all.yaml \
sbatch scripts/train_pilot_a6000.sbatch
```

以上都沿用既有 GPU、内存、时间和并发设置，不代表运行这些命令会自动取消旧任务。同一 RUN_DIR 的作业仍受 `.job.lock` 保护，不能同时写入。正在执行的旧 Python 和旧脚本不会自动切换本计划；需在原任务退出后重新启动新版脚本。不要同时提交 build 和 train 指望它们自动建立 Slurm 依赖。

持久数据仍是 `runs/locomo_longmemeval/prepared/<source>/`。原始 `config.yaml` 不必改写或删除。每次作业在 `jobs/<attempt>/` 保存 `workflow.yaml`、`build_config.yaml`、`train_config.yaml`，所以历史快照有 M 也不会导致默认继续构建 M。

训练输出按 experiment 隔离：

- `experiments/locomo_s/train/`：本轮 LoCoMo/S 的 checkpoint、TensorBoard、rollouts 和 `train.complete`。
- `experiments/longmemeval_m/train/`：未来 M-only 实验。
- `experiments/locomo_s_m/train/`：未来三来源实验。

同一 experiment 的重启可自动恢复自身的已完成 checkpoint。不同数据选择不能复用同一个 experiment 名称；改变训练来源需要新名称，不会把旧 optimizer/epoch 位置错误地当作新数据的续训位置。新实验默认从配置中的初始 Reader 模型开始；本次不实现从 LoCoMo/S checkpoint 自动迁移到 M 的第二阶段训练。

此 workflow 只用于 pilot 脚本；直接执行 `python -m agmem --config configs/pilot.yaml ...` 仍按该配置中的三来源运行。如需手动执行本轮某阶段，使用本次作业保存的 build_config.yaml 或 train_config.yaml。
