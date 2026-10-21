# AG-Mem

目录引导的多步记忆检索与 Reader 全参 GRPO。Reader 使用本地 Qwen3.5-4B；冻结的 Qwen3.5-27B 共用一个服务，分别承担离线 Writer 和 Task Agent。

实现遵循当前约定：每个原始 session 一个目录项；Writer 看到当前 session 和前 5 个原始 session，只概括当前 session。Reader 每次选择 1–10 个 ID，读取整份 session 原文，最多 8 次调用并自主停止。一个 GRPO group 对同一问题采样 8 条完整轨迹。

```text
reward = outcome_f1 - 0.01 * logical_read_calls - 0.1 * format_failure
choice outcome: option-label F1, correct=1 / wrong or invalid=0
open QA outcome: deterministic answer-text F1
format_failure: 1 if any Reader format/protocol failure occurs, otherwise 0
```

每条完整 rollout 最多扣一次格式惩罚，合法动作不加格式分。JSON/schema 错误、`stop=true` 但 IDs 非空，以及继续检索时空批次、超过 10 个或同批重复 ID，均计格式失败；未知 ID、跨批重复与容量拒绝保持原有成本处理。系数由 `reward.format_penalty` 配置，默认 0.1（smoke 继承），设为 0 可消融。训练和评估共用评分；历史日志不会重新计分。详细边界见 [格式奖励设计](doc/format_reward_20261020.md)。

## 环境

在项目根目录执行，环境安装步骤与 CUDA 版本说明见 [环境配置](doc/environment_setup.md)。本次没有修改当前环境，也没有安装新的 GPU 软件。

```bash
conda env create -f environment.yml
conda activate agmem
uv pip install -r requirements-serving.txt --torch-backend=auto
python -m pip install --no-deps -e .
python -m pip check
```

依赖是有明确版本约束的候选组合，尚未在分配的两张 80GB GPU 上验证。先确认实际驱动支持对应 CUDA wheel，再进行模型与单组梯度检查；依赖安装成功不等于全参训练显存检查已通过。

换机器运行见 [Git 提交与迁移说明](doc/migration.md)。仓库保存代码、配置和固定划分清单；Parquet 数据、prepared JSONL、模型权重和运行输出不随 Git 提交，需要单独同步或重建。

## 数据

默认入口：[configs/default.yaml](configs/default.yaml)。输入为 `data/processed/query_dedup_val10_seed42_20261018/<source>/{train,validation,test}.parquet`：12244 条 train、90 条 validation、3100 条 test。按来源适配后全局 shuffle，不额外按来源均衡，也不重新去重或划分。

```bash
python -m agmem inspect-data
python -m agmem prepare
```

`prepare` 验证每个输入 Parquet 的 SHA-256，清洗完整历史并保留 session ID、顺序和空 session 占位，写入 `data/prepared/session_jsonl_v2/`：

```text
memory_records.jsonl       # 每行一条完整清洗后历史，包含有序 records/messages
train_examples.jsonl       # 每条 query 引用其 memory_id，包含私有参考答案
validation_examples.jsonl
test_examples.jsonl
manifest.json              # 数据来源、准备状态和存储格式版本
preparation_report.json
```

仅当清洗后的完整有序历史及 session ID、时间等元数据一致时，多个 query 才复用同一 `memory_id` 和目录。只有部分 session 相同、顺序变化或追加 session 的轨迹，各自保存完整原文并分别生成目录。模型输入继续通过显式字段白名单构造。

存储已由 SQLite 改为 JSONL。旧的 `data/prepared/session_v1/` 和 `smoke_session_v1/` 保留，不再被默认配置读取；更新后先重新 `prepare`。当前已验证新 smoke 目录包含 3 条完整历史、78 个 session、2 条 train 和 2 条 validation；全量新格式数据通过上面的命令生成。格式和验证细节见 [JSONL 存储说明](doc/jsonl_storage_20261019.md)。

“有效”在此指结构、字段和参考标签校验通过。后续补做的编码检查仍在历史正文中检出 28 个可疑 session 消息体，涉及 train 113 行、validation 6 行、test 55 行，均来自 LongMemEval s/m；题目、选项与答案字段未检出同类编码异常。这些候选尚未清理，不代表已完成全部内容质量审查，详见 [历史文本编码检查](doc/prepared_text_integrity_audit_20261018.md)。

原始数据与划分文件不变。`preparation_report.json` 列出有效数量、异常样本和处理原因。已知 ZH4O 非法参考标签进入隔离报告；答案字段适配审计确认有效 train 为 **12243 条**、validation 90 条、test 3100 条；完整历史结构的最终检查以 preparation report 为准。PersonaMem-v2 两条记录中的 null 错误选项填充被忽略并记录，样本保留。其选择题的所有选项按稳定样本 ID 确定性打乱，正确标签仅传给 scorer。

PerLTQA 历史中的空 session 占位也保留原始位置，并各自有目录项，摘要明确写为 `Empty session; no recorded messages.`，不复制前文事实、不删除对应 QA。空 session 的确定性摘要不消耗 Writer 生成调用。

## 启动冻结的 27B 服务

在分配的双卡节点上，以下使用第 2 张卡部署 27B；Reader 进程使用第 1 张卡。两个进程可以使用同一个 Conda 环境。

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve /home/share/models/Qwen3.5-27B \
  --served-model-name agmem-27b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 65536 --gpu-memory-utilization 0.90
```

调用端每个请求显式关闭 thinking；Writer 和 Task Agent 使用不同 prompt，共用冻结权重。`service.base_url` 默认 `http://127.0.0.1:8000/v1`，模型服务名称必须对应配置。若修改服务上下文，同步 `task_agent.max_context_tokens`。

## 构建目录

```bash
python -m agmem build-directories
```

Writer 输入中没有当前 query、选项标签或 evidence 标注。Writer 摘要和响应缓存均按完整历史 ID、模型指纹、prompt 和生成配置隔离，只有完整历史一致的 query 共享目录。不同轨迹即使当前 session 与前文窗口相同，也分别请求 Writer 生成摘要。

`content` 使用事实短句，以分号分隔，明确主体、动作/状态及结果；优先保留具有辨识度的实体、时间、数量、偏好和约束。在预算内覆盖多个独立话题，优先压缩寒暄和重复；区分事件日期与 session 日期，无可靠依据时保留相对时间，并保留计划/进行/完成/取消等状态及变化。所有摘要事实必须来自当前 session，前 5 个 session 仅辅助有依据的指代消解；保留人物归属、否定和不确定性，不把提问或假设写成既定事实。这些内容要求由提示词引导，格式和长度校验本身不判断事实准确性。

目录写入 `data/prepared/session_jsonl_v2/directories.jsonl`，每行一个 session 目录项，包含 `memory_id`、`version`、`id`、`session_id`、`order` 和最多 200 tokens 的 `content`。Reader 根据当前问题的 `memory_id` 加载完整目录，按选中 ID 从 `memory_records.jsonl` 中该轨迹的有序 sessions 定位原文。训练与检索直接读取这些 JSONL 文件。

每完成一项就追加保存，`directory_manifest.json` 标记完整生成并校验过的目录；中断后重跑可继续同一历史尚未完成的部分。`*.index.json` 仅保存可重建的字节偏移索引，`completions.jsonl` 保存模型响应缓存。

`writer.content_max_tokens: 200` 限制摘要正文，`writer.max_output_tokens: 384` 为包含 JSON 结构与转义字符的完整输出预留空间。首次 Writer 提示明确要求仅输出一个 JSON 对象，只有 `content` 键，值为非空字符串；禁止 Markdown 围栏、附加字段、注释和对象之外的文字。正文上限为 200 tokens，无额外较低目标，也无需凑满。校验失败时反馈上一次输出与实际错误：格式错误要求修复格式，超长错误要求保留关键事实并压缩至同一上限；重试不再递减目标长度。提示词不等于约束解码，仍保留实际 JSON 解析与 tokenizer 长度检查。重试耗尽时报告实际 token 数与上限。提示和格式重试协议变更会更新目录版本；仅提示变更时重新运行 `build-directories`，存储格式变更则先重新 `prepare`。

每个 session 只建一个目录项，不进行 1024-token 切分、层级目录或 embedding 预筛选。按当前决定，超长目录的结构性方案尚未实现；达到模型/服务容量时显式报错，不静默截断或删除数据。

## 先验证一组真实模型训练

小配置只取 LoCoMo，每个 split 先取两条，使用独立的 prepared/output 目录。模型服务启动后执行：

```bash
python -m agmem --config configs/smoke.yaml prepare --limit-per-source 2
python -m agmem --config configs/smoke.yaml build-directories
CUDA_VISIBLE_DEVICES=0 python -m agmem --config configs/smoke.yaml train --max-groups 1
```

该命令会真实采样一组 8 条完整轨迹、调用 Task Agent、执行 full-parameter backward / optimizer step，并保存 checkpoint。当前工作环境未运行此 GPU 命令；不要将下面 CPU 单元测试结果当成真实模型训练通过。

### 使用 Slurm 提交 smoke

在项目根目录提交 [scripts/smoke.sbatch](scripts/smoke.sbatch)：

```bash
sbatch scripts/smoke.sbatch
# 集群要求指定分区或账户时：
sbatch --partition=你的分区 --account=你的账户 scripts/smoke.sbatch
```

脚本申请同一节点的 3 张 `a100-pcie-40gb`、120G 主机内存、16 个 CPU，最长运行 10 小时。GPU 类型名称必须与集群 `sinfo -o '%P %G'` 中的 GRES 名称一致；如不同，在提交时用 `--gres=gpu:实际类型:3` 覆盖。脚本保留 Slurm 分配的 GPU ID：第一张训练 4B，另外两张以 TP=2 部署 27B，服务上下文 65536、BF16、关闭视觉分支。

默认激活 `/home/hongyshen/miniconda3` 下的 `agmem` 环境；需要调整时通过提交环境变量 `AGMEM_CONDA_SH`（conda.sh 路径）、`AGMEM_CONDA_ENV`（环境名称或路径）和 `AGMEM_PROJECT_DIR`（项目根目录）覆盖。依赖应提前安装，包括 TensorBoard；脚本不会自动安装环境。

每次作业在 `runs/slurm_smoke_<job_id>/` 独立准备 LoCoMo train/validation 各 2 条，使用最新 Writer 提示词构建 78 个 session 目录项，再训练 1 个 GRPO group（8 条完整轨迹）；不自动运行 validation 评测。它等待 27B 服务就绪后才执行后续步骤，退出时清理服务及 worker。`config.yaml` 保存本次有效配置；`job.log`、`vllm.log`、`prepare.log`、`directories.log`、`train.log` 保存各阶段输出；训练结果和 TensorBoard 事件位于 `train/`。Slurm 本身的输出在提交目录的 `agmem-smoke-<job_id>.out/.err`。重新排队同一作业 ID 时拒绝复用输出目录，重新提交将获得新的作业 ID。

### 使用 Slurm 提交全量训练

在项目根目录提交 [scripts/train.sbatch](scripts/train.sbatch)：

```bash
sbatch scripts/train.sbatch
# 集群要求指定分区或账户时：
sbatch --partition=你的分区 --account=你的账户 scripts/train.sbatch
```

申请资源与 `scripts/smoke.sbatch` 一致：同一节点 3 张 `a100-pcie-40gb`、120G 主机内存、16 个 CPU、10 小时。GPU 分工和 27B 服务参数也一致。脚本读取 [configs/full.yaml](configs/full.yaml)，准备全部 9 个来源的 train 与 validation，构建全部对应 session 目录，然后对合并后的 train 运行完整 1 epoch；每组采样 8 条轨迹，没有 smoke 的样本数或 group 数限制。validation 每 100 次更新评估一次；本脚本不准备 test。

默认持久目录为 `runs/slurm_full_train/`，其中 `prepared/` 保存清洗历史、Writer 缓存和目录，`train/` 保存模型、rollout、指标和 TensorBoard，`jobs/<job_id>.<随机后缀>/` 保存该次提交的 `job.log`、`vllm.log`、`prepare.log`、`directories.log`、`train.log`。Slurm 输出为提交目录中的 `agmem-train-<job_id>.out/.err`。首次提交保存有效配置至 `config.yaml`，后续提交使用该快照；修改默认配置不会改变已有实验。

全量目录构建和训练可能超过单次 10 小时限额。再次提交相同命令会复用已落盘的 Writer 缓存，并从最新完整 checkpoint 续训。为减少超时丢失的更新，全量 Slurm 配置每次更新保存 checkpoint，保留最近两个；这会增加磁盘写入，每份含优化器状态的 checkpoint 约 61 GB。中断时尚未完成的更新需要重跑，尚未写完的 checkpoint 会归档到本次作业日志目录；若首次 checkpoint 尚未完成，保留旧训练日志后从初始 Reader 重启。全部训练成功才写入 `train.complete`，存在该标记时再次提交会直接结束。

默认服务端口固定为 8000，以保持 Writer 缓存标识一致；端口被占用会报错。可以在**首次提交新实验时**通过 `AGMEM_SERVICE_PORT` 指定其他端口，后续续训沿用快照中的端口。新实验通过 `AGMEM_RUN_DIR` 指定新的持久目录，同一目录不允许并发作业；conda 和项目目录覆盖变量与 smoke 脚本相同。

```bash
# 新建一个独立实验（后续续训继续使用同一个 AGMEM_RUN_DIR）：
AGMEM_RUN_DIR="$PWD/runs/full_experiment_2" AGMEM_SERVICE_PORT=8001 sbatch scripts/train.sbatch

tensorboard --logdir runs/slurm_full_train/train/tensorboard --host 127.0.0.1 --port 6006
```

脚本沿用当前上下文和显存设置，不自动截断超长历史或跳过样本；全量数据的容量限制仍需在实际运行中验证。

## 训练与验证

```bash
CUDA_VISIBLE_DEVICES=0 python -m agmem train
CUDA_VISIBLE_DEVICES=0 python -m agmem evaluate --split validation --output runs/initial_reader_validation
CUDA_VISIBLE_DEVICES=0 python -m agmem evaluate \
  --checkpoint runs/session_grpo_v1/checkpoint-00000100/model \
  --split validation --output runs/trained_reader_validation
```

第一条为正式训练；第二条评估初始 Reader；第三条评估指定训练 checkpoint。训练默认每 100 个更新在固定 validation 上评估、保存 checkpoint，保留最近两个 checkpoint；路径按实际输出调整。test 只在显式 `evaluate --split test` 时参与评测，不参与参数更新。

```bash
CUDA_VISIBLE_DEVICES=0 python -m agmem train \
  --resume runs/session_grpo_v1/checkpoint-00000100
```

恢复时检查输入清单、目录版本、原始模型指纹和训练配置；保存模型、FP32 master weights、Adam 状态、epoch、样本位置与 policy version。新实验使用新的 `training.output_dir`，不会覆盖已有实验。

## 日志与 TensorBoard

训练默认开启 `training.tensorboard: true`，每次更新写入 `<output_dir>/tensorboard/`。`train/reward` 是当前问题的 8 条轨迹平均奖励，横轴为更新步数；同时记录 F1、读取成本、reward 标准差、loss、梯度、协议失败率等。周期验证结果写入 `validation/reward` 等标签。事件每步刷新，退出时关闭；恢复训练时清理 checkpoint 之后的旧曲线点。[PyTorch TensorBoard 接口](https://docs.pytorch.org/docs/2.10/tensorboard.html)

已有环境只需补装新依赖。下面将新的 smoke 放到独立目录，完整 stdout/stderr 写入日志并在后台执行：

```bash
python -m pip install 'tensorboard>=2.20,<3'
run_dir="runs/smoke_tb_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
nohup env CUDA_VISIBLE_DEVICES=0 python -u -m agmem \
  --config configs/smoke.yaml train --max-groups 1 \
  --output-dir "$run_dir" \
  > "$run_dir/train.log" 2>&1 < /dev/null &
echo $! > "$run_dir/train.pid"
tail -f "$run_dir/train.log"
```

`--output-dir` 覆盖本次训练输出目录，数据与 Writer 目录沿用已有配置，不需要重新 `prepare` 或 `build-directories`。`Ctrl+C` 退出 `tail` 不会停止后台训练。正式训练时省略 `--config configs/smoke.yaml` 和 `--max-groups 1`，选择新的输出目录。

另开终端启动 TensorBoard：

```bash
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

浏览器访问 `http://127.0.0.1:6006`；远程运行时可在本机使用 `ssh -N -L 6006:127.0.0.1:6006 用户名@训练服务器` 转发后访问。单组 smoke 只有一个曲线点，多次更新后才能观察变化。

之前已经结束的训练也可以从指标文件导入，不加载模型、不重新训练：

```bash
python -m agmem export-tensorboard runs/smoke_session_grpo
```

导入保持原始 `train_metrics.jsonl` 和 checkpoint 不变。旧运行未记录的终端输出无法从指标文件还原。JSON 代码块解析修复及验证记录见 [训练监控说明](doc/training_monitoring_20261019.md)。

## 训练实现与资源

使用原生 PyTorch 多步 GRPO，按每条完整轨迹内的生成 token 平均，再按 8 条轨迹平均；仅 Reader 生成 token（含自然 stop）进入 policy loss，prompt、工具返回、Task Agent 输出都不作生成目标。每个决策单独保存真实 prompt、token IDs、采样时的实际处理后 logits 所对应 logprob，训练时使用相同温度与模板重算带梯度的 logprob。不使用 top-k/top-p 截断或 JSON 约束解码改变行为分布。

默认 `CPUAdamW` 将 FP32 master weights 和 Adam moments 放在 CPU，模型 BF16 参数与梯度在 GPU；这是全参更新。reference policy 平时在 CPU，组内计算 reference logprob 时移入 GPU，计算完移回 CPU。Writer/Task Agent 只有 HTTP 推理调用，不进入优化器。全参更新只会为参与文本计算图的参数产生梯度，未使用的视觉分支没有梯度。

按当前决定，已取消采样 logprob 的一致性阈值检查：采样后不再额外进行 teacher forcing 校验，也不再保存原始 logits 来与 sampling scores 比较。训练必需的带梯度重算保留，采样/重算的最大绝对差只记为 `max_old_logprob_error` 指标，不因差异大小中止。GRPO 比率继续使用实际采样 logprob，不以重算结果覆盖旧概率。变更及历史排查记录见 [logprob 说明](doc/logprob_consistency_20261019.md)。

CPU offload 显著使用主机内存、传输与磁盘；4B 全参的 master/moments/CPU gradients 加 reference 需要数十 GB 主机内存，建议在至少 128GB 主机 RAM 的节点先实测。checkpoint 包含全量优化器状态，远大于单份 BF16 权重。此实现没有宣称已适配具体 80GB GPU，也不保证任意长度目录都可训练。

`runs/.../groups/*.json.gz` 保存完整轨迹和状态指纹；`train_metrics.jsonl` 保存更新、F1、调用成本、方差和 logprob 差异指标；评估目录保存逐样本 episodes 和按来源/总体/来源平均的指标。指标是本地划分的标签或文本 F1、reward 和成本，尚未接入各 benchmark 额外的官方 Judge 协议，也未接入下一阶段 MemoryArena 环境。

每个训练 group 还自动保存一份便于查看的精简轨迹：`runs/.../rollouts/group-00000000.json`（与完整 group 文件使用相同的从零开始编号）。`rollouts` 数组中每条包含 `reader_steps`：逐轮记录 `reader_output`（原始输出，含可能的格式错误）与 `action`（解析成功的 `ids`、`stop`，失败为 null）；`task_agent_output` 是 Task Agent 的最终答案。同时保留样本 ID、结束原因、F1 和 reward；新运行额外记录 `read_cost`、`format_failure`、`format_reward`。精简文件不包含输入 prompt、目录内容、检索原文、token IDs 或 logprob；完整审计文件仍保留。

已完成的训练可以补导出，无需模型、GPU 或重跑训练：

```bash
python -m agmem export-rollouts runs/slurm_smoke_180646/train
```

## CPU 验证与代码入口

```bash
python -m unittest discover -s tests -v
python -m compileall -q agmem tests
```

- `agmem/data.py`：来源题型、选项与参考答案适配、消息白名单、完整 session 记录。
- `agmem/store.py` / `services.py`：完整历史 JSONL、逐 session 目录及按完整历史隔离的 Writer 缓存。
- `agmem/env.py`：反馈状态、批量读取、去重、成本计数及终止原因。
- `agmem/policy.py` / `grpo.py`：本地 Qwen3.5 接入、实际行为 logprob、轨迹 GRPO 与全参 CPU 优化器。
- `agmem/pipeline.py` / `cli.py`：准备、目录构建、训练、恢复和评估入口。

当前验证记录见 [实现验证](doc/implementation_validation_20261018.md)。

模型快速指纹包含 config/tokenizer JSON 的内容 hash、权重文件大小和 mtime；需要完整权重 SHA-256 时可额外执行 `python -m agmem fingerprint-model /home/share/models/Qwen3.5-4B --full-weights`。






可以按 **GPU 0 训练 4B，GPU 1、2 共同部署一个 27B 服务**启动。27B 使用张量并行，Writer 和 Task Agent 共用这个服务。双卡部署对应 `--tensor-parallel-size 2`。[vLLM 并行部署说明](<https://docs.vllm.ai/en/v0.19.1/serving/parallelism_scaling/>)

以下命令均在项目根目录执行，假设已安装项目依赖。

**1．终端一：启动 27B，保持运行**

```
conda activate agmem


nohup env CUDA_VISIBLE_DEVICES=1,2 \
  vllm serve /home/share/models/Qwen3.5-27B \
  --served-model-name agmem-27b \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  > logs/vllm-27b.log 2>&1 < /dev/null &
```

这里限制每次处理一条序列，并关闭多模态输入，适合当前纯文本任务。[参数说明](<https://docs.vllm.ai/en/v0.19.1/configuration/engine_args/>)

**2．终端二：先验证一组真实 GRPO 训练**

```
conda activate agmem

python -m agmem --config configs/smoke.yaml prepare --limit-per-source 2

python -m agmem --config configs/smoke.yaml build-directories

CUDA_VISIBLE_DEVICES=0 python -m agmem \
  --config configs/smoke.yaml train --max-groups 1
```

这会完成一组 **8 条完整轨迹**的采样、Task Agent 评分、全参反向传播和参数更新。当前实现依次采样这 8 条轨迹。

**3．单组验证通过后，构建完整目录并正式训练**

```
# 切换到 JSONL 存储后先执行 prepare
python -m agmem prepare

python -m agmem build-directories

CUDA_VISIBLE_DEVICES=0 python -m agmem train
```

默认配置已经使用你提供的模型路径，无须修改 `reader.device: cuda:0`；训练进程只看得到物理 GPU 0。

**需要实测的是单张 40GB 的训练显存。** 当前已有 CPU 优化器卸载和梯度检查点，但长上下文仍可能超显存，建议主机内存至少 128GB。上面的命令尚未在三张 40GB 卡上验证。27B 的 `65536` 上下文与当前配置一致；如果启动时提示 KV cache 容量不足而需要调小它，也要同步修改配置中的 `task_agent.max_context_tokens`。




**是的，如果在另一台机器上重新构建目录并开始训练，数据只需要上传 `./data/processed/` 中当前使用的完整版本目录。** 原始数据、旧审计和 smoke 产物都不需要迁移；模型需要另外准备。

我核对了代码，当前读取的是：

```
data/processed/query_dedup_val10_seed42_20261018/
```

该目录约 **2\.02 GB（1.89 GiB）**&#65292;包含 9 个来源的 **27 个 Parquet 文件**，以及划分清单和校验信息。

| 内容 | 放在哪里 |
|---|---|
| `agmem/`、`configs/`、`scripts/`、`tests/`、`doc/`、依赖文件 | GitHub |
| 完整的 processed 数据版本目录 | Hugging Face Dataset 仓库 |
| Qwen3.5-4B、Qwen3.5-27B 完整模型目录 | 目标机器另行准备 |
| `runs/`、prepared 历史与目录、checkpoint | 从头训练不需要；复用目录或恢复训练时另行迁移 |

**代码上传 GitHub。** 当前 .gitignore 已排除 Parquet、prepared 数据、运行结果和模型权重，同时保留 processed 中的小型清单，适合这种分开存储的方式。

当前工作区的 `git status` 返回“不是 Git 仓库”。如果你准备新建一个空 GitHub 仓库，可以在项目根目录执行，替换用户名：

```
git init -b main
git add .gitignore README.md agmem configs scripts tests doc \
  pyproject.toml environment.yml requirements*.txt data/processed

git diff --cached --stat
git commit -m "Add AG-Mem training code and dataset manifests"

git remote add origin git@github.com:你的用户名/AG-Mem.git
git push -u origin main
```

**数据上传 Hugging Face。** 先创建一个 Dataset 仓库，例如 `你的用户名/agmem-data`。下面将整个 `data/processed` 上传到仓库根目录，保留版本目录名；请把清单和 Parquet 一起上传，`prepare` 会用 `manifest.json` 校验文件 SHA-256。上传、登录和指定下载目录的命令已核对 [Hugging Face 官方 CLI 文档](<https://huggingface.co/docs/huggingface_hub/guides/cli>)。

```
# 如果尚未安装 hf CLI
python -m pip install -U huggingface_hub

hf auth login

hf upload 你的用户名/agmem-data \
  ./data/processed \
  . \
  --repo-type dataset
```

Hugging Face 上的目录结构将是：

```
query_dedup_val10_seed42_20261018/
├── manifest.json
├── train_manifest.jsonl
├── validation_manifest.jsonl
├── locomo/
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
├── perltqa_en/
└── ...
```

**另一台机器下载时**，先克隆代码，再把数据下载回原来的相对路径：

```
git clone git@github.com:你的用户名/AG-Mem.git
cd AG-Mem

# 在安装了 hf CLI 的环境中执行
hf auth login
hf download 你的用户名/agmem-data \
  --repo-type dataset \
  --local-dir ./data/processed
```

随后按 环境安装说明 (doc/environment\_setup.md) 创建环境，并准备两个完整模型目录。

如果继续使用 `scripts/train.sbatch`，需要核对目标机器的 **模型路径、conda 路径及 Slurm 资源名称**。该脚本读取 `configs/full.yaml`，继承 `configs/default.yaml`，**不会自动读取 `configs/local.yaml`**&#65307;目标机器的模型路径可以覆盖写在 `configs/full.yaml` 中。首次提交后，脚本会重新执行 `prepare → build-directories → train`。