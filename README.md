# AG-Mem

2026-11-04：LoCoMo Judge 改为按 gold 的必要语义判分。时间题接受有明确基准的相对／绝对日期转换及两者并列；额外的不矛盾细节或列表项不扣分。遗漏必要事实、时间换算错误或与 gold 矛盾仍判错。模板 SHA256 和缓存作用域随之更新；已启动的训练进程仍使用启动时加载的规则。

2026-11-03：四卡入口默认使用 `configs/train_dapo_reader_loop_v3_query9010_format05.yaml`，训练、验证和测试均只使用 LoCoMo，启用 `reader.prompt_version: evidence_loop_v3`。单次最多读取 **1 个 ID**，每次返回原文后再次调用 Reader，整题最多 **8 次 READ**。每一步输入都有 `max_ids_per_call`、`read_calls_used`、`read_calls_remaining`，并继续展示累计 `read_ids` 与绑定的原文。Reader 根据还缺哪些事实决定继续读取或 STOP，不要求用完预算。

采样继续以最多 4 个 group 的固定有序窗口并发，每题保留完整 8 条 rollout；Reader 最多合批 4 个 decision，token 预算 16384，长上下文拆成更小批次且不截断；两个评分 worker 并发执行 Task Agent → Judge。当前窗口全部完成后才进入下一窗口。每轮最多 32 个候选，按原候选顺序选前 16 个有效 group，凑齐后完成当前窗口并停止补采，含首轮最多采样 4 轮。采样和评分全部结束后，按 mini=8、micro=2 更新 2 次，每次更新使用 8 个 group、共 64 条 rollout；多决策轨迹与 token 预算会影响实际 backward 次数，不能将其固定理解为每组 4 次。

继续检索时必须提供 1 个 ID；请求 2 个及以上时记录 `too_many_ids`，只尝试读取原列表第一个 ID，后续 ID 不读取，也不加入 `read_ids`。保留原有 format 计数规则，其中超量、同批重复、无效 ID 和已读 ID 逐 ID 计数：每个违规 ID 出现位置扣 0.5，同一位置触犯多条规则只扣一次；同批重复从第二次出现开始计罚。合法 stop 不读取 ID、也不因其 ID 列表长度扣分。Reader prompt、环境执行和 reward 判定共用 `retrieval.max_ids_per_call=1`。

奖励使用 `judge_cost_v2` 和 `reward.format_penalty_mode: per_id`：`R=2J-1-0.01N_read-0.5(N_bad_id+N_other)`，答对、答错都按消耗的 READ 调用数扣 0.01；空列表、无效 ID、重复读取和容量拒绝只要消耗调用也计入 cost，STOP 和超预算未执行的请求不新增调用成本。`N_bad_id` 是超出单次上限、同批重复（第二次及以后）、目录中不存在或已经成功读过的 ID 出现次数；同一 ID 位置满足多项条件只计一次。JSON/schema、空列表、超预算等其他原有 format 违规按动作扣 0.5，计入 `N_other`，与违规 ID 数相加；同一动作中的其他违规最多计一次。例如超预算并请求 3 个已读 ID，扣超预算 0.5 加 3 个违规 ID 的 1.5，共 2.0。四类 ID 违规不额外叠加统一动作惩罚。合法 STOP 忽略 ID 列表。

Qwen3.5-27B Judge 给出二元 `J`。新 `filter_metric=judge_or_all_correct_format` 保留两类 group：实际可训练轨迹中既有答对也有答错；或完整 group 全对，且实际可训练轨迹中同时存在零 format 扣分与有 format 扣分的轨迹。全错、仅调用成本不同的全对、全部都有 format 扣分的全对组仍过滤。差异标准差须超过 `filter_min_std=1e-6`，且存在非零 advantage；只有截断或空 token 轨迹提供差异的组不纳入。有效组保存完整 8 条轨迹，advantage 仍在完整 group 上计算。Writer 和 Task Agent 的 prompt 不变，Judge 使用上文更新的语义判分规则，D 默认关闭。每 4 次 optimizer step 验证并保存 checkpoint，即 mini=8 时累计完成 2 批、32 个有效 group 后验证；额外周期 checkpoint 每 25 步保存。`sampling/accepted_all_correct_format_rate` 记录通过新分支的组占全部候选的比例，`sampling/pipeline_round_*` 记录流水线统计。正常 HTTP 完成先后不改变 Reader 合批及候选选择；故障重试不承诺逐 token 复现。

LoCoMo 将原训练集与验证集合并后，按精确 query 字符串去重分组，以 seed=42 随机打乱，按 9:1 重新划分。共 1,134 条记录、1,126 个不同 query；验证 query 数向上取整为 113，训练为 1,013 个 query。对应训练 1,019 条、验证 115 条，相同 query 的全部记录始终在同一侧，不删除重复记录。LoCoMo test 的 400 条记录逐字节保留；当前实验的训练、验证和测试均仅选择 `data.sources: [locomo]`，LongMemEval-S 不参与采样、评分或指标汇总；其历史数据文件保留。新配置复用该数据划分、prepared 与 Writer 结果。

若旧实验仍在运行，先在原启动终端按 Ctrl+C，等待其训练与服务进程退出；保留现有四卡 Slurm 作业，不执行 `scancel`。然后在该四卡作业内、项目根目录检查并启动新实验：

```bash
conda activate agmem
python -m agmem.four_gpu --train-config configs/train_dapo_reader_loop_v3_query9010_format05.yaml --check-only
python -u -m agmem.four_gpu --train-config configs/train_dapo_reader_loop_v3_query9010_format05.yaml
```

`--check-only` 不加载 GPU 模型。新输出目录为 `runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train/`。新实验从初始4B权重开始，不加 `--resume`；旧 reward/filter 的 checkpoint 不能直接切换到新配置续训。原 `query9010_id_cost` 及其 restart/windowstop 配置保留 0.1 惩罚和混合 Judge 过滤，供旧实验恢复。正在运行的进程不会热加载新配置。

迁移到其他机器后，在旧 val100 数据与 prepared 已就绪且新输出目录尚不存在时运行：

```bash
python -m agmem.query_resplit \
  --config configs/train_dapo_reader_loop_v3_val100.yaml \
  --fraction 0.1 --seed 42 \
  --output-data data/processed/locomo_query9010_seed42_20261102 \
  --output-prepared runs/locomo_longmemeval/prepared_locomo_query9010_seed42_20261102
```

`manifest.json` 记录划分规则、query/记录数量及文件哈希；`row_mapping.jsonl` 保存每条记录对应的原 split 和行号。工具逐字段校验 Parquet 内容与 schema，并更新 prepared 样本 ID、记忆映射及来源信息，不调用模型。

旧实验可显式使用原配置，例如 `--train-config configs/train_dapo_max1_no_call_cost_mixed_v2.yaml --resume`；只有新实验自身的checkpoint才能使用新默认配置续训。以下历史GRPO配置和奖励说明不代表当前默认行为。检查、续训及指标解释见 [DAPO训练说明](doc/dapo_judge_training.md)。

新增独立 A100 入口：`scripts/build_directories_all_a100.sbatch` 申请 1 张 `a100-sxm4-80gb`，单卡部署 27B，从 LongMemEval-M 开始为除 LoCoMo/S 外的七个来源构建目录；`scripts/train_eval_locomo_s_a100.sbatch` 申请 4 张 A100 40GB 卡，其中两张分层运行 Reader 全参 GRPO，另外两张部署 27B，负责 LoCoMo/S 的目录、训练和测试。两个任务按来源加锁共享缓存，test 使用独立存储。提交与结果路径见 [独立任务说明](doc/independent_a100_jobs.md)。

以下为历史 pilot 配置说明，其 LoCoMo＋LongMemEval-S 来源选择不代表当前仅 LoCoMo 的 DAPO 默认配置。`configs/pilot.yaml` 保留三种可选数据源，`configs/pilot_workflow.yaml` 独立选择目录构建和训练来源。支持按数据集独立准备、构建和续跑，训练时汇总选定来源，train 与 validation 保持分开。已有集中存储可以迁移复用摘要。操作与共享历史审计见 [按数据集构建说明](doc/dataset_directory_workflow_20261022.md)。

pilot 脚本支持 `AGMEM_STAGE=build|train|all`（默认 all），默认完成 LoCoMo/S 目录后立即训练这两部分，结果保存到 `runs/locomo_longmemeval/experiments/locomo_s/train/`。后续 M-only 与三来源实验分别使用 `configs/pilot_workflow_m.yaml`、`configs/pilot_workflow_all.yaml`。

目录缓存可继续复用。旧脚本的 Writer、Reader 与 Task Agent 并发覆盖见 [并发执行说明](doc/concurrency_20261023.md)；当前四卡 DAPO 的跨 group 窗口与 mini/micro 设置以 [DAPO 训练说明](doc/dapo_judge_training.md) 为准。

以下为历史 GRPO 配置说明，奖励与单次格式扣分规则不代表当前默认 DAPO 配置。

目录引导的多步记忆检索与 Reader 全参 GRPO。Reader 使用本地 Qwen3.5-4B；冻结的 Qwen3.5-27B 共用一个服务，分别承担离线 Writer 和 Task Agent。

实现遵循当前约定：每个原始 session 一个目录项；Writer 看到当前 session 和前 3 个原始 session，只概括当前 session。Reader 每次选择 1–4 个 ID，读取整份 session 原文，最多 8 次调用并自主停止。超量选择只尝试读取列表前四个 ID，并计格式惩罚。一个 GRPO group 对同一问题采样 8 条完整轨迹。

```text
reward = 2 * outcome_f1 - 1 - 0.01 * outcome_f1 * logical_read_calls - 0.1 * format_failure
choice outcome: option-label F1, correct=1 / wrong or invalid=0
open QA outcome: deterministic answer-text F1
format_failure: 1 if any Reader format/protocol failure occurs, otherwise 0
```

每条完整 rollout 最多扣一次格式惩罚，合法动作不加格式分。JSON/schema 错误，以及继续检索时空批次、超过 `retrieval.max_ids_per_call`（默认 4）或同批重复 ID，均计格式失败。超量请求按原列表顺序只尝试读取前 N 个 ID，后续 ID 不读取；前 N 个仍需通过重复、未知 ID 与上下文容量检查，一次请求仅计一次调用。未知 ID、跨批重复与容量拒绝保持原有成本处理。`stop=true` 时忽略 ids 列表内容，直接停止，不新增读取成本或格式惩罚；JSON 字段及类型仍须合法。系数由 `reward.format_penalty` 配置，默认 0.1（smoke 继承），设为 0 可消融。训练和评估共用评分；历史日志不会重新计分。详细边界见 [格式奖励设计](doc/format_reward_20261020.md)。

Reader 不展示剩余 token 预算或 session 原文 token 数。超量读取的精简 rollout 额外记录 `ids_truncated`、`selected_ids` 和 `returned_ids`，区分原始动作、被选中执行的前 N 个 ID 与实际成功返回的 ID，不包含检索原文。此次 ID 限制修改不启用 32K 上限或左侧截断；累计上下文仍按原配置检查。旧运行的独立配置快照不会被默认配置覆盖，继续使用其中记录的 ID 上限；修改限制时需同步运行配置，已有目录可复用。

从旧 checkpoint 改变单次 ID 上限并续训时，使用 `train --resume <checkpoint> --fork-retrieval-resume`，配置中的 `training.output_dir` 必须指向新的空目录。该选项保留权重、优化器、step 和数据游标，只允许检索配置中的 `max_ids_per_call` 变化及输出路径变化；reward、Reader 上下文、最大调用次数和训练超参数仍检查一致性。新目录的 `retrieval_changes.jsonl` 记录来源 checkpoint 与切换点，旧日志和验证最佳 checkpoint 不复制；新规则下的最佳模型重新评估。后续从新目录生成的 checkpoint 续跑，使用普通 `--resume`，不再添加 fork 选项。

默认启用 `reward.formula: outcome_gated_v2`；F1=0 时不扣检索成本。`read_cost` 是 F1 加权后的实际扣款，`raw_read_cost` 保留未加权成本。保留最近两个 checkpoint，并保护固定验证集宏平均 reward/F1 各自最佳的 checkpoint；选择记录在 `best_checkpoints.json`。独立 LoCoMo/S 新实验使用 `runs/locomo_longmemeval/experiments/locomo_s_reward_v2_time_v1/train/`（提问日期与时间回答修复后，从初始 Reader 重训并复用目录），最终 test 优先使用 validation 最佳 reward 模型。旧运行不回写。详见 [奖励与 checkpoint 更新](doc/reward_outcome_gated_20261024.md)。

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

## 超算申请 GPU 并部署 Qwen3.5-4B，复用作业

在可直接访问 GPU 的主机上，无 Slurm 作业时可后台启动四卡服务：

```bash
mkdir -p runs
CUDA_VISIBLE_DEVICES=2,3,4,5 nohup bash scripts/serve_qwen35_4b.sh > runs/serve_qwen35_4b_launcher.log 2>&1 &
tail -n 50 runs/serve_qwen35_4b_launcher.log
```

直接模式默认使用物理 GPU `2,3,4,5`（`CUDA_DEVICE_ORDER=PCI_BUS_ID`）、TP=4、本地 `models/qwen3.5-4b` 和端口 `8976`，模型名为 `agmem-4b`，接口为 `http://127.0.0.1:8976/v1`。可用 `CUDA_VISIBLE_DEVICES`、`AGMEM_EXPECTED_GPUS`、`AGMEM_TENSOR_PARALLEL_SIZE` 和 `AGMEM_SERVICE_PORT` 覆盖。每次启动的日志目录为 `runs/serve_qwen35_4b_local_<时间戳>_<PID>/`；出现 `READY` 后可用服务，在该目录创建 `STOP_SERVING` 会停止服务并退出。本地模式遇到启动失败会记录 `SERVICE_FAILED` 并非零退出。

只申请一张 GPU、运行两天时，在登录节点的项目根目录提交：

```bash
sbatch scripts/serve_qwen35_4b_1gpu.sbatch
```

[单卡脚本](scripts/serve_qwen35_4b_1gpu.sbatch) 申请 `gpu02` 分区单节点、1 张 GPU、8 个 CPU、48G 主机内存、48 小时 (`2-00:00:00`)，以 TP=1 部署本地 Qwen3.5-4B。共享部署脚本按申请卡数进行预检，默认设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，避免采样器 JIT 使用旧版 nvcc。单卡作业的终端入口使用 `srun --jobid=<作业ID> --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --pty bash`。日志与状态文件路径和下文一致。

保留三张 GPU 供后续训练时，提交原三卡脚本：

```bash
sbatch scripts/serve_qwen35_4b.sbatch
```

[提交脚本](scripts/serve_qwen35_4b.sbatch) 当前申请 `gpu02` 分区单节点、3 张 GPU、16 个 CPU、120G 主机内存、3 天 (`3-00:00:00`)；由调度器选择节点。需要 6 天时可以使用 `sbatch --time=6-00:00:00 scripts/serve_qwen35_4b.sbatch` 覆盖。实际可获批的时限由分区与账户限制决定。[部署脚本](scripts/serve_qwen35_4b.sh) 默认激活 `agmem` 环境，依次检测当前 `CONDA_EXE` 所属安装目录、`$HOME/miniconda3` 和 `$HOME/anaconda3`，也可通过 `AGMEM_CONDA_SH` 显式指定。模型默认读取项目内的 `models/qwen3.5-4b`，旧路径可通过 `AGMEM_MODEL_PATH` 指定。Slurm 模式保留调度器分配的 GPU 编号，以 vLLM TP=1、BF16、32768 上下文、纯文本模式启动 `agmem-4b` 服务。该模型的 16 个注意力头不支持 TP=3，因此服务运行在一张 GPU 上，其余两张仍由这个 Slurm 作业保留。

Slurm 模式的日志和状态保存在 `runs/serve_qwen35_4b_<作业ID>/`：`job.log` 是控制日志，`vllm.log` 是服务日志，`node.txt` 与 `port.txt` 记录节点和端口，`READY` 表示服务健康检查通过。服务仅监听计算节点的 `127.0.0.1`，Slurm 模式默认自动选择空闲端口；可在提交时通过 `AGMEM_SERVICE_PORT=8000 sbatch ...` 指定。其他覆盖变量包括 `AGMEM_CONDA_SH`、`AGMEM_CONDA_ENV`、`AGMEM_MODEL_PATH`、`AGMEM_MAX_MODEL_LEN`、`AGMEM_MAX_NUM_SEQS` 和 `AGMEM_GPU_MEMORY_UTILIZATION`。

查看日志，将示例 ID 替换为实际作业 ID：

```bash
job_id=123456
squeue -j "$job_id"
tail -f "runs/serve_qwen35_4b_${job_id}/vllm.log"
```

准备训练时，在登录节点创建停止服务标记：

```bash
touch "runs/serve_qwen35_4b_${job_id}/STOP_SERVING"
tail -f "runs/serve_qwen35_4b_${job_id}/job.log"
```

看到 `TRAINING_READY` 后按 Ctrl+C 退出 tail；脚本已经停止服务进程组，主作业继续保留资源。使用同一作业进入计算节点（`--overlap` 允许与仍在等待的 batch 步骤共享已分配资源）：

```bash
srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:3 --pty bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate agmem
cd /share/home/wtma/AG-mem
nvidia-smi
# 在此运行训练命令；进程可见该作业分配的 3 张 GPU。
```

训练并行方式和 Reader/Task Agent 分卡仍由训练入口配置；使用这个作业时，训练入口需要支持三张 GPU。退出交互 shell 后主作业继续等待；训练完成后，在登录节点执行 `scancel "$job_id"` 释放整个作业。时限从原作业开始运行时计算，切换训练不会重置计时。

环境缺少 vLLM、GPU 检查失败、服务异常退出或启动超时时，会记录 `SERVICE_FAILED`，清理服务进程后继续保留作业；`ALLOCATION_READY` 表示已进入保留资源状态。此时可直接用上述 `srun --jobid` 进入节点修复环境或运行训练。服务不会自动重启，修好后可在交互 shell 中手动执行 `vllm serve`。服务成功运行时仍通过 `STOP_SERVING` 切换到训练。`scancel` 和作业时限到达都会正常结束作业。脚本尚需实际 GPU 节点验证。

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

`content` 使用事实短句，以分号分隔，明确主体、动作/状态及结果；优先保留具有辨识度的实体、时间、数量、偏好和约束。在预算内覆盖多个独立话题，优先压缩寒暄和重复；区分事件日期与 session 日期，无可靠依据时保留相对时间，并保留计划/进行/完成/取消等状态及变化。所有摘要事实必须来自当前 session，前 3 个 session 仅辅助有依据的指代消解；保留人物归属、否定和不确定性，不把提问或假设写成既定事实。这些内容要求由提示词引导，格式和长度校验本身不判断事实准确性。

目录写入 `data/prepared/session_jsonl_v2/directories.jsonl`，每行一个 session 目录项，包含 `memory_id`、`version`、`id`、`session_id`、`order` 和最多 300 tokens 的 `content`。Reader 根据当前问题的 `memory_id` 加载完整目录，按选中 ID 从 `memory_records.jsonl` 中该轨迹的有序 sessions 定位原文。训练与检索直接读取这些 JSONL 文件。

每完成一项就追加保存，`directory_manifest.json` 标记完整生成并校验过的目录；中断后重跑可继续同一历史尚未完成的部分。`*.index.json` 仅保存可重建的字节偏移索引，`completions.jsonl` 保存模型响应缓存。

Writer 只生成摘要纯文本，`writer.content_max_tokens: 300` 同时作为模型请求的 `max_tokens` 和最终 `content` 上限；不再输出 JSON、`content` 字段名或目录元数据。事实短句、实体细节、时间状态、多个话题及当前 session 证据边界等摘要要求保持不变。程序提取 session ID、顺序、时间、人物等元数据，将摘要放入 `content`，再序列化为 JSONL 目录项。300 tokens 是上限，无需凑满。

Writer 返回非空正文时，即使 `finish_reason=length` 也直接接受，不做长度或格式重试。若服务异常返回超过上限的文本，程序按本地 tokenizer 截取前 300 tokens；允许摘要末尾不完整。移除独立的 `writer.max_output_tokens`、`writer.format_retries` 和重试预算扩展；HTTP/连接故障仍按 service 配置重试。空正文会记录并保存为 `content: ""`、`writer_status: "empty_pending"` 的待补目录项，继续后续 session 和后续流程；不在同一次构建中立即重试，也不把空响应写入成功 completion 缓存。下次运行 `build-directories` 或全量脚本时只重试待补项，成功后清除标记，正常摘要继续复用。`prepared/manifest.json` 的 `directory_empty_pending` 记录本次构建结束时的待补数。原始 session 本身为空仍使用固定无记录摘要，不标记待补。Task Agent 仍拒绝截断答案，处理规则保持不变。成功响应缓存于 `completions.jsonl`，失败响应及结束原因、用量记录于 `service_failures.jsonl`。本次纯文本协议与 300-token 预算会更新目录版本，清洗历史可以复用，旧 JSON/200-token 目录项保留但不会被新版本复用，需要重新构建。旧运行的 `config.yaml` 快照也需将 writer 改为 `lookback_sessions: 3`、`content_max_tokens: 300`，否则仍会沿用快照中的长度。详见 [Writer 纯文本输出说明](doc/writer_plain_content_20261021.md)。

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

申请资源与 `scripts/smoke.sbatch` 一致：同一节点 3 张 `a100-pcie-40gb`、120G 主机内存、16 个 CPU、10 小时。GPU 分工和 27B 服务参数也一致。脚本读取 [configs/full.yaml](configs/full.yaml)，准备全部 9 个来源的 train 与 validation，构建全部对应 session 目录，然后对合并后的 train 运行完整 1 epoch；每组采样 8 条轨迹，没有 smoke 的样本数或 group 数限制。validation 每 50 次更新评估一次；本脚本不准备 test。

默认持久目录为 `runs/slurm_full_train/`，其中 `prepared/` 保存清洗历史、Writer 缓存和目录，`train/` 保存模型、rollout、指标和 TensorBoard，`jobs/<job_id>.<随机后缀>/` 保存该次提交的 `job.log`、`vllm.log`、`prepare.log`、`directories.log`、`train.log`。Slurm 输出为提交目录中的 `agmem-train-<job_id>.out/.err`。首次提交保存有效配置至 `config.yaml`，后续提交使用该快照；修改默认配置不会改变已有实验。

全量目录构建和训练可能超过单次 10 小时限额。再次提交相同命令会复用已落盘的 Writer 缓存，并从最新完整 checkpoint 续训。为减少超时丢失的更新，全量 Slurm 配置每次更新保存 checkpoint，保留最近两个，并额外保护 validation reward/F1 最佳点；这会增加磁盘写入，每份含优化器状态的 checkpoint 约 61 GB。中断时尚未完成的更新需要重跑，尚未写完的 checkpoint 会归档到本次作业日志目录；若首次 checkpoint 尚未完成，保留旧训练日志后从初始 Reader 重启。全部训练成功才写入 `train.complete`，存在该标记且没有待补空摘要时再次提交会直接结束；若还有待补项，再次提交会只补目录，不重复已经完成的训练。

三个全量脚本均先把完整 Bash 执行流程解析为函数，再开始启动服务和准备数据；正常结束也在该函数内退出。这样直接使用 `bash scripts/train_a6000.sbatch` 运行时，即使源 sbatch 文件随后被编辑，已启动的 Bash 流程也不会继续从被改写文件中读取错位内容。

服务端口默认尝试 8000，以保持 Writer 缓存标识一致；如果端口被占用，且尚未创建 prepared 或训练输出目录、没有显式指定 `AGMEM_SERVICE_PORT`，脚本会自动选择空闲端口并更新配置快照。只有配置快照、尚无上述产物时，也可以通过 `AGMEM_SERVICE_PORT` 修改端口。已有产物时，后续续训沿用快照中的端口；冲突会明确报错，避免改变缓存和 checkpoint 的版本标识。显式指定的端口被占用时也会报错。新实验通过 `AGMEM_RUN_DIR` 指定新的持久目录，同一目录不允许并发作业；conda 和项目目录覆盖变量与 smoke 脚本相同。此逻辑适用于 A100、H100 和 A6000 三个全量脚本。

```bash
# 新建一个独立实验（后续续训继续使用同一个 AGMEM_RUN_DIR）：
AGMEM_RUN_DIR="$PWD/runs/full_experiment_2" AGMEM_SERVICE_PORT=8001 sbatch scripts/train.sbatch

tensorboard --logdir runs/slurm_full_train/train/tensorboard --host 127.0.0.1 --port 6006
```

脚本沿用当前上下文和显存设置，不自动截断超长历史或跳过样本；全量数据的容量限制仍需在实际运行中验证。

### 两张 H100 的全量训练

```bash
sbatch scripts/train_h100.sbatch
```

[scripts/train_h100.sbatch](scripts/train_h100.sbatch) 申请同一节点的 2 张 `nvidia_h100_80gb_hbm3`：第一张训练 4B Reader，第二张以 TP=1 部署 27B Writer/Task Agent，BF16 和 65536 上下文保持一致。内存、CPU 和时限沿用创建时的 `scripts/train.sbatch`：120G、16 个 CPU、60 小时（当前原脚本的时限已从最初的 10 小时调整为 60 小时）。

数据准备、目录构建、训练配置、日志及自动续训逻辑均沿用原脚本，默认共用 `runs/slurm_full_train/`；同一目录的作业不能并发。若要新建独立实验，可使用 `AGMEM_RUN_DIR="$PWD/runs/slurm_full_train_h100" sbatch scripts/train_h100.sbatch`。Slurm 输出文件为 `agmem-train-h100-<job_id>.out/.err`。

### 三张 RTX A6000 的全量训练

```bash
sbatch scripts/train_a6000.sbatch
```

[scripts/train_a6000.sbatch](scripts/train_a6000.sbatch) 申请同一节点的 3 张 `nvidia_rtx_a6000`：第一张训练 4B Reader，另外两张以 TP=2 部署 27B Writer/Task Agent。沿用原全量脚本的 120G 主机内存、16 个 CPU、60 小时时限，以及 BF16、65536 服务上下文和完整训练流程。

默认复用 `runs/slurm_full_train/` 的数据、配置快照和 checkpoint，同一目录的作业不能并发。独立实验使用 `AGMEM_RUN_DIR="$PWD/runs/slurm_full_train_a6000" sbatch scripts/train_a6000.sbatch`。Slurm 输出文件为 `agmem-train-a6000-<job_id>.out/.err`。

## 训练与验证

```bash
CUDA_VISIBLE_DEVICES=0 python -m agmem train
CUDA_VISIBLE_DEVICES=0 python -m agmem evaluate --split validation --output runs/initial_reader_validation
CUDA_VISIBLE_DEVICES=0 python -m agmem evaluate \
  --checkpoint runs/session_grpo_v1/checkpoint-00000100/model \
  --split validation --output runs/trained_reader_validation
```

第一条为正式训练；第二条评估初始 Reader；第三条评估指定训练 checkpoint。训练默认每 50 个更新在固定 validation 上评估，周期 checkpoint 每 100 个更新保存（LoCoMo/S 独立脚本每步保存）；验证步也保存 checkpoint，保留最近两个及验证最佳 checkpoint；路径按实际输出调整。test 只在显式 `evaluate --split test` 时参与评测，不参与参数更新。

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

训练入口在导入 PyTorch 前默认设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`，缓解变长输入的显存碎片；如果已有该变量或旧名称 `PYTORCH_CUDA_ALLOC_CONF`，则尊重显式配置。每次策略更新的实际输入长度（prompt + completion 去掉最后一个目标 token）达到 16,384 tokens 时，默认用 `torch.autograd.graph.save_on_cpu(pin_memory=False)` 将反向所需保存张量暂存 CPU，并在反向时恢复到各自原设备。这与非重入梯度检查点组合使用，不截断上下文、不 detach 计算图、不改变 group size 或全参目标。`AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS` 可调整阈值，`0` 禁用、`1` 对所有 CUDA 更新序列启用；采样和无梯度 reference 计算不启用此卸载。它增加 CPU 内存和传输耗时，仍需实测峰值，不能保证任意长序列不 OOM。

这些内存选项属于运行时设置，不加入冻结的训练超参数，可直接恢复旧 checkpoint。`training_memory` 日志和 `run.json.runtime_memory` 记录设置；`grpo_update_started`、`grpo_backward_progress` 记录长序列更新进度；训练指标新增 `activation_offload_decisions`、`max_update_input_tokens` 和 `gpu_<编号>_peak_allocated_gib` / `gpu_<编号>_peak_reserved_gib`（更新阶段，含 reference 前向与优化器更新）。OOM 仍明确报错，不静默跳过样本或自动改变奖励。

按当前决定，已取消采样 logprob 的一致性阈值检查：采样后不再额外进行 teacher forcing 校验，也不再保存原始 logits 来与 sampling scores 比较。训练必需的带梯度重算保留，采样/重算的最大绝对差只记为 `max_old_logprob_error` 指标，不因差异大小中止。GRPO 比率继续使用实际采样 logprob，不以重算结果覆盖旧概率。变更及历史排查记录见 [logprob 说明](doc/logprob_consistency_20261019.md)。

CPU offload 显著使用主机内存、传输与磁盘；4B 全参的 master/moments/CPU gradients 加 reference 需要数十 GB 主机内存，建议在至少 128GB 主机 RAM 的节点先实测。checkpoint 包含全量优化器状态，远大于单份 BF16 权重。此实现没有宣称已适配具体 80GB GPU，也不保证任意长度目录都可训练。

`runs/.../groups/*.json.gz` 保存完整轨迹和状态指纹；`train_metrics.jsonl` 保存更新、F1、调用成本、方差和 logprob 差异指标；评估目录保存逐样本 episodes 和按来源/总体/来源平均的指标。指标是本地划分的标签或文本 F1、reward 和成本，尚未接入各 benchmark 额外的官方 Judge 协议，也未接入下一阶段 MemoryArena 环境。

每个训练 group 还自动保存一份便于查看的精简轨迹：`runs/.../rollouts/group-00000000.json`（与完整 group 文件使用相同的从零开始编号）。`rollouts` 数组中每条包含 `reader_steps`：逐轮记录 `reader_output`（原始输出，含可能的格式错误）与 `action`（解析成功的 `ids`、`stop`，失败为 null）；`task_agent_output` 是 Task Agent 的最终答案。同时保留样本 ID、结束原因、F1 和 reward；新运行额外记录 `outcome_reward`、`raw_read_cost`、`read_cost`、`format_failure`、`format_violation_count`、`format_reward`；`format_violation_count` 在当前 `per_id` 配置中是违规 ID 出现次数加其他 format 违规动作数；历史 `per_decision` 配置中是违规动作数。精简文件不包含输入 prompt、目录内容、检索原文、token IDs 或 logprob；完整审计文件仍保留。

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
  --max-num-seqs 2 \
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

这会完成一组 **8 条完整轨迹**的采样、Task Agent 评分、全参反向传播和参数更新。默认每批推进 2 条 Reader 轨迹，Task Agent 最多并发 2 个请求；8 条轨迹及评分全部完成后统一更新参数。

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

### Reader ID 修复协议与 checkpoint 100 续跑

2026-10-26 新实验使用 `configs/resume_locomo_s_idrepair.yaml`，显式启用 `repair_v2`。有限补零修复、保留同批有效记录、重复无效调用终止均保留原始采样文本；协议违规每条 rollout 最多扣 0.1。旧配置缺省仍为 `strict_v1`。新分支恢复模型和优化器，并先评估新协议下的 step 100 基线，再从 step 101 继续。执行命令、审计字段及后续续跑说明见 [Reader ID 协议 v2](doc/reader_id_protocol_v2.md)。

### Checkpoint 20/32 的 LoCoMo / LongMemEval-S 测试

配置入口为 `configs/eval_locomo_longmemeval_checkpoints.yaml`。评估读取已保存的训练 `run.json`，固定 Reader prompt、每次 1 个 ID、最多 8 次 READ、Task prompt，以及本地 Qwen3.5-27B Judge。`configs/protocols/format05_training_judge.json` 精确保存该轮训练使用的 Judge 模板；不会覆盖当前训练代码中修改后的模板。

本次使用仓库的固定 test split：LoCoMo 400 个问题、LongMemEval-S 150 个问题（不是 LongMemEval-S 全部 500 个问题）。LoCoMo 的 10 份历史与训练历史完全相同，可复用原目录；这是按 query 留出的测试，不是未见历史测试。LongMemEval-S 的 150 份测试历史需要用本地 Qwen 构建 7,163 个 session 目录项。准备数据独立保存在 `runs/evaluation/format05_best_final/prepared_test/`，不会写回训练 Store。

在已有四卡 allocation 的 GPU 节点、项目根目录执行：

```bash
conda activate agmem
python -u -m agmem.evaluate_gpu --task-agents local
```

默认配置为 `task_agents: [local]`，GPU 节点不需要外网或 OpenAI 凭据。`export_remote_inputs: true` 会在每个 checkpoint 的本地评测完成后导出同一批 Task 输入，供联网机器单独运行 GPT；远端不可用不会阻止下一个 checkpoint 的本地评测。

默认配置启用 `parallel_directory_build: true`。在 `--stage all` 且选择两个数据集时，启动器使用 `CUDA_VISIBLE_DEVICES` 中的四张卡并行分工：第 1 张运行 Qwen Writer，续建 LongMemEval-S 测试目录；第 2、3 张运行 Reader；第 4 张运行另一个 Qwen 服务，处理 LoCoMo 的 Task 和 Judge。LoCoMo 优先评测全部选中的 checkpoint，不等待 LongMemEval-S 目录构建。两项任务都完成后，Reader 和 Task/Judge 自动转入 LongMemEval-S 评测。

该模式可显式指定 `--parallel-directories`；`--no-parallel-directories` 使用原来的顺序流程。只选一个数据集或使用分阶段命令时，第 1 张卡仍预留，评测使用第 2、3、4 张卡。启动器只关闭自己启动的进程，不终止其他任务或 Slurm allocation。`--local-base-url http://127.0.0.1:PORT/v1` 可以复用 Task/Judge 服务，并行目录构建仍使用第 1 张卡上的独立 Writer。

#### GPU 无外网时：本地评测 → GPT 回答 → 本地 Judge

先完成 LoCoMo 两个 checkpoint 的本地评测。已完成的 best/checkpoint 20 会直接复用，缺失的 final/checkpoint 32 会继续运行：

```bash
python -u -m agmem.evaluate_gpu --datasets locomo --task-agents local
```

如果现有任务已经在 GPU 3 部署了 Task/Judge Qwen，应在命令末尾添加 `--local-base-url http://127.0.0.1:PORT/v1`，填入该任务日志里的端口，复用现有服务。只选 LoCoMo 不占用 GPU 0，可以让已有 LongMemEval-S 目录构建继续执行；不要在占用中的 GPU 3 上启动第二个 Qwen 服务。

每个本地组合完成后，输入包位于：

```text
runs/evaluation/format05_best_final/results/best/locomo/offline/requests.json.gz
runs/evaluation/format05_best_final/results/final/locomo/offline/requests.json.gz
```

旧版本已完成的本地结果也可以只补导出，无需 GPU、网络或重新运行模型：

```bash
python -u -m agmem.evaluate_checkpoints --stage export-remote --datasets locomo
```

把这两个包分别复制为联网机器上的 `best_requests.json.gz`、`final_requests.json.gz`，同时复制 `agmem/remote_task_runner.py`。该脚本仅依赖 Python 标准库，不需要整个仓库、GPU、模型权重或训练数据。在联网机器执行：

```bash
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY
export OPENAI_BASE_URL='https://YOUR_ENDPOINT/v1'
python remote_task_runner.py --requests best_requests.json.gz \
  --output returned/best/locomo --workers 4
python remote_task_runner.py --requests final_requests.json.gz \
  --output returned/final/locomo --workers 4
```

每个 checkpoint 使用自己缓存的 Reader 证据；导出会核对与本地 Qwen 使用的 Task 输入哈希一致。包内只包含请求所需的问题、原文证据和协议元数据，不包含 gold answer、本地 Qwen 回答或 Judge 标签。联网步骤只生成 GPT 回答，不重新运行 Reader，也不评分。每题成功后立即保存，网络失败可原命令续跑；空回答不会当作错误答案入分，非空但截断的回答按原训练/评测规则保留截断标记。

GPT 的 `max_completion_tokens` 包含推理 token。联网脚本首次仍按输入包请求 512 tokens；只有实际收到“正文为空、`finish_reason=length`”的响应时，才按 512 → 2,048 → 8,192 → 16,384 逐级扩大预算重试。CLI 的 `--empty-answer-max-tokens` 默认 16,384。`--retries 3` 指每个 token 档位最多三次普通错误重试，预算扩容独立计数：即使前三次网络失败、第四次才返回空正文，也仍然会进入 2,048 档位，并重新获得该档位的普通重试额度。其他错误不会扩大预算；达到最高档位仍失败时会明确报错，不会无限重试。此策略写入响应 manifest，每条回答记录实际请求预算，导回 Judge 后仍保留该信息。旧的成功回答原样保留，首次续跑会在运行锁内升级 manifest，并校验旧回答的原始签名。它是显式的空回答重试策略，不代表 GPT 与本地 Qwen 有相同的可见输出 token 上限。

请求开始、失败原因、HTTP 状态和 token 用量会写入对应 `responses/events.jsonl`，不记录密钥或原文。等待期间每 30 秒输出在途 query 和已等待时间；某题彻底失败会立即报告，然后保存其余在途请求的成功结果再退出。诊断时可用 `--timeout 120` 缩短单次 socket 等待；超时不是整个评测的时限。原脚本隐藏的 `ValueError` 现在会给出 `empty_text`、`invalid_usage` 等原因码。代码更新不会改变已启动进程，需待旧进程退出后再执行原命令；不要删除已保存的 responses 目录。

把整个 `returned/` 目录复制回 GPU 节点后，沿用训练的本地 Qwen Judge 打分：

```bash
python -u -m agmem.evaluate_gpu --stage judge-remote --datasets locomo \
  --responses-root /path/to/returned
```

如 Qwen 服务仍在运行，同样添加 `--local-base-url http://127.0.0.1:PORT/v1`。此阶段不加载 Reader，不调用远端 API，也不需要 OpenAI 凭据。省略 `--responses-root` 时，会分别从 `results/{best,final}/locomo/offline/responses/` 读取回答。每份回答必须与对应输入包、query、模型和检索版本一致；不完整或不匹配的返回包会被拒绝。评分结果保存到原有 `answers/remote/`，并加入 `results/summary.json` 和 `summary.csv`，可与 `answers/local/` 直接比较。

也可以在准备好提交时使用单卡评分脚本：

```bash
sbatch scripts/judge_locomo_gpt.sbatch
```

该脚本申请 `gpu02` 分区 **1 张 GPU、8 小时、8 个 CPU**，部署本地 Qwen3.5-27B（TP=1），依次评分 step 20 和 step 32 的 LoCoMo GPT 回答，共 800 条。它复用 `configs/protocols/format05_training_judge.json`，并核对保存的训练 `run.json` 中的 Judge 模板哈希和评分协议；不会使用当前代码中另行修改的训练 Judge 模板。成功结果逐题保存，重启可续跑，完成后退出并释放本作业服务。Slurm 日志为 `agmem-gpt-judge-<作业ID>.out/.err`；逐题评分和指标仍写入 `results/{best,final}/locomo/answers/remote/`。脚本创建不会自动提交或申请资源。

#### 单独使用 GPT-5-mini 作为 Judge

`scripts/judge_locomo_gpt5mini.sh` 在联网 CPU/登录节点运行，只对已保存的 GPT Task 回答评分，不生成新答案、不运行 Reader，也不申请 GPU。它与本地 Qwen Judge 使用同一份冻结训练模板及相同的完整 gold 参考答案规则，Judge 模型则改为 GPT-5-mini。只向 Judge 发送问题、gold 和已有回答。

```bash
conda activate agmem
# 可先检查 800 条回答和训练 prompt 哈希；不调用 API，也不需要凭据。
bash scripts/judge_locomo_gpt5mini.sh --check-only

read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY
export OPENAI_BASE_URL='https://YOUR_ENDPOINT/v1'
bash scripts/judge_locomo_gpt5mini.sh --workers 4 --timeout 120 --retries 3
```

独立结果目录为 `runs/evaluation/format05_best_final/gpt_judge_results/{best,final}/locomo/answers/remote/metrics.json`，汇总为同根目录的 `summary.json` 和 `summary.csv`。原 Qwen Judge 结果仍在 `results/`，不会覆盖。逐题评分和每次有效 Judge 标签都会缓存，使用原命令可续跑；更换输入或 Judge 协议时需指定新的 `--output-root`。

GPT Judge 为推理预留首次 2,048、最高 16,384 completion tokens，实际预算和 Judge 身份写入评分协议及调用记录。Judge 输出必须通过原严格标签解析器；空回答、被 `length` 截断的标签、非法 JSON 或无效标签不会被当作 WRONG。请求失败会记录诊断并保留成功进度。两个 Judge 的 prompt 一致，模型及生成参数分别明确记录，分数应按 Judge 分开比较。

DNS、连接超时或一般 TLS 连接故障耗尽单轮重试后，默认再进行最多 6 轮网络恢复，轮间等待 30 秒、60 秒、60 秒……；可用 `--network-retries` 和 `--network-retry-delay` 调整，设 `--network-retries 0` 可关闭。恢复参数只影响执行并记录在事件日志中，不改变评分协议，已有评分可继续复用。原 `--retries 3` 同时控制标签重试并已写入评分协议，已有结果续跑时保持该值。认证错误、证书校验错误、无效标签及输出预算耗尽不会触发这层网络恢复。网络长期不可用时仍会明确报错，保留进度且不给失败题赋分；重试不能修复 DNS 或代理配置本身。

#### 使用 History Chats / answer 标签模板重新评测

`configs/eval_locomo_answer_tags.yaml` 为 Qwen3.5-27B 和 GPT-5-mini 选择同一个 `history_answer_v1` Task 模板，使用用户指定的 `History Chats: ... / Question: ...` 格式，要求答案包在 `<answer>...</answer>` 内。历史部分保留检索到的原始会话文字、人物和会话时间，不再加入目录摘要或要求补充相对/绝对日期。Judge 仍使用 `configs/protocols/format05_training_judge.json` 中的严格 LoCoMo-Refined 模板，已经与用户提供的全文逐字核对。

在联网节点激活 `agmem` 并设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 后执行：

```bash
bash scripts/eval_locomo_gpt_answer_tags.sh
```

脚本先复用已有 step20/step32 检索轨迹导出新请求，再依次完成每个 checkpoint 的 GPT 回答和 GPT Judge 评分；无需 GPU，也无需先重跑 Qwen。默认并发 2，可通过 `AGMEM_EVAL_WORKERS` 调整。重新执行原命令会跳过已经完成的任务。回答原文保存在响应包及评分轨迹的 `task_completion.raw_text`，Judge 与 F1 只使用标签内的答案。无效/空/重复标签以及 `length` 截断不能被当作有效回答；GPT 初始预算 2048、最高 16384，预算扩容策略独立记录。

新回答保存到 `runs/evaluation/format05_best_final/answer_tags/results/{best,final}/locomo/offline/responses/`，GPT Judge 汇总保存到 `runs/evaluation/format05_best_final/answer_tags/gpt_judge_results/summary.csv`。旧版结果仍在原目录；历史 prompt 契约保留以便读取和续跑旧任务。

仅导出可在无网络、无 API 凭据环境运行：

```bash
python -u -m agmem.evaluate_checkpoints \
  --config configs/eval_locomo_answer_tags.yaml --stage export-remote \
  --datasets locomo --checkpoints best final
```

若之后重跑本地 Qwen，复用已部署的 Qwen3.5-27B 服务执行下列命令；同样不加载 Reader，生成的本地答案会采用新模板：

```bash
python -u -m agmem.evaluate_checkpoints \
  --config configs/eval_locomo_answer_tags.yaml --stage score --task-agents local \
  --datasets locomo --checkpoints best final \
  --local-base-url http://127.0.0.1:PORT/v1
```

#### GPU 可以联网时：可选的直接远端评测

远端 Task 使用标准 Chat Completions 接口，端点应包含服务要求的 `/v1` 前缀。凭据只从环境读取。默认模型为 `gpt-5-mini`；若服务采用其他名称，在导出或直接评测命令中用 `--remote-model MODEL_ID` 指定。远端请求不发送 vLLM 专用字段，默认省略 temperature，并用 `max_completion_tokens` 设置输出上限；该上限包含可见输出和推理 token。供应商参数差异可在 `remote_task` 中显式配置。只有显式选择 `remote` 时才会从 GPU 节点直接访问远端：

```bash
read -rsp 'OPENAI_API_KEY: ' OPENAI_API_KEY
export OPENAI_API_KEY
export OPENAI_BASE_URL='https://YOUR_ENDPOINT/v1'
python -u -m agmem.evaluate_gpu --task-agents local remote
```

从旧的顺序评测切换时，在原终端停止旧启动器，等待它释放自己的 GPU 服务和锁，再运行上述命令。已经保存的 LongMemEval-S 目录项会直接复用；已有检索和评分也会跳过。不同数据集使用独立锁，可以同时构建和评测；同一数据集的 Store 操作仍互斥。背景目录失败会立即报告，但会让已启动的 LoCoMo 评测继续完成并保存结果。

每个 checkpoint × 数据集只检索一次，两个 Task 模型使用相同的原文证据，默认评测流程使用本地 Qwen Judge；独立 GPT Judge 脚本另存评分结果。两者共享训练时的 Qwen tokenizer/65,536-token 证据容量检查；这不表示远端模型也具有相同的真实 token 上限。远端容量、网络或空回答错误会中断并保留进度，不会算作回答错误。

评测并发由该 YAML 的 `evaluation_execution` 独立控制：默认同时推进 4 个 query，每次 Reader batch 最多 4 个决策，按 16,384 个 padded tokens 的预算自动拆批，长输入保持完整。本地 Task→Judge 有 2 个并发 query，远端 Task→Judge 有 4 个并发 query；同一道题先完成回答再评分，不同题的回答与 Judge 可以同时进行。本地 Qwen 服务仍最多同时处理 2 个请求，多余请求排队。

默认 `--stage all` 中，一道题检索完成并保存后立即送入本地 Task 的评分池，Reader 在主线程继续推进其他题；本地评测全部完成后导出 GPT 输入。显式选择多个 Task 模型时，第二个 Task 模型复用已保存证据并发评分。各评测组合依次使用 Reader，另一张卡上的目录构建同时进行。每题仍一次 READ 1 个 ID；并发不增加单题检索预算。更改并发数可以继续复用已有结果，运行设置另存于 `executions/`，逐步轨迹记录实际 Reader batch 大小。greedy 批处理可能存在浮点差异，不承诺和逐题推理逐 token 完全一致。

所有阶段可重入；逐样本原子保存，不重复执行已经完成的结果。更改模型、目录、数据、prompt 或评分协议时，已有输出会拒绝混用，需要在配置中指定新的 `output_root`。常用分阶段命令：

```bash
# CPU：准备测试数据、复用已有目录，不调用模型。
python -u -m agmem.evaluate_checkpoints --stage prepare
# GPU：仅构建缺失目录。
python -u -m agmem.evaluate_gpu --stage build
# 仅运行 Reader，缓存两 checkpoint 的测试检索证据。
python -u -m agmem.evaluate_gpu --stage retrieve
# 已有检索结果后，仅补远端 Task + 固定本地 Judge。
python -u -m agmem.evaluate_gpu --stage score --task-agents remote
```

`--datasets locomo`、`--checkpoints best` 可缩小运行范围；`--limit 2` 是独立 smoke 输出，不会混入完整测试结果。运行日志保存在 `runs/evaluation/format05_best_final/jobs/`；结果位于 `results/{best,final}/{locomo,split_longmemeval_s}/`，包含共享 `retrieval/`、独立的 `answers/{local,remote}/metrics.json` 与压缩逐题轨迹。全部已完成结果汇总到 `results/summary.json` 和 `results/summary.csv`。

并行模式的日志分别为 `directories.log`（LongMemEval-S 目录）、`eval_locomo.log`、`eval_split_longmemeval_s.log`，两个 Qwen 服务日志为 `writer_vllm.log` 和 `qwen_vllm.log`。同一 job 下的 `schedule.json` 记录 GPU 分工，`progress.json` 记录各阶段状态；LoCoMo 完成后即可查看它的 `metrics.json` 与共享 `summary.csv`，无需等待 LongMemEval-S。

### 目录引导的过程奖励 DAPO 训练

新实验入口为 `configs/train_dapo_directory_process_v1.yaml`，对应 `method.png` 的奖励归因、动作归因和逐步优势，优化器及动态过滤继续使用 DAPO。只训练 Reader；Writer、静态目录、Task Agent 和 Judge 均固定。沿用 LoCoMo query 90:10 数据及已经构建的目录，每个目录项保留稳定 ID、`time_order.time`、`time_order.order` 和摘要。

复用静态目录时，在已有三卡 allocation 的 GPU 节点、项目根目录执行以下命令；它不会申请 Slurm 资源。`--train-only` 将前两张可见 GPU 分配给 Reader，第三张分配给固定的 Task/Judge 及过程 likelihood 服务，不启动 Writer 或后台目录构建：

```bash
conda activate agmem
export VLLM_USE_FLASHINFER_SAMPLER=0
python -u -m agmem.four_gpu \
  --train-only \
  --train-config configs/train_dapo_directory_process_v1.yaml
```

可先追加 `--check-only` 核对已有训练/验证数据及目录（不使用 GPU）。三卡启动日志位于 `runs/three_gpu/<SLURM_JOB_ID>-<timestamp>/`。若确实需要四卡同时运行后台目录构建，使用原来的四卡 allocation 并去掉 `--train-only`。

该配置从配置中的基础 Reader 模型启动独立实验，当前输出到 `runs/locomo_longmemeval/experiments/locomo_dapo_directory_process_v2/train/`，保留原 v1 运行记录。新的 Reader/Task 输入协议和过程评分协议不能直接续接旧的 outcome-only DAPO 或 v1 PRM run；新实验产生 checkpoint 后，只能用一致的配置与评分协议恢复。

Reader 每次 READ 可以选择 1–2 个 ID，每题最多执行 8 次 READ；可以一次完成，也可以根据返回原文继续检索。`interaction_history_v5` 输入保存此前每次 Reader 原始输出、解析动作及完整工具响应，并提供去重后的已读证据；同时明确告诉 Reader，Task 可以看到已检索 session 的原文、时间和目录摘要，以及公开 observation/options。Task 使用 `history_answer_evidence_v2`：按 `History Chats: ... / Question: ...` 模板接收这些内容和 query，不包含未检索的目录、Reader 轨迹或 gold。提示词要求在保留答题必要信息的前提下简短作答（Keep the answer brief），只输出答案标签，不展开分析。

Task 的生成上限仍为 2048 token。正常结束或命中上限后都使用现有输出，不因 `finish_reason=length` 重试或扩容：存在完整 `<answer>...</answer>` 时取第一组标签内正文；有 `<answer>` 但无后续 `</answer>` 时取开标签后全部文本；没有 `<answer>` 时答案为空字符串。Judge 只收到提取结果，包括空字符串。日志仍保留原始响应、finish reason、截断标记和答案标签完整性，Task 截断不计为 Reader format 违规或 Reader overlong masking。HTTP/传输失败、非文本响应及其他异常结束原因仍作为基础设施错误处理。旧 Task 协议保留原行为和缓存；新协议使用独立缓存身份。过程奖励的 gold likelihood 同样使用新 Task 输入，仍只评分 `<answer>` 后的 gold 正文 token。

轨迹终局 reward 保持原设计：`2 × Judge − 1 − 0.01 × READ调用数 − 0.5 × format违规数`。正确答案的 outcome reward 为 +1，错误为 −1；调用成本对两者均生效。超出每次两个 ID 的位置、同批重复位置、无效 ID 和已成功读取的 ID，按每个违规 ID 位置累计，每个位置即使同时违反多项规则也只扣一次 0.5；无效 JSON/动作结构、空 READ、超过总调用预算等非 ID 协议违规仍按违规 decision 计一次。第 9 次 READ 请求不执行、不增加调用成本，但保留格式违规。过程奖励另用于计算每步优势，不加到这项终局 reward 中。

新方案过滤为 `exclude_all_wrong`：剔除全错组，混合组和全对组均可进入，包括 format、cost 和终局 reward 完全相同的全对组。不再要求全对组有 format 差异；仍保留截断及空 token 防护，至少一条正确轨迹必须可以训练。该模式要求启用过程奖励，旧 ORM 配置及其过滤规则保持不变。通过过滤后才在参数更新前计算过程归因，被过滤的候选轨迹仍记录采样指标。

奖励归因对每一步已经返回的证据前缀做冻结 Task 的标准答案 teacher forcing，使用 gold 正文 token 的平均 log-probability 增量；gold 不进入 Reader。动作归因固定实际采样的下一步 READ/STOP，用同一冻结 Reader 比较保留与删除本轮完整交互时该动作的平均 token log-probability。`mean_tau` 在单条轨迹中具有有效下一动作的 transition 上计算；没有有效下一动作时权重为 1。过程配置默认 `alpha: 1.0`、`gamma: 1.0`，动作权重裁剪至 `[0.5, 2.0]`。`normalization: group_standardize` 按图对同题全部轨迹的 READ 奖励一起标准化，STOP 的过程项为 0；可显式切换 `rms` 只缩放尺度、保留零点。逐步减均值会改变零即时收益动作的累计回报，比较两种设置时应使用独立实验目录。

终局优势为 0 不再阻止过程训练：每步优势仍为终局优势加上加权的过程奖励累计回报。若过程奖励也完全相同，组内标准化后仍可能没有信号，这种组保留在批次中；只有整个批次全部可训练动作的最终优势都为 0 时，才跳过 optimizer step，继续采样，不增加训练 step 或 policy version，也不触发 Adam 动量更新。`skipped_updates.jsonl` 与 `zero_signal_batches/` 保存这类批次及其轨迹。续跑仍以 checkpoint 已提交的采样游标为准，之后未提交的零信号批次会归档并重新采样。采样指标 `accepted_all_correct`、`accepted_zero_terminal_advantage` 和更新指标 `process_zero_terminal_groups`、`process_zero_combined_advantage_groups` 分别记录终局及最终优势的情况，后两项按未被截断掩码排除的训练轨迹统计。

每题采样 8 条 rollout，凑齐 8 个有效 group 后进行一次参数更新（最多 64 条 rollout，仍应用原有截断掩码；整批无信号时按上述规则跳过），然后用当前 Reader 重新采样；`target_groups` 与 `mini_batch_groups` 均为 8。学习率继承为 `1e-6`，每 4 个实际更新 step 验证一次，验证完成后保存 checkpoint；`checkpoint_every: 0` 关闭额外的周期保存，正常训练结束仍保存最终 checkpoint。数据 epoch 为 3。训练轨迹中的 `process_attribution` 保存每步证据增量、动作支持度、权重、过程奖励和优势；训练指标记录相应统计，日志中的 `process_attribution_started/progress` 表示更新前正在计算过程评分。新 Task 协议已做离线单元与集成测试，尚未启动真实 GPU 训练。

### 三卡重建 LoCoMo / LongMemEval 父子目录

`scripts/build_longmemeval_qwen.sh` 现按 [父子目录设计](doc/build_directory.md) 构建 LoCoMo、LongMemEval-S/M 的 train、validation、test。每张 80GB GPU 部署一份 Qwen3.5-27B（TP=1），默认共 48 个并发请求，每个服务最多 16 个活动序列、prefill 分块 4096 token。三份权重依次初始化，之后同时推理。启动前检查可用显存，不会自动终止已有训练或模型服务。

先在原训练终端退出训练并等待显存释放，再在同一个三卡作业中执行：

```bash
cd /share/home/wtma/AG-mem
conda activate agmem
bash scripts/build_longmemeval_qwen.sh
```

登录节点可只检查输入与构建配置，不申请 GPU、不发请求、不写构建数据：

```bash
bash scripts/build_longmemeval_qwen.sh --dry-run
```

如需另外申请三张 A100 80GB，提供独立的两天 Slurm 脚本，手动提交：

```bash
sbatch scripts/build_longmemeval_qwen_3gpu.sbatch
```

也支持复用已经部署的三个相同 Qwen 服务；此模式不启动或停止这些服务：

```bash
bash scripts/build_longmemeval_qwen.sh --base-urls \
  http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 http://127.0.0.1:8002/v1
```

默认配置为 `configs/build_parent_child_qwen.yaml`，新结果写入 `runs/parent_child_directories/qwen27b_child300_v1/`。子摘要预算已从 160 调整为 300 token，因此使用独立输出路径；此前 `qwen27b_v1/` 保留。运行中的进程使用启动时加载的预算，需重启才会应用新配置。旧单层摘要无法充当新父子目录，因此从现有 session 正文重新生成；原 GPT/Qwen 目录保留。LoCoMo 按 conversation 共享 bank，LongMemEval 按题独立 bank。输入相同、构建身份一致的 session 可以跨 bank 复用新生成且已校验的结果，重新绑定本 bank 的 ID；query、gold、has_answer 和原始 answer_ ID 不进入 Writer。

Writer 只看当前 session，不再拼接前 3 个 session。对话中每条完整消息为一个 Block，模型按连续事件、话题、需求或子目标划分子项，输出父项 `title/content` 及每个子项的 `end_block/title/content/entities`。程序生成 ID、起止范围、时间、原文指针与版本。边界必须连续且完整覆盖全部 Block；换说话人或单纯换实体不是切分依据。完整提示词与 Schema 位于 `agmem/hierarchical/resources/`，主要为 `writer_system.txt` 和 `writer_user.txt`。

上下文上限 65,536 token，使用本地真实 tokenizer 和 chat template 计数；标题上限 24、父概述 200、子摘要 300 token，实体最多 12 个。输出预算从 4,096 增至最多 16,384 token；`length` 截断的 JSON 不被接受。超长 session 按完整 Block 分窗，最多带前 2 个 Block 作只读上下文，最后汇总父概述。语义修复最多 2 次；后备固定边界或无事实占位会明确标为 `fallback`，不会冒充正常语义摘要。单条不可分 Block 超出输入预算或请求基础设施故障会记录失败。

默认顺序为 LoCoMo → LongMemEval-S → LongMemEval-M，各自处理 train、validation、test。可加 `--datasets locomo`，或 `--datasets split_longmemeval_s --splits test` 选择子集。`--workers 24 --max-num-seqs 8` 可降低并发。相同命令可续建：成功 session 跳过，失败及默认的 `fallback` 重试；并发和服务端口可更改。更改输入文件、模型身份或提示词需使用新输出路径。

每个 session 完成即提交到 `build_journal.sqlite3`；正常结束或可捕获异常退出时导出 `records.jsonl`、`blocks.jsonl`、`directories.jsonl`、`source_map.jsonl`、`bank_manifest.jsonl`、`writer_traces.jsonl`、`build_report.json` 和 `retry_failed.json`。构建过程中以日志进度和 journal 为准；强制终止后再次运行会恢复并重新导出。报告分别记录 `complete/fallback/empty/failed/pending`；存在失败时整体非零退出，但已完成结果保留。

本地当前只有处理后的 LoCoMo Parquet，没有官方原始 JSON；适配器保留 Parquet 中现有正文，并记录兼容模式警告，不猜测还原先前预处理。若提供官方 JSON，可将该数据源改为 `format: locomo_json` 并配置 `path`，另用新输出路径。

`agmem/hierarchical/reader.py` 提供独立的展开和回读接口：初始只显示父目录，READ 父 ID 展开子项，下一轮才允许 READ 已展示的子 ID 回读对应目录项与原文；父目录始终可见，已读证据保留。展开列表的保留策略需调用方显式选择 `retention="cumulative"` 或 `"latest"`。这些接口尚未替换当前训练/评估的旧 Reader 环境，新数据也不会自动写入现有训练配置。

若确需继续旧单层目录的补建，可显式执行 `bash scripts/build_longmemeval_qwen.sh --legacy-flat --config configs/build_longmemeval_qwen.yaml`；该配置与新父子目录分开保存。

启动器会打印日志目录 `runs/longmemeval_qwen_build/jobs/<jobid>-<timestamp>/`。`directories.log` 记录构建进度，`writer_0.log`、`writer_1.log`、`writer_2.log` 分别记录三卡服务，`cuda_preflight.log` 记录显存检查。退出时只清理本次启动器创建的进程组。

## 相邻语义聚类目录构建

新入口仅处理 LoCoMo 和 LongMemEval-S 的 train、validation、test，配置为
`configs/build_adjacent_qwen_locomo_lmes.yaml`。Qwen3-Embedding-4B 先将同一 session
中的相邻 Block 合并为连续语义段，灰区由 Qwen3.5-27B 判断；随后为固定范围生成子项，
最后依据子项和 session 原文生成父目录条目。原文不重排，范围完整覆盖，无强制“两条消息一个子项”。
每个 bank 的所有 session 条目共同组成父目录；每个 session 有自己的子目录。

默认直接读取本地 `../model/qwen3-embedding-4b/`，使用原生 2560 维向量。
启动前检查模型配置、pooling 维度与客户端维度一致，服务就绪后再用真实 embedding 请求检查输出。
Writer 继续使用 `/share/home/wtma/model/qwen3.5-27b`。
若在其他机器上缺少 embedding 权重，可在能访问 Hugging Face 的节点下载：

```bash
conda activate agmem
bash scripts/download_directory_embedding.sh
```

下载脚本默认写入 `../model/qwen3-embedding-4b/`，保存实际模型 revision 以支持复现。
模型已存在其他路径时，启动命令可加 `--embedding-model-path /path/to/Qwen3-Embedding-4B`。

在已有三卡 allocation 的 GPU 终端运行。如果旧构建仍占用这三张卡，先在原终端停止它并等待显存释放；
启动器不会结束其他进程，也不会提交 Slurm 作业：

```bash
cd /share/home/wtma/AG-mem
conda activate agmem
bash scripts/build_adjacent_directories_qwen.sh
```

| 可见 GPU | 服务 | 默认并发 |
|---|---|---|
| 0 | Qwen3-Embedding-4B | 4 个批请求，每批最多 32 条、16384 token |
| 1 | Qwen3.5-27B Writer | 最多 64 个请求 |
| 2 | Qwen3.5-27B Writer | 最多 64 个请求 |

Writer 总并发 128，所有数据源共享两个副本；单个数据集也可以同时利用两个 Writer。
服务依次启动以降低加载时的 CPU 内存峰值，启动完成后并行处理。
Writer 上下文为 65536 token，标题 / 父摘要 / 子摘要预算为 24 / 200 / 300 token。
聚类阈值为待校准初值：高阈值 0.78、低阈值 0.55、内部一致性下限 0.40。
每段原文最多 4096 token；超过预算的单个完整 Block 保留并标记。
灰区每 session 最多判断 32 次，超过判断预算保留边界，并单独计数；服务失败会明确标为 fallback。

可先检查六份输入文件及参数，再用独立输出目录做小规模构建；前一条命令无需 GPU，缺少 encoder 权重时会明确报告未就绪：

```bash
bash scripts/build_adjacent_directories_qwen.sh --dry-run
bash scripts/build_adjacent_directories_qwen.sh \
  --limit-sessions 20 --output-root runs/parent_child_directories/adjacent_smoke_v1
```

默认正式输出为 `runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1/`。
相同命令可以续跑，复用已提交成功目录、embedding 缓存及成功边界决策，默认重试失败与 fallback。
旧目录不会被覆盖；新构建不读取 LongMemEval-M，也不会自动切换现有训练/评估配置。
更改模型、提示词、聚类参数或输入内容需另设输出路径；并发可以在续跑时调整。
日志位于 `runs/adjacent_directory_build/jobs/<jobid>-<timestamp>/`，包括 `directories.log`、
`embedding.log`、`writer_1.log`、`writer_2.log` 和 `cuda_preflight.log`。
结果与统计仍使用 `directories.jsonl`、`build_report.json`、`writer_traces.jsonl` 等既有文件格式，
另外保存 `embedding_cache.sqlite3`；构建期间进度以日志和 `build_journal.sqlite3` 为准。

需要新申请资源时，可自行提交三张 A100 80GB、两天的脚本：

```bash
sbatch scripts/build_adjacent_directories_qwen_3gpu.sbatch
```

## LongMemEval 当前 8:1:1 数据与三卡构建

后续 LongMemEval-S/M 的训练、验证和测试统一使用
`data/processed/longmemeval_811_seed42_20261111/`，两版均为 400/50/50 题。
原 train 340 题、validation 10 题保留；从原 test 调出 60/40 题，新的 test 50 题
是原 test 150 题的子集。同题在 S/M 中保持同一划分。`row_mapping.jsonl` 保存旧、新行号、
文件 SHA 和 bank ID，`manifest.json` 保存分层抽样与全列内容校验结果。
旧数据、运行配置及已冻结 prepared 仅用于复现历史实验，不应作为后续 LongMemEval 的数据入口。

在已分配三张 80GB GPU 的终端依次运行下列两套脚本；每套启动器结束时会释放自己启动的服务。

```bash
conda activate agmem
# 1. 新 S/M 测试集：构建旧 ORM 单层目录，随后自动评估 checkpoint20/32。
bash scripts/build_longmemeval_811_flat_qwen.sh
# 2. 新 S 的 train/validation/test：保留已有父子目录，缺失部分按新话题/事件规则构建。
bash scripts/build_longmemeval_811_hierarchical_qwen.sh
```

第一套使用三份 Qwen3.5-27B，Writer 总并发 192、每卡最多 64。
Writer prompt 与旧版本一致：当前 session 是摘要目标，前 3 个原始 session 作背景，摘要上限 300 tokens。
已有 GPT/Qwen 摘要保留原内容与来源，缺失项才调用本地 Qwen；输出为
`runs/locomo_longmemeval/prepared_longmemeval_811_flat/test/<dataset>/`。
默认只处理 S/M 测试集；`--splits train validation test` 可显式扩展范围。
目录构建成功并释放三份 Writer 后，同一脚本自动切换为前两张卡运行 Reader、
第三张卡运行 Qwen3.5-27B Task/Judge，对 checkpoint20 和 checkpoint32 各评估 S/M 的 50 道测试题。
若只需要目录构建，加 `--build-only`。旧版脚本已经启动的进程不会自动接入新增评估阶段；
等待它结束并释放 GPU 后执行：

```bash
bash scripts/build_longmemeval_811_flat_qwen.sh --eval-only
```

评估配置为 `configs/eval_longmemeval_811_orm_checkpoints.yaml`。
Reader 使用保存的旧 ORM 训练配置与 prompt（逐项校验输入协议），最多 8 次 READ、每次 1 个 ID；
目录必须来自本节新 8:1:1 测试数据且全部完成，评估不会另建目录或混入父子目录。
Task 使用此前统一的 `history_answer_v1`，输入为问题、公开 observation 和检索原文/时间，
输出完整 `<answer>...</answer>`，上限 2048 tokens；本地截断或缺失标签作为请求失败，不能当成答错评分。
Judge 使用冻结的 `format05_training_judge.json` 中与
[LongMemEval 官方评估脚本](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py)
一致的题型 prompt，`_abs` 题优先使用 abstention 模板。本地严格解析 yes/no，不将格式异常计为错误答案。
本地结果存放于 `runs/evaluation/longmemeval_811_orm/results/summary.csv`；
`best` 对应 step20，`final` 对应 step32，逐题轨迹、按题型指标位于各 checkpoint/dataset 下。
采样和本地回答不访问外网。

每组结果自动导出 `offline/requests.json.gz`，保存相同 Reader 证据与 Task 消息。
之后在能联网且能访问项目结果的 CPU/login 节点设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`，运行：

```bash
bash scripts/eval_longmemeval_811_gpt.sh
```

该脚本用 GPT-5-mini 重新回答并按相同 LongMemEval prompt 判分，不重新运行 Reader。
GPT 结果在 `runs/evaluation/longmemeval_811_orm/gpt/judge_results/summary.csv`，
不会覆盖 Qwen 结果。重跑默认续接已完成请求；设置 `AGMEM_EVAL_OUTPUT_ROOT` 可另开一次远端实验。

第二套先在 CPU 上验证新旧原始行和目录来源，再向独立输出库重绑定可复用目录的
bank/source/snapshot/version，保留摘要、实体、原文和子项边界。
已存在的 complete/empty/fallback 目录原样保留，并记录旧版构建身份；fallback 标记也保留，
不将它改记为正常完成。缺失、pending、failed 的 session 使用 topic/event v2 规则新建。
这是明确记录来源的混合版本实验，不是把全库目录升级成 v2。
不复制旧分段决策或 Writer 缓存，也不复制 10GB 的 embedding 缓存，不覆盖源库和目标中已提交的目录。
然后部署 GPU0 Qwen3-Embedding-4B、GPU1/2 Qwen3.5-27B，总 Writer 并发 128、每卡最多 64；
新建部分的父/子摘要上限为 400/300 tokens，高/低阈值为 0.72/0.50。
复用目录仍记录其原来的 0.78/0.55；旧规则与新规则的阈值分别写入逐 session 来源清单。
该增量配置 `retry_fallback: false`，新产生的 fallback 同样保留标记，后续需要主动定向修复。
服务单次超时 1200 秒，最多重试 1 次，以容纳慢速的长输出。
配置为 `configs/build_adjacent_qwen_longmemeval_811_incremental.yaml`，
输出为 `runs/parent_child_directories/qwen27b_adjacent_qwen4b_811_incremental_v2/`；
`reuse_report.json`、`reuse_provenance.jsonl` 记录导入结果，
`directory_rule_provenance.jsonl` 区分每个 session 的旧版保留目录、新版生成目录及待构建状态。
构建完成后导出最终 JSONL。
当前第二套默认仅构建 S；M 的相邻语义父子目录不属于本次默认范围。

两套脚本均支持 `--dry-run`（只读、不启动服务、不创建结果库），以及原样重跑续建。
父子目录导入前必须让旧构建释放 `.build.lock`；不要删除锁文件或把旧输出库当成新数据的输出库。

### 按具体话题或事件判断边界（v2）

新的边界复核 prompt 位于
`agmem/hierarchical/resources/adjacent/topic_event_v2/boundary.txt`。
同一事件中的提问、澄清、改期、等待、确认和简短收尾保持连贯；独立事件仍分开，
不再把对话行为或计划阶段的变化本身当作切分依据。高/低阈值为 0.72/0.50。

v2 使用独立的构建版本和输出库；旧 v1 的续跑及 fallback 修复仍保留原规则。
如需复现原先全部按 v1 规则补建的 8:1:1 流程，使用
`scripts/build_longmemeval_811_hierarchical_v1_qwen.sh` 和原配置。
旧目录不会因 prompt 更新自动重新分段，也不能将其边界缓存作为 v2 的结果导入。
上一节增量脚本显式保存旧目录来源，只对没有目录的 session 使用新规则。
新入口默认处理 LoCoMo 和新 8:1:1 LongMemEval-S 的 train/validation/test，父/子摘要预算 400/300，
GPU0 部署 Qwen3-Embedding-4B，GPU1/2 部署 Qwen3.5-27B，每个 Writer 最多 64 请求，总并发 128。

```bash
# 只检查输入和配置，不启动推理服务。
bash scripts/build_topic_event_directories_qwen.sh --dry-run
# 如需构建，可先在独立小样本输出中验证新规则。
bash scripts/build_topic_event_directories_qwen.sh --datasets locomo \
  --limit-sessions 20 --output-root runs/parent_child_directories/topic_event_v2_smoke
```

正式配置为 `configs/build_adjacent_qwen_topic_event_v2.yaml`，输出库为
`runs/parent_child_directories/qwen27b_adjacent_qwen4b_topic_event_v2/`。
相同配置和输出路径支持续跑；更改阈值或 prompt 后应使用新输出路径。

## 当前 LongMemEval 固定划分与双卡语义目录（2026-11-13）

后续 LongMemEval-S、LongMemEval-M 统一使用 **315 train / 35 validation / 150 test**：
`data/processed/longmemeval_315_35_150_seed42_20261113/`。
原始150题测试文件逐字节保留；原10题验证保留，再从原340题训练中按题型分层、seed42选25题。
S/M共享question_id分配，三个split无交集。此前811配置和数据保留用于历史实验，不能与此测试集混用。

当前目录入口为 `scripts/build_longmemeval_semantic_2gpu.sh`，配置为
`configs/build_adjacent_qwen_longmemeval_31535150.yaml`。
GPU4部署 `./models/qwen3-embedding-4b`，GPU5部署 `./models/qwen3.5-27b`；
Writer并发64，每卡TP=1，父/子摘要预算400/300。
新增目录使用topic/event v2灰区prompt和高/低阈值0.72/0.50。
已有相邻语义目录按来源校验后导入，保留原有complete/fallback状态及旧规则来源，不复制旧分段决策缓存。

```bash
conda activate agmem
CUDA_VISIBLE_DEVICES=4,5 bash scripts/build_longmemeval_semantic_2gpu.sh
```

默认先进行CPU划分校验与全split目录迁移，再补建S的test，最后构建M的train/validation/test。
`--import-only`仅进行CPU准备；`--s-test-only`或`--m-only`分别运行一个构建阶段；
`--dry-run`检查计划。重复执行同一命令续跑。
S的train/validation已有目录随新划分重新绑定，本次推理不补建其原有缺口，不重建原有fallback。
每个构建阶段会逐题核验所选范围并写入 `selected_scope_*.json`；全库报告仍保留未选范围的pending，
例如S训练的2个既有缺口不会阻塞S测试和M构建，也不会被标记为已完成。
输出为 `runs/parent_child_directories/qwen27b_adjacent_qwen4b_31535150_incremental_v2/`，
`directory_rule_provenance.jsonl`区分保留的旧目录与按新规则构建的目录。

## LongMemEval-S fallback 定点修复（2026-11-14）

当前相邻语义目录中LoCoMo没有fallback；S有55个fallback session：train12、validation35、test8，
影响30道query。55个父目录均为占位摘要，其中3个session还各有1个占位子摘要。
使用 `configs/repair_longmemeval_s_fallbacks.yaml` 和 `scripts/repair_longmemeval_s_fallbacks.sh`，
只在GPU5部署Qwen3.5-27B，默认64并发，无需Embedding。保留全部子项边界/ID和440个正常子摘要，
只修3个占位子摘要，再生成55个父摘要。父/子内容硬上限400/300 token，标题硬上限24 token。

```bash
conda run --no-capture-output -n agmem bash scripts/repair_longmemeval_s_fallbacks.sh --dry-run
conda run --no-capture-output -n agmem bash scripts/repair_longmemeval_s_fallbacks.sh
```

这是原目录库之上的独立修复结果，输出到
`runs/parent_child_directories/longmemeval_s_31535150_fallback_repair_20261114/`。
`repair_report.json`记录成功/剩余数量，全部选中项修复成功后才有`complete.json`；重跑会复用已成功项。
原目录库保持只读，数量不会因为生成修复结果而自动变化。`repair_patches.jsonl`包含替换前后哈希和完整新目录，
后续冻结训练数据时使用`agmem.repair_topic_event_fallbacks.apply_repair_patch`验证并叠加，
不能将仅含修复session的输出目录当作完整训练目录库。
S训练另有2个pending session，未计入55个fallback，也不在此次定点修复范围。

首轮49/55成功后，用户已批准将最后6项的父/子摘要上限调整为500/500 token、父标题30 token，
子标题仍按24 token检查。当前`repair_longmemeval_s_fallbacks.sh`已转到
`repair_longmemeval_s_fallbacks_len500.sh`，配置为`configs/repair_longmemeval_s_fallbacks_len500.yaml`。
新入口完整保留首轮49项，复用4个已有父输出和2个已有子输出，仅补写后两项尚未生成的父摘要。
新结果为`runs/parent_child_directories/longmemeval_s_31535150_fallback_repair_len500_20261114/`，
包含合并后的修复patch；后续应使用这份新patch清单，不重复叠加旧清单中的49项。
`length_acceptance.json`记录原生成请求/响应哈希、实际token数及用户批准的长度上限。

另两个S训练pending已追溯到旧Writer父目录请求ReadTimeout；各自的1个和3个子项均有成功验证记录。
`scripts/repair_longmemeval_s_pending_parents.sh`会核对原请求/响应哈希、源block完整覆盖和新bank映射，
复用这4个子项，仅在GPU5补写2个父标题/摘要（父500token、标题30token，并发2），无需Embedding。
输出`runs/parent_child_directories/longmemeval_s_31535150_pending_parents_20261114/`，
`pending_parent_report.json`及`complete.json`标记补全进度。此输出是两条缺失目录的插入结果，
训练数据冻结时通过`repair_pending_parents.apply_pending_parent_patch`验证原状态为pending且目录为空后应用。
它与55条fallback修复patch分开保存，原始构建库保持不变。

## LoCoMo + LongMemEval-S 双卡 Judge + F1 ORM

当前训练配置为`configs/train_dapo_locomo_lmes_hierarchical_f1_2gpu.yaml`，入口为
`scripts/train_locomo_lmes_f1_2gpu.sh`。从`./models/qwen3.5-4b`启动全参数Reader训练，
GPU4只运行Reader；GPU5部署`./models/qwen3.5-27b`，由固定Task Agent与Judge共享一个服务。
F1在本地计算，不部署额外Writer或Embedding；无需Slurm。
Judge恢复此前放宽的LoCoMo语义评分：接受有明确锚点的等价时间表达和额外无矛盾内容；
LongMemEval仍使用官方提示词。`reward.judge.template_file: null`选择代码内模板，
CPU预检核验模板哈希及题目路由；旧严格版快照和已保存run保持不变。

训练冻结目录为`runs/locomo_lmes_context3_open1_nav06/prepared`，包含LoCoMo1019训练/115验证，
LongMemEval-S315训练/35验证，共1334训练/150验证、360条历史、16976个session。
LoCoMo使用新构建的`qwen27b_locomo_topic_event_context3_v1`目录：0.72/0.50阈值、
topic/event v2边界、当前segment及前三个完整segment原文生成摘要，272个session共767个子项。
训练需要的构建journal、manifest和原文导出已保存到`runs/parent_child_directories/qwen27b_locomo_topic_event_context3_v1`；
其embedding缓存及请求日志仍保留在原`/tmp/agmem_locomo_topic_event_context3_v1`。
S训练的12个fallback和2个pending、验证的35个fallback修复均在冻结时经哈希核验后应用，
冻结结果全部为complete；测试query不进入训练或验证。
源原文、旧目录、修复文件均保留。该冻结目录不跟随后续构建库变化。

每个epoch对两源query合并后按seed+epoch随机打乱，自然比例约76.4%/23.6%，不按数据集顺序训练。
每个query仍独立采样8条rollout、独立计算优势。DAPO每凑齐8个有效group更新一次，
过滤全Judge=0、没有可训练Judge=1轨迹及无非零可训练优势的group；全对组有奖励差异仍可训练。
Reader每次仍只允许1个ID、最多10次READ，同时只展开1个子目录。
输入中的`currently_open_parent_id`明确对应父目录中的ID；已展开仅表示候选目录，
Reader必须判断是否相关，必要时更换父目录，不能把展开当作已经找到目标或取得原文。
`q=Judge*(0.8+0.2*F1)`，PRM关闭。每次READ成本0.05；
所有format违规均扣0.1，包括重复请求当前已展开子目录对应父ID的位置：
`reward.format_penalty=0.1`、`reward.visible_navigation_penalty=0.1`。
总奖励为`2*q-1-0.05*READ次数-0.1*format违规数`。
已被移出列表的目录可以合法重开；已读取原文子项的重复读取仍扣0.1。
成本和format仍独立累计扣分，没有质量门控或新增惩罚总额上限。
当前启用`training.visible_navigation_negative_advantage: {enabled: true, magnitude: 1.0, include_repeated_segments: true}`：
完整group先照常计算每条轨迹的优势`A`，动态过滤仍使用原规则；对重复请求当前已展开子目录、
以及再次读取已经成功获取的segment的动作，将其生成token的训练优势改为`min(A, -1)`，其余动作保留原`A`。
这是动作级优势覆盖，不把整条轨迹的优势改成负数，也不额外改变轨迹reward。
例如原`A=0.8`时仅该违规动作使用`-1`；原`A=-1.5`时该动作仍使用`-1.5`。
首次成功读取segment、失败读取后的合法重试、合法重开已关闭目录不触发这项覆盖。
判断采用每次动作执行前的`already_read_ids`，不会用最终证据集合把首次读取误算成重复；
即使segment所属目录已被替换，已经读取的原文仍保留，再读该segment仍受此规则约束。
其他format违规仍仅累计0.1轨迹罚分。v2规则单独记录`segment_rule_*`训练指标及逐步优势；
没有配置`include_repeated_segments`的旧实验继续使用v1导航规则，冻结记录和恢复协议保持兼容。
层级Reader的STOP必须使用空`ids`。`stop=true`但`ids`非空时，记录一次
`stop_with_ids`格式违规（当前扣0.1），仍立即停止，不读取所列ID、不新增READ成本，
此前已经成功读取的证据继续保留。训练、采样及验证指标增加`stop_with_ids_rate`和
`format_decisions_stop_with_ids`。旧单层Reader保留原有STOP兼容行为。
该规则更新了Reader输入和评分协议指纹；已启动进程不会热更新，旧协议run不能直接混用新规则续训。
学习率1e-6、3epoch、每4次更新验证并保存checkpoint；额外周期保存关闭。
单卡执行默认micro_batch_rollouts=1、Reader采样batch=1；activation offload从8192token开始。
训练、采样及验证分别记录`judge_accuracy`、原始`answer_f1`/`outcome_f1`、`outcome_q`与`outcome_reward`。
验证仍以两源F1的source_macro选择checkpoint；TensorBoard额外记录
`sampling/source_candidates`、`source_accepted`、`source_selected`及`sampling/by_source/`。

```bash
# CPU准备/校验，不占用GPU。
conda run --no-capture-output -n agmem bash scripts/train_locomo_lmes_f1_2gpu.sh --check-only

# 存储足够后，在正常可访问GPU的终端执行。
AGMEM_TRAIN_GPUS=4,5 conda run --no-capture-output -n agmem \
  bash scripts/train_locomo_lmes_f1_2gpu.sh

# 可用AGMEM_TRAIN_OUTPUT_DIR指向其他容量足够的文件系统；新配方自身的同一目录可加--resume续训。
# 一次更新的验证性运行可追加--max-updates 1，仍会保存完整checkpoint。
```

新配方默认输出为`runs/locomo_lmes_context3_open1_repeatneg/train`，实验名为
`locomo_lmes_context3_open1_repeatneg`，复用`runs/locomo_lmes_context3_open1_nav06/prepared`冻结数据。
Judge缓存使用模型与提示词指纹隔离。正在运行的进程不会热加载配置，重新启动后才应用新的
0.1 format罚分和动作级负优势；新实验从初始4B权重开始，不加`--resume`。
旧实验的checkpoint不能直接用新配置`--resume`混入新规则；旧训练数据、结果和评分记录保留。
可选顺序对照配置为`configs/train_dapo_locomo_lmes_hierarchical_batch2first.yaml`，
它将navneg旧实验第二轮的17个候选query（candidate 12--28）提前到epoch 0开头，
之后保留剩余query的原seed打乱顺序；epoch 1及以后仍采用原来的`seed+epoch`打乱。
该配置与默认新配方使用相同的重复动作负优势规则，便于单独比较顺序影响。
`training.initial_query_ids`只重排query，不搬用旧rollout/答案/优势，不改变数据划分和目录；
候选仍要从初始模型重新生成并经过原过滤，所以不能保证首批仍选中原来的8个group。
默认启动脚本仍用普通顺序；选择对照配置时设置`AGMEM_TRAIN_CONFIG`，并清除或更新旧的
`AGMEM_TRAIN_OUTPUT_DIR`覆盖值。两种配置分别写入独立实验目录。
训练保留模型和完整CPUAdamW状态。`training.checkpoint_retention: best_and_latest`
仅保留验证主指标（当前为source-macro F1）最佳和最新已提交checkpoint，重合时只留一份；
不会为了best_reward额外保留第三个。每4次更新验证并保存的频率不变。
必须等新checkpoint完整写入并提交后才删除旧checkpoint；失败的`.building`不参与最佳选择，
也不会自动删除。因此正常保留最多两份，写入时峰值仍可能有三份完整checkpoint。
当前每份完整checkpoint约56.7GiB；这项策略减少长期占用，并不能免除新写入时的额外空间需求。
容量估算会继续报告80GiB首次启动余量和包含原子保存临时副本的建议容量。
2026-11-14按用户要求，当前混合配置设置`training.enforce_storage_reserve: false`，
暂时取消磁盘余量拦截，仅打印容量信息。`--check-only`会同时输出实际`storage_ready`和`enforced`，
空间低于估算值不再阻止当前流程启动。将配置改回`true`可恢复启动前拦截。
实际单卡显存峰值需首轮采样/更新确认，CPU检查不等同于GPU训练已验证。

## LoCoMo父子目录：新版边界与前三个segment原文

`scripts/build_locomo_topic_event_context3_2gpu.sh` 全量重建LoCoMo的train/validation/test，
使用`configs/build_locomo_topic_event_context3.yaml`。默认GPU4部署
`models/qwen3-embedding-4b`，GPU5部署`models/qwen3.5-27b`，Writer并发和max-num-seqs均为64。
相邻语义分段阈值为0.72/0.50，灰区判断使用topic/event v2。

子摘要输入包含当前完整segment、同session紧邻此前最多3个完整segment原文，以及session时间、
参与者等元信息。首段没有前文，第2/3段分别提供此前1/2段；不会输入前文摘要、未来段或其他session。
前文用于明确当前事实的指代对象和人物／事件关系，不把前文独立事件归入当前子项。
输入容量不足时明确失败，不截断、不丢弃规定的前文。子摘要300token、父摘要400token、标题24token；
父摘要继续在全部子摘要之后，结合子项与session原文生成。

新协议`adjacent_topic_event_context3_v1`使用独立输出目录
`runs/parent_child_directories/qwen27b_locomo_topic_event_context3_v1`；不会导入旧目录或旧分段缓存。
三个划分中相同历史只构建一次。再次执行会续建此新协议目录中的未完成项，旧目录和训练冻结数据不变。
新提示词位于`agmem/hierarchical/resources/adjacent/segment_context_v1/child.txt`。

```bash
conda activate agmem
bash scripts/build_locomo_topic_event_context3_2gpu.sh --dry-run
# GPU4、5释放后执行；不会自动停止已有训练或服务。
bash scripts/build_locomo_topic_event_context3_2gpu.sh
# 可将构建结果和缓存放到其他磁盘；每次调用的job-dir必须使用新路径。
bash scripts/build_locomo_topic_event_context3_2gpu.sh \
  --output-root /tmp/agmem_locomo_topic_event_context3_v1 \
  --job-dir "/tmp/agmem_locomo_context3_jobs/$(date +%Y%m%d_%H%M%S)"
```

`AGMEM_DIRECTORY_GPUS`默认4,5，`AGMEM_WRITER_WORKERS`及`AGMEM_WRITER_MAX_NUM_SEQS`默认64。
`--limit-sessions N`可限制首次检查的session数，后续去掉该参数即可继续构建。
`/tmp`是节点本地临时存储，需在清理或迁移节点前另行保存构建结果。

## LongMemEval-M：DeepSeek-Flash 与双卡 embedding 全新构建

`scripts/build_longmemeval_m_deepseek.sh` 使用
`configs/build_longmemeval_m_deepseek.yaml`，为固定划分的M训练315题、验证35题、测试150题
重新构建相邻语义聚类父子目录。输出隔离到
`runs/parent_child_directories/deepseek_flash_lmem_adjacent_v1`，不导入已有Qwen目录、分段或缓存。
后续重跑同一新配置，会从该DeepSeek实验自己的SQLite journal和缓存续跑。

语义聚类保持高/低阈值0.72/0.50、topic/event灰区提示词、连续覆盖原始block。
GPU4、5分别部署一个`models/qwen3-embedding-4b`副本（TP=1、每卡max-num-seqs=64）。
两个副本共享8个embedding HTTP请求并发，每请求最多32个文本且合计不超过16384 token；
调度轮转副本，已有缓存与同键请求合并保留。目录Writer远端请求并发默认64。
灰区判断、子摘要、父摘要均由`deepseek-flash`生成；父摘要在子目录完成后，结合子项及session原文生成。
标题24、子摘要300、父摘要400的长度上限继续按本地Qwen参考tokenizer计算，以保持目录预算一致。
配置里的`model_path: models/qwen3.5-27b`仅供CPU加载参考tokenizer，不部署27B推理服务；
它不代表DeepSeek实际用量，API计费token以响应`usage`为准。

接口使用官方`/chat/completions`、`thinking: {type: disabled}`、`response_format: {type: json_object}`，
不发送本地vLLM专用参数或未承诺的随机seed。JSON格式之外仍由本地执行schema、长度和原文边界校验。
传输层附加的JSON Schema计入输入预算；历史Writer trace的`input_tokens`是附加schema之前的参考计数，
完整HTTP request和API usage在writer traces中可核对。`length`响应触发原有输出预算扩展与校验，
空白非length响应、429及服务端暂时错误有限重试；耗尽重试后停止补采并保存失败状态，避免错误服务扫完整库。
HTTP 400/401/402立即失败；密钥仅从环境读取，不写入配置或日志。
DeepSeek模型名是服务端别名，construction identity如实记录别名和文档版本说明，不声称具有不可变权重hash。

```bash
cd /data/home/wtma/AG-mem
conda activate agmem

# 无付费调用、无GPU加载；只检查本地输入和计划。
bash scripts/build_longmemeval_m_deepseek.sh --dry-run

# 先在终端设置DEEPSEEK_API_KEY；可按需设置DEEPSEEK_BASE_URL。
# 正式运行前GPU4、5需空闲，脚本不会停止既有训练/服务。
mkdir -p runs/deepseek_directory_build
AGMEM_EMBEDDING_GPUS=4,5 AGMEM_WRITER_WORKERS=64 \
nohup /data/home/wtma/anaconda3/bin/conda run --no-capture-output -n agmem \
  bash scripts/build_longmemeval_m_deepseek.sh \
  >> runs/deepseek_directory_build/longmemeval_m.log 2>&1 &
```

正式启动先通过免费`GET /models`检查认证和模型名，再启动embedding服务。
`--limit-sessions 2`可用于小规模API试运行（会产生真实调用费用），完成后去掉该参数原命令续跑。
并发可通过`AGMEM_WRITER_WORKERS`调整，不改变新构建目录的协议；`--embedding-base-urls URL1 URL2`
可接入已经运行的embedding服务。脚本只清理本次创建的进程。

真实结构示例见`reports/longmemeval_m_deepseek_directory_example.json`及
`reports/longmemeval_m_deepseek_directory_example_storage_and_raw.json`。
示例是旧Qwen生成的一个父目录项和对应完整子目录，仅展示兼容结构，不是DeepSeek输出或导入材料。
整条query历史只有一个父目录，每个session对应一个父目录项和一个子目录。

## 全量父子目录可见的 ORM（每次最多两个 segment）

新配置`configs/train_dapo_locomo_lmes_full_directory_orm.yaml`及入口
`scripts/train_locomo_lmes_full_directory_2gpu.sh`使用独立的`hierarchical_full_v1` Reader协议。
冻结数据继续复用当前LoCoMo与LongMemEval-S父子目录，存储协议仍是`hierarchical_v1`。
每一步输入包括query、公开observation/options、完整父目录及所有对应子目录、已读ID、
保留的segment摘要和原文、动作反馈及剩余预算。父目录不再是可执行的导航动作。

READ输出`{"ids":["p0001:c0001","p0002:c0003"],"stop":false}`，允许一次读取1或2个
不同父目录下的叶子项；STOP必须为`{"ids":[],"stop":true}`。首次展示全部目录不消耗READ。
每次READ尝试最多消耗一次预算/成本，默认10次。混合请求返回有效、未读的ID，重复和无效位置
单独计罚；超量请求整体不执行任何ID，但仍消耗一次READ预算。容量失败不会标已读、不会
截断原文/目录。预算用完后仍请求READ只计预算违规，不继续执行或收取一次READ成本。

轨迹奖励采用`full_directory_orm_v1`，四项相加：

- `q = 1 if Judge == 1 else raw_word_F1`，无F1折扣、无是否取得证据的额外门控。
- `outcome_reward = 2*q - 1`。
- `format_reward`为整条轨迹的0/1值，所有输出协议合法才为1；JSON/schema、空READ、STOP携带ID、
  单次超过2个ID使其为0。不会对每个合法动作重复加分，也不乘旧的0.1 format系数。
- `cost_reward = -0.01 * READ次数`。
- `repetition_reward = -0.1 * 检索违规数`：已成功读过的ID、同批后续重复位置、未知/不可用/非叶子ID，
  每个位置最多计一次；预算耗尽后的READ每个动作额外计一次。首次成功读取不算重复。

最终为`reward = outcome_reward + format_reward + cost_reward + repetition_reward`。
JSON格式合法但ID无效时，format仍为1，仅检索违规项扣分；合法但无关的证据主要由outcome/cost评价。
不存在“未展开”窗口限制。诊断性的重复ID数和无效ID数可重叠，总扣分使用位置并集避免重复收费。

此协议明确关闭PRM和两种动作级强制负优势：所有参与loss的Reader输出token均使用原轨迹的DAPO组内优势。
旧导航式实验与其frozen action advantages仍可按原协议读取，但不能直接用新配置`--resume`。
依旧过滤Judge全0组及无训练优势信号组，按8个有效group更新；这意味着错答F1只影响混合组的排序。
训练、采样（含by_source）、验证、轨迹文件分别记录`outcome_q`、`outcome_reward`、`format_accuracy`、
`format_reward`、`cost_reward`、`repetition_reward`、`retrieval_violation_count`等独立字段。

新默认输出为`runs/locomo_lmes_full_directory_orm/train`，保留已配置的第一轮17个query前置对照顺序、
3个epoch、每4步验证及“最佳主指标＋最新”checkpoint保留规则。Task/Judge继续共享本地Qwen3.5-27B，
Judge使用放宽时间口径的LoCoMo模板与LongMemEval官方模板。

```bash
cd /data/home/wtma/AG-mem
conda activate agmem
# CPU校验冻结数据、模型文件、全部初始Reader输入token长度；不启动GPU服务。
bash scripts/train_locomo_lmes_full_directory_2gpu.sh --check-only

# GPU4/5空闲且磁盘空间足够时，从基础4B模型开始新的实验。
AGMEM_TRAIN_GPUS=4,5 \
AGMEM_TRAIN_CONFIG=configs/train_dapo_locomo_lmes_full_directory_orm.yaml \
AGMEM_TRAIN_OUTPUT_DIR=runs/locomo_lmes_full_directory_orm/train \
bash scripts/train_locomo_lmes_full_directory_2gpu.sh
```

全量目录不得静默截断。初始输入检查通过只证明满足模型上下文上限，不代表单GPU训练显存或后续
累积原文必然足够；运行时仍逐步检查Reader/Task完整输入容量。独立的旧单层checkpoint评测套件
保持原协议；本配置的训练内验证与`agmem --config ... evaluate`使用新协议。
