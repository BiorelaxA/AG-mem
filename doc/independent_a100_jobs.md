# 独立的全量目录任务与 LoCoMo/S 训练评估任务

目录任务申请 1 张 `a100-sxm4-80gb`，时间沿用当前脚本的 96 小时；训练评估任务申请 4 张 `a100-pcie-40gb`，时间 72 小时。两者均申请 CPU 16、内存 120G。资源数量独立计费/排队，同时运行共需 5 张 GPU。未自动提交任务，也未中断旧任务。

```bash
cd /home/hongyshen/AG-Mem

# 从 LongMemEval-M 开始构建其余七来源的 train/validation/test 目录，只部署 27B。
sbatch scripts/build_directories_all_a100.sbatch

# 只补齐 LoCoMo/S → 训练 → test 评估。
sbatch scripts/train_eval_locomo_s_a100.sbatch
```

新入口不依赖彼此完成，可以分别提交。旧 pilot 任务仍运行时持有独占锁，新任务会拒绝启动；需要先让旧任务退出。不要与旧的 full_train/pilot 构建脚本一起写同一 prepared 数据。两个新入口的所有协调锁由脚本/调度模块负责，直接运行 CLI 不会自动获得这些工作流锁。

## 全量目录任务

配置为 `configs/directory_all.yaml`，负责除 LoCoMo/S 外的七个来源。执行顺序为 LongMemEval-M、PerLTQA EN/ZH、PersonaMem 128k/32k/v2、ZH4O。LoCoMo/S 完全由训练评估任务构建，本任务不会补建或等待这两个来源。每个来源先完成 train/validation，再独立准备 test，生成其所需目录。

1 张 A100 SXM4 80GB 卡以 TP=1 部署冻结 Qwen3.5-27B，不加载 4B Reader、不创建优化器，也不执行训练或问答评估。Writer 默认并发 2；内容规则仍是当前 session + 前 3 个原始 session，最多输出 300 tokens，长度截断接受，空正文保留 empty_pending。

共用现有 `runs/locomo_longmemeval/prepared/<source>/`，已有摘要按原版本复用。新来源如果尚无 manifest，会尝试从 `runs/slurm_full_train/prepared` 导入同来源的已准备数据/目录；旧 full_train 仍持锁时跳过导入，按 processed Parquet 新建，而不是读取正在修改的旧文件。导入会核对输入 manifest、适配器和 split provenance，不静默混用不同数据。

## LoCoMo/S 训练评估任务

配置为 `configs/train_eval_locomo_s.yaml`，固定 1,464 train、20 validation；test 的数量由已有 processed 数据实际适配得到。

1. 第 1、2 张卡运行 4B Reader，第 3、4 张卡以 TP=2 部署 27B。
2. 复用 LoCoMo/S 的已完成准备结果和目录，只补未完成项。目录完整后立即训练，不等待其他来源，也不提前补建 test。
3. 执行 1 epoch、group size=8 的全参 GRPO，每步保存 checkpoint，每 50 步验证。Reader 默认批大小 2，Task Agent 并发 2。
4. 训练结束后准备两来源的 test 及其目录，使用固定 validation 的 source_macro.reward 最佳 checkpoint 完成全量 test 评估；没有验证 checkpoint 时回退到最新模型并记录原因。保留最近两个及 validation reward/F1 最佳模型，未训练基线不在此脚本内自动运行。
5. `metrics.json` 包含 pooled、by_source、source_macro 指标，分别可查看 LoCoMo 和 LongMemEval-S 的结果。

新训练采用 `2*F1-1-0.01*F1*read_calls-0.1*format_failure`。修复提问日期与时间回答后，默认写入独立 `locomo_s_reward_v2_time_v1` 实验，首次提交从原始 Qwen3.5-4B 开始，复用已有目录；旧 `locomo_s_reward_v2` 实验保留供比较，不加载其 checkpoint。新实验中断后仍从新目录的最新 checkpoint 续跑。详见 [奖励与 checkpoint 更新](reward_outcome_gated_20261024.md)。

训练会按最新 checkpoint 自动续跑；没有完整 checkpoint 的中断训练目录会归档到本次 job 目录后从初始 Reader 重启。有 train.complete 时不重训；测试中断后保留不完整输出并重新评估，有完整 metrics.json 则复用。改变数据、模型或奖励等训练配置不能冒充旧实验续跑。

### 问题日期和时间回答（2026-10-24）

LongMemEval-S/M 的 `question_date` 进入 Reader 与 Task Agent 的公开 observation。
新 `prepare` 会直接保存该字段对应的 observation；既有 prepared 数据无需重新构建目录，训练和评估会依据样本的 source/split/原始行号从 Parquet 补齐公开日期。加载时核对输入 manifest、Parquet 哈希和问题/答案对应关系，拒绝错配；只读取 question/answer/question_date 标量列，答案仅用于一致性检查，不加入 prompt。旧 JSONL 和历史 ID 不修改。

Task Agent 回答时间题时输出简洁的“绝对日期或范围 + 相对时间表达”，按证据中的 session `time` 解释历史的 yesterday/last week，按 observation 的提问日期计算多少天前；缺少锚点则说明不足，不猜测具体日期。选择题仍只输出选项标签。此修改只改变任务背景和 Task prompt，不改变 Writer、reward 或 F1 实现；措辞增加也不保证文本 F1 单调提高。

Task 服务缓存键包含完整 messages，因此新 prompt/日期自动与旧回答缓存隔离。任务上下文版本和 prompt 指纹保存在 run.json、checkpoint 和新评估报告中；从旧 checkpoint 继续时记录 `task_input_contract_changed` 和 `task_input_changes.jsonl`，保留优化器、step 和数据顺序。修改前后的指标需要按此边界解读。正在运行的 Python 进程不会热加载本次修改，需要结束旧作业后重新提交才会使用新逻辑；已有历史 rollout/评估结果不会自动重算。

### Reader 双卡分层

训练脚本仅向 Reader 进程暴露前两张分配卡，并设置 `AGMEM_READER_MODEL_PARALLEL_SIZE=2`。
Reader 使用单进程模型并行：前半 decoder layers 放在逻辑 `cuda:0`，后半放在 `cuda:1`；embedding、最终 norm 和 lm_head 在第一张卡。
层入口通过可微的 Tensor 设备传输处理 hidden states、位置编码和 mask，生成缓存按各层所属设备保存。反向传播跨两张卡，全参 CPUAdamW 和梯度检查点继续启用。
冻结的 KL reference 仍从 CPU 临时移至第一张训练卡计算无梯度概率，再卸载回 CPU。

这不是两个独立训练副本，也不改变 group size=8、reward 或梯度累积规则。训练日志会输出 `reader_model_parallel` 及各层的设备，模型保存格式和优化器参数顺序保持兼容，可恢复已有单卡 checkpoint。双卡部署属于运行时设置，不修改旧实验冻结的训练超参数；不设置该环境变量时，其他单卡入口保持原逻辑。测试评估同样使用双卡 Reader。

两张卡分担参数、梯度与各层激活，但不构成一个统一的 80GB 显存池，不保证任意长上下文都能训练。CPU 小模型测试覆盖缓存生成、梯度检查点反向传播、优化器恢复和模型保存。提交脚本在启动 27B 前会先用随机小模型执行两卡跨设备测试，失败即退出，日志为本次 job 下的 `reader_parallel_check.log`。也可在已分配的 GPU 上手动检查：

```bash
# 只暴露分配给 Reader 的两张卡；不要使用正被其他任务占用的卡。
python -m unittest tests.test_model_parallel -v
```

### 在已有 allocation 内恢复长序列 OOM

训练入口现在默认启用可扩展显存分配（未显式设置 allocator 时），以及输入达到 16,384 tokens 的激活 CPU 卸载。两项设置无需修改旧 job 的冻结配置，也不改变 checkpoint 权重、优化器状态或续训位置。依然使用两张卡训练 Reader、两张卡服务 27B。

在保留了原 allocation 和 27B 服务的 tmux/gpu16 会话中，可以显式指定：

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS=16384
```

继续用原实验配置运行 `python -m agmem.jobs train-eval --config <原配置> --job-dir <新的日志目录>`，调度器从该实验最近的完整 checkpoint 恢复，不重新申请 GPU。先确认旧 Reader 已退出，服务地址仍可访问，并保留 `CUDA_VISIBLE_DEVICES=0,1` 和 `AGMEM_READER_MODEL_PARALLEL_SIZE=2`。不要再次启动一份 vLLM，也不要取消仍需复用的 allocation。需要重新启动训练进程才能应用内存设置。

阈值 `0` 关闭激活卸载，`1` 对所有更新输入启用。卸载使用非锁页主机内存，避免大量激活与 CPU Adam 同时占用锁页内存；代价是 CPU/GPU 传输开销。更新日志记录卸载决策数、每条轨迹反向完成进度和成功更新的双卡峰值显存，OOM 则记录具体 rollout/decision 及其输入长度。日志没有宣称已通过 4B/长上下文 GPU 压测，续训时需验证原失败 group 并监控主机内存。新增双卡卸载一致性测试也会随脚本原有的 `tests.test_model_parallel` 检查运行。

## 测试集隔离与缓存复用

为了兼容已存在的 train/validation preparation identity，不把 test 直接追加到原 store。测试集保存到 `runs/locomo_longmemeval/prepared_test/<source>/`。

测试历史与 train/validation 的完整 memory_id、全部清洗 records 完全一致时，可以复制成功摘要；不同完整历史不因共享部分 session 而复制目录。测试 query 和标签只写入 test store，不会进入训练集。之后 Writer 只生成缺失项。跨 split 的相同完整历史可复用目录，并不代表进行了历史隔离；当前实验仍沿用原 query 去重划分。

## 同时执行与端口

- 两个新任务共享旧 pilot 的 `.job.lock`（共享模式），与旧 pilot 的独占锁互斥。
- 每个来源使用 `prepared/.locks/<source>.lock` 独占锁，保护 prepare、目录生成和对应 test store。
- 训练评估任务从补建目录到评估结束一直持有 LoCoMo/S 的来源锁，避免读取期间目录变化。
- 两个任务默认处理的来源不重叠，目录任务完成七个来源后退出，不等待 LoCoMo/S。若其他任务占用目录任务负责的来源，则先处理其余来源，最后每 30 秒重试被占用的来源。

每个任务自动选择自己的本机可用服务端口，避免两个任务恰好分配到同一节点时端口冲突。`service.request_base_url` 仅决定 HTTP 实际连接地址；`service.base_url` 保留既有逻辑身份 `http://127.0.0.1:50543/v1`，与模型指纹共同构成缓存身份，所以更换部署端口不重建目录。请求仍发送到本任务部署的同一指定模型。该传输地址不写入目录服务身份。

## 输出和调整参数

全量 Writer 日志：

```text
runs/directory_all/jobs/<job-attempt>/job.log
runs/directory_all/jobs/<job-attempt>/vllm.log
runs/directory_all/jobs/<job-attempt>/pipeline.log
```

训练评估日志：

```text
runs/train_eval_locomo_s/jobs/<job-attempt>/job.log
runs/train_eval_locomo_s/jobs/<job-attempt>/vllm.log
runs/train_eval_locomo_s/jobs/<job-attempt>/pipeline.log
runs/train_eval_locomo_s/jobs/<job-attempt>/train.log
runs/train_eval_locomo_s/jobs/<job-attempt>/test.log
```

持久结果：

```text
runs/locomo_longmemeval/experiments/locomo_s_reward_v2_time_v1/train/checkpoint-*/
runs/locomo_longmemeval/experiments/locomo_s_reward_v2_time_v1/train/rollouts/
runs/locomo_longmemeval/experiments/locomo_s_reward_v2_time_v1/train/tensorboard/
runs/locomo_longmemeval/experiments/locomo_s_reward_v2_time_v1/evaluation/checkpoint-*/test/metrics.json
```

并发参数继续使用 `AGMEM_WRITER_WORKERS`、`AGMEM_TASK_AGENT_WORKERS`、`AGMEM_READER_BATCH_SIZE`、`AGMEM_READER_BATCH_MAX_TOKENS`。vLLM max-num-seqs 与 Writer/Task 并发容量同步；Reader 长输入按批预算拆批，不截断原文。

```bash
AGMEM_WRITER_WORKERS=2 sbatch scripts/build_directories_all_a100.sbatch
AGMEM_WRITER_WORKERS=2 AGMEM_TASK_AGENT_WORKERS=2 AGMEM_READER_BATCH_SIZE=2 \
sbatch scripts/train_eval_locomo_s_a100.sbatch
```

`AGMEM_PREPARED_ROOT` 可覆盖共享 train/validation 数据根目录，其父目录下派生 prepared_test 和 experiments；两个任务需要指向同一个根才能复用。`AGMEM_JOB_ROOT` 只覆盖作业日志/锁根目录，不改变训练输出位置。`AGMEM_SERVICE_PORT` 可指定实际部署端口；不指定则自动选择，不会更改逻辑缓存 URL。

本次包含 CPU 回归与调度顺序测试、shell/内嵌 Python 语法检查；未申请 GPU 实测这两个新作业，也未确认并发吞吐提升。
