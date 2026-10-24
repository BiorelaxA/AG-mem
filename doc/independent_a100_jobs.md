# 独立的全量目录任务与 LoCoMo/S 训练评估任务

目录任务申请 2 张 `a100-pcie-40gb`，时间沿用当前脚本的 96 小时；训练评估任务申请 4 张 `a100-pcie-40gb`，时间 72 小时。两者均申请 CPU 16、内存 120G。资源数量独立计费/排队，同时运行共需 6 张 GPU。未自动提交任务，也未中断旧任务。

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

2 张 A100 40GB 卡以 TP=2 部署冻结 Qwen3.5-27B，不加载 4B Reader、不创建优化器，也不执行训练或问答评估。Writer 默认并发 2；内容规则仍是当前 session + 前 3 个原始 session，最多输出 300 tokens，长度截断接受，空正文保留 empty_pending。

共用现有 `runs/locomo_longmemeval/prepared/<source>/`，已有摘要按原版本复用。新来源如果尚无 manifest，会尝试从 `runs/slurm_full_train/prepared` 导入同来源的已准备数据/目录；旧 full_train 仍持锁时跳过导入，按 processed Parquet 新建，而不是读取正在修改的旧文件。导入会核对输入 manifest、适配器和 split provenance，不静默混用不同数据。

## LoCoMo/S 训练评估任务

配置为 `configs/train_eval_locomo_s.yaml`，固定 1,464 train、20 validation；test 的数量由已有 processed 数据实际适配得到。

1. 第 1、2 张卡运行 4B Reader，第 3、4 张卡以 TP=2 部署 27B。
2. 复用 LoCoMo/S 的已完成准备结果和目录，只补未完成项。目录完整后立即训练，不等待其他来源，也不提前补建 test。
3. 执行 1 epoch、group size=8 的全参 GRPO，每步保存 checkpoint，每 50 步验证。Reader 默认批大小 2，Task Agent 并发 2。
4. 训练结束后准备两来源的 test 及其目录，使用固定 validation 的 source_macro.reward 最佳 checkpoint 完成全量 test 评估；没有验证 checkpoint 时回退到最新模型并记录原因。保留最近两个及 validation reward/F1 最佳模型，未训练基线不在此脚本内自动运行。
5. `metrics.json` 包含 pooled、by_source、source_macro 指标，分别可查看 LoCoMo 和 LongMemEval-S 的结果。

新训练采用 `2*F1-1-0.01*F1*read_calls-0.1*format_failure`，写入独立 `locomo_s_reward_v2` 实验，复用已有目录，避免接续已退化的旧 reward 实验。详见 [奖励与 checkpoint 更新](reward_outcome_gated_20261024.md)。

训练会按最新 checkpoint 自动续跑；没有完整 checkpoint 的中断训练目录会归档到本次 job 目录后从初始 Reader 重启。有 train.complete 时不重训；测试中断后保留不完整输出并重新评估，有完整 metrics.json 则复用。改变数据、模型或奖励等训练配置不能冒充旧实验续跑。

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
runs/locomo_longmemeval/experiments/locomo_s_reward_v2/train/checkpoint-*/
runs/locomo_longmemeval/experiments/locomo_s_reward_v2/train/rollouts/
runs/locomo_longmemeval/experiments/locomo_s_reward_v2/train/tensorboard/
runs/locomo_longmemeval/experiments/locomo_s_reward_v2/evaluation/checkpoint-*/test/metrics.json
```

并发参数继续使用 `AGMEM_WRITER_WORKERS`、`AGMEM_TASK_AGENT_WORKERS`、`AGMEM_READER_BATCH_SIZE`、`AGMEM_READER_BATCH_MAX_TOKENS`。vLLM max-num-seqs 与 Writer/Task 并发容量同步；Reader 长输入按批预算拆批，不截断原文。

```bash
AGMEM_WRITER_WORKERS=2 sbatch scripts/build_directories_all_a100.sbatch
AGMEM_WRITER_WORKERS=2 AGMEM_TASK_AGENT_WORKERS=2 AGMEM_READER_BATCH_SIZE=2 \
sbatch scripts/train_eval_locomo_s_a100.sbatch
```

`AGMEM_PREPARED_ROOT` 可覆盖共享 train/validation 数据根目录，其父目录下派生 prepared_test 和 experiments；两个任务需要指向同一个根才能复用。`AGMEM_JOB_ROOT` 只覆盖作业日志/锁根目录，不改变训练输出位置。`AGMEM_SERVICE_PORT` 可指定实际部署端口；不指定则自动选择，不会更改逻辑缓存 URL。

本次包含 CPU 回归与调度顺序测试、shell/内嵌 Python 语法检查；未申请 GPU 实测这两个新作业，也未确认并发吞吐提升。
