# 并发目录构建与 GRPO 采样

本次只改变执行调度，不改变 Writer 提示、300-token 上限、当前 session 加前 3 个原始 session 的窗口、历史 ID、目录版本或 reward 定义。默认配置为：

```yaml
execution:
  writer_workers: 2
  task_agent_workers: 2
  reader_batch_size: 2
  reader_batch_max_tokens: 16384
```

没有 `execution` 的历史配置，直接用 `python -m agmem --config ...` 执行时保持串行；使用新版 sbatch 时会创建独立的本次作业配置，缺失的并发配置默认补为 2。三个并发数都设为 1 即恢复串行调度。`reader_batch_max_tokens` 必须为正整数。

## Writer

仍按所选数据集、历史顺序推进；一条历史内最多同时请求 `writer_workers` 个 session。前文来自原始清洗历史，不等待前文摘要，所以这些请求可独立执行。只保留有界的在途任务，避免一次创建十几万个请求。

主线程准备输入并保存目录项，HTTP 请求在线程池执行；完成一项就保存一项。`directories.jsonl` 的物理追加顺序可能改变，但每行的历史 ID、session ID、order 不变；发布与读取完整目录始终按原 session 顺序。成功目录项直接复用，空正文仍保存 `empty_pending`，下次补写。

遇到基础设施错误时停止提交新任务，收齐并保存当前在途任务的成功结果后报错，不发布不完整历史。下次运行复用已写目录与成功 HTTP 响应缓存。并发 HTTP 缓存读写加锁；相同服务实例中完全相同的请求共享一个在途响应，不重复请求 Task Agent。

这些锁只保护一个进程内的线程，不支持两个独立 `build-directories` 进程同时写同一个 prepared store。sbatch 保留原有作业锁。

## GRPO

每个 group 仍是同一 query 的 8 条完整 rollout，并使用固定的同一 Reader policy_version：

1. 每次最多取 `reader_batch_size` 条轨迹，分别维护 evidence、错误反馈、读取次数和结束状态。
2. 将仍在进行的轨迹的下一次 Reader 决策合并推理；同一条轨迹仍必须先执行工具，再构造下一步输入。
3. 某一批轨迹完成后，将 Task Agent 回答与评分交给最多 `task_agent_workers` 个工作线程，可与下一批 Reader 检索重叠。
4. 8 条检索轨迹及对应评分全部成功后，按原 rollout_index 排列，计算 advantage、保存轨迹并调用一次 GRPO update。基础设施失败仍重试整个 group，不赋一个人为的错误答案 reward。

Reader 模型始终由主线程调用，不使用多个线程同时调用同一个 GPU 模型。反向传播、CPU Adam 和 reference logprob 计算仍逐条执行，未实现多 GPU 梯度并行；此次主要加速 rollout 收集和远端评分。

批量输入左侧补齐，并提供 attention_mask。完整审计保留未补齐的 prompt、真正生成的 token（包括 EOS）及其采样时 logprob；已结束样本后续强制 padding 不进入训练损失。格式非法时的终止/反馈、stop=true 忽略 ids、成本和格式惩罚均保持原规则。

批次预算按 `(最长 prompt token 数 + 每次输出上限) × 批大小` 计算，超出时拆成更小批次。单条输入超过批预算但未超过模型上下文限制时，仍单独生成，不截断目录或证据。该预算仅控制合批，并不是显存保证；OOM 时应减小 Reader batch size，单条过长输入仍可能 OOM。

一个真实推理批次使用由有序决策 seeds 推导的联合随机种子，完整 `groups/*.json.gz` 中每个 Completion 新增 `sampling_seed`、`sampling_batch_size`。批大小或预算改变后，采样结果可能改变，不承诺与串行逐 token 一致；单条推理沿用原决策 seed。行为 logprob 始终来自实际生成 scores，不重新启用采样 logprob 偏差阈值检查。改变调度可续训已提交 checkpoint，会记录 `resume_execution_changed`；复现实验必须保存实际作业配置。

验证仍使用原来的逐样本贪心流程，不并发更新参数。精简 `rollouts/*.json` 仍只保留 Reader 输出/动作和 Task Agent 答案等字段，不加入原文。

## sbatch 使用与续跑

A100、A6000、H100、pilot 和 smoke 脚本根据 Writer/Task Agent 并发数设置 vLLM `--max-num-seqs`，取两者最大值；这两个阶段使用同一个冻结的 27B 服务，但不会同时构建目录和训练。资源申请不因本次修改改变。

```bash
# 默认各为 2，先小规模确认显存和吞吐。
sbatch scripts/train_pilot_a6000.sbatch

# 提高 Writer 并发；Reader 仍用小批次。
AGMEM_WRITER_WORKERS=4 AGMEM_TASK_AGENT_WORKERS=2 \
AGMEM_READER_BATCH_SIZE=2 AGMEM_READER_BATCH_MAX_TOKENS=16384 \
sbatch scripts/train_pilot_a6000.sbatch

# 串行回退（同样可用于 smoke.sbatch 等脚本）。
AGMEM_WRITER_WORKERS=1 AGMEM_TASK_AGENT_WORKERS=1 AGMEM_READER_BATCH_SIZE=1 \
sbatch scripts/train_pilot_a6000.sbatch
```

全量/pilot 每次启动生成 `RUN_DIR/jobs/<attempt>/config.yaml`，保留原始 `RUN_DIR/config.yaml`；只有 execution 设置允许通过上述环境变量覆盖。服务 URL、模型、Writer 配置、数据根目录保持一致，已生成摘要可继续复用。smoke 对应 `RUN_DIR/execution_config.yaml`，原有 smoke 仍要求新的运行目录。

新作业日志记录实际配置路径及 vLLM 容量；`directories.log` 记录 `directory_execution`，`train.log` 记录 `training_execution`，训练 `run.json` 和 checkpoint 包含实际调度配置。直接启动外部 vLLM 服务时，也需将 `--max-num-seqs` 调到不低于客户端 Writer/Task Agent 并发数，否则客户端并发未必提升吞吐。

已有运行中的 Python 进程不会热加载这些更改。本次不修改 runs 中正在使用的冻结配置，不自动取消或重提任务；下次启动使用新代码与配置覆盖。不要为同一个 RUN_DIR 同时启动两个任务。吞吐提升与请求长度、KV cache、GPU 利用率有关，不能简单按并发倍数推算剩余时间。
