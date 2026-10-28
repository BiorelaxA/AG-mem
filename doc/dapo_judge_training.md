# DAPO、Judge reward 与训练监控

更新：2026-10-28。

当前四卡入口默认使用 `configs/train_dapo_locomo_s_v2.yaml`，训练 LoCoMo 和 LongMemEval-S。v2 对所有轨迹扣检索成本，只训练 Judge 有对有错的 group，并通过 `strict_v2` 处理重复读取。训练执行引擎是 AG-mem 的独立 DAPO 实现，继续使用现有两卡 Reader 分层模型并行、CPU FP32 AdamW 和多轮检索环境；没有引入 verl/Ray 运行时。旧 DAPO v1 与 GRPO 路径保留，可显式选择。

Reader、Writer、Task Agent 的 prompt 模板及 Judge 评判模板保持不变；新的重复读取反馈会作为工具执行结果进入后续 Reader 上下文。冻结目录继续复用。此变更不新增 Python 依赖，也不会重启已运行的训练。CPU 单元测试不等于真实 GPU 训练验证；新配置尚未进行实际 GPU 复测，批量处理是否提速及显存峰值需在分配的 GPU 上测量。

本次完整 unittest 共 236 项：233 项通过，3 项依赖 GPU 的测试跳过。新默认配置的 `python -m agmem.four_gpu --check-only` 已通过，核验输入为 LoCoMo train 1124 / validation 10 条、LongMemEval-S train 340 / validation 10 条；检查未加载 GPU 模型。

## 四卡分工与启动

在已申请的四张 A100 80GB GPU 作业内、项目根目录执行。入口沿用 Slurm 的 `CUDA_VISIBLE_DEVICES`，不会自行申请资源或重排物理卡号。

| 可见 GPU 的顺序 | 角色 |
|---|---|
| 第 1 张 | 单卡 Qwen3.5-27B Writer，后台构建其余来源的目录 |
| 第 2、3 张 | Qwen3.5-4B Reader，全参训练、分层模型并行 |
| 第 4 张 | 单卡 Qwen3.5-27B，分别处理 Task Agent、Judge 和启用时的 D 评分请求 |

只检查本地模型、配置和准备数据，不加载 GPU 模型：

```bash
conda activate agmem
python -m agmem.four_gpu --check-only
```

开始新的 DAPO 实验：

```bash
python -u -m agmem.four_gpu
```

只进行一次 Reader 参数更新并保存可恢复断点：

```bash
python -u -m agmem.four_gpu --max-updates 1
```

`--max-updates` 限制本次调用的参数更新次数；动态采样仍需先收集一个完整有效批次。独立 Writer 目录任务仍按配置运行，这个参数不限制其目录构建数量。

从此 DAPO 实验最新的已提交 checkpoint 继续：

```bash
python -u -m agmem.four_gpu --resume
```

默认训练输出：

```text
runs/locomo_longmemeval/experiments/locomo_s_dapo_mixed_v2/train
```

两个 27B 服务、训练和目录构建的进程日志分别放在 `runs/four_gpu/<jobid>-<timestamp>/`。新实验从配置中的初始 Qwen3.5-4B 权重开始。v2 改变了奖励、过滤及检索协议，应使用新的输出目录，不能用旧 v1/GRPO checkpoint 假装延续同一实验；普通 `--resume` 仅用于配置及协议匹配的同一实验。

旧 DAPO v1 实验使用显式配置，保留其门控成本、reward 差异过滤和 D reward 能力：

```bash
python -u -m agmem.four_gpu --train-config configs/train_dapo_locomo_s_4x80gb.yaml
```

旧 GRPO 实验使用显式配置：

```bash
python -u -m agmem.four_gpu --train-config configs/train_locomo_s_4x80gb.yaml
```

若该旧实验已经有 checkpoint，则在此命令末尾加 `--resume`。两条路径使用独立输出目录。

## 采样、更新与 step 的含义

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `candidate_groups` | 32 | 一轮生成的候选 query/group 数 |
| `group_size` | 8 | 同一 query、同一记忆快照下的完整 rollout 数 |
| `target_groups` | 16 | 一个行为策略批次保留的有效 group 数 |
| `mini_batch_groups` | 1 | 一次 optimizer step 使用的完整 query group 数 |
| `micro_batch_rollouts` | 1 | 一次真实训练前向/反向最多同时处理的 rollout 数，每条选一个待计算 decision |
| `max_micro_batch_tokens` | 16384 | 一次 micro 的 padding 后输入 token 总数预算；超长单条允许独立处理 |
| `ppo_epochs` | 1 | 每批轨迹只优化一遍 |
| `max_num_gen_batches` | 4 | 一批至多进行 4 轮候选生成 |
| `filter_metric` | `judge` | 仅保留二元 Judge 同时有 0 和 1 的 group |
| `filter_min_std` | `1e-6` | Judge 差异过滤的数值阈值，不是 advantage 标准差下限 |

首轮最多产生 `32 × 8 = 256` 条 rollout。按采样顺序保留前 16 个有效 group，每组保存完整 8 条轨迹，包括正、负 advantage；不做组内 top-k 奖励筛选。全对或全错 group 即使存在成本、格式差异也不进入训练。开启 overlong filtering 后，实际留下训练 token 的轨迹还必须同时包含 J=0 和 J=1；例如唯一答对轨迹被截断屏蔽，该组也会过滤。当前轮超出目标的有效 group 仍记录诊断，但不进入这批更新。

不足 16 组时继续补采。默认累计 4 轮仍不足则明确报错，并保留候选诊断；`max_num_gen_batches <= 0` 取消这个补采轮数上限，但数据集和配置中的 epoch 总数仍然有限。数据耗尽时不足目标的尾批会丢弃并记录 `dropped_tail_groups`；不会把不完整尾批静默当作 16 组训练。如果整个实验没有收集到任何完整批次，会明确失败。

采样期间 Reader 参数保持不变。组内 advantage 在各自完整 group 上计算后固定：

```text
A_i = (R_i - mean_group(R)) / (std_group(R) + epsilon)
epsilon = 1e-6
```

`std_group` 使用总体标准差。v2 不设置 advantage 分子上限、advantage 裁剪或标准差下限；下述 PPO/DAPO ratio clipping 仍保留。真实生成 `old_logprobs`、行为策略版本和批次 ID 一并保存。默认 mini=1 时该批 16 个 group 产生 16 次 optimizer step；mini=2 时产生 8 次。两张 Reader GPU 是模型并行，不能据此将 batch 数量乘以 2。

`mini_batch_groups` 与 `micro_batch_rollouts` 是独立参数：mini=2 将两道 query 的 16 条 rollout 累积到一次参数更新，各 query 的 advantage 仍独立计算。micro=2 在单次 batched forward/backward 中处理来自最多两条 rollout 的 decision；一条 rollout 有多次检索时仍需多个 block。它不表示每两条 rollout 就执行一次 optimizer step。整个 mini 共用有效 completion token 总数分母，只做一次梯度裁剪及 optimizer step。

micro 按待处理 decision 的输入长度组织，使用 attention mask 和独立 position IDs 隔离 padding；只有 completion token 参与 loss。预算按 `batch_size × padded_input_length` 计算，超过预算则拆小；单条 decision 已超过 16384 token 时单独运行，不截断其历史。这是控制合批的预算，不是上下文上限，也不能保证所有长样本都不会 OOM。默认 micro=1 保留逐 decision 的串行前向路径。

独立试验配置与命令如下；各自使用独立输出目录，默认配置不会自动切换为这些参数：

```bash
# mini=1，micro=2；输出 .../locomo_s_dapo_mixed_v2_micro2/train
python -u -m agmem.four_gpu --train-config configs/train_dapo_micro2_v2.yaml

# mini=2，micro=2；输出 .../locomo_s_dapo_mixed_v2_mini2_micro2/train
python -u -m agmem.four_gpu --train-config configs/train_dapo_mini2_micro2_v2.yaml
```

`target_groups` 必须能被 mini 整除；checkpoint 中已消费 group 游标也必须位于 mini 边界。对比不同 mini 时，应同时查看累计训练 group 和 token 数，不能仅对齐 optimizer step。

DAPO 更新使用：

- 非对称裁剪：`clip_ratio_low=0.2`，`clip_ratio_high=0.28`。
- Token-level loss：所有参与本次更新的 Reader completion token 使用同一个总 token 数分母；不包含输入 prompt、检索文本或 Task Agent 输出 token。
- `kl_coefficient=0`，不加载冻结的 Reader reference model。
- `overlong_filter=true`：Reader 某次生成因 token 上限截断时，将这条轨迹的训练 loss 屏蔽。它与普通格式错误、检索调用次数耗尽分开记录；该选项是 overlong loss filtering，没有额外引入 soft length reward。

每 **25 次实际 optimizer step** 执行 validation 并保存 checkpoint，mini=2 时也仍是 25 次更新。被过滤的候选 group 不增加 optimizer step。checkpoint 保存优化器、采样游标以及尚未消费完的行为策略批次，包括原始 logprobs、固定 advantage 和已消费 group 数，恢复后继续该批次，不重新计算剩余轨迹的行为概率。一个 mini 只推进一次 policy version，消费的 group 数按 mini 大小推进。

恢复时按 checkpoint 的两个游标整理产物：超出优化步的训练/验证记录，以及超出候选游标的采样记录，会先归档到 `resume_orphans/`。未完成的 `.building` checkpoint 也会归档。随后根据保留的 JSONL 重建 TensorBoard，避免重复统计或用优化步横轴误删已提交的采样曲线。尚未消费的 active buffer 候选属于已提交采样，会继续保留。

## Judge 与 reward

Judge 使用与 Task Agent 相同的固定 Qwen3.5-27B 权重和服务，独立请求、独立缓存角色；temperature 为 0，关闭 thinking。

| 数据来源 | Judge prompt |
|---|---|
| LoCoMo | LoCoMo-Refined 的 `refined` 原始模板 |
| LongMemEval-S／M | 官方模板，按 `question_type` 以及原始 `question_id` 中的 `_abs` 分支选择 |

来源、许可归属、冻结模板位于 `agmem/judge_prompts.py`。运行元数据记录模板内容 SHA256、模型指纹、解析协议和解码参数。官方 `yes/no` 或 `CORRECT/WRONG` 在代码内映射为 1/0，不改写官方模板的判分规则。LongMemEval 的私有评分元数据从经过来源校验的原始数据恢复，不进入 Reader、Writer、Task Agent 的输入。

此处的 LoCoMo 数据保持原样，实验标注为“LoCoMo 数据＋Refined Judge prompt＋Qwen3.5-27B”，不冒充 LoCoMo-Refined 官方完整配置。训练、validation 和 test 对同一来源使用相同 Judge 协议。

默认 `reward.formula: judge_cost_v2`：

```text
R = 2 * J - 1 - 0.01 * read_calls - 0.1 * format_failure
```

`J` 为二元语义正确性分数；检索成本无 J 门控，答对、答错都按实际读取请求数扣除。每条轨迹最多扣一次格式惩罚。`answer_f1` 继续作为独立词面指标，兼容字段 `outcome_f1` 也仍表示 F1，不会改成 Judge 分数；`outcome_reward` 是 `2J-1`，总 `reward` 再减成本和格式惩罚。`judge_accuracy` 单独记录，并用于验证集宏平均的 `best_outcome` checkpoint 选择；`best_reward` 仍按验证宏平均 reward 选择。

`retrieval.id_protocol: strict_v2` 要求复制目录中的精确 ID，保留前导零及哈希，不自动修复。跨调用再次请求已经成功读过的 ID 记录 `repeated_read`，与已有 JSON/schema、空 ID、同批重复、未知 ID 和超量 ID 一起计入 `format_failure`；不额外增加另一项重复次数惩罚。

一次请求同时包含旧 ID 和新 ID 时，仅返回允许前缀内的新 ID 原文，保留已读证据，且仍记录重复读取违规。全是旧 ID 时不再次读取原文，反馈 `already_read_ids`、`new_ids`、`no_new_evidence`、`read_calls_used` 和 `read_calls_remaining`，提醒复用已读证据或停止。空、未知或全重复的读取请求在预算未耗尽时仍消耗一次调用；合法 stop 不消耗读取调用。重复读取不会清除已有证据，也不会仅凭一次重复就把该 rollout 丢弃。

Judge 超时、截断或标签解析失败是基础设施错误，不记成答错。输出通过严格解析后才能写入缓存；失败重试耗尽时使整组失败，避免由服务故障制造奖励差异。

Task Agent 的非空回答达到输出上限（`finish_reason=length`）时，保留实际文本并交给
Judge 正常评分，继续收集完整的 8 条 rollout，不因此丢弃轨迹或重试同一回答。
该规则同时应用于训练和验证。轨迹及 `candidate_rollouts` / `rollouts` 中的
`task_completion` 保存 `truncated`、`finish_reason`、`usage`、`max_tokens`；缓存命中和
checkpoint 恢复也保留这些字段。TensorBoard 的 `train/task_truncation_rate` 和
`validation/task_truncation_rate` 记录 Task 截断比例，过滤前候选另记录
`task_truncated_rollout_fraction`，与 Reader 的 `truncated_rollout_fraction` 分开。
Reader 的 overlong loss mask 只看 Reader 自己的生成截断，不屏蔽 Task 回答截断的轨迹。

空 Task 回答、Judge 标签截断或格式错误、HTTP 失败及服务 OOM 仍按评分失败处理，
按既有次数重试后退出训练，不将错误伪装成 J=0，也不跳过损坏的 group。
Task 输出策略以 `score_nonempty_truncated_v1` 写入评分协议和版本化缓存。
旧 Task 文本缓存会重新生成，Judge/Writer 缓存不受影响；旧 DAPO checkpoint 的评分协议
与该策略不同，恢复校验会拒绝混用，应在新的输出目录开始实验。

## 独立的 D 辅助项

`reward.likelihood.mode` 默认是字符串 `"off"`。当前 v2 支持关闭或仅记录；旧 v1 为复现实验保留 D 加入 reward 的能力：

| 模式 | 行为 |
|---|---|
| `off` | 不加载评分 tokenizer、不创建 likelihood HTTP 客户端、不执行评分前向或基线计算；D 记为未计算 |
| `diagnostic` | 计算并记录 D，但不加入 reward，也不改变采样过滤结果 |
| `reward` | 仅旧 `judge_gated_v1` 支持；新 `judge_cost_v2` 会拒绝该配置 |

默认命令已关闭 D 计算。只记录 D、仍使用同一 v2 奖励与混合 Judge 过滤规则时，使用独立配置：

```bash
python -u -m agmem.four_gpu --train-config configs/train_dapo_diagnostic_v2.yaml
```

输出至 `runs/locomo_longmemeval/experiments/locomo_s_dapo_mixed_v2_diagnostic/train/`。这会增加 Task 服务的 likelihood 请求及显存开销；D 不进入 reward、advantage 或 group 有效性判定。

对完整标准答案 \(y\)，计算冻结 Task Agent 在实际任务上下文 \(C_i\) 下的平均条件 logprob：

\[
\ell_i = \frac{1}{|y|}\sum_t \log p_{\mathrm{Task}}(y_t \mid C_i,y_{<t}),
\qquad D_i = \tanh\left(\frac{\ell_i-\ell_0}{\tau}\right).
\]

\(C_i\) 复用实际 Task Agent 的 system prompt、query、observation、options 和检索记忆；\(\ell_0\) 保留相同输入，只把检索记忆置空。只统计标准答案 continuation token，不统计上下文或模板 token，概率温度为 1。多个完整 gold 先各自按答案 token 求平均，再等权平均各 gold；两个上下文使用相同的 gold token 序列。评分结果及无记忆基线可缓存复用。

以下公式只解释保留的旧 v1 实验：在 `judge_gated_v1` 中启用 `reward` 模式后：

```text
R = 2 * J - 1 - 0.01 * J * read_calls - 0.1 * format_failure
    + coefficient * (1 - J) * D
```

旧 v1 默认 `coefficient=0.1`、`tau=1.0`；其 `reward` 模式使用 `reward.likelihood.advantage_std_floor=0.1`。这些历史设计不应用于新 v2：v2 的 D 默认不计算，diagnostic 只记录，reward 模式被拒绝，advantage 没有标准差下限。切换评分模式会改变实验协议，应使用单独配置和新的输出目录。

开启 `diagnostic` 或 `reward` 时，`agmem.four_gpu` 自动为 Task 27B 使用
`--max-num-batched-tokens 1024 --gpu-memory-utilization 0.80 --max-num-seqs 1`，
并启用 chunked prefill，保留 65536 token 的上下文上限。vLLM 0.22.1 即使使用
`prompt_logprobs=0`，仍会为 prefill chunk 计算全词表 FP32 log-softmax；Qwen3.5
在 4096 token 的 chunk 下仅该临时张量就约需 3.79 GiB，可能超出默认 KV 缓存分配后的
空闲显存。1024 token 的 chunk 将其降至约 0.95 GiB，并通过降低缓存预算预留更多工作显存。
此设置可能降低服务吞吐；D 仍按完整上下文评分。Writer 保持原有服务预算。
实际服务参数保存在 launcher 日志目录的 `task_server_limits.json` 和 `writer_server_limits.json`。

## TensorBoard：entropy 与过滤前诊断

```bash
tensorboard --logdir runs/locomo_longmemeval/experiments/locomo_s_dapo_mixed_v2/train/tensorboard --host 127.0.0.1 --port 6123
```

`train/entropy` 是训练前向中实际计算的全词表分布熵：

\[
H_t = -\sum_v p_\theta(v\mid s_t)\log p_\theta(v\mid s_t).
\]

单位为 **nats**，对本次更新未被屏蔽的 Reader completion token 按 token 数加权平均，在执行 optimizer step 前计算，以该次更新的 step 写入。它不是所选 token 的负 logprob；后者单独记为 `train/sampled_surprisal`，行为策略对应值为 `train/behavior_sampled_surprisal`。`entropy_chunk_size=32` 控制按 token 位置分块计算熵，以限制额外临时显存，不近似词表分布。

同时记录 `train/judge_accuracy`、`train/answer_f1`、`train/reward`、`train/clip_fraction`、`train/grad_norm` 等。验证曲线在 `validation/*`，来源宏平均在 `validation/source_macro/*`。

新增读取行为统计包括 `repeated_read_rate`（包含重复读取的轨迹比例）、`repeated_read_calls`（每条轨迹平均重复调用数）、`repeated_read_call_rate`（读取调用中重复的比例）、`no_new_evidence_calls` 和 `no_new_evidence_call_rate`，分别写入对应 train/validation/sampling 统计。与 `format_failure_rate`、`judge_accuracy` 一起检查，避免把“重复次数减少”单独当作正确率提升。

`train/cumulative_trained_groups` 记录累计已消费 group；`train/update_groups`、`train/update_rollouts` 和 `train/training_tokens` 分别记录本次更新的 group、完整 rollout 和实际参与 loss 的 completion token 数。`training_tokens` 排除 overlong mask 的轨迹及所有 prompt token。`micro_batches` 表示实际 decision forward/backward 次数；`micro_batch_input_tokens`、`micro_batch_padding_tokens`、`max_micro_batch_input_tokens` 和 `micro_batch_over_budget_singletons` 可用于检查合批及 token 预算是否生效。增大 micro 只改变执行分块；增大 mini 会改变每次更新的样本数与优化步数，应据此比较实验。

`sampling/*` 的横轴是累计候选 group 数，与 optimizer step 分开。以下指标在过滤前统计，包含被拒绝的 group：

- `sampling/judge_all_correct_rate`、`judge_all_wrong_rate`、`judge_mixed_rate`。
- `sampling/zero_reward_variance_rate`：最终 reward 无可靠组内差异的比例。
- `sampling/cost_format_only_signal_rate`：Judge 同分，仅成本或格式提供差异的比例。
- `sampling/auxiliary_only_signal_rate`：启用 D 后仅辅助项提供差异的比例。
- `sampling/no_unmasked_signal_rate`：屏蔽截断轨迹后缺乏有效更新信号的比例。
- `sampling/no_unmasked_judge_signal_rate`：屏蔽 Reader 截断轨迹后，实际有训练 token 的轨迹不能同时提供 J=0 和 J=1 的比例；v2 拒绝这些 group。
- `sampling/accepted_rate`、`generation_batches`、`candidate_groups`、`selected_groups`。
- `sampling/source_candidates/*`：当前采样批次累计的各来源候选数量。

权威文本记录为 `train_metrics.jsonl`、`sampling_metrics.jsonl`、`candidate_metrics.jsonl`。`candidate_rollouts/` 保留所有候选的精简动作和回答，包括被过滤的候选；不能仅凭进入训练后的 group 判断 Judge 是否产生大量同分轨迹。

参考：[DAPO 论文](https://arxiv.org/html/2503.14476v1)、[LoCoMo-Refined Judge](https://github.com/mem-eval-suite/LoCoMo_refined/blob/main/src/llm_judge.py)、[LongMemEval 官方评测](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py)。
