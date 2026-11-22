# DAPO、Judge reward 与训练监控

更新：2026-11-04。

当前四卡入口默认使用 `configs/train_dapo_reader_loop_v3_query9010_format05.yaml`，继承原逐 ID 惩罚配置 `configs/train_dapo_reader_loop_v3_query9010_id_cost.yaml`，单次检索最多 1 个 ID，训练、验证和测试仅使用 LoCoMo。默认使用 `reader.prompt_version: evidence_loop_v3`，明确每次 READ 返回后会再次决策、整题最多 8 次 READ，并始终展示已读 ID、对应证据及剩余调用次数；采样使用最多 4 个 group 的固定窗口，每题生成 8 条 rollout，以 mini=8、micro=2 更新，每 4 个 optimizer step 验证。

奖励保留 `read_call_cost=0.01` 和 `format_penalty_mode: per_id`，将所有 format 计罚单位的系数提高为 0.5：违规 ID 按出现位置累计，其他协议违规按动作累计。过滤使用 `judge_or_all_correct_format`：保留原混合 Judge 组，额外允许全部答对且可训练轨迹中同时有无 format 惩罚和有 format 惩罚的组；全错组仍过滤。两类组均须存在实际可训练的非零 advantage，具体条件见下文。

重复读取仍通过 `strict_v2` 处理。训练执行引擎是 AG-mem 的独立 DAPO 实现，继续使用现有两卡 Reader 分层模型并行、CPU FP32 AdamW 和多轮检索环境；没有引入 verl/Ray 运行时。旧逐 ID 0.1 惩罚、Reader IDs、四组并发、逐组采样 v2、DAPO v1 与 GRPO 配置保留，可显式选择。

此次实验使用每次 READ 扣 0.01、单次 1 个 ID 上限、每计罚单位 0.5 的格式惩罚和上述扩展过滤规则，D 默认关闭；Reader 使用上文的 evidence_loop_v3 多步决策提示，Writer 和 Task Agent 的 prompt 不变。LoCoMo Judge 的时间与额外答案规则已于 2026-11-04 更新，具体见下文。冻结目录继续复用。此变更不新增 Python 依赖，也不会重启或改变已运行的训练；新实验对多步检索、停止行为及正确率的影响需要实际训练验证。


LoCoMo 将原训练集与验证集合并后，按精确 query 字符串去重分组，以 seed=42 随机打乱，按 9:1 重新划分。共 1,134 条记录、1,126 个不同 query；验证 query 数向上取整为 113，训练为 1,013 个 query。对应训练 1,019 条、验证 115 条，相同 query 的全部记录始终在同一侧，不删除重复记录。LoCoMo test 的 400 条记录逐字节保留；当前实验的训练、验证和测试均仅选择 `data.sources: [locomo]`，LongMemEval-S 不参与采样、评分或指标汇总；其历史数据文件保留。新数据与 prepared 使用独立目录，复用已有 Writer 结果；多轮检索设置不变，使用上述 reward 和过滤规则。

迁移到其他机器后，在旧 val100 数据与 prepared 已就绪且新输出目录尚不存在时运行：

```bash
python -m agmem.query_resplit \
  --config configs/train_dapo_reader_loop_v3_val100.yaml \
  --fraction 0.1 --seed 42 \
  --output-data data/processed/locomo_query9010_seed42_20261102 \
  --output-prepared runs/locomo_longmemeval/prepared_locomo_query9010_seed42_20261102
```

`manifest.json` 记录划分规则、query/记录数量及文件哈希；`row_mapping.jsonl` 保存每条记录对应的原 split 和行号。工具逐字段校验 Parquet 内容与 schema，并更新 prepared 样本 ID、记忆映射及来源信息，不调用模型。

测试专用配置为 `configs/eval_dapo_locomo_query9010_format05.yaml`：只选择 LoCoMo 的 `test` split（400 条记录、399 个不同 query），使用独立的 `runs/locomo_longmemeval/prepared_test_locomo_query9010_seed42_20261102`，不混入 train/validation，并使用同一 0.5 format 系数。训练配置的 `jobs.test_prepared_root` 与它一致，train-eval 工作流也继承相同的 `[locomo]` 来源选择。

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
python -m agmem.four_gpu --train-config configs/train_dapo_reader_loop_v3_query9010_format05.yaml --check-only
```

若旧实验仍在运行，先在原启动终端按 Ctrl+C，等待训练与服务进程退出；保留当前四卡 Slurm 作业，不执行 `scancel`。然后从初始 4B 权重开始新的 DAPO 实验：

```bash
python -u -m agmem.four_gpu --train-config configs/train_dapo_reader_loop_v3_query9010_format05.yaml
```

只进行一次 Reader 参数更新并保存可恢复断点：

```bash
python -u -m agmem.four_gpu --train-config configs/train_dapo_reader_loop_v3_query9010_format05.yaml --max-updates 1
```

`--max-updates` 限制本次调用的参数更新次数；动态采样仍需先收集一个完整有效批次。独立 Writer 目录任务仍按配置运行，这个参数不限制其目录构建数量。

从此 DAPO 实验最新的已提交 checkpoint 继续：

```bash
python -u -m agmem.four_gpu --train-config configs/train_dapo_reader_loop_v3_query9010_format05.yaml --resume
```

默认训练输出：

```text
runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train
```

两个 27B 服务、训练和目录构建的进程日志分别放在 `runs/four_gpu/<jobid>-<timestamp>/`。新实验从配置中的初始 Qwen3.5-4B 权重开始。并发窗口、Reader 合批大小和 token 预算写入 `sampling_protocol`；Reader 的输入协议与实际渲染 prompt 哈希写入 `reader_input_contract`，并随 checkpoint 和评估结果保存。切换采样协议、Reader prompt 版本、单次 ID 上限、奖励或过滤规则需要新实验，不能把旧协议下保存的轨迹和 reward 直接用于新协议续训。本次 format 系数和过滤规则均已改变，首次启动 `format05` 时不要加 `--resume`；上面的恢复命令仅适用于 `format05` 实验自身配置匹配的 checkpoint。旧 v1/GRPO checkpoint 同样不能切换为新 v2 续训。

原 `configs/train_dapo_reader_loop_v3_query9010_id_cost.yaml`、`configs/train_dapo_reader_loop_v3_query9010_id_cost_val4_restart.yaml` 和 `configs/train_dapo_reader_loop_v3_query9010_id_cost_val4_windowstop.yaml` 保留系数 0.1、`filter_metric=judge` 及各自的原输出目录，可用对应原配置加 `--resume` 继续。原 `configs/eval_dapo_locomo_query9010_id_cost.yaml` 同样保留 0.1 的评估设置。这些实验不能改用 `format05` 配置普通续跑。

原 `configs/train_dapo_max1_no_call_cost_v2.yaml` 保留零调用成本、`filter_metric=reward` 的实验及原 `.../locomo_s_dapo_reward_filter_micro2_parallel4_reader_ids_v2_format_count_max1_no_call_cost/train` 输出目录。其 checkpoint 不可直接切换为本次扩展 Judge 过滤配置续训。

原 `configs/train_dapo_max1_v2.yaml` 保留调用成本 0.01、`filter_metric=judge` 的混合正确性过滤及原输出目录。要继续这一旧实验，显式使用 `--train-config configs/train_dapo_max1_v2.yaml --resume`，不能用新配置读取其 checkpoint。

原 `configs/train_dapo_max2_v2.yaml` 保留每次最多 2 个 ID 的累计惩罚实验及原输出目录，可显式选择该配置继续旧运行。

原 `configs/train_dapo_reader_ids_format_count_v2.yaml` 保留每次最多 4 个 ID 的累计惩罚实验及原输出目录，可显式选择该配置继续旧运行。

原 `configs/train_dapo_reader_ids_v2.yaml` 保留每条 rollout 最多扣一次 0.1 的惩罚模式和原输出目录。未显式配置 `format_penalty_mode` 的旧 recipe 均继续使用 `per_rollout`。

恢复仍使用旧 Reader 提示的四组并发实验时，显式选择原配置，其输出目录仍为 `.../locomo_s_dapo_mixed_v2_micro2_parallel4/train`：

```bash
python -u -m agmem.four_gpu --train-config configs/train_dapo_parallel_v2.yaml --resume
```

旧 DAPO v1 实验使用显式配置，保留其门控成本、reward 差异过滤和 D reward 能力：

```bash
python -u -m agmem.four_gpu --train-config configs/train_dapo_locomo_s_4x80gb.yaml
```

旧 GRPO 实验使用显式配置：

```bash
python -u -m agmem.four_gpu --train-config configs/train_locomo_s_4x80gb.yaml
```

若该旧实验已经有 checkpoint，则在此命令末尾加 `--resume`。两条路径使用独立输出目录。

## Reader 已读 ID 与证据

`reader.prompt_version` 缺省为 `legacy_v1`，旧配置保持原来的消息结构与系统提示。新默认配置显式选择 `evidence_loop_v3`，旧 `evidence_ids_v2` 的prompt和输入契约保持不变。原有 `retrieved_evidence` 的每个对象已经含有 `id`；新版本额外将所有成功读取的 ID 汇总为顶层 `read_ids`，并在系统提示中明确它们和目录、原文的关系。简化的输入示例：

```json
{
  "query": "Alice 打算去哪里旅行？",
  "observation": "",
  "max_ids_per_call": 1,
  "read_calls_used": 1,
  "read_calls_remaining": 7,
  "directory": [{"id": "r_00001_example", "content": "Alice 的旅行计划"}],
  "read_ids": ["r_00001_example"],
  "retrieved_evidence": [
    {
      "id": "r_00001_example",
      "content": "Alice 的旅行计划",
      "raw_text": "Alice: 我打算去北京旅行。"
    }
  ]
}
```

每条证据的 `id` 对应目录中的同一个 ID；`content` 是目录摘要，`raw_text` 是该 session 的原文，实际记录还保留来源、时间和顺序字段。原文只在证据对象中保存一份。`read_ids` 由当前成功读取的证据生成，初始为空，读取成功后累积；未知 ID、容量拒绝或其他没有取得证据的请求不会被误标成已读。

Reader 每轮先结合 query、observation、选项和现有证据判断是否还缺信息。若证据足够，或目录中没有能够补充缺失信息的未读条目，输出 `{"ids":[],"stop":true}`；需要继续时，只选择能补充新信息的未读 ID，并输出原有 `{"ids":["ID"],"stop":false}`。动作 schema 不变，不要求额外输出解释或推理文字。已有 `last_tool_error` 反馈继续保留。

这项改动明确输入状态和跨调用指令，没有加入硬约束解码或在采样后改写动作。模型仍可能生成旧 ID，环境会按现有规则去重、反馈并计罚；能否减少重复和提高主动停止比例，需要通过同题评估及后续采样验证。

## 采样、更新与 step 的含义

| 配置 | 默认值 | 含义 |
|---|---:|---|
| `candidate_groups` | 32 | 每轮候选 query/group 数的上限，凑齐目标后在窗口边界提前结束 |
| `group_size` | 8 | 同一 query、同一记忆快照下的完整 rollout 数 |
| `target_groups` | 16 | 一个行为策略批次保留的有效 group 数 |
| `mini_batch_groups` | 8 | 一次 optimizer step 使用的完整 query group 数 |
| `micro_batch_rollouts` | 2 | 一次真实训练前向/反向最多同时处理的 rollout 数，每条选一个待计算 decision |
| `max_micro_batch_tokens` | 16384 | 一次 micro 的 padding 后输入 token 总数预算；超长单条允许独立处理 |
| `ppo_epochs` | 1 | 每批轨迹只优化一遍 |
| `max_num_gen_batches` | 4 | 一批至多进行 4 轮候选生成 |
| `filter_metric` | `judge_or_all_correct_format` | 保留混合 Judge 组，或全部答对且可训练轨迹同时有无惩罚和有惩罚的组 |
| `filter_min_std` | `1e-6` | 混合分支检查完整组 Judge 标准差；全对分支检查可训练轨迹的 format reward 标准差；均须严格大于此值 |

采样阶段的执行参数与训练 mini/micro 分开配置：

| `execution` 配置 | 默认值 | 含义 |
|---|---:|---|
| `sampling_group_concurrency` | 4 | 同一固定窗口内最多同时处理的候选 group 数 |
| `reader_batch_size` | 4 | Reader 一次调度最多合批的 decision 数，可来自不同 query |
| `reader_batch_max_tokens` | 16384 | Reader 生成合批的 token 预算，包含 padding 后的 prompt 和预留生成长度 |
| `task_agent_workers` | 2 | 并发评分 worker 数，每个 worker 顺序执行一条 rollout 的 Task Agent、Judge 评分 |

窗口内采用固定轮转顺序生成 Reader decision，由单一线程调用同一个 Reader 模型；完成检索的 rollout 交给两个 HTTP 评分 worker，使 Reader 推理与 Task/Judge 评分重叠。每组使用自己的 query、目录快照、8 个 rollout seed 和证据，结果按原候选顺序交付。已完成但等待前序候选的 group 仍占窗口名额。窗口内全部 group 完成生成和评分后，才接纳下一窗口的候选。因此这是带窗口边界等待的固定窗口流水线，不是某个 group 一结束就补入新 query 的全异步滑动窗口。

正常 HTTP 响应的完成先后不改变 Reader 的合批组合或最终候选选择。评分故障沿用完整 group 重试规则，失败评分不会当成 J=0；重试会复用 rollout seed，但可能改变合批组成，不承诺故障重试后逐 token 复现。退出采样或出现异常时会取消待执行评分并等待已运行请求结束。

Reader 生成预算按 `batch_size × (padded_prompt_length + max_new_tokens_per_decision)` 检查，超过 16384 时拆小批次；单条长上下文可单独执行，不为满足合批预算截断历史。该预算不改变已有上下文上限或生成截断规则，也不等于训练的 `max_micro_batch_tokens`。

首轮最多产生 `32 × 8 = 256` 条 rollout，分为最多 8 个四组窗口。凑齐 16 个有效 group 后停止提交新候选，完成当前窗口中已经启动的全部 group 的生成、评分及记录，然后进入更新；不会继续补足本轮的 32 个候选。串行采样在当前 group 完成后停止。尚未提交的 query 不推进数据游标，下批继续使用。按原采样顺序保留前 16 个有效 group，每组保存完整 8 条轨迹，包括正、负 advantage；不做组内 top-k 奖励筛选。

当前 `judge_or_all_correct_format` 按以下规则判断 group；“可训练轨迹”指未被 Reader overlong mask 屏蔽且包含 completion token 的轨迹：

| group 情况 | 是否通过过滤 |
|---|---|
| 混合 Judge | 完整组 Judge 标准差大于阈值，且可训练轨迹中同时有 J=0 和 J=1 |
| 全部答对，存在无 format 惩罚和有 format 惩罚的轨迹 | 完整组每条均为 J=1，可训练轨迹中同时有 `format_reward=0` 和 `<0`，且这些轨迹的 format reward 标准差大于阈值 |
| 全部答对，仅调用成本不同 | 过滤 |
| 全部答对，全部有 format 惩罚但扣分不同 | 过滤；缺少无 format 惩罚的对照 |
| 全部答错，即使存在 format 差异 | 过滤 |

两条通过分支还都要求至少一条可训练轨迹具有非零 advantage；整组 reward 相同、advantage 全为 0 时过滤。混合组唯一答对或唯一答错轨迹被屏蔽时，不能通过混合分支；不能因错误轨迹被屏蔽而将原混合组视为全对。全对组唯一无惩罚或唯一有惩罚轨迹被屏蔽或没有 token 时，也不能通过全对分支。屏蔽只影响 loss，不裁掉保存的完整 8 条轨迹，也不重算剩余轨迹的 advantage。当前窗口超出目标的有效 group 仍记录诊断，但不进入这批更新。

例如第一轮 32 个候选留下 14 个有效 group，第二轮第一个四组窗口补齐剩余 2 个，则累计评估 36 个候选就进入更新，不再等待 64 个。`target_groups=16`、`mini_batch_groups=8` 和验证间隔保持原配置。已启动的旧进程不会热加载修改；已有完整 checkpoint 可按原配置 `--resume`，保存的未消费轨迹先继续更新，随后采样沿用窗口边界停止规则。这不允许跨 reward 或 filter 切换：0.1 惩罚的旧实验仍用其原规则，`format05` 使用新输出目录。尚无 checkpoint 时，候选可读日志缺少完整 token 和行为策略概率，不能据此恢复内存中的采样批次；要保留它需等待旧进程生成 checkpoint 后重启，否则只能使用新输出目录重新采样。

不足 16 组时继续补采。默认含首轮累计最多 4 轮，即最多 128 个候选、1024 条 rollout（不计故障重试），仍不足则明确报错，并保留候选诊断；`max_num_gen_batches <= 0` 取消这个补采轮数上限，但数据集和配置中的 epoch 总数仍然有限。数据耗尽时不足目标的尾批会丢弃并记录 `dropped_tail_groups`；不会把不完整尾批静默当作 16 组训练。如果整个实验没有收集到任何完整批次，会明确失败。

采样期间 Reader 参数保持不变，所有候选的 Reader 推理、Task/Judge 评分及记录全部完成后才开始更新。采样与 backward 不重叠。组内 advantage 在各自完整 group 上计算后固定：

```text
A_i = (R_i - mean_group(R)) / (std_group(R) + epsilon)
epsilon = 1e-6
```

全对 format 对照组与混合 Judge 组使用相同的完整组 reward 和 advantage 计算，不另加损失或改为按动作单独归一化。全错组继续过滤；提高 format 惩罚并加入全对对照组后，仍需同时检查正确率、重复读取和停止行为。

`std_group` 使用总体标准差。v2 不设置 advantage 分子上限、advantage 裁剪或标准差下限；下述 PPO/DAPO ratio clipping 仍保留。真实生成 `old_logprobs`、行为策略版本和批次 ID 一并保存。默认 mini=8 时该批 16 个 group 产生 2 次 optimizer step；历史 mini=1 和 mini=2 配置分别产生 16 次和 8 次。两张 Reader GPU 是模型并行，不能据此将 batch 数量乘以 2。

`mini_batch_groups` 与 `micro_batch_rollouts` 是独立参数：默认 mini=8 累积 8 个 query group、共 64 条 rollout 的有效训练 token 后更新一次，16 个有效 group 共更新 2 次，再采样下一批；各 query 的 advantage 仍在各自的 8 条 rollout 内独立计算。micro=2 在单次 batched forward/backward 中处理来自最多两条 rollout 的 decision；一条 rollout 有多次检索时仍需多个 block，长输入还可能拆小，因此一组 8 条 rollout 不保证恰好执行 4 次 backward。它不表示每两条 rollout 就执行一次 optimizer step。整个 mini 共用有效 completion token 总数分母，只做一次梯度裁剪及 optimizer step。

micro 按待处理 decision 的输入长度组织，使用 attention mask 和独立 position IDs 隔离 padding；只有 completion token 参与 loss。预算按 `batch_size × padded_input_length` 计算，超过预算则拆小；单条 decision 已超过 16384 token 时单独运行，不截断其历史。这是控制合批的预算，不是上下文上限，也不能保证所有长样本都不会 OOM。显式设置 micro=1 时保留逐 decision 的串行前向路径。

旧逐组采样的独立配置仍可显式运行，各自使用原有输出目录；它们不启用四组并发窗口：

```bash
# mini=1，micro=1；输出 .../locomo_s_dapo_mixed_v2/train
python -u -m agmem.four_gpu --train-config configs/train_dapo_locomo_s_v2.yaml

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

当前默认 `train_dapo_reader_loop_v3_query9010_format05.yaml` 每 **4 次实际 optimizer step** 执行 validation，并保存该步 checkpoint。当前 `target_groups=16`、`mini_batch_groups=8`，每批更新 2 次，因此每累计完成 2 批、32 个有效 group 后验证，步数为 4、8、12……。独立的 `checkpoint_every=25` 保持不变，仍会在 25、50……额外保存，且允许已有同一 Reader 输入协议的实验仅调整 validation 间隔后普通续训。配置在启动时读取，正在运行的进程不会自动切换间隔。

原 Reader IDs recipe 仍每 16 步验证，其余旧 recipe 保留每 25 步验证的配置；间隔始终按 optimizer step 计数，被过滤的候选 group 不增加 step。若另行改变 mini 大小，每批更新次数为 `target_groups / mini_batch_groups`，需要相应选择验证间隔。checkpoint 保存优化器、采样游标以及尚未消费完的行为策略批次，包括原始 logprobs、固定 advantage 和已消费 group 数，恢复后继续该批次，不重新计算剩余轨迹的行为概率。一个 mini 只推进一次 policy version，消费的 group 数按 mini 大小推进。

恢复时按 checkpoint 的两个游标整理产物：超出优化步的训练/验证记录，以及超出候选游标的采样记录，会先归档到 `resume_orphans/`。未完成的 `.building` checkpoint 也会归档。随后根据保留的 JSONL 重建 TensorBoard，避免重复统计或用优化步横轴误删已提交的采样曲线。尚未消费的 active buffer 候选属于已提交采样，会继续保留。

## Judge 与 reward

Judge 使用与 Task Agent 相同的固定 Qwen3.5-27B 权重和服务，独立请求、独立缓存角色；temperature 为 0，关闭 thinking。

| 数据来源 | Judge prompt |
|---|---|
| LoCoMo | 基于 LoCoMo-Refined `refined` 模板，调整时间等价与额外答案规则 |
| LongMemEval-S／M | 官方模板，按 `question_type` 以及原始 `question_id` 中的 `_abs` 分支选择 |

来源、许可归属、模板及本地修改标记位于 `agmem/judge_prompts.py`。运行元数据记录模板内容 SHA256、模型指纹、解析协议和解码参数。`yes/no` 或 `CORRECT/WRONG` 在代码内映射为 1/0。LoCoMo 按 gold 的必要语义判分：时间题接受有明确基准的相对／绝对日期转换及两者并列，不因表示形式、兼容的时间精度或明确标注的会话基准而判错；缺失基准时不猜日期，错误换算、相互矛盾的事件日期或把模糊区间擅自替换成某个精确日仍判错。列表必须覆盖必要事实，额外的不矛盾项目和细节不扣分。LongMemEval 模板保持官方规则，其私有评分元数据从经过来源校验的原始数据恢复，不进入 Reader、Writer、Task Agent 的输入。

模板修改会改变评分缓存作用域，新的 Judge 实例不会复用旧模板的评分。已经启动的训练进程仍使用启动时加载的模板；旧 checkpoint 的直接续跑会检查 `scoring_protocol`，新旧规则不一致时拒绝混用，需通过新实验使用新规则。

此处的 LoCoMo 数据保持原样，实验标注为“LoCoMo 数据＋本地调整的 Refined Judge prompt＋Qwen3.5-27B”，不冒充 LoCoMo-Refined 官方完整配置。训练、validation 和 test 对同一来源使用相同 Judge 协议。

当前默认 `reward.formula: judge_cost_v2`、`reward.read_call_cost: 0.01`、`reward.format_penalty_mode: per_id`、`reward.format_penalty: 0.5`：

```text
R = 2 * J - 1 - 0.01 * n_read_calls - 0.5 * format_violation_count
```

`J` 为二元语义正确性分数。答对、答错都扣调用成本；`n_read_calls` 沿用环境中消耗预算的 READ 次数，包括空列表、无效 ID、重复读取和容量拒绝。STOP 和超预算未执行的请求不新增调用成本。`raw_read_cost` 与 `read_cost` 均为 `0.01 * n_read_calls`。

`format_violation_count = N_bad_id + N_other`。原有 format 规则继续生效：其中四类 ID 问题按出现位置累计为 `N_bad_id`，其他协议违规按动作累计为 `N_other`。四类 ID 计罚方式为：

| 规则 | 计罚方式 |
|---|---|
| 单次超量 | 上限 1 时，第 2 个及之后的每个 ID 计一次 |
| 同批重复 | 同一动作中相同 ID 从第二次出现开始，各计一次 |
| 无效 ID | 不属于当前目录的每个 ID 出现位置计一次 |
| 重复读取 | 请求已成功读取的 ID，每个出现位置计一次 |

同一位置触犯多条规则只扣一次 0.5；跨位置、跨动作累计。例：未读且有效的 `[A,B,C]` 扣 1.0；未读且有效的 `[A,A,A]` 扣 1.0；如果 A 已读，`[A,B,C]` 扣 1.5。超量请求仍只尝试原列表第一个 ID，后续位置不读取，不用后续 ID 替补无效首项。容量拒绝、因超量未执行的 ID 不进入已读历史。

JSON/schema、空列表、超预算等其他原有格式或协议违规也使用 0.5 系数，同一动作中此类违规最多计一次，累计为 `N_other`。这部分与 `N_bad_id` 相加：例如超预算且请求 3 个已读 ID，总 format 扣分为 0.5 + 1.5 = 2.0。四类 ID 违规不再额外叠加原来统一的动作惩罚；没有其他违规时，3 个坏 ID 就扣 1.5。合法 STOP 忽略 ID 列表，单纯容量拒绝不扣 format。`format_failure` 表示有任一被计罚的违规，`format_reward` 保存总扣分；`format_decisions_*` 继续记录原始协议诊断。

`answer_f1` 和兼容字段 `outcome_f1` 仍为独立 F1 指标。`outcome_reward` 为 `2J-1`，最终 `reward` 再减调用成本和 format。`best_outcome` 依据验证集 Judge 宏平均正确率，`best_reward` 依据验证集 reward 宏平均。逐 ID 累计没有原先“最多扣 0.9”的保证，因为一个动作可以输出多个违规 ID。

新配置继承 query 9:1 数据版本，使用独立输出目录。旧 `train_dapo_reader_loop_v3_query9010.yaml` 保留零调用成本和按动作计罚，可复现原实验；`id_cost` 及其 restart/windowstop 配置保留逐 ID 0.1 惩罚；未配置模式的历史实验保持 `per_rollout`。恢复训练时检查 reward、training 过滤配置及 `reader_id_with_protocol_v2` 计数版本。此次计数版本相同，但系数和过滤规则改变，仍不能将旧 checkpoint 作为新实验普通续训。

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

默认命令已关闭 D 计算。以下是保留的旧 v2 diagnostic 配置：它扣除调用成本、仅保留混合 Judge 组，但使用历史 format 规则，不是本次逐 ID 计罚实验的 diagnostic 版本：

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
tensorboard --logdir runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train/tensorboard --host 127.0.0.1 --port 6123
```

`train/entropy` 是训练前向中实际计算的全词表分布熵：

\[
H_t = -\sum_v p_\theta(v\mid s_t)\log p_\theta(v\mid s_t).
\]

单位为 **nats**，对本次更新未被屏蔽的 Reader completion token 按 token 数加权平均，在执行 optimizer step 前计算，以该次更新的 step 写入。它不是所选 token 的负 logprob；后者单独记为 `train/sampled_surprisal`，行为策略对应值为 `train/behavior_sampled_surprisal`。`entropy_chunk_size=32` 控制按 token 位置分块计算熵，以限制额外临时显存，不近似词表分布。

同时记录 `train/judge_accuracy`、`train/answer_f1`、`train/reward`、`train/clip_fraction`、`train/grad_norm` 等。验证曲线在 `validation/*`，来源宏平均在 `validation/source_macro/*`。

新增读取行为统计包括 `repeated_read_rate`（包含重复读取的轨迹比例）、`repeated_read_calls`（每条轨迹平均重复调用数）、`repeated_read_call_rate`（读取调用中重复的比例）、`no_new_evidence_calls` 和 `no_new_evidence_call_rate`，分别写入对应 train/validation/sampling 统计。另记录 `format_violation_count`（当前配置为每条轨迹平均违规 ID 出现次数加其他 format 违规动作数）与 `format_reward`（每条轨迹平均实际格式扣分），分别写入 train/validation/sampling 统计；完整及精简轨迹保存各自的原始计数。与 `format_failure_rate`、`judge_accuracy` 一起检查，避免把“重复次数减少”单独当作正确率提升。

`train/cumulative_trained_groups` 记录累计已消费 group；`train/update_groups`、`train/update_rollouts` 和 `train/training_tokens` 分别记录本次更新的 group、完整 rollout 和实际参与 loss 的 completion token 数。`training_tokens` 排除 overlong mask 的轨迹及所有 prompt token。`micro_batches` 表示实际 decision forward/backward 次数；`micro_batch_input_tokens`、`micro_batch_padding_tokens`、`max_micro_batch_input_tokens` 和 `micro_batch_over_budget_singletons` 可用于检查合批及 token 预算是否生效。增大 micro 只改变执行分块；增大 mini 会改变每次更新的样本数与优化步数，应据此比较实验。

`sampling/*` 的横轴是累计候选 group 数，与 optimizer step 分开。以下比例的分母均为当前行为策略批次已经评估的全部候选 group 数，包含被拒绝的 group；不是入选训练的 group 数：

- `sampling/judge_all_correct_rate`、`judge_all_wrong_rate`、`judge_mixed_rate`。
- `sampling/zero_reward_variance_rate`：最终 reward 组内标准差不大于 `filter_min_std=1e-6` 的比例；这是 reward 诊断，当前入选门槛分别检查 Judge 或 format 差异，并要求非零可训练 advantage。
- `sampling/cost_format_only_signal_rate`：Judge 同分，仅成本或格式提供差异的比例；当前调用成本为 0.01。该指标可能包含全错、纯成本差异或全对 format 对照，不能据此判断是否通过过滤。
- `sampling/all_correct_format_signal_rate`：完整组全对、可训练轨迹中同时有无 format 惩罚和有 format 惩罚，且其 format reward 标准差超过阈值的候选比例；此诊断尚不要求非零 advantage。
- `sampling/accepted_all_correct_format_rate`：满足上述全对 format 条件并通过过滤的候选比例。它以全部候选为分母；通过过滤但因当前窗口已凑齐目标而未被选入更新的组也计入，不能直接当作训练组中全对 format 组的占比。实际入选需查看 `candidate_metrics.jsonl` 的 `selected` 字段。
- `sampling/auxiliary_only_signal_rate`：启用 D 后仅辅助项提供差异的比例。
- `sampling/no_unmasked_signal_rate`：屏蔽截断轨迹后缺乏有效更新信号的比例。
- `sampling/no_unmasked_judge_signal_rate`：屏蔽 Reader 截断轨迹后，实际有训练 token 的轨迹不能同时提供 J=0 和 J=1 的比例；全对 format 对照组也计入此项，因此当前新过滤模式下，该值不等于无效组比例。旧 `filter_metric=judge` 模式仍拒绝这些组。
- `sampling/accepted_rate`、`generation_batches`、`candidate_groups`、`selected_groups`。
- `sampling/source_candidates/*`：当前采样批次累计的各来源候选数量。

并发采样每完成一个窗口，在 stdout 输出一条 `dapo_sampling_pipeline`，其 `progress` 包含以下累计值。每轮候选全部结束时，相同数值加上 `pipeline_round_` 前缀写入 `sampling_metrics.jsonl`，并显示为 TensorBoard 的 `sampling/pipeline_round_*`：

| 字段后缀 | 含义 |
|---|---|
| `windows_completed`、`groups_completed` | 本轮完成的窗口与候选 group 数 |
| `reader_batches`、`reader_decisions` | 本轮 Reader 调度批数与 decision 数；模型内部因 token 预算拆批时，实际生成调用可更多 |
| `reader_seconds` | 本轮 Reader 生成耗时之和 |
| `scoring_seconds_sum` | 所有评分 worker 的执行用时之和；并发部分会重叠，不能直接当作总墙钟时间 |
| `scoring_wait_seconds` | 调度线程等待评分结果的时间 |
| `peak_groups` | 本轮同时接纳的候选 group 峰值，默认最多 4 |
| `peak_scoring_inflight` | 尚未结束的评分任务峰值，包含线程池队列中等待的任务，不等于 HTTP worker 数 |
| `group_retries` | 本轮完整 group 的基础设施重试次数 |

这些 `pipeline_round_*` 只累计当前一轮最多 32 个候选的处理量，每轮补采重新计数；其余上述过滤比例仍累计整个行为策略批次，可能跨越多轮补采。窗口进度用于观察正在采样时是否仍有推进，实际加速幅度需结合相同数据与配置下的总采样时间测量。

权威文本记录为 `train_metrics.jsonl`、`sampling_metrics.jsonl`、`candidate_metrics.jsonl`。`candidate_rollouts/` 保留所有候选的精简动作和回答，包括被过滤的候选；不能仅凭进入训练后的 group 判断 Judge 是否产生大量同分轨迹。

参考：[DAPO 论文](https://arxiv.org/html/2503.14476v1)、[LoCoMo-Refined Judge](https://github.com/mem-eval-suite/LoCoMo_refined/blob/main/src/llm_judge.py)、[LongMemEval 官方评测](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py)。
