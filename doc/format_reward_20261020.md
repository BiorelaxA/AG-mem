# Reader 格式奖励设计与实现

日期：2026-10-20。此版本按用户确认加入格式惩罚，替代此前仅 outcome + cost 的奖励设计。

2026-10-22 更新：`stop=true` 时忽略 ids 字符串列表中的内容，直接停止，不读取、不增加成本，不因非空、重复、未知或超过 10 个 ID 判格式失败。字段及类型校验保留，原始输出和解析动作中的 IDs 原样记录。此前步骤的格式惩罚不会被清除。

2026-10-24 更新：当前默认采用 `2*F1-1-0.01*F1*read_calls-0.1*format_failure`。以下保留旧版本公式和案例以便解释历史日志；当前字段及兼容规则以 [奖励与 checkpoint 更新](reward_outcome_gated_20261024.md) 为准，格式合法性规则保持一致。

## 历史奖励公式

2026-10-25 更新：当前默认每次最多 4 个 ID，环境、prompt 和 reward 共同使用 `retrieval.max_ids_per_call`。`stop=false` 超量时仅尝试读取原列表前 N 个 ID，后续 ID 不读取、不参与工具校验；原始完整动作保留用于 GRPO 与格式评分，整条 rollout 最多扣一次 0.1。前 N 个重复、未知或超过上下文容量仍拒绝本次读取。`stop=true` 忽略全部 ID。下文 10-ID 整批拒绝规则为历史行为，以本段及最新实现规格为准。

```text
reward = outcome_f1 - 0.01 * read_calls - format_penalty * format_failure
format_penalty = 0.1
```

`format_failure` 为整条 Reader rollout 的二值标志。任意一轮格式失败记 1，否则记 0。每条轨迹最多扣一次；合法输出无格式加分，不按合法步骤数或错误次数累加。即使最终答案正确，格式惩罚也保留；总奖励不裁剪，且该惩罚是软约束。

选择题 outcome 仍按选项标签 F1（0/1），开放问答仍按答案文本 F1。每次最多 10 个 ID、最多 8 次批量检索、每次成本 0.01、GRPO group size=8 均保持不变。

## 格式检查边界

| 情况 | 格式失败 | 检索成本与执行 |
|---|---|---|
| 单一合法 JSON，只有 ids/stop，类型正确 | 否 | 依动作执行 |
| 合法停止：`{"ids":[],"stop":true}` | 否 | 不读取，成本不增加 |
| JSON 语法错误、额外/缺少字段、错误字段类型 | 是 | Reader 标记 protocol_failure 并终止，不提交本次读取 |
| `stop=true` 且 ids 非空（字符串数组） | 否 | 忽略列表内容，立即停止，不执行读取、不新增成本 |
| `stop=false`，空批次、超过 10 个或同批重复 ID | 是 | 工具整批拒绝，仍计一次检索成本，保留结构化错误反馈 |
| 未知/越权 ID | 否 | 工具拒绝，计一次检索成本 |
| 跨调用重复 ID | 否 | 本次调用仍计成本，证据按 ID 去重 |
| 上下文容量拒绝 | 否 | 整批拒绝并计一次成本，不截断原文 |
| 服务/存储基础设施故障 | 不作为格式负例 | 按既有组重试机制处理 |

解析器继续兼容一个完整外围 Markdown JSON/无语言代码块；代码块之外的说明、多个代码块或围栏内的非法 JSON 仍失败。Prompt 要求直接输出 JSON；动作解析不改写原始生成文本、token 或采样 logprob。

工具层批次格式错误后可以继续检索或停止，但整条轨迹的格式失败标志不会被后续合法动作清除。因而 `format_failure_rate` 可以高于只统计立即终止的 `protocol_failure_rate`。

## GRPO 与评估

Task Agent 仍基于当前成功检索的证据给出最终答案；计算 F1、已有检索成本与一次格式惩罚后，再对 8 条总奖励计算组内均值、标准差和 advantage。完整轨迹的所有 Reader 生成 token 共用该 advantage，不新增逐步奖励，不丢弃格式失败样本，不重新采样“修正后的”动作。

若组内总奖励仍完全相同，advantage 为 0；统一扣分本身不能制造组内差异。训练、validation 和显式 test 使用同一个评分函数。

## 配置、记录与历史兼容

```yaml
reward:
  read_call_cost: 0.01
  format_penalty: 0.1
```

默认配置及继承它的 smoke 配置启用 0.1。系数须为有限非负数；设为 0 可以消融。旧运行保存的独立配置若缺少该字段，按 0 保持旧奖励口径，不自动改成新目标。

单条完整/精简 rollout 增加：

- `read_cost`：正值检索成本。
- `format_failure`：整条轨迹的格式失败标志。
- `format_reward`：0 或负系数。
- `reward = outcome - read_cost + format_reward`。

训练 JSONL 与 TensorBoard 新增 `train/format_failure_rate` 和 `train/format_reward`，分别表示组内失败比例和平均格式奖励；评估 pooled、by_source、source_macro 同样记录这些分量。原有 `protocol_failure_rate` 保留。

已完成运行的原始 group、reward、advantage、TensorBoard 和 checkpoint 不回写。`export-rollouts` 只复制已记录字段，不对历史轨迹重新评分。恢复训练会检查 reward 配置，不能把新增惩罚配置静默接到旧奖励实验上；切换目标应使用新的输出目录。仅改变 reward 无需重做 prepare 或 Writer 目录。

## 历史 smoke 的离线示例（2026-10-20 旧规则）

以下保留 2026-10-20 的假设计算，不代表当前停止协议；其中 rollout 3 的非空 ids 停止动作按新规则合法。原始日志不回写：

| rollout | 原 reward | format_failure | 新 reward |
|---|---:|---:|---:|
| 0 | 0.272941 | 0 | 0.272941 |
| 1 | 0.606667 | 0 | 0.606667 |
| 2 | -0.01 | 0 | -0.01 |
| 3 | -0.01 | 1 | -0.11 |
| 4 | 0.92 | 0 | 0.92 |
| 5 | -0.01 | 0 | -0.01 |
| 6 | 0 | 1 | -0.10 |
| 7 | 0.97 | 0 | 0.97 |

平均 reward 从 0.34245098 变为 0.31745098；该示例的平均格式奖励为 -0.025、格式失败率为 0.25。实际历史日志保持原值。
