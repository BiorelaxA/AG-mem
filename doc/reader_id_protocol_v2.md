# Reader ID 执行协议 repair_v2（2026-10-26）

通过 `retrieval.id_protocol: repair_v2` 显式启用。缺省值仍为 `strict_v1`，旧独立配置的执行与奖励口径不变。本协议更新 ID 执行、错误恢复及 format 的定义，不改变 Writer、目录、Reader 上下文上限、每次最多 4 个 ID、最多 8 次调用、group size 8 或优化器超参数。

## 执行规则

- `stop=true` 忽略 ID 内容，不读取也不因 ID 内容受罚；JSON/字段类型仍须合法。
- `stop=false` 只处理原始列表的前 4 个位置；多余位置不查找、不修复，不用后面的 ID 补足无效位置。超限照常计 format 违规。
- 精确存在于当前历史目录的 ID 直接读取。仅对 `r_数字_12位小写十六进制哈希` 的编号缺前导零情况尝试修复：编号整数、哈希必须完全一致，当前目录中必须唯一匹配，规范编号至少五位。不猜测哈希或跨历史查找。
- 需要修复和无法修复的 ID 均计 format 违规。修复成功的 ID 正常执行；同批中其他有效 ID 也正常执行。重复 ID 去重读取，原始列表重复或修复后重复均计违规。
- 容量检查仍检查所有候选有效记录；容量不足则本批有效记录全部不提交，保留此前证据，不自动缩短上下文。容量问题本身不算 format。
- 首次无效 ID 返回具体无效列表及纠错提示。相邻两次调用包含相同无效 ID 集合，且第二次没有新增证据，第二次调用后以 `repeated_invalid_action` 结束检索。两次调用都计入成本。正常跨轮重读合法 ID 不算 format 违规。
- 结束后仍调用 Task Agent，使用累计证据；即使没有证据也不终止训练或跳过 group。
- 原始 action、completion、token IDs、行为 logprob 不改写；修复只作用于工具执行，GRPO 仍针对实际采样序列计算。

## 奖励与日志

`R = 2*S - 1 - 0.01*S*N_read - 0.1*f` 保持不变。`f` 是整条 rollout 的二值标记：JSON/字段非法、空读取、超过 ID 上限、同批重复、修复 ID、未知 ID、重复无效调用中任意一种成立即为 1。多个原因、多次违规不累计扣分，最多扣 0.1。

每个 decision 保存 `format_violations`。工具结果保存 `id_repairs`、`invalid_ids`、`selected_ids`、实际 records。精简 rollout 保存原始输出、修复映射、返回 ID、错误和违规原因，不保存原文。

训练和验证/TensorBoard 新增 `id_repair_rate`、`unknown_id_rate`、`repeated_invalid_action_rate`、`empty_evidence_rate`；这些 rate 均以 rollout 为分母。`format_decisions_<reason>` 为每条 rollout 的该类违规决策平均次数，不参与累计奖励。

## 从 checkpoint 100 建立新实验

配置：`configs/resume_locomo_s_idrepair.yaml`。独立输出：`runs/locomo_longmemeval/experiments/locomo_s_reward_v2_time_v1_ids4_idrepair_v2/train`。

`--fork-id-protocol-resume` 要求新输出目录为空，允许更改 ID 协议、训练输出目录与验证间隔；数据、Reader、调用预算、奖励系数和训练参数仍校验一致。与 `--fork-retrieval-resume` 互斥。恢复原模型、优化器、policy version 和训练游标。`run.json` 和 `retrieval_changes.jsonl` 保存来源。

首次运行使用 `--validate-before-training`：先在修复协议下评估恢复的 step 100，写入本实验 `validation/step-00000100` 与 TensorBoard，再保存本实验 checkpoint 100（含优化器，参与 best 保留），随后从 step 101 继续。新版配置每 25 步验证（125、150、175……）；每步保存及保留最近两份与最佳 checkpoint 的策略保持不变。普通续跑也允许调整验证间隔，并记录 `resume_validation_interval_changed`。此基线用于区分协议修复收益与后续训练收益。

后续中断续跑应使用新实验最新完整 checkpoint，去掉两个首次运行参数：`--fork-id-protocol-resume`、`--validate-before-training`。不要把新配置直接用于旧输出目录。

运行命令（在原 allocation 的 GPU 节点，停止旧 Reader 训练进程并保留 27B 服务后，在 tmux 中执行）：

```bash
cd /home/hongyshen/AG-Mem
mkdir -p runs/train_eval_locomo_s/jobs/190770.idrepair_resume
curl --noproxy '*' --fail --max-time 10 http://127.0.0.1:50559/v1/models && \
CUDA_VISIBLE_DEVICES=0,1 \
AGMEM_READER_MODEL_PARALLEL_SIZE=2 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
AGMEM_ACTIVATION_OFFLOAD_MIN_TOKENS=16384 \
/home/hongyshen/miniconda3/envs/agmem/bin/python -u -m agmem \
  --config configs/resume_locomo_s_idrepair.yaml train \
  --resume runs/locomo_longmemeval/experiments/locomo_s_reward_v2_time_v1_ids4/train/checkpoint-00000100 \
  --fork-id-protocol-resume --validate-before-training \
  > runs/train_eval_locomo_s/jobs/190770.idrepair_resume/train.log 2>&1
```
