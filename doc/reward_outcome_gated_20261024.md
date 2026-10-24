# Outcome 加权成本与最佳 checkpoint

2026-10-24：本轮修改奖励和 checkpoint 保留；保留 group=8、temperature=1、组内减均值并除标准差的 advantage。没有引入重采样、强制首次检索或固定检索奖励，也不自动停止或重启现有 Slurm 作业。

## 当前奖励

默认配置 `reward.formula: outcome_gated_v2`：

```text
s = outcome_f1                       # 选择题为标签 0/1，开放问答为文本 F1
n = n_read_calls
f = format_failure                   # 一条 rollout 最多扣一次
outcome_reward = 2*s - 1
raw_read_cost = 0.01*n
read_cost = s*raw_read_cost           # 实际从 reward 扣除的成本
format_reward = -0.1*f
reward = outcome_reward - read_cost + format_reward
```

同样答错且格式正确时，无论检索 0 次、1 次还是 8 次，reward 均为 -1，不再把少尝试当成更好结果。同样完全答对时，一次检索得 0.99，三次得 0.97。F1=0.5 且一次检索得 -0.005。格式错误独立扣分，完全答对也不能免除；正确格式不额外奖励。

成本随连续 F1 加权，并非仅在 F1=1 时才生效；这是加权目标，不是保证任意微小 F1 提升都压倒所有成本差异的字典序目标。检索次数上限仍为 8。全组 reward 相同时仍没有相对奖励信号，统一负分不会自动恢复探索。

训练、validation、test 共用评分函数。TensorBoard 的 `train/reward` 是 8 条 rollout 的最终奖励均值，`train/outcome_f1` 保持原始 F1，增加 `train/outcome_reward`、`train/raw_read_cost`；`train/read_cost` 改为实际加权扣款。逐条计算成本后取平均，不能用平均 F1 乘平均调用次数。完整/精简 rollout 及验证指标同步记录这些分量。

独立冻结旧配置缺少 `formula` 时沿用 `outcome_cost_v1`（原始 F1 减未加权成本与格式惩罚）。历史文件不重算；旧 reward 的 checkpoint 不允许在同一实验中静默续训为新 reward。新的独立 LoCoMo/S 脚本使用 `locomo_s_reward_v2` 实验目录，从配置中的初始 4B 启动，同时复用现有 train/validation 目录缓存。旧实验 `locomo_s` 保留。

## checkpoint 规则

- 保留最近 `training.keep_last_checkpoints` 个完整 checkpoint（默认 2），用于续训。
- 额外保留固定 validation 的 `source_macro.reward` 最高和 `source_macro.outcome_f1` 最高 checkpoint，分别称 `best_reward`、`best_outcome`；不使用训练 group 的 reward 选优。
- 最佳点相同不重复保存；分数完全相同保留更早的模型。最多保留最近两个加两个不同最佳点，共四份完整 checkpoint。
- 每次 validation 后必保存完整 checkpoint，即使该 step 不满足周期保存间隔。模型、优化器、配置和 `validation_metrics.json` 一起通过 `.building` 目录重命名提交，提交后才更新选择与清理。
- `<train_dir>/best_checkpoints.json` 记录最佳点的相对 checkpoint 名、step、指标和 scope；这是指针文件，不另复制大模型。恢复时可从完整 checkpoint 内的验证指标重建选择，未提交的 `.building` 不参与。
- 验证频率为每 50 step，保存频率保持 LoCoMo/S 每 step，未新增 step 0 baseline。最佳模型仅在已执行验证并成功保存的 checkpoint 中选择。
- 独立训练评估脚本最终 test 使用 `best_reward`；尚无验证 checkpoint 时明确记录回退到最新 checkpoint。续训始终使用最新 checkpoint，不能用 best 替代 latest。

新实验路径：

```text
runs/locomo_longmemeval/experiments/locomo_s_reward_v2/train/
  checkpoint-00000100/
    model/
    trainer.pt
    validation_metrics.json
  best_checkpoints.json
  tensorboard/
  rollouts/
```

提交入口仍为 `sbatch scripts/train_eval_locomo_s_a100.sbatch`。正在运行的旧任务继续使用其已加载代码和冻结配置；需要先结束占用相同 LoCoMo/S 数据锁的旧任务，才能启动新任务。本次不修改目录生成的模型、prompt 或缓存版本。
