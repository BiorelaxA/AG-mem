# 采样 logprob 与训练重算

日期：2026-10-19。

## 当前实现：取消一致性阈值检查

按用户决定，已移除采样 logprob 的一致性阈值检查：

- `HFPolicy.complete` 只记录实际 sampling scores 对应的选中 token logprob，不再额外执行 teacher forcing 重算，也不再请求原始 logits 来比较采样分布。
- `GRPOUpdater.update` 保留训练必需的带梯度重算，真实采样 logprob 继续作为 GRPO ratio 的旧概率；不会用重算值覆盖旧概率。
- 更新前的最大绝对差仍记录为 `max_old_logprob_error`，不因差异超过某个阈值报错或丢弃轨迹。
- 删除 `reader.sampling_logprob_tolerance`、`reader.processor_logprob_tolerance` 和 `training.old_logprob_tolerance` 配置，以及不再计算的 `Completion` 误差字段。
- 保留 token/score 数量对齐、完整 group=8、策略版本、非有限损失和梯度检查。

此次修改不需要重新准备数据或生成 Writer 目录。下文记录此前排查过程；其中的阈值方案与拒绝行为已撤销。

## 历史报错

4B smoke 训练在 `HFPolicy.complete` 记录真实采样概率后，对同一 prompt 和生成 token 进行 teacher forcing 重算，最大绝对 logprob 差为 `0.113252`，超过原来的 `0.05`，因此主动终止。该差值是自然对数概率之差，所对应的概率比约为 1.12。

生成使用逐 token 解码及 cache，重算使用整段前向；Qwen3.5 的 Gated DeltaNet 在 cached decode 与整段处理时采用不同计算分支。即使关闭 cache，输入序列长度和 LM head 矩阵形状仍可能不同。BF16 下数学等价的计算不保证逐位一致；PyTorch 官方也明确说明批量计算与切片计算、不同计算顺序可能产生数值差异。[PyTorch 2.10 数值精度说明](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html)

同架构的缩小模型已复现这一类差异。这支持数值误差的判断，但尚未在完整 4B GPU 路径重新运行并获得原始/处理后 logits 诊断。

## 首次调整（阈值方案现已撤销）

1. `reader.sampling_logprob_tolerance` 与 `training.old_logprob_tolerance` 从 `0.05` 改为 `0.18`。`exp(±0.18)` 约为 `[0.835, 1.197]`，处于当前 clip ratio=0.2 的 `[0.8, 1.2]` 内。此范围内仍记录并保留实际偏差，超限仍终止。
2. 新增 `reader.processor_logprob_tolerance: 0.00001`。生成同时返回同一前向中的原始 logits 与 sampling scores；原始 logits 仅按配置温度处理后的选中 token logprob 必须与实际采样 scores 一致。额外过滤、温度或惩罚改变分布会单独报错，不能依靠较大的 BF16 数值容差通过校验。
3. `Completion` 记录 `sampling_logprob_error` 和 `processor_logprob_error`。生成时实际 scores 对应的旧 logprob 继续进入 GRPO 比率，没有以重算结果替换、重采样或删除轨迹。
4. teacher forcing 输入改为 `prompt_ids + token_ids[:-1]`，对最后 n 个位置预测 n 个生成 token。最后一个生成 token 只作目标，无需再作为输入；causal 对齐与仅生成 token 计入 loss 的规则保持一致。

这些阈值是当时采用的工程容差，并非 GRPO 理论要求。初始概率比处于 PPO clip 区间内，不能单独证明数值误差可接受。

Reader 继续使用 BF16 和已配置的 cache；group=8、全参更新、F1 outcome、检索成本与 PPO/KL 公式保持原定义。FLA/causal-conv1d 为可选加速依赖，安装它们不自动保证不同计算形状逐位一致。

## 首次调整时的历史验证

- `python -m unittest discover -s tests -v`：28 项全部通过；`python -m compileall -q agmem tests` 通过。
- 回归验证小于数值容差的额外 logits penalty 仍被严格拒绝；超出数值容差的执行差异仍报错。
- 回归给真实采样旧概率加入已知偏差，解析验证 GRPO loss 为 `sinh(0.08)`，确认旧概率没有被 teacher forcing 结果覆盖。
- 在实际 `agmem` 环境（torch 2.10.0+cu129 / transformers 5.5.3）使用真实 Qwen3.5 模型代码构造小模型。4 层 FP32 对比最大误差约 `4.77e-7`，BF16 约 `0.0039`。
- 32 层 BF16 压力配置：hidden=128、保留 3 个 linear attention + 1 个 full attention 的重复结构，随机初始化，LM head 权重乘 10 放大数值差异。80-token prompt、12-token 输出预算下，cached 生成与整段重算误差约 `0.0811`，关闭 cache 仍约 `0.0724`，而逐前缀同形状重算可为零。因此仅关闭 cache 不能充分解决原先的严格阈值问题。
- 该 32 层小模型完成 8 次实际采样、反向传播与一次全参 CPUAdamW GRPO 更新（本压力测试 KL 系数=0）。最大实际 logprob 差 `0.1346499`，同前向 processor 差为 0，梯度范数 `63.25`，policy version 从 0 更新到 1。

上述小模型验证使用 CPU，并非本地预训练 4B 权重；完整 4B/A100-40GB 的采样、精度和显存仍需在 GPU 节点重试，不能由小模型结果替代。

## 取消检查后的回归验证

- `python -m unittest discover -s tests -v`：28 项全部通过；`python -m compileall -q agmem tests` 通过。此次验证为 CPU 测试，未重新运行完整 4B GPU 训练。
- 采样直接保留实际 scores 的 logprob，不要求原始 logits；mock 明确禁止采样后调用 teacher forcing 重算，确认没有额外校验前向。
- 使用已知采样概率偏差，确认不再触发一致性报错，且保存的仍是真实行为概率。
- 对 8 条轨迹分别注入 `0.08`、`0.4` 的 logprob 差，均完成全参更新，最大差异只作为指标。解析验证损失为 `sinh(drift)`，确认没有用重算概率替换采样旧概率。
- token 与 score 数量不一致仍报错，防止保存错位的概率。

## 重试

这次失败只留下 `run.json`，没有 group 日志、指标或 checkpoint，已保留在 `runs/smoke_session_grpo_failed_logprob_20261019T183513/`。该目录另保存 CPU 压力测试脚本 `tiny_qwen_logprob_probe.py` 和实测结果 `tiny_qwen_probe_result.json`。

```bash
CUDA_VISIBLE_DEVICES=0 python -m agmem \
  --config configs/smoke.yaml train --max-groups 1
```

此次改动不需要重新准备数据或生成 Writer 目录。
