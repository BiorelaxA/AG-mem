# 训练日志与 TensorBoard

日期：2026-10-19；格式奖励指标补充于 2026-10-20。首节保留原 smoke 历史结果，不回写旧奖励。

## 本次 smoke 的问题

`step=1`、checkpoint 保存和进程正常退出说明训练代码链路执行完成，不代表本组产生了有效学习信号。已保存的 8 条轨迹全部是 `protocol_failure`，没有发生读取，reward/advantage 均为 0，loss 和梯度也为 0。

主要原因是 Reader 返回了 Markdown JSON 代码块，旧解析器直接调用 `json.loads`，拒绝整个文本。解析器现在接受裸 JSON 或一个完整的 JSON/无语言代码块，依旧严格检查字段与类型，并要求 `stop=true` 时 `ids=[]`。代码块外有解释、多个代码块或非法 schema 均不接受。Reader 提示也明确要求不输出围栏。

解析仅处理动作文本，不修改记录中的原始输出、生成 token 或实际采样 logprob，不加入约束解码。用原运行的 8 条输出回放解析：7 条成为合法读取动作，1 条仍因 `stop=true` 且 ID 非空而非法；历史运行的 reward 不会因此回写。

## 新训练的指标

默认 `training.tensorboard: true`，事件保存到 `<training.output_dir>/tensorboard/`。每次 GRPO 更新完成后，以更新步数为横轴写入数值指标并立即 flush：

- `train/reward`：本组 8 条完整轨迹的平均 `outcome - 0.01 * read_calls - 0.1 * format_failure`（默认新配置）。
- `train/reward_min`、`train/reward_max`、`train/reward_std`：组内奖励分布。
- `train/outcome_f1`、`train/read_calls`、`train/read_cost`、`train/unique_ids_read`：任务结果与读取开销。
- `train/loss`、`train/grad_norm`、`train/sampled_kl`、`train/max_old_logprob_error`：训练指标；logprob 差异仅记录，不触发阈值检查。
- `train/format_failure_rate`、`train/format_reward`：整条轨迹任一格式失败的比例及平均格式奖励；每条最多扣一次，默认单条为 0 或 -0.1。批次格式错误即使随后正常 stop，也计入格式失败率。
- `train/protocol_failure_rate`、`train/natural_stop_rate`、`train/zero_variance_group`：协议、停止与零方差组监控。

周期 validation 的总体、来源平均及各来源数值指标分别写入 `validation/*`、`validation/source_macro/*`、`validation/by_source/<source>/*`。smoke 配置关闭周期验证，因而没有这部分点。

退出时关闭 writer 并刷新剩余事件。恢复 checkpoint 时从保存步数的下一步继续，使用 `purge_step` 隐藏该位置之后的旧事件；TensorBoard 开关不影响模型/优化器配置兼容性。[PyTorch SummaryWriter 文档](https://docs.pytorch.org/docs/2.10/tensorboard.html)

## 已结束运行的导入

```bash
python -m agmem export-tensorboard runs/smoke_session_grpo
```

命令读取 `train_metrics.jsonl` 及已有 `validation/step-*/metrics.json`，不加载模型，不访问 GPU。重复导入会清理旧图上的 scalar 点后重新写入，不修改原始 JSONL 或 checkpoint。单步 smoke 只有一个 reward=0 的点，不能显示多步趋势。

已经把本次 smoke 的 1 条训练指标导入 `runs/smoke_session_grpo/tensorboard/`。之前未保存的终端原始输出无法还原；后续通过 stdout/stderr 重定向保存，具体命令见根目录 README。

## 依赖与验证

`requirements.txt` 增加 `tensorboard>=2.20,<3`。现有 `agmem` 环境尚未安装这个新依赖，执行 `python -m pip install 'tensorboard>=2.20,<3'` 后即可使用；此次没有修改该 Conda 环境。

CPU 单元测试 33 项全部通过，编译检查通过。新增用例使用真实 event 文件读取验证 reward 曲线、validation 标签、重复导入及恢复时旧点清理；环境回归验证带围栏动作可以读取原文并停止，原始输出保持不变。完整 4B/A100 的修复后检索和非零梯度训练仍需在 GPU 节点重跑。
