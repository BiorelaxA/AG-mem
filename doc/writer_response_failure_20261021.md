# Writer 空响应和截断重试修复

本文件记录历史 v1.16 行为；当前已按用户要求切换到 [v1.17 的 300-token 纯文本输出](writer_plain_content_20261021.md)，不再进行这里描述的 Writer 格式或长度修复。

作业 `185193.vyyeAX` 和 `185193.kLnZqq` 均在构建目录时失败，GRPO 尚未开始。失败目标是第一条历史的第 4 个 session，`r_00003_a90539ec0c38`；完整重建输入为 28,060 tokens，加初始完整输出预算 384 后为 28,444，低于服务的 65,536-token 上限。离线计数与前 3 次成功请求的缓存键已核对一致。

旧实现把 `finish_reason=length` 与空 content 合并为 `Truncated or empty service completion`，并在服务层使用相同输入、温度 0、输出预算 384 重试。异常最终以 InfrastructureError 退出，Writer 的 JSON/长度修复循环没有机会处理它。失败响应没有保存，因此历史这两次失败到底是截断还是空正文，不能从现有日志断言。

修复后：

- 服务返回空正文或截断响应时，Writer 立即进入现有格式修复循环，加入上次非空输出和明确的纠正提示；空正文不作为 assistant 空消息传回模型。
- 首次完整输出预算仍取 `writer.max_output_tokens`（当前 384）。若截断，再次尝试将预算翻倍，默认最多 1536，即当前配置下依次为 384、768、1536。`service.writer_retry_max_output_tokens` 可覆盖重试上限，上限至少为初始预算。
- `writer.content_max_tokens=200` 始终约束最终摘要，较大的重试输出预算不会放宽摘要要求。完整 JSON、非空字符串和内容长度校验均通过才写入目录。
- `writer.format_retries=2` 仍表示最多首次加两次修复。HTTP/连接错误仍由服务层重试。Task Agent 的截断和空正文仍按基础设施失败重试，不参与 Writer 格式修复，也不接受截断答案作为任务结果。
- 每次失败追加写入 prepared 下的 `service_failures.jsonl`，包括时间戳、角色、请求缓存键、历史缓存作用域、尝试次数、输出预算、状态码、异常信息、完整 JSON 响应。若响应含 `reasoning_content`、`finish_reason`、`usage`，这些字段会保留。记录不含授权头或输入历史。网络故障未取得 JSON 响应时 response 为 null。

已有成功目录摘要的生成请求、提示词、200-token 校验与版本身份保持一致，修复路径只影响未完成的项。因此 `runs/slurm_full_train/prepared/` 已生成的 3 个目录项可直接复用，不需要修改运行配置快照或清空 prepared。

重新执行 `bash scripts/train_a6000.sbatch` 可继续构建。失败修复仍耗尽时将停止并报告目标历史/session 和具体原因；查看 `runs/slurm_full_train/prepared/service_failures.jsonl` 的 response 即可区分空正文、reasoning-only 与截断。当前未通过实际 GPU 请求验证修复后的第 4 个 session 能否成功生成。
