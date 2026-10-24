# Writer 只输出 300-token 摘要文本

Writer 输入为当前 session 和同一历史中前面最多 3 个 session。原有内容要求保持：主体和动作明确、事实短句与分号、保留实体细节和时间状态、覆盖多个话题、保留否定和不确定性；前文仅用于有依据的指代消解，事实必须来自当前 session。

模型只输出摘要纯文本，不生成 JSON、content 字段名、session ID 或其他目录元数据。程序从清洗历史提取 ID、顺序、时间和人物信息，把生成文本放入 content，再统一转义并写入 directories.jsonl。

配置仅需要：

```yaml
writer:
  lookback_sessions: 3
  content_max_tokens: 300
```

content_max_tokens 同时用于请求的 max_tokens 和最终摘要上限；不再使用独立 max_output_tokens、format_retries 或 writer_retry_max_output_tokens。finish_reason=length 且正文非空时直接缓存并使用，不补全句子、不请求模型重写。如果服务异常返回更多文本，用本地 tokenizer 截取前 300 tokens。允许最后一句不完整，程序生成的目录 JSON 始终保持合法。

Writer 返回空正文时记录 service_failures.jsonl，并保存 content 为空字符串、writer_status 为 empty_pending 的目录项；随后继续后面的 session，不中断构建或后续流程。空响应不写入成功响应缓存，也不在本次构建中立即重试。原始 session 本身为空使用固定无记录摘要，不标记待补。网络/HTTP 故障沿用 service.retries；重试耗尽仍报错。Task Agent 的空正文和截断答案处理保持不变。

纯文本协议版本为 4，与旧 JSON/200-token 协议使用不同的目录及响应缓存版本。旧数据不删除：清洗历史和固定数据划分继续使用，旧目录项保留，但新版本会重新生成全部摘要；之后相同版本的中断重跑仍逐 session 复用成功项。

本工作区的 runs/slurm_full_train/config.yaml 已同步为上述配置，原快照保留在 config.writer_json_v3.yaml。其他机器若已有运行快照，需要同步修改该快照的 writer 配置；仅修改 configs/default.yaml 不会覆盖已冻结的运行配置。无需清空 prepared 或重新划分数据。

在已分配三张 A6000 的终端中继续执行：

```bash
bash scripts/train_a6000.sbatch
```

## 待补空摘要的续跑

- 下次 build-directories 会检查完整目录中是否含 empty_pending，避免整份目录缓存让空项被直接跳过；每个待补项本次只重新生成一次，正常目录项继续复用。
- 待补项成功后写入非空 content 并移除 writer_status 标记。只有明确的空占位项允许在同一版本补齐；ID、顺序和来源元数据不能改变，正常摘要仍不可覆盖。
- 目录 JSONL 以追加方式记录补齐结果，不删除旧行；同一 memory_id/version/id 以索引指向的最新行为准。因此直接统计原始文件所有旧标记行不等于当前待补数量。
- 补齐时同步使旧目录快照及内存缓存失效，整份目录重新发布校验清单；关闭、重新打开以及中途失败后的续跑均可恢复。
- 每条 directory_built 日志包含 empty_pending 数量，构建结束的 directories_finished 给出总数；prepared/manifest.json 的 directory_empty_pending 保存当前构建的总数。
- 全量脚本发现 train.complete 但仍有待补项时，会再跑目录构建；训练已完成时不会再次启动训练。待补项修复规则不改变 Writer 提示、300-token 预算和目录版本，现有成功摘要可继续复用。

2026-10-21：参考窗口从前 5 个改为前 3 个 session。窗口和提示指纹变化会生成新的目录版本；已有清洗历史继续复用，旧窗口生成的摘要保留在旧版本下，新窗口需要重新生成摘要。相同新版本中断后的续跑仍复用成功项。
