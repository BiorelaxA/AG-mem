# 首版实现验证

日期：2026-10-18。此报告区分 CPU 验证与尚未执行的真实模型/GPU 运行。

本文的数据库统计记录首版 SQLite 实现。2026-10-19 已按用户要求切换为完整轨迹 JSONL 与逐 session 目录；当前格式、复用范围及验证结果见 [JSONL 存储说明](jsonl_storage_20261019.md)。

本报告的“有效”限于结构与参考标签。后续 [文本编码补查](prepared_text_integrity_audit_20261018.md) 仍发现 28 个可疑历史 session 消息体，尚未改写或移除；不能将本报告解释为所有正文均无乱码或语义错误。

## 已通过

- `python -m unittest discover -s tests -v`：16 项测试全部通过。
- `python -m compileall -q agmem tests`：通过。
- CLI 帮助、默认与 smoke 配置解析、依赖文件与 Conda YAML 语法检查：通过。
- 全部 15434 条处理后 QA 的题型/答案适配：除已知 ZH4O 非法选项标签外均通过。
- 全量历史适配、输入 Parquet SHA-256 核对、SQLite `PRAGMA quick_check`：通过。

测试覆盖 session 粒度、空 session 保留、前 5 个 session 上下文及缓存失效、追加历史复用、标注白名单、Reader 反馈与 episode reset、原文/摘要最终交付、批量 1/10/11 个 ID 的成本及原子拒绝、未知/重复 ID、容量拒绝、强制结束与自然 stop 区分、标签/文本 F1、零方差优势，以及 HTTP 缓存/基础设施失败。

梯度测试使用 CPU 小型因果网络，确认生成 token 的 causal logprob 对齐、实际采样 scores 的记录与重算检查、trajectory/token 平均的 clipped 目标、完整 8 条 group 的参数更新、参考模型不变、优化器状态恢复及陈旧 policy version 拒绝。这是算法与连接逻辑验证，不是真实 Qwen3.5 训练结果。

## 数据准备结果

路径：`data/prepared/session_v1/`。

| split | 原始处理后行数 | 有效行数 | 隔离 |
|---|---:|---:|---:|
| train | 12244 | 12243 | 1 |
| validation | 90 | 90 | 0 |
| test | 3100 | 3100 | 0 |

共 2972 个 memory snapshot、238304 个不同 session 消息体。所有 9 个来源均保留。唯一隔离条目为 ZH4O train 当前行号 533（原始 train 行号 542）、`question_id=question_5_821`，参考答案不是 A/B/C/D。

PersonaMem-v2 两条列表末尾为 null 的错误选项填充不作为候选选项，样本保留。PerLTQA 中英文每份历史都包含 3004 个 session，其中 123 个为空占位；占位位置、目录项和 QA 样本全部保留。首轮完整历史验证暴露空 session 兼容问题后已修正，重新验证并补入 PerLTQA；迁移过程和文件/历史指纹核验记录在 `preparation_report.json` 的 `adapter_migration` 中。新适配器和 fresh prepare 回归测试均覆盖该情况。

原始 Parquet 和处理后的固定 train/validation/test Parquet 均未修改。准备报告详见 [preparation_report.json](../data/prepared/session_v1/preparation_report.json)。

## 尚未执行

- 新 Conda 环境的完整依赖解析/安装及 CUDA wheel 匹配；当前 CPU 测试使用现有 torch 2.9.0+cu126。
- Transformers 5.5.3 对本地真实 Qwen3.5 checkpoint 的加载、生成、logprob 一致性与 backward。
- 冻结 27B 服务、真实 Writer 目录生成、真实 8-rollout group 和 80GB 单卡显存验证。
- 超长完整目录在实际资源上的容量验证及后续结构性解决方案。

已提供这些步骤对应的候选依赖、服务命令、`configs/smoke.yaml` 与单组训练命令，见根目录 [README](../README.md) 和 [环境配置](environment_setup.md)。

## 2026-10-19：Writer 摘要长度修正

实际 smoke 目录构建在首个 session 因摘要超过 96 tokens 而失败。用本地 Qwen3.5-27B tokenizer 对 SQLite 中三次缓存输出复核，摘要正文分别为 101、97、97 tokens。首次提示缺少明确的正文长度约束，旧重试仅追加通用要求，没有携带上一次输出和实际校验错误。

修正后，首次提示声明配置的正文上限并要求预留余量；重试包含上一条输出、实际 token 数及逐步收紧的建议长度。保留原有 96-token 硬上限和 192-token 完整输出预算，耗尽重试后明确报告实际数量与上限。提示和格式重试协议计入目录版本，防止误用旧目录缓存；原始数据和 prepared 数据库无需重建。

`python -m unittest discover -s tests -v`：19 项全部通过；新增回归覆盖超长摘要重写、恰好达到上限的合法摘要、合法缓存复用、重试耗尽时不发布无效目录，以及 JSON/字段类型修复。`python -m compileall -q agmem tests` 通过。本次修正尚未重新调用真实 27B 服务构建目录；上述测试使用 HTTP 模拟响应。
