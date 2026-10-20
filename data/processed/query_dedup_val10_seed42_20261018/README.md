# Query 去重与每来源 10 条验证集

日期：2026-10-18。随机种子：42。状态：`split_complete_pending_model_input_adaptation`。

本目录已完成用户指定的数据划分，包含全部 9 个来源（包括 PerLTQA 中英文）。原始 `data/<source>/train.parquet` 和 `test.parquet` 未修改。

用户已确认后续使用：将本目录 9 个来源的 `train.parquet` 组合作为训练集（12244 条），将对应的 `validation.parquet` 组合作为验证集（90 条），不重新划分。来源 schema 不同，加载时先按来源适配，再逻辑组合；test 独立保留用于评测。

## 处理规则

1. 在每个来源内部，以其原始 test 的 `question` 原始字符串为集合，从原始 train 删除 query 完全相同的全部行。大小写、Unicode、标点和空白均不归一化；不要求答案、选项、历史同时相同。
2. 在去重后的每个来源中抽取 10 条 validation，其余作为 train。为避免验证问题进入其他来源的训练或测试，普通候选 query 在全部剩余训练来源中只出现一次，且不在任何来源 test 中出现。
3. LongMemEval s/m 是同一批问题的不同历史长度版本：候选 query 在每个版本各出现一次、其他来源不出现，并同时留出相同的 10 个问题。因此 validation 共 90 行，对应 80 个不同 query。
4. 每个来源独立执行 `random.Random(42).sample(sorted(eligible_query_strings), 10)`；输出保留原始行顺序。已知 ZH4O 非法参考样本不参与 validation 抽样，仍保留在 train。
5. query 去重范围为同来源 train/test。跨来源 train/test query 重复仅记录，不额外删除。历史或部分 session 重叠不作为删除条件，本次不是历史隔离划分。
6. 原始列、类型、schema 元数据及每个字段值均保留；每个 test 是原始文件的逐字节副本。本阶段不清除标注字段、不构建 Writer 目录、不调用模型、不启动训练。

## 数量

| 来源 | 原始 train | 删除同 test query 的行 | 最终 train | validation | test |
|---|---:|---:|---:|---:|---:|
| locomo | 1136 | 2 | 1124 | 10 | 400 |
| perltqa_en | 2438 | 6 | 2422 | 10 | 400 |
| perltqa_zh | 2438 | 3 | 2425 | 10 | 400 |
| personamem_128k | 2303 | 418 | 1875 | 10 | 400 |
| personamem_32k | 189 | 102 | 77 | 10 | 400 |
| personamem_v2_128k | 3000 | 7 | 2983 | 10 | 400 |
| split_longmemeval_m | 350 | 0 | 340 | 10 | 150 |
| split_longmemeval_s | 350 | 0 | 340 | 10 | 150 |
| zh4o | 668 | 0 | 658 | 10 | 400 |
| **合计** | **12872** | **538** | **12244** | **90** | **3100** |

## 文件与复现

每个 `<source>/` 下都有 `train.parquet`、`validation.parquet`、`test.parquet`。沿用各来源自己的 schema，后续加载时先按来源适配。

- [manifest.json](manifest.json)：固定策略、各来源数量、validation 原始行号、输入/输出 SHA-256、完整性核验与重叠统计。
- [train_manifest.jsonl](train_manifest.jsonl)、[validation_manifest.jsonl](validation_manifest.jsonl)：每条保留样本的来源文件、0 起算原始行号、输出行号、query 指纹、全字段行指纹和有序历史指纹。
- [removed_train_manifest.jsonl](removed_train_manifest.jsonl)：全部 538 条移除样本的原始位置与原因。
- [counts.csv](counts.csv)：便于读取的逐来源数量。
- [independent_verification.json](independent_verification.json)：独立复查 schema、query 和原始行完整分配的结果。
- [train_test_overlap_audit.md](train_test_overlap_audit.md)、[机器报告](train_test_overlap_audit.json)：处理后 train/test 的完整历史和 query 重叠数量，包含同来源与跨来源口径。

复现时以 manifest 中的原始文件 SHA-256 固定输入，再按各行清单的 `original_row_index` 取行即可恢复相同的样本内容、顺序和成员关系；重新编码后的 Parquet 文件字节不必相同。历史内容指纹采用原始全量审计口径，保留有序 session/消息，仅移除 `has_answer`、`num_tokens`，不作为本次筛除条件。

## 核验结果及范围

写出后已重新读取 train/validation，比对包含全部字段的有序行指纹与 schema；test 副本 SHA-256 与原件相同。全部原始 train 行恰好落入 train、validation、removed 三者之一，各来源 validation 恰好 10 条。

- 同来源 train/test、validation/test、train/validation 的 query 交集均为 0。
- 跨所有来源汇总后，validation/train、validation/test 的 query 交集均为 0。
- 跨来源 train/test 仍有 47 种相同 query，按同来源去重规则保留；不能声称所有来源合并后 train/test 的 query 完全互斥。
- validation 中有 55 行的完整有序历史也出现在 train；跨两侧共有 33 种相同完整历史。PerLTQA 每种语言内全部样本共享一份历史，因此本划分评估的是留出问题表现，不是未见历史泛化。

数据划分已完成，模型输入适配尚未完成。后续必须隔离原始答案/证据标注并检查来源任务格式与上下文容量，再送入 Writer、Reader 或 Task Agent。已知 `data/zh4o/train.parquet` 原始第 542 行（0 起算，`question_id=question_5_821`）的参考答案为 `coleção`，不是 A/B/C/D；该行仍位于 train、未被静默删除或修正，已在 manifest 标记供后续处理。

旧的 `data/audits/history_split_candidate_20261018/`（7195 train / 800 validation）仅保留为历史隔离候选记录，不用于本次训练配置。
