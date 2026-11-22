# Train/Test 完整历史原文与 query 重复核对

日期：2026-10-18。范围：当时启用的 7 个来源、14 个原始 Parquet 文件、10296 条 QA 样本。PerLTQA 不包含在本报告范围内。

后续已重新启用 PerLTQA；包含全部 9 个来源的原始文件审计见 `data/audits/train_test_trajectory_content_20261018/`。当前采用 [按 query 去重、每来源留出 10 条的划分](../data/processed/query_dedup_val10_seed42_20261018/README.md)，不再使用本文末尾的 7195/800 历史隔离候选。下文数字仍描述未修改的原始 train/test。

本次结论来自全量历史原文比对，不以 session ID 集合替代原文比较。原始数据未修改，也未调用 Writer 或训练模型。

## 比较单位

一条 QA 样本包含 `question`、`answer` 和 `haystack_sessions`。后者是多个 session 的有序列表，每个 session 又是多条消息的有序列表。这里的“完整历史轨迹”指该样本实际携带的整份历史，不保证它是单一 Agent 在环境中连续发生的一条执行轨迹。

```text
一条 QA 样本
├── query
├── 完整历史 = [session 1, session 2, ..., session N]
│   └── 每个 session = [message 1, message 2, ...]
└── reference answer
```

原文比对口径：

- 从消息中移除 `has_answer`、`num_tokens` 两类标注/派生字段。
- 保留其他全部消息字段，包括角色、正文、人物名称、消息时间和 `dia_id`；正文不做大小写、空白或 Unicode 归一化。
- 保留 session 边界、session 顺序和消息顺序。JSON 对象字段顺序不影响相等性。
- 先计算每个清洗后完整 session 的 SHA-256，再对有序 session 指纹列表计算完整历史 SHA-256。session ID 不参与内容指纹。
- 另计算包含完整 `haystack_dates` 的指纹；在本报告 7 个来源中，其跨 split 命中条数与有序原文指纹结果一致。
- query 相同默认指原始 `question` 字符串逐字相同；格式归一化后的比较另存机器报告。

## 全量结果

下表的三个重复数量列均按“测试样本条数”统计：该测试样本的对应内容可在同来源 train 中找到。它们不是不同轨迹数或不同 query 文本数。

| 来源 | 每条历史的 session 数范围 | test 总条数 | 完整有序历史相同 | query 相同 | 历史与 query 同时相同 |
|---|---:|---:|---:|---:|---:|
| LoCoMo | 19–32 | 400 | 400 | 2 | 2 |
| LongMemEval-s | 38–62 | 150 | 0 | 0 | 0 |
| LongMemEval-m | 460–490 | 150 | 0 | 0 | 0 |
| ZH4O | 50–94 | 400 | 400 | 0 | 0 |
| PersonaMem 32k | 4–7 | 400 | 280 | 124 | 59 |
| PersonaMem 128k | 16–20 | 400 | 369 | 282 | 68 |
| PersonaMem-v2 128k | 81–201 | 400 | 296 | 5 | 0 |

不同重复内容的数量如下：

| 来源 | 跨 train/test 共享的不同完整有序历史数 | 跨 train/test 共享的不同 query 文本数 |
|---|---:|---:|
| LoCoMo | 10 | 2 |
| LongMemEval-s | 0 | 0 |
| LongMemEval-m | 0 | 0 |
| ZH4O | 28 | 0 |
| PersonaMem 32k | 69 | 96 |
| PersonaMem 128k | 96 | 271 |
| PersonaMem-v2 128k | 253 | 5 |

若仅忽略 session 之间的排列顺序，仍保留各 session 内消息顺序和重复 session 的数量，PersonaMem 32k 的匹配测试条数从 280 增至 289；其他来源的匹配测试条数不变。

LongMemEval 的整份历史无重复，不排除它们共享部分 session。此前“共享 session 关系将 LongMemEval 连成一个组”的结果描述的是局部重叠连通关系，不能解释成全部问题共用一条完全相同的历史。

## 为什么先前的 session ID 集合统计不能代替这些数字

已经找到具体反例：`data/personamem_128k/train.parquet` 第 0 行和第 34 行（均为 0 起算）具有完全相同、顺序也一致的 20 个 session ID，但最后一个 session 分别包含 31 条、29 条消息；较短者是较长者的相同消息前缀。两条样本的完整历史实际不一样。

因此，先前按 ID 集合给出的 PersonaMem 32k `388/400`、128k `400/400`、v2 `374/400`，只能表示 ID 集合出现过，不能作为完整原文轨迹完全相同的数量。对应的有序原文核对结果应分别修正为 `280/400`、`369/400`、`296/400`。

## query 重复的解释

相同 query 不自动等于相同完整输入或相同 QA：

- PersonaMem-v2 有 5 条测试 query 与 train 逐字相同，但完整历史均不同，原始参考答案也均不同。
- PersonaMem 32k/128k 的 `all_options` 单独存储。相同 `question` 不代表四个选项相同。若同时要求完整有序历史、question 和选项列表都相同，32k 命中 1 条测试样本，128k 命中 0 条。
- 原始“question＋answer”组合跨 split 重复的测试样本条数为：LoCoMo 2、PersonaMem 32k 81、PersonaMem 128k 247；其余 4 个来源为 0。这与不同 QA 组合数是两种计数，128k 的不同重复 QA 组合数为 240。
- 上述输入比较是数据样本级比较，不是声称多步检索后最终 Task Agent 实际收到相同证据。

另查到跨来源重复 query：PersonaMem 128k 的 train 与 32k 的 test 共享 52 种 query 文本，覆盖后者 71 条测试样本；32k 的 train 与 128k 的 test 共享 15 种 query 文本，覆盖后者 18 条测试样本。两方向均未发现完整有序历史跨来源完全相同。

## 对已有 validation 候选的补充核验

对先前 7195 条 train / 800 条 validation 候选，用本次全量原文指纹重新检查：完整有序历史交集为 0；忽略 session 排列后的完整历史交集也为 0；原始 QA 组合交集为 0。

但两侧存在 8 种相同 query 文本，涉及 8 条 validation 样本，参考答案及完整历史不同。当前候选满足本次检查口径下的完整历史隔离，不等于 query 文本也完全隔离；最终划分应明确是否额外要求 query 去重。

这些结果不覆盖语义改写、历史前缀/局部片段近重复或真实用户身份关系。完整历史不同也不能反推它们完全无关联。

## 可复核文件

- [7 个来源的完整统计与比较方法](../data/audits/active_sources_history_content_20261018/summary.json)
- [逐样本内容指纹与原文件行号](../data/audits/active_sources_history_content_20261018/row_fingerprints.jsonl)
- [重复历史/query 的具体行号和 query 示例](../data/audits/active_sources_history_content_20261018/overlap_examples.json)
- [独立复算、ID 反例及候选验证集检查](../data/audits/active_sources_history_content_20261018/additional_checks.json)

逐样本指纹包含来源、split、行号、question ID、session 数以及有序/无序历史、query、QA 的指纹，不复制历史正文或参考答案。全部行数和跨 split 指纹交集已独立复算通过。
