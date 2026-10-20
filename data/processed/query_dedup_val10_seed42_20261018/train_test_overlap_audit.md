# 处理后 train/test 的完整历史与 query 重叠

日期：2026-10-18。比较对象是处理后的 12244 条 train 与保留的 3100 条 test，不包含 validation。

完整历史按一条 QA 携带的全部 `haystack_sessions` 比较，保留 session 边界、session 顺序、消息顺序及所有真实消息字段；只移除 `has_answer`、`num_tokens`。不以共享部分 session 或相同 session ID 集合代替完整历史相同。query 为 `question` 原始字符串完全相同。

当前文件 SHA-256 与划分 manifest 一致；query 从当前 Parquet 重新读取核对，历史使用已全量核验且保留原行内容的指纹与训练清单映射。

## 各来源内部的 train/test

共享轨迹种数为两侧共同出现的不同完整历史数；命中样本数按 QA 行计数，同一轨迹可对应多条 query。

| 来源 | train 条数 | test 条数 | 共享完整轨迹种数 | train 命中条数 | test 命中条数 | 共享 query 种数 |
|---|---:|---:|---:|---:|---:|---:|
| locomo | 1124 | 400 | 10 | 1124 | 400 | 0 |
| perltqa_en | 2422 | 400 | 1 | 2422 | 400 | 0 |
| perltqa_zh | 2425 | 400 | 1 | 2425 | 400 | 0 |
| personamem_128k | 1875 | 400 | 60 | 1567 | 333 | 0 |
| personamem_32k | 77 | 400 | 28 | 58 | 193 | 0 |
| personamem_v2_128k | 2983 | 400 | 253 | 723 | 296 | 0 |
| split_longmemeval_m | 340 | 150 | 0 | 0 | 0 | 0 |
| split_longmemeval_s | 340 | 150 | 0 | 0 | 0 | 0 |
| zh4o | 658 | 400 | 28 | 658 | 400 | 0 |

## 所有来源合并后的 train/test

| 比较项 | 共享内容种数 | train 命中条数 | test 命中条数 |
|---|---:|---:|---:|
| 完整有序历史 | 381 | 8977 | 2422 |
| 原始 query | 47 | 79 | 58 |
| 完整历史＋query 同时相同 | 0 | 0 | 0 |

## 跨来源重复

| train 来源 | test 来源 | 共享 query 种数 | train 命中条数 | test 命中条数 | 共享完整轨迹种数 |
|---|---|---:|---:|---:|---:|
| personamem_128k | personamem_32k | 40 | 71 | 51 | 0 |
| personamem_32k | personamem_128k | 7 | 8 | 7 | 0 |

同来源 train/test 的 query 重叠均为 0。全来源合并后的 query 重复来自跨来源，按本次同来源去重规则保留；此报告仅统计，不修改数据。

忽略 session 排列的完整历史重叠、包含 session 日期的重叠等辅助口径及全部精确计数见 [机器报告](train_test_overlap_audit.json)。
