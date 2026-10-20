# 排除 PerLTQA 后的历史隔离验证集候选审计

日期：2026-10-18。状态：`candidate_not_training_ready`。

**已被新划分取代：** 用户后续确定包含 PerLTQA，在每个来源内删除与 test 完全相同的 query，再各留出 10 条 validation。当前使用 [query 去重划分](../data/processed/query_dedup_val10_seed42_20261018/README.md)；下文仅保留当时的历史隔离候选及其审计结论，不再是默认训练清单。

补充说明：本报告的“历史关联组”和“session ID 集合”不能直接解释为完整原文轨迹相同。后续已完成 7 个启用来源的全量原文与 query 比对，修正 PersonaMem 的完整轨迹重复数量，见 [完整历史原文核对](train_test_history_content_audit_20261018.md)。已有候选的完整有序/无序历史交集均为 0，但另有 8 条 validation 的 query 文本在 train 出现过，历史和参考答案不同；候选仍待定稿。

## 范围与结论

用户确定排除 `perltqa_en`、`perltqa_zh`，要求 validation 与训练集隔离来源历史。本次仅做数据审计和候选清单，不实现训练代码、不调用 Writer、不修改原始 Parquet。

剩余 7 个来源原始 train 共 7996 条。问题和答案非空检查，以及 ZH4O 选项标签检查发现 1 条异常；其余 7995 条作为本次候选池。暂以约 10%、seed=42 生成训练 7195 条、validation 800 条。10% 为候选工程设置，不是用户已最终确认的比例。

这证明元数据和当前精确重复关联层面可以形成历史隔离的 train/validation 候选，不表示已经完成全部内容清洗、隐含标注检查、语义近重复审计、原始 persona 身份溯源或上下文容量过滤。

## 候选数量

| 来源 | 原始 train | 关联后的训练历史组 | 候选 train | 候选 validation | 隔离 |
|---|---:|---:|---:|---:|---:|
| LoCoMo | 1136 | 10 | 1013 | 123 | 0 |
| PersonaMem 32k | 189 | 21 | 167 | 22 | 0 |
| PersonaMem 128k | 2303 | 20 | 2052 | 251 | 0 |
| PersonaMem-v2 128k | 3000 | 788 | 2669 | 331 | 0 |
| LongMemEval-s | 350 | 1（与 m 同组） | 350 | 0 | 0 |
| LongMemEval-m | 350 | 1（与 s 同组） | 350 | 0 | 0 |
| ZH4O | 668 | 28 | 594 | 73 | 1 |
| **合计** | **7996** | **848 个不重复关联组** | **7195** | **800** | **1** |

PersonaMem 32k/128k 的关联组存在跨来源重叠，LongMemEval s/m 也共享一个组，因此来源组数不能直接相加。候选 validation 包含 89 个关联组：LoCoMo 1 个、PersonaMem-v1 3 个、PersonaMem-v2 82 个、ZH4O 3 个。

## 历史分组方法

1. LoCoMo：使用 `conv-*` 来源会话分组，全部问题绑定来源历史。
2. ZH4O：使用 `session_*` 来源历史分组，不将内部 session 或 QA 拆成独立组。
3. PersonaMem-v1：先按本地 session ID 的历史前缀分组，再将相同完整会话原文及相同归一化 QA 关联的变体合组。原始 128k 的 60 个训练历史 ID 组由此关联为 20 组；32k 的 36 个训练历史 ID 组关联为 21 组，并与 128k 关联组联合分配。
4. PersonaMem-v2：按来源 `personaN` 分组，合并同一 persona 的不同历史快照和生成批次；3000 条训练样本对应 788 个 persona 组。
5. LongMemEval：先绑定 s/m 同题，再按共享来源 session 关联，审计时去除 `answer_` 标注前缀以核对真实来源身份。全部历史形成一个连通组；排除 `sharegpt_`、`ultrachat_` 前缀后仍是一个连通组。因此本候选保守地将 LongMemEval 全部留在训练侧。

LongMemEval 的“一组”表示共享 session 关系连通，不表示每个 query 的完整历史相同。官方构造会从用户对话、ShareGPT 和 UltraChat 等候选中抽取填充 session，再加入任务对话；公共背景复用与任务历史复用需要区分。本候选没有仅依据“不是该题证据”就把其他对话全部视为可自由共享的公共背景。[官方构造脚本](https://github.com/xiaowu0162/LongMemEval/blob/main/data/custom_history/sample_haystack_and_timestamp.py)

PersonaMem 原始格式包含 `persona_id` 和 `shared_context_id`，但本地转换文件没有保留前者。当前按可识别来源历史和原文变体关联，尚不声称已经证明原始用户身份完全互斥。[官方字段说明](https://github.com/bowen-upenn/PersonaMem)

QA 归一化仅用于离线重复审计：Unicode NFKC、casefold 和空白归一化；问题与答案联合计算 hash，保留数字与标点差别。这些答案和诊断标注不进入 Writer/Reader/Task Agent 输入，不参与额外奖励。当前保留同组中的历史顺序变体，不将所有同题变体简单删除。

## 检查结果与覆盖范围

| 检查项 | train/validation 交集 |
|---|---:|
| 关联历史组 ID | 0 |
| 来源 session ID（全量候选行） | 0 |
| 归一化问题＋答案 hash（全量候选行） | 0 |
| PersonaMem-v1 代表历史完整会话内容 hash | 0 |

PersonaMem-v1 内容检查覆盖全部 97 个可识别历史 ID 的各一个代表快照，共 1367 个 session 出现、968 个不同会话内容 hash。会话指纹保留角色、人物名称和消息正文，去除答案标注与派生 token 数等字段。它不代替所有来源每一行的完整内容审计。

隔离样本：`data/zh4o/train.parquet` 第 542 行（0 起算），`question_id=question_5_821`，参考答案为 `coleção`，不属于合法选项 `A/B/C/D`。未猜测或自动修正其答案。

## 现有 test 的限制

排除 PerLTQA 后现有 test 有 2300 条，其中 2052 条属于候选训练历史组，241 条属于候选验证历史组，只有 7 条属于两侧都未出现的组。原 test 不可直接作为历史独立的最终测试集。

若同时要求保留这些 test 文件，并将全部与 test 历史关联的 train 样本排除，按本次关联规则只剩 1742 条训练候选，均来自 PersonaMem-v2。此数字也是候选审计结果，不表示已通过后续清洗。

当前清单只回答 train/validation 能否隔离，尚未决定保留原 test 的新问题评测口径，还是重新按历史划分最终 test。不能据此报告完整官方 benchmark 的独立测试结果。

## 对最终验证集的建议

- 保持按历史关联组划分和跨长度/顺序变体绑定，固定清单后不反复按验证分数重抽。
- 目标可从约 10% 起步，同时保证少组来源的代表性：LoCoMo 至少 2 个历史组、PersonaMem-v1 至少 3 个组、ZH4O 至少 3 个组。允许样本比例随整组分配略有浮动。
- 当前 800 条草案只有 1 个 LoCoMo 验证历史，尚未按上述建议调整，因此不直接作为最终版。
- 分来源报告 F1、调用次数和总奖励；总体指标同时报告按样本汇总及按来源平均，避免单一来源掩盖退化。
- LongMemEval 在当前草案无 validation 覆盖。若需要它的独立验证，须另行明确并审计允许共享的公共背景边界，或重建互斥背景池，或整源留作外部验证/测试；不能以不同 query ID 代替历史隔离。

## 可检查的文件

- [候选 train 清单](../data/audits/history_split_candidate_20261018/train.jsonl)
- [候选 validation 清单](../data/audits/history_split_candidate_20261018/validation.jsonl)
- [隔离清单](../data/audits/history_split_candidate_20261018/quarantined.jsonl)
- [历史组与来源关联](../data/audits/history_split_candidate_20261018/history_groups.jsonl)
- [数量、检查结果和原始文件 SHA-256](../data/audits/history_split_candidate_20261018/summary.json)

清单使用来源文件、0 起算行号和组 ID 定位，未复制历史正文或参考答案。`summary.json` 固定了 14 个输入 Parquet 的 hash，并记录候选选择方式。正式训练前应完成剩余审计、上下文过滤与 test 协议，再生成最终清单并重新检查交集。
