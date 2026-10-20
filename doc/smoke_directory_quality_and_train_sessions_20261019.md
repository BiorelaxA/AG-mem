# Smoke 目录摘要质量与训练集 session 数量

日期：2026-10-19。只做读取、统计和人工对照，未调用 Writer 重建目录，未修改模型、数据或 96-token 配置。

## 范围与口径

96 tokens 是 Writer 生成的目录项 `content` 上限，不是 Reader 动作输出上限。

检查对象为 `data/prepared/smoke_session_jsonl_v2/` 中已生成的 78 个目录项及其完整 session 原文。smoke 的两个 train query 共用 conv-42 的 29 个 session；两个 validation query 分别对应 conv-48 的 30 个 session 和 conv-26 的 19 个 session，共 3 条不同历史。

全部 78 个摘要均阅读并进行精确 token 统计；其中 20 个 session 人工逐原文对照，包括四个 smoke 问题对应的 session、最长的七个 session 及较短的对照项。这是定向检查，不据此估计所有目录的事实错误率，也没有执行检索召回率对照实验。

长度以本地 `/home/share/models/Qwen3.5-27B` tokenizer、`add_special_tokens=False` 计算。原文长度是工具实际返回的 `raw_text`，包含角色和消息字段；因此不等于仅正文的长度。

## 判断

现有摘要通常能指示 session 的主要人物和话题，可以作为粗粒度导航；但不能据此声称已准确覆盖所有关键事实。已经发现相对时间被压平、独特地点缺失、动作归属变得笼统，以及不充分的跨 session 指代推断。

96 tokens 可以作为低成本导航基线，但不是“全部关键内容都能保留”的保证。当前更直接的问题是摘要方式：出现问题的摘要只有 56、66、68、71、83 tokens，并没有用满 96；仅提高上限不能保证问题消失。应优先保留可检索的事实锚点、事件时间、主体/动作、否定和计划状态，减少寒暄、泛泛支持语和元数据中已有的 session 日期重复。

较长、多主题 session 可进一步比较 128/192-token 预算，但应依据检索效果与目录输入成本决定，不能仅从更长的摘要推断更好。若调整 `content_max_tokens`，也要给完整 JSON 的 `max_output_tokens` 留足空间；当前完整输出预算为 192 tokens。

## Token 统计

| 项目 | 最小 | 平均 | 中位数 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|
| session 原文 raw_text | 614 | 1255.2 | 1177 | 2309 | 2501 |
| 目录 content | 52 | 77.6 | 79 | 94 | 96 |

78 条摘要合计 6054 tokens；8 条达到或超过 90 tokens，只有 1 条恰好为 96。摘要总长度约为上述 raw_text 总长度的 6.18%，但这只是压缩比例，不是信息覆盖率。

## 原文对照的具体例子

| session | 原文 / 摘要 tokens | 观察 |
|---|---:|---|
| conv-42_session_17 | 1227 / 78 | 摘要保留 Joanna、研究 road trip、Woodhaven、Midwest、图书馆旧书及剧本灵感，足以导航到地点问题对应原文。 |
| conv-42_session_19 | 1302 / 56 | 原文在 2022-08-22 的 session 中说国际赛是 yesterday 获胜，即 2022-08-21；摘要写成 “On Aug 22, Nate won…”，容易把对话日期当事件日期。 |
| conv-42_session_28 | 1958 / 71 | 原文明确说 Joanna 去年夏天在 Fort Wayne 附近徒步拍日落；这是本条历史中唯一一次 Fort Wayne，目录却完全遗漏该地点和徒步事件。相关 query 可能因此漏选。 |
| conv-48_session_4 | 2501 / 83 | 净水器、瑜伽、书籍和纪念物等主题保留，但原文净水器里程碑是 last week，摘要写成 session 当天；Bogota 旅行等独特细节缺失。 |
| conv-26_session_14 | 2309 / 66 | 原文明确 Caroline 为当地教堂制作彩色玻璃；摘要只说讨论教堂彩色玻璃窗，没有明确创作者和制作动作。 |
| conv-26_session_18 | 1308 / 68 | 原文有孩子害怕、父母安抚、表现坚强的直接证据；摘要保留事故主题却遗漏这些情绪细节。另外原文登山时间仅说 yesterday，摘要写成事故 following day，增加了没有依据的时间关系。 |

目录无需直接回答每个问题，Task Agent 最终仍应读原文。但摘要至少应让 Reader 看到相关主题和独特实体，不能依赖 Reader 从未出现的地点或模糊的动作归属中猜测。

作为表达方式的对照，人工把 conv-42_session_28 改写为下面的事实列表，仍只有 **66 tokens**，并保留 Fort Wayne 等原摘要漏掉的信息：

> Nate: tournament postponed; previous event missed finals; considering a new team (undecided); starting YouTube gaming videos; made coconut ice cream; bought a third turtle. Joanna: producers liked yesterday's script pitch; writing a Midwest-town thriller; sunset photo from last summer's hike near Fort Wayne. They plan to meet tomorrow.

这是人工审阅示例，不是新的 Writer 输出，也没有写回训练目录。另两个手写示例为 73 和 74 tokens；它们说明原有预算内仍有改进空间，不证明模型改提示后必然达到同样效果。

## 完整训练集 session 数量

按当前规则计算：同一清洗后的完整有序历史只建一套目录；不同历史即使共享某些 session，也分别建目录。不包含 validation/test，仅统计有效 train。

| 来源 | 有效 train QA | 不同完整历史 | 所需 session 目录项 |
|---|---:|---:|---:|
| LoCoMo | 1124 | 10 | 272 |
| PerLTQA EN | 2422 | 1 | 3004 |
| PerLTQA ZH | 2425 | 1 | 3004 |
| PersonaMem 128k | 1875 | 294 | 5735 |
| PersonaMem 32k | 77 | 40 | 214 |
| PersonaMem v2 128k | 2983 | 1319 | 221813 |
| LongMemEval m | 340 | 340 | 161599 |
| LongMemEval s | 340 | 340 | 16223 |
| ZH4O | 657 | 28 | 1752 |
| **合计** | **12243** | **2373** | **413616** |

当前训练目录的工作量约为 **41.36 万个目录项**。PerLTQA 中每种语言的 3004 个位置包括 123 个空 session，占位目录仍保留；这些空项使用固定摘要，不需要 Writer 调用。

三个统计口径不要混淆：

- 当前完整历史复用规则：**413616** 个目录项，是本次主要数字。
- 若把每个 query 附带的 sessions 全部重复累加：**15350616** 次出现，不等于需要生成这么多目录项。
- 若不考虑所属历史、只对 session 消息体全局去重：**230769** 种正文；当前实现不采用这个数量来复用目录。

处理后 Parquet 的 train 有 12244 行，其中 1 条 ZH4O 非法答案标签在适配时隔离，因此有效 train 为 12243。

## 数量来源与可复核材料

全量新版 JSONL 尚未执行 prepare，本次没有为统计重建全量数据。只读使用此前已经完成全量清洗的 `data/prepared/session_v1/artifacts.sqlite` 索引：核对固定划分 manifest SHA-256 和当前 adapter 版本，提取 train 引用的完整历史，并重新计算每条历史的 `memory_id`，确认其身份规则与当前 JSONL 一致。按历史内 records 数累加，不对不同历史的 session 正文再次去重。

详细机器报告位于 `runs/audits/smoke_directory_review_20261019/`：

- `smoke_token_stats.json`：78 项的逐 session token 长度和统计。
- `manual_review.json`：20 个原文对照项及具体判断。
- `manual_rewrite_examples.json`：仅供审阅的手写改写及实测长度。
- `train_session_counts.json` / `.csv`：训练集分来源统计、口径与完整历史数量。
