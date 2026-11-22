# 200-token Writer 目录摘要复查

日期：2026-10-20。检查对象：`runs/slurm_smoke_180646/prepared/`。

## 结论

新提示词与 200-token 预算组合使摘要的实体覆盖、主体动作和相对时间表达明显改善，已能提供比旧版更具体的检索导航。之前发现的若干错误已修复，但关键事实遗漏、未决定状态丢失和未经唯一确认的跨 session 指代仍存在。不能认定“前面的问题已经全部解决”，也不能把摘要当作完整、可靠的事实替代品。

目前不建议仅为这些问题继续增加 token 上限：新摘要平均 97.0 tokens、最长 158 tokens；遗漏孩子情绪的那条仅 61 tokens。剩余问题主要来自事实选择和关系表达。

## 范围与比较方法

- 新旧目录各 78 项，来自 3 条完整 LoCoMo 历史；逐项核对后确认清洗后的历史原文完全相同。
- 使用保存的服务身份、Writer 配置与当前提示词重算目录版本，确认与本次运行记录的版本一致：`827ec5a87b87380eeacdee1367f4758d5f1733f7b5cd71830c8ff262e9f5a913`。
- 阅读全部 78 条新摘要；选择 12 个 session 逐原文对照，覆盖此前重点问题、4 个 smoke query 对应的证据 session 及额外实体/人物归属案例。
- 这是定向审阅，不据此计算总体事实准确率。旧版 96-token 与新版 200-token 同时改变了提示词，不能把改善单独归因于长度上限。
- 未调用模型重新生成摘要、未改写训练数据或原目录，也未进行新旧检索召回率对照实验。

## Token 成本

本地 Qwen3.5-27B tokenizer，`add_special_tokens=False`，只计算 content 正文。

| 项目 | 旧目录 | 新目录 |
|---|---:|---:|
| 配置上限 | 96 | 200 |
| 最短 | 52 | 41 |
| 平均 | 77.62 | 97.03 |
| 中位数 | 79 | 95 |
| 最长 | 96 | 158 |
| 78 项正文合计 | 6054 | 7568 |

正文 token 总量增长约 25.0%，没有摘要达到 200 上限。train 所属 conv-42 的 29 项正文合计从 2222 增至 2632 tokens。这些数字不含目录元数据、JSON 结构、Reader 提示与已读取证据，不能当作完整输入长度。

## 已确认的改善

| session | 旧问题 | 新目录表现 |
|---|---|---|
| conv-42_session_19 | 把对话日期 8 月 22 日写成国际赛获胜日期 | 保留 `won ... yesterday`，结合目录日期正确指向 8 月 21 日 |
| conv-42_session_28 | 漏掉 Fort Wayne 与去年夏天徒步 | 保留地点、时间、徒步与日落照片之间的关系 |
| conv-48_session_4 | 净水器里程碑写成对话当天；漏掉 Bogota 等实体 | 保留 `last week`，恢复 Bogota、last summer、Neal Stephenson 等信息 |
| conv-26_session_14 | 仅笼统提到教堂彩色玻璃，丢失制作主体 | 明确 Caroline 为当地教堂制作彩色玻璃窗，并保留作品含义 |
| conv-26_session_8 | 漏掉蓝色偏好，陶艺和绘画混在一起 | 恢复 Caroline 最喜欢蓝色，分别描述陶艺与绘画活动 |
| conv-42_session_23 | 电影和电子游戏类型混淆 | 明确 Inception 为电影、Cyberpunk 2077 为电子游戏 |
| conv-48_session_2 | 表达成两条蛇都买于巴黎 | 明确买于巴黎的是 Seraphim |
| conv-26_session_18 | 把徒步写成事故次日 | 改为原文的 `yesterday`，不再强行关联为次日 |

conv-42_session_17 保留了 Joanna 为研究下一部电影去 Woodhaven 的明确关联，地点问题的导航信息仍然充分。conv-48_session_22 对 partner 对话、工程计划、两只猫的名字、年龄和来源也有直接原文支持。

## 仍然存在的问题

### 1. 关键反应遗漏：conv-26_session_18

原文 D18:6–9 明确问到孩子如何应对事故，并说明他们害怕，父母安抚并解释哥哥会没事，孩子表现坚强、有恢复力。新摘要保留了事故、儿子无事、大峡谷和徒步，但仍没有这些反应。

这正对应 validation 的问题 `How did Melanie's children handle the accident?`，参考答案为 `They were scared but resilient`。Reader 可以从事故与家庭主题判断 session 相关，但无法仅由 content 了解到这段关键反应。该摘要只有 61 tokens，属于信息选择问题。

### 2. 尚未决定被弱化：conv-42_session_28

原文 D28:5 是正在考虑加入新队，有几个邀请，但尚未决定。新摘要仍写 `plans to join a new gaming team after the next one`，丢失 `haven't decided yet`。Fort Wayne 遗漏已修复，但计划状态的问题仍在。

### 3. 指代缺乏唯一依据：conv-48_session_15

新摘要继续将逃跑的蛇指定为 Seraphim。当前 session D15:16–22 没有名字，D15:30 还明确说 `my snakes`。前五个 session 中 D12:6 也说 snakes，D14:4 提到 Seraphim 的新水箱；这些信息不足以唯一确认第 15 个 session 的逃跑事件属于 Seraphim。应保留“她的一条蛇”而非自动绑定最近出现的名字。

### 4. 更具体的表达也可能引入关系误写

- conv-42_session_19：Nate 寻求推荐后，提出 ACOTAR 询问 Joanna 的看法；摘要写成 Nate 向 Joanna 推荐该系列。后续 Nate 真正向 Joanna 推荐的是 space opera 系列（D19:13–20）。
- conv-48_session_2：原文说 Deborah 在旧家练瑜伽，以及坐在窗边让她平静；新摘要 `specifically on a bench by the window` 容易将两件事合并成在长凳上练瑜伽，原文没有明确这一关系（D2:11–17）。

另一个应保留区别的案例是 conv-48_session_13：摘要的 solar-powered vehicle 来自数据中的图像描述，原话为完成 tough engineering project；验证问题称 robotics project。这里不能直接认定是 Writer 幻觉，但摘要未出现 robotics，仍需 Reader 结合历史理解项目指代；该验证问题的召回未在本次 smoke 中实测。

## 对 Reader 导航能力的判断

已保存的 smoke 只有一个训练 query：国际赛获胜日期。8 条 rollout 中，0、1、2、3、4、5、7 均实际读取了目标 conv-42_session_19；rollout 6 首轮 JSON 语法失败，未执行检索。这支持“当前目录能为这个问题提供有效导航”，不等价于整体检索召回率已验证。

即使读到了正确 session，Task Agent 仍可能给出相对日期、错误日期或冗长答案。因此应分别检查目录导航、Reader 选择与 Task Agent 最终回答，不能把所有答案错误都归因于摘要。

后续优先改进事实选择与关系保真：把反应、原因、意图、犹豫/否定当作可检索事实；跨 session 人物/实体绑定必须有唯一依据；明确谁向谁推荐、询问或建议。可以用上述失败案例构造通用示例，随后重新生成同一批目录进行对照；目前不据此自动修改提示词或提高 token 上限。

完整对照数据：`runs/audits/writer_content_review_20261020/comparison.json`；12 项原文审阅结论与统计：`runs/audits/writer_content_review_20261020/review.json`。
