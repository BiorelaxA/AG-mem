# 完整历史与 JSONL 目录

日期：2026-10-19。

原文存储、目录构建、Reader 检索及训练/评估均改为 JSONL，不再读取 SQLite。复用单位是清洗后的完整历史：有序 session 的正文、清洗后来源 ID、时间等可信元数据共同决定 `memory_id`。query 与参考答案不参与历史标识。

完整历史一致的多个 query 引用同一个 `memory_id`，共用同一版本目录；仅共享部分 session、追加/删除 session、改变顺序或可信元数据的轨迹分别保存完整历史，分别生成全部目录项。Writer 的模型响应缓存也包含完整历史 ID 和目录版本，即使两个轨迹的某一 Writer 输入窗口完全相同，也会分别发起生成请求。空 session 各自保留目录项，沿用确定性的无记录摘要。

## 文件格式

默认路径为 `data/prepared/session_jsonl_v2/`，smoke 为 `data/prepared/smoke_session_jsonl_v2/`。

- `memory_records.jsonl`：每行一条完整历史，含 `memory_id` 与有序 `records`。每个 record 包含目录可读 ID、来源 session ID、顺序、时间和完整清洗后消息。
- `train_examples.jsonl`、`validation_examples.jsonl`、`test_examples.jsonl`：各 query 引用对应 `memory_id`，包含仅供评分使用的私有答案；未使用的 split 可以没有文件。
- `directories.jsonl`：每行一个 session 目录项，按 `(memory_id, version, id)` 定位。包含顶层 `session_id`、`order`、`content`，以及角色/时间等目录元数据。
- `manifest.json`：存储格式、数据输入与准备状态、当前目录版本。
- `directory_manifest.json`：已完成历史/目录版本的条目数与内容 SHA-256。训练只加载完整构建并通过一致性校验的目录。
- `completions.jsonl`：确定性模型响应缓存；Writer 缓存按完整历史隔离。
- `*.index.json`：可重建的 JSONL 字节偏移索引。正式数据保存在 JSONL；索引缺失或文件大小/mtime 不匹配时会重建。

目录行的简化示例：

```json
{"memory_id":"mem_example","version":"writer_version","id":"r_00000_example","session_id":"session_0","order":0,"content":"A short session summary.","task_session":{"task_id":null,"session_id":"session_0"},"actors_source":{"actors":["user"],"sources":[]},"time_order":{"time":null,"order":0}}
```

Reader 根据 query 的 `memory_id` 载入该完整目录并输出目录 `id`；工具仅在同一历史中定位对应 session，返回完整原文和对应摘要。顶层来源 session ID 保留用于追溯；目录 `id` 同时区分重复来源 ID 和不同位置。消息中的评测标注不进入清洗后原文，gold-derived `answer_` 来源 ID 前缀仍按原约定移除。

Writer 仍为第 t 个 session 读取同一轨迹的当前 session 与前 5 个 session，仅概括当前 session。摘要正文现上限 200 tokens，完整 JSON 输出预算 384 tokens（2026-10-19 后续更新）。每生成一个合法目录项即写入 JSONL，中断后可以在同一轨迹和版本下续建；完整结束后发布目录校验清单。已完成的目录文件重新加载后直接用于训练，保留 8-rollout GRPO、F1 outcome 和每次检索 0.01 成本；2026-10-20 新增每条轨迹最多一次 0.1 的 Reader 格式惩罚，见 `format_reward_20261020.md`。

## 从 SQLite 切换

旧 `data/prepared/session_v1/`、`data/prepared/smoke_session_v1/` 及其数据库保留。当前配置改用新目录，先从固定 processed Parquet 生成新格式：

```bash
python -m agmem --config configs/smoke.yaml prepare --limit-per-source 2
python -m agmem --config configs/smoke.yaml build-directories
CUDA_VISIBLE_DEVICES=0 python -m agmem --config configs/smoke.yaml train --max-groups 1
```

全量流程：

```bash
python -m agmem prepare
python -m agmem build-directories
CUDA_VISIBLE_DEVICES=0 python -m agmem train
```

不重新划分或改写原始/processed Parquet。旧目录与 checkpoint 不冒充新版本产物；恢复训练继续执行数据、目录、模型指纹和训练配置的一致性检查。

## 验证

- `python -m unittest discover -s tests -v`：24 项全部通过；`python -m compileall -q agmem tests` 通过。
- CPU 回归覆盖完整历史复用、部分共享轨迹独立 Writer 请求、相同输入也不跨轨迹命中响应缓存、JSONL 重新打开与 UTF-8 字节定位、删除索引后重建、同一轨迹中断续建、版本隔离，以及从重载的 JSONL 采样完整 8-rollout group。
- prepare → build-directories → 训练环境装配的集成检查使用临时 Parquet、JSONL 与模拟 Writer，确认正式管线不依赖数据库。
- 真实 processed 数据的 smoke prepare 已完成：2 条 train、2 条 validation、3 条完整历史、78 个 session。
- 真实 27B 的新格式目录构建和实际 GPU 梯度更新需要在 GPU 节点执行上述命令；CPU 模拟结果不代表显存验证通过。
