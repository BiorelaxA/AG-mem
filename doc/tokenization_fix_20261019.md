# Transformers 5 分词返回值修正

日期：2026-10-19。

真实 smoke 训练在 `torch.tensor([prompt_ids])` 处报 `Could not infer dtype of tokenizers.Encoding`。当前环境为 PyTorch 2.10.0+cu129、Transformers 5.5.3、tokenizers 0.22.2。

## 原因与修改

Transformers 5.5.3 的 `apply_chat_template` 默认 `return_dict=True`，返回 `BatchEncoding`。本地 Qwen3.5 tokenizer 返回的对象包含 `input_ids`、`attention_mask` 两个字段，`len(result)` 为 2；按整数索引访问时会得到底层 `tokenizers.Encoding`，并非 token ID。旧代码将这个对象直接当作整数列表构造张量，因此失败，也会错误计算 Reader/Task Agent 的上下文长度。[对应版本接口源码](https://github.com/huggingface/transformers/blob/v5.5.3/src/transformers/tokenization_utils_base.py)

`agmem.policy.chat_token_ids` 统一显式指定 `return_dict=False`、`return_tensors=None`、`padding=False`、`truncation=False`，验证结果为非空的一维整数 ID 列表。Reader 编码与 Task Agent 容量检查均使用这一函数，Reader 输入张量显式使用 `torch.long`。保留原来的聊天模板、关闭 thinking、输出预算和容量限制。

## 验证

- `python -m unittest discover -s tests -v`：27 项通过。新增测试使用真实 `BatchEncoding` 和 `tokenizers.Encoding` 重现默认返回类型，验证 Reader/Task Agent 按实际 token 数判定容量，而非按字段数量计数。
- `python -m compileall -q agmem tests` 通过。
- 在实际 `agmem` 环境加载本地 4B 与 27B tokenizer；同一短提示均返回 17 个整数 ID，成功构造形状 `[1, 17]`、类型 `torch.int64` 的 CPU 张量。没有加载真实模型权重或执行 GPU 梯度更新。

首行 `The fast path is not available ... Falling back to torch implementation` 是可选 Gated DeltaNet 加速依赖缺失时的性能提示；模型选择 PyTorch 实现。它与本次分词对象转换错误属于不同位置的处理。

## 重新运行

这次失败只留下 `run.json`，没有 group 日志、训练指标或 checkpoint。该目录已保留为 `runs/smoke_session_grpo_failed_tokenization_20261019T181714/`，可以用原命令重新开始：

```bash
CUDA_VISIBLE_DEVICES=0 python -m agmem \
  --config configs/smoke.yaml train --max-groups 1
```

分词修正不改变原文或 Writer 目录，无需重做数据准备和目录生成。实际 GPU 训练结果仍以重新执行该命令为准。
