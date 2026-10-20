# 新 Conda 环境与依赖

本仓库提供 `environment.yml`、`requirements.txt`、`requirements-serving.txt` 和 `requirements-dev.txt`。当前环境未修改；GPU 依赖组合尚待实际分配的双 80GB 节点验证。

候选组合为 Python 3.11、PyTorch 2.10.0、Transformers 5.5.3、vLLM 0.19.1。不是沿用当前无法识别 qwen3_5 的 Transformers 4.57.3。

vLLM 0.19.1 的官方 [CUDA 依赖](https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/cuda.txt) 固定 torch 2.10.0；[通用依赖](https://github.com/vllm-project/vllm/blob/v0.19.1/requirements/common.txt) 排除了 Transformers 5.0–5.4 与 5.5.0，因此这里选用 5.5.3。其[支持列表](https://docs.vllm.ai/en/v0.19.1/models/supported_models/)包含 `Qwen3_5ForConditionalGeneration`。这是依据元数据约束选出的候选，尚未完成新环境 resolver 与 CUDA 实机验证。

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate agmem
nvidia-smi
uv pip install -r requirements-serving.txt --torch-backend=auto
python -m pip install --no-deps -e .
python -m pip check
python -m unittest discover -s tests -v
```

Conda 负责 Python 环境；GPU PyTorch / vLLM 使用 pip wheel。按 [vLLM 0.19.1 安装说明](https://docs.vllm.ai/en/v0.19.1/getting_started/installation/gpu/)，默认二进制使用 CUDA 12.9，并提供其他 CUDA 变体；`uv --torch-backend=auto` 检查实际驱动并选择 PyTorch 源。若节点驱动不支持默认 vLLM wheel，需要按照该文档选择**同一 vLLM 版本**的兼容变体或在节点构建，不能只替换 torch、强行绕过依赖检查。

训练只使用 `requirements.txt` 中的库；`requirements-serving.txt` 在此基础上增加 27B 推理服务，同一个 Conda 环境可供两个进程使用。可选开发依赖安装命令：

```bash
uv pip install -r requirements-dev.txt --torch-backend=auto
```

当前训练后端是原生 PyTorch 和本仓库完整轨迹 GRPO；不依赖某个 Trainer 对自定义多轮环境的隐式实现。CPU AdamW master weights 与状态使用 FP32，Reader 仍为全参更新。

TensorBoard 已列入 `requirements.txt`，使用 PyTorch 的 `SummaryWriter` 写入训练和验证指标，不需要安装 TensorFlow。已建好的 `agmem` 环境只需补装：

```bash
python -m pip install 'tensorboard>=2.20,<3'
```

训练默认写入 `<training.output_dir>/tensorboard/`。缺少依赖时会在加载模型前报告安装命令；可在本地配置设置 `training.tensorboard: false` 关闭。已完成的训练通过 `python -m agmem export-tensorboard <run_dir>` 导入。后台运行、控制台日志与浏览器访问命令见根目录 README。

Qwen3.5 的 Gated DeltaNet 在 Transformers 5.5.3 中有纯 PyTorch fallback；[模型源码](https://github.com/huggingface/transformers/blob/v5.5.3/src/transformers/models/qwen3_5/modeling_qwen3_5.py)同时支持可选的 flash-linear-attention、causal-conv1d 快速路径。基础依赖先验证正确性；若后续安装可选 CUDA kernel，必须在分配节点与固定 torch/CUDA 组合下验证 forward、generation/teacher-forcing logprob 及 backward 一致性，然后更新锁定清单。

新环境通过单组真实模型梯度检查后保存最终环境：

```bash
python -m pip freeze > requirements.lock.txt
conda env export --from-history > environment.resolved.yml
```

保存节点 GPU/driver 型号、CUDA wheel 变体、上述锁定文件、模型指纹与 run.json，作为实际实验复现入口。根目录候选 requirements 不冒充已经在真实 GPU 上通过的最终锁文件。
