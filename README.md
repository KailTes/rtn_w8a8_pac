# RTN W8A8 INT8 量化 + PPL 评测

纯 safetensors 离线 INT8 量化（CPU，无需 GPU/NPU），适配昇腾 NPU + omni-infer 推理。

## 特点

- **纯 CPU 量化**：不依赖 GPU/NPU/模型代码，直接对 safetensors 权重做 RTN per-channel INT8 量化
- **compressed-tensors 格式**：输出兼容 vllm / omni-infer INT8 推理
- **一键评测**：集成 lm-eval WikiText-2 PPL 评测
- **多模型支持**：自动检测模型类型，PanGu 使用 `run_pangu.sh`，其他模型直接启动 vllm serve

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/KailTes/rtn_w8a8_pac.git
cd rtn_w8a8_pac
```

### 2. 依赖

```bash
pip install torch safetensors          # 量化脚本
pip install "lm-eval[api]"             # PPL 评测
```

### 3. 量化

```bash
# 纯 CPU 量化，不需要 GPU/NPU
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python3 quantize_safetensors_int8.py \
    --model /path/to/fp16-model \
    --output /path/to/output-w8a8
```

多 shard safetensors 自动处理，会重新生成 `model.safetensors.index.json`。

### 4. 验证推理 (omni-infer / vllm)

```bash
# 启动服务
bash rtn_eval.sh serve /path/to/output-w8a8

# 测试推理
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"test","prompt":"hello","max_tokens":50}'

# 停止服务
bash rtn_eval.sh stop
```

### 5. PPL 评测

```bash
# FP16 baseline
bash rtn_eval.sh eval_fp16 /path/to/fp16-model

# 停止 FP16 服务
bash rtn_eval.sh stop

# W8A8 评测
bash rtn_eval.sh eval_w8a8 /path/to/fp16-model

# 或一键全部
bash rtn_eval.sh all /path/to/fp16-model
```

## 在昇腾私密服务器上使用

### 环境准备

```bash
# SSH 到服务器
ssh root@<server-ip>

# 进入容器 (omni-infer)
docker exec -it <container-name> bash

# 克隆仓库
cd /data
git clone https://github.com/KailTes/rtn_w8a8_pac.git
cd rtn_w8a8_pac

# 安装 lm-eval (如果未安装)
pip install "lm-eval[api]"
```

### 量化私密模型

```bash
# /models 通常是只读挂载，用 QUANT_OUTPUT_BASE 指定可写目录
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
QUANT_OUTPUT_BASE=/data/models \
bash rtn_eval.sh quantize /models/your-private-model
# 输出: /data/models/your-private-model-RTN-W8A8
```

### PanGu 模型

PanGu 模型自动检测 (`pangu_v2_moe`)，使用 `run_pangu.sh` 启动 8 卡 TP 服务：

```bash
# 设置 run_pangu.sh 路径 (如果不在默认位置)
export RUN_PANGU_SCRIPT=/path/to/run_pangu.sh

# 量化
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
QUANT_OUTPUT_BASE=/data/models \
bash rtn_eval.sh quantize /data/weights/pangu_v2/92B/iter_0059000_hf/

# 评测
bash rtn_eval.sh eval_w8a8 /data/weights/pangu_v2/92B/iter_0059000_hf/
```

### 通用模型 (Qwen3 等)

```bash
# 指定 NPU 设备和端口
ASCEND_RT_VISIBLE_DEVICES=3 \
API_BASE=http://localhost:8000 \
QUANT_OUTPUT_BASE=/data/models \
bash rtn_eval.sh all /models/Qwen3-0.6B
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_BASE` | `http://7.150.11.4:8000` | API 地址 |
| `SERVE_PORT` | `8000` | vllm serve 端口 |
| `TP_SIZE` | `1` | tensor parallel size |
| `QUANT_OUTPUT_BASE` | 模型所在目录 | 量化输出的父目录 |
| `RUN_PANGU_SCRIPT` | 自动查找 | `run_pangu.sh` 路径 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | NPU 设备 |
| `VLLM_USE_V1` | `0` | vllm v1 引擎开关 |
| `EXTRA_SERVE_ARGS` | 空 | 额外 vllm serve 参数 |

## 命令一览

```
bash rtn_eval.sh quantize  /path/to/model   # RTN W8A8 离线量化 (CPU)
bash rtn_eval.sh serve     /path/to/model   # 启动 vllm serve
bash rtn_eval.sh eval      /path/to/model   # 通过 API 评测 PPL
bash rtn_eval.sh stop                       # 停止 vllm serve
bash rtn_eval.sh eval_fp16 /path/to/model   # FP16: serve + eval
bash rtn_eval.sh eval_w8a8 /path/to/model   # W8A8: serve + eval
bash rtn_eval.sh all       /path/to/model   # 全部依次执行
```

## 量化原理

RTN (Round-To-Nearest) per-output-channel symmetric INT8:

```
scale = abs(weight).max(dim=output_channel) / 127
int8_weight = round(weight / scale).clamp(-128, 127)
```

跳过的层：`embed`, `norm`, `lm_head`（这些层对量化敏感，保持原精度）。

## 参考结果 (Qwen3-0.6B, WikiText-2)

| 平台 | FP16 | W8A8 RTN | 劣化 |
|------|------|----------|------|
| NVIDIA RTX 5090 (torchao) | 60.0424 | 60.9461 | +1.50% |
| 昇腾 910B2 (omni-infer v1.0.0) | 60.0255 | 61.1840 | +1.93% |
