# QunYou-LLM 🗣️

**群聊数据驱动的对话大模型微调框架**  
Fine-tuning LLMs on Group Chat Data — Data Pipeline → QLoRA → RAG → Deployment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 项目简介 / Overview

从 QQ / 微信群聊记录出发，构建完整的 **数据清洗 → 大模型微调 → RAG 检索增强 → 模型部署** 全链路。

通过 QLoRA 参数高效微调技术，在消费级 GPU (RTX 4090) 上将 Qwen2.5-3B 适配为群聊风格对话模型，并配套语义检索系统实现知识增强。

A complete LLM fine-tuning pipeline from group chat data: **Data Cleaning → QLoRA Fine-tuning → RAG Retrieval-Augmented Generation → Model Deployment**. Built on Qwen2.5-3B with consumer-grade GPU.

---

## 🚀 快速开始 / Quick Start

### 安装

```bash
pip install -r requirements.txt
```

### 1️⃣ 数据清洗管道

```bash
# 准备数据: 将你的聊天记录 JSON 放到项目根目录
# 格式参考 data/sample/liaotian_sample.json

# 运行清洗管道
python scripts/01_data_pipeline.py
```

输出:
- `data/formatted/train.jsonl` — 训练集
- `data/formatted/val.jsonl` — 验证集
- `data/formatted/cpt_corpus.txt` — 继续预训练语料
- 自动生成数据质量报告

### 2️⃣ QLoRA 微调

```bash
# 单卡 RTX 4090 训练
python scripts/02_train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --output_dir output/qlora-checkpoints
```

训练完成后自动输出:
- LoRA 适配器 `output/qlora-checkpoints/final_adapter/`
- 合并后完整模型 `output/qlora-checkpoints/merged_model/`
- Loss 曲线图 `output/qlora-checkpoints/loss_history.png`

### 3️⃣ RAG 向量库构建

```bash
python scripts/04_build_rag.py
```

输出:
- `data/rag/faiss_index.bin` — FAISS 向量索引
- `data/rag/messages_meta.json` — 消息元数据

### 4️⃣ 启动推理服务

```bash
# 方式 A: 基础推理 API
python scripts/serve.py --port 8000

# 方式 B: RAG 增强 API
python scripts/rag_chat.py --serve --port 8001

# 方式 C: 工具调用 + RAG API
python scripts/tool_rag.py --serve --port 8002
```

```bash
# 调用示例
curl -X POST http://localhost:8001/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "晚上吃什么"}'
# → {"response": "烧烤自助", ...}
```

### 5️⃣ 工具调用链 (Tool-use)

```bash
# 先查天气，再 RAG 聊天
python scripts/tool_rag.py \
  --query "今天适合出去玩吗" \
  --tool get_weather \
  --tool-params '{"city":"北京"}'
# → 🔧 get_weather → 🌤 北京天气: 晴, 28°C
# → 💬 "明天才出门"
```


---

## 📁 项目结构 / Project Structure

```
qunyou-llm/
├── scripts/
│   ├── 01_data_pipeline.py      # 数据清洗管道
│   ├── 02_train.py              # QLoRA 微调 (含 Loss 绘图)
│   ├── 03_export_ollama.sh      # Ollama 导出脚本
│   ├── 04_build_rag.py          # RAG 向量库构建
│   ├── rag_chat.py              # RAG 检索增强生成 (核心函数)
│   ├── tool_rag.py              # Tool-use 模式 (API + RAG 链)
│   ├── serve.py                 # 推理 API 服务器
│   └── convert_to_gguf.py       # GGUF 格式转换
├── data/
│   └── sample/
│       └── liaotian_sample.json # 数据格式示例
├── output/
│   └── gguf/
│       └── Modelfile            # Ollama 模型配置
├── requirements.txt             # Python 依赖
└── README.md                    # 本文件
```

---

## 🔧 技术栈 / Tech Stack

| 领域 | 技术 |
|------|------|
| 基座模型 | Qwen2.5-3B-Instruct |
| 微调框架 | HuggingFace Transformers + TRL + PEFT |
| 量化 | bitsandbytes 4-bit NF4 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 |
| 向量检索 | FAISS (IVF) |
| 部署 | FastAPI + Ollama / llama.cpp |
| 硬件 | NVIDIA RTX 4090 24GB |

---

## 📄 License

MIT

---

## 🙏 致谢 / Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen) — 基座模型
- [HuggingFace TRL](https://github.com/huggingface/trl) — 微调框架
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — 推理引擎
- [Ollama](https://ollama.com) — 模型部署
