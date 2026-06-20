#!/usr/bin/env python3
"""
群聊模型推理服务器 — OpenAI 兼容 API
无需 Ollama，直接加载微调后的 HF 模型提供服务

用法:
  # 启动服务器
  python3 scripts/serve.py

  # 调用 API（另一个终端）
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qunyou-chat","messages":[{"role":"user","content":"你好"}]}'

简历对应技能:
  - 模型服务化部署 (Model Serving)
  - OpenAI 兼容 API (vLLM / TGI)
  - 推理优化
"""

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import os
import sys
import json
import time
import argparse
import torch
from typing import Optional, List, Dict

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========== 参数 ==========
parser = argparse.ArgumentParser(description="群聊模型推理服务器")
parser.add_argument("--model-path", default="output/qlora-checkpoints/merged_model",
                    help="模型路径")
parser.add_argument("--port", type=int, default=8000, help="服务端口")
parser.add_argument("--host", default="0.0.0.0", help="监听地址")
parser.add_argument("--dtype", default="auto", help="推理精度")
parser.add_argument("--max-tokens", type=int, default=512, help="最大生成长度")
args = parser.parse_args()

# ========== 加载模型 ==========
print(f"加载模型: {args.model_path}")
print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# 合并后的模型保存为 4-bit NF4 格式，需要用相同量化配置加载
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)

model.eval()
print(f"模型就绪! 设备: {model.device}")

# ========== FastAPI 应用 ==========
app = FastAPI(title="QunYou Chat API")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qunyou-chat"
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict


@app.get("/")
async def root():
    return {"service": "QunYou Chat API", "model": "qunyou-chat", "status": "running"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "qunyou-chat", "object": "model", "created": int(time.time()), "owned_by": "qunyou"}]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 构建 ChatML
    messages = []
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response.strip()},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": inputs['input_ids'].shape[1],
            "completion_tokens": outputs.shape[1] - inputs['input_ids'].shape[1],
            "total_tokens": outputs.shape[1],
        },
    }


# ========== Main ==========
if __name__ == "__main__":
    print(f"\n🚀 启动推理服务器: http://{args.host}:{args.port}")
    print(f"   API: POST /v1/chat/completions")
    print(f"   示例: curl http://{args.host}:{args.port}/v1/chat/completions -H ...")
    print(f"   模型: {args.model_path}\n")

    uvicorn.run(app, host=args.host, port=args.port)
