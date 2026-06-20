#!/usr/bin/env python3
"""
RAG 检索增强生成模块
======================

功能：
  1. 群消息语义检索（FAISS + BGE嵌入）
  2. 上下文游走（Context Walking）：从匹配消息游走到邻近聊天记录
  3. 检索增强生成：检索结果拼入 Prompt → 大模型生成
  4. 封装的调用函数：rag_chat(query) → response

用法：
  # 交互式聊天
  python3 scripts/rag_chat.py --interactive

  # 单次查询
  python3 scripts/rag_chat.py --query "晚上吃什么"

  # API 服务
  python3 scripts/rag_chat.py --serve

简历关键词：
  - RAG (Retrieval-Augmented Generation)
  - 语义检索 (Semantic Search / FAISS)
  - 向量数据库 (Vector Database)
  - 上下文增强 (Context Augmentation)
  - 大模型应用开发 (LLM Application)
"""

import os
import sys
import json
import time
import argparse
import re
from typing import List, Dict, Optional, Tuple
import numpy as np

# ========== 全局配置 ==========
RAG_DIR = "data/rag"
MODEL_PATH = "output/qlora-checkpoints/merged_model"
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CACHE_DIR = "models"
MAX_CONTEXT_MSGS = 6       # 上下文游走：最多取多少条邻近消息
MAX_QUERY_LENGTH = 128     # 检索时 query 截断长度
OUTPUT_DIR = "output/rag"  # RAG 服务输出

# ========== 延迟加载（按需导入） ==========
_faiss_index = None
_messages_meta = None
_boundaries = None
_embed_model = None
_llm_tokenizer = None
_llm_model = None
_config = None


# ====================================================================
#  第一部分：向量检索 (Vector Retrieval)
# ====================================================================

def load_rag_data():
    """加载 FAISS 索引 + 消息元数据（全局单例）"""
    global _faiss_index, _messages_meta, _boundaries, _config
    import faiss

    if _faiss_index is not None:
        return

    print("[RAG] 加载向量库数据...")

    # 加载 FAISS 索引
    index_path = os.path.join(RAG_DIR, "faiss_index.bin")
    _faiss_index = faiss.read_index(index_path)
    print(f"  FAISS 索引: {_faiss_index.ntotal} 个向量")

    # 加载消息元数据
    meta_path = os.path.join(RAG_DIR, "messages_meta.json")
    with open(meta_path, 'r', encoding='utf-8') as f:
        _messages_meta = json.load(f)
    print(f"  消息元数据: {len(_messages_meta)} 条")

    # 加载对话边界
    boundaries_path = os.path.join(RAG_DIR, "boundaries.json")
    with open(boundaries_path, 'r', encoding='utf-8') as f:
        _boundaries = json.load(f)
    print(f"  对话边界: {len(_boundaries)} 段")

    # 加载配置
    config_path = os.path.join(RAG_DIR, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        _config = json.load(f)
    print(f"  ✅ RAG 数据就绪")


def load_embed_model():
    """加载嵌入模型（全局单例）"""
    global _embed_model
    if _embed_model is not None:
        return

    print("[嵌入] 加载模型...")
    from sentence_transformers import SentenceTransformer

    # 强制使用 HuggingFace 镜像（解决国内网络问题）
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # 优先从本地缓存加载（已有缓存时跳过网络请求）
    model_path = EMBED_MODEL_NAME

    # 检查本地缓存是否存在
    local_path = os.path.join(CACHE_DIR, f"models--{EMBED_MODEL_NAME.replace('/', '--')}")
    snapshot_dir = os.path.join(local_path, "snapshots")
    if os.path.isdir(snapshot_dir):
        snapshots = os.listdir(snapshot_dir)
        if snapshots:
            full_path = os.path.join(snapshot_dir, snapshots[0])
            if os.path.isfile(os.path.join(full_path, "model.safetensors")):
                model_path = full_path
                print(f"  [缓存] 使用本地模型: {full_path}")

    _embed_model = SentenceTransformer(
        model_path,
        cache_folder=CACHE_DIR,
        device='cuda',
    )
    print(f"  ✅ 嵌入模型就绪 ({EMBED_MODEL_NAME})")


def embed_query(query: str) -> np.ndarray:
    """将用户 query 转为向量（BGE 需要加指令前缀）"""
    load_embed_model()
    # BGE 推荐：为 query 加指令前缀
    prefixed = f"为这个句子生成向量: {query}"
    vec = _embed_model.encode(
        [prefixed],
        normalize_embeddings=True,
        device='cuda',
    )
    return vec


def retrieve_top_k(query_vec: np.ndarray, k: int = 5) -> List[Dict]:
    """
    检索 top-k 最相似消息
    返回按相似度排序的消息列表
    """
    distances, indices = _faiss_index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(_messages_meta):
            continue
        msg = _messages_meta[idx]
        results.append({
            'index': int(idx),
            'text': msg['text'],
            'sender': msg['sender'],
            'uid': msg.get('uid', msg['sender']),
            'time': msg['time'],
            'timestamp': msg['timestamp'],
            'conv_id': msg['conv_id'],
            'pos_in_conv': msg['pos_in_conv'],
            'conv_len': msg['conv_len'],
            'score': float(dist),
        })

    return results


def weighted_retrieve(
    query_vec: np.ndarray,
    top_k: int = 50,
    final_k: int = 1,
    semantic_weight: float = 0.7,
    time_weight: float = 0.3,
    now_ts: Optional[int] = None,
) -> List[Dict]:
    """
    加权检索：语义相似度 × α + 时间衰减 × (1-α)

    策略：
      1. 先用 FAISS 取 top-K（纯语义）
      2. 对每个候选计算时间衰减分
      3. 加权排序：score = α × sim_norm + (1-α) × recency
      4. 取 top final_k

    参数:
      semantic_weight: 语义权重 (α)，默认为 0.7（必须 >0.5）
      time_weight:     时间权重，默认为 0.3
    """
    # 纯语义检索 top-K
    raw_results = retrieve_top_k(query_vec, k=top_k)
    if not raw_results:
        return []

    # 取当前时间（或最后一条消息的时间）
    if now_ts is None:
        last_ts = max(m.get('timestamp', 0) for m in _messages_meta)
        now_ts = last_ts if last_ts > 0 else int(time.time() * 1000)

    # 相似度归一化到 [0, 1]
    sim_scores = np.array([r['score'] for r in raw_results])
    sim_min, sim_max = sim_scores.min(), sim_scores.max()
    sim_range = sim_max - sim_min if sim_max > sim_min else 1.0

    # 时间衰减分：越新越高，指数衰减
    timestamps = np.array([r.get('timestamp', 0) for r in raw_results])
    hours_ago = (now_ts - timestamps) / (3600 * 1000)
    hours_ago = np.maximum(hours_ago, 0)  # 防止负值
    recency_scores = np.exp(-hours_ago / 48)  # 48小时半衰期

    # 加权融合
    scored = []
    for i, r in enumerate(raw_results):
        sim_norm = (r['score'] - sim_min) / sim_range
        final = semantic_weight * sim_norm + time_weight * recency_scores[i]
        r['similarity_raw'] = r['score']
        r['sim_norm'] = float(sim_norm)
        r['recency_score'] = float(recency_scores[i])
        r['final_score'] = float(final)
        scored.append(r)

    # 按 final_score 排序
    scored.sort(key=lambda x: x['final_score'], reverse=True)

    return scored[:final_k]


# ====================================================================
#  第二部分：上下文游走 (Context Walking)
# ====================================================================

def context_walk(matched_msg: Dict, half_window: int = 3) -> List[Dict]:
    """
    上下文游走：从匹配到的消息出发，沿对话游走到邻近消息

    策略：
      1. 优先从时间窗口对话中取上下文
      2. 如果对话太短（<3条），从全局消息列表中补充邻近消息
      3. 去重 + 按时间排序
      4. **确保至少包含2个不同发言人**（体现多人对话）

    参数:
      matched_msg:  检索命中的消息元数据
      half_window:  向前/后各取多少条

    返回: 按时间排序的上下文消息列表
    """
    msg_index = matched_msg['index']
    context_indices = set()
    context_indices.add(msg_index)

    # 策略1：从时间窗口对话中取
    conv_id = matched_msg['conv_id']
    if conv_id >= 0 and conv_id < len(_boundaries):
        start, end = _boundaries[conv_id]
        pos = matched_msg['pos_in_conv']
        ctx_start = max(start, pos + start - half_window)
        ctx_end = min(end, pos + start + half_window + 1)
        for i in range(ctx_start, ctx_end):
            context_indices.add(_messages_meta[i]['index'])

    # 策略2：如果上下文太少，从全局消息列表中补充邻近消息
    if len(context_indices) < half_window * 2 + 1:
        for offset in range(1, half_window + 1):
            if msg_index - offset >= 0:
                context_indices.add(msg_index - offset)
            if msg_index + offset < len(_messages_meta):
                context_indices.add(msg_index + offset)

    # 策略3：确保至少2个不同发言人
    sorted_indices = sorted(context_indices)
    unique_senders = set()
    for i in sorted_indices:
        unique_senders.add(_messages_meta[i].get('sender', ''))
    # 如果只有一个发言人，尝试扩大窗口
    if len(unique_senders) < 2:
        for offset in range(half_window + 1, half_window + 5):
            if msg_index - offset >= 0:
                context_indices.add(msg_index - offset)
            if msg_index + offset < len(_messages_meta):
                context_indices.add(msg_index + offset)
        # 重新检查
        sorted_indices = sorted(context_indices)

    context = [_messages_meta[i] for i in sorted_indices]
    return context


def format_context(context: List[Dict]) -> str:
    """
    将上下文消息格式化为可读文本
    用于拼入 LLM prompt
    """
    if not context:
        return ""

    lines = []
    for msg in context:
        sender = msg['sender']
        text = msg['text'].strip()
        # 清理 @ 提及和图片标记
        text = re.sub(r'@\S+', '', text).strip()
        text = re.sub(r'\[图片:.*?\]', '[图片]', text).strip()
        if text:
            lines.append(f"{sender}: {text}")

    return "\n".join(lines)


# ====================================================================
#  第三部分：检索增强生成 (RAG Generation)
# ====================================================================

def load_llm():
    """加载微调后的大模型（全局单例）"""
    global _llm_tokenizer, _llm_model
    if _llm_model is not None:
        return

    print("[LLM] 加载微调模型...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    _llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if _llm_tokenizer.pad_token is None:
        _llm_tokenizer.pad_token = _llm_tokenizer.eos_token

    _llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    _llm_model.eval()
    print(f"  ✅ 微调模型就绪")


def build_rag_prompt(query: str, context_text: str) -> str:
    """
    构建 RAG 增强的 Prompt
    将检索到的上下文 + 用户 query 拼入 ChatML 模板

    结构:
      user: [群聊上下文]
      user: {当前问题}
      assistant: (模型生成)
    """
    # 如果上下文非空，作为前置对话历史
    if context_text:
        prompt = (
            f"<|im_start|>user\n"
            f"以下是群聊中相关的聊天记录：\n{context_text}\n<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        # 无检索结果时，走普通对话
        prompt = (
            f"<|im_start|>user\n"
            f"{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    return prompt


def generate_response(prompt: str, max_new_tokens: int = 128,
                      temperature: float = 0.7) -> str:
    """调用微调模型生成回复"""
    import torch

    inputs = _llm_tokenizer(prompt, return_tensors="pt").to(_llm_model.device)

    with torch.no_grad():
        outputs = _llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=_llm_tokenizer.eos_token_id,
        )

    response = _llm_tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    return response.strip()


# ====================================================================
#  第四部分：封装的核心函数 (The One Function)
# ====================================================================

def rag_chat(
    query: str,
    top_k: int = 1,
    context_window: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    verbose: bool = False,
) -> Dict:
    """
    ═══════════════════════════════════════════════════════════════
    RAG 检索增强生成 — 主函数
    ═══════════════════════════════════════════════════════════════

    输入:  query — 用户消息文本
    输出:  {
             'response':  str,   # LLM 生成的回复
             'retrieved': {...},  # 检索到的消息详情
             'context':   str,   # 上下文文本
             'prompt':    str,   # 实际输入的 prompt
             'latency':   float, # 总耗时(秒)
           }

    流程:
      query → 嵌入向量 → FAISS检索 → top-1命中
        → 上下文游走(±N条) → 格式化context
        → 构建RAG prompt → LLM生成 → 回复
    """
    start_time = time.time()
    timeline = {}

    # 1. 确保数据已加载
    load_rag_data()
    load_llm()
    t1 = time.time()
    timeline['load'] = t1 - start_time

    # 2. 嵌入 query
    query_vec = embed_query(query)
    t2 = time.time()
    timeline['embed'] = t2 - t1

    # 3. 加权检索（语义 + 时间衰减）
    results = weighted_retrieve(
        query_vec,
        top_k=50,
        final_k=top_k,
        semantic_weight=0.7,
        time_weight=0.3,
    )
    t3 = time.time()
    timeline['retrieve'] = t3 - t2

    # 4. 上下文游走
    retrieved = {}
    context_text = ""
    if results:
        best = results[0]
        context_msgs = context_walk(best, half_window=context_window)
        context_text = format_context(context_msgs)
        retrieved = {
            'matched_text': best['text'],
            'matched_sender': best['sender'],
            'similarity_score': best.get('score', 0),
            'similarity_norm': best.get('sim_norm', 0),
            'recency_score': best.get('recency_score', 0),
            'final_weighted_score': best.get('final_score', 0),
            'conv_id': best['conv_id'],
            'context_messages': [
                {'sender': m['sender'], 'text': m['text']}
                for m in context_msgs
            ],
        }

    t4 = time.time()
    timeline['walk'] = t4 - t3

    t4 = time.time()
    timeline['walk'] = t4 - t3

    # 5. 构建 RAG prompt
    prompt = build_rag_prompt(query, context_text)
    t5 = time.time()
    timeline['build_prompt'] = t5 - t4

    # 6. 生成回复
    response = generate_response(prompt, max_new_tokens, temperature)
    t6 = time.time()
    timeline['generate'] = t6 - t5

    total_latency = t6 - start_time

    if verbose:
        print(f"\n[⏱] 耗时明细:")
        for step, sec in timeline.items():
            print(f"     {step:>15}: {sec*1000:6.1f}ms")
        print(f"     {'─'*26}")
        print(f"     {'total':>15}: {total_latency*1000:6.1f}ms")
        print(f"\n[🔍] 检索命中: {retrieved.get('matched_text', '无')}")
        print(f"    发送者: {retrieved.get('matched_sender', '?')}")
        print(f"    语义相似度: {retrieved.get('similarity_norm', 0):.4f}")
        print(f"    时间衰减分: {retrieved.get('recency_score', 0):.4f}")
        print(f"    加权总分:   {retrieved.get('final_weighted_score', 0):.4f}  (α=0.7×语义 + 0.3×时间)")
        if retrieved.get('context_messages'):
            sender_set = set(m['sender'] for m in retrieved['context_messages'])
            print(f"    上下文: {len(retrieved['context_messages'])} 条消息, "
                  f"{len(sender_set)} 位发言人")
        print(f"\n[📋] Prompt:\n{prompt}\n")
        print(f"[💬] 回复: {response}")

    return {
        'response': response,
        'retrieved': retrieved,
        'context': context_text,
        'prompt': prompt,
        'latency': {
            'total_seconds': round(total_latency, 3),
            'detail': {k: round(v, 3) for k, v in timeline.items()},
        },
    }


# ====================================================================
#  第五部分：交互式聊天 & API 服务
# ====================================================================

def interactive_chat():
    """交互式聊天模式"""
    print("\n" + "=" * 60)
    print("  🤖 RAG 增强群聊机器人")
    print("  输入 'quit' 退出, 'verbose' 切换详细模式")
    print("=" * 60)

    verbose = False
    history = []

    while True:
        try:
            query = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() == 'quit':
            break
        if query.lower() == 'verbose':
            verbose = not verbose
            print(f"详细模式: {'ON' if verbose else 'OFF'}")
            continue

        history.append({"role": "user", "content": query})
        result = rag_chat(query, verbose=verbose)
        response = result['response']

        print(f"\n🤖 群聊AI: {response}")
        history.append({"role": "assistant", "content": response})


def start_api_server(port: int = 8001):
    """启动 OpenAI 兼容的 RAG API 服务器"""
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="RAG-Enhanced Chat API")

    class ChatRequest(BaseModel):
        query: str
        top_k: int = 1
        context_window: int = 3
        max_tokens: int = 128
        temperature: float = 0.7
        verbose: bool = False

    class ChatResponse(BaseModel):
        response: str
        retrieved: dict
        context: str
        latency: dict

    @app.get("/")
    def root():
        return {"service": "RAG QunYou Chat", "status": "running",
                "vectors": _faiss_index.ntotal if _faiss_index else 0}

    @app.post("/rag/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        result = rag_chat(
            query=req.query,
            top_k=req.top_k,
            context_window=req.context_window,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            verbose=req.verbose,
        )
        return ChatResponse(
            response=result['response'],
            retrieved=result['retrieved'],
            context=result['context'],
            latency=result['latency'],
        )

    print(f"\n🚀 RAG API 服务器启动: http://0.0.0.0:{port}")
    print(f"   POST /rag/chat — RAG 增强对话")
    print(f"   curl -X POST http://localhost:{port}/rag/chat \\")
    print(f'     -H "Content-Type: application/json" \\')
    print(f'     -d \'{{"query": "晚上吃什么"}}\'')
    uvicorn.run(app, host="0.0.0.0", port=port)


# ====================================================================
#  命令行入口
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 检索增强群聊机器人")
    parser.add_argument("--query", "-q", type=str, help="单次查询")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--serve", "-s", action="store_true", help="启动 API 服务")
    parser.add_argument("--port", "-p", type=int, default=8001, help="API 端口")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--top-k", type=int, default=1, help="检索 top-k")
    parser.add_argument("--context-window", type=int, default=3, help="上下文窗口大小")
    args = parser.parse_args()

    if args.serve:
        load_rag_data()
        start_api_server(port=args.port)
    elif args.query:
        result = rag_chat(
            query=args.query,
            top_k=args.top_k,
            context_window=args.context_window,
            verbose=args.verbose,
        )
        print(f"\n📝 Query: {args.query}")
        print(f"💬 Response: {result['response']}")
        if args.verbose:
            print(f"\n🔍 Retrieved: {result['retrieved'].get('matched_text', 'N/A')}")
            print(f"📋 Context:\n{result['context']}")
            print(f"⏱ Latency: {result['latency']['total_seconds']:.2f}s")
    elif args.interactive:
        interactive_chat()
    else:
        parser.print_help()
