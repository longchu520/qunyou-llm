#!/usr/bin/env python3
"""
RAG 构建脚本 — 群消息向量库构建
1. 清洗并展平所有群消息
2. 用 sentence-transformers 嵌入
3. 构建 FAISS 索引 + 消息元数据存储
4. 支持按对话上下文游走 (context walking)

输出:
  data/rag/faiss_index.bin    — FAISS 向量索引
  data/rag/messages_meta.json — 消息元数据（含所属对话ID、位置）
  data/rag/embeddings_config.json — 配置信息

简历关键词:
  - RAG (Retrieval-Augmented Generation)
  - 向量数据库 (Vector Database)
  - 语义检索 (Semantic Search)
  - 嵌入模型 (Embedding Model)
"""

import json
import os
import re
import pickle
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# ========== 配置 ==========
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"  # 33MB，中文嵌入效果优秀
CACHE_DIR = "models"
INPUT_PATH = "liaotian.json"
OUTPUT_DIR = "data/rag"

# 与 data_pipeline 保持一致的消息清洗规则
BOT_UIDS = {
    "u_5FL3cNsFBVvGfqzcwHR4gg", "u_WFYIz8BcxwF9ew--0eIMxw",
    "u_F2pnFnRack26HRMpEGXXXw", "u_4lw_uPnkYgC5kKNkHpx3Qg",
    "u_yfyJVYgWEAAoWMBiX9S4kw", "u_PmxGsJErxkwN0ilC07NLWw",
    "1004706062", "u_Sc5yhos_lS-SIt7KI1x-UA",
}
BOT_NAME_SUBSTR = ["小冰(", "老皮(", "秀妍(", "小几(", "小狐狸(",
                   "元梦", "开心农场", "MaiBot", "Q群管家"]
NOISE_PATTERNS = [
    r'^\[(图片|语音|视频|文件|表情|动画表情|红包|转账|QQ红包|小程序|链接).*?\]$',
    r'^[~.。，、…\s]+$', r'^收到$', r'^好的$', r'^嗯{1,3}$',
    r'^哈{2,}$', r'^来了$', r'^明白$', r'^ok$', r'^OK$',
    r'^\.{2,}$', r'^-{2,}$', r'^_+$', r'^\d{1,2}$',
]


def load_and_clean_messages() -> Tuple[List[Dict], List[List[int]]]:
    """
    加载并清洗所有消息
    返回: (所有有效消息的列表, 对话边界索引)
    对话边界: [[start, end], ...] 指向 messages 列表中的索引范围
    """
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    raw_msgs = raw['messages']
    MAX_CONVERSATION_GAP = 1800  # 30分钟

    # 清洗
    cleaned = []
    for m in raw_msgs:
        if m.get('system', False) or m.get('type') != 'type_1':
            continue
        uid = m.get('sender', {}).get('uid', '')
        name = m.get('sender', {}).get('name', '')
        if uid in BOT_UIDS or any(kw in name for kw in BOT_NAME_SUBSTR):
            continue
        text = m.get('content', {}).get('text', '').strip()
        if not text or len(text) < 3:
            continue
        if any(re.match(p, text) for p in NOISE_PATTERNS):
            continue

        cleaned.append({
            'text': text,
            'sender': name,
            'timestamp': m['timestamp'],
            'time': m['time'],
            'uid': uid,
        })

    # 对话切分（按时间窗口）
    cleaned.sort(key=lambda x: x['timestamp'])
    conversations = []
    boundaries = []
    start = 0
    for i in range(1, len(cleaned)):
        gap = cleaned[i]['timestamp'] - cleaned[i-1]['timestamp']
        if gap > MAX_CONVERSATION_GAP:
            if i - start >= 2:
                conversations.append(cleaned[start:i])
                boundaries.append([start, i])
            start = i
    if len(cleaned) - start >= 2:
        conversations.append(cleaned[start:])
        boundaries.append([start, len(cleaned)])

    print(f"[加载] 原始消息: {len(raw_msgs)}")
    print(f"[清洗] 有效消息: {len(cleaned)} (保留率: {len(cleaned)/len(raw_msgs)*100:.1f}%)")
    print(f"[对话] 构建对话: {len(conversations)} 段")

    return cleaned, boundaries, conversations


def build_embeddings(messages: List[Dict]) -> Tuple[np.ndarray, object]:
    """
    用 sentence-transformers 嵌入所有消息
    BGE 模型规范：需要对 query 添加 "为文本生成向量:" 前缀，
    但对 passage 不需要
    """
    print(f"\n[嵌入] 加载嵌入模型: {EMBED_MODEL}")

    from sentence_transformers import SentenceTransformer

    # BGE 模型需要设置 export HF_ENDPOINT 或者用本地缓存
    model = SentenceTransformer(
        EMBED_MODEL,
        cache_folder=CACHE_DIR,
        device='cuda',
    )

    texts = [m['text'] for m in messages]
    batch_size = 512

    print(f"[嵌入] 开始嵌入 {len(texts)} 条消息, batch_size={batch_size}...")

    # BGE 推荐对 query 加指令前缀，但对 passage 不需要
    # 这里直接嵌入原始文本作为 passage
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2归一化，方便余弦相似度
        device='cuda',
    )
    elapsed = time.time() - start

    print(f"[嵌入] 完成! 耗时: {elapsed:.1f}s ({len(texts)/elapsed:.0f} 条/秒)")
    print(f"[嵌入] 向量维度: {embeddings.shape[1]}")

    return embeddings, model


def build_faiss_index(embeddings: np.ndarray) -> object:
    """构建 FAISS 索引"""
    import faiss

    dim = embeddings.shape[1]
    print(f"\n[FAISS] 构建索引, 维度={dim}, 向量数={embeddings.shape[0]}")

    # 使用 IVF 索引（更适合大规模检索）
    nlist = min(int(np.sqrt(embeddings.shape[0])), 100)  # 聚类中心数
    quantizer = faiss.IndexFlatIP(dim)  # 内积（等价于归一化后的余弦相似度）
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # 训练并添加
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = min(nlist, 10)  # 检索时探索的聚类数

    print(f"[FAISS] 索引构建完成! 类型: IVF, nlist={nlist}, nprobe={index.nprobe}")
    print(f"[FAISS] 索引中向量数: {index.ntotal}")

    return index


def build_message_metadata(
    messages: List[Dict],
    boundaries: List[List[int]],
) -> List[Dict]:
    """
    构建消息元数据，每条消息记录:
      - 文本内容
      - 发送者
      - 所属对话ID (conversation_id)
      - 在对话中的位置 (pos_in_conv)
      - 对话总长度
    """
    # 构建 conv_id 查找表: msg_index → conv_id
    msg_to_conv = {}
    for conv_id, (start, end) in enumerate(boundaries):
        for idx in range(start, end):
            msg_to_conv[idx] = conv_id

    meta = []
    for i, m in enumerate(messages):
        conv_id = msg_to_conv.get(i, -1)
        # 计算在对话中的位置
        if conv_id >= 0:
            start, end = boundaries[conv_id]
            pos = i - start
            conv_len = end - start
        else:
            pos = -1
            conv_len = 1
        meta.append({
            'index': i,
            'text': m['text'],
            'sender': m['sender'],
            'timestamp': m['timestamp'],
            'time': m['time'],
            'conv_id': conv_id,
            'pos_in_conv': pos,
            'conv_len': conv_len,
        })

    print(f"[元数据] 构建 {len(meta)} 条元数据记录")
    return meta


def save_rag_data(index, meta, config: dict):
    """保存 FAISS 索引 + 元数据"""
    import faiss
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # FAISS 索引
    faiss_path = os.path.join(OUTPUT_DIR, "faiss_index.bin")
    faiss.write_index(index, faiss_path)
    print(f"[保存] FAISS 索引 -> {faiss_path} ({os.path.getsize(faiss_path)/1024/1024:.1f}MB)")

    # 消息元数据 (JSON, 带缩进便于调试)
    meta_path = os.path.join(OUTPUT_DIR, "messages_meta.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"[保存] 消息元数据 -> {meta_path} ({os.path.getsize(meta_path)/1024/1024:.1f}MB)")

    # 配置信息
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[保存] 配置 -> {config_path}")

    # 边界索引（用于快速重建）
    boundaries_path = os.path.join(OUTPUT_DIR, "boundaries.json")
    with open(boundaries_path, 'w', encoding='utf-8') as f:
        json.dump(config['boundaries'], f)
    print(f"[保存] 对话边界 -> {boundaries_path}")

    print(f"\n✅ RAG 数据构建完成! 输出目录: {OUTPUT_DIR}/")


def main():
    print("=" * 60)
    print("  RAG 向量库构建")
    print("  群消息 → 嵌入 → FAISS 索引")
    print("=" * 60)

    # 1. 清洗消息
    messages, boundaries, conversations = load_and_clean_messages()

    # 2. 嵌入
    embeddings, model = build_embeddings(messages)

    # 3. 构建 FAISS 索引
    index = build_faiss_index(embeddings)

    # 4. 构建元数据
    meta = build_message_metadata(messages, boundaries)

    # 5. 保存
    config = {
        'embed_model': EMBED_MODEL,
        'total_messages': len(messages),
        'num_conversations': len(conversations),
        'vector_dim': embeddings.shape[1],
        'index_type': 'IVFFlat (Inner Product)',
        'boundaries': boundaries,
        'timestamp': time.time(),
    }
    save_rag_data(index, meta, config)

    # 打印统计
    print(f"\n{'='*60}")
    print(f"  📊 RAG 数据统计")
    print(f"  {'='*60}")
    print(f"  总消息数:      {len(messages):>8,}")
    print(f"  对话数:        {len(conversations):>8,}")
    print(f"  向量维度:      {embeddings.shape[1]:>8}")
    print(f"  嵌入模型:      {EMBED_MODEL}")


if __name__ == "__main__":
    main()
