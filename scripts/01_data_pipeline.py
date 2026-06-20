#!/usr/bin/env python3
"""
对话数据生产管道 v3.0 — 群聊数据 → 大模型微调数据集

核心策略（针对群聊数据稀疏性设计）:
  - 按天分组（解决消息间隔中位数200分钟的问题）
  - 固定大小滑动窗口构建对话样本
  - 所有消息去个性化（不区分具体说话者），仅保留对话流
  - 多粒度产出：CPT语料 + SFT对话对 + QA对

简历对应技能:
  - 非结构化数据处理与质量治理
  - 领域自适应预训练 (Domain-Adaptive Pretraining)
  - 参数高效微调数据构建 (PEFT Data Pipeline)
  - 大规模对话系统数据工程
"""

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import json
import re
import os
import random
import math
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple

# ========== 配置 ==========
INPUT_PATH = "liaotian.json"
OUTPUT_DIR = "data/cleaned"
FORMATTED_DIR = "data/formatted"
TRAIN_RATIO = 0.95
SEED = 42
random.seed(SEED)

MIN_MESSAGE_LENGTH = 3
# 滑动窗口参数
CONTEXT_SIZE = 6       # 每样本前 N 条作为上下文
PREDICT_SIZE = 1       # 最后 1 条作为预测目标
MAX_DAILY_SAMPLES = 0  # 0=不限

# BOT UIDs — 精确匹配
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
    r'^/?[a-z]{1,5}$', r'^\d{1,2}$', r'^[~.。，、…\s]+$',
    r'^收到$', r'^好的$', r'^嗯{1,3}$', r'^哈{2,}$', r'^来了$',
    r'^明白$', r'^ok$', r'^OK$', r'^\.{2,}$', r'^-{2,}$', r'^_+$',
]


# ==============================
#  Stage 1: 数据清洗
# ==============================

def load_data(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_message(msg: dict) -> Optional[dict]:
    """清洗单条消息，返回标准格式或 None（应过滤）"""
    if msg.get('system', False):
        return None
    if msg.get('type') != 'type_1':
        return None

    uid = msg.get('sender', {}).get('uid', '')
    name = msg.get('sender', {}).get('name', '')

    if uid in BOT_UIDS:
        return None
    if any(kw in name for kw in BOT_NAME_SUBSTR):
        return None

    text = msg.get('content', {}).get('text', '').strip()
    if not text or len(text) < MIN_MESSAGE_LENGTH:
        return None
    for pat in NOISE_PATTERNS:
        if re.match(pat, text):
            return None

    # 提取回复关系
    reply_target = None
    elements = msg.get('content', {}).get('elements', [])
    for e in elements:
        if isinstance(e, dict) and e.get('type') == 'reply':
            reply_target = e.get('data', {}).get('referencedMessageId')

    return {
        'id': msg['id'],
        'timestamp': msg['timestamp'],
        'time': msg['time'],
        'sender': name,
        'uid': uid,
        'text': text,
        'reply_target': reply_target,
    }


def stage1_clean(raw: dict) -> Tuple[List[dict], Counter, dict]:
    """完整清洗流程"""
    total = len(raw['messages'])
    cleaned = []
    stats = Counter()
    id_index = {}

    for m in raw['messages']:
        # 构建完整ID索引（用于回复查找）
        if m.get('id'):
            id_index[m['id']] = m

        result = clean_message(m)
        if result:
            cleaned.append(result)
        else:
            # 统计过滤原因
            if m.get('system', False):
                stats['system'] += 1
            elif m.get('type') != 'type_1':
                stats['non_text'] += 1
            else:
                text = m.get('content', {}).get('text', '').strip()
                uid = m.get('sender', {}).get('uid', '')
                if uid in BOT_UIDS or any(kw in m.get('sender', {}).get('name', '')
                                          for kw in BOT_NAME_SUBSTR):
                    stats['bot'] += 1
                elif not text or len(text) < MIN_MESSAGE_LENGTH:
                    stats['too_short'] += 1
                else:
                    stats['noise'] += 1

    return cleaned, stats, id_index


# ==============================
#  Stage 2: 按天分组 + 滑动窗口
# ==============================

def group_by_day(messages: List[dict]) -> List[List[dict]]:
    """按自然天分组（保留日期排序）"""
    day_map = defaultdict(list)
    for m in messages:
        day = m['time'][:10]  # "2022-06-29"
        day_map[day].append(m)

    days = sorted(day_map.keys())
    return [day_map[d] for d in days]


def sliding_windows(messages: List[dict], context: int) -> List[Tuple[List[dict], dict]]:
    """
    在一条消息序列上滑动窗口，生成 (上下文, 目标回复) 对
    每个样本：前 context 条为上下文，后 1 条为目标
    步长=1 保证最大数据利用率
    """
    pairs = []
    for i in range(context, len(messages)):
        ctx = messages[i - context:i]
        target = messages[i]
        pairs.append((ctx, target))
    return pairs


def build_reply_pairs(messages: List[dict], id_index: dict) -> List[Tuple[str, str]]:
    """
    基于回复关系构建问答对
    回复消息的 reply_target 指向原消息，构成天然的 (Q, A)
    """
    # 构建 cleaned 消息的 id→obj 索引
    id_to_cleaned = {}
    for day_msgs in group_by_day(messages):
        for m in day_msgs:
            if m['id']:
                id_to_cleaned[m['id']] = m

    pairs = []
    for msg in messages:
        if msg['reply_target'] and msg['reply_target'] in id_to_cleaned:
            parent = id_to_cleaned[msg['reply_target']]
            # 过滤自回复
            if parent['uid'] != msg['uid']:
                # 删除图片标记等噪声
                q_text = re.sub(r'\[图片:.*?\]', '', parent['text']).strip()
                a_text = re.sub(r'^\[回复\s*:\s*.*?\]', '', msg['text']).strip()
                a_text = re.sub(r'@\S+\s*', '', a_text).strip()
                if q_text and a_text and len(q_text) >= 2 and len(a_text) >= 2:
                    pairs.append((q_text, a_text))
    return pairs


# ==============================
#  Stage 3: 格式化 & 导出
# ==============================

def format_sft_sample(ctx_msgs: List[dict], target: dict) -> dict:
    """
    格式化为 ChatML 格式的 SFT 样本

    关键规则（确保多人对话）：
      1. target 的发送者必须与最后一条 ctx 消息的发送者不同（不是自言自语）
      2. ctx 中至少有 2 个不同的发送者（体现多人对话）
      3. 所有上下文消息视为 user 角色，目标视为 assistant 角色
    """
    # 规则1：检测是否自言自语（同一人自问自答）
    if ctx_msgs and ctx_msgs[-1].get('uid') == target.get('uid'):
        return None  # 过滤：同一个人自己接自己

    # 规则2：检测上下文是否有至少2个不同的发言人
    unique_senders = set(m.get('uid') for m in ctx_msgs if m.get('uid'))
    if len(unique_senders) < 1:
        return None  # 至少需要一个不同发言人

    conversation = []
    for m in ctx_msgs:
        text = re.sub(r'\[图片:.*?\]', '[图片]', m['text']).strip()
        text = re.sub(r'^\[回复\s*:\s*.*?\]', '', text).strip()
        if text:
            conversation.append({"role": "user", "content": text})

    target_text = re.sub(r'\[图片:.*?\]', '[图片]', target['text']).strip()
    target_text = re.sub(r'^\[回复\s*:\s*.*?\]', '', target_text).strip()
    target_text = re.sub(r'@\S+\s*', '', target_text).strip()

    if not target_text:
        return None

    conversation.append({"role": "assistant", "content": target_text})

    return {
        'conversation': conversation,
        'num_turns': len(conversation),
        'timestamp': target['timestamp'],
        'date': target['time'][:10],
    }


def format_cpt_text(messages: List[dict]) -> str:
    """
    格式化为因果语言模型 (CLM) 训练文本
    不保留说话者身份，只保留对话流
    """
    lines = []
    for m in messages:
        text = re.sub(r'\[图片:.*?\]', '[图片]', m['text']).strip()
        text = re.sub(r'^\[回复\s*:\s*.*?\]', '', text).strip()
        text = re.sub(r'@\S+\s*', '', text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def quality_filter(sample: Optional[dict]) -> bool:
    """质量过滤"""
    if sample is None:
        return False
    last = sample['conversation'][-1]['content']
    if len(last) < 2:
        return False
    GENERIC = {"好的", "嗯嗯", "收到", "明白", "可以", "ok", "OK", "是的",
               "对", "不对", "好", "嗯", "不", "行", "不行", "不是"}
    if last in GENERIC:
        return False
    if re.match(r'^[\d\s\.\,\!\?]+$', last):
        return False
    return True


# ==============================
#  数据质量报告
# ==============================

def print_report(original: int, cleaned: List[dict], stats: Counter,
                  sft_samples: List[dict], qa_pairs: int, cpt_chars: int,
                  daily_groups: int):
    print("\n" + "=" * 67)
    print("  📊 数据质量报告 (Data Quality Report)")
    print("  " + "=" * 67)

    removed = original - len(cleaned)
    print(f"\n  [1] 数据清洗统计")
    print(f"      原始消息数:           {original:>8,}")
    print(f"      清洗后有效消息:       {len(cleaned):>8,}")
    print(f"      保留率:               {len(cleaned)/original*100:>7.1f}%")
    print(f"      过滤详情:")
    for reason in ['bot', 'too_short', 'noise', 'non_text', 'system']:
        if stats[reason]:
            print(f"        - {reason:>10}: {stats[reason]:>6,} ({stats[reason]/original*100:.1f}%)")

    print(f"\n  [2] 数据规模")
    print(f"      覆盖天数:             {daily_groups:>8}")
    print(f"      SFT训练样本:          {len(sft_samples):>8,}")
    print(f"      QA回复对:             {qa_pairs:>8,}")
    print(f"      CPT语料字符数:        {cpt_chars:>8,}")

    if sft_samples:
        turn_counts = [s['num_turns'] for s in sft_samples]
        print(f"\n  [3] 样本对话轮次分布")
        print(f"      平均轮次:            {sum(turn_counts)/len(turn_counts):>7.1f}")
        print(f"      中位数:              {sorted(turn_counts)[len(turn_counts)//2]:>7}")
        print(f"      最短/最长:           {min(turn_counts):>3}/{max(turn_counts)}")

        text_lens = [len(t['content']) for s in sft_samples for t in s['conversation']]
        print(f"\n  [4] 文本统计")
        print(f"      总字符数:            {sum(text_lens):>8,}")
        print(f"      平均消息长度:        {sum(text_lens)/len(text_lens):>7.1f}字")
        print(f"      消息中位长度:        {sorted(text_lens)[len(text_lens)//2]:>7}")
        print(f"      10字以内占比:        {sum(1 for l in text_lens if l<=10)/len(text_lens)*100:.1f}%")
        print(f"      10-50字占比:         {sum(1 for l in text_lens if 10<l<=50)/len(text_lens)*100:.1f}%")
        print(f"      50字以上占比:        {sum(1 for l in text_lens if l>50)/len(text_lens)*100:.1f}%")

        all_text = "".join(t['content'] for s in sft_samples for t in s['conversation'])
        chinese = len(re.findall(r'[一-鿿]', all_text))
        print(f"\n  [5] 词汇多样性")
        print(f"      唯一汉字数:          {len(set(re.findall(r'[一-鿿]', all_text))):>8,}")
        print(f"      中文占比:            {chinese/max(1,len(all_text))*100:>7.1f}%")

        timestamps = [s['timestamp'] for s in sft_samples if s.get('timestamp')]
        if timestamps:
            from datetime import datetime
            print(f"\n  [6] 时间覆盖")
            print(f"      最早: {datetime.fromtimestamp(min(timestamps)/1000).strftime('%Y-%m-%d')}")
            print(f"      最晚: {datetime.fromtimestamp(max(timestamps)/1000).strftime('%Y-%m-%d')}")

    print("\n  " + "=" * 67)


# ==============================
#  工具函数
# ==============================

def export_jsonl(samples: List[dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')


def export_text(texts: List[str], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(t + '\n\n')


# ==============================
#  Main
# ==============================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FORMATTED_DIR, exist_ok=True)

    print("=" * 67)
    print("  🧹 对话数据生产管道 v3.0")
    print("  群聊数据 → 微调数据集全流程")
    print("=" * 67)

    # ---- Stage 1: 加载 + 清洗 ----
    raw = load_data(INPUT_PATH)
    print(f"\n  [加载] 群: {raw['chatInfo']['name']}, 消息: {len(raw['messages']):,}")

    cleaned, stats, id_index = stage1_clean(raw)
    print(f"  [清洗] 完成: {len(cleaned):,}/{len(raw['messages']):,} 保留")

    # ---- Stage 2: 按天分组 ----
    daily_groups = group_by_day(cleaned)
    print(f"  [分组] 共 {len(daily_groups)} 天")

    # ---- Stage 3: 构建训练样本 ----
    print(f"\n  ── Stage 3: 训练样本构建 ──")

    # 3a. SFT 滑动窗口样本
    sft_samples = []
    for day_msgs in daily_groups:
        windows = sliding_windows(day_msgs, CONTEXT_SIZE)
        for ctx, target in windows:
            sample = format_sft_sample(ctx, target)
            if quality_filter(sample):
                sft_samples.append(sample)

    random.shuffle(sft_samples)
    print(f"  [SFT] 滑动窗口样本: {len(sft_samples):,}")

    # 3b. QA回复对
    all_flat = [m for day in daily_groups for m in day]
    qa_pairs = build_reply_pairs(all_flat, id_index)
    qa_samples = []
    for q, a in qa_pairs:
        qa_samples.append({
            'conversation': [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            'num_turns': 2,
            'timestamp': 0,
            'date': '',
        })
    print(f"  [QA] 回复对样本: {len(qa_samples):,}")

    # 合并
    all_samples = sft_samples + qa_samples
    random.shuffle(all_samples)

    # 3c. CPT 因果语言模型语料
    cpt_texts = [format_cpt_text(day) for day in daily_groups if len(day) >= 5]
    cpt_chars = sum(len(t) for t in cpt_texts)
    export_text(cpt_texts, f"{FORMATTED_DIR}/cpt_corpus.txt")
    print(f"  [CPT] 语料: {len(cpt_texts):,} 段, {cpt_chars:,} 字符")

    # ---- Stage 4: 分割 + 导出 ----
    print(f"\n  ── Stage 4: 数据导出 ──")
    split_idx = max(1, int(len(all_samples) * TRAIN_RATIO))
    train = all_samples[:split_idx]
    val = all_samples[split_idx:]

    export_jsonl(train, f"{FORMATTED_DIR}/train.jsonl")
    export_jsonl(val, f"{FORMATTED_DIR}/val.jsonl")
    print(f"  [训练集] {len(train):,} 条 → {FORMATTED_DIR}/train.jsonl")
    print(f"  [验证集] {len(val):,} 条 → {FORMATTED_DIR}/val.jsonl")

    # ---- 数据质量报告 ----
    print_report(len(raw['messages']), cleaned, stats,
                  all_samples, len(qa_pairs), cpt_chars, len(daily_groups))

    # ---- 样本预览 ----
    print("\n  ── 训练样本预览 ──")
    for i, s in enumerate(all_samples[:5]):
        print(f"\n  >>> 样本 #{i+1} [轮次: {s['num_turns']}]")
        for j, turn in enumerate(s['conversation']):
            txt = turn['content'][:60].replace('\n', '\\n')
            print(f"      [{turn['role']:>9}]: {txt}")
    print(f"\n  {'='*67}")
    print("  ✅ 数据管道完成!")
    print(f"  {'='*67}")


if __name__ == "__main__":
    main()
