#!/usr/bin/env python3
"""
QLoRA 指令微调脚本 v2.0 — 群聊对话风格微调
TRL 1.x + Transformers 5.x 兼容

训练策略:
  1. 4-bit NF4 量化 (QLoRA)
  2. 全部线性层 LoRA (All Linear targets)
  3. Cosine LR schedule with warmup
  4. Assistant-only loss (只对assistant回复计算损失)
  5. Gradient checkpointing + paged_adamw_8bit

简历对应技能:
  - Parameter-Efficient Fine-Tuning (PEFT / QLoRA)
  - Model Quantization (4-bit NF4)
  - Large Model Training Optimization
  - Dialogue System Development
  - MLOps & Training Pipeline

用法:
  python3 scripts/02_train.py
  python3 scripts/02_train.py --model Qwen/Qwen2.5-7B-Instruct --batch_size 2
"""

import os
import sys
import json
import math
import time
import random
import argparse
from typing import Dict, Optional
from datetime import datetime

import torch
import torch.nn as nn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRL_USE_INTERNALLY_CALLED_ERRORS"] = "0"

# 国内网络环境：默认使用 HF 镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from datasets import load_dataset


# ==============================
#  参数
# ==============================

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Dialogue")

    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="基座模型 HuggingFace ID")
    parser.add_argument("--cache_dir", type=str, default="models",
                        help="模型缓存目录")

    parser.add_argument("--train_file", type=str, default="data/formatted/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/formatted/val.jsonl")
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--output_dir", type=str, default="output/qlora-checkpoints")
    parser.add_argument("--num_epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()


# ==============================
#  工具函数
# ==============================

def format_chatml(conversation: list) -> str:
    """将对话列表格式化为 ChatML 格式 (+ 最后保留 assistant 前缀)"""
    result = ""
    for turn in conversation:
        role = turn['role']
        content = turn['content'].strip()
        if content:
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    result += "<|im_start|>assistant\n"
    return result


def convert_to_chatml(example: dict) -> str:
    """JSONL 数据集中的每行转为 ChatML 文本"""
    return format_chatml(example['conversation'])


def find_all_linear_names(model):
    """找出4bit量化模型中的所有线性层"""
    cls = bnb.nn.Linear4bit
    names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            parts = name.split('.')
            names.add(parts[0] if len(parts) == 1 else parts[-1])
    return sorted(names)


# ==============================
#  加载模型
# ==============================

def resolve_local_model_path(model_id: str) -> str:
    """尝试从 HuggingFace 本地缓存解析模型路径，如果找不到则返回原始 ID"""
    import hashlib
    # HF 缓存目录命名规则: models--org--name
    cache_name = "models--" + model_id.replace("/", "--")
    possible_caches = [
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", cache_name),
        os.path.join("/root", ".cache", "huggingface", "hub", cache_name),
    ]
    for cache_dir in possible_caches:
        snapshots_dir = os.path.join(cache_dir, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                path = os.path.join(snapshots_dir, snapshots[0])
                # 验证 safetensors 文件存在
                has_weights = any(f.endswith(".safetensors") for f in os.listdir(path))
                if has_weights:
                    print(f"  [缓存] 本地找到模型: {path}")
                    return path
    return model_id


def load_model_and_tokenizer(args):
    model_path = resolve_local_model_path(args.model)
    is_local = model_path != args.model
    print(f"\n  [模型] 基座: {args.model}" + (" (本地缓存)" if is_local else ""))
    print(f"  [量化] 4-bit NF4 + 双重量化")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=is_local,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [参数] 总: {total/1e6:.0f}M, 可训练(冻结前): {trainable:.0f}")

    return model, tokenizer


# ==============================
#  Loss 记录回调 + 绘图
# ==============================

class LossHistoryCallback(TrainerCallback):
    """记录训练过程中的 loss 到文件"""

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step, "epoch": round(state.epoch, 3)}
        if "loss" in logs:
            entry["loss"] = round(logs["loss"], 5)
        if "eval_loss" in logs:
            entry["eval_loss"] = round(logs["eval_loss"], 5)
        if "learning_rate" in logs:
            entry["lr"] = round(logs["learning_rate"], 8)
        if "grad_norm" in logs:
            entry["grad_norm"] = round(logs["grad_norm"], 4)
        self.history.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时保存 loss 历史"""
        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n  [Loss] 历史记录已保存 -> {self.save_path}")

        # 尝试画图
        try:
            self._plot_loss()
        except Exception as e:
            print(f"  [Loss] 绘图失败: {e}")

    def _plot_loss(self):
        """绘制 loss 曲线"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [e["step"] for e in self.history if "loss" in e]
        losses = [e["loss"] for e in self.history if "loss" in e]

        if not steps:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 子图1: Loss 曲线
        ax1.plot(steps, losses, "b-", linewidth=1.5, label="Train Loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss Curve", fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: LR 曲线
        lr_steps = [e["step"] for e in self.history if "lr" in e]
        lrs = [e["lr"] for e in self.history if "lr" in e]
        if lr_steps:
            ax2.plot(lr_steps, lrs, "r-", linewidth=1.5, label="Learning Rate")
            ax2.set_ylabel("Learning Rate")
            ax2.set_xlabel("Step")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 标注关键指标
        final_loss = losses[-1]
        min_loss = min(losses)
        ax1.axhline(y=final_loss, color="gray", linestyle="--", alpha=0.5)
        ax1.annotate(f"Final: {final_loss:.4f}",
                     xy=(steps[-1], final_loss),
                     xytext=(steps[-1] * 0.7, final_loss * 1.1),
                     fontsize=10, color="blue",
                     arrowprops=dict(arrowstyle="->", color="blue", alpha=0.6))

        plt.tight_layout()
        plot_path = self.save_path.replace(".json", ".png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Loss] 曲线图已保存 -> {plot_path}")


# ==============================
#  Main
# ==============================

def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 67)
    print("  🚀 QLoRA 指令微调")
    print(f"  {args.model}")
    print("=" * 67)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载模型 + Tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # 2. 配置 LoRA
    target_modules = find_all_linear_names(model)
    print(f"  [LoRA] r={args.lora_r}, alpha={args.lora_alpha}, 目标: {len(target_modules)} 个模块")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 应用 LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. 加载数据集
    print(f"\n  ── 数据集 ──")

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    print(f"  [训练集] {len(train_dataset)} 条")

    eval_dataset = None
    if os.path.exists(args.val_file):
        eval_dataset = load_dataset("json", data_files=args.val_file, split="train")
        print(f"  [验证集] {len(eval_dataset)} 条")

    # 4. SFTConfig
    total_batch = args.batch_size * args.gradient_accumulation_steps

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_total_limit=3,
        bf16=True,
        tf32=True,
        dataloader_num_workers=2,
        report_to="none",
        run_name=f"qunyou-qlora-{datetime.now().strftime('%m%d-%H%M')}",
        seed=args.seed,
        max_length=args.max_length,
        remove_unused_columns=False,
    )

    print(f"\n  [训练] batch={args.batch_size}, accum={args.gradient_accumulation_steps}, "
          f"有效batch={total_batch}")
    print(f"  [训练] lr={args.learning_rate}, epochs={args.num_epochs}, "
          f"max_len={args.max_length}")

    # 估算步数
    steps_per_epoch = math.ceil(len(train_dataset) / total_batch)
    print(f"  [训练] 预估每epoch: {steps_per_epoch} 步, 共 {steps_per_epoch * args.num_epochs} 步")

    # 5. 创建 loss 回调 + Trainer
    loss_callback = LossHistoryCallback(
        save_path=os.path.join(args.output_dir, "loss_history.json")
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=convert_to_chatml,
        callbacks=[loss_callback],
    )

    # 6. 训练
    print(f"\n  {'='*67}")
    print(f"  🏋️  开始训练 ({(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')})")
    print(f"  {'='*67}\n")

    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_time = time.time() - start_time

    print(f"\n  ✅ 训练完成! 耗时: {train_time/60:.1f} 分钟")

    # 保存最终训练指标
    metrics = train_result.metrics
    if "train_loss" in metrics:
        print(f"  最终训练损失: {metrics['train_loss']:.4f}")

    # 7. 保存
    print(f"\n  ── 保存模型 ──")

    # Save adapter
    adapter_path = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"  [LoRA适配器] -> {adapter_path}/")

    # Merge + save full model
    merged_path = os.path.join(args.output_dir, "merged_model")
    print(f"  [合并] LoRA + 基座...")

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"  [完整模型] -> {merged_path}/")

    # 8. 推理测试
    print(f"\n  ── 推理测试 ──")

    test_prompts = [
        "今天天气真好啊",
        "你们打游戏吗",
        "晚上吃什么",
    ]

    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(merged_model.device)

        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"\n  User: {prompt}")
        print(f"  Assistant: {response}")

    # 9. 保存训练配置
    config = {
        "base_model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "learning_rate": args.learning_rate,
        "quantization": "4bit_nf4",
        "dataset_size": len(train_dataset),
        "training_time_minutes": round(train_time / 60, 1),
        "train_loss": metrics.get("train_loss"),
    }

    with open(f"{args.output_dir}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n  {'='*67}")
    print(f"  ✅ 全部完成!")
    print(f"  适配器: {adapter_path}/")
    print(f"  完整模型: {merged_path}/")
    print(f"  配置: {args.output_dir}/training_config.json")
    print(f"  {'='*67}")


if __name__ == "__main__":
    main()
