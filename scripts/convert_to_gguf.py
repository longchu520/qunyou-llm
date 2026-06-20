#!/usr/bin/env python3
"""
HF → GGUF 转换器 (Qwen2.5 专用)
使用 gguf 库直接转换，无需外部脚本

用法:
  python3 scripts/convert_to_gguf.py \
    --model-dir output/qlora-checkpoints/merged_model \
    --output output/gguf/qunyou-chat.gguf

简历对应:
  - 模型格式转换 (GGUF Export)
  - 推理优化与部署
  - 跨平台模型兼容 (llama.cpp / Ollama)
"""

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import os
import sys
import json
import struct
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# gguf 库转换工具
import gguf
from gguf import GGUFWriter, MODEL_ARCH, MODEL_TENSOR
from gguf import TensorNameMap, get_tensor_name_map
from gguf.gguf_writer import GGUFValueType
from gguf.vocab import LlamaHfVocab
from gguf.utility import SafetensorsLocal


QWEN2_ARCH = MODEL_ARCH.QWEN2


def get_tensor_names():
    """获取 Qwen2 架构的张量名称映射"""
    name_map = get_tensor_name_map(QWEN2_ARCH)
    return name_map


def convert(args):
    print("=" * 60)
    print("  HF → GGUF 转换器 (Qwen2.5)")
    print("=" * 60)

    model_dir = Path(args.model_dir)
    output_path = Path(args.output)

    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}")
        sys.exit(1)

    os.makedirs(output_path.parent, exist_ok=True)

    # 1. 加载配置
    print(f"\n[1/5] 加载模型配置...")
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # 解析关键参数
    hidden_size = config.get("hidden_size", config.get("dim", 2048))
    num_attention_heads = config.get("num_attention_heads", config.get("n_head", 32))
    num_kv_heads = config.get("num_key_value_heads", config.get("n_kv_head", num_attention_heads))
    num_layers = config.get("num_hidden_layers", config.get("n_layer", 24))
    feed_forward_length = config.get("intermediate_size", config.get("n_inner", 8192))
    max_position_embeddings = config.get("max_position_embeddings", config.get("n_positions", 8192))
    head_dim = config.get("head_dim", hidden_size // num_attention_heads)
    rms_norm_eps = config.get("rms_norm_eps", config.get("layer_norm_eps", 1e-6))
    rope_theta = config.get("rope_theta", config.get("rope_theta", 10000.0))

    print(f"  hidden_size: {hidden_size}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  num_layers: {num_layers}")
    print(f"  feed_forward_length: {feed_forward_length}")
    print(f"  max_position_embeddings: {max_position_embeddings}")

    # 2. 创建 GGUF Writer
    print(f"\n[2/5] 创建 GGUF 文件...")
    gguf_writer = GGUFWriter(str(output_path), "qwen2")

    # 写入架构
    gguf_writer.add_architecture("Qwen2ForCausalLM")

    # 写入超参数
    gguf_writer.add_block_count(num_layers)
    gguf_writer.add_context_length(max_position_embeddings)
    gguf_writer.add_embedding_length(hidden_size)
    gguf_writer.add_feed_forward_length(feed_forward_length)
    gguf_writer.add_head_count(num_attention_heads)
    gguf_writer.add_head_count_kv(num_kv_heads)
    gguf_writer.add_layer_norm_rms_epsilon(rms_norm_eps)
    gguf_writer.add_rope_freq_base(rope_theta)
    gguf_writer.add_rope_dimension_count(head_dim)

    # 文件类型 (all f32 or f16)
    gguf_writer.add_file_type(gguf.GGMLQuantizationType.F16)

    # 3. 转换 Tokenizer
    print(f"\n[3/5] 转换 Tokenizer...")
    try:
        # llama.cpp 的 tokenizer 转换
        vocab = LlamaHfVocab(model_dir)
        vocab_output = gguf_writer.add_tokenizer_model(vocab.tokenizer_model)
        gguf_writer.add_token_list(vocab.token_list)
        gguf_writer.add_token_merges(vocab.merges_list)
        gguf_writer.add_token_types(vocab.token_types)

        # special tokens
        if vocab.bos_token_id is not None:
            gguf_writer.add_bos_token_id(vocab.bos_token_id)
        if vocab.eos_token_id is not None:
            gguf_writer.add_eos_token_id(vocab.eos_token_id)
        if vocab.unk_token_id is not None:
            gguf_writer.add_unknown_token_id(vocab.unk_token_id)
        if vocab.pad_token_id is not None:
            gguf_writer.add_pad_token_id(vocab.pad_token_id)

        print(f"  Tokenizer model: {vocab.tokenizer_model}")
        print(f"  Vocab size: {len(vocab.token_list)}")
        print(f"  BOS: {vocab.bos_token_id}, EOS: {vocab.eos_token_id}")
    except Exception as e:
        print(f"  Tokenizer 转换出错 (使用基本配置): {e}")
        # 基本 tokenizer 配置
        gguf_writer.add_tokenizer_model("gpt2")

    # 4. 转换权重
    print(f"\n[4/5] 转换模型权重...")

    # 读取 safetensors 文件列表
    index_file = model_dir / "model.safetensors.index.json"
    shard_files = []

    if index_file.exists():
        with open(index_file) as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        # 收集唯一的 shard 文件
        shard_set = set(weight_map.values())
        shard_files = [model_dir / s for s in sorted(shard_set)]
        print(f"  使用 index 文件: {len(shard_files)} 个 shard")
    else:
        # 单个文件
        safetensor_files = sorted(model_dir.glob("*.safetensors"))
        if not safetensor_files:
            safetensor_files = sorted(model_dir.glob("model*.safetensors"))
        shard_files = safetensor_files
        print(f"  未找到 index 文件，使用本地文件: {[f.name for f in shard_files]}")

    if not shard_files:
        print(f"  错误: 找不到 safetensors 文件")
        # 尝试 bin 文件
        bin_files = sorted(model_dir.glob("*.bin"))
        if bin_files:
            print("  safetensors 不存在，但找到了 bin 文件，跳过转换")
            print("  模型可能已经被 PyTorch 格式保存")
        sys.exit(1)

    # 加载所有 shard 的权重
    print(f"  加载权重...")
    try:
        tensors = SafetensorsLocal(shard_files)
    except Exception as e:
        print(f"  SafetensorsLocal 出错: {e}")
        print("  尝试直接加载...")
        # fallback: 手动读取
        tensors = {}
        from safetensors import safe_open
        for sf in shard_files:
            with safe_open(sf, framework="np") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
        print(f"  手动加载了 {len(tensors)} 个张量")

    # 定义张量名称映射 (Qwen2 → GGUF)
    tensor_mapping = {
        "model.embed_tokens.weight": "token_embd.weight",
        "model.norm.weight": "output_norm.weight",
        "lm_head.weight": "output.weight",
    }

    # 逐层映射
    layer_prefix = "model.layers."
    for layer_idx in range(num_layers):
        layer_tensors = {
            f"{layer_prefix}{layer_idx}.self_attn.q_proj.weight": f"blk.{layer_idx}.attn_q.weight",
            f"{layer_prefix}{layer_idx}.self_attn.k_proj.weight": f"blk.{layer_idx}.attn_k.weight",
            f"{layer_prefix}{layer_idx}.self_attn.v_proj.weight": f"blk.{layer_idx}.attn_v.weight",
            f"{layer_prefix}{layer_idx}.self_attn.o_proj.weight": f"blk.{layer_idx}.attn_output.weight",
            f"{layer_prefix}{layer_idx}.mlp.gate_proj.weight": f"blk.{layer_idx}.ffn_gate.weight",
            f"{layer_prefix}{layer_idx}.mlp.up_proj.weight": f"blk.{layer_idx}.ffn_up.weight",
            f"{layer_prefix}{layer_idx}.mlp.down_proj.weight": f"blk.{layer_idx}.ffn_down.weight",
            f"{layer_prefix}{layer_idx}.input_layernorm.weight": f"blk.{layer_idx}.attn_norm.weight",
            f"{layer_prefix}{layer_idx}.post_attention_layernorm.weight": f"blk.{layer_idx}.ffn_norm.weight",
        }

        for hf_name, gguf_name in layer_tensors.items():
            _write_tensor(gguf_writer, tensors, hf_name, gguf_name)

    # 写入顶层张量
    for hf_name, gguf_name in tensor_mapping.items():
        _write_tensor(gguf_writer, tensors, hf_name, gguf_name)

    # 5. 完成并关闭
    print(f"\n[5/5] 写入文件并关闭...")
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    file_size = output_path.stat().st_size
    print(f"\n✅ 转换完成!")
    print(f"  输出: {output_path}")
    print(f"  大小: {file_size / 1024 / 1024:.1f} MB")


def _write_tensor(gguf_writer, tensors, hf_name, gguf_name):
    """写入单个张量到 GGUF"""
    try:
        if isinstance(tensors, dict):
            # 手动加载的 dict
            if hf_name in tensors:
                data = tensors[hf_name]
                gguf_writer.add_tensor(gguf_name, data)
                print(f"  ✓ {gguf_name} ({data.shape})")
                return True
        else:
            # SafetensorsLocal 对象
            if hf_name in tensors:
                tensor_info = tensors[hf_name]
                data = tensor_info.data if hasattr(tensor_info, 'data') else tensor_info
                if hasattr(data, 'numpy'):
                    data = data.numpy()
                gguf_writer.add_tensor(gguf_name, data)
                print(f"  ✓ {gguf_name}")
                return True
    except Exception as e:
        print(f"  ✗ {hf_name} → {gguf_name}: {e}")
        return False

    print(f"  - {hf_name}: 未找到")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HF → GGUF Converter for Qwen2.5")
    parser.add_argument("--model-dir", required=True, help="HuggingFace 模型目录")
    parser.add_argument("--output", required=True, help="输出 GGUF 文件路径")
    args = parser.parse_args()
    convert(args)
