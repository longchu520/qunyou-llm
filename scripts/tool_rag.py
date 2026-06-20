#!/usr/bin/env python3
"""
Tool-use 模式：API 调用 → RAG 函数链

核心模式:
  1. 定义工具函数 (tools)
  2. 调用外部 API 获取信息
  3. 将结果注入 RAG 上下文
  4. 调用 rag_chat() 生成回复

示例:
  >>> from scripts.tool_rag import get_weather, ask_with_weather
  >>> ask_with_weather("北京")
  # 内部流程: 天气API → 格式化结果 → rag_chat(增强query)

简历关键词:
  - Tool-use / Function Calling 模式
  - LLM Agent 开发
  - API 集成与编排
"""

import os
import sys
import json
import time
import re
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime

# 引入 RAG 核心函数
from scripts.rag_chat import rag_chat, load_rag_data, load_llm, load_embed_model


# ====================================================================
#  工具注册系统 (Tool Registry)
# ====================================================================

class Tool:
    """工具描述符"""
    def __init__(self, name: str, desc: str, fn: Callable, parameters: dict):
        self.name = name
        self.description = desc
        self.fn = fn
        self.parameters = parameters  # JSON Schema

    def __call__(self, **kwargs) -> str:
        return self.fn(**kwargs)

    def to_openai_schema(self) -> dict:
        """转为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


_tool_registry: Dict[str, Tool] = {}


def register_tool(name: str, desc: str, parameters: dict = None):
    """装饰器：注册工具"""
    def decorator(func):
        _tool_registry[name] = Tool(name, desc, func, parameters or {})
        return func
    return decorator


def get_tool(name: str) -> Optional[Tool]:
    return _tool_registry.get(name)


def list_tools() -> Dict[str, Tool]:
    return dict(_tool_registry)


# ====================================================================
#  工具函数实现
# ====================================================================

@register_tool(
    name="get_weather",
    desc="获取指定城市的当前天气情况",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称，如 北京、上海"}
        },
        "required": ["city"]
    }
)
def get_weather(city: str) -> str:
    """
    模拟天气 API 调用
    真实场景可替换为: requests.get(f"https://api.weather.com/v1/{city}")
    """
    # 模拟天气数据（无网络环境可用）
    mock_weather = {
        "北京": "晴, 28°C, 湿度30%",
        "上海": "多云, 26°C, 湿度65%",
        "广州": "阵雨, 30°C, 湿度80%",
        "深圳": "多云, 29°C, 湿度75%",
        "杭州": "阴, 24°C, 湿度70%",
        "成都": "阴, 22°C, 湿度60%",
        "武汉": "晴, 33°C, 湿度45%",
        "南京": "多云, 27°C, 湿度55%",
    }
    result = mock_weather.get(city, f"{city}, 未知天气")
    return f"🌤 {city}天气: {result}"


@register_tool(
    name="get_current_time",
    desc="获取当前日期和时间",
    parameters={}
)
def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M')}"


@register_tool(
    name="search_group_history",
    desc="搜索群聊历史中的相关消息",
    parameters={
        "type": "object",
        "properties": {
            "keyword": {"type": "string", "description": "搜索关键词"}
        },
        "required": ["keyword"]
    }
)
def search_group_history(keyword: str) -> str:
    """
    在群聊历史中搜索关键词
    实际使用 RAG 检索，这里作为独立的 tool 暴露
    """
    # 使用 RAG 的检索能力
    from scripts.rag_chat import embed_query, weighted_retrieve
    load_rag_data()
    load_embed_model()

    query_vec = embed_query(keyword)
    results = weighted_retrieve(query_vec, top_k=50, final_k=3)

    if not results:
        return f"未找到关于「{keyword}」的聊天记录"

    lines = [f"历史相关消息 (共{len(results)}条):"]
    for r in results:
        lines.append(f"  [{r['sender']}]: {r['text']}")
    return "\n".join(lines)


# ====================================================================
#  核心模式：API调用 → 函数链
# ====================================================================

def call_tool(tool_name: str, **params) -> str:
    """
    调用指定工具，返回结果文本

    >>> call_tool("get_weather", city="北京")
    '🌤 北京天气: 晴, 28°C, 湿度30%'
    """
    tool = get_tool(tool_name)
    if not tool:
        return f"错误: 未找到工具 '{tool_name}'"
    try:
        result = tool(**params)
        return result
    except Exception as e:
        return f"工具调用失败: {e}"


def tool_rag_chat(query: str, tool_name: Optional[str] = None,
                  tool_params: Optional[dict] = None,
                  verbose: bool = False) -> Dict:
    """
    ═══════════════════════════════════════════════════════════════
    工具增强 RAG — 先调 API，再调 RAG
    ═══════════════════════════════════════════════════════════════

    流程:
      1. (可选) 调用外部工具/API → 获取结构化信息
      2. 工具结果拼入 query 上下文
      3. 调 rag_chat() 生成回复

    用法:
      # 直接 RAG
      result = tool_rag_chat("晚上吃什么")

      # 天气 + RAG
      result = tool_rag_chat("今天适合出去玩吗",
                             tool_name="get_weather",
                             tool_params={"city": "北京"})

    返回:
      {
        'response':  str,    # 最终回复
        'tool_result': str,  # 工具返回的原始结果
        'augmented_query': str,  # 增强后的 query
        'rag_result': dict,  # rag_chat 的完整返回
      }
    """
    start = time.time()
    tool_result = None

    # 阶段1: 调用工具
    if tool_name:
        print(f"\n[🔧] 调用工具: {tool_name}{tool_params or {}}")
        tool_result = call_tool(tool_name, **(tool_params or {}))
        print(f"    工具返回: {tool_result}")
    else:
        tool_result = None

    # 阶段2: 构建增强后的 query
    if tool_result:
        augmented_query = (
            f"{query}\n\n"
            f"[参考信息]\n{tool_result}"
        )
    else:
        augmented_query = query

    # 阶段3: 调用 RAG
    print(f"\n[🤖] 调用 RAG...")
    rag_result = rag_chat(
        query=augmented_query,
        top_k=1,
        context_window=3,
        verbose=verbose,
    )

    elapsed = time.time() - start

    if verbose:
        print(f"\n[⏱] 工具链总耗时: {elapsed:.2f}s")
        if tool_result:
            print(f"[🔧] 使用的工具: {tool_name}")
            print(f"[📎] 工具结果: {tool_result[:100]}...")

    return {
        'response': rag_result['response'],
        'tool_used': tool_name,
        'tool_result': tool_result,
        'augmented_query': augmented_query,
        'rag_result': rag_result,
        'latency_seconds': round(elapsed, 3),
    }


# ====================================================================
#  交互式 Tool-RAG 聊天
# ====================================================================

def interactive_tool_chat():
    """支持工具调用的交互式聊天"""
    print("\n" + "=" * 65)
    print("  🤖 Tool-RAG 增强群聊机器人")
    print("  " + "=" * 65)
    print("  可用工具:")
    for name, tool in _tool_registry.items():
        print(f"    /{name} - {tool.description}")
    print("  使用: 直接输入消息聊天")
    print("        /tool 工具名 参数 来调用工具")
    print("        /weather 北京 来查天气")
    print("        quit 退出")
    print("  " + "=" * 65)

    while True:
        try:
            raw = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw.lower() == 'quit':
            break

        # 解析工具指令
        tool_name = None
        tool_params = {}
        query = raw

        if raw.startswith('/'):
            parts = raw[1:].split(maxsplit=1)
            if len(parts) >= 1 and parts[0] in _tool_registry:
                tool_name = parts[0]
                tool_params = _parse_tool_args(tool_name, parts[1] if len(parts) > 1 else "")
                query = input(f"  (附带问题) > ").strip() or "随便聊聊"

        # 执行
        result = tool_rag_chat(query, tool_name=tool_name, tool_params=tool_params, verbose=True)
        print(f"\n🤖 群聊AI: {result['response']}")


def _parse_tool_args(tool_name: str, raw_args: str) -> dict:
    """解析工具参数"""
    tool = get_tool(tool_name)
    if not tool:
        return {}
    params = tool.parameters.get("properties", {})
    if not params:
        return {}
    # 简单解析：首个参数
    first_key = list(params.keys())[0]
    return {first_key: raw_args.strip()}


# ====================================================================
#  API 服务器（含工具调用端点）
# ====================================================================

def start_tool_api_server(port: int = 8002):
    """启动含工具调用的 API 服务器"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    # 确保 RAG 组件已加载
    load_rag_data()
    load_llm()
    load_embed_model()

    app = FastAPI(title="Tool-RAG API")

    class ToolCallRequest(BaseModel):
        query: str
        tool_name: Optional[str] = None
        tool_params: Optional[dict] = None
        verbose: bool = False

    class ToolCallResponse(BaseModel):
        response: str
        tool_used: Optional[str]
        tool_result: Optional[str]
        augmented_query: str
        latency_seconds: float

    @app.get("/")
    def root():
        return {
            "service": "Tool-RAG API",
            "tools": {n: t.description for n, t in _tool_registry.items()}
        }

    @app.get("/tools")
    def list_available_tools():
        return {
            "tools": [
                {"name": n, "description": t.description, "parameters": t.parameters}
                for n, t in _tool_registry.items()
            ]
        }

    @app.post("/tool-rag/chat", response_model=ToolCallResponse)
    def tool_rag_endpoint(req: ToolCallRequest):
        """工具增强 RAG 端点"""
        if req.tool_name and req.tool_name not in _tool_registry:
            raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found")
        result = tool_rag_chat(
            query=req.query,
            tool_name=req.tool_name,
            tool_params=req.tool_params,
            verbose=req.verbose,
        )
        return ToolCallResponse(
            response=result['response'],
            tool_used=result['tool_used'],
            tool_result=result['tool_result'],
            augmented_query=result['augmented_query'],
            latency_seconds=result['latency_seconds'],
        )

    print(f"\n🚀 Tool-RAG API 服务器: http://0.0.0.0:{port}")
    print(f"   GET  /tools           — 列出可用工具")
    print(f"   POST /tool-rag/chat   — 工具增强对话")
    print(f"\n  示例:")
    print(f"  curl -X POST http://localhost:{port}/tool-rag/chat \\")
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"query": "今天适合出去玩吗", "tool_name": "get_weather", "tool_params": {{"city": "北京"}}}}\'')
    uvicorn.run(app, host="0.0.0.0", port=port)


# ====================================================================
#  命令行
# ====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tool-RAG 工具增强检索生成")
    parser.add_argument("--query", "-q", type=str, help="查询内容")
    parser.add_argument("--tool", "-t", type=str, help="工具名称")
    parser.add_argument("--tool-params", "-p", type=str, default="{}",
                        help='工具参数 JSON, 如 \'{"city":"北京"}\'')
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--serve", "-s", action="store_true", help="启动 API 服务")
    parser.add_argument("--port", type=int, default=8002, help="API 端口")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    if args.serve:
        start_tool_api_server(port=args.port)
    elif args.interactive:
        interactive_tool_chat()
    elif args.query:
        tool_params = json.loads(args.tool_params) if args.tool else None
        result = tool_rag_chat(
            query=args.query,
            tool_name=args.tool,
            tool_params=tool_params,
            verbose=args.verbose,
        )
        print(f"\n📝 Query: {args.query}")
        if result['tool_used']:
            print(f"🔧 Tool: {result['tool_used']} → {result['tool_result']}")
        print(f"💬 Response: {result['response']}")
        print(f"⏱ Latency: {result['latency_seconds']:.2f}s")
    else:
        parser.print_help()
