"""
AstrBot 插件示例 — 将微调模型部署到 QQ 机器人

用法：
  1. 将本项目 clone 到 AstrBot 的 plugins/ 目录下
  2. 在 AstrBot 配置中加载此插件
  3. 在 QQ 群中 @机器人 即可对话

AstrBot 文档: https://github.com/Soulter/AstrBot
"""

from astra.plugin import PluginBase
from astra.event import MessageEvent
from astra.message import Chain

import sys
sys.path.append("plugins/qunyou-llm")

from scripts.rag_chat import rag_chat


class QunYouPlugin(PluginBase):
    priority = 10

    async def on_message(self, event: MessageEvent) -> None:
        if not event.is_at_me:
            return

        query = event.text.strip()
        if not query:
            return

        try:
            # 调用 RAG 增强对话
            result = rag_chat(query)
            reply = result['response']

            # 可选: 附带检索来源
            if result['retrieved'].get('matched_text'):
                source = result['retrieved']['matched_text'][:30]
                reply += f"\n(参考: {source})"

            await event.reply(Chain().text(reply))
        except Exception as e:
            await event.reply(Chain().text(f"啊哦: {e}"))
