"""离线测试用的“剧本”模型：按顺序返回预设的回复，支持 bind_tools、with_structured_output 和流式输出。"""
import json
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda

# 假检索器（tests/conftest.py）返回的固定片段
DOCS = [
    Document(page_content="【2.3 什么是 KV Cache】K、V 算一次就缓存。",
             metadata={"source": "02.md", "section": "第 2 章 > 2.3 什么是 KV Cache"}),
    Document(page_content="【2.4 显存】KV Cache 占显存。",
             metadata={"source": "02.md", "section": "第 2 章 > 2.4 KV Cache 带来的新问题：显存"}),
]


def tool_call(name: str, call_id: str = "call_1", **args: Any) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    """responses 里放 AIMessage（普通调用的回复）或 pydantic 对象（with_structured_output 的结果），按顺序消费。"""

    responses: list[Any]
    received: list[list[BaseMessage]] = []
    bound_tools: list[list[str]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next(self) -> Any:
        if not self.responses:
            raise AssertionError("ScriptedChatModel 的剧本已经用完，图里多调用了一次 LLM")
        return self.responses.pop(0)

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received.append(list(messages))
        reply = self._next()
        assert isinstance(reply, AIMessage), f"期望普通调用，剧本里却是 {type(reply).__name__}"
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _stream(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs):
        """普通回复逐字输出；工具调用整块输出。"""
        self.received.append(list(messages))
        reply = self._next()
        assert isinstance(reply, AIMessage), f"期望普通调用，剧本里却是 {type(reply).__name__}"
        if reply.tool_calls:
            chunks = [{"name": tc["name"], "args": json.dumps(tc["args"], ensure_ascii=False), "id": tc["id"],
                       "index": i, "type": "tool_call_chunk"} for i, tc in enumerate(reply.tool_calls)]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for char in reply.content:
            yield ChatGenerationChunk(message=AIMessageChunk(content=char))

    def bind_tools(self, tools, **kwargs):
        self.bound_tools.append([getattr(t, "name", str(t)) for t in tools])
        return self

    def with_structured_output(self, schema, **kwargs):
        def run(_input):
            reply = self._next()
            assert isinstance(reply, schema), f"期望 {schema.__name__}，剧本里却是 {type(reply).__name__}"
            return reply
        return RunnableLambda(run)
