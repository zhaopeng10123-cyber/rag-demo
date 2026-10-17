"""以 MCP client 身份连接 mcp_server.py（langchain-mcp-adapters，stdio 子进程），验证 MCP 工具和进程内 tool 等价。

1. 同一个问题分别调用进程内的 search_knowledge_base 和 MCP 的 search_knowledge，逐条比较来源、章节和正文；
2. 再把 MCP 工具交给 create_agent 回答一题，看模型能否只凭 MCP 工具的说明正确调用它。

用法（在 rag-demo 目录下）：
  python mcp_client_demo.py                  默认问题
  python mcp_client_demo.py "你的问题"
"""
import argparse
import asyncio
import sys

import config
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from tools import AGENT_SYSTEM_PROMPT, CallCounter, search_knowledge_base

SERVER = {"rag": {"transport": "stdio", "command": sys.executable,
                  "args": [str(config.ROOT / "mcp_server.py")], "cwd": str(config.ROOT)}}


def tool_call(name: str, args: dict) -> dict:
    # 用 ToolCall 的形式调用，才能拿到 ToolMessage 里的 artifact（原始结果），而不只是给模型看的文本
    return {"name": name, "args": args, "id": "demo", "type": "tool_call"}


async def compare_search(mcp_search, question: str) -> bool:
    local = search_knowledge_base.invoke(tool_call(search_knowledge_base.name, {"query": question})).artifact
    remote = (await mcp_search.ainvoke(tool_call(mcp_search.name, {"query": question}))).artifact
    chunks = remote["structured_content"]["chunks"]

    print(f"=== 检索对比：{question}")
    same = len(local) == len(chunks)
    for i, (doc, chunk) in enumerate(zip(local, chunks), 1):
        ok = (doc.metadata["source"], doc.metadata["section"], doc.page_content) == \
             (chunk["source"], chunk["section"], chunk["content"])
        same &= ok
        print(f"  [{i}] {'一致' if ok else '不一致'}  {chunk['source']} > {chunk['section']}")
    print(f"进程内 {len(local)} 条，MCP {len(chunks)} 条，{'完全一致' if same else '有差异'}\n")
    return same


async def agent_answer(mcp_tools, question: str) -> None:
    # 主线 Prompt 里写的是进程内 tool 的名字，换成 MCP 工具的名字，其余规则不变
    prompt = AGENT_SYSTEM_PROMPT.replace(search_knowledge_base.name, "search_knowledge")
    agent = create_agent(model=config.get_llm(), tools=mcp_tools, system_prompt=prompt)
    counter = CallCounter()
    state = await agent.ainvoke({"messages": [HumanMessage(question)]}, {"callbacks": [counter]})

    print(f"=== create_agent + MCP 工具：{question}")
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  model -> {tc['name']}({tc['args']})")
    print(f"助手：{state['messages'][-1].text}")
    print(f"（LLM 调用 {counter.calls} 次，{counter.tokens} tokens）")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()
    question = " ".join(args.question) or "RadixAttention 是怎么复用 KV Cache 的？"

    client = MultiServerMCPClient(SERVER)
    # 用同一个 session 调用所有工具；直接 client.get_tools() 的话每次调用都会新起一个 server 进程
    async with client.session("rag") as session:
        mcp_tools = await load_mcp_tools(session)
        print("MCP 工具：", ", ".join(t.name for t in mcp_tools), "\n")
        search = next(t for t in mcp_tools if t.name == "search_knowledge")
        await compare_search(search, question)
        # 只给 Agent 检索工具：answer 工具内部还会再调一次 LLM，Agent 自己作答时用不着
        await agent_answer([search], question)


if __name__ == "__main__":
    asyncio.run(main())
