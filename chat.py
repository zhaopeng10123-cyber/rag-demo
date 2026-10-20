"""命令行多轮对话：结合历史改写问题 -> 检索 -> 流式生成 -> 打印引用。

用法：python chat.py      命令：/clear 清空历史，/exit 退出
"""
import uuid

import config
import tracing
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rag_chain import build_answer_chain, format_docs, get_retriever, print_sources

MAX_TURNS = 5

contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据对话历史，把用户的最新问题改写成一个不依赖上下文、可以单独用于检索的完整问题。"
               "把“它”“这个”等指代替换成具体名称。只输出改写后的问题，不要回答问题。"
               "如果最新问题本身已经完整，原样输出。"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def build_rewrite_chain(llm):
    return contextualize_prompt | llm | StrOutputParser()


def main() -> None:
    llm = config.get_llm()
    retriever = get_retriever()
    rewrite_chain = build_rewrite_chain(llm)
    answer_chain = build_answer_chain(llm)
    history: list[BaseMessage] = []
    session_id = uuid.uuid4().hex

    print("知识库问答已启动。/clear 清空历史，/exit 退出。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/clear":
            history.clear()
            session_id = uuid.uuid4().hex
            print("（历史已清空）")
            continue

        trace_id = tracing.new_trace_id()
        cfg = tracing.traced(None, "chat", trace_id=trace_id, session_id=session_id, tags=["chain"])
        # 没有历史时问题本身就是完整的，省一次 LLM 调用
        standalone = question
        if history:
            standalone = rewrite_chain.invoke({"history": history, "question": question}, cfg).strip()
            if standalone != question:
                print(f"（检索问题改写为：{standalone}）")

        docs = retriever.invoke(standalone, cfg)
        print("助手：", end="", flush=True)
        answer = ""
        for token in answer_chain.stream({"context": format_docs(docs), "question": standalone}, cfg):
            print(token, end="", flush=True)
            answer += token
        print("\n")
        print_sources(docs)
        if url := tracing.trace_url(trace_id):
            print(f"（Langfuse：{url}）")

        history += [HumanMessage(question), AIMessage(answer)]
        history = history[-MAX_TURNS * 2:]

    print("再见。")


if __name__ == "__main__":
    main()
