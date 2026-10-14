"""单轮 RAG 链：检索 -> 拼接带来源的上下文 -> LLM 生成，输出同时包含 answer 和 docs。

用法：python rag_chain.py "你的问题"     不带参数则跑内置的 4 个示例问题
"""
import sys

import config
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

REFUSAL = "知识库中没有相关信息"

SYSTEM_PROMPT = f"""你是一个知识库问答助手，只能根据下面的【参考资料】回答问题。

规则：
1. 只使用参考资料中的信息，不要用你自己的知识补充；
2. 如果参考资料不足以回答，直接回答“{REFUSAL}”，不要编造；
3. 回答用中文，条理清晰；
4. 在答案末尾另起一行，写“来源：”并列出你实际用到的资料编号，例如“来源：[1][3]”。

【参考资料】
{{context}}"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def get_retriever(k: int = config.TOP_K, collection: str = config.COLLECTION):
    return config.get_vectorstore(collection).as_retriever(search_kwargs={"k": k})


def doc_label(doc: Document) -> str:
    return f"{doc.metadata.get('source', '?')} > {doc.metadata.get('section', '')}".rstrip(" >")


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i}] 来源: {doc_label(d)}\n{d.page_content}" for i, d in enumerate(docs, 1))


def unique_sources(docs: list[Document]) -> list[str]:
    return list(dict.fromkeys(doc_label(d) for d in docs))


def print_sources(docs: list[Document]) -> None:
    """按 format_docs 的编号打印来源，同一章节的多个片段合并成一行，和答案里的 [n] 对得上。"""
    ids: dict[str, list[int]] = {}
    for i, d in enumerate(docs, 1):
        ids.setdefault(doc_label(d), []).append(i)
    print("检索到的片段：")
    for label, nums in ids.items():
        print(f"  {''.join(f'[{n}]' for n in nums)} {label}")


def build_answer_chain(llm=None) -> Runnable:
    """输入 {"context": str, "question": str}，输出答案字符串。chat.py 流式输出时直接用它。"""
    return answer_prompt | (llm or config.get_llm()) | StrOutputParser()


def build_rag_chain(retriever=None, llm=None) -> Runnable:
    """输入问题字符串，输出 {"question", "docs", "answer"}。"""
    retriever = retriever or get_retriever()
    to_inputs = lambda x: {"context": format_docs(x["docs"]), "question": x["question"]}
    return (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        .assign(answer=to_inputs | build_answer_chain(llm))
    )


DEMO_QUESTIONS = [
    "LCEL 是什么？用它串起来的链自动获得哪些能力？",
    "SGLang 的 RadixAttention 解决了什么问题？",
    "LangChain 和 LlamaIndex 应该怎么选？",
    "今天北京的天气怎么样？",
]


def main() -> None:
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUESTIONS
    chain = build_rag_chain()
    for q in questions:
        result = chain.invoke(q)
        print(f"\n问：{q}\n答：{result['answer']}")
        print_sources(result["docs"])


if __name__ == "__main__":
    main()
