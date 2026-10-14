"""离线建库：加载 Markdown -> 按标题切分 -> 按长度二次切分 -> 向量化 -> 写入 Chroma。

用法：python ingest.py [--chunk-size 500] [--chunk-overlap 80] [--collection kb]
可重复执行，每次都会先删除同名 collection 再重建。
"""
import argparse
import random
from pathlib import Path

import config
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


def load_markdown(kb_dir: Path = config.KB_DIR) -> list[Document]:
    # Windows 默认按 GBK 解码，必须显式指定 utf-8
    return [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": str(p)})
        for p in sorted(kb_dir.glob("*.md"))
    ]


def section_of(meta: dict) -> str:
    # h1 是整篇文档的标题，和 source 重复，不放进标题路径
    return " > ".join(meta[h] for h in ("h2", "h3") if meta.get(h))


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=SEPARATORS
    )
    chunks = []
    for doc in docs:
        source = Path(doc.metadata["source"]).name
        for section_doc in header_splitter.split_text(doc.page_content):
            section = section_of(section_doc.metadata)
            for piece in char_splitter.split_text(section_doc.page_content):
                # 把标题路径拼进正文一起向量化，"第3章 RAG" 这类上下文对检索帮助很大
                chunks.append(Document(
                    page_content=f"【{section}】\n{piece}" if section else piece,
                    metadata={**section_doc.metadata, "source": source, "section": section},
                ))
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_index(chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP,
                collection: str = config.COLLECTION, verbose: bool = True):
    docs = load_markdown()
    chunks = split_documents(docs, chunk_size, chunk_overlap)

    store = config.get_vectorstore(collection)
    store.delete_collection()
    store = config.get_vectorstore(collection)
    store.add_documents(chunks, ids=[f"{collection}-{c.metadata['chunk_id']}" for c in chunks])

    if verbose:
        lengths = [len(c.page_content) for c in chunks]
        print(f"文档 {len(docs)} 篇，片段 {len(chunks)} 个，平均 {sum(lengths) / len(lengths):.0f} 字，"
              f"最长 {max(lengths)} 字 -> collection '{collection}'")
    return store, chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--collection", default=config.COLLECTION)
    args = parser.parse_args()

    store, chunks = build_index(args.chunk_size, args.chunk_overlap, args.collection)

    print("\n=== 随机抽查 3 个片段 ===")
    for c in random.Random(0).sample(chunks, 3):
        print(f"[{c.metadata['source']}] {c.metadata['section']}")
        print(c.page_content[:150].replace("\n", " "), "...\n")

    query = "什么是 LCEL"
    print(f"=== 召回测试：{query} ===")
    for doc, score in store.similarity_search_with_score(query, k=3):
        print(f"距离 {score:.3f}  [{doc.metadata['source']}] {doc.metadata['section']}")


if __name__ == "__main__":
    main()
