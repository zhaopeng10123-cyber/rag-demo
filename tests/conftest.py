import pytest
import tools
from langchain_core.runnables import RunnableLambda

from tests.fake_models import DOCS


@pytest.fixture(autouse=True)
def fake_retriever(monkeypatch):
    """检索词里带“空”时返回空列表，其余返回固定的 DOCS，不加载 Embedding 模型。"""
    queries = []

    def search(query: str):
        queries.append(query)
        return [] if "空" in query else DOCS

    monkeypatch.setattr(tools, "_retriever", RunnableLambda(search))
    return queries
