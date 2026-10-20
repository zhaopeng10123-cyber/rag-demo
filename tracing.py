"""Langfuse 全链路追踪。

.env 里配置了 LANGFUSE_PUBLIC_KEY 和 LANGFUSE_SECRET_KEY 才开启，没配置时这里的函数都是空操作，
离线测试和 CI 不受影响。

每次问答用 traced(cfg, name, ...) 给 RunnableConfig 加上 Langfuse 的回调和会话、用户、标签等属性，
LangChain / LangGraph 里的每个节点、LLM 调用（含 token 用量）、检索、工具调用都会自动挂到这条 trace 下。
trace id 提前生成，同一个问题里分几次调用的链（比如 chain 模式的改写和作答）也能归到同一条 trace，
评测脚本再用它把分数写回去。
"""
import hashlib
import os
from functools import cache
from typing import Any

import config  # noqa: F401  确保先加载 .env


def enabled() -> bool:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false":
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@cache
def client():
    from langfuse import get_client

    return get_client()


def new_trace_id() -> str | None:
    return client().create_trace_id() if enabled() else None


def traced(cfg: dict | None, name: str, *, trace_id: str | None = None, session_id: str | None = None,
           user_id: str | None = None, tags: list[str] | None = None, metadata: dict | None = None) -> dict:
    """返回加上 Langfuse 回调和 trace 属性的新 config；未开启时原样返回。"""
    cfg = dict(cfg or {})
    if not enabled():
        return cfg
    from langfuse.langchain import CallbackHandler

    handler = CallbackHandler(trace_context={"trace_id": trace_id} if trace_id else None)
    attrs: dict[str, Any] = {"langfuse_tags": [t for t in (*(tags or []), os.getenv("LLM_MODEL")) if t]}
    if session_id:
        attrs["langfuse_session_id"] = session_id
    if user_id:
        attrs["langfuse_user_id"] = user_id
    cfg["callbacks"] = [*cfg.get("callbacks", []), handler]
    cfg["metadata"] = {**cfg.get("metadata", {}), **(metadata or {}), **attrs}
    cfg["run_name"] = name
    return cfg


def score(trace_id: str | None, name: str, value: float | str | bool, comment: str | None = None) -> None:
    """把评测分数写回 trace：bool 记为 BOOLEAN，str 记为 CATEGORICAL，其余为 NUMERIC。"""
    if not (enabled() and trace_id):
        return
    if isinstance(value, bool):
        value, data_type = float(value), "BOOLEAN"
    else:
        data_type = "CATEGORICAL" if isinstance(value, str) else "NUMERIC"
    client().create_score(trace_id=trace_id, name=name, value=value, data_type=data_type, comment=comment)


def trace_url(trace_id: str | None) -> str | None:
    return client().get_trace_url(trace_id=trace_id) if enabled() and trace_id else None


def flush() -> None:
    if enabled():
        client().flush()


def anonymize(identity: str) -> str:
    """API Key 之类的身份标识不能原样写进 trace，取哈希前缀作为 user_id。"""
    kind, _, raw = identity.partition(":")
    return f"{kind}:{hashlib.sha256(raw.encode()).hexdigest()[:12]}" if kind == "key" else identity
