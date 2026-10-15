"""FastAPI 服务：把三种问答实现包成 SSE 流式接口，并做限流。

启动：uvicorn api:app --host 127.0.0.1 --port 8000
调用：python api_client.py "什么是 LCEL？"，或者打开 http://127.0.0.1:8000/docs

限流分三层，都在建立 SSE 连接之前检查，所以能返回正常的 HTTP 状态码：
  429  单个客户端（有 API_KEYS 时按密钥，否则按 IP）超过 RATE_LIMIT
  409  同一个会话上一个问题还没回答完（同一线程并发写会把对话历史写乱）
  503  全局同时进行中的回答达到 MAX_CONCURRENT
"""
import json
import logging
import math
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Literal

import config
import openai
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from ratelimit import ConcurrencyLimiter, TokenBucket, parse_rate
from streaming import MODES, Backends

log = logging.getLogger("rag_demo.api")

Mode = Literal["graph", "agent", "chain"]
SESSION_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, pattern=SESSION_PATTERN,
                                   description="多轮对话的会话 ID；不传则新建，从 session 事件里取回")
    mode: Mode = "graph"


def sse(event: str, data: dict) -> ServerSentEvent:
    # FastAPI 默认的 JSON 编码会把中文转义成 \uXXXX，用 raw_data 保持可读
    return ServerSentEvent(event=event, raw_data=json.dumps(data, ensure_ascii=False))


def error_payload(e: Exception) -> dict:
    if isinstance(e, openai.APIStatusError):
        return {"message": f"LLM 服务返回错误（HTTP {e.status_code}）", "status": e.status_code}
    if isinstance(e, openai.APIConnectionError):
        return {"message": "无法连接 LLM 服务", "status": 502}
    return {"message": "服务内部错误", "status": 500}


def create_app(llm=None, rate_limit: str | None = None, max_concurrent: int | None = None,
               api_keys: list[str] | None = None) -> FastAPI:
    capacity, period = parse_rate(rate_limit or config.RATE_LIMIT)
    limiter = TokenBucket(capacity, period)
    slots = ConcurrencyLimiter(max_concurrent or config.MAX_CONCURRENT)
    keys = set(config.API_KEYS if api_keys is None else api_keys)
    backends = Backends(llm)
    busy: set[str] = set()

    app = FastAPI(title="rag-demo API", version="0.1.0",
                  description="个人技术笔记知识库问答：RAG 链 / create_agent / LangGraph 状态图，SSE 流式输出")
    app.state.limiter, app.state.slots, app.state.busy, app.state.backends = limiter, slots, busy, backends

    def client_id(request: Request) -> str:
        if keys:
            key = request.headers.get("x-api-key", "")
            if key not in keys:
                raise HTTPException(401, "缺少或无效的 X-API-Key")
            return f"key:{key}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def rate_limited(response: Response, client: str = Depends(client_id)) -> str:
        decision = limiter.acquire(client)
        headers = {"X-RateLimit-Limit": f"{capacity}/{int(period)}s", "X-RateLimit-Remaining": str(decision.remaining)}
        if not decision.allowed:
            headers["Retry-After"] = str(math.ceil(decision.retry_after))
            raise HTTPException(429, "请求太频繁，请稍后再试", headers=headers)
        response.headers.update(headers)
        return client

    def backend_for(req: ChatRequest):
        try:
            return backends.get(req.mode)
        except RuntimeError as e:  # 例如没有配置 LLM_API_KEY
            raise HTTPException(503, str(e))

    async def stream_slot(req: ChatRequest, _client: str = Depends(rate_limited)) -> AsyncIterator[str]:
        session_id = req.session_id or uuid.uuid4().hex
        key = f"{req.mode}:{session_id}"
        if key in busy:
            raise HTTPException(409, "该会话的上一个问题还在回答中")
        if not slots.try_acquire():
            raise HTTPException(503, "服务繁忙，请稍后再试", headers={"Retry-After": "5"})
        busy.add(key)
        try:
            yield session_id
        finally:
            # yield 依赖在整个流式响应结束（含客户端中途断开）后才退出
            busy.discard(key)
            slots.release()

    @app.post("/chat/stream", response_class=EventSourceResponse, summary="流式问答（SSE）",
              responses={429: {"description": "请求太频繁"}, 409: {"description": "会话忙"},
                         503: {"description": "服务繁忙或未配置 LLM"}})
    async def chat_stream(req: ChatRequest, backend=Depends(backend_for),
                          session_id: str = Depends(stream_slot)) -> AsyncIterator[ServerSentEvent]:
        """事件依次为 session、step*、token*、sources?、done；出错时以 error 事件结束。"""
        yield sse("session", {"session_id": session_id, "mode": req.mode})
        start, first_token = time.perf_counter(), None
        try:
            async for event, data in backend.stream(req.question, session_id):
                if event == "token" and first_token is None:
                    first_token = time.perf_counter() - start
                if event == "done":
                    data["first_token_latency"] = round(first_token or 0, 2)
                yield sse(event, data)
        except Exception as e:
            log.exception("流式回答失败 mode=%s session=%s", req.mode, session_id)
            yield sse("error", error_payload(e))

    @app.delete("/sessions/{session_id}", status_code=204, summary="清空会话历史（所有模式）")
    def clear_session(session_id: str, _client: str = Depends(client_id)) -> None:
        backends.clear(session_id)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": os.getenv("LLM_MODEL", ""), "modes": list(MODES),
                "rate_limit": f"{capacity}/{int(period)}s", "max_concurrent": slots.limit, "active": slots.active}

    return app


app = create_app()
