"""api.py 的命令行客户端：实时打印处理进度和流式答案，支持多轮对话。

用法：
  python api_client.py                      交互式多轮对话（/clear 开新会话，/exit 退出）
  python api_client.py "什么是 LCEL？"      只问一个问题
选项：--mode graph|agent|chain  --url http://127.0.0.1:8000  --api-key 你的密钥
"""
import argparse
import json
from collections.abc import Iterator

import httpx


def iter_sse(response: httpx.Response) -> Iterator[tuple[str, dict]]:
    event, data = "message", []
    for line in response.iter_lines():
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith(":"):
            continue  # 心跳注释
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))


def print_sources(sources: list[dict]) -> None:
    ids: dict[str, list[int]] = {}
    for s in sources:
        ids.setdefault(f"{s['source']} > {s['section']}".rstrip(" >"), []).append(s["id"])
    print("检索到的片段：")
    for label, nums in ids.items():
        print(f"  {''.join(f'[{n}]' for n in nums)} {label}")


def ask(client: httpx.Client, args, question: str, session_id: str | None) -> str | None:
    payload = {"question": question, "mode": args.mode, "session_id": session_id}
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with client.stream("POST", f"{args.url}/chat/stream", json=payload, headers=headers) as r:
        if r.status_code != 200:
            r.read()
            retry = f"，{r.headers['Retry-After']} 秒后重试" if "Retry-After" in r.headers else ""
            print(f"[HTTP {r.status_code}] {r.json().get('detail')}{retry}")
            return session_id

        answering = False
        for event, data in iter_sse(r):
            if event == "session":
                session_id = data["session_id"]
            elif event == "step" and not answering:
                print(f"  · {data['detail']}")
            elif event == "token":
                text = data["text"]
                if not answering:
                    # 推理模型的答案常以空行开头
                    text = text.lstrip()
                    if not text:
                        continue
                    print("助手：", end="")
                    answering = True
                print(text, end="", flush=True)
            elif event == "sources":
                print()
                print_sources(data["sources"])
            elif event == "done":
                print(f"\n（路由 {data['route']}，LLM 调用 {data['llm_calls']} 次，{data['tokens']} tokens，"
                      f"首字 {data['first_token_latency']}s，总耗时 {data['latency']}s）")
            elif event == "error":
                print(f"\n[错误] {data['message']}")
    return session_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--mode", choices=["graph", "agent", "chain"], default="graph")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    args = parser.parse_args()

    # 服务端每 15 秒发一次心跳，读超时比它长即可；本地服务不走系统代理
    with httpx.Client(timeout=httpx.Timeout(10, read=60), trust_env=False) as client:
        if args.question:
            ask(client, args, " ".join(args.question), None)
            return
        session_id = None
        print(f"已连接 {args.url}（{args.mode} 模式）。/clear 开新会话，/exit 退出。")
        while True:
            try:
                question = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question == "/exit":
                break
            if question == "/clear":
                session_id = None
                print("（已开始新会话）")
            elif question:
                session_id = ask(client, args, question, session_id)


if __name__ == "__main__":
    main()
