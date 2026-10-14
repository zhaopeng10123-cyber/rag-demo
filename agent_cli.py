"""Agent 多轮对话命令行。

用法：python agent_cli.py --mode graph|agent      命令：/clear 开启新会话，/exit 退出
多轮历史由 checkpointer 按 thread_id 保存，指代改写由模型在调用检索 tool 时自己完成。
"""
import argparse
import uuid

import agent
import graph_agent

MODES = {
    "agent": (agent.build_agent, agent.ask, agent.print_result),
    "graph": (graph_agent.build_graph, graph_agent.ask, graph_agent.print_result),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="graph")
    args = parser.parse_args()

    build, ask, print_result = MODES[args.mode]
    app = build()
    thread_id = uuid.uuid4().hex

    print(f"知识库 Agent（{args.mode} 模式）已启动。/clear 开启新会话，/exit 退出。")
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
            thread_id = uuid.uuid4().hex
            print("（已开启新会话）")
            continue
        print_result(ask(app, question, thread_id))

    print("再见。")


if __name__ == "__main__":
    main()
