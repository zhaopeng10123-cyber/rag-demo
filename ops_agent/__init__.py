"""云主机告警根因定位 Agent：合成告警 case + 4 个诊断工具 + 手写 LangGraph ReAct 循环。

用法（在 rag-demo 目录下）：
  python -m ops_agent --list             列出所有 case
  python -m ops_agent --case case01      排查一个 case，打印工具调用轨迹和根因报告
  python -m ops_agent.evaluate           跑全部 case，统计根因类别、影响范围、证据和工具覆盖
"""
