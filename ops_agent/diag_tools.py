"""4 个只读诊断工具：指标、变更、日志、同类主机对比。

当前排查的是哪个 case 不由模型决定，而是调用图时通过 context 传入（OpsContext），工具从 ToolRuntime 里取，
模型看到的参数只有主机名、指标名这些。时间一律输出成绝对时间（月-日 时:分）加"多少分钟前"。
"""
from dataclasses import dataclass
from statistics import mean
from typing import Annotated, Literal

from langchain.tools import ToolRuntime, tool

from ops_agent.data import DEFAULT_METRICS, HISTORY_MINUTES, Case, load_cases, series

MAX_LOG_LINES = 15
METRIC_LIST = "、".join(f"{m}（{desc}）" for m, (_, _, desc) in DEFAULT_METRICS.items())


@dataclass
class OpsContext:
    case_id: str


def _case(runtime: ToolRuntime[OpsContext]) -> Case:
    return load_cases()[runtime.context.case_id]


def _when(case: Case, minutes_ago: int) -> str:
    ago = "告警时刻" if minutes_ago == 0 else f"{minutes_ago} 分钟前"
    return f"{case.at(minutes_ago):%m-%d %H:%M}（{ago}）"


def _unknown_host(case: Case, host: str) -> str | None:
    if case.host(host):
        return None
    return f"没有主机 {host}。可查询的主机：{'、'.join(h.name for h in case.hosts)}"


def _window(values: tuple[float, ...], start_ago: int, end_ago: int) -> list[float]:
    """minutes_ago 落在 (end_ago, start_ago] 区间的点，start_ago > end_ago。"""
    return list(values[HISTORY_MINUTES - start_ago:HISTORY_MINUTES - end_ago])


@tool(description=f"""查询一台主机某个指标过去一段时间的变化。
返回窗口开头的基线、最近 5 分钟均值、最大值及时间、前一天同时段均值，以及把窗口切成 12 段的每段均值/最大值。
用来判断指标是突变、缓慢上涨、周期性波动还是一直如此。
可用指标：{METRIC_LIST}""")
def query_metrics(
    host: Annotated[str, "主机名"],
    metric: Annotated[str, "指标名"],
    runtime: ToolRuntime[OpsContext],
    minutes: Annotated[int, "往前看多少分钟，10~4320，默认 60；看日周期或缓慢泄漏时用 1440 以上"] = 60,
) -> str:
    case = _case(runtime)
    if err := _unknown_host(case, host):
        return err
    if metric not in DEFAULT_METRICS:
        return f"没有指标 {metric}。可用指标：{METRIC_LIST}"
    minutes = max(10, min(HISTORY_MINUTES, minutes))
    values = series(case.id, host, metric)
    window = _window(values, minutes, 0)
    peak = max(range(len(window)), key=window.__getitem__)
    lines = [
        f"{host} {metric}（{DEFAULT_METRICS[metric][2]}），最近 {minutes} 分钟：",
        f"窗口开头 1/4 均值 {mean(window[:max(1, minutes // 4)]):.2f}，最近 5 分钟均值 {mean(window[-5:]):.2f}，"
        f"最大值 {window[peak]:.2f} 出现在 {_when(case, minutes - 1 - peak)}",
    ]
    if minutes + 1440 <= HISTORY_MINUTES:
        lines.append(f"前一天同时段均值 {mean(_window(values, minutes + 1440, 1440)):.2f}")
    size = minutes / 12
    lines.append("分段（时间段起点：均值/最大值）：")
    for i in range(12):
        seg = window[round(i * size):round((i + 1) * size)]
        lines.append(f"  {case.at(minutes - 1 - round(i * size)):%m-%d %H:%M}  {mean(seg):.2f}/{max(seg):.2f}")
    return "\n".join(lines)


@tool
def list_changes(
    target: Annotated[str, "服务名、主机名，或日志里出现的数据库等其他对象名"],
    runtime: ToolRuntime[OpsContext],
    hours: Annotated[int, "往前看多少小时，默认 72"] = 72,
) -> str:
    """查询变更记录（发布、配置变更、重启、数据库变更、agent 升级等）。
    按主机名查返回涉及该主机的变更（包括作用于它所属服务的变更）；按服务名查返回该服务的全部变更，包括只灰度到部分主机的。
    结果里带变更涉及的主机列表，可以和异常主机对照，判断变更是否只影响了部分主机。"""
    case = _case(runtime)
    host = case.host(target)

    def related(c) -> bool:
        if host:
            return target in c.hosts or c.target == target or (c.target == host.service and not c.hosts)
        return c.target == target

    hits = [c for c in case.changes if c.minutes_ago <= hours * 60 and related(c)]
    if not hits:
        known = sorted({h.name for h in case.hosts} | {h.service for h in case.hosts})
        hint = "" if host or target in known else f"。已知的主机和服务：{'、'.join(known)}"
        return f"最近 {hours} 小时没有 {target} 的变更记录{hint}"
    lines = [f"{target} 最近 {hours} 小时的变更（新的在前）："]
    for c in sorted(hits, key=lambda c: c.minutes_ago):
        scope = f"涉及主机：{'、'.join(c.hosts)}" if c.hosts else "未指定主机"
        lines.append(f"- {_when(case, c.minutes_ago)} [{c.type}] {c.target}：{c.desc}（{scope}；操作人 {c.operator}）")
    return "\n".join(lines)


@tool
def search_logs(
    host: Annotated[str, "主机名"],
    runtime: ToolRuntime[OpsContext],
    keyword: Annotated[str | None, "只看包含该关键词的日志，不区分大小写"] = None,
    level: Annotated[Literal["ERROR", "WARN", "INFO"] | None, "只看该级别"] = None,
    minutes: Annotated[int, "往前看多少分钟，1~4320，默认 60"] = 60,
) -> str:
    """搜索一台主机的应用和系统日志，同样内容的日志已聚合，带出现次数。
    最多返回 15 条，新的在前；结果太多时用 keyword 或 level 缩小范围，找历史规律时把 minutes 调大。"""
    case = _case(runtime)
    if err := _unknown_host(case, host):
        return err
    minutes = max(1, min(HISTORY_MINUTES, minutes))
    hits = [e for e in case.logs.get(host, [])
            if e.minutes_ago < minutes
            and (level is None or e.level == level)
            and (not keyword or keyword.lower() in e.msg.lower())]
    cond = "，".join(x for x in (f"关键词 {keyword}" if keyword else "", f"级别 {level}" if level else "") if x)
    head = f"{host} 最近 {minutes} 分钟的日志" + (f"（{cond}）" if cond else "")
    if not hits:
        return f"{head}：没有匹配的日志"
    hits.sort(key=lambda e: e.minutes_ago)
    lines = [f"{head}，共 {len(hits)} 条（新的在前）："]
    lines += [f"- {_when(case, e.minutes_ago)} {e.level} ×{e.count} {e.msg}" for e in hits[:MAX_LOG_LINES]]
    if len(hits) > MAX_LOG_LINES:
        lines.append(f"……还有 {len(hits) - MAX_LOG_LINES} 条未显示")
    return "\n".join(lines)


@tool(description=f"""对比同一指标在"同服务的其他主机"和"同可用区的其他服务主机"上的情况。
返回每台主机最近 10 分钟均值、1 小时前和 1 天前的值，以及版本等属性。用来判断问题是单机、部分主机、整个服务还是整个可用区，
以及异常主机和正常主机在属性上有什么差别。
可用指标：{METRIC_LIST}""")
def compare_peers(
    host: Annotated[str, "作为参照的主机名，通常是告警主机"],
    metric: Annotated[str, "指标名"],
    runtime: ToolRuntime[OpsContext],
) -> str:
    case = _case(runtime)
    if err := _unknown_host(case, host):
        return err
    if metric not in DEFAULT_METRICS:
        return f"没有指标 {metric}。可用指标：{METRIC_LIST}"
    me = case.host(host)
    same_service = [h for h in case.hosts if h.service == me.service]
    same_az = [h for h in case.hosts if h.az == me.az and h.service != me.service]

    def row(h) -> str:
        values = series(case.id, h.name, metric)
        attrs = "，".join(f"{k}={v}" for k, v in h.attrs.items())
        mark = "（参照）" if h.name == host else ""
        return (f"- {h.name}{mark} {h.service} {h.az}：最近 10 分钟 {mean(_window(values, 10, 0)):.2f}，"
                f"1 小时前 {mean(_window(values, 70, 60)):.2f}，1 天前 {mean(_window(values, 1450, 1440)):.2f}"
                + (f"；{attrs}" if attrs else ""))

    lines = [f"{metric}（{DEFAULT_METRICS[metric][2]}）对比，参照主机 {host}：", f"同服务 {me.service}："]
    lines += [row(h) for h in same_service]
    lines.append(f"同可用区 {me.az} 的其他服务：")
    lines += [row(h) for h in same_az] or ["- 无"]
    return "\n".join(lines)


TOOLS = [query_metrics, list_changes, search_logs, compare_peers]
