"""影响范围的规则计算：不让模型判断，按 compare_peers 用到的同一份数据算出来。

分两层：
1. 异常是否真实：模型给的根因类别是监控误报，或者百分比指标超过 100%（物理上不可能），判为无实际影响；
2. 异常分布：在告警主机的同服务主机、同可用区其他服务主机里找出异常主机，再按分布映射成范围。

异常主机的判定：最近 10 分钟均值超过告警阈值，或者相对自身基线的涨幅达到告警主机涨幅的一半。
基线取 1 小时前和 1 天前两个窗口里较低的一个，这样突变（1 小时前正常）和缓慢上涨（1 天前正常）都能识别；
每天固定时段的峰值（如定时备份）1 小时前是正常的，也会被识别为异常，但只要其他主机没有同样的峰值，结果仍是单台主机。
"""
from dataclasses import dataclass
from statistics import mean

from ops_agent.data import PERCENT_METRICS, Case, series
from ops_agent.diag_tools import _window

RISE_RATIO = 0.5


@dataclass
class HostStat:
    name: str
    service: str
    az: str
    current: float
    baseline: float
    abnormal: bool = False


def host_stat(case: Case, name: str, metric: str) -> HostStat:
    h = case.host(name)
    values = series(case.id, name, metric)
    baseline = min(mean(_window(values, 70, 60)), mean(_window(values, 1450, 1440)))
    return HostStat(h.name, h.service, h.az, mean(_window(values, 10, 0)), baseline)


def rule_scope(case: Case, category: str | None) -> tuple[str, str]:
    """返回 (影响范围, 依据)。category 是模型报告里的根因类别，没有报告时传 None。"""
    alert = case.alert
    if category == "false_alarm":
        return "no_impact", "根因类别为监控误报，业务没有实际影响"
    me = host_stat(case, alert.host, alert.metric)
    if alert.metric in PERCENT_METRICS and me.current > 100:
        return "no_impact", f"{alert.metric} 是百分比指标，当前 {me.current:.1f} 超过 100，监控数据本身不可信"

    rise = me.current - me.baseline
    peers = [host_stat(case, h.name, alert.metric) for h in case.hosts
             if h.name != me.name and (h.service == me.service or h.az == me.az)]
    for p in peers:
        p.abnormal = p.current >= alert.threshold or (rise > 0 and p.current - p.baseline >= RISE_RATIO * rise)
    service = [p for p in peers if p.service == me.service]
    az_others = [p for p in peers if p.service != me.service and p.az == me.az]
    bad_service = [p.name for p in service if p.abnormal]
    bad_az = [p.name for p in az_others if p.abnormal]

    if az_others and len(bad_az) * 2 >= len(az_others):
        return "whole_az", f"同可用区 {me.az} 的其他服务主机也异常：{'、'.join(bad_az)}"
    if service and len(bad_service) == len(service):
        return "whole_service", f"同服务 {me.service} 的 {len(service) + 1} 台主机都异常"
    if bad_service:
        return "partial_hosts", (f"同服务 {len(service) + 1} 台主机中 {len(bad_service) + 1} 台异常："
                                 f"{'、'.join([me.name, *bad_service])}")
    return "single_host", f"同服务和同可用区的其他主机都正常，只有 {me.name} 异常"
