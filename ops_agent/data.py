"""合成告警 case 的数据模型、加载和指标生成。

每个 case 一个 JSON 文件（cases/）。指标不存原始数据点，只写基线、噪声和异常事件，按固定随机种子逐分钟生成，
同一个 case 每次生成的数据完全相同。所有时间都以告警时刻为基准往前推（minutes_ago）。
"""
import random
import zlib
from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CASES_DIR = Path(__file__).parent / "cases"
HISTORY_MINUTES = 3 * 24 * 60

# case 里没有单独描述的主机和指标都用这里的正常基线：(基线, 噪声标准差, 说明)
DEFAULT_METRICS: dict[str, tuple[float, float, str]] = {
    "cpu_usage": (30, 4, "CPU 使用率，%"),
    "mem_usage": (55, 2, "内存使用率，%"),
    "disk_usage": (48, 0.3, "根分区磁盘使用率，%"),
    "disk_io_util": (12, 4, "磁盘 IO 利用率，%"),
    "net_latency_ms": (0.8, 0.15, "到同可用区网关的网络延迟，ms"),
    "packet_loss": (0.0, 0.02, "丢包率，%"),
    "http_p99_ms": (120, 15, "接口 P99 延迟，ms"),
    "conn_pool_usage": (35, 5, "数据库连接池使用率，%"),
}
PERCENT_METRICS = {m for m, (_, _, desc) in DEFAULT_METRICS.items() if desc.endswith("%")}

# 根因类别和影响范围是固定枚举，评估时直接比较
Category = Literal["release_defect", "config_change", "resource_leak", "downstream_dependency",
                   "infrastructure", "planned_load", "false_alarm"]
Scope = Literal["single_host", "partial_hosts", "whole_service", "whole_az", "no_impact"]
CATEGORY_NAMES: dict[str, str] = {
    "release_defect": "发布引入缺陷", "config_change": "配置变更错误", "resource_leak": "资源泄漏",
    "downstream_dependency": "下游依赖故障", "infrastructure": "底层基础设施故障",
    "planned_load": "计划内负载", "false_alarm": "监控误报",
}
CATEGORY_DEFINITIONS: dict[str, str] = {
    "release_defect": "新版本代码有问题，通常只有升级了的主机异常",
    "config_change": "业务或系统配置改错，比如参数、限额、日志策略",
    "resource_leak": "进程持续占用内存、连接、句柄等资源不释放，常见锯齿或单调上涨",
    "downstream_dependency": "数据库、缓存、第三方接口等下游变慢或故障，本机资源往往并不紧张",
    "infrastructure": "网络、存储、宿主机、可用区等底层设施故障，跨服务影响",
    "planned_load": "备份、批处理、大促等已知的计划内负载，有固定时间规律",
    "false_alarm": "监控数据本身不对（采集方式、口径、agent 问题），业务实际正常",
}
SCOPE_NAMES: dict[str, str] = {
    "single_host": "单台主机", "partial_hosts": "部分主机", "whole_service": "整个服务",
    "whole_az": "整个可用区", "no_impact": "无实际影响",
}


class Event(BaseModel):
    """异常事件，生效期间替换基线：
    step      从 start 起变成 value
    ramp      从 start 起线性涨到 value（告警时刻达到）
    sawtooth  从 start 起每 period 分钟从基线涨到 value 再跌回基线
    daily     每天 at 开始持续 duration 分钟变成 value
    """
    type: Literal["step", "ramp", "sawtooth", "daily"]
    value: float
    start: int = -HISTORY_MINUTES  # 相对告警时刻的分钟数，负数
    period: int = 0
    at: str = ""
    duration: int = 0
    noise: float | None = None  # 事件生效期间的噪声，不设则沿用基线噪声


class MetricSpec(BaseModel):
    base: float | None = None
    noise: float | None = None
    events: list[Event] = []


class Host(BaseModel):
    name: str
    service: str
    az: str
    attrs: dict[str, str] = {}


class Alert(BaseModel):
    host: str
    metric: str
    value: float
    threshold: float
    time: datetime
    title: str


class Change(BaseModel):
    minutes_ago: int
    target: str  # 服务名、主机名或数据库等其他对象
    hosts: list[str] = []
    type: str
    desc: str
    operator: str


class LogEntry(BaseModel):
    minutes_ago: int
    level: Literal["ERROR", "WARN", "INFO"]
    msg: str
    count: int = 1


class Answer(BaseModel):
    category: Category
    scope: Scope
    evidence: list[str] = Field(description="报告里必须出现的关键证据，每项用 | 分隔同义写法，命中任意一个即可")
    required_tools: list[str]
    summary: str


class Case(BaseModel):
    id: str
    alert: Alert
    hosts: list[Host]
    metrics: dict[str, dict[str, MetricSpec]] = {}
    changes: list[Change] = []
    logs: dict[str, list[LogEntry]] = {}
    answer: Answer

    def host(self, name: str) -> Host | None:
        return next((h for h in self.hosts if h.name == name), None)

    def at(self, minutes_ago: int) -> datetime:
        return self.alert.time - timedelta(minutes=minutes_ago)


@cache
def load_cases() -> dict[str, Case]:
    cases = [Case.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(CASES_DIR.glob("*.json"))]
    return {c.id: c for c in cases}


def _level(spec: MetricSpec, base: float, noise: float, minutes_ago: int, clock: datetime) -> tuple[float, float]:
    t = -minutes_ago
    level = base
    for e in spec.events:
        if t < e.start:
            continue
        if e.type == "daily":
            hour, minute = map(int, e.at.split(":"))
            since = (clock.hour * 60 + clock.minute) - (hour * 60 + minute)
            if not 0 <= since % 1440 < e.duration:
                continue
            level = e.value
        elif e.type == "step":
            level = e.value
        elif e.type == "ramp":
            level = base + (e.value - base) * (t - e.start) / -e.start
        elif e.type == "sawtooth":
            level = base + (e.value - base) * ((t - e.start) % e.period) / e.period
        noise = e.noise if e.noise is not None else noise
    return level, noise


@cache
def series(case_id: str, host: str, metric: str) -> tuple[float, ...]:
    """过去 HISTORY_MINUTES 分钟逐分钟的值，下标 i 对应 minutes_ago = HISTORY_MINUTES - 1 - i。"""
    case = load_cases()[case_id]
    base, noise, _ = DEFAULT_METRICS[metric]
    spec = case.metrics.get(host, {}).get(metric, MetricSpec())
    base = spec.base if spec.base is not None else base
    noise = spec.noise if spec.noise is not None else noise
    rng = random.Random(zlib.crc32(f"{case_id}/{host}/{metric}".encode()))
    values = []
    for i in range(HISTORY_MINUTES):
        minutes_ago = HISTORY_MINUTES - 1 - i
        level, sigma = _level(spec, base, noise, minutes_ago, case.at(minutes_ago))
        value = max(0.0, level + rng.gauss(0, sigma))
        if metric in PERCENT_METRICS and level <= 100:
            value = min(value, 100.0)
        values.append(round(value, 2))
    return tuple(values)


def category_choices() -> str:
    return "；".join(f"{k}（{v}：{CATEGORY_DEFINITIONS[k]}）" for k, v in CATEGORY_NAMES.items())


def scope_choices() -> str:
    return "；".join(f"{k}（{v}）" for k, v in SCOPE_NAMES.items())
