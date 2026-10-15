"""进程内限流：按客户端的令牌桶，加一个全局并发上限。

状态都在当前进程的内存里。多 worker 或多实例部署时各算各的，需要换成 Redis 等共享存储。
"""
import re
import threading
import time
from dataclasses import dataclass

PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def parse_rate(text: str) -> tuple[int, float]:
    """"10/minute" -> (10, 60.0)"""
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(second|minute|hour|day)\s*", text)
    if not m or int(m[1]) == 0:
        raise ValueError(f"无法解析限流配置 {text!r}，格式如 10/minute")
    return int(m[1]), float(PERIODS[m[2]])


@dataclass
class Decision:
    allowed: bool
    remaining: int
    retry_after: float  # 秒，allowed 时为 0


class TokenBucket:
    """每个 key 一个容量为 capacity 的桶，每 period 秒匀速补满：允许突发 capacity 个请求，长期平均不超过 capacity/period。"""

    def __init__(self, capacity: int, period: float, clock=time.monotonic, max_keys: int = 10_000) -> None:
        self.capacity = capacity
        self.rate = capacity / period
        self.clock = clock
        self.max_keys = max_keys
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (剩余令牌, 上次更新时间)
        self._lock = threading.Lock()

    def acquire(self, key: str) -> Decision:
        with self._lock:
            now = self.clock()
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            allowed = tokens >= 1
            if allowed:
                tokens -= 1
            self._buckets[key] = (tokens, now)
            if len(self._buckets) > self.max_keys:
                self._prune(now)
            return Decision(allowed, int(tokens), 0.0 if allowed else (1 - tokens) / self.rate)

    def _prune(self, now: float) -> None:
        # 闲置到已经补满的桶和新建的桶没有区别，可以直接丢掉
        refill_time = self.capacity / self.rate
        self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < refill_time}


class ConcurrencyLimiter:
    """限制同时进行中的流式回答数。只在事件循环线程里调用，不需要加锁。"""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.active = 0

    def try_acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)
