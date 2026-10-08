"""垃圾回收的时机。

工程里有几十万个页面对象，Python 自动做一次全量垃圾回收要停一百多毫秒，正在画或转视角时会明显卡一下。
这里关掉全量回收的自动触发，改成用户停手、画面也不动的时候再做。小范围的回收（第 0、1 代）照常自动进行，
每次只要零点几毫秒。
"""
from __future__ import annotations

import gc
import logging
import time

log = logging.getLogger("splender.gc")


class GcPolicy:
    def __init__(self) -> None:
        self.installed = False
        self.last_ms = 0.0
        self.count = 0
        self.last_time = time.perf_counter()

    def install(self) -> None:
        """全量回收不再自动触发（第二代门槛调到极大）。"""
        if self.installed:
            return
        first, second, _third = gc.get_threshold()
        gc.set_threshold(first, second, 1 << 30)
        self.installed = True

    def uninstall(self) -> None:
        if self.installed:
            first, second, _third = gc.get_threshold()
            gc.set_threshold(first, second, 10)
            self.installed = False

    @staticmethod
    def due() -> bool:
        """按 Python 默认节奏，这时候本该做一次全量回收了。"""
        return gc.get_count()[2] >= 10

    def collect_if_due(self, force: bool = False) -> float:
        """需要时做一次全量回收，返回耗时（毫秒）。只应在用户停手时调用。"""
        if not force and not self.due():
            return 0.0
        start = time.perf_counter()
        gc.collect()
        self.last_ms = (time.perf_counter() - start) * 1000.0
        self.last_time = time.perf_counter()
        self.count += 1
        log.debug("空闲时整理内存，用时 %.1f ms", self.last_ms)
        return self.last_ms


POLICY = GcPolicy()
