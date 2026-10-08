"""日志：写文件，同时保留一份给「信息」编辑器显示。"""
from __future__ import annotations

import logging
import sys
import time
import traceback
from collections import deque

from .core.signals import Signal
from .paths import log_dir

#: 界面显示用的记录：(时间, 级别, 文字)
records: deque = deque(maxlen=2000)
record_added = Signal()


class _MemoryHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = record.getMessage()
            if record.exc_info:
                text += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
            item = (time.strftime("%H:%M:%S", time.localtime(record.created)), record.levelname, text)
            records.append(item)
            record_added.emit(item)
        except Exception:  # noqa: BLE001
            pass


def setup() -> None:
    root = logging.getLogger("splender")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        path = log_dir() / (time.strftime("%Y-%m-%d") + ".log")
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:  # noqa: BLE001
        pass
    root.addHandler(_MemoryHandler())
    if sys.stderr is not None:
        try:
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            root.addHandler(sh)
        except Exception:  # noqa: BLE001
            pass

    def hook(kind, value, tb):  # 未捕获的异常进日志，不让程序无声消失
        root.error("未处理的错误", exc_info=(kind, value, tb))

    sys.excepthook = hook
