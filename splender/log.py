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
#: 没被接住的错误记进日志之后再交给它 (类型, 值, 回溯)，程序用来弹提示
on_unhandled = None
_crash_file = None


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
        callback = on_unhandled
        if callback is not None and not issubclass(kind, (KeyboardInterrupt, SystemExit)):
            try:
                callback(kind, value, tb)
            except Exception:  # noqa: BLE001
                root.error("显示错误提示时又出错", exc_info=True)

    sys.excepthook = hook


def enable_crash_file(path) -> None:
    """程序崩溃（Python 自己都来不及报错的那种）时把各线程的调用栈写进 path。正常退出时用 disable_crash_file 收尾。"""
    global _crash_file
    import faulthandler

    try:
        handle = open(path, "w", encoding="utf-8")      # noqa: SIM115  进程活着就一直开着
    except OSError:
        logging.getLogger("splender").warning("崩溃报告文件建不了：%s", path)
        return
    disable_crash_file(delete_if_empty=True)
    _crash_file = handle
    faulthandler.enable(file=handle, all_threads=True)


def disable_crash_file(delete_if_empty: bool = True) -> None:
    global _crash_file
    import faulthandler
    import os

    handle, _crash_file = _crash_file, None
    if handle is None:
        return
    try:
        faulthandler.disable()
    except Exception:  # noqa: BLE001
        pass
    path = handle.name
    try:
        handle.close()
        if delete_if_empty and os.path.getsize(path) == 0:
            os.remove(path)
    except OSError:
        pass
