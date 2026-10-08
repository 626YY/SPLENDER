"""忙碌光标：几秒钟的同步操作（重构网格、展开 UV）期间把鼠标换成等待样式。"""
from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def busy_cursor():
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        active = QApplication.instance() is not None
    except Exception:  # noqa: BLE001
        active = False
    if active:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        if active:
            QApplication.restoreOverrideCursor()
