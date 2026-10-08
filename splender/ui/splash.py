"""启动画面。"""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QSplashScreen

from .. import __version__
from ..paths import resource


def first_launch() -> bool:
    """编译缓存还是空的（第一次启动，或者刚换了版本、换了地方放）：这次启动要多等几秒。"""
    import os
    from pathlib import Path

    folder = os.environ.get("NUMBA_CACHE_DIR")
    if not folder or not Path(folder).is_dir():
        return False
    return not any(Path(folder).rglob("meshprep.*.nbi"))


def show_splash(qt) -> QSplashScreen | None:
    path = resource("logo", "splash.png")
    if not path.is_file():
        return None
    pixmap = QPixmap(str(path))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    font = qt.font()
    font.setPixelSize(12)
    painter.setFont(font)
    painter.setPen(QColor(150, 150, 158))
    text = "第一次启动，正在准备（大约十秒，以后会快很多）…" if first_launch() else "正在启动…"
    painter.drawText(QRect(24, pixmap.height() - 40, pixmap.width() - 48, 28), Qt.AlignVCenter | Qt.AlignLeft, text)
    painter.drawText(QRect(24, pixmap.height() - 40, pixmap.width() - 48, 28), Qt.AlignVCenter | Qt.AlignRight,
                     "版本 " + __version__)
    painter.end()
    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, False)
    splash.show()
    qt.processEvents()
    return splash
