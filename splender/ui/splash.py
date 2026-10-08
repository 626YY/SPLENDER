"""启动画面。"""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QSplashScreen

from .. import __version__
from ..paths import resource


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
    painter.drawText(QRect(24, pixmap.height() - 40, pixmap.width() - 48, 28), Qt.AlignVCenter | Qt.AlignLeft, "正在启动…")
    painter.drawText(QRect(24, pixmap.height() - 40, pixmap.width() - 48, 28), Qt.AlignVCenter | Qt.AlignRight,
                     "版本 " + __version__)
    painter.end()
    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, False)
    splash.show()
    qt.processEvents()
    return splash
