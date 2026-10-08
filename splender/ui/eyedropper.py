"""吸管（照 Blender：鼠标停在颜色上按 E）：光标变成十字，在屏幕上任何地方点一下取那里的颜色；右键或 Esc 取消。"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QApplication

_ACTIVE: list = []


def sample_screen(pos) -> tuple | None:
    """屏幕上一点的颜色（sRGB，0..1）。"""
    screen = QGuiApplication.screenAt(pos)
    if screen is None:
        return None
    geo = screen.geometry()
    pixmap = screen.grabWindow(0, pos.x() - geo.x(), pos.y() - geo.y(), 1, 1)
    image = pixmap.toImage()
    if image.isNull() or image.width() < 1:
        return None
    color = image.pixelColor(0, 0)
    return color.redF(), color.greenF(), color.blueF()


class Eyedropper(QObject):
    def __init__(self, write, size: int = 3) -> None:
        super().__init__()
        self.write = write
        self.size = size
        app = QApplication.instance()
        app.installEventFilter(self)
        QApplication.setOverrideCursor(Qt.CrossCursor)
        _ACTIVE.append(self)

    def finish(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        QApplication.restoreOverrideCursor()
        if self in _ACTIVE:
            _ACTIVE.remove(self)

    def eventFilter(self, obj: QObject, event: Any) -> bool:  # noqa: N802
        kind = event.type()
        if kind == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                color = sample_screen(QCursor.pos())
                if color is not None:
                    self.write(color if self.size == 3 else (*color, 1.0))
            self.finish()
            return True
        if kind in (QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
            return True
        if kind == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.finish()
            return True
        return False


def start(binding) -> Eyedropper:
    size = int(getattr(binding.prop, "size", 3) or 3)
    return Eyedropper(binding.write, size)


def active() -> bool:
    return bool(_ACTIVE)
