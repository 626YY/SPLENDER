"""色阶控件（照 Photoshop 的色阶）：上面是直方图和输入的黑、灰、白三个三角，下面是输出的渐变条和黑、白两个三角。

拖三角直接改图层的五个属性（输入黑场、输入白场、灰度系数、输出黑场、输出白场），松开记一步撤销。
灰三角的位置是「变成中间灰的那个输入」：黑场 + (白场 - 黑场) × 0.5^灰度系数。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import theme
from .widgets import begin_ui_edit, end_ui_edit

HIST_COLORS = {"RGB": QColor(200, 200, 200), "R": QColor(230, 90, 90), "G": QColor(100, 200, 100),
               "B": QColor(100, 140, 240)}


def draw_histogram(p: QPainter, rect: QRectF, counts, color: QColor) -> None:
    """在 rect 里画直方图（按第 99 百分位定高，个别尖峰不会把其余的压扁）。"""
    if counts is None:
        return
    counts = np.asarray(counts, np.float64)
    if counts.sum() <= 0:
        return
    top = max(float(np.percentile(counts[1:-1], 99)), 1.0)
    heights = np.clip(counts / top, 0.0, 1.0)
    path = QPainterPath(QPointF(rect.left(), rect.bottom()))
    n = len(heights)
    for k, h in enumerate(heights):
        x0 = rect.left() + rect.width() * k / n
        x1 = rect.left() + rect.width() * (k + 1) / n
        y = rect.bottom() - h * rect.height()
        path.lineTo(QPointF(x0, y))
        path.lineTo(QPointF(x1, y))
    path.lineTo(QPointF(rect.right(), rect.bottom()))
    path.closeSubpath()
    fill = QColor(color)
    fill.setAlpha(110)
    p.setPen(Qt.NoPen)
    p.setBrush(fill)
    p.drawPath(path)


class LevelsWidget(QWidget):
    def __init__(self, target: Any, names: list[str], counts=None, channel: str = "RGB", history: Any = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.target = target
        self.names = list(names)          # 输入黑场、输入白场、灰度系数、输出黑场、输出白场
        self.counts = counts
        self.channel = channel
        self.history = history
        self._drag: str | None = None
        self._start: list | None = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(theme.px(150))

    def sizeHint(self) -> QSize:
        return QSize(theme.px(220), theme.px(150))

    # ---- 值 ----
    def values(self) -> list[float]:
        return [float(getattr(self.target, name)) for name in self.names]

    def _set(self, index: int, value: float) -> None:
        setattr(self.target, self.names[index], float(value))

    # ---- 布局 ----
    def _hist_rect(self) -> QRectF:
        pad = float(theme.px(8))
        return QRectF(pad, float(theme.px(4)), max(1.0, self.width() - 2 * pad), float(theme.px(88)))

    def _in_row(self) -> float:
        return self._hist_rect().bottom() + theme.px(1)

    def _out_bar(self) -> QRectF:
        r = self._hist_rect()
        return QRectF(r.left(), r.bottom() + theme.px(20), r.width(), float(theme.px(12)))

    def _out_row(self) -> float:
        return self._out_bar().bottom() + theme.px(1)

    def _x(self, value: float) -> float:
        r = self._hist_rect()
        return r.left() + float(np.clip(value, 0.0, 1.0)) * r.width()

    def _value_at(self, x: float) -> float:
        r = self._hist_rect()
        return float(np.clip((x - r.left()) / max(r.width(), 1.0), 0.0, 1.0))

    def marker_positions(self) -> dict:
        """各个三角的中心（自检用）：in_black、gamma、in_white、out_black、out_white。"""
        in_black, in_white, gamma, out_black, out_white = self.values()
        mid = in_black + (in_white - in_black) * 0.5 ** gamma
        s = theme.px(6)
        y_in, y_out = self._in_row() + s, self._out_row() + s
        return {"in_black": QPointF(self._x(in_black), y_in), "gamma": QPointF(self._x(mid), y_in),
                "in_white": QPointF(self._x(in_white), y_in), "out_black": QPointF(self._x(out_black), y_out),
                "out_white": QPointF(self._x(out_white), y_out)}

    # ---- 画 ----
    def paintEvent(self, event: Any) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self._hist_rect()
        p.fillRect(r, theme.qcolor("bg.widget"))
        draw_histogram(p, r, self.counts, HIST_COLORS.get(self.channel, QColor(200, 200, 200)))
        p.setPen(QPen(theme.qcolor("line"), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRect(r)
        bar = self._out_bar()
        gradient = QLinearGradient(bar.topLeft(), bar.topRight())
        end = {"R": QColor(255, 0, 0), "G": QColor(0, 255, 0), "B": QColor(0, 0, 255)}.get(self.channel,
                                                                                         QColor(255, 255, 255))
        gradient.setColorAt(0.0, QColor(0, 0, 0))
        gradient.setColorAt(1.0, end)
        p.fillRect(bar, gradient)
        p.drawRect(bar)
        fills = {"in_black": QColor(0, 0, 0), "gamma": QColor(128, 128, 128), "in_white": QColor(255, 255, 255),
                 "out_black": QColor(0, 0, 0), "out_white": QColor(255, 255, 255)}
        s = float(theme.px(6))
        for key, c in self.marker_positions().items():
            tri = QPainterPath(QPointF(c.x(), c.y() - s))
            tri.lineTo(QPointF(c.x() + s, c.y() + s * 0.8))
            tri.lineTo(QPointF(c.x() - s, c.y() + s * 0.8))
            tri.closeSubpath()
            p.setPen(QPen(QColor(255, 255, 255) if key == self._drag else theme.qcolor("text.faint"), 1.0))
            p.setBrush(fills[key])
            p.drawPath(tri)
        p.end()

    # ---- 拖动 ----
    def _marker_at(self, pos: QPointF) -> str | None:
        best, best_d = None, float(theme.px(10))
        for key, c in self.marker_positions().items():
            d = math.hypot(c.x() - pos.x(), c.y() - pos.y())
            if d <= best_d:
                best, best_d = key, d
        return best

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        key = self._marker_at(event.position())
        if key is None:
            event.accept()
            return
        self._drag = key
        self._start = self.values()
        begin_ui_edit(self)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self._drag is None:
            self.setCursor(Qt.SizeHorCursor if self._marker_at(event.position()) else Qt.ArrowCursor)
            return
        value = self._value_at(event.position().x())
        in_black, in_white, gamma, out_black, out_white = self.values()
        key = self._drag
        if key == "in_black":
            self._set(0, min(value, in_white - 0.01))
        elif key == "in_white":
            self._set(1, max(value, in_black + 0.01))
        elif key == "gamma":
            t = float(np.clip((value - in_black) / max(in_white - in_black, 1e-4), 0.01, 0.99))
            self._set(2, float(np.clip(math.log(t) / math.log(0.5), 0.1, 10.0)))
        elif key == "out_black":
            self._set(3, value)
        elif key == "out_white":
            self._set(4, value)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if self._drag is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._drag = None
        start, end = self._start, self.values()
        self._start = None
        end_ui_edit(self)
        if self.history is not None and start is not None and start != end:
            target, names = self.target, list(self.names)

            def undo() -> None:
                for name, value in zip(names, start):
                    setattr(target, name, value)

            def redo() -> None:
                for name, value in zip(names, end):
                    setattr(target, name, value)

            self.history.push("色阶", undo, redo)
        self.update()
        event.accept()
