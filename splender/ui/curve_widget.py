"""曲线控件（照 Photoshop 的曲线）：方框里一条曲线和几个控制点。

- 点曲线外的空白处加一个点（加在曲线上，然后可以接着拖）；按住点拖动；
- 点选中后按 Delete 或右键删掉（两头的点不删，只能拖）；方向键微调选中的点（按住 Shift 走得多）；
- 改动时发 curveChanged(值)；拖动开始、结束发 editingStarted / editingFinished（撤销合成一步）。
- state：面板重画后接着用的状态（当前控制点），见 ui/widget_state.py。
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..core import curve
from . import theme
from .widgets import begin_ui_edit, end_ui_edit

CHANNEL_COLORS = {"RGB": QColor(230, 230, 230), "R": QColor(235, 90, 90), "G": QColor(110, 210, 110),
                  "B": QColor(100, 150, 245)}


class CurveWidget(QWidget):
    curveChanged = Signal(object)
    editingStarted = Signal()
    editingFinished = Signal()

    def __init__(self, value: Any = None, channel: str = "RGB", parent: QWidget | None = None,
                 state: dict | None = None) -> None:
        super().__init__(parent)
        self._value = curve.normalize(value if value is not None else curve.DEFAULT)
        self.channel = channel
        self._state = state if state is not None else {}
        active = int(self._state.get("active", -1))
        self.active = active if 0 <= active < len(self._value) else -1
        self._drag = False
        self.counts = None                 # 背后画的直方图（256 档），没有就不画
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(theme.px(200))

    def sizeHint(self) -> QSize:
        return QSize(theme.px(200), theme.px(200))

    def minimumSizeHint(self) -> QSize:
        return QSize(theme.px(120), theme.px(200))

    # ---- 值 ----
    def value(self) -> tuple:
        return self._value

    def setValue(self, value: Any, emit: bool = False) -> None:
        value = curve.normalize(value)
        if value == self._value:
            return
        self._value = value
        self._set_active(min(self.active, len(value) - 1))
        self.update()
        if emit:
            self.curveChanged.emit(value)

    def _commit(self, value: tuple) -> None:
        value = curve.normalize(value)
        if value != self._value:
            self._value = value
            self.curveChanged.emit(value)
        self.update()

    def _set_active(self, index: int) -> None:
        self.active = index
        self._state["active"] = index

    # ---- 坐标 ----
    def _plot(self) -> QRectF:
        """曲线区占满控件（横向越宽，拖点越准）。"""
        pad = float(theme.px(6))
        return QRectF(pad, pad, max(1.0, self.width() - 2 * pad), max(1.0, self.height() - 2 * pad))

    def _to_screen(self, x: float, y: float) -> QPointF:
        r = self._plot()
        return QPointF(r.left() + x * r.width(), r.bottom() - y * r.height())

    def _to_value(self, pos: QPointF) -> tuple[float, float]:
        r = self._plot()
        x = (pos.x() - r.left()) / max(r.width(), 1.0)
        y = (r.bottom() - pos.y()) / max(r.height(), 1.0)
        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))

    def point_position(self, index: int) -> QPointF:
        """第 index 个控制点在控件里的位置（自检用）。"""
        x, y = self._value[index]
        return self._to_screen(x, y)

    def _point_at(self, pos: QPointF) -> int:
        best, best_d = -1, float(theme.px(8))
        for k, (x, y) in enumerate(self._value):
            p = self._to_screen(x, y)
            d = ((p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = k, d
        return best

    # ---- 画 ----
    def paintEvent(self, event: Any) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self._plot()
        p.fillRect(r, theme.qcolor("bg.widget"))
        if self.counts is not None:
            from .levels_widget import HIST_COLORS, draw_histogram

            draw_histogram(p, r, self.counts, HIST_COLORS.get(self.channel, QColor(200, 200, 200)))
        p.setPen(QPen(theme.qcolor("line"), 1.0))
        for k in range(1, 4):
            x = r.left() + r.width() * k / 4.0
            y = r.top() + r.height() * k / 4.0
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QPen(theme.qcolor("text.faint"), 1.0, Qt.DashLine))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.top()))
        xs = np.linspace(0.0, 1.0, 128)
        ys = curve.evaluate(self._value, xs)
        path = QPainterPath(self._to_screen(xs[0], ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(self._to_screen(float(x), float(y)))
        color = CHANNEL_COLORS.get(self.channel, QColor(230, 230, 230))
        p.setPen(QPen(color, 1.8))
        p.drawPath(path)
        s = float(theme.px(4))
        for k, (x, y) in enumerate(self._value):
            c = self._to_screen(x, y)
            active = k == self.active
            p.setPen(QPen(QColor(255, 255, 255) if active else QColor(20, 20, 20), 1.2))
            p.setBrush(QColor(255, 255, 255) if active else color)
            p.drawRect(QRectF(c.x() - s, c.y() - s, 2 * s, 2 * s))
        p.setPen(QPen(theme.qcolor("line"), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawRect(r)
        if 0 <= self.active < len(self._value):
            x, y = self._value[self.active]
            p.setPen(theme.qcolor("text.dim"))
            pad = float(theme.px(4))
            p.drawText(r.adjusted(pad, pad, -pad, -pad), Qt.AlignLeft | Qt.AlignTop,
                       "输入 %d   输出 %d" % (round(x * 255), round(y * 255)))
        p.end()

    # ---- 鼠标、键盘 ----
    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        pos = event.position()
        k = self._point_at(pos)
        if event.button() == Qt.RightButton:
            if 0 < k < len(self._value) - 1:
                self.editingStarted.emit()
                self._commit(self._value[:k] + self._value[k + 1:])
                self._set_active(-1)
                self.editingFinished.emit()
            event.accept()
            return
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        begin_ui_edit(self)
        self.editingStarted.emit()
        if k < 0 and len(self._value) < curve.MAX_POINTS:
            x, _y = self._to_value(pos)
            y = float(curve.evaluate(self._value, x))
            points = sorted(list(self._value) + [(x, y)], key=lambda q: q[0])
            self._commit(tuple(points))
            k = min(range(len(self._value)), key=lambda i: abs(self._value[i][0] - x))
        self._set_active(k)
        self._drag = True
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if not self._drag or self.active < 0:
            self.setCursor(Qt.PointingHandCursor if self._point_at(event.position()) >= 0 else Qt.CrossCursor)
            return
        x, y = self._to_value(event.position())
        self._move_active(x, y)
        event.accept()

    def _move_active(self, x: float, y: float) -> None:
        points = list(self._value)
        k = self.active
        lo = points[k - 1][0] + 1e-3 if k > 0 else 0.0
        hi = points[k + 1][0] - 1e-3 if k < len(points) - 1 else 1.0
        points[k] = (float(np.clip(x, lo, hi)), float(np.clip(y, 0.0, 1.0)))
        self._commit(tuple(points))

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if self._drag and event.button() == Qt.LeftButton:
            self._drag = False
            self.editingFinished.emit()
            end_ui_edit(self)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        k = self.active
        key = event.key()
        if key in (Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_X) and 0 < k < len(self._value) - 1:
            self.editingStarted.emit()
            self._commit(self._value[:k] + self._value[k + 1:])
            self._set_active(-1)
            self.editingFinished.emit()
            event.accept()
            return
        steps = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0), Qt.Key_Up: (0, 1), Qt.Key_Down: (0, -1)}
        if key in steps and 0 <= k < len(self._value):
            dx, dy = steps[key]
            amount = (10.0 if event.modifiers() & Qt.ShiftModifier else 1.0) / 255.0
            x, y = self._value[k]
            self.editingStarted.emit()
            self._move_active(x + dx * amount, y + dy * amount)
            self.editingFinished.emit()
            event.accept()
            return
        super().keyPressEvent(event)
