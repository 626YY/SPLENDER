"""色带控件（照 Blender 的颜色渐变）：上面一排 加色标 / 减色标 / 插值方式 / 翻转，中间是色带和下方的色标三角，
下面一排是当前色标的序号、位置和颜色。

- 点色标选中，按住拖动改位置（越过别的色标时顺序跟着变）；点色带空白处选中最近的色标并开始拖；
- Ctrl+点击色带（或双击）在那里加一个色标，颜色取那里现在的颜色；Delete 删掉当前色标；
- 改动时发 rampChanged(值)；拖动、改数值、取色的开始和结束发 editingStarted / editingFinished（撤销合成一步）。
"""
from __future__ import annotations

from typing import Any

import shiboken6
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from ..core import ramp
from . import theme
from .widgets import (ColorButton, EnumField, IconButton, NumberField, begin_ui_edit, draw_checker, end_ui_edit,
                      rounded_path)


class _RampBar(QWidget):
    """色带本身和色标三角。"""

    def __init__(self, owner: "ColorRampWidget") -> None:
        super().__init__(owner)
        self.owner = owner
        self._drag = False
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(theme.px(200), theme.px(34))

    def minimumSizeHint(self) -> QSize:
        return QSize(theme.px(80), theme.px(34))

    def _bar_rect(self) -> QRectF:
        pad = theme.px(6)
        return QRectF(pad, theme.px(2), max(1.0, self.width() - 2 * pad), theme.px(20))

    def _x_of(self, pos: float) -> float:
        rect = self._bar_rect()
        return rect.left() + pos * rect.width()

    def _pos_of(self, x: float) -> float:
        rect = self._bar_rect()
        return min(1.0, max(0.0, (x - rect.left()) / max(rect.width(), 1.0)))

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self._bar_rect()
        radius = float(theme.size("radius"))
        path = rounded_path(rect, radius)
        _mode, stops = self.owner.value()
        if any(s[4] < 1.0 for s in stops):
            draw_checker(p, path, rect)
        # 按插值方式取 64 个点画（常值、样条都对）
        grad = QLinearGradient(rect.left(), 0.0, rect.right(), 0.0)
        samples = ramp.evaluate(self.owner.value(), [i / 63.0 for i in range(64)])
        for i, c in enumerate(samples):
            grad.setColorAt(i / 63.0, QColor.fromRgbF(float(c[0]), float(c[1]), float(c[2]), float(c[3])))
        if self.owner.value()[0] == "CONSTANT":
            grad = QLinearGradient(rect.left(), 0.0, rect.right(), 0.0)
            for k, s in enumerate(stops):
                c = QColor.fromRgbF(s[1], s[2], s[3], s[4])
                grad.setColorAt(max(0.0, s[0]), c)
                nxt = stops[k + 1][0] if k + 1 < len(stops) else 1.0
                grad.setColorAt(max(0.0, min(1.0, nxt - 1e-4)), c)
            first = stops[0]
            grad.setColorAt(0.0, QColor.fromRgbF(first[1], first[2], first[3], first[4]))
        p.fillPath(path, grad)
        p.setPen(QPen(theme.qcolor("line"), 1.0))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        # 色标：色带上一条竖线，下面一个小三角（当前色标白边）
        active = self.owner.active
        tri = theme.px(6)
        for k, s in enumerate(stops):
            x = self._x_of(s[0])
            is_active = k == active
            p.setPen(QPen(QColor(255, 255, 255) if is_active else QColor(0, 0, 0), 1.5 if is_active else 1.0))
            p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            shape = QPainterPath(QPointF(x, rect.bottom() + 1))
            shape.lineTo(x - tri, rect.bottom() + 1 + tri * 1.5)
            shape.lineTo(x + tri, rect.bottom() + 1 + tri * 1.5)
            shape.closeSubpath()
            p.setBrush(QColor.fromRgbF(s[1], s[2], s[3], 1.0))
            p.setPen(QPen(QColor(255, 255, 255) if is_active else QColor(20, 20, 20), 1.5 if is_active else 1.0))
            p.drawPath(shape)
        p.end()

    def _stop_at(self, x: float) -> int:
        _mode, stops = self.owner.value()
        best, best_d = -1, theme.px(8)
        for k, s in enumerate(stops):
            d = abs(self._x_of(s[0]) - x)
            if d <= best_d:
                best, best_d = k, d
        return best

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.LeftButton or not self.isEnabled():
            super().mousePressEvent(event)
            return
        x = event.position().x()
        if event.modifiers() & Qt.ControlModifier:
            self.owner.add_stop(self._pos_of(x))
            event.accept()
            return
        k = self._stop_at(x)
        if k < 0:
            # 点在空白处：选最近的色标，接着拖它
            _mode, stops = self.owner.value()
            pos = self._pos_of(x)
            k = min(range(len(stops)), key=lambda i: abs(stops[i][0] - pos))
        self.owner.set_active(k)
        self._drag = True
        begin_ui_edit(self)
        self.owner.editingStarted.emit()
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag:
            self.owner.move_active(self._pos_of(event.position().x()))
            event.accept()
            return
        self.setCursor(Qt.SizeHorCursor if self._stop_at(event.position().x()) >= 0 else Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._drag and event.button() == Qt.LeftButton:
            self._drag = False
            self.owner.editingFinished.emit()
            end_ui_edit(self)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self._stop_at(event.position().x()) < 0:
            self.owner.add_stop(self._pos_of(event.position().x()))
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace, Qt.Key_X):
            self.owner.remove_active()
            event.accept()
            return
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            _mode, stops = self.owner.value()
            step = -1 if event.key() == Qt.Key_Left else 1
            self.owner.set_active(max(0, min(len(stops) - 1, self.owner.active + step)))
            event.accept()
            return
        super().keyPressEvent(event)


class ColorRampWidget(QWidget):
    rampChanged = Signal(object)
    editingStarted = Signal()
    editingFinished = Signal()

    def __init__(self, value: Any = None, parent: QWidget | None = None, state: dict | None = None) -> None:
        super().__init__(parent)
        self._value = ramp.normalize(value if value is not None else ramp.DEFAULT)
        # 面板重画后接着用的状态（当前色标），见 ui/widget_state.py
        self._state = state if state is not None else {}
        active = int(self._state.get("active", 0))
        self.active = active if 0 <= active < len(self._value[1]) else 0
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.size("gap"))
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(theme.size("gap"))
        self.add_button = IconButton("add", "加一个色标（也可以按住 Ctrl 点色带）")
        self.remove_button = IconButton("minus", "删掉当前色标")
        self.flip_button = IconButton("swap", "左右翻转")
        self.mode_field = EnumField([(k, label, desc) for k, label, desc in ramp.INTERPOLATIONS])
        self.mode_field.setToolTip("插值方式")
        top.addWidget(self.add_button)
        top.addWidget(self.remove_button)
        top.addWidget(self.mode_field, 1)
        top.addWidget(self.flip_button)
        box.addLayout(top)
        self.bar = _RampBar(self)
        box.addWidget(self.bar)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(theme.size("gap"))
        self.index_field = NumberField(0, 0, ramp.MAX_STOPS - 1, step=1, integer=True, label="色标")
        self.pos_field = NumberField(0.0, 0.0, 1.0, step=0.01, precision=3, label="位置", slider=True)
        self.color_button = ColorButton((0.0, 0.0, 0.0, 1.0))
        bottom.addWidget(self.index_field, 2)
        bottom.addWidget(self.pos_field, 3)
        bottom.addWidget(self.color_button, 2)
        box.addLayout(bottom)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.add_button.clicked.connect(lambda: self.add_stop(None))
        self.remove_button.clicked.connect(self.remove_active)
        self.flip_button.clicked.connect(self._flip)
        self.mode_field.currentChanged.connect(self._set_mode)
        self.index_field.valueChanged.connect(lambda v: self.set_active(int(v)))
        self.pos_field.valueChanged.connect(self.move_active)
        self.pos_field.editingStarted.connect(self.editingStarted)
        self.pos_field.editingFinished.connect(self.editingFinished)
        self.color_button.colorChanged.connect(self._set_color)
        self.color_button.editingStarted.connect(self.editingStarted)
        self.color_button.editingFinished.connect(self.editingFinished)
        self._sync()

    # ---- 值 ----
    def value(self) -> tuple:
        return self._value

    def setValue(self, value: Any, emit: bool = False) -> None:
        value = ramp.normalize(value)
        if value == self._value:
            return
        self._value = value
        self.active = min(self.active, len(value[1]) - 1)
        self._sync()
        if emit:
            self.rampChanged.emit(value)

    def _commit(self, value: tuple, active: int | None = None) -> None:
        if active is not None:
            self.active = active
        if value != self._value:
            self._value = value
            self._sync()
            self.rampChanged.emit(value)
        else:
            self._sync()

    # ---- 操作 ----
    def set_active(self, index: int) -> None:
        index = max(0, min(len(self._value[1]) - 1, int(index)))
        if index != self.active:
            self.active = index
            self._sync()

    def add_stop(self, position: float | None) -> None:
        value, index = ramp.add_stop(self._value, position, self.active)
        self.editingStarted.emit()
        self._commit(value, index)
        self.editingFinished.emit()

    def remove_active(self) -> None:
        value, index = ramp.remove_stop(self._value, self.active)
        self.editingStarted.emit()
        self._commit(value, index)
        self.editingFinished.emit()

    def move_active(self, position: float) -> None:
        value, index = ramp.move_stop(self._value, self.active, float(position))
        self._commit(value, index)

    def _set_color(self, color: tuple) -> None:
        self._commit(ramp.set_color(self._value, self.active, color))

    def _set_mode(self, mode: Any) -> None:
        self.editingStarted.emit()
        self._commit((mode, self._value[1]))
        self.editingFinished.emit()

    def _flip(self) -> None:
        self.editingStarted.emit()
        stops = self._value[1]
        self._commit(ramp.flip(self._value), len(stops) - 1 - self.active)
        self.editingFinished.emit()

    # ---- 显示 ----
    def _sync(self) -> None:
        if not shiboken6.isValid(self):
            return
        mode, stops = self._value
        stop = stops[self.active]
        self._state["active"] = self.active
        self.mode_field.setCurrent(mode)
        self.index_field.setRange(0, len(stops) - 1)
        self.index_field.setValue(self.active)
        self.pos_field.setValue(stop[0])
        self.color_button.setColor(stop[1:])
        self.remove_button.setEnabled(len(stops) > 1)
        self.add_button.setEnabled(len(stops) < ramp.MAX_STOPS)
        self.bar.update()
