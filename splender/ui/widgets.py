"""界面控件。

全部自绘并跟随主题：尺寸只从 theme.px()/theme.size() 取，颜色只从 theme.color()/theme.qcolor() 取。
theme.apply(app, 缩放) 之后新建的控件按新尺寸生成，已有控件在下一次排版和重画时跟上。

各种状态在所有控件上表现一致：
- 悬停：底色变亮（bg.widget → bg.widget_hover）。
- 按下：底色变暗（bg.widget_pressed）。
- 选中、开启、正在拖动的填充：强调色。
- 键盘焦点：强调色描边，只在用 Tab 等键盘方式取得焦点时画，鼠标点击不画。
- 禁用：底色退到 bg.panel_header，文字用 text.disabled。
"""
from __future__ import annotations

import ast
import colorsys
import logging
import math
import operator
import re
import unicodedata
import weakref
from functools import partial
from typing import Any, Callable, Iterable

import shiboken6
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QConicalGradient, QCursor, QFont, QFontMetricsF, QIcon, QKeySequence,
                           QLinearGradient, QPainter, QPainterPath, QPen)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QHBoxLayout, QLineEdit, QMenu, QSizePolicy, QToolTip,
                               QVBoxLayout, QWidget)

from ..core import registry
from ..core.signals import Signal as CoreSignal
from . import icons, theme

log = logging.getLogger("splender.ui")

# ---------------------------------------------------------------------------
# 圆角位置：对齐成组的控件只圆外侧的角
# ---------------------------------------------------------------------------
CORNER_TL = 1
CORNER_TR = 2
CORNER_BL = 4
CORNER_BR = 8
CORNERS_NONE = 0
CORNERS_ALL = CORNER_TL | CORNER_TR | CORNER_BL | CORNER_BR
CORNERS_LEFT = CORNER_TL | CORNER_BL
CORNERS_RIGHT = CORNER_TR | CORNER_BR
CORNERS_TOP = CORNER_TL | CORNER_TR
CORNERS_BOTTOM = CORNER_BL | CORNER_BR

_KEYBOARD_REASONS = (Qt.TabFocusReason, Qt.BacktabFocusReason, Qt.ShortcutFocusReason)
_TIGHT_UNITS = ("°", "%", "′", "″")


# ---------------------------------------------------------------------------
# 绘制工具
# ---------------------------------------------------------------------------
_font_cache: dict[tuple, QFont] = {}


def ui_font(kind: str = "font", bold: bool = False) -> QFont:
    """主题字体（按当前界面缩放）。自绘控件统一用它，不依赖样式表推导出的控件字体。"""
    key = (kind, bold, theme.scale())
    cached = _font_cache.get(key)
    if cached is None:
        cached = theme.font(kind, bold)
        _font_cache[key] = cached
    return QFont(cached)


def line_width() -> float:
    """细线宽度。与样式表里的 1 像素边框一致，界面放大到 2 倍及以上才随之加粗。"""
    return float(max(1, int(theme.scale())))


def mix(a: QColor, b: QColor, t: float) -> QColor:
    """按 t 混合两种颜色，t=1 得到 a，t=0 得到 b。"""
    t = max(0.0, min(1.0, float(t)))
    return QColor.fromRgbF(a.redF() * t + b.redF() * (1.0 - t), a.greenF() * t + b.greenF() * (1.0 - t),
                           a.blueF() * t + b.blueF() * (1.0 - t), a.alphaF() * t + b.alphaF() * (1.0 - t))


def _align_int(*flags: Any) -> int:
    value = 0
    for flag in flags:
        value |= flag.value if hasattr(flag, "value") else int(flag)
    return value


def rounded_path(rect: QRectF, radius: float, corners: int = CORNERS_ALL) -> QPainterPath:
    """圆角矩形路径，corners 指定哪几个角是圆的。"""
    r = max(0.0, min(float(radius), rect.width() / 2.0, rect.height() / 2.0))
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    tl = r if corners & CORNER_TL else 0.0
    tr = r if corners & CORNER_TR else 0.0
    bl = r if corners & CORNER_BL else 0.0
    br = r if corners & CORNER_BR else 0.0
    path = QPainterPath()
    path.moveTo(x + tl, y)
    path.lineTo(x + w - tr, y)
    if tr:
        path.arcTo(QRectF(x + w - 2 * tr, y, 2 * tr, 2 * tr), 90.0, -90.0)
    path.lineTo(x + w, y + h - br)
    if br:
        path.arcTo(QRectF(x + w - 2 * br, y + h - 2 * br, 2 * br, 2 * br), 0.0, -90.0)
    path.lineTo(x + bl, y + h)
    if bl:
        path.arcTo(QRectF(x, y + h - 2 * bl, 2 * bl, 2 * bl), 270.0, -90.0)
    path.lineTo(x, y + tl)
    if tl:
        path.arcTo(QRectF(x, y, 2 * tl, 2 * tl), 180.0, -90.0)
    path.closeSubpath()
    return path


def draw_text(p: QPainter, rect: QRectF, text: str, color: QColor, align: Any = Qt.AlignLeft,
              font: QFont | None = None, elide: bool = True) -> None:
    """单行文字，垂直居中，放不下时末尾省略。"""
    if font is not None:
        p.setFont(font)
    if not text or rect.width() <= 0:
        return
    if elide:
        fm = QFontMetricsF(p.font())
        if fm.horizontalAdvance(text) > rect.width():
            text = fm.elidedText(text, Qt.ElideRight, rect.width())
    p.setPen(color)
    p.drawText(rect, _align_int(align, Qt.AlignVCenter), text)


def draw_focus(p: QPainter, rect: QRectF, radius: float, corners: int = CORNERS_ALL) -> None:
    """键盘焦点描边：画在控件边缘以内，不会被裁掉。"""
    lw = line_width()
    p.save()
    p.setPen(QPen(theme.qcolor("line.focus"), lw))
    p.setBrush(Qt.NoBrush)
    half = lw / 2.0
    p.drawPath(rounded_path(rect.adjusted(half, half, -half, -half), max(0.0, radius - half), corners))
    p.restore()


def draw_outline(p: QPainter, rect: QRectF, radius: float, color: QColor, corners: int = CORNERS_ALL) -> None:
    lw = line_width()
    p.save()
    p.setPen(QPen(color, lw))
    p.setBrush(Qt.NoBrush)
    half = lw / 2.0
    p.drawPath(rounded_path(rect.adjusted(half, half, -half, -half), max(0.0, radius - half), corners))
    p.restore()


def draw_icon(p: QPainter, widget: QWidget, source: Any, color_name: str, size: int, center: QPointF,
              enabled: bool = True, checked: bool = False) -> None:
    """在 center 画一个图标。source 是图标名或 QIcon；名称图标按 color_name 着色。"""
    if source is None or source == "":
        return
    dpr = widget.devicePixelRatioF() if widget is not None else 1.0
    if isinstance(source, QIcon):
        mode = QIcon.Normal if enabled else QIcon.Disabled
        state = QIcon.On if checked else QIcon.Off
        pm = source.pixmap(QSize(size, size), dpr, mode, state)
    else:
        pm = icons.pixmap(str(source), theme.color(color_name), size, dpr)
    x = round(center.x() - size / 2.0)
    y = round(center.y() - size / 2.0)
    p.drawPixmap(QPointF(x, y), pm)


def draw_chevron(p: QPainter, center: QPointF, size: float, direction: str, color: QColor,
                 width: float | None = None) -> None:
    """细线箭头（‹ › ˅ ˄），size 是箭头的宽。"""
    half = size / 2.0
    quarter = size / 4.0
    cx, cy = center.x(), center.y()
    path = QPainterPath()
    if direction == "left":
        path.moveTo(cx + quarter, cy - half)
        path.lineTo(cx - quarter, cy)
        path.lineTo(cx + quarter, cy + half)
    elif direction == "right":
        path.moveTo(cx - quarter, cy - half)
        path.lineTo(cx + quarter, cy)
        path.lineTo(cx - quarter, cy + half)
    elif direction == "up":
        path.moveTo(cx - half, cy + quarter)
        path.lineTo(cx, cy - quarter)
        path.lineTo(cx + half, cy + quarter)
    else:
        path.moveTo(cx - half, cy - quarter)
        path.lineTo(cx, cy + quarter)
        path.lineTo(cx + half, cy - quarter)
    pen = QPen(color, width if width is not None else line_width() * 1.35)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.save()
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    p.restore()


def draw_triangle(p: QPainter, center: QPointF, size: float, direction: str, color: QColor) -> None:
    """实心小三角（折叠面板的开合标记）。"""
    half = size / 2.0
    cx, cy = center.x(), center.y()
    path = QPainterPath()
    if direction == "down":
        h = size * 0.55
        path.moveTo(cx - half, cy - h / 2.0)
        path.lineTo(cx + half, cy - h / 2.0)
        path.lineTo(cx, cy + h / 2.0)
    else:
        w = size * 0.55
        path.moveTo(cx - w / 2.0, cy - half)
        path.lineTo(cx + w / 2.0, cy)
        path.lineTo(cx - w / 2.0, cy + half)
    path.closeSubpath()
    p.save()
    p.setPen(Qt.NoPen)
    p.setBrush(color)
    p.drawPath(path)
    p.restore()


def draw_check(p: QPainter, rect: QRectF, color: QColor) -> None:
    w = rect.width()
    path = QPainterPath()
    path.moveTo(rect.left() + w * 0.25, rect.top() + w * 0.52)
    path.lineTo(rect.left() + w * 0.43, rect.top() + w * 0.70)
    path.lineTo(rect.left() + w * 0.76, rect.top() + w * 0.32)
    pen = QPen(color, max(line_width(), w * 0.14))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.save()
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    p.restore()


def draw_checker(p: QPainter, path: QPainterPath, rect: QRectF) -> None:
    """透明底的棋盘格，用来衬托带透明度的颜色。"""
    cell = float(theme.px(5))
    light = theme.qcolor("text.faint")
    dark = theme.qcolor("bg.widget")
    p.save()
    p.setClipPath(path)
    p.fillRect(rect, dark)
    y = rect.top()
    row = 0
    while y < rect.bottom():
        x = rect.left() + (cell if row % 2 else 0.0)
        while x < rect.right():
            p.fillRect(QRectF(x, y, cell, cell), light)
            x += cell * 2
        y += cell
        row += 1
    p.restore()


def normalize_items(raw: Any) -> list[tuple]:
    """把 [(id, 文字, 说明, 图标)] 这类列表整理成统一的四元组，缺的项补空。"""
    if callable(raw):
        raw = raw()
    out: list[tuple] = []
    for item in raw or ():
        if isinstance(item, str):
            out.append((item, item, "", ""))
            continue
        item = tuple(item)
        if not item:
            continue
        ident = item[0]
        text = item[1] if len(item) > 1 else str(ident)
        desc = item[2] if len(item) > 2 else ""
        icon = item[3] if len(item) > 3 else ""
        out.append((ident, str(text), str(desc or ""), icon or ""))
    return out


def _icon_ok(icon: Any) -> bool:
    if isinstance(icon, QIcon):
        return not icon.isNull()
    return bool(icon) and icons.exists(str(icon))


# ---------------------------------------------------------------------------
# 正在进行的界面编辑（拖动数值、输入文字、打开的取色浮层）
# 面板宿主在编辑进行中不重建内容，等编辑结束再刷新，避免把正在拖的控件删掉。
# ---------------------------------------------------------------------------
_active_edits: dict[int, list] = {}

#: 有界面编辑结束时发出（无参数）。
edit_state_changed = CoreSignal()


def begin_ui_edit(widget: QWidget) -> None:
    key = id(widget)
    entry = _active_edits.get(key)
    if entry is None or entry[0]() is not widget:
        _active_edits[key] = [weakref.ref(widget), 1]
    else:
        entry[1] += 1


def end_ui_edit(widget: QWidget) -> None:
    key = id(widget)
    entry = _active_edits.get(key)
    if entry is not None:
        entry[1] -= 1
        if entry[1] <= 0:
            _active_edits.pop(key, None)
    edit_state_changed.emit()


def ui_edit_active(container: QWidget | None = None) -> bool:
    """是否有编辑正在进行；给出 container 时只看它里面的控件。"""
    for key, (ref, _count) in list(_active_edits.items()):
        widget = ref()
        if widget is None or not shiboken6.isValid(widget):
            _active_edits.pop(key, None)
            continue
        if container is None or widget is container:
            return True
        try:
            if container.isAncestorOf(widget):
                return True
        except RuntimeError:
            return False
    return False


# ---------------------------------------------------------------------------
# 数字表达式：数值框里可以输入 60*2、1/3、50% 这类写法
# ---------------------------------------------------------------------------
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_NAMES = {"pi": math.pi, "tau": math.tau, "e": math.e}


def evaluate_number(text: str) -> float | None:
    """安全地算出一个算术表达式，算不出返回 None。"""
    source = str(text).strip()
    if not source or len(source) > 200:
        return None
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 64 or abs(left) > 1e6):
                raise ValueError("指数过大")
            return float(_BINOPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return float(_UNARY[type(node.op)](ev(node.operand)))
        if isinstance(node, ast.Name) and node.id in _NAMES:
            return _NAMES[node.id]
        raise ValueError("不支持的写法")

    try:
        value = ev(tree)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# 公共混入：悬停、键盘焦点、对齐圆角、右键菜单钩子
# ---------------------------------------------------------------------------
class _Interactive:
    """自绘控件的公共行为。混在 Qt 基类前面：class X(_Interactive, QWidget)。"""

    _corners = CORNERS_ALL
    _kbd_focus = False
    _suppress_context = False
    #: 右键菜单钩子 fn(全局坐标)。布局语句用它挂「恢复默认值、复制数值、粘贴数值」。
    context_menu_hook: Callable[[QPoint], None] | None = None

    def set_corners(self, corners: int) -> None:
        """对齐成组时由布局调用：只圆组外侧的角。"""
        if corners != self._corners:
            self._corners = corners
            self.update()

    def corners(self) -> int:
        return self._corners

    def _hovered(self) -> bool:
        return self.isEnabled() and self.underMouse()

    def _focus_visible(self) -> bool:
        return self._kbd_focus and self.hasFocus()

    def enterEvent(self, event: Any) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self.update()
        super().leaveEvent(event)

    def focusInEvent(self, event: Any) -> None:
        self._kbd_focus = event.reason() in _KEYBOARD_REASONS
        self.update()
        super().focusInEvent(event)

    def focusOutEvent(self, event: Any) -> None:
        self._kbd_focus = False
        self.update()
        super().focusOutEvent(event)

    def changeEvent(self, event: Any) -> None:
        if event.type() == QEvent.EnabledChange:
            self.update()
        super().changeEvent(event)

    def contextMenuEvent(self, event: Any) -> None:
        if self._suppress_context:
            self._suppress_context = False
            event.accept()
            return
        hook = self.context_menu_hook
        if hook is not None and self.isEnabled():
            event.accept()
            hook(event.globalPos())
            return
        self._default_context_menu(event)

    def _default_context_menu(self, event: Any) -> None:
        event.ignore()


def _popup_menu(menu: QMenu, global_pos: QPoint) -> None:
    menu.setAttribute(Qt.WA_DeleteOnClose, True)
    menu.popup(global_pos)


def _text_edit_menu(edit: QLineEdit, extra: Callable[[QMenu], None] | None = None) -> QMenu:
    """文本框的右键菜单（中文）。extra 可以在末尾追加条目。"""
    menu = QMenu(edit)
    has_sel = edit.hasSelectedText()
    read_only = edit.isReadOnly()
    normal = edit.echoMode() == QLineEdit.Normal
    menu.addAction("撤销", edit.undo).setEnabled(edit.isUndoAvailable() and not read_only)
    menu.addAction("重做", edit.redo).setEnabled(edit.isRedoAvailable() and not read_only)
    menu.addSeparator()
    menu.addAction("剪切", edit.cut).setEnabled(has_sel and normal and not read_only)
    menu.addAction("复制", edit.copy).setEnabled(has_sel and normal)
    menu.addAction("粘贴", edit.paste).setEnabled(not read_only and bool(QApplication.clipboard().text()))
    menu.addAction("删除", edit.del_).setEnabled(has_sel and not read_only)
    menu.addSeparator()
    menu.addAction("全选", edit.selectAll).setEnabled(bool(edit.text()))
    if extra is not None:
        menu.addSeparator()
        extra(menu)
    return menu


# ---------------------------------------------------------------------------
# 数值框
# ---------------------------------------------------------------------------
class NumberField(_Interactive, QWidget):
    """Blender 式数值框。

    - 横向拖动改值，拖过整个软范围所需的距离见 drag_span（滑杆模式下是控件宽度）；拖动只在软范围内走。
    - 拖动时按住 Shift 精调（fine_factor 倍），按住 Ctrl 按步长取整。右键或 Esc 取消本次拖动。
    - 单击（没拖动）进入文字编辑：回车确认，Esc 取消，双击全选，可以输入 60*2、1/3、50% 这类算式。
    - 悬停时两端出现小箭头，单击箭头按步长加减。
    - 有键盘焦点时方向键、滚轮按步长加减；没有焦点时滚轮交给外层滚动，避免滚动面板时误改。
    - slider=True 时按软范围画填充条。label 画在框内左侧。
    """

    valueChanged = Signal(float)
    editingStarted = Signal()
    editingFinished = Signal()

    #: 非滑杆模式下，拖过整个软范围需要的逻辑像素。
    drag_span = 300
    #: 按下后移动超过这么多逻辑像素才算拖动，否则当作单击。
    drag_threshold = 3
    #: Shift 精调倍率。
    fine_factor = 0.1
    #: 滚轮或方向键连续调整时，停多久（毫秒）算一次编辑结束。
    burst_ms = 600
    #: 悬停箭头区宽度（逻辑像素）。
    arrow_width = 16

    def __init__(self, value: float = 0, minimum: float = -math.inf, maximum: float = math.inf,
                 soft_min: float | None = None, soft_max: float | None = None, step: float | None = None,
                 precision: int = 2, unit: str = "", label: str = "", slider: bool = False, integer: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._integer = bool(integer)
        self._precision = 0 if self._integer else max(0, int(precision))
        self._step = float(step) if step else None
        self._unit = unit or ""
        self._label = label or ""
        self._slider = bool(slider)
        self._value = 0.0
        self.setRange(minimum, maximum, soft_min, soft_max)
        self._value = self._clamp(self._number(value, 0.0))
        self._pressed = False
        self._dragging = False
        self._press_x = 0.0
        self._last_x = 0.0
        self._press_value = self._value
        self._drag_raw = self._value
        self._drag_lo = -math.inf
        self._drag_hi = math.inf
        self._hover_zone = 0
        self._editor: _InlineEdit | None = None
        self._editing = False
        self._edit_from_keyboard = False
        self._burst = False
        self._wheel_accum = 0.0
        self._burst_timer = QTimer(self)
        self._burst_timer.setSingleShot(True)
        self._burst_timer.timeout.connect(self._end_burst)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAccessibleName(self._label)

    # ---- 取值与设置 ----
    @staticmethod
    def _number(value: Any, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        return number if math.isfinite(number) else fallback

    def _clamp(self, value: float) -> float:
        value = min(self._max, max(self._min, value))
        if self._integer:
            value = float(round(value))
        return value

    def value(self) -> float | int:
        return int(self._value) if self._integer else self._value

    def setValue(self, value: float, emit: bool = False) -> None:
        number = self._number(value, self._value)
        self._set(number, emit)

    def _set(self, number: float, emit: bool) -> bool:
        number = self._clamp(round(number, 10))
        if number == self._value:
            return False
        self._value = number
        self.update()
        if emit:
            self.valueChanged.emit(float(number))
        return True

    def setRange(self, minimum: float | None = -math.inf, maximum: float | None = math.inf,
                 soft_min: float | None = None, soft_max: float | None = None) -> None:
        """硬范围限制所有输入；软范围决定拖动的行程和滑杆的填充。软范围不给就等于硬范围。"""
        lo = -math.inf if minimum is None else float(minimum)
        hi = math.inf if maximum is None else float(maximum)
        if lo > hi:
            lo, hi = hi, lo
        smin = lo if soft_min is None else max(lo, float(soft_min))
        smax = hi if soft_max is None else min(hi, float(soft_max))
        if smin > smax:
            smin, smax = lo, hi
        self._min, self._max, self._soft_min, self._soft_max = lo, hi, smin, smax
        self._value = self._clamp(self._value)
        self.update()

    def minimum(self) -> float:
        return self._min

    def maximum(self) -> float:
        return self._max

    def softRange(self) -> tuple[float, float]:
        return self._soft_min, self._soft_max

    def label(self) -> str:
        return self._label

    def setLabel(self, label: str) -> None:
        self._label = label or ""
        self.setAccessibleName(self._label)
        self.updateGeometry()
        self.update()

    def unit(self) -> str:
        return self._unit

    def setUnit(self, unit: str) -> None:
        self._unit = unit or ""
        self.updateGeometry()
        self.update()

    def precision(self) -> int:
        return self._precision

    def setPrecision(self, precision: int) -> None:
        if not self._integer:
            self._precision = max(0, int(precision))
            self.update()

    def step(self) -> float | None:
        return self._step

    def setStep(self, step: float | None) -> None:
        self._step = float(step) if step else None

    def isSlider(self) -> bool:
        return self._slider

    def setSlider(self, slider: bool) -> None:
        self._slider = bool(slider)
        self.update()

    def isInteger(self) -> bool:
        return self._integer

    def isEditing(self) -> bool:
        """正在拖动、输入文字或连续调整中。"""
        return self._dragging or self._editing or self._burst

    def isDragging(self) -> bool:
        return self._dragging

    def isTextEditing(self) -> bool:
        return self._editing

    def editor(self) -> QLineEdit | None:
        """文字编辑时盖在上面的输入框（测试和外部检查用）。"""
        return self._editor

    # ---- 文本 ----
    def format_value(self, value: float, with_unit: bool = True) -> str:
        if self._integer:
            text = str(int(round(value)))
        else:
            if abs(value) < 0.5 * 10.0 ** -self._precision:
                value = 0.0
            text = "%.*f" % (self._precision, value)
        if with_unit and self._unit:
            text += self._unit if self._unit in _TIGHT_UNITS else " " + self._unit
        return text

    def text(self, with_unit: bool = True) -> str:
        return self.format_value(self._value, with_unit)

    def parse_text(self, text: str) -> float | None:
        """把输入的文字换成数值：去掉单位，支持算式和百分号，全角字符也认。"""
        source = unicodedata.normalize("NFKC", str(text)).strip()
        if not source:
            return None
        unit = self._unit.strip()
        if unit and source.lower().endswith(unit.lower()):
            source = source[: -len(unit)].strip()
        percent = False
        if source.endswith("%") and unit != "%":
            percent = True
            source = source[:-1].strip()
        source = source.replace("×", "*").replace("÷", "/").replace("^", "**")
        if source.count(",") == 1 and "." not in source:
            source = source.replace(",", ".")
        value = evaluate_number(source)
        if value is None:
            return None
        return value / 100.0 if percent else value

    # ---- 步长 ----
    def _soft_span(self) -> float | None:
        span = self._soft_max - self._soft_min
        return span if math.isfinite(span) and 0.0 < span <= 1e7 else None

    def snap_step(self) -> float:
        """Ctrl 拖动时取整的步长：有给定步长就用它，否则按软范围大小取 0.1、1、10、100。"""
        if self._step and (not self._integer or self._step > 1):
            return self._step
        span = self._soft_span()
        if span is None:
            base = 1.0 if self._integer else 10.0 ** (1 - self._precision)
        elif span < 2.1:
            base = 0.1
        elif span < 21.0:
            base = 1.0
        elif span < 2100.0:
            base = 10.0
        else:
            base = 100.0
        return max(1.0, base) if self._integer else base

    def increment(self) -> float:
        """方向键、滚轮、两端箭头每次改变的量。"""
        if self._step:
            return float(max(1, round(self._step))) if self._integer else self._step
        if self._integer:
            return 1.0
        return max(10.0 ** -self._precision, self.snap_step() / 10.0)

    def _units_per_px(self) -> float:
        span = self._soft_span()
        if span is not None:
            length = float(self.width()) if self._slider else float(theme.px(self.drag_span))
            return span / max(1.0, length)
        if self._integer:
            return 0.5 / theme.scale()
        return 10.0 ** (1 - self._precision) / 4.0 / theme.scale()

    def _fraction(self) -> float:
        span = self._soft_max - self._soft_min
        if not math.isfinite(span) or span <= 0:
            return 0.0
        return max(0.0, min(1.0, (self._value - self._soft_min) / span))

    # ---- 绘制 ----
    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(ui_font())
        pad = theme.size("pad")
        width = fm.horizontalAdvance(self.text()) + 2 * pad
        if self._label:
            width += fm.horizontalAdvance(self._label) + pad
        width = max(width, theme.px(64))
        return QSize(int(math.ceil(width)), theme.size("widget"))

    def minimumSizeHint(self) -> QSize:
        fm = QFontMetricsF(ui_font())
        width = fm.horizontalAdvance(self.text()) + 2 * theme.size("pad")
        return QSize(int(math.ceil(max(width, theme.px(32)))), theme.size("widget"))

    def _arrows_visible(self) -> bool:
        return (not self._slider and self.isEnabled() and not self._dragging and not self._editing
                and self.underMouse() and self.width() >= theme.px(self.arrow_width) * 4)

    def _zone_at(self, pos: QPointF) -> int:
        if self._slider or self.width() < theme.px(self.arrow_width) * 4:
            return 0
        aw = theme.px(self.arrow_width)
        if pos.x() < aw:
            return -1
        if pos.x() > self.width() - aw:
            return 1
        return 0

    def _fill_color(self, enabled: bool, hover: bool) -> QColor:
        soft = theme.qcolor("accent.soft")
        if not enabled:
            return mix(soft, theme.qcolor("bg.panel_header"), 0.45)
        if hover:
            return mix(theme.qcolor("accent"), soft, 0.22)
        return soft

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        radius = float(theme.size("radius"))
        enabled = self.isEnabled()
        hover = enabled and (self.underMouse() or self._dragging)
        if not enabled:
            bg = theme.qcolor("bg.panel_header")
        elif hover:
            bg = theme.qcolor("bg.widget_hover")
        else:
            bg = theme.qcolor("bg.widget")
        path = rounded_path(rect, radius, self._corners)
        p.fillPath(path, bg)
        if self._slider:
            width = round(rect.width() * self._fraction())
            if width > 0:
                p.save()
                p.setClipRect(QRectF(rect.left(), rect.top(), width, rect.height()))
                p.fillPath(path, self._fill_color(enabled, hover))
                p.restore()
        if self._editing:
            p.end()
            return
        font = ui_font()
        p.setFont(font)
        fm = QFontMetricsF(font)
        pad = float(theme.size("pad"))
        inner = rect.adjusted(pad, 0, -pad, 0)
        if self._arrows_visible():
            aw = float(theme.px(self.arrow_width))
            size = float(theme.px(7))
            for zone, direction, x in ((-1, "left", aw / 2.0 + line_width()), (1, "right", rect.width() - aw / 2.0 - line_width())):
                color = theme.qcolor("text" if self._hover_zone == zone else "text.faint")
                draw_chevron(p, QPointF(x, rect.center().y()), size, direction, color)
            inner = rect.adjusted(aw, 0, -aw, 0)
        value_text = self.text()
        value_color = theme.qcolor("text" if enabled else "text.disabled")
        if self._label:
            label_color = theme.qcolor("text.dim" if enabled else "text.disabled")
            value_w = fm.horizontalAdvance(value_text)
            label_w = inner.width() - value_w - pad * 0.75
            if label_w >= fm.horizontalAdvance("…") * 2:
                draw_text(p, QRectF(inner.left(), rect.top(), label_w, rect.height()), self._label, label_color,
                          Qt.AlignLeft)
            draw_text(p, QRectF(inner.left(), rect.top(), inner.width(), rect.height()), value_text, value_color,
                      Qt.AlignRight)
        else:
            draw_text(p, QRectF(inner.left(), rect.top(), inner.width(), rect.height()), value_text, value_color,
                      Qt.AlignHCenter)
        if self._focus_visible():
            draw_focus(p, rect, radius, self._corners)
        p.end()

    # ---- 鼠标 ----
    def mousePressEvent(self, event: Any) -> None:
        if not self.isEnabled():
            event.ignore()
            return
        button = event.button()
        if button == Qt.LeftButton and not self._editing:
            self._end_burst()
            self._pressed = True
            self._dragging = False
            x = event.position().x()
            self._press_x = x
            self._last_x = x
            self._press_value = self._value
            event.accept()
            return
        if button == Qt.RightButton and (self._pressed or self._dragging):
            self._cancel_drag()
            self._suppress_context = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        x = event.position().x()
        if self._pressed and event.buttons() & Qt.LeftButton:
            if not self._dragging and abs(x - self._press_x) >= theme.px(self.drag_threshold):
                self._begin_drag(x)
            if self._dragging:
                self._drag_to(x, event.modifiers())
            event.accept()
            return
        zone = self._zone_at(event.position())
        if zone != self._hover_zone:
            self._hover_zone = zone
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self._pressed:
            self._pressed = False
            if self._dragging:
                self._finish_drag()
            else:
                zone = self._zone_at(event.position())
                if zone:
                    self._step_once(zone)
                else:
                    self.begin_text_edit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        if self._editing and self._editor is not None:
            self._editor.selectAll()
            event.accept()
            return
        self.mousePressEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._hover_zone = 0
        super().leaveEvent(event)

    def _begin_drag(self, x: float) -> None:
        self._dragging = True
        self._last_x = x
        self._drag_raw = self._value
        self._drag_lo = max(self._min, min(self._soft_min, self._value))
        self._drag_hi = min(self._max, max(self._soft_max, self._value))
        self.setCursor(Qt.SizeHorCursor)
        begin_ui_edit(self)
        self.editingStarted.emit()
        self.update()

    def _drag_to(self, x: float, modifiers: Any) -> None:
        dx = x - self._last_x
        self._last_x = x
        fine = bool(modifiers & Qt.ShiftModifier)
        snap = bool(modifiers & Qt.ControlModifier)
        rate = self._units_per_px() * (self.fine_factor if fine else 1.0)
        self._drag_raw = min(self._drag_hi, max(self._drag_lo, self._drag_raw + dx * rate))
        value = self._drag_raw
        if snap:
            step = self.snap_step() * (self.fine_factor if fine else 1.0)
            if step > 0:
                value = round(round(value / step) * step, 10)
        elif not self._integer:
            value = round(value, self._precision + (1 if fine else 0))
        value = min(self._drag_hi, max(self._drag_lo, value))
        self._set(value, emit=True)

    def _finish_drag(self) -> None:
        self._dragging = False
        self.unsetCursor()
        self.editingFinished.emit()
        end_ui_edit(self)
        if not self._kbd_focus:
            self.clearFocus()
        self.update()

    def _cancel_drag(self) -> None:
        was_dragging = self._dragging
        self._pressed = False
        self._dragging = False
        self.unsetCursor()
        if was_dragging:
            self._set(self._press_value, emit=True)
            self.editingFinished.emit()
            end_ui_edit(self)
        self.update()

    def cancel_drag(self) -> None:
        """取消正在进行的拖动，恢复拖动前的值。"""
        if self._pressed or self._dragging:
            self._cancel_drag()

    def _step_once(self, direction: int) -> None:
        begin_ui_edit(self)
        self.editingStarted.emit()
        self._set(self._value + direction * self.increment(), emit=True)
        self.editingFinished.emit()
        end_ui_edit(self)
        if not self._kbd_focus:
            self.clearFocus()

    # ---- 文字编辑 ----
    def begin_text_edit(self) -> None:
        """进入文字编辑。"""
        if self._editing or not self.isEnabled():
            return
        self._end_burst()
        if self._editor is None:
            self._editor = _InlineEdit(self)
        editor = self._editor
        editor.apply_style(self._corners)
        editor.setGeometry(self.rect())
        editor.setText(self.format_value(self._value, with_unit=False))
        self._edit_from_keyboard = self._kbd_focus and self.hasFocus()
        self._editing = True
        begin_ui_edit(self)
        self.editingStarted.emit()
        editor.show()
        editor.setFocus(Qt.OtherFocusReason)
        editor.selectAll()
        self.update()

    def finish_text_edit(self, commit: bool = True) -> None:
        """结束文字编辑；commit=False 时放弃输入。"""
        self._finish_text_edit(commit)

    def _finish_text_edit(self, commit: bool) -> None:
        if not self._editing:
            return
        self._editing = False
        editor = self._editor
        if commit and editor is not None:
            parsed = self.parse_text(editor.text())
            if parsed is not None:
                self._set(parsed, emit=True)
        if editor is not None:
            if editor.hasFocus():
                # 先把焦点收回来再藏输入框，否则 Qt 会把焦点交给下一个控件
                self.setFocus(Qt.OtherFocusReason)
                if self._edit_from_keyboard:
                    self._kbd_focus = True
                else:
                    self.clearFocus()
            editor.hide()
        self.editingFinished.emit()
        end_ui_edit(self)
        self.update()

    # ---- 键盘与滚轮 ----
    def keyPressEvent(self, event: Any) -> None:
        key = event.key()
        mods = event.modifiers()
        if self._dragging:
            if key == Qt.Key_Escape:
                self._cancel_drag()
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_F2, Qt.Key_Space):
            self.begin_text_edit()
            event.accept()
            return
        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            sign = 1 if key in (Qt.Key_Right, Qt.Key_Up) else -1
            amount = self.increment()
            if mods & Qt.ControlModifier:
                amount = self.snap_step()
            if mods & Qt.ShiftModifier:
                amount *= self.fine_factor
            self._bump(sign * amount)
            event.accept()
            return
        if event.matches(QKeySequence.Copy):
            self.copy_value()
            event.accept()
            return
        if event.matches(QKeySequence.Paste):
            self.paste_value()
            event.accept()
            return
        text = event.text()
        if text and (text.isdigit() or text in "-+.") and not (mods & (Qt.ControlModifier | Qt.AltModifier)):
            self.begin_text_edit()
            if self._editor is not None:
                self._editor.setText(text)
            event.accept()
            return
        if key == Qt.Key_Escape and self.hasFocus():
            self.clearFocus()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: Any) -> None:
        if not (self.isEnabled() and self._focus_visible()) or self._dragging or self._editing:
            event.ignore()
            return
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            event.ignore()
            return
        self._wheel_accum += delta / 120.0
        steps = int(self._wheel_accum)
        if steps:
            self._wheel_accum -= steps
            amount = self.increment() * (self.fine_factor if event.modifiers() & Qt.ShiftModifier else 1.0)
            self._bump(steps * amount)
        event.accept()

    def _bump(self, delta: float) -> None:
        if not self._burst:
            self._burst = True
            begin_ui_edit(self)
            self.editingStarted.emit()
        self._set(self._value + delta, emit=True)
        self._burst_timer.start(self.burst_ms)

    def _end_burst(self) -> None:
        if not self._burst:
            return
        self._burst = False
        self._burst_timer.stop()
        self.editingFinished.emit()
        end_ui_edit(self)

    def focusOutEvent(self, event: Any) -> None:
        self._end_burst()
        super().focusOutEvent(event)

    def hideEvent(self, event: Any) -> None:
        if self._dragging:
            self._pressed = False
            self._finish_drag()
        if self._editing:
            self._finish_text_edit(True)
        self._end_burst()
        super().hideEvent(event)

    def changeEvent(self, event: Any) -> None:
        if event.type() == QEvent.EnabledChange and not self.isEnabled():
            if self._dragging:
                self._cancel_drag()
            if self._editing:
                self._finish_text_edit(False)
            self._end_burst()
        super().changeEvent(event)

    # ---- 剪贴板 ----
    def copy_value(self) -> None:
        QApplication.clipboard().setText(self.format_value(self._value, with_unit=False))

    def paste_value(self) -> bool:
        value = self.parse_text(QApplication.clipboard().text())
        if value is None:
            return False
        begin_ui_edit(self)
        self.editingStarted.emit()
        self._set(value, emit=True)
        self.editingFinished.emit()
        end_ui_edit(self)
        return True

    def _default_context_menu(self, event: Any) -> None:
        menu = QMenu(self)
        menu.addAction("复制数值", self.copy_value)
        menu.addAction("粘贴数值", self.paste_value).setEnabled(
            self.parse_text(QApplication.clipboard().text()) is not None)
        _popup_menu(menu, event.globalPos())
        event.accept()


class _InlineEdit(QLineEdit):
    """数值框进入文字编辑时盖在上面的输入框。"""

    def __init__(self, owner: NumberField) -> None:
        super().__init__(owner)
        self._owner = owner
        self.hide()

    def apply_style(self, corners: int) -> None:
        r = theme.size("radius")
        lw = int(line_width())
        radii = [r if corners & flag else 0 for flag in (CORNER_TL, CORNER_TR, CORNER_BR, CORNER_BL)]
        self.setStyleSheet(
            "QLineEdit {"
            f" background: {theme.color('bg.field')}; color: {theme.color('text')};"
            f" border: {lw}px solid {theme.color('line.focus')};"
            f" border-top-left-radius: {radii[0]}px; border-top-right-radius: {radii[1]}px;"
            f" border-bottom-right-radius: {radii[2]}px; border-bottom-left-radius: {radii[3]}px;"
            f" padding: 0px {max(0, theme.size('pad') - lw)}px; min-height: 0px;"
            f" selection-background-color: {theme.color('accent')}; selection-color: {theme.color('text.on_accent')};"
            " }")

    def keyPressEvent(self, event: Any) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            self._owner._finish_text_edit(False)
            event.accept()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._owner._finish_text_edit(True)
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event: Any) -> None:
        super().focusOutEvent(event)
        if event.reason() == Qt.PopupFocusReason:
            return
        if self._owner._editing:
            self._owner._finish_text_edit(True)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        self.selectAll()
        event.accept()

    def contextMenuEvent(self, event: Any) -> None:
        _popup_menu(_text_edit_menu(self), event.globalPos())
        event.accept()


# ---------------------------------------------------------------------------
# 按钮
# ---------------------------------------------------------------------------
class _ButtonBase(_Interactive, QAbstractButton):
    """自绘按钮：底色、图标、文字、可选的下拉箭头。"""

    def __init__(self, text: str = "", icon: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icon_src = icon if _icon_ok(icon) or isinstance(icon, QIcon) else None
        self._role: str | None = None
        self._flat = False
        self._menu_indicator = False
        self._content_align = Qt.AlignHCenter
        self._icon_logical: int | None = None
        self.setText(text or "")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        if text:
            self.setAccessibleName(text)

    # ---- 外观设置 ----
    def set_icon(self, icon: Any) -> None:
        self._icon_src = icon if _icon_ok(icon) or isinstance(icon, QIcon) else None
        self.updateGeometry()
        self.update()

    def icon_name(self) -> Any:
        return self._icon_src

    def set_icon_size(self, logical: int | None) -> None:
        self._icon_logical = logical
        self.update()

    def set_role(self, role: str | None) -> None:
        """role="primary" 用强调色。"""
        self._role = role
        self.update()

    def role(self) -> str | None:
        return self._role

    def set_flat(self, flat: bool) -> None:
        """平时透明，悬停才出底色（标题栏里的按钮常用）。"""
        self._flat = bool(flat)
        self.update()

    def set_menu_indicator(self, on: bool) -> None:
        """右侧画一个向下的小箭头，表示点开是菜单或浮层。"""
        self._menu_indicator = bool(on)
        self.updateGeometry()
        self.update()

    def set_content_alignment(self, align: Any) -> None:
        self._content_align = align
        self.update()

    def _icon_size(self) -> int:
        if self._icon_logical:
            return theme.px(self._icon_logical)
        return theme.size("icon")

    # ---- 尺寸 ----
    def sizeHint(self) -> QSize:
        h = theme.size("widget")
        text = self.text()
        if not text and self._icon_src is not None and not self._menu_indicator:
            return QSize(h, h)
        fm = QFontMetricsF(ui_font())
        pad = theme.size("pad")
        width = 2 * pad + fm.horizontalAdvance(text)
        if self._icon_src is not None:
            width += self._icon_size() + (theme.px(6) if text else 0)
        if self._menu_indicator:
            width += theme.px(8) + theme.px(6)
        return QSize(int(math.ceil(max(width, h))), h)

    def minimumSizeHint(self) -> QSize:
        h = theme.size("widget")
        if not self.text():
            return self.sizeHint()
        return QSize(h, h)

    # ---- 绘制 ----
    def _state_colors(self) -> tuple[QColor | None, str]:
        enabled = self.isEnabled()
        hover = self._hovered()
        down = self.isDown()
        strong = (self.isCheckable() and self.isChecked()) or self._role == "primary"
        if not enabled:
            if strong:
                return mix(theme.qcolor("accent.soft"), theme.qcolor("bg.panel_header"), 0.5), "text.disabled"
            return (None if self._flat else theme.qcolor("bg.panel_header")), "text.disabled"
        if strong:
            name = "accent.pressed" if down else "accent.hover" if hover else "accent"
            return theme.qcolor(name), "text.on_accent"
        if down:
            return theme.qcolor("bg.widget_pressed"), "text"
        if hover:
            return theme.qcolor("bg.widget_hover"), "text"
        if self._flat:
            return None, "text.dim"
        return theme.qcolor("bg.widget"), "text"

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        radius = float(theme.size("radius"))
        bg, fg = self._state_colors()
        if bg is not None:
            p.fillPath(rounded_path(rect, radius, self._corners), bg)
        self._paint_content(p, rect, fg)
        if self._focus_visible():
            draw_focus(p, rect, radius, self._corners)
        p.end()

    def _paint_content(self, p: QPainter, rect: QRectF, fg: str) -> None:
        text = self.text()
        enabled = self.isEnabled()
        checked = self.isCheckable() and self.isChecked()
        pad = float(theme.size("pad"))
        icon_px = self._icon_size()
        inner = rect.adjusted(pad, 0, -pad, 0)
        if self._menu_indicator:
            arrow = float(theme.px(8))
            draw_chevron(p, QPointF(inner.right() - arrow / 2.0, rect.center().y()), arrow, "down",
                         theme.qcolor("text.dim" if fg == "text" else fg))
            inner.setRight(inner.right() - arrow - theme.px(6))
        if not text:
            if self._icon_src is not None:
                center = rect.center() if not self._menu_indicator else QPointF(inner.center().x(), rect.center().y())
                draw_icon(p, self, self._icon_src, fg, icon_px, center, enabled, checked)
            return
        font = ui_font()
        fm = QFontMetricsF(font)
        gap = float(theme.px(6))
        text_w = fm.horizontalAdvance(text)
        content_w = text_w + (icon_px + gap if self._icon_src is not None else 0.0)
        if self._content_align == Qt.AlignLeft or content_w > inner.width():
            x = inner.left()
        else:
            x = inner.left() + (inner.width() - content_w) / 2.0
        if self._icon_src is not None:
            draw_icon(p, self, self._icon_src, fg, icon_px, QPointF(x + icon_px / 2.0, rect.center().y()), enabled,
                      checked)
            x += icon_px + gap
        draw_text(p, QRectF(x, rect.top(), max(0.0, inner.right() - x), rect.height()), text, theme.qcolor(fg),
                  Qt.AlignLeft, font)


class Button(_ButtonBase):
    """普通按钮：文字、图标或两者。role="primary" 用强调色。"""

    def __init__(self, text: str = "", icon: Any = None, role: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(text, icon, parent)
        self._role = role


class Toggle(_ButtonBase):
    """开关按钮：按下保持，开启时用强调色。信号 toggled(bool)。"""

    def __init__(self, text: str = "", icon: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(text, icon, parent)
        self.setCheckable(True)


class IconButton(_ButtonBase):
    """纯图标按钮。平时透明，悬停出底色；checkable 时开启用强调色。tooltip 写这个按钮是做什么的。

    size 是按钮边长（逻辑像素），不给就和控件同高；边长达到工具按钮尺寸时用大号图标。
    """

    def __init__(self, icon: Any, tooltip: str = "", checkable: bool = False, size: int | None = None,
                 parent: QWidget | None = None, *, icon_size: int | None = None) -> None:
        super().__init__("", icon, parent)
        self._flat = True
        self._side = size
        self._icon_logical = icon_size
        self.setCheckable(bool(checkable))
        if tooltip:
            self.setToolTip(tooltip)
            self.setAccessibleName(tooltip)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def _side_px(self) -> int:
        return theme.px(self._side) if self._side else theme.size("widget")

    def _icon_size(self) -> int:
        if self._icon_logical:
            return theme.px(self._icon_logical)
        return theme.size("icon.tool") if self._side_px() >= theme.size("tool_button") else theme.size("icon")

    def sizeHint(self) -> QSize:
        side = self._side_px()
        return QSize(side, side)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class CheckBox(_Interactive, QAbstractButton):
    """勾选框，文字在右。整行都可以点。"""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setText(text or "")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        if text:
            self.setAccessibleName(text)

    def _box(self) -> float:
        return float(theme.px(14))

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(ui_font())
        width = self._box()
        if self.text():
            width += theme.px(6) + fm.horizontalAdvance(self.text()) + line_width()
        return QSize(int(math.ceil(width)), theme.size("widget"))

    def minimumSizeHint(self) -> QSize:
        return QSize(int(self._box()), theme.size("widget"))

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        box = self._box()
        top = math.floor((rect.height() - box) / 2.0)
        indicator = QRectF(0.0, top, box, box)
        enabled = self.isEnabled()
        hover = self._hovered()
        down = self.isDown()
        checked = self.isChecked()
        radius = float(theme.size("radius.small"))
        if checked:
            if enabled:
                fill = theme.qcolor("accent.pressed" if down else "accent.hover" if hover else "accent")
            else:
                fill = mix(theme.qcolor("accent.soft"), theme.qcolor("bg.panel_header"), 0.5)
        elif enabled:
            fill = theme.qcolor("bg.widget_pressed" if down else "bg.widget_hover" if hover else "bg.widget")
        else:
            fill = theme.qcolor("bg.panel_header")
        p.fillPath(rounded_path(indicator, radius), fill)
        if checked:
            draw_check(p, indicator, theme.qcolor("text.on_accent" if enabled else "text.disabled"))
        if self._focus_visible():
            draw_focus(p, indicator, radius)
        if self.text():
            gap = float(theme.px(6))
            draw_text(p, QRectF(box + gap, 0.0, rect.width() - box - gap, rect.height()), self.text(),
                      theme.qcolor("text" if enabled else "text.disabled"), Qt.AlignLeft, ui_font())
        p.end()


# ---------------------------------------------------------------------------
# 下拉与分段按钮
# ---------------------------------------------------------------------------
class EnumField(_Interactive, QWidget):
    """下拉选择。items 是 [(id, 文字, 说明, 图标)]，也可以是返回这种列表的函数（每次打开时重新取）。"""

    currentChanged = Signal(object)

    def __init__(self, items: Any = (), parent: QWidget | None = None, icon_only: bool = False) -> None:
        super().__init__(parent)
        self._items_source = items
        self._icon_only = icon_only
        first = self.items()
        self._current: Any = first[0][0] if first else None
        self._menu: QMenu | None = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def items(self) -> list[tuple]:
        try:
            return normalize_items(self._items_source)
        except Exception:  # noqa: BLE001  动态列表出错时不让界面崩
            log.exception("下拉列表的条目取不出来")
            return []

    def setItems(self, items: Any) -> None:
        self._items_source = items
        self.updateGeometry()
        self.update()

    def _item(self, ident: Any) -> tuple | None:
        for item in self.items():
            if item[0] == ident:
                return item
        return None

    def current(self) -> Any:
        return self._current

    def currentText(self) -> str:
        item = self._item(self._current)
        if item is not None:
            return item[1]
        return "" if self._current is None else str(self._current)

    def setCurrent(self, ident: Any, emit: bool = False) -> None:
        if ident == self._current:
            return
        self._current = ident
        if self._icon_only:
            self.setToolTip(self.currentText())
        self.update()
        if emit:
            self.currentChanged.emit(ident)

    def isPopupVisible(self) -> bool:
        return self._menu is not None

    def showPopup(self) -> QMenu | None:
        """打开选项菜单（不阻塞）。返回菜单对象。"""
        if not self.isEnabled():
            return None
        menu = QMenu(self)
        menu.setAttribute(Qt.WA_DeleteOnClose, True)
        menu.setToolTipsVisible(True)
        active = None
        for ident, text, desc, icon in self.items():
            action = menu.addAction(text)
            if _icon_ok(icon):
                action.setIcon(icons.icon(icon) if not isinstance(icon, QIcon) else icon)
            if desc:
                action.setToolTip(desc)
            action.triggered.connect(partial(self._choose, ident))
            if ident == self._current:
                active = action
        if not self._icon_only:
            menu.setMinimumWidth(self.width())
        menu.aboutToHide.connect(self._menu_closed)
        self._menu = menu
        if active is not None:
            menu.setActiveAction(active)
        begin_ui_edit(self)
        menu.popup(self.mapToGlobal(QPoint(0, self.height())))
        self.update()
        return menu

    def _menu_closed(self) -> None:
        if self._menu is not None:
            self._menu = None
            end_ui_edit(self)
            self.update()

    def _choose(self, ident: Any, *_args: Any) -> None:
        self.setCurrent(ident, emit=True)

    def _cycle(self, delta: int) -> None:
        items = self.items()
        if not items:
            return
        idents = [item[0] for item in items]
        index = idents.index(self._current) if self._current in idents else 0
        index = max(0, min(len(idents) - 1, index + delta))
        self.setCurrent(idents[index], emit=True)

    def sizeHint(self) -> QSize:
        if self._icon_only:
            width = theme.size("icon") + 2 * theme.size("pad") + theme.px(8) + theme.px(4)
            return QSize(int(width), theme.size("widget"))
        fm = QFontMetricsF(ui_font())
        widest = max([fm.horizontalAdvance(item[1]) for item in self.items()] or [0.0])
        if any(_icon_ok(item[3]) for item in self.items()):
            widest += theme.size("icon") + theme.px(6)
        width = widest + 2 * theme.size("pad") + theme.px(8) + theme.px(6)
        return QSize(int(math.ceil(max(width, theme.px(64)))), theme.size("widget"))

    def minimumSizeHint(self) -> QSize:
        return QSize(theme.px(40), theme.size("widget"))

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        radius = float(theme.size("radius"))
        enabled = self.isEnabled()
        open_ = self._menu is not None
        if not enabled:
            bg, fg = theme.qcolor("bg.panel_header"), "text.disabled"
        elif open_:
            bg, fg = theme.qcolor("bg.widget_pressed"), "text"
        elif self._hovered():
            bg, fg = theme.qcolor("bg.widget_hover"), "text"
        else:
            bg, fg = theme.qcolor("bg.widget"), "text"
        p.fillPath(rounded_path(rect, radius, self._corners), bg)
        pad = float(theme.size("pad"))
        arrow = float(theme.px(8))
        draw_chevron(p, QPointF(rect.right() - pad - arrow / 2.0, rect.center().y()), arrow, "down",
                     theme.qcolor("text.dim" if enabled else "text.disabled"))
        x = rect.left() + pad
        item = self._item(self._current)
        if item is not None and _icon_ok(item[3]):
            size = theme.size("icon")
            draw_icon(p, self, item[3], fg, size, QPointF(x + size / 2.0, rect.center().y()), enabled)
            x += size + theme.px(6)
        right = rect.right() - pad - arrow - theme.px(6)
        if not self._icon_only:
            draw_text(p, QRectF(x, rect.top(), max(0.0, right - x), rect.height()), self.currentText(),
                      theme.qcolor(fg), Qt.AlignLeft, ui_font())
        if self._focus_visible():
            draw_focus(p, rect, radius, self._corners)
        p.end()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.showPopup()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        key = event.key()
        if key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter, Qt.Key_F4) or (
                key == Qt.Key_Down and event.modifiers() & Qt.AltModifier):
            self.showPopup()
            event.accept()
            return
        if key in (Qt.Key_Up, Qt.Key_Down):
            self._cycle(-1 if key == Qt.Key_Up else 1)
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: Any) -> None:
        if not (self.isEnabled() and self._focus_visible()):
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta:
            self._cycle(-1 if delta > 0 else 1)
        event.accept()


class SegmentedControl(_Interactive, QWidget):
    """一排互斥按钮。items 是 [(id, 文字, 说明, 图标)]；icon_only 时只显示图标，文字进提示。"""

    currentChanged = Signal(object)

    def __init__(self, items: Any = (), icon_only: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items = normalize_items(items)
        self._current: Any = self._items[0][0] if self._items else None
        self._icon_only = bool(icon_only)
        self._hover_index = -1
        self._pressed_index = -1
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def items(self) -> list[tuple]:
        return list(self._items)

    def setItems(self, items: Any) -> None:
        self._items = normalize_items(items)
        if self._current not in [item[0] for item in self._items]:
            self._current = self._items[0][0] if self._items else None
        self.updateGeometry()
        self.update()

    def current(self) -> Any:
        return self._current

    def setCurrent(self, ident: Any, emit: bool = False) -> None:
        if ident == self._current:
            return
        self._current = ident
        self.update()
        if emit:
            self.currentChanged.emit(ident)

    def _show_icon(self, item: tuple) -> bool:
        return _icon_ok(item[3])

    def _segment_width(self, item: tuple, fm: QFontMetricsF) -> float:
        h = theme.size("widget")
        if self._icon_only and self._show_icon(item):
            return float(h)
        width = fm.horizontalAdvance(item[1]) + 2 * theme.size("pad")
        if self._show_icon(item):
            width += theme.size("icon") + theme.px(6)
        return max(width, float(h))

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(ui_font())
        if not self._items:
            return QSize(theme.size("widget"), theme.size("widget"))
        widest = max(self._segment_width(item, fm) for item in self._items)
        gaps = line_width() * (len(self._items) - 1)
        return QSize(int(math.ceil(widest * len(self._items) + gaps)), theme.size("widget"))

    def minimumSizeHint(self) -> QSize:
        n = max(1, len(self._items))
        return QSize(theme.px(20) * n, theme.size("widget"))

    def segment_rects(self) -> list[QRectF]:
        n = len(self._items)
        if n == 0:
            return []
        gap = int(line_width())
        total = max(0, self.width() - gap * (n - 1))
        rects = []
        x = 0
        for i in range(n):
            w = (total * (i + 1)) // n - (total * i) // n
            rects.append(QRectF(x, 0, w, self.height()))
            x += w + gap
        return rects

    def _index_at(self, pos: QPointF) -> int:
        for i, rect in enumerate(self.segment_rects()):
            if rect.adjusted(0, 0, line_width(), 0).contains(pos):
                return i
        return -1

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        radius = float(theme.size("radius"))
        enabled = self.isEnabled()
        font = ui_font()
        rects = self.segment_rects()
        n = len(rects)
        for i, (rect, item) in enumerate(zip(rects, self._items)):
            corners = 0
            if i == 0:
                corners |= self._corners & CORNERS_LEFT
            if i == n - 1:
                corners |= self._corners & CORNERS_RIGHT
            selected = item[0] == self._current
            hover = enabled and i == self._hover_index and self.underMouse()
            down = enabled and i == self._pressed_index
            if not enabled:
                bg = mix(theme.qcolor("accent.soft"), theme.qcolor("bg.panel_header"), 0.5) if selected \
                    else theme.qcolor("bg.panel_header")
                fg = "text.disabled"
            elif selected:
                bg = theme.qcolor("accent.pressed" if down else "accent.hover" if hover else "accent")
                fg = "text.on_accent"
            else:
                bg = theme.qcolor("bg.widget_pressed" if down else "bg.widget_hover" if hover else "bg.widget")
                fg = "text"
            p.fillPath(rounded_path(rect, radius, corners), bg)
            show_icon = self._show_icon(item)
            if self._icon_only and show_icon:
                draw_icon(p, self, item[3], fg, theme.size("icon"), rect.center(), enabled, selected)
            else:
                fm = QFontMetricsF(font)
                pad = float(theme.px(6))
                text_w = fm.horizontalAdvance(item[1])
                icon_px = theme.size("icon")
                if show_icon and text_w + icon_px + theme.px(6) + 2 * pad > rect.width():
                    show_icon = False      # 放不下时只留文字
                content = text_w + (icon_px + theme.px(6) if show_icon else 0)
                x = rect.left() + max(pad, (rect.width() - content) / 2.0)
                if show_icon:
                    draw_icon(p, self, item[3], fg, icon_px, QPointF(x + icon_px / 2.0, rect.center().y()), enabled,
                              selected)
                    x += icon_px + theme.px(6)
                draw_text(p, QRectF(x, rect.top(), max(0.0, rect.right() - pad - x), rect.height()), item[1],
                          theme.qcolor(fg), Qt.AlignLeft, font)
            if selected and self._focus_visible():
                draw_focus(p, rect, radius, corners)
        p.end()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            index = self._index_at(event.position())
            if index >= 0:
                self._pressed_index = index
                self.setCurrent(self._items[index][0], emit=True)
                self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._pressed_index >= 0:
            self._pressed_index = -1
            self.update()
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        index = self._index_at(event.position())
        if index != self._hover_index:
            self._hover_index = index
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._hover_index = -1
        super().leaveEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        key = event.key()
        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Home, Qt.Key_End) and self._items:
            idents = [item[0] for item in self._items]
            index = idents.index(self._current) if self._current in idents else 0
            if key == Qt.Key_Left:
                index -= 1
            elif key == Qt.Key_Right:
                index += 1
            elif key == Qt.Key_Home:
                index = 0
            else:
                index = len(idents) - 1
            index = max(0, min(len(idents) - 1, index))
            self.setCurrent(idents[index], emit=True)
            event.accept()
            return
        super().keyPressEvent(event)

    def event(self, event: Any) -> bool:
        if event.type() == QEvent.ToolTip:
            index = self._index_at(QPointF(event.pos()))
            if index >= 0:
                _ident, text, desc, _icon = self._items[index]
                parts = []
                if self._icon_only or not text:
                    parts.append(text)
                if desc:
                    parts.append(desc)
                tip = "\n".join(part for part in parts if part) or self.toolTip()
                if tip:
                    QToolTip.showText(event.globalPos(), tip, self, self.segment_rects()[index].toRect())
                    return True
            if not self.toolTip():
                QToolTip.hideText()
                event.ignore()
                return True
        return super().event(event)


# ---------------------------------------------------------------------------
# 颜色
# ---------------------------------------------------------------------------
_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def color_tuple(color: Any, size: int | None = None) -> tuple:
    """把各种颜色写法整理成 0..1 的分量元组（sRGB）。"""
    if isinstance(color, QColor):
        values = [color.redF(), color.greenF(), color.blueF(), color.alphaF()]
        values = values[: size or 3]
    else:
        try:
            values = [_unit(v) for v in color]
        except TypeError:
            values = [1.0, 1.0, 1.0]
    want = size or (len(values) if len(values) in (3, 4) else 3)
    values = (list(values) + [1.0, 1.0, 1.0, 1.0])[:want]
    return tuple(_unit(v) for v in values)


def to_qcolor(color: Any, opaque: bool = False) -> QColor:
    values = color_tuple(color)
    alpha = 1.0 if opaque or len(values) < 4 else values[3]
    return QColor.fromRgbF(values[0], values[1], values[2], alpha)


def color_to_hex(color: Any) -> str:
    values = color_tuple(color)
    text = "#" + "".join("%02X" % int(round(v * 255)) for v in values[:3])
    if len(values) == 4:
        text += "%02X" % int(round(values[3] * 255))
    return text


def parse_color(text: str, size: int = 3) -> tuple | None:
    """认十六进制（#RGB、#RRGGBB、#RRGGBBAA）和用逗号或空格分开的分量（0..1 或 0..255）。"""
    source = unicodedata.normalize("NFKC", str(text)).strip()
    if not source:
        return None
    match = _HEX_RE.match(source)
    if match:
        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        values = [int(digits[i:i + 2], 16) / 255.0 for i in range(0, len(digits), 2)]
        if size == 4 and len(values) == 3:
            values.append(1.0)
        return tuple(values[:size]) if len(values) >= size else tuple(values + [1.0])[:size]
    parts = [part for part in re.split(r"[\s,;()\[\]]+", source) if part]
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) not in (3, 4) or not all(math.isfinite(n) for n in numbers):
        return None
    if any(n > 1.0 for n in numbers[:3]) and all(0.0 <= n <= 255.0 for n in numbers[:3]):
        numbers = [n / 255.0 for n in numbers[:3]] + numbers[3:]
    if size == 4 and len(numbers) == 3:
        numbers.append(1.0)
    return tuple(_unit(n) for n in numbers[:size])


_recent_colors: list[tuple] = []

#: 最近用过的颜色变化时发出。
recent_colors_changed = CoreSignal()


def recent_colors() -> list[tuple]:
    return list(_recent_colors)


def remember_color(color: Any) -> None:
    """把一个颜色记进「最近用过」。"""
    rgb = tuple(round(v, 4) for v in color_tuple(color)[:3])
    for existing in list(_recent_colors):
        if max(abs(a - b) for a, b in zip(existing, rgb)) < 1.0 / 512.0:
            _recent_colors.remove(existing)
    _recent_colors.insert(0, rgb)
    del _recent_colors[max(1, ColorPicker.recent_limit):]
    recent_colors_changed.emit()


def clear_recent_colors() -> None:
    _recent_colors.clear()
    recent_colors_changed.emit()


class ColorButton(_Interactive, QWidget):
    """色块。单击打开取色浮层，拖动取色时颜色实时写回。颜色是 0..1 的 sRGB 三元组（或带透明的四元组）。"""

    colorChanged = Signal(tuple)
    editingStarted = Signal()
    editingFinished = Signal()

    def __init__(self, color: Any = (1.0, 1.0, 1.0), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color_tuple(color)
        self._popover: Popover | None = None
        self._picker: ColorPicker | None = None
        self._down = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def color(self) -> tuple:
        return self._color

    def setColor(self, color: Any, emit: bool = False) -> None:
        value = color_tuple(color, len(self._color))
        if value == self._color:
            return
        self._color = value
        self.update()
        picker = self._picker
        if picker is not None and shiboken6.isValid(picker):
            picker.setColor(value)
        if emit:
            self.colorChanged.emit(value)

    def picker(self) -> "ColorPicker | None":
        """打开着的取色器（没打开时为 None）。"""
        picker = self._picker
        return picker if picker is not None and shiboken6.isValid(picker) else None

    def popover(self) -> "Popover | None":
        pop = self._popover
        return pop if pop is not None and shiboken6.isValid(pop) else None

    def openPicker(self) -> "Popover | None":
        if not self.isEnabled():
            return None
        if self.popover() is not None:
            return self._popover
        pop = Popover(self)
        pop.setAttribute(Qt.WA_DeleteOnClose, True)
        picker = ColorPicker(self._color, pop.body)
        lay = QVBoxLayout(pop.body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(picker)
        picker.set_reference(self._color)
        picker.colorChanged.connect(self._from_picker)
        picker.editingStarted.connect(self.editingStarted)
        picker.editingFinished.connect(self.editingFinished)
        pop.closed.connect(self._picker_closed)
        self._popover = pop
        self._picker = picker
        pop.popup()
        self.update()
        return pop

    def _from_picker(self, color: tuple) -> None:
        value = color_tuple(color, len(self._color))
        if value != self._color:
            self._color = value
            self.update()
            self.colorChanged.emit(value)

    def _picker_closed(self) -> None:
        self._popover = None
        self._picker = None
        if shiboken6.isValid(self):
            self.update()

    def sizeHint(self) -> QSize:
        return QSize(theme.px(64), theme.size("widget"))

    def minimumSizeHint(self) -> QSize:
        return QSize(theme.size("widget"), theme.size("widget"))

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        radius = float(theme.size("radius"))
        path = rounded_path(rect, radius, self._corners)
        enabled = self.isEnabled()
        values = self._color
        if len(values) == 4 and values[3] < 1.0:
            half = QRectF(rect.left(), rect.top(), math.floor(rect.width() / 2.0), rect.height())
            draw_checker(p, path, rect)
            p.save()
            p.setClipRect(half)
            p.fillPath(path, to_qcolor(values, opaque=True))
            p.restore()
            p.save()
            p.setClipRect(QRectF(half.right(), rect.top(), rect.width() - half.width(), rect.height()))
            p.fillPath(path, to_qcolor(values))
            p.restore()
        else:
            p.fillPath(path, to_qcolor(values, opaque=True))
        if not enabled:
            p.fillPath(path, theme.qcolor("bg.panel_header", 0.6))
        if self.popover() is not None:
            draw_outline(p, rect, radius, theme.qcolor("line.focus"), self._corners)
        elif self._focus_visible():
            draw_focus(p, rect, radius, self._corners)
        elif self._hovered():
            draw_outline(p, rect, radius, theme.qcolor("text.faint"), self._corners)
        else:
            draw_outline(p, rect, radius, theme.qcolor("line"), self._corners)
        p.end()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            self._down = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self._down:
            self._down = False
            if self.rect().contains(event.position().toPoint()):
                self.openPicker()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.openPicker()
            event.accept()
            return
        super().keyPressEvent(event)

    def event(self, event: Any) -> bool:
        if event.type() == QEvent.ToolTip:
            tip = self.toolTip()
            text = color_to_hex(self._color) + ("\n" + tip if tip else "")
            QToolTip.showText(event.globalPos(), text, self)
            return True
        return super().event(event)


class ColorPicker(QWidget):
    """取色器：色相环加中间的饱和度/明度方块，十六进制输入，RGB 与 HSV 数值，最近用过的颜色。

    可以直接嵌进面板，也是 ColorButton 浮层里的内容。颜色是 0..1 的 sRGB 分量元组。
    一次拖动、一次输入算一次编辑（editingStarted / editingFinished），便于撤销分组。
    """

    colorChanged = Signal(tuple)
    editingStarted = Signal()
    editingFinished = Signal()

    #: 最近用过的颜色最多记几个。
    recent_limit = 10

    def __init__(self, color: Any = (1.0, 1.0, 1.0), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rgb = color_tuple(color)
        h, s, v = colorsys.rgb_to_hsv(*self._rgb[:3])
        self._hsv = [h, s, v]
        self._reference = self._rgb
        self._mode = "RGB"
        self._edit_depth = 0
        self._edit_start = self._rgb

        gap = theme.size("gap")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(gap)

        self.wheel = _HueWheel(self)
        lay.addWidget(self.wheel)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(gap)
        self.compare = _CompareSwatch(self)
        self.hex_field = TextField("", "#RRGGBB")
        self.hex_field.setToolTip("十六进制颜色")
        self.hex_field.textCommitted.connect(self._on_hex)
        row.addWidget(self.compare)
        row.addWidget(self.hex_field, 1)
        lay.addLayout(row)

        self.mode_switch = SegmentedControl([("RGB", "RGB", "按红、绿、蓝三个分量调整"),
                                             ("HSV", "HSV", "按色相、饱和度、明度调整")])
        self.mode_switch.currentChanged.connect(self.set_mode)
        lay.addWidget(self.mode_switch)

        fields_box = QVBoxLayout()
        fields_box.setContentsMargins(0, 0, 0, 0)
        fields_box.setSpacing(int(line_width()))
        self.fields: list[NumberField] = []
        count = len(self._rgb)
        for index in range(count):
            field = NumberField(0.0, 0.0, 1.0, precision=3, slider=True)
            field.setStep(0.01)
            field.valueChanged.connect(partial(self._on_field, index))
            field.editingStarted.connect(self._begin_edit)
            field.editingFinished.connect(self._end_edit)
            if count > 1:
                field.set_corners(CORNERS_TOP if index == 0 else CORNERS_BOTTOM if index == count - 1 else 0)
            fields_box.addWidget(field)
            self.fields.append(field)
        lay.addLayout(fields_box)

        self.recent = _RecentSwatches(self)
        lay.addWidget(self.recent)
        self._sync()

    # ---- 对外 ----
    def color(self) -> tuple:
        return self._rgb

    def setColor(self, color: Any, emit: bool = False) -> None:
        value = color_tuple(color, len(self._rgb))
        if max(abs(a - b) for a, b in zip(value, self._rgb)) < 1e-7:
            return
        self._set_rgb(value, emit)

    def hsv(self) -> tuple[float, float, float]:
        return tuple(self._hsv)

    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        mode = "HSV" if str(mode).upper() == "HSV" else "RGB"
        if mode == self._mode:
            return
        self._mode = mode
        self.mode_switch.setCurrent(mode)
        self._sync()

    def set_reference(self, color: Any) -> None:
        """对比色块左半边显示的「原来的颜色」。"""
        self._reference = color_tuple(color, len(self._rgb))
        self.compare.update()

    def reference(self) -> tuple:
        return self._reference

    def revert(self) -> None:
        """恢复到原来的颜色。"""
        self._begin_edit()
        self._set_rgb(self._reference, True)
        self._end_edit()

    def apply_color(self, color: Any) -> None:
        """当作一次完整的编辑设置颜色（最近颜色、十六进制输入都走这里）。"""
        self._begin_edit()
        self._set_rgb(color_tuple(color, len(self._rgb)), True)
        self._end_edit()

    # ---- 内部 ----
    def _begin_edit(self) -> None:
        if self._edit_depth == 0:
            self._edit_start = self._rgb
            begin_ui_edit(self)
            self.editingStarted.emit()
        self._edit_depth += 1

    def _end_edit(self) -> None:
        if self._edit_depth == 0:
            return
        self._edit_depth -= 1
        if self._edit_depth == 0:
            if self._rgb != self._edit_start:
                remember_color(self._rgb)
            self.editingFinished.emit()
            end_ui_edit(self)

    def _set_hsv(self, h: float, s: float, v: float, emit: bool = True) -> None:
        h = h % 1.0
        s = max(0.0, min(1.0, s))
        v = max(0.0, min(1.0, v))
        self._hsv = [h, s, v]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        self._rgb = (r, g, b) + tuple(self._rgb[3:])
        self._sync()
        if emit:
            self.colorChanged.emit(self._rgb)

    def _set_rgb(self, rgb: tuple, emit: bool = True) -> None:
        rgb = color_tuple(rgb, len(self._rgb))
        h, s, v = colorsys.rgb_to_hsv(*rgb[:3])
        if s <= 1e-6 or v <= 1e-6:
            h = self._hsv[0]      # 灰色没有色相，保留原来的，避免色相标记乱跳
        if v <= 1e-6:
            s = self._hsv[1]
        self._hsv = [h, s, v]
        self._rgb = rgb
        self._sync()
        if emit:
            self.colorChanged.emit(self._rgb)

    def _sync(self) -> None:
        labels = ("R", "G", "B") if self._mode == "RGB" else ("H", "S", "V")
        values = list(self._rgb[:3]) if self._mode == "RGB" else list(self._hsv)
        for index, field in enumerate(self.fields):
            if index < 3:
                field.setLabel(labels[index])
                field.setValue(values[index])
            else:
                field.setLabel("A")
                field.setValue(self._rgb[3])
        if not self.hex_field.hasFocus():
            self.hex_field.setText(color_to_hex(self._rgb[:3]))
        self.wheel.update()
        self.compare.update()

    def _on_field(self, index: int, value: float) -> None:
        if index == 3 or self._mode == "RGB":
            rgb = list(self._rgb)
            rgb[index] = value
            self._set_rgb(tuple(rgb), True)
        else:
            hsv = list(self._hsv)
            hsv[index] = value
            self._set_hsv(*hsv)

    def _on_hex(self, text: str) -> None:
        parsed = parse_color(text, 3)
        if parsed is None:
            self.hex_field.setText(color_to_hex(self._rgb[:3]))
            return
        self.apply_color(tuple(parsed) + tuple(self._rgb[3:]))


class _HueWheel(QWidget):
    """色相环加饱和度/明度方块。"""

    #: 推荐边长（逻辑像素）。
    preferred_side = 196

    def __init__(self, picker: ColorPicker) -> None:
        super().__init__(picker)
        self._picker = picker
        self._drag: str | None = None
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:
        side = theme.px(self.preferred_side)
        return QSize(side, side)

    def minimumSizeHint(self) -> QSize:
        side = theme.px(120)
        return QSize(side, side)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(theme.px(120), min(width, theme.px(self.preferred_side)))

    def geometry_parts(self) -> tuple[QPointF, float, float, QRectF]:
        """(圆心, 外半径, 内半径, 方块)"""
        side = float(min(self.width(), self.height()))
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        outer = side / 2.0 - line_width()
        ring = max(float(theme.px(12)), side * 0.075)
        inner = outer - ring
        half = inner * math.sqrt(0.5) - theme.px(6)
        square = QRectF(round(center.x() - half), round(center.y() - half), round(2 * half), round(2 * half))
        return center, outer, inner, square

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        center, outer, inner, square = self.geometry_parts()
        h, s, v = self._picker.hsv()
        enabled = self.isEnabled()
        # 色相环：红色在正上方，顺时针依次是黄、绿、青、蓝、品红
        gradient = QConicalGradient(center, 90.0)
        stops = 36
        for i in range(stops + 1):
            t = i / stops
            gradient.setColorAt(t, QColor.fromHsvF((1.0 - t) % 1.0, 1.0, 1.0))
        ring = QPainterPath()
        ring.addEllipse(center, outer, outer)
        ring.addEllipse(center, inner, inner)
        p.fillPath(ring, gradient)
        # 饱和度（横向）与明度（纵向）方块
        radius = float(theme.size("radius.small"))
        box = rounded_path(square, radius)
        horizontal = QLinearGradient(square.topLeft(), square.topRight())
        horizontal.setColorAt(0.0, QColor.fromRgbF(1.0, 1.0, 1.0))
        horizontal.setColorAt(1.0, QColor.fromHsvF(h, 1.0, 1.0))
        p.fillPath(box, horizontal)
        vertical = QLinearGradient(square.topLeft(), square.bottomLeft())
        vertical.setColorAt(0.0, QColor.fromRgbF(0.0, 0.0, 0.0, 0.0))
        vertical.setColorAt(1.0, QColor.fromRgbF(0.0, 0.0, 0.0, 1.0))
        p.fillPath(box, vertical)
        # 标记
        mid = (outer + inner) / 2.0
        angle = math.radians(90.0 - h * 360.0)
        hue_pos = QPointF(center.x() + math.cos(angle) * mid, center.y() - math.sin(angle) * mid)
        marker = (outer - inner) / 2.0 + line_width()
        self._marker(p, hue_pos, marker, QColor.fromHsvF(h, 1.0, 1.0))
        sv_pos = QPointF(square.left() + s * square.width(), square.top() + (1.0 - v) * square.height())
        self._marker(p, sv_pos, float(theme.px(6)), to_qcolor(self._picker.color(), opaque=True))
        if not enabled:
            p.setPen(Qt.NoPen)
            p.setBrush(theme.qcolor("bg.panel", 0.6))
            p.drawRect(QRectF(self.rect()))
        p.end()

    def _marker(self, p: QPainter, pos: QPointF, radius: float, fill: QColor) -> None:
        lw = line_width()
        p.save()
        p.setBrush(fill)
        p.setPen(QPen(theme.qcolor("line", 0.85), lw * 3.0))
        p.drawEllipse(pos, radius, radius)
        p.setPen(QPen(theme.qcolor("text.on_accent"), lw * 1.5))
        p.drawEllipse(pos, radius, radius)
        p.restore()

    def _zone(self, pos: QPointF) -> str | None:
        center, outer, inner, square = self.geometry_parts()
        dist = math.hypot(pos.x() - center.x(), pos.y() - center.y())
        tol = float(theme.px(4))
        if inner - tol <= dist <= outer + tol:
            return "ring"
        if square.adjusted(-tol, -tol, tol, tol).contains(pos):
            return "square"
        return None

    def _apply(self, pos: QPointF) -> None:
        center, _outer, _inner, square = self.geometry_parts()
        h, s, v = self._picker.hsv()
        if self._drag == "ring":
            angle = math.degrees(math.atan2(-(pos.y() - center.y()), pos.x() - center.x()))
            h = ((90.0 - angle) / 360.0) % 1.0
        elif self._drag == "square":
            s = max(0.0, min(1.0, (pos.x() - square.left()) / max(1.0, square.width())))
            v = max(0.0, min(1.0, 1.0 - (pos.y() - square.top()) / max(1.0, square.height())))
        self._picker._set_hsv(h, s, v)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() != Qt.LeftButton or not self.isEnabled():
            super().mousePressEvent(event)
            return
        zone = self._zone(event.position())
        if zone is None:
            event.ignore()
            return
        self._drag = zone
        self._picker._begin_edit()
        self._apply(event.position())
        event.accept()

    def mouseMoveEvent(self, event: Any) -> None:
        if self._drag is not None and event.buttons() & Qt.LeftButton:
            self._apply(event.position())
            event.accept()
            return
        self.setCursor(Qt.CrossCursor if self._zone(event.position()) else Qt.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self._drag is not None:
            self._drag = None
            self._picker._end_edit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _CompareSwatch(_Interactive, QWidget):
    """对比色块：左半边是原来的颜色（单击恢复），右半边是现在的颜色。"""

    def __init__(self, picker: ColorPicker) -> None:
        super().__init__(picker)
        self._picker = picker
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(theme.px(48), theme.size("widget"))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        radius = float(theme.size("radius"))
        path = rounded_path(rect, radius)
        half = math.floor(rect.width() / 2.0)
        p.save()
        p.setClipRect(QRectF(rect.left(), rect.top(), half, rect.height()))
        p.fillPath(path, to_qcolor(self._picker.reference(), opaque=True))
        p.restore()
        p.save()
        p.setClipRect(QRectF(rect.left() + half, rect.top(), rect.width() - half, rect.height()))
        p.fillPath(path, to_qcolor(self._picker.color(), opaque=True))
        p.restore()
        hover_left = self._hovered() and self.mapFromGlobal(QCursor.pos()).x() < half
        draw_outline(p, rect, radius, theme.qcolor("text.faint" if hover_left else "line"))
        p.end()

    def mouseMoveEvent(self, event: Any) -> None:
        self.update()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and event.position().x() < self.width() / 2.0:
            self._picker.revert()
            event.accept()
            return
        super().mousePressEvent(event)

    def event(self, event: Any) -> bool:
        if event.type() == QEvent.ToolTip:
            left = event.pos().x() < self.width() / 2.0
            tip = ("原来的颜色 %s，单击恢复" % color_to_hex(self._picker.reference()[:3])) if left \
                else ("现在的颜色 %s" % color_to_hex(self._picker.color()[:3]))
            QToolTip.showText(event.globalPos(), tip, self)
            return True
        return super().event(event)


class _RecentSwatches(QWidget):
    """最近用过的颜色，一排小色块，单击取用。"""

    def __init__(self, picker: ColorPicker) -> None:
        super().__init__(picker)
        self._picker = picker
        self._hover = -1
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        recent_colors_changed.connect(self._on_recent_changed)

    def _on_recent_changed(self) -> None:
        if shiboken6.isValid(self):
            self.update()

    def sizeHint(self) -> QSize:
        return QSize(theme.px(160), theme.px(18))

    def minimumSizeHint(self) -> QSize:
        return QSize(theme.px(60), theme.px(18))

    def slot_rects(self) -> list[QRectF]:
        count = max(1, ColorPicker.recent_limit)
        gap = theme.size("gap")
        total = max(0, self.width() - gap * (count - 1))
        rects = []
        x = 0
        for i in range(count):
            w = (total * (i + 1)) // count - (total * i) // count
            rects.append(QRectF(x, 0, w, self.height()))
            x += w + gap
        return rects

    def _index_at(self, pos: QPointF) -> int:
        for i, rect in enumerate(self.slot_rects()):
            if rect.contains(pos):
                return i
        return -1

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        radius = float(theme.size("radius.small"))
        colors = recent_colors()
        for i, rect in enumerate(self.slot_rects()):
            path = rounded_path(rect, radius)
            if i < len(colors):
                p.fillPath(path, to_qcolor(colors[i], opaque=True))
                outline = "text.faint" if i == self._hover and self.isEnabled() else "line"
                draw_outline(p, rect, radius, theme.qcolor(outline))
            else:
                p.fillPath(path, theme.qcolor("bg.field"))
        p.end()

    def mouseMoveEvent(self, event: Any) -> None:
        index = self._index_at(event.position())
        if index != self._hover:
            self._hover = index
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._hover = -1
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            index = self._index_at(event.position())
            colors = recent_colors()
            if 0 <= index < len(colors):
                self._picker.apply_color(tuple(colors[index]) + tuple(self._picker.color()[3:]))
            event.accept()
            return
        super().mousePressEvent(event)

    def event(self, event: Any) -> bool:
        if event.type() == QEvent.ToolTip:
            index = self._index_at(QPointF(event.pos()))
            colors = recent_colors()
            if 0 <= index < len(colors):
                QToolTip.showText(event.globalPos(), "最近用过 %s" % color_to_hex(colors[index]), self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)


# ---------------------------------------------------------------------------
# 折叠面板
# ---------------------------------------------------------------------------
class _FoldoutHeader(_Interactive, QWidget):
    def __init__(self, foldout: "Foldout") -> None:
        super().__init__(foldout)
        self._foldout = foldout
        self.setFocusPolicy(Qt.TabFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        lay = QHBoxLayout(self)
        pad = theme.size("pad")
        lay.setContentsMargins(pad, 0, theme.size("gap"), 0)
        lay.setSpacing(0)
        lay.addStretch(1)
        self.extra = QWidget(self)
        lay.addWidget(self.extra, 0, Qt.AlignVCenter)

    def height_px(self) -> int:
        return theme.size("widget") + theme.px(4)

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(ui_font())
        width = theme.size("pad") * 3 + theme.px(8) + fm.horizontalAdvance(self._foldout.title())
        return QSize(int(math.ceil(width)), self.height_px())

    def minimumSizeHint(self) -> QSize:
        return QSize(theme.px(60), self.height_px())

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        foldout = self._foldout
        opened = foldout.is_open()
        radius = float(theme.size("radius"))
        enabled = self.isEnabled()
        hover = self._hovered()
        if foldout.is_flat():
            if hover:
                p.fillPath(rounded_path(rect, radius), theme.qcolor("bg.row_hover"))
        else:
            base = theme.qcolor("bg.panel_header")
            bg = mix(theme.qcolor("bg.widget"), base, 0.35) if hover else base
            p.fillPath(rounded_path(rect, radius, CORNERS_TOP if opened else CORNERS_ALL), bg)
        pad = float(theme.size("pad"))
        tri = float(theme.px(8))
        tri_color = theme.qcolor("text" if hover else "text.dim") if enabled else theme.qcolor("text.disabled")
        draw_triangle(p, QPointF(pad + tri / 2.0, rect.center().y()), tri, "down" if opened else "right", tri_color)
        x = pad + tri + theme.px(6)
        icon = foldout.icon()
        if _icon_ok(icon):
            size = theme.size("icon")
            draw_icon(p, self, icon, "text.dim" if enabled else "text.disabled", size,
                      QPointF(x + size / 2.0, rect.center().y()), enabled)
            x += size + theme.px(6)
        right = rect.width() - theme.size("gap")
        if self.extra.isVisible() and self.extra.width() > 0 and self.extra.layout() is not None \
                and self.extra.layout().count():
            right = self.extra.geometry().left() - theme.size("gap")
        draw_text(p, QRectF(x, rect.top(), max(0.0, right - x), rect.height()), foldout.title(),
                  theme.qcolor("text" if enabled else "text.disabled"), Qt.AlignLeft, ui_font())
        if self._focus_visible():
            draw_focus(p, rect, radius, CORNERS_TOP if opened else CORNERS_ALL)
        p.end()

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self._foldout.set_open(True)
                self._foldout.soloRequested.emit()
            else:
                self._foldout.toggle()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self._foldout.toggle()
            event.accept()
            return
        if event.key() == Qt.Key_Left:
            self._foldout.set_open(False)
            event.accept()
            return
        if event.key() == Qt.Key_Right:
            self._foldout.set_open(True)
            event.accept()
            return
        super().keyPressEvent(event)


class Foldout(QWidget):
    """可折叠面板。标题行整行可点；左侧小三角表示开合。

    body 是内容容器（没有预设布局，可以直接 UILayout(foldout.body, ctx)）；
    header_layout 是标题行右侧的水平布局，可以放小控件；header_widget 是它所在的容器。
    信号 toggled(bool)：展开为 True。Ctrl+单击标题发出 soloRequested（只展开这一个）。
    flat=True 用于面板里套的小节：标题行没有底色。
    """

    toggled = Signal(bool)
    soloRequested = Signal()

    def __init__(self, title: str = "", icon: Any = None, closed: bool = False, parent: QWidget | None = None,
                 *, flat: bool = False) -> None:
        super().__init__(parent)
        self._title = title or ""
        self._icon = icon or None
        self._open = not closed
        self._flat = bool(flat)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = _FoldoutHeader(self)
        self.header_widget = self.header.extra
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.header_layout.setSpacing(theme.size("gap"))
        self.body = QWidget(self)
        pad = theme.size("pad")
        if self._flat:
            self.body.setContentsMargins(theme.size("gap"), theme.size("gap"), 0, theme.size("gap"))
        else:
            self.body.setContentsMargins(pad, pad - theme.size("gap"), pad, pad)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self.body.setVisible(self._open)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.header.setAccessibleName(self._title)

    def title(self) -> str:
        return self._title

    def setTitle(self, title: str) -> None:
        self._title = title or ""
        self.header.setAccessibleName(self._title)
        self.header.updateGeometry()
        self.header.update()

    def icon(self) -> Any:
        return self._icon

    def setIcon(self, icon: Any) -> None:
        self._icon = icon or None
        self.header.update()

    def is_flat(self) -> bool:
        return self._flat

    def is_open(self) -> bool:
        return self._open

    def isOpen(self) -> bool:
        return self._open

    def set_open(self, opened: bool) -> None:
        opened = bool(opened)
        if opened == self._open:
            return
        self._open = opened
        self.body.setVisible(opened)
        self.header.update()
        self.update()
        self.toggled.emit(opened)

    def setOpen(self, opened: bool) -> None:
        self.set_open(opened)

    def toggle(self) -> None:
        self.set_open(not self._open)

    def paintEvent(self, event: Any) -> None:
        if self._flat or not self._open:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        top = self.header.geometry().bottom() + 1
        rect = QRectF(0, top, self.width(), self.height() - top)
        if rect.height() > 0:
            p.fillPath(rounded_path(rect, float(theme.size("radius")), CORNERS_BOTTOM), theme.qcolor("bg.panel"))
        p.end()


# ---------------------------------------------------------------------------
# 文本输入
# ---------------------------------------------------------------------------
class TextField(QLineEdit):
    """单行文本输入。回车或失去焦点时提交（文字有变化才发 textCommitted），Esc 放弃并退出编辑。"""

    textCommitted = Signal(str)

    def __init__(self, text: str = "", placeholder: str = "", parent: QWidget | None = None) -> None:
        super().__init__(str(text or ""), parent)
        self._start_text = self.text()
        self._editing = False
        #: 右键菜单末尾追加条目的钩子 fn(QMenu)。
        self.extra_menu_hook: Callable[[QMenu], None] | None = None
        #: 回车后是否退出编辑。
        self.clear_focus_on_enter = True
        if placeholder:
            self.setPlaceholderText(placeholder)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.editingFinished.connect(self._commit)

    def sizeHint(self) -> QSize:
        return QSize(theme.px(140), theme.size("widget"))

    def minimumSizeHint(self) -> QSize:
        return QSize(theme.px(40), theme.size("widget"))

    def setText(self, text: str) -> None:
        super().setText(str(text or ""))
        if not self.hasFocus():
            self._start_text = self.text()

    def _commit(self) -> None:
        text = self.text()
        if text != self._start_text:
            self._start_text = text
            self.textCommitted.emit(text)

    def focusInEvent(self, event: Any) -> None:
        if not self._editing:
            self._editing = True
            self._start_text = self.text()
            begin_ui_edit(self)
        super().focusInEvent(event)

    def focusOutEvent(self, event: Any) -> None:
        super().focusOutEvent(event)
        if event.reason() == Qt.PopupFocusReason:
            return
        if self._editing:
            self._editing = False
            end_ui_edit(self)

    def keyPressEvent(self, event: Any) -> None:
        key = event.key()
        if key == Qt.Key_Escape:
            if self.text() != self._start_text:
                super().setText(self._start_text)
            self.clearFocus()
            event.accept()
            return
        super().keyPressEvent(event)
        if key in (Qt.Key_Return, Qt.Key_Enter) and self.clear_focus_on_enter:
            self.clearFocus()

    def enterEvent(self, event: Any) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        if self.isEnabled() and self.underMouse() and not self.hasFocus():
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing, True)
            draw_outline(p, QRectF(self.rect()), float(theme.size("radius")), theme.qcolor("line.soft"))
            p.end()

    def contextMenuEvent(self, event: Any) -> None:
        _popup_menu(_text_edit_menu(self, self.extra_menu_hook), event.globalPos())
        event.accept()


class SearchField(TextField):
    """搜索框：左边放大镜，有文字时右边出现清空按钮。Esc 先清空，再按一次退出。"""

    def __init__(self, text: str = "", placeholder: str = "搜索", parent: QWidget | None = None) -> None:
        super().__init__(text, placeholder, parent)
        self.clear_focus_on_enter = False
        side = theme.SIZES["widget"] - theme.SIZES["gap"]
        self._clear_button = IconButton("close", "清空", size=side, parent=self,
                                        icon_size=theme.SIZES["icon"] - theme.SIZES["gap"])
        self._clear_button.setFocusPolicy(Qt.NoFocus)
        self._clear_button.setCursor(Qt.ArrowCursor)
        self._clear_button.clicked.connect(self._clear_text)
        self._clear_button.setVisible(bool(self.text()))
        self.textChanged.connect(self._update_clear)
        self.setTextMargins(theme.size("icon") + theme.px(4), 0, theme.px(side) - theme.px(4), 0)

    def _clear_text(self) -> None:
        self.clear()
        self.setFocus(Qt.OtherFocusReason)

    def _update_clear(self, text: str) -> None:
        self._clear_button.setVisible(bool(text))

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        hint = self._clear_button.sizeHint()
        lw = int(line_width())
        self._clear_button.setGeometry(self.width() - hint.width() - lw - theme.px(1),
                                       (self.height() - hint.height()) // 2, hint.width(), hint.height())

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        p = QPainter(self)
        size = theme.size("icon")
        x = line_width() + theme.px(6) + size / 2.0
        color = "text.dim" if self.hasFocus() else "text.faint"
        if not self.isEnabled():
            color = "text.disabled"
        draw_icon(p, self, "search", color, size, QPointF(x, self.height() / 2.0), self.isEnabled())
        p.end()

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key_Escape and self.text():
            self.clear()
            event.accept()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# 浮层
# ---------------------------------------------------------------------------
class Popover(QWidget):
    """浮层：点外面或按 Esc 关闭。body 是内容容器（没有预设布局），popup() 显示在锚点控件下方。"""

    closed = Signal()
    #: 最小宽度（逻辑像素）。
    min_width = 220

    def __init__(self, anchor_widget: QWidget | None, parent: QWidget | None = None) -> None:
        owner = parent if parent is not None else anchor_widget
        super().__init__(owner, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.anchor = anchor_widget
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoMouseReplay, True)
        lay = QVBoxLayout(self)
        margin = theme.size("pad")
        lay.setContentsMargins(margin, margin, margin, margin)
        lay.setSpacing(0)
        self.body = QWidget(self)
        lay.addWidget(self.body)
        self._shown = False

    def popup(self, pos: QPoint | None = None) -> None:
        self.ensurePolished()
        lay = self.layout()
        lay.activate()
        hint = self.sizeHint()
        anchor_w = self.anchor.width() if self.anchor is not None and shiboken6.isValid(self.anchor) else 0
        width = max(hint.width(), theme.px(self.min_width), anchor_w)
        height = hint.height()
        if lay.hasHeightForWidth():
            height = max(height, lay.totalHeightForWidth(width))
        self.resize(width, height)
        self.move(pos if pos is not None else self._place(width, height))
        if not self._shown:
            self._shown = True
            if self.anchor is not None:
                begin_ui_edit(self.anchor)
        self.show()

    def _place(self, width: int, height: int) -> QPoint:
        anchor = self.anchor
        if anchor is None or not shiboken6.isValid(anchor):
            return QCursor.pos()
        gap = theme.px(2)
        top_left = anchor.mapToGlobal(QPoint(0, 0))
        x = top_left.x()
        y = top_left.y() + anchor.height() + gap
        screen = anchor.screen() or QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            if y + height > area.bottom() + 1 and top_left.y() - height - gap >= area.top():
                y = top_left.y() - height - gap
            x = max(area.left(), min(x, area.right() + 1 - width))
            y = max(area.top(), min(y, area.bottom() + 1 - height))
        return QPoint(x, y)

    def isOpen(self) -> bool:
        return self._shown and self.isVisible()

    def hideEvent(self, event: Any) -> None:
        super().hideEvent(event)
        if self._shown:
            self._shown = False
            if self.anchor is not None:
                end_ui_edit(self.anchor)
            self.closed.emit()

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        radius = float(theme.px(6))
        p.fillPath(rounded_path(rect, radius), theme.qcolor("bg.menu"))
        draw_outline(p, rect, radius, theme.qcolor("line.soft"))
        p.end()


# ---------------------------------------------------------------------------
# 分隔线与文字
# ---------------------------------------------------------------------------
class Separator(QWidget):
    """细分隔线。vertical=True 用在一行控件之间。"""

    def __init__(self, vertical: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._vertical = bool(vertical)
        if self._vertical:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        else:
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def isVertical(self) -> bool:
        return self._vertical

    def sizeHint(self) -> QSize:
        thickness = theme.size("gap") * 2 + int(line_width())
        if self._vertical:
            return QSize(thickness, theme.size("widget"))
        return QSize(theme.px(16), thickness)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        lw = line_width()
        color = theme.qcolor("line.soft")
        if self._vertical:
            inset = float(theme.size("gap"))
            x = math.floor(self.width() / 2.0)
            p.fillRect(QRectF(x, inset, lw, max(0.0, self.height() - 2 * inset)), color)
        else:
            y = math.floor(self.height() / 2.0)
            p.fillRect(QRectF(0, y, self.width(), lw), color)
        p.end()


class Label(QWidget):
    """文字标签。role：None 正文、"dim" 次要、"faint" 更淡的小字、"title" 小标题、"warn" 提醒（橙色）。

    默认单行，放不下时末尾省略并在悬停提示里给出全文；setWordWrap(True) 后自动换行。
    """

    def __init__(self, text: str = "", role: str | None = None, icon: Any = None, parent: QWidget | None = None,
                 *, align: Any = Qt.AlignLeft) -> None:
        super().__init__(parent)
        self._text = str(text or "")
        self._role = role
        self._icon = icon if _icon_ok(icon) or isinstance(icon, QIcon) else None
        self._align = align
        self._wrap = False
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setAccessibleName(self._text)

    # ---- 属性 ----
    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = str(text or "")
        self.setAccessibleName(self._text)
        self.updateGeometry()
        self.update()

    def role(self) -> str | None:
        return self._role

    def setRole(self, role: str | None) -> None:
        self._role = role
        self.updateGeometry()
        self.update()

    def setIcon(self, icon: Any) -> None:
        self._icon = icon if _icon_ok(icon) or isinstance(icon, QIcon) else None
        self.updateGeometry()
        self.update()

    def alignment(self) -> Any:
        return self._align

    def setAlignment(self, align: Any) -> None:
        self._align = align
        self.update()

    def wordWrap(self) -> bool:
        return self._wrap

    def setWordWrap(self, wrap: bool) -> None:
        self._wrap = bool(wrap)
        policy = self.sizePolicy()
        policy.setHeightForWidth(self._wrap)
        policy.setVerticalPolicy(QSizePolicy.Preferred if self._wrap else QSizePolicy.Fixed)
        self.setSizePolicy(policy)
        self.updateGeometry()
        self.update()

    # ---- 尺寸 ----
    def _font(self) -> QFont:
        if self._role == "title":
            return ui_font("font.title", bold=True)
        if self._role == "faint":
            return ui_font("font.small")
        return ui_font()

    def _color(self) -> QColor:
        if not self.isEnabled():
            return theme.qcolor("text.disabled")
        if self._role == "dim":
            return theme.qcolor("text.dim")
        if self._role == "faint":
            return theme.qcolor("text.faint")
        if self._role == "warn":
            return theme.qcolor("warn")
        return theme.qcolor("text")

    def _icon_part(self) -> float:
        return float(theme.size("icon") + theme.px(6)) if self._icon is not None else 0.0

    def sizeHint(self) -> QSize:
        fm = QFontMetricsF(self._font())
        width = fm.horizontalAdvance(self._text) + self._icon_part() + line_width()
        if self._wrap:
            width = min(width, float(theme.px(240)))
            return QSize(int(math.ceil(width)), self.heightForWidth(int(math.ceil(width))))
        return QSize(int(math.ceil(width)), theme.size("widget"))

    def minimumSizeHint(self) -> QSize:
        if self._wrap:
            return QSize(theme.px(40), theme.size("widget"))
        return QSize(int(self._icon_part()), theme.size("widget"))

    def hasHeightForWidth(self) -> bool:
        return self._wrap

    def heightForWidth(self, width: int) -> int:
        if not self._wrap:
            return -1
        fm = QFontMetricsF(self._font())
        available = max(1.0, width - self._icon_part())
        bound = fm.boundingRect(QRectF(0, 0, available, 1e6), _align_int(Qt.AlignLeft, Qt.TextWordWrap), self._text)
        pad = (theme.size("widget") - fm.height()) / 2.0
        return max(theme.size("widget"), int(math.ceil(bound.height() + 2 * pad)))

    def _elided(self) -> bool:
        fm = QFontMetricsF(self._font())
        return not self._wrap and fm.horizontalAdvance(self._text) > self.width() - self._icon_part()

    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        line_h = float(theme.size("widget"))
        font = self._font()
        p.setFont(font)
        color = self._color()
        x = rect.left()
        if self._icon is not None:
            size = theme.size("icon")
            name = "text.disabled" if not self.isEnabled() else ("text.dim" if self._role in ("dim", "faint") else "text")
            draw_icon(p, self, self._icon, name, size, QPointF(x + size / 2.0, min(rect.center().y(), line_h / 2.0)),
                      self.isEnabled())
            x += self._icon_part()
        area = QRectF(x, rect.top(), max(0.0, rect.right() - x), rect.height())
        if self._wrap:
            fm = QFontMetricsF(font)
            pad = (line_h - fm.height()) / 2.0
            p.setPen(color)
            p.drawText(area.adjusted(0, pad, 0, -pad), _align_int(self._align, Qt.AlignTop, Qt.TextWordWrap),
                       self._text)
        else:
            draw_text(p, area, self._text, color, self._align)
        p.end()

    def changeEvent(self, event: Any) -> None:
        if event.type() == QEvent.EnabledChange:
            self.update()
        super().changeEvent(event)

    def event(self, event: Any) -> bool:
        if event.type() == QEvent.ToolTip and not self.toolTip():
            if self._elided():
                QToolTip.showText(event.globalPos(), self._text, self)
            else:
                QToolTip.hideText()
            return True
        return super().event(event)


# ---------------------------------------------------------------------------
# 标签条
# ---------------------------------------------------------------------------
def _has_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if 0x2E80 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF or 0xFF00 <= code <= 0xFFEF:
            return True
    return False


class TabStrip(_Interactive, QWidget):
    """标签条。items 是 [(id, 文字, 说明, 图标)]。

    vertical=True 竖排（侧栏分类、属性编辑器）：中文逐字竖排，其他文字旋转。icon_only 只显示图标，文字进提示。
    信号：currentChanged(id)，tabDoubleClicked(id)，tabContextMenuRequested(id, 全局坐标)。
    """

    currentChanged = Signal(object)
    tabDoubleClicked = Signal(object)
    tabContextMenuRequested = Signal(object, QPoint)

    def __init__(self, items: Any = (), vertical: bool = False, icon_only: bool = False,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items = normalize_items(items)
        self._current: Any = self._items[0][0] if self._items else None
        self._vertical = bool(vertical)
        self._icon_only = bool(icon_only)
        self._hover = -1
        self.setFocusPolicy(Qt.TabFocus)
        self.setMouseTracking(True)
        if self._vertical:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        else:
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def items(self) -> list[tuple]:
        return list(self._items)

    def setItems(self, items: Any) -> None:
        self._items = normalize_items(items)
        if self._current not in [item[0] for item in self._items]:
            self._current = self._items[0][0] if self._items else None
        self.updateGeometry()
        self.update()

    def current(self) -> Any:
        return self._current

    def setCurrent(self, ident: Any, emit: bool = False) -> None:
        if ident == self._current:
            return
        self._current = ident
        self.update()
        if emit:
            self.currentChanged.emit(ident)

    def isVertical(self) -> bool:
        return self._vertical

    # ---- 几何 ----
    def _thickness(self) -> int:
        if self._icon_only:
            return theme.size("widget") + theme.px(4)
        return theme.size("widget")

    def _extent(self, item: tuple, fm: QFontMetricsF) -> float:
        pad = float(theme.size("pad"))
        if self._icon_only and _icon_ok(item[3]):
            return float(self._thickness())
        if self._vertical:
            if _has_cjk(item[1]):
                length = len(item[1]) * fm.height()
            else:
                length = fm.horizontalAdvance(item[1])
            return length + 2 * pad
        width = fm.horizontalAdvance(item[1]) + 2 * pad
        if _icon_ok(item[3]):
            width += theme.size("icon") + theme.px(6)
        return width

    def tab_rects(self) -> list[QRectF]:
        fm = QFontMetricsF(ui_font())
        gap = float(theme.px(2))
        pos = 0.0
        rects = []
        thick = float(self.width() if self._vertical else self.height())
        for item in self._items:
            length = math.ceil(self._extent(item, fm))
            if self._vertical:
                rects.append(QRectF(0.0, pos, thick, length))
            else:
                rects.append(QRectF(pos, 0.0, length, thick))
            pos += length + gap
        return rects

    def sizeHint(self) -> QSize:
        rects = self.tab_rects()
        total = int(math.ceil(rects[-1].bottom() if self._vertical else rects[-1].right())) if rects else 0
        thick = self._thickness()
        return QSize(thick, total) if self._vertical else QSize(total, thick)

    def minimumSizeHint(self) -> QSize:
        thick = self._thickness()
        return QSize(thick, thick)

    def _index_at(self, pos: QPointF) -> int:
        for i, rect in enumerate(self.tab_rects()):
            if rect.contains(pos):
                return i
        return -1

    # ---- 绘制 ----
    def paintEvent(self, event: Any) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        font = ui_font()
        p.setFont(font)
        fm = QFontMetricsF(font)
        radius = float(theme.size("radius"))
        enabled = self.isEnabled()
        for i, (rect, item) in enumerate(zip(self.tab_rects(), self._items)):
            selected = item[0] == self._current
            hover = enabled and i == self._hover and self.underMouse()
            if selected:
                p.fillPath(rounded_path(rect, radius), theme.qcolor("bg.widget"))
                bar = float(theme.px(2))
                accent = theme.qcolor("accent" if enabled else "text.disabled")
                if self._vertical:
                    p.fillPath(rounded_path(QRectF(rect.left(), rect.top() + radius, bar, rect.height() - 2 * radius),
                                            bar / 2.0), accent)
                else:
                    p.fillPath(rounded_path(QRectF(rect.left() + radius, rect.bottom() - bar,
                                                   rect.width() - 2 * radius, bar), bar / 2.0), accent)
            elif hover:
                p.fillPath(rounded_path(rect, radius), theme.qcolor("bg.row_hover"))
            fg = "text.disabled" if not enabled else ("text" if selected or hover else "text.dim")
            color = theme.qcolor(fg)
            if self._icon_only and _icon_ok(item[3]):
                draw_icon(p, self, item[3], fg, theme.size("icon"), rect.center(), enabled)
            elif self._vertical:
                self._paint_vertical_text(p, rect, item[1], color, fm)
            else:
                pad = float(theme.size("pad"))
                x = rect.left() + pad
                if _icon_ok(item[3]):
                    size = theme.size("icon")
                    draw_icon(p, self, item[3], fg, size, QPointF(x + size / 2.0, rect.center().y()), enabled)
                    x += size + theme.px(6)
                draw_text(p, QRectF(x, rect.top(), rect.right() - pad - x, rect.height()), item[1], color,
                          Qt.AlignLeft)
            if selected and self._focus_visible():
                draw_focus(p, rect, radius)
        p.end()

    def _paint_vertical_text(self, p: QPainter, rect: QRectF, text: str, color: QColor, fm: QFontMetricsF) -> None:
        p.setPen(color)
        if _has_cjk(text):
            step = fm.height()
            y = rect.top() + (rect.height() - step * len(text)) / 2.0
            for ch in text:
                p.drawText(QRectF(rect.left(), y, rect.width(), step), _align_int(Qt.AlignHCenter, Qt.AlignVCenter), ch)
                y += step
            return
        p.save()
        p.translate(rect.center())
        p.rotate(90.0)
        box = QRectF(-rect.height() / 2.0, -rect.width() / 2.0, rect.height(), rect.width())
        p.drawText(box, _align_int(Qt.AlignHCenter, Qt.AlignVCenter), text)
        p.restore()

    # ---- 交互 ----
    def mousePressEvent(self, event: Any) -> None:
        index = self._index_at(event.position())
        if index < 0:
            super().mousePressEvent(event)
            return
        ident = self._items[index][0]
        if event.button() == Qt.LeftButton:
            self.setCurrent(ident, emit=True)
            event.accept()
            return
        if event.button() == Qt.RightButton:
            self.tabContextMenuRequested.emit(ident, event.globalPosition().toPoint())
            self._suppress_context = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        index = self._index_at(event.position())
        if index >= 0 and event.button() == Qt.LeftButton:
            self.tabDoubleClicked.emit(self._items[index][0])
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        index = self._index_at(event.position())
        if index != self._hover:
            self._hover = index
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._hover = -1
        super().leaveEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        key = event.key()
        back = (Qt.Key_Up,) if self._vertical else (Qt.Key_Left,)
        forward = (Qt.Key_Down,) if self._vertical else (Qt.Key_Right,)
        if key in back + forward and self._items:
            idents = [item[0] for item in self._items]
            index = idents.index(self._current) if self._current in idents else 0
            index = max(0, min(len(idents) - 1, index + (1 if key in forward else -1)))
            self.setCurrent(idents[index], emit=True)
            event.accept()
            return
        super().keyPressEvent(event)

    def event(self, event: Any) -> bool:
        if event.type() == QEvent.ToolTip:
            index = self._index_at(QPointF(event.pos()))
            if index >= 0:
                _ident, text, desc, _icon = self._items[index]
                parts = [text] if self._icon_only else []
                if desc:
                    parts.append(desc)
                tip = "\n".join(part for part in parts if part)
                if tip:
                    QToolTip.showText(event.globalPos(), tip, self, self.tab_rects()[index].toRect())
                    return True
            QToolTip.hideText()
            return True
        return super().event(event)


# ---------------------------------------------------------------------------
# 菜单
# ---------------------------------------------------------------------------
def build_menu(menu_cls_or_idname: Any, ctx: Any, parent: QWidget | None = None) -> QMenu:
    """把 registry.Menu 变成 QMenu。操作项右侧显示当前快捷键，不可用的置灰；每次打开前重新判断是否可用。"""
    from .layout import MenuLayout  # 布局语句依赖本模块，这里延迟导入

    if isinstance(menu_cls_or_idname, str):
        idname = menu_cls_or_idname
        cls = registry.menu(idname)
    else:
        cls = menu_cls_or_idname
        idname = getattr(cls, "idname", "") or getattr(cls, "__name__", "")
    qmenu = QMenu(parent)
    qmenu.setToolTipsVisible(True)
    if cls is None:
        log.warning("没有名为 %s 的菜单", idname)
        qmenu.setTitle(idname)
        return qmenu
    qmenu.setTitle(getattr(cls, "label", "") or idname)
    layout = MenuLayout(qmenu, ctx)
    extras = registry.menu_extras(idname)
    for draw_fn in [f for f in extras if getattr(f, "_splender_prepend", False)]:
        try:
            draw_fn(layout, ctx)
        except Exception:  # noqa: BLE001
            log.exception("菜单 %s 的追加内容出错", idname)
    try:
        cls().draw(layout, ctx)
    except Exception:  # noqa: BLE001  一个菜单出错不该拖垮整个窗口
        log.exception("菜单 %s 生成出错", idname)
    for draw_fn in [f for f in extras if not getattr(f, "_splender_prepend", False)]:
        try:
            draw_fn(layout, ctx)
        except Exception:  # noqa: BLE001
            log.exception("菜单 %s 的追加内容出错", idname)
    qmenu.aboutToShow.connect(layout.refresh_polls)
    qmenu._splender_layout = layout  # 让布局对象和菜单同寿
    return qmenu


# ---------------------------------------------------------------------------
# 主题变化：已有控件按新尺寸重新排版
# ---------------------------------------------------------------------------
_THEMED = (NumberField, _ButtonBase, CheckBox, EnumField, SegmentedControl, ColorButton, _HueWheel, _CompareSwatch,
           _RecentSwatches, _FoldoutHeader, Foldout, TextField, Separator, Label, TabStrip)


def _on_theme_changed() -> None:
    _font_cache.clear()
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.allWidgets():
        if isinstance(widget, _THEMED):
            widget.updateGeometry()
            widget.update()


theme.changed.connect(_on_theme_changed)
