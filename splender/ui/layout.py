"""布局语句：面板、标题栏、菜单的内容都用它排。

    layout = UILayout(parent_widget, ctx)            # 把自己装进 parent_widget
    layout.use_property_split = True                 # 标签在左、控件在右（面板里由面板宿主打开）
    col = layout.column(align=True)
    col.prop(brush, "size")
    col.prop(brush, "flow")
    row = layout.row(align=True)
    row.operator("layer.add_paint", icon="add")

规则摘要：
- prop 按属性类型自动选控件，并与数据双向绑定：控件改数据，数据的 changed(name) 改控件，互不触发；
  控件销毁后绑定自动断开。
- 数据对象 undoable 为真且 ctx.history 存在时，一次编辑（拖动从按下到松开）结束后向历史推一步。
- 右键菜单：恢复默认值、复制数值、粘贴数值。提示文字取属性说明。
- operator 按钮文字取操作的 label，提示取说明加快捷键；创建时和 refresh_operator_states() 时判断 poll。
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
import weakref
from functools import partial
from typing import Any, Callable

import shiboken6
from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QPainter
from PySide6.QtWidgets import (QApplication, QBoxLayout, QFileDialog, QHBoxLayout, QLayout, QMenu, QSizePolicy,
                               QWidget, QWidgetAction, QWidgetItem)

from ..core import ops, registry
from ..core.props import resolve_path
from ..core.signals import Signal as CoreSignal
from . import icons, theme
from .widgets import (CORNERS_ALL, CORNERS_BOTTOM, CORNERS_LEFT, CORNERS_RIGHT, CORNERS_TOP, Button, CheckBox,
                      ColorButton, EnumField, IconButton, Label, NumberField, Popover, SegmentedControl, TextField,
                      Toggle, _icon_ok, build_menu, color_to_hex, draw_outline, evaluate_number, line_width,
                      parse_color, rounded_path)

log = logging.getLogger("splender.ui")

_COMPONENT_LABELS = ("X", "Y", "Z", "W")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001
        return a is b


def _descriptor(target: Any, attr: str) -> Any:
    properties = getattr(type(target), "properties", None)
    if callable(properties):
        found = properties().get(attr)
        if found is not None:
            return found
    raise AttributeError("%s 没有名为 %s 的属性" % (type(target).__name__, attr))


def _unit_of(prop: Any) -> str:
    unit = getattr(prop, "unit", "") or ""
    if not unit:
        unit = {"PIXEL": "px", "ANGLE": "°"}.get(getattr(prop, "subtype", ""), "")
    return unit


def _history_of(ctx: Any) -> Any:
    if ctx is None:
        return None
    try:
        return getattr(ctx, "history", None)
    except Exception:  # noqa: BLE001
        return None


def _shortcut(idname: str, props: dict, ctx: Any) -> str:
    if ctx is None:
        return ""
    try:
        keyconfig = getattr(ctx, "keyconfig", None)
    except Exception:  # noqa: BLE001
        keyconfig = None
    if keyconfig is None:
        return ""
    try:
        return keyconfig.shortcut_for(idname, dict(props) if props else None)
    except Exception:  # noqa: BLE001
        return ""


def operator_tooltip(idname: str, props: dict | None = None, ctx: Any = None) -> str:
    """操作按钮的提示：说明（没有说明时用名称）加当前快捷键。"""
    cls = ops.get(idname)
    parts = []
    if cls is not None:
        parts.append(getattr(cls, "description", "") or getattr(cls, "label", "") or idname)
    shortcut = _shortcut(idname, props or {}, ctx)
    if shortcut:
        parts.append("快捷键  " + shortcut)
    return "\n".join(part for part in parts if part)


def _call_operator(idname: str, props: dict, ctx: Any, *_args: Any) -> str:
    return ops.call(idname, ctx, **props)


# ---------------------------------------------------------------------------
# 操作按钮的可用状态
# ---------------------------------------------------------------------------
_operator_buttons: "weakref.WeakSet[OperatorButton]" = weakref.WeakSet()
_watched_histories: "weakref.WeakSet[Any]" = weakref.WeakSet()
_poll_scheduled = False


def refresh_operator_states() -> None:
    """重新判断所有操作按钮能不能用。窗口管理器在合并后的 notify 里调用它。"""
    global _poll_scheduled
    _poll_scheduled = False
    for button in list(_operator_buttons):
        if shiboken6.isValid(button):
            button.update_poll()


def schedule_operator_refresh(*_args: Any) -> None:
    """在下一轮事件循环里刷新按钮状态（同一轮里的多次请求合并成一次）。"""
    global _poll_scheduled
    if _poll_scheduled or QApplication.instance() is None:
        return
    _poll_scheduled = True
    QTimer.singleShot(0, refresh_operator_states)


ops.executed.connect(schedule_operator_refresh)


def _watch_history(ctx: Any) -> None:
    history = _history_of(ctx)
    signal = getattr(history, "changed", None)
    if history is None or not isinstance(signal, CoreSignal):
        return
    try:
        if history in _watched_histories:
            return
        _watched_histories.add(history)
    except TypeError:
        return
    signal.connect(schedule_operator_refresh)


class OperatorButton(Button):
    """执行一个操作的按钮。poll 为假时置灰。"""

    def __init__(self, idname: str, props: dict | None, ctx: Any, text: str = "", icon: Any = None,
                 role: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(text, icon, role, parent)
        self.idname = idname
        self.op_props = dict(props or {})
        self.ctx = ctx
        self._layout_enabled = True
        self.clicked.connect(self.run)
        tip = operator_tooltip(idname, self.op_props, ctx)
        if tip:
            self.setToolTip(tip)
        if not text:
            self.setAccessibleName(getattr(ops.get(idname), "label", "") or idname)
        _operator_buttons.add(self)
        _watch_history(ctx)
        self.update_poll()

    def set_layout_enabled(self, enabled: bool) -> None:
        self._layout_enabled = bool(enabled)
        self.update_poll()

    def update_poll(self) -> None:
        ok = self._layout_enabled and ops.poll(self.idname, self.ctx)
        if ok != self.isEnabled():
            self.setEnabled(ok)

    def run(self, *_args: Any) -> str:
        if not ops.poll(self.idname, self.ctx):
            self.update_poll()
            return ops.CANCELLED
        return ops.call(self.idname, self.ctx, **self.op_props)


# ---------------------------------------------------------------------------
# 数据与控件的双向绑定
# ---------------------------------------------------------------------------
_BOOL_WORDS = {"1": True, "true": True, "yes": True, "on": True, "是": True, "开": True, "真": True,
               "0": False, "false": False, "no": False, "off": False, "否": False, "关": False, "假": False}


class _Binding:
    """一个属性与它的控件之间的双向绑定。

    数据 → 控件：连到数据组的 changed 信号（绑定方法，弱引用）。控件 → 数据：连控件的信号。
    控件销毁时（destroyed 或已失效）自动断开，不会再去碰已删除的 Qt 对象。
    """

    def __init__(self, layout: Any, root: Any, target: Any, attr: str, prop: Any) -> None:
        self.ctx = layout.ctx
        self.root = root
        self.target = target
        self.attr = attr
        self.prop = prop
        self.undoable = bool(getattr(root, "undoable", False) or getattr(target, "undoable", False))
        self.kind = ""
        self._refs: list[weakref.ref] = []
        self._alive = True
        self._depth = 0
        self._start: Any = None
        signal = getattr(target, "changed", None)
        self._signal = signal if isinstance(signal, CoreSignal) else None
        if self._signal is not None:
            self._signal.connect(self._on_data_changed)
        layout._bindings.append(self)

    # ---- 生命周期 ----
    @property
    def alive(self) -> bool:
        return self._alive

    def own(self, *objects: Any) -> None:
        for obj in objects:
            self._refs.append(weakref.ref(obj))
            store = getattr(obj, "_splender_bindings", None)
            if store is None:
                store = []
                obj._splender_bindings = store
            store.append(self)
            obj.destroyed.connect(self.detach)

    def detach(self, *_args: Any) -> None:
        if not self._alive:
            return
        self._alive = False
        if self._signal is not None:
            try:
                self._signal.disconnect(self._on_data_changed)
            except Exception:  # noqa: BLE001
                pass
            self._signal = None

    def objects(self) -> list | None:
        out = []
        for ref in self._refs:
            obj = ref()
            if obj is None or not shiboken6.isValid(obj):
                return None
            out.append(obj)
        return out

    # ---- 读写 ----
    def read(self) -> Any:
        return getattr(self.target, self.attr)

    def write(self, value: Any) -> None:
        if not self._alive:
            return
        old = self.read()
        try:
            setattr(self.target, self.attr, value)
        except Exception:  # noqa: BLE001
            log.exception("属性 %s 写入失败", self.attr)
            return
        if self._depth == 0:
            new = self.read()
            if not _same(old, new):
                self._push(old, new)

    def begin(self, *_args: Any) -> None:
        """一次编辑开始（拖动按下、进入文字编辑）。"""
        if self._depth == 0:
            self._start = self.read()
        self._depth += 1

    def end(self, *_args: Any) -> None:
        """一次编辑结束：值变了就向历史推一步。"""
        if self._depth == 0:
            return
        self._depth -= 1
        if self._depth == 0:
            start, self._start = self._start, None
            if self._alive:
                end = self.read()
                if not _same(start, end):
                    self._push(start, end)

    def _push(self, old: Any, new: Any) -> None:
        if not self.undoable:
            return
        history = _history_of(self.ctx)
        if history is None:
            return
        target, attr = self.target, self.attr

        def undo() -> None:
            setattr(target, attr, old)

        def redo() -> None:
            setattr(target, attr, new)

        history.push(self.prop.label or attr, undo, redo)

    # ---- 数据变了 ----
    def _on_data_changed(self, name: str) -> None:
        if not self._alive or name != self.attr:
            return
        objects = self.objects()
        if objects is None:
            self.detach()
            return
        try:
            self._apply(objects, self.read())
        except RuntimeError:
            self.detach()

    def _apply(self, objects: list, value: Any) -> None:
        kind = self.kind
        if kind == "number":
            objects[0].setValue(value)
        elif kind == "bool":
            button = objects[0]
            blocked = button.blockSignals(True)
            button.setChecked(bool(value))
            button.blockSignals(blocked)
        elif kind == "enum":
            objects[0].setCurrent(value)
        elif kind == "color":
            objects[0].setColor(value)
        elif kind in ("ramp", "curve"):
            objects[0].setValue(value)
        elif kind == "vector":
            for index, field in enumerate(objects):
                if index < len(value):
                    field.setValue(value[index])
        elif kind == "string":
            field = objects[0]
            if not field.hasFocus():
                field.setText("" if value is None else str(value))
        elif kind == "action_bool":
            action = objects[0]
            blocked = action.blockSignals(True)
            action.setChecked(bool(value))
            action.blockSignals(blocked)
        elif kind == "action_enum":
            for action in objects:
                blocked = action.blockSignals(True)
                action.setChecked(action.data() == value)
                action.blockSignals(blocked)

    # ---- 控件接线 ----
    def bind_number(self, field: NumberField) -> None:
        self.kind = "number"
        self.own(field)
        field.valueChanged.connect(self._number_changed)
        field.editingStarted.connect(self.begin)
        field.editingFinished.connect(self.end)
        field.context_menu_hook = self.show_menu

    def _number_changed(self, value: float) -> None:
        if getattr(self.prop, "kind", "") == "INT":
            value = int(round(value))
        self.write(value)

    def bind_vector(self, fields: list[NumberField]) -> None:
        self.kind = "vector"
        self.own(*fields)
        for index, field in enumerate(fields):
            field.valueChanged.connect(partial(self._vector_changed, index))
            field.editingStarted.connect(self.begin)
            field.editingFinished.connect(self.end)
            field.context_menu_hook = self.show_menu

    def _vector_changed(self, index: int, value: float, *_args: Any) -> None:
        values = list(self.read())
        if index < len(values):
            values[index] = value
            self.write(tuple(values))

    def bind_bool(self, button: Any) -> None:
        self.kind = "bool"
        self.own(button)
        button.setChecked(bool(self.read()))
        button.clicked.connect(self._bool_clicked)
        button.context_menu_hook = self.show_menu

    def _bool_clicked(self, checked: bool = False, *_args: Any) -> None:
        self.write(bool(checked))

    def bind_enum(self, widget: Any) -> None:
        self.kind = "enum"
        self.own(widget)
        widget.setCurrent(self.read())
        widget.currentChanged.connect(self.write)
        widget.context_menu_hook = self.show_menu

    def bind_color(self, button: ColorButton) -> None:
        self.kind = "color"
        self.own(button)
        button.colorChanged.connect(self.write)
        button.editingStarted.connect(self.begin)
        button.editingFinished.connect(self.end)
        button.context_menu_hook = self.show_menu

    def bind_curve(self, widget: Any) -> None:
        self.kind = "curve"
        self.own(widget)
        widget.curveChanged.connect(self.write)
        widget.editingStarted.connect(self.begin)
        widget.editingFinished.connect(self.end)

    def bind_ramp(self, widget: Any) -> None:
        self.kind = "ramp"
        self.own(widget)
        widget.rampChanged.connect(self.write)
        widget.editingStarted.connect(self.begin)
        widget.editingFinished.connect(self.end)

    def bind_string(self, field: TextField) -> None:
        self.kind = "string"
        self.own(field)
        field.textCommitted.connect(self.write)
        field.extra_menu_hook = self.fill_menu

    def bind_action_bool(self, action: QAction) -> None:
        self.kind = "action_bool"
        self.own(action)
        action.setCheckable(True)
        action.setChecked(bool(self.read()))
        action.triggered.connect(self._bool_clicked)

    def bind_action_enum(self, actions: list[QAction]) -> None:
        self.kind = "action_enum"
        self.own(*actions)
        for action in actions:
            action.triggered.connect(partial(self._enum_action, action.data()))

    def _enum_action(self, ident: Any, *_args: Any) -> None:
        self.write(ident)

    # ---- 右键菜单 ----
    def is_default(self) -> bool:
        checker = getattr(self.target, "is_default", None)
        if callable(checker):
            try:
                return bool(checker(self.attr))
            except Exception:  # noqa: BLE001
                return False
        return False

    def reset_default(self, *_args: Any) -> None:
        default = self.prop.default_value(self.target)
        self.write(default)

    def format_value(self, value: Any) -> str:
        kind = getattr(self.prop, "kind", "")
        if kind == "INT":
            return str(int(value))
        if kind == "FLOAT":
            return format(float(value), ".10g")
        if kind == "BOOL":
            return "true" if value else "false"
        if kind == "COLOR":
            return color_to_hex(value)
        if kind == "VECTOR":
            return json.dumps([round(float(v), 6) for v in value])
        return "" if value is None else str(value)

    def parse_value(self, text: str) -> Any:
        """把剪贴板文字换成这个属性能接受的值，换不了返回 None。"""
        kind = getattr(self.prop, "kind", "")
        source = unicodedata.normalize("NFKC", str(text or "")).strip()
        if kind in ("FLOAT", "INT"):
            unit = _unit_of(self.prop).strip()
            if unit and source.lower().endswith(unit.lower()):
                source = source[: -len(unit)].strip()
            value = evaluate_number(source)
            if value is None:
                return None
            return int(round(value)) if kind == "INT" else value
        if kind == "BOOL":
            return _BOOL_WORDS.get(source.lower())
        if kind == "ENUM":
            for ident, label, _desc, _icon in self.prop.items(self.target):
                if source == str(ident) or source == label:
                    return ident
            return None
        if kind == "COLOR":
            return parse_color(source, len(self.read()))
        if kind == "VECTOR":
            parts = [part for part in re.split(r"[\s,;()\[\]]+", source) if part]
            try:
                numbers = [float(part) for part in parts]
            except ValueError:
                return None
            size = len(self.read())
            if len(numbers) != size or not all(math.isfinite(n) for n in numbers):
                return None
            return tuple(numbers)
        if kind == "STRING":
            return str(text or "").strip("\r\n")
        return None

    def copy_value(self, *_args: Any) -> None:
        QApplication.clipboard().setText(self.format_value(self.read()))

    def paste_value(self, *_args: Any) -> bool:
        value = self.parse_value(QApplication.clipboard().text())
        if value is None:
            return False
        self.write(value)
        return True

    def fill_menu(self, menu: QMenu) -> None:
        reset = menu.addAction("恢复默认值", self.reset_default)
        reset.setEnabled(not self.is_default())
        menu.addSeparator()
        menu.addAction("复制数值", self.copy_value)
        paste = menu.addAction("粘贴数值", self.paste_value)
        paste.setEnabled(self.parse_value(QApplication.clipboard().text()) is not None)

    def show_menu(self, global_pos: QPoint) -> QMenu | None:
        objects = self.objects()
        if not objects:
            return None
        menu = QMenu(objects[0] if isinstance(objects[0], QWidget) else None)
        self.fill_menu(menu)
        menu.setAttribute(Qt.WA_DeleteOnClose, True)
        menu.popup(global_pos)
        return menu


# ---------------------------------------------------------------------------
# 布局用的容器
# ---------------------------------------------------------------------------
class _Block(QWidget):
    """子布局的容器。boxed=True 时画一个内凹的框（layout.box()）。"""

    def __init__(self, parent: QWidget | None = None, boxed: bool = False) -> None:
        super().__init__(parent)
        self._boxed = boxed

    def paintEvent(self, event: Any) -> None:
        if not self._boxed:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect())
        radius = float(theme.size("radius"))
        p.fillPath(rounded_path(rect, radius), theme.qcolor("bg.area"))
        draw_outline(p, rect, radius, theme.qcolor("line"))
        p.end()


class _SplitLayout(QLayout):
    """属性拆分行的排版：标签占左边固定比例（右对齐），控件占右边。比例相同，多行自然对成两列。"""

    def __init__(self, parent: QWidget, factor: float) -> None:
        super().__init__(parent)
        self._items: list[QWidgetItem] = []
        self.factor = factor
        self.split = True
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(0)

    def add_widget(self, widget: QWidget) -> None:
        self.addChildWidget(widget)
        self._items.append(QWidgetItem(widget))
        self.invalidate()

    def addItem(self, item: Any) -> None:  # Qt 接口；这里只经 add_widget 加入
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Any:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> Any:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Any:
        return Qt.Orientation.Horizontal

    def _label(self) -> Any:
        return self._items[0] if self._items else None

    def _content(self) -> Any:
        return self._items[1] if len(self._items) > 1 else None

    def label_width(self, total: int) -> int:
        return int(round(total * self.factor)) if self.split else 0

    def hasHeightForWidth(self) -> bool:
        content = self._content()
        return content is not None and content.hasHeightForWidth()

    def heightForWidth(self, width: int) -> int:
        content = self._content()
        if content is None:
            return theme.size("widget")
        available = max(0, width - self.label_width(width))
        height = content.heightForWidth(available) if content.hasHeightForWidth() else content.sizeHint().height()
        return max(height, theme.size("widget") if self.split else 0)

    def sizeHint(self) -> QSize:
        content = self._content()
        hint = content.sizeHint() if content is not None else QSize(0, 0)
        if not self.split:
            return hint
        width = int(math.ceil(hint.width() / max(0.1, 1.0 - self.factor)))
        return QSize(width, max(hint.height(), theme.size("widget")))

    def minimumSize(self) -> QSize:
        content = self._content()
        hint = content.minimumSize() if content is not None else QSize(0, 0)
        if not self.split:
            return hint
        width = int(math.ceil(hint.width() / max(0.1, 1.0 - self.factor)))
        return QSize(width, max(hint.height(), theme.size("widget")))

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        label, content = self._label(), self._content()
        if self.split:
            left = self.label_width(rect.width())
            gap = theme.size("pad")
            if label is not None:
                label.setGeometry(QRect(rect.x(), rect.y(), max(0, left - gap),
                                        min(rect.height(), theme.size("widget"))))
            if content is not None:
                content.setGeometry(QRect(rect.x() + left, rect.y(), rect.width() - left, rect.height()))
        else:
            if label is not None:
                label.setGeometry(QRect(rect.x(), rect.y(), 0, 0))
            if content is not None:
                content.setGeometry(rect)


class _SplitRow(QWidget):
    """属性拆分的一行：左边是右对齐的次要色标签，右边是控件。"""

    def __init__(self, text: str, factor: float, split: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = Label(text, role="dim", align=Qt.AlignRight)
        self.content: QWidget | None = None
        self._layout = _SplitLayout(self, factor)
        self._layout.add_widget(self.label)
        self.set_split(split)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def set_content(self, widget: QWidget) -> None:
        self.content = widget
        self._layout.add_widget(widget)
        policy = widget.sizePolicy()
        if policy.verticalPolicy() == QSizePolicy.Fixed and not policy.hasHeightForWidth():
            own = self.sizePolicy()
            own.setVerticalPolicy(QSizePolicy.Fixed)
            self.setSizePolicy(own)

    def set_split(self, split: bool) -> None:
        self._layout.split = bool(split)
        self.label.setVisible(bool(split))
        self._layout.invalidate()
        self.updateGeometry()

    def is_split(self) -> bool:
        return self._layout.split

    def set_text(self, text: str) -> None:
        self.label.setText(text)

    def text(self) -> str:
        return self.label.text()


# ---------------------------------------------------------------------------
# 布局语句
# ---------------------------------------------------------------------------
class UILayout:
    """面板、标题栏、浮层的内容排版。见模块说明。"""

    #: 属性拆分时标签占的宽度比例。
    split_factor = 0.38

    def __init__(self, parent_widget: QWidget, ctx: Any, direction: str = "COLUMN", *, align: bool = False,
                 _nested: bool = False) -> None:
        self.ctx = ctx
        self.direction = "ROW" if str(direction).upper() == "ROW" else "COLUMN"
        self.align = bool(align)
        #: 标签在左、控件在右。面板里默认开（由面板宿主设置），标题栏里默认关。
        self.use_property_split = False
        self._enabled = True
        self._parent: UILayout | None = None
        self._expand = False
        self._split_row: _SplitRow | None = None
        self._split_label_taken = False
        self._children: list[UILayout] = []
        self._bindings: list[_Binding] = []
        self._align_groups: list[list[weakref.ref]] = [[]]
        existing = parent_widget.layout()
        if existing is None:
            container = parent_widget
        else:
            container = _Block(parent_widget)
            existing.addWidget(container)
        #: 本布局装内容的控件。
        self.container = container
        direction_flag = QBoxLayout.LeftToRight if self.direction == "ROW" else QBoxLayout.TopToBottom
        self._box = QBoxLayout(direction_flag, container)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(self._spacing())
        # 竖排的根布局末尾留一段伸缩空白：容器比内容高时，内容靠上排，不被拉散
        self._tail = self.direction == "COLUMN" and not _nested
        if self._tail:
            self._box.addStretch(1)

    def _index(self) -> int:
        """新内容插入的位置（在末尾的伸缩空白之前）。"""
        return self._box.count() - (1 if self._tail else 0)

    # ---- 状态 ----
    @property
    def enabled(self) -> bool:
        """设为 False 后，其后加入的内容都置灰。"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def _effective_enabled(self) -> bool:
        node: UILayout | None = self
        while node is not None:
            if not node._enabled:
                return False
            node = node._parent
        return True

    def _spacing(self) -> int:
        return int(line_width()) if self.align else theme.size("gap")

    def _splits(self) -> bool:
        return self.use_property_split and self.direction == "COLUMN" and self._split_row is None

    def _mode(self) -> str:
        if self._splits():
            return "split"
        if self._split_row is not None and not self._split_label_taken:
            return "row_first"
        return "plain"

    def bindings(self) -> list:
        """本布局及子布局里的全部绑定（测试与调试用）。"""
        out = list(self._bindings)
        for child in self._children:
            out.extend(child.bindings())
        return out

    # ---- 放控件 ----
    def _add(self, widget: QWidget, *, stretch: int | None = None, align_item: QWidget | None = None) -> QWidget:
        if stretch is None:
            stretch = 1 if (self.direction == "ROW" and self._expand) else 0
        if self.direction == "ROW" and not self._expand and stretch == 0:
            # 标题栏这类不铺满的行：控件按自身大小排，不抢空间
            policy = widget.sizePolicy()
            if policy.horizontalPolicy() in (QSizePolicy.Expanding, QSizePolicy.MinimumExpanding, QSizePolicy.Preferred):
                policy.setHorizontalPolicy(QSizePolicy.Maximum)
                widget.setSizePolicy(policy)
        self._box.insertWidget(self._index(), widget, stretch)
        if not self._effective_enabled():
            self._disable(widget)
        if align_item is not None:
            self._align_groups[-1].append(weakref.ref(align_item))
            self._update_corners()
        if self._split_row is not None:
            self._split_label_taken = True
        return widget

    @staticmethod
    def _disable(widget: QWidget) -> None:
        setter = getattr(widget, "set_layout_enabled", None)
        if callable(setter):
            setter(False)
        else:
            widget.setEnabled(False)

    def _update_corners(self) -> None:
        if not self.align:
            return
        for group in self._align_groups:
            items = []
            for ref in group:
                widget = ref()
                if widget is not None and shiboken6.isValid(widget) and hasattr(widget, "set_corners"):
                    items.append(widget)
            count = len(items)
            for index, widget in enumerate(items):
                if count == 1:
                    corners = CORNERS_ALL
                elif self.direction == "ROW":
                    corners = CORNERS_LEFT if index == 0 else CORNERS_RIGHT if index == count - 1 else 0
                else:
                    corners = CORNERS_TOP if index == 0 else CORNERS_BOTTOM if index == count - 1 else 0
                widget.set_corners(corners)

    def _take_row_label(self, text: str) -> None:
        row = self._split_row
        if row is not None:
            row.set_text(text)
            row.set_split(True)
        self._split_label_taken = True

    # ---- 子布局 ----
    def _child(self, container: QWidget, direction: str, align: bool) -> "UILayout":
        child = UILayout(container, self.ctx, direction, align=align, _nested=True)
        child._parent = self
        child.use_property_split = self.use_property_split
        child._expand = direction == "ROW" and (self.direction == "COLUMN" or self._expand)
        self._children.append(child)
        return child

    def row(self, align: bool = False, heading: str | None = None) -> "UILayout":
        """横排。属性拆分时这一行整体放在控件列：左边显示 heading，没有 heading 时显示行里第一个属性的名称。"""
        if self._splits():
            row = _SplitRow(heading or "", self.split_factor, split=heading is not None)
            content = _Block()
            row.set_content(content)
            self._add(row)
            child = self._child(content, "ROW", align)
            child.use_property_split = False
            child._split_row = row
            child._split_label_taken = heading is not None
            child._expand = True
            return child
        block = _Block()
        self._add(block)
        return self._child(block, "ROW", align)

    def column(self, align: bool = False) -> "UILayout":
        block = _Block()
        self._add(block)
        return self._child(block, "COLUMN", align)

    def box(self) -> "UILayout":
        block = _Block(boxed=True)
        self._add(block)
        child = self._child(block, "COLUMN", False)
        margin = theme.size("pad")
        child._box.setContentsMargins(margin, margin, margin, margin)
        return child

    # ---- 内容 ----
    def label(self, text: str = "", icon: Any = None, role: str | None = None) -> Label:
        widget = Label(text, role=role, icon=icon)
        self._add(widget)
        return widget

    def separator(self, factor: float = 1.0) -> None:
        """留一段空白；对齐成组的控件在这里断开。"""
        self._box.insertSpacing(self._index(), max(0, int(round(theme.size("pad") * float(factor)))))
        self._align_groups.append([])

    def widget(self, w: QWidget, stretch: int = 0) -> QWidget:
        """放进自定义控件。stretch 为 0 时按所在行的规则（面板行里等分，标题栏里按自身大小）。"""
        self._add(w, stretch=stretch if stretch else None)
        return w

    def stretch(self) -> None:
        self._box.insertStretch(self._index(), 1)

    def operator(self, idname: str, text: str | None = None, icon: Any = None, role: str | None = None,
                 **props: Any) -> QWidget:
        cls = ops.get(idname)
        label = text if text is not None else (getattr(cls, "label", "") or idname)
        if icon is None and cls is not None:
            icon = getattr(cls, "icon", "") or None
        button = OperatorButton(idname, props, self.ctx, label, icon, role)
        self._add(button, align_item=button)
        return button

    def menu(self, menu_idname: str, text: str | None = None, icon: Any = None) -> QWidget:
        cls = registry.menu(menu_idname)
        label = text if text is not None else (getattr(cls, "label", "") or menu_idname)
        button = Button(label, icon)
        button.set_menu_indicator(True)
        button.clicked.connect(partial(self._open_menu, weakref.ref(button), menu_idname))
        self._add(button, align_item=button)
        return button

    def _open_menu(self, button_ref: weakref.ref, menu_idname: str, *_args: Any) -> QMenu | None:
        button = button_ref()
        if button is None or not shiboken6.isValid(button):
            return None
        qmenu = build_menu(menu_idname, self.ctx, button)
        qmenu.setAttribute(Qt.WA_DeleteOnClose, True)
        qmenu.popup(button.mapToGlobal(QPoint(0, button.height())))
        return qmenu

    def popover(self, text: str, icon: Any, draw_fn: Callable[["UILayout", Any], None]) -> QWidget:
        """点开是浮层的按钮。draw_fn(layout, ctx) 每次打开时填充浮层内容（属性拆分默认开）。"""
        if not text and icon == "chevron.down":
            icon = None              # 只留一个下拉箭头（按钮自己会画），不再叠一个箭头图标
        button = Button(text or "", icon)
        button.set_menu_indicator(True)
        button.clicked.connect(partial(self._open_popover, weakref.ref(button), draw_fn))
        self._add(button, align_item=button)
        return button

    def _open_popover(self, button_ref: weakref.ref, draw_fn: Callable, *_args: Any) -> Popover | None:
        button = button_ref()
        if button is None or not shiboken6.isValid(button):
            return None
        pop = Popover(button)
        pop.setAttribute(Qt.WA_DeleteOnClose, True)
        layout = UILayout(pop.body, self.ctx)
        layout.use_property_split = True
        try:
            draw_fn(layout, self.ctx)
        except Exception as error:  # noqa: BLE001
            log.exception("浮层内容生成出错")
            layout.label("内容没能显示：%s" % error, icon="warning", role="dim")
        pop._splender_layout = layout
        button._splender_popover = pop
        pop.popup()
        return pop

    # ---- 属性 ----
    def prop(self, data: Any, name: str, text: str | None = None, icon: Any = None, slider: bool | None = None,
             expand: bool = False, toggle: bool = False, icon_only: bool = False) -> QWidget:
        """按属性类型生成控件并双向绑定。返回主控件。"""
        target, attr = resolve_path(data, name)
        prop = _descriptor(target, attr)
        label = prop.label if text is None else str(text)
        if icon is None:
            icon = getattr(prop, "icon", "") or None
        binding = _Binding(self, data, target, attr, prop)
        kind = getattr(prop, "kind", "")
        if kind in ("FLOAT", "INT"):
            return self._prop_number(binding, label, slider)
        if kind == "BOOL":
            return self._prop_bool(binding, label, icon, toggle, icon_only)
        if kind == "ENUM":
            return self._prop_enum(binding, label, expand, icon_only)
        if kind == "COLOR":
            return self._prop_color(binding, label)
        if kind == "VECTOR":
            return self._prop_vector(binding, label, slider)
        if kind == "STRING":
            return self._prop_string(binding, label)
        if kind == "RAMP":
            return self._prop_ramp(binding, label)
        if kind == "CURVE":
            return self._prop_curve(binding, label)
        binding.detach()
        self._bindings.remove(binding)
        raise TypeError("属性 %s 的类型 %s 不能直接生成控件" % (name, kind))

    def _labeled(self, binding: _Binding, control: QWidget, label: str, outer: QWidget | None = None) -> None:
        """放一个自身不带标签的控件：拆分或在竖排里时左边放标签。"""
        mode = self._mode()
        desc = getattr(binding.prop, "description", "") or ""
        outer = outer or control
        if mode == "row_first":
            self._take_row_label(label)
            self._add(outer, align_item=control)
        elif self.direction == "COLUMN" and (mode == "split" or label):
            row = _SplitRow(label, self.split_factor)
            row.set_content(outer)
            if desc:
                row.label.setToolTip(desc)
            self._add(row, align_item=control)
        elif label:
            caption = Label(label, role="dim")
            if desc:
                caption.setToolTip(desc)
            self._add(caption)
            self._add(outer, align_item=control)
        else:
            self._add(outer, align_item=control)

    def _prop_number(self, binding: _Binding, label: str, slider: bool | None) -> NumberField:
        prop = binding.prop
        mode = self._mode()
        is_int = getattr(prop, "kind", "") == "INT"
        field = NumberField(binding.read(), prop.min, prop.max, prop.soft_min, prop.soft_max, prop.step,
                            getattr(prop, "precision", 0 if is_int else 2), _unit_of(prop),
                            label if mode == "plain" else "",
                            slider if slider is not None else getattr(prop, "subtype", "") == "FACTOR", is_int)
        desc = getattr(prop, "description", "") or ""
        if desc:
            field.setToolTip(desc)
        binding.bind_number(field)
        if mode == "plain":
            self._add(field, align_item=field)
        else:
            self._labeled(binding, field, label)
        return field

    def _prop_bool(self, binding: _Binding, label: str, icon: Any, toggle: bool, icon_only: bool) -> QWidget:
        prop = binding.prop
        has_icon = _icon_ok(icon)
        if icon_only and has_icon:
            widget: QWidget = Toggle("", icon)
            widget.setToolTip(label + ("\n" + prop.description if getattr(prop, "description", "") else ""))
            widget.setAccessibleName(label)
            text_inside = False
        elif toggle or icon_only:
            widget = Toggle(label, icon if has_icon else None)
            text_inside = True
        else:
            widget = CheckBox(label)
            text_inside = True
        desc = getattr(prop, "description", "") or ""
        if desc and text_inside:
            widget.setToolTip(desc)
        binding.bind_bool(widget)
        mode = self._mode()
        if mode == "split":
            row = _SplitRow("" if text_inside else label, self.split_factor)
            row.set_content(widget)
            self._add(row, align_item=widget)
        elif mode == "row_first":
            self._take_row_label("" if text_inside else label)
            self._add(widget, align_item=widget)
        else:
            self._add(widget, align_item=widget)
        return widget

    def _prop_enum(self, binding: _Binding, label: str, expand: bool, icon_only: bool) -> QWidget:
        prop, target = binding.prop, binding.target
        if expand:
            widget: QWidget = SegmentedControl(prop.items(target), icon_only=icon_only)
        else:
            widget = EnumField(partial(prop.items, target), icon_only=icon_only)
        desc = getattr(prop, "description", "") or ""
        if desc:
            widget.setToolTip(desc)
        binding.bind_enum(widget)
        self._labeled(binding, widget, label)
        return widget

    def _prop_color(self, binding: _Binding, label: str) -> ColorButton:
        button = ColorButton(binding.read())
        desc = getattr(binding.prop, "description", "") or ""
        if desc:
            button.setToolTip(desc)
        binding.bind_color(button)
        self._labeled(binding, button, label)
        return button

    def _prop_ramp(self, binding: _Binding, label: str) -> QWidget:
        """色带：占满一行（和 Blender 一样不分左右栏），名字在上面。"""
        from .ramp_widget import ColorRampWidget
        from .widget_state import state_for

        widget = ColorRampWidget(binding.read(), state=state_for(binding.target, binding.attr))
        widget.bar.focus_key = ("ramp", id(binding.target), binding.attr)
        desc = getattr(binding.prop, "description", "") or ""
        if desc:
            widget.setToolTip(desc)
        binding.bind_ramp(widget)
        if label:
            self.label(label)
        self._add(widget, align_item=widget)
        return widget

    def _prop_curve(self, binding: _Binding, label: str) -> QWidget:
        """曲线：占满一行，名字在上面。"""
        from .curve_widget import CurveWidget
        from .widget_state import state_for

        widget = CurveWidget(binding.read(), channel=getattr(binding.prop, "subtype", "RGB") or "RGB",
                             state=state_for(binding.target, binding.attr))
        widget.focus_key = ("curve", id(binding.target), binding.attr)
        desc = getattr(binding.prop, "description", "") or ""
        if desc:
            widget.setToolTip(desc)
        binding.bind_curve(widget)
        if label:
            self.label(label)
        self._add(widget, align_item=widget)
        return widget

    def _prop_string(self, binding: _Binding, label: str) -> TextField:
        prop = binding.prop
        field = TextField(binding.read())
        desc = getattr(prop, "description", "") or ""
        if desc:
            field.setToolTip(desc)
        binding.bind_string(field)
        outer: QWidget = field
        subtype = getattr(prop, "subtype", "")
        if subtype in ("DIR_PATH", "FILE_PATH"):
            outer = _Block()
            row = QHBoxLayout(outer)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(theme.size("gap"))
            browse = IconButton("open", "选择文件夹" if subtype == "DIR_PATH" else "选择文件")
            browse.clicked.connect(partial(_browse_path, binding, subtype, weakref.ref(field)))
            row.addWidget(field, 1)
            row.addWidget(browse)
            outer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            field.browse_button = browse
        self._labeled(binding, field, label, outer)
        return field

    def _prop_vector(self, binding: _Binding, label: str, slider: bool | None) -> QWidget:
        prop = binding.prop
        values = binding.read()
        count = len(values)
        mode = self._mode()
        unit = _unit_of(prop)
        precision = getattr(prop, "precision", 3)
        desc = getattr(prop, "description", "") or ""
        fields: list[NumberField] = []

        def make(index: int, in_label: str) -> NumberField:
            field = NumberField(values[index], prop.min, prop.max, None, None, None, precision, unit, in_label,
                                bool(slider))
            if desc:
                field.setToolTip(desc)
            fields.append(field)
            return field

        if mode == "split":
            column = self.column(align=True)
            for index in range(count):
                comp = _COMPONENT_LABELS[index] if index < len(_COMPONENT_LABELS) else str(index + 1)
                text = ("%s %s" % (label, comp)) if index == 0 and label else comp
                row = _SplitRow(text, self.split_factor)
                row.set_content(make(index, ""))
                column._add(row, align_item=fields[-1])
            binding.bind_vector(fields)
            return column.container
        holder = _Block()
        holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_box = QHBoxLayout(holder)
        row_box.setContentsMargins(0, 0, 0, 0)
        row_box.setSpacing(int(line_width()))
        for index in range(count):
            comp = _COMPONENT_LABELS[index] if index < len(_COMPONENT_LABELS) else str(index + 1)
            field = make(index, comp)
            if count > 1:
                field.set_corners(CORNERS_LEFT if index == 0 else CORNERS_RIGHT if index == count - 1 else 0)
            row_box.addWidget(field, 1)
        binding.bind_vector(fields)
        self._labeled(binding, holder, label)
        return holder

    # ---- 清空 ----
    def _detach_all(self) -> None:
        for binding in self._bindings:
            binding.detach()
        for child in self._children:
            child._detach_all()

    def clear(self) -> None:
        """删掉全部内容并断开绑定，可以重新往里排。"""
        self._detach_all()
        while self._box.count():
            item = self._box.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        if self._tail:
            self._box.addStretch(1)
        self._children.clear()
        self._bindings.clear()
        self._align_groups = [[]]
        self._split_label_taken = False


def _browse_path(binding: _Binding, subtype: str, field_ref: weakref.ref, *_args: Any) -> None:
    field = field_ref()
    if field is None or not shiboken6.isValid(field):
        return
    start = str(binding.read() or "")
    if subtype == "DIR_PATH":
        path = QFileDialog.getExistingDirectory(field, "选择文件夹", start)
    else:
        path, _filter = QFileDialog.getOpenFileName(field, "选择文件", start)
    if path:
        binding.write(path)


# ---------------------------------------------------------------------------
# 菜单里的布局语句
# ---------------------------------------------------------------------------
_floating_popovers: set = set()


def _open_floating_popover(draw_fn: Callable, ctx: Any, *_args: Any) -> Popover:
    pop = Popover(None)
    pop.setAttribute(Qt.WA_DeleteOnClose, True)
    layout = UILayout(pop.body, ctx)
    layout.use_property_split = True
    try:
        draw_fn(layout, ctx)
    except Exception as error:  # noqa: BLE001
        log.exception("浮层内容生成出错")
        layout.label("内容没能显示：%s" % error, icon="warning", role="dim")
    pop._splender_layout = layout
    _floating_popovers.add(pop)
    pop.closed.connect(partial(_floating_popovers.discard, pop))
    pop.popup()
    return pop


class MenuLayout:
    """菜单里的布局语句：operator / menu / prop / label / separator 变成 QMenu 的条目。

    行、列、框在菜单里没有意义，row()/column()/box() 直接返回自己。
    """

    def __init__(self, qmenu: QMenu, ctx: Any) -> None:
        self.qmenu = qmenu
        self.ctx = ctx
        self.direction = "COLUMN"
        self.align = False
        self.use_property_split = False
        self.enabled = True
        self._op_actions: list[tuple[QAction, str, bool]] = []
        self._bindings: list[_Binding] = []
        self._holders: list[UILayout] = []

    def row(self, align: bool = False, heading: str | None = None) -> "MenuLayout":
        return self

    def column(self, align: bool = False) -> "MenuLayout":
        return self

    def box(self) -> "MenuLayout":
        return self

    def label(self, text: str = "", icon: Any = None, role: str | None = None) -> QAction:
        action = self.qmenu.addAction(text)
        if _icon_ok(icon):
            action.setIcon(icons.icon(icon))
        action.setEnabled(False)
        return action

    def operator(self, idname: str, text: str | None = None, icon: Any = None, role: str | None = None,
                 **props: Any) -> QAction:
        cls = ops.get(idname)
        label = text if text is not None else (getattr(cls, "label", "") or idname)
        shortcut = _shortcut(idname, props, self.ctx)
        action = QAction(label + ("\t" + shortcut if shortcut else ""), self.qmenu)
        if icon is None and cls is not None:
            icon = getattr(cls, "icon", "") or None
        if _icon_ok(icon):
            action.setIcon(icons.icon(icon))
        desc = getattr(cls, "description", "") if cls is not None else ""
        if desc:
            action.setToolTip(desc)
        action.triggered.connect(partial(_call_operator, idname, dict(props), self.ctx))
        self.qmenu.addAction(action)
        self._op_actions.append((action, idname, self.enabled))
        action.setEnabled(self.enabled and ops.poll(idname, self.ctx))
        return action

    def refresh_polls(self) -> None:
        """重新判断各操作项能不能用（菜单每次打开前调用）。"""
        for action, idname, layout_enabled in self._op_actions:
            if shiboken6.isValid(action):
                action.setEnabled(layout_enabled and ops.poll(idname, self.ctx))

    def menu(self, menu_idname: str, text: str | None = None, icon: Any = None) -> QMenu:
        sub = build_menu(menu_idname, self.ctx, self.qmenu)
        if text is not None:
            sub.setTitle(text)
        if _icon_ok(icon):
            sub.setIcon(icons.icon(icon))
        self.qmenu.addMenu(sub)
        sub.menuAction().setEnabled(self.enabled)
        return sub

    def popover(self, text: str, icon: Any, draw_fn: Callable) -> QAction:
        action = self.qmenu.addAction(text)
        if _icon_ok(icon):
            action.setIcon(icons.icon(icon))
        action.triggered.connect(partial(_open_floating_popover, draw_fn, self.ctx))
        action.setEnabled(self.enabled)
        return action

    def prop(self, data: Any, name: str, text: str | None = None, icon: Any = None, slider: bool | None = None,
             expand: bool = False, toggle: bool = False, icon_only: bool = False) -> Any:
        target, attr = resolve_path(data, name)
        prop = _descriptor(target, attr)
        label = prop.label if text is None else str(text)
        kind = getattr(prop, "kind", "")
        if kind == "BOOL":
            binding = _Binding(self, data, target, attr, prop)
            action = QAction(label, self.qmenu)
            if getattr(prop, "description", ""):
                action.setToolTip(prop.description)
            binding.bind_action_bool(action)
            action.setEnabled(self.enabled)
            self.qmenu.addAction(action)
            return action
        if kind == "ENUM":
            binding = _Binding(self, data, target, attr, prop)
            holder = self.qmenu if expand else self.qmenu.addMenu(label)
            group = QActionGroup(holder)
            group.setExclusive(True)
            actions = []
            current = binding.read()
            for ident, item_text, desc, item_icon in prop.items(target):
                action = QAction(item_text, holder)
                action.setCheckable(True)
                action.setData(ident)
                action.setChecked(ident == current)
                if desc:
                    action.setToolTip(desc)
                if _icon_ok(item_icon):
                    action.setIcon(icons.icon(item_icon))
                group.addAction(action)
                holder.addAction(action)
                actions.append(action)
            binding.bind_action_enum(actions)
            if holder is not self.qmenu:
                holder.menuAction().setEnabled(self.enabled)
            return holder
        # 其他类型：把普通控件嵌进菜单
        holder_widget = QWidget()
        inner = UILayout(holder_widget, self.ctx, "ROW")
        margin = theme.size("pad")
        inner._box.setContentsMargins(margin, int(line_width()), margin, int(line_width()))
        inner._expand = True
        inner.enabled = self.enabled
        widget = inner.prop(data, name, text, icon, slider, expand, toggle, icon_only)
        action = QWidgetAction(self.qmenu)
        action.setDefaultWidget(holder_widget)
        self.qmenu.addAction(action)
        self._holders.append(inner)
        return widget

    def separator(self, factor: float = 1.0) -> None:
        self.qmenu.addSeparator()

    def widget(self, w: QWidget, stretch: int = 0) -> QWidget:
        action = QWidgetAction(self.qmenu)
        action.setDefaultWidget(w)
        self.qmenu.addAction(action)
        if not self.enabled:
            w.setEnabled(False)
        return w

    def stretch(self) -> None:
        """菜单里没有伸缩空白。"""

    def clear(self) -> None:
        for binding in self._bindings:
            binding.detach()
        self._bindings.clear()
        for holder in self._holders:
            holder.clear()
        self._holders.clear()
        self._op_actions.clear()
        self.qmenu.clear()
