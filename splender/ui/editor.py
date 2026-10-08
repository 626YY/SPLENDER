"""编辑器基类，以及把 Qt 输入事件转成 core.keymap.Event 的辅助函数。

一个编辑器分四块，和 Blender 一样：标题栏在上，下面是 工具栏 | 主区域 | 侧栏。
子类实现 build_main() 和 draw_header()，其余由基类负责：

- 标题栏：最左是区域的编辑器类型按钮（由 AreaWidget 放进来），右边是 draw_header() 排的内容；
- 工具栏（has_toolbar）：按 registry.tools(idname) 生成工具按钮，按 T 开关；
- 侧栏（has_sidebar）：按 registry.panels(idname, "SIDEBAR") 的标签页分页，按 N 开关，宽度可拖；
- 键位分发：handle_event(event) 依次查 活动工具键位表 → keymaps → "Screen" → "Window"；
- 状态：save_state() / load_state() 存取工具栏、侧栏的开关和宽度，子类在此基础上追加自己的字段。

主区域控件把鼠标、滚轮、数位板事件转成 Event 后调 handle_event。本模块的
key_event() / mouse_event() / wheel_event() / tablet_event() 负责转换，
install_input(widget) 一步装好（打开鼠标跟踪、点击取得焦点、关闭输入法）。

约定：
- 子类的 build_main() 在基类 __init__ 末尾被调用，需要的属性请在 super().__init__() 之前设置或在 build_main 里设置。
- 构造之后由区域依次调用 load_state(state) 和 refresh()，标题栏在 refresh() 时第一次画出。
- 子类覆盖 showEvent / hideEvent 时要调用基类实现，on_show() / on_hide() 由它们触发。
"""
from __future__ import annotations

import copy
import logging
import sys
from typing import Any, Callable, Iterable

import shiboken6
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QCursor, QPainter, QPointingDevice
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea, QStackedWidget, QToolButton, QVBoxLayout,
                               QWidget)

from ..core import ops, registry
from ..core.context import Context
from ..core.keymap import (CLICK, CLICK_DRAG, DOUBLE, LEFTMOUSE, MIDDLEMOUSE, MOUSEMOVE, MOVE, PRESS, RELEASE,
                           RIGHTMOUSE, WHEEL, WHEELDOWN, WHEELUP, Event)
from . import icons, theme

log = logging.getLogger("splender.ui.editor")

#: 事件已经交给过模态操作的标记（写在 Event 实例上，防止同一事件被模态操作处理两次）
MODAL_SEEN = "_wm_modal_seen"


def alive(obj: Any) -> bool:
    """Qt 对象存在且还没被销毁。"""
    if obj is None:
        return False
    try:
        return bool(shiboken6.isValid(obj))
    except Exception:  # noqa: BLE001
        return False


# ======================================================================
# 键名转换
# ======================================================================

_KEY_NAMES: dict[int, str] = {}
_KEYPAD_NAMES: dict[int, str] = {}
_VK_NAMES: dict[int, str] = {}
#: 修饰键自身按下时，事件里不再带它自己的修饰标志
_MODIFIER_KEYS = {"LEFT_CTRL": "ctrl", "RIGHT_CTRL": "ctrl", "LEFT_SHIFT": "shift", "RIGHT_SHIFT": "shift",
                  "LEFT_ALT": "alt", "RIGHT_ALT": "alt"}


def _init_tables() -> None:
    if _KEY_NAMES:
        return
    K = Qt.Key
    for i in range(26):
        _KEY_NAMES[int(K.Key_A) + i] = chr(ord("A") + i)
    for i in range(10):
        _KEY_NAMES[int(K.Key_0) + i] = str(i)
        _KEYPAD_NAMES[int(K.Key_0) + i] = "NUMPAD_%d" % i
    for i in range(24):
        _KEY_NAMES[int(K.Key_F1) + i] = "F%d" % (i + 1)
    named = {
        "Key_Space": "SPACE", "Key_Escape": "ESC", "Key_Return": "RET", "Key_Enter": "RET", "Key_Tab": "TAB",
        "Key_Backtab": "TAB", "Key_Backspace": "BACK_SPACE", "Key_Delete": "DEL", "Key_Insert": "INSERT",
        "Key_Home": "HOME", "Key_End": "END", "Key_PageUp": "PAGE_UP", "Key_PageDown": "PAGE_DOWN",
        "Key_Left": "LEFT_ARROW", "Key_Right": "RIGHT_ARROW", "Key_Up": "UP_ARROW", "Key_Down": "DOWN_ARROW",
        "Key_BracketLeft": "LEFT_BRACKET", "Key_BraceLeft": "LEFT_BRACKET",
        "Key_BracketRight": "RIGHT_BRACKET", "Key_BraceRight": "RIGHT_BRACKET",
        "Key_Comma": "COMMA", "Key_Less": "COMMA", "Key_Period": "PERIOD", "Key_Greater": "PERIOD",
        "Key_Slash": "SLASH", "Key_Question": "SLASH", "Key_Minus": "MINUS", "Key_Underscore": "MINUS",
        "Key_Equal": "EQUAL", "Key_Plus": "EQUAL", "Key_Semicolon": "SEMI_COLON", "Key_Colon": "SEMI_COLON",
        "Key_Apostrophe": "QUOTE", "Key_QuoteDbl": "QUOTE", "Key_QuoteLeft": "ACCENT_GRAVE",
        "Key_AsciiTilde": "ACCENT_GRAVE", "Key_Backslash": "BACK_SLASH", "Key_Bar": "BACK_SLASH",
        # 主键盘上 Shift 打出的符号，按所在的键算（和 Blender 一样认物理键）
        "Key_Exclam": "1", "Key_At": "2", "Key_NumberSign": "3", "Key_Dollar": "4", "Key_Percent": "5",
        "Key_AsciiCircum": "6", "Key_Ampersand": "7", "Key_Asterisk": "8", "Key_ParenLeft": "9",
        "Key_ParenRight": "0",
        "Key_Control": "LEFT_CTRL", "Key_Shift": "LEFT_SHIFT", "Key_Alt": "LEFT_ALT", "Key_AltGr": "RIGHT_ALT",
        "Key_Meta": "OSKEY", "Key_Super_L": "OSKEY", "Key_Super_R": "OSKEY", "Key_Menu": "APP",
        "Key_Pause": "PAUSE", "Key_Print": "PRINT", "Key_Clear": "NUMPAD_5",
    }
    for qt_name, name in named.items():
        value = getattr(K, qt_name, None)
        if value is not None:
            _KEY_NAMES[int(value)] = name
    keypad = {
        "Key_Period": "NUMPAD_PERIOD", "Key_Comma": "NUMPAD_PERIOD", "Key_Delete": "NUMPAD_PERIOD",
        "Key_Plus": "NUMPAD_PLUS", "Key_Minus": "NUMPAD_MINUS", "Key_Asterisk": "NUMPAD_ASTERIX",
        "Key_Slash": "NUMPAD_SLASH", "Key_Enter": "NUMPAD_ENTER", "Key_Return": "NUMPAD_ENTER",
        "Key_Equal": "NUMPAD_EQUAL", "Key_Clear": "NUMPAD_5",
    }
    if sys.platform != "darwin":
        # 关闭数字锁定时小键盘发出的是方向和翻页键，Qt 会带上 KeypadModifier。
        # macOS 的独立方向键也带这个标志，所以那里不做这层映射。
        keypad.update({
            "Key_Insert": "NUMPAD_0", "Key_End": "NUMPAD_1", "Key_Down": "NUMPAD_2", "Key_PageDown": "NUMPAD_3",
            "Key_Left": "NUMPAD_4", "Key_Right": "NUMPAD_6", "Key_Home": "NUMPAD_7", "Key_Up": "NUMPAD_8",
            "Key_PageUp": "NUMPAD_9",
        })
    for qt_name, name in keypad.items():
        value = getattr(K, qt_name, None)
        if value is not None:
            _KEYPAD_NAMES[int(value)] = name
    # Windows 虚拟键码：不受 Shift 和键盘布局的字符影响，用来认物理键
    for i in range(10):
        _VK_NAMES[0x30 + i] = str(i)
        _VK_NAMES[0x60 + i] = "NUMPAD_%d" % i
    for i in range(26):
        _VK_NAMES[0x41 + i] = chr(ord("A") + i)
    _VK_NAMES.update({
        0x6A: "NUMPAD_ASTERIX", 0x6B: "NUMPAD_PLUS", 0x6D: "NUMPAD_MINUS", 0x6E: "NUMPAD_PERIOD",
        0x6F: "NUMPAD_SLASH", 0xBA: "SEMI_COLON", 0xBB: "EQUAL", 0xBC: "COMMA", 0xBD: "MINUS", 0xBE: "PERIOD",
        0xBF: "SLASH", 0xC0: "ACCENT_GRAVE", 0xDB: "LEFT_BRACKET", 0xDC: "BACK_SLASH", 0xDD: "RIGHT_BRACKET",
        0xDE: "QUOTE",
    })


def key_name(qt_key: int, modifiers: Any = Qt.NoModifier, native_virtual_key: int = 0) -> str:
    """Qt 键值转成键位表里的键名。认不出时返回空字符串。

    小键盘上的键（Qt 带 KeypadModifier）转成 NUMPAD_*，和主键盘数字区分开。
    """
    _init_tables()
    try:
        key = int(qt_key)
    except (TypeError, ValueError):
        return ""
    keypad = bool(modifiers & Qt.KeypadModifier) if modifiers is not None else False
    if keypad and key in _KEYPAD_NAMES:
        return _KEYPAD_NAMES[key]
    if native_virtual_key and sys.platform == "win32":
        name = _VK_NAMES.get(int(native_virtual_key))
        if name:
            return name
    return _KEY_NAMES.get(key, "")


def modifier_state(modifiers: Any) -> tuple[bool, bool, bool]:
    """Qt 修饰键标志转成 (ctrl, shift, alt)。"""
    if modifiers is None:
        return False, False, False
    return (bool(modifiers & Qt.ControlModifier), bool(modifiers & Qt.ShiftModifier),
            bool(modifiers & Qt.AltModifier))


def live_modifier_state(modifiers: Any = None) -> tuple[bool, bool, bool]:
    """修饰键的实际状态：事件自带的标志再并上键盘此刻的状态（数位笔事件在 Windows 上常常不带修饰键）。"""
    ctrl, shift, alt = modifier_state(modifiers)
    try:
        from PySide6.QtGui import QGuiApplication
        k_ctrl, k_shift, k_alt = modifier_state(QGuiApplication.queryKeyboardModifiers())
    except Exception:  # noqa: BLE001
        return ctrl, shift, alt
    return ctrl or k_ctrl, shift or k_shift, alt or k_alt


_BUTTON_NAMES = {
    Qt.MouseButton.LeftButton: LEFTMOUSE,
    Qt.MouseButton.MiddleButton: MIDDLEMOUSE,
    Qt.MouseButton.RightButton: RIGHTMOUSE,
    Qt.MouseButton.BackButton: "BUTTON4MOUSE",
    Qt.MouseButton.ForwardButton: "BUTTON5MOUSE",
}


def mouse_button_name(button: Any) -> str:
    """Qt 鼠标键转成键名（LEFTMOUSE / MIDDLEMOUSE / RIGHTMOUSE …）。"""
    return _BUTTON_NAMES.get(button, "")


def key_event(qev: Any) -> Event | None:
    """QKeyEvent 转成 Event。认不出的键返回 None。"""
    t = qev.type()
    if t not in (QEvent.KeyPress, QEvent.KeyRelease):
        return None
    mods = qev.modifiers()
    try:
        native = int(qev.nativeVirtualKey())
    except Exception:  # noqa: BLE001
        native = 0
    name = key_name(qev.key(), mods, native)
    if not name:
        return None
    ctrl, shift, alt = modifier_state(mods)
    own = _MODIFIER_KEYS.get(name)
    if own == "ctrl":
        ctrl = False
    elif own == "shift":
        shift = False
    elif own == "alt":
        alt = False
    return Event(type=name, value=RELEASE if t == QEvent.KeyRelease else PRESS, ctrl=ctrl, shift=shift, alt=alt,
                 is_repeat=bool(qev.isAutoRepeat()), text=qev.text() or "")


def mouse_event(qev: Any, widget: Any = None, prev: tuple[float, float] | None = None) -> Event | None:
    """QMouseEvent 转成 Event。坐标是 widget 内的逻辑像素。"""
    t = qev.type()
    if t == QEvent.MouseMove:
        etype, value = MOUSEMOVE, MOVE
    elif t in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick, QEvent.MouseButtonRelease):
        etype = mouse_button_name(qev.button())
        if not etype:
            return None
        value = {QEvent.MouseButtonPress: PRESS, QEvent.MouseButtonDblClick: DOUBLE}.get(t, RELEASE)
    else:
        return None
    pos = qev.position()
    if t == QEvent.MouseMove:
        ctrl, shift, alt = modifier_state(qev.modifiers())
    else:
        ctrl, shift, alt = live_modifier_state(qev.modifiers())
    event = Event(type=etype, value=value, ctrl=ctrl, shift=shift, alt=alt, x=pos.x(), y=pos.y(), source=widget)
    if prev is not None:
        event.prev_x, event.prev_y = prev
    else:
        event.prev_x, event.prev_y = event.x, event.y
    return event


def wheel_event(qev: Any, widget: Any = None) -> Event | None:
    """QWheelEvent 转成 Event：WHEELUPMOUSE / WHEELDOWNMOUSE，wheel 是格数（一格 = 1.0）。"""
    delta = qev.angleDelta()
    amount = delta.y() if delta.y() != 0 else delta.x()   # Alt 会把竖直滚动变成水平
    steps = amount / 120.0
    if amount == 0:
        pixel = qev.pixelDelta()
        amount = pixel.y() if pixel.y() != 0 else pixel.x()
        steps = amount / 40.0
    if amount == 0:
        return None
    pos = qev.position()
    ctrl, shift, alt = modifier_state(qev.modifiers())
    return Event(type=WHEELUP if amount > 0 else WHEELDOWN, value=WHEEL, ctrl=ctrl, shift=shift, alt=alt,
                 x=pos.x(), y=pos.y(), prev_x=pos.x(), prev_y=pos.y(), wheel=steps, source=widget)


def tablet_event(qev: Any, widget: Any = None, prev: tuple[float, float] | None = None) -> Event | None:
    """QTabletEvent 转成 Event：带压感（0..1）、倾斜角、是否橡皮端。"""
    t = qev.type()
    if t == QEvent.TabletMove:
        etype, value = MOUSEMOVE, MOVE
    elif t in (QEvent.TabletPress, QEvent.TabletRelease):
        etype = mouse_button_name(qev.button()) or LEFTMOUSE
        value = PRESS if t == QEvent.TabletPress else RELEASE
    else:
        return None
    pos = qev.position()
    ctrl, shift, alt = live_modifier_state(qev.modifiers())
    try:
        eraser = qev.pointerType() == QPointingDevice.PointerType.Eraser
    except Exception:  # noqa: BLE001
        eraser = False
    pressure = float(qev.pressure())
    event = Event(type=etype, value=value, ctrl=ctrl, shift=shift, alt=alt, x=pos.x(), y=pos.y(),
                  pressure=min(1.0, max(0.0, pressure)), tilt_x=float(qev.xTilt()), tilt_y=float(qev.yTilt()),
                  is_tablet=True, is_eraser=eraser, source=widget)
    if prev is not None:
        event.prev_x, event.prev_y = prev
    else:
        event.prev_x, event.prev_y = event.x, event.y
    return event


class InputForwarder(QObject):
    """装在主区域控件上：把鼠标、滚轮、数位板事件转成 Event 交给编辑器的 handle_event。

    编辑器处理了就吞掉事件；数位板事件处理了就不再让 Qt 合成鼠标事件。
    偏好里开了「Alt+左键模拟中键」时在这里换算。
    """

    _MOUSE = (QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick, QEvent.MouseMove)
    _TABLET = (QEvent.TabletPress, QEvent.TabletRelease, QEvent.TabletMove)

    def __init__(self, editor: "Editor", widget: QWidget) -> None:
        super().__init__(widget)
        self._editor = editor
        self._widget = widget
        self._prev: tuple[float, float] | None = None
        self._emulating = False
        widget.setMouseTracking(True)
        widget.setAttribute(Qt.WA_TabletTracking, True)
        widget.setAttribute(Qt.WA_InputMethodEnabled, False)   # 视口里按字母键是快捷键，不是打字
        if widget.focusPolicy() == Qt.NoFocus:
            widget.setFocusPolicy(Qt.ClickFocus)                  # 点视口时让文本框交出焦点
        widget.installEventFilter(self)

    def _emulate_3_button(self) -> bool:
        prefs = getattr(self._editor.app, "prefs", None)
        nav = getattr(prefs, "navigation", None)
        return bool(getattr(nav, "emulate_3_button", False))

    def _tablet_mode(self) -> str:
        app = getattr(self._editor, "app", None)
        mode = getattr(app, "tablet_mode", None)
        return mode() if callable(mode) else "AUTO"

    def _from_wintab(self, event: Event) -> None:
        """鼠标事件：WinTab 开着、笔就在板子上时，补上压感、倾斜、橡皮端（驱动里关了 Windows Ink 时靠它）。"""
        tablet = getattr(getattr(self._editor, "app", None), "wintab", None)
        if tablet is None or not tablet.ok:
            return
        from .wintab import annotate

        annotate(event, tablet.poll())

    def eventFilter(self, obj: QObject, qev: QEvent) -> bool:  # noqa: N802
        t = qev.type()
        if t in self._MOUSE:
            event = mouse_event(qev, self._widget, self._prev)
            if event is not None:
                self._from_wintab(event)
        elif t == QEvent.Wheel:
            event = wheel_event(qev, self._widget)
        elif t in self._TABLET:
            if self._tablet_mode() == "WINTAB":
                qev.ignore()                     # 不用 Windows Ink 的笔事件：让 Qt 合成鼠标事件，压感从 WinTab 补
                return False
            event = tablet_event(qev, self._widget, self._prev)
        else:
            return False
        if event is None or not alive(self._editor):
            return False
        if event.type == LEFTMOUSE and self._emulate_3_button():
            if event.value in (PRESS, DOUBLE) and event.alt:
                event.type, event.alt, self._emulating = MIDDLEMOUSE, False, True
            elif event.value == RELEASE and self._emulating:
                event.type, event.alt, self._emulating = MIDDLEMOUSE, False, False
        if event.type == MOUSEMOVE or event.value in (PRESS, RELEASE, DOUBLE):
            self._prev = (event.x, event.y)
        try:
            handled = bool(self._editor.handle_event(event))
        except Exception:  # noqa: BLE001
            log.exception("编辑器 %s 处理输入时出错", getattr(self._editor, "idname", "?"))
            handled = False
        if t in self._TABLET:
            qev.setAccepted(handled)
        return handled


# ======================================================================
# 键位分发
# ======================================================================

def dispatch_keymaps(app: Any, names: Iterable[str], event: Event, make_context: Callable[[], Any]) -> bool:
    """按顺序在多张键位表里找匹配的操作并执行。

    可用条件不满足或返回 PASS_THROUGH 的，继续找下一条；执行了就返回 True。
    """
    kc = getattr(app, "keyconfig", None)
    if kc is None:
        return False
    ctx = None
    for name in names:
        km = kc.keymaps.get(name)
        if km is None:
            continue
        for item in km.items:
            if not item.matches(event) or ops.get(item.op) is None:
                continue
            if ctx is None:
                ctx = make_context()
            if not ops.poll(item.op, ctx):
                continue
            result = ops.call(item.op, ctx, event, **dict(item.props))
            if result == ops.PASS_THROUGH:
                continue
            return True
    return False


def _unique(names: Iterable[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        if name and name not in out:
            out.append(name)
    return out


def _layout_class() -> Any:
    try:
        from .layout import UILayout
        return UILayout
    except Exception:  # noqa: BLE001
        if not getattr(_layout_class, "_warned", False):
            log.exception("布局模块不可用，标题栏只显示编辑器类型按钮")
            _layout_class._warned = True  # type: ignore[attr-defined]
        return None


# ======================================================================
# 编辑器的四块
# ======================================================================

class _HeaderScroll(QScrollArea):
    """标题栏内容放不下时（区域太窄）不挤成省略号，而是像 Blender 一样能用滚轮左右滚动。"""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setAutoFillBackground(False)
        self.viewport().setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")
        self.setFocusPolicy(Qt.NoFocus)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        bar = self.horizontalScrollBar()
        if bar.maximum() > 0:
            delta = event.angleDelta().y() or event.angleDelta().x()
            bar.setValue(bar.value() - int(delta / 2))
            event.accept()
            return
        event.ignore()


class EditorHeader(QWidget):
    """标题栏：最左是区域的编辑器类型按钮，右边是编辑器自己的内容。右键弹出区域菜单。"""

    def __init__(self, editor: "Editor") -> None:
        super().__init__(editor)
        self._editor = editor
        self._box = QHBoxLayout(self)
        self.lead = QHBoxLayout()
        self.lead.setContentsMargins(0, 0, 0, 0)
        self.lead.setSpacing(0)
        self._box.addLayout(self.lead)
        self.scroll = _HeaderScroll(self)
        self.content = QWidget()
        self.content.setAutoFillBackground(False)
        self.content.setStyleSheet("background: transparent;")
        self.scroll.setWidget(self.content)
        self._box.addWidget(self.scroll, 1)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setFixedHeight(theme.size("header"))
        pad = theme.px(4)
        self._box.setContentsMargins(pad, 0, pad, 0)
        self._box.setSpacing(theme.px(6))
        self.update()

    def set_lead(self, widget: QWidget) -> None:
        self.lead.addWidget(widget, 0, Qt.AlignVCenter)
        widget.show()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.header"))

    def contextMenuEvent(self, event: Any) -> None:  # noqa: N802
        area = self._editor.area
        if area is not None and hasattr(area, "show_header_menu"):
            area.show_header_menu(event.globalPos())
            event.accept()
        else:
            event.ignore()


def _tool_shortcut(app: Any, tool_idname: str) -> str:
    kc = getattr(app, "keyconfig", None)
    if kc is None:
        return ""
    for km in kc.keymaps.values():
        for item in km.items:
            if item.active and item.props.get("tool") == tool_idname:
                return item.shortcut_text()
    return ""


class ToolbarRegion(QWidget):
    """工具栏：一列工具按钮，点哪个就把 app.tool_settings.tool 设成哪个。"""

    def __init__(self, editor: "Editor") -> None:
        super().__init__(editor)
        self._editor = editor
        self._buttons: dict[str, QWidget] = {}
        self._box = QVBoxLayout(self)
        self._box.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.apply_theme()
        self.rebuild()

    def apply_theme(self) -> None:
        self.setFixedWidth(theme.size("toolbar"))
        pad = theme.px(4)
        self._box.setContentsMargins(0, pad, 0, pad)
        self._box.setSpacing(theme.px(2))
        for button in self._buttons.values():
            if isinstance(button, QToolButton):
                side = theme.size("tool_button")
                button.setFixedSize(side, side)
                button.setIconSize(QSize(theme.size("icon.tool"), theme.size("icon.tool")))
            else:
                button.updateGeometry()
        self.update()

    def rebuild(self) -> None:
        for button in self._buttons.values():
            button.hide()
            button.setParent(None)
            button.deleteLater()
        self._buttons.clear()
        app = self._editor.app
        ts = getattr(app, "tool_settings", None)
        self._mode = getattr(ts, "mode", "PAINT") if ts is not None else "PAINT"
        for tool in registry.tools(self._editor.idname):
            if getattr(tool, "mode", self._mode) != self._mode:
                continue
            tip = tool.label or tool.idname
            shortcut = _tool_shortcut(app, tool.idname)
            if shortcut:
                tip += "  " + shortcut
            if tool.description:
                tip += "\n" + tool.description
            button = self._make_button(tool.icon or "tool.brush", tip)
            button.clicked.connect(lambda _checked=False, name=tool.idname: self._choose(name))
            self._box.addWidget(button, 0, Qt.AlignHCenter)
            self._buttons[tool.idname] = button
        self.sync()

    def _make_button(self, icon_name: str, tip: str) -> QWidget:
        try:
            from .widgets import IconButton
            # IconButton 的 size 是逻辑像素，它自己乘界面缩放
            button = IconButton(icon_name, tip, checkable=True, size=theme.SIZES["tool_button"], parent=self)
        except Exception:  # noqa: BLE001
            log.exception("图标按钮不可用，工具栏改用普通按钮")
            button = QToolButton(self)
            side = theme.size("tool_button")
            button.setIcon(icons.icon(icon_name, size=theme.size("icon.tool")))
            button.setIconSize(QSize(theme.size("icon.tool"), theme.size("icon.tool")))
            button.setCheckable(True)
            button.setToolTip(tip)
            button.setFixedSize(side, side)
        button.setFocusPolicy(Qt.NoFocus)
        return button

    def _choose(self, idname: str) -> None:
        ts = getattr(self._editor.app, "tool_settings", None)
        if ts is not None:
            ts.tool = idname
        self.sync()

    def sync(self) -> None:
        ts = getattr(self._editor.app, "tool_settings", None)
        if ts is not None and getattr(self, "_mode", None) not in (None, getattr(ts, "mode", "PAINT")):
            self.rebuild()                         # 换了模式：换一套工具
            return
        active = getattr(ts, "tool", "") if ts is not None else ""
        ctx = None
        for idname, button in self._buttons.items():
            if not alive(button):
                continue
            button.blockSignals(True)
            button.setChecked(idname == active)
            button.blockSignals(False)
            tool = registry.tool(self._editor.idname, idname)
            if tool is not None:
                if ctx is None:
                    ctx = self._editor.context()
                try:
                    button.setEnabled(bool(tool.poll(ctx)))
                except Exception:  # noqa: BLE001
                    button.setEnabled(False)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.panel"))
        painter.fillRect(QRect(self.width() - 1, 0, 1, self.height()), theme.qcolor("line"))


class SidebarGrip(QWidget):
    """侧栏左边的拖动条，拖动改侧栏宽度。"""

    def __init__(self, editor: "Editor") -> None:
        super().__init__(editor)
        self._editor = editor
        self._dragging = False
        self.setCursor(Qt.SplitHCursor)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setFixedWidth(max(3, theme.px(4)))

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.panel"))
        painter.fillRect(QRect(0, 0, 1, self.height()), theme.qcolor("line"))

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._dragging = True
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if not self._dragging:
            return
        body = self._editor.body
        local = body.mapFromGlobal(event.globalPosition().toPoint())
        width = body.width() - local.x() - self.width() // 2
        self._editor.set_sidebar_width_px(width)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            area = self._editor.area
            screen = area.parent_screen() if area is not None and hasattr(area, "parent_screen") else None
            if screen is not None:
                screen.mark_changed()


class SidebarRegion(QWidget):
    """侧栏：按标签页分组的面板，标签竖排在最右边。"""

    def __init__(self, editor: "Editor") -> None:
        super().__init__(editor)
        self._editor = editor
        self._categories: list[str] = []
        self._hosts: dict[str, QWidget] = {}
        self._saved_panels: dict[str, Any] = {}
        self._current = ""
        self._tabs: Any = None
        self._box = QHBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(0)
        self._stack = QStackedWidget(self)
        self._box.addWidget(self._stack, 1)
        self._empty = QLabel("", self._stack)
        self._stack.addWidget(self._empty)
        self.rebuild()

    @property
    def current_category(self) -> str:
        return self._current

    def rebuild(self) -> None:
        editor = self._editor
        categories = registry.panel_categories(editor.idname, "SIDEBAR")
        for name in list(self._hosts):
            if name not in categories:
                host = self._hosts.pop(name)
                self._remember(name, host)
                self._stack.removeWidget(host)
                host.deleteLater()
        for name in categories:
            if name not in self._hosts:
                host = self._make_host(name)
                if host is not None:
                    self._hosts[name] = host
                    self._stack.addWidget(host)
                    saved = self._saved_panels.get(name)
                    if saved is not None and hasattr(host, "set_state"):
                        try:
                            host.set_state(saved)
                        except Exception:  # noqa: BLE001
                            log.exception("恢复侧栏面板状态出错")
        self._categories = [c for c in categories if c in self._hosts]
        self._rebuild_tabs()
        if self._current not in self._hosts:
            self._current = self._categories[0] if self._categories else ""
        self._show_current()

    def _make_host(self, category: str) -> QWidget | None:
        try:
            from .panels import PanelHost
            return PanelHost(self._editor.context, self._editor.idname, "SIDEBAR", category)
        except Exception:  # noqa: BLE001
            log.exception("侧栏面板宿主不可用")
            return None

    def _rebuild_tabs(self) -> None:
        if self._tabs is not None:
            self._tabs.hide()
            self._tabs.setParent(None)
            self._tabs.deleteLater()
            self._tabs = None
        if len(self._categories) < 2:
            return
        items = [(name, name, "", "") for name in self._categories]
        try:
            from .widgets import TabStrip
            tabs = TabStrip(items, vertical=True)
        except Exception:  # noqa: BLE001
            log.exception("标签条不可用")
            return
        tabs.setParent(self)
        signal = getattr(tabs, "currentChanged", None)
        if signal is not None:
            signal.connect(self.set_category)
        self._box.addWidget(tabs, 0)
        tabs.show()
        self._tabs = tabs
        self._sync_tabs()

    def _sync_tabs(self) -> None:
        if self._tabs is None or not self._current:
            return
        setter = getattr(self._tabs, "setCurrent", None)
        if setter is not None:
            self._tabs.blockSignals(True)
            try:
                setter(self._current)
            finally:
                self._tabs.blockSignals(False)

    def set_category(self, name: Any) -> None:
        name = str(name)
        if name in self._hosts and name != self._current:
            self._current = name
            self._show_current()
            self._sync_tabs()

    def _show_current(self) -> None:
        host = self._hosts.get(self._current)
        self._stack.setCurrentWidget(host if host is not None else self._empty)

    def refresh(self) -> None:
        host = self._hosts.get(self._current)
        if host is not None and hasattr(host, "refresh"):
            try:
                host.refresh()
            except Exception:  # noqa: BLE001
                log.exception("侧栏刷新出错")

    def _remember(self, name: str, host: QWidget) -> None:
        if hasattr(host, "state"):
            try:
                self._saved_panels[name] = host.state()
            except Exception:  # noqa: BLE001
                pass

    def save_state(self) -> dict:
        panels = dict(self._saved_panels)
        for name, host in self._hosts.items():
            if hasattr(host, "state"):
                try:
                    panels[name] = host.state()
                except Exception:  # noqa: BLE001
                    continue
        data: dict[str, Any] = {"sidebar_tab": self._current}
        if panels:
            data["panels"] = panels
        return data

    def load_state(self, state: dict) -> None:
        panels = state.get("panels")
        if isinstance(panels, dict):
            self._saved_panels.update(panels)
            for name, host in self._hosts.items():
                if name in panels and hasattr(host, "set_state"):
                    try:
                        host.set_state(panels[name])
                    except Exception:  # noqa: BLE001
                        log.exception("恢复侧栏面板状态出错")
        tab = state.get("sidebar_tab")
        if isinstance(tab, str) and tab in self._hosts:
            self.set_category(tab)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.panel"))


# ======================================================================
# 编辑器基类
# ======================================================================

class Editor(QWidget):
    """所有编辑器的基类。见模块说明。"""

    idname = ""
    label = ""
    icon = ""
    category = "通用"
    order = 100
    description = ""
    has_toolbar = False          # 左侧工具栏（registry.tools(idname)）
    has_sidebar = False          # 右侧侧栏（registry.panels(idname, "SIDEBAR") 按 category 分标签）
    toolbar_default = True       # 新建时工具栏是否显示
    sidebar_default = False      # 新建时侧栏是否显示
    keymaps: list[str] = []      # 本编辑器启用的键位表名，靠前的优先

    def __init__(self, app: Any, area: Any = None) -> None:
        super().__init__(area)
        self.app = app
        self.area = area
        self._toolbar_visible = bool(self.toolbar_default)
        self._sidebar_visible = bool(self.sidebar_default)
        self._sidebar_width = float(theme.SIZES.get("sidebar", 280))     # 逻辑像素，不含界面缩放
        self._header_ui: Any = None
        self._header_drawn = False
        self._shown = False
        self._forwarders: list[InputForwarder] = []
        self._press: Event | None = None      # 没有操作接的鼠标按下，等着变成单击或拖动

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.header = EditorHeader(self)
        root.addWidget(self.header)
        self.body = QWidget(self)
        root.addWidget(self.body, 1)
        body = QHBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self._body_layout = body

        self.toolbar: ToolbarRegion | None = ToolbarRegion(self) if self.has_toolbar else None
        if self.toolbar is not None:
            self.toolbar.setParent(self.body)
            body.addWidget(self.toolbar)
        main = self.build_main()
        if main is None:
            main = QWidget()
        main.setParent(self.body)
        self.main = main
        body.addWidget(main, 1)
        self.sidebar_grip: SidebarGrip | None = None
        self.sidebar: SidebarRegion | None = None
        if self.has_sidebar:
            self.sidebar_grip = SidebarGrip(self)
            self.sidebar_grip.setParent(self.body)
            body.addWidget(self.sidebar_grip)
            self.sidebar = SidebarRegion(self)
            self.sidebar.setParent(self.body)
            body.addWidget(self.sidebar)
        self._apply_regions()

        theme.changed.connect(self._on_theme_changed)
        registry.changed.connect(self._on_registry_changed)
        ts = getattr(app, "tool_settings", None)
        changed = getattr(ts, "changed", None)
        if changed is not None and hasattr(changed, "connect"):
            changed.connect(self._on_tool_settings_changed)

    # ---- 子类实现 ----
    def build_main(self) -> QWidget:
        """主区域控件。"""
        return QWidget()

    def draw_header(self, layout: Any, ctx: Any) -> None:
        """标题栏里编辑器类型按钮右边的内容。"""

    # ---- 上下文 ----
    @property
    def wm(self) -> Any:
        return getattr(self.app, "wm", None)

    def context(self, event: Any = None) -> Context:
        wm = self.wm
        if wm is not None and hasattr(wm, "context") and self.area is not None:
            ctx = wm.context(area=self.area, event=event)
            if ctx.editor is not self:
                ctx.editor = self
            return ctx
        screen = self.area.parent_screen() if self.area is not None and hasattr(self.area, "parent_screen") else None
        return Context(app=self.app, wm=wm, window=self.window(), screen=screen, area=self.area, editor=self,
                       event=event)

    # ---- 键位 ----
    def active_keymaps(self) -> list[str]:
        """本编辑器当前启用的键位表。子类可以按模式增减。"""
        return list(self.keymaps)

    def tool_keymap(self) -> str:
        """当前工具额外启用的键位表名。"""
        ts = getattr(self.app, "tool_settings", None)
        name = getattr(ts, "tool", "") if ts is not None else ""
        tool = registry.tool(self.idname, name) if name and self.idname else None
        return getattr(tool, "keymap", "") if tool is not None else ""

    def keymap_names(self) -> list[str]:
        """查找顺序：活动工具 → 本编辑器 → Screen → Window。"""
        return _unique([self.tool_keymap(), *self.active_keymaps(), "Screen", "Window"])

    def handle_event(self, event: Event) -> bool:
        """处理一个输入事件。有模态操作时先交给它，然后按键位表找操作执行。处理了返回 True。

        和 Blender 一样：鼠标按下时没有操作接，就记下这次按下；之后拖过阈值发一次「拖动」（CLICK_DRAG，
        坐标是按下的位置，修饰键也按按下时的算），没拖就松开则在松开之后发一次「单击」（CLICK）。"""
        wm = self.wm
        if wm is not None and not event.__dict__.get(MODAL_SEEN):
            event.__dict__[MODAL_SEEN] = True
            has_modal = getattr(wm, "has_modal", None)
            if has_modal is not None and has_modal() and wm.modal_handle(event, self):
                if event.value in (PRESS, DOUBLE):
                    self._press = None
                return True
        press = self._press
        if press is not None:
            if event.type == MOUSEMOVE:
                if self._past_drag_threshold(press, event):
                    self._press = None
                    drag = copy.copy(press)
                    drag.value = CLICK_DRAG
                    drag.__dict__[MODAL_SEEN] = True
                    if self.dispatch_keymap(drag):
                        # 拖动开始的操作（框选、移动）接着收到这一次移动
                        if wm is not None and getattr(wm, "has_modal", lambda: False)():
                            wm.modal_handle(event, self)
                        return True
            elif event.type == press.type and event.value == RELEASE:
                self._press = None
                handled = self.dispatch_keymap(event)
                click = copy.copy(event)
                click.value = CLICK
                click.ctrl, click.shift, click.alt = press.ctrl, press.shift, press.alt
                return self.dispatch_keymap(click) or handled
            elif event.value in (PRESS, DOUBLE):
                self._press = None
        if self.dispatch_keymap(event):
            return True
        if event.value == DOUBLE:
            # 没有操作接双击时，当作普通按下再找一遍（和 Blender 相同）
            retry = copy.copy(event)
            retry.value = PRESS
            if self.dispatch_keymap(retry):
                return True
        if event.value in (PRESS, DOUBLE) and event.type in (LEFTMOUSE, MIDDLEMOUSE, RIGHTMOUSE):
            self._press = copy.copy(event)
            self._press.value = PRESS
            return True
        return False

    def _past_drag_threshold(self, press: Event, event: Event) -> bool:
        nav = getattr(getattr(self.app, "prefs", None), "navigation", None)
        if press.is_tablet:
            limit = float(getattr(nav, "drag_threshold_tablet", 10))
        else:
            limit = float(getattr(nav, "drag_threshold", 3))
        return max(abs(event.x - press.x), abs(event.y - press.y)) > limit

    def cancel_press(self) -> None:
        """忘掉记着的按下（例如按下后弹出了菜单）。"""
        self._press = None

    def dispatch_keymap(self, event: Event) -> bool:
        return dispatch_keymaps(self.app, self.keymap_names(), event, lambda: self.context(event))

    def localize(self, event: Event, global_pos: Any = None) -> Event:
        """把事件坐标换算到主区域控件内（键盘事件用当前鼠标位置）。"""
        target = self.main if alive(getattr(self, "main", None)) else self
        pos = global_pos if global_pos is not None else QCursor.pos()
        if not isinstance(pos, QPoint):
            pos = pos.toPoint()
        local = target.mapFromGlobal(pos)
        event.x, event.y = float(local.x()), float(local.y())
        event.prev_x, event.prev_y = event.x, event.y
        if event.source is None:
            event.source = target
        return event

    def install_input(self, widget: QWidget) -> InputForwarder:
        """让 widget 的鼠标、滚轮、数位板事件经由 handle_event 处理。"""
        forwarder = InputForwarder(self, widget)
        self._forwarders.append(forwarder)
        return forwarder

    # ---- 状态 ----
    def save_state(self) -> dict:
        state: dict[str, Any] = {}
        if self.has_toolbar:
            state["toolbar"] = self._toolbar_visible
        if self.has_sidebar:
            state["sidebar"] = self._sidebar_visible
            state["sidebar_width"] = round(self._sidebar_width, 1)
            if self.sidebar is not None:
                state.update(self.sidebar.save_state())
        return state

    def load_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if self.has_toolbar and "toolbar" in state:
            self._toolbar_visible = bool(state["toolbar"])
        if self.has_sidebar:
            if "sidebar" in state:
                self._sidebar_visible = bool(state["sidebar"])
            try:
                width = float(state.get("sidebar_width", self._sidebar_width))
                if width == width:   # 不是 NaN
                    self._sidebar_width = min(4000.0, max(80.0, width))
            except (TypeError, ValueError):
                pass
            if self.sidebar is not None:
                self.sidebar.load_state(state)
        self._apply_regions()

    # ---- 工具栏、侧栏 ----
    @property
    def toolbar_visible(self) -> bool:
        return self.has_toolbar and self._toolbar_visible

    @toolbar_visible.setter
    def toolbar_visible(self, value: bool) -> None:
        self._toolbar_visible = bool(value)
        self._apply_regions()

    @property
    def sidebar_visible(self) -> bool:
        return self.has_sidebar and self._sidebar_visible

    @sidebar_visible.setter
    def sidebar_visible(self, value: bool) -> None:
        self._sidebar_visible = bool(value)
        self._apply_regions()

    @property
    def sidebar_width(self) -> int:
        """侧栏当前宽度（像素，已乘界面缩放）。"""
        return theme.px(self._sidebar_width)

    def set_sidebar_width_px(self, width: int) -> None:
        low = theme.px(160)
        high = max(low, self.body.width() - theme.px(120))
        width = max(low, min(high, int(width)))
        self._sidebar_width = width / max(0.01, theme.scale())
        if self.sidebar is not None:
            self.sidebar.setFixedWidth(width)

    def _apply_regions(self) -> None:
        if self.toolbar is not None:
            self.toolbar.setVisible(self.toolbar_visible)
        if self.sidebar is not None:
            self.sidebar.setFixedWidth(theme.px(self._sidebar_width))
            self.sidebar.setVisible(self.sidebar_visible)
            if self.sidebar_grip is not None:
                self.sidebar_grip.setVisible(self.sidebar_visible)

    # ---- 刷新 ----
    def refresh_header(self) -> None:
        """重画标题栏内容。"""
        self._header_drawn = True
        ui = self._header_ui
        ctx = self.context()
        if ui is None:
            cls = _layout_class()
            if cls is None:
                return
            try:
                ui = cls(self.header.content, ctx, direction="ROW")
            except Exception:  # noqa: BLE001
                log.exception("编辑器 %s 的标题栏布局创建失败", self.idname)
                return
            try:
                ui.use_property_split = False
            except Exception:  # noqa: BLE001
                pass
            self._header_ui = ui
        else:
            try:
                ui.clear()
            except Exception:  # noqa: BLE001
                log.exception("清空标题栏出错")
            if hasattr(ui, "ctx"):
                try:
                    ui.ctx = ctx
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.draw_header(ui, ctx)
        except Exception:  # noqa: BLE001
            log.exception("编辑器 %s 的标题栏绘制出错", self.idname)

    def refresh(self) -> None:
        """数据变了：重画标题栏、同步工具栏、刷新侧栏。子类覆盖时调用基类。"""
        self.refresh_header()
        if self.toolbar is not None:
            self.toolbar.sync()
        if self.sidebar is not None and self.sidebar_visible:
            self.sidebar.refresh()

    def on_show(self) -> None:
        """编辑器出现在屏幕上时调用。"""

    def on_hide(self) -> None:
        """编辑器被隐藏（换工作区、被最大化的区域盖住、换成别的编辑器）时调用。"""

    def dispose(self) -> None:
        """编辑器被销毁前调用，子类在这里释放显卡资源等。"""

    def attach_type_button(self, button: QWidget) -> None:
        self.header.set_lead(button)

    # ---- Qt ----
    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._header_drawn:
            self.refresh_header()
        if not self._shown:
            self._shown = True
            try:
                self.on_show()
            except Exception:  # noqa: BLE001
                log.exception("编辑器 %s 的 on_show 出错", self.idname)

    def hideEvent(self, event: Any) -> None:  # noqa: N802
        super().hideEvent(event)
        if self._shown:
            self._shown = False
            try:
                self.on_hide()
            except Exception:  # noqa: BLE001
                log.exception("编辑器 %s 的 on_hide 出错", self.idname)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.area"))

    def _on_theme_changed(self) -> None:
        if not alive(self):
            return
        self.header.apply_theme()
        if self.toolbar is not None:
            self.toolbar.apply_theme()
        if self.sidebar_grip is not None:
            self.sidebar_grip.apply_theme()
        self._apply_regions()
        self.update()

    def _on_registry_changed(self, kind: str = "") -> None:
        if not alive(self):
            return
        if kind == "tools" and self.toolbar is not None:
            self.toolbar.rebuild()
        elif kind == "panels" and self.sidebar is not None:
            self.sidebar.rebuild()

    def _on_tool_settings_changed(self, name: str = "") -> None:
        if not alive(self):
            return
        if name == "tool" and self.toolbar is not None:
            self.toolbar.sync()


class UnavailableEditor(Editor):
    """布局里引用了没注册（或创建失败）的编辑器时用它顶上。记住原来的类型和状态，存盘时原样写回。"""

    label = "不可用"
    icon = "warning"
    category = ""

    def __init__(self, app: Any, area: Any = None, missing_idname: str = "", missing_state: dict | None = None,
                 reason: str = "") -> None:
        self.missing_idname = str(missing_idname or "")
        self.missing_state = copy.deepcopy(missing_state) if isinstance(missing_state, dict) else {}
        self.reason = reason
        super().__init__(app, area)

    def build_main(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setAlignment(Qt.AlignCenter)
        box.setSpacing(theme.px(6))
        mark = QLabel(page)
        side = theme.px(32)
        mark.setPixmap(icons.pixmap("warning", theme.color("text.faint"), side, self.devicePixelRatioF()))
        mark.setAlignment(Qt.AlignCenter)
        title = QLabel("这种编辑器不可用", page)
        title.setProperty("role", "title")
        title.setAlignment(Qt.AlignCenter)
        box.addWidget(mark)
        box.addWidget(title)
        if self.missing_idname:
            name = QLabel(self.missing_idname, page)
            name.setProperty("role", "faint")
            name.setAlignment(Qt.AlignCenter)
            box.addWidget(name)
        return page

    def save_state(self) -> dict:
        return copy.deepcopy(self.missing_state)

    def load_state(self, state: dict) -> None:
        if isinstance(state, dict) and state:
            self.missing_state = copy.deepcopy(state)
