"""窗口管理器：事件路由、模态操作栈、状态栏消息、数据变更通知、上下文、弹出窗口。

按键路由（程序级事件过滤器）：
1. 有弹出菜单或别的程序级对话框时不管，交给它们；
2. 栈上有模态操作时，先把按键转成 Event 交给栈顶操作的 modal(ctx, event)；
3. 焦点在文本输入控件（QLineEdit、QTextEdit、QPlainTextEdit、QAbstractSpinBox、可编辑的 QComboBox）里时，
   不带 Ctrl / Alt 的按键、以及复制粘贴撤销这类标准编辑快捷键，放行给控件；
4. 否则找鼠标所在的区域，交给它的编辑器 handle_event(event)，处理了就吞掉；
   鼠标不在任何区域上时，只查 "Screen" 和 "Window" 两张键位表。

鼠标和数位板事件由编辑器主区域控件自己转换后调 handle_event（见 editor.install_input），
handle_event 里同样先给模态操作。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import shiboken6
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QSize, Qt, QTimer
from PySide6.QtGui import QCursor, QKeySequence
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QPlainTextEdit, QTextEdit,
                               QWidget)

from ..core import ops
from ..core.context import Context
from ..core.signals import Signal
from ..ops import screen_ops  # noqa: F401  注册屏幕操作
from . import theme
from .area import AreaWidget, Screen
from .editor import MODAL_SEEN, alive, dispatch_keymaps, key_event

log = logging.getLogger("splender.wm")

_TEXT_TYPES = (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)
#: 文本框有焦点时交给文本框自己的标准编辑快捷键
_TEXT_KEYS = (
    QKeySequence.StandardKey.Undo, QKeySequence.StandardKey.Redo, QKeySequence.StandardKey.Copy,
    QKeySequence.StandardKey.Cut, QKeySequence.StandardKey.Paste, QKeySequence.StandardKey.SelectAll,
    QKeySequence.StandardKey.MoveToNextWord, QKeySequence.StandardKey.MoveToPreviousWord,
    QKeySequence.StandardKey.SelectNextWord, QKeySequence.StandardKey.SelectPreviousWord,
    QKeySequence.StandardKey.DeleteStartOfWord, QKeySequence.StandardKey.DeleteEndOfWord,
    QKeySequence.StandardKey.MoveToStartOfDocument, QKeySequence.StandardKey.MoveToEndOfDocument,
    QKeySequence.StandardKey.SelectStartOfDocument, QKeySequence.StandardKey.SelectEndOfDocument,
)
_LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "OPERATOR": logging.INFO, "WARNING": logging.WARNING,
               "ERROR": logging.ERROR}


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if a is None or b is None:
        return False
    try:
        return shiboken6.getCppPointer(a)[0] == shiboken6.getCppPointer(b)[0]
    except Exception:  # noqa: BLE001
        return False


def _to_point(pos: Any) -> QPoint:
    if isinstance(pos, QPoint):
        return QPoint(pos)
    if isinstance(pos, QPointF):
        return pos.toPoint()
    return QPoint(int(pos[0]), int(pos[1]))


def is_text_input(widget: Any) -> bool:
    """焦点控件是不是在打字的文本输入框（只读的不算）。"""
    if widget is None or not alive(widget):
        return False
    if isinstance(widget, QComboBox):
        return widget.isEditable()
    if isinstance(widget, _TEXT_TYPES):
        read_only = getattr(widget, "isReadOnly", None)
        try:
            return not (read_only() if callable(read_only) else False)
        except Exception:  # noqa: BLE001
            return True
    parent = widget.parentWidget()
    return isinstance(parent, QComboBox) and parent.isEditable()


def _is_text_shortcut(qev: Any) -> bool:
    for key in _TEXT_KEYS:
        try:
            if qev.matches(key):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False


class WindowManager(QObject):
    """见模块说明。"""

    def __init__(self, app: Any, *, install_filter: bool = True) -> None:
        super().__init__()
        self.app = app
        try:
            if getattr(app, "wm", None) is None:
                app.wm = self
        except Exception:  # noqa: BLE001
            pass
        self.windows: list[QWidget] = []
        self.main_window: Any = None
        self._modal: list[tuple[Any, Any]] = []
        self.last_input = time.perf_counter()     # 最近一次键盘、鼠标、数位板输入
        self._pending: set[str] = set()
        self._flush_scheduled = False
        #: 每次合并刷新之后发出 (tags: frozenset[str])。界面里需要跟着刷新的控件连这个。
        self.notified = Signal()
        #: report 时发出 (text, level)
        self.reported = Signal()
        #: 合并后实际刷新的次数
        self.flush_count = 0
        self._installed = False
        if install_filter:
            self.install()

    # ---- 安装与关闭 ----
    def install(self) -> None:
        app = QApplication.instance()
        if app is not None and not self._installed:
            app.installEventFilter(self)
            self._installed = True

    def shutdown(self) -> None:
        self.modal_cancel_all()
        app = QApplication.instance()
        if app is not None and self._installed:
            app.removeEventFilter(self)
        self._installed = False

    # ---- 窗口 ----
    def register_window(self, window: QWidget, main: bool = False) -> None:
        if not any(_same(window, w) for w in self.windows):
            self.windows.append(window)
        if main or self.main_window is None and hasattr(window, "workspace_names"):
            self.main_window = window

    def unregister_window(self, window: QWidget) -> None:
        self.windows = [w for w in self.windows if not _same(w, window) and alive(w)]
        if _same(window, self.main_window):
            self.main_window = None

    def close_popouts(self) -> None:
        for window in list(self.windows):
            if not _same(window, self.main_window) and alive(window):
                window.close()

    def _owns(self, window: Any) -> bool:
        if window is None or not alive(window):
            return False
        if any(_same(window, w) for w in self.windows):
            return True
        try:
            return window.findChild(Screen) is not None
        except Exception:  # noqa: BLE001
            return False

    def active_window(self) -> Any:
        window = QApplication.activeWindow()
        if window is not None and self._owns(window.window()):
            return window.window()
        return self.main_window

    def _screen_of(self, window: Any) -> Screen | None:
        if window is None or not alive(window):
            return None
        getter = getattr(window, "current_screen", None)
        if callable(getter):
            return getter()
        try:
            return window.findChild(Screen)
        except Exception:  # noqa: BLE001
            return None

    def active_screen(self) -> Screen | None:
        """当前活动窗口里显示的 Screen。"""
        return self._screen_of(self.active_window())

    def screens(self) -> list[Screen]:
        """各窗口当前显示的 Screen。"""
        out = []
        for window in self.windows:
            screen = self._screen_of(window)
            if screen is not None:
                out.append(screen)
        return out

    def area_under_mouse(self, global_pos: Any = None, window: Any = None) -> AreaWidget | None:
        """鼠标所在的区域。给了 window 时只认这个窗口里的。"""
        pos = QCursor.pos() if global_pos is None else _to_point(global_pos)
        widget = QApplication.widgetAt(pos)
        area = None
        while widget is not None:
            if isinstance(widget, AreaWidget):
                area = widget
                break
            if isinstance(widget, Screen):
                area = widget.area_at(pos)
                break
            if widget.isWindow():
                break
            widget = widget.parentWidget()
        if area is not None and window is not None and not _same(area.window(), window):
            return None
        return area

    # ---- 上下文 ----
    def context(self, area: Any = None, event: Any = None, *, window: Any = None, use_mouse: bool = True) -> Context:
        """组装上下文。area 为空时取鼠标所在的区域。"""
        if area is None and use_mouse:
            area = self.area_under_mouse(window=window)
        if area is not None and alive(area):
            screen = area.parent_screen()
            win = area.window()
            editor = area.editor
        else:
            area = None
            editor = None
            win = window if window is not None else self.active_window()
            screen = self._screen_of(win)
        return Context(app=self.app, wm=self, window=win, screen=screen, area=area, editor=editor, region="WINDOW",
                       event=event)

    # ---- 事件路由 ----
    _INPUT_EVENTS = (QEvent.KeyPress, QEvent.KeyRelease, QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                     QEvent.MouseMove, QEvent.Wheel, QEvent.TabletPress, QEvent.TabletMove, QEvent.TabletRelease)

    def eventFilter(self, obj: QObject, qev: QEvent) -> bool:  # noqa: N802
        kind = qev.type()
        if kind in self._INPUT_EVENTS:
            self.last_input = time.perf_counter()
        if kind == QEvent.Wheel and qev.modifiers() & Qt.ControlModifier and isinstance(obj, QWidget) \
                and obj.isWindow() is False and not self._modal:
            try:
                if self._hover_wheel(qev):
                    return True
            except Exception:  # noqa: BLE001
                log.exception("Ctrl+滚轮改值出错")
        if kind in (QEvent.KeyPress, QEvent.KeyRelease):
            # 单独按下、松开 Alt 不交给菜单栏（否则会进菜单的键盘导航，后面的快捷键都被它吃掉）。
            # Alt 作为修饰键的状态照常带在其它按键和鼠标事件上。
            if qev.key() == Qt.Key_Alt and not is_text_input(QApplication.focusWidget()):
                return True
            try:
                return self._route_key(obj, qev)
            except Exception:  # noqa: BLE001
                log.exception("按键分发出错")
        return False

    def _route_key(self, obj: Any, qev: Any) -> bool:
        if not isinstance(obj, QWidget):
            return False
        # 一次按键会从焦点控件一路传给父控件，只在第一站处理
        target = QWidget.keyboardGrabber() or QApplication.focusWidget()
        if target is None:
            if not obj.isWindow():
                return False
        elif not _same(obj, target):
            return False
        if QApplication.activePopupWidget() is not None:
            return False
        modal_widget = QApplication.activeModalWidget()
        if modal_widget is not None and not self._owns(modal_widget):
            return False
        window = obj.window()
        if not self._owns(window):
            return False
        event = key_event(qev)
        if event is None:
            return False
        if self._modal and self.modal_handle(event):
            return True
        event.__dict__[MODAL_SEEN] = True
        if is_text_input(QApplication.focusWidget()):
            if not (event.ctrl or event.alt) or _is_text_shortcut(qev):
                return False
        if qev.type() == QEvent.KeyPress and self._hover_property_key(qev):
            return True
        area = self.area_under_mouse(window=window)
        if area is not None and alive(area.editor):
            editor = area.editor
            editor.localize(event)
            return bool(editor.handle_event(event))
        return self.dispatch_window_event(event, window)

    # ---- 鼠标停在属性上时的快捷键（照 Blender）----
    @staticmethod
    def hover_binding():
        """鼠标下面那个属性控件的绑定（没有就是 None）。"""
        widget = QApplication.widgetAt(QCursor.pos())
        while widget is not None:
            store = getattr(widget, "_splender_bindings", None)
            if store:
                for binding in store:
                    if binding.alive and binding.objects() is not None:
                        return binding
            widget = widget.parentWidget()
        return None

    def _hover_property_key(self, qev: Any) -> bool:
        """Ctrl+C / Ctrl+V 复制粘贴数值，Backspace 恢复默认值，减号把数取反，颜色上按 E 用吸管。"""
        key = qev.key()
        if key not in (Qt.Key_C, Qt.Key_V, Qt.Key_Backspace, Qt.Key_Minus, Qt.Key_E):
            return False
        mods = qev.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier | Qt.AltModifier)
        binding = self.hover_binding()
        if binding is None:
            return False
        kind = getattr(binding.prop, "kind", "")
        if key == Qt.Key_C and mods == Qt.ControlModifier:
            binding.copy_value()
            self.report("已复制：%s" % binding.format_value(binding.read()))
            return True
        if key == Qt.Key_V and mods == Qt.ControlModifier:
            if binding.paste_value():
                self.notify("property")
            return True
        if mods:
            return False
        if key == Qt.Key_Backspace:
            binding.reset_default()
            self.notify("property")
            return True
        if key == Qt.Key_Minus and kind in ("FLOAT", "INT"):
            binding.write(-binding.read())
            self.notify("property")
            return True
        if key == Qt.Key_E and kind == "COLOR":
            from . import eyedropper

            eyedropper.start(binding)
            return True
        return False

    def _hover_wheel(self, qev: Any) -> bool:
        """Ctrl+滚轮：鼠标下的下拉框换到上一项/下一项，数值框按步长加减。"""
        from .widgets import EnumField, NumberField

        binding = self.hover_binding()
        if binding is None:
            return False
        objects = binding.objects() or []
        steps = qev.angleDelta().y()
        if not steps:
            return False
        sign = 1 if steps > 0 else -1
        for widget in objects:
            if isinstance(widget, EnumField):
                items = widget.items()
                idents = [item[0] for item in items]
                if not idents:
                    return False
                index = idents.index(widget.current()) if widget.current() in idents else 0
                widget.setCurrent(idents[(index - sign) % len(idents)], emit=True)
                return True
            if isinstance(widget, NumberField):
                widget._bump(sign * widget.increment())
                return True
        return False

    def dispatch_window_event(self, event: Any, window: Any = None) -> bool:
        """鼠标不在任何区域上时的按键：只查 Screen 和 Window 键位表，上下文里没有区域。"""
        window = window if window is not None else self.active_window()
        screen = self._screen_of(window)

        def make() -> Context:
            return Context(app=self.app, wm=self, window=window, screen=screen, area=None, editor=None, event=event)

        return dispatch_keymaps(self.app, ("Screen", "Window"), event, make)

    # ---- 模态操作栈 ----
    def modal_push(self, op: Any, ctx: Any) -> None:
        """操作的 invoke 返回 RUNNING_MODAL 后由 ops.call 调用。之后的输入先交给它的 modal。"""
        self._modal.append((op, ctx))

    def has_modal(self) -> bool:
        return bool(self._modal)

    def modal_ops(self) -> list[Any]:
        return [op for op, _ctx in self._modal]

    def modal_handle(self, event: Any, editor: Any = None) -> bool:
        """把事件交给模态操作，从栈顶往下。处理了返回 True；都 PASS_THROUGH 时返回 False。"""
        for index in range(len(self._modal) - 1, -1, -1):
            if index >= len(self._modal):
                continue
            op, ctx = self._modal[index]
            try:
                local = ctx.copy(event=event) if hasattr(ctx, "copy") else ctx
                result = op.modal(local, event)
            except Exception:  # noqa: BLE001
                log.exception("模态操作 %s 出错", getattr(op, "idname", "?"))
                self._remove_modal(op)
                try:
                    op.cancel(ctx)
                except Exception:  # noqa: BLE001
                    log.exception("模态操作取消时出错")
                self.report("操作中止：%s" % (getattr(op, "label", "") or getattr(op, "idname", "")), "ERROR")
                return True
            if isinstance(result, (set, frozenset)):
                result = next(iter(result)) if result else ops.FINISHED
            result = result or ops.FINISHED
            if result == ops.RUNNING_MODAL:
                return True
            if result == ops.PASS_THROUGH:
                continue
            self._remove_modal(op)
            self.notify("modal")
            try:
                ops.finished.emit(op, ctx, result)
            except Exception:  # noqa: BLE001
                log.exception("操作结束的通知出错")
            return True
        return False

    def _remove_modal(self, op: Any) -> None:
        self._modal = [(o, c) for o, c in self._modal if o is not op]

    def modal_cancel_all(self) -> None:
        """打断所有模态操作（关窗口时）。"""
        items, self._modal = self._modal, []
        for op, ctx in reversed(items):
            try:
                op.cancel(ctx)
            except Exception:  # noqa: BLE001
                log.exception("模态操作取消时出错")

    # ---- 消息与通知 ----
    def report(self, text: str, level: str = "INFO") -> None:
        """在状态栏显示几秒并写日志。level：INFO / WARNING / ERROR。"""
        level = str(level or "INFO").upper()
        log.log(_LOG_LEVELS.get(level, logging.INFO), "%s", text)
        window = self.main_window
        if window is not None and alive(window) and hasattr(window, "show_report"):
            window.show_report(str(text), level)
        self.reported.emit(str(text), level)

    def notify(self, tag: Any = "") -> None:
        """数据变了。同一轮事件循环里的多次通知合并成一次刷新。"""
        if isinstance(tag, str):
            self._pending.add(tag)
        else:
            try:
                self._pending.update(str(t) for t in tag)
            except TypeError:
                self._pending.add(str(tag))
        if not self._flush_scheduled:
            self._flush_scheduled = True
            QTimer.singleShot(0, self._flush)

    def _flush(self) -> None:
        self._flush_scheduled = False
        tags = frozenset(self._pending)
        self._pending.clear()
        self.flush_count += 1
        for window in list(self.windows):
            if not alive(window):
                continue
            screen = self._screen_of(window)
            if screen is not None:
                screen.refresh()
            hook = getattr(window, "on_notify", None)
            if callable(hook):
                try:
                    hook(tags)
                except Exception:  # noqa: BLE001
                    log.exception("窗口刷新出错")
        self.notified.emit(tags)

    # ---- 状态栏 ----
    def set_hint(self, text: str) -> None:
        if alive(self.main_window):
            self.main_window.set_hint(text)

    def set_progress(self, text: str, fraction: float | None) -> None:
        if alive(self.main_window):
            self.main_window.set_progress(text, fraction)

    def set_stats(self, text: str) -> None:
        if alive(self.main_window):
            self.main_window.set_stats(text)

    def save_workspaces(self) -> None:
        """把各工作区的当前布局写进 prefs.workspaces。"""
        if alive(self.main_window) and hasattr(self.main_window, "store_workspaces"):
            self.main_window.store_workspaces()

    # ---- 弹出窗口 ----
    def popout(self, area: AreaWidget) -> Any:
        """把区域复制到一个新的顶层窗口：只有一个区域，编辑器类型和状态与原区域相同。"""
        from .window import PopoutWindow

        if area is None or not alive(area):
            return None
        data = area.to_dict()
        data.pop("stash", None)
        window = PopoutWindow(self.app, self, data)
        width = max(area.width(), theme.px(360))
        height = max(area.height(), theme.px(260))
        window.resize(QSize(width, height))
        window.move(area.mapToGlobal(QPoint(0, 0)) + QPoint(theme.px(32), theme.px(32)))
        window.show()
        return window
