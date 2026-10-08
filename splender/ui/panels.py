"""面板宿主：把注册给某个编辑器、某个区的面板依次画成可折叠的 Foldout，放进一个可滚动的区域。

    host = PanelHost(lambda: editor.context(), "VIEW_3D", "SIDEBAR", category="笔刷")
    host.refresh()                 # 数据变了：重新判断 poll 并重画
    state = host.state()           # 各面板的折叠状态和滚动位置，随编辑器状态保存
    host.set_state(state)

正在拖动数值、输入文字或开着取色浮层时，refresh() 推迟到编辑结束再做，免得把正在用的控件删掉。
"""
from __future__ import annotations

import logging
import weakref
from functools import partial
from typing import Any, Callable

import shiboken6
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QFrame, QScrollArea, QVBoxLayout, QWidget

from ..core import registry
from . import theme, widgets
from .layout import UILayout
from .widgets import Foldout

log = logging.getLogger("splender.ui")


def panel_key(cls: type) -> str:
    """面板的稳定标识：有 idname 用 idname，否则用类的完整名字。"""
    return getattr(cls, "idname", "") or "%s.%s" % (cls.__module__, cls.__qualname__)


class _Entry:
    """一个正在显示的面板。"""

    __slots__ = ("key", "cls", "panel", "foldout", "layout", "header_host", "header_layout")

    def __init__(self, key: str, cls: type, panel: Any, foldout: Foldout) -> None:
        self.key = key
        self.cls = cls
        self.panel = panel
        self.foldout = foldout
        self.layout: UILayout | None = None
        self.header_host: QWidget | None = None
        self.header_layout: UILayout | None = None


class PanelHost(QScrollArea):
    """可滚动的面板列。ctx_provider 是返回 Context 的函数（也可以直接给一个 Context）。"""

    def __init__(self, ctx_provider: Callable[[], Any] | Any, space: str, region: str = "SIDEBAR",
                 category: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctx_provider = ctx_provider
        self.space = space
        self.region = region
        self.category = category
        self._entries: dict[str, _Entry] = {}
        self._open: dict[str, bool] = {}
        self._pending = False
        self._queued = False
        self._restore_scroll: int | None = None
        self.setFrameShape(QFrame.NoFrame)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viewport().setAutoFillBackground(False)
        self._content = QWidget()
        self._content.setAutoFillBackground(False)
        self._column = QVBoxLayout(self._content)
        gap = theme.size("gap")
        self._column.setContentsMargins(gap, gap, gap, theme.size("pad"))
        self._column.setSpacing(gap)
        self._column.addStretch(1)
        self.setWidget(self._content)
        registry.changed.connect(self._on_registry_changed)
        widgets.edit_state_changed.connect(self._on_edit_state_changed)
        self.refresh()

    # ---- 上下文 ----
    def context(self) -> Any:
        provider = self._ctx_provider
        try:
            return provider() if callable(provider) else provider
        except Exception:  # noqa: BLE001
            log.exception("面板宿主取上下文出错")
            return None

    def set_category(self, category: str | None) -> None:
        """切换显示哪一个标签页的面板（侧栏分类）。"""
        if category != self.category:
            self.category = category
            self.refresh()

    # ---- 刷新 ----
    def refresh(self) -> None:
        """重新判断每个面板的 poll，按顺序重画通过的面板。编辑进行中时推迟到编辑结束。"""
        if not shiboken6.isValid(self):
            return
        if widgets.ui_edit_active(self):
            self._pending = True
            return
        self._pending = False
        ctx = self.context()
        visible: list[type] = []
        app = getattr(ctx, "app", None)
        mode = getattr(getattr(app, "tool_settings", None), "mode", None)
        for cls in registry.panels(self.space, self.region, self.category):
            panel_mode = getattr(cls, "mode", None)
            if panel_mode is not None and mode is not None and panel_mode != mode:
                continue
            try:
                ok = bool(cls.poll(ctx))
            except Exception:  # noqa: BLE001
                log.exception("面板 %s 的显示条件判断出错", panel_key(cls))
                ok = False
            if ok:
                visible.append(cls)
        wanted = [panel_key(cls) for cls in visible]
        focus = QApplication.focusWidget()
        focus_key = None
        if focus is not None and shiboken6.isValid(focus) and self._content.isAncestorOf(focus):
            focus_key = getattr(focus, "focus_key", None)
        bar = self.verticalScrollBar()
        scroll = self._restore_scroll if self._restore_scroll is not None else bar.value()
        self._restore_scroll = None
        self._content.setUpdatesEnabled(False)
        try:
            for key in list(self._entries):
                entry = self._entries[key]
                index = wanted.index(key) if key in wanted else -1
                if index < 0 or visible[index] is not entry.cls:
                    self._drop(key)
            for index, cls in enumerate(visible):
                key = wanted[index]
                entry = self._entries.get(key)
                if entry is None:
                    entry = self._create(key, cls)
                if self._column.indexOf(entry.foldout) != index:
                    self._column.removeWidget(entry.foldout)
                    self._column.insertWidget(index, entry.foldout)
                self._draw(entry, ctx)
        finally:
            self._content.setUpdatesEnabled(True)
        self._column.activate()
        self._content.adjustSize()
        bar.setValue(min(scroll, bar.maximum()))
        if focus_key is not None:
            self._restore_focus(focus_key)

    def _restore_focus(self, focus_key) -> None:
        """新建的控件要等下一轮事件循环才显示出来，那时再给焦点。"""
        ref = weakref.ref(self)

        def run() -> None:
            host = ref()
            if host is None or not shiboken6.isValid(host):
                return
            for widget in host._content.findChildren(QWidget):
                if getattr(widget, "focus_key", None) == focus_key and widget.isVisible():
                    widget.setFocus(Qt.OtherFocusReason)
                    return

        QTimer.singleShot(0, run)

    def _queue_refresh(self) -> None:
        if self._queued:
            return
        self._queued = True
        ref = weakref.ref(self)

        def run() -> None:
            host = ref()
            if host is None or not shiboken6.isValid(host):
                return
            host._queued = False
            host.refresh()

        QTimer.singleShot(0, run)

    def _on_registry_changed(self, kind: str = "") -> None:
        if not shiboken6.isValid(self):
            return
        if kind in ("panels", ""):
            self._queue_refresh()

    def _on_edit_state_changed(self) -> None:
        if not shiboken6.isValid(self):
            return
        if self._pending and not widgets.ui_edit_active(self):
            self._queue_refresh()

    def is_refresh_pending(self) -> bool:
        """有一次刷新因为编辑进行中被推迟了。"""
        return self._pending

    # ---- 面板 ----
    def _create(self, key: str, cls: type) -> _Entry:
        opened = self._open.get(key, not getattr(cls, "default_closed", False))
        foldout = Foldout(getattr(cls, "label", "") or key, icon=getattr(cls, "icon", "") or None,
                          closed=not opened)
        foldout.setObjectName(key)
        foldout.toggled.connect(partial(self._on_toggled, key))
        foldout.soloRequested.connect(partial(self._on_solo, key))
        try:
            panel = cls()
        except Exception:  # noqa: BLE001
            log.exception("面板 %s 创建出错", key)
            panel = None
        entry = _Entry(key, cls, panel, foldout)
        header_host = QWidget()
        foldout.header_layout.addWidget(header_host)
        entry.header_host = header_host
        self._entries[key] = entry
        self._open.setdefault(key, opened)
        return entry

    def _drop(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        if entry.layout is not None:
            entry.layout.clear()
        if entry.header_layout is not None:
            entry.header_layout.clear()
        self._column.removeWidget(entry.foldout)
        entry.foldout.hide()
        entry.foldout.deleteLater()

    def _draw(self, entry: _Entry, ctx: Any) -> None:
        foldout = entry.foldout
        foldout.setTitle(getattr(entry.cls, "label", "") or entry.key)
        if entry.layout is None:
            entry.layout = UILayout(foldout.body, ctx)
        else:
            entry.layout.clear()
            entry.layout.ctx = ctx
        layout = entry.layout
        layout.use_property_split = True
        layout.enabled = True
        if entry.panel is None:
            layout.label("这个面板没能创建", icon="warning", role="dim")
        else:
            try:
                entry.panel.draw(layout, ctx)
            except Exception as error:  # noqa: BLE001
                log.exception("面板 %s 绘制出错", entry.key)
                layout.enabled = True
                note = layout.label("这个面板的内容没能显示：%s" % error, icon="warning", role="dim")
                note.setWordWrap(True)
        if entry.header_layout is None:
            entry.header_layout = UILayout(entry.header_host, ctx, "ROW")
        else:
            entry.header_layout.clear()
            entry.header_layout.ctx = ctx
        if entry.panel is not None:
            try:
                entry.panel.draw_header(entry.header_layout, ctx)
            except Exception:  # noqa: BLE001
                log.exception("面板 %s 的标题行出错", entry.key)
        foldout.header.update()

    def _on_toggled(self, key: str, opened: bool) -> None:
        self._open[key] = bool(opened)

    def _on_solo(self, key: str) -> None:
        for other, entry in self._entries.items():
            entry.foldout.set_open(other == key)

    # ---- 查询 ----
    def visible_panels(self) -> list[str]:
        """正在显示的面板标识，按显示顺序。"""
        keys = []
        for index in range(self._column.count()):
            widget = self._column.itemAt(index).widget()
            if isinstance(widget, Foldout) and widget.objectName() in self._entries:
                keys.append(widget.objectName())
        return keys

    def foldout(self, key: str) -> Foldout | None:
        entry = self._entries.get(key)
        return entry.foldout if entry is not None else None

    def panel_layout(self, key: str) -> UILayout | None:
        entry = self._entries.get(key)
        return entry.layout if entry is not None else None

    # ---- 状态 ----
    def state(self) -> dict:
        """各面板的折叠状态和滚动位置。"""
        return {"open": dict(self._open), "scroll": int(self.verticalScrollBar().value())}

    def set_state(self, state: dict | None) -> None:
        if not state:
            return
        opened = state.get("open") or {}
        if isinstance(opened, dict):
            for key, value in opened.items():
                self._open[str(key)] = bool(value)
                entry = self._entries.get(str(key))
                if entry is not None:
                    entry.foldout.set_open(bool(value))
        scroll = state.get("scroll")
        if isinstance(scroll, (int, float)):
            self._restore_scroll = int(scroll)
            bar = self.verticalScrollBar()
            bar.setValue(min(int(scroll), bar.maximum()))
            ref = weakref.ref(self)

            def apply_scroll() -> None:
                host = ref()
                if host is not None and shiboken6.isValid(host) and host._restore_scroll is not None:
                    value = host._restore_scroll
                    host._restore_scroll = None
                    host.verticalScrollBar().setValue(value)

            QTimer.singleShot(0, apply_scroll)
