"""屏幕类操作：区域最大化、拆分、关闭、弹出，工具栏与侧栏开关，工作区切换与增删改名，操作搜索（F3）。

这些操作作用在上下文里的区域上，快捷键按下时就是鼠标所在的那个区域。
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QCursor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget

from ..core import ops
from ..core.ops import CANCELLED, FINISHED, Operator, register
from ..core.props import EnumProperty, FloatProperty, IntProperty, StringProperty
from ..ui import icons, theme

log = logging.getLogger("splender.ops.screen")


def _main_window(ctx: Any) -> Any:
    wm = getattr(ctx, "wm", None)
    return getattr(wm, "main_window", None) if wm is not None else None


def _has_area(ctx: Any) -> bool:
    return getattr(ctx, "screen", None) is not None and getattr(ctx, "area", None) is not None


# ======================================================================
# 区域
# ======================================================================

@register
class ScreenAreaMaximize(Operator):
    idname = "screen.area_maximize"
    label = "最大化区域"
    description = "让鼠标所在的区域铺满整个窗口，再执行一次还原"
    icon = "maximize"

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        screen = getattr(ctx, "screen", None)
        return screen is not None and (getattr(ctx, "area", None) is not None or
                                       getattr(screen, "maximized_area", None) is not None)

    def execute(self, ctx: Any) -> str:
        screen = ctx.screen
        if screen.maximized_area is not None:
            screen.restore()
        else:
            screen.maximize(ctx.area)
        return FINISHED


@register
class ScreenAreaSplit(Operator):
    idname = "screen.area_split"
    label = "拆分区域"
    description = "把区域拆成两块，新的一块显示同样的编辑器"
    icon = "split.h"

    orientation = EnumProperty("方向", items=[("H", "左右", "拆成左右两块"), ("V", "上下", "拆成上下两块")],
                               default="H")
    ratio = FloatProperty("位置", default=0.5, min=0.0, max=1.0, subtype="FACTOR",
                          description="左（上）块占的比例")

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return _has_area(ctx)

    def execute(self, ctx: Any) -> str:
        new_area = ctx.screen.split_area(ctx.area, self.orientation, self.ratio)
        if new_area is None:
            self.report(ctx, "区域太小，放不下两块", "WARNING")
            return CANCELLED
        return FINISHED


@register
class ScreenAreaClose(Operator):
    idname = "screen.area_close"
    label = "关闭区域"
    description = "关掉这个区域，旁边的区域补上来"
    icon = "close"

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return _has_area(ctx) and len(ctx.screen.areas()) > 1

    def execute(self, ctx: Any) -> str:
        return FINISHED if ctx.screen.close_area(ctx.area) else CANCELLED


@register
class ScreenAreaPopout(Operator):
    idname = "screen.area_popout"
    label = "在新窗口打开"
    description = "把这个区域复制到一个独立窗口，可以拖到副屏"
    icon = "popout"

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return getattr(ctx, "wm", None) is not None and getattr(ctx, "area", None) is not None

    def execute(self, ctx: Any) -> str:
        return FINISHED if ctx.wm.popout(ctx.area) is not None else CANCELLED


@register
class ScreenRegionToggle(Operator):
    idname = "screen.region_toggle"
    label = "显示或隐藏工具栏、侧栏"
    description = "开关鼠标所在编辑器的工具栏或侧栏"

    region = EnumProperty("区域", items=[("TOOLBAR", "工具栏", "编辑器左边的工具按钮"),
                                       ("SIDEBAR", "侧栏", "编辑器右边的面板")], default="SIDEBAR")

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        editor = getattr(ctx, "editor", None)
        return editor is not None and (getattr(editor, "has_toolbar", False) or getattr(editor, "has_sidebar", False))

    def execute(self, ctx: Any) -> str:
        editor = ctx.editor
        if self.region == "TOOLBAR":
            if not editor.has_toolbar:
                return CANCELLED
            editor.toolbar_visible = not editor.toolbar_visible
        else:
            if not editor.has_sidebar:
                return CANCELLED
            editor.sidebar_visible = not editor.sidebar_visible
        screen = getattr(ctx, "screen", None)
        if screen is not None and hasattr(screen, "mark_changed"):
            screen.mark_changed()
        return FINISHED


# ======================================================================
# 工作区
# ======================================================================

def _workspace_sources(_owner: Any = None) -> list[tuple]:
    from ..ui import workspaces
    items = [("CURRENT", "复制当前", "复制当前工作区的布局")]
    items += [(name, name, "默认的「%s」布局" % name) for name in workspaces.DEFAULT_ORDER]
    return items


@register
class ScreenWorkspaceCycle(Operator):
    idname = "screen.workspace_cycle"
    label = "切换工作区"
    description = "切到下一个或上一个工作区"
    searchable = False

    direction = IntProperty("方向", default=1, min=-1, max=1, description="1 下一个，-1 上一个")

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        window = _main_window(ctx)
        return window is not None and len(window.workspace_names()) > 1

    def execute(self, ctx: Any) -> str:
        return FINISHED if _main_window(ctx).cycle_workspace(self.direction) else CANCELLED


@register
class ScreenWorkspaceNext(Operator):
    idname = "screen.workspace_next"
    label = "下一个工作区"
    description = "切到右边的工作区"

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return ScreenWorkspaceCycle.poll(ctx)

    def execute(self, ctx: Any) -> str:
        return FINISHED if _main_window(ctx).cycle_workspace(1) else CANCELLED


@register
class ScreenWorkspacePrevious(Operator):
    idname = "screen.workspace_previous"
    label = "上一个工作区"
    description = "切到左边的工作区"

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return ScreenWorkspaceCycle.poll(ctx)

    def execute(self, ctx: Any) -> str:
        return FINISHED if _main_window(ctx).cycle_workspace(-1) else CANCELLED


@register
class ScreenWorkspaceAdd(Operator):
    idname = "screen.workspace_add"
    label = "新建工作区"
    description = "复制当前工作区，或按默认布局新建一个"
    icon = "add"

    source = EnumProperty("来源", items=_workspace_sources, default="CURRENT")
    name = StringProperty("名称", default="", description="留空时按来源取名")

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return _main_window(ctx) is not None

    def execute(self, ctx: Any) -> str:
        from ..ui import workspaces
        window = _main_window(ctx)
        if self.source == "CURRENT" or workspaces.default_layout(self.source) is None:
            index = window.duplicate_workspace()
            if self.name and index >= 0:
                window.rename_workspace(index, self.name)
        else:
            window.add_workspace(self.name or self.source, workspaces.default_layout(self.source))
        return FINISHED


@register
class ScreenWorkspaceDelete(Operator):
    idname = "screen.workspace_delete"
    label = "删除工作区"
    description = "删除当前工作区"
    icon = "remove"

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        window = _main_window(ctx)
        return window is not None and len(window.workspace_names()) > 1

    def execute(self, ctx: Any) -> str:
        return FINISHED if _main_window(ctx).remove_workspace() else CANCELLED


@register
class ScreenWorkspaceRename(Operator):
    idname = "screen.workspace_rename"
    label = "重命名工作区"
    description = "给当前工作区改名"

    name = StringProperty("名称", default="")

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return _main_window(ctx) is not None

    def execute(self, ctx: Any) -> str:
        window = _main_window(ctx)
        index = window.active_workspace_index()
        if self.name.strip():
            return FINISHED if window.rename_workspace(index, self.name) else CANCELLED
        window.topbar.tabs.start_rename(index)
        return FINISHED


# ======================================================================
# 操作搜索（F3）
# ======================================================================

_recent: list[str] = []
_popups: list[Any] = []


def recent_searches() -> list[str]:
    """最近从搜索里执行过的操作，新的在前。"""
    return list(_recent)


def last_search_popup() -> Any:
    """最近打开的搜索浮层（可能已关闭）。"""
    return _popups[-1] if _popups else None


class _Entry:
    __slots__ = ("idname", "label", "description", "shortcut", "haystack")

    def __init__(self, idname: str, label: str, description: str, shortcut: str) -> None:
        self.idname = idname
        self.label = label
        self.description = description
        self.shortcut = shortcut
        self.haystack = (label + " " + idname + " " + description).lower()


class _ResultList(QWidget):
    """搜索结果列表：操作名在左，快捷键在右。"""

    def __init__(self, popup: "SearchPopup") -> None:
        super().__init__(popup)
        self._popup = popup
        self.setMouseTracking(True)

    def row_height(self) -> int:
        return theme.size("row")

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(theme.px(420), self.row_height() * self._popup.visible_rows)

    def row_at(self, pos: QPoint) -> int:
        index = pos.y() // max(1, self.row_height()) + self._popup.scroll
        return index if 0 <= index < len(self._popup.results) else -1

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        popup = self._popup
        row_h = self.row_height()
        pad = theme.px(8)
        radius = theme.size("radius.small")
        font = theme.font()
        small = theme.font("font.small")
        fm_small = QFontMetrics(small)
        if not popup.results:
            painter.setFont(font)
            painter.setPen(theme.qcolor("text.faint"))
            painter.drawText(self.rect().adjusted(pad, 0, -pad, 0), Qt.AlignLeft | Qt.AlignVCenter,
                             "没有找到匹配的操作")
            return
        for row in range(popup.visible_rows):
            index = popup.scroll + row
            if index >= len(popup.results):
                break
            entry = popup.results[index]
            rect = QRect(0, row * row_h, self.width(), row_h)
            selected = index == popup.selected
            if selected:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.qcolor("accent"))
                painter.drawRoundedRect(rect, radius, radius)
            shortcut_w = fm_small.horizontalAdvance(entry.shortcut) if entry.shortcut else 0
            if entry.shortcut:
                painter.setFont(small)
                painter.setPen(theme.qcolor("text.on_accent" if selected else "text.faint"))
                painter.drawText(rect.adjusted(pad, 0, -pad, 0), Qt.AlignRight | Qt.AlignVCenter, entry.shortcut)
            painter.setFont(font)
            painter.setPen(theme.qcolor("text.on_accent" if selected else "text"))
            text_rect = rect.adjusted(pad, 0, -pad - shortcut_w - (pad if shortcut_w else 0), 0)
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter,
                             QFontMetrics(font).elidedText(entry.label, Qt.ElideRight, text_rect.width()))

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        index = self.row_at(event.position().toPoint())
        if index >= 0 and index != self._popup.selected:
            self._popup.selected = index
            self.update()
            self._popup.sync_tooltip()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            index = self.row_at(event.position().toPoint())
            if index >= 0:
                self._popup.execute(index)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        steps = -1 if event.angleDelta().y() > 0 else 1
        self._popup.scroll_by(steps * 3)


class SearchPopup(QWidget):
    """操作搜索浮层：输入文字筛选可用的操作，上下键选择，回车执行，Esc 关闭。"""

    MAX_ROWS = 12

    def __init__(self, ctx: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.ctx = ctx
        self.results: list[_Entry] = []
        self.selected = 0
        self.scroll = 0
        self.visible_rows = self.MAX_ROWS
        self.executed = ""
        self._entries = self._collect(ctx)
        box = QVBoxLayout(self)
        pad = theme.px(6)
        box.setContentsMargins(pad, pad, pad, pad)
        box.setSpacing(theme.px(4))
        self.field = QLineEdit(self)
        self.field.setPlaceholderText("搜索操作")
        self.field.addAction(icons.icon("search", theme.color("text.faint")), QLineEdit.LeadingPosition)
        self.field.setMinimumHeight(theme.size("widget"))
        self.field.textChanged.connect(self.set_query)
        self.field.installEventFilter(self)
        self.list = _ResultList(self)
        box.addWidget(self.field)
        box.addWidget(self.list)
        self.set_query("")

    @staticmethod
    def _collect(ctx: Any) -> list[_Entry]:
        kc = getattr(ctx, "keyconfig", None)
        entries = []
        for cls in ops.all_operators():
            if not getattr(cls, "searchable", True) or cls.idname == "wm.search":
                continue
            if not ops.poll(cls.idname, ctx):
                continue
            shortcut = kc.shortcut_for(cls.idname) if kc is not None else ""
            entries.append(_Entry(cls.idname, cls.label or cls.idname, cls.description or "", shortcut))
        return entries

    def set_query(self, text: str) -> None:
        terms = (text or "").lower().split()
        if not terms:
            order = {name: i for i, name in enumerate(_recent)}
            results = sorted(self._entries, key=lambda e: (order.get(e.idname, len(order)), e.label))
        else:
            query = " ".join(terms)
            scored = []
            for entry in self._entries:
                if not all(term in entry.haystack for term in terms):
                    continue
                label = entry.label.lower()
                if label.startswith(query):
                    score = 0
                elif query in label:
                    score = 1
                elif all(term in label for term in terms):
                    score = 2
                elif query in entry.idname.lower():
                    score = 3
                else:
                    score = 4
                scored.append((score, entry.label, entry))
            results = [item[2] for item in sorted(scored, key=lambda s: (s[0], s[1]))]
        self.results = results
        self.selected = 0
        self.scroll = 0
        self.visible_rows = max(1, min(self.MAX_ROWS, len(results) or 1))
        self.list.setFixedHeight(self.list.row_height() * self.visible_rows)
        self.adjustSize()
        self.list.update()
        self.sync_tooltip()

    def sync_tooltip(self) -> None:
        if 0 <= self.selected < len(self.results):
            entry = self.results[self.selected]
            self.list.setToolTip(entry.description)

    def move_selection(self, delta: int) -> None:
        if not self.results:
            return
        self.selected = max(0, min(len(self.results) - 1, self.selected + delta))
        if self.selected < self.scroll:
            self.scroll = self.selected
        elif self.selected >= self.scroll + self.visible_rows:
            self.scroll = self.selected - self.visible_rows + 1
        self.list.update()
        self.sync_tooltip()

    def scroll_by(self, rows: int) -> None:
        limit = max(0, len(self.results) - self.visible_rows)
        self.scroll = max(0, min(limit, self.scroll + rows))
        self.list.update()

    def execute(self, index: int | None = None) -> str:
        index = self.selected if index is None else index
        if not 0 <= index < len(self.results):
            return CANCELLED
        idname = self.results[index].idname
        self.executed = idname
        self.close()
        if idname in _recent:
            _recent.remove(idname)
        _recent.insert(0, idname)
        del _recent[20:]
        return ops.call(idname, self.ctx)

    def popup_at(self, global_pos: QPoint) -> None:
        self.setFixedWidth(theme.px(440))
        self.adjustSize()
        x = global_pos.x() - self.width() // 2
        y = global_pos.y() - theme.size("widget")
        screen = QApplication.screenAt(global_pos)
        if screen is not None:
            avail = screen.availableGeometry()
            x = max(avail.x(), min(avail.x() + avail.width() - self.width(), x))
            y = max(avail.y(), min(avail.y() + avail.height() - self.height(), y))
        self.move(QPoint(x, y))
        self.show()
        self.field.setFocus(Qt.PopupFocusReason)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.field and event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Down:
                self.move_selection(1)
                return True
            if key == Qt.Key_Up:
                self.move_selection(-1)
                return True
            if key == Qt.Key_PageDown:
                self.move_selection(self.visible_rows)
                return True
            if key == Qt.Key_PageUp:
                self.move_selection(-self.visible_rows)
                return True
            if key in (Qt.Key_Return, Qt.Key_Enter):
                self.execute()
                return True
            if key == Qt.Key_Escape:
                self.close()
                return True
        return False

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        radius = theme.px(6)
        painter.setPen(QPen(theme.qcolor("line.soft"), 1))
        painter.setBrush(theme.qcolor("bg.menu"))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), radius, radius)


@register
class WMSearch(Operator):
    idname = "wm.search"
    label = "搜索操作"
    description = "按名称查找并执行任何操作"
    icon = "search"
    searchable = False

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return QApplication.instance() is not None

    def invoke(self, ctx: Any, event: Any) -> str:
        parent = ctx.window if isinstance(getattr(ctx, "window", None), QWidget) else None
        popup = SearchPopup(ctx, parent)
        _popups[:] = [p for p in _popups[-4:]]
        _popups.append(popup)
        popup.popup_at(QCursor.pos())
        return FINISHED

    def execute(self, ctx: Any) -> str:
        return self.invoke(ctx, None)
