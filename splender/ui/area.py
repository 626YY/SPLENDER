"""区域与屏幕：Blender 式的模块化窗口。

Screen 是一个工作区的区域拆分树。树的每个叶子是一个 AreaWidget，每个区域装一个编辑器。
Screen 自己管几何：给每个节点算矩形，区域是 Screen 的直接子控件，用 setGeometry 摆放；
区域之间留 theme.size("splitter") 宽的缝，缝上盖一条透明的拖动条。窗口缩放时按比例重排，
每个区域有最小尺寸，放不下时按最小尺寸分配。

交互：
- 拖缝改比例。和 Blender 一样，只动挨着这条缝的区域，嵌套在里面的平行分隔缝留在原处；
- 区域四角各有一块四分之一圆的热区（鼠标移上去变十字）：往区域里面拖是拆分（横向拖左右拆、
  纵向拖上下拆，分隔线跟着鼠标走，松手确定）；拖进相邻区域是合并（两个区域必须共用一整条边，
  被拖入的区域消失）；按住 Ctrl 拖到任意别的区域是交换编辑器；Esc 或右键取消；
- 拖动中的预览画在一个盖在 Screen 上的透明覆盖层里，只在拖动期间显示；
- 标题栏右键菜单：垂直拆分、水平拆分、最大化、在新窗口打开、关闭区域；
- 最大化：maximize(area) 把一个区域铺满整个 Screen，其它区域隐藏但不销毁，restore() 还原。

序列化格式（docs/ARCHITECTURE.md 3.6）：
    {"type": "split", "orientation": "H" | "V", "ratio": 0.7, "a": {...}, "b": {...}}
    {"type": "area", "editor": "VIEW_3D", "state": {...}, "stash": {其它编辑器类型: 状态}}
H 表示左右排列，a 在左（上），b 在右（下），ratio 是 a 占去掉缝之后的比例。
"stash" 是这个区域里换过的其它编辑器类型的状态，换回来时恢复；没有时省略。
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Any, Callable

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QCursor, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF, QRegion,
                           QTransform)
from PySide6.QtWidgets import QAbstractButton, QApplication, QMenu, QWidget

from ..core import ops, registry
from ..core.context import Context
from ..core.props import FloatProperty, IntProperty, PropertyGroup
from . import icons, theme
from .editor import Editor, UnavailableEditor, alive

log = logging.getLogger("splender.ui.area")

H = "H"   # 左右排列（分隔线是竖的）
V = "V"   # 上下排列（分隔线是横的）
_CORNERS = ("TL", "TR", "BL", "BR")
_MAX_DEPTH = 64


class ScreenSettings(PropertyGroup):
    """区域框架的可调项。"""

    corner_size = IntProperty("角落热区半径", default=12, min=4, max=48, unit="px",
                              description="区域四角可以拖动来拆分、合并、交换的范围")
    drag_threshold = IntProperty("拖动起始距离", default=6, min=1, max=48, unit="px",
                                 description="从角落按下后移动超过这个距离才开始拆分或合并")
    edge_grab = IntProperty("分隔缝两侧抓取宽度", default=2, min=0, max=12, unit="px",
                            description="分隔缝两侧额外可以抓住拖动的宽度")
    min_area_width = IntProperty("区域最小宽度", default=64, min=24, max=600, unit="px",
                                 description="拖动分隔缝或拆分时区域不会窄过这个宽度")
    min_area_height = IntProperty("区域最小高度", default=40, min=24, max=600, unit="px",
                                  description="拖动分隔缝或拆分时区域不会矮过这个高度，至少放得下标题栏")
    report_seconds = FloatProperty("消息停留时间", default=4.0, min=0.5, max=60.0, unit="s", precision=1,
                                   description="状态栏里操作结果的消息显示多久后换回提示文字")


#: 全局的一份。偏好编辑器可以直接把它画出来。
settings = ScreenSettings()


def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(1.0, max(0.0, number))


def _to_point(pos: Any) -> QPoint:
    if isinstance(pos, QPoint):
        return QPoint(pos)
    if isinstance(pos, QPointF):
        return pos.toPoint()
    return QPoint(int(pos[0]), int(pos[1]))


# ======================================================================
# 拆分树
# ======================================================================

class _Leaf:
    __slots__ = ("area", "parent", "rect")

    def __init__(self, area: "AreaWidget") -> None:
        self.area = area
        self.parent: _Split | None = None
        self.rect = QRect()


class _Split:
    __slots__ = ("orientation", "ratio", "a", "b", "parent", "rect", "divider", "handle")

    def __init__(self, orientation: str, ratio: float, a: Any, b: Any) -> None:
        self.orientation = orientation
        self.ratio = ratio
        self.a = a
        self.b = b
        self.parent: _Split | None = None
        self.rect = QRect()
        self.divider = QRect()
        self.handle: SplitterHandle | None = None
        a.parent = self
        b.parent = self


def _iter_leaves(node: Any) -> list[_Leaf]:
    out: list[_Leaf] = []
    stack = [node] if node is not None else []
    while stack:
        item = stack.pop()
        if isinstance(item, _Leaf):
            out.append(item)
        else:
            stack.append(item.b)
            stack.append(item.a)
    return out


def _iter_splits(node: Any) -> list[_Split]:
    """自上而下（广度优先）列出所有拆分节点。"""
    out: list[_Split] = []
    queue = [node] if isinstance(node, _Split) else []
    while queue:
        item = queue.pop(0)
        out.append(item)
        for child in (item.a, item.b):
            if isinstance(child, _Split):
                queue.append(child)
    return out


# ======================================================================
# 编辑器类型按钮与下拉
# ======================================================================

class EditorTypeButton(QAbstractButton):
    """区域标题栏最左边的按钮：编辑器图标加一个小下拉箭头。"""

    def __init__(self, area: "AreaWidget") -> None:
        super().__init__(area)
        self._area = area
        self._open = False
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_Hover, True)
        self.clicked.connect(self._on_clicked)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setFixedSize(theme.px(40), max(theme.px(16), theme.size("widget") - theme.px(2)))
        self.update()

    def set_open(self, value: bool) -> None:
        self._open = bool(value)
        self.update()

    def sync(self) -> None:
        editor = self._area.editor
        label = getattr(editor, "label", "") if editor is not None else ""
        if isinstance(editor, UnavailableEditor) and editor.missing_idname:
            label = "%s（%s）" % (editor.label, editor.missing_idname)
        self.setToolTip("编辑器类型：%s" % label if label else "编辑器类型")
        self.update()

    def _on_clicked(self) -> None:
        self._area.open_type_menu()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(theme.px(40), theme.size("widget") - theme.px(2))

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        hot = self._open or self.isDown() or self.underMouse()
        radius = theme.size("radius")
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.qcolor("bg.widget_hover" if hot else "bg.widget"))
        painter.drawRoundedRect(QRectF(self.rect()), radius, radius)
        editor = self._area.editor
        name = getattr(editor, "icon", "") or "editor.view3d"
        dpr = self.devicePixelRatioF()
        side = theme.size("icon")
        pix = icons.pixmap(name, theme.color("text"), side, dpr)
        painter.drawPixmap(QPoint(theme.px(6), (self.height() - side) // 2), pix)
        small = theme.px(10)
        chevron = icons.pixmap("chevron.down", theme.color("text.dim"), small, dpr)
        painter.drawPixmap(QPoint(self.width() - theme.px(5) - small, (self.height() - small) // 2), chevron)


class _TypeEntry:
    __slots__ = ("rect", "idname", "label", "icon", "column", "row")

    def __init__(self, rect: QRect, idname: str, label: str, icon: str, column: int, row: int) -> None:
        self.rect = rect
        self.idname = idname
        self.label = label
        self.icon = icon
        self.column = column
        self.row = row


class EditorTypeMenu(QWidget):
    """编辑器类型下拉：按 category 分栏列出全部已注册的编辑器，带图标。"""

    chosen = Signal(str)

    def __init__(self, area: "AreaWidget", on_close: Callable[[], None] | None = None) -> None:
        super().__init__(area.window(), Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._current = area.editor_idname()
        self._on_close = on_close
        self._entries: list[_TypeEntry] = []
        self._titles: list[tuple[QRect, str]] = []
        self._hover = -1
        self._build()

    @property
    def entries(self) -> list[_TypeEntry]:
        return list(self._entries)

    def _build(self) -> None:
        groups: dict[str, list[type]] = {}
        order: list[str] = []
        for cls in registry.editors():
            category = getattr(cls, "category", "") or "通用"
            if category not in groups:
                groups[category] = []
                order.append(category)
            groups[category].append(cls)
        fm = QFontMetrics(theme.font())
        fm_small = QFontMetrics(theme.font("font.small"))
        pad = theme.px(5)
        gap = theme.px(4)
        row = theme.size("row")
        icon = theme.size("icon")
        widths = []
        for category in order:
            width = fm_small.horizontalAdvance(category) + theme.px(16)
            for cls in groups[category]:
                text = getattr(cls, "label", "") or cls.idname
                width = max(width, theme.px(8) + icon + theme.px(8) + fm.horizontalAdvance(text) + theme.px(18))
            widths.append(max(width, theme.px(150)))
        rows = max((len(items) for items in groups.values()), default=0)
        x = pad
        for column, category in enumerate(order):
            width = widths[column]
            self._titles.append((QRect(x, pad, width, row), category))
            for index, cls in enumerate(groups[category]):
                rect = QRect(x, pad + row + index * row, width, row)
                self._entries.append(_TypeEntry(rect, cls.idname, getattr(cls, "label", "") or cls.idname,
                                                getattr(cls, "icon", "") or "editor.view3d", column, index))
            x += width + gap
        if not order:
            width = fm.horizontalAdvance("没有可用的编辑器") + theme.px(32)
            self._titles.append((QRect(pad, pad, width, row), "没有可用的编辑器"))
            x = pad + width + gap
            rows = 0
        self.resize(x - gap + pad, pad + row + rows * row + pad)
        for index, entry in enumerate(self._entries):
            if entry.idname == self._current:
                self._hover = index
                break

    def popup_under(self, anchor: QWidget) -> None:
        pos = anchor.mapToGlobal(QPoint(0, anchor.height() + theme.px(2)))
        screen = anchor.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            if pos.x() + self.width() > avail.x() + avail.width():
                pos.setX(max(avail.x(), avail.x() + avail.width() - self.width()))
            if pos.y() + self.height() > avail.y() + avail.height():
                above = anchor.mapToGlobal(QPoint(0, -self.height() - theme.px(2)))
                pos.setY(max(avail.y(), above.y()))
        self.move(pos)
        self.show()
        self.setFocus(Qt.PopupFocusReason)

    def entry_at(self, pos: QPoint) -> int:
        for index, entry in enumerate(self._entries):
            if entry.rect.contains(pos):
                return index
        return -1

    def choose(self, idname: str) -> None:
        self.close()
        self.chosen.emit(idname)

    # ---- Qt ----
    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        radius = theme.px(6)
        painter.setPen(QPen(theme.qcolor("line.soft"), 1))
        painter.setBrush(theme.qcolor("bg.menu"))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        inset = theme.px(8)
        painter.setFont(theme.font("font.small"))
        painter.setPen(theme.qcolor("text.faint"))
        for rect, text in self._titles:
            painter.drawText(rect.adjusted(inset, 0, -inset, 0), Qt.AlignLeft | Qt.AlignVCenter, text)
        painter.setFont(theme.font())
        dpr = self.devicePixelRatioF()
        side = theme.size("icon")
        small = theme.size("radius")
        for index, entry in enumerate(self._entries):
            rect = entry.rect
            hot = index == self._hover
            current = entry.idname == self._current
            if hot:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.qcolor("accent"))
                painter.drawRoundedRect(QRectF(rect), small, small)
            elif current:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.qcolor("accent.softer"))
                painter.drawRoundedRect(QRectF(rect), small, small)
            color = theme.color("text.on_accent") if hot else theme.color("text")
            pix = icons.pixmap(entry.icon, color, side, dpr)
            painter.drawPixmap(QPoint(rect.x() + inset, rect.y() + (rect.height() - side) // 2), pix)
            painter.setPen(QColor(color))
            text_rect = rect.adjusted(inset + side + theme.px(8), 0, -inset, 0)
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, entry.label)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        index = self.entry_at(event.position().toPoint())
        if index != self._hover and index >= 0:
            self._hover = index
            self.update()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            return
        index = self.entry_at(event.position().toPoint())
        if index >= 0:
            self.choose(self._entries[index].idname)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
            return
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            if 0 <= self._hover < len(self._entries):
                self.choose(self._entries[self._hover].idname)
            return
        if not self._entries:
            return
        current = self._entries[self._hover] if 0 <= self._hover < len(self._entries) else self._entries[0]
        column, row = current.column, current.row
        if key == Qt.Key_Up:
            row -= 1
        elif key == Qt.Key_Down:
            row += 1
        elif key == Qt.Key_Left:
            column -= 1
        elif key == Qt.Key_Right:
            column += 1
        else:
            super().keyPressEvent(event)
            return
        columns = sorted({e.column for e in self._entries})
        column = min(max(column, columns[0]), columns[-1])
        in_column = [i for i, e in enumerate(self._entries) if e.column == column]
        row = min(max(row, 0), len(in_column) - 1)
        self._hover = in_column[row]
        self.update()

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        callback, self._on_close = self._on_close, None
        if callback is not None:
            try:
                callback()
            except Exception:  # noqa: BLE001
                log.exception("关闭编辑器类型菜单出错")
        super().closeEvent(event)


# ======================================================================
# 角落热区、分隔缝、覆盖层
# ======================================================================

class CornerZone(QWidget):
    """区域一角的热区：四分之一圆，十字光标；顺便画出区域的圆角。"""

    def __init__(self, area: "AreaWidget", corner: str) -> None:
        super().__init__(area)
        self._area = area
        self.corner = corner
        self._pressed = False
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setCursor(Qt.CrossCursor)
        self.place()

    @property
    def left(self) -> bool:
        return self.corner in ("TL", "BL")

    @property
    def top(self) -> bool:
        return self.corner in ("TL", "TR")

    def place(self) -> None:
        size = theme.px(settings.corner_size)
        width, height = self._area.width(), self._area.height()
        x = 0 if self.left else width - size
        y = 0 if self.top else height - size
        self.setGeometry(x, y, size, size)
        cx = 0 if self.left else size
        cy = 0 if self.top else size
        self.setMask(QRegion(QRect(cx - size, cy - size, 2 * size, 2 * size), QRegion.Ellipse))
        self.update()

    def set_enabled_for_drag(self, enabled: bool) -> None:
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        radius = float(theme.size("radius"))
        if radius <= 0:
            return
        size = float(self.width())
        path = QPainterPath()
        path.moveTo(0.0, 0.0)
        path.lineTo(radius, 0.0)
        path.arcTo(QRectF(0.0, 0.0, 2 * radius, 2 * radius), 90.0, 90.0)
        path.closeSubpath()
        transform = QTransform(-1.0 if not self.left else 1.0, 0.0, 0.0, -1.0 if not self.top else 1.0,
                               size if not self.left else 0.0, size if not self.top else 0.0)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setTransform(transform)
        painter.fillPath(path, theme.qcolor("bg.window"))

    def _screen(self) -> "Screen | None":
        return self._area.parent_screen()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        screen = self._screen()
        if screen is None:
            event.ignore()
            return
        if event.button() == Qt.LeftButton:
            if screen.begin_corner_drag(self._area, self.corner, event.globalPosition().toPoint()):
                self._pressed = True
                event.accept()
                return
        elif self._pressed:
            screen.cancel_drag()          # 拖动中按右键取消
            self._pressed = False
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        screen = self._screen()
        if self._pressed and screen is not None:
            screen.update_corner_drag(event.globalPosition().toPoint(), event.modifiers())
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        screen = self._screen()
        if self._pressed and event.button() == Qt.LeftButton:
            self._pressed = False
            if screen is not None:
                screen.finish_corner_drag(event.globalPosition().toPoint(), event.modifiers())
            event.accept()
        else:
            event.ignore()


class SplitterHandle(QWidget):
    """盖在分隔缝上的透明拖动条，比缝略宽，方便抓。"""

    def __init__(self, screen: "Screen", node: _Split) -> None:
        super().__init__(screen)
        self._screen = screen
        self.node = node
        self._pressed = False
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setCursor(Qt.SplitHCursor if node.orientation == H else Qt.SplitVCursor)

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._screen.begin_splitter_drag(self.node, event.globalPosition().toPoint()):
            self._pressed = True
            event.accept()
        elif event.button() == Qt.RightButton:
            self.show_options(event.globalPosition().toPoint())
            event.accept()
        else:
            event.ignore()

    def show_options(self, global_pos: Any) -> QMenu:
        """右键分隔缝：区域选项（照 Blender）——拆分旁边的区域、合并两边、交换两边。"""
        screen = self._screen
        node = self.node
        menu = QMenu(self)
        menu.setAttribute(Qt.WA_DeleteOnClose, True)
        menu.addSection("区域选项")
        area = screen.area_at(global_pos + (QPoint(-6, 0) if node.orientation == H else QPoint(0, -6)))
        if area is not None:
            menu.addAction("垂直拆分", lambda: screen.split_area(area, H))
            menu.addAction("水平拆分", lambda: screen.split_area(area, V))
        a = node.a.area if isinstance(node.a, _Leaf) else None
        b = node.b.area if isinstance(node.b, _Leaf) else None
        join = menu.addAction("合并两边（留下%s）" % ("左边" if node.orientation == H else "上边"),
                              lambda: screen.join_areas(a, b))
        join.setEnabled(a is not None and b is not None and screen.can_join(a, b))
        swap = menu.addAction("交换两边", lambda: screen.swap_areas(a, b))
        swap.setEnabled(a is not None and b is not None)
        menu.popup(global_pos)
        return menu

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self._pressed:
            self._screen.update_splitter_drag(event.globalPosition().toPoint())
            event.accept()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if self._pressed and event.button() == Qt.LeftButton:
            self._pressed = False
            self._screen.finish_splitter_drag(event.globalPosition().toPoint())
            event.accept()


class DragOverlay(QWidget):
    """拖动角落时盖在整个 Screen 上的透明层，画拆分线、合并箭头、交换提示。"""

    def __init__(self, screen: "Screen") -> None:
        super().__init__(screen)
        self._screen = screen
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.hide()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._screen.paint_drag_preview(painter)


class _DragKeyFilter(QObject):
    """拖动角落期间截住按键：Esc 取消，Ctrl 切换交换预览，其它键不往下传。"""

    def __init__(self, screen: "Screen") -> None:
        super().__init__(screen)
        self._screen = screen

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        t = event.type()
        if t in (QEvent.KeyPress, QEvent.KeyRelease):
            if not isinstance(obj, QWidget):
                return False
            key = event.key()
            if key == Qt.Key_Escape:
                if t == QEvent.KeyPress:
                    self._screen.cancel_drag()
                return True
            if key == Qt.Key_Control:
                self._screen.set_drag_ctrl(t == QEvent.KeyPress)
            return True
        if t in (QEvent.ApplicationDeactivate, QEvent.WindowDeactivate) and obj is self._screen.window():
            self._screen.cancel_drag()
        return False


class _CornerDrag:
    __slots__ = ("area", "corner", "start", "pos", "active", "mode", "orientation", "target", "valid",
                 "split_pos", "ctrl")

    def __init__(self, area: "AreaWidget", corner: str, start: QPoint) -> None:
        self.area = area
        self.corner = corner
        self.start = QPoint(start)
        self.pos = QPoint(start)
        self.active = False
        self.mode: str | None = None        # "SPLIT" / "JOIN" / "SWAP"
        self.orientation: str | None = None
        self.target: AreaWidget | None = None
        self.valid = False
        self.split_pos = 0                  # 拆分线（缝的起点）在 Screen 里的坐标
        self.ctrl = False


# ======================================================================
# 区域
# ======================================================================

class AreaWidget(QWidget):
    """一个区域：装一个编辑器。标题栏最左是编辑器类型按钮。换编辑器类型时保留各类型上次的状态。"""

    editorChanged = Signal(str)

    def __init__(self, app: Any, screen: "Screen", idname: str = "", state: dict | None = None,
                 stash: dict | None = None) -> None:
        super().__init__(screen)
        self.app = app
        self._screen = screen
        self._editor: Editor | None = None
        self._stash: dict[str, dict] = {}
        if isinstance(stash, dict):
            self._stash = {str(k): copy.deepcopy(v) for k, v in stash.items() if isinstance(v, dict)}
        self._menu: EditorTypeMenu | None = None
        self._type_button = EditorTypeButton(self)
        self._corners = [CornerZone(self, corner) for corner in _CORNERS]
        theme.changed.connect(self._on_theme_changed)
        registry.changed.connect(self._on_registry_changed)
        self.set_editor(idname, state if isinstance(state, dict) else {})

    # ---- 查询 ----
    @property
    def editor(self) -> Editor | None:
        return self._editor

    @property
    def type_button(self) -> EditorTypeButton:
        return self._type_button

    @property
    def corner_zones(self) -> list[CornerZone]:
        return list(self._corners)

    def parent_screen(self) -> "Screen | None":
        return self._screen if alive(self._screen) else None

    def editor_idname(self) -> str:
        """当前编辑器类型。顶替用的不可用编辑器返回它顶替的那个类型。"""
        editor = self._editor
        if isinstance(editor, UnavailableEditor):
            return editor.missing_idname
        return getattr(editor, "idname", "") if editor is not None else ""

    def editor_state(self) -> dict:
        editor = self._editor
        if editor is None:
            return {}
        try:
            state = editor.save_state()
        except Exception:  # noqa: BLE001
            log.exception("编辑器 %s 保存状态出错", getattr(editor, "idname", "?"))
            state = {}
        return copy.deepcopy(state) if isinstance(state, dict) else {}

    def stashed_states(self) -> dict[str, dict]:
        return copy.deepcopy(self._stash)

    # ---- 换编辑器 ----
    def set_editor(self, idname: str, state: dict | None = None) -> Editor:
        """换成 idname 类型的编辑器。state 为 None 时恢复这个类型在本区域上次的状态。"""
        idname = str(idname or "")
        old = self._editor
        if old is not None:
            old_key = self.editor_idname()
            if old_key == idname and state is None:
                return old
            if old_key:
                self._stash[old_key] = self.editor_state()
            self._detach(old)
        if state is None:
            state = self._stash.pop(idname, None)
        else:
            self._stash.pop(idname, None)
        editor = self._create(idname, state or {})
        self._editor = editor
        editor.attach_type_button(self._type_button)
        editor.setGeometry(self.rect())
        editor.show()
        for zone in self._corners:
            zone.raise_()
        self._type_button.sync()
        try:
            editor.refresh()
        except Exception:  # noqa: BLE001
            log.exception("编辑器 %s 刷新出错", idname)
        if old is not None:
            self.editorChanged.emit(idname)
            screen = self.parent_screen()
            if screen is not None:
                screen.mark_changed()
        return editor

    def _create(self, idname: str, state: dict) -> Editor:
        cls = registry.editor(idname) if idname else None
        if cls is None:
            if idname:
                log.warning("编辑器 %s 没有注册，用占位编辑器顶上", idname)
            return UnavailableEditor(self.app, self, idname, state)
        try:
            editor = cls(self.app, self)
        except Exception:  # noqa: BLE001
            log.exception("创建编辑器 %s 失败", idname)
            return UnavailableEditor(self.app, self, idname, state, reason="error")
        try:
            editor.load_state(copy.deepcopy(state))
        except Exception:  # noqa: BLE001
            log.exception("编辑器 %s 恢复状态出错", idname)
        return editor

    def _detach(self, editor: Editor) -> None:
        self._type_button.setParent(self)        # 先从旧标题栏里取出来，免得跟着删掉
        editor.hide()
        try:
            editor.dispose()
        except Exception:  # noqa: BLE001
            log.exception("编辑器 %s 释放资源出错", getattr(editor, "idname", "?"))
        editor.setParent(None)
        editor.deleteLater()

    def dispose(self) -> None:
        """区域被删除前调用。"""
        if self._menu is not None and alive(self._menu):
            self._menu.close()
        editor = self._editor
        self._editor = None
        if editor is not None and alive(editor):
            self._detach(editor)

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        data: dict[str, Any] = {"type": "area", "editor": self.editor_idname(), "state": self.editor_state()}
        if self._stash:
            data["stash"] = copy.deepcopy(self._stash)
        return data

    def from_dict(self, data: dict) -> None:
        if not isinstance(data, dict):
            return
        stash = data.get("stash")
        self._stash = ({str(k): copy.deepcopy(v) for k, v in stash.items() if isinstance(v, dict)}
                       if isinstance(stash, dict) else {})
        state = data.get("state")
        self.set_editor(str(data.get("editor") or ""), copy.deepcopy(state) if isinstance(state, dict) else {})

    # ---- 下拉菜单与右键菜单 ----
    def open_type_menu(self) -> EditorTypeMenu:
        if self._menu is not None and alive(self._menu):
            self._menu.close()
        menu = EditorTypeMenu(self, on_close=self._on_menu_closed)
        menu.chosen.connect(self._on_type_chosen)
        self._menu = menu
        self._type_button.set_open(True)
        menu.popup_under(self._type_button)
        return menu

    def _on_menu_closed(self) -> None:
        self._menu = None
        if alive(self._type_button):
            self._type_button.set_open(False)

    def _on_type_chosen(self, idname: str) -> None:
        if alive(self) and idname != self.editor_idname():
            self.set_editor(idname)

    def _context(self) -> Any:
        wm = getattr(self.app, "wm", None)
        if wm is not None and hasattr(wm, "context"):
            return wm.context(area=self)
        return Context(app=self.app, screen=self.parent_screen(), area=self, editor=self._editor,
                       window=self.window())

    def header_menu_items(self) -> list[tuple]:
        """右键菜单的条目：(文字, 图标, 操作名, 属性, 直接执行的后备函数)；None 是分隔线。"""
        screen = self.parent_screen()
        maximized = screen is not None and screen.maximized_area is not None
        items: list[Any] = [
            ("垂直拆分", "split.h", "screen.area_split", {"orientation": H},
             lambda: screen.split_area(self, H) if screen else None),
            ("水平拆分", "split.v", "screen.area_split", {"orientation": V},
             lambda: screen.split_area(self, V) if screen else None),
            None,
            ("还原" if maximized else "最大化", "restore" if maximized else "maximize", "screen.area_maximize", {},
             lambda: (screen.restore() if maximized else screen.maximize(self)) if screen else None),
            ("在新窗口打开", "popout", "screen.area_popout", {}, None),
        ]
        editor = self._editor
        if editor is not None and (editor.has_toolbar or editor.has_sidebar):
            items.append(None)
            if editor.has_toolbar:
                items.append(("工具栏", "", "screen.region_toggle", {"region": "TOOLBAR"},
                              lambda: setattr(editor, "toolbar_visible", not editor.toolbar_visible)))
            if editor.has_sidebar:
                items.append(("侧栏", "", "screen.region_toggle", {"region": "SIDEBAR"},
                              lambda: setattr(editor, "sidebar_visible", not editor.sidebar_visible)))
        items.append(None)
        items.append(("关闭区域", "close", "screen.area_close", {},
                      lambda: screen.close_area(self) if screen else None))
        return items

    def show_header_menu(self, global_pos: Any) -> QMenu:
        """在 global_pos 处弹出标题栏右键菜单。"""
        menu = QMenu(self)
        menu.setAttribute(Qt.WA_DeleteOnClose, True)
        ctx = self._context()
        kc = getattr(self.app, "keyconfig", None)
        editor = self._editor
        for item in self.header_menu_items():
            if item is None:
                menu.addSeparator()
                continue
            text, icon_name, op_name, props, fallback = item
            shortcut = kc.shortcut_for(op_name, props or None) if kc is not None else ""
            action = menu.addAction(text + ("\t" + shortcut if shortcut else ""))
            if icon_name:
                action.setIcon(icons.icon(icon_name))
            if op_name == "screen.region_toggle" and editor is not None:
                action.setCheckable(True)
                action.setChecked(editor.toolbar_visible if props.get("region") == "TOOLBAR"
                                  else editor.sidebar_visible)
            registered = ops.get(op_name) is not None
            if registered:
                action.setEnabled(ops.poll(op_name, ctx))
            else:
                action.setEnabled(fallback is not None and self._fallback_enabled(op_name))
            action.triggered.connect(lambda _checked=False, o=op_name, p=dict(props), f=fallback:
                                     self._run(o, p, f))
        menu.popup(_to_point(global_pos))
        return menu

    def _fallback_enabled(self, op_name: str) -> bool:
        screen = self.parent_screen()
        if op_name == "screen.area_close":
            return screen is not None and len(screen.areas()) > 1
        return screen is not None

    def _run(self, op_name: str, props: dict, fallback: Callable[[], Any] | None) -> None:
        if not alive(self):
            return
        if ops.get(op_name) is not None:
            ops.call(op_name, self._context(), **props)
        elif fallback is not None:
            fallback()

    # ---- Qt ----
    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._editor is not None:
            self._editor.setGeometry(self.rect())
        for zone in self._corners:
            zone.place()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.area"))

    def _on_theme_changed(self) -> None:
        if not alive(self):
            return
        self._type_button.apply_theme()
        for zone in self._corners:
            zone.place()

    def _on_registry_changed(self, kind: str = "") -> None:
        if not alive(self) or kind != "editors":
            return
        editor = self._editor
        if isinstance(editor, UnavailableEditor) and editor.missing_idname and editor.reason != "error":
            if registry.editor(editor.missing_idname) is not None:
                # 编辑器后注册进来了：换成真的，状态照旧
                self.set_editor(editor.missing_idname, editor.save_state())
                return
        self._type_button.sync()


# ======================================================================
# 屏幕
# ======================================================================

class Screen(QWidget):
    """一个工作区的区域拆分树。见模块说明。"""

    layoutChanged = Signal()          # 结构、比例、编辑器类型或区域开关变了（用来触发保存）
    maximizedChanged = Signal(bool)

    def __init__(self, app: Any, layout: dict | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app = app
        self._root: Any = None
        self._maximized: AreaWidget | None = None
        self._drag: _CornerDrag | None = None
        self._split_drag: tuple[_Split, int, dict] | None = None
        self._key_filter: _DragKeyFilter | None = None
        self._cursor_overridden = False
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self._overlay = DragOverlay(self)
        theme.changed.connect(self._on_theme_changed)
        self.from_dict(layout if isinstance(layout, dict) else {"type": "area", "editor": "VIEW_3D", "state": {}})

    # ---- 查询 ----
    @property
    def maximized_area(self) -> AreaWidget | None:
        return self._maximized

    @property
    def dragging(self) -> bool:
        return self._drag is not None or self._split_drag is not None

    @property
    def drag_state(self) -> dict:
        """拖动角落时的预览状态（测试和展示用）。"""
        d = self._drag
        if d is None:
            return {}
        return {"active": d.active, "mode": d.mode, "orientation": d.orientation, "target": d.target,
                "valid": d.valid, "split_pos": d.split_pos, "area": d.area}

    @property
    def overlay(self) -> DragOverlay:
        return self._overlay

    def areas(self) -> list[AreaWidget]:
        """全部区域，按从左到右、从上到下的树序。"""
        return [leaf.area for leaf in _iter_leaves(self._root)]

    def splitter_handles(self) -> list[SplitterHandle]:
        return [s.handle for s in _iter_splits(self._root) if s.handle is not None]

    def area_at(self, global_pos: Any) -> AreaWidget | None:
        return self._area_at_local(self.mapFromGlobal(_to_point(global_pos)))

    def find_editor(self, idname: str) -> Editor | None:
        """第一个 idname 类型的编辑器。"""
        for area in self.areas():
            if area.editor_idname() == idname and area.editor is not None:
                return area.editor
        return None

    def find_area(self, idname: str) -> AreaWidget | None:
        for area in self.areas():
            if area.editor_idname() == idname:
                return area
        return None

    def splitter_for(self, area: AreaWidget, side: str) -> SplitterHandle | None:
        """区域某一边（"left" "right" "top" "bottom"）紧挨着的那条分隔缝。"""
        leaf = self._leaf_of(area)
        node: Any = leaf
        want = {"left": (H, "b"), "right": (H, "a"), "top": (V, "b"), "bottom": (V, "a")}.get(side)
        if want is None:
            return None
        while node is not None and node.parent is not None:
            parent = node.parent
            if parent.orientation == want[0] and getattr(parent, want[1]) is node:
                return parent.handle
            node = parent
        return None

    @property
    def wm(self) -> Any:
        return getattr(self.app, "wm", None)

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return self._node_dict(self._root) if self._root is not None else {}

    def _node_dict(self, node: Any) -> dict:
        if isinstance(node, _Leaf):
            return node.area.to_dict()
        return {"type": "split", "orientation": node.orientation, "ratio": node.ratio,
                "a": self._node_dict(node.a), "b": self._node_dict(node.b)}

    def from_dict(self, data: dict) -> None:
        """按字典重建整个布局。坏数据不会让它崩：认不出的节点变成一个空区域。"""
        self._end_corner_drag()
        self._split_drag = None
        self._maximized = None
        for leaf in _iter_leaves(self._root):
            self._dispose_area(leaf.area)
        for split in _iter_splits(self._root):
            self._dispose_handle(split)
        self._root = self._build_node(data, 0)
        self._root.parent = None
        self._structure_changed(emit=False)

    def _build_node(self, data: Any, depth: int) -> Any:
        if isinstance(data, dict) and data.get("type") == "split" and depth < _MAX_DEPTH:
            orientation = V if str(data.get("orientation", H)).upper() == V else H
            ratio = _clamp01(data.get("ratio", 0.5))
            a = self._build_node(data.get("a"), depth + 1)
            b = self._build_node(data.get("b"), depth + 1)
            return _Split(orientation, ratio, a, b)
        if not isinstance(data, dict):
            data = {}
        state = data.get("state")
        area = AreaWidget(self.app, self, str(data.get("editor") or ""),
                          copy.deepcopy(state) if isinstance(state, dict) else {}, data.get("stash"))
        return _Leaf(area)

    # ---- 结构修改 ----
    def split_area(self, area: AreaWidget, orientation: str = H, ratio: float = 0.5, *, new_first: bool = False,
                   editor: str | None = None) -> AreaWidget | None:
        """把区域拆成两块，返回新区域。orientation：H 左右，V 上下。ratio 是左（上）块的比例。

        新区域默认在右（下）边，new_first=True 时在左（上）边；编辑器类型和状态复制原区域。
        区域太小放不下两块时返回 None。
        """
        leaf = self._leaf_of(area)
        if leaf is None:
            return None
        orientation = V if str(orientation).upper() == V else H
        ratio = _clamp01(ratio)
        if self._maximized is not None:
            self.restore()
        if not self._can_split(leaf, orientation):
            return None
        idname = editor if editor is not None else area.editor_idname()
        state = area.editor_state() if idname == area.editor_idname() else None
        new_area = AreaWidget(self.app, self, idname, state)
        new_leaf = _Leaf(new_area)
        parent = leaf.parent
        was_a = parent is not None and parent.a is leaf
        node = _Split(orientation, ratio, new_leaf, leaf) if new_first else _Split(orientation, ratio, leaf, new_leaf)
        node.parent = parent
        if parent is None:
            self._root = node
        elif was_a:
            parent.a = node
        else:
            parent.b = node
        self._structure_changed()
        return new_area

    def can_join(self, keep: AreaWidget, remove: AreaWidget) -> bool:
        """两个区域能不能合并：必须相邻并且共用一整条边。"""
        lk, lr = self._leaf_of(keep), self._leaf_of(remove)
        if lk is None or lr is None or lk is lr:
            return False
        if lk.parent is not None and lk.parent is lr.parent:
            return True
        return self._join_plan(lk, lr) is not None

    def join_areas(self, keep: AreaWidget, remove: AreaWidget) -> bool:
        """把 remove 并进 keep：remove 消失，keep 占据两者的位置。不能合并时返回 False，布局不变。"""
        lk, lr = self._leaf_of(keep), self._leaf_of(remove)
        if lk is None or lr is None or lk is lr:
            return False
        if self._maximized is not None:
            self.restore()
        if lk.parent is not None and lk.parent is lr.parent:
            parent = lk.parent
            self._dispose_handle(parent)
            self._replace(parent, lk)
        else:
            plan = self._join_plan(lk, lr)
            if plan is None:
                return False
            for split in _iter_splits(self._root):
                self._dispose_handle(split)
            self._root = self._materialize(plan)
            self._root.parent = None
        self._dispose_area(remove)
        self._structure_changed()
        return True

    def swap_areas(self, a: AreaWidget, b: AreaWidget) -> bool:
        """交换两个区域的位置（编辑器连同状态一起换）。"""
        la, lb = self._leaf_of(a), self._leaf_of(b)
        if la is None or lb is None or la is lb:
            return False
        if self._maximized is not None:
            self.restore()
        la.area, lb.area = b, a
        self._structure_changed()
        return True

    def close_area(self, area: AreaWidget) -> bool:
        """关掉一个区域，它旁边的那块补上来。只剩一个区域时不能关。"""
        leaf = self._leaf_of(area)
        if leaf is None or leaf.parent is None:
            return False
        if self._maximized is not None:
            self.restore()
        parent = leaf.parent
        sibling = parent.b if parent.a is leaf else parent.a
        self._dispose_handle(parent)
        self._replace(parent, sibling)
        self._dispose_area(area)
        self._structure_changed()
        return True

    def maximize(self, area: AreaWidget) -> bool:
        """把区域铺满整个 Screen，其它区域隐藏但不销毁。"""
        if self._leaf_of(area) is None:
            return False
        self._end_corner_drag()
        self._maximized = area
        for other in self.areas():
            other.setVisible(other is area)
        for handle in self.splitter_handles():
            handle.hide()
        self._sync_corner_zones()
        self._relayout()
        self.maximizedChanged.emit(True)
        return True

    def restore(self) -> None:
        """从最大化还原。"""
        if self._maximized is None:
            return
        self._maximized = None
        for area in self.areas():
            area.show()
        for handle in self.splitter_handles():
            handle.show()
        self._sync_corner_zones()
        self._relayout()
        self._restack()
        self.maximizedChanged.emit(False)

    def toggle_maximize(self, area: AreaWidget) -> None:
        if self._maximized is not None:
            self.restore()
        else:
            self.maximize(area)

    def refresh(self) -> None:
        """让每个编辑器重画。"""
        for area in self.areas():
            editor = area.editor
            if editor is not None and alive(editor):
                try:
                    editor.refresh()
                except Exception:  # noqa: BLE001
                    log.exception("编辑器 %s 刷新出错", getattr(editor, "idname", "?"))

    def mark_changed(self) -> None:
        """布局以外的区域状态变了（如工具栏开关），也需要保存。"""
        self.layoutChanged.emit()

    # ---- 树操作的内部实现 ----
    def _leaf_of(self, area: Any) -> _Leaf | None:
        for leaf in _iter_leaves(self._root):
            if leaf.area is area:
                return leaf
        return None

    def _replace(self, old: Any, new: Any) -> None:
        parent = old.parent
        new.parent = parent
        if parent is None:
            self._root = new
        elif parent.a is old:
            parent.a = new
        else:
            parent.b = new

    def _dispose_area(self, area: AreaWidget) -> None:
        if self._maximized is area:
            self._maximized = None
        if not alive(area):
            return
        area.dispose()
        area.hide()
        area.setParent(None)
        area.deleteLater()

    def _dispose_handle(self, split: _Split) -> None:
        handle = split.handle
        split.handle = None
        if handle is not None and alive(handle):
            handle.hide()
            handle.setParent(None)
            handle.deleteLater()

    def _structure_changed(self, emit: bool = True) -> None:
        for split in _iter_splits(self._root):
            if split.handle is None:
                split.handle = SplitterHandle(self, split)
        for area in self.areas():
            if area is self._maximized or self._maximized is None:
                area.show()
        for handle in self.splitter_handles():
            handle.setVisible(self._maximized is None)
        self._sync_corner_zones()
        self._update_minimum()
        self._relayout()
        self._restack()
        if emit:
            self.layoutChanged.emit()

    def _sync_corner_zones(self) -> None:
        enabled = self._maximized is None
        for area in self.areas():
            for zone in area.corner_zones:
                zone.set_enabled_for_drag(enabled)

    def _restack(self) -> None:
        for handle in self.splitter_handles():
            handle.raise_()
        if self._overlay.isVisible():
            self._overlay.raise_()

    # ---- 几何 ----
    def gap(self) -> int:
        """区域之间缝的宽度。"""
        return theme.size("splitter")

    def margin(self) -> int:
        """Screen 四周留的缝，和区域之间一样宽，所以每块区域四周的缝都均匀。"""
        return theme.size("splitter")

    def min_area_size(self) -> QSize:
        width = theme.px(settings.min_area_width)
        height = max(theme.px(settings.min_area_height), theme.size("header") + theme.px(8))
        return QSize(width, height)

    def content_rect(self) -> QRect:
        m = self.margin()
        return QRect(m, m, max(0, self.width() - 2 * m), max(0, self.height() - 2 * m))

    def _min_size(self, node: Any, cache: dict) -> tuple[int, int]:
        cached = cache.get(id(node))
        if cached is not None:
            return cached
        if isinstance(node, _Leaf):
            size = self.min_area_size()
            result = (size.width(), size.height())
        else:
            a = self._min_size(node.a, cache)
            b = self._min_size(node.b, cache)
            gap = self.gap()
            if node.orientation == H:
                result = (a[0] + gap + b[0], max(a[1], b[1]))
            else:
                result = (max(a[0], b[0]), a[1] + gap + b[1])
        cache[id(node)] = result
        return result

    def _extent(self, node: _Split, avail: int, cache: dict) -> int:
        index = 0 if node.orientation == H else 1
        low = self._min_size(node.a, cache)[index]
        low_b = self._min_size(node.b, cache)[index]
        high = avail - low_b
        if high < low:
            # 放不下：按最小尺寸的比例分（窗口有最小尺寸，正常走不到这里）
            total = low + low_b
            return max(0, min(avail, int(round(avail * low / total)))) if total > 0 else avail // 2
        return max(low, min(high, int(round(avail * node.ratio)))) if avail > 0 else 0

    def _compute(self, node: Any, rect: QRect, out: dict, pins: dict | None, cache: dict) -> None:
        """算出每个节点的矩形（不动控件）。pins 里的拆分节点按给定的分隔线位置反推比例。"""
        if isinstance(node, _Leaf):
            out[node] = (QRect(rect), None)
            return
        gap = self.gap()
        horizontal = node.orientation == H
        total = rect.width() if horizontal else rect.height()
        avail = max(0, total - gap)
        if pins and node in pins and avail > 0:
            origin = rect.x() if horizontal else rect.y()
            node.ratio = _clamp01((pins[node] - origin) / avail)
        extent = self._extent(node, avail, cache)
        if horizontal:
            ra = QRect(rect.x(), rect.y(), extent, rect.height())
            divider = QRect(rect.x() + extent, rect.y(), gap, rect.height())
            rb = QRect(rect.x() + extent + gap, rect.y(), max(0, total - extent - gap), rect.height())
        else:
            ra = QRect(rect.x(), rect.y(), rect.width(), extent)
            divider = QRect(rect.x(), rect.y() + extent, rect.width(), gap)
            rb = QRect(rect.x(), rect.y() + extent + gap, rect.width(), max(0, total - extent - gap))
        out[node] = (QRect(rect), divider)
        self._compute(node.a, ra, out, pins, cache)
        self._compute(node.b, rb, out, pins, cache)

    def _layout_rects(self, rect: QRect | None = None) -> dict:
        out: dict = {}
        if self._root is not None:
            if rect is None:
                rect = self.content_rect()
                if rect.width() <= 0 or rect.height() <= 0:
                    rect = QRect(0, 0, 4096, 4096)    # 还没摆上屏幕时用一块足够大的虚拟矩形判断
            self._compute(self._root, rect, out, None, {})
        return out

    def _relayout(self, pins: dict | None = None) -> None:
        if self._root is None:
            return
        rect = self.content_rect()
        self._overlay.setGeometry(self.rect())
        out: dict = {}
        self._compute(self._root, rect, out, pins, {})
        grab = theme.px(settings.edge_grab) if settings.edge_grab > 0 else 0
        for node, (node_rect, divider) in out.items():
            node.rect = node_rect
            if isinstance(node, _Leaf):
                if self._maximized is None and node.area.geometry() != node_rect:
                    node.area.setGeometry(node_rect)
                continue
            node.divider = divider
            if node.handle is not None:
                if node.orientation == H:
                    hit = QRect(divider.x() - grab, divider.y(), divider.width() + 2 * grab, divider.height())
                else:
                    hit = QRect(divider.x(), divider.y() - grab, divider.width(), divider.height() + 2 * grab)
                node.handle.setGeometry(hit)
        if self._maximized is not None and alive(self._maximized):
            self._maximized.setGeometry(rect)

    def _update_minimum(self) -> None:
        if self._root is None:
            return
        width, height = self._min_size(self._root, {})
        m = self.margin()
        self.setMinimumSize(width + 2 * m, height + 2 * m)

    def _area_at_local(self, pos: QPoint) -> AreaWidget | None:
        if self._maximized is not None:
            return self._maximized if self._maximized.geometry().contains(pos) else None
        for area in self.areas():
            if area.isVisible() and area.geometry().contains(pos):
                return area
        return None

    def _can_split(self, leaf: _Leaf, orientation: str) -> bool:
        rect = leaf.rect
        if self.width() <= 0 or rect.width() <= 0 or rect.height() <= 0:
            return True     # 还没摆上屏幕，按编程调用处理，不判断大小
        size = self.min_area_size()
        need = size.width() if orientation == H else size.height()
        extent = rect.width() if orientation == H else rect.height()
        return extent >= 2 * need + self.gap()

    # ---- 合并：任意两块共用整条边的区域 ----
    def _shares_full_edge(self, ra: QRect, rb: QRect) -> bool:
        gap = self.gap()
        if ra.y() == rb.y() and ra.height() == rb.height():
            return ra.x() + ra.width() + gap == rb.x() or rb.x() + rb.width() + gap == ra.x()
        if ra.x() == rb.x() and ra.width() == rb.width():
            return ra.y() + ra.height() + gap == rb.y() or rb.y() + rb.height() + gap == ra.y()
        return False

    def _join_plan(self, keep: _Leaf, remove: _Leaf) -> Any:
        """两块不是兄弟节点但共用整条边时，按合并后的矩形重新切出一棵树。切不出时返回 None。"""
        rects = self._layout_rects()
        if keep not in rects or remove not in rects:
            return None
        ra, rb = rects[keep][0], rects[remove][0]
        if not self._shares_full_edge(ra, rb):
            return None
        items = [(value[0], node) for node, value in rects.items()
                 if isinstance(node, _Leaf) and node is not keep and node is not remove]
        items.append((ra.united(rb), keep))
        prefer = []
        for node, (_rect, divider) in rects.items():
            if isinstance(node, _Split):
                prefer.append((node.orientation, divider.x() if node.orientation == H else divider.y()))
        bounds = rects[self._root][0]
        return self._guillotine(items, bounds, prefer, 0)

    def _guillotine(self, items: list, bounds: QRect, prefer: list, depth: int) -> Any:
        if not items or depth > _MAX_DEPTH:
            return None
        if len(items) == 1:
            rect, leaf = items[0]
            return ("leaf", leaf) if rect == bounds else None
        gap = self.gap()
        right_edge = bounds.x() + bounds.width()
        bottom_edge = bounds.y() + bounds.height()
        candidates = []
        for x in sorted({r.x() + r.width() for r, _ in items if r.x() + r.width() < right_edge}):
            if all(r.x() + r.width() <= x or r.x() >= x + gap for r, _ in items):
                candidates.append((H, x))
        for y in sorted({r.y() + r.height() for r, _ in items if r.y() + r.height() < bottom_edge}):
            if all(r.y() + r.height() <= y or r.y() >= y + gap for r, _ in items):
                candidates.append((V, y))
        if not candidates:
            return None
        choice = next((p for p in prefer if p in candidates), candidates[0])
        orientation, pos = choice
        if orientation == H:
            first = [(r, n) for r, n in items if r.x() + r.width() <= pos]
            second = [(r, n) for r, n in items if r.x() >= pos + gap]
            box_a = QRect(bounds.x(), bounds.y(), pos - bounds.x(), bounds.height())
            box_b = QRect(pos + gap, bounds.y(), right_edge - pos - gap, bounds.height())
            avail = bounds.width() - gap
            ratio = (pos - bounds.x()) / avail if avail > 0 else 0.5
        else:
            first = [(r, n) for r, n in items if r.y() + r.height() <= pos]
            second = [(r, n) for r, n in items if r.y() >= pos + gap]
            box_a = QRect(bounds.x(), bounds.y(), bounds.width(), pos - bounds.y())
            box_b = QRect(bounds.x(), pos + gap, bounds.width(), bottom_edge - pos - gap)
            avail = bounds.height() - gap
            ratio = (pos - bounds.y()) / avail if avail > 0 else 0.5
        plan_a = self._guillotine(first, box_a, prefer, depth + 1)
        plan_b = self._guillotine(second, box_b, prefer, depth + 1)
        if plan_a is None or plan_b is None:
            return None
        return ("split", orientation, _clamp01(ratio), plan_a, plan_b)

    def _materialize(self, plan: Any) -> Any:
        if plan[0] == "leaf":
            leaf = plan[1]
            leaf.parent = None
            return leaf
        _kind, orientation, ratio, plan_a, plan_b = plan
        return _Split(orientation, ratio, self._materialize(plan_a), self._materialize(plan_b))

    # ---- 拖分隔缝 ----
    def begin_splitter_drag(self, node: _Split, global_pos: QPoint) -> bool:
        if self._maximized is not None or self._drag is not None:
            return False
        pos = self.mapFromGlobal(global_pos)
        horizontal = node.orientation == H
        offset = pos.x() - node.divider.x() if horizontal else pos.y() - node.divider.y()
        pins = {}
        for child in (node.a, node.b):
            for split in _iter_splits(child):
                if split.orientation == node.orientation:
                    pins[split] = split.divider.x() if horizontal else split.divider.y()
        self._split_drag = (node, offset, pins)
        self._override_cursor(Qt.SplitHCursor if horizontal else Qt.SplitVCursor)
        return True

    def update_splitter_drag(self, global_pos: QPoint) -> None:
        if self._split_drag is None:
            return
        node, offset, pins = self._split_drag
        pos = self.mapFromGlobal(global_pos)
        horizontal = node.orientation == H
        rect = node.rect
        gap = self.gap()
        avail = (rect.width() if horizontal else rect.height()) - gap
        if avail <= 0:
            return
        want = (pos.x() - offset - rect.x()) if horizontal else (pos.y() - offset - rect.y())
        cache: dict = {}
        index = 0 if horizontal else 1
        low = self._min_size(node.a, cache)[index]
        high = avail - self._min_size(node.b, cache)[index]
        if high < low:
            return
        extent = max(low, min(high, int(want)))
        node.ratio = extent / avail
        self._relayout(pins)

    def finish_splitter_drag(self, global_pos: QPoint) -> None:
        if self._split_drag is None:
            return
        self.update_splitter_drag(global_pos)
        self._split_drag = None
        self._restore_cursor()
        self.layoutChanged.emit()

    # ---- 拖角落 ----
    def begin_corner_drag(self, area: AreaWidget, corner: str, global_pos: QPoint) -> bool:
        if self._maximized is not None or self._drag is not None or self._split_drag is not None:
            return False
        if self._leaf_of(area) is None:
            return False
        self._drag = _CornerDrag(area, corner, self.mapFromGlobal(global_pos))
        self._key_filter = _DragKeyFilter(self)
        QApplication.instance().installEventFilter(self._key_filter)
        return True

    def update_corner_drag(self, global_pos: QPoint, modifiers: Any = None) -> None:
        d = self._drag
        if d is None:
            return
        d.pos = self.mapFromGlobal(global_pos)
        if modifiers is not None:
            d.ctrl = bool(modifiers & Qt.ControlModifier)
        if not d.active:
            delta = d.pos - d.start
            if max(abs(delta.x()), abs(delta.y())) < theme.px(settings.drag_threshold):
                return
            d.active = True
            self._overlay.setGeometry(self.rect())
            self._overlay.show()
            self._overlay.raise_()
            self._override_cursor(Qt.CrossCursor)
        self._evaluate_drag()

    def set_drag_ctrl(self, pressed: bool) -> None:
        d = self._drag
        if d is None or d.ctrl == pressed:
            return
        d.ctrl = pressed
        if d.active:
            self._evaluate_drag()

    def finish_corner_drag(self, global_pos: QPoint, modifiers: Any = None) -> None:
        d = self._drag
        if d is None:
            return
        self.update_corner_drag(global_pos, modifiers)
        self._end_corner_drag()
        if not d.active or not d.valid or d.mode is None:
            return
        if d.mode == "SWAP" and d.target is not None:
            self.swap_areas(d.area, d.target)
        elif d.mode == "JOIN" and d.target is not None:
            self.join_areas(d.area, d.target)
        elif d.mode == "SPLIT":
            rect = d.area.geometry()
            gap = self.gap()
            if d.orientation == H:
                avail = rect.width() - gap
                ratio = (d.split_pos - rect.x()) / avail if avail > 0 else 0.5
                new_first = d.corner in ("TL", "BL")
            else:
                avail = rect.height() - gap
                ratio = (d.split_pos - rect.y()) / avail if avail > 0 else 0.5
                new_first = d.corner in ("TL", "TR")
            self.split_area(d.area, d.orientation or H, ratio, new_first=new_first)

    def cancel_drag(self) -> None:
        """取消正在进行的角落拖动或分隔缝拖动。"""
        if self._split_drag is not None:
            self._split_drag = None
            self._restore_cursor()
        self._end_corner_drag()

    def _end_corner_drag(self) -> None:
        if self._key_filter is not None:
            app = QApplication.instance()
            if app is not None:
                app.removeEventFilter(self._key_filter)
            self._key_filter.deleteLater()
            self._key_filter = None
        if self._drag is not None:
            self._drag = None
            self._overlay.hide()
            self._restore_cursor()

    def _evaluate_drag(self) -> None:
        d = self._drag
        if d is None:
            return
        source = d.area
        over = self._area_at_local(d.pos)
        threshold = theme.px(settings.drag_threshold)
        if d.ctrl:
            d.mode = "SWAP"
            d.target = over if over is not None and over is not source else None
            d.valid = d.target is not None
        elif over is source:
            d.mode = "SPLIT"
            d.target = None
            delta = d.pos - d.start
            dx, dy = abs(delta.x()), abs(delta.y())
            if d.orientation is None:
                d.orientation = H if dx >= dy else V
            elif d.orientation == H and dy > 2 * dx and dy > 2 * threshold:
                d.orientation = V
            elif d.orientation == V and dx > 2 * dy and dx > 2 * threshold:
                d.orientation = H
            rect = source.geometry()
            gap = self.gap()
            size = self.min_area_size()
            if d.orientation == H:
                low = rect.x() + size.width()
                high = rect.x() + rect.width() - size.width() - gap
                want = d.pos.x() - gap // 2
            else:
                low = rect.y() + size.height()
                high = rect.y() + rect.height() - size.height() - gap
                want = d.pos.y() - gap // 2
            d.valid = high >= low
            d.split_pos = max(low, min(high, want)) if d.valid else want
        elif over is not None:
            d.mode = "JOIN"
            d.target = over
            d.valid = self.can_join(source, over)
        else:
            d.mode = None
            d.target = None
            d.valid = False
        if not d.valid:
            cursor = Qt.ForbiddenCursor
        elif d.mode == "SPLIT":
            cursor = Qt.SplitHCursor if d.orientation == H else Qt.SplitVCursor
        elif d.mode == "SWAP":
            cursor = Qt.DragMoveCursor
        else:
            cursor = Qt.CrossCursor
        self._override_cursor(cursor)
        self._overlay.update()

    def _override_cursor(self, shape: Any) -> None:
        if self._cursor_overridden:
            QApplication.changeOverrideCursor(QCursor(shape))
        else:
            QApplication.setOverrideCursor(QCursor(shape))
            self._cursor_overridden = True

    def _restore_cursor(self) -> None:
        if self._cursor_overridden:
            QApplication.restoreOverrideCursor()
            self._cursor_overridden = False

    # ---- 预览绘制 ----
    def paint_drag_preview(self, painter: QPainter) -> None:
        d = self._drag
        if d is None or not d.active or d.mode is None or not d.valid:
            return
        radius = float(theme.size("radius"))
        if d.mode == "SPLIT":
            rect = d.area.geometry()
            line = max(theme.px(2), self.gap())
            if d.orientation == H:
                bar = QRectF(d.split_pos + (self.gap() - line) / 2.0, rect.y(), line, rect.height())
                new_side = (QRectF(rect.x(), rect.y(), d.split_pos - rect.x(), rect.height())
                            if d.corner in ("TL", "BL") else
                            QRectF(d.split_pos + self.gap(), rect.y(),
                                   rect.x() + rect.width() - d.split_pos - self.gap(), rect.height()))
            else:
                bar = QRectF(rect.x(), d.split_pos + (self.gap() - line) / 2.0, rect.width(), line)
                new_side = (QRectF(rect.x(), rect.y(), rect.width(), d.split_pos - rect.y())
                            if d.corner in ("TL", "TR") else
                            QRectF(rect.x(), d.split_pos + self.gap(), rect.width(),
                                   rect.y() + rect.height() - d.split_pos - self.gap()))
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme.qcolor("accent", 0.10))
            painter.drawRoundedRect(new_side, radius, radius)
            painter.setBrush(theme.qcolor("text", 0.92))
            painter.drawRect(bar)
            return
        target = d.target
        if target is None:
            return
        trect = QRectF(target.geometry())
        if d.mode == "JOIN":
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme.qcolor("bg.tooltip", 0.62))
            painter.drawRoundedRect(trect, radius, radius)
            srect = QRectF(d.area.geometry())
            dx = trect.center().x() - srect.center().x()
            dy = trect.center().y() - srect.center().y()
            direction = (1 if dx > 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy > 0 else -1)
            length = max(theme.px(28), min(theme.px(120), min(trect.width(), trect.height()) * 0.42))
            painter.setBrush(theme.qcolor("text", 0.88))
            painter.drawPath(_arrow_path(trect.center(), direction, length))
        elif d.mode == "SWAP":
            srect = QRectF(d.area.geometry())
            painter.setBrush(theme.qcolor("accent", 0.16))
            painter.setPen(QPen(theme.qcolor("accent"), theme.px(2)))
            inset = theme.px(1)
            painter.drawRoundedRect(trect.adjusted(inset, inset, -inset, -inset), radius, radius)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(theme.qcolor("accent", 0.55), theme.px(2), Qt.DashLine))
            painter.drawRoundedRect(srect.adjusted(inset, inset, -inset, -inset), radius, radius)
            side = int(max(theme.px(24), min(theme.px(64), min(trect.width(), trect.height()) * 0.3)))
            pix = icons.pixmap("swap", theme.color("text"), side, painter.device().devicePixelRatioF())
            center = trect.center()
            painter.drawPixmap(QPoint(int(center.x() - side / 2), int(center.y() - side / 2)), pix)

    # ---- Qt ----
    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.window"))

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._relayout()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self._relayout()

    def _on_theme_changed(self) -> None:
        if not alive(self):
            return
        self._update_minimum()
        self._relayout()
        self.update()


def _arrow_path(center: QPointF, direction: tuple[int, int], length: float) -> QPainterPath:
    """以 center 为中心、指向 direction 的实心箭头。"""
    half = length / 2.0
    shaft = length * 0.2
    head_w = length * 0.62
    head_l = length * 0.46
    points = [(-half, -shaft / 2), (half - head_l, -shaft / 2), (half - head_l, -head_w / 2), (half, 0.0),
              (half - head_l, head_w / 2), (half - head_l, shaft / 2), (-half, shaft / 2)]
    angle = {(1, 0): 0.0, (0, 1): 90.0, (-1, 0): 180.0, (0, -1): 270.0}.get(direction, 0.0)
    transform = QTransform()
    transform.translate(center.x(), center.y())
    transform.rotate(angle)
    polygon = QPolygonF([transform.map(QPointF(x, y)) for x, y in points])
    path = QPainterPath()
    path.addPolygon(polygon)
    path.closeSubpath()
    return path
