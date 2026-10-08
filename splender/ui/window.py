"""主窗口：顶栏（标志、程序菜单、工作区标签、工程名）、当前工作区的 Screen、状态栏；以及弹出的独立窗口。

工作区：
- 每个工作区记住自己的区域布局和编辑器状态。第一次切过去时才建 Screen，之后切走只是隐藏；
- 标签点击切换，双击改名，右键复制、改名、删除、左右移动，拖动排序，加号新建；
- 布局有变动就写进 app.prefs.workspaces（同一轮事件循环里合并成一次）。
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPolygonF
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMenuBar,
                               QProgressBar, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..core import registry
from ..paths import resource
from . import icons, theme, workspaces
from .area import Screen, settings
from .editor import alive

log = logging.getLogger("splender.ui.window")


# ======================================================================
# 顶栏
# ======================================================================

class LogoMark(QWidget):
    """顶栏最左的小标志：程序图标（resources/logo/splender_64.png）；没有时用 logo.svg，再没有就画一颗宝石形的记号。"""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setToolTip("%s %s" % (APP_NAME, __version__))
        self._svg = None
        self._pixmap = None
        png = resource("logo", "splender_64.png")
        if png.is_file():
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(str(png))
            self._pixmap = pixmap if not pixmap.isNull() else None
        path = resource("logo", "logo.svg")
        if path.is_file():
            try:
                from PySide6.QtSvg import QSvgRenderer
                self._svg = QSvgRenderer(str(path))
            except Exception:  # noqa: BLE001
                self._svg = None
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setFixedSize(theme.px(26), theme.size("topbar"))
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        side = theme.px(16)
        box = QRectF((self.width() - side) / 2.0, (self.height() - side) / 2.0, side, side)
        if self._pixmap is not None:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            big = theme.px(18)
            painter.drawPixmap(QRectF((self.width() - big) / 2.0, (self.height() - big) / 2.0, big, big),
                               self._pixmap, QRectF(self._pixmap.rect()))
            return
        if self._svg is not None and self._svg.isValid():
            self._svg.render(painter, box)
            return
        c = box.center()
        r = side / 2.0
        outer = QPolygonF([QPointF(c.x(), c.y() - r), QPointF(c.x() + r * 0.86, c.y()),
                           QPointF(c.x(), c.y() + r), QPointF(c.x() - r * 0.86, c.y())])
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.qcolor("accent"))
        painter.drawPolygon(outer)
        facet = QPolygonF([QPointF(c.x(), c.y() - r), QPointF(c.x() + r * 0.86, c.y()), QPointF(c.x(), c.y())])
        painter.setBrush(theme.qcolor("text.on_accent", 0.35))
        painter.drawPolygon(facet)
        facet2 = QPolygonF([QPointF(c.x(), c.y()), QPointF(c.x() - r * 0.86, c.y()), QPointF(c.x(), c.y() + r)])
        painter.setBrush(theme.qcolor("bg.window", 0.25))
        painter.drawPolygon(facet2)


class TopMenuBar(QMenuBar):
    """程序菜单：文件、编辑、窗口、帮助。内容在打开时由 registry.menu(...) 现生成。"""

    MENUS = (("文件", "TOPBAR_MT_file"), ("编辑", "TOPBAR_MT_edit"), ("窗口", "TOPBAR_MT_window"),
             ("帮助", "TOPBAR_MT_help"))

    def __init__(self, window: "MainWindow", parent: QWidget) -> None:
        super().__init__(parent)
        self.setNativeMenuBar(False)
        self._window = window
        self._built: dict[str, Any] = {}
        self.menus: dict[str, QMenu] = {}
        for title, idname in self.MENUS:
            menu = self.addMenu(title)
            menu.setObjectName(idname)
            menu.aboutToShow.connect(lambda m=menu, i=idname: self.populate(m, i))
            self.menus[idname] = menu
        self.apply_theme()

    def apply_theme(self) -> None:
        c = theme.color
        r = theme.size("radius")
        self.setFont(theme.font())
        self.setStyleSheet(
            "QMenuBar { background: transparent; border: none; spacing: %dpx; }"
            "QMenuBar::item { background: transparent; color: %s; padding: %dpx %dpx; border-radius: %dpx; }"
            "QMenuBar::item:selected { background: %s; }"
            "QMenuBar::item:pressed { background: %s; color: %s; }"
            % (theme.px(1), c("text"), theme.px(3), theme.px(9), r, c("bg.widget_hover"), c("accent"),
               c("text.on_accent")))
        self.setFixedHeight(theme.size("topbar"))

    def populate(self, menu: QMenu, idname: str) -> None:
        """打开菜单前现生成内容。菜单类没注册时就是空菜单。"""
        old = self._built.pop(idname, None)
        menu.clear()
        if old is not None and alive(old):
            old.deleteLater()
        if registry.menu(idname) is None:
            return
        try:
            from .widgets import build_menu
            wm = getattr(self._window.app, "wm", None)
            ctx = wm.context(window=self._window) if wm is not None else None
            built = build_menu(idname, ctx, menu)
        except Exception:  # noqa: BLE001
            log.exception("生成菜单 %s 出错", idname)
            return
        if built is None or built is menu:
            return
        for action in built.actions():
            menu.addAction(action)
        self._built[idname] = built


class _RenameEdit(QLineEdit):
    """工作区标签上的改名输入框：回车确认，Esc 取消，点别处也确认。"""

    cancelled = Signal()

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class WorkspaceTabs(QWidget):
    """工作区标签：点击切换，双击改名，右键复制与删除，拖动排序，加号新建。"""

    PLUS = -2

    def __init__(self, window: "MainWindow", parent: QWidget) -> None:
        super().__init__(parent)
        self._window = window
        self._names: list[str] = []
        self._current = -1
        self._hover = -1
        self._rects: list[QRect] = []
        self._plus = QRect()
        self._press = -1
        self._press_pos = QPoint()
        self._dragging = False
        self._edit: _RenameEdit | None = None
        self._editing = -1
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.apply_theme()

    # ---- 数据 ----
    def set_items(self, names: list[str], current: int) -> None:
        self._names = list(names)
        self._current = current
        self._relayout()
        self.updateGeometry()
        self.update()

    def names(self) -> list[str]:
        return list(self._names)

    def tab_rect(self, index: int) -> QRect:
        return QRect(self._rects[index]) if 0 <= index < len(self._rects) else QRect()

    def plus_rect(self) -> QRect:
        return QRect(self._plus)

    @property
    def rename_editor(self) -> QLineEdit | None:
        return self._edit

    # ---- 几何 ----
    def apply_theme(self) -> None:
        self.setFixedHeight(theme.size("topbar"))
        self._relayout()
        self.updateGeometry()
        self.update()

    def _relayout(self) -> None:
        fm = QFontMetrics(theme.font())
        height = theme.size("topbar")
        tab_h = max(theme.px(16), height - theme.px(8))
        y = (height - tab_h) // 2
        x = 0
        pad = theme.px(11)
        self._rects = []
        for name in self._names:
            width = fm.horizontalAdvance(name) + 2 * pad
            self._rects.append(QRect(x, y, width, tab_h))
            x += width + theme.px(2)
        self._plus = QRect(x + theme.px(2), y, tab_h, tab_h)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self._plus.x() + self._plus.width() + theme.px(2), theme.size("topbar"))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    def index_at(self, pos: QPoint) -> int:
        for index, rect in enumerate(self._rects):
            if rect.contains(pos):
                return index
        if self._plus.contains(pos):
            return self.PLUS
        return -1

    # ---- 绘制 ----
    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setFont(theme.font())
        radius = theme.size("radius")
        for index, rect in enumerate(self._rects):
            if index == self._editing:
                continue
            active = index == self._current
            hover = index == self._hover
            if active or hover:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.qcolor("bg.widget") if active else theme.qcolor("bg.widget", 0.55))
                painter.drawRoundedRect(QRectF(rect), radius, radius)
            painter.setPen(theme.qcolor("text" if active or hover else "text.dim"))
            painter.drawText(rect, Qt.AlignCenter, self._names[index])
        if self._hover == self.PLUS:
            painter.setPen(Qt.NoPen)
            painter.setBrush(theme.qcolor("bg.widget", 0.55))
            painter.drawRoundedRect(QRectF(self._plus), radius, radius)
        side = theme.px(14)
        pix = icons.pixmap("add", theme.color("text" if self._hover == self.PLUS else "text.dim"), side,
                           self.devicePixelRatioF())
        painter.drawPixmap(QPoint(self._plus.x() + (self._plus.width() - side) // 2,
                                  self._plus.y() + (self._plus.height() - side) // 2), pix)

    # ---- 鼠标 ----
    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        pos = event.position().toPoint()
        index = self.index_at(pos)
        if event.button() == Qt.LeftButton:
            if index >= 0:
                self._press = index
                self._press_pos = pos
                self._dragging = False
                self._window.set_workspace(index)
            elif index == self.PLUS:
                self.show_add_menu()
        elif event.button() == Qt.RightButton and index >= 0:
            self.show_context_menu(index, event.globalPosition().toPoint())

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802
        index = self.index_at(event.position().toPoint())
        if event.button() == Qt.LeftButton and index >= 0:
            self.start_rename(index)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        pos = event.position().toPoint()
        if self._press >= 0 and event.buttons() & Qt.LeftButton:
            if not self._dragging and abs(pos.x() - self._press_pos.x()) >= QApplication.startDragDistance():
                self._dragging = True
            if self._dragging:
                target = self._drop_index(pos.x())
                if target != self._press:
                    self._window.move_workspace(self._press, target)
                    self._press = target
            return
        hover = self.index_at(pos)
        if hover != self._hover:
            self._hover = hover
            self.update()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        self._press = -1
        self._dragging = False

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        if self._hover != -1:
            self._hover = -1
            self.update()

    def _drop_index(self, x: int) -> int:
        for index, rect in enumerate(self._rects):
            if x < rect.x() + rect.width():
                return index
        return max(0, len(self._rects) - 1)

    # ---- 菜单 ----
    def show_context_menu(self, index: int, global_pos: QPoint) -> QMenu:
        window = self._window
        menu = QMenu(self)
        menu.setAttribute(Qt.WA_DeleteOnClose, True)
        count = len(self._names)

        def add(text: str, icon_name: str, callback: Any, enabled: bool = True) -> None:
            action = menu.addAction(text)
            if icon_name:
                action.setIcon(icons.icon(icon_name))
            action.setEnabled(enabled)
            action.triggered.connect(lambda _checked=False: callback())

        add("复制", "duplicate", lambda: window.duplicate_workspace(index))
        add("重命名", "", lambda: self.start_rename(index))
        add("删除", "remove", lambda: window.remove_workspace(index), count > 1)
        menu.addSeparator()
        add("向左移动", "chevron.left", lambda: window.move_workspace(index, index - 1), index > 0)
        add("向右移动", "chevron.right", lambda: window.move_workspace(index, index + 1), index < count - 1)
        menu.popup(global_pos)
        return menu

    def show_add_menu(self) -> QMenu:
        window = self._window
        menu = QMenu(self)
        menu.setAttribute(Qt.WA_DeleteOnClose, True)
        action = menu.addAction("复制当前工作区")
        action.setIcon(icons.icon("duplicate"))
        action.triggered.connect(lambda _checked=False: window.duplicate_workspace())
        menu.addSeparator()
        for name in workspaces.DEFAULT_ORDER:
            item = menu.addAction(name)
            item.setIcon(icons.icon("workspace"))
            item.triggered.connect(lambda _checked=False, n=name: window.add_workspace(
                n, workspaces.default_layout(n)))
        menu.popup(self.mapToGlobal(QPoint(self._plus.x(), self._plus.y() + self._plus.height() + theme.px(2))))
        return menu

    # ---- 改名 ----
    def start_rename(self, index: int) -> QLineEdit | None:
        if not 0 <= index < len(self._names):
            return None
        self._finish_rename(commit=False)
        rect = self._rects[index]
        edit = _RenameEdit(self)
        edit.setText(self._names[index])
        edit.setAlignment(Qt.AlignCenter)
        width = max(rect.width(), theme.px(96))
        edit.setGeometry(QRect(rect.x(), rect.y(), width, rect.height()))
        edit.selectAll()
        edit.editingFinished.connect(lambda: self._finish_rename(commit=True))
        edit.cancelled.connect(lambda: self._finish_rename(commit=False))
        self._edit = edit
        self._editing = index
        edit.show()
        edit.setFocus(Qt.OtherFocusReason)
        self.update()
        return edit

    def _finish_rename(self, commit: bool) -> None:
        edit, index = self._edit, self._editing
        if edit is None:
            return
        self._edit = None
        self._editing = -1
        text = edit.text()
        edit.hide()
        edit.deleteLater()
        if commit:
            self._window.rename_workspace(index, text)
        self.update()


class WindowButton(QWidget):
    """顶栏右端的窗口按钮：最小化、最大化（还原）、关闭。图形用线条画，鼠标移上去变色，关闭键是红色。"""

    def __init__(self, parent: QWidget, kind: str, callback: Any) -> None:
        super().__init__(parent)
        self.kind = kind
        self.callback = callback
        self.maximized = False
        self._hover = False
        self._down = False
        self.setToolTip({"min": "最小化", "max": "最大化", "close": "关闭"}[kind])
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setFixedSize(theme.px(46), theme.size("topbar"))
        self.update()

    def enterEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event: Any) -> None:  # noqa: N802
        self._hover = False
        self._down = False
        self.update()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._down = True
            self.update()

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._down:
            self._down = False
            self.update()
            if self.rect().contains(event.position().toPoint()):
                self.callback()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        from PySide6.QtGui import QPen

        painter = QPainter(self)
        if self._hover or self._down:
            if self.kind == "close":
                color = QColor("#c42b1c") if not self._down else QColor("#a6231a")
            else:
                color = theme.qcolor("bg.widget_hover" if not self._down else "bg.widget_pressed")
            painter.fillRect(self.rect(), color)
        pen_color = QColor("#ffffff") if (self.kind == "close" and self._hover) else theme.qcolor("text")
        painter.setPen(QPen(pen_color, max(1.0, float(theme.px(1)))))
        painter.setRenderHint(QPainter.Antialiasing, self.kind == "close")
        side = float(theme.px(10))
        cx, cy = self.width() / 2.0, self.height() / 2.0
        box = QRectF(cx - side / 2.0, cy - side / 2.0, side, side)
        if self.kind == "min":
            painter.drawLine(QPointF(box.left(), cy), QPointF(box.right(), cy))
        elif self.kind == "max":
            if self.maximized:
                gap = float(theme.px(2))
                front = box.adjusted(0, gap, -gap, 0)
                painter.drawRect(front)
                painter.drawLine(QPointF(box.left() + gap, front.top()), QPointF(box.left() + gap, box.top()))
                painter.drawLine(QPointF(box.left() + gap, box.top()), QPointF(box.right(), box.top()))
                painter.drawLine(QPointF(box.right(), box.top()), QPointF(box.right(), box.bottom() - gap))
                painter.drawLine(QPointF(box.right(), box.bottom() - gap), QPointF(front.right(), box.bottom() - gap))
            else:
                painter.drawRect(box)
        else:
            painter.drawLine(box.topLeft(), box.bottomRight())
            painter.drawLine(box.topRight(), box.bottomLeft())


class TopBar(QWidget):
    """顶栏：标志、程序菜单、工作区标签，右边是工程名；没有系统标题栏时最右边是窗口按钮。"""

    def __init__(self, window: "MainWindow", window_buttons: bool = False) -> None:
        super().__init__(window)
        self._box = QHBoxLayout(self)
        self.logo = LogoMark(self)
        self.menus = TopMenuBar(window, self)
        self.tabs = WorkspaceTabs(window, self)
        self.title = QLabel(self)
        self.title.setProperty("role", "dim")
        self._box.addWidget(self.logo)
        self._box.addWidget(self.menus)
        self._gap = self._box.addSpacing(theme.px(14))
        self._box.addWidget(self.tabs)
        self._box.addStretch(1)
        self._box.addWidget(self.title)
        self.buttons: list[WindowButton] = []
        if window_buttons:
            self._box.addSpacing(theme.px(10))
            for kind, callback in (("min", window.showMinimized), ("max", window.toggle_maximized),
                                   ("close", window.close)):
                button = WindowButton(self, kind, callback)
                self._box.addWidget(button)
                self.buttons.append(button)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setFixedHeight(theme.size("topbar"))
        self._box.setContentsMargins(theme.px(4), 0, 0 if self.buttons else theme.px(12), 0)
        self._box.setSpacing(theme.px(2))
        self.logo.apply_theme()
        self.menus.apply_theme()
        self.tabs.apply_theme()
        for button in self.buttons:
            button.apply_theme()
        self.update()

    def set_maximized(self, value: bool) -> None:
        for button in self.buttons:
            if button.kind == "max":
                button.maximized = bool(value)
                button.setToolTip("向下还原" if value else "最大化")
                button.update()

    def is_caption(self, pos: QPoint) -> bool:
        """顶栏里这一点是不是可以拖动窗口的空白处（不是菜单、标签、按钮）。"""
        child = self.childAt(pos)
        if child is None or child is self.title:
            return True
        if child is self.menus:
            return self.menus.actionAt(self.menus.mapFrom(self, pos)) is None
        if child is self.tabs:
            local = self.tabs.mapFrom(self, pos)
            return self.tabs.index_at(local) < 0 and not self.tabs.plus_rect().contains(local)
        return False

    def set_title(self, text: str) -> None:
        self.title.setText(text)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.header"))


# ======================================================================
# 状态栏
# ======================================================================

class ElidedLabel(QWidget):
    """一行文字，放不下时末尾省略；可以带一个小图标。"""

    def __init__(self, parent: QWidget, align: Any = Qt.AlignLeft) -> None:
        super().__init__(parent)
        self._text = ""
        self._color = "text.dim"
        self._icon = ""
        self._align = align
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def set_text(self, text: str, color: str = "text.dim", icon: str = "") -> None:
        self._text = text or ""
        self._color = color
        self._icon = icon
        self.setToolTip(self._text if len(self._text) > 40 else "")
        self.update()

    def text(self) -> str:
        return self._text

    @property
    def color_name(self) -> str:
        return self._color

    def sizeHint(self) -> QSize:  # noqa: N802
        fm = QFontMetrics(theme.font())
        return QSize(fm.horizontalAdvance(self._text) + theme.px(24), fm.height())

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setFont(theme.font())
        rect = self.rect()
        if self._icon:
            side = theme.px(14)
            pix = icons.pixmap(self._icon, theme.color(self._color), side, self.devicePixelRatioF())
            painter.drawPixmap(QPoint(rect.x(), rect.y() + (rect.height() - side) // 2), pix)
            rect = rect.adjusted(side + theme.px(6), 0, 0, 0)
        fm = QFontMetrics(painter.font())
        text = fm.elidedText(self._text, Qt.ElideRight, max(0, rect.width()))
        painter.setPen(QColor(theme.color(self._color)))
        painter.drawText(rect, self._align | Qt.AlignVCenter, text)


class StatusBar(QWidget):
    """状态栏：左边操作提示（和临时消息），中间后台任务进度，右边显存内存等统计。"""

    _LEVEL_STYLE = {"ERROR": ("error", "error"), "WARNING": ("warn", "warning"), "INFO": ("text", "info"),
                    "OPERATOR": ("text", "info")}

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self._hint = ""
        self._reporting = False
        self.report_level = ""
        box = QHBoxLayout(self)
        self._box = box
        self.hint_label = ElidedLabel(self)
        self._progress = QWidget(self)
        progress_box = QHBoxLayout(self._progress)
        progress_box.setContentsMargins(0, 0, 0, 0)
        self._progress_box = progress_box
        self.progress_label = QLabel(self._progress)
        self.progress_label.setProperty("role", "dim")
        self.progress_bar = QProgressBar(self._progress)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(False)
        progress_box.addWidget(self.progress_label)
        progress_box.addWidget(self.progress_bar)
        self.stats_label = ElidedLabel(self, Qt.AlignRight)
        box.addWidget(self.hint_label, 1)
        box.addWidget(self._progress, 0)
        box.addWidget(self.stats_label, 1)
        self._progress.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._restore_hint)
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setFixedHeight(theme.size("statusbar"))
        self._box.setContentsMargins(theme.px(10), 0, theme.px(10), 0)
        self._box.setSpacing(theme.px(12))
        self._progress_box.setSpacing(theme.px(8))
        self.progress_bar.setFixedSize(theme.px(160), max(2, theme.px(4)))
        self.update()

    # ---- 接口 ----
    def set_hint(self, text: str) -> None:
        self._hint = text or ""
        if not self._reporting:
            self.hint_label.set_text(self._hint, "text.dim")

    def hint(self) -> str:
        return self._hint

    def show_report(self, text: str, level: str = "INFO", seconds: float | None = None) -> None:
        level = str(level or "INFO").upper()
        color, icon_name = self._LEVEL_STYLE.get(level, ("text", "info"))
        self._reporting = True
        self.report_level = level
        self.hint_label.set_text(text, color, icon_name)
        duration = seconds if seconds is not None else settings.report_seconds
        self._timer.start(max(100, int(duration * 1000)))

    def _restore_hint(self) -> None:
        self._reporting = False
        self.report_level = ""
        self.hint_label.set_text(self._hint, "text.dim")

    def set_progress(self, text: str, fraction: float | None) -> None:
        if fraction is None:
            self._progress.hide()
            return
        try:
            value = float(fraction)
        except (TypeError, ValueError):
            value = 0.0
        if value != value:      # NaN
            value = 0.0
        self.progress_label.setText(text or "")
        self.progress_label.setVisible(bool(text))
        self.progress_bar.setValue(int(round(min(1.0, max(0.0, value)) * 1000)))
        self._progress.show()

    def progress_visible(self) -> bool:
        return not self._progress.isHidden()

    def set_stats(self, text: str) -> None:
        self.stats_label.set_text(text or "", "text.dim")

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.header"))


# ======================================================================
# 主窗口
# ======================================================================

class MainWindow(QMainWindow):
    """程序主窗口。见模块说明。"""

    workspaceChanged = Signal(int)

    def __init__(self, app: Any, wm: Any = None) -> None:
        super().__init__()
        self.app = app
        self.wm = wm if wm is not None else getattr(app, "wm", None)
        self._project: Any = None
        self._save_pending = False
        self.setWindowTitle(APP_NAME)
        from .frameless import ENABLED as FRAMELESS, WinFrame
        self.frame = WinFrame(self, self._hit_test) if FRAMELESS else None
        central = QWidget(self)
        central.setObjectName("central")
        box = QVBoxLayout(central)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.topbar = TopBar(self, window_buttons=self.frame is not None)
        self.stack = QStackedWidget(central)
        self.status = StatusBar(self)
        box.addWidget(self.topbar)
        box.addWidget(self.stack, 1)
        box.addWidget(self.status)
        self.setCentralWidget(central)

        items, active = workspaces.load(getattr(app, "prefs", None))
        self._workspaces: list[workspaces.Workspace] = items
        self._active = -1
        if self.wm is not None:
            self.wm.register_window(self, main=True)
        self.set_workspace(active)
        self._hook_project()
        theme.changed.connect(self._on_theme_changed)
        registry.changed.connect(self._on_registry_changed)

    # ---- 工作区：查询 ----
    def workspace_names(self) -> list[str]:
        return [ws.name for ws in self._workspaces]

    def active_workspace_index(self) -> int:
        return self._active

    def active_workspace(self) -> workspaces.Workspace | None:
        return self._workspaces[self._active] if 0 <= self._active < len(self._workspaces) else None

    def workspace(self, index_or_name: Any) -> workspaces.Workspace | None:
        index = self._index_of(index_or_name)
        return self._workspaces[index] if index is not None else None

    def current_screen(self) -> Screen | None:
        ws = self.active_workspace()
        return ws.screen if ws is not None else None

    def _index_of(self, index_or_name: Any) -> int | None:
        if isinstance(index_or_name, str):
            names = self.workspace_names()
            return names.index(index_or_name) if index_or_name in names else None
        try:
            index = int(index_or_name)
        except (TypeError, ValueError):
            return None
        return index if 0 <= index < len(self._workspaces) else None

    # ---- 工作区：修改 ----
    def set_workspace(self, index_or_name: Any) -> bool:
        """切换到某个工作区（序号或名字）。"""
        index = self._index_of(index_or_name)
        if index is None:
            return False
        ws = self._workspaces[index]
        if ws.screen is None or not alive(ws.screen):
            ws.screen = Screen(self.app, ws.layout, parent=self.stack)
            self.stack.addWidget(ws.screen)
            ws.screen.layoutChanged.connect(self._schedule_save)
        changed = index != self._active
        self._active = index
        self.stack.setCurrentWidget(ws.screen)
        if changed:
            ws.screen.refresh()
        self._sync_tabs()
        if changed:
            self.workspaceChanged.emit(index)
            self._schedule_save()
        return True

    def cycle_workspace(self, direction: int = 1) -> bool:
        count = len(self._workspaces)
        if count < 2:
            return False
        return self.set_workspace((self._active + (1 if direction >= 0 else -1)) % count)

    def add_workspace(self, name: str | None = None, layout: dict | None = None, index: int | None = None,
                      activate: bool = True) -> int:
        """新建工作区，返回它的序号。layout 为空时复制当前工作区的布局。"""
        current = self.active_workspace()
        if layout is None or not workspaces.is_valid_layout(layout):
            layout = current.snapshot() if current is not None else workspaces.fallback_layout()
        base = name or (current.name if current is not None else "工作区")
        name = workspaces.unique_name(base, self.workspace_names())
        ws = workspaces.Workspace(name, layout)
        if index is None:
            index = self._active + 1 if self._active >= 0 else len(self._workspaces)
        index = max(0, min(len(self._workspaces), index))
        self._workspaces.insert(index, ws)
        if self._active >= index:
            self._active += 1
        if activate:
            self.set_workspace(index)
        else:
            self._sync_tabs()
        self._schedule_save()
        return index

    def duplicate_workspace(self, index: int | None = None) -> int:
        source = self._workspaces[index] if index is not None and 0 <= index < len(self._workspaces) \
            else self.active_workspace()
        if source is None:
            return -1
        position = self._workspaces.index(source) + 1
        return self.add_workspace(source.name, source.snapshot(), index=position)

    def remove_workspace(self, index: int | None = None) -> bool:
        """删除工作区。只剩一个时不能删。"""
        index = self._active if index is None else index
        if len(self._workspaces) < 2 or not 0 <= index < len(self._workspaces):
            return False
        ws = self._workspaces.pop(index)
        if ws.screen is not None and alive(ws.screen):
            self.stack.removeWidget(ws.screen)
            for area in ws.screen.areas():
                area.dispose()
            ws.screen.deleteLater()
        ws.screen = None
        if index < self._active:
            self._active -= 1
        elif index == self._active:
            self._active = -1
            self.set_workspace(max(0, index - 1))
        self._sync_tabs()
        self._schedule_save()
        return True

    def rename_workspace(self, index: int, name: str) -> bool:
        if not 0 <= index < len(self._workspaces):
            return False
        name = (name or "").strip()
        if not name:
            return False
        others = [ws.name for i, ws in enumerate(self._workspaces) if i != index]
        self._workspaces[index].name = workspaces.unique_name(name, others)
        self._sync_tabs()
        self._schedule_save()
        return True

    def move_workspace(self, index: int, new_index: int) -> bool:
        count = len(self._workspaces)
        if not 0 <= index < count:
            return False
        new_index = max(0, min(count - 1, new_index))
        if new_index == index:
            return False
        active = self.active_workspace()
        self._workspaces.insert(new_index, self._workspaces.pop(index))
        if active is not None:
            self._active = self._workspaces.index(active)
        self._sync_tabs()
        self._schedule_save()
        return True

    def _sync_tabs(self) -> None:
        self.topbar.tabs.set_items(self.workspace_names(), self._active)

    # ---- 存取 ----
    def _schedule_save(self) -> None:
        if not self._save_pending:
            self._save_pending = True
            QTimer.singleShot(0, self.store_workspaces)

    def store_workspaces(self) -> str:
        """把全部工作区的布局写进 app.prefs.workspaces，返回写入的文字。"""
        self._save_pending = False
        return workspaces.save(getattr(self.app, "prefs", None), self._workspaces, max(0, self._active))

    # ---- 状态栏 ----
    def set_hint(self, text: str) -> None:
        self.status.set_hint(text)

    def set_progress(self, text: str, fraction: float | None) -> None:
        self.status.set_progress(text, fraction)

    def set_stats(self, text: str) -> None:
        self.status.set_stats(text)

    def show_report(self, text: str, level: str = "INFO") -> None:
        self.status.show_report(text, level)

    # ---- 工程名 ----
    def _hook_project(self) -> None:
        project = getattr(self.app, "project", None)
        if project is not self._project:
            old = self._project
            if old is not None and hasattr(old, "changed"):
                try:
                    old.changed.disconnect(self._on_project_changed)
                except Exception:  # noqa: BLE001
                    pass
            self._project = project
            if project is not None and hasattr(project, "changed"):
                project.changed.connect(self._on_project_changed)
        self._update_title()

    def _on_project_changed(self, kind: str = "") -> None:
        if alive(self):
            self._update_title()

    def _update_title(self) -> None:
        title = str(getattr(self._project, "title", "") or "") if self._project is not None else ""
        self.topbar.set_title(title)
        self.setWindowTitle("%s - %s" % (title, APP_NAME) if title else APP_NAME)

    def on_notify(self, tags: frozenset) -> None:
        self._hook_project()

    # ---- 窗口边框 ----
    def toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _hit_test(self, pos: QPoint) -> str:
        """窗口里这一点（本地逻辑坐标）是标题栏（可拖动、双击最大化）还是内容。"""
        bar = self.topbar
        local = bar.mapFrom(self, pos)
        if 0 <= local.y() < bar.height() and 0 <= local.x() < bar.width():
            return "caption" if bar.is_caption(local) else "client"
        return "client"

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        if self.frame is not None:
            self.frame.attach()

    def nativeEvent(self, event_type: Any, message: Any):  # noqa: N802
        if self.frame is not None:
            result = self.frame.native_event(event_type, message)
            if result is not None:
                return result
        return super().nativeEvent(event_type, message)

    def changeEvent(self, event: Any) -> None:  # noqa: N802
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.WindowStateChange:
            self.topbar.set_maximized(self.isMaximized())
        super().changeEvent(event)

    # ---- Qt ----
    def _on_theme_changed(self) -> None:
        if not alive(self):
            return
        self.topbar.apply_theme()
        self.status.apply_theme()
        self._sync_tabs()

    def _on_registry_changed(self, kind: str = "") -> None:
        pass

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self.frame is not None:
            self.frame.active = False            # 窗口拆掉的过程中不再处理系统消息
        self.store_workspaces()
        wm = self.wm
        if wm is not None:
            wm.modal_cancel_all()
            wm.close_popouts()
        super().closeEvent(event)


class PopoutWindow(QWidget):
    """从区域弹出的独立窗口：里面是一个只有单个区域的 Screen，同样接受键位路由。"""

    def __init__(self, app: Any, wm: Any, layout: dict, title: str = "") -> None:
        super().__init__(None, Qt.Window)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.app = app
        self.wm = wm
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self._screen = Screen(app, layout, parent=self)
        box.addWidget(self._screen)
        area = self._screen.areas()[0] if self._screen.areas() else None
        label = getattr(area.editor, "label", "") if area is not None and area.editor is not None else ""
        self.setWindowTitle(title or ("%s - %s" % (label, APP_NAME) if label else APP_NAME))
        if wm is not None:
            wm.register_window(self)

    def current_screen(self) -> Screen:
        return self._screen

    def on_notify(self, tags: frozenset) -> None:
        pass

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.window"))

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        if self.wm is not None:
            self.wm.unregister_window(self)
        super().closeEvent(event)
