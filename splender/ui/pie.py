"""饼菜单（和 Blender 一样）：在鼠标处弹出，条目排在八个方向上。

- 鼠标往哪个方向移，哪个条目就亮；点一下选中；
- 按住快捷键拖向某个方向再松开快捷键，直接选中（按一下就松开的话菜单留着，再点）；
- Esc、右键、点中间取消。
条目顺序和 Blender 相同：左、右、下、上、左上、右上、左下、右下。
"""
from __future__ import annotations

import math
from typing import Any, Callable

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QWidget

from . import icons, theme

#: 八个位置的方向（度，0 = 右，逆时针），按 Blender 的条目顺序
SLOT_ANGLES = [180.0, 0.0, 270.0, 90.0, 135.0, 45.0, 225.0, 315.0]


class PieItem:
    def __init__(self, text: str, callback: Callable[[], Any] | None, icon: str = "", enabled: bool = True,
                 shortcut: str = "") -> None:
        self.text = text
        self.callback = callback
        self.icon = icon
        self.enabled = enabled
        self.shortcut = shortcut


class PieMenu(QWidget):
    RADIUS = 110          # 条目离中心的距离（逻辑像素，按界面缩放）
    DEAD = 18             # 离中心这么近时不选任何条目

    def __init__(self, items: list[PieItem], title: str = "", hold_key: int | None = None) -> None:
        super().__init__(None, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMouseTracking(True)
        self.items = items[:8]
        self.title = title
        self.hold_key = hold_key
        self.highlight = -1
        self.moved = False
        self.result: PieItem | None = None
        side = theme.px(2 * self.RADIUS + 260)
        self.resize(side, side)
        self._center = QPointF(side / 2.0, side / 2.0)
        self._rects = self._layout()

    # ---- 排版 ----
    def _layout(self) -> list[QRectF]:
        fm = QFontMetrics(theme.font())
        rects = []
        radius = theme.px(self.RADIUS)
        height = theme.px(26)
        for index, item in enumerate(self.items):
            angle = math.radians(SLOT_ANGLES[index])
            width = fm.horizontalAdvance(item.text) + theme.px(34 if item.icon else 20)
            if item.shortcut:
                width += fm.horizontalAdvance(item.shortcut) + theme.px(12)
            cx = self._center.x() + math.cos(angle) * radius
            cy = self._center.y() - math.sin(angle) * radius
            # 左右两侧的条目向外对齐（像 Blender 一样不压到中心）
            if abs(math.cos(angle)) > 0.3:
                x = cx if math.cos(angle) > 0 else cx - width
                if abs(math.sin(angle)) < 0.3:
                    x = cx - width * 0.15 if math.cos(angle) > 0 else cx - width * 0.85
            else:
                x = cx - width / 2.0
            rects.append(QRectF(x, cy - height / 2.0, width, height))
        return rects

    def popup(self, global_pos: QPoint | None = None) -> None:
        pos = global_pos if global_pos is not None else QCursor.pos()
        self.move(pos - QPoint(int(self._center.x()), int(self._center.y())))
        self.show()
        self.raise_()
        self.activateWindow()
        self.grabKeyboard()

    # ---- 选择 ----
    def _slot_at(self, pos: QPointF) -> int:
        for index, rect in enumerate(self._rects):
            if rect.adjusted(-4, -4, 4, 4).contains(pos):
                return index
        d = pos - self._center
        if math.hypot(d.x(), d.y()) < theme.px(self.DEAD):
            return -1
        angle = math.degrees(math.atan2(-d.y(), d.x())) % 360.0
        best, best_diff = -1, 360.0
        for index in range(len(self.items)):
            diff = abs((angle - SLOT_ANGLES[index] + 180.0) % 360.0 - 180.0)
            if diff < best_diff:
                best, best_diff = index, diff
        return best if best_diff <= 45.0 else -1

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        pos = event.position()
        if math.hypot(pos.x() - self._center.x(), pos.y() - self._center.y()) > theme.px(self.DEAD):
            self.moved = True
        slot = self._slot_at(pos)
        if slot != self.highlight:
            self.highlight = slot
            self.update()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.RightButton:
            self._finish(None)
            return
        if event.button() == Qt.LeftButton:
            slot = self._slot_at(event.position())
            self._finish(self.items[slot] if slot >= 0 else None)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self._finish(None)
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.highlight >= 0:
            self._finish(self.items[self.highlight])
            return
        # 数字键 1–8 直接选
        if Qt.Key_1 <= event.key() <= Qt.Key_8:
            index = event.key() - Qt.Key_1
            if index < len(self.items):
                self._finish(self.items[index])
                return
        event.accept()

    def keyReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return
        if self.hold_key is not None and event.key() == self.hold_key and self.moved:
            self._finish(self.items[self.highlight] if self.highlight >= 0 else None)

    def _finish(self, item: PieItem | None) -> None:
        if item is not None and not item.enabled:
            return
        self.result = item
        try:
            self.releaseKeyboard()
        except Exception:  # noqa: BLE001
            pass
        self.close()
        if item is not None and item.callback is not None:
            QTimer.singleShot(0, item.callback)

    # ---- 画 ----
    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        c = self._center
        ring = theme.px(16)
        painter.setPen(QPen(theme.qcolor("line.soft"), theme.px(3)))
        painter.setBrush(theme.qcolor("bg.menu", 0.85))
        painter.drawEllipse(c, ring, ring)
        if self.highlight >= 0:
            angle = SLOT_ANGLES[self.highlight]
            pen = QPen(theme.qcolor("accent"), theme.px(3))
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            rect = QRectF(c.x() - ring, c.y() - ring, 2 * ring, 2 * ring)
            painter.drawArc(rect, int((angle - 30) * 16), int(60 * 16))
        if self.title:
            painter.setFont(theme.font("font.small", bold=True))
            painter.setPen(theme.qcolor("text.dim"))
            painter.drawText(QRectF(c.x() - 120, c.y() - ring - theme.px(28), 240, theme.px(18)), Qt.AlignCenter,
                             self.title)
        painter.setFont(theme.font())
        fm = QFontMetrics(theme.font())
        for index, (item, rect) in enumerate(zip(self.items, self._rects)):
            path = QPainterPath()
            path.addRoundedRect(rect, theme.px(5), theme.px(5))
            active = index == self.highlight and item.enabled
            painter.fillPath(path, theme.qcolor("accent") if active else theme.qcolor("bg.menu"))
            painter.setPen(QPen(theme.qcolor("line"), 1))
            painter.drawPath(path)
            x = rect.left() + theme.px(10)
            if item.icon:
                try:
                    pixmap = icons.icon(item.icon).pixmap(theme.px(16), theme.px(16))
                    painter.drawPixmap(int(x), int(rect.center().y() - theme.px(8)), pixmap)
                except Exception:  # noqa: BLE001
                    pass
                x += theme.px(22)
            color = "text.on_accent" if active else ("text" if item.enabled else "text.disabled")
            painter.setPen(theme.qcolor(color))
            painter.drawText(QRectF(x, rect.top(), rect.right() - x, rect.height()), Qt.AlignVCenter | Qt.AlignLeft,
                             item.text)
            if item.shortcut:
                painter.setPen(theme.qcolor("text.dim" if not active else "text.on_accent"))
                painter.drawText(QRectF(x, rect.top(), rect.right() - x - theme.px(8), rect.height()),
                                 Qt.AlignVCenter | Qt.AlignRight, item.shortcut)
        del fm


def pie_from_menu(idname: str, ctx: Any) -> tuple[list[PieItem], str]:
    """把注册的菜单（registry.Menu）变成饼菜单条目：菜单里的操作项按顺序排到八个方向上。"""
    from .widgets import build_menu

    qmenu = build_menu(idname, ctx)
    layout = getattr(qmenu, "_splender_layout", None)
    if layout is not None:
        try:
            layout.refresh_polls()
        except Exception:  # noqa: BLE001
            pass
    items = []
    for action in qmenu.actions():
        if action.isSeparator() or action.menu() is not None or not action.text():
            continue
        text = action.text().split("\t")[0]
        shortcut = action.text().split("\t")[1] if "\t" in action.text() else action.shortcut().toString()
        items.append(PieItem(text, action.trigger, "", action.isEnabled(), shortcut))
    items_keep = qmenu  # 动作属于这个菜单，菜单要活到用户选完
    for item in items:
        item._menu = items_keep
    return items, qmenu.title()


def show_pie(idname: str, ctx: Any, hold_key: int | None = None, global_pos: QPoint | None = None) -> PieMenu | None:
    items, title = pie_from_menu(idname, ctx)
    if not items:
        return None
    pie = PieMenu(items, title, hold_key)
    pie.popup(global_pos)
    return pie


def qt_key_of(name: str) -> int | None:
    """键名（Z、ACCENT_GRAVE、TAB…）→ Qt 键值，按住松开选择用。"""
    table = {"ACCENT_GRAVE": Qt.Key_QuoteLeft, "TAB": Qt.Key_Tab, "SPACE": Qt.Key_Space, "PERIOD": Qt.Key_Period,
             "COMMA": Qt.Key_Comma}
    if name in table:
        return int(table[name])
    if len(name) == 1 and name.isalpha():
        return int(getattr(Qt, "Key_" + name.upper()))
    return None


def app_has_popup() -> bool:
    return QApplication.activePopupWidget() is not None
