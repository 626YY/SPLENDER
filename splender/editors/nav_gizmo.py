"""3D 视口右上角的导航小部件（照 Blender）：坐标轴球（点一个轴转到那个方向的视图，再点一次转到对面；按住拖动旋转视图），
下面一列按钮：缩放（按住拖）、平移（按住拖）、摄像机视图、透视 / 正交。坐标都是视口控件的逻辑像素。"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from ..core import ops
from ..core.keymap import LEFTMOUSE
from ..doc.objects import to_internal
from ..ui import icons, theme

AXES = [("X", (1.0, 0.0, 0.0), QColor(230, 62, 72)), ("Y", (0.0, 1.0, 0.0), QColor(123, 178, 61)),
        ("Z", (0.0, 0.0, 1.0), QColor(69, 122, 230))]
#: 点哪个轴转到哪个视图（Blender：点 X 看右视图，再点一次看左视图）
AXIS_VIEWS = {("X", 1): "RIGHT", ("X", -1): "LEFT", ("Y", 1): "BACK", ("Y", -1): "FRONT", ("Z", 1): "TOP",
              ("Z", -1): "BOTTOM"}
BUTTONS = [("zoom", "zoom-in", "按住拖动缩放视图"), ("pan", "hand", "按住拖动平移视图"),
           ("camera", "object.camera", "切换摄像机视图"), ("ortho", "grid-3x3", "切换透视 / 正交")]


class NavGizmo:
    def __init__(self, editor) -> None:
        self.editor = editor
        self.hover = None          # ("axis", 名字, 正负) / ("ball",) / ("button", 名字)
        self.pressed = None        # 按下时的目标和位置

    # ---------------------------------------------------------------- 排版
    def enabled(self) -> bool:
        overlay = self.editor.overlay
        return bool(overlay.show_overlays and overlay.show_gizmo) and self.editor.widget is not None

    def ball(self) -> tuple[QPointF, float]:
        widget = self.editor.widget
        radius = float(theme.px(40))
        margin = float(theme.px(14))
        return QPointF(widget.width() - margin - radius, margin + radius + float(theme.px(4))), radius

    def button_rects(self) -> list[tuple[str, QRectF]]:
        center, radius = self.ball()
        size = float(theme.px(26))
        gap = float(theme.px(4))
        top = center.y() + radius + float(theme.px(12))
        return [(name, QRectF(center.x() - size / 2.0, top + i * (size + gap), size, size))
                for i, (name, _icon, _tip) in enumerate(BUTTONS)]

    def bubbles(self) -> list[tuple[float, QPointF, str, int, QColor]]:
        """每个轴两端的小圆：(深度, 位置, 轴名, 正负, 颜色)，按深度从远到近排。"""
        center, radius = self.ball()
        camera = self.editor.view.camera
        right, up, back = camera.basis()
        out = []
        for name, axis, color in AXES:
            v = to_internal(np.asarray(axis, np.float64))
            for sign in (1, -1):
                d = v * sign
                x = float(d @ right)
                y = float(d @ up)
                depth = float(d @ back)
                pos = QPointF(center.x() + x * radius * 0.78, center.y() - y * radius * 0.78)
                out.append((depth, pos, name, sign, color))
        out.sort(key=lambda item: item[0])
        return out

    def hit(self, x: float, y: float):
        if not self.enabled():
            return None
        point = QPointF(x, y)
        for name, rect in self.button_rects():
            if rect.contains(point):
                return ("button", name)
        bubble_r = float(theme.px(9))
        for _depth, pos, name, sign, _color in reversed(self.bubbles()):
            if math.hypot(pos.x() - x, pos.y() - y) <= bubble_r:
                return ("axis", name, sign)
        center, radius = self.ball()
        if math.hypot(center.x() - x, center.y() - y) <= radius:
            return ("ball",)
        return None

    # ---------------------------------------------------------------- 画
    def paint(self, painter: QPainter) -> None:
        if not self.enabled():
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        center, radius = self.ball()
        if self.hover is not None and self.hover[0] in ("ball", "axis"):
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 28))
            painter.drawEllipse(center, radius, radius)
        bubble_r = float(theme.px(9))
        font = QFont(theme.font("font.small"))
        font.setBold(True)
        painter.setFont(font)
        for depth, pos, name, sign, color in self.bubbles():
            if sign > 0:
                painter.setPen(QPen(color, max(1.5, float(theme.px(2)))))
                painter.drawLine(center, pos)
            fill = QColor(color)
            if sign < 0:
                fill.setAlpha(110)
            hovered = self.hover == ("axis", name, sign)
            painter.setPen(QPen(QColor(255, 255, 255), 1.5) if hovered else Qt.NoPen)
            painter.setBrush(fill)
            r = bubble_r if sign > 0 else bubble_r * 0.8
            painter.drawEllipse(pos, r, r)
            if sign > 0 or hovered:
                painter.setPen(QColor(20, 20, 20) if sign > 0 else QColor(240, 240, 240))
                painter.drawText(QRectF(pos.x() - r, pos.y() - r, 2 * r, 2 * r), Qt.AlignCenter,
                                 name if sign > 0 else "-" + name)
        dpr = self.editor.widget.devicePixelRatioF()
        camera = self.editor.view.camera
        for name, rect in self.button_rects():
            active = (name == "camera" and self.editor.in_camera_view()) or (name == "ortho" and camera.ortho)
            hovered = self.hover == ("button", name)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(75, 139, 245, 200) if active else QColor(40, 40, 44, 200 if hovered else 150))
            painter.drawEllipse(rect)
            icon = next(i for n, i, _t in BUTTONS if n == name)
            if name == "ortho" and not camera.ortho:
                icon = "perspective"
            size = int(theme.px(16))
            pm = icons.pixmap(icon, "#ffffff" if (active or hovered) else theme.color("text.dim"), size, dpr)
            painter.drawPixmap(int(rect.center().x() - size / 2), int(rect.center().y() - size / 2), pm)

    # ---------------------------------------------------------------- 鼠标
    def on_move(self, x: float, y: float) -> bool:
        target = self.hit(x, y)
        if target != self.hover:
            self.hover = target
            self.editor.widget.overlay.update()
        if self.pressed is not None:
            kind, start = self.pressed
            if kind[0] in ("ball", "axis") and math.hypot(x - start[0], y - start[1]) > 3:
                # 拖动：旋转视图（接着交给旋转操作，松开左键结束）
                self.pressed = None
                event = self._event(start)
                ops.call("view3d.orbit", self.editor.context(event), event)
                return True
            return True
        return target is not None

    def on_press(self, event) -> bool:
        target = self.hit(event.x, event.y)
        if target is None or event.type != LEFTMOUSE:
            return False
        if target[0] == "button":
            name = target[1]
            ctx = self.editor.context(event)
            if name == "zoom":
                ops.call("view3d.zoom_drag", ctx, event)
            elif name == "pan":
                ops.call("view3d.pan", ctx, event)
            elif name == "camera":
                ops.call("view3d.view_camera", ctx, None)
            elif name == "ortho":
                ops.call("view3d.view_persportho", ctx, None)
            return True
        self.pressed = (target, (event.x, event.y))
        return True

    def on_release(self, event) -> bool:
        if self.pressed is None:
            return False
        target, _start = self.pressed
        self.pressed = None
        if target[0] == "axis":
            _kind, name, sign = target
            camera = self.editor.view.camera
            view = AXIS_VIEWS[(name, sign)]
            if camera.axis_view == view:                 # 再点一次同一个轴：看对面
                view = AXIS_VIEWS[(name, -sign)]
            ops.call("view3d.view_axis", self.editor.context(event), None, invoke=False, axis=view)
        return True

    def _event(self, pos):
        from ..core.keymap import PRESS, Event

        return Event(type=LEFTMOUSE, value=PRESS, x=pos[0], y=pos[1], source=self.editor.widget)
