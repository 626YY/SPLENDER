"""移动、旋转、缩放工具的操纵杆（照 Blender）：画在选中部分的轴心上，按住拖动就是带约束的 G / R / S。

- 移动：三根箭头（沿轴）、三个小方块（在平面里）、中间的圈（在屏幕平面里自由移动）；
- 旋转：三个环（绕轴）、外面一个白环（绕视线）；
- 缩放：三根带方块的线（沿轴）、三个小方块（在平面里）、中间的圈（整体缩放）。
坐标轴按标题栏里的「坐标系」（全局、局部、法向、视图）。物体模式和编辑模式都能用。
屏幕坐标用视口控件的逻辑像素（和导航小部件一样）。
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen

from ..core import ops
from ..core.keymap import LEFTMOUSE
from ..doc.objects import to_internal
from ..ui import theme

AXIS_COLORS = [QColor(230, 62, 72), QColor(123, 178, 61), QColor(69, 122, 230)]
VIEW_COLOR = QColor(235, 235, 235)
HIGHLIGHT = QColor(255, 220, 90)
PLANES = {0: "YZ", 1: "XZ", 2: "XY"}          # 法线是哪个轴 → 平面约束
KINDS = {"move": "TRANSLATE", "rotate": "ROTATE", "scale": "RESIZE"}
OPS = {"TRANSLATE": "transform.translate", "ROTATE": "transform.rotate", "RESIZE": "transform.resize"}


class TransformGizmo:
    def __init__(self, editor) -> None:
        self.editor = editor
        self.hover = None             # ("axis", k) / ("plane", k) / ("center",) / ("view",)
        self._cache = None

    # ---------------------------------------------------------------- 状态
    def kind(self) -> str | None:
        tool = self.editor.app.tool_settings.tool
        return KINDS.get(tool.split(".")[-1]) if tool.startswith(("object.", "mesh.")) else None

    def enabled(self) -> bool:
        editor = self.editor
        overlay = editor.overlay
        if editor.widget is None or editor.view is None or not (overlay.show_overlays and overlay.show_gizmo):
            return False
        return self.kind() is not None and editor.app.tool_settings.mode in ("OBJECT", "EDIT")

    def frame(self):
        """(轴心, 三个轴（显示坐标，按列排）)。没有选中的东西时返回 None。编辑模式按网格版本缓存。"""
        from ..ops.object_ops import edit_active, object_mode, orientation_basis, pivot_points

        editor = self.editor
        app = editor.app
        ctx = editor.context()
        settings = app.tool_settings
        project = app.project
        if project is None:
            return None
        if edit_active(ctx):
            from types import SimpleNamespace

            from ..ops.edit_transform import edit_pivot, to_disp

            session = app.engine.edit
            mesh = session.mesh
            key = (id(mesh), mesh.geometry_version, mesh.topology_version, int(mesh.vert_sel.sum()), mesh.active,
                   settings.transform_pivot_point, settings.transform_orientation,
                   tuple(np.round(project.cursor_location, 6)))
            if self._cache is not None and self._cache[0] == key:
                return self._cache[1]
            sel = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
            if not len(sel):
                self._cache = (key, None)
                return None
            disp = to_disp(mesh.verts)
            pivot, _groups = edit_pivot(SimpleNamespace(), ctx, mesh, disp, sel)
            basis = orientation_basis(ctx, settings.transform_orientation, session.obj, editor)
            result = (np.asarray(pivot, np.float64), basis)
            self._cache = (key, result)
            return result
        if not object_mode(ctx):
            return None
        objects = project.selected_objects()
        if not objects:
            return None
        pivot, _individual = pivot_points(ctx, objects, settings.transform_pivot_point)
        active = project.active_object if project.active_object in objects else objects[0]
        basis = orientation_basis(ctx, settings.transform_orientation, active, editor)
        return np.asarray(pivot, np.float64), basis

    # ---------------------------------------------------------------- 投影
    def _scale(self) -> float:
        widget = self.editor.widget
        return self.editor.view.width / max(1.0, float(widget.width()))

    def _screen(self, point_display) -> QPointF | None:
        view = self.editor.view
        p = to_internal(np.asarray(point_display, np.float64))
        clip = view.camera.view_projection(view.aspect) @ np.array([p[0], p[1], p[2], 1.0])
        if clip[3] <= 1e-9:
            return None
        ratio = self._scale()
        x = (clip[0] / clip[3] * 0.5 + 0.5) * view.width / ratio
        y = (0.5 - clip[1] / clip[3] * 0.5) * view.height / ratio
        return QPointF(float(x), float(y))

    def _world_per_px(self, pivot) -> float:
        view = self.editor.view
        camera = view.camera
        per_physical = camera.world_per_pixel(camera.depth_of(to_internal(pivot)), view.height, view.aspect)
        return per_physical * self._scale()

    def geometry(self) -> dict | None:
        """当前要画的各部分（逻辑像素）。"""
        if not self.enabled():
            return None
        frame = self.frame()
        if frame is None:
            return None
        pivot, basis = frame
        center = self._screen(pivot)
        if center is None:
            return None
        kind = self.kind()
        unit = self._world_per_px(pivot)
        length = theme.px(78) * unit
        out = {"kind": kind, "center": center, "axes": [], "planes": [], "rings": []}
        min_len = float(theme.px(14))
        for k in range(3):
            axis = basis[:, k]
            tip = self._screen(pivot + axis * length)
            if tip is None:
                continue
            start = self._screen(pivot + axis * length * 0.22)
            # 几乎正对视线的轴缩成一点，藏起来（和 Blender 一样），免得挡住中间的圈
            if start is not None and math.hypot(tip.x() - center.x(), tip.y() - center.y()) >= min_len:
                out["axes"].append((k, start, tip))
            if kind in ("TRANSLATE", "RESIZE"):
                a, b = [j for j in range(3) if j != k]
                off = length * 0.32
                size = length * 0.09
                corners = [pivot + basis[:, a] * (off + sa * size) + basis[:, b] * (off + sb * size)
                           for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                pts = [self._screen(c) for c in corners]
                if all(p is not None for p in pts):
                    area = 0.5 * abs(sum(pts[i].x() * pts[(i + 1) % 4].y() - pts[(i + 1) % 4].x() * pts[i].y()
                                         for i in range(4)))
                    if area >= float(theme.px(5)) ** 2:          # 侧着看的平面块藏起来
                        out["planes"].append((k, pts))
            if kind == "ROTATE":
                a, b = [j for j in range(3) if j != k]
                ring = []
                for i in range(65):
                    t = 2.0 * math.pi * i / 64.0
                    ring.append(self._screen(pivot + (basis[:, a] * math.cos(t) + basis[:, b] * math.sin(t))
                                             * length * 0.82))
                if all(p is not None for p in ring):
                    out["rings"].append((k, ring))
        return out

    # ---------------------------------------------------------------- 命中
    def hit(self, x: float, y: float):
        geo = self.geometry()
        if geo is None:
            return None
        point = QPointF(x, y)
        tolerance = float(theme.px(7))
        center = geo["center"]
        kind = geo["kind"]
        best, best_d = None, tolerance
        for k, start, tip in geo["axes"] if kind != "ROTATE" else []:
            d = _segment_distance(point, start, tip)
            if d < best_d:
                best, best_d = ("axis", k), d
        for k, pts in geo["planes"]:
            path = QPainterPath(pts[0])
            for p in pts[1:]:
                path.lineTo(p)
            path.closeSubpath()
            if path.contains(point):
                return ("plane", k)
        for k, ring in geo["rings"]:
            for p, q in zip(ring, ring[1:]):
                d = _segment_distance(point, p, q)
                if d < best_d:
                    best, best_d = ("axis", k), d
        if best is not None:
            return best
        r = math.hypot(x - center.x(), y - center.y())
        if kind == "ROTATE":
            view_r = float(theme.px(92))
            if abs(r - view_r) <= tolerance:
                return ("view",)
            return None
        if r <= float(theme.px(11)):
            return ("center",)
        return None

    def on_move(self, x: float, y: float) -> bool:
        hover = self.hit(x, y)
        if hover != self.hover:
            self.hover = hover
            if self.editor.widget is not None:
                self.editor.widget.overlay.update()
        return False

    def on_press(self, event) -> bool:
        """左键按在操纵杆上：开始带约束的移动、旋转、缩放（松开确认）。"""
        if event.type != LEFTMOUSE or event.ctrl or event.alt or event.shift:
            return False
        part = self.hit(event.x, event.y)
        if part is None:
            return False
        kind = self.kind()
        props = {"release_confirm": True}
        orient = self.editor.app.tool_settings.transform_orientation
        if part[0] == "axis":
            props.update(constraint="XYZ"[part[1]], orient_type=orient)
        elif part[0] == "plane":
            props.update(constraint=PLANES[part[1]], orient_type=orient)
        ctx = self.editor.context(event)
        ops.call(OPS[kind], ctx, event, **props)
        return True

    # ---------------------------------------------------------------- 画
    def paint(self, painter: QPainter) -> None:
        geo = self.geometry()
        if geo is None:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        kind = geo["kind"]
        center = geo["center"]
        hover = self.hover
        width = float(theme.px(2))
        if kind != "ROTATE":
            for k, start, tip in geo["axes"]:
                color = HIGHLIGHT if hover == ("axis", k) else AXIS_COLORS[k]
                painter.setPen(QPen(color, width))
                painter.drawLine(start, tip)
                painter.setBrush(color)
                painter.setPen(Qt.NoPen)
                if kind == "TRANSLATE":
                    _arrow_head(painter, start, tip, float(theme.px(9)))
                else:
                    s = float(theme.px(5))
                    painter.drawRect(QRectF(tip.x() - s, tip.y() - s, 2 * s, 2 * s))
            for k, pts in geo["planes"]:
                color = HIGHLIGHT if hover == ("plane", k) else AXIS_COLORS[k]
                path = QPainterPath(pts[0])
                for p in pts[1:]:
                    path.lineTo(p)
                path.closeSubpath()
                fill = QColor(color)
                fill.setAlpha(170)
                painter.setPen(Qt.NoPen)
                painter.setBrush(fill)
                painter.drawPath(path)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(HIGHLIGHT if hover == ("center",) else VIEW_COLOR, width))
            r = float(theme.px(11))
            painter.drawEllipse(center, r, r)
        else:
            for k, ring in geo["rings"]:
                color = HIGHLIGHT if hover == ("axis", k) else AXIS_COLORS[k]
                painter.setPen(QPen(color, width))
                path = QPainterPath(ring[0])
                for p in ring[1:]:
                    path.lineTo(p)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(path)
            painter.setPen(QPen(HIGHLIGHT if hover == ("view",) else VIEW_COLOR, width))
            r = float(theme.px(92))
            painter.drawEllipse(center, r, r)
        painter.restore()


def _segment_distance(p: QPointF, a: QPointF, b: QPointF) -> float:
    ax, ay, bx, by = a.x(), a.y(), b.x(), b.y()
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    t = 0.0 if length2 < 1e-9 else max(0.0, min(1.0, ((p.x() - ax) * vx + (p.y() - ay) * vy) / length2))
    return math.hypot(p.x() - (ax + vx * t), p.y() - (ay + vy * t))


def _arrow_head(painter: QPainter, start: QPointF, tip: QPointF, size: float) -> None:
    dx, dy = tip.x() - start.x(), tip.y() - start.y()
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    base = QPointF(tip.x() - ux * size, tip.y() - uy * size)
    left = QPointF(base.x() - uy * size * 0.45, base.y() + ux * size * 0.45)
    right = QPointF(base.x() + uy * size * 0.45, base.y() - ux * size * 0.45)
    path = QPainterPath(tip)
    path.lineTo(left)
    path.lineTo(right)
    path.closeSubpath()
    painter.drawPath(path)
