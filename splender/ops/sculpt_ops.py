"""雕刻相关的操作：落笔、调笔刷、细分、清除遮罩、切换模式。"""
from __future__ import annotations

import numpy as np

from ..core import ops
from ..core.keymap import LEFTMOUSE, MOUSEMOVE, PRESS, RELEASE, RIGHTMOUSE
from ..core.ops import CANCELLED, FINISHED, RUNNING_MODAL, Operator
from ..core.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from .view3d_ops import view3d


def _session(ctx):
    engine = getattr(ctx, "engine", None)
    return engine.sculpt if engine is not None else None


def _sculpting(ctx) -> bool:
    return ctx.app.tool_settings.mode == "SCULPT" and _session(ctx) is not None


@ops.register
class SculptStrokeOp(Operator):
    idname = "sculpt.stroke"
    label = "雕刻"
    searchable = False
    invert = BoolProperty("反向", default=False)
    smooth = BoolProperty("临时平滑", default=False)

    @classmethod
    def poll(cls, ctx) -> bool:
        return _sculpting(ctx) and getattr(ctx.editor, "idname", "") == "VIEW_3D"

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        app = ctx.app
        tools = app.tool_settings
        brush = "SMOOTH" if self.smooth else tools.tool.split(".", 1)[1].upper()
        self.editor = editor
        self.button = event.type
        x, y = editor.pixel(event)
        pressure = event.pressure if event.is_tablet else 1.0
        host = app.host
        host.make_current()
        session = host.engine.sculpt
        sym = tools.symmetry
        if not session.stroke_begin(editor.view, x, y, pressure, brush, tools.sculpt, (sym.x, sym.y, sym.z),
                                    invert=bool(self.invert)):
            return CANCELLED
        host.note_input(event.time)
        host.request_frame()
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        host = ctx.app.host
        session = host.engine.sculpt
        if session is None:
            return CANCELLED
        if event.type == MOUSEMOVE:
            if event.source is self.editor.widget:
                host.make_current()
                x, y = self.editor.pixel(event)
                session.stroke_move(x, y, event.pressure if event.is_tablet else 1.0)
                host.note_input(event.time)
                host.request_frame()
            return RUNNING_MODAL
        if event.type == self.button and event.value == RELEASE:
            host.make_current()
            session.stroke_end()
            host.request_frame()
            return FINISHED
        if (event.type == "ESC" or event.type == RIGHTMOUSE) and event.value == PRESS:
            self.cancel(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def cancel(self, ctx) -> None:
        host = ctx.app.host
        host.make_current()
        if host.engine.sculpt is not None:
            host.engine.sculpt.stroke_cancel()
        host.request_frame()


@ops.register
class SculptBrushResize(Operator):
    idname = "sculpt.brush_resize"
    label = "调整雕刻笔刷"
    description = "左右移动鼠标调整，左键或回车确认，右键或 Esc 取消"
    searchable = False
    target = EnumProperty("对象", items=[("size", "直径", ""), ("strength", "力度", ""), ("hardness", "硬度", "")],
                          default="size")

    @classmethod
    def poll(cls, ctx) -> bool:
        return _sculpting(ctx)

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.brush = ctx.app.tool_settings.sculpt
        self.initial = float(getattr(self.brush, self.target))
        self.start_x = event.x if event is not None else 0.0
        self.anchor = getattr(self.editor, "_mouse", None)
        self._hint(ctx)
        return RUNNING_MODAL

    def _hint(self, ctx) -> None:
        prop = type(self.brush).prop(self.target)
        value = getattr(self.brush, self.target)
        text = "%s：%s    左键确认 · 右键取消" % (prop.label, ("%.0f px" % value) if self.target == "size" else ("%.2f" % value))
        if ctx.wm is not None:
            ctx.wm.set_hint(text)

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            if event.source is self.editor.widget:
                dx = event.x - self.start_x
                value = self.initial + dx if self.target == "size" else self.initial + dx / 240.0
                setattr(self.brush, self.target, value)
                if self.anchor is not None:
                    self.editor._mouse = self.anchor
                    self.editor.refresh_cursor()
                self._hint(ctx)
            return RUNNING_MODAL
        if event.value == PRESS and event.type in (LEFTMOUSE, "RET", "NUMPAD_ENTER", "SPACE"):
            self._finish(ctx)
            return FINISHED
        if event.value == PRESS and event.type in (RIGHTMOUSE, "ESC"):
            setattr(self.brush, self.target, self.initial)
            self._finish(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def _finish(self, ctx) -> None:
        if ctx.wm is not None:
            ctx.wm.set_hint("")
        self.editor.refresh_cursor()

    def cancel(self, ctx) -> None:
        setattr(self.brush, self.target, self.initial)


@ops.register
class SculptSubdivide(Operator):
    idname = "sculpt.subdivide"
    label = "细分"
    description = "把每个三角形分成四个并让表面更圆滑，雕刻能刻出更细的细节"
    icon = "grid"
    levels = IntProperty("级数", default=1, min=1, max=3)
    smooth = BoolProperty("圆滑", default=True, description="关掉时只加密网格，形状不变")

    @classmethod
    def poll(cls, ctx) -> bool:
        return _sculpting(ctx)

    def execute(self, ctx) -> str:
        host = ctx.app.host
        host.make_current()
        session = host.engine.sculpt
        triangles = session.base.triangle_count * (4 ** int(self.levels))
        limit = int(ctx.app.prefs.sculpt.max_triangles) if hasattr(ctx.app.prefs, "sculpt") else 40_000_000
        if triangles > limit:
            self.report(ctx, "细分后会有 %s 个三角形，超过上限 %s（可在偏好设置里调）" % (format(triangles, ","),
                                                                        format(limit, ",")), "WARNING")
            return CANCELLED
        session.subdivide(int(self.levels), bool(self.smooth))
        self.report(ctx, "已细分：%s 个三角形" % format(session.base.triangle_count, ","))
        host.request_frame()
        return FINISHED


@ops.register
class SculptVoxelRemesh(Operator):
    idname = "sculpt.voxel_remesh"
    label = "体素重构"
    description = "按体素重新生成均匀、封闭的网格，重叠的部分合成一块。新网格没有 UV，回到绘制前需要展开 UV"
    icon = "remesh"

    @classmethod
    def poll(cls, ctx) -> bool:
        return _sculpting(ctx)

    def execute(self, ctx) -> str:
        from ..ui.busy import busy_cursor

        app = ctx.app
        host = app.host
        host.make_current()
        session = host.engine.sculpt
        settings = app.tool_settings.remesh
        estimate = session.remesh_estimate(int(settings.resolution))
        limit = int(app.prefs.sculpt.max_triangles) if hasattr(app.prefs, "sculpt") else 40_000_000
        if estimate["triangles"] > limit:
            self.report(ctx, "重构后约有 %s 个三角形，超过上限 %s（可以降低分辨率，或在偏好设置里调上限）"
                        % (format(estimate["triangles"], ","), format(limit, ",")), "WARNING")
            return CANCELLED
        try:
            with busy_cursor():
                stats = session.voxel_remesh(settings.resolution, settings.relax, settings.close_holes,
                                             settings.keep_mask, settings.min_island)
        except ValueError as exc:
            self.report(ctx, str(exc), "WARNING")
            return CANCELLED
        self.report(ctx, "已重构：%s 个三角形，用时 %.1f 秒" % (format(stats["triangles"], ","),
                                                         stats["remesh_ms"] / 1000.0))
        host.request_frame()
        return FINISHED


class _MaskOp(Operator):
    """遮罩整体编辑：读出全部顶点的遮罩，按规则改完写回，一步撤销。"""

    @classmethod
    def poll(cls, ctx) -> bool:
        return _sculpting(ctx)

    def transform(self, session, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def execute(self, ctx) -> str:
        host = ctx.app.host
        host.make_current()
        session = host.engine.sculpt
        changed = session.edit_mask(lambda m: self.transform(session, m), self.label)
        host.request_frame()
        return FINISHED if changed else CANCELLED


def _spread(session, mask: np.ndarray, op) -> np.ndarray:
    """每个顶点和相邻顶点取最大（扩大）或最小（缩小）。"""
    edges = session.neighbor_edges()
    out = mask.copy()
    op.at(out, edges[:, 0], mask[edges[:, 1]])
    return out


def _smooth(session, mask: np.ndarray, iterations: int) -> np.ndarray:
    edges = session.neighbor_edges()
    count = np.bincount(edges[:, 0], minlength=len(mask)).astype(np.float32)
    out = mask.astype(np.float32)
    for _ in range(max(1, iterations)):
        total = np.bincount(edges[:, 0], weights=out[edges[:, 1]], minlength=len(mask)).astype(np.float32)
        out = np.where(count > 0, (out + total) / (count + 1.0), out)
    return out


@ops.register
class SculptMaskInvert(_MaskOp):
    idname = "sculpt.mask_invert"
    label = "反转遮罩"
    description = "遮住的放开、没遮住的遮住"
    icon = "square-dashed"

    def transform(self, session, mask):
        return 1.0 - mask


@ops.register
class SculptMaskFill(_MaskOp):
    idname = "sculpt.mask_fill"
    label = "填满遮罩"
    description = "整个模型都遮住"
    value = FloatProperty("遮罩值", default=1.0, min=0.0, max=1.0, subtype="FACTOR")

    def transform(self, session, mask):
        return np.full_like(mask, float(self.value))


@ops.register
class SculptMaskSmooth(_MaskOp):
    idname = "sculpt.mask_smooth"
    label = "柔化遮罩"
    description = "遮罩的边缘变柔和"
    iterations = IntProperty("次数", default=2, min=1, max=50)

    def transform(self, session, mask):
        return _smooth(session, mask, int(self.iterations))


@ops.register
class SculptMaskSharpen(_MaskOp):
    idname = "sculpt.mask_sharpen"
    label = "锐化遮罩"
    description = "遮罩的边缘变硬"
    strength = FloatProperty("力度", default=0.5, min=0.0, max=1.0, subtype="FACTOR")

    def transform(self, session, mask):
        smooth = _smooth(session, mask, 1)
        return mask + (mask - smooth) * (1.0 + 4.0 * float(self.strength))


@ops.register
class SculptMaskGrow(_MaskOp):
    idname = "sculpt.mask_grow"
    label = "扩大遮罩"
    description = "遮罩向外扩一圈"
    iterations = IntProperty("圈数", default=1, min=1, max=100)

    def transform(self, session, mask):
        for _ in range(int(self.iterations)):
            mask = _spread(session, mask, np.maximum)
        return mask


@ops.register
class SculptMaskShrink(_MaskOp):
    idname = "sculpt.mask_shrink"
    label = "缩小遮罩"
    description = "遮罩向里收一圈"
    iterations = IntProperty("圈数", default=1, min=1, max=100)

    def transform(self, session, mask):
        for _ in range(int(self.iterations)):
            mask = _spread(session, mask, np.minimum)
        return mask


@ops.register
class SculptMaskContrast(_MaskOp):
    idname = "sculpt.mask_contrast"
    label = "增加遮罩对比度"
    description = "半遮的地方往全遮或不遮靠"
    factor = FloatProperty("倍数", default=1.5, min=0.1, max=10.0)

    def transform(self, session, mask):
        return (mask - 0.5) * float(self.factor) + 0.5


@ops.register
class SculptMaskClear(Operator):
    idname = "sculpt.mask_clear"
    label = "清除遮罩"
    description = "去掉全部遮罩"
    icon = "square-dashed"

    @classmethod
    def poll(cls, ctx) -> bool:
        return _sculpting(ctx)

    def execute(self, ctx) -> str:
        from ..sculpt.engine import BRUSH_INDEX, DAB_DTYPE

        host = ctx.app.host
        host.make_current()
        session = host.engine.sculpt
        center, radius = session.sculptor.bounds()
        dab = np.zeros(1, DAB_DTYPE)
        dab["center_radius"][0] = (*center, radius * 4.0)
        dab["normal_strength"][0] = (0.0, 1.0, 0.0, 1.0)
        dab["params"][0] = (0.95, -1.0, 0.0, 0.0)
        session.apply_once(dab, BRUSH_INDEX["MASK"])
        host.request_frame()
        return FINISHED


# ====================================================================== 框选、套索遮罩（照 Blender：B、Ctrl+右键拖）
def _mask_in_shape(ctx, editor, test, value: float, label: str) -> bool:
    """屏幕上落在形状里的顶点，遮罩设成 value（一步撤销）。"""
    from ..mesh.select import project

    host = ctx.app.host
    host.make_current()
    session = host.engine.sculpt
    sculptor = session.sculptor
    v = sculptor.vertex_count
    pos = np.frombuffer(sculptor.buffers["pos"].read(size=v * 16), np.float32).reshape(v, 4)[:, :3]
    sx, sy, front, _dist = project(editor.view, pos.astype(np.float64))
    inside = front & test(sx, sy)
    if not inside.any():
        return False

    def apply(mask):
        out = mask.copy()
        out[inside] = float(value)
        return out

    changed = session.edit_mask(apply, label)
    host.request_frame()
    return changed


class _MaskGesture(Operator):
    value = FloatProperty("遮罩值", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
                          description="1 是遮住，0 是放开")

    @classmethod
    def poll(cls, ctx) -> bool:
        return _sculpting(ctx) and view3d(ctx) is not None

    def _pixel(self, event):
        editor = self.editor
        if event is not None and getattr(event, "source", None) is editor.widget:
            return editor.pixel(event)
        return editor.cursor_pixel()

    def _finish(self, ctx) -> None:
        self.editor.set_overlay_shape(None)
        if getattr(ctx, "wm", None) is not None:
            ctx.wm.set_hint("")


@ops.register
class PaintMaskBoxGesture(_MaskGesture):
    idname = "paint.mask_box_gesture"
    label = "框选遮罩"
    description = "拖出一个框：左键把框里的遮住，中键把框里的放开（B）"

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.started = False
        self.current = self._pixel(event)
        self.start = self.current
        self.editor.set_overlay_shape(("cross", self.current))
        if getattr(ctx, "wm", None) is not None:
            ctx.wm.set_hint("框选遮罩：左键拖动遮住，中键拖动放开；右键或 Esc 取消")
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        from ..core.keymap import MIDDLEMOUSE

        editor = self.editor
        if event.type == MOUSEMOVE:
            self.current = self._pixel(event)
            editor.set_overlay_shape(("box", self.start, self.current) if self.started else ("cross", self.current))
            return RUNNING_MODAL
        if not self.started and event.value == PRESS and event.type in (LEFTMOUSE, MIDDLEMOUSE):
            self.started = True
            self.button = event.type
            self.mask_value = float(self.value) if event.type == LEFTMOUSE and not event.ctrl else \
                1.0 - float(self.value)
            self.start = self.current = self._pixel(event)
            return RUNNING_MODAL
        if self.started and event.value == RELEASE and event.type == self.button:
            self.current = self._pixel(event)
            self._finish(ctx)
            (x0, y0), (x1, y1) = self.start, self.current
            lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            if hi_x - lo_x < 1 or hi_y - lo_y < 1:
                return CANCELLED
            done = _mask_in_shape(ctx, editor, lambda sx, sy: (sx >= lo_x) & (sx <= hi_x) & (sy >= lo_y) &
                                  (sy <= hi_y), self.mask_value, self.label)
            return FINISHED if done else CANCELLED
        if event.value == PRESS and event.type in (RIGHTMOUSE, "ESC"):
            self._finish(ctx)
            return CANCELLED
        return RUNNING_MODAL


@ops.register
class PaintMaskLassoGesture(_MaskGesture):
    idname = "paint.mask_lasso_gesture"
    label = "套索遮罩"
    description = "按住拖出一圈，圈里的顶点遮罩设成给定的值（Ctrl+Shift+右键拖遮住，Ctrl+右键拖放开）"

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.button = event.type if event is not None and event.type in (LEFTMOUSE, RIGHTMOUSE) else RIGHTMOUSE
        self.path = [self._pixel(event)]
        self.editor.set_overlay_shape(("lasso", list(self.path)))
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            point = self._pixel(event)
            last = self.path[-1]
            if abs(point[0] - last[0]) + abs(point[1] - last[1]) >= 2.0:
                self.path.append(point)
                self.editor.set_overlay_shape(("lasso", list(self.path)))
            return RUNNING_MODAL
        if event.value == RELEASE and event.type == self.button:
            self._finish(ctx)
            if len(self.path) < 3:
                return CANCELLED
            from .object_ops import points_in_polygon

            path = np.asarray(self.path, np.float64)
            done = _mask_in_shape(ctx, self.editor, lambda sx, sy: points_in_polygon(sx, sy, path), float(self.value),
                                  self.label)
            return FINISHED if done else CANCELLED
        if event.value == PRESS and event.type == "ESC":
            self._finish(ctx)
            return CANCELLED
        return RUNNING_MODAL
