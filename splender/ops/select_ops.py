"""选区（照 Photoshop）：选框（矩形、椭圆）、套索、多边形套索、魔棒、快速选择这些工具，和「选择」菜单里的全选、
取消选择、重新选择、反选、修改（羽化、扩展、收缩、平滑、边界）、蒙版和选区互转；有选区时 Delete 清掉选区里的内容。

三维视口里画的形状按屏幕投到模型上（只选看得见的地方），UV 视图里按 UV 盖。选区本身在 engine/selection.py。
拖之前按住 Shift 是添加、Alt 是减去、Shift+Alt 是交叉；没有选区时 Shift 是正方形（正圆）、Alt 是从中心拉，
拖的时候再按 Shift、Alt 也是。不拖、点一下是取消选择。
"""
from __future__ import annotations

import json
import math

import numpy as np

from ..core import ops, registry
from ..core.keymap import DOUBLE, LEFTMOUSE, MOUSEMOVE, PRESS, RELEASE, RIGHTMOUSE
from ..core.ops import CANCELLED, FINISHED, RUNNING_MODAL, Operator
from ..core.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, StringProperty
from ..doc.project import PLANE_COLOR, PLANE_HEIGHT, PLANE_MASK, PLANE_MR, SELECT_MODE_ITEMS
from .tool_ops import (MAX_DETAIL, SPACE_ITEMS, _after_bake, _engine, _screen_setup, _set_shape, _texel_density,
                       _tool_editor, _uv_scale)

SHAPE_KINDS = [("RECT", "矩形", ""), ("ELLIPSE", "椭圆", ""), ("POLYGON", "多边形", "")]
MODIFY_ITEMS = [("FEATHER", "羽化", "选区边缘柔和过渡"), ("EXPAND", "扩展", "选区往外长"),
                ("CONTRACT", "收缩", "选区往里缩"), ("SMOOTH", "平滑", "去掉选区边上的小尖角、小洞"),
                ("BORDER", "边界", "只留选区边上的一圈")]
CLOSE_PX = 8.0           # 多边形套索：点回到第一个点附近这么近（屏幕像素）就闭合


def _selection(ctx):
    """(引擎, 这套贴图的选区)；没有贴图时 (None, None)。"""
    engine = _engine(ctx)
    ts = getattr(ctx, "texture_set", None)
    if engine is None or ts is None or ts.uid not in engine.sets:
        return None, None
    return engine, engine.selection(ts)


def _has_set(ctx) -> bool:
    ts = getattr(ctx, "texture_set", None)
    return ts is not None


def _modifier_mode(event, mode: str, active: bool) -> str:
    """拖之前按住的键定合并方式（有选区时才算）：Shift 添加、Alt 减去、Shift+Alt 交叉；都没按用工具设置里的。"""
    if active and event is not None:
        if event.shift and event.alt:
            return "INTERSECT"
        if event.shift:
            return "ADD"
        if event.alt:
            return "SUBTRACT"
    return mode


def _changed(ctx) -> None:
    _engine_, selection = _selection(ctx)
    if selection is not None and selection.emptied:
        selection.emptied = False
        ctx.app.report("选区里什么都没选上，已取消选择", "WARNING")
    ctx.app.notify("selection")
    request = getattr(ctx.app, "request_frame", None)
    if request is not None:
        request()


# ====================================================================== 选框、套索、多边形套索
class _ShapeSelect(Operator):
    """拖（点）出一个形状，合进选区。三维视口：点是屏幕像素，投到模型上；UV 视图：点是纹素（u × 边长, v × 边长）。"""
    redo = True
    mode = EnumProperty("方式", items=SELECT_MODE_ITEMS, default="SET", description="新画的和原来的选区怎么合")
    feather = FloatProperty("羽化", default=0.0, min=0.0, max=500.0, precision=1, unit="px",
                            description="选区边缘柔和过渡多宽（屏幕像素）")
    antialias = BoolProperty("消除锯齿", default=True, description="选区边缘平滑")
    occlude = BoolProperty("只选看得见的地方", default=True, description="在模型上选时，被挡住的地方不选")
    kind = EnumProperty("形状", items=SHAPE_KINDS, default="RECT", hidden=True)
    points = StringProperty("点", default="[]", hidden=True)
    space = EnumProperty("空间", items=SPACE_ITEMS, default="UV", hidden=True)
    px_scale = FloatProperty("每像素纹素", default=1.0, min=1e-6, hidden=True)
    detail = FloatProperty("精细", default=2.0, min=1.0, hidden=True)
    matrix = FloatVectorProperty("投影", default=tuple(np.identity(4).ravel()), hidden=True)
    view_size = FloatVectorProperty("视口大小", default=(1.0, 1.0), hidden=True)
    capture = IntProperty("遮挡", default=0, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return _has_set(ctx)

    def draw_redo(self, layout) -> None:
        layout.prop(self, "mode", expand=True)
        layout.prop(self, "feather")
        layout.prop(self, "antialias")
        if self.space == "SCREEN":
            layout.prop(self, "occlude")

    # ---- 开始拖 ----
    def _begin(self, ctx, event) -> bool:
        editor = _tool_editor(ctx)
        engine, selection = _selection(ctx)
        if editor is None or selection is None:
            return False
        settings = ctx.app.tool_settings.selection
        self.editor = editor
        self.engine = engine
        self.active = selection.active
        self.mode = _modifier_mode(event, settings.mode, selection.active)
        self.feather = settings.feather
        self.antialias = settings.antialias
        self.occlude = settings.occlude
        self.held = (bool(event.shift), bool(event.alt))
        self.button = event.type
        return True

    def _constrain(self, event) -> tuple:
        """拖的时候：正方形（正圆）、从中心拉。有选区时拖之前就按着的键是合并方式，不算。"""
        square = bool(event.shift) and not (self.active and self.held[0])
        center = bool(event.alt) and not (self.active and self.held[1])
        return square, center

    # ---- 拖完：记下几何，做 ----
    def _finish(self, ctx, kind: str, pixels: list) -> str:
        """pixels 是视口像素的点。点一下（太小）时：新选区方式下取消选择。"""
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        if max(xs) - min(xs) < 3.0 and max(ys) - min(ys) < 3.0:
            if self.mode == "SET" and self.active:
                return ops.call("select.none", ctx, invoke=False)
            return CANCELLED
        editor = self.editor
        view = editor.view
        ts = ctx.texture_set
        res = float(ts.size)
        self.kind = kind
        if getattr(view, "is_2d", False):
            self.space = "UV"
            points = [tuple(float(c) * res for c in view.pixel_to_uv(x, y)) for x, y in pixels]
            self.px_scale = _uv_scale(view, pixels[0][0], pixels[0][1], res)
        else:
            _screen_setup(self, editor, self.engine)
            points = [(float(x), float(y)) for x, y in pixels]
            mid = (sum(xs) / len(xs), sum(ys) / len(ys))
            self.detail = min(MAX_DETAIL, max(1.0, _texel_density(self.engine, view, mid[0], mid[1], res)))
        self.points = json.dumps([[round(x, 3), round(y, 3)] for x, y in points])
        return self.execute(ctx)

    def execute(self, ctx) -> str:
        from ..engine import raster

        engine, selection = _selection(ctx)
        if selection is None:
            return CANCELLED
        points = json.loads(self.points or "[]")
        if len(points) < 2:
            return CANCELLED
        res = float(ctx.texture_set.size)
        screen = self.space == "SCREEN"
        if screen and engine.meshmap(ctx.texture_set.uid) is None:
            return _after_bake(ctx, self, ctx.texture_set, "选上")    # 投到模型上要模型贴图：先烘焙，好了自动选
        if screen:
            scale = min(MAX_DETAIL, max(1.0, self.detail))           # 图上几个像素一个屏幕像素
            feather = float(self.feather)
        else:
            scale = 1.0                                              # 一个纹素一个像素（太大时自动画粗）
            feather = float(self.feather) * max(1e-6, self.px_scale)
        mask, (x0, y0, x1, y1) = raster.mask_shape(self.kind, points, scale, antialias=self.antialias, feather=feather)
        if screen:
            params = {"image": mask, "rect": (x0, y0, x1, y1), "opacity": 1.0, "matrix": tuple(self.matrix),
                      "view_size": tuple(self.view_size), "capture": int(self.capture), "occlude": bool(self.occlude)}
            name = "SCREEN_IMAGE"
        else:
            params = {"image": mask, "rect": (x0 / res, y0 / res, x1 / res, y1 / res), "opacity": 1.0}
            name = "IMAGE"
        try:
            done = selection.combine(self.mode, name, params, self.label)
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "没选上：%s" % error, "ERROR")
            return CANCELLED
        if not done:
            return CANCELLED
        _changed(ctx)
        return FINISHED

    def cancel(self, ctx) -> None:
        _set_shape(getattr(self, "editor", None), None)


@ops.register
class SelectMarquee(_ShapeSelect):
    idname = "select.marquee"
    label = "选框"
    description = "拖出矩形或椭圆选区。拖之前按住 Shift 添加、Alt 减去；拖的时候按 Shift 正方形、Alt 从中心拉"
    shape = EnumProperty("形状", items=SHAPE_KINDS[:2], default="RECT")

    def invoke(self, ctx, event) -> str:
        if event is None:
            return self.execute(ctx)
        if not self._begin(ctx, event):
            return CANCELLED
        self.p0 = self.editor.pixel(event)
        self.p1 = self.p0
        return RUNNING_MODAL

    def _corners(self, event) -> tuple:
        (x0, y0), (x1, y1) = self.p0, self.p1
        square, center = self._constrain(event)
        if square:
            side = max(abs(x1 - x0), abs(y1 - y0))
            x1 = x0 + math.copysign(side, (x1 - x0) or 1.0)
            y1 = y0 + math.copysign(side, (y1 - y0) or 1.0)
        if center:
            x0, y0 = 2 * x0 - x1, 2 * y0 - y1
        return (x0, y0), (x1, y1)

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            self.p1 = editor.pixel(event)
            a, b = self._corners(event)
            _set_shape(editor, ("box" if self.shape == "RECT" else "ellipse", a, b))
            return RUNNING_MODAL
        if event.type == self.button and event.value == RELEASE:
            _set_shape(editor, None)
            a, b = self._corners(event)
            return self._finish(ctx, self.shape, [a, b])
        if (event.type == "ESC" or event.type == RIGHTMOUSE) and event.value == PRESS:
            _set_shape(editor, None)
            return CANCELLED
        return RUNNING_MODAL


@ops.register
class SelectLasso(_ShapeSelect):
    idname = "select.lasso"
    label = "套索"
    description = "按住拖，圈出一块随手画的选区。拖之前按住 Shift 添加、Alt 减去"

    def invoke(self, ctx, event) -> str:
        if event is None:
            return self.execute(ctx)
        if not self._begin(ctx, event):
            return CANCELLED
        self.trail = [self.editor.pixel(event)]
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            x, y = editor.pixel(event)
            last = self.trail[-1]
            if math.hypot(x - last[0], y - last[1]) >= 2.0:
                self.trail.append((x, y))
                _set_shape(editor, ("lasso", list(self.trail)))
            return RUNNING_MODAL
        if event.type == self.button and event.value == RELEASE:
            _set_shape(editor, None)
            if len(self.trail) < 3:
                return self._finish(ctx, "POLYGON", self.trail[:1] * 2)
            return self._finish(ctx, "POLYGON", self.trail)
        if (event.type == "ESC" or event.type == RIGHTMOUSE) and event.value == PRESS:
            _set_shape(editor, None)
            return CANCELLED
        return RUNNING_MODAL


@ops.register
class SelectPolygon(_ShapeSelect):
    idname = "select.polygon"
    label = "多边形套索"
    description = ("一下一下点出多边形的顶点：点回第一个点、双击或按回车闭合；退格去掉上一个点，Esc 放弃。"
                   "按住 Shift 按 45° 吸附")

    def invoke(self, ctx, event) -> str:
        if event is None:
            return self.execute(ctx)
        if not self._begin(ctx, event):
            return CANCELLED
        start = self.editor.pixel(event)
        self.corners = [start]
        self.cursor = start
        self._show()
        return RUNNING_MODAL

    def _show(self) -> None:
        trail = list(self.corners) + [self.cursor]
        _set_shape(self.editor, ("multi", [("lasso", trail), ("points", list(self.corners), (1.0, 1.0, 1.0), 5.0)]))

    def _snap(self, event, x: float, y: float) -> tuple:
        if not event.shift or not self.corners:
            return x, y
        px, py = self.corners[-1]
        angle = round(math.atan2(y - py, x - px) / (math.pi / 4)) * (math.pi / 4)
        length = math.hypot(x - px, y - py)
        return px + math.cos(angle) * length, py + math.sin(angle) * length

    def _close(self, ctx) -> str:
        _set_shape(self.editor, None)
        if len(self.corners) < 3:
            return CANCELLED
        return self._finish(ctx, "POLYGON", self.corners)

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            self.cursor = self._snap(event, *editor.pixel(event))
            self._show()
            return RUNNING_MODAL
        if event.type == LEFTMOUSE and event.value == DOUBLE:
            return self._close(ctx)
        if event.type == LEFTMOUSE and event.value == PRESS:
            x, y = self._snap(event, *editor.pixel(event))
            first = self.corners[0]
            if len(self.corners) >= 3 and math.hypot(x - first[0], y - first[1]) <= CLOSE_PX:
                return self._close(ctx)
            last = self.corners[-1]
            if math.hypot(x - last[0], y - last[1]) >= 1.0:
                self.corners.append((x, y))
            self._show()
            return RUNNING_MODAL
        if event.type in ("RET", "NUMPAD_ENTER") and event.value == PRESS:
            return self._close(ctx)
        if event.type in ("BACK_SPACE", "DEL") and event.value == PRESS:
            if len(self.corners) > 1:
                self.corners.pop()
                self._show()
                return RUNNING_MODAL
            _set_shape(editor, None)
            return CANCELLED
        if (event.type == "ESC" or event.type == RIGHTMOUSE) and event.value == PRESS:
            _set_shape(editor, None)
            return CANCELLED
        return RUNNING_MODAL


# ====================================================================== 魔棒、快速选择
def _sample_image(ctx, engine, ts, size: int, sample_all: bool):
    """找相近颜色用的图（行序和贴图一样）：合成结果（8 位）或者当前图层自己。"""
    from ..engine import fill
    from ..engine.projectio import composite_image

    if sample_all:
        return composite_image(engine, ts, size, dtype=np.uint8)
    layer = ts.active_layer
    return fill.layer_image(engine, layer, PLANE_COLOR, size) if layer is not None else composite_image(
        engine, ts, size, dtype=np.uint8)


def _surface_uv(ctx, editor, engine, x: float, y: float, what: str, op) -> tuple | None:
    """鼠标下的贴图位置；三维视口里没点到这套贴图时报告原因、返回 None。"""
    view = editor.view
    if getattr(view, "is_2d", False):
        return tuple(float(c) for c in view.pixel_to_uv(x, y))
    hit = engine.uv_hit(view, x, y)
    if hit is None:
        op.report(ctx, "没点到模型", "WARNING")
        return None
    if hit[0] is not ctx.texture_set:
        op.report(ctx, "点到的是另一套贴图「%s」，先在图层里换到它再%s" % (hit[0].name, what), "WARNING")
        return None
    return hit[1]


def _texels_per_px(engine, view, x: float, y: float, res: float) -> float:
    if getattr(view, "is_2d", False):
        return _uv_scale(view, x, y, res)
    return _texel_density(engine, view, x, y, res)


@ops.register
class SelectWand(Operator):
    idname = "select.wand"
    label = "魔棒"
    description = "点一下，选上颜色相近的一片。拖之前按住 Shift 添加、Alt 减去"
    redo = True
    mode = EnumProperty("方式", items=SELECT_MODE_ITEMS, default="SET")
    tolerance = FloatProperty("容差", default=32.0 / 255.0, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                              description="颜色差多少以内算一片")
    contiguous = BoolProperty("连续的", default=True, description="只选和点到的地方连成一片的")
    sample_all = BoolProperty("对所有图层取样", default=True, description="按看到的合成结果找相近颜色")
    antialias = BoolProperty("消除锯齿", default=True, description="选区边缘平滑")
    feather = FloatProperty("羽化", default=0.0, min=0.0, max=500.0, precision=1, unit="px",
                            description="选区边缘柔和过渡多宽（屏幕像素）")
    uv = FloatVectorProperty("位置", default=(0.5, 0.5), precision=4, hidden=True)
    px_scale = FloatProperty("每像素纹素", default=1.0, min=1e-6, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return _has_set(ctx)

    def draw_redo(self, layout) -> None:
        layout.prop(self, "mode", expand=True)
        layout.prop(self, "tolerance", slider=True)
        layout.prop(self, "contiguous")
        layout.prop(self, "sample_all")
        layout.prop(self, "antialias")
        layout.prop(self, "feather")

    def invoke(self, ctx, event) -> str:
        if event is None:
            return self.execute(ctx)
        editor = _tool_editor(ctx)
        engine, selection = _selection(ctx)
        if editor is None or selection is None:
            return CANCELLED
        settings = ctx.app.tool_settings.selection
        self.mode = _modifier_mode(event, settings.mode, selection.active)
        for name in ("tolerance", "contiguous", "sample_all", "antialias", "feather"):
            setattr(self, name, getattr(settings, name))
        x, y = editor.pixel(event)
        uv = _surface_uv(ctx, editor, engine, x, y, "用魔棒", self)
        if uv is None:
            return CANCELLED
        self.uv = uv
        self.px_scale = _texels_per_px(engine, editor.view, x, y, float(ctx.texture_set.size))
        return self.execute(ctx)

    def execute(self, ctx) -> str:
        import cv2

        from ..engine import fill

        engine, selection = _selection(ctx)
        if selection is None:
            return CANCELLED
        ts = ctx.texture_set
        size = fill.mask_size(ts.size)
        image = _sample_image(ctx, engine, ts, size, self.sample_all)
        u, v = self.uv
        mask = fill.similar_mask(image, int(u * size), int(v * size), float(self.tolerance), bool(self.contiguous))
        if not mask.any():
            self.report(ctx, "这里没有可选的地方", "WARNING")
            return CANCELLED
        mask = (mask * 255.0).astype(np.uint8)
        pixels_per_texel = size / float(ts.size)
        if self.antialias:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.6, sigmaY=0.6)
        if self.feather > 0:
            sigma = max(0.3, self.feather * self.px_scale * pixels_per_texel * 0.5)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
        try:
            done = selection.combine(self.mode, "IMAGE", {"image": mask, "rect": (0.0, 0.0, 1.0, 1.0), "opacity": 1.0},
                                     self.label)
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "没选上：%s" % error, "ERROR")
            return CANCELLED
        if not done:
            return CANCELLED
        _changed(ctx)
        return FINISHED


@ops.register
class SelectQuick(Operator):
    idname = "select.quick"
    label = "快速选择"
    description = "像画画一样拖：笔刷碰到的地方连同颜色相近、连成一片的一起选上，边停在颜色变化的地方。按住 Alt 减去"
    _count = 0

    @classmethod
    def poll(cls, ctx) -> bool:
        return _has_set(ctx)

    def invoke(self, ctx, event) -> str:
        from ..engine import fill

        if event is None:
            return CANCELLED
        editor = _tool_editor(ctx)
        engine, selection = _selection(ctx)
        if editor is None or selection is None:
            return CANCELLED
        settings = ctx.app.tool_settings.selection
        ts = ctx.texture_set
        self.editor = editor
        self.engine = engine
        self.selection = selection
        self.settings = settings
        self.size = fill.mask_size(ts.size)
        self.image = _sample_image(ctx, engine, ts, self.size, settings.sample_all)
        if event.alt or settings.mode == "SUBTRACT":
            self.mode = "SUBTRACT"
        else:
            self.mode = "ADD" if selection.active else "SET"
        self.button = event.type
        self.steps = 0
        self.last = None
        self.per_px = None
        selection.begin_batch()                      # 拖一笔只记一步撤销（界面也只在松手时刷新）
        self._dab(ctx, *editor.pixel(event))
        return RUNNING_MODAL

    def _radius(self, x: float, y: float) -> float:
        return max(1.0, float(self.settings.quick_size) * 0.5)

    def _dab(self, ctx, x: float, y: float) -> None:
        """在 (x, y) 这一下：笔刷盖住的地方，加上和它颜色相近、连成一片的（只在附近找），合进选区。"""
        from ..engine import fill

        editor = self.editor
        radius_px = self._radius(x, y)
        _set_shape(editor, ("circle", (x, y), radius_px))
        self.last = (x, y)
        view = editor.view
        if getattr(view, "is_2d", False):
            uv = tuple(float(c) for c in view.pixel_to_uv(x, y))
        else:
            hit = self.engine.uv_hit(view, x, y)
            if hit is None or hit[0] is not ctx.texture_set:
                return
            uv = hit[1]
        res = float(ctx.texture_set.size)
        if getattr(self, "per_px", None) is None:
            # 图上每屏幕像素几个像素：一笔里差不多，第一下算出来就一直用（省掉每一下多打一条射线）
            self.per_px = _texels_per_px(self.engine, view, x, y, res) * self.size / res
        r = max(1.0, radius_px * self.per_px)
        size = self.size
        cx, cy = int(uv[0] * size), int(uv[1] * size)
        reach = int(math.ceil(r * 3.0)) + 2
        x0, y0 = max(0, cx - reach), max(0, cy - reach)
        x1, y1 = min(size, cx + reach + 1), min(size, cy + reach + 1)
        if x1 <= x0 or y1 <= y0:
            return
        crop = np.ascontiguousarray(self.image[y0:y1, x0:x1])
        grown = fill.similar_mask(crop, cx - x0, cy - y0, float(self.settings.tolerance), True)
        yy, xx = np.ogrid[y0:y1, x0:x1]              # 按行、列广播，不用生成整块坐标
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        mask = np.maximum(grown, disk.astype(np.float32))
        mask = (mask * 255.0).astype(np.uint8)
        if self.settings.antialias:
            import cv2

            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.6, sigmaY=0.6)
        rect = (x0 / size, y0 / size, x1 / size, y1 / size)
        try:
            if self.selection.combine(self.mode, "IMAGE", {"image": mask, "rect": rect, "opacity": 1.0}, self.label):
                self.steps += 1
                if self.mode == "SET":
                    self.mode = "ADD"
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "没选上：%s" % error, "ERROR")
        request = getattr(ctx.app, "request_frame", None)
        if request is not None:
            request()

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            x, y = editor.pixel(event)
            radius = self._radius(x, y)
            if self.last is None or math.hypot(x - self.last[0], y - self.last[1]) >= max(2.0, radius * 0.35):
                self._dab(ctx, x, y)
            else:
                _set_shape(editor, ("circle", (x, y), radius))
            return RUNNING_MODAL
        if event.type == self.button and event.value == RELEASE:
            _set_shape(editor, None)
            done = self.selection.end_batch(self.label)
            _changed(ctx)
            return FINISHED if done else CANCELLED
        if (event.type == "ESC" or event.type == RIGHTMOUSE) and event.value == PRESS:
            _set_shape(editor, None)
            self.selection.cancel_batch()
            _changed(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def cancel(self, ctx) -> None:
        _set_shape(getattr(self, "editor", None), None)
        selection = getattr(self, "selection", None)
        if selection is not None:
            selection.cancel_batch()


# ====================================================================== 「选择」菜单
class _SelectionOp(Operator):
    @classmethod
    def poll(cls, ctx) -> bool:
        return _has_set(ctx)

    def _run(self, ctx, method: str, *args) -> str:
        _engine_, selection = _selection(ctx)
        if selection is None:
            return CANCELLED
        if not getattr(selection, method)(*args):
            return CANCELLED
        _changed(ctx)
        return FINISHED


@ops.register
class SelectAll(_SelectionOp):
    idname = "select.all"
    label = "全选"
    description = "整张贴图都选上"

    def execute(self, ctx) -> str:
        return self._run(ctx, "select_all")


@ops.register
class SelectNone(_SelectionOp):
    idname = "select.none"
    label = "取消选择"
    description = "不要选区了（画笔、滤镜又能改所有地方）"

    @classmethod
    def poll(cls, ctx) -> bool:
        _engine_, selection = _selection(ctx)
        return selection is not None and selection.active

    def execute(self, ctx) -> str:
        return self._run(ctx, "deselect")


@ops.register
class SelectReselect(_SelectionOp):
    idname = "select.reselect"
    label = "重新选择"
    description = "把刚取消的选区找回来"

    @classmethod
    def poll(cls, ctx) -> bool:
        _engine_, selection = _selection(ctx)
        return selection is not None and selection.can_reselect()

    def execute(self, ctx) -> str:
        return self._run(ctx, "reselect")


@ops.register
class SelectInvert(_SelectionOp):
    idname = "select.invert"
    label = "反选"
    description = "选上原来没选的，原来选上的去掉"

    @classmethod
    def poll(cls, ctx) -> bool:
        _engine_, selection = _selection(ctx)
        return selection is not None and selection.active

    def execute(self, ctx) -> str:
        return self._run(ctx, "invert")


@ops.register
class SelectModify(_SelectionOp):
    idname = "select.modify"
    label = "修改选区"
    description = "羽化、扩展、收缩、平滑选区，或者只留选区边上的一圈"
    redo = True
    kind = EnumProperty("方式", items=MODIFY_ITEMS, default="FEATHER")
    amount = FloatProperty("宽度", default=8.0, min=0.5, max=2048.0, precision=1, unit="px",
                           description="按贴图的纹素算")

    @classmethod
    def poll(cls, ctx) -> bool:
        _engine_, selection = _selection(ctx)
        return selection is not None and selection.active

    def execute(self, ctx) -> str:
        return self._run(ctx, "modify", self.kind, float(self.amount))


@ops.register
class SelectFromMask(_SelectionOp):
    idname = "select.from_mask"
    label = "载入蒙版为选区"
    description = "把当前图层的蒙版当成选区（白的选上、黑的不选）"

    @classmethod
    def poll(cls, ctx) -> bool:
        ts = getattr(ctx, "texture_set", None)
        layer = ts.active_layer if ts is not None else None
        return layer is not None and layer.has_mask

    def execute(self, ctx) -> str:
        return self._run(ctx, "load_mask", ctx.texture_set.active_layer)


@ops.register
class SelectToMask(_SelectionOp):
    idname = "select.to_mask"
    label = "按选区加蒙版"
    description = "给当前图层加一个蒙版：选区里的显示，选区外的藏起来"

    @classmethod
    def poll(cls, ctx) -> bool:
        _engine_, selection = _selection(ctx)
        layer = ctx.texture_set.active_layer if selection is not None else None
        return selection is not None and selection.active and layer is not None and not layer.has_mask

    def execute(self, ctx) -> str:
        result = self._run(ctx, "to_mask", ctx.texture_set.active_layer)
        if result == FINISHED:
            ctx.app.notify("layers")
        return result


def _content_planes(engine, layer, target_mask: bool) -> dict:
    """「清掉选区里的」「只留选区里的」要改哪些页：画蒙版时是蒙版，否则图层已有的内容页。"""
    if target_mask:
        return {PLANE_MASK: (True,)}
    store = engine.layers.stores.get(layer.uid)
    planes = {}
    for plane, use in ((PLANE_COLOR, (True,)), (PLANE_MR, (True, True)), (PLANE_HEIGHT, (True,))):
        if store is not None and plane in store.planes:
            planes[plane] = use
    return planes


@ops.register
class LayerClearSelection(Operator):
    idname = "layer.clear_selection"
    label = "清除选区里的内容"
    description = "把当前图层在选区里的内容擦掉（画蒙版时把选区里的蒙版涂黑）"

    @classmethod
    def poll(cls, ctx) -> bool:
        _engine_, selection = _selection(ctx)
        layer = ctx.texture_set.active_layer if selection is not None else None
        return (selection is not None and selection.active and layer is not None and not layer.locked
                and (layer.kind == "PAINT" or (ctx.texture_set.paint_target == "MASK" and layer.has_mask)))

    def execute(self, ctx) -> str:
        engine, _selection_ = _selection(ctx)
        ts = ctx.texture_set
        layer = ts.active_layer
        target_mask = ts.paint_target == "MASK" and layer.has_mask
        planes = _content_planes(engine, layer, target_mask)
        if not planes:
            return CANCELLED
        try:
            count = engine.apply_filter(ts, layer, "SEL_CLEAR", {}, planes, self.label)
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "没清掉：%s" % error, "ERROR")
            return CANCELLED
        if count == 0:
            return CANCELLED
        ctx.app.notify("layers")
        return FINISHED


# ====================================================================== 图层操作碰到选区时（照 Photoshop）
def mask_new_layer(ctx, layer, label: str) -> None:
    """新建的填充层、调整层：有选区时给它加一个和选区一样的蒙版，和新建合成一步撤销。"""
    engine, selection = _selection(ctx)
    if selection is None or not selection.active or layer.has_mask:
        return
    if selection.to_mask(layer, label) and engine.history is not None:
        engine.history.merge_last(2, label)


def keep_selected(ctx, layer, label: str, steps: int = 2) -> bool:
    """刚复制出来的图层：有选区时只留选区里的内容（通过拷贝的图层），和前面 steps − 1 步合成一步撤销。"""
    engine, selection = _selection(ctx)
    if selection is None or not selection.active or layer.kind != "PAINT":
        return False
    planes = _content_planes(engine, layer, False)
    if not planes:
        return False
    if engine.apply_filter(ctx.texture_set, layer, "SEL_KEEP", {}, planes, label) and engine.history is not None:
        engine.history.merge_last(steps, label)
    return True


@ops.register
class LayerCutSelection(Operator):
    idname = "layer.cut_selection"
    label = "通过剪切的图层"
    description = "把当前图层在选区里的内容挪到一个新图层上（原图层上擦掉）"

    @classmethod
    def poll(cls, ctx) -> bool:
        _engine_, selection = _selection(ctx)
        layer = ctx.texture_set.active_layer if selection is not None else None
        return (selection is not None and selection.active and layer is not None and layer.kind == "PAINT"
                and not layer.locked)

    def execute(self, ctx) -> str:
        engine, _selection_ = _selection(ctx)
        ts = ctx.texture_set
        source = ts.active_layer
        planes = _content_planes(engine, source, False)
        if not planes:
            return CANCELLED
        # 复制图层在有选区时已经只留选区里的（通过拷贝，一步）；再把原图层选区里的擦掉，合成一步
        if ops.call("layer.duplicate", ctx, invoke=False, uid=source.uid) != FINISHED:
            return CANCELLED
        copy = ts.active_layer
        copy.name = ts.unique_name(source.name + " 剪切")
        engine.apply_filter(ts, source, "SEL_CLEAR", {}, planes, self.label)
        if engine.history is not None:
            engine.history.merge_last(2, self.label)
        ctx.app.notify("layers")
        return FINISHED


# ====================================================================== 菜单（照 Photoshop 的「选择」菜单）
@registry.register_menu
class SelectModifyMenu(registry.Menu):
    idname = "PAINT_MT_select_modify"
    label = "修改"

    def draw(self, layout, ctx) -> None:
        for kind in ("BORDER", "SMOOTH", "EXPAND", "CONTRACT", "FEATHER"):
            label = next(item[1] for item in MODIFY_ITEMS if item[0] == kind)
            layout.operator("select.modify", text=label, kind=kind)


@registry.register_menu
class SelectMenu(registry.Menu):
    idname = "PAINT_MT_select"
    label = "选择"

    def draw(self, layout, ctx) -> None:
        layout.operator("select.all")
        layout.operator("select.none")
        layout.operator("select.reselect")
        layout.operator("select.invert")
        layout.separator()
        layout.menu("PAINT_MT_select_modify")
        layout.separator()
        layout.operator("select.from_mask")
        layout.operator("select.to_mask")
        layout.separator()
        layout.operator("layer.clear_selection")
        layout.operator("layer.cut_selection")


def draw_tool_header(layout, tools) -> bool:
    """绘制模式标题栏里选区工具的设置；当前工具不是选区工具时返回 False。"""
    from ..tools.paint_tools import SELECT_TOOLS

    if tools.tool not in SELECT_TOOLS:
        return False
    settings = tools.selection
    layout.prop(settings, "mode", text="", expand=True, icon_only=True)
    if tools.tool in ("paint.select_wand", "paint.select_quick"):
        if tools.tool == "paint.select_quick":
            layout.prop(settings, "quick_size", text="大小")
        layout.prop(settings, "tolerance", text="容差", slider=True)
        if tools.tool == "paint.select_wand":
            layout.prop(settings, "contiguous", toggle=True)
        layout.prop(settings, "sample_all", toggle=True)
    else:
        layout.prop(settings, "feather", text="羽化")
        layout.prop(settings, "occlude", toggle=True)
    layout.prop(settings, "antialias", toggle=True)
    return True
