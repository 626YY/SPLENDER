"""绘制模式里不是笔刷的工具（照 Photoshop）：渐变、油漆桶。在三维视口和 UV 视图里都能用。"""
from __future__ import annotations

import math

import numpy as np

from ..core import ops
from ..core.keymap import LEFTMOUSE, MOUSEMOVE, PRESS, RELEASE, RIGHTMOUSE
from ..core.ops import CANCELLED, FINISHED, RUNNING_MODAL, Operator
from ..core.props import (BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, RampProperty,
                          StringProperty)
from ..doc.project import (ALIGN_ITEMS, FILL_MODES, GRADIENT_SHAPE_CODE, GRADIENT_SHAPES, PLANE_COLOR, PLANE_HEIGHT,
                           PLANE_MASK, PLANE_MR, SHAPE_ITEMS, _font_items)


def _engine(ctx):
    host = getattr(ctx.app, "host", None)
    if host is None:
        return None
    host.make_current()
    return host.engine


def paint_target(ctx):
    """要改的图层和页：画蒙版时改蒙版（任何有蒙版的层），否则改绘制层的颜色。返回 (图层, {页: 用法}) 或 (None, 原因)。"""
    ts = getattr(ctx, "texture_set", None)
    layer = ts.active_layer if ts is not None else None
    if layer is None:
        return None, "先新建一个绘制图层"
    if layer.locked:
        return None, "图层「%s」已锁定" % layer.name
    if ts.paint_target == "MASK" and layer.has_mask:
        return layer, {PLANE_MASK: (True,)}
    if layer.kind != "PAINT":
        return None, "「%s」不是绘制图层，不能直接改。可以给它加蒙版，或新建绘制图层" % layer.name
    return layer, {PLANE_COLOR: (True,)}


def _tool_editor(ctx):
    editor = getattr(ctx, "editor", None)
    idname = getattr(editor, "idname", "")
    if idname in ("VIEW_3D", "IMAGE_EDITOR") and getattr(editor, "view", None) is not None:
        return editor
    return None


def _set_shape(editor, shape) -> None:
    setter = getattr(editor, "set_overlay_shape", None)
    if callable(setter):
        setter(shape)


SPACE_ITEMS = [("UV", "UV", "按贴图上的位置"), ("SCREEN", "屏幕", "按模型投到屏幕上的位置")]
MAX_DETAIL = 16.0                 # 往模型上投字、形状时，图最多画得比屏幕细几倍
MAX_PIXELS = 16 * 1024 * 1024     # 字、形状画成的图最多多少像素（再大就画粗一点、放大了盖上）


def _after_bake(ctx, op, ts, what: str) -> str:
    """在模型上做这一步要用模型贴图，还没烘焙：先烘焙，烘好了自动按同样的参数再做一次。"""
    app = ctx.app
    engine = _engine(ctx)
    ensure = getattr(app, "ensure_meshmaps", None)
    if engine is None or ensure is None:
        op.report(ctx, "在模型上%s要先烘焙模型贴图" % what, "WARNING")
        return CANCELLED
    props = {name: getattr(op, name) for name in type(op).properties()}
    area = getattr(getattr(op, "editor", None), "area", None) or getattr(ctx, "area", None)
    idname = op.idname

    def finished(job) -> None:
        if getattr(job, "ts", None) is not ts:
            return
        if finished in engine.bake_finished:
            engine.bake_finished.remove(finished)
        if job.result is None or engine.meshmap(ts.uid) is None:
            return

        def run() -> None:
            wm = getattr(app, "wm", None)
            if wm is None:
                return
            later = wm.context(area=area, use_mouse=False)
            if getattr(later, "texture_set", None) is not ts:
                app.report("已经换到别的贴图，%s没做" % what, "WARNING")
                return
            ops.call(idname, later, invoke=False, **props)

        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, run)

    engine.bake_finished.append(finished)
    ensure(ts)
    if engine.meshmap(ts.uid) is None and engine.bake is None and ts not in getattr(app, "_bake_queue", ()):
        engine.bake_finished.remove(finished)
        op.report(ctx, "在模型上%s要先烘焙模型贴图，没能开始烘焙" % what, "WARNING")
        return CANCELLED
    op.report(ctx, "先准备模型贴图，好了自动%s" % what)
    return CANCELLED


def _uv_scale(view, x: float, y: float, res: float) -> float:
    """UV 视图里一个屏幕像素是几个纹素。"""
    u0, v0 = view.pixel_to_uv(x, y)
    u1, v1 = view.pixel_to_uv(x + 1.0, y)
    return max(1e-6, math.hypot(float(u1) - float(u0), float(v1) - float(v0)) * res)


def _texel_density(engine, view, x: float, y: float, res: float) -> float:
    """三维视口里 (x, y) 处一个屏幕像素大约盖住几个纹素（按点到的三角形算，没点到模型时按 2）。"""
    hit = engine.uv_hit(view, x, y)
    if hit is None:
        return 2.0
    _ts, _uv, obj, tri = hit
    corners = np.asarray(obj.data.positions, np.float64).reshape(-1, 3, 3)[tri]
    uvs = np.asarray(obj.data.uvs, np.float64).reshape(-1, 3, 2)[tri]
    edges = np.stack([corners[1] - corners[0], corners[2] - corners[0]], 1)
    duv = np.stack([uvs[1] - uvs[0], uvs[2] - uvs[0]], 1)
    try:
        stretch = float(np.linalg.norm(duv @ np.linalg.pinv(edges), 2))      # 每单位长度最多走多少 UV
    except np.linalg.LinAlgError:
        return 2.0
    matrix = np.asarray(view.camera.view_projection(view.aspect), np.float64)
    depth = max(1e-6, float(matrix[3, :3] @ corners.mean(0) + matrix[3, 3]))
    pixel = 2.0 * depth / max(float(np.linalg.norm(matrix[1, :3])), 1e-12) / max(1.0, float(view.height))
    return stretch * pixel * res if stretch > 0 else 2.0


def _fit(width: float, height: float) -> float:
    """画成图时缩小的比例（≤ 1）：边长不超过 MAX_SIDE，像素数不超过 MAX_PIXELS。"""
    from ..engine.raster import MAX_SIDE

    width, height = max(1.0, float(width)), max(1.0, float(height))
    return min(1.0, MAX_SIDE / max(width, height), math.sqrt(MAX_PIXELS / (width * height)))


def _screen_setup(op, editor, engine) -> None:
    """三维视口：记下这一刻的投影、视口大小、看到的表面（判断遮挡用）。"""
    from ..engine.filters import capture_view

    view = editor.view
    op.space = "SCREEN"
    op.matrix = tuple(np.asarray(view.camera.view_projection(view.aspect), np.float64).ravel())
    op.view_size = (float(view.width), float(view.height))
    op.capture = capture_view(engine, view)


@ops.register
class PaintGradient(Operator):
    idname = "paint.gradient"
    label = "渐变"
    description = "拖一条线，在当前绘制层（画蒙版时是蒙版）上铺渐变。按住 Shift 角度按 45° 吸附"
    redo = True
    shape = EnumProperty("形状", items=GRADIENT_SHAPES, default="LINEAR")
    reverse = BoolProperty("反向", default=False, description="两头的颜色对调")
    opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2)
    space = EnumProperty("空间", items=SPACE_ITEMS, default="UV", hidden=True)
    start = FloatVectorProperty("起点", default=(0.0, 0.0), hidden=True)
    end = FloatVectorProperty("终点", default=(1.0, 0.0), hidden=True)
    matrix = FloatVectorProperty("投影", default=tuple(np.identity(4).ravel()), hidden=True)
    view_size = FloatVectorProperty("视口大小", default=(1.0, 1.0), hidden=True)
    ramp = RampProperty("色带", hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return paint_target(ctx)[0] is not None

    # ---- 拖线 ----
    def invoke(self, ctx, event) -> str:
        if event is None:                       # 脚本、自检直接调用：按给的参数做
            return self.execute(ctx)
        editor = _tool_editor(ctx)
        if editor is None:
            return CANCELLED
        layer, reason = paint_target(ctx)
        if layer is None:
            self.report(ctx, reason, "WARNING")
            return CANCELLED
        settings = ctx.app.tool_settings.gradient
        self.shape = settings.shape
        self.reverse = settings.reverse
        self.opacity = settings.opacity
        if settings.source == "RAMP":
            self.ramp = settings.ramp
        else:
            brush = ctx.app.tool_settings.brush
            self.ramp = ("LINEAR", ((0.0, *brush.color[:3], 1.0), (1.0, *brush.secondary_color[:3], 1.0)))
        self.editor = editor
        self.button = event.type
        self.p0 = editor.pixel(event)
        self.p1 = self.p0
        _set_shape(editor, ("line", self.p0, self.p1))
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            x, y = editor.pixel(event)
            if event.shift:
                # 按 45° 吸附（和 Photoshop 一样）
                dx, dy = x - self.p0[0], y - self.p0[1]
                angle = round(math.atan2(dy, dx) / (math.pi / 4)) * (math.pi / 4)
                length = math.hypot(dx, dy)
                x, y = self.p0[0] + math.cos(angle) * length, self.p0[1] + math.sin(angle) * length
            self.p1 = (x, y)
            _set_shape(editor, ("line", self.p0, self.p1))
            return RUNNING_MODAL
        if event.type == self.button and event.value == RELEASE:
            _set_shape(editor, None)
            if math.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]) < 3.0:
                return CANCELLED
            view = editor.view
            if getattr(view, "is_2d", False):
                self.space = "UV"
                self.start = tuple(view.pixel_to_uv(*self.p0))
                self.end = tuple(view.pixel_to_uv(*self.p1))
            else:
                self.space = "SCREEN"
                self.start = self.p0
                self.end = self.p1
                self.matrix = tuple(np.asarray(view.camera.view_projection(view.aspect), np.float64).ravel())
                self.view_size = (float(view.width), float(view.height))
            return self.execute(ctx)
        if (event.type == "ESC" or event.type == RIGHTMOUSE) and event.value == PRESS:
            _set_shape(editor, None)
            return CANCELLED
        return RUNNING_MODAL

    def cancel(self, ctx) -> None:
        _set_shape(getattr(self, "editor", None), None)

    # ---- 铺上去 ----
    def execute(self, ctx) -> str:
        layer, planes = paint_target(ctx)
        if layer is None:
            self.report(ctx, planes, "WARNING")
            return CANCELLED
        engine = _engine(ctx)
        if engine is None:
            return CANCELLED
        ts = ctx.texture_set
        if self.space == "SCREEN" and engine.meshmap(ts.uid) is None:
            return _after_bake(ctx, self, ts, "铺上渐变")
        params = {"shape": GRADIENT_SHAPE_CODE.get(self.shape, 0), "reverse": bool(self.reverse),
                  "opacity": float(self.opacity), "space": self.space, "a": tuple(self.start), "b": tuple(self.end),
                  "matrix": tuple(self.matrix), "view_size": tuple(self.view_size), "ramp": self.ramp}
        try:
            count = engine.apply_filter(ts, layer, "GRADIENT", params, planes, self.label)
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "渐变没做成：%s" % error, "ERROR")
            return CANCELLED
        if count == 0:
            return CANCELLED
        ctx.app.notify("layers")
        return FINISHED


def brush_planes(ctx):
    """油漆桶、文字、形状这类「盖上去」的工具要改哪些页：画蒙版时是蒙版，否则按笔刷开着的通道。
    返回 (图层, {页: 用法}) 或 (None, 原因)。"""
    layer, planes = paint_target(ctx)
    if layer is None or PLANE_MASK in planes:
        return layer, planes
    brush = ctx.app.tool_settings.brush
    planes = {}
    if brush.use_basecolor:
        planes[PLANE_COLOR] = (True,)
    if brush.use_metallic or brush.use_roughness:
        planes[PLANE_MR] = (bool(brush.use_metallic), bool(brush.use_roughness))
    if brush.use_height:
        planes[PLANE_HEIGHT] = (True,)
    if not planes:
        return None, "笔刷没有开任何通道"
    return layer, planes


def stamp_screen(ctx, layer, planes, image: np.ndarray, rect, op, label: str) -> str:
    """把一张 RGBA 图（第一行在上）按屏幕上的矩形（像素）投到模型上，数值页用笔刷的金属度、粗糙度、高度。"""
    engine = _engine(ctx)
    if engine is None:
        return CANCELLED
    brush = ctx.app.tool_settings.brush
    params = {"image": image, "rect": tuple(float(v) for v in rect), "opacity": float(op.opacity),
              "matrix": tuple(op.matrix), "view_size": tuple(op.view_size), "capture": int(op.capture),
              "occlude": bool(op.occlude),
              "values": (float(brush.metallic), float(brush.roughness), float(brush.height), 0.0)}
    count = engine.apply_filter(ctx.texture_set, layer, "SCREEN_IMAGE", params, planes, label)
    if count == 0:
        op.report(ctx, "没落到模型上", "WARNING")
        return CANCELLED
    ctx.app.notify("layers")
    return FINISHED


def stamp_image(ctx, layer, planes, image: np.ndarray, rect, opacity: float, label: str, color=None) -> str:
    """把一张 RGBA 图（行序和贴图一样）按 UV 矩形盖到图层上，数值页用笔刷的金属度、粗糙度、高度。
    image 也可以是单通道的遮罩（8 位），这时颜色是 color。"""
    engine = _engine(ctx)
    if engine is None:
        return CANCELLED
    brush = ctx.app.tool_settings.brush
    params = {"image": image, "rect": tuple(float(v) for v in rect), "opacity": float(opacity),
              "values": (float(brush.metallic), float(brush.roughness), float(brush.height), 0.0)}
    if color is not None:
        params["color2"] = (float(color[0]), float(color[1]), float(color[2]), 1.0)
    count = engine.apply_filter(ctx.texture_set, layer, "IMAGE", params, planes, label)
    if count == 0:
        return CANCELLED
    ctx.app.notify("layers")
    return FINISHED


@ops.register
class PaintFill(Operator):
    idname = "paint.fill"
    label = "油漆桶"
    description = "点一下，把一片地方填上笔刷的颜色和材质：相近颜色、UV 岛、网格部件或整层"
    redo = True
    mode = EnumProperty("范围", items=FILL_MODES, default="SIMILAR")
    tolerance = FloatProperty("容差", default=32.0 / 255.0, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                              description="颜色差多少以内算一片")
    contiguous = BoolProperty("连续的", default=True, description="只填和点到的地方连成一片的")
    sample_all = BoolProperty("对所有图层取样", default=True, description="按看到的合成结果找相近颜色")
    opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2)
    uv = FloatVectorProperty("位置", default=(0.5, 0.5), hidden=True)
    object_uid = IntProperty("物体", default=0, hidden=True)
    triangle = IntProperty("三角形", default=-1, min=-1, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return paint_target(ctx)[0] is not None

    def draw_redo(self, layout) -> None:
        layout.prop(self, "mode")
        if self.mode == "SIMILAR":
            layout.prop(self, "tolerance", slider=True)
            layout.prop(self, "contiguous")
            layout.prop(self, "sample_all")
        layout.prop(self, "opacity", slider=True)

    def invoke(self, ctx, event) -> str:
        if event is None:                       # 脚本、自检直接调用：按给的参数做
            return self.execute(ctx)
        editor = _tool_editor(ctx)
        if editor is None:
            return CANCELLED
        settings = ctx.app.tool_settings.fill
        for name in ("mode", "tolerance", "contiguous", "sample_all", "opacity"):
            setattr(self, name, getattr(settings, name))
        ts = ctx.texture_set
        x, y = editor.pixel(event)
        view = editor.view
        if getattr(view, "is_2d", False):
            u, v = (float(c) for c in view.pixel_to_uv(x, y))
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                return CANCELLED
            from ..engine.fill import uv_triangle_at

            found = uv_triangle_at(ctx.app.project, ts, u, v)
            self.object_uid, self.triangle = (found[0].uid, found[1]) if found is not None else (0, -1)
        else:
            engine = _engine(ctx)
            hit = engine.uv_hit(view, x, y) if engine is not None else None
            if hit is None:
                self.report(ctx, "没点到模型", "WARNING")
                return CANCELLED
            hit_ts, (u, v), obj, tri = hit
            if hit_ts is not ts:
                self.report(ctx, "点到的是另一套贴图「%s」，先在图层里换到它" % hit_ts.name, "WARNING")
                return CANCELLED
            self.object_uid, self.triangle = obj.uid, tri
        self.uv = (u, v)
        return self.execute(ctx)

    def execute(self, ctx) -> str:
        from ..engine import fill
        from ..engine.projectio import composite_image

        layer, planes = brush_planes(ctx)
        if layer is None:
            self.report(ctx, planes, "WARNING")
            return CANCELLED
        engine = _engine(ctx)
        if engine is None:
            return CANCELLED
        ts = ctx.texture_set
        size = fill.mask_size(ts.size)
        u, v = self.uv
        if self.mode == "LAYER":
            mask = np.ones((size, size), np.float32)
        elif self.mode == "SIMILAR":
            if self.sample_all:
                image = composite_image(engine, ts, size, dtype=np.uint8)
            else:
                image = fill.layer_image(engine, layer, PLANE_COLOR, size)
            mask = fill.similar_mask(image, int(u * size), int(v * size), float(self.tolerance), bool(self.contiguous))
        else:
            obj = next((o for o in ctx.app.project.objects if o.uid == self.object_uid), None)
            if obj is None or self.triangle < 0:
                self.report(ctx, "这里没有模型的三角形", "WARNING")
                return CANCELLED
            mask = fill.triangle_mask(engine.ctx, fill.group_triangles(obj, self.triangle, self.mode), size)
        if not mask.any():
            self.report(ctx, "没有可填的地方", "WARNING")
            return CANCELLED
        coverage = np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8)
        try:
            return stamp_image(ctx, layer, planes, coverage, (0.0, 0.0, 1.0, 1.0), self.opacity, self.label,
                               color=_fill_color(ctx, planes))
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "没填上：%s" % error, "ERROR")
            return CANCELLED


@ops.register
class PaintCloneSource(Operator):
    idname = "paint.clone_source"
    label = "定下仿制来源"
    description = "仿制图章、修复画笔从哪里取样（按住 Alt 点一下）"
    searchable = False
    uv = FloatVectorProperty("位置", default=(0.5, 0.5), precision=4, description="来源在贴图上的位置")

    @classmethod
    def poll(cls, ctx) -> bool:
        return _tool_editor(ctx) is not None

    def invoke(self, ctx, event) -> str:
        if event is None:                       # 脚本、自检直接调用：按给的参数做
            return self.execute(ctx)
        editor = _tool_editor(ctx)
        engine = _engine(ctx)
        if editor is None or engine is None:
            return CANCELLED
        x, y = editor.pixel(event)
        uv = engine.surface_uv(editor.view, x, y)
        if uv is None:
            self.report(ctx, "没点到模型", "WARNING")
            return CANCELLED
        self.uv = uv
        return self.execute(ctx)

    def execute(self, ctx) -> str:
        settings = ctx.app.tool_settings.effect
        settings.source_u, settings.source_v = self.uv
        settings.has_source = True
        settings.offset_set = False
        self.report(ctx, "已定下仿制来源")
        ctx.app.notify("tool")
        return FINISHED


def _fill_color(ctx, planes) -> tuple:
    brush = ctx.app.tool_settings.brush
    return (brush.mask_value,) * 3 if PLANE_MASK in planes else tuple(brush.color[:3])


@ops.register
class PaintText(Operator):
    idname = "paint.text"
    label = "文字"
    description = "点一下，在那里写字（笔刷的颜色和材质）。做完在「调整上一步」里还能改字、字号、旋转"
    redo = True
    text = StringProperty("文字", default="", description="要写的字，可以换行")
    font = EnumProperty("字体", items=_font_items, default="")
    size = FloatProperty("字号", default=48.0, min=2.0, max=2048.0, precision=0, unit="px",
                         description="字有多高（屏幕像素，按写字时的缩放）")
    bold = BoolProperty("粗体", default=False)
    italic = BoolProperty("斜体", default=False)
    rotation = FloatProperty("旋转", default=0.0, min=-180.0, max=180.0, unit="°", precision=0)
    align = EnumProperty("对齐", items=ALIGN_ITEMS, default="LEFT")
    line_spacing = FloatProperty("行距", default=1.2, min=0.5, max=4.0, precision=2)
    opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2)
    occlude = BoolProperty("只写在看得见的地方", default=True, description="在模型上写字时，被挡住的地方不写")
    space = EnumProperty("空间", items=SPACE_ITEMS, default="UV", hidden=True)
    uv = FloatVectorProperty("位置", default=(0.5, 0.5), precision=4, hidden=True, description="字的中心在贴图上的位置")
    point = FloatVectorProperty("屏幕位置", default=(0.0, 0.0), hidden=True)
    px_scale = FloatProperty("每像素纹素", default=1.0, min=1e-6, hidden=True)
    detail = FloatProperty("精细", default=2.0, min=1.0, hidden=True)
    matrix = FloatVectorProperty("投影", default=tuple(np.identity(4).ravel()), hidden=True)
    view_size = FloatVectorProperty("视口大小", default=(1.0, 1.0), hidden=True)
    capture = IntProperty("遮挡", default=0, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return paint_target(ctx)[0] is not None

    def draw_redo(self, layout) -> None:
        for name in ("text", "font", "size"):
            layout.prop(self, name)
        row = layout.row(align=True, heading="样式")
        row.prop(self, "bold", toggle=True)
        row.prop(self, "italic", toggle=True)
        layout.prop(self, "align", expand=True)
        layout.prop(self, "line_spacing")
        layout.prop(self, "rotation")
        layout.prop(self, "opacity", slider=True)
        if self.space == "SCREEN":
            layout.prop(self, "occlude")

    def invoke(self, ctx, event) -> str:
        if event is None:                       # 脚本、自检直接调用：按给的参数做
            return self.execute(ctx)
        editor = _tool_editor(ctx)
        engine = _engine(ctx)
        if editor is None or engine is None:
            return CANCELLED
        layer, reason = paint_target(ctx)
        if layer is None:
            self.report(ctx, reason, "WARNING")
            return CANCELLED
        settings = ctx.app.tool_settings.text
        for name in ("font", "size", "bold", "italic", "rotation", "align", "line_spacing", "opacity", "occlude"):
            setattr(self, name, getattr(settings, name))
        x, y = editor.pixel(event)
        view = editor.view
        res = float(ctx.texture_set.size)
        if getattr(view, "is_2d", False):
            self.space = "UV"
            self.uv = tuple(float(c) for c in view.pixel_to_uv(x, y))
            self.px_scale = _uv_scale(view, x, y, res)
        else:
            self.editor = editor
            _screen_setup(self, editor, engine)
            self.point = (float(x), float(y))
            self.detail = min(MAX_DETAIL, max(1.0, _texel_density(engine, view, x, y, res)))
        if not self.text:
            from PySide6.QtWidgets import QInputDialog

            value, ok = QInputDialog.getMultiLineText(None, "文字", "要写的字", getattr(PaintText, "_last", "") or "")
            if not ok or not value.strip():
                return CANCELLED
            self.text = value
            PaintText._last = value
        return self.execute(ctx)

    def execute(self, ctx) -> str:
        from ..engine import raster

        layer, planes = brush_planes(ctx)
        if layer is None:
            self.report(ctx, planes, "WARNING")
            return CANCELLED
        if not self.text.strip():
            return CANCELLED
        ts = ctx.texture_set
        screen = self.space == "SCREEN"
        if screen:
            engine = _engine(ctx)
            if engine is None:
                return CANCELLED
            if engine.meshmap(ts.uid) is None:
                return _after_bake(ctx, self, ts, "写上字")
        # 图上几个像素是一个单位（屏幕像素或纹素）：在模型上按那里的纹素密度画细一点，图太大就画粗一点
        factor = min(MAX_DETAIL, max(1.0, self.detail)) if screen else max(1e-6, self.px_scale)
        style = {"bold": self.bold, "italic": self.italic, "rotation": self.rotation, "line_spacing": self.line_spacing}
        want = self.size * factor
        extent = raster.text_extent(self.text, self.font, want, **style)
        shrink = _fit(*extent)
        image = raster.tint(raster.text_image(self.text, self.font, want * shrink, align=self.align, **style),
                            _fill_color(ctx, planes))
        h, w = image.shape[:2]
        try:
            if screen:
                ratio = factor * shrink                 # 图上几个像素是屏幕上一个像素
                cx, cy = self.point
                rect = (cx - w / 2.0 / ratio, cy - h / 2.0 / ratio, cx + w / 2.0 / ratio, cy + h / 2.0 / ratio)
                return stamp_screen(ctx, layer, planes, np.ascontiguousarray(image[::-1]), rect, self, self.label)
            ratio = shrink * float(ts.size)             # 图上几个像素是 UV 的 1
            u, v = self.uv
            rect = (u - w / 2.0 / ratio, v - h / 2.0 / ratio, u + w / 2.0 / ratio, v + h / 2.0 / ratio)
            return stamp_image(ctx, layer, planes, image, rect, self.opacity, self.label)
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "没写上：%s" % error, "ERROR")
            return CANCELLED


@ops.register
class PaintShape(Operator):
    idname = "paint.shape"
    label = "形状"
    description = "拖出矩形、椭圆、多边形或直线（笔刷的颜色和材质）。按住 Shift 等比，按住 Alt 从中心往外拉"
    redo = True
    kind = EnumProperty("形状", items=SHAPE_ITEMS, default="RECT")
    fill = BoolProperty("填充", default=True)
    stroke = FloatProperty("描边", default=0.0, min=0.0, max=1024.0, precision=0, unit="px")
    radius = FloatProperty("圆角", default=32.0, min=0.0, max=4096.0, precision=0, unit="px")
    sides = IntProperty("边数", default=6, min=3, max=64)
    opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2)
    occlude = BoolProperty("只画在看得见的地方", default=True, description="在模型上画形状时，被挡住的地方不画")
    space = EnumProperty("空间", items=SPACE_ITEMS, default="UV", hidden=True)
    start = FloatVectorProperty("起点", default=(0.25, 0.25), precision=4, hidden=True)
    end = FloatVectorProperty("终点", default=(0.75, 0.75), precision=4, hidden=True)
    px_scale = FloatProperty("每像素纹素", default=1.0, min=1e-6, hidden=True)
    detail = FloatProperty("精细", default=2.0, min=1.0, hidden=True)
    matrix = FloatVectorProperty("投影", default=tuple(np.identity(4).ravel()), hidden=True)
    view_size = FloatVectorProperty("视口大小", default=(1.0, 1.0), hidden=True)
    capture = IntProperty("遮挡", default=0, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return paint_target(ctx)[0] is not None

    def invoke(self, ctx, event) -> str:
        if event is None:                       # 脚本、自检直接调用：按给的参数做
            return self.execute(ctx)
        editor = _tool_editor(ctx)
        if editor is None:
            return CANCELLED
        settings = ctx.app.tool_settings.shape
        for name in ("kind", "fill", "stroke", "radius", "sides", "opacity", "occlude"):
            setattr(self, name, getattr(settings, name))
        self.editor = editor
        self.button = event.type
        self.p0 = editor.pixel(event)
        self.p1 = self.p0
        return RUNNING_MODAL

    def draw_redo(self, layout) -> None:
        layout.prop(self, "kind")
        layout.prop(self, "fill")
        layout.prop(self, "stroke")
        if self.kind == "ROUNDED":
            layout.prop(self, "radius")
        if self.kind == "POLYGON":
            layout.prop(self, "sides")
        layout.prop(self, "opacity", slider=True)
        if self.space == "SCREEN":
            layout.prop(self, "occlude")

    def _corners(self, event) -> tuple:
        (x0, y0), (x1, y1) = self.p0, self.p1
        if event.shift:
            side = max(abs(x1 - x0), abs(y1 - y0))
            x1 = x0 + math.copysign(side, (x1 - x0) or 1.0)
            y1 = y0 + math.copysign(side, (y1 - y0) or 1.0)
        if event.alt:
            x0, y0 = 2 * x0 - x1, 2 * y0 - y1
        return (x0, y0), (x1, y1)

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            self.p1 = editor.pixel(event)
            a, b = self._corners(event)
            _set_shape(editor, ("line", a, b) if self.kind == "LINE" else ("box", a, b))
            return RUNNING_MODAL
        if event.type == self.button and event.value == RELEASE:
            _set_shape(editor, None)
            a, b = self._corners(event)
            if math.hypot(b[0] - a[0], b[1] - a[1]) < 3.0:
                return CANCELLED
            engine = _engine(ctx)
            if engine is None:
                return CANCELLED
            view = editor.view
            res = float(ctx.texture_set.size)
            if getattr(view, "is_2d", False):
                self.space = "UV"
                self.start = tuple(float(c) for c in view.pixel_to_uv(*a))
                self.end = tuple(float(c) for c in view.pixel_to_uv(*b))
                self.px_scale = _uv_scale(view, a[0], a[1], res)
            else:
                _screen_setup(self, editor, engine)
                self.start, self.end = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
                mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
                self.detail = min(MAX_DETAIL, max(1.0, _texel_density(engine, view, mid[0], mid[1], res)))
            return self.execute(ctx)
        if (event.type == "ESC" or event.type == RIGHTMOUSE) and event.value == PRESS:
            _set_shape(editor, None)
            return CANCELLED
        return RUNNING_MODAL

    def cancel(self, ctx) -> None:
        _set_shape(getattr(self, "editor", None), None)

    def execute(self, ctx) -> str:
        from ..engine import raster

        layer, planes = brush_planes(ctx)
        if layer is None:
            self.report(ctx, planes, "WARNING")
            return CANCELLED
        ts = ctx.texture_set
        res = float(ts.size)
        screen = self.space == "SCREEN"
        if screen:
            engine = _engine(ctx)
            if engine is None:
                return CANCELLED
            if engine.meshmap(ts.uid) is None:
                return _after_bake(ctx, self, ts, "画上形状")
        (x0, y0), (x1, y1) = self.start, self.end
        # 图上几个像素是一个单位：在模型上是屏幕像素（按纹素密度画细一点），UV 视图里是纹素（描边、圆角按屏幕像素换算）
        if screen:
            unit = min(MAX_DETAIL, max(1.0, self.detail))
            width, height, px = abs(x1 - x0), abs(y1 - y0), 1.0
        else:
            unit = 1.0
            width, height, px = abs(x1 - x0) * res, abs(y1 - y0) * res, max(1e-6, self.px_scale)
        stroke = (self.stroke if (self.fill or self.stroke > 0) else 8.0) * px
        pad = 2.0 * max(2.0, stroke)
        ratio = unit * _fit((width + pad) * unit, (height + pad) * unit)
        image = raster.shape_image(self.kind, width * ratio, height * ratio, fill=self.fill, stroke=stroke * ratio,
                                   radius=self.radius * px * ratio, sides=self.sides)
        if screen:
            image = image[::-1]                                   # 第一行在上（屏幕的方向）
            flip = (x1 > x0) != (y1 > y0)
        else:
            flip = (x1 > x0) == (y1 > y0)
        if self.kind == "LINE" and flip:
            image = image[:, ::-1]                                # 拖的方向是另一条对角线
        image = raster.tint(np.ascontiguousarray(image), _fill_color(ctx, planes))
        h, w = image.shape[:2]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        try:
            if screen:
                rect = (cx - w / 2.0 / ratio, cy - h / 2.0 / ratio, cx + w / 2.0 / ratio, cy + h / 2.0 / ratio)
                return stamp_screen(ctx, layer, planes, image, rect, self, self.label)
            scale = ratio * res                                   # 图上几个像素是 UV 的 1
            rect = (cx - w / 2.0 / scale, cy - h / 2.0 / scale, cx + w / 2.0 / scale, cy + h / 2.0 / scale)
            return stamp_image(ctx, layer, planes, image, rect, self.opacity, self.label)
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "没画上：%s" % error, "ERROR")
            return CANCELLED
