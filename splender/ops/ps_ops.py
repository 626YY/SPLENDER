"""照 Photoshop 习惯的几个快捷操作（绘制时、图像编辑器里）：D 默认黑白色、数字键设不透明度（Shift+数字设流量）、
Shift+S 稳定器开关、Alt+Backspace / Ctrl+Backspace 用前景色 / 背景色填一层、Ctrl+Shift+U 去色；
图像编辑器里按比例看（小键盘 1 / 2 / 4 / 8，Ctrl+小键盘 2 / 4 / 8 放大，Ctrl+1 一比一，Ctrl+0 看全部）和框选放大（Shift+B）。"""
from __future__ import annotations

from ..core import ops
from ..core.keymap import LEFTMOUSE, MIDDLEMOUSE, MOUSEMOVE, PRESS, RELEASE, RIGHTMOUSE
from ..core.ops import CANCELLED, FINISHED, RUNNING_MODAL, Operator
from ..core.props import EnumProperty, FloatProperty

_STABILIZER = {"last": 0.5}


def _brush(ctx):
    ts = ctx.app.tool_settings
    return ts.active_brush() if hasattr(ts, "active_brush") else ts.brush


@ops.register
class PaintDefaultColors(Operator):
    idname = "paint.default_colors"
    label = "默认颜色"
    description = "笔刷颜色换成黑色、备用颜色换成白色（D）"

    def execute(self, ctx) -> str:
        brush = ctx.app.tool_settings.brush
        brush.color = (0.0, 0.0, 0.0)
        brush.secondary_color = (1.0, 1.0, 1.0)
        return FINISHED


@ops.register
class PaintBrushSetValue(Operator):
    idname = "paint.brush_set_value"
    label = "设笔刷数值"
    description = "数字键设不透明度（1 是 10%，0 是 100%），Shift+数字设流量"
    searchable = False
    target = EnumProperty("设", items=[("opacity", "不透明度", ""), ("flow", "流量", "")], default="opacity")
    value = FloatProperty("数值", default=1.0, min=0.01, max=1.0, subtype="FACTOR")

    def execute(self, ctx) -> str:
        brush = _brush(ctx)
        if not hasattr(brush, self.target):
            return CANCELLED
        setattr(brush, self.target, float(self.value))
        label = "不透明度" if self.target == "opacity" else "流量"
        self.report(ctx, "%s %d%%" % (label, round(float(self.value) * 100)))
        ctx.app.notify("brush")
        return FINISHED


@ops.register
class PaintToggleStabilizer(Operator):
    idname = "paint.toggle_stabilizer"
    label = "稳定器开关"
    description = "笔划稳定器（让线条更顺）开或关（Shift+S）"

    def execute(self, ctx) -> str:
        brush = ctx.app.tool_settings.brush
        if float(brush.smooth) > 0.0:
            _STABILIZER["last"] = float(brush.smooth)
            brush.smooth = 0.0
            self.report(ctx, "稳定器关")
        else:
            brush.smooth = _STABILIZER["last"] or 0.5
            self.report(ctx, "稳定器开（%d%%）" % round(float(brush.smooth) * 100))
        ctx.app.notify("brush")
        return FINISHED


@ops.register
class LayerFillColor(Operator):
    idname = "layer.fill_color"
    label = "用颜色填充"
    description = "在当前图层上面加一层填充，颜色是笔刷颜色（Alt+Backspace）或备用颜色（Ctrl+Backspace）"
    which = EnumProperty("颜色", items=[("PRIMARY", "笔刷颜色", ""), ("SECONDARY", "备用颜色", "")], default="PRIMARY")

    @classmethod
    def poll(cls, ctx) -> bool:
        return getattr(ctx, "texture_set", None) is not None

    def execute(self, ctx) -> str:
        from ..doc.project import Layer
        from .layer_ops import _record, _room

        ts = ctx.texture_set
        if not _room(ts, 1):
            self.report(ctx, "一套贴图的图层已经满了", "WARNING")
            return CANCELLED
        brush = ctx.app.tool_settings.brush
        color = brush.color if self.which == "PRIMARY" else brush.secondary_color
        before = ts.structure()
        layer = Layer(name=ts.unique_name("颜色填充"), kind="FILL", fill_color=tuple(color), use_basecolor=True,
                      use_metallic=False, use_roughness=False, use_height=False)
        ts.add_layer(layer)
        _record(ctx, "用颜色填充", ts, before, added=[layer])
        from .select_ops import mask_new_layer

        mask_new_layer(ctx, layer, "用颜色填充")         # 有选区时只填选区里（照 Photoshop）
        return FINISHED


@ops.register
class LayerAddDesaturate(Operator):
    idname = "layer.add_desaturate"
    label = "去色"
    description = "加一个把下面变成灰度的调整层（Ctrl+Shift+U）"

    @classmethod
    def poll(cls, ctx) -> bool:
        return getattr(ctx, "texture_set", None) is not None

    def execute(self, ctx) -> str:
        from ..doc.project import Layer
        from .layer_ops import _record, _room

        ts = ctx.texture_set
        if not _room(ts, 1):
            self.report(ctx, "一套贴图的图层已经满了", "WARNING")
            return CANCELLED
        before = ts.structure()
        layer = Layer(name=ts.unique_name("去色"), kind="ADJUST", adjust_type="HSV", adj_saturation=-1.0,
                      use_metallic=False, use_roughness=False, use_height=False)
        ts.add_layer(layer)
        _record(ctx, "去色", ts, before, added=[layer])
        return FINISHED


# ====================================================================== 图像编辑器
def _image(ctx):
    editor = getattr(ctx, "editor", None)
    return editor if getattr(editor, "idname", "") == "IMAGE_EDITOR" else None


@ops.register
class ImageViewZoomRatio(Operator):
    idname = "image.view_zoom_ratio"
    label = "按比例看"
    description = "贴图的一个纹素占屏幕上几个像素（1 是一比一）"
    ratio = FloatProperty("比例", default=1.0, min=1.0 / 64.0, max=64.0)

    @classmethod
    def poll(cls, ctx) -> bool:
        editor = _image(ctx)
        return editor is not None and editor.view is not None and ctx.texture_set is not None

    def execute(self, ctx) -> str:
        editor = _image(ctx)
        ts = ctx.texture_set
        view = editor.view
        x, y = view.width / 2, view.height / 2
        target = float(ts.size) * float(self.ratio)
        view.zoom_at(target / max(view.zoom, 1e-9), x, y)
        editor.view_changed()
        return FINISHED


@ops.register
class ImageZoomBorder(Operator):
    idname = "image.zoom_border"
    label = "框选放大"
    description = "拖出一个框，把框里的部分放大到满屏（Shift+B）"

    @classmethod
    def poll(cls, ctx) -> bool:
        editor = _image(ctx)
        return editor is not None and editor.view is not None

    def invoke(self, ctx, event) -> str:
        self.editor = _image(ctx)
        self.started = False
        self.start = self.current = self._pixel(event)
        if getattr(ctx, "wm", None) is not None:
            ctx.wm.set_hint("拖出一个框：放大到框里的范围；右键或 Esc 取消")
        return RUNNING_MODAL

    def _pixel(self, event):
        editor = self.editor
        if event is not None and getattr(event, "source", None) is editor.widget:
            return editor.pixel(event)
        return (editor.view.width / 2, editor.view.height / 2)

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            self.current = self._pixel(event)
            return RUNNING_MODAL
        if not self.started and event.value == PRESS and event.type in (LEFTMOUSE, MIDDLEMOUSE):
            self.started = True
            self.start = self.current = self._pixel(event)
            return RUNNING_MODAL
        if self.started and event.value == RELEASE and event.type in (LEFTMOUSE, MIDDLEMOUSE):
            self.current = self._pixel(event)
            self._finish(ctx)
            (x0, y0), (x1, y1) = self.start, self.current
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w < 4 or h < 4:
                return CANCELLED
            view = self.editor.view
            center = view.pixel_to_uv((x0 + x1) / 2, (y0 + y1) / 2)
            factor = min(view.width / w, view.height / h)
            view.zoom = float(min(max(view.zoom * factor, 16.0), 16384.0 * 64.0))
            view.center = center
            view.dirty = True
            self.editor.view_changed()
            return FINISHED
        if event.value == PRESS and event.type in (RIGHTMOUSE, "ESC"):
            self._finish(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def _finish(self, ctx) -> None:
        if getattr(ctx, "wm", None) is not None:
            ctx.wm.set_hint("")
