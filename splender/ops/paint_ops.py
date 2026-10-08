"""绘制操作：落笔、调笔刷、取色。"""
from __future__ import annotations

import time

from ..core import ops
from ..core.keymap import LEFTMOUSE, MOUSEMOVE, PRESS, RELEASE, RIGHTMOUSE
from ..core.ops import CANCELLED, FINISHED, PASS_THROUGH, RUNNING_MODAL, Operator
from ..core.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from .view3d_ops import view3d


def _painting(ctx) -> bool:
    """当前是用笔刷画的工具（渐变这类工具有自己的操作）。"""
    from ..tools.paint_tools import STROKE_TOOLS

    tool = ctx.app.tool_settings.tool
    return view3d(ctx) is not None and (tool in STROKE_TOOLS or tool.startswith("sculpt."))


def _paint_editor(ctx):
    """可以落笔的编辑器：三维视口，或者用绘制工具时的 UV 视图。"""
    editor = ctx.editor
    idname = getattr(editor, "idname", "")
    if idname == "IMAGE_EDITOR" and getattr(editor, "view", None) is not None:
        from ..tools.paint_tools import STROKE_TOOLS

        return editor if ctx.app.tool_settings.tool in STROKE_TOOLS else None
    if idname == "VIEW_3D" and _painting(ctx):
        return editor
    return None


@ops.register
class PaintStroke(Operator):
    idname = "paint.stroke"
    label = "绘制"
    searchable = False
    invert = BoolProperty("反向", default=False, description="画笔时擦除，橡皮时绘制")

    @classmethod
    def poll(cls, ctx) -> bool:
        return _paint_editor(ctx) is not None

    def invoke(self, ctx, event) -> str:
        editor = _paint_editor(ctx)
        app = ctx.app
        tools = app.tool_settings
        erase = (tools.tool == "paint.eraser") != bool(self.invert)
        if event.is_eraser:
            erase = True
        self.editor = editor
        self.button = event.type
        x, y = editor.pixel(event)
        pressure = event.pressure if event.is_tablet else 1.0
        host = app.host
        host.make_current()
        engine = host.engine
        if not engine.stroke_begin(editor.view, x, y, pressure, event.time, tools, erase):
            if engine.stroke_reject:
                self.report(ctx, engine.stroke_reject, "WARNING")
            return CANCELLED
        host.note_input(event.time)
        host.request_frame()
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        app = ctx.app
        host = app.host
        engine = host.engine
        if event.type == MOUSEMOVE:
            if event.source is self.editor.widget:
                x, y = self.editor.pixel(event)
                engine.stroke_move(x, y, event.pressure if event.is_tablet else 1.0, event.time)
                host.note_input(event.time)
                host.request_frame()
            return RUNNING_MODAL
        if event.type == self.button and event.value == RELEASE:
            host.make_current()
            engine.stroke_end()
            host.request_frame()
            return FINISHED
        if (event.type == "ESC" or event.type == RIGHTMOUSE) and event.value == PRESS:
            self.cancel(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def cancel(self, ctx) -> None:
        host = ctx.app.host
        host.make_current()
        host.engine.stroke_cancel()
        host.request_frame()


@ops.register
class PaintBrushResize(Operator):
    idname = "paint.brush_resize"
    label = "调整笔刷"
    description = "移动鼠标调整，左键或回车确认，右键或 Esc 取消"
    searchable = False
    target = EnumProperty("对象", items=[("size", "直径", ""), ("flow", "流量", ""), ("hardness", "硬度", ""),
                                       ("opacity", "不透明度", "")], default="size")

    @classmethod
    def poll(cls, ctx) -> bool:
        return _painting(ctx)

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.brush = ctx.app.tool_settings.active_brush()
        self.initial = float(getattr(self.brush, self.target))
        self.start_x = event.x if event is not None else 0.0
        self.anchor = self.editor._mouse
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
                if self.target == "size":
                    value = self.initial + dx
                else:
                    value = self.initial + dx / 240.0
                setattr(self.brush, self.target, value)
                # 调整期间光标留在原地，只让圆圈变化
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
class PaintBrushScale(Operator):
    idname = "paint.brush_scale"
    label = "缩放笔刷直径"
    searchable = False
    factor = FloatProperty("倍率", default=1.1, min=0.1, max=10.0)

    @classmethod
    def poll(cls, ctx) -> bool:
        return _painting(ctx)

    def execute(self, ctx) -> str:
        brush = ctx.app.tool_settings.active_brush()
        if brush.size_unit == "SCENE":
            brush.scene_size = brush.scene_size * self.factor
        else:
            brush.size = max(1.0, brush.size * self.factor)
        editor = view3d(ctx)
        if editor is not None:
            editor.refresh_cursor()
        return FINISHED


@ops.register
class PaintSampleColor(Operator):
    idname = "paint.sample_color"
    label = "吸取颜色"
    description = "把鼠标下的颜色设为笔刷颜色"

    @classmethod
    def poll(cls, ctx) -> bool:
        return view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        if event is None or editor._mouse is None:
            return CANCELLED
        host = ctx.app.host
        host.make_current()
        x, y = editor._mouse
        # 临时切到只看基础色再取，得到的是贴图本身的颜色而不是受光后的颜色
        view = editor.view
        shading = view.shading
        cursor = view.cursor
        saved = (shading.mode, shading.channel)
        with shading.changed.block():
            shading.mode, shading.channel = "CHANNEL", "basecolor"
            view.cursor = None
            host.engine.renderer.render(view)
            color = host.engine.sample_color(view, x, y)
            shading.mode, shading.channel = saved
            view.cursor = cursor
        view.dirty = True
        host.request_frame()
        if color is None:
            return CANCELLED
        ctx.app.tool_settings.brush.color = color
        return FINISHED


@ops.register
class PaintSwapColors(Operator):
    idname = "paint.swap_colors"
    label = "交换颜色"
    description = "交换笔刷颜色和备用颜色"

    def execute(self, ctx) -> str:
        brush = ctx.app.tool_settings.brush
        brush.color, brush.secondary_color = brush.secondary_color, brush.color
        return FINISHED


@ops.register
class PaintToolSet(Operator):
    idname = "paint.tool_set"
    label = "切换工具"
    searchable = False
    tool = StringProperty("工具", default="paint.brush")

    def execute(self, ctx) -> str:
        ctx.app.tool_settings.tool = self.tool
        return FINISHED


@ops.register
class PaintToolCycle(Operator):
    idname = "paint.tool_cycle"
    label = "轮换工具"
    description = "在几种工具之间轮换（和 Photoshop 一样，再按一次换到同组的下一个）"
    searchable = False
    tools = StringProperty("工具", default="paint.gradient,paint.fill", description="逗号隔开")

    def execute(self, ctx) -> str:
        names = [name.strip() for name in self.tools.split(",") if name.strip()]
        if not names:
            return CANCELLED
        current = ctx.app.tool_settings.tool
        ctx.app.tool_settings.tool = names[(names.index(current) + 1) % len(names)] if current in names else names[0]
        return FINISHED


@ops.register
class PaintTargetSet(Operator):
    idname = "paint.target_set"
    label = "切换绘制目标"
    searchable = False
    target = EnumProperty("目标", items=[("CONTENT", "内容", ""), ("MASK", "蒙版", "")], default="CONTENT")

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.texture_set is not None

    def execute(self, ctx) -> str:
        ctx.texture_set.paint_target = self.target
        return FINISHED


def draw_brush_popover(layout, ctx, holder=None) -> None:
    """右键浮层的内容：当前笔刷的常用设置和颜色。holder 用来保存取色盘的绑定，浮层关闭前不能被回收。"""
    from types import SimpleNamespace

    from ..editors.properties_editor import BrushMaterialPanel, BrushPanel

    holder = holder if holder is not None else SimpleNamespace()
    BrushPanel.draw(holder, layout, ctx)
    if BrushMaterialPanel.poll(ctx):
        layout.separator()
        BrushMaterialPanel.draw(holder, layout, ctx)
    return holder


@ops.register
class PaintContextMenu(Operator):
    idname = "paint.context_menu"
    label = "笔刷设置"
    description = "在鼠标位置弹出当前笔刷的常用设置和颜色"
    searchable = False

    @classmethod
    def poll(cls, ctx) -> bool:
        return _painting(ctx)

    def invoke(self, ctx, event) -> str:
        from types import SimpleNamespace

        from ..ui.layout import _open_floating_popover

        holder = SimpleNamespace()
        popover = _open_floating_popover(lambda layout, c: draw_brush_popover(layout, c, holder), ctx)
        popover._splender_holder = holder
        return FINISHED
