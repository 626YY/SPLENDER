"""属性编辑器的「渲染」页：引擎、输出尺寸、显示变换、引擎自己的设置。"""
from __future__ import annotations

from ..core import registry
from . import engines


class _RenderPanel(registry.Panel):
    space = "PROPERTIES"
    region = "MAIN"
    category = "RENDER"

    @classmethod
    def poll(cls, ctx) -> bool:
        return getattr(ctx, "project", None) is not None


@registry.register_panel
class RenderEnginePanel(_RenderPanel):
    idname = "RENDER_PT_engine"
    label = "渲染引擎"
    order = 10
    icon = "render"

    def draw(self, layout, ctx) -> None:
        props = ctx.project.render
        layout.prop(props, "engine")
        cls = engines.engine(props.engine)
        if cls is not None:
            ok, reason = cls.available()
            if not ok:
                layout.label(reason, icon="warning", role="dim")
            elif cls.description:
                layout.label(cls.description, role="dim")
        col = layout.column()
        col.use_property_split = False
        col.operator("render.render", text="渲染图片", icon="render")
        col.operator("render.view_result", text="查看结果", icon="image")


@registry.register_panel
class RenderOutputPanel(_RenderPanel):
    idname = "RENDER_PT_output"
    label = "输出"
    order = 20
    icon = "image"

    def draw(self, layout, ctx) -> None:
        props = ctx.project.render
        layout.prop(props, "resolution_x")
        layout.prop(props, "resolution_y")
        layout.prop(props, "resolution_percentage", slider=True)
        width, height = props.size
        layout.label("出图 %d × %d" % (width, height), role="dim")
        layout.prop(props, "texture_size")
        layout.prop(props, "transparent")


@registry.register_panel
class RenderColorPanel(_RenderPanel):
    idname = "RENDER_PT_color"
    label = "色彩"
    order = 30
    icon = "contrast"

    def draw(self, layout, ctx) -> None:
        props = ctx.project.render
        layout.prop(props, "view_transform", expand=True)
        layout.prop(props, "exposure")


@registry.register_panel
class RenderEngineSettingsPanel(_RenderPanel):
    idname = "RENDER_PT_engine_settings"
    label = "引擎设置"
    order = 40
    icon = "settings"

    @classmethod
    def poll(cls, ctx) -> bool:
        if getattr(ctx, "project", None) is None:
            return False
        return ctx.project.render.engine_settings() is not None

    def draw(self, layout, ctx) -> None:
        props = ctx.project.render
        cls = engines.engine(props.engine)
        settings = props.engine_settings()
        custom = getattr(cls, "draw_settings", None)
        if custom is not None:
            custom(layout, settings, ctx)
            return
        for name, prop in type(settings).properties().items():
            if not prop.hidden:
                layout.prop(settings, name)
