"""属性编辑器「工具」页里雕刻模式的面板。绘制模式的面板标成只在绘制模式显示。"""
from __future__ import annotations

from ..core import registry


class _SculptPanel(registry.Panel):
    space = "PROPERTIES"
    region = "MAIN"
    category = "TOOL"
    mode = "SCULPT"


@registry.register_panel
class SculptBrushPanel(_SculptPanel):
    idname = "SCULPT_PT_brush"
    label = "雕刻笔刷"
    order = 5
    icon = "sculpt"

    def draw(self, layout, ctx) -> None:
        tools = ctx.app.tool_settings
        brush = tools.sculpt
        name = tools.tool.split(".", 1)[-1].upper()
        from ..sculpt.engine import BRUSHES

        for idname, label, desc in BRUSHES:
            if idname == name:
                layout.label("%s：%s" % (label, desc), role="dim")
        layout.prop(brush, "size")
        layout.prop(brush, "strength", slider=True)
        layout.prop(brush, "smooth_strength", slider=True)
        layout.prop(brush, "hardness", slider=True)
        layout.prop(brush, "spacing", slider=True)
        layout.prop(brush, "pressure_size")
        layout.prop(brush, "pressure_strength")
        layout.prop(brush, "invert")


@registry.register_panel
class SculptSymmetryPanel(_SculptPanel):
    idname = "SCULPT_PT_symmetry"
    label = "对称"
    order = 6
    icon = "symmetry"

    def draw(self, layout, ctx) -> None:
        sym = ctx.app.tool_settings.symmetry
        row = layout.row(align=True)
        row.use_property_split = False
        row.prop(sym, "x", toggle=True)
        row.prop(sym, "y", toggle=True)
        row.prop(sym, "z", toggle=True)


@registry.register_panel
class SculptMeshPanel(_SculptPanel):
    idname = "SCULPT_PT_mesh"
    label = "网格"
    order = 7
    icon = "mesh"

    def draw(self, layout, ctx) -> None:
        engine = ctx.app.engine
        session = engine.sculpt if engine is not None else None
        if session is None:
            layout.label("没有正在雕刻的模型", role="dim")
            return
        row = layout.row(heading="模型")
        row.label(session.obj.name)
        row = layout.row(heading="面数")
        row.label("%s 个三角形 · %s 个顶点" % (format(session.base.triangle_count, ","),
                                          format(session.base.vertex_count, ",")))
        if session.base.corner_uvs is None:
            layout.label("新网格还没有 UV，回到绘制模式后可以自动展开", role="dim")
        col = layout.column()
        col.use_property_split = False
        col.operator("sculpt.subdivide", text="细分一级", icon="grid")
        col.operator("sculpt.mask_clear", text="清除遮罩", icon="square-dashed")
        col.operator("object.mode_set", text="回到绘制模式", icon="paint", mode="PAINT")


@registry.register_panel
class SculptRemeshPanel(_SculptPanel):
    idname = "SCULPT_PT_remesh"
    label = "体素重构"
    order = 8
    icon = "remesh"

    def draw(self, layout, ctx) -> None:
        settings = ctx.app.tool_settings.remesh
        engine = ctx.app.engine
        session = engine.sculpt if engine is not None else None
        layout.prop(settings, "resolution")
        if session is not None:
            estimate = session.remesh_estimate(int(settings.resolution))
            row = layout.row(heading="预计面数")
            row.label("约 %s个三角形" % _count_text(estimate["triangles"]), role="dim")
            row = layout.row(heading="体素大小")
            row.label("%.4g" % estimate["voxel"], role="dim")
        layout.prop(settings, "relax")
        layout.prop(settings, "min_island")
        layout.prop(settings, "close_holes")
        layout.prop(settings, "keep_mask")
        col = layout.column()
        col.use_property_split = False
        col.operator("sculpt.voxel_remesh", text="重构网格", icon="remesh")


def _count_text(count: int) -> str:
    """数量写成中文习惯：「15 万」「1.2 亿」「9,024 」（后面直接接量词）。"""
    if count >= 100_000_000:
        return "%.1f 亿" % (count / 100_000_000)
    if count >= 10_000:
        return "%.0f 万" % (count / 10_000)
    return format(count, ",") + " "


def mark_paint_panels() -> None:
    """「工具」页里原有的绘制面板只在绘制模式显示。"""
    for cls in registry.panels("PROPERTIES", "MAIN", "TOOL"):
        if getattr(cls, "mode", None) is None:
            cls.mode = "PAINT"
