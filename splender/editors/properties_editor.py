"""属性编辑器：笔刷与工具、当前图层、纹理集、工程。左边一列标签，右边是对应的面板。"""
from __future__ import annotations

import os

import shiboken6
from PySide6.QtWidgets import QHBoxLayout, QWidget

from ..core import registry
from ..doc.project import CHANNELS, CURVE_PROPS, SELECTIVE_INKS, level_prop_names
from ..ops.adjust_ops import AUTO_ITEMS, layer_histogram
from ..tools.paint_tools import EFFECT_TOOLS, STROKE_TOOLS
from ..ui import theme
from ..ui.editor import Editor
from ..ui.panels import PanelHost
from ..ui.widgets import ColorPicker, TabStrip

SPACE = "PROPERTIES"
TABS = [("TOOL", "工具", "笔刷和当前工具的设置", "tool.brush"),
        ("OBJECT", "物体", "当前物体的名字、显示、变换和材质", "mode.object"),
        ("MESH", "数据", "当前物体的数据：模型的面数和 UV，摄像机的镜头，灯光的参数", "object.mesh"),
        ("LAYER", "图层", "当前图层的设置", "editor.layers"),
        ("SET", "纹理集", "这套贴图的设置", "texture_set"),
        ("RENDER", "渲染", "出图的引擎、尺寸和画质", "render"),
        ("PROJECT", "工程", "工程信息", "info")]


class _PickerLink:
    """把嵌在面板里的取色器和一个颜色属性连起来；取色器销毁后自动断开。"""

    def __init__(self, picker: ColorPicker, group, name: str) -> None:
        self.picker = picker
        self.group = group
        self.name = name
        self._busy = False
        picker.setColor(getattr(group, name))
        picker.colorChanged.connect(self.from_picker)
        group.changed.connect(self.from_data)
        picker.destroyed.connect(self.detach)

    def from_picker(self, color) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            setattr(self.group, self.name, tuple(color)[:3])
        finally:
            self._busy = False

    def from_data(self, name: str) -> None:
        if name != self.name or self._busy:
            return
        if not shiboken6.isValid(self.picker):
            self.detach()
            return
        self._busy = True
        try:
            self.picker.setColor(getattr(self.group, self.name))
        finally:
            self._busy = False

    def detach(self, *_args) -> None:
        self.group.changed.disconnect(self.from_data)


def _brush(ctx):
    return ctx.app.tool_settings.active_brush()


def _layer(ctx):
    return ctx.layer


class _Panel(registry.Panel):
    space = SPACE
    region = "MAIN"


# ====================================================================== 工具
def _stroke_tool(ctx) -> bool:
    """当前工具是用笔刷画的（画笔、橡皮、雕刻笔刷）。"""
    tool = ctx.app.tool_settings.tool
    return tool in STROKE_TOOLS or tool.startswith("sculpt.")


@registry.register_panel
class GradientPanel(_Panel):
    idname = "PROPERTIES_PT_gradient"
    label = "渐变"
    category = "TOOL"
    order = 12
    icon = "tool.gradient"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.app.tool_settings.tool == "paint.gradient"

    def draw(self, layout, ctx) -> None:
        settings = ctx.app.tool_settings.gradient
        layout.prop(settings, "shape")
        layout.prop(settings, "source", expand=True)
        if settings.source == "RAMP":
            layout.prop(settings, "ramp", text="")
        layout.prop(settings, "reverse")
        layout.prop(settings, "opacity", slider=True)


@registry.register_panel
class EffectPanel(_Panel):
    idname = "PROPERTIES_PT_effect"
    label = "效果"
    category = "TOOL"
    order = 11
    icon = "adjust"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.app.tool_settings.tool in EFFECT_TOOLS

    def draw(self, layout, ctx) -> None:
        tool = ctx.app.tool_settings.tool
        settings = ctx.app.tool_settings.effect
        layout.prop(settings, "strength", slider=True)
        if tool in ("paint.dodge", "paint.burn"):
            layout.prop(settings, "tone_range", expand=True)
            layout.prop(settings, "protect_tones")
        elif tool == "paint.sponge":
            layout.prop(settings, "sponge_mode", expand=True)
            layout.prop(settings, "vibrance")
        elif tool == "paint.blur":
            layout.prop(settings, "blur_radius")
        elif tool == "paint.sharpen":
            layout.prop(settings, "sharpen_amount")
        elif tool in ("paint.clone", "paint.heal"):
            layout.prop(settings, "clone_aligned")
            if tool == "paint.heal":
                layout.prop(settings, "blur_radius", text="融合范围")
            text = ("来源 U %.3f · V %.3f" % (settings.source_u, settings.source_v)) if settings.has_source \
                else "还没有来源：按住 Alt 点一下"
            layout.label(text, role="dim")


@registry.register_panel
class SelectionPanel(_Panel):
    idname = "PROPERTIES_PT_selection"
    label = "选区"
    category = "TOOL"
    order = 12
    icon = "tool.select_rect"

    @classmethod
    def poll(cls, ctx) -> bool:
        from ..tools.paint_tools import SELECT_TOOLS

        return ctx.app.tool_settings.tool in SELECT_TOOLS

    def draw(self, layout, ctx) -> None:
        tools = ctx.app.tool_settings
        settings = tools.selection
        layout.prop(settings, "mode", expand=True)
        if tools.tool == "paint.select_quick":
            layout.prop(settings, "quick_size")
        if tools.tool in ("paint.select_wand", "paint.select_quick"):
            layout.prop(settings, "tolerance", slider=True)
            if tools.tool == "paint.select_wand":
                layout.prop(settings, "contiguous")
            layout.prop(settings, "sample_all")
        else:
            layout.prop(settings, "feather")
            layout.prop(settings, "occlude")
        layout.prop(settings, "antialias")
        layout.separator()
        row = layout.row(align=True)
        row.operator("select.all")
        row.operator("select.none")
        row.operator("select.invert")


@registry.register_panel
class TextPanel(_Panel):
    idname = "PROPERTIES_PT_text"
    label = "文字"
    category = "TOOL"
    order = 12
    icon = "tool.text"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.app.tool_settings.tool == "paint.text"

    def draw(self, layout, ctx) -> None:
        settings = ctx.app.tool_settings.text
        layout.prop(settings, "font")
        layout.prop(settings, "size")
        row = layout.row(align=True, heading="样式")
        row.prop(settings, "bold", toggle=True)
        row.prop(settings, "italic", toggle=True)
        layout.prop(settings, "align", expand=True)
        layout.prop(settings, "line_spacing")
        layout.prop(settings, "rotation")
        layout.prop(settings, "opacity", slider=True)
        layout.prop(settings, "occlude")


@registry.register_panel
class ShapePanel(_Panel):
    idname = "PROPERTIES_PT_shape"
    label = "形状"
    category = "TOOL"
    order = 12
    icon = "tool.shape"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.app.tool_settings.tool == "paint.shape"

    def draw(self, layout, ctx) -> None:
        settings = ctx.app.tool_settings.shape
        layout.prop(settings, "kind")
        layout.prop(settings, "fill")
        layout.prop(settings, "stroke")
        if settings.kind == "ROUNDED":
            layout.prop(settings, "radius")
        if settings.kind == "POLYGON":
            layout.prop(settings, "sides")
        layout.prop(settings, "opacity", slider=True)
        layout.prop(settings, "occlude")


@registry.register_panel
class FillPanel(_Panel):
    idname = "PROPERTIES_PT_fill"
    label = "油漆桶"
    category = "TOOL"
    order = 12
    icon = "tool.fill"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.app.tool_settings.tool == "paint.fill"

    def draw(self, layout, ctx) -> None:
        settings = ctx.app.tool_settings.fill
        layout.prop(settings, "mode")
        col = layout.column()
        col.enabled = settings.mode == "SIMILAR"
        col.prop(settings, "tolerance", slider=True)
        col.prop(settings, "contiguous")
        col.prop(settings, "sample_all")
        layout.prop(settings, "opacity", slider=True)


@registry.register_panel
class BrushPanel(_Panel):
    idname = "PROPERTIES_PT_brush"
    label = "笔刷"
    category = "TOOL"
    order = 10
    icon = "tool.brush"

    @classmethod
    def poll(cls, ctx) -> bool:
        return _stroke_tool(ctx)

    def draw(self, layout, ctx) -> None:
        brush = _brush(ctx)
        layout.prop(brush, "size_unit", expand=True)
        if brush.size_unit == "SCENE":
            layout.prop(brush, "scene_size")
        else:
            layout.prop(brush, "size")
        layout.prop(brush, "flow", slider=True)
        layout.prop(brush, "opacity", slider=True)
        layout.prop(brush, "hardness", slider=True)
        layout.prop(brush, "spacing", slider=True)


@registry.register_panel
class BrushMaterialPanel(_Panel):
    idname = "PROPERTIES_PT_brush_material"
    label = "颜色与材质"
    category = "TOOL"
    order = 20
    icon = "palette"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.app.tool_settings.tool != "paint.eraser" and not ctx.app.tool_settings.tool.startswith("sculpt.")

    def draw(self, layout, ctx) -> None:
        brush = ctx.app.tool_settings.brush
        ts = ctx.texture_set
        if ts is not None and ts.paint_target == "MASK" and ctx.layer is not None and ctx.layer.has_mask:
            layout.label("正在画蒙版：1 显示，0 隐藏", role="dim")
            layout.prop(brush, "mask_value", slider=True)
            return
        row = layout.row(align=True)
        row.use_property_split = False
        for channel, _label, _plane in CHANNELS:
            row.prop(brush, "use_" + channel, toggle=True)
        picker = ColorPicker()
        self._link = _PickerLink(picker, brush, "color")
        box = layout.column()
        box.use_property_split = False
        box.widget(picker)
        box.enabled = brush.use_basecolor
        colors = layout.row(align=True)
        colors.prop(brush, "secondary_color", text="备用颜色")
        colors.operator("paint.swap_colors", text="", icon="swap")
        col = layout.column()
        col.enabled = brush.use_metallic
        col.prop(brush, "metallic", slider=True)
        col = layout.column()
        col.enabled = brush.use_roughness
        col.prop(brush, "roughness", slider=True)
        col = layout.column()
        col.enabled = brush.use_height
        col.prop(brush, "height")


@registry.register_panel
class BrushDynamicsPanel(_Panel):
    idname = "PROPERTIES_PT_brush_dynamics"
    label = "压感与稳定"
    category = "TOOL"
    order = 30
    icon = "pressure"

    @classmethod
    def poll(cls, ctx) -> bool:
        return _stroke_tool(ctx)

    def draw(self, layout, ctx) -> None:
        brush = _brush(ctx)
        layout.prop(brush, "pressure_size")
        layout.prop(brush, "pressure_flow")
        layout.prop(ctx.prefs.paint, "pressure_curve")
        layout.prop(brush, "smooth", slider=True)


@registry.register_panel
class SymmetryPanel(_Panel):
    idname = "PROPERTIES_PT_symmetry"
    label = "对称"
    category = "TOOL"
    order = 40
    icon = "symmetry"

    @classmethod
    def poll(cls, ctx) -> bool:
        return _stroke_tool(ctx)

    def draw(self, layout, ctx) -> None:
        sym = ctx.app.tool_settings.symmetry
        row = layout.row(align=True, heading="镜像轴")
        row.prop(sym, "x", toggle=True)
        row.prop(sym, "y", toggle=True)
        row.prop(sym, "z", toggle=True)


@registry.register_panel
class BrushSurfacePanel(_Panel):
    idname = "PROPERTIES_PT_brush_surface"
    label = "表面"
    category = "TOOL"
    order = 50
    default_closed = True

    @classmethod
    def poll(cls, ctx) -> bool:
        return _stroke_tool(ctx)

    def draw(self, layout, ctx) -> None:
        brush = _brush(ctx)
        layout.prop(brush, "backface_cull")
        col = layout.column()
        col.enabled = brush.backface_cull
        col.prop(brush, "normal_falloff")


# ====================================================================== 图层
class _LayerPanel(_Panel):
    category = "LAYER"

    @classmethod
    def poll(cls, ctx) -> bool:
        return _layer(ctx) is not None


@registry.register_panel
class LayerPanel(_LayerPanel):
    idname = "PROPERTIES_PT_layer"
    label = "图层"
    order = 10
    icon = "editor.layers"

    def draw(self, layout, ctx) -> None:
        layer = _layer(ctx)
        layout.prop(layer, "name")
        layout.prop(layer, "opacity", slider=True)
        layout.prop(layer, "visible")
        layout.prop(layer, "locked")


@registry.register_panel
class LayerBlendPanel(_LayerPanel):
    idname = "PROPERTIES_PT_layer_blend"
    label = "通道混合"
    order = 20
    icon = "channel"

    @classmethod
    def poll(cls, ctx) -> bool:
        layer = _layer(ctx)
        return layer is not None and layer.kind != "ADJUST"

    def draw(self, layout, ctx) -> None:
        layer = _layer(ctx)
        for channel, label, _plane in CHANNELS:
            row = layout.row(align=True, heading=label)
            row.prop(layer, "blend_" + channel, text="")
            row.prop(layer, "opacity_" + channel, text="", slider=True)


@registry.register_panel
class LayerFillPanel(_LayerPanel):
    idname = "PROPERTIES_PT_layer_fill"
    label = "填充内容"
    order = 30
    icon = "layer.fill"

    @classmethod
    def poll(cls, ctx) -> bool:
        layer = _layer(ctx)
        return layer is not None and layer.kind == "FILL"

    def draw(self, layout, ctx) -> None:
        layer = _layer(ctx)
        row = layout.row(align=True)
        row.use_property_split = False
        for channel, _label, _plane in CHANNELS:
            row.prop(layer, "use_" + channel, toggle=True)
        project = ctx.project
        graph = project.graph(layer.fill_graph) if (project is not None and layer.fill_graph) else None
        row = layout.row(heading="来源")
        row.menu("LAYERS_MT_fill_graph", text=("节点图「%s」" % graph.name) if graph is not None else "固定数值",
                 icon="editor.nodes" if graph is not None else "layer.fill")
        if graph is not None:
            layout.prop(layer, "graph_tiling")
            layout.prop(layer, "graph_offset_u", slider=True)
            layout.prop(layer, "graph_offset_v", slider=True)
            col = layout.column()
            col.enabled = layer.use_height
            col.prop(layer, "graph_height")
            if graph.output_node() is None:
                layout.label("这张节点图没有「材质输出」节点", icon="warning")
            col = layout.column()
            col.use_property_split = False
            col.operator("node.graph_edit", text="编辑节点图", icon="editor.nodes", graph=graph.uid)
            return
        col = layout.column()
        col.enabled = layer.use_basecolor
        col.prop(layer, "fill_color")
        col = layout.column()
        col.enabled = layer.use_metallic
        col.prop(layer, "fill_metallic", slider=True)
        col = layout.column()
        col.enabled = layer.use_roughness
        col.prop(layer, "fill_roughness", slider=True)
        col = layout.column()
        col.enabled = layer.use_height
        col.prop(layer, "fill_height")


def _auto_buttons(layout) -> None:
    row = layout.row(align=True)
    for mode, label, _desc in AUTO_ITEMS:
        row.operator("layer.adjust_auto", text=label, mode=mode)


@registry.register_panel
class LayerAdjustPanel(_LayerPanel):
    idname = "PROPERTIES_PT_layer_adjust"
    label = "调整"
    order = 30
    icon = "adjust"

    @classmethod
    def poll(cls, ctx) -> bool:
        layer = _layer(ctx)
        return layer is not None and layer.kind == "ADJUST"

    def draw_header(self, layout, ctx) -> None:
        button = layout.menu("LAYERS_MT_adjust_presets", text="", icon="list")
        button.setToolTip("预设")

    def draw(self, layout, ctx) -> None:
        layer = _layer(ctx)
        layout.prop(layer, "adjust_type")
        kind = layer.adjust_type
        if kind == "HSV":
            layout.prop(layer, "adj_colorize")
            if layer.adj_colorize:
                layout.prop(layer, "adj_colorize_hue", slider=True)
                layout.prop(layer, "adj_colorize_sat", slider=True)
                layout.prop(layer, "adj_value", slider=True)
            else:
                layout.prop(layer, "adj_hue_edit")
                if layer.adj_hue_edit == "MASTER":
                    layout.prop(layer, "adj_hue", slider=True)
                    layout.prop(layer, "adj_saturation", slider=True)
                    layout.prop(layer, "adj_value", slider=True)
                else:
                    prefix = "adj_hsv_%s_" % layer.adj_hue_edit.lower()
                    for field in ("hue", "sat", "light"):
                        layout.prop(layer, prefix + field, slider=True)
        elif kind == "CURVES":
            layout.prop(layer, "adj_curve_channel", expand=True)
            channel = layer.adj_curve_channel
            widget = layout.prop(layer, CURVE_PROPS.get(channel, "adj_curve"), text="")
            counts = layer_histogram(ctx, layer)
            if counts is not None:
                widget.counts = counts["RGB".index(channel) if channel != "RGB" else 3]
            _auto_buttons(layout)
        elif kind == "GRADIENT_MAP":
            layout.prop(layer, "adj_ramp", text="")
            layout.prop(layer, "adj_ramp_reverse")
        elif kind == "COLOR_BALANCE":
            layout.prop(layer, "adj_cb_tone", expand=True)
            tone = layer.adj_cb_tone.lower()
            for channel in "rgb":
                layout.prop(layer, "adj_cb_%s_%s" % (tone, channel), slider=True)
            layout.prop(layer, "adj_cb_preserve")
        elif kind == "POSTERIZE":
            layout.prop(layer, "adj_posterize")
        elif kind == "VIBRANCE":
            layout.prop(layer, "adj_vibrance", slider=True)
            layout.prop(layer, "adj_vib_saturation", slider=True)
        elif kind == "PHOTO_FILTER":
            layout.prop(layer, "adj_filter_color")
            layout.prop(layer, "adj_filter_density", slider=True)
            layout.prop(layer, "adj_filter_preserve")
        elif kind == "EXPOSURE":
            layout.prop(layer, "adj_exposure")
            layout.prop(layer, "adj_exp_offset")
            layout.prop(layer, "adj_exp_gamma")
        elif kind == "BLACK_WHITE":
            for name in ("red", "yellow", "green", "cyan", "blue", "magenta"):
                layout.prop(layer, "adj_bw_" + name, slider=True)
            layout.prop(layer, "adj_bw_tint")
            col = layout.column()
            col.enabled = layer.adj_bw_tint
            col.prop(layer, "adj_bw_tint_color")
        elif kind == "CHANNEL_MIXER":
            layout.prop(layer, "adj_mix_mono")
            if layer.adj_mix_mono:
                layout.label("灰色输出", role="dim")
                out = "r"
            else:
                layout.prop(layer, "adj_mix_output", expand=True)
                out = layer.adj_mix_output.lower()
            for source in "rgbc":
                layout.prop(layer, "adj_mix_%s%s" % (out, source), slider=True)
            total = sum(getattr(layer, "adj_mix_%s%s" % (out, source)) for source in "rgb")
            layout.label("总计 %+d%%" % round(total * 100), icon="warning" if total > 1.0 + 1e-6 else None,
                         role="dim")
        elif kind == "COLOR_LOOKUP":
            layout.prop(layer, "adj_lut_look")
            if layer.adj_lut_look == "FILE":
                layout.label(layer.adj_lut_name or "还没有导入文件", role="dim")
            row = layout.row(align=True)
            row.operator("layer.adjust_lut_import", text="导入 .cube…", icon="import")
            row.operator("layer.adjust_lut_export", text="导出 .cube…", icon="export")
        elif kind == "SELECTIVE_COLOR":
            layout.prop(layer, "adj_sel_range")
            key = layer.adj_sel_range.lower()
            for ink, _label, _desc in SELECTIVE_INKS:
                layout.prop(layer, "adj_sel_%s_%s" % (key, ink), slider=True)
            layout.prop(layer, "adj_sel_method", expand=True)
        elif kind == "BRIGHTNESS":
            layout.prop(layer, "adj_brightness", slider=True)
            layout.prop(layer, "adj_contrast", slider=True)
        elif kind == "LEVELS":
            from ..ui.levels_widget import LevelsWidget

            layout.prop(layer, "adj_levels_channel", expand=True)
            channel = layer.adj_levels_channel
            names = level_prop_names(channel)
            counts = layer_histogram(ctx, layer)
            levels = LevelsWidget(layer, names, counts[3 if channel == "RGB" else "RGB".index(channel)]
                                  if counts is not None else None, channel, getattr(ctx.app, "history", None))
            layout.widget(levels)
            for name in names:
                layout.prop(layer, name, slider=not name.endswith("gamma"))
            _auto_buttons(layout)
        elif kind == "THRESHOLD":
            layout.prop(layer, "adj_threshold", slider=True)
        row = layout.row(align=True, heading="作用于")
        for channel, _label, _plane in CHANNELS:
            row.prop(layer, "use_" + channel, toggle=True)
        layout.label("调整的是它下面（同一个文件夹里）已经合好的结果", role="dim")


@registry.register_panel
class LayerMaskPanel(_LayerPanel):
    idname = "PROPERTIES_PT_layer_mask"
    label = "蒙版"
    order = 40
    icon = "layer.mask"

    def draw(self, layout, ctx) -> None:
        layer = _layer(ctx)
        ts = ctx.texture_set
        if not layer.has_mask:
            row = layout.row(align=True)
            row.use_property_split = False
            row.operator("layer.mask_add", text="白色蒙版", icon="add", fill="WHITE")
            row.operator("layer.mask_add", text="黑色蒙版", icon="add", fill="BLACK")
            return
        layout.prop(layer, "mask_enabled")
        layout.prop(ts, "paint_target", expand=True)
        layout.operator("layer.mask_remove", icon="remove")


@registry.register_panel
class LayerGeneratorPanel(_LayerPanel):
    idname = "PROPERTIES_PT_layer_generator"
    label = "生成器"
    order = 45
    icon = "generator"

    def draw(self, layout, ctx) -> None:
        layer = _layer(ctx)
        row = layout.row()
        row.use_property_split = False
        row.menu("LAYERS_MT_smart_mask", text="智能遮罩", icon="generator")
        layout.prop(layer, "mask_generator")
        if layer.mask_generator == "NONE":
            layout.label("按模型的边缘、凹陷、朝向等自动算出蒙版，和画的蒙版相乘", role="dim")
            return
        if layer.mask_generator == "PARTS":
            self._draw_parts(layout, ctx, layer)
            return
        if layer.mask_generator in ("FACING", "GRADIENT"):
            layout.prop(layer, "gen_direction")
        layout.prop(layer, "gen_range", slider=True)
        layout.prop(layer, "gen_softness", slider=True)
        layout.prop(layer, "gen_noise", slider=True)
        col = layout.column()
        col.enabled = layer.gen_noise > 0.0
        col.prop(layer, "gen_noise_size")
        col.prop(layer, "gen_seed")
        layout.prop(layer, "gen_invert")
        self._draw_bake_hint(layout, ctx)

    def _draw_parts(self, layout, ctx, layer) -> None:
        from ..bake.parts import parse_ids

        engine = ctx.app.engine
        ts = ctx.texture_set
        maps = engine.meshmap(ts.uid) if engine is not None and ts is not None else None
        chosen = len(parse_ids(layer.gen_parts))
        row = layout.row(heading="选中")
        if maps is not None and maps.part_count:
            row.label("%d 个部件（共 %d 个）" % (chosen, maps.part_count), role="dim")
        else:
            row.label("%d 个部件" % chosen, role="dim")
        col = layout.column()
        col.use_property_split = False
        col.operator("layer.pick_parts", text="在视口里点选部件", icon="generator", role="primary")
        col.operator("layer.clear_parts", text="清空", icon="remove")
        layout.prop(layer, "gen_invert")
        if ts is not None:
            layout.prop(ts.meshmap, "id_source")
        if maps is not None and not maps.part_count and engine.bake is None:
            layout.label("这份模型贴图还没分部件，重新烘焙一次", icon="warning")
            col = layout.column()
            col.use_property_split = False
            col.operator("bake.mesh_maps", text="重新烘焙", icon="bake")
            return
        self._draw_bake_hint(layout, ctx)

    def _draw_bake_hint(self, layout, ctx) -> None:
        engine = ctx.app.engine
        ts = ctx.texture_set
        if engine is not None and ts is not None and engine.meshmap(ts.uid) is None:
            if engine.bake is not None:
                layout.label("正在烘焙模型贴图，好了就会显示", role="dim")
            else:
                layout.label("要先烘焙模型贴图才有效果", icon="warning")
                col = layout.column()
                col.use_property_split = False
                col.operator("bake.mesh_maps", text="烘焙模型贴图", icon="bake")


# ====================================================================== 纹理集
@registry.register_panel
class TextureSetPanel(_Panel):
    idname = "PROPERTIES_PT_texture_set"
    label = "纹理集"
    category = "SET"
    order = 10
    icon = "texture_set"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.texture_set is not None

    def draw(self, layout, ctx) -> None:
        ts = ctx.texture_set
        layout.prop(ts, "name")
        row = layout.row(heading="分辨率")
        row.menu("TEXTURE_SET_MT_resolution", text="%d × %d" % (ts.size, ts.size))
        layout.prop(ts, "base_color")
        layout.prop(ts, "base_metallic", slider=True)
        layout.prop(ts, "base_roughness", slider=True)
        layout.prop(ts, "height_scale")


class MapStrip(QWidget):
    """模型贴图的缩略图，每行四张。"""

    KINDS = (("ao", "遮蔽"), ("curvature", "曲率"), ("thickness", "厚度"), ("normal", "法线"), ("position", "位置"),
             ("tangent", "法线贴图"), ("parts", "部件"))
    PER_ROW = 4

    def __init__(self, maps) -> None:
        super().__init__()
        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import QGridLayout, QLabel

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(theme.px(4))
        grid.setVerticalSpacing(theme.px(2))
        side = theme.px(52)
        self._images = []
        for column, (kind, label) in enumerate(self.KINDS):
            try:
                data = maps.preview(kind, 96)
            except Exception:  # noqa: BLE001
                continue
            image = QImage(data.tobytes(), data.shape[1], data.shape[0], data.shape[1] * 3, QImage.Format_RGB888)
            self._images.append(data)
            picture = QLabel()
            picture.setPixmap(QPixmap.fromImage(image).scaled(side, side, _Qt.KeepAspectRatio,
                                                              _Qt.SmoothTransformation))
            picture.setAlignment(_Qt.AlignCenter)
            caption = QLabel(label)
            caption.setProperty("role", "dim")
            caption.setAlignment(_Qt.AlignCenter)
            row = (column // self.PER_ROW) * 2
            grid.addWidget(picture, row, column % self.PER_ROW)
            grid.addWidget(caption, row + 1, column % self.PER_ROW)


@registry.register_panel
class MeshMapsPanel(_Panel):
    idname = "PROPERTIES_PT_meshmaps"
    label = "模型贴图"
    category = "SET"
    order = 15
    icon = "bake"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.texture_set is not None

    def draw(self, layout, ctx) -> None:
        ts = ctx.texture_set
        settings = ts.meshmap
        engine = ctx.app.engine
        maps = engine.meshmap(ts.uid) if engine is not None else None
        job = engine.bake if engine is not None else None
        if job is not None and job.ts is ts:
            row = layout.row(heading="正在烘焙")
            row.label("%s · %d%%" % (job.status, int(job.progress * 100)), role="dim")
        elif maps is not None:
            layout.widget(MapStrip(maps))
            stats = maps.meta.get("stats", {})
            row = layout.row(heading="大小")
            row.label("%d × %d · 用时 %.1f 秒" % (maps.size, maps.size, float(stats.get("seconds", 0.0))), role="dim")
            if maps.meta.get("stale"):
                layout.label("模型的形状改过，重新烘焙才能对上", icon="warning")
        else:
            layout.label("还没有烘焙。生成器蒙版和智能材质靠它们决定盖住哪里", role="dim")
        layout.prop(settings, "high_poly")
        if settings.high_poly:
            layout.prop(settings, "cage_front", slider=True)
            layout.prop(settings, "cage_back", slider=True)
            layout.prop(settings, "normal_samples")
        layout.prop(settings, "normal_format")
        layout.prop(settings, "id_source")
        layout.prop(settings, "resolution")
        layout.prop(settings, "ao_samples")
        layout.prop(settings, "ao_distance", slider=True)
        layout.prop(settings, "bake_thickness")
        col = layout.column()
        col.enabled = settings.bake_thickness
        col.prop(settings, "thickness_distance", slider=True)
        layout.prop(settings, "curvature_smooth")
        layout.prop(settings, "curvature_contrast")
        layout.prop(settings, "padding")
        layout.prop(settings, "self_only")
        layout.prop(settings, "auto_rebake")
        layout.separator()
        layout.prop(settings, "show_ao")
        col = layout.column()
        col.enabled = settings.show_ao
        col.prop(settings, "ao_strength", slider=True)
        col = layout.column()
        col.use_property_split = False
        if job is not None:
            col.operator("bake.cancel", text="停止烘焙", icon="stop")
        else:
            col.operator("bake.mesh_maps", text="重新烘焙" if maps is not None else "烘焙", icon="bake",
                         role="primary")


@registry.register_panel
class ExportPanel(_Panel):
    idname = "PROPERTIES_PT_export"
    label = "导出"
    category = "SET"
    order = 20
    icon = "export"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.texture_set is not None

    def draw(self, layout, ctx) -> None:
        ts = ctx.texture_set
        settings = ts.export
        layout.label("勾选的通道各导出成一张 PNG", role="dim")
        for names in (("basecolor", "metallic", "roughness"), ("height", "normal", "ao")):
            row = layout.row(align=True)
            row.use_property_split = False
            for name in names:
                row.prop(settings, name, toggle=True)
        layout.prop(settings, "size")
        layout.prop(settings, "name_pattern")
        col = layout.column()
        col.enabled = settings.normal
        col.prop(ts.meshmap, "normal_format")
        col.prop(settings, "normal_bake")
        col.prop(settings, "normal_strength", slider=True)
        col.prop(settings, "normal_depth")
        col = layout.column()
        col.use_property_split = False
        col.operator("wm.export_textures", text="导出贴图…", icon="export", role="primary")


# ====================================================================== 工程
@registry.register_panel
class ProjectPanel(_Panel):
    idname = "PROPERTIES_PT_project"
    label = "工程"
    category = "PROJECT"
    order = 10
    icon = "file"

    def draw(self, layout, ctx) -> None:
        project = ctx.project
        if project is None:
            return
        row = layout.row(heading="名称")
        row.label(project.name)
        row = layout.row(heading="位置")
        row.label(os.path.dirname(project.path) if project.path else "还没有保存", role="dim")
        for obj in project.objects:
            row = layout.row(heading="模型")
            row.label("%s · %s 个三角形" % (obj.name, format(obj.data.triangle_count, ",")))
        row = layout.row(heading="纹理集")
        row.label("%d 套" % len(project.texture_sets))
        col = layout.column()
        col.use_property_split = False
        col.operator("wm.import_mesh", icon="import")
        col.operator("wm.save", icon="save")


# ====================================================================== 模型
@registry.register_panel
class MeshObjectsPanel(_Panel):
    idname = "PROPERTIES_PT_mesh_objects"
    label = "模型"
    category = "MESH"
    order = 1
    icon = "mesh"

    @classmethod
    def poll(cls, ctx) -> bool:
        project = ctx.project
        active = project.active_object if project is not None else None
        return active is None or active.kind == "MESH"

    def draw(self, layout, ctx) -> None:
        project = ctx.project
        if project is None:
            return
        for obj in project.objects:
            data = obj.data
            uv = "有 UV" if getattr(data, "has_uvs", True) else "没有 UV"
            row = layout.row(heading=obj.name)
            row.label("%s 个三角形 · %s" % (format(data.triangle_count, ","), uv),
                      icon=None if getattr(data, "has_uvs", True) else "warning")


@registry.register_panel
class MeshUnwrapPanel(_Panel):
    idname = "PROPERTIES_PT_mesh_unwrap"
    label = "自动展开 UV"
    category = "MESH"
    order = 2
    icon = "uv_unwrap"

    @classmethod
    def poll(cls, ctx) -> bool:
        project = ctx.project
        active = project.active_object if project is not None else None
        return active is None or active.kind == "MESH"

    def draw(self, layout, ctx) -> None:
        settings = ctx.app.tool_settings.unwrap
        layout.prop(settings, "angle_limit", slider=True)
        layout.prop(settings, "margin")
        layout.prop(settings, "min_chart")
        layout.prop(settings, "rotate")
        layout.prop(settings, "transfer")
        col = layout.column()
        col.use_property_split = False
        missing = ctx.project is not None and any(not getattr(o.data, "has_uvs", True) for o in ctx.project.objects)
        if missing:
            col.operator("mesh.uv_unwrap", text="展开没有 UV 的模型", icon="uv_unwrap", only_missing=True)
        col.operator("mesh.uv_unwrap", text="全部重新展开", icon="uv_unwrap", only_missing=False)


@registry.register_editor
class PropertiesEditor(Editor):
    idname = SPACE
    label = "属性"
    icon = "editor.properties"
    category = "通用"
    order = 30
    description = "笔刷、图层、纹理集和工程的设置"

    def __init__(self, app, area=None) -> None:
        self._tab = "TOOL"
        self._pending_panels = None
        super().__init__(app, area)

    def build_main(self):
        root = QWidget()
        box = QHBoxLayout(root)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.tabs = TabStrip(TABS, vertical=True, icon_only=True)
        self.tabs.setCurrent(self._tab)
        self.tabs.currentChanged.connect(self._on_tab)
        box.addWidget(self.tabs)
        self.panels = PanelHost(self.context, SPACE, "MAIN", self._tab)
        box.addWidget(self.panels, 1)
        return root

    def draw_header(self, layout, ctx) -> None:
        label = next((item[1] for item in TABS if item[0] == self._tab), "")
        layout.label(label, role="dim")

    def _on_tab(self, ident) -> None:
        self._tab = ident
        self.panels.set_category(ident)
        self.refresh_header()

    def refresh(self) -> None:
        super().refresh()
        self.panels.refresh()

    def save_state(self) -> dict:
        state = super().save_state()
        state["tab"] = self._tab
        state["panels"] = self.panels.state()
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        state = state or {}
        tab = state.get("tab", self._tab)
        if tab in [item[0] for item in TABS]:
            self._tab = tab
            self.tabs.setCurrent(tab)
            self.panels.set_category(tab)
        self.panels.set_state(state.get("panels"))
