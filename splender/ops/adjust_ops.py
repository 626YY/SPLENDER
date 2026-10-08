"""调整层的预设（照 Photoshop 调整面板里的预设）：一组数值一键套上，可以撤销。
每种调整的第一项「默认值」把这种调整的参数都恢复成默认。"""
from __future__ import annotations

import logging

from ..core import ops, registry
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import EnumProperty, FloatProperty, IntProperty, StringProperty
from ..engine.adjust import TYPE_PROPS, auto_levels, histogram, levels_to_curve

log = logging.getLogger("splender.ops.adjust")

_IDENTITY = ((0.0, 0.0), (1.0, 1.0))


def _curves(rgb=None, r=None, g=None, b=None) -> dict:
    return {"adj_curve": rgb or _IDENTITY, "adj_curve_r": r or _IDENTITY, "adj_curve_g": g or _IDENTITY,
            "adj_curve_b": b or _IDENTITY}


def _bw(red, yellow, green, cyan, blue, magenta) -> dict:
    """按 Photoshop 的百分数写。"""
    return {"adj_bw_red": red / 100.0, "adj_bw_yellow": yellow / 100.0, "adj_bw_green": green / 100.0,
            "adj_bw_cyan": cyan / 100.0, "adj_bw_blue": blue / 100.0, "adj_bw_magenta": magenta / 100.0}


def _mixer(r=(100, 0, 0, 0), g=(0, 100, 0, 0), b=(0, 0, 100, 0), mono=False) -> dict:
    values = {"adj_mix_mono": mono}
    for out, row in (("r", r), ("g", g), ("b", b)):
        for src, amount in zip("rgbc", row):
            values["adj_mix_%s%s" % (out, src)] = amount / 100.0
    return values


def _filter(r, g, b) -> dict:
    return {"adj_filter_color": (r / 255.0, g / 255.0, b / 255.0)}


def _gradient(*stops, mode="LINEAR") -> dict:
    return {"adj_ramp": (mode, tuple((float(pos), r / 255.0, g / 255.0, b / 255.0, 1.0) for pos, (r, g, b) in stops))}


def _brush_gradient(ctx) -> dict:
    brush = ctx.app.tool_settings.brush
    first, second = tuple(brush.color)[:3], tuple(brush.secondary_color)[:3]
    return {"adj_ramp": ("LINEAR", ((0.0, *first, 1.0), (1.0, *second, 1.0)))}


PRESETS = {
    "CURVES": [
        ("中对比度", _curves(((0.0, 0.0), (0.25, 0.2), (0.75, 0.8), (1.0, 1.0)))),
        ("强对比度", _curves(((0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)))),
        ("线性对比度", _curves(((0.0, 0.0), (0.25, 0.22), (0.75, 0.78), (1.0, 1.0)))),
        ("较亮", _curves(((0.0, 0.0), (0.5, 0.62), (1.0, 1.0)))),
        ("较暗", _curves(((0.0, 0.0), (0.5, 0.38), (1.0, 1.0)))),
        ("提亮暗部", _curves(((0.0, 0.0), (0.25, 0.35), (0.6, 0.66), (1.0, 1.0)))),
        ("褪色", _curves(((0.0, 0.1), (0.5, 0.5), (1.0, 0.92)))),
        ("负片", _curves(((0.0, 1.0), (1.0, 0.0)))),
        ("反冲", _curves(r=((0.0, 0.0), (0.25, 0.17), (0.75, 0.87), (1.0, 1.0)),
                       g=((0.0, 0.0), (0.25, 0.2), (0.75, 0.84), (1.0, 1.0)),
                       b=((0.0, 0.12), (1.0, 0.88)))),
        ("暖调", _curves(r=((0.0, 0.0), (0.5, 0.56), (1.0, 1.0)), b=((0.0, 0.0), (0.5, 0.44), (1.0, 1.0)))),
        ("冷调", _curves(r=((0.0, 0.0), (0.5, 0.45), (1.0, 1.0)), b=((0.0, 0.0), (0.5, 0.56), (1.0, 1.0)))),
    ],
    "LEVELS": [
        ("增加对比度", {"adj_in_black": 0.06, "adj_in_white": 0.94}),
        ("强对比度", {"adj_in_black": 0.12, "adj_in_white": 0.88}),
        ("较亮", {"adj_gamma": 1.3}),
        ("较暗", {"adj_gamma": 0.77}),
        ("压低反差", {"adj_out_black": 0.08, "adj_out_white": 0.92}),
    ],
    "EXPOSURE": [
        ("减 2.0", {"adj_exposure": -2.0}),
        ("减 1.0", {"adj_exposure": -1.0}),
        ("加 1.0", {"adj_exposure": 1.0}),
        ("加 2.0", {"adj_exposure": 2.0}),
    ],
    "VIBRANCE": [
        ("轻微", {"adj_vibrance": 0.25}),
        ("明显", {"adj_vibrance": 0.6}),
        ("压低", {"adj_vibrance": -0.5}),
    ],
    "HSV": [
        ("增加饱和度", {"adj_saturation": 0.2}),
        ("进一步增加饱和度", {"adj_saturation": 0.4}),
        ("强饱和度", {"adj_saturation": 0.7}),
        ("降低饱和度", {"adj_saturation": -0.5}),
        ("旧样式", {"adj_saturation": -0.6, "adj_value": -0.05}),
        ("深褐", {"adj_colorize": True, "adj_colorize_hue": 35.0, "adj_colorize_sat": 0.25}),
        ("氰版照相", {"adj_colorize": True, "adj_colorize_hue": 205.0, "adj_colorize_sat": 0.25}),
    ],
    "COLOR_BALANCE": [
        ("暖调", {"adj_cb_midtones_r": 0.15, "adj_cb_midtones_b": -0.15, "adj_cb_highlights_r": 0.05}),
        ("冷调", {"adj_cb_midtones_r": -0.12, "adj_cb_midtones_b": 0.15}),
        ("青橙", {"adj_cb_shadows_r": -0.15, "adj_cb_shadows_b": 0.15, "adj_cb_highlights_r": 0.15,
                "adj_cb_highlights_b": -0.15}),
        ("复古", {"adj_cb_shadows_b": 0.1, "adj_cb_midtones_r": 0.1, "adj_cb_midtones_g": 0.05,
                "adj_cb_highlights_b": -0.2}),
    ],
    "BLACK_WHITE": [
        ("红色滤镜", _bw(120, 110, -10, -50, -50, 120)),
        ("黄色滤镜", _bw(120, 110, 40, -30, 0, 70)),
        ("绿色滤镜", _bw(40, 100, 100, 40, 0, 40)),
        ("蓝色滤镜", _bw(0, 0, 0, 110, 110, 110)),
        ("高对比度红色滤镜", _bw(150, 140, -30, -70, -70, 150)),
        ("高对比度蓝色滤镜", _bw(-50, -50, -10, 150, 150, 150)),
        ("红外线", _bw(-40, 235, 144, -68, -3, -107)),
        ("较亮", _bw(60, 80, 60, 80, 40, 100)),
        ("较暗", _bw(20, 40, 20, 40, 0, 60)),
        ("最白", _bw(100, 100, 100, 100, 100, 100)),
        ("最黑", _bw(0, 0, 0, 0, 0, 0)),
        ("中灰密度", _bw(128, 128, 100, 100, 128, 100)),
    ],
    "PHOTO_FILTER": [
        ("加温滤镜 (85)", _filter(236, 138, 0)),
        ("加温滤镜 (LBA)", _filter(250, 150, 0)),
        ("加温滤镜 (81)", _filter(235, 177, 19)),
        ("冷却滤镜 (80)", _filter(0, 109, 255)),
        ("冷却滤镜 (LBB)", _filter(0, 93, 255)),
        ("冷却滤镜 (82)", _filter(0, 181, 255)),
        ("红", _filter(234, 26, 26)),
        ("橙", _filter(243, 132, 23)),
        ("黄", _filter(249, 227, 28)),
        ("绿", _filter(25, 201, 25)),
        ("青", _filter(29, 203, 234)),
        ("蓝", _filter(29, 53, 234)),
        ("紫", _filter(155, 29, 234)),
        ("洋红", _filter(227, 24, 227)),
        ("深褐", _filter(172, 122, 51)),
        ("深红", _filter(255, 0, 0)),
        ("深蓝", _filter(0, 34, 205)),
        ("深祖母绿", _filter(0, 140, 0)),
        ("深黄", _filter(255, 213, 0)),
        ("水下", _filter(0, 194, 177)),
    ],
    "CHANNEL_MIXER": [
        ("黑白（红色滤镜）", _mixer(r=(100, 0, 0, 0), mono=True)),
        ("黑白（橙色滤镜）", _mixer(r=(50, 50, 0, 0), mono=True)),
        ("黑白（黄色滤镜）", _mixer(r=(34, 66, 0, 0), mono=True)),
        ("黑白（绿色滤镜）", _mixer(r=(10, 80, 10, 0), mono=True)),
        ("黑白（蓝色滤镜）", _mixer(r=(0, 0, 100, 0), mono=True)),
        ("黑白（按亮度）", _mixer(r=(21, 72, 7, 0), mono=True)),
        ("黑白红外线", _mixer(r=(-70, 200, -30, 0), mono=True)),
        ("红蓝互换", _mixer(r=(0, 0, 100, 0), b=(100, 0, 0, 0))),
        ("红绿互换", _mixer(r=(0, 100, 0, 0), g=(100, 0, 0, 0))),
    ],
    "POSTERIZE": [
        ("2 档", {"adj_posterize": 2}),
        ("3 档", {"adj_posterize": 3}),
        ("8 档", {"adj_posterize": 8}),
    ],
    "GRADIENT_MAP": [
        ("笔刷颜色到备用颜色", _brush_gradient),
        ("黑到白", _gradient((0.0, (0, 0, 0)), (1.0, (255, 255, 255)))),
        ("紫到橙", _gradient((0.0, (41, 10, 89)), (1.0, (255, 124, 0)))),
        ("蓝红黄", _gradient((0.0, (10, 0, 178)), (0.5, (255, 0, 0)), (1.0, (255, 252, 0)))),
        ("铜色", _gradient((0.0, (151, 70, 26)), (0.3, (251, 216, 197)), (0.83, (108, 46, 22)),
                         (1.0, (239, 219, 205)))),
        ("色谱", _gradient((0.0, (255, 0, 0)), (0.17, (255, 255, 0)), (0.33, (0, 255, 0)), (0.5, (0, 255, 255)),
                         (0.67, (0, 0, 255)), (0.83, (255, 0, 255)), (1.0, (255, 0, 0)))),
        ("暖色双色调", _gradient((0.0, (38, 20, 8)), (1.0, (255, 236, 204)))),
        ("冷色双色调", _gradient((0.0, (8, 18, 40)), (1.0, (214, 236, 255)))),
        ("热成像", _gradient((0.0, (0, 0, 0)), (0.3, (110, 0, 140)), (0.55, (230, 40, 0)), (0.8, (255, 200, 0)),
                          (1.0, (255, 255, 255)))),
    ],
}


def presets(kind: str) -> list:
    """这种调整的预设：[(名字, 数值字典或 None 或 函数(ctx))]，第一项是「默认值」（None）。"""
    return [("默认值", None)] + list(PRESETS.get(kind, []))


def preset_values(layer, kind: str, index: int, ctx=None) -> tuple[str, dict] | None:
    items = presets(kind)
    if not 0 <= index < len(items):
        return None
    name, values = items[index]
    if values is None:
        values = {prop: layer.prop(prop).default_value(layer) for prop in TYPE_PROPS.get(kind, ())}
    elif callable(values):
        values = values(ctx)
    return name, dict(values)


@ops.register
class LayerAdjustPreset(Operator):
    idname = "layer.adjust_preset"
    label = "调整预设"
    description = "给当前调整层套一组预设的数值（第一项「默认值」恢复默认）"
    searchable = False
    index = IntProperty("预设", default=0, min=0)

    @classmethod
    def poll(cls, ctx) -> bool:
        layer = getattr(ctx, "layer", None)
        return layer is not None and layer.kind == "ADJUST"

    def execute(self, ctx) -> str:
        layer = ctx.layer
        found = preset_values(layer, layer.adjust_type, int(self.index), ctx)
        if found is None:
            return CANCELLED
        name, values = found
        _apply_values(ctx, layer, values, "调整预设「%s」" % name)
        return FINISHED


def _apply_values(ctx, layer, values: dict, label: str) -> None:
    """一次改几个属性，记一步撤销。"""
    before = {prop: getattr(layer, prop) for prop in values}
    for prop, value in values.items():
        setattr(layer, prop, value)
    after = {prop: getattr(layer, prop) for prop in values}
    history = getattr(ctx.app, "history", None)
    if history is None or before == after:
        return

    def undo(layer=layer, values=before) -> None:
        for prop, value in values.items():
            setattr(layer, prop, value)

    def redo(layer=layer, values=after) -> None:
        for prop, value in values.items():
            setattr(layer, prop, value)

    history.push(label, undo, redo)


# ---------------------------------------------------------------- 直方图、自动色阶
_histograms: dict = {}


def _engine(ctx):
    app = ctx.app
    host = getattr(app, "host", None)
    engine = getattr(app, "engine", None)
    if host is None or engine is None:
        return None
    host.make_current()
    return engine


def composite_below(ctx, ts, layer=None, size: int = 256):
    """这套贴图合成结果的小图（layer 给出时只看它下面的）。拿不到时返回 None。"""
    from ..engine.projectio import composite_image

    engine = _engine(ctx)
    if engine is None or ts is None or ts.uid not in engine.sets:
        return None
    try:
        return composite_image(engine, ts, size, below=layer)
    except Exception:  # noqa: BLE001
        log.exception("合成预览出错")
        return None


def layer_histogram(ctx, layer):
    """调整层下面合成结果的直方图 (4, 256)：红、绿、蓝、明暗。画面没变时用上次算的。"""
    ts = getattr(ctx, "texture_set", None)
    engine = getattr(ctx.app, "engine", None)
    if ts is None or engine is None:
        return None
    key = (ts.uid, layer.uid, engine.content_version)
    cached = _histograms.get((ts.uid, layer.uid))
    if cached is not None and cached[0] == key:
        return cached[1]
    image = composite_below(ctx, ts, layer)
    counts = histogram(image) if image is not None else None
    if len(_histograms) > 64:
        _histograms.clear()
    _histograms[(ts.uid, layer.uid)] = (key, counts)
    return counts


AUTO_ITEMS = [("TONE", "自动色调", "红、绿、蓝各自拉满"), ("CONTRAST", "自动对比度", "只拉开反差，不改颜色"),
              ("COLOR", "自动颜色", "各自拉满，再把中间调校正成中性灰")]


@ops.register
class LayerAdjustAuto(Operator):
    idname = "layer.adjust_auto"
    label = "自动色阶"
    description = "按画面明暗的分布自动定黑白场：当前是色阶或曲线调整层时改它，否则在上面加一个色阶调整层"
    mode = EnumProperty("方式", items=AUTO_ITEMS, default="TONE")
    clip = FloatProperty("剪切", default=0.001, min=0.0, max=0.1, precision=4,
                         description="最暗、最亮的两头各舍掉多少比例再定黑白场")

    @classmethod
    def poll(cls, ctx) -> bool:
        return getattr(ctx, "texture_set", None) is not None and getattr(ctx.app, "engine", None) is not None

    def execute(self, ctx) -> str:
        from ..doc.project import Layer
        from .layer_ops import _record, _room

        ts = ctx.texture_set
        label = next(item[1] for item in AUTO_ITEMS if item[0] == self.mode)
        layer = ts.active_layer
        target = layer if (layer is not None and layer.kind == "ADJUST"
                           and layer.adjust_type in ("LEVELS", "CURVES")) else None
        image = composite_below(ctx, ts, target)
        if image is None:
            self.report(ctx, "现在拿不到合成结果", "WARNING")
            return CANCELLED
        values = auto_levels(image, self.mode, float(self.clip))
        if target is None:
            if not _room(ts, 1):
                self.report(ctx, "一套贴图的图层已经满了", "WARNING")
                return CANCELLED
            before = ts.structure()
            new = Layer(name=ts.unique_name(label), kind="ADJUST", adjust_type="LEVELS", use_metallic=False,
                        use_roughness=False, use_height=False, **values)
            ts.add_layer(new)
            _record(ctx, label, ts, before, added=[new])
            return FINISHED
        if target.adjust_type == "LEVELS":
            full = {prop: target.prop(prop).default_value(target) for prop in TYPE_PROPS["LEVELS"]}
            full.update(values)
        else:
            identity = ((0.0, 0.0), (1.0, 1.0))
            full = {"adj_curve": identity, "adj_curve_r": identity, "adj_curve_g": identity, "adj_curve_b": identity}
            if self.mode == "CONTRAST":
                full["adj_curve"] = levels_to_curve(values["adj_in_black"], values["adj_in_white"], 1.0)
            else:
                for channel in "rgb":
                    full["adj_curve_" + channel] = levels_to_curve(
                        values["adj_lv_%s_in_black" % channel], values["adj_lv_%s_in_white" % channel],
                        values.get("adj_lv_%s_gamma" % channel, 1.0))
        _apply_values(ctx, target, full, label)
        return FINISHED


# ---------------------------------------------------------------- 颜色查找：导入、导出 .cube
CUBE_FILTER = "查找表 (*.cube)"


@ops.register
class LayerAdjustLutImport(Operator):
    idname = "layer.adjust_lut_import"
    label = "导入查找表…"
    description = "读一个 .cube 文件（一维或三维）给当前调整层当颜色查找表，文件内容存进工程"
    icon = "import"
    filepath = StringProperty("文件", default="", subtype="FILE_PATH")

    @classmethod
    def poll(cls, ctx) -> bool:
        layer = getattr(ctx, "layer", None)
        return layer is not None and layer.kind == "ADJUST"

    def execute(self, ctx) -> str:
        import os

        from ..engine import cube

        path = self.filepath
        if not path:
            from PySide6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getOpenFileName(ctx.app.window, "导入查找表", "", CUBE_FILTER)
        if not path:
            return CANCELLED
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                found = cube.parse_cube(handle.read())
        except (OSError, cube.CubeError) as error:
            self.report(ctx, "没有导入：%s" % error, "ERROR")
            return CANCELLED
        name = os.path.splitext(os.path.basename(path))[0]
        _apply_values(ctx, ctx.layer, {"adjust_type": "COLOR_LOOKUP", "adj_lut_look": "FILE", "adj_lut_name": name,
                                       "adj_lut_data": cube.encode(found)}, "导入查找表「%s」" % name)
        self.report(ctx, "已导入查找表「%s」（%s，%d 格）" % (name, "三维" if found["dims"] == 3 else "一维", found["size"]))
        return FINISHED


@ops.register
class LayerAdjustLutExport(Operator):
    idname = "layer.adjust_lut_export"
    label = "导出查找表…"
    description = "把当前颜色查找层用的查找表存成 .cube 文件，别的软件也能用"
    icon = "export"
    filepath = StringProperty("文件", default="", subtype="FILE_PATH")

    @classmethod
    def poll(cls, ctx) -> bool:
        layer = getattr(ctx, "layer", None)
        return layer is not None and layer.kind == "ADJUST" and layer.adjust_type == "COLOR_LOOKUP"

    def execute(self, ctx) -> str:
        from ..engine import cube
        from ..engine.adjust import lookup_cube

        layer = ctx.layer
        found = lookup_cube(layer)
        if found is None:
            self.report(ctx, "这一层还没有查找表", "WARNING")
            return CANCELLED
        path = self.filepath
        if not path:
            from PySide6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getSaveFileName(ctx.app.window, "导出查找表", layer.name + ".cube", CUBE_FILTER)
        if not path:
            return CANCELLED
        if found["dims"] == 3:
            data, size = found["data"], found["size"]
        else:
            size = cube.BUILTIN_SIZE
            data = cube.apply(found, cube.grid(size))
        try:
            cube.write_cube(path, data, size, layer.name)
        except OSError as error:
            self.report(ctx, "没有导出：%s" % error, "ERROR")
            return CANCELLED
        self.report(ctx, "已导出 %s" % path)
        return FINISHED


@registry.register_menu
class AdjustPresetMenu(registry.Menu):
    idname = "LAYERS_MT_adjust_presets"
    label = "预设"

    def draw(self, layout, ctx) -> None:
        layer = getattr(ctx, "layer", None)
        if layer is None or layer.kind != "ADJUST":
            layout.label("先选一个调整层", role="dim")
            return
        for index, (name, _values) in enumerate(presets(layer.adjust_type)):
            if index == 1:
                layout.separator()
            layout.operator("layer.adjust_preset", text=name, index=index)
