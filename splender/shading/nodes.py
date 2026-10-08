"""材质节点的类型（照 Blender 的着色器节点，名字、插口、参数和算法都按 Blender）。

每种节点：
- inputs：(名字, 显示名, 插口类型, 默认值, 选项)。插口类型 FLOAT / COLOR / VECTOR；选项里 implicit="GENERATED" 之类
  表示没接线时用的隐含坐标（这种输入不显示数值框，和 Blender 一样）；min / max / subtype 给数值框用。
- outputs：(名字, 显示名, 插口类型)。
- params：不在插口上的参数（下拉、开关、色带），按名字放在 node.values 里。
- emit(cx)：生成着色器语句。cx.i(名字) 取输入（已转成这个插口的类型），cx.o(名字) 是输出变量名，
  cx.p(名字) 取参数的 Python 值，cx.ramp(名字) 是色带数据在 mg[] 里的起点，cx.slot(名字) 是存储缓冲里的值。
颜色插口按线性颜色算；界面上的颜色（和色带）按 sRGB 显示，进着色器前换成线性。
"""
from __future__ import annotations

from ..core.props import BoolProperty, ColorProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty
from ..core.props import PropertyGroup, RampProperty

SOCKET_COLORS = {"FLOAT": "#a1a1a1", "COLOR": "#c7c729", "VECTOR": "#6363c7"}
CATEGORIES = [("INPUT", "输入"), ("TEXTURE", "纹理"), ("COLOR", "颜色"), ("VECTOR", "矢量"), ("CONVERTER", "转换器"),
              ("OUTPUT", "输出"), ("LAYOUT", "布局")]
CATEGORY_COLORS = {"INPUT": "#83314a", "TEXTURE": "#79461d", "COLOR": "#6c6c18", "VECTOR": "#3c3c83",
                   "CONVERTER": "#246283", "OUTPUT": "#4b1c1c", "LAYOUT": "#3a3a3a"}
GLSL_TYPE = {"FLOAT": "float", "COLOR": "vec3", "VECTOR": "vec3"}

_TYPES: dict[str, type] = {}


def register(cls):
    _TYPES[cls.id] = cls
    cls.Values = _values_class(cls)
    return cls


def get(type_id: str):
    return _TYPES.get(type_id)


def all_types() -> list:
    order = {key: index for index, (key, _label) in enumerate(CATEGORIES)}
    ids = list(_TYPES)
    return sorted(_TYPES.values(), key=lambda t: (order.get(t.category, 99), ids.index(t.id)))


class _Values(PropertyGroup):
    undoable = True


def _values_class(node_type) -> type:
    """一种节点的数值组：每个可以填数的输入一个属性（名字 in_插口名），再加上参数。"""
    attrs = {}
    for name, label, kind, default, opts in node_type.inputs:
        if opts.get("implicit") or kind == "ANY":
            continue
        if kind == "FLOAT":
            attrs["in_" + name] = FloatProperty(label, default=float(default), min=opts.get("min", -1e9),
                                                max=opts.get("max", 1e9), soft_min=opts.get("soft_min"),
                                                soft_max=opts.get("soft_max"), precision=opts.get("precision", 3),
                                                subtype=opts.get("subtype", "NONE"), unit=opts.get("unit", ""))
        elif kind == "COLOR":
            attrs["in_" + name] = ColorProperty(label, default=tuple(default)[:3])
        else:
            attrs["in_" + name] = FloatVectorProperty(label, default=tuple(default), size=3, precision=3,
                                                      unit=opts.get("unit", ""))
    for name, prop in node_type.params.items():
        attrs[name] = prop
    return type(node_type.__name__ + "Values", (_Values,), attrs)


class ShaderNodeType:
    id = ""
    label = ""
    category = "INPUT"
    description = ""
    width = 150
    inputs: tuple = ()
    outputs: tuple = ()
    params: dict = {}
    Values = _Values

    @staticmethod
    def emit(cx) -> list[str]:
        return []


def _f(name, label, default=0.0, **opts):
    return (name, label, "FLOAT", default, opts)


def _c(name, label, default=(0.8, 0.8, 0.8)):
    return (name, label, "COLOR", default, {})


def _v(name, label, default=(0.0, 0.0, 0.0), **opts):
    return (name, label, "VECTOR", default, opts)


def _enum(label, items, default, description=""):
    return EnumProperty(label, items=items, default=default, description=description)


def _index(items, value) -> int:
    for k, item in enumerate(items):
        if item[0] == value:
            return k
    return 0


# ====================================================================== 输入
@register
class LayerStackNode(ShaderNodeType):
    id = "LAYER_STACK"
    label = "图层堆栈"
    category = "INPUT"
    description = "这套贴图的图层（绘制、填充、调整……）叠出来的结果"
    outputs = (("base_color", "基础色", "COLOR"), ("metallic", "金属度", "FLOAT"), ("roughness", "粗糙度", "FLOAT"),
               ("height", "高度", "FLOAT"))

    @staticmethod
    def emit(cx):
        return ["%s = st_base;" % cx.o("base_color"), "%s = st_metal;" % cx.o("metallic"),
                "%s = st_rough;" % cx.o("roughness"), "%s = st_height;" % cx.o("height")]


@register
class MeshMapsNode(ShaderNodeType):
    id = "MESH_MAPS"
    label = "模型贴图"
    category = "INPUT"
    description = "烘焙出来的模型信息：遮蔽、曲率、厚度、位置、法线、部件编号"
    outputs = (("ao", "环境遮蔽", "FLOAT"), ("curvature", "曲率", "FLOAT"), ("convex", "凸边", "FLOAT"),
               ("concave", "凹缝", "FLOAT"), ("thickness", "厚度", "FLOAT"), ("position", "位置", "VECTOR"),
               ("normal", "法线", "VECTOR"), ("part", "部件编号", "FLOAT"))
    width = 140

    @staticmethod
    def emit(cx):
        return ["%s = g_ma.r;" % cx.o("ao"), "%s = g_ma.g;" % cx.o("curvature"),
                "%s = clamp(g_ma.g * 2.0 - 1.0, 0.0, 1.0);" % cx.o("convex"),
                "%s = clamp(1.0 - g_ma.g * 2.0, 0.0, 1.0);" % cx.o("concave"),
                "%s = g_ma.b;" % cx.o("thickness"), "%s = mg_position();" % cx.o("position"),
                "%s = mg_normal();" % cx.o("normal"), "%s = max(g_id, 0.0);" % cx.o("part")]


@register
class TexCoordNode(ShaderNodeType):
    id = "TEX_COORD"
    label = "纹理坐标"
    category = "INPUT"
    description = "生成（按模型包围盒 0..1）、物体、UV、法线坐标"
    outputs = (("generated", "生成", "VECTOR"), ("object", "物体", "VECTOR"), ("uv", "UV", "VECTOR"),
               ("normal", "法线", "VECTOR"))
    width = 120

    @staticmethod
    def emit(cx):
        return ["%s = mg_generated();" % cx.o("generated"), "%s = mg_position();" % cx.o("object"),
                "%s = vec3(g_uv, 0.0);" % cx.o("uv"), "%s = mg_normal();" % cx.o("normal")]


@register
class ValueNode(ShaderNodeType):
    id = "VALUE"
    label = "数值"
    category = "INPUT"
    outputs = (("value", "值", "FLOAT"),)
    width = 120
    params = {"value": FloatProperty("值", default=0.5, precision=3)}

    @staticmethod
    def emit(cx):
        return ["%s = %s;" % (cx.o("value"), cx.slot("value", "FLOAT"))]


@register
class RGBNode(ShaderNodeType):
    id = "RGB"
    label = "RGB"
    category = "INPUT"
    outputs = (("color", "颜色", "COLOR"),)
    width = 130
    params = {"color": ColorProperty("颜色", default=(0.5, 0.5, 0.5))}

    @staticmethod
    def emit(cx):
        return ["%s = %s;" % (cx.o("color"), cx.slot("color", "COLOR"))]


# ====================================================================== 纹理
_GEN = {"implicit": "GENERATED"}


@register
class NoiseTextureNode(ShaderNodeType):
    id = "TEX_NOISE"
    label = "噪波纹理"
    category = "TEXTURE"
    description = "分形噪波（按模型表面算，UV 接缝两边连续）"
    inputs = (_v("vector", "矢量", **_GEN), _f("scale", "缩放", 5.0, min=-1000.0, max=1000.0),
              _f("detail", "细节", 2.0, min=0.0, max=15.0), _f("roughness", "粗糙度", 0.5, min=0.0, max=1.0,
                                                             subtype="FACTOR"),
              _f("distortion", "扭曲", 0.0, min=-1000.0, max=1000.0))
    outputs = (("fac", "系数", "FLOAT"), ("color", "颜色", "COLOR"))

    @staticmethod
    def emit(cx):
        return ["mg_tex_noise(%s, %s, %s, %s, %s, %s, %s);" % (cx.i("vector"), cx.i("scale"), cx.i("detail"),
                                                               cx.i("roughness"), cx.i("distortion"), cx.o("fac"),
                                                               cx.o("color"))]


VORONOI_METRIC = [("EUCLIDEAN", "欧几里得", ""), ("MANHATTAN", "曼哈顿", ""), ("CHEBYCHEV", "切比雪夫", "")]
VORONOI_FEATURE = [("F1", "F1（最近）", "到最近的点"), ("F2", "F2（第二近）", "到第二近的点"),
                   ("DISTANCE_TO_EDGE", "到边距离", "离细胞边缘多远")]


@register
class VoronoiTextureNode(ShaderNodeType):
    id = "TEX_VORONOI"
    label = "沃罗诺伊纹理"
    category = "TEXTURE"
    description = "细胞状的图案：石块、鳞片、裂纹"
    inputs = (_v("vector", "矢量", **_GEN), _f("scale", "缩放", 5.0, min=-1000.0, max=1000.0),
              _f("randomness", "随机度", 1.0, min=0.0, max=1.0, subtype="FACTOR"))
    outputs = (("distance", "距离", "FLOAT"), ("color", "颜色", "COLOR"), ("position", "位置", "VECTOR"))
    params = {"feature": _enum("特征", VORONOI_FEATURE, "F1"), "distance": _enum("距离", VORONOI_METRIC, "EUCLIDEAN")}

    @staticmethod
    def emit(cx):
        return ["mg_tex_voronoi(%s, %s, %s, %d, %d, %s, %s, %s);" % (
            cx.i("vector"), cx.i("scale"), cx.i("randomness"), _index(VORONOI_METRIC, cx.p("distance")),
            _index(VORONOI_FEATURE, cx.p("feature")), cx.o("distance"), cx.o("color"), cx.o("position"))]


WAVE_TYPE = [("BANDS", "条带", ""), ("RINGS", "环形", "")]
WAVE_DIRECTION = [("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", ""), ("DIAGONAL", "对角 / 球形", "")]
WAVE_PROFILE = [("SIN", "正弦", ""), ("SAW", "锯齿", ""), ("TRI", "三角", "")]


@register
class WaveTextureNode(ShaderNodeType):
    id = "TEX_WAVE"
    label = "波浪纹理"
    category = "TEXTURE"
    description = "条带或同心圆的波纹，可以用噪波扭曲（木纹、水纹）"
    inputs = (_v("vector", "矢量", **_GEN), _f("scale", "缩放", 5.0, min=-1000.0, max=1000.0),
              _f("distortion", "扭曲", 0.0, min=-1000.0, max=1000.0), _f("detail", "细节", 2.0, min=0.0, max=15.0),
              _f("detail_scale", "细节缩放", 1.0, min=-1000.0, max=1000.0),
              _f("detail_roughness", "细节粗糙度", 0.5, min=0.0, max=1.0, subtype="FACTOR"),
              _f("phase", "相位偏移", 0.0, min=-1000.0, max=1000.0))
    outputs = (("color", "颜色", "COLOR"), ("fac", "系数", "FLOAT"))
    params = {"wave_type": _enum("类型", WAVE_TYPE, "BANDS"), "direction": _enum("方向", WAVE_DIRECTION, "X"),
              "profile": _enum("波形", WAVE_PROFILE, "SIN")}

    @staticmethod
    def emit(cx):
        return ["%s = mg_tex_wave(%s, %s, %s, %s, %s, %s, %s, %d, %d, %d);" % (
            cx.o("fac"), cx.i("vector"), cx.i("scale"), cx.i("distortion"), cx.i("detail"), cx.i("detail_scale"),
            cx.i("detail_roughness"), cx.i("phase"), _index(WAVE_TYPE, cx.p("wave_type")),
            _index(WAVE_DIRECTION, cx.p("direction")), _index(WAVE_PROFILE, cx.p("profile"))),
            "%s = vec3(%s);" % (cx.o("color"), cx.o("fac"))]


GRADIENT_TYPE = [("LINEAR", "线性", ""), ("QUADRATIC", "二次方", ""), ("EASING", "缓动", ""),
                 ("DIAGONAL", "对角", ""), ("SPHERICAL", "球形", ""), ("QUADRATIC_SPHERE", "二次球形", ""),
                 ("RADIAL", "径向", "")]


@register
class GradientTextureNode(ShaderNodeType):
    id = "TEX_GRADIENT"
    label = "渐变纹理"
    category = "TEXTURE"
    description = "沿 X 方向（或成球形、径向）的渐变"
    inputs = (_v("vector", "矢量", **_GEN),)
    outputs = (("color", "颜色", "COLOR"), ("fac", "系数", "FLOAT"))
    params = {"gradient_type": _enum("类型", GRADIENT_TYPE, "LINEAR")}

    @staticmethod
    def emit(cx):
        return ["%s = mg_tex_gradient(%s, %d);" % (cx.o("fac"), cx.i("vector"),
                                                    _index(GRADIENT_TYPE, cx.p("gradient_type"))),
                "%s = vec3(%s);" % (cx.o("color"), cx.o("fac"))]


@register
class CheckerTextureNode(ShaderNodeType):
    id = "TEX_CHECKER"
    label = "棋盘格纹理"
    category = "TEXTURE"
    inputs = (_v("vector", "矢量", **_GEN), _c("color1", "颜色 1", (0.8, 0.8, 0.8)),
              _c("color2", "颜色 2", (0.2, 0.2, 0.2)), _f("scale", "缩放", 5.0, min=-1000.0, max=1000.0))
    outputs = (("color", "颜色", "COLOR"), ("fac", "系数", "FLOAT"))

    @staticmethod
    def emit(cx):
        return ["%s = mg_tex_checker(%s, %s);" % (cx.o("fac"), cx.i("vector"), cx.i("scale")),
                "%s = mix(%s, %s, %s);" % (cx.o("color"), cx.i("color2"), cx.i("color1"), cx.o("fac"))]


@register
class BrickTextureNode(ShaderNodeType):
    id = "TEX_BRICK"
    label = "砖块纹理"
    category = "TEXTURE"
    description = "砖墙：一排排错开的砖，砖之间是灰浆"
    width = 170
    inputs = (_v("vector", "矢量", **_GEN), _c("color1", "颜色 1", (0.8, 0.8, 0.8)), _c("color2", "颜色 2", (0.2, 0.2, 0.2)),
              _c("mortar", "灰浆", (0.0, 0.0, 0.0)), _f("scale", "缩放", 5.0, min=-1000.0, max=1000.0),
              _f("mortar_size", "灰浆宽度", 0.02, min=0.0, max=0.125), _f("mortar_smooth", "灰浆平滑", 0.1, min=0.0, max=1.0),
              _f("bias", "偏向", 0.0, min=-1.0, max=1.0), _f("brick_width", "砖宽", 0.5, min=0.01, max=100.0),
              _f("row_height", "行高", 0.25, min=0.01, max=100.0))
    outputs = (("color", "颜色", "COLOR"), ("fac", "系数", "FLOAT"))
    params = {"offset": FloatProperty("偏移", default=0.5, min=0.0, max=1.0, subtype="FACTOR"),
              "offset_frequency": IntProperty("偏移频率", default=2, min=1, max=99),
              "squash": FloatProperty("挤压", default=1.0, min=0.0, max=99.0),
              "squash_frequency": IntProperty("挤压频率", default=2, min=1, max=99)}

    @staticmethod
    def emit(cx):
        b = "b%d" % cx.uid
        return ["vec2 %s = mg_tex_brick(%s * %s, %s, %s, %s, %s, %s, %s, %d, %s, %d);" % (
            b, cx.i("vector"), cx.i("scale"), cx.i("mortar_size"), cx.i("mortar_smooth"), cx.i("bias"),
            cx.i("brick_width"), cx.i("row_height"), cx.slot("offset", "FLOAT"), int(cx.p("offset_frequency")),
            cx.slot("squash", "FLOAT"), int(cx.p("squash_frequency"))),
            "%s = mix(mix(%s, %s, %s.x), %s, %s.y);" % (cx.o("color"), cx.i("color1"), cx.i("color2"), b,
                                                       cx.i("mortar"), b),
            "%s = %s.y;" % (cx.o("fac"), b)]


@register
class WhiteNoiseTextureNode(ShaderNodeType):
    id = "TEX_WHITE_NOISE"
    label = "白噪波纹理"
    category = "TEXTURE"
    description = "每个点各自随机的值（没有连续性）"
    inputs = (_v("vector", "矢量", **_GEN),)
    outputs = (("value", "值", "FLOAT"), ("color", "颜色", "COLOR"))
    width = 130

    @staticmethod
    def emit(cx):
        return ["%s = mg_hash3(%s);" % (cx.o("color"), cx.i("vector")), "%s = %s.x;" % (cx.o("value"), cx.o("color"))]


_project_getter = [lambda: None]


def set_project_getter(getter) -> None:
    """程序启动时告诉节点去哪找当前工程（「节点图纹理」要列出工程里的节点图）。"""
    _project_getter[0] = getter


def _graph_items(owner=None):
    try:
        project = _project_getter[0]()
    except Exception:  # noqa: BLE001
        project = None
    items = [("0", "（选一张节点图）", "")]
    if project is not None:
        items += [(str(g.uid), g.name, "") for g in project.graphs]
    return items


@register
class GraphTextureNode(ShaderNodeType):
    id = "TEX_GRAPH"
    label = "节点图纹理"
    category = "TEXTURE"
    description = "用「节点」工作区里做的程序化贴图（按 UV 平铺）"
    width = 170
    inputs = (_v("vector", "矢量", implicit="UV"), _f("tiling", "平铺次数", 1.0, min=0.01, max=1000.0))
    outputs = (("color", "颜色", "COLOR"), ("metallic", "金属度", "FLOAT"), ("roughness", "粗糙度", "FLOAT"),
               ("height", "高度", "FLOAT"))
    params = {"graph": EnumProperty("节点图", items=_graph_items, default="0")}

    @staticmethod
    def emit(cx):
        g = "g%d" % cx.uid
        m = "m%d" % cx.uid
        slot = "int(%s + 0.5)" % cx.slot("@graph_slot", "FLOAT")
        return ["vec4 %s = mg_graph(%s, %s, %s, 0);" % (g, slot, cx.i("vector"), cx.i("tiling")),
                "vec4 %s = mg_graph(%s, %s, %s, 1);" % (m, slot, cx.i("vector"), cx.i("tiling")),
                "%s = srgb_to_linear(clamp(%s.rgb, 0.0, 1.0));" % (cx.o("color"), g),
                "%s = %s.r;" % (cx.o("metallic"), m), "%s = %s.g;" % (cx.o("roughness"), m),
                "%s = (%s.b - 0.5) * 2.0;" % (cx.o("height"), m)]


# ====================================================================== 颜色
MIX_BLEND = [("MIX", "混合", ""), ("DARKEN", "变暗", ""), ("MULTIPLY", "正片叠底", ""), ("BURN", "颜色加深", ""),
             ("LIGHTEN", "变亮", ""), ("SCREEN", "滤色", ""), ("DODGE", "颜色减淡", ""), ("ADD", "相加", ""),
             ("OVERLAY", "叠加", ""), ("SOFT_LIGHT", "柔光", ""), ("LINEAR_LIGHT", "线性光", ""),
             ("DIFFERENCE", "差值", ""), ("EXCLUSION", "排除", ""), ("SUBTRACT", "相减", ""), ("DIVIDE", "相除", ""),
             ("HUE", "色相", ""), ("SATURATION", "饱和度", ""), ("COLOR", "颜色", ""), ("VALUE", "明度", "")]


@register
class MixNode(ShaderNodeType):
    id = "MIX_RGB"
    label = "混合颜色"
    category = "COLOR"
    description = "两种颜色按混合模式叠在一起"
    inputs = (_f("fac", "系数", 0.5, min=0.0, max=1.0, subtype="FACTOR"), _c("a", "A", (0.5, 0.5, 0.5)),
              _c("b", "B", (0.5, 0.5, 0.5)))
    outputs = (("result", "结果", "COLOR"),)
    params = {"blend_type": _enum("混合", MIX_BLEND, "MIX"),
              "use_clamp": BoolProperty("钳制结果", default=False, description="结果夹到 0..1")}

    @staticmethod
    def emit(cx):
        expr = "mg_mix(%d, clamp(%s, 0.0, 1.0), %s, %s)" % (_index(MIX_BLEND, cx.p("blend_type")), cx.i("fac"),
                                                           cx.i("a"), cx.i("b"))
        if cx.p("use_clamp"):
            expr = "clamp(%s, 0.0, 1.0)" % expr
        return ["%s = %s;" % (cx.o("result"), expr)]


@register
class InvertNode(ShaderNodeType):
    id = "INVERT"
    label = "反相"
    category = "COLOR"
    inputs = (_f("fac", "系数", 1.0, min=0.0, max=1.0, subtype="FACTOR"), _c("color", "颜色", (0.0, 0.0, 0.0)))
    outputs = (("color", "颜色", "COLOR"),)
    width = 130

    @staticmethod
    def emit(cx):
        return ["%s = mix(%s, 1.0 - %s, %s);" % (cx.o("color"), cx.i("color"), cx.i("color"), cx.i("fac"))]


@register
class HueSaturationNode(ShaderNodeType):
    id = "HUE_SAT"
    label = "色相/饱和度/明度"
    category = "COLOR"
    inputs = (_f("hue", "色相", 0.5, min=0.0, max=1.0, subtype="FACTOR"), _f("saturation", "饱和度", 1.0, min=0.0, max=2.0),
              _f("value", "明度", 1.0, min=0.0, max=2.0), _f("fac", "系数", 1.0, min=0.0, max=1.0, subtype="FACTOR"),
              _c("color", "颜色", (0.8, 0.8, 0.8)))
    outputs = (("color", "颜色", "COLOR"),)

    @staticmethod
    def emit(cx):
        h = "h%d" % cx.uid
        return ["vec3 %s = rgb2hsv(max(%s, 0.0));" % (h, cx.i("color")),
                "%s.x = fract(%s.x + %s + 0.5);" % (h, h, cx.i("hue")),
                "%s.y = clamp(%s.y * %s, 0.0, 1.0);" % (h, h, cx.i("saturation")),
                "%s.z = %s.z * %s;" % (h, h, cx.i("value")),
                "%s = mix(%s, hsv2rgb(%s), %s);" % (cx.o("color"), cx.i("color"), h, cx.i("fac"))]


@register
class BrightContrastNode(ShaderNodeType):
    id = "BRIGHTCONTRAST"
    label = "亮度/对比度"
    category = "COLOR"
    inputs = (_c("color", "颜色", (1.0, 1.0, 1.0)), _f("bright", "亮度", 0.0, min=-100.0, max=100.0),
              _f("contrast", "对比度", 0.0, min=-100.0, max=100.0))
    outputs = (("color", "颜色", "COLOR"),)

    @staticmethod
    def emit(cx):
        return ["%s = max((1.0 + %s) * %s + (%s - %s * 0.5), 0.0);" % (
            cx.o("color"), cx.i("contrast"), cx.i("color"), cx.i("bright"), cx.i("contrast"))]


@register
class GammaNode(ShaderNodeType):
    id = "GAMMA"
    label = "伽马"
    category = "COLOR"
    inputs = (_c("color", "颜色", (1.0, 1.0, 1.0)), _f("gamma", "伽马", 1.0, min=0.001, max=10.0))
    outputs = (("color", "颜色", "COLOR"),)
    width = 130

    @staticmethod
    def emit(cx):
        return ["%s = pow(max(%s, 0.0), vec3(%s));" % (cx.o("color"), cx.i("color"), cx.i("gamma"))]


# ====================================================================== 转换器
@register
class ColorRampNode(ShaderNodeType):
    id = "VALTORGB"
    label = "颜色渐变"
    category = "CONVERTER"
    description = "按系数从色带上取颜色（色标可以拖、可以加减，插值方式可选）"
    width = 240
    inputs = (_f("fac", "系数", 0.5, min=0.0, max=1.0, subtype="FACTOR"),)
    outputs = (("color", "颜色", "COLOR"), ("alpha", "Alpha", "FLOAT"))
    params = {"ramp": RampProperty("色带")}

    @staticmethod
    def emit(cx):
        r = "r%d" % cx.uid
        return ["vec4 %s = mg_ramp(%d, %s);" % (r, cx.ramp("ramp"), cx.i("fac")),
                "%s = srgb_to_linear(%s.rgb);" % (cx.o("color"), r), "%s = %s.a;" % (cx.o("alpha"), r)]


MATH_OPS = [("ADD", "相加", ""), ("SUBTRACT", "相减", ""), ("MULTIPLY", "相乘", ""), ("DIVIDE", "相除", ""),
            ("MULTIPLY_ADD", "乘后再加", ""), ("POWER", "幂", ""), ("LOGARITHM", "对数", ""), ("SQRT", "平方根", ""),
            ("INVERSE_SQRT", "平方根倒数", ""), ("ABSOLUTE", "绝对值", ""), ("EXPONENT", "指数", ""),
            ("MINIMUM", "最小值", ""), ("MAXIMUM", "最大值", ""), ("LESS_THAN", "小于", ""), ("GREATER_THAN", "大于", ""),
            ("SIGN", "符号", ""), ("COMPARE", "比较", ""), ("SMOOTH_MIN", "平滑最小值", ""), ("SMOOTH_MAX", "平滑最大值", ""),
            ("ROUND", "四舍五入", ""), ("FLOOR", "向下取整", ""), ("CEIL", "向上取整", ""), ("TRUNC", "截断", ""),
            ("FRACT", "小数部分", ""), ("MODULO", "截断取余", ""), ("FLOORED_MODULO", "向下取余", ""),
            ("WRAP", "循环", ""), ("SNAP", "吸附", ""), ("PINGPONG", "往返", ""), ("SINE", "正弦", ""),
            ("COSINE", "余弦", ""), ("TANGENT", "正切", ""), ("ARCSINE", "反正弦", ""), ("ARCCOSINE", "反余弦", ""),
            ("ARCTANGENT", "反正切", ""), ("ARCTAN2", "二参数反正切", ""), ("SINH", "双曲正弦", ""),
            ("COSH", "双曲余弦", ""), ("TANH", "双曲正切", ""), ("RADIANS", "转为弧度", ""), ("DEGREES", "转为角度", "")]


@register
class MathNode(ShaderNodeType):
    id = "MATH"
    label = "运算"
    category = "CONVERTER"
    description = "两三个数做加减乘除、比较、取整、三角函数……"
    inputs = (_f("value1", "值", 0.5), _f("value2", "值", 0.5), _f("value3", "值", 0.0))
    outputs = (("value", "值", "FLOAT"),)
    params = {"operation": _enum("运算", MATH_OPS, "ADD"),
              "use_clamp": BoolProperty("钳制结果", default=False, description="结果夹到 0..1")}

    @staticmethod
    def emit(cx):
        expr = "mg_math(%d, %s, %s, %s)" % (_index(MATH_OPS, cx.p("operation")), cx.i("value1"), cx.i("value2"),
                                            cx.i("value3"))
        if cx.p("use_clamp"):
            expr = "clamp(%s, 0.0, 1.0)" % expr
        return ["%s = %s;" % (cx.o("value"), expr)]


VMATH_OPS = [("ADD", "相加", ""), ("SUBTRACT", "相减", ""), ("MULTIPLY", "相乘", ""), ("DIVIDE", "相除", ""),
             ("CROSS_PRODUCT", "叉积", ""), ("PROJECT", "投影", ""), ("REFLECT", "反射", ""), ("DOT_PRODUCT", "点积", ""),
             ("DISTANCE", "距离", ""), ("LENGTH", "长度", ""), ("SCALE", "缩放", ""), ("NORMALIZE", "归一化", ""),
             ("ABSOLUTE", "绝对值", ""), ("MINIMUM", "最小值", ""), ("MAXIMUM", "最大值", ""), ("FLOOR", "向下取整", ""),
             ("CEIL", "向上取整", ""), ("FRACTION", "小数部分", ""), ("MODULO", "取余", ""), ("SNAP", "吸附", ""),
             ("SINE", "正弦", ""), ("COSINE", "余弦", ""), ("TANGENT", "正切", "")]


@register
class VectorMathNode(ShaderNodeType):
    id = "VECT_MATH"
    label = "矢量运算"
    category = "CONVERTER"
    inputs = (_v("a", "矢量"), _v("b", "矢量"), _f("scale", "缩放", 1.0))
    outputs = (("vector", "矢量", "VECTOR"), ("value", "值", "FLOAT"))
    params = {"operation": _enum("运算", VMATH_OPS, "ADD")}

    @staticmethod
    def emit(cx):
        return ["mg_vmath(%d, %s, %s, %s, %s, %s);" % (_index(VMATH_OPS, cx.p("operation")), cx.i("a"), cx.i("b"),
                                                       cx.i("scale"), cx.o("vector"), cx.o("value"))]


MAP_RANGE_TYPES = [("LINEAR", "线性", ""), ("STEPPED", "阶梯", ""), ("SMOOTHSTEP", "平滑阶梯", ""),
                   ("SMOOTHERSTEP", "更平滑阶梯", "")]


@register
class MapRangeNode(ShaderNodeType):
    id = "MAP_RANGE"
    label = "映射范围"
    category = "CONVERTER"
    description = "把一个范围里的数换算到另一个范围"
    inputs = (_f("value", "值", 1.0), _f("from_min", "从最小值", 0.0), _f("from_max", "从最大值", 1.0),
              _f("to_min", "到最小值", 0.0), _f("to_max", "到最大值", 1.0), _f("steps", "步数", 4.0, min=0.0))
    outputs = (("result", "结果", "FLOAT"),)
    params = {"interpolation_type": _enum("插值", MAP_RANGE_TYPES, "LINEAR"),
              "clamp": BoolProperty("钳制", default=True, description="结果不超出目标范围")}

    @staticmethod
    def emit(cx):
        return ["%s = mg_map_range(%d, %s, %s, %s, %s, %s, %s, %s);" % (
            cx.o("result"), _index(MAP_RANGE_TYPES, cx.p("interpolation_type")), cx.i("value"), cx.i("from_min"),
            cx.i("from_max"), cx.i("to_min"), cx.i("to_max"), cx.i("steps"), "true" if cx.p("clamp") else "false")]


@register
class ClampNode(ShaderNodeType):
    id = "CLAMP"
    label = "钳制"
    category = "CONVERTER"
    inputs = (_f("value", "值", 1.0), _f("min", "最小值", 0.0), _f("max", "最大值", 1.0))
    outputs = (("result", "结果", "FLOAT"),)
    params = {"clamp_type": _enum("类型", [("MINMAX", "最小最大", ""), ("RANGE", "范围", "最小、最大可以反过来")],
                                  "MINMAX")}
    width = 130

    @staticmethod
    def emit(cx):
        if cx.p("clamp_type") == "RANGE":
            return ["%s = clamp(%s, min(%s, %s), max(%s, %s));" % (cx.o("result"), cx.i("value"), cx.i("min"),
                                                                   cx.i("max"), cx.i("min"), cx.i("max"))]
        return ["%s = clamp(%s, %s, %s);" % (cx.o("result"), cx.i("value"), cx.i("min"), cx.i("max"))]


@register
class RGBToBWNode(ShaderNodeType):
    id = "RGBTOBW"
    label = "RGB 转 BW"
    category = "CONVERTER"
    inputs = (_c("color", "颜色", (0.5, 0.5, 0.5)),)
    outputs = (("value", "值", "FLOAT"),)
    width = 120

    @staticmethod
    def emit(cx):
        return ["%s = mg_lum(%s);" % (cx.o("value"), cx.i("color"))]


@register
class SeparateRGBNode(ShaderNodeType):
    id = "SEPRGB"
    label = "分离 RGB"
    category = "CONVERTER"
    inputs = (_c("color", "颜色", (0.8, 0.8, 0.8)),)
    outputs = (("r", "R", "FLOAT"), ("g", "G", "FLOAT"), ("b", "B", "FLOAT"))
    width = 120

    @staticmethod
    def emit(cx):
        c = cx.i("color")
        return ["%s = %s.r;" % (cx.o("r"), c), "%s = %s.g;" % (cx.o("g"), c), "%s = %s.b;" % (cx.o("b"), c)]


@register
class CombineRGBNode(ShaderNodeType):
    id = "COMBRGB"
    label = "合并 RGB"
    category = "CONVERTER"
    inputs = (_f("r", "R", 0.0, min=0.0, max=1.0, subtype="FACTOR"), _f("g", "G", 0.0, min=0.0, max=1.0, subtype="FACTOR"),
              _f("b", "B", 0.0, min=0.0, max=1.0, subtype="FACTOR"))
    outputs = (("color", "颜色", "COLOR"),)
    width = 120

    @staticmethod
    def emit(cx):
        return ["%s = vec3(%s, %s, %s);" % (cx.o("color"), cx.i("r"), cx.i("g"), cx.i("b"))]


@register
class SeparateXYZNode(ShaderNodeType):
    id = "SEPXYZ"
    label = "分离 XYZ"
    category = "CONVERTER"
    inputs = (_v("vector", "矢量"),)
    outputs = (("x", "X", "FLOAT"), ("y", "Y", "FLOAT"), ("z", "Z", "FLOAT"))
    width = 120

    @staticmethod
    def emit(cx):
        v = cx.i("vector")
        return ["%s = %s.x;" % (cx.o("x"), v), "%s = %s.y;" % (cx.o("y"), v), "%s = %s.z;" % (cx.o("z"), v)]


@register
class CombineXYZNode(ShaderNodeType):
    id = "COMBXYZ"
    label = "合并 XYZ"
    category = "CONVERTER"
    inputs = (_f("x", "X", 0.0), _f("y", "Y", 0.0), _f("z", "Z", 0.0))
    outputs = (("vector", "矢量", "VECTOR"),)
    width = 120

    @staticmethod
    def emit(cx):
        return ["%s = vec3(%s, %s, %s);" % (cx.o("vector"), cx.i("x"), cx.i("y"), cx.i("z"))]


@register
class SeparateHSVNode(ShaderNodeType):
    id = "SEPHSV"
    label = "分离 HSV"
    category = "CONVERTER"
    inputs = (_c("color", "颜色", (0.8, 0.8, 0.8)),)
    outputs = (("h", "H", "FLOAT"), ("s", "S", "FLOAT"), ("v", "V", "FLOAT"))
    width = 120

    @staticmethod
    def emit(cx):
        h = "h%d" % cx.uid
        return ["vec3 %s = rgb2hsv(max(%s, 0.0));" % (h, cx.i("color")), "%s = %s.x;" % (cx.o("h"), h),
                "%s = %s.y;" % (cx.o("s"), h), "%s = %s.z;" % (cx.o("v"), h)]


@register
class CombineHSVNode(ShaderNodeType):
    id = "COMBHSV"
    label = "合并 HSV"
    category = "CONVERTER"
    inputs = (_f("h", "H", 0.0, min=0.0, max=1.0, subtype="FACTOR"), _f("s", "S", 0.0, min=0.0, max=1.0, subtype="FACTOR"),
              _f("v", "V", 0.0, min=0.0, max=1.0, subtype="FACTOR"))
    outputs = (("color", "颜色", "COLOR"),)
    width = 120

    @staticmethod
    def emit(cx):
        return ["%s = hsv2rgb(vec3(fract(%s), clamp(%s, 0.0, 1.0), max(%s, 0.0)));" % (
            cx.o("color"), cx.i("h"), cx.i("s"), cx.i("v"))]


# ====================================================================== 矢量
MAPPING_TYPES = [("POINT", "点", ""), ("TEXTURE", "纹理", "反过来的变换（挪纹理而不是挪坐标）"), ("VECTOR", "矢量", ""),
                 ("NORMAL", "法线", "")]


@register
class MappingNode(ShaderNodeType):
    id = "MAPPING"
    label = "映射"
    category = "VECTOR"
    description = "坐标的平移、旋转、缩放"
    width = 170
    inputs = (_v("vector", "矢量", **_GEN), _v("location", "位置", unit="m"), _v("rotation", "旋转", unit="°"),
              _v("scale", "缩放", (1.0, 1.0, 1.0)))
    outputs = (("vector", "矢量", "VECTOR"),)
    params = {"vector_type": _enum("类型", MAPPING_TYPES, "POINT")}

    @staticmethod
    def emit(cx):
        return ["%s = mg_mapping(%d, %s, %s, %s, %s);" % (
            cx.o("vector"), _index(MAPPING_TYPES, cx.p("vector_type")), cx.i("vector"), cx.i("location"),
            cx.i("rotation"), cx.i("scale"))]


# ====================================================================== 布局、输出
@register
class RerouteNode(ShaderNodeType):
    id = "REROUTE"
    label = "转接点"
    category = "LAYOUT"
    description = "线的转弯点，让连线整齐"
    width = 16
    inputs = (("input", "", "ANY", 0.0, {}),)
    outputs = (("output", "", "ANY"),)


@register
class MaterialOutputNode(ShaderNodeType):
    id = "OUTPUT_MATERIAL"
    label = "材质输出"
    category = "OUTPUT"
    description = "这套贴图最后的基础色、金属度、粗糙度、高度"
    inputs = (_c("base_color", "基础色", (0.8, 0.8, 0.8)),
              _f("metallic", "金属度", 0.0, min=0.0, max=1.0, subtype="FACTOR"),
              _f("roughness", "粗糙度", 0.5, min=0.0, max=1.0, subtype="FACTOR"),
              _f("height", "高度", 0.0, min=-1.0, max=1.0))
    outputs = ()

    @staticmethod
    def emit(cx):
        return ["base = %s;" % cx.i("base_color"), "metal = clamp(%s, 0.0, 1.0);" % cx.i("metallic"),
                "rough = clamp(%s, 0.0, 1.0);" % cx.i("roughness"), "height = clamp(%s, -1.0, 1.0);" % cx.i("height")]
