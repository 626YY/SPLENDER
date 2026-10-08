"""节点类型。每种节点：显示名、分类、输入（名字、显示名、没接线时的值）、参数（属性组）、着色器（vec4 node(vec2 uv)）。

着色器里用 IN0(uv)..IN3(uv) 取输入（颜色），G0(uv)..G3(uv) 取灰度，参数按 p_名字 取（见 glsl.py）。
"""
from __future__ import annotations

from ..core.props import ColorProperty, EnumProperty, FloatProperty, IntProperty, PropertyGroup, RampProperty

_TYPES: dict[str, "NodeType"] = {}

CATEGORIES = [("GENERATE", "生成"), ("ADJUST", "调整"), ("FILTER", "滤镜"), ("OUTPUT", "输出")]
CATEGORY_LABELS = dict(CATEGORIES)

WHITE = (1.0, 1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)
MID = (0.5, 0.5, 0.5, 1.0)


class NodeType:
    id = ""
    label = ""
    category = "GENERATE"
    description = ""
    inputs: tuple = ()               # ((名字, 显示名, 默认值 RGBA, 灰度?), ...)
    output_kind = "gray"             # gray / color / input（和第一个输入一样） / none（输出节点）
    Params = PropertyGroup
    glsl = ""


def register(cls):
    _TYPES[cls.id] = cls
    return cls


def get(type_id: str):
    return _TYPES.get(type_id)


def all_types() -> list:
    order = {key: index for index, (key, _label) in enumerate(CATEGORIES)}
    return sorted(_TYPES.values(), key=lambda t: (order.get(t.category, 99), list(_TYPES).index(t.id)))


class _Params(PropertyGroup):
    undoable = True


def _seed(label="随机种子"):
    return IntProperty(label, default=0, min=0, max=99999, description="换一个数，图案就换一种")


# ====================================================================== 生成
@register
class NoiseNode(NodeType):
    id = "NOISE"
    label = "噪波"
    description = "连绵起伏的云状噪波，可以无缝平铺"

    class Params(_Params):
        scale = IntProperty("尺度", default=4, min=1, max=256, description="一张图里大起伏的个数，越大越细碎")
        octaves = IntProperty("层次", default=5, min=1, max=12, description="叠几层越来越细的细节")
        persistence = FloatProperty("细节强度", default=0.5, min=0.0, max=1.0, subtype="FACTOR",
                                    description="越细的层次占多大分量")
        kind = EnumProperty("样子", items=[("SMOOTH", "平滑", "柔和的云"), ("BILLOW", "云絮", "鼓起的团块"),
                                         ("RIDGED", "山脊", "尖锐的脊线")], default="SMOOTH")
        seed = _seed()

    glsl = "vec4 node(vec2 uv) { return gray(fbm(uv, p_scale, p_octaves, p_persistence, p_kind, p_seed)); }"


@register
class CellsNode(NodeType):
    id = "CELLS"
    label = "细胞"
    description = "一个个细胞：可以出距离、裂缝或每块不同的灰度"

    class Params(_Params):
        scale = IntProperty("尺度", default=6, min=1, max=256, description="一行有几个细胞")
        jitter = FloatProperty("错乱", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
                               description="0 是整齐的格子，1 是完全随机")
        mode = EnumProperty("输出", items=[("DISTANCE", "距离", "离细胞中心越远越亮"),
                                         ("CRACKS", "裂缝", "细胞之间的缝"),
                                         ("PATCHES", "色块", "每个细胞一个随机灰度")], default="DISTANCE")
        width = FloatProperty("缝宽", default=0.08, min=0.0, max=1.0, precision=3, description="「裂缝」的宽度")
        seed = _seed()

    glsl = """
vec4 node(vec2 uv) {
  vec4 v = voronoi(uv, p_scale, p_jitter, p_seed);
  if (p_mode == 0) return gray(clamp(v.x, 0.0, 1.0));
  if (p_mode == 1) return gray(smoothstep(0.0, max(p_width, 1e-4), v.y - v.x));
  return gray(v.z);
}"""


@register
class BricksNode(NodeType):
    id = "BRICKS"
    label = "砖块"
    description = "砌砖、地砖、瓦片：带缝和倒角，每块可以有不同的明暗"

    class Params(_Params):
        columns = IntProperty("每行块数", default=4, min=1, max=256)
        rows = IntProperty("行数", default=8, min=1, max=256)
        offset = FloatProperty("错缝", default=0.5, min=0.0, max=1.0, subtype="FACTOR",
                               description="隔行错开多少，0 是对齐的方砖")
        gap = FloatProperty("缝宽", default=0.08, min=0.0, max=1.0, subtype="FACTOR")
        bevel = FloatProperty("倒角", default=0.2, min=0.0, max=1.0, subtype="FACTOR", description="砖边圆滑的程度")
        output = EnumProperty("输出", items=[("HEIGHT", "高度", "砖面高、缝低"),
                                           ("RANDOM", "明暗", "每块砖一个随机灰度"),
                                           ("MASK", "砖面", "砖是白色、缝是黑色")], default="HEIGHT")
        seed = _seed()

    glsl = """
vec4 node(vec2 uv) {
  vec4 b = bricks(uv, p_columns, p_rows, p_offset, p_gap, p_bevel, p_seed);
  if (p_output == 0) return gray(b.x);
  if (p_output == 1) return gray(b.y * step(1e-4, b.x));
  return gray(step(1e-4, b.x));
}"""


@register
class ShapeNode(NodeType):
    id = "SHAPE"
    label = "形状"
    description = "排成格子的圆点、方块、圆环、菱形、星星"

    class Params(_Params):
        count = IntProperty("每行个数", default=4, min=1, max=256)
        shape = EnumProperty("形状", items=[("CIRCLE", "圆", ""), ("SQUARE", "方", ""), ("RING", "环", ""),
                                          ("DIAMOND", "菱形", ""), ("STAR", "星", "")], default="CIRCLE")
        size = FloatProperty("大小", default=0.7, min=0.0, max=1.5, precision=3)
        softness = FloatProperty("边缘柔和", default=0.02, min=0.0, max=1.0, precision=3)
        angle = FloatProperty("转角", default=0.0, min=-180.0, max=180.0, unit="°", precision=0)

    glsl = """
vec4 node(vec2 uv) {
  vec2 g = rotate2(fract(uv * float(p_count)) - 0.5, p_angle);
  float s = p_size * 0.5;
  float d;
  if (p_shape == 0) d = length(g) - s;
  else if (p_shape == 1) d = max(abs(g.x), abs(g.y)) - s;
  else if (p_shape == 2) d = abs(length(g) - s * 0.75) - s * 0.25;
  else if (p_shape == 3) d = (abs(g.x) + abs(g.y)) - s;
  else {
    float a = atan(g.y, g.x);
    d = length(g) - s * mix(0.45, 1.0, 0.5 + 0.5 * cos(5.0 * a));
  }
  float aa = max(fwidth(d), 1e-5) + p_softness * 0.5;
  return gray(1.0 - smoothstep(-aa, aa, d));
}"""


@register
class WavesNode(NodeType):
    id = "WAVES"
    label = "波纹"
    description = "一条条平行的波纹，可以当条纹、织物纹理的底子"

    class Params(_Params):
        count = IntProperty("条数", default=8, min=1, max=512)
        direction = EnumProperty("方向", items=[("U", "横向", "沿 U 排列"), ("V", "纵向", "沿 V 排列"),
                                              ("DIAGONAL", "斜向", "沿对角线排列")], default="U")
        sharpness = FloatProperty("锐利", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
                                  description="0 是柔和的正弦，1 接近方波")

    glsl = """
vec4 node(vec2 uv) {
  float t = p_direction == 0 ? uv.x : (p_direction == 1 ? uv.y : uv.x + uv.y);
  float s = sin(TAU * t * float(p_count));
  float k = mix(1.0, 12.0, p_sharpness * p_sharpness);
  return gray(clamp(0.5 + 0.5 * clamp(s * k, -1.0, 1.0), 0.0, 1.0));
}"""


@register
class GradientNode(NodeType):
    id = "GRADIENT"
    label = "渐变"
    description = "从黑到白的渐变"

    class Params(_Params):
        direction = EnumProperty("方向", items=[("U", "左到右", ""), ("V", "下到上", ""),
                                              ("RADIAL", "中心向外", "")], default="U")

    glsl = """
vec4 node(vec2 uv) {
  if (p_direction == 0) return gray(uv.x);
  if (p_direction == 1) return gray(uv.y);
  return gray(clamp(length(uv - 0.5) * 2.0, 0.0, 1.0));
}"""


@register
class ScratchesNode(NodeType):
    id = "SCRATCHES"
    label = "划痕"
    description = "大致朝一个方向的细划痕，两头渐淡"

    class Params(_Params):
        count = IntProperty("数量", default=64, min=1, max=4096, description="大约有多少道")
        length = FloatProperty("长度", default=0.15, min=0.0, max=1.0, precision=3)
        width = FloatProperty("宽度", default=0.003, min=0.0, max=0.1, precision=4)
        angle = FloatProperty("方向", default=30.0, min=-180.0, max=180.0, unit="°", precision=0)
        spread = FloatProperty("方向散乱", default=20.0, min=0.0, max=180.0, unit="°", precision=0)
        seed = _seed()

    glsl = """
vec4 node(vec2 uv) {
  int n = max(1, int(sqrt(float(max(p_count, 1)))));
  vec2 g = uv * float(n);
  ivec2 c0 = ivec2(floor(g));
  float best = 0.0;
  for (int y = -2; y <= 2; y++) {
    for (int x = -2; x <= 2; x++) {
      ivec2 c = c0 + ivec2(x, y);
      vec3 h = hash3(wrap2(c, ivec2(n)), p_seed);
      vec3 h2 = hash3(wrap2(c, ivec2(n)), p_seed + 977);
      vec2 center = vec2(c) + h.xy;
      float a = radians(p_angle + (h.z - 0.5) * 2.0 * p_spread);
      vec2 dir = vec2(cos(a), sin(a));
      float half_len = min(p_length * float(n) * (0.5 + h2.x) * 0.5, 1.9);
      vec2 d = g - center;
      float t = clamp(dot(d, dir), -half_len, half_len);
      float dist = length(d - dir * t) / float(n);
      float line = 1.0 - smoothstep(0.0, max(p_width, 1e-5) + 0.5 / u_size.x, dist);
      float fade = 1.0 - abs(t) / max(half_len, 1e-5);
      best = max(best, line * fade * mix(0.4, 1.0, h2.y));
    }
  }
  return gray(best);
}"""


@register
class ValueNode(NodeType):
    id = "VALUE"
    label = "数值"
    description = "一个固定的灰度"

    class Params(_Params):
        value = FloatProperty("数值", default=0.5, min=0.0, max=1.0, subtype="FACTOR")

    glsl = "vec4 node(vec2 uv) { return gray(p_value); }"


@register
class ColorNode(NodeType):
    id = "COLOR"
    label = "颜色"
    description = "一个固定的颜色"
    output_kind = "color"

    class Params(_Params):
        color = ColorProperty("颜色", default=(0.8, 0.35, 0.2))

    glsl = "vec4 node(vec2 uv) { return vec4(p_color, 1.0); }"


# ====================================================================== 调整
@register
class BlendNode(NodeType):
    id = "BLEND"
    label = "混合"
    category = "ADJUST"
    description = "把前景按某种方式叠到背景上，可以用蒙版控制叠在哪里"
    inputs = (("fg", "前景", WHITE, False), ("bg", "背景", BLACK, False), ("mask", "蒙版", WHITE, True))
    output_kind = "input"

    class Params(_Params):
        mode = EnumProperty("方式", items=[("NORMAL", "覆盖", ""), ("MULTIPLY", "正片叠底", ""), ("ADD", "相加", ""),
                                         ("SUBTRACT", "相减", ""), ("SCREEN", "滤色", ""), ("OVERLAY", "叠加", ""),
                                         ("LIGHTEN", "变亮", ""), ("DARKEN", "变暗", ""), ("DIFFERENCE", "差值", "")],
                            default="NORMAL")
        opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR")

    glsl = """
vec3 blend_mode(vec3 a, vec3 b, int mode) {
  if (mode == 0) return a;
  if (mode == 1) return a * b;
  if (mode == 2) return min(a + b, vec3(1.0));
  if (mode == 3) return max(b - a, vec3(0.0));
  if (mode == 4) return 1.0 - (1.0 - a) * (1.0 - b);
  if (mode == 5) return mix(2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b), step(0.5, b));
  if (mode == 6) return max(a, b);
  if (mode == 7) return min(a, b);
  return abs(a - b);
}
vec4 node(vec2 uv) {
  vec4 a = IN0(uv);
  vec4 b = IN1(uv);
  float m = clamp(G2(uv) * p_opacity, 0.0, 1.0);
  return vec4(mix(b.rgb, blend_mode(a.rgb, b.rgb, p_mode), m), 1.0);
}"""


@register
class LevelsNode(NodeType):
    id = "LEVELS"
    label = "色阶"
    category = "ADJUST"
    description = "调黑场、白场和中间调"
    inputs = (("input", "输入", MID, False),)
    output_kind = "input"

    class Params(_Params):
        in_low = FloatProperty("输入黑场", default=0.0, min=0.0, max=1.0, subtype="FACTOR", precision=3)
        in_high = FloatProperty("输入白场", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=3)
        gamma = FloatProperty("中间调", default=1.0, min=0.05, max=10.0, precision=2,
                              description="大于 1 中间调变亮，小于 1 变暗")
        out_low = FloatProperty("输出黑场", default=0.0, min=0.0, max=1.0, subtype="FACTOR", precision=3)
        out_high = FloatProperty("输出白场", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=3)

    glsl = """
vec4 node(vec2 uv) {
  vec4 c = IN0(uv);
  vec3 t = clamp((c.rgb - p_in_low) / max(p_in_high - p_in_low, 1e-5), 0.0, 1.0);
  t = pow(t, vec3(1.0 / max(p_gamma, 1e-3)));
  return vec4(mix(vec3(p_out_low), vec3(p_out_high), t), 1.0);
}"""


@register
class InvertNode(NodeType):
    id = "INVERT"
    label = "反相"
    category = "ADJUST"
    description = "黑白（颜色）反过来"
    inputs = (("input", "输入", MID, False),)
    output_kind = "input"
    Params = _Params
    glsl = "vec4 node(vec2 uv) { vec4 c = IN0(uv); return vec4(1.0 - c.rgb, 1.0); }"


@register
class ThresholdNode(NodeType):
    id = "THRESHOLD"
    label = "阈值"
    category = "ADJUST"
    description = "比阈值亮的变白，暗的变黑"
    inputs = (("input", "输入", MID, True),)

    class Params(_Params):
        threshold = FloatProperty("阈值", default=0.5, min=0.0, max=1.0, subtype="FACTOR", precision=3)
        softness = FloatProperty("过渡", default=0.02, min=0.0, max=1.0, subtype="FACTOR", precision=3)

    glsl = """
vec4 node(vec2 uv) {
  float s = max(p_softness * 0.5, 1e-4);
  return gray(smoothstep(p_threshold - s, p_threshold + s, G0(uv)));
}"""


@register
class GradientMapNode(NodeType):
    id = "GRADIENT_MAP"
    label = "渐变映射"
    category = "ADJUST"
    description = "按灰度从色带上取色（色标可以拖、可以加减）"
    inputs = (("input", "灰度", MID, True),)
    output_kind = "color"

    class Params(_Params):
        ramp = RampProperty("色带", default=("LINEAR", ((0.0, 0.08, 0.06, 0.05, 1.0), (0.5, 0.45, 0.30, 0.18, 1.0),
                                                         (1.0, 0.92, 0.85, 0.70, 1.0))))

        def from_dict(self, data) -> None:
            data = dict(data or {})
            if "ramp" not in data and "dark" in data:
                # 旧版的三色参数换成三个色标
                m = float(data.get("position", 0.5))
                stops = [(0.0, *data["dark"][:3], 1.0), (m, *data.get("middle", (0.5, 0.5, 0.5))[:3], 1.0),
                         (1.0, *data.get("light", (1.0, 1.0, 1.0))[:3], 1.0)]
                data["ramp"] = {"interpolation": "LINEAR", "stops": stops}
            super().from_dict(data)

    glsl = """
vec4 node(vec2 uv) {
  return vec4(ramp_ramp(clamp(G0(uv), 0.0, 1.0)).rgb, 1.0);
}"""


@register
class HsvNode(NodeType):
    id = "HSV"
    label = "色相饱和度"
    category = "ADJUST"
    description = "转色相，调饱和度和明度"
    inputs = (("input", "颜色", MID, False),)
    output_kind = "color"

    class Params(_Params):
        hue = FloatProperty("色相", default=0.0, min=-180.0, max=180.0, unit="°", precision=0)
        saturation = FloatProperty("饱和度", default=0.0, min=-1.0, max=1.0, precision=2, description="负数变灰，正数更艳")
        value = FloatProperty("明度", default=0.0, min=-1.0, max=1.0, precision=2)

    glsl = """
vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + 1e-10)), d / (q.x + 1e-10), q.x);
}
vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}
vec4 node(vec2 uv) {
  vec3 h = rgb2hsv(clamp(IN0(uv).rgb, 0.0, 1.0));
  h.x = fract(h.x + p_hue / 360.0);
  h.y = clamp(p_saturation < 0.0 ? h.y * (1.0 + p_saturation) : h.y + (1.0 - h.y) * p_saturation, 0.0, 1.0);
  h.z = clamp(p_value < 0.0 ? h.z * (1.0 + p_value) : h.z + (1.0 - h.z) * p_value, 0.0, 1.0);
  return vec4(hsv2rgb(h), 1.0);
}"""


@register
class GrayscaleNode(NodeType):
    id = "GRAYSCALE"
    label = "转灰度"
    category = "ADJUST"
    description = "颜色按亮度变成灰度"
    inputs = (("input", "颜色", MID, False),)
    Params = _Params
    glsl = "vec4 node(vec2 uv) { return gray(G0(uv)); }"


# ====================================================================== 滤镜
@register
class BlurNode(NodeType):
    id = "BLUR"
    label = "模糊"
    category = "FILTER"
    description = "高斯模糊，边界接着另一边（平铺不露缝）"
    inputs = (("input", "输入", MID, False),)
    output_kind = "input"

    class Params(_Params):
        radius = FloatProperty("半径", default=0.01, min=0.0, max=0.5, precision=4, description="按整张图的宽度算")

    glsl = """
vec4 node(vec2 uv) {
  float r = max(p_radius, 0.25 / u_size.x);
  float step_uv = r / 4.0;
  float lod = log2(max(step_uv * u_size.x, 1.0));
  float sigma = r * 0.5;
  vec4 sum = vec4(0.0);
  float wsum = 0.0;
  for (int y = -4; y <= 4; y++) {
    for (int x = -4; x <= 4; x++) {
      vec2 o = vec2(float(x), float(y)) * step_uv;
      float w = exp(-dot(o, o) / (2.0 * sigma * sigma));
      sum += IN0L(uv + o, lod) * w;
      wsum += w;
    }
  }
  return sum / wsum;
}"""


@register
class WarpNode(NodeType):
    id = "WARP"
    label = "扭曲"
    category = "FILTER"
    description = "顺着扭曲图的坡度推开图案，让它变得自然、不规整"
    inputs = (("input", "输入", MID, False), ("warp", "扭曲图", MID, True))
    output_kind = "input"

    class Params(_Params):
        intensity = FloatProperty("强度", default=1.0, min=0.0, max=50.0, soft_max=10.0, precision=2)

    glsl = """
vec4 node(vec2 uv) {
  float e = 1.0 / u_size.x;
  vec2 g = vec2(G1(uv + vec2(e, 0.0)) - G1(uv - vec2(e, 0.0)), G1(uv + vec2(0.0, e)) - G1(uv - vec2(0.0, e)));
  g /= 2.0 * e;
  return IN0(uv + g * p_intensity * 0.002);
}"""


@register
class DirectionalWarpNode(NodeType):
    id = "DIR_WARP"
    label = "方向扭曲"
    category = "FILTER"
    description = "按扭曲图的明暗，把图案朝一个方向推"
    inputs = (("input", "输入", MID, False), ("warp", "扭曲图", MID, True))
    output_kind = "input"

    class Params(_Params):
        angle = FloatProperty("方向", default=0.0, min=-180.0, max=180.0, unit="°", precision=0)
        intensity = FloatProperty("强度", default=0.05, min=0.0, max=1.0, precision=3)

    glsl = """
vec4 node(vec2 uv) {
  float a = radians(p_angle);
  return IN0(uv + vec2(cos(a), sin(a)) * (G1(uv) - 0.5) * p_intensity);
}"""


@register
class TransformNode(NodeType):
    id = "TRANSFORM"
    label = "变换"
    category = "FILTER"
    description = "重复、旋转、平移"
    inputs = (("input", "输入", MID, False),)
    output_kind = "input"

    class Params(_Params):
        tile = IntProperty("重复", default=1, min=1, max=256, description="横竖各重复几次")
        rotation = FloatProperty("旋转", default=0.0, min=-180.0, max=180.0, unit="°", precision=0)
        offset_u = FloatProperty("横移", default=0.0, min=-1.0, max=1.0, precision=3)
        offset_v = FloatProperty("纵移", default=0.0, min=-1.0, max=1.0, precision=3)

    glsl = """
vec4 node(vec2 uv) {
  float k = float(max(p_tile, 1));
  vec2 p = rotate2(uv - 0.5, -p_rotation) * k + 0.5 - vec2(p_offset_u, p_offset_v);
  return IN0L(p, log2(k));
}"""


@register
class EdgeNode(NodeType):
    id = "EDGE"
    label = "描边"
    category = "FILTER"
    description = "找出明暗变化的地方（边缘）"
    inputs = (("input", "输入", MID, True),)

    class Params(_Params):
        width = FloatProperty("宽度", default=2.0, min=0.5, max=64.0, precision=1, unit="px")
        contrast = FloatProperty("强度", default=4.0, min=0.0, max=100.0, soft_max=20.0, precision=1)

    glsl = """
vec4 node(vec2 uv) {
  float e = p_width / u_size.x;
  float gx = G0(uv + vec2(e, 0.0)) - G0(uv - vec2(e, 0.0));
  float gy = G0(uv + vec2(0.0, e)) - G0(uv - vec2(0.0, e));
  return gray(clamp(length(vec2(gx, gy)) * p_contrast, 0.0, 1.0));
}"""


# ====================================================================== 输出
@register
class OutputNode(NodeType):
    id = "OUTPUT"
    label = "材质输出"
    category = "OUTPUT"
    description = "这张图作为材质用时，各通道取自这里"
    inputs = (("basecolor", "基础色", (0.5, 0.5, 0.5, 1.0), False), ("metallic", "金属度", BLACK, True),
              ("roughness", "粗糙度", MID, True), ("height", "高度", MID, True))
    output_kind = "none"
    Params = _Params
    glsl = ""
