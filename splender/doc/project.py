"""文档模型：工程、纹理集、图层、笔刷设置。

这里只有数据和信号，不碰显卡。界面改这里的数据，引擎监听这里的信号。
"""
from __future__ import annotations

import itertools
import logging
from typing import Any

from ..core.props import (BoolProperty, ColorProperty, CurveProperty, EnumProperty, FloatProperty, IntProperty,
                          PointerProperty, PropertyGroup, RampProperty, StringProperty)
from ..core.signals import Signal

log = logging.getLogger("splender.doc")

# ---- 页面种类（一个图层的一类数据存成一种页）----
PLANE_COLOR = 0      # RGBA8：基础色 RGB + 覆盖度
PLANE_MR = 1         # RGBA8：金属度、金属度覆盖度、粗糙度、粗糙度覆盖度
PLANE_HEIGHT = 2     # RG16F：高度、覆盖度
PLANE_MASK = 3       # R8：图层蒙版
PLANE_STROKE = 4     # RG16F：进行中的笔划（累积覆盖度、批次标记），只存在于引擎里
PLANE_NAMES = {PLANE_COLOR: "color", PLANE_MR: "mr", PLANE_HEIGHT: "height", PLANE_MASK: "mask",
               PLANE_STROKE: "stroke"}
PLANE_FORMAT = {PLANE_COLOR: "rgba8", PLANE_MR: "rgba8", PLANE_HEIGHT: "rg16f", PLANE_MASK: "r8",
                PLANE_STROKE: "rg16f"}
CONTENT_PLANES = (PLANE_COLOR, PLANE_MR, PLANE_HEIGHT)

# ---- 通道 ----
CHANNELS = [
    ("basecolor", "基础色", PLANE_COLOR),
    ("metallic", "金属度", PLANE_MR),
    ("roughness", "粗糙度", PLANE_MR),
    ("height", "高度", PLANE_HEIGHT),
]
CHANNEL_IDS = [c[0] for c in CHANNELS]
CHANNEL_LABELS = {c[0]: c[1] for c in CHANNELS}
# 导出的贴图：除了画的通道，还有合成出来的法线和烘焙的环境遮蔽
EXPORT_ITEMS = [("basecolor", "基础色"), ("metallic", "金属度"), ("roughness", "粗糙度"), ("height", "高度"),
                ("normal", "法线"), ("ao", "环境遮蔽")]
EXPORT_IDS = [c[0] for c in EXPORT_ITEMS]
MESH_FORMAT_ITEMS = [("GLB", "glTF 二进制 (.glb)", "一个文件，基础色、法线和遮蔽粗糙度金属度打包贴图都装在里面"),
                     ("OBJ", "OBJ", "模型和材质文件（.mtl），材质文件引用导出的 PNG")]
EXPORT_LABELS = dict(EXPORT_ITEMS)
EXPORT_SIZE_ITEMS = [("0", "原始尺寸", "和纹理集一样大"), ("8192", "8K", ""), ("4096", "4K", ""), ("2048", "2K", ""),
                     ("1024", "1K", ""), ("512", "512", "")]

BLEND_ITEMS = [
    ("NORMAL", "正常", "直接覆盖"),
    ("MULTIPLY", "正片叠底", "变暗"),
    ("SCREEN", "滤色", "变亮"),
    ("OVERLAY", "叠加", "加强对比"),
    ("ADD", "相加", "数值相加，高度常用"),
    ("SUBTRACT", "相减", "数值相减"),
    ("DARKEN", "变暗", "取较暗的"),
    ("LIGHTEN", "变亮", "取较亮的"),
]
BLEND_INDEX = {item[0]: i for i, item in enumerate(BLEND_ITEMS)}

RESOLUTION_ITEMS = [("256", "256", ""), ("512", "512", ""), ("1024", "1K", ""), ("2048", "2K", ""),
                    ("4096", "4K", ""), ("8192", "8K", ""), ("16384", "16K", "")]

# ---- 生成器蒙版：按模型贴图（曲率、遮蔽、厚度、法线、位置）自动算出的蒙版 ----
GENERATOR_ITEMS = [
    ("NONE", "无", "不用生成器"),
    ("EDGES", "边缘", "凸起的棱角和边缘：磨损、掉漆、擦亮"),
    ("CAVITY", "凹陷", "凹进去的缝和沟：积灰、锈迹、污垢"),
    ("OCCLUSION", "遮蔽", "被周围挡住、光照不到的地方"),
    ("FACING", "朝向", "朝某个方向的面：朝上的积灰、积雪、苔藓"),
    ("THIN", "薄处", "模型薄的地方：透光、磨穿"),
    ("GRADIENT", "渐变", "沿一个方向从无到有：底部的泥、顶部的褪色"),
    ("NOISE", "斑驳", "只有随机纹理"),
    ("PARTS", "部件", "只盖住选中的部件：模型的各个零件、UV 岛或高模的零件"),
]
GENERATOR_CODE = {item[0]: index for index, item in enumerate(GENERATOR_ITEMS)}
DIRECTION_ITEMS = [("UP", "上", "朝上（+Y）"), ("DOWN", "下", "朝下（-Y）"), ("FRONT", "前", "朝前（+Z）"),
                   ("BACK", "后", "朝后（-Z）"), ("RIGHT", "右", "朝右（+X）"), ("LEFT", "左", "朝左（-X）")]
DIRECTION_VECTOR = {"UP": (0.0, 1.0, 0.0), "DOWN": (0.0, -1.0, 0.0), "FRONT": (0.0, 0.0, 1.0),
                    "BACK": (0.0, 0.0, -1.0), "RIGHT": (1.0, 0.0, 0.0), "LEFT": (-1.0, 0.0, 0.0)}
MESHMAP_RESOLUTION_ITEMS = [("1024", "1K", ""), ("2048", "2K", ""), ("4096", "4K", ""), ("8192", "8K", ""),
                            ("16384", "16K", "")]

# ---- 图层种类 ----
LAYER_KIND_ITEMS = [("PAINT", "绘制层", "用笔刷画的内容"), ("FILL", "填充层", "整层统一的数值"),
                    ("FOLDER", "文件夹", "把几层合成一组，整组一起调不透明度、混合和蒙版"),
                    ("ADJUST", "调整层", "调整它下面（同一文件夹内）合成好的结果")]
LAYER_KIND_CODE = {"PAINT": 0, "FILL": 1, "FOLDER": 2, "ADJUST": 3}
#: 文件夹最多套几层（合成着色器按这个开累加栈）
MAX_FOLDER_DEPTH = 4

# ---- 调整层（顺序和 Photoshop 的调整菜单一样）----
ADJUST_ITEMS = [
    ("BRIGHTNESS", "亮度对比度", "整体提亮压暗、拉开或压缩反差"),
    ("LEVELS", "色阶", "输入黑白场、灰度系数、输出黑白场"),
    ("CURVES", "曲线", "拖曲线改明暗和反差，也可以分红、绿、蓝单独调"),
    ("EXPOSURE", "曝光度", "像相机加减曝光：按挡提亮压暗，还能调位移和灰度系数"),
    ("VIBRANCE", "自然饱和度", "先提不太艳的颜色，已经很艳的少动"),
    ("HSV", "色相饱和度", "转色相、调饱和度和明度，也可以整体着一种颜色"),
    ("COLOR_BALANCE", "色彩平衡", "让阴影、中间调、高光分别偏青或红、偏洋红或绿、偏黄或蓝"),
    ("BLACK_WHITE", "黑白", "转成黑白，原来每种颜色变得多亮可以单独调，还能整体染一种颜色"),
    ("PHOTO_FILTER", "照片滤镜", "像在镜头前加一片有色滤镜"),
    ("CHANNEL_MIXER", "通道混合器", "每个输出通道由红、绿、蓝按比例混出来"),
    ("COLOR_LOOKUP", "颜色查找", "按查找表整体换一种色调风格，可以导入 .cube 文件"),
    ("INVERT", "反相", "数值和颜色取反"),
    ("POSTERIZE", "色调分离", "把连续的明暗压成几档"),
    ("THRESHOLD", "阈值", "按亮度分成黑白两档"),
    ("GRADIENT_MAP", "渐变映射", "按明暗从色带上取颜色：最暗取左端，最亮取右端"),
    ("SELECTIVE_COLOR", "可选颜色", "只改某一类颜色里青、洋红、黄、黑的多少"),
]
ADJUST_CODE = {item[0]: index for index, item in enumerate(ADJUST_ITEMS)}
#: 调整层菜单在这几项后面加分隔线（和 Photoshop 的分组一样）
ADJUST_MENU_BREAKS = ("EXPOSURE", "COLOR_LOOKUP")
#: 颜色查找的内置风格（最后一项是导入的 .cube 文件）
LOOK_ITEMS = [("WARM_FILM", "暖调胶片", "高光偏暖、黑色微微抬起、反差柔和"),
              ("TEAL_ORANGE", "青橙", "阴影偏青、高光偏橙，电影常用的配色"),
              ("FADED", "褪色", "黑色发灰、颜色变淡，像放旧了的照片"),
              ("COLD", "冷峻", "整体偏冷、反差更硬"),
              ("VINTAGE", "复古", "泛黄的褪色暖调"),
              ("BLEACH", "漂白", "颜色变淡、反差变强，金属感的冷硬画面"),
              ("NIGHT", "夜色", "偏蓝偏暗，像月光下"),
              ("CROSS", "交叉冲印", "高光泛黄绿、阴影泛蓝紫的强烈色偏"),
              ("FILE", "导入的文件", "从 .cube 文件读进来的查找表")]
CURVE_CHANNEL_ITEMS = [("RGB", "RGB", "三个通道一起改"), ("R", "红", "只改红通道"), ("G", "绿", "只改绿通道"),
                       ("B", "蓝", "只改蓝通道")]
#: 曲线属性按通道
CURVE_PROPS = {"RGB": "adj_curve", "R": "adj_curve_r", "G": "adj_curve_g", "B": "adj_curve_b"}
TONE_ITEMS = [("SHADOWS", "阴影", "暗的部分"), ("MIDTONES", "中间调", "不暗不亮的部分"), ("HIGHLIGHTS", "高光", "亮的部分")]
MIX_OUTPUT_ITEMS = [("R", "红", "输出的红通道"), ("G", "绿", "输出的绿通道"), ("B", "蓝", "输出的蓝通道")]
SELECTIVE_RANGES = [("REDS", "红色", "偏红的颜色"), ("YELLOWS", "黄色", "偏黄的颜色"), ("GREENS", "绿色", "偏绿的颜色"),
                    ("CYANS", "青色", "偏青的颜色"), ("BLUES", "蓝色", "偏蓝的颜色"), ("MAGENTAS", "洋红", "偏洋红的颜色"),
                    ("WHITES", "白色", "很亮的颜色"), ("NEUTRALS", "中性色", "不太艳、不太亮也不太暗的颜色"),
                    ("BLACKS", "黑色", "很暗的颜色")]
SELECTIVE_INKS = [("c", "青色", "负数少一些青（偏红），正数多一些青"),
                  ("m", "洋红", "负数少一些洋红（偏绿），正数多一些洋红"),
                  ("y", "黄色", "负数少一些黄（偏蓝），正数多一些黄"),
                  ("k", "黑色", "负数变亮，正数变暗")]
SELECTIVE_METHOD_ITEMS = [("RELATIVE", "相对", "按原来有多少按比例加减"), ("ABSOLUTE", "绝对", "直接加减")]
#: 色阶的五个数（名字后缀、标题、默认值）；红、绿、蓝各一套：adj_lv_r_in_black 这样，RGB 用 adj_in_black
LEVEL_FIELDS = [("in_black", "输入黑场", 0.0), ("in_white", "输入白场", 1.0), ("gamma", "灰度系数", 1.0),
                ("out_black", "输出黑场", 0.0), ("out_white", "输出白场", 1.0)]
#: 色相饱和度里能单独改的六类颜色（正中的色相）
HUE_RANGES = [("REDS", "红色", 0.0), ("YELLOWS", "黄色", 60.0), ("GREENS", "绿色", 120.0), ("CYANS", "青色", 180.0),
              ("BLUES", "蓝色", 240.0), ("MAGENTAS", "洋红", 300.0)]
HUE_EDIT_ITEMS = [("MASTER", "全图", "所有颜色一起改")] + [(key, label, "只改偏%s的颜色" % label.rstrip("色"))
                                                         for key, label, _center in HUE_RANGES]


def level_prop_names(channel: str) -> list[str]:
    """某个通道（RGB / R / G / B）色阶的五个属性名。"""
    if channel == "RGB":
        return ["adj_" + field for field, _label, _default in LEVEL_FIELDS]
    return ["adj_lv_%s_%s" % (channel.lower(), field) for field, _label, _default in LEVEL_FIELDS]

_uid = itertools.count(1)


def new_uid() -> int:
    return next(_uid)


def reserve_uid(value: int) -> None:
    """读取工程后把计数器推到已用编号之后。"""
    global _uid
    current = next(_uid)
    _uid = itertools.count(max(current, int(value) + 1))


class Layer(PropertyGroup):
    """一个图层。uid 稳定不变，页面数据以 uid 归属。"""

    undoable = True

    name = StringProperty("名称", default="图层")
    kind = EnumProperty("类型", items=LAYER_KIND_ITEMS, default="PAINT")
    parent_uid = IntProperty("所属文件夹", default=0, hidden=True)
    visible = BoolProperty("可见", default=True)
    locked = BoolProperty("锁定", default=False, description="锁定后不能在这层上绘制")
    opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2)

    blend_basecolor = EnumProperty("基础色混合", items=BLEND_ITEMS, default="NORMAL")
    blend_metallic = EnumProperty("金属度混合", items=BLEND_ITEMS, default="NORMAL")
    blend_roughness = EnumProperty("粗糙度混合", items=BLEND_ITEMS, default="NORMAL")
    blend_height = EnumProperty("高度混合", items=BLEND_ITEMS, default="ADD")

    opacity_basecolor = FloatProperty("基础色强度", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    opacity_metallic = FloatProperty("金属度强度", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    opacity_roughness = FloatProperty("粗糙度强度", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    opacity_height = FloatProperty("高度强度", default=1.0, min=0.0, max=1.0, subtype="FACTOR")

    # 填充层的数值
    use_basecolor = BoolProperty("基础色", default=True)
    use_metallic = BoolProperty("金属度", default=True)
    use_roughness = BoolProperty("粗糙度", default=True)
    use_height = BoolProperty("高度", default=False)
    fill_color = ColorProperty("颜色", default=(0.8, 0.8, 0.8))
    fill_metallic = FloatProperty("金属度", default=0.0, min=0.0, max=1.0, subtype="FACTOR")
    fill_roughness = FloatProperty("粗糙度", default=0.5, min=0.0, max=1.0, subtype="FACTOR")
    fill_height = FloatProperty("高度", default=0.0, min=-1.0, max=1.0, precision=3)
    # 填充内容来自节点图（0 表示不用，用上面的固定值）
    fill_graph = IntProperty("节点图", default=0, hidden=True)
    graph_tiling = FloatProperty("平铺次数", default=1.0, min=0.01, max=1000.0, soft_max=64.0, precision=2,
                                 description="节点图在 UV 上横竖各重复几次")
    graph_offset_u = FloatProperty("横移", default=0.0, min=-1.0, max=1.0, precision=3)
    graph_offset_v = FloatProperty("纵移", default=0.0, min=-1.0, max=1.0, precision=3)
    graph_height = FloatProperty("高度强度", default=1.0, min=0.0, max=10.0, precision=2,
                                 description="节点图里的高度起伏有多大")

    has_mask = BoolProperty("有蒙版", default=False)
    mask_enabled = BoolProperty("启用蒙版", default=True, description="关闭后暂时不受蒙版影响")
    mask_default = FloatProperty("蒙版初始值", default=1.0, min=0.0, max=1.0, hidden=True)

    # 生成器蒙版（和画出来的蒙版相乘）
    mask_generator = EnumProperty("生成器", items=GENERATOR_ITEMS, default="NONE",
                                  description="按模型的形状自动生成蒙版，需要先烘焙模型贴图")
    gen_range = FloatProperty("范围", default=0.5, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                              description="越大盖住的地方越多")
    gen_softness = FloatProperty("过渡", default=0.3, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                                 description="边界的软硬，越大越柔和")
    gen_noise = FloatProperty("斑驳", default=0.3, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                              description="用随机纹理打乱边界，越大越不规则")
    gen_noise_size = FloatProperty("斑驳大小", default=0.05, min=0.001, max=1.0, precision=3,
                                   description="随机纹理颗粒的大小，按模型尺寸算")
    gen_seed = IntProperty("随机种子", default=0, min=0, max=9999, description="换一个数，随机纹理就换一种")
    gen_direction = EnumProperty("方向", items=DIRECTION_ITEMS, default="UP", description="「朝向」和「渐变」用的方向")
    gen_invert = BoolProperty("反转", default=False, description="盖住和露出的地方对调")
    gen_parts = StringProperty("选中的部件", default="", hidden=True,
                               description="「部件」生成器盖住的部件编号，逗号隔开")

    # 调整层
    adjust_type = EnumProperty("调整", items=ADJUST_ITEMS, default="HSV")
    adj_hue = FloatProperty("色相", default=0.0, min=-180.0, max=180.0, unit="°", precision=0)
    adj_saturation = FloatProperty("饱和度", default=0.0, min=-1.0, max=1.0, precision=2,
                                   description="负数变灰，正数更艳")
    adj_value = FloatProperty("明度", default=0.0, min=-1.0, max=1.0, precision=2)
    adj_brightness = FloatProperty("亮度", default=0.0, min=-1.0, max=1.0, precision=2)
    adj_contrast = FloatProperty("对比度", default=0.0, min=-1.0, max=1.0, precision=2)
    adj_in_black = FloatProperty("输入黑场", default=0.0, min=0.0, max=1.0, subtype="FACTOR", precision=3)
    adj_in_white = FloatProperty("输入白场", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=3)
    adj_gamma = FloatProperty("灰度系数", default=1.0, min=0.1, max=10.0, precision=2,
                              description="大于 1 中间调变亮，小于 1 变暗")
    adj_out_black = FloatProperty("输出黑场", default=0.0, min=0.0, max=1.0, subtype="FACTOR", precision=3)
    adj_out_white = FloatProperty("输出白场", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=3)
    adj_threshold = FloatProperty("阈值", default=0.5, min=0.0, max=1.0, subtype="FACTOR", precision=3)
    adj_levels_channel = EnumProperty("通道", items=CURVE_CHANNEL_ITEMS, default="RGB", description="现在改的是哪个通道的色阶")
    adj_hue_edit = EnumProperty("编辑", items=HUE_EDIT_ITEMS, default="MASTER", description="改所有颜色，还是只改某一类颜色")
    adj_colorize = BoolProperty("着色", default=False, description="整体换成一种色相，只留明暗变化")
    adj_colorize_hue = FloatProperty("色相", default=30.0, min=0.0, max=360.0, unit="°", precision=0,
                                     description="着成什么色相")
    adj_colorize_sat = FloatProperty("饱和度", default=0.25, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                                     description="着的颜色有多艳")
    # 曲线：先过红、绿、蓝各自的曲线，再过 RGB 总曲线；金属度、粗糙度、高度只过总曲线
    adj_curve_channel = EnumProperty("通道", items=CURVE_CHANNEL_ITEMS, default="RGB", description="现在改的是哪条曲线")
    adj_curve = CurveProperty("RGB 曲线", subtype="RGB",
                              description="横向是原来的明暗，竖向是改成的明暗。点空白处加点，拖点改形状，右键或 Delete 删点")
    adj_curve_r = CurveProperty("红曲线", subtype="R", description="只改红通道。点空白处加点，拖点改形状，右键或 Delete 删点")
    adj_curve_g = CurveProperty("绿曲线", subtype="G", description="只改绿通道。点空白处加点，拖点改形状，右键或 Delete 删点")
    adj_curve_b = CurveProperty("蓝曲线", subtype="B", description="只改蓝通道。点空白处加点，拖点改形状，右键或 Delete 删点")
    # 渐变映射
    adj_ramp = RampProperty("渐变", description="最暗的地方取左端的颜色，最亮的地方取右端；色标越透明，那一段改得越少")
    adj_ramp_reverse = BoolProperty("反向", default=False, description="色带左右对调着用")
    # 色彩平衡：负数偏青、洋红、黄，正数偏红、绿、蓝
    adj_cb_tone = EnumProperty("色调", items=TONE_ITEMS, default="MIDTONES", description="下面三条改的是哪一段明暗")
    adj_cb_shadows_r = FloatProperty("青色 / 红色", default=0.0, min=-1.0, max=1.0, precision=2,
                                     description="阴影偏青还是偏红")
    adj_cb_shadows_g = FloatProperty("洋红 / 绿色", default=0.0, min=-1.0, max=1.0, precision=2,
                                     description="阴影偏洋红还是偏绿")
    adj_cb_shadows_b = FloatProperty("黄色 / 蓝色", default=0.0, min=-1.0, max=1.0, precision=2,
                                     description="阴影偏黄还是偏蓝")
    adj_cb_midtones_r = FloatProperty("青色 / 红色", default=0.0, min=-1.0, max=1.0, precision=2,
                                      description="中间调偏青还是偏红")
    adj_cb_midtones_g = FloatProperty("洋红 / 绿色", default=0.0, min=-1.0, max=1.0, precision=2,
                                      description="中间调偏洋红还是偏绿")
    adj_cb_midtones_b = FloatProperty("黄色 / 蓝色", default=0.0, min=-1.0, max=1.0, precision=2,
                                      description="中间调偏黄还是偏蓝")
    adj_cb_highlights_r = FloatProperty("青色 / 红色", default=0.0, min=-1.0, max=1.0, precision=2,
                                        description="高光偏青还是偏红")
    adj_cb_highlights_g = FloatProperty("洋红 / 绿色", default=0.0, min=-1.0, max=1.0, precision=2,
                                        description="高光偏洋红还是偏绿")
    adj_cb_highlights_b = FloatProperty("黄色 / 蓝色", default=0.0, min=-1.0, max=1.0, precision=2,
                                        description="高光偏黄还是偏蓝")
    adj_cb_preserve = BoolProperty("保持明度", default=True, description="只改颜色，明暗不变")
    # 色调分离
    adj_posterize = IntProperty("色阶数", default=4, min=2, max=255, description="压成几档")
    # 自然饱和度
    adj_vibrance = FloatProperty("自然饱和度", default=0.0, min=-1.0, max=1.0, precision=2,
                                 description="正数先提不太艳的颜色，负数先压不太艳的颜色；很艳的颜色少动")
    adj_vib_saturation = FloatProperty("饱和度", default=0.0, min=-1.0, max=1.0, precision=2,
                                       description="所有颜色一样多地变艳或变灰")
    # 照片滤镜
    adj_filter_color = ColorProperty("颜色", default=(236 / 255.0, 138 / 255.0, 0.0), description="滤镜的颜色")
    adj_filter_density = FloatProperty("浓度", default=0.25, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                                       description="滤镜颜色有多浓")
    adj_filter_preserve = BoolProperty("保持明度", default=True, description="只改颜色，画面不跟着变暗")
    # 曝光度
    adj_exposure = FloatProperty("曝光度", default=0.0, min=-20.0, max=20.0, soft_min=-5.0, soft_max=5.0, precision=2,
                                 description="每加 1 亮一倍，每减 1 暗一半")
    adj_exp_offset = FloatProperty("位移", default=0.0, min=-0.5, max=0.5, precision=4,
                                   description="整体加减一个数：主要改暗部，亮部变化很小")
    adj_exp_gamma = FloatProperty("灰度系数校正", default=1.0, min=0.01, max=9.99, precision=2,
                                  description="大于 1 中间调变亮，小于 1 变暗")
    # 黑白：原来是这种颜色的地方变得多亮
    adj_bw_red = FloatProperty("红色", default=0.4, min=-2.0, max=3.0, precision=2, description="原来是红色的地方变得多亮")
    adj_bw_yellow = FloatProperty("黄色", default=0.6, min=-2.0, max=3.0, precision=2,
                                  description="原来是黄色的地方变得多亮")
    adj_bw_green = FloatProperty("绿色", default=0.4, min=-2.0, max=3.0, precision=2,
                                 description="原来是绿色的地方变得多亮")
    adj_bw_cyan = FloatProperty("青色", default=0.6, min=-2.0, max=3.0, precision=2, description="原来是青色的地方变得多亮")
    adj_bw_blue = FloatProperty("蓝色", default=0.2, min=-2.0, max=3.0, precision=2, description="原来是蓝色的地方变得多亮")
    adj_bw_magenta = FloatProperty("洋红", default=0.8, min=-2.0, max=3.0, precision=2,
                                   description="原来是洋红的地方变得多亮")
    adj_bw_tint = BoolProperty("色调", default=False, description="把黑白结果整体染成一种颜色")
    adj_bw_tint_color = ColorProperty("颜色", default=(0.6, 0.54, 0.4), description="染成什么颜色，明暗还按黑白的结果")
    # 通道混合器：输出的每个通道 = 红 × 比例 + 绿 × 比例 + 蓝 × 比例 + 常数
    adj_mix_output = EnumProperty("输出通道", items=MIX_OUTPUT_ITEMS, default="R", description="现在改的是哪个输出通道")
    adj_mix_mono = BoolProperty("单色", default=False, description="三个通道用同一组比例，结果是黑白的")
    adj_mix_rr = FloatProperty("红色", default=1.0, min=-2.0, max=2.0, precision=2, description="原来的红通道混进来多少")
    adj_mix_rg = FloatProperty("绿色", default=0.0, min=-2.0, max=2.0, precision=2, description="原来的绿通道混进来多少")
    adj_mix_rb = FloatProperty("蓝色", default=0.0, min=-2.0, max=2.0, precision=2, description="原来的蓝通道混进来多少")
    adj_mix_rc = FloatProperty("常数", default=0.0, min=-2.0, max=2.0, precision=2, description="再整体加减多少")
    adj_mix_gr = FloatProperty("红色", default=0.0, min=-2.0, max=2.0, precision=2, description="原来的红通道混进来多少")
    adj_mix_gg = FloatProperty("绿色", default=1.0, min=-2.0, max=2.0, precision=2, description="原来的绿通道混进来多少")
    adj_mix_gb = FloatProperty("蓝色", default=0.0, min=-2.0, max=2.0, precision=2, description="原来的蓝通道混进来多少")
    adj_mix_gc = FloatProperty("常数", default=0.0, min=-2.0, max=2.0, precision=2, description="再整体加减多少")
    adj_mix_br = FloatProperty("红色", default=0.0, min=-2.0, max=2.0, precision=2, description="原来的红通道混进来多少")
    adj_mix_bg = FloatProperty("绿色", default=0.0, min=-2.0, max=2.0, precision=2, description="原来的绿通道混进来多少")
    adj_mix_bb = FloatProperty("蓝色", default=1.0, min=-2.0, max=2.0, precision=2, description="原来的蓝通道混进来多少")
    adj_mix_bc = FloatProperty("常数", default=0.0, min=-2.0, max=2.0, precision=2, description="再整体加减多少")
    # 颜色查找
    adj_lut_look = EnumProperty("查找表", items=LOOK_ITEMS, default="WARM_FILM", description="换成哪一种色调风格")
    adj_lut_name = StringProperty("文件", default="", description="导入的查找表文件名")
    adj_lut_data = StringProperty("查找表数据", default="", hidden=True)
    # 可选颜色（每类颜色的青、洋红、黄、黑在类定义后面按表加上：adj_sel_reds_c 这样）
    adj_sel_range = EnumProperty("颜色", items=SELECTIVE_RANGES, default="REDS", description="现在改的是哪一类颜色")
    adj_sel_method = EnumProperty("方法", items=SELECTIVE_METHOD_ITEMS, default="RELATIVE")

    def __init__(self, uid: int | None = None, **values: Any) -> None:
        super().__init__(**values)
        self.uid = uid if uid is not None else new_uid()

    def blend(self, channel: str) -> str:
        return getattr(self, "blend_" + channel)

    def channel_opacity(self, channel: str) -> float:
        return getattr(self, "opacity_" + channel)

    def to_dict(self) -> dict:
        data = super().to_dict()
        if self.kind != "ADJUST":
            # 调整参数只对调整层有意义，别的图层不存
            data = {key: value for key, value in data.items() if not key.startswith("adj")}
        data["uid"] = self.uid
        return data

    @classmethod
    def from_saved(cls, data: dict) -> "Layer":
        layer = cls(uid=int(data.get("uid", 0)) or None)
        layer.from_dict(data)
        reserve_uid(layer.uid)
        return layer


def _add_generated_props() -> None:
    """成套的调整参数按表加到 Layer 上：可选颜色（九类颜色 × 青洋红黄黑）、分通道色阶、分颜色的色相饱和度。"""
    def add(name: str, prop) -> None:
        prop.__set_name__(Layer, name)
        setattr(Layer, name, prop)

    for key, label, _desc in SELECTIVE_RANGES:
        for ink, ink_label, ink_desc in SELECTIVE_INKS:
            add("adj_sel_%s_%s" % (key.lower(), ink),
                FloatProperty(ink_label, default=0.0, min=-1.0, max=1.0, precision=2,
                              description="%s：%s" % (label, ink_desc)))
    for channel, channel_label in (("r", "红"), ("g", "绿"), ("b", "蓝")):
        for field, label, default in LEVEL_FIELDS:
            if field == "gamma":
                prop = FloatProperty(label, default=default, min=0.1, max=10.0, precision=2,
                                     description="%s通道：大于 1 中间调变亮，小于 1 变暗" % channel_label)
            else:
                prop = FloatProperty(label, default=default, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                                     description="%s通道的%s" % (channel_label, label))
            add("adj_lv_%s_%s" % (channel, field), prop)
    for key, label, _center in HUE_RANGES:
        prefix = "adj_hsv_%s_" % key.lower()
        add(prefix + "hue", FloatProperty("色相", default=0.0, min=-180.0, max=180.0, unit="°", precision=0,
                                          description="偏%s的颜色转多少色相" % label.rstrip("色")))
        add(prefix + "sat", FloatProperty("饱和度", default=0.0, min=-1.0, max=1.0, precision=2,
                                          description="偏%s的颜色变艳还是变灰" % label.rstrip("色")))
        add(prefix + "light", FloatProperty("明度", default=0.0, min=-1.0, max=1.0, precision=2,
                                            description="偏%s的颜色变亮还是变暗" % label.rstrip("色")))
    if "_props_cache" in Layer.__dict__:
        del Layer._props_cache


_add_generated_props()


class MeshMapSettings(PropertyGroup):
    """模型贴图的烘焙设置。"""

    resolution = EnumProperty("分辨率", items=MESHMAP_RESOLUTION_ITEMS, default="2048",
                              description="遮蔽、曲率、厚度、法线贴图的大小。生成器蒙版按它取样，随机纹理不受影响")
    smooth_resolution = EnumProperty("位置/朝向/部件最大", items=MESHMAP_RESOLUTION_ITEMS, default="4096",
                                     description="世界位置、世界朝向、部件这三张变化平缓（部件只看边界）的贴图最多做多大。"
                                                 "比上面的分辨率小时省很多显存：16K 时从约 8 GB 降到约 3 GB")
    ao_samples = IntProperty("遮蔽采样", default=64, min=4, max=1024,
                             description="每个像素发多少条光线算遮蔽。越多越干净，也越慢")
    ao_distance = FloatProperty("遮蔽距离", default=0.15, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                                description="多远以内的遮挡才算数，按模型尺寸算")
    bake_thickness = BoolProperty("烘焙厚度", default=True, description="量出模型各处有多厚，给「薄处」生成器用")
    thickness_distance = FloatProperty("厚度距离", default=0.25, min=0.001, max=1.0, subtype="FACTOR", precision=3,
                                       description="比这更厚的地方都算最厚，按模型尺寸算")
    curvature_smooth = IntProperty("曲率平滑", default=2, min=0, max=200,
                                   description="越大曲率越平滑，只留下大的起伏")
    curvature_contrast = FloatProperty("曲率对比", default=1.0, min=0.1, max=10.0, precision=2,
                                       description="越大，边缘和凹陷越明显")
    self_only = BoolProperty("只算自身遮挡", default=False,
                             description="只让这套贴图对应的部分互相遮挡，不算别的模型")
    padding = IntProperty("扩边", default=16, min=0, max=256, unit="px",
                          description="接缝外面延伸多少像素，缩小看时接缝不露底色")
    show_ao = BoolProperty("显示遮蔽", default=True, description="在视口里把环境遮蔽叠到光照上")
    ao_strength = FloatProperty("遮蔽强度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2)
    auto_rebake = BoolProperty("形状改了自动重烘", default=True,
                               description="雕刻、重构之后回到绘制模式时，自动重新烘焙模型贴图")
    high_poly_object = EnumProperty("高模（场景里的）", items=lambda owner: _scene_mesh_items(owner), default="0",
                                    description="用场景里的一个模型当高模（比如雕刻好的那个，可以隐藏起来）。选了它就不用下面的文件")
    high_poly = StringProperty("高模", default="", subtype="FILE_PATH",
                               description="从这个模型文件（OBJ、glTF）烘焙细节：法线贴图、遮蔽、曲率、厚度都按它算。空着就只用当前模型")
    cage_front = FloatProperty("向外找", default=0.02, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                               description="从低模表面往外多远以内找高模，按模型尺寸算")
    cage_back = FloatProperty("向里找", default=0.02, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                              description="从低模表面往里多远以内找高模，按模型尺寸算")
    normal_samples = EnumProperty("法线抗锯齿", items=[("1", "关", ""), ("4", "2×2", ""), ("9", "3×3", ""),
                                                    ("16", "4×4", "")], default="4",
                                  description="每个像素取几个点平均，细节边缘更平滑")
    id_source = EnumProperty("部件划分", items=[("PART", "网格部件", "连在一起的面算一个部件"),
                                             ("UV_ISLAND", "UV 岛", "UV 上连在一起的面算一个部件"),
                                             ("HIGH_POLY", "高模部件", "高模里连在一起的面算一个部件")],
                             default="PART", description="「部件」生成器按什么把模型分成一块一块")
    normal_format = EnumProperty("法线格式", items=[("OPENGL", "OpenGL", "绿色朝上（Blender、Unity）"),
                                                  ("DIRECTX", "DirectX", "绿色朝下（虚幻、3ds Max）")],
                                 default="OPENGL", description="导出法线贴图时绿色通道的方向")


class ExportSettings(PropertyGroup):
    """导出贴图的设置：导哪些、多大、文件怎么命名，法线怎么合成。"""

    basecolor = BoolProperty("基础色", default=True)
    metallic = BoolProperty("金属度", default=True)
    roughness = BoolProperty("粗糙度", default=True)
    height = BoolProperty("高度", default=True, description="16 位灰度，中灰是零")
    normal = BoolProperty("法线", default=True, description="从高模烘的法线加上手绘高度的凹凸，合成一张")
    ao = BoolProperty("环境遮蔽", default=False, description="烘焙出的环境遮蔽")
    size = EnumProperty("尺寸", items=EXPORT_SIZE_ITEMS, default="0")
    name_pattern = StringProperty("文件名", default="{工程}_{纹理集}_{通道}",
                                  description="{工程}、{纹理集}、{通道} 会换成对应的名字")
    normal_bake = BoolProperty("含高模细节", default=True, description="把从高模烘焙的法线合进去")
    normal_strength = FloatProperty("高度凹凸强度", default=1.0, min=0.0, max=50.0, soft_max=4.0, precision=2,
                                    description="手绘高度变成法线时的强弱。1 和视口里看到的一样")
    normal_depth = EnumProperty("法线位深", items=[("8", "8 位", "通用"), ("16", "16 位", "渐变更平滑，文件更大")],
                                default="8")
    with_mesh = BoolProperty("连模型一起导出", default=False,
                             description="导出贴图时，把用到这套贴图的模型也导出到同一个文件夹")
    mesh_format = EnumProperty("模型格式", items=MESH_FORMAT_ITEMS, default="GLB")


class TextureSet(PropertyGroup):
    """一套贴图：对应模型的一个材质。图层自下而上排列。"""

    undoable = True

    name = StringProperty("名称", default="纹理集")
    resolution = EnumProperty("分辨率", items=RESOLUTION_ITEMS, default="16384")
    base_color = ColorProperty("底色", default=(0.5, 0.5, 0.5), description="没有任何图层覆盖时的基础色")
    base_metallic = FloatProperty("底金属度", default=0.0, min=0.0, max=1.0, subtype="FACTOR")
    base_roughness = FloatProperty("底粗糙度", default=0.5, min=0.0, max=1.0, subtype="FACTOR")
    height_scale = FloatProperty("高度强度", default=1.0, min=0.0, max=10.0, precision=2,
                                 description="高度通道在视口里凹凸的强弱")
    paint_target = EnumProperty("绘制目标", items=[("CONTENT", "内容", "画在图层的通道上"),
                                                ("MASK", "蒙版", "画在图层的蒙版上")], default="CONTENT")
    meshmap = PointerProperty("模型贴图", type=MeshMapSettings)
    export = PointerProperty("导出", type=ExportSettings)

    def __init__(self, uid: int | None = None, **values: Any) -> None:
        super().__init__(**values)
        self.uid = uid if uid is not None else new_uid()
        self.layers: list[Layer] = []
        self.active_layer_uid = 0
        self._watched: set[int] = set()
        #: (kind, layer|None)。kind: "structure" 增删排序；"props" 图层属性；"active" 当前层；"pixels" 像素；
        #: "material" 材质节点的结构；"material_values" 材质节点的数值
        self.layers_changed = Signal()
        #: 材质节点图（shading.graph.ShaderGraph）；None 表示直接用图层堆栈的结果
        self.material = None

    # ---- 查询 ----
    @property
    def size(self) -> int:
        return int(self.resolution)

    @property
    def active_layer(self) -> Layer | None:
        return self.layer(self.active_layer_uid)

    def layer(self, uid: int) -> Layer | None:
        for item in self.layers:
            if item.uid == uid:
                return item
        return None

    def index_of(self, layer: Layer) -> int:
        return self.layers.index(layer)

    # ---- 修改 ----
    def _watch(self, layer: Layer) -> None:
        if layer.uid in self._watched:
            return
        self._watched.add(layer.uid)
        layer.changed.connect(lambda name, target=layer: self.layers_changed.emit("props", target))

    # ---- 文件夹结构 ----
    # 约定：列表自下而上；文件夹排在它的内容上面，内容紧挨着它、连成一段：
    #   [..., 子1, 子2, 子文件夹的内容..., 子文件夹, 子3, 文件夹, ...]
    # 每层的 parent_uid 是直接所属的文件夹（0 表示最外层）。
    def parent_of(self, layer: Layer) -> Layer | None:
        return self.layer(layer.parent_uid) if layer.parent_uid else None

    def depth(self, layer: Layer) -> int:
        depth = 0
        parent = self.parent_of(layer)
        while parent is not None and depth < 64:
            depth += 1
            parent = self.parent_of(parent)
        return depth

    def is_ancestor(self, folder: Layer, layer: Layer) -> bool:
        """folder 是不是 layer 的上级（任意层）。"""
        parent = self.parent_of(layer)
        while parent is not None:
            if parent is folder:
                return True
            parent = self.parent_of(parent)
        return False

    def subtree_range(self, layer: Layer) -> tuple[int, int]:
        """layer 和它全部内容在列表里的范围 [start, end)，end 是 layer 自己的下标 + 1。"""
        end = self.layers.index(layer) + 1
        start = end - 1
        if layer.kind == "FOLDER":
            while start > 0 and self.is_ancestor(layer, self.layers[start - 1]):
                start -= 1
        return start, end

    def subtree(self, layer: Layer) -> list[Layer]:
        start, end = self.subtree_range(layer)
        return self.layers[start:end]

    def children(self, folder: Layer | None) -> list[Layer]:
        uid = folder.uid if folder is not None else 0
        return [item for item in self.layers if item.parent_uid == uid]

    def folder_depth_below(self, layer: Layer) -> int:
        """layer 自己往下还套着几层文件夹（普通图层为 0，空文件夹为 1）。"""
        if layer.kind != "FOLDER":
            return 0
        deepest = 1
        for item in self.subtree(layer)[:-1]:
            if item.kind == "FOLDER":
                deepest = max(deepest, self.depth(item) - self.depth(layer) + 1)
        return deepest

    def structure(self) -> tuple:
        """当前结构的快照（撤销用）：图层顺序、各自所属文件夹、当前层。"""
        return (tuple(self.layers), tuple(item.parent_uid for item in self.layers), self.active_layer_uid)

    def set_structure(self, snapshot: tuple) -> None:
        layers, parents, active = snapshot
        for item, parent in zip(layers, parents):
            if item.parent_uid != parent:
                item.parent_uid = parent
        self.layers[:] = list(layers)
        for item in self.layers:
            self._watch(item)
        self.active_layer_uid = active if self.layer(active) is not None else (self.layers[-1].uid if self.layers else 0)
        self.layers_changed.emit("structure", None)

    def move_subtree(self, layer: Layer, parent_uid: int, index: int) -> None:
        """把 layer（文件夹连同内容）挪到 parent_uid 文件夹里、列表下标 index 处（按挪走之前的下标算）。"""
        start, end = self.subtree_range(layer)
        block = self.layers[start:end]
        if start <= index <= end:
            index = start                           # 落点就在自己身上：位置不变，只换所属
        elif index > end:
            index -= end - start
        del self.layers[start:end]
        self.layers[index:index] = block
        layer.parent_uid = parent_uid
        self.layers_changed.emit("structure", layer)

    def repair_structure(self) -> None:
        """读入的工程结构不合法（引用了不存在的文件夹、内容不连续）时修正：找不到的上级归到最外层，然后按树重排。"""
        uids = {item.uid for item in self.layers if item.kind == "FOLDER"}
        for item in self.layers:
            if item.parent_uid and item.parent_uid not in uids:
                item.parent_uid = 0
        ordered: list[Layer] = []

        def emit(parent_uid: int, seen: set) -> None:
            for item in [x for x in self.layers if x.parent_uid == parent_uid]:
                if item.uid in seen:
                    continue
                seen.add(item.uid)
                if item.kind == "FOLDER":
                    emit(item.uid, seen)
                ordered.append(item)

        emit(0, set())
        for item in self.layers:                    # 成环等异常：剩下的放到最外层
            if item not in ordered:
                item.parent_uid = 0
                ordered.append(item)
        self.layers[:] = ordered

    def add_layer(self, layer: Layer, index: int | None = None) -> Layer:
        """新图层默认放在当前层上面、和当前层同一个文件夹里（当前层是文件夹时放在文件夹上面）。"""
        if index is None:
            active = self.active_layer
            if active in self.layers:
                index = self.layers.index(active) + 1
                layer.parent_uid = active.parent_uid
            else:
                index = len(self.layers)
        self.layers.insert(index, layer)
        self._watch(layer)
        self.active_layer_uid = layer.uid
        self.layers_changed.emit("structure", layer)
        return layer

    def remove_layer(self, layer: Layer) -> int:
        index = self.layers.index(layer)
        self.layers.pop(index)
        if self.active_layer_uid == layer.uid:
            self.active_layer_uid = self.layers[min(index, len(self.layers) - 1)].uid if self.layers else 0
        self.layers_changed.emit("structure", layer)
        return index

    def move_layer(self, layer: Layer, new_index: int) -> None:
        old = self.layers.index(layer)
        new_index = max(0, min(len(self.layers) - 1, new_index))
        if old == new_index:
            return
        self.layers.insert(new_index, self.layers.pop(old))
        self.layers_changed.emit("structure", layer)

    def set_active(self, layer: Layer | None) -> None:
        uid = layer.uid if layer is not None else 0
        if uid != self.active_layer_uid:
            self.active_layer_uid = uid
            self.layers_changed.emit("active", layer)

    def unique_name(self, base: str) -> str:
        names = {layer.name for layer in self.layers}
        if base not in names:
            return base
        for i in itertools.count(2):
            candidate = "%s %d" % (base, i)
            if candidate not in names:
                return candidate
        return base

    def on_changed(self, name: str) -> None:
        signal = self.__dict__.get("layers_changed")
        if signal is not None:
            signal.emit("set", None)

    # ---- 材质节点 ----
    def set_material(self, graph) -> None:
        """换材质节点图（None 去掉材质节点）。"""
        old = self.material
        if old is graph:
            return
        if old is not None:
            try:
                old.changed.disconnect(self._on_material_changed)
            except Exception:  # noqa: BLE001
                pass
        self.material = graph
        if graph is not None:
            graph.changed.connect(self._on_material_changed)
        self.layers_changed.emit("material", None)

    def ensure_material(self):
        """没有材质节点时建一张默认的（图层堆栈直接接到材质输出）。"""
        if self.material is None:
            from ..shading.graph import ShaderGraph

            self.set_material(ShaderGraph.default())
        return self.material

    def _on_material_changed(self, kind: str, node) -> None:
        if kind == "structure":
            self.layers_changed.emit("material", None)
        elif kind == "values":
            self.layers_changed.emit("material_values", None)

    # ---- 存取 ----
    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update(uid=self.uid, active=self.active_layer_uid, layers=[layer.to_dict() for layer in self.layers])
        if self.material is not None:
            data["material"] = self.material.to_dict()
        return data

    @classmethod
    def from_saved(cls, data: dict) -> "TextureSet":
        ts = cls(uid=int(data.get("uid", 0)) or None)
        ts.from_dict(data)
        reserve_uid(ts.uid)
        for item in data.get("layers", []):
            layer = Layer.from_saved(item)
            ts.layers.append(layer)
            ts._watch(layer)
        ts.active_layer_uid = int(data.get("active", 0))
        if ts.layer(ts.active_layer_uid) is None and ts.layers:
            ts.active_layer_uid = ts.layers[-1].uid
        ts.repair_structure()
        if data.get("material"):
            from ..shading.graph import ShaderGraph

            try:
                ts.set_material(ShaderGraph.from_dict(data["material"]))
            except Exception:  # noqa: BLE001  坏的材质节点不该让整个工程打不开
                pass
        return ts


class Brush(PropertyGroup):
    """笔刷设置。"""

    size = FloatProperty("直径", default=60.0, min=1.0, max=4000.0, soft_max=600.0, unit="px", precision=0,
                         description="笔刷直径")
    size_unit = EnumProperty("直径单位", items=[
        ("VIEW", "视口像素", "屏幕上看到多大就画多大"),
        ("SCENE", "场景单位", "在模型表面上保持同样大小")], default="VIEW")
    scene_size = FloatProperty("场景直径", default=0.1, min=0.0001, max=100.0, precision=4,
                               description="以场景单位计的笔刷直径")
    flow = FloatProperty("流量", default=1.0, min=0.01, max=1.0, subtype="FACTOR", precision=2,
                         description="每个笔触叠加多少")
    opacity = FloatProperty("不透明度", default=1.0, min=0.01, max=1.0, subtype="FACTOR", precision=2,
                            description="一笔之内能达到的最大浓度")
    hardness = FloatProperty("硬度", default=0.8, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                             description="边缘的软硬")
    spacing = FloatProperty("间距", default=0.1, min=0.01, max=2.0, subtype="FACTOR", precision=2,
                            description="相邻笔触之间的距离，相对直径")
    smooth = FloatProperty("稳定器", default=0.0, min=0.0, max=0.98, subtype="FACTOR", precision=2,
                           description="平滑手抖，数值越大线条越稳也越滞后")

    color = ColorProperty("颜色", default=(0.80, 0.30, 0.20))
    secondary_color = ColorProperty("备用颜色", default=(1.0, 1.0, 1.0))
    metallic = FloatProperty("金属度", default=0.0, min=0.0, max=1.0, subtype="FACTOR")
    roughness = FloatProperty("粗糙度", default=0.5, min=0.0, max=1.0, subtype="FACTOR")
    height = FloatProperty("高度", default=0.05, min=-1.0, max=1.0, precision=3,
                           description="正值凸起，负值凹陷")
    mask_value = FloatProperty("蒙版值", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
                               description="画蒙版时写入的值，1 显示，0 隐藏")

    use_basecolor = BoolProperty("基础色", default=True, description="笔刷是否写入基础色")
    use_metallic = BoolProperty("金属度", default=True)
    use_roughness = BoolProperty("粗糙度", default=True)
    use_height = BoolProperty("高度", default=False)

    pressure_size = BoolProperty("压感控制直径", default=False)
    pressure_flow = BoolProperty("压感控制流量", default=True)
    backface_cull = BoolProperty("不画背面", default=True, description="朝向相反的表面不受影响")
    normal_falloff = FloatProperty("法线衰减角", default=80.0, min=10.0, max=180.0, unit="°", precision=0,
                                   description="表面朝向与落笔处相差超过这个角度就不受影响")

    def channel_enabled(self, channel: str) -> bool:
        return getattr(self, "use_" + channel)


class Symmetry(PropertyGroup):
    x = BoolProperty("X", default=False, description="沿 X 轴镜像")
    y = BoolProperty("Y", default=False, description="沿 Y 轴镜像")
    z = BoolProperty("Z", default=False, description="沿 Z 轴镜像")


class SculptBrush(PropertyGroup):
    """雕刻笔刷的设置（各种雕刻笔刷共用）。"""

    size_unit = "VIEW"            # 光标按屏幕像素换算（和绘制笔刷的视口像素一致）
    scene_size = 0.0
    size = FloatProperty("直径", default=90.0, min=1.0, max=4000.0, unit="px", precision=0,
                         description="笔刷在屏幕上的直径")
    strength = FloatProperty("力度", default=0.5, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                             description="每一下改变多少")
    smooth_strength = FloatProperty("平滑力度", default=0.6, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                                    description="平滑笔刷（也包括按住 Shift 临时平滑）的力度")
    hardness = FloatProperty("硬度", default=0.0, min=0.0, max=0.95, subtype="FACTOR", precision=2,
                             description="越硬，笔刷边缘越陡")
    spacing = FloatProperty("间距", default=0.12, min=0.02, max=1.0, subtype="FACTOR", precision=2,
                            description="相邻两下之间隔多远，按直径算")
    pressure_size = BoolProperty("压感控制大小", default=False)
    pressure_strength = BoolProperty("压感控制力度", default=True)
    invert = BoolProperty("反向", default=False, description="堆变挖、加遮罩变去遮罩。按住 Ctrl 临时反向")


class RemeshSettings(PropertyGroup):
    """体素重构的设置。"""

    resolution = IntProperty("分辨率", default=256, min=16, max=2048,
                             description="模型最长的一边分成多少格。越大细节越多，面数大约按平方增长")
    relax = IntProperty("放松", default=2, min=0, max=20, description="让新网格的三角形更均匀的次数")
    close_holes = BoolProperty("封住开口", default=True, description="先把模型上的开口补上，生成封闭的网格")
    keep_mask = BoolProperty("保留遮罩", default=True, description="把遮罩搬到新网格上")
    min_island = IntProperty("去掉碎块", default=64, min=0, max=1_000_000,
                             description="少于这么多三角形的小碎块会被去掉，0 表示都保留")


class UnwrapSettings(PropertyGroup):
    """自动展开 UV 的设置。"""

    angle_limit = FloatProperty("角度限制", default=45.0, min=5.0, max=85.0, unit="°", precision=0,
                                description="同一块里各处朝向最多相差多少。越小拉伸越少，块和接缝越多")
    margin = IntProperty("块间距", default=16, min=2, max=512, unit="px",
                         description="块与块之间留多少贴图像素（按模型用的贴图大小算），给接缝处扩边")
    min_chart = IntProperty("最小块", default=32, min=0, max=1_000_000,
                            description="少于这么多三角形的小块并进旁边的块，减少零碎的接缝")
    rotate = BoolProperty("允许旋转", default=True, description="排列时可以把块转 90 度，贴图用得更满")
    transfer = BoolProperty("搬运已画内容", default=True,
                            description="换了 UV 之后，把原来画在旧 UV 上的内容按模型表面搬到新 UV 上")


#: 渐变的形状（照 Photoshop 的五种渐变）
GRADIENT_SHAPES = [("LINEAR", "线性", "沿拖动的方向从一头过渡到另一头"),
                   ("RADIAL", "径向", "从起点往外一圈一圈地过渡"),
                   ("ANGLE", "角度", "绕起点转一圈过渡"),
                   ("REFLECTED", "对称", "从起点往两边对称地过渡"),
                   ("DIAMOND", "菱形", "从起点往外按菱形过渡")]
GRADIENT_SHAPE_CODE = {item[0]: index for index, item in enumerate(GRADIENT_SHAPES)}


class GradientSettings(PropertyGroup):
    """渐变工具（照 Photoshop）：拖一条线，整层按这条线铺上渐变。"""

    shape = EnumProperty("形状", items=GRADIENT_SHAPES, default="LINEAR")
    source = EnumProperty("颜色", items=[("BRUSH", "笔刷颜色到备用颜色", "从笔刷颜色过渡到备用颜色"),
                                        ("RAMP", "色带", "按下面的色带过渡，色标可以带透明")], default="BRUSH")
    ramp = RampProperty("色带", description="左端是起点的颜色，右端是终点的颜色；色标越透明，那一段盖得越少")
    reverse = BoolProperty("反向", default=False, description="两头的颜色对调")
    opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                            description="铺上去有多实")


FILL_MODES = [("SIMILAR", "相近颜色", "点到的那片颜色差不多的地方"),
              ("ISLAND", "UV 岛", "点到的那块 UV 岛整块"),
              ("PART", "部件", "点到的那个网格部件整块"),
              ("LAYER", "整层", "整层都填")]


class FillSettings(PropertyGroup):
    """油漆桶（照 Photoshop）：点一下，把一片地方填上笔刷的颜色和材质。"""

    mode = EnumProperty("范围", items=FILL_MODES, default="SIMILAR")
    tolerance = FloatProperty("容差", default=32.0 / 255.0, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                              description="颜色差多少以内算一片")
    contiguous = BoolProperty("连续的", default=True, description="只填和点到的地方连成一片的；关掉时整张图里颜色相近的都填")
    sample_all = BoolProperty("对所有图层取样", default=True,
                              description="按看到的合成结果找相近颜色；关掉时只看当前图层自己画的")
    opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                            description="填上去有多实")


TONE_RANGE_ITEMS = [("SHADOWS", "阴影", "主要改暗的地方"), ("MIDTONES", "中间调", "主要改不暗不亮的地方"),
                    ("HIGHLIGHTS", "高光", "主要改亮的地方")]
TONE_RANGE_CODE = {item[0]: index for index, item in enumerate(TONE_RANGE_ITEMS)}


class EffectSettings(PropertyGroup):
    """减淡、加深、海绵、模糊、锐化、涂抹、仿制图章、修复画笔这几支笔刷（照 Photoshop）。
    笔刷的大小、硬度、流量、间距用画笔的设置。"""

    strength = FloatProperty("强度", default=0.5, min=0.01, max=1.0, subtype="FACTOR", precision=2,
                             description="每划一下改多少，来回划会越改越多")
    tone_range = EnumProperty("范围", items=TONE_RANGE_ITEMS, default="MIDTONES", description="减淡、加深主要改哪一段明暗")
    protect_tones = BoolProperty("保护色调", default=True, description="减淡、加深时只改明暗，颜色不跑偏")
    sponge_mode = EnumProperty("模式", items=[("DESATURATE", "去色", "颜色变灰"), ("SATURATE", "加色", "颜色变艳")],
                               default="DESATURATE")
    vibrance = BoolProperty("自然饱和度", default=True, description="海绵先动不太艳的颜色，已经很艳的少动")
    blur_radius = FloatProperty("模糊半径", default=3.0, min=0.5, max=100.0, precision=1, unit="px",
                                description="模糊笔每划一下糊开多宽（贴图纹素）")
    sharpen_amount = FloatProperty("锐化数量", default=1.0, min=0.1, max=5.0, precision=2, description="锐化笔加强边缘的力度")
    clone_aligned = BoolProperty("对齐", default=True,
                                 description="每一笔都接着上一笔的位置取样；关掉时每一笔都从来源点重新开始")
    has_source = BoolProperty("有来源", default=False, hidden=True, save=False)
    source_u = FloatProperty("来源 U", default=0.5, hidden=True, save=False)
    source_v = FloatProperty("来源 V", default=0.5, hidden=True, save=False)
    offset_set = BoolProperty("偏移已定", default=False, hidden=True, save=False)
    offset_u = FloatProperty("偏移 U", default=0.0, hidden=True, save=False)
    offset_v = FloatProperty("偏移 V", default=0.0, hidden=True, save=False)


def _scene_mesh_items(_owner=None):
    """可以当高模的：当前工程里的模型（标识是模型的编号）。"""
    items = [("0", "不用", "不用场景里的模型")]
    try:
        from ..app import instance

        app = instance()
        project = getattr(app, "project", None)
    except Exception:  # noqa: BLE001
        project = None
    for obj in getattr(project, "objects", []) or []:
        items.append((str(obj.uid), obj.name, "用场景里的「%s」当高模" % obj.name))
    return items


def _font_items(_owner=None):
    """系统里装的字体（第一项是界面默认字体）。"""
    cached = getattr(_font_items, "cache", None)
    if cached is None:
        try:
            from PySide6.QtGui import QFontDatabase

            families = sorted(set(QFontDatabase.families()))
        except Exception:  # noqa: BLE001
            families = []
        cached = [("", "默认字体", "界面用的字体")] + [(name, name, "") for name in families]
        if len(cached) > 1:
            _font_items.cache = cached
    return cached


ALIGN_ITEMS = [("LEFT", "左对齐", ""), ("CENTER", "居中", ""), ("RIGHT", "右对齐", "")]
SHAPE_ITEMS = [("RECT", "矩形", ""), ("ROUNDED", "圆角矩形", ""), ("ELLIPSE", "椭圆", ""), ("POLYGON", "多边形", ""),
               ("LINE", "直线", "")]


class TextSettings(PropertyGroup):
    """文字工具（照 Photoshop）：点一下，在那里写字；做完在「调整上一步」里还能改字、字号、旋转。"""

    font = EnumProperty("字体", items=_font_items, default="")
    size = FloatProperty("字号", default=48.0, min=2.0, max=2048.0, precision=0, unit="px",
                         description="字有多高（屏幕像素，按写字时的缩放）")
    bold = BoolProperty("粗体", default=False)
    italic = BoolProperty("斜体", default=False)
    rotation = FloatProperty("旋转", default=0.0, min=-180.0, max=180.0, unit="°", precision=0)
    align = EnumProperty("对齐", items=ALIGN_ITEMS, default="LEFT")
    line_spacing = FloatProperty("行距", default=1.2, min=0.5, max=4.0, precision=2, description="行与行隔多远（按字高的倍数）")
    opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2)
    occlude = BoolProperty("只写在看得见的地方", default=True, description="在模型上写字时，被挡住的地方不写")


class ShapeSettings(PropertyGroup):
    """形状工具（照 Photoshop）：拖出矩形、椭圆、多边形或直线。按住 Shift 等比，按住 Alt 从中心往外拉。"""

    kind = EnumProperty("形状", items=SHAPE_ITEMS, default="RECT")
    fill = BoolProperty("填充", default=True, description="里面填上颜色")
    stroke = FloatProperty("描边", default=0.0, min=0.0, max=1024.0, precision=0, unit="px",
                           description="边线有多粗（屏幕像素），0 是不描边")
    radius = FloatProperty("圆角", default=32.0, min=0.0, max=4096.0, precision=0, unit="px",
                           description="圆角矩形的圆角半径（屏幕像素）")
    sides = IntProperty("边数", default=6, min=3, max=64, description="多边形有几条边")
    opacity = FloatProperty("不透明度", default=1.0, min=0.0, max=1.0, subtype="FACTOR", precision=2)
    occlude = BoolProperty("只画在看得见的地方", default=True, description="在模型上画形状时，被挡住的地方不画")


SELECT_MODE_ITEMS = [("SET", "新建", "换成新画的选区", "sel.set"),
                     ("ADD", "添加", "加到原来的选区里（拖之前按住 Shift 也是）", "sel.add"),
                     ("SUBTRACT", "减去", "从原来的选区里去掉（拖之前按住 Alt 也是）", "sel.subtract"),
                     ("INTERSECT", "交叉", "只留和原来的选区重叠的部分（拖之前按住 Shift+Alt 也是）", "sel.intersect")]


class SelectionSettings(PropertyGroup):
    """选区工具（照 Photoshop）：矩形、椭圆选框，套索、多边形套索，魔棒，快速选择。
    有选区时，画笔、滤镜、填充只改选区里面。"""

    mode = EnumProperty("方式", items=SELECT_MODE_ITEMS, default="SET", description="新画的和原来的选区怎么合")
    feather = FloatProperty("羽化", default=0.0, min=0.0, max=500.0, precision=1, unit="px",
                            description="选区边缘柔和过渡多宽（屏幕像素），0 是硬边")
    antialias = BoolProperty("消除锯齿", default=True, description="选区边缘平滑，不出锯齿")
    occlude = BoolProperty("只选看得见的地方", default=True, description="在模型上框选、圈选时，被挡住的地方不选")
    tolerance = FloatProperty("容差", default=32.0 / 255.0, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                              description="魔棒、快速选择：颜色差多少以内算一片")
    contiguous = BoolProperty("连续的", default=True,
                              description="魔棒只选和点到的地方连成一片的；关掉时整张图里颜色相近的都选上")
    sample_all = BoolProperty("对所有图层取样", default=True,
                              description="按看到的合成结果找相近颜色；关掉时只看当前图层自己画的")
    quick_size = FloatProperty("大小", default=40.0, min=2.0, max=1000.0, precision=0, unit="px",
                               description="快速选择的笔刷有多大（屏幕像素）")


#: 3D 视口的模式（和 Blender 一样，在任何工作区都能切换）
MODE_ITEMS = [("OBJECT", "物体模式", "选择、移动、添加和删除物体", "mode.object"),
              ("EDIT", "编辑模式", "编辑模型的点、边、面", "mode.edit"),
              ("SCULPT", "雕刻模式", "改变模型的形状", "mode.sculpt"),
              ("PAINT", "纹理绘制", "在模型上画贴图", "mode.paint")]


class ToolSettings(PropertyGroup):
    mode = EnumProperty("模式", items=MODE_ITEMS, default="PAINT")
    # 变换（G、R、S）
    use_snap = BoolProperty("吸附", default=False, description="移动、旋转、缩放时按步长吸附。拖动时按住 Ctrl 临时反过来")
    snap_elements = EnumProperty("吸附到", items=[
        ("INCREMENT", "增量", "移动的距离按步长取整"),
        ("GRID", "网格", "位置对齐到地面网格的格点"),
        ("SURFACE", "表面", "移动时贴到鼠标下的模型表面上")], default="INCREMENT")
    snap_translate = FloatProperty("移动步长", default=1.0, min=0.0001, max=10000.0, precision=3, unit="m")
    snap_rotate = FloatProperty("旋转步长", default=5.0, min=0.01, max=180.0, precision=2, unit="°")
    snap_scale = FloatProperty("缩放步长", default=0.1, min=0.0001, max=100.0, precision=3)
    precision_factor = FloatProperty("精细倍率", default=0.1, min=0.001, max=1.0, precision=3,
                                     description="拖动时按住 Shift，移动量乘以这个数")
    transform_pivot_point = EnumProperty("轴心点", items=[
        ("BOUNDING_BOX_CENTER", "边界框中心", "选中物体合起来的边界框中心", "pivot.bbox"),
        ("CURSOR", "3D 游标", "绕 3D 游标旋转、缩放", "pivot.cursor"),
        ("INDIVIDUAL_ORIGINS", "各自的原点", "每个物体绕自己的原点旋转、缩放", "pivot.individual"),
        ("MEDIAN_POINT", "质心点", "选中物体原点的平均位置", "pivot.median"),
        ("ACTIVE_ELEMENT", "活动元素", "绕当前物体的原点", "pivot.active")], default="MEDIAN_POINT")
    use_proportional_edit = BoolProperty("衰减编辑", default=False,
                                         description="编辑模式里移动选中的点时，附近没选中的点按距离跟着动")
    proportional_edit_falloff = EnumProperty("衰减方式", items=[
        ("SMOOTH", "平滑", ""), ("SPHERE", "球状", ""), ("ROOT", "根凸", ""), ("INVERSE_SQUARE", "平方倒数", ""),
        ("SHARP", "锐利", ""), ("LINEAR", "线性", ""), ("CONSTANT", "常量", ""), ("RANDOM", "随机", "")],
        default="SMOOTH")
    proportional_size = FloatProperty("衰减半径", default=1.0, min=0.0001, max=100000.0, precision=3, unit="m",
                                      description="变换时滚轮改大小")
    use_proportional_connected = BoolProperty("只影响相连的", default=False)
    transform_orientation = EnumProperty("坐标系", items=[
        ("GLOBAL", "全局", "按世界的 X、Y、Z 轴", "orientation.global"),
        ("LOCAL", "局部", "按当前物体自己的轴", "orientation.local"),
        ("NORMAL", "法向", "编辑模式里按选中部分的法线", "orientation.normal"),
        ("VIEW", "视图", "按屏幕的横、竖和视线方向", "orientation.view"),
        ("CURSOR", "游标", "按 3D 游标的朝向", "orientation.cursor")], default="GLOBAL")
    tool = StringProperty("当前工具", default="paint.brush")
    brush = PointerProperty("笔刷", type=Brush)
    eraser = PointerProperty("橡皮", type=Brush)
    sculpt = PointerProperty("雕刻笔刷", type=SculptBrush)
    symmetry = PointerProperty("对称", type=Symmetry)
    remesh = PointerProperty("体素重构", type=RemeshSettings)
    unwrap = PointerProperty("展开 UV", type=UnwrapSettings)
    gradient = PointerProperty("渐变", type=GradientSettings)
    fill = PointerProperty("油漆桶", type=FillSettings)
    effect = PointerProperty("效果笔刷", type=EffectSettings)
    text = PointerProperty("文字", type=TextSettings)
    shape = PointerProperty("形状", type=ShapeSettings)
    selection = PointerProperty("选区", type=SelectionSettings)

    def active_brush(self):
        if self.tool.startswith("sculpt."):
            return self.sculpt
        return self.eraser if self.tool == "paint.eraser" else self.brush


def _resource_items(kind: str):
    """环境、Matcap、工作室光的下拉选项：随资源目录里有什么而变。"""
    def items(_owner=None):
        try:
            from ..engine import ibl
            names = {"hdri": ibl.list_hdris, "matcap": ibl.list_matcaps, "studio": ibl.list_studio_lights}[kind]()
            return [(name, ibl.label(name, kind), "") for name in names]
        except Exception:  # noqa: BLE001
            return []
    return items


class ViewShading(PropertyGroup):
    """一个视口的显示设置。"""

    mode = EnumProperty("着色模式", items=[
        ("WIREFRAME", "线框", "只画模型的边", "wireframe"),
        ("SOLID", "实体", "工作室光，适合看形体", "shade_solid"),
        ("MATERIAL", "材质预览", "PBR 材质加环境光", "shade_material"),
        ("CHANNEL", "通道", "只看一个通道", "shade_channel"),
        ("RENDERED", "渲染", "用渲染引擎实时出图（默认路径追踪）", "shade_rendered")], default="MATERIAL")
    show_xray = BoolProperty("X 光", default=False, description="模型半透明，能看到、选到后面的东西")
    xray_alpha = FloatProperty("X 光透明度", default=0.5, min=0.0, max=1.0, subtype="FACTOR", precision=2)
    channel = EnumProperty("查看通道", items=[(c[0], c[1], "") for c in CHANNELS], default="basecolor")
    hdri = EnumProperty("环境", items=_resource_items("hdri"), default="forest",
                        description="照亮模型的环境图")
    hdri_rotation = FloatProperty("环境旋转", default=0.0, min=-180.0, max=180.0, unit="°", precision=0)
    hdri_strength = FloatProperty("环境强度", default=1.0, min=0.0, max=10.0, soft_max=3.0, precision=2)
    background_opacity = FloatProperty("背景可见度", default=0.0, min=0.0, max=1.0, subtype="FACTOR")
    background_blur = FloatProperty("背景模糊", default=0.5, min=0.0, max=1.0, subtype="FACTOR")
    exposure = FloatProperty("曝光", default=0.0, min=-10.0, max=10.0, soft_min=-4.0, soft_max=4.0, unit="EV",
                             precision=2)
    view_transform = EnumProperty("显示变换", items=[("STANDARD", "标准", ""), ("FILMIC", "Filmic", ""),
                                                 ("AGX", "AgX", "")], default="AGX")
    solid_light = EnumProperty("实体光照", items=[("STUDIO", "工作室光", ""), ("MATCAP", "Matcap", ""),
                                              ("FLAT", "无光照", "")], default="STUDIO")
    studio_light = EnumProperty("工作室光", items=_resource_items("studio"), default="studio")
    matcap = EnumProperty("Matcap", items=_resource_items("matcap"), default="basic_1")
    solid_color = EnumProperty("实体颜色", items=[("TEXTURE", "贴图", "显示基础色"), ("SINGLE", "单色", "统一灰色")],
                               default="TEXTURE")
    show_wireframe = BoolProperty("线框", default=False)
    show_grid = BoolProperty("地面网格", default=False)
    bump = BoolProperty("显示高度凹凸", default=True)


class ViewOverlay(PropertyGroup):
    """视口叠加层（和 Blender 的「叠加层」一样）。"""

    show_overlays = BoolProperty("叠加层", default=True, description="网格、坐标轴、游标、摄像机和灯光的线框等全部开关")
    show_floor = BoolProperty("地面网格", default=True)
    show_axis_x = BoolProperty("X 轴", default=True)
    show_axis_y = BoolProperty("Y 轴", default=True)
    grid_scale = FloatProperty("网格大小", default=1.0, min=0.001, max=1000.0, precision=3,
                               description="地面网格一格多大")
    show_cursor = BoolProperty("3D 游标", default=True)
    show_extras = BoolProperty("摄像机、灯光", default=True, description="摄像机、灯光、空物体的线框")
    show_outline = BoolProperty("选中描边", default=True)
    show_object_origins = BoolProperty("原点", default=True, description="选中物体的原点画成小点")
    show_object_origins_all = BoolProperty("全部原点", default=False, description="没选中的物体也显示原点")
    show_stats = BoolProperty("统计信息", default=False, description="在视口左上角显示物体数、面数")
    show_gizmo = BoolProperty("导航小部件", default=True, description="右上角的坐标轴和平移、缩放按钮")
    show_wireframes = BoolProperty("线框", default=False, description="在模型表面上叠一层线框")
    wireframe_opacity = FloatProperty("线框不透明度", default=0.6, min=0.0, max=1.0, subtype="FACTOR", precision=2)


#: 模型形状的版本号（全局递增，不同模型、删了又建的模型也不会撞号）：自动保存据此判断形状变没变
_geometry_serial = itertools.count(1)


class MeshObject(PropertyGroup):
    """一个模型。data 是 meshio.MeshData；material_sets[i] 是第 i 个材质对应的纹理集 uid。

    geometry_dirty：形状相对工程文件改过、保存时要重写模型数据。每次标成 True 都换一个新的 geometry_version。
    """

    kind = "MESH"
    undoable = True
    name = StringProperty("名称", default="模型")
    visible = BoolProperty("在视口中显示", default=True)
    select = BoolProperty("选中", default=False, hidden=True)

    def __init__(self, name: str, data: Any, material_sets: list[int]) -> None:
        from .objects import ObjectTransform

        super().__init__()
        self.uid = new_uid()
        self.name = name
        self.data = data
        self.material_sets = material_sets
        self.transform = ObjectTransform()      # 当前变换（顶点已经是变换后的世界坐标）
        self._geometry_dirty = False
        self.geometry_version = next(_geometry_serial)

    @property
    def geometry_dirty(self) -> bool:
        return self._geometry_dirty

    @geometry_dirty.setter
    def geometry_dirty(self, value: bool) -> None:
        self._geometry_dirty = bool(value)
        if value:
            self.geometry_version = next(_geometry_serial)


class Project:
    """一个工程。"""

    def __init__(self, name: str = "未命名") -> None:
        self.name = name
        self.path = ""
        self.dirty = False
        self.objects: list[MeshObject] = []
        self.texture_sets: list[TextureSet] = []
        self.active_set_uid = 0
        from ..render.props import RenderProps

        self.render = RenderProps()          # 出图设置（渲染引擎、分辨率、各引擎自己的设置）
        self.scene_objects: list = []         # 摄像机、灯光、空物体（doc.objects.SceneObject）
        self.active_object_uid = 0            # 当前物体（模型、摄像机、灯光都算）
        self.active_camera_uid = 0            # 出图和「从摄像机看」用的摄像机
        self.cursor_location = (0.0, 0.0, 0.0)   # 3D 游标：新加的物体放在这里
        self.graphs: list = []                # 节点图（nodes.graph.NodeGraph）
        self.active_graph_uid = 0             # 节点编辑器正在编辑的图
        #: (图, 变化种类)：节点图的内容变了（种类见 NodeGraph.changed）；图本身增删时图为 None、种类 "list"
        self.graphs_changed = Signal()
        #: kind: "loaded" 整体更换；"objects"；"sets"；"active_set"；"dirty"；"path"
        self.changed = Signal()

    @property
    def active_texture_set(self) -> TextureSet | None:
        for ts in self.texture_sets:
            if ts.uid == self.active_set_uid:
                return ts
        return self.texture_sets[0] if self.texture_sets else None

    def texture_set(self, uid: int) -> TextureSet | None:
        for ts in self.texture_sets:
            if ts.uid == uid:
                return ts
        return None

    def add_texture_set(self, ts: TextureSet, index: int | None = None) -> TextureSet:
        if index is None:
            self.texture_sets.append(ts)
        else:
            self.texture_sets.insert(max(0, min(len(self.texture_sets), index)), ts)
        ts.layers_changed.connect(self._on_layers_changed)
        if not self.active_set_uid:
            self.active_set_uid = ts.uid
        self.changed.emit("sets")
        return ts

    def remove_texture_set(self, ts: TextureSet) -> int:
        """拿掉一套纹理集（它的图层像素由引擎留着，撤销时放回），返回原来的位置。"""
        if ts not in self.texture_sets:
            return -1
        index = self.texture_sets.index(ts)
        self.texture_sets.remove(ts)
        try:
            ts.layers_changed.disconnect(self._on_layers_changed)
        except Exception:  # noqa: BLE001
            pass
        if self.active_set_uid == ts.uid:
            self.active_set_uid = self.texture_sets[0].uid if self.texture_sets else 0
        self.mark_dirty()
        self.changed.emit("sets")
        return index

    def set_active_set(self, ts: TextureSet) -> None:
        if ts.uid != self.active_set_uid:
            self.active_set_uid = ts.uid
            self.changed.emit("active_set")

    # ---- 场景物体（模型 + 摄像机、灯光、空物体）----
    def all_objects(self) -> list:
        return list(self.objects) + list(self.scene_objects)

    def object_by_uid(self, uid: int):
        for obj in self.all_objects():
            if obj.uid == uid:
                return obj
        return None

    @property
    def active_object(self):
        return self.object_by_uid(self.active_object_uid)

    def selected_objects(self) -> list:
        return [o for o in self.all_objects() if getattr(o, "select", False) and o.visible]

    def set_active_object(self, obj) -> None:
        uid = obj.uid if obj is not None else 0
        if uid != self.active_object_uid:
            self.active_object_uid = uid
            self.changed.emit("selection")

    def selection_changed(self) -> None:
        self.changed.emit("selection")

    def add_scene_object(self, obj, index: int | None = None):
        if index is None:
            self.scene_objects.append(obj)
        else:
            self.scene_objects.insert(max(0, min(len(self.scene_objects), index)), obj)
        if obj.kind == "CAMERA" and not self.active_camera_uid:
            self.active_camera_uid = obj.uid
        self.mark_dirty()
        self.changed.emit("objects")
        return obj

    def remove_object(self, obj) -> int:
        """拿掉一个物体（模型或摄像机、灯光），返回它原来在列表里的位置。"""
        if obj in self.objects:
            index = self.objects.index(obj)
            self.objects.remove(obj)
        elif obj in self.scene_objects:
            index = self.scene_objects.index(obj)
            self.scene_objects.remove(obj)
        else:
            return -1
        if self.active_object_uid == obj.uid:
            self.active_object_uid = 0
        if self.active_camera_uid == obj.uid:
            self.active_camera_uid = next((o.uid for o in self.scene_objects if o.kind == "CAMERA"), 0)
        self.mark_dirty()
        self.changed.emit("objects")
        return index

    @property
    def active_camera(self):
        obj = self.object_by_uid(self.active_camera_uid)
        return obj if obj is not None and getattr(obj, "kind", "") == "CAMERA" else None

    # ---- 节点图 ----
    @property
    def active_graph(self):
        return self.graph(self.active_graph_uid) or (self.graphs[0] if self.graphs else None)

    def set_active_graph(self, graph) -> None:
        uid = graph.uid if graph is not None else 0
        if uid != self.active_graph_uid:
            self.active_graph_uid = uid
            self.graphs_changed.emit(None, "active")

    def graph(self, uid: int):
        for graph in self.graphs:
            if graph.uid == uid:
                return graph
        return None

    def add_graph(self, graph, index: int | None = None):
        if index is None:
            self.graphs.append(graph)
        else:
            self.graphs.insert(max(0, min(len(self.graphs), index)), graph)
        graph.changed.connect(lambda kind, _node, target=graph: self._on_graph_changed(target, kind))
        self.mark_dirty()
        self.graphs_changed.emit(None, "list")
        return graph

    def remove_graph(self, uid: int):
        graph = self.graph(uid)
        if graph is not None:
            self.graphs.remove(graph)
            self.mark_dirty()
            self.graphs_changed.emit(None, "list")
        return graph

    def _on_graph_changed(self, graph, kind: str) -> None:
        if graph not in self.graphs:
            return
        if kind != "active":
            self.mark_dirty()
        self.graphs_changed.emit(graph, kind)

    def add_object(self, obj: MeshObject, index: int | None = None) -> MeshObject:
        if index is None:
            self.objects.append(obj)
        else:
            self.objects.insert(max(0, min(len(self.objects), index)), obj)
        self.mark_dirty()
        self.changed.emit("objects")
        return obj

    def mark_dirty(self, value: bool = True) -> None:
        if self.dirty != value:
            self.dirty = value
            self.changed.emit("dirty")

    def _on_layers_changed(self, kind: str, layer: Any) -> None:
        if kind != "active":
            self.mark_dirty()

    @property
    def title(self) -> str:
        return self.name + (" *" if self.dirty else "")

    # ---- 存取（不含像素与模型数据）----
    def to_dict(self) -> dict:
        return {"name": self.name, "active_set": self.active_set_uid, "render": self.render.to_saved(),
                "graphs": [graph.to_dict() for graph in self.graphs], "active_graph": self.active_graph_uid,
                "texture_sets": [ts.to_dict() for ts in self.texture_sets],
                "objects": [{"uid": o.uid, "name": o.name, "material_sets": list(o.material_sets),
                             "visible": o.visible, "select": bool(getattr(o, "select", False)),
                             "transform": o.transform.to_dict()} for o in self.objects],
                "scene_objects": [o.to_dict() for o in self.scene_objects],
                "active_object": self.active_object_uid, "active_camera": self.active_camera_uid,
                "cursor": [float(v) for v in self.cursor_location]}
