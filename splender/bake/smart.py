"""智能材质：几层带生成器蒙版的填充层，一键加到当前纹理集，按模型的形状自动落在该落的地方。

每个材质：名字、一句说明、缩略图用的几种颜色、自下而上的若干层（图层属性的值）。
颜色按界面里的取色写（sRGB）。生成器的含义见 doc.project.GENERATOR_ITEMS。
"""
from __future__ import annotations


def _fill(name: str, color, metallic: float = 0.0, roughness: float = 0.5, **extra) -> dict:
    values = {"name": name, "kind": "FILL", "fill_color": tuple(color), "fill_metallic": metallic,
              "fill_roughness": roughness, "use_basecolor": True, "use_metallic": True, "use_roughness": True}
    values.update(extra)
    if "fill_height" in values:
        values["use_height"] = True
    return values


SMART_MATERIALS = [
    {
        "id": "bronze", "name": "青铜古器", "description": "出土青铜：凹处生铜绿，棱角被摸得发亮，底部带土沁",
        "swatch": [(0.55, 0.38, 0.20), (0.30, 0.55, 0.45), (0.85, 0.62, 0.35)],
        "layers": [
            _fill("青铜", (0.55, 0.38, 0.20), 1.0, 0.35),
            _fill("铜绿", (0.30, 0.55, 0.45), 0.0, 0.85, mask_generator="CAVITY", gen_range=0.55, gen_softness=0.55,
                  gen_noise=0.75, gen_noise_size=0.03, fill_height=0.01),
            _fill("包浆", (0.85, 0.62, 0.35), 1.0, 0.18, mask_generator="EDGES", gen_range=0.35, gen_softness=0.3,
                  gen_noise=0.4, gen_noise_size=0.03),
            _fill("土沁", (0.42, 0.33, 0.22), 0.0, 0.9, mask_generator="GRADIENT", gen_direction="DOWN",
                  gen_range=0.3, gen_softness=0.3, gen_noise=0.7, gen_noise_size=0.06),
        ],
    },
    {
        "id": "rust_iron", "name": "锈铁", "description": "风吹雨打的铁件：缝里锈透，表面浮锈成片，棱角磨出铁色",
        "swatch": [(0.42, 0.42, 0.43), (0.45, 0.20, 0.08), (0.65, 0.65, 0.66)],
        "layers": [
            _fill("铁", (0.42, 0.42, 0.43), 1.0, 0.45),
            _fill("浮锈", (0.55, 0.28, 0.12), 0.0, 0.85, mask_generator="NOISE", gen_range=0.3, gen_softness=0.4,
                  gen_noise=1.0, gen_noise_size=0.035),
            _fill("锈蚀", (0.45, 0.20, 0.08), 0.0, 0.92, mask_generator="OCCLUSION", gen_range=0.6, gen_softness=0.3,
                  gen_noise=0.7, gen_noise_size=0.03, fill_height=0.015),
            _fill("磨亮", (0.65, 0.65, 0.66), 1.0, 0.3, mask_generator="EDGES", gen_range=0.3, gen_softness=0.2,
                  gen_noise=0.5, gen_noise_size=0.02),
        ],
    },
    {
        "id": "worn_paint", "name": "掉漆机甲", "description": "红漆机甲：棱角掉漆露出金属，缝里积着油污",
        "swatch": [(0.75, 0.18, 0.12), (0.55, 0.55, 0.57), (0.18, 0.15, 0.12)],
        "layers": [
            _fill("金属底", (0.55, 0.55, 0.57), 1.0, 0.35),
            _fill("红漆", (0.75, 0.18, 0.12), 0.0, 0.45, mask_generator="EDGES", gen_invert=True, gen_range=0.25,
                  gen_softness=0.05, gen_noise=0.6, gen_noise_size=0.02),
            _fill("油污", (0.18, 0.15, 0.12), 0.0, 0.8, opacity=0.75, mask_generator="CAVITY", gen_range=0.5,
                  gen_softness=0.3, gen_noise=0.5, gen_noise_size=0.04),
        ],
    },
    {
        "id": "lacquer_gold", "name": "黑漆描金", "description": "漆器：乌黑的漆面，棱线描金，朝上的面落了一层薄灰",
        "swatch": [(0.03, 0.03, 0.035), (1.0, 0.77, 0.34), (0.35, 0.33, 0.30)],
        "layers": [
            _fill("黑漆", (0.03, 0.03, 0.035), 0.0, 0.12),
            _fill("描金", (1.0, 0.77, 0.34), 1.0, 0.25, mask_generator="EDGES", gen_range=0.3, gen_softness=0.1,
                  gen_noise=0.15, gen_noise_size=0.05),
            _fill("微尘", (0.35, 0.33, 0.30), 0.0, 0.7, opacity=0.35, mask_generator="FACING", gen_range=0.25,
                  gen_softness=0.2, gen_noise=0.8, gen_noise_size=0.05),
        ],
    },
    {
        "id": "celadon", "name": "青瓷", "description": "雨过天青：凹处积釉色深，棱线釉薄发白",
        "swatch": [(0.55, 0.72, 0.62), (0.35, 0.55, 0.47), (0.82, 0.86, 0.80)],
        "layers": [
            _fill("青釉", (0.55, 0.72, 0.62), 0.0, 0.08),
            _fill("积釉", (0.35, 0.55, 0.47), 0.0, 0.05, mask_generator="CAVITY", gen_range=0.5, gen_softness=0.5,
                  gen_noise=0.2, gen_noise_size=0.08),
            _fill("出筋", (0.82, 0.86, 0.80), 0.0, 0.1, mask_generator="EDGES", gen_range=0.3, gen_softness=0.4,
                  gen_noise=0.1, gen_noise_size=0.05),
        ],
    },
    {
        "id": "mossy_stone", "name": "苔石", "description": "山间旧石：石缝发暗，朝上的面长满青苔",
        "swatch": [(0.45, 0.44, 0.42), (0.28, 0.42, 0.15), (0.18, 0.17, 0.16)],
        "layers": [
            _fill("石", (0.45, 0.44, 0.42), 0.0, 0.85),
            _fill("石缝", (0.18, 0.17, 0.16), 0.0, 0.95, mask_generator="OCCLUSION", gen_range=0.5, gen_softness=0.4,
                  gen_noise=0.5, gen_noise_size=0.03),
            _fill("青苔", (0.28, 0.42, 0.15), 0.0, 0.95, mask_generator="FACING", gen_range=0.35, gen_softness=0.25,
                  gen_noise=0.8, gen_noise_size=0.04, fill_height=0.01),
        ],
    },
    {
        "id": "snow", "name": "初雪", "description": "深色木石上落了一场雪：朝上的面积雪，边缘参差",
        "swatch": [(0.25, 0.20, 0.17), (0.95, 0.97, 1.0)],
        "layers": [
            _fill("木石", (0.25, 0.20, 0.17), 0.0, 0.8),
            _fill("积雪", (0.95, 0.97, 1.0), 0.0, 0.6, mask_generator="FACING", gen_range=0.4, gen_softness=0.15,
                  gen_noise=0.6, gen_noise_size=0.05, fill_height=0.03),
        ],
    },
    {
        "id": "gilt", "name": "鎏金", "description": "鎏金铜器：棱角处金层剥落露出红铜胎，凹处氧化发乌",
        "swatch": [(1.0, 0.78, 0.35), (0.45, 0.20, 0.12), (0.30, 0.22, 0.10)],
        "layers": [
            _fill("金", (1.0, 0.78, 0.35), 1.0, 0.2),
            _fill("红铜胎", (0.62, 0.30, 0.18), 1.0, 0.45, mask_generator="EDGES", gen_range=0.15, gen_softness=0.1,
                  gen_noise=0.8, gen_noise_size=0.015),
            _fill("氧化", (0.30, 0.22, 0.10), 1.0, 0.6, mask_generator="CAVITY", gen_range=0.5, gen_softness=0.35,
                  gen_noise=0.4, gen_noise_size=0.04),
        ],
    },
    {
        "id": "leather", "name": "旧皮革", "description": "用了多年的皮具：边缘磨白，褶缝里发黑",
        "swatch": [(0.36, 0.20, 0.11), (0.62, 0.45, 0.32), (0.15, 0.08, 0.04)],
        "layers": [
            _fill("皮", (0.36, 0.20, 0.11), 0.0, 0.55),
            _fill("磨白", (0.62, 0.45, 0.32), 0.0, 0.75, mask_generator="EDGES", gen_range=0.35, gen_softness=0.3,
                  gen_noise=0.5, gen_noise_size=0.03),
            _fill("污渍", (0.15, 0.08, 0.04), 0.0, 0.6, mask_generator="CAVITY", gen_range=0.5, gen_softness=0.3,
                  gen_noise=0.5, gen_noise_size=0.04),
        ],
    },
    {
        "id": "charred", "name": "焦木", "description": "火里抢出来的木头：从下往上烧黑，裂缝里积着灰",
        "swatch": [(0.45, 0.30, 0.18), (0.03, 0.025, 0.02), (0.55, 0.52, 0.50)],
        "layers": [
            _fill("木", (0.45, 0.30, 0.18), 0.0, 0.7),
            _fill("炭化", (0.03, 0.025, 0.02), 0.0, 0.9, mask_generator="GRADIENT", gen_direction="DOWN",
                  gen_range=0.55, gen_softness=0.3, gen_noise=0.8, gen_noise_size=0.06),
            _fill("灰烬", (0.55, 0.52, 0.50), 0.0, 1.0, mask_generator="CAVITY", gen_range=0.4, gen_softness=0.3,
                  gen_noise=0.7, gen_noise_size=0.03),
        ],
    },
    {
        "id": "jade", "name": "古玉", "description": "碧玉：薄处透亮，表面带几块黄褐色的沁",
        "swatch": [(0.25, 0.55, 0.35), (0.65, 0.85, 0.60), (0.55, 0.42, 0.22)],
        "layers": [
            _fill("碧玉", (0.25, 0.55, 0.35), 0.0, 0.15),
            _fill("透光", (0.65, 0.85, 0.60), 0.0, 0.12, mask_generator="THIN", gen_range=0.5, gen_softness=0.5,
                  gen_noise=0.3, gen_noise_size=0.06),
            _fill("沁色", (0.55, 0.42, 0.22), 0.0, 0.3, mask_generator="NOISE", gen_range=0.2, gen_softness=0.25,
                  gen_noise=1.0, gen_noise_size=0.08),
        ],
    },
]

_BY_ID = {item["id"]: item for item in SMART_MATERIALS}


def get(material_id: str) -> dict | None:
    return _BY_ID.get(material_id)


# ====================================================================== 智能遮罩
# 只是一套生成器设置：套到当前图层上，蒙版就按模型的形状落在该落的地方。图层本身画什么、填什么不变。
def _mask(mask_id: str, name: str, description: str, generator: str, **values) -> dict:
    settings = {"mask_generator": generator, "gen_invert": False}
    settings.update(values)
    return {"id": mask_id, "name": name, "description": description, "values": settings}


SMART_MASKS = [
    _mask("edge_wear", "边缘磨损", "棱角上细碎的磨损", "EDGES",
          gen_range=0.3, gen_softness=0.15, gen_noise=0.6, gen_noise_size=0.02),
    _mask("edge_chips", "棱角崩口", "棱角成块掉漆，边界又硬又碎", "EDGES",
          gen_range=0.4, gen_softness=0.04, gen_noise=0.85, gen_noise_size=0.012),
    _mask("hand_polish", "手摸包浆", "常被手摸的棱线，柔和发亮", "EDGES",
          gen_range=0.22, gen_softness=0.45, gen_noise=0.15, gen_noise_size=0.08),
    _mask("crevice_dust", "缝里积灰", "凹缝里积着灰", "CAVITY",
          gen_range=0.55, gen_softness=0.5, gen_noise=0.4, gen_noise_size=0.05),
    _mask("deep_grime", "深处油泥", "光照不到的深处又黑又腻", "OCCLUSION",
          gen_range=0.5, gen_softness=0.35, gen_noise=0.5, gen_noise_size=0.03),
    _mask("top_dust", "顶面落尘", "朝上的面落了一层灰", "FACING", gen_direction="UP",
          gen_range=0.35, gen_softness=0.3, gen_noise=0.7, gen_noise_size=0.05),
    _mask("snow_cap", "一夜积雪", "朝上的面盖着雪，边界干净", "FACING", gen_direction="UP",
          gen_range=0.28, gen_softness=0.08, gen_noise=0.35, gen_noise_size=0.09),
    _mask("moss_shade", "背阴苔痕", "朝下、背光的面长出苔", "FACING", gen_direction="DOWN",
          gen_range=0.3, gen_softness=0.3, gen_noise=0.8, gen_noise_size=0.04),
    _mask("ground_mud", "溅泥", "从地面往上溅的泥点", "GRADIENT", gen_direction="DOWN",
          gen_range=0.3, gen_softness=0.35, gen_noise=0.75, gen_noise_size=0.06),
    _mask("sun_fade", "日晒褪色", "顶上晒褪了色，往下渐浅", "GRADIENT", gen_direction="UP",
          gen_range=0.45, gen_softness=0.6, gen_noise=0.3, gen_noise_size=0.12),
    _mask("thin_glow", "薄处透光", "薄的地方透出光", "THIN",
          gen_range=0.4, gen_softness=0.4, gen_noise=0.1, gen_noise_size=0.05),
    _mask("water_stains", "水渍", "大片不规则的水渍", "NOISE",
          gen_range=0.35, gen_softness=0.4, gen_noise=1.0, gen_noise_size=0.08),
    _mask("speckles", "细碎斑点", "满身细小的斑点", "NOISE",
          gen_range=0.15, gen_softness=0.1, gen_noise=1.0, gen_noise_size=0.01),
    _mask("lichen", "地衣斑块", "一块一块的地衣，边缘毛糙", "NOISE",
          gen_range=0.25, gen_softness=0.06, gen_noise=1.0, gen_noise_size=0.035),
]

_MASK_BY_ID = {item["id"]: item for item in SMART_MASKS}


def get_mask(mask_id: str) -> dict | None:
    return _MASK_BY_ID.get(mask_id)
