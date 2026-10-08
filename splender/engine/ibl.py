"""环境光：HDRI、Matcap、工作室光资源的读取，环境贴图预过滤，以及着色用的 GLSL 库。

方向约定（世界坐标 Y 轴向上、右手系；GLSL 和 numpy 两边用同一套公式）
====================================================================

单位方向 d = (x, y, z) 与等距柱状贴图坐标 (u, v) 互换::

    u = 0.5 + atan2(z, x) / (2π)          u ∈ [0, 1)，图的左边到右边
    v = 0.5 + asin(y) / π                 v ∈ [0, 1]，v = 1 是天顶，v = 0 是天底

    φ = (u − 0.5)·2π，θ = (v − 0.5)·π
    d = (cos θ·cos φ, sin θ, cos θ·sin φ)

- 图的正中 (u = 0.5) 是 +X；u = 0.25 是 −Z；u = 0.75 是 +Z；左右边缘 (u = 0 / 1) 是 −X。
- 默认相机在 +Z 看向 −Z，所以画面正中央看到的是 u = 0.25 那一列。
- 这和 Blender 的等距柱状约定一致：Blender 场景按常规的 Y 轴向上导出（OBJ / glTF 默认，
  (x, y, z) = (x_b, z_b, −y_b)）后，同一张 HDRI 在两边出现在同一个方向。
- 文件里的图像行序自上而下（第 0 行是天顶）。上传到显卡时上下翻转，所以 GLSL 里
  纹理坐标 t 就等于 v（和「UV 原点在左下角」的约定一致）。
- 环境旋转 u_env_rotation（弧度）绕 +Y 轴按右手定则转，从上往下看是逆时针。
  查询时先把世界方向反向旋转再换算 (u, v)，所以正的旋转让图里的景物往 u 减小的方向移动。

漫反射球谐
----------
``Environment.sh`` 是 9×3 系数，已经乘上余弦卷积因子并除以 π：在 GLSL 里按
基函数求和，得到的就是白色朗伯面的出射亮度（辐照度 / π）。常数亮度 L 的环境求出来正好是 L。
基函数顺序（x, y, z 是世界坐标方向分量）::

    0.282095,  0.488603·y,  0.488603·z,  0.488603·x,
    1.092548·xy,  1.092548·yz,  0.315392·(3z² − 1),  1.092548·xz,  0.546274·(x² − y²)

高光预过滤
----------
等距柱状，多级渐远的第 i 级是一个粗糙度 r_i 的 GGX 预过滤结果，最后一级 r = 1。
粗糙度与级的关系（GLSL 与 Python 一致）::

    lod = (N − 1)·r·(2 − r)          N = u_env_levels
    r_i = 1 − sqrt(1 − i / (N − 1))

第 0 级是镜面反射（原图缩到第 0 级宽度），最模糊一级默认宽 32 像素。

显示变换
--------
- 标准：sRGB 编码，超出 1 的部分截断（Blender 的 Standard）。
- Filmic：曲线和高光去饱和按 Blender 4.3「Filmic sRGB / Base Contrast」的输出拟合。
- AgX：结构沿用 Benjamin Wrensch「Minimal AgX」(https://iolite-engine.com/blog_posts/minimal_agx_implementation)
  与 Filament / three.js 的 AgX 写法（内缩矩阵 → log2 → S 形曲线 → 外扩矩阵 → 幂次解码）。
  S 形曲线采用 Blender AgX（Troy Sobotka 原作，Eary Chow 改进版）的带枢轴可调 S 形曲线，参数
  对比度 2.4、趾部 1.5、肩部 1.5；三个矩阵按 Blender 4.3「AgX Base sRGB」的实际输出重新拟合。
  拟合只用了 Blender 的输出数值，程序里不带 Blender 的 LUT。

着色器拼接
----------
GLSL 片段不带 ``#version``，拼在渲染器自己的 ``#version 430`` 之后。GLSL_PBR 依赖 GLSL_ENV，
要拼在它后面。``env_background`` 用到屏幕导数，只能用在片元着色器；在其他阶段拼入时先
``#define SPL_NO_DERIVATIVES``，改用 ``env_background_lod``。
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.props import FloatProperty, IntProperty, PropertyGroup
from ..paths import RESOURCE_DIR, ensure_vendor_path

log = logging.getLogger("splender.ibl")

HDRI_DIR = RESOURCE_DIR / "hdri"
MATCAP_DIR = RESOURCE_DIR / "matcap"
STUDIO_DIR = RESOURCE_DIR / "studio"

#: 显示变换编号，对应 GLSL tonemap() 的 transform 参数
TRANSFORM_STANDARD = 0
TRANSFORM_FILMIC = 1
TRANSFORM_AGX = 2
TRANSFORMS = {"STANDARD": TRANSFORM_STANDARD, "FILMIC": TRANSFORM_FILMIC, "AGX": TRANSFORM_AGX}

#: fp16 能表示的最大值，上传前夹到这里
HALF_MAX = 65504.0

_LABELS = {
    "hdri": {
        "city": "城市", "courtyard": "庭院", "forest": "森林", "interior": "室内",
        "night": "夜晚", "studio": "影棚", "sunrise": "日出", "sunset": "日落",
    },
    "studio": {"basic": "基础", "outdoor": "户外", "paint": "绘画", "rim": "轮廓光", "studio": "影棚"},
    "matcap": {
        "basic_1": "基础 1", "basic_2": "基础 2", "basic_dark": "基础 暗", "basic_side": "基础 侧光",
        "ceramic_dark": "陶瓷 暗", "ceramic_lightbulb": "陶瓷 灯泡", "check_normal+y": "法线检查",
        "check_rim_dark": "轮廓检查 暗", "check_rim_light": "轮廓检查 亮", "clay_brown": "黏土 棕",
        "clay_muddy": "黏土 泥", "clay_studio": "黏土 影棚", "jade": "玉石", "metal_anisotropic": "金属 拉丝",
        "metal_carpaint": "金属 车漆", "metal_lead": "金属 铅", "metal_shiny": "金属 亮面", "pearl": "珍珠",
        "reflection_check_horizontal": "反射检查 横", "reflection_check_vertical": "反射检查 竖",
        "resin": "树脂", "skin": "皮肤", "toon": "卡通",
    },
}


# ======================================================================
# 方向约定（numpy 版，和 GLSL 的 env_dir_to_uv / env_uv_to_dir 一致）
# ======================================================================

def direction_to_uv(d: np.ndarray) -> np.ndarray:
    """单位方向 (..., 3) → 等距柱状坐标 (..., 2)，v = 1 是天顶。"""
    d = np.asarray(d, dtype=np.float64)
    u = 0.5 + np.arctan2(d[..., 2], d[..., 0]) / (2.0 * math.pi)
    v = 0.5 + np.arcsin(np.clip(d[..., 1], -1.0, 1.0)) / math.pi
    return np.stack([np.mod(u, 1.0), v], axis=-1)


def uv_to_direction(uv: np.ndarray) -> np.ndarray:
    """等距柱状坐标 (..., 2) → 单位方向 (..., 3)。"""
    uv = np.asarray(uv, dtype=np.float64)
    phi = (uv[..., 0] - 0.5) * 2.0 * math.pi
    th = (uv[..., 1] - 0.5) * math.pi
    ct = np.cos(th)
    return np.stack([ct * np.cos(phi), np.sin(th), ct * np.sin(phi)], axis=-1)


def pixel_to_direction(col: float, row: float, width: int, height: int) -> np.ndarray:
    """图像像素坐标（行序自上而下，取像素中心时传 i + 0.5）→ 单位方向。"""
    return uv_to_direction(np.array([col / width, 1.0 - row / height]))


def direction_to_pixel(d: np.ndarray, width: int, height: int) -> np.ndarray:
    """单位方向 → 图像像素坐标 (列, 行)，行序自上而下。"""
    uv = direction_to_uv(d)
    return np.stack([uv[..., 0] * width, (1.0 - uv[..., 1]) * height], axis=-1)


def equirect_directions(width: int, height: int) -> np.ndarray:
    """每个像素中心的方向，形状 (height, width, 3)，行序自上而下。"""
    u = (np.arange(width) + 0.5) / width
    v = 1.0 - (np.arange(height) + 0.5) / height
    uu, vv = np.meshgrid(u, v)
    return uv_to_direction(np.stack([uu, vv], axis=-1))


def equirect_solid_angles(width: int, height: int) -> np.ndarray:
    """每一行像素的立体角，形状 (height, 1)。整张图加起来是 4π。"""
    edges = math.pi / 2.0 - np.arange(height + 1) * (math.pi / height)
    return ((np.sin(edges[:-1]) - np.sin(edges[1:])) * (2.0 * math.pi / width))[:, None]


def rotate_y(d: np.ndarray, angle: float) -> np.ndarray:
    """绕 +Y 轴按右手定则旋转 angle 弧度。"""
    c, s = math.cos(angle), math.sin(angle)
    d = np.asarray(d, dtype=np.float64)
    return np.stack([c * d[..., 0] + s * d[..., 2], d[..., 1], -s * d[..., 0] + c * d[..., 2]], axis=-1)


def brightest_direction(rgb: np.ndarray) -> np.ndarray:
    """等距柱状图里最亮的方向（按亮度，先做 3×3 平均避免单个噪点）。用于找太阳。"""
    img = np.asarray(rgb, dtype=np.float64)
    lum = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    pad = np.pad(lum, ((1, 1), (0, 0)), mode="edge")
    pad = np.concatenate([pad[:, -1:], pad, pad[:, :1]], axis=1)
    acc = sum(pad[dy:dy + lum.shape[0], dx:dx + lum.shape[1]] for dy in range(3) for dx in range(3))
    row, col = np.unravel_index(int(np.argmax(acc)), acc.shape)
    return pixel_to_direction(col + 0.5, row + 0.5, lum.shape[1], lum.shape[0])


# ======================================================================
# 球谐
# ======================================================================

#: 余弦卷积因子除以 π：第 0 阶 1，第 1 阶 2/3，第 2 阶 1/4
SH_IRRADIANCE_FACTORS = np.array([1.0, 2 / 3, 2 / 3, 2 / 3, 0.25, 0.25, 0.25, 0.25, 0.25])


def sh_basis(d: np.ndarray) -> np.ndarray:
    """9 个实球谐基函数在方向 d (..., 3) 上的值，形状 (..., 9)。顺序见模块说明。"""
    d = np.asarray(d, dtype=np.float64)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([
        np.full_like(x, 0.282095),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z, 0.546274 * (x * x - y * y),
    ], axis=-1)


def area_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """按面积平均缩小（能量守恒）。放大时用双线性。"""
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img
    if w % width == 0 and h % height == 0 and w >= width and h >= height:
        fx, fy = w // width, h // height
        return img.reshape(height, fy, width, fx, -1).mean(axis=(1, 3)).astype(np.float32)
    import cv2
    interp = cv2.INTER_AREA if (width <= w and height <= h) else cv2.INTER_LINEAR
    out = cv2.resize(img, (width, height), interpolation=interp)
    return out.reshape(height, width, -1).astype(np.float32)


def sh_project(rgb: np.ndarray, max_width: int = 256) -> np.ndarray:
    """把等距柱状图投影成漫反射球谐系数 (9, 3)，已含余弦卷积与 1/π。"""
    img = np.asarray(rgb, dtype=np.float32)
    h, w = img.shape[:2]
    tw = min(w, max(8, int(max_width)))
    th = max(4, int(round(tw * h / w)))
    small = area_resize(img, tw, th).astype(np.float64)
    dirs = equirect_directions(tw, th)
    basis = sh_basis(dirs)                                  # (th, tw, 9)
    weights = np.broadcast_to(equirect_solid_angles(tw, th), (th, tw))
    coeffs = np.einsum("hwk,hwc,hw->kc", basis, small, weights)
    return (coeffs * SH_IRRADIANCE_FACTORS[:, None]).astype(np.float32)


def sh_evaluate(sh: np.ndarray, d: np.ndarray) -> np.ndarray:
    """用系数求方向 d 上的漫反射亮度（和 GLSL env_diffuse 一致，未乘强度），形状 (..., 3)。"""
    return np.maximum(sh_basis(d) @ np.asarray(sh, dtype=np.float64), 0.0)


# ======================================================================
# 图像读取
# ======================================================================

def sanitize(rgb: np.ndarray) -> np.ndarray:
    """转成 float32 (H, W, 3)：去掉透明通道，灰度扩成三通道，NaN 置 0，负数置 0，无穷夹到 fp16 上限。"""
    a = np.asarray(rgb)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3:
        raise ValueError(f"图像形状不对：{a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.shape[2] >= 3:
        a = a[..., :3]
    else:
        raise ValueError(f"图像通道数不对：{a.shape}")
    a = a.astype(np.float32, copy=True)
    np.nan_to_num(a, copy=False, nan=0.0, posinf=HALF_MAX, neginf=0.0)
    np.maximum(a, 0.0, out=a)
    return np.ascontiguousarray(a)


def read_exr(path: str | Path) -> np.ndarray:
    """用 OpenEXR 读一张 EXR，返回 float32 (H, W, 3)，行序自上而下。"""
    ensure_vendor_path()
    import OpenEXR  # noqa: PLC0415

    with OpenEXR.File(str(path), separate_channels=True) as f:
        channels = {name: ch.pixels for name, ch in f.channels().items()}
    if not channels:
        raise ValueError(f"EXR 里没有像素通道：{path}")

    def pick(prefix: str) -> np.ndarray | None:
        names = [prefix + c for c in "RGB"]
        if all(n in channels for n in names):
            return np.stack([channels[n] for n in names], axis=-1)
        return None

    out = pick("")
    if out is None:
        for name in sorted(channels):
            if name.endswith(".R"):
                out = pick(name[:-1])
                if out is not None:
                    break
    if out is None:
        lum = channels.get("Y")
        if lum is None:
            lum = next(iter(channels.values()))
        out = np.repeat(np.asarray(lum)[..., None], 3, axis=2)
    return sanitize(out)


def read_hdr(path: str | Path) -> np.ndarray:
    """用 OpenCV 读 Radiance .hdr（支持中文路径），返回 float32 (H, W, 3) RGB。"""
    import cv2  # noqa: PLC0415

    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"读不出这张 HDR：{path}")
    return sanitize(img[..., ::-1])


def read_npz(path: str | Path) -> np.ndarray:
    with np.load(str(path)) as data:
        key = "rgb" if "rgb" in data.files else data.files[0]
        return sanitize(data[key])


def load_image(path: str | Path) -> np.ndarray:
    """按扩展名读 .npz / .hdr / .exr，返回线性 float32 (H, W, 3)，行序自上而下。"""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".npz":
        return read_npz(p)
    if ext in (".hdr", ".rgbe", ".pic"):
        return read_hdr(p)
    if ext == ".exr":
        return read_exr(p)
    raise ValueError(f"不支持的环境图格式：{p.suffix}")


def _is_path(name_or_path: str | Path) -> bool:
    if isinstance(name_or_path, Path):
        return True
    text = str(name_or_path)
    return any(ch in text for ch in "/\\:") or Path(text).suffix.lower() in (".npz", ".hdr", ".exr", ".json", ".sl")


def _list(folder: Path, suffix: str) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*" + suffix))


def label(name: str, kind: str = "hdri") -> str:
    """资源的中文显示名。kind：hdri、matcap、studio。没有译名时返回原名。"""
    return _LABELS.get(kind, {}).get(name, name)


# ---- HDRI ----

def list_hdris() -> list[str]:
    """自带 HDRI 的名字列表。"""
    return _list(HDRI_DIR, ".npz")


def hdri_path(name: str) -> Path:
    return HDRI_DIR / f"{name}.npz"


def load_hdri(name_or_path: str | Path) -> np.ndarray:
    """读一张环境图：自带名字（如 "forest"）或文件路径（.npz / .hdr / .exr）。返回线性 float32 (H, W, 3)。"""
    if _is_path(name_or_path) and Path(name_or_path).suffix:
        return load_image(name_or_path)
    path = hdri_path(str(name_or_path))
    if not path.is_file():
        raise FileNotFoundError(f"没有这张环境图：{name_or_path}")
    return read_npz(path)


def thumbnail(name: str) -> Path:
    """自带 HDRI 的缩略图路径（256×128 PNG）。"""
    path = HDRI_DIR / f"{name}.png"
    if not path.is_file():
        raise FileNotFoundError(f"没有这张环境图的缩略图：{name}")
    return path


# ---- Matcap ----

def list_matcaps() -> list[str]:
    return _list(MATCAP_DIR, ".npz")


def load_matcap(name_or_path: str | Path) -> np.ndarray:
    """读一张 Matcap：自带名字或文件路径。返回线性 float32 (H, W, 3)，行序自上而下。"""
    if _is_path(name_or_path) and Path(name_or_path).suffix:
        return load_image(name_or_path)
    path = MATCAP_DIR / f"{name_or_path}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"没有这个 Matcap：{name_or_path}")
    return read_npz(path)


def matcap_thumbnail(name: str) -> Path:
    path = MATCAP_DIR / f"{name}.png"
    if not path.is_file():
        raise FileNotFoundError(f"没有这个 Matcap 的缩略图：{name}")
    return path


# ---- 工作室光 ----

def list_studio_lights() -> list[str]:
    return _list(STUDIO_DIR, ".json")


def parse_studio_sl(text: str, name: str = "") -> dict:
    """解析 Blender 的工作室光预设（.sl，键值文本），返回和 resources/studio/*.json 相同的字典。"""
    values: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = float(parts[1])
        except ValueError:
            continue

    def vec(prefix: str) -> list[float]:
        return [values.get(f"{prefix}.{c}", 0.0) for c in "xyz"]

    lights = []
    for i in range(4):
        p = f"light[{i}]"
        direction = np.array(vec(p + ".vec"), dtype=np.float64)
        length = float(np.linalg.norm(direction))
        lights.append({
            "enabled": bool(values.get(p + ".flag", 0.0)),
            "direction": (direction / length).tolist() if length > 1e-8 else [0.0, 0.0, 1.0],
            "diffuse": vec(p + ".col"),
            "specular": vec(p + ".spec"),
            "smooth": values.get(p + ".smooth", 0.0),
        })
    return {"name": name, "space": "view", "ambient": vec("light_ambient"), "lights": lights}


def load_studio_light(name_or_path: str | Path) -> dict:
    """读工作室光：自带名字、.json，或 Blender 的 .sl 文件。

    返回 {"name", "space": "view", "ambient": [r,g,b], "lights": [4 × {"enabled", "direction",
    "diffuse", "specular", "smooth"}]}。方向在视图空间：X 向右、Y 向上、Z 指向观察者，指向灯。
    """
    if _is_path(name_or_path) and Path(name_or_path).suffix:
        p = Path(name_or_path)
        if p.suffix.lower() == ".sl":
            return parse_studio_sl(p.read_text(encoding="utf-8", errors="replace"), p.stem)
        return json.loads(p.read_text(encoding="utf-8"))
    path = STUDIO_DIR / f"{name_or_path}.json"
    if not path.is_file():
        raise FileNotFoundError(f"没有这个工作室光：{name_or_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def studio_thumbnail(name: str) -> Path:
    path = STUDIO_DIR / f"{name}.png"
    if not path.is_file():
        raise FileNotFoundError(f"没有这个工作室光的缩略图：{name}")
    return path


# ======================================================================
# 显示变换（numpy 版，和 GLSL_TONEMAP 一致；用于缩略图和测试）
# ======================================================================

_AGX_LOG_MIN = -12.47393
_AGX_LOG_MAX = 4.026069
_AGX_SLOPE = 2.4
_AGX_TOE = 1.5
_AGX_SHOULDER = 1.5
_AGX_PIVOT_X = 10.0 / 16.5
_AGX_PIVOT_Y = 0.18 ** (1.0 / 2.4)
_AGX_IN = np.array([
    [0.546667680942795, 0.32748324794988865, 0.1258490711073163],
    [0.1360294338155839, 0.7471025400701325, 0.11686802611428351],
    [0.1145785058558556, 0.19371978260482298, 0.6917017115393214]])
_AGX_OUT = np.array([
    [0.9697101994007058, 0.035915020679113945, -0.005625220079819716],
    [0.014696117185316894, 1.0141508999854987, -0.028847017170815545],
    [-0.02369039741190403, 0.014679238674602097, 1.0090111587373018]])
_AGX_POST = np.array([
    [2.0210798830584467, -0.7554913868033621, -0.2655884962550844],
    [-0.31125807988942916, 1.3113932849063326, -0.00013520501690350974],
    [-0.23299218809370464, -0.2714327926028438, 1.5044249806965484]])

_FILMIC_LOG_MIN = -12.473931188
_FILMIC_LOG_RANGE = 16.5
_FILMIC_PIVOT_X = 0.6354359458227553
_FILMIC_PIVOT_Y = 0.564164819801117
_FILMIC_SLOPE = 2.1762946603035744
_FILMIC_TOE = 2.7523484780872742
_FILMIC_SHOULDER = 2.6705347173712286
#: 高光去饱和权重 w = exp2(a + b·L + c·L²)·(1 − exp(−4 t²))，L = log2(最大通道)，t = max(L − x0, 0)
_FILMIC_DESAT = (-19.396179011126584, 2.562729187387057, -0.09691842445298048, 3.6836896386973765)
_FILMIC_DESAT_LOG_MAX = 12.526068812


def _sigmoid_scales(xp: float, yp: float, slope: float, toe: float, shoulder: float) -> tuple[float, float]:
    """带枢轴 S 形曲线两侧的缩放，使曲线经过 (0, 0) 和 (1, 1)。"""
    t_sh = ((slope * (1.0 - xp) / (1.0 - yp)) ** shoulder - 1.0) ** (1.0 / shoulder)
    t_to = ((slope * xp / yp) ** toe - 1.0) ** (1.0 / toe)
    return slope * (1.0 - xp) / t_sh, slope * xp / t_to


_AGX_S_SH, _AGX_S_TO = _sigmoid_scales(_AGX_PIVOT_X, _AGX_PIVOT_Y, _AGX_SLOPE, _AGX_TOE, _AGX_SHOULDER)
_FILMIC_S_SH, _FILMIC_S_TO = _sigmoid_scales(_FILMIC_PIVOT_X, _FILMIC_PIVOT_Y, _FILMIC_SLOPE, _FILMIC_TOE,
                                             _FILMIC_SHOULDER)


def srgb_encode(c: np.ndarray) -> np.ndarray:
    """线性 → sRGB 编码（先夹到 0..1）。"""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def srgb_decode(c: np.ndarray) -> np.ndarray:
    """sRGB 编码 → 线性。"""
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def _sigmoid_np(x, xp, yp, slope, s_to, p_to, s_sh, p_sh):
    d = x - xp
    ts = np.maximum(d, 0.0) * (slope / s_sh)
    tt = np.maximum(-d, 0.0) * (slope / s_to)
    sh = yp + s_sh * ts / np.power(1.0 + np.power(ts, p_sh), 1.0 / p_sh)
    to = yp - s_to * tt / np.power(1.0 + np.power(tt, p_to), 1.0 / p_to)
    return np.where(d >= 0.0, sh, to)


def _agx_np(c: np.ndarray) -> np.ndarray:
    v = c @ _AGX_IN.T
    x = np.clip((np.log2(np.maximum(v, 1e-10)) - _AGX_LOG_MIN) / (_AGX_LOG_MAX - _AGX_LOG_MIN), 0.0, 1.0)
    s = _sigmoid_np(x, _AGX_PIVOT_X, _AGX_PIVOT_Y, _AGX_SLOPE, _AGX_S_TO, _AGX_TOE, _AGX_S_SH, _AGX_SHOULDER)
    lin = np.power(np.maximum(s @ _AGX_OUT.T, 0.0), 2.4)
    return srgb_encode(np.clip(lin @ _AGX_POST.T, 0.0, 1.0))


def _filmic_np(c: np.ndarray) -> np.ndarray:
    a, b, cq, x0 = _FILMIC_DESAT
    m = np.max(c, axis=-1, keepdims=True)
    lx = np.minimum(np.log2(np.maximum(m, 1e-10)), _FILMIC_DESAT_LOG_MAX)
    t = np.maximum(lx - x0, 0.0)
    w = np.clip(np.exp2(a + b * lx + cq * lx * lx) * (1.0 - np.exp(-4.0 * t * t)), 0.0, 1.0)
    c = c + (m - c) * w
    x = np.clip((np.log2(np.maximum(c, 1e-10)) - _FILMIC_LOG_MIN) / _FILMIC_LOG_RANGE, 0.0, 1.0)
    return np.clip(_sigmoid_np(x, _FILMIC_PIVOT_X, _FILMIC_PIVOT_Y, _FILMIC_SLOPE, _FILMIC_S_TO, _FILMIC_TOE,
                               _FILMIC_S_SH, _FILMIC_SHOULDER), 0.0, 1.0)


def tonemap(rgb: np.ndarray, transform: int | str = TRANSFORM_AGX, exposure_ev: float = 0.0) -> np.ndarray:
    """线性亮度 → 可直接显示的 sRGB 编码值（0..1，float64）。和 GLSL tonemap() 一致。"""
    if isinstance(transform, str):
        transform = TRANSFORMS[transform.upper()]
    c = np.nan_to_num(np.asarray(rgb, dtype=np.float64), nan=0.0, posinf=1e30, neginf=0.0)
    c = np.minimum(np.maximum(c * (2.0 ** float(exposure_ev)), 0.0), 1e30)
    if transform == TRANSFORM_FILMIC:
        return _filmic_np(c)
    if transform == TRANSFORM_AGX:
        return _agx_np(c)
    return srgb_encode(c)


def to_uint8(srgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(srgb) * 255.0), 0, 255).astype(np.uint8)


# ======================================================================
# 工作室光与 Matcap 的 numpy 版（缩略图、测试用；和 GLSL 一致）
# ======================================================================

def _studio_arrays(data: dict) -> tuple[np.ndarray, ...]:
    lights = list(data.get("lights", []))[:4]
    while len(lights) < 4:
        lights.append({"enabled": False})
    dirs, cols, specs, smooth = [], [], [], []
    for light in lights:
        on = bool(light.get("enabled", True))
        d = np.array(light.get("direction", [1.0, 0.0, 0.0]), dtype=np.float64)
        n = np.linalg.norm(d)
        dirs.append(d / n if n > 1e-8 else np.array([1.0, 0.0, 0.0]))
        cols.append(np.array(light.get("diffuse", [0, 0, 0]), dtype=np.float64) * on)
        specs.append(np.array(light.get("specular", [0, 0, 0]), dtype=np.float64) * on)
        smooth.append(float(light.get("smooth", 0.0)) * on)
    amb = np.array(data.get("ambient", [0, 0, 0]), dtype=np.float64)
    return np.array(dirs), np.array(cols), np.array(specs), np.array(smooth), amb


def shade_studio_np(n: np.ndarray, v: np.ndarray, base_color: Iterable[float], data: dict,
                    roughness: float = 0.4, metallic: float = 0.0, specular: bool = True) -> np.ndarray:
    """工作室光着色（Blender workbench 算法），n、v 在视图空间，形状 (..., 3)。返回线性亮度。"""
    n = np.asarray(n, dtype=np.float64)
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    v = np.broadcast_to(np.asarray(v, dtype=np.float64), n.shape)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    base = np.broadcast_to(np.asarray(base_color, dtype=np.float64), n.shape)
    dirs, cols, specs, wrap, amb = _studio_arrays(data)

    def wrapped(nl, w):
        w1 = w + 1.0
        return np.clip((nl + w) / (w1 * w1), 0.0, 1.0)

    if specular:
        diff_color = base * (1.0 - metallic)
        spec_color = 0.05 * (1.0 - metallic) + base * metallic
    else:
        diff_color = base
        spec_color = np.zeros_like(base)
    spec_light = np.broadcast_to(amb, n.shape).copy()
    diff_light = np.broadcast_to(amb, n.shape).copy()
    if specular:
        r = 2.0 * np.sum(n * v, axis=-1, keepdims=True) * n - v
        wrap_nl = r @ dirs.T                                            # (..., 4)
        half = dirs[None, :, :] + v.reshape(-1, 1, 3)
        half = half / np.maximum(np.linalg.norm(half, axis=-1, keepdims=True), 1e-12)
        spec_angle = np.clip(np.sum(half * n.reshape(-1, 1, 3), axis=-1), 0.0, 1.0).reshape(wrap_nl.shape)
        spec_nl = np.clip(n @ dirs.T, 0.0, 1.0)
        gloss = (1.0 - roughness) * (1.0 - wrap)
        shininess = np.exp2(10.0 * gloss + 1.0)
        spec = np.power(spec_angle, shininess) * spec_nl * (shininess * 0.125 + 1.0)
        w = wrap + (1.0 - wrap) * roughness
        spec_env = wrapped(wrap_nl, w)
        spec = spec + (spec_env - spec) * (wrap * wrap)
        spec_light = spec_light + spec @ specs
        nv = np.clip(np.sum(n * v, axis=-1, keepdims=True), 0.0, 1.0)
        fresnel = np.exp2(-8.35 * nv) * (1.0 - roughness)
        spec_color = spec_color + (1.0 - spec_color) * fresnel
    spec_light = spec_light * spec_color
    diff = wrapped(n @ dirs.T, wrap)
    diff_light = diff_light + diff @ cols
    spec_energy = np.mean(spec_color, axis=-1, keepdims=True)
    return diff_light * diff_color * (1.0 - spec_energy) + spec_light


def sphere_normals(size: int, supersample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """正交相机看单位球：返回 (法线 (H, W, 3)，覆盖遮罩 (H, W))，视图空间，行序自上而下。"""
    n = size * supersample
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x, y = np.meshgrid(c, -c)
    r2 = x * x + y * y
    mask = r2 < 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    return np.stack([x, y, z], axis=-1), mask


def _downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    return img.reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3))


def make_hdri_thumbnail(rgb: np.ndarray, width: int = 256, height: int = 128,
                        transform: int = TRANSFORM_AGX, exposure_ev: float = 0.0) -> np.ndarray:
    """环境图缩略图：按面积缩小后做显示变换，返回 uint8 (height, width, 3)。"""
    small = area_resize(sanitize(rgb), width, height)
    return to_uint8(tonemap(small, transform, exposure_ev))


def make_matcap_thumbnail(rgb: np.ndarray, size: int = 128) -> np.ndarray:
    """Matcap 缩略图：按面积缩小，标准 sRGB 编码（和实体模式一致），uint8。"""
    small = area_resize(sanitize(rgb), size, size)
    return to_uint8(tonemap(small, TRANSFORM_STANDARD))


def make_studio_thumbnail(data: dict, size: int = 128, base_color=(0.8, 0.8, 0.8), roughness: float = 0.4,
                          background=(0.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    """工作室光缩略图：正交看一个灰球，返回 uint8 RGBA (size, size, 4)，球外透明。"""
    ss = 4
    normals, mask = sphere_normals(size, ss)
    rgb = shade_studio_np(normals, np.array([0.0, 0.0, 1.0]), base_color, data, roughness=roughness)
    srgb = tonemap(rgb, TRANSFORM_STANDARD)
    alpha = mask.astype(np.float64)[..., None]
    bg = np.array(background, dtype=np.float64)
    rgba = np.concatenate([srgb * alpha, alpha], axis=-1)
    rgba = _downsample(rgba, ss)
    out_alpha = rgba[..., 3:4]
    color = np.where(out_alpha > 1e-6, rgba[..., :3] / np.maximum(out_alpha, 1e-6), bg[:3])
    return to_uint8(np.concatenate([color, out_alpha + bg[3] * (1.0 - out_alpha)], axis=-1))


# ======================================================================
# GLSL
# ======================================================================

def _f(x: float) -> str:
    return f"{float(x):.10g}"


def _mat3(m: np.ndarray) -> str:
    cols = [", ".join(_f(m[r][c]) for r in range(3)) for c in range(3)]
    return "mat3(" + ", ".join(f"vec3({col})" for col in cols) + ")"


_GLSL_EQUIRECT = """
vec2 env_dir_to_uv(vec3 d) {
    return vec2(0.5 + atan(d.z, d.x) * (0.5 / SPL_PI), 0.5 + asin(clamp(d.y, -1.0, 1.0)) * (1.0 / SPL_PI));
}

vec3 env_uv_to_dir(vec2 uv) {
    float phi = (uv.x - 0.5) * (2.0 * SPL_PI);
    float th = (uv.y - 0.5) * SPL_PI;
    return vec3(cos(th) * cos(phi), sin(th), cos(th) * sin(phi));
}
"""

_GLSL_PI = """
#ifndef SPL_PI
#define SPL_PI 3.14159265358979
#endif
"""

GLSL_ENV = _GLSL_PI + """
// ==== SPLENDER 环境光（engine/ibl.py GLSL_ENV）====
// 方向一律是世界空间单位向量。u_env_rotation 是绕 +Y 的弧度。
uniform sampler2D u_env_spec;      // 高光预过滤，等距柱状，第 i 级 = 一个粗糙度
uniform sampler2D u_env_bg;        // 背景，等距柱状，带多级渐远
uniform sampler2D u_env_brdf;      // 环境 BRDF 表（x = N·V，y = 粗糙度）
uniform vec3 u_env_sh[9];          // 漫反射球谐（已含余弦卷积与 1/π）
uniform float u_env_rotation;
uniform float u_env_strength;
uniform int u_env_levels;
""" + _GLSL_EQUIRECT + """
// 世界方向 → 环境图自己的方向（撤销旋转）
vec3 env_to_local(vec3 d) {
    float c = cos(u_env_rotation);
    float s = sin(u_env_rotation);
    return vec3(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}

float env_roughness_lod(float roughness) {
    float r = clamp(roughness, 0.0, 1.0);
    return r * (2.0 - r) * float(max(u_env_levels - 1, 0));
}

vec3 env_sh_eval(vec3 n) {
    vec3 c = u_env_sh[0] * 0.282095
           + u_env_sh[1] * (0.488603 * n.y)
           + u_env_sh[2] * (0.488603 * n.z)
           + u_env_sh[3] * (0.488603 * n.x)
           + u_env_sh[4] * (1.092548 * n.x * n.y)
           + u_env_sh[5] * (1.092548 * n.y * n.z)
           + u_env_sh[6] * (0.315392 * (3.0 * n.z * n.z - 1.0))
           + u_env_sh[7] * (1.092548 * n.x * n.z)
           + u_env_sh[8] * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(c, vec3(0.0));
}

// 白色朗伯面在法线 n 处的出射亮度（辐照度 / π），已乘强度
vec3 env_diffuse(vec3 n) {
    return env_sh_eval(env_to_local(normalize(n))) * u_env_strength;
}

// 反射方向 r、粗糙度 roughness（0..1，感知粗糙度）的预过滤环境亮度，已乘强度
vec3 env_specular(vec3 r, float roughness) {
    vec2 uv = env_dir_to_uv(env_to_local(normalize(r)));
    return textureLod(u_env_spec, uv, env_roughness_lod(roughness)).rgb * u_env_strength;
}

// 在预过滤贴图的某一级做三次 B 样条采样（4 次双线性），放大显示时没有菱形折痕
vec3 env_bspline_level(vec2 uv, int level) {
    vec2 size = vec2(textureSize(u_env_spec, level));
    vec2 p = uv * size - 0.5;
    vec2 i = floor(p);
    vec2 f = p - i;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;
    vec2 w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    vec2 w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    vec2 w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    vec2 w3 = f3 / 6.0;
    vec2 g0 = w0 + w1;
    vec2 g1 = w2 + w3;
    vec2 p0 = (i - 0.5 + w1 / g0) / size;
    vec2 p1 = (i + 1.5 + w3 / g1) / size;
    float lv = float(level);
    vec3 a = textureLod(u_env_spec, vec2(p0.x, p0.y), lv).rgb * g0.x + textureLod(u_env_spec, vec2(p1.x, p0.y), lv).rgb * g1.x;
    vec3 b = textureLod(u_env_spec, vec2(p0.x, p1.y), lv).rgb * g0.x + textureLod(u_env_spec, vec2(p1.x, p1.y), lv).rgb * g1.x;
    return a * g0.y + b * g1.y;
}

vec3 env_spec_smooth(vec2 uv, float lod) {
    int top = max(u_env_levels - 1, 0);
    float l = clamp(lod, 0.0, float(top));
    int l0 = int(floor(l));
    int l1 = min(l0 + 1, top);
    float t = l - float(l0);
    vec3 a = env_bspline_level(uv, l0);
    if (t <= 0.0 || l1 == l0) {
        return a;
    }
    return mix(a, env_bspline_level(uv, l1), t);
}

// 背景（不依赖屏幕导数的版本）：blur > 0 时用预过滤贴图，lod 是背景贴图的级
vec3 env_background_lod(vec3 dir, float blur, float lod) {
    vec2 uv = env_dir_to_uv(env_to_local(normalize(dir)));
    vec3 c = blur > 0.0 ? env_spec_smooth(uv, env_roughness_lod(blur)) : textureLod(u_env_bg, uv, lod).rgb;
    return c * u_env_strength;
}

#ifndef SPL_NO_DERIVATIVES
// 背景：dir 是视线方向，blur 0..1（0 = 清晰）。已乘强度。只能在片元着色器里用。
vec3 env_background(vec3 dir, float blur) {
    vec2 uv = env_dir_to_uv(env_to_local(normalize(dir)));
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    dx.x -= floor(dx.x + 0.5);     // 接缝处 u 从 1 跳到 0，导数按最短方向算
    dy.x -= floor(dy.x + 0.5);
    vec3 c = blur > 0.0 ? env_spec_smooth(uv, env_roughness_lod(blur)) : textureGrad(u_env_bg, uv, dx, dy).rgb;
    return c * u_env_strength;
}
#endif

// split-sum 的环境 BRDF：返回 (A, B)，镜面反射率 = F0·A + F90·B
vec2 env_brdf(float nv, float roughness) {
    vec2 size = vec2(textureSize(u_env_brdf, 0));
    vec2 uv = (clamp(vec2(nv, roughness), 0.0, 1.0) * (size - 1.0) + 0.5) / size;
    return textureLod(u_env_brdf, uv, 0.0).rg;
}
"""

GLSL_PBR = """
// ==== SPLENDER PBR（engine/ibl.py GLSL_PBR，要拼在 GLSL_ENV 后面）====
// 金属度工作流，GGX + 多次散射能量补偿（Fdez-Agüera 2019），split-sum 环境光。
// n：世界空间法线；v：从表面指向相机的单位向量；base_color：线性基础色；roughness：感知粗糙度。
// 返回线性亮度（未做显示变换）。

float pbr_specular_occlusion(float nv, float ao, float alpha) {
    return clamp(pow(nv + ao, exp2(-16.0 * alpha - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

vec3 shade_pbr(vec3 n, vec3 v, vec3 base_color, float metallic, float roughness, float ao) {
    n = normalize(n);
    v = normalize(v);
    float nv = dot(n, v);
    if (nv < 1e-3) {
        // 插值法线或法线贴图背对视线时，把法线掰回可见半球，边缘不会发黑
        n = normalize(n + v * (1e-3 - nv));
        nv = 1e-3;
    }
    float r = clamp(roughness, 0.0, 1.0);
    float m = clamp(metallic, 0.0, 1.0);
    float a = r * r;
    vec3 refl = reflect(-v, n);
    // 粗糙表面的反射主方向偏向法线（Lagarde & de Rousiers 2014）
    vec3 dir = normalize(mix(n, refl, (1.0 - a) * (sqrt(1.0 - a) + a)));
    vec3 f0 = mix(vec3(0.04), base_color, m);
    vec2 ab = env_brdf(nv, r);
    vec3 fss_ess = f0 * ab.x + ab.y;
    float ems = max(1.0 - (ab.x + ab.y), 0.0);
    vec3 f_avg = f0 + (1.0 - f0) / 21.0;
    vec3 fms_ems = ems * fss_ess * f_avg / (1.0 - f_avg * ems);
    vec3 kd = base_color * (1.0 - m) * max(vec3(1.0) - fss_ess - fms_ems, vec3(0.0));
    float occ = clamp(ao, 0.0, 1.0);
    vec3 spec = env_specular(dir, r) * fss_ess * pbr_specular_occlusion(nv, occ, a);
    vec3 diff = env_diffuse(n) * (fms_ems + kd) * occ;
    return spec + diff;
}
"""

GLSL_TONEMAP = f"""
// ==== SPLENDER 显示变换（engine/ibl.py GLSL_TONEMAP）====
// transform：0 标准，1 Filmic，2 AgX。exposure_ev 是曝光（档）。返回 sRGB 编码值，可直接显示。

vec3 tm_srgb_encode(vec3 c) {{
    c = clamp(c, 0.0, 1.0);
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}}

vec3 tm_srgb_decode(vec3 c) {{
    c = clamp(c, 0.0, 1.0);
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}}

// 带枢轴的可调 S 形曲线：过 (0,0)、(xp,yp)、(1,1)，枢轴处斜率 slope
vec3 tm_sigmoid(vec3 x, float xp, float yp, float slope, float s_to, float p_to, float s_sh, float p_sh) {{
    vec3 d = x - xp;
    vec3 ts = max(d, 0.0) * (slope / s_sh);
    vec3 tt = max(-d, 0.0) * (slope / s_to);
    vec3 sh = yp + s_sh * ts / pow(1.0 + pow(ts, vec3(p_sh)), vec3(1.0 / p_sh));
    vec3 to = yp - s_to * tt / pow(1.0 + pow(tt, vec3(p_to)), vec3(1.0 / p_to));
    return mix(to, sh, step(0.0, d));
}}

vec3 tm_agx(vec3 c) {{
    const mat3 m_in = {_mat3(_AGX_IN)};
    const mat3 m_out = {_mat3(_AGX_OUT)};
    const mat3 m_post = {_mat3(_AGX_POST)};
    vec3 v = m_in * c;
    vec3 x = clamp((log2(max(v, vec3(1e-10))) - ({_f(_AGX_LOG_MIN)})) / {_f(_AGX_LOG_MAX - _AGX_LOG_MIN)}, 0.0, 1.0);
    vec3 s = tm_sigmoid(x, {_f(_AGX_PIVOT_X)}, {_f(_AGX_PIVOT_Y)}, {_f(_AGX_SLOPE)}, {_f(_AGX_S_TO)}, {_f(_AGX_TOE)},
                        {_f(_AGX_S_SH)}, {_f(_AGX_SHOULDER)});
    vec3 lin = pow(max(m_out * s, vec3(0.0)), vec3(2.4));
    return tm_srgb_encode(clamp(m_post * lin, 0.0, 1.0));
}}

vec3 tm_filmic(vec3 c) {{
    float m = max(max(c.r, c.g), c.b);
    float lx = min(log2(max(m, 1e-10)), {_f(_FILMIC_DESAT_LOG_MAX)});
    float t = max(lx - ({_f(_FILMIC_DESAT[3])}), 0.0);
    float w = clamp(exp2({_f(_FILMIC_DESAT[0])} + {_f(_FILMIC_DESAT[1])} * lx + ({_f(_FILMIC_DESAT[2])}) * lx * lx)
                    * (1.0 - exp(-4.0 * t * t)), 0.0, 1.0);
    c = c + (m - c) * w;
    vec3 x = clamp((log2(max(c, vec3(1e-10))) - ({_f(_FILMIC_LOG_MIN)})) / {_f(_FILMIC_LOG_RANGE)}, 0.0, 1.0);
    return clamp(tm_sigmoid(x, {_f(_FILMIC_PIVOT_X)}, {_f(_FILMIC_PIVOT_Y)}, {_f(_FILMIC_SLOPE)}, {_f(_FILMIC_S_TO)},
                            {_f(_FILMIC_TOE)}, {_f(_FILMIC_S_SH)}, {_f(_FILMIC_SHOULDER)}), 0.0, 1.0);
}}

vec3 tonemap(vec3 linear_rgb, int transform, float exposure_ev) {{
    vec3 c = mix(linear_rgb, vec3(0.0), isnan(linear_rgb));
    c = clamp(c * exp2(exposure_ev), 0.0, 1e30);
    if (transform == 1) {{
        return tm_filmic(c);
    }}
    if (transform == 2) {{
        return tm_agx(c);
    }}
    return tm_srgb_encode(c);
}}

// 输出到 8 位屏幕前加一点三角分布抖动，消除平滑渐变里的色带。frag_coord 用 gl_FragCoord.xy。
vec3 tonemap_dither(vec3 srgb, vec2 frag_coord) {{
    uvec2 p = uvec2(frag_coord);
    uint h = p.x * 1664525u + p.y * 1013904223u;
    h ^= h >> 16u; h *= 0x7feb352du; h ^= h >> 15u; h *= 0x846ca68bu; h ^= h >> 16u;
    uint k = h * 747796405u + 2891336453u;
    k ^= k >> 16u; k *= 0x7feb352du; k ^= k >> 15u;
    float n = float(h & 0xffffu) / 65535.0 + float(k & 0xffffu) / 65535.0 - 1.0;
    return srgb + n / 255.0;
}}
"""

GLSL_STUDIO = """
// ==== SPLENDER 工作室光（engine/ibl.py GLSL_STUDIO）====
// 算法同 Blender workbench 的工作室光。方向在视图空间（X 右、Y 上、Z 指向观察者），指向灯。
// n_view：视图空间法线；v_view：从表面指向相机的单位向量；返回线性亮度（按 Blender 实体模式用标准显示变换）。
uniform vec3 u_studio_dir[4];
uniform vec3 u_studio_col[4];
uniform vec3 u_studio_spec[4];
uniform float u_studio_smooth[4];
uniform vec3 u_studio_ambient;
uniform float u_studio_roughness;   // 材质粗糙度，Blender 默认 0.4
uniform float u_studio_metallic;    // 材质金属度，默认 0
uniform int u_studio_specular;      // 1 = 计算高光（Blender 实体模式的「高光照明」）

vec4 studio_wrapped(vec4 nl, vec4 w) {
    vec4 w1 = w + 1.0;
    return clamp((nl + w) / (w1 * w1), 0.0, 1.0);
}

vec3 shade_studio_ex(vec3 n_view, vec3 v_view, vec3 base_color, float roughness, float metallic) {
    vec3 n = normalize(n_view);
    vec3 v = normalize(v_view);
    bool use_spec = u_studio_specular != 0;
    vec3 diff_color = use_spec ? base_color * (1.0 - metallic) : base_color;
    vec3 spec_color = use_spec ? mix(vec3(0.05), base_color, metallic) : vec3(0.0);
    vec4 wrap = vec4(u_studio_smooth[0], u_studio_smooth[1], u_studio_smooth[2], u_studio_smooth[3]);
    vec3 spec_light = u_studio_ambient;
    vec3 diff_light = u_studio_ambient;
    if (use_spec) {
        vec3 r = -reflect(v, n);
        vec4 spec_angle = vec4(0.0);
        vec4 spec_nl = vec4(0.0);
        vec4 wrap_nl = vec4(0.0);
        for (int i = 0; i < 4; ++i) {
            vec3 l = u_studio_dir[i];
            wrap_nl[i] = dot(l, r);
            spec_angle[i] = clamp(dot(normalize(l + v), n), 0.0, 1.0);
            spec_nl[i] = clamp(dot(l, n), 0.0, 1.0);
        }
        vec4 gloss = vec4(1.0 - roughness) * (1.0 - wrap);
        vec4 shininess = exp2(10.0 * gloss + 1.0);
        vec4 spec = pow(spec_angle, shininess) * spec_nl * (shininess * 0.125 + 1.0);
        vec4 w = mix(wrap, vec4(1.0), roughness);
        spec = mix(spec, studio_wrapped(wrap_nl, w), wrap * wrap);
        spec_light += spec.x * u_studio_spec[0] + spec.y * u_studio_spec[1]
                    + spec.z * u_studio_spec[2] + spec.w * u_studio_spec[3];
        float nv = clamp(dot(n, v), 0.0, 1.0);
        spec_color = mix(spec_color, vec3(1.0), exp2(-8.35 * nv) * (1.0 - roughness));
    }
    spec_light *= spec_color;
    vec4 diff_nl = vec4(dot(u_studio_dir[0], n), dot(u_studio_dir[1], n), dot(u_studio_dir[2], n), dot(u_studio_dir[3], n));
    vec4 diff = studio_wrapped(diff_nl, wrap);
    diff_light += diff.x * u_studio_col[0] + diff.y * u_studio_col[1] + diff.z * u_studio_col[2] + diff.w * u_studio_col[3];
    float spec_energy = dot(spec_color, vec3(0.33333));
    return diff_light * diff_color * (1.0 - spec_energy) + spec_light;
}

vec3 shade_studio(vec3 n_view, vec3 v_view, vec3 base_color) {
    return shade_studio_ex(n_view, v_view, base_color, u_studio_roughness, u_studio_metallic);
}
"""

GLSL_MATCAP = """
// ==== SPLENDER Matcap（engine/ibl.py GLSL_MATCAP）====
// 算法同 Blender workbench。n_view 是视图空间法线。返回线性亮度（按 Blender 实体模式用标准显示变换）。
uniform sampler2D u_matcap;
uniform int u_matcap_flip;          // 1 = 左右翻转

// 透视相机下用视线方向修正，球边缘不会被拉伸；正交相机 v_view = (0, 0, 1)
vec2 matcap_uv(vec3 n_view, vec3 v_view) {
    vec3 i = normalize(v_view);
    float a = 1.0 / (1.0 + i.z);
    float b = -i.x * i.y * a;
    vec3 b1 = vec3(1.0 - i.x * i.x * a, b, -i.x);
    vec3 b2 = vec3(b, 1.0 - i.y * i.y * a, -i.y);
    vec2 uv = vec2(dot(b1, n_view), dot(b2, n_view));
    if (u_matcap_flip != 0) {
        uv.x = -uv.x;
    }
    return uv * 0.496 + 0.5;
}

vec3 shade_matcap(vec3 n_view, vec3 v_view, vec3 base_color) {
    return textureLod(u_matcap, matcap_uv(normalize(n_view), v_view), 0.0).rgb * base_color;
}

vec3 shade_matcap(vec3 n_view, vec3 base_color) {
    return shade_matcap(n_view, vec3(0.0, 0.0, 1.0), base_color);
}
"""


# ======================================================================
# 内部着色器：预过滤与 BRDF 表
# ======================================================================

_VS_FULLSCREEN = """
#version 430
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_GLSL_HAMMERSLEY = """
vec2 hammersley(uint i, uint n) {
    return vec2(float(i) / float(n), float(bitfieldReverse(i)) * 2.3283064365386963e-10);
}
"""

_FS_PREFILTER = "#version 430\n" + _GLSL_PI + _GLSL_EQUIRECT + _GLSL_HAMMERSLEY + """
uniform sampler2D u_src;
uniform vec2 u_dst_size;
uniform float u_alpha;
uniform int u_samples;
uniform float u_footprint;     // 采样足迹放大倍数（2 的幂次）
uniform float u_copy_lod;      // >= 0：直接按这个级复制（镜面级）
out vec4 frag;

void main() {
    vec2 uv = gl_FragCoord.xy / u_dst_size;
    if (u_copy_lod >= 0.0) {
        frag = vec4(textureLod(u_src, uv, u_copy_lod).rgb, 1.0);
        return;
    }
    vec3 n = env_uv_to_dir(uv);
    // 东-北切线框架：除两极外处处连续（纹素中心不会落在极点上）
    vec3 t = normalize(cross(vec3(0.0, 1.0, 0.0), n));
    vec3 b = cross(n, t);
    float a2 = u_alpha * u_alpha;
    uint count = uint(u_samples);
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (uint i = 0u; i < count; ++i) {
        vec2 xi = hammersley(i, count);
        float c2 = (1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y);
        float ct = sqrt(c2);
        float st = sqrt(max(1.0 - c2, 0.0));
        float ph = 2.0 * SPL_PI * xi.x;
        vec3 h = t * (st * cos(ph)) + b * (st * sin(ph)) + n * ct;
        vec3 l = 2.0 * ct * h - n;
        float nl = dot(n, l);
        if (nl <= 0.0) {
            continue;
        }
        // 过滤重要性采样（Křivánek & Colbert 2008）：每个样本读一块与其立体角相当的区域
        float dd = c2 * (a2 - 1.0) + 1.0;
        float pdf = a2 / (SPL_PI * dd * dd) * 0.25;
        float radius = sqrt(1.0 / (float(count) * pdf)) * u_footprint;
        float cos_el = max(sqrt(max(1.0 - l.y * l.y, 0.0)), 1e-3);
        vec2 gu = vec2(radius / (2.0 * SPL_PI * cos_el), 0.0);
        vec2 gv = vec2(0.0, radius / SPL_PI);
        sum += textureGrad(u_src, env_dir_to_uv(l), gu, gv).rgb * nl;
        wsum += nl;
    }
    frag = vec4(sum / max(wsum, 1e-8), 1.0);
}
"""

_FS_BRDF = "#version 430\n" + _GLSL_PI + _GLSL_HAMMERSLEY + """
uniform vec2 u_size;
uniform int u_samples;
out vec4 frag;

float vis_smith_ggx_correlated(float nv, float nl, float a) {
    float a2 = a * a;
    float gv = nl * sqrt(nv * nv * (1.0 - a2) + a2);
    float gl = nv * sqrt(nl * nl * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-12);
}

void main() {
    vec2 idx = gl_FragCoord.xy - 0.5;
    float nv = max(idx.x / (u_size.x - 1.0), 1e-4);
    float r = idx.y / (u_size.y - 1.0);
    float a = r * r;
    float a2 = a * a;
    vec3 v = vec3(sqrt(1.0 - nv * nv), 0.0, nv);
    float sa = 0.0;
    float sb = 0.0;
    uint count = uint(u_samples);
    for (uint i = 0u; i < count; ++i) {
        vec2 xi = hammersley(i, count);
        float ct = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
        float st = sqrt(max(1.0 - ct * ct, 0.0));
        float ph = 2.0 * SPL_PI * xi.x;
        vec3 h = vec3(st * cos(ph), st * sin(ph), ct);
        float vh = dot(v, h);
        vec3 l = 2.0 * vh * h - v;
        float nl = l.z;
        if (nl > 0.0 && vh > 0.0) {
            float g = 4.0 * vis_smith_ggx_correlated(nv, nl, a) * nl * vh / max(ct, 1e-8);
            float fc = pow(1.0 - vh, 5.0);
            sa += (1.0 - fc) * g;
            sb += fc * g;
        }
    }
    frag = vec4(sa / float(count), sb / float(count), 0.0, 1.0);
}
"""


# ======================================================================
# 质量设置
# ======================================================================

class EnvironmentSettings(PropertyGroup):
    """环境光贴图的质量。换环境图时按这里生成贴图。"""

    background_max_size = IntProperty(
        "背景贴图最大宽度", default=4096, min=256, max=16384, unit="px",
        description="环境图当背景显示时的最大宽度，更大的图先按面积缩小")
    specular_size = IntProperty(
        "反射贴图宽度", default=1024, min=64, max=4096, unit="px",
        description="最清晰一级反射贴图的宽度，决定光滑表面的反射有多清楚")
    specular_min_size = IntProperty(
        "最模糊一级宽度", default=32, min=8, max=256, unit="px",
        description="粗糙度为 1 的那一级反射贴图的宽度")
    samples = IntProperty(
        "反射采样数", default=512, min=16, max=16384,
        description="最清晰的模糊级每个纹素的采样数，往后每级加倍，越多越平滑")
    samples_max = IntProperty(
        "反射采样数上限", default=4096, min=16, max=65536,
        description="每个纹素最多采样多少次")
    footprint = FloatProperty(
        "采样平滑度", default=2.0, min=1.0, max=8.0, precision=2,
        description="每个采样覆盖范围的放大倍数。越大越不容易出亮斑，过大反射会偏糊")
    sh_size = IntProperty(
        "漫反射积分宽度", default=256, min=16, max=2048, unit="px",
        description="计算漫反射光照前把环境图缩到多宽")
    brdf_size = IntProperty("BRDF 表尺寸", default=128, min=16, max=512, unit="px")
    brdf_samples = IntProperty("BRDF 表采样数", default=2048, min=64, max=16384)
    anisotropy = FloatProperty("背景各向异性过滤", default=16.0, min=1.0, max=16.0, precision=0)
    batch_samples = IntProperty(
        "单批采样上限", default=16_000_000, min=100_000, max=1_000_000_000,
        description="预过滤时每次提交给显卡的采样数上限，越小越不影响别的程序用显卡")


def specular_level_roughness(level: int, levels: int) -> float:
    """预过滤第 level 级对应的粗糙度：r = 1 − sqrt(1 − i / (N − 1))。"""
    if levels <= 1:
        return 0.0
    t = min(max(level / (levels - 1), 0.0), 1.0)
    return 1.0 - math.sqrt(1.0 - t)


def roughness_to_lod(roughness: float, levels: int) -> float:
    """粗糙度 → 预过滤贴图的级（和 GLSL env_roughness_lod 一致）。"""
    r = min(max(float(roughness), 0.0), 1.0)
    return r * (2.0 - r) * max(levels - 1, 0)


def _set_uniform(program: Any, name: str, value: Any) -> bool:
    member = program.get(name, None)
    if member is None:
        return False
    member.value = value
    return True


def _write_uniform(program: Any, name: str, array: np.ndarray) -> bool:
    member = program.get(name, None)
    if member is None:
        return False
    member.write(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    return True


# ======================================================================
# Environment
# ======================================================================

class Environment:
    """一张环境图在显卡上的全部数据：背景贴图、高光预过滤贴图、BRDF 表、漫反射球谐。

    用法::

        env = Environment(ctx)
        env.set_image(ibl.load_hdri("forest"))
        env.rotation = math.radians(view_shading.hdri_rotation)
        env.strength = view_shading.hdri_strength
        env.bind(program, unit_base=4)      # 占用 4、5、6 三个纹理单元

    set_image 会切换当前帧缓冲和视口，完成后恢复调用前的帧缓冲与开关状态。
    """

    TEXTURE_UNITS = 3

    def __init__(self, ctx: Any, settings: EnvironmentSettings | None = None) -> None:
        import moderngl  # noqa: PLC0415

        self._mgl = moderngl
        self.ctx = ctx
        self.settings = settings if settings is not None else EnvironmentSettings()
        self.rotation = 0.0
        self.strength = 1.0
        self.sh = np.zeros((9, 3), dtype=np.float32)
        self.levels = 1
        self.level_roughness: list[float] = [0.0]
        self.bg_texture = None
        self.spec_texture = None
        self.brdf_texture = None
        self.source_size = (0, 0)
        self.stats: dict[str, float] = {}
        self._released = False
        self._prog_prefilter = ctx.program(vertex_shader=_VS_FULLSCREEN, fragment_shader=_FS_PREFILTER)
        self._prog_brdf = ctx.program(vertex_shader=_VS_FULLSCREEN, fragment_shader=_FS_BRDF)
        self._vao_prefilter = ctx.vertex_array(self._prog_prefilter, [])
        self._vao_brdf = ctx.vertex_array(self._prog_brdf, [])
        self._build_brdf()
        self.set_image(np.full((32, 64, 3), 0.5, dtype=np.float32))

    # ---- 公共接口 ----
    def set_hdri(self, name_or_path: str | Path) -> None:
        """按名字或路径读一张环境图并生成贴图。"""
        self.set_image(load_hdri(name_or_path))

    def set_image(self, rgb: np.ndarray) -> None:
        """用线性等距柱状图 (H, W, 3)（行序自上而下）生成背景、预过滤贴图和球谐。"""
        if self._released:
            raise RuntimeError("环境光已释放")
        t0 = time.perf_counter()
        img = sanitize(rgb)
        h, w = img.shape[:2]
        if w < 2 or h < 1:
            raise ValueError(f"环境图太小：{w}×{h}")
        s = self.settings
        self.source_size = (w, h)

        # 漫反射球谐用原始精度的数据
        t1 = time.perf_counter()
        self.sh = sh_project(img, s.sh_size)
        t_sh = time.perf_counter() - t1

        # 背景贴图
        t1 = time.perf_counter()
        bw = min(w, int(s.background_max_size))
        bh = max(1, int(round(bw * h / w)))
        bg_img = area_resize(img, bw, bh) if (bw, bh) != (w, h) else img
        self._release_textures(keep_brdf=True)
        self.bg_texture = self._make_texture(bg_img, mipmaps=True)
        self.bg_texture.repeat_x = True
        self.bg_texture.repeat_y = False
        self.bg_texture.anisotropy = float(min(s.anisotropy, self.ctx.max_anisotropy or 1.0))
        t_bg = time.perf_counter() - t1

        # 高光预过滤
        t1 = time.perf_counter()
        self._prefilter(bw)
        t_spec = time.perf_counter() - t1
        self.stats = {"total_ms": (time.perf_counter() - t0) * 1000.0, "sh_ms": t_sh * 1000.0,
                      "background_ms": t_bg * 1000.0, "prefilter_ms": t_spec * 1000.0}
        log.debug("环境光生成 %s×%s，%.1f ms", w, h, self.stats["total_ms"])

    def bind(self, program: Any, unit_base: int = 0, *, rotation: float | None = None,
             strength: float | None = None) -> int:
        """把贴图绑到 unit_base 起的三个纹理单元并设置 uniform。着色器里没用到的 uniform 跳过。返回下一个空闲单元。"""
        if self._released:
            raise RuntimeError("环境光已释放")
        self.spec_texture.use(location=unit_base)
        self.bg_texture.use(location=unit_base + 1)
        self.brdf_texture.use(location=unit_base + 2)
        _set_uniform(program, "u_env_spec", unit_base)
        _set_uniform(program, "u_env_bg", unit_base + 1)
        _set_uniform(program, "u_env_brdf", unit_base + 2)
        _write_uniform(program, "u_env_sh", self.sh)
        _set_uniform(program, "u_env_rotation", float(self.rotation if rotation is None else rotation))
        _set_uniform(program, "u_env_strength", float(self.strength if strength is None else strength))
        _set_uniform(program, "u_env_levels", int(self.levels))
        return unit_base + self.TEXTURE_UNITS

    def read_specular_level(self, level: int) -> np.ndarray:
        """读回预过滤贴图的某一级，float32 (H, W, 3)，行序自上而下。测试与检查用。"""
        w, h = self.level_size(level)
        data = np.frombuffer(self.spec_texture.read(level=level), dtype=np.float16)
        return np.flipud(data.reshape(h, w, 3)).astype(np.float32)

    def level_size(self, level: int) -> tuple[int, int]:
        w, h = self.spec_texture.size
        return max(1, w >> level), max(1, h >> level)

    def release(self) -> None:
        if self._released:
            return
        self._release_textures(keep_brdf=False)
        for obj in (self._vao_prefilter, self._vao_brdf, self._prog_prefilter, self._prog_brdf):
            try:
                obj.release()
            except Exception:  # noqa: BLE001
                pass
        self._released = True

    def __del__(self) -> None:  # 上下文已失效时静默
        try:
            self.release()
        except Exception:  # noqa: BLE001
            pass

    # ---- 内部 ----
    def _make_texture(self, img: np.ndarray, mipmaps: bool) -> Any:
        h, w = img.shape[:2]
        data = np.ascontiguousarray(np.flipud(np.minimum(img, HALF_MAX)).astype(np.float16))
        tex = self.ctx.texture((w, h), 3, data.tobytes(), dtype="f2")
        if mipmaps:
            tex.build_mipmaps()
            tex.filter = (self._mgl.LINEAR_MIPMAP_LINEAR, self._mgl.LINEAR)
        else:
            tex.filter = (self._mgl.LINEAR, self._mgl.LINEAR)
        return tex

    def _release_textures(self, keep_brdf: bool) -> None:
        names = ["bg_texture", "spec_texture"] + ([] if keep_brdf else ["brdf_texture"])
        for name in names:
            tex = getattr(self, name)
            if tex is not None:
                try:
                    tex.release()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, name, None)

    def _build_brdf(self) -> None:
        s = self.settings
        n = int(s.brdf_size)
        tex = self.ctx.texture((n, n), 2, dtype="f4")
        tex.filter = (self._mgl.LINEAR, self._mgl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        try:
            _set_uniform(self._prog_brdf, "u_size", (float(n), float(n)))
            _set_uniform(self._prog_brdf, "u_samples", int(s.brdf_samples))
            with self.ctx.scope(fbo, enable_only=self._mgl.NOTHING):
                fbo.viewport = (0, 0, n, n)
                self._vao_brdf.render(mode=self._mgl.TRIANGLES, vertices=3)
            self.ctx.finish()
        finally:
            fbo.release()
        self.brdf_texture = tex

    def _prefilter(self, bg_width: int) -> None:
        s = self.settings
        mgl = self._mgl
        target = int(s.specular_size)
        min_w = max(4, int(s.specular_min_size))
        # 第 0 级宽度：取最接近原图宽度的 2 的幂，夹在 [4×最模糊级, 设置宽度] 之间
        base = 1 << max(0, int(round(math.log2(max(bg_width, 1)))))
        base = max(min(base, 1 << int(math.floor(math.log2(max(target, 2))))), min_w * 4)
        levels = int(math.floor(math.log2(base / min_w))) + 1
        self.levels = max(1, levels)
        self.level_roughness = [specular_level_roughness(i, self.levels) for i in range(self.levels)]

        spec = self.ctx.texture((base, base // 2), 3, dtype="f2")
        spec.build_mipmaps(0, self.levels - 1)
        spec.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        spec.repeat_x = True
        spec.repeat_y = False
        self.spec_texture = spec

        prog = self._prog_prefilter
        bw, bh = self.bg_texture.size
        _set_uniform(prog, "u_src", 0)
        _set_uniform(prog, "u_footprint", float(s.footprint))
        self.bg_texture.use(location=0)
        for level in range(self.levels):
            w, h = max(1, base >> level), max(1, (base // 2) >> level)
            rough = self.level_roughness[level]
            samples = 0 if level == 0 else int(min(s.samples_max, s.samples * (2 ** (level - 1))))
            tmp = self.ctx.texture((w, h), 4, dtype="f4")
            fbo = self.ctx.framebuffer(color_attachments=[tmp])
            try:
                _set_uniform(prog, "u_dst_size", (float(w), float(h)))
                _set_uniform(prog, "u_alpha", float(rough * rough))
                _set_uniform(prog, "u_samples", max(1, samples))
                _set_uniform(prog, "u_copy_lod", float(max(math.log2(bw / w), 0.0)) if level == 0 else -1.0)
                rows = h if samples == 0 else max(1, int(s.batch_samples) // max(1, w * samples))
                with self.ctx.scope(fbo, enable_only=mgl.NOTHING):
                    y = 0
                    while y < h:
                        band = min(rows, h - y)
                        fbo.viewport = (0, y, w, band)
                        self._vao_prefilter.render(mode=mgl.TRIANGLES, vertices=3)
                        if band < h:
                            self.ctx.finish()
                        y += band
                data = np.frombuffer(tmp.read(), dtype=np.float32).reshape(h, w, 4)[..., :3]
                data = np.nan_to_num(data, nan=0.0, posinf=HALF_MAX, neginf=0.0)
                spec.write(np.ascontiguousarray(np.minimum(data, HALF_MAX).astype(np.float16)).tobytes(), level=level)
            finally:
                fbo.release()
                tmp.release()


# ======================================================================
# 工作室光与 Matcap 的绑定辅助
# ======================================================================

def bind_studio(program: Any, data: dict, *, roughness: float = 0.4, metallic: float = 0.0,
                specular: bool = True) -> None:
    """把工作室光字典（load_studio_light 的返回值）写进着色器的 u_studio_* uniform。"""
    dirs, cols, specs, smooth, amb = _studio_arrays(data)
    _write_uniform(program, "u_studio_dir", dirs)
    _write_uniform(program, "u_studio_col", cols)
    _write_uniform(program, "u_studio_spec", specs)
    _write_uniform(program, "u_studio_smooth", smooth)
    _set_uniform(program, "u_studio_ambient", tuple(float(x) for x in amb))
    _set_uniform(program, "u_studio_roughness", float(roughness))
    _set_uniform(program, "u_studio_metallic", float(metallic))
    _set_uniform(program, "u_studio_specular", 1 if specular else 0)


def create_matcap_texture(ctx: Any, rgb: np.ndarray) -> Any:
    """把 Matcap 图（float (H, W, 3)，行序自上而下）建成贴图：RGB16F、多级渐远、边缘夹紧。"""
    import moderngl  # noqa: PLC0415

    img = sanitize(rgb)
    h, w = img.shape[:2]
    data = np.ascontiguousarray(np.flipud(np.minimum(img, HALF_MAX)).astype(np.float16))
    tex = ctx.texture((w, h), 3, data.tobytes(), dtype="f2")
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


def bind_matcap(program: Any, texture: Any, unit: int = 0, flip: bool = False) -> int:
    """把 Matcap 贴图绑到纹理单元 unit 并设置 u_matcap / u_matcap_flip。返回下一个空闲单元。"""
    texture.use(location=unit)
    _set_uniform(program, "u_matcap", unit)
    _set_uniform(program, "u_matcap_flip", 1 if flip else 0)
    return unit + 1


__all__ = [
    "Environment", "EnvironmentSettings", "GLSL_ENV", "GLSL_PBR", "GLSL_TONEMAP", "GLSL_STUDIO", "GLSL_MATCAP",
    "TRANSFORM_STANDARD", "TRANSFORM_FILMIC", "TRANSFORM_AGX", "TRANSFORMS",
    "list_hdris", "load_hdri", "thumbnail", "hdri_path", "list_matcaps", "load_matcap", "matcap_thumbnail",
    "list_studio_lights", "load_studio_light", "studio_thumbnail", "parse_studio_sl", "label",
    "load_image", "read_exr", "read_hdr", "read_npz", "sanitize", "area_resize",
    "direction_to_uv", "uv_to_direction", "pixel_to_direction", "direction_to_pixel", "equirect_directions",
    "equirect_solid_angles", "rotate_y", "brightest_direction",
    "sh_basis", "sh_project", "sh_evaluate", "tonemap", "srgb_encode", "srgb_decode", "to_uint8",
    "shade_studio_np", "sphere_normals", "make_hdri_thumbnail", "make_matcap_thumbnail", "make_studio_thumbnail",
    "bind_studio", "create_matcap_texture", "bind_matcap", "specular_level_roughness", "roughness_to_lod",
]
