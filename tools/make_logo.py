"""生成 SPLENDER 的程序标志和启动画面。

用法：python tools/make_logo.py [--source 图片] [--vector]

标志默认取 splender/resources/logo/source.png（LibTV 生成的晶体 S，2048² 透明底，原图和其他方案在
docs/design/logo/）：各尺寸按预乘透明度缩小，32 像素及以下再锐化、略提饱和度；48 像素及以上留一点边。
--vector 改用下面的程序画法（一笔写成的 S），没有 source.png 时也用它。

输出到 splender/resources/logo/：
  splender.ico          程序图标，含 16 20 24 32 40 48 64 128 256 各尺寸
  splender_256.png      标志 256 像素
  splender_64.png       标志 64 像素
  splender_32.png       标志 32 像素
  splash.png            启动画面 720×400
  splash@2x.png         启动画面高分屏版 1440×800

标志是深色圆角方块上一笔写成的 S。S 的中心线是几段三次贝塞尔曲线，沿线用一支
扁平的椭圆笔尖一路盖章，笔尖朝向固定，所以横向走笔细、竖向走笔粗，起笔和收笔
有压力变化；颜色沿笔画从青到蓝到紫。所有图都在高倍分辨率下画好，再按面积平均
缩小，保证边缘平滑。16~32 像素单独加粗、放大字形、减小留边，保证任务栏上看得清。

可以反复运行，结果只由本文件里的参数决定。
"""
from __future__ import annotations

import argparse
import io
import math
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "splender" / "resources" / "logo"
FONT_DIR = Path("C:/Windows/Fonts")

ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
PNG_SIZES = (256, 64, 32)

# ---------------------------------------------------------------- 颜色

#: 笔画渐变：(沿笔画位置 0~1, 颜色)。在 OKLab 空间插值，中间色不发灰。
STROKE_GRADIENT = (
    (0.00, "#25d8ef"),
    (0.50, "#4a80f6"),
    (1.00, "#a45cf4"),
)
#: 方块底色：上、下
TILE_TOP = "#262833"
TILE_BOTTOM = "#16171d"
#: 方块内描边（白色，透明度），让深色方块在深色任务栏上也有轮廓
TILE_RIM_ALPHA = 0.07

#: 启动画面
SPLASH_SIZE = (720, 400)
SPLASH_BG_TOP = "#18191f"
SPLASH_BG_BOTTOM = "#121318"
SPLASH_TITLE = "#ececf0"
SPLASH_SUBTITLE = "#9a9ba5"
SPLASH_RULE = "#25262d"
#: 底部状态条的高度（1 倍尺寸下的像素）。程序在这一条里画加载文字和版本号。
SPLASH_STATUS_HEIGHT = 48

# ---------------------------------------------------------------- 字形

#: S 的中心线，设计空间 100×100，y 向下。每段是 (起点, 控制点1, 控制点2, 终点)，
#: 相邻段在接点处切线共线，所以整条线是光滑的一笔。
S_SEGMENTS = (
    ((77.0, 23.0), (73.0, 13.5), (63.5, 9.0), (50.0, 9.0)),
    ((50.0, 9.0), (35.0, 9.0), (25.5, 18.0), (25.5, 29.0)),
    ((25.5, 29.0), (25.5, 41.0), (36.0, 45.5), (50.0, 50.0)),
    ((50.0, 50.0), (64.0, 54.5), (75.5, 60.0), (75.5, 71.0)),
    ((75.5, 71.0), (75.5, 83.5), (64.5, 91.0), (49.5, 91.0)),
    ((49.5, 91.0), (36.0, 91.0), (26.0, 85.0), (22.5, 77.5)),
)


@dataclass(frozen=True)
class MarkStyle:
    """一个尺寸下标志的画法。"""

    weight: float = 17.5        # 笔尖长轴，设计单位（字高约 100）
    nib_ratio: float = 0.62     # 笔尖短轴 / 长轴，越小粗细对比越强
    nib_angle: float = -40.0    # 笔尖长轴方向，度，屏幕坐标（负数=向右上）
    slant: float = 9.0          # 字形前倾角度，度
    start_pressure: float = 0.65  # 起笔时的压力
    start_len: float = 0.14       # 起笔加压所占笔画比例
    end_pressure: float = 0.15    # 收笔尖端的压力
    end_len: float = 0.32         # 收笔减压所占笔画比例
    glyph_height: float = 0.66    # 字形外框高度 / 方块边长
    optical_dx: float = 0.0       # 字形相对方块中心的视觉修正，占边长比例
    optical_dy: float = 0.0
    margin: float = 0.035         # 方块到图边的留白，占边长比例
    squircle_n: float = 4.6       # 超椭圆指数，越大越接近直角方块
    rim: float = 1.0              # 内描边宽度（输出像素）
    supersample: int = 8


#: 大尺寸的画法
LARGE = MarkStyle()
#: 小尺寸单独调整：笔画更粗、字形更大、压力变化更平缓、方块几乎铺满
SMALL_STYLES = {
    16: replace(LARGE, weight=21.0, nib_ratio=0.70, start_pressure=1.0, end_pressure=0.55, end_len=0.16,
                glyph_height=0.80, margin=0.0, squircle_n=4.0, rim=0.0, supersample=16),
    20: replace(LARGE, weight=20.5, nib_ratio=0.68, start_pressure=1.0, end_pressure=0.48, end_len=0.18,
                glyph_height=0.78, margin=0.0, squircle_n=4.0, rim=0.0, supersample=16),
    24: replace(LARGE, weight=20.0, nib_ratio=0.66, start_pressure=1.0, end_pressure=0.42, end_len=0.20,
                glyph_height=0.77, margin=0.0, squircle_n=4.2, rim=0.0, supersample=16),
    32: replace(LARGE, weight=19.0, nib_ratio=0.62, start_pressure=0.95, end_pressure=0.32, end_len=0.24,
                glyph_height=0.73, margin=0.02, squircle_n=4.4, rim=0.0, supersample=16),
    40: replace(LARGE, weight=18.0, nib_ratio=0.60, end_pressure=0.24, end_len=0.26, glyph_height=0.70,
                margin=0.025, rim=0.0, supersample=12),
    48: replace(LARGE, weight=17.5, nib_ratio=0.60, end_pressure=0.20, glyph_height=0.68, margin=0.03,
                rim=0.0, supersample=12),
    64: replace(LARGE, glyph_height=0.67, supersample=12),
}


def style_for(size: int) -> MarkStyle:
    return SMALL_STYLES.get(size, LARGE)


# ---------------------------------------------------------------- 颜色工具

def hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float64)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lms = np.cbrt(rgb @ _M1.T)
    return lms @ _M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lms = lab @ np.linalg.inv(_M2).T
    return (lms ** 3) @ np.linalg.inv(_M1).T


def gradient_linear(ts: np.ndarray, stops=STROKE_GRADIENT) -> np.ndarray:
    """沿笔画的颜色（线性光 RGB），在 OKLab 里分段插值。"""
    pos = np.array([p for p, _ in stops])
    labs = np.array([linear_to_oklab(srgb_to_linear(hex_rgb(c))) for _, c in stops])
    out = np.empty((len(ts), 3))
    for k in range(3):
        out[:, k] = np.interp(ts, pos, labs[:, k])
    return np.clip(oklab_to_linear(out), 0.0, 1.0)


# ---------------------------------------------------------------- 字形几何

def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def centerline(count: int, slant: float = 0.0) -> np.ndarray:
    """按弧长均匀取 count 个点，返回 (count, 2)，设计单位。slant 是前倾角度（度）。"""
    dense = []
    u = np.linspace(0.0, 1.0, 600)[:, None]
    for p0, p1, p2, p3 in S_SEGMENTS:
        p0, p1, p2, p3 = (np.array(p, dtype=np.float64) for p in (p0, p1, p2, p3))
        mu = 1.0 - u
        pts = mu ** 3 * p0 + 3 * mu * mu * u * p1 + 3 * mu * u * u * p2 + u ** 3 * p3
        dense.append(pts if not dense else pts[1:])
    dense = np.concatenate(dense)
    if slant:
        dense[:, 0] += (50.0 - dense[:, 1]) * math.tan(math.radians(slant))
    seg = np.hypot(*np.diff(dense, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, cum[-1], count)
    return np.stack([np.interp(target, cum, dense[:, 0]), np.interp(target, cum, dense[:, 1])], axis=1)


def pressure(ts: np.ndarray, style: MarkStyle) -> np.ndarray:
    rise = style.start_pressure + (1.0 - style.start_pressure) * _smoothstep(0.0, style.start_len, ts)
    fall = 1.0 - (1.0 - style.end_pressure) * _smoothstep(1.0 - style.end_len, 1.0, ts)
    return rise * fall


def nib_axes(style: MarkStyle) -> tuple[float, float, float]:
    """笔尖半长轴、半短轴（设计单位）和朝向（弧度）。"""
    a = style.weight / 2.0
    return a, a * style.nib_ratio, math.radians(style.nib_angle)


def glyph_bounds(style: MarkStyle, count: int = 4000) -> tuple[float, float, float, float]:
    """整笔（含笔尖宽度）的外框，设计单位。"""
    pts = centerline(count, style.slant)
    ts = np.linspace(0.0, 1.0, count)
    p = pressure(ts, style)
    a, b, ang = nib_axes(style)
    hx = np.sqrt((a * math.cos(ang)) ** 2 + (b * math.sin(ang)) ** 2) * p
    hy = np.sqrt((a * math.sin(ang)) ** 2 + (b * math.cos(ang)) ** 2) * p
    return (float(np.min(pts[:, 0] - hx)), float(np.min(pts[:, 1] - hy)),
            float(np.max(pts[:, 0] + hx)), float(np.max(pts[:, 1] + hy)))


def stroke_index_map(width: int, height: int, transform, style: MarkStyle, scale_px: float) -> np.ndarray:
    """沿中心线盖椭圆笔尖章，返回每个像素被哪一个章最后盖到（0 表示没盖到）。

    transform 把设计坐标变成像素坐标；scale_px 是一个设计单位等于多少像素。
    """
    pts0 = centerline(4000, style.slant)
    length_px = float(np.sum(np.hypot(*np.diff(pts0, axis=0).T))) * scale_px
    count = max(200, int(length_px / 0.5))
    pts = centerline(count, style.slant)
    ts = np.linspace(0.0, 1.0, count)
    p = pressure(ts, style)
    a, b, ang = nib_axes(style)
    k = 48
    u = np.linspace(0.0, 2 * math.pi, k, endpoint=False)
    ex = np.cos(u) * a * math.cos(ang) - np.sin(u) * b * math.sin(ang)
    ey = np.cos(u) * a * math.sin(ang) + np.sin(u) * b * math.cos(ang)
    image = Image.new("I", (width, height), 0)
    draw = ImageDraw.Draw(image)
    centers = transform(pts)
    for i in range(count):
        cx, cy = centers[i]
        s = p[i] * scale_px
        poly = list(zip((cx + ex * s).tolist(), (cy + ey * s).tolist()))
        draw.polygon(poly, fill=i + 1)
    return np.asarray(image, dtype=np.int64), count


# ---------------------------------------------------------------- 栅格工具

def box_reduce(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape[:2]
    shape = (h // factor, factor, w // factor, factor) + arr.shape[2:]
    return arr.reshape(shape).mean(axis=(1, 3))


def superellipse_mask(size: int, half: float, n: float) -> np.ndarray:
    """以图中心为中心、半边长 half 的超椭圆，返回布尔掩码。"""
    c = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    return (np.abs((x - c) / half) ** n + np.abs((y - c) / half) ** n) <= 1.0


def to_image(premul_linear: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """预乘的线性光颜色 + 覆盖率 → 8 位 sRGB RGBA。"""
    safe = np.where(alpha > 1e-6, alpha, 1.0)[..., None]
    rgb = linear_to_srgb(premul_linear / safe)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    rgba[alpha <= 1e-6] = 0.0
    return Image.fromarray(np.round(rgba * 255.0).astype(np.uint8), "RGBA")


# ---------------------------------------------------------------- 标志

def render_layers(size: int, style: MarkStyle, tile: bool = True, glyph_height_px: float | None = None):
    """在 size×size（输出像素）上画标志，返回高倍分辨率下的 (预乘线性颜色, 覆盖率)。

    tile=False 时只画笔画、背景透明（启动画面用）。glyph_height_px 指定字形高度时忽略 style.glyph_height。
    """
    ss = style.supersample
    big = size * ss
    x0, y0, x1, y1 = glyph_bounds(style)
    gh = glyph_height_px if glyph_height_px is not None else size * style.glyph_height
    scale_px = gh * ss / (y1 - y0)
    cx_design = (x0 + x1) / 2.0
    cy_design = (y0 + y1) / 2.0
    center_px = big / 2.0
    dx = style.optical_dx * big
    dy = style.optical_dy * big

    def transform(pts: np.ndarray) -> np.ndarray:
        return np.stack([(pts[:, 0] - cx_design) * scale_px + center_px + dx,
                         (pts[:, 1] - cy_design) * scale_px + center_px + dy], axis=1)

    index, count = stroke_index_map(big, big, transform, style, scale_px)
    stroke = index > 0
    colors = gradient_linear(np.linspace(0.0, 1.0, count))
    color = np.zeros((big, big, 3))
    color[stroke] = colors[index[stroke] - 1]

    if not tile:
        return color * stroke[..., None], stroke.astype(np.float64)

    half = big * (0.5 - style.margin)
    inside = superellipse_mask(big, half, style.squircle_n)
    rows = np.linspace(0.0, 1.0, big)[:, None]
    top = srgb_to_linear(hex_rgb(TILE_TOP))
    bottom = srgb_to_linear(hex_rgb(TILE_BOTTOM))
    bg = (top[None, None, :] * (1 - rows[..., None]) + bottom[None, None, :] * rows[..., None])
    bg = np.broadcast_to(bg, (big, big, 3)).copy()
    if style.rim > 0 and TILE_RIM_ALPHA > 0:
        inner = superellipse_mask(big, half - style.rim * ss, style.squircle_n)
        ring = inside & ~inner
        bg[ring] = bg[ring] * (1 - TILE_RIM_ALPHA) + TILE_RIM_ALPHA
    out = np.where(stroke[..., None], color, bg)
    alpha = inside.astype(np.float64)
    return out * alpha[..., None], alpha


def render_mark(size: int, style: MarkStyle | None = None, tile: bool = True,
                glyph_height_px: float | None = None) -> Image.Image:
    style = style or style_for(size)
    premul, alpha = render_layers(size, style, tile, glyph_height_px)
    ss = style.supersample
    return to_image(box_reduce(premul, ss), box_reduce(alpha, ss))


# ---------------------------------------------------------------- ICO

def _bmp_entry(image: Image.Image) -> bytes:
    """32 位带透明度的 BMP 图标数据（含 AND 掩码，兼容老程序）。"""
    w, h = image.size
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    bgra = rgba[::-1, :, [2, 1, 0, 3]].tobytes()
    row_bytes = ((w + 31) // 32) * 4
    mask = bytearray(row_bytes * h)
    transparent = rgba[::-1, :, 3] == 0
    for y in range(h):
        for x in np.nonzero(transparent[y])[0]:
            mask[y * row_bytes + x // 8] |= 0x80 >> (x % 8)
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0, len(bgra) + len(mask), 0, 0, 0, 0)
    return header + bgra + bytes(mask)


def _png_entry(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def write_ico(path: Path, images: dict[int, Image.Image]) -> None:
    sizes = sorted(images)
    blobs = [(_png_entry if s >= 256 else _bmp_entry)(images[s]) for s in sizes]
    offset = 6 + 16 * len(sizes)
    out = bytearray(struct.pack("<HHH", 0, 1, len(sizes)))
    for s, blob in zip(sizes, blobs):
        dim = 0 if s >= 256 else s
        out += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    for blob in blobs:
        out += blob
    path.write_bytes(bytes(out))


# ---------------------------------------------------------------- 启动画面

def _font(name: str, size: float, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size=round(size), index=index)


def _draw_tracked(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, tracking: float) -> None:
    """逐字画，字距额外加 tracking 像素。"""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + tracking


def _tracked_width(text: str, font, tracking: float) -> float:
    return sum(font.getlength(ch) for ch in text) + tracking * (len(text) - 1)


def _vertical_gradient(width: int, height: int, top: str, bottom: str, seed: int = 7) -> Image.Image:
    """带轻微抖动的竖向渐变，避免深色渐变出现色带。"""
    rows = np.linspace(0.0, 1.0, height)[:, None, None]
    a = srgb_to_linear(hex_rgb(top))
    b = srgb_to_linear(hex_rgb(bottom))
    lin = a * (1 - rows) + b * rows
    srgb = linear_to_srgb(np.broadcast_to(lin, (height, width, 3))) * 255.0
    noise = np.random.default_rng(seed).uniform(-0.5, 0.5, size=srgb.shape)
    return Image.fromarray(np.clip(np.round(srgb + noise), 0, 255).astype(np.uint8), "RGB")


#: 启动画面上的字
SPLASH_TITLE_TEXT = "SPLENDER"
SPLASH_SUBTITLE_TEXT = "雕刻 · 绘制 · 材质 · 渲染"
#: 字体：(文件名, 字号, ttc 序号, 额外字距)，字号和字距按 1 倍尺寸
SPLASH_TITLE_FONT = ("segoeuisl.ttf", 50, 0, 9.0)       # Segoe UI Semilight
SPLASH_SUBTITLE_FONT = ("msyh.ttc", 16, 1, 2.0)         # Microsoft YaHei UI
SPLASH_MARK_SIZE = 128          # 标志边长
SPLASH_MARK_GAP = 30            # 标志和文字之间
SPLASH_TEXT_GAP = 14            # 标题和副标题之间
SPLASH_BORDER = "#2a2b32"       # 外框细线，深色桌面上也看得出边界


SOURCE = OUT_DIR / "source.png"
#: 用图片做标志时，各尺寸到图边的留白（占边长比例）；小尺寸铺满，看得清
SOURCE_MARGIN = {16: 0.0, 20: 0.0, 24: 0.0, 32: 0.0, 40: 0.02, 48: 0.03}
SOURCE_MARGIN_LARGE = 0.035


def load_source(path: Path) -> Image.Image:
    """读入标志原图：裁到不透明的范围，补成正方形。"""
    image = Image.open(path).convert("RGBA")
    alpha = np.asarray(image)[:, :, 3]
    ys, xs = np.nonzero(alpha > 8)
    crop = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    side = max(crop.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square


def mark_from_source(source: Image.Image, size: int) -> Image.Image:
    """按预乘透明度缩小（边缘不发黑），小尺寸锐化、略提饱和度。"""
    from PIL import ImageEnhance, ImageFilter

    margin = SOURCE_MARGIN.get(size, SOURCE_MARGIN_LARGE)
    inner = max(1, round(size * (1.0 - 2.0 * margin)))
    image = source.convert("RGBa").resize((inner, inner), Image.LANCZOS).convert("RGBA")
    if size <= 32:
        alpha = image.split()[3]
        rgb = image.convert("RGB").filter(ImageFilter.UnsharpMask(radius=0.7, percent=80, threshold=0))
        rgb = ImageEnhance.Color(rgb).enhance(1.12)
        image = Image.merge("RGBA", (*rgb.split(), alpha))
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(image, ((size - inner) // 2, (size - inner) // 2))
    return canvas


_SOURCE_IMAGE: Image.Image | None = None


def mark_image(size: int) -> Image.Image:
    """某个尺寸的标志：有原图就用原图，否则用程序画法。"""
    if _SOURCE_IMAGE is not None:
        return mark_from_source(_SOURCE_IMAGE, size)
    return render_mark(size, replace(style_for(size), supersample=max(8, style_for(size).supersample)))


def make_splash(scale: int = 1) -> Image.Image:
    """启动画面。标志和文字作为一组在上方区域居中；底部 SPLASH_STATUS_HEIGHT 高的一条留给程序画字。"""
    w, h = SPLASH_SIZE[0] * scale, SPLASH_SIZE[1] * scale
    s = float(scale)
    canvas = _vertical_gradient(w, h, SPLASH_BG_TOP, SPLASH_BG_BOTTOM).convert("RGBA")
    status_top = h - round(SPLASH_STATUS_HEIGHT * s)
    body_mid = status_top / 2.0
    draw = ImageDraw.Draw(canvas)

    t_name, t_size, t_index, t_track = SPLASH_TITLE_FONT
    s_name, s_size, s_index, s_track = SPLASH_SUBTITLE_FONT
    title_font = _font(t_name, t_size * s, t_index)
    sub_font = _font(s_name, s_size * s, s_index)
    t_box = title_font.getbbox(SPLASH_TITLE_TEXT)
    s_box = sub_font.getbbox(SPLASH_SUBTITLE_TEXT)
    title_ink_h = t_box[3] - t_box[1]
    sub_ink_h = s_box[3] - s_box[1]
    text_w = max(_tracked_width(SPLASH_TITLE_TEXT, title_font, t_track * s),
                 _tracked_width(SPLASH_SUBTITLE_TEXT, sub_font, s_track * s))

    mark_size = round(SPLASH_MARK_SIZE * s)
    gap_mark = SPLASH_MARK_GAP * s
    left = round((w - (mark_size + gap_mark + text_w)) / 2.0)
    if _SOURCE_IMAGE is not None:
        mark = mark_from_source(_SOURCE_IMAGE, mark_size)
    else:
        mark = render_mark(mark_size, replace(LARGE, supersample=8))
    canvas.alpha_composite(mark, (left, round(body_mid - mark_size / 2.0)))

    # 以字的实际墨迹高度对齐，标题加副标题这一块的中线对准标志中线
    gap_text = SPLASH_TEXT_GAP * s
    title_top = body_mid - (title_ink_h + gap_text + sub_ink_h) / 2.0
    text_x = left + mark_size + gap_mark
    _draw_tracked(draw, (text_x, title_top - t_box[1]), SPLASH_TITLE_TEXT, title_font, SPLASH_TITLE, t_track * s)
    sub_top = title_top + title_ink_h + gap_text
    _draw_tracked(draw, (text_x + s, sub_top - s_box[1]), SPLASH_SUBTITLE_TEXT, sub_font, SPLASH_SUBTITLE,
                  s_track * s)

    # 底部状态条只画一条分隔线；加载文字和版本号由程序运行时画
    line = max(1, round(s))
    draw.rectangle([0, status_top, w - 1, status_top + line - 1], fill=SPLASH_RULE)
    draw.rectangle([0, 0, w - 1, h - 1], outline=SPLASH_BORDER, width=line)
    return canvas.convert("RGB")


# ---------------------------------------------------------------- 入口

def build(out_dir: Path = OUT_DIR) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = {size: mark_image(size) for size in ICO_SIZES}
    written = {}
    ico = out_dir / "splender.ico"
    write_ico(ico, images)
    written["ico"] = ico
    for size in PNG_SIZES:
        path = out_dir / f"splender_{size}.png"
        images[size].save(path, "PNG", optimize=True)
        written[f"png{size}"] = path
    splash = out_dir / "splash.png"
    make_splash(1).save(splash, "PNG", optimize=True)
    written["splash"] = splash
    splash2 = out_dir / "splash@2x.png"
    make_splash(2).save(splash2, "PNG", optimize=True)
    written["splash2x"] = splash2
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 SPLENDER 标志和启动画面")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="输出目录")
    parser.add_argument("--source", type=Path, default=SOURCE, help="标志原图（透明底的正方形大图）")
    parser.add_argument("--vector", action="store_true", help="不用原图，用程序画的 S")
    args = parser.parse_args()
    global _SOURCE_IMAGE
    if not args.vector and args.source.is_file():
        _SOURCE_IMAGE = load_source(args.source)
    for key, path in build(args.out).items():
        print(f"{key:9s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
