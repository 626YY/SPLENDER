"""调整层：参数打包、查找表，以及和合成着色器一样算法的 CPU 实现（生成查找表、测试、预览用）。

合成着色器里每层四组数（adj、adj2、adj3、adj4；adj3 的第四个数是调整种类）加一张查找表，图层参数 flags2 的
第三个数是它从查找表缓冲的第几段（每段 LUT_SIZE 行 RGBA）开始：
- 第一段是「按通道」的函数：RGB 三列给颜色的三个通道用（亮度对比度、色阶、曲线、曝光度、反相、色调分离），
  第四列给金属度、粗糙度、高度用（每种调整都有；只对颜色有意义的几种，取把这个数当成灰色改完后的明暗）；
- 后面接着这种调整自己的数据：渐变的色带、六类颜色的色相饱和度、可选颜色的九类参数、颜色查找的表。
颜色都在 sRGB 编码下调（和常见修图软件一致），曝光度在线性光下算。
"""
from __future__ import annotations

import numpy as np

from ..core import curve, ramp
from ..doc.project import ADJUST_CODE, HUE_RANGES, SELECTIVE_INKS, SELECTIVE_RANGES, level_prop_names

LUT_SIZE = 1024
LUMA = np.array([0.2126, 0.7152, 0.0722])
#: 颜色三个通道各改各的几种（全靠查找表第一段）
PER_CHANNEL = ("BRIGHTNESS", "LEVELS", "CURVES", "EXPOSURE", "INVERT", "POSTERIZE")

#: 每种调整用到的属性（预设里的「默认值」把它们恢复成默认）
TYPE_PROPS = {
    "BRIGHTNESS": ("adj_brightness", "adj_contrast"),
    "LEVELS": tuple(name for channel in ("RGB", "R", "G", "B") for name in level_prop_names(channel)),
    "CURVES": ("adj_curve", "adj_curve_r", "adj_curve_g", "adj_curve_b"),
    "EXPOSURE": ("adj_exposure", "adj_exp_offset", "adj_exp_gamma"),
    "VIBRANCE": ("adj_vibrance", "adj_vib_saturation"),
    "HSV": ("adj_hue", "adj_saturation", "adj_value", "adj_colorize", "adj_colorize_hue", "adj_colorize_sat")
           + tuple("adj_hsv_%s_%s" % (key.lower(), field) for key, _l, _c in HUE_RANGES
                   for field in ("hue", "sat", "light")),
    "COLOR_BALANCE": tuple("adj_cb_%s_%s" % (tone, ch) for tone in ("shadows", "midtones", "highlights")
                           for ch in "rgb") + ("adj_cb_preserve",),
    "BLACK_WHITE": ("adj_bw_red", "adj_bw_yellow", "adj_bw_green", "adj_bw_cyan", "adj_bw_blue", "adj_bw_magenta",
                    "adj_bw_tint", "adj_bw_tint_color"),
    "PHOTO_FILTER": ("adj_filter_color", "adj_filter_density", "adj_filter_preserve"),
    "CHANNEL_MIXER": tuple("adj_mix_%s%s" % (out, src) for out in "rgb" for src in "rgbc") + ("adj_mix_mono",),
    "INVERT": (),
    "POSTERIZE": ("adj_posterize",),
    "THRESHOLD": ("adj_threshold",),
    "GRADIENT_MAP": ("adj_ramp", "adj_ramp_reverse"),
    "COLOR_LOOKUP": ("adj_lut_look", "adj_lut_name", "adj_lut_data"),
    "SELECTIVE_COLOR": tuple("adj_sel_%s_%s" % (key.lower(), ink) for key, _l, _d in SELECTIVE_RANGES
                             for ink, _il, _id in SELECTIVE_INKS) + ("adj_sel_method",),
}


# ====================================================================== 基础（和着色器里的一样）
def luma(c) -> np.ndarray:
    return np.asarray(c, np.float64) @ LUMA


def srgb_to_linear(c) -> np.ndarray:
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0.0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c) -> np.ndarray:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def clip_color(c) -> np.ndarray:
    """超出 0..1 的颜色往同样明暗的灰收（色相不变），和「明度」混合模式一样。c：(n, 3)。"""
    c = np.asarray(c, np.float64)
    lum = luma(c)[:, None]
    lo = c.min(axis=1, keepdims=True)
    hi = c.max(axis=1, keepdims=True)
    c = np.where(lo < 0.0, lum + (c - lum) * (lum / np.maximum(lum - lo, 1e-6)), c)
    c = np.where(hi > 1.0, lum + (c - lum) * ((1.0 - lum) / np.maximum(hi - lum, 1e-6)), c)
    return np.clip(c, 0.0, 1.0)


def set_lum(c, lum) -> np.ndarray:
    c = np.asarray(c, np.float64)
    return clip_color(c + (np.asarray(lum, np.float64) - luma(c))[:, None])


def lift(v, amount) -> np.ndarray:
    """正数往 1 提、负数往 0 压。"""
    v = np.asarray(v, np.float64)
    return np.where(np.asarray(amount) >= 0.0, v + (1.0 - v) * amount, v * (1.0 + amount))


def rgb_to_hsv(c) -> tuple:
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    hi = c.max(axis=1)
    lo = c.min(axis=1)
    d = hi - lo
    safe = np.maximum(d, 1e-12)
    h = np.where(hi == r, np.mod((g - b) / safe, 6.0), np.where(hi == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    h = np.where(d > 0.0, h / 6.0, 0.0)
    s = np.where(hi > 0.0, d / np.maximum(hi, 1e-12), 0.0)
    return h, s, hi


def hsv_to_rgb(h, s, v) -> np.ndarray:
    k = np.array([0.0, 2.0 / 3.0, 1.0 / 3.0])
    q = np.clip(np.abs(np.mod(np.asarray(h)[:, None] + k, 1.0) * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    return np.asarray(v)[:, None] * (1.0 + (q - 1.0) * np.asarray(s)[:, None])


def levels(x, in_black: float, in_white: float, gamma: float, out_black: float, out_white: float) -> np.ndarray:
    """色阶：输入黑白场拉开、灰度系数弯中间调、再压到输出黑白场之间。"""
    t = np.clip((np.asarray(x, np.float64) - in_black) / max(in_white - in_black, 1e-4), 0.0, 1.0)
    t = t ** (1.0 / max(gamma, 1e-3))
    return out_black + (out_white - out_black) * t


def _brightness(x, brightness: float, contrast: float) -> np.ndarray:
    factor = 1.0 / max(1.0 - contrast * 0.99, 0.01) if contrast >= 0.0 else 1.0 + contrast
    return np.clip((np.asarray(x, np.float64) + brightness - 0.5) * factor + 0.5, 0.0, 1.0)


def _posterize(x, count: int) -> np.ndarray:
    n = float(max(int(count), 2))
    return np.minimum(np.floor(np.clip(np.asarray(x, np.float64), 0.0, 1.0) * n), n - 1.0) / (n - 1.0)


def _exposure(linear, exposure: float, offset: float, gamma: float) -> np.ndarray:
    value = np.maximum(np.asarray(linear, np.float64) * 2.0 ** exposure + offset, 0.0)
    return np.clip(value ** (1.0 / max(gamma, 1e-3)), 0.0, 1.0)


# ====================================================================== 按通道的几种
def per_channel(layer, xs) -> tuple[np.ndarray, np.ndarray]:
    """按通道改的几种：返回 (颜色三个通道各自的函数值 (n, 3), 数值贴图的函数值 (n,))，xs 是输入。"""
    xs = np.atleast_1d(np.asarray(xs, np.float64))
    kind = layer.adjust_type
    if kind == "BRIGHTNESS":
        y = _brightness(xs, layer.adj_brightness, layer.adj_contrast)
        return np.stack([y, y, y], axis=1), y
    if kind == "LEVELS":
        master = [getattr(layer, name) for name in level_prop_names("RGB")]
        rgb = np.stack([levels(levels(xs, *[getattr(layer, n) for n in level_prop_names(ch)]), *master)
                        for ch in "RGB"], axis=1)
        return rgb, levels(xs, *master)
    if kind == "CURVES":
        master = layer.adj_curve
        rgb = np.stack([curve.evaluate(master, curve.evaluate(getattr(layer, name), xs))
                        for name in ("adj_curve_r", "adj_curve_g", "adj_curve_b")], axis=1)
        return rgb, curve.evaluate(master, xs)
    if kind == "EXPOSURE":
        args = (layer.adj_exposure, layer.adj_exp_offset, layer.adj_exp_gamma)
        y = linear_to_srgb(_exposure(srgb_to_linear(xs), *args))
        return np.stack([y, y, y], axis=1), _exposure(xs, *args)
    if kind == "INVERT":
        y = 1.0 - xs
        return np.stack([y, y, y], axis=1), y
    if kind == "POSTERIZE":
        y = _posterize(xs, layer.adj_posterize)
        return np.stack([y, y, y], axis=1), y
    return np.stack([xs, xs, xs], axis=1), xs


# ====================================================================== 只对颜色有意义的几种
def _cb_map(v, lum, sh, mi, hi) -> np.ndarray:
    sh = sh * np.clip((lum - 0.333) / -0.25 + 0.5, 0.0, 1.0) * 0.7
    mi = mi * np.clip((lum - 0.333) / 0.25 + 0.5, 0.0, 1.0) * np.clip((lum + 0.333 - 1.0) / -0.25 + 0.5, 0.0, 1.0) * 0.7
    hi = hi * np.clip((lum + 0.333 - 1.0) / 0.25 + 0.5, 0.0, 1.0) * 0.7
    return np.clip(v + sh + mi + hi, 0.0, 1.0)


def _bw_gray(c, primary, secondary) -> np.ndarray:
    """primary：红、绿、蓝；secondary：青、洋红、黄。"""
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    lo = c.min(axis=1)
    hi = c.max(axis=1)
    md = c.sum(axis=1) - lo - hi
    pr, pg, pb = primary
    sc, sm, sy = secondary
    red_max = (r >= g) & (r >= b)
    green_max = ~red_max & (g >= b)
    wp = np.where(red_max, pr, np.where(green_max, pg, pb))
    ws = np.where(red_max, np.where(g >= b, sy, sm),
                  np.where(green_max, np.where(r >= b, sy, sc), np.where(r >= g, sm, sc)))
    return np.clip(lo + (md - lo) * ws + (hi - md) * wp, 0.0, 1.0)


def _selective(c, table, relative: bool) -> np.ndarray:
    """可选颜色：九类颜色（红黄绿青蓝洋红白中性黑）各占多少，按各自的 (青, 洋红, 黄, 黑) 改，改动按份额加起来。"""
    c = np.clip(np.asarray(c, np.float64), 0.0, 1.0)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    lo = c.min(axis=1)
    hi = c.max(axis=1)
    md = c.sum(axis=1) - lo - hi
    weights = [np.where(r >= hi, hi - md, 0.0), np.where(b <= lo, md - lo, 0.0), np.where(g >= hi, hi - md, 0.0),
               np.where(r <= lo, md - lo, 0.0), np.where(b >= hi, hi - md, 0.0), np.where(g <= lo, md - lo, 0.0),
               np.maximum(lo - 0.5, 0.0) * 2.0, np.maximum(1.0 - (np.abs(hi - 0.5) + np.abs(lo - 0.5)), 0.0),
               np.maximum(0.5 - hi, 0.0) * 2.0]
    k0 = (1.0 - hi)[:, None]
    cmy0 = np.where(k0 < 0.9999, (1.0 - c - k0) / np.maximum(1.0 - k0, 1e-6), 0.0)
    delta = np.zeros_like(c)
    for k, w in enumerate(weights):
        a = np.asarray(table[k], np.float64)
        cmy = np.clip(cmy0 * (1.0 + a[:3]) if relative else cmy0 + a[:3], 0.0, 1.0)
        kk = np.clip(k0 * (1.0 + a[3]) if relative else k0 + a[3], 0.0, 1.0)
        delta += ((1.0 - cmy) * (1.0 - kk) - c) * w[:, None]
    return np.clip(c + delta, 0.0, 1.0)


def _hue_ranges(layer) -> np.ndarray:
    """六类颜色各自的 (色相/360, 饱和度, 明度)。"""
    return np.asarray([[getattr(layer, "adj_hsv_%s_hue" % key.lower()) / 360.0,
                        getattr(layer, "adj_hsv_%s_sat" % key.lower()),
                        getattr(layer, "adj_hsv_%s_light" % key.lower())] for key, _l, _c in HUE_RANGES])


def _hsv(layer, c) -> np.ndarray:
    if layer.adj_colorize:
        tint = hsv_to_rgb(np.array([layer.adj_colorize_hue / 360.0]), np.array([layer.adj_colorize_sat]),
                          np.array([1.0]))
        out = set_lum(np.repeat(tint, len(c), axis=0), luma(np.clip(c, 0.0, 1.0)))
        return lift(out, layer.adj_value)
    h, s, v = rgb_to_hsv(c)
    dh = np.full(len(c), layer.adj_hue / 360.0)
    ds = np.full(len(c), layer.adj_saturation)
    dv = np.full(len(c), layer.adj_value)
    ranges = _hue_ranges(layer)
    if np.any(ranges):
        colored = np.clip(s / 0.05, 0.0, 1.0)
        for k in range(6):
            d = np.abs(np.mod(h - k / 6.0 + 0.5, 1.0) - 0.5) * 360.0
            w = np.clip((45.0 - d) / 30.0, 0.0, 1.0) * colored
            dh += w * ranges[k, 0]
            ds += w * ranges[k, 1]
            dv += w * ranges[k, 2]
    h = np.mod(h + dh, 1.0)
    s = np.clip(s * (1.0 + ds), 0.0, 1.0)
    v = np.clip(lift(v, dv), 0.0, 1.0)
    return hsv_to_rgb(h, s, v)


def apply_color(layer, colors) -> np.ndarray:
    """调整层对一组 sRGB 颜色 (n, 3) 的效果（不算不透明度和蒙版），和着色器一样。"""
    c = np.atleast_2d(np.asarray(colors, np.float64))
    kind = layer.adjust_type
    if kind in PER_CHANNEL:
        return np.stack([per_channel(layer, c[:, k])[0][:, k] for k in range(3)], axis=1)
    if kind == "HSV":
        return _hsv(layer, c)
    if kind == "THRESHOLD":
        y = (luma(c) >= layer.adj_threshold).astype(np.float64)
        return np.stack([y, y, y], axis=1)
    if kind == "GRADIENT_MAP":
        t = luma(c)
        g = ramp.evaluate(layer.adj_ramp, 1.0 - t if layer.adj_ramp_reverse else t)
        return c + (g[:, :3] - c) * g[:, 3:4]
    if kind == "SELECTIVE_COLOR":
        return _selective(c, _selective_rows(layer), layer.adj_sel_method == "RELATIVE")
    if kind == "COLOR_BALANCE":
        lum = (c.max(axis=1) + c.min(axis=1)) * 0.5
        tones = [[getattr(layer, "adj_cb_%s_%s" % (tone, ch)) for tone in ("shadows", "midtones", "highlights")]
                 for ch in "rgb"]
        out = np.stack([_cb_map(c[:, k], lum, *tones[k]) for k in range(3)], axis=1)
        return set_lum(out, luma(c)) if layer.adj_cb_preserve else out
    if kind == "VIBRANCE":
        lum = luma(c)[:, None]
        sat = (c.max(axis=1) - c.min(axis=1))[:, None]
        return clip_color(lum + (c - lum) * ((1.0 + layer.adj_vibrance * (1.0 - sat))
                                             * (1.0 + layer.adj_vib_saturation)))
    if kind == "PHOTO_FILTER":
        color = np.asarray(layer.adj_filter_color[:3], np.float64)
        out = c + (c * color - c) * layer.adj_filter_density
        return set_lum(out, luma(c)) if layer.adj_filter_preserve else out
    if kind == "BLACK_WHITE":
        gray = _bw_gray(np.clip(c, 0.0, 1.0), (layer.adj_bw_red, layer.adj_bw_green, layer.adj_bw_blue),
                        (layer.adj_bw_cyan, layer.adj_bw_magenta, layer.adj_bw_yellow))
        if layer.adj_bw_tint:
            tint = np.repeat(np.asarray(layer.adj_bw_tint_color[:3], np.float64)[None], len(c), axis=0)
            return set_lum(tint, gray)
        return np.stack([gray, gray, gray], axis=1)
    if kind == "CHANNEL_MIXER":
        rows = np.asarray([[getattr(layer, "adj_mix_%s%s" % (out, src)) for src in "rgbc"] for out in "rgb"])
        if layer.adj_mix_mono:
            y = np.clip(c @ rows[0, :3] + rows[0, 3], 0.0, 1.0)
            return np.stack([y, y, y], axis=1)
        return np.clip(c @ rows[:, :3].T + rows[:, 3], 0.0, 1.0)
    if kind == "COLOR_LOOKUP":
        from . import cube

        found = lookup_cube(layer)
        return c if found is None else np.clip(cube.apply(found, c), 0.0, 1.0)
    return c


def scalar_function(layer, xs) -> np.ndarray:
    """金属度、粗糙度、高度这种单个数怎么改：按通道的几种照样改，只对颜色有意义的几种取把它当灰色改完后的明暗。"""
    single = np.ndim(xs) == 0
    xs = np.atleast_1d(np.asarray(xs, np.float64))
    kind = layer.adjust_type
    if kind in PER_CHANNEL:
        out = per_channel(layer, xs)[1]
    elif kind == "HSV":
        out = np.clip(lift(xs, layer.adj_value), 0.0, 1.0)
    elif kind == "THRESHOLD":
        out = (xs >= layer.adj_threshold).astype(np.float64)
    elif kind == "VIBRANCE":
        out = xs
    else:
        out = np.clip(luma(apply_color(layer, np.stack([xs, xs, xs], axis=1))), 0.0, 1.0)
    return out[0] if single else out


# ====================================================================== 查找表
def _selective_rows(layer) -> np.ndarray:
    return np.asarray([[getattr(layer, "adj_sel_%s_%s" % (key.lower(), ink)) for ink, _l, _d in SELECTIVE_INKS]
                       for key, _label, _desc in SELECTIVE_RANGES], np.float64)


def lookup_cube(layer) -> dict | None:
    """颜色查找层用的查找表：导入的文件或内置风格。"""
    from . import cube

    if layer.adj_lut_look == "FILE":
        return cube.decode(layer.adj_lut_data)
    return cube.builtin(layer.adj_lut_look)


def lookup_table(found: dict) -> np.ndarray:
    """查找表放进缓冲的样子：三维的按格子排（补到 LUT_SIZE 的整数倍），一维的重采样成 LUT_SIZE 行。"""
    data = np.asarray(found["data"], np.float32)
    if found["dims"] == 1:
        xs = np.linspace(0.0, 1.0, LUT_SIZE)
        src = np.linspace(0.0, 1.0, len(data))
        table = np.ones((LUT_SIZE, 4), np.float32)
        for k in range(3):
            table[:, k] = np.interp(xs, src, data[:, k])
        return table
    rows = -(-len(data) // LUT_SIZE)
    table = np.ones((rows * LUT_SIZE, 4), np.float32)
    table[:len(data), :3] = data
    return table


def _block(rows) -> np.ndarray:
    table = np.zeros((LUT_SIZE, 4), np.float32)
    rows = np.asarray(rows, np.float32)
    table[:len(rows), :rows.shape[1]] = rows
    return table


def pack(layer) -> tuple:
    """返回 (adj, adj2, adj3, adj4, 查找表)。查找表的行数是 LUT_SIZE 的整数倍。"""
    kind = layer.adjust_type
    code = float(ADJUST_CODE.get(kind, ADJUST_CODE["HSV"]))
    xs = np.linspace(0.0, 1.0, LUT_SIZE)
    first = np.empty((LUT_SIZE, 4), np.float32)
    if kind in PER_CHANNEL:
        rgb, scalar = per_channel(layer, xs)
        first[:, :3] = rgb
        first[:, 3] = scalar
    else:
        first[:, :3] = xs[:, None]
        first[:, 3] = scalar_function(layer, xs)
    a, a2, a3, a4 = [0.0] * 4, [0.0] * 4, [0.0, 0.0, 0.0, code], [0.0] * 4
    extra = None
    if kind == "HSV":
        a = [layer.adj_hue / 360.0, layer.adj_saturation, layer.adj_value, 0.0]
        a4 = [1.0 if layer.adj_colorize else 0.0, layer.adj_colorize_hue / 360.0, layer.adj_colorize_sat, 0.0]
        ranges = _hue_ranges(layer)
        if np.any(ranges):
            extra = _block(ranges)
            a3[0] = 1.0
    elif kind == "THRESHOLD":
        a3[2] = layer.adj_threshold
    elif kind == "GRADIENT_MAP":
        extra = ramp.lut(layer.adj_ramp, LUT_SIZE)
        if layer.adj_ramp_reverse:
            extra = extra[::-1]
    elif kind == "SELECTIVE_COLOR":
        extra = _block(_selective_rows(layer))
        a[0] = 1.0 if layer.adj_sel_method == "RELATIVE" else 0.0
    elif kind == "COLOR_LOOKUP":
        found = lookup_cube(layer)
        if found is not None:
            extra = lookup_table(found)
            lo = np.asarray(found["domain_min"], np.float64)
            hi = np.asarray(found["domain_max"], np.float64)
            a = [float(found["size"]), float(found["dims"]), 1.0, 0.0]
            a2 = [*lo, 0.0]
            a4 = [*(1.0 / np.maximum(hi - lo, 1e-6)), 0.0]
    elif kind == "COLOR_BALANCE":
        a = [layer.adj_cb_shadows_r, layer.adj_cb_shadows_g, layer.adj_cb_shadows_b,
             1.0 if layer.adj_cb_preserve else 0.0]
        a2 = [layer.adj_cb_midtones_r, layer.adj_cb_midtones_g, layer.adj_cb_midtones_b, 0.0]
        a4 = [layer.adj_cb_highlights_r, layer.adj_cb_highlights_g, layer.adj_cb_highlights_b, 0.0]
    elif kind == "VIBRANCE":
        a = [layer.adj_vibrance, layer.adj_vib_saturation, 0.0, 0.0]
    elif kind == "PHOTO_FILTER":
        a = [*layer.adj_filter_color[:3], layer.adj_filter_density]
        a2 = [1.0 if layer.adj_filter_preserve else 0.0, 0.0, 0.0, 0.0]
    elif kind == "BLACK_WHITE":
        a = [layer.adj_bw_red, layer.adj_bw_green, layer.adj_bw_blue, 1.0 if layer.adj_bw_tint else 0.0]
        a2 = [layer.adj_bw_cyan, layer.adj_bw_magenta, layer.adj_bw_yellow, 0.0]
        a4 = [*layer.adj_bw_tint_color[:3], 0.0]
    elif kind == "CHANNEL_MIXER":
        a, a2, a4 = [[getattr(layer, "adj_mix_%s%s" % (out, src)) for src in "rgbc"] for out in "rgb"]
        a3[0] = 1.0 if layer.adj_mix_mono else 0.0
    table = first if extra is None else np.concatenate([first, np.asarray(extra, np.float32)])
    return tuple(a), tuple(a2), tuple(a3), tuple(a4), table


# ====================================================================== 直方图、自动色阶
def histogram(image: np.ndarray) -> np.ndarray:
    """(4, 256)：红、绿、蓝、明暗各 256 档的像素数。image 是 0..1 的 (..., 3)。"""
    image = np.asarray(image, np.float32).reshape(-1, 3)
    q = np.clip(np.rint(image * 255.0), 0, 255).astype(np.int64)
    lum = np.clip(np.rint((image @ LUMA.astype(np.float32)) * 255.0), 0, 255).astype(np.int64)
    return np.stack([np.bincount(q[:, k], minlength=256) for k in range(3)] + [np.bincount(lum, minlength=256)])


def _bounds(values: np.ndarray, clip: float) -> tuple[float, float]:
    """两头各舍掉 clip 比例后的最暗、最亮。几乎没有起伏时不拉。"""
    lo, hi = np.quantile(values, [clip, 1.0 - clip]) if clip > 0 else (values.min(), values.max())
    if hi - lo < 2.0 / 255.0:
        return 0.0, 1.0
    return float(lo), float(hi)


def auto_levels(image: np.ndarray, mode: str, clip: float = 0.001) -> dict:
    """自动色阶（照 Photoshop）：返回要设的色阶属性（没提到的恢复默认）。
    TONE 自动色调：红绿蓝各自拉满；CONTRAST 自动对比度：三个通道一起拉（不改颜色）；
    COLOR 自动颜色：各自拉满，再让中间调回到中性灰。"""
    pixels = np.asarray(image, np.float64).reshape(-1, 3)
    values: dict = {}
    if mode == "CONTRAST":
        lo, hi = _bounds(pixels.ravel(), clip)
        values.update(adj_in_black=lo, adj_in_white=hi)
        return values
    stretched = np.empty_like(pixels)
    for k, channel in enumerate("rgb"):
        lo, hi = _bounds(pixels[:, k], clip)
        values["adj_lv_%s_in_black" % channel] = lo
        values["adj_lv_%s_in_white" % channel] = hi
        stretched[:, k] = levels(pixels[:, k], lo, hi, 1.0, 0.0, 1.0)
    if mode == "COLOR":
        lum = luma(stretched)
        mid = stretched[(lum > 0.2) & (lum < 0.8)]
        if len(mid) >= 16:
            means = np.clip(mid.mean(axis=0), 0.02, 0.98)
            target = float(np.clip(means.mean(), 0.02, 0.98))
            for k, channel in enumerate("rgb"):
                # 让这个通道中间调的平均值变成三个通道的平均值：m^(1/γ) = 目标
                gamma = np.log(means[k]) / np.log(target)
                values["adj_lv_%s_gamma" % channel] = float(np.clip(gamma, 0.1, 10.0))
    return values


def levels_to_curve(in_black: float, in_white: float, gamma: float, points: int = 5) -> tuple:
    """把一组色阶（不算输出黑白场）换成曲线的控制点。"""
    xs = np.linspace(in_black, in_white, points)
    ys = levels(xs, in_black, in_white, gamma, 0.0, 1.0)
    out = [(0.0, 0.0)] if in_black > 1e-4 else []
    out += [(float(x), float(y)) for x, y in zip(xs, ys)]
    if in_white < 1.0 - 1e-4:
        out.append((1.0, 1.0))
    return tuple(out)
