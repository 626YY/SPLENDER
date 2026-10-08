"""颜色查找（照 Photoshop 的「颜色查找」调整）：三维查找表（.cube 文件，或内置的几种色调风格）。

查找表在合成着色器里放在调整层查找表缓冲里：三维的按 红最快、绿次之、蓝最慢 排成 尺寸³ 行 RGB，
一维的重采样成和曲线一样的 LUT_SIZE 行（RGB 三列各管各的通道）。
导入的文件存进图层（压缩后的文本），工程自带，不依赖原文件。
"""
from __future__ import annotations

import base64
import zlib

import numpy as np

from ..core import curve

MAX_SIZE = 65                  # 三维查找表每边最多几格（常见的 17、33、65）
BUILTIN_SIZE = 33
LUMA = np.array([0.2126, 0.7152, 0.0722])


class CubeError(ValueError):
    pass


# ---------------------------------------------------------------- .cube 文件
def parse_cube(text: str) -> dict:
    """读 .cube 文本：返回 {dims: 1 或 3, size, data (n, 3), domain_min, domain_max, title}。"""
    size3 = size1 = 0
    domain_min, domain_max = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    title = ""
    values: list = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        try:
            if key == "TITLE":
                title = line[5:].strip().strip('"')
            elif key == "LUT_3D_SIZE":
                size3 = int(parts[1])
            elif key == "LUT_1D_SIZE":
                size1 = int(parts[1])
            elif key == "DOMAIN_MIN":
                domain_min = tuple(float(v) for v in parts[1:4])
            elif key == "DOMAIN_MAX":
                domain_max = tuple(float(v) for v in parts[1:4])
            elif key in ("LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE"):
                lo, hi = float(parts[1]), float(parts[2])
                domain_min, domain_max = (lo, lo, lo), (hi, hi, hi)
            elif key[0].isdigit() or key[0] in "-+.":
                values.append([float(v) for v in parts[:3]])
        except (IndexError, ValueError) as error:
            raise CubeError("这一行读不懂：%s" % raw.strip()) from error
    if size3 and size1:
        raise CubeError("同时写了一维和三维的尺寸")
    if size3:
        if not 2 <= size3 <= MAX_SIZE:
            raise CubeError("三维查找表的尺寸 %d 不支持（2 到 %d）" % (size3, MAX_SIZE))
        dims, size, expected = 3, size3, size3 ** 3
    elif size1:
        if not 2 <= size1 <= 65536:
            raise CubeError("一维查找表的尺寸 %d 不支持" % size1)
        dims, size, expected = 1, size1, size1
    else:
        raise CubeError("没有写查找表的尺寸（LUT_3D_SIZE 或 LUT_1D_SIZE）")
    if len(values) != expected:
        raise CubeError("应有 %d 行颜色，实际 %d 行" % (expected, len(values)))
    data = np.asarray(values, np.float32)
    if any(hi <= lo for lo, hi in zip(domain_min, domain_max)):
        raise CubeError("输入范围不对")
    return {"dims": dims, "size": size, "data": data, "domain_min": tuple(domain_min),
            "domain_max": tuple(domain_max), "title": title}


def encode(cube: dict) -> str:
    """存进图层的文本：头 + 压缩的半精度数据。"""
    head = "SPLUT1;%d;%d;%s;%s;" % (cube["dims"], cube["size"], ",".join("%r" % v for v in cube["domain_min"]),
                                   ",".join("%r" % v for v in cube["domain_max"]))
    body = zlib.compress(np.asarray(cube["data"], np.float16).tobytes(), 6)
    return head + base64.b64encode(body).decode("ascii")


_decoded: dict = {}


def decode(text: str) -> dict | None:
    """encode 的反过来；坏数据返回 None。同一份文本只解一次。"""
    if not text:
        return None
    cached = _decoded.get(text)
    if cached is not None:
        return cached
    try:
        magic, dims, size, lo, hi, body = text.split(";", 5)
        if magic != "SPLUT1":
            return None
        dims, size = int(dims), int(size)
        data = np.frombuffer(zlib.decompress(base64.b64decode(body)), np.float16).astype(np.float32).reshape(-1, 3)
        cube = {"dims": dims, "size": size, "data": data,
                "domain_min": tuple(float(v) for v in lo.split(",")),
                "domain_max": tuple(float(v) for v in hi.split(","))}
        if len(data) != (size ** 3 if dims == 3 else size):
            return None
    except Exception:  # noqa: BLE001
        return None
    if len(_decoded) > 16:
        _decoded.clear()
    _decoded[text] = cube
    return cube


def write_cube(path: str, data: np.ndarray, size: int, title: str = "") -> None:
    """写一个三维 .cube 文件（红最快）。"""
    lines = ['TITLE "%s"' % title] if title else []
    lines.append("LUT_3D_SIZE %d" % size)
    lines += ["%.6f %.6f %.6f" % tuple(row) for row in np.asarray(data, np.float64)]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- 内置的色调风格
def grid(size: int) -> np.ndarray:
    """三维查找表的输入颜色：(size³, 3)，红最快、蓝最慢。"""
    axis = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)


def _luma(c: np.ndarray) -> np.ndarray:
    return c @ LUMA


def _saturate(c: np.ndarray, amount: float) -> np.ndarray:
    lum = _luma(c)[:, None]
    return lum + (c - lum) * amount


def _set_lum(c: np.ndarray, lum: np.ndarray) -> np.ndarray:
    return np.clip(c + (lum - _luma(c))[:, None], 0.0, 1.0)


def _curves(c: np.ndarray, rgb=None, r=None, g=None, b=None) -> np.ndarray:
    out = np.empty_like(c)
    for k, own in enumerate((r, g, b)):
        v = c[:, k]
        if own is not None:
            v = curve.evaluate(own, v)
        if rgb is not None:
            v = curve.evaluate(rgb, v)
        out[:, k] = v
    return out


def _split_tone(c: np.ndarray, shadow, highlight, amount: float) -> np.ndarray:
    """阴影染一种颜色、高光染另一种，明暗不变。"""
    lum = _luma(c)
    tint = (np.outer(1.0 - lum, np.asarray(shadow) - 0.5) + np.outer(lum, np.asarray(highlight) - 0.5)) * amount
    return _set_lum(c + tint, lum)


def _look(name: str, c: np.ndarray) -> np.ndarray:
    if name == "WARM_FILM":
        c = _curves(c, rgb=((0.0, 0.03), (0.25, 0.22), (0.75, 0.8), (1.0, 0.97)),
                    r=((0.0, 0.02), (0.5, 0.54), (1.0, 1.0)), b=((0.0, 0.0), (0.5, 0.45), (1.0, 0.92)))
        return _saturate(c, 0.9)
    if name == "TEAL_ORANGE":
        c = _split_tone(c, (0.1, 0.55, 0.6), (1.0, 0.62, 0.3), 0.35)
        c = _curves(c, rgb=((0.0, 0.0), (0.25, 0.21), (0.75, 0.8), (1.0, 1.0)))
        return _saturate(c, 1.1)
    if name == "FADED":
        c = _curves(c, rgb=((0.0, 0.12), (0.5, 0.52), (1.0, 0.92)))
        return _saturate(c, 0.72)
    if name == "COLD":
        c = _curves(c, rgb=((0.0, 0.0), (0.25, 0.19), (0.75, 0.82), (1.0, 1.0)),
                    r=((0.0, 0.0), (0.5, 0.45), (1.0, 0.96)), b=((0.0, 0.04), (0.5, 0.56), (1.0, 1.0)))
        return _saturate(c, 0.88)
    if name == "VINTAGE":
        sepia = np.clip(np.stack([_luma(c) * 1.07 + 0.03, _luma(c) * 0.95, _luma(c) * 0.76], axis=1), 0, 1)
        c = c * 0.65 + sepia * 0.35
        return _curves(c, rgb=((0.0, 0.08), (0.5, 0.52), (1.0, 0.93)))
    if name == "BLEACH":
        lum = _luma(c)[:, None]
        overlay = np.where(c < 0.5, 2.0 * c * lum, 1.0 - 2.0 * (1.0 - c) * (1.0 - lum))
        c = c * 0.4 + overlay * 0.6
        return _saturate(c, 0.55)
    if name == "NIGHT":
        lum = _luma(c)
        c = _set_lum(c * 0.6 + np.array([0.18, 0.28, 0.55]) * 0.4, lum)
        c = _curves(c, rgb=((0.0, 0.0), (0.5, 0.36), (1.0, 0.85)))
        return _saturate(c, 0.8)
    if name == "CROSS":
        return _curves(c, r=((0.0, 0.0), (0.25, 0.18), (0.75, 0.86), (1.0, 1.0)),
                       g=((0.0, 0.0), (0.25, 0.2), (0.75, 0.88), (1.0, 1.0)),
                       b=((0.0, 0.2), (1.0, 0.8)))
    return c


_builtin: dict = {}


def builtin(name: str, size: int = BUILTIN_SIZE) -> dict:
    """内置风格的三维查找表（和导入的文件同一种形式）。"""
    cached = _builtin.get((name, size))
    if cached is None:
        data = np.clip(_look(name, grid(size)), 0.0, 1.0).astype(np.float32)
        cached = {"dims": 3, "size": size, "data": data, "domain_min": (0.0, 0.0, 0.0),
                  "domain_max": (1.0, 1.0, 1.0)}
        _builtin[(name, size)] = cached
    return cached


# ---------------------------------------------------------------- 查表（CPU 上，和着色器一样）
def apply(cube: dict, colors: np.ndarray) -> np.ndarray:
    """用查找表改一组颜色（(n, 3)，0..1），三维的按三线性插值。测试和预览用。"""
    c = np.asarray(colors, np.float64)
    lo = np.asarray(cube["domain_min"])
    hi = np.asarray(cube["domain_max"])
    c = np.clip((c - lo) / (hi - lo), 0.0, 1.0)
    size, data = cube["size"], np.asarray(cube["data"], np.float64)
    if cube["dims"] == 1:
        xs = np.linspace(0.0, 1.0, size)
        return np.stack([np.interp(c[:, k], xs, data[:, k]) for k in range(3)], axis=1)
    p = c * (size - 1)
    i0 = np.minimum(np.floor(p).astype(np.int64), size - 2)
    f = p - i0
    out = np.zeros_like(c)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                     * (f[:, 2] if dz else 1 - f[:, 2]))
                index = ((i0[:, 2] + dz) * size + (i0[:, 1] + dy)) * size + (i0[:, 0] + dx)
                out += w[:, None] * data[index]
    return out
