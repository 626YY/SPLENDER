"""把渲染结果存成图片：PNG（8 位，可带透明）或 EXR（32 位浮点线性，保留高光细节）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_png(image_u8: np.ndarray, path: str | Path, alpha: bool = True) -> None:
    from PIL import Image

    data = np.ascontiguousarray(image_u8 if alpha else image_u8[:, :, :3])
    Image.fromarray(data, "RGBA" if data.shape[2] == 4 else "RGB").save(str(path))


def srgb_to_linear(image_u8: np.ndarray) -> np.ndarray:
    c = image_u8[:, :, :3].astype(np.float32) / 255.0
    rgb = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    alpha = image_u8[:, :, 3:4].astype(np.float32) / 255.0 if image_u8.shape[2] == 4 else np.ones_like(c[:, :, :1])
    return np.concatenate([rgb, alpha], axis=2).astype(np.float32)


def save_exr(linear_rgba: np.ndarray, path: str | Path) -> None:
    from ..paths import ensure_vendor_path

    ensure_vendor_path()
    import OpenEXR

    data = np.ascontiguousarray(linear_rgba.astype(np.float32))
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    channels = {"R": np.ascontiguousarray(data[:, :, 0]), "G": np.ascontiguousarray(data[:, :, 1]),
                "B": np.ascontiguousarray(data[:, :, 2])}
    if data.shape[2] > 3:
        channels["A"] = np.ascontiguousarray(data[:, :, 3])
    with OpenEXR.File(header, channels) as handle:
        handle.write(str(path))


def save_image(path: str | Path, preview_u8: np.ndarray | None, linear: np.ndarray | None) -> str:
    """按扩展名存：.exr 优先用线性结果（没有时把 sRGB 画面转回线性），其余存 PNG。返回实际写入的路径。"""
    path = Path(path)
    if path.suffix.lower() == ".exr":
        data = linear if linear is not None else srgb_to_linear(preview_u8)
        save_exr(data, path)
    else:
        if path.suffix.lower() not in (".png",):
            path = path.with_suffix(".png")
        save_png(preview_u8, path)
    return str(path)
