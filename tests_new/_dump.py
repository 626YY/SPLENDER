"""调试辅助：把某个图层某种页的某一级拼成一张图。"""
import numpy as np
from PIL import Image

from splender.engine.pagepool import PAGE


def plane_image(engine, layer_uid, plane, level, channels=(0, 1, 2), scale=1.0):
    store = engine.layers.stores[layer_uid].planes[plane]
    grid = store.levels[level]
    comps = {"rgba8": 4, "rg16f": 2, "r8": 1}[store.fmt]
    dtype = np.float16 if store.fmt == "rg16f" else np.uint8
    out = np.zeros((grid.size * PAGE, grid.size * PAGE, 3), np.float32)
    for (tx, ty), page in grid.pages.items():
        raw = np.frombuffer(engine.cache.page_bytes(page), dtype).reshape(PAGE, PAGE, comps).astype(np.float32)
        if dtype == np.uint8:
            raw = raw / 255.0
        tile = np.zeros((PAGE, PAGE, 3), np.float32)
        for i, c in enumerate(channels):
            if c < comps:
                tile[:, :, i] = raw[:, :, c] * scale
        out[ty * PAGE:(ty + 1) * PAGE, tx * PAGE:(tx + 1) * PAGE] = tile
    return Image.fromarray((np.clip(out[::-1], 0, 1) * 255).astype(np.uint8))
