"""纹理集换分辨率：每个图层的像素按新边长重新采样，生成新的 LayerStore（调用方换上去并记撤销）。

- 缩小 2^k 倍：新图的第 0 级直接取旧图的第 k 级（多级本来就是按 2×2 平均出来的，等于盒式滤波），逐页拷过去；
- 放大 2^k 倍：新图每个纹素在旧图第 0 级上双线性取样（颜色和数值按覆盖度加权，和合成器的混合一致）；
- 按旧图的一行格子分批：只把这一行和上下相邻两行的旧页换入显存并钉住，16K 也不会把页池挤爆；
- 最后逐级生成新图的上一级（和贴图搬迁共用那段代码）。
"""
from __future__ import annotations

import logging
import math
import time
from types import SimpleNamespace

import numpy as np

from ..doc.project import PLANE_FORMAT
from .layerstore import PLANE_KIND, LayerStore
from .pageops import KIND_COLOR, KIND_HEIGHT, KIND_MASK, KIND_MR, SRGB_GLSL
from .pagepool import FORMATS, MAX_ARRAYS, PAGE, bind_samplers, glsl_fetch

log = logging.getLogger("splender.engine.resize")


def _resample_shader(kind: int, fetch: str) -> str:
    fmt = {KIND_COLOR: "rgba8", KIND_MR: "rgba8", KIND_HEIGHT: "rg16f", KIND_MASK: "r8"}[kind]
    glsl = FORMATS[fmt][5]
    return f"""#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout({glsl}, binding = 0) writeonly uniform image2DArray u_dst;
layout(std430, binding = 1) readonly buffer Jobs {{ int jobs[]; }};
layout(std430, binding = 2) readonly buffer Table {{ int table[]; }};
uniform int u_base; uniform int u_layer_mask; uniform int u_grid; uniform float u_src_res; uniform float u_dst_res;
uniform float u_mask_default;
{fetch}
{SRGB_GLSL}
const int KIND = {kind};
vec4 empty_value() {{ return KIND == 3 ? vec4(u_mask_default) : vec4(0.0); }}
vec4 tap(ivec2 t) {{
  int size = int(u_src_res);
  t = clamp(t, ivec2(0), ivec2(size - 1));
  ivec2 tile = t >> 8;
  int slot = table[tile.y * u_grid + tile.x];
  if (slot < 0) return empty_value();
  ivec2 q = t & 255;
  if (KIND == 0 || KIND == 1) return fetch_rgba8(slot, q);
  if (KIND == 2) return fetch_rg16f(slot, q);
  return fetch_r8(slot, q);
}}
vec2 pair(vec2 a, vec2 b, vec2 c, vec2 d, vec4 w) {{
  float cov = a.y * w.x + b.y * w.y + c.y * w.z + d.y * w.w;
  float val = a.x * a.y * w.x + b.x * b.y * w.y + c.x * c.y * w.z + d.x * d.y * w.w;
  return vec2(cov > 0.0 ? val / cov : 0.0, cov);
}}
void main() {{
  int j = (u_base + int(gl_GlobalInvocationID.z)) * 8;
  int dst = jobs[j];
  ivec2 tile = ivec2(jobs[j + 1], jobs[j + 2]);
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  vec2 uv = (vec2(tile * 256 + p) + 0.5) / u_dst_res;
  vec2 st = uv * u_src_res - 0.5;
  ivec2 i0 = ivec2(floor(st));
  vec2 f = st - vec2(i0);
  vec4 w = vec4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
  vec4 a = tap(i0), b = tap(i0 + ivec2(1, 0)), c = tap(i0 + ivec2(0, 1)), d = tap(i0 + ivec2(1, 1));
  vec4 value;
  if (KIND == 0) {{
    float cov = a.a * w.x + b.a * w.y + c.a * w.z + d.a * w.w;
    vec3 rgb = srgb_to_linear(a.rgb) * a.a * w.x + srgb_to_linear(b.rgb) * b.a * w.y
             + srgb_to_linear(c.rgb) * c.a * w.z + srgb_to_linear(d.rgb) * d.a * w.w;
    value = cov > 0.0 ? vec4(linear_to_srgb(rgb / cov), cov) : vec4(0.0);
  }} else if (KIND == 1) {{
    value = vec4(pair(a.rg, b.rg, c.rg, d.rg, w), pair(a.ba, b.ba, c.ba, d.ba, w));
  }} else if (KIND == 2) {{
    value = vec4(pair(a.rg, b.rg, c.rg, d.rg, w), 0.0, 0.0);
  }} else {{
    value = vec4(a.r * w.x + b.r * w.y + c.r * w.z + d.r * w.w);
  }}
  imageStore(u_dst, ivec3(p, dst & u_layer_mask), value);
}}
"""


class TextureResize:
    """一套纹理集的全部图层换成 new_size 边长。同步完成，返回 {图层 uid: 新的 LayerStore}。"""

    def __init__(self, engine, ts, new_size: int) -> None:
        self.engine = engine
        self.ts = ts
        self.old_size = int(ts.size)
        self.new_size = int(new_size)
        self.old_grid = self.old_size // PAGE
        self.new_grid = self.new_size // PAGE
        self.stats = {}

    def run(self) -> dict:
        started = time.perf_counter()
        engine = self.engine
        sources = {}
        for layer in self.ts.layers:
            store = engine.layers.stores.get(layer.uid)
            if store is not None and store.page_count() > 0:
                sources[layer.uid] = (layer, store)
        if not sources or self.new_size == self.old_size:
            return {}
        ctx = engine.ctx
        fetch = "\n".join(glsl_fetch(fmt, MAX_ARRAYS[fmt], engine.pools.shift) for fmt in FORMATS)
        programs = {kind: ctx.compute_shader(_resample_shader(kind, fetch))
                    for kind in (KIND_COLOR, KIND_MR, KIND_HEIGHT, KIND_MASK)}
        table = ctx.buffer(reserve=max(1, self.old_grid * self.old_grid) * 4)
        results = {uid: LayerStore(self.new_grid) for uid in sources}
        try:
            for uid, (layer, store) in sources.items():
                for plane, pstore in store.planes.items():
                    self._plane(plane, pstore, results[uid], programs, table, float(layer.mask_default))
            from .transfer import TextureTransfer

            TextureTransfer._mips(SimpleNamespace(engine=engine), sources, results)
        finally:
            for program in programs.values():
                program.release()
            table.release()
        self.stats.update(seconds=round(time.perf_counter() - started, 2), layers=len(results),
                          pages=int(sum(store.page_count() for store in results.values())))
        log.info("纹理集「%s」换分辨率 %d → %d：%d 层，%d 页，用时 %.2f 秒", self.ts.name, self.old_size, self.new_size,
                 len(results), self.stats["pages"], self.stats["seconds"])
        return results

    def _plane(self, plane: int, pstore, result: LayerStore, programs: dict, table, mask_default: float) -> None:
        engine = self.engine
        cache = engine.cache
        fmt = PLANE_FORMAT[plane]
        kind = PLANE_KIND[plane]
        if self.new_size < self.old_size:
            level = int(round(math.log2(self.old_size / self.new_size)))
            shift = 0
        else:
            level = 0
            shift = int(round(math.log2(self.new_size / self.old_size)))
        if level >= len(pstore.levels):
            return
        source = pstore.levels[level]
        if not source.pages:
            return
        src_grid = self.old_grid >> level
        src_res = float(src_grid * PAGE)
        target = result.plane(plane).levels[0]
        rows = sorted({y for (_x, y) in source.pages})
        for row in rows:
            # 这一行的源页生成哪些新格：缩小是一一对应，放大是每个源格变成 2^k × 2^k 个新格
            columns = sorted(x for (x, y) in source.pages if y == row)
            tiles = []
            for x in columns:
                for dy in range(1 << shift):
                    for dx in range(1 << shift):
                        tiles.append(((x << shift) + dx, (row << shift) + dy))
            if not tiles:
                continue
            needed = [(x, y) for (x, y) in source.pages if abs(y - row) <= 1]
            pages = [source.pages[key] for key in needed]
            cache.make_resident(pages)
            cache.pin_many(pages)
            try:
                slots = np.full(src_grid * src_grid, -1, np.int32)
                for (x, y), page in zip(needed, pages):
                    slots[y * src_grid + x] = page.slot
                table.write(slots.tobytes())
                news = cache.try_new_pages(fmt, len(tiles), block=True)
                jobs = np.full((len(news), 8), -1, np.int32)
                jobs[:, 0] = np.fromiter((page.slot for page in news), np.int32, len(news))
                jobs[:, 1] = [t[0] for t in tiles]
                jobs[:, 2] = [t[1] for t in tiles]
                program = programs[kind]
                unit = 0
                for name in FORMATS:
                    unit = bind_samplers(program, engine.pools, name, unit)
                table.bind_to_storage_buffer(2)
                program["u_grid"] = int(src_grid)
                program["u_src_res"] = src_res
                program["u_dst_res"] = float(self.new_size)
                if "u_mask_default" in program:
                    program["u_mask_default"] = float(mask_default)
                engine.ops._run_by_array(program, fmt, jobs)
                cache.note_new_pages(news)
                for (x, y), page in zip(tiles, news):
                    cache.set_page(target, int(x), int(y), page)
            finally:
                cache.pin_many(pages, -1)
