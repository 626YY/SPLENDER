"""把图层像素从旧 UV 搬到新 UV：重构、重新展开 UV 之后，原来画的内容跟着模型走，不会乱。

做法（显卡，按格行处理，每行 grid 个 256² 的格）：
1. 旧网格（这套贴图对应的三角形）建 BVH，三角形的旧 UV 一起传上去。
2. 新网格按新 UV 光栅化这一行：岛内三角形在前，扩边在后（深度测试保证扩边不盖住岛内），
   每个纹素得到三维位置和法线。
3. 每个纹素沿法线两个方向各发一条射线，取离得最近的交点，按重心坐标换算出旧 UV（找不到就空着）。
4. 找旧 UV 的同时在显卡上记下每个新格用到了哪些旧格（位图，读回只有几十 KB）；只给真有内容可取的新格分配页。
5. 每种页（颜色、金属粗糙、高度、蒙版）：要用到的旧页换入显存并钉住，按页表间接做双线性取样，
   颜色和数值按覆盖度加权（和合成器的混合一致），写进新页。
6. 全部做完后逐级生成上一级的页。蒙版缺的象限按蒙版初始值补。

结果是每个图层一份新的 LayerStore；调用方把它换上去并记撤销（换回旧的）。
"""
from __future__ import annotations

import logging
import time

import moderngl
import numpy as np

from ..doc.project import PLANE_FORMAT, PLANE_MASK
from ..render.pathtracer.bvh import build_bvh
from ..render.pathtracer.shaders import BVH_GLSL, STACK
from .layerstore import PLANE_KIND, LayerStore
from .pageops import KIND_COLOR, KIND_HEIGHT, KIND_MASK, KIND_MR, SRGB_GLSL
from .pagepool import FORMATS, MAX_ARRAYS, PAGE, bind_samplers, glsl_fetch

log = logging.getLogger("splender.engine.transfer")

_RASTER_MAIN_VERT = """#version 430
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
uniform vec2 u_origin; uniform vec2 u_size; uniform float u_res; uniform float u_depth;
out vec3 v_pos; out vec3 v_nrm;
void main() {
  vec2 t = (in_uv * u_res - u_origin) / u_size;
  gl_Position = vec4(t * 2.0 - 1.0, u_depth, 1.0);
  v_pos = in_pos; v_nrm = in_nrm;
}
"""

_RASTER_SKIRT_VERT = _RASTER_MAIN_VERT.replace("in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;",
                                               "in vec2 in_uv; in vec3 in_pos; in vec3 in_nrm;")

_RASTER_FRAG = """#version 430
in vec3 v_pos; in vec3 v_nrm;
layout(location = 0) out vec4 o_pos;
layout(location = 1) out vec4 o_nrm;
layout(location = 2) out vec4 o_known;
void main() {
  float l = length(v_nrm);
  o_pos = vec4(v_pos, 1.0);
  o_nrm = vec4(l > 1e-20 ? v_nrm / l : vec3(0.0, 1.0, 0.0), 1.0);
  o_known = vec4(0.0);
}
"""

# 形状没变（只换了 UV）的三角形：旧 UV 直接当顶点属性插值，不用发射线
_RASTER_KNOWN_VERT = """#version 430
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in vec2 in_old_uv;
uniform vec2 u_origin; uniform vec2 u_size; uniform float u_res; uniform float u_depth;
out vec3 v_pos; out vec3 v_nrm; out vec2 v_old;
void main() {
  vec2 t = (in_uv * u_res - u_origin) / u_size;
  gl_Position = vec4(t * 2.0 - 1.0, u_depth, 1.0);
  v_pos = in_pos; v_nrm = in_nrm; v_old = in_old_uv;
}
"""

_RASTER_KNOWN_FRAG = """#version 430
in vec3 v_pos; in vec3 v_nrm; in vec2 v_old;
layout(location = 0) out vec4 o_pos;
layout(location = 1) out vec4 o_nrm;
layout(location = 2) out vec4 o_known;
void main() {
  float l = length(v_nrm);
  o_pos = vec4(v_pos, 1.0);
  o_nrm = vec4(l > 1e-20 ? v_nrm / l : vec3(0.0, 1.0, 0.0), 1.0);
  o_known = vec4(v_old, 1.0, 1.0);
}
"""

_MAP = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) readonly uniform image2D u_pos;
layout(rgba16f, binding = 1) readonly uniform image2D u_nrm;
layout(rg32f, binding = 2) writeonly uniform image2D o_uv;
layout(rgba32f, binding = 3) readonly uniform image2D u_known;
struct Node { vec3 bmin; int first; vec3 bmax; int count; };
layout(std430, binding = 1) readonly buffer Nodes { Node nodes[]; };
layout(std430, binding = 2) readonly buffer TriPos { vec4 tri_pos[]; };
layout(std430, binding = 3) readonly buffer TriUV { vec4 tri_uv[]; };
layout(std430, binding = 4) buffer Refs { uint refs[]; };   // 每个新格一行位图：用到了哪些旧格
#define BIG 1e30
uniform ivec2 u_size;
uniform float u_dist;
uniform float u_eps;
uniform int u_grid;
uniform float u_res;
""" + BVH_GLSL + """
void mark_tile(int column, int tx, int ty) {
  int index = ty * u_grid + tx;
  int words = (u_grid * u_grid + 31) / 32;
  atomicOr(refs[column * words + (index >> 5)], 1u << uint(index & 31));
}
// 一个工作组（8×8 纹素，同在一个新格里）先在共享内存里合出用到的旧格范围，范围小就由一个线程整块标记，
// 避免几百万个线程往同几个字上做原子操作；范围大（跨了接缝）才各自标记
shared int s_box[4];
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  bool inside = p.x < u_size.x && p.y < u_size.y;
  if (gl_LocalInvocationIndex == 0u) {
    s_box[0] = 1 << 30; s_box[1] = 1 << 30; s_box[2] = -1; s_box[3] = -1;
  }
  barrier();
  vec2 uv = vec2(-1.0);
  if (inside) {
    vec4 P = imageLoad(u_pos, p);
    if (P.w > 0.0) {
      vec4 known = imageLoad(u_known, p);
      if (known.z > 0.5) {
        uv = known.xy;
      } else {
        vec3 n = normalize(imageLoad(u_nrm, p).xyz);
        int t1, t2; vec2 b1, b2; float d1, d2;
        // 两个方向各一条：起点往外挪一点，重合的表面也能打到；第一条已经贴着表面打中就不用第二条
        bool h1 = trace(P.xyz + n * u_eps, -n, u_dist + u_eps, false, t1, b1, d1);
        float e1 = h1 ? abs(d1 - u_eps) : BIG;
        float e2 = BIG;
        if (e1 > u_eps * 4.0) {
          bool h2 = trace(P.xyz - n * u_eps, n, u_dist + u_eps, false, t2, b2, d2);
          e2 = h2 ? abs(d2 - u_eps) : BIG;
        }
        if (e1 < BIG || e2 < BIG) {
          int tri = e1 <= e2 ? t1 : t2;
          vec2 bc = e1 <= e2 ? b1 : b2;
          vec4 a = tri_uv[tri * 2];
          vec4 b = tri_uv[tri * 2 + 1];
          uv = a.xy * (1.0 - bc.x - bc.y) + a.zw * bc.x + b.xy * bc.y;
        }
      }
    }
    imageStore(o_uv, p, vec4(uv, 0.0, 0.0));
  }
  bool valid = uv.x >= 0.0;
  ivec2 lo = ivec2(0), hi = ivec2(-1);
  if (valid) {
    int last = int(u_res) - 1;
    ivec2 i0 = ivec2(floor(uv * u_res - 0.5));
    lo = clamp(i0, ivec2(0), ivec2(last)) >> 8;
    hi = clamp(i0 + 1, ivec2(0), ivec2(last)) >> 8;
    atomicMin(s_box[0], lo.x);
    atomicMin(s_box[1], lo.y);
    atomicMax(s_box[2], hi.x);
    atomicMax(s_box[3], hi.y);
  }
  barrier();
  int column = int(gl_WorkGroupID.x * 8u) >> 8;
  bool spread = (s_box[2] - s_box[0] + 1) * (s_box[3] - s_box[1] + 1) > 16;
  if (!spread) {
    if (gl_LocalInvocationIndex == 0u && s_box[2] >= 0) {
      for (int ty = s_box[1]; ty <= s_box[3]; ty++)
        for (int tx = s_box[0]; tx <= s_box[2]; tx++)
          mark_tile(column, tx, ty);
    }
  } else if (valid) {
    mark_tile(column, lo.x, lo.y);
    mark_tile(column, hi.x, lo.y);
    mark_tile(column, lo.x, hi.y);
    mark_tile(column, hi.x, hi.y);
  }
}
"""


def _sample_shader(kind: int, fetch: str) -> str:
    fmt = {KIND_COLOR: "rgba8", KIND_MR: "rgba8", KIND_HEIGHT: "rg16f", KIND_MASK: "r8"}[kind]
    glsl = FORMATS[fmt][5]
    return f"""#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout({glsl}, binding = 0) writeonly uniform image2DArray u_dst;
layout(std430, binding = 1) readonly buffer Jobs {{ int jobs[]; }};
layout(std430, binding = 2) readonly buffer Table {{ int table[]; }};
uniform sampler2D u_uv;
uniform int u_base; uniform int u_layer_mask; uniform int u_grid; uniform float u_res; uniform float u_mask_default;
{fetch}
{SRGB_GLSL}
const int KIND = {kind};
vec4 empty_value() {{ return KIND == 3 ? vec4(u_mask_default) : vec4(0.0); }}
vec4 tap(ivec2 t) {{
  int size = int(u_res);
  t = clamp(t, ivec2(0), ivec2(size - 1));
  ivec2 tile = t >> 8;
  int slot = table[tile.y * u_grid + tile.x];
  if (slot < 0) return empty_value();
  ivec2 q = t & 255;
  if (KIND == 0 || KIND == 1) return fetch_rgba8(slot, q);
  if (KIND == 2) return fetch_rg16f(slot, q);
  return fetch_r8(slot, q);
}}
vec2 pair(vec2 a, vec2 b, vec2 c, vec2 d, vec4 w) {{       // (数值, 覆盖度) 按覆盖度加权
  float cov = a.y * w.x + b.y * w.y + c.y * w.z + d.y * w.w;
  float val = a.x * a.y * w.x + b.x * b.y * w.y + c.x * c.y * w.z + d.x * d.y * w.w;
  return vec2(cov > 0.0 ? val / cov : 0.0, cov);
}}
void main() {{
  int j = (u_base + int(gl_GlobalInvocationID.z)) * 8;
  int dst = jobs[j];
  int column = jobs[j + 1];
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  vec2 uv = texelFetch(u_uv, ivec2(column * 256 + p.x, p.y), 0).xy;
  vec4 value = empty_value();
  if (uv.x >= 0.0) {{
    vec2 st = uv * u_res - 0.5;
    ivec2 i0 = ivec2(floor(st));
    vec2 f = st - vec2(i0);
    vec4 w = vec4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    vec4 a = tap(i0), b = tap(i0 + ivec2(1, 0)), c = tap(i0 + ivec2(0, 1)), d = tap(i0 + ivec2(1, 1));
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
  }}
  imageStore(u_dst, ivec3(p, dst & u_layer_mask), value);
}}
"""


class TextureTransfer:
    """一套纹理集的全部图层，从旧几何搬到当前几何。同步完成。

    old_tris (N,3,3)、old_uvs (N,3,2)：旧几何里属于这套纹理集的三角形（所有用到它的模型，没变的模型也算上，
    它们的内容就原样搬过去）。new_sets：当前几何在这套纹理集上的 SetGPU 列表。"""

    def __init__(self, engine, ts, old_tris: np.ndarray, old_uvs: np.ndarray, new_sets: list,
                 distance: float | None = None, known_uvs: dict | None = None) -> None:
        self.engine = engine
        self.ctx = engine.ctx
        self.ts = ts
        self.grid = ts.size // PAGE
        self.res = ts.size
        corners = np.ascontiguousarray(old_tris, np.float32).reshape(-1, 3, 3)
        uvs = np.ascontiguousarray(old_uvs, np.float32).reshape(-1, 3, 2)
        self.old_tris = corners
        self.old_uvs = uvs
        self.new_sets = new_sets                     # 当前几何在这套纹理集上的 SetGPU 列表
        self.known_uvs = known_uvs or {}             # id(SetGPU) -> 形状没变时每个角点的旧 UV (3T, 2)
        lo = corners.reshape(-1, 3).min(axis=0) if len(corners) else np.zeros(3)
        hi = corners.reshape(-1, 3).max(axis=0) if len(corners) else np.ones(3)
        diag = float(np.linalg.norm(hi - lo)) or 1.0
        self.distance = float(distance) if distance is not None else diag * 0.02
        self.eps = diag * 1e-5
        self.stats = {}

    # ---------------------------------------------------------------- 主流程
    def run(self) -> dict:
        """返回 {图层 uid: 新的 LayerStore}。没有可搬的内容时返回空字典。"""
        started = time.perf_counter()
        engine = self.engine
        sources = {}
        for layer in self.ts.layers:
            store = engine.layers.stores.get(layer.uid)
            if store is not None and store.page_count() > 0:
                sources[layer.uid] = (layer, store)
        if not sources or len(self.old_tris) == 0 or not self.new_sets:
            return {}
        ctx = self.ctx
        bvh = build_bvh(self.old_tris[:, 0], self.old_tris[:, 1], self.old_tris[:, 2])
        if bvh["depth"] >= STACK:
            raise RuntimeError("旧模型的 BVH 太深（%d 层），超过上限 %d" % (bvh["depth"], STACK))
        order = bvh["order"]
        tri_pos = np.zeros((len(order), 3, 4), np.float32)
        v0 = self.old_tris[order, 0]
        tri_pos[:, 0, :3] = v0
        tri_pos[:, 1, :3] = self.old_tris[order, 1] - v0
        tri_pos[:, 2, :3] = self.old_tris[order, 2] - v0
        tri_uv = np.zeros((len(order), 2, 4), np.float32)
        tri_uv[:, 0, :2] = self.old_uvs[order, 0]
        tri_uv[:, 0, 2:] = self.old_uvs[order, 1]
        tri_uv[:, 1, :2] = self.old_uvs[order, 2]
        width = self.grid * PAGE
        res = {}
        try:
            res["nodes"] = ctx.buffer(np.ascontiguousarray(bvh["nodes"], np.float32).tobytes())
            res["tri_pos"] = ctx.buffer(tri_pos.tobytes())
            res["tri_uv"] = ctx.buffer(tri_uv.tobytes())
            res["pos"] = ctx.texture((width, PAGE), 4, dtype="f4")
            res["nrm"] = ctx.texture((width, PAGE), 4, dtype="f2")
            res["uv"] = ctx.texture((width, PAGE), 2, dtype="f4")
            res["uv"].filter = (moderngl.NEAREST, moderngl.NEAREST)
            res["known"] = ctx.texture((width, PAGE), 4, dtype="f4")
            res["depth"] = ctx.depth_texture((width, PAGE))
            res["fbo"] = ctx.framebuffer([res["pos"], res["nrm"], res["known"]], res["depth"])
            res["main"] = ctx.program(vertex_shader=_RASTER_MAIN_VERT, fragment_shader=_RASTER_FRAG)
            res["main_known"] = ctx.program(vertex_shader=_RASTER_KNOWN_VERT, fragment_shader=_RASTER_KNOWN_FRAG)
            res["skirt"] = ctx.program(vertex_shader=_RASTER_SKIRT_VERT, fragment_shader=_RASTER_FRAG)
            res["map"] = ctx.compute_shader(_MAP)
            fetch = "\n".join(glsl_fetch(fmt, MAX_ARRAYS[fmt], engine.pools.shift) for fmt in FORMATS)
            res["sample"] = {kind: ctx.compute_shader(_sample_shader(kind, fetch))
                             for kind in (KIND_COLOR, KIND_MR, KIND_HEIGHT, KIND_MASK)}
            res["table"] = ctx.buffer(reserve=self.grid * self.grid * 4)
            words = (self.grid * self.grid + 31) // 32
            res["refs"] = ctx.buffer(reserve=self.grid * words * 4)
            res["vaos"] = self._vaos(res)
            results = {uid: LayerStore(self.grid) for uid in sources}
            for row in range(self.grid):
                self._row(row, res, sources, results)
            self._mips(sources, results)
        finally:
            for _kind, vao, _set in res.pop("vaos", []):
                vao.release()
            for item in res.pop("old_uv_vbos", []):
                item.release()
            for item in res.pop("sample", {}).values():
                item.release()
            for item in res.values():
                item.release()
        self.stats.update(seconds=round(time.perf_counter() - started, 2), layers=len(results),
                          pages=int(sum(store.page_count() for store in results.values())))
        log.info("搬运贴图：%s，%d 层，%d 页，用时 %.2f 秒", self.ts.name, len(results), self.stats["pages"],
                 self.stats["seconds"])
        return results

    def _vaos(self, res) -> list:
        ctx = self.ctx
        vaos = []
        res["old_uv_vbos"] = []
        for set_gpu in self.new_sets:
            known = self.known_uvs.get(id(set_gpu))
            if set_gpu.ibo is not None and known is not None:
                old_vbo = ctx.buffer(np.ascontiguousarray(known, np.float32).tobytes())
                res["old_uv_vbos"].append(old_vbo)
                vaos.append(("main_known", ctx.vertex_array(res["main_known"], [
                    (set_gpu.mesh.vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv"), (old_vbo, "2f", "in_old_uv")],
                    index_buffer=set_gpu.ibo, index_element_size=4), set_gpu))
            elif set_gpu.ibo is not None:
                vaos.append(("main", ctx.vertex_array(res["main"], [(set_gpu.mesh.vbo, "3f 3f 2f", "in_pos", "in_nrm",
                                                                     "in_uv")],
                                                      index_buffer=set_gpu.ibo, index_element_size=4), set_gpu))
            if set_gpu.skirt_vbo is not None:
                vaos.append(("skirt", ctx.vertex_array(res["skirt"], [(set_gpu.skirt_vbo, "2f 3f 3f", "in_uv", "in_pos",
                                                                       "in_nrm")]), set_gpu))
        return vaos

    # ---------------------------------------------------------------- 一行
    def _row(self, row: int, res: dict, sources: dict, results: dict) -> None:
        ctx = self.ctx
        engine = self.engine
        cache = engine.cache
        grid = self.grid
        width = grid * PAGE
        # 1. 光栅化这一行的新几何
        fbo = res["fbo"]
        fbo.use()
        ctx.viewport = (0, 0, width, PAGE)
        fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE | moderngl.BLEND)
        ctx.depth_func = "<"
        drew = False
        for kind, vao, set_gpu in res["vaos"]:
            first, count = (set_gpu.tile_first, set_gpu.tile_count) if kind != "skirt" else \
                (set_gpu.skirt_first, set_gpu.skirt_count)
            ids = np.arange(row * grid, (row + 1) * grid)
            counts = count[ids]
            if counts.sum() == 0:
                continue
            start = int(first[ids][counts > 0].min())
            end = int((first[ids] + counts)[counts > 0].max())
            program = res[kind]
            program["u_origin"] = (0.0, float(row * PAGE))
            program["u_size"] = (float(width), float(PAGE))
            program["u_res"] = float(self.res)
            program["u_depth"] = -0.5 if kind != "skirt" else 0.5
            vao.render(moderngl.TRIANGLES, vertices=end - start, first=start)
            drew = True
        ctx.disable(moderngl.DEPTH_TEST)
        if not drew:
            return
        # 2. 找旧 UV
        program = res["map"]
        res["pos"].bind_to_image(0, read=True, write=False)
        res["nrm"].bind_to_image(1, read=True, write=False)
        res["uv"].bind_to_image(2, read=False, write=True)
        res["known"].bind_to_image(3, read=True, write=False)
        res["nodes"].bind_to_storage_buffer(1)
        res["tri_pos"].bind_to_storage_buffer(2)
        res["tri_uv"].bind_to_storage_buffer(3)
        res["refs"].clear()
        res["refs"].bind_to_storage_buffer(4)
        program["u_grid"] = int(grid)
        program["u_res"] = float(self.res)
        program["u_size"] = (width, PAGE)
        program["u_dist"] = float(self.distance)
        if "u_eps" in program:
            program["u_eps"] = float(self.eps)
        program.run((width + 7) // 8, (PAGE + 7) // 8, 1)
        ctx.memory_barrier()
        words = (grid * grid + 31) // 32
        bits = np.frombuffer(res["refs"].read(), np.uint32).reshape(grid, words)
        refs = np.unpackbits(bits.view(np.uint8), axis=1, bitorder="little")[:, :grid * grid].astype(bool)
        used_columns = np.nonzero(refs.any(axis=1))[0]
        if len(used_columns) == 0:
            return
        # 3. 每个图层的每种页
        for uid, (layer, store) in sources.items():
            for plane, pstore in store.planes.items():
                level0 = pstore.levels[0]
                if not level0.pages:
                    continue
                exists = level0.exists.reshape(-1)
                wanted = refs[used_columns] & exists[None, :]
                columns = used_columns[wanted.any(axis=1)]
                if len(columns) == 0:
                    continue
                needed = np.nonzero(wanted.any(axis=0))[0]
                pages = [level0.pages[(int(t % grid), int(t // grid))] for t in needed]
                cache.make_resident(pages)
                cache.pin_many(pages)
                try:
                    table = np.full(grid * grid, -1, np.int32)
                    table[needed] = [page.slot for page in pages]
                    res["table"].write(table.tobytes())
                    fmt = PLANE_FORMAT[plane]
                    news = cache.try_new_pages(fmt, len(columns), block=True)
                    dst = np.fromiter((page.slot for page in news), np.int32, len(news))
                    self._sample(PLANE_KIND[plane], fmt, dst, columns, res, float(layer.mask_default))
                    cache.note_new_pages(news)
                    target = results[uid].plane(plane).levels[0]
                    for column, page in zip(columns.tolist(), news):
                        cache.set_page(target, int(column), row, page)
                finally:
                    cache.pin_many(pages, -1)

    def _sample(self, kind: int, fmt: str, dst: np.ndarray, columns: np.ndarray, res: dict, mask_default: float) -> None:
        engine = self.engine
        ops = engine.ops
        pool = engine.pools[fmt]
        program = res["sample"][kind]
        unit = 0
        for name in FORMATS:
            unit = bind_samplers(program, engine.pools, name, unit)
        res["uv"].use(unit)
        program["u_uv"] = unit
        res["table"].bind_to_storage_buffer(2)
        program["u_grid"] = int(self.grid)
        program["u_res"] = float(self.res)
        if "u_mask_default" in program:
            program["u_mask_default"] = float(mask_default)
        jobs = np.full((len(dst), 8), -1, np.int32)
        jobs[:, 0] = dst
        jobs[:, 1] = columns
        ops._run_by_array(program, fmt, jobs)

    # ---------------------------------------------------------------- 上一级
    def _mips(self, sources: dict, results: dict) -> None:
        cache = self.engine.cache
        ops = self.engine.ops
        for uid, store in results.items():
            layer = sources[uid][0]
            for plane, pstore in store.planes.items():
                kind = PLANE_KIND[plane]
                fmt = pstore.fmt
                filler = None
                if plane == PLANE_MASK:
                    filler = cache.new_page(fmt, pinned=True)
                    value = float(layer.mask_default)
                    ops.clear(fmt, np.array([filler.slot], np.int32), (value, value, value, value))
                try:
                    for mip in range(1, len(pstore.levels)):
                        below = pstore.levels[mip - 1]
                        here = pstore.levels[mip]
                        if not below.pages:
                            break
                        parents = sorted({(x >> 1, y >> 1) for (x, y) in below.pages})
                        for start in range(0, len(parents), 256):
                            chunk = parents[start:start + 256]
                            children_pages = []
                            children = np.full((len(chunk), 4), -1, np.int32)
                            for i, (x, y) in enumerate(chunk):
                                for q in range(4):
                                    page = below.pages.get((2 * x + (q & 1), 2 * y + (q >> 1)))
                                    if page is not None:
                                        children_pages.append((i, q, page))
                            sources_pages = [item[2] for item in children_pages]
                            cache.make_resident(sources_pages)
                            cache.pin_many(sources_pages)
                            try:
                                for i, q, page in children_pages:
                                    children[i, q] = page.slot
                                news = cache.try_new_pages(fmt, len(chunk), block=True)
                                dst = np.fromiter((page.slot for page in news), np.int32, len(news))
                                old = np.full(len(news), filler.slot if filler is not None else -1, np.int32)
                                ops.downsample(kind, dst, old, children)
                                cache.note_new_pages(news)
                                for (x, y), page in zip(chunk, news):
                                    cache.set_page(here, x, y, page)
                            finally:
                                cache.pin_many(sources_pages, -1)
                finally:
                    if filler is not None:
                        cache.pin_many([filler], -1)
                        cache.free_page(filler)
