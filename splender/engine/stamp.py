"""盖章：把一帧里的所有笔触一次画到笔划页上。

做法是把模型三角形按 UV 光栅化到 256×256 的页里（一个多重间接绘制调用覆盖所有页），
每个纹素直接拿到自己的三维位置和法线，再按与笔触中心的距离累积覆盖度。
岛内三角形画完后再画一圈扩边四边形，让 UV 接缝外侧也有数据，贴图过滤时不露边。
在 UV 视图里画的笔触（params.z = 1）中心是 UV 坐标、半径按 UV 算，按纹素自己的 UV 量距离。
有选区时每格的选区页槽号放在任务的第 7 个数（in_job1.z，-1 是没有页、按 u_sel_default），
覆盖度往选区值那里累积（a += (选区 − a) × 这一下），结果正好是「整笔的覆盖度 × 选区」：
画笔、橡皮、效果笔刷的实时样子和抬笔后的合并都只在选区里，选区外的格也不会被标成画过。
"""
from __future__ import annotations

import moderngl
import numpy as np

from .cache import PageCache, PlaneStore
from .meshgpu import SetGPU
from .pageops import KIND_STROKE, PageOps
from .pagepool import MAX_ARRAYS, PAGE, PoolSet, bind_samplers, glsl_fetch

DAB_DTYPE = np.dtype([("center_radius", "f4", 4), ("normal_hardness", "f4", 4), ("params", "f4", 4)])
MAX_DABS = 8192
MAX_PAIRS = 1 << 20
MAX_JOBS = 16384

_VERT_COMMON = """
uniform float u_res;
in ivec4 in_job0;      // 格 x, 格 y, 数组里的层号, 笔触索引起点
in ivec4 in_job1;      // 笔触个数, 格编号, 选区页槽号（-1 是没有页）, 0
out vec3 v_pos; out vec3 v_nrm; out vec2 v_uv;
flat out ivec4 v_job0; flat out ivec4 v_job1;
void emit(vec2 uv, vec3 pos, vec3 nrm){
  vec2 t = (uv * u_res - vec2(in_job0.xy) * 256.0) / 256.0;
  gl_Position = vec4(t * 2.0 - 1.0, 0.0, 1.0);
  v_pos = pos; v_nrm = nrm; v_uv = uv; v_job0 = in_job0; v_job1 = in_job1;
}
"""

_FILL_VERT = "#version 430\nin vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;" + _VERT_COMMON + """
void main(){ emit(in_uv, in_pos, in_nrm); }
"""

_SKIRT_VERT = "#version 430\nin vec2 in_uv; in vec3 in_pos; in vec3 in_nrm;" + _VERT_COMMON + """
void main(){ emit(in_uv, in_pos, in_nrm); }
"""

_FRAG = """#version 430
layout(rg16f, binding=0) uniform image2DArray u_stroke;
struct Dab { vec4 center_radius; vec4 normal_hardness; vec4 params; };
layout(std430, binding=1) readonly buffer Dabs { Dab dabs[]; };
layout(std430, binding=2) readonly buffer DabIndex { int dab_index[]; };
layout(std430, binding=3) buffer Touched { uint touched[]; };
uniform float u_batch;     // 本批的编号，用来让扩边不覆盖岛内纹素
uniform int u_skirt;
uniform float u_res;
uniform int u_sel_on; uniform float u_sel_default;
{sel_fetch}
in vec3 v_pos; in vec3 v_nrm; in vec2 v_uv;
flat in ivec4 v_job0; flat in ivec4 v_job1;
void main(){
  ivec3 texel = ivec3(ivec2(gl_FragCoord.xy), v_job0.z);
  vec2 cur = imageLoad(u_stroke, texel).rg;
  if (u_skirt == 1 && cur.g == u_batch) return;
  float a = cur.r;
  vec3 n = normalize(v_nrm);
  float texel_world = length(fwidth(v_pos));
  float sel = 1.0;
  if (u_sel_on == 1) sel = v_job1.z >= 0 ? clamp(fetch_r8(v_job1.z, ivec2(gl_FragCoord.xy)).r, 0.0, 1.0) : u_sel_default;
  if (sel <= 0.0) return;
  for (int i = 0; i < v_job1.x; i++) {
    Dab d = dabs[dab_index[v_job0.w + i]];
    float radius = d.center_radius.w;
    bool flat_dab = d.params.z > 0.5;                    // UV 视图里画的：按 UV 量距离
    float dist = flat_dab ? length(v_uv - d.center_radius.xy) / radius
                          : length(v_pos - d.center_radius.xyz) / radius;
    if (dist >= 1.0) continue;
    float limit = d.params.y;                            // 法线夹角余弦下限，小于 -1 表示不限制
    float facing = (flat_dab || limit < -1.0) ? 1.0 : smoothstep(limit, limit + 0.12, dot(n, d.normal_hardness.xyz));
    float texel = flat_dab ? 1.0 / u_res : texel_world;
    float soft = max(1.0 - d.normal_hardness.w, min(0.5, texel / radius * 1.5));
    float fall = 1.0 - smoothstep(1.0 - soft, 1.0, dist);
    a += (sel - a) * clamp(d.params.x * fall * facing, 0.0, 1.0);
  }
  imageStore(u_stroke, texel, vec4(a, u_batch, 0.0, 0.0));
  if (a > cur.r) touched[v_job1.y] = 1u;
}
"""


class StrokeBuffer:
    """一个纹理集上进行中的笔划：稀疏的笔划页（含各级）和「哪些格子真的被画到」的标记。"""

    def __init__(self, ctx: moderngl.Context, grid: int) -> None:
        self.grid = grid
        self.store = PlaneStore("rg16f", grid)
        self.touched = ctx.buffer(reserve=grid * grid * 4)
        self.touched.write(bytes(grid * grid * 4))
        self.dirty_tiles: set[int] = set()          # 用过的 L0 格子编号

    def touched_tiles(self) -> np.ndarray:
        flags = np.frombuffer(self.touched.read(), np.uint32)
        return np.nonzero(flags)[0].astype(np.int64)

    def release(self, cache: PageCache) -> None:
        for level in self.store.levels:
            for page in list(level.pages.values()):
                cache.unpin(page)
                cache.free_page(page)
            level.pages.clear()
            level.slots[:] = -1
            level.exists[:] = False
        self.touched.release()


class Stamper:
    def __init__(self, ctx: moderngl.Context, pools: PoolSet, cache: PageCache, ops: PageOps) -> None:
        self.ctx = ctx
        self.pools = pools
        self.cache = cache
        self.ops = ops
        frag = _FRAG.replace("{sel_fetch}", glsl_fetch("r8", MAX_ARRAYS["r8"], pools.shift))
        self.fill = ctx.program(vertex_shader=_FILL_VERT, fragment_shader=frag)
        self.skirt = ctx.program(vertex_shader=_SKIRT_VERT, fragment_shader=frag)
        self.dab_buffer = ctx.buffer(reserve=DAB_DTYPE.itemsize * MAX_DABS)
        self.index_buffer = ctx.buffer(reserve=4 * MAX_PAIRS)
        self.job_buffer = ctx.buffer(reserve=32 * MAX_JOBS)
        self.cmd_buffer = ctx.buffer(reserve=20 * MAX_JOBS)
        self.skirt_cmd_buffer = ctx.buffer(reserve=20 * MAX_JOBS)
        self._target_tex = ctx.texture((PAGE, PAGE), 1, dtype="f1")
        self._target = ctx.framebuffer(color_attachments=[self._target_tex])
        self._vaos: dict[int, tuple] = {}
        self.batch = 0
        self.stat_pairs = 0
        self.stat_jobs = 0

    def _vao(self, set_gpu: SetGPU):
        key = id(set_gpu)
        found = self._vaos.get(key)
        if found is None:
            fill = None
            if set_gpu.ibo is not None:
                fill = self.ctx.vertex_array(self.fill, [(set_gpu.mesh.vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv"),
                                                         (self.job_buffer, "4i 4i /i", "in_job0", "in_job1")],
                                             index_buffer=set_gpu.ibo, index_element_size=4)
            skirt = None
            if set_gpu.skirt_vbo is not None:
                skirt = self.ctx.vertex_array(self.skirt, [(set_gpu.skirt_vbo, "2f 3f 3f", "in_uv", "in_pos", "in_nrm"),
                                                           (self.job_buffer, "4i 4i /i", "in_job0", "in_job1")])
            found = (fill, skirt, set_gpu)
            self._vaos[key] = found
        return found

    def forget(self, set_gpu: SetGPU) -> None:
        found = self._vaos.pop(id(set_gpu), None)
        if found:
            for vao in found[:2]:
                if vao is not None:
                    vao.release()

    # ------------------------------------------------------------------
    def stamp(self, set_gpu: SetGPU, stroke: StrokeBuffer, dabs: np.ndarray, dilate: bool = True,
              selection=None) -> np.ndarray:
        """把 dabs 盖到 stroke 上。返回这次涉及的 L0 格子编号（已去重）。selection 给了就只盖在选区里。"""
        if len(dabs) == 0:
            return np.empty(0, np.int64)
        dabs = dabs[:MAX_DABS]
        centers = np.ascontiguousarray(dabs["center_radius"][:, :3])
        radii = np.ascontiguousarray(dabs["center_radius"][:, 3])
        if dabs["params"][0, 2] > 0.5:                   # 一笔里的笔触要么都在 UV 空间，要么都在三维空间
            dab_index, tile_index = set_gpu.tiles_for_uv_dabs(np.ascontiguousarray(centers[:, :2]), radii)
        else:
            dab_index, tile_index = set_gpu.tiles_for_dabs(centers, radii)
        if len(tile_index) == 0:
            return np.empty(0, np.int64)
        if len(tile_index) > MAX_PAIRS:
            dab_index, tile_index = dab_index[:MAX_PAIRS], tile_index[:MAX_PAIRS]
        order = np.argsort(tile_index, kind="stable")
        tile_sorted = tile_index[order]
        dab_sorted = dab_index[order].astype(np.int32)
        tiles, starts, counts = np.unique(tile_sorted, return_index=True, return_counts=True)
        if len(tiles) > MAX_JOBS:
            tiles, starts, counts = tiles[:MAX_JOBS], starts[:MAX_JOBS], counts[:MAX_JOBS]
        grid = set_gpu.grid
        tx = (tiles % grid).astype(np.int32)
        ty = (tiles // grid).astype(np.int32)
        if selection is not None and selection.default <= 0.0:
            # 选区外（没有选区页、按 0 算）的格不用盖，也不用给它们开笔划页
            inside = selection.exists_at(tx, ty)
            if not inside.any():
                return np.empty(0, np.int64)
            tiles, starts, counts, tx, ty = tiles[inside], starts[inside], counts[inside], tx[inside], ty[inside]
        level0 = stroke.store.levels[0]
        slots = level0.slots[ty, tx]
        missing = np.nonzero(slots < 0)[0]
        if len(missing):
            pages = self.cache.try_new_pages("rg16f", len(missing), block=True)
            self.cache.pin_many(pages)
            for page, j in zip(pages, missing.tolist()):
                self.cache.set_page(level0, int(tx[j]), int(ty[j]), page)
            self.ops.clear("rg16f", np.fromiter((p.slot for p in pages), np.int32, len(pages)))
            slots = level0.slots[ty, tx]
        pool = self.pools["rg16f"]
        pool.touch(slots, self.cache.frame)
        stroke.dirty_tiles.update(tiles.tolist())

        # 按笔划页所在的数组分组，每组一次绘制
        arrays = slots >> pool.shift
        group = np.argsort(arrays, kind="stable")
        tiles, starts, counts, tx, ty, slots, arrays = (tiles[group], starts[group], counts[group], tx[group],
                                                       ty[group], slots[group], arrays[group])
        njobs = len(tiles)
        jobs = np.zeros((njobs, 8), np.int32)
        jobs[:, 0] = tx
        jobs[:, 1] = ty
        jobs[:, 2] = slots & (pool.layers - 1)
        jobs[:, 3] = starts
        jobs[:, 4] = counts
        jobs[:, 5] = tiles
        selected = []
        if selection is not None:
            jobs[:, 6], selected = selection.slots_for(tx, ty)
        job_index = np.arange(njobs, dtype=np.uint32)
        cmds = np.zeros((njobs, 5), np.uint32)
        cmds[:, 0] = set_gpu.tile_count[tiles]
        cmds[:, 1] = 1
        cmds[:, 2] = set_gpu.tile_first[tiles]
        cmds[:, 4] = job_index
        skirt_cmds = np.zeros((njobs, 5), np.uint32)
        skirt_cmds[:, 0] = set_gpu.skirt_count[tiles]
        skirt_cmds[:, 1] = 1
        skirt_cmds[:, 2] = set_gpu.skirt_first[tiles]
        skirt_cmds[:, 3] = job_index

        self.dab_buffer.write(dabs.tobytes())
        self.index_buffer.write(dab_sorted.tobytes())
        self.job_buffer.write(jobs.tobytes())
        self.cmd_buffer.write(cmds.tobytes())
        self.skirt_cmd_buffer.write(skirt_cmds.tobytes())

        self.batch = self.batch % 2000 + 1
        fill_vao, skirt_vao, _ = self._vao(set_gpu)
        ctx = self.ctx
        self._target.use()
        ctx.viewport = (0, 0, PAGE, PAGE)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self.dab_buffer.bind_to_storage_buffer(1)
        self.index_buffer.bind_to_storage_buffer(2)
        stroke.touched.bind_to_storage_buffer(3)
        bounds = np.flatnonzero(np.r_[True, arrays[1:] != arrays[:-1]])
        ends = np.r_[bounds[1:], njobs]
        for program in (self.fill, self.skirt):
            program["u_res"] = float(set_gpu.resolution)
            program["u_batch"] = float(self.batch)
            if "u_sel_on" in program:
                program["u_sel_on"] = 1 if selection is not None else 0
                program["u_sel_default"] = float(selection.default) if selection is not None else 1.0
            if selection is not None:
                bind_samplers(program, self.pools, "r8", 0)
        for start, end in zip(bounds, ends):
            pool.arrays[int(arrays[start])].bind_to_image(0, read=True, write=True)
            if fill_vao is not None:
                self.fill["u_skirt"] = 0
                fill_vao.render_indirect(self.cmd_buffer, moderngl.TRIANGLES, int(end - start), first=int(start))
                ctx.memory_barrier()
            if dilate and skirt_vao is not None:
                self.skirt["u_skirt"] = 1
                skirt_vao.render_indirect(self.skirt_cmd_buffer, moderngl.TRIANGLES, int(end - start), first=int(start))
                ctx.memory_barrier()
        if selection is not None:
            selection.unpin(selected)
        self.stat_pairs = len(tile_sorted)
        self.stat_jobs = njobs
        return tiles

    # ------------------------------------------------------------------
    def update_mips(self, stroke: StrokeBuffer, tiles: np.ndarray) -> list[np.ndarray]:
        """把刚盖过的 L0 格子逐级缩小到上面各级。返回每一级受影响的格子编号（下标 0 是 L0）。"""
        levels = stroke.store.levels
        affected = [np.asarray(tiles, np.int64)]
        child_tiles = affected[0]
        for mip in range(1, len(levels)):
            below = levels[mip - 1]
            here = levels[mip]
            cx = child_tiles % below.size
            cy = child_tiles // below.size
            parent = (cy >> 1) * here.size + (cx >> 1)
            parents, inverse = np.unique(parent, return_inverse=True)
            px = (parents % here.size).astype(np.int32)
            py = (parents // here.size).astype(np.int32)
            slots = here.slots[py, px]
            missing = np.nonzero(slots < 0)[0]
            if len(missing):
                pages = self.cache.try_new_pages("rg16f", len(missing), block=True)
                self.cache.pin_many(pages)
                for page, j in zip(pages, missing.tolist()):
                    self.cache.set_page(here, int(px[j]), int(py[j]), page)
                self.ops.clear("rg16f", np.fromiter((p.slot for p in pages), np.int32, len(pages)))
                slots = here.slots[py, px]
            children = np.full((len(parents), 4), -1, np.int32)
            quadrant = (cx & 1) + ((cy & 1) << 1)
            children[inverse, quadrant] = below.slots[cy, cx]
            self.ops.downsample(KIND_STROKE, slots, slots, children)
            affected.append(parents)
            child_tiles = parents
        return affected

    def release(self) -> None:
        for key in list(self._vaos):
            found = self._vaos.pop(key)
            for vao in found[:2]:
                if vao is not None:
                    vao.release()
        for item in (self.dab_buffer, self.index_buffer, self.job_buffer, self.cmd_buffer, self.skirt_cmd_buffer,
                     self._target, self._target_tex, self.fill, self.skirt):
            item.release()
