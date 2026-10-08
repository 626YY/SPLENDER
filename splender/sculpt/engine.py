"""显卡雕刻引擎（ZBrush 部分）。

网格全部放在显存里：顶点位置（带遮罩）、法线、三角形、顶点到三角形的邻接。顶点按空间顺序每 256 个一块。
每帧把攒下的笔触交给显卡：挑出被碰到的块 → （需要时）求笔触下的平均平面 → 执行笔刷 → 重算法线 → 更新包围盒。
CPU 只上传笔触，不碰顶点。

撤销按块写时复制：一笔里第一次改动某块时先把原值拷进备份区；抬笔时读回改动过的块的前后数据，记一步撤销。
"""
from __future__ import annotations

import logging
import math
import time

import moderngl
import numpy as np

from . import shaders as S
from .topology import SculptMesh, spatial_sort, vertex_faces

log = logging.getLogger("splender.sculpt")

BRUSHES = [("DRAW", "标准", "沿表面法线堆起或挖下"), ("CLAY", "黏土", "像抹黏土一样堆高，表面趋于平整"),
           ("CLAY_STRIPS", "黏土条", "比黏土更硬朗的堆积"), ("SMOOTH", "平滑", "抹平起伏"),
           ("GRAB", "抓取", "抓住一块拖动"), ("INFLATE", "膨胀", "沿各自法线鼓起"), ("PINCH", "捏合", "把表面往笔刷中心收紧"),
           ("FLATTEN", "压平", "压成平面"), ("CREASE", "折痕", "刻出锐利的折痕"), ("MASK", "遮罩", "冻结区域，其他笔刷不改动它")]
BRUSH_INDEX = {name: index for index, (name, _label, _desc) in enumerate(BRUSHES)}
PLANE_BRUSHES = {S.DRAW, S.CLAY, S.CLAY_STRIPS, S.PINCH, S.FLATTEN, S.CREASE}
DAB_DTYPE = np.dtype([("center_radius", "f4", 4), ("normal_strength", "f4", 4), ("params", "f4", 4)])
MAX_DABS = 256
C = S.CLUSTER


class SculptEngine:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.programs = {name: ctx.compute_shader(getattr(S, name)) for name in
                         ("CULL", "ARGS", "PLANE", "APPLY", "SMOOTH_A", "SMOOTH_B", "GRAB_INIT", "GRAB", "NORMALS",
                          "BOUNDS", "GATHER", "SCATTER")}
        from ..engine import ibl

        self.color_program = ctx.program(vertex_shader=S.RENDER_VERT, fragment_shader=S.RENDER_FRAG_HEAD +
                                         ibl.GLSL_STUDIO + ibl.GLSL_MATCAP + S.RENDER_FRAG_BODY)
        self.gbuf_program = ctx.program(vertex_shader=S.RENDER_VERT, fragment_shader=S.GBUF_FRAG)
        self.buffers: dict[str, moderngl.Buffer] = {}
        self.vaos: dict[str, moderngl.VertexArray] = {}
        self.mesh: SculptMesh | None = None
        self.order: np.ndarray | None = None          # 排序后的第 i 个顶点是原来的第 order[i] 个
        self.vertex_count = 0
        self.cluster_count = 0
        self.arena_capacity = 0
        self.margin = 0.0
        self.version = 0                               # 形状每变一次加一（几何缓冲、同步据此判断）
        self.stats = {"load_ms": 0.0, "dab_batches": 0, "last_gpu_ms": 0.0}

    # ------------------------------------------------------------------ 载入与读回
    def load(self, mesh: SculptMesh, arena_fraction: float = 1.0) -> None:
        started = time.perf_counter()
        self.release_buffers()
        sorted_mesh, order = spatial_sort(mesh)
        self.mesh = sorted_mesh
        self.order = order
        v = sorted_mesh.vertex_count
        clusters = max(1, (v + C - 1) // C)
        padded = clusters * C
        pos = np.zeros((padded, 4), np.float32)
        pos[:v, :3] = sorted_mesh.vertices
        if v and padded > v:
            pos[v:, :3] = sorted_mesh.vertices[-1]
        mask = getattr(mesh, "mask", None)
        if mask is not None:
            pos[:v, 3] = np.asarray(mask, np.float32)[order]
        offsets, faces = vertex_faces(padded, sorted_mesh.triangles)
        self.vertex_count = v
        self.cluster_count = clusters
        self.arena_capacity = max(1, min(clusters, int(math.ceil(clusters * max(0.05, arena_fraction)))))
        tri = np.ascontiguousarray(sorted_mesh.triangles.astype(np.int32))
        buf = self.ctx.buffer
        self.buffers = {
            "pos": buf(pos.tobytes()),
            "nrm": buf(reserve=padded * 16),
            "tris": buf(tri.tobytes()),
            "vf_off": buf(offsets.astype(np.int32).tobytes()),
            "vf_idx": buf(faces.astype(np.int32).tobytes() if len(faces) else bytes(4)),
            "bounds": buf(reserve=clusters * 16),
            "work": buf(reserve=clusters * 4),
            "meta": buf(reserve=(8 + clusters) * 4),
            "dabs": buf(reserve=MAX_DABS * DAB_DTYPE.itemsize),
            "cow": buf(reserve=clusters * 4),
            "fresh": buf(reserve=clusters * 4),
            "arena": buf(reserve=self.arena_capacity * C * 16),
            "plane": buf(reserve=MAX_DABS * 8 * 4),
            "scratch": buf(reserve=padded * 16),
            "grab_w": buf(reserve=padded * 4),
        }
        for name in ("cow", "fresh", "meta"):
            self.buffers[name].clear()
        self._reset_meta()
        edges = sorted_mesh.vertices[sorted_mesh.triangles[:, 1]] - sorted_mesh.vertices[sorted_mesh.triangles[:, 0]]
        self.margin = float(np.sqrt((edges.astype(np.float64) ** 2).sum(axis=1)).mean() * 2.5) if len(edges) else 0.0
        self.vaos = {}
        for name, program in (("color", self.color_program), ("gbuf", self.gbuf_program)):
            self.vaos[name] = self.ctx.vertex_array(program, [(self.buffers["pos"], "4f", "in_pos"),
                                                              (self.buffers["nrm"], "4f", "in_nrm")],
                                                    index_buffer=self.buffers["tris"], index_element_size=4)
        self._bind()
        self._run_all("NORMALS")
        self._run_all("BOUNDS")
        self.ctx.memory_barrier()
        self.version += 1
        self.stats["load_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        log.info("雕刻网格载入：%d 个顶点，%d 个三角形，%d 块，用时 %.0f ms", v, sorted_mesh.triangle_count, clusters,
                 self.stats["load_ms"])

    def _reset_meta(self) -> None:
        meta = np.zeros(8, np.uint32)
        meta[2] = self.arena_capacity
        self.buffers["meta"].write(meta.tobytes(), offset=0)

    def read_vertices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读回 (位置 (V,3), 法线 (V,3), 遮罩 (V,))，按载入时的原始顶点顺序。"""
        v = self.vertex_count
        pos = np.frombuffer(self.buffers["pos"].read(size=v * 16), np.float32).reshape(v, 4)
        nrm = np.frombuffer(self.buffers["nrm"].read(size=v * 16), np.float32).reshape(v, 4)
        out_pos = np.empty((v, 3), np.float32)
        out_nrm = np.empty((v, 3), np.float32)
        out_mask = np.empty(v, np.float32)
        out_pos[self.order] = pos[:, :3]
        out_nrm[self.order] = nrm[:, :3]
        out_mask[self.order] = pos[:, 3]
        return out_pos, out_nrm, out_mask

    def bounds(self) -> tuple[np.ndarray, float]:
        data = np.frombuffer(self.buffers["bounds"].read(), np.float32).reshape(-1, 4)
        lo = (data[:, :3] - data[:, 3:4]).min(axis=0)
        hi = (data[:, :3] + data[:, 3:4]).max(axis=0)
        return (lo + hi) * 0.5, float(np.linalg.norm(hi - lo) * 0.5) or 1.0

    # ------------------------------------------------------------------ 调度
    def _bind(self) -> None:
        names = ("pos", "nrm", "tris", "vf_off", "vf_idx", "bounds", "work", "meta", "dabs", "cow", "fresh", "arena",
                 "plane", "scratch", "grab_w")
        for binding, name in enumerate(names):
            self.buffers[name].bind_to_storage_buffer(binding)

    def _setup(self, name: str, all_clusters: bool, dab_count: int = 0, brush: int = 0):
        program = self.programs[name]
        for key, value in (("u_vertex_count", self.vertex_count), ("u_cluster_count", self.cluster_count),
                           ("u_dab_count", dab_count), ("u_type", brush), ("u_all", int(all_clusters))):
            if key in program:
                program[key] = value
        return program

    def _run_all(self, name: str) -> None:
        program = self._setup(name, True)
        program.run(self.cluster_count, 1, 1)
        self.ctx.memory_barrier()

    def _run_work(self, name: str, dab_count: int = 0, brush: int = 0, **uniforms) -> None:
        program = self._setup(name, False, dab_count, brush)
        for key, value in uniforms.items():
            if key in program:
                program[key] = value
        program.run_indirect(self.buffers["meta"], offset=16)
        self.ctx.memory_barrier()

    def _cull(self, dab_count: int, backup: bool, margin: float | None = None) -> None:
        self.buffers["meta"].write(np.zeros(1, np.uint32).tobytes(), offset=0)      # 工作列表清空
        program = self._setup("CULL", False, dab_count)
        program["u_margin"] = float(self.margin if margin is None else margin)
        program["u_backup"] = int(backup)
        program.run((self.cluster_count + 63) // 64, 1, 1)
        self.ctx.memory_barrier()
        self.programs["ARGS"].run(1, 1, 1)
        self.ctx.memory_barrier()

    # ------------------------------------------------------------------ 笔划
    def begin_stroke(self) -> None:
        """一笔开始：清空撤销备份的记录。"""
        self.buffers["cow"].clear()
        self.buffers["fresh"].clear()
        self._reset_meta()

    def apply_dabs(self, dabs: np.ndarray, brush: int) -> None:
        """把一批笔触（DAB_DTYPE）作用到网格上。"""
        if self.mesh is None or len(dabs) == 0:
            return
        started = time.perf_counter()
        self._bind()
        for start in range(0, len(dabs), MAX_DABS):
            part = np.ascontiguousarray(dabs[start:start + MAX_DABS])
            count = len(part)
            self.buffers["dabs"].write(part.tobytes())
            self._cull(count, backup=True)
            if brush in PLANE_BRUSHES:
                self.buffers["plane"].clear()
                self._run_work("PLANE", count, brush)
            if brush == S.SMOOTH:
                self._run_work("SMOOTH_A", count, brush)
                self._run_work("SMOOTH_B", count, brush)
            else:
                self._run_work("APPLY", count, brush)
            if brush != S.MASK:
                self._run_work("NORMALS", count, brush)
            self._run_work("BOUNDS", count, brush)
        self.version += 1
        self.stats["dab_batches"] += 1
        self.stats["last_cpu_ms"] = round((time.perf_counter() - started) * 1000.0, 2)

    def grab_begin(self, dab: np.ndarray) -> None:
        """抓取：记下笔刷范围内每个顶点的权重。之后 grab_move 只改这些顶点。"""
        self._bind()
        self.buffers["dabs"].write(np.ascontiguousarray(dab[:1]).tobytes())
        self._cull(1, backup=True)
        self._run_work("GRAB_INIT", 1, S.GRAB)

    def grab_move(self, delta: np.ndarray) -> None:
        self._bind()
        self._run_work("GRAB", 1, S.GRAB, u_delta=tuple(float(v) for v in delta))
        self._run_work("NORMALS", 1, S.GRAB)
        self._run_work("BOUNDS", 1, S.GRAB)
        self.version += 1

    def end_stroke(self) -> dict | None:
        """一笔结束：读回改过的块的前后数据，返回撤销记录（没改动时返回 None）。"""
        meta = np.frombuffer(self.buffers["meta"].read(size=16), np.uint32)
        touched = int(min(meta[3], self.arena_capacity))
        if touched == 0:
            return None
        self._bind()
        program = self._setup("GATHER", False)
        program.run(touched, 1, 1)
        self.ctx.memory_barrier()
        clusters = np.frombuffer(self.buffers["meta"].read(size=touched * 4, offset=32), np.int32).copy()
        cow = np.frombuffer(self.buffers["cow"].read(), np.int32)
        slots = cow[clusters] - 1
        used = int(slots.max()) + 1
        before = np.frombuffer(self.buffers["arena"].read(size=used * C * 16), np.float32).reshape(used, C, 4)[slots]
        after = np.frombuffer(self.buffers["scratch"].read(size=used * C * 16), np.float32).reshape(used, C, 4)[slots]
        return {"clusters": clusters, "before": np.ascontiguousarray(before), "after": np.ascontiguousarray(after),
                "overflow": bool(meta[1] > self.arena_capacity)}

    def restore(self, clusters: np.ndarray, data: np.ndarray) -> None:
        """把这些块的数据写回（撤销、重做）。data 形状 (块数, 256, 4)。"""
        if len(clusters) == 0:
            return
        self._bind()
        self.buffers["work"].write(np.ascontiguousarray(clusters.astype(np.int32)).tobytes())
        self.buffers["scratch"].write(np.ascontiguousarray(data.astype(np.float32)).tobytes())
        program = self._setup("SCATTER", False)
        program.run(len(clusters), 1, 1)
        self.ctx.memory_barrier()
        self._run_all("NORMALS")
        self._run_all("BOUNDS")
        self.version += 1

    # ------------------------------------------------------------------ 绘制
    def draw(self, view_proj: np.ndarray, view_matrix: np.ndarray, eye, *, solid: int, tint, matcap_tex=None,
             studio=None, cursor=None, pixel_world: float = 0.001, mask_dim: float = 0.35) -> None:
        """画进当前绑定的帧缓冲（深度测试由调用方打开）。"""
        if self.mesh is None:
            return
        from ..engine import ibl

        program = self.color_program
        program["u_view_proj"].write(view_proj.astype("f4").T.tobytes())
        program["u_view"].write(view_matrix.astype("f4").T.tobytes())
        program["u_eye"] = tuple(float(v) for v in eye)
        program["u_solid"] = int(solid)
        program["u_tint"] = tuple(float(v) for v in tint)
        if "u_mask_dim" in program:
            program["u_mask_dim"] = float(mask_dim)
        if "u_pixel_world" in program:
            program["u_pixel_world"] = float(pixel_world)
        if cursor is not None:
            position, _normal, radius, hardness = cursor
            program["u_cursor"] = (float(position[0]), float(position[1]), float(position[2]), float(radius))
            program["u_cursor_hardness"] = float(hardness)
            program["u_cursor_on"] = 1
        else:
            program["u_cursor_on"] = 0
        if matcap_tex is not None:
            ibl.bind_matcap(program, matcap_tex, 0)
        if studio is not None:
            ibl.bind_studio(program, studio)
        self.vaos["color"].render(moderngl.TRIANGLES)

    def draw_gbuffer(self, view_proj: np.ndarray) -> None:
        if self.mesh is None:
            return
        self.gbuf_program["u_view_proj"].write(view_proj.astype("f4").T.tobytes())
        self.vaos["gbuf"].render(moderngl.TRIANGLES)

    # ------------------------------------------------------------------ 释放
    def release_buffers(self) -> None:
        for vao in self.vaos.values():
            vao.release()
        self.vaos = {}
        for buffer in self.buffers.values():
            buffer.release()
        self.buffers = {}
        self.mesh = None

    def release(self) -> None:
        self.release_buffers()
        for program in list(self.programs.values()) + [self.color_program, self.gbuf_program]:
            program.release()
