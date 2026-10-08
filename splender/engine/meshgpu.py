"""模型在显卡上的数据：视口绘制用的顶点缓冲，和每个纹理集按格分桶的绘制几何。"""
from __future__ import annotations

import moderngl
import numpy as np


def _vertex_order(first: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """每个格子的顶点范围 (起点, 个数) 展开成顶点下标序列。"""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64)
    ends = np.cumsum(counts)
    within = np.arange(total) - np.repeat(ends - counts, counts)
    return np.repeat(first, counts) + within


def mesh_layout(program, vbo):
    """按着色器实际用到的输入拼顶点格式（没用到的输入会被编译器去掉，要跳过对应字节）。"""
    fmt = []
    names = []
    for name, count in (("in_pos", 3), ("in_nrm", 3), ("in_uv", 2)):
        if name in program:
            fmt.append("%df" % count)
            names.append(name)
        else:
            fmt.append("%dx" % (count * 4))
    return (vbo, " ".join(fmt), *names)


class MeshGPU:
    """一个模型。顶点是逐角点（位置 3、法线 3、UV 2），按材质各有一份索引。"""

    def __init__(self, ctx: moderngl.Context, mesh) -> None:
        self.ctx = ctx
        self.mesh = mesh
        data = np.concatenate([mesh.positions, mesh.normals, mesh.uvs], axis=1).astype("f4")
        self.vbo = ctx.buffer(np.ascontiguousarray(data).tobytes())
        self.corner_count = len(mesh.positions)
        self.material_ibo: list[moderngl.Buffer | None] = []
        self.material_count: list[int] = []
        ids = np.asarray(mesh.material_ids, np.int32)
        for index in range(max(1, len(mesh.materials))):
            tris = np.nonzero(ids == index)[0].astype(np.uint32)
            if len(tris) == 0:
                self.material_ibo.append(None)
                self.material_count.append(0)
                continue
            corners = (tris[:, None] * 3 + np.arange(3, dtype=np.uint32)[None, :]).ravel()
            self.material_ibo.append(ctx.buffer(corners.astype("u4").tobytes()))
            self.material_count.append(int(len(corners)))
        self.bounds_min = np.asarray(mesh.bounds_min, np.float32)
        self.bounds_max = np.asarray(mesh.bounds_max, np.float32)

    @property
    def uv_edge(self) -> float:
        """较短那部分 UV 边的长度（20% 分位数，UV 单位）。线框看起来密不密由短边决定，UV 视图据此在缩小时淡出线框。"""
        found = getattr(self, "_uv_edge", None)
        if found is None:
            uv = np.asarray(self.mesh.uvs, np.float32).reshape(-1, 3, 2)
            if len(uv) > 200000:
                uv = uv[:: len(uv) // 200000 + 1]
            edges = np.linalg.norm(uv[:, [1, 2, 0]] - uv, axis=2)
            found = float(np.percentile(edges, 20)) if edges.size else 0.0
            self._uv_edge = found
        return found

    def update_vertices(self, mesh) -> None:
        """顶点位置、法线改了（拓扑不变）：重传顶点缓冲。"""
        data = np.empty((len(mesh.positions), 8), np.float32)
        data[:, 0:3] = mesh.positions
        data[:, 3:6] = mesh.normals
        data[:, 6:8] = mesh.uvs
        self.vbo.write(data)
        self.bounds_min = np.asarray(mesh.bounds_min, np.float32)
        self.bounds_max = np.asarray(mesh.bounds_max, np.float32)

    def edge_buffer(self):
        """线框用的边：逐角点顶点缓冲里的下标对（GL_LINES）。位置重合的点算一个；四边形、多边形内部的对角线不画。
        第一次要用时才算，之后缓存（变换不改拓扑）。"""
        found = getattr(self, "_edge_ibo", None)
        if found is None:
            pairs = edge_pairs(self.mesh.positions, getattr(self.mesh, "polygon_sizes", None))
            self._edge_count = int(pairs.size)
            found = self.ctx.buffer(np.ascontiguousarray(pairs, np.uint32).tobytes()) if pairs.size else None
            self._edge_ibo = found if found is not None else False
        return found or None

    @property
    def edge_count(self) -> int:
        self.edge_buffer()
        return int(getattr(self, "_edge_count", 0))

    @property
    def center(self) -> np.ndarray:
        return (self.bounds_min + self.bounds_max) * 0.5

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self.bounds_max - self.bounds_min) * 0.5) or 1.0

    def release(self) -> None:
        self.vbo.release()
        for ibo in self.material_ibo:
            if ibo is not None:
                ibo.release()
        edge = getattr(self, "_edge_ibo", None)
        if edge:
            edge.release()


def edge_pairs(positions: np.ndarray, polygon_sizes=None) -> np.ndarray:
    """逐角点三角形 → 要画的边 [(角点 a, 角点 b), ...]。

    多边形按扇形拆成三角形（(v0,v1,v2)(v0,v2,v3)…）时，扇形内部的对角线不算边。位置完全相同的点焊成一个，
    重复的边只留一条。"""
    from ..sculpt.topology import weld

    corners = len(positions)
    tris = corners // 3
    if tris == 0:
        return np.zeros((0, 2), np.uint32)
    keep = np.ones((tris, 3), bool)            # 每个三角形的三条边：0=(c0,c1) 1=(c1,c2) 2=(c2,c0)
    if polygon_sizes is not None and len(polygon_sizes):
        sizes = np.asarray(polygon_sizes, np.int64)
        counts = np.maximum(sizes - 2, 1)
        if int(counts.sum()) == tris:
            starts = np.repeat(np.cumsum(counts) - counts, counts)
            k = np.arange(tris) - starts                    # 三角形在自己多边形里的序号
            last = np.repeat(counts - 1, counts)
            keep[:, 0] = k == 0
            keep[:, 2] = k == last
    ids = weld(positions, None, None, tolerance=1e-6, drop_degenerate=False).triangles.reshape(-1).astype(np.int64)
    if polygon_sizes is None:
        keep &= ~quad_diagonals(positions, ids)
    base = np.arange(tris, dtype=np.int64) * 3
    a = np.stack([base, base + 1, base + 2], axis=1)
    b = np.stack([base + 1, base + 2, base], axis=1)
    a = a[keep]
    b = b[keep]
    wa, wb = ids[a], ids[b]
    lo, hi = np.minimum(wa, wb), np.maximum(wa, wb)
    valid = lo != hi
    key = lo[valid] * (int(ids.max()) + 1) + hi[valid]
    _unique, first = np.unique(key, return_index=True)
    out = np.stack([a[valid][first], b[valid][first]], axis=1)
    return out.astype(np.uint32)


def quad_diagonals(positions: np.ndarray, ids: np.ndarray, min_share: float = 0.9,
                   max_angle: float = 3.0) -> np.ndarray:
    """三角化过的四边形网格里的对角线：[三角形数, 3] 布尔，True 表示这条边是对角线（线框里不画）。

    一条边是两个三角形各自最长的那条边、两个面几乎共面，就当它是拆四边形时加的对角线。只有大多数三角形
    （min_share）都能这样两两配上时才认为原来是四边形网格，否则（雕刻、重构出来的三角网格）一条都不去掉。"""
    tris = len(positions) // 3
    out = np.zeros((tris, 3), bool)
    if tris < 2:
        return out
    p = np.asarray(positions, np.float64).reshape(-1, 3, 3)
    edge_vec = p[:, [1, 2, 0]] - p                       # 边 k = (角 k, 角 k+1)
    length = np.einsum("tki,tki->tk", edge_vec, edge_vec)
    longest = np.argmax(length, axis=1)
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-30)
    w = ids.reshape(-1, 3)
    a = w[np.arange(tris), longest]
    b = w[np.arange(tris), (longest + 1) % 3]
    key = np.minimum(a, b) * (int(ids.max()) + 1) + np.maximum(a, b)
    order = np.argsort(key, kind="stable")
    sk = key[order]
    same = np.nonzero(sk[1:] == sk[:-1])[0]
    if len(same) == 0:
        return out
    t1, t2 = order[same], order[same + 1]
    # 一条边只能配一对（三个三角形共用的边不算）
    triple = np.zeros(len(sk), bool)
    triple[1:-1] = (sk[1:-1] == sk[:-2]) & (sk[1:-1] == sk[2:])
    bad = triple[same] | triple[np.minimum(same + 1, len(sk) - 1)]
    coplanar = np.einsum("ti,ti->t", normal[t1], normal[t2]) >= np.cos(np.radians(max_angle))
    good = coplanar & ~bad
    if 2 * int(good.sum()) < min_share * tris:
        return out
    out[t1[good], longest[t1[good]]] = True
    out[t2[good], longest[t2[good]]] = True
    return out


def _transform_boxes(lo: np.ndarray, hi: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """一组包围盒（空盒是 +inf/-inf）按 4×4 变换后的新包围盒。"""
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    if len(lo) == 0:
        return lo.astype(np.float32), hi.astype(np.float32)
    valid = np.all(np.isfinite(lo), axis=1) & np.all(np.isfinite(hi), axis=1)
    rot = delta[:3, :3]
    with np.errstate(invalid="ignore", over="ignore"):       # 空盒（±inf）算出来的 nan 下面会被原值换回
        center = (lo + hi) * 0.5
        half = (hi - lo) * 0.5
        new_center = center @ rot.T + delta[:3, 3]
        new_half = half @ np.abs(rot).T
    out_lo = np.where(valid[:, None], new_center - new_half, lo)
    out_hi = np.where(valid[:, None], new_center + new_half, hi)
    return out_lo.astype(np.float32), out_hi.astype(np.float32)


class SetGPU:
    """一个模型的一个材质在某个纹理集分辨率下的绘制几何（见 meshprep.SetGeometry）。"""

    def __init__(self, ctx: moderngl.Context, mesh_gpu: MeshGPU, geometry) -> None:
        self.ctx = ctx
        self.mesh = mesh_gpu
        self.geometry = geometry
        self.grid = int(geometry.grid)
        self.resolution = int(geometry.resolution)
        indices = np.ascontiguousarray(geometry.indices, np.uint32)
        self.ibo = ctx.buffer(indices.tobytes()) if len(indices) else None
        skirt = np.ascontiguousarray(geometry.skirt_vertices, np.float32)
        self.skirt_vbo = ctx.buffer(skirt.tobytes()) if len(skirt) else None
        self.tile_first = np.asarray(geometry.tile_first, np.int64)
        self.tile_count = np.asarray(geometry.tile_count, np.int64)
        self.skirt_first = np.asarray(geometry.skirt_first, np.int64)
        self.skirt_count = np.asarray(geometry.skirt_count, np.int64)
        # 包围盒并入扩边四边形的位置：只有扩边的格子也要能被笔触选中
        tile_min = np.array(geometry.tile_min, np.float32, copy=True)
        tile_max = np.array(geometry.tile_max, np.float32, copy=True)
        if len(skirt):
            owner = np.repeat(np.arange(len(self.skirt_count)), self.skirt_count)
            positions = skirt[_vertex_order(self.skirt_first, self.skirt_count), 2:5]
            np.minimum.at(tile_min, owner, positions)
            np.maximum.at(tile_max, owner, positions)
        # 只保留有几何的格子，剔除时少算很多
        self.tile_ids = np.nonzero((self.tile_count > 0) | (self.skirt_count > 0))[0].astype(np.int64)
        self.tile_min = np.ascontiguousarray(tile_min[self.tile_ids])
        self.tile_max = np.ascontiguousarray(tile_max[self.tile_ids])

    def apply_transform(self, delta: np.ndarray) -> None:
        """模型整体变换了（拓扑、UV 不变）：扩边顶点跟着变，每格的包围盒按变换后的八个角重新框（偏大一点，不会漏）。"""
        delta = np.asarray(delta, np.float64)
        rot = delta[:3, :3]
        skirt = np.array(self.geometry.skirt_vertices, np.float32, copy=True)
        if len(skirt):
            if np.allclose(rot, np.identity(3), atol=1e-12):
                skirt[:, 2:5] += delta[:3, 3].astype(np.float32)
            else:
                skirt[:, 2:5] = skirt[:, 2:5] @ rot.T.astype(np.float32) + delta[:3, 3].astype(np.float32)
                n = skirt[:, 5:8] @ np.linalg.inv(rot).astype(np.float32)
                skirt[:, 5:8] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
            self.geometry.skirt_vertices = skirt
            if self.skirt_vbo is not None:
                self.skirt_vbo.write(np.ascontiguousarray(self.geometry.skirt_vertices).tobytes())
        self.geometry.tile_min, self.geometry.tile_max = _transform_boxes(self.geometry.tile_min,
                                                                          self.geometry.tile_max, delta)
        self.tile_min, self.tile_max = _transform_boxes(self.tile_min, self.tile_max, delta)

    def tiles_for_dabs(self, centers: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """哪些笔触碰到哪些格子。返回 (笔触序号数组, 格子编号数组)，一一对应。"""
        if len(self.tile_ids) == 0 or len(centers) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        # 先用这一批笔触的总包围盒粗筛格子
        lo = (centers - radii[:, None]).min(axis=0)
        hi = (centers + radii[:, None]).max(axis=0)
        near = np.nonzero(np.all(self.tile_max >= lo, axis=1) & np.all(self.tile_min <= hi, axis=1))[0]
        if len(near) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        tmin = self.tile_min[near]
        tmax = self.tile_max[near]
        c = centers[:, None, :]
        gap = np.maximum(tmin[None] - c, 0.0) + np.maximum(c - tmax[None], 0.0)
        hit = np.einsum("dtk,dtk->dt", gap, gap) <= (radii * radii)[:, None]
        dab_index, local = np.nonzero(hit)
        return dab_index.astype(np.int64), self.tile_ids[near[local]]

    def tiles_for_uv_dabs(self, centers: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """UV 视图里的笔触（中心、半径按 UV）碰到哪些有几何的格子。返回 (笔触序号数组, 格子编号数组)。"""
        empty = (np.empty(0, np.int64), np.empty(0, np.int64))
        if len(self.tile_ids) == 0 or len(centers) == 0:
            return empty
        grid = self.grid
        has = getattr(self, "_has_tile", None)
        if has is None:
            has = np.zeros(grid * grid, bool)
            has[self.tile_ids] = True
            self._has_tile = has
        lo = np.floor((centers - radii[:, None]) * grid).astype(np.int64)
        hi = np.floor((centers + radii[:, None]) * grid).astype(np.int64)
        inside = np.all(hi >= 0, axis=1) & np.all(lo < grid, axis=1)
        lo = np.clip(lo, 0, grid - 1)
        hi = np.clip(hi, 0, grid - 1)
        dab_parts, tile_parts = [], []
        for d in np.nonzero(inside)[0]:
            xs = np.arange(lo[d, 0], hi[d, 0] + 1)
            ys = np.arange(lo[d, 1], hi[d, 1] + 1)
            tiles = (ys[:, None] * grid + xs[None, :]).ravel()
            tiles = tiles[has[tiles]]
            if len(tiles):
                dab_parts.append(np.full(len(tiles), d, np.int64))
                tile_parts.append(tiles)
        if not tile_parts:
            return empty
        return np.concatenate(dab_parts), np.concatenate(tile_parts)

    def release(self) -> None:
        if self.ibo is not None:
            self.ibo.release()
        if self.skirt_vbo is not None:
            self.skirt_vbo.release()
