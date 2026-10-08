"""雕刻网格的拓扑工具：焊接、法线、邻接、空间排序、细分、默认底模。全部 numpy 进 numpy 出。

约定：
- SculptMesh 是索引网格：vertices (V,3) float32，triangles (T,3) int32，逆时针为正面。
- corner_uvs (T*3, 2) 是逐角点 UV（UV 接缝处同一个顶点在不同三角形里 UV 不同），没有 UV 时为 None。
- 绘制引擎用的 MeshData 是逐角点排列；weld / unweld 在两者之间转换。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange


@dataclass
class SculptMesh:
    vertices: np.ndarray                     # (V, 3) float32
    triangles: np.ndarray                    # (T, 3) int32
    corner_uvs: np.ndarray | None = None     # (T*3, 2) float32
    material_ids: np.ndarray | None = None   # (T,) int32
    stats: dict = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))


# ====================================================================== 焊接
def weld(positions: np.ndarray, uvs: np.ndarray | None = None, material_ids: np.ndarray | None = None,
         tolerance: float = 1e-6, drop_degenerate: bool = True) -> SculptMesh:
    """逐角点（每 3 行一个三角形）变成索引网格：位置相同（容差内）的角点合成一个顶点，UV 留在角点上。
    drop_degenerate=False 时保留焊接后退化的三角形，三角形和原来的逐角点一一对应。"""
    started = time.perf_counter()
    p = np.asarray(positions, np.float64).reshape(-1, 3)
    corners = len(p)
    if corners == 0:
        return SculptMesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    extent = float(np.max(np.ptp(p, axis=0))) or 1.0
    step = max(tolerance * extent, 1e-12)
    q = np.round((p - p.min(axis=0)) / step).astype(np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    sq = q[order]
    new_group = np.ones(corners, bool)
    new_group[1:] = np.any(sq[1:] != sq[:-1], axis=1)
    group_id = np.cumsum(new_group) - 1
    inverse = np.empty(corners, np.int64)
    inverse[order] = group_id
    first = order[new_group]
    vertices = p[first].astype(np.float32)
    triangles = inverse.reshape(-1, 3).astype(np.int32)
    mats = None if material_ids is None else np.asarray(material_ids, np.int32).copy()
    cuv = None if uvs is None else np.asarray(uvs, np.float32).reshape(-1, 2).copy()
    # 焊接后退化（两个角是同一个顶点）的三角形去掉
    t = triangles
    keep = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    if not drop_degenerate:
        keep[:] = True
    if not keep.all():
        triangles = triangles[keep]
        if mats is not None:
            mats = mats[keep]
        if cuv is not None:
            cuv = cuv.reshape(-1, 3, 2)[keep].reshape(-1, 2)
    mesh = SculptMesh(vertices, triangles, cuv, mats if mats is not None else np.zeros(len(triangles), np.int32))
    mesh.stats = {"weld_ms": round((time.perf_counter() - started) * 1000.0, 1), "removed": int((~keep).sum())}
    return mesh


def unweld(mesh: SculptMesh, normals: np.ndarray | None = None) -> tuple:
    """索引网格变回逐角点：(positions (3T,3), normals (3T,3), uvs (3T,2), material_ids (T,))。"""
    tri = mesh.triangles.reshape(-1)
    positions = mesh.vertices[tri].astype(np.float32)
    if normals is None:
        normals = vertex_normals(mesh.vertices, mesh.triangles)
    corner_normals = np.asarray(normals, np.float32)[tri]
    uvs = mesh.corner_uvs if mesh.corner_uvs is not None else np.zeros((len(tri), 2), np.float32)
    mats = mesh.material_ids if mesh.material_ids is not None else np.zeros(len(mesh.triangles), np.int32)
    return positions, corner_normals, np.asarray(uvs, np.float32), np.asarray(mats, np.int32)


# ====================================================================== 法线与邻接
def face_normals(vertices: np.ndarray, triangles: np.ndarray, normalize: bool = False) -> np.ndarray:
    v = np.asarray(vertices, np.float64)
    t = np.asarray(triangles, np.int64)
    n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    if normalize:
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return n


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """按面积加权的顶点法线，单位长度。"""
    count = len(vertices)
    fn = face_normals(vertices, triangles)
    index = np.asarray(triangles, np.int64).reshape(-1)
    out = np.empty((count, 3), np.float64)
    for k in range(3):
        out[:, k] = np.bincount(index, weights=np.repeat(fn[:, k], 3), minlength=count)
    length = np.linalg.norm(out, axis=1, keepdims=True)
    out = np.where(length > 1e-30, out / np.maximum(length, 1e-30), np.array([0.0, 1.0, 0.0]))
    return out.astype(np.float32)


def vertex_faces(vertex_count: int, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """顶点到三角形的 CSR：(offsets (V+1,), faces (3T,))。"""
    flat = np.asarray(triangles, np.int64).reshape(-1)
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=vertex_count)
    offsets = np.zeros(vertex_count + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets.astype(np.int32), (order // 3).astype(np.int32)


def edges_of(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """唯一边 (E,2)（小编号在前）、每条三角形边对应的边编号 (T,3)、每条边被几个三角形用。"""
    t = np.asarray(triangles, np.int64)
    a = t[:, [0, 1, 2]].reshape(-1)
    b = t[:, [1, 2, 0]].reshape(-1)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    vcount = int(t.max()) + 1 if len(t) else 1
    key = lo * vcount + hi
    unique, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    edges = np.stack([unique // vcount, unique % vcount], axis=1).astype(np.int32)
    return edges, inverse.reshape(-1, 3).astype(np.int64), counts


def mesh_stats(mesh: SculptMesh) -> dict:
    edges, _tri_edges, counts = edges_of(mesh.triangles)
    v, e, f = mesh.vertex_count, len(edges), mesh.triangle_count
    return {"vertices": v, "triangles": f, "edges": e, "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum()), "euler": int(v - e + f)}


# ====================================================================== 空间排序
@njit(cache=True, nogil=True)
def _spread(x):
    x = x & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


@njit(cache=True, nogil=True, parallel=True)
def _morton(q):
    out = np.empty(q.shape[0], np.uint64)
    for i in prange(q.shape[0]):
        out[i] = _spread(np.uint64(q[i, 0])) | (_spread(np.uint64(q[i, 1])) << np.uint64(1)) | \
            (_spread(np.uint64(q[i, 2])) << np.uint64(2))
    return out


def morton_order(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, np.float64)
    if len(p) == 0:
        return np.zeros(0, np.int64)
    lo = p.min(axis=0)
    span = np.maximum(p.max(axis=0) - lo, 1e-12)
    q = np.clip(((p - lo) / span * 2097151.0), 0, 2097151).astype(np.uint64)
    return np.argsort(_morton(q), kind="stable")


def spatial_sort(mesh: SculptMesh) -> tuple[SculptMesh, np.ndarray]:
    """按空间位置（Morton 序）重排顶点和三角形，相邻的顶点在内存里也相邻。返回 (新网格, old_of_new)。"""
    order = morton_order(mesh.vertices)
    new_of_old = np.empty(len(order), np.int64)
    new_of_old[order] = np.arange(len(order))
    vertices = mesh.vertices[order]
    triangles = new_of_old[mesh.triangles].astype(np.int32)
    tri_order = np.argsort(triangles.min(axis=1), kind="stable")
    triangles = triangles[tri_order]
    uvs = None
    if mesh.corner_uvs is not None:
        uvs = mesh.corner_uvs.reshape(-1, 3, 2)[tri_order].reshape(-1, 2)
    mats = mesh.material_ids[tri_order] if mesh.material_ids is not None else None
    return SculptMesh(vertices, triangles, uvs, mats), order


# ====================================================================== 非流形修复
@njit(cache=True, nogil=True)
def _halfedge_count(offsets, targets, a, b):
    n = 0
    for e in range(offsets[a], offsets[a + 1]):
        if targets[e] == b:
            n += 1
    return n


@njit(cache=True, nogil=True)
def _local_edge(tris, t, a, b):
    for k in range(3):
        p = tris[t, k]
        q = tris[t, (k + 1) % 3]
        if (p == a and q == b) or (p == b and q == a):
            return k
    return -1


@njit(cache=True, nogil=True)
def _split_fans(tris, verts, vertex_count):
    """非流形边（三个以上三角形共用）两端的顶点，按「三角形扇」拆成几个顶点。
    非流形边上的三角形按绕边的角度排好，夹着实体的相邻两片配成一对（这样两块只碰到一条边的实体会分开），
    配对的两片在两端的顶点上都算连通，保证拆完两端一致、不留缝。
    返回 (新三角形, 每个新顶点来自哪个旧顶点 (V',), 拆出的顶点数)。"""
    counts = np.zeros(vertex_count + 1, np.int64)
    for t in range(tris.shape[0]):
        for k in range(3):
            counts[tris[t, k] + 1] += 1
    offsets = np.cumsum(counts)
    fill = offsets[:-1].copy()
    targets = np.empty(tris.shape[0] * 3, np.int64)
    faces = np.empty(tris.shape[0] * 3, np.int64)
    for t in range(tris.shape[0]):
        for k in range(3):
            a = tris[t, k]
            targets[fill[a]] = tris[t, (k + 1) % 3]
            faces[fill[a]] = t
            fill[a] += 1
    candidate = np.zeros(vertex_count, np.bool_)
    pair_of = np.full(tris.shape[0] * 3, -1, np.int64)
    ring = np.empty(64, np.int64)
    forward = np.empty(64, np.bool_)
    angle = np.empty(64, np.float64)
    for t in range(tris.shape[0]):
        for k in range(3):
            a = tris[t, k]
            b = tris[t, (k + 1) % 3]
            n_ab = _halfedge_count(offsets, targets, a, b)
            n_ba = _halfedge_count(offsets, targets, b, a)
            if n_ab + n_ba <= 2:
                continue
            candidate[a] = True
            candidate[b] = True
            # 每条非流形边只处理一次：轮到这条边上编号最小的三角形时处理
            lo = min(a, b)
            hi = max(a, b)
            first = -1
            for e in range(offsets[lo], offsets[lo + 1]):
                if targets[e] == hi and (first < 0 or faces[e] < first):
                    first = faces[e]
            for e in range(offsets[hi], offsets[hi + 1]):
                if targets[e] == lo and (first < 0 or faces[e] < first):
                    first = faces[e]
            if first != t:
                continue
            m = 0
            for e in range(offsets[lo], offsets[lo + 1]):
                if targets[e] == hi and m < 64:
                    ring[m] = faces[e]
                    forward[m] = True          # lo→hi
                    m += 1
            for e in range(offsets[hi], offsets[hi + 1]):
                if targets[e] == lo and m < 64:
                    ring[m] = faces[e]
                    forward[m] = False         # hi→lo
                    m += 1
            if m % 2 == 1:
                continue
            dx = verts[hi, 0] - verts[lo, 0]
            dy = verts[hi, 1] - verts[lo, 1]
            dz = verts[hi, 2] - verts[lo, 2]
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            if length <= 0.0:
                continue
            dx, dy, dz = dx / length, dy / length, dz / length
            # 垂直于边的一组基 (u, w)，w = d × u
            if abs(dx) < 0.9:
                ux, uy, uz = 0.0, -dz, dy
            else:
                ux, uy, uz = dz, 0.0, -dx
            ul = math.sqrt(ux * ux + uy * uy + uz * uz)
            ux, uy, uz = ux / ul, uy / ul, uz / ul
            wx = dy * uz - dz * uy
            wy = dz * ux - dx * uz
            wz = dx * uy - dy * ux
            for i in range(m):
                f = ring[i]
                x = tris[f, 0]
                if x == lo or x == hi:
                    x = tris[f, 1]
                    if x == lo or x == hi:
                        x = tris[f, 2]
                rx = verts[x, 0] - verts[lo, 0]
                ry = verts[x, 1] - verts[lo, 1]
                rz = verts[x, 2] - verts[lo, 2]
                angle[i] = math.atan2(rx * wx + ry * wy + rz * wz, rx * ux + ry * uy + rz * uz)
            order = np.argsort(angle[:m])
            ok = True
            for i in range(m):
                if forward[order[i]] == forward[order[(i + 1) % m]]:
                    ok = False
            if not ok:
                continue
            # 绕 lo→hi 按右手方向转：hi→lo 的三角形法线朝角度减小的一侧，实体在它后面，和下一片配对
            for i in range(m):
                if forward[order[i]]:
                    continue
                f0 = ring[order[i]]
                f1 = ring[order[(i + 1) % m]]
                k0 = _local_edge(tris, f0, lo, hi)
                k1 = _local_edge(tris, f1, lo, hi)
                pair_of[f0 * 3 + k0] = f1
                pair_of[f1 * 3 + k1] = f0
    out = tris.copy()
    source = list(range(vertex_count))
    split = 0
    for v in range(vertex_count):
        if not candidate[v]:
            continue
        s, e = offsets[v], offsets[v + 1]
        m = e - s
        if m < 2:
            continue
        parent = np.arange(m)
        for i in range(m):
            ti = faces[s + i]
            for j in range(i + 1, m):
                tj = faces[s + j]
                for ki in range(3):
                    w = tris[ti, ki]
                    if w == v:
                        continue
                    shared = False
                    for kj in range(3):
                        if tris[tj, kj] == w:
                            shared = True
                    if not shared:
                        continue
                    if _halfedge_count(offsets, targets, v, w) + _halfedge_count(offsets, targets, w, v) == 2:
                        joined = True
                    else:
                        k = _local_edge(tris, ti, v, w)
                        joined = k >= 0 and pair_of[ti * 3 + k] == tj
                    if not joined:
                        continue
                    ri = i
                    while parent[ri] != ri:
                        ri = parent[ri]
                    rj = j
                    while parent[rj] != rj:
                        rj = parent[rj]
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
        new_id = np.full(m, -1, np.int64)
        for i in range(m):
            r = i
            while parent[r] != r:
                r = parent[r]
            if r == 0:
                continue
            if new_id[r] < 0:
                new_id[r] = len(source)
                source.append(v)
                split += 1
            t = faces[s + i]
            for k in range(3):
                if out[t, k] == v:
                    out[t, k] = new_id[r]
    return out, np.array(source, np.int64), split


def split_nonmanifold(mesh: SculptMesh) -> tuple[SculptMesh, int]:
    """把非流形边拆开（每个三角形扇用自己的顶点），后续的细分、平滑、展开 UV 都按流形处理。"""
    if mesh.triangle_count == 0:
        return mesh, 0
    tris, source, split = _split_fans(np.ascontiguousarray(mesh.triangles, np.int64),
                                      np.ascontiguousarray(mesh.vertices, np.float64), mesh.vertex_count)
    if split == 0:
        return mesh, 0
    out = SculptMesh(np.asarray(mesh.vertices)[source], tris.astype(np.int32), mesh.corner_uvs, mesh.material_ids)
    mask = getattr(mesh, "mask", None)
    if mask is not None:
        out.mask = np.asarray(mask)[source]
    return out, int(split)


# ====================================================================== 细分（Loop）
def subdivide(mesh: SculptMesh, levels: int = 1, smooth: bool = True) -> SculptMesh:
    """Loop 细分：每级三角形数 ×4。smooth=False 时只在边中点切分、不移动顶点。逐角点 UV 线性插值，接缝保持。"""
    for _ in range(max(0, int(levels))):
        mesh = _subdivide_once(mesh, smooth)
    return mesh


def _subdivide_once(mesh: SculptMesh, smooth: bool) -> SculptMesh:
    v = np.asarray(mesh.vertices, np.float64)
    t = np.asarray(mesh.triangles, np.int64)
    vcount = len(v)
    edges, tri_edges, counts = edges_of(t)
    ecount = len(edges)
    # 新的边点
    mid = (v[edges[:, 0]] + v[edges[:, 1]]) * 0.5
    if smooth:
        opposite = np.zeros((ecount, 3), np.float64)
        third = t[:, [2, 0, 1]].reshape(-1)        # 边 (0,1) 对面是 2，边 (1,2) 对面是 0，边 (2,0) 对面是 1
        flat_edges = tri_edges.reshape(-1)
        for k in range(3):
            opposite[:, k] = np.bincount(flat_edges, weights=v[third, k], minlength=ecount)
        interior = counts == 2
        edge_points = mid.copy()
        edge_points[interior] = (v[edges[interior, 0]] + v[edges[interior, 1]]) * 0.375 + opposite[interior] * 0.125
        # 原顶点
        boundary_edge = counts == 1
        boundary_vertex = np.zeros(vcount, bool)
        boundary_vertex[edges[boundary_edge].reshape(-1)] = True
        valence = np.bincount(edges.reshape(-1), minlength=vcount).astype(np.float64)
        neighbor_sum = np.zeros((vcount, 3), np.float64)
        for k in range(3):
            neighbor_sum[:, k] = (np.bincount(edges[:, 0], weights=v[edges[:, 1], k], minlength=vcount)
                                  + np.bincount(edges[:, 1], weights=v[edges[:, 0], k], minlength=vcount))
        n = np.maximum(valence, 1.0)
        beta = np.where(n > 3, 3.0 / (8.0 * n), 3.0 / 16.0)
        vertex_points = v * (1.0 - n * beta)[:, None] + neighbor_sum * beta[:, None]
        if boundary_vertex.any():
            bsum = np.zeros((vcount, 3), np.float64)
            be = edges[boundary_edge]
            for k in range(3):
                bsum[:, k] = (np.bincount(be[:, 0], weights=v[be[:, 1], k], minlength=vcount)
                              + np.bincount(be[:, 1], weights=v[be[:, 0], k], minlength=vcount))
            vertex_points[boundary_vertex] = v[boundary_vertex] * 0.75 + bsum[boundary_vertex] * 0.125
            edge_points[boundary_edge] = mid[boundary_edge]
    else:
        edge_points = mid
        vertex_points = v
    vertices = np.concatenate([vertex_points, edge_points]).astype(np.float32)
    e = tri_edges + vcount                         # 边点的新编号：e01, e12, e20
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    e01, e12, e20 = e[:, 0], e[:, 1], e[:, 2]
    new_tris = np.stack([
        np.stack([a, e01, e20], axis=1),
        np.stack([e01, b, e12], axis=1),
        np.stack([e20, e12, c], axis=1),
        np.stack([e01, e12, e20], axis=1)], axis=1).reshape(-1, 3).astype(np.int32)
    uvs = None
    if mesh.corner_uvs is not None:
        cu = np.asarray(mesh.corner_uvs, np.float32).reshape(-1, 3, 2)
        u0, u1, u2 = cu[:, 0], cu[:, 1], cu[:, 2]
        m01, m12, m20 = (u0 + u1) * 0.5, (u1 + u2) * 0.5, (u2 + u0) * 0.5
        uvs = np.stack([
            np.stack([u0, m01, m20], axis=1),
            np.stack([m01, u1, m12], axis=1),
            np.stack([m20, m12, u2], axis=1),
            np.stack([m01, m12, m20], axis=1)], axis=1).reshape(-1, 2).astype(np.float32)
    mats = None
    if mesh.material_ids is not None:
        mats = np.repeat(np.asarray(mesh.material_ids, np.int32), 4)
    return SculptMesh(vertices, new_tris, uvs, mats)


# ====================================================================== 默认底模
def make_sphere(level: int = 64, radius: float = 1.0) -> SculptMesh:
    """立方体每面切成 level×level 的网格再投到球面（ZBrush 式均匀），四边形拆两个三角形。
    UV 把六个面排成 3×2 的格子。"""
    n = max(1, int(level))
    s = np.linspace(-1.0, 1.0, n + 1)
    gu, gv = np.meshgrid(s, s, indexing="xy")
    faces = [
        (np.stack([np.ones_like(gu), gv, -gu], -1)),     # +X
        (np.stack([-np.ones_like(gu), gv, gu], -1)),     # -X
        (np.stack([gu, np.ones_like(gu), -gv], -1)),     # +Y
        (np.stack([gu, -np.ones_like(gu), gv], -1)),     # -Y
        (np.stack([gu, gv, np.ones_like(gu)], -1)),      # +Z
        (np.stack([-gu, gv, -np.ones_like(gu)], -1)),    # -Z
    ]
    positions, uvs = [], []
    for index, grid in enumerate(faces):
        p = grid.reshape(-1, 3)
        # 等面积一些的投影：先做正切变换让格子在球面上更均匀
        p = np.tan(p * (math.pi / 4.0))
        p = p / np.linalg.norm(p, axis=1, keepdims=True) * radius
        col, row = index % 3, index // 3
        uv = np.stack([(gu.reshape(-1) * 0.5 + 0.5 + col) / 3.0, (gv.reshape(-1) * 0.5 + 0.5 + row) / 2.0], -1)
        i = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
        a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, 1:].ravel(), i[1:, :-1].ravel()
        tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
        positions.append(p[tris.reshape(-1)])
        uvs.append(uv[tris.reshape(-1)])
    corners = np.concatenate(positions).astype(np.float32)
    corner_uvs = np.concatenate(uvs).astype(np.float32)
    # 让三角形朝外（逆时针）：检查第一个三角形的法线方向，整体翻转
    mesh = weld(corners, corner_uvs, np.zeros(len(corners) // 3, np.int32), tolerance=1e-7)
    fn = face_normals(mesh.vertices, mesh.triangles)
    centers = mesh.vertices[mesh.triangles].mean(axis=1)
    inward = np.einsum("ij,ij->i", fn, centers) < 0
    if inward.any():
        tri = mesh.triangles.copy()
        tri[inward] = tri[inward][:, [0, 2, 1]]
        cu = mesh.corner_uvs.reshape(-1, 3, 2).copy()
        cu[inward] = cu[inward][:, [0, 2, 1]]
        mesh = SculptMesh(mesh.vertices, tri, cu.reshape(-1, 2), mesh.material_ids)
    return mesh
