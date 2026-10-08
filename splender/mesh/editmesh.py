"""编辑模式的网格（照 Blender 的 BMesh）：点、多边形面，边由面推出来（另外可以有不属于任何面的散边、散点）；
每个面角存一份 UV（同一个点在不同面上可以有不同 UV，接缝就是这样）；点、边、面各有选中、隐藏标记。

坐标都是内部坐标（Y 朝上，和模型数据一样，已经含物体变换）。拓扑变了（加面、删面、合并点…）由操作重建数组，
边表按需重算。退出编辑模式时三角化成绘制用的逐角点三角形（MeshData），多边形大小记在 polygon_sizes 里。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

SELECT_MODES = ("VERT", "EDGE", "FACE")


@dataclass
class EdgeTable:
    edges: np.ndarray          # (E, 2) 点号，小号在前
    loop_edge: np.ndarray      # (L,) 每个面角到下一个面角的那条边
    face_count: np.ndarray     # (E,) 每条边挨着几个面
    loose: np.ndarray          # (E,) 是不是散边（不属于任何面）


class EditMesh:
    def __init__(self, verts, face_start, loop_vert, loop_uv=None, face_mat=None, face_smooth=None,
                 materials=None, has_uvs: bool = True, loose_edges=None) -> None:
        self.verts = np.ascontiguousarray(verts, np.float64).reshape(-1, 3)
        self.face_start = np.ascontiguousarray(face_start, np.int64)
        self.loop_vert = np.ascontiguousarray(loop_vert, np.int64)
        loops = len(self.loop_vert)
        faces = len(self.face_start) - 1
        self.loop_uv = np.zeros((loops, 2), np.float32) if loop_uv is None else \
            np.ascontiguousarray(loop_uv, np.float32).reshape(-1, 2)
        self.face_mat = np.zeros(faces, np.int32) if face_mat is None else np.ascontiguousarray(face_mat, np.int32)
        self.face_smooth = np.zeros(faces, bool) if face_smooth is None else np.ascontiguousarray(face_smooth, bool)
        self.materials = list(materials or ["材质"])
        self.has_uvs = bool(has_uvs)
        self.loose_edges = np.zeros((0, 2), np.int64) if loose_edges is None else \
            np.ascontiguousarray(loose_edges, np.int64).reshape(-1, 2)
        v = len(self.verts)
        self.vert_sel = np.zeros(v, bool)
        self.vert_hide = np.zeros(v, bool)
        self.face_sel = np.zeros(faces, bool)
        self.face_hide = np.zeros(faces, bool)
        self._edges: EdgeTable | None = None
        self.edge_sel = np.zeros(len(self.edges().edges), bool)
        self.select_mode = {"VERT"}
        self.active: tuple | None = None          # ("VERT" / "EDGE" / "FACE", 编号)
        self.topology_version = 0
        self.geometry_version = 0

    # ---------------------------------------------------------------- 基本量
    @property
    def vert_count(self) -> int:
        return int(len(self.verts))

    @property
    def face_count(self) -> int:
        return int(len(self.face_start) - 1)

    @property
    def loop_count(self) -> int:
        return int(len(self.loop_vert))

    def face_sizes(self) -> np.ndarray:
        return np.diff(self.face_start)

    def loop_face(self) -> np.ndarray:
        return np.repeat(np.arange(self.face_count, dtype=np.int64), self.face_sizes())

    def loop_next(self) -> np.ndarray:
        """每个面角在同一个面里的下一个面角。"""
        nxt = np.arange(1, self.loop_count + 1, dtype=np.int64)
        if self.face_count:
            nxt[self.face_start[1:] - 1] = self.face_start[:-1]
        return nxt

    def loop_prev(self) -> np.ndarray:
        prv = np.arange(-1, self.loop_count - 1, dtype=np.int64)
        if self.face_count:
            prv[self.face_start[:-1]] = self.face_start[1:] - 1
        return prv

    def face_verts(self, face: int) -> np.ndarray:
        return self.loop_vert[self.face_start[face]:self.face_start[face + 1]]

    # ---------------------------------------------------------------- 边表
    def edges(self) -> EdgeTable:
        if self._edges is not None:
            return self._edges
        v = max(1, self.vert_count)
        a = self.loop_vert
        b = self.loop_vert[self.loop_next()] if self.loop_count else np.zeros(0, np.int64)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        loose = self.loose_edges
        llo = np.minimum(loose[:, 0], loose[:, 1]) if len(loose) else np.zeros(0, np.int64)
        lhi = np.maximum(loose[:, 0], loose[:, 1]) if len(loose) else np.zeros(0, np.int64)
        keys = np.concatenate([lo * v + hi, llo * v + lhi])
        uniq, inverse = np.unique(keys, return_inverse=True)
        edges = np.stack([uniq // v, uniq % v], axis=1).astype(np.int64)
        loop_edge = inverse[:self.loop_count].astype(np.int64)
        face_count = np.bincount(loop_edge, minlength=len(edges)).astype(np.int64)
        self._edges = EdgeTable(edges, loop_edge, face_count, face_count == 0)
        return self._edges

    def invalidate(self, topology: bool = True) -> None:
        """拓扑变了（数组换过）：边表重算，选中标记补齐长度。"""
        old_edges = self._edges
        old_sel = getattr(self, "edge_sel", None)
        self._edges = None
        table = self.edges()
        if old_sel is None or old_edges is None or len(old_sel) != len(table.edges) or topology:
            # 边的选中按点重新推（拓扑操作之后最可靠）
            self.edge_sel = self.vert_sel[table.edges[:, 0]] & self.vert_sel[table.edges[:, 1]] \
                if len(table.edges) else np.zeros(0, bool)
        if topology:
            self.topology_version += 1
        self.geometry_version += 1

    # ---------------------------------------------------------------- 法线、中心
    def face_normals(self) -> np.ndarray:
        """每个面的单位法线（按多边形的 Newell 法则，凹多边形也对）。"""
        if not self.face_count:
            return np.zeros((0, 3))
        p = self.verts[self.loop_vert]
        q = self.verts[self.loop_vert[self.loop_next()]]
        cross = np.stack([(p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2]),
                          (p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0]),
                          (p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])], axis=1)
        n = np.add.reduceat(cross, self.face_start[:-1], axis=0) if self.loop_count else np.zeros((0, 3))
        length = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(length, 1e-30)

    def face_centers(self) -> np.ndarray:
        if not self.face_count:
            return np.zeros((0, 3))
        p = self.verts[self.loop_vert]
        sums = np.add.reduceat(p, self.face_start[:-1], axis=0)
        return sums / self.face_sizes()[:, None]

    def vert_normals(self) -> np.ndarray:
        """点的法线：周围面法线按面积加权平均。"""
        out = np.zeros_like(self.verts)
        if self.face_count:
            fn = self.face_normals()
            area = self.face_areas()
            lf = self.loop_face()
            weight = (fn * area[:, None])[lf]
            np.add.at(out, self.loop_vert, weight)
        length = np.linalg.norm(out, axis=1, keepdims=True)
        return np.where(length > 1e-30, out / np.maximum(length, 1e-30), np.array([0.0, 1.0, 0.0]))

    def face_areas(self) -> np.ndarray:
        if not self.face_count:
            return np.zeros(0)
        p = self.verts[self.loop_vert]
        q = self.verts[self.loop_vert[self.loop_next()]]
        cross = np.stack([(p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2]),
                          (p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0]),
                          (p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])], axis=1)
        n = np.add.reduceat(cross, self.face_start[:-1], axis=0)
        return 0.5 * np.linalg.norm(n, axis=1)

    # ---------------------------------------------------------------- 选中
    def set_select_mode(self, modes) -> None:
        modes = {m for m in modes if m in SELECT_MODES} or {"VERT"}
        old = set(self.select_mode)
        self.select_mode = modes
        # 从细到粗换模式时：边模式只留两端都选中的边，面模式只留全选中的面（和 Blender 一样）
        if "VERT" not in modes and "VERT" in old:
            self.flush_from_verts()
        self.flush()

    def flush_from_verts(self) -> None:
        table = self.edges()
        self.edge_sel = self.vert_sel[table.edges[:, 0]] & self.vert_sel[table.edges[:, 1]] \
            if len(table.edges) else np.zeros(0, bool)
        self.face_sel = self._faces_all(self.vert_sel[self.loop_vert]) if self.face_count else np.zeros(0, bool)
        self._apply_hidden()

    def flush_from_edges(self) -> None:
        table = self.edges()
        vert = np.zeros(self.vert_count, bool)
        chosen = table.edges[self.edge_sel]
        vert[chosen.reshape(-1)] = True
        self.vert_sel = vert
        self.face_sel = self._faces_all(self.edge_sel[table.loop_edge]) if self.face_count else np.zeros(0, bool)
        self._apply_hidden()

    def flush_from_faces(self) -> None:
        table = self.edges()
        lf = self.loop_face()
        loop_on = self.face_sel[lf] if self.face_count else np.zeros(0, bool)
        vert = np.zeros(self.vert_count, bool)
        vert[self.loop_vert[loop_on]] = True
        edge = np.zeros(len(table.edges), bool)
        edge[table.loop_edge[loop_on]] = True
        self.vert_sel = vert
        self.edge_sel = edge
        self._apply_hidden()

    def flush(self) -> None:
        """按当前选择模式把选中状态推到其它元素上。"""
        if "VERT" in self.select_mode:
            self.flush_from_verts()
        elif "EDGE" in self.select_mode:
            self.flush_from_edges()
        else:
            self.flush_from_faces()

    def _faces_all(self, loop_values: np.ndarray) -> np.ndarray:
        """每个面的面角是不是全为真。"""
        if not self.face_count:
            return np.zeros(0, bool)
        return np.logical_and.reduceat(loop_values, self.face_start[:-1])

    def _apply_hidden(self) -> None:
        self.vert_sel &= ~self.vert_hide
        self.face_sel &= ~self.face_hide
        table = self.edges()
        if len(table.edges):
            hidden_edge = self.vert_hide[table.edges[:, 0]] | self.vert_hide[table.edges[:, 1]]
            self.edge_sel &= ~hidden_edge

    def select_all(self, action: str) -> None:
        visible_v = ~self.vert_hide
        if action == "TOGGLE":
            action = "DESELECT" if self.vert_sel.any() or self.face_sel.any() else "SELECT"
        if action == "SELECT":
            self.vert_sel = visible_v.copy()
            self.face_sel = ~self.face_hide
            table = self.edges()
            self.edge_sel = np.ones(len(table.edges), bool)
        elif action == "DESELECT":
            self.vert_sel[:] = False
            self.face_sel[:] = False
            self.edge_sel[:] = False
            self.active = None
        else:   # INVERT：按当前模式反选
            if "VERT" in self.select_mode:
                self.vert_sel = ~self.vert_sel & visible_v
                self.flush_from_verts()
            elif "EDGE" in self.select_mode:
                self.edge_sel = ~self.edge_sel
                self.flush_from_edges()
            else:
                self.face_sel = ~self.face_sel & ~self.face_hide
                self.flush_from_faces()
            return
        self._apply_hidden()

    def selected_vert_indices(self) -> np.ndarray:
        return np.nonzero(self.vert_sel)[0]

    # ---------------------------------------------------------------- 撤销用的快照
    def snapshot(self) -> dict:
        return {name: copy.deepcopy(getattr(self, name)) for name in (
            "verts", "face_start", "loop_vert", "loop_uv", "face_mat", "face_smooth", "materials", "has_uvs",
            "loose_edges", "vert_sel", "vert_hide", "face_sel", "face_hide", "edge_sel", "select_mode", "active")}

    def restore(self, snap: dict) -> None:
        for name, value in snap.items():
            setattr(self, name, copy.deepcopy(value))
        sel = self.edge_sel
        self._edges = None
        self.edges()
        self.edge_sel = sel if len(sel) == len(self._edges.edges) else np.zeros(len(self._edges.edges), bool)
        self.topology_version += 1
        self.geometry_version += 1

    def copy(self) -> "EditMesh":
        out = EditMesh(self.verts, self.face_start, self.loop_vert, self.loop_uv, self.face_mat, self.face_smooth,
                       self.materials, self.has_uvs, self.loose_edges)
        out.restore(self.snapshot())
        return out


# ====================================================================== 模型数据 → 编辑网格
def weld_corners(positions: np.ndarray, tolerance: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """逐角点位置焊成点：(点坐标 (V,3), 每个角点的点号 (3T,))。容差按模型尺寸算。"""
    p = np.asarray(positions, np.float64).reshape(-1, 3)
    if len(p) == 0:
        return np.zeros((0, 3)), np.zeros(0, np.int64)
    extent = float(np.max(np.ptp(p, axis=0))) or 1.0
    step = max(tolerance * extent, 1e-12)
    q = np.round((p - p.min(axis=0)) / step).astype(np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    sq = q[order]
    new = np.ones(len(p), bool)
    new[1:] = np.any(sq[1:] != sq[:-1], axis=1)
    group = np.cumsum(new) - 1
    ids = np.empty(len(p), np.int64)
    ids[order] = group
    verts = p[order[new]]
    return verts, ids


def polygons_of(data, corner_vert: np.ndarray) -> np.ndarray:
    """每个多边形由哪几个三角形组成：返回多边形大小列表（对应三角形顺序，扇形拆分）。
    模型带 polygon_sizes 且对得上就用它；否则从三角化的四边形里找对角线配对（找不到就都是三角形）。"""
    tris = len(corner_vert) // 3
    sizes = getattr(data, "polygon_sizes", None)
    if sizes is not None and len(sizes):
        sizes = np.asarray(sizes, np.int64)
        if int(np.maximum(sizes - 2, 1).sum()) == tris:
            return sizes
    return None


def from_mesh_data(data, tolerance: float = 1e-6) -> EditMesh:
    """绘制用的逐角点三角形 → 编辑网格。四边形、多边形按 polygon_sizes 还原；没有就把拆过的四边形配回来。"""
    positions = np.asarray(data.positions, np.float64).reshape(-1, 3)
    uvs = np.asarray(data.uvs, np.float32).reshape(-1, 2)
    normals = np.asarray(data.normals, np.float64).reshape(-1, 3)
    mats = np.asarray(data.material_ids, np.int32)
    verts, corner_vert = weld_corners(positions, tolerance)
    tris = len(corner_vert) // 3
    sizes = polygons_of(data, corner_vert)
    if sizes is not None:
        # 扇形：第 k 个三角形是 (v0, v_{k+1}, v_{k+2})
        counts = np.maximum(sizes - 2, 1)
        tri_start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        face_sizes = sizes
        face_of_loop = np.repeat(np.arange(len(sizes)), sizes)
        k = np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        t0 = tri_start[face_of_loop]
        loop_corner = np.where(k == 0, 3 * t0, np.where(k == 1, 3 * t0 + 1, 3 * (t0 + k - 2) + 2)).astype(np.int64)
        face_first_tri = tri_start
    else:
        from ..engine.meshgpu import quad_diagonals

        diag = quad_diagonals(positions, corner_vert)
        loop_corner, face_sizes, face_first_tri = _pair_quads(diag, corner_vert)
    face_start = np.concatenate([[0], np.cumsum(face_sizes)]).astype(np.int64)
    loop_vert = corner_vert[loop_corner]
    loop_uv = uvs[loop_corner]
    face_mat = mats[np.asarray(face_first_tri, np.int64)] if tris else np.zeros(0, np.int32)
    # 平滑标记：面角法线和面法线不一样就是平滑面
    mesh = EditMesh(verts, face_start, loop_vert, loop_uv, face_mat, None, list(data.materials),
                    bool(getattr(data, "has_uvs", True)))
    if mesh.face_count:
        fn = mesh.face_normals()
        lf = mesh.loop_face()
        dots = np.einsum("ij,ij->i", normals[loop_corner], fn[lf])
        rough = dots < 0.9999
        mesh.face_smooth = np.logical_or.reduceat(rough, face_start[:-1])
    loose = getattr(data, "loose_edges", None)
    if loose is not None and len(loose):
        lv, lids = weld_corners(np.concatenate([verts, np.asarray(loose, np.float64).reshape(-1, 3)]), tolerance)
        if len(lv) == len(verts):
            mesh.loose_edges = lids[len(verts):].reshape(-1, 2)
            mesh.invalidate()
    return mesh


def _pair_quads(diag: np.ndarray, corner_vert: np.ndarray):
    """按对角线把成对的三角形拼成四边形。返回 (面角 → 原角点号, 每个面的大小, 每个面的第一个三角形)。"""
    tris = len(corner_vert) // 3
    w = corner_vert.reshape(-1, 3)
    has = diag.any(axis=1)
    edge_index = np.argmax(diag, axis=1)
    a = w[np.arange(tris), edge_index]
    b = w[np.arange(tris), (edge_index + 1) % 3]
    vmax = int(corner_vert.max()) + 1 if len(corner_vert) else 1
    key = np.where(has, np.minimum(a, b) * vmax + np.maximum(a, b), -1 - np.arange(tris))
    order = np.argsort(key, kind="stable")
    sk = key[order]
    partner = np.full(tris, -1, np.int64)
    same = np.nonzero((sk[1:] == sk[:-1]) & (sk[1:] >= 0))[0]
    if len(same):
        # 三个三角形共用一条「对角线」的情况不配对
        triple = np.zeros(len(sk), bool)
        triple[1:-1] = (sk[1:-1] == sk[:-2]) & (sk[1:-1] == sk[2:])
        ok = ~(triple[same] | triple[np.minimum(same + 1, len(sk) - 1)])
        same = same[ok]
        partner[order[same]] = order[same + 1]
        partner[order[same + 1]] = order[same]
    ids = np.arange(tris)
    lead = (partner >= 0) & (ids < partner)
    single = partner < 0
    firsts = np.nonzero(single | lead)[0]
    sizes = np.where(single[firsts], 3, 4).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    loop_corner = np.empty(int(sizes.sum()), np.int64)
    s = firsts[sizes == 3]
    s_start = starts[sizes == 3]
    for c in range(3):
        loop_corner[s_start + c] = 3 * s + c
    q = firsts[sizes == 4]
    q_start = starts[sizes == 4]
    e = edge_index[q]
    p = partner[q]
    pe = edge_index[p]
    loop_corner[q_start] = 3 * q + e
    loop_corner[q_start + 1] = 3 * p + (pe + 2) % 3
    loop_corner[q_start + 2] = 3 * q + (e + 1) % 3
    loop_corner[q_start + 3] = 3 * q + (e + 2) % 3
    return loop_corner, sizes, firsts


# ====================================================================== 编辑网格 → 模型数据
def triangulate(mesh: EditMesh) -> tuple[np.ndarray, np.ndarray]:
    """多边形拆成三角形：返回 (每个三角形的三个面角 (T,3), 每个面从哪个面角开始扇形)。
    四边形沿较短的对角线拆（起点挪到那条对角线的一端），更大的多边形从第一个面角扇形拆。"""
    sizes = mesh.face_sizes()
    starts = mesh.face_start[:-1].copy()
    start_loop = starts.copy()
    quads = np.nonzero(sizes == 4)[0]
    if len(quads):
        base = starts[quads]
        p = mesh.verts[mesh.loop_vert[base[:, None] + np.arange(4)]]
        d02 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
        d13 = np.linalg.norm(p[:, 1] - p[:, 3], axis=1)
        start_loop[quads] = np.where(d13 < d02 - 1e-12, base + 1, base)
    count = np.maximum(sizes - 2, 0)
    total = int(count.sum())
    out = np.empty((total, 3), np.int64)
    face_of_tri = np.repeat(np.arange(mesh.face_count), count)
    k = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
    first = starts[face_of_tri]
    n = sizes[face_of_tri]
    offset = start_loop[face_of_tri] - first
    out[:, 0] = first + (offset % n)
    out[:, 1] = first + ((offset + k + 1) % n)
    out[:, 2] = first + ((offset + k + 2) % n)
    return out, face_of_tri


def loop_normals(mesh: EditMesh) -> np.ndarray:
    """每个面角的法线：平滑面用周围平滑面的面积加权平均，平直面用面法线。"""
    fn = mesh.face_normals()
    lf = mesh.loop_face()
    out = fn[lf].copy() if mesh.loop_count else np.zeros((0, 3))
    smooth_faces = mesh.face_smooth
    if smooth_faces.any():
        area = mesh.face_areas()
        acc = np.zeros_like(mesh.verts)
        loops = np.nonzero(smooth_faces[lf])[0]
        np.add.at(acc, mesh.loop_vert[loops], (fn * area[:, None])[lf[loops]])
        length = np.linalg.norm(acc, axis=1, keepdims=True)
        vn = acc / np.maximum(length, 1e-30)
        out[loops] = vn[mesh.loop_vert[loops]]
    return out


def to_mesh_data(mesh: EditMesh, base=None, skip_hidden: bool = False):
    """编辑网格 → 绘制用的 MeshData（逐角点三角形，多边形大小记在 polygon_sizes）。
    skip_hidden 时隐藏的面不放进去（编辑模式里显示用）。"""
    from ..doc.meshio import MeshData

    tris, face_of_tri = triangulate(mesh)
    if skip_hidden and mesh.face_hide.any():
        shown = ~mesh.face_hide[face_of_tri]
        tris, face_of_tri = tris[shown], face_of_tri[shown]
    corner_loop = tris.reshape(-1)
    positions = mesh.verts[mesh.loop_vert[corner_loop]].astype(np.float32)
    normals = loop_normals(mesh)[corner_loop].astype(np.float32)
    uvs = mesh.loop_uv[corner_loop].astype(np.float32)
    material_ids = mesh.face_mat[face_of_tri].astype(np.int32)
    if len(positions):
        bmin, bmax = positions.min(axis=0), positions.max(axis=0)
    else:
        bmin = bmax = np.zeros(3, np.float32)
    data = MeshData(name=getattr(base, "name", "网格"), positions=positions, normals=normals, uvs=uvs,
                    material_ids=material_ids, materials=list(mesh.materials), bounds_min=bmin, bounds_max=bmax,
                    source_path=getattr(base, "source_path", ""), has_uvs=bool(mesh.has_uvs))
    sizes = mesh.face_sizes()
    if skip_hidden and mesh.face_hide.any():
        sizes = sizes[~mesh.face_hide]
    data.polygon_sizes = sizes[sizes >= 3].astype(np.int32)
    table = mesh.edges()
    loose = table.edges[table.loose] if len(table.edges) else np.zeros((0, 2), np.int64)
    data.loose_edges = mesh.verts[loose].astype(np.float32) if len(loose) else None
    return data, corner_loop
