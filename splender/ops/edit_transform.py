"""编辑模式里的 G/R/S：作用在选中的点上（变换的计算和物体模式共用 object_ops._TransformBase），
轴心点、法向坐标系、衰减编辑（O）在这里。"""
from __future__ import annotations

import heapq
import math

import numpy as np
from numba import njit

from ..doc.objects import D

FALLOFF = ("SMOOTH", "SPHERE", "ROOT", "INVERSE_SQUARE", "SHARP", "LINEAR", "CONSTANT", "RANDOM")


def to_disp(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, np.float64) @ D.T


def to_int(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, np.float64) @ D


def normal_basis(mesh, sel: np.ndarray) -> np.ndarray:
    """选中部分的法向坐标系（显示坐标，按列排 X、Y、Z）：Z 是平均法线。"""
    faces = np.nonzero(mesh.face_sel)[0]
    if len(faces):
        n = (mesh.face_normals()[faces] * mesh.face_areas()[faces, None]).sum(axis=0)
    elif len(sel):
        n = mesh.vert_normals()[sel].sum(axis=0)
    else:
        n = np.array([0.0, 1.0, 0.0])
    if np.linalg.norm(n) < 1e-12:
        n = np.array([0.0, 1.0, 0.0])
    z = to_disp(n / np.linalg.norm(n))
    # X 取和 Z 最不平行的世界轴投影到垂直面上（选中一条边时沿那条边）
    x = None
    if mesh.edge_sel.sum() == 1:
        a, b = mesh.edges().edges[np.nonzero(mesh.edge_sel)[0][0]]
        x = to_disp(mesh.verts[b] - mesh.verts[a])
        x = x - z * float(x @ z)
    if x is None or np.linalg.norm(x) < 1e-9:
        axis = np.eye(3)[int(np.argmin(np.abs(z)))]
        x = axis - z * float(axis @ z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def edit_pivot(op, ctx, mesh, disp: np.ndarray, sel: np.ndarray):
    """(轴心点, 各自原点的分组 [(点号, 中心)] 或 None)，显示坐标。"""
    mode = ctx.app.tool_settings.transform_pivot_point
    pts = disp[sel]
    if mode == "CURSOR":
        return np.asarray(ctx.app.project.cursor_location, np.float64), None
    if mode == "BOUNDING_BOX_CENTER":
        return (pts.min(axis=0) + pts.max(axis=0)) * 0.5, None
    if mode == "ACTIVE_ELEMENT" and mesh.active is not None:
        kind, index = mesh.active
        if kind == "VERT":
            return disp[index].copy(), None
        if kind == "EDGE":
            a, b = mesh.edges().edges[index]
            return (disp[a] + disp[b]) * 0.5, None
        if kind == "FACE":
            return to_disp(mesh.face_centers()[index]), None
    if mode == "INDIVIDUAL_ORIGINS" and mesh.face_sel.any():
        from ..mesh.topology import _face_islands

        groups = []
        for island in _face_islands(mesh, np.nonzero(mesh.face_sel)[0]):
            verts = np.unique(np.concatenate([mesh.face_verts(f) for f in island]))
            groups.append((verts, disp[verts].mean(axis=0)))
        return pts.mean(axis=0), groups
    return pts.mean(axis=0), None


def falloff(kind: str, d: np.ndarray, seed: int = 0) -> np.ndarray:
    """d 是 距离/半径（0..1），返回权重（1 在中心，0 在半径处）。和 Blender 的衰减曲线一样。"""
    f = np.clip(1.0 - d, 0.0, 1.0)
    if kind == "SMOOTH":
        return 3.0 * f * f - 2.0 * f * f * f
    if kind == "SPHERE":
        return np.sqrt(np.maximum(2.0 * f - f * f, 0.0))
    if kind == "ROOT":
        return np.sqrt(f)
    if kind == "INVERSE_SQUARE":
        return f * (2.0 - f)
    if kind == "SHARP":
        return f * f
    if kind == "CONSTANT":
        return (d < 1.0).astype(np.float64)
    if kind == "RANDOM":
        rng = np.random.default_rng(seed)
        return f * rng.random(len(f))
    return f


@njit(cache=True, nogil=True)
def _nearest_selected(points, sel_points, radius):
    """每个点到最近的选中点的距离（超过 radius 的记成 inf），用格子加速。"""
    n = points.shape[0]
    m = sel_points.shape[0]
    out = np.full(n, np.inf)
    if m == 0:
        return out
    inv = 1.0 / radius
    keys = np.empty(m, np.int64)
    base = 2097152
    for i in range(m):
        cx = int(math.floor(sel_points[i, 0] * inv)) + base // 2
        cy = int(math.floor(sel_points[i, 1] * inv)) + base // 2
        cz = int(math.floor(sel_points[i, 2] * inv)) + base // 2
        keys[i] = (cx * base + cy) * base + cz
    order = np.argsort(keys)
    sorted_keys = keys[order]
    r2 = radius * radius
    for i in range(n):
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        cx = int(math.floor(px * inv)) + base // 2
        cy = int(math.floor(py * inv)) + base // 2
        cz = int(math.floor(pz * inv)) + base // 2
        best = np.inf
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    key = ((cx + dx) * base + (cy + dy)) * base + (cz + dz)
                    lo = np.searchsorted(sorted_keys, key)
                    j = lo
                    while j < m and sorted_keys[j] == key:
                        s = order[j]
                        ddx = px - sel_points[s, 0]
                        ddy = py - sel_points[s, 1]
                        ddz = pz - sel_points[s, 2]
                        d2 = ddx * ddx + ddy * ddy + ddz * ddz
                        if d2 < best:
                            best = d2
                        j += 1
        if best <= r2:
            out[i] = math.sqrt(best)
    return out


def proportional_weights(mesh, disp: np.ndarray, sel: np.ndarray, radius: float, kind: str,
                         connected: bool) -> tuple[np.ndarray, np.ndarray]:
    """衰减编辑：(受影响的没选中的点, 它们的权重)。connected 时沿边量距离，只影响相连的。"""
    radius = max(float(radius), 1e-9)
    chosen = np.zeros(mesh.vert_count, bool)
    chosen[sel] = True
    candidates = np.nonzero(~chosen & ~mesh.vert_hide)[0]
    if not len(candidates):
        return np.zeros(0, np.int64), np.zeros(0)
    if connected:
        dist = _geodesic(mesh, disp, sel, radius)
        d = dist[candidates]
    else:
        d = _nearest_selected(np.ascontiguousarray(disp[candidates]), np.ascontiguousarray(disp[sel]), radius)
    inside = np.isfinite(d) & (d < radius)
    idx = candidates[inside]
    return idx, falloff(kind, d[inside] / radius)


def _geodesic(mesh, disp, sel, radius) -> np.ndarray:
    table = mesh.edges()
    adj: dict[int, list[tuple[int, float]]] = {}
    lengths = np.linalg.norm(disp[table.edges[:, 0]] - disp[table.edges[:, 1]], axis=1)
    for (a, b), length in zip(table.edges, lengths):
        adj.setdefault(int(a), []).append((int(b), float(length)))
        adj.setdefault(int(b), []).append((int(a), float(length)))
    dist = np.full(mesh.vert_count, np.inf)
    heap = []
    for s in sel:
        dist[s] = 0.0
        heap.append((0.0, int(s)))
    heapq.heapify(heap)
    while heap:
        d, v = heapq.heappop(heap)
        if d > dist[v] or d > radius:
            continue
        for w, length in adj.get(v, ()):
            nd = d + length
            if nd < dist[w] and nd <= radius:
                dist[w] = nd
                heapq.heappush(heap, (nd, w))
    return dist


def apply_edit_delta(op, delta: np.ndarray) -> None:
    """把显示坐标里的变换 delta（绕 op.pivot）用到选中的点上（各自原点时每组绕自己的中心），衰减的点按权重跟着动。"""
    session = op.edit
    mesh = session.mesh
    disp0 = op.edit_disp
    disp = disp0.copy()
    linear = delta[:3, :3]
    shift = delta[:3, 3] - (op.pivot - linear @ op.pivot)       # 平移部分（旋转缩放时是 0）
    if op.edit_islands:
        for verts, center in op.edit_islands:
            disp[verts] = disp0[verts] @ linear.T + (center - linear @ center + shift)
    else:
        disp[op.edit_sel] = disp0[op.edit_sel] @ linear.T + delta[:3, 3]
    if op.prop_idx is not None and len(op.prop_idx):
        full = disp0[op.prop_idx] @ linear.T + delta[:3, 3]
        disp[op.prop_idx] = disp0[op.prop_idx] + (full - disp0[op.prop_idx]) * op.prop_weight[:, None]
    mesh.verts = to_int(disp)
    mesh.geometry_version += 1
    session.sync(topology=False)
