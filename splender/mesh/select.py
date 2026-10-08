"""编辑模式里的拾取和区域选择：屏幕上离鼠标最近的点、边，鼠标下的面；框、圈、套索里的元素。
不开 X 光时被挡住的点不算（和 Blender 一样），用几何缓冲里的深度判断。"""
from __future__ import annotations

import numpy as np

from .editmesh import EditMesh, triangulate


def project(view, points: np.ndarray):
    """内部坐标的点 → (屏幕 x, 屏幕 y, 在相机前面, 到眼睛的距离)。屏幕坐标是视口像素。"""
    camera = view.camera
    vp = camera.view_projection(view.aspect)
    p = np.asarray(points, np.float64).reshape(-1, 3)
    clip = p @ vp[:3, :3].T + vp[:3, 3]
    w = p @ vp[3, :3] + vp[3, 3]
    clip = np.concatenate([clip, w[:, None]], axis=1)
    front = w > 1e-9
    safe = np.where(front, w, 1.0)
    sx = (clip[:, 0] / safe * 0.5 + 0.5) * view.width
    sy = (0.5 - clip[:, 1] / safe * 0.5) * view.height
    if camera.ortho:
        dist = (p - camera.eye) @ camera.forward
    else:
        dist = np.linalg.norm(p - camera.eye, axis=1)
    return sx, sy, front, dist


class DepthProbe:
    """一次读回整个几何缓冲的位置，判断点有没有被挡住。"""

    def __init__(self, engine, view) -> None:
        self.view = view
        engine.ctx_make_current()
        pos, _nrm = view.read_surface(0, 0, view.width - 1, view.height - 1)
        self.pos = pos
        self.camera = view.camera

    def visible(self, points: np.ndarray, sx: np.ndarray, sy: np.ndarray, dist: np.ndarray) -> np.ndarray:
        view = self.view
        ix = np.clip(sx.astype(np.int64), 0, view.width - 1)
        iy = np.clip(sy.astype(np.int64), 0, view.height - 1)
        surf = self.pos[iy, ix]
        empty = surf[:, 3] < 0.5
        camera = self.camera
        if camera.ortho:
            surf_dist = (surf[:, :3] - camera.eye) @ camera.forward
        else:
            surf_dist = np.linalg.norm(surf[:, :3] - camera.eye, axis=1)
        tan_y, _tan_x = camera.tan_half(view.aspect)
        span = np.full(len(dist), camera.distance) if camera.ortho else np.maximum(np.abs(dist), 1e-6)
        per_px = 2.0 * span * tan_y / max(1, view.height)          # 一个像素多少世界长度
        tolerance = np.maximum(np.abs(dist) * 2e-3, per_px * 3.0)
        return empty | (dist <= surf_dist + tolerance)


def _visible_filter(engine, view, mesh_points, sx, sy, dist, front, xray: bool, inside: np.ndarray,
                    probe=None) -> np.ndarray:
    ok = inside & front
    if xray or not ok.any():
        return ok
    probe = probe if probe is not None else DepthProbe(engine, view)
    idx = np.nonzero(ok)[0]
    vis = probe.visible(mesh_points[idx], sx[idx], sy[idx], dist[idx])
    out = np.zeros(len(ok), bool)
    out[idx[vis]] = True
    return out


def pick(engine, view, mesh: EditMesh, x: float, y: float, xray: bool, radius_px: float = 30.0, probe=None):
    """鼠标下的元素：按选择模式，点模式找最近的点、边模式找最近的边、面模式找鼠标下的面。
    返回 ("VERT"|"EDGE"|"FACE", 编号) 或 None。"""
    modes = mesh.select_mode
    best = None
    best_d = radius_px
    if "VERT" in modes:
        found = _nearest_vert(engine, view, mesh, x, y, xray, radius_px, probe=probe)
        if found is not None and found[1] < best_d:
            best, best_d = ("VERT", found[0]), found[1]
    if "EDGE" in modes:
        found = _nearest_edge(engine, view, mesh, x, y, xray, radius_px, probe=probe)
        if found is not None and (best is None or found[1] < best_d * 0.75):
            best, best_d = ("EDGE", found[0]), found[1]
    if "FACE" in modes and best is None:
        face = face_under(view, mesh, x, y)
        if face is not None:
            best = ("FACE", face)
    return best


def _closest(candidates: np.ndarray, screen_d: np.ndarray, eye_d: np.ndarray) -> int:
    """屏幕上离鼠标最近的；差不到一个像素的几个（比如前后重叠的点）取离眼睛近的。"""
    best = float(screen_d[candidates].min())
    tied = candidates[screen_d[candidates] <= best + 1.0]
    return int(tied[np.argmin(eye_d[tied])])


def _nearest_vert(engine, view, mesh, x, y, xray, radius_px, probe=None):
    shown = np.nonzero(~mesh.vert_hide)[0]
    if not len(shown):
        return None
    sx, sy, front, dist = project(view, mesh.verts[shown])
    d = np.hypot(sx - x, sy - y)
    near = front & (d <= radius_px)
    if not near.any():
        return None
    vis = _visible_filter(engine, view, mesh.verts[shown], sx, sy, dist, front, xray, near, probe)
    if not vis.any():
        return None
    k = _closest(np.nonzero(vis)[0], d, dist)
    return int(shown[k]), float(d[k])


def _nearest_edge(engine, view, mesh, x, y, xray, radius_px, probe=None):
    table = mesh.edges()
    if not len(table.edges):
        return None
    e = table.edges
    shown = ~(mesh.vert_hide[e[:, 0]] | mesh.vert_hide[e[:, 1]])
    ids = np.nonzero(shown)[0]
    if not len(ids):
        return None
    ax, ay, af, ad = project(view, mesh.verts[e[ids, 0]])
    bx, by, bf, bd = project(view, mesh.verts[e[ids, 1]])
    vx, vy = bx - ax, by - ay
    length2 = np.maximum(vx * vx + vy * vy, 1e-12)
    t = np.clip(((x - ax) * vx + (y - ay) * vy) / length2, 0.0, 1.0)
    px, py = ax + vx * t, ay + vy * t
    d = np.hypot(px - x, py - y)
    near = af & bf & (d <= radius_px)
    if not near.any():
        return None
    mid = (mesh.verts[e[ids, 0]] + mesh.verts[e[ids, 1]]) * 0.5
    mx, my, mf, md = project(view, mid)
    vis = _visible_filter(engine, view, mid, mx, my, md, mf, xray, near, probe)
    if not vis.any():
        return None
    k = _closest(np.nonzero(vis)[0], d, md)
    return int(ids[k]), float(d[k])


def _triangles(mesh: EditMesh):
    """面拆成的三角形（拓扑不变时复用，悬停拾取每帧都要用）。"""
    cache = getattr(mesh, "_pick_tris", None)
    key = (mesh.topology_version, mesh.face_count, mesh.loop_count)
    if cache is None or cache[0] != key:
        cache = (key, *triangulate(mesh))
        mesh._pick_tris = cache
    return cache[1], cache[2]


def face_under(view, mesh: EditMesh, x: float, y: float):
    """鼠标视线最先打到的面（隐藏的面不算）。"""
    from ..bake.parts import ray_hit

    if not mesh.face_count:
        return None
    tris, face_of_tri = _triangles(mesh)
    shown = ~mesh.face_hide[face_of_tri]
    if not shown.any():
        return None
    tris = tris[shown]
    face_of_tri = face_of_tri[shown]
    corners = mesh.verts[mesh.loop_vert[tris.reshape(-1)]].astype(np.float32)
    origin, direction = view.camera.ray(x, y, view.width, view.height)
    tri, _u, _v, _t = ray_hit(corners, origin, direction)
    if tri < 0:
        return None
    return int(face_of_tri[tri])


def elements_in(engine, view, mesh: EditMesh, test, xray: bool) -> dict:
    """按屏幕上的形状（test(sx, sy) → 布尔数组）找元素：{"VERT": 点号, "EDGE": 边号, "FACE": 面号}。
    边要两端都在里面，面看中心点（和 Blender 面模式的面点一样）。"""
    out = {}
    shown = np.nonzero(~mesh.vert_hide)[0]
    sx, sy, front, dist = project(view, mesh.verts[shown])
    inside_v = test(sx, sy)
    vis_v = _visible_filter(engine, view, mesh.verts[shown], sx, sy, dist, front, xray, inside_v)
    chosen = np.zeros(mesh.vert_count, bool)
    chosen[shown[vis_v]] = True
    out["VERT"] = np.nonzero(chosen)[0]
    table = mesh.edges()
    if len(table.edges):
        out["EDGE"] = np.nonzero(chosen[table.edges[:, 0]] & chosen[table.edges[:, 1]])[0]
    else:
        out["EDGE"] = np.zeros(0, np.int64)
    if mesh.face_count:
        centers = mesh.face_centers()
        faces = np.nonzero(~mesh.face_hide)[0]
        cx, cy, cf, cd = project(view, centers[faces])
        inside_f = test(cx, cy)
        vis_f = _visible_filter(engine, view, centers[faces], cx, cy, cd, cf, xray, inside_f)
        out["FACE"] = faces[vis_f]
    else:
        out["FACE"] = np.zeros(0, np.int64)
    return out


def apply_region(mesh: EditMesh, found: dict, mode: str) -> None:
    """按选择方式（SET/ADD/SUB/XOR/AND）把区域里的元素并进选中，再按选择模式推平。"""
    if "VERT" in mesh.select_mode:
        kind, current = "VERT", mesh.vert_sel
    elif "EDGE" in mesh.select_mode:
        kind, current = "EDGE", mesh.edge_sel
    else:
        kind, current = "FACE", mesh.face_sel
    inside = np.zeros(len(current), bool)
    inside[found[kind]] = True
    if mode == "SET":
        new = inside
    elif mode == "ADD":
        new = current | inside
    elif mode == "SUB":
        new = current & ~inside
    elif mode == "XOR":
        new = current ^ inside
    else:
        new = current & inside
    if kind == "VERT":
        mesh.vert_sel = new
        mesh.flush_from_verts()
    elif kind == "EDGE":
        mesh.edge_sel = new
        mesh.flush_from_edges()
    else:
        mesh.face_sel = new
        mesh.flush_from_faces()


def select_element(mesh: EditMesh, element, toggle: bool = False, extend: bool = False,
                   deselect: bool = False) -> None:
    """点选一个元素：替换选中、Shift 切换（选中的当前元素再点取消，选中的非当前元素变成当前）。"""
    kind, index = element
    arrays = {"VERT": "vert_sel", "EDGE": "edge_sel", "FACE": "face_sel"}
    sel = getattr(mesh, arrays[kind])
    if not (toggle or extend or deselect):
        mesh.vert_sel[:] = False
        mesh.edge_sel[:] = False
        mesh.face_sel[:] = False
        sel = getattr(mesh, arrays[kind])
        sel[index] = True
        mesh.active = (kind, int(index))
    elif deselect:
        sel[index] = False
    elif toggle and sel[index] and mesh.active == (kind, int(index)):
        sel[index] = False
        mesh.active = None
    else:
        sel[index] = True
        mesh.active = (kind, int(index))
    {"VERT": mesh.flush_from_verts, "EDGE": mesh.flush_from_edges, "FACE": mesh.flush_from_faces}[kind]()


def linked(mesh: EditMesh, seeds_verts: np.ndarray, delimit_seam: bool = False) -> np.ndarray:
    """从这些点出发，沿边连着的所有点（同一块）。"""
    table = mesh.edges()
    parent = np.arange(mesh.vert_count)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in table.edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(mesh.vert_count)])
    wanted = np.unique(roots[np.asarray(seeds_verts, np.int64)])
    return np.nonzero(np.isin(roots, wanted) & ~mesh.vert_hide)[0]


def grow(mesh: EditMesh, steps: int = 1) -> None:
    """Ctrl+小键盘加：选中向外扩一圈。"""
    table = mesh.edges()
    for _ in range(max(1, steps)):
        if "FACE" in mesh.select_mode and "VERT" not in mesh.select_mode:
            lf = mesh.loop_face()
            touched = np.zeros(mesh.vert_count, bool)
            touched[mesh.loop_vert[mesh.face_sel[lf]]] = True
            mesh.face_sel = np.logical_or.reduceat(touched[mesh.loop_vert], mesh.face_start[:-1]) & ~mesh.face_hide
            mesh.flush_from_faces()
        else:
            sel = mesh.vert_sel.copy()
            e = table.edges
            sel[e[mesh.vert_sel[e[:, 0]], 1]] = True
            sel[e[mesh.vert_sel[e[:, 1]], 0]] = True
            mesh.vert_sel = sel & ~mesh.vert_hide
            mesh.flush_from_verts()


def shrink(mesh: EditMesh, steps: int = 1) -> None:
    """Ctrl+小键盘减：选中向里收一圈（边界上的点不选）。"""
    table = mesh.edges()
    e = table.edges
    for _ in range(max(1, steps)):
        sel = mesh.vert_sel.copy()
        outside = ~mesh.vert_sel
        sel[e[outside[e[:, 0]], 1]] = False
        sel[e[outside[e[:, 1]], 0]] = False
        mesh.vert_sel = sel
        mesh.flush_from_verts()
