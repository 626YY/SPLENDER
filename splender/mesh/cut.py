"""刀切（K）、连接点（J）：沿一串点把穿过的面切开（照 Blender 的切刀）。

刀点有三种：正好在点上（VERT）、在边上（EDGE，带边上的参数 t）、在面里（FACE）。两个相邻刀点之间，
切割面（过眼睛和这两点的平面）和网格的交点就是这一段切到的边、点；按屏幕上的先后排成一串。
同一个面里一前一后两个落在面边界上的刀点之间连一刀，把面分成两个；没连到边界的一段留成散边。
"""
from __future__ import annotations

import numpy as np

from .editmesh import EditMesh
from .topology import FaceList, rebuild


def point(kind: str, index: int, pos, t: float = 0.0) -> dict:
    return {"kind": kind, "index": int(index), "pos": np.asarray(pos, np.float64), "t": float(t)}


# ====================================================================== 两刀点之间切到什么
def segment_hits(mesh: EditMesh, view, a: dict, b: dict, probe=None, only_selected: bool = False) -> list[dict]:
    """刀点 a、b 之间切到的边和点（不含 a、b），按屏幕上从 a 到 b 的顺序。probe 给了就去掉被挡住的。"""
    from .select import project

    camera = view.camera
    pa, pb = a["pos"], b["pos"]
    seg = pb - pa
    if np.linalg.norm(seg) < 1e-12 or not mesh.vert_count:
        return []
    if camera.ortho:
        normal = np.cross(seg, np.asarray(camera.forward, np.float64))
    else:
        normal = np.cross(seg, np.asarray(camera.eye, np.float64) - pa)
    length = np.linalg.norm(normal)
    if length < 1e-12:
        return []
    normal /= length
    sa = project(view, pa[None])
    sb = project(view, pb[None])
    ax, ay, bx, by = float(sa[0][0]), float(sa[1][0]), float(sb[0][0]), float(sb[1][0])
    vx, vy = bx - ax, by - ay
    screen_len = max(float(np.hypot(vx, vy)), 1e-9)
    margin = 0.5 / screen_len                       # 离两头不到半个像素的不算

    def screen_param(points):
        sx, sy, front, dist = project(view, points)
        s = ((sx - ax) * vx + (sy - ay) * vy) / (screen_len * screen_len)
        ok = front & (s > margin) & (s < 1.0 - margin)
        if probe is not None and ok.any():
            idx = np.nonzero(ok)[0]
            ok[idx] &= probe.visible(points[idx], sx[idx], sy[idx], dist[idx])
        return s, ok

    scale = max(float(np.abs(mesh.verts).max()), 1.0)
    eps = 1e-7 * scale
    d = (mesh.verts - pa) @ normal
    table = mesh.edges()
    allowed_v = ~mesh.vert_hide
    allowed_e = None
    if only_selected or mesh.face_hide.any():
        faces_ok = ~mesh.face_hide
        if only_selected:
            faces_ok = faces_ok & mesh.face_sel
        lf = mesh.loop_face()
        allowed_e = np.zeros(len(table.edges), bool)
        allowed_v = np.zeros(mesh.vert_count, bool)
        if mesh.loop_count:
            allowed_e[table.loop_edge[faces_ok[lf]]] = True
            allowed_v[mesh.loop_vert[faces_ok[lf]]] = True
    skip_v = {x["index"] for x in (a, b) if x["kind"] == "VERT"}
    skip_e = {x["index"] for x in (a, b) if x["kind"] == "EDGE"}
    hits: list[dict] = []
    # 正好在切割面上的点
    on_plane = np.nonzero((np.abs(d) <= eps) & allowed_v)[0]
    if len(on_plane):
        s, ok = screen_param(mesh.verts[on_plane])
        for k in np.nonzero(ok)[0]:
            v = int(on_plane[k])
            if v not in skip_v:
                hits.append({**point("VERT", v, mesh.verts[v]), "s": float(s[k])})
    # 穿过切割面的边
    e = table.edges
    if len(e):
        d0, d1 = d[e[:, 0]], d[e[:, 1]]
        cross = (np.abs(d0) > eps) & (np.abs(d1) > eps) & ((d0 > 0) != (d1 > 0))
        if allowed_e is not None:
            cross &= allowed_e
        ids = np.nonzero(cross)[0]
        if len(ids):
            t = d0[ids] / (d0[ids] - d1[ids])
            pos = mesh.verts[e[ids, 0]] + (mesh.verts[e[ids, 1]] - mesh.verts[e[ids, 0]]) * t[:, None]
            s, ok = screen_param(pos)
            for k in np.nonzero(ok)[0]:
                edge = int(ids[k])
                if edge not in skip_e:
                    hits.append({**point("EDGE", edge, pos[k], float(t[k])), "s": float(s[k])})
    hits.sort(key=lambda h: h["s"])
    return hits


# ====================================================================== 切
def _faces_of(mesh: EditMesh, p: dict, lf: np.ndarray, loop_edge: np.ndarray) -> set:
    if p["kind"] == "FACE":
        return {int(p["index"])}
    if p["kind"] == "EDGE":
        return set(lf[loop_edge == p["index"]].tolist())
    return set(lf[mesh.loop_vert == p["index"]].tolist())


def _interp_uv(mesh: EditMesh, face: int, pos: np.ndarray) -> np.ndarray:
    """面里一点的 UV：投到面的平面上，按扇形三角形的重心坐标插值。"""
    s, e = mesh.face_start[face], mesh.face_start[face + 1]
    verts = mesh.verts[mesh.loop_vert[s:e]]
    uvs = mesh.loop_uv[s:e].astype(np.float64)
    n = mesh.face_normals()[face]
    best, best_err = uvs.mean(axis=0), np.inf
    for k in range(1, len(verts) - 1):
        a, b, c = verts[0], verts[k], verts[k + 1]
        v0, v1, v2 = b - a, c - a, pos - a
        v2 = v2 - n * float(v2 @ n)
        d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
        d20, d21 = v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-30:
            continue
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        w0 = 1.0 - w1 - w2
        err = max(0.0, -min(w0, w1, w2))
        if err < best_err:
            best_err = err
            best = uvs[0] * w0 + uvs[k] * w1 + uvs[k + 1] * w2
    return np.asarray(best, np.float64)


def _runs(mesh: EditMesh, chains: list[list[dict]], lf: np.ndarray, loop_edge: np.ndarray, edges: np.ndarray):
    """把刀点串分成一刀一刀：(面, 刀点串)，串的两头在面的边界上（在面里的一头是悬空的）。"""
    runs = []
    for chain in chains:
        run, run_face = None, None
        for i in range(len(chain) - 1):
            p, q = chain[i], chain[i + 1]
            common = _faces_of(mesh, p, lf, loop_edge) & _faces_of(mesh, q, lf, loop_edge)
            if p["kind"] != "FACE" and q["kind"] != "FACE":
                along_edge = len(common) != 1
                if p["kind"] == "VERT" and q["kind"] == "VERT":
                    a, b = sorted((p["index"], q["index"]))
                    along_edge |= bool(np.any((edges[:, 0] == a) & (edges[:, 1] == b)))
                if along_edge:
                    common = set()
            if not common:
                if run:
                    runs.append((run_face, run))
                run, run_face = None, None
                continue
            face = next(iter(common))
            if run is not None and run_face == face and run[-1] is p:
                run.append(q)
            else:
                if run:
                    runs.append((run_face, run))
                run, run_face = [p, q], face
            if q["kind"] != "FACE":
                runs.append((run_face, run))
                run, run_face = None, None
        if run:
            runs.append((run_face, run))
    return runs


def knife(mesh: EditMesh, chains: list[list[dict]]) -> np.ndarray:
    """沿几串刀点切开网格。返回切出来的点（已选中，切出来的边也选中）。"""
    table = mesh.edges()
    lf = mesh.loop_face()
    loop_edge = table.loop_edge
    base = mesh.vert_count
    new_pos: list[np.ndarray] = []
    vid: dict[int, int] = {}                     # 刀点 → 点号（同一个刀点对象只建一个点）

    def vertex_of(p: dict) -> int:
        key = id(p)
        if key not in vid:
            if p["kind"] == "VERT":
                vid[key] = int(p["index"])
            else:
                vid[key] = base + len(new_pos)
                new_pos.append(np.asarray(p["pos"], np.float64))
        return vid[key]

    # 边上的刀点：插进边里（按边上的参数排）
    edge_points: dict[int, dict[int, float]] = {}
    for chain in chains:
        for p in chain:
            if p["kind"] == "EDGE":
                edge_points.setdefault(int(p["index"]), {})[vertex_of(p)] = float(p["t"])
    ordered = {e: sorted(((t, x) for x, t in pts.items())) for e, pts in edge_points.items()}
    runs = _runs(mesh, chains, lf, loop_edge, table.edges)
    touched = set()
    for e in ordered:
        touched.update(lf[loop_edge == e].tolist())
    for face, _run in runs:
        touched.add(int(face))
    # 碰到的面换成点串（边上的刀点插进去，UV 沿边插值）
    polys: dict[int, list[list[tuple[int, np.ndarray]]]] = {}
    nxt = mesh.loop_next()
    for f in sorted(touched):
        s, e = mesh.face_start[f], mesh.face_start[f + 1]
        out = []
        for loop in range(s, e):
            v = int(mesh.loop_vert[loop])
            uv = mesh.loop_uv[loop].astype(np.float64)
            out.append((v, uv))
            edge = int(loop_edge[loop])
            if edge in ordered:
                uv_w = mesh.loop_uv[nxt[loop]].astype(np.float64)
                forward = int(table.edges[edge][0]) == v
                pts = ordered[edge] if forward else [(1.0 - t, x) for t, x in reversed(ordered[edge])]
                for t, x in pts:
                    out.append((x, uv * (1.0 - t) + uv_w * t))
        polys[f] = [out]
    # 一刀一刀切
    loose: list[tuple[int, int]] = []
    cut_pairs: set[tuple[int, int]] = set()
    for face, run in runs:
        ids = [vertex_of(p) for p in run]
        pairs = [(ids[k], ids[k + 1]) for k in range(len(ids) - 1) if ids[k] != ids[k + 1]]
        if run[0]["kind"] == "FACE" or run[-1]["kind"] == "FACE":
            loose.extend(pairs)                       # 没切到边界的一段留成散边
            cut_pairs.update((min(a, b), max(a, b)) for a, b in pairs)
            continue
        u, w = ids[0], ids[-1]
        if u == w:
            continue
        pieces = polys.get(int(face))
        if not pieces:
            continue
        for index, poly in enumerate(pieces):
            verts = [v for v, _uv in poly]
            if u in verts and w in verts:
                break
        else:
            continue
        i, j = verts.index(u), verts.index(w)
        n = len(poly)
        inner = [(ids[k], _interp_uv(mesh, int(face), run[k]["pos"])) for k in range(1, len(run) - 1)]
        if not inner and (abs(i - j) == 1 or abs(i - j) == n - 1):
            continue                                  # 两点本来就相邻
        first = [poly[(i + k) % n] for k in range((j - i) % n + 1)] + inner[::-1]
        second = [poly[(j + k) % n] for k in range((i - j) % n + 1)] + inner
        pieces[index:index + 1] = [first, second]
        cut_pairs.update((min(a, b), max(a, b)) for a, b in pairs)
    # 重建
    new = FaceList()
    remove = np.zeros(mesh.face_count, bool)
    for f, pieces in polys.items():
        remove[f] = True
        for poly in pieces:
            new.add([v for v, _uv in poly], [uv for _v, uv in poly], mesh.face_mat[f], mesh.face_smooth[f])
    loose_edges = [tuple(int(x) for x in pair) for pair in mesh.loose_edges]
    for e, pts in ordered.items():
        a, b = (int(x) for x in table.edges[e])
        if (a, b) in loose_edges or (b, a) in loose_edges:
            # 散边上的刀点把散边拆成几段
            loose_edges = [pair for pair in loose_edges if pair not in ((a, b), (b, a))]
            chain_ids = [a] + [x for _t, x in pts] + [b]
            loose_edges += [(chain_ids[k], chain_ids[k + 1]) for k in range(len(chain_ids) - 1)]
    loose_edges += loose
    rebuild(mesh, ~remove, new, np.asarray(new_pos).reshape(-1, 3) if new_pos else None,
            loose_edges=np.asarray(loose_edges, np.int64).reshape(-1, 2), compact=False)
    cut_ids = np.asarray(sorted(set(vid.values())), np.int64)
    mesh.select_all("DESELECT")
    if len(cut_ids):
        table = mesh.edges()
        v = max(1, mesh.vert_count)
        keys = table.edges[:, 0] * v + table.edges[:, 1]
        want = np.asarray([a * v + b for a, b in cut_pairs], np.int64)
        mesh.edge_sel = np.isin(keys, want)
        mesh.vert_sel[cut_ids] = True
        if mesh.edge_sel.any():
            mesh.vert_sel[table.edges[mesh.edge_sel].reshape(-1)] = True
    return cut_ids


# ====================================================================== 连接点（J）
def connect_verts(mesh: EditMesh) -> int:
    """选中的点里，同一个面上、又不相邻的点之间连一刀（把面分开）。返回切了几刀。"""
    sel = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
    if len(sel) < 2:
        return 0
    chosen = np.zeros(mesh.vert_count, bool)
    chosen[sel] = True
    lf = mesh.loop_face()
    counts = np.bincount(lf[chosen[mesh.loop_vert]], minlength=mesh.face_count) if mesh.loop_count else \
        np.zeros(0, np.int64)
    faces = np.nonzero((counts >= 2) & ~mesh.face_hide)[0]
    shared: dict[int, dict] = {}

    def p_of(v: int) -> dict:
        if v not in shared:
            shared[v] = point("VERT", v, mesh.verts[v])
        return shared[v]

    chains = []
    for f in faces:
        verts = [int(v) for v in mesh.face_verts(f)]
        n = len(verts)
        picked = [k for k, v in enumerate(verts) if chosen[v]]
        links = list(zip(picked, picked[1:] + picked[:1])) if len(picked) > 2 else [tuple(picked)]
        for i, j in links:
            if abs(i - j) in (0, 1, n - 1):
                continue
            chains.append([p_of(verts[i]), p_of(verts[j])])
    if not chains:
        return 0
    knife(mesh, chains)
    mesh.select_all("DESELECT")
    mesh.vert_sel[sel] = True
    mesh.flush_from_verts()
    return len(chains)
