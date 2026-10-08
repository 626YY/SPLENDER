"""倒角（Ctrl+B，照 Blender 的倒角边）：选中的边向两侧各退进 width，原来的棱换成一条（或几段圆弧形的）斜面；
几条倒角边在点上汇合时补一个面把角封上。

做法：每个面在倒角点上的那个角，按它两条边是否倒角换成新点——
- 两条边都不倒角：这个角换成两条边上各自滑出的点；
- 一条倒角：换成另一条边上滑出的点（倒角面的边线就落在这里）；
- 两条都倒角：换成两条偏移线在面内的交点。
每条倒角边两侧的新点连成倒角面；段数大于 1 时按圆弧（以原来的点为控制点）细分。最后把点上剩下的洞补上。
"""
from __future__ import annotations

import numpy as np

from .editmesh import EditMesh
from .topology import FaceList, rebuild


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-30 else v


def bevel_edges(mesh: EditMesh, width: float, segments: int = 1, clamp: bool = True) -> dict:
    table = mesh.edges()
    chosen = np.nonzero(mesh.edge_sel & (table.face_count == 2))[0]
    if not len(chosen) or width <= 0:
        return {"faces": np.zeros(0, np.int64)}
    segments = max(1, int(segments))
    bevel = np.zeros(len(table.edges), bool)
    bevel[chosen] = True
    lf = mesh.loop_face()
    nxt = mesh.loop_next()
    prv = mesh.loop_prev()
    verts = mesh.verts
    fn = mesh.face_normals()
    bevel_verts = np.zeros(mesh.vert_count, bool)
    bevel_verts[table.edges[chosen].reshape(-1)] = True
    # 宽度不超过相邻边长的一半（防止新点跑出边外，和 Blender 的「限制重叠」一样）
    if clamp:
        lengths = np.linalg.norm(verts[table.edges[:, 0]] - verts[table.edges[:, 1]], axis=1)
        touching = bevel_verts[table.edges[:, 0]] | bevel_verts[table.edges[:, 1]]
        if touching.any():
            width = min(width, float(lengths[touching].min()) * 0.49)
    new_pos: list[np.ndarray] = []
    base = mesh.vert_count

    def add_point(p) -> int:
        new_pos.append(np.asarray(p, np.float64))
        return base + len(new_pos) - 1

    slide: dict[tuple[int, int], int] = {}        # (点, 没倒角的边) → 滑出的新点
    corner_point: dict[int, list[int]] = {}       # 面角号 → 换成的新点（按面的走向）

    def slide_point(v: int, e: int, beveled_neighbor_dir) -> int:
        key = (v, e)
        if key in slide:
            return slide[key]
        a, b = table.edges[e]
        other = int(b) if int(a) == v else int(a)
        direction = verts[other] - verts[v]
        length = np.linalg.norm(direction)
        direction = direction / max(length, 1e-30)
        if beveled_neighbor_dir is not None:
            s = np.linalg.norm(np.cross(direction, beveled_neighbor_dir))
            d = width / max(s, 0.2)
        else:
            d = width
        d = min(d, length * 0.98)
        slide[key] = add_point(verts[v] + direction * d)
        return slide[key]

    for loop in range(mesh.loop_count):
        v = int(mesh.loop_vert[loop])
        if not bevel_verts[v]:
            continue
        f = int(lf[loop])
        e_out = int(table.loop_edge[loop])               # v → next
        e_in = int(table.loop_edge[prv[loop]])           # prev → v
        p_next = verts[mesh.loop_vert[nxt[loop]]]
        p_prev = verts[mesh.loop_vert[prv[loop]]]
        d_out = _unit(p_next - verts[v])
        d_in = _unit(p_prev - verts[v])
        if bevel[e_in] and bevel[e_out]:
            # 两条偏移线（各自往面里退 width）的交点
            n = fn[f]
            off_out = _unit(np.cross(n, d_out)) * width       # 出边左侧（面内）
            off_in = _unit(np.cross(d_in, n)) * width         # 入边面内一侧
            p = _line_intersection(verts[v] + off_out, d_out, verts[v] + off_in, d_in)
            corner_point[loop] = [add_point(p)]
        elif bevel[e_out]:
            corner_point[loop] = [slide_point(v, e_in, d_out)]
        elif bevel[e_in]:
            corner_point[loop] = [slide_point(v, e_out, d_in)]
        else:
            corner_point[loop] = [slide_point(v, e_in, None), slide_point(v, e_out, None)]
    # 原来的面：倒角点上的角换成新点
    new = FaceList()
    for f in range(mesh.face_count):
        s, e = mesh.face_start[f], mesh.face_start[f + 1]
        loops = range(s, e)
        if not any(loop in corner_point for loop in loops):
            new.add(mesh.loop_vert[s:e], mesh.loop_uv[s:e], mesh.face_mat[f], mesh.face_smooth[f],
                    bool(mesh.face_sel[f]))
            continue
        out_v, out_uv = [], []
        n = e - s
        for k, loop in enumerate(loops):
            if loop in corner_point:
                pts = corner_point[loop]
                for j, p in enumerate(pts):
                    out_v.append(p)
                    # UV：往相邻面角挪一点
                    if len(pts) == 2:
                        other = s + (k - 1) % n if j == 0 else s + (k + 1) % n
                    else:
                        other = loop
                    t = 0.0 if other == loop else 0.15
                    out_uv.append(mesh.loop_uv[loop] * (1 - t) + mesh.loop_uv[other] * t)
            else:
                out_v.append(int(mesh.loop_vert[loop]))
                out_uv.append(mesh.loop_uv[loop])
        new.add(out_v, out_uv, mesh.face_mat[f], mesh.face_smooth[f], False)
    # 倒角面
    pos_all = lambda i: verts[i] if i < base else new_pos[i - base]  # noqa: E731
    bevel_faces = []
    edge_loops: dict[int, list[int]] = {}
    for loop in range(mesh.loop_count):
        e = int(table.loop_edge[loop])
        if bevel[e]:
            edge_loops.setdefault(e, []).append(loop)
    for e, loops in edge_loops.items():
        if len(loops) != 2:
            continue
        l1, l2 = loops                                       # l1：a → b，l2：b → a
        a, b = int(mesh.loop_vert[l1]), int(mesh.loop_vert[nxt[l1]])
        f1_a = _side(corner_point[l1], 0)                    # 面 1 在 a 处、靠这条边的新点
        f1_b = _side(corner_point[nxt[l1]], -1)
        f2_b = _side(corner_point[l2], 0)
        f2_a = _side(corner_point[nxt[l2]], -1)
        uv1a, uv1b = mesh.loop_uv[l1], mesh.loop_uv[nxt[l1]]
        uv2b, uv2a = mesh.loop_uv[l2], mesh.loop_uv[nxt[l2]]
        mat, smooth = mesh.face_mat[lf[l1]], mesh.face_smooth[lf[l1]] or segments > 1
        if segments == 1:
            new.add([f1_b, f1_a, f2_a, f2_b], [uv1b, uv1a, uv2a, uv2b], mat, smooth, True)
            bevel_faces.append(len(new) - 1)
            continue
        # 圆弧：两端各自从面 1 的点到面 2 的点，以原来的点为控制点
        rows_a = [f1_a]
        rows_b = [f1_b]
        for k in range(1, segments):
            t = k / segments
            rows_a.append(add_point(_bezier(pos_all(f1_a), verts[a], pos_all(f2_a), t)))
            rows_b.append(add_point(_bezier(pos_all(f1_b), verts[b], pos_all(f2_b), t)))
        rows_a.append(f2_a)
        rows_b.append(f2_b)
        for k in range(segments):
            t0, t1 = k / segments, (k + 1) / segments
            new.add([rows_b[k], rows_a[k], rows_a[k + 1], rows_b[k + 1]],
                    [uv1b * (1 - t0) + uv2b * t0, uv1a * (1 - t0) + uv2a * t0, uv1a * (1 - t1) + uv2a * t1,
                     uv1b * (1 - t1) + uv2b * t1], mat, smooth, True)
            bevel_faces.append(len(new) - 1)
    remap = rebuild(mesh, np.zeros(mesh.face_count, bool), new, np.asarray(new_pos) if new_pos else None)
    fresh = remap[base:base + len(new_pos)]
    _fill_corner_holes(mesh, set(int(v) for v in fresh[fresh >= 0]))
    mesh.flush_from_faces()
    return {"faces": np.nonzero(mesh.face_sel)[0], "width": width}


def bevel_vertices(mesh: EditMesh, width: float, clamp: bool = True) -> int:
    """倒角点（Ctrl+Shift+B）：选中的点切掉，连着的每条边上退进 width 处各出一个新点，
    原来的面在这里换成两个新点，点周围补一个面把角封上。返回倒了几个点。"""
    table = mesh.edges()
    chosen = np.zeros(mesh.vert_count, bool)
    chosen[np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]] = True
    edges = table.edges
    touching = chosen[edges[:, 0]] | chosen[edges[:, 1]]
    if not chosen.any() or not touching.any() or width <= 0:
        return 0
    lengths = np.linalg.norm(mesh.verts[edges[:, 0]] - mesh.verts[edges[:, 1]], axis=1)
    if clamp:
        both = chosen[edges[:, 0]] & chosen[edges[:, 1]]
        limit = np.where(both, lengths * 0.49, lengths * 0.98)
        width = min(width, float(limit[touching].min()))
    base = mesh.vert_count
    new_pos: list[np.ndarray] = []
    slid: dict[tuple[int, int], int] = {}

    def slide(v: int, e: int) -> int:
        key = (v, e)
        if key not in slid:
            a, b = (int(x) for x in edges[e])
            w = b if a == v else a
            d = mesh.verts[w] - mesh.verts[v]
            t = width / max(float(lengths[e]), 1e-30)
            slid[key] = base + len(new_pos)
            new_pos.append(mesh.verts[v] + d * t)
        return slid[key]

    nxt = mesh.loop_next()
    prv = mesh.loop_prev()
    lf = mesh.loop_face()
    loop_edge = table.loop_edge
    faces = np.unique(lf[chosen[mesh.loop_vert]])
    new = FaceList()
    remove = np.zeros(mesh.face_count, bool)
    cap_edges: dict[int, dict[int, int]] = {}         # 点 → {新点: 下一个新点}（封口面的走向）
    point_uv: dict[int, np.ndarray] = {}
    for f in faces:
        s, e = mesh.face_start[f], mesh.face_start[f + 1]
        out_v, out_uv = [], []
        for loop in range(s, e):
            v = int(mesh.loop_vert[loop])
            uv = mesh.loop_uv[loop].astype(np.float64)
            if not chosen[v]:
                out_v.append(v)
                out_uv.append(uv)
                continue
            e_in, e_out = int(loop_edge[prv[loop]]), int(loop_edge[loop])
            a_id, b_id = slide(v, e_in), slide(v, e_out)
            t_in = width / max(float(lengths[e_in]), 1e-30)
            t_out = width / max(float(lengths[e_out]), 1e-30)
            uv_prev = mesh.loop_uv[prv[loop]].astype(np.float64)
            uv_next = mesh.loop_uv[nxt[loop]].astype(np.float64)
            uv_a, uv_b = uv * (1 - t_in) + uv_prev * t_in, uv * (1 - t_out) + uv_next * t_out
            out_v += [a_id, b_id]
            out_uv += [uv_a, uv_b]
            point_uv.setdefault(a_id, uv_a)
            point_uv.setdefault(b_id, uv_b)
            cap_edges.setdefault(v, {})[b_id] = a_id
        new.add(out_v, out_uv, mesh.face_mat[f], mesh.face_smooth[f])
        remove[f] = True
    caps = []
    for v, links in cap_edges.items():
        start = next(iter(links))
        cycle = [start]
        while True:
            following = links.get(cycle[-1])
            if following is None or following == start or len(cycle) > len(links):
                break
            cycle.append(following)
        if links.get(cycle[-1]) == start and len(cycle) >= 3:
            caps.append(cycle)
    face_count = mesh.face_count - int(remove.sum()) + len(new)
    for cycle in caps:
        new.add(cycle, [point_uv.get(x, np.zeros(2)) for x in cycle], 0, False, True)
    loose = mesh.loose_edges.copy()
    for k, (a, b) in enumerate(loose):
        e = int(np.nonzero((edges[:, 0] == min(a, b)) & (edges[:, 1] == max(a, b)))[0][0])
        if chosen[a]:
            loose[k, 0] = slide(int(a), e)
        if chosen[b]:
            loose[k, 1] = slide(int(b), e)
    mesh.vert_sel[:] = False
    rebuild(mesh, ~remove, new, np.asarray(new_pos).reshape(-1, 3), loose_edges=loose,
            keep_verts=None, compact=True)
    mesh.face_sel[:] = False
    if caps:
        mesh.face_sel[face_count:face_count + len(caps)] = True
        mesh.flush_from_faces()
    return int(chosen.sum())


def _side(points: list[int], index: int) -> int:
    return points[index]


def _bezier(p0, p1, p2, t):
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


def _line_intersection(p1, d1, p2, d2):
    """两条直线最近点的中点（共面时就是交点）。"""
    w = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        return (p1 + p2) * 0.5
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    return ((p1 + d1 * s) + (p2 + d2 * t)) * 0.5


def _fill_corner_holes(mesh: EditMesh, fresh: set) -> None:
    """倒角后点上留下的洞（边界边围成一圈、全是新点）补成一个面，走向和周围相反。"""
    table = mesh.edges()
    open_edges = np.nonzero(table.face_count == 1)[0]
    if not len(open_edges):
        return
    loop_of_edge = {}
    nxt = mesh.loop_next()
    for loop, e in enumerate(table.loop_edge):
        if table.face_count[e] == 1:
            loop_of_edge[int(e)] = loop
    # 洞的走向：沿已有面的反方向（b → a）
    succ: dict[int, int] = {}
    for e in open_edges:
        loop = loop_of_edge.get(int(e))
        if loop is None:
            continue
        a, b = int(mesh.loop_vert[loop]), int(mesh.loop_vert[nxt[loop]])
        if a not in fresh or b not in fresh:
            continue
        succ[b] = a
    new = FaceList()
    used = set()
    lf = mesh.loop_face()
    for start in list(succ):
        if start in used:
            continue
        cycle = [start]
        v = succ.get(start)
        while v is not None and v != start and v not in used and len(cycle) <= len(succ):
            cycle.append(v)
            v = succ.get(v)
        if v != start or len(cycle) < 3:
            continue
        used.update(cycle)
        uv = []
        for p in cycle:
            loops = np.nonzero(mesh.loop_vert == p)[0]
            uv.append(mesh.loop_uv[loops[0]] if len(loops) else np.zeros(2, np.float32))
        loops_any = np.nonzero(np.isin(mesh.loop_vert, cycle))[0]
        mat = int(mesh.face_mat[lf[loops_any[0]]]) if len(loops_any) else 0
        new.add(cycle, uv, mat, False, True)
    if len(new):
        rebuild(mesh, np.ones(mesh.face_count, bool), new, compact=False)
