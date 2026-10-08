"""环切（Ctrl+R）和边的选择工具：循环边、并排边、最短路径。照 Blender：

- 循环边（Alt+点击）：从一条边出发，在四边形里「穿过对面的点」一直走下去（点有 4 条边时才继续）；
- 并排边（Ctrl+Alt+点击）：从一条边出发，在四边形里跳到对面那条边；
- 环切：沿一圈并排边，每条边上加 cuts 个点，把经过的四边形切开。
"""
from __future__ import annotations

import numpy as np

from .editmesh import EditMesh
from .topology import FaceList, rebuild


def _adjacency(mesh: EditMesh):
    """边 → 用到它的面角列表；点 → 连着的边列表。"""
    table = mesh.edges()
    by_edge: dict[int, list[int]] = {}
    for loop, e in enumerate(table.loop_edge):
        by_edge.setdefault(int(e), []).append(loop)
    by_vert: dict[int, list[int]] = {}
    for e, (a, b) in enumerate(table.edges):
        by_vert.setdefault(int(a), []).append(e)
        by_vert.setdefault(int(b), []).append(e)
    return table, by_edge, by_vert


def edge_ring(mesh: EditMesh, edge: int) -> tuple[list[int], bool]:
    """从 edge 出发的一圈并排边（经过的面都得是四边形）。返回 (边列表, 是否闭合)。"""
    table, by_edge, _by_vert = _adjacency(mesh)
    lf = mesh.loop_face()
    nxt = mesh.loop_next()
    sizes = mesh.face_sizes()

    def opposite(loop):
        # 四边形里面角 loop（边 loop→next）对面的那条边
        return int(table.loop_edge[nxt[nxt[loop]]])

    ring = [int(edge)]
    closed = False
    for direction in (0, 1):
        loops = by_edge.get(int(edge), [])
        if direction >= len(loops):
            continue
        current_loop = loops[direction]
        visited_faces = set()
        while True:
            f = int(lf[current_loop])
            if sizes[f] != 4 or f in visited_faces or mesh.face_hide[f]:
                break
            visited_faces.add(f)
            e = opposite(current_loop)
            if e == ring[0] and direction == 0:
                closed = True
                break
            if direction == 0:
                ring.append(e)
            else:
                ring.insert(0, e)
            others = [l for l in by_edge.get(e, []) if int(lf[l]) != f]
            if not others:
                break
            current_loop = others[0]
        if closed:
            break
    return ring, closed


def edge_loop(mesh: EditMesh, edge: int) -> list[int]:
    """从 edge 出发的循环边：在有 4 条边的点上走向「对面」的边；碰到边界时沿边界走。"""
    table, by_edge, by_vert = _adjacency(mesh)
    lf = mesh.loop_face()
    nxt = mesh.loop_next()
    prv = mesh.loop_prev()
    result = [int(edge)]
    boundary = table.face_count[int(edge)] == 1

    def step(e, v):
        """沿边 e 走到点 v 之后的下一条边。"""
        edges_here = by_vert.get(int(v), [])
        if boundary:
            # 边界边：接着走点 v 上另一条边界边
            cands = [x for x in edges_here if x != e and table.face_count[x] == 1]
            return cands[0] if len(cands) == 1 else None
        if len(edges_here) != 4:
            return None
        # 和 e 不共面的那条边就是「对面」的
        faces_e = {int(lf[l]) for l in by_edge.get(int(e), [])}
        for x in edges_here:
            if x == e:
                continue
            faces_x = {int(lf[l]) for l in by_edge.get(int(x), [])}
            if not (faces_e & faces_x):
                return x
        return None

    del nxt, prv
    seen = {int(edge)}
    for start_vert_index in (1, 0):
        e = int(edge)
        v = int(table.edges[e][start_vert_index])
        while True:
            x = step(e, v)
            if x is None:
                break
            if x in seen:
                if x == int(edge):
                    return result            # 绕了一整圈回到起点：闭合的循环边
                break
            seen.add(x)
            if start_vert_index == 1:
                result.append(x)
            else:
                result.insert(0, x)
            a, b = table.edges[x]
            v = int(b) if int(a) == v else int(a)
            e = x
    return result


def shortest_path(mesh: EditMesh, start: int, end: int) -> list[int]:
    """两点之间沿边的最短路径（按边长），返回经过的点。"""
    import heapq

    table, _by_edge, by_vert = _adjacency(mesh)
    dist = {start: 0.0}
    prev: dict[int, int] = {}
    heap = [(0.0, start)]
    while heap:
        d, v = heapq.heappop(heap)
        if v == end:
            break
        if d > dist.get(v, float("inf")):
            continue
        for e in by_vert.get(v, []):
            a, b = table.edges[e]
            w = int(b) if int(a) == v else int(a)
            if mesh.vert_hide[w]:
                continue
            nd = d + float(np.linalg.norm(mesh.verts[w] - mesh.verts[v]))
            if nd < dist.get(w, float("inf")):
                dist[w] = nd
                prev[w] = v
                heapq.heappush(heap, (nd, w))
    if end not in dist:
        return []
    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]


def loop_cut(mesh: EditMesh, edge: int, cuts: int = 1, factor: float = 0.0) -> np.ndarray:
    """沿 edge 所在的一圈并排边切 cuts 刀。factor（-1..1）把只切一刀时的切口往一边挪。返回新点号（已选中）。"""
    ring, closed = edge_ring(mesh, edge)
    if not ring:
        return np.zeros(0, np.int64)
    cuts = max(1, int(cuts))
    table, by_edge, _by_vert = _adjacency(mesh)
    lf = mesh.loop_face()
    nxt = mesh.loop_next()
    ring_set = {int(e): i for i, e in enumerate(ring)}
    # 每条并排边上的新点：按「参考方向」排，相邻两条边的参考方向保持一致（顺着四边形走）
    base = mesh.vert_count
    new_pos = []
    points: dict[int, np.ndarray] = {}
    ref_start: dict[int, int] = {}
    # 第一条边随便定方向，之后每走过一个四边形，对边的方向由面的走向决定
    first = ring[0]
    ref_start[first] = int(table.edges[first][0])
    faces_done = set()
    order = list(ring)
    for e in order:
        for loop in by_edge.get(e, []):
            f = int(lf[loop])
            if f in faces_done or mesh.face_sizes()[f] != 4:
                continue
            opp_loop = nxt[nxt[loop]]
            opp = int(table.loop_edge[opp_loop])
            if opp not in ring_set or opp in ref_start:
                continue
            a = int(mesh.loop_vert[loop])
            # 面里 loop 的起点 a 对应对边的终点（四边形里两条对边走向相反）
            opp_end = int(mesh.loop_vert[nxt[opp_loop]])
            ref_start[opp] = opp_end if ref_start[e] == a else int(mesh.loop_vert[opp_loop])
            faces_done.add(f)
    count = base
    for e in ring:
        a, b = table.edges[e]
        s = ref_start.get(e, int(a))
        t_end = int(b) if s == int(a) else int(a)
        pts = []
        for k in range(1, cuts + 1):
            t = k / (cuts + 1)
            if cuts == 1:
                t = 0.5 + 0.5 * float(np.clip(factor, -0.999, 0.999))
            new_pos.append(mesh.verts[s] * (1 - t) + mesh.verts[t_end] * t)
            pts.append(count)
            count += 1
        points[e] = (s, np.asarray(pts, np.int64))
    # 经过的四边形切成 cuts+1 条
    new = FaceList()
    remove = np.zeros(mesh.face_count, bool)
    for f in range(mesh.face_count):
        if mesh.face_sizes()[f] != 4:
            continue
        s0 = mesh.face_start[f]
        loops = [s0, s0 + 1, s0 + 2, s0 + 3]
        edges = [int(table.loop_edge[l]) for l in loops]
        hit = [k for k in range(4) if edges[k] in ring_set]
        if len(hit) != 2 or (hit[1] - hit[0]) != 2:
            continue
        k0 = hit[0]
        la, lb = loops[k0], loops[(k0 + 2) % 4]
        # 第一条被切的边按面的走向 (la → la.next)，对边 (lb → lb.next) 和它反向
        va, va2 = int(mesh.loop_vert[la]), int(mesh.loop_vert[nxt[la]])
        vb, vb2 = int(mesh.loop_vert[lb]), int(mesh.loop_vert[nxt[lb]])
        ua, ua2 = mesh.loop_uv[la], mesh.loop_uv[nxt[la]]
        ub, ub2 = mesh.loop_uv[lb], mesh.loop_uv[nxt[lb]]
        start_a, pts_a = points[edges[k0]]
        start_b, pts_b = points[edges[(k0 + 2) % 4]]
        run_a = pts_a if start_a == va else pts_a[::-1]           # 从 va 到 va2
        run_b = pts_b if start_b == vb2 else pts_b[::-1]          # 从 vb2 到 vb（和 run_a 平行）
        chain_a = [va] + list(run_a) + [va2]
        chain_b = [vb2] + list(run_b) + [vb]
        n = cuts + 1
        uv_a = [ua * (1 - j / n) + ua2 * (j / n) for j in range(n + 1)]
        uv_b = [ub2 * (1 - j / n) + ub * (j / n) for j in range(n + 1)]
        for j in range(n):
            new.add([chain_a[j], chain_a[j + 1], chain_b[j + 1], chain_b[j]],
                    [uv_a[j], uv_a[j + 1], uv_b[j + 1], uv_b[j]], mesh.face_mat[f], mesh.face_smooth[f])
        remove[f] = True
    # 环的两头碰到的不是四边形的面：边上多了点，插进面里
    touched = np.zeros(mesh.face_count, bool)
    for e in ring:
        for loop in by_edge.get(e, []):
            f = int(lf[loop])
            if not remove[f]:
                touched[f] = True
    for f in np.nonzero(touched)[0]:
        s, e = mesh.face_start[f], mesh.face_start[f + 1]
        verts_out, uvs_out = [], []
        n = e - s
        for k, loop in enumerate(range(s, e)):
            verts_out.append(int(mesh.loop_vert[loop]))
            uvs_out.append(mesh.loop_uv[loop])
            edge_id = int(table.loop_edge[loop])
            if edge_id in points:
                start, pts = points[edge_id]
                run = pts if start == int(mesh.loop_vert[loop]) else pts[::-1]
                for j, p in enumerate(run):
                    verts_out.append(int(p))
                    t = (j + 1) / (len(run) + 1)
                    uvs_out.append(mesh.loop_uv[loop] * (1 - t) + mesh.loop_uv[s + (k + 1) % n] * t)
        new.add(verts_out, uvs_out, mesh.face_mat[f], mesh.face_smooth[f])
        remove[f] = True
    rebuild(mesh, ~remove, new, np.asarray(new_pos), compact=False)
    mesh.select_all("DESELECT")
    new_ids = np.arange(base, count)
    mesh.vert_sel[new_ids] = True
    mesh.flush_from_verts()
    return new_ids
