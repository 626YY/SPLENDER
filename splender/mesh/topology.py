"""编辑模式的拓扑操作（照 Blender）：删除、融并、合并、挤出、内插、填充、复制、分离、拆分、细分、三角化、
三角转四边、法线、平滑、隐藏。都直接改 EditMesh（数组重建后调用 invalidate），返回新加的点或面（接着要移动时用）。

约定：面的点按逆时针排（从面的法线方向看）；新面的 UV 从相邻的老面角里取。
"""
from __future__ import annotations

import math

import numpy as np

from .editmesh import EditMesh


# ====================================================================== 重建
class FaceList:
    """新加的面：点号、每个面角的 UV、材质、平滑、要不要选中。"""

    def __init__(self) -> None:
        self.verts: list[np.ndarray] = []
        self.uvs: list[np.ndarray] = []
        self.mats: list[int] = []
        self.smooth: list[bool] = []
        self.select: list[bool] = []

    def add(self, verts, uvs, mat: int = 0, smooth: bool = False, select: bool = False) -> None:
        verts = np.asarray(verts, np.int64)
        if len(verts) < 3 or len(set(verts.tolist())) < 3:
            return
        self.verts.append(verts)
        self.uvs.append(np.asarray(uvs, np.float32).reshape(-1, 2))
        self.mats.append(int(mat))
        self.smooth.append(bool(smooth))
        self.select.append(bool(select))

    def __len__(self) -> int:
        return len(self.verts)


def rebuild(mesh: EditMesh, keep_faces: np.ndarray, new: FaceList | None = None, new_verts=None,
            keep_verts: np.ndarray | None = None, loose_edges=None, compact: bool = True,
            loop_vert_override: np.ndarray | None = None) -> np.ndarray:
    """留下 keep_faces 这些老面，加上 new 里的新面；new_verts 先接在点数组后面（新面的点号可以指向它们）。
    compact 时去掉没人用的点（keep_verts 里的点保留，散边用到的也保留）。返回老点号 → 新点号（删掉的是 -1）。"""
    if new_verts is not None and len(new_verts):
        add = np.asarray(new_verts, np.float64).reshape(-1, 3)
        mesh.verts = np.concatenate([mesh.verts, add])
        mesh.vert_sel = np.concatenate([mesh.vert_sel, np.zeros(len(add), bool)])
        mesh.vert_hide = np.concatenate([mesh.vert_hide, np.zeros(len(add), bool)])
    loop_vert = mesh.loop_vert if loop_vert_override is None else loop_vert_override
    sizes = mesh.face_sizes()
    keep_faces = np.asarray(keep_faces, bool)
    keep_loops = np.repeat(keep_faces, sizes)
    kept_sizes = sizes[keep_faces]
    parts_v = [loop_vert[keep_loops]]
    parts_uv = [mesh.loop_uv[keep_loops]]
    sizes_all = [kept_sizes]
    mats = [mesh.face_mat[keep_faces]]
    smooth = [mesh.face_smooth[keep_faces]]
    fsel = [mesh.face_sel[keep_faces]]
    fhide = [mesh.face_hide[keep_faces]]
    if new is not None and len(new):
        parts_v += new.verts
        parts_uv += new.uvs
        sizes_all.append(np.asarray([len(v) for v in new.verts], np.int64))
        mats.append(np.asarray(new.mats, np.int32))
        smooth.append(np.asarray(new.smooth, bool))
        fsel.append(np.asarray(new.select, bool))
        fhide.append(np.zeros(len(new), bool))
    face_sizes = np.concatenate(sizes_all) if sizes_all else np.zeros(0, np.int64)
    mesh.loop_vert = np.concatenate(parts_v).astype(np.int64) if parts_v else np.zeros(0, np.int64)
    mesh.loop_uv = np.concatenate(parts_uv).astype(np.float32) if parts_uv else np.zeros((0, 2), np.float32)
    mesh.face_start = np.concatenate([[0], np.cumsum(face_sizes)]).astype(np.int64)
    mesh.face_mat = np.concatenate(mats).astype(np.int32)
    mesh.face_smooth = np.concatenate(smooth).astype(bool)
    mesh.face_sel = np.concatenate(fsel).astype(bool)
    mesh.face_hide = np.concatenate(fhide).astype(bool)
    if loose_edges is not None:
        mesh.loose_edges = np.asarray(loose_edges, np.int64).reshape(-1, 2)
    remap = np.arange(mesh.vert_count, dtype=np.int64)
    if compact:
        used = np.zeros(mesh.vert_count, bool)
        used[mesh.loop_vert] = True
        if len(mesh.loose_edges):
            used[mesh.loose_edges.reshape(-1)] = True
        if keep_verts is not None:
            keep = np.asarray(keep_verts, bool)
            used[:len(keep)] |= keep
        remap = np.full(mesh.vert_count, -1, np.int64)
        remap[used] = np.arange(int(used.sum()))
        mesh.verts = mesh.verts[used]
        mesh.vert_sel = mesh.vert_sel[used]
        mesh.vert_hide = mesh.vert_hide[used]
        mesh.loop_vert = remap[mesh.loop_vert]
        if len(mesh.loose_edges):
            mesh.loose_edges = remap[mesh.loose_edges]
    _drop_bad_loose(mesh)
    if mesh.active is not None:
        kind, index = mesh.active
        if kind == "VERT":
            mesh.active = ("VERT", int(remap[index])) if index < len(remap) and remap[index] >= 0 else None
        else:
            mesh.active = None
    mesh.invalidate()
    return remap


def _drop_bad_loose(mesh: EditMesh) -> None:
    """散边里去掉两端相同的、和面的边重复的。"""
    if not len(mesh.loose_edges):
        return
    e = mesh.loose_edges
    e = e[e[:, 0] != e[:, 1]]
    if len(e) and mesh.loop_count:
        v = max(1, mesh.vert_count)
        a = mesh.loop_vert
        b = mesh.loop_vert[mesh.loop_next()]
        face_keys = np.unique(np.minimum(a, b) * v + np.maximum(a, b))
        keys = np.minimum(e[:, 0], e[:, 1]) * v + np.maximum(e[:, 0], e[:, 1])
        e = e[~np.isin(keys, face_keys)]
        keys = np.minimum(e[:, 0], e[:, 1]) * v + np.maximum(e[:, 0], e[:, 1])
        _u, first = np.unique(keys, return_index=True)
        e = e[np.sort(first)]
    mesh.loose_edges = e


def select_new(mesh: EditMesh, verts=None, faces=None) -> None:
    """只选中这些新点、新面（按选择模式推到边）。"""
    mesh.vert_sel[:] = False
    mesh.face_sel[:] = False
    if verts is not None:
        mesh.vert_sel[np.asarray(verts, np.int64)] = True
    if faces is not None:
        mesh.face_sel[np.asarray(faces, np.int64)] = True
        mesh.flush_from_faces()
    else:
        mesh.flush_from_verts()


# ====================================================================== 删除
def delete(mesh: EditMesh, kind: str = "VERT") -> None:
    """X 菜单：VERT 删点（连着的边、面一起没）、EDGE 删边（连着的面一起没）、FACE 删面（留下的散点散边去掉）、
    ONLY_FACE 只删面（边和点留着）、EDGE_FACE 删边和面（点留着）。"""
    table = mesh.edges()
    lf = mesh.loop_face()
    if kind == "VERT":
        gone_v = mesh.vert_sel.copy()
        bad_loop = gone_v[mesh.loop_vert]
        keep_faces = ~np.logical_or.reduceat(bad_loop, mesh.face_start[:-1]) if mesh.face_count else np.zeros(0, bool)
        loose = mesh.loose_edges[~(gone_v[mesh.loose_edges[:, 0]] | gone_v[mesh.loose_edges[:, 1]])] \
            if len(mesh.loose_edges) else mesh.loose_edges
        # 原来就是散点（不属于面和边）、又没被删的留着
        rebuild(mesh, keep_faces, loose_edges=loose, keep_verts=~gone_v & _isolated(mesh))
    elif kind in ("EDGE", "EDGE_FACE"):
        gone_e = mesh.edge_sel.copy()
        bad_loop = gone_e[table.loop_edge] if mesh.loop_count else np.zeros(0, bool)
        keep_faces = ~np.logical_or.reduceat(bad_loop, mesh.face_start[:-1]) if mesh.face_count else np.zeros(0, bool)
        # 被删的面上没被删的边变成散边（和 Blender 一样留着）
        faces_gone = ~keep_faces
        orphan = np.zeros(len(table.edges), bool)
        if mesh.loop_count:
            orphan[table.loop_edge[faces_gone[lf]]] = True
        still_used = np.zeros(len(table.edges), bool)
        if mesh.loop_count:
            still_used[table.loop_edge[keep_faces[lf]]] = True
        new_loose = table.edges[(orphan & ~still_used & ~gone_e) | (table.loose & ~gone_e)]
        keep_v = None
        if kind == "EDGE_FACE":
            keep_v = np.zeros(mesh.vert_count, bool)
            keep_v[table.edges[gone_e].reshape(-1)] = True
        rebuild(mesh, keep_faces, loose_edges=new_loose, keep_verts=keep_v)
    elif kind in ("FACE", "ONLY_FACE"):
        gone_f = mesh.face_sel.copy()
        keep_faces = ~gone_f
        loose = mesh.loose_edges
        if kind == "ONLY_FACE":
            still = np.zeros(len(table.edges), bool)
            if mesh.loop_count:
                still[table.loop_edge[keep_faces[lf]]] = True
            orphan = np.zeros(len(table.edges), bool)
            if mesh.loop_count:
                orphan[table.loop_edge[gone_f[lf]]] = True
            loose = table.edges[(orphan & ~still) | table.loose]
            rebuild(mesh, keep_faces, loose_edges=loose)
        else:
            rebuild(mesh, keep_faces, loose_edges=loose)
    mesh.select_all("DESELECT")


def _isolated(mesh: EditMesh) -> np.ndarray:
    """不属于任何面、任何边的散点。"""
    used = np.zeros(mesh.vert_count, bool)
    used[mesh.loop_vert] = True
    if len(mesh.loose_edges):
        used[mesh.loose_edges.reshape(-1)] = True
    return ~used


# ====================================================================== 面区域、边界
def region_boundary(mesh: EditMesh, faces: np.ndarray):
    """一组面的边界：每条边界边按区域里那个面的走向给出 (面角号, 起点, 终点)。"""
    table = mesh.edges()
    lf = mesh.loop_face()
    in_region = np.zeros(mesh.face_count, bool)
    in_region[faces] = True
    loops = np.nonzero(in_region[lf])[0]
    edge_count = np.bincount(table.loop_edge[loops], minlength=len(table.edges))
    boundary_loops = loops[edge_count[table.loop_edge[loops]] == 1]
    nxt = mesh.loop_next()
    return boundary_loops, mesh.loop_vert[boundary_loops], mesh.loop_vert[nxt[boundary_loops]]


def _face_uv_at(mesh: EditMesh, loop: int) -> np.ndarray:
    return mesh.loop_uv[loop]


# ====================================================================== 挤出
def extrude_region(mesh: EditMesh, individual: bool = False) -> np.ndarray:
    """挤出选中的面（整块一起，或者每个面各自挤）；没有选中的面时挤出选中的边或点。
    返回新点的编号（已选中），接着沿法线移动。"""
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
    if len(faces):
        if individual:
            return _extrude_faces_individual(mesh, faces)
        return _extrude_faces(mesh, faces)
    edges = np.nonzero(mesh.edge_sel)[0]
    if len(edges):
        return _extrude_edges(mesh, edges)
    verts = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
    if len(verts):
        return _extrude_verts(mesh, verts)
    return np.zeros(0, np.int64)


def _extrude_faces(mesh: EditMesh, faces: np.ndarray) -> np.ndarray:
    lf = mesh.loop_face()
    in_region = np.zeros(mesh.face_count, bool)
    in_region[faces] = True
    region_loops = np.nonzero(in_region[lf])[0]
    region_verts = np.unique(mesh.loop_vert[region_loops])
    base = mesh.vert_count
    dup = np.full(base, -1, np.int64)
    dup[region_verts] = base + np.arange(len(region_verts))
    new_verts = mesh.verts[region_verts].copy()
    b_loops, a_ids, b_ids = region_boundary(mesh, faces)
    new = FaceList()
    for loop, a, b in zip(b_loops, a_ids, b_ids):
        f = lf[loop]
        uv_a = mesh.loop_uv[loop]
        uv_b = mesh.loop_uv[mesh.loop_next()[loop]]
        new.add([a, b, dup[b], dup[a]], [uv_a, uv_b, uv_b, uv_a], mesh.face_mat[f], mesh.face_smooth[f], False)
    # 区域里的面换成新点（这就是挤出去的「顶盖」）
    loop_vert = mesh.loop_vert.copy()
    loop_vert[region_loops] = dup[loop_vert[region_loops]]
    keep = np.ones(mesh.face_count, bool)
    old_face_count = mesh.face_count
    mesh.face_sel[:] = False
    mesh.face_sel[faces] = True
    rebuild(mesh, keep, new, new_verts, loop_vert_override=loop_vert, compact=True)
    cap_faces = faces[faces < old_face_count]
    select_new(mesh, faces=cap_faces)
    return np.nonzero(mesh.vert_sel)[0]


def _extrude_faces_individual(mesh: EditMesh, faces: np.ndarray) -> np.ndarray:
    """每个面各自挤出（沿各自法线），面和面之间也长出侧面。"""
    lf = mesh.loop_face()
    base = mesh.vert_count
    new_verts = []
    loop_vert = mesh.loop_vert.copy()
    new = FaceList()
    nxt = mesh.loop_next()
    count = base
    for f in faces:
        s, e = mesh.face_start[f], mesh.face_start[f + 1]
        loops = np.arange(s, e)
        verts = mesh.loop_vert[loops]
        ids = count + np.arange(len(verts))
        count += len(verts)
        new_verts.append(mesh.verts[verts])
        for k, loop in enumerate(loops):
            a, b = verts[k], verts[(k + 1) % len(verts)]
            uv_a, uv_b = mesh.loop_uv[loop], mesh.loop_uv[nxt[loop]]
            new.add([a, b, ids[(k + 1) % len(verts)], ids[k]], [uv_a, uv_b, uv_b, uv_a], mesh.face_mat[f],
                    mesh.face_smooth[f])
        loop_vert[loops] = ids
    del lf
    rebuild(mesh, np.ones(mesh.face_count, bool), new, np.concatenate(new_verts), loop_vert_override=loop_vert)
    select_new(mesh, faces=faces)
    return np.nonzero(mesh.vert_sel)[0]


def _extrude_edges(mesh: EditMesh, edges: np.ndarray) -> np.ndarray:
    table = mesh.edges()
    pairs = table.edges[edges]
    verts = np.unique(pairs.reshape(-1))
    base = mesh.vert_count
    dup = np.full(base, -1, np.int64)
    dup[verts] = base + np.arange(len(verts))
    new = FaceList()
    # 新面的走向和这条边已有的面相反（法线朝同一侧）；UV 用相邻面角的
    edge_loop = _edge_first_loop(mesh)
    nxt = mesh.loop_next()
    for e, (a, b) in zip(edges, pairs):
        loop = edge_loop[e]
        if loop >= 0:
            la, lb = mesh.loop_vert[loop], mesh.loop_vert[nxt[loop]]
            uv_a, uv_b = mesh.loop_uv[loop], mesh.loop_uv[nxt[loop]]
            f = mesh.loop_face()[loop]
            new.add([lb, la, dup[la], dup[lb]], [uv_b, uv_a, uv_a, uv_b], mesh.face_mat[f], mesh.face_smooth[f])
        else:
            new.add([a, b, dup[b], dup[a]], np.zeros((4, 2)))
    rebuild(mesh, np.ones(mesh.face_count, bool), new, mesh.verts[verts].copy(), compact=False)
    select_new(mesh, verts=base + np.arange(len(verts)))
    return base + np.arange(len(verts))


def _extrude_verts(mesh: EditMesh, verts: np.ndarray) -> np.ndarray:
    base = mesh.vert_count
    new_ids = base + np.arange(len(verts))
    loose = np.concatenate([mesh.loose_edges, np.stack([verts, new_ids], axis=1)])
    rebuild(mesh, np.ones(mesh.face_count, bool), None, mesh.verts[verts].copy(), loose_edges=loose, compact=False)
    select_new(mesh, verts=new_ids)
    return new_ids


def _edge_first_loop(mesh: EditMesh) -> np.ndarray:
    """每条边对应的一个面角（没有面的边是 -1）。"""
    table = mesh.edges()
    out = np.full(len(table.edges), -1, np.int64)
    if mesh.loop_count:
        out[table.loop_edge[::-1]] = np.arange(mesh.loop_count)[::-1]
    return out


# ====================================================================== 内插
def inset_region(mesh: EditMesh, thickness: float, depth: float = 0.0, individual: bool = False,
                 base=None) -> dict:
    """内插选中的面：边界往里缩 thickness，中间那块往法线方向挪 depth，原来的边界和缩进去的边界之间长出一圈面。
    base 是上一次内插前的快照（拖动调厚度时每次从同一个起点算）。"""
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
    if not len(faces):
        return {"faces": np.zeros(0, np.int64)}
    if individual:
        # 每个面各自内插：一个面一个面地做（每次只选中一个）
        all_faces = faces.copy()
        for f in all_faces[::-1]:
            mesh.face_sel[:] = False
            mesh.face_sel[f] = True
            inset_region(mesh, thickness, depth, False)
        mesh.face_sel[:] = False
        mesh.face_sel[all_faces] = True
        mesh.flush_from_faces()
        return {"faces": all_faces}
    fn = mesh.face_normals()
    lf = mesh.loop_face()
    nxt = mesh.loop_next()
    b_loops, a_ids, b_ids = region_boundary(mesh, faces)
    if not len(b_loops):
        return {"faces": faces}
    # 每个边界点：进来的边和出去的边，往里的方向取两边往里方向的角平分线
    out_edge = {}
    in_edge = {}
    for loop, a, b in zip(b_loops, a_ids, b_ids):
        out_edge[int(a)] = loop
        in_edge[int(b)] = loop
    region_normal_at = {}
    move = {}
    for v, loop_out in out_edge.items():
        loop_in = in_edge.get(v)
        f_out = lf[loop_out]
        n_out = fn[f_out]
        d_out = mesh.verts[mesh.loop_vert[nxt[loop_out]]] - mesh.verts[v]
        inward_out = np.cross(n_out, d_out)
        inward_out /= max(np.linalg.norm(inward_out), 1e-30)
        if loop_in is None:
            direction = inward_out
            scale = 1.0
            normal = n_out
        else:
            f_in = lf[loop_in]
            n_in = fn[f_in]
            d_in = mesh.verts[v] - mesh.verts[mesh.loop_vert[loop_in]]
            inward_in = np.cross(n_in, d_in)
            inward_in /= max(np.linalg.norm(inward_in), 1e-30)
            direction = inward_out + inward_in
            length = np.linalg.norm(direction)
            if length < 1e-9:
                direction = inward_out
                scale = 1.0
            else:
                direction /= length
                scale = 1.0 / max(float(direction @ inward_out), 0.2)
            normal = n_out + n_in
            normal /= max(np.linalg.norm(normal), 1e-30)
        move[v] = direction * scale
        region_normal_at[v] = normal
    base_count = mesh.vert_count
    boundary_verts = np.asarray(sorted(move), np.int64)
    dup = np.full(base_count, -1, np.int64)
    dup[boundary_verts] = base_count + np.arange(len(boundary_verts))
    new_pos = np.stack([mesh.verts[v] + move[int(v)] * thickness for v in boundary_verts])
    new = FaceList()
    for loop, a, b in zip(b_loops, a_ids, b_ids):
        f = lf[loop]
        uv_a = mesh.loop_uv[loop]
        uv_b = mesh.loop_uv[nxt[loop]]
        new.add([a, b, dup[b], dup[a]], [uv_a, uv_b, uv_b, uv_a], mesh.face_mat[f], mesh.face_smooth[f])
    in_region = np.zeros(mesh.face_count, bool)
    in_region[faces] = True
    region_loops = np.nonzero(in_region[lf])[0]
    loop_vert = mesh.loop_vert.copy()
    on_boundary = dup[loop_vert[region_loops]] >= 0
    loop_vert[region_loops[on_boundary]] = dup[loop_vert[region_loops[on_boundary]]]
    rebuild(mesh, np.ones(mesh.face_count, bool), new, new_pos, loop_vert_override=loop_vert, compact=False)
    if depth:
        inner = np.unique(mesh.loop_vert[np.nonzero(np.repeat(np.isin(np.arange(mesh.face_count), faces),
                                                                mesh.face_sizes()))[0]])
        avg = fn[faces].mean(axis=0)
        avg /= max(np.linalg.norm(avg), 1e-30)
        mesh.verts[inner] += avg * depth
    select_new(mesh, faces=faces)
    return {"faces": faces}


# ====================================================================== 合并、融并、填充
def merge_verts(mesh: EditMesh, groups: list[np.ndarray], targets: np.ndarray) -> None:
    """每组点合成一个点，放在 targets 对应的位置。面里连续重复的点去掉，少于三个点的面删掉。"""
    remap = np.arange(mesh.vert_count, dtype=np.int64)
    for group, target in zip(groups, targets):
        group = np.asarray(group, np.int64)
        if len(group) == 0:
            continue
        keep = int(group[0])
        remap[group] = keep
        mesh.verts[keep] = target
    _apply_vert_remap(mesh, remap)


def _apply_vert_remap(mesh: EditMesh, remap: np.ndarray) -> None:
    loop_vert = remap[mesh.loop_vert]
    # 去掉面里连续重复的点
    keep_loop = np.ones(mesh.loop_count, bool)
    if mesh.loop_count:
        nxt = mesh.loop_next()
        keep_loop = loop_vert != loop_vert[nxt]
    sizes = np.add.reduceat(keep_loop.astype(np.int64), mesh.face_start[:-1]) if mesh.face_count else \
        np.zeros(0, np.int64)
    good_face = sizes >= 3
    keep_loop &= np.repeat(good_face, mesh.face_sizes())
    lf = mesh.loop_face()
    mesh.loop_vert = loop_vert[keep_loop]
    mesh.loop_uv = mesh.loop_uv[keep_loop]
    new_sizes = sizes[good_face]
    mesh.face_start = np.concatenate([[0], np.cumsum(new_sizes)]).astype(np.int64)
    mesh.face_mat = mesh.face_mat[good_face]
    mesh.face_smooth = mesh.face_smooth[good_face]
    mesh.face_sel = mesh.face_sel[good_face]
    mesh.face_hide = mesh.face_hide[good_face]
    del lf
    if len(mesh.loose_edges):
        mesh.loose_edges = remap[mesh.loose_edges]
    rebuild(mesh, _unique_faces(mesh), keep_verts=_isolated_before(mesh, remap))


def _unique_faces(mesh: EditMesh) -> np.ndarray:
    """点集合完全相同的面只留第一个（合并点以后重叠的面）。"""
    keep = np.ones(mesh.face_count, bool)
    seen = set()
    for f in range(mesh.face_count):
        key = tuple(sorted(mesh.loop_vert[mesh.face_start[f]:mesh.face_start[f + 1]].tolist()))
        if key in seen:
            keep[f] = False
        else:
            seen.add(key)
    return keep


def _isolated_before(mesh: EditMesh, remap: np.ndarray) -> np.ndarray:
    keep = np.zeros(mesh.vert_count, bool)
    keep[np.unique(remap)] = True
    return keep & mesh.vert_sel


def merge(mesh: EditMesh, how: str = "CENTER", cursor=None) -> int:
    """M 菜单：CENTER 到中心、CURSOR 到游标、FIRST/LAST 到第一个/最后一个选中的点、COLLAPSE 每块各自合成一点。
    返回合并掉的点数。"""
    verts = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
    if len(verts) < 2 and how != "COLLAPSE":
        return 0
    before = mesh.vert_count
    if how == "COLLAPSE":
        groups = _selected_islands(mesh, verts)
        targets = np.stack([mesh.verts[g].mean(axis=0) for g in groups]) if groups else np.zeros((0, 3))
    else:
        if how == "CURSOR" and cursor is not None:
            target = np.asarray(cursor, np.float64)
        elif how in ("FIRST", "LAST") and mesh.active is not None and mesh.active[0] == "VERT":
            target = mesh.verts[mesh.active[1]].copy()
        else:
            target = mesh.verts[verts].mean(axis=0)
        groups = [verts]
        targets = np.asarray([target])
    merge_verts(mesh, groups, targets)
    mesh.select_all("DESELECT")
    return before - mesh.vert_count


def merge_by_distance(mesh: EditMesh, distance: float, only_selected: bool = True) -> int:
    """按距离合并（去重点）：距离小于 distance 的点合成一个。返回合并掉的点数。"""
    candidates = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0] if only_selected else np.arange(mesh.vert_count)
    if len(candidates) < 2:
        return 0
    p = mesh.verts[candidates]
    step = max(float(distance), 1e-12)
    cell = np.floor(p / step).astype(np.int64)
    order = np.lexsort((cell[:, 2], cell[:, 1], cell[:, 0]))
    parent = np.arange(len(candidates))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # 同格和相邻格里的点两两比较（体量通常不大）
    from collections import defaultdict

    buckets = defaultdict(list)
    for i in order:
        buckets[tuple(cell[i])].append(i)
    for key, members in buckets.items():
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbors.extend(buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ()))
        for i in members:
            for j in neighbors:
                if j <= i:
                    continue
                if np.linalg.norm(p[i] - p[j]) <= distance:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(candidates))])
    groups = []
    targets = []
    for root in np.unique(roots):
        members = candidates[roots == root]
        if len(members) > 1:
            groups.append(members)
            targets.append(mesh.verts[members].mean(axis=0))
    if not groups:
        return 0
    before = mesh.vert_count
    merge_verts(mesh, groups, np.asarray(targets))
    return before - mesh.vert_count


def _selected_islands(mesh: EditMesh, verts: np.ndarray) -> list[np.ndarray]:
    """选中的点按选中的边连成几块。"""
    table = mesh.edges()
    chosen = set(int(v) for v in verts)
    parent = {v: v for v in chosen}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in table.edges:
        a, b = int(a), int(b)
        if a in chosen and b in chosen:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for v in chosen:
        groups.setdefault(find(v), []).append(v)
    return [np.asarray(sorted(g), np.int64) for g in groups.values()]


def dissolve_faces(mesh: EditMesh, faces: np.ndarray | None = None) -> int:
    """融并面：连在一起的选中面合成一个多边形（只处理边界是一圈的块）。返回生成的面数。"""
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0] if faces is None else np.asarray(faces, np.int64)
    if len(faces) < 2:
        return 0
    islands = _face_islands(mesh, faces)
    new = FaceList()
    remove = np.zeros(mesh.face_count, bool)
    lf = mesh.loop_face()
    for island in islands:
        if len(island) < 2:
            continue
        loops = _boundary_cycle(mesh, island)
        if loops is None:
            continue
        verts = mesh.loop_vert[loops]
        new.add(verts, mesh.loop_uv[loops], mesh.face_mat[lf[loops[0]]], bool(mesh.face_smooth[island].any()), True)
        remove[island] = True
    if not len(new):
        return 0
    remap = rebuild(mesh, ~remove, new)
    mesh.flush_from_faces()
    mesh._last_remap = remap
    return len(new)


def _face_islands(mesh: EditMesh, faces: np.ndarray) -> list[np.ndarray]:
    """面按共用的边连成几块。"""
    table = mesh.edges()
    lf = mesh.loop_face()
    chosen = np.zeros(mesh.face_count, bool)
    chosen[faces] = True
    parent = np.arange(mesh.face_count)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    loops = np.nonzero(chosen[lf])[0]
    by_edge: dict[int, int] = {}
    for loop in loops:
        e = int(table.loop_edge[loop])
        f = int(lf[loop])
        if e in by_edge:
            ra, rb = find(by_edge[e]), find(f)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        else:
            by_edge[e] = f
    groups: dict[int, list[int]] = {}
    for f in faces:
        groups.setdefault(find(int(f)), []).append(int(f))
    return [np.asarray(sorted(g), np.int64) for g in groups.values()]


def _boundary_cycle(mesh: EditMesh, island: np.ndarray):
    """一块面的外边界按顺序排成一圈（面角号，走向和块里的面一致）。边界不是一整圈时返回 None。"""
    b_loops, a_ids, b_ids = region_boundary(mesh, island)
    if not len(b_loops):
        return None
    start_of = {}
    for loop, a in zip(b_loops, a_ids):
        if int(a) in start_of:
            return None                 # 点上有两条边界边出去：不是简单的一圈
        start_of[int(a)] = (int(loop), None)
    nxt_vert = dict(zip((int(a) for a in a_ids), (int(b) for b in b_ids)))
    first = int(a_ids[0])
    cycle = []
    v = first
    for _ in range(len(b_loops) + 1):
        cycle.append(start_of[v][0])
        v = nxt_vert[v]
        if v == first:
            break
        if v not in start_of:
            return None
    if len(cycle) != len(b_loops):
        return None
    return np.asarray(cycle, np.int64)


def dissolve_edges(mesh: EditMesh) -> int:
    """融并边：选中边两边的面合成一个面。"""
    table = mesh.edges()
    edges = np.nonzero(mesh.edge_sel)[0]
    if not len(edges):
        return 0
    lf = mesh.loop_face()
    loops = np.nonzero(np.isin(table.loop_edge, edges))[0]
    faces = np.unique(lf[loops])
    # 只融并选中边连起来的面：按选中边做并查集
    parent = {int(f): int(f) for f in faces}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_edge: dict[int, int] = {}
    for loop in loops:
        e, f = int(table.loop_edge[loop]), int(lf[loop])
        if e in by_edge:
            ra, rb = find(by_edge[e]), find(f)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        else:
            by_edge[e] = f
    groups: dict[int, list[int]] = {}
    for f in faces:
        groups.setdefault(find(int(f)), []).append(int(f))
    total = 0
    mesh.face_sel[:] = False
    targets = [np.asarray(g, np.int64) for g in groups.values() if len(g) > 1]
    if not targets:
        return 0
    for group in targets:
        mesh.face_sel[group] = True
    total = dissolve_faces(mesh, np.concatenate(targets))
    return total


def dissolve_verts(mesh: EditMesh) -> int:
    """融并点：去掉选中的点，周围的面合成一个面（点在边界上、或者只连两条边时把两条边接成一条）。"""
    verts = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
    if not len(verts):
        return 0
    lf = mesh.loop_face()
    loops = np.nonzero(np.isin(mesh.loop_vert, verts))[0]
    faces = np.unique(lf[loops])
    mesh.face_sel[:] = False
    mesh.face_sel[faces] = True
    mesh._last_remap = None
    count = dissolve_faces(mesh, faces) if len(faces) > 1 else 0
    remap = getattr(mesh, "_last_remap", None)
    if remap is not None:
        verts = remap[verts]
        verts = verts[verts >= 0]
    # 合并后的面里去掉这些点（它们已经不在别的面上）
    remove_loop = np.isin(mesh.loop_vert, verts)
    if remove_loop.any():
        sizes = np.add.reduceat((~remove_loop).astype(np.int64), mesh.face_start[:-1]) if mesh.face_count else \
            np.zeros(0, np.int64)
        good = sizes >= 3
        keep_loop = ~remove_loop & np.repeat(good, mesh.face_sizes())
        mesh.loop_vert = mesh.loop_vert[keep_loop]
        mesh.loop_uv = mesh.loop_uv[keep_loop]
        mesh.face_start = np.concatenate([[0], np.cumsum(sizes[good])]).astype(np.int64)
        mesh.face_mat = mesh.face_mat[good]
        mesh.face_smooth = mesh.face_smooth[good]
        mesh.face_sel = mesh.face_sel[good]
        mesh.face_hide = mesh.face_hide[good]
        rebuild(mesh, np.ones(mesh.face_count, bool))
    mesh.select_all("DESELECT")
    return count


def fill(mesh: EditMesh) -> str:
    """F：选中两个点连一条边；选中的边围成一圈（或选中点在边界上排成一圈）就补一个面。返回做了什么。"""
    verts = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
    table = mesh.edges()
    if len(verts) == 2:
        a, b = int(verts[0]), int(verts[1])
        exists = np.any((table.edges[:, 0] == min(a, b)) & (table.edges[:, 1] == max(a, b)))
        if exists:
            return ""
        mesh.loose_edges = np.concatenate([mesh.loose_edges, [[a, b]]])
        mesh.invalidate()
        mesh.vert_sel[[a, b]] = True
        mesh.flush_from_verts()
        return "EDGE"
    if len(verts) < 3:
        return ""
    chosen = np.zeros(mesh.vert_count, bool)
    chosen[verts] = True
    cand = (table.face_count <= 1) & chosen[table.edges[:, 0]] & chosen[table.edges[:, 1]]
    edge_ids = np.nonzero(cand)[0]
    if len(edge_ids) < 3:
        return ""
    # 沿边界边把选中的点串成一圈
    adjacency: dict[int, list[int]] = {}
    for a, b in table.edges[edge_ids]:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if any(len(n) != 2 for n in adjacency.values()):
        return ""
    start = next(iter(adjacency))
    cycle = [start]
    prev = None
    cur = start
    while True:
        options = [n for n in adjacency[cur] if n != prev]
        nxt = options[0] if options else None
        if nxt is None or nxt == start:
            break
        cycle.append(nxt)
        prev, cur = cur, nxt
        if len(cycle) > len(adjacency):
            return ""
    if len(cycle) != len(adjacency):
        return ""
    # 走向：和这些边上已有的面相反
    edge_loop = _edge_first_loop(mesh)
    nxt_loop = mesh.loop_next()
    flip = False
    for e in edge_ids:
        loop = edge_loop[e]
        if loop >= 0:
            a, b = int(mesh.loop_vert[loop]), int(mesh.loop_vert[nxt_loop[loop]])
            i = cycle.index(a)
            flip = cycle[(i + 1) % len(cycle)] == b
            break
    if flip:
        cycle = cycle[::-1]
    uv = []
    for v in cycle:
        loops = np.nonzero(mesh.loop_vert == v)[0]
        uv.append(mesh.loop_uv[loops[0]] if len(loops) else np.zeros(2, np.float32))
    new = FaceList()
    mat = 0
    loops_any = np.nonzero(np.isin(mesh.loop_vert, cycle))[0]
    if len(loops_any):
        mat = int(mesh.face_mat[mesh.loop_face()[loops_any[0]]])
    new.add(cycle, uv, mat, False, True)
    rebuild(mesh, np.ones(mesh.face_count, bool), new, compact=False)
    mesh.flush_from_faces()
    return "FACE"


# ====================================================================== 复制、拆分、分离
def duplicate(mesh: EditMesh) -> np.ndarray:
    """复制选中的面（连着它们的点）；没选面时复制选中的边和点（成散边、散点）。返回新点号。"""
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
    table = mesh.edges()
    lf = mesh.loop_face()
    base = mesh.vert_count
    if len(faces):
        loops = np.nonzero(np.isin(lf, faces))[0]
        verts = np.unique(mesh.loop_vert[loops])
    else:
        verts = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
    if not len(verts):
        return np.zeros(0, np.int64)
    dup = np.full(base, -1, np.int64)
    dup[verts] = base + np.arange(len(verts))
    new = FaceList()
    for f in faces:
        s, e = mesh.face_start[f], mesh.face_start[f + 1]
        new.add(dup[mesh.loop_vert[s:e]], mesh.loop_uv[s:e], mesh.face_mat[f], mesh.face_smooth[f], True)
    loose = mesh.loose_edges
    if not len(faces):
        pairs = table.edges[mesh.edge_sel]
        if len(pairs):
            loose = np.concatenate([loose, dup[pairs]])
    face_count = mesh.face_count
    rebuild(mesh, np.ones(face_count, bool), new, mesh.verts[verts].copy(), loose_edges=loose, compact=False)
    new_faces = np.arange(face_count, mesh.face_count)
    if len(new_faces):
        select_new(mesh, faces=new_faces)
    else:
        select_new(mesh, verts=base + np.arange(len(verts)))
    del lf
    return base + np.arange(len(verts))


def split(mesh: EditMesh) -> None:
    """Y：选中的面和其余部分断开（边界上的点复制一份）。"""
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
    if not len(faces):
        return
    lf = mesh.loop_face()
    region = np.zeros(mesh.face_count, bool)
    region[faces] = True
    inside = np.zeros(mesh.vert_count, bool)
    outside = np.zeros(mesh.vert_count, bool)
    inside[mesh.loop_vert[region[lf]]] = True
    outside[mesh.loop_vert[~region[lf]]] = True
    shared = np.nonzero(inside & outside)[0]
    base = mesh.vert_count
    dup = np.full(base, -1, np.int64)
    dup[shared] = base + np.arange(len(shared))
    loop_vert = mesh.loop_vert.copy()
    loops = np.nonzero(region[lf] & (dup[loop_vert] >= 0))[0]
    loop_vert[loops] = dup[loop_vert[loops]]
    rebuild(mesh, np.ones(mesh.face_count, bool), None, mesh.verts[shared].copy(), loop_vert_override=loop_vert,
            compact=False)
    select_new(mesh, faces=faces)


def separate_selection(mesh: EditMesh) -> EditMesh | None:
    """P：选中的面拿出来做成另一个网格（返回它），从这个网格里删掉。"""
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
    if not len(faces):
        return None
    keep = np.zeros(mesh.face_count, bool)
    keep[faces] = True
    other = mesh.copy()
    rebuild(other, keep)
    other.select_all("DESELECT")
    rebuild(mesh, ~keep)
    mesh.select_all("DESELECT")
    return other


# ====================================================================== 细分
def subdivide(mesh: EditMesh, cuts: int = 1, smooth: float = 0.0) -> int:
    """细分选中的面：每条边切 cuts 刀，四边形分成网格，三角形分成小三角形，多边形以中心点分成四边形。
    返回新面数。"""
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
    if not len(faces):
        return 0
    cuts = max(1, int(cuts))
    table = mesh.edges()
    lf = mesh.loop_face()
    region = np.zeros(mesh.face_count, bool)
    region[faces] = True
    cut_edges = np.unique(table.loop_edge[region[lf]])
    # 每条要切的边上加 cuts 个点（按边的正向 a<b 排）
    base = mesh.vert_count
    edge_points = {}
    new_pos = []
    count = base
    for e in cut_edges:
        a, b = table.edges[e]
        pts = []
        for k in range(1, cuts + 1):
            t = k / (cuts + 1)
            new_pos.append(mesh.verts[a] * (1 - t) + mesh.verts[b] * t)
            pts.append(count)
            count += 1
        edge_points[int(e)] = np.asarray(pts, np.int64)
    nxt = mesh.loop_next()
    new = FaceList()

    def edge_run(loop):
        """面角 loop 到下一个面角这条边上的新点（按面的走向排）。"""
        e = int(table.loop_edge[loop])
        pts = edge_points.get(e)
        if pts is None:
            return np.zeros(0, np.int64)
        a = mesh.loop_vert[loop]
        return pts if a == table.edges[e][0] else pts[::-1]

    def lerp_uv(u0, u1, k):
        t = k / (cuts + 1)
        return u0 * (1 - t) + u1 * t

    grid_points = []
    for f in faces:
        s, e = mesh.face_start[f], mesh.face_start[f + 1]
        loops = np.arange(s, e)
        n = len(loops)
        verts = mesh.loop_vert[loops]
        uvs = mesh.loop_uv[loops]
        mat, sm = mesh.face_mat[f], mesh.face_smooth[f]
        runs = [edge_run(loop) for loop in loops]
        if n == 4 and cuts >= 1:
            # 四边形：(cuts+2)×(cuts+2) 的网格
            m = cuts + 2
            grid = np.empty((m, m), np.int64)
            uvg = np.zeros((m, m, 2), np.float32)
            corners = verts
            grid[0, 0], grid[0, m - 1], grid[m - 1, m - 1], grid[m - 1, 0] = corners
            uvg[0, 0], uvg[0, m - 1], uvg[m - 1, m - 1], uvg[m - 1, 0] = uvs
            # 边：0→1 在第 0 行，1→2 在最后一列，2→3 在最后一行（反向），3→0 在第 0 列（反向）
            for k in range(cuts):
                grid[0, k + 1] = runs[0][k]
                uvg[0, k + 1] = lerp_uv(uvs[0], uvs[1], k + 1)
                grid[k + 1, m - 1] = runs[1][k]
                uvg[k + 1, m - 1] = lerp_uv(uvs[1], uvs[2], k + 1)
                grid[m - 1, m - 2 - k] = runs[2][k]
                uvg[m - 1, m - 2 - k] = lerp_uv(uvs[2], uvs[3], k + 1)
                grid[m - 2 - k, 0] = runs[3][k]
                uvg[m - 2 - k, 0] = lerp_uv(uvs[3], uvs[0], k + 1)
            for i in range(1, m - 1):
                for j in range(1, m - 1):
                    u, v = j / (m - 1), i / (m - 1)
                    p = (mesh.verts[corners[0]] * (1 - u) * (1 - v) + mesh.verts[corners[1]] * u * (1 - v) +
                         mesh.verts[corners[2]] * u * v + mesh.verts[corners[3]] * (1 - u) * v)
                    new_pos.append(p)
                    grid[i, j] = count
                    count += 1
                    uvg[i, j] = (uvs[0] * (1 - u) * (1 - v) + uvs[1] * u * (1 - v) + uvs[2] * u * v +
                                 uvs[3] * (1 - u) * v)
            for i in range(m - 1):
                for j in range(m - 1):
                    new.add([grid[i, j], grid[i, j + 1], grid[i + 1, j + 1], grid[i + 1, j]],
                            [uvg[i, j], uvg[i, j + 1], uvg[i + 1, j + 1], uvg[i + 1, j]], mat, sm, True)
            continue
        if n == 3 and cuts == 1:
            m01, m12, m20 = runs[0][0], runs[1][0], runs[2][0]
            u01 = lerp_uv(uvs[0], uvs[1], 1)
            u12 = lerp_uv(uvs[1], uvs[2], 1)
            u20 = lerp_uv(uvs[2], uvs[0], 1)
            new.add([verts[0], m01, m20], [uvs[0], u01, u20], mat, sm, True)
            new.add([m01, verts[1], m12], [u01, uvs[1], u12], mat, sm, True)
            new.add([m20, m12, verts[2]], [u20, u12, uvs[2]], mat, sm, True)
            new.add([m01, m12, m20], [u01, u12, u20], mat, sm, True)
            continue
        # 其它：中心加一点，每个原角和两侧的新点围成四边形（每边只切一刀时），切多刀时扇形连到中心
        center = mesh.verts[verts].mean(axis=0)
        new_pos.append(center)
        c = count
        count += 1
        cu = uvs.mean(axis=0)
        ring = []
        ring_uv = []
        for k in range(n):
            ring.append(verts[k])
            ring_uv.append(uvs[k])
            for j, p in enumerate(runs[k]):
                ring.append(p)
                ring_uv.append(lerp_uv(uvs[k], uvs[(k + 1) % n], j + 1))
        r = len(ring)
        for k in range(r):
            new.add([c, ring[k], ring[(k + 1) % r]], [cu, ring_uv[k], ring_uv[(k + 1) % r]], mat, sm, True)
    del grid_points, nxt
    # 没被细分的相邻面：边上多了点，把它们插进面里（成多边形，和 Blender 不三角化时一样）
    loop_vert = []
    loop_uv = []
    sizes = []
    keep_mask = ~region
    adjust_faces = np.nonzero(keep_mask)[0]
    touched = np.zeros(mesh.face_count, bool)
    if len(cut_edges):
        edge_cut = np.zeros(len(table.edges), bool)
        edge_cut[cut_edges] = True
        touched = np.logical_or.reduceat(edge_cut[table.loop_edge], mesh.face_start[:-1]) & keep_mask
    rebuilt = FaceList()
    for f in np.nonzero(touched)[0]:
        s, e = mesh.face_start[f], mesh.face_start[f + 1]
        verts_out = []
        uvs_out = []
        n = e - s
        for k, loop in enumerate(range(s, e)):
            verts_out.append(mesh.loop_vert[loop])
            uvs_out.append(mesh.loop_uv[loop])
            run = edge_run(loop)
            for j, p in enumerate(run):
                verts_out.append(p)
                uvs_out.append(lerp_uv(mesh.loop_uv[loop], mesh.loop_uv[s + (k + 1) % n], j + 1))
        rebuilt.add(verts_out, uvs_out, mesh.face_mat[f], mesh.face_smooth[f], False)
    del loop_vert, loop_uv, sizes, adjust_faces
    for i in range(len(rebuilt)):
        new.add(rebuilt.verts[i], rebuilt.uvs[i], rebuilt.mats[i], rebuilt.smooth[i], False)
    remove = region | touched
    new_count = len(new)
    rebuild(mesh, ~remove, new, np.asarray(new_pos) if new_pos else None, compact=False)
    if smooth > 0.0:
        smooth_verts(mesh, np.arange(base, mesh.vert_count), smooth, 1)
    mesh.flush_from_faces()
    return new_count


# ====================================================================== 三角化、三角转四边
def triangulate_faces(mesh: EditMesh) -> int:
    from .editmesh import triangulate

    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide & (mesh.face_sizes() > 3))[0]
    if not len(faces):
        return 0
    tris, face_of_tri = triangulate(mesh)
    chosen = np.isin(face_of_tri, faces)
    new = FaceList()
    for (a, b, c), f in zip(tris[chosen], face_of_tri[chosen]):
        loops = [a, b, c]
        new.add(mesh.loop_vert[loops], mesh.loop_uv[loops], mesh.face_mat[f], mesh.face_smooth[f], True)
    remove = np.zeros(mesh.face_count, bool)
    remove[faces] = True
    rebuild(mesh, ~remove, new, compact=False)
    mesh.flush_from_faces()
    return len(new)


def tris_to_quads(mesh: EditMesh, max_angle: float = 40.0) -> int:
    """Alt+J：选中的相邻三角形两两拼成四边形（两个面夹角小于 max_angle、拼出来是凸四边形的才拼）。"""
    table = mesh.edges()
    tri = np.nonzero(mesh.face_sel & ~mesh.face_hide & (mesh.face_sizes() == 3))[0]
    if len(tri) < 2:
        return 0
    lf = mesh.loop_face()
    fn = mesh.face_normals()
    is_tri = np.zeros(mesh.face_count, bool)
    is_tri[tri] = True
    loops = np.nonzero(is_tri[lf])[0]
    by_edge: dict[int, list[int]] = {}
    for loop in loops:
        by_edge.setdefault(int(table.loop_edge[loop]), []).append(int(loop))
    candidates = []
    cos_limit = math.cos(math.radians(max_angle))
    for e, pair in by_edge.items():
        if len(pair) != 2:
            continue
        f1, f2 = int(lf[pair[0]]), int(lf[pair[1]])
        dot = float(fn[f1] @ fn[f2])
        if dot < cos_limit:
            continue
        a, b = table.edges[e]
        length = float(np.linalg.norm(mesh.verts[a] - mesh.verts[b]))
        candidates.append((-dot, -length, e, pair))
    candidates.sort()
    used = np.zeros(mesh.face_count, bool)
    new = FaceList()
    nxt = mesh.loop_next()
    for _d, _l, e, (l1, l2) in candidates:
        f1, f2 = int(lf[l1]), int(lf[l2])
        if used[f1] or used[f2]:
            continue
        # f1 的走向：l1 → nxt(l1) 是共用边；四边形 = [l1 的点, f2 里对面的点, nxt(l1) 的点, f1 剩下的点]
        third1 = nxt[nxt[l1]]
        third2 = nxt[nxt[l2]]
        loops4 = [l1, third2, nxt[l1], third1]
        quad = mesh.verts[mesh.loop_vert[loops4]]
        if not _convex(quad):
            continue
        used[f1] = used[f2] = True
        new.add(mesh.loop_vert[loops4], mesh.loop_uv[loops4], mesh.face_mat[f1], mesh.face_smooth[f1], True)
    if not len(new):
        return 0
    rebuild(mesh, ~used, new, compact=False)
    mesh.flush_from_faces()
    return len(new)


def _convex(quad: np.ndarray) -> bool:
    n = np.cross(quad[1] - quad[0], quad[2] - quad[0]) + np.cross(quad[2] - quad[0], quad[3] - quad[0])
    for k in range(4):
        a, b, c = quad[k], quad[(k + 1) % 4], quad[(k + 2) % 4]
        if float(np.cross(b - a, c - b) @ n) <= 0.0:
            return False
    return True


# ====================================================================== 法线
def flip_normals(mesh: EditMesh, faces: np.ndarray | None = None) -> None:
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0] if faces is None else faces
    for f in faces:
        s, e = mesh.face_start[f], mesh.face_start[f + 1]
        # 第一个面角不动，后面的倒过来（走向反了，点和 UV 一起倒）
        mesh.loop_vert[s + 1:e] = mesh.loop_vert[s + 1:e][::-1].copy()
        mesh.loop_uv[s + 1:e] = mesh.loop_uv[s + 1:e][::-1].copy()
    mesh.invalidate()


def recalc_normals(mesh: EditMesh, inside: bool = False) -> int:
    """Shift+N：让选中面的走向一致，并且朝外（按每块的有向体积判断）。返回翻转的面数。"""
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
    if not len(faces):
        return 0
    table = mesh.edges()
    lf = mesh.loop_face()
    nxt = mesh.loop_next()
    chosen = np.zeros(mesh.face_count, bool)
    chosen[faces] = True
    by_edge: dict[int, list[int]] = {}
    for loop in np.nonzero(chosen[lf])[0]:
        by_edge.setdefault(int(table.loop_edge[loop]), []).append(int(loop))
    flip = np.zeros(mesh.face_count, bool)
    seen = np.zeros(mesh.face_count, bool)
    face_loops: dict[int, list[int]] = {}
    for e, loops in by_edge.items():
        for loop in loops:
            face_loops.setdefault(int(lf[loop]), []).append(loop)
    flipped = 0
    for start in faces:
        if seen[start]:
            continue
        component = []
        stack = [int(start)]
        seen[start] = True
        while stack:
            f = stack.pop()
            component.append(f)
            for loop in face_loops.get(f, []):
                for other in by_edge.get(int(table.loop_edge[loop]), []):
                    g = int(lf[other])
                    if g == f or seen[g]:
                        continue
                    # 两个面在这条边上应该走相反的方向
                    same_dir = bool(mesh.loop_vert[loop] == mesh.loop_vert[other])
                    flip[g] = bool(flip[f]) ^ same_dir
                    seen[g] = True
                    stack.append(g)
        comp = np.asarray(component, np.int64)
        # 按翻转后的走向算有向体积，负的话整块再翻
        volume = 0.0
        for f in comp:
            s, e = mesh.face_start[f], mesh.face_start[f + 1]
            pts = mesh.verts[mesh.loop_vert[s:e]]
            if flip[f]:
                pts = pts[::-1]
            for k in range(1, len(pts) - 1):
                volume += float(np.dot(pts[0], np.cross(pts[k], pts[k + 1])))
        if (volume < 0) != inside:
            flip[comp] = ~flip[comp]
    targets = np.nonzero(flip & chosen)[0]
    flipped = len(targets)
    if flipped:
        flip_normals(mesh, targets)
    del nxt
    return flipped


# ====================================================================== 平滑、隐藏
def smooth_verts(mesh: EditMesh, verts: np.ndarray | None = None, factor: float = 0.5, repeat: int = 1) -> None:
    verts = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0] if verts is None else np.asarray(verts, np.int64)
    if not len(verts):
        return
    table = mesh.edges()
    edges = table.edges
    chosen = np.zeros(mesh.vert_count, bool)
    chosen[verts] = True
    boundary = np.zeros(mesh.vert_count, bool)
    open_edges = edges[table.face_count == 1]
    boundary[open_edges.reshape(-1)] = True
    for _ in range(max(1, int(repeat))):
        acc = np.zeros_like(mesh.verts)
        count = np.zeros(mesh.vert_count)
        np.add.at(acc, edges[:, 0], mesh.verts[edges[:, 1]])
        np.add.at(acc, edges[:, 1], mesh.verts[edges[:, 0]])
        np.add.at(count, edges[:, 0], 1)
        np.add.at(count, edges[:, 1], 1)
        avg = acc / np.maximum(count, 1)[:, None]
        move = chosen & (count > 0) & ~boundary
        mesh.verts[move] = mesh.verts[move] * (1 - factor) + avg[move] * factor
    mesh.geometry_version += 1


def hide(mesh: EditMesh, unselected: bool = False) -> None:
    if unselected:
        mesh.vert_hide |= ~mesh.vert_sel
        mesh.face_hide |= ~mesh.face_sel
    else:
        if "FACE" in mesh.select_mode:
            mesh.face_hide |= mesh.face_sel
            # 只属于隐藏面的点也藏起来
            lf = mesh.loop_face()
            visible = np.zeros(mesh.vert_count, bool)
            visible[mesh.loop_vert[~mesh.face_hide[lf]]] = True
            mesh.vert_hide |= ~visible & np.isin(np.arange(mesh.vert_count), mesh.loop_vert)
        else:
            mesh.vert_hide |= mesh.vert_sel
            lf = mesh.loop_face()
            mesh.face_hide |= np.logical_or.reduceat(mesh.vert_hide[mesh.loop_vert], mesh.face_start[:-1]) \
                if mesh.face_count else np.zeros(0, bool)
            del lf
    mesh.select_all("DESELECT")
    mesh.geometry_version += 1


def reveal(mesh: EditMesh, select: bool = True) -> None:
    was_v, was_f = mesh.vert_hide.copy(), mesh.face_hide.copy()
    mesh.vert_hide[:] = False
    mesh.face_hide[:] = False
    if select:
        mesh.vert_sel |= was_v
        mesh.face_sel |= was_f
        mesh.flush()
    mesh.geometry_version += 1


# ====================================================================== 接上另一份网格（编辑模式里加形体）
def append(mesh: EditMesh, other: EditMesh, select: bool = True) -> np.ndarray:
    """把 other 接到后面。select 时只选中新加的部分。返回新点号。"""
    base_v, base_l, base_f = mesh.vert_count, mesh.loop_count, mesh.face_count
    mesh.verts = np.concatenate([mesh.verts, other.verts])
    mesh.loop_vert = np.concatenate([mesh.loop_vert, other.loop_vert + base_v]).astype(np.int64)
    mesh.face_start = np.concatenate([mesh.face_start[:-1], other.face_start + base_l]).astype(np.int64)
    mesh.loop_uv = np.concatenate([mesh.loop_uv, other.loop_uv]).astype(np.float32)
    mesh.face_mat = np.concatenate([mesh.face_mat, np.zeros(other.face_count, np.int32)]).astype(np.int32)
    mesh.face_smooth = np.concatenate([mesh.face_smooth, other.face_smooth]).astype(bool)
    mesh.loose_edges = np.concatenate([mesh.loose_edges, other.loose_edges + base_v]).astype(np.int64).reshape(-1, 2)
    mesh.vert_hide = np.concatenate([mesh.vert_hide, np.zeros(other.vert_count, bool)])
    mesh.face_hide = np.concatenate([mesh.face_hide, np.zeros(other.face_count, bool)])
    if select:
        mesh.vert_sel = np.concatenate([np.zeros(base_v, bool), np.ones(other.vert_count, bool)])
        mesh.face_sel = np.concatenate([np.zeros(base_f, bool), np.ones(other.face_count, bool)])
        mesh.active = None
    else:
        mesh.vert_sel = np.concatenate([mesh.vert_sel, np.zeros(other.vert_count, bool)])
        mesh.face_sel = np.concatenate([mesh.face_sel, np.zeros(other.face_count, bool)])
    mesh.invalidate()
    return np.arange(base_v, mesh.vert_count, dtype=np.int64)


# ====================================================================== 沿边拆开（撕开 V、按边拆分）
def _find(parent: dict, x: int) -> int:
    root = x
    while parent.get(root, root) != root:
        root = parent[root]
    while parent.get(x, x) != root:
        parent[x], x = root, parent[x]
    return root


def edge_split(mesh: EditMesh, edges: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """沿这些边把两侧的面拆开：边上的点按被隔开的几片面各复制一份。
    返回 (新点号, 每个新点是从哪个点复制的)。"""
    table = mesh.edges()
    edges = np.nonzero(mesh.edge_sel)[0] if edges is None else np.asarray(edges, np.int64)
    empty = np.zeros(0, np.int64)
    if not len(edges) or not mesh.loop_count:
        return empty, empty
    cut = np.zeros(len(table.edges), bool)
    cut[edges] = True
    verts = np.unique(table.edges[cut].reshape(-1))
    involved = np.zeros(mesh.vert_count, bool)
    involved[verts] = True
    nxt = mesh.loop_next()
    loop_edge = table.loop_edge
    # 同一个点上的两个面角，隔着一条没切的边相邻，就算同一片
    parent: dict[int, int] = {}
    by_edge: dict[int, list[int]] = {}
    for loop in np.nonzero(involved[mesh.loop_vert] | involved[mesh.loop_vert[nxt]])[0]:
        e = int(loop_edge[loop])
        if not cut[e]:
            by_edge.setdefault(e, []).append(int(loop))
    for e, loops in by_edge.items():
        lo = int(table.edges[e][0])
        at_lo = [l if int(mesh.loop_vert[l]) == lo else int(nxt[l]) for l in loops]
        at_hi = [int(nxt[l]) if int(mesh.loop_vert[l]) == lo else l for l in loops]
        for group in (at_lo, at_hi):
            root = _find(parent, group[0])
            for other in group[1:]:
                r = _find(parent, other)
                if r != root:
                    parent[r] = root
    loop_vert = mesh.loop_vert.copy()
    new_pos, source = [], []
    base = mesh.vert_count
    corners = np.nonzero(involved[mesh.loop_vert])[0]
    by_vert: dict[int, dict[int, list[int]]] = {}
    for loop in corners:
        v = int(mesh.loop_vert[loop])
        by_vert.setdefault(v, {}).setdefault(_find(parent, int(loop)), []).append(int(loop))
    for v, groups in by_vert.items():
        if len(groups) < 2:
            continue
        ordered = sorted(groups.values(), key=lambda g: min(g))
        for group in ordered[1:]:
            new_id = base + len(new_pos)
            new_pos.append(mesh.verts[v].copy())
            source.append(v)
            loop_vert[group] = new_id
    if not new_pos:
        return empty, empty
    sel = mesh.vert_sel.copy()
    rebuild(mesh, np.ones(mesh.face_count, bool), None, np.asarray(new_pos), loop_vert_override=loop_vert,
            compact=False)
    new_ids = base + np.arange(len(new_pos), dtype=np.int64)
    source = np.asarray(source, np.int64)
    mesh.vert_sel[:len(sel)] = sel
    mesh.vert_sel[new_ids] = sel[source]
    mesh.flush_from_verts()
    return new_ids, source


# ====================================================================== 补洞（Alt+F：三角形填充）
def fill_triangles(mesh: EditMesh) -> int:
    """选中的边界边围成的圈补成三角形面。返回补了几个三角形。"""
    old = mesh.face_count
    if fill(mesh) != "FACE":
        return 0
    mesh.face_sel[:] = False
    mesh.face_sel[old:] = True
    return triangulate_faces(mesh) or 1


# ====================================================================== 桥接循环边
def _chains(mesh: EditMesh, edges: np.ndarray) -> list[tuple[list[int], bool]] | None:
    """选中的边连成的几条线或几个圈：[(按顺序的点, 是否闭合)]。有分叉时返回 None。"""
    table = mesh.edges()
    adj: dict[int, list[int]] = {}
    for a, b in table.edges[edges]:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    if any(len(n) > 2 for n in adj.values()):
        return None
    seen: set[int] = set()
    out = []
    for start in sorted(adj, key=lambda v: (len(adj[v]), v)):
        if start in seen:
            continue
        walk, prev, cur = [], None, start
        while cur is not None and cur not in seen:
            seen.add(cur)
            walk.append(cur)
            following = [n for n in adj[cur] if n != prev and n not in seen]
            prev, cur = cur, (following[0] if following else None)
        closed = len(walk) > 2 and walk[0] in adj[walk[-1]]
        out.append((walk, closed))
    return out


def bridge_edge_loops(mesh: EditMesh) -> int:
    """两圈（或两条）选中的边之间连上一圈四边形。选中了面时先去掉这些面，用它们的边界来桥接。返回加了几个面。"""
    if mesh.face_sel.any():
        faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
        _loops, a_ids, b_ids = region_boundary(mesh, faces)
        if not len(a_ids):
            return 0
        boundary = np.unique(np.concatenate([a_ids, b_ids]))
        remove = np.zeros(mesh.face_count, bool)
        remove[faces] = True
        remap = rebuild(mesh, ~remove, loose_edges=mesh.loose_edges)
        mesh.select_all("DESELECT")
        kept = remap[boundary]
        mesh.vert_sel[kept[kept >= 0]] = True
        mesh.flush_from_verts()
    table = mesh.edges()
    chosen = np.nonzero(mesh.edge_sel)[0]
    chains = _chains(mesh, chosen) if len(chosen) else None
    if not chains or len(chains) != 2:
        return 0
    (a, a_closed), (b, b_closed) = chains
    if a_closed != b_closed or len(a) != len(b) or len(a) < 2:
        return 0
    pa = mesh.verts[a]
    n = len(a)
    best, best_cost = None, np.inf
    for direction in (1, -1):
        order = b if direction == 1 else b[::-1]
        offsets = range(n) if a_closed else [0]
        for off in offsets:
            cand = order[off:] + order[:off]
            cost = float(np.linalg.norm(pa - mesh.verts[cand], axis=1).sum())
            if cost < best_cost:
                best, best_cost = cand, cost
    b = best
    count = n if a_closed else n - 1
    # 走向：和第一圈上已有的面相反
    nxt = mesh.loop_next()
    flip = False
    for i in range(count):
        u, w = a[i], a[(i + 1) % n]
        if np.any((mesh.loop_vert == u) & (mesh.loop_vert[nxt] == w)):
            flip = True
            break
        if np.any((mesh.loop_vert == w) & (mesh.loop_vert[nxt] == u)):
            break
    new = FaceList()
    for i in range(count):
        j = (i + 1) % n
        quad = [a[i], a[j], b[j], b[i]]
        uv = [(i / count, 0.0), (j / count if j else 1.0, 0.0), (j / count if j else 1.0, 1.0), (i / count, 1.0)]
        if flip:
            quad, uv = quad[::-1], uv[::-1]
        new.add(quad, uv, 0, False, True)
    face_count = mesh.face_count
    rebuild(mesh, np.ones(face_count, bool), new, compact=False)
    mesh.face_sel[:] = False
    mesh.face_sel[face_count:] = True
    mesh.flush_from_faces()
    return len(new)


# ====================================================================== 旋转边
def edge_rotate(mesh: EditMesh, ccw: bool = False) -> int:
    """两个面之间的边转到相邻的点上（两面合成的多边形换一条对角线）。返回转了几条边。"""
    table = mesh.edges()
    nxt = mesh.loop_next()
    lf = mesh.loop_face()
    used: set[int] = set()
    new = FaceList()
    remove = np.zeros(mesh.face_count, bool)
    turned = []
    for e in np.nonzero(mesh.edge_sel & (table.face_count == 2))[0]:
        loops = np.nonzero(table.loop_edge == e)[0]
        f1, f2 = int(lf[loops[0]]), int(lf[loops[1]])
        if f1 == f2 or f1 in used or f2 in used:
            continue
        ring = []
        for loop, face in ((loops[0], f1), (loops[1], f2)):
            # 从这条边的终点开始绕这个面走到起点（不含起点）
            s, t = mesh.face_start[face], mesh.face_start[face + 1]
            size = t - s
            start = nxt[loop]
            for k in range(size - 1):
                l = s + ((start - s + k) % size)
                ring.append((int(mesh.loop_vert[l]), mesh.loop_uv[l].astype(np.float64)))
        m = len(ring)
        len1 = int(mesh.face_start[f1 + 1] - mesh.face_start[f1]) - 1     # 第一个面在 ring 里占几个点
        # 两个面合成的多边形 ring = [边的终点, 第一个面其余的点, 边的起点, 第二个面其余的点]，
        # 原来的边是 ring[0]—ring[len1] 这条对角线；顺时针转就是两头各往后挪一个点
        i, j = 0, len1
        step = -1 if ccw else 1
        i, j = (i + step) % m, (j + step) % m
        first = [ring[(i + k) % m] for k in range((j - i) % m + 1)]
        second = [ring[(j + k) % m] for k in range((i - j) % m + 1)]
        if len(first) < 3 or len(second) < 3:
            continue
        for poly, face in ((first, f1), (second, f2)):
            new.add([v for v, _uv in poly], [uv for _v, uv in poly], mesh.face_mat[face], mesh.face_smooth[face])
        remove[[f1, f2]] = True
        used.update((f1, f2))
        turned.append((ring[i][0], ring[j][0]))
    if not turned:
        return 0
    rebuild(mesh, ~remove, new, compact=False)
    table = mesh.edges()
    mesh.select_all("DESELECT")
    for a, b in turned:
        hit = np.nonzero((table.edges[:, 0] == min(a, b)) & (table.edges[:, 1] == max(a, b)))[0]
        mesh.edge_sel[hit] = True
    mesh.flush_from_edges()
    return len(turned)


# ====================================================================== 戳面
def poke(mesh: EditMesh, offset: float = 0.0) -> int:
    """选中的每个面在中心加一个点，连成一圈三角形（中心点沿法线挪 offset）。返回戳了几个面。"""
    faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
    if not len(faces):
        return 0
    centers = mesh.face_centers()[faces] + mesh.face_normals()[faces] * float(offset)
    base = mesh.vert_count
    new = FaceList()
    for k, f in enumerate(faces):
        s, t = mesh.face_start[f], mesh.face_start[f + 1]
        verts = mesh.loop_vert[s:t]
        uvs = mesh.loop_uv[s:t].astype(np.float64)
        center_uv = uvs.mean(axis=0)
        for i in range(t - s):
            j = (i + 1) % (t - s)
            new.add([verts[i], verts[j], base + k], [uvs[i], uvs[j], center_uv], mesh.face_mat[f],
                    mesh.face_smooth[f], True)
    remove = np.zeros(mesh.face_count, bool)
    remove[faces] = True
    rebuild(mesh, ~remove, new, centers, compact=False)
    mesh.flush_from_faces()
    return len(faces)
