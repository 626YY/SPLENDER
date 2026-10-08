"""编辑模式里交互较多的操作（照 Blender）：切刀 K、连接点 J、撕开 V、按边拆分、滑移边（GG）和滑移点（Shift+V）、
只挤出边或点、挤出到鼠标（Ctrl+右键）、镜像选择、补洞 Alt+F，以及工具栏里拖动类工具的入口。"""
from __future__ import annotations

import math

import numpy as np

from ..core import ops
from ..core.keymap import (DOUBLE, LEFTMOUSE, MIDDLEMOUSE, MOUSEMOVE, PRESS, RELEASE, RIGHTMOUSE, WHEEL,
                           WHEELDOWN, WHEELUP)
from ..core.ops import CANCELLED, FINISHED, PASS_THROUGH, RUNNING_MODAL, Operator
from ..core.props import BoolProperty, EnumProperty, FloatProperty
from ..doc.objects import to_internal
from ..mesh import cut as C
from ..mesh import select as S
from ..mesh import topology as T
from .mesh_edit_ops import _EditOp, _engine, _hint, _redraw, _xray, edit_mode, edit_session
from .view3d_ops import view3d

KNIFE_LINE = (0.25, 1.0, 0.45)           # 切线（亮绿）
KNIFE_CUT = (1.0, 0.85, 0.2)             # 切到的点（黄）
KNIFE_HOVER = (0.35, 0.8, 1.0)           # 鼠标下的刀点（蓝）


def _pixel(editor, event):
    if event is not None and getattr(event, "source", None) is editor.widget:
        return editor.pixel(event)
    return editor.cursor_pixel()


def _screen(view, points) -> list[tuple[float, float]]:
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(pts):
        return []
    sx, sy, _front, _dist = S.project(view, pts)
    return list(zip(sx.tolist(), sy.tolist()))


def _probe_key(view) -> bytes:
    return np.asarray(view.camera.view_projection(view.aspect), np.float64).tobytes() + \
        bytes(str((view.width, view.height)), "ascii")


class _ProbeCache:
    """被挡住的判断要读回整个几何缓冲：视角不变时只读一次。"""

    def __init__(self, engine, view) -> None:
        self.engine = engine
        self.view = view
        self.key = None
        self.probe = None

    def get(self):
        key = _probe_key(self.view)
        if self.probe is None or key != self.key:
            self.probe = S.DepthProbe(self.engine, self.view)
            self.key = key
        return self.probe


def _edge_param(view, mesh, edge: int, x: float, y: float) -> float:
    """鼠标视线和这条边最近的地方在边上的参数（0..1）。"""
    a, b = (int(v) for v in mesh.edges().edges[edge])
    pa, pb = mesh.verts[a], mesh.verts[b]
    origin, direction = view.camera.ray(x, y, view.width, view.height)
    origin = np.asarray(origin, np.float64)
    direction = np.asarray(direction, np.float64)
    u = pb - pa
    w0 = pa - origin
    aa, bb, cc = u @ u, u @ direction, direction @ direction
    dd, ee = u @ w0, direction @ w0
    den = aa * cc - bb * bb
    s = (bb * ee - cc * dd) / den if abs(den) > 1e-18 else 0.5
    return float(np.clip(s, 0.0, 1.0))


def _face_hit(view, mesh, face: int, x: float, y: float) -> np.ndarray:
    origin, direction = view.camera.ray(x, y, view.width, view.height)
    origin = np.asarray(origin, np.float64)
    direction = np.asarray(direction, np.float64)
    center = mesh.verts[mesh.face_verts(face)].mean(axis=0)
    normal = mesh.face_normals()[face]
    den = float(direction @ normal)
    if abs(den) < 1e-12:
        return center
    t = float((center - origin) @ normal) / den
    return origin + direction * t


# ====================================================================== 切刀
@ops.register
class MeshKnifeTool(Operator):
    idname = "mesh.knife_tool"
    label = "切刀"
    description = "在面上点出一条线，沿线把面切开（K）：回车确认，右键结束这一刀，E 另起一刀，Z 切穿，Ctrl 吸附到边的中点"
    icon = "tool.knife"
    use_occlude_geometry = BoolProperty("只切看得见的", default=True, description="关掉时连背面一起切（切穿）")
    only_selected = BoolProperty("只切选中的面", default=False)
    snap_px = FloatProperty("吸附范围", default=12.0, min=1.0, max=100.0, unit="px", hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.session = edit_session(ctx)
        self.engine = _engine(ctx)
        self.engine.ctx_make_current()
        self.probes = _ProbeCache(self.engine, self.editor.view)
        self.chains: list[list[dict]] = [[]]
        self.hits: dict[tuple[int, int], list[dict]] = {}
        self.hover = None
        self.hover_hits: list[dict] = []
        self.mouse = _pixel(self.editor, event)
        self.ctrl = bool(getattr(event, "ctrl", False))
        self.shift = bool(getattr(event, "shift", False))
        self._update_hover(ctx)
        if event is not None and event.type == LEFTMOUSE and event.value == PRESS:
            self._add_point(ctx)
        self._status(ctx)
        return RUNNING_MODAL

    # ---- 刀点 ----
    def _xray(self) -> bool:
        return not self.use_occlude_geometry or _xray(self.editor)

    def _probe(self):
        return None if self._xray() else self.probes.get()

    def _snap_point(self, x: float, y: float):
        mesh = self.session.mesh
        view = self.editor.view
        xray = self._xray()
        probe = self._probe()
        radius = float(self.snap_px)
        if not self.shift:
            found = S._nearest_vert(self.engine, view, mesh, x, y, xray, radius, probe=probe)
            if found is not None:
                v = found[0]
                return C.point("VERT", v, mesh.verts[v])
            found = S._nearest_edge(self.engine, view, mesh, x, y, xray, radius, probe=probe)
            if found is not None:
                e = found[0]
                t = 0.5 if self.ctrl else _edge_param(view, mesh, e, x, y)
                a, b = (int(v) for v in mesh.edges().edges[e])
                if t <= 1e-4:
                    return C.point("VERT", a, mesh.verts[a])
                if t >= 1.0 - 1e-4:
                    return C.point("VERT", b, mesh.verts[b])
                return C.point("EDGE", e, mesh.verts[a] + (mesh.verts[b] - mesh.verts[a]) * t, t)
        face = S.face_under(view, mesh, x, y)
        if face is None or (self.only_selected and not mesh.face_sel[face]):
            return None
        return C.point("FACE", face, _face_hit(view, mesh, face, x, y))

    def _update_hover(self, ctx) -> None:
        x, y = self.mouse
        self.hover = self._snap_point(x, y)
        chain = self.chains[-1]
        if chain and self.hover is not None and len(chain) >= 2 and self._near_first(chain):
            self.hover = chain[0]
        self.hover_hits = []
        if chain and self.hover is not None:
            self.hover_hits = C.segment_hits(self.session.mesh, self.editor.view, chain[-1], self.hover,
                                             self._probe(), bool(self.only_selected))
        self._draw()

    def _near_first(self, chain) -> bool:
        first = _screen(self.editor.view, chain[0]["pos"])[0]
        return math.hypot(first[0] - self.mouse[0], first[1] - self.mouse[1]) <= float(self.snap_px)

    def _add_point(self, ctx) -> None:
        if self.hover is None:
            return
        chain = self.chains[-1]
        if chain and chain[-1] is self.hover:
            return
        if chain:
            self.hits[(len(self.chains) - 1, len(chain) - 1)] = list(self.hover_hits)
        chain.append(self.hover)
        self.hover_hits = []
        self._draw()

    def _end_chain(self) -> None:
        if self.chains[-1]:
            self.chains.append([])
        self._draw()

    def _undo_point(self) -> None:
        chain = self.chains[-1]
        if not chain and len(self.chains) > 1:
            self.chains.pop()
            chain = self.chains[-1]
        if chain:
            chain.pop()
            self.hits.pop((len(self.chains) - 1, len(chain) - 1), None)
        self._draw()

    def _full_chains(self) -> list[list[dict]]:
        out = []
        for ci, chain in enumerate(self.chains):
            if len(chain) < 2:
                continue
            full = [chain[0]]
            for k in range(len(chain) - 1):
                full += self.hits.get((ci, k), [])
                full.append(chain[k + 1])
            out.append(full)
        return out

    # ---- 显示 ----
    def _draw(self) -> None:
        view = self.editor.view
        shapes = []
        for ci, chain in enumerate(self.chains):
            pts = [p["pos"] for p in chain]
            if ci == len(self.chains) - 1 and chain and self.hover is not None:
                pts.append(self.hover["pos"])
            if len(pts) >= 2:
                shapes.append(("polyline", _screen(view, pts), KNIFE_LINE))
            cuts = [h["pos"] for k in range(len(chain)) for h in self.hits.get((ci, k), [])]
            if ci == len(self.chains) - 1:
                cuts += [h["pos"] for h in self.hover_hits]
            if cuts:
                shapes.append(("points", _screen(view, cuts), KNIFE_CUT, 5.0))
            if chain:
                shapes.append(("points", _screen(view, [p["pos"] for p in chain]), KNIFE_LINE, 6.0))
        if self.hover is not None:
            shapes.append(("points", _screen(view, [self.hover["pos"]]), KNIFE_HOVER, 8.0))
        self.editor.set_overlay_shape(("multi", tuple(shapes)) if shapes else None)

    def _status(self, ctx) -> None:
        _hint(ctx, "切刀：左键加点 · 回车或空格确认 · 右键结束这一刀 · E 另起一刀 · Z 切穿%s · Ctrl 边中点 · Shift 不吸附 · "
              "Ctrl+Z 退一点 · Esc 取消" % ("（开）" if not self.use_occlude_geometry else ""))

    # ---- 模态 ----
    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            self.mouse = _pixel(self.editor, event)
            self.ctrl, self.shift = bool(event.ctrl), bool(event.shift)
            self._update_hover(ctx)
            return RUNNING_MODAL
        if event.type in (MIDDLEMOUSE, WHEELUP, WHEELDOWN) or event.value == WHEEL:
            return PASS_THROUGH                       # 切的时候也能转视角、缩放
        if event.type in ("LEFT_CTRL", "RIGHT_CTRL", "LEFT_SHIFT", "RIGHT_SHIFT"):
            self.ctrl, self.shift = bool(event.ctrl), bool(event.shift)
            if event.value == PRESS:
                self.ctrl |= event.type in ("LEFT_CTRL", "RIGHT_CTRL")
                self.shift |= event.type in ("LEFT_SHIFT", "RIGHT_SHIFT")
            self._update_hover(ctx)
            return RUNNING_MODAL
        if event.value == DOUBLE and event.type == LEFTMOUSE:
            # 双击：加上这一点并结束这一刀
            self._add_point(ctx)
            self._end_chain()
            return RUNNING_MODAL
        if event.value == PRESS:
            key = event.type
            if key == LEFTMOUSE:
                self._add_point(ctx)
                return RUNNING_MODAL
            if key in ("RET", "NUMPAD_ENTER", "SPACE"):
                return self._confirm(ctx)
            if key == RIGHTMOUSE:
                self._end_chain()
                return RUNNING_MODAL
            if key == "E":
                self._end_chain()
                return RUNNING_MODAL
            if key == "Z" and event.ctrl:
                self._undo_point()
                return RUNNING_MODAL
            if key == "Z":
                self.use_occlude_geometry = not self.use_occlude_geometry
                self._recompute(ctx)
                self._status(ctx)
                return RUNNING_MODAL
            if key == "BACK_SPACE":
                self._undo_point()
                return RUNNING_MODAL
            if key == "ESC":
                self.cancel(ctx)
                return CANCELLED
        return RUNNING_MODAL

    def _recompute(self, ctx) -> None:
        """切穿开关变了：已经点下的各段重新算切到哪些边。"""
        mesh = self.session.mesh
        for ci, chain in enumerate(self.chains):
            for k in range(len(chain) - 1):
                self.hits[(ci, k)] = C.segment_hits(mesh, self.editor.view, chain[k], chain[k + 1], self._probe(),
                                                    bool(self.only_selected))
        self._update_hover(ctx)

    def _confirm(self, ctx) -> str:
        chains = self._full_chains()
        self._finish(ctx)
        if not chains:
            return CANCELLED
        session = self.session
        before = session.begin()
        self.engine.ctx_make_current()
        cut = C.knife(session.mesh, chains)
        if not len(cut):
            session.mesh.restore(before)
            session.update_overlay()
            return CANCELLED
        session.push("切刀", before)
        _redraw(ctx)
        return FINISHED

    def _finish(self, ctx) -> None:
        self.editor.set_overlay_shape(None)
        _hint(ctx, "")
        ctx.app.request_frame()

    def cancel(self, ctx) -> None:
        self._finish(ctx)


# ====================================================================== 连接点（J）、按边拆分、补洞
@ops.register
class MeshVertConnectPath(_EditOp):
    idname = "mesh.vert_connect_path"
    label = "连接顶点路径"
    description = "同一个面上选中的点之间连一条边，把面分开（J）"

    def apply(self, ctx, session, mesh) -> bool:
        return C.connect_verts(mesh) > 0


@ops.register
class MeshVertConnect(_EditOp):
    idname = "mesh.vert_connect"
    label = "连接顶点对"
    description = "同一个面上选中的两个点之间连一条边"

    def apply(self, ctx, session, mesh) -> bool:
        return C.connect_verts(mesh) > 0


@ops.register
class MeshEdgeSplit(_EditOp):
    idname = "mesh.edge_split"
    label = "按边拆分"
    description = "沿选中的边把两侧的面拆开（点各复制一份）；按点拆分时选中的点周围每个面各用一份"
    type = EnumProperty("拆分", items=[("EDGE", "按边", "沿选中的边拆开"),
                                     ("VERT", "按点", "选中的点周围的面全部拆开")], default="EDGE")

    def apply(self, ctx, session, mesh) -> bool:
        if self.type == "VERT":
            table = mesh.edges()
            chosen = mesh.vert_sel & ~mesh.vert_hide
            edges = np.nonzero(chosen[table.edges[:, 0]] | chosen[table.edges[:, 1]])[0]
        else:
            edges = np.nonzero(mesh.edge_sel)[0]
        new_ids, _src = T.edge_split(mesh, edges)
        return len(new_ids) > 0


@ops.register
class MeshBridgeEdgeLoops(_EditOp):
    idname = "mesh.bridge_edge_loops"
    label = "桥接循环边"
    description = "两圈（或两条）选中的边之间连上一圈面；选中两块面时去掉它们再连通"

    def apply(self, ctx, session, mesh) -> bool:
        if T.bridge_edge_loops(mesh):
            return True
        self.report(ctx, "桥接要选中点数一样的两圈边（或两块面）", "WARNING")
        return False


@ops.register
class MeshEdgeRotate(_EditOp):
    idname = "mesh.edge_rotate"
    label = "旋转边"
    description = "两个面之间的边转到相邻的点上"
    use_ccw = BoolProperty("逆时针", default=False)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.edge_sel.any())

    def apply(self, ctx, session, mesh) -> bool:
        return T.edge_rotate(mesh, bool(self.use_ccw)) > 0


@ops.register
class MeshPoke(_EditOp):
    idname = "mesh.poke"
    label = "戳面"
    description = "选中的每个面中心加一个点，连成一圈三角形"
    offset = FloatProperty("中心外移", default=0.0, min=-1000.0, max=1000.0, precision=4, unit="m")

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.face_sel.any())

    def apply(self, ctx, session, mesh) -> bool:
        return T.poke(mesh, float(self.offset)) > 0


@ops.register
class MeshLoopMultiSelect(_EditOp):
    idname = "mesh.loop_multi_select"
    label = "循环选择"
    description = "选中的每条边，把它所在的一圈循环边（或并排边）都选上"
    redo = False
    ring = BoolProperty("并排边", default=False)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.edge_sel.any())

    def execute(self, ctx) -> str:
        from ..mesh import loopcut as LC

        session = edit_session(ctx)
        mesh = session.mesh
        before = session.begin()
        chosen = np.nonzero(mesh.edge_sel)[0]
        picked = set()
        for e in chosen:
            if int(e) in picked:
                continue
            edges = LC.edge_ring(mesh, int(e))[0] if self.ring else LC.edge_loop(mesh, int(e))
            picked.update(int(x) for x in edges)
        mesh.edge_sel[np.asarray(sorted(picked), np.int64)] = True
        mesh.flush_from_edges()
        session.push_selection(self.label, before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshExtrudeMoveShrinkFatten(Operator):
    idname = "view3d.edit_mesh_extrude_move_shrink_fatten"
    label = "沿法线挤出面"
    description = "挤出选中的面，每个点沿自己的法线推出去（面不会歪）"

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.face_sel.any())

    def invoke(self, ctx, event) -> str:
        session = edit_session(ctx)
        before = session.begin()
        new = T.extrude_region(session.mesh)
        if not len(new):
            return CANCELLED
        session.push("挤出", before)
        if view3d(ctx) is not None:
            ops.call("mesh.shrink_fatten", ctx, event, merge_previous=1)
        return FINISHED


@ops.register
class MeshFill(_EditOp):
    idname = "mesh.fill"
    label = "补洞"
    description = "选中的边围成的洞用三角形补上（Alt+F）"

    def apply(self, ctx, session, mesh) -> bool:
        return T.fill_triangles(mesh) > 0


# ====================================================================== 撕开（V）
def _vertex_rip_edges(mesh, view, v: int, mouse) -> list[int]:
    """撕开一个点：鼠标那一侧的面和另一侧的面之间的边。"""
    table = mesh.edges()
    lf = mesh.loop_face()
    loops = np.nonzero(mesh.loop_vert == v)[0]
    faces = np.unique(lf[loops])
    if len(faces) < 2:
        return []
    sv = np.asarray(_screen(view, mesh.verts[v])[0])
    m = np.asarray(mouse, np.float64) - sv
    if np.hypot(*m) < 1.0:
        m = np.array([1.0, 0.0])
    centers = np.asarray(_screen(view, mesh.face_centers()[faces]))
    side = {int(f): float((c - sv) @ m) > 0 for f, c in zip(faces, centers)}
    incident = np.nonzero((table.edges[:, 0] == v) | (table.edges[:, 1] == v))[0]
    cuts = []
    for e in incident:
        around = set(lf[table.loop_edge == e].tolist())
        if len(around) == 2:
            f1, f2 = around
            if side[f1] != side[f2]:
                cuts.append(int(e))
    boundary = bool(np.any(table.face_count[incident] == 1))
    if (len(cuts) < 2 and not boundary) or not cuts:
        # 面都在一边：切和鼠标方向最垂直的边
        dirs = []
        for e in incident:
            if table.face_count[e] != 2:
                continue
            a, b = (int(x) for x in table.edges[e])
            w = b if a == v else a
            d = np.asarray(_screen(view, mesh.verts[w])[0]) - sv
            n = np.hypot(*d)
            dirs.append((abs(float(d @ m)) / max(n * np.hypot(*m), 1e-9), int(e)))
        dirs.sort()
        cuts = [e for _c, e in dirs[:1 if boundary else 2]]
    return cuts


@ops.register
class MeshRipMove(Operator):
    idname = "mesh.rip_move"
    label = "撕开"
    description = "把选中的点或边从面上撕开，鼠标那一侧跟着移动（V；Alt+V 撕开后补上缝）"
    icon = "split"
    use_fill = BoolProperty("补上缝", default=False)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None and bool(edit_session(ctx).mesh.vert_sel.any())

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        session = edit_session(ctx)
        mesh = session.mesh
        _engine(ctx).ctx_make_current()
        view = editor.view
        mouse = _pixel(editor, event)
        before = session.begin()
        table = mesh.edges()
        if mesh.edge_sel.any():
            cuts = np.nonzero(mesh.edge_sel & (table.face_count == 2))[0]
        else:
            cuts = []
            for v in np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]:
                cuts += _vertex_rip_edges(mesh, view, int(v), mouse)
            cuts = np.unique(np.asarray(cuts, np.int64))
        cut_pairs = [tuple(int(x) for x in table.edges[e]) for e in cuts]
        new_ids, source = T.edge_split(mesh, cuts) if len(cuts) else (np.zeros(0, np.int64), np.zeros(0, np.int64))
        if not len(new_ids):
            mesh.restore(before)
            session.update_overlay()
            self.report(ctx, "这里撕不开：选中的边两侧要都有面", "WARNING")
            return CANCELLED
        moving = self._choose_side(mesh, view, new_ids, source, mouse)
        if self.use_fill:
            self._fill(mesh, cut_pairs, new_ids, source, moving)
        session.push("撕开", before)
        _redraw(ctx)
        ops.call("transform.translate", ctx, event, merge_previous=1)
        return FINISHED

    def _choose_side(self, mesh, view, new_ids, source, mouse) -> dict:
        """每个被拆开的点留鼠标那一侧的那份选中，返回 原点号 → 跟着鼠标走的那份。"""
        lf = mesh.loop_face()
        centers = mesh.face_centers()
        mouse = np.asarray(mouse, np.float64)
        moving = {}
        mesh.vert_sel[:] = False
        for v in np.unique(source):
            copies = [int(v)] + [int(n) for n, s in zip(new_ids, source) if s == v]
            best, best_d = copies[0], np.inf
            for c in copies:
                faces = np.unique(lf[mesh.loop_vert == c])
                if not len(faces):
                    continue
                pos = np.asarray(_screen(view, centers[faces].mean(axis=0))[0])
                d = float(np.hypot(*(pos - mouse)))
                if d < best_d:
                    best, best_d = c, d
            moving[int(v)] = best
            mesh.vert_sel[best] = True
        mesh.flush_from_verts()
        return moving

    def _fill(self, mesh, cut_pairs, new_ids, source, moving) -> None:
        """撕开的缝用面补上（Alt+V）：每条撕开的边，两侧的两份连成一个四边形（端点没拆开时是三角形）。"""
        copies: dict[int, list[int]] = {}
        for n, s in zip(new_ids, source):
            copies.setdefault(int(s), [int(s)]).append(int(n))
        new = T.FaceList()
        lf = mesh.loop_face()
        nxt = mesh.loop_next()
        for a, b in cut_pairs:
            a1, b1 = moving.get(a, a), moving.get(b, b)
            a0 = next((c for c in copies.get(a, [a]) if c != a1), a1)
            b0 = next((c for c in copies.get(b, [b]) if c != b1), b1)
            ring = [a1, b1, b0, a0]
            # 走向：和移动那一侧挨着这条边的面相反
            forward = np.nonzero((mesh.loop_vert == a1) & (mesh.loop_vert[nxt] == b1))[0]
            if len(forward):
                ring = ring[::-1]
            ring = [x for k, x in enumerate(ring) if x != ring[k - 1]]
            if len(set(ring)) >= 3:
                face = int(lf[forward[0]]) if len(forward) else 0
                new.add(ring, np.zeros((len(ring), 2)), mesh.face_mat[face] if mesh.face_count else 0, False)
        if len(new):
            sel = mesh.vert_sel.copy()
            T.rebuild(mesh, np.ones(mesh.face_count, bool), new, compact=False)
            mesh.vert_sel[:] = sel
            mesh.flush_from_verts()


# ====================================================================== 滑移点（Shift+V）、滑移边（GG）
class _SlideBase(Operator):
    redo = True
    release_confirm = BoolProperty("松开确认", default=False, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.vert_sel.any())

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.session = edit_session(ctx)
        if self.editor is None:
            return self.execute(ctx)
        _engine(ctx).ctx_make_current()
        self.button = event.type if event is not None and event.type in (LEFTMOUSE, MIDDLEMOUSE, RIGHTMOUSE) else None
        self.before = self.session.begin()
        self.base = self.session.mesh.verts.copy()
        self.mouse0 = np.asarray(_pixel(self.editor, event), np.float64)
        self.numeric = ""
        if not self.setup(ctx):
            self.report(ctx, self.fail_text, "WARNING")
            return CANCELLED
        self._apply(ctx)
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            if not self.numeric:
                self.factor = self.mouse_factor(np.asarray(_pixel(self.editor, event), np.float64) - self.mouse0,
                                                bool(event.shift))
            self._apply(ctx)
            return RUNNING_MODAL
        if self.release_confirm and event.value == RELEASE and event.type == self.button:
            return self._confirm(ctx)
        if event.value != PRESS:
            return RUNNING_MODAL
        from .object_ops import DIGIT_KEYS

        if event.type in DIGIT_KEYS:
            char = DIGIT_KEYS[event.type]
            self.numeric = (self.numeric[1:] if self.numeric.startswith("-") else "-" + self.numeric) \
                if char == "-" else self.numeric + char
            try:
                self.factor = float(self.numeric)
            except ValueError:
                pass
            self._apply(ctx)
            return RUNNING_MODAL
        if event.type == "BACK_SPACE":
            self.numeric = self.numeric[:-1]
            return RUNNING_MODAL
        if event.type in (LEFTMOUSE, "RET", "NUMPAD_ENTER", "SPACE"):
            return self._confirm(ctx)
        if event.type in (RIGHTMOUSE, "ESC"):
            self.cancel(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def _apply(self, ctx) -> None:
        mesh = self.session.mesh
        mesh.verts = self.base.copy()
        self.move(mesh, float(self.factor))
        mesh.geometry_version += 1
        self.session.sync(topology=False)
        _hint(ctx, "%s  %.3f    左键确认 · 右键取消 · 输入数字" % (self.label, self.factor))
        ctx.app.request_frame()

    def _confirm(self, ctx) -> str:
        _hint(ctx, "")
        self.session.push(self.label, self.before, topology=False)
        _redraw(ctx)
        return FINISHED

    def cancel(self, ctx) -> None:
        mesh = self.session.mesh
        mesh.verts = self.base.copy()
        mesh.geometry_version += 1
        self.session.sync(topology=False)
        _hint(ctx, "")

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        self.session = session
        if not hasattr(self, "base"):
            self.base = session.mesh.verts.copy()
            self.mouse0 = np.zeros(2)
            if not self.setup(ctx):
                return CANCELLED
        before = session.begin()
        mesh = session.mesh
        mesh.verts = mesh.verts.copy()
        self.move(mesh, float(self.factor))
        mesh.geometry_version += 1
        session.push(self.label, before, topology=False)
        return FINISHED


@ops.register
class TransformVertSlide(_SlideBase):
    idname = "transform.vert_slide"
    label = "滑移顶点"
    description = "选中的点沿着连着它的边滑动（Shift+V）：往哪条边的方向拖就沿哪条边"
    factor = FloatProperty("系数", default=0.0, min=-1.0, max=1.0, precision=3, subtype="FACTOR")
    fail_text = "先选中要滑动的点"

    def setup(self, ctx) -> bool:
        mesh = self.session.mesh
        self.verts = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
        table = mesh.edges()
        self.neighbors = {}
        for v in self.verts:
            incident = table.edges[(table.edges[:, 0] == v) | (table.edges[:, 1] == v)]
            others = [int(b) if int(a) == v else int(a) for a, b in incident]
            if others:
                self.neighbors[int(v)] = others
        self.target = {v: n[0] for v, n in self.neighbors.items()}
        return bool(self.neighbors)

    def mouse_factor(self, delta, precise) -> float:
        view = self.editor.view
        best = 0.0
        if np.hypot(*delta) < 2.0:
            return 0.0
        for v, others in self.neighbors.items():
            sv = np.asarray(_screen(view, self.base[v])[0])
            dirs = [np.asarray(_screen(view, self.base[w])[0]) - sv for w in others]
            scores = [float(d @ delta) / max(np.hypot(*d) * np.hypot(*delta), 1e-9) for d in dirs]
            k = int(np.argmax(scores))
            self.target[v] = others[k]
            if v == next(iter(self.neighbors)):
                d = dirs[k]
                best = float(d @ delta) / max(float(d @ d), 1e-9)
        return float(np.clip(best * (0.1 if precise else 1.0), -1.0, 1.0))

    def move(self, mesh, factor: float) -> None:
        for v, w in self.target.items():
            mesh.verts[v] = self.base[v] + (self.base[w] - self.base[v]) * factor


@ops.register
class TransformEdgeSlide(_SlideBase):
    idname = "transform.edge_slide"
    label = "滑移边"
    description = "选中的一圈边沿着两侧的边滑动（连按两次 G）"
    factor = FloatProperty("系数", default=0.0, min=-1.0, max=1.0, precision=3, subtype="FACTOR")
    fail_text = "滑移边要选中连成一条线或一圈的边"

    def setup(self, ctx) -> bool:
        mesh = self.session.mesh
        table = mesh.edges()
        chosen = np.nonzero(mesh.edge_sel)[0]
        if not len(chosen):
            return False
        degree = np.bincount(table.edges[chosen].reshape(-1), minlength=mesh.vert_count)
        if degree.max() > 2:
            return False
        nxt = mesh.loop_next()
        prv = mesh.loop_prev()
        loop_edge = table.loop_edge
        chosen_set = set(chosen.tolist())
        # 每个点两侧的滑轨：面里沿选中边走的方向在左边（面的走向和边同向）为 +，否则为 −
        self.rails: dict[int, dict[int, int]] = {}
        for e in chosen:
            for loop in np.nonzero(loop_edge == e)[0]:
                a, b = int(mesh.loop_vert[loop]), int(mesh.loop_vert[nxt[loop]])
                side = 1 if self._forward(a, b) else -1
                # 面里 a 之前的边、b 之后的边就是这两个点在这一侧的滑轨
                rail_a = int(loop_edge[prv[loop]])
                rail_b = int(loop_edge[nxt[loop]])
                for v, rail in ((a, rail_a), (b, rail_b)):
                    if rail in chosen_set:
                        continue
                    ra, rb = (int(x) for x in table.edges[rail])
                    self.rails.setdefault(v, {}).setdefault(side, rb if ra == v else ra)
        return bool(self.rails)

    def _forward(self, a: int, b: int) -> bool:
        """选中的边按统一的方向排：沿着每条线（或每一圈）从头走到尾，a → b 是不是顺着走。"""
        if not hasattr(self, "_seq"):
            mesh = self.session.mesh
            table = mesh.edges()
            adj: dict[int, list[int]] = {}
            for x, y in table.edges[np.nonzero(mesh.edge_sel)[0]]:
                adj.setdefault(int(x), []).append(int(y))
                adj.setdefault(int(y), []).append(int(x))
            seq: dict[int, int] = {}
            # 先从线的端点（只连一条选中边的点）出发，剩下的是闭合的圈
            for start in sorted(adj, key=lambda v: (len(adj[v]), v)):
                if start in seq:
                    continue
                prev, cur = None, start
                while cur is not None:
                    seq[cur] = len(seq)
                    following = [n for n in adj[cur] if n != prev and n not in seq]
                    prev, cur = cur, (following[0] if following else None)
            self._seq = seq
        da, db = self._seq.get(a, 0), self._seq.get(b, 0)
        if abs(da - db) == 1:
            return db > da
        return db < da                                # 一圈的首尾相接那条边

    def mouse_factor(self, delta, precise) -> float:
        view = self.editor.view
        if np.hypot(*delta) < 2.0:
            return 0.0
        v = next(iter(self.rails))
        sv = np.asarray(_screen(view, self.base[v])[0])
        best, score = 0.0, -np.inf
        for side, w in self.rails[v].items():
            d = np.asarray(_screen(view, self.base[w])[0]) - sv
            along = float(d @ delta) / max(float(d @ d), 1e-9)
            if along > score:
                best, score = along * side, along
        return float(np.clip(best * (0.1 if precise else 1.0), -1.0, 1.0))

    def move(self, mesh, factor: float) -> None:
        side = 1 if factor >= 0 else -1
        t = abs(factor)
        for v, rails in self.rails.items():
            w = rails.get(side)
            if w is not None:
                mesh.verts[v] = self.base[v] + (self.base[w] - self.base[v]) * t


@ops.register
class TransformToSphere(_SlideBase):
    idname = "transform.tosphere"
    label = "球形化"
    description = "选中的点往同一个球面上靠（Shift+Alt+S）：往右拖越圆"
    factor = FloatProperty("系数", default=0.0, min=0.0, max=1.0, precision=3, subtype="FACTOR")
    fail_text = "先选中要球形化的点"

    def setup(self, ctx) -> bool:
        mesh = self.session.mesh
        self.verts = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
        if len(self.verts) < 2:
            return False
        pts = self.base[self.verts]
        self.center = pts.mean(axis=0)
        self.radius = float(np.linalg.norm(pts - self.center, axis=1).mean())
        return self.radius > 1e-12

    def mouse_factor(self, delta, precise) -> float:
        return float(np.clip(delta[0] / 300.0 * (0.1 if precise else 1.0), 0.0, 1.0))

    def move(self, mesh, factor: float) -> None:
        p0 = self.base[self.verts]
        d = p0 - self.center
        n = np.linalg.norm(d, axis=1, keepdims=True)
        on_sphere = self.center + d / np.maximum(n, 1e-30) * self.radius
        mesh.verts[self.verts] = p0 + (on_sphere - p0) * factor


@ops.register
class TransformShear(_SlideBase):
    idname = "transform.shear"
    label = "切变"
    description = "选中的点按屏幕上的高低往左右推（Ctrl+Shift+Alt+S）"
    factor = FloatProperty("系数", default=0.0, min=-10.0, max=10.0, precision=3)
    fail_text = "先选中要切变的点"

    def setup(self, ctx) -> bool:
        mesh = self.session.mesh
        self.verts = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
        if not len(self.verts):
            return False
        self.center = self.base[self.verts].mean(axis=0)
        editor = getattr(self, "editor", None) or view3d(ctx)
        if editor is not None and editor.view is not None:
            right, up, _back = editor.view.camera.basis()
            self.right, self.up = np.asarray(right, np.float64), np.asarray(up, np.float64)
        elif not hasattr(self, "right"):
            self.right, self.up = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        return True

    def mouse_factor(self, delta, precise) -> float:
        return float(delta[0] / 200.0 * (0.1 if precise else 1.0))

    def move(self, mesh, factor: float) -> None:
        p0 = self.base[self.verts]
        height = (p0 - self.center) @ self.up
        mesh.verts[self.verts] = p0 + np.outer(height * factor, self.right)


# ====================================================================== 只挤出边、点；挤出到鼠标
@ops.register
class MeshExtrudeEdgesMove(Operator):
    idname = "mesh.extrude_edges_move"
    label = "只挤出边"
    description = "把选中的边挤出成面（不管选中的面），接着移动"

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.edge_sel.any())

    def invoke(self, ctx, event) -> str:
        session = edit_session(ctx)
        before = session.begin()
        new = T._extrude_edges(session.mesh, np.nonzero(session.mesh.edge_sel)[0])
        if not len(new):
            return CANCELLED
        session.push(self.label, before)
        if view3d(ctx) is not None:
            ops.call("transform.translate", ctx, event, merge_previous=1)
        return FINISHED


@ops.register
class MeshExtrudeVerticesMove(Operator):
    idname = "mesh.extrude_vertices_move"
    label = "只挤出点"
    description = "从选中的点各拉出一条边，接着移动"

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.vert_sel.any())

    def invoke(self, ctx, event) -> str:
        session = edit_session(ctx)
        before = session.begin()
        mesh = session.mesh
        new = T._extrude_verts(mesh, np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0])
        if not len(new):
            return CANCELLED
        session.push(self.label, before)
        if view3d(ctx) is not None:
            ops.call("transform.translate", ctx, event, merge_previous=1)
        return FINISHED


@ops.register
class MeshDupliExtrudeCursor(Operator):
    idname = "mesh.dupli_extrude_cursor"
    label = "挤出到鼠标"
    description = "选中的部分挤出到鼠标点的位置（Ctrl+右键）；什么都没选时在那里加一个点"
    searchable = False

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        session = edit_session(ctx)
        mesh = session.mesh
        view = editor.view
        x, y = _pixel(editor, event)
        sel = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
        if len(sel):
            center = mesh.verts[sel].mean(axis=0)
        else:
            center = to_internal(np.asarray(ctx.app.project.cursor_location, np.float64))
        origin, direction = view.camera.ray(x, y, view.width, view.height)
        origin = np.asarray(origin, np.float64)
        direction = np.asarray(direction, np.float64)
        normal = -np.asarray(view.camera.forward, np.float64)
        den = float(direction @ normal)
        if abs(den) < 1e-12:
            return CANCELLED
        target = origin + direction * (float((center - origin) @ normal) / den)
        before = session.begin()
        if len(sel):
            new = T.extrude_region(mesh)
            if not len(new):
                return CANCELLED
            mesh.verts[new] += target - center
        else:
            T.rebuild(mesh, np.ones(mesh.face_count, bool), None, target[None], compact=False)
            T.select_new(mesh, verts=[mesh.vert_count - 1])
        session.push(self.label, before)
        _redraw(ctx)
        return FINISHED


# ====================================================================== 选择：并排边、镜像
@ops.register
class MeshEdgeringSelect(Operator):
    idname = "mesh.edgering_select"
    label = "选择并排边"
    description = "Ctrl+Alt+点击一条边：和它并排的一圈边都选上（Shift 加选）"
    searchable = False
    toggle = BoolProperty("加选", default=False)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        return ops.call("mesh.loop_select", ctx, event, ring=True, toggle=bool(self.toggle))


@ops.register
class MeshSelectMirror(_EditOp):
    idname = "mesh.select_mirror"
    label = "镜像选择"
    description = "选中物体另一半对称位置上的点（沿物体自己的轴）"
    redo = True
    topology = False
    axis = EnumProperty("轴", items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")], default="X")
    extend = BoolProperty("保留原来的选择", default=False)

    def execute(self, ctx) -> str:
        from ..doc.objects import display_to_internal_matrix

        session = edit_session(ctx)
        mesh = session.mesh
        before = session.begin()
        m = display_to_internal_matrix(session.obj.transform.matrix())
        inv = np.linalg.inv(m)
        local = mesh.verts @ inv[:3, :3].T + inv[:3, 3]
        k = "XYZ".index(self.axis)
        # 显示坐标的 Z 是内部坐标的 Y（取反方向不影响镜像）
        k_internal = {0: 0, 1: 2, 2: 1}[k]
        size = float(np.ptp(local, axis=0).max()) if mesh.vert_count else 1.0
        tol = max(size * 1e-4, 1e-6)
        keys = np.round(local / tol).astype(np.int64)
        lookup = {tuple(key): i for i, key in enumerate(keys.tolist())}
        chosen = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
        found = []
        for v in chosen:
            p = local[v].copy()
            p[k_internal] = -p[k_internal]
            base = np.round(p / tol).astype(np.int64)
            hit = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        hit = lookup.get((int(base[0] + dx), int(base[1] + dy), int(base[2] + dz)))
                        if hit is not None:
                            break
                    if hit is not None:
                        break
                if hit is not None:
                    break
            if hit is not None:
                found.append(hit)
        if not found:
            self.report(ctx, "没找到对称位置上的点", "WARNING")
            return CANCELLED
        if not self.extend:
            mesh.vert_sel[:] = False
        mesh.vert_sel[np.asarray(found, np.int64)] = True
        mesh.flush_from_verts()
        if "VERT" not in mesh.select_mode:
            mesh.flush()
        session.push_selection(self.label, before)
        _redraw(ctx)
        return FINISHED


# ====================================================================== 工具栏里拖动类工具的入口
TOOL_ACTIONS = {
    "EXTRUDE": ("view3d.edit_mesh_extrude_move_normal", {}),
    "INSET": ("mesh.inset", {}),
    "BEVEL": ("mesh.bevel", {}),
    "SHRINK_FATTEN": ("mesh.shrink_fatten", {}),
    "SMOOTH": ("mesh.vertices_smooth", {"interactive": True}),
}


@ops.register
class MeshToolDrag(Operator):
    idname = "mesh.tool_drag"
    label = "工具拖动"
    description = "在选中的部分上按住拖动就用当前工具；在别处拖动是框选"
    searchable = False
    action = EnumProperty("动作", items=[(k, k, "") for k in TOOL_ACTIONS], default="EXTRUDE")

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        session = edit_session(ctx)
        mesh = session.mesh
        engine = _engine(ctx)
        engine.ctx_make_current()
        x, y = _pixel(editor, event)
        element = S.pick(engine, editor.view, mesh, x, y, _xray(editor))
        on_selection = False
        if element is not None:
            kind, index = element
            on_selection = bool({"VERT": mesh.vert_sel, "EDGE": mesh.edge_sel, "FACE": mesh.face_sel}[kind][index])
        if not on_selection:
            face = S.face_under(editor.view, mesh, x, y)
            on_selection = face is not None and bool(mesh.face_sel[face])
        if not on_selection or not mesh.vert_sel.any():
            mode = "SUB" if event is not None and event.ctrl else ("ADD" if event is not None and event.shift else "SET")
            ops.call("view3d.select_box", ctx, event, mode=mode)
            return FINISHED
        idname, props = TOOL_ACTIONS[self.action]
        ops.call(idname, ctx, event, release_confirm=True, **props)
        return FINISHED
