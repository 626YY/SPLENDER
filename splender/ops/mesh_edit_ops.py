"""编辑模式的操作（照 Blender 的网格编辑）：选择模式 1/2/3、全选、相连、循环边、并排边、最短路径、扩大缩小选区、
相似、隐藏；挤出 E、内插 I、倒角 Ctrl+B、环切 Ctrl+R、刀切 K、合并 M、填充 F、复制 Shift+D、拆分 Y、分离 P、
删除 X、融并 Ctrl+X、细分、三角化 Ctrl+T、三角转四边 Alt+J、法线 Shift+N、平滑、UV 展开 U。

每个操作：先拍快照（session.begin），改 EditMesh，再 session.push 记一步撤销并刷新显示。
"""
from __future__ import annotations

import math

import numpy as np

from ..core import ops
from ..core.keymap import LEFTMOUSE, MIDDLEMOUSE, MOUSEMOVE, PRESS, RELEASE, RIGHTMOUSE, WHEEL, WHEELDOWN, WHEELUP
from ..core.ops import CANCELLED, FINISHED, PASS_THROUGH, RUNNING_MODAL, Operator
from ..core.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from ..doc.objects import to_display, to_internal
from ..mesh import loopcut as LC
from ..mesh import select as S
from ..mesh import topology as T
from ..mesh.bevel import bevel_edges, bevel_vertices
from .view3d_ops import view3d


def _engine(ctx):
    host = getattr(ctx.app, "host", None)
    return host.engine if host is not None else None


def edit_session(ctx):
    engine = _engine(ctx)
    return engine.edit if engine is not None else None


def edit_mode(ctx) -> bool:
    return ctx.app.tool_settings.mode == "EDIT" and edit_session(ctx) is not None


def _xray(editor) -> bool:
    shading = getattr(editor, "shading", None)
    return bool(getattr(shading, "show_xray", False)) or getattr(shading, "mode", "") == "WIREFRAME"


def _hint(ctx, text: str) -> None:
    if getattr(ctx, "wm", None) is not None:
        ctx.wm.set_hint(text)


def _redraw(ctx) -> None:
    ctx.app.request_frame()
    ctx.app.notify("edit")


class _EditOp(Operator):
    """编辑模式里、对选中部分做的一步操作。子类实现 apply(ctx, session, mesh)，返回 False 表示没做成。"""

    redo = True
    topology = True
    needs_selection = True

    @classmethod
    def poll(cls, ctx) -> bool:
        if not edit_mode(ctx):
            return False
        if not cls.needs_selection:
            return True
        mesh = edit_session(ctx).mesh
        return bool(mesh.vert_sel.any() or mesh.face_sel.any())

    def apply(self, ctx, session, mesh) -> bool:
        raise NotImplementedError

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        before = session.begin()
        _engine(ctx).ctx_make_current()
        try:
            ok = self.apply(ctx, session, session.mesh)
        except Exception:
            session.mesh.restore(before)
            raise
        if ok is False:
            session.mesh.restore(before)
            session.update_overlay()
            return CANCELLED
        session.push(self.label, before, topology=self.topology)
        _redraw(ctx)
        return FINISHED


# ====================================================================== 选择
@ops.register
class MeshSelectMode(_EditOp):
    idname = "mesh.select_mode"
    label = "选择模式"
    description = "选点、选边还是选面（Shift 同时开几种，Ctrl 按现在的选中扩展到新模式）"
    redo = False
    needs_selection = False
    type = EnumProperty("模式", items=[("VERT", "点", "", "select.vert"), ("EDGE", "边", "", "select.edge"),
                                     ("FACE", "面", "", "select.face")], default="VERT")
    use_extend = BoolProperty("叠加", default=False)
    use_expand = BoolProperty("扩展", default=False)

    def invoke(self, ctx, event) -> str:
        if event is None:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication

            mods = QApplication.keyboardModifiers()
            self.use_extend = bool(self.use_extend or mods & Qt.ShiftModifier)
            self.use_expand = bool(self.use_expand or mods & Qt.ControlModifier)
        return self.execute(ctx)

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        mesh = session.mesh
        before = session.begin()
        modes = set(mesh.select_mode)
        if self.use_extend:
            modes ^= {self.type}
            if not modes:
                modes = {self.type}
        else:
            modes = {self.type}
        if self.use_expand and self.type in ("EDGE", "FACE") and "VERT" in mesh.select_mode:
            # 从点模式扩展：沾到选中点的边、面都选上
            table = mesh.edges()
            if self.type == "EDGE":
                mesh.edge_sel = mesh.vert_sel[table.edges[:, 0]] | mesh.vert_sel[table.edges[:, 1]]
                mesh.select_mode = modes
                mesh.flush_from_edges()
            else:
                mesh.face_sel = np.logical_or.reduceat(mesh.vert_sel[mesh.loop_vert], mesh.face_start[:-1]) \
                    if mesh.face_count else np.zeros(0, bool)
                mesh.select_mode = modes
                mesh.flush_from_faces()
        else:
            mesh.set_select_mode(modes)
        session.push_selection("选择模式", before)
        _redraw(ctx)
        ctx.app.notify("mode")
        return FINISHED


@ops.register
class MeshSelectAll(_EditOp):
    idname = "mesh.select_all"
    label = "全选"
    redo = False
    needs_selection = False
    action = EnumProperty("动作", items=[("TOGGLE", "切换", ""), ("SELECT", "全选", ""), ("DESELECT", "全不选", ""),
                                       ("INVERT", "反选", "")], default="TOGGLE")

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        before = session.begin()
        session.mesh.select_all(self.action)
        session.push_selection({"SELECT": "全选", "DESELECT": "全不选", "INVERT": "反选"}.get(self.action, "全选"),
                               before)
        _redraw(ctx)
        return FINISHED


def edit_click_select(ctx, editor, event, toggle=False, extend=False, deselect=False, deselect_all=True) -> str:
    """编辑模式里的点选（view3d.select 在编辑模式时走这里）。"""
    session = edit_session(ctx)
    mesh = session.mesh
    engine = _engine(ctx)
    engine.ctx_make_current()
    x, y = editor.pixel(event)
    element = S.pick(engine, editor.view, mesh, x, y, _xray(editor))
    before = session.begin()
    if element is None:
        if not (deselect_all and not (toggle or extend or deselect)):
            return CANCELLED
        mesh.select_all("DESELECT")
    else:
        S.select_element(mesh, element, toggle=toggle, extend=extend, deselect=deselect)
    session.push_selection("选择", before)
    _redraw(ctx)
    return FINISHED


def edit_region_select(ctx, editor, test, mode: str, label: str) -> None:
    session = edit_session(ctx)
    engine = _engine(ctx)
    engine.ctx_make_current()
    before = session.begin()
    found = S.elements_in(engine, editor.view, session.mesh, test, _xray(editor))
    S.apply_region(session.mesh, found, mode)
    session.push_selection(label, before)
    _redraw(ctx)


@ops.register
class MeshSelectLinked(_EditOp):
    idname = "mesh.select_linked"
    label = "选择相连"
    description = "和选中部分连着的整块都选上"
    redo = False

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        mesh = session.mesh
        before = session.begin()
        verts = S.linked(mesh, np.nonzero(mesh.vert_sel)[0])
        mesh.vert_sel[verts] = True
        mesh.flush_from_verts()
        if "VERT" not in mesh.select_mode:
            mesh.flush()
        session.push_selection("选择相连", before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshSelectLinkedPick(Operator):
    idname = "mesh.select_linked_pick"
    label = "选择鼠标下相连的部分"
    description = "鼠标指着的那一整块：L 选上，Shift+L 取消"
    searchable = False
    deselect = BoolProperty("取消选择", default=False)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        session = edit_session(ctx)
        mesh = session.mesh
        engine = _engine(ctx)
        engine.ctx_make_current()
        x, y = editor.cursor_pixel() if event is None or event.source is not editor.widget else editor.pixel(event)
        element = S.pick(engine, editor.view, mesh, x, y, _xray(editor), radius_px=60.0)
        if element is None:
            face = S.face_under(editor.view, mesh, x, y)
            element = ("FACE", face) if face is not None else None
        if element is None:
            return CANCELLED
        kind, index = element
        if kind == "VERT":
            seeds = [index]
        elif kind == "EDGE":
            seeds = list(mesh.edges().edges[index])
        else:
            seeds = list(mesh.face_verts(index))
        before = session.begin()
        verts = S.linked(mesh, np.asarray(seeds, np.int64))
        mesh.vert_sel[verts] = not self.deselect
        mesh.flush_from_verts()
        session.push_selection("选择相连", before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshLoopSelect(Operator):
    idname = "mesh.loop_select"
    label = "选择循环边"
    description = "Alt+点击一条边：沿着它一整圈都选上（Shift 加选）"
    searchable = False
    toggle = BoolProperty("加选", default=False)
    ring = BoolProperty("并排边", default=False, description="选的是一圈并排的边（Ctrl+Alt+点击）")

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        session = edit_session(ctx)
        mesh = session.mesh
        engine = _engine(ctx)
        engine.ctx_make_current()
        x, y = editor.pixel(event)
        modes = mesh.select_mode
        mesh.select_mode = {"EDGE"}
        found = S._nearest_edge(engine, editor.view, mesh, x, y, _xray(editor), 40.0)
        mesh.select_mode = modes
        if found is None:
            return CANCELLED
        edge = found[0]
        before = session.begin()
        if self.ring:
            edges, _closed = LC.edge_ring(mesh, edge)
        else:
            edges = LC.edge_loop(mesh, edge)
        if not self.toggle:
            mesh.select_all("DESELECT")
        table = mesh.edges()
        if "FACE" in mesh.select_mode and "VERT" not in mesh.select_mode and "EDGE" not in mesh.select_mode:
            # 面模式：循环面（两条相邻循环边之间的面）
            lf = mesh.loop_face()
            chosen = np.isin(table.loop_edge, edges)
            faces = np.unique(lf[chosen])
            mesh.face_sel[faces] = True
            mesh.flush_from_faces()
        else:
            mesh.edge_sel[edges] = True
            mesh.flush_from_edges()
        mesh.active = ("EDGE", int(edge))
        session.push_selection("选择循环边", before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshShortestPathPick(Operator):
    idname = "mesh.shortest_path_pick"
    label = "选择最短路径"
    description = "Ctrl+点击一个点：从当前点到它沿边最短的路径都选上"
    searchable = False

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        session = edit_session(ctx)
        mesh = session.mesh
        engine = _engine(ctx)
        engine.ctx_make_current()
        x, y = editor.pixel(event)
        modes = mesh.select_mode
        mesh.select_mode = {"VERT"}
        target = S._nearest_vert(engine, editor.view, mesh, x, y, _xray(editor), 40.0)
        mesh.select_mode = modes
        if target is None:
            return CANCELLED
        start = mesh.active[1] if mesh.active is not None and mesh.active[0] == "VERT" else None
        if start is None:
            chosen = np.nonzero(mesh.vert_sel)[0]
            if not len(chosen):
                return CANCELLED
            start = int(chosen[0])
        before = session.begin()
        path = LC.shortest_path(mesh, int(start), int(target[0]))
        mesh.vert_sel[path] = True
        mesh.flush_from_verts()
        mesh.active = ("VERT", int(target[0]))
        session.push_selection("选择最短路径", before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshSelectMore(_EditOp):
    idname = "mesh.select_more"
    label = "扩大选区"
    redo = False

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        before = session.begin()
        S.grow(session.mesh)
        session.push_selection(self.label, before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshSelectLess(_EditOp):
    idname = "mesh.select_less"
    label = "缩小选区"
    redo = False

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        before = session.begin()
        S.shrink(session.mesh)
        session.push_selection(self.label, before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshSelectSimilar(_EditOp):
    idname = "mesh.select_similar"
    label = "选择相似"
    description = "和选中的面（边、点）差不多的都选上"
    redo = True
    type = EnumProperty("依据", items=[("NORMAL", "朝向", "法线方向差不多"), ("AREA", "面积", ""),
                                     ("MATERIAL", "材质", ""), ("SIDES", "边数", "面的边数一样"),
                                     ("LENGTH", "边长", "边的长度差不多"), ("FACES", "连着的面数", "点连着的面数一样")],
                        default="NORMAL")
    threshold = FloatProperty("容差", default=0.1, min=0.0, max=1.0, precision=3, subtype="FACTOR")

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        mesh = session.mesh
        before = session.begin()
        kind = self.type
        th = float(self.threshold)
        if kind in ("NORMAL", "AREA", "MATERIAL", "SIDES") and mesh.face_sel.any():
            faces = np.nonzero(mesh.face_sel)[0]
            if kind == "NORMAL":
                fn = mesh.face_normals()
                ref = fn[faces]
                hit = (fn @ ref.T).max(axis=1) >= 1.0 - th * 2.0
            elif kind == "AREA":
                area = mesh.face_areas()
                ref = area[faces]
                hit = np.min(np.abs(area[:, None] - ref[None, :]) / np.maximum(ref[None, :], 1e-12), axis=1) <= th
            elif kind == "MATERIAL":
                hit = np.isin(mesh.face_mat, mesh.face_mat[faces])
            else:
                sizes = mesh.face_sizes()
                hit = np.isin(sizes, sizes[faces])
            mesh.face_sel = hit & ~mesh.face_hide
            mesh.flush_from_faces()
        elif kind == "LENGTH" and mesh.edge_sel.any():
            table = mesh.edges()
            length = np.linalg.norm(mesh.verts[table.edges[:, 0]] - mesh.verts[table.edges[:, 1]], axis=1)
            ref = length[mesh.edge_sel]
            hit = np.min(np.abs(length[:, None] - ref[None, :]) / np.maximum(ref[None, :], 1e-12), axis=1) <= th
            mesh.edge_sel = hit
            mesh.flush_from_edges()
        elif kind == "FACES" and mesh.vert_sel.any():
            counts = np.bincount(mesh.loop_vert, minlength=mesh.vert_count)
            hit = np.isin(counts, counts[mesh.vert_sel])
            mesh.vert_sel = hit & ~mesh.vert_hide
            mesh.flush_from_verts()
        else:
            return CANCELLED
        session.push_selection(self.label, before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshSelectNonManifold(_EditOp):
    idname = "mesh.select_non_manifold"
    label = "选择非流形"
    description = "选中开口边、三个以上面共用的边、散边"
    redo = False
    needs_selection = False

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        mesh = session.mesh
        before = session.begin()
        table = mesh.edges()
        bad = (table.face_count != 2)
        mesh.set_select_mode(mesh.select_mode | {"VERT"} if "FACE" in mesh.select_mode else mesh.select_mode)
        mesh.edge_sel = bad
        mesh.flush_from_edges()
        session.push_selection(self.label, before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshSelectRandom(_EditOp):
    idname = "mesh.select_random"
    label = "随机选择"
    redo = True
    needs_selection = False
    ratio = FloatProperty("比例", default=0.5, min=0.0, max=1.0, subtype="FACTOR")
    seed = IntProperty("随机种子", default=0, min=0, max=100000)
    action = EnumProperty("动作", items=[("SELECT", "选择", ""), ("DESELECT", "取消选择", "")], default="SELECT")

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        mesh = session.mesh
        before = session.begin()
        rng = np.random.default_rng(int(self.seed))
        value = self.action == "SELECT"
        if "VERT" in mesh.select_mode:
            pick = rng.random(mesh.vert_count) < float(self.ratio)
            mesh.vert_sel[pick & ~mesh.vert_hide] = value
            mesh.flush_from_verts()
        elif "EDGE" in mesh.select_mode:
            pick = rng.random(len(mesh.edge_sel)) < float(self.ratio)
            mesh.edge_sel[pick] = value
            mesh.flush_from_edges()
        else:
            pick = rng.random(mesh.face_count) < float(self.ratio)
            mesh.face_sel[pick & ~mesh.face_hide] = value
            mesh.flush_from_faces()
        session.push_selection(self.label, before)
        _redraw(ctx)
        return FINISHED


# ====================================================================== 隐藏
@ops.register
class MeshHide(_EditOp):
    idname = "mesh.hide"
    label = "隐藏"
    description = "藏起选中的部分（H；Shift+H 藏起没选中的）"
    unselected = BoolProperty("隐藏没选中的", default=False)

    def apply(self, ctx, session, mesh) -> bool:
        T.hide(mesh, bool(self.unselected))
        return True


@ops.register
class MeshReveal(_EditOp):
    idname = "mesh.reveal"
    label = "显示隐藏的部分"
    description = "藏起来的都显示出来（Alt+H）"
    needs_selection = False
    select = BoolProperty("选中它们", default=True)

    def apply(self, ctx, session, mesh) -> bool:
        if not (mesh.vert_hide.any() or mesh.face_hide.any()):
            return False
        T.reveal(mesh, bool(self.select))
        return True


# ====================================================================== 删除、融并
@ops.register
class MeshDelete(_EditOp):
    idname = "mesh.delete"
    label = "删除"
    type = EnumProperty("删除", items=[("VERT", "点", ""), ("EDGE", "边", ""), ("FACE", "面", ""),
                                     ("EDGE_FACE", "只删边和面", ""), ("ONLY_FACE", "只删面", "")], default="VERT")

    def apply(self, ctx, session, mesh) -> bool:
        T.delete(mesh, self.type)
        return True


@ops.register
class MeshDissolveVerts(_EditOp):
    idname = "mesh.dissolve_verts"
    label = "融并点"
    description = "去掉选中的点，周围的面合成一个"

    def apply(self, ctx, session, mesh) -> bool:
        T.dissolve_verts(mesh)
        return True


@ops.register
class MeshDissolveEdges(_EditOp):
    idname = "mesh.dissolve_edges"
    label = "融并边"
    description = "去掉选中的边，两边的面合成一个"

    def apply(self, ctx, session, mesh) -> bool:
        return T.dissolve_edges(mesh) > 0


@ops.register
class MeshDissolveFaces(_EditOp):
    idname = "mesh.dissolve_faces"
    label = "融并面"
    description = "连在一起的选中面合成一个面"

    def apply(self, ctx, session, mesh) -> bool:
        return T.dissolve_faces(mesh) > 0


@ops.register
class MeshDissolveMode(_EditOp):
    idname = "mesh.dissolve_mode"
    label = "融并"
    description = "按选择模式融并点、边或面"

    def apply(self, ctx, session, mesh) -> bool:
        if "VERT" in mesh.select_mode:
            T.dissolve_verts(mesh)
            return True
        if "EDGE" in mesh.select_mode:
            return T.dissolve_edges(mesh) > 0
        return T.dissolve_faces(mesh) > 0


# ====================================================================== 合并、填充、复制、拆分、分离
@ops.register
class MeshMerge(_EditOp):
    idname = "mesh.merge"
    label = "合并"
    type = EnumProperty("合并到", items=[("CENTER", "中心", ""), ("CURSOR", "3D 游标", ""), ("COLLAPSE", "各自塌陷", ""),
                                       ("FIRST", "第一个", ""), ("LAST", "最后一个", "")], default="CENTER")

    def apply(self, ctx, session, mesh) -> bool:
        cursor = to_internal(np.asarray(ctx.app.project.cursor_location, np.float64))
        removed = T.merge(mesh, self.type, cursor)
        self.report(ctx, "合并掉 %d 个点" % removed)
        return removed > 0 or self.type == "COLLAPSE"


@ops.register
class MeshRemoveDoubles(_EditOp):
    idname = "mesh.remove_doubles"
    label = "按距离合并"
    description = "离得很近的点合成一个（去重点）"
    threshold = FloatProperty("合并距离", default=0.0001, min=0.0, max=10.0, precision=5, unit="m")
    use_unselected = BoolProperty("连同没选中的", default=False)

    def apply(self, ctx, session, mesh) -> bool:
        removed = T.merge_by_distance(mesh, float(self.threshold), only_selected=not self.use_unselected)
        self.report(ctx, "合并掉 %d 个点" % removed)
        return True


@ops.register
class MeshEdgeFaceAdd(_EditOp):
    idname = "mesh.edge_face_add"
    label = "填充"
    description = "两个点连成边，边界上的一圈点补成面（F）"

    def apply(self, ctx, session, mesh) -> bool:
        return bool(T.fill(mesh))


@ops.register
class MeshSplit(_EditOp):
    idname = "mesh.split"
    label = "拆分"
    description = "选中的面和其余部分断开（Y）"

    def apply(self, ctx, session, mesh) -> bool:
        if not mesh.face_sel.any():
            return False
        T.split(mesh)
        return True


@ops.register
class MeshSeparate(_EditOp):
    idname = "mesh.separate"
    label = "分离"
    description = "把一部分拿出来做成新物体（P）"
    type = EnumProperty("分离", items=[("SELECTED", "选中项", ""), ("MATERIAL", "按材质", ""),
                                     ("LOOSE", "按松散块", "")], default="SELECTED")

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx)

    def execute(self, ctx) -> str:
        from ..mesh.editmesh import to_mesh_data
        from ..doc.project import MeshObject
        from .object_ops import add_objects_with_undo, unique_name

        session = edit_session(ctx)
        mesh = session.mesh
        before = session.begin()
        parts = []
        if self.type == "SELECTED":
            other = T.separate_selection(mesh)
            if other is not None:
                parts.append(other)
        elif self.type == "MATERIAL":
            for mat in np.unique(mesh.face_mat)[1:]:
                mesh.face_sel = (mesh.face_mat == mat) & ~mesh.face_hide
                mesh.flush_from_faces()
                other = T.separate_selection(mesh)
                if other is not None:
                    parts.append(other)
        else:
            from ..mesh.topology import _face_islands

            islands = _face_islands(mesh, np.arange(mesh.face_count))
            for island in sorted(islands, key=len)[:-1]:
                mesh.face_sel[:] = False
                mesh.face_sel[island] = True
                mesh.flush_from_faces()
                other = T.separate_selection(mesh)
                if other is not None:
                    parts.append(other)
        if not parts:
            mesh.restore(before)
            return CANCELLED
        session.push("分离", before)
        obj = session.obj
        project = ctx.app.project
        new_objects = []
        for part in parts:
            data, _corner = to_mesh_data(part, obj.data)
            data.name = unique_name(project, obj.name)
            new = MeshObject(data.name, data, list(obj.material_sets))
            new.geometry_dirty = True
            new_objects.append(new)
        add_objects_with_undo(ctx, new_objects, [], "分离", select=False)
        self.report(ctx, "分离出 %d 个物体" % len(new_objects))
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshDuplicateMove(Operator):
    idname = "mesh.duplicate_move"
    label = "复制"
    description = "复制选中的部分并跟着鼠标移动（Shift+D）"
    icon = "duplicate"

    release_confirm = BoolProperty("松开确认", default=False, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.vert_sel.any())

    def invoke(self, ctx, event) -> str:
        session = edit_session(ctx)
        before = session.begin()
        new = T.duplicate(session.mesh)
        if not len(new):
            return CANCELLED
        session.push("复制", before)
        if view3d(ctx) is not None:
            ops.call("transform.translate", ctx, event, merge_previous=1, release_confirm=bool(self.release_confirm))
        return FINISHED

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        before = session.begin()
        T.duplicate(session.mesh)
        session.push("复制", before)
        return FINISHED


# ====================================================================== 挤出
@ops.register
class MeshExtrudeMoveNormal(Operator):
    idname = "view3d.edit_mesh_extrude_move_normal"
    label = "挤出"
    description = "挤出选中的面（边、点）并沿法线拉出去（E）"
    icon = "tool.extrude"
    release_confirm = BoolProperty("松开确认", default=False, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.vert_sel.any())

    def invoke(self, ctx, event) -> str:
        session = edit_session(ctx)
        mesh = session.mesh
        before = session.begin()
        had_faces = bool(mesh.face_sel.any())
        normal = _selection_normal(mesh) if had_faces else None
        new = T.extrude_region(mesh)
        if not len(new):
            return CANCELLED
        session.push("挤出", before)
        if view3d(ctx) is None:
            return FINISHED
        confirm = bool(self.release_confirm)
        if had_faces and normal is not None:
            ops.call("transform.translate", ctx, event, merge_previous=1, constraint="Z", orient_type="NORMAL",
                     release_confirm=confirm)
        else:
            ops.call("transform.translate", ctx, event, merge_previous=1, release_confirm=confirm)
        return FINISHED

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        before = session.begin()
        T.extrude_region(session.mesh)
        session.push("挤出", before)
        return FINISHED


@ops.register
class MeshExtrudeFacesMove(Operator):
    idname = "mesh.extrude_faces_move"
    label = "挤出各个面"
    description = "每个选中的面各自沿自己的法线挤出"

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.face_sel.any())

    def invoke(self, ctx, event) -> str:
        session = edit_session(ctx)
        before = session.begin()
        T.extrude_region(session.mesh, individual=True)
        session.push("挤出各个面", before)
        if view3d(ctx) is not None:
            ops.call("mesh.shrink_fatten", ctx, event, merge_previous=1)
        return FINISHED

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        before = session.begin()
        T.extrude_region(session.mesh, individual=True)
        session.push("挤出各个面", before)
        return FINISHED


def _selection_normal(mesh) -> np.ndarray | None:
    """选中面的平均法线（内部坐标）。"""
    faces = np.nonzero(mesh.face_sel)[0]
    if not len(faces):
        return None
    n = (mesh.face_normals()[faces] * mesh.face_areas()[faces, None]).sum(axis=0)
    length = np.linalg.norm(n)
    return n / length if length > 1e-12 else None


# ====================================================================== 内插、倒角、法向缩放（交互）
class _DragAmountOp(Operator):
    """按住拖动调一个数值（内插厚度、倒角宽度…）：鼠标离中心越远数值越大（drag="HORIZONTAL" 时往右拖越大）；
    Shift 精细、Ctrl 按 0.1 吸附；输入数字给精确值；滚轮改段数。"""

    redo = True
    amount_name = "thickness"
    drag = "RADIAL"
    allow_negative = False
    merge_previous = IntProperty("并入前几步", default=0, hidden=True)
    release_confirm = BoolProperty("松开确认", default=False, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx)

    def invoke(self, ctx, event) -> str:
        session = edit_session(ctx)
        self.editor = view3d(ctx)
        if self.editor is None:
            return self.execute(ctx)
        self.session = session
        self.base = session.begin()
        mesh = session.mesh
        sel = np.nonzero(mesh.vert_sel)[0]
        if not len(sel):
            return CANCELLED
        center3d = mesh.verts[sel].mean(axis=0)
        sx, sy, _f, dist = S.project(self.editor.view, center3d[None])
        self.center = (float(sx[0]), float(sy[0]))
        view = self.editor.view
        tan_y, _ = view.camera.tan_half(view.aspect)
        span = view.camera.distance if view.camera.ortho else float(dist[0])
        self.world_per_px = 2.0 * max(span, 1e-6) * tan_y / max(1, view.height)
        mouse = self.editor.cursor_pixel() if event is None or event.source is not self.editor.widget else \
            self.editor.pixel(event)
        self.start_dist = max(math.hypot(mouse[0] - self.center[0], mouse[1] - self.center[1]), 1.0)
        self.mouse0 = (float(mouse[0]), float(mouse[1]))
        self.button = event.type if event is not None and event.type in (LEFTMOUSE, MIDDLEMOUSE, RIGHTMOUSE) else None
        self._anchor = None
        self.numeric = ""
        self._apply_amount(ctx, self._initial())
        return RUNNING_MODAL

    def _drag_target(self, event) -> str:
        """拖动时改哪个数（内插：按住 Ctrl 改深度）。"""
        return self.amount_name

    def _drag(self, ctx, event, x: float, y: float) -> None:
        """按鼠标移动改数值：从开始拖（或换了 Shift、Ctrl）那一刻的值接着算，不跳。"""
        target = self._drag_target(event)
        precise = bool(event.shift)
        if self.drag == "HORIZONTAL":
            raw = (x - self.mouse0[0]) / 300.0
        else:
            raw = (math.hypot(x - self.center[0], y - self.center[1]) - self.start_dist) * self.world_per_px
        key = (target, precise)
        if self._anchor is None or self._anchor[0] != key:
            self._anchor = (key, raw, float(getattr(self, target)))
        _key, raw0, value0 = self._anchor
        value = value0 + (raw - raw0) * (0.1 if precise else 1.0)
        if event.ctrl and target == self.amount_name:
            value = round(value / 0.1) * 0.1
        if not self.allow_negative and target == self.amount_name:
            value = max(0.0, value)
        setattr(self, target, float(value))
        self._apply_amount(ctx, None)

    def _initial(self) -> float:
        return float(getattr(self, self.amount_name))

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            x, y = editor.pixel(event) if event.source is editor.widget else editor.cursor_pixel()
            if not self.numeric:
                self._drag(ctx, event, x, y)
            if self.drag == "RADIAL":
                editor.set_overlay_shape(("line", self.center, (x, y)))
            return RUNNING_MODAL
        if self.release_confirm and event.value == RELEASE and event.type == self.button:
            return self._confirm(ctx)
        if event.type in (WHEELUP, WHEELDOWN) or event.value == WHEEL:
            up = event.type == WHEELUP or getattr(event, "wheel", 0.0) > 0
            if self.on_wheel(1 if up else -1):
                self._apply_amount(ctx, float(getattr(self, self.amount_name)))
            return RUNNING_MODAL
        if event.value != PRESS:
            return RUNNING_MODAL
        key = event.type
        from .object_ops import DIGIT_KEYS

        if key in DIGIT_KEYS:
            char = DIGIT_KEYS[key]
            self.numeric = (self.numeric[1:] if self.numeric.startswith("-") else "-" + self.numeric) \
                if char == "-" else self.numeric + char
            try:
                value = float(self.numeric)
                self._apply_amount(ctx, value if self.allow_negative else abs(value))
            except ValueError:
                pass
            return RUNNING_MODAL
        if key == "BACK_SPACE":
            self.numeric = self.numeric[:-1]
            return RUNNING_MODAL
        if self.on_key(key):
            self._apply_amount(ctx, float(getattr(self, self.amount_name)))
            return RUNNING_MODAL
        if key in (LEFTMOUSE, "RET", "NUMPAD_ENTER", "SPACE"):
            return self._confirm(ctx)
        if key in (RIGHTMOUSE, "ESC"):
            self.cancel(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def _confirm(self, ctx) -> str:
        self._finish(ctx)
        self.session.push(self.label, self.base)
        history = ctx.app.history
        if self.merge_previous and history is not None:
            history.merge_last(int(self.merge_previous) + 1)
        _redraw(ctx)
        return FINISHED

    def _initial_amount(self) -> float:
        return 0.0

    def on_wheel(self, step: int) -> bool:
        return False

    def on_key(self, key: str) -> bool:
        return False

    def _apply_amount(self, ctx, amount: float | None) -> None:
        if amount is not None:
            setattr(self, self.amount_name, float(amount))
        self.session.mesh.restore(self.base)
        _engine(ctx).ctx_make_current()
        self.build(ctx, self.session.mesh)
        self.session.sync(topology=True)
        _hint(ctx, self.status())
        ctx.app.request_frame()

    def build(self, ctx, mesh) -> None:
        raise NotImplementedError

    def status(self) -> str:
        return ""

    def _finish(self, ctx) -> None:
        self.editor.set_overlay_shape(None)
        _hint(ctx, "")

    def cancel(self, ctx) -> None:
        self.session.mesh.restore(self.base)
        self.session.sync(topology=True)
        self._finish(ctx)

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        before = session.begin()
        _engine(ctx).ctx_make_current()
        self.build(ctx, session.mesh)
        session.push(self.label, before)
        _redraw(ctx)
        return FINISHED


@ops.register
class MeshInset(_DragAmountOp):
    idname = "mesh.inset"
    label = "内插面"
    description = "选中的面往里缩出一圈（I）：拖动调厚度，Ctrl 拖深度方向，再按 I 切换各自内插"
    thickness = FloatProperty("厚度", default=0.0, min=0.0, max=10000.0, precision=4, unit="m")
    depth = FloatProperty("深度", default=0.0, min=-10000.0, max=10000.0, precision=4, unit="m")
    use_individual = BoolProperty("各自内插", default=False)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.face_sel.any())

    def on_key(self, key: str) -> bool:
        if key == "I":
            self.use_individual = not self.use_individual
            return True
        return False

    def _drag_target(self, event) -> str:
        return "depth" if event.ctrl else "thickness"

    def build(self, ctx, mesh) -> None:
        T.inset_region(mesh, float(self.thickness), float(self.depth), bool(self.use_individual))

    def status(self) -> str:
        return "内插 厚度 %.4f  深度 %.4f%s    左键确认 · 右键取消 · Ctrl 拖深度 · I 各自内插 · 输入数字" % (
            self.thickness, self.depth, "（各自）" if self.use_individual else "")


@ops.register
class MeshBevel(_DragAmountOp):
    idname = "mesh.bevel"
    label = "倒角"
    description = "选中的边倒成斜面或圆角（Ctrl+B）；倒点时把选中的点切掉一个角（Ctrl+Shift+B）。拖动调宽度，滚轮改段数"
    icon = "tool.bevel"
    amount_name = "offset"
    affect = EnumProperty("倒", items=[("EDGES", "边", ""), ("VERTICES", "点", "")], default="EDGES")
    offset = FloatProperty("宽度", default=0.0, min=0.0, max=10000.0, precision=4, unit="m")
    segments = IntProperty("段数", default=1, min=1, max=100)
    clamp_overlap = BoolProperty("限制重叠", default=True, description="宽度不超过相邻边长的一半")

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.vert_sel.any())

    def on_wheel(self, step: int) -> bool:
        self.segments = max(1, int(self.segments) + step)
        return True

    def on_key(self, key: str) -> bool:
        if key == "V":
            self.affect = "VERTICES" if self.affect == "EDGES" else "EDGES"
            return True
        return False

    def build(self, ctx, mesh) -> None:
        if self.offset <= 0:
            return
        if self.affect == "VERTICES":
            bevel_vertices(mesh, float(self.offset), bool(self.clamp_overlap))
        else:
            bevel_edges(mesh, float(self.offset), int(self.segments), bool(self.clamp_overlap))

    def status(self) -> str:
        what = "倒点" if self.affect == "VERTICES" else "倒角"
        return "%s 宽度 %.4f  段数 %d    左键确认 · 右键取消 · 滚轮改段数 · V 倒边或倒点 · 输入数字" % (
            what, self.offset, self.segments)


@ops.register
class MeshShrinkFatten(_DragAmountOp):
    idname = "mesh.shrink_fatten"
    label = "法向缩放"
    description = "选中的点沿各自的法线往外推或往里收（Alt+S）"
    icon = "tool.shrink_fatten"
    amount_name = "value"
    allow_negative = True
    value = FloatProperty("距离", default=0.0, min=-10000.0, max=10000.0, precision=4, unit="m")

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.vert_sel.any())

    def build(self, ctx, mesh) -> None:
        sel = np.nonzero(mesh.vert_sel)[0]
        normals = mesh.vert_normals()
        mesh.verts[sel] += normals[sel] * float(self.value)

    def status(self) -> str:
        return "法向缩放 %.4f    左键确认 · 右键取消" % self.value


# ====================================================================== 环切、刀切
@ops.register
class MeshLoopcutSlide(Operator):
    idname = "mesh.loopcut_slide"
    label = "环切"
    description = "鼠标指着一条边，沿它那一圈切一刀（Ctrl+R）：滚轮改刀数，左键确认后可以滑动位置"
    redo = True
    icon = "tool.loopcut"
    number_cuts = IntProperty("刀数", default=1, min=1, max=100)
    edge_index = IntProperty("边", default=-1, hidden=True)
    factor = FloatProperty("位置", default=0.0, min=-1.0, max=1.0, precision=3)
    release_confirm = BoolProperty("松开确认", default=False, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.session = edit_session(ctx)
        self.stage = "PICK"
        self.base = self.session.begin()
        self.button = event.type if event is not None and event.type in (LEFTMOUSE, MIDDLEMOUSE, RIGHTMOUSE) else None
        from .mesh_edit_tools import _ProbeCache, _pixel

        self.probes = _ProbeCache(_engine(ctx), self.editor.view)
        self.mouse = _pixel(self.editor, event)
        self._preview(ctx)
        if self.release_confirm:
            # 工具栏的环切：按下就切，拖动滑动，松开确认；没指着边时交给点选
            if self.edge_index < 0:
                self.editor.set_overlay_shape(None)
                return PASS_THROUGH
            self._cut(ctx)
            self.stage = "SLIDE"
            self.slide_start = self.mouse
            _hint(ctx, "滑动切口：拖动鼠标，松开确认")
            return RUNNING_MODAL
        _hint(ctx, "环切：移动鼠标选位置，滚轮改刀数，左键确认后拖动滑动，右键或 Esc 取消")
        return RUNNING_MODAL

    def _preview(self, ctx) -> None:
        editor = self.editor
        mesh = self.session.mesh
        engine = _engine(ctx)
        engine.ctx_make_current()
        x, y = self.mouse
        xray = _xray(editor)
        found = S._nearest_edge(engine, editor.view, mesh, x, y, xray, 60.0,
                                probe=None if xray else self.probes.get())
        if found is None:
            editor.set_overlay_shape(None)
            self.edge_index = -1
            return
        self.edge_index = int(found[0])
        ring, closed = LC.edge_ring(mesh, self.edge_index)
        table = mesh.edges()
        # 预览：每一刀画一条黄线，穿过每条并排边上将来切出的点
        a = mesh.verts[table.edges[ring, 0]]
        b = mesh.verts[table.edges[ring, 1]]
        flip = _ring_flips(mesh, ring)
        a, b = np.where(flip[:, None], b, a), np.where(flip[:, None], a, b)
        shapes = []
        cuts = int(self.number_cuts)
        for k in range(1, cuts + 1):
            t = k / (cuts + 1) if cuts > 1 else 0.5 + 0.5 * float(np.clip(self.factor, -0.999, 0.999))
            sx, sy, _f, _d = S.project(editor.view, a + (b - a) * t)
            pts = list(zip(sx.tolist(), sy.tolist()))
            if closed and pts:
                pts.append(pts[0])
            shapes.append(("polyline", pts, (1.0, 0.85, 0.15)))
        editor.set_overlay_shape(("multi", tuple(shapes)))

    def modal(self, ctx, event) -> str:
        if event.type == MIDDLEMOUSE:
            return PASS_THROUGH                        # 选位置时也能转视角
        if event.type == MOUSEMOVE:
            from .mesh_edit_tools import _pixel

            self.mouse = _pixel(self.editor, event)
        if self.stage == "PICK":
            if event.type == MOUSEMOVE:
                self._preview(ctx)
                return RUNNING_MODAL
            if event.type in (WHEELUP, WHEELDOWN) or event.value == WHEEL:
                up = event.type == WHEELUP or getattr(event, "wheel", 0.0) > 0
                self.number_cuts = max(1, int(self.number_cuts) + (1 if up else -1))
                _hint(ctx, "环切 %d 刀：左键确认，滚轮改刀数" % self.number_cuts)
                self._preview(ctx)
                return RUNNING_MODAL
            if event.value == PRESS and event.type in ("NUMPAD_PLUS", "PAGE_UP"):
                self.number_cuts += 1
                self._preview(ctx)
                return RUNNING_MODAL
            if event.value == PRESS and event.type in ("NUMPAD_MINUS", "PAGE_DOWN"):
                self.number_cuts = max(1, self.number_cuts - 1)
                self._preview(ctx)
                return RUNNING_MODAL
            if event.value == PRESS and event.type == LEFTMOUSE:
                if self.edge_index < 0:
                    return RUNNING_MODAL
                self._cut(ctx)
                if self.number_cuts == 1:
                    self.stage = "SLIDE"
                    self.slide_start = self.mouse
                    _hint(ctx, "滑动切口：移动鼠标，左键确认，右键放在正中间")
                    return RUNNING_MODAL
                self._done(ctx)
                return FINISHED
            if event.value == PRESS and event.type in (RIGHTMOUSE, "ESC"):
                self.cancel(ctx)
                return CANCELLED
            return RUNNING_MODAL
        # 滑动
        if event.type == MOUSEMOVE:
            x, _y = self.mouse
            self.factor = float(np.clip((x - self.slide_start[0]) / 200.0, -0.95, 0.95))
            self._cut(ctx)
            return RUNNING_MODAL
        if self.release_confirm and event.value == RELEASE and event.type == self.button:
            self._done(ctx)
            return FINISHED
        if event.value == PRESS and event.type in (LEFTMOUSE, "RET", "NUMPAD_ENTER"):
            self._done(ctx)
            return FINISHED
        if event.value == PRESS and event.type in (RIGHTMOUSE, "ESC"):
            self.factor = 0.0
            self._cut(ctx)
            self._done(ctx)
            return FINISHED
        return RUNNING_MODAL

    def _cut(self, ctx) -> None:
        mesh = self.session.mesh
        mesh.restore(self.base)
        _engine(ctx).ctx_make_current()
        LC.loop_cut(mesh, int(self.edge_index), int(self.number_cuts), float(self.factor))
        self.session.sync(topology=True)
        self.editor.set_overlay_shape(None)
        ctx.app.request_frame()

    def _done(self, ctx) -> None:
        self.editor.set_overlay_shape(None)
        _hint(ctx, "")
        self.session.push("环切", self.base)
        _redraw(ctx)

    def cancel(self, ctx) -> None:
        self.session.mesh.restore(self.base)
        self.session.sync(topology=True)
        self.editor.set_overlay_shape(None)
        _hint(ctx, "")

    def execute(self, ctx) -> str:
        session = edit_session(ctx)
        if self.edge_index < 0:
            return CANCELLED
        before = session.begin()
        LC.loop_cut(session.mesh, int(self.edge_index), int(self.number_cuts), float(self.factor))
        session.push("环切", before)
        return FINISHED


def _ring_flips(mesh, ring) -> np.ndarray:
    """并排边各自要不要掉头，让一圈边的方向一致（预览线才连得顺）。"""
    table = mesh.edges()
    pts = mesh.verts[table.edges[ring]]
    flip = np.zeros(len(ring), bool)
    for k in range(1, len(ring)):
        a0, b0 = (pts[k - 1][1], pts[k - 1][0]) if flip[k - 1] else (pts[k - 1][0], pts[k - 1][1])
        a1, b1 = pts[k][0], pts[k][1]
        same = np.linalg.norm(a1 - a0) + np.linalg.norm(b1 - b0)
        cross = np.linalg.norm(b1 - a0) + np.linalg.norm(a1 - b0)
        flip[k] = cross < same
    return flip


# ====================================================================== 细分、三角化、法线、平滑
@ops.register
class MeshSubdivide(_EditOp):
    idname = "mesh.subdivide"
    label = "细分"
    description = "选中的面切成更小的面"
    number_cuts = IntProperty("刀数", default=1, min=1, max=100)
    smoothness = FloatProperty("平滑", default=0.0, min=0.0, max=1.0, subtype="FACTOR")

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.face_sel.any())

    def apply(self, ctx, session, mesh) -> bool:
        return T.subdivide(mesh, int(self.number_cuts), float(self.smoothness)) > 0


@ops.register
class MeshTriangulate(_EditOp):
    idname = "mesh.quads_convert_to_tris"
    label = "三角化面"

    def apply(self, ctx, session, mesh) -> bool:
        return T.triangulate_faces(mesh) > 0


@ops.register
class MeshTrisToQuads(_EditOp):
    idname = "mesh.tris_convert_to_quads"
    label = "三角面转四边面"
    face_threshold = FloatProperty("最大夹角", default=40.0, min=0.0, max=180.0, unit="°")

    def apply(self, ctx, session, mesh) -> bool:
        return T.tris_to_quads(mesh, float(self.face_threshold)) > 0


@ops.register
class MeshNormalsMakeConsistent(_EditOp):
    idname = "mesh.normals_make_consistent"
    label = "重新计算法线"
    description = "让选中面的朝向一致并朝外（Shift+N），勾「朝内」则朝里"
    inside = BoolProperty("朝内", default=False)

    def apply(self, ctx, session, mesh) -> bool:
        T.recalc_normals(mesh, bool(self.inside))
        return True


@ops.register
class MeshFlipNormals(_EditOp):
    idname = "mesh.flip_normals"
    label = "翻转法线"

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.face_sel.any())

    def apply(self, ctx, session, mesh) -> bool:
        T.flip_normals(mesh)
        return True


@ops.register
class MeshVerticesSmooth(_DragAmountOp):
    idname = "mesh.vertices_smooth"
    label = "平滑顶点"
    description = "选中的点往周围点的平均位置靠"
    icon = "tool.smooth"
    amount_name = "factor"
    drag = "HORIZONTAL"
    factor = FloatProperty("平滑度", default=0.5, min=0.0, max=1.0, subtype="FACTOR")
    repeat = IntProperty("次数", default=1, min=1, max=100)
    interactive = BoolProperty("拖动调节", default=False, hidden=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.vert_sel.any())

    def invoke(self, ctx, event) -> str:
        if not self.interactive:
            return self.execute(ctx)
        self.factor = 0.0
        return super().invoke(ctx, event)

    def on_wheel(self, step: int) -> bool:
        self.repeat = max(1, int(self.repeat) + step)
        return True

    def build(self, ctx, mesh) -> None:
        T.smooth_verts(mesh, None, float(self.factor), int(self.repeat))

    def status(self) -> str:
        return "平滑 %.3f  次数 %d    左右拖动 · 滚轮改次数 · 左键确认 · 右键取消" % (self.factor, self.repeat)


@ops.register
class MeshFacesShade(_EditOp):
    idname = "mesh.faces_shade"
    label = "面的着色"
    smooth = BoolProperty("平滑", default=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.face_sel.any())

    def apply(self, ctx, session, mesh) -> bool:
        mesh.face_smooth[mesh.face_sel] = bool(self.smooth)
        return True


# ====================================================================== UV
@ops.register
class MeshUVUnwrap(_EditOp):
    idname = "uv.smart_project"
    label = "智能 UV 投射"
    description = "选中的面按朝向分块展开，排进 0–1 的贴图空间"
    angle_limit = FloatProperty("角度限制", default=66.0, min=1.0, max=89.0, unit="°")
    island_margin = FloatProperty("块间距", default=0.005, min=0.0, max=1.0, precision=4)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and bool(edit_session(ctx).mesh.face_sel.any())

    def apply(self, ctx, session, mesh) -> bool:
        from ..geometry.unwrap import smart_unwrap
        from ..mesh.editmesh import triangulate

        faces = np.nonzero(mesh.face_sel)[0]
        tris, face_of_tri = triangulate(mesh)
        tris = tris[np.isin(face_of_tri, faces)]
        if not len(tris):
            return False
        uvs, _stats = smart_unwrap(mesh.verts, mesh.loop_vert[tris], angle_limit=float(self.angle_limit),
                                   margin_texels=max(1.0, float(self.island_margin) * 1024.0), resolution=1024)
        mesh.loop_uv[tris.reshape(-1)] = np.asarray(uvs, np.float32).reshape(-1, 2)
        mesh.has_uvs = True
        return True


@ops.register
class MeshUVReset(_EditOp):
    idname = "uv.reset"
    label = "重置 UV"
    description = "每个面铺满整个贴图"

    def apply(self, ctx, session, mesh) -> bool:
        faces = np.nonzero(mesh.face_sel)[0]
        for f in faces:
            s, e = mesh.face_start[f], mesh.face_start[f + 1]
            n = e - s
            if n == 4:
                mesh.loop_uv[s:e] = [(0, 0), (1, 0), (1, 1), (0, 1)]
            else:
                ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
                mesh.loop_uv[s:e] = np.stack([0.5 + 0.5 * np.cos(ang), 0.5 + 0.5 * np.sin(ang)], axis=1)
        mesh.has_uvs = True
        return True


@ops.register
class MeshUVProjectFromView(_EditOp):
    idname = "uv.project_from_view"
    label = "从视角投射"
    description = "按现在屏幕上看到的样子投到贴图上"
    scale_to_bounds = BoolProperty("铺满", default=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return edit_mode(ctx) and view3d(ctx) is not None and bool(edit_session(ctx).mesh.face_sel.any())

    def apply(self, ctx, session, mesh) -> bool:
        editor = view3d(ctx)
        lf = mesh.loop_face()
        loops = np.nonzero(mesh.face_sel[lf])[0]
        sx, sy, _f, _d = S.project(editor.view, mesh.verts[mesh.loop_vert[loops]])
        u = sx / max(1, editor.view.width)
        v = 1.0 - sy / max(1, editor.view.height)
        if self.scale_to_bounds:
            u = (u - u.min()) / max(np.ptp(u), 1e-9)
            v = (v - v.min()) / max(np.ptp(v), 1e-9)
        mesh.loop_uv[loops] = np.stack([u, v], axis=1)
        mesh.has_uvs = True
        return True
