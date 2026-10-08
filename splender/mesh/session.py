"""编辑模式会话：进入编辑模式时把物体的三角形数据变成可编辑的多边形网格（EditMesh），退出时三角化写回。

编辑过程中：
- 视口里照常用物体的材质显示（模型的显卡数据随编辑更新，绘制用的分块几何等退出时再重建）；
- 叠加层画点、边、选中的面（照 Blender 的颜色：没选中黑色、选中橙色、当前元素白色）；
- 每一步操作是一步撤销（整份网格的快照）；退出时这些步骤合成一步「编辑网格」。
"""
from __future__ import annotations

import logging
import time

import moderngl
import numpy as np

from .editmesh import EditMesh, from_mesh_data, loop_normals, to_mesh_data

log = logging.getLogger("splender.mesh")

_VERT = """#version 430
in vec3 in_pos;
in float in_state;
uniform mat4 u_view_proj;
uniform float u_bias;
uniform float u_point;
out float v_state;
void main() {
  gl_Position = u_view_proj * vec4(in_pos, 1.0);
  gl_Position.z -= u_bias * gl_Position.w;
  gl_PointSize = u_point * (in_state > 1.5 ? 1.3 : 1.0);
  v_state = in_state;
}
"""

_FRAG = """#version 430
in float v_state;
uniform vec4 u_normal;
uniform vec4 u_selected;
uniform vec4 u_active;
uniform int u_round;
out vec4 o_color;
void main() {
  if (u_round == 1) {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    if (dot(c, c) > 1.0) discard;
  }
  o_color = v_state > 1.5 ? u_active : (v_state > 0.5 ? u_selected : u_normal);
}
"""

# Blender 默认主题的颜色
VERT_COLOR = (0.0, 0.0, 0.0, 1.0)
VERT_SELECT = (1.0, 0.478, 0.0, 1.0)
EDGE_COLOR = (0.0, 0.0, 0.0, 0.85)
EDGE_SELECT = (1.0, 0.627, 0.0, 1.0)
ACTIVE_COLOR = (1.0, 1.0, 1.0, 1.0)
FACE_SELECT = (1.0, 0.647, 0.0, 0.2)
FACE_DOT = (0.0, 0.0, 0.0, 1.0)


from ..sculpt.session import snapshot_mesh  # noqa: E402  形状快照（含多边形大小、散边）


class EditOverlay:
    """点、边、选中的面的显卡数据。"""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self.buffers: dict[str, moderngl.Buffer] = {}
        self.vaos: dict[str, moderngl.VertexArray] = {}
        self.counts = {"points": 0, "lines": 0, "faces": 0, "dots": 0}

    def _upload(self, name: str, data: np.ndarray) -> None:
        raw = np.ascontiguousarray(data, np.float32).tobytes()
        old = self.buffers.get(name)
        if old is not None and old.size >= len(raw) > 0:
            old.write(raw)
        else:
            if old is not None:
                old.release()
                vao = self.vaos.pop(name, None)
                if vao is not None:
                    vao.release()
            self.buffers[name] = self.ctx.buffer(raw if raw else bytes(16), dynamic=True)
        if name not in self.vaos:
            self.vaos[name] = self.ctx.vertex_array(self.program, [(self.buffers[name], "3f 1f", "in_pos", "in_state")])

    def update(self, mesh: EditMesh, show_face_dots: bool) -> None:
        verts = mesh.verts.astype(np.float32)
        vis_v = ~mesh.vert_hide
        # 点：0 没选、1 选中、2 当前
        state_v = mesh.vert_sel.astype(np.float32)
        if mesh.active is not None and mesh.active[0] == "VERT" and mesh.active[1] < len(state_v):
            state_v[mesh.active[1]] = 2.0
        pts = np.concatenate([verts[vis_v], state_v[vis_v, None]], axis=1) if vis_v.any() else np.zeros((0, 4))
        self.counts["points"] = len(pts)
        self._upload("points", pts)
        table = mesh.edges()
        edges = table.edges
        if len(edges):
            vis_e = ~(mesh.vert_hide[edges[:, 0]] | mesh.vert_hide[edges[:, 1]])
            if len(mesh.face_hide) and mesh.loop_count:
                # 只属于隐藏面的边也不画
                lf = mesh.loop_face()
                shown = np.zeros(len(edges), bool)
                shown[table.loop_edge[~mesh.face_hide[lf]]] = True
                vis_e &= shown | table.loose
            state_e = mesh.edge_sel.astype(np.float32)
            if mesh.active is not None and mesh.active[0] == "EDGE" and mesh.active[1] < len(state_e):
                state_e[mesh.active[1]] = 2.0
            e = edges[vis_e]
            s = np.repeat(state_e[vis_e], 2)
            lines = np.concatenate([verts[e.reshape(-1)], s[:, None]], axis=1)
        else:
            lines = np.zeros((0, 4), np.float32)
        self.counts["lines"] = len(lines)
        self._upload("lines", lines)
        # 选中的面：半透明橙色
        faces = np.nonzero(mesh.face_sel & ~mesh.face_hide)[0]
        if len(faces):
            from .editmesh import triangulate

            tris, face_of_tri = triangulate(mesh)
            keep = np.isin(face_of_tri, faces)
            corners = mesh.loop_vert[tris[keep].reshape(-1)]
            tri = np.concatenate([verts[corners], np.ones((len(corners), 1), np.float32)], axis=1)
        else:
            tri = np.zeros((0, 4), np.float32)
        self.counts["faces"] = len(tri)
        self._upload("faces", tri)
        if show_face_dots and mesh.face_count:
            centers = mesh.face_centers().astype(np.float32)
            shown = ~mesh.face_hide
            state_f = mesh.face_sel.astype(np.float32)
            if mesh.active is not None and mesh.active[0] == "FACE" and mesh.active[1] < len(state_f):
                state_f[mesh.active[1]] = 2.0
            dots = np.concatenate([centers[shown], state_f[shown, None]], axis=1)
        else:
            dots = np.zeros((0, 4), np.float32)
        self.counts["dots"] = len(dots)
        self._upload("dots", dots)

    def draw(self, view_proj: np.ndarray, *, xray: bool, point_px: float, edge_px: float, modes: set) -> None:
        ctx = self.ctx
        program = self.program
        program["u_view_proj"].write(np.asarray(view_proj, np.float32).T.tobytes())
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE, moderngl.ONE
        if xray:
            ctx.disable(moderngl.DEPTH_TEST)
        else:
            ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_mask = False
        if self.counts["faces"]:
            program["u_bias"] = 1e-4
            program["u_round"] = 0
            program["u_normal"] = FACE_SELECT
            program["u_selected"] = FACE_SELECT
            program["u_active"] = FACE_SELECT
            self.vaos["faces"].render(moderngl.TRIANGLES, vertices=self.counts["faces"])
        if self.counts["lines"]:
            program["u_bias"] = 3e-4
            program["u_round"] = 0
            program["u_normal"] = EDGE_COLOR
            program["u_selected"] = EDGE_SELECT
            program["u_active"] = ACTIVE_COLOR
            ctx.line_width = float(edge_px)
            self.vaos["lines"].render(moderngl.LINES, vertices=self.counts["lines"])
            ctx.line_width = 1.0
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        if "VERT" in modes and self.counts["points"]:
            program["u_bias"] = 5e-4
            program["u_round"] = 1
            program["u_point"] = float(point_px)
            program["u_normal"] = VERT_COLOR
            program["u_selected"] = VERT_SELECT
            program["u_active"] = ACTIVE_COLOR
            self.vaos["points"].render(moderngl.POINTS, vertices=self.counts["points"])
        if "FACE" in modes and self.counts["dots"]:
            program["u_bias"] = 5e-4
            program["u_round"] = 1
            program["u_point"] = float(point_px)
            program["u_normal"] = FACE_DOT
            program["u_selected"] = VERT_SELECT
            program["u_active"] = ACTIVE_COLOR
            self.vaos["dots"].render(moderngl.POINTS, vertices=self.counts["dots"])
        ctx.disable(moderngl.PROGRAM_POINT_SIZE)
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

    def release(self) -> None:
        for vao in self.vaos.values():
            vao.release()
        for buf in self.buffers.values():
            buf.release()
        self.program.release()
        self.vaos.clear()
        self.buffers.clear()


class EditSession:
    def __init__(self, engine, obj) -> None:
        started = time.perf_counter()
        self.version = 0
        self._changed = False
        self.engine = engine
        self.obj = obj
        self.entry_snapshot = snapshot_mesh(obj.data)
        self.mesh: EditMesh = from_mesh_data(obj.data)
        self.tag = object()
        self.changed = False
        self.overlay = EditOverlay(engine.ctx)
        self.point_px = 4.0
        self.edge_px = 1.0
        self.corner_loop = None                 # 显示用的三角形每个角是哪个面角（第一次改形状时才换显示数据）
        engine.renderer.extra_draws.append(self._draw)
        self.update_overlay()
        log.info("进入编辑模式：%s，%d 个点、%d 个面，用时 %.2f 秒", obj.name, self.mesh.vert_count, self.mesh.face_count,
                 time.perf_counter() - started)

    # 「编辑以来改没改过形状」；每次标成 True 都换一个新版本号（自动保存据此判断形状变没变）
    @property
    def changed(self) -> bool:
        return self._changed

    @changed.setter
    def changed(self, value: bool) -> None:
        self._changed = bool(value)
        if value:
            self.version += 1

    # ---------------------------------------------------------------- 显示
    def sync(self, topology: bool = True) -> None:
        """网格变了：更新视口里的模型和叠加层。topology=False 时只是点挪了位置（快）。"""
        mesh = self.mesh
        obj = self.obj
        engine = self.engine
        if topology or self.corner_loop is None:
            data, corner_loop = to_mesh_data(mesh, obj.data, skip_hidden=True)
            data.name = obj.data.name
            self.corner_loop = corner_loop
            obj.data = data
            engine.edit_mesh_changed(obj, topology=True)
        else:
            data = obj.data
            corner = self.corner_loop
            data.positions = mesh.verts[mesh.loop_vert[corner]].astype(np.float32)
            data.normals = loop_normals(mesh)[corner].astype(np.float32)
            if len(data.positions):
                data.bounds_min = data.positions.min(axis=0)
                data.bounds_max = data.positions.max(axis=0)
            engine.edit_mesh_changed(obj, topology=False)
        self.update_overlay()

    def update_overlay(self) -> None:
        self.engine.ctx_make_current()
        self.overlay.update(self.mesh, "FACE" in self.mesh.select_mode)
        for view in self.engine.views:
            view.dirty = True
        self.engine._pending_work = True

    def _draw(self, renderer, view, pass_name: str) -> None:
        if pass_name != "color":
            return
        overlay = getattr(view, "overlay", None)
        if overlay is not None and not overlay.show_overlays:
            return
        xray = bool(getattr(view.shading, "show_xray", False)) or view.shading.mode == "WIREFRAME"
        self.overlay.draw(view.camera.view_projection(view.aspect), xray=xray, point_px=self.point_px,
                          edge_px=self.edge_px, modes=self.mesh.select_mode)

    def full_data(self):
        """完整的网格（含隐藏的面），存盘用。"""
        data, _corner = to_mesh_data(self.mesh, self.obj.data)
        data.name = getattr(self.obj.data, "name", data.name)
        return data

    # ---------------------------------------------------------------- 撤销
    def begin(self) -> dict:
        """一步操作开始前的快照。"""
        return self.mesh.snapshot()

    def push(self, label: str, before: dict, topology: bool = True) -> None:
        """一步操作做完：更新显示，记一步撤销。"""
        self.changed = True
        self.sync(topology=topology)
        after = self.mesh.snapshot()
        history = self.engine.history
        if history is None:
            return
        session = self

        def restore(snap) -> None:
            if session.engine.edit is not session:
                return
            session.engine.ctx_make_current()
            session.mesh.restore(snap)
            session.sync(topology=True)

        nbytes = sum(getattr(v, "nbytes", 0) for v in before.values()) * 2
        history.push(label, lambda: restore(before), lambda: restore(after), nbytes)
        history.steps[-1].tag = self.tag

    def push_selection(self, label: str, before: dict) -> None:
        """只改了选中（不动形状）。"""
        self.update_overlay()
        after = {k: self.mesh.snapshot()[k] for k in ("vert_sel", "edge_sel", "face_sel", "active", "select_mode")}
        before = {k: before[k] for k in after}
        history = self.engine.history
        if history is None:
            return
        session = self

        def restore(snap) -> None:
            if session.engine.edit is not session:
                return
            for key, value in snap.items():
                setattr(session.mesh, key, value.copy() if hasattr(value, "copy") else value)
            session.update_overlay()

        history.push(label, lambda: restore(before), lambda: restore(after))
        history.steps[-1].tag = self.tag

    # ---------------------------------------------------------------- 退出
    def close(self) -> dict | None:
        """退出：有改动就三角化写回物体（调用方重建绘制数据），没改动什么都不动。"""
        renderer = self.engine.renderer
        if self._draw in renderer.extra_draws:
            renderer.extra_draws.remove(self._draw)
        self.engine.ctx_make_current()
        self.overlay.release()
        history = self.engine.history
        if history is not None:
            history.remove_tagged(self.tag)
        for view in self.engine.views:
            view.dirty = True
        if not self.changed:
            return {"changed": False}
        data, _corner = to_mesh_data(self.mesh, self.obj.data)
        data.name = getattr(self.obj.data, "name", data.name)
        self.obj.data = data
        self.obj.geometry_dirty = True
        return {"changed": True, "before": self.entry_snapshot, "after": snapshot_mesh(self.obj.data)}
