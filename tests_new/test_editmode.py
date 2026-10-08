"""编辑模式（照 Blender）：点选、框选、G/R/S（数值、约束、法向、衰减编辑）、挤出后接着移动和「调整上一步」、
切刀、环切、倒角、内插、撕开、滑移边、加形体、隐藏、吸附、镜像选择，撤销重做都对；编辑模式里存盘写进去的是编辑着的网格。
隐藏上下文，不开窗口。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parent))     # 同目录的 test_objects（隔离模式下脚本目录不在路径里）
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from splender.core import ops  # noqa: E402
from splender.core.context import Context  # noqa: E402
from splender.core.history import History  # noqa: E402
from splender.core.keymap import Event  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.objects import to_display, to_internal  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet, ViewOverlay, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from splender.geometry import primitives  # noqa: E402

QT = QApplication.instance() or QApplication(sys.argv)
from splender.ops import mesh_edit_ops, mesh_edit_tools, object_ops, view3d_ops, wm_ops  # noqa: E402,F401
from test_objects import FakeApp, FakeEditor, key, move  # noqa: E402


def click(editor, x, y, value="CLICK", **mods):
    return Event(type="LEFTMOUSE", value=value, x=float(x), y=float(y), source=editor.widget, **mods)


class EditModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = Path(tempfile.mkdtemp())

    def make(self, mesh, name="立方体", top=False):
        project = Project("编辑")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="1024"))
        ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.6, 0.6, 0.62)))
        obj = MeshObject(name, mesh, [ts.uid])
        project.add_object(obj)
        obj.select = True
        project.active_object_uid = obj.uid
        self.project = project
        self.obj = obj
        self.engine = Engine(self.ctx, Preferences(), History())
        self.addCleanup(self.engine.release)
        self.engine.set_project(project)
        self.view = self.engine.create_view(ViewShading(mode="SOLID"), ViewOverlay())
        self.view.resize(640, 480)
        camera = self.view.camera
        camera.target = np.zeros(3)
        camera.distance = 12.0
        camera.yaw, camera.pitch = 0.0, 0.0        # 前视图：屏幕右 = X，屏幕上 = Z
        if top:
            camera.pitch = 89.999                  # 顶视图：屏幕右 = X，屏幕上 = Y
        camera.ortho = True
        self.view.invalidate_camera()
        self.app = FakeApp(self.engine, project)
        self.engine.history = self.app.history
        self.editor = FakeEditor(self.view)
        self.app.active_view3d = self.editor
        self.ctx_op = Context(app=self.app, wm=None, editor=self.editor)
        self.engine.settle()
        self.session = self.engine.enter_edit(obj)
        self.app.tool_settings.mode = "EDIT"
        self.engine.settle()
        return self.session.mesh

    def call(self, idname, event=None, invoke=True, **props):
        return ops.call(idname, self.ctx_op, event, invoke=invoke, **props)

    def screen_of(self, point_display):
        p = to_internal(np.asarray(point_display, np.float64))
        clip = self.view.camera.view_projection(self.view.aspect) @ np.array([p[0], p[1], p[2], 1.0])
        return ((clip[0] / clip[3] * 0.5 + 0.5) * self.view.width, (0.5 - clip[1] / clip[3] * 0.5) * self.view.height)

    def disp(self, mesh):
        return to_display(mesh.verts.T).T

    def top_face(self, mesh):
        fn = mesh.face_normals()
        return int(np.argmax(fn @ to_internal(np.array([0.0, 0.0, 1.0]))))

    def select_face(self, mesh, f):
        mesh.set_select_mode({"FACE"})
        mesh.select_all("DESELECT")
        mesh.face_sel[f] = True
        mesh.flush_from_faces()

    # ------------------------------------------------------------------
    def test_select_and_transform(self):
        mesh = self.make(primitives.cube(2.0))
        self.assertEqual((mesh.vert_count, mesh.face_count), (8, 6))
        # 点选正前方右上角的点（前视图里它在最前面）
        d = self.disp(mesh)
        corner = int(np.argmin(np.linalg.norm(d - np.array([1.0, -1.0, 1.0]), axis=1)))
        x, y = self.screen_of(d[corner])
        self.assertEqual(self.call("view3d.select", click(self.editor, x, y), deselect_all=True), "FINISHED")
        self.assertEqual(list(np.nonzero(mesh.vert_sel)[0]), [corner])
        # 框选整个立方体（前视图里背面的点被挡住，开 X 光框全部）
        self.view.shading.show_xray = True
        self.editor.shading = self.view.shading
        x0, y0 = self.screen_of((-1.5, 0.0, 1.5))
        x1, y1 = self.screen_of((1.5, 0.0, -1.5))
        op = object_ops.View3DSelectBox()
        op.invoke(self.ctx_op, click(self.editor, x0, y0, "CLICK_DRAG"))
        op.modal(self.ctx_op, move(self.editor, x1, y1))
        self.assertEqual(op.modal(self.ctx_op, click(self.editor, x1, y1, "RELEASE")), "FINISHED")
        self.assertEqual(int(mesh.vert_sel.sum()), 8)
        # G X 1 回车：所有点沿 X 移 1
        before = self.disp(mesh).copy()
        op = object_ops.TransformTranslate()
        self.editor.cursor = self.screen_of((0.0, 0.0, 0.0))
        self.assertEqual(op.invoke(self.ctx_op, key("G")), "RUNNING_MODAL")
        op.modal(self.ctx_op, key("X"))
        op.modal(self.ctx_op, key("1"))
        self.assertEqual(op.modal(self.ctx_op, key("RET")), "FINISHED")
        np.testing.assert_allclose(self.disp(mesh) - before, np.tile([1.0, 0.0, 0.0], (8, 1)), atol=1e-6)
        # 显示用的模型跟着变了
        lo = to_display(np.asarray(self.obj.data.bounds_min, np.float64))
        hi = to_display(np.asarray(self.obj.data.bounds_max, np.float64))
        np.testing.assert_allclose((np.minimum(lo, hi) + np.maximum(lo, hi)) * 0.5, [1.0, 0.0, 0.0], atol=1e-5)
        # R Z 90、S 2
        op = object_ops.TransformRotate()
        op.invoke(self.ctx_op, key("R"))
        op.modal(self.ctx_op, key("Z"))
        for ch in "90":
            op.modal(self.ctx_op, key(ch))
        op.modal(self.ctx_op, key("RET"))
        op = object_ops.TransformResize()
        op.invoke(self.ctx_op, key("S"))
        op.modal(self.ctx_op, key("2"))
        op.modal(self.ctx_op, key("RET"))
        span = self.disp(mesh).max(axis=0) - self.disp(mesh).min(axis=0)
        np.testing.assert_allclose(span, [4.0, 4.0, 4.0], atol=1e-5)
        # 右键取消：点回到原处
        snapshot = mesh.verts.copy()
        op = object_ops.TransformTranslate()
        op.invoke(self.ctx_op, key("G"))
        op.modal(self.ctx_op, move(self.editor, 400, 100))
        self.assertEqual(op.modal(self.ctx_op, key("RIGHTMOUSE")), "CANCELLED")
        np.testing.assert_allclose(mesh.verts, snapshot)
        # 撤销三步回到移动前
        for _ in range(3):
            self.app.history.undo()
        np.testing.assert_allclose(self.disp(mesh), before, atol=1e-6)
        self.app.history.redo()
        np.testing.assert_allclose(self.disp(mesh) - before, np.tile([1.0, 0.0, 0.0], (8, 1)), atol=1e-6)

    def test_extrude_move_and_adjust(self):
        from splender.ui.redo import RedoController

        mesh = self.make(primitives.cube(2.0))
        controller = RedoController(self.app)
        self.addCleanup(ops.finished.disconnect, controller._on_finished)
        top = self.top_face(mesh)
        self.select_face(mesh, top)
        self.editor.cursor = self.screen_of((0.0, 0.0, 1.0))
        steps = len(self.app.history.steps)
        self.assertEqual(self.call("view3d.edit_mesh_extrude_move_normal", key("E")), "FINISHED")
        translate = self.app_modal_op()
        for ch in "1":
            translate.modal(self.ctx_op, key(ch))
        self.assertEqual(translate.modal(self.ctx_op, key("RET")), "FINISHED")
        ops.finished.emit(translate, self.ctx_op, "FINISHED")
        self.assertEqual((mesh.vert_count, mesh.face_count), (12, 10))
        top_z = self.disp(mesh)[mesh.vert_sel][:, 2]
        np.testing.assert_allclose(top_z, 2.0, atol=1e-6)
        # 挤出和移动合成一步撤销
        self.assertEqual(len(self.app.history.steps), steps + 1)
        self.assertEqual(self.app.history.steps[-1].label, "挤出")
        # 调整上一步：只重做移动，挤出留着
        translate.value = (0.0, 0.0, 3.0)
        self.assertTrue(controller.adjust())
        self.assertEqual((mesh.vert_count, mesh.face_count), (12, 10))
        np.testing.assert_allclose(self.disp(mesh)[mesh.vert_sel][:, 2], 4.0, atol=1e-6)
        self.assertEqual(len(self.app.history.steps), steps + 1)
        self.app.history.undo()
        self.assertEqual((mesh.vert_count, mesh.face_count), (8, 6))

    def app_modal_op(self):
        """ops.call 把模态操作交给窗口管理器；测试里没有窗口管理器，从 ops 的记录里取最近的一个。"""
        return self._last_modal

    def setUp(self):
        self._last_modal = None
        original = ops.call

        def tracking_call(idname, ctx, event=None, *, invoke=True, **props):
            cls = ops.get(idname)
            if cls is None or not ops.poll(idname, ctx):
                return "CANCELLED"
            op = cls()
            for name, value in props.items():
                if name in cls.properties():
                    setattr(op, name, value)
            result = op.invoke(ctx, event) if invoke else op.execute(ctx)
            if result == "RUNNING_MODAL":
                self._last_modal = op
            return result

        ops.call = tracking_call
        self.addCleanup(setattr, ops, "call", original)

    def test_proportional_edit(self):
        mesh = self.make(primitives.grid(10, 10, 4.0), "栅格", top=True)
        d = self.disp(mesh)
        center = int(np.argmin(np.linalg.norm(d, axis=1)))
        mesh.select_all("DESELECT")
        mesh.vert_sel[center] = True
        mesh.flush_from_verts()
        settings = self.app.tool_settings
        settings.use_proportional_edit = True
        settings.proportional_size = 1.0
        op = object_ops.TransformTranslate()
        self.editor.cursor = self.screen_of((0.0, 0.0, 0.0))
        op.invoke(self.ctx_op, key("G"))
        op.modal(self.ctx_op, key("Z"))
        op.modal(self.ctx_op, key("1"))
        op.modal(self.ctx_op, key("RET"))
        after = self.disp(mesh)
        dist = np.linalg.norm(d[:, :2], axis=1)
        dz = after[:, 2] - d[:, 2]
        self.assertAlmostEqual(float(dz[center]), 1.0, places=6)
        near = (dist > 1e-6) & (dist < 0.99)
        self.assertTrue(near.any())
        self.assertTrue(((dz[near] > 0) & (dz[near] < 1)).all())
        np.testing.assert_allclose(dz[dist > 1.01], 0.0, atol=1e-9)

    def test_knife(self):
        mesh = self.make(primitives.cube(2.0))
        # 前视图：从前面那个面的左边中点切到右边中点
        a = self.screen_of((-1.0, -1.0, 0.0))
        b = self.screen_of((1.0, -1.0, 0.0))
        op = mesh_edit_tools.MeshKnifeTool()
        self.editor.cursor = a
        self.assertEqual(op.invoke(self.ctx_op, click(self.editor, a[0], a[1], "PRESS")), "RUNNING_MODAL")
        op.modal(self.ctx_op, move(self.editor, *b))
        op.modal(self.ctx_op, click(self.editor, b[0], b[1], "PRESS"))
        self.assertEqual(op.modal(self.ctx_op, key("RET")), "FINISHED")
        self.assertEqual((mesh.vert_count, mesh.face_count), (10, 7))
        self.app.history.undo()
        self.assertEqual((mesh.vert_count, mesh.face_count), (8, 6))

    def test_loopcut_bevel_inset(self):
        mesh = self.make(primitives.cube(2.0))
        self.assertEqual(self.call("mesh.loopcut_slide", invoke=False, edge_index=0, number_cuts=2), "FINISHED")
        self.assertEqual((mesh.vert_count, mesh.face_count), (16, 14))
        self.app.history.undo()
        mesh.set_select_mode({"EDGE"})
        mesh.select_all("SELECT")
        self.assertEqual(self.call("mesh.bevel", invoke=False, offset=0.2), "FINISHED")
        self.assertEqual(mesh.face_count, 26)
        self.app.history.undo()
        self.select_face(mesh, self.top_face(mesh))
        self.assertEqual(self.call("mesh.inset", invoke=False, thickness=0.3), "FINISHED")
        self.assertEqual((mesh.vert_count, mesh.face_count), (12, 10))
        self.app.history.undo()
        mesh.set_select_mode({"VERT"})
        mesh.select_all("DESELECT")
        mesh.vert_sel[0] = True
        mesh.flush_from_verts()
        self.assertEqual(self.call("mesh.bevel", invoke=False, offset=0.3, affect="VERTICES"), "FINISHED")
        self.assertEqual((mesh.vert_count, mesh.face_count), (10, 7))

    def test_rip_and_slide(self):
        mesh = self.make(primitives.grid(4, 4, 4.0), "栅格", top=True)
        d = self.disp(mesh)
        # 选中间一行边（x == 0 那条线）撕开
        table = mesh.edges()
        on_line = np.isclose(d[:, 0], 0.0, atol=1e-6)
        mesh.set_select_mode({"VERT"})
        mesh.select_all("DESELECT")
        mesh.vert_sel[on_line] = True
        mesh.flush_from_verts()
        self.assertEqual(int(mesh.edge_sel.sum()), 4)
        self.editor.cursor = self.screen_of((0.5, 0.0, 0.0))
        v0 = mesh.vert_count
        self.assertEqual(self.call("mesh.rip_move", key("V")), "FINISHED")
        translate = self._last_modal
        self.assertIsNotNone(translate)
        translate.modal(self.ctx_op, key("ESC"))
        self.assertEqual(mesh.vert_count, v0 + 5)
        # 跟着鼠标走的是右边（x > 0 那一侧）的那份
        sel = np.nonzero(mesh.vert_sel)[0]
        self.assertEqual(len(sel), 5)
        lf = mesh.loop_face()
        centers = to_display(mesh.face_centers().T).T
        for v in sel:
            faces = np.unique(lf[mesh.loop_vert == v])
            self.assertTrue((centers[faces][:, 0] > 0).all())
        self.app.history.undo()
        self.assertEqual(mesh.vert_count, v0)
        # 滑移边：中间那一列边往一侧滑一半
        mesh.select_all("DESELECT")
        mesh.vert_sel[on_line] = True
        mesh.flush_from_verts()
        before = self.disp(mesh).copy()
        self.editor.cursor = self.screen_of((0.0, 0.0, 0.0))
        op = mesh_edit_tools.TransformEdgeSlide()
        self.assertEqual(op.invoke(self.ctx_op, key("G")), "RUNNING_MODAL")
        for ch in "0.5":
            op.modal(self.ctx_op, key({"0": "0", ".": "PERIOD", "5": "5"}[ch]))
        self.assertEqual(op.modal(self.ctx_op, key("RET")), "FINISHED")
        moved = self.disp(mesh)[on_line] - before[on_line]
        np.testing.assert_allclose(np.abs(moved[:, 0]), 0.5, atol=1e-6)     # 栅格间距 1，滑一半
        self.assertTrue(np.all(np.sign(moved[:, 0]) == np.sign(moved[0, 0])))

    def test_add_primitive_hide_snap_mirror(self):
        mesh = self.make(primitives.cube(2.0))
        self.project.cursor_location = (4.0, 0.0, 0.0)
        self.assertEqual(self.call("mesh.primitive_cube_add"), "FINISHED")
        self.assertEqual((mesh.vert_count, mesh.face_count), (16, 12))
        np.testing.assert_allclose(self.disp(mesh)[mesh.vert_sel].mean(axis=0), [4.0, 0.0, 0.0], atol=1e-5)
        self.assertEqual(int(mesh.vert_sel.sum()), 8)
        # 游标 → 选中项、选中项 → 网格点
        self.project.cursor_location = (0.0, 0.0, 0.0)
        self.call("view3d.snap_cursor_to_selected")
        np.testing.assert_allclose(self.project.cursor_location, [4.0, 0.0, 0.0], atol=1e-5)
        # 隐藏新加的立方体：视口里只画原来那个
        tris_before = len(self.obj.data.positions)
        self.assertEqual(self.call("mesh.hide", unselected=False), "FINISHED")
        self.assertEqual(len(self.obj.data.positions), tris_before // 2)
        self.assertEqual(self.call("mesh.reveal"), "FINISHED")
        self.assertEqual(len(self.obj.data.positions), tris_before)
        # 镜像选择：选 +X 的一个点，镜像到 -X
        mesh.select_all("DESELECT")
        d = self.disp(mesh)
        v = int(np.argmin(np.linalg.norm(d - np.array([1.0, 1.0, 1.0]), axis=1)))
        mesh.vert_sel[v] = True
        mesh.flush_from_verts()
        self.assertEqual(self.call("mesh.select_mirror", invoke=False, axis="X"), "FINISHED")
        picked = np.nonzero(mesh.vert_sel)[0]
        self.assertEqual(len(picked), 1)
        np.testing.assert_allclose(d[picked[0]], [-1.0, 1.0, 1.0], atol=1e-6)

    def test_save_while_editing(self):
        from splender.engine.projectio import load_project

        mesh = self.make(primitives.cube(2.0))
        self.select_face(mesh, self.top_face(mesh))
        self.assertEqual(self.call("mesh.poke", invoke=False), "FINISHED")
        self.assertEqual(mesh.face_count, 9)
        path = str(self.tmp / "edit.splender")
        self.engine.save_project(path)
        # 存盘后视口里还是编辑用的显示数据，接着能编辑
        self.assertIs(self.engine.edit, self.session)
        other = Engine(self.ctx, Preferences(), History())
        self.addCleanup(other.release)
        project, _meta = load_project(other, path)
        loaded = project.objects[0].data
        self.assertEqual(len(loaded.positions), 3 * 14)          # 4 个三角形 + 5 个四边形 × 2
        self.assertEqual(int(np.sum(loaded.polygon_sizes)), 3 * 4 + 4 * 5)


if __name__ == "__main__":
    unittest.main()
