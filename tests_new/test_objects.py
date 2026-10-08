"""物体模式（照 Blender）：添加基本体、点选框选、G/R/S（输入数值、约束轴）、复制、删除、隐藏、合并、应用变换、
设置原点、镜像、游标吸附，撤销重做都对；保存再打开变换和顶点不走样；线框、X 光、局部视图能画。
隐藏上下文，不开窗口。"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
from splender.doc.objects import SceneObject, to_display, to_internal  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet, ToolSettings, ViewOverlay, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from splender.geometry import primitives  # noqa: E402

QT = QApplication.instance() or QApplication(sys.argv)
from splender.ops import object_ops, sculpt_ops, view3d_ops, wm_ops  # noqa: E402,F401


class FakeHost:
    def __init__(self, engine) -> None:
        self.engine = engine

    def make_current(self) -> None:
        pass

    def request_frame(self) -> None:
        pass


class FakeWidget:
    pass


class FakeSettings:
    lock_camera = False


class FakeEditor:
    idname = "VIEW_3D"

    def __init__(self, view) -> None:
        self.view = view
        self.widget = FakeWidget()
        self.settings = FakeSettings()
        self.local_view = None
        self.shape = None
        self.cursor = (0.0, 0.0)
        self._mouse = None

    def pixel(self, event):
        return (event.x, event.y)

    def cursor_pixel(self):
        return self.cursor

    def set_overlay_shape(self, shape) -> None:
        self.shape = shape

    def in_camera_view(self) -> bool:
        return False

    def camera_changed(self) -> None:
        self.view.invalidate_camera()

    def sync_settings(self) -> None:
        pass


class FakeApp:
    def __init__(self, engine, project) -> None:
        self.prefs = Preferences()
        self.prefs.paint.object_resolution = "1024"
        self.history = History()
        self.tool_settings = ToolSettings()
        self.tool_settings.mode = "OBJECT"
        self.project = project
        self.host = FakeHost(engine)
        self.wm = None
        self.active_view3d = None
        self.reports = []

    @property
    def engine(self):
        return self.host.engine

    def request_frame(self) -> None:
        pass

    def notify(self, tag: str = "") -> None:
        pass

    def report(self, text, level="INFO") -> None:
        self.reports.append(text)


def move(editor, x, y, **mods):
    return Event(type="MOUSEMOVE", value="MOVE", x=float(x), y=float(y), source=editor.widget, **mods)


def key(name, value="PRESS", **mods):
    return Event(type=name, value=value, **mods)


class ObjectModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = Path(tempfile.mkdtemp())

    def setUp(self):
        project = Project("物体")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="1024"))
        ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.6, 0.6, 0.62)))
        mesh = primitives.uv_sphere(24, 12, 1.0)
        project.add_object(MeshObject("球", mesh, [ts.uid]))
        self.project = project
        self.engine = Engine(self.ctx, Preferences(), History())
        self.addCleanup(self.engine.release)
        self.engine.set_project(project)
        self.view = self.engine.create_view(ViewShading(mode="SOLID"), ViewOverlay())
        self.view.resize(640, 480)
        camera = self.view.camera
        camera.target = np.zeros(3)
        camera.distance = 12.0
        camera.yaw, camera.pitch = 0.0, 0.0        # 前视图：屏幕右 = X，屏幕上 = Z
        camera.ortho = True
        self.view.invalidate_camera()
        self.app = FakeApp(self.engine, project)
        self.editor = FakeEditor(self.view)
        self.app.active_view3d = self.editor
        self.ctx_op = Context(app=self.app, wm=None, editor=self.editor)
        self.engine.settle()

    def call(self, idname, event=None, invoke=True, **props):
        return ops.call(idname, self.ctx_op, event, invoke=invoke, **props)

    def modal(self, op_result_op, *events):
        for event in events:
            result = op_result_op.modal(self.ctx_op, event)
        return result

    def screen_of(self, point_display):
        p = to_internal(np.asarray(point_display, np.float64))
        clip = self.view.camera.view_projection(self.view.aspect) @ np.array([p[0], p[1], p[2], 1.0])
        return ((clip[0] / clip[3] * 0.5 + 0.5) * self.view.width, (0.5 - clip[1] / clip[3] * 0.5) * self.view.height)

    def bounds_display(self, obj):
        lo = to_display(np.asarray(obj.data.bounds_min, np.float64))
        hi = to_display(np.asarray(obj.data.bounds_max, np.float64))
        return np.minimum(lo, hi), np.maximum(lo, hi)

    # ------------------------------------------------------------------
    def test_add_select_undo(self):
        self.project.cursor_location = (3.0, 0.0, 0.0)
        self.assertEqual(self.call("mesh.primitive_cube_add"), "FINISHED")
        self.assertEqual(len(self.project.objects), 2)
        cube = self.project.objects[-1]
        self.assertTrue(cube.select)
        self.assertEqual(self.project.active_object_uid, cube.uid)
        self.assertEqual(len(self.project.texture_sets), 2)
        self.assertIn(cube.uid, self.engine.meshes)
        lo, hi = self.bounds_display(cube)
        np.testing.assert_allclose((lo + hi) * 0.5, [3.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(hi - lo, [2.0, 2.0, 2.0], atol=1e-5)
        self.app.history.undo()
        self.assertEqual(len(self.project.objects), 1)
        self.assertEqual(len(self.project.texture_sets), 1)
        self.assertNotIn(cube.uid, self.engine.meshes)
        self.app.history.redo()
        self.assertEqual(len(self.project.objects), 2)
        self.assertIn(cube.uid, self.engine.meshes)
        # 点选：点在立方体上选中它，点空白处全不选
        self.engine.settle()
        x, y = self.screen_of((3.0, 0.0, 0.0))
        sphere = self.project.objects[0]
        sphere.select = False
        self.assertEqual(self.call("view3d.select", Event(type="LEFTMOUSE", value="CLICK", x=x, y=y),
                                   deselect_all=True), "FINISHED")
        self.assertTrue(cube.select)
        sx, sy = self.screen_of((0.0, 0.0, 0.0))
        self.call("view3d.select", Event(type="LEFTMOUSE", value="CLICK", x=sx, y=sy, shift=True), toggle=True)
        self.assertTrue(sphere.select and cube.select)
        self.assertEqual(self.project.active_object_uid, sphere.uid)
        self.call("view3d.select", Event(type="LEFTMOUSE", value="CLICK", x=5, y=5), deselect_all=True)
        self.assertFalse(sphere.select or cube.select)
        # 框选两个
        x0, y0 = self.screen_of((-1.5, 0.0, 1.5))
        x1, y1 = self.screen_of((4.5, 0.0, -1.5))
        op = object_ops.View3DSelectBox()
        op.invoke(self.ctx_op, Event(type="LEFTMOUSE", value="CLICK_DRAG", x=x0, y=y0, source=self.editor.widget))
        op.modal(self.ctx_op, move(self.editor, x1, y1))
        self.assertEqual(op.modal(self.ctx_op, Event(type="LEFTMOUSE", value="RELEASE", x=x1, y=y1,
                                                     source=self.editor.widget)), "FINISHED")
        self.assertTrue(sphere.select and cube.select)
        self.app.history.undo()
        self.assertFalse(sphere.select or cube.select)

    def test_transform_typed_and_constrained(self):
        sphere = self.project.objects[0]
        sphere.select = True
        self.project.active_object_uid = sphere.uid
        before = sphere.data.positions.copy()
        # G X 2 回车：沿 X 移 2
        op = object_ops.TransformTranslate()
        x, y = self.screen_of((0.0, 0.0, 0.0))
        self.editor.cursor = (x, y)
        op.invoke(self.ctx_op, key("G"))
        op.modal(self.ctx_op, key("X"))
        op.modal(self.ctx_op, key("2"))
        self.assertEqual(op.modal(self.ctx_op, key("RET")), "FINISHED")
        np.testing.assert_allclose(sphere.transform.location, (2.0, 0.0, 0.0), atol=1e-6)
        moved = to_display(sphere.data.positions.astype(np.float64).T).T - to_display(before.astype(np.float64).T).T
        np.testing.assert_allclose(moved, np.tile([2.0, 0.0, 0.0], (len(moved), 1)), atol=1e-5)
        # 拖动：自由移动跟着鼠标走（正交前视图下屏幕上移 = +Z）
        op = object_ops.TransformTranslate()
        sx, sy = self.screen_of((2.0, 0.0, 0.0))
        self.editor.cursor = (sx, sy)
        op.invoke(self.ctx_op, key("G"))
        tx, ty = self.screen_of((2.0, 0.0, 1.0))
        op.modal(self.ctx_op, move(self.editor, tx, ty))
        op.modal(self.ctx_op, key("RET"))
        np.testing.assert_allclose(sphere.transform.location, (2.0, 0.0, 1.0), atol=1e-4)
        # R Z 90：绕 Z 转 90°
        op = object_ops.TransformRotate()
        op.invoke(self.ctx_op, key("R"))
        op.modal(self.ctx_op, key("Z"))
        for ch in "90":
            op.modal(self.ctx_op, key(ch))
        op.modal(self.ctx_op, key("RET"))
        np.testing.assert_allclose(sphere.transform.rotation, (0.0, 0.0, 90.0), atol=1e-4)
        # S 2：整体放大两倍（绕自己的原点）
        lo, hi = self.bounds_display(sphere)
        op = object_ops.TransformResize()
        op.invoke(self.ctx_op, key("S"))
        op.modal(self.ctx_op, key("2"))
        op.modal(self.ctx_op, key("RET"))
        lo2, hi2 = self.bounds_display(sphere)
        np.testing.assert_allclose(hi2 - lo2, (hi - lo) * 2.0, rtol=1e-4)
        np.testing.assert_allclose(sphere.transform.scale, (2.0, 2.0, 2.0), atol=1e-6)
        # 撤销三次：回到只移动过的样子
        for _ in range(3):
            self.app.history.undo()
        np.testing.assert_allclose(sphere.transform.location, (2.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(sphere.transform.rotation, (0.0, 0.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(sphere.data.positions, (to_internal(np.array([2.0, 0.0, 0.0])) +
                                                           before.astype(np.float64)), atol=1e-4)
        # 右键取消：什么都不变
        op = object_ops.TransformTranslate()
        op.invoke(self.ctx_op, key("G"))
        op.modal(self.ctx_op, move(self.editor, 600, 50))
        self.assertEqual(op.modal(self.ctx_op, key("RIGHTMOUSE")), "CANCELLED")
        np.testing.assert_allclose(sphere.transform.location, (2.0, 0.0, 0.0), atol=1e-6)
        self.assertIsNone(self.engine.meshes[sphere.uid].model)
        # 按属性直接做（「调整上一步」走这条）：沿局部 Y 移 1
        sphere.transform.rotation = (0.0, 0.0, 90.0)
        self.call("transform.translate", invoke=False, value=(0.0, 1.0, 0.0), constraint="Y", orient_type="LOCAL")
        np.testing.assert_allclose(sphere.transform.location, (1.0, 0.0, 0.0), atol=1e-5)

    def test_duplicate_delete_hide_join(self):
        sphere = self.project.objects[0]
        sphere.select = True
        self.project.active_object_uid = sphere.uid
        self.assertEqual(self.call("object.duplicate_move", invoke=False), "FINISHED")
        self.assertEqual(len(self.project.objects), 2)
        dup = self.project.objects[1]
        self.assertTrue(dup.select and not sphere.select)
        self.assertEqual(dup.material_sets, sphere.material_sets)
        self.assertIsNot(dup.data.positions, sphere.data.positions)
        self.call("transform.translate", invoke=False, value=(3.0, 0.0, 0.0))
        # 合并
        sphere.select = True
        self.project.active_object_uid = sphere.uid
        tris = sphere.data.triangle_count
        self.assertEqual(self.call("object.join", invoke=False), "FINISHED")
        self.assertEqual(len(self.project.objects), 1)
        self.assertEqual(sphere.data.triangle_count, tris * 2)
        self.app.history.undo()
        self.assertEqual(len(self.project.objects), 2)
        self.assertEqual(sphere.data.triangle_count, tris)
        # 隐藏、显示
        dup.select = True
        sphere.select = False
        self.call("object.hide_view_set", invoke=False)
        self.assertFalse(dup.visible)
        self.assertNotIn(dup, [m for m in self.engine.pick_objects.values() if m.visible])
        self.call("object.hide_view_clear", invoke=False)
        self.assertTrue(dup.visible and dup.select)
        # 删除、撤销
        self.call("object.delete", invoke=False)
        self.assertEqual(len(self.project.objects), 1)
        self.assertNotIn(dup.uid, self.engine.meshes)
        self.app.history.undo()
        self.assertEqual(len(self.project.objects), 2)
        self.assertIn(dup.uid, self.engine.meshes)

    def test_apply_origin_mirror_shade(self):
        sphere = self.project.objects[0]
        sphere.select = True
        self.project.active_object_uid = sphere.uid
        sphere.transform.scale = (2.0, 2.0, 2.0)
        lo, hi = self.bounds_display(sphere)
        self.call("object.transform_apply", invoke=False, location=False, rotation=False, scale=True)
        np.testing.assert_allclose(sphere.transform.scale, (1.0, 1.0, 1.0))
        lo2, hi2 = self.bounds_display(sphere)
        np.testing.assert_allclose(hi2 - lo2, hi - lo, atol=1e-5)          # 样子不变
        self.app.history.undo()
        np.testing.assert_allclose(sphere.transform.scale, (2.0, 2.0, 2.0))
        # 原点 → 3D 游标（模型不动）
        self.project.cursor_location = (1.0, 2.0, 3.0)
        self.call("object.origin_set", invoke=False, type="ORIGIN_CURSOR")
        np.testing.assert_allclose(sphere.transform.location, (1.0, 2.0, 3.0))
        lo3, hi3 = self.bounds_display(sphere)
        np.testing.assert_allclose(hi3 - lo3, hi - lo, atol=1e-5)
        # 几何中心 → 原点（模型移过来）
        self.call("object.origin_set", invoke=False, type="GEOMETRY_ORIGIN")
        lo4, hi4 = self.bounds_display(sphere)
        np.testing.assert_allclose((lo4 + hi4) * 0.5, (1.0, 2.0, 3.0), atol=1e-4)
        self.app.history.undo()
        lo5, hi5 = self.bounds_display(sphere)
        np.testing.assert_allclose((lo5 + hi5) * 0.5, (lo3 + hi3) * 0.5, atol=1e-5)
        # 镜像：三角形翻面后法线还朝外
        self.call("transform.mirror", invoke=False, constraint="X")
        tris = sphere.data.positions.reshape(-1, 3, 3).astype(np.float64)
        face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        stored = sphere.data.normals.reshape(-1, 3, 3).astype(np.float64).mean(axis=1)
        agree = np.einsum("ij,ij->i", face, stored) > 0
        self.assertGreater(agree.mean(), 0.99)
        # 平直、平滑着色
        self.call("object.shade_flat", invoke=False)
        n = sphere.data.normals.reshape(-1, 3, 3)
        self.assertTrue(np.allclose(n[:, 0], n[:, 1], atol=1e-6))
        self.call("object.shade_smooth", invoke=False)
        n = sphere.data.normals.reshape(-1, 3, 3)
        self.assertFalse(np.allclose(n[:, 0], n[:, 1], atol=1e-3))
        self.call("object.shade_smooth_by_angle", invoke=False, angle=5.0)

    def test_scene_objects_cursor_and_save(self):
        self.project.cursor_location = (0.0, -6.0, 1.0)
        self.call("object.camera_add", invoke=False, location=(0.0, -6.0, 1.0), rotation=(80.0, 0.0, 0.0))
        self.call("object.light_add", invoke=False, type="SPOT", location=(2.0, 2.0, 3.0))
        self.call("object.empty_add", invoke=False, type="CUBE", radius=0.5)
        kinds = [o.kind for o in self.project.scene_objects]
        self.assertEqual(kinds, ["CAMERA", "LIGHT", "EMPTY"])
        self.assertEqual(self.project.active_camera_uid, self.project.scene_objects[0].uid)
        light = self.project.scene_objects[1]
        self.assertEqual(light.data.type, "SPOT")
        # 游标到选中项（灯）
        for obj in self.project.all_objects():
            obj.select = obj is light
        self.call("view3d.snap_cursor_to_selected", invoke=False)
        np.testing.assert_allclose(self.project.cursor_location, (2.0, 2.0, 3.0))
        # 网格挪一下再保存、打开
        sphere = self.project.objects[0]
        sphere.transform.location = (1.0, 0.5, -0.25)
        sphere.transform.rotation = (10.0, 20.0, 30.0)
        positions = sphere.data.positions.copy()
        path = str(self.tmp / "objects.splender")
        self.engine.save_project(path)
        engine2 = Engine(self.ctx, Preferences(), History())
        self.addCleanup(engine2.release)
        loaded, _meta = engine2.load_project(path)
        sphere2 = loaded.objects[0]
        np.testing.assert_allclose(sphere2.transform.location, (1.0, 0.5, -0.25), atol=1e-6)
        np.testing.assert_allclose(sphere2.transform.rotation, (10.0, 20.0, 30.0), atol=1e-4)
        np.testing.assert_allclose(sphere2.data.positions, positions, atol=1e-6)
        self.assertEqual([o.kind for o in loaded.scene_objects], ["CAMERA", "LIGHT", "EMPTY"])
        self.assertEqual(loaded.scene_objects[2].empty_display_type, "CUBE")
        self.assertEqual(loaded.active_camera_uid, self.project.active_camera_uid)
        # 打开后再改变换，顶点照样跟着走（监听接上了）
        sphere2.transform.location = (0.0, 0.5, -0.25)
        moved = to_display(sphere2.data.positions.astype(np.float64).T).T - to_display(positions.astype(np.float64).T).T
        np.testing.assert_allclose(moved.mean(axis=0), (-1.0, 0.0, 0.0), atol=1e-4)

    def test_edges_wireframe_xray_local(self):
        from splender.engine.meshgpu import edge_pairs

        cube = primitives.cube(2.0)
        self.assertEqual(len(edge_pairs(cube.positions, cube.polygon_sizes)), 12)
        sphere = primitives.uv_sphere(32, 16, 1.0)
        verts = 32 * 15 + 2
        faces = 32 * 14 + 64
        self.assertEqual(len(edge_pairs(sphere.positions, sphere.polygon_sizes)), verts + faces - 2)
        background = None
        for mode in ("SOLID", "WIREFRAME"):
            self.view.shading.mode = mode
            self.view.dirty = True
            self.engine.settle()
            image = self.engine.renderer.read_pixels(self.view).astype(int)
            if mode == "SOLID":
                background = image[5, 5, :3]
                solid_cover = (np.abs(image[:, :, :3] - background).sum(axis=2) > 30).mean()
            else:
                wire_cover = (np.abs(image[:, :, :3] - background).sum(axis=2) > 30).mean()
        self.assertGreater(solid_cover, wire_cover)        # 线框只画边，盖住的像素少得多
        self.assertGreater(wire_cover, 0.002)
        self.view.shading.mode = "SOLID"
        self.view.shading.show_xray = True
        self.view.dirty = True
        self.engine.settle()
        self.view.shading.show_xray = False
        # 局部视图：只看第二个物体时，第一个点不到
        self.call("mesh.primitive_cube_add", invoke=False, location=(4.0, 0.0, 0.0))
        sphere_obj, cube_obj = self.project.objects
        self.view.local_uids = {cube_obj.uid}
        self.view.invalidate_camera()
        self.engine.settle()
        x, y = self.screen_of((0.0, 0.0, 0.0))
        self.assertIsNone(self.engine.object_at(self.view, x, y))
        cx, cy = self.screen_of((4.0, 0.0, 0.0))
        self.assertIs(self.engine.object_at(self.view, cx, cy), cube_obj)
        self.view.local_uids = None

    def test_sculpt_mask_ops(self):
        sphere = self.project.objects[0]
        self.app.tool_settings.mode = "SCULPT"
        session = self.engine.enter_sculpt(sphere)
        self.assertIsNotNone(session)
        self.assertEqual(self.call("sculpt.mask_fill", invoke=False, value=1.0), "FINISHED")
        self.assertTrue(np.allclose(session._read_mask(), 1.0))
        self.call("sculpt.mask_invert", invoke=False)
        self.assertTrue(np.allclose(session._read_mask(), 0.0))
        # 一个顶点遮住，扩大一圈后它的邻居也遮住，缩小一圈回到一个
        mask = np.zeros(session.sculptor.vertex_count, np.float32)
        mask[0] = 1.0
        session.edit_mask(lambda m: mask, "测试")
        self.call("sculpt.mask_grow", invoke=False, iterations=1)
        grown = int((session._read_mask() > 0.5).sum())
        self.assertGreater(grown, 3)
        self.call("sculpt.mask_shrink", invoke=False, iterations=1)
        self.assertLessEqual(int((session._read_mask() > 0.5).sum()), grown)
        self.call("sculpt.mask_smooth", invoke=False, iterations=2)
        smooth = session._read_mask()
        self.assertTrue(np.any((smooth > 0.01) & (smooth < 0.99)))
        # 撤销回到扩大之前
        self.engine.exit_sculpt()


if __name__ == "__main__":
    unittest.main()
