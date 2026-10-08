"""纹理集改分辨率：画过的内容按新尺寸重新采样（缩小、放大、到 256），显示贴图和分块几何跟着换；
撤销回到原来的像素；改完还能接着画。隐藏上下文，不开窗口。"""
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parent))     # 同目录的 test_objects（隔离模式下脚本目录不在路径里）
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet, ToolSettings, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402


class ResizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = Path(tempfile.mkdtemp())

    def setUp(self):
        project = Project("改分辨率")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="2048"))
        ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.5, 0.5, 0.52)))
        ts.add_layer(Layer(name="绘制 1"))
        mesh = make_test_mesh("plane")
        project.add_object(MeshObject(mesh.name, mesh, [ts.uid]))
        self.project, self.ts = project, ts
        self.engine = Engine(self.ctx, Preferences(), History())
        self.addCleanup(self.engine.release)
        self.engine.set_project(project)
        view = self.engine.create_view(ViewShading())
        view.resize(640, 640)
        camera = view.camera
        camera.target = np.zeros(3)
        camera.distance = 2.4
        camera.yaw, camera.pitch = 0.0, 89.0
        view.invalidate_camera()
        self.view = view
        self.engine.settle()

    def paint(self, y_offset=0.0, color=(0.9, 0.15, 0.1)):
        tools = ToolSettings()
        tools.brush.size = 70
        tools.brush.color = color
        tools.brush.use_height = True
        tools.brush.height = 0.1
        view = self.view
        cx, cy = view.width / 2, view.height / 2 + y_offset
        self.assertTrue(self.engine.stroke_begin(view, cx - 120, cy, 1.0, time.perf_counter(), tools))
        for i in range(1, 41):
            t = i / 40
            self.engine.stroke_move(cx - 120 + 240 * t, cy + 40 * np.sin(t * 6.283), 1.0, time.perf_counter())
            self.engine.frame()
        self.engine.stroke_end()
        self.engine.settle()

    def export(self, name="base.png", channel="basecolor") -> np.ndarray:
        out = self.tmp / name
        self.engine.export_channel(self.ts, channel, str(out), size=512)
        return np.asarray(Image.open(out)).astype(np.float64) / 255.0

    def resize(self, size: int):
        pairs = self.engine.resize_texture_set(self.ts, size)
        old = self.ts.resolution
        self.ts.resolution = str(size)
        self.engine.apply_texture_set_size(self.ts, pairs, True)
        self.engine.settle()
        return pairs, old

    def test_down_up_and_undo(self):
        self.paint()
        before = self.export("before.png")
        self.assertGreater(before[:, :, 0].max() - before[:, :, 0].min(), 0.2, "画上去了")
        state = self.engine.sets[self.ts.uid]
        # 缩小到 1K：导出到 512 和原来几乎一样
        pairs, old = self.resize(1024)
        self.assertEqual(state.display.size, 1024)
        self.assertTrue(state.set_gpus, "分块几何要按新尺寸重建")
        down = self.export("down.png")
        diff = np.abs(down - before)
        print("\n2K→1K：导出差 平均 %.4f 最大 %.3f" % (diff.mean(), diff.max()))
        self.assertLess(diff.mean(), 0.01)
        # 再放大到 4K：比原来略糊，但整体一样
        pairs2, old2 = self.resize(4096)
        up = self.export("up.png")
        diff = np.abs(up - before)
        print("1K→4K：导出差 平均 %.4f 最大 %.3f" % (diff.mean(), diff.max()))
        self.assertLess(diff.mean(), 0.02)
        # 高度通道也跟着搬了
        height = self.export("height.png", "height")
        self.assertGreater(height.max() - height.min(), 0.02)
        # 改完还能接着画（按新尺寸）
        self.paint(y_offset=90.0, color=(0.1, 0.3, 0.9))
        painted = self.export("painted.png")
        self.assertGreater(painted[:, :, 2].max(), 0.6, "新画的蓝色要在")
        # 撤销两次（先撤掉第二次改尺寸）：回到 2K 和原来的像素
        self.ts.resolution = old2
        self.engine.apply_texture_set_size(self.ts, pairs2, False)
        self.ts.resolution = old
        self.engine.apply_texture_set_size(self.ts, pairs, False)
        self.engine.settle()
        self.assertEqual(state.display.size, 2048)
        back = self.export("back.png")
        self.assertLess(np.abs(back - before).max(), 0.02)

    def test_tiny_and_operator(self):
        from splender.core import ops
        from splender.core.context import Context
        from splender.ops import texture_set_ops  # noqa: F401
        from test_objects import FakeApp

        self.paint()
        before = self.export("before2.png")
        app = FakeApp(self.engine, self.project)
        self.engine.history = app.history
        ctx = Context(app=app, wm=None, editor=None)
        self.assertEqual(ops.call("texture_set.resize", ctx, invoke=False, resolution="256"), "FINISHED")
        self.engine.settle()
        self.assertEqual(self.ts.size, 256)
        self.assertEqual(self.engine.sets[self.ts.uid].display.size, 256)
        tiny = self.export("tiny.png")
        # 256 只剩大块颜色：红色区域还在
        self.assertGreater(tiny[:, :, 0].max() - tiny[:, :, 0].min(), 0.1)
        app.history.undo()
        self.engine.settle()
        self.assertEqual(self.ts.size, 2048)
        np.testing.assert_allclose(self.export("undo.png"), before, atol=0.02)
        app.history.redo()
        self.engine.settle()
        self.assertEqual(self.ts.size, 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
