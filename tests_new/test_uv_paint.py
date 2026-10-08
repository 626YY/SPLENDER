"""UV 视图里绘制：笔迹落在 UV 上对应的位置，粗细按屏幕上的笔刷大小换算，三维视口里同样看得到；撤销能撤掉。
隐藏上下文，不开窗口。"""
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/engine")
OUT.mkdir(parents=True, exist_ok=True)


class UvPaintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = Path(tempfile.mkdtemp())

    def test_paint_in_uv_view(self):
        project = Project("UV 绘制")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="4096"))
        ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.5, 0.5, 0.5)))
        ts.add_layer(Layer(name="绘制"))
        project.add_object(MeshObject("平面", make_test_mesh("plane"), [ts.uid]))
        history = History()
        engine = Engine(self.ctx, Preferences(), history)
        self.addCleanup(engine.release)
        engine.set_project(project)
        flat = engine.create_view2d()
        flat.resize(800, 800)
        flat.fit()
        view = engine.create_view(ViewShading(mode="MATERIAL"))
        view.resize(640, 640)
        camera = view.camera
        camera.target = np.zeros(3)
        camera.distance = 2.4
        camera.yaw, camera.pitch = 0.0, 89.0
        view.invalidate_camera()
        engine.settle()
        before3d = engine.renderer.read_pixels(view).copy()
        tools = ToolSettings()
        tools.brush.size = 40
        tools.brush.hardness = 1.0
        tools.brush.color = (0.95, 0.1, 0.05)
        y = 300.0
        ok = engine.stroke_begin(flat, 150, y, 1.0, time.perf_counter(), tools)
        self.assertTrue(ok, "UV 视图里应该能落笔：%r" % engine.stroke_reject)
        for i in range(1, 51):
            engine.stroke_move(150 + i * 10, y, 1.0, time.perf_counter())
            engine.frame()
        engine.stroke_end()
        engine.settle()
        self.assertTrue(history.can_undo)
        out = self.tmp / "basecolor.png"
        engine.export_channel(ts, "basecolor", str(out))
        color = np.asarray(Image.open(out)).astype(int)
        red = (color[:, :, 0] - color[:, :, 1]) > 100
        Image.fromarray(color.astype(np.uint8)[::8, ::8]).save(OUT / "uv_paint_basecolor.png")
        rows = np.nonzero(red.any(axis=1))[0]
        cols = np.nonzero(red.any(axis=0))[0]
        self.assertGreater(len(rows), 0, "应该画上了")
        size = color.shape[0]
        # 屏幕位置 → UV → 图片行列（图片第一行是 v=1）
        uv_start = flat.pixel_to_uv(150, y)
        uv_end = flat.pixel_to_uv(650, y)
        expect_row = (1.0 - uv_start[1]) * size
        expect_width = 40.0 / flat.zoom * size
        print()
        print("UV 绘制：笔迹行 %d-%d（应以 %.0f 为中心，宽约 %.0f），列 %d-%d（应为 %.0f-%.0f）" % (
            rows.min(), rows.max(), expect_row, expect_width, cols.min(), cols.max(),
            (uv_start[0] - 20 / flat.zoom) * size, (uv_end[0] + 20 / flat.zoom) * size))
        self.assertAlmostEqual((rows.min() + rows.max()) * 0.5, expect_row, delta=size * 0.004)
        self.assertAlmostEqual(rows.max() - rows.min(), expect_width, delta=expect_width * 0.1)
        self.assertAlmostEqual(cols.min(), (uv_start[0] - 20 / flat.zoom) * size, delta=size * 0.006)
        self.assertAlmostEqual(cols.max(), (uv_end[0] + 20 / flat.zoom) * size, delta=size * 0.006)
        # 三维视口里也看得到
        after3d = engine.renderer.read_pixels(view).astype(int)
        changed = np.abs(after3d - before3d.astype(int)).max(axis=2) > 30
        Image.fromarray(after3d[:, :, :3].astype(np.uint8)).save(OUT / "uv_paint_view3d.png")
        print("三维视口里变了的像素：%d" % changed.sum())
        self.assertGreater(changed.sum(), 2000)
        # 撤销
        history.undo()
        engine.settle()
        undone = engine.renderer.read_pixels(view).astype(int)
        self.assertLess(np.abs(undone - before3d.astype(int)).max(), 4, "撤销后回到原样")


if __name__ == "__main__":
    unittest.main(verbosity=2)
