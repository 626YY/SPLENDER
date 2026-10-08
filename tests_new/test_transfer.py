"""贴图搬运：画几笔 → 换一套完全不同的 UV → 把图层像素搬过去 → 从同一个角度看，画面应该几乎不变；
撤销后换回原来的像素。隐藏上下文，不开窗口。截图存到 docs/evidence/transfer/。"""
import sys
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
from splender.geometry.unwrap import smart_unwrap  # noqa: E402
from splender.sculpt.topology import weld  # noqa: E402

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/transfer")
OUT.mkdir(parents=True, exist_ok=True)


def paint(engine, view, tools, points):
    assert engine.stroke_begin(view, points[0][0], points[0][1], 1.0, time.perf_counter(), tools)
    for x, y in points[1:]:
        engine.stroke_move(x, y, 1.0, time.perf_counter())
        engine.frame()
    engine.stroke_end()
    engine.settle()


class TransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)

    def test_relayout_keeps_appearance(self):
        project = Project("搬运")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="4096"))
        ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.5, 0.5, 0.52), fill_roughness=0.5))
        ts.add_layer(Layer(name="绘制"))
        mesh = make_test_mesh("sphere", segments=128, rings=64)
        obj = project.add_object(MeshObject(mesh.name, mesh, [ts.uid]))
        history = History()
        engine = Engine(self.ctx, Preferences(), history)
        self.addCleanup(engine.release)
        engine.set_project(project)
        view = engine.create_view(ViewShading())
        view.resize(900, 700)
        engine.frame_view(view)
        engine.settle()
        tools = ToolSettings()
        tools.brush.size = 70
        tools.brush.use_height = True
        tools.brush.height = 0.06
        cx, cy = view.width / 2, view.height / 2
        for color, offset in (((0.9, 0.2, 0.1), -80), ((0.1, 0.5, 0.9), 0), ((0.95, 0.8, 0.2), 80)):
            tools.brush.color = color
            paint(engine, view, tools, [(cx - 120 + 240 * t, cy + offset + 30 * np.sin(t * 9.0))
                                        for t in np.linspace(0, 1, 40)])
        before = engine.renderer.read_pixels(view)[:, :, :3].astype(np.float32)
        Image.fromarray(before.astype(np.uint8)).save(OUT / "01_before.png")

        # 换一套完全不同的 UV（自动展开）
        old = {"positions": mesh.positions.copy(), "uvs": mesh.uvs.copy(), "material_ids": mesh.material_ids.copy()}
        welded = weld(mesh.positions, None, mesh.material_ids, drop_degenerate=False)
        uvs, _stats = smart_unwrap(welded.vertices, welded.triangles, welded.material_ids, resolution=4096,
                                   margin_texels=24)
        mesh.uvs = uvs
        engine.rebuild_object(obj)
        engine.settle()
        scrambled = engine.renderer.read_pixels(view)[:, :, :3].astype(np.float32)
        Image.fromarray(scrambled.astype(np.uint8)).save(OUT / "02_new_uv_without_transfer.png")
        t0 = time.perf_counter()
        swaps = engine.transfer_layers(obj, old)
        seconds = time.perf_counter() - t0
        engine.settle()
        after = engine.renderer.read_pixels(view)[:, :, :3].astype(np.float32)
        Image.fromarray(after.astype(np.uint8)).save(OUT / "03_after_transfer.png")
        covered = before.sum(axis=2) > 120
        diff_after = np.abs(after - before)[covered].mean()
        diff_scrambled = np.abs(scrambled - before)[covered].mean()
        p99 = np.percentile(np.abs(after - before)[covered].max(axis=1), 99)
        print("\n搬运 %.2f 秒，%s；搬之前和原来差 %.2f，搬之后差 %.2f（99%% 分位 %.1f）" % (
            seconds, [(t.name, len(p)) for t, p in swaps], diff_scrambled, diff_after, p99))
        self.assertTrue(swaps, "应该有图层被搬运")
        self.assertGreater(diff_scrambled, 2.0, "换 UV 不搬运时画面应该乱掉")
        self.assertLess(diff_after, 2.0, "搬运后应该和原来几乎一样")
        self.assertLess(p99, 40.0)

        # 撤销：换回旧像素和旧 UV
        engine.swap_layer_stores(swaps, use_new=False)
        mesh.uvs = old["uvs"]
        engine.rebuild_object(obj)
        engine.settle()
        restored = engine.renderer.read_pixels(view)[:, :, :3].astype(np.float32)
        self.assertLess(np.abs(restored - before)[covered].mean(), 0.5, "撤销应完全回到原来")
        # 重做
        mesh.uvs = uvs
        engine.rebuild_object(obj)
        engine.swap_layer_stores(swaps, use_new=True)
        engine.settle()
        again = engine.renderer.read_pixels(view)[:, :, :3].astype(np.float32)
        self.assertLess(np.abs(again - after)[covered].mean(), 0.5, "重做应回到搬运后的样子")
        engine.free_layer_stores(swaps, keep_new=True)

    def test_mask_default_kept(self):
        """白色蒙版的图层：蒙版没画过的地方搬过去后仍是白的（缩小看也不出黑块）。"""
        project = Project("蒙版")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="2048"))
        ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.2, 0.2, 0.2)))
        top = ts.add_layer(Layer(name="填充", kind="FILL", fill_color=(0.9, 0.9, 0.9), has_mask=True, mask_default=1.0))
        mesh = make_test_mesh("sphere", segments=96, rings=48)
        obj = project.add_object(MeshObject(mesh.name, mesh, [ts.uid]))
        engine = Engine(self.ctx, Preferences(), History())
        self.addCleanup(engine.release)
        engine.set_project(project)
        view = engine.create_view(ViewShading())
        view.resize(700, 600)
        engine.frame_view(view)
        engine.settle()
        tools = ToolSettings()
        tools.brush.size = 80
        tools.brush.mask_value = 0.0
        ts.paint_target = "MASK"
        cx, cy = view.width / 2, view.height / 2
        paint(engine, view, tools, [(cx - 120 + 240 * t, cy) for t in np.linspace(0, 1, 30)])
        before = engine.renderer.read_pixels(view)[:, :, :3].astype(np.float32)
        view.camera.distance *= 6.0
        view.invalidate_camera()
        engine.settle()
        far_before = engine.renderer.read_pixels(view)[:, :, :3].astype(np.float32)
        view.camera.distance /= 6.0
        view.invalidate_camera()
        engine.settle()
        old = {"positions": mesh.positions.copy(), "uvs": mesh.uvs.copy(), "material_ids": mesh.material_ids.copy()}
        welded = weld(mesh.positions, None, mesh.material_ids, drop_degenerate=False)
        mesh.uvs, _stats = smart_unwrap(welded.vertices, welded.triangles, welded.material_ids, resolution=2048,
                                        margin_texels=16)
        engine.rebuild_object(obj)
        swaps = engine.transfer_layers(obj, old)
        engine.settle()
        after = engine.renderer.read_pixels(view)[:, :, :3].astype(np.float32)
        Image.fromarray(after.astype(np.uint8)).save(OUT / "04_mask_after_transfer.png")
        covered = before.sum(axis=2) > 60
        print("\n蒙版搬运：差 %.2f" % np.abs(after - before)[covered].mean())
        self.assertLess(np.abs(after - before)[covered].mean(), 2.5)
        # 缩小看（远处，用到上面几级）也不能出黑块
        view.camera.distance *= 6.0
        view.invalidate_camera()
        engine.settle()
        far = engine.renderer.read_pixels(view)[:, :, :3].astype(np.float32)
        Image.fromarray(far.astype(np.uint8)).save(OUT / "05_mask_far_after.png")
        Image.fromarray(far_before.astype(np.uint8)).save(OUT / "05_mask_far_before.png")
        lit = far_before.sum(axis=2) > 60
        print("远处看：差 %.2f" % np.abs(far - far_before)[lit].mean())
        self.assertLess(np.abs(far - far_before)[lit].mean(), 4.0)
        self.assertTrue(top.has_mask)
        engine.free_layer_stores(swaps, keep_new=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
