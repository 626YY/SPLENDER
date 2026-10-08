"""效果笔刷（照 Photoshop）：减淡、加深、海绵、模糊、锐化、涂抹、仿制图章、修复画笔。在 UV 视图里划一笔，
只改划过的地方、方向对、一笔记一步撤销、取消能放回原样。隐藏上下文，不开窗口。"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.doc.project import PLANE_COLOR, Layer, MeshObject, Project, TextureSet, ToolSettings  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from splender.engine.filters import FilterJob  # noqa: E402
from splender.engine.pagepool import PAGE  # noqa: E402

RES = 1024
GRID = RES // PAGE


class EffectBrushTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        project = Project("效果笔刷")
        cls.ts = project.add_texture_set(TextureSet(name="材质", resolution=str(RES)))
        cls.layer = Layer(name="画")
        cls.ts.add_layer(cls.layer)
        project.add_object(MeshObject("平面", make_test_mesh("plane"), [cls.ts.uid]))
        cls.history = History()
        cls.engine = Engine(cls.ctx, Preferences(), cls.history)
        cls.engine.set_project(project)
        cls.flat = cls.engine.create_view2d()
        cls.flat.resize(800, 800)
        cls.flat.center = np.array([0.5, 0.5])
        cls.flat.zoom = 800.0                     # 整张贴图正好铺满视图：一个视图像素是 1.28 个纹素
        cls.engine.settle()

    @classmethod
    def tearDownClass(cls):
        cls.engine.release()

    # ---- 图层像素 ----
    def write(self, image: np.ndarray) -> None:
        """(RES, RES, 4) 的 0..1 颜色（含覆盖度）写成图层的颜色页。"""
        engine = self.engine
        engine.layers.drop_layer(self.layer.uid)
        data = np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)
        cells = [(x, y) for y in range(GRID) for x in range(GRID)]
        news = engine.cache.try_new_pages("rgba8", len(cells), block=True)
        for (x, y), page in zip(cells, news):
            engine.cache.pools["rgba8"].write(page.slot, np.ascontiguousarray(
                data[y * PAGE:(y + 1) * PAGE, x * PAGE:(x + 1) * PAGE]).tobytes())
        engine.cache.note_new_pages(news)
        job = FilterJob(engine.layers, self.layer.uid, GRID, ["rgba8"])
        job.add(PLANE_COLOR, [c[0] for c in cells], [c[1] for c in cells], news)
        job.finish()
        self.history.clear() if hasattr(self.history, "clear") else None

    def read(self) -> np.ndarray:
        out = np.zeros((RES, RES, 4), np.float32)
        store = self.engine.layers.stores[self.layer.uid]
        for (x, y), page in store.planes[PLANE_COLOR].levels[0].pages.items():
            raw = np.frombuffer(self.engine.cache.page_bytes(page), np.uint8).reshape(PAGE, PAGE, 4)
            out[y * PAGE:(y + 1) * PAGE, x * PAGE:(x + 1) * PAGE] = raw / 255.0
        return out

    @staticmethod
    def solid(rgb) -> np.ndarray:
        image = np.ones((RES, RES, 4), np.float32)
        image[..., :3] = rgb
        return image

    # ---- 划一笔 ----
    def tools(self, tool: str, **effect) -> ToolSettings:
        tools = ToolSettings()
        tools.tool = tool
        tools.brush.size = 60
        tools.brush.hardness = 1.0
        tools.brush.flow = 1.0
        tools.brush.spacing = 0.1
        tools.brush.use_metallic = tools.brush.use_roughness = tools.brush.use_height = False
        tools.effect.strength = 1.0
        for name, value in effect.items():
            setattr(tools.effect, name, value)
        return tools

    def stroke(self, tools, x0=200.0, x1=600.0, y=400.0, invert=False, cancel=False) -> bool:
        engine = self.engine
        ok = engine.stroke_begin(self.flat, x0, y, 1.0, time.perf_counter(), tools, invert)
        if not ok:
            return False
        for i in range(1, 21):
            engine.stroke_move(x0 + (x1 - x0) * i / 20, y, 1.0, time.perf_counter())
            engine.frame()
        if cancel:
            engine.stroke_cancel()
        else:
            engine.stroke_end()
        engine.settle()
        return True

    @staticmethod
    def row(y_view=400.0) -> int:
        """视图里的一行对应贴图的第几行（v 往上）。"""
        return int((0.5 + (400.0 - y_view) / 800.0) * RES)

    def texel(self, image, x_view, y_view=400.0):
        u = 0.5 + (x_view - 400.0) / 800.0
        v = 0.5 + (400.0 - y_view) / 800.0
        return image[int(v * RES), int(u * RES)]

    # ---- 测 ----
    def test_dodge_burn_and_invert(self):
        self.write(self.solid((0.5, 0.5, 0.5)))
        steps = self.history.index
        self.assertTrue(self.stroke(self.tools("paint.dodge")), self.engine.stroke_reject)
        self.assertEqual(self.history.index, steps + 1, "一笔一步撤销")
        self.assertEqual(self.history.steps[steps].label, "减淡")
        out = self.read()
        self.assertGreater(float(self.texel(out, 400)[0]), 0.56, "划过的地方变亮")
        self.assertAlmostEqual(float(self.texel(out, 400, 150)[0]), 0.5, delta=1.5 / 255, msg="别处不动")
        self.history.undo()
        self.assertAlmostEqual(float(self.texel(self.read(), 400)[0]), 0.5, delta=1.5 / 255, msg="撤销回到原样")
        self.assertTrue(self.stroke(self.tools("paint.dodge"), invert=True))
        self.assertLess(float(self.texel(self.read(), 400)[0]), 0.44, "按住 Ctrl 变成加深")
        self.write(self.solid((0.5, 0.5, 0.5)))
        self.assertTrue(self.stroke(self.tools("paint.burn")))
        self.assertLess(float(self.texel(self.read(), 400)[0]), 0.44)

    def test_sponge(self):
        self.write(self.solid((0.8, 0.3, 0.2)))
        self.assertTrue(self.stroke(self.tools("paint.sponge", sponge_mode="DESATURATE")))
        c = self.texel(self.read(), 400)
        self.assertLess(float(c[0] - c[2]), 0.55, "颜色变灰")
        self.write(self.solid((0.6, 0.4, 0.35)))
        self.assertTrue(self.stroke(self.tools("paint.sponge", sponge_mode="SATURATE")))
        c = self.texel(self.read(), 400)
        self.assertGreater(float(c[0] - c[2]), 0.3, "颜色变艳")

    def test_blur_and_sharpen(self):
        checker = np.zeros((RES, RES, 4), np.float32)
        yy, xx = np.mgrid[0:RES, 0:RES]
        checker[..., :3] = (((xx // 4) + (yy // 4)) % 2)[..., None] * 1.0
        checker[..., 3] = 1.0
        self.write(checker)
        self.assertTrue(self.stroke(self.tools("paint.blur", blur_radius=4.0)))
        out = self.read()
        r = self.row()
        center = out[r - 8:r + 8, 400:600, 0]
        away = out[100:116, 400:600, 0]
        self.assertLess(float(center.std()), float(away.std()) * 0.5, "划过的地方糊了")
        self.assertAlmostEqual(float(away.std()), float(checker[100:116, 400:600, 0].std()), delta=0.01)
        soft = self.solid((0.0, 0.0, 0.0))
        soft[..., :3] = np.clip((xx - 506) / 12.0, 0, 1)[..., None]
        self.write(soft)
        self.assertTrue(self.stroke(self.tools("paint.sharpen", sharpen_amount=2.0)))
        out = self.read()
        edge = out[r, 490:530, 0]
        self.assertLess(float(edge.min()), 0.0 + 2.0 / 255 + 1e-6)
        self.assertGreater(float(np.abs(np.diff(edge)).max()), float(np.abs(np.diff(soft[r, 490:530, 0])).max()) + 0.02,
                           "边缘更陡")

    def test_smudge_drags_color(self):
        image = self.solid((0.0, 0.0, 1.0))
        image[:, :RES // 2, :3] = (1.0, 0.0, 0.0)
        self.write(image)
        self.assertTrue(self.stroke(self.tools("paint.smudge", strength=0.9), x0=300.0, x1=520.0))
        out = self.read()
        r = self.row()
        right = out[r, RES // 2 + 20:RES // 2 + 80]
        self.assertGreater(float(right[:, 0].max()), 0.3, "红色被拖进了蓝的那边")
        self.assertLess(float(out[r - 120, RES // 2 + 40, 0]), 0.02, "没划到的地方不动")

    def test_clone_and_heal(self):
        image = self.solid((0.0, 0.0, 1.0))
        image[:RES // 4, :, :3] = (1.0, 0.0, 0.0)          # 底下一条红的（v 小）
        self.write(image)
        tools = self.tools("paint.clone")
        self.assertFalse(self.stroke(tools), "没有来源时不能仿制")
        self.assertIn("Alt", self.engine.stroke_reject)
        tools.effect.has_source = True
        tools.effect.source_u, tools.effect.source_v = 0.25, 0.1        # 红条里
        self.assertTrue(self.stroke(tools, x0=200.0, x1=600.0))          # 从 (0.25, 0.5) 起划
        out = self.read()
        r = self.row()
        self.assertGreater(float(out[r, int(0.4 * RES), 0]), 0.9, "照搬来了红色")
        self.assertLess(float(out[r + 100, int(0.4 * RES), 0]), 0.02, "没划到的地方还是蓝")
        # 修复画笔：来源的纹理，颜色跟着落笔处（蓝）
        rng = np.random.default_rng(1)
        image = self.solid((0.0, 0.0, 0.8))
        noise = rng.uniform(-0.15, 0.15, (RES // 4, RES))
        image[:RES // 4, :, :3] = np.clip(0.5 + noise[..., None], 0, 1)     # 灰色带纹理
        self.write(image)
        tools = self.tools("paint.heal", blur_radius=4.0)
        tools.effect.has_source = True
        tools.effect.source_u, tools.effect.source_v = 0.25, 0.12
        self.assertTrue(self.stroke(tools, x0=200.0, x1=600.0))
        out = self.read()
        patch = out[r - 6:r + 6, int(0.35 * RES):int(0.6 * RES), :3]
        self.assertGreater(float(patch[..., 2].mean()), 0.6, "颜色跟着落笔处（还是偏蓝）")
        self.assertGreater(float(patch[..., 0].std()), 0.03, "带上了来源的纹理")

    def test_cancel_restores(self):
        self.write(self.solid((0.5, 0.5, 0.5)))
        steps = self.history.index
        self.assertTrue(self.stroke(self.tools("paint.dodge"), cancel=True))
        self.assertEqual(self.history.index, steps, "取消不记撤销")
        self.assertAlmostEqual(float(self.texel(self.read(), 400)[0]), 0.5, delta=1.5 / 255)


if __name__ == "__main__":
    unittest.main(verbosity=2)
