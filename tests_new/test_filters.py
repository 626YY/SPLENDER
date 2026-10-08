"""滤镜（engine/filters.py）：直接在图层的页上做模糊、锐化、杂色……结果和预期的数学性质一致，跨页连续，可以撤销。
隐藏上下文、512² 贴图，图层内容直接写进页里。不开窗口。"""
import sys
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
from splender.doc.project import (PLANE_COLOR, PLANE_FORMAT, PLANE_MASK, PLANE_MR, Layer, MeshObject, Project,  # noqa: E402
                                  TextureSet)
from splender.engine.engine import Engine  # noqa: E402
from splender.engine.filters import FilterJob  # noqa: E402
from splender.engine.pagepool import PAGE  # noqa: E402

RES = 512
GRID = RES // PAGE
CHANNELS = {"rgba8": 4, "rg16f": 2, "r8": 1}
DTYPES = {"rgba8": np.uint8, "rg16f": np.float16, "r8": np.uint8}


class FilterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        project = Project("滤镜")
        cls.ts = project.add_texture_set(TextureSet(name="材质", resolution=str(RES)))
        cls.layer = Layer(name="画", kind="PAINT")
        cls.ts.layers.append(cls.layer)
        cls.ts._watch(cls.layer)
        mesh = make_test_mesh("plane")
        project.add_object(MeshObject(mesh.name, mesh, [cls.ts.uid]))
        cls.history = History()
        cls.engine = Engine(cls.ctx, Preferences(), cls.history)
        cls.engine.set_project(project)
        cls.engine.settle()

    @classmethod
    def tearDownClass(cls):
        cls.engine.release()

    def setUp(self):
        self.engine.layers.drop_layer(self.layer.uid)
        self.ts.paint_target = "CONTENT"
        self.layer.has_mask = False

    # ---- 读写图层像素 ----
    def write(self, plane: int, image: np.ndarray) -> None:
        """把 (RES, RES, 通道) 的数组写成这一层的页（全空的格不写），再逐级生成多级。"""
        engine = self.engine
        fmt = PLANE_FORMAT[plane]
        image = np.ascontiguousarray(image.astype(DTYPES[fmt]))
        cells = [(x, y) for y in range(GRID) for x in range(GRID)
                 if np.any(image[y * PAGE:(y + 1) * PAGE, x * PAGE:(x + 1) * PAGE])]
        cache = engine.cache
        news = cache.try_new_pages(fmt, len(cells), block=True)
        for (x, y), page in zip(cells, news):
            cache.pools[fmt].write(page.slot, image[y * PAGE:(y + 1) * PAGE, x * PAGE:(x + 1) * PAGE].tobytes())
        cache.note_new_pages(news)
        job = FilterJob(engine.layers, self.layer.uid, GRID, [fmt])
        job.add(plane, [c[0] for c in cells], [c[1] for c in cells], news)
        job.finish()

    def read(self, plane: int) -> np.ndarray:
        """这一层第 0 级的像素 (RES, RES, 通道)，浮点 0..1（高度原样）。"""
        fmt = PLANE_FORMAT[plane]
        channels = CHANNELS[fmt]
        out = np.zeros((RES, RES, channels), np.float32)
        store = self.engine.layers.stores.get(self.layer.uid)
        if store is None or plane not in store.planes:
            return out
        for (x, y), page in store.planes[plane].levels[0].pages.items():
            raw = np.frombuffer(self.engine.cache.page_bytes(page), DTYPES[fmt]).reshape(PAGE, PAGE, channels)
            out[y * PAGE:(y + 1) * PAGE, x * PAGE:(x + 1) * PAGE] = raw.astype(np.float32) / (
                255.0 if fmt != "rg16f" else 1.0)
        return out

    def run_filter(self, name: str, planes=None, **params) -> int:
        planes = planes or {PLANE_COLOR: (True,)}
        return self.engine.apply_filter(self.ts, self.layer, name, params, planes, name)

    @staticmethod
    def color(rgb, coverage) -> np.ndarray:
        """颜色页的原始值：rgb（0..1）+ 覆盖度，转成 0..255。"""
        img = np.zeros((RES, RES, 4), np.float32)
        img[..., :3] = rgb
        img[..., 3] = coverage
        img[..., :3] *= (coverage > 0)[..., None]
        return np.rint(np.clip(img, 0, 1) * 255.0)

    # ---- 模糊 ----
    def test_gaussian_blur_small_and_large(self):
        """竖线（正好跨在两页的分界上）模糊后每行的覆盖度总和不变、宽度合乎半径；大半径在粗一级算也对。"""
        for sigma, tol in ((4.0, 0.08), (40.0, 0.15)):
            coverage = np.zeros((RES, RES), np.float32)
            coverage[:, 255:257] = 1.0
            self.write(PLANE_COLOR, self.color((1.0, 0.2, 0.1), coverage))
            pages = self.run_filter("GAUSSIAN_BLUR", radius=sigma)
            self.assertGreater(pages, 0)
            out = self.read(PLANE_COLOR)
            row = out[100, :, 3]
            self.assertAlmostEqual(float(row.sum()), 2.0, delta=0.2, msg="σ=%s 覆盖度总和" % sigma)
            xs = np.arange(RES) + 0.5
            mean = float((row * xs).sum() / row.sum())
            std = float(np.sqrt((row * (xs - mean) ** 2).sum() / row.sum()))
            self.assertAlmostEqual(mean, 256.0, delta=1.0)
            self.assertAlmostEqual(std, float(np.hypot(sigma, 0.58)), delta=sigma * tol, msg="σ=%s 宽度" % sigma)
            # 颜色还是那个颜色（预乘后再除回来）
            covered = out[100, :, 3] > 0.05
            self.assertTrue(np.allclose(out[100, covered, 0], 1.0, atol=0.03))
            # 上下各行一样（竖线）
            self.assertTrue(np.allclose(out[50, :, 3], out[400, :, 3], atol=0.01))

    def test_motion_blur(self):
        coverage = np.zeros((RES, RES), np.float32)
        coverage[300:302, 120:122] = 1.0
        self.write(PLANE_COLOR, self.color((0.2, 0.9, 0.3), coverage))
        self.run_filter("MOTION_BLUR", angle=0.0, distance=60.0)
        out = self.read(PLANE_COLOR)[..., 3]
        rows = np.nonzero(out.sum(axis=1) > 0.05)[0]
        cols = np.nonzero(out.sum(axis=0) > 0.01)[0]
        self.assertLessEqual(len(rows), 4, "只往横向拖")
        self.assertGreater(cols.max() - cols.min(), 50)
        self.assertAlmostEqual(float(out.sum()), 4.0, delta=0.4)

    def test_radial_spin(self):
        """绕中心旋转：离中心一段距离的点变成一段圆弧，到中心的距离不变。"""
        coverage = np.zeros((RES, RES), np.float32)
        coverage[255:257, 400:402] = 1.0        # 中心 (256, 256) 右边 145 处
        self.write(PLANE_COLOR, self.color((1.0, 1.0, 1.0), coverage))
        self.run_filter("RADIAL_BLUR", mode="SPIN", amount=30.0, center_u=0.5, center_v=0.5)
        out = self.read(PLANE_COLOR)[..., 3]
        ys, xs = np.nonzero(out > 0.002)
        radius = np.hypot(xs + 0.5 - 256.0, ys + 0.5 - 256.0)
        angles = np.degrees(np.arctan2(ys + 0.5 - 256.0, xs + 0.5 - 256.0))
        self.assertAlmostEqual(float(np.median(radius)), 145.0, delta=4.0)
        self.assertGreater(float(angles.max() - angles.min()), 20.0)
        self.assertLess(float(angles.max() - angles.min()), 40.0)

    def test_surface_blur_keeps_edges(self):
        rng = np.random.default_rng(1)
        base = np.where(np.arange(RES)[None, :] < 256, 0.25, 0.75) * np.ones((RES, 1))
        noisy = np.clip(base + rng.normal(0, 0.02, (RES, RES)), 0, 1)
        self.write(PLANE_COLOR, self.color(np.stack([noisy] * 3, axis=-1), np.ones((RES, RES))))
        self.run_filter("SURFACE_BLUR", radius=4.0, threshold=0.15)
        out = self.read(PLANE_COLOR)[..., 0]
        self.assertLess(float(out[:, 20:230].std()), 0.012, "平的地方噪点变少")
        self.assertAlmostEqual(float(out[:, 250].mean()), 0.25, delta=0.02, msg="边缘左边不被右边带亮")
        self.assertAlmostEqual(float(out[:, 262].mean()), 0.75, delta=0.02, msg="边缘右边不被左边带暗")

    # ---- 锐化、高反差 ----
    def test_unsharp_mask_and_high_pass(self):
        step = np.where(np.arange(RES)[None, :] < 300, 0.3, 0.7) * np.ones((RES, 1))
        self.write(PLANE_COLOR, self.color(np.stack([step] * 3, axis=-1), np.ones((RES, RES))))
        self.run_filter("UNSHARP_MASK", amount=1.0, radius=3.0, threshold=0.0)
        out = self.read(PLANE_COLOR)[100, :, 0]
        self.assertLess(float(out[295:300].min()), 0.27, "暗的一侧压得更暗")
        self.assertGreater(float(out[300:305].max()), 0.73, "亮的一侧提得更亮")
        self.assertAlmostEqual(float(out[100]), 0.3, delta=0.01, msg="远离边缘不变")
        self.write(PLANE_COLOR, self.color(np.stack([step] * 3, axis=-1), np.ones((RES, RES))))
        self.run_filter("HIGH_PASS", radius=4.0)
        out = self.read(PLANE_COLOR)[100, :, 0]
        self.assertAlmostEqual(float(out[50]), 0.5, delta=0.01, msg="平的地方变成中性灰")
        self.assertGreater(float(abs(out[300] - 0.5)), 0.05, "边缘留下来")

    # ---- 杂色 ----
    def test_add_noise_and_median(self):
        gray = np.full((RES, RES, 3), 0.5)
        self.write(PLANE_COLOR, self.color(gray, np.ones((RES, RES))))
        self.run_filter("ADD_NOISE", amount=0.2, distribution="UNIFORM", monochromatic=False, seed=3)
        out = self.read(PLANE_COLOR)
        values = out[..., :3]
        self.assertGreaterEqual(float(values.min()), 0.39)
        self.assertLessEqual(float(values.max()), 0.61)
        self.assertAlmostEqual(float(values.std()), 0.2 / np.sqrt(12.0), delta=0.008)
        self.assertGreater(float(np.abs(out[..., 0] - out[..., 1]).mean()), 0.02, "彩色杂色各通道不一样")
        self.assertTrue(np.allclose(out[..., 3], 1.0), "覆盖度不变")
        # 中间值去掉零星的白点
        speckled = np.full((RES, RES), 0.5)
        speckled[::17, ::13] = 1.0
        self.write(PLANE_COLOR, self.color(np.stack([speckled] * 3, axis=-1), np.ones((RES, RES))))
        self.run_filter("MEDIAN", radius=1.0)
        out = self.read(PLANE_COLOR)[..., 0]
        self.assertLess(float(np.abs(out - 0.5).max()), 0.01)

    # ---- 其它 ----
    def test_minimum_maximum(self):
        coverage = np.zeros((RES, RES), np.float32)
        coverage[100, 100] = 1.0
        self.write(PLANE_COLOR, self.color((1.0, 1.0, 1.0), coverage))
        self.run_filter("MAXIMUM", radius=2.0)
        out = self.read(PLANE_COLOR)[..., 3]
        self.assertEqual(int((out > 0.5).sum()), 25, "一个点扩成 5×5")
        self.run_filter("MINIMUM", radius=2.0)
        out = self.read(PLANE_COLOR)[..., 3]
        self.assertEqual(int((out > 0.5).sum()), 1, "再缩回一个点")

    def test_emboss_and_mosaic(self):
        ramp = np.tile(np.linspace(0.0, 1.0, RES)[None, :], (RES, 1))
        self.write(PLANE_COLOR, self.color(np.stack([ramp] * 3, axis=-1), np.ones((RES, RES))))
        self.run_filter("MOSAIC", cell=16.0)
        out = self.read(PLANE_COLOR)[..., 0]
        self.assertLess(float(np.abs(out[:, 0:16] - out[:, 0:1]).max()), 0.01, "一格里是同一个颜色")
        self.assertAlmostEqual(float(out[0, 40]), float(ramp[0, 32:48].mean()), delta=0.01, msg="格子取平均")
        flat = np.full((RES, RES, 3), 0.3)
        flat[:, 256:] = 0.8
        self.write(PLANE_COLOR, self.color(flat, np.ones((RES, RES))))
        self.run_filter("EMBOSS", angle=0.0, height=2.0, amount=1.0)
        out = self.read(PLANE_COLOR)[..., 0]
        self.assertAlmostEqual(float(out[10, 100]), 0.5, delta=0.01, msg="平的地方是中性灰")
        self.assertGreater(float(out[10, 256]), 0.8, "亮的一边凸起")

    def test_clouds_fill_whole_layer(self):
        self.engine.layers.drop_layer(self.layer.uid)
        self.run_filter("CLOUDS", scale=64.0, seed=2, contrast=1.0, height=0.5,
                        color1=(0.0, 0.0, 0.5, 1.0), color2=(1.0, 0.9, 0.2, 1.0))
        out = self.read(PLANE_COLOR)
        self.assertTrue(np.allclose(out[..., 3], 1.0), "整层铺满")
        self.assertGreater(float(out[..., 0].std()), 0.05, "有起伏")
        self.assertTrue(np.all(out[..., 2] <= 0.51) and np.all(out[..., 1] <= 0.91))

    # ---- 渐变 ----
    def gradient(self, **params) -> np.ndarray:
        base = {"shape": 0, "reverse": False, "opacity": 1.0, "space": "UV", "a": (0.25, 0.5), "b": (0.75, 0.5),
                "ramp": ("LINEAR", ((0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0)))}
        base.update(params)
        self.run_filter("GRADIENT", **base)
        return self.read(PLANE_COLOR)

    def test_gradient_shapes_in_uv(self):
        xs = (np.arange(RES) + 0.5) / RES
        out = self.gradient()
        expected = np.clip((xs - 0.25) / 0.5, 0, 1)
        self.assertLess(float(np.abs(out[200, :, 0] - expected).max()), 1.5 / 255.0, "线性")
        self.assertTrue(np.allclose(out[..., 3], 1.0), "整层盖满")
        self.assertLess(float(np.abs(out[10, :, 0] - out[500, :, 0]).max()), 1e-6, "和竖向位置无关")
        out = self.gradient(shape=1, a=(0.5, 0.5), b=(0.75, 0.5))
        y, x = 256 + 64, 256
        r = np.hypot((x + 0.5) / RES - 0.5, (y + 0.5) / RES - 0.5) / 0.25
        self.assertAlmostEqual(float(out[y, x, 0]), r, delta=1.5 / 255.0, msg="径向")
        out = self.gradient(shape=3, a=(0.5, 0.5), b=(0.75, 0.5))
        self.assertAlmostEqual(float(out[100, 192, 0]), float(out[100, 319, 0]), delta=1.5 / 255.0, msg="对称")
        out = self.gradient(reverse=True)
        self.assertLess(float(np.abs(out[200, :, 0] - (1.0 - expected)).max()), 1.5 / 255.0, "反向")

    def test_gradient_over_existing_and_mask(self):
        self.write(PLANE_COLOR, self.color((1.0, 0.0, 0.0), np.ones((RES, RES))))
        out = self.gradient(opacity=0.5, ramp=("LINEAR", ((0.0, 0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0, 1.0))))
        half = 1.055 * 0.5 ** (1 / 2.4) - 0.055          # 线性光里各一半，换回 sRGB
        self.assertTrue(np.allclose(out[100, 100, :3], (half, 0.0, half), atol=2.0 / 255.0), "一半盖上去")
        # 色标透明：那一头不盖
        self.write(PLANE_COLOR, self.color((1.0, 0.0, 0.0), np.ones((RES, RES))))
        out = self.gradient(ramp=("LINEAR", ((0.0, 0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0, 0.0))))
        self.assertTrue(np.allclose(out[100, 10, :3], (0.0, 0.0, 1.0), atol=2.0 / 255.0))
        self.assertTrue(np.allclose(out[100, 500, :3], (1.0, 0.0, 0.0), atol=2.0 / 255.0))
        # 蒙版：按明暗
        self.layer.has_mask = True
        self.run_filter("GRADIENT", planes={PLANE_MASK: (True,)}, shape=0, opacity=1.0, space="UV",
                        a=(0.0, 0.0), b=(0.0, 1.0), ramp=("LINEAR", ((0.0, 0, 0, 0, 1.0), (1.0, 1, 1, 1, 1.0))))
        mask = self.read(PLANE_MASK)[..., 0]
        self.assertLess(float(mask[10, 200]), 0.05)
        self.assertGreater(float(mask[500, 200]), 0.95)

    def test_gradient_on_screen(self):
        """三维：每个纹素按位置贴图投到屏幕上再算。用一张假的位置贴图（UV 直接当投影后的位置）。"""
        from types import SimpleNamespace

        side = 64
        uv = (np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="xy"), axis=-1) + 0.5) / side
        pos = np.zeros((side, side, 4), np.float32)
        pos[..., 0] = uv[..., 0] * 2 - 1
        pos[..., 1] = uv[..., 1] * 2 - 1
        texture = self.ctx.texture((side, side), 4, pos.tobytes(), dtype="f4")
        texture.repeat_x = texture.repeat_y = False          # 和烘焙出来的模型贴图一样不重复
        self.engine.meshmaps[self.ts.uid] = SimpleNamespace(textures={"p": texture}, size=side,
                                                            bounds=(np.zeros(3), np.ones(3)))
        try:
            out = self.gradient(space="SCREEN", a=(25.0, 30.0), b=(75.0, 30.0), view_size=(100.0, 100.0),
                                matrix=tuple(np.identity(4).ravel()))
        finally:
            del self.engine.meshmaps[self.ts.uid]
            texture.release()
        xs = (np.arange(RES) + 0.5) / RES
        expected = np.clip((xs * 100.0 - 25.0) / 50.0, 0, 1)
        self.assertLess(float(np.abs(out[300, :, 0] - expected).max()), 3.0 / 255.0)

    def test_screen_image_projection(self):
        """三维：屏幕上的一张图投到模型上。从上往下看那块平面（位置贴图和平面一致）：图的上半边在屏幕上方；
        被别的东西挡住的地方不盖；数值页用给的数。"""
        from types import SimpleNamespace

        side = 64
        uv = (np.stack(np.meshgrid(np.arange(side), np.arange(side), indexing="xy"), axis=-1) + 0.5) / side
        pos = np.zeros((side, side, 4), np.float32)
        pos[..., 0] = uv[..., 0] * 2 - 1                 # 平面：x = 2u - 1，z = 1 - 2v（XZ 平面）
        pos[..., 2] = 1 - uv[..., 1] * 2
        pos[..., 3] = 1.0
        texture = self.ctx.texture((side, side), 4, pos.tobytes(), dtype="f4")
        texture.repeat_x = texture.repeat_y = False
        self.engine.meshmaps[self.ts.uid] = SimpleNamespace(textures={"p": texture}, size=side,
                                                            bounds=(-np.ones(3), np.ones(3)))
        top = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], np.float64)  # 从上往下看，正交
        image = np.zeros((64, 64, 4), np.uint8)
        image[..., 0] = 255
        image[:32, :, 3] = 255                            # 上半边实（第一行在上）
        params = dict(image=image, rect=(10.0, 10.0, 40.0, 40.0), opacity=1.0, matrix=tuple(top.ravel()),
                      view_size=(100.0, 100.0))
        # 那一刻看到的表面：左边四分之一被一块浮在上面的东西挡住
        px = (np.arange(100) + 0.5) / 100.0
        su, sv = np.meshgrid(px, 1.0 - px, indexing="xy")
        seen = np.zeros((100, 100, 4), np.float32)
        seen[..., 0] = su * 2 - 1
        seen[..., 2] = 1 - sv * 2
        seen[..., 3] = 1.0
        seen[:, :25, 1] = 0.5
        try:
            self.write(PLANE_COLOR, self.color((0.0, 0.0, 1.0), np.ones((RES, RES))))
            planes = {PLANE_COLOR: (True,), PLANE_MR: (False, True)}
            # 两个三角形都碰到矩形（粗选是四格），真有纹素会变的只有左上那一格：两种页各一格
            self.assertEqual(self.run_filter("SCREEN_IMAGE", planes=planes, values=(0.9, 0.2, 0.0, 0.0), **params), 2)
            out = self.read(PLANE_COLOR)
            red, blue = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
            self.assertTrue(np.allclose(out[420, 120, :3], red, atol=2 / 255), "图的上半边落在屏幕上方（v 大）")
            self.assertTrue(np.allclose(out[330, 120, :3], blue, atol=2 / 255), "透明的下半边不动")
            self.assertTrue(np.allclose(out[420, 300, :3], blue, atol=2 / 255), "矩形外不动")
            self.assertTrue(np.allclose(out[100, 120, :3], blue, atol=2 / 255))
            self.assertAlmostEqual(float(self.read(PLANE_MR)[420, 120, 2]), 0.2, delta=2 / 255, msg="粗糙度盖上")
            # 只盖看得见的地方
            self.write(PLANE_COLOR, self.color((0.0, 0.0, 1.0), np.ones((RES, RES))))
            self.run_filter("SCREEN_IMAGE", gbuf=seen, **params)
            out = self.read(PLANE_COLOR)
            self.assertTrue(np.allclose(out[420, 80, :3], blue, atol=2 / 255), "被挡住的不盖")
            self.assertTrue(np.allclose(out[420, 160, :3], red, atol=2 / 255), "看得见的盖上")
            # 关掉遮挡判断：挡住的也盖
            self.write(PLANE_COLOR, self.color((0.0, 0.0, 1.0), np.ones((RES, RES))))
            self.run_filter("SCREEN_IMAGE", gbuf=seen, occlude=False, **params)
            self.assertTrue(np.allclose(self.read(PLANE_COLOR)[420, 80, :3], red, atol=2 / 255))
        finally:
            del self.engine.meshmaps[self.ts.uid]
            texture.release()

    # ---- 盖一张图（油漆桶、文字、形状）----
    def test_image_stamp(self):
        """一张 RGBA 图盖在 UV 的一块矩形上：矩形外不动；颜色按透明度盖；金属粗糙、高度用给的数。"""
        self.write(PLANE_COLOR, self.color((0.0, 0.0, 1.0), np.ones((RES, RES))))
        image = np.zeros((64, 64, 4), np.uint8)
        image[..., 0] = 255
        image[:, :32, 3] = 255          # 左半边实，右半边透明
        planes = {PLANE_COLOR: (True,), PLANE_MR: (False, True)}
        self.run_filter("IMAGE", planes=planes, image=image, rect=(0.25, 0.25, 0.75, 0.75), opacity=1.0,
                        values=(0.9, 0.2, 0.0, 0.0))
        out = self.read(PLANE_COLOR)
        self.assertTrue(np.allclose(out[200, 160, :3], (1.0, 0.0, 0.0), atol=2 / 255), "矩形里左半边盖成红的")
        self.assertTrue(np.allclose(out[200, 320, :3], (0.0, 0.0, 1.0), atol=2 / 255), "透明的那半边不动")
        self.assertTrue(np.allclose(out[40, 40, :3], (0.0, 0.0, 1.0), atol=2 / 255), "矩形外不动")
        mr = self.read(PLANE_MR)
        self.assertAlmostEqual(float(mr[200, 160, 2]), 0.2, delta=2 / 255, msg="粗糙度盖上")
        self.assertAlmostEqual(float(mr[200, 160, 3]), 1.0, delta=2 / 255)
        self.assertAlmostEqual(float(mr[200, 160, 1]), 0.0, delta=2 / 255, msg="金属度没开，不动")
        # 半透明：一半盖上去
        self.write(PLANE_COLOR, self.color((0.0, 0.0, 1.0), np.ones((RES, RES))))
        image[..., 3] = 128
        self.run_filter("IMAGE", image=image, rect=(0.0, 0.0, 1.0, 1.0), opacity=1.0)
        out = self.read(PLANE_COLOR)
        a = 128 / 255.0
        lin = np.array([a, 0.0, 1.0 - a])
        expected = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
        self.assertTrue(np.allclose(out[300, 300, :3], expected, atol=3 / 255), "线性光里按透明度混")

    def test_mask_stamp(self):
        """只给一张单通道遮罩加一种颜色（油漆桶用）：和给整张 RGBA 图一样。"""
        self.write(PLANE_COLOR, self.color((0.0, 0.0, 1.0), np.ones((RES, RES))))
        mask = np.zeros((64, 64), np.uint8)
        mask[:, :32] = 255
        planes = {PLANE_COLOR: (True,), PLANE_MR: (False, True)}
        self.run_filter("IMAGE", planes=planes, image=mask, rect=(0.25, 0.25, 0.75, 0.75), opacity=1.0,
                        values=(0.9, 0.2, 0.0, 0.0), color2=(1.0, 0.0, 0.0, 1.0))
        out = self.read(PLANE_COLOR)
        self.assertTrue(np.allclose(out[200, 160, :3], (1.0, 0.0, 0.0), atol=2 / 255), "遮罩实的地方盖成给的颜色")
        self.assertTrue(np.allclose(out[200, 320, :3], (0.0, 0.0, 1.0), atol=2 / 255), "遮罩空的地方不动")
        self.assertAlmostEqual(float(self.read(PLANE_MR)[200, 160, 2]), 0.2, delta=2 / 255, msg="粗糙度盖上")

    def test_fill_masks(self):
        from splender.engine import fill

        image = np.zeros((64, 64, 3), np.float32)
        image[:, 32:] = 1.0
        image[10:20, 40:50] = 0.0               # 右边白里一块黑（和左边不连着）
        contiguous = fill.similar_mask(image, 5, 5, 0.1, True)
        self.assertEqual(int(contiguous.sum()), 64 * 32)
        everywhere = fill.similar_mask(image, 5, 5, 0.1, False)
        self.assertEqual(int(everywhere.sum()), 64 * 32 + 100)
        # 8 位的图：容差照样按 0..1 给，结果一样
        image8 = np.rint(image * 255.0).astype(np.uint8)
        self.assertTrue(np.array_equal(fill.similar_mask(image8, 5, 5, 0.1, True), contiguous))
        self.assertTrue(np.array_equal(fill.similar_mask(image8, 5, 5, 0.1, False), everywhere))
        image8[0, 0] = 30                                   # 差 30/255 ≈ 0.118：容差 0.1 不算一片，0.12 算
        self.assertEqual(float(fill.similar_mask(image8, 5, 5, 0.1, True)[0, 0]), 0.0)
        self.assertEqual(float(fill.similar_mask(image8, 5, 5, 0.12, True)[0, 0]), 1.0)
        tris = np.array([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]])
        mask = fill.triangle_mask(self.ctx, tris, 64, dilate=0)
        self.assertGreater(float(mask[5, 5]), 0.5, "三角形里")
        self.assertLess(float(mask[50, 50]), 0.5, "三角形外")
        self.assertAlmostEqual(float(mask.sum()), 64 * 64 / 8, delta=64)

    # ---- 页种类、通道选择、撤销 ----
    def test_roughness_only_and_mask(self):
        mr = np.zeros((RES, RES, 4), np.float32)
        mr[:, :256, 0] = 1.0           # 金属度：左半边 1
        mr[:, :, 1] = 1.0
        mr[:, :256, 2] = 1.0           # 粗糙度：左半边 1
        mr[:, :, 3] = 1.0
        self.write(PLANE_MR, np.rint(mr * 255))
        self.run_filter("GAUSSIAN_BLUR", planes={PLANE_MR: (False, True)}, radius=6.0)
        out = self.read(PLANE_MR)
        self.assertTrue(np.array_equal(out[:, 250:262, 0] > 0.5, mr[:, 250:262, 0] > 0.5), "金属度没动")
        self.assertGreater(float(out[100, 256, 2]), 0.2, "粗糙度在边上被糊开")
        self.assertLess(float(out[100, 256, 2]), 0.8)
        # 蒙版
        self.layer.has_mask = True
        mask = np.zeros((RES, RES, 1), np.float32)
        mask[200:300, 200:300] = 1.0
        self.write(PLANE_MASK, np.rint(mask * 255))
        self.run_filter("GAUSSIAN_BLUR", planes={PLANE_MASK: (True,)}, radius=5.0)
        out = self.read(PLANE_MASK)[..., 0]
        self.assertGreater(float(out[200, 250]), 0.3)
        self.assertLess(float(out[200, 250]), 0.7)
        self.assertAlmostEqual(float(out[250, 250]), 1.0, delta=0.02)

    def test_undo_restores_pixels(self):
        coverage = np.zeros((RES, RES), np.float32)
        coverage[50:80, 300:330] = 1.0
        original = self.color((0.9, 0.4, 0.2), coverage)
        self.write(PLANE_COLOR, original)
        before = self.read(PLANE_COLOR)
        steps = len(self.history.steps)
        self.run_filter("GAUSSIAN_BLUR", radius=8.0)
        self.assertEqual(len(self.history.steps), steps + 1)
        self.assertFalse(np.allclose(self.read(PLANE_COLOR), before))
        self.history.undo()
        self.assertTrue(np.allclose(self.read(PLANE_COLOR), before), "撤销后回到原样")
        self.history.redo()
        self.assertFalse(np.allclose(self.read(PLANE_COLOR), before))


if __name__ == "__main__":
    unittest.main(verbosity=2)
