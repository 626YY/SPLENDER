"""雕刻引擎测试：笔刷形变、撤销重做、平滑、遮罩、抓取、性能和画面。隐藏上下文，不开窗口。"""
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

from splender.engine import ibl  # noqa: E402
from splender.engine.camera import Camera  # noqa: E402
from splender.sculpt import topology as tp  # noqa: E402
from splender.sculpt.engine import BRUSH_INDEX, DAB_DTYPE, SculptEngine  # noqa: E402

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/sculpt")
OUT.mkdir(parents=True, exist_ok=True)


def dab(center, normal, radius, strength=1.0, hardness=0.0, sign=1.0):
    d = np.zeros(1, DAB_DTYPE)
    d["center_radius"][0] = (*center, radius)
    d["normal_strength"][0] = (*normal, strength)
    d["params"][0] = (hardness, sign, 0.0, 0.0)
    return d


def line_of_dabs(start, end, normal, radius, count, strength=0.5, sign=1.0):
    pts = np.linspace(np.asarray(start, float), np.asarray(end, float), count)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    out = np.zeros(count, DAB_DTYPE)
    out["center_radius"][:, :3] = pts
    out["center_radius"][:, 3] = radius
    out["normal_strength"][:, :3] = pts if normal is None else normal
    out["normal_strength"][:, 3] = strength
    out["params"][:, 1] = sign
    return out


class SculptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.engine = SculptEngine(cls.ctx)

    def setUp(self):
        self.mesh = tp.make_sphere(48)
        self.engine.load(self.mesh)
        self.original, _n, _m = self.engine.read_vertices()

    def test_draw_displaces_outward_and_undo_restores(self):
        engine = self.engine
        engine.begin_stroke()
        engine.apply_dabs(dab((0, 0, 1), (0, 0, 1), 0.35, 1.0), BRUSH_INDEX["DRAW"])
        record = engine.end_stroke()
        moved, normals, _mask = engine.read_vertices()
        radius = np.linalg.norm(moved, axis=1)
        pole = np.argmax(self.original[:, 2])
        print("\n标准笔刷：极点半径 %.4f（原来 1.0），改动的块 %d" % (radius[pole], len(record["clusters"])))
        self.assertGreater(radius[pole], 1.02)
        far = self.original[:, 2] < 0.0
        self.assertLess(np.abs(radius[far] - 1.0).max(), 1e-5, "笔刷范围外的顶点不应移动")
        self.assertLess(np.abs(np.linalg.norm(normals, axis=1) - 1.0).max(), 1e-3)
        engine.restore(record["clusters"], record["before"])
        undone, _n, _m = engine.read_vertices()
        self.assertLess(np.abs(undone - self.original).max(), 1e-6, "撤销后应回到原样")
        engine.restore(record["clusters"], record["after"])
        redone, _n, _m = engine.read_vertices()
        self.assertLess(np.abs(redone - moved).max(), 1e-6, "重做后应与改动后一致")

    def test_smooth_reduces_bump(self):
        engine = self.engine
        engine.begin_stroke()
        engine.apply_dabs(dab((0, 0, 1), (0, 0, 1), 0.25, 1.0), BRUSH_INDEX["DRAW"])
        engine.end_stroke()
        bumped, _n, _m = engine.read_vertices()
        engine.begin_stroke()
        for _ in range(20):
            engine.apply_dabs(dab((0, 0, 1), (0, 0, 1), 0.4, 1.0), BRUSH_INDEX["SMOOTH"])
        engine.end_stroke()
        smoothed, _n, _m = engine.read_vertices()
        pole = np.argmax(self.original[:, 2])
        before = np.linalg.norm(bumped[pole]) - 1.0
        after = np.linalg.norm(smoothed[pole]) - 1.0
        print("平滑：极点凸起 %.4f -> %.4f" % (before, after))
        self.assertLess(after, before * 0.7)

    def test_mask_freezes(self):
        engine = self.engine
        engine.begin_stroke()
        engine.apply_dabs(dab((0, 0, 1), (0, 0, 1), 0.3, 1.0), BRUSH_INDEX["MASK"])
        engine.end_stroke()
        _p, _n, mask = engine.read_vertices()
        pole = np.argmax(self.original[:, 2])
        self.assertGreater(mask[pole], 0.99)
        engine.begin_stroke()
        engine.apply_dabs(dab((0, 0, 1), (0, 0, 1), 0.5, 1.0), BRUSH_INDEX["DRAW"])
        engine.end_stroke()
        moved, _n, _m = engine.read_vertices()
        self.assertLess(abs(np.linalg.norm(moved[pole]) - 1.0), 1e-5, "被遮罩冻结的顶点不应移动")
        ring = (np.abs(self.original[:, 2] - 0.93) < 0.01)          # 离极点约 0.37：遮罩（半径 0.3）外、笔刷（半径 0.5）内
        self.assertGreater(np.abs(np.linalg.norm(moved[ring], axis=1) - 1.0).max(), 1e-3, "遮罩外的顶点应该移动")

    def test_grab_moves_region(self):
        engine = self.engine
        engine.begin_stroke()
        engine.grab_begin(dab((0, 0, 1), (0, 0, 1), 0.5, 1.0))
        for step in range(1, 6):
            engine.grab_move(np.array([0.0, 0.0, 0.1 * step]))
        engine.end_stroke()
        moved, _n, _m = engine.read_vertices()
        pole = np.argmax(self.original[:, 2])
        print("抓取：极点 z %.3f（原来 1.0，拖了 0.5）" % moved[pole, 2])
        self.assertAlmostEqual(float(moved[pole, 2]), 1.5, delta=0.01)
        self.assertLess(np.abs(moved[self.original[:, 2] < 0.3] - self.original[self.original[:, 2] < 0.3]).max(), 1e-6)

    def test_all_brushes_and_render(self):
        engine = self.engine
        for name, start, end, sign in (("CLAY", (-0.6, 0.2, 0.8), (0.6, 0.2, 0.8), 1.0),
                                       ("CREASE", (-0.5, -0.25, 0.85), (0.5, -0.25, 0.85), 1.0),
                                       ("INFLATE", (0.8, 0.5, 0.3), (0.8, 0.5, 0.31), 1.0),
                                       ("PINCH", (-0.8, 0.5, 0.3), (-0.8, 0.5, 0.31), 1.0),
                                       ("FLATTEN", (0.0, 0.8, 0.6), (0.0, 0.8, 0.61), 1.0),
                                       ("CLAY_STRIPS", (-0.5, -0.6, 0.6), (0.5, -0.6, 0.6), -1.0)):
            engine.begin_stroke()
            engine.apply_dabs(line_of_dabs(start, end, None, 0.18, 40, 0.5, sign), BRUSH_INDEX[name])
            self.assertIsNotNone(engine.end_stroke(), name)
        positions, normals, _m = engine.read_vertices()
        self.assertTrue(np.isfinite(positions).all() and np.isfinite(normals).all())
        image = render_image(self.ctx, engine)
        Image.fromarray(image).save(OUT / "sculpt_brushes.png")

    def test_performance_millions(self):
        engine = self.engine
        big = tp.subdivide(tp.make_sphere(128), 2)          # 约 300 万三角形
        t0 = time.perf_counter()
        engine.load(big)
        load = time.perf_counter() - t0
        self.ctx.finish()
        dabs = line_of_dabs((-0.7, 0.3, 0.65), (0.7, 0.3, 0.65), None, 0.12, 32, 0.4)
        timings = []
        engine.begin_stroke()
        for frame in range(30):
            part = dabs.copy()
            part["center_radius"][:, 1] -= frame * 0.02
            t1 = time.perf_counter()
            engine.apply_dabs(part, BRUSH_INDEX["CLAY"])
            self.ctx.finish()
            timings.append((time.perf_counter() - t1) * 1000.0)
        t2 = time.perf_counter()
        record = engine.end_stroke()
        end_ms = (time.perf_counter() - t2) * 1000.0
        timings = np.array(timings)
        print("性能：%d 个三角形，载入 %.2f 秒；每帧 32 个笔触：中位数 %.2f ms，最长 %.2f ms；抬笔读回 %d 块用 %.0f ms" % (
            big.triangle_count, load, np.median(timings), timings.max(), len(record["clusters"]), end_ms))
        self.assertLess(np.median(timings), 16.0)
        image = render_image(self.ctx, engine)
        Image.fromarray(image).save(OUT / "sculpt_3m.png")


def render_image(ctx, engine, size=(900, 700)):
    color = ctx.texture(size, 4)
    depth = ctx.depth_renderbuffer(size)
    fbo = ctx.framebuffer([color], depth)
    fbo.use()
    fbo.clear(0.16, 0.16, 0.17, 1.0, depth=1.0)
    ctx.enable(moderngl.DEPTH_TEST)
    camera = Camera()
    camera.yaw, camera.pitch = 20.0, 15.0
    camera.frame(np.zeros(3), 1.1, size[0] / size[1])
    matcap = ibl.create_matcap_texture(ctx, ibl.load_matcap(ibl.list_matcaps()[0]))
    engine.draw(camera.view_projection(size[0] / size[1]), camera.view_matrix(), camera.eye, solid=1,
                tint=(0.8, 0.8, 0.8), matcap_tex=matcap)
    data = np.frombuffer(fbo.read(components=3), np.uint8).reshape(size[1], size[0], 3)[::-1]
    for item in (fbo, color, depth, matcap):
        item.release()
    return np.ascontiguousarray(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
