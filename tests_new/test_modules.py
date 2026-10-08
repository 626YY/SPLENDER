"""网格预处理、PNG 流式导出、环境光照资源的测试（不需要窗口）。"""
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.engine import ibl  # noqa: E402
from splender.engine.export_png import PngStreamWriter  # noqa: E402
from splender.engine.meshprep import build_set_geometry  # noqa: E402


class MeshPrepTest(unittest.TestCase):
    def test_every_triangle_lands_in_its_tiles(self):
        mesh = make_test_mesh("sphere", segments=96, rings=48)
        geo = build_set_geometry(mesh, 0, 16384, skirt_texels=32)
        grid = geo.grid
        self.assertEqual(grid, 64)
        uvs = geo.uvs if geo.uvs is not None else mesh.uvs
        # 抽查：每个三角形的 UV 包围盒中心所在的格，一定列出了这个三角形
        rng = np.random.default_rng(5)
        tris = rng.choice(mesh.triangle_count, 200, replace=False)
        for t in tris:
            corners = uvs[t * 3:t * 3 + 3]
            center = corners.mean(axis=0)
            tx = min(grid - 1, int(center[0] * grid))
            ty = min(grid - 1, int(center[1] * grid))
            listed = geo.tile_indices(tx, ty)
            self.assertIn(t * 3, set((listed // 3 * 3).tolist()), "三角形 %d 不在它所在的格 (%d, %d) 里" % (t, tx, ty))
        self.assertGreater(len(geo.skirt_vertices), 0, "球的 UV 有接缝，应该生成扩边")
        self.assertEqual(geo.skirt_vertices.shape[1], 8)
        # 有三角形的格，包围盒有限；空格是无穷
        filled = geo.tile_count > 0
        self.assertTrue(np.isfinite(geo.tile_min[filled]).all())

    def test_million_triangles_speed(self):
        mesh = make_test_mesh("sphere", segments=1000, rings=500)
        build_set_geometry(mesh, 0, 4096, skirt_texels=8)          # 预热（编译）
        t0 = time.perf_counter()
        geo = build_set_geometry(mesh, 0, 16384, skirt_texels=32)
        elapsed = time.perf_counter() - t0
        print("\n%d 个三角形、16K 预处理用时 %.2f 秒，扩边顶点 %d" % (mesh.triangle_count, elapsed, len(geo.skirt_vertices)))
        self.assertGreater(mesh.triangle_count, 900_000)
        self.assertLess(elapsed, 20.0)


class PngExportTest(unittest.TestCase):
    def test_rgba8_and_gray16_roundtrip(self):
        folder = Path(tempfile.mkdtemp(prefix="splender_png_"))
        rng = np.random.default_rng(1)
        rgba = rng.integers(0, 255, (203, 301, 4), dtype=np.uint8)
        path = folder / "颜色.png"
        writer = PngStreamWriter(str(path), 301, 203, 4, bit_depth=8)
        for start in range(0, 203, 37):
            writer.write_rows(rgba[start:start + 37])
        writer.close()
        back = np.asarray(Image.open(path).convert("RGBA"))
        np.testing.assert_array_equal(back, rgba)
        gray = rng.integers(0, 65535, (64, 128), dtype=np.uint16)
        path16 = folder / "高度.png"
        writer = PngStreamWriter(str(path16), 128, 64, 1, bit_depth=16)
        writer.write_rows(gray[:20])
        writer.write_rows(gray[20:])
        writer.close()
        back16 = np.asarray(Image.open(path16))
        np.testing.assert_array_equal(back16.astype(np.uint16), gray)

    def test_too_many_rows_rejected(self):
        folder = Path(tempfile.mkdtemp(prefix="splender_png_"))
        writer = PngStreamWriter(str(folder / "x.png"), 16, 8, 3)
        writer.write_rows(np.zeros((8, 16, 3), np.uint8))
        with self.assertRaises(ValueError):
            writer.write_rows(np.zeros((1, 16, 3), np.uint8))
        writer.close()
        self.assertEqual(Image.open(folder / "x.png").size, (16, 8))


class EnvironmentAssetsTest(unittest.TestCase):
    def test_hdris_matcaps_studio(self):
        hdris = ibl.list_hdris()
        self.assertGreaterEqual(len(hdris), 1)
        image = ibl.load_hdri(hdris[0])
        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)
        self.assertTrue(np.isfinite(image).all())
        sh = ibl.sh_project(image)
        up = ibl.sh_evaluate(sh, np.array([[0.0, 1.0, 0.0]]))
        self.assertTrue(np.isfinite(up).all() and (up >= 0).all())
        self.assertGreaterEqual(len(ibl.list_matcaps()), 1)
        lights = ibl.list_studio_lights()
        self.assertGreaterEqual(len(lights), 1)
        self.assertIsInstance(ibl.load_studio_light(lights[0]), dict)
        for name in hdris:
            self.assertTrue(ibl.label(name, "hdri"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
