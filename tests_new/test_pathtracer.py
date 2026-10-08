"""路径追踪渲染器的测试：白炉能量守恒、BVH 与暴力求交一致、贴图与环境光、材质递变、透明背景、性能。

用隐藏的显卡上下文，不开窗口。图片存到 docs/evidence/render/。
"""
import json
import math
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

from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.engine import ibl  # noqa: E402
from splender.render.pathtracer import PathTracer  # noqa: E402
from splender.render.scene import (RenderCamera, RenderEnvironment, RenderMaterial, RenderMesh, RenderScene,  # noqa: E402
                                   RenderSettings, RenderTexture)

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/render")
OUT.mkdir(parents=True, exist_ok=True)


def sphere_mesh(segments=96, rings=48, offset=(0.0, 0.0, 0.0), radius=1.0, material=0):
    mesh = make_test_mesh("sphere", segments=segments, rings=rings)
    positions = mesh.positions * radius + np.asarray(offset, np.float32)
    return RenderMesh("球", positions.astype(np.float32), mesh.normals.astype(np.float32), mesh.uvs.astype(np.float32),
                      np.full(mesh.triangle_count, material, np.int32))


def look_at(eye, target, tan_half_y=math.tan(math.radians(20.0))):
    eye = np.asarray(eye, np.float64)
    forward = np.asarray(target, np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return RenderCamera(eye=eye, forward=forward, up=up, tan_half_y=tan_half_y)


def render(tracer, samples, budget=200.0):
    t0 = time.perf_counter()
    while not tracer.finished:
        tracer.render_pass(budget)
    return time.perf_counter() - t0


def save(image, name):
    Image.fromarray(image[:, :, :3] if image.shape[2] == 4 else image).save(OUT / name)


class PathTracerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tracer = PathTracer(cls.ctx)
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        cls.tracer.release()
        (OUT / "pathtracer_results.json").write_text(json.dumps(cls.results, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")

    def test_1_white_furnace(self):
        """反照率 1 的白色漫反射球，放在亮度 1 的均匀环境里，每个像素都应约等于 1。"""
        settings = RenderSettings(width=96, height=96, samples=96, max_bounces=8, view_transform="STANDARD")
        settings.debug_mode = 1
        scene = RenderScene(meshes=[sphere_mesh()], materials=[RenderMaterial(base_color=(1.0, 1.0, 1.0))],
                            environment=RenderEnvironment(image=None, color=(1.0, 1.0, 1.0), background=True),
                            camera=look_at((0, 0, 4.2), (0, 0, 0)), settings=settings)
        tracer = self.tracer
        tracer.set_scene(scene)
        render(tracer, settings.samples)
        image = tracer.read_image(linear=True)
        rgb = image[:, :, :3]
        center = rgb[24:72, 24:72]
        mean = float(center.mean())
        print("\n白炉测试：球上平均 %.4f，最小 %.3f，最大 %.3f" % (mean, center.min(), center.max()))
        self.results["white_furnace_mean"] = round(mean, 4)
        self.assertAlmostEqual(mean, 1.0, delta=0.02)

    def test_2_bvh_matches_brute_force(self):
        """5 万三角形的球加两个小球，主射线交点与 numpy 暴力求交一致。"""
        meshes = [sphere_mesh(200, 125), sphere_mesh(32, 16, offset=(0.9, 0.4, 0.9), radius=0.35),
                  sphere_mesh(32, 16, offset=(-0.8, -0.5, 0.8), radius=0.3)]
        total = sum(m.triangle_count for m in meshes)
        settings = RenderSettings(width=100, height=100, samples=1)
        settings.debug_mode = 2
        camera = look_at((0.3, 0.6, 3.5), (0, 0, 0))
        scene = RenderScene(meshes=meshes, materials=[RenderMaterial()], camera=camera, settings=settings)
        tracer = self.tracer
        tracer.set_scene(scene)
        tracer.render_pass(500.0)
        gpu = np.frombuffer(tracer._accum.read(), np.float32).reshape(100, 100, 4)
        positions = np.concatenate([m.positions for m in meshes]).astype(np.float64)
        v0, v1, v2 = positions[0::3], positions[1::3], positions[2::3]
        e1, e2 = v1 - v0, v2 - v0
        rng = np.random.default_rng(3)
        checked = mismatched = 0
        for _ in range(400):
            px, py = int(rng.integers(0, 100)), int(rng.integers(0, 100))
            origin, direction = camera.ray(px + 0.5, py + 0.5, 100, 100)
            p = np.cross(direction, e2)
            det = np.einsum("ij,ij->i", e1, p)
            ok = np.abs(det) > 1e-14
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            s = origin - v0
            u = np.einsum("ij,ij->i", s, p) * inv
            q = np.cross(s, e1)
            v = (q @ direction) * inv
            t = np.einsum("ij,ij->i", e2, q) * inv
            valid = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 0)
            ref_t = float(t[valid].min()) if valid.any() else -1.0
            got_t = float(gpu[py, px, 0])
            checked += 1
            if (ref_t < 0) != (got_t < 0) or (ref_t >= 0 and abs(ref_t - got_t) > 1e-3 * max(1.0, ref_t)):
                mismatched += 1
        print("BVH：%d 个三角形，节点 %d，深度 %d，建树 %.1f ms；抽查 %d 条射线，不一致 %d" % (
            total, tracer.stats()["nodes"], tracer.stats()["bvh_depth"], tracer.stats()["bvh_ms"], checked, mismatched))
        self.results["bvh_mismatch"] = mismatched
        self.assertEqual(mismatched, 0)

    def _textured_scene(self, width, height, samples, mesh=None, size=1024):
        tex = np.zeros((size, size, 4), np.uint8)
        cells = (np.add.outer(np.arange(size) // (size // 16), np.arange(size) // (size // 16)) % 2).astype(bool)
        tex[cells] = (220, 70, 50, 255)
        tex[~cells] = (235, 230, 220, 255)
        mr = np.zeros((size, size, 2), np.uint8)
        mr[:, :, 0] = np.where(cells, 255, 0)
        mr[:, :, 1] = np.where(cells, 70, 160)
        yy, xx = np.mgrid[0:size, 0:size] / size
        height_map = (0.5 + 0.5 * np.sin(xx * 80.0) * np.sin(yy * 40.0)).astype(np.float32)
        material = RenderMaterial(base_color_tex=RenderTexture(tex, srgb=True), mr_tex=RenderTexture(mr),
                                  height_tex=RenderTexture(height_map), height_scale=0.0004)
        hdri = ibl.list_hdris()[0]
        env = RenderEnvironment(image=ibl.load_hdri(hdri), rotation=0.6, strength=1.0, background=True)
        settings = RenderSettings(width=width, height=height, samples=samples, max_bounces=4, texture_size=size)
        return RenderScene(meshes=[mesh or sphere_mesh(128, 64)], materials=[material], environment=env,
                           camera=look_at((1.4, 0.8, 3.6), (0, 0, 0)), settings=settings), hdri

    def test_3_textured_with_hdri(self):
        scene, hdri = self._textured_scene(640, 400, 64)
        tracer = self.tracer
        tracer.set_scene(scene)
        seconds = render(tracer, 64)
        image = tracer.read_image()
        save(image, "pt_textured_hdri.png")
        print("贴图加环境光（%s）：640×400，64 次采样，%.2f 秒" % (hdri, seconds))
        self.assertGreater(image[:, :, :3].std(), 10)

    def test_4_material_sweep(self):
        meshes, materials = [], []
        for row, metal in enumerate((0.0, 1.0)):
            for col in range(6):
                rough = col / 5.0
                materials.append(RenderMaterial(base_color=(0.9, 0.55, 0.2) if metal else (0.75, 0.1, 0.1),
                                                metallic=metal, roughness=rough))
                meshes.append(sphere_mesh(48, 24, offset=(-2.75 + col * 1.1, 0.6 - row * 1.2, 0.0), radius=0.5,
                                          material=len(materials) - 1))
        hdri = ibl.list_hdris()[0]
        settings = RenderSettings(width=900, height=360, samples=96, max_bounces=4)
        scene = RenderScene(meshes=meshes, materials=materials,
                            environment=RenderEnvironment(image=ibl.load_hdri(hdri), strength=1.0),
                            camera=look_at((0, 0, 7.5), (0, 0, 0), math.tan(math.radians(16))), settings=settings)
        tracer = self.tracer
        tracer.set_scene(scene)
        seconds = render(tracer, 96)
        image = tracer.read_image()
        save(image, "pt_material_sweep.png")
        print("材质递变：上排非金属、下排金属，粗糙度从左到右 0 到 1；%.2f 秒" % seconds)
        # 光滑金属球上能看到环境的锐利细节，粗糙的则模糊：比较球心区域的局部对比度
        def detail(cx, cy):
            patch = image[cy - 12:cy + 12, cx - 12:cx + 12, :3].astype(float)
            return float(np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean())
        cam = scene.camera

        def project(x, y):
            rel = np.array([x, y, 0.0]) - cam.eye
            depth = rel @ cam.forward
            sx = (rel @ cam.right) / depth / (cam.tan_half_y * 900 / 360)
            sy = (rel @ cam.up) / depth / cam.tan_half_y
            return int((sx * 0.5 + 0.5) * 900), int((0.5 - sy * 0.5) * 360)
        sharp = detail(*project(-2.75, -0.6))
        blurry = detail(*project(-2.75 + 5 * 1.1, -0.6))
        print("光滑金属球细节 %.2f，粗糙金属球细节 %.2f" % (sharp, blurry))
        self.assertGreater(sharp, blurry)

    def test_5_transparent_background(self):
        settings = RenderSettings(width=80, height=80, samples=8, transparent=True)
        scene = RenderScene(meshes=[sphere_mesh(32, 16, radius=0.6)], materials=[RenderMaterial()],
                            environment=RenderEnvironment(color=(0.5, 0.5, 0.5)),
                            camera=look_at((0, 0, 4), (0, 0, 0)), settings=settings)
        tracer = self.tracer
        tracer.set_scene(scene)
        render(tracer, 8)
        image = tracer.read_image(linear=True)
        self.assertLess(float(image[2, 2, 3]), 0.01, "角落是背景，应透明")
        self.assertGreater(float(image[40, 40, 3]), 0.99, "中间是球，应不透明")

    def test_7_denoise(self):
        """4 次采样降噪后，比不降噪的 4 次采样更接近 512 次采样的参考图。"""
        tracer = self.tracer
        scene, _hdri = self._textured_scene(320, 200, 512)
        tracer.set_scene(scene)
        render(tracer, 512)
        reference = tracer.read_image(linear=True)[:, :, :3].copy()
        scene, _hdri = self._textured_scene(320, 200, 4)
        tracer.set_scene(scene)
        render(tracer, 4)
        noisy = tracer.read_image(linear=True)[:, :, :3].copy()
        stats = tracer.denoise_now("AUTO")
        clean = tracer.read_image(linear=True)[:, :, :3]
        save(tracer.read_image(), "pt_denoised_4spp.png")
        err_noisy = float(np.mean((np.clip(noisy, 0, 4) - np.clip(reference, 0, 4)) ** 2))
        err_clean = float(np.mean((np.clip(clean, 0, 4) - np.clip(reference, 0, 4)) ** 2))
        print()
        print("降噪（%s，%.0f 毫秒）：4 次采样误差 %.5f，降噪后 %.5f" % (stats["device"], stats["ms"], err_noisy, err_clean))
        self.results["denoise"] = {"device": stats["device"], "ms": stats["ms"], "mse_noisy": err_noisy,
                                   "mse_denoised": err_clean}
        self.assertLess(err_clean, err_noisy * 0.5)
        self.assertEqual(tracer.denoised_samples, 4)
        tracer.reset()
        self.assertEqual(tracer.denoised_samples, 0, "重新开始后降噪结果作废")

    def test_6_performance_million_triangles(self):
        mesh = make_test_mesh("sphere", segments=1000, rings=500)
        render_mesh = RenderMesh("百万三角形球", mesh.positions, mesh.normals, mesh.uvs,
                                 np.zeros(mesh.triangle_count, np.int32))
        scene, _hdri = self._textured_scene(1920, 1080, 12, mesh=render_mesh, size=4096)
        tracer = self.tracer
        t0 = time.perf_counter()
        tracer.set_scene(scene)
        setup = time.perf_counter() - t0
        seconds = render(tracer, 12)
        stats = tracer.stats()
        save(tracer.read_image(), "pt_million_1080p.png")
        print("性能：%d 个三角形，建 BVH %.0f ms，准备场景共 %.2f 秒；1920×1080 最大反弹 4 次，"
              "%d 次采样用 %.2f 秒，每秒 %.1f 次采样，每秒 %.0f 百万像素样本" % (
                  stats["triangles"], stats["bvh_ms"], setup, stats["samples"], seconds, stats.get("spp_per_second", 0),
                  stats.get("mpix_samples_per_second", 0)))
        self.results["million"] = {"triangles": stats["triangles"], "bvh_ms": stats["bvh_ms"], "setup_s": round(setup, 2),
                                   "spp_per_second": stats.get("spp_per_second"), "bvh_depth": stats["bvh_depth"]}
        self.assertGreater(stats.get("spp_per_second", 0), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
