"""法线贴图：低模 + 从高模烘的法线贴图，在视口和路径追踪里看起来应该接近直接渲染高模。
隐藏上下文，不开窗口。图存到 docs/evidence/bake/。"""
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
from splender.doc.meshio import MeshData, make_test_mesh  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet, ToolSettings, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from splender.sculpt import topology as tp  # noqa: E402

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/bake")
OUT.mkdir(parents=True, exist_ok=True)


def bumpy_mesh() -> MeshData:
    mesh = tp.subdivide(tp.make_sphere(96), 1)
    v = mesh.vertices.astype(np.float64)
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    r = 1.0 + 0.05 * np.sin(6.0 * np.arctan2(n[:, 2], n[:, 0])) * np.cos(6.0 * np.arcsin(n[:, 1]))
    mesh.vertices = (n * r[:, None]).astype(np.float32)
    positions, normals, uvs, mats = tp.unweld(mesh)
    return MeshData(name="高模", positions=positions, normals=normals, uvs=uvs, material_ids=mats, materials=["材质"],
                    bounds_min=positions.min(axis=0), bounds_max=positions.max(axis=0))


def write_obj(path, mesh: MeshData) -> None:
    lines = ["v %.6f %.6f %.6f" % tuple(p) for p in mesh.positions]
    lines += ["vn %.6f %.6f %.6f" % tuple(q) for q in mesh.normals]
    for t in range(len(mesh.positions) // 3):
        a, b, c = 3 * t + 1, 3 * t + 2, 3 * t + 3
        lines.append("f %d//%d %d//%d %d//%d" % (a, a, b, b, c, c))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def luminance(image) -> np.ndarray:
    rgb = image[:, :, :3].astype(np.float64)
    return rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722


def correlation(a, b, mask) -> float:
    x = a[mask] - a[mask].mean()
    y = b[mask] - b[mask].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))


def box(image, radius) -> np.ndarray:
    pad = np.pad(image, radius, mode="edge")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * radius + 1
    h, w = image.shape
    total = c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]
    return total / (k * k)


def inner_mask(covered, shrink=0.8) -> np.ndarray:
    """球的内部：按覆盖面积算出半径再缩一圈（轮廓附近法线贴图改不了，不算）。"""
    ys, xs = np.nonzero(covered)
    cy, cx = ys.mean(), xs.mean()
    radius = np.sqrt(covered.sum() / np.pi)
    yy, xx = np.mgrid[0:covered.shape[0], 0:covered.shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) < (radius * shrink) ** 2


def bump_agreement(flat, mapped, high, covered, blur=0) -> tuple:
    """法线贴图带来的明暗变化（加贴图减平滑）和高模起伏带来的明暗变化（高模减平滑）有多一致。"""
    mask = inner_mask(covered)
    d_map = luminance(mapped) - luminance(flat)
    d_high = luminance(high) - luminance(flat)
    if blur:
        d_map, d_high = box(d_map, blur), box(d_high, blur)
    ratio = float(d_map[mask].std() / max(d_high[mask].std(), 1e-6))
    return correlation(d_map, d_high, mask), ratio


class _SceneBase(unittest.TestCase):
    """共用：隐藏上下文、高模文件、搭场景。"""

    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = Path(tempfile.mkdtemp())
        cls.high = bumpy_mesh()
        write_obj(cls.tmp / "high.obj", cls.high)

    def scene(self, mesh):
        project = Project("法线")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="2048", base_color=(0.7, 0.7, 0.72)))
        ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.7, 0.7, 0.72), fill_roughness=0.35))
        project.add_object(MeshObject(mesh.name, mesh, [ts.uid]))
        engine = Engine(self.ctx, Preferences(), History())
        self.addCleanup(engine.release)
        engine.set_project(project)
        view = engine.create_view(ViewShading(mode="MATERIAL", hdri="interior"))
        view.resize(640, 640)
        camera = view.camera
        camera.target = np.zeros(3)
        camera.distance = 3.4
        camera.yaw, camera.pitch = 20.0, 15.0
        view.invalidate_camera()
        return engine, ts, view


class NormalMapTest(_SceneBase):
    def trace(self, engine, view) -> np.ndarray:
        from splender.render.build import build_scene
        from splender.render.pathtracer import PathTracer
        from splender.render.scene import RenderSettings
        settings = RenderSettings(width=320, height=320, samples=128, texture_size=2048, transparent=True)
        tracer = PathTracer(self.ctx)
        try:
            scene = build_scene(engine, engine.project, view.camera, settings, view.shading, aspect=1.0)
            tracer.set_scene(scene)
            while not tracer.finished:
                tracer.render_pass(200.0)
            return tracer.read_image(), scene
        finally:
            tracer.release()

    def test_viewport_and_pathtrace(self):
        engine, ts, view = self.scene(bumpy_mesh())
        engine.settle()
        high = engine.renderer.read_pixels(view).copy()
        pt_high, _scene = self.trace(engine, view)
        low_mesh = make_test_mesh("sphere", segments=96, rings=48)
        engine, ts, view = self.scene(low_mesh)
        engine.settle()
        flat = engine.renderer.read_pixels(view).copy()
        pt_flat, scene = self.trace(engine, view)
        self.assertIsNone(scene.materials[0].normal_tex, "没烘之前不该有法线贴图")
        ts.meshmap.resolution = "2048"
        ts.meshmap.ao_samples = 8
        ts.meshmap.high_poly = str(self.tmp / "high.obj")
        ts.meshmap.cage_front = ts.meshmap.cage_back = 0.06
        engine.bake_start(ts)
        while engine.bake is not None:
            engine.frame()
        engine.settle()
        mapped = engine.renderer.read_pixels(view).copy()
        pt_mapped, scene = self.trace(engine, view)
        self.assertIsNotNone(scene.materials[0].normal_tex, "出图场景里应带上法线贴图")
        for name, image in (("nmap_high.png", high), ("nmap_low_flat.png", flat), ("nmap_low_mapped.png", mapped),
                            ("nmap_pt_high.png", pt_high), ("nmap_pt_low_flat.png", pt_flat),
                            ("nmap_pt_low_mapped.png", pt_mapped)):
            Image.fromarray(np.ascontiguousarray(image[:, :, :3])).save(OUT / name)
        # 视口
        covered = np.abs(luminance(flat) - luminance(flat)[0, 0]) > 2.0
        c_vp, ratio_vp = bump_agreement(flat, mapped, high, covered)
        print()
        print("视口：法线贴图的明暗变化和高模起伏的相关 %.3f，幅度比 %.2f" % (c_vp, ratio_vp))
        # 路径追踪：背景透明，按不透明度取覆盖；有采样噪点，差分先稍微平滑
        covered = pt_flat[:, :, 3] > 250
        c_pt, ratio_pt = bump_agreement(pt_flat, pt_mapped, pt_high, covered, blur=2)
        print("路径追踪：法线贴图的明暗变化和高模起伏的相关 %.3f，幅度比 %.2f" % (c_pt, ratio_pt))
        self.assertGreater(c_vp, 0.8)
        self.assertGreater(ratio_vp, 0.6)
        self.assertGreater(c_pt, 0.7)
        self.assertGreater(ratio_pt, 0.5)
        # 导出：没有手绘高度时，导出的法线就是烘出来的那张
        baked = engine.meshmap(ts.uid).read_arrays()["t"][::-1, :, :3].astype(int)
        out_gl = self.tmp / "normal_gl.png"
        info = engine.export_channel(ts, "normal", str(out_gl))
        self.assertTrue(info["baked"])
        exported = np.asarray(Image.open(out_gl)).astype(int)
        diff = np.abs(exported - baked)
        print("导出法线和烘焙法线：平均差 %.3f，最大差 %d" % (diff.mean(), diff.max()))
        self.assertLess(diff.mean(), 0.5)
        self.assertLessEqual(diff.max(), 3)
        ts.meshmap.normal_format = "DIRECTX"
        out_dx = self.tmp / "normal_dx.png"
        engine.export_channel(ts, "normal", str(out_dx))
        directx = np.asarray(Image.open(out_dx)).astype(int)
        self.assertLessEqual(np.abs(directx[:, :, 1] - (255 - exported[:, :, 1])).max(), 1, "DirectX 绿色反过来")
        self.assertEqual(np.abs(directx[:, :, [0, 2]] - exported[:, :, [0, 2]]).max(), 0)
        ts.meshmap.normal_format = "OPENGL"
        ts.export.normal_bake = False
        flat_only = engine.export_channel(ts, "normal", str(self.tmp / "normal_flat.png"))
        self.assertFalse(flat_only["baked"])
        flat_image = np.asarray(Image.open(self.tmp / "normal_flat.png")).astype(int)
        self.assertLessEqual(np.abs(flat_image - np.array([128, 128, 255])).max(), 1, "不含高模细节时是平的")

    def test_export_height_normal(self):
        """手绘高度导出成法线：方向、强度和视口里看到的凹凸一致。"""
        project = Project("高度")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="2048"))
        ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.6, 0.6, 0.62), fill_roughness=0.4))
        ts.add_layer(Layer(name="绘制"))
        project.add_object(MeshObject("平面", make_test_mesh("plane"), [ts.uid]))
        engine = Engine(self.ctx, Preferences(), History())
        self.addCleanup(engine.release)
        engine.set_project(project)
        view = engine.create_view(ViewShading(mode="MATERIAL", hdri="interior"))
        view.resize(640, 640)
        camera = view.camera
        camera.target = np.zeros(3)
        camera.distance = 2.2
        camera.yaw, camera.pitch = 0.0, 55.0
        view.invalidate_camera()
        engine.settle()
        tools = ToolSettings()
        tools.brush.size = 90
        tools.brush.color = (0.6, 0.6, 0.62)
        tools.brush.use_height = True
        for dy in (-100, 0, 100):
            cx, cy = 320, 320 + dy
            ok = engine.stroke_begin(view, cx - 150, cy, 1.0, time.perf_counter(), tools)
            self.assertTrue(ok, "落笔失败：%r" % (engine.stroke_reject,))
            for i in range(1, 31):
                engine.stroke_move(cx - 150 + i * 10, cy + 30 * np.sin(i * 0.4), 1.0, time.perf_counter())
                engine.frame()
            engine.stroke_end()
            engine.settle()
        bumped = engine.renderer.read_pixels(view).copy()
        # 高度和法线各导一张（不含烘焙细节）
        ts.export.normal_bake = False
        out_h = self.tmp / "plane_height.png"
        out_n = self.tmp / "plane_normal.png"
        engine.export_channel(ts, "height", str(out_h))
        info = engine.export_channel(ts, "normal", str(out_n))
        height = np.asarray(Image.open(out_h)).astype(np.float64) / 65535.0 * 2.0 - 1.0
        normal = np.asarray(Image.open(out_n)).astype(np.float64) / 255.0 * 2.0 - 1.0
        self.assertGreater(np.abs(height).max(), 0.02, "应该画出了高度")
        # 按导出的高度自己算一遍：x 向右是 u，图片往上是 v
        h = np.pad(height, 1, mode="edge")
        du = (h[1:-1, 2:] - h[1:-1, :-2]) * 0.5
        dv = (h[:-2, 1:-1] - h[2:, 1:-1]) * 0.5
        expect = np.stack([-du * info["slope"], -dv * info["slope"], np.ones_like(du)], axis=2)
        expect /= np.linalg.norm(expect, axis=2, keepdims=True)
        busy = np.abs(du) + np.abs(dv) > 1e-4
        err = np.abs(normal - expect)[busy]
        print()
        print("高度换算法线：有坡的像素 %d 个，平均差 %.4f，最大差 %.4f，坡度系数 %.1f" % (busy.sum(), err.mean(), err.max(),
                                                                       info["slope"]))
        self.assertGreater(busy.sum(), 2000)
        self.assertLess(err.mean(), 0.01)
        # 把导出的法线当成法线贴图、关掉高度凹凸，视口里应该和原来的凹凸一样
        ts.meshmap.resolution = "2048"
        ts.meshmap.ao_samples = 4
        ts.meshmap.show_ao = False
        engine.bake_start(ts)
        while engine.bake is not None:
            engine.frame()
        maps = engine.meshmap(ts.uid)
        rows = np.asarray(Image.open(out_n))[::-1]
        rgba = np.concatenate([rows, np.full(rows.shape[:2] + (1,), 255, np.uint8)], axis=2)
        texture = self.ctx.texture((rgba.shape[1], rgba.shape[0]), 4, np.ascontiguousarray(rgba).tobytes(), dtype="f1")
        texture.build_mipmaps()
        maps.textures["t"].release()
        maps.textures["t"] = texture
        maps.meta["high_poly"] = "导出的法线"
        view.shading.bump = False
        engine.set_meshmap(ts.uid, maps)
        engine.settle()
        mapped = engine.renderer.read_pixels(view).copy()
        maps.meta["high_poly"] = ""
        engine.set_meshmap(ts.uid, maps)
        engine.settle()
        plain = engine.renderer.read_pixels(view).copy()
        for name, image in (("height_bump.png", bumped), ("height_as_normal.png", mapped), ("height_plain.png", plain)):
            Image.fromarray(image[:, :, :3]).save(OUT / name)
        covered = np.abs(luminance(plain) - luminance(plain)[0, 0]) > 2.0
        c, ratio = bump_agreement(plain, mapped, bumped, covered)
        print("视口：导出法线的明暗变化和高度凹凸的相关 %.3f，幅度比 %.2f" % (c, ratio))
        self.assertGreater(c, 0.85)
        self.assertGreater(ratio, 0.75)
        self.assertLess(ratio, 1.3)


def arnold_found() -> bool:
    try:
        from splender.addons import render_arnold as ra
        return ra.find_arnold("") is not None
    except Exception:  # noqa: BLE001
        return False


class ArnoldNormalMapTest(_SceneBase):
    """Arnold 也用上法线贴图：低模加贴图的起伏和高模一致。"""

    @unittest.skipUnless(arnold_found(), "这台电脑上没有 Arnold")
    def test_arnold(self):
        from splender.addons import render_arnold as ra
        from splender.render.build import build_scene
        from splender.render.engines import RenderRequest
        from splender.render.scene import RenderSettings

        ra._STATE["license"] = False                  # 只看画面，不等授权检查
        settings = RenderSettings(width=240, height=240, samples=64, texture_size=2048, transparent=True)

        def render(engine, view):
            scene = build_scene(engine, engine.project, view.camera, settings, view.shading, aspect=1.0)
            es = ra.ArnoldSettings()
            es.camera_aa = 4
            es.progressive = False
            job = ra.ArnoldJob(RenderRequest(scene=scene, settings=settings, engine_settings=es))
            while not job.done:
                time.sleep(0.05)
            if job.error:
                raise RuntimeError(job.error)
            return job.preview()

        engine, ts, view = self.scene(bumpy_mesh())
        engine.settle()
        high = render(engine, view)
        engine, ts, view = self.scene(make_test_mesh("sphere", segments=96, rings=48))
        engine.settle()
        flat = render(engine, view)
        ts.meshmap.resolution = "2048"
        ts.meshmap.ao_samples = 8
        ts.meshmap.high_poly = str(self.tmp / "high.obj")
        ts.meshmap.cage_front = ts.meshmap.cage_back = 0.06
        engine.bake_start(ts)
        while engine.bake is not None:
            engine.frame()
        engine.settle()
        mapped = render(engine, view)
        for name, image in (("nmap_ai_high.png", high), ("nmap_ai_low_flat.png", flat),
                            ("nmap_ai_low_mapped.png", mapped)):
            Image.fromarray(np.ascontiguousarray(image[:, :, :3])).save(OUT / name)
        covered = flat[:, :, 3] > 250
        c, ratio = bump_agreement(flat, mapped, high, covered, blur=1)
        print()
        print("Arnold：法线贴图的明暗变化和高模起伏的相关 %.3f，幅度比 %.2f" % (c, ratio))
        self.assertGreater(c, 0.7)
        self.assertGreater(ratio, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
