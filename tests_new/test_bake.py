"""烘焙和生成器蒙版：模型贴图（遮蔽、曲率、厚度、法线、位置）的数值对不对，生成器盖住的地方对不对，
存取一致，大模型的速度。隐藏上下文，不开窗口。缩略图存到 docs/evidence/bake/。"""
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

from splender.bake.baker import MeshMapSet  # noqa: E402
from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet  # noqa: E402
from splender.engine import projectio  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/bake")
OUT.mkdir(parents=True, exist_ok=True)


def make_project(kind, **kw):
    project = Project("烘焙测试")
    ts = project.add_texture_set(TextureSet(name="材质", resolution="2048"))
    ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.5, 0.5, 0.5)))
    mesh = make_test_mesh(kind, **kw)
    project.add_object(MeshObject(mesh.name, mesh, [ts.uid]))
    return project, ts


def bake(engine, ts, budget=50.0, limit=60.0):
    job = engine.bake_start(ts)
    t0 = time.perf_counter()
    while engine.bake is not None and time.perf_counter() - t0 < limit:
        engine.frame()
    assert engine.bake is None, "烘焙超时"
    assert not job.error, job.error
    return job, time.perf_counter() - t0


def save_previews(maps, prefix):
    for kind in ("ao", "curvature", "thickness", "normal", "position"):
        Image.fromarray(maps.preview(kind, 256)).save(OUT / ("%s_%s.png" % (prefix, kind)))


def write_obj(path, positions, normals):
    """逐角点的三角形写成 OBJ（每个角点各自一个顶点和法线）。"""
    with open(path, "w", encoding="utf-8") as f:
        lines = ["v %.6f %.6f %.6f" % tuple(p) for p in positions.reshape(-1, 3)]
        lines += ["vn %.6f %.6f %.6f" % tuple(n) for n in normals.reshape(-1, 3)]
        for t in range(len(positions) // 3):
            a, b, c = 3 * t + 1, 3 * t + 2, 3 * t + 3
            lines.append("f %d//%d %d//%d %d//%d" % (a, a, b, b, c, c))
        f.write("\n".join(lines) + "\n")


def bumpy_sphere(level=96, amount=0.05, frequency=6.0):
    """球面上一圈圈起伏：位置和精确的法线。"""
    from splender.sculpt import topology as tp
    mesh = tp.subdivide(tp.make_sphere(level), 1)
    v = mesh.vertices.astype(np.float64)
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    r = 1.0 + amount * np.sin(frequency * np.arctan2(n[:, 2], n[:, 0])) * np.cos(frequency * np.arcsin(n[:, 1]))
    mesh.vertices = (n * r[:, None]).astype(np.float32)
    positions, normals, _uv, _m = tp.unweld(mesh)
    return positions, normals


class BakeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)

    def make_engine(self, project):
        engine = Engine(self.ctx, Preferences(), History())
        engine.set_project(project)
        self.addCleanup(engine.release)
        return engine

    def test_torus_maps(self):
        project, ts = make_project("torus", segments=96, sides=48)
        ts.meshmap.resolution = "1024"
        ts.meshmap.ao_samples = 48
        ts.meshmap.ao_distance = 0.3
        engine = self.make_engine(project)
        job, seconds = bake(engine, ts)
        maps = engine.meshmap(ts.uid)
        self.assertIsNotNone(maps)
        save_previews(maps, "torus")
        arrays = maps.read_arrays()
        a = arrays["a"].astype(np.float32) / 255.0
        p = arrays["p"].astype(np.float32)
        covered = a[:, :, 3] > 0.9
        radial = np.sqrt(p[:, :, 0] ** 2 + p[:, :, 2] ** 2)
        flat = np.abs(p[:, :, 1]) < 0.05
        inner = covered & flat & (radial < 0.7)
        outer = covered & flat & (radial > 1.3)
        self.assertGreater(inner.sum(), 50)
        self.assertGreater(outer.sum(), 50)
        ao_inner, ao_outer = a[inner, 0].mean(), a[outer, 0].mean()
        print("\n圆环：遮蔽 内圈 %.3f 外圈 %.3f；曲率 外圈 %.3f；用时 %.2f 秒，%s" % (
            ao_inner, ao_outer, a[outer, 1].mean(), seconds, maps.meta["stats"]))
        self.assertLess(ao_inner, ao_outer - 0.1, "内圈应该比外圈暗")
        self.assertGreater(ao_outer, 0.85, "外圈几乎不被遮挡")
        self.assertGreater(a[outer, 1].mean(), 0.55, "外圈是凸的")
        # 法线：外圈朝外
        n = arrays["n"][:, :, :3].astype(np.float32) / 255.0 * 2.0 - 1.0
        outward = np.stack([p[:, :, 0], np.zeros_like(radial), p[:, :, 2]], -1) / np.maximum(radial, 1e-6)[:, :, None]
        self.assertGreater(float((n[outer] * outward[outer]).sum(axis=1).mean()), 0.9)

    def test_thickness(self):
        # 一大一小两个球：小球薄（厚度小）
        project, ts = make_project("sphere", segments=96, rings=48)
        small = make_test_mesh("sphere", segments=96, rings=48, radius=0.2)
        small.positions = small.positions + np.array([1.6, 0.0, 0.0], np.float32)
        small.uvs = small.uvs * 0.3 + 0.7
        big = project.objects[0].data
        big.uvs = big.uvs * 0.65
        project.add_object(MeshObject("小球", small, [ts.uid]))
        ts.meshmap.resolution = "1024"
        ts.meshmap.ao_samples = 32
        ts.meshmap.thickness_distance = 0.5
        engine = self.make_engine(project)
        bake(engine, ts)
        arrays = engine.meshmap(ts.uid).read_arrays()
        a = arrays["a"].astype(np.float32) / 255.0
        p = arrays["p"].astype(np.float32)
        covered = a[:, :, 3] > 0.9
        on_small = covered & (p[:, :, 0] > 1.3)
        on_big = covered & (p[:, :, 0] < 0.9)
        print("\n厚度：小球 %.3f 大球 %.3f" % (a[on_small, 2].mean(), a[on_big, 2].mean()))
        self.assertLess(a[on_small, 2].mean(), a[on_big, 2].mean() - 0.2)

    def test_generator_edges_on_cube(self):
        project, ts = make_project("cube", divisions=16)
        ts.meshmap.resolution = "1024"
        ts.meshmap.ao_samples = 16
        ts.meshmap.curvature_smooth = 1
        engine = self.make_engine(project)
        bake(engine, ts)
        arrays = engine.meshmap(ts.uid).read_arrays()
        save_previews(engine.meshmap(ts.uid), "cube")
        # 扩边：UV 块之间的空隙里，紧挨着块的一圈也有值（标记为扩出来的）
        alpha = arrays["a"][:, :, 3]
        padded = (alpha > 0) & (alpha < 250)
        self.assertGreater(int(padded.sum()), 1000)
        layer = Layer(name="边缘", kind="FILL", fill_color=(1.0, 0.0, 0.0), mask_generator="EDGES", gen_range=0.6,
                      gen_softness=0.1, gen_noise=0.0)
        ts.add_layer(layer)
        path = OUT / "cube_edges_basecolor.png"
        engine.settle()
        projectio.export_channel(engine, ts, "basecolor", str(path), size=1024)
        image = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)[::-1] / 255.0
        p = arrays["p"].astype(np.float32)
        covered = arrays["a"][:, :, 3] > 250
        near_edge = (np.abs(p[:, :, :3]) > 0.97).sum(axis=2) >= 2
        face_center = (np.abs(p[:, :, :3]) < 0.6).sum(axis=2) >= 2
        red = (image[:, :, 0] > 0.8) & (image[:, :, 1] < 0.2)
        edge_red = red[covered & near_edge].mean()
        center_red = red[covered & face_center].mean()
        print("\n立方体边缘生成器：棱上变红 %.2f，面中间变红 %.2f" % (edge_red, center_red))
        self.assertGreater(edge_red, 0.6)
        self.assertLess(center_red, 0.05)
        # 反转之后反过来
        layer.gen_invert = True
        engine.settle()
        projectio.export_channel(engine, ts, "basecolor", str(path), size=1024)
        image = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)[::-1] / 255.0
        red = (image[:, :, 0] > 0.8) & (image[:, :, 1] < 0.2)
        self.assertLess(red[covered & near_edge].mean(), 0.4)
        self.assertGreater(red[covered & face_center].mean(), 0.95)

    def test_facing_and_noise(self):
        project, ts = make_project("sphere", segments=96, rings=48)
        ts.meshmap.resolution = "1024"
        ts.meshmap.ao_samples = 8
        engine = self.make_engine(project)
        bake(engine, ts)
        arrays = engine.meshmap(ts.uid).read_arrays()
        ts.add_layer(Layer(name="积灰", kind="FILL", fill_color=(1.0, 1.0, 1.0), mask_generator="FACING",
                           gen_range=0.3, gen_softness=0.1, gen_noise=0.6, gen_noise_size=0.1))
        path = OUT / "sphere_facing_basecolor.png"
        engine.settle()
        projectio.export_channel(engine, ts, "basecolor", str(path), size=1024)
        image = np.asarray(Image.open(path).convert("L")).astype(np.float32)[::-1] / 255.0
        n = arrays["n"].astype(np.float32) / 255.0 * 2.0 - 1.0
        covered = arrays["a"][:, :, 3] > 250
        top = covered & (n[:, :, 1] > 0.9)
        bottom = covered & (n[:, :, 1] < -0.5)
        white_top = (image[top] > 0.95).mean()
        white_bottom = (image[bottom] > 0.95).mean()
        middle = covered & (np.abs(n[:, :, 1] - 0.4) < 0.1)
        print("\n朝上生成器：顶部变白 %.2f，底部变白 %.2f，过渡带变白 %.2f（有斑驳时应在中间）" % (
            white_top, white_bottom, (image[middle] > 0.95).mean()))
        self.assertGreater(white_top, 0.9)
        self.assertLess(white_bottom, 0.01)
        self.assertTrue(0.05 < (image[middle] > 0.95).mean() < 0.95, "斑驳应让边界参差不齐")

    def test_save_load_roundtrip(self):
        project, ts = make_project("torus")
        ts.meshmap.resolution = "1024"
        ts.meshmap.ao_samples = 8
        engine = self.make_engine(project)
        bake(engine, ts)
        maps = engine.meshmap(ts.uid)
        data = maps.to_bytes()
        again = MeshMapSet.from_bytes(self.ctx, data)
        try:
            for key in ("a", "n", "p"):
                self.assertTrue(np.array_equal(maps.read_arrays()[key], again.read_arrays()[key]))
            self.assertEqual(again.meta["bounds_min"], maps.meta["bounds_min"])
        finally:
            again.release()
        print("\n模型贴图存档 %.1f MB（1024²）" % (len(data) / 1e6))
        # 整个工程存取
        path = OUT / "roundtrip.splender"
        if path.exists():
            path.unlink()
        engine.save_project(str(path))
        engine2 = Engine(self.ctx, Preferences(), History())
        self.addCleanup(engine2.release)
        loaded, _meta = engine2.load_project(str(path))
        engine2.cache.storage.close()
        maps2 = engine2.meshmap(loaded.texture_sets[0].uid)
        self.assertIsNotNone(maps2)
        self.assertTrue(np.array_equal(maps2.read_arrays()["a"], maps.read_arrays()["a"]))

    def test_dense_mesh_speed(self):
        project, ts = make_project("dense_sphere", triangles=1_000_000)
        ts.meshmap.resolution = "2048"
        ts.meshmap.ao_samples = 64
        engine = self.make_engine(project)
        job, seconds = bake(engine, ts, limit=180.0)
        stats = engine.meshmap(ts.uid).meta["stats"]
        print("\n100 万三角形、2048²、64 采样：共 %.2f 秒，%s" % (seconds, stats))
        self.assertLess(seconds, 60.0)

    def test_high_poly(self):
        """从高模烘：同形状的细高模烘出来是平的；起伏的高模烘出的世界法线和高模一致，法线贴图有起伏。"""
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        from splender.sculpt import topology as tp
        same = tp.unweld(tp.subdivide(tp.make_sphere(64), 1))
        write_obj(tmp / "same.obj", same[0], same[1])
        project, ts = make_project("sphere", segments=96, rings=48)
        ts.meshmap.resolution = "1024"
        ts.meshmap.ao_samples = 8
        ts.meshmap.high_poly = str(tmp / "same.obj")
        engine = self.make_engine(project)
        bake(engine, ts)
        maps = engine.meshmap(ts.uid)
        self.assertTrue(maps.has_normal)
        arrays = maps.read_arrays()
        covered = arrays["a"][:, :, 3] > 250
        tn = arrays["t"][:, :, :3].astype(np.float32) / 255.0 * 2.0 - 1.0
        flat_angle = np.degrees(np.arccos(np.clip(tn[covered, 2] / np.linalg.norm(tn[covered], axis=1), -1, 1)))
        print()
        print("同形状高模：法线贴图偏离平面 平均 %.2f°，95%% 分位 %.2f°" % (flat_angle.mean(), np.percentile(flat_angle, 95)))
        self.assertLess(flat_angle.mean(), 3.0)
        positions, normals = bumpy_sphere()
        write_obj(tmp / "bumpy.obj", positions, normals)
        ts.meshmap.high_poly = str(tmp / "bumpy.obj")
        ts.meshmap.cage_front = 0.06
        ts.meshmap.cage_back = 0.06
        bake(engine, ts)
        arrays = engine.meshmap(ts.uid).read_arrays()
        save_previews(engine.meshmap(ts.uid), "highpoly")
        Image.fromarray(engine.meshmap(ts.uid).preview("tangent", 512)).save(OUT / "highpoly_tangent.png")
        covered = arrays["a"][:, :, 3] > 250
        p = arrays["p"][:, :, :3].astype(np.float64)
        n_map = arrays["n"][:, :, :3].astype(np.float64) / 255.0 * 2.0 - 1.0
        n_map /= np.maximum(np.linalg.norm(n_map, axis=2, keepdims=True), 1e-9)
        # 高模在投影点的精确法线：数值求导
        def radius(q):
            q = q / np.linalg.norm(q, axis=-1, keepdims=True)
            return 1.0 + 0.05 * np.sin(6.0 * np.arctan2(q[..., 2], q[..., 0])) * np.cos(6.0 * np.arcsin(np.clip(q[..., 1], -1, 1)))
        pts = p[covered]
        sel = np.random.default_rng(3).choice(len(pts), size=min(4000, len(pts)), replace=False)
        pts = pts[sel]
        u = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        helper = np.where(np.abs(u[:, 1:2]) < 0.9, np.array([[0.0, 1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
        e1 = np.cross(u, helper); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(u, e1)
        h = 1e-4
        def surf(q):
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
            return q * radius(q)[:, None]
        tu = surf(u + e1 * h) - surf(u - e1 * h)
        tv = surf(u + e2 * h) - surf(u - e2 * h)
        exact = np.cross(tu, tv)
        exact /= np.linalg.norm(exact, axis=1, keepdims=True)
        exact *= np.sign(np.sum(exact * u, axis=1, keepdims=True))
        got = n_map[covered][sel]
        angle = np.degrees(np.arccos(np.clip(np.sum(got * exact, axis=1), -1, 1)))
        tn = arrays["t"][:, :, :3].astype(np.float32) / 255.0 * 2.0 - 1.0
        tilt = np.degrees(np.arccos(np.clip(tn[covered, 2] / np.linalg.norm(tn[covered], axis=1), -1, 1)))
        print("起伏高模：世界法线和精确值差 平均 %.2f°，95%% 分位 %.2f°；法线贴图平均倾斜 %.1f°" % (
            angle.mean(), np.percentile(angle, 95), tilt.mean()))
        self.assertLess(np.median(angle), 4.0)
        self.assertGreater(tilt.mean(), 5.0, "起伏的高模应该烘出有起伏的法线贴图")

    def test_cancel_on_geometry_change(self):
        project, ts = make_project("torus")
        engine = self.make_engine(project)
        job = engine.bake_start(ts)
        engine.frame()
        engine.rebuild_object(project.objects[0])
        self.assertIsNone(engine.bake)
        self.assertTrue(job.done and job.error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
