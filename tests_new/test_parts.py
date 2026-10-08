"""部件（ID 贴图）和智能遮罩：烘焙分部件、「部件」生成器只盖住选中的部件、视口点选、高模部件、存取。
隐藏上下文，不开窗口。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from splender.bake.baker import MeshMapSet  # noqa: E402
from splender.bake.parts import format_ids, parse_ids, ray_hit, triangle_parts  # noqa: E402
from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import MeshData, make_test_mesh  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/bake")
OUT.mkdir(parents=True, exist_ok=True)


def two_spheres() -> MeshData:
    """左右两个分开的球，UV 各占一半（左球在 u<0.5，右球在 u>0.5）。"""
    a = make_test_mesh("sphere", segments=48, rings=24)
    pos = []
    uvs = []
    for shift, u0 in ((-1.2, 0.0), (1.2, 0.5)):
        p = np.asarray(a.positions, np.float32).copy()
        p[:, 0] += shift
        pos.append(p)
        uv = np.asarray(a.uvs, np.float32).copy()
        uv[:, 0] = u0 + uv[:, 0] * 0.5
        uvs.append(uv)
    positions = np.concatenate(pos)
    normals = np.concatenate([a.normals, a.normals])
    return MeshData(name="双球", positions=positions, normals=normals, uvs=np.concatenate(uvs),
                    material_ids=np.zeros(len(positions) // 3, np.int32), materials=["材质"],
                    bounds_min=positions.min(axis=0), bounds_max=positions.max(axis=0))


def write_obj(path, positions) -> None:
    lines = ["v %.6f %.6f %.6f" % tuple(p) for p in positions]
    for t in range(len(positions) // 3):
        lines.append("f %d %d %d" % (3 * t + 1, 3 * t + 2, 3 * t + 3))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class PartsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = Path(tempfile.mkdtemp())

    def scene(self, mesh, id_source="PART"):
        project = Project("部件")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="2048"))
        ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.5, 0.5, 0.5), fill_roughness=0.6))
        project.add_object(MeshObject(mesh.name, mesh, [ts.uid]))
        history = History()
        engine = Engine(self.ctx, Preferences(), history)
        self.addCleanup(engine.release)
        engine.set_project(project)
        ts.meshmap.resolution = "1024"
        ts.meshmap.ao_samples = 4
        ts.meshmap.id_source = id_source
        return engine, ts, history

    def bake(self, engine, ts):
        engine.bake_start(ts)
        while engine.bake is not None:
            engine.frame()
        engine.settle()
        maps = engine.meshmap(ts.uid)
        self.assertIsNotNone(maps, "烘焙应该成功")
        return maps

    def test_triangle_parts(self):
        mesh = two_spheres()
        ids, count = triangle_parts(mesh.positions, mesh.uvs, "PART")
        self.assertEqual(count, 2)
        half = len(ids) // 2
        self.assertEqual(len(set(ids[:half].tolist())), 1)
        self.assertNotEqual(ids[0], ids[-1])
        cube = make_test_mesh("cube")
        _ids, islands = triangle_parts(cube.positions, cube.uvs, "UV_ISLAND")
        _ids, parts = triangle_parts(cube.positions, cube.uvs, "PART")
        print()
        print("立方体：网格部件 %d 个，UV 岛 %d 个" % (parts, islands))
        self.assertEqual(parts, 1)
        self.assertGreater(islands, 1)
        self.assertEqual(parse_ids("3, 7，x,3"), [3, 7])
        self.assertEqual(format_ids([7, 3, 3]), "3,7")
        tri, _u, _v, t = ray_hit(mesh.positions, (1.2, 0.0, 5.0), (0.0, 0.0, -1.0))
        self.assertGreaterEqual(tri, half, "打中右边的球")
        self.assertAlmostEqual(t, 4.0, delta=0.05)

    def test_bake_generator_pick(self):
        engine, ts, history = self.scene(two_spheres())
        maps = self.bake(engine, ts)
        self.assertEqual(maps.part_count, 2)
        ids = maps.read_arrays()["i"]
        covered = maps.read_arrays()["a"][:, :, 3] == 255
        half = ids.shape[1] // 2
        left = np.unique(np.rint(ids[:, :half][covered[:, :half]]))
        right = np.unique(np.rint(ids[:, half:][covered[:, half:]]))
        print()
        print("ID 贴图：左半边部件 %s，右半边部件 %s" % (left.tolist(), right.tolist()))
        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertNotEqual(left[0], right[0])
        Image.fromarray(maps.preview("parts", 256)).save(OUT / "parts_two_spheres.png")
        # 视口点选：右边的球
        view = engine.create_view(ViewShading(mode="MATERIAL", hdri="interior"))
        view.resize(640, 320)
        camera = view.camera
        camera.target = np.zeros(3)
        camera.distance = 5.5
        camera.yaw, camera.pitch = 0.0, 0.0
        view.invalidate_camera()
        engine.settle()
        found = engine.part_at(view, 480, 160)
        self.assertIsNotNone(found)
        self.assertIs(found[0], ts)
        self.assertEqual(found[1], int(right[0]))
        self.assertEqual(engine.part_at(view, 160, 160)[1], int(left[0]))
        self.assertIsNone(engine.part_at(view, 320, 10), "点到空白处")
        # 「部件」生成器：红色填充只盖住右边的球
        red = Layer(name="红", kind="FILL", fill_color=(0.9, 0.1, 0.1), fill_roughness=0.5)
        ts.add_layer(red)
        red.mask_generator = "PARTS"
        red.gen_parts = format_ids([found[1]])
        engine.settle()
        image = engine.renderer.read_pixels(view).astype(int)
        Image.fromarray(image[:, :, :3].astype(np.uint8)).save(OUT / "parts_generator.png")
        redness = image[:, :, 0] - image[:, :, 1]
        right_red = redness[110:210, 430:530].mean()
        left_red = redness[110:210, 110:210].mean()
        print("部件生成器：右球偏红 %.1f，左球偏红 %.1f" % (right_red, left_red))
        self.assertGreater(right_red, 60)
        self.assertLess(abs(left_red), 10)
        red.gen_invert = True
        engine.settle()
        image = engine.renderer.read_pixels(view).astype(int)
        redness = image[:, :, 0] - image[:, :, 1]
        self.assertGreater(redness[110:210, 110:210].mean(), 60, "反转后盖住左球")
        self.assertLess(abs(redness[110:210, 430:530].mean()), 10)
        # 存取：部件编号跟着模型贴图存下来
        again = MeshMapSet.from_bytes(self.ctx, maps.to_bytes())
        self.addCleanup(again.release)
        self.assertEqual(again.part_count, 2)
        self.assertTrue(np.array_equal(again.read_arrays()["i"], maps.read_arrays()["i"]))

    def test_high_poly_parts(self):
        """高模分成上下两块（中间错开一点），低模是一个整球：按高模部件分出两块。"""
        sphere = make_test_mesh("sphere", segments=64, rings=32)
        pos = np.asarray(sphere.positions, np.float32).reshape(-1, 3, 3).copy()
        top = pos[:, :, 1].mean(axis=1) > 0.0
        pos[top, :, 1] += 0.01
        write_obj(self.tmp / "split.obj", pos.reshape(-1, 3))
        engine, ts, _history = self.scene(make_test_mesh("sphere", segments=64, rings=32), id_source="HIGH_POLY")
        ts.meshmap.high_poly = str(self.tmp / "split.obj")
        ts.meshmap.cage_front = ts.meshmap.cage_back = 0.05
        maps = self.bake(engine, ts)
        self.assertEqual(maps.part_count, 2)
        arrays = maps.read_arrays()
        ids = np.rint(arrays["i"])
        y = arrays["p"][:, :, 1].astype(np.float32)
        covered = arrays["a"][:, :, 3] == 255
        upper = np.unique(ids[covered & (y > 0.1)])
        lower = np.unique(ids[covered & (y < -0.1)])
        print()
        print("高模部件：上半 %s，下半 %s" % (upper.tolist(), lower.tolist()))
        self.assertEqual(len(upper), 1)
        self.assertEqual(len(lower), 1)
        self.assertNotEqual(upper[0], lower[0])

    def test_smart_mask_undo(self):
        from splender.bake.smart import SMART_MASKS, get_mask
        from splender.core import ops
        import splender.ops.layer_ops  # noqa: F401

        self.assertGreaterEqual(len(SMART_MASKS), 10)
        self.assertEqual(len({m["id"] for m in SMART_MASKS}), len(SMART_MASKS))
        engine, ts, history = self.scene(two_spheres())
        layer = Layer(name="灰", kind="FILL", fill_color=(0.3, 0.3, 0.3))
        ts.add_layer(layer)

        class _App:
            pass

        class _Ctx:
            pass

        app = _App()
        app.history = history
        app.engine = engine
        ctx = _Ctx()
        ctx.app = app
        ctx.texture_set = ts
        ctx.wm = None
        result = ops.call("layer.apply_smart_mask", ctx, mask="crevice_dust")
        self.assertEqual(result, "FINISHED")
        preset = get_mask("crevice_dust")["values"]
        self.assertEqual(layer.mask_generator, "CAVITY")
        self.assertAlmostEqual(layer.gen_range, preset["gen_range"])
        history.undo()
        self.assertEqual(layer.mask_generator, "NONE")
        history.redo()
        self.assertEqual(layer.mask_generator, "CAVITY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
