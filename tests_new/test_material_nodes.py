"""材质节点（照 Blender 着色器节点）：每种节点都能编进合成器的着色器；图层堆栈经过节点再输出，导出的贴图跟着变；
改数值不重新编译；色带、噪波、节点图纹理有效；存取、撤销。隐藏上下文，不开窗口。"""
import os
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

from splender.core import ramp  # noqa: E402
from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from splender.shading import nodes  # noqa: E402
from splender.shading.compile import compile_graph, srgb_to_linear  # noqa: E402
from splender.shading.graph import ShaderGraph  # noqa: E402

FILL = (0.8, 0.3, 0.1)


def to_srgb(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0.0), 1.0 / 2.4) - 0.055)


class MaterialNodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = Path(tempfile.mkdtemp())

    def setUp(self):
        project = Project("材质节点")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="1024"))
        ts.add_layer(Layer(name="填充", kind="FILL", fill_color=FILL, use_basecolor=True))
        project.add_object(MeshObject("平面", make_test_mesh("plane"), [ts.uid]))
        nodes.set_project_getter(lambda: project)
        self.project, self.ts = project, ts
        self.engine = Engine(self.ctx, Preferences(), History())
        self.addCleanup(self.engine.release)
        self.engine.set_project(project)
        self.engine.settle()

    def export(self, channel="basecolor", size=256) -> np.ndarray:
        out = self.tmp / ("%s.png" % channel)
        self.engine.export_channel(self.ts, channel, str(out), size=size)
        image = np.asarray(Image.open(out)).astype(np.float64)
        return image / (65535.0 if image.max() > 255 else 255.0)

    # ------------------------------------------------------------------
    def test_every_node_compiles(self):
        """把每种节点都放进一张图、尽量都接上，编进合成器的着色器。"""
        graph = ShaderGraph()
        out = graph.add_node("OUTPUT_MATERIAL")
        previous_color = graph.add_node("LAYER_STACK").uid
        for node_type in nodes.all_types():
            if node_type.id in ("OUTPUT_MATERIAL", "LAYER_STACK"):
                continue
            node = graph.add_node(node_type.id)
            # 上一个的颜色（或第一个输出）接到这个的第一个输入，让所有节点都在输出的上游
            ins = [i[0] for i in node_type.inputs]
            if ins:
                src = graph.nodes[previous_color]
                graph.link(previous_color, src.output_names()[0], node.uid, ins[0])
            if node_type.outputs:
                previous_color = node.uid
        graph.link(previous_color, graph.nodes[previous_color].output_names()[0], out.uid, "base_color")
        graph.set_active(out.uid)
        compiled = compile_graph(graph)
        self.assertIsNotNone(compiled)
        used = {graph.nodes[uid].type_id for uid in graph.nodes}
        self.assertEqual(len(used), len(nodes.all_types()))
        values = compiled.values(graph, lambda uid: -1)
        compositor = self.engine.compositor
        self.assertTrue(compositor.set_material(compiled, values), "所有节点编进去的着色器要能编译")
        # 每种节点也单独编一次（各自的代码路径都走到）
        for node_type in nodes.all_types():
            if node_type.id in ("OUTPUT_MATERIAL", "REROUTE"):
                continue
            g = ShaderGraph()
            n = g.add_node(node_type.id)
            o = g.add_node("OUTPUT_MATERIAL")
            if n.output_names():
                g.link(n.uid, n.output_names()[0], o.uid, "base_color")
            c = compile_graph(g)
            self.assertTrue(compositor.set_material(c, c.values(g, lambda uid: -1)), node_type.id)
        compositor.set_material(None, None)

    def test_layer_stack_through_nodes(self):
        base = self.export()[:, :, :3].mean(axis=(0, 1))
        np.testing.assert_allclose(base, FILL, atol=0.01)
        # 默认材质：图层堆栈直接接输出，和没有材质节点一样
        self.ts.ensure_material()
        self.engine.settle()
        np.testing.assert_allclose(self.export()[:, :, :3].mean(axis=(0, 1)), FILL, atol=0.01)
        # 中间插一个反相（线性颜色里反相，和 Blender 一样）
        graph = self.ts.material
        stack = next(n for n in graph.nodes.values() if n.type_id == "LAYER_STACK")
        out = graph.output_node()
        invert = graph.add_node("INVERT")
        graph.link(stack.uid, "base_color", invert.uid, "color")
        graph.link(invert.uid, "color", out.uid, "base_color")
        self.engine.settle()
        expect = to_srgb(1.0 - srgb_to_linear(FILL))
        got = self.export()[:, :, :3].mean(axis=(0, 1))
        np.testing.assert_allclose(got, expect, atol=0.01)
        # 改数值（反相系数）不重新编译，结果按比例变
        programs = len(self.engine.compositor.material_programs)
        invert.values.in_fac = 0.5
        self.engine.settle()
        expect_half = to_srgb(srgb_to_linear(FILL) * 0.5 + (1.0 - srgb_to_linear(FILL)) * 0.5)
        np.testing.assert_allclose(self.export()[:, :, :3].mean(axis=(0, 1)), expect_half, atol=0.01)
        self.assertEqual(len(self.engine.compositor.material_programs), programs)
        # 静音反相：结果回到原色
        graph.set_mute(invert.uid, True)
        self.engine.settle()
        np.testing.assert_allclose(self.export()[:, :, :3].mean(axis=(0, 1)), FILL, atol=0.01)
        # 粗糙度从数值节点来
        graph.set_mute(invert.uid, False)
        value = graph.add_node("VALUE")
        value.values.value = 0.9
        graph.link(value.uid, "value", out.uid, "roughness")
        self.engine.settle()
        mrao = self.export("roughness")
        self.assertAlmostEqual(float(mrao.mean()), 0.9, delta=0.01)

    def test_color_ramp_and_noise(self):
        graph = self.ts.ensure_material()
        out = graph.output_node()
        noise = graph.add_node("TEX_NOISE")
        noise.values.in_scale = 8.0
        noise.values.in_detail = 3.0
        colors = graph.add_node("VALTORGB")
        colors.values.ramp = ("LINEAR", ((0.3, 0.0, 0.0, 1.0, 1.0), (0.7, 1.0, 1.0, 0.0, 1.0)))
        graph.link(noise.uid, "fac", colors.uid, "fac")
        graph.link(colors.uid, "color", out.uid, "base_color")
        self.engine.settle()
        image = self.export(size=512)[:, :, :3]
        Image.fromarray((image * 255).astype(np.uint8)).save(self.tmp / "noise_ramp.png")
        # 色带两头是蓝和黄：既有偏蓝的地方也有偏黄的地方，中间有渐变
        blue = image[:, :, 2] - image[:, :, 0]
        self.assertGreater(blue.max(), 0.5)
        self.assertLess(blue.min(), -0.5)
        self.assertGreater(image.std(), 0.1)
        # 换成常值插值：只剩两种颜色
        colors.values.ramp = ("CONSTANT", colors.values.ramp[1])
        self.engine.settle()
        image = self.export(size=256)[:, :, :3]
        distinct = np.unique((image * 10).round().reshape(-1, 3), axis=0)
        self.assertLessEqual(len(distinct), 4)

    def test_graph_texture_node(self):
        from splender.nodes.graph import NodeGraph

        tex = NodeGraph("棋盘")
        tex.resolution = "256"
        shape = tex.add_node("COLOR", (-200, 0))
        shape.params.color = (0.1, 0.7, 0.2)
        output = tex.add_node("OUTPUT", (100, 0))
        tex.link(shape.uid, output.uid, 0)
        self.project.add_graph(tex)
        graph = self.ts.ensure_material()
        node = graph.add_node("TEX_GRAPH")
        node.values.graph = str(tex.uid)
        graph.link(node.uid, "color", graph.output_node().uid, "base_color")
        self.engine.settle()
        np.testing.assert_allclose(self.export()[:, :, :3].mean(axis=(0, 1)), (0.1, 0.7, 0.2), atol=0.02)

    def test_save_load_roundtrip(self):
        graph = self.ts.ensure_material()
        mix = graph.add_node("MIX_RGB", (10.0, 20.0))
        mix.values.blend_type = "MULTIPLY"
        mix.values.in_b = (0.5, 0.25, 1.0)
        stack = next(n for n in graph.nodes.values() if n.type_id == "LAYER_STACK")
        graph.link(stack.uid, "base_color", mix.uid, "a")
        graph.link(mix.uid, "result", graph.output_node().uid, "base_color")
        mix.values.in_fac = 1.0
        self.engine.settle()
        before = self.export()[:, :, :3].mean(axis=(0, 1))
        path = str(self.tmp / "mat.splender")
        self.engine.save_project(path)
        other = Engine(self.ctx, Preferences(), History())
        self.addCleanup(other.release)
        project, _meta = other.load_project(path)
        other.set_project(project)
        other.settle()
        ts = project.texture_sets[0]
        self.assertIsNotNone(ts.material)
        loaded = next(n for n in ts.material.nodes.values() if n.type_id == "MIX_RGB")
        self.assertEqual(loaded.values.blend_type, "MULTIPLY")
        self.assertEqual(len(ts.material.links), len(graph.links))
        out = self.tmp / "loaded.png"
        other.export_channel(ts, "basecolor", str(out), size=256)
        after = (np.asarray(Image.open(out)).astype(np.float64) / 255.0)[:, :, :3].mean(axis=(0, 1))
        np.testing.assert_allclose(after, before, atol=0.01)
        # 撤销用的整图快照：换回去结构和数值都回来
        snap = graph.to_dict()
        graph.remove_node(mix.uid)
        self.assertNotIn(mix.uid, graph.nodes)
        graph.load(snap)
        self.assertEqual(graph.nodes[mix.uid].values.blend_type, "MULTIPLY")

    def test_ramp_math_matches_python(self):
        """着色器里的色带和 Python 里的算法一致（四种插值）。"""
        value = ("CARDINAL", ((0.0, 0.0, 0.0, 0.0, 1.0), (0.4, 1.0, 0.2, 0.0, 1.0), (1.0, 0.1, 0.9, 1.0, 1.0)))
        graph = self.ts.ensure_material()
        coords = graph.add_node("TEX_COORD")
        sep = graph.add_node("SEPXYZ")
        colors = graph.add_node("VALTORGB")
        graph.link(coords.uid, "uv", sep.uid, "vector")
        graph.link(sep.uid, "x", colors.uid, "fac")
        graph.link(colors.uid, "color", graph.output_node().uid, "base_color")
        for mode in ("LINEAR", "EASE", "CARDINAL", "B_SPLINE"):
            colors.values.ramp = (mode, value[1])
            self.engine.settle()
            image = self.export(size=256)[:, :, :3]
            row = image[128]                                    # 一行：u 从 0 到 1
            u = (np.arange(256) + 0.5) / 256.0
            expect = ramp.evaluate((mode, value[1]), u)[:, :3]
            # 导出按 sRGB 存：色带的颜色本来就是 sRGB，节点里转线性、导出再转回来
            diff = np.abs(row - expect)
            self.assertLess(diff.mean(), 0.01, mode)


if __name__ == "__main__":
    unittest.main(verbosity=2)
