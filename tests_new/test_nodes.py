"""节点系统：每种节点能求值、生成器无缝平铺、改参数只重算下游、不许成环、撤销、存取；
填充层用节点图当材质，导出的贴图和节点图的输出对得上。隐藏上下文，不开窗口。"""
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
from splender.doc.project import Layer, MeshObject, Project, TextureSet, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from splender.nodes import library  # noqa: E402
from splender.nodes.evaluate import GraphEvaluator  # noqa: E402
from splender.nodes.graph import NodeGraph, default_graph, record  # noqa: E402

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/nodes")
OUT.mkdir(parents=True, exist_ok=True)
GENERATORS = ("NOISE", "CELLS", "BRICKS", "SHAPE", "WAVES", "SCRATCHES")


def bricks_graph(name="砖墙") -> NodeGraph:
    graph = NodeGraph(name)
    graph.resolution = "512"
    bricks = graph.add_node("BRICKS", (-400, 0))
    bricks.params.columns, bricks.params.rows = 2, 4
    colors = graph.add_node("GRADIENT_MAP", (-150, -80))
    colors.params.ramp = ("LINEAR", ((0.0, 0.05, 0.05, 0.05, 1.0), (0.5, 0.6, 0.15, 0.1, 1.0), (1.0, 0.9, 0.3, 0.2, 1.0)))
    output = graph.add_node("OUTPUT", (100, 0))
    graph.link(bricks.uid, colors.uid, 0)
    graph.link(colors.uid, output.uid, 0)
    graph.link(bricks.uid, output.uid, 3)
    return graph


class NodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.ev = GraphEvaluator(cls.ctx)
        cls.tmp = Path(tempfile.mkdtemp())

    def test_every_node_and_tiling(self):
        graph = NodeGraph("全部")
        graph.resolution = "256"
        sheet = []
        for node_type in library.all_types():
            if node_type.id == "OUTPUT":
                continue
            node = graph.add_node(node_type.id)
            data = self.ev.read_node(graph, node.uid)
            self.assertTrue(np.isfinite(data).all(), node_type.id)
            sheet.append(np.clip(data[::-1, :, :3] * 255, 0, 255).astype(np.uint8))
            if node_type.id in GENERATORS:
                g = data[:, :, 0]
                # 接缝两边的差，和图里任何一处相邻两列（行）的差比：能平铺的话不会更大
                seam_u = np.abs(g[:, 0] - g[:, -1]).mean()
                seam_v = np.abs(g[0, :] - g[-1, :]).mean()
                inner_u = np.abs(np.diff(g, axis=1)).mean(axis=0).max()
                inner_v = np.abs(np.diff(g, axis=0)).mean(axis=1).max()
                print("%-10s 左右接缝 %.4f（内部最大 %.4f） 上下接缝 %.4f（内部最大 %.4f）" % (
                    node_type.label, seam_u, inner_u, seam_v, inner_v))
                self.assertLessEqual(seam_u, inner_u * 1.25 + 1e-3, "%s 左右要能接上" % node_type.id)
                self.assertLessEqual(seam_v, inner_v * 1.25 + 1e-3, "%s 上下要能接上" % node_type.id)
        rows = [np.concatenate(sheet[i:i + 7], axis=1) for i in range(0, len(sheet), 7)]
        width = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
        Image.fromarray(np.concatenate(rows, axis=0)).save(OUT / "all_nodes.png")

    def test_cache_and_cycles(self):
        graph = bricks_graph()
        self.ev.output(graph)
        before = self.ev.stat_rendered
        self.ev.output(graph)
        self.assertEqual(self.ev.stat_rendered, before, "没变就不重算")
        colors = next(n for n in graph.nodes.values() if n.type_id == "GRADIENT_MAP")
        colors.params.ramp = ramp.move_stop(colors.params.ramp, 1, 0.3)[0]
        self.ev.output(graph)
        self.assertEqual(self.ev.stat_rendered, before + 1, "只重算改了参数的节点")
        bricks = next(n for n in graph.nodes.values() if n.type_id == "BRICKS")
        self.assertFalse(graph.link(colors.uid, bricks.uid, 0), "砖块没有输入")
        blur = graph.add_node("BLUR")
        graph.link(colors.uid, blur.uid, 0)
        self.assertFalse(graph.link(blur.uid, colors.uid, 0), "不许连成环")
        self.assertFalse(graph.link(blur.uid, blur.uid, 0))

    def test_undo_and_save_load(self):
        history = History()
        graph = bricks_graph()
        before = graph.to_dict()
        noise = graph.add_node("NOISE")
        blend = graph.add_node("BLEND")
        graph.link(noise.uid, blend.uid, 0)
        record(history, "加节点", graph, before)
        self.assertEqual(len(graph.nodes), 5)
        history.undo()
        self.assertEqual(len(graph.nodes), 3)
        history.redo()
        self.assertEqual(len(graph.nodes), 5)
        self.assertIn((noise.uid, blend.uid, 0), graph.links)
        # 存进工程再读回
        project = Project("节点存取")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="1024"))
        ts.add_layer(Layer(name="砖", kind="FILL", fill_graph=graph.uid, graph_tiling=3.0))
        project.add_graph(graph)
        project.add_object(MeshObject("平面", make_test_mesh("plane"), [ts.uid]))
        engine = Engine(self.ctx, Preferences(), History())
        self.addCleanup(engine.release)
        engine.set_project(project)
        engine.settle()
        path = os.path.join(self.tmp, "节点.splender")
        engine.save_project(path)
        engine.close_storage()
        engine2 = Engine(self.ctx, Preferences(), History())
        self.addCleanup(engine2.release)
        loaded, _meta = engine2.load_project(path)
        self.assertEqual(len(loaded.graphs), 1)
        self.assertEqual(loaded.graphs[0].to_dict(), graph.to_dict())
        self.assertEqual(loaded.texture_sets[0].layers[0].fill_graph, graph.uid)
        self.assertAlmostEqual(loaded.texture_sets[0].layers[0].graph_tiling, 3.0)
        engine2.close_storage()

    def test_fill_layer_uses_graph(self):
        graph = bricks_graph()
        project = Project("节点填充")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="2048"))
        layer = Layer(name="砖墙", kind="FILL", fill_graph=graph.uid, graph_tiling=2.0, use_height=True)
        ts.add_layer(layer)
        project.add_graph(graph)
        project.add_object(MeshObject("平面", make_test_mesh("plane"), [ts.uid]))
        engine = Engine(self.ctx, Preferences(), History())
        self.addCleanup(engine.release)
        engine.set_project(project)
        view = engine.create_view(ViewShading(mode="MATERIAL"))
        view.resize(480, 480)
        camera = view.camera
        camera.target = np.zeros(3)
        camera.distance = 2.4
        camera.yaw, camera.pitch = 0.0, 89.0
        view.invalidate_camera()
        engine.settle()
        Image.fromarray(engine.renderer.read_pixels(view)[:, :, :3]).save(OUT / "fill_graph_view.png")
        out = self.tmp / "base.png"
        engine.export_channel(ts, "basecolor", str(out), size=512)
        exported = np.asarray(Image.open(out)).astype(np.float64) / 255.0
        # 节点图的基础色平铺两次，缩到 512：应该和导出的一样
        base = self.ev.read_output(graph)[0, :, :, :3]          # 第 0 行是 v=0
        tiled = np.tile(base[::-1], (2, 2, 1))                  # 图片朝向，平铺两次 → 1024
        expect = tiled.reshape(512, 2, 512, 2, 3).mean(axis=(1, 3))
        diff = np.abs(exported - expect)
        print()
        print("节点图填充：导出和节点图输出的平均差 %.4f，最大差 %.3f" % (diff.mean(), diff.max()))
        self.assertLess(diff.mean(), 0.02)
        # 高度通道跟着节点图的高度
        engine.export_channel(ts, "height", str(self.tmp / "height.png"), size=512)
        height = np.asarray(Image.open(self.tmp / "height.png")).astype(np.float64) / 65535.0 * 2.0 - 1.0
        self.assertGreater(height.max() - height.min(), 0.8, "砖面高、缝低")
        # 改节点参数：显示跟着变
        colors = next(n for n in graph.nodes.values() if n.type_id == "GRADIENT_MAP")
        colors.params.ramp = ramp.set_color(colors.params.ramp, 2, (0.2, 0.4, 0.9))
        engine.settle()
        engine.export_channel(ts, "basecolor", str(out), size=512)
        changed = np.asarray(Image.open(out)).astype(np.float64) / 255.0
        self.assertGreater(changed[:, :, 2].mean(), exported[:, :, 2].mean() + 0.1, "改成偏蓝之后应该更蓝")
        # 删掉节点图：图层回到固定的填充色
        project.remove_graph(graph.uid)
        engine.settle()
        engine.export_channel(ts, "basecolor", str(out), size=512)
        plain = np.asarray(Image.open(out)).astype(np.float64)
        self.assertLess(plain.std(), 1.0, "没有节点图就是一种颜色")


if __name__ == "__main__":
    unittest.main(verbosity=2)
