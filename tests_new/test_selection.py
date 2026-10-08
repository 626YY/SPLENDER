"""选区（engine/selection.py）：新选区、添加、减去、交叉、反选、全选、取消和重新选择、羽化扩展收缩边界、撤销重做；
有选区时滤镜和笔刷只改选区里面；只留、清掉选区里的；蒙版和选区互转；上面各级和第 0 级对得上。
隐藏上下文、512² 贴图（2×2 格），一块平面。不开窗口。"""
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
from splender.doc.project import PLANE_COLOR, PLANE_MASK, Layer, MeshObject, Project, TextureSet  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from splender.engine.pagepool import PAGE  # noqa: E402
from splender.engine.stamp import DAB_DTYPE, StrokeBuffer  # noqa: E402

RES = 512
GRID = RES // PAGE


class SelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        project = Project("选区")
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
        self.engine.drop_selection(self.ts)
        self.engine.layers.drop_layer(self.layer.uid)
        self.layer.has_mask = False
        self.layer.mask_default = 1.0
        self.sel = self.engine.selection(self.ts)

    # ---- 工具 ----
    def plane_image(self, plane_store, mip: int, default: float, channels: int = 1, dtype=np.uint8) -> np.ndarray:
        """一种页在第 mip 级的样子 (边长, 边长, 通道)，0..1；缺页的格按 default。"""
        side = RES >> mip
        out = np.full((side, side, channels), default, np.float32)
        if plane_store is None:
            return out
        level = plane_store.levels[mip]
        for (x, y), page in level.pages.items():
            raw = np.frombuffer(self.engine.cache.page_bytes(page), dtype).reshape(PAGE, PAGE, channels)
            tile = raw.astype(np.float32) / (255.0 if dtype == np.uint8 else 1.0)
            out[y * PAGE:(y + 1) * PAGE, x * PAGE:(x + 1) * PAGE] = tile[:side - y * PAGE, :side - x * PAGE]
        return out

    def selection_image(self, mip: int = 0) -> np.ndarray:
        return self.plane_image(self.sel.plane(), mip, self.sel.default)[..., 0]

    def color(self) -> np.ndarray:
        store = self.engine.layers.stores.get(self.layer.uid)
        plane = store.planes.get(PLANE_COLOR) if store is not None else None
        return self.plane_image(plane, 0, 0.0, 4)

    def rect(self, mode: str, rect, label="框选") -> bool:
        """按 UV 矩形（u0, v0, u1, v1）合一个实心的矩形进选区。"""
        mask = np.full((16, 16), 255, np.uint8)
        return self.sel.combine(mode, "IMAGE", {"image": mask, "rect": rect, "opacity": 1.0}, label)

    def fill(self, rgb=(1.0, 0.0, 0.0)) -> None:
        """整层盖一种颜色（走滤镜写回，有选区时按选区）。"""
        mask = np.full((4, 4), 255, np.uint8)
        self.engine.apply_filter(self.ts, self.layer, "IMAGE",
                                 {"image": mask, "rect": (0.0, 0.0, 1.0, 1.0), "opacity": 1.0, "color2": (*rgb, 1.0)},
                                 {PLANE_COLOR: (True,)}, "填充")

    def assert_mips(self) -> None:
        """第 1 级的每个纹素等于第 0 级对应 2×2 的平均（差一点取整误差）。"""
        level0 = self.selection_image(0)
        level1 = self.selection_image(1)
        expected = level0.reshape(RES // 2, 2, RES // 2, 2).mean(axis=(1, 3))
        self.assertLess(float(np.abs(level1 - expected).max()), 3.0 / 255.0)

    # ---- 合进选区 ----
    def test_set_add_subtract_intersect(self):
        self.assertTrue(self.rect("SET", (0.1, 0.1, 0.5, 0.5)))
        self.assertTrue(self.sel.active)
        image = self.selection_image()
        self.assertGreater(image[150, 150], 0.99)
        self.assertLess(image[400, 400], 0.01)
        self.assert_mips()
        self.rect("ADD", (0.6, 0.6, 0.9, 0.9), "添加")
        image = self.selection_image()
        self.assertGreater(image[150, 150], 0.99)
        self.assertGreater(image[380, 380], 0.99)
        self.rect("SUBTRACT", (0.2, 0.2, 0.3, 0.3), "减去")
        image = self.selection_image()
        self.assertLess(image[128, 128], 0.01, "减去的洞")
        self.assertGreater(image[200, 200], 0.99)
        self.rect("INTERSECT", (0.4, 0.4, 0.7, 0.7), "交叉")
        image = self.selection_image()
        self.assertGreater(image[230, 230], 0.99, "两块重叠的地方留下")
        self.assertGreater(image[330, 330], 0.99)
        self.assertLess(image[150, 150], 0.01, "原来的选区在交叉矩形外的部分去掉")
        self.assertLess(image[440, 440], 0.01)
        self.assertEqual(self.sel.default, 0.0)
        self.assert_mips()

    def test_invert_all_none_reselect(self):
        self.rect("SET", (0.0, 0.0, 0.5, 1.0))
        self.assertTrue(self.sel.invert())
        self.assertEqual(self.sel.default, 1.0)
        image = self.selection_image()
        self.assertLess(image[100, 50], 0.01)
        self.assertGreater(image[100, 400], 0.99)
        self.assert_mips()
        self.assertTrue(self.sel.select_all())
        self.assertEqual(self.sel.default, 1.0)
        self.assertFalse(self.sel.cells())
        self.assertTrue(self.sel.deselect())
        self.assertFalse(self.sel.active)
        self.assertIsNone(self.engine.active_selection(self.ts))
        self.assertTrue(self.sel.reselect())
        self.assertTrue(self.sel.active)

    def test_empty_deselects(self):
        """全选后反选：什么都没选上，自动取消选择（画笔不会哪里都画不上）。撤销回到全选。"""
        self.sel.select_all()
        self.assertTrue(self.sel.invert())
        self.assertFalse(self.sel.active)
        self.assertTrue(self.sel.emptied)
        self.history.undo()
        self.assertTrue(self.sel.active)
        self.assertEqual(self.sel.default, 1.0)

    def test_modify(self):
        self.rect("SET", (0.25, 0.25, 0.75, 0.75))
        before = self.selection_image()
        self.assertTrue(self.sel.modify("EXPAND", 20))
        grown = self.selection_image()
        self.assertGreater(grown[118, 256], 0.99, "扩展：边外 10 个纹素也选上了")
        self.assertLess(before[118, 256], 0.01)
        self.history.undo()
        self.assertTrue(self.sel.modify("CONTRACT", 20))
        shrunk = self.selection_image()
        self.assertLess(shrunk[138, 256], 0.01, "收缩：边内 10 个纹素去掉了")
        self.history.undo()
        self.assertTrue(self.sel.modify("FEATHER", 8))
        soft = self.selection_image()
        self.assertTrue(0.2 < soft[128, 256] < 0.8, "羽化：边上是半选")
        self.history.undo()
        self.assertTrue(self.sel.modify("BORDER", 16))
        ring = self.selection_image()
        self.assertGreater(ring[128, 256], 0.7, "边界：边上选上")
        self.assertLess(ring[256, 256], 0.05, "边界：中间不选")
        self.assertLess(ring[20, 20], 0.05)
        self.assert_mips()

    def test_undo_redo(self):
        self.rect("SET", (0.1, 0.1, 0.4, 0.4))
        first = self.selection_image()
        self.rect("SET", (0.6, 0.6, 0.9, 0.9))
        self.history.undo()
        self.assertTrue(np.array_equal(self.selection_image(), first), "撤销回到第一块")
        self.history.undo()
        self.assertFalse(self.sel.active, "再撤销就没有选区了")
        self.history.redo()
        self.history.redo()
        self.assertGreater(self.selection_image()[400, 400], 0.99)
        self.sel.deselect()
        self.history.undo()
        self.assertTrue(self.sel.active, "撤销取消选择")

    # ---- 有选区时只改选区里面 ----
    def test_filter_only_inside(self):
        self.rect("SET", (0.0, 0.0, 0.5, 0.5))
        self.fill((1.0, 0.0, 0.0))
        color = self.color()
        self.assertTrue(np.allclose(color[100, 100], (1.0, 0.0, 0.0, 1.0), atol=2 / 255))
        self.assertLess(float(color[400, 400, 3]), 0.01, "选区外不动")
        # 羽化过的边按选区值过渡
        self.sel.modify("FEATHER", 12)
        self.engine.layers.drop_layer(self.layer.uid)
        self.fill((0.0, 1.0, 0.0))
        edge = self.color()[256, 100]
        self.assertTrue(0.2 < float(edge[3]) < 0.8, edge)
        # 取消选择后哪里都能改
        self.sel.deselect()
        self.fill((0.0, 0.0, 1.0))
        self.assertGreater(float(self.color()[400, 400, 3]), 0.99)

    def test_brush_stamp_only_inside(self):
        self.rect("SET", (0.0, 0.0, 0.5, 1.0))
        state = self.engine.sets[self.ts.uid]
        buffer = StrokeBuffer(self.ctx, GRID)
        try:
            dabs = np.zeros(1, DAB_DTYPE)
            dabs["center_radius"][0] = (0.5, 0.5, 0.0, 0.3)       # UV 视图里的一笔，跨过选区的边
            dabs["normal_hardness"][0] = (0.0, 0.0, 1.0, 1.0)
            dabs["params"][0] = (1.0, -2.0, 1.0, 0.0)
            tiles = self.engine.stamper.stamp(state.set_gpus[0], buffer, dabs, False, self.sel)
            self.assertTrue(len(tiles))
            coverage = self.plane_image(buffer.store, 0, 0.0, 2, np.float16)[..., 0]
            self.assertGreater(float(coverage[256, 200]), 0.99, "选区里画上了")
            self.assertLess(float(coverage[256, 300]), 0.01, "选区外没画上")
        finally:
            buffer.release(self.engine.cache)

    def test_keep_and_clear(self):
        self.fill((1.0, 0.0, 0.0))
        self.rect("SET", (0.0, 0.0, 0.5, 0.5))
        self.engine.apply_filter(self.ts, self.layer, "SEL_CLEAR", {}, {PLANE_COLOR: (True,)}, "清除")
        color = self.color()
        self.assertLess(float(color[100, 100, 3]), 0.01, "选区里清掉了")
        self.assertGreater(float(color[400, 400, 3]), 0.99, "选区外留着")
        self.sel.invert()
        self.engine.apply_filter(self.ts, self.layer, "SEL_KEEP", {}, {PLANE_COLOR: (True,)}, "只留")
        color = self.color()
        self.assertGreater(float(color[400, 400, 3]), 0.99)

    def test_mask_round_trip(self):
        self.rect("SET", (0.5, 0.5, 1.0, 1.0))
        self.assertTrue(self.sel.to_mask(self.layer))
        self.assertTrue(self.layer.has_mask)
        self.assertEqual(self.layer.mask_default, 0.0)
        store = self.engine.layers.stores[self.layer.uid]
        mask = self.plane_image(store.planes[PLANE_MASK], 0, self.layer.mask_default)[..., 0]
        self.assertGreater(mask[400, 400], 0.99)
        self.assertLess(mask[100, 100], 0.01)
        self.history.undo()
        self.assertFalse(self.layer.has_mask)
        self.history.redo()
        self.sel.deselect()
        self.assertTrue(self.sel.load_mask(self.layer))
        self.assertTrue(self.sel.active)
        self.assertGreater(self.selection_image()[400, 400], 0.99)

    def test_mask_mips_use_default(self):
        """蒙版初始值是白时，只在一格里画过：上一级没画过的象限还是白的（以前填成黑的）。"""
        self.layer.has_mask = True
        self.layer.mask_default = 1.0
        mask = np.full((8, 8), 255, np.uint8)
        self.engine.apply_filter(self.ts, self.layer, "IMAGE",
                                 {"image": mask, "rect": (0.0, 0.0, 0.25, 0.25), "opacity": 1.0,
                                  "color2": (0.0, 0.0, 0.0, 1.0)}, {PLANE_MASK: (True,)}, "画蒙版")
        store = self.engine.layers.stores[self.layer.uid].planes[PLANE_MASK]
        level1 = self.plane_image(store, 1, 1.0)[..., 0]
        self.assertLess(float(level1[10, 10]), 0.05, "画黑的那块")
        self.assertGreater(float(level1[200, 200]), 0.99, "没画过的地方还是白")


if __name__ == "__main__":
    unittest.main()
