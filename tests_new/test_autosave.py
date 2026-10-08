"""自动保存与恢复：写恢复文件、只补写变了的页、撤销后删掉多余的页、还原成工程文件后和当时的画面一致。
隐藏上下文，不开窗口。"""
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.project import ToolSettings, ViewShading  # noqa: E402
from splender.doc.storage import ProjectFile  # noqa: E402
from splender.engine.autosave import restore  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from splender.engine.projectio import composite_image  # noqa: E402
from test_engine_paint import make_project  # noqa: E402


class AutosaveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="splender_autosave_")
        self.count = 0

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- 工具 ----
    def new_engine(self, resolution="2048"):
        history = History()
        engine = Engine(self.ctx, Preferences(), history)
        project, ts = make_project(resolution=resolution)
        engine.set_project(project)
        view = engine.create_view(ViewShading())
        view.resize(800, 600)
        engine.frame_view(view)
        engine.settle()
        saver = engine.autosave
        saver.new_path = self.next_path
        return engine, history, project, ts, view

    def next_path(self):
        self.count += 1
        return os.path.join(self.tmp, "会话-%d.splender-recover" % self.count)

    def paint(self, engine, view, color, dy, project):
        tools = ToolSettings()
        tools.brush.size = 70
        tools.brush.color = color
        tools.brush.use_height = True
        cx, cy = view.width / 2, view.height / 2
        self.assertTrue(engine.stroke_begin(view, cx - 120, cy + dy, 1.0, time.perf_counter(), tools))
        for i in range(1, 25):
            engine.stroke_move(cx - 120 + i * 10, cy + dy, 1.0, time.perf_counter())
            engine.frame()
        engine.stroke_end()
        engine.settle()
        project.mark_dirty(True)             # 程序里由撤销历史标记，这里没有程序对象

    def autosave(self, engine, project):
        started = engine.autosave.start(project, info={"session": "测试"})
        if not started:
            return None
        result = engine.autosave.wait()
        self.assertIsNotNone(result, "自动保存没做完")
        self.assertNotIn("error", result)
        return result

    @staticmethod
    def image(engine, ts):
        return composite_image(engine, ts, 512).astype(np.float32)

    def reopen(self, path):
        engine = Engine(self.ctx, Preferences(), History())
        project, _meta = engine.load_project(path)
        return engine, project

    def assert_same(self, a, b, what):
        diff = float(np.abs(a - b).max())
        self.assertLessEqual(diff, 1.5 / 255.0, "%s：最大差 %.4f" % (what, diff))

    # ---- 测试 ----
    def test_untitled_incremental_undo_restore(self):
        engine, history, project, ts, view = self.new_engine()
        self.paint(engine, view, (0.9, 0.2, 0.1), -40, project)
        self.paint(engine, view, (0.1, 0.6, 0.3), 40, project)
        first = self.autosave(engine, project)
        self.assertIsNotNone(first)
        self.assertGreater(first["pages"], 0)
        path = engine.autosave.path
        self.assertTrue(os.path.isfile(path))
        print("\n第一次：", first)
        # 什么都没变：不存
        self.assertIsNone(self.autosave(engine, project))
        # 再画一笔：只补写这一笔碰到的页
        self.paint(engine, view, (0.2, 0.3, 0.9), 0, project)
        second = self.autosave(engine, project)
        print("第二次：", second)
        self.assertGreater(second["pages"], 0)
        self.assertLess(second["pages"], first["pages"] + 1)
        self.assertEqual(engine.autosave.path, path, "同一个工程一直用同一个恢复文件")
        with ProjectFile(path, journal="TRUNCATE") as rec:
            pages_before_undo = len(rec.page_keys())
        # 撤销这一笔：这一笔新加的页从恢复文件里删掉，改过的页换回原来的
        history.undo()
        engine.settle()
        third = self.autosave(engine, project)
        print("撤销后：", third)
        self.assertIsNotNone(third)
        reference = self.image(engine, ts)
        with ProjectFile(path, journal="TRUNCATE") as rec:
            pages_after_undo = len(rec.page_keys())
        self.assertLessEqual(pages_after_undo, pages_before_undo)
        # 还原成工程文件，换一个引擎打开，和当时一样
        target = os.path.join(self.tmp, "恢复出来.splender")
        info = restore(path, target)
        self.assertEqual(info["warnings"], [])
        engine2, project2 = self.reopen(target)
        try:
            self.assertEqual([layer.uid for layer in project2.texture_sets[0].layers],
                             [layer.uid for layer in ts.layers])
            self.assert_same(self.image(engine2, project2.texture_sets[0]), reference, "没存过的工程恢复")
        finally:
            engine2.close_storage()
            engine2.release()
        engine.release()
        self.assertTrue(os.path.isfile(path), "引擎关掉时不删恢复文件（留着给下次恢复）")

    def test_based_project_delta(self):
        engine, history, project, ts, view = self.new_engine()
        self.paint(engine, view, (0.9, 0.2, 0.1), -40, project)
        base = os.path.join(self.tmp, "原工程.splender")
        saved = engine.save_project(base)
        self.assertFalse(project.dirty)
        self.assertIsNone(self.autosave(engine, project), "刚存过、没改动：不用自动保存")
        # 改：再画一笔、挪一下模型、删掉一层
        self.paint(engine, view, (0.1, 0.3, 0.9), 30, project)
        obj = project.objects[0]
        delta = np.identity(4)
        delta[:3, 3] = (0.25, 0.0, 0.0)
        engine.transform_object(obj, delta)
        extra = ts.add_layer(type(ts.layers[0])(name="会被删的层"))
        engine.settle()
        total = unchanged = 0
        for layer in ts.layers:
            store = engine.layers.stores.get(layer.uid)
            for plane, mip, tx, ty, page in (store.pages() if store is not None else ()):
                total += 1
                unchanged += int(not page.dirty and page.disk_key == (layer.uid, plane, mip, tx, ty))
        result = self.autosave(engine, project)
        print("\n有底的第一次：", result, "工程文件写了 %d 页；现在共 %d 页，没变的 %d 页" % (saved["pages"], total, unchanged))
        self.assertGreater(unchanged, 0)
        self.assertEqual(result["pages"], total - unchanged, "只写相对工程文件改过的页")
        with ProjectFile(engine.autosave.path, journal="TRUNCATE") as rec:
            self.assertIn("mesh/%d.npz" % obj.uid, rec.list_assets(), "挪过的模型要存")
            meta = rec.read_meta()
        self.assertEqual(meta["recover"]["base"], os.path.abspath(base))
        # 删掉画过的那一层：工程文件里它的页在恢复记录里记成「不要了」
        painted = ts.layers[1]
        engine.complete_strokes(include_active=True)
        store = engine.detach_layer_pixels(painted.uid)
        ts.remove_layer(painted)
        engine.settle()
        result2 = self.autosave(engine, project)
        self.assertIsNotNone(result2)
        with ProjectFile(engine.autosave.path, journal="TRUNCATE") as rec:
            meta = rec.read_meta()
            self.assertFalse([k for k in rec.page_keys() if k[0] == painted.uid], "删掉的层不留页")
        self.assertTrue([k for k in meta["recover"]["deleted"] if k[0] == painted.uid], "工程文件里那层的页记成不要了")
        reference = self.image(engine, ts)
        positions = obj.data.positions.copy()
        target = os.path.join(self.tmp, "原工程（恢复）.splender")
        info = restore(engine.autosave.path, target)
        self.assertEqual(info["warnings"], [])
        with ProjectFile(target) as restored:
            self.assertFalse([k for k in restored.page_keys() if k[0] == painted.uid])
        engine2, project2 = self.reopen(target)
        try:
            ts2 = project2.texture_sets[0]
            self.assertEqual([layer.name for layer in ts2.layers], [layer.name for layer in ts.layers])
            self.assert_same(self.image(engine2, ts2), reference, "有底的工程恢复")
            self.assertTrue(np.allclose(project2.objects[0].data.positions, positions), "模型形状也恢复")
        finally:
            engine2.close_storage()
            engine2.release()
        if store is not None:
            engine.free_layer_pixels(store)
        # 存盘后恢复文件作废
        old = engine.autosave.path
        engine.save_project(base)
        self.assertIsNone(engine.autosave.path)
        engine.autosave.writer.sync()
        self.assertFalse(os.path.exists(old), "存盘后旧的恢复文件删掉")
        self.assertIsNotNone(extra)
        engine.close_storage()
        engine.release()

    def test_meshmaps_read_back_in_strips(self):
        """刚烘好的模型贴图还没读回：缩略图直接读贴图的小一级，不整张读回；自动保存分帧按横条读回再写；
        恢复出来的模型贴图（大小各不相同）和原来一样。"""
        engine, history, project, ts, view = self.new_engine()
        ts.meshmap.resolution = "2048"
        ts.meshmap.smooth_resolution = "1024"
        ts.meshmap.ao_samples = 4
        engine.bake_start(ts)
        started = time.perf_counter()
        while engine.bake is not None and time.perf_counter() - started < 60:
            engine.frame()
        maps = engine.meshmap(ts.uid)
        self.assertIsNotNone(maps)
        self.assertFalse(maps.has_arrays, "刚烘完不读回")
        thumb = maps.preview("ao", 64)
        self.assertEqual(thumb.shape, (64, 64, 3))
        self.assertFalse(maps.has_arrays, "缩略图不用整张读回")
        project.mark_dirty(True)
        result = self.autosave(engine, project)
        self.assertIsNotNone(result)
        self.assertTrue(maps.has_arrays, "自动保存时分帧读回了")
        reference = {key: value.copy() for key, value in maps.read_arrays().items()}
        self.assertEqual(reference["a"].shape[:2], (2048, 2048))
        self.assertEqual(reference["p"].shape[:2], (1024, 1024))
        self.assertTrue(np.array_equal(maps.preview("ao", 64), thumb), "读回前后缩略图一样")
        target = os.path.join(self.tmp, "带模型贴图.splender")
        restore(engine.autosave.path, target)
        engine2, project2 = self.reopen(target)
        try:
            maps2 = engine2.meshmap(project2.texture_sets[0].uid)
            self.assertIsNotNone(maps2)
            for key, value in maps2.read_arrays().items():
                self.assertTrue(np.array_equal(value, reference[key]), key)
        finally:
            engine2.close_storage()
            engine2.release()
        engine.release()

    def test_base_changed_warns(self):
        engine, history, project, ts, view = self.new_engine()
        base = os.path.join(self.tmp, "原.splender")
        engine.save_project(base)
        self.paint(engine, view, (0.5, 0.5, 0.1), 0, project)
        self.autosave(engine, project)
        path = engine.autosave.path
        engine.close_storage()
        engine.release()
        os.utime(base, (time.time() + 100, time.time() + 100))
        info = restore(path, os.path.join(self.tmp, "x.splender"))
        self.assertTrue(info["warnings"], "原工程文件后来改过要提醒")
        os.remove(base)
        info = restore(path, os.path.join(self.tmp, "y.splender"))
        self.assertTrue(any("找不到" in w for w in info["warnings"]))


if __name__ == "__main__":
    unittest.main()
