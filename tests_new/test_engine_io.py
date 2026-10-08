"""引擎的保存、打开、导出。隐藏上下文，不开窗口。"""
import os
import shutil
import sys
from pathlib import Path
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.project import ToolSettings, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from tests_new.test_engine_paint import make_project  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


class IoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = tempfile.mkdtemp(prefix="splender_io_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def paint(self, engine, view, color, dy):
        tools = ToolSettings()
        tools.brush.size = 80
        tools.brush.color = color
        tools.brush.use_height = True
        cx, cy = view.width / 2, view.height / 2
        ok = engine.stroke_begin(view, cx - 110, cy + dy, 1.0, time.perf_counter(), tools)
        self.assertTrue(ok, "落笔失败：%r 当前层=%r" % (engine.stroke_reject, engine.project.active_texture_set.active_layer))
        for i in range(1, 23):
            engine.stroke_move(cx - 110 + i * 10, cy + dy, 1.0, time.perf_counter())
            engine.frame()
        engine.stroke_end()
        engine.settle()

    def test_save_load_export(self):
        path = os.path.join(self.tmp, "测试 工程.splender")
        history = History()
        engine = Engine(self.ctx, Preferences(), history)
        project, ts = make_project(resolution="4096")
        engine.set_project(project)
        view = engine.create_view(ViewShading())
        view.resize(960, 600)
        engine.frame_view(view)
        engine.settle()
        self.paint(engine, view, (0.9, 0.2, 0.1), -40)
        self.paint(engine, view, (0.1, 0.6, 0.3), 50)
        before = engine.renderer.read_pixels(view).copy()
        info = engine.save_project(path, {"ui": {"hello": 1}})
        print("\n保存:", info, "文件大小 %.1f KB" % (os.path.getsize(path) / 1024))
        self.assertFalse(project.dirty)
        # 再画一笔后保存：只写改过的页
        self.paint(engine, view, (0.2, 0.3, 0.9), 0)
        after_third = engine.renderer.read_pixels(view).copy()
        info2 = engine.save_project(path)
        print("增量保存:", info2)
        self.assertLess(info2["pages"], info["pages"] + 400)
        # 撤销最后一笔：旧页还在内存，画面回到两笔
        history.undo()
        engine.settle()
        self.assertLess(np.abs(engine.renderer.read_pixels(view).astype(int) - before.astype(int)).max(), 4)
        history.redo()
        engine.settle()
        # 导出
        out_color = os.path.join(self.tmp, "basecolor.png")
        out_height = os.path.join(self.tmp, "height.png")
        r1 = engine.export_channel(ts, "basecolor", out_color)
        r2 = engine.export_channel(ts, "height", out_height)
        print("导出:", {k: round(v, 3) if isinstance(v, float) else v for k, v in r1.items()}, round(r2["seconds"], 3))
        color = np.asarray(Image.open(out_color))
        height = np.asarray(Image.open(out_height))
        self.assertEqual(color.shape, (4096, 4096, 3))
        self.assertEqual(height.shape, (4096, 4096))
        self.assertGreater(int((color[:, :, 0] > 200).sum()), 1000, "应有红色笔迹")
        self.assertGreater(int(height.max()), 33000, "高度应有凸起")
        self.assertEqual(int(np.median(height)), 32768, "没画的地方高度为零点")
        Image.fromarray(color).resize((512, 512)).save(str(Path(__file__).resolve().parents[1] / "docs" / "evidence" / "engine" / "export_basecolor_preview.png"))
        engine.close_storage()
        engine.release()
        leftovers = [n for n in os.listdir(self.tmp) if n.startswith("测试 工程")]
        self.assertEqual(leftovers, [os.path.basename(path)], "关闭后只应剩工程文件本身")

        # 另开一个引擎读回
        engine2 = Engine(self.ctx, Preferences(), History())
        t0 = time.perf_counter()
        project2, meta = engine2.load_project(path)
        view2 = engine2.create_view(ViewShading())
        view2.resize(960, 600)
        engine2.frame_view(view2)
        frames = engine2.settle()
        print("打开并显示完整画面：%.0f ms，%d 帧" % ((time.perf_counter() - t0) * 1000, frames))
        self.assertEqual(meta.get("ui"), {"hello": 1})
        self.assertEqual([layer.name for layer in project2.texture_sets[0].layers], ["底色", "绘制 1"])
        loaded = engine2.renderer.read_pixels(view2)
        diff = np.abs(loaded.astype(int) - after_third.astype(int))
        print("读回后画面最大差值:", int(diff.max()), "统计:", {k: v for k, v in engine2.stats().items() if k in ("disk_reads", "sync_disk_reads", "uploaded")})
        self.assertLess(int(diff.max()), 4)
        # 读回后还能继续画、继续存
        self.paint(engine2, view2, (0.9, 0.8, 0.1), 90)
        info3 = engine2.save_project(path)
        self.assertGreater(info3["pages"], 0)
        engine2.close_storage()
        engine2.release()


if __name__ == "__main__":
    unittest.main(verbosity=2)
