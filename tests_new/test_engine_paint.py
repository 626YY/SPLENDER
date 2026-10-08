"""引擎端到端：建工程 → 画一笔 → 合成 → 渲染 → 撤销 → 重做。隐藏上下文，不开窗口。

同时把画面存成图片放到 docs/evidence/engine/，供人工检查。
"""
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

from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet, ToolSettings, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/engine")
OUT.mkdir(parents=True, exist_ok=True)


def make_project(kind="sphere", resolution="16384"):
    project = Project("测试")
    ts = project.add_texture_set(TextureSet(name="材质", resolution=resolution))
    ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.55, 0.55, 0.58), fill_roughness=0.45))
    ts.add_layer(Layer(name="绘制 1"))
    mesh = make_test_mesh(kind)
    project.add_object(MeshObject(mesh.name, mesh, [ts.uid]))
    return project, ts


def save(engine, view, name):
    image = engine.renderer.read_pixels(view)
    Image.fromarray(image[:, :, :3]).save(OUT / name)
    return image


class PaintTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)

    def test_paint_undo_redo(self):
        prefs = Preferences()
        history = History()
        engine = Engine(self.ctx, prefs, history)
        project, ts = make_project()
        t0 = time.perf_counter()
        engine.set_project(project)
        print("\n载入场景 %.0f ms" % ((time.perf_counter() - t0) * 1000))
        view = engine.create_view(ViewShading())
        view.resize(1280, 800)
        engine.frame_view(view)
        frames = engine.settle()
        print("首次稳定用了 %d 帧" % frames)
        before = save(engine, view, "01_before.png").copy()

        tools = ToolSettings()
        tools.brush.size = 90
        tools.brush.color = (0.85, 0.2, 0.15)
        tools.brush.use_height = True
        tools.brush.height = 0.08
        tools.brush.roughness = 0.25
        cx, cy = view.width / 2, view.height / 2
        self.assertTrue(engine.stroke_begin(view, cx - 170, cy - 40, 1.0, time.perf_counter(), tools))
        steps = 40
        for i in range(1, steps + 1):
            t = i / steps
            engine.stroke_move(cx - 170 + 340 * t, cy - 40 + 90 * np.sin(t * 6.283), 0.4 + 0.6 * t, time.perf_counter())
            engine.frame()
        mid = save(engine, view, "02_during_stroke.png").copy()
        engine.stroke_end()
        engine.settle()
        after = save(engine, view, "03_after_stroke.png").copy()
        self.assertGreater(np.abs(after.astype(int) - before.astype(int)).sum(), 100000, "画完画面应有变化")
        diff_merge = np.abs(after.astype(int) - mid.astype(int))
        print("落笔中与合并后的画面最大差值:", int(diff_merge.max()), "差异像素:", int((diff_merge.max(axis=2) > 6).sum()))
        # 合并后不重新合成显示；这里强制全部重新合成，确认并进图层的数据和叠加显示一致
        for state in engine.sets.values():
            state.display.mark_all_dirty()
        view.dirty = True
        engine.settle()
        recomposed = save(engine, view, "03b_recomposed.png").copy()
        diff_re = np.abs(recomposed.astype(int) - after.astype(int))
        print("重新合成与叠加显示的最大差值:", int(diff_re.max()), "差异像素:", int((diff_re.max(axis=2) > 6).sum()))
        self.assertLessEqual(int(diff_re.max()), 4, "并进图层的数据应与叠加显示一致")
        self.assertTrue(history.can_undo)
        stats = engine.stats()
        print("统计:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()})

        history.undo()
        engine.settle()
        undone = save(engine, view, "04_undo.png")
        self.assertLess(np.abs(undone.astype(int) - before.astype(int)).max(), 4, "撤销后应与画之前一致")
        history.redo()
        engine.settle()
        redone = save(engine, view, "05_redo.png")
        self.assertLess(np.abs(redone.astype(int) - after.astype(int)).max(), 4, "重做后应与画完一致")

        # 橡皮
        self.assertTrue(engine.stroke_begin(view, cx - 60, cy - 120, 1.0, time.perf_counter(), tools, erase=True))
        for i in range(1, 25):
            engine.stroke_move(cx - 60 + i * 2, cy - 120 + i * 9, 1.0, time.perf_counter())
            engine.frame()
        engine.stroke_end()
        engine.settle()
        save(engine, view, "06_erase.png")

        # 换视角、通道查看
        view.camera.orbit(-140, -60)
        view.camera.zoom(2.2)
        view.invalidate_camera()
        engine.settle()
        save(engine, view, "07_closeup.png")
        view.shading.mode = "CHANNEL"
        view.shading.channel = "height"
        view.dirty = True
        engine.settle()
        save(engine, view, "08_height_channel.png")
        view.shading.mode = "SOLID"
        view.dirty = True
        engine.settle()
        save(engine, view, "09_solid.png")
        engine.release()

    def test_symmetry_and_mask(self):
        """X 对称：画右边，左边对称位置也有笔迹。蒙版：在填充层蒙版上画 0，那里露出下面的颜色。"""
        prefs = Preferences()
        engine = Engine(self.ctx, prefs, History())
        project, ts = make_project()
        engine.set_project(project)
        view = engine.create_view(ViewShading())
        view.resize(900, 900)
        engine.frame_view(view)
        view.camera.yaw = 0.0
        view.camera.pitch = 0.0
        view.invalidate_camera()
        engine.settle()
        base = save(engine, view, "12_sym_before.png").astype(int)
        tools = ToolSettings()
        tools.brush.size = 40
        tools.brush.color = (0.9, 0.1, 0.1)
        tools.symmetry.x = True
        cx, cy = view.width / 2, view.height / 2
        self.assertTrue(engine.stroke_begin(view, cx + 110, cy - 60, 1.0, time.perf_counter(), tools))
        for i in range(1, 16):
            engine.stroke_move(cx + 110, cy - 60 + i * 8, 1.0, time.perf_counter())
            engine.frame()
        engine.stroke_end()
        engine.settle()
        after = save(engine, view, "13_sym_after.png").astype(int)
        diff = np.abs(after - base).sum(axis=2)
        right = diff[int(cy - 60):int(cy + 60), int(cx + 90):int(cx + 130)].mean()
        left = diff[int(cy - 60):int(cy + 60), int(cx - 130):int(cx - 90)].mean()
        middle = diff[int(cy - 60):int(cy + 60), int(cx - 20):int(cx + 20)].mean()
        print("对称：右 %.1f 左 %.1f 中间 %.1f" % (right, left, middle))
        self.assertGreater(right, 40)
        self.assertGreater(left, 40, "X 对称：左边对称位置应有笔迹")
        self.assertLess(middle, 5)
        # 蒙版：给「底色」填充层加蒙版，画 0 的地方填充层消失，露出纹理集的默认底色
        fill = ts.layers[0]
        fill.fill_color = (0.1, 0.8, 0.2)
        fill.has_mask = True
        ts.set_active(fill)
        ts.paint_target = "MASK"
        engine.settle()
        green = save(engine, view, "14_mask_before.png").astype(int)
        tools.symmetry.x = False
        tools.brush.mask_value = 0.0
        self.assertTrue(engine.stroke_begin(view, cx - 40, cy + 120, 1.0, time.perf_counter(), tools),
                        engine.stroke_reject)
        for i in range(1, 12):
            engine.stroke_move(cx - 40 + i * 8, cy + 120, 1.0, time.perf_counter())
            engine.frame()
        engine.stroke_end()
        engine.settle()
        masked = save(engine, view, "15_mask_after.png").astype(int)
        spot = (slice(int(cy + 110), int(cy + 130)), slice(int(cx - 30), int(cx + 40)))
        before_g = green[spot][:, :, 1].mean() - green[spot][:, :, 0].mean()
        after_g = masked[spot][:, :, 1].mean() - masked[spot][:, :, 0].mean()
        print("蒙版：绿色程度 %.1f -> %.1f" % (before_g, after_g))
        self.assertGreater(before_g, 40)
        self.assertLess(after_g, before_g * 0.3, "蒙版画 0 的地方填充层应该消失")
        engine.release()

    def test_rapid_strokes_match_sequential(self):
        """上一笔还在合并时马上落下一笔，结果应和一笔一笔完整合并完全一样。"""
        results = []
        for rapid in (False, True):
            prefs = Preferences()
            prefs.paint.merge_pages_per_frame = 24          # 故意让合并拖很多帧
            history = History()
            engine = Engine(self.ctx, prefs, history)
            project, ts = make_project()
            engine.set_project(project)
            view = engine.create_view(ViewShading())
            view.resize(1024, 700)
            engine.frame_view(view)
            engine.settle()
            tools = ToolSettings()
            tools.brush.size = 70
            tools.brush.use_height = True
            tools.brush.height = 0.05
            cx, cy = view.width / 2, view.height / 2
            max_pending = 0
            for k in range(6):
                tools.brush.color = (0.15 + 0.13 * k, 0.8 - 0.1 * k, 0.3)
                y = cy - 90 + k * 36
                self.assertTrue(engine.stroke_begin(view, cx - 110, y, 1.0, time.perf_counter(), tools), "第 %d 笔没落上" % k)
                for i in range(1, 21):
                    engine.stroke_move(cx - 110 + i * 11, y + 18 * np.sin(i * 0.5 + k), 1.0, time.perf_counter())
                    engine.frame()
                engine.stroke_end()
                engine.frame()
                max_pending = max(max_pending, len(engine.merges))
                if not rapid:
                    engine.complete_strokes()
                    engine.settle()
            engine.settle()
            self.assertFalse(engine.merges)
            self.assertEqual(len(history.steps), 6)
            for state in engine.sets.values():
                state.display.mark_all_dirty()
            view.dirty = True
            engine.settle()
            results.append(save(engine, view, "10_rapid_%d.png" % int(rapid)).copy())
            print("连续落笔：同时等待合并的笔划最多 %d 笔" % max_pending if rapid else "逐笔合并完成")
            if rapid:
                self.assertGreaterEqual(max_pending, 1, "测试应该覆盖到合并未完成就落笔的情况")
                for _ in range(6):
                    history.undo()
                engine.settle()
                for state in engine.sets.values():
                    state.display.mark_all_dirty()
                view.dirty = True
                engine.settle()
                blank = save(engine, view, "11_rapid_undo_all.png")
                results.append(blank.copy())
            engine.release()
        diff = np.abs(results[0].astype(int) - results[1].astype(int))
        print("连续落笔与逐笔合并的最大差值:", int(diff.max()))
        self.assertLessEqual(int(diff.max()), 2, "连续落笔的结果应与逐笔合并一致")


if __name__ == "__main__":
    unittest.main(verbosity=2)
