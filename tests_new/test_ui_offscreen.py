"""界面组件与区域系统的测试。用 Qt 的离屏平台，不出现任何窗口，也不抢焦点。"""
import os
import sys
from pathlib import Path
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from splender.core.props import BoolProperty, EnumProperty, FloatProperty, PropertyGroup  # noqa: E402
from splender.ui import theme  # noqa: E402
from splender.ui.layout import UILayout  # noqa: E402
from splender.ui.widgets import EnumField, NumberField  # noqa: E402

QT = QApplication.instance() or QApplication(sys.argv)
theme.apply(QT, 1.0)


class Settings(PropertyGroup):
    size = FloatProperty("直径", default=40.0, min=1.0, max=500.0)
    mode = EnumProperty("模式", items=[("A", "甲", ""), ("B", "乙", ""), ("C", "丙", "")], default="A")
    flag = BoolProperty("开关", default=False)


class WidgetBindingTest(unittest.TestCase):
    def test_two_way_binding(self):
        host = QWidget()
        data = Settings()
        layout = UILayout(host, None)
        number = layout.prop(data, "size")
        enum = layout.prop(data, "mode")
        self.assertIsInstance(number, NumberField)
        self.assertIsInstance(enum, EnumField)
        data.size = 123.0                        # 数据变了，控件跟着变
        QT.processEvents()
        self.assertAlmostEqual(float(number.value()), 123.0)
        number.setValue(77.0, emit=True)         # 控件变了，数据跟着变
        QT.processEvents()
        self.assertAlmostEqual(data.size, 77.0)
        number.setValue(9999.0, emit=True)       # 超出范围按上限
        QT.processEvents()
        self.assertAlmostEqual(data.size, 500.0)
        enum.setCurrent("C", emit=True)
        QT.processEvents()
        self.assertEqual(data.mode, "C")
        data.mode = "B"
        QT.processEvents()
        self.assertEqual(enum.current(), "B")
        host.deleteLater()


class AreaSystemTest(unittest.TestCase):
    def test_split_join_swap_maximize_and_save(self):
        import area_gallery as ag
        from splender.core import registry
        from splender.ui.window import MainWindow
        from splender.ui.wm import WindowManager

        for idname, label, icon, color in (("VIEW_3D", "3D 视口", "editor.view3d", "#35353a"),
                                           ("IMAGE_EDITOR", "UV / 图像", "editor.image", "#303438"),
                                           ("LAYERS", "图层", "editor.layers", "#2c2c30"),
                                           ("PROPERTIES", "属性", "editor.properties", "#2c2c30"),
                                           ("ASSETS", "资源库", "editor.assets", "#2a2c2e"),
                                           ("INFO", "信息", "editor.info", "#2a2a2c"),
                                           ("OUTLINER", "大纲视图", "editor.outliner", "#2a2a2c"),
                                           ("NODE_EDITOR", "节点编辑器", "editor.nodes", "#2a2a2c")):
            if registry.editor(idname) is None:
                registry.register_editor(ag.fake_editor(idname, label, icon, color))
        app = ag.FakeApp()
        wm = WindowManager(app, install_filter=False)
        window = MainWindow(app, wm)
        window.resize(1400, 900)
        window.show()
        QT.processEvents()
        screen = window.current_screen()
        before = len(screen.areas())
        first = screen.areas()[0]
        new = screen.split_area(first, "H", 0.5)
        QT.processEvents()
        self.assertEqual(len(screen.areas()), before + 1)
        new.set_editor("IMAGE_EDITOR")
        QT.processEvents()
        self.assertEqual(new.editor_idname(), "IMAGE_EDITOR")
        order = screen.areas()
        spot_first, spot_new = order.index(first), order.index(new)
        self.assertTrue(screen.swap_areas(first, new))  # 两块区域连同编辑器互换位置
        QT.processEvents()
        order = screen.areas()
        self.assertEqual(order.index(first), spot_new)
        self.assertEqual(order.index(new), spot_first)
        self.assertEqual(new.editor_idname(), "IMAGE_EDITOR")
        self.assertTrue(screen.maximize(new))
        QT.processEvents()
        self.assertIs(screen.maximized_area, new)
        screen.restore()
        QT.processEvents()
        self.assertIsNone(screen.maximized_area)
        saved = screen.to_dict()
        self.assertTrue(screen.join_areas(first, new))
        QT.processEvents()
        self.assertEqual(len(screen.areas()), before)
        screen.from_dict(saved)                  # 布局可以存下来再还原
        QT.processEvents()
        self.assertEqual(len(screen.areas()), before + 1)
        self.assertIn("IMAGE_EDITOR", [a.editor_idname() for a in screen.areas()])
        start_name = window.active_workspace().name
        for name in ("UV", "材质", "绘制", start_name):     # 最后回到一开始那个工作区（下面接着用它的布局）
            if name == window.active_workspace().name:
                continue
            self.assertTrue(window.set_workspace(name), "切换到工作区「%s」失败" % name)
            QT.processEvents()
        # 弹出：区域复制到一个独立窗口，编辑器类型相同
        source = screen.areas()[0]
        popped = wm.popout(source)
        QT.processEvents()
        self.assertIsNotNone(popped)
        inner = popped.current_screen().areas()
        self.assertEqual(len(inner), 1)
        self.assertEqual(inner[0].editor_idname(), source.editor_idname())
        popped.close()
        # 布局变化写进偏好，重新打开窗口时还原
        count = len(screen.areas())
        screen.split_area(screen.areas()[0], "V", 0.4)
        for _ in range(5):
            QT.processEvents()
        self.assertTrue(app.prefs.workspaces, "布局没有写进偏好")
        window.close()
        second = MainWindow(app, wm)
        second.resize(1400, 900)
        QT.processEvents()
        self.assertEqual(len(second.current_screen().areas()), count + 1, "重新打开后布局没有还原")
        second.close()


class BrushPopoverTest(unittest.TestCase):
    def test_brush_popover_builds(self):
        """右键笔刷浮层的内容能生成：直径、流量、颜色等控件都在。"""
        from types import SimpleNamespace

        import area_gallery as ag
        from splender.ops.paint_ops import draw_brush_popover

        app = ag.FakeApp()
        ctx = SimpleNamespace(app=app, texture_set=None, layer=None, editor=None, area=None, window=None, wm=None)
        host = QWidget()
        layout = UILayout(host, ctx)
        layout.use_property_split = True
        holder = draw_brush_popover(layout, ctx)
        QT.processEvents()
        self.assertTrue(hasattr(holder, "_link"), "颜色选择没有绑定到笔刷")
        numbers = host.findChildren(NumberField)
        self.assertGreaterEqual(len(numbers), 4, "直径、流量、不透明度、硬度等数值控件应该都在")
        app.tool_settings.brush.size = 88.0
        QT.processEvents()
        self.assertTrue(any(abs(float(n.value()) - 88.0) < 1e-6 for n in numbers), "改笔刷直径后浮层里的数值应跟着变")
        host.deleteLater()


if __name__ == "__main__":
    unittest.main(verbosity=2)
