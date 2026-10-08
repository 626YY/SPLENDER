"""窗口系统展示：离屏创建主窗口并截图到 docs/evidence/area-system/。不弹窗口。

用法：QT_QPA_PLATFORM=offscreen python tools/area_gallery.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from splender.core import registry  # noqa: E402
from splender.core.history import History  # noqa: E402
from splender.core.keymap import default_keyconfig  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.project import Project, ToolSettings  # noqa: E402
from splender.ui import theme  # noqa: E402
from splender.ui.editor import Editor  # noqa: E402
from splender.ui.window import MainWindow  # noqa: E402
from splender.ui.wm import WindowManager  # noqa: E402

OUT = ROOT / "docs" / "evidence" / "area-system"


class FakeApp:
    def __init__(self):
        self.prefs = Preferences()
        self.keyconfig = default_keyconfig()
        self.project = Project("示例工程")
        self.history = History()
        self.tool_settings = ToolSettings()
        self.engine = None
        self.wm = None


def fake_editor(idname, label, icon, color):
    class _Fake(Editor):
        def build_main(self):
            box = QLabel(label)
            box.setStyleSheet("background:%s; color:#ddd; font-size:22px;" % color)
            box.setAlignment(box.alignment() | 0x84)
            return box

        def draw_header(self, layout, ctx):
            layout.label(label)

    _Fake.idname, _Fake.label, _Fake.icon = idname, label, icon
    _Fake.__name__ = "Fake_" + idname
    return _Fake


def main():
    qt = QApplication.instance() or QApplication(sys.argv)
    QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    theme.apply(qt, 1.0)
    for idname, label, icon, color in (("VIEW_3D", "3D 视口", "editor.view3d", "#35353a"),
                                       ("IMAGE_EDITOR", "UV / 图像", "editor.image", "#303438"),
                                       ("LAYERS", "图层", "editor.layers", "#2c2c30"),
                                       ("PROPERTIES", "属性", "editor.properties", "#2c2c30"),
                                       ("ASSETS", "资源库", "editor.assets", "#2a2c2e"),
                                       ("INFO", "信息", "editor.info", "#2a2a2c")):
        if registry.editor(idname) is None:
            registry.register_editor(fake_editor(idname, label, icon, color))
    app = FakeApp()
    wm = WindowManager(app, install_filter=False)
    window = MainWindow(app, wm)
    window.resize(1600, 1000)
    window.show()
    qt.processEvents()
    OUT.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(OUT / "01_default.png"))
    screen = window.current_screen()
    areas = screen.areas()
    print("areas", [a.editor_idname for a in areas])
    new = screen.split_area(areas[0], "H", 0.5)
    qt.processEvents()
    window.grab().save(str(OUT / "02_split.png"))
    screen.maximize(areas[0])
    qt.processEvents()
    window.grab().save(str(OUT / "03_maximized.png"))
    screen.restore()
    screen.join_areas(areas[0], new)
    window.set_workspace(1)
    qt.processEvents()
    window.grab().save(str(OUT / "04_workspace_material.png"))
    print("workspaces", window.workspace_names())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
