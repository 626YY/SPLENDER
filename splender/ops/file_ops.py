"""文件操作：新建、打开、保存、导入模型、导出贴图、偏好设置、退出。

带 filepath / directory 属性的操作，属性为空时弹系统对话框，给了就直接用（脚本和测试走这条路）。
"""
from __future__ import annotations

import logging
import os

from ..core import ops
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import EnumProperty, StringProperty
from ..paths import log_dir

log = logging.getLogger("splender.ops.file")

MESH_FILTER = "模型 (*.obj *.gltf *.glb);;所有文件 (*)"
PROJECT_FILTER = "SPLENDER 工程 (*.splender)"


def confirm_discard(ctx) -> bool:
    """当前工程有没保存的改动时问一句。返回 True 表示可以继续。"""
    project = ctx.project
    if project is None or not project.dirty:
        return True
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(ctx.app.window)
    box.setWindowTitle("SPLENDER")
    box.setIcon(QMessageBox.Question)
    box.setText("「%s」有没保存的改动。" % project.name)
    box.setInformativeText("要先保存吗？")
    save = box.addButton("保存", QMessageBox.AcceptRole)
    box.addButton("不保存", QMessageBox.DestructiveRole)
    cancel = box.addButton("取消", QMessageBox.RejectRole)
    box.setDefaultButton(save)
    box.exec()
    clicked = box.clickedButton()
    if clicked is cancel:
        return False
    if clicked is save:
        return ops.call("wm.save", ctx) == FINISHED
    return True


def _start_dir(ctx) -> str:
    folder = ctx.app.prefs.files.last_dir
    return folder if folder and os.path.isdir(folder) else os.path.expanduser("~")


@ops.register
class WMNew(Operator):
    idname = "wm.new"
    label = "新建工程"
    description = "新建一个带预览球的空工程"
    icon = "new"

    def execute(self, ctx) -> str:
        if not confirm_discard(ctx):
            return CANCELLED
        ctx.app.new_project()
        return FINISHED


@ops.register
class WMImportMesh(Operator):
    idname = "wm.import_mesh"
    label = "导入模型…"
    description = "用一个模型文件新建工程，支持 OBJ 和 glTF"
    icon = "import"
    filepath = StringProperty("文件", default="", subtype="FILE_PATH")

    def execute(self, ctx) -> str:
        path = self.filepath
        if not path:
            from PySide6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getOpenFileName(ctx.app.window, "导入模型", _start_dir(ctx), MESH_FILTER)
        if not path:
            return CANCELLED
        if not confirm_discard(ctx):
            return CANCELLED
        try:
            ctx.app.import_mesh(path)
        except Exception as error:  # noqa: BLE001
            log.exception("导入模型失败")
            self.report(ctx, "没有导入：%s" % error, "ERROR")
            return CANCELLED
        return FINISHED


@ops.register
class WMOpen(Operator):
    idname = "wm.open"
    label = "打开工程…"
    icon = "open"
    filepath = StringProperty("文件", default="", subtype="FILE_PATH")

    def execute(self, ctx) -> str:
        path = self.filepath
        if not path:
            from PySide6.QtWidgets import QFileDialog

            path, _ = QFileDialog.getOpenFileName(ctx.app.window, "打开工程", _start_dir(ctx), PROJECT_FILTER)
        if not path:
            return CANCELLED
        if not confirm_discard(ctx):
            return CANCELLED
        try:
            ctx.app.open_project(path)
        except Exception as error:  # noqa: BLE001
            log.exception("打开工程失败")
            self.report(ctx, "没有打开：%s" % error, "ERROR")
            return CANCELLED
        return FINISHED


@ops.register
class WMSave(Operator):
    idname = "wm.save"
    label = "保存"
    icon = "save"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.project is not None and bool(ctx.project.objects)

    def execute(self, ctx) -> str:
        if not ctx.project.path:
            return ops.call("wm.save_as", ctx)
        try:
            ctx.app.save_project()
        except Exception as error:  # noqa: BLE001
            log.exception("保存失败")
            self.report(ctx, "没有保存：%s" % error, "ERROR")
            return CANCELLED
        return FINISHED


@ops.register
class WMSaveAs(Operator):
    idname = "wm.save_as"
    label = "另存为…"
    filepath = StringProperty("文件", default="", subtype="FILE_PATH")

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.project is not None and bool(ctx.project.objects)

    def execute(self, ctx) -> str:
        path = self.filepath
        if not path:
            from PySide6.QtWidgets import QFileDialog

            start = ctx.project.path or os.path.join(_start_dir(ctx), ctx.project.name + ".splender")
            path, _ = QFileDialog.getSaveFileName(ctx.app.window, "保存工程", start, PROJECT_FILTER)
        if not path:
            return CANCELLED
        if not path.lower().endswith(".splender"):
            path += ".splender"
        try:
            ctx.app.save_project(path)
        except Exception as error:  # noqa: BLE001
            log.exception("保存失败")
            self.report(ctx, "没有保存：%s" % error, "ERROR")
            return CANCELLED
        return FINISHED


def incremented_path(path: str) -> str:
    """「名字.splender」→「名字1.splender」，「名字7.splender」→「名字8.splender」（和 Blender 的增量保存一样）。"""
    import re

    folder, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    match = re.search(r"(\d+)$", stem)
    if match:
        number = int(match.group(1)) + 1
        stem = stem[:match.start()] + str(number).zfill(len(match.group(1)))
    else:
        stem += "1"
    return os.path.join(folder, stem + (ext or ".splender"))


@ops.register
class WMSaveIncremental(Operator):
    idname = "wm.save_incremental"
    label = "增量保存"
    description = "存成一个编号加一的新文件（原来的文件留着）"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.project is not None and bool(ctx.project.path)

    def execute(self, ctx) -> str:
        path = incremented_path(ctx.project.path)
        while os.path.exists(path):
            path = incremented_path(path)
        return ops.call("wm.save_as", ctx, invoke=False, filepath=path)


@ops.register
class WMOpenRecent(Operator):
    idname = "wm.open_recent"
    label = "打开最近的工程"
    searchable = False
    filepath = StringProperty("文件", default="")

    def execute(self, ctx) -> str:
        if not os.path.isfile(self.filepath):
            self.report(ctx, "文件已经不在了：%s" % self.filepath, "WARNING")
            return CANCELLED
        return ops.call("wm.open", ctx, filepath=self.filepath)


@ops.register
class WMExportTextures(Operator):
    idname = "wm.export_textures"
    label = "导出贴图…"
    description = "把勾选的通道各导出成一张 PNG"
    icon = "export"
    directory = StringProperty("文件夹", default="", subtype="DIR_PATH")
    size = EnumProperty("尺寸", items=[("0", "按纹理集设置", ""), ("8192", "8K", ""), ("4096", "4K", ""),
                                     ("2048", "2K", ""), ("1024", "1K", "")], default="0")

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.project is not None and bool(ctx.project.texture_sets) and ctx.engine is not None

    def execute(self, ctx) -> str:
        folder = self.directory
        if not folder:
            from PySide6.QtWidgets import QFileDialog

            start = ctx.app.prefs.files.export_dir or _start_dir(ctx)
            folder = QFileDialog.getExistingDirectory(ctx.app.window, "导出贴图到", start)
        if not folder:
            return CANCELLED
        try:
            os.makedirs(folder, exist_ok=True)
            ctx.app.export_textures(folder, size=int(self.size) or None)
            ctx.app.prefs.files.export_dir = folder
        except Exception as error:  # noqa: BLE001
            log.exception("导出失败")
            ctx.wm.set_progress("", None)
            self.report(ctx, "没有导出：%s" % error, "ERROR")
            return CANCELLED
        return FINISHED


@ops.register
class WMPreferences(Operator):
    idname = "wm.preferences"
    label = "偏好设置…"
    icon = "settings"

    def execute(self, ctx) -> str:
        from ..ui import theme
        from ..ui.window import PopoutWindow

        existing = getattr(ctx.app, "_prefs_window", None)
        try:
            if existing is not None and existing.isVisible():
                existing.raise_()
                existing.activateWindow()
                return FINISHED
        except RuntimeError:
            pass
        window = PopoutWindow(ctx.app, ctx.wm, {"type": "area", "editor": "PREFERENCES", "state": {}}, "偏好设置")
        window.resize(theme.px(760), theme.px(640))
        if ctx.app.window is not None:
            center = ctx.app.window.frameGeometry().center()
            window.move(center.x() - window.width() // 2, center.y() - window.height() // 2)
        window.show()
        ctx.app._prefs_window = window
        return FINISHED


@ops.register
class WMOpenLogFolder(Operator):
    idname = "wm.open_log_folder"
    label = "打开日志文件夹"

    def execute(self, ctx) -> str:
        os.startfile(str(log_dir()))  # noqa: S606
        return FINISHED


@ops.register
class WMAbout(Operator):
    idname = "wm.about"
    label = "关于 SPLENDER"

    def execute(self, ctx) -> str:
        from PySide6.QtWidgets import QMessageBox

        from .. import __version__

        engine = ctx.engine
        gpu = engine.ctx.info.get("GL_RENDERER", "") if engine is not None else ""
        QMessageBox.about(ctx.app.window, "关于 SPLENDER",
                          "<b>SPLENDER</b> %s<br>16K 贴图创作<br><br>显卡：%s" % (__version__, gpu))
        return FINISHED


@ops.register
class WMQuit(Operator):
    idname = "wm.quit"
    label = "退出"

    def execute(self, ctx) -> str:
        if ctx.app.window is not None:
            ctx.app.window.close()
        return FINISHED
