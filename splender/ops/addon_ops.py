"""插件的启用、停用、安装、移除。"""
from __future__ import annotations

import os

from ..core import addons, ops
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import StringProperty


@ops.register
class AddonToggle(Operator):
    idname = "preferences.addon_toggle"
    label = "启用或停用插件"
    searchable = False
    module = StringProperty("插件", default="")

    def execute(self, ctx) -> str:
        addon = addons.get(self.module)
        if addon is None:
            return CANCELLED
        if addon.enabled:
            addons.disable(self.module)
            self.report(ctx, "已停用插件：%s" % addon.name)
        else:
            if ctx.app.host is not None:
                ctx.app.host.make_current()
            if addons.enable(self.module):
                self.report(ctx, "已启用插件：%s" % addon.name)
            else:
                self.report(ctx, "插件「%s」启用失败：%s" % (addon.name, addon.error), "WARNING")
        ctx.app.notify("addons")
        return FINISHED


@ops.register
class AddonInstall(Operator):
    idname = "preferences.addon_install"
    label = "安装插件"
    description = "从 .zip、.py 文件或文件夹安装插件"
    icon = "import"

    def invoke(self, ctx, event) -> str:
        from PySide6.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getOpenFileName(None, "安装插件", "", "插件 (*.zip *.py)")
        if not path:
            return CANCELLED
        try:
            installed = addons.install(path)
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "安装失败：%s" % error, "ERROR")
            return CANCELLED
        names = [addons.get(m).name for m in installed if addons.get(m) is not None]
        self.report(ctx, "已安装：%s。在列表里勾选启用" % ("、".join(names) or "插件"))
        ctx.app.notify("addons")
        return FINISHED


@ops.register
class AddonRemove(Operator):
    idname = "preferences.addon_remove"
    label = "移除插件"
    description = "删除这个用户插件的文件"
    searchable = False
    module = StringProperty("插件", default="")

    def invoke(self, ctx, event) -> str:
        from PySide6.QtWidgets import QMessageBox

        addon = addons.get(self.module)
        if addon is None or addon.builtin:
            return CANCELLED
        answer = QMessageBox.question(None, "移除插件", "删除插件「%s」的文件？" % addon.name)
        if answer != QMessageBox.Yes:
            return CANCELLED
        addons.remove(self.module)
        ctx.app.notify("addons")
        return FINISHED


@ops.register
class AddonRefresh(Operator):
    idname = "preferences.addon_refresh"
    label = "重新扫描插件"
    icon = "reset"

    def execute(self, ctx) -> str:
        addons.discover()
        ctx.app.notify("addons")
        return FINISHED


@ops.register
class AddonOpenFolder(Operator):
    idname = "preferences.addon_open_folder"
    label = "打开插件文件夹"
    icon = "folder"

    def execute(self, ctx) -> str:
        os.startfile(str(addons.user_addon_dir()))  # noqa: S606
        return FINISHED
