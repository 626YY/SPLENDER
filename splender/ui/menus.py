"""顶栏菜单和各编辑器的菜单。菜单内容只是把操作排出来，快捷键文字由界面库从键位表里取。"""
from __future__ import annotations

import os

from ..core import registry


@registry.register_menu
class FileMenu(registry.Menu):
    idname = "TOPBAR_MT_file"
    label = "文件"

    def draw(self, layout, ctx) -> None:
        layout.operator("wm.new", icon="new")
        layout.operator("wm.open", icon="open")
        layout.menu("TOPBAR_MT_recent", text="最近打开")
        layout.operator("wm.recover_autosave", icon="recover")
        layout.separator()
        layout.operator("wm.save", icon="save")
        layout.operator("wm.save_as")
        layout.operator("wm.save_incremental")
        layout.separator()
        layout.operator("wm.import_mesh", icon="import")
        layout.operator("wm.import_to_scene", icon="import")
        layout.operator("wm.export_textures", icon="export")
        layout.operator("wm.export_mesh", icon="export")
        layout.separator()
        layout.operator("wm.preferences", icon="settings")
        layout.separator()
        layout.operator("wm.quit")


@registry.register_menu
class FileContextMenu(registry.Menu):
    """F4：文件相关的常用命令（和 Blender 的 F4 菜单一样）。"""

    idname = "TOPBAR_MT_file_context"
    label = "文件"

    def draw(self, layout, ctx) -> None:
        layout.operator("wm.new", icon="new")
        layout.operator("wm.open", icon="open")
        layout.menu("TOPBAR_MT_recent", text="最近打开")
        layout.operator("wm.recover_autosave", icon="recover")
        layout.separator()
        layout.operator("wm.save", icon="save")
        layout.operator("wm.save_as")
        layout.operator("wm.save_incremental")
        layout.separator()
        layout.operator("wm.import_mesh", icon="import")
        layout.operator("wm.import_to_scene", icon="import")
        layout.operator("wm.export_textures", icon="export")
        layout.operator("wm.export_mesh", icon="export")
        layout.separator()
        layout.operator("wm.preferences", icon="settings")


@registry.register_menu
class RecentMenu(registry.Menu):
    idname = "TOPBAR_MT_recent"
    label = "最近打开"

    def draw(self, layout, ctx) -> None:
        files = ctx.prefs.recent_files() if ctx.prefs is not None else []
        if not files:
            layout.label("还没有打开过工程", role="dim")
            return
        for path in files:
            layout.operator("wm.open_recent", text=os.path.basename(path), filepath=path)


@registry.register_menu
class EditMenu(registry.Menu):
    idname = "TOPBAR_MT_edit"
    label = "编辑"

    def draw(self, layout, ctx) -> None:
        history = ctx.history
        undo = "撤销" + (" " + history.undo_label if history is not None and history.can_undo else "")
        redo = "重做" + (" " + history.redo_label if history is not None and history.can_redo else "")
        layout.operator("ed.undo", text=undo, icon="undo")
        layout.operator("ed.redo", text=redo, icon="redo")
        layout.separator()
        layout.operator("wm.search", text="搜索操作…", icon="search")


@registry.register_menu
class WindowMenu(registry.Menu):
    idname = "TOPBAR_MT_window"
    label = "窗口"

    def draw(self, layout, ctx) -> None:
        layout.operator("screen.area_maximize", text="最大化鼠标所在区域", icon="maximize")
        layout.operator("screen.region_toggle", text="显示或隐藏工具栏", region="TOOLBAR")
        layout.operator("screen.region_toggle", text="显示或隐藏侧栏", region="SIDEBAR")
        layout.separator()
        layout.operator("screen.workspace_add", text="新建工作区", icon="add")
        layout.operator("screen.workspace_cycle", text="下一个工作区", direction=1)
        layout.operator("screen.workspace_cycle", text="上一个工作区", direction=-1)


@registry.register_menu
class HelpMenu(registry.Menu):
    idname = "TOPBAR_MT_help"
    label = "帮助"

    def draw(self, layout, ctx) -> None:
        layout.operator("wm.open_log_folder", icon="folder")
        layout.operator("wm.copy_diagnostics", icon="diagnostics")
        layout.separator()
        layout.operator("wm.about", icon="info")


@registry.register_menu
class LayerAddMenu(registry.Menu):
    idname = "LAYERS_MT_add"
    label = "新建"

    def draw(self, layout, ctx) -> None:
        layout.operator("layer.add_paint", icon="layer.paint")
        layout.operator("layer.add_fill", icon="layer.fill")
        layout.operator("layer.add_folder", icon="layer.folder")
        layout.menu("LAYERS_MT_adjust", text="调整层", icon="adjust")
        layout.menu("LAYERS_MT_smart", text="智能材质", icon="smart_material")
        layout.operator("layer.add_graph_fill", text="节点图材质层", icon="editor.nodes")
        layout.separator()
        layout.operator("layer.mask_add", text="添加白色蒙版", icon="layer.mask", fill="WHITE")
        layout.operator("layer.mask_add", text="添加黑色蒙版", icon="layer.mask", fill="BLACK")


@registry.register_menu
class AdjustLayerMenu(registry.Menu):
    idname = "LAYERS_MT_adjust"
    label = "调整层"

    def draw(self, layout, ctx) -> None:
        from ..doc.project import ADJUST_ITEMS, ADJUST_MENU_BREAKS

        for ident, label, _desc in ADJUST_ITEMS:
            layout.operator("layer.add_adjust", text=label, icon="adjust", adjust=ident)
            if ident in ADJUST_MENU_BREAKS:
                layout.separator()


@registry.register_menu
class SmartMaterialMenu(registry.Menu):
    idname = "LAYERS_MT_smart"
    label = "智能材质"

    def draw(self, layout, ctx) -> None:
        from ..bake.smart import SMART_MATERIALS

        for item in SMART_MATERIALS:
            layout.operator("layer.add_smart_material", text=item["name"], icon="smart_material",
                            material=item["id"])


@registry.register_menu
class NodeGraphsMenu(registry.Menu):
    idname = "NODE_MT_graphs"
    label = "节点图"

    def draw(self, layout, ctx) -> None:
        project = ctx.project
        for graph in (project.graphs if project is not None else []):
            layout.operator("node.graph_select", text=graph.name, icon="editor.nodes", graph=graph.uid)
        if project is not None and project.graphs:
            layout.separator()
        layout.operator("node.graph_new", icon="add")
        layout.operator("node.graph_delete", icon="remove")


@registry.register_menu
class FillGraphMenu(registry.Menu):
    idname = "LAYERS_MT_fill_graph"
    label = "填充来源"

    def draw(self, layout, ctx) -> None:
        layout.operator("layer.set_fill_graph", text="固定数值", icon="layer.fill", graph=0)
        project = ctx.project
        for graph in (project.graphs if project is not None else []):
            layout.operator("layer.set_fill_graph", text="节点图「%s」" % graph.name, icon="editor.nodes",
                            graph=graph.uid)
        layout.separator()
        layout.operator("node.graph_new_for_layer", icon="add")


@registry.register_menu
class SmartMaskMenu(registry.Menu):
    idname = "LAYERS_MT_smart_mask"
    label = "智能遮罩"

    def draw(self, layout, ctx) -> None:
        from ..bake.smart import SMART_MASKS

        for item in SMART_MASKS:
            layout.operator("layer.apply_smart_mask", text=item["name"], icon="generator", mask=item["id"])


@registry.register_menu
class LayerContextMenu(registry.Menu):
    idname = "LAYERS_MT_context"
    label = "图层"

    def draw(self, layout, ctx) -> None:
        layout.operator("layer.duplicate", icon="duplicate")
        layout.operator("layer.move", text="上移", icon="up", direction="UP")
        layout.operator("layer.move", text="下移", icon="down", direction="DOWN")
        layout.separator()
        layout.operator("layer.group", icon="layer.folder")
        layout.operator("layer.ungroup", icon="layer.folder")
        layout.separator()
        layout.operator("layer.mask_add", text="添加白色蒙版", icon="layer.mask", fill="WHITE")
        layout.operator("layer.mask_add", text="添加黑色蒙版", icon="layer.mask", fill="BLACK")
        layout.operator("layer.mask_remove", icon="remove")
        layout.menu("LAYERS_MT_smart_mask", text="智能遮罩", icon="generator")
        layout.separator()
        layout.operator("layer.delete", icon="remove")
