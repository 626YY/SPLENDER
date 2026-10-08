"""着色器编辑器的菜单（照 Blender 节点编辑器）：视图、选择、添加（按分类）、节点、换材质、右键菜单。"""
from __future__ import annotations

from ..core import registry
from ..shading import nodes as shader_nodes


def _menu(cls):
    return registry.register_menu(cls)


@_menu
class ShaderViewMenu(registry.Menu):
    idname = "SHADER_MT_view"
    label = "视图"

    def draw(self, layout, ctx) -> None:
        layout.operator("screen.region_toggle", text="侧栏", region="SIDEBAR")
        layout.separator()
        layout.operator("shader.view_selected", text="查看所选")
        layout.operator("shader.view_all", text="查看全部")
        layout.separator()
        layout.operator("screen.area_maximize", text="最大化区域", icon="maximize")


@_menu
class ShaderSelectMenu(registry.Menu):
    idname = "SHADER_MT_select"
    label = "选择"

    def draw(self, layout, ctx) -> None:
        layout.operator("shader.select_all", text="全选", action="SELECT")
        layout.operator("shader.select_all", text="全不选", action="DESELECT")
        layout.operator("shader.select_all", text="反选", action="INVERT")


@_menu
class ShaderAddMenu(registry.Menu):
    idname = "SHADER_MT_add"
    label = "添加"

    def draw(self, layout, ctx) -> None:
        for key, label in shader_nodes.CATEGORIES:
            layout.menu("SHADER_MT_add_" + key.lower(), text=label)


def _category_menu(key: str, label: str):
    class CategoryMenu(registry.Menu):
        idname = "SHADER_MT_add_" + key.lower()

        def draw(self, layout, ctx) -> None:
            for node_type in shader_nodes.all_types():
                if node_type.category == key:
                    layout.operator("shader.add_node", text=node_type.label, type=node_type.id)

    CategoryMenu.label = label
    CategoryMenu.__name__ = "ShaderAdd%sMenu" % key.title()
    return _menu(CategoryMenu)


for _key, _label in shader_nodes.CATEGORIES:
    _category_menu(_key, _label)


@_menu
class ShaderNodeMenu(registry.Menu):
    idname = "SHADER_MT_node"
    label = "节点"

    def draw(self, layout, ctx) -> None:
        layout.operator("shader.translate", text="移动")
        layout.separator()
        layout.operator("shader.duplicate_move", text="复制", icon="duplicate")
        layout.operator("shader.clipboard_copy", text="复制到剪贴板")
        layout.operator("shader.clipboard_paste", text="粘贴")
        layout.separator()
        layout.operator("shader.delete", text="删除", icon="remove")
        layout.operator("shader.delete_reconnect", text="删除并接通")
        layout.separator()
        layout.operator("shader.link_make", text="连接选中的节点")
        layout.separator()
        layout.operator("shader.mute_toggle", text="静音")
        layout.operator("shader.hide_toggle", text="折叠")
        layout.operator("shader.rename", text="改名")


@_menu
class ShaderContextMenu(registry.Menu):
    idname = "SHADER_MT_context_menu"
    label = "节点"

    def draw(self, layout, ctx) -> None:
        layout.operator("shader.duplicate_move", text="复制", icon="duplicate")
        layout.operator("shader.delete", text="删除", icon="remove")
        layout.operator("shader.delete_reconnect", text="删除并接通")
        layout.separator()
        layout.operator("shader.link_make", text="连接选中的节点")
        layout.separator()
        layout.operator("shader.mute_toggle", text="静音")
        layout.operator("shader.hide_toggle", text="折叠")
        layout.operator("shader.rename", text="改名")
        layout.separator()
        layout.menu("SHADER_MT_add", text="添加")


@_menu
class ShaderTextureSetsMenu(registry.Menu):
    idname = "SHADER_MT_texture_sets"
    label = "材质"

    def draw(self, layout, ctx) -> None:
        project = getattr(ctx.app, "project", None)
        if project is None:
            return
        for ts in project.texture_sets:
            mark = "（节点）" if ts.material is not None else ""
            layout.operator("shader.texture_set", text=ts.name + mark, icon="texture_set", uid=ts.uid)
