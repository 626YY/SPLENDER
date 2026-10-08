"""着色器编辑器里的操作（照 Blender 节点编辑器的键位）：加节点、删除（并接通）、复制、复制粘贴、静音、折叠、
连接选中的节点、全选、看全部/看选中、移动、启用材质节点、换材质。"""
from __future__ import annotations

import numpy as np

from ..core import ops
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import BoolProperty, EnumProperty, IntProperty, StringProperty


def shader_editor(ctx):
    from ..editors.shader_editor import ShaderEditor

    editor = getattr(ctx, "editor", None)
    return editor if isinstance(editor, ShaderEditor) else None


def _graph(ctx):
    editor = shader_editor(ctx)
    return editor.graph() if editor is not None else None


class _ShaderOp(Operator):
    @classmethod
    def poll(cls, ctx) -> bool:
        return _graph(ctx) is not None


class _SelectionOp(_ShaderOp):
    @classmethod
    def poll(cls, ctx) -> bool:
        editor = shader_editor(ctx)
        return editor is not None and editor.graph() is not None and bool(editor.selected_uids())


@ops.register
class ShaderUseNodes(Operator):
    idname = "shader.use_nodes"
    label = "使用节点"
    description = "这套贴图用材质节点（图层堆栈接到材质输出，可以在中间加节点）"
    icon = "editor.shader"
    enable = BoolProperty("使用", default=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        project = getattr(ctx.app, "project", None)
        return project is not None and project.active_texture_set is not None

    def execute(self, ctx) -> str:
        from ..shading.graph import ShaderGraph

        ts = ctx.app.project.active_texture_set
        before = ts.material
        after = ShaderGraph.default() if self.enable and before is None else (before if self.enable else None)
        if after is before:
            return CANCELLED
        ts.set_material(after)
        history = getattr(ctx.app, "history", None)
        if history is not None:
            history.push("使用节点" if self.enable else "停用节点", lambda: ts.set_material(before),
                         lambda: ts.set_material(after))
        ctx.app.notify("material")
        return FINISHED


@ops.register
class ShaderSetTextureSet(Operator):
    idname = "shader.texture_set"
    label = "换材质"
    description = "着色器编辑器（和图层编辑器）显示这套贴图"
    searchable = False
    uid = IntProperty("纹理集", default=0)

    @classmethod
    def poll(cls, ctx) -> bool:
        return getattr(ctx.app, "project", None) is not None

    def execute(self, ctx) -> str:
        return ops.call("project.texture_set_activate", ctx, invoke=False, uid=int(self.uid))


def _type_items(owner=None):
    from ..shading import nodes

    return [(t.id, t.label, t.description) for t in nodes.all_types()]


@ops.register
class ShaderAddNode(_ShaderOp):
    idname = "shader.add_node"
    label = "添加节点"
    description = "在鼠标处加一个材质节点（跟着鼠标走，点一下放下）"
    searchable = False
    type = EnumProperty("节点", items=_type_items, default="VALUE")

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        graph = editor.graph()
        before = graph.to_dict()
        pos = editor.cursor_scene_pos()
        node = graph.add_node(self.type, (pos.x() - 60.0, pos.y() - 20.0))
        editor._sync()
        editor.select_only([node.uid])
        graph.set_active(node.uid)
        editor.start_grab([node.uid], "添加节点", before)
        return FINISHED


@ops.register
class ShaderDelete(_SelectionOp):
    idname = "shader.delete"
    label = "删除"
    description = "删掉选中的节点（X、Delete）"
    icon = "remove"

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        graph = editor.graph()
        before = graph.to_dict()
        for uid in editor.selected_uids():
            graph.remove_node(uid)
        editor.record("删除节点", before)
        return FINISHED


@ops.register
class ShaderDeleteReconnect(_SelectionOp):
    idname = "shader.delete_reconnect"
    label = "删除并接通"
    description = "删掉选中的节点，把它前后的线接起来（Ctrl+X）"

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        graph = editor.graph()
        before = graph.to_dict()
        for uid in editor.selected_uids():
            node = graph.nodes.get(uid)
            if node is None:
                continue
            # 第一个接了线的输入，接到第一个接了线的输出原来接的地方（和 Blender 一样）
            upstream = next((graph.input_link(uid, name) for name in node.input_names()
                             if graph.input_link(uid, name) is not None), None)
            targets = []
            for name in node.output_names():
                targets = graph.output_links(uid, name)
                if targets:
                    break
            graph.remove_node(uid)
            if upstream is not None:
                for b, sb in targets:
                    graph.link(upstream[0], upstream[1], b, sb)
        editor.record("删除并接通", before)
        return FINISHED


def _copy_nodes(graph, uids) -> dict:
    chosen = set(uids)
    return {"nodes": [graph.nodes[uid].to_dict() for uid in uids if uid in graph.nodes],
            "links": [list(link) for link in graph.links if link[0] in chosen and link[2] in chosen]}


def _paste_nodes(graph, data: dict, offset=(0.0, 0.0)) -> list[int]:
    mapping = {}
    created = []
    for item in data.get("nodes", []):
        node = graph.add_node(item["type"], (item["location"][0] + offset[0], item["location"][1] + offset[1]))
        node.name = item.get("name", "")
        node.mute = bool(item.get("mute", False))
        node.hide = bool(item.get("hide", False))
        node.values.from_dict(item.get("values") or {})
        mapping[int(item["uid"])] = node.uid
        created.append(node.uid)
    for a, sa, b, sb in data.get("links", []):
        if int(a) in mapping and int(b) in mapping:
            graph.link(mapping[int(a)], sa, mapping[int(b)], sb)
    return created


@ops.register
class ShaderDuplicateMove(_SelectionOp):
    idname = "shader.duplicate_move"
    label = "复制"
    description = "复制选中的节点（连同它们之间的线），跟着鼠标走（Shift+D）"
    icon = "duplicate"

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        graph = editor.graph()
        before = graph.to_dict()
        created = _paste_nodes(graph, _copy_nodes(graph, editor.selected_uids()), (30.0, 30.0))
        editor._sync()
        editor.select_only(created)
        editor.start_grab(created, "复制节点", before)
        return FINISHED


@ops.register
class ShaderCopy(_SelectionOp):
    idname = "shader.clipboard_copy"
    label = "复制到剪贴板"
    description = "复制选中的节点（Ctrl+C）"

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        editor.clipboard = _copy_nodes(editor.graph(), editor.selected_uids())
        _SHARED["clipboard"] = editor.clipboard
        return FINISHED


_SHARED: dict = {}


@ops.register
class ShaderPaste(_ShaderOp):
    idname = "shader.clipboard_paste"
    label = "粘贴"
    description = "把复制的节点贴到鼠标处（Ctrl+V）"

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        data = editor.clipboard or _SHARED.get("clipboard")
        if not data or not data.get("nodes"):
            return CANCELLED
        graph = editor.graph()
        before = graph.to_dict()
        xs = [item["location"][0] for item in data["nodes"]]
        ys = [item["location"][1] for item in data["nodes"]]
        pos = editor.cursor_scene_pos()
        created = _paste_nodes(graph, data, (pos.x() - float(np.mean(xs)), pos.y() - float(np.mean(ys))))
        editor._sync()
        editor.select_only(created)
        editor.record("粘贴节点", before)
        return FINISHED


@ops.register
class ShaderMuteToggle(_SelectionOp):
    idname = "shader.mute_toggle"
    label = "静音"
    description = "选中的节点不起作用（输入直接传到输出），再按一次恢复（M）"

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        graph = editor.graph()
        before = graph.to_dict()
        uids = editor.selected_uids()
        mute = not all(graph.nodes[uid].mute for uid in uids)
        for uid in uids:
            graph.set_mute(uid, mute)
        editor.record("静音节点", before)
        return FINISHED


@ops.register
class ShaderHideToggle(_SelectionOp):
    idname = "shader.hide_toggle"
    label = "折叠"
    description = "选中的节点收起成一行，再按一次展开（H）"

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        graph = editor.graph()
        before = graph.to_dict()
        uids = editor.selected_uids()
        hide = not all(graph.nodes[uid].hide for uid in uids)
        for uid in uids:
            graph.set_hide(uid, hide)
        editor._sync()
        editor.record("折叠节点", before)
        return FINISHED


@ops.register
class ShaderLinkMake(_SelectionOp):
    idname = "shader.link_make"
    label = "连接"
    description = "选中的节点从左到右依次连起来：前一个的第一个输出接到后一个第一个空着的输入（F）"

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        graph = editor.graph()
        uids = sorted(editor.selected_uids(), key=lambda uid: graph.nodes[uid].location[0])
        if len(uids) < 2:
            return CANCELLED
        before = graph.to_dict()
        made = 0
        for a, b in zip(uids, uids[1:]):
            na, nb = graph.nodes[a], graph.nodes[b]
            outs = na.output_names()
            free = [name for name in nb.input_names() if graph.input_link(b, name) is None]
            if outs and free:
                # 优先接类型一样的
                kind = na.output_type(outs[0])
                target = next((name for name in free if nb.input_type(name) == kind), free[0])
                made += int(graph.link(a, outs[0], b, target))
        if not made:
            return CANCELLED
        editor.record("连接节点", before)
        return FINISHED


@ops.register
class ShaderSelectAll(_ShaderOp):
    idname = "shader.select_all"
    label = "全选"
    action = EnumProperty("动作", items=[("SELECT", "全选", ""), ("DESELECT", "全不选", ""), ("INVERT", "反选", "")],
                          default="SELECT")

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        if self.action == "SELECT":
            editor.select_only(list(editor.items))
        elif self.action == "DESELECT":
            editor.select_only([])
        else:
            editor.select_only([uid for uid, item in editor.items.items() if not item.isSelected()])
        return FINISHED


@ops.register
class ShaderViewAll(_ShaderOp):
    idname = "shader.view_all"
    label = "查看全部"

    def execute(self, ctx) -> str:
        shader_editor(ctx).frame_all()
        return FINISHED


@ops.register
class ShaderViewSelected(_ShaderOp):
    idname = "shader.view_selected"
    label = "查看所选"

    def execute(self, ctx) -> str:
        shader_editor(ctx).frame_all(only_selected=True)
        return FINISHED


@ops.register
class ShaderTranslate(_SelectionOp):
    idname = "shader.translate"
    label = "移动"
    description = "选中的节点跟着鼠标走，点一下放下（G）"

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        editor.start_grab(editor.selected_uids(), "移动节点")
        return FINISHED


@ops.register
class ShaderRename(_SelectionOp):
    idname = "shader.rename"
    label = "改名"
    description = "给当前节点起个名字（标题上显示）"
    name = StringProperty("名字", default="")

    def invoke(self, ctx, event) -> str:
        from PySide6.QtWidgets import QInputDialog

        graph = _graph(ctx)
        node = graph.active
        if node is None:
            return CANCELLED
        text, ok = QInputDialog.getText(None, "节点名字", "名字", text=node.label)
        if not ok:
            return CANCELLED
        self.name = text
        return self.execute(ctx)

    def execute(self, ctx) -> str:
        editor = shader_editor(ctx)
        graph = editor.graph()
        node = graph.active
        if node is None:
            return CANCELLED
        before = graph.to_dict()
        node.name = "" if self.name == node.type.label else self.name
        graph._emit("layout", node)
        editor._sync()
        if editor.items.get(node.uid) is not None:
            editor.items[node.uid].update()
        editor.record("节点改名", before)
        return FINISHED
