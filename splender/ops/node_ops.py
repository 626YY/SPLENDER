"""节点图操作：新建、删除节点图，加、删、复制节点，连线，把节点图用到填充层上。都进撤销历史。"""
from __future__ import annotations

from ..core import ops
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import BoolProperty, FloatProperty, IntProperty, StringProperty
from ..doc.project import Layer


def node_editor(ctx):
    editor = ctx.editor
    return editor if getattr(editor, "idname", "") == "NODE_EDITOR" else None


def active_graph(ctx):
    project = ctx.project
    if project is None:
        return None
    editor = node_editor(ctx)
    uid = getattr(editor, "graph_uid", 0) if editor is not None else 0
    return project.graph(uid) if uid else project.active_graph


def _history(ctx):
    return getattr(ctx.app, "history", None)


def _record(ctx, label: str, graph, before: dict) -> None:
    from ..nodes.graph import record

    record(_history(ctx), label, graph, before)


def _unique_graph_name(project, base: str = "节点图") -> str:
    names = {g.name for g in project.graphs}
    if base not in names:
        return base
    number = 2
    while "%s %d" % (base, number) in names:
        number += 1
    return "%s %d" % (base, number)


class _GraphOp(Operator):
    @classmethod
    def poll(cls, ctx) -> bool:
        return active_graph(ctx) is not None


@ops.register
class NodeGraphNew(Operator):
    idname = "node.graph_new"
    label = "新建节点图"
    description = "新建一张节点图（带一个现成的小例子），在节点编辑器里打开"
    icon = "add"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.project is not None

    def execute(self, ctx) -> str:
        from ..nodes.graph import default_graph

        project = ctx.project
        graph = default_graph(_unique_graph_name(project))
        previous = project.active_graph_uid
        project.add_graph(graph)
        project.set_active_graph(graph)
        editor = node_editor(ctx)
        if editor is not None:
            editor.show_graph(graph.uid)
        history = _history(ctx)
        if history is not None:
            def undo() -> None:
                project.remove_graph(graph.uid)
                project.active_graph_uid = previous
                project.graphs_changed.emit(None, "active")

            def redo() -> None:
                project.add_graph(graph)
                project.set_active_graph(graph)

            history.push("新建节点图", undo, redo)
        self.report(ctx, "已新建节点图「%s」" % graph.name)
        return FINISHED


@ops.register
class NodeGraphDelete(_GraphOp):
    idname = "node.graph_delete"
    label = "删除节点图"
    description = "删掉正在编辑的节点图。用它的填充层回到固定颜色"
    icon = "remove"

    def execute(self, ctx) -> str:
        project = ctx.project
        graph = active_graph(ctx)
        index = project.graphs.index(graph)
        project.remove_graph(graph.uid)
        project.set_active_graph(project.graphs[min(index, len(project.graphs) - 1)] if project.graphs else None)
        history = _history(ctx)
        if history is not None:
            def undo() -> None:
                project.add_graph(graph, index)
                project.set_active_graph(graph)

            def redo() -> None:
                project.remove_graph(graph.uid)

            history.push("删除节点图", undo, redo)
        return FINISHED


@ops.register
class NodeGraphSelect(Operator):
    idname = "node.graph_select"
    label = "切换节点图"
    searchable = False
    graph = IntProperty("节点图", default=0)

    def execute(self, ctx) -> str:
        project = ctx.project
        graph = project.graph(self.graph) if project is not None else None
        if graph is None:
            return CANCELLED
        project.set_active_graph(graph)
        editor = node_editor(ctx)
        if editor is not None:
            editor.show_graph(graph.uid)
        return FINISHED


@ops.register
class NodeAdd(_GraphOp):
    idname = "node.add"
    label = "添加节点"
    searchable = False
    type = StringProperty("类型", default="")
    x = FloatProperty("横坐标", default=0.0)
    y = FloatProperty("纵坐标", default=0.0)
    at_center = BoolProperty("放在视图中间", default=True)

    def execute(self, ctx) -> str:
        from ..nodes import library

        node_type = library.get(self.type)
        graph = active_graph(ctx)
        if node_type is None:
            return CANCELLED
        location = (self.x, self.y)
        editor = node_editor(ctx)
        if self.at_center and editor is not None:
            location = editor.view_center()
        before = graph.to_dict()
        node = graph.add_node(self.type, location)
        _record(ctx, "添加节点「%s」" % node_type.label, graph, before)
        if editor is not None:
            editor.select_only([node.uid])
        return FINISHED


@ops.register
class NodeAddMenu(_GraphOp):
    idname = "node.add_menu"
    label = "添加节点…"
    description = "在鼠标处添加节点"
    icon = "add"

    def invoke(self, ctx, event) -> str:
        editor = node_editor(ctx)
        if editor is None:
            return CANCELLED
        editor.open_add_menu()
        return FINISHED

    def execute(self, ctx) -> str:
        return self.invoke(ctx, None)


@ops.register
class NodeDelete(_GraphOp):
    idname = "node.delete"
    label = "删除节点"
    description = "删掉选中的节点"
    icon = "remove"

    def execute(self, ctx) -> str:
        graph = active_graph(ctx)
        editor = node_editor(ctx)
        uids = editor.selected_uids() if editor is not None else []
        if not uids and graph.active_uid:
            uids = [graph.active_uid]
        if not uids:
            return CANCELLED
        before = graph.to_dict()
        for uid in uids:
            graph.remove_node(uid)
        _record(ctx, "删除节点", graph, before)
        return FINISHED


@ops.register
class NodeDuplicate(_GraphOp):
    idname = "node.duplicate"
    label = "复制节点"
    description = "复制选中的节点（它们之间的连线一起复制）"
    icon = "duplicate"

    def execute(self, ctx) -> str:
        graph = active_graph(ctx)
        editor = node_editor(ctx)
        uids = editor.selected_uids() if editor is not None else []
        if not uids and graph.active_uid:
            uids = [graph.active_uid]
        uids = [uid for uid in uids if uid in graph.nodes and graph.nodes[uid].type_id != "OUTPUT"]
        if not uids:
            return CANCELLED
        before = graph.to_dict()
        mapping = {}
        for uid in uids:
            source = graph.nodes[uid]
            copy = graph.add_node(source.type_id, (source.location[0] + 40.0, source.location[1] + 40.0))
            copy.params.copy_from(source.params)
            copy.name = source.name
            mapping[uid] = copy.uid
        for src, dst, slot in list(graph.links):
            if src in mapping and dst in mapping:
                graph.link(mapping[src], mapping[dst], slot)
            elif dst in mapping:
                graph.link(src, mapping[dst], slot)
        _record(ctx, "复制节点", graph, before)
        if editor is not None:
            editor.select_only(list(mapping.values()))
        return FINISHED


@ops.register
class NodeLink(_GraphOp):
    idname = "node.link"
    label = "连线"
    searchable = False
    source = IntProperty("上游", default=0)
    target = IntProperty("下游", default=0)
    index = IntProperty("输入", default=0)

    def execute(self, ctx) -> str:
        graph = active_graph(ctx)
        before = graph.to_dict()
        if not graph.link(self.source, self.target, self.index):
            self.report(ctx, "这样连会绕成一个圈，或者接不上", "WARNING")
            return CANCELLED
        _record(ctx, "连线", graph, before)
        return FINISHED


@ops.register
class NodeUnlink(_GraphOp):
    idname = "node.unlink"
    label = "断开连线"
    searchable = False
    target = IntProperty("下游", default=0)
    index = IntProperty("输入", default=0)

    def execute(self, ctx) -> str:
        graph = active_graph(ctx)
        before = graph.to_dict()
        if not graph.unlink(self.target, self.index):
            return CANCELLED
        _record(ctx, "断开连线", graph, before)
        return FINISHED


@ops.register
class NodeViewAll(Operator):
    idname = "node.view_all"
    label = "看全部节点"
    icon = "focus"

    @classmethod
    def poll(cls, ctx) -> bool:
        return node_editor(ctx) is not None

    def execute(self, ctx) -> str:
        node_editor(ctx).frame_all()
        return FINISHED


@ops.register
class NodeGraphEdit(Operator):
    idname = "node.graph_edit"
    label = "编辑节点图"
    description = "到「节点」工作区打开这张节点图"
    icon = "editor.nodes"
    graph = IntProperty("节点图", default=0)

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.project is not None and bool(ctx.project.graphs)

    def execute(self, ctx) -> str:
        project = ctx.project
        graph = project.graph(self.graph) or project.active_graph
        if graph is None:
            return CANCELLED
        project.set_active_graph(graph)
        window = getattr(ctx.app, "window", None)
        if window is not None:
            try:
                window.set_workspace("节点")
            except Exception:  # noqa: BLE001
                pass
            screen = window.current_screen() if hasattr(window, "current_screen") else None
            for area in (screen.areas() if screen is not None else []):
                editor = area.editor
                if getattr(editor, "idname", "") == "NODE_EDITOR":
                    editor.show_graph(graph.uid)
        return FINISHED


@ops.register
class NodeGraphNewForLayer(Operator):
    idname = "node.graph_new_for_layer"
    label = "新建节点图并用上"
    description = "新建一张节点图，当前填充层的内容改成来自它"
    icon = "add"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.project is not None and _fill_layer(ctx) is not None

    def execute(self, ctx) -> str:
        if ops.call("node.graph_new", ctx, invoke=False) != FINISHED:
            return CANCELLED
        return ops.call("layer.set_fill_graph", ctx, invoke=False, graph=ctx.project.active_graph_uid)


# ---------------------------------------------------------------- 填充层用节点图
def _fill_layer(ctx):
    ts = ctx.texture_set
    layer = ts.active_layer if ts is not None else None
    return layer if layer is not None and layer.kind == "FILL" else None


@ops.register
class LayerSetFillGraph(Operator):
    idname = "layer.set_fill_graph"
    label = "填充用节点图"
    description = "当前填充层的内容改成来自这张节点图（0 表示用固定值）"
    searchable = False
    graph = IntProperty("节点图", default=0)

    @classmethod
    def poll(cls, ctx) -> bool:
        return _fill_layer(ctx) is not None

    def execute(self, ctx) -> str:
        layer = _fill_layer(ctx)
        names = ("fill_graph", "use_basecolor", "use_metallic", "use_roughness", "use_height")
        before = {name: getattr(layer, name) for name in names}
        layer.fill_graph = int(self.graph)
        if self.graph:
            layer.use_basecolor = layer.use_metallic = layer.use_roughness = layer.use_height = True
        after = {name: getattr(layer, name) for name in names}
        history = _history(ctx)
        if history is not None and before != after:
            def apply(values) -> None:
                for name, value in values.items():
                    setattr(layer, name, value)

            history.push("填充用节点图", lambda: apply(before), lambda: apply(after))
        return FINISHED


@ops.register
class LayerAddGraphFill(Operator):
    idname = "layer.add_graph_fill"
    label = "新建节点图材质层"
    description = "新建一个填充层，内容来自正在编辑的节点图"
    icon = "layer.fill"
    graph = IntProperty("节点图", default=0)

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.texture_set is not None and ctx.project is not None and bool(ctx.project.graphs)

    def execute(self, ctx) -> str:
        from .layer_ops import _record as record_structure

        project = ctx.project
        graph = project.graph(self.graph) if self.graph else active_graph(ctx)
        ts = ctx.texture_set
        if graph is None:
            return CANCELLED
        before = ts.structure()
        layer = Layer(name=ts.unique_name(graph.name), kind="FILL", fill_graph=graph.uid, use_basecolor=True,
                      use_metallic=True, use_roughness=True, use_height=True)
        ts.add_layer(layer)
        ts.set_active(layer)
        record_structure(ctx, "新建节点图材质层", ts, before, added=[layer])
        self.report(ctx, "已新建材质层「%s」" % layer.name)
        return FINISHED
