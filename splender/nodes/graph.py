"""节点图：节点和连线，存进工程。

每个节点只有一个输出；连线是 (上游节点, 下游节点, 下游的第几个输入)。一个输入最多接一根线，接新线替换旧的；
不允许连成环。图有变化时发 changed(kind, node)：
- "structure"：增删节点、连线变化（要重新求值）；
- "params"：某个节点的参数变了（要重新求值）；
- "layout"：节点挪了位置、改了名字（不用重新求值）；
- "active"：当前节点变了；
- "settings"：图的设置（分辨率、名字）变了。
"""
from __future__ import annotations

from typing import Any

from ..core.props import EnumProperty, PropertyGroup, StringProperty
from ..core.signals import Signal
from ..doc.project import new_uid, reserve_uid
from . import library

RESOLUTION_ITEMS = [("256", "256", ""), ("512", "512", ""), ("1024", "1K", ""), ("2048", "2K", ""), ("4096", "4K", "")]


class GraphSettings(PropertyGroup):
    undoable = True
    name = StringProperty("名称", default="节点图")
    resolution = EnumProperty("分辨率", items=RESOLUTION_ITEMS, default="1024",
                              description="节点图按多大的图计算：越大越细，越慢")


class Node:
    def __init__(self, type_id: str, uid: int | None = None, location: tuple = (0.0, 0.0), name: str = "") -> None:
        node_type = library.get(type_id)
        if node_type is None:
            raise ValueError("没有这种节点：%s" % type_id)
        self.type_id = type_id
        self.uid = uid if uid is not None else new_uid()
        self.location = (float(location[0]), float(location[1]))
        self.name = name
        self.params = node_type.Params()

    @property
    def type(self):
        return library.get(self.type_id)

    @property
    def label(self) -> str:
        return self.name or self.type.label

    def to_dict(self) -> dict:
        return {"uid": self.uid, "type": self.type_id, "location": list(self.location), "name": self.name,
                "params": self.params.to_dict()}


class NodeGraph:
    def __init__(self, name: str = "节点图", uid: int | None = None) -> None:
        self.uid = uid if uid is not None else new_uid()
        self.settings = GraphSettings(name=name)
        self.settings.changed.connect(self._on_settings)
        self.nodes: dict[int, Node] = {}
        self.links: list[tuple[int, int, int]] = []
        self.active_uid = 0
        self.version = 0                  # 影响求值的变化每次加一
        self.changed = Signal()

    # ---- 名字、分辨率（在 settings 里，界面上能直接改）----
    @property
    def name(self) -> str:
        return self.settings.name

    @name.setter
    def name(self, value: str) -> None:
        self.settings.name = str(value)

    @property
    def resolution(self) -> str:
        return self.settings.resolution

    @resolution.setter
    def resolution(self, value: str) -> None:
        self.settings.resolution = str(value)

    def _on_settings(self, name: str) -> None:
        self._touch("settings" if name == "resolution" else "layout")

    # ---- 查询 ----
    @property
    def size(self) -> int:
        return int(self.resolution)

    @property
    def active(self) -> Node | None:
        return self.nodes.get(self.active_uid)

    def node(self, uid: int) -> Node | None:
        return self.nodes.get(uid)

    def input_link(self, to_uid: int, index: int) -> int | None:
        for src, dst, slot in self.links:
            if dst == to_uid and slot == index:
                return src
        return None

    def output_node(self) -> Node | None:
        for node in self.nodes.values():
            if node.type_id == "OUTPUT":
                return node
        return None

    def upstream(self, uid: int) -> set[int]:
        """uid 依赖的所有节点（不含自己）。"""
        found: set[int] = set()
        stack = [uid]
        while stack:
            current = stack.pop()
            for src, dst, _slot in self.links:
                if dst == current and src not in found:
                    found.add(src)
                    stack.append(src)
        return found

    def downstream(self, uid: int) -> set[int]:
        found: set[int] = set()
        stack = [uid]
        while stack:
            current = stack.pop()
            for src, dst, _slot in self.links:
                if src == current and dst not in found:
                    found.add(dst)
                    stack.append(dst)
        return found

    # ---- 修改 ----
    def _touch(self, kind: str, node: Node | None = None) -> None:
        if kind in ("structure", "params", "settings"):
            self.version += 1
        self.changed.emit(kind, node)

    def add_node(self, type_id: str, location: tuple = (0.0, 0.0), uid: int | None = None) -> Node:
        node = Node(type_id, uid=uid, location=location)
        self.nodes[node.uid] = node
        node.params.changed.connect(lambda _name, target=node: self._touch("params", target))
        self.active_uid = node.uid
        self._touch("structure", node)
        return node

    def remove_node(self, uid: int) -> None:
        node = self.nodes.pop(uid, None)
        if node is None:
            return
        self.links = [link for link in self.links if link[0] != uid and link[1] != uid]
        if self.active_uid == uid:
            self.active_uid = 0
        self._touch("structure", node)

    def can_link(self, from_uid: int, to_uid: int, index: int) -> bool:
        source, target = self.nodes.get(from_uid), self.nodes.get(to_uid)
        if source is None or target is None or from_uid == to_uid:
            return False
        if source.type.output_kind == "none" or not 0 <= index < len(target.type.inputs):
            return False
        return from_uid not in self.downstream(to_uid)

    def link(self, from_uid: int, to_uid: int, index: int) -> bool:
        """接线；会成环、类型不对时不接，返回 False。"""
        if not self.can_link(from_uid, to_uid, index):
            return False
        self.links = [link for link in self.links if not (link[1] == to_uid and link[2] == index)]
        self.links.append((from_uid, to_uid, index))
        self._touch("structure", self.nodes[to_uid])
        return True

    def unlink(self, to_uid: int, index: int) -> bool:
        before = len(self.links)
        self.links = [link for link in self.links if not (link[1] == to_uid and link[2] == index)]
        if len(self.links) != before:
            self._touch("structure", self.nodes.get(to_uid))
            return True
        return False

    def move_node(self, uid: int, location: tuple) -> None:
        node = self.nodes.get(uid)
        if node is not None:
            node.location = (float(location[0]), float(location[1]))
            self._touch("layout", node)

    def set_active(self, uid: int) -> None:
        if uid != self.active_uid and (uid == 0 or uid in self.nodes):
            self.active_uid = uid
            self._touch("active", self.nodes.get(uid))

    def set_resolution(self, value: str) -> None:
        if value != self.resolution and value in dict(RESOLUTION_ITEMS):
            self.resolution = value

    def rename(self, name: str) -> None:
        name = str(name).strip() or self.name
        if name != self.name:
            self.name = name

    # ---- 存取、撤销 ----
    def to_dict(self) -> dict:
        return {"uid": self.uid, "name": self.name, "resolution": self.resolution, "active": self.active_uid,
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "links": [list(link) for link in self.links]}

    def restore(self, data: dict) -> None:
        """换成 data 描述的样子（撤销用）：参数相同的节点保留原对象。"""
        with self.settings.changed.block():
            self.settings.name = str(data.get("name", self.name))
            self.settings.resolution = str(data.get("resolution", self.resolution))
        old = self.nodes
        self.nodes = {}
        for item in data.get("nodes", []):
            uid = int(item["uid"])
            node = old.get(uid)
            if node is None or node.type_id != item.get("type"):
                if library.get(item.get("type", "")) is None:
                    continue
                node = Node(item["type"], uid=uid)
                node.params.changed.connect(lambda _name, target=node: self._touch("params", target))
                reserve_uid(uid)
            node.location = tuple(float(v) for v in item.get("location", (0.0, 0.0)))[:2]
            node.name = str(item.get("name", ""))
            with node.params.changed.block():
                node.params.from_dict(item.get("params") or {})
            self.nodes[uid] = node
        self.links = [(int(a), int(b), int(c)) for a, b, c in data.get("links", [])
                      if int(a) in self.nodes and int(b) in self.nodes]
        active = int(data.get("active", 0))
        self.active_uid = active if active in self.nodes else 0
        self._touch("structure")

    @classmethod
    def from_dict(cls, data: dict) -> "NodeGraph":
        uid = int(data.get("uid", 0)) or None
        if uid is not None:
            reserve_uid(uid)
        graph = cls(str(data.get("name", "节点图")), uid=uid)
        graph.restore(data)
        graph.version = 0
        return graph


def default_graph(name: str = "节点图") -> NodeGraph:
    """新建的图：一个噪波经过渐变映射接到材质输出，打开就能看到东西。"""
    graph = NodeGraph(name)
    noise = graph.add_node("NOISE", (-460.0, -40.0))
    colors = graph.add_node("GRADIENT_MAP", (-200.0, -120.0))
    output = graph.add_node("OUTPUT", (80.0, 0.0))
    graph.link(noise.uid, colors.uid, 0)
    graph.link(colors.uid, output.uid, 0)
    graph.link(noise.uid, output.uid, 3)
    graph.set_active(output.uid)
    return graph


def snapshot(graph: NodeGraph) -> dict:
    return graph.to_dict()


def record(history: Any, label: str, graph: NodeGraph, before: dict) -> None:
    """图已经改好（before 是改之前的 to_dict()），记一步撤销。"""
    if history is None:
        return
    after = graph.to_dict()
    if after == before:
        return
    history.push(label, lambda: graph.restore(before), lambda: graph.restore(after))
