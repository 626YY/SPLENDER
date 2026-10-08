"""材质节点图：节点、按插口的连线（上游节点, 上游插口, 下游节点, 下游插口），存进纹理集。

一个输入插口最多接一根线（接新的替换旧的）；不许连成环。变化时发 changed(kind, node)：
- "structure"：增删节点、连线、静音、改了下拉之类会换着色器代码的参数（要重新编译）；
- "values"：数值、颜色、色带变了（只换存储缓冲里的值）；
- "layout"：位置、折叠、名字（不影响结果）；
- "active"：当前节点。
"""
from __future__ import annotations

from typing import Any

from ..core.signals import Signal
from ..doc.project import new_uid, reserve_uid
from . import nodes

# 改了要重新编译的参数种类（其余的只换数值）
_STRUCTURAL_KINDS = ("ENUM", "BOOL", "INT")


class ShaderNode:
    def __init__(self, type_id: str, uid: int | None = None, location: tuple = (0.0, 0.0), name: str = "") -> None:
        node_type = nodes.get(type_id)
        if node_type is None:
            raise ValueError("没有这种材质节点：%s" % type_id)
        self.type_id = type_id
        self.uid = uid if uid is not None else new_uid()
        self.location = (float(location[0]), float(location[1]))
        self.name = name
        self.mute = False
        self.hide = False            # 折叠（只显示接了线的插口）
        self.values = node_type.Values()

    @property
    def type(self):
        return nodes.get(self.type_id)

    @property
    def label(self) -> str:
        return self.name or self.type.label

    def input_names(self) -> list[str]:
        return [item[0] for item in self.type.inputs]

    def output_names(self) -> list[str]:
        return [item[0] for item in self.type.outputs]

    def input_type(self, name: str) -> str:
        for item in self.type.inputs:
            if item[0] == name:
                return item[2]
        return "FLOAT"

    def output_type(self, name: str) -> str:
        for item in self.type.outputs:
            if item[0] == name:
                return item[2]
        return "FLOAT"

    def to_dict(self) -> dict:
        return {"uid": self.uid, "type": self.type_id, "location": list(self.location), "name": self.name,
                "mute": self.mute, "hide": self.hide, "values": self.values.to_dict()}


class ShaderGraph:
    def __init__(self, uid: int | None = None) -> None:
        self.uid = uid if uid is not None else new_uid()
        self.nodes: dict[int, ShaderNode] = {}
        self.links: list[tuple[int, str, int, str]] = []
        self.active_uid = 0
        self.changed = Signal()
        self._muted = 0

    # ---- 查询 ----
    def node(self, uid: int) -> ShaderNode | None:
        return self.nodes.get(uid)

    @property
    def active(self) -> ShaderNode | None:
        return self.nodes.get(self.active_uid)

    def output_node(self) -> ShaderNode | None:
        """生效的材质输出：当前选中的那个输出节点，否则第一个。"""
        outputs = [n for n in self.nodes.values() if n.type_id == "OUTPUT_MATERIAL"]
        if not outputs:
            return None
        active = self.active
        return active if active in outputs else outputs[0]

    def input_link(self, uid: int, socket: str) -> tuple[int, str] | None:
        for a, sa, b, sb in self.links:
            if b == uid and sb == socket:
                return a, sa
        return None

    def output_links(self, uid: int, socket: str | None = None) -> list[tuple[int, str]]:
        return [(b, sb) for a, sa, b, sb in self.links if a == uid and (socket is None or sa == socket)]

    def upstream(self, uid: int) -> set[int]:
        """uid 依赖的所有节点（不含自己）。"""
        seen: set[int] = set()
        stack = [uid]
        while stack:
            current = stack.pop()
            for a, _sa, b, _sb in self.links:
                if b == current and a not in seen:
                    seen.add(a)
                    stack.append(a)
        return seen

    # ---- 修改 ----
    def _emit(self, kind: str, node=None) -> None:
        if not self._muted:
            self.changed.emit(kind, node)

    def _watch(self, node: ShaderNode) -> None:
        def on_value(name: str, node=node) -> None:
            prop = type(node.values).properties().get(name)
            structural = getattr(prop, "kind", "") in _STRUCTURAL_KINDS
            self._emit("structure" if structural else "values", node)

        node.values.changed.connect(on_value)

    def add_node(self, type_id: str, location: tuple = (0.0, 0.0), uid: int | None = None) -> ShaderNode:
        node = ShaderNode(type_id, uid, location)
        self.nodes[node.uid] = node
        self._watch(node)
        self.active_uid = node.uid
        self._emit("structure", node)
        return node

    def remove_node(self, uid: int) -> None:
        if uid not in self.nodes:
            return
        node = self.nodes.pop(uid)
        self.links = [link for link in self.links if link[0] != uid and link[2] != uid]
        if self.active_uid == uid:
            self.active_uid = 0
        self._emit("structure", node)

    def link(self, a: int, socket_a: str, b: int, socket_b: str) -> bool:
        """接线（替换下游插口原来的线）。成环、插口不存在时不接，返回 False。"""
        if a == b or a not in self.nodes or b not in self.nodes:
            return False
        if socket_a not in self.nodes[a].output_names() or socket_b not in self.nodes[b].input_names():
            return False
        if b in self.upstream(a) or a == b:
            return False
        self.links = [link for link in self.links if not (link[2] == b and link[3] == socket_b)]
        self.links.append((a, socket_a, b, socket_b))
        self._emit("structure", self.nodes[b])
        return True

    def unlink(self, b: int, socket_b: str) -> bool:
        before = len(self.links)
        self.links = [link for link in self.links if not (link[2] == b and link[3] == socket_b)]
        if len(self.links) != before:
            self._emit("structure", self.nodes.get(b))
            return True
        return False

    def remove_links(self, links: list) -> None:
        drop = set(links)
        if not drop:
            return
        self.links = [link for link in self.links if link not in drop]
        self._emit("structure", None)

    def set_mute(self, uid: int, mute: bool) -> None:
        node = self.nodes.get(uid)
        if node is not None and node.mute != bool(mute):
            node.mute = bool(mute)
            self._emit("structure", node)

    def set_hide(self, uid: int, hide: bool) -> None:
        node = self.nodes.get(uid)
        if node is not None and node.hide != bool(hide):
            node.hide = bool(hide)
            self._emit("layout", node)

    def move_node(self, uid: int, location: tuple) -> None:
        node = self.nodes.get(uid)
        if node is not None:
            node.location = (float(location[0]), float(location[1]))
            self._emit("layout", node)

    def set_active(self, uid: int) -> None:
        if uid != self.active_uid and (uid == 0 or uid in self.nodes):
            self.active_uid = uid
            self._emit("active", self.nodes.get(uid))

    # ---- 存取、撤销 ----
    def to_dict(self) -> dict:
        return {"uid": self.uid, "nodes": [n.to_dict() for n in self.nodes.values()],
                "links": [list(link) for link in self.links], "active": self.active_uid}

    def load(self, data: dict, emit: bool = True) -> None:
        """整图换成 data 的内容（撤销、打开工程时用），节点编号不变。"""
        self._muted += 1
        try:
            # 编号和类型都对得上的节点沿用原来的对象（界面上的控件、撤销步骤还指着它们）
            old = self.nodes
            fresh: dict[int, ShaderNode] = {}
            for item in data.get("nodes", []):
                type_id = item.get("type", "")
                if nodes.get(type_id) is None:
                    continue
                uid = int(item.get("uid", 0)) or None
                node = old.get(uid) if uid is not None else None
                if node is None or node.type_id != type_id:
                    node = ShaderNode(type_id, uid)
                    self._watch(node)
                node.location = tuple(float(v) for v in item.get("location", (0, 0)))
                node.name = item.get("name", "")
                node.mute = bool(item.get("mute", False))
                node.hide = bool(item.get("hide", False))
                node.values.from_dict(item.get("values") or {})
                reserve_uid(node.uid)
                fresh[node.uid] = node
            self.nodes = fresh
            self.links = []
            for item in data.get("links", []):
                try:
                    a, sa, b, sb = int(item[0]), str(item[1]), int(item[2]), str(item[3])
                except (TypeError, ValueError, IndexError):
                    continue
                if a in self.nodes and b in self.nodes and sa in self.nodes[a].output_names() and \
                        sb in self.nodes[b].input_names():
                    self.links.append((a, sa, b, sb))
            self.active_uid = int(data.get("active", 0))
            if self.active_uid not in self.nodes:
                self.active_uid = 0
        finally:
            self._muted -= 1
        if emit:
            self._emit("structure", None)

    @classmethod
    def from_dict(cls, data: dict) -> "ShaderGraph":
        graph = cls(int(data.get("uid", 0)) or None)
        reserve_uid(graph.uid)
        graph.load(data, emit=False)
        return graph

    @classmethod
    def default(cls) -> "ShaderGraph":
        """新材质：图层堆栈直接接到材质输出（和没有材质节点时一样）。"""
        graph = cls()
        graph._muted += 1
        try:
            stack = graph.add_node("LAYER_STACK", (-260.0, 0.0))
            out = graph.add_node("OUTPUT_MATERIAL", (60.0, 0.0))
            for name in ("base_color", "metallic", "roughness", "height"):
                graph.link(stack.uid, name, out.uid, name)
            graph.active_uid = out.uid
        finally:
            graph._muted -= 1
        return graph

    def copy(self) -> "ShaderGraph":
        data = self.to_dict()
        return ShaderGraph.from_dict(dict(data, uid=0))


def snapshot(graph: ShaderGraph) -> dict:
    return graph.to_dict()


def restore(graph: ShaderGraph, data: dict) -> None:
    graph.load(data)


def describe(graph: ShaderGraph) -> Any:
    """调试用：节点类型和连线的简表。"""
    return [(n.uid, n.type_id) for n in graph.nodes.values()], list(graph.links)
