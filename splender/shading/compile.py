"""材质节点图 → 合成器着色器里的一个函数：

    void material_graph(inout vec3 base, inout float metal, inout float rough, inout float height)

进来的是图层堆栈叠好的结果（「图层堆栈」节点读它），材质输出节点写回去。只编译材质输出用到的节点。
没接线的输入、数值参数、色带都放在存储缓冲 mg[] 里（Compiled.values 现算），改数值不用重新编译；
下拉、开关、连线变了才换代码（Compiled.code 是缓存着色器的键）。
"""
from __future__ import annotations

import numpy as np

from ..core import ramp as ramp_mod
from . import nodes
from .graph import ShaderGraph

RAMP_BLOCK = 41
_IMPLICIT = {"GENERATED": "mg_generated()", "UV": "vec3(g_uv, 0.0)", "OBJECT": "mg_position()",
             "NORMAL": "mg_normal()"}
_ZERO = {"FLOAT": "0.0", "COLOR": "vec3(0.0)", "VECTOR": "vec3(0.0)"}


def srgb_to_linear(c) -> np.ndarray:
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def convert(expr: str, src: str, dst: str) -> str:
    """插口类型之间的自动转换（照 Blender：颜色转数值取亮度，矢量转数值取平均）。"""
    if src == dst or dst == "ANY":
        return expr
    if dst == "FLOAT":
        return "mg_lum(%s)" % expr if src == "COLOR" else "dot(%s, vec3(1.0 / 3.0))" % expr
    if src == "FLOAT":
        return "vec3(%s)" % expr
    return expr


class Compiled:
    def __init__(self, code: str, slots: list, size: int, graph_nodes: set[int]) -> None:
        self.code = code
        self.slots = slots              # [(起点, 种类, 节点编号, 名字, 插口类型)]
        self.size = max(1, size)
        self.graph_nodes = graph_nodes  # 用到「节点图纹理」的节点

    def values(self, graph: ShaderGraph, graph_slot_of=None) -> np.ndarray:
        """当前参数下的存储缓冲内容 (size, 4) float32。graph_slot_of(节点图编号) → 贴图槽序号（-1 没有）。"""
        out = np.zeros((self.size, 4), np.float32)
        for base, kind, uid, name, socket_type in self.slots:
            node = graph.nodes.get(uid)
            if node is None:
                continue
            if kind == "ramp":
                mode, stops = ramp_mod.normalize(getattr(node.values, name))
                out[base, 0] = len(stops)
                out[base, 1] = ramp_mod.MODE_INDEX[mode]
                for k, stop in enumerate(stops):
                    out[base + 1 + k // 4, k % 4] = stop[0]
                    out[base + 9 + k] = stop[1:]
                continue
            if kind == "graph_slot":
                uid_text = getattr(node.values, "graph", "0")
                try:
                    graph_uid = int(uid_text)
                except (TypeError, ValueError):
                    graph_uid = 0
                slot = graph_slot_of(graph_uid) if (graph_slot_of is not None and graph_uid) else -1
                out[base, 0] = float(slot if slot is not None else -1)
                continue
            value = getattr(node.values, ("in_" + name) if kind == "in" else name)
            if socket_type == "FLOAT":
                out[base, 0] = float(value)
            elif socket_type == "COLOR":
                out[base, :3] = srgb_to_linear(np.asarray(value, np.float64)[:3])
            else:
                out[base, :3] = np.asarray(value, np.float64)[:3]
        return out

    @property
    def needs_maps(self) -> bool:
        """用到了模型贴图（位置、法线、遮蔽、曲率、厚度、部件）：没烘焙过的要去烘焙。"""
        return any(key in self.code for key in ("mg_generated", "mg_position", "mg_normal", "g_ma", "g_id"))

    def graph_uids(self, graph: ShaderGraph) -> list[int]:
        """「节点图纹理」节点用到的节点图编号（按出现顺序，不重复）。"""
        out = []
        for uid in sorted(self.graph_nodes):
            node = graph.nodes.get(uid)
            if node is None:
                continue
            try:
                graph_uid = int(getattr(node.values, "graph", "0"))
            except (TypeError, ValueError):
                graph_uid = 0
            if graph_uid and graph_uid not in out:
                out.append(graph_uid)
        return out


class _Compiler:
    def __init__(self, graph: ShaderGraph) -> None:
        self.graph = graph
        self.slots: list = []
        self.size = 0
        self.keys: dict = {}
        self.out_types: dict[tuple[int, str], str] = {}
        self.lines: list[str] = []
        self.graph_nodes: set[int] = set()

    def alloc(self, key: tuple, count: int = 1) -> int:
        if key in self.keys:
            return self.keys[key]
        base = self.size
        self.size += count
        self.keys[key] = base
        return base


class _Cx:
    def __init__(self, compiler: _Compiler, node) -> None:
        self.c = compiler
        self.node = node
        self.uid = node.uid

    def o(self, name: str) -> str:
        return "n%d_%s" % (self.uid, name)

    def p(self, name: str):
        return getattr(self.node.values, name)

    def i(self, name: str) -> str:
        node = self.node
        socket_type = node.input_type(name)
        link = self.c.graph.input_link(self.uid, name)
        if link is not None and link in self.c.out_types:
            return convert("n%d_%s" % link, self.c.out_types[link], socket_type)
        opts = next((item[4] for item in node.type.inputs if item[0] == name), {})
        if opts.get("implicit"):
            return _IMPLICIT[opts["implicit"]]
        if socket_type == "ANY":
            return "0.0"
        base = self.c.alloc(("in", self.uid, name))
        self.c.slots.append((base, "in", self.uid, name, socket_type))
        return "mg[%d].x" % base if socket_type == "FLOAT" else "mg[%d].xyz" % base

    def slot(self, name: str, socket_type: str) -> str:
        if name == "@graph_slot":
            self.c.graph_nodes.add(self.uid)
            key = ("graph_slot", self.uid, "graph")
            if key not in self.c.keys:
                base = self.c.alloc(key)
                self.c.slots.append((base, "graph_slot", self.uid, "graph", "FLOAT"))
            return "mg[%d].x" % self.c.keys[key]
        key = ("param", self.uid, name)
        if key not in self.c.keys:
            base = self.c.alloc(key)
            self.c.slots.append((base, "param", self.uid, name, socket_type))
        base = self.c.keys[key]
        return "mg[%d].x" % base if socket_type == "FLOAT" else "mg[%d].xyz" % base

    def ramp(self, name: str) -> int:
        key = ("ramp", self.uid, name)
        if key not in self.c.keys:
            base = self.c.alloc(key, RAMP_BLOCK)
            self.c.slots.append((base, "ramp", self.uid, name, "COLOR"))
        return self.c.keys[key]


def _order(graph: ShaderGraph, out_uid: int) -> list[int]:
    """输出节点依赖的节点，按先算上游的顺序。"""
    needed = graph.upstream(out_uid) | {out_uid}
    order: list[int] = []
    visiting: set[int] = set()

    def visit(uid: int) -> None:
        if uid in order or uid in visiting:
            return
        visiting.add(uid)
        for a, _sa, b, _sb in graph.links:
            if b == uid and a in needed:
                visit(a)
        visiting.discard(uid)
        order.append(uid)

    visit(out_uid)
    return order


def compile_graph(graph: ShaderGraph | None) -> Compiled | None:
    """没有图、没有输出节点时返回 None（直接用图层堆栈）。"""
    if graph is None:
        return None
    out = graph.output_node()
    if out is None:
        return None
    c = _Compiler(graph)
    for uid in _order(graph, out.uid):
        node = graph.nodes[uid]
        cx = _Cx(c, node)
        node_type = node.type
        # 转接点：输出类型跟着接进来的线
        if node.type_id == "REROUTE":
            link = graph.input_link(uid, "input")
            src_type = c.out_types.get(link, "FLOAT") if link is not None else "FLOAT"
            expr = ("n%d_%s" % link) if link is not None and link in c.out_types else "0.0"
            c.lines.append("%s n%d_output = %s;" % (nodes.GLSL_TYPE[src_type], uid, expr))
            c.out_types[(uid, "output")] = src_type
            continue
        for name, _label, kind in node_type.outputs:
            c.lines.append("%s %s = %s;" % (nodes.GLSL_TYPE[kind], cx.o(name), _ZERO[kind]))
            c.out_types[(uid, name)] = kind
        if node.mute:
            # 静音：每个输出取第一个同类型的输入（没有就是 0），和 Blender 一样
            for name, _label, kind in node_type.outputs:
                same = next((item[0] for item in node_type.inputs if item[2] == kind), None)
                expr = cx.i(same) if same is not None else _ZERO[kind]
                c.lines.append("%s = %s;" % (cx.o(name), expr))
            continue
        c.lines.append("// %s" % node_type.id)
        c.lines += node_type.emit(cx)
    body = "\n  ".join(c.lines)
    code = ("void material_graph(inout vec3 base, inout float metal, inout float rough, inout float height) {\n"
            "  vec3 st_base = base; float st_metal = metal; float st_rough = rough; float st_height = height;\n"
            "  %s\n}\n" % body)
    return Compiled(code, c.slots, c.size, c.graph_nodes)
