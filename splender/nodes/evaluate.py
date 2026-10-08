"""节点图在显卡上求值。

- 每个节点画一遍全屏三角形，输出一张 RGBA16F 贴图（可平铺、带多级）；
- 节点的签名 = 类型 + 参数 + 分辨率 + 各输入的签名，签名没变就直接用上次的结果，拖一个参数只重算它和下游；
- 材质输出用计算着色器写进两层的贴图数组：第 0 层基础色（sRGB），第 1 层 (金属度, 粗糙度, 高度, 1)。
要求调用时引擎的显卡上下文是当前上下文。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

import moderngl
import numpy as np

from . import glsl, library

log = logging.getLogger("splender.nodes")

_VERT = """#version 430
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FRAG_MAIN = """
out vec4 o_color;
void main() {
  vec2 uv = gl_FragCoord.xy / u_size;
  o_color = node(uv);
}
"""

_OUTPUT_COMPUTE = """
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba16f, binding = 0) writeonly uniform image2DArray o_out;
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= int(u_size.x) || p.y >= int(u_size.y)) return;
  vec2 uv = (vec2(p) + 0.5) / u_size;
  vec4 base = IN0(uv);
  imageStore(o_out, ivec3(p, 0), vec4(clamp(base.rgb, 0.0, 1.0), 1.0));
  imageStore(o_out, ivec3(p, 1), vec4(clamp(G1(uv), 0.0, 1.0), clamp(G2(uv), 0.0, 1.0), clamp(G3(uv), 0.0, 1.0), 1.0));
}
"""

_THUMB_FRAG = """#version 430
uniform sampler2D u_src;
uniform sampler2DArray u_src_array;
uniform int u_array;
uniform float u_lod;
uniform vec2 u_size;
out vec4 o_color;
void main() {
  vec2 uv = gl_FragCoord.xy / u_size;
  vec4 c = u_array == 1 ? textureLod(u_src_array, vec3(uv, 0.0), u_lod) : textureLod(u_src, uv, u_lod);
  o_color = vec4(clamp(c.rgb, 0.0, 1.0), 1.0);
}
"""


def _enum_index(prop, value) -> int:
    for index, item in enumerate(prop.items()):
        if item[0] == value:
            return index
    return 0


def set_params(program, params) -> None:
    """参数组的值写进 p_名字 uniform。"""
    _set_ramps(program, params)
    for name, prop in params.properties().items():
        key = "p_" + name
        if key not in program:
            continue
        value = getattr(params, name)
        kind = getattr(prop, "kind", "")
        if kind == "FLOAT":
            program[key] = float(value)
        elif kind == "INT":
            program[key] = int(value)
        elif kind == "BOOL":
            program[key] = 1 if value else 0
        elif kind == "ENUM":
            program[key] = _enum_index(prop, value)
        elif kind == "COLOR":
            program[key] = tuple(float(v) for v in value)


def _set_ramps(program, params) -> None:
    from ..core import ramp

    for name, prop in params.properties().items():
        if getattr(prop, "kind", "") == "RAMP":
            ramp.set_uniforms(program, name, getattr(params, name))


class GraphEvaluator:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._programs: dict[str, tuple] = {}
        self._nodes: dict[tuple, dict] = {}        # (图, 节点) -> {"sig", "tex", "fbo"}
        self._outputs: dict[int, dict] = {}        # 图 -> {"sig", "tex"}
        self._thumb_prog = None
        self._thumb_vao = None
        self.stat_rendered = 0
        self.stat_ms = 0.0

    # ---------------------------------------------------------------- 着色器
    def _program(self, type_id: str):
        found = self._programs.get(type_id)
        if found is not None:
            return found
        node_type = library.get(type_id)
        source = glsl.HEADER + glsl.uniform_lines(node_type.Params) + glsl.LIBRARY
        if type_id == "OUTPUT":
            program = self.ctx.compute_shader(source + _OUTPUT_COMPUTE)
            found = (program, None)
        else:
            program = self.ctx.program(vertex_shader=_VERT, fragment_shader=source + node_type.glsl + _FRAG_MAIN)
            found = (program, self.ctx.vertex_array(program, []))
        self._programs[type_id] = found
        return found

    # ---------------------------------------------------------------- 签名
    def _signature(self, graph, uid: int, memo: dict) -> str:
        found = memo.get(uid)
        if found is not None:
            return found
        node = graph.nodes[uid]
        parts = [node.type_id, graph.resolution, json.dumps(node.params.to_dict(), sort_keys=True)]
        for index in range(len(node.type.inputs)):
            src = graph.input_link(uid, index)
            parts.append(self._signature(graph, src, memo) if src is not None else "-")
        sig = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=16).hexdigest()
        memo[uid] = sig
        return sig

    # ---------------------------------------------------------------- 求值
    def _bind_inputs(self, program, graph, node, memo: dict) -> None:
        linked = [0, 0, 0, 0]
        for index, (_name, _label, default, _gray) in enumerate(node.type.inputs[:4]):
            src = graph.input_link(node.uid, index)
            key = "u_def%d" % index
            if key in program:
                program[key] = tuple(float(v) for v in default)
            if src is None:
                continue
            texture = self.node_texture(graph, src, memo)
            if texture is None:
                continue
            texture.use(index)
            if "u_in%d" % index in program:
                program["u_in%d" % index] = index
            linked[index] = 1
        if "u_linked" in program:
            program["u_linked"] = tuple(linked)

    def node_texture(self, graph, uid: int, memo: dict | None = None):
        """节点的输出贴图（按需求值）。输出节点没有贴图，返回 None。"""
        memo = {} if memo is None else memo
        node = graph.nodes.get(uid)
        if node is None or node.type_id == "OUTPUT":
            return None
        sig = self._signature(graph, uid, memo)
        key = (graph.uid, uid)
        entry = self._nodes.get(key)
        size = graph.size
        if entry is not None and entry["sig"] == sig:
            return entry["tex"]
        started = time.perf_counter()
        if entry is None or entry["tex"].size != (size, size):
            if entry is not None:
                entry["fbo"].release()
                entry["tex"].release()
            tex = self.ctx.texture((size, size), 4, dtype="f2")
            tex.repeat_x = tex.repeat_y = True
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            entry = {"tex": tex, "fbo": self.ctx.framebuffer([tex]), "sig": ""}
            self._nodes[key] = entry
        program, vao = self._program(node.type_id)
        self._bind_inputs(program, graph, node, memo)
        set_params(program, node.params)
        if "u_size" in program:
            program["u_size"] = (float(size), float(size))
        ctx = self.ctx
        entry["fbo"].use()
        ctx.viewport = (0, 0, size, size)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        vao.render(moderngl.TRIANGLES, vertices=3)
        entry["tex"].build_mipmaps()
        entry["sig"] = sig
        self.stat_rendered += 1
        self.stat_ms += (time.perf_counter() - started) * 1000.0
        return entry["tex"]

    def output(self, graph):
        """材质输出（两层的贴图数组），图里没有输出节点时返回 None。"""
        node = graph.output_node()
        if node is None:
            return None
        memo: dict = {}
        sig = self._signature(graph, node.uid, memo)
        entry = self._outputs.get(graph.uid)
        size = graph.size
        if entry is not None and entry["sig"] == sig:
            return entry["tex"]
        if entry is None or entry["tex"].size[:2] != (size, size):
            if entry is not None:
                entry["tex"].release()
            # 先不设多级过滤：没生成多级之前贴图「不完整」，往不完整的贴图做图像写入会被忽略
            tex = self.ctx.texture_array((size, size, 2), 4, dtype="f2")
            tex.repeat_x = tex.repeat_y = True
            entry = {"tex": tex, "sig": ""}
            self._outputs[graph.uid] = entry
        program, _vao = self._program("OUTPUT")
        self._bind_inputs(program, graph, node, memo)
        if "u_size" in program:
            program["u_size"] = (float(size), float(size))
        entry["tex"].bind_to_image(0, read=False, write=True)
        program.run((size + 7) // 8, (size + 7) // 8, 1)
        self.ctx.memory_barrier()
        entry["tex"].build_mipmaps()
        entry["tex"].filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        entry["sig"] = sig
        return entry["tex"]

    def output_signature(self, graph) -> str:
        node = graph.output_node()
        return self._signature(graph, node.uid, {}) if node is not None else ""

    # ---------------------------------------------------------------- 读回
    def read_node(self, graph, uid: int) -> np.ndarray:
        """节点输出读回成 (size, size, 4) float32，第 0 行是 v=0（测试用）。"""
        texture = self.node_texture(graph, uid)
        size = texture.size[0]
        return np.frombuffer(texture.read(), np.float16).reshape(size, size, 4).astype(np.float32)

    def read_output(self, graph) -> np.ndarray | None:
        texture = self.output(graph)
        if texture is None:
            return None
        size = texture.size[0]
        return np.frombuffer(texture.read(), np.float16).reshape(2, size, size, 4).astype(np.float32)

    def thumbnail(self, graph, uid: int, size: int = 96) -> np.ndarray | None:
        """节点的缩略图 (size, size, 4) uint8，第 0 行是图片顶部。输出节点显示基础色。"""
        node = graph.nodes.get(uid)
        if node is None:
            return None
        if node.type_id == "OUTPUT":
            source = self.output(graph)
        else:
            source = self.node_texture(graph, uid)
        if source is None:
            return None
        ctx = self.ctx
        if self._thumb_prog is None:
            self._thumb_prog = ctx.program(vertex_shader=_VERT, fragment_shader=_THUMB_FRAG)
            self._thumb_vao = ctx.vertex_array(self._thumb_prog, [])
        target = ctx.texture((size, size), 4, dtype="f1")
        fbo = ctx.framebuffer([target])
        try:
            program = self._thumb_prog
            is_array = node.type_id == "OUTPUT"
            # 两种采样器必须指向不同的贴图单元，否则这次绘制无效
            source.use(1 if is_array else 0)
            program["u_src"] = 0
            program["u_src_array"] = 1
            program["u_array"] = 1 if is_array else 0
            program["u_lod"] = float(max(0.0, np.log2(max(graph.size / size, 1.0))))
            program["u_size"] = (float(size), float(size))
            fbo.use()
            ctx.viewport = (0, 0, size, size)
            ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
            self._thumb_vao.render(moderngl.TRIANGLES, vertices=3)
            data = np.frombuffer(fbo.read(components=4, alignment=1), np.uint8).reshape(size, size, 4)
            return np.ascontiguousarray(data[::-1])
        finally:
            fbo.release()
            target.release()

    # ---------------------------------------------------------------- 释放
    def forget(self, graph_uid: int, keep: set | None = None) -> None:
        """丢掉一张图（或它已经删掉的节点）的缓存。keep 给了就只丢不在里面的节点。"""
        for key in [k for k in self._nodes if k[0] == graph_uid and (keep is None or k[1] not in keep)]:
            entry = self._nodes.pop(key)
            entry["fbo"].release()
            entry["tex"].release()
        if keep is None:
            entry = self._outputs.pop(graph_uid, None)
            if entry is not None:
                entry["tex"].release()

    def release(self) -> None:
        for graph_uid in {k[0] for k in self._nodes} | set(self._outputs):
            self.forget(graph_uid)
        for program, vao in self._programs.values():
            if vao is not None:
                vao.release()
            program.release()
        self._programs.clear()
        if self._thumb_prog is not None:
            self._thumb_vao.release()
            self._thumb_prog.release()
            self._thumb_prog = None
