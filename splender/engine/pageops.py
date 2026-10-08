"""对页面的批量操作：清零、生成上一级、把笔划并进图层。

每种操作一帧里只发很少几次显卡指令（按目标数组分组），页数多少都一样。
"""
from __future__ import annotations

import moderngl
import numpy as np

from . import glx
from .pagepool import FORMATS, MAX_ARRAYS, PoolSet, bind_samplers, glsl_fetch

PACK_KIND = {"rgba8": 0, "rg16f": 1, "r8": 2}

# 页面内容的种类，决定缩小和合并时怎么算
KIND_COLOR = 0       # rgba8：sRGB 颜色 + 覆盖度
KIND_MR = 1          # rgba8：两组（数值, 覆盖度）
KIND_HEIGHT = 2      # rg16f：（数值, 覆盖度）
KIND_MASK = 3        # r8：单值
KIND_STROKE = 4      # rg16f：笔划累积覆盖度

KIND_FORMAT = {KIND_COLOR: "rgba8", KIND_MR: "rgba8", KIND_HEIGHT: "rg16f", KIND_MASK: "r8", KIND_STROKE: "rg16f"}

SRGB_GLSL = """
vec3 srgb_to_linear(vec3 c){ return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, lessThanEqual(c, vec3(0.04045))); }
vec3 linear_to_srgb(vec3 c){ c = clamp(c, 0.0, 1.0);
  return mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, c * 12.92, lessThanEqual(c, vec3(0.0031308))); }
"""


class PageOps:
    def __init__(self, ctx: moderngl.Context, pools: PoolSet) -> None:
        self.ctx = ctx
        self.pools = pools
        shift = pools.shift
        self._fetch = "\n".join(glsl_fetch(fmt, MAX_ARRAYS[fmt], shift) for fmt in FORMATS)
        self.job_buffer = ctx.buffer(reserve=8 * 4 * 16384)
        self.aux_buffer = ctx.buffer(reserve=4 * 4 * 16384)
        self._job_capacity = 16384
        self._clear = {}
        self._down = {}
        self._apply = {}
        for fmt in FORMATS:
            glsl = FORMATS[fmt][5]
            self._clear[fmt] = ctx.compute_shader(f"""#version 430
layout(local_size_x=16,local_size_y=16) in;
layout({glsl}, binding=0) writeonly uniform image2DArray u_dst;
layout(std430, binding=1) readonly buffer Jobs {{ int jobs[]; }};
uniform int u_base; uniform vec4 u_value;
void main(){{ int layer = jobs[(u_base + int(gl_GlobalInvocationID.z)) * 8];
  imageStore(u_dst, ivec3(gl_GlobalInvocationID.xy, layer), u_value); }}
""")
            self._down[fmt] = ctx.compute_shader(f"""#version 430
layout(local_size_x=16,local_size_y=16) in;
layout({glsl}, binding=0) uniform image2DArray u_dst;
layout(std430, binding=1) readonly buffer Jobs {{ int jobs[]; }};
layout(std430, binding=2) readonly buffer Aux {{ int aux[]; }};
uniform int u_base; uniform int u_kind; uniform int u_layer_mask; uniform int u_use_aux;
uniform vec4 u_empty;          // 这一象限原来没有页、也没有新的子页时填的值（蒙版是它的初始值，别的是 0）
{self._fetch}
{SRGB_GLSL}
vec4 src(int slot, ivec2 p){{
  if (u_kind == 0 || u_kind == 1) return fetch_rgba8(slot, p);
  if (u_kind == 3) return fetch_r8(slot, p);
  return fetch_rg16f(slot, p);
}}
vec4 reduce(vec4 a, vec4 b, vec4 c, vec4 d, vec4 w){{
  // w：四个子纹素是否有几何覆盖（岛内或扩边）。没覆盖的不参与平均，接缝处的数值才不会被稀释。
  float n = w.x + w.y + w.z + w.w;
  float inv = n > 0.0 ? 1.0 / n : 0.0;
  if (u_kind == 4) {{
    return vec4((a.r*w.x + b.r*w.y + c.r*w.z + d.r*w.w) * inv, n > 0.0 ? 1.0 : 0.0, 0.0, 0.0);
  }}
  if (u_kind == 0) {{
    float s = a.a + b.a + c.a + d.a;
    if (s <= 0.0) return vec4(0.0);
    vec3 rgb = (srgb_to_linear(a.rgb)*a.a + srgb_to_linear(b.rgb)*b.a + srgb_to_linear(c.rgb)*c.a + srgb_to_linear(d.rgb)*d.a) / s;
    return vec4(linear_to_srgb(rgb), (a.a*w.x + b.a*w.y + c.a*w.z + d.a*w.w) * inv);
  }}
  if (u_kind == 1) {{
    float s0 = a.g + b.g + c.g + d.g, s1 = a.a + b.a + c.a + d.a;
    float v0 = s0 > 0.0 ? (a.r*a.g + b.r*b.g + c.r*c.g + d.r*d.g) / s0 : 0.0;
    float v1 = s1 > 0.0 ? (a.b*a.a + b.b*b.a + c.b*c.a + d.b*d.a) / s1 : 0.0;
    return vec4(v0, (a.g*w.x + b.g*w.y + c.g*w.z + d.g*w.w) * inv, v1, (a.a*w.x + b.a*w.y + c.a*w.z + d.a*w.w) * inv);
  }}
  if (u_kind == 2) {{
    float s = a.g + b.g + c.g + d.g;
    float v = s > 0.0 ? (a.r*a.g + b.r*b.g + c.r*c.g + d.r*d.g) / s : 0.0;
    return vec4(v, (a.g*w.x + b.g*w.y + c.g*w.z + d.g*w.w) * inv, 0.0, 0.0);
  }}
  return n > 0.0 ? (a*w.x + b*w.y + c*w.z + d*w.w) * inv : (a + b + c + d) * 0.25;
}}
float covered(int slot, ivec2 p){{ return slot >= 0 ? step(0.5, fetch_rg16f(slot, p).g) : 1.0; }}
void main(){{
  int j = (u_base + int(gl_GlobalInvocationID.z)) * 8;
  int dst = jobs[j]; int old = jobs[j + 1];
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  int q = (p.x >= 128 ? 1 : 0) + (p.y >= 128 ? 2 : 0);
  int child = jobs[j + 2 + q];
  ivec3 target = ivec3(p, dst & u_layer_mask);
  if (child >= 0) {{
    ivec2 s = (p - ivec2((q & 1) * 128, (q >> 1) * 128)) * 2;
    vec4 a = src(child, s), b = src(child, s + ivec2(1, 0)), c = src(child, s + ivec2(0, 1)), d = src(child, s + ivec2(1, 1));
    vec4 w = vec4(1.0);
    if (u_use_aux == 1) {{
      int v = aux[(u_base + int(gl_GlobalInvocationID.z)) * 4 + q];
      w = vec4(covered(v, s), covered(v, s + ivec2(1, 0)), covered(v, s + ivec2(0, 1)), covered(v, s + ivec2(1, 1)));
    }}
    imageStore(u_dst, target, reduce(a, b, c, d, w));
  }} else if (old == dst) {{
    return;                                   // 原地更新：这一象限保持不变
  }} else if (old >= 0) {{
    imageStore(u_dst, target, src(old, p));
  }} else {{
    imageStore(u_dst, target, u_empty);
  }}
}}
""")
        for kind in (KIND_COLOR, KIND_MR, KIND_HEIGHT, KIND_MASK):
            fmt = KIND_FORMAT[kind]
            glsl = FORMATS[fmt][5]
            self._apply[kind] = ctx.compute_shader(f"""#version 430
layout(local_size_x=16,local_size_y=16) in;
layout({glsl}, binding=0) writeonly uniform image2DArray u_dst;
layout(std430, binding=1) readonly buffer Jobs {{ int jobs[]; }};
uniform int u_base; uniform int u_layer_mask;
uniform vec3 u_color;        // 笔刷颜色（sRGB）
uniform vec4 u_values;       // 金属度、粗糙度、高度、蒙版值
uniform ivec4 u_enable;      // 金属度、粗糙度 是否写入
uniform float u_opacity; uniform int u_erase; uniform float u_mask_default;
{self._fetch}
{SRGB_GLSL}
const int KIND = {kind};
vec2 over(vec2 old, float value, float a){{       // old = (数值, 覆盖度)
  float c = a + old.y * (1.0 - a);
  return vec2(c > 0.0 ? (value * a + old.x * old.y * (1.0 - a)) / c : 0.0, c);
}}
void main(){{
  int j = (u_base + int(gl_GlobalInvocationID.z)) * 8;
  int dst = jobs[j]; int old = jobs[j + 1]; int stroke = jobs[j + 2];
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  float a = clamp(fetch_rg16f(stroke, p).r, 0.0, 1.0) * u_opacity;
  ivec3 target = ivec3(p, dst & u_layer_mask);
  if (KIND == 0) {{
    vec4 o = old >= 0 ? fetch_rgba8(old, p) : vec4(0.0);
    if (u_erase == 1) {{ imageStore(u_dst, target, vec4(o.rgb, o.a * (1.0 - a))); return; }}
    float c = a + o.a * (1.0 - a);
    vec3 rgb = c > 0.0 ? (srgb_to_linear(u_color) * a + srgb_to_linear(o.rgb) * o.a * (1.0 - a)) / c : vec3(0.0);
    imageStore(u_dst, target, vec4(linear_to_srgb(rgb), c));
  }} else if (KIND == 1) {{
    vec4 o = old >= 0 ? fetch_rgba8(old, p) : vec4(0.0);
    vec2 m = o.rg, r = o.ba;
    if (u_erase == 1) {{ if (u_enable.x == 1) m.y *= (1.0 - a); if (u_enable.y == 1) r.y *= (1.0 - a); }}
    else {{ if (u_enable.x == 1) m = over(m, u_values.x, a); if (u_enable.y == 1) r = over(r, u_values.y, a); }}
    imageStore(u_dst, target, vec4(m, r));
  }} else if (KIND == 2) {{
    vec2 o = old >= 0 ? fetch_rg16f(old, p).rg : vec2(0.0);
    if (u_erase == 1) o.y *= (1.0 - a); else o = over(o, u_values.z, a);
    imageStore(u_dst, target, vec4(o, 0.0, 0.0));
  }} else {{
    float o = old >= 0 ? fetch_r8(old, p).r : u_mask_default;
    imageStore(u_dst, target, vec4(mix(o, u_values.w, a)));
  }}
}}
""")

        # 读回打包：把若干页按 glGetTexImage 的字节排列写进一块常驻映射缓冲，一次调度完成
        self._pack = ctx.compute_shader(f"""#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(std430, binding=1) readonly buffer Jobs {{ int jobs[]; }};
layout(std430, binding=3) writeonly buffer Out {{ uint words[]; }};
uniform int u_kind;
{self._fetch}
void main(){{
  int j = int(gl_GlobalInvocationID.z) * 8;
  int slot = jobs[j]; int offset = jobs[j + 1];
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (u_kind == 0) {{
    words[offset + p.y * 256 + p.x] = packUnorm4x8(fetch_rgba8(slot, p));
  }} else if (u_kind == 1) {{
    words[offset + p.y * 256 + p.x] = packHalf2x16(fetch_rg16f(slot, p).rg);
  }} else if (p.x < 64) {{
    ivec2 q = ivec2(p.x * 4, p.y);
    vec4 v = vec4(fetch_r8(slot, q).r, fetch_r8(slot, q + ivec2(1, 0)).r, fetch_r8(slot, q + ivec2(2, 0)).r,
                  fetch_r8(slot, q + ivec2(3, 0)).r);
    words[offset + p.y * 64 + p.x] = packUnorm4x8(v);
  }}
}}
""")

        # 纯色检测，分两遍：第一遍每个 16×16 小块判断是否与左下角纹素相同；第二遍每页汇总 256 个小块的结果
        bits = """
uint texel_bits(int slot, ivec2 p){
  if (u_kind == 0) return packUnorm4x8(fetch_rgba8(slot, p));
  if (u_kind == 1) return packHalf2x16(fetch_rg16f(slot, p).rg);
  return packUnorm4x8(vec4(fetch_r8(slot, p).r, 0.0, 0.0, 0.0));
}
"""
        self._uniform_tiles = ctx.compute_shader(f"""#version 430
layout(local_size_x=16,local_size_y=16) in;
layout(std430, binding=1) readonly buffer Jobs {{ int jobs[]; }};
layout(std430, binding=4) writeonly buffer Partial {{ uint partial[]; }};
uniform int u_kind;
{self._fetch}
{bits}
shared uint differs;
void main(){{
  int page = int(gl_WorkGroupID.z);
  int slot = jobs[page * 8];
  if (gl_LocalInvocationIndex == 0u) differs = 0u;
  barrier();
  uint first = texel_bits(slot, ivec2(0));
  if (texel_bits(slot, ivec2(gl_GlobalInvocationID.xy)) != first) atomicOr(differs, 1u);
  barrier();
  if (gl_LocalInvocationIndex == 0u) partial[page * 256 + int(gl_WorkGroupID.y) * 16 + int(gl_WorkGroupID.x)] = differs;
}}
""")
        self._uniform_reduce = ctx.compute_shader(f"""#version 430
layout(local_size_x=64) in;
layout(std430, binding=1) readonly buffer Jobs {{ int jobs[]; }};
layout(std430, binding=4) readonly buffer Partial {{ uint partial[]; }};
layout(std430, binding=3) writeonly buffer Out {{ uint words[]; }};
uniform int u_kind; uniform int u_count;
{self._fetch}
{bits}
void main(){{
  int page = int(gl_GlobalInvocationID.x);
  if (page >= u_count) return;
  uint d = 0u;
  for (int i = 0; i < 256; i++) d |= partial[page * 256 + i];
  int slot = jobs[page * 8];
  words[jobs[page * 8 + 1]] = d == 0u ? 1u : 0u;
  words[jobs[page * 8 + 1] + 1] = texel_bits(slot, ivec2(0));
}}
""")
        self._partial = None
        # 填色：每个任务给一个打包好的值，把整页填成这个值（纯色页换入用，不走 PCIe）
        self._fill = {}
        for fmt in FORMATS:
            glsl = FORMATS[fmt][5]
            self._fill[fmt] = ctx.compute_shader(f"""#version 430
layout(local_size_x=16,local_size_y=16) in;
layout({glsl}, binding=0) writeonly uniform image2DArray u_dst;
layout(std430, binding=1) readonly buffer Jobs {{ int jobs[]; }};
uniform int u_base; uniform int u_layer_mask; uniform int u_kind;
void main(){{
  int j = (u_base + int(gl_GlobalInvocationID.z)) * 8;
  uint v = uint(jobs[j + 1]);
  vec4 value = u_kind == 1 ? vec4(unpackHalf2x16(v), 0.0, 0.0) : unpackUnorm4x8(v);
  imageStore(u_dst, ivec3(gl_GlobalInvocationID.xy, jobs[j] & u_layer_mask), value);
}}
""")

    # ------------------------------------------------------------------ 内部
    def _upload_jobs(self, jobs: np.ndarray) -> None:
        count = len(jobs)
        if count > self._job_capacity:
            self._job_capacity = int(count * 1.5)
            self.job_buffer.orphan(self._job_capacity * 32)
            self.aux_buffer.orphan(self._job_capacity * 16)
        self.job_buffer.write(np.ascontiguousarray(jobs, np.int32).tobytes())

    def _bind_fetch(self, program, first_unit: int = 0) -> None:
        unit = first_unit
        for fmt in FORMATS:
            unit = bind_samplers(program, self.pools, fmt, unit)

    def _run_by_array(self, program, fmt: str, jobs: np.ndarray, aux: np.ndarray | None = None) -> None:
        """jobs 的第 0 列是目标槽号。按目标所在的数组分组各发一次。aux 是每个任务附带的 4 个整数。"""
        pool = self.pools[fmt]
        arrays = jobs[:, 0] >> pool.shift
        order = np.argsort(arrays, kind="stable")
        jobs = jobs[order]
        arrays = arrays[order]
        self._upload_jobs(jobs)
        self.job_buffer.bind_to_storage_buffer(1)
        if "u_use_aux" in program:
            program["u_use_aux"] = int(aux is not None)
        if aux is not None:
            self.aux_buffer.write(np.ascontiguousarray(aux[order], np.int32).tobytes())
            self.aux_buffer.bind_to_storage_buffer(2)
        if "u_layer_mask" in program:
            program["u_layer_mask"] = pool.layers - 1
        starts = np.flatnonzero(np.r_[True, arrays[1:] != arrays[:-1]])
        ends = np.r_[starts[1:], len(arrays)]
        for start, end in zip(starts, ends):
            pool.arrays[int(arrays[start])].bind_to_image(0, read=True, write=True)
            program["u_base"] = int(start)
            program.run(16, 16, int(end - start))
        self.ctx.memory_barrier()

    # ------------------------------------------------------------------ 公开操作
    def clear(self, fmt: str, slots: np.ndarray, value=(0.0, 0.0, 0.0, 0.0)) -> None:
        if len(slots) == 0:
            return
        pool = self.pools[fmt]
        jobs = np.zeros((len(slots), 8), np.int32)
        jobs[:, 0] = slots
        program = self._clear[fmt]
        program["u_value"] = tuple(float(v) for v in value)
        # 清零着色器直接用层号，先把槽号换成层号再按数组分组
        arrays = jobs[:, 0] >> pool.shift
        order = np.argsort(arrays, kind="stable")
        jobs = jobs[order]
        arrays = arrays[order]
        jobs[:, 0] &= pool.layers - 1
        self._upload_jobs(jobs)
        self.job_buffer.bind_to_storage_buffer(1)
        starts = np.flatnonzero(np.r_[True, arrays[1:] != arrays[:-1]])
        ends = np.r_[starts[1:], len(arrays)]
        for start, end in zip(starts, ends):
            pool.arrays[int(arrays[start])].bind_to_image(0, read=False, write=True)
            program["u_base"] = int(start)
            program.run(16, 16, int(end - start))
        self.ctx.memory_barrier()

    def downsample(self, kind: int, dst: np.ndarray, old: np.ndarray, children: np.ndarray,
                   coverage: np.ndarray | None = None, empty: float = 0.0) -> None:
        """生成上一级的页。dst/old 形状 (n,)，children 形状 (n,4)，-1 表示该象限不变。
        empty：没有旧页的象限、又没有新的子页时填什么（蒙版是它的初始值，缺页的格按它算；别的种类 0）。

        coverage 形状 (n,4)：与 children 对应的「几何覆盖」页槽号（rg16f，第二分量大于 0.5 表示有几何），
        给了就只在有几何的子纹素之间平均；-1 或不给表示全部有效。目前没有调用方传它：
        笔划页的覆盖标记只在本笔碰过的格子里可靠，接缝靠足够宽的扩边解决。
        """
        if len(dst) == 0:
            return
        fmt = KIND_FORMAT[kind]
        jobs = np.full((len(dst), 8), -1, np.int32)
        jobs[:, 0] = dst
        jobs[:, 1] = old
        jobs[:, 2:6] = children
        program = self._down[fmt]
        program["u_kind"] = kind
        program["u_empty"] = (float(empty), 0.0, 0.0, 0.0) if kind == KIND_MASK else (0.0, 0.0, 0.0, 0.0)
        self._bind_fetch(program)
        self._run_by_array(program, fmt, jobs, coverage)

    def apply_stroke(self, kind: int, dst: np.ndarray, old: np.ndarray, stroke: np.ndarray, *, color, values,
                     enable, opacity: float, erase: bool, mask_default: float = 1.0) -> None:
        """把笔划并进图层：dst 是新页，old 是原来的页（-1 表示没有），stroke 是笔划页。"""
        if len(dst) == 0:
            return
        fmt = KIND_FORMAT[kind]
        jobs = np.full((len(dst), 8), -1, np.int32)
        jobs[:, 0] = dst
        jobs[:, 1] = old
        jobs[:, 2] = stroke
        program = self._apply[kind]
        for name, value in (("u_color", tuple(color)), ("u_values", tuple(values)), ("u_enable", tuple(enable)),
                            ("u_opacity", float(opacity)), ("u_erase", int(erase)), ("u_mask_default", float(mask_default))):
            if name in program:
                program[name] = value
        self._bind_fetch(program)
        self._run_by_array(program, fmt, jobs)

    def pack(self, fmt: str, slots, byte_offsets, buffer_handle: int) -> None:
        """把这些槽的页读进常驻映射缓冲（异步；之后用 Fence 判断完成）。byte_offsets 是每页在缓冲里的起点。"""
        count = len(slots)
        if count == 0:
            return
        jobs = np.zeros((count, 8), np.int32)
        jobs[:, 0] = slots
        jobs[:, 1] = np.asarray(byte_offsets, np.int64) // 4
        self._upload_jobs(jobs)
        self.job_buffer.bind_to_storage_buffer(1)
        glx.bind_storage(3, buffer_handle)
        program = self._pack
        program["u_kind"] = PACK_KIND[fmt]
        self._bind_fetch(program)
        program.run(16, 16, count)
        glx.client_barrier()
        glx.bind_storage(3, 0)

    def check_uniform(self, fmt: str, slots, word_offsets, buffer_handle: int) -> None:
        """检查这些页是不是纯色。结果写进常驻映射缓冲：每页两个 uint（是否纯色, 打包的值）。"""
        count = len(slots)
        if count == 0:
            return
        jobs = np.zeros((count, 8), np.int32)
        jobs[:, 0] = slots
        jobs[:, 1] = word_offsets
        self._upload_jobs(jobs)
        if self._partial is None or self._partial.size < count * 1024:
            if self._partial is not None:
                self._partial.release()
            self._partial = self.ctx.buffer(reserve=max(count, 1024) * 1024)
        self.job_buffer.bind_to_storage_buffer(1)
        self._partial.bind_to_storage_buffer(4)
        kind = PACK_KIND[fmt]
        first = self._uniform_tiles
        first["u_kind"] = kind
        self._bind_fetch(first)
        first.run(16, 16, count)
        self.ctx.memory_barrier()
        second = self._uniform_reduce
        second["u_kind"] = kind
        second["u_count"] = count
        self._bind_fetch(second)
        glx.bind_storage(3, buffer_handle)
        second.run((count + 63) // 64, 1, 1)
        glx.client_barrier()
        glx.bind_storage(3, 0)

    def fill_packed(self, fmt: str, slots, packed_values) -> None:
        """把这些页整页填成各自的值（值按 check_uniform 的打包方式）。"""
        if len(slots) == 0:
            return
        jobs = np.zeros((len(slots), 8), np.int32)
        jobs[:, 0] = slots
        jobs[:, 1] = np.asarray(packed_values, np.uint32).view(np.int32)
        program = self._fill[fmt]
        if "u_kind" in program:
            program["u_kind"] = PACK_KIND[fmt]
        self._run_by_array(program, fmt, jobs)

    def release(self) -> None:
        self.job_buffer.release()
        self.aux_buffer.release()
        if self._partial is not None:
            self._partial.release()
