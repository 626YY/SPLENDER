"""模型贴图烘焙（显卡）。

一套纹理集的模型贴图是三张带 mip 的贴图，按纹理集的 UV 排布：
- a（RGBA8）：环境遮蔽、曲率（0.5 为平，越亮越凸）、厚度、覆盖
- n（RGBA8）：世界空间法线（xyz × 0.5 + 0.5）、覆盖
- p（RGBA16F）：世界空间位置、覆盖
- t（RGBA8）：切线空间法线（指定了高模时是高模的细节，否则是平的）、覆盖
- i（R32F）：部件编号（-1 是空白），按「部件划分」把模型分成一块一块，给「部件」生成器用

指定了高模时，先从低模表面沿法线找到高模（投影），法线贴图、遮蔽、曲率、厚度都按高模的细节算。

烘焙分几步，每帧推进一点，界面不卡：
1. 后台线程：把模型的三角形建成 BVH（和路径追踪同一套），算逐角点曲率。
2. 按块（最大 2048²）在 UV 空间光栅化目标三角形，得到每个纹素的位置、平滑法线、几何法线、曲率。
3. 计算着色器：每个纹素朝法线半球发射线算环境遮蔽，朝反方向发射线量厚度。采样方向用每个纹素随机旋转的
   Hammersley 序列，同样的采样数噪点更少。按横条分批派发，每批十几毫秒。
4. 收尾：写进三张贴图；用跳跃泛洪找每个空纹素最近的有效纹素，把边界向外扩（mip 和双线性采样在接缝处不漏底色）；
   生成 mip；读回一份存盘和缩略图用。
"""
from __future__ import annotations

import io
import logging
import math
import threading
import time

import moderngl
import numpy as np

from ..engine import glx
from ..render.pathtracer.bvh import build_bvh
from ..render.pathtracer.shaders import BVH_GLSL, STACK
from .curvature import corner_curvature
from .parts import compact, triangle_parts

log = logging.getLogger("splender.bake")

TILE = 2048
MAX_SIZE = 4096

_RASTER_VERT = """#version 430
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_curv; in float in_id;
uniform vec2 u_origin;        // 这一块左下角（纹素）
uniform float u_res;          // 整张图的边长（纹素）
uniform float u_tile;         // 这一块的边长（纹素）
out vec3 v_pos; out vec3 v_nrm; out float v_curv; flat out float v_id;
void main() {
  v_pos = in_pos; v_nrm = in_nrm; v_curv = in_curv; v_id = in_id;
  vec2 q = (in_uv * u_res - u_origin) / u_tile;
  gl_Position = vec4(q * 2.0 - 1.0, 0.0, 1.0);
}
"""

_RASTER_FRAG = """#version 430
in vec3 v_pos; in vec3 v_nrm; in float v_curv; flat in float v_id;
layout(location = 0) out vec4 o_pos;
layout(location = 1) out vec4 o_nrm;
layout(location = 2) out vec4 o_geo;
layout(location = 3) out vec4 o_tan;     // 每个纹素沿 u、v 方向位置变多少（切线空间用）
layout(location = 4) out vec4 o_bit;
layout(location = 5) out vec4 o_id;      // 部件编号
void main() {
  o_id = vec4(v_id, 0.0, 0.0, 1.0);
  o_tan = vec4(dFdx(v_pos), 0.0);
  o_bit = vec4(dFdy(v_pos), 0.0);
  vec3 n = v_nrm;
  float ln = length(n);
  n = ln > 1e-20 ? n / ln : vec3(0.0, 1.0, 0.0);
  vec3 g = cross(dFdx(v_pos), dFdy(v_pos));
  float lg = length(g);
  g = lg > 1e-30 ? g / lg : n;
  if (dot(g, n) < 0.0) g = -g;
  o_pos = vec4(v_pos, 1.0);
  o_nrm = vec4(n, v_curv);
  o_geo = vec4(g, 1.0);
}
"""

# 高模投影：从低模表面沿法线找高模（每个纹素可以多采几点抗锯齿），得到切线空间法线和高模曲率；
# 打中的话把位置、法线换成高模上的，后面算遮蔽、厚度的射线就从高模表面出发。
_PROJECT = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) uniform image2D u_gpos;
layout(rgba16f, binding = 1) uniform image2D u_gnrm;
layout(rgba16f, binding = 2) uniform image2D u_ggeo;
layout(rgba32f, binding = 3) readonly uniform image2D u_gtan;
layout(rgba32f, binding = 4) readonly uniform image2D u_gbit;
layout(rgba16f, binding = 5) writeonly uniform image2D o_tnorm;
layout(r32f, binding = 6) uniform image2D u_gid;
struct Node { vec3 bmin; int first; vec3 bmax; int count; };
layout(std430, binding = 1) readonly buffer Nodes { Node nodes[]; };
layout(std430, binding = 2) readonly buffer TriPos { vec4 tri_pos[]; };
layout(std430, binding = 3) readonly buffer TriNrm { vec4 tri_nrm[]; };   // 每个三角形 3 项：角点法线、曲率
layout(std430, binding = 4) readonly buffer TriId { float tri_id[]; };    // 高模每个三角形的部件号
uniform int u_high_id;        // 1：部件按高模划分
#define BIG 1e30
uniform ivec2 u_size;
uniform float u_front;
uniform float u_back;
uniform int u_grid;           // 每个纹素 u_grid × u_grid 个采样点
uniform ivec2 u_offset;       // 这一批从哪一行开始（每个纹素只能处理一次：处理后位置、法线就换成了高模的）
""" + BVH_GLSL + """
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy) + u_offset;
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  vec4 P = imageLoad(u_gpos, p);
  if (P.w < 0.5) { imageStore(o_tnorm, p, vec4(0.0, 0.0, 1.0, 0.0)); return; }
  vec3 n = normalize(imageLoad(u_gnrm, p).xyz);
  vec3 du = imageLoad(u_gtan, p).xyz;
  vec3 dv = imageLoad(u_gbit, p).xyz;
  // 切线空间：T 沿 u 方向（去掉法线分量），B 和 dP/dv 同侧。渲染时按同样的约定解开
  vec3 T = du - n * dot(n, du);
  float lt = length(T);
  T = lt > 1e-20 ? T / lt : normalize(abs(n.x) < 0.9 ? cross(n, vec3(1.0, 0.0, 0.0)) : cross(n, vec3(0.0, 1.0, 0.0)));
  vec3 B = cross(n, T);
  if (dot(B, dv) < 0.0) B = -B;
  vec3 sum = vec3(0.0);
  float curv = 0.0;
  float hits = 0.0;
  float best = BIG;
  float hid = -1.0;
  vec3 hp = P.xyz; vec3 hn = n; vec3 hg = n;
  for (int sy = 0; sy < u_grid; sy++) {
    for (int sx = 0; sx < u_grid; sx++) {
      vec2 o = (vec2(float(sx), float(sy)) + 0.5) / float(u_grid) - 0.5;
      vec3 ps = P.xyz + du * o.x + dv * o.y;
      int tri; vec2 bc; float t;
      if (!trace(ps + n * u_front, -n, u_front + u_back, false, tri, bc, t)) continue;
      vec4 a = tri_nrm[tri * 3];
      vec4 b = tri_nrm[tri * 3 + 1];
      vec4 c = tri_nrm[tri * 3 + 2];
      float w0 = 1.0 - bc.x - bc.y;
      vec3 h = a.xyz * w0 + b.xyz * bc.x + c.xyz * bc.y;
      float lh = length(h);
      h = lh > 1e-20 ? h / lh : n;
      float hc = a.w * w0 + b.w * bc.x + c.w * bc.y;
      sum += vec3(dot(h, T), dot(h, B), dot(h, n));
      curv += hc;
      hits += 1.0;
      float centered = dot(o, o);
      if (centered < best) {
        best = centered;
        if (u_high_id == 1) hid = tri_id[tri];
        hp = ps + n * u_front - n * t;
        hn = h;
        vec3 g = cross(tri_pos[tri * 3 + 1].xyz, tri_pos[tri * 3 + 2].xyz);
        float lg = length(g);
        hg = lg > 1e-30 ? g / lg : h;
        if (dot(hg, h) < 0.0) hg = -hg;
      }
    }
  }
  if (hits > 0.0) {
    vec3 tn = sum / hits;
    float l = length(tn);
    tn = l > 1e-20 ? tn / l : vec3(0.0, 0.0, 1.0);
    imageStore(o_tnorm, p, vec4(tn, 1.0));
    imageStore(u_gpos, p, vec4(hp, 1.0));
    imageStore(u_gnrm, p, vec4(hn, curv / hits));
    imageStore(u_ggeo, p, vec4(hg, 1.0));
  } else {
    imageStore(o_tnorm, p, vec4(0.0, 0.0, 1.0, 0.0));
  }
  if (u_high_id == 1) imageStore(u_gid, p, vec4(hid, 0.0, 0.0, 0.0));
}
"""

_RAYS = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) readonly uniform image2D u_gpos;
layout(rgba16f, binding = 1) readonly uniform image2D u_gnrm;
layout(rgba16f, binding = 2) readonly uniform image2D u_ggeo;
layout(rgba32f, binding = 3) uniform image2D u_accum;
struct Node { vec3 bmin; int first; vec3 bmax; int count; };
layout(std430, binding = 1) readonly buffer Nodes { Node nodes[]; };
layout(std430, binding = 2) readonly buffer TriPos { vec4 tri_pos[]; };
#define TWO_PI 6.28318530717958648
#define BIG 1e30
uniform ivec2 u_offset;      // 这一批的起点（块内纹素）
uniform ivec2 u_size;        // 块的大小
uniform int u_first;         // 这一批从第几个采样开始
uniform int u_count;         // 这一批做几个采样
uniform int u_total;         // 每个纹素一共多少采样
uniform float u_ao_dist;     // 0 表示不算环境遮蔽
uniform float u_thick_dist;  // 0 表示不算厚度
uniform float u_eps;
uniform ivec2 u_tile_origin;
""" + BVH_GLSL + """
uint pcg(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
float radical_inverse(uint bits) {
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return float(bits) * 2.3283064365386963e-10;
}
vec3 cosine_dir(vec3 n, vec2 xi) {
  float phi = TWO_PI * xi.x;
  float r = sqrt(xi.y);
  vec3 t = normalize(abs(n.x) > 0.5 ? cross(n, vec3(0.0, 1.0, 0.0)) : cross(n, vec3(1.0, 0.0, 0.0)));
  vec3 b = cross(n, t);
  return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - xi.y)));
}
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy) + u_offset;
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  vec4 P = imageLoad(u_gpos, p);
  if (P.w < 0.5) return;
  vec3 n = normalize(imageLoad(u_gnrm, p).xyz);
  vec3 g = imageLoad(u_ggeo, p).xyz;
  g = dot(g, g) > 1e-12 ? normalize(g) : n;
  ivec2 gp = p + u_tile_origin;
  uint h = pcg(uint(gp.x) * 0x9E3779B9u ^ pcg(uint(gp.y) + 0x85EBCA6Bu));
  vec2 rot = vec2(float(h & 0xFFFFu), float(h >> 16u)) * (1.0 / 65536.0);
  uint h2 = pcg(h);
  vec2 rot2 = vec2(float(h2 & 0xFFFFu), float(h2 >> 16u)) * (1.0 / 65536.0);
  vec4 acc = imageLoad(u_accum, p);
  for (int k = 0; k < u_count; k++) {
    uint s = uint(u_first + k);
    vec2 base = vec2((float(s) + 0.5) / float(u_total), radical_inverse(s));
    int tri; vec2 bc; float th;
    if (u_ao_dist > 0.0) {
      vec3 d = cosine_dir(n, fract(base + rot));
      float side = dot(d, g);
      if (side <= 0.0) d = normalize(d - 2.0 * side * g);
      acc.x += trace(P.xyz + g * u_eps, d, u_ao_dist, true, tri, bc, th) ? 1.0 : 0.0;
    }
    if (u_thick_dist > 0.0) {
      vec3 d = cosine_dir(-n, fract(base + rot2));
      float side = dot(d, -g);
      if (side <= 0.0) d = normalize(d + 2.0 * side * g);
      acc.y += trace(P.xyz - g * u_eps, d, u_thick_dist, false, tri, bc, th) ? th : u_thick_dist;
    }
    acc.z += 1.0;
  }
  imageStore(u_accum, p, acc);
}
"""

_FINISH = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) readonly uniform image2D u_gpos;
layout(rgba16f, binding = 1) readonly uniform image2D u_gnrm;
layout(rgba32f, binding = 3) readonly uniform image2D u_accum;
layout(rgba8, binding = 4) writeonly uniform image2D o_a;
layout(rgba8, binding = 5) writeonly uniform image2D o_n;
layout(rgba16f, binding = 6) writeonly uniform image2D o_p;
layout(rgba16f, binding = 2) readonly uniform image2D u_tnorm;
layout(rgba8, binding = 7) writeonly uniform image2D o_t;
uniform int u_has_tnorm;
uniform ivec2 u_size;
uniform ivec2 u_tile_origin;
uniform float u_thick_dist;
uniform int u_has_ao;
uniform int u_has_thick;
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  ivec2 q = p + u_tile_origin;
  vec4 P = imageLoad(u_gpos, p);
  if (P.w < 0.5) {
    imageStore(o_a, q, vec4(1.0, 0.5, 1.0, 0.0));
    imageStore(o_n, q, vec4(0.5, 1.0, 0.5, 0.0));
    imageStore(o_p, q, vec4(0.0));
    imageStore(o_t, q, vec4(0.5, 0.5, 1.0, 0.0));
    return;
  }
  vec3 tn = u_has_tnorm == 1 ? imageLoad(u_tnorm, p).xyz : vec3(0.0, 0.0, 1.0);
  imageStore(o_t, q, vec4(normalize(tn) * 0.5 + 0.5, 1.0));
  vec4 N = imageLoad(u_gnrm, p);
  vec4 acc = imageLoad(u_accum, p);
  float samples = max(acc.z, 1.0);
  float ao = u_has_ao == 1 ? 1.0 - acc.x / samples : 1.0;
  float thick = u_has_thick == 1 ? clamp(acc.y / samples / max(u_thick_dist, 1e-20), 0.0, 1.0) : 1.0;
  float curv = clamp(N.w, -1.0, 1.0) * 0.5 + 0.5;
  vec3 n = normalize(N.xyz);
  imageStore(o_a, q, vec4(ao, curv, thick, 1.0));
  imageStore(o_n, q, vec4(n * 0.5 + 0.5, 1.0));
  imageStore(o_p, q, vec4(P.xyz, 1.0));
}
"""

_FINISH_ID = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) readonly uniform image2D u_gpos;
layout(r32f, binding = 1) readonly uniform image2D u_gid;
layout(r32f, binding = 2) writeonly uniform image2D o_i;
uniform ivec2 u_size;
uniform ivec2 u_tile_origin;
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  float id = imageLoad(u_gpos, p).w < 0.5 ? -1.0 : imageLoad(u_gid, p).r;
  imageStore(o_i, p + u_tile_origin, vec4(id, 0.0, 0.0, 0.0));
}
"""

# 跳跃泛洪：种子图里存最近的有效纹素坐标（无效为 -1）
_JFA_INIT = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba8, binding = 0) readonly uniform image2D u_a;
layout(rg32i, binding = 1) writeonly uniform iimage2D o_seed;
uniform ivec2 u_size;
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  imageStore(o_seed, p, imageLoad(u_a, p).w > 0.5 ? ivec4(p, 0, 0) : ivec4(-1, -1, 0, 0));
}
"""

_JFA_STEP = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rg32i, binding = 1) readonly uniform iimage2D i_seed;
layout(rg32i, binding = 2) writeonly uniform iimage2D o_seed;
uniform ivec2 u_size;
uniform int u_step;
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  ivec2 best = imageLoad(i_seed, p).xy;
  float best_d = best.x >= 0 ? dot(vec2(best - p), vec2(best - p)) : 1e30;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 q = p + ivec2(dx, dy) * u_step;
      if (q.x < 0 || q.y < 0 || q.x >= u_size.x || q.y >= u_size.y) continue;
      ivec2 s = imageLoad(i_seed, q).xy;
      if (s.x < 0) continue;
      float d = dot(vec2(s - p), vec2(s - p));
      if (d < best_d) { best_d = d; best = s; }
    }
  }
  imageStore(o_seed, p, ivec4(best, 0, 0));
}
"""

_JFA_FILL = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba8, binding = 0) uniform image2D u_a;
layout(rgba8, binding = 3) uniform image2D u_n;
layout(rgba16f, binding = 4) uniform image2D u_p;
layout(rgba8, binding = 5) uniform image2D u_t;
layout(r32f, binding = 6) uniform image2D u_i;
layout(rg32i, binding = 1) readonly uniform iimage2D i_seed;
uniform ivec2 u_size;
uniform float u_radius;
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  if (imageLoad(u_a, p).w > 0.5) return;
  ivec2 s = imageLoad(i_seed, p).xy;
  if (s.x < 0) return;
  vec2 d = vec2(s - p);
  if (dot(d, d) > u_radius * u_radius) return;
  // 只读有效纹素（它们在这一步里不会被写），结果和执行顺序无关
  vec4 a = imageLoad(u_a, s);
  imageStore(u_a, p, vec4(a.xyz, 0.75));
  imageStore(u_n, p, vec4(imageLoad(u_n, s).xyz, 0.75));
  imageStore(u_p, p, vec4(imageLoad(u_p, s).xyz, 0.75));
  imageStore(u_t, p, vec4(imageLoad(u_t, s).xyz, 0.75));
  imageStore(u_i, p, vec4(imageLoad(u_i, s).r, 0.0, 0.0, 0.0));
}
"""


def _set(program, name: str, value) -> None:
    if name in program:
        program[name] = value


# ====================================================================== 一套模型贴图
class MeshMapSet:
    """一套纹理集的模型贴图：显卡上三张带 mip 的贴图，和 CPU 上的副本（存盘、缩略图用）。"""

    def __init__(self, ctx, size: int, textures: dict, meta: dict, arrays: dict | None = None) -> None:
        self.ctx = ctx
        self.size = int(size)
        self.textures = textures          # "a"、"n"、"p"、"t"、"i" -> moderngl.Texture
        self.meta = dict(meta)            # bounds_min、bounds_max、settings、stats
        self.arrays = arrays              # "a"、"n"、"p" -> numpy（第 0 行是 v=0）
        self.saved = False                # 已经写进当前工程文件

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.meta.get("bounds_min", (0, 0, 0)), np.float32),
                np.asarray(self.meta.get("bounds_max", (1, 1, 1)), np.float32))

    def read_arrays(self) -> dict:
        if self.arrays is None:
            size = self.size
            self.arrays = {
                "a": np.frombuffer(self.textures["a"].read(), np.uint8).reshape(size, size, 4).copy(),
                "n": np.frombuffer(self.textures["n"].read(), np.uint8).reshape(size, size, 4).copy(),
                "p": np.frombuffer(self.textures["p"].read(), np.float16).reshape(size, size, 4).copy(),
                "t": np.frombuffer(self.textures["t"].read(), np.uint8).reshape(size, size, 4).copy(),
                "i": np.frombuffer(self.textures["i"].read(), np.float32).reshape(size, size).copy(),
            }
        return self.arrays

    @property
    def part_count(self) -> int:
        """ID 贴图里有几个部件（0 表示还没分）。"""
        return int(self.meta.get("parts", 0))

    def part_at_uv(self, u: float, v: float) -> int:
        """UV 处的部件号（-1 是空白）。"""
        ids = self.read_arrays()["i"]
        size = self.size
        x = min(size - 1, max(0, int(u * size)))
        y = min(size - 1, max(0, int(v * size)))
        return int(round(float(ids[y, x])))

    @property
    def has_normal(self) -> bool:
        """法线贴图是从高模烘出来的（否则是平的，不用画）。"""
        return bool(self.meta.get("high_poly"))

    def preview(self, kind: str, size: int = 128) -> np.ndarray:
        """缩略图 (size, size, 3) uint8，第 0 行是图片顶部。kind：ao、curvature、thickness、normal、position。"""
        arrays = self.read_arrays()
        step = max(1, self.size // size)
        if kind in ("ao", "curvature", "thickness"):
            channel = {"ao": 0, "curvature": 1, "thickness": 2}[kind]
            gray = arrays["a"][::step, ::step, channel]
            image = np.repeat(gray[:, :, None], 3, axis=2)
        elif kind == "normal":
            image = arrays["n"][::step, ::step, :3]
        elif kind == "tangent":
            image = arrays["t"][::step, ::step, :3]
        elif kind == "parts":
            from .parts import part_color

            image = part_color(np.rint(arrays["i"][::step, ::step]).astype(np.int64))
        else:
            lo, hi = self.bounds
            p = arrays["p"][::step, ::step, :3].astype(np.float32)
            image = np.clip((p - lo) / np.maximum(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
            image[arrays["p"][::step, ::step, 3] <= 0] = 0
        return np.ascontiguousarray(image[::-1])

    def to_bytes(self) -> bytes:
        import json

        import zstandard

        arrays = self.read_arrays()
        buffer = io.BytesIO()
        np.savez(buffer, a=arrays["a"], n=arrays["n"], p=arrays["p"], t=arrays["t"], i=arrays["i"],
                 meta=np.frombuffer(json.dumps(self.meta).encode("utf-8"), np.uint8))
        return zstandard.ZstdCompressor(level=3).compress(buffer.getvalue())

    @classmethod
    def from_bytes(cls, ctx, data: bytes) -> "MeshMapSet":
        import json

        import zstandard

        raw = zstandard.ZstdDecompressor().decompress(data)
        with np.load(io.BytesIO(raw)) as z:
            arrays = {"a": z["a"], "n": z["n"], "p": z["p"]}
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            if "t" in z:
                arrays["t"] = z["t"]
            if "i" in z:
                arrays["i"] = z["i"]
        size = int(arrays["a"].shape[0])
        if "t" not in arrays:                      # 早先存的没有法线贴图：补一张平的
            flat = np.zeros((size, size, 4), np.uint8)
            flat[:, :] = (128, 128, 255, 255)
            arrays["t"] = flat
        if "i" not in arrays:                      # 早先存的没有部件：全部空白，重新烘焙才有
            arrays["i"] = np.full((size, size), -1.0, np.float32)
        textures = {
            "a": ctx.texture((size, size), 4, np.ascontiguousarray(arrays["a"]).tobytes(), dtype="f1"),
            "n": ctx.texture((size, size), 4, np.ascontiguousarray(arrays["n"]).tobytes(), dtype="f1"),
            "p": ctx.texture((size, size), 4, np.ascontiguousarray(arrays["p"]).tobytes(), dtype="f2"),
            "t": ctx.texture((size, size), 4, np.ascontiguousarray(arrays["t"]).tobytes(), dtype="f1"),
        }
        for tex in textures.values():
            _finish_texture(tex)
        textures["i"] = ctx.texture((size, size), 1, np.ascontiguousarray(arrays["i"], np.float32).tobytes(),
                                    dtype="f4")
        _finish_id_texture(textures["i"])
        maps = cls(ctx, size, textures, meta, arrays)
        maps.saved = True
        return maps

    def release(self) -> None:
        for tex in self.textures.values():
            tex.release()
        self.textures = {}


def _finish_texture(tex) -> None:
    tex.repeat_x = False
    tex.repeat_y = False
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)


def _finish_id_texture(tex) -> None:
    """部件号不能插值，也不做多级。"""
    tex.repeat_x = False
    tex.repeat_y = False
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)


# ====================================================================== 烘焙任务
class BakeJob:
    """一次烘焙。engine 每帧调用 step(budget_ms)，done 之后 result 是新的 MeshMapSet（出错时 error 有原因）。"""

    PREPARE, RASTER, RAYS, FINISH, DONE = range(5)

    def __init__(self, engine, ts, settings) -> None:
        self.engine = engine
        self.ctx = engine.ctx
        self.ts = ts
        self.settings = {
            "size": int(min(MAX_SIZE, max(64, int(settings.resolution)))),
            "ao_samples": int(max(1, settings.ao_samples)),
            "ao_distance": float(settings.ao_distance),
            "thickness": bool(settings.bake_thickness),
            "thickness_distance": float(settings.thickness_distance),
            "curvature_smooth": int(settings.curvature_smooth),
            "curvature_contrast": float(settings.curvature_contrast),
            "self_only": bool(settings.self_only),
            "padding": int(settings.padding),
            "high_poly": str(getattr(settings, "high_poly", "") or "").strip(),
            "cage_front": float(getattr(settings, "cage_front", 0.02)),
            "cage_back": float(getattr(settings, "cage_back", 0.02)),
            "normal_samples": int(getattr(settings, "normal_samples", "4")),
            "id_source": str(getattr(settings, "id_source", "PART")),
        }
        self.stage = self.PREPARE
        self.started = time.perf_counter()
        self.error = ""
        self.result: MeshMapSet | None = None
        self.cancelled = False
        self.status = "准备模型"
        self._prep: dict | None = None
        self._prep_error: str = ""
        self._gpu: dict = {}
        self._targets = self._collect_targets()
        if not self._targets:
            raise ValueError("这套贴图没有对应的、带 UV 的模型")
        self._thread = threading.Thread(target=self._prepare, name="bake-prepare", daemon=True)
        self._thread.start()
        size = self.settings["size"]
        self.tile = min(TILE, size)
        self.tiles = [(x, y) for y in range(0, size, self.tile) for x in range(0, size, self.tile)]
        self.tile_index = 0
        self.sample = 0
        self.row = 0
        self.rows_per_band = 64
        self.samples_per_band = 4
        self.ray_seconds = 0.0

    # ---------------------------------------------------------------- 准备（后台线程）
    def _collect_targets(self) -> list:
        engine = self.engine
        project = engine.project
        targets = []
        for obj in project.objects if project is not None else []:
            if not obj.visible or not getattr(obj.data, "has_uvs", True):
                continue
            mesh = engine.meshes.get(obj.uid)
            if mesh is None:
                continue
            materials = [m for m, uid in enumerate(obj.material_sets)
                         if uid == self.ts.uid and m < len(mesh.material_count) and mesh.material_count[m] > 0]
            if materials:
                targets.append((obj, mesh, materials))
        return targets

    def _prepare(self) -> None:
        try:
            settings = self.settings
            engine = self.engine
            occluders = []
            curvature = {}
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            target_uids = {obj.uid for obj, _mesh, _mats in self._targets}
            part_ids = {}
            parts = 0
            low_mode = "UV_ISLAND" if settings["id_source"] == "UV_ISLAND" else "PART"
            for obj, _mesh, materials in self._targets:
                data = obj.data
                tri_parts, _count = triangle_parts(data.positions, data.uvs, low_mode)
                chosen = np.isin(np.asarray(data.material_ids), materials)
                ids, count = compact(tri_parts, chosen, start=parts)
                parts += count
                part_ids[obj.uid] = np.repeat(ids.astype(np.float32), 3)
                curv, _stats = corner_curvature(data.positions, settings["curvature_smooth"],
                                                settings["curvature_contrast"])
                curvature[obj.uid] = curv
                mats = np.asarray(data.material_ids)
                tri_mask = np.isin(mats, materials)
                corners = np.asarray(data.positions, np.float32).reshape(-1, 3, 3)[tri_mask]
                if len(corners):
                    lo = np.minimum(lo, corners.reshape(-1, 3).min(axis=0))
                    hi = np.maximum(hi, corners.reshape(-1, 3).max(axis=0))
            high = None
            if settings["high_poly"]:
                from ..doc.meshio import load_mesh

                self.status = "读取高模"
                high = load_mesh(settings["high_poly"])
                high_curv, _stats = corner_curvature(high.positions, settings["curvature_smooth"],
                                                     settings["curvature_contrast"])
                occluders.append(np.asarray(high.positions, np.float32).reshape(-1, 3, 3))
            else:
                for obj in engine.project.objects:
                    if not obj.visible:
                        continue
                    corners = np.asarray(obj.data.positions, np.float32).reshape(-1, 3, 3)
                    if settings["self_only"]:
                        if obj.uid not in target_uids:
                            continue
                        materials = next(m for o, _x, m in self._targets if o.uid == obj.uid)
                        corners = corners[np.isin(np.asarray(obj.data.material_ids), materials)]
                    occluders.append(corners)
            tris = np.concatenate(occluders) if occluders else np.zeros((0, 3, 3), np.float32)
            bvh = build_bvh(tris[:, 0], tris[:, 1], tris[:, 2])
            order = bvh["order"]
            tri_pos = np.zeros((max(1, len(order)), 3, 4), np.float32)
            if len(order):
                v0 = tris[order, 0]
                tri_pos[:, 0, :3] = v0
                tri_pos[:, 1, :3] = tris[order, 1] - v0
                tri_pos[:, 2, :3] = tris[order, 2] - v0
            if not np.isfinite(lo).all():
                lo, hi = np.zeros(3), np.ones(3)
            tri_id = None
            if high is not None and settings["id_source"] == "HIGH_POLY":
                high_parts, parts = triangle_parts(high.positions, None, "PART")
                tri_id = high_parts[order].astype(np.float32) if len(order) else np.zeros(1, np.float32)
            tri_nrm = None
            if high is not None and len(order):
                normals = np.asarray(high.normals, np.float32).reshape(-1, 3, 3)[order]
                curv = np.asarray(high_curv, np.float32).reshape(-1, 3)[order]
                tri_nrm = np.zeros((len(order), 3, 4), np.float32)
                tri_nrm[:, :, :3] = normals
                tri_nrm[:, :, 3] = curv
            self._prep = {"curvature": curvature, "nodes": bvh["nodes"], "tri_pos": tri_pos, "depth": bvh["depth"],
                          "bounds": (lo.astype(np.float32), hi.astype(np.float32)), "triangles": int(len(order)),
                          "bvh_ms": round(bvh["ms"], 1), "tri_nrm": tri_nrm, "part_ids": part_ids,
                          "tri_id": tri_id, "parts": int(parts),
                          "high_triangles": int(high.triangle_count) if high is not None else 0}
        except Exception as exc:  # noqa: BLE001
            log.exception("烘焙准备出错")
            self._prep_error = str(exc) or type(exc).__name__

    # ---------------------------------------------------------------- 进度
    @property
    def done(self) -> bool:
        return self.stage == self.DONE

    @property
    def progress(self) -> float:
        if self.stage == self.PREPARE:
            return 0.0
        if self.stage >= self.FINISH:
            return 1.0
        total = len(self.tiles) * max(1, self.settings["ao_samples"])
        return min(0.99, (self.tile_index * self.settings["ao_samples"] + self.sample) / total)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def cancel(self) -> None:
        self.cancelled = True

    # ---------------------------------------------------------------- 推进
    def step(self, budget_ms: float = 12.0) -> bool:
        """推进一点，返回是否已经结束（完成、出错或取消）。"""
        if self.done:
            return True
        try:
            if self.cancelled:
                self._finish_with("已取消")
                return True
            if self.stage == self.PREPARE:
                if self._thread.is_alive():
                    return False
                if self._prep_error or self._prep is None:
                    self._finish_with("准备模型时出错：%s" % (self._prep_error or "未知原因"))
                    return True
                if self._prep["depth"] >= STACK:
                    self._finish_with("模型的 BVH 太深（%d 层），超过上限 %d" % (self._prep["depth"], STACK))
                    return True
                self._setup_gpu()
                self.stage = self.RASTER
            started = time.perf_counter()
            while not self.done and (time.perf_counter() - started) * 1000.0 < budget_ms:
                if self.stage == self.RASTER:
                    self._raster_tile()
                    if "project" in self._gpu:
                        self._project_tile()
                    self.stage = self.RAYS
                    self.status = "计算遮蔽和厚度"
                elif self.stage == self.RAYS:
                    remaining = budget_ms - (time.perf_counter() - started) * 1000.0
                    if self._rays(max(2.0, remaining)):
                        self._finish_tile()
                        self.tile_index += 1
                        self.sample = 0
                        self.row = 0
                        self.stage = self.RASTER if self.tile_index < len(self.tiles) else self.FINISH
                elif self.stage == self.FINISH:
                    self.status = "扩边、生成多级贴图"
                    self._finalize()
                    self.stage = self.DONE
        except Exception as exc:  # noqa: BLE001
            log.exception("烘焙出错")
            self._finish_with("出错：%s" % (str(exc) or type(exc).__name__))
        return self.done

    def _finish_with(self, error: str) -> None:
        self.error = error
        self.stage = self.DONE
        self._release_gpu(keep_outputs=False)

    # ---------------------------------------------------------------- 显卡资源
    def _setup_gpu(self) -> None:
        ctx = self.ctx
        prep = self._prep
        size = self.settings["size"]
        tile = self.tile
        gpu = self._gpu
        gpu["gpos"] = ctx.texture((tile, tile), 4, dtype="f4")
        gpu["gnrm"] = ctx.texture((tile, tile), 4, dtype="f2")
        gpu["ggeo"] = ctx.texture((tile, tile), 4, dtype="f2")
        gpu["accum"] = ctx.texture((tile, tile), 4, dtype="f4")
        gpu["gtan"] = ctx.texture((tile, tile), 4, dtype="f4")
        gpu["gbit"] = ctx.texture((tile, tile), 4, dtype="f4")
        gpu["gtnorm"] = ctx.texture((tile, tile), 4, dtype="f2")
        gpu["gid"] = ctx.texture((tile, tile), 1, dtype="f4")
        for name in ("gpos", "gnrm", "ggeo", "accum", "gtan", "gbit", "gtnorm", "gid"):
            gpu[name].filter = (moderngl.NEAREST, moderngl.NEAREST)
        gpu["fbo"] = ctx.framebuffer([gpu["gpos"], gpu["gnrm"], gpu["ggeo"], gpu["gtan"], gpu["gbit"], gpu["gid"]])
        gpu["out_a"] = ctx.texture((size, size), 4, dtype="f1")
        gpu["out_n"] = ctx.texture((size, size), 4, dtype="f1")
        gpu["out_p"] = ctx.texture((size, size), 4, dtype="f2")
        gpu["out_t"] = ctx.texture((size, size), 4, dtype="f1")
        gpu["out_i"] = ctx.texture((size, size), 1, dtype="f4")
        gpu["finish_id"] = ctx.compute_shader(_FINISH_ID)
        if prep.get("tri_nrm") is not None:
            gpu["project"] = ctx.compute_shader(_PROJECT)
            gpu["tri_nrm"] = ctx.buffer(np.ascontiguousarray(prep["tri_nrm"], np.float32).tobytes())
            tri_id = prep.get("tri_id")
            gpu["tri_id"] = ctx.buffer(np.ascontiguousarray(tri_id if tri_id is not None else np.zeros(1),
                                                            np.float32).tobytes())
        gpu["raster"] = ctx.program(vertex_shader=_RASTER_VERT, fragment_shader=_RASTER_FRAG)
        gpu["rays"] = ctx.compute_shader(_RAYS)
        gpu["finish"] = ctx.compute_shader(_FINISH)
        gpu["nodes"] = ctx.buffer(np.ascontiguousarray(prep["nodes"], np.float32).tobytes())
        gpu["tri_pos"] = ctx.buffer(np.ascontiguousarray(prep["tri_pos"], np.float32).tobytes())
        gpu["vaos"] = []
        gpu["curv"] = []
        program = gpu["raster"]
        for obj, mesh, materials in self._targets:
            curv = np.ascontiguousarray(self._prep["curvature"][obj.uid], np.float32)
            curv_vbo = ctx.buffer(curv.tobytes())
            gpu["curv"].append(curv_vbo)
            id_vbo = ctx.buffer(np.ascontiguousarray(self._prep["part_ids"][obj.uid], np.float32).tobytes())
            gpu["curv"].append(id_vbo)
            for material in materials:
                ibo = mesh.material_ibo[material]
                if ibo is None:
                    continue
                vao = ctx.vertex_array(program, [(mesh.vbo, "3f 3f 2f", "in_pos", "in_nrm", "in_uv"),
                                                 (curv_vbo, "1f", "in_curv"), (id_vbo, "1f", "in_id")],
                                       index_buffer=ibo, index_element_size=4)
                gpu["vaos"].append(vao)
        lo, hi = prep["bounds"]
        diag = float(np.linalg.norm(hi - lo)) or 1.0
        self.diag = diag
        self.eps = diag * 5e-5
        self.status = "光栅化"

    def _release_gpu(self, keep_outputs: bool) -> None:
        gpu = self._gpu
        for name in ("fbo", "gpos", "gnrm", "ggeo", "accum", "gtan", "gbit", "gtnorm", "gid", "raster", "rays",
                     "finish", "finish_id", "nodes", "tri_pos", "tri_nrm", "tri_id", "project", "seed_a", "seed_b",
                     "jfa_init", "jfa_step", "jfa_fill"):
            item = gpu.pop(name, None)
            if item is not None:
                item.release()
        for item in gpu.pop("vaos", []) + gpu.pop("curv", []):
            item.release()
        if not keep_outputs:
            for name in ("out_a", "out_n", "out_p", "out_t", "out_i"):
                item = gpu.pop(name, None)
                if item is not None:
                    item.release()

    # ---------------------------------------------------------------- 各步
    def _raster_tile(self) -> None:
        ctx = self.ctx
        gpu = self._gpu
        x0, y0 = self.tiles[self.tile_index]
        tile = self.tile
        gpu["fbo"].use()
        ctx.viewport = (0, 0, tile, tile)
        gpu["fbo"].clear(0.0, 0.0, 0.0, 0.0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        program = gpu["raster"]
        _set(program, "u_origin", (float(x0), float(y0)))
        _set(program, "u_res", float(self.settings["size"]))
        _set(program, "u_tile", float(tile))
        for vao in gpu["vaos"]:
            vao.render(moderngl.TRIANGLES)
        gpu["accum"].write(np.zeros((tile, tile, 4), np.float32).tobytes())
        ctx.memory_barrier()

    def _project_tile(self) -> None:
        gpu = self._gpu
        program = gpu["project"]
        tile = self.tile
        settings = self.settings
        gpu["gpos"].bind_to_image(0, read=True, write=True)
        gpu["gnrm"].bind_to_image(1, read=True, write=True)
        gpu["ggeo"].bind_to_image(2, read=True, write=True)
        gpu["gtan"].bind_to_image(3, read=True, write=False)
        gpu["gbit"].bind_to_image(4, read=True, write=False)
        gpu["gtnorm"].bind_to_image(5, read=False, write=True)
        gpu["nodes"].bind_to_storage_buffer(1)
        gpu["tri_pos"].bind_to_storage_buffer(2)
        gpu["tri_nrm"].bind_to_storage_buffer(3)
        gpu["tri_id"].bind_to_storage_buffer(4)
        gpu["gid"].bind_to_image(6, read=True, write=True)
        _set(program, "u_high_id", int(self._prep.get("tri_id") is not None))
        _set(program, "u_size", (tile, tile))
        _set(program, "u_front", float(settings["cage_front"] * self.diag))
        _set(program, "u_back", float(settings["cage_back"] * self.diag))
        grid = {1: 1, 4: 2, 9: 3, 16: 4}.get(int(settings["normal_samples"]), 2)
        _set(program, "u_grid", int(grid))
        rows = max(8, min(tile, 2048 * 64 // max(1, tile * grid * grid)))
        for row in range(0, tile, rows):
            sub = min(rows, tile - row)
            program["u_size"] = (tile, row + sub)
            program["u_offset"] = (0, row)
            program.run((tile + 7) // 8, (sub + 7) // 8, 1)
            fence = glx.Fence()
            while not fence.ready(wait_ns=2_000_000):
                pass
            fence.release()
        self.ctx.memory_barrier()

    def _rays(self, budget_ms: float) -> bool:
        """推进遮蔽和厚度的射线，返回这一块是否做完。"""
        settings = self.settings
        total = settings["ao_samples"]
        ao_dist = settings["ao_distance"] * self.diag
        thick_dist = settings["thickness_distance"] * self.diag if settings["thickness"] else 0.0
        if ao_dist <= 0.0 and thick_dist <= 0.0:
            return True
        gpu = self._gpu
        program = gpu["rays"]
        gpu["gpos"].bind_to_image(0, read=True, write=False)
        gpu["gnrm"].bind_to_image(1, read=True, write=False)
        gpu["ggeo"].bind_to_image(2, read=True, write=False)
        gpu["accum"].bind_to_image(3, read=True, write=True)
        gpu["nodes"].bind_to_storage_buffer(1)
        gpu["tri_pos"].bind_to_storage_buffer(2)
        tile = self.tile
        x0, y0 = self.tiles[self.tile_index]
        _set(program, "u_size", (tile, tile))
        _set(program, "u_total", int(total))
        _set(program, "u_ao_dist", float(ao_dist))
        _set(program, "u_thick_dist", float(thick_dist))
        _set(program, "u_eps", float(self.eps))
        _set(program, "u_tile_origin", (int(x0), int(y0)))
        spent = 0.0
        while self.sample < total:
            rows = max(8, min(tile - self.row, self.rows_per_band))
            count = min(self.samples_per_band, total - self.sample)
            _set(program, "u_offset", (0, int(self.row)))
            _set(program, "u_first", int(self.sample))
            _set(program, "u_count", int(count))
            t0 = time.perf_counter()
            program.run((tile + 7) // 8, (rows + 7) // 8, 1)
            fence = glx.Fence()
            while not fence.ready(wait_ns=2_000_000):
                pass
            fence.release()
            elapsed = (time.perf_counter() - t0) * 1000.0
            spent += elapsed
            self.ray_seconds += elapsed / 1000.0
            per_row = elapsed / max(1, rows)
            target = min(16.0, max(2.0, budget_ms * 0.6))
            self.rows_per_band = int(max(8, min(tile, target / max(per_row, 1e-4))))
            self.row += rows
            if self.row >= tile:
                self.row = 0
                self.sample += count
            if spent + per_row * self.rows_per_band > budget_ms:
                break
        self.ctx.memory_barrier()
        return self.sample >= total

    def _finish_tile(self) -> None:
        gpu = self._gpu
        program = gpu["finish"]
        tile = self.tile
        x0, y0 = self.tiles[self.tile_index]
        settings = self.settings
        gpu["gpos"].bind_to_image(0, read=True, write=False)
        gpu["gnrm"].bind_to_image(1, read=True, write=False)
        gpu["accum"].bind_to_image(3, read=True, write=False)
        gpu["out_a"].bind_to_image(4, read=False, write=True)
        gpu["out_n"].bind_to_image(5, read=False, write=True)
        gpu["out_p"].bind_to_image(6, read=False, write=True)
        gpu["gtnorm"].bind_to_image(2, read=True, write=False)
        gpu["out_t"].bind_to_image(7, read=False, write=True)
        _set(program, "u_has_tnorm", int("project" in gpu))
        _set(program, "u_size", (tile, tile))
        _set(program, "u_tile_origin", (int(x0), int(y0)))
        _set(program, "u_thick_dist", float(settings["thickness_distance"] * self.diag))
        _set(program, "u_has_ao", int(settings["ao_distance"] > 0.0))
        _set(program, "u_has_thick", int(settings["thickness"] and settings["thickness_distance"] > 0.0))
        program.run((tile + 7) // 8, (tile + 7) // 8, 1)
        self.ctx.memory_barrier()
        program = gpu["finish_id"]
        gpu["gpos"].bind_to_image(0, read=True, write=False)
        gpu["gid"].bind_to_image(1, read=True, write=False)
        gpu["out_i"].bind_to_image(2, read=False, write=True)
        _set(program, "u_size", (tile, tile))
        _set(program, "u_tile_origin", (int(x0), int(y0)))
        program.run((tile + 7) // 8, (tile + 7) // 8, 1)
        self.ctx.memory_barrier()

    def _finalize(self) -> None:
        ctx = self.ctx
        gpu = self._gpu
        size = self.settings["size"]
        radius = float(self.settings["padding"])
        if radius > 0:
            gpu["seed_a"] = ctx.texture((size, size), 2, dtype="i4")
            gpu["seed_b"] = ctx.texture((size, size), 2, dtype="i4")
            gpu["jfa_init"] = ctx.compute_shader(_JFA_INIT)
            gpu["jfa_step"] = ctx.compute_shader(_JFA_STEP)
            gpu["jfa_fill"] = ctx.compute_shader(_JFA_FILL)
            groups = ((size + 7) // 8, (size + 7) // 8, 1)
            gpu["out_a"].bind_to_image(0, read=True, write=False)
            gpu["seed_a"].bind_to_image(1, read=False, write=True)
            _set(gpu["jfa_init"], "u_size", (size, size))
            gpu["jfa_init"].run(*groups)
            ctx.memory_barrier()
            step = 1
            while step * 2 <= radius:
                step *= 2
            source, target = gpu["seed_a"], gpu["seed_b"]
            program = gpu["jfa_step"]
            _set(program, "u_size", (size, size))
            while step >= 1:
                source.bind_to_image(1, read=True, write=False)
                target.bind_to_image(2, read=False, write=True)
                _set(program, "u_step", int(step))
                program.run(*groups)
                ctx.memory_barrier()
                source, target = target, source
                step //= 2
            program = gpu["jfa_fill"]
            gpu["out_a"].bind_to_image(0, read=True, write=True)
            gpu["out_n"].bind_to_image(3, read=True, write=True)
            gpu["out_p"].bind_to_image(4, read=True, write=True)
            gpu["out_t"].bind_to_image(5, read=True, write=True)
            gpu["out_i"].bind_to_image(6, read=True, write=True)
            source.bind_to_image(1, read=True, write=False)
            _set(program, "u_size", (size, size))
            _set(program, "u_radius", radius)
            program.run(*groups)
            ctx.memory_barrier()
        textures = {"a": gpu.pop("out_a"), "n": gpu.pop("out_n"), "p": gpu.pop("out_p"), "t": gpu.pop("out_t")}
        for tex in textures.values():
            _finish_texture(tex)
        textures["i"] = gpu.pop("out_i")
        _finish_id_texture(textures["i"])
        lo, hi = self._prep["bounds"]
        meta = {"bounds_min": [float(v) for v in lo], "bounds_max": [float(v) for v in hi],
                "settings": dict(self.settings), "high_poly": self.settings["high_poly"],
                "parts": int(self._prep.get("parts", 0)),
                "id_source": "HIGH_POLY" if self._prep.get("tri_id") is not None else (
                    "UV_ISLAND" if self.settings["id_source"] == "UV_ISLAND" else "PART"),
                "stats": {"triangles": self._prep["triangles"], "bvh_ms": self._prep["bvh_ms"],
                          "ray_seconds": round(self.ray_seconds, 2), "seconds": round(self.elapsed, 2)}}
        self._release_gpu(keep_outputs=False)
        self.result = MeshMapSet(ctx, size, textures, meta)
        self.status = "完成"
        log.info("模型贴图烘焙完成：%s，%d²，%d 个遮挡三角形，射线 %.2f 秒，共 %.2f 秒", self.ts.name, size,
                 self._prep["triangles"], self.ray_seconds, self.elapsed)
