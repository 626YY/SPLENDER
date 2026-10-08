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

import copy
import io
import itertools
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
MAX_SIZE = 16384
#: 每格占多少字节（含多级）：遮蔽曲率厚度 RGBA8、切线法线 RGBA8 按分辨率；朝向 RGBA8、位置 RGBA16F 按「最大」，部件 R32F 不做多级
_FULL_BYTES = (4 + 4) * 4.0 / 3.0
_SMALL_BYTES = (4 + 8) * 4.0 / 3.0 + 4.0


def estimate_vram_mb(size: int, smooth: int) -> float:
    """一套模型贴图大约占多少显存（MB）。"""
    small = min(int(size), int(smooth))
    return (float(size) ** 2 * _FULL_BYTES + float(small) ** 2 * _SMALL_BYTES) / 1048576.0

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
uniform int u_shrink;            // 位置、朝向贴图比整张小几倍（1 是一样大）
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  ivec2 q = p + u_tile_origin;
  bool small = (q.x % u_shrink) == u_shrink / 2 && (q.y % u_shrink) == u_shrink / 2;
  ivec2 qs = q / u_shrink;
  vec4 P = imageLoad(u_gpos, p);
  if (P.w < 0.5) {
    imageStore(o_a, q, vec4(1.0, 0.5, 1.0, 0.0));
    if (small) {
      imageStore(o_n, qs, vec4(0.5, 1.0, 0.5, 0.0));
      imageStore(o_p, qs, vec4(0.0));
    }
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
  if (small) {
    imageStore(o_n, qs, vec4(n * 0.5 + 0.5, 1.0));
    imageStore(o_p, qs, vec4(P.xyz, 1.0));
  }
}
"""

_FINISH_ID = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) readonly uniform image2D u_gpos;
layout(r32f, binding = 1) readonly uniform image2D u_gid;
layout(r32f, binding = 2) writeonly uniform image2D o_i;
uniform ivec2 u_size;
uniform ivec2 u_tile_origin;
uniform int u_shrink;
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  ivec2 q = p + u_tile_origin;
  if ((q.x % u_shrink) != u_shrink / 2 || (q.y % u_shrink) != u_shrink / 2) return;
  float id = imageLoad(u_gpos, p).w < 0.5 ? -1.0 : imageLoad(u_gid, p).r;
  imageStore(o_i, q / u_shrink, vec4(id, 0.0, 0.0, 0.0));
}
"""

# 跳跃泛洪（按块做）：一块加上外圈（外圈宽 = 扩边半径）的种子图里存最近的有效纹素的整图坐标（无效为 -1）。
# 只往外扩「扩边半径」那么远，所以最近的有效纹素一定在外圈以内，按块做和整张做结果一样，种子图却只要一块那么大。
_JFA_INIT = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(%s, binding = 0) readonly uniform image2D u_cov;
layout(rg32i, binding = 1) writeonly uniform iimage2D o_seed;
uniform ivec2 u_origin;      // 这一块（含外圈）的左下角，整图坐标
uniform ivec2 u_region;      // 这一块（含外圈）的大小
void main() {
  ivec2 l = ivec2(gl_GlobalInvocationID.xy);
  if (l.x >= u_region.x || l.y >= u_region.y) return;
  ivec2 p = u_origin + l;
  // 只有真正覆盖到的格（1.0）当种子；前面的块扩出来的格是 0.75，不能再当种子往外扩
  imageStore(o_seed, l, imageLoad(u_cov, p).w > 0.9 ? ivec4(p, 0, 0) : ivec4(-1, -1, 0, 0));
}
"""

_JFA_STEP = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rg32i, binding = 1) readonly uniform iimage2D i_seed;
layout(rg32i, binding = 2) writeonly uniform iimage2D o_seed;
uniform ivec2 u_origin;
uniform ivec2 u_region;
uniform int u_step;
void main() {
  ivec2 l = ivec2(gl_GlobalInvocationID.xy);
  if (l.x >= u_region.x || l.y >= u_region.y) return;
  ivec2 p = u_origin + l;
  ivec2 best = imageLoad(i_seed, l).xy;
  float best_d = best.x >= 0 ? dot(vec2(best - p), vec2(best - p)) : 1e30;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      ivec2 m = l + ivec2(dx, dy) * u_step;
      if (m.x < 0 || m.y < 0 || m.x >= u_region.x || m.y >= u_region.y) continue;
      ivec2 s = imageLoad(i_seed, m).xy;
      if (s.x < 0) continue;
      float d = dot(vec2(s - p), vec2(s - p));
      if (d < best_d) { best_d = d; best = s; }
    }
  }
  imageStore(o_seed, l, ivec4(best, 0, 0));
}
"""

# 填：只写这一块里面（不含外圈）没覆盖到的格，只读有效纹素（它们在这一步里不会被写），结果和执行顺序无关
_JFA_FILL_FULL = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba8, binding = 0) uniform image2D u_a;
layout(rgba8, binding = 5) uniform image2D u_t;
layout(rg32i, binding = 1) readonly uniform iimage2D i_seed;
uniform ivec2 u_origin;
uniform ivec2 u_tile_origin;
uniform ivec2 u_tile_size;
uniform float u_radius;
void main() {
  ivec2 k = ivec2(gl_GlobalInvocationID.xy);
  if (k.x >= u_tile_size.x || k.y >= u_tile_size.y) return;
  ivec2 p = u_tile_origin + k;
  if (imageLoad(u_a, p).w > 0.5) return;
  ivec2 s = imageLoad(i_seed, p - u_origin).xy;
  if (s.x < 0) return;
  vec2 d = vec2(s - p);
  if (dot(d, d) > u_radius * u_radius) return;
  imageStore(u_a, p, vec4(imageLoad(u_a, s).xyz, 0.75));
  imageStore(u_t, p, vec4(imageLoad(u_t, s).xyz, 0.75));
}
"""

_JFA_FILL_SMALL = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba8, binding = 3) uniform image2D u_n;
layout(rgba16f, binding = 4) uniform image2D u_p;
layout(r32f, binding = 6) uniform image2D u_i;
layout(rg32i, binding = 1) readonly uniform iimage2D i_seed;
uniform ivec2 u_origin;
uniform ivec2 u_tile_origin;
uniform ivec2 u_tile_size;
uniform float u_radius;
void main() {
  ivec2 k = ivec2(gl_GlobalInvocationID.xy);
  if (k.x >= u_tile_size.x || k.y >= u_tile_size.y) return;
  ivec2 p = u_tile_origin + k;
  if (imageLoad(u_p, p).w > 0.5) return;
  ivec2 s = imageLoad(i_seed, p - u_origin).xy;
  if (s.x < 0) return;
  vec2 d = vec2(s - p);
  if (dot(d, d) > u_radius * u_radius) return;
  imageStore(u_n, p, vec4(imageLoad(u_n, s).xyz, 0.75));
  imageStore(u_p, p, vec4(imageLoad(u_p, s).xyz, 0.75));
  imageStore(u_i, p, vec4(imageLoad(u_i, s).r, 0.0, 0.0, 0.0));
}
"""


def _set(program, name: str, value) -> None:
    if name in program:
        program[name] = value


# ====================================================================== 一套模型贴图
_MAP_SERIAL = itertools.count(1)


class MeshMapSet:
    """一套纹理集的模型贴图：显卡上三张带 mip 的贴图，和 CPU 上的副本（存盘、缩略图用）。"""

    def __init__(self, ctx, size: int, textures: dict, meta: dict, arrays: dict | None = None) -> None:
        self.ctx = ctx
        self.size = int(size)
        self.textures = textures          # "a"、"n"、"p"、"t"、"i" -> moderngl.Texture
        self.meta = dict(meta)            # bounds_min、bounds_max、settings、stats
        self.arrays = arrays              # "a"、"n"、"p" -> numpy（第 0 行是 v=0）
        self.saved = False                # 已经写进当前工程文件
        self.serial = next(_MAP_SERIAL)   # 每套新烘的贴图一个号（自动保存据此判断换没换）

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(self.meta.get("bounds_min", (0, 0, 0)), np.float32),
                np.asarray(self.meta.get("bounds_max", (1, 1, 1)), np.float32))

    #: 每张贴图在 CPU 上的格式：(numpy 类型, 分量数, moderngl 类型)
    FORMATS = {"a": (np.uint8, 4, "f1"), "n": (np.uint8, 4, "f1"), "p": (np.float16, 4, "f2"),
               "t": (np.uint8, 4, "f1"), "i": (np.float32, 1, "f4")}

    def read_array(self, name: str) -> np.ndarray:
        """一张贴图在 CPU 上的副本（第一次用时读回，之后留着）。"""
        if self.arrays is None:
            self.arrays = {}
        if name not in self.arrays:
            dtype, comps, _kind = self.FORMATS[name]
            w, h = self.textures[name].size
            shape = (h, w, comps) if comps > 1 else (h, w)
            self.arrays[name] = np.frombuffer(self.textures[name].read(), dtype).reshape(shape).copy()
        return self.arrays[name]

    def read_arrays(self) -> dict:
        """CPU 上的副本（存盘用）。位置、朝向、部件可能比遮蔽和法线贴图小。16K 时有两三 GB，读回要一两秒。"""
        for name in self.FORMATS:
            self.read_array(name)
        return self.arrays

    @property
    def has_arrays(self) -> bool:
        return self.arrays is not None and all(name in self.arrays for name in self.FORMATS)

    def readback_steps(self, strip_bytes: int = 16 << 20):
        """生成器：把还没读回的贴图按横条分几步读回（每步约 strip_bytes），读完 has_arrays 为真。
        自动保存用：16K 的模型贴图一次读回会卡一两秒。"""
        for name, (dtype, comps, kind) in self.FORMATS.items():
            if self.arrays is not None and name in self.arrays:
                continue
            texture = self.textures[name]
            w, h = texture.size
            out = np.empty((h, w, comps) if comps > 1 else (h, w), dtype)
            fbo = self.ctx.framebuffer([texture])
            try:
                rows = max(1, int(strip_bytes) // max(1, w * comps * np.dtype(dtype).itemsize))
                for y in range(0, h, rows):
                    n = min(rows, h - y)
                    data = fbo.read(viewport=(0, y, w, n), components=comps, dtype=kind, alignment=1)
                    out[y:y + n] = np.frombuffer(data, dtype).reshape((n, w, comps) if comps > 1 else (n, w))
                    yield
            finally:
                fbo.release()
            if self.arrays is None:
                self.arrays = {}
            self.arrays[name] = out

    def _preview_source(self, name: str, want: int) -> np.ndarray:
        """缩略图用的一张图：已经读回了就隔行取，否则直接读贴图合适的那一级（不用把整张读回）。"""
        if self.arrays is not None and name in self.arrays:
            array = self.arrays[name]
            step = max(1, int(array.shape[0]) // max(1, want))
            return array[::step, ::step]
        texture = self.textures[name]
        dtype, comps, _kind = self.FORMATS[name]
        w, h = texture.size
        level = 0
        mipmapped = texture.filter[0] in (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR_MIPMAP_NEAREST)
        while mipmapped and (w >> (level + 1)) >= want and (h >> (level + 1)) >= 1:
            level += 1
        lw, lh = max(1, w >> level), max(1, h >> level)
        array = np.frombuffer(texture.read(level=level), dtype).reshape((lh, lw, comps) if comps > 1 else (lh, lw))
        step = max(1, lw // max(1, want))
        return array[::step, ::step]

    @property
    def small_size(self) -> int:
        """位置、朝向、部件贴图的大小。"""
        texture = self.textures.get("p")
        return int(texture.size[0]) if texture is not None else self.size

    @property
    def vram_mb(self) -> float:
        return sum(_texture_mb(texture) for texture in self.textures.values())

    @property
    def part_count(self) -> int:
        """ID 贴图里有几个部件（0 表示还没分）。"""
        return int(self.meta.get("parts", 0))

    def part_at_uv(self, u: float, v: float) -> int:
        """UV 处的部件号（-1 是空白）。"""
        ids = self.read_array("i")
        size = int(ids.shape[0])
        x = min(size - 1, max(0, int(u * size)))
        y = min(size - 1, max(0, int(v * size)))
        return int(round(float(ids[y, x])))

    @property
    def has_normal(self) -> bool:
        """法线贴图是从高模烘出来的（否则是平的，不用画）。"""
        return bool(self.meta.get("high_poly"))

    def preview(self, kind: str, size: int = 128) -> np.ndarray:
        """缩略图 (size, size, 3) uint8，第 0 行是图片顶部。kind：ao、curvature、thickness、normal、position。"""
        if kind in ("ao", "curvature", "thickness"):
            channel = {"ao": 0, "curvature": 1, "thickness": 2}[kind]
            gray = self._preview_source("a", size)[:, :, channel]
            image = np.repeat(gray[:, :, None], 3, axis=2)
        elif kind == "normal":
            image = self._preview_source("n", size)[:, :, :3]
        elif kind == "tangent":
            image = self._preview_source("t", size)[:, :, :3]
        elif kind == "parts":
            from .parts import part_color

            image = part_color(np.rint(self._preview_source("i", size)).astype(np.int64))
        else:
            lo, hi = self.bounds
            source = self._preview_source("p", size)
            p = source[:, :, :3].astype(np.float32)
            image = np.clip((p - lo) / np.maximum(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
            image[source[:, :, 3] <= 0] = 0
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
        small = int(arrays["p"].shape[0])
        if "t" not in arrays:                      # 早先存的没有法线贴图：补一张平的
            flat = np.zeros((size, size, 4), np.uint8)
            flat[:, :] = (128, 128, 255, 255)
            arrays["t"] = flat
        if "i" not in arrays:                      # 早先存的没有部件：全部空白，重新烘焙才有
            arrays["i"] = np.full((small, small), -1.0, np.float32)

        def upload(name, comps, dtype):
            array = np.ascontiguousarray(arrays[name])
            return ctx.texture((int(array.shape[1]), int(array.shape[0])), comps, array.tobytes(), dtype=dtype)

        textures = {"a": upload("a", 4, "f1"), "n": upload("n", 4, "f1"), "p": upload("p", 4, "f2"),
                    "t": upload("t", 4, "f1")}
        for tex in textures.values():
            _finish_texture(tex)
        textures["i"] = upload("i", 1, "f4") if arrays["i"].dtype == np.float32 else ctx.texture(
            (int(arrays["i"].shape[1]), int(arrays["i"].shape[0])), 1,
            np.ascontiguousarray(arrays["i"], np.float32).tobytes(), dtype="f4")
        _finish_id_texture(textures["i"])
        maps = cls(ctx, size, textures, meta, arrays)
        maps.saved = True
        return maps

    def release(self) -> None:
        for tex in self.textures.values():
            tex.release()
        self.textures = {}


def _texture_mb(texture) -> float:
    """一张贴图大约占多少显存（MB，按格式和有没有多级）。"""
    w, h = texture.size
    per = {"f1": 1, "f2": 2, "f4": 4}.get(texture.dtype, 4) * int(texture.components)
    factor = 4.0 / 3.0 if texture.filter[0] in (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR_MIPMAP_NEAREST) else 1.0
    return w * h * per * factor / 1048576.0


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

    PREPARE, RASTER, PROJECT, RAYS, FINISH, DONE = range(6)
    #: 着色器程序在引擎上留着反复用（编译一次要几百毫秒，每次烘焙都编会卡一下）
    CACHED = ("raster", "rays", "finish", "finish_id", "project", "jfa_init_full", "jfa_init_small", "jfa_step",
              "jfa_fill_full", "jfa_fill_small")

    def __init__(self, engine, ts, settings) -> None:
        self.engine = engine
        self.ctx = engine.ctx
        self.ts = ts
        self.settings = {
            "size": int(min(MAX_SIZE, max(64, int(settings.resolution)))),
            "smooth_size": int(min(MAX_SIZE, max(64, int(getattr(settings, "smooth_resolution", "4096") or 4096)))),
            "ao_samples": int(max(1, settings.ao_samples)),
            "ao_distance": float(settings.ao_distance),
            "thickness": bool(settings.bake_thickness),
            "thickness_distance": float(settings.thickness_distance),
            "curvature_smooth": int(settings.curvature_smooth),
            "curvature_contrast": float(settings.curvature_contrast),
            "self_only": bool(settings.self_only),
            "padding": int(settings.padding),
            "high_poly": str(getattr(settings, "high_poly", "") or "").strip(),
            "high_poly_object": 0,
            "cage_front": float(getattr(settings, "cage_front", 0.02)),
            "cage_back": float(getattr(settings, "cage_back", 0.02)),
            "normal_samples": int(getattr(settings, "normal_samples", "4")),
            "id_source": str(getattr(settings, "id_source", "PART")),
        }
        self._high_mesh = None              # 场景里当高模的模型（这一刻的形状）
        self._high_uid = 0
        try:
            high_uid = int(getattr(settings, "high_poly_object", "0") or 0)
        except (TypeError, ValueError):
            high_uid = 0
        if high_uid:
            high_obj = next((o for o in engine.project.objects if o.uid == high_uid), None)
            if high_obj is not None:
                sculpt = getattr(engine, "sculpt", None)
                if sculpt is not None and sculpt.obj is high_obj:
                    sculpt.sync_to_object()
                edit = getattr(engine, "edit", None)
                data = edit.full_data() if (edit is not None and edit.obj is high_obj and edit.changed) else high_obj.data
                self._high_mesh = copy.copy(data)     # 数组不会原地改，浅拷贝就把这一刻的形状定住了
                self._high_uid = high_uid
                self.settings["high_poly"] = "场景：%s" % high_obj.name
                self.settings["high_poly_object"] = high_uid
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
        self.small_size = min(size, self.settings["smooth_size"])
        self.shrink = max(1, size // self.small_size)     # 都是 2 的幂
        self._finish_queue = None
        self._setup = None
        self.project_row = 0
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
            if not obj.visible or not getattr(obj.data, "has_uvs", True) or obj.uid == self._high_uid:
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
            if self._high_mesh is not None:
                high = self._high_mesh
            elif settings["high_poly"]:
                from ..doc.meshio import load_mesh

                self.status = "读取高模"
                high = load_mesh(settings["high_poly"])
            if high is not None:
                self.status = "准备高模"
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
        if self.stage == self.DONE:
            return 1.0
        if self.stage == self.FINISH:                 # 收尾占最后 5%
            queue = self._finish_queue
            total = max(1, getattr(self, "_finish_total", 1))
            left = len(queue) if queue is not None else total
            return min(0.999, 0.95 + 0.05 * (1.0 - left / total))
        total = len(self.tiles) * max(1, self.settings["ao_samples"])
        return min(0.95, 0.95 * (self.tile_index * self.settings["ao_samples"] + self.sample) / total)

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
                if self._warm_programs():             # 准备线程干活的时候，每帧先编一个着色器（第一次烘焙才要编）
                    return False
                if self._thread.is_alive():
                    return False
                if self._prep_error or self._prep is None:
                    self._finish_with("准备模型时出错：%s" % (self._prep_error or "未知原因"))
                    return True
                if self._prep["depth"] >= STACK:
                    self._finish_with("模型的 BVH 太深（%d 层），超过上限 %d" % (self._prep["depth"], STACK))
                    return True
                if self._setup is None:
                    self._setup = self._setup_steps()
                    self.status = "准备显卡资源"
                try:
                    next(self._setup)                 # 一帧一小步
                    return False
                except StopIteration:
                    self._setup = None
                self.stage = self.RASTER
            started = time.perf_counter()
            while not self.done and (time.perf_counter() - started) * 1000.0 < budget_ms:
                if self.stage == self.RASTER:
                    self._raster_tile()
                    self.project_row = 0
                    if "project" in self._gpu:
                        self.stage = self.PROJECT
                        self.status = "从高模取细节"
                    else:
                        self.stage = self.RAYS
                        self.status = "计算遮蔽和厚度"
                elif self.stage == self.PROJECT:
                    remaining = budget_ms - (time.perf_counter() - started) * 1000.0
                    if self._project_rows(max(2.0, remaining)):
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
                    if self._finish_queue is None:
                        self._finish_begin()
                    if self._finish_queue:
                        name, work = self._finish_queue.popleft()
                        self.status = name
                        work()
                    else:
                        self._finish_end()
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
        """一次把显卡资源准备好（测试用；烘焙时走 _setup_steps，分几帧做）。"""
        for _ in self._setup_steps():
            pass

    def _setup_steps(self):
        """准备显卡资源，分成几小步（生成器，每 yield 一次算一步）：建大贴图、传大缓冲时驱动要真正分配显存，
        一下子全做会卡几百毫秒。"""
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
        yield
        small = self.small_size
        gpu["out_a"] = ctx.texture((size, size), 4, dtype="f1")
        gpu["out_n"] = ctx.texture((small, small), 4, dtype="f1")
        gpu["out_p"] = ctx.texture((small, small), 4, dtype="f2")
        gpu["out_t"] = ctx.texture((size, size), 4, dtype="f1")
        gpu["out_i"] = ctx.texture((small, small), 1, dtype="f4")
        yield
        gpu["finish_id"] = self._program("finish_id", lambda: ctx.compute_shader(_FINISH_ID))
        if prep.get("tri_nrm") is not None:
            gpu["project"] = self._program("project", lambda: ctx.compute_shader(_PROJECT))
            gpu["tri_nrm"] = ctx.buffer(np.ascontiguousarray(prep["tri_nrm"], np.float32).tobytes())
            yield
            tri_id = prep.get("tri_id")
            gpu["tri_id"] = ctx.buffer(np.ascontiguousarray(tri_id if tri_id is not None else np.zeros(1),
                                                            np.float32).tobytes())
            yield
        gpu["raster"] = self._program("raster", lambda: ctx.program(vertex_shader=_RASTER_VERT,
                                                                  fragment_shader=_RASTER_FRAG))
        gpu["rays"] = self._program("rays", lambda: ctx.compute_shader(_RAYS))
        gpu["finish"] = self._program("finish", lambda: ctx.compute_shader(_FINISH))
        gpu["accum_fbo"] = ctx.framebuffer([gpu["accum"]])
        yield
        gpu["nodes"] = ctx.buffer(np.ascontiguousarray(prep["nodes"], np.float32).tobytes())
        yield
        gpu["tri_pos"] = ctx.buffer(np.ascontiguousarray(prep["tri_pos"], np.float32).tobytes())
        yield
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

    def _warm_programs(self) -> bool:
        """还有没编过的着色器就编一个，返回这一帧是不是编了。一帧只编一个，免得一下子卡几百毫秒。"""
        ctx = self.ctx
        cache = self.engine.__dict__.setdefault("bake_programs", {})
        builders = (("rays", lambda: ctx.compute_shader(_RAYS)),
                    ("project", lambda: ctx.compute_shader(_PROJECT)),
                    ("raster", lambda: ctx.program(vertex_shader=_RASTER_VERT, fragment_shader=_RASTER_FRAG)),
                    ("finish", lambda: ctx.compute_shader(_FINISH)),
                    ("finish_id", lambda: ctx.compute_shader(_FINISH_ID)),
                    ("jfa_init_full", lambda: ctx.compute_shader(_JFA_INIT % "rgba8")),
                    ("jfa_init_small", lambda: ctx.compute_shader(_JFA_INIT % "rgba16f")),
                    ("jfa_step", lambda: ctx.compute_shader(_JFA_STEP)),
                    ("jfa_fill_full", lambda: ctx.compute_shader(_JFA_FILL_FULL)),
                    ("jfa_fill_small", lambda: ctx.compute_shader(_JFA_FILL_SMALL)))
        for key, build in builders:
            if key not in cache:
                self.status = "准备着色器"
                self._program(key, build)
                return True
        return False

    def _program(self, key: str, build):
        cache = self.engine.__dict__.setdefault("bake_programs", {})
        program = cache.get(key)
        if program is None:
            program = cache[key] = build()
        return program

    def _release_gpu(self, keep_outputs: bool) -> None:
        gpu = self._gpu
        for name in self.CACHED:
            gpu.pop(name, None)
        for name in ("fbo", "accum_fbo", "gpos", "gnrm", "ggeo", "accum", "gtan", "gbit", "gtnorm", "gid", "nodes",
                     "tri_pos", "tri_nrm", "tri_id", "seed_a", "seed_b"):
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
        gpu["accum_fbo"].use()
        gpu["accum_fbo"].clear(0.0, 0.0, 0.0, 0.0)
        ctx.memory_barrier()

    def _project_rows(self, budget_ms: float) -> bool:
        """从高模取细节（按横条分批），返回这一块是否做完。"""
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
        deadline = time.perf_counter() + budget_ms / 1000.0
        while self.project_row < tile:
            row = self.project_row
            sub = min(rows, tile - row)
            program["u_size"] = (tile, row + sub)
            program["u_offset"] = (0, row)
            program.run((tile + 7) // 8, (sub + 7) // 8, 1)
            fence = glx.Fence()
            while not fence.ready(wait_ns=2_000_000):
                pass
            fence.release()
            self.project_row = row + sub
            if time.perf_counter() >= deadline:
                break
        self.ctx.memory_barrier()
        return self.project_row >= tile

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
        _set(program, "u_shrink", int(self.shrink))
        program.run((tile + 7) // 8, (tile + 7) // 8, 1)
        self.ctx.memory_barrier()
        program = gpu["finish_id"]
        gpu["gpos"].bind_to_image(0, read=True, write=False)
        gpu["gid"].bind_to_image(1, read=True, write=False)
        gpu["out_i"].bind_to_image(2, read=False, write=True)
        _set(program, "u_size", (tile, tile))
        _set(program, "u_tile_origin", (int(x0), int(y0)))
        _set(program, "u_shrink", int(self.shrink))
        program.run((tile + 7) // 8, (tile + 7) // 8, 1)
        self.ctx.memory_barrier()

    def _finish_begin(self) -> None:
        """收尾排成一串小步：先按块扩边（整图的和小图的），再逐张生成多级。"""
        from collections import deque

        ctx = self.ctx
        gpu = self._gpu
        size = self.settings["size"]
        small = self.small_size
        radius = float(self.settings["padding"])
        queue: deque = deque()
        if radius > 0:
            small_radius = max(1.0, radius / self.shrink)
            margin = int(math.ceil(radius))
            region = min(size, TILE + 2 * margin)
            gpu["seed_a"] = ctx.texture((region, region), 2, dtype="i4")
            gpu["seed_b"] = ctx.texture((region, region), 2, dtype="i4")
            gpu["jfa_init_full"] = self._program("jfa_init_full", lambda: ctx.compute_shader(_JFA_INIT % "rgba8"))
            gpu["jfa_init_small"] = self._program("jfa_init_small", lambda: ctx.compute_shader(_JFA_INIT % "rgba16f"))
            gpu["jfa_step"] = self._program("jfa_step", lambda: ctx.compute_shader(_JFA_STEP))
            gpu["jfa_fill_full"] = self._program("jfa_fill_full", lambda: ctx.compute_shader(_JFA_FILL_FULL))
            gpu["jfa_fill_small"] = self._program("jfa_fill_small", lambda: ctx.compute_shader(_JFA_FILL_SMALL))
            full_tiles = [(x, y) for y in range(0, size, TILE) for x in range(0, size, TILE)]
            small_tiles = [(x, y) for y in range(0, small, TILE) for x in range(0, small, TILE)]
            total = len(full_tiles) + len(small_tiles)
            for index, (x, y) in enumerate(full_tiles):
                queue.append(("扩边 %d/%d" % (index + 1, total),
                              lambda x=x, y=y: self._dilate_tile(False, x, y, size, radius)))
            for index, (x, y) in enumerate(small_tiles):
                queue.append(("扩边 %d/%d" % (len(full_tiles) + index + 1, total),
                              lambda x=x, y=y: self._dilate_tile(True, x, y, small, small_radius)))
        for name in ("out_a", "out_t", "out_n", "out_p"):
            queue.append(("生成多级贴图", lambda name=name: _finish_texture(gpu[name])))
        queue.append(("生成多级贴图", lambda: _finish_id_texture(gpu["out_i"])))
        self._finish_queue = queue
        self._finish_total = len(queue)

    def _dilate_tile(self, small: bool, x0: int, y0: int, size: int, radius: float) -> None:
        """扩一块的边：这一块加外圈做跳跃泛洪，再把这一块里没覆盖的格填上最近的有效纹素。"""
        ctx = self.ctx
        gpu = self._gpu
        margin = int(math.ceil(radius))
        tile_w = min(TILE, size - x0)
        tile_h = min(TILE, size - y0)
        ox, oy = max(0, x0 - margin), max(0, y0 - margin)
        rw = min(size, x0 + tile_w + margin) - ox
        rh = min(size, y0 + tile_h + margin) - oy
        groups = ((rw + 7) // 8, (rh + 7) // 8, 1)
        init = gpu["jfa_init_small" if small else "jfa_init_full"]
        (gpu["out_p"] if small else gpu["out_a"]).bind_to_image(0, read=True, write=False)
        gpu["seed_a"].bind_to_image(1, read=False, write=True)
        _set(init, "u_origin", (ox, oy))
        _set(init, "u_region", (rw, rh))
        init.run(*groups)
        ctx.memory_barrier()
        step = 1
        while step * 2 <= radius:
            step *= 2
        source, target = gpu["seed_a"], gpu["seed_b"]
        program = gpu["jfa_step"]
        _set(program, "u_origin", (ox, oy))
        _set(program, "u_region", (rw, rh))
        while step >= 1:
            source.bind_to_image(1, read=True, write=False)
            target.bind_to_image(2, read=False, write=True)
            _set(program, "u_step", int(step))
            program.run(*groups)
            ctx.memory_barrier()
            source, target = target, source
            step //= 2
        fill = gpu["jfa_fill_small" if small else "jfa_fill_full"]
        if small:
            gpu["out_n"].bind_to_image(3, read=True, write=True)
            gpu["out_p"].bind_to_image(4, read=True, write=True)
            gpu["out_i"].bind_to_image(6, read=True, write=True)
        else:
            gpu["out_a"].bind_to_image(0, read=True, write=True)
            gpu["out_t"].bind_to_image(5, read=True, write=True)
        source.bind_to_image(1, read=True, write=False)
        _set(fill, "u_origin", (ox, oy))
        _set(fill, "u_tile_origin", (x0, y0))
        _set(fill, "u_tile_size", (tile_w, tile_h))
        _set(fill, "u_radius", float(radius))
        fill.run((tile_w + 7) // 8, (tile_h + 7) // 8, 1)
        ctx.memory_barrier()

    def _finish_end(self) -> None:
        ctx = self.ctx
        gpu = self._gpu
        size = self.settings["size"]
        textures = {"a": gpu.pop("out_a"), "n": gpu.pop("out_n"), "p": gpu.pop("out_p"), "t": gpu.pop("out_t"),
                    "i": gpu.pop("out_i")}
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
