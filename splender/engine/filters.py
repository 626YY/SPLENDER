"""绘制层的滤镜（照 Photoshop 的滤镜菜单）：直接改图层的像素（或蒙版），可以撤销。

按块做：每块 TILE × TILE 格（第 0 级）。先把这一块连同四周一圈（核能够到的范围）从页里取到临时贴图
（预乘覆盖度、半精度），在临时贴图上跑滤镜的几遍，最后按格写成新页。全部写完再逐级生成多级、记撤销
（借用并入笔划的那一套：页面写好不再改，撤销就是把页表指回旧页）。
核很大时（大半径模糊、马赛克、径向模糊离中心远的地方）在粗一级上算，写回第 0 级时双线性放大：
这时模糊本身有十几个纹素宽，粗一级算出来和直接在第 0 级算看不出差别，快得多。
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque

import moderngl
import numpy as np

from ..doc.project import PLANE_FORMAT
from .layerstore import PLANE_KIND, MergeJob, RecordGroup
from .pageops import KIND_COLOR, KIND_HEIGHT, KIND_MASK, KIND_MR, SRGB_GLSL
from .pagepool import FORMATS, MAX_ARRAYS, PAGE, bind_samplers, glsl_fetch

log = logging.getLogger("splender.engine.filters")

TILE = 4                 # 一块每边几格（第 0 级）
WIDE_TILE = 16           # 不用取周围的（贴图、渐变、杂色、减淡……）一次做这么大一块，少些来回
SERVICE_EVERY = 256      # 大任务每生成这么多新页推进一帧（页面缓存要隔一帧才能把新页备份、换出）
MAX_HALO = 384           # 临时贴图四周最多留多少纹素
MAX_SPREAD_PAGES = 6     # 往空白处扩散最多扩几格

#: 中值、表面模糊、浮雕按哪个数排大小：颜色按明暗，金属粗糙两组平均，高度、蒙版就是那个数
KEYS = {KIND_COLOR: (0.2126, 0.7152, 0.0722, 0.0), KIND_MR: (0.5, 0.0, 0.5, 0.0), KIND_HEIGHT: (1.0, 0.0, 0.0, 0.0),
        KIND_MASK: (1.0, 0.0, 0.0, 0.0)}

# 遍的种类（临时贴图上）
P_GAUSS_H, P_GAUSS_V, P_MIN_H, P_MIN_V, P_MAX_H, P_MAX_V, P_MEDIAN, P_MOTION, P_RADIAL, P_SURFACE = range(1, 11)
# 写回的方式
(S_REPLACE, S_USM, S_HIGH_PASS, S_NOISE, S_MOSAIC, S_CLOUDS, S_EMBOSS, S_GRADIENT, S_IMAGE, S_TONE, S_SPONGE,
 S_HEAL, S_SCREEN_IMAGE, S_SELECT) = range(14)
#: S_SELECT 的几种：反选（选区自己）、只留选区里的、清掉选区里的、选区边界（选区自己，取模糊后的）
SEL_INVERT, SEL_KEEP, SEL_CLEAR, SEL_BORDER = range(4)
DEPTH_TOLERANCE = 3.0    # 投到模型上时，纹素和那个像素看到的表面相差几个像素宽以内算同一处（看得见）
#: 笔刷类的效果（减淡、加深……）：滤镜按这一帧新画的笔触当遮罩，只改笔刷扫过的地方
EFFECTS = ("DODGE", "BURN", "SPONGE", "BLUR", "SHARPEN", "SMUDGE", "CLONE", "HEAL")


# ====================================================================== 着色器
# 投到模型上：第 0 级纹素 g 投到屏幕上落在图的哪里（st，0..1）。没落在图里、或者被挡住（u_project.x = 1 时才判断）返回 false。
# 遮挡：这个像素和周围一圈看到的表面里取最远的，纹素不比它远太多就算看得见（贴着看的地方相邻像素远近差得多，轮廓边上的也算上）
_PROJECT_GLSL = """
bool project_texel(ivec2 g, out vec2 st) {
  st = vec2(-1.0);
  vec4 P = textureLod(u_pos, (vec2(g) + 0.5) / float(u_res), 0.0);
  vec4 clip = u_grad_matrix * vec4(P.xyz, 1.0);
  if (P.w <= 0.5 || clip.w <= 1e-6) return false;
  vec2 ndc = clip.xy / clip.w;
  vec2 q = vec2((ndc.x * 0.5 + 0.5) * u_view_size.x, (0.5 - ndc.y * 0.5) * u_view_size.y);
  st = (q - u_img_rect.xy) / max(u_img_rect.zw - u_img_rect.xy, vec2(1e-6));
  if (st.x < 0.0 || st.y < 0.0 || st.x > 1.0 || st.y > 1.0) return false;
  if (u_project.x < 0.5) return true;
  ivec2 size = textureSize(u_gbuf, 0);
  vec2 cq = q / u_view_size * vec2(size);
  ivec2 c = ivec2(int(floor(cq.x)), size.y - 1 - int(floor(cq.y)));
  float far = -1e30;
  bool seen = false;
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      vec4 s = texelFetch(u_gbuf, clamp(c + ivec2(dx, dy), ivec2(0), size - 1), 0);
      if (s.w > 0.5) {
        far = max(far, dot(u_depth_row, vec4(s.xyz, 1.0)));
        seen = true;
      }
    }
  }
  return seen && dot(u_depth_row, vec4(P.xyz, 1.0)) <= far + u_eps.x + u_eps.y * clip.w;
}
"""

# 「投到模型上」真会改到哪些格：每格一层（z），格里每个纹素都试一下，落在图里不透明的地方就记下这一格
_PROBE_SHADER = """#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 1) readonly buffer Cells { ivec2 cells[]; };
layout(std430, binding = 2) buffer Flags { uint flags[]; };
uniform int u_res; uniform float u_opacity; uniform int u_page;
uniform sampler2D u_pos; uniform mat4 u_grad_matrix; uniform vec2 u_view_size;
uniform sampler2D u_img; uniform vec4 u_img_rect;
uniform sampler2D u_gbuf; uniform vec2 u_eps; uniform vec4 u_project; uniform vec4 u_depth_row;
""" + _PROJECT_GLSL + """
void main() {
  uint index = gl_WorkGroupID.z;
  ivec2 g = cells[index] * u_page + ivec2(gl_GlobalInvocationID.xy);
  vec2 st;
  if (project_texel(g, st) && texture(u_img, st).a * u_opacity > 0.0) flags[index] = 1u;
}
"""


def _kind_glsl(kind: int, fmt: str) -> str:
    value = {KIND_COLOR: "vec4(1.0, 1.0, 1.0, 0.0)", KIND_MR: "vec4(1.0, 0.0, 1.0, 0.0)"}.get(kind, "vec4(1.0, 0.0, 0.0, 0.0)")
    return f"""
const int KIND = {kind};
const vec4 VALUE = {value};                  // 哪几个分量是数值（其余是覆盖度）
const float NEUTRAL = {0.0 if kind == KIND_HEIGHT else 0.5};
const float LO = {-1.0 if kind == KIND_HEIGHT else 0.0};
uniform int u_grid; uniform int u_res; uniform float u_mask_default;
layout(std430, binding = 2) readonly buffer Table {{ int table[]; }};
vec4 empty_u() {{ return KIND == 3 ? vec4(u_mask_default, 1.0, 0.0, 0.0) : vec4(0.0); }}
// 页里的原始值 → (数值, 覆盖度…)：颜色 rgb+覆盖；金属、覆盖、粗糙、覆盖；高度、覆盖；蒙版、1
vec4 to_u(vec4 raw) {{
  if (KIND == 2) return vec4(raw.rg, 0.0, 0.0);
  if (KIND == 3) return vec4(raw.r, 1.0, 0.0, 0.0);
  return raw;
}}
vec4 to_raw(vec4 u) {{
  if (KIND == 2) return vec4(u.xy, 0.0, 0.0);
  if (KIND == 3) return vec4(u.x);
  return u;
}}
vec4 premul(vec4 u) {{
  if (KIND == 0) return vec4(u.rgb * u.a, u.a);
  if (KIND == 1) return vec4(u.x * u.y, u.y, u.z * u.w, u.w);
  return vec4(u.x * u.y, u.y, 0.0, 0.0);
}}
vec4 unpremul(vec4 p) {{
  if (KIND == 0) return p.a > 1e-5 ? vec4(p.rgb / p.a, p.a) : vec4(0.0);
  if (KIND == 1) return vec4(p.y > 1e-5 ? p.x / p.y : 0.0, p.y, p.w > 1e-5 ? p.z / p.w : 0.0, p.w);
  return vec4(p.y > 1e-5 ? p.x / p.y : 0.0, p.y, 0.0, 0.0);
}}
vec4 read_u(ivec2 t) {{
  t = clamp(t, ivec2(0), ivec2(u_res - 1));
  ivec2 cell = t >> 8;
  int slot = table[cell.y * u_grid + cell.x];
  if (slot < 0) return empty_u();
  return to_u(fetch_{fmt}(slot, t & 255));
}}
"""


def _gather_shader(kind: int, fetch: str, fmt: str) -> str:
    return f"""#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(rgba16f, binding = 1) writeonly uniform image2D u_out;
uniform ivec2 u_origin; uniform ivec2 u_size;
{fetch}
{_kind_glsl(kind, fmt)}
void main() {{
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  imageStore(u_out, p, premul(read_u(u_origin + p)));
}}
"""


_PASS_SHADER = """#version 430
layout(local_size_x = 16, local_size_y = 16) in;
uniform sampler2D u_src;
layout(rgba16f, binding = 1) writeonly uniform image2D u_out;
uniform ivec2 u_size; uniform int u_mode; uniform vec4 u_p0; uniform vec4 u_p1; uniform vec4 u_key;
vec4 at(ivec2 q) { return texelFetch(u_src, clamp(q, ivec2(0), u_size - 1), 0); }
vec4 smooth_at(vec2 q) {
  q = clamp(q, vec2(0.0), vec2(u_size - 1));
  return texture(u_src, (q + 0.5) / vec2(textureSize(u_src, 0)));
}
void main() {
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  if (p.x >= u_size.x || p.y >= u_size.y) return;
  vec4 r = at(p);
  if (u_mode == 1 || u_mode == 2) {
    // 高斯模糊（横、竖各一遍）
    float sigma = max(u_p0.x, 0.05);
    int R = int(ceil(sigma * 3.0));
    ivec2 dir = u_mode == 1 ? ivec2(1, 0) : ivec2(0, 1);
    vec4 sum = vec4(0.0); float wsum = 0.0;
    for (int i = -R; i <= R; i++) {
      float w = exp(-0.5 * float(i * i) / (sigma * sigma));
      sum += w * at(p + dir * i);
      wsum += w;
    }
    r = sum / wsum;
  } else if (u_mode >= 3 && u_mode <= 6) {
    // 最小值、最大值（横、竖各一遍，方形）
    int R = int(u_p0.x);
    ivec2 dir = (u_mode == 3 || u_mode == 5) ? ivec2(1, 0) : ivec2(0, 1);
    bool lowest = u_mode <= 4;
    for (int i = -R; i <= R; i++) {
      vec4 s = at(p + dir * i);
      r = lowest ? min(r, s) : max(r, s);
    }
  } else if (u_mode == 7) {
    // 中值：按 u_key 排大小，二分找中值，取最接近它的那个样本（颜色不会混出新的）
    int R = int(u_p0.x);
    float lo = 1e9; float hi = -1e9;
    for (int y = -R; y <= R; y++) for (int x = -R; x <= R; x++) {
      float k = dot(at(p + ivec2(x, y)), u_key);
      lo = min(lo, k); hi = max(hi, k);
    }
    int need = ((2 * R + 1) * (2 * R + 1) + 1) / 2;
    for (int it = 0; it < 10; it++) {
      float mid = 0.5 * (lo + hi);
      int count = 0;
      for (int y = -R; y <= R; y++) for (int x = -R; x <= R; x++) {
        if (dot(at(p + ivec2(x, y)), u_key) <= mid) count++;
      }
      if (count >= need) hi = mid; else lo = mid;
    }
    float best = 1e9;
    for (int y = -R; y <= R; y++) for (int x = -R; x <= R; x++) {
      vec4 s = at(p + ivec2(x, y));
      float d = abs(dot(s, u_key) - hi);
      if (d < best) { best = d; r = s; }
    }
  } else if (u_mode == 8) {
    // 动感模糊：沿一个方向等权取样
    int taps = max(int(u_p0.w), 2);
    vec4 sum = vec4(0.0);
    for (int i = 0; i < taps; i++) {
      float t = float(i) / float(taps - 1) * 2.0 - 1.0;
      sum += smooth_at(vec2(p) + u_p0.xy * (u_p0.z * t));
    }
    r = sum / float(taps);
  } else if (u_mode == 9) {
    // 径向模糊：旋转（沿圆弧）或缩放（沿指向中心的直线）
    int taps = max(int(u_p0.w), 2);
    vec2 v = vec2(p) - u_p0.xy;
    vec4 sum = vec4(0.0);
    for (int i = 0; i < taps; i++) {
      float t = float(i) / float(taps - 1);
      vec2 q;
      if (u_p1.x > 0.5) {
        q = u_p0.xy + v * (1.0 - u_p0.z * t);
      } else {
        float a = u_p0.z * (t - 0.5);
        q = u_p0.xy + vec2(v.x * cos(a) - v.y * sin(a), v.x * sin(a) + v.y * cos(a));
      }
      sum += smooth_at(q);
    }
    r = sum / float(taps);
  } else if (u_mode == 10) {
    // 表面模糊：只和差不多的邻居平均，边缘保持清楚
    int R = int(u_p0.x);
    float threshold = max(u_p0.y, 1e-4);
    vec4 c = r;
    vec4 sum = vec4(0.0); float wsum = 0.0;
    for (int y = -R; y <= R; y++) for (int x = -R; x <= R; x++) {
      vec4 s = at(p + ivec2(x, y));
      float d = abs(dot(s - c, u_key)) + abs(s.a - c.a) * 0.25;
      float w = clamp(1.0 - d / (2.5 * threshold), 0.0, 1.0);
      sum += s * w; wsum += w;
    }
    r = wsum > 0.0 ? sum / wsum : c;
  }
  imageStore(u_out, p, r);
}
"""


def _scatter_shader(kind: int, fetch: str, fmt: str, stroke_fetch: str, sel_fetch: str) -> str:
    glsl = FORMATS[fmt][5]
    return f"""#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout({glsl}, binding = 0) writeonly uniform image2DArray u_dst;
layout(std430, binding = 1) readonly buffer Jobs {{ int jobs[]; }};
uniform int u_base; uniform int u_layer_mask;
uniform sampler2D u_proc; uniform ivec2 u_proc_size; uniform vec2 u_proc_origin; uniform float u_scale;
// 修复画笔：来源的低频（u_proc_b，和 u_proc 同一片）、落笔处的低频（u_proc_c，从 u_proc_c_origin 起）
uniform sampler2D u_proc_b; uniform sampler2D u_proc_c; uniform vec2 u_proc_c_origin;
uniform int u_mode; uniform vec4 u_p0; uniform vec4 u_p1; uniform vec4 u_c0; uniform vec4 u_c1; uniform vec4 u_key;
uniform ivec2 u_use;
// 渐变：位置贴图（三维时把每个纹素投到屏幕上）、色带（256 × 1）、起点终点（UV 或屏幕像素）
uniform sampler2D u_pos; uniform sampler2D u_ramp; uniform mat4 u_grad_matrix; uniform vec4 u_grad_ab;
uniform vec2 u_view_size;
// 贴图：一张 RGBA（不预乘）铺在 UV 的一块矩形上，按它的透明度盖上去；数值页用 u_c0 =（金属, 粗糙, 高度）
uniform sampler2D u_img; uniform vec4 u_img_rect;
// 投到模型上：u_img_rect 是图在视口里的位置（像素，左上原点）；u_gbuf 是那时视口看到的表面位置（第一行在下）；
// u_depth_row 点乘 (位置, 1) 是离镜头多远；比看到的表面远不超过 u_eps.x + u_eps.y × w 算看得见；
// u_project.x = 1 时只投到看得见的地方
uniform sampler2D u_gbuf; uniform vec2 u_eps; uniform vec4 u_project; uniform vec4 u_depth_row;
// 笔刷类效果：这一帧新画的笔触（笔划页的覆盖度）当遮罩，乘强度；取样位置可以整体挪开（仿制、涂抹）
layout(std430, binding = 3) readonly buffer MaskTable {{ int mask_table[]; }};
uniform int u_mask_on; uniform float u_strength; uniform vec2 u_src_offset;
// 选区：每格的选区页槽号（-1 是没有页，按 u_sel_default）；u_sel_on 取不取选区值，u_sel_mix 最后按它过渡
layout(std430, binding = 4) readonly buffer SelTable {{ int sel_table[]; }};
uniform int u_sel_on; uniform int u_sel_mix; uniform float u_sel_default;
{fetch}
{stroke_fetch}
{sel_fetch}
{SRGB_GLSL}
// 颜色盖上去（不预乘、sRGB）：和笔刷、图层叠加一样在线性光里混，半透明盖上去和半透明图层看起来一样
vec4 over_color(vec4 under, vec3 color, float a) {{
  float cov = a + under.a * (1.0 - a);
  vec3 lin = cov > 1e-6 ? (srgb_to_linear(color) * a + srgb_to_linear(under.rgb) * under.a * (1.0 - a)) / cov : vec3(0.0);
  return vec4(linear_to_srgb(lin), cov);
}}
{_kind_glsl(kind, fmt)}
{_PROJECT_GLSL}
// 临时贴图（第 k 级，从 origin 起）在第 0 级纹素 g 处的值，双线性
vec4 sample_temp(sampler2D s, vec2 origin, vec2 g) {{
  vec2 q = (g + 0.5) / u_scale - 0.5 - origin;
  q = clamp(q, vec2(0.0), vec2(u_proc_size - 1));
  return texture(s, (q + 0.5) / vec2(textureSize(s, 0)));
}}
vec4 proc_at(vec2 g) {{ return sample_temp(u_proc, u_proc_origin, g); }}
float sel_at(ivec2 tile, ivec2 p) {{
  if (u_sel_on == 0) return 1.0;
  int s = sel_table[tile.y * u_grid + tile.x];
  return s >= 0 ? clamp(fetch_r8(s, p).r, 0.0, 1.0) : u_sel_default;
}}
// 按 m 从原值过渡到新值：颜色在线性光里（和笔刷边缘一样），别的在预乘的样子里
vec4 blend_by(vec4 orig, vec4 res, float m) {{
  vec4 out_;
  if (KIND == 0) {{
    vec4 a = vec4(srgb_to_linear(orig.rgb) * orig.a, orig.a);
    vec4 b = vec4(srgb_to_linear(res.rgb) * res.a, res.a);
    vec4 mixed = mix(a, b, m);
    out_ = mixed.a > 1e-5 ? vec4(linear_to_srgb(mixed.rgb / mixed.a), mixed.a) : vec4(0.0);
  }} else {{
    out_ = unpremul(mix(premul(orig), premul(res), m));
  }}
  if (KIND == 3) out_.y = 1.0;
  return out_;
}}
uint pcg(uint v) {{
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}}
float rnd(ivec2 g, uint salt) {{
  return float(pcg(uint(g.x) * 1973u + pcg(uint(g.y) * 9277u + salt * 26699u + uint(u_p1.x) * 7919u))) / 4294967295.0;
}}
float gauss(ivec2 g, uint salt) {{
  float a = max(rnd(g, salt), 1e-7);
  float b = rnd(g, salt + 101u);
  return sqrt(-2.0 * log(a)) * cos(6.2831853 * b);
}}
// 云彩：平滑的值噪声叠几层
float vnoise(vec2 x, float seed) {{
  vec2 i = floor(x); vec2 f = x - i;
  vec2 u = f * f * (3.0 - 2.0 * f);
  uint s = uint(seed);
  float a = float(pcg(uint(int(i.x) & 65535) * 7919u + pcg(uint(int(i.y) & 65535) + s))) / 4294967295.0;
  float b = float(pcg(uint(int(i.x + 1.0) & 65535) * 7919u + pcg(uint(int(i.y) & 65535) + s))) / 4294967295.0;
  float c = float(pcg(uint(int(i.x) & 65535) * 7919u + pcg(uint(int(i.y + 1.0) & 65535) + s))) / 4294967295.0;
  float d = float(pcg(uint(int(i.x + 1.0) & 65535) * 7919u + pcg(uint(int(i.y + 1.0) & 65535) + s))) / 4294967295.0;
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}}
float clouds(vec2 x, float seed) {{
  float sum = 0.0; float amp = 0.5; float norm = 0.0;
  for (int o = 0; o < 7; o++) {{
    sum += amp * vnoise(x, seed + float(o) * 13.0);
    norm += amp; amp *= 0.5; x *= 2.0;
  }}
  return sum / norm;
}}
void main() {{
  int j = (u_base + int(gl_GlobalInvocationID.z)) * 8;
  int dst = jobs[j];
  ivec2 tile = ivec2(jobs[j + 1], jobs[j + 2]);
  ivec2 p = ivec2(gl_GlobalInvocationID.xy);
  ivec2 g = tile * 256 + p;
  vec4 orig = read_u(g);
  vec4 res = orig;
  if (u_mode == 0) {{
    res = unpremul(proc_at(vec2(g) + u_src_offset));
  }} else if (u_mode == 1 || u_mode == 2) {{
    // USM 锐化：原值 + 数量 ×（原值 − 模糊），差别小于阈值的不动；高反差保留：中性值 +（原值 − 模糊）
    vec4 blur = unpremul(proc_at(vec2(g)));
    vec4 diff = (orig - blur) * VALUE;
    if (u_mode == 1) res = orig + diff * step(vec4(u_p0.y), abs(diff)) * u_p0.x;
    else res = orig * (1.0 - VALUE) + (vec4(NEUTRAL) + diff) * VALUE;
  }} else if (u_mode == 3) {{
    // 添加杂色：u_p0 = (数量, 高斯分布, 单色)
    vec4 n;
    for (int k = 0; k < 4; k++) {{
      uint salt = uint(u_p0.z > 0.5 ? 0 : k);
      n[k] = u_p0.y > 0.5 ? gauss(g, salt) * 0.5 : rnd(g, salt) - 0.5;
    }}
    if (KIND == 1) n.z = u_p0.z > 0.5 ? n.x : n.y;
    res = orig + n * VALUE * u_p0.x;
  }} else if (u_mode == 4) {{
    // 马赛克：格子（u_p0.x 纹素，对齐图的原点）里求平均
    float c = u_p0.x;
    vec2 cell0 = floor(vec2(g) / c) * c;
    vec2 a = cell0 / u_scale - u_proc_origin;
    vec2 b = (cell0 + c) / u_scale - u_proc_origin;
    vec4 sum = vec4(0.0); float wsum = 0.0;
    ivec2 i0 = ivec2(floor(a));
    ivec2 i1 = ivec2(ceil(b)) - 1;
    for (int y = i0.y; y <= i1.y; y++) for (int x = i0.x; x <= i1.x; x++) {{
      float wx = min(b.x, float(x + 1)) - max(a.x, float(x));
      float wy = min(b.y, float(y + 1)) - max(a.y, float(y));
      if (wx <= 0.0 || wy <= 0.0) continue;
      sum += texelFetch(u_proc, clamp(ivec2(x, y), ivec2(0), u_proc_size - 1), 0) * wx * wy;
      wsum += wx * wy;
    }}
    res = unpremul(sum / max(wsum, 1e-6));
  }} else if (u_mode == 5) {{
    // 云彩：u_p0 = (块大小, 种子, 对比度, 高度幅度)，颜色在两种颜色之间
    float v = clouds(vec2(g) / max(u_p0.x, 1.0), u_p0.y * 131.0);
    v = clamp((v - 0.5) * u_p0.z + 0.5, 0.0, 1.0);
    if (KIND == 0) res = vec4(mix(u_c0.rgb, u_c1.rgb, v), 1.0);
    else if (KIND == 1) res = vec4(v, 1.0, v, 1.0);
    else if (KIND == 2) res = vec4((v - 0.5) * 2.0 * u_p0.w, 1.0, 0.0, 0.0);
    else res = vec4(v, 1.0, 0.0, 0.0);
  }} else if (u_mode == 6) {{
    // 浮雕：沿角度方向前后两点的差，加到中性值上；u_p0 = (角度, 距离, 数量)
    vec2 d = vec2(cos(u_p0.x), sin(u_p0.x)) * u_p0.y;
    float ka = dot(unpremul(proc_at(vec2(g) + d)), u_key);
    float kb = dot(unpremul(proc_at(vec2(g) - d)), u_key);
    float v = NEUTRAL + (ka - kb) * u_p0.z;
    res = orig * (1.0 - VALUE) + vec4(v) * VALUE;
  }}
  else if (u_mode == 7) {{
    // 渐变：u_p0 = (形状, 反向, 不透明度, 用屏幕坐标)
    vec2 uv = (vec2(g) + 0.5) / float(u_res);
    vec2 q = uv;
    if (u_p0.w > 0.5) {{
      vec4 clip = u_grad_matrix * vec4(texture(u_pos, uv).xyz, 1.0);
      vec2 ndc = clip.xy / max(clip.w, 1e-6);
      q = vec2((ndc.x * 0.5 + 0.5) * u_view_size.x, (0.5 - ndc.y * 0.5) * u_view_size.y);
    }}
    vec2 a = u_grad_ab.xy;
    vec2 d = u_grad_ab.zw - a;
    float len = max(length(d), 1e-9);
    vec2 r = q - a;
    int shape = int(u_p0.x + 0.5);
    float t;
    if (shape == 0) t = dot(r, d) / (len * len);
    else if (shape == 1) t = length(r) / len;
    else if (shape == 2) t = fract((atan(r.y, r.x) - atan(d.y, d.x)) / 6.2831853);
    else if (shape == 3) t = abs(dot(r, d)) / (len * len);
    else {{
      vec2 n = d / len;
      t = (abs(dot(r, n)) + abs(dot(r, vec2(-n.y, n.x)))) / len;
    }}
    t = clamp(t, 0.0, 1.0);
    if (u_p0.y > 0.5) t = 1.0 - t;
    vec4 c = texture(u_ramp, vec2((t * 255.0 + 0.5) / 256.0, 0.5));
    float a2 = clamp(c.a * u_p0.z, 0.0, 1.0);
    if (KIND == 0) {{
      res = over_color(orig, c.rgb, a2);
    }} else {{
      float v = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
      res = orig * (1.0 - VALUE) + mix(orig, vec4(v), a2) * VALUE;
    }}
  }}
  else if (u_mode == 8) {{
    vec2 uv = (vec2(g) + 0.5) / float(u_res);
    vec2 st = (uv - u_img_rect.xy) / max(u_img_rect.zw - u_img_rect.xy, vec2(1e-9));
    if (st.x >= 0.0 && st.y >= 0.0 && st.x <= 1.0 && st.y <= 1.0) {{
      vec4 c = texture(u_img, st);
      if (u_p0.y > 0.5) c = vec4(u_c1.rgb, c.r);        // 只给了遮罩：颜色是 u_c1
      float a2 = clamp(c.a * u_p0.x, 0.0, 1.0);
      if (KIND == 3 && u_p0.z > 2.5) {{
        res = vec4(orig.x * a2, 1.0, 0.0, 0.0);         // 交叉：相乘
      }} else if (KIND == 0) {{
        res = over_color(orig, c.rgb, a2);
      }} else if (KIND == 1) {{
        float cm = a2 + orig.y * (1.0 - a2);
        float cr = a2 + orig.w * (1.0 - a2);
        res = vec4(cm > 1e-6 ? (u_c0.x * a2 + orig.x * orig.y * (1.0 - a2)) / cm : 0.0, cm,
                   cr > 1e-6 ? (u_c0.y * a2 + orig.z * orig.w * (1.0 - a2)) / cr : 0.0, cr);
      }} else if (KIND == 2) {{
        float ch = a2 + orig.y * (1.0 - a2);
        res = vec4(ch > 1e-6 ? (u_c0.z * a2 + orig.x * orig.y * (1.0 - a2)) / ch : 0.0, ch, 0.0, 0.0);
      }} else {{
        res = vec4(mix(orig.x, dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)), a2), 1.0, 0.0, 0.0);
      }}
    }} else if (KIND == 3 && u_p0.z > 2.5) {{
      res = vec4(0.0, 1.0, 0.0, 0.0);                   // 交叉：图外归 0
    }}
  }}
  else if (u_mode == 9) {{
    // 减淡（u_p0.x = 1）、加深（-1）：u_p0.y = 范围（0 阴影、1 中间调、2 高光），u_p0.z = 保护色调
    float l = KIND == 0 ? dot(orig.rgb, vec3(0.2126, 0.7152, 0.0722)) : orig.x;
    int range = int(u_p0.y + 0.5);
    float w = range == 0 ? (1.0 - l) * (1.0 - l) : (range == 1 ? 4.0 * l * (1.0 - l) : l * l);
    w = 0.25 + 0.75 * w;
    vec4 up = orig + (1.0 - orig) * (0.5 * w);
    vec4 down = orig * (1.0 - 0.5 * w);
    vec4 changed = u_p0.x > 0.0 ? up : down;
    if (KIND == 0 && u_p0.z > 0.5) {{
      // 保护色调：只改明暗，色相、饱和度尽量不变
      float target = dot(changed.rgb, vec3(0.2126, 0.7152, 0.0722));
      vec3 c = orig.rgb + (target - l);
      float n = min(c.r, min(c.g, c.b));
      float x = max(c.r, max(c.g, c.b));
      if (n < 0.0) c = target + (c - target) * (target / max(target - n, 1e-6));
      if (x > 1.0) c = target + (c - target) * ((1.0 - target) / max(x - target, 1e-6));
      changed.rgb = c;
    }}
    res = orig * (1.0 - VALUE) + changed * VALUE;
  }}
  else if (u_mode == 10) {{
    // 海绵：u_p0.x = 1 加色、-1 去色；u_p0.y = 自然饱和度（不太艳的先动）
    if (KIND == 0) {{
      float l = dot(orig.rgb, vec3(0.2126, 0.7152, 0.0722));
      float s = max(orig.r, max(orig.g, orig.b)) - min(orig.r, min(orig.g, orig.b));
      float amount = 0.5 * (u_p0.y > 0.5 ? (1.0 - s) : 1.0);
      vec3 c = l + (orig.rgb - l) * (1.0 + u_p0.x * amount);
      float n = min(c.r, min(c.g, c.b));
      float x = max(c.r, max(c.g, c.b));
      if (n < 0.0) c = l + (c - l) * (l / max(l - n, 1e-6));
      if (x > 1.0) c = l + (c - l) * ((1.0 - l) / max(x - l, 1e-6));
      res = vec4(c, orig.a);
    }}
  }}
  else if (u_mode == 11) {{
    // 修复画笔：来源的细节 +（落笔处的低频 − 来源的低频），颜色、明暗跟着落笔处走
    vec2 src = vec2(g) + u_src_offset;
    vec4 a = unpremul(proc_at(src));
    vec4 b = unpremul(sample_temp(u_proc_b, u_proc_origin, src));
    vec4 c = unpremul(sample_temp(u_proc_c, u_proc_c_origin, vec2(g)));
    res = orig * (1.0 - VALUE) + (a + (c - b)) * VALUE;
  }}
  else if (u_mode == 12) {{
    // 纹素的位置投到屏幕上，落在图里（而且没被挡住）才盖
    vec2 st;
    if (project_texel(g, st)) {{
      vec4 c = texture(u_img, st);
      if (u_p0.y > 0.5) c = vec4(u_c1.rgb, c.r);        // 只给了遮罩：颜色是 u_c1
      float a2 = clamp(c.a * u_p0.x, 0.0, 1.0);
      if (KIND == 3 && u_p0.z > 2.5) {{
        res = vec4(orig.x * a2, 1.0, 0.0, 0.0);         // 交叉：相乘
      }} else if (KIND == 0) {{
        res = over_color(orig, c.rgb, a2);
      }} else if (KIND == 1) {{
        float cm = a2 + orig.y * (1.0 - a2);
        float cr = a2 + orig.w * (1.0 - a2);
        res = vec4(cm > 1e-6 ? (u_c0.x * a2 + orig.x * orig.y * (1.0 - a2)) / cm : 0.0, cm,
                   cr > 1e-6 ? (u_c0.y * a2 + orig.z * orig.w * (1.0 - a2)) / cr : 0.0, cr);
      }} else if (KIND == 2) {{
        float ch = a2 + orig.y * (1.0 - a2);
        res = vec4(ch > 1e-6 ? (u_c0.z * a2 + orig.x * orig.y * (1.0 - a2)) / ch : 0.0, ch, 0.0, 0.0);
      }} else {{
        res = vec4(mix(orig.x, dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)), a2), 1.0, 0.0, 0.0);
      }}
    }} else if (KIND == 3 && u_p0.z > 2.5) {{
      res = vec4(0.0, 1.0, 0.0, 0.0);                   // 交叉：没投到（或被挡住）的地方归 0
    }}
  }}
  else if (u_mode == 13) {{
    // 选区的运算：u_p0.x = 0 反选、1 只留选区里的、2 清掉选区里的、3 边界（取模糊后的，0.5 处最亮）
    int op = int(u_p0.x + 0.5);
    if (op == 0) {{
      res = vec4(1.0 - orig.x, 1.0, 0.0, 0.0);
    }} else if (op == 3) {{
      float b = unpremul(proc_at(vec2(g))).x;
      res = vec4(clamp(1.0 - abs(2.0 * b - 1.0), 0.0, 1.0), 1.0, 0.0, 0.0);
    }} else if (KIND != 3 || op == 2) {{
      float s = sel_at(tile, p);
      float k = op == 1 ? s : 1.0 - s;
      if (KIND == 0) res = vec4(orig.rgb, orig.a * k);
      else if (KIND == 1) res = vec4(orig.x, orig.y * k, orig.z, orig.w * k);
      else if (KIND == 2) res = vec4(orig.x, orig.y * k, 0.0, 0.0);
      else res = vec4(orig.x * k, 1.0, 0.0, 0.0);
    }}
  }}
  res = mix(clamp(res, 0.0, 1.0), clamp(res, LO, 1.0), VALUE);
  if (u_mask_on == 1) {{
    // 只改这一帧笔刷扫过的地方：按笔划页的覆盖度乘强度，在预乘的样子里过渡
    int mslot = mask_table[tile.y * u_grid + tile.x];
    float m = mslot >= 0 ? clamp(fetch_rg16f(mslot, p).r, 0.0, 1.0) * u_strength : 0.0;
    res = blend_by(orig, res, m);
  }}
  if (u_sel_mix == 1) {{
    // 有选区：只改选区里面（羽化过的边按选区值过渡）
    res = blend_by(orig, res, sel_at(tile, p));
  }}
  if (KIND == 1) {{
    if (u_use.x == 0) res.xy = orig.xy;
    if (u_use.y == 0) res.zw = orig.zw;
  }}
  imageStore(u_dst, ivec3(p, dst & u_layer_mask), to_raw(res));
}}
"""


# ====================================================================== 多级、撤销
class FilterJob(MergeJob):
    """滤镜写好的第 0 级新页交给并入笔划那一套：逐级生成多级、记撤销、最后一次换上。"""

    def __init__(self, pixels, layer_uid: int, grid: int, formats, mask_default: float = 1.0) -> None:  # noqa: D107
        # 不调用 MergeJob.__init__（那是给笔划用的）：只借它往上逐级生成、记录、提交的部分
        self.mask_default = float(mask_default)
        self.cache = pixels.cache
        self.ops = pixels.ops
        self.stroke = None
        self.params = None
        self.store = pixels.store(layer_uid, grid)
        self.records = []
        self.staged = {}
        self.queue = deque()
        self.stall = 0
        self.pages_done = 0
        self.formats = sorted(set(formats))

    def add(self, plane: int, xs: list, ys: list, news: list) -> None:
        here = self.store.plane(plane).levels[0]
        olds = [here.pages.get((x, y)) for x, y in zip(xs, ys)]
        staged = self.staged.setdefault((plane, 0), {})
        for x, y, page in zip(xs, ys, news):
            staged[(x, y)] = page
        self.records.append(RecordGroup(here, 0, xs, ys, olds, news))

    def finish(self) -> list:
        for (plane, mip), staged in list(self.staged.items()):
            levels = self.store.plane(plane).levels
            if mip != 0 or len(levels) < 2 or not staged:
                continue
            size = levels[0].size
            cells = np.fromiter((y * size + x for (x, y) in staged), np.int64, len(staged))
            parents = np.unique(((cells // size) >> 1) * (size >> 1) + ((cells % size) >> 1))
            self.queue.append([plane, 1, parents, 0])
        # 和 MergeJob.step(block=True) 一样逐级往上做，只是每一小块之后推进一帧：
        # 早先生成的页这样才能备份、换出，16K 整层也不会把显存页池挤满
        cache = self.cache
        made = 0
        while self.queue:
            item = self.queue[0]
            plane, mip, tiles, cursor = item
            count = self._process(plane, mip, tiles[cursor:cursor + 128], True)
            item[3] += count
            made += count
            if item[3] >= len(tiles):
                self.queue.popleft()
                levels = self.store.plane(plane).levels
                if mip + 1 < len(levels):
                    size = levels[mip].size
                    parents = np.unique(((tiles // size) >> 1) * (size >> 1) + ((tiles % size) >> 1))
                    self.queue.appendleft([plane, mip + 1, parents, 0])
            if made >= SERVICE_EVERY:
                made = 0
                cache.service(cache.frame + 1, budget_ms=2.0)
        self._clear_reserve()
        return self.commit()



def capture_view(engine, view) -> int:
    """记下视口此刻看到的表面位置（留在显卡上，第一行在下），往模型上投图时用它判断哪里被挡住。
    只留最近一份，返回它的编号（参数 capture 用）。"""
    view.renderer.ensure_gbuffer(view)
    old = engine.__dict__.get("screen_capture")
    if old is not None:
        old[1].release()
    ctx = engine.ctx
    texture = ctx.texture((view.width, view.height), 4, dtype="f4")
    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    texture.repeat_x = False
    texture.repeat_y = False
    # 帧缓冲之间拷（直接拷进贴图会被压成 0..1 的 8 位）：目标的附件要和来源一样多，后面几个拷完就扔
    spares = [ctx.texture(a.size, a.components, dtype=a.dtype) for a in view.gbuf_fbo.color_attachments[1:]]
    target = ctx.framebuffer([texture] + spares)
    try:
        ctx.copy_framebuffer(target, view.gbuf_fbo)
    finally:
        target.release()
        for spare in spares:
            spare.release()
    token = (old[0] + 1) if old is not None else 1
    engine.screen_capture = (token, texture)
    return token

# ====================================================================== 滤镜本体
class LayerFilter:
    """对一个图层的几种页做一次滤镜。run() 返回撤销记录（新页已经换上）。

    planes：{页种类: (改第一组, 改第二组)}——金属粗糙页两组（金属、粗糙）各管各，别的页只看第一个。"""

    def __init__(self, engine, ts, layer, name: str, params: dict, planes: dict) -> None:
        self.engine = engine
        self.ts = ts
        self.layer = layer
        self.name = name
        self.p = dict(params)
        self.planes = dict(planes)
        self.grid = int(ts.size) // PAGE
        self.res0 = self.grid * PAGE
        self.stats: dict = {}
        self._owned: list = []          # 这一次用完就放掉的贴图（图、色带）

    # ---- 入口 ----
    def run(self) -> list:
        started = time.perf_counter()
        engine = self.engine
        self._ramp_texture = None
        self._image_texture = None
        self._gbuf_texture = None
        self._screen_targets = None
        self._selection = None
        self._sel_mix = False
        found = engine.active_selection(self.ts) if hasattr(engine, "active_selection") else None
        if found is not None and found.applies_to(self.layer):
            spec_needs = self.name in ("SEL_KEEP", "SEL_CLEAR")
            if spec_needs or not self.p.get("ignore_selection"):
                self._selection = found
                self._sel_mix = not spec_needs
        elif self.name in ("SEL_KEEP", "SEL_CLEAR"):
            raise ValueError("没有选区")
        self._eps = (0.0, 0.0)
        self._depth_row = (0.0, 0.0, 0.0, 0.0)
        if self.name == "SCREEN_IMAGE":
            if self._position_texture() is None:
                raise ValueError("还没有模型贴图")
            if self.p.get("occlude", True):
                self._gbuf_texture = self._capture_texture()
            self._eps = self._depth_eps()
            self._depth_row = self._depth_measure()
        if self.name in ("IMAGE", "SCREEN_IMAGE"):
            image = np.ascontiguousarray(self.p["image"])
            h, w = image.shape[:2]
            dtype = "f1" if image.dtype == np.uint8 else "f4"
            data = image if dtype == "f1" else image.astype(np.float32)
            components = 1 if image.ndim == 2 else 4
            self._image_texture = engine.ctx.texture((w, h), components, data.tobytes(), dtype=dtype)
            self._image_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._image_texture.repeat_x = False
            self._image_texture.repeat_y = False
            self._owned.append(self._image_texture)
        if self.name == "GRADIENT":
            from ..core import ramp as ramp_mod

            table = np.clip(ramp_mod.lut(self.p.get("ramp"), 256), 0.0, 1.0).astype(np.float32)
            self._ramp_texture = engine.ctx.texture((256, 1), 4, table.tobytes(), dtype="f4")
            self._ramp_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._ramp_texture.repeat_x = False
            self._ramp_texture.repeat_y = False
            self._owned.append(self._ramp_texture)
        job = FilterJob(engine.layers, self.layer.uid, self.grid, [PLANE_FORMAT[p] for p in self.planes],
                        float(getattr(self.layer, "mask_default", 1.0)))
        pages = 0
        try:
            for plane, use in self.planes.items():
                pages += self._plane(plane, use, job)
        except Exception:
            job.cancel()
            raise
        finally:
            for texture in self._owned:
                texture.release()
            self._owned = []
        if not job.records:
            return []
        records = job.finish()
        self.stats.update(seconds=round(time.perf_counter() - started, 3), pages=pages)
        log.info("滤镜「%s」：图层「%s」%d 格，用时 %.2f 秒", self.name, self.layer.name, pages, self.stats["seconds"])
        return records

    # ---- 着色器（按引擎缓存） ----
    def _program(self, key: tuple, source):
        cache = self.engine.__dict__.setdefault("filter_programs", {})
        program = cache.get(key)
        if program is None:
            program = cache[key] = self.engine.ctx.compute_shader(source())
        return program

    def _programs(self, kind: int, fmt: str) -> tuple:
        fetch = glsl_fetch(fmt, MAX_ARRAYS[fmt], self.engine.pools.shift)
        gather = self._program(("gather", kind), lambda: _gather_shader(kind, fetch, fmt))
        stroke_fetch = "" if fmt == "rg16f" else glsl_fetch("rg16f", MAX_ARRAYS["rg16f"], self.engine.pools.shift)
        sel_fetch = "" if fmt == "r8" else glsl_fetch("r8", MAX_ARRAYS["r8"], self.engine.pools.shift)
        scatter = self._program(("scatter", kind), lambda: _scatter_shader(kind, fetch, fmt, stroke_fetch, sel_fetch))
        passes = self._program(("pass",), lambda: _PASS_SHADER)
        return gather, scatter, passes

    # ---- 每种滤镜怎么做 ----
    def _spec(self, kind: int, levels: int) -> dict:
        """level：在第几级算；halo：临时贴图四周留几个纹素（那一级的）；passes：[(种类, p0, p1)]；
        scatter：(写回方式, p0, p1)；gather：要不要取图；spread：会往外扩多远（第 0 级纹素，None 表示铺满整张）。"""
        p = self.p
        name = self.name
        top = levels - 1

        def level_for(reach: float, target: float) -> int:
            if reach <= target:
                return 0
            return int(min(top, max(0, math.ceil(math.log2(reach / target)))))

        spec = {"level": 0, "halo": 2, "passes": [], "scatter": (S_REPLACE, (0, 0, 0, 0), (0, 0, 0, 0)),
                "gather": True, "spread": 0.0}
        if name in ("GAUSSIAN_BLUR", "UNSHARP_MASK", "HIGH_PASS", "SHARPEN"):
            sigma = 1.0 if name == "SHARPEN" else max(0.1, float(p.get("radius", 1.0)))
            level = level_for(sigma, 12.0)
            sk = sigma / (1 << level)
            spec.update(level=level, halo=int(math.ceil(3.0 * sk)) + 2,
                        passes=[(P_GAUSS_H, (sk, 0, 0, 0), (0, 0, 0, 0)), (P_GAUSS_V, (sk, 0, 0, 0), (0, 0, 0, 0))])
            if name == "GAUSSIAN_BLUR":
                spec["spread"] = 3.0 * sigma
            elif name == "HIGH_PASS":
                spec["scatter"] = (S_HIGH_PASS, (0, 0, 0, 0), (0, 0, 0, 0))
            else:
                amount = float(p.get("amount", 0.5 if name == "SHARPEN" else 1.0))
                spec["scatter"] = (S_USM, (amount, float(p.get("threshold", 0.0)), 0, 0), (0, 0, 0, 0))
        elif name == "MOTION_BLUR":
            half = max(0.5, float(p.get("distance", 20.0)) * 0.5)
            level = level_for(half, 48.0)
            hk = half / (1 << level)
            angle = math.radians(float(p.get("angle", 0.0)))
            taps = int(min(160, max(3, math.ceil(2.0 * hk) + 1)))
            spec.update(level=level, halo=int(math.ceil(hk)) + 2, spread=half,
                        passes=[(P_MOTION, (math.cos(angle), math.sin(angle), hk, taps), (0, 0, 0, 0))])
        elif name == "RADIAL_BLUR":
            spec.update(radial=True, spread=None)
        elif name in ("MINIMUM", "MAXIMUM"):
            radius = max(1.0, float(p.get("radius", 1.0)))
            level = level_for(radius, 16.0)
            rk = max(1, int(round(radius / (1 << level))))
            first, second = (P_MIN_H, P_MIN_V) if name == "MINIMUM" else (P_MAX_H, P_MAX_V)
            spec.update(level=level, halo=rk + 2, spread=radius if name == "MAXIMUM" else 0.0,
                        passes=[(first, (rk, 0, 0, 0), (0, 0, 0, 0)), (second, (rk, 0, 0, 0), (0, 0, 0, 0))])
        elif name == "MEDIAN":
            radius = max(1.0, float(p.get("radius", 1.0)))
            level = level_for(radius, 4.0)
            rk = max(1, int(round(radius / (1 << level))))
            spec.update(level=level, halo=rk + 2, spread=radius, passes=[(P_MEDIAN, (rk, 0, 0, 0), (0, 0, 0, 0))])
        elif name == "SURFACE_BLUR":
            radius = max(1.0, float(p.get("radius", 5.0)))
            level = level_for(radius, 8.0)
            rk = max(1, int(round(radius / (1 << level))))
            spec.update(level=level, halo=rk + 2,
                        passes=[(P_SURFACE, (rk, float(p.get("threshold", 0.1)), 0, 0), (0, 0, 0, 0))])
        elif name == "ADD_NOISE":
            spec.update(gather=False, scatter=(S_NOISE, (float(p.get("amount", 0.1)),
                                                         1.0 if p.get("distribution", "UNIFORM") == "GAUSSIAN" else 0.0,
                                                         1.0 if p.get("monochromatic", False) else 0.0, 0),
                                               (float(p.get("seed", 0)), 0, 0, 0)))
        elif name == "EMBOSS":
            height = max(1.0, float(p.get("height", 3.0)))
            level = level_for(height, 32.0)
            spec.update(level=level, halo=int(math.ceil(height / (1 << level))) + 3,
                        scatter=(S_EMBOSS, (math.radians(float(p.get("angle", 135.0))), height,
                                            float(p.get("amount", 1.0)), 0), (0, 0, 0, 0)))
        elif name == "MOSAIC":
            cell = max(2.0, float(p.get("cell", 16.0)))
            level = int(min(top, max(0, math.floor(math.log2(cell)))))
            spec.update(level=level, halo=int(math.ceil(cell / (1 << level))) + 3, spread=cell,
                        scatter=(S_MOSAIC, (cell, 0, 0, 0), (0, 0, 0, 0)))
        elif name == "GRADIENT":
            spec.update(gather=False, spread=None,
                        scatter=(S_GRADIENT, (float(p.get("shape", 0)), 1.0 if p.get("reverse") else 0.0,
                                              float(p.get("opacity", 1.0)), 1.0 if p.get("space") == "SCREEN" else 0.0),
                                 (0, 0, 0, 0)))
        elif name in ("DODGE", "BURN"):
            spec.update(gather=False, scatter=(S_TONE, (1.0 if name == "DODGE" else -1.0, float(p.get("range", 1)),
                                                        1.0 if p.get("protect", True) else 0.0, 0), (0, 0, 0, 0)))
        elif name == "SPONGE":
            spec.update(gather=False, scatter=(S_SPONGE, (1.0 if p.get("saturate", False) else -1.0,
                                                          1.0 if p.get("vibrance", True) else 0.0, 0, 0), (0, 0, 0, 0)))
        elif name == "BLUR":
            sigma = max(0.3, float(p.get("radius", 3.0)))
            level = level_for(sigma, 12.0)
            sk = sigma / (1 << level)
            spec.update(level=level, halo=int(math.ceil(3.0 * sk)) + 2,
                        passes=[(P_GAUSS_H, (sk, 0, 0, 0), (0, 0, 0, 0)), (P_GAUSS_V, (sk, 0, 0, 0), (0, 0, 0, 0))])
        elif name in ("SMUDGE", "CLONE"):
            spec.update(halo=2)
        elif name == "HEAL":
            sigma = max(1.0, float(p.get("radius", 9.0)))
            level = level_for(sigma, 12.0)
            sk = sigma / (1 << level)
            spec.update(level=level, halo=int(math.ceil(3.0 * sk)) + 2, heal=True,
                        passes=[(P_GAUSS_H, (sk, 0, 0, 0), (0, 0, 0, 0)), (P_GAUSS_V, (sk, 0, 0, 0), (0, 0, 0, 0))],
                        scatter=(S_HEAL, (0, 0, 0, 0), (0, 0, 0, 0)))
        elif name in ("SCREEN_IMAGE", "IMAGE"):
            mask = 1.0 if np.ndim(p["image"]) == 2 else 0.0          # 只给遮罩：颜色用 color2
            combine = 3.0 if p.get("combine") == "MULTIPLY" else 0.0  # 交叉（蒙版页）：相乘，图外归 0
            screen = name == "SCREEN_IMAGE"
            spec.update(gather=False, spread=None, screen=screen, image=not screen,
                        scatter=(S_SCREEN_IMAGE if screen else S_IMAGE, (float(p.get("opacity", 1.0)), mask, combine, 0),
                                 (0, 0, 0, 0)))
        elif name == "SEL_INVERT":
            spec.update(gather=False, scatter=(S_SELECT, (SEL_INVERT, 0, 0, 0), (0, 0, 0, 0)))
        elif name in ("SEL_KEEP", "SEL_CLEAR"):
            spec.update(gather=False, needs_selection=True,
                        scatter=(S_SELECT, (SEL_KEEP if name == "SEL_KEEP" else SEL_CLEAR, 0, 0, 0), (0, 0, 0, 0)))
        elif name == "SEL_BORDER":
            # 边界：模糊到大约一半宽，取 0.5 附近（边缘）那一圈
            sigma = max(0.5, float(p.get("radius", 8.0)) / 1.35)
            level = level_for(sigma, 12.0)
            sk = sigma / (1 << level)
            spec.update(level=level, halo=int(math.ceil(3.0 * sk)) + 2, spread=3.0 * sigma,
                        passes=[(P_GAUSS_H, (sk, 0, 0, 0), (0, 0, 0, 0)), (P_GAUSS_V, (sk, 0, 0, 0), (0, 0, 0, 0))],
                        scatter=(S_SELECT, (SEL_BORDER, 0, 0, 0), (0, 0, 0, 0)))
        elif name == "CLOUDS":
            spec.update(gather=False, spread=None,
                        scatter=(S_CLOUDS, (max(1.0, float(p.get("scale", 256.0))), float(p.get("seed", 0)),
                                            float(p.get("contrast", 1.0)), float(p.get("height", 0.5))),
                                 (0, 0, 0, 0)))
        else:
            raise ValueError("没有这种滤镜：%s" % name)
        return spec

    # ---- 哪些格要重写 ----
    def _targets(self, existing: set, spec: dict) -> set:
        targets = self._base_targets(existing, spec)
        extra = self.p.get("extra_cells")
        if extra:
            grid = self.grid
            targets = set(targets) | {(int(x), int(y)) for x, y in extra if 0 <= x < grid and 0 <= y < grid}
        if spec.get("needs_selection") and self._selection is not None:
            # 只留：缺页的格按 default，是 1 的不用动；清除：是 0 的不用动。只剩选区页碰到的格要改
            if (self.name == "SEL_KEEP") == (self._selection.default > 0.5):
                targets = set(targets) & self._selection.cells()
        return targets

    def _base_targets(self, existing: set, spec: dict) -> set:
        grid = self.grid
        if "cells" in self.p:
            return {(int(x), int(y)) for x, y in self.p["cells"] if 0 <= x < grid and 0 <= y < grid}
        if spec.get("image"):
            return self._image_cells()
        if spec.get("screen"):
            if self._screen_targets is None:
                self._screen_targets = self._probe_cells(self._screen_cells())
            return set(self._screen_targets)
        if spec.get("spread") is None and not spec.get("radial"):
            return {(x, y) for y in range(grid) for x in range(grid)}
        if not existing:
            return set()
        cells = np.array(sorted(existing), np.int64).reshape(-1, 2)
        if spec.get("radial"):
            reached = self._radial_reach(cells)
        else:
            steps = min(MAX_SPREAD_PAGES, int(math.ceil(float(spec["spread"]) / PAGE)))
            reached = cells
            if steps > 0:
                offsets = np.array([(dx, dy) for dy in range(-steps, steps + 1) for dx in range(-steps, steps + 1)])
                reached = (cells[:, None, :] + offsets[None, :, :]).reshape(-1, 2)
        reached = reached[(reached[:, 0] >= 0) & (reached[:, 1] >= 0) & (reached[:, 0] < grid) & (reached[:, 1] < grid)]
        return {(int(x), int(y)) for x, y in np.unique(reached, axis=0)}

    def _image_cells(self) -> set:
        """「贴图」盖得到的格：图的透明度按格取最大，大于 0 的格。"""
        image = np.asarray(self.p["image"])
        alpha = image[..., 3] if image.ndim == 3 else image
        u0, v0, u1, v1 = (float(v) for v in self.p.get("rect", (0.0, 0.0, 1.0, 1.0)))
        grid = self.grid
        lo_x = max(0, int(math.floor(u0 * grid)))
        lo_y = max(0, int(math.floor(v0 * grid)))
        hi_x = min(grid - 1, int(math.ceil(u1 * grid)) - 1)
        hi_y = min(grid - 1, int(math.ceil(v1 * grid)) - 1)
        h, w = alpha.shape
        cells = set()
        for y in range(lo_y, hi_y + 1):
            # 这一格在图里对应的行列（往外多看一行一列，双线性取样会碰到）
            r0 = int(math.floor(((y / grid) - v0) / max(v1 - v0, 1e-9) * h)) - 1
            r1 = int(math.ceil((((y + 1) / grid) - v0) / max(v1 - v0, 1e-9) * h)) + 1
            if r1 <= 0 or r0 >= h:
                continue
            rows = alpha[max(0, r0):min(h, r1)]
            for x in range(lo_x, hi_x + 1):
                c0 = int(math.floor(((x / grid) - u0) / max(u1 - u0, 1e-9) * w)) - 1
                c1 = int(math.ceil((((x + 1) / grid) - u0) / max(u1 - u0, 1e-9) * w)) + 1
                if c1 <= 0 or c0 >= w:
                    continue
                if np.any(rows[:, max(0, c0):min(w, c1)] > 0):
                    cells.add((x, y))
        return cells

    def _radial(self) -> tuple:
        center = np.array([float(self.p.get("center_u", 0.5)), float(self.p.get("center_v", 0.5))]) * self.res0
        zoom = self.p.get("mode", "SPIN") == "ZOOM"
        amount = float(self.p.get("amount", 10.0))
        amount = min(0.95, max(0.0, amount / 100.0)) if zoom else math.radians(max(0.0, amount))
        return center, zoom, amount

    def _radial_reach(self, cells: np.ndarray) -> np.ndarray:
        """径向模糊：每格的内容沿圆弧（旋转）或沿半径往外（缩放）会跑到哪些格。"""
        center, zoom, amount = self._radial()
        corners = np.array([(0.5, 0.5), (0, 0), (1, 0), (0, 1), (1, 1)]) * PAGE
        points = (cells[:, None, :] * PAGE + corners[None]).reshape(-1, 2) - center
        out = []
        for t in np.linspace(0.0, 1.0, 17):
            if zoom:
                q = center + points / max(1.0 - amount * t, 0.05)
            else:
                a = amount * (t - 0.5)
                rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
                q = center + points @ rot.T
            out.append(np.floor(q / PAGE).astype(np.int64))
        reached = np.concatenate([cells] + out)
        offsets = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        return (reached[:, None, :] + offsets[None, :, :]).reshape(-1, 2)

    def _radial_tile(self, tx: int, ty: int, levels: int) -> tuple:
        """这一块在第几级算、留多宽、遍的参数。离中心越远模糊越长，就在越粗的一级算。"""
        center, zoom, amount = self._radial()
        x0, y0 = tx * TILE * PAGE, ty * TILE * PAGE
        x1, y1 = x0 + TILE * PAGE, y0 + TILE * PAGE
        nearest = np.array([min(max(center[0], x0), x1), min(max(center[1], y0), y1)])
        rho_min = float(np.linalg.norm(nearest - center))
        rho_max = max(float(np.linalg.norm(np.array(c) - center)) for c in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)))
        if zoom:
            reach_min, reach_max = rho_min * amount, rho_max * amount
        else:
            reach_min, reach_max = rho_min * amount * 0.5, rho_max * amount * 0.5
        level = 0
        if reach_min > 16.0:
            level = int(math.floor(math.log2(reach_min / 16.0)))
        level = min(levels - 1, level)
        while level < levels - 1 and reach_max / (1 << level) + 2 > MAX_HALO:
            level += 1
        scale = 1 << level
        halo = int(min(MAX_HALO, math.ceil(reach_max / scale) + 2))
        length = (reach_max * (1.0 if zoom else 2.0)) / scale
        taps = int(min(256, max(8, math.ceil(length * 1.5))))
        return level, halo, (center, zoom, amount, taps)

    # ---- 一种页 ----
    def _plane(self, plane: int, use: tuple, job: FilterJob) -> int:
        ctx = self.engine.ctx
        cache = self.engine.cache
        fmt = PLANE_FORMAT[plane]
        kind = PLANE_KIND[plane]
        pstore = job.store.plane(plane)
        levels = pstore.levels
        spec = self._spec(kind, len(levels))
        targets = self._targets(set(levels[0].pages), spec)
        if not targets:
            return 0
        gather, scatter, passes = self._programs(kind, fmt)
        mask_default = float(getattr(self.layer, "mask_default", 1.0))
        tiles: dict = {}
        span = TILE if spec["gather"] else WIDE_TILE
        for x, y in targets:
            tiles.setdefault((x // span, y // span), []).append((x, y))
        table0, tablek, self._mask_table, self._sel_table = self._tables()
        made = 0
        for (tx, ty) in sorted(tiles, key=lambda t: (t[1], t[0])):
            cells = sorted(tiles[(tx, ty)], key=lambda c: (c[1], c[0]))
            self._tile(plane, kind, fmt, use, spec, levels, tx, ty, cells, job, (gather, scatter, passes),
                       (table0, tablek), mask_default)
            made += len(cells)
            if made >= SERVICE_EVERY:
                # 推进一帧：前面几块生成的新页可以备份、换出，大图整层做也不会把显存页池挤满
                made = 0
                cache.service(cache.frame + 1, budget_ms=2.0)
        return len(targets)

    def _tables(self) -> tuple:
        """四张页表缓冲（第 0 级、第 k 级、笔划遮罩、选区），按格数留在引擎上反复用。"""
        size = max(1, self.grid * self.grid) * 4
        cached = self.engine.__dict__.setdefault("filter_tables", {})
        tables = cached.get(size)
        if tables is None:
            ctx = self.engine.ctx
            tables = cached[size] = tuple(ctx.buffer(reserve=size) for _ in range(4))
        tables[2].write(np.full(size // 4, -1, np.int32).tobytes())
        return tables

    def _capture_texture(self):
        """判断遮挡用的表面位置（第一行在下）：视口记下的那份（按编号对上），或参数里直接给的数组（第一行在上）。都没有时 None。"""
        capture = self.engine.__dict__.get("screen_capture")
        token = int(self.p.get("capture", 0) or 0)
        if capture is not None and token and capture[0] == token:
            return capture[1]
        array = self.p.get("gbuf")
        if array is None:
            return None
        array = np.ascontiguousarray(np.asarray(array, np.float32)[::-1])
        texture = self.engine.ctx.texture((array.shape[1], array.shape[0]), 4, array.tobytes(), dtype="f4")
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._owned.append(texture)
        return texture

    def _depth_eps(self) -> tuple:
        """位置相差多少以内算同一处：(固定的一点, 每单位深度多少)。固定的那点是位置贴图半精度的误差，
        跟深度走的是 DEPTH_TOLERANCE 个像素在那个深度上有多宽。"""
        matrix = np.asarray(self.p.get("matrix", np.identity(4).ravel()), np.float64).reshape(4, 4)
        height = max(1.0, float(self.p.get("view_size", (1.0, 1.0))[1]))
        pixel = 2.0 / max(float(np.linalg.norm(matrix[1, :3])), 1e-12) / height
        maps = self.engine.meshmaps.get(self.ts.uid)
        reach = 1.0
        if maps is not None:
            lo, hi = maps.bounds
            reach = max(float(np.abs(lo).max()), float(np.abs(hi).max()), 1e-6)
        tolerance = float(self.p.get("depth_tolerance", DEPTH_TOLERANCE))
        return reach * 2.0 ** -9, pixel * tolerance

    def _depth_measure(self) -> tuple:
        """离镜头多远（按长度算）= 这一行点乘 (位置, 1)：透视时就是 w；正交时 w 恒为 1，改用 z 那行（除掉它的缩放）。"""
        matrix = np.asarray(self.p.get("matrix", np.identity(4).ravel()), np.float64).reshape(4, 4)
        if np.linalg.norm(matrix[3, :3]) > 1e-9:
            return tuple(matrix[3])
        scale = float(np.linalg.norm(matrix[2, :3]))
        return tuple(matrix[2] / scale) if scale > 1e-12 else (0.0, 0.0, 0.0, 0.0)

    def _screen_cells(self) -> set:
        """「投到模型上」碰得到的格：这套贴图里投到屏幕上和图的矩形有重叠的三角形，它们在 UV 上占的范围（往外多算补边那么宽）。
        按三角形算，拉得再近也不会漏格。"""
        from .fill import triangles_of_set

        project = getattr(self.engine, "project", None)
        if project is None:
            return set()
        grid = self.grid
        matrix = np.asarray(self.p.get("matrix", np.identity(4).ravel()), np.float64).reshape(4, 4)
        width, height = (float(v) for v in self.p.get("view_size", (1.0, 1.0)))
        x0, y0, x1, y1 = (float(v) for v in self.p.get("rect", (0.0, 0.0, 1.0, 1.0)))
        maps = self.engine.meshmaps.get(self.ts.uid)
        pad = 0.0
        if maps is not None:
            settings = getattr(maps, "meta", {}).get("settings", {})
            pad = float(settings.get("padding", 16)) / max(1, int(maps.size))
        marks = np.zeros((grid + 1, grid + 1), np.int64)
        for obj, tris in triangles_of_set(project, self.ts):
            corners = np.asarray(obj.data.positions, np.float64).reshape(-1, 3, 3)[tris]
            clip = corners @ matrix[:2, :3].T + matrix[:2, 3]
            w = corners @ matrix[3, :3] + matrix[3, 3]
            front = w > 1e-6
            safe = np.where(front, w, 1.0)
            sx = (clip[..., 0] / safe * 0.5 + 0.5) * width
            sy = (0.5 - clip[..., 1] / safe * 0.5) * height
            overlap = (sx.max(1) >= x0) & (sx.min(1) <= x1) & (sy.max(1) >= y0) & (sy.min(1) <= y1)
            # 一部分在镜头后面的三角形投不准，算它碰得到；整个在后面的看不见
            chosen = np.where(front.all(1), overlap, front.any(1))
            if not chosen.any():
                continue
            uvs = np.asarray(obj.data.uvs, np.float64).reshape(-1, 3, 2)[tris[chosen]]
            lo = np.clip(uvs.min(1) - pad, 0.0, 1.0)
            hi = np.clip(uvs.max(1) + pad, 0.0, 1.0)
            c0 = np.minimum((lo * grid).astype(np.int64), grid - 1)
            c1 = np.minimum((hi * grid).astype(np.int64), grid - 1)
            np.add.at(marks, (c0[:, 1], c0[:, 0]), 1)
            np.add.at(marks, (c0[:, 1], c1[:, 0] + 1), -1)
            np.add.at(marks, (c1[:, 1] + 1, c0[:, 0]), -1)
            np.add.at(marks, (c1[:, 1] + 1, c1[:, 0] + 1), 1)
        covered = marks.cumsum(0).cumsum(1)[:grid, :grid] > 0
        return {(int(x), int(y)) for y, x in np.argwhere(covered)}

    def _projection_values(self) -> dict:
        """投到模型上那几个统一变量（写回、探测共用）。"""
        return {"u_res": int(self.res0), "u_view_size": tuple(float(v) for v in self.p.get("view_size", (1.0, 1.0))),
                "u_img_rect": tuple(float(v) for v in self.p.get("rect", (0.0, 0.0, 1.0, 1.0))),
                "u_eps": tuple(float(v) for v in self._eps), "u_depth_row": tuple(float(v) for v in self._depth_row),
                "u_project": (1.0 if self._gbuf_texture is not None else 0.0, 0.0, 0.0, 0.0)}

    def _probe_cells(self, cells: set) -> set:
        """粗选出来的格里，哪些真有纹素会变（投到图里不透明的地方、没被挡住）：显卡上逐纹素试一遍，只读回每格一个标记。"""
        if not cells:
            return set()
        engine = self.engine
        ctx = engine.ctx
        program = self._program(("probe",), lambda: _PROBE_SHADER)
        order = sorted(cells, key=lambda c: (c[1], c[0]))
        cell_buffer = ctx.buffer(np.asarray(order, np.int32).tobytes())
        flags = ctx.buffer(np.zeros(len(order), np.uint32).tobytes())
        try:
            unit = 0
            for name, texture in (("u_pos", self._position_texture()), ("u_img", self._image_texture),
                                  ("u_gbuf", self._gbuf_texture or self._image_texture)):
                texture.use(unit)
                if name in program:
                    program[name] = unit
                unit += 1
            matrix = np.asarray(self.p.get("matrix", np.identity(4).ravel()), np.float32).reshape(4, 4)
            program["u_grad_matrix"].write(np.ascontiguousarray(matrix.T).tobytes())
            values = dict(self._projection_values(), u_opacity=float(self.p.get("opacity", 1.0)), u_page=PAGE)
            for name, value in values.items():
                if name in program:
                    program[name] = value
            cell_buffer.bind_to_storage_buffer(1)
            flags.bind_to_storage_buffer(2)
            program.run(PAGE // 16, PAGE // 16, len(order))
            ctx.memory_barrier()
            hit = np.frombuffer(flags.read(), np.uint32)
        finally:
            cell_buffer.release()
            flags.release()
        kept = {cell for cell, flag in zip(order, hit) if flag}
        self.stats["probed"] = len(order)
        return kept

    def _position_texture(self):
        """三维渐变、投到模型上的图用的位置贴图（没烘焙过时为 None）。"""
        if self.p.get("space") != "SCREEN" and self.name != "SCREEN_IMAGE":
            return None
        maps = self.engine.meshmaps.get(self.ts.uid)
        return maps.textures.get("p") if maps is not None and getattr(maps, "textures", None) else None

    def _chain(self, chain: tuple, origin: tuple, pass_list: list, temps: tuple):
        """从 origin 起取一片到临时贴图 temps[0]，再在 temps[0]、temps[1] 之间来回跑这几遍。返回最后那张。"""
        kind, fmt, gather, passes, tablek, grid_k, res_k, side, mask_default = chain
        engine = self.engine
        ctx = engine.ctx
        current = self._temp(side, temps[0])
        bind_samplers(gather, engine.pools, fmt, 0)
        tablek.bind_to_storage_buffer(2)
        current.bind_to_image(1, read=False, write=True)
        gather["u_grid"] = int(grid_k)
        gather["u_res"] = int(res_k)
        gather["u_origin"] = (int(origin[0]), int(origin[1]))
        gather["u_size"] = (int(side), int(side))
        if "u_mask_default" in gather:
            gather["u_mask_default"] = mask_default
        groups = (side + 15) // 16
        gather.run(groups, groups, 1)
        ctx.memory_barrier()
        spare = temps[1]
        for mode, p0, p1 in pass_list:
            target = self._temp(side, spare)
            current.use(0)
            passes["u_src"] = 0
            target.bind_to_image(1, read=False, write=True)
            passes["u_size"] = (int(side), int(side))
            passes["u_mode"] = int(mode)
            passes["u_p0"] = tuple(float(v) for v in p0)
            passes["u_p1"] = tuple(float(v) for v in p1)
            if "u_key" in passes:
                passes["u_key"] = KEYS[kind]
            passes.run(groups, groups, 1)
            ctx.memory_barrier()
            spare = temps[0] if spare == temps[1] else temps[1]
            current = target
        return current

    def _temp(self, size: int, index: int):
        """第 index 张临时贴图（至少 size 边长）。留在引擎上反复用，只在不够大时换大的。"""
        temps = self.engine.__dict__.setdefault("filter_temps", {})
        texture = temps.get(index)
        if texture is None or texture.size[0] < size:
            if texture is not None:
                texture.release()
            side = max(size, 512)
            texture = self.engine.ctx.texture((side, side), 4, dtype="f2")
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
            temps[index] = texture
        return texture

    def _tile(self, plane, kind, fmt, use, spec, levels, tx, ty, cells, job, programs, tables, mask_default) -> None:
        engine = self.engine
        cache = engine.cache
        ctx = engine.ctx
        gather, scatter, passes = programs
        table0, tablek = tables
        level, halo, pass_list = spec["level"], spec["halo"], list(spec["passes"])
        if spec.get("radial"):
            level, halo, (center, zoom, amount, taps) = self._radial_tile(tx, ty, len(levels))
            scale = 1 << level
            origin_k = (tx * TILE * PAGE) // scale - halo, (ty * TILE * PAGE) // scale - halo
            ck = center / scale - 0.5 - np.array(origin_k, np.float64)
            pass_list = [(P_RADIAL, (float(ck[0]), float(ck[1]), amount, taps), (1.0 if zoom else 0.0, 0, 0, 0))]
        scale = 1 << level
        grid_k = levels[level].size
        res_k = grid_k * PAGE
        offset = self.p.get("offset", (0, 0))
        off = (int(round(float(offset[0]))) // scale, int(round(float(offset[1]))) // scale)
        origin = ((tx * TILE * PAGE) // scale - halo + off[0], (ty * TILE * PAGE) // scale - halo + off[1])
        side = max(1, (TILE * PAGE) // scale) + 2 * halo
        # 需要的页：第 0 级这几格的原页（写回时要原值），第 k 级覆盖临时贴图那一片的页
        here = levels[0]
        orig_pages = {(x, y): here.pages[(x, y)] for (x, y) in cells if (x, y) in here.pages}
        source_pages = {}
        origin_dst = (origin[0] - off[0], origin[1] - off[1])
        regions = [origin, origin_dst] if spec.get("heal") else [origin]
        if spec["gather"]:
            for region in regions:
                lo_x = max(0, region[0] // PAGE)
                lo_y = max(0, region[1] // PAGE)
                hi_x = min(grid_k - 1, (region[0] + side - 1) // PAGE)
                hi_y = min(grid_k - 1, (region[1] + side - 1) // PAGE)
                for y in range(lo_y, hi_y + 1):
                    for x in range(lo_x, hi_x + 1):
                        page = levels[level].pages.get((x, y))
                        if page is not None:
                            source_pages[(x, y)] = page
        needed = list(orig_pages.values()) + list(source_pages.values())
        cache.make_resident(needed)
        cache.pin_many(needed)
        mask_store = self.p.get("mask_store")
        selected = []
        try:
            if self._selection is not None:
                xs = np.fromiter((c[0] for c in cells), np.int64, len(cells))
                ys = np.fromiter((c[1] for c in cells), np.int64, len(cells))
                sel_slots, selected = self._selection.slots_for(xs, ys)
                table = np.full(self.grid * self.grid, -1, np.int32)
                table[ys * self.grid + xs] = sel_slots
                self._sel_table.write(table.tobytes())
            if mask_store is not None:
                mask_slots = np.full(self.grid * self.grid, -1, np.int32)
                slots = mask_store.levels[0].slots
                for x, y in cells:
                    mask_slots[y * self.grid + x] = int(slots[y, x])
                self._mask_table.write(mask_slots.tobytes())
            slots0 = np.full(self.grid * self.grid, -1, np.int32)
            for (x, y), page in orig_pages.items():
                slots0[y * self.grid + x] = page.slot
            table0.write(slots0.tobytes())
            current = extra_b = extra_c = None
            if spec["gather"]:
                slotsk = np.full(self.grid * self.grid, -1, np.int32)
                for (x, y), page in source_pages.items():
                    slotsk[y * grid_k + x] = page.slot
                tablek.write(slotsk.tobytes())
                chain = (kind, fmt, gather, passes, tablek, grid_k, res_k, side, mask_default)
                if spec.get("heal"):
                    # 来源原样、来源的低频、落笔处的低频
                    current = self._chain(chain, origin, [], (0, 0))
                    extra_b = self._chain(chain, origin, pass_list, (1, 2))
                    extra_c = self._chain(chain, origin_dst, pass_list, (3, 4))
                else:
                    current = self._chain(chain, origin, pass_list, (0, 1))
            # 写回：每格一页
            news = cache.try_new_pages(fmt, len(cells), block=True)
            jobs = np.full((len(news), 8), -1, np.int32)
            jobs[:, 0] = np.fromiter((page.slot for page in news), np.int32, len(news))
            jobs[:, 1] = [c[0] for c in cells[:len(news)]]
            jobs[:, 2] = [c[1] for c in cells[:len(news)]]
            unit = bind_samplers(scatter, engine.pools, fmt, 0)
            if fmt != "rg16f":
                unit = bind_samplers(scatter, engine.pools, "rg16f", unit)
            if fmt != "r8":
                unit = bind_samplers(scatter, engine.pools, "r8", unit)
            if current is None:
                current = self._temp(16, 0)
            current.use(unit)
            if "u_proc" in scatter:
                scatter["u_proc"] = unit
            for name, texture in (("u_proc_b", extra_b), ("u_proc_c", extra_c)):
                unit += 1
                (texture or current).use(unit)
                if name in scatter:
                    scatter[name] = unit
            unit += 1
            for name, texture in (("u_pos", self._position_texture()), ("u_ramp", self._ramp_texture),
                                  ("u_img", self._image_texture), ("u_gbuf", self._gbuf_texture)):
                (texture or current).use(unit)
                if name in scatter:
                    scatter[name] = unit
                unit += 1
            if "u_grad_matrix" in scatter:
                matrix = np.asarray(self.p.get("matrix", np.identity(4).ravel()), np.float32).reshape(4, 4)
                scatter["u_grad_matrix"].write(np.ascontiguousarray(matrix.T).tobytes())
            mode, p0, p1 = spec["scatter"]
            values = {"u_grid": int(self.grid), "u_res": int(self.res0), "u_mask_default": mask_default,
                      "u_grad_ab": tuple(float(v) for v in (*self.p.get("a", (0.0, 0.0)), *self.p.get("b", (1.0, 0.0)))),
                      "u_view_size": tuple(float(v) for v in self.p.get("view_size", (1.0, 1.0))),
                      "u_img_rect": tuple(float(v) for v in self.p.get("rect", (0.0, 0.0, 1.0, 1.0))),
                      "u_mask_on": 1 if mask_store is not None else 0,
                      "u_sel_on": 1 if self._selection is not None else 0,
                      "u_sel_mix": 1 if self._sel_mix else 0,
                      "u_sel_default": float(self._selection.default) if self._selection is not None else 1.0,
                      "u_proc_c_origin": (float(origin_dst[0]), float(origin_dst[1])),
                      "u_strength": float(self.p.get("strength", 1.0)),
                      "u_src_offset": (float(offset[0]), float(offset[1])),
                      "u_proc_size": (int(side), int(side)), "u_proc_origin": (float(origin[0]), float(origin[1])),
                      "u_scale": float(scale), "u_mode": int(mode), "u_p0": tuple(float(v) for v in p0),
                      "u_p1": tuple(float(v) for v in p1), "u_key": KEYS[kind],
                      "u_c0": tuple(float(v) for v in self.p.get("color1", self.p.get("values", (0.0, 0.0, 0.0, 1.0)))),
                      "u_c1": tuple(float(v) for v in self.p.get("color2", (1.0, 1.0, 1.0, 1.0))),
                      "u_use": (int(bool(use[0])), int(bool(use[1]) if len(use) > 1 else 1))}
            if self.name == "SCREEN_IMAGE":
                values.update({k: v for k, v in self._projection_values().items() if k != "u_res"})
            for name, value in values.items():
                if name in scatter:
                    scatter[name] = value
            table0.bind_to_storage_buffer(2)
            self._mask_table.bind_to_storage_buffer(3)
            self._sel_table.bind_to_storage_buffer(4)
            engine.ops._run_by_array(scatter, fmt, jobs)
            cache.note_new_pages(news)
            job.add(plane, [c[0] for c in cells[:len(news)]], [c[1] for c in cells[:len(news)]], news)
        finally:
            cache.pin_many(needed, -1)
            if self._selection is not None:
                self._selection.unpin(selected)
