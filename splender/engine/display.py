"""显示贴图：图层合成的结果，放在硬件稀疏贴图里，只为屏幕上看得见的页付显存。

每个纹理集三张：基础色（sRGB）、金属度/粗糙度/AO、高度。每一级切成 256 的页，按需提交。
clamp_map 记录每个 L0 格子当前最细能用到哪一级，着色器据此不去采还没准备好的层级。
"""
from __future__ import annotations

import logging
import math

import moderngl
import numpy as np

from . import glx
from .cache import PageCache
from .adjust import LUT_SIZE
from .pageops import SRGB_GLSL
from ..doc.project import ADJUST_CODE, MAX_FOLDER_DEPTH
from .pagepool import FORMATS, MAX_ARRAYS, PAGE, PoolSet, bind_samplers, glsl_fetch

MAX_PART_IDS = 16384         # 「部件」生成器能选的部件编号上限

log = logging.getLogger("splender.engine.display")

MAX_LAYERS = 64
LAYER_STRIDE = MAX_LAYERS + 1          # 每个任务多一行放笔划页（最多 4 笔同时叠加）
MAX_OVERLAYS = 4                       # 同时叠加显示的笔划数：正在画的一笔，加上还在并入图层的几笔
PARAM_DTYPE = np.dtype([("fill_color", "f4", 4), ("fill_mrh", "f4", 4), ("blend", "i4", 4), ("opacity", "f4", 4),
                        ("flags", "i4", 4), ("gen", "f4", 4), ("gen2", "f4", 4), ("gen3", "f4", 4),
                        ("adj", "f4", 4), ("adj2", "f4", 4), ("adj3", "f4", 4), ("flags2", "i4", 4),
                        ("graph", "f4", 4), ("adj4", "f4", 4)])
#: 调整种类在着色器里的编号
ADJ_DEFINES = "\n".join("#define ADJ_%s %d" % (key, code) for key, code in ADJUST_CODE.items())
MAX_GRAPHS = 6                   # 一套贴图同时能用几张节点图（每张占一个贴图单元）


class DisplaySet:
    """一个纹理集的显示贴图和每页状态。"""

    def __init__(self, ctx: moderngl.Context, caps: glx.Caps, size: int, anisotropy: float = 16.0) -> None:
        self.ctx = ctx
        self.size = size
        self.grid = size // PAGE
        self.levels = int(math.log2(self.grid)) + 1
        formats = (glx.GL_SRGB8_ALPHA8, glx.GL_RGBA8, glx.GL_R16F)
        self.sparse = bool(caps.sparse and size <= max(caps.max_sparse_size, 0))
        if self.sparse:
            for fmt in formats:
                px, py = caps.sparse_page_size(fmt)
                if px == 0 or PAGE % px or PAGE % py:
                    self.sparse = False
        make = glx.create_sparse_texture if self.sparse else glx.create_dense_texture
        aniso = anisotropy if caps.anisotropy else 1.0
        self.tex_color = make(glx.GL_SRGB8_ALPHA8, size, self.levels, anisotropy=aniso)
        self.tex_mrao = make(glx.GL_RGBA8, size, self.levels, anisotropy=aniso)
        self.tex_height = make(glx.GL_R16F, size, self.levels, anisotropy=1.0)
        self.textures = (self.tex_color, self.tex_mrao, self.tex_height)
        self.page_bytes = PAGE * PAGE * (4 + 4 + 2)
        self.fbos = [0] * self.levels
        self.level_size = [self.grid >> m for m in range(self.levels)]
        self.committed = [np.full((g, g), not self.sparse, bool) for g in self.level_size]
        self.valid = [np.zeros((g, g), bool) for g in self.level_size]      # 合成过，可以显示
        self.stale = [np.ones((g, g), bool) for g in self.level_size]       # 内容已过期，需要重新合成
        self.needed = [np.zeros((g, g), np.int32) for g in self.level_size]  # 最近一次被需要的帧
        self.base_level = max(0, self.levels - 3)                            # 这一级及更粗的常驻
        self.clamp = np.full((self.grid, self.grid), self.levels - 1, np.uint8)
        self.clamp_tex = ctx.texture((self.grid, self.grid), 1, dtype="u1")
        self.clamp_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.clamp_tex.write(self.clamp.tobytes(), alignment=1)
        self.committed_pages = 0
        if not self.sparse:
            self.committed_pages = sum(g * g for g in self.level_size)
        for level in range(self.base_level, self.levels):
            self.commit(level, np.arange(self.level_size[level] ** 2))

    # ---- 提交与撤销 ----
    def fbo(self, level: int) -> int:
        if not self.fbos[level]:
            self.fbos[level] = glx.create_framebuffer(list(self.textures), level)
        return self.fbos[level]

    def commit(self, level: int, tiles: np.ndarray) -> None:
        g = self.level_size[level]
        ty, tx = np.divmod(np.asarray(tiles, np.int64), g)
        fresh = ~self.committed[level][ty, tx]
        if self.sparse:
            for x, y in zip(tx[fresh].tolist(), ty[fresh].tolist()):
                for tex in self.textures:
                    glx.commit_pages(tex, level, x * PAGE, y * PAGE, PAGE, PAGE, True)
            glx.fn("glBindTexture")(glx.GL_TEXTURE_2D, 0)
        self.committed_pages += int(fresh.sum())
        self.committed[level][ty, tx] = True
        self.stale[level][ty, tx] = True
        self.valid[level][ty[fresh], tx[fresh]] = False

    def decommit(self, level: int, tiles: np.ndarray) -> None:
        if not self.sparse or level >= self.base_level:
            return
        g = self.level_size[level]
        ty, tx = np.divmod(np.asarray(tiles, np.int64), g)
        held = self.committed[level][ty, tx]
        for x, y in zip(tx[held].tolist(), ty[held].tolist()):
            for tex in self.textures:
                glx.commit_pages(tex, level, x * PAGE, y * PAGE, PAGE, PAGE, False)
        glx.fn("glBindTexture")(glx.GL_TEXTURE_2D, 0)
        self.committed_pages -= int(held.sum())
        self.committed[level][ty, tx] = False
        self.valid[level][ty, tx] = False
        self.stale[level][ty, tx] = True

    @property
    def committed_bytes(self) -> int:
        return self.committed_pages * self.page_bytes

    # ---- 标脏 ----
    def mark_dirty(self, level: int, tiles: np.ndarray) -> None:
        g = self.level_size[level]
        ty, tx = np.divmod(np.asarray(tiles, np.int64), g)
        self.stale[level][ty, tx] = True

    def mark_all_dirty(self) -> None:
        for level in range(self.levels):
            self.stale[level][:] = True

    # ---- 层级下限图 ----
    def update_clamp(self) -> bool:
        """按「本页和周围一圈都已就绪」重新计算每个 L0 格子能用的最细层级。"""
        result = np.full((self.grid, self.grid), self.levels - 1, np.uint8)
        for level in range(self.levels - 2, -1, -1):
            ready = self.valid[level]
            g = self.level_size[level]
            padded = np.pad(ready, 1, constant_values=True)
            ring = np.ones((g, g), bool)
            for dy in (0, 1, 2):
                for dx in (0, 1, 2):
                    ring &= padded[dy:dy + g, dx:dx + g]
            scale = 1 << level
            fine = np.repeat(np.repeat(ring, scale, axis=0), scale, axis=1)
            # 只有上一级也可用时才能继续往细走（保证三线性过滤的两级都在）
            previous_ok = result <= level + 1
            result[fine & previous_ok] = level
        if np.array_equal(result, self.clamp):
            return False
        self.clamp = result
        self.clamp_tex.write(result.tobytes(), alignment=1)
        return True

    def release(self) -> None:
        for fbo in self.fbos:
            if fbo:
                glx.delete_framebuffer(fbo)
        for tex in self.textures:
            glx.delete_texture(tex)
        self.clamp_tex.release()


_COMPOSITE_VERT = """#version 430
in vec2 in_corner;
in ivec4 in_job;                 // 格 x, 格 y, 任务序号, 0
uniform vec2 u_target;           // 目标的尺寸（纹素）
uniform vec2 u_origin;           // 目标左下角对应的格坐标
flat out ivec4 v_job;
void main(){
  vec2 p = (vec2(in_job.xy) - u_origin + in_corner) * 256.0 / u_target;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
  v_job = in_job;
}
"""


def _composite_frag(fetch: str, material: str = "") -> str:
    """material：材质节点图编出来的 material_graph 函数（空串表示没有材质节点，直接用图层堆栈）。"""
    from ..shading.glsl import LIBRARY as MATERIAL_LIBRARY

    material_block = (MATERIAL_LIBRARY + material) if material else ""
    material_call = ("  {\n    vec3 mb = ac.rgb; float mm = am.x; float mr = ar.x; float mh = ah.x;\n"
                     "    material_graph(mb, mm, mr, mh);\n"
                     "    ac.rgb = mb; am.x = mm; ar.x = mr; ah.x = mh;\n  }\n") if material else ""
    return f"""#version 430
struct LayerParams {{ vec4 fill_color; vec4 fill_mrh; ivec4 blend; vec4 opacity; ivec4 flags; vec4 gen; vec4 gen2;
                      vec4 gen3; vec4 adj; vec4 adj2; vec4 adj3; ivec4 flags2; vec4 graph; vec4 adj4; }};
layout(std430, binding=1) readonly buffer Slots {{ ivec4 slots[]; }};
layout(std430, binding=2) readonly buffer Params {{ LayerParams layers[]; }};
uniform int u_count;
uniform vec3 u_base_color; uniform vec2 u_base_mr;
uniform int u_ov_count;
uniform int u_ov_layer[{MAX_OVERLAYS}]; uniform vec3 u_ov_color[{MAX_OVERLAYS}]; uniform vec4 u_ov_values[{MAX_OVERLAYS}];
uniform ivec4 u_ov_enable[{MAX_OVERLAYS}]; uniform float u_ov_opacity[{MAX_OVERLAYS}];
uniform int u_ov_erase[{MAX_OVERLAYS}]; uniform int u_ov_target[{MAX_OVERLAYS}];
uniform ivec2 u_origin_i; uniform int u_encode;
uniform sampler2D u_map_a; uniform sampler2D u_map_n; uniform sampler2D u_map_p;
uniform sampler2D u_map_i;    // 部件编号（-1 空白）
uniform sampler2D u_idsel;    // 每个图层一行：这个部件选中了没有
// 节点图的材质输出：第 0 层基础色（sRGB），第 1 层 (金属度, 粗糙度, 高度)
uniform sampler2DArray u_graph0; uniform sampler2DArray u_graph1; uniform sampler2DArray u_graph2;
uniform sampler2DArray u_graph3; uniform sampler2DArray u_graph4; uniform sampler2DArray u_graph5;
uniform float u_graph_res[6];
uniform int u_has_maps; uniform float u_map_lod; uniform float u_level_texels; uniform float u_ao_mix;
uniform vec3 u_bmin; uniform vec3 u_bmax;
flat in ivec4 v_job;
layout(location=0) out vec4 o_color;
layout(location=1) out vec4 o_mrao;
layout(location=2) out vec4 o_height;
{fetch}
{SRGB_GLSL}
vec3 blend3(vec3 b, vec3 s, int mode){{
  switch(mode){{
    case 1: return b * s;
    case 2: return 1.0 - (1.0 - b) * (1.0 - s);
    case 3: return mix(2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s), step(0.5, b));
    case 4: return b + s;
    case 5: return b - s;
    case 6: return min(b, s);
    case 7: return max(b, s);
  }}
  return s;
}}
float blend1(float b, float s, int mode){{ return blend3(vec3(b), vec3(s), mode).x; }}
// ---- 生成器蒙版：模型贴图 + 三维噪声（按世界位置算，接缝两边连续）----
vec3 hash33(vec3 p) {{
  p = fract(p * vec3(0.1031, 0.1030, 0.0973));
  p += dot(p, p.yxz + 33.33);
  return fract((p.xxy + p.yxx) * p.zyx) * 2.0 - 1.0;
}}
float gnoise(vec3 p) {{
  vec3 i = floor(p);
  vec3 f = p - i;
  vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float n000 = dot(hash33(i), f);
  float n100 = dot(hash33(i + vec3(1.0, 0.0, 0.0)), f - vec3(1.0, 0.0, 0.0));
  float n010 = dot(hash33(i + vec3(0.0, 1.0, 0.0)), f - vec3(0.0, 1.0, 0.0));
  float n110 = dot(hash33(i + vec3(1.0, 1.0, 0.0)), f - vec3(1.0, 1.0, 0.0));
  float n001 = dot(hash33(i + vec3(0.0, 0.0, 1.0)), f - vec3(0.0, 0.0, 1.0));
  float n101 = dot(hash33(i + vec3(1.0, 0.0, 1.0)), f - vec3(1.0, 0.0, 1.0));
  float n011 = dot(hash33(i + vec3(0.0, 1.0, 1.0)), f - vec3(0.0, 1.0, 1.0));
  float n111 = dot(hash33(i + vec3(1.0, 1.0, 1.0)), f - vec3(1.0, 1.0, 1.0));
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
             mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}}

// ---- 混合（W3C 合成公式，带覆盖度：在文件夹里从「空」开始合成也对；最外层覆盖度为 1 时和原来一样）----
#define MAXD {MAX_FOLDER_DEPTH + 1}
vec4 comp3(vec4 back, vec3 s, float a, int mode) {{
  vec3 mixed = mix(s, blend3(back.rgb, s, mode), back.a);
  float ao = a + back.a * (1.0 - a);
  return vec4(ao > 0.0 ? (a * mixed + back.a * (1.0 - a) * back.rgb) / ao : vec3(0.0), ao);
}}
vec2 comp1(vec2 back, float s, float a, int mode) {{
  float mixed = mix(s, blend1(back.x, s, mode), back.y);
  float ao = a + back.y * (1.0 - a);
  return vec2(ao > 0.0 ? (a * mixed + back.y * (1.0 - a) * back.x) / ao : 0.0, ao);
}}
// ---- 调整层（颜色在 sRGB 编码下调，和常见修图软件一致）----
vec3 rgb2hsv(vec3 c) {{
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 q = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 r = mix(vec4(q.xyw, c.r), vec4(c.r, q.yzx), step(q.x, c.r));
  float dd = r.x - min(r.w, r.y);
  return vec3(abs(r.z + (r.w - r.y) / (6.0 * dd + 1e-10)), dd / (r.x + 1e-10), r.x);
}}
vec3 hsv2rgb(vec3 c) {{
  vec3 q = clamp(abs(fract(c.x + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
  return c.z * mix(vec3(1.0), q, c.y);
}}
float lift(float v, float amount) {{ return amount >= 0.0 ? mix(v, 1.0, amount) : v * (1.0 + amount); }}
{ADJ_DEFINES}
#define LUT_SIZE {LUT_SIZE}
// 调整层的查找表：每层从 flags2.z 那一段开始。第一段是按通道的函数（RGB 三列管颜色，第四列管金属度、粗糙度、高度），
// 后面接着这种调整自己的数据（色带、分类参数、颜色查找表）
layout(std430, binding=4) readonly buffer Luts {{ vec4 luts[]; }};
vec4 lut_at(int row, float x) {{
  float f = clamp(x, 0.0, 1.0) * float(LUT_SIZE - 1);
  int k = min(int(f), LUT_SIZE - 2);
  int base = row * LUT_SIZE + k;
  return mix(luts[base], luts[base + 1], f - float(k));
}}
// 三维查找表（红最快、蓝最慢），三线性插值
vec3 lut3d(int base, vec3 c, int n) {{
  vec3 p = clamp(c, 0.0, 1.0) * float(n - 1);
  ivec3 i = min(ivec3(floor(p)), ivec3(n - 2));
  vec3 f = p - vec3(i);
  int b0 = base + (i.z * n + i.y) * n + i.x;
  int b1 = b0 + n;
  int b2 = b0 + n * n;
  int b3 = b2 + n;
  vec3 c00 = mix(luts[b0].rgb, luts[b0 + 1].rgb, f.x);
  vec3 c10 = mix(luts[b1].rgb, luts[b1 + 1].rgb, f.x);
  vec3 c01 = mix(luts[b2].rgb, luts[b2 + 1].rgb, f.x);
  vec3 c11 = mix(luts[b3].rgb, luts[b3 + 1].rgb, f.x);
  return mix(mix(c00, c10, f.y), mix(c01, c11, f.y), f.z);
}}
float luma(vec3 c) {{ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }}
// 超出 0..1 的颜色往同样明暗的灰收（色相不变），和「明度」混合模式一样
vec3 clip_color(vec3 c) {{
  float l = luma(c);
  float n = min(c.r, min(c.g, c.b));
  float x = max(c.r, max(c.g, c.b));
  if (n < 0.0) c = l + (c - l) * (l / max(l - n, 1e-6));
  if (x > 1.0) c = l + (c - l) * ((1.0 - l) / max(x - l, 1e-6));
  return clamp(c, 0.0, 1.0);
}}
vec3 set_lum(vec3 c, float l) {{ return clip_color(c + (l - luma(c))); }}
// 色彩平衡：阴影、中间调、高光三段的偏移按明暗分配（三段的份额加起来是 1）
vec3 cb_weights(float l) {{
  float sh = clamp((l - 0.333) / -0.25 + 0.5, 0.0, 1.0);
  float mi = clamp((l - 0.333) / 0.25 + 0.5, 0.0, 1.0) * clamp((l + 0.333 - 1.0) / -0.25 + 0.5, 0.0, 1.0);
  float hi = clamp((l + 0.333 - 1.0) / 0.25 + 0.5, 0.0, 1.0);
  return vec3(sh, mi, hi) * 0.7;
}}
// 黑白：最亮通道对应的原色（红、绿、蓝）和最亮两个通道对应的间色（黄、青、洋红）各管一段
float bw_gray(vec3 c, vec4 P, vec4 S) {{
  float mn = min(c.r, min(c.g, c.b));
  float mx = max(c.r, max(c.g, c.b));
  float md = c.r + c.g + c.b - mn - mx;
  float wp; float ws;
  if (c.r >= c.g && c.r >= c.b) {{ wp = P.x; ws = c.g >= c.b ? S.z : S.y; }}
  else if (c.g >= c.b) {{ wp = P.y; ws = c.r >= c.b ? S.z : S.x; }}
  else {{ wp = P.z; ws = c.r >= c.g ? S.y : S.x; }}
  return clamp(mn + (md - mn) * ws + (mx - md) * wp, 0.0, 1.0);
}}
// 可选颜色：先算这个颜色属于九类颜色各多少，每类按自己的青、洋红、黄、黑改（印刷四色），改动按份额加起来
vec3 selective(vec3 c, int base, bool rel) {{
  float mn = min(c.r, min(c.g, c.b));
  float mx = max(c.r, max(c.g, c.b));
  float md = c.r + c.g + c.b - mn - mx;
  float k0 = 1.0 - mx;
  vec3 cmy0 = k0 < 0.9999 ? (1.0 - c - k0) / (1.0 - k0) : vec3(0.0);
  vec3 delta = vec3(0.0);
  for (int k = 0; k < 9; k++) {{
    float w;
    if (k == 0) w = c.r >= mx ? mx - md : 0.0;
    else if (k == 1) w = c.b <= mn ? md - mn : 0.0;
    else if (k == 2) w = c.g >= mx ? mx - md : 0.0;
    else if (k == 3) w = c.r <= mn ? md - mn : 0.0;
    else if (k == 4) w = c.b >= mx ? mx - md : 0.0;
    else if (k == 5) w = c.g <= mn ? md - mn : 0.0;
    else if (k == 6) w = max(mn - 0.5, 0.0) * 2.0;
    else if (k == 7) w = max(1.0 - (abs(mx - 0.5) + abs(mn - 0.5)), 0.0);
    else w = max(0.5 - mx, 0.0) * 2.0;
    if (w <= 0.0) continue;
    vec4 a = luts[base + k];
    vec3 cmy = clamp(rel ? cmy0 * (1.0 + a.xyz) : cmy0 + a.xyz, 0.0, 1.0);
    float kk = clamp(rel ? k0 * (1.0 + a.w) : k0 + a.w, 0.0, 1.0);
    delta += ((1.0 - cmy) * (1.0 - kk) - c) * w;
  }}
  return clamp(c + delta, 0.0, 1.0);
}}
// 调整层改颜色（sRGB 编码）。按通道的几种（亮度对比度、色阶、曲线、曝光度、反相、色调分离）全在查找表第一段
vec3 adjust3(vec3 c, LayerParams L) {{
  int type = int(L.adj3.w);
  vec4 A = L.adj; vec4 A2 = L.adj2; vec4 A3 = L.adj3; vec4 A4 = L.adj4;
  int row = L.flags2.z;
  int extra = (row + 1) * LUT_SIZE;
  if (type == ADJ_HSV) {{
    if (A4.x > 0.5) {{
      // 着色：整体换成一种色相，明暗不变
      vec3 o = set_lum(hsv2rgb(vec3(A4.y, A4.z, 1.0)), luma(clamp(c, 0.0, 1.0)));
      return vec3(lift(o.r, A.z), lift(o.g, A.z), lift(o.b, A.z));
    }}
    vec3 hsv = rgb2hsv(clamp(c, 0.0, 1.0));
    float dh = A.x; float ds = A.y; float dv = A.z;
    if (A3.x > 0.5) {{
      // 六类颜色：正中 ±15° 全量，再往外 30° 淡出；越接近灰色越不受分类调整影响
      float colored = clamp(hsv.y / 0.05, 0.0, 1.0);
      for (int k = 0; k < 6; k++) {{
        float d = abs(fract(hsv.x - float(k) / 6.0 + 0.5) - 0.5) * 360.0;
        float w = clamp((45.0 - d) / 30.0, 0.0, 1.0) * colored;
        vec4 r = luts[extra + k];
        dh += w * r.x; ds += w * r.y; dv += w * r.z;
      }}
    }}
    hsv.x = fract(hsv.x + dh);
    hsv.y = clamp(hsv.y * (1.0 + ds), 0.0, 1.0);
    hsv.z = clamp(lift(hsv.z, dv), 0.0, 1.0);
    return hsv2rgb(hsv);
  }}
  if (type == ADJ_THRESHOLD) return vec3(step(A3.z, luma(c)));
  if (type == ADJ_GRADIENT_MAP) {{
    vec4 g = lut_at(row + 1, luma(c));
    return mix(c, g.rgb, g.a);
  }}
  if (type == ADJ_SELECTIVE_COLOR) return selective(clamp(c, 0.0, 1.0), extra, A.x > 0.5);
  if (type == ADJ_COLOR_LOOKUP) {{
    if (A.z < 0.5) return c;
    vec3 x = clamp((c - A2.xyz) * A4.xyz, 0.0, 1.0);
    if (A.y < 1.5) return vec3(lut_at(row + 1, x.r).r, lut_at(row + 1, x.g).g, lut_at(row + 1, x.b).b);
    return clamp(lut3d(extra, x, int(A.x + 0.5)), 0.0, 1.0);
  }}
  if (type == ADJ_COLOR_BALANCE) {{
    vec3 w = cb_weights((max(c.r, max(c.g, c.b)) + min(c.r, min(c.g, c.b))) * 0.5);
    vec3 o = clamp(c + A.xyz * w.x + A2.xyz * w.y + A4.xyz * w.z, 0.0, 1.0);
    return A.w > 0.5 ? set_lum(o, luma(c)) : o;
  }}
  if (type == ADJ_VIBRANCE) {{
    float l = luma(c);
    float s = max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
    return clip_color(l + (c - l) * ((1.0 + A.x * (1.0 - s)) * (1.0 + A.y)));
  }}
  if (type == ADJ_PHOTO_FILTER) {{
    vec3 o = mix(c, c * A.rgb, A.w);
    return A2.x > 0.5 ? set_lum(o, luma(c)) : o;
  }}
  if (type == ADJ_BLACK_WHITE) {{
    float g = bw_gray(clamp(c, 0.0, 1.0), A, A2);
    return A.w > 0.5 ? set_lum(A4.rgb, g) : vec3(g);
  }}
  if (type == ADJ_CHANNEL_MIXER) {{
    float r = dot(c, A.xyz) + A.w;
    if (A3.x > 0.5) return vec3(clamp(r, 0.0, 1.0));
    return clamp(vec3(r, dot(c, A2.xyz) + A2.w, dot(c, A4.xyz) + A4.w), 0.0, 1.0);
  }}
  return vec3(lut_at(row, c.r).r, lut_at(row, c.g).g, lut_at(row, c.b).b);
}}
// 金属度、粗糙度、高度这种单个数：查找表第一段的第四列
float adjust1(float v, LayerParams L) {{ return lut_at(L.flags2.z, v).a; }}
// footprint：一个纹素在噪声空间里有多大。比它还细的倍频淡出，缩小看时不闪
float fbm(vec3 p, float footprint) {{
  float sum = 0.0, amp = 0.5, norm = 0.0, freq = 1.0;
  for (int o = 0; o < 6; o++) {{
    float fade = clamp(1.5 - footprint * freq * 2.0, 0.0, 1.0);
    sum += amp * fade * gnoise(p * freq);
    norm += amp;
    amp *= 0.5;
    freq *= 2.0;
  }}
  return clamp(sum / norm * 2.2, -1.0, 1.0);
}}
float g_id;                   // 这个纹素的部件编号
vec2 g_uv;                    // 这个纹素的 UV
vec4 graph_tex(int gi, vec3 p, float lod) {{
  if (gi == 0) return textureLod(u_graph0, p, lod);
  if (gi == 1) return textureLod(u_graph1, p, lod);
  if (gi == 2) return textureLod(u_graph2, p, lod);
  if (gi == 3) return textureLod(u_graph3, p, lod);
  if (gi == 4) return textureLod(u_graph4, p, lod);
  return textureLod(u_graph5, p, lod);
}}
// G：阈值、半宽、斑驳量、斑驳频率；G2：方向 xyz、反转；G3：随机偏移
float generator(int kind, vec4 G, vec4 G2, vec4 G3, vec4 ma, vec3 mn, vec3 mp, float footprint, int layer) {{
  if (kind == 8) {{
    float sel = 0.0;
    if (g_id >= 0.0) {{
      int id = int(g_id + 0.5);
      if (id < {MAX_PART_IDS}) sel = texelFetch(u_idsel, ivec2(id, layer), 0).r;
    }}
    return G2.w > 0.5 ? 1.0 - sel : sel;
  }}
  float base = 0.5;
  if (kind == 1) base = clamp(ma.g * 2.0 - 1.0, 0.0, 1.0);
  else if (kind == 2) base = clamp(1.0 - ma.g * 2.0, 0.0, 1.0);
  else if (kind == 3) base = 1.0 - ma.r;
  else if (kind == 4) base = dot(mn, G2.xyz) * 0.5 + 0.5;
  else if (kind == 5) base = 1.0 - ma.b;
  else if (kind == 6) {{
    vec3 ext = max(u_bmax - u_bmin, vec3(1e-6));
    float s = dot((mp - u_bmin) / ext, abs(G2.xyz));
    base = dot(G2.xyz, vec3(1.0)) < 0.0 ? 1.0 - s : s;
  }}
  if (G.z > 0.0) base += fbm(mp * G.w + G3.xyz, footprint * G.w) * G.z * 0.5;
  float v = smoothstep(G.x - G.y - 1e-4, G.x + G.y + 1e-4, base);
  return G2.w > 0.5 ? 1.0 - v : v;
}}

vec2 over(vec2 old, float value, float a){{
  float c = a + old.y * (1.0 - a);
  return vec2(c > 0.0 ? (value * a + old.x * old.y * (1.0 - a)) / c : 0.0, c);
}}
// ---- 每个纹素的公共数据（main 里填好，下面几个函数共用）----
ivec2 g_p;
int g_row;
float g_ov_a[{MAX_OVERLAYS}];
vec4 g_ma; vec3 g_mn; vec3 g_mp; float g_footprint;

// 图层（或文件夹）的蒙版：画的蒙版、正在画的蒙版笔划、生成器
float layer_mask(int i, LayerParams L, ivec4 s) {{
  float mask = 1.0;
  if (L.flags.z == 1) mask = s.w >= 0 ? fetch_r8(s.w, g_p).r : L.fill_mrh.w;
  for (int k = 0; k < u_ov_count; k++) {{
    if (u_ov_layer[k] == i && u_ov_target[k] == 1 && L.flags.z == 1 && g_ov_a[k] > 0.0)
      mask = mix(mask, u_ov_values[k].w, g_ov_a[k]);
  }}
  if (L.flags.w > 0) mask *= generator(L.flags.w, L.gen, L.gen2, L.gen3, g_ma, g_mn, g_mp, g_footprint, i);
  return mask;
}}

// 绘制层、填充层叠到累加器上；调整层改累加器
void apply_layer(int i, LayerParams L, inout vec4 ac, inout vec2 am, inout vec2 ar, inout vec2 ah) {{
  ivec4 s = slots[g_row + i];
  int kind = L.flags.x;
  if (kind == 3) {{
    float t = clamp(L.opacity.x * layer_mask(i, L, s), 0.0, 1.0);
    if ((L.flags.y & 1) != 0) {{
      vec3 srgb = linear_to_srgb(ac.rgb);
      ac.rgb = mix(ac.rgb, srgb_to_linear(adjust3(srgb, L)), t * L.opacity.y);
    }}
    if (((L.flags.y >> 1) & 1) != 0) am.x = mix(am.x, adjust1(am.x, L), t * L.opacity.z);
    if (((L.flags.y >> 2) & 1) != 0) ar.x = mix(ar.x, adjust1(ar.x, L), t * L.opacity.w);
    if (((L.flags.y >> 3) & 1) != 0) {{
      float hv = ah.x * 0.5 + 0.5;
      ah.x = mix(ah.x, adjust1(hv, L) * 2.0 - 1.0, t * L.fill_color.a);
    }}
    return;
  }}
  vec4 c; vec2 m; vec2 r; vec2 h;
  if (kind == 1) {{
    c = vec4(srgb_to_linear(L.fill_color.rgb), float(L.flags.y & 1));
    m = vec2(L.fill_mrh.x, float((L.flags.y >> 1) & 1));
    r = vec2(L.fill_mrh.y, float((L.flags.y >> 2) & 1));
    h = vec2(L.fill_mrh.z, float((L.flags.y >> 3) & 1));
    if (L.flags2.y > 0) {{
      // 填充内容来自节点图：按平铺次数和偏移贴到 UV 上，按这一级的纹素大小选多级
      int gi = L.flags2.y - 1;
      vec2 guv = g_uv * L.graph.x + L.graph.yz;
      float lod = log2(max(u_graph_res[gi] * L.graph.x / u_level_texels, 1e-6));
      vec4 gb = graph_tex(gi, vec3(guv, 0.0), lod);
      vec4 gm = graph_tex(gi, vec3(guv, 1.0), lod);
      c.rgb = srgb_to_linear(clamp(gb.rgb, 0.0, 1.0));
      m.x = gm.r;
      r.x = gm.g;
      h.x = clamp((gm.b - 0.5) * 2.0 * L.graph.w, -1.0, 1.0);
    }}
  }} else {{
    vec4 t = s.x >= 0 ? fetch_rgba8(s.x, g_p) : vec4(0.0);
    c = vec4(srgb_to_linear(t.rgb), t.a);
    vec4 q = s.y >= 0 ? fetch_rgba8(s.y, g_p) : vec4(0.0);
    m = q.rg; r = q.ba;
    h = s.z >= 0 ? fetch_rg16f(s.z, g_p).rg : vec2(0.0);
  }}
  float mask = 1.0;
  if (L.flags.z == 1) mask = s.w >= 0 ? fetch_r8(s.w, g_p).r : L.fill_mrh.w;
  for (int k = 0; k < u_ov_count; k++) {{
    if (u_ov_layer[k] != i) continue;
    float a = g_ov_a[k];
    if (a <= 0.0) continue;
    ivec4 en = u_ov_enable[k];
    vec4 val = u_ov_values[k];
    if (u_ov_target[k] == 1) {{
      if (L.flags.z == 1) mask = mix(mask, val.w, a);
    }} else if (u_ov_erase[k] == 1) {{
      if (en.x == 1) c.a *= (1.0 - a);
      if (en.y == 1) m.y *= (1.0 - a);
      if (en.z == 1) r.y *= (1.0 - a);
      if (en.w == 1) h.y *= (1.0 - a);
    }} else {{
      if (en.x == 1) {{
        float ca = a + c.a * (1.0 - a);
        c = vec4(ca > 0.0 ? (srgb_to_linear(u_ov_color[k]) * a + c.rgb * c.a * (1.0 - a)) / ca : vec3(0.0), ca);
      }}
      if (en.y == 1) m = over(m, val.x, a);
      if (en.z == 1) r = over(r, val.y, a);
      if (en.w == 1) h = over(h, val.z, a);
    }}
  }}
  if (L.flags.w > 0) mask *= generator(L.flags.w, L.gen, L.gen2, L.gen3, g_ma, g_mn, g_mp, g_footprint, i);
  float w = L.opacity.x * mask;
  ac = comp3(ac, c.rgb, clamp(c.a * w * L.opacity.y, 0.0, 1.0), L.blend.x);
  am = comp1(am, m.x, clamp(m.y * w * L.opacity.z, 0.0, 1.0), L.blend.y);
  ar = comp1(ar, r.x, clamp(r.y * w * L.opacity.w, 0.0, 1.0), L.blend.z);
  ah = comp1(ah, h.x, clamp(h.y * w * L.fill_color.a, 0.0, 1.0), L.blend.w);
}}

{material_block}
void main(){{
  g_p = ivec2(gl_FragCoord.xy) - (v_job.xy - u_origin_i) * 256;
  g_row = v_job.z * {LAYER_STRIDE};
  ivec4 ov_slot = slots[g_row + {MAX_LAYERS}];
  for (int k = 0; k < {MAX_OVERLAYS}; k++) {{
    int s = k < u_ov_count ? ov_slot[k] : -1;
    g_ov_a[k] = s >= 0 ? clamp(fetch_rg16f(s, g_p).r, 0.0, 1.0) * u_ov_opacity[k] : 0.0;
  }}
  // 模型贴图：每个纹素只取一次，所有生成器共用
  vec2 map_uv = (vec2(v_job.xy * 256 + g_p) + 0.5) / u_level_texels;
  g_uv = map_uv;
  g_ma = vec4(1.0, 0.5, 1.0, 0.0);
  g_mn = vec3(0.0, 1.0, 0.0);
  g_mp = vec3(0.0);
  g_id = -1.0;
  if (u_has_maps == 1) {{
    g_ma = textureLod(u_map_a, map_uv, u_map_lod);
    g_mn = textureLod(u_map_n, map_uv, u_map_lod).xyz * 2.0 - 1.0;
    g_mp = textureLod(u_map_p, map_uv, u_map_lod).xyz;
    ivec2 isz = textureSize(u_map_i, 0);
    g_id = texelFetch(u_map_i, clamp(ivec2(map_uv * vec2(isz)), ivec2(0), isz - 1), 0).r;
  }}
  g_footprint = max(length(dFdx(g_mp)), length(dFdy(g_mp)));
  vec4 ac = vec4(srgb_to_linear(u_base_color), 1.0);
  vec2 am = vec2(u_base_mr.x, 1.0);
  vec2 ar = vec2(u_base_mr.y, 1.0);
  vec2 ah = vec2(0.0, 1.0);
  // 只有一个循环（图层的着色代码只内联一份，着色器编译快）：当前这一级的累加器在寄存器里；
  // 进文件夹时把上一级压进栈、从「空」开始合成，文件夹收口时弹出上一级，把这一级按文件夹的设置叠上去
  vec4 stk_c[MAXD]; vec2 stk_m[MAXD]; vec2 stk_r[MAXD]; vec2 stk_h[MAXD];
  int d = 0;
  for (int i = 0; i < u_count; i++) {{
    LayerParams L = layers[i];
    int depth = L.flags2.x;
    if (L.flags.x == 2) {{
      if (d != depth + 1) continue;
      vec4 gc = ac; vec2 gm = am; vec2 gr = ar; vec2 gh = ah;
      d -= 1;
      ac = stk_c[d]; am = stk_m[d]; ar = stk_r[d]; ah = stk_h[d];
      if (L.opacity.x <= 0.0) continue;
      float w = L.opacity.x * layer_mask(i, L, slots[g_row + i]);
      ac = comp3(ac, gc.rgb, clamp(gc.a * w * L.opacity.y, 0.0, 1.0), L.blend.x);
      am = comp1(am, gm.x, clamp(gm.y * w * L.opacity.z, 0.0, 1.0), L.blend.y);
      ar = comp1(ar, gr.x, clamp(gr.y * w * L.opacity.w, 0.0, 1.0), L.blend.z);
      ah = comp1(ah, gh.x, clamp(gh.y * w * L.fill_color.a, 0.0, 1.0), L.blend.w);
      continue;
    }}
    if (L.opacity.x <= 0.0) continue;
    while (d < depth && d < MAXD - 1) {{
      stk_c[d] = ac; stk_m[d] = am; stk_r[d] = ar; stk_h[d] = ah;
      d += 1;
      ac = vec4(0.0); am = vec2(0.0); ar = vec2(0.0); ah = vec2(0.0);
    }}
    apply_layer(i, L, ac, am, ar, ah);
  }}
  // 结构不完整（文件夹没收口）时按正常混合收起来
  while (d > 0) {{
    d -= 1;
    ac = comp3(stk_c[d], ac.rgb, ac.a, 0);
    am = comp1(stk_m[d], am.x, am.y, 0);
    ar = comp1(stk_r[d], ar.x, ar.y, 0);
    ah = comp1(stk_h[d], ah.x, ah.y, 0);
  }}
{material_call}  vec3 bc = ac.rgb;
  o_color = vec4(u_encode == 1 ? linear_to_srgb(bc) : clamp(bc, 0.0, 1.0), 1.0);
  float ao = u_has_maps == 1 ? mix(1.0, g_ma.r, u_ao_mix) : 1.0;
  o_mrao = vec4(clamp(am.x, 0.0, 1.0), clamp(ar.x, 0.0, 1.0), ao, 1.0);
  o_height = vec4(ah.x, 0.0, 0.0, 1.0);
}}
"""


class Compositor:
    """把图层栈合成进显示贴图。一级一次绘制，页数多少都一样。"""

    def __init__(self, ctx: moderngl.Context, pools: PoolSet, cache: PageCache) -> None:
        self.ctx = ctx
        self.pools = pools
        self.cache = cache
        fetch = "\n".join(glsl_fetch(fmt, MAX_ARRAYS[fmt], pools.shift) for fmt in FORMATS)
        self._fetch = fetch
        self.program = ctx.program(vertex_shader=_COMPOSITE_VERT, fragment_shader=_composite_frag(fetch))
        quad = np.array([0, 0, 1, 0, 0, 1, 1, 1], "f4")
        self.quad = ctx.buffer(quad.tobytes())
        self.capacity = 4096
        self.job_buffer = ctx.buffer(reserve=16 * self.capacity)
        self.slot_buffer = ctx.buffer(reserve=16 * LAYER_STRIDE * self.capacity)
        self.param_buffer = ctx.buffer(reserve=PARAM_DTYPE.itemsize * MAX_LAYERS)
        self.vao = self._make_vao(self.program)
        self._base = (self.program, self.vao)
        # 材质节点：每种材质代码一个着色器（最近用过的几个留着）；参数在存储缓冲里
        self.material_programs: dict[str, tuple] = {}
        self.material_failed: set[str] = set()
        self.material_code = ""
        self.material_buffer = ctx.buffer(reserve=16 * 256)
        # 调整层的查找表（曲线、渐变映射、可选颜色），每层 LUT_SIZE 行 RGBA
        self.lut_buffer = ctx.buffer(reserve=16 * LUT_SIZE)
        self._lut_key = None
        self.stat_pages = 0
        # 没有烘焙模型贴图时用的 1×1 贴图（不遮蔽、平、厚、朝上、原点）
        self._default_maps = {
            "a": ctx.texture((1, 1), 4, bytes([255, 128, 255, 0]), dtype="f1"),
            "n": ctx.texture((1, 1), 4, bytes([128, 255, 128, 0]), dtype="f1"),
            "p": ctx.texture((1, 1), 4, np.zeros(4, np.float16).tobytes(), dtype="f2"),
            "i": ctx.texture((1, 1), 1, np.full(1, -1.0, np.float32).tobytes(), dtype="f4"),
        }
        self._default_maps["i"].filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._default_idsel = ctx.texture((1, 1), 1, bytes([0]), dtype="f1")
        self.idsel = None
        self._default_graph = ctx.texture_array((1, 1, 2), 4, np.array([[128, 128, 128, 255], [0, 128, 128, 255]],
                                                                       np.uint8).tobytes(), dtype="f1")
        self.graph_textures: list = []
        self.maps = None
        self.ao_mix = 0.0

    def _make_vao(self, program):
        return self.ctx.vertex_array(program, [(self.quad, "2f", "in_corner"), (self.job_buffer, "4i /i", "in_job")])

    def set_material(self, compiled, values) -> bool:
        """换成这套贴图的材质节点（compiled 为 None 时直接用图层堆栈）。要在设其它参数之前调用。
        返回材质节点有没有生效（编译失败时退回图层堆栈）。"""
        code = compiled.code if compiled is not None else ""
        if code in self.material_failed:
            code = ""
        entry = None
        if code:
            entry = self.material_programs.pop(code, None)
            if entry is None:
                try:
                    program = self.ctx.program(vertex_shader=_COMPOSITE_VERT,
                                               fragment_shader=_composite_frag(self._fetch, code))
                except Exception:  # noqa: BLE001
                    log.exception("材质节点编译失败，先用图层堆栈的结果")
                    self.material_failed.add(code)
                    code = ""
                else:
                    entry = (program, self._make_vao(program))
            if entry is not None:
                self.material_programs[code] = entry          # 放到最后：最近用过
                while len(self.material_programs) > 8:
                    old_code = next(iter(self.material_programs))
                    old = self.material_programs.pop(old_code)
                    old[1].release()
                    old[0].release()
        self.program, self.vao = entry if entry is not None else self._base
        self.material_code = code
        if code and values is not None:
            data = np.ascontiguousarray(values, np.float32).tobytes()
            if self.material_buffer.size < len(data):
                self.material_buffer.orphan(len(data))
            self.material_buffer.write(data)
        return bool(code)

    def set_layers(self, params: np.ndarray, count: int, base_color, base_mr) -> None:
        self.param_buffer.write(params.tobytes())
        self.program["u_count"] = int(count)
        if "u_flat" in self.program:
            self.program["u_flat"] = 0 if bool(np.any(params["flags"][:count, 0] == 2)) else 1
        self.program["u_base_color"] = tuple(float(v) for v in base_color)
        self.program["u_base_mr"] = tuple(float(v) for v in base_mr)

    def set_luts(self, table, key=None) -> None:
        """当前纹理集调整层的查找表（形状 (行数, 4)，或 None）。key 没变时不重新上传。"""
        if table is None or len(table) == 0:
            return
        if key is not None and key == self._lut_key:
            return
        data = np.ascontiguousarray(table, np.float32).tobytes()
        if self.lut_buffer.size < len(data):
            self.lut_buffer.orphan(len(data))
        self.lut_buffer.write(data)
        self._lut_key = key

    def set_maps(self, maps, ao_mix: float = 0.0) -> None:
        """当前纹理集的模型贴图（MeshMapSet 或 None），和视口里叠多少遮蔽。"""
        self.maps = maps if maps is not None and getattr(maps, "textures", None) else None
        self.ao_mix = float(ao_mix) if self.maps is not None else 0.0
        program = self.program
        if "u_has_maps" in program:
            program["u_has_maps"] = 1 if self.maps is not None else 0
        if "u_ao_mix" in program:
            program["u_ao_mix"] = self.ao_mix
        lo, hi = maps.bounds if self.maps is not None else ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        if "u_bmin" in program:
            program["u_bmin"] = tuple(float(v) for v in lo)
        if "u_bmax" in program:
            program["u_bmax"] = tuple(float(v) for v in hi)

    def set_graphs(self, textures: list) -> None:
        """当前纹理集用到的节点图材质输出（两层贴图数组），按图层参数里的序号对应。"""
        self.graph_textures = list(textures)[:MAX_GRAPHS]

    def set_selection(self, texture) -> None:
        """当前纹理集的部件选择表（MAX_PART_IDS × MAX_LAYERS 的 R8 贴图，或 None）。"""
        self.idsel = texture

    def set_overlays(self, overlays: list) -> None:
        """按先后顺序叠加显示的笔划。每项：(图层序号, 颜色, 数值, 启用, 不透明度, 擦除, 画蒙版)。"""
        count = min(len(overlays), MAX_OVERLAYS)
        layer = np.full(MAX_OVERLAYS, -1, np.int32)
        color = np.zeros((MAX_OVERLAYS, 3), np.float32)
        values = np.zeros((MAX_OVERLAYS, 4), np.float32)
        enable = np.zeros((MAX_OVERLAYS, 4), np.int32)
        opacity = np.zeros(MAX_OVERLAYS, np.float32)
        erase = np.zeros(MAX_OVERLAYS, np.int32)
        target = np.zeros(MAX_OVERLAYS, np.int32)
        for k, (index, c, v, e, o, er, t) in enumerate(overlays[:count]):
            layer[k] = int(index)
            color[k] = c
            values[k] = v
            enable[k] = e
            opacity[k] = float(o)
            erase[k] = int(er)
            target[k] = int(t)
        self._uniform("u_ov_count", np.int32(count))
        for name, data in (("u_ov_layer", layer), ("u_ov_color", color), ("u_ov_values", values),
                           ("u_ov_enable", enable), ("u_ov_opacity", opacity), ("u_ov_erase", erase),
                           ("u_ov_target", target)):
            self._uniform(name, data)

    def _uniform(self, name: str, data) -> None:
        program = self.program
        member = program[name] if name in program else (program[name + "[0]"] if (name + "[0]") in program else None)
        if member is None:
            return
        data = np.asarray(data)
        length = max(1, int(getattr(member, "array_length", 1)))
        if data.ndim >= 1 and data.shape[0] == MAX_OVERLAYS and length < MAX_OVERLAYS:
            data = data[:length]
        member.write(np.ascontiguousarray(data).tobytes())

    def compose(self, display: DisplaySet, level: int, tx: np.ndarray, ty: np.ndarray, table: np.ndarray) -> None:
        """table 形状 (n, LAYER_STRIDE, 4)：每个任务每个图层的 颜色/金属粗糙/高度/蒙版 槽号，最后一行第 0 项是笔划页。"""
        count = len(tx)
        if count == 0:
            return
        self._ensure_capacity(count)
        size = display.level_size[level] * PAGE
        glx.bind_framebuffer(display.fbo(level), size, size, 3)
        glx.enable(glx.GL_FRAMEBUFFER_SRGB, True)
        self._draw(count, tx, ty, table, (size, size), (0, 0), False, size)
        glx.enable(glx.GL_FRAMEBUFFER_SRGB, False)
        display.stale[level][ty, tx] = False
        display.valid[level][ty, tx] = True
        self.stat_pages += count

    def compose_to(self, fbo, size: tuple, origin: tuple, tx: np.ndarray, ty: np.ndarray, table: np.ndarray,
                   srgb_encode: bool = True, level_texels: int = 0) -> None:
        """合成到任意目标（导出用）：fbo 是带三个颜色附件的帧缓冲，origin 是它左下角对应的格坐标。
        level_texels：这一级整张图的边长（纹素），模型贴图按它换算 UV。"""
        count = len(tx)
        if count == 0:
            return
        self._ensure_capacity(count)
        fbo.use()
        self.ctx.viewport = (0, 0, int(size[0]), int(size[1]))
        self._draw(count, tx, ty, table, size, origin, srgb_encode, level_texels or int(size[0]))

    def _ensure_capacity(self, count: int) -> None:
        if count > self.capacity:
            self.capacity = int(count * 1.5)
            self.job_buffer.orphan(16 * self.capacity)
            self.slot_buffer.orphan(16 * LAYER_STRIDE * self.capacity)
            base_program, base_vao = self._base
            base_vao.release()
            self._base = (base_program, self._make_vao(base_program))
            for code, (program, vao) in list(self.material_programs.items()):
                vao.release()
                self.material_programs[code] = (program, self._make_vao(program))
            current = self.material_programs.get(self.material_code) if self.material_code else None
            self.program, self.vao = current if current is not None else self._base

    def _draw(self, count: int, tx, ty, table, size: tuple, origin: tuple, encode: bool, level_texels: int) -> None:
        jobs = np.zeros((count, 4), np.int32)
        jobs[:, 0] = tx
        jobs[:, 1] = ty
        jobs[:, 2] = np.arange(count)
        self.job_buffer.write(jobs.tobytes())
        self.slot_buffer.write(np.ascontiguousarray(table, np.int32).tobytes())
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self.slot_buffer.bind_to_storage_buffer(1)
        self.param_buffer.bind_to_storage_buffer(2)
        self.lut_buffer.bind_to_storage_buffer(4)
        if self.material_code:
            self.material_buffer.bind_to_storage_buffer(3)
        unit = 0
        for fmt in FORMATS:
            unit = bind_samplers(self.program, self.pools, fmt, unit)
        program = self.program
        textures = self.maps.textures if self.maps is not None else self._default_maps
        for key in ("a", "n", "p", "i"):
            name = "u_map_" + key
            if name in program:
                (textures.get(key) or self._default_maps[key]).use(unit)
                program[name] = unit
                unit += 1
        if "u_idsel" in program:
            (self.idsel or self._default_idsel).use(unit)
            program["u_idsel"] = unit
            unit += 1
        if "u_graph0" in program:
            sizes = []
            for index in range(MAX_GRAPHS):
                texture = self.graph_textures[index] if index < len(self.graph_textures) else None
                (texture or self._default_graph).use(unit)
                program["u_graph%d" % index] = unit
                unit += 1
                sizes.append(float(texture.size[0]) if texture is not None else 1.0)
            program["u_graph_res"] = sizes
        if "u_level_texels" in program:
            program["u_level_texels"] = float(level_texels)
        if "u_map_lod" in program:
            lod = math.log2(self.maps.size / max(1.0, float(level_texels))) if self.maps is not None else 0.0
            program["u_map_lod"] = float(max(0.0, lod))
        program["u_target"] = (float(size[0]), float(size[1]))
        program["u_origin"] = (float(origin[0]), float(origin[1]))
        program["u_origin_i"] = (int(origin[0]), int(origin[1]))
        program["u_encode"] = int(encode)
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=count)

    def release(self) -> None:
        for program, vao in self.material_programs.values():
            vao.release()
            program.release()
        self.material_programs.clear()
        base_program, base_vao = self._base
        for item in (base_vao, self.quad, self.job_buffer, self.slot_buffer, self.param_buffer, base_program,
                     self.material_buffer, self.lut_buffer, self._default_idsel, self._default_graph,
                     *self._default_maps.values()):
            item.release()
