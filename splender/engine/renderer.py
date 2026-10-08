"""视口渲染：材质预览（PBR + 环境光）、实体、单通道查看；几何缓冲；页面需求反馈。

每个视口画进自己的贴图，界面控件只负责把这张贴图贴到屏幕上。
"""
from __future__ import annotations

import logging
import math

import moderngl
import numpy as np

from . import glx, ibl
from .camera import Camera
from .display import DisplaySet
from .meshgpu import MeshGPU, mesh_layout

log = logging.getLogger("splender.engine.renderer")

MODE_INDEX = {"SOLID": 0, "MATERIAL": 1, "CHANNEL": 2, "WIREFRAME": 0}

_WIRE_VERT = """#version 430
in vec3 in_pos;
uniform mat4 u_view_proj;
uniform mat4 u_model;
uniform float u_bias;          // 往相机方向挪一点，线压在面上不打架
void main(){
  gl_Position = u_view_proj * (u_model * vec4(in_pos, 1.0));
  gl_Position.z -= u_bias * gl_Position.w;
}
"""

_WIRE_FRAG = """#version 430
uniform vec4 u_color;
out vec4 o_color;
void main(){ o_color = u_color; }
"""

WIRE_UNSELECTED = (0.0, 0.0, 0.0)
WIRE_SELECTED = (0.93, 0.34, 0.0)
WIRE_ACTIVE = (1.0, 0.67, 0.25)
CHANNEL_INDEX = {"basecolor": 0, "metallic": 1, "roughness": 2, "height": 3}
TRANSFORM_INDEX = {"STANDARD": 0, "FILMIC": 1, "AGX": 2}
SOLID_INDEX = {"STUDIO": 0, "MATCAP": 1, "FLAT": 2}

_VERT = """#version 430
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
uniform mat4 u_view_proj;
uniform mat4 u_model;          // 拖动变换时实时显示用；平时是单位矩阵（顶点已经是世界坐标）
uniform mat3 u_nmat;
out vec3 v_pos; out vec3 v_nrm; out vec2 v_uv;
void main(){
  vec4 wp = u_model * vec4(in_pos, 1.0);
  v_pos = wp.xyz; v_nrm = u_nmat * in_nrm; v_uv = in_uv;
  gl_Position = u_view_proj * wp;
}
"""

_IDENTITY4 = np.identity(4, np.float32).T.tobytes()
_IDENTITY3 = np.identity(3, np.float32).T.tobytes()


def set_model(program, mesh) -> None:
    """模型矩阵和法线矩阵写进着色器（网格没有显示矩阵时用单位矩阵）。"""
    matrix = getattr(mesh, "model", None)
    if "u_model" in program:
        if matrix is None:
            program["u_model"].write(_IDENTITY4)
        else:
            program["u_model"].write(np.asarray(matrix, np.float32).T.tobytes())
    if "u_nmat" in program:
        if matrix is None:
            program["u_nmat"].write(_IDENTITY3)
        else:
            normal = np.linalg.inv(np.asarray(matrix, np.float64)[:3, :3]).T
            program["u_nmat"].write(normal.astype(np.float32).T.tobytes())

_OUTLINE_VERT = """#version 430
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_OUTLINE_FRAG = """#version 430
uniform sampler2D u_ids;      // 几何缓冲的法线附件，w 是物体编号（0 是背景）
uniform sampler2D u_state;    // 物体编号 → 0 没选、0.5 选中、1 当前物体
uniform int u_width;
uniform vec4 u_selected; uniform vec4 u_active;
out vec4 o_color;
float state_of(float id) {
  if (id < 0.5) return 0.0;
  return texelFetch(u_state, ivec2(int(id + 0.5), 0), 0).r;
}
void main() {
  ivec2 size = textureSize(u_ids, 0);
  ivec2 p = ivec2(gl_FragCoord.xy);
  float id = texelFetch(u_ids, p, 0).w;
  float best = 0.0;
  for (int dy = -u_width; dy <= u_width; dy++) {
    for (int dx = -u_width; dx <= u_width; dx++) {
      if (dx * dx + dy * dy > u_width * u_width) continue;
      ivec2 q = clamp(p + ivec2(dx, dy), ivec2(0), size - 1);
      float other = texelFetch(u_ids, q, 0).w;
      if (abs(other - id) > 0.5) best = max(best, state_of(other));
    }
  }
  if (best <= 0.0) discard;
  o_color = best > 0.75 ? u_active : u_selected;
}
"""

_VT_GLSL = """
uniform sampler2D u_color; uniform sampler2D u_mrao; uniform sampler2D u_height; uniform usampler2D u_clamp;
uniform float u_tex_size; uniform float u_grid; uniform float u_aniso; uniform int u_textured;
// 把导数放大到「已准备好的最细层级」，保留各向异性比例
float vt_scale(vec2 uv, vec2 dx, vec2 dy){
  float min_lod = float(texelFetch(u_clamp, ivec2(clamp(uv, 0.0, 0.999999) * u_grid), 0).r);
  vec2 tx = dx * u_tex_size, ty = dy * u_tex_size;
  float a = dot(tx, tx), b = dot(ty, ty);
  float major = sqrt(max(a, b)), minor = sqrt(min(a, b));
  float ratio = min(major / max(minor, 1e-8), max(u_aniso, 1.0));
  float lod = log2(max(major / ratio, 1e-8));
  return exp2(max(0.0, min_lod - lod));
}
"""

_SHADE_FRAG = """#version 430
in vec3 v_pos; in vec3 v_nrm; in vec2 v_uv;
out vec4 o_color;
uniform vec3 u_eye; uniform mat4 u_view;
uniform int u_mode; uniform int u_channel; uniform int u_solid; uniform int u_solid_textured;
uniform int u_transform; uniform float u_exposure;
uniform int u_bump; uniform float u_height_world;
uniform sampler2D u_nmap; uniform int u_has_nmap;     // 从高模烘的切线空间法线贴图
uniform vec4 u_cursor; uniform float u_cursor_hardness; uniform int u_cursor_on;
uniform float u_pixel_world;
uniform float u_alpha;          // X 光时模型半透明
""" + _VT_GLSL + """
%(ENV)s
%(PBR)s
%(TONEMAP)s
%(STUDIO)s
%(MATCAP)s
vec3 to_display(vec3 c){ return mix(1.055 * pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.4)) - 0.055, c * 12.92, lessThanEqual(c, vec3(0.0031308))); }
void main(){
  vec3 n = normalize(v_nrm);
  vec3 v = normalize(u_eye - v_pos);
  if (!gl_FrontFacing) n = -n;
  vec3 base = vec3(0.8); float metal = 0.0, rough = 0.5, ao = 1.0, height = 0.0;
  if (u_textured == 1) {
    vec2 dx = dFdx(v_uv), dy = dFdy(v_uv);
    float k = vt_scale(v_uv, dx, dy);
    vec2 gx = dx * k, gy = dy * k;
    base = textureGrad(u_color, v_uv, gx, gy).rgb;
    vec4 mr = textureGrad(u_mrao, v_uv, gx, gy);
    metal = mr.r; rough = mr.g; ao = mr.b;
    height = textureGrad(u_height, v_uv, gx, gy).r;
    if (u_has_nmap == 1 && u_mode != 2) {
      // 切线空间和烘焙器同一个约定：T 是 dP/du 去掉法线分量，B = N×T，和 dP/dv 同侧
      vec3 dp1 = dFdx(v_pos), dp2 = dFdy(v_pos);
      float det = dx.x * dy.y - dy.x * dx.y;
      if (abs(det) > 1e-20) {
        vec3 dpdu = (dp1 * dy.y - dp2 * dx.y) / det;
        vec3 dpdv = (dp2 * dx.x - dp1 * dy.x) / det;
        vec3 t = dpdu - n * dot(n, dpdu);
        float lt = length(t);
        if (lt > 1e-20) {
          t /= lt;
          vec3 b = cross(n, t);
          if (dot(b, dpdv) < 0.0) b = -b;
          vec3 tn = texture(u_nmap, v_uv).xyz * 2.0 - 1.0;
          n = normalize(tn.x * t + tn.y * b + tn.z * n);
        }
      }
    }
    if (u_bump == 1 && u_mode != 2) {
      float hx = textureGrad(u_height, v_uv + dx, gx, gy).r;
      float hy = textureGrad(u_height, v_uv + dy, gx, gy).r;
      vec3 dpdx = dFdx(v_pos), dpdy = dFdy(v_pos);
      vec3 r1 = cross(dpdy, n), r2 = cross(n, dpdx);
      float det = dot(dpdx, r1);
      vec3 grad = sign(det) * ((hx - height) * r1 + (hy - height) * r2) * u_height_world;
      n = normalize(abs(det) * n - grad);
    }
  }
  vec3 color;
  if (u_mode == 1) {
    color = tonemap(shade_pbr(n, v, base, metal, max(rough, 0.03), ao), u_transform, u_exposure);
  } else if (u_mode == 0) {
    vec3 nv = normalize(mat3(u_view) * n);
    vec3 vv = normalize(mat3(u_view) * v);
    vec3 tint = u_solid_textured == 1 ? base : vec3(0.8);
    if (u_solid == 0) color = to_display(shade_studio(nv, vv, tint));
    else if (u_solid == 1) color = to_display(shade_matcap(nv, vv, tint));
    else color = to_display(tint);
  } else {
    if (u_channel == 0) color = to_display(base);
    else if (u_channel == 1) color = vec3(metal);
    else if (u_channel == 2) color = vec3(rough);
    else color = vec3(clamp(0.5 + height * 0.5, 0.0, 1.0));
  }
  if (u_cursor_on == 1) {
    float d = length(v_pos - u_cursor.xyz);
    float w = max(fwidth(d), u_pixel_world * 0.5) * 1.3;
    float ring = 1.0 - smoothstep(0.0, w, abs(d - u_cursor.w));
    float inner = (1.0 - smoothstep(0.0, w * 0.8, abs(d - u_cursor.w * u_cursor_hardness))) * 0.35;
    float halo = 1.0 - smoothstep(w, w * 2.2, abs(d - u_cursor.w));
    color = mix(color, vec3(0.0), halo * 0.35);
    color = mix(color, vec3(1.0), max(ring, inner) * 0.9);
  }
  o_color = vec4(color, u_alpha);
}
"""

_GBUF_FRAG = """#version 430
in vec3 v_pos; in vec3 v_nrm; in vec2 v_uv;
uniform float u_set;
uniform float u_object;       // 物体编号 + 1（点选、描边用）
layout(location=0) out vec4 o_pos;
layout(location=1) out vec4 o_nrm;
void main(){ vec3 n = normalize(v_nrm); if (!gl_FrontFacing) n = -n; o_pos = vec4(v_pos, u_set); o_nrm = vec4(n, u_object); }
"""

_FEEDBACK_FRAG = """#version 430
in vec3 v_pos; in vec3 v_nrm; in vec2 v_uv;
uniform float u_tex_size; uniform float u_scale; uniform float u_set; uniform float u_max_level; uniform float u_aniso;
out vec4 o_request;
void main(){
  vec2 tx = dFdx(v_uv) * u_tex_size / u_scale, ty = dFdy(v_uv) * u_tex_size / u_scale;
  float a = dot(tx, tx), b = dot(ty, ty);
  float major = sqrt(max(a, b)), minor = sqrt(min(a, b));
  float ratio = min(major / max(minor, 1e-8), max(u_aniso, 1.0));
  float level = clamp(floor(log2(max(major / ratio, 1e-8))), 0.0, u_max_level);
  vec2 tile = floor(clamp(v_uv, 0.0, 0.999999) * u_tex_size / 256.0 / exp2(level));
  o_request = vec4(tile, level, u_set) / 255.0;
}
"""

_BG_VERT = """#version 430
in vec2 in_p; out vec2 v_p;
void main(){ v_p = in_p; gl_Position = vec4(in_p, 1.0, 1.0); }
"""

_BG_FRAG = """#version 430
in vec2 v_p; out vec4 o_color;
uniform mat4 u_inv_view_rot; uniform vec2 u_tan; uniform int u_ortho;
uniform vec3 u_bg_color; uniform float u_bg_opacity; uniform float u_bg_blur;
uniform int u_transform; uniform float u_exposure; uniform int u_mode;
%(ENV)s
%(TONEMAP)s
void main(){
  vec3 color = u_bg_color;
  if (u_mode == 1 && u_bg_opacity > 0.0) {
    vec3 dir = u_ortho == 1 ? vec3(0.0, 0.0, -1.0) : normalize(vec3(v_p * u_tan, -1.0));
    dir = normalize(mat3(u_inv_view_rot) * dir);
    vec3 env = tonemap(env_background(dir, u_bg_blur), u_transform, u_exposure);
    color = mix(color, env, u_bg_opacity);
  }
  o_color = vec4(color, 1.0);
}
"""


class View:
    """一个视口：相机、尺寸、显示设置和它的渲染目标。"""

    is_2d = False

    def __init__(self, renderer: "Renderer", shading) -> None:
        self.renderer = renderer
        self.ctx = renderer.ctx
        self.camera = Camera()
        self.shading = shading
        self.width = 0
        self.height = 0
        self.color_tex = None
        self.fbo = None
        self.msaa_fbo = None
        self.gbuf_fbo = None
        self.gbuf_valid = False
        self.feedback_fbo = None
        self.feedback_pending = False
        self.feedback_buffer = None   # 页面需求图异步读回到这里
        self.feedback_fence = None
        self.dirty = True
        self.cursor = None            # (位置, 法线, 世界半径, 硬度)
        self.symmetry = (0, 0, 0)
        self.frame_rendered = 0
        self.serial = 0               # 每次重画加一，界面据此判断要不要重贴
        self._resources: list = []

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)

    def resize(self, width: int, height: int) -> None:
        width, height = max(8, int(width)), max(8, int(height))
        if (width, height) == (self.width, self.height) and self.fbo is not None:
            return
        self._release_targets()
        ctx = self.ctx
        self.width, self.height = width, height
        size = (width, height)
        samples = self.renderer.samples

        def keep(item):
            self._resources.append(item)
            return item

        self.color_tex = keep(ctx.texture(size, 4, dtype="f1"))
        self.color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        depth = keep(ctx.depth_renderbuffer(size))
        self.fbo = keep(ctx.framebuffer([self.color_tex], depth))
        if samples > 1:
            self.msaa_fbo = keep(ctx.framebuffer([keep(ctx.renderbuffer(size, 4, samples=samples))],
                                                 keep(ctx.depth_renderbuffer(size, samples=samples))))
        self.gbuf_pos = keep(ctx.texture(size, 4, dtype="f4"))
        self.gbuf_nrm = keep(ctx.texture(size, 4, dtype="f2"))
        self.gbuf_fbo = keep(ctx.framebuffer([self.gbuf_pos, self.gbuf_nrm], keep(ctx.depth_renderbuffer(size))))
        div = max(2, self.renderer.feedback_divisor)
        self.feedback_size = (max(8, width // div), max(8, height // div))
        self.feedback_tex = keep(ctx.texture(self.feedback_size, 4, dtype="f1"))
        self.feedback_fbo = keep(ctx.framebuffer([self.feedback_tex], keep(ctx.depth_renderbuffer(self.feedback_size))))
        self.feedback_buffer = keep(ctx.buffer(reserve=self.feedback_size[0] * self.feedback_size[1] * 4))
        if self.feedback_fence is not None:
            self.feedback_fence.release()
            self.feedback_fence = None
        self.gbuf_valid = False
        self.feedback_pending = False
        self.dirty = True

    def invalidate_camera(self) -> None:
        self.gbuf_valid = False
        self.dirty = True

    # ---- 几何缓冲查询 ----
    def read_surface(self, x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
        """读一块几何缓冲。坐标是视口像素（原点左上，含端点）。返回 (位置+纹理集号 [h,w,4], 法线 [h,w,3])，行序自上而下。"""
        self.renderer.ensure_gbuffer(self)
        x0, x1 = max(0, min(x0, x1)), min(self.width - 1, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(self.height - 1, max(y0, y1))
        w, h = x1 - x0 + 1, y1 - y0 + 1
        viewport = (x0, self.height - 1 - y1, w, h)
        pos = np.frombuffer(self.gbuf_fbo.read(viewport=viewport, components=4, attachment=0, dtype="f4", alignment=1),
                            np.float32).reshape(h, w, 4)[::-1]
        nrm = np.frombuffer(self.gbuf_fbo.read(viewport=viewport, components=4, attachment=1, dtype="f2", alignment=1),
                            np.float16).reshape(h, w, 4)[::-1, :, :3].astype(np.float32)
        return pos, nrm

    def surface_at(self, x: float, y: float):
        """像素 (x, y) 处的表面：(位置, 法线, 纹理集序号) 或 None。"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None
        pos, nrm = self.read_surface(int(x), int(y), int(x), int(y))
        if pos[0, 0, 3] < 0.5:
            return None
        return pos[0, 0, :3].astype(np.float64), nrm[0, 0].astype(np.float64), int(round(float(pos[0, 0, 3]))) - 1

    def _release_targets(self) -> None:
        for item in reversed(self._resources):
            try:
                item.release()
            except Exception:  # noqa: BLE001
                pass
        self._resources.clear()
        self.fbo = self.msaa_fbo = self.gbuf_fbo = self.feedback_fbo = None

    def release(self) -> None:
        self._release_targets()


class Renderer:
    def __init__(self, ctx: moderngl.Context, *, samples: int = 4, anisotropy: float = 16.0,
                 feedback_divisor: int = 8) -> None:
        self.ctx = ctx
        self.samples = max(1, min(int(samples), ctx.max_samples))
        self.anisotropy = anisotropy
        self.feedback_divisor = feedback_divisor
        parts = {"ENV": ibl.GLSL_ENV, "PBR": ibl.GLSL_PBR, "TONEMAP": ibl.GLSL_TONEMAP, "STUDIO": ibl.GLSL_STUDIO,
                 "MATCAP": ibl.GLSL_MATCAP}
        self.shade = ctx.program(vertex_shader=_VERT, fragment_shader=_SHADE_FRAG % parts)
        self.gbuffer = ctx.program(vertex_shader=_VERT, fragment_shader=_GBUF_FRAG)
        self.feedback = ctx.program(vertex_shader=_VERT, fragment_shader=_FEEDBACK_FRAG)
        self.outline = ctx.program(vertex_shader=_OUTLINE_VERT, fragment_shader=_OUTLINE_FRAG)
        self.wire = ctx.program(vertex_shader=_WIRE_VERT, fragment_shader=_WIRE_FRAG)
        self._wire_vaos: dict[int, moderngl.VertexArray] = {}
        self._outline_vao = ctx.vertex_array(self.outline, [])
        self._state_tex = ctx.texture((2048, 1), 1, dtype="f1")
        self._state_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.selection_states: dict[int, float] = {}   # 物体编号 → 0.5 选中 / 1.0 当前（引擎每帧设）
        self._state_key = None
        self.outline_width = 2
        self.overlays = None                           # engine.overlays.Overlays（引擎设置）
        self.background = ctx.program(vertex_shader=_BG_VERT, fragment_shader=_BG_FRAG % parts)
        self._bg_vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        self._bg_vao = ctx.vertex_array(self.background, [(self._bg_vbo, "2f", "in_p")])
        self._vaos: dict[tuple, moderngl.VertexArray] = {}
        self.environment = ibl.Environment(ctx)
        self._env_name = None
        self._matcap_name = None
        self._matcap_tex = None
        self.hidden: set[int] = set()              # 不画的模型（id(MeshGPU)），例如正在雕刻、由雕刻引擎自己画的
        self.meshmaps: dict = {}                   # 纹理集 uid -> MeshMapSet（引擎设置，用其中的法线贴图）
        self.extra_draws: list = []                # 额外的绘制：fn(renderer, view, "color" 或 "gbuffer")
        self._studio_name = None
        self._studio_data = None
        self._display_wrappers: dict[int, tuple] = {}
        self.bg_color = (0.21, 0.21, 0.22)
        self.scene: list[tuple] = []          # [(MeshGPU, 材质序号, DisplaySet | None, TextureSet | None, 集合序号)]
        self._white = ctx.texture((1, 1), 4, data=bytes([204, 204, 204, 255]))
        self._clamp0 = ctx.texture((1, 1), 1, data=bytes([0]), dtype="u1")

    # ---- 资源 ----
    def _vao(self, program, mesh: MeshGPU, material: int) -> moderngl.VertexArray | None:
        ibo = mesh.material_ibo[material] if material < len(mesh.material_ibo) else None
        if ibo is None:
            return None
        key = (id(program), id(mesh), material)
        vao = self._vaos.get(key)
        if vao is None:
            vao = self.ctx.vertex_array(program, [mesh_layout(program, mesh.vbo)], index_buffer=ibo,
                                        index_element_size=4)
            self._vaos[key] = vao
        return vao

    def forget_mesh(self, mesh: MeshGPU) -> None:
        for key in [k for k in self._vaos if k[1] == id(mesh)]:
            self._vaos.pop(key).release()
        vao = self._wire_vaos.pop(id(mesh), None)
        if vao is not None:
            vao.release()

    @staticmethod
    def _skip(view, mesh) -> bool:
        """局部视图里只画选进来的物体。"""
        local = getattr(view, "local_uids", None)
        return local is not None and getattr(mesh, "object_uid", 0) not in local

    def _wire_vao(self, mesh: MeshGPU):
        vao = self._wire_vaos.get(id(mesh))
        if vao is None:
            ibo = mesh.edge_buffer()
            if ibo is None:
                return None
            vao = self.ctx.vertex_array(self.wire, [mesh_layout(self.wire, mesh.vbo)], index_buffer=ibo,
                                        index_element_size=4)
            self._wire_vaos[id(mesh)] = vao
        return vao

    def draw_wires(self, view, alpha: float, depth: bool, colored: bool = True) -> None:
        """每个模型的边。colored：选中的画橙色（当前物体亮一些），没选中的黑色。"""
        ctx = self.ctx
        program = self.wire
        program["u_view_proj"].write(view.camera.view_projection(view.aspect).astype("f4").T.tobytes())
        program["u_bias"] = 2e-4 if depth else 0.0
        if depth:
            ctx.enable(moderngl.DEPTH_TEST)
        else:
            ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE, moderngl.ONE
        ctx.depth_mask = False
        drawn = set()
        for mesh, _material, _display, _ts, _set_index in self.scene:
            if id(mesh) in drawn or id(mesh) in self.hidden or self._skip(view, mesh):
                continue
            drawn.add(id(mesh))
            vao = self._wire_vao(mesh)
            if vao is None:
                continue
            state = self.selection_states.get(getattr(mesh, "pick_id", 0), 0.0) if colored else 0.0
            rgb = WIRE_ACTIVE if state > 0.75 else (WIRE_SELECTED if state > 0.25 else WIRE_UNSELECTED)
            program["u_color"] = (*rgb, float(alpha))
            set_model(program, mesh)
            vao.render(moderngl.LINES)
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

    def _wrap(self, display: DisplaySet) -> tuple:
        found = self._display_wrappers.get(id(display))
        if found is None:
            size = (display.size, display.size)
            found = (self.ctx.external_texture(display.tex_color, size, 4, 0, "f1"),
                     self.ctx.external_texture(display.tex_mrao, size, 4, 0, "f1"),
                     self.ctx.external_texture(display.tex_height, size, 1, 0, "f2"))
            self._display_wrappers[id(display)] = found
        return found

    def forget_display(self, display: DisplaySet) -> None:
        self._display_wrappers.pop(id(display), None)

    def _prepare_lighting(self, shading) -> None:
        if shading.mode == "MATERIAL" and shading.hdri != self._env_name:
            try:
                self.environment.set_image(ibl.load_hdri(shading.hdri))
                self._env_name = shading.hdri
            except Exception:  # noqa: BLE001
                log.exception("环境 %s 加载失败", shading.hdri)
                self._env_name = shading.hdri
        if self._env_name is None:
            names = ibl.list_hdris()
            if names:
                self.environment.set_image(ibl.load_hdri(names[0]))
            self._env_name = names[0] if names else ""
        if shading.matcap != self._matcap_name:
            try:
                if self._matcap_tex is not None:
                    self._matcap_tex.release()
                self._matcap_tex = ibl.create_matcap_texture(self.ctx, ibl.load_matcap(shading.matcap))
            except Exception:  # noqa: BLE001
                log.exception("Matcap %s 加载失败", shading.matcap)
            self._matcap_name = shading.matcap
        if shading.studio_light != self._studio_name:
            try:
                self._studio_data = ibl.load_studio_light(shading.studio_light)
            except Exception:  # noqa: BLE001
                log.exception("工作室光 %s 加载失败", shading.studio_light)
            self._studio_name = shading.studio_light

    # ---- 几何缓冲 ----
    def ensure_gbuffer(self, view: View) -> None:
        if view.gbuf_valid or view.gbuf_fbo is None:
            return
        ctx = self.ctx
        view.gbuf_fbo.use()
        ctx.viewport = (0, 0, view.width, view.height)
        view.gbuf_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE | moderngl.BLEND)
        self.gbuffer["u_view_proj"].write(view.camera.view_projection(view.aspect).astype("f4").T.tobytes())
        for mesh, material, _display, _ts, set_index in self.scene:
            if id(mesh) in self.hidden or self._skip(view, mesh):
                continue
            vao = self._vao(self.gbuffer, mesh, material)
            if vao is None:
                continue
            self.gbuffer["u_set"] = float(set_index + 1)
            if "u_object" in self.gbuffer:
                self.gbuffer["u_object"] = float(getattr(mesh, "pick_id", 0))
            set_model(self.gbuffer, mesh)
            vao.render(moderngl.TRIANGLES)
        for draw in list(self.extra_draws):
            draw(self, view, "gbuffer")
        view.gbuf_valid = True

    # ---- 页面需求反馈 ----
    def render_feedback(self, view: View) -> None:
        ctx = self.ctx
        view.feedback_fbo.use()
        ctx.viewport = (0, 0) + view.feedback_size
        view.feedback_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE | moderngl.BLEND)
        program = self.feedback
        program["u_view_proj"].write(view.camera.view_projection(view.aspect).astype("f4").T.tobytes())
        program["u_scale"] = view.width / view.feedback_size[0]
        program["u_aniso"] = float(self.anisotropy)
        for mesh, material, display, _ts, set_index in self.scene:
            if display is None or id(mesh) in self.hidden or self._skip(view, mesh):
                continue
            vao = self._vao(program, mesh, material)
            if vao is None:
                continue
            program["u_tex_size"] = float(display.size)
            program["u_max_level"] = float(display.levels - 1)
            program["u_set"] = float(set_index + 1)
            set_model(program, mesh)
            vao.render(moderngl.TRIANGLES)
        # 异步读回：先读进缓冲，之后用 Fence 判断好了没有，不让 CPU 等显卡
        view.feedback_fbo.read_into(view.feedback_buffer, components=4, alignment=1)
        if view.feedback_fence is not None:
            view.feedback_fence.release()
        view.feedback_fence = glx.Fence()
        view.feedback_pending = True

    def read_feedback(self, view: View, wait: bool = False) -> np.ndarray | None:
        """最近一次需求图（读回完成才有）：去重后的 (x, y, 级, 集合序号+1) 数组。"""
        if not view.feedback_pending or view.feedback_fbo is None or view.feedback_fence is None:
            return None
        if not view.feedback_fence.ready(wait_ns=100_000_000 if wait else 0):
            return None
        view.feedback_fence.release()
        view.feedback_fence = None
        view.feedback_pending = False
        raw = np.frombuffer(view.feedback_buffer.read(), np.uint8).reshape(-1, 4)
        packed = raw.view(np.uint32).ravel()
        unique = np.unique(packed)
        requests = unique.view(np.uint8).reshape(-1, 4)
        return requests[requests[:, 3] > 0]

    # ---- 主渲染 ----
    def render(self, view: View) -> None:
        if view.fbo is None:
            return
        hook = getattr(view, "before_render", None)
        if hook is not None:
            try:
                hook()
            except Exception:  # noqa: BLE001
                log.exception("视口渲染前的同步出错")
        ctx = self.ctx
        shading = view.shading
        self._prepare_lighting(shading)
        target = view.msaa_fbo or view.fbo
        target.use()
        ctx.viewport = (0, 0, view.width, view.height)
        target.clear(*self.bg_color, 1.0, depth=1.0)
        camera = view.camera
        view_matrix = camera.view_matrix()
        mode = MODE_INDEX.get(shading.mode, 1)
        transform = TRANSFORM_INDEX.get(shading.view_transform, 2)

        # 背景
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        bg = self.background
        inv_rot = np.identity(4)
        inv_rot[:3, :3] = view_matrix[:3, :3].T
        bg["u_inv_view_rot"].write(inv_rot.astype("f4").T.tobytes())
        tan_y, tan_x = camera.tan_half(view.aspect)
        bg["u_tan"] = (tan_x, tan_y)
        bg["u_ortho"] = int(camera.ortho)
        bg["u_bg_color"] = self.bg_color
        bg["u_bg_opacity"] = float(shading.background_opacity)
        bg["u_bg_blur"] = float(shading.background_blur)
        bg["u_transform"] = transform
        bg["u_exposure"] = float(shading.exposure)
        bg["u_mode"] = mode
        unit = self._bind_environment(bg, 0, shading)
        self._bg_vao.render(moderngl.TRIANGLES)

        # 模型
        ctx.enable(moderngl.DEPTH_TEST)
        program = self.shade
        program["u_view_proj"].write(camera.view_projection(view.aspect).astype("f4").T.tobytes())
        program["u_view"].write(view_matrix.astype("f4").T.tobytes())
        program["u_eye"] = tuple(float(v) for v in camera.eye)
        program["u_mode"] = mode
        program["u_channel"] = CHANNEL_INDEX.get(shading.channel, 0)
        program["u_solid"] = SOLID_INDEX.get(shading.solid_light, 0)
        program["u_solid_textured"] = int(shading.solid_color == "TEXTURE")
        program["u_transform"] = transform
        program["u_exposure"] = float(shading.exposure)
        program["u_bump"] = int(shading.bump)
        program["u_aniso"] = float(self.anisotropy)
        program["u_pixel_world"] = camera.world_per_pixel(camera.distance, view.height, view.aspect)
        if view.cursor is not None:
            position, normal, radius, hardness = view.cursor
            program["u_cursor"] = (float(position[0]), float(position[1]), float(position[2]), float(radius))
            program["u_cursor_hardness"] = float(hardness)
            program["u_cursor_on"] = 1
        else:
            program["u_cursor_on"] = 0
        unit = self._bind_environment(program, 4, shading)
        if self._matcap_tex is not None:
            unit = ibl.bind_matcap(program, self._matcap_tex, unit)
        if self._studio_data is not None:
            ibl.bind_studio(program, self._studio_data)
        nmap_unit = unit + 1
        wireframe = shading.mode == "WIREFRAME"
        xray = bool(getattr(shading, "show_xray", False)) and not wireframe
        program["u_alpha"] = float(getattr(shading, "xray_alpha", 0.5)) if xray else 1.0
        if xray:
            # X 光：模型半透明、不挡后面的东西
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA, moderngl.ONE, moderngl.ONE
        for mesh, material, display, ts, _set_index in (() if wireframe else self.scene):
            if id(mesh) in self.hidden or self._skip(view, mesh):
                continue
            vao = self._vao(program, mesh, material)
            if vao is None:
                continue
            if display is not None:
                color, mrao, height = self._wrap(display)
                color.use(0)
                mrao.use(1)
                height.use(2)
                display.clamp_tex.use(3)
                program["u_textured"] = 1
                program["u_tex_size"] = float(display.size)
                program["u_grid"] = float(display.grid)
                scale = ts.height_scale if ts is not None else 1.0
                program["u_height_world"] = 0.05 * mesh.radius * float(scale)
            else:
                self._white.use(0)
                self._white.use(1)
                self._white.use(2)
                self._clamp0.use(3)
                program["u_textured"] = 0
            program["u_color"] = 0
            program["u_mrao"] = 1
            program["u_height"] = 2
            program["u_clamp"] = 3
            set_model(program, mesh)
            maps = self.meshmaps.get(ts.uid) if (ts is not None and display is not None) else None
            if "u_has_nmap" in program:
                if maps is not None and maps.has_normal and "t" in maps.textures:
                    maps.textures["t"].use(nmap_unit)
                    program["u_nmap"] = nmap_unit
                    program["u_has_nmap"] = 1
                else:
                    program["u_has_nmap"] = 0
            vao.render(moderngl.TRIANGLES)
        if xray:
            ctx.disable(moderngl.BLEND)
            ctx.enable(moderngl.DEPTH_TEST)
        overlay = getattr(view, "overlay", None)
        if wireframe:
            self.draw_wires(view, 1.0, depth=False)
        elif overlay is not None and overlay.show_overlays and getattr(overlay, "show_wireframes", False):
            self.draw_wires(view, float(getattr(overlay, "wireframe_opacity", 0.6)), depth=not xray, colored=True)
        if self.overlays is not None:
            try:
                self.overlays.draw_scene(self, view)
            except Exception:  # noqa: BLE001
                log.exception("叠加层出错")
        overlay = getattr(view, "overlay", None)
        if self.selection_states and (overlay is None or (overlay.show_overlays and overlay.show_outline)):
            self._draw_outline(view, target)
        for draw in list(self.extra_draws):
            draw(self, view, "color")
        if self.overlays is not None:
            try:
                self.overlays.draw_top(self, view)
            except Exception:  # noqa: BLE001
                log.exception("叠加层出错")
        if view.msaa_fbo is not None:
            ctx.copy_framebuffer(view.fbo, view.msaa_fbo)
        view.dirty = False
        view.serial += 1

    def _draw_outline(self, view: View, target) -> None:
        """选中物体的橙色描边（当前物体亮一些），按几何缓冲里的物体编号找边。"""
        ctx = self.ctx
        key = tuple(sorted(self.selection_states.items()))
        if key != self._state_key:
            states = np.zeros(2048, np.uint8)
            for pick_id, value in self.selection_states.items():
                if 0 < pick_id < 2048:
                    states[pick_id] = int(round(value * 255))
            self._state_tex.write(states.tobytes())
            self._state_key = key
        self.ensure_gbuffer(view)
        target.use()
        ctx.viewport = (0, 0, view.width, view.height)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        view.gbuf_nrm.use(0)
        self._state_tex.use(1)
        program = self.outline
        program["u_ids"] = 0
        program["u_state"] = 1
        program["u_width"] = int(max(1, self.outline_width))
        program["u_selected"] = (0.93, 0.34, 0.0, 1.0)
        program["u_active"] = (1.0, 0.67, 0.25, 1.0)
        self._outline_vao.render(moderngl.TRIANGLES, vertices=3)
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

    def object_id_at(self, view: View, x: float, y: float) -> int:
        """像素 (x, y) 处的物体编号（0 表示没有物体）。"""
        if not (0 <= x < view.width and 0 <= y < view.height) or view.gbuf_fbo is None:
            return 0
        self.ensure_gbuffer(view)
        viewport = (int(x), view.height - 1 - int(y), 1, 1)
        raw = view.gbuf_fbo.read(viewport=viewport, components=4, attachment=1, dtype="f2", alignment=1)
        return int(round(float(np.frombuffer(raw, np.float16)[3])))

    def object_id_image(self, view: View, x0: int, y0: int, x1: int, y1: int):
        """一块区域里每个像素的物体编号 [h, w]（行序自上而下）和区域左上角 (x, y)。套索、刷选用。"""
        if view.gbuf_fbo is None:
            return np.zeros((0, 0), np.int64), (0, 0)
        self.ensure_gbuffer(view)
        x0, x1 = max(0, int(min(x0, x1))), min(view.width - 1, int(max(x0, x1)))
        y0, y1 = max(0, int(min(y0, y1))), min(view.height - 1, int(max(y0, y1)))
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w <= 0 or h <= 0:
            return np.zeros((0, 0), np.int64), (x0, y0)
        raw = view.gbuf_fbo.read(viewport=(x0, view.height - 1 - y1, w, h), components=4, attachment=1, dtype="f2",
                                 alignment=1)
        ids = np.rint(np.frombuffer(raw, np.float16).reshape(h, w, 4)[::-1, :, 3].astype(np.float32)).astype(np.int64)
        return ids, (x0, y0)

    def object_ids_in(self, view: View, x0: int, y0: int, x1: int, y1: int) -> set:
        """一个矩形里出现过的物体编号（框选用）。"""
        if view.gbuf_fbo is None:
            return set()
        self.ensure_gbuffer(view)
        x0, x1 = max(0, min(x0, x1)), min(view.width - 1, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(view.height - 1, max(y0, y1))
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w <= 0 or h <= 0:
            return set()
        raw = view.gbuf_fbo.read(viewport=(x0, view.height - 1 - y1, w, h), components=4, attachment=1, dtype="f2",
                                 alignment=1)
        ids = np.rint(np.frombuffer(raw, np.float16).reshape(-1, 4)[:, 3].astype(np.float32)).astype(np.int64)
        return {int(v) for v in np.unique(ids) if v > 0}

    def _bind_environment(self, program, unit: int, shading) -> int:
        return self.environment.bind(program, unit, rotation=math.radians(float(shading.hdri_rotation)),
                                     strength=float(shading.hdri_strength))

    def read_pixels(self, view: View) -> np.ndarray:
        """把视口画面读成 [h, w, 4] 的 uint8 数组（测试和截图用）。"""
        raw = view.fbo.read(components=4, alignment=1)
        return np.frombuffer(raw, np.uint8).reshape(view.height, view.width, 4)[::-1]

    def release(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        self._vaos.clear()
        self.environment.release()
