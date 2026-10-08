"""平面视图：把纹理集的合成结果按 UV 铺开显示，叠加 UV 线框。给 UV / 图像编辑器用。"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from .meshgpu import mesh_layout
from .renderer import _VT_GLSL
from .selection import ANTS_GLSL, ants_values

CHANNEL_INDEX = {"basecolor": 0, "metallic": 1, "roughness": 2, "height": 3}

_VERT = """#version 430
in vec2 in_p;
uniform vec2 u_center; uniform float u_zoom; uniform vec2 u_size;
out vec2 v_uv;
void main(){ v_uv = in_p; gl_Position = vec4((in_p - u_center) * u_zoom * 2.0 / u_size, 0.0, 1.0); }
"""

_FRAG = """#version 430
in vec2 v_uv; out vec4 o_color;
uniform int u_channel;
uniform sampler2D u_sel; uniform int u_sel_on;     // 选区（小贴图），有选区时画边线
""" + _VT_GLSL + ANTS_GLSL + """
vec3 to_display(vec3 c){ return mix(1.055 * pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.4)) - 0.055, c * 12.92, lessThanEqual(c, vec3(0.0031308))); }
void main(){
  vec2 dx = dFdx(v_uv), dy = dFdy(v_uv);
  float k = vt_scale(v_uv, dx, dy);
  vec2 gx = dx * k, gy = dy * k;
  vec3 color;
  if (u_channel == 0) color = to_display(textureGrad(u_color, v_uv, gx, gy).rgb);
  else if (u_channel == 1) color = vec3(textureGrad(u_mrao, v_uv, gx, gy).r);
  else if (u_channel == 2) color = vec3(textureGrad(u_mrao, v_uv, gx, gy).g);
  else color = vec3(clamp(0.5 + textureGrad(u_height, v_uv, gx, gy).r * 0.5, 0.0, 1.0));
  o_color = vec4(color, 1.0);
  if (u_sel_on == 1) {
    vec4 a = ants(u_sel, v_uv, gl_FragCoord.xy);
    o_color.rgb = mix(o_color.rgb, a.rgb, a.a);
  }
}
"""

_WIRE_VERT = """#version 430
in vec2 in_uv;
uniform vec2 u_center; uniform float u_zoom; uniform vec2 u_size;
void main(){ gl_Position = vec4((in_uv - u_center) * u_zoom * 2.0 / u_size, 0.0, 1.0); }
"""

_WIRE_FRAG = """#version 430
uniform vec4 u_line; out vec4 o_color;
void main(){ o_color = u_line; }
"""


class View2D:
    """一个平面视图。center 是视口中心对应的 UV，zoom 是一个 UV 单位占多少像素。"""

    is_2d = True

    def __init__(self, renderer2d: "Renderer2D") -> None:
        self.renderer2d = renderer2d
        self.ctx = renderer2d.ctx
        self.width = 0
        self.height = 0
        self.center = np.array([0.5, 0.5])
        self.zoom = 512.0
        self.channel = "basecolor"
        self.show_wire = True
        self.wire_opacity = 0.35         # UV 线框的不透明度（缩小到线挤在一起时还会自动变淡）
        self.set_uid = 0
        self.color_tex = None
        self.fbo = None
        self.dirty = True
        self.feedback_pending = False
        self.cursor = None
        self.serial = 0
        self.requests = None
        self._fit_pending = True

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)

    def resize(self, width: int, height: int) -> None:
        width, height = max(8, int(width)), max(8, int(height))
        if (width, height) == (self.width, self.height) and self.fbo is not None:
            return
        self.release()
        self.width, self.height = width, height
        self.color_tex = self.ctx.texture((width, height), 4, dtype="f1")
        self.color_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = self.ctx.framebuffer([self.color_tex])
        if self._fit_pending:
            self.fit()
            self._fit_pending = False
        self.dirty = True

    def fit(self) -> None:
        self.center = np.array([0.5, 0.5])
        self.zoom = max(16.0, min(self.width, self.height) * 0.92)
        self.dirty = True

    def invalidate_camera(self) -> None:
        self.dirty = True

    def pan(self, dx: float, dy: float) -> None:
        self.center = self.center - np.array([dx, -dy]) / self.zoom
        self.dirty = True

    def zoom_at(self, factor: float, x: float, y: float) -> None:
        """以像素 (x, y)（原点左上）为不动点缩放。"""
        uv = self.pixel_to_uv(x, y)
        self.zoom = float(min(max(self.zoom * factor, 16.0), 16384.0 * 64.0))
        offset = np.array([x - self.width * 0.5, (self.height * 0.5 - y)]) / self.zoom
        self.center = uv - offset
        self.dirty = True

    def pixel_to_uv(self, x: float, y: float) -> np.ndarray:
        return self.center + np.array([x - self.width * 0.5, self.height * 0.5 - y]) / self.zoom

    def compute_requests(self, display, set_index: int) -> np.ndarray:
        """按可见范围和缩放直接算出需要哪些页，不用显卡反馈。"""
        level = int(min(display.levels - 1, max(0, math.floor(math.log2(max(display.size / self.zoom, 1e-9))))))
        grid = display.level_size[level]
        half = np.array([self.width, self.height]) * 0.5 / self.zoom
        lo = np.clip(np.floor((self.center - half) * grid), 0, grid - 1).astype(int)
        hi = np.clip(np.floor((self.center + half) * grid), 0, grid - 1).astype(int)
        if (self.center + half < 0).any() or (self.center - half > 1).any():
            return np.zeros((0, 4), np.uint8)
        xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        out = np.zeros((xs.size, 4), np.uint8)
        out[:, 0] = xs.ravel()
        out[:, 1] = ys.ravel()
        out[:, 2] = level
        out[:, 3] = set_index + 1
        return out

    def state(self) -> dict:
        return {"center": [float(v) for v in self.center], "zoom": float(self.zoom), "channel": self.channel,
                "wire": self.show_wire, "wire_opacity": self.wire_opacity}

    def set_state(self, data: dict) -> None:
        if not data:
            return
        self.center = np.asarray(data.get("center", self.center), float)
        self.zoom = float(data.get("zoom", self.zoom))
        self.channel = data.get("channel", self.channel)
        self.show_wire = bool(data.get("wire", self.show_wire))
        self.wire_opacity = float(data.get("wire_opacity", self.wire_opacity))
        self._fit_pending = False

    def release(self) -> None:
        for item in (self.fbo, self.color_tex):
            if item is not None:
                item.release()
        self.fbo = self.color_tex = None


class Renderer2D:
    def __init__(self, ctx: moderngl.Context, renderer) -> None:
        self.ctx = ctx
        self.renderer = renderer
        self.prefs = None                                   # 引擎设置：选区边线的样子
        self.program = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self.wire = ctx.program(vertex_shader=_WIRE_VERT, fragment_shader=_WIRE_FRAG)
        quad = np.array([0, 0, 1, 0, 0, 1, 1, 1], "f4")
        self._quad = ctx.buffer(quad.tobytes())
        self._vao = ctx.vertex_array(self.program, [(self._quad, "2f", "in_p")])
        frame = np.array([0, 0, 1, 0, 1, 1, 0, 1], "f4")
        self._frame = ctx.buffer(frame.tobytes())
        self._frame_vao = ctx.vertex_array(self.wire, [(self._frame, "2f", "in_uv")])
        self._wire_vaos: dict[tuple, moderngl.VertexArray] = {}
        self.bg_color = (0.13, 0.13, 0.14)
        self.wire_color = (1.0, 1.0, 1.0, 0.28)

    def forget_mesh(self, mesh) -> None:
        for key in [k for k in self._wire_vaos if k[0] == id(mesh)]:
            self._wire_vaos.pop(key).release()

    def _wire_vao(self, mesh, material: int):
        ibo = mesh.material_ibo[material] if material < len(mesh.material_ibo) else None
        if ibo is None:
            return None
        key = (id(mesh), material)
        vao = self._wire_vaos.get(key)
        if vao is None:
            vao = self.ctx.vertex_array(self.wire, [mesh_layout(self.wire, mesh.vbo)], index_buffer=ibo,
                                        index_element_size=4)
            self._wire_vaos[key] = vao
        return vao

    def render(self, view: View2D, state, scene: list) -> None:
        """state 是引擎里的纹理集状态（带 display 和 index），scene 是渲染器的场景列表。"""
        if view.fbo is None:
            return
        ctx = self.ctx
        view.fbo.use()
        ctx.viewport = (0, 0, view.width, view.height)
        view.fbo.clear(*self.bg_color, 1.0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        if state is not None:
            display = state.display
            program = self.program
            color, mrao, height = self.renderer._wrap(display)
            color.use(0)
            mrao.use(1)
            height.use(2)
            display.clamp_tex.use(3)
            program["u_color"] = 0
            program["u_mrao"] = 1
            program["u_height"] = 2
            program["u_clamp"] = 3
            program["u_tex_size"] = float(display.size)
            program["u_grid"] = float(display.grid)
            program["u_aniso"] = 1.0
            program["u_channel"] = CHANNEL_INDEX.get(view.channel, 0)
            selection = getattr(state, "selection", None)
            viewport = getattr(self.prefs, "viewport", None)
            sel_tex = selection.preview() if (selection is not None and getattr(viewport, "selection_ants", True)) else None
            if "u_sel_on" in program:
                program["u_sel_on"] = 1 if sel_tex is not None else 0
            if sel_tex is not None:
                sel_tex.use(4)
                program["u_sel"] = 4
                program["u_ants"] = ants_values(self.prefs)
            for target in (program, self.wire):
                target["u_center"] = (float(view.center[0]), float(view.center[1]))
                target["u_zoom"] = float(view.zoom)
                target["u_size"] = (float(view.width), float(view.height))
            self._vao.render(moderngl.TRIANGLE_STRIP)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            if view.show_wire and view.wire_opacity > 0.0:
                ctx.wireframe = True
                for mesh, material, mesh_display, _ts, _set_index in scene:
                    if mesh_display is not display:
                        continue
                    edge_px = mesh.uv_edge * float(view.zoom)          # 屏幕上一条 UV 边大约多长
                    fade = min(1.0, max(0.0, (edge_px - 6.0) / 18.0))
                    alpha = float(view.wire_opacity) * fade
                    if alpha <= 0.01:
                        continue
                    self.wire["u_line"] = (*self.wire_color[:3], alpha)
                    vao = self._wire_vao(mesh, material)
                    if vao is not None:
                        vao.render(moderngl.TRIANGLES)
                ctx.wireframe = False
            self.wire["u_line"] = (1.0, 1.0, 1.0, 0.45)
            self._frame_vao.render(moderngl.LINE_LOOP)
            ctx.disable(moderngl.BLEND)
        view.dirty = False
        view.serial += 1

    def release(self) -> None:
        for vao in self._wire_vaos.values():
            vao.release()
        self._wire_vaos.clear()
