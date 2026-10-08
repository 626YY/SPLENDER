"""视口叠加层（和 Blender 一样）：地面网格和坐标轴线、3D 游标、摄像机、灯光、空物体。

坐标：界面上按 Blender 的 Z 轴朝上显示和操作；内部渲染是 Y 轴朝上，两者用 doc.objects 里的固定换算互转。
地面网格画在内部的 XZ 平面（就是 Blender 的 XY 平面）上，X 轴红、Y 轴绿。
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from ..doc.objects import camera_fov, local_to_internal_matrix

_GRID_VERT = """#version 430
in vec2 in_xz;
uniform mat4 u_view_proj;
uniform vec2 u_center;
uniform float u_extent;
out vec3 v_world;
void main() {
  vec3 p = vec3(u_center.x + in_xz.x * u_extent, 0.0, u_center.y + in_xz.y * u_extent);
  v_world = p;
  gl_Position = u_view_proj * vec4(p, 1.0);
}
"""

_GRID_FRAG = """#version 430
in vec3 v_world;
uniform vec3 u_eye;
uniform float u_scale;          // 网格的一格多大（世界单位）
uniform float u_fade;           // 多远开始淡出
uniform vec4 u_line;
uniform vec4 u_axis_x;
uniform vec4 u_axis_y;
uniform int u_show_grid;
uniform int u_show_x;
uniform int u_show_y;
out vec4 o_color;

float lines(vec2 p, float spacing) {
  vec2 c = p / spacing;
  vec2 w = fwidth(c);
  vec2 g = abs(fract(c - 0.5) - 0.5) / max(w, vec2(1e-6));
  return 1.0 - min(min(g.x, g.y), 1.0);
}

void main() {
  vec2 p = v_world.xz;
  vec2 px = fwidth(p);
  float per_px = max(px.x, px.y);           // 一个像素多少世界单位
  float alpha = 0.0;
  if (u_show_grid == 1) {
    // 三层：间距每层 ×10，太密（每格不到 6 像素）的层淡掉
    float spacing = u_scale;
    for (int k = 0; k < 4; k++) {
      float cell_px = spacing / max(per_px, 1e-9);
      float vis = smoothstep(6.0, 30.0, cell_px);
      alpha = max(alpha, lines(p, spacing) * vis * (0.35 + 0.25 * float(k > 0)));
      spacing *= 10.0;
    }
  }
  vec4 color = vec4(u_line.rgb, alpha * u_line.a);
  float ax = 1.0 - min(abs(p.y) / max(px.y * 1.2, 1e-9), 1.0);   // X 轴：内部 z = 0 的线
  float ay = 1.0 - min(abs(p.x) / max(px.x * 1.2, 1e-9), 1.0);   // Y 轴：内部 x = 0 的线
  if (u_show_x == 1 && ax > 0.0) color = mix(color, vec4(u_axis_x.rgb, u_axis_x.a), ax);
  if (u_show_y == 1 && ay > 0.0) color = mix(color, vec4(u_axis_y.rgb, u_axis_y.a), ay);
  float dist = length(v_world - u_eye);
  color.a *= 1.0 - smoothstep(u_fade * 0.5, u_fade, dist);
  if (color.a < 0.004) discard;
  o_color = color;
}
"""

_LINE_VERT = """#version 430
in vec3 in_pos;
in vec4 in_color;
uniform mat4 u_view_proj;
out vec4 v_color;
void main() { v_color = in_color; gl_Position = u_view_proj * vec4(in_pos, 1.0); }
"""

_LINE_FRAG = """#version 430
in vec4 v_color;
out vec4 o_color;
void main() { o_color = v_color; }
"""

UNSELECTED = (0.0, 0.0, 0.0, 1.0)
SELECTED = (0.93, 0.34, 0.0, 1.0)
ACTIVE = (1.0, 0.67, 0.25, 1.0)


class Overlays:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.grid_prog = ctx.program(vertex_shader=_GRID_VERT, fragment_shader=_GRID_FRAG)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
        self.grid_vbo = ctx.buffer(quad.tobytes())
        self.grid_vao = ctx.vertex_array(self.grid_prog, [(self.grid_vbo, "2f", "in_xz")])
        self.line_prog = ctx.program(vertex_shader=_LINE_VERT, fragment_shader=_LINE_FRAG)
        self.line_vbo = ctx.buffer(reserve=7 * 4 * 4096)
        self.line_vao = ctx.vertex_array(self.line_prog, [(self.line_vbo, "3f 4f", "in_pos", "in_color")])
        self.project = None                 # 引擎设置
        self.guide = None                   # 变换时的约束轴线 [(点, 方向, 颜色)]，内部坐标

    # ---------------------------------------------------------------- 入口
    def draw_scene(self, renderer, view) -> None:
        """模型之后、描边之前：地面网格、摄像机、灯光、空物体（有深度测试）。"""
        overlay = getattr(view, "overlay", None)
        if overlay is None or not overlay.show_overlays:
            return
        ctx = self.ctx
        camera = view.camera
        view_proj = camera.view_projection(view.aspect)
        if overlay.show_floor or overlay.show_axis_x or overlay.show_axis_y:
            self._draw_grid(view, overlay, view_proj)
        if overlay.show_extras and self.project is not None:
            segments = []
            looking_through = getattr(view, "camera_object_uid", 0)
            local = getattr(view, "local_uids", None)
            for obj in self.project.scene_objects:
                if not obj.visible or obj.uid == looking_through or (local is not None and obj.uid not in local):
                    continue
                segments.extend(self._glyph(obj, view))
            self._draw_lines(segments, view_proj, depth=True)

    def draw_top(self, renderer, view) -> None:
        """最上面（不被模型挡住）：变换的约束轴线、物体原点、3D 游标。"""
        overlay = getattr(view, "overlay", None)
        view_proj = view.camera.view_projection(view.aspect)
        if self.guide:
            segments = []
            reach = max(view.camera.distance * 50.0, view.camera.scene_radius * 20.0)
            for point, direction, color in self.guide:
                p = np.asarray(point, np.float64)
                d = np.asarray(direction, np.float64)
                d = d / max(np.linalg.norm(d), 1e-12)
                a, b = p - d * reach, p + d * reach
                segments.append([*a, *color, 1.0])
                segments.append([*b, *color, 1.0])
            self._draw_lines(segments, view_proj, depth=False)
        if overlay is None or not overlay.show_overlays or self.project is None:
            return
        if getattr(overlay, "show_object_origins", True):
            self._draw_lines(self._origins(view, bool(getattr(overlay, "show_object_origins_all", False))),
                             view_proj, depth=False)
        if overlay.show_cursor:
            self._draw_lines(self._cursor(view), view_proj, depth=False)

    # ---------------------------------------------------------------- 网格
    def _draw_grid(self, view, overlay, view_proj) -> None:
        ctx = self.ctx
        camera = view.camera
        program = self.grid_prog
        program["u_view_proj"].write(view_proj.astype("f4").T.tobytes())
        eye = camera.eye
        reach = max(camera.distance * 40.0, camera.scene_radius * 20.0, 50.0)
        program["u_center"] = (float(eye[0]), float(eye[2]))
        program["u_extent"] = float(reach)
        program["u_eye"] = tuple(float(v) for v in eye)
        program["u_scale"] = float(max(1e-4, overlay.grid_scale))
        program["u_fade"] = float(reach)
        program["u_line"] = (0.33, 0.33, 0.33, 1.0)
        program["u_axis_x"] = (0.9, 0.24, 0.27, 0.9)
        program["u_axis_y"] = (0.48, 0.7, 0.24, 0.9)
        program["u_show_grid"] = int(bool(overlay.show_floor))
        program["u_show_x"] = int(bool(overlay.show_axis_x))
        program["u_show_y"] = int(bool(overlay.show_axis_y))
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.depth_mask = False
        self.grid_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)

    # ---------------------------------------------------------------- 线
    def _draw_lines(self, segments: list, view_proj, depth: bool) -> None:
        if not segments:
            return
        data = np.asarray(segments, np.float32).reshape(-1, 7)
        needed = data.nbytes
        if needed > self.line_vbo.size:
            self.line_vbo.orphan(needed * 2)
        self.line_vbo.write(data.tobytes())
        ctx = self.ctx
        self.line_prog["u_view_proj"].write(view_proj.astype("f4").T.tobytes())
        if depth:
            ctx.enable(moderngl.DEPTH_TEST)
        else:
            ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.line_vao.render(moderngl.LINES, vertices=len(data))
        ctx.disable(moderngl.BLEND)
        ctx.enable(moderngl.DEPTH_TEST)

    def _color_of(self, obj) -> tuple:
        project = self.project
        if obj.select and project is not None and project.active_object_uid == obj.uid:
            return ACTIVE
        return SELECTED if obj.select else UNSELECTED

    def _glyph(self, obj, view) -> list:
        """摄像机、灯光、空物体的线段（内部坐标）。每段两个点，每个点 7 个数（xyz + rgba）。"""
        m = local_to_internal_matrix(obj.transform.matrix())
        color = self._color_of(obj)
        out = []

        def seg(a, b, c=color):
            pa = m[:3, :3] @ np.asarray(a, np.float64) + m[:3, 3]
            pb = m[:3, :3] @ np.asarray(b, np.float64) + m[:3, 3]
            out.append([*pa, *c])
            out.append([*pb, *c])

        # 局部坐标按 Blender：摄像机、灯光朝本地 -Z 看，+Y 朝上（这里是 Blender 的局部坐标，矩阵里已含换算）
        if obj.kind == "CAMERA":
            data = obj.data
            size = float(data.display_size)
            aspect = 16.0 / 9.0
            if data.projection == "ORTHO":
                half_w = float(data.ortho_scale) * 0.5
                depth = size
            else:
                fov = camera_fov(data, aspect)
                depth = size
                half_w = math.tan(fov * 0.5) * depth * aspect if aspect >= 1.0 else math.tan(fov * 0.5) * depth
            half_h = half_w / aspect
            corners = [(-half_w, -half_h, -depth), (half_w, -half_h, -depth), (half_w, half_h, -depth),
                       (-half_w, half_h, -depth)]
            for i in range(4):
                seg(corners[i], corners[(i + 1) % 4])
                if data.projection != "ORTHO":
                    seg((0.0, 0.0, 0.0), corners[i])
            top = half_h * 1.15
            tri = [(-half_w * 0.6, top, -depth), (half_w * 0.6, top, -depth), (0.0, top + half_h * 0.6, -depth)]
            for i in range(3):
                seg(tri[i], tri[(i + 1) % 3])
        elif obj.kind == "LIGHT":
            data = obj.data
            r = 0.12 * max(0.2, view.camera.world_per_pixel(view.camera.depth_of(m[:3, 3]), view.height,
                                                               view.aspect) * 90.0)
            ring = [(math.cos(t) * r, math.sin(t) * r, 0.0) for t in np.linspace(0, 2 * math.pi, 25)]
            # 圆圈朝向相机：在相机平面里画
            cam_right, cam_up, _back = view.camera.basis()
            center = m[:3, 3]
            for i in range(24):
                a = center + cam_right * ring[i][0] + cam_up * ring[i][1]
                b = center + cam_right * ring[i + 1][0] + cam_up * ring[i + 1][1]
                out.append([*a, *color])
                out.append([*b, *color])
            if data.type in ("SUN", "SPOT", "AREA"):
                seg((0.0, 0.0, 0.0), (0.0, 0.0, -1.5))
            if data.type == "SPOT":
                half = math.radians(float(data.spot_size)) * 0.5
                length = 1.5
                rr = math.tan(half) * length
                ring2 = [(math.cos(t) * rr, math.sin(t) * rr, -length) for t in np.linspace(0, 2 * math.pi, 25)]
                for i in range(24):
                    seg(ring2[i], ring2[i + 1])
                for i in range(0, 24, 6):
                    seg((0.0, 0.0, 0.0), ring2[i])
            if data.type == "AREA":
                s = float(data.size) * 0.5
                quad = [(-s, -s, 0.0), (s, -s, 0.0), (s, s, 0.0), (-s, s, 0.0)]
                for i in range(4):
                    seg(quad[i], quad[(i + 1) % 4])
        else:
            self._empty_shape(obj, seg)
        return out

    @staticmethod
    def _empty_shape(obj, seg) -> None:
        """空物体的样子（局部坐标，和 Blender 一样）：坐标轴、箭头、单箭头、圆、立方体、球、锥。"""
        s = float(obj.empty_size)
        kind = getattr(obj, "empty_display_type", "PLAIN_AXES")
        circle = [(math.cos(t), math.sin(t)) for t in np.linspace(0.0, 2.0 * math.pi, 33)]
        if kind == "ARROWS":
            for axis in range(3):
                tip = [0.0, 0.0, 0.0]
                tip[axis] = s
                seg((0.0, 0.0, 0.0), tuple(tip))
                other = [(axis + 1) % 3, (axis + 2) % 3]
                for o in other:
                    for sign in (-1.0, 1.0):
                        p = [0.0, 0.0, 0.0]
                        p[axis] = s * 0.85
                        p[o] = sign * s * 0.06
                        seg(tuple(tip), tuple(p))
        elif kind == "SINGLE_ARROW":
            seg((0.0, 0.0, 0.0), (0.0, 0.0, s))
            for dx, dy in ((0.06, 0.0), (-0.06, 0.0), (0.0, 0.06), (0.0, -0.06)):
                seg((0.0, 0.0, s), (dx * s, dy * s, s * 0.85))
        elif kind == "CIRCLE":
            for i in range(32):
                seg((circle[i][0] * s, circle[i][1] * s, 0.0), (circle[i + 1][0] * s, circle[i + 1][1] * s, 0.0))
        elif kind == "CUBE":
            corners = [(x, y, z) for x in (-s, s) for y in (-s, s) for z in (-s, s)]
            for i, a in enumerate(corners):
                for b in corners[i + 1:]:
                    if sum(1 for u, v in zip(a, b) if u != v) == 1:
                        seg(a, b)
        elif kind == "SPHERE":
            for i in range(32):
                (c0, s0), (c1, s1) = circle[i], circle[i + 1]
                seg((c0 * s, s0 * s, 0.0), (c1 * s, s1 * s, 0.0))
                seg((c0 * s, 0.0, s0 * s), (c1 * s, 0.0, s1 * s))
                seg((0.0, c0 * s, s0 * s), (0.0, c1 * s, s1 * s))
        elif kind == "CONE":
            for i in range(32):
                seg((circle[i][0] * s, circle[i][1] * s, 0.0), (circle[i + 1][0] * s, circle[i + 1][1] * s, 0.0))
            for i in range(0, 32, 8):
                seg((circle[i][0] * s, circle[i][1] * s, 0.0), (0.0, 0.0, 2.0 * s))
        else:
            seg((-s, 0, 0), (s, 0, 0))
            seg((0, -s, 0), (0, s, 0))
            seg((0, 0, -s), (0, 0, s))

    def _origins(self, view, show_all: bool) -> list:
        """物体原点：选中的画橙色小点（当前物体亮一些），show_all 时没选中的也画（白色）。"""
        from ..doc.objects import to_internal

        project = self.project
        camera = view.camera
        right, up, _back = camera.basis()
        out = []
        for obj in project.all_objects():
            if not obj.visible or (not obj.select and not show_all):
                continue
            if obj.kind != "MESH" and not obj.select:
                continue
            position = to_internal(np.asarray(obj.transform.location, np.float64))
            per_px = camera.world_per_pixel(camera.depth_of(position), view.height, view.aspect)
            if obj.select:
                color = ACTIVE if project.active_object_uid == obj.uid else SELECTED
            else:
                color = (0.85, 0.85, 0.85, 1.0)
            radius = 2.5 * per_px
            steps = 10
            for i in range(steps):
                t0 = 2.0 * math.pi * i / steps
                t1 = 2.0 * math.pi * (i + 1) / steps
                for scale in (1.0, 0.55):
                    a = position + (right * math.cos(t0) + up * math.sin(t0)) * radius * scale
                    b = position + (right * math.cos(t1) + up * math.sin(t1)) * radius * scale
                    out.append([*a, *color])
                    out.append([*b, *color])
        return out

    def _cursor(self, view) -> list:
        from ..doc.objects import to_internal

        project = self.project
        position = to_internal(np.asarray(project.cursor_location, np.float64))
        camera = view.camera
        per_px = camera.world_per_pixel(camera.depth_of(position), view.height, view.aspect)
        right, up, _back = camera.basis()
        out = []
        radius = 10.0 * per_px
        steps = 24
        for i in range(steps):
            t0 = 2.0 * math.pi * i / steps
            t1 = 2.0 * math.pi * (i + 1) / steps
            color = (1.0, 0.15, 0.15, 1.0) if i % 2 == 0 else (1.0, 1.0, 1.0, 1.0)
            a = position + right * math.cos(t0) * radius + up * math.sin(t0) * radius
            b = position + right * math.cos(t1) * radius + up * math.sin(t1) * radius
            out.append([*a, *color])
            out.append([*b, *color])
        inner, outer = 5.0 * per_px, 18.0 * per_px
        for direction in (right, -right, up, -up):
            a = position + direction * inner
            b = position + direction * outer
            out.append([*a, 0.0, 0.0, 0.0, 1.0])
            out.append([*b, 0.0, 0.0, 0.0, 1.0])
        return out

    def release(self) -> None:
        for item in (self.grid_vao, self.grid_vbo, self.grid_prog, self.line_vao, self.line_vbo, self.line_prog):
            item.release()
