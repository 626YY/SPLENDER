"""导出法线贴图：从高模烘的切线空间法线，加上手绘高度换算出的凹凸，合成一张。
高度逐条合成，留三条（上、中、下）滑动着求坡度，整张图不进内存。"""
from __future__ import annotations

import logging
import time

import moderngl
import numpy as np

from .pagepool import PAGE

log = logging.getLogger("splender.engine.io")

_VERT = """
#version 430
void main(){
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_FRAG = """
#version 430
#define PAGE %d
uniform sampler2D u_up;            // 上面一条（v 更大）
uniform sampler2D u_mid;           // 这一条
uniform sampler2D u_down;          // 下面一条（v 更小）
uniform int u_has_up;
uniform int u_has_down;
uniform sampler2D u_baked;         // 烘焙的切线空间法线（×0.5+0.5）
uniform int u_has_baked;
uniform int u_row;                 // 这一条在这一级的格行号
uniform float u_size;              // 这一级整张图的边长（纹素）
uniform float u_slope;             // 相邻纹素高度差 → 坡度
uniform int u_flip_green;
out vec4 o_color;

float h_at(int x, int y) {
  int w = textureSize(u_mid, 0).x;
  x = clamp(x, 0, w - 1);
  if (y >= PAGE) {
    if (u_has_up == 1) return texelFetch(u_up, ivec2(x, y - PAGE), 0).r;
    y = PAGE - 1;
  }
  if (y < 0) {
    if (u_has_down == 1) return texelFetch(u_down, ivec2(x, y + PAGE), 0).r;
    y = 0;
  }
  return texelFetch(u_mid, ivec2(x, y), 0).r;
}

void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  float dx = (h_at(p.x + 1, p.y) - h_at(p.x - 1, p.y)) * 0.5;
  float dy = (h_at(p.x, p.y + 1) - h_at(p.x, p.y - 1)) * 0.5;
  vec3 n = normalize(vec3(-dx * u_slope, -dy * u_slope, 1.0));
  if (u_has_baked == 1) {
    vec2 uv = (vec2(p) + vec2(0.5, 0.5 + float(u_row * PAGE))) / u_size;
    vec3 b = texture(u_baked, uv).xyz * 2.0 - 1.0;
    n = normalize(vec3(b.xy + n.xy, b.z * n.z));       // 两张法线叠加（whiteout）
  }
  if (u_flip_green == 1) n.y = -n.y;
  o_color = vec4(n * 0.5 + 0.5, 1.0);
}
"""


def surface_scale(project, ts) -> tuple:
    """这套贴图对应的模型：每单位 UV 平均对应多长的表面（世界单位），以及模型半径。"""
    world = flat = 0.0
    radius = 0.0
    for obj in project.objects:
        mesh = obj.data
        ids = [index for index, uid in enumerate(obj.material_sets) if uid == ts.uid]
        if not ids or mesh.uvs is None:
            continue
        pos = np.asarray(mesh.positions, np.float64).reshape(-1, 3, 3)
        uvs = np.asarray(mesh.uvs, np.float64).reshape(-1, 3, 2)
        mats = np.asarray(mesh.material_ids) if mesh.material_ids is not None else np.zeros(len(pos), np.int32)
        chosen = np.isin(mats, ids)
        p, t = pos[chosen], uvs[chosen]
        world += 0.5 * float(np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum())
        e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        flat += 0.5 * float(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum())
        extent = np.asarray(mesh.bounds_max, np.float64) - np.asarray(mesh.bounds_min, np.float64)
        radius = max(radius, float(np.linalg.norm(extent) * 0.5))
    per_uv = float(np.sqrt(world / flat)) if world > 0 and flat > 0 else 1.0
    return per_uv, (radius or 1.0)


def export_normal(engine, ts, path: str, progress=None, size: int | None = None) -> dict:
    """把这套贴图的法线导出成 PNG（RGB，8 或 16 位）。"""
    from .export_png import PngStreamWriter
    from .projectio import compose_row

    started = time.perf_counter()
    settings = ts.export
    state = engine.sets[ts.uid]
    display = state.display
    full = display.size
    size = int(size or full)
    level = max(0, min(display.levels - 1, int(round(np.log2(full / size))))) if size < full else 0
    size = full >> level
    grid = display.level_size[level]
    width = grid * PAGE
    ctx = engine.ctx
    depth = 16 if settings.normal_depth == "16" else 8
    per_uv, radius = surface_scale(engine.project, ts)
    texel_world = per_uv / float(size)
    height_world = 0.05 * radius * float(ts.height_scale)
    slope = height_world / max(texel_world, 1e-12) * float(settings.normal_strength)
    maps = engine.meshmap(ts.uid) if hasattr(engine, "meshmap") else None
    baked = maps.textures["t"] if (settings.normal_bake and maps is not None and maps.has_normal) else None

    strip_color = ctx.texture((width, PAGE), 4, dtype="f1")
    strip_mrao = ctx.texture((width, PAGE), 4, dtype="f1")
    heights = [ctx.texture((width, PAGE), 1, dtype="f2") for _ in range(3)]
    for item in heights:
        item.filter = (moderngl.NEAREST, moderngl.NEAREST)
    fbos = [ctx.framebuffer([strip_color, strip_mrao, item]) for item in heights]
    out_tex = ctx.texture((width, PAGE), 4, dtype="f2")
    out_fbo = ctx.framebuffer([out_tex])
    program = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG % PAGE)
    vao = ctx.vertex_array(program, [])
    writer = PngStreamWriter(path, size, size, 3, bit_depth=depth)
    try:
        engine._prepare_composite(state, with_stroke=False)

        def emit(row: int) -> None:
            heights[row % 3].use(0)
            heights[(row + 1) % 3].use(1)
            heights[(row - 1) % 3].use(2)
            program["u_mid"] = 0
            program["u_up"] = 1
            program["u_down"] = 2
            program["u_has_up"] = 1 if row + 1 < grid else 0
            program["u_has_down"] = 1 if row > 0 else 0
            if baked is not None:
                baked.use(3)
                program["u_baked"] = 3
                program["u_has_baked"] = 1
            else:
                program["u_has_baked"] = 0
            program["u_row"] = int(row)
            program["u_size"] = float(size)
            program["u_slope"] = float(slope)
            program["u_flip_green"] = 1 if ts.meshmap.normal_format == "DIRECTX" else 0
            out_fbo.use()
            ctx.viewport = (0, 0, width, PAGE)
            ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
            vao.render(moderngl.TRIANGLES, vertices=3)
            raw = np.frombuffer(out_fbo.read(components=4, dtype="f2", alignment=1), np.float16)
            rows = raw.reshape(PAGE, width, 4)[::-1, :, :3].astype(np.float32)
            if depth == 16:
                data = np.clip(np.rint(rows * 65535.0), 0, 65535).astype(np.uint16)
            else:
                data = np.clip(np.rint(rows * 255.0), 0, 255).astype(np.uint8)
            writer.write_rows(np.ascontiguousarray(data))

        compose_row(engine, state, level, grid - 1, grid, fbos[(grid - 1) % 3], width, srgb_encode=False)
        for row in range(grid - 1, -1, -1):           # 图片第一行对应 v=1
            if row > 0:
                compose_row(engine, state, level, row - 1, grid, fbos[(row - 1) % 3], width, srgb_encode=False)
            emit(row)
            if progress is not None:
                progress((grid - row) / grid)
        writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        vao.release()
        program.release()
        out_fbo.release()
        out_tex.release()
        for item in fbos:
            item.release()
        for item in heights:
            item.release()
        strip_color.release()
        strip_mrao.release()
    elapsed = time.perf_counter() - started
    log.info("已导出 %s（法线，%d×%d，%d 位），用时 %.2f 秒", path, size, size, depth, elapsed)
    return {"path": path, "size": size, "seconds": elapsed, "slope": slope, "baked": baked is not None}
