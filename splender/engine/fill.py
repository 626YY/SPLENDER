"""油漆桶要填的范围（照 Photoshop，外加贴图特有的两种）：相近颜色、UV 岛、网格部件。结果是一张 0..1 的遮罩图，
交给滤镜框架的「贴图」写回方式铺上去。遮罩最大 MASK_SIZE 边长：更大的贴图按它放大（边缘柔一点，看不出来）。"""
from __future__ import annotations

import moderngl
import numba
import numpy as np

MASK_SIZE = 4096


def mask_size(texture_size: int) -> int:
    return int(min(int(texture_size), MASK_SIZE))


@numba.njit(cache=True)
def _flood(image, x, y, tolerance):
    """扫描线洪泛：只走和 (x, y) 连成一片、各通道相差都不超过 tolerance 的纹素（4 邻接），走不到的不看。"""
    h, w, c = image.shape
    out = np.zeros((h, w), np.float32)
    seen = np.zeros((h, w), np.bool_)
    ref = image[y, x].copy()
    stack_x = np.empty(h * 4 + 16, np.int64)
    stack_y = np.empty(h * 4 + 16, np.int64)
    top = 0
    stack_x[0] = x
    stack_y[0] = y
    top = 1
    while top > 0:
        top -= 1
        sx = stack_x[top]
        sy = stack_y[top]
        if seen[sy, sx]:
            continue
        # 往左找到这一段的头
        lx = sx
        while lx >= 0 and not seen[sy, lx]:
            ok = True
            for k in range(c):
                if abs(float(image[sy, lx, k]) - float(ref[k])) > tolerance:
                    ok = False
                    break
            if not ok:
                break
            lx -= 1
        lx += 1
        rx = lx
        while rx < w and not seen[sy, rx]:
            ok = True
            for k in range(c):
                if abs(float(image[sy, rx, k]) - float(ref[k])) > tolerance:
                    ok = False
                    break
            if not ok:
                break
            seen[sy, rx] = True
            out[sy, rx] = 1.0
            rx += 1
        # 上下两行里这一段下面的新段，各压一个起点
        for ny in (sy - 1, sy + 1):
            if ny < 0 or ny >= h:
                continue
            inside = False
            for nx in range(lx, rx):
                ok = not seen[ny, nx]
                if ok:
                    for k in range(c):
                        if abs(float(image[ny, nx, k]) - float(ref[k])) > tolerance:
                            ok = False
                            break
                if ok and not inside:
                    if top >= len(stack_x):
                        grown_x = np.empty(len(stack_x) * 2, np.int64)
                        grown_y = np.empty(len(stack_y) * 2, np.int64)
                        grown_x[:top] = stack_x[:top]
                        grown_y[:top] = stack_y[:top]
                        stack_x = grown_x
                        stack_y = grown_y
                    stack_x[top] = nx
                    stack_y[top] = ny
                    top += 1
                    inside = True
                elif not ok:
                    inside = False
    return out


def similar_mask(image: np.ndarray, x: int, y: int, tolerance: float, contiguous: bool) -> np.ndarray:
    """和 (x, y) 处颜色差不多（各通道相差都不超过 tolerance，按 0..1 算）的地方；contiguous 时只要和它连成一片的。
    图可以是 0..1 的浮点，也可以是 0..255 的 8 位。"""
    if np.asarray(image).dtype == np.uint8:
        image = np.ascontiguousarray(image)
        tolerance = float(tolerance) * 255.0 + 1e-3
    else:
        image = np.ascontiguousarray(image, np.float32)
        tolerance = float(tolerance) + 1e-6
    h, w = image.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    if contiguous:
        return _flood(image, x, y, tolerance)
    reference = image[y, x].astype(np.float32)
    return (np.abs(image.astype(np.float32) - reference).max(axis=-1) <= tolerance).astype(np.float32)


def triangle_mask(ctx, uv_triangles: np.ndarray, size: int, dilate: int = 2) -> np.ndarray:
    """把这些 UV 三角形 (n, 3, 2) 画成 size × size 的遮罩，再往外扩 dilate 个纹素（接缝外侧也盖上，缩小看不露边）。"""
    tris = np.ascontiguousarray(np.asarray(uv_triangles, np.float32).reshape(-1, 2))
    if len(tris) == 0:
        return np.zeros((size, size), np.float32)
    program = ctx.program(vertex_shader="""#version 330
in vec2 in_uv;
void main() { gl_Position = vec4(in_uv * 2.0 - 1.0, 0.0, 1.0); }
""", fragment_shader="""#version 330
out vec4 o;
void main() { o = vec4(1.0); }
""")
    target = ctx.texture((size, size), 1, dtype="f1")
    fbo = ctx.framebuffer([target])
    vbo = ctx.buffer(tris.tobytes())
    vao = ctx.vertex_array(program, [(vbo, "2f", "in_uv")])
    try:
        fbo.use()
        ctx.viewport = (0, 0, size, size)
        fbo.clear(0.0, 0.0, 0.0, 0.0)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        vao.render()
        # 细小的三角形可能一个像素中心都没盖到：把边也画一遍
        ctx.wireframe = True
        try:
            vao.render()
        finally:
            ctx.wireframe = False
        data = np.frombuffer(fbo.read(components=1, dtype="f1"), np.uint8).reshape(size, size)
    finally:
        for item in (vao, vbo, fbo, target, program):
            item.release()
    mask = data > 127
    if dilate > 0:
        from scipy import ndimage

        mask = ndimage.binary_dilation(mask, iterations=int(dilate))
    return mask.astype(np.float32)


def triangles_of_set(project, ts) -> list:
    """用这套贴图的每个物体：[(物体, 这套贴图的三角形序号 (k,))]。"""
    out = []
    for obj in project.objects:
        data = getattr(obj, "data", None)
        if data is None or not getattr(data, "has_uvs", True) or not getattr(obj, "visible", True):
            continue
        mats = np.asarray(getattr(data, "material_ids", []), np.int64)
        count = len(np.asarray(data.positions).reshape(-1, 3)) // 3
        if len(mats) == 0:
            mats = np.zeros(count, np.int64)
        sets = list(getattr(obj, "material_sets", []))
        wanted = [index for index, uid in enumerate(sets) if uid == ts.uid]
        if not wanted:
            continue
        tris = np.nonzero(np.isin(mats, wanted))[0]
        if len(tris):
            out.append((obj, tris))
    return out


def uv_triangle_at(project, ts, u: float, v: float):
    """UV 视图里点到的三角形：(物体, 三角形序号) 或 None。"""
    point = np.array([u, v], np.float64)
    for obj, tris in triangles_of_set(project, ts):
        corners = np.asarray(obj.data.uvs, np.float64).reshape(-1, 3, 2)[tris]
        a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
        v0, v1, v2 = c - a, b - a, point - a
        d00 = (v0 * v0).sum(1)
        d01 = (v0 * v1).sum(1)
        d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1)
        d21 = (v2 * v1).sum(1)
        denom = d00 * d11 - d01 * d01
        ok = np.abs(denom) > 1e-20
        safe = np.where(ok, denom, 1.0)
        s = (d11 * d20 - d01 * d21) / safe
        t = (d00 * d21 - d01 * d20) / safe
        inside = ok & (s >= -1e-6) & (t >= -1e-6) & (s + t <= 1.0 + 1e-6)
        hits = np.nonzero(inside)[0]
        if len(hits):
            return obj, int(tris[hits[0]])
    return None


def group_triangles(obj, tri: int, mode: str) -> np.ndarray:
    """和 tri 同一块（UV 岛或网格部件）的三角形的 UV (n, 3, 2)。"""
    from ..bake.parts import triangle_parts

    data = obj.data
    ids, _count = triangle_parts(data.positions, data.uvs, "UV_ISLAND" if mode == "ISLAND" else "PART")
    chosen = np.nonzero(ids == ids[int(tri)])[0]
    return np.asarray(data.uvs, np.float64).reshape(-1, 3, 2)[chosen]


def layer_image(engine, layer, plane: int, size: int) -> np.ndarray:
    """图层某种页在边长 size 那一级的样子（不合成别的层），(size, size, 通道) 浮点 0..1，行序和贴图一样（第一行是 v=0）。"""
    from ..doc.project import PLANE_FORMAT
    from .pagepool import PAGE

    fmt = PLANE_FORMAT[plane]
    channels = {"rgba8": 4, "rg16f": 2, "r8": 1}[fmt]
    dtype = np.float16 if fmt == "rg16f" else np.uint8
    out = np.zeros((size, size, channels), np.float32)
    store = engine.layers.stores.get(layer.uid)
    if store is None or plane not in store.planes:
        return out
    levels = store.planes[plane].levels
    full = levels[0].size * PAGE
    level = min(len(levels) - 1, max(0, int(round(np.log2(max(1, full // size))))))
    pages = levels[level].pages
    keys = list(pages)
    raws = engine.cache.fetch_bytes([pages[key] for key in keys]) if keys else []
    for (x, y), raw in zip(keys, raws):
        tile = np.frombuffer(raw, dtype).reshape(PAGE, PAGE, channels).astype(np.float32)
        if fmt != "rg16f":
            tile = tile / 255.0
        y0, x0 = y * PAGE, x * PAGE
        if y0 < size and x0 < size:
            out[y0:y0 + PAGE, x0:x0 + PAGE] = tile[:size - y0, :size - x0]
    return out
