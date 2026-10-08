"""自动展开 UV（思路同 Blender 的「智能 UV 投射」）：

1. 按法线角度长块：从一个三角形出发做广度优先，邻居的法线和起点法线的夹角不超过限制就并进来。
2. 太小的块并到法线最接近的邻块里（放宽一点角度），减少碎接缝。
3. 每块沿面积加权的平均法线投影到平面（有三角形翻面时改用起点法线），世界单位不缩放，所以各块的贴图密度一致。
4. 转到包围矩形面积最小的方向（凸包 + 旋转卡壳），再从大到小用天际线左下角法排进方格（可以转 90°），
   二分查找能放下的最大比例。块与块之间留出间距（贴图像素），给绘制的接缝扩边用。
5. 不同材质各用一张贴图，分开排，互不影响。

全部 numba 编译，几百万个三角形也只要几秒。结果是逐角点 UV (T*3, 2)，接缝处同一个顶点在不同块里 UV 不同。
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _face_neighbors(tris, vertex_count):
    """每个三角形三条边（k→k+1）对面的三角形 (T,3)，没有为 -1。"""
    counts = np.zeros(vertex_count + 1, np.int64)
    for t in range(tris.shape[0]):
        for k in range(3):
            counts[tris[t, k] + 1] += 1
    offsets = np.cumsum(counts)
    fill = offsets[:-1].copy()
    targets = np.empty(tris.shape[0] * 3, np.int64)
    faces = np.empty(tris.shape[0] * 3, np.int64)
    for t in range(tris.shape[0]):
        for k in range(3):
            a = tris[t, k]
            targets[fill[a]] = tris[t, (k + 1) % 3]
            faces[fill[a]] = t
            fill[a] += 1
    nb = np.full((tris.shape[0], 3), -1, np.int64)
    for t in range(tris.shape[0]):
        for k in range(3):
            a = tris[t, k]
            b = tris[t, (k + 1) % 3]
            for e in range(offsets[b], offsets[b + 1]):
                if targets[e] == a:
                    nb[t, k] = faces[e]
                    break
    return nb


@njit(cache=True, nogil=True)
def _grow_charts(normals, mats, nb, cos_limit):
    count_t = normals.shape[0]
    chart = np.full(count_t, -1, np.int64)
    ref = np.zeros((count_t, 3), np.float64)
    queue = np.empty(count_t, np.int64)
    count = 0
    for s in range(count_t):
        if chart[s] >= 0:
            continue
        c = count
        count += 1
        chart[s] = c
        rx, ry, rz = normals[s, 0], normals[s, 1], normals[s, 2]
        ref[c, 0], ref[c, 1], ref[c, 2] = rx, ry, rz
        head = 0
        tail = 1
        queue[0] = s
        while head < tail:
            f = queue[head]
            head += 1
            for k in range(3):
                g = nb[f, k]
                if g < 0 or chart[g] >= 0 or mats[g] != mats[s]:
                    continue
                if normals[g, 0] * rx + normals[g, 1] * ry + normals[g, 2] * rz >= cos_limit:
                    chart[g] = c
                    queue[tail] = g
                    tail += 1
    return chart, ref[:count].copy(), count


@njit(cache=True, nogil=True)
def _absorb_small(chart, count, ref, nb, normals, mats, min_faces, cos_relaxed):
    sizes = np.zeros(count, np.int64)
    for f in range(chart.shape[0]):
        sizes[chart[f]] += 1
    for _pass in range(16):
        changed = False
        for f in range(chart.shape[0]):
            c = chart[f]
            if sizes[c] >= min_faces:
                continue
            best = -1
            best_d = cos_relaxed
            for k in range(3):
                g = nb[f, k]
                if g < 0 or mats[g] != mats[f]:
                    continue
                cg = chart[g]
                if cg == c or sizes[cg] < min_faces:
                    continue
                d = normals[f, 0] * ref[cg, 0] + normals[f, 1] * ref[cg, 1] + normals[f, 2] * ref[cg, 2]
                if d > best_d:
                    best_d = d
                    best = cg
            if best >= 0:
                sizes[c] -= 1
                sizes[best] += 1
                chart[f] = best
                changed = True
        if not changed:
            break
    # 重新编号，去掉空块
    remap = np.full(count, -1, np.int64)
    n = 0
    for c in range(count):
        if sizes[c] > 0:
            remap[c] = n
            n += 1
    out_ref = np.empty((n, 3), np.float64)
    for c in range(count):
        if remap[c] >= 0:
            out_ref[remap[c]] = ref[c]
    for f in range(chart.shape[0]):
        chart[f] = remap[chart[f]]
    return chart, out_ref, n


@njit(cache=True, nogil=True)
def _hull_min_rect(xs, ys):
    """点集的凸包，再找包围矩形面积最小的方向。返回角度（把点旋转 -angle 后矩形与坐标轴对齐）。"""
    n = xs.shape[0]
    if n < 3:
        return 0.0
    order = np.argsort(xs + ys * 1e-9)
    hx = np.empty(2 * n, np.float64)
    hy = np.empty(2 * n, np.float64)
    k = 0
    for i in range(n):
        px, py = xs[order[i]], ys[order[i]]
        while k >= 2 and (hx[k - 1] - hx[k - 2]) * (py - hy[k - 2]) - (hy[k - 1] - hy[k - 2]) * (px - hx[k - 2]) <= 0.0:
            k -= 1
        hx[k] = px
        hy[k] = py
        k += 1
    lower = k + 1
    for i in range(n - 2, -1, -1):
        px, py = xs[order[i]], ys[order[i]]
        while k >= lower and (hx[k - 1] - hx[k - 2]) * (py - hy[k - 2]) - (hy[k - 1] - hy[k - 2]) * (px - hx[k - 2]) <= 0.0:
            k -= 1
        hx[k] = px
        hy[k] = py
        k += 1
    h = k - 1
    if h < 3:
        return 0.0
    best_area = np.inf
    best_angle = 0.0
    for i in range(h):
        ex = hx[i + 1] - hx[i]
        ey = hy[i + 1] - hy[i]
        length = math.sqrt(ex * ex + ey * ey)
        if length <= 0.0:
            continue
        ux, uy = ex / length, ey / length
        lo_u = np.inf
        hi_u = -np.inf
        lo_v = np.inf
        hi_v = -np.inf
        for j in range(h):
            pu = hx[j] * ux + hy[j] * uy
            pv = -hx[j] * uy + hy[j] * ux
            lo_u = min(lo_u, pu)
            hi_u = max(hi_u, pu)
            lo_v = min(lo_v, pv)
            hi_v = max(hi_v, pv)
        area = (hi_u - lo_u) * (hi_v - lo_v)
        if area < best_area:
            best_area = area
            best_angle = math.atan2(uy, ux)
    return best_angle


@njit(cache=True, nogil=True)
def _project_charts(verts, tris, normals, areas, chart, ref, count):
    """每块投影到平面并对齐到最小包围矩形。返回逐角点坐标 (T*3,2)（世界单位，每块左下角在原点）和每块的宽高。"""
    count_t = tris.shape[0]
    counts = np.zeros(count + 1, np.int64)
    for f in range(count_t):
        counts[chart[f] + 1] += 1
    offsets = np.cumsum(counts)
    fill = offsets[:-1].copy()
    members = np.empty(count_t, np.int64)
    for f in range(count_t):
        c = chart[f]
        members[fill[c]] = f
        fill[c] += 1
    uv = np.zeros((count_t * 3, 2), np.float64)
    size = np.zeros((count, 2), np.float64)
    for c in range(count):
        s, e = offsets[c], offsets[c + 1]
        nx = 0.0
        ny = 0.0
        nz = 0.0
        for i in range(s, e):
            f = members[i]
            nx += normals[f, 0] * areas[f]
            ny += normals[f, 1] * areas[f]
            nz += normals[f, 2] * areas[f]
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 0.0:
            nx, ny, nz = nx / length, ny / length, nz / length
        else:
            nx, ny, nz = ref[c, 0], ref[c, 1], ref[c, 2]
        flipped = False
        for i in range(s, e):
            f = members[i]
            if normals[f, 0] * nx + normals[f, 1] * ny + normals[f, 2] * nz < 0.05:
                flipped = True
                break
        if flipped:
            nx, ny, nz = ref[c, 0], ref[c, 1], ref[c, 2]
        # 平面上的一组基
        if abs(nx) < 0.9:
            ux, uy, uz = 0.0, -nz, ny
        else:
            ux, uy, uz = nz, 0.0, -nx
        ul = math.sqrt(ux * ux + uy * uy + uz * uz)
        ux, uy, uz = ux / ul, uy / ul, uz / ul
        wx = ny * uz - nz * uy
        wy = nz * ux - nx * uz
        wz = nx * uy - ny * ux
        m = (e - s) * 3
        xs = np.empty(m, np.float64)
        ys = np.empty(m, np.float64)
        j = 0
        for i in range(s, e):
            f = members[i]
            for k in range(3):
                v = tris[f, k]
                xs[j] = verts[v, 0] * ux + verts[v, 1] * uy + verts[v, 2] * uz
                ys[j] = verts[v, 0] * wx + verts[v, 1] * wy + verts[v, 2] * wz
                j += 1
        angle = _hull_min_rect(xs, ys)
        ca, sa = math.cos(angle), math.sin(angle)
        lo_x = np.inf
        lo_y = np.inf
        hi_x = -np.inf
        hi_y = -np.inf
        for q in range(m):
            rx = xs[q] * ca + ys[q] * sa
            ry = -xs[q] * sa + ys[q] * ca
            xs[q] = rx
            ys[q] = ry
            lo_x = min(lo_x, rx)
            lo_y = min(lo_y, ry)
            hi_x = max(hi_x, rx)
            hi_y = max(hi_y, ry)
        wide = (hi_x - lo_x) >= (hi_y - lo_y)
        j = 0
        for i in range(s, e):
            f = members[i]
            for k in range(3):
                if wide:
                    uv[f * 3 + k, 0] = xs[j] - lo_x
                    uv[f * 3 + k, 1] = ys[j] - lo_y
                else:                       # 竖长的块先转成横的，排的时候再决定转不转
                    uv[f * 3 + k, 0] = ys[j] - lo_y
                    uv[f * 3 + k, 1] = hi_x - xs[j]
                j += 1
        if wide:
            size[c, 0] = hi_x - lo_x
            size[c, 1] = hi_y - lo_y
        else:
            size[c, 0] = hi_y - lo_y
            size[c, 1] = hi_x - lo_x
    return uv, size


@njit(cache=True, nogil=True)
def _skyline(size, order, scale, pad, width, height, allow_rotate, out_x, out_y, out_rot):
    """天际线左下角法：每块放到让顶边最低的位置（可以转 90°）。放得下返回 True。"""
    n = order.shape[0]
    cap = n + 2
    sx = np.empty(cap, np.float64)      # 天际线各段的起点、宽度、高度
    sw = np.empty(cap, np.float64)
    sh = np.empty(cap, np.float64)
    nx = np.empty(cap + 2, np.float64)
    nw = np.empty(cap + 2, np.float64)
    nh = np.empty(cap + 2, np.float64)
    segs = 1
    sx[0] = 0.0
    sw[0] = width
    sh[0] = 0.0
    for idx in range(n):
        c = order[idx]
        best_top = np.inf
        best_x = 0.0
        best_y = 0.0
        best_rot = False
        best_waste = np.inf
        for rot in range(2 if allow_rotate else 1):
            if rot == 0:
                w = size[c, 0] * scale + 2.0 * pad
                h = size[c, 1] * scale + 2.0 * pad
            else:
                w = size[c, 1] * scale + 2.0 * pad
                h = size[c, 0] * scale + 2.0 * pad
            for i in range(segs):
                x = sx[i]
                if x + w > width + 1e-9:
                    break
                reach = x + w
                y = 0.0
                j = i
                while j < segs and sx[j] < reach - 1e-12:
                    y = max(y, sh[j])
                    j += 1
                top = y + h
                if top > height + 1e-9:
                    continue
                waste = 0.0
                j = i
                while j < segs and sx[j] < reach - 1e-12:
                    span = min(sx[j] + sw[j], reach) - max(sx[j], x)
                    waste += (y - sh[j]) * span
                    j += 1
                if top < best_top - 1e-9 or (abs(top - best_top) <= 1e-9 and waste < best_waste):
                    best_top = top
                    best_x = x
                    best_y = y
                    best_rot = rot == 1
                    best_waste = waste
        if best_top == np.inf:
            return False
        if best_rot:
            w = size[c, 1] * scale + 2.0 * pad
        else:
            w = size[c, 0] * scale + 2.0 * pad
        out_x[c] = best_x + pad
        out_y[c] = best_y + pad
        out_rot[c] = best_rot
        # 更新天际线：[left, right) 变成高 best_top 的一段
        left = best_x
        right = best_x + w
        m = 0
        inserted = False
        for i in range(segs):
            a = sx[i]
            b = sx[i] + sw[i]
            if b <= left + 1e-12 or a >= right - 1e-12:
                if a >= right - 1e-12 and not inserted:
                    nx[m] = left
                    nw[m] = right - left
                    nh[m] = best_top
                    m += 1
                    inserted = True
                nx[m] = a
                nw[m] = b - a
                nh[m] = sh[i]
                m += 1
                continue
            if a < left:
                nx[m] = a
                nw[m] = left - a
                nh[m] = sh[i]
                m += 1
            if not inserted:
                nx[m] = left
                nw[m] = right - left
                nh[m] = best_top
                m += 1
                inserted = True
            if b > right:
                nx[m] = right
                nw[m] = b - right
                nh[m] = sh[i]
                m += 1
        if not inserted:
            nx[m] = left
            nw[m] = right - left
            nh[m] = best_top
            m += 1
        # 合并相邻同高的段
        segs = 0
        for i in range(m):
            if segs > 0 and abs(sh[segs - 1] - nh[i]) <= 1e-9:
                sw[segs - 1] += nw[i]
            elif segs < cap:
                sx[segs] = nx[i]
                sw[segs] = nw[i]
                sh[segs] = nh[i]
                segs += 1
    return True


def _pack(size: np.ndarray, charts: np.ndarray, resolution: int, pad: float,
          allow_rotate: bool = True) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """把一组块排进 resolution×resolution 的方格：几种排序各试一遍，每种二分查找能放下的最大比例
    （贴图像素/世界单位），取最大的。"""
    sub = np.ascontiguousarray(size[charts])
    res = float(resolution)
    total = float((sub[:, 0] * sub[:, 1]).sum()) or 1e-12
    orders = [np.lexsort((-sub[:, 1], -np.maximum(sub[:, 0], sub[:, 1]))),
              np.argsort(-sub[:, 1], kind="stable"),
              np.argsort(-(sub[:, 0] * sub[:, 1]), kind="stable"),
              np.argsort(-sub[:, 0], kind="stable")]
    best = None
    for order in orders:
        out_x = np.zeros(len(charts))
        out_y = np.zeros(len(charts))
        out_rot = np.zeros(len(charts), np.bool_)
        hi = math.sqrt(res * res / total) * 1.05
        lo = 0.0
        if best is not None and not _skyline(sub, order, best[0], pad, res, res, allow_rotate, out_x, out_y, out_rot):
            continue                    # 连当前最好的比例都放不下，不用细找
        if best is not None:
            lo = best[0]
        for _ in range(30):
            mid = (lo + hi) * 0.5
            if _skyline(sub, order, mid, pad, res, res, allow_rotate, out_x, out_y, out_rot):
                lo = mid
            else:
                hi = mid
        if lo <= 0.0:
            lo = 1e-9
        _skyline(sub, order, lo, pad, res, res, allow_rotate, out_x, out_y, out_rot)
        if best is None or lo > best[0]:
            best = (lo, out_x, out_y, out_rot)
    return best


def smart_unwrap(vertices: np.ndarray, triangles: np.ndarray, material_ids: np.ndarray | None = None,
                 angle_limit: float = 45.0, min_chart_faces: int = 32, margin_texels: float = 8.0,
                 resolution: int = 4096, allow_rotate: bool = True) -> tuple[np.ndarray, dict]:
    """返回 (逐角点 UV (T*3,2) float32, 统计)。angle_limit：一块里法线和起点法线的最大夹角（度）。
    margin_texels：按 resolution 大小的贴图算，块与块的间距（像素，每边一半）。"""
    started = time.perf_counter()
    verts = np.ascontiguousarray(vertices, np.float64)
    tris = np.ascontiguousarray(triangles, np.int64)
    count_t = len(tris)
    if count_t == 0:
        return np.zeros((0, 2), np.float32), {"charts": 0}
    mats = np.zeros(count_t, np.int64) if material_ids is None else np.ascontiguousarray(material_ids, np.int64)
    cross = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    double_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(double_area, 1e-30)[:, None]
    normals[double_area <= 1e-30] = (0.0, 0.0, 1.0)
    areas = double_area * 0.5
    nb = _face_neighbors(tris, len(verts))
    limit = math.radians(float(np.clip(angle_limit, 1.0, 89.0)))
    chart, ref, count = _grow_charts(normals, mats, nb, math.cos(limit))
    relaxed = math.cos(min(limit + math.radians(20.0), math.radians(85.0)))
    chart, ref, count = _absorb_small(chart, count, ref, nb, normals, mats, int(min_chart_faces), relaxed)
    uv, size = _project_charts(verts, tris, normals, areas, chart, ref, count)
    pad = max(0.5, float(margin_texels) * 0.5)
    out = np.zeros((count_t * 3, 2), np.float32)
    chart_mat = np.zeros(count, np.int64)
    chart_mat[chart] = mats
    scales = {}
    for material in np.unique(mats):
        charts = np.nonzero(chart_mat == material)[0]
        scale, ox, oy, rot = _pack(size, charts, int(resolution), pad, bool(allow_rotate))
        scales[int(material)] = scale
        offset_x = np.zeros(count)
        offset_y = np.zeros(count)
        rotated = np.zeros(count, bool)
        offset_x[charts] = ox
        offset_y[charts] = oy
        rotated[charts] = rot
        faces = np.nonzero(mats == material)[0]
        corners = (faces[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
        cc = np.repeat(chart[faces], 3)
        lx = uv[corners, 0]
        ly = uv[corners, 1]
        turn = rotated[cc]
        # 转 90°：(x, y) → (高 - y, x)，宽高互换；绕向不变
        px = np.where(turn, size[cc, 1] - ly, lx)
        py = np.where(turn, lx, ly)
        out[corners, 0] = ((px * scale + offset_x[cc]) / resolution).astype(np.float32)
        out[corners, 1] = ((py * scale + offset_y[cc]) / resolution).astype(np.float32)
    total_area = float(areas.sum())
    uv_area = 0.0
    if count_t:
        u = out.reshape(-1, 3, 2).astype(np.float64)
        uv_area = float(np.abs((u[:, 1, 0] - u[:, 0, 0]) * (u[:, 2, 1] - u[:, 0, 1])
                               - (u[:, 2, 0] - u[:, 0, 0]) * (u[:, 1, 1] - u[:, 0, 1])).sum() * 0.5)
    stats = {"charts": int(count), "unwrap_ms": round((time.perf_counter() - started) * 1000.0, 1),
             "coverage": round(uv_area / max(1, len(np.unique(mats))), 4), "surface_area": total_area,
             "texels_per_unit": {k: v for k, v in scales.items()}}
    return out, stats
