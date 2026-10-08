"""体素重构（类似 ZBrush 的 DynaMesh）：任意三角网格 → 带符号距离场 → Surface Nets 等值面 → 均匀、封闭的新网格。

做法（稀疏分块，只在表面附近的 8³ 小块里存数据，所以分辨率能开到 2048）：
1. 标出离网格不超过 band 的小块，把三角形分到它们碰到的小块里。
2. 每个小块里的格点算到网格的最近距离（多线程，一个小块一个线程，没有写冲突）。
3. 内外靠射线绕数判断：沿 x、y、z 三个方向的格线各做一次三角形光栅化，记下穿过点和穿过方向，
   格点的绕数不为 0 就在里面。重叠的几块自然合成一块（并集），里外翻转的模型也能对。
   一条格线上进出不配对（模型有洞）就作废，由另外两个方向投票；都作废时才看最近三角形的法线。
4. Surface Nets：内外不同的格子放一个顶点（各边零点的平均），每条内外不同的格边连成四边形，再按短对角线拆三角形。
5. 切向放松几次，让三角形更均匀而不缩水。遮罩和材质从原网格最近的三角形带过来。

思路参考 SculptGL 的体素重构（MIT 许可）；三角形内外判定用一致的边函数加平局规则，共边的格点只算一次。
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

from .topology import SculptMesh, _split_fans

B = 8                     # 小块边长（格点数）
BAND = 1.8                # 窄带宽度（体素）：要大于格子对角线（√3 ≈ 1.73），内外不同的格子角点才都有距离
PAD = 3                   # 包围盒外留的空格（体素）
MAX_RESOLUTION = 2048


# ====================================================================== 几何小工具
@njit(cache=True, nogil=True, inline="always")
def _closest_on_triangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
    """点到三角形的最近点（Ericson《实时碰撞检测》5.1.5），返回最近点和重心坐标 (v, w)。"""
    abx, aby, abz = bx - ax, by - ay, bz - az
    acx, acy, acz = cx - ax, cy - ay, cz - az
    apx, apy, apz = px - ax, py - ay, pz - az
    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return ax, ay, az, 0.0, 0.0
    bpx, bpy, bpz = px - bx, py - by, pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bx, by, bz, 1.0, 0.0
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return ax + abx * v, ay + aby * v, az + abz * v, v, 0.0
    cpx, cpy, cpz = px - cx, py - cy, pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cx, cy, cz, 0.0, 1.0
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return ax + acx * w, ay + acy * w, az + acz * w, 0.0, w
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return bx + (cx - bx) * w, by + (cy - by) * w, bz + (cz - bz) * w, 1.0 - w, w
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return ax + abx * v + acx * w, ay + aby * v + acy * w, az + abz * v + acz * w, v, w


@njit(cache=True, nogil=True, inline="always")
def _tri_range(verts, tris, t, origin, h, dims, band, out):
    """三角形外扩 band 后覆盖的格点范围（闭区间），写进 out[0..5]。"""
    for ax in range(3):
        a = verts[tris[t, 0], ax]
        b = verts[tris[t, 1], ax]
        c = verts[tris[t, 2], ax]
        lo = min(a, min(b, c)) - band
        hi = max(a, max(b, c)) + band
        i0 = int(math.ceil((lo - origin[ax]) / h))
        i1 = int(math.floor((hi - origin[ax]) / h))
        out[2 * ax] = max(i0, 0)
        out[2 * ax + 1] = min(i1, dims[ax] - 1)


# ====================================================================== 1. 分块
@njit(cache=True, nogil=True)
def _mark_blocks(verts, tris, origin, h, dims, band, bdims):
    flags = np.zeros((bdims[0], bdims[1], bdims[2]), np.uint8)
    rng = np.empty(6, np.int64)
    for t in range(tris.shape[0]):
        _tri_range(verts, tris, t, origin, h, dims, band, rng)
        for bi in range(rng[0] // B, rng[1] // B + 1):
            for bj in range(rng[2] // B, rng[3] // B + 1):
                for bk in range(rng[4] // B, rng[5] // B + 1):
                    flags[bi, bj, bk] = 1
    return flags


@njit(cache=True, nogil=True)
def _bin_triangles(verts, tris, origin, h, dims, band, block_id, block_count):
    counts = np.zeros(block_count + 1, np.int64)
    rng = np.empty(6, np.int64)
    for t in range(tris.shape[0]):
        _tri_range(verts, tris, t, origin, h, dims, band, rng)
        for bi in range(rng[0] // B, rng[1] // B + 1):
            for bj in range(rng[2] // B, rng[3] // B + 1):
                for bk in range(rng[4] // B, rng[5] // B + 1):
                    counts[block_id[bi, bj, bk] + 1] += 1
    offsets = np.cumsum(counts)
    fill = offsets[:-1].copy()
    items = np.empty(offsets[-1], np.int32)
    for t in range(tris.shape[0]):
        _tri_range(verts, tris, t, origin, h, dims, band, rng)
        for bi in range(rng[0] // B, rng[1] // B + 1):
            for bj in range(rng[2] // B, rng[3] // B + 1):
                for bk in range(rng[4] // B, rng[5] // B + 1):
                    b = block_id[bi, bj, bk]
                    items[fill[b]] = t
                    fill[b] += 1
    return offsets, items


# ====================================================================== 2. 距离
@njit(cache=True, parallel=True)
def _block_distances(verts, tris, origin, h, dims, band, coords, offsets, items):
    n = coords.shape[0]
    dist = np.full((n, B, B, B), np.inf, np.float32)
    near = np.full((n, B, B, B), -1, np.int32)
    for b in prange(n):
        rng = np.empty(6, np.int64)
        bi = coords[b, 0] * B
        bj = coords[b, 1] * B
        bk = coords[b, 2] * B
        for e in range(offsets[b], offsets[b + 1]):
            t = items[e]
            _tri_range(verts, tris, t, origin, h, dims, band, rng)
            i0, i1 = max(rng[0], bi), min(rng[1], bi + B - 1)
            j0, j1 = max(rng[2], bj), min(rng[3], bj + B - 1)
            k0, k1 = max(rng[4], bk), min(rng[5], bk + B - 1)
            a, bb, c = tris[t, 0], tris[t, 1], tris[t, 2]
            ax, ay, az = verts[a, 0], verts[a, 1], verts[a, 2]
            bx, by, bz = verts[bb, 0], verts[bb, 1], verts[bb, 2]
            cx, cy, cz = verts[c, 0], verts[c, 1], verts[c, 2]
            for i in range(i0, i1 + 1):
                px = origin[0] + i * h
                for j in range(j0, j1 + 1):
                    py = origin[1] + j * h
                    for k in range(k0, k1 + 1):
                        pz = origin[2] + k * h
                        qx, qy, qz, _v, _w = _closest_on_triangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz)
                        dx, dy, dz = px - qx, py - qy, pz - qz
                        d = math.sqrt(dx * dx + dy * dy + dz * dz)
                        if d < dist[b, i - bi, j - bj, k - bk]:
                            dist[b, i - bi, j - bj, k - bk] = d
                            near[b, i - bi, j - bj, k - bk] = t
    return dist, near


# ====================================================================== 3. 内外（射线绕数）
@njit(cache=True, nogil=True)
def _crossings(verts, tris, origin, h, dims, axis, counting, cursor, pos, delta):
    """沿 axis 方向的每条格线和三角形的交点。counting=True 时只数个数（cursor[c+1] += 1），
    否则按 cursor 写入位置和方向（+1 进入，-1 离开，以朝外的法线为准）。"""
    u = (axis + 1) % 3
    v = (axis + 2) % 3
    nv = dims[v]
    for t in range(tris.shape[0]):
        ia, ib, ic = tris[t, 0], tris[t, 1], tris[t, 2]
        au, av, aa = verts[ia, u], verts[ia, v], verts[ia, axis]
        bu, bv, ba = verts[ib, u], verts[ib, v], verts[ib, axis]
        cu, cv, ca = verts[ic, u], verts[ic, v], verts[ic, axis]
        d = (bu - au) * (cv - av) - (cu - au) * (bv - av)
        if d == 0.0:
            continue
        ccw = d > 0.0
        iu0 = max(int(math.ceil((min(au, min(bu, cu)) - origin[u]) / h)), 0)
        iu1 = min(int(math.floor((max(au, max(bu, cu)) - origin[u]) / h)), dims[u] - 1)
        iv0 = max(int(math.ceil((min(av, min(bv, cv)) - origin[v]) / h)), 0)
        iv1 = min(int(math.floor((max(av, max(bv, cv)) - origin[v]) / h)), dims[v] - 1)
        for iu in range(iu0, iu1 + 1):
            pu = origin[u] + iu * h
            for iv in range(iv0, iv1 + 1):
                pv = origin[v] + iv * h
                inside = True
                for edge in range(3):
                    if edge == 0:
                        x0, y0, x1, y1 = au, av, bu, bv
                    elif edge == 1:
                        x0, y0, x1, y1 = bu, bv, cu, cv
                    else:
                        x0, y0, x1, y1 = cu, cv, au, av
                    forward = x0 < x1 or (x0 == x1 and y0 < y1)
                    if not forward:
                        x0, y0, x1, y1 = x1, y1, x0, y0
                    # 共边的两个三角形用同一组端点、同一个式子算，结果逐位相同；等于 0 时一律算正（一致的微扰）
                    e = (x1 - x0) * (pv - y0) - (y1 - y0) * (pu - x0)
                    if (e >= 0.0) != (forward == ccw):
                        inside = False
                        break
                if not inside:
                    continue
                c = iu * nv + iv
                if counting:
                    cursor[c + 1] += 1
                    continue
                l1 = ((pu - au) * (cv - av) - (cu - au) * (pv - av)) / d
                l2 = ((bu - au) * (pv - av) - (pu - au) * (bv - av)) / d
                l1 = min(max(l1, 0.0), 1.0)
                l2 = min(max(l2, 0.0), 1.0 - l1)
                slot = cursor[c]
                cursor[c] += 1
                pos[slot] = (1.0 - l1 - l2) * aa + l1 * ba + l2 * ca
                delta[slot] = -1 if ccw else 1


@njit(cache=True, parallel=True)
def _sort_columns(offsets, pos, delta):
    for c in prange(offsets.shape[0] - 1):
        s, e = offsets[c], offsets[c + 1]
        for i in range(s + 1, e):
            p = pos[i]
            d = delta[i]
            j = i - 1
            while j >= s and pos[j] > p:
                pos[j + 1] = pos[j]
                delta[j + 1] = delta[j]
                j -= 1
            pos[j + 1] = p
            delta[j + 1] = d


@njit(cache=True, parallel=True)
def _vote_axis(coords, origin, h, dims, axis, offsets, pos, delta, votes, valid):
    u = (axis + 1) % 3
    v = (axis + 2) % 3
    for b in prange(coords.shape[0]):
        base_a = coords[b, axis] * B
        base_u = coords[b, u] * B
        base_v = coords[b, v] * B
        for lu in range(B):
            gu = base_u + lu
            if gu >= dims[u]:
                continue
            for lv in range(B):
                gv = base_v + lv
                if gv >= dims[v]:
                    continue
                c = gu * dims[v] + gv
                s, e = offsets[c], offsets[c + 1]
                total = 0
                for k in range(s, e):
                    total += delta[k]
                ok = total == 0
                k = s
                w = 0
                for la in range(B):
                    x = origin[axis] + (base_a + la) * h
                    while k < e and pos[k] < x:
                        w += delta[k]
                        k += 1
                    if axis == 0:
                        li, lj, lk = la, lu, lv
                    elif axis == 1:
                        li, lj, lk = lv, la, lu
                    else:
                        li, lj, lk = lu, lv, la
                    if ok:
                        valid[b, li, lj, lk] += 1
                        if w != 0:
                            votes[b, li, lj, lk] += 1


@njit(cache=True, parallel=True)
def _resolve_inside(verts, tris, origin, h, coords, votes, valid, near):
    n = coords.shape[0]
    inside = np.zeros((n, B, B, B), np.uint8)
    undecided = np.zeros(n, np.int64)
    for b in prange(n):
        for li in range(B):
            for lj in range(B):
                for lk in range(B):
                    va = int(valid[b, li, lj, lk])
                    vo = int(votes[b, li, lj, lk])
                    if va > 0 and vo * 2 != va:
                        inside[b, li, lj, lk] = 1 if vo * 2 > va else 0
                        continue
                    undecided[b] += 1
                    t = near[b, li, lj, lk]
                    if t < 0:
                        continue
                    px = origin[0] + (coords[b, 0] * B + li) * h
                    py = origin[1] + (coords[b, 1] * B + lj) * h
                    pz = origin[2] + (coords[b, 2] * B + lk) * h
                    a, bb, c = tris[t, 0], tris[t, 1], tris[t, 2]
                    ax, ay, az = verts[a, 0], verts[a, 1], verts[a, 2]
                    bx, by, bz = verts[bb, 0], verts[bb, 1], verts[bb, 2]
                    cx, cy, cz = verts[c, 0], verts[c, 1], verts[c, 2]
                    qx, qy, qz, _v, _w = _closest_on_triangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz)
                    nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay)
                    ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az)
                    nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                    if (px - qx) * nx + (py - qy) * ny + (pz - qz) * nz < 0.0:
                        inside[b, li, lj, lk] = 1
    return inside, undecided.sum()


# ====================================================================== 4. Surface Nets
@njit(cache=True, nogil=True, inline="always")
def _lookup(block_id, bdims, i, j, k):
    bi, bj, bk = i // B, j // B, k // B
    if bi >= bdims[0] or bj >= bdims[1] or bk >= bdims[2]:
        return -1
    return block_id[bi, bj, bk]


@njit(cache=True, parallel=True)
def _count_cells(coords, block_id, bdims, dims, inside):
    n = coords.shape[0]
    counts = np.zeros(n, np.int64)
    flags = np.zeros((n, B, B, B), np.uint8)
    for b in prange(n):
        for li in range(B):
            i = coords[b, 0] * B + li
            if i + 1 >= dims[0]:
                continue
            for lj in range(B):
                j = coords[b, 1] * B + lj
                if j + 1 >= dims[1]:
                    continue
                for lk in range(B):
                    k = coords[b, 2] * B + lk
                    if k + 1 >= dims[2]:
                        continue
                    seen_in = False
                    seen_out = False
                    broken = False
                    for c in range(8):
                        ci = i + (c & 1)
                        cj = j + ((c >> 1) & 1)
                        ck = k + ((c >> 2) & 1)
                        cb = _lookup(block_id, bdims, ci, cj, ck)
                        if cb < 0:
                            broken = True
                            break
                        if inside[cb, ci - (ci // B) * B, cj - (cj // B) * B, ck - (ck // B) * B]:
                            seen_in = True
                        else:
                            seen_out = True
                    if not broken and seen_in and seen_out:
                        flags[b, li, lj, lk] = 1
                        counts[b] += 1
    return counts, flags


@njit(cache=True, parallel=True)
def _place_vertices(coords, block_id, bdims, origin, h, inside, dist, near, flags, base, total):
    n = coords.shape[0]
    out = np.empty((total, 3), np.float32)
    vnear = np.empty(total, np.int32)
    vidx = np.full((n, B, B, B), -1, np.int32)
    for b in prange(n):
        slot = base[b]
        cin = np.empty(8, np.uint8)
        cd = np.empty(8, np.float64)
        for li in range(B):
            for lj in range(B):
                for lk in range(B):
                    if not flags[b, li, lj, lk]:
                        continue
                    i = coords[b, 0] * B + li
                    j = coords[b, 1] * B + lj
                    k = coords[b, 2] * B + lk
                    best = -1
                    best_d = np.inf
                    for c in range(8):
                        ci = i + (c & 1)
                        cj = j + ((c >> 1) & 1)
                        ck = k + ((c >> 2) & 1)
                        cb = _lookup(block_id, bdims, ci, cj, ck)
                        oi, oj, ok = ci - (ci // B) * B, cj - (cj // B) * B, ck - (ck // B) * B
                        cin[c] = inside[cb, oi, oj, ok]
                        d = dist[cb, oi, oj, ok]
                        cd[c] = min(d, 4.0 * h)
                        if d < best_d and near[cb, oi, oj, ok] >= 0:
                            best_d = d
                            best = near[cb, oi, oj, ok]
                    sx = 0.0
                    sy = 0.0
                    sz = 0.0
                    m = 0
                    for e in range(12):
                        # 12 条边：沿 x 的 4 条、沿 y 的 4 条、沿 z 的 4 条（角点编号 bit0=x、bit1=y、bit2=z）
                        q = e & 3
                        if e < 4:
                            c0 = ((q & 1) << 1) | ((q >> 1) << 2)
                            c1 = c0 | 1
                        elif e < 8:
                            c0 = (q & 1) | ((q >> 1) << 2)
                            c1 = c0 | 2
                        else:
                            c0 = (q & 1) | ((q >> 1) << 1)
                            c1 = c0 | 4
                        if cin[c0] == cin[c1]:
                            continue
                        d0 = cd[c0]
                        d1 = cd[c1]
                        t = 0.5 if d0 + d1 <= 0.0 else d0 / (d0 + d1)
                        x0, y0, z0 = c0 & 1, (c0 >> 1) & 1, (c0 >> 2) & 1
                        x1, y1, z1 = c1 & 1, (c1 >> 1) & 1, (c1 >> 2) & 1
                        sx += x0 + (x1 - x0) * t
                        sy += y0 + (y1 - y0) * t
                        sz += z0 + (z1 - z0) * t
                        m += 1
                    if m == 0:
                        sx, sy, sz, m = 0.5, 0.5, 0.5, 1
                    out[slot, 0] = origin[0] + (i + sx / m) * h
                    out[slot, 1] = origin[1] + (j + sy / m) * h
                    out[slot, 2] = origin[2] + (k + sz / m) * h
                    vnear[slot] = best
                    vidx[b, li, lj, lk] = slot
                    slot += 1
    return out, vnear, vidx


@njit(cache=True, nogil=True, inline="always")
def _vid(vidx, block_id, bdims, i, j, k):
    if i < 0 or j < 0 or k < 0:
        return -1
    cb = _lookup(block_id, bdims, i, j, k)
    if cb < 0:
        return -1
    return vidx[cb, i - (i // B) * B, j - (j // B) * B, k - (k // B) * B]


@njit(cache=True, parallel=True)
def _quads(coords, block_id, bdims, dims, inside, vidx, counting, base, out):
    n = coords.shape[0]
    counts = np.zeros(n, np.int64)
    for b in prange(n):
        slot = base[b]
        for li in range(B):
            i = coords[b, 0] * B + li
            for lj in range(B):
                j = coords[b, 1] * B + lj
                for lk in range(B):
                    k = coords[b, 2] * B + lk
                    if i >= dims[0] or j >= dims[1] or k >= dims[2]:
                        continue
                    here = inside[b, li, lj, lk]
                    for axis in range(3):
                        a, bb, c = i, j, k
                        if axis == 0:
                            a += 1
                        elif axis == 1:
                            bb += 1
                        else:
                            c += 1
                        if a >= dims[0] or bb >= dims[1] or c >= dims[2]:
                            continue
                        nb = _lookup(block_id, bdims, a, bb, c)
                        if nb < 0:
                            continue
                        there = inside[nb, a - (a // B) * B, bb - (bb // B) * B, c - (c // B) * B]
                        if here == there:
                            continue
                        if axis == 0:
                            q0 = _vid(vidx, block_id, bdims, i, j - 1, k - 1)
                            q1 = _vid(vidx, block_id, bdims, i, j, k - 1)
                            q2 = _vid(vidx, block_id, bdims, i, j, k)
                            q3 = _vid(vidx, block_id, bdims, i, j - 1, k)
                        elif axis == 1:
                            q0 = _vid(vidx, block_id, bdims, i - 1, j, k - 1)
                            q1 = _vid(vidx, block_id, bdims, i - 1, j, k)
                            q2 = _vid(vidx, block_id, bdims, i, j, k)
                            q3 = _vid(vidx, block_id, bdims, i, j, k - 1)
                        else:
                            q0 = _vid(vidx, block_id, bdims, i - 1, j - 1, k)
                            q1 = _vid(vidx, block_id, bdims, i, j - 1, k)
                            q2 = _vid(vidx, block_id, bdims, i, j, k)
                            q3 = _vid(vidx, block_id, bdims, i - 1, j, k)
                        if q0 < 0 or q1 < 0 or q2 < 0 or q3 < 0:
                            continue
                        if counting:
                            counts[b] += 1
                            continue
                        if here:            # 低端在里面：面朝 +axis
                            out[slot, 0] = q0
                            out[slot, 1] = q1
                            out[slot, 2] = q2
                            out[slot, 3] = q3
                        else:
                            out[slot, 0] = q0
                            out[slot, 1] = q3
                            out[slot, 2] = q2
                            out[slot, 3] = q1
                        slot += 1
    return counts


@njit(cache=True, parallel=True)
def _split_quads(verts, quads):
    """四边形按短对角线拆成两个三角形。"""
    q = quads.shape[0]
    tris = np.empty((q * 2, 3), np.int32)
    for n in prange(q):
        a, b, c, d = quads[n, 0], quads[n, 1], quads[n, 2], quads[n, 3]
        ac = 0.0
        bd = 0.0
        for k in range(3):
            ac += (verts[a, k] - verts[c, k]) ** 2
            bd += (verts[b, k] - verts[d, k]) ** 2
        if ac <= bd:
            tris[2 * n, 0], tris[2 * n, 1], tris[2 * n, 2] = a, b, c
            tris[2 * n + 1, 0], tris[2 * n + 1, 1], tris[2 * n + 1, 2] = a, c, d
        else:
            tris[2 * n, 0], tris[2 * n, 1], tris[2 * n, 2] = a, b, d
            tris[2 * n + 1, 0], tris[2 * n + 1, 1], tris[2 * n + 1, 2] = b, c, d
    return tris


# ====================================================================== 5. 补洞、放松与属性搬运
@njit(cache=True, nogil=True)
def _outgoing(tris, vertex_count):
    """每个顶点出发的半边（a→b）的 CSR：offsets (V+1,)，targets (3T,)，halfedge (3T,) 记原半边编号。"""
    counts = np.zeros(vertex_count + 1, np.int64)
    for t in range(tris.shape[0]):
        for k in range(3):
            counts[tris[t, k] + 1] += 1
    offsets = np.cumsum(counts)
    fill = offsets[:-1].copy()
    targets = np.empty(tris.shape[0] * 3, np.int64)
    halfedge = np.empty(tris.shape[0] * 3, np.int64)
    for t in range(tris.shape[0]):
        for k in range(3):
            a = tris[t, k]
            targets[fill[a]] = tris[t, (k + 1) % 3]
            halfedge[fill[a]] = t * 3 + k
            fill[a] += 1
    return offsets, targets, halfedge


@njit(cache=True, nogil=True)
def _boundary_halfedges(tris, vertex_count):
    """没有反向半边的半边（开口的边）：返回布尔 (3T,)，半边 t*3+k 是 tris[t,k]→tris[t,(k+1)%3]。"""
    offsets, targets, _he = _outgoing(tris, vertex_count)
    out = np.zeros(tris.shape[0] * 3, np.bool_)
    for t in range(tris.shape[0]):
        for k in range(3):
            a = tris[t, k]
            b = tris[t, (k + 1) % 3]
            found = False
            for e in range(offsets[b], offsets[b + 1]):
                if targets[e] == a:
                    found = True
                    break
            out[t * 3 + k] = not found
    return out


@njit(cache=True, nogil=True)
def _signed_volume(verts, tris):
    total = 0.0
    for t in range(tris.shape[0]):
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        ax, ay, az = verts[a, 0], verts[a, 1], verts[a, 2]
        bx, by, bz = verts[b, 0], verts[b, 1], verts[b, 2]
        cx, cy, cz = verts[c, 0], verts[c, 1], verts[c, 2]
        total += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
    return total / 6.0


@njit(cache=True, nogil=True)
def _components(tris, vertex_count):
    """按共用顶点把三角形分成连通块：返回每个三角形的块编号 (T,) 和块数。"""
    parent = np.arange(vertex_count)
    for t in range(tris.shape[0]):
        for k in range(1, 3):
            a = tris[t, 0]
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            b = tris[t, k]
            while parent[b] != b:
                parent[b] = parent[parent[b]]
                b = parent[b]
            if a != b:
                parent[max(a, b)] = min(a, b)
    label = np.full(vertex_count, -1, np.int64)
    count = 0
    out = np.empty(tris.shape[0], np.int64)
    for t in range(tris.shape[0]):
        a = tris[t, 0]
        while parent[a] != a:
            a = parent[a]
        if label[a] < 0:
            label[a] = count
            count += 1
        out[t] = label[a]
    return out, count


def drop_islands(vertices: np.ndarray, triangles: np.ndarray, min_triangles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """去掉三角形少于 min_triangles 的小碎块。返回 (顶点, 三角形, 旧顶点编号, 去掉的块数)。"""
    comp, count = _components(np.ascontiguousarray(triangles, np.int64), len(vertices))
    if count <= 1 or min_triangles <= 0:
        return vertices, triangles, np.arange(len(vertices)), 0
    sizes = np.bincount(comp, minlength=count)
    small = sizes < min_triangles
    if not small.any() or small.all():
        return vertices, triangles, np.arange(len(vertices)), 0
    keep_tris = ~small[comp]
    kept = triangles[keep_tris]
    used = np.zeros(len(vertices), bool)
    used[kept.reshape(-1)] = True
    old = np.nonzero(used)[0]
    remap = np.full(len(vertices), -1, np.int64)
    remap[old] = np.arange(len(old))
    return vertices[old], remap[kept].astype(np.int32), old, int(small.sum())


def fill_holes(mesh: SculptMesh, max_boundary_edges: int = 500_000) -> tuple[SculptMesh, int]:
    """把开口（边界环）用扇形三角形封上，方向和相邻三角形一致。返回 (新网格, 补了几个洞)。
    边界边太多（三角形汤）时不补，交给三个方向的投票。"""
    tris = np.ascontiguousarray(mesh.triangles, np.int64)
    if len(tris) == 0:
        return mesh, 0
    a = tris.reshape(-1)
    b = tris[:, [1, 2, 0]].reshape(-1)
    boundary = _boundary_halfedges(tris, int(tris.max()) + 1)
    if not boundary.any() or int(boundary.sum()) > max_boundary_edges:
        return mesh, 0
    starts = a[boundary]
    ends = b[boundary]
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    first = {}
    for index, vertex in enumerate(starts.tolist()):
        first.setdefault(vertex, []).append(index)
    used = np.zeros(len(starts), bool)
    verts = np.asarray(mesh.vertices, np.float64)
    new_verts = []
    new_tris = []
    holes = 0
    next_index = len(verts)
    for seed in range(len(starts)):
        if used[seed]:
            continue
        loop = []
        edge = seed
        while edge is not None and not used[edge]:
            used[edge] = True
            loop.append(int(starts[edge]))
            candidates = first.get(int(ends[edge]), [])
            edge = next((c for c in candidates if not used[c]), None)
        if len(loop) < 3:
            continue
        holes += 1
        if len(loop) == 3:
            new_tris.append((loop[2], loop[1], loop[0]))
            continue
        center = verts[loop].mean(axis=0)
        new_verts.append(center)
        for i in range(len(loop)):
            new_tris.append((next_index, loop[(i + 1) % len(loop)], loop[i]))
        next_index += 1
    if not new_tris:
        return mesh, 0
    vertices = np.concatenate([verts, np.asarray(new_verts, np.float64).reshape(-1, 3)]).astype(np.float32)
    triangles = np.concatenate([tris, np.asarray(new_tris, np.int64)]).astype(np.int32)
    mats = None
    if mesh.material_ids is not None:
        mats = np.concatenate([np.asarray(mesh.material_ids, np.int32), np.zeros(len(new_tris), np.int32)])
    out = SculptMesh(vertices, triangles, None, mats)
    mask = getattr(mesh, "mask", None)
    if mask is not None:
        out.mask = np.concatenate([np.asarray(mask, np.float32), np.zeros(len(new_verts), np.float32)])
    return out, holes


@njit(cache=True, nogil=True)
def _accumulate(verts, tris, normals, sums, counts):
    normals[:] = 0.0
    sums[:] = 0.0
    counts[:] = 0
    for t in range(tris.shape[0]):
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        ux, uy, uz = verts[b, 0] - verts[a, 0], verts[b, 1] - verts[a, 1], verts[b, 2] - verts[a, 2]
        vx, vy, vz = verts[c, 0] - verts[a, 0], verts[c, 1] - verts[a, 1], verts[c, 2] - verts[a, 2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        for k in range(3):
            p = tris[t, k]
            q = tris[t, (k + 1) % 3]
            r = tris[t, (k + 2) % 3]
            normals[p, 0] += nx
            normals[p, 1] += ny
            normals[p, 2] += nz
            sums[p, 0] += verts[q, 0] + verts[r, 0]
            sums[p, 1] += verts[q, 1] + verts[r, 1]
            sums[p, 2] += verts[q, 2] + verts[r, 2]
            counts[p] += 2


@njit(cache=True, parallel=True)
def _relax_move(verts, normals, sums, counts, amount, out):
    for v in prange(verts.shape[0]):
        if counts[v] == 0:
            out[v, 0] = verts[v, 0]
            out[v, 1] = verts[v, 1]
            out[v, 2] = verts[v, 2]
            continue
        nx, ny, nz = normals[v, 0], normals[v, 1], normals[v, 2]
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 0.0:
            nx, ny, nz = nx / length, ny / length, nz / length
        inv = 1.0 / counts[v]
        dx = sums[v, 0] * inv - verts[v, 0]
        dy = sums[v, 1] * inv - verts[v, 1]
        dz = sums[v, 2] * inv - verts[v, 2]
        dn = dx * nx + dy * ny + dz * nz
        out[v, 0] = verts[v, 0] + (dx - dn * nx) * amount
        out[v, 1] = verts[v, 1] + (dy - dn * ny) * amount
        out[v, 2] = verts[v, 2] + (dz - dn * nz) * amount


def tangential_relax(vertices: np.ndarray, triangles: np.ndarray, iterations: int = 2,
                     amount: float = 0.5) -> np.ndarray:
    """沿表面切向挪向邻居的平均位置：三角形更均匀，形状和体积基本不变。"""
    if iterations <= 0 or len(triangles) == 0:
        return np.asarray(vertices, np.float32)
    v = np.ascontiguousarray(vertices, np.float64)
    t = np.ascontiguousarray(triangles, np.int64)
    normals = np.empty_like(v)
    sums = np.empty_like(v)
    counts = np.empty(len(v), np.int64)
    out = np.empty_like(v)
    for _ in range(int(iterations)):
        _accumulate(v, t, normals, sums, counts)
        _relax_move(v, normals, sums, counts, float(amount), out)
        v, out = out, v
    return v.astype(np.float32)


@njit(cache=True, parallel=True)
def _transfer_mask(new_verts, vnear, verts, tris, mask):
    out = np.zeros(new_verts.shape[0], np.float32)
    for n in prange(new_verts.shape[0]):
        t = vnear[n]
        if t < 0:
            continue
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        _x, _y, _z, v, w = _closest_on_triangle(new_verts[n, 0], new_verts[n, 1], new_verts[n, 2],
                                                verts[a, 0], verts[a, 1], verts[a, 2],
                                                verts[b, 0], verts[b, 1], verts[b, 2],
                                                verts[c, 0], verts[c, 1], verts[c, 2])
        out[n] = (1.0 - v - w) * mask[a] + v * mask[b] + w * mask[c]
    return out


# ====================================================================== 入口
def grid_for(mesh: SculptMesh, resolution: int) -> dict:
    """给定分辨率（包围盒最长边上的体素数）时的体素大小和格子尺寸，界面上预估用。"""
    verts = np.asarray(mesh.vertices, np.float64)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    extent = float(np.max(hi - lo)) or 1.0
    resolution = int(max(8, min(MAX_RESOLUTION, resolution)))
    h = extent / resolution
    dims = np.ceil((hi - lo) / h).astype(np.int64) + 2 * PAD + 1
    return {"voxel": h, "dims": dims, "origin": lo - PAD * h, "resolution": resolution}


def voxel_remesh(mesh: SculptMesh, resolution: int = 256, relax: int = 2, mask: np.ndarray | None = None,
                 close_holes: bool = True, min_island: int = 64, progress=None) -> SculptMesh:
    """resolution：包围盒最长边上的体素数。结果没有 UV（corner_uvs=None），材质和遮罩从最近的原三角形带过来。
    close_holes：先把开口封上（否则开口处会留下破洞）。min_island：少于这么多三角形的碎块去掉。"""
    started = time.perf_counter()
    timings = {}
    holes = 0
    if close_holes:
        if mask is not None:
            mesh = SculptMesh(mesh.vertices, mesh.triangles, mesh.corner_uvs, mesh.material_ids)
            mesh.mask = mask
        mesh, holes = fill_holes(mesh)
        if holes and mask is not None:
            mask = mesh.mask

    def mark(name: str, t0: float) -> float:
        now = time.perf_counter()
        timings[name] = round((now - t0) * 1000.0, 1)
        if progress is not None:
            progress(name)
        return now

    verts = np.ascontiguousarray(mesh.vertices, np.float64)
    tris = np.ascontiguousarray(mesh.triangles, np.int64)
    if len(tris) == 0:
        raise ValueError("模型没有三角形")
    grid = grid_for(mesh, resolution)
    h = float(grid["voxel"])
    dims = grid["dims"]
    origin = np.asarray(grid["origin"], np.float64)
    band = BAND * h
    bdims = (dims + B - 1) // B
    t0 = time.perf_counter()
    flags = _mark_blocks(verts, tris, origin, h, dims, band, bdims)
    coords = np.argwhere(flags).astype(np.int64)
    del flags
    block_id = np.full(tuple(int(x) for x in bdims), -1, np.int32)
    block_id[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(len(coords), dtype=np.int32)
    offsets, items = _bin_triangles(verts, tris, origin, h, dims, band, block_id, len(coords))
    t0 = mark("blocks", t0)
    dist, near = _block_distances(verts, tris, origin, h, dims, band, coords, offsets, items)
    del items
    t0 = mark("distance", t0)
    votes = np.zeros((len(coords), B, B, B), np.uint8)
    valid = np.zeros((len(coords), B, B, B), np.uint8)
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        columns = int(dims[u] * dims[v])
        cursor = np.zeros(columns + 1, np.int64)
        _crossings(verts, tris, origin, h, dims, axis, True, cursor, np.empty(0, np.float64), np.empty(0, np.int8))
        col_offsets = np.cumsum(cursor)
        pos = np.empty(int(col_offsets[-1]), np.float64)
        delta = np.empty(int(col_offsets[-1]), np.int8)
        fill = col_offsets[:-1].copy()
        _crossings(verts, tris, origin, h, dims, axis, False, fill, pos, delta)
        _sort_columns(col_offsets, pos, delta)
        _vote_axis(coords, origin, h, dims, axis, col_offsets, pos, delta, votes, valid)
    inside, undecided = _resolve_inside(verts, tris, origin, h, coords, votes, valid, near)
    del votes, valid
    t0 = mark("sign", t0)
    counts, cell_flags = _count_cells(coords, block_id, bdims, dims, inside)
    base = np.zeros(len(coords), np.int64)
    np.cumsum(counts[:-1], out=base[1:])
    total = int(counts.sum())
    new_verts, vnear, vidx = _place_vertices(coords, block_id, bdims, origin, h, inside, dist, near, cell_flags,
                                             base, total)
    del cell_flags
    qcounts = _quads(coords, block_id, bdims, dims, inside, vidx, True, base, np.empty((0, 4), np.int32))
    qbase = np.zeros(len(coords), np.int64)
    np.cumsum(qcounts[:-1], out=qbase[1:])
    quads = np.empty((int(qcounts.sum()), 4), np.int32)
    _quads(coords, block_id, bdims, dims, inside, vidx, False, qbase, quads)
    del vidx, dist, near, inside
    t0 = mark("surface", t0)
    if len(quads) == 0:
        raise ValueError("重构后没有剩下表面（模型可能是开口的薄片，或者分辨率太低）")
    triangles = _split_quads(new_verts, quads)
    # 朝向：带符号体积为负就整体翻转
    volume = float(_signed_volume(new_verts, triangles))
    if volume < 0.0:
        triangles = np.ascontiguousarray(triangles[:, [0, 2, 1]])
    # 内外相间的格子会让几片表面挤在同一条边上：拆开成各自的顶点
    fixed_tris, source, split = _split_fans(np.ascontiguousarray(triangles, np.int64),
                                            new_verts.astype(np.float64), len(new_verts))
    if split:
        new_verts = new_verts[source]
        vnear = vnear[source]
        triangles = fixed_tris.astype(np.int32)
    new_verts, triangles, old, islands = drop_islands(new_verts, triangles, int(min_island))
    if islands:
        vnear = vnear[old]
    relaxed = tangential_relax(new_verts, triangles, relax) if relax > 0 else new_verts
    t0 = mark("relax", t0)
    mats = np.zeros(len(triangles), np.int32)
    if mesh.material_ids is not None and len(mesh.material_ids) == len(tris):
        corner_near = vnear[triangles[:, 0]]
        source = np.asarray(mesh.material_ids, np.int32)
        mats = np.where(corner_near >= 0, source[np.maximum(corner_near, 0)], 0).astype(np.int32)
    out = SculptMesh(relaxed, triangles, None, mats)
    if mask is not None and len(mask) == len(verts):
        out.mask = _transfer_mask(relaxed.astype(np.float64), vnear, verts, tris, np.asarray(mask, np.float64))
    out.stats = {"remesh_ms": round((time.perf_counter() - started) * 1000.0, 1), "steps_ms": timings,
                 "grid": [int(d) for d in dims], "voxel": h, "blocks": int(len(coords)),
                 "undecided": int(undecided), "volume": abs(volume), "holes": holes, "split": int(split),
                 "islands_removed": int(islands)}
    return out
