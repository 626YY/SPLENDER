"""法线：按角度平滑（夹角小于阈值的相邻面一起平滑，大于的保持锐利的棱）。"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def angle_smooth(pos, weld, count, cos_limit):
    """pos：逐角点 (3T, 3)；weld：每个角点的焊接组号；count：组数；cos_limit：夹角阈值的余弦。
    每个角点的法线 = 同一焊接点上、和本面夹角不超过阈值的那些面的单位法线按角点夹角加权平均。"""
    corners = pos.shape[0]
    tris = corners // 3
    fn = np.zeros((tris, 3))
    corner_angle = np.zeros(corners)
    for k in range(tris):
        b = 3 * k
        ax = pos[b + 1, 0] - pos[b, 0]
        ay = pos[b + 1, 1] - pos[b, 1]
        az = pos[b + 1, 2] - pos[b, 2]
        bx = pos[b + 2, 0] - pos[b, 0]
        by = pos[b + 2, 1] - pos[b, 1]
        bz = pos[b + 2, 2] - pos[b, 2]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        length = math.sqrt(cx * cx + cy * cy + cz * cz)
        if length > 0.0:
            fn[k, 0] = cx / length
            fn[k, 1] = cy / length
            fn[k, 2] = cz / length
        for c in range(3):
            i0 = b + c
            i1 = b + (c + 1) % 3
            i2 = b + (c + 2) % 3
            ux = pos[i1, 0] - pos[i0, 0]
            uy = pos[i1, 1] - pos[i0, 1]
            uz = pos[i1, 2] - pos[i0, 2]
            vx = pos[i2, 0] - pos[i0, 0]
            vy = pos[i2, 1] - pos[i0, 1]
            vz = pos[i2, 2] - pos[i0, 2]
            lu = math.sqrt(ux * ux + uy * uy + uz * uz)
            lv = math.sqrt(vx * vx + vy * vy + vz * vz)
            if lu > 0.0 and lv > 0.0:
                d = (ux * vx + uy * vy + uz * vz) / (lu * lv)
                d = min(1.0, max(-1.0, d))
                corner_angle[i0] = math.acos(d)
    start = np.zeros(count + 1, np.int64)
    for i in range(corners):
        start[weld[i] + 1] += 1
    for v in range(count):
        start[v + 1] += start[v]
    fill = start[:-1].copy()
    members = np.empty(corners, np.int64)
    for i in range(corners):
        w = weld[i]
        members[fill[w]] = i
        fill[w] += 1
    out = np.empty((corners, 3), np.float32)
    for i in range(corners):
        w = weld[i]
        f = i // 3
        sx = 0.0
        sy = 0.0
        sz = 0.0
        for j in range(start[w], start[w + 1]):
            other = members[j]
            g = other // 3
            dot = fn[f, 0] * fn[g, 0] + fn[f, 1] * fn[g, 1] + fn[f, 2] * fn[g, 2]
            if g == f or dot >= cos_limit:
                a = corner_angle[other]
                sx += fn[g, 0] * a
                sy += fn[g, 1] * a
                sz += fn[g, 2] * a
        length = math.sqrt(sx * sx + sy * sy + sz * sz)
        if length > 1e-12:
            out[i, 0] = sx / length
            out[i, 1] = sy / length
            out[i, 2] = sz / length
        else:
            out[i, 0] = fn[f, 0]
            out[i, 1] = fn[f, 1]
            out[i, 2] = fn[f, 2]
    return out
