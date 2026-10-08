"""逐顶点曲率：给模型贴图里的「曲率」用。烘焙时在后台线程里跑，所以这里的 numba 函数都不开多线程（nogil 即可）。

做法：位置完全相同的角点算同一个顶点（按浮点位模式哈希，O(n)）；顶点法线按面积加权；
每条边的法曲率 k = (n_j - n_i)·(p_j - p_i) / |p_j - p_i|²（凸为正、凹为负，半径 r 的球上处处是 1/r）；
顶点取相邻边的平均，再按需平滑几次。最后按面积加权的 95% 分位数归一到 [-1, 1]，不同大小、不同密度的模型手感一致。
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _hash_weld(bits):
    """bits：(N, 3) uint32，同一位置的角点位模式相同。返回 (每个角点的顶点号, 顶点数)。"""
    n = bits.shape[0]
    cap = 1
    while cap < n * 2:
        cap <<= 1
    table = np.full(cap, -1, np.int64)
    ids = np.empty(n, np.int64)
    first = np.empty(n, np.int64)
    count = 0
    mask = cap - 1
    for i in range(n):
        h = (np.uint64(bits[i, 0]) * np.uint64(73856093)) ^ (np.uint64(bits[i, 1]) * np.uint64(19349663)) \
            ^ (np.uint64(bits[i, 2]) * np.uint64(83492791))
        slot = np.int64(h & np.uint64(mask))
        while True:
            j = table[slot]
            if j < 0:
                table[slot] = count
                first[count] = i
                ids[i] = count
                count += 1
                break
            k = first[j]
            if bits[k, 0] == bits[i, 0] and bits[k, 1] == bits[i, 1] and bits[k, 2] == bits[i, 2]:
                ids[i] = j
                break
            slot = (slot + 1) & mask
    return ids, first[:count].copy(), count


@njit(cache=True, nogil=True)
def _normals_and_curvature(pos, tri, count):
    normals = np.zeros((count, 3), np.float64)
    area = np.zeros(count, np.float64)
    for t in range(tri.shape[0]):
        a, b, c = tri[t, 0], tri[t, 1], tri[t, 2]
        ux, uy, uz = pos[b, 0] - pos[a, 0], pos[b, 1] - pos[a, 1], pos[b, 2] - pos[a, 2]
        vx, vy, vz = pos[c, 0] - pos[a, 0], pos[c, 1] - pos[a, 1], pos[c, 2] - pos[a, 2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        double = math.sqrt(nx * nx + ny * ny + nz * nz)
        for k in range(3):
            v = tri[t, k]
            normals[v, 0] += nx
            normals[v, 1] += ny
            normals[v, 2] += nz
            area[v] += double / 6.0
    for v in range(count):
        length = math.sqrt(normals[v, 0] ** 2 + normals[v, 1] ** 2 + normals[v, 2] ** 2)
        if length > 0.0:
            normals[v, 0] /= length
            normals[v, 1] /= length
            normals[v, 2] /= length
    curv = np.zeros(count, np.float64)
    weight = np.zeros(count, np.float64)
    for t in range(tri.shape[0]):
        for k in range(3):
            i = tri[t, k]
            for m in range(1, 3):
                j = tri[t, (k + m) % 3]
                ex = pos[j, 0] - pos[i, 0]
                ey = pos[j, 1] - pos[i, 1]
                ez = pos[j, 2] - pos[i, 2]
                ee = ex * ex + ey * ey + ez * ez
                if ee <= 0.0:
                    continue
                dn = (normals[j, 0] - normals[i, 0]) * ex + (normals[j, 1] - normals[i, 1]) * ey \
                    + (normals[j, 2] - normals[i, 2]) * ez
                curv[i] += dn / ee
                weight[i] += 1.0
    for v in range(count):
        if weight[v] > 0.0:
            curv[v] /= weight[v]
    return normals, curv, area


@njit(cache=True, nogil=True)
def _neighbor_mean(values, tri, out):
    sums = np.zeros(values.shape[0], np.float64)
    counts = np.zeros(values.shape[0], np.float64)
    for t in range(tri.shape[0]):
        for k in range(3):
            i = tri[t, k]
            sums[i] += values[tri[t, (k + 1) % 3]] + values[tri[t, (k + 2) % 3]]
            counts[i] += 2.0
    for v in range(values.shape[0]):
        out[v] = sums[v] / counts[v] if counts[v] > 0.0 else values[v]


def weld_corners(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """逐角点位置 (3T, 3) → (每个角点的顶点号 (3T,), 每个顶点的代表角点 (V,), 顶点数)。只合并位置完全相同的角点。"""
    p = np.ascontiguousarray(positions, np.float32).reshape(-1, 3)
    bits = p.view(np.uint32).reshape(-1, 3)
    # -0.0 和 0.0 当成同一个位置
    bits = np.where(bits == np.uint32(0x80000000), np.uint32(0), bits)
    ids, first, count = _hash_weld(np.ascontiguousarray(bits))
    return ids, first, int(count)


def corner_curvature(positions: np.ndarray, smooth: int = 2, contrast: float = 1.0,
                     percentile: float = 95.0) -> tuple[np.ndarray, dict]:
    """逐角点曲率 (3T,) float32，范围 [-1, 1]，凸为正。smooth：平滑次数；contrast：放大倍数。"""
    started = time.perf_counter()
    ids, first, count = weld_corners(positions)
    p = np.ascontiguousarray(positions, np.float32).reshape(-1, 3)
    pos = p[first].astype(np.float64)
    tri = ids.reshape(-1, 3)
    _normals, curv, area = _normals_and_curvature(pos, tri, count)
    buffer = np.empty_like(curv)
    for _ in range(max(0, int(smooth))):
        _neighbor_mean(curv, tri, buffer)
        curv = curv * 0.5 + buffer * 0.5
    magnitude = np.abs(curv)
    scale = 0.0
    if count:
        order = np.argsort(magnitude)
        cumulative = np.cumsum(area[order])
        cut = np.searchsorted(cumulative, cumulative[-1] * float(percentile) / 100.0)
        scale = float(magnitude[order[min(cut, count - 1)]])
    if scale <= 0.0:
        scale = 1.0
    normalized = np.clip(curv / scale * float(contrast), -1.0, 1.0).astype(np.float32)
    stats = {"vertices": count, "scale": scale, "ms": round((time.perf_counter() - started) * 1000.0, 1)}
    return normalized[ids], stats
