"""在 CPU 上为三角形建 BVH（分箱 SAH），压平成着色器直接读的数组。

节点数组 nodes 形状 (节点数, 2, 4) float32：
  nodes[i, 0] = (包围盒最小 x, y, z, 整数位：内部节点是左孩子下标，叶子是第一个三角形下标)
  nodes[i, 1] = (包围盒最大 x, y, z, 整数位：叶子是三角形个数，内部节点是 0)
内部节点的两个孩子相邻存放：左孩子 left，右孩子 left + 1。根是 0 号节点。
三角形按 order 重新排列，叶子引用的是排列后的下标。
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

LEAF_SIZE = 4          # 叶子最多几个三角形（划分不划算时允许到 MAX_LEAF）
MAX_LEAF = 16
BINS = 16


@njit(cache=True, nogil=True)
def _area(x0, y0, z0, x1, y1, z1):
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    if dx < 0.0 or dy < 0.0 or dz < 0.0:
        return 0.0
    return 2.0 * (dx * dy + dy * dz + dz * dx)


@njit(cache=True, nogil=True)
def _build(tri_min, tri_max, cent, leaf_size, max_leaf, bins):
    n = cent.shape[0]
    order = np.arange(n).astype(np.int32)
    capacity = max(1, 2 * n)
    node_min = np.empty((capacity, 3), np.float32)
    node_max = np.empty((capacity, 3), np.float32)
    node_first = np.zeros(capacity, np.int32)
    node_count = np.zeros(capacity, np.int32)
    stack_node = np.empty(capacity, np.int32)
    stack_start = np.empty(capacity, np.int32)
    stack_end = np.empty(capacity, np.int32)
    sp = 0
    stack_node[0] = 0
    stack_start[0] = 0
    stack_end[0] = n
    sp = 1
    used = 1
    bin_count = np.zeros(bins, np.int64)
    bin_min = np.empty((bins, 3), np.float64)
    bin_max = np.empty((bins, 3), np.float64)
    left_area = np.zeros(bins, np.float64)
    left_count = np.zeros(bins, np.int64)
    max_depth_seen = 0
    while sp > 0:
        sp -= 1
        node = stack_node[sp]
        start = stack_start[sp]
        end = stack_end[sp]
        bx0 = by0 = bz0 = 1e30
        bx1 = by1 = bz1 = -1e30
        cx0 = cy0 = cz0 = 1e30
        cx1 = cy1 = cz1 = -1e30
        for i in range(start, end):
            t = order[i]
            if tri_min[t, 0] < bx0:
                bx0 = tri_min[t, 0]
            if tri_min[t, 1] < by0:
                by0 = tri_min[t, 1]
            if tri_min[t, 2] < bz0:
                bz0 = tri_min[t, 2]
            if tri_max[t, 0] > bx1:
                bx1 = tri_max[t, 0]
            if tri_max[t, 1] > by1:
                by1 = tri_max[t, 1]
            if tri_max[t, 2] > bz1:
                bz1 = tri_max[t, 2]
            c0 = cent[t, 0]
            c1 = cent[t, 1]
            c2 = cent[t, 2]
            if c0 < cx0:
                cx0 = c0
            if c1 < cy0:
                cy0 = c1
            if c2 < cz0:
                cz0 = c2
            if c0 > cx1:
                cx1 = c0
            if c1 > cy1:
                cy1 = c1
            if c2 > cz1:
                cz1 = c2
        node_min[node, 0] = bx0
        node_min[node, 1] = by0
        node_min[node, 2] = bz0
        node_max[node, 0] = bx1
        node_max[node, 1] = by1
        node_max[node, 2] = bz1
        count = end - start
        if count <= leaf_size:
            node_first[node] = start
            node_count[node] = count
            continue
        ex = cx1 - cx0
        ey = cy1 - cy0
        ez = cz1 - cz0
        axis = 0
        extent = ex
        lo = cx0
        if ey > extent:
            axis = 1
            extent = ey
            lo = cy0
        if ez > extent:
            axis = 2
            extent = ez
            lo = cz0
        mid = -1
        if extent > 1e-12:
            for b in range(bins):
                bin_count[b] = 0
                bin_min[b, 0] = 1e30
                bin_min[b, 1] = 1e30
                bin_min[b, 2] = 1e30
                bin_max[b, 0] = -1e30
                bin_max[b, 1] = -1e30
                bin_max[b, 2] = -1e30
            scale = bins / extent * 0.999999
            for i in range(start, end):
                t = order[i]
                b = int((cent[t, axis] - lo) * scale)
                if b < 0:
                    b = 0
                elif b >= bins:
                    b = bins - 1
                bin_count[b] += 1
                for k in range(3):
                    if tri_min[t, k] < bin_min[b, k]:
                        bin_min[b, k] = tri_min[t, k]
                    if tri_max[t, k] > bin_max[b, k]:
                        bin_max[b, k] = tri_max[t, k]
            # 从左往右累积
            ax0 = ay0 = az0 = 1e30
            ax1 = ay1 = az1 = -1e30
            running = 0
            for b in range(bins - 1):
                running += bin_count[b]
                if bin_count[b] > 0:
                    ax0 = min(ax0, bin_min[b, 0])
                    ay0 = min(ay0, bin_min[b, 1])
                    az0 = min(az0, bin_min[b, 2])
                    ax1 = max(ax1, bin_max[b, 0])
                    ay1 = max(ay1, bin_max[b, 1])
                    az1 = max(az1, bin_max[b, 2])
                left_count[b] = running
                left_area[b] = _area(ax0, ay0, az0, ax1, ay1, az1) if running > 0 else 0.0
            # 从右往左累积并求最省的切分
            ax0 = ay0 = az0 = 1e30
            ax1 = ay1 = az1 = -1e30
            running = 0
            best_cost = 1e300
            best_split = -1
            for b in range(bins - 1, 0, -1):
                running += bin_count[b]
                if bin_count[b] > 0:
                    ax0 = min(ax0, bin_min[b, 0])
                    ay0 = min(ay0, bin_min[b, 1])
                    az0 = min(az0, bin_min[b, 2])
                    ax1 = max(ax1, bin_max[b, 0])
                    ay1 = max(ay1, bin_max[b, 1])
                    az1 = max(az1, bin_max[b, 2])
                lc = left_count[b - 1]
                if lc == 0 or running == 0:
                    continue
                cost = lc * left_area[b - 1] + running * _area(ax0, ay0, az0, ax1, ay1, az1)
                if cost < best_cost:
                    best_cost = cost
                    best_split = b
            parent_area = _area(bx0, by0, bz0, bx1, by1, bz1)
            leaf_cost = count * parent_area
            if best_split > 0 and (best_cost < leaf_cost or count > max_leaf):
                i = start
                j = end - 1
                while i <= j:
                    t = order[i]
                    b = int((cent[t, axis] - lo) * scale)
                    if b < 0:
                        b = 0
                    elif b >= bins:
                        b = bins - 1
                    if b < best_split:
                        i += 1
                    else:
                        order[i] = order[j]
                        order[j] = t
                        j -= 1
                mid = i
            elif count <= max_leaf:
                node_first[node] = start
                node_count[node] = count
                continue
        if mid <= start or mid >= end:
            mid = (start + end) // 2        # 退化（重心重合等）：按序号对半分
        left = used
        used += 2
        node_first[node] = left
        node_count[node] = 0
        stack_node[sp] = left + 1
        stack_start[sp] = mid
        stack_end[sp] = end
        sp += 1
        stack_node[sp] = left
        stack_start[sp] = start
        stack_end[sp] = mid
        sp += 1
    return order, node_min[:used], node_max[:used], node_first[:used], node_count[:used]


@njit(cache=True, nogil=True)
def _depth(node_first, node_count):
    """最大深度（检查着色器的栈够不够用）。"""
    n = node_first.shape[0]
    depth = np.zeros(n, np.int32)
    best = 0
    for i in range(n):
        if node_count[i] == 0:
            left = node_first[i]
            depth[left] = depth[i] + 1
            depth[left + 1] = depth[i] + 1
            if depth[i] + 1 > best:
                best = depth[i] + 1
    return best


def build_bvh(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, leaf_size: int = LEAF_SIZE) -> dict:
    """v0, v1, v2：(T, 3) 三个角的位置。返回 {order, nodes, depth, ms}。"""
    started = time.perf_counter()
    v0 = np.asarray(v0, np.float32)
    v1 = np.asarray(v1, np.float32)
    v2 = np.asarray(v2, np.float32)
    tri_min = np.minimum(np.minimum(v0, v1), v2)
    tri_max = np.maximum(np.maximum(v0, v1), v2)
    cent = (tri_min + tri_max) * 0.5
    if len(cent) == 0:
        nodes = np.zeros((1, 2, 4), np.float32)
        nodes[0, 0, :3] = 1e30
        nodes[0, 1, :3] = -1e30
        return {"order": np.zeros(0, np.int32), "nodes": nodes, "depth": 0,
                "ms": (time.perf_counter() - started) * 1000.0}
    order, nmin, nmax, first, count = _build(tri_min, tri_max, cent, int(leaf_size), MAX_LEAF, BINS)
    nodes = np.zeros((len(first), 2, 4), np.float32)
    nodes[:, 0, :3] = nmin
    nodes[:, 1, :3] = nmax
    nodes[:, 0, 3] = first.view(np.float32)
    nodes[:, 1, 3] = count.view(np.float32)
    depth = int(_depth(first, count))
    return {"order": order, "nodes": nodes, "depth": depth, "ms": (time.perf_counter() - started) * 1000.0}
