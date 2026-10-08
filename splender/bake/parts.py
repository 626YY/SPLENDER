"""把模型分成部件，每个三角形一个部件编号（ID 贴图用）；以及点选：屏幕上一点打中哪个三角形。

部件的划分：
- 网格部件：共用顶点（位置相同）连在一起的面算一个部件；
- UV 岛：位置和 UV 都相同的角点才算连着，UV 上连在一起的面算一个部件。
"""
from __future__ import annotations

import numba
import numpy as np


def _group(keys: np.ndarray) -> np.ndarray:
    """每行一个键（整数），相同的行编成同一个号。"""
    order = np.lexsort(keys.T[::-1])
    sorted_keys = keys[order]
    new_group = np.ones(len(keys), bool)
    new_group[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
    group = np.cumsum(new_group) - 1
    inverse = np.empty(len(keys), np.int64)
    inverse[order] = group
    return inverse


def _quantize(values: np.ndarray, tolerance: float) -> np.ndarray:
    values = np.asarray(values, np.float64)
    extent = float(np.max(np.ptp(values, axis=0))) if len(values) else 1.0
    step = max(tolerance * (extent or 1.0), 1e-12)
    return np.round((values - values.min(axis=0)) / step).astype(np.int64)


@numba.njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@numba.njit(cache=True)
def _components(corner_vertex, vertex_count):
    """每 3 个角点一个三角形；按共用顶点连成部件，按三角形出现的先后编号。"""
    parent = np.arange(vertex_count)
    tri_count = len(corner_vertex) // 3
    for t in range(tri_count):
        a = _find(parent, corner_vertex[3 * t])
        for k in range(1, 3):
            b = _find(parent, corner_vertex[3 * t + k])
            if a != b:
                parent[b] = a
    label = np.full(vertex_count, -1, np.int64)
    out = np.empty(tri_count, np.int32)
    count = 0
    for t in range(tri_count):
        r = _find(parent, corner_vertex[3 * t])
        if label[r] < 0:
            label[r] = count
            count += 1
        out[t] = label[r]
    return out, count


def triangle_parts(positions: np.ndarray, uvs: np.ndarray | None = None, mode: str = "PART",
                   tolerance: float = 1e-6) -> tuple[np.ndarray, int]:
    """逐角点的模型（每 3 行一个三角形）→ (每个三角形的部件号 (T,) int32, 部件数)。
    mode：PART 网格部件；UV_ISLAND UV 岛（要给 uvs）。"""
    p = np.asarray(positions, np.float64).reshape(-1, 3)
    if len(p) == 0:
        return np.zeros(0, np.int32), 0
    keys = _quantize(p, tolerance)
    if mode == "UV_ISLAND" and uvs is not None:
        keys = np.concatenate([keys, _quantize(np.asarray(uvs, np.float64).reshape(-1, 2), tolerance)], axis=1)
    corner_vertex = _group(keys)
    ids, count = _components(corner_vertex, int(corner_vertex.max()) + 1)
    return ids, int(count)


def compact(ids: np.ndarray, chosen: np.ndarray, start: int = 0) -> tuple[np.ndarray, int]:
    """只保留 chosen 的三角形的部件，从 start 开始连续编号；其余三角形记 -1。返回 (新编号, 部件数)。"""
    out = np.full(len(ids), -1, np.int32)
    if not chosen.any():
        return out, 0
    used, inverse = np.unique(ids[chosen], return_inverse=True)
    out[chosen] = (inverse + start).astype(np.int32)
    return out, int(len(used))


def part_color(ids) -> np.ndarray:
    """部件号 → 好分辨的颜色 (…, 3) uint8（色相按黄金角错开），-1 是黑。"""
    ids = np.asarray(ids, np.float64)
    hue = (ids * 0.61803398875) % 1.0
    sat = 0.55 + 0.35 * ((ids * 0.38196601125) % 1.0)
    val = 0.75 + 0.25 * ((ids * 0.7548776662) % 1.0)
    h6 = hue * 6.0
    c = val * sat
    x = c * (1.0 - np.abs(h6 % 2.0 - 1.0))
    m = val - c
    zeros = np.zeros_like(h6)
    sector = np.floor(h6).astype(int) % 6
    r = np.choose(sector, [c, x, zeros, zeros, x, c])
    g = np.choose(sector, [x, c, c, x, zeros, zeros])
    b = np.choose(sector, [zeros, zeros, x, c, c, x])
    rgb = np.stack([r + m, g + m, b + m], axis=-1)
    rgb[ids < 0] = 0.0
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


@numba.njit(cache=True)
def _ray_hit(corners, origin, direction):
    best_t = 1e30
    best = -1
    best_u = 0.0
    best_v = 0.0
    ox, oy, oz = origin[0], origin[1], origin[2]
    dx, dy, dz = direction[0], direction[1], direction[2]
    for t in range(len(corners) // 3):
        ax, ay, az = corners[3 * t, 0], corners[3 * t, 1], corners[3 * t, 2]
        e1x = corners[3 * t + 1, 0] - ax
        e1y = corners[3 * t + 1, 1] - ay
        e1z = corners[3 * t + 1, 2] - az
        e2x = corners[3 * t + 2, 0] - ax
        e2y = corners[3 * t + 2, 1] - ay
        e2z = corners[3 * t + 2, 2] - az
        px = dy * e2z - dz * e2y
        py = dz * e2x - dx * e2z
        pz = dx * e2y - dy * e2x
        det = e1x * px + e1y * py + e1z * pz
        if abs(det) < 1e-20:
            continue
        inv = 1.0 / det
        sx, sy, sz = ox - ax, oy - ay, oz - az
        u = (sx * px + sy * py + sz * pz) * inv
        if u < 0.0 or u > 1.0:
            continue
        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        v = (dx * qx + dy * qy + dz * qz) * inv
        if v < 0.0 or u + v > 1.0:
            continue
        dist = (e2x * qx + e2y * qy + e2z * qz) * inv
        if dist > 1e-9 and dist < best_t:
            best_t = dist
            best = t
            best_u = u
            best_v = v
    return best, best_u, best_v, best_t


def ray_hit(positions: np.ndarray, origin, direction) -> tuple[int, float, float, float]:
    """射线打中的第一个三角形：(三角形号或 -1, 重心坐标 u, v, 距离)。positions 是逐角点 (3T, 3)。"""
    corners = np.ascontiguousarray(positions, np.float32).reshape(-1, 3)
    o = np.asarray(origin, np.float64)
    d = np.asarray(direction, np.float64)
    tri, u, v, t = _ray_hit(corners, o, d)
    return int(tri), float(u), float(v), float(t)


def parse_ids(text: str) -> list[int]:
    """「3,7,12」→ [3, 7, 12]（去重、排序，丢掉不是数的）。"""
    found = set()
    for item in str(text or "").replace("，", ",").split(","):
        item = item.strip()
        if item.isdigit():
            found.add(int(item))
    return sorted(found)


def format_ids(ids) -> str:
    return ",".join(str(int(i)) for i in sorted(set(int(i) for i in ids)))
