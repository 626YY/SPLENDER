"""绘制前的几何预处理：按 256 纹素的格分桶、每格三维包围盒、UV 岛扩边、UV 重叠估计。

输入 MeshData 里一个材质的三角形（一个纹理集），输出 SetGeometry 给绘制引擎上传到显卡。
绘制时引擎只取「被笔触碰到的格」的三角形，按 UV 光栅化到这一格的页里，
每个纹素由此拿到自己的三维位置和法线。

约定（见 docs/ARCHITECTURE.md）：
- UV 原点在左下角，纹素 (x, y) = (u·分辨率, v·分辨率)。
- 格 (tx, ty) 覆盖纹素 [256·tx, 256·(tx+1)] × [256·ty, 256·(ty+1)]，编号 t = ty·grid + tx。
- 三角形与格「相交」指两者内部有面积大于零的公共部分；只在格线上擦边、只碰到一个角的不算。
  显卡按纹素中心光栅化，纹素中心永远不在格线上，所以这个判定既不漏纹素，也不多放三角形。
- UV 超出 [0, 1] 的三角形整体按重心所在的整数格平移回来（u - floor(重心 u)），
  平移后的 UV 放在 SetGeometry.uvs 里，绘制时要用它代替 MeshData.uvs。
  所有判定都基于这份 float32 数据本身，和显卡拿到的完全一致。

所有重活用 numba 编译（cache=True、nogil=True），大模型分块后用线程池并行。
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numba import njit

__all__ = ["SetGeometry", "build_set_geometry", "TILE"]

#: 页的边长（纹素）。
TILE = 256

# 三角形标记
F_INVALID = 1     # 坐标或 UV 不是有效数值
F_DEGEN = 2       # UV 面积为零
F_OUT = 4         # UV 超出 [0, 1]
F_SHIFT = 8       # 整体平移过
F_CROSS = 16      # 平移后仍有一部分在 [0, 1] 之外
_SKIP = F_INVALID | F_DEGEN

#: 扩边转角处扇形每一片最大张角（弧度）。
_JOIN_STEP = math.pi / 4.0


@dataclass
class SetGeometry:
    """一个纹理集（一个材质）在某个分辨率下的绘制几何。"""

    material_index: int
    resolution: int
    grid: int                    # resolution // 256
    indices: np.ndarray          # (M,) uint32，按格排序的角点索引（指向 MeshData 的逐角数组）
    tile_first: np.ndarray       # (grid*grid,) int32，格 ty*grid+tx 在 indices 里的起点
    tile_count: np.ndarray       # (grid*grid,) int32，索引个数
    tile_min: np.ndarray         # (grid*grid, 3) float32，该格内表面的三维包围盒；空格填 +inf
    tile_max: np.ndarray         # 空格填 -inf
    skirt_vertices: np.ndarray   # (K, 8) float32：u, v, px, py, pz, nx, ny, nz。三角形列表，已按格排序
    skirt_first: np.ndarray      # (grid*grid,) int32，顶点起点
    skirt_count: np.ndarray      # (grid*grid,) int32，顶点个数
    overlap_ratio: float         # UV 重叠面积占比的估计
    uv_out_of_range: bool
    #: 平移回 [0, 1] 后的 UV，(T*3, 2) float32，与 MeshData.uvs 同样排列；没有平移时为 None（直接用 MeshData.uvs）
    uvs: np.ndarray | None = None
    #: 给用户看的提示（坏三角形、UV 越界、UV 重叠等）
    warnings: list[str] = field(default_factory=list)
    #: 统计与耗时，供性能面板和测试使用
    stats: dict = field(default_factory=dict)

    def tile_indices(self, tx: int, ty: int) -> np.ndarray:
        """格 (tx, ty) 的角点索引。"""
        t = ty * self.grid + tx
        a = int(self.tile_first[t])
        return self.indices[a:a + int(self.tile_count[t])]

    def tile_skirt(self, tx: int, ty: int) -> np.ndarray:
        """格 (tx, ty) 的扩边顶点 (n, 8)。"""
        t = ty * self.grid + tx
        a = int(self.skirt_first[t])
        return self.skirt_vertices[a:a + int(self.skirt_count[t])]


# =====================================================================================
# 三角形逐个检查
# =====================================================================================

@njit(cache=True, nogil=True)
def _classify(uvs, positions, tri_ids, tol, shift, flags):
    """检查数值，算 UV 平移量（整数），标记越界与跨界。"""
    lo = -tol
    hi = 1.0 + tol
    for k in range(tri_ids.shape[0]):
        b = tri_ids[k] * 3
        shift[k, 0] = 0.0
        shift[k, 1] = 0.0
        ok = True
        for c in range(3):
            if not (np.isfinite(uvs[b + c, 0]) and np.isfinite(uvs[b + c, 1])):
                ok = False
            if not (np.isfinite(positions[b + c, 0]) and np.isfinite(positions[b + c, 1])
                    and np.isfinite(positions[b + c, 2])):
                ok = False
        if not ok:
            flags[k] = F_INVALID
            continue
        u0 = float(uvs[b, 0])
        v0 = float(uvs[b, 1])
        u1 = float(uvs[b + 1, 0])
        v1 = float(uvs[b + 1, 1])
        u2 = float(uvs[b + 2, 0])
        v2 = float(uvs[b + 2, 1])
        f = 0
        umin = min(u0, min(u1, u2))
        umax = max(u0, max(u1, u2))
        vmin = min(v0, min(v1, v2))
        vmax = max(v0, max(v1, v2))
        if umin < lo or umax > hi or vmin < lo or vmax > hi:
            f |= F_OUT
            su = -math.floor((u0 + u1 + u2) / 3.0)
            sv = -math.floor((v0 + v1 + v2) / 3.0)
            if su != 0.0 or sv != 0.0:
                f |= F_SHIFT
                shift[k, 0] = su
                shift[k, 1] = sv
            if umin + su < lo or umax + su > hi or vmin + sv < lo or vmax + sv > hi:
                f |= F_CROSS
        flags[k] = f


@njit(cache=True, nogil=True)
def _mark_degenerate(uvs, tri_ids, flags):
    """UV 面积恰好为零的三角形打上 F_DEGEN（用最终的 float32 UV 判）。"""
    for k in range(tri_ids.shape[0]):
        if flags[k] & F_INVALID:
            continue
        b = tri_ids[k] * 3
        u0 = float(uvs[b, 0])
        v0 = float(uvs[b, 1])
        area = (uvs[b + 1, 0] - u0) * (uvs[b + 2, 1] - v0) - (uvs[b + 1, 1] - v0) * (uvs[b + 2, 0] - u0)
        if area == 0.0:
            flags[k] |= F_DEGEN


# =====================================================================================
# 三角形与格的相交、裁剪
# =====================================================================================

@njit(cache=True, nogil=True, inline="always")
def _separated(px, py, qx, qy, x0, x1, y0, y1):
    """有向边 p→q（三角形在左侧）所在直线能否把矩形 [x0,x1]×[y0,y1] 隔在外面（允许贴边）。"""
    dx = qx - px
    dy = qy - py
    # 边函数 E(x, y) = dx·(y - py) - dy·(x - px)，三角形内部 E > 0；取矩形上让 E 最大的角
    x = x0 if dy > 0.0 else x1
    y = y1 if dx > 0.0 else y0
    return dx * (y - py) - dy * (x - px) <= 0.0


@njit(cache=True, nogil=True)
def _tri_tiles(ax, ay, bx, by, cx, cy, grid, out):
    """三角形（格单位坐标，格线在整数上）与哪些格的内部相交。格号写进 out，返回个数。

    分离轴判定：x 轴、y 轴由候选范围保证严格重叠，再查三角形三条边的法向。
    """
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if not (area > 0.0 or area < 0.0):
        return 0
    if area < 0.0:
        bx, cx = cx, bx
        by, cy = cy, by
    minx = min(ax, min(bx, cx))
    maxx = max(ax, max(bx, cx))
    miny = min(ay, min(by, cy))
    maxy = max(ay, max(by, cy))
    g = float(grid)
    tx0 = int(math.floor(min(max(minx, 0.0), g)))
    tx1 = int(math.ceil(max(min(maxx, g), 0.0))) - 1
    ty0 = int(math.floor(min(max(miny, 0.0), g)))
    ty1 = int(math.ceil(max(min(maxy, g), 0.0))) - 1
    if tx0 > tx1 or ty0 > ty1:
        return 0
    if tx0 == tx1 and ty0 == ty1 and minx >= tx0 and maxx <= tx0 + 1 and miny >= ty0 and maxy <= ty0 + 1:
        out[0] = ty0 * grid + tx0
        return 1
    n = 0
    for ty in range(ty0, ty1 + 1):
        y0 = float(ty)
        y1 = y0 + 1.0
        for tx in range(tx0, tx1 + 1):
            x0 = float(tx)
            x1 = x0 + 1.0
            if _separated(ax, ay, bx, by, x0, x1, y0, y1):
                continue
            if _separated(bx, by, cx, cy, x0, x1, y0, y1):
                continue
            if _separated(cx, cy, ax, ay, x0, x1, y0, y1):
                continue
            out[n] = ty * grid + tx
            n += 1
    return n


@njit(cache=True, nogil=True)
def _clip_axis(src, n, axis, bound, keep_ge, dst):
    """Sutherland–Hodgman：把多边形 src[:n]（每行 u, v, x, y, z）裁到 coord[axis] >= bound（keep_ge）
    或 <= bound。三维坐标随 UV 线性插值。返回 dst 里的顶点数。"""
    if n == 0:
        return 0
    m = 0
    p = n - 1
    pv = src[p, axis]
    pin = pv >= bound if keep_ge else pv <= bound
    for i in range(n):
        cv = src[i, axis]
        cin = cv >= bound if keep_ge else cv <= bound
        if cin != pin:
            t = (bound - pv) / (cv - pv)
            for d in range(5):
                dst[m, d] = src[p, d] + t * (src[i, d] - src[p, d])
            dst[m, axis] = bound
            m += 1
        if cin:
            for d in range(5):
                dst[m, d] = src[i, d]
            m += 1
        p = i
        pv = cv
        pin = cin
    return m


@njit(cache=True, nogil=True)
def _clip_rect(poly, n, x0, x1, y0, y1, tmp):
    """把 poly[:n] 裁到矩形，结果留在 poly 里，返回顶点数。"""
    n = _clip_axis(poly, n, 0, x0, True, tmp)
    n = _clip_axis(tmp, n, 0, x1, False, poly)
    n = _clip_axis(poly, n, 1, y0, True, tmp)
    n = _clip_axis(tmp, n, 1, y1, False, poly)
    return n


@njit(cache=True, nogil=True)
def _bounds_part(poly, tmp, t, grid, minx, maxx, miny, maxy, bmin, bmax):
    """poly[:3] 是一个三角形（u, v 为格单位，后三列是三维坐标）。把它落在格 t 内那部分的三维范围并进 bmin/bmax。
    返回 False 表示数值上裁空了（只在极端擦边时出现），调用方改用整个三角形。"""
    tx = t % grid
    ty = t // grid
    if minx >= tx and maxx <= tx + 1 and miny >= ty and maxy <= ty + 1:
        m = 3
    else:
        m = _clip_rect(poly, 3, float(tx), float(tx + 1), float(ty), float(ty + 1), tmp)
        if m == 0:
            return False
    for j in range(m):
        for d in range(3):
            x = poly[j, 2 + d]
            if x < bmin[t, d]:
                bmin[t, d] = x
            if x > bmax[t, d]:
                bmax[t, d] = x
    return True


# =====================================================================================
# 主三角形分桶：两遍法
# =====================================================================================

@njit(cache=True, nogil=True)
def _count_pass(uvs, positions, tri_ids, flags, grid, k0, k1, counts, bmin, bmax):
    """第一遍：数每格的三角形个数，同时求每格包围盒。处理 tri_ids[k0:k1]。"""
    tiles = np.empty(grid * grid, np.int64)
    poly = np.empty((16, 5))
    tmp = np.empty((16, 5))
    g = float(grid)
    for k in range(k0, k1):
        if flags[k] & _SKIP:
            continue
        b = tri_ids[k] * 3
        ax = uvs[b, 0] * g
        ay = uvs[b, 1] * g
        bx = uvs[b + 1, 0] * g
        by = uvs[b + 1, 1] * g
        cx = uvs[b + 2, 0] * g
        cy = uvs[b + 2, 1] * g
        n = _tri_tiles(ax, ay, bx, by, cx, cy, grid, tiles)
        if n == 0:
            continue
        minx = min(ax, min(bx, cx))
        maxx = max(ax, max(bx, cx))
        miny = min(ay, min(by, cy))
        maxy = max(ay, max(by, cy))
        for i in range(n):
            t = tiles[i]
            counts[t] += 1
            for c in range(3):
                poly[c, 0] = uvs[b + c, 0] * g
                poly[c, 1] = uvs[b + c, 1] * g
                poly[c, 2] = positions[b + c, 0]
                poly[c, 3] = positions[b + c, 1]
                poly[c, 4] = positions[b + c, 2]
            if not _bounds_part(poly, tmp, t, grid, minx, maxx, miny, maxy, bmin, bmax):
                for c in range(3):
                    for d in range(3):
                        x = positions[b + c, d]
                        if x < bmin[t, d]:
                            bmin[t, d] = x
                        if x > bmax[t, d]:
                            bmax[t, d] = x


@njit(cache=True, nogil=True)
def _fill_pass(uvs, tri_ids, flags, grid, k0, k1, cursor, out):
    """第二遍：按 cursor 给出的写入位置，把三角形的三个角点索引写进每个相交的格。"""
    tiles = np.empty(grid * grid, np.int64)
    g = float(grid)
    for k in range(k0, k1):
        if flags[k] & _SKIP:
            continue
        b = tri_ids[k] * 3
        n = _tri_tiles(uvs[b, 0] * g, uvs[b, 1] * g, uvs[b + 1, 0] * g, uvs[b + 1, 1] * g,
                       uvs[b + 2, 0] * g, uvs[b + 2, 1] * g, grid, tiles)
        for i in range(n):
            t = tiles[i]
            p = cursor[t]
            out[p] = b
            out[p + 1] = b + 1
            out[p + 2] = b + 2
            cursor[t] = p + 3


# =====================================================================================
# 扩边
# =====================================================================================

@njit(cache=True, nogil=True)
def _corner_keys(uvs, tri_ids, act, inv_tol, qu, qv):
    """参与扩边的三角形，每个角点的 UV 量化成整数键（容差 1/inv_tol）。"""
    lim = 4.0e18
    for a in range(act.shape[0]):
        b = tri_ids[act[a]] * 3
        for c in range(3):
            x = math.floor(uvs[b + c, 0] * inv_tol + 0.5)
            y = math.floor(uvs[b + c, 1] * inv_tol + 0.5)
            qu[3 * a + c] = np.int64(min(max(x, -lim), lim))
            qv[3 * a + c] = np.int64(min(max(y, -lim), lim))


@njit(cache=True, nogil=True)
def _edge_keys(uvs, tri_ids, act, wid, nv, ekey, eside):
    """每条边的使用记录：无向边键 lo·nv+hi，以及三角形在规范方向（lo→hi）的左侧(1)还是右侧(0)。"""
    for a in range(act.shape[0]):
        b = tri_ids[act[a]] * 3
        u0 = float(uvs[b, 0])
        v0 = float(uvs[b, 1])
        ccw = (uvs[b + 1, 0] - u0) * (uvs[b + 2, 1] - v0) - (uvs[b + 1, 1] - v0) * (uvs[b + 2, 0] - u0) > 0.0
        for i in range(3):
            e = 3 * a + i
            va = wid[3 * a + i]
            vb = wid[3 * a + (i + 1) % 3]
            if va == vb:
                ekey[e] = -1
                eside[e] = 0
                continue
            if va < vb:
                ekey[e] = va * nv + vb
                eside[e] = 1 if ccw else 0
            else:
                ekey[e] = vb * nv + va
                eside[e] = 0 if ccw else 1


@njit(cache=True, nogil=True)
def _boundary_uses(ekey, eside, order):
    """按键排好序后逐组看：同一条 UV 边只在一侧有三角形，就是岛的边界。返回边界上的边使用记录号。

    「只被一个三角形使用」是它的特例；两片镜像重叠的 UV 在同一侧共用一条边时，这条边也算边界。
    """
    n = order.shape[0]
    flag = np.zeros(n, np.bool_)
    i = 0
    while i < n:
        key = ekey[order[i]]
        j = i
        left = False
        right = False
        while j < n and ekey[order[j]] == key:
            if eside[order[j]] == 1:
                left = True
            else:
                right = True
            j += 1
        if key >= 0 and not (left and right):
            for q in range(i, j):
                flag[order[q]] = True
        i = j
    return np.nonzero(flag)[0]


@njit(cache=True, nogil=True, inline="always")
def _put(out, s, c, u, v, positions, normals, corner):
    out[s, c, 0] = u
    out[s, c, 1] = v
    out[s, c, 2] = positions[corner, 0]
    out[s, c, 3] = positions[corner, 1]
    out[s, c, 4] = positions[corner, 2]
    out[s, c, 5] = normals[corner, 0]
    out[s, c, 6] = normals[corner, 1]
    out[s, c, 7] = normals[corner, 2]


@njit(cache=True, nogil=True)
def _skirt_quads(uvs, positions, normals, tri_ids, act, wid, buses, width, quads,
                 valid, e_from, e_to, e_dir, e_nrm, e_uvto, e_cto):
    """每条边界边向岛外（远离所属三角形第三个顶点的一侧）挤出宽 width 的四边形，拆成两个三角形。

    边先定向成「三角形在左侧」，外法线就是右侧。四个顶点的位置和法线取所在端点的值。
    同时记下每条边的端点（焊接后的顶点号）、方向、外法线，供转角补扇形用。
    """
    for q in range(buses.shape[0]):
        e = buses[q]
        a = e // 3
        i = e - 3 * a
        b = tri_ids[act[a]] * 3
        ca = b + i
        cb = b + (i + 1) % 3
        cc = b + (i + 2) % 3
        wa = wid[3 * a + i]
        wb = wid[3 * a + (i + 1) % 3]
        au = float(uvs[ca, 0])
        av = float(uvs[ca, 1])
        bu = float(uvs[cb, 0])
        bv = float(uvs[cb, 1])
        cu = float(uvs[cc, 0])
        cv = float(uvs[cc, 1])
        if (bu - au) * (cv - av) - (bv - av) * (cu - au) < 0.0:
            au, bu = bu, au
            av, bv = bv, av
            ca, cb = cb, ca
            wa, wb = wb, wa
        dx = bu - au
        dy = bv - av
        L = math.sqrt(dx * dx + dy * dy)
        s0 = 2 * q
        if not (L > 0.0):
            valid[q] = False
            for s in range(s0, s0 + 2):
                for c in range(3):
                    _put(quads, s, c, au, av, positions, normals, ca)
            continue
        nx = dy / L
        ny = -dx / L
        ox = nx * width
        oy = ny * width
        # 外侧 a' = a + o，b' = b + o；三角形 (a', b', b)、(a', b, a)，都是逆时针
        _put(quads, s0, 0, au + ox, av + oy, positions, normals, ca)
        _put(quads, s0, 1, bu + ox, bv + oy, positions, normals, cb)
        _put(quads, s0, 2, bu, bv, positions, normals, cb)
        _put(quads, s0 + 1, 0, au + ox, av + oy, positions, normals, ca)
        _put(quads, s0 + 1, 1, bu, bv, positions, normals, cb)
        _put(quads, s0 + 1, 2, au, av, positions, normals, ca)
        valid[q] = True
        e_from[q] = wa
        e_to[q] = wb
        e_dir[q, 0] = dx
        e_dir[q, 1] = dy
        e_nrm[q, 0] = nx
        e_nrm[q, 1] = ny
        e_uvto[q, 0] = bu
        e_uvto[q, 1] = bv
        e_cto[q] = cb


@njit(cache=True, nogil=True)
def _skirt_joins(valid, vf, vt, e_dir, e_nrm, e_uvto, e_cto, positions, normals, width, nverts, step):
    """岛的凸角处，相邻两条边的扩边之间会空出一个楔形，用扇形三角形补上。

    只处理恰好一进一出两条边界边的顶点（普通的岛边界都是这样）。返回 (J, 3, 8)。
    """
    nb = valid.shape[0]
    in_cnt = np.zeros(nverts, np.int64)
    out_cnt = np.zeros(nverts, np.int64)
    in_e = np.full(nverts, -1, np.int64)
    out_e = np.full(nverts, -1, np.int64)
    for q in range(nb):
        if not valid[q]:
            continue
        out_cnt[vf[q]] += 1
        out_e[vf[q]] = q
        in_cnt[vt[q]] += 1
        in_e[vt[q]] = q
    total = 0
    for v in range(nverts):
        if in_cnt[v] == 1 and out_cnt[v] == 1:
            qi = in_e[v]
            qo = out_e[v]
            turn = e_dir[qi, 0] * e_dir[qo, 1] - e_dir[qi, 1] * e_dir[qo, 0]
            if turn > 0.0:
                phi = math.atan2(e_nrm[qi, 0] * e_nrm[qo, 1] - e_nrm[qi, 1] * e_nrm[qo, 0],
                                 e_nrm[qi, 0] * e_nrm[qo, 0] + e_nrm[qi, 1] * e_nrm[qo, 1])
                if phi > 1e-6:
                    total += max(1, int(math.ceil(phi / step - 1e-9)))
    out = np.empty((total, 3, 8))
    m = 0
    for v in range(nverts):
        if not (in_cnt[v] == 1 and out_cnt[v] == 1):
            continue
        qi = in_e[v]
        qo = out_e[v]
        turn = e_dir[qi, 0] * e_dir[qo, 1] - e_dir[qi, 1] * e_dir[qo, 0]
        if not (turn > 0.0):
            continue
        n0x = e_nrm[qi, 0]
        n0y = e_nrm[qi, 1]
        n1x = e_nrm[qo, 0]
        n1y = e_nrm[qo, 1]
        phi = math.atan2(n0x * n1y - n0y * n1x, n0x * n1x + n0y * n1y)
        if not (phi > 1e-6):
            continue
        ns = max(1, int(math.ceil(phi / step - 1e-9)))
        cu = e_uvto[qi, 0]
        cv = e_uvto[qi, 1]
        corner = e_cto[qi]
        for s in range(ns):
            if s == 0:
                r0x = n0x
                r0y = n0y
            else:
                ang = phi * s / ns
                r0x = math.cos(ang) * n0x - math.sin(ang) * n0y
                r0y = math.sin(ang) * n0x + math.cos(ang) * n0y
            if s == ns - 1:
                r1x = n1x
                r1y = n1y
            else:
                ang = phi * (s + 1) / ns
                r1x = math.cos(ang) * n0x - math.sin(ang) * n0y
                r1y = math.sin(ang) * n0x + math.cos(ang) * n0y
            _put(out, m, 0, cu, cv, positions, normals, corner)
            _put(out, m, 1, cu + width * r0x, cv + width * r0y, positions, normals, corner)
            _put(out, m, 2, cu + width * r1x, cv + width * r1y, positions, normals, corner)
            m += 1
    return out


@njit(cache=True, nogil=True)
def _soup_count(tris, grid, counts, bmin, bmax, with_bounds):
    """扩边三角形（UV 单位）分桶第一遍：数个数，可选求包围盒。"""
    tiles = np.empty(grid * grid, np.int64)
    poly = np.empty((16, 5))
    tmp = np.empty((16, 5))
    g = float(grid)
    for s in range(tris.shape[0]):
        ax = tris[s, 0, 0] * g
        ay = tris[s, 0, 1] * g
        bx = tris[s, 1, 0] * g
        by = tris[s, 1, 1] * g
        cx = tris[s, 2, 0] * g
        cy = tris[s, 2, 1] * g
        n = _tri_tiles(ax, ay, bx, by, cx, cy, grid, tiles)
        if n == 0:
            continue
        minx = min(ax, min(bx, cx))
        maxx = max(ax, max(bx, cx))
        miny = min(ay, min(by, cy))
        maxy = max(ay, max(by, cy))
        for i in range(n):
            t = tiles[i]
            counts[t] += 1
            if not with_bounds:
                continue
            for c in range(3):
                poly[c, 0] = tris[s, c, 0] * g
                poly[c, 1] = tris[s, c, 1] * g
                poly[c, 2] = tris[s, c, 2]
                poly[c, 3] = tris[s, c, 3]
                poly[c, 4] = tris[s, c, 4]
            if not _bounds_part(poly, tmp, t, grid, minx, maxx, miny, maxy, bmin, bmax):
                for c in range(3):
                    for d in range(3):
                        x = tris[s, c, 2 + d]
                        if x < bmin[t, d]:
                            bmin[t, d] = x
                        if x > bmax[t, d]:
                            bmax[t, d] = x


@njit(cache=True, nogil=True)
def _soup_fill(tris, grid, cursor, out):
    """扩边三角形分桶第二遍：把顶点写进每个相交的格。"""
    tiles = np.empty(grid * grid, np.int64)
    g = float(grid)
    for s in range(tris.shape[0]):
        n = _tri_tiles(tris[s, 0, 0] * g, tris[s, 0, 1] * g, tris[s, 1, 0] * g, tris[s, 1, 1] * g,
                       tris[s, 2, 0] * g, tris[s, 2, 1] * g, grid, tiles)
        for i in range(n):
            t = tiles[i]
            p = cursor[t]
            for c in range(3):
                for d in range(8):
                    out[p + c, d] = tris[s, c, d]
            cursor[t] = p + 3


# =====================================================================================
# UV 重叠估计
# =====================================================================================

@njit(cache=True, nogil=True)
def _overlap_raster(uvs, tri_ids, flags, n):
    """把三角形光栅化到 n×n 的网格，按纹素中心数覆盖次数。返回 (被覆盖的纹素数, 被覆盖两次以上的纹素数)。

    用 8 位亚像素的整数坐标，边上的点按固定规则只归一侧，共边的两个三角形不会重复计数，
    所以互不重叠的 UV 估计值恰好为 0。
    """
    S = 256
    half = 128
    scale = float(n * S)
    lim = 16.0
    cover = np.zeros((n, n), np.uint8)
    for k in range(tri_ids.shape[0]):
        if flags[k] & _SKIP:
            continue
        b = tri_ids[k] * 3
        u0 = float(uvs[b, 0])
        v0 = float(uvs[b, 1])
        u1 = float(uvs[b + 1, 0])
        v1 = float(uvs[b + 1, 1])
        u2 = float(uvs[b + 2, 0])
        v2 = float(uvs[b + 2, 1])
        if (abs(u0) > lim or abs(u1) > lim or abs(u2) > lim or abs(v0) > lim or abs(v1) > lim
                or abs(v2) > lim):
            continue
        X0 = np.int64(math.floor(u0 * scale + 0.5))
        Y0 = np.int64(math.floor(v0 * scale + 0.5))
        X1 = np.int64(math.floor(u1 * scale + 0.5))
        Y1 = np.int64(math.floor(v1 * scale + 0.5))
        X2 = np.int64(math.floor(u2 * scale + 0.5))
        Y2 = np.int64(math.floor(v2 * scale + 0.5))
        area = (X1 - X0) * (Y2 - Y0) - (Y1 - Y0) * (X2 - X0)
        if area == 0:
            continue
        if area < 0:
            X1, X2 = X2, X1
            Y1, Y2 = Y2, Y1
        minX = min(X0, min(X1, X2))
        maxX = max(X0, max(X1, X2))
        minY = min(Y0, min(Y1, Y2))
        maxY = max(Y0, max(Y1, Y2))
        px0 = -((half - minX) // S)          # ceil((minX - half) / S)
        px1 = (maxX - half) // S
        py0 = -((half - minY) // S)
        py1 = (maxY - half) // S
        if px0 < 0:
            px0 = 0
        if py0 < 0:
            py0 = 0
        if px1 > n - 1:
            px1 = n - 1
        if py1 > n - 1:
            py1 = n - 1
        dx0 = X1 - X0
        dy0 = Y1 - Y0
        dx1 = X2 - X1
        dy1 = Y2 - Y1
        dx2 = X0 - X2
        dy2 = Y0 - Y2
        # 边上的点归哪一侧：等价于把采样点朝 (1, ε) 方向挪一点点再判严格在内
        tl0 = dy0 < 0 or (dy0 == 0 and dx0 > 0)
        tl1 = dy1 < 0 or (dy1 == 0 and dx1 > 0)
        tl2 = dy2 < 0 or (dy2 == 0 and dx2 > 0)
        for py in range(py0, py1 + 1):
            cy = py * S + half
            for px in range(px0, px1 + 1):
                cx = px * S + half
                w0 = dx0 * (cy - Y0) - dy0 * (cx - X0)
                if w0 < 0 or (w0 == 0 and not tl0):
                    continue
                w1 = dx1 * (cy - Y1) - dy1 * (cx - X1)
                if w1 < 0 or (w1 == 0 and not tl1):
                    continue
                w2 = dx2 * (cy - Y2) - dy2 * (cx - X2)
                if w2 < 0 or (w2 == 0 and not tl2):
                    continue
                if cover[py, px] < 255:
                    cover[py, px] += 1
    covered = 0
    over = 0
    for y in range(n):
        for x in range(n):
            c = cover[y, x]
            if c >= 1:
                covered += 1
                if c >= 2:
                    over += 1
    return covered, over


# =====================================================================================
# 入口
# =====================================================================================

def _workers(threads: int) -> int:
    if threads and threads > 0:
        return int(threads)
    return max(1, min(8, os.cpu_count() or 1))


def _run_jobs(jobs, workers: int):
    """依次或用线程池执行一组无参函数。numba 函数带 nogil，线程能真正并行。"""
    if workers <= 1 or len(jobs) <= 1:
        return [job() for job in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(lambda job: job(), jobs))


def _chunks(n: int, workers: int, min_size: int = 1 << 15) -> list[tuple[int, int]]:
    k = max(1, min(workers, n // min_size))
    edges = np.linspace(0, n, k + 1).astype(np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]


def _as_f32(arr, ncol: int) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float32).reshape(-1, ncol))


def _bounds_f32(lo64: np.ndarray, hi64: np.ndarray):
    """float64 包围盒转 float32，向外取整保证仍然包住。"""
    lo = lo64.astype(np.float32)
    hi = hi64.astype(np.float32)
    m = np.isfinite(lo64) & (lo.astype(np.float64) > lo64)
    lo[m] = np.nextafter(lo[m], np.float32(-np.inf))
    m = np.isfinite(hi64) & (hi.astype(np.float64) < hi64)
    hi[m] = np.nextafter(hi[m], np.float32(np.inf))
    return np.ascontiguousarray(lo), np.ascontiguousarray(hi)


def _prefix(counts_per_tile: np.ndarray) -> np.ndarray:
    first = np.zeros(counts_per_tile.shape[0], np.int64)
    if counts_per_tile.shape[0] > 1:
        first[1:] = np.cumsum(counts_per_tile)[:-1]
    return first


def _build_skirt(uvs, pos, nrm, tri_ids, flags, width, weld_tol, corners):
    """找 UV 边界边，生成扩边三角形 (S, 3, 8)。返回 (三角形, 边界边数, 补角三角形数)。"""
    empty = np.zeros((0, 3, 8))
    act = np.flatnonzero((flags & _SKIP) == 0).astype(np.int64)
    if act.size == 0 or not width > 0.0:
        return empty, 0, 0
    A = act.size
    qu = np.empty(3 * A, np.int64)
    qv = np.empty(3 * A, np.int64)
    _corner_keys(uvs, tri_ids, act, 1.0 / weld_tol, qu, qv)
    umin, umax = int(qu.min()), int(qu.max())
    vmin, vmax = int(qv.min()), int(qv.max())
    ru, rv = umax - umin + 1, vmax - vmin + 1
    if ru * rv < (1 << 62):
        keys = (qu - umin) * rv + (qv - vmin)
        uniq, wid = np.unique(keys, return_inverse=True)
    else:
        uniq, wid = np.unique(np.stack([qu, qv], axis=1), axis=0, return_inverse=True)
    wid = np.ascontiguousarray(wid.reshape(-1), dtype=np.int64)
    nv = int(uniq.shape[0])
    ekey = np.empty(3 * A, np.int64)
    eside = np.empty(3 * A, np.uint8)
    _edge_keys(uvs, tri_ids, act, wid, nv, ekey, eside)
    order = np.argsort(ekey, kind="stable")
    buses = _boundary_uses(ekey, eside, order)
    nb = int(buses.shape[0])
    if nb == 0:
        return empty, 0, 0
    quads = np.empty((2 * nb, 3, 8))
    valid = np.zeros(nb, np.bool_)
    e_from = np.full(nb, -1, np.int64)
    e_to = np.full(nb, -1, np.int64)
    e_dir = np.zeros((nb, 2))
    e_nrm = np.zeros((nb, 2))
    e_uvto = np.zeros((nb, 2))
    e_cto = np.zeros(nb, np.int64)
    _skirt_quads(uvs, pos, nrm, tri_ids, act, wid, buses, float(width), quads,
                 valid, e_from, e_to, e_dir, e_nrm, e_uvto, e_cto)
    n_edges = int(valid.sum())
    if not corners or n_edges == 0:
        return quads, n_edges, 0
    verts = np.unique(np.concatenate([e_from[valid], e_to[valid]]))
    vf = np.searchsorted(verts, e_from).astype(np.int64)
    vt = np.searchsorted(verts, e_to).astype(np.int64)
    vf[~valid] = 0
    vt[~valid] = 0
    joins = _skirt_joins(valid, vf, vt, e_dir, e_nrm, e_uvto, e_cto, pos, nrm, float(width),
                         int(verts.shape[0]), _JOIN_STEP)
    return np.concatenate([quads, joins]), n_edges, int(joins.shape[0])


def build_set_geometry(mesh, material_index: int, resolution: int, skirt_texels: float = 2.0, *,
                       skirt_corners: bool = True, skirt_in_bounds: bool = True,
                       weld_tolerance: float = 1e-6, uv_tolerance: float = 1e-5,
                       overlap_grid: int = 1024, overlap_warn_ratio: float = 0.001,
                       threads: int = 0) -> SetGeometry:
    """为 mesh 里材质 material_index 的三角形生成分辨率 resolution 下的绘制几何。

    resolution：贴图边长，256 的整数倍（通常 512 到 16384），格数 grid = resolution // 256。
    skirt_texels：扩边宽度（纹素），0 表示不扩边。
    skirt_corners：在岛的凸角处用扇形补上相邻扩边之间的楔形空隙。
    skirt_in_bounds：每格包围盒是否把落在该格内的扩边也算进去（扩边会写进这一格，按包围盒挑格时不能漏掉它）。
    weld_tolerance：找边界边时把 UV 角点焊在一起的容差（UV 单位）。
    uv_tolerance：UV 超出 [0, 1] 多少才算越界。
    overlap_grid：估计 UV 重叠用的网格边长。
    overlap_warn_ratio：重叠比例超过它时在 warnings 里提示。
    threads：线程数，0 表示自动。

    扩边三角形应先于模型三角形绘制，或只写没被模型三角形覆盖的纹素，免得盖住相邻岛的内容。
    """
    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    res = int(resolution)
    if res < TILE or res % TILE != 0 or res > 65536:
        raise ValueError(f"分辨率必须是 {TILE} 的整数倍且不超过 65536，收到 {resolution}")
    if not weld_tolerance > 0.0:
        raise ValueError("weld_tolerance 必须大于 0")
    grid = res // TILE
    ntiles = grid * grid
    uvs_in = _as_f32(mesh.uvs, 2)
    pos = _as_f32(mesh.positions, 3)
    nrm = _as_f32(mesh.normals, 3)
    mids = np.asarray(mesh.material_ids).reshape(-1)
    T = int(mids.shape[0])
    if uvs_in.shape[0] != 3 * T or pos.shape[0] != 3 * T or nrm.shape[0] != 3 * T:
        raise ValueError("MeshData 的数组长度和三角形数对不上")
    tri_ids = np.flatnonzero(mids == int(material_index)).astype(np.int64)
    n = int(tri_ids.shape[0])
    warnings: list[str] = []
    workers = _workers(threads)

    # ---- 逐三角形检查、UV 平移 ----
    t0 = time.perf_counter()
    shift = np.zeros((n, 2))
    flags = np.zeros(n, np.uint8)
    _classify(uvs_in, pos, tri_ids, float(uv_tolerance), shift, flags)
    n_shift = int(np.count_nonzero(flags & F_SHIFT))
    wrapped = None
    uvs = uvs_in
    if n_shift:
        wrapped = uvs_in.copy()
        sel = np.flatnonzero(flags & F_SHIFT)
        view = wrapped.reshape(-1, 3, 2)
        view[tri_ids[sel]] += shift[sel][:, None, :].astype(np.float32)
        uvs = wrapped
    _mark_degenerate(uvs, tri_ids, flags)
    n_invalid = int(np.count_nonzero(flags & F_INVALID))
    n_degen = int(np.count_nonzero(flags & F_DEGEN))
    n_out = int(np.count_nonzero(flags & F_OUT))
    n_cross = int(np.count_nonzero(flags & F_CROSS))
    timings["classify"] = time.perf_counter() - t0

    # ---- 主三角形分桶 ----
    t0 = time.perf_counter()
    chunks = _chunks(n, workers)
    nch = len(chunks)
    counts = np.zeros((nch, ntiles), np.int64)
    bmin = np.full((nch, ntiles, 3), np.inf)
    bmax = np.full((nch, ntiles, 3), -np.inf)
    _run_jobs([lambda c=c, a=a, b=b: _count_pass(uvs, pos, tri_ids, flags, grid, a, b, counts[c], bmin[c], bmax[c])
               for c, (a, b) in enumerate(chunks)], workers)
    per_tile = counts.sum(axis=0)
    entries = int(per_tile.sum())
    if entries * 3 > np.iinfo(np.int32).max:
        raise ValueError("这个材质的三角形太多，超出单个纹理集的索引上限")
    tile_first64 = _prefix(per_tile * 3)
    chunk_before = np.zeros_like(counts)
    if nch > 1:
        chunk_before[1:] = np.cumsum(counts, axis=0)[:-1]
    cursor = np.ascontiguousarray(tile_first64[None, :] + 3 * chunk_before)
    indices = np.empty(entries * 3, np.uint32)
    _run_jobs([lambda c=c, a=a, b=b: _fill_pass(uvs, tri_ids, flags, grid, a, b, cursor[c], indices)
               for c, (a, b) in enumerate(chunks)], workers)
    lo64 = bmin.min(axis=0)
    hi64 = bmax.max(axis=0)
    timings["tiles"] = time.perf_counter() - t0

    # ---- 扩边 ----
    t0 = time.perf_counter()
    width = float(skirt_texels) / res if skirt_texels and skirt_texels > 0 else 0.0
    skirt_tris, n_edges, n_joins = _build_skirt(uvs, pos, nrm, tri_ids, flags, width,
                                                float(weld_tolerance), bool(skirt_corners))
    # 先取成 float32 再分桶：分桶依据的就是交给显卡的数据本身
    skirt_tris = skirt_tris.astype(np.float32).astype(np.float64)
    timings["skirt_build"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sk_counts = np.zeros(ntiles, np.int64)
    sk_lo = np.full((ntiles, 3), np.inf)
    sk_hi = np.full((ntiles, 3), -np.inf)
    if skirt_tris.shape[0]:
        _soup_count(skirt_tris, grid, sk_counts, sk_lo, sk_hi, bool(skirt_in_bounds))
    sk_first64 = _prefix(sk_counts * 3)
    skirt_vertices = np.empty((int(sk_counts.sum()) * 3, 8), np.float32)
    if skirt_tris.shape[0]:
        _soup_fill(skirt_tris, grid, sk_first64.copy(), skirt_vertices)
    if skirt_in_bounds:
        lo64 = np.minimum(lo64, sk_lo)
        hi64 = np.maximum(hi64, sk_hi)
    tile_min, tile_max = _bounds_f32(lo64, hi64)
    timings["skirt_tiles"] = time.perf_counter() - t0

    # ---- UV 重叠 ----
    t0 = time.perf_counter()
    og = max(16, int(overlap_grid))
    covered, overlapped = _overlap_raster(uvs, tri_ids, flags, og)
    overlap_ratio = float(overlapped) / float(covered) if covered else 0.0
    timings["overlap"] = time.perf_counter() - t0

    # ---- 提示 ----
    if n == 0:
        warnings.append("这个材质没有三角形")
    if n_invalid:
        warnings.append(f"{n_invalid} 个三角形的坐标或 UV 不是有效数值，已跳过")
    if n_degen:
        warnings.append(f"{n_degen} 个三角形的 UV 面积为零，在贴图上画不到")
    if n_out:
        warnings.append(f"{n_out} 个三角形的 UV 超出 0 到 1 的范围，已按所在整数格平移回来")
    if n_cross:
        warnings.append(f"{n_cross} 个三角形跨越 UV 的整数边界，超出 0 到 1 的部分画不到")
    if overlap_ratio > overlap_warn_ratio:
        warnings.append(f"UV 有重叠：约 {overlap_ratio:.1%} 的已用纹素被多处表面共用，这些地方画一处会同时改变另一处")

    timings["total"] = time.perf_counter() - t_start
    stats = {
        "triangles": n,
        "painted_triangles": n - n_invalid - n_degen,
        "invalid_triangles": n_invalid,
        "uv_degenerate_triangles": n_degen,
        "out_of_range_triangles": n_out,
        "shifted_triangles": n_shift,
        "crossing_triangles": n_cross,
        "tile_entries": entries,
        "occupied_tiles": int(np.count_nonzero(per_tile)),
        "boundary_edges": n_edges,
        "skirt_join_triangles": n_joins,
        "skirt_triangles": int(skirt_tris.shape[0]),
        "skirt_entries": int(sk_counts.sum()),
        "overlap_grid": og,
        "covered_texels": int(covered),
        "overlapped_texels": int(overlapped),
        "threads": workers,
        "chunks": nch,
        "seconds": timings,
    }
    return SetGeometry(
        material_index=int(material_index), resolution=res, grid=grid,
        indices=indices,
        tile_first=tile_first64.astype(np.int32), tile_count=(per_tile * 3).astype(np.int32),
        tile_min=tile_min, tile_max=tile_max,
        skirt_vertices=skirt_vertices,
        skirt_first=sk_first64.astype(np.int32), skirt_count=(sk_counts * 3).astype(np.int32),
        overlap_ratio=overlap_ratio, uv_out_of_range=n_out > 0,
        uvs=wrapped, warnings=warnings, stats=stats)
