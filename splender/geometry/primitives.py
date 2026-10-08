"""网格基本体（和 Blender「添加 → 网格」一样的默认尺寸和分段），都带 UV。

坐标按 Blender 的 Z 朝上写（显示坐标），最后统一换成内部的 Y 朝上。返回逐角点的 MeshData；
四边形记在 polygon_sizes 里（导出时写成四边形）。
"""
from __future__ import annotations

import math

import numpy as np

from ..doc.meshio import MeshData
from ..doc.objects import D


def _finish(name: str, corners: np.ndarray, uvs: np.ndarray, sizes: list[int], smooth: bool) -> MeshData:
    """corners：逐角点的显示坐标 (3T,3)；sizes：每个多边形几个角（四边形拆成 (a,b,c)(a,c,d)）。"""
    p = np.asarray(corners, np.float64) @ D            # 显示 → 内部：内部 = Dᵀ·显示，行向量写成 p @ D
    tris = p.reshape(-1, 3, 3)
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    face_n /= np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), 1e-30)
    if smooth:
        from ..sculpt.topology import vertex_normals, weld

        mesh = weld(p.astype(np.float32), None, None, tolerance=1e-7, drop_degenerate=False)
        vn = vertex_normals(mesh.vertices, mesh.triangles)
        normals = vn[mesh.triangles.reshape(-1)]
    else:
        normals = np.repeat(face_n, 3, axis=0)
    positions = p.astype(np.float32)
    data = MeshData(name=name, positions=positions, normals=np.asarray(normals, np.float32),
                    uvs=np.asarray(uvs, np.float32).reshape(-1, 2), material_ids=np.zeros(len(tris), np.int32),
                    materials=["材质"], bounds_min=positions.min(axis=0), bounds_max=positions.max(axis=0))
    data.polygon_sizes = np.asarray(sizes, np.int32)
    return data


class _Builder:
    def __init__(self) -> None:
        self.corners: list = []
        self.uvs: list = []
        self.sizes: list[int] = []

    def quad(self, a, b, c, d, ua, ub, uc, ud) -> None:
        self.corners += [a, b, c, a, c, d]
        self.uvs += [ua, ub, uc, ua, uc, ud]
        self.sizes.append(4)

    def tri(self, a, b, c, ua, ub, uc) -> None:
        self.corners += [a, b, c]
        self.uvs += [ua, ub, uc]
        self.sizes.append(3)

    def build(self, name: str, smooth: bool) -> MeshData:
        return _finish(name, np.asarray(self.corners, np.float64), np.asarray(self.uvs, np.float64), self.sizes,
                       smooth)


def plane(size: float = 2.0) -> MeshData:
    s = size * 0.5
    b = _Builder()
    b.quad((-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0), (0, 0), (1, 0), (1, 1), (0, 1))
    return b.build("平面", False)


def grid(x_subdivisions: int = 10, y_subdivisions: int = 10, size: float = 2.0) -> MeshData:
    nx, ny = max(1, int(x_subdivisions)), max(1, int(y_subdivisions))
    b = _Builder()
    for j in range(ny):
        for i in range(nx):
            u0, u1, v0, v1 = i / nx, (i + 1) / nx, j / ny, (j + 1) / ny
            p = lambda u, v: ((u - 0.5) * size, (v - 0.5) * size, 0.0)  # noqa: E731
            b.quad(p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1), (u0, v0), (u1, v0), (u1, v1), (u0, v1))
    return b.build("栅格", False)


def cube(size: float = 2.0) -> MeshData:
    """六个面按 Blender 立方体的十字 UV 排。"""
    s = size * 0.5
    b = _Builder()
    # 每个面：四个角（逆时针，从外面看）和 UV 矩形 (u0, v0, u1, v1)
    faces = [
        (((-s, -s, -s), (-s, -s, s), (-s, s, s), (-s, s, -s)), (0.375, 0.0, 0.625, 0.25)),     # -X
        (((-s, s, -s), (-s, s, s), (s, s, s), (s, s, -s)), (0.375, 0.25, 0.625, 0.5)),       # +Y
        (((s, s, -s), (s, s, s), (s, -s, s), (s, -s, -s)), (0.375, 0.5, 0.625, 0.75)),       # +X
        (((s, -s, -s), (s, -s, s), (-s, -s, s), (-s, -s, -s)), (0.375, 0.75, 0.625, 1.0)),   # -Y
        (((-s, s, -s), (s, s, -s), (s, -s, -s), (-s, -s, -s)), (0.125, 0.5, 0.375, 0.75)),   # -Z
        (((s, s, s), (-s, s, s), (-s, -s, s), (s, -s, s)), (0.625, 0.5, 0.875, 0.75)),       # +Z
    ]
    for (a, bb, c, d), (u0, v0, u1, v1) in faces:
        b.quad(a, bb, c, d, (u0, v0), (u1, v0), (u1, v1), (u0, v1))
    return b.build("立方体", False)


def circle(vertices: int = 32, radius: float = 1.0, fill: bool = True) -> MeshData:
    n = max(3, int(vertices))
    b = _Builder()
    ring = [(math.cos(2 * math.pi * i / n) * radius, math.sin(2 * math.pi * i / n) * radius, 0.0) for i in range(n)]
    uv = [(0.5 + 0.5 * math.cos(2 * math.pi * i / n), 0.5 + 0.5 * math.sin(2 * math.pi * i / n)) for i in range(n)]
    for i in range(n):
        j = (i + 1) % n
        b.tri((0.0, 0.0, 0.0), ring[i], ring[j], (0.5, 0.5), uv[i], uv[j])
    return b.build("圆", False)


def uv_sphere(segments: int = 32, rings: int = 16, radius: float = 1.0) -> MeshData:
    seg, rng = max(3, int(segments)), max(3, int(rings))
    b = _Builder()

    def p(i, j):
        # 接缝处和两极用完全相同的坐标（焊接、线框、平滑法线都靠位置相同来认同一个点）
        if j == 0:
            return (0.0, 0.0, -radius)
        if j == rng:
            return (0.0, 0.0, radius)
        theta = 2 * math.pi * (i % seg) / seg
        phi = math.pi * j / rng - math.pi / 2
        return (math.cos(phi) * math.cos(theta) * radius, math.cos(phi) * math.sin(theta) * radius,
                math.sin(phi) * radius)

    for j in range(rng):
        for i in range(seg):
            u0, u1, v0, v1 = i / seg, (i + 1) / seg, j / rng, (j + 1) / rng
            if j == 0:
                b.tri(p(i, 0), p(i + 1, 1), p(i, 1), ((u0 + u1) / 2, v0), (u1, v1), (u0, v1))
            elif j == rng - 1:
                b.tri(p(i, j), p(i + 1, j), p(i, j + 1), (u0, v0), (u1, v0), ((u0 + u1) / 2, v1))
            else:
                b.quad(p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1), (u0, v0), (u1, v0), (u1, v1), (u0, v1))
    return b.build("球体", True)


def ico_sphere(subdivisions: int = 2, radius: float = 1.0) -> MeshData:
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
             (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [np.asarray(v, np.float64) / np.linalg.norm(v) for v in verts]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
             (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
             (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(max(0, int(subdivisions) - 1)):
        cache: dict = {}
        new_faces = []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    builder = _Builder()
    for a, b, c in faces:
        pa, pb, pc = verts[a] * radius, verts[b] * radius, verts[c] * radius
        uvs = []
        for q in (pa, pb, pc):
            n = q / radius
            uvs.append((0.5 + math.atan2(n[1], n[0]) / (2 * math.pi), 0.5 + math.asin(max(-1.0, min(1.0, n[2]))) / math.pi))
        # 跨过 UV 接缝的三角形：把小于 0.25 的 u 加 1，避免拉满一整圈
        us = [u for u, _v in uvs]
        if max(us) - min(us) > 0.5:
            uvs = [(u + 1.0 if u < 0.5 else u, v) for u, v in uvs]
            uvs = [(u * 0.5, v) for u, v in uvs]
        builder.tri(tuple(pa), tuple(pb), tuple(pc), *uvs)
    return builder.build("棱角球", False)


def cylinder(vertices: int = 32, radius: float = 1.0, depth: float = 2.0, cap: bool = True) -> MeshData:
    return _lathe("柱体", vertices, radius, radius, depth, cap)


def cone(vertices: int = 32, radius1: float = 1.0, radius2: float = 0.0, depth: float = 2.0) -> MeshData:
    return _lathe("锥体", vertices, radius1, radius2, depth, True)


def _lathe(name: str, vertices: int, r_bottom: float, r_top: float, depth: float, cap: bool) -> MeshData:
    n = max(3, int(vertices))
    h = depth * 0.5
    b = _Builder()
    ang = [2 * math.pi * (i % n) / n for i in range(n + 1)]       # 接缝处的点和起点坐标完全相同
    bottom = [(math.cos(a) * r_bottom, math.sin(a) * r_bottom, -h) for a in ang]
    top = [(math.cos(a) * r_top, math.sin(a) * r_top, h) for a in ang]
    for i in range(n):
        u0, u1 = i / n * 0.5, (i + 1) / n * 0.5     # 侧面占 UV 左半边
        if r_top <= 1e-9:
            b.tri(bottom[i], bottom[i + 1], top[i], (u0, 0.0), (u1, 0.0), ((u0 + u1) / 2, 1.0))
        else:
            b.quad(bottom[i], bottom[i + 1], top[i + 1], top[i], (u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0))
    if cap:
        def cap_uv(a, cx, cy):
            return (cx + 0.12 * math.cos(a), cy + 0.12 * math.sin(a))

        for i in range(n):
            a0, a1 = ang[i], ang[i + 1]
            if r_bottom > 1e-9:
                b.tri((0.0, 0.0, -h), bottom[i + 1], bottom[i], (0.75, 0.25), cap_uv(a1, 0.75, 0.25),
                      cap_uv(a0, 0.75, 0.25))
            if r_top > 1e-9:
                b.tri((0.0, 0.0, h), top[i], top[i + 1], (0.75, 0.75), cap_uv(a0, 0.75, 0.75),
                      cap_uv(a1, 0.75, 0.75))
    data = b.build(name, False)
    # 侧面用平滑法线（顶、底面保持平的）
    return data


def torus(major_segments: int = 48, minor_segments: int = 12, major_radius: float = 1.0,
          minor_radius: float = 0.25) -> MeshData:
    nu, nv = max(3, int(major_segments)), max(3, int(minor_segments))
    b = _Builder()

    def p(i, j):
        u = 2 * math.pi * (i % nu) / nu
        v = 2 * math.pi * (j % nv) / nv
        r = major_radius + minor_radius * math.cos(v)
        return (r * math.cos(u), r * math.sin(u), minor_radius * math.sin(v))

    for i in range(nu):
        for j in range(nv):
            u0, u1, v0, v1 = i / nu, (i + 1) / nu, j / nv, (j + 1) / nv
            b.quad(p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1), (u0, v0), (u1, v0), (u1, v1), (u0, v1))
    return b.build("环体", True)


PRIMITIVES = {
    "PLANE": ("平面", plane), "CUBE": ("立方体", cube), "CIRCLE": ("圆", circle), "UV_SPHERE": ("经纬球", uv_sphere),
    "ICO_SPHERE": ("棱角球", ico_sphere), "CYLINDER": ("柱体", cylinder), "CONE": ("锥体", cone),
    "TORUS": ("环体", torus), "GRID": ("栅格", grid),
}
