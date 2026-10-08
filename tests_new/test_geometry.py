"""几何工具测试：体素重构（封闭、并集、空心、开口、翻面、遮罩搬运）、非流形拆分、自动展开 UV（不重叠、不翻面、密度一致）。
纯 CPU，不开窗口。"""
import math
import sys
from pathlib import Path
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from splender.geometry.unwrap import smart_unwrap  # noqa: E402
from splender.sculpt import topology as tp  # noqa: E402
from splender.sculpt.remesh import drop_islands, fill_holes, voxel_remesh  # noqa: E402


def volume(mesh) -> float:
    v = mesh.vertices.astype(np.float64)
    t = mesh.triangles
    return float(np.einsum("ij,ij->i", v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6.0)


def closed(mesh) -> dict:
    stats = tp.mesh_stats(mesh)
    return {"boundary": stats["boundary_edges"], "nonmanifold": stats["nonmanifold_edges"], "euler": stats["euler"]}


def two_spheres(offset=0.9, scale=0.7, level=24):
    a = tp.make_sphere(level)
    b = tp.make_sphere(level)
    b.vertices = b.vertices * scale + np.array([offset, 0, 0], np.float32)
    return tp.SculptMesh(np.concatenate([a.vertices, b.vertices]),
                         np.concatenate([a.triangles, b.triangles + a.vertex_count]))


class RemeshTest(unittest.TestCase):
    def test_sphere_closed_and_volume(self):
        result = voxel_remesh(tp.make_sphere(32), 128)
        self.assertEqual(closed(result), {"boundary": 0, "nonmanifold": 0, "euler": 2})
        self.assertAlmostEqual(volume(result), 4.0 / 3.0 * math.pi, delta=0.03)
        self.assertGreater(volume(result), 0.0, "三角形应朝外")
        self.assertIsNone(result.corner_uvs)

    def test_overlap_becomes_union(self):
        result = voxel_remesh(two_spheres(), 160)
        self.assertEqual(closed(result), {"boundary": 0, "nonmanifold": 0, "euler": 2})
        # 并集体积：大球 + 小球 - 重叠部分，比两者之和小、比大球大
        big = 4.0 / 3.0 * math.pi
        small = big * 0.343
        self.assertGreater(volume(result), big + 0.1)
        self.assertLess(volume(result), big + small - 0.05)

    def test_inverted_and_hollow(self):
        sphere = tp.make_sphere(24)
        inverted = tp.SculptMesh(sphere.vertices.copy(), sphere.triangles[:, [0, 2, 1]].copy())
        result = voxel_remesh(inverted, 96)
        self.assertEqual(closed(result)["euler"], 2)
        self.assertGreater(volume(result), 4.0)
        inner = tp.SculptMesh(sphere.vertices * 0.6, sphere.triangles[:, [0, 2, 1]].copy())
        hollow = tp.SculptMesh(np.concatenate([sphere.vertices, inner.vertices]),
                               np.concatenate([sphere.triangles, inner.triangles + sphere.vertex_count]))
        result = voxel_remesh(hollow, 128)
        self.assertEqual(closed(result), {"boundary": 0, "nonmanifold": 0, "euler": 4})
        self.assertAlmostEqual(volume(result), 4.0 / 3.0 * math.pi * (1 - 0.216), delta=0.05)

    def test_open_mesh_is_closed(self):
        sphere = tp.make_sphere(24)
        keep = sphere.vertices[sphere.triangles].mean(axis=1)[:, 1] < 0.8
        holed = tp.SculptMesh(sphere.vertices.copy(), sphere.triangles[keep].copy())
        filled, holes = fill_holes(holed)
        self.assertEqual(holes, 1)
        self.assertEqual(tp.mesh_stats(filled)["boundary_edges"], 0)
        result = voxel_remesh(holed, 96)
        self.assertEqual(closed(result), {"boundary": 0, "nonmanifold": 0, "euler": 2})

    def test_mask_and_materials_carry_over(self):
        sphere = tp.make_sphere(24)
        mask = (sphere.vertices[:, 1] > 0.5).astype(np.float32)
        mats = (sphere.vertices[sphere.triangles].mean(axis=1)[:, 0] > 0).astype(np.int32)
        mesh = tp.SculptMesh(sphere.vertices, sphere.triangles, None, mats)
        result = voxel_remesh(mesh, 96, mask=mask)
        top = result.vertices[:, 1] > 0.6
        bottom = result.vertices[:, 1] < 0.4
        self.assertGreater(float(result.mask[top].mean()), 0.95)
        self.assertLess(float(result.mask[bottom].mean()), 0.05)
        centers = result.vertices[result.triangles].mean(axis=1)
        right = centers[:, 0] > 0.1
        left = centers[:, 0] < -0.1
        self.assertGreater(float(result.material_ids[right].mean()), 0.95)
        self.assertLess(float(result.material_ids[left].mean()), 0.05)

    def test_islands_dropped(self):
        a = tp.make_sphere(8)
        far = tp.make_sphere(1)
        far.vertices = far.vertices * 0.01 + np.array([5, 0, 0], np.float32)
        verts = np.concatenate([a.vertices, far.vertices])
        tris = np.concatenate([a.triangles, far.triangles + a.vertex_count])
        _v, kept, _old, removed = drop_islands(verts, tris, 64)
        self.assertEqual(removed, 1)
        self.assertEqual(len(kept), a.triangle_count)

    def test_nonmanifold_split(self):
        # 两个四面体只共用一条边：拆开后每条边正好两个三角形
        v = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0.2], [0.5, 0.3, 1], [0.5, -1, 0.2], [0.5, -0.3, -1]],
                     np.float32)
        t1 = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
        t2 = np.array([[0, 1, 4], [0, 5, 1], [1, 5, 4], [0, 4, 5]])
        mesh = tp.SculptMesh(v, np.concatenate([t1, t2]).astype(np.int32))
        self.assertGreater(tp.mesh_stats(mesh)["nonmanifold_edges"], 0)
        fixed, split = tp.split_nonmanifold(mesh)
        self.assertGreater(split, 0)
        stats = tp.mesh_stats(fixed)
        self.assertEqual(stats["nonmanifold_edges"], 0)
        self.assertEqual(stats["boundary_edges"], 0)

    def test_speed(self):
        big = tp.subdivide(tp.make_sphere(128), 1)
        voxel_remesh(tp.make_sphere(8), 32)
        t0 = time.perf_counter()
        result = voxel_remesh(big, 384)
        seconds = time.perf_counter() - t0
        print("78 万三角形重构到 %d 个：%.2f 秒，%s" % (result.triangle_count, seconds, result.stats["steps_ms"]))
        self.assertLess(seconds, 6.0)


class UnwrapTest(unittest.TestCase):
    def check(self, mesh, **kw):
        uv, stats = smart_unwrap(mesh.vertices, mesh.triangles, mesh.material_ids, **kw)
        self.assertEqual(uv.shape, (mesh.triangle_count * 3, 2))
        self.assertTrue((uv >= 0).all() and (uv <= 1).all(), "UV 应在 0-1 之内")
        u = uv.reshape(-1, 3, 2).astype(np.float64)
        signed = (u[:, 1, 0] - u[:, 0, 0]) * (u[:, 2, 1] - u[:, 0, 1]) - (u[:, 2, 0] - u[:, 0, 0]) * (u[:, 1, 1] - u[:, 0, 1])
        self.assertEqual(int((signed < 0).sum()), 0, "不应有翻面的三角形")
        v = mesh.vertices.astype(np.float64)
        t = mesh.triangles
        world = np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
        ratio = np.abs(signed) / np.maximum(world, 1e-30)
        ratio /= np.median(ratio)
        self.assertGreater(np.percentile(ratio, 5), 0.7, "拉伸太大")
        self.assertLess(np.percentile(ratio, 95), 1.3, "拉伸太大")
        # 不重叠：在 512 格上按三角形覆盖的格子数，不同块不应占同一格（允许少量边界误差）
        grid = np.full((512, 512), -1, np.int64)
        overlap = 0
        chart_of = self.chart_ids(uv, mesh)
        centers = (u.mean(axis=1) * 512).astype(np.int64).clip(0, 511)
        for (x, y), c in zip(centers, chart_of):
            if grid[y, x] >= 0 and grid[y, x] != c:
                overlap += 1
            grid[y, x] = c
        self.assertLess(overlap, mesh.triangle_count * 0.002, "块之间不应重叠")
        return uv, stats

    @staticmethod
    def chart_ids(uv, mesh):
        """按 UV 连通性给三角形分块：共用一条边且两端 UV 相同的算同一块。"""
        tris = mesh.triangles
        parent = np.arange(len(tris))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        edge_owner = {}
        u = np.round(uv.reshape(-1, 3, 2) * 1e6).astype(np.int64)
        for f, tri in enumerate(tris):
            for k in range(3):
                a, b = int(tri[k]), int(tri[(k + 1) % 3])
                key = (min(a, b), max(a, b))
                uva = tuple(u[f, k])
                uvb = tuple(u[f, (k + 1) % 3])
                sig = (uva, uvb) if a < b else (uvb, uva)
                other = edge_owner.get(key)
                if other is not None and other[1] == sig:
                    ra, rb = find(f), find(other[0])
                    if ra != rb:
                        parent[ra] = rb
                else:
                    edge_owner[key] = (f, sig)
        return np.array([find(f) for f in range(len(tris))])

    def test_sphere(self):
        mesh = voxel_remesh(tp.make_sphere(16), 48)
        _uv, stats = self.check(mesh)
        self.assertGreater(stats["coverage"], 0.45)
        self.assertLess(stats["charts"], 60)

    def test_materials_separate(self):
        sphere = voxel_remesh(tp.make_sphere(16), 48)
        mats = (sphere.vertices[sphere.triangles].mean(axis=1)[:, 0] > 0).astype(np.int32)
        mesh = tp.SculptMesh(sphere.vertices, sphere.triangles, None, mats)
        uv, stats = smart_unwrap(mesh.vertices, mesh.triangles, mats)
        self.assertEqual(len(stats["texels_per_unit"]), 2)
        # 两个材质各自排满一张图
        for m in (0, 1):
            corners = np.repeat(mats == m, 3)
            self.assertGreater(float(uv[corners].max()), 0.8)

    def test_speed(self):
        mesh = voxel_remesh(tp.subdivide(tp.make_sphere(128), 1), 384)
        smart_unwrap(tp.make_sphere(4).vertices, tp.make_sphere(4).triangles)
        t0 = time.perf_counter()
        _uv, stats = smart_unwrap(mesh.vertices, mesh.triangles)
        seconds = time.perf_counter() - t0
        print("%d 个三角形展开：%.2f 秒，%d 块，覆盖 %.0f%%" % (mesh.triangle_count, seconds, stats["charts"],
                                                       stats["coverage"] * 100))
        self.assertLess(seconds, 4.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
