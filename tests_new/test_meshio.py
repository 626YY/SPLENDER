"""splender.doc.meshio 的测试。

覆盖：OBJ 的各种面写法、负索引、多边形、多材质、缺法线、缺 UV、坏面、切块并行解析、浮点解析；
glTF（外部 .bin、data: URI）与 GLB 的节点变换、多图元、稀疏访问器、三角形带与扇；测试模型的朝向与法线。

运行：python -m unittest tests_new.test_meshio -v
"""
from __future__ import annotations

import base64
import json
import math
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splender.doc import meshio  # noqa: E402
from splender.doc.meshio import DEFAULT_MATERIAL, MeshImportError, load_mesh, make_test_mesh  # noqa: E402


def warn_number(mesh, pattern: str) -> int:
    """在 warnings 里找匹配 pattern 的一条，返回其中的第一个整数；找不到返回 0。"""
    for w in mesh.warnings:
        if re.search(pattern, w):
            m = re.search(r"\d+", w)
            return int(m.group()) if m else -1
    return 0


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_text(self, name: str, text: str, newline: str = "\n", encoding: str = "utf-8",
                   bom: bool = False) -> Path:
        data = text.replace("\n", newline).encode(encoding)
        if bom:
            data = b"\xef\xbb\xbf" + data
        p = self.dir / name
        p.write_bytes(data)
        return p

    def assert_mesh_contract(self, m) -> None:
        """MeshData 的通用约定：形状、类型、C 连续、法线单位长度、包围盒。"""
        T = m.triangle_count
        self.assertEqual(m.positions.shape, (3 * T, 3))
        self.assertEqual(m.normals.shape, (3 * T, 3))
        self.assertEqual(m.uvs.shape, (3 * T, 2))
        self.assertEqual(m.material_ids.shape, (T,))
        for arr, dt in ((m.positions, np.float32), (m.normals, np.float32), (m.uvs, np.float32),
                        (m.material_ids, np.int32), (m.bounds_min, np.float32), (m.bounds_max, np.float32)):
            self.assertEqual(arr.dtype, dt)
            self.assertTrue(arr.flags.c_contiguous)
        self.assertTrue(np.isfinite(m.positions).all())
        self.assertTrue(np.isfinite(m.normals).all())
        self.assertTrue(np.isfinite(m.uvs).all())
        np.testing.assert_allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-5)
        self.assertTrue((m.material_ids >= 0).all() and (m.material_ids < len(m.materials)).all())
        np.testing.assert_array_equal(m.bounds_min, m.positions.min(axis=0))
        np.testing.assert_array_equal(m.bounds_max, m.positions.max(axis=0))


# =====================================================================================
# OBJ
# =====================================================================================

class ObjFormatTest(TempCase):
    """一个文件里覆盖四种面写法、负索引、多边形、多材质、缺 UV、缺法线、各种空白与注释。"""

    V = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)]
    VT = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
    VN = [(0, 0.6, 0.8), (0, 0, 2)]

    TEXT = (
        "# 注释：各种写法\n"
        "mtllib test.mtl\n"
        "\n"
        "o 物体一\n"
        "g 组一\n"
        "s 1\n"
        "v 0 0 0\n"
        "   v 1 0 0\n"
        "\tv 1 1 0\n"
        "v 0 1 0 1.0\n"
        "v 2 0 0 0.5 0.5 0.5\n"
        "v 2.0e0 1E0 -0\n"
        "vt 0 0\n"
        "vt 1 0 0\n"
        "vt 1 1\n"
        "vt 0 1\n"
        "vt .5 +.5\n"
        "vn 0 0.6 0.8\n"
        "vn 0 0 2\n"
        "s off\n"
        "usemtl 红\n"
        "f 1 2 3 # 尾注\n"
        "f 1/1 3/3 4/4\n"
        "usemtl 蓝\n"
        "f 2//1 5//1 6//1\n"
        "f -5/-4/-1 -2/-1/-1 -1/-3/-1\n"
        "usemtl 红\n"
        "f 1/1/1 2/2/1 5/5/1 \\\n"
        "   6/5/1 3/3/1\n"
    )

    # (v 索引, vt 索引或 None, vn 索引或 None, 材质)，都从 0 开始
    TRIS = [
        ((0, 1, 2), None, None, 0),
        ((0, 2, 3), (0, 2, 3), None, 0),
        ((1, 4, 5), None, (0, 0, 0), 1),
        ((1, 4, 5), (1, 4, 2), (1, 1, 1), 1),
        # 五边形 1 2 5 6 3 扇形切出的第一个三角形 (0, 1, 4) 三点共线、面积为零，读取时跳过
        ((0, 4, 5), (0, 4, 4), (0, 0, 0), 0),
        ((0, 5, 2), (0, 4, 2), (0, 0, 0), 0),
    ]

    def expected(self):
        pos, uv, nrm, mat = [], [], [], []
        for v, t, n, m in self.TRIS:
            for c in range(3):
                pos.append(self.V[v[c]])
                uv.append(self.VT[t[c]] if t else (0.0, 0.0))
                if n:
                    vec = np.asarray(self.VN[n[c]], float)
                    nrm.append(vec / np.linalg.norm(vec))
                else:
                    nrm.append((0.0, 0.0, 1.0))      # 全部面都在 z=0 平面且逆时针，平滑法线就是 +Z
            mat.append(m)
        return (np.asarray(pos, np.float32), np.asarray(uv, np.float32), np.asarray(nrm, np.float32),
                np.asarray(mat, np.int32))

    def check(self, m):
        self.assert_mesh_contract(m)
        pos, uv, nrm, mat = self.expected()
        np.testing.assert_array_equal(m.positions, pos)
        np.testing.assert_array_equal(m.uvs, uv)
        np.testing.assert_allclose(m.normals, nrm, atol=1e-6)
        np.testing.assert_array_equal(m.material_ids, mat)
        self.assertEqual(m.materials, ["红", "蓝"])
        self.assertTrue(m.has_uvs)
        self.assertEqual(warn_number(m, "缺少 UV"), 2)
        self.assertEqual(warn_number(m, "法线缺失"), 6)
        self.assertEqual(warn_number(m, "面积为零"), 1)

    def test_lf(self):
        self.check(load_mesh(self.write_text("a.obj", self.TEXT)))

    def test_crlf_bom(self):
        self.check(load_mesh(self.write_text("b.obj", self.TEXT, newline="\r\n", bom=True)))

    def test_cr_only(self):
        # 老式 Mac 换行；续行在这种文件里不成立，所以去掉续行再比
        text = self.TEXT.replace("\\\n   ", "")
        self.check(load_mesh(self.write_text("c.obj", text, newline="\r")))

    def test_name_and_source(self):
        p = self.write_text("模型 名字.obj", self.TEXT)
        m = load_mesh(p)
        self.assertEqual(m.name, "模型 名字")
        self.assertEqual(Path(m.source_path), p)


class ObjNormalsTest(TempCase):
    def test_angle_weighted_cube(self):
        """8 个共享顶点的立方体，每个四边形按扇形拆，角上的三角形数不对称；按夹角加权后法线应对称。"""
        text = "\n".join([
            "v -1 -1 -1", "v 1 -1 -1", "v 1 1 -1", "v -1 1 -1",
            "v -1 -1 1", "v 1 -1 1", "v 1 1 1", "v -1 1 1",
            "f 5 6 7 8", "f 1 4 3 2", "f 2 3 7 6", "f 1 5 8 4", "f 4 8 7 3", "f 1 2 6 5",
        ]) + "\n"
        m = load_mesh(self.write_text("cube.obj", text))
        self.assert_mesh_contract(m)
        self.assertEqual(m.triangle_count, 12)
        expect = m.positions / np.linalg.norm(m.positions, axis=1, keepdims=True)
        np.testing.assert_allclose(m.normals, expect, atol=1e-6)
        self.assertFalse(m.has_uvs)

    def test_weld_by_position(self):
        """两个三角形用重复的顶点（坐标相同、索引不同）相连，平滑法线要按位置焊接。"""
        text = "\n".join([
            "v 0 0 0", "v 1 0 0", "v 0 1 0",
            "v 0 0 0", "v 0 1 0", "v 0 0 -1",
            "vt 0 0", "f 1/1 2/1 3/1", "f 4/1 5/1 6/1",
        ]) + "\n"
        m = load_mesh(self.write_text("fold.obj", text))
        self.assert_mesh_contract(m)
        s = 1.0 / math.sqrt(2.0)
        expect = np.array([
            [-s, 0, s], [0, 0, 1], [-s, 0, s],      # A：(0,0,0) 共享，(1,0,0) 独有，(0,1,0) 共享
            [-s, 0, s], [-s, 0, s], [-1, 0, 0],     # B
        ], np.float32)
        np.testing.assert_allclose(m.normals, expect, atol=1e-6)

    def test_flat_option(self):
        text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 -1\nf 1 2 3\nf 1 3 4\n"
        m = load_mesh(self.write_text("flat.obj", text), missing_normals="flat")
        np.testing.assert_allclose(m.normals[:3], [[0, 0, 1]] * 3, atol=1e-7)
        np.testing.assert_allclose(m.normals[3:], [[-1, 0, 0]] * 3, atol=1e-7)

    def test_invalid_file_normals_recomputed(self):
        text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 0\nvn 0 0 5\nf 1//1 2//2 3//2\n"
        m = load_mesh(self.write_text("zero.obj", text))
        np.testing.assert_allclose(m.normals, [[0, 0, 1]] * 3, atol=1e-7)
        self.assertEqual(warn_number(m, "法线缺失"), 1)


class ObjBadInputTest(TempCase):
    def test_missing_uvs(self):
        m = load_mesh(self.write_text("nouv.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"))
        self.assert_mesh_contract(m)
        self.assertFalse(m.has_uvs)
        self.assertTrue((m.uvs == 0).all())
        self.assertTrue(any("没有 UV" in w for w in m.warnings))

    def test_bad_faces(self):
        text = "\n".join([
            "v 0 0 0", "v 1 0 0", "v 0 1 0", "v 1 1 0",
            "f 1 2 3",          # 好
            "f 1 2 9",          # 越界
            "f 0 1 2",          # 索引 0
            "f 1 2",            # 顶点不足
            "f 1 1 2",          # 面积为零
            "f -9 1 2",         # 负索引越过开头
            "f 1/5 2/1 3/1",    # vt 越界（文件里没有 vt）
            "f a b c",          # 不是数字
            "f 2 4 3",          # 好
            "f 1.5 2 3",        # 不是整数
            "f 1//3 2//1 3//1",  # vn 越界
        ]) + "\n"
        m = load_mesh(self.write_text("bad.obj", text))
        self.assert_mesh_contract(m)
        self.assertEqual(m.triangle_count, 2)
        np.testing.assert_array_equal(m.positions, np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                                               [1, 0, 0], [1, 1, 0], [0, 1, 0]]))
        self.assertEqual(warn_number(m, "不存在的顶点"), 7)
        self.assertEqual(warn_number(m, "不足 3"), 1)
        self.assertEqual(warn_number(m, "面积为零"), 1)

    def test_nan_vertex(self):
        text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv nan 0 0\nv 1 1\nf 1 2 3\nf 1 2 4\nf 2 5 3\n"
        m = load_mesh(self.write_text("nan.obj", text))
        self.assertEqual(m.triangle_count, 1)
        self.assertEqual(warn_number(m, "不是有效数值"), 2)

    def test_no_faces(self):
        with self.assertRaises(MeshImportError):
            load_mesh(self.write_text("empty.obj", "# 空\nv 0 0 0\n"))

    def test_all_faces_bad(self):
        with self.assertRaises(MeshImportError) as cm:
            load_mesh(self.write_text("allbad.obj", "v 0 0 0\nv 1 0 0\nf 1 2 2\nf 1 2 7\n"))
        self.assertIn("没有可用的三角形", str(cm.exception))

    def test_missing_file_and_format(self):
        with self.assertRaises(MeshImportError):
            load_mesh(self.dir / "不存在.obj")
        p = self.write_text("x.fbx", "abc")
        with self.assertRaises(MeshImportError):
            load_mesh(p)


class ObjMaterialTest(TempCase):
    def test_default_material(self):
        m = load_mesh(self.write_text("d.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"))
        self.assertEqual(m.materials, [DEFAULT_MATERIAL])

    def test_faces_before_usemtl(self):
        text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nusemtl 皮肤 材质\nf 1 2 3\nusemtl\nf 1 2 3\n"
        m = load_mesh(self.write_text("e.obj", text))
        self.assertEqual(m.materials, [DEFAULT_MATERIAL, "皮肤 材质"])
        np.testing.assert_array_equal(m.material_ids, [0, 1, 0])

    def test_unused_material_dropped(self):
        text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl 空\nusemtl 实\nf 1 2 3\nusemtl 坏\nf 1 1 1\n"
        m = load_mesh(self.write_text("f.obj", text))
        self.assertEqual(m.materials, ["实"])

    def test_gbk_names(self):
        p = self.dir / "gbk.obj"
        p.write_bytes("v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl 金属\nf 1 2 3\n".encode("gbk"))
        self.assertEqual(load_mesh(p).materials, ["金属"])


class FloatParseTest(unittest.TestCase):
    def test_against_python(self):
        rng = np.random.default_rng(7)
        tokens = ["0", "-0", "1", "-1", ".5", "-.5", "+.25", "5.", "1e3", "1E-3", "-2.5e+2", "1e-30", "3.4e38",
                  "1e-45", "123456789012345678901234567890", "0.000000000000000000000000123456789",
                  "3.14159265358979323846264338327950288", "1.17549435e-38", "007", "-0.0e0"]
        vals = rng.standard_normal(3000) * 10.0 ** rng.integers(-12, 12, 3000)
        for i, x in enumerate(vals):
            fmt = ("%.6f", "%.9e", "%g", "%.17g", "%+.3f", "%.12E")[i % 6]
            tokens.append(fmt % x)
        for tok in tokens:
            arr = np.frombuffer(tok.encode(), np.uint8).copy()
            val, end, ok = meshio._parse_float(arr, 0, arr.shape[0])
            self.assertTrue(ok, tok)
            self.assertEqual(end, len(tok), tok)
            want = np.float32(float(tok))
            got = np.float32(val)
            if want != got:
                # 允许极少数落在 float32 舍入边界上的数差 1 ulp
                self.assertEqual(np.nextafter(want, got), got, tok)

    def test_not_a_number(self):
        for tok in ("abc", "-", "+", ".", "e5"):
            arr = np.frombuffer(tok.encode(), np.uint8).copy()
            val, end, ok = meshio._parse_float(arr, 0, arr.shape[0])
            self.assertFalse(ok, tok)


def _python_reference_obj(text: str):
    """很慢但直白的 OBJ 参考解析（只认本测试生成的写法）。返回 (坐标, uv, 法线, 材质名列表)。"""
    V, VT, VN = [], [], []
    pos, uv, nrm, mats = [], [], [], []
    cur = None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] == "v":
            V.append(tuple(float(x) for x in parts[1:4]))
        elif parts[0] == "vt":
            VT.append((float(parts[1]), float(parts[2])))
        elif parts[0] == "vn":
            VN.append(tuple(float(x) for x in parts[1:4]))
        elif parts[0] == "usemtl":
            cur = line[len("usemtl"):].strip()
        elif parts[0] == "f":
            corners = []
            for tok in parts[1:]:
                a, b, c = tok.split("/")

                def res(s, n):
                    i = int(s)
                    return i - 1 if i > 0 else n + i
                corners.append((res(a, len(V)), res(b, len(VT)), res(c, len(VN))))
            for j in range(1, len(corners) - 1):
                for cidx in (corners[0], corners[j], corners[j + 1]):
                    pos.append(V[cidx[0]])
                    uv.append(VT[cidx[1]])
                    nrm.append(VN[cidx[2]])
                mats.append(cur)
    return np.asarray(pos), np.asarray(uv), np.asarray(nrm), mats


class ObjChunkedParseTest(TempCase):
    def make_random_obj(self, seed: int = 3) -> str:
        rng = np.random.default_rng(seed)
        lines = ["# 随机测试文件"]
        nv = nvt = nvn = 0
        names = ["石头", "木头", "metal", "布 料", "皮"]
        for block in range(60):
            for _ in range(int(rng.integers(20, 80))):
                p = rng.uniform(-5, 5, 3)
                lines.append("v %.6f %.6f %.6f" % tuple(p))
                nv += 1
            for _ in range(int(rng.integers(20, 80))):
                lines.append("vt %.6f %.6f" % tuple(rng.uniform(0, 1, 2)))
                nvt += 1
            for _ in range(int(rng.integers(5, 30))):
                n = rng.standard_normal(3)
                n /= np.linalg.norm(n)
                lines.append("vn %.6f %.6f %.6f" % tuple(n))
                nvn += 1
            if rng.random() < 0.5:
                lines.append("usemtl " + names[int(rng.integers(0, len(names)))])
            if rng.random() < 0.2:
                lines.append("")
                lines.append("# 块 %d" % block)
            for _ in range(int(rng.integers(100, 400))):
                k = int(rng.integers(3, 7))
                vs = rng.choice(nv, size=k, replace=False)
                toks = []
                for v in vs:
                    t = int(rng.integers(0, nvt))
                    q = int(rng.integers(0, nvn))
                    sv = str(v + 1) if rng.random() < 0.5 else str(v - nv)
                    st = str(t + 1) if rng.random() < 0.5 else str(t - nvt)
                    sn = str(q + 1) if rng.random() < 0.5 else str(q - nvn)
                    toks.append(f"{sv}/{st}/{sn}")
                lines.append("f " + " ".join(toks))
        return "\n".join(lines) + "\n"

    def test_chunked_equals_reference(self):
        text = self.make_random_obj()
        p = self.write_text("rand.obj", text, newline="\r\n")
        whole = meshio._load_obj(p, chunk_bytes=1 << 30, threads=1)
        parts = meshio._load_obj(p, chunk_bytes=4096, threads=4)
        self.assert_mesh_contract(whole)
        for name in ("positions", "normals", "uvs", "material_ids"):
            np.testing.assert_array_equal(getattr(whole, name), getattr(parts, name), err_msg=name)
        self.assertEqual(whole.materials, parts.materials)
        pos, uv, nrm, mats = _python_reference_obj(text)
        self.assertEqual(whole.triangle_count, len(mats))
        np.testing.assert_allclose(whole.positions, pos, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(whole.uvs, uv, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(whole.normals, nrm / np.linalg.norm(nrm, axis=1, keepdims=True), atol=2e-6)
        names = [DEFAULT_MATERIAL if n is None else n for n in mats]
        self.assertEqual([whole.materials[i] for i in whole.material_ids], names)


# =====================================================================================
# glTF / GLB
# =====================================================================================

def _rot_y90() -> np.ndarray:
    return np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def _mat4(m3=None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    out = np.eye(4)
    if m3 is not None:
        out[:3, :3] = m3
    out[:3, 3] = t
    return out


class GltfBuilder:
    """按 glTF 2.0 规范拼一个测试场景：两个图元（交错缓冲 + 稀疏访问器 + 归一化 UV），三个节点（含镜像）。"""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, data: bytes, stride: int | None = None) -> int:
        self.buf.extend(b"\0" * ((-len(self.buf)) % 4))
        v = {"buffer": 0, "byteOffset": len(self.buf), "byteLength": len(data)}
        if stride:
            v["byteStride"] = stride
        self.buf.extend(data)
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, view: int, offset: int, ctype: int, count: int, kind: str, **extra) -> int:
        a = {"bufferView": view, "byteOffset": offset, "componentType": ctype, "count": count, "type": kind}
        a.update(extra)
        self.accessors.append(a)
        return len(self.accessors) - 1

    def scene(self):
        # 图元 A：位置、法线、UV 交错在一个缓冲视图里，步长 32；索引 uint16
        inter = np.zeros(4, dtype=[("p", "<f4", 3), ("n", "<f4", 3), ("t", "<f4", 2)])
        self.pa = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
        self.na = np.array([[0, 0.6, 0.8]] * 4, np.float32)
        self.ua = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)       # glTF 的 v 向下
        inter["p"], inter["n"], inter["t"] = self.pa, self.na, self.ua
        va = self.view(inter.tobytes(), stride=32)
        a_pos = self.accessor(va, 0, 5126, 4, "VEC3", min=[0, 0, 0], max=[1, 1, 0])
        a_nrm = self.accessor(va, 12, 5126, 4, "VEC3")
        a_uv = self.accessor(va, 24, 5126, 4, "VEC2")
        a_ia = self.accessor(self.view(np.array([0, 1, 2, 0, 2, 3], np.uint16).tobytes()), 0, 5123, 6, "SCALAR")
        # 图元 B：位置紧密排列，第 2 个顶点由稀疏访问器替换；UV 是归一化 uint16；没有法线；索引 uint8
        pb = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], np.float32)
        self.pb = pb.copy()
        self.pb[2] = (0, 2, 1)
        vpb = self.view(pb.tobytes())
        vsi = self.view(np.array([2], np.uint8).tobytes())
        vsv = self.view(np.array([[0, 2, 1]], np.float32).tobytes())
        a_pb = self.accessor(vpb, 0, 5126, 3, "VEC3", sparse={
            "count": 1, "indices": {"bufferView": vsi, "componentType": 5121}, "values": {"bufferView": vsv}})
        ub = np.array([[0, 65535], [65535, 65535], [0, 0]], np.uint16)
        self.ub = ub.astype(np.float64) / 65535.0
        a_ub = self.accessor(self.view(ub.tobytes()), 0, 5123, 3, "VEC2", normalized=True)
        a_ib = self.accessor(self.view(np.array([0, 1, 2], np.uint8).tobytes()), 0, 5121, 3, "SCALAR")

        s = math.sin(math.pi / 4)
        mirror = _mat4(np.diag([-1.0, 1.0, 1.0]), (0.0, 1.0, 0.0))
        doc = {
            "asset": {"version": "2.0", "generator": "SPLENDER 测试"},
            "scene": 0,
            "scenes": [{"nodes": [0, 2]}],
            "nodes": [
                {"name": "父", "translation": [1, 2, 3], "rotation": [0, s, 0, s], "scale": [2, 2, 2],
                 "children": [1]},
                {"name": "镜像子", "matrix": mirror.T.reshape(-1).tolist(), "mesh": 0},
                {"name": "原样", "mesh": 0},
            ],
            "meshes": [{"name": "网格", "primitives": [
                {"attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv}, "indices": a_ia,
                 "material": 0},
                {"attributes": {"POSITION": a_pb, "TEXCOORD_0": a_ub}, "indices": a_ib, "material": 1},
            ]}],
            "materials": [{"name": "金属"}, {}],
            "buffers": [{"byteLength": len(self.buf)}],
            "bufferViews": self.views,
            "accessors": self.accessors,
        }
        parent = _mat4(_rot_y90() * 2.0, (1.0, 2.0, 3.0))
        self.instances = [parent @ mirror, np.eye(4)]
        return doc, bytes(self.buf)

    def expected(self):
        pos, nrm, uv, mat = [], [], [], []
        for W in self.instances:
            m3 = W[:3, :3]
            flip = np.linalg.det(m3) < 0
            # 图元 A
            tris = np.array([[0, 1, 2], [0, 2, 3]])
            if flip:
                tris = tris[:, [0, 2, 1]]
            P = self.pa.astype(float) @ m3.T + W[:3, 3]
            Nm = self.na.astype(float) @ np.linalg.inv(m3)
            Nm /= np.linalg.norm(Nm, axis=1, keepdims=True)
            U = np.stack([self.ua[:, 0], 1.0 - self.ua[:, 1]], 1)
            for t in tris:
                pos.extend(P[t])
                nrm.extend(Nm[t])
                uv.extend(U[t])
                mat.append(0)
            # 图元 B：平直法线
            t = np.array([0, 2, 1]) if flip else np.array([0, 1, 2])
            P = self.pb.astype(float) @ m3.T + W[:3, 3]
            fn = np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])
            fn /= np.linalg.norm(fn)
            U = np.stack([self.ub[:, 0], 1.0 - self.ub[:, 1]], 1)
            pos.extend(P[t])
            nrm.extend([fn] * 3)
            uv.extend(U[t])
            mat.append(1)
        return np.asarray(pos), np.asarray(nrm), np.asarray(uv), np.asarray(mat)


def write_glb(path: Path, doc: dict, bin_bytes: bytes) -> None:
    js = json.dumps(doc, ensure_ascii=False).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    bb = bin_bytes + b"\0" * ((-len(bin_bytes)) % 4)
    total = 12 + 8 + len(js) + 8 + len(bb)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
                     + struct.pack("<II", len(bb), 0x004E4942) + bb)


class GltfTest(TempCase):
    def setUp(self) -> None:
        super().setUp()
        self.builder = GltfBuilder()
        self.doc, self.bin = self.builder.scene()

    def check(self, m):
        self.assert_mesh_contract(m)
        pos, nrm, uv, mat = self.builder.expected()
        self.assertEqual(m.triangle_count, 6)
        np.testing.assert_allclose(m.positions, pos, atol=1e-6)
        np.testing.assert_allclose(m.normals, nrm, atol=1e-6)
        np.testing.assert_allclose(m.uvs, uv, atol=1e-7)
        np.testing.assert_array_equal(m.material_ids, mat)
        self.assertEqual(m.materials, ["金属", "材质1"])
        self.assertTrue(m.has_uvs)
        # 镜像节点的三角形保持逆时针朝外：几何法线与给定法线同向
        P = m.positions.reshape(-1, 3, 3).astype(float)
        gn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        self.assertTrue((np.einsum("ij,ij->i", gn, m.normals.reshape(-1, 3, 3)[:, 0]) > 0).all())

    def test_gltf_external_bin(self):
        (self.dir / "mesh data.bin").write_bytes(self.bin)
        doc = json.loads(json.dumps(self.doc))
        doc["buffers"][0]["uri"] = "mesh%20data.bin"
        p = self.dir / "scene.gltf"
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        self.check(load_mesh(p))

    def test_glb(self):
        p = self.dir / "scene.glb"
        write_glb(p, self.doc, self.bin)
        self.check(load_mesh(p))

    def test_data_uri(self):
        doc = json.loads(json.dumps(self.doc))
        doc["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(self.bin).decode()
        p = self.dir / "embedded.gltf"
        p.write_text(json.dumps(doc), encoding="utf-8")
        self.check(load_mesh(p))

    def test_missing_bin(self):
        doc = json.loads(json.dumps(self.doc))
        doc["buffers"][0]["uri"] = "nothere.bin"
        p = self.dir / "missing.gltf"
        p.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaises(MeshImportError) as cm:
            load_mesh(p)
        self.assertIn("nothere.bin", str(cm.exception))

    def test_draco_rejected(self):
        doc = json.loads(json.dumps(self.doc))
        doc["extensionsRequired"] = ["KHR_draco_mesh_compression"]
        doc["extensionsUsed"] = ["KHR_draco_mesh_compression"]
        p = self.dir / "draco.glb"
        write_glb(p, doc, self.bin)
        with self.assertRaises(MeshImportError) as cm:
            load_mesh(p)
        self.assertIn("Draco", str(cm.exception))

    def test_accessor_out_of_range(self):
        doc = json.loads(json.dumps(self.doc))
        doc["accessors"][0]["count"] = 1000
        p = self.dir / "broken.glb"
        write_glb(p, doc, self.bin)
        with self.assertRaises(MeshImportError) as cm:
            load_mesh(p)
        self.assertIn("访问器", str(cm.exception))

    def test_version_1_rejected(self):
        p = self.dir / "old.gltf"
        p.write_text(json.dumps({"asset": {"version": "1.0"}}), encoding="utf-8")
        with self.assertRaises(MeshImportError):
            load_mesh(p)

    def test_no_nodes(self):
        doc = json.loads(json.dumps(self.doc))
        for key in ("nodes", "scenes", "scene"):
            doc.pop(key)
        p = self.dir / "nonodes.glb"
        write_glb(p, doc, self.bin)
        m = load_mesh(p)
        self.assertEqual(m.triangle_count, 3)
        self.assertTrue(any("没有节点" in w for w in m.warnings))

    def test_smooth_option(self):
        p = self.dir / "smooth.glb"
        write_glb(p, self.doc, self.bin)
        m = load_mesh(p, missing_normals="smooth")
        self.assert_mesh_contract(m)
        # 图元 A 的法线来自文件，不受影响
        pos, nrm, uv, mat = self.builder.expected()
        np.testing.assert_allclose(m.normals[:6], nrm[:6], atol=1e-6)


class GltfModesTest(TempCase):
    def test_strip_fan_and_bad_index(self):
        b = GltfBuilder()
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 2, 0]], np.float32)
        a_pos = b.accessor(b.view(pts.tobytes()), 0, 5126, 5, "VEC3")
        strip = b.accessor(b.view(np.array([0, 1, 2, 3, 4], np.uint32).tobytes()), 0, 5125, 5, "SCALAR")
        fan = b.accessor(b.view(np.array([0, 1, 3, 4], np.uint16).tobytes()), 0, 5123, 4, "SCALAR")
        bad = b.accessor(b.view(np.array([0, 1, 9, 0, 1, 2], np.uint16).tobytes()), 0, 5123, 6, "SCALAR")
        doc = {"asset": {"version": "2.0"},
               "meshes": [{"primitives": [
                   {"attributes": {"POSITION": a_pos}, "indices": strip, "mode": 5},
                   {"attributes": {"POSITION": a_pos}, "indices": fan, "mode": 6},
                   {"attributes": {"POSITION": a_pos}, "indices": bad},
                   {"attributes": {"POSITION": a_pos}, "mode": 1},
               ]}],
               "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}],
               "buffers": [{"byteLength": len(b.buf)}], "bufferViews": b.views, "accessors": b.accessors}
        p = self.dir / "modes.glb"
        write_glb(p, doc, bytes(b.buf))
        m = load_mesh(p)
        # 带：(0,1,2) (1,3,2) (2,3,4)；扇：(1,3,0) (3,4,0)；坏索引那组只剩 (0,1,2)
        expect_idx = [(0, 1, 2), (1, 3, 2), (2, 3, 4), (1, 3, 0), (3, 4, 0), (0, 1, 2)]
        np.testing.assert_array_equal(m.positions, pts[np.array(expect_idx).reshape(-1)])
        self.assertFalse(m.has_uvs)
        self.assertEqual(warn_number(m, "索引越界"), 1)
        self.assertEqual(warn_number(m, "点或线"), 1)
        # 三角形带的奇数个三角形按规范换了顺序，所以全部朝 +Z；缺法线时按平直法线补
        P = m.positions.reshape(-1, 3, 3).astype(float)
        gn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        self.assertTrue((gn[:, 2] > 0).all())
        np.testing.assert_allclose(m.normals, np.tile([0.0, 0.0, 1.0], (18, 1)), atol=1e-7)


# =====================================================================================
# 测试模型
# =====================================================================================

class TestMeshesTest(TempCase):
    def check_common(self, m):
        self.assert_mesh_contract(m)
        self.assertTrue((m.uvs >= 0).all() and (m.uvs <= 1).all())
        P = m.positions.reshape(-1, 3, 3).astype(float)
        gn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        self.assertTrue((np.linalg.norm(gn, axis=1) > 0).all())
        # 顶点法线与几何法线同向（三角形逆时针 = 正面）
        N = m.normals.reshape(-1, 3, 3)
        for c in range(3):
            self.assertTrue((np.einsum("ij,ij->i", gn, N[:, c]) > 0).all())
        # UV 逆时针与三维逆时针一致（贴图不是镜像的）
        U = m.uvs.reshape(-1, 3, 2).astype(float)
        uv_area = (U[:, 1, 0] - U[:, 0, 0]) * (U[:, 2, 1] - U[:, 0, 1]) - (U[:, 1, 1] - U[:, 0, 1]) * (U[:, 2, 0] - U[:, 0, 0])
        self.assertTrue((uv_area > 0).all())
        return P, gn

    def test_sphere(self):
        m = make_test_mesh("sphere", segments=24, rings=12, radius=2.5)
        self.assertEqual(m.triangle_count, 2 * 24 * 11)
        P, gn = self.check_common(m)
        np.testing.assert_allclose(np.linalg.norm(m.positions, axis=1), 2.5, rtol=1e-6)
        self.assertTrue((np.einsum("ij,ij->i", m.normals, m.positions) > 0).all())
        self.assertTrue((np.einsum("ij,ij->i", gn, P.mean(axis=1)) > 0).all())

    def test_dense_sphere(self):
        m = make_test_mesh("dense_sphere", triangles=20000)
        self.assertGreaterEqual(m.triangle_count, 20000)
        self.assertLess(m.triangle_count, 21000)
        self.check_common(m)

    def test_cube(self):
        m = make_test_mesh("cube", size=3.0, margin=0.03, divisions=2)
        self.assertEqual(m.triangle_count, 6 * 2 * 4)
        P, gn = self.check_common(m)
        self.assertTrue((np.einsum("ij,ij->i", gn, P.mean(axis=1)) > 0).all())
        np.testing.assert_allclose(np.abs(m.positions).max(axis=1), 1.5, rtol=1e-6)
        # 六块 UV 各自在自己的正方形里，正方形之间至少隔 2·margin
        U = m.uvs.reshape(6, -1, 2)
        rects = [meshio.cube_face_uv_rect(f, 0.03) for f in range(6)]
        for f in range(6):
            u0, v0, u1, v1 = rects[f]
            self.assertTrue((U[f, :, 0] >= u0 - 1e-6).all() and (U[f, :, 0] <= u1 + 1e-6).all())
            self.assertTrue((U[f, :, 1] >= v0 - 1e-6).all() and (U[f, :, 1] <= v1 + 1e-6).all())
            for g in range(f + 1, 6):
                a, b = rects[f], rects[g]
                gap = max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])
                self.assertGreaterEqual(gap, 2 * 0.03 - 1e-9)

    def test_plane(self):
        m = make_test_mesh("plane", divisions=3)
        self.assertEqual(m.triangle_count, 18)
        self.check_common(m)
        np.testing.assert_allclose(m.normals, np.tile([0, 1, 0], (54, 1)), atol=1e-7)
        self.assertAlmostEqual(float(m.uvs.min()), 0.0)
        self.assertAlmostEqual(float(m.uvs.max()), 1.0)

    def test_torus(self):
        m = make_test_mesh("torus", major_radius=2.0, minor_radius=0.5, segments=32, sides=16)
        self.assertEqual(m.triangle_count, 2 * 32 * 16)
        P, gn = self.check_common(m)
        # 管心：表面点在大圆上的投影
        flat = m.positions.astype(float).copy()
        flat[:, 1] = 0.0
        center = 2.0 * flat / np.linalg.norm(flat, axis=1, keepdims=True)
        out = m.positions - center
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 0.5, rtol=1e-5)
        self.assertTrue((np.einsum("ij,ij->i", m.normals, out) > 0).all())
        cflat = P.mean(axis=1).copy()
        cflat[:, 1] = 0.0
        ccenter = 2.0 * cflat / np.linalg.norm(cflat, axis=1, keepdims=True)
        self.assertTrue((np.einsum("ij,ij->i", gn, P.mean(axis=1) - ccenter) > 0).all())

    def test_unknown(self):
        with self.assertRaises(ValueError):
            make_test_mesh("teapot")
        with self.assertRaises(TypeError):
            make_test_mesh("cube", radius=2)


if __name__ == "__main__":
    unittest.main()
