"""导出模型：OBJ（四边形、按材质分组、MTL 引用贴图）和 glTF 二进制（结构、UV 翻转、贴图打包），导出后再读回来对得上；
100 万三角形的导出时间。不用显卡。"""
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from splender.doc.meshexport import export_glb, export_obj  # noqa: E402
from splender.doc.meshio import load_mesh, make_test_mesh  # noqa: E402


def png_bytes(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, "PNG")
    return buffer.getvalue()


def read_glb(path: str) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67 and version == 2 and total == len(data)
    json_len, json_type = struct.unpack_from("<II", data, 12)
    assert json_type == 0x4E4F534A
    document = json.loads(data[20:20 + json_len].decode("utf-8"))
    bin_len, bin_type = struct.unpack_from("<II", data, 20 + json_len)
    assert bin_type == 0x004E4942
    return document, data[28 + json_len:28 + json_len + bin_len]


def accessor(document: dict, binary: bytes, index: int) -> np.ndarray:
    acc = document["accessors"][index]
    view = document["bufferViews"][acc["bufferView"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
    dtype = {5126: np.float32, 5125: np.uint32}[acc["componentType"]]
    raw = np.frombuffer(binary, dtype, acc["count"] * width, view["byteOffset"])
    return raw.reshape(-1, width) if width > 1 else raw


class MeshExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="splender_meshexport_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def two_material_cube(self):
        mesh = make_test_mesh("cube")
        tris = mesh.triangle_count
        mesh.material_ids = (np.arange(tris) >= tris // 2).astype(np.int32)   # 前一半一个材质，后一半另一个
        mesh.materials = ["甲", "乙"]
        mesh.polygon_sizes = np.full(tris // 2, 4, np.int32)                  # 立方体两个三角形一个四边形
        return mesh

    def test_obj_quads_materials_roundtrip(self):
        mesh = self.two_material_cube()
        path = os.path.join(self.tmp, "方块.obj")
        info = export_obj(path, [("方块", mesh, ["铁皮", "木头"])],
                          textures={"铁皮": {"basecolor": "铁皮_basecolor.png", "normal": "铁皮_normal.png",
                                            "roughness": "铁皮_roughness.png", "metallic": "铁皮_metallic.png"}})
        self.assertEqual(info["triangles"], mesh.triangle_count)
        self.assertEqual(info["quads"], mesh.triangle_count // 2)
        text = Path(path).read_text(encoding="utf-8")
        faces = [line for line in text.splitlines() if line.startswith("f ")]
        self.assertEqual(len(faces), mesh.triangle_count // 2)
        self.assertTrue(all(len(line.split()) == 5 for line in faces), "四边形写成 4 个角")
        self.assertEqual(text.count("usemtl "), 2)
        mtl = Path(self.tmp, "方块.mtl").read_text(encoding="utf-8")
        for line in ("newmtl 铁皮", "newmtl 木头", "map_Kd 铁皮_basecolor.png", "map_Pr 铁皮_roughness.png",
                     "map_Pm 铁皮_metallic.png", "norm 铁皮_normal.png"):
            self.assertIn(line, mtl)
        back = load_mesh(path)
        self.assertEqual(back.triangle_count, mesh.triangle_count)
        self.assertTrue(np.allclose(np.sort(back.positions.reshape(-1)), np.sort(mesh.positions.reshape(-1)), atol=1e-5))
        self.assertTrue(np.allclose(np.sort(back.uvs.reshape(-1)), np.sort(mesh.uvs.reshape(-1)), atol=1e-5))

    def test_glb_structure_uv_textures_roundtrip(self):
        mesh = make_test_mesh("sphere", segments=24, rings=12)
        path = os.path.join(self.tmp, "球.glb")
        materials = {"漆面": {"basecolor": png_bytes((200, 30, 20)), "orm": png_bytes((255, 128, 0)),
                              "normal": png_bytes((128, 128, 255)), "color": (0.5, 0.5, 0.5)}}
        info = export_glb(path, [("球", mesh, ["漆面"])], materials=materials)
        self.assertEqual(info["images"], 3)
        document, binary = read_glb(path)
        material = document["materials"][0]
        pbr = material["pbrMetallicRoughness"]
        self.assertEqual(material["name"], "漆面")
        self.assertIn("baseColorTexture", pbr)
        self.assertIn("metallicRoughnessTexture", pbr)
        self.assertEqual(material["occlusionTexture"]["index"], pbr["metallicRoughnessTexture"]["index"])
        self.assertIn("normalTexture", material)
        for image in document["images"]:
            view = document["bufferViews"][image["bufferView"]]
            Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])).verify()
        prim = document["meshes"][0]["primitives"][0]
        positions = accessor(document, binary, prim["attributes"]["POSITION"])
        uvs = accessor(document, binary, prim["attributes"]["TEXCOORD_0"])
        index = accessor(document, binary, prim["indices"])
        self.assertEqual(len(index), mesh.triangle_count * 3)
        self.assertLess(int(index.max()), len(positions))
        # 逐角对得上：位置一样，UV 的 V 翻过来（glTF 原点在左上）
        self.assertTrue(np.allclose(positions[index], mesh.positions, atol=1e-6))
        self.assertTrue(np.allclose(uvs[index][:, 0], mesh.uvs[:, 0], atol=1e-6))
        self.assertTrue(np.allclose(1.0 - uvs[index][:, 1], mesh.uvs[:, 1], atol=1e-6))
        back = load_mesh(path)
        self.assertEqual(back.triangle_count, mesh.triangle_count)
        self.assertTrue(np.allclose(back.uvs, mesh.uvs, atol=1e-5), "读回来 UV 一样")

    def test_million_triangles_speed(self):
        mesh = make_test_mesh("dense_sphere", triangles=1_000_000)
        self.assertGreaterEqual(mesh.triangle_count, 1_000_000)
        export_glb(os.path.join(self.tmp, "热身.glb"), [("热身", make_test_mesh("sphere"), ["材质"])])
        export_obj(os.path.join(self.tmp, "热身.obj"), [("热身", make_test_mesh("sphere"), ["材质"])])
        started = time.perf_counter()
        glb = export_glb(os.path.join(self.tmp, "大.glb"), [("大", mesh, ["材质"])])
        glb_seconds = time.perf_counter() - started
        started = time.perf_counter()
        obj = export_obj(os.path.join(self.tmp, "大.obj"), [("大", mesh, ["材质"])])
        obj_seconds = time.perf_counter() - started
        print("\n%s 个三角形：glb %.2f 秒（%.0f MB），obj %.2f 秒" % (format(mesh.triangle_count, ","), glb_seconds,
                                                          glb["bytes"] / 2 ** 20, obj_seconds))
        self.assertLess(glb_seconds, 5.0)
        self.assertLess(obj_seconds, 5.0)
        self.assertEqual(obj["triangles"], mesh.triangle_count)


if __name__ == "__main__":
    unittest.main()
