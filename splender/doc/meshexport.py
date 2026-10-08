"""导出模型：OBJ（可带四边形、按材质分组、MTL 引用贴图）和 glTF 二进制（.glb，PBR 材质和贴图打包在里面）。

模型按 MeshData 逐角点存放（三角形 k 的三个角是 3k..3k+2）；导出时把位置、UV、法线各自去重。
MeshData.polygon_sizes 给了的话（每个多边形 3 或 4 个角，四边形占连续两个三角形 (a,b,c)(a,c,d)），OBJ 写成四边形。
数字转文字用 numba 直接写字节。
"""
from __future__ import annotations

import io
import json
import logging
import os
import struct
import time

import numpy as np
from numba import njit

log = logging.getLogger("splender.export")


# ====================================================================== 去重
def unique_rows(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按字节完全相同去重：返回 (唯一行, 每行在唯一行里的编号)。保持第一次出现的先后。"""
    a = np.ascontiguousarray(array)
    if len(a) == 0:
        return a, np.zeros(0, np.int64)
    view = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))[:, 0]
    _keys, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")             # 按第一次出现排
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return a[first[order]], rank[inverse.reshape(-1)]


# ====================================================================== 文字（numba 写字节）
@njit(cache=True, nogil=True)
def _put_int(buf, at, value):
    if value < 0:
        buf[at] = 45
        at += 1
        value = -value
    if value == 0:
        buf[at] = 48
        return at + 1
    digits = 0
    tmp = value
    while tmp > 0:
        digits += 1
        tmp //= 10
    for k in range(digits - 1, -1, -1):
        buf[at + k] = 48 + value % 10
        value //= 10
    return at + digits


@njit(cache=True, nogil=True)
def _put_float(buf, at, value):
    if value != value:
        value = 0.0
    if value < 0.0:
        buf[at] = 45
        at += 1
        value = -value
    whole = np.int64(value)
    frac = np.int64((value - whole) * 1000000.0 + 0.5)
    if frac >= 1000000:
        whole += 1
        frac -= 1000000
    at = _put_int(buf, at, whole)
    buf[at] = 46
    at += 1
    for k in range(5, -1, -1):
        buf[at + k] = 48 + frac % 10
        frac //= 10
    return at + 6


@njit(cache=True, nogil=True)
def _rows_text(prefix, values):
    """每行：前缀 + 空格分隔的小数 + 换行。"""
    rows, cols = values.shape
    buf = np.empty(rows * (len(prefix) + cols * 24 + 2), np.uint8)
    at = 0
    for r in range(rows):
        for k in range(len(prefix)):
            buf[at] = prefix[k]
            at += 1
        for c in range(cols):
            buf[at] = 32
            at += 1
            at = _put_float(buf, at, values[r, c])
        buf[at] = 10
        at += 1
    return buf[:at]


@njit(cache=True, nogil=True)
def _faces_text(pos_idx, uv_idx, nrm_idx, sizes, has_uv, has_nrm):
    """f 行：sizes[i] 个角（3 或 4），角的编号已经是从 1 开始的。四边形取两个三角形 (a,b,c)(a,c,d) 的 a b c d。"""
    total = 0
    for s in sizes:
        total += s
    buf = np.empty(len(sizes) * 4 + total * 36, np.uint8)
    at = 0
    corner = 0
    for i in range(len(sizes)):
        buf[at] = 102
        at += 1
        if sizes[i] == 4:
            picks = (corner, corner + 1, corner + 2, corner + 5)
            step = 6
        else:
            picks = (corner, corner + 1, corner + 2, -1)
            step = 3
        for j in range(4):
            c = picks[j]
            if c < 0:
                break
            buf[at] = 32
            at += 1
            at = _put_int(buf, at, pos_idx[c])
            if has_uv or has_nrm:
                buf[at] = 47
                at += 1
                if has_uv:
                    at = _put_int(buf, at, uv_idx[c])
                if has_nrm:
                    buf[at] = 47
                    at += 1
                    at = _put_int(buf, at, nrm_idx[c])
        buf[at] = 10
        at += 1
        corner += step
    return buf[:at]


def polygon_sizes(mesh, count: int | None = None) -> np.ndarray:
    """每个多边形几个角；没有四边形信息时全是三角形。"""
    sizes = getattr(mesh, "polygon_sizes", None)
    tris = int(count if count is not None else mesh.triangle_count)
    if sizes is None:
        return np.full(tris, 3, np.int64)
    sizes = np.asarray(sizes, np.int64)
    if int(np.where(sizes == 4, 2, 1).sum()) != tris:
        return np.full(tris, 3, np.int64)
    return sizes


# ====================================================================== OBJ
def export_obj(path: str, objects: list, *, textures: dict | None = None, write_mtl: bool = True,
               scale: float = 1.0) -> dict:
    """objects：[(名字, MeshData, 材质名列表)]。textures：{材质名: {"basecolor": 文件名, "normal": …, …}}，写进 MTL。"""
    started = time.perf_counter()
    path = str(path)
    folder = os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    mtl_name = stem + ".mtl"
    out = io.BytesIO()
    out.write(("# SPLENDER\n" + ("mtllib %s\n" % mtl_name if write_mtl else "")).encode("utf-8"))
    base_v = base_vt = base_vn = 0
    totals = {"triangles": 0, "quads": 0, "vertices": 0}
    used_materials: list[str] = []
    for name, mesh, material_names in objects:
        positions = np.asarray(mesh.positions, np.float32).reshape(-1, 3) * np.float32(scale)
        uvs = np.asarray(mesh.uvs, np.float32).reshape(-1, 2)
        normals = np.asarray(mesh.normals, np.float32).reshape(-1, 3)
        has_uv = bool(getattr(mesh, "has_uvs", True)) and len(uvs) == len(positions)
        mats = np.asarray(mesh.material_ids, np.int64)
        sizes = polygon_sizes(mesh)
        # 多边形按材质排（四边形的两个三角形材质相同，取第一个）
        tri_of_poly = np.concatenate([[0], np.cumsum(np.where(sizes == 4, 2, 1))[:-1]]).astype(np.int64)
        poly_mat = mats[tri_of_poly] if len(mats) else np.zeros(len(sizes), np.int64)
        order = np.argsort(poly_mat, kind="stable")
        sel = _corner_selection(tri_of_poly[order] * 3, np.where(sizes[order] == 4, 6, 3))
        pos_unique, pos_idx = unique_rows(positions[sel])
        out.write(("o %s\n" % _clean(name)).encode("utf-8"))
        out.write(_rows_text(np.frombuffer(b"v", np.uint8), pos_unique.astype(np.float64)).tobytes())
        uv_idx = np.zeros(len(sel), np.int64)
        uv_count = 0
        if has_uv:
            uv_unique, uv_idx = unique_rows(uvs[sel])
            uv_count = len(uv_unique)
            out.write(_rows_text(np.frombuffer(b"vt", np.uint8), uv_unique.astype(np.float64)).tobytes())
        nrm_unique, nrm_idx = unique_rows(normals[sel])
        out.write(_rows_text(np.frombuffer(b"vn", np.uint8), nrm_unique.astype(np.float64)).tobytes())
        sorted_sizes = sizes[order]
        sorted_mats = poly_mat[order]
        poly_first = np.concatenate([[0], np.cumsum(np.where(sorted_sizes == 4, 6, 3))[:-1]]).astype(np.int64)
        for material in np.unique(sorted_mats):
            picked = np.nonzero(sorted_mats == material)[0]
            label = material_names[int(material)] if int(material) < len(material_names) else "material_%d" % material
            used_materials.append(label)
            out.write(("usemtl %s\n" % _clean(label)).encode("utf-8"))
            corners = _corner_selection(poly_first[picked], np.where(sorted_sizes[picked] == 4, 6, 3))
            out.write(_faces_text(pos_idx[corners] + 1 + base_v, uv_idx[corners] + 1 + base_vt,
                                  nrm_idx[corners] + 1 + base_vn, sorted_sizes[picked], has_uv, True).tobytes())
        base_v += len(pos_unique)
        base_vt += uv_count
        base_vn += len(nrm_unique)
        totals["triangles"] += mesh.triangle_count
        totals["quads"] += int((sizes == 4).sum())
        totals["vertices"] += len(pos_unique)
    _write_bytes(path, out.getvalue())
    if write_mtl:
        lines = ["# SPLENDER"]
        for label in dict.fromkeys(used_materials):
            maps = (textures or {}).get(label, {})
            lines += ["", "newmtl %s" % _clean(label), "Kd 0.8 0.8 0.8", "Ks 0.04 0.04 0.04", "Ns 200", "d 1", "illum 2"]
            if maps.get("basecolor"):
                lines.append("map_Kd %s" % maps["basecolor"])
            if maps.get("roughness"):
                lines.append("map_Pr %s" % maps["roughness"])
            if maps.get("metallic"):
                lines.append("map_Pm %s" % maps["metallic"])
            if maps.get("normal"):
                lines.append("norm %s" % maps["normal"])
                lines.append("map_Bump -bm 1.0 %s" % maps["normal"])
        _write_bytes(os.path.join(folder, mtl_name), ("\n".join(lines) + "\n").encode("utf-8"))
    totals["seconds"] = round(time.perf_counter() - started, 3)
    totals["path"] = path
    log.info("已导出模型 %s：%d 个三角形（%d 个四边形），用时 %.2f 秒", path, totals["triangles"], totals["quads"],
             totals["seconds"])
    return totals


def _corner_selection(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """一串 [start, start+count) 区间拼起来（向量化，不用 Python 循环）。"""
    starts = np.asarray(starts, np.int64)
    counts = np.asarray(counts, np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    ends = np.cumsum(counts)
    within = np.arange(total) - np.repeat(ends - counts, counts)
    return np.repeat(starts, counts) + within


def _clean(text: str) -> str:
    return "".join("_" if ch.isspace() else ch for ch in str(text)) or "_"


def _write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".part"
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


# ====================================================================== glTF 二进制
def export_glb(path: str, objects: list, *, materials: dict | None = None, scale: float = 1.0) -> dict:
    """objects：[(名字, MeshData, 材质名列表)]。materials：{材质名: {"basecolor": PNG 字节, "orm": PNG 字节（R 遮蔽、
    G 粗糙度、B 金属度）, "normal": PNG 字节, "metallic": 数, "roughness": 数, "color": (r,g,b)}}。"""
    started = time.perf_counter()
    blobs: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []
    offset = 0

    def add_view(data: bytes, target: int | None = None) -> int:
        nonlocal offset
        pad = (-offset) % 4
        if pad:
            blobs.append(b"\0" * pad)
            offset += pad
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        views.append(view)
        blobs.append(data)
        offset += len(data)
        return len(views) - 1

    def add_accessor(array: np.ndarray, kind: str, component: int, target: int | None, bounds: bool = False) -> int:
        view = add_view(np.ascontiguousarray(array).tobytes(), target)
        acc = {"bufferView": view, "componentType": component, "count": int(len(array)), "type": kind}
        if bounds and len(array):
            acc["min"] = [float(v) for v in array.min(axis=0)]
            acc["max"] = [float(v) for v in array.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    gl_materials: list[dict] = []
    images: list[dict] = []
    textures: list[dict] = []
    material_index: dict[str, int] = {}

    def texture_of(png: bytes | None) -> int | None:
        if not png:
            return None
        view = add_view(png)
        images.append({"bufferView": view, "mimeType": "image/png"})
        textures.append({"source": len(images) - 1, "sampler": 0})
        return len(textures) - 1

    def material_of(label: str) -> int:
        if label in material_index:
            return material_index[label]
        info = (materials or {}).get(label, {})
        color = list(info.get("color", (0.8, 0.8, 0.8)))
        pbr = {"baseColorFactor": [float(color[0]), float(color[1]), float(color[2]), 1.0],
               "metallicFactor": float(info.get("metallic", 0.0)), "roughnessFactor": float(info.get("roughness", 0.5))}
        material = {"name": label, "pbrMetallicRoughness": pbr}
        base = texture_of(info.get("basecolor"))
        if base is not None:
            pbr["baseColorTexture"] = {"index": base}
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        orm = texture_of(info.get("orm"))
        if orm is not None:
            pbr["metallicRoughnessTexture"] = {"index": orm}
            pbr["metallicFactor"] = 1.0
            pbr["roughnessFactor"] = 1.0
            material["occlusionTexture"] = {"index": orm}
        normal = texture_of(info.get("normal"))
        if normal is not None:
            material["normalTexture"] = {"index": normal}
        gl_materials.append(material)
        material_index[label] = len(gl_materials) - 1
        return material_index[label]

    meshes, nodes = [], []
    triangles = 0
    for name, mesh, material_names in objects:
        positions = np.asarray(mesh.positions, np.float32).reshape(-1, 3) * np.float32(scale)
        normals = np.asarray(mesh.normals, np.float32).reshape(-1, 3)
        uvs = np.asarray(mesh.uvs, np.float32).reshape(-1, 2).copy()
        uvs[:, 1] = 1.0 - uvs[:, 1]                  # glTF 的 UV 原点在左上
        mats = np.asarray(mesh.material_ids, np.int64)
        primitives = []
        for material in np.unique(mats):
            tris = np.nonzero(mats == material)[0]
            corners = (tris[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
            packed = np.concatenate([positions[corners], normals[corners], uvs[corners]], axis=1)
            verts, index = unique_rows(packed)
            label = material_names[int(material)] if int(material) < len(material_names) else "material_%d" % material
            prim = {"attributes": {"POSITION": add_accessor(verts[:, 0:3], "VEC3", 5126, 34962, bounds=True),
                                   "NORMAL": add_accessor(verts[:, 3:6], "VEC3", 5126, 34962),
                                   "TEXCOORD_0": add_accessor(verts[:, 6:8], "VEC2", 5126, 34962)},
                    "indices": add_accessor(index.astype(np.uint32), "SCALAR", 5125, 34963),
                    "material": material_of(label)}
            primitives.append(prim)
        meshes.append({"name": name, "primitives": primitives})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
        triangles += mesh.triangle_count
    document = {"asset": {"version": "2.0", "generator": "SPLENDER"}, "scene": 0,
                "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
                "materials": gl_materials, "accessors": accessors, "bufferViews": views,
                "buffers": [{"byteLength": 0}]}
    if images:
        document["images"] = images
        document["textures"] = textures
        document["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    binary = b"".join(blobs)
    binary += b"\0" * ((-len(binary)) % 4)
    document["buffers"][0]["byteLength"] = len(binary)
    text = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    text += b" " * ((-len(text)) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    data = (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(text), 0x4E4F534A) + text
            + struct.pack("<II", len(binary), 0x004E4942) + binary)
    _write_bytes(str(path), data)
    seconds = round(time.perf_counter() - started, 3)
    log.info("已导出 glTF %s：%d 个三角形，%d 张贴图，用时 %.2f 秒", path, triangles, len(images), seconds)
    return {"path": str(path), "triangles": triangles, "images": len(images), "seconds": seconds, "bytes": len(data)}
