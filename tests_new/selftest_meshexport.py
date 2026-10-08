"""整机自检：导出模型。雕刻里体素重构 → 回到绘制自动展开 UV → 画两笔 → 导出 glb（贴图装在里面）、obj（旁边写 PNG 和 .mtl）
→ 读回检查：位置、UV、材质贴图齐全，glb 里的基础色贴图在每个角的 UV 处和画面合成一致；导出贴图时「连模型一起导出」。
变量 app 和 d 由自检框架提供。"""
import io
import json
import logging
import math
import os
import shutil
import struct
import tempfile

import numpy as np
from PIL import Image

errors = []


class _Catch(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())


logging.getLogger("splender").addHandler(_Catch())
from splender.engine.projectio import composite_image  # noqa: E402

results = {}
d.out.mkdir(parents=True, exist_ok=True)
folder = tempfile.mkdtemp(prefix="splender_meshexport_")


def read_glb(path):
    data = open(path, "rb").read()
    json_len = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + json_len].decode("utf-8"))
    bin_len = struct.unpack_from("<I", data, 20 + json_len)[0]
    return document, data[28 + json_len:28 + json_len + bin_len]


def accessor(document, binary, index):
    acc = document["accessors"][index]
    view = document["bufferViews"][acc["bufferView"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
    dtype = {5126: np.float32, 5125: np.uint32}[acc["componentType"]]
    raw = np.frombuffer(binary, dtype, acc["count"] * width, view["byteOffset"])
    return raw.reshape(-1, width) if width > 1 else raw


try:
    # 1. 体素重构 + 自动展开 UV
    app.window.set_workspace("雕刻")
    d.settle()
    assert app.tool_settings.mode == "SCULPT"
    d.call("sculpt.voxel_remesh")
    d.settle()
    app.window.set_workspace("绘制")
    d.settle()
    obj = app.project.objects[0]
    assert getattr(obj.data, "has_uvs", False), "重构后回到绘制应自动展开 UV"
    results["triangles"] = obj.data.triangle_count
    # 2. 画两笔
    view = d.editor("VIEW_3D")
    cx, cy = view.widget.width() / 2, view.widget.height() / 2
    for dy, color in ((-40, (0.9, 0.15, 0.1)), (40, (0.1, 0.35, 0.9))):
        app.tool_settings.brush.color = color
        d.stroke(view, [(cx - 150 + 300 * i / 40, cy + dy + 15 * math.sin(i * 0.3)) for i in range(41)])
        d.settle()
    ts = app.project.active_texture_set

    # 3. glb：贴图装在里面
    glb = os.path.join(folder, "导出.glb")
    assert d.call("wm.export_mesh", filepath=glb, size="1024") == "FINISHED"
    document, binary = read_glb(glb)
    material = document["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    assert material["name"] == ts.name
    for key in ("baseColorTexture", "metallicRoughnessTexture"):
        assert key in pbr, key
    assert "normalTexture" in material and "occlusionTexture" in material
    images = []
    for image in document["images"]:
        view_info = document["bufferViews"][image["bufferView"]]
        images.append(Image.open(io.BytesIO(binary[view_info["byteOffset"]:view_info["byteOffset"] + view_info["byteLength"]])))
    results["glb_images"] = [im.size for im in images]
    assert all(im.size == (1024, 1024) for im in images)
    prim = document["meshes"][0]["primitives"][0]
    index = accessor(document, binary, prim["indices"])
    positions = accessor(document, binary, prim["attributes"]["POSITION"])[index]
    uvs = accessor(document, binary, prim["attributes"]["TEXCOORD_0"])[index]
    assert np.allclose(positions, obj.data.positions, atol=1e-5), "位置对得上"
    assert np.allclose(uvs[:, 0], obj.data.uvs[:, 0], atol=1e-6) and np.allclose(1 - uvs[:, 1], obj.data.uvs[:, 1], atol=1e-6)
    # 基础色贴图：按每个三角形重心的 UV 取色，和画面合成（同一尺寸）一致
    base = np.asarray(images[document["textures"][pbr["baseColorTexture"]["index"]]["source"]].convert("RGB"),
                      np.int16)
    reference = composite_image(app.engine, ts, 1024, dtype=np.uint8)[:, :, :3].astype(np.int16)
    centers = obj.data.uvs.reshape(-1, 3, 2).mean(axis=1)
    pick = np.linspace(0, len(centers) - 1, 4000).astype(np.int64)
    px = np.clip((centers[pick, 0] * 1024).astype(np.int64), 0, 1023)
    py_img = np.clip(((1.0 - centers[pick, 1]) * 1024).astype(np.int64), 0, 1023)      # 图片第一行是 v=1
    from_glb = base[py_img, px]
    # composite_image 的行序和贴图一样（第 0 行是 v=0）
    py_tex = np.clip((centers[pick, 1] * 1024).astype(np.int64), 0, 1023)
    from_engine = reference[py_tex, px]
    diff = np.abs(from_glb - from_engine)
    results["glb_basecolor_diff_p99"] = float(np.percentile(diff, 99))
    results["glb_basecolor_colorful"] = int((np.ptp(from_glb, axis=1) > 60).sum())
    assert results["glb_basecolor_diff_p99"] <= 3, results["glb_basecolor_diff_p99"]
    assert results["glb_basecolor_colorful"] > 20, "两笔颜色应该在导出的贴图里"

    # 4. obj：旁边写 PNG 和 .mtl
    obj_path = os.path.join(folder, "导出.obj")
    assert d.call("wm.export_mesh", filepath=obj_path, size="1024") == "FINISHED"
    mtl = open(os.path.join(folder, "导出.mtl"), encoding="utf-8").read()
    names = [line.split(" ", 1)[1].strip() for line in mtl.splitlines() if line.startswith(("map_Kd", "map_Pr", "map_Pm", "norm"))]
    results["obj_textures"] = names
    assert len(names) == 4 and all(os.path.isfile(os.path.join(folder, name)) for name in names), names
    faces = sum(1 for line in open(obj_path, encoding="utf-8") if line.startswith("f "))
    assert faces == obj.data.triangle_count, (faces, obj.data.triangle_count)

    # 5. 导出贴图时连模型一起导出
    textures_dir = os.path.join(folder, "贴图")
    os.makedirs(textures_dir)
    ts.export.with_mesh = True
    ts.export.mesh_format = "OBJ"
    ts.export.size = "1024"
    assert d.call("wm.export_textures", directory=textures_dir) == "FINISHED"
    listing = sorted(os.listdir(textures_dir))
    results["with_mesh_files"] = listing
    assert any(name.endswith(".obj") for name in listing) and any(name.endswith(".mtl") for name in listing)
    mtl_name = next(name for name in listing if name.endswith(".mtl"))
    refs = [line.split(" ", 1)[1].strip() for line in open(os.path.join(textures_dir, mtl_name), encoding="utf-8")
            if line.startswith("map_Kd")]
    assert refs and all(ref in listing for ref in refs), "材质文件引用刚导出的贴图"

    # 6. 导出面板
    props = d.editor("PROPERTIES")
    if props is not None:
        props.tabs.setCurrent("SET")
        props._on_tab("SET")
        d.settle()
        try:
            from PySide6.QtWidgets import QScrollArea

            for area in getattr(props, "widget", props).findChildren(QScrollArea):
                bar = area.verticalScrollBar()
                bar.setValue(bar.maximum())
        except Exception as error:  # noqa: BLE001
            results["scroll_error"] = str(error)
        d.settle()
        d.shot("meshexport_01_panel.png")
finally:
    shutil.rmtree(folder, ignore_errors=True)
(d.out / "meshexport_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
