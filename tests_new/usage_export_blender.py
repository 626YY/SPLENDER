"""实际使用测：雕刻里体素重构 → 自动展开 UV → 画几笔 → 导出 glb → 用 Blender（后台、出厂设置，不开窗口）导入，
按 Blender 读到的 UV 去它读到的贴图里取色，和 SPLENDER 自己的合成比：贴图对位对不对、颜色对不对、位置对不对；
材质的各张贴图接没接上。结果写到 docs/evidence/selftest/usage_export_blender.json。
用法：python -m splender --selftest tests_new/usage_export_blender.py。变量 app 和 d 由自检框架提供。
找不到 Blender 时只做导出这一半。"""
import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree

from splender.engine.projectio import composite_image

BLENDER_SCRIPT = r'''
import bpy, json, sys
import numpy as np
argv = sys.argv[sys.argv.index("--") + 1:]
glb, out = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
started = __import__("time").perf_counter()
bpy.ops.import_scene.gltf(filepath=glb)
seconds = __import__("time").perf_counter() - started
obj = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
me = obj.data
mat = me.materials[0]
nodes = mat.node_tree.nodes
bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
def source(name):
    links = bsdf.inputs[name].links
    return links[0].from_node if links else None
base_node = source("Base Color")
image = base_node.image
w, h = image.size
pixels = np.array(image.pixels[:], np.float32).reshape(h, w, image.channels)
uv = me.uv_layers.active.data
count = len(me.polygons)
pick = np.linspace(0, count - 1, min(4000, count)).astype(int)
mw = obj.matrix_world
rows = []
for k in pick:
    poly = me.polygons[int(k)]
    cu = sum(uv[i].uv[0] for i in poly.loop_indices) / len(poly.loop_indices)
    cv = sum(uv[i].uv[1] for i in poly.loop_indices) / len(poly.loop_indices)
    x = min(w - 1, max(0, int(cu * w)))
    y = min(h - 1, max(0, int(cv * h)))
    color = pixels[y, x, :3]
    center = mw @ poly.center
    rows.append([float(cu), float(cv), float(color[0]), float(color[1]), float(color[2]),
                 float(center.x), float(center.y), float(center.z)])
links = {name: (source(name).type if source(name) else None) for name in ("Base Color", "Metallic", "Roughness", "Normal")}
json.dump({"polygons": count, "image": [w, h], "colorspace": image.colorspace_settings.name, "links": links,
           "import_seconds": seconds, "samples": rows}, open(out, "w"))
'''


def find_blender():
    found = sorted(glob.glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe"))
    return found[-1] if found else None


report = {}
work = tempfile.mkdtemp(prefix="splender_export_blender_")
try:
    app.window.set_workspace("雕刻")
    d.settle()
    started = time.perf_counter()
    d.call("sculpt.voxel_remesh")
    d.settle()
    app.window.set_workspace("绘制")
    d.settle()
    report["remesh_and_unwrap_seconds"] = round(time.perf_counter() - started, 2)
    obj = app.project.objects[0]
    report["triangles"] = obj.data.triangle_count
    view = d.editor("VIEW_3D")
    cx, cy = view.widget.width() / 2, view.widget.height() / 2
    for k, color in enumerate(((0.9, 0.1, 0.1), (0.1, 0.8, 0.2), (0.1, 0.2, 0.9))):
        app.tool_settings.brush.color = color
        app.tool_settings.brush.size = 60
        d.stroke(view, [(cx - 150 + 300 * i / 40, cy - 70 + 70 * k + 15 * math.sin(i * 0.3)) for i in range(41)])
        d.settle()
    ts = app.project.active_texture_set
    glb = os.path.join(work, "导出.glb")
    started = time.perf_counter()
    assert d.call("wm.export_mesh", filepath=glb, size="2048") == "FINISHED"
    report["export_seconds"] = round(time.perf_counter() - started, 2)
    report["glb_mb"] = round(os.path.getsize(glb) / 2 ** 20, 1)
    blender = find_blender()
    report["blender"] = blender
    if blender:
        script = os.path.join(work, "check.py")
        open(script, "w", encoding="utf-8").write(BLENDER_SCRIPT)
        out = os.path.join(work, "blender.json")
        started = time.perf_counter()
        proc = subprocess.run([blender, "--background", "--factory-startup", "--python-exit-code", "1", "--python", script,
                               "--", glb, out], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=600)
        report["blender_seconds"] = round(time.perf_counter() - started, 2)
        assert proc.returncode == 0 and os.path.exists(out), (proc.stdout + proc.stderr)[-3000:]
        info = json.load(open(out, encoding="utf-8"))
        samples = np.asarray(info["samples"], np.float64)
        report.update(polygons=info["polygons"], links=info["links"], colorspace=info["colorspace"],
                      import_seconds=round(info["import_seconds"], 2))
        # Blender 是 Z 朝上：(x, y, z)_Blender = (x, -z, y)_内部
        centers_b = np.stack([samples[:, 5], samples[:, 7], -samples[:, 6]], axis=1)
        tri = obj.data.positions.reshape(-1, 3, 3).mean(axis=1)
        dist, which = cKDTree(tri).query(centers_b)
        report["position_err_max"] = float(dist.max())
        ours_uv = obj.data.uvs.reshape(-1, 3, 2).mean(axis=1)[which]
        report["uv_err_max"] = float(np.abs(ours_uv - samples[:, 0:2]).max())
        size = info["image"][0]
        reference = composite_image(app.engine, ts, size, dtype=np.uint8)[:, :, :3].astype(np.float64) / 255.0
        px = np.clip((samples[:, 0] * size).astype(np.int64), 0, size - 1)
        py = np.clip((samples[:, 1] * size).astype(np.int64), 0, size - 1)
        diff = np.abs(reference[py, px] - samples[:, 2:5]) * 255.0
        report["color_diff_p99"] = round(float(np.percentile(diff, 99)), 2)
        report["color_diff_max"] = round(float(diff.max()), 2)
        report["painted_samples"] = int((np.ptp(samples[:, 2:5], axis=1) > 0.25).sum())
        assert report["position_err_max"] < 1e-3, report
        assert report["uv_err_max"] < 1e-4, report
        assert report["color_diff_p99"] <= 3.0, report
        assert report["painted_samples"] > 20, report
        assert report["links"]["Base Color"] == "TEX_IMAGE" and report["links"]["Normal"] == "NORMAL_MAP", report
        assert report["links"]["Metallic"] and report["links"]["Roughness"], report
finally:
    (d.out / "usage_export_blender.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    shutil.rmtree(work, ignore_errors=True)
