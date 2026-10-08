"""给每个智能材质渲染一张缩略图，存到 splender/resources/smart/<id>.png。

用法：python tools/render_smart_thumbs.py [--size 160]

做法：离屏建一个引擎和一颗「材质球」（带一圈凹槽、下半部竖棱、顶上几个鼓包，边缘、凹陷、朝上的面都有），
自动展开 UV、烘焙一次模型贴图；每个智能材质加上后用材质预览渲染一张，四倍超采样后缩小，存成带透明底的 PNG。
材质改了（bake/smart.py）就重新跑一遍。
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from splender.bake import smart  # noqa: E402
from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import MeshData  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402
from splender.geometry.unwrap import smart_unwrap  # noqa: E402
from splender.sculpt import topology as tp  # noqa: E402

OUT = ROOT / "splender" / "resources" / "smart"


def material_ball() -> MeshData:
    """一颗带特征的球：赤道上方一圈凹槽、下半部一圈竖棱、顶上几个鼓包。"""
    mesh = tp.subdivide(tp.make_sphere(64), 1)
    v = mesh.vertices.astype(np.float64)
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    y = n[:, 1]
    angle = np.arctan2(n[:, 2], n[:, 0])
    groove = -0.075 * np.exp(-((y - 0.28) / 0.045) ** 2)
    ridges = 0.05 * np.clip(np.sin(angle * 10.0), 0.0, None) ** 3 * np.clip((-0.15 - y) / 0.25, 0.0, 1.0) * \
        np.clip((y + 0.9) / 0.2, 0.0, 1.0)
    bumps = np.zeros(len(v))
    for cx, cz in ((0.25, 0.1), (-0.2, 0.3), (0.05, -0.3)):
        center = np.array([cx, math.sqrt(max(0.0, 1.0 - cx * cx - cz * cz)), cz])
        bumps += 0.06 * np.exp(-np.sum((n - center) ** 2, axis=1) / 0.012)
    radius = 1.0 + groove + ridges + bumps
    mesh.vertices = (n * radius[:, None]).astype(np.float32)
    uvs, _stats = smart_unwrap(mesh.vertices, mesh.triangles, None, angle_limit=50.0, resolution=2048,
                               margin_texels=24)
    positions, normals, _uv, _mats = tp.unweld(mesh)
    count = len(mesh.triangles)
    return MeshData(name="材质球", positions=positions, normals=normals, uvs=uvs,
                    material_ids=np.zeros(count, np.int32), materials=["材质"],
                    bounds_min=positions.min(axis=0), bounds_max=positions.max(axis=0))


def main() -> int:
    parser = argparse.ArgumentParser(description="渲染智能材质缩略图")
    parser.add_argument("--size", type=int, default=160, help="输出边长（像素）")
    parser.add_argument("--only", default="", help="只渲染这个材质（id）")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = moderngl.create_standalone_context(require=430)
    engine = Engine(ctx, Preferences(), History())
    project = Project("缩略图")
    ts = project.add_texture_set(TextureSet(name="材质", resolution="2048"))
    ts.meshmap.resolution = "2048"
    ts.meshmap.ao_samples = 96
    ts.meshmap.ao_distance = 0.12
    mesh = material_ball()
    project.add_object(MeshObject(mesh.name, mesh, [ts.uid]))
    engine.set_project(project)
    started = time.perf_counter()
    engine.bake_start(ts)
    while engine.bake is not None:
        engine.frame()
    print("模型贴图烘焙 %.1f 秒" % (time.perf_counter() - started))
    shading = ViewShading(mode="MATERIAL", hdri="interior", exposure=0.3)
    view = engine.create_view(shading)
    side = args.size * 4
    view.resize(side, side)
    engine.frame_view(view)
    camera = view.camera
    camera.yaw, camera.pitch = 30.0, 18.0
    camera.distance *= 0.82
    view.invalidate_camera()
    for item in smart.SMART_MATERIALS:
        if args.only and item["id"] != args.only:
            continue
        ts.layers.clear()
        ts.active_layer_uid = 0
        folder = Layer(name=item["name"], kind="FOLDER")
        ts.add_layer(folder)
        for values in item["layers"]:
            values = dict(values)
            layer = Layer(name=values.pop("name"), **values)
            layer.parent_uid = folder.uid
            ts.add_layer(layer, ts.index_of(folder))
        engine.settle(2000)
        rgba = engine.renderer.read_pixels(view).astype(np.float32)
        image = Image.fromarray(rgba.astype(np.uint8), "RGBA")
        # 背景是统一的底色：按和底色的差别做透明度（球的边缘抗锯齿过渡保留）
        bg = rgba[2, 2, :3]
        diff = np.abs(rgba[:, :, :3] - bg).max(axis=2)
        alpha = np.clip(diff * 12.0, 0, 255)
        coverage = engine.renderer.read_pixels(view)            # 再读一次，确认画面已稳定
        if not np.array_equal(coverage, rgba.astype(np.uint8)):
            engine.settle(2000)
        out = np.dstack([rgba[:, :, :3], alpha]).astype(np.uint8)
        image = Image.fromarray(out, "RGBA").resize((args.size, args.size), Image.LANCZOS)
        path = OUT / ("%s.png" % item["id"])
        image.save(path, optimize=True)
        print("已渲染", item["name"], path)
    engine.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
