"""人工检查用：跨 UV 接缝画一笔，在几个缩放级别下渲染，看接缝处有没有线。不开窗口。"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path

ensure_vendor_path()
import moderngl
import numpy as np
from PIL import Image

from splender.core.history import History
from splender.core.prefs import Preferences
from splender.doc.project import ToolSettings, ViewShading
from splender.engine.engine import Engine
from tests_new.test_engine_paint import make_project

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/engine/seam")
OUT.mkdir(parents=True, exist_ok=True)
ctx = moderngl.create_standalone_context(require=430)
engine = Engine(ctx, Preferences(), History())
project, ts = make_project()
engine.set_project(project)
mesh = project.objects[0].data
# 找到 u≈0 的经线在三维里的方向，让相机正对它
uvs, pos = mesh.uvs, mesh.positions
pick = np.argmin(np.abs(uvs[:, 0]) + np.abs(uvs[:, 1] - 0.5))
direction = pos[pick] / np.linalg.norm(pos[pick])
yaw = math.degrees(math.atan2(direction[0], direction[2]))
print("接缝方向", direction, "相机偏航", yaw)
view = engine.create_view(ViewShading())
view.resize(1280, 800)
engine.frame_view(view)
view.camera.yaw = yaw
view.camera.pitch = 8.0
view.invalidate_camera()
engine.settle()

tools = ToolSettings()
tools.brush.size = 70
tools.brush.color = (0.1, 0.45, 0.85)
tools.brush.use_height = True
tools.brush.height = 0.06
tools.brush.hardness = 0.6
cx, cy = view.width / 2, view.height / 2
for row, (dy, color) in enumerate(((-70, (0.1, 0.45, 0.85)), (40, (0.9, 0.75, 0.1)))):
    tools.brush.color = color
    engine.stroke_begin(view, cx - 170, cy + dy, 1.0, time.perf_counter(), tools)
    for i in range(1, 35):
        engine.stroke_move(cx - 170 + i * 10, cy + dy + 12 * math.sin(i * 0.5), 1.0, time.perf_counter())
        engine.frame()
    engine.stroke_end()
    engine.settle()


def shot(name):
    engine.settle()
    Image.fromarray(engine.renderer.read_pixels(view)[:, :, :3]).save(OUT / name)


shot("zoom_1x.png")
base = view.camera.distance
for factor in (3.0, 8.0, 20.0):
    view.camera.distance = base / factor
    view.invalidate_camera()
    shot("zoom_%dx.png" % factor)
for factor in (0.5, 0.25):
    view.camera.distance = base / factor
    view.invalidate_camera()
    shot("zoom_out_%s.png" % str(factor).replace(".", "_"))
view.camera.distance = base / 3.0
view.shading.mode = "CHANNEL"
view.shading.channel = "basecolor"
view.invalidate_camera()
shot("zoom_3x_basecolor.png")
print(engine.stats())

from tests_new._dump import plane_image
from splender.doc.project import PLANE_COLOR
layer = ts.layers[1]
img = plane_image(engine, layer.uid, PLANE_COLOR, 3, channels=(3, 3, 3))
img.save(OUT / "layer_coverage_level3.png")
img = plane_image(engine, layer.uid, PLANE_COLOR, 2, channels=(3, 3, 3))
box = img.getbbox()
print("level2 coverage bbox", box, img.size)
if box:
    img.crop(box).save(OUT / "layer_coverage_level2_crop.png")
print("mesh", mesh.triangle_count, "uv range", mesh.uvs.min(0), mesh.uvs.max(0))
