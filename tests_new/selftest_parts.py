"""整机自检：智能遮罩和部件。两个分开的球，烘焙后分成两个部件；加一层红色填充，先套智能遮罩，
再在视口里点右边的球（只盖住它），撤销回到智能遮罩。变量 app 和 d 由自检框架提供。"""
import json
import logging
import time

import numpy as np

from splender.core.keymap import LEFTMOUSE, PRESS, RELEASE
from splender.doc.meshio import MeshData, make_test_mesh

errors = []


class _Catch(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())


logging.getLogger("splender").addHandler(_Catch())
app.window.set_workspace("绘制")       # 这个自检从绘制工作区（纹理绘制模式）开始
results = {}
d.settle()
d.out.mkdir(parents=True, exist_ok=True)

# 1. 两个分开的球，UV 各占一半
base = make_test_mesh("sphere", segments=64, rings=32)
positions, uvs = [], []
for shift, u0 in ((-1.2, 0.0), (1.2, 0.5)):
    p = np.asarray(base.positions, np.float32).copy()
    p[:, 0] += shift
    positions.append(p)
    uv = np.asarray(base.uvs, np.float32).copy()
    uv[:, 0] = u0 + uv[:, 0] * 0.5
    uvs.append(uv)
positions = np.concatenate(positions)
mesh = MeshData(name="双球", positions=positions, normals=np.concatenate([base.normals, base.normals]),
                uvs=np.concatenate(uvs), material_ids=np.zeros(len(positions) // 3, np.int32), materials=["材质"],
                bounds_min=positions.min(axis=0), bounds_max=positions.max(axis=0))
app.new_project(mesh, name="部件自检", resolution="2048")
d.settle()
ts = app.project.active_texture_set
ts.meshmap.resolution = "1024"
ts.meshmap.ao_samples = 16
view = d.editor("VIEW_3D")
app.engine.frame_view(view.view)
d.settle()

# 2. 烘焙
assert d.call("bake.mesh_maps") == "FINISHED"
t0 = time.perf_counter()
while app.engine.bake is not None and time.perf_counter() - t0 < 120:
    d.wait(50)
d.settle()
maps = app.engine.meshmap(ts.uid)
results["parts"] = maps.part_count
assert maps.part_count == 2, "两个球应该分成两个部件：%s" % maps.part_count

# 3. 红色填充层，套智能遮罩
assert d.call("layer.add_fill") == "FINISHED"
layer = ts.active_layer
layer.fill_color = (0.85, 0.12, 0.08)
assert d.call("layer.apply_smart_mask", mask="edge_chips") == "FINISHED"
results["after_smart_mask"] = layer.mask_generator
assert layer.mask_generator == "EDGES"
d.settle()
d.shot("parts_01_smart_mask.png")

# 4. 视口里点选右边的球
w, h = view.widget.width(), view.widget.height()
found = None
for fx in (0.70, 0.65, 0.75, 0.6):
    hit = app.engine.part_at(view.view, w * fx, h * 0.5)
    if hit is not None and hit[1] >= 0:
        found = (w * fx, h * 0.5, hit[1])
        break
assert found is not None, "视口右半边应该能点到球"
x, y, part = found
assert d.call("layer.pick_parts") == "RUNNING_MODAL"
d.move(view, x, y)
d.event(view, LEFTMOUSE, PRESS, x, y)
d.event(view, LEFTMOUSE, RELEASE, x, y)
d.key(view, "ESC")
d.settle()
results["picked"] = layer.gen_parts
results["generator"] = layer.mask_generator
assert layer.mask_generator == "PARTS"
assert layer.gen_parts == str(part), layer.gen_parts
image = app.engine.renderer.read_pixels(view.view).astype(int)
red = image[:, :, 0] - image[:, :, 1]
ys = int(image.shape[0] * 0.5)
px_right = int(image.shape[1] * (x / w))
px_left = image.shape[1] - px_right
results["red_right"] = float(red[ys - 10:ys + 10, px_right - 10:px_right + 10].mean())
results["red_left"] = float(red[ys - 10:ys + 10, px_left - 10:px_left + 10].mean())
assert results["red_right"] > 40, results
assert abs(results["red_left"]) < 15, results
d.shot("parts_02_picked.png")

# 5. 面板
props = d.editor("PROPERTIES")
if props is not None:
    props.tabs.setCurrent("LAYER")
    props._on_tab("LAYER")
    d.settle()
    from PySide6.QtWidgets import QScrollArea
    for area in getattr(props, "widget", props).findChildren(QScrollArea):
        area.verticalScrollBar().setValue(area.verticalScrollBar().maximum())
    d.settle()
    d.shot("parts_03_panel.png")
    props.tabs.setCurrent("SET")
    props._on_tab("SET")
    d.settle()
    d.shot("parts_04_meshmaps.png")

# 6. 撤销：回到智能遮罩
d.call("ed.undo")
d.settle()
results["after_undo"] = (layer.mask_generator, layer.gen_parts)
assert layer.mask_generator == "EDGES" and layer.gen_parts == "", results["after_undo"]
d.call("ed.redo")
d.settle()
assert layer.mask_generator == "PARTS" and layer.gen_parts == str(part)
(d.out / "parts_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
