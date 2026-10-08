"""整机自检：材质节点（着色器编辑器）。「材质」工作区里：上面视口、下面着色器编辑器；点「使用节点」得到图层堆栈接材质输出；
加噪波纹理和颜色渐变节点并连到基础色，视口跟着变；节点里的数值框、色带是真控件，改了立刻生效；X 删除、Ctrl+Z 撤销；
M 静音、H 折叠；存盘再打开材质节点还在。变量 app 和 d 由自检框架提供。"""
import json

import numpy as np

from splender.core.keymap import LEFTMOUSE, MOUSEMOVE, MOVE, PRESS, RELEASE

results = {}
d.settle()
win = app.window
win.set_workspace("材质")
d.settle()
shader = d.editor("SHADER_EDITOR")
v3d = d.editor("VIEW_3D")
assert shader is not None, "材质工作区里要有着色器编辑器"
assert v3d is not None
project = app.project
ts = project.active_texture_set
assert ts is not None
results["material_before"] = ts.material is not None
# ---- 使用节点
if ts.material is None:
    assert d.call("shader.use_nodes", shader) == "FINISHED"
d.settle()
graph = ts.material
assert graph is not None and graph.output_node() is not None
kinds = sorted(n.type_id for n in graph.nodes.values())
results["default_nodes"] = kinds
assert "LAYER_STACK" in kinds and "OUTPUT_MATERIAL" in kinds
assert len(shader.items) == len(graph.nodes), (len(shader.items), len(graph.nodes))
d.shot("shader_00_default.png", widget=shader.view)
# ---- 加噪波纹理、颜色渐变，连到基础色
before_color = None
v3d.shading.mode = "MATERIAL"
d.settle()
pixels = app.engine.renderer.read_pixels(v3d.view)[:, :, :3].astype(np.float64)
before_mean = pixels.mean(axis=(0, 1))
stack = next(n for n in graph.nodes.values() if n.type_id == "LAYER_STACK")
out = graph.output_node()
noise = graph.add_node("TEX_NOISE", (stack.location[0], stack.location[1] + 260))
ramp_node = graph.add_node("VALTORGB", (stack.location[0] + 220, stack.location[1] + 260))
ramp_node.values.ramp = ("LINEAR", ((0.35, 0.05, 0.1, 0.9, 1.0), (0.65, 0.95, 0.75, 0.1, 1.0)))
graph.link(noise.uid, "fac", ramp_node.uid, "fac")
graph.link(ramp_node.uid, "color", out.uid, "base_color")
d.settle()
d.wait(200)
d.settle()
assert len(shader.items) == len(graph.nodes)
item = shader.items[ramp_node.uid]
assert item.proxy is not None and item.body is not None, "颜色渐变节点里要有真控件"
from splender.ui.ramp_widget import ColorRampWidget  # noqa: E402

ramps = item.body.findChildren(ColorRampWidget)
results["ramp_widgets"] = len(ramps)
assert len(ramps) == 1, "颜色渐变节点里要有一个色带控件"
noise_item = shader.items[noise.uid]
assert "vector" in noise_item.inputs and "fac" in noise_item.outputs
shader.frame_all()
d.settle()
d.shot("shader_01_noise_ramp.png", widget=shader.view)
pixels = app.engine.renderer.read_pixels(v3d.view)[:, :, :3].astype(np.float64)
after_mean = pixels.mean(axis=(0, 1))
results["viewport_mean"] = [list(map(float, before_mean)), list(map(float, after_mean))]
assert np.abs(after_mean - before_mean).max() > 3.0, "接上噪波色带之后视口要变"
d.shot("shader_02_viewport.png", widget=v3d.widget)
# ---- 节点里的色带控件改颜色：视口跟着变
widget = ramps[0]
widget.set_active(1)
widget.editingStarted.emit()
widget._set_color((0.9, 0.1, 0.1, 1.0))
widget.editingFinished.emit()
d.settle()
assert abs(ramp_node.values.ramp[1][1][1] - 0.9) < 1e-6, ramp_node.values.ramp
pixels2 = app.engine.renderer.read_pixels(v3d.view)[:, :, :3].astype(np.float64)
results["after_ramp_edit"] = list(map(float, pixels2.mean(axis=(0, 1))))
assert np.abs(pixels2.mean(axis=(0, 1)) - after_mean).max() > 1.0
# Ctrl+Z 撤销这次改色
d.key(shader, "Z", ctrl=True)
d.settle()
assert abs(ramp_node.values.ramp[1][1][1] - 0.95) < 1e-6, ramp_node.values.ramp
# ---- 选中噪波节点：M 静音、H 折叠、X 删除再撤销
shader.select_only([noise.uid])
graph.set_active(noise.uid)
center = shader.view.mapFromScene(noise_item.mapToScene(noise_item.boundingRect().center()))
d.move(shader, center.x(), center.y())
d.key(shader, "M")
d.settle()
assert noise.mute
d.key(shader, "M")
d.key(shader, "H")
d.settle()
assert noise.hide and shader.items[noise.uid].height() < 40
d.shot("shader_03_collapsed.png", widget=shader.view)
d.key(shader, "H")
d.settle()
assert not noise.hide
count = len(graph.nodes)
d.key(shader, "X")
d.settle()
assert noise.uid not in graph.nodes and len(graph.nodes) == count - 1
d.key(shader, "Z", ctrl=True)
d.settle()
assert noise.uid in graph.nodes, "撤销删除要把节点和线都放回来"
assert graph.input_link(ramp_node.uid, "fac") == (noise.uid, "fac")
# ---- Shift+A 菜单在内存里建（不弹出），分类齐全
from splender.ui.widgets import build_menu  # noqa: E402

menu = build_menu("SHADER_MT_add", shader.context())
names = [a.text().split(chr(9))[0] for a in menu.actions() if a.text()]
results["add_menu"] = names
assert {"输入", "纹理", "颜色", "转换器", "输出"} <= set(names), names
menu.deleteLater()
# ---- 存盘再打开：材质节点还在
import tempfile, os  # noqa: E401,E402

path = os.path.join(tempfile.mkdtemp(), "shader_selftest.splender")
app.save_project(path)
d.settle()
saved_nodes = len(graph.nodes)
from splender.engine.projectio import load_project  # noqa: E402

project2, _meta = load_project(app.engine, path)
ts2 = next(t for t in project2.texture_sets if t.uid == ts.uid)
results["reloaded_nodes"] = len(ts2.material.nodes) if ts2.material is not None else 0
assert ts2.material is not None and len(ts2.material.nodes) == saved_nodes
(d.out / "shader_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                           encoding="utf-8")
