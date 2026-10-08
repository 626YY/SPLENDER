"""整机自检：图层文件夹和调整层。画一笔 → 建文件夹 → 拖进去 → 文件夹半透明、隐藏 → 调整层改色 →
解散、撤销 → 删除文件夹、撤销（像素回来）→ 复制文件夹 → 智能材质成组 → 折叠。变量 app 和 d 由自检框架提供。"""
import json
import logging
import math

import numpy as np

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
view = d.editor("VIEW_3D")
ts = app.project.active_texture_set
w, h = view.widget.width(), view.widget.height()
cx, cy = w / 2, h / 2


def pixels():
    app.host.make_current()
    return app.engine.renderer.read_pixels(view.view)[:, :, :3].astype(np.float32)


def red_amount(image) -> float:
    """偏红的程度：红减去绿蓝平均，只看偏红的像素。"""
    redness = image[:, :, 0] - (image[:, :, 1] + image[:, :, 2]) * 0.5
    return float(np.clip(redness - 40.0, 0, None).sum())          # 环境光的暖色不算


def green_amount(image) -> float:
    greenness = image[:, :, 1] - (image[:, :, 0] + image[:, :, 2]) * 0.5
    return float(np.clip(greenness - 40.0, 0, None).sum())


paint_layer = next(layer for layer in ts.layers if layer.kind == "PAINT")
ts.set_active(paint_layer)
app.tool_settings.brush.color = (0.95, 0.1, 0.1)
app.tool_settings.brush.size = 70
d.stroke(view, [(cx - 130 + i * 9, cy + 20 * math.sin(i * 0.4)) for i in range(30)])
d.settle()
base_red = red_amount(pixels())
results["red_painted"] = base_red
assert base_red > 1000, results

# 建文件夹，把绘制层拖进去
assert d.call("layer.add_folder") == "FINISHED"
folder = ts.active_layer
assert folder.kind == "FOLDER"
assert d.call("layer.move_to", uid=paint_layer.uid, index=ts.index_of(folder), parent=folder.uid) == "FINISHED"
assert paint_layer.parent_uid == folder.uid and ts.depth(paint_layer) == 1
d.settle()
results["red_in_folder"] = red_amount(pixels())
assert abs(results["red_in_folder"] - base_red) < base_red * 0.05, "放进文件夹不应改变画面：%s" % results

folder.opacity = 0.5
d.settle()
results["red_half"] = red_amount(pixels())
assert results["red_half"] < base_red * 0.75, results
d.shot("folders_01_half.png")
folder.opacity = 1.0
folder.visible = False
d.settle()
hidden_image = pixels()
results["red_hidden"] = red_amount(hidden_image)
from PIL import Image  # noqa: E402
Image.fromarray(hidden_image.astype(np.uint8)).save(d.out / "folders_00_hidden.png")
assert results["red_hidden"] < base_red * 0.05, "隐藏文件夹应连同里面的图层一起隐藏：%s" % results
folder.visible = True
d.settle()

# 调整层：色相转 120 度，红变绿
ts.set_active(folder)
assert d.call("layer.add_adjust", adjust="HSV") == "FINISHED"
adjust = ts.active_layer
adjust.adj_hue = 120.0
d.settle()
image = pixels()
results["green_after_hue"] = green_amount(image)
results["red_after_hue"] = red_amount(image)
assert results["green_after_hue"] > base_red * 0.3 and results["red_after_hue"] < base_red * 0.3, results
d.shot("folders_02_hue.png")
d.call("ed.undo")
d.settle()
assert adjust not in ts.layers

# 解散文件夹再撤销
ts.set_active(folder)
assert d.call("layer.ungroup") == "FINISHED"
assert folder not in ts.layers and paint_layer.parent_uid == 0
d.call("ed.undo")
d.settle()
assert folder in ts.layers and paint_layer.parent_uid == folder.uid

# 删除文件夹（连同里面的层）再撤销：像素要回来
ts.set_active(folder)
assert d.call("layer.delete") == "FINISHED"
assert paint_layer not in ts.layers
d.settle()
results["red_deleted"] = red_amount(pixels())
d.call("ed.undo")
d.settle()
results["red_restored"] = red_amount(pixels())
assert results["red_deleted"] < base_red * 0.05 and abs(results["red_restored"] - base_red) < base_red * 0.05, results

# 复制文件夹：连同里面的层和像素
ts.set_active(folder)
count = len(ts.layers)
assert d.call("layer.duplicate") == "FINISHED"
assert len(ts.layers) == count + 2
copy = ts.active_layer
assert copy.kind == "FOLDER" and len(ts.children(copy)) == 1
copy_child = ts.children(copy)[0]
assert app.engine.layers.stores.get(copy_child.uid) is not None, "复制的图层应带着像素"
d.call("ed.undo")
d.settle()
assert len(ts.layers) == count

# 智能材质成组
assert d.call("layer.add_smart_material", material="lacquer_gold") == "FINISHED"
smart_folder = ts.active_layer
results["smart_folder"] = [smart_folder.name, smart_folder.kind, [x.name for x in ts.children(smart_folder)]]
assert smart_folder.kind == "FOLDER" and len(ts.children(smart_folder)) == 3
for _ in range(400):
    if app.engine.bake is None:
        break
    d.wait(50)
d.settle()
layers_editor = d.editor("LAYERS")
props = d.editor("PROPERTIES")
if props is not None:
    props.tabs.setCurrent("LAYER")
    props._on_tab("LAYER")
d.settle()
d.shot("folders_03_smart_group.png")
if layers_editor is not None:
    layers_editor.collapsed.add(smart_folder.uid)
    layers_editor.refresh()
    d.settle()
    d.shot("folders_04_collapsed.png")
    rows = layers_editor.list.rows()
    assert all(ts.parent_of(layer) is not smart_folder for layer in rows), "折叠后里面的图层不显示"

results["history_tail"] = [step.label for step in app.history.steps][-6:]
results["errors"] = errors
(d.out / "folders_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
