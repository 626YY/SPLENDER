"""整机自检：雕刻。切到雕刻工作区、几种笔刷、撤销重做、细分、回到绘制模式在雕过的形状上画、撤销整段雕刻。
变量 app 和 d 由自检框架提供。"""
import json
import math
import time

import numpy as np

app.window.set_workspace("绘制")       # 这个自检从绘制工作区（纹理绘制模式）开始
results = {}
d.settle()
d.out.mkdir(parents=True, exist_ok=True)
obj = app.project.objects[0]
original = obj.data.positions.copy()


def green_pixels(editor, name="") -> int:
    app.host.make_current()
    rgb = app.engine.renderer.read_pixels(editor.view)[:, :, :3].astype(np.int32)
    if name:
        from PIL import Image
        Image.fromarray(rgb.astype(np.uint8)).save(d.out / name)
    return int(((rgb[:, :, 1] > 120) & (rgb[:, :, 0] < 90) & (rgb[:, :, 2] < 110)).sum())


# 先在绘制模式里画一道绿色：重构、自动展开 UV 之后它应该跟着模型走
pview = d.editor("VIEW_3D")
app.tool_settings.brush.color = (0.1, 0.8, 0.2)
app.tool_settings.brush.size = 60
pw, ph = pview.widget.width(), pview.widget.height()
d.stroke(pview, [(pw / 2 - 120 + i * 8, ph / 2 + 60) for i in range(31)])
d.settle()
results["green_before"] = green_pixels(pview, "sculpt_00_green_before.png")
assert results["green_before"] > 500, results

# 切到雕刻工作区：自动进入雕刻模式
app.window.set_workspace("雕刻")
d.settle()
d.wait(300)
assert app.tool_settings.mode == "SCULPT", "切到雕刻工作区后应进入雕刻模式"
assert app.engine.sculpt is not None, "应该有雕刻会话"
view = d.editor("VIEW_3D")
w, h = view.widget.width(), view.widget.height()
cx, cy = w / 2, h / 2
results["triangles"] = app.engine.sculpt.base.triangle_count
d.shot("sculpt_01_enter.png")


def stroke(tool, points, **kw):
    app.tool_settings.tool = tool
    d.settle()
    d.stroke(view, points, **kw)
    d.settle()


def wave(offset_y, amp=25, count=40, x0=-140, x1=140):
    return [(cx + x0 + (x1 - x0) * i / (count - 1), cy + offset_y + amp * math.sin(i * 0.35)) for i in range(count)]


app.tool_settings.sculpt.size = 70
stroke("sculpt.draw", wave(-110))
stroke("sculpt.clay", wave(-40))
stroke("sculpt.crease", wave(30, 15))
stroke("sculpt.inflate", [(cx + 150, cy + 100 + i) for i in range(12)])
d.shot("sculpt_02_brushes.png")
steps = len(app.history.steps)
results["history_after_brushes"] = steps

# 撤销、重做最后一笔
d.call("ed.undo")
d.settle()
d.shot("sculpt_03_undo.png")
d.call("ed.redo")
d.settle()

# 抓取
stroke("sculpt.grab", [(cx - 60, cy + 120), (cx - 60, cy + 150), (cx - 70, cy + 190)])
# 遮罩后平滑：被遮罩的区域不受影响
stroke("sculpt.mask", wave(-110, 10, 20))
stroke("sculpt.smooth", wave(-110, 30))
d.call("sculpt.mask_clear")
d.settle()
d.shot("sculpt_04_grab_mask.png")

# 细分一级
before_subdiv = app.engine.sculpt.base.triangle_count
t0 = time.perf_counter()
d.call("sculpt.subdivide")
d.settle()
results["subdivide"] = {"before": before_subdiv, "after": app.engine.sculpt.base.triangle_count,
                        "seconds": round(time.perf_counter() - t0, 2)}
assert results["subdivide"]["after"] == before_subdiv * 4
stroke("sculpt.draw", wave(90, 10), )
d.shot("sculpt_05_subdivided.png")

# 体素重构：均匀、封闭的新网格（没有 UV），重构后接着雕
from splender.sculpt import topology as tp  # noqa: E402

app.tool_settings.remesh.resolution = 128
t0 = time.perf_counter()
d.call("sculpt.voxel_remesh")
d.settle()
session = app.engine.sculpt
mesh_stats = tp.mesh_stats(session.base)
results["remesh"] = {"triangles": session.base.triangle_count, "seconds": round(time.perf_counter() - t0, 2),
                     "boundary": mesh_stats["boundary_edges"], "nonmanifold": mesh_stats["nonmanifold_edges"],
                     "has_uvs": session.base.corner_uvs is not None}
assert mesh_stats["boundary_edges"] == 0 and mesh_stats["nonmanifold_edges"] == 0, results["remesh"]
assert session.base.corner_uvs is None
stroke("sculpt.clay_strips", wave(20, 20))
d.shot("sculpt_05b_remeshed.png")
d.call("ed.undo")
d.settle()
d.call("ed.undo")
d.settle()
results["remesh_undo_triangles"] = app.engine.sculpt.base.triangle_count
assert results["remesh_undo_triangles"] == results["subdivide"]["after"], "撤销重构应回到细分后的网格"
d.call("ed.redo")
d.settle()
d.call("ed.redo")
d.settle()
assert app.engine.sculpt.base.triangle_count == results["remesh"]["triangles"]

# 回到绘制模式：形状写回，能在上面画
app.window.set_workspace("绘制")
d.settle()
assert app.tool_settings.mode == "PAINT" and app.engine.sculpt is None
results["paint_triangles"] = obj.data.triangle_count
d.settle()
results["green_after_remesh"] = green_pixels(d.editor("VIEW_3D"), "sculpt_06a_green_after_remesh.png")
results["auto_unwrapped"] = bool(getattr(obj.data, "has_uvs", False))
assert results["auto_unwrapped"], "重构过的模型回到绘制模式应自动展开 UV"
results["shape_changed"] = bool(len(obj.data.positions) != len(original) or not np.allclose(obj.data.positions[:10], original[:10]))
app.tool_settings.brush.color = (0.9, 0.3, 0.1)
app.tool_settings.brush.size = 50
d.stroke(view if view.widget.isVisible() else d.editor("VIEW_3D"), wave(0, 30))
d.settle()
d.shot("sculpt_06_paint_on_sculpt.png")

# 撤销：先撤掉刚画的，再撤掉整段雕刻
labels = [s.label for s in app.history.steps]
results["history_labels_tail"] = labels[-4:]
assert labels[-1] == "笔划" and labels[-2] == "雕刻", labels[-4:]
d.call("ed.undo")
d.settle()
d.call("ed.undo")
d.settle()
results["restored_original"] = bool(len(obj.data.positions) == len(original) and np.allclose(obj.data.positions, original))
d.settle()
results["green_after_undo"] = green_pixels(d.editor("VIEW_3D"), "sculpt_08_green_after_undo.png")
d.shot("sculpt_07_undo_sculpt.png")
(d.out / "sculpt_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert results["shape_changed"], "回到绘制模式后形状应已改变"
assert results["restored_original"], "撤销整段雕刻后应回到原来的形状"
assert results["green_after_remesh"] > results["green_before"] * 0.5, "重构后画好的内容应跟着搬过去：%s" % results
# 笔刷光标的圆圈会挡住一部分，所以留些余地
assert abs(results["green_after_undo"] - results["green_before"]) < results["green_before"] * 0.25, results
