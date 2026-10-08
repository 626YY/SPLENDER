"""整机自检：启动、画一笔、撤销重做、图层操作、切工作区、拆分区域。变量 app 和 d 由自检框架提供。"""
import json
import math
import time

app.window.set_workspace("绘制")       # 这个自检从绘制工作区（纹理绘制模式）开始
results = {}
d.settle()
d.shot("01_startup.png")
view = d.editor("VIEW_3D")
assert view is not None and view.view is not None, "没有 3D 视口"
w, h = view.widget.width(), view.widget.height()
cx, cy = w / 2, h / 2
results["viewport_size"] = [w, h]

# 画一笔（鼠标）
points = [(cx - 160 + i * 6.5, cy - 40 + 45 * math.sin(i * 0.22)) for i in range(50)]
app.tool_settings.brush.size = 46
app.tool_settings.brush.use_height = True
t0 = time.perf_counter()
d.stroke(view, points)
d.settle()
results["stroke_ms"] = round((time.perf_counter() - t0) * 1000)
d.shot("02_after_stroke.png")
results["can_undo_after_stroke"] = app.history.can_undo

# 带压感的一笔，换颜色
app.tool_settings.brush.color = (0.15, 0.5, 0.85)
app.tool_settings.brush.pressure_size = True
points = [(cx - 150 + i * 6, cy + 60 + 20 * math.cos(i * 0.3)) for i in range(50)]
d.stroke(view, points, pressure=lambda t: 0.15 + 0.85 * math.sin(t * math.pi))
d.settle()
d.shot("03_pressure_stroke.png")

# 撤销、重做
d.call("ed.undo")
d.settle()
d.shot("04_undo.png")
d.call("ed.redo")
d.settle()

# 图层：新建填充层、调不透明度、隐藏
d.call("layer.add_fill")
ts = app.project.active_texture_set
ts.active_layer.fill_color = (0.85, 0.65, 0.2)
ts.active_layer.opacity = 0.5
d.settle()
d.shot("05_fill_layer.png")
results["layers"] = [layer.name for layer in ts.layers]
d.call("ed.undo")
d.settle()

# 导航：旋转、缩放
d.stroke(view, [(cx, cy), (cx + 120, cy + 40), (cx + 200, cy + 60)], button="MIDDLEMOUSE")
d.event(view, "WHEELUPMOUSE", "WHEEL", cx, cy, wheel=3.0)
d.settle()
d.shot("06_orbit_zoom.png")
d.call("view3d.view_all", view)
d.settle()

# 通道查看
view.shading.mode = "CHANNEL"
view.shading.channel = "height"
d.settle()
d.shot("07_channel_height.png")
view.shading.mode = "SOLID"
d.settle()
d.shot("08_solid.png")
view.shading.mode = "MATERIAL"

# 拆分区域、换编辑器
screen = d.screen()
area = view.area
new_area = screen.split_area(area, "H", 0.5)
new_area.set_editor("IMAGE_EDITOR")
d.settle()
d.shot("09_split_with_uv.png")
screen.join_areas(area, new_area)
d.settle()

# 切工作区
app.window.set_workspace("UV")
d.settle()
d.shot("10_workspace_uv.png")
app.window.set_workspace("材质")
d.settle()
d.wait(700)
d.shot("11_workspace_material.png")
app.window.set_workspace("绘制")
d.settle()

stats = app.engine.stats()
results["engine"] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()}
results["present_ms"] = [round(v, 1) for v in list(app.host.present_ms)[-40:]]
d.out.mkdir(parents=True, exist_ok=True)
(d.out / "basic_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
