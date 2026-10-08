"""整机自检：Blender 式物体模式。「布局」工作区（物体模式）里：添加立方体、点选、拖动框选、G 移动（输入数值）、
「调整上一步」面板改尺寸、大纲视图、属性编辑器的物体页、按模式换的键位、线框和 X 光、摄像机视图、局部视图、
模式切换（物体 → 雕刻 → 绘制 → 物体）。变量 app 和 d 由自检框架提供。"""
import json

import numpy as np

from splender.core.keymap import LEFTMOUSE, MOUSEMOVE, MOVE, PRESS, RELEASE
from splender.doc.objects import to_internal

results = {}
d.settle()
win = app.window
names = win.workspace_names()
results["workspaces"] = names
assert names[0] == "布局", names
win.set_workspace("布局")
d.settle()
assert app.tool_settings.mode == "OBJECT", app.tool_settings.mode
assert app.tool_settings.tool.startswith("object."), app.tool_settings.tool
v3d = d.editor("VIEW_3D")
outliner = d.editor("OUTLINER")
props = d.editor("PROPERTIES")
assert v3d is not None and outliner is not None and props is not None
results["keymaps"] = v3d.keymap_names()
assert v3d.keymap_names()[:3] == ["3D View Tool: Select Box", "Object Mode", "Object Non-modal"], v3d.keymap_names()
project = app.project
view = v3d.view
ratio = v3d.widget.devicePixelRatioF()


def screen_of(point):
    p = to_internal(np.asarray(point, np.float64))
    clip = view.camera.view_projection(view.aspect) @ np.array([p[0], p[1], p[2], 1.0])
    px = (clip[0] / clip[3] * 0.5 + 0.5) * view.width
    py = (0.5 - clip[1] / clip[3] * 0.5) * view.height
    return px / ratio, py / ratio                 # 事件坐标是逻辑像素


# 前视图正交，好算位置
d.move(v3d, 200, 200)
d.key(v3d, "NUMPAD_1")
d.settle()
results["axis_view"] = view.camera.axis_view
# ---- Shift+A → 立方体（菜单本身靠人点，这里直接调同一个操作），放在游标处
project.cursor_location = (3.0, 0.0, 0.0)
assert d.call("mesh.primitive_cube_add", v3d) == "FINISHED"
d.settle()
cube = project.objects[-1]
results["after_add"] = [o.name for o in project.all_objects()]
assert cube.select and project.active_object is cube
# 「调整上一步」面板出现了，改尺寸就重做
panel = v3d.redo_panel
results["redo_panel_visible"] = bool(panel.isVisible())
assert panel.isVisible(), "加完立方体左下角应该出现调整面板"
op = panel._op
op.size = 1.0
d.wait(50)
d.settle()
cube = project.objects[-1]
lo = np.asarray(cube.data.bounds_min)
hi = np.asarray(cube.data.bounds_max)
results["cube_size_after_adjust"] = [float(v) for v in (hi - lo)]
assert np.allclose(hi - lo, 1.0, atol=1e-4), hi - lo
assert len(project.objects) == 2, "调整只是重做那一步，不该多出物体"
d.shot("objects_00_added.png")
# ---- 点选：点在球上
sphere = project.objects[0]
d.wait(100)
d.settle()
sx, sy = screen_of((0.0, 0.0, 0.0))
d.move(v3d, sx, sy)
d.event(v3d, LEFTMOUSE, PRESS, sx, sy)
d.event(v3d, LEFTMOUSE, RELEASE, sx, sy)
d.settle()
results["click_select"] = [o.name for o in project.selected_objects()]
assert project.active_object is sphere and sphere.select and not cube.select, results["click_select"]
# ---- 拖动框选两个（工具是框选：按下拖动产生 CLICK_DRAG）
x0, y0 = screen_of((-1.6, 0.0, 1.6))
x1, y1 = screen_of((4.0, 0.0, -1.6))
d.move(v3d, x0, y0)
d.event(v3d, LEFTMOUSE, PRESS, x0, y0)
for t in np.linspace(0.0, 1.0, 8):
    d.move(v3d, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
d.shot("objects_01_box_select.png")
d.event(v3d, LEFTMOUSE, RELEASE, x1, y1)
d.settle()
results["box_select"] = sorted(o.name for o in project.selected_objects())
assert cube.select and sphere.select, results["box_select"]
# ---- G X 1 回车（键盘走真实的键位表）
before = [tuple(o.transform.location) for o in (sphere, cube)]
d.move(v3d, x1, y1)
d.key(v3d, "G")
d.key(v3d, "X")
d.key(v3d, "1")
d.key(v3d, "RET")
d.settle()
after = [tuple(o.transform.location) for o in (sphere, cube)]
results["move_x1"] = [before, after]
assert np.allclose(np.subtract(after, before), [[1, 0, 0], [1, 0, 0]], atol=1e-5), results["move_x1"]
# Ctrl+Z 撤销移动
d.key(v3d, "Z", ctrl=True)
d.settle()
assert np.allclose([tuple(o.transform.location) for o in (sphere, cube)], before, atol=1e-5)
# ---- 大纲视图：两行物体，点立方体那行只选它
outliner.refresh()
rows = outliner.tree.rows()
results["outliner_rows"] = [r.label for r in rows]
assert any(r.label == cube.name for r in rows)
index = next(i for i, r in enumerate(rows) if r.kind == "OBJECT" and r.obj is cube)
rect = outliner.tree.row_rect(index)
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtCore import QEvent  # noqa: E402

pos = QPointF(rect.center().x(), rect.center().y())
press = QMouseEvent(QEvent.MouseButtonPress, pos, outliner.tree.mapToGlobal(pos.toPoint()), Qt.LeftButton,
                    Qt.LeftButton, Qt.NoModifier)
outliner.tree.mousePressEvent(press)
d.settle()
results["outliner_click"] = [o.name for o in project.selected_objects()]
assert project.active_object is cube and not sphere.select, results["outliner_click"]
# ---- 属性编辑器：物体页显示当前物体的名字和变换
props.tabs.setCurrent("OBJECT")
props._on_tab("OBJECT")
d.settle()
cube.transform.location = (4.0, 0.0, 0.5)
d.settle()
d.shot("objects_02_properties.png", widget=props)
results["props_tab"] = props._tab
# ---- 线框、X 光
d.move(v3d, 200, 200)
d.key(v3d, "Z", shift=True)
d.settle()
results["wireframe"] = v3d.shading.mode
assert v3d.shading.mode == "WIREFRAME"
d.shot("objects_03_wireframe.png", widget=v3d)
d.key(v3d, "Z", shift=True)
d.key(v3d, "Z", alt=True)
d.settle()
assert v3d.shading.show_xray
d.shot("objects_04_xray.png", widget=v3d)
d.key(v3d, "Z", alt=True)
# ---- 摄像机：加一台，Ctrl+Alt+小键盘0 对齐视图，小键盘 0 进出摄像机视图
assert d.call("object.camera_add", v3d) == "FINISHED"
d.settle()
camera_obj = project.active_object
d.key(v3d, "NUMPAD_0", ctrl=True, alt=True)
d.settle()
results["camera_view"] = v3d.in_camera_view()
assert v3d.in_camera_view()
d.shot("objects_05_camera_view.png", widget=v3d)
d.key(v3d, "NUMPAD_0")
d.settle()
assert not v3d.in_camera_view()
# ---- 局部视图：只看立方体
for o in project.all_objects():
    o.select = o is cube
project.active_object_uid = cube.uid
project.selection_changed()
d.key(v3d, "SLASH")
d.settle()
results["local_view"] = sorted(v3d.local_view["uids"]) if v3d.local_view else None
assert v3d.local_view is not None
d.shot("objects_06_local_view.png", widget=v3d)
d.key(v3d, "SLASH")
d.settle()
assert v3d.local_view is None
# ---- 模式：物体 → 雕刻（雕当前物体）→ 绘制 → 物体
for o in project.all_objects():
    o.select = o is sphere
project.active_object_uid = sphere.uid
assert d.call("object.mode_set", v3d, mode="SCULPT") == "FINISHED"
d.settle()
results["sculpt_object"] = app.engine.sculpt.obj.name if app.engine.sculpt is not None else None
assert app.engine.sculpt is not None and app.engine.sculpt.obj is sphere
assert "Sculpt" in v3d.keymap_names()[:2] and "Object Mode" not in v3d.keymap_names(), v3d.keymap_names()
assert d.call("object.mode_set", v3d, mode="PAINT") == "FINISHED"
d.settle()
assert app.engine.sculpt is None and app.tool_settings.tool.startswith("paint.")
assert d.call("object.mode_set", v3d, mode="OBJECT") == "FINISHED"
d.settle()
assert app.tool_settings.tool.startswith("object.")
results["previous_mode"] = app.previous_mode
d.shot("objects_07_full.png")
(d.out / "objects_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                            encoding="utf-8")
