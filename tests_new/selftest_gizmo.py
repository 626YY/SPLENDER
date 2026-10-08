"""整机自检：移动、旋转、缩放工具的操纵杆。物体模式里拖 X 箭头只沿 X 移动、拖 Z 环绕 Z 旋转、拖中间圈整体缩放；
编辑模式里拖 Z 箭头只动选中的点；变换进行中操纵杆不画。变量 app 和 d 由自检框架提供。"""
import json

import numpy as np

from splender.core.keymap import LEFTMOUSE, MOUSEMOVE, MOVE, PRESS, RELEASE
from splender.doc.objects import to_display

results = {}
d.settle()
app.window.set_workspace("布局")
d.settle()
v3d = d.editor("VIEW_3D")
project = app.project
project.cursor_location = (0.0, 0.0, 0.0)
for obj in list(project.all_objects()):
    obj.select = False
assert d.call("mesh.primitive_cube_add", v3d) == "FINISHED"
d.settle()
cube = project.active_object
for obj in project.all_objects():
    obj.visible = obj is cube
d.move(v3d, 300, 250)
d.key(v3d, "NUMPAD_1")
d.key(v3d, "NUMPAD_PERIOD")
d.settle()
gizmo = v3d.transform_gizmo


def drag(start, end, steps=8):
    d.move(v3d, start.x(), start.y())
    d.event(v3d, LEFTMOUSE, PRESS, start.x(), start.y())
    for t in np.linspace(0.0, 1.0, steps):
        d.move(v3d, start.x() + (end.x() - start.x()) * t, start.y() + (end.y() - start.y()) * t)
    d.event(v3d, LEFTMOUSE, RELEASE, end.x(), end.y())
    d.settle()


# ---- 移动工具：拖 X 箭头
app.tool_settings.tool = "object.move"
d.settle()
geo = gizmo.geometry()
assert geo is not None and geo["kind"] == "TRANSLATE", geo
results["axes"] = [(k, (s.x(), s.y()), (t.x(), t.y())) for k, s, t in geo["axes"]]
x_axis = next(item for item in geo["axes"] if item[0] == 0)
tip = x_axis[2]
grab = x_axis[1] + (tip - x_axis[1]) * 0.7
assert gizmo.hit(grab.x(), grab.y()) == ("axis", 0), gizmo.hit(grab.x(), grab.y())
d.shot("gizmo_00_move.png", widget=v3d.widget)
before = np.array(cube.transform.location)
from PySide6.QtCore import QPointF  # noqa: E402

drag(grab, grab + QPointF(80.0, -60.0))
after = np.array(cube.transform.location)
results["move_x"] = [list(before), list(after)]
delta = after - before
assert delta[0] > 0.1 and abs(delta[1]) < 1e-6 and abs(delta[2]) < 1e-6, delta
# ---- 旋转工具：拖 Z 环（前视图里 Z 环是一条横线，拖白色视线环）
app.tool_settings.tool = "object.rotate"
d.settle()
geo = gizmo.geometry()
assert geo["kind"] == "ROTATE" and len(geo["rings"]) == 3
center = geo["center"]
from splender.ui import theme  # noqa: E402

r = float(theme.px(92))
start = QPointF(center.x() + r, center.y())
assert gizmo.hit(start.x(), start.y()) == ("view",), gizmo.hit(start.x(), start.y())
d.shot("gizmo_01_rotate.png", widget=v3d.widget)
rot_before = np.array(cube.transform.rotation)
drag(start, QPointF(center.x(), center.y() - r))
rot_after = np.array(cube.transform.rotation)
results["rotate_view"] = [list(rot_before), list(rot_after)]
assert np.abs(rot_after - rot_before).max() > 30.0, (rot_before, rot_after)
# ---- 缩放工具：拖中间的圈整体放大
app.tool_settings.tool = "object.scale"
d.settle()
geo = gizmo.geometry()
center = geo["center"]
scale_before = np.array(cube.transform.scale)
start = QPointF(center.x() + 6, center.y())
assert gizmo.hit(start.x(), start.y()) == ("center",)
drag(start, QPointF(center.x() + 30, center.y()))
scale_after = np.array(cube.transform.scale)
results["scale"] = [list(scale_before), list(scale_after)]
assert np.allclose(scale_after / scale_before, scale_after[0] / scale_before[0], atol=1e-4)
assert scale_after[0] > scale_before[0] * 1.5
d.shot("gizmo_02_scale.png", widget=v3d.widget)
# ---- 编辑模式：选一个面，拖 Z 箭头
d.key(v3d, "TAB")
d.settle()
assert app.tool_settings.mode == "EDIT"
app.tool_settings.tool = "mesh.move"
d.key(v3d, "3")
mesh = app.engine.edit.mesh
fn = mesh.face_normals()
from splender.doc.objects import to_internal  # noqa: E402

top = int(np.argmax(fn @ to_internal(np.array([0.0, 0.0, 1.0]))))
mesh.select_all("DESELECT")
mesh.face_sel[top] = True
mesh.flush_from_faces()
app.engine.edit.update_overlay()
d.settle()
geo = gizmo.geometry()
z_axis = next(item for item in geo["axes"] if item[0] == 2)
grab = z_axis[1] + (z_axis[2] - z_axis[1]) * 0.7
assert gizmo.hit(grab.x(), grab.y()) == ("axis", 2), gizmo.hit(grab.x(), grab.y())
verts_before = to_display(mesh.verts.T).T.copy()
d.shot("gizmo_03_edit.png", widget=v3d.widget)
drag(grab, grab + QPointF(0.0, -50.0))
mesh = app.engine.edit.mesh
verts_after = to_display(mesh.verts.T).T
moved = np.abs(verts_after - verts_before).max(axis=1) > 1e-6
results["edit_moved"] = int(moved.sum())
assert moved.sum() == 4, moved.sum()
dz = (verts_after - verts_before)[moved]
assert np.allclose(dz[:, :2], 0.0, atol=1e-6) and (dz[:, 2] > 0.05).all(), dz
d.key(v3d, "TAB")
d.settle()
(d.out / "gizmo_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
