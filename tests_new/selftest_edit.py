"""整机自检：Blender 式编辑模式。「布局」工作区里加一个立方体，Tab 进编辑模式：工具栏换成编辑工具、标题栏有
选择/添加/网格/点/边/面/UV 菜单和点边面按钮、键位换成 Mesh 表；1/2/3 换选择模式、A 全选、点选、G Z 1、
E 挤出、I 内插（拖动）、Ctrl+R 环切、K 切刀、右键菜单按选择模式换内容、Ctrl+Z 一步步撤回；Tab 退出时形状写回物体，
整个编辑过程是一步撤销。弹出菜单只在内存里建（不弹到屏幕上）。变量 app 和 d 由自检框架提供。"""
import json

import numpy as np

from splender.core.keymap import LEFTMOUSE, PRESS, RELEASE
from splender.doc.objects import to_display, to_internal

results = {}
d.settle()
win = app.window
win.set_workspace("布局")
d.settle()
assert app.tool_settings.mode == "OBJECT", app.tool_settings.mode
v3d = d.editor("VIEW_3D")
project = app.project
view = v3d.view
ratio = v3d.widget.devicePixelRatioF()


def screen_of(point):
    p = to_internal(np.asarray(point, np.float64))
    clip = view.camera.view_projection(view.aspect) @ np.array([p[0], p[1], p[2], 1.0])
    px = (clip[0] / clip[3] * 0.5 + 0.5) * view.width
    py = (0.5 - clip[1] / clip[3] * 0.5) * view.height
    return px / ratio, py / ratio


def counts():
    mesh = app.engine.edit.mesh
    return mesh.vert_count, mesh.face_count


def click(x, y, **mods):
    d.move(v3d, x, y)
    d.event(v3d, LEFTMOUSE, PRESS, x, y, **mods)
    d.event(v3d, LEFTMOUSE, RELEASE, x, y, **mods)
    d.settle()


# 只留一个立方体，前视图正交
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
shape_before = cube.data.positions.copy()
steps_before = len(app.history.steps)
# ---- Tab 进编辑模式
d.key(v3d, "TAB")
d.settle()
results["mode"] = app.tool_settings.mode
assert app.tool_settings.mode == "EDIT" and app.engine.edit is not None and app.engine.edit.obj is cube
results["keymaps"] = v3d.keymap_names()
assert v3d.keymap_names()[:2] == ["3D View Tool: Select Box", "Mesh"], v3d.keymap_names()
toolbar = v3d.toolbar
results["tools"] = list(toolbar._buttons)
for name in ("mesh.select_box", "mesh.tool_extrude", "mesh.tool_inset", "mesh.tool_bevel", "mesh.tool_loopcut",
             "mesh.tool_knife"):
    assert name in toolbar._buttons, (name, list(toolbar._buttons))
assert counts() == (8, 6), counts()
d.shot("edit_00_entered.png")
# ---- 1/2/3 选择模式，A 全选
d.key(v3d, "2")
assert app.engine.edit.mesh.select_mode == {"EDGE"}
d.key(v3d, "3")
assert app.engine.edit.mesh.select_mode == {"FACE"}
d.key(v3d, "1", shift=True)
assert app.engine.edit.mesh.select_mode == {"FACE", "VERT"}
d.key(v3d, "1")
assert app.engine.edit.mesh.select_mode == {"VERT"}
d.key(v3d, "A")
d.settle()
assert app.engine.edit.mesh.vert_sel.all()
d.key(v3d, "A", alt=True)
assert not app.engine.edit.mesh.vert_sel.any()
# ---- 点选前面右上角的点，G Z 1
mesh = app.engine.edit.mesh
disp = to_display(mesh.verts.T).T
corner = int(np.argmin(np.linalg.norm(disp - np.array([1.0, -1.0, 1.0]), axis=1)))
x, y = screen_of(disp[corner])
click(x, y)
results["click_vertex"] = [int(v) for v in np.nonzero(mesh.vert_sel)[0]]
assert results["click_vertex"] == [corner], (results["click_vertex"], corner)
d.key(v3d, "G")
d.key(v3d, "Z")
d.key(v3d, "1")
d.key(v3d, "RET")
d.settle()
mesh = app.engine.edit.mesh
moved = to_display(mesh.verts[corner])
results["moved_vertex"] = [float(v) for v in moved]
assert np.allclose(moved, [1.0, -1.0, 2.0], atol=1e-5), moved
d.key(v3d, "Z", ctrl=True)
d.settle()
assert np.allclose(to_display(app.engine.edit.mesh.verts[corner]), [1.0, -1.0, 1.0], atol=1e-5)
# ---- 面模式点前面那个面，E 1 回车：沿法线挤出 1
d.key(v3d, "3")
fx, fy = screen_of((0.0, -1.0, 0.0))
click(fx, fy)
mesh = app.engine.edit.mesh
assert int(mesh.face_sel.sum()) == 1, int(mesh.face_sel.sum())
d.key(v3d, "E")
d.key(v3d, "1")
d.key(v3d, "RET")
d.settle()
results["after_extrude"] = counts()
assert counts() == (12, 10), counts()
mesh = app.engine.edit.mesh
front = to_display(mesh.verts[mesh.vert_sel].T).T
assert np.allclose(front[:, 1], -2.0, atol=1e-5), front
d.shot("edit_01_extruded.png", widget=v3d)
# ---- I 拖动内插，左键确认
d.move(v3d, fx, fy)
d.key(v3d, "I")
for t in np.linspace(0.0, 1.0, 6):
    d.move(v3d, fx + 30 + 40 * t, fy)
d.event(v3d, LEFTMOUSE, PRESS, fx + 70, fy)
d.settle()
results["after_inset"] = counts()
assert counts() == (16, 14), counts()
# ---- Ctrl+R 环切：指着侧面的一条竖边，点下后右键放在正中
ex, ey = screen_of((1.0, -0.5, 0.0))
d.move(v3d, ex, ey)
d.key(v3d, "R", ctrl=True)
d.move(v3d, ex, ey)
d.event(v3d, LEFTMOUSE, PRESS, ex, ey)
d.event(v3d, "RIGHTMOUSE", PRESS, ex, ey)
d.settle()
results["after_loopcut"] = counts()
assert counts()[0] > 16, counts()
d.shot("edit_02_loopcut.png", widget=v3d)
# ---- K 切刀：在内插出来的中间那个面上横着切一刀
d.key(v3d, "1")
mesh = app.engine.edit.mesh
ax, ay = screen_of((-0.9, -2.0, 0.1))
bx, by = screen_of((0.9, -2.0, 0.1))
v_before = counts()[0]
d.move(v3d, ax, ay)
d.key(v3d, "K")
d.event(v3d, LEFTMOUSE, PRESS, ax, ay)
d.move(v3d, bx, by)
d.event(v3d, LEFTMOUSE, PRESS, bx, by)
d.shot("edit_03_knife.png", widget=v3d)
d.key(v3d, "RET")
d.settle()
results["after_knife"] = counts()
assert counts()[0] >= v_before + 2, (counts(), v_before)
# ---- 右键菜单按选择模式换内容（只在内存里建，不弹出来）
from splender.ui.widgets import build_menu  # noqa: E402

menus = {}
for kind in ("1", "2", "3"):
    d.key(v3d, kind)
    menu = build_menu("VIEW3D_MT_edit_mesh_context_menu", v3d.context())
    menus[kind] = [a.text().split(chr(9))[0] for a in menu.actions() if a.text()]
    if kind == "3":
        menu.adjustSize()
        menu.grab().save(str(d.out / "edit_04_context_menu.png"))
    menu.deleteLater()
results["context_menus"] = menus
assert "戳面" in menus["3"] and "桥接循环边" in menus["2"] and "撕开" in menus["1"], menus
for name in ("VIEW3D_MT_select_edit_mesh", "VIEW3D_MT_edit_mesh", "VIEW3D_MT_edit_mesh_vertices",
             "VIEW3D_MT_edit_mesh_edges", "VIEW3D_MT_edit_mesh_faces", "VIEW3D_MT_uv_map",
             "VIEW3D_MT_edit_mesh_delete", "VIEW3D_MT_edit_mesh_merge", "VIEW3D_MT_mesh_add"):
    menu = build_menu(name, v3d.context())
    assert len([a for a in menu.actions() if a.text()]) >= 3, name
    menu.deleteLater()
# ---- Shift+A 加一个立方体到正在编辑的网格里
v_now, f_now = counts()
project.cursor_location = (4.0, 0.0, 0.0)
assert d.call("mesh.primitive_cube_add", v3d) == "FINISHED"
d.settle()
assert counts() == (v_now + 8, f_now + 6), counts()
d.shot("edit_05_added_cube.png", widget=v3d)
# ---- Tab 退出：形状写回物体，整个编辑过程合成一步撤销
d.key(v3d, "TAB")
d.settle()
assert app.tool_settings.mode == "OBJECT" and app.engine.edit is None
results["steps"] = [len(app.history.steps), steps_before]
assert len(app.history.steps) == steps_before + 1, (len(app.history.steps), steps_before)
assert app.history.steps[-1].label == "编辑网格", app.history.steps[-1].label
results["triangles_after"] = int(len(cube.data.positions) // 3)
assert len(cube.data.positions) > len(shape_before)
d.shot("edit_06_object_mode.png", widget=v3d)
d.key(v3d, "Z", ctrl=True)
d.settle()
assert np.allclose(cube.data.positions, shape_before, atol=1e-5), "撤销编辑网格这一步要回到进编辑模式前的形状"
(d.out / "edit_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                         encoding="utf-8")
