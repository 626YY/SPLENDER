"""物体模式的操作，照 Blender：点选、框选、刷选、套索、全选/反选；G/R/S 移动、旋转、缩放（X/Y/Z 约束、坐标系、
轴心点、吸附、输入数值、Shift 精细）；清除、应用变换；Shift+A 添加；Shift+D 复制；X/Delete 删除；H/Shift+H/Alt+H 隐藏；
3D 游标和 Shift+S 吸附；设置原点、合并、镜像、平滑/平直着色。每一步都进撤销历史。

坐标都按 Blender 的 Z 朝上（显示坐标）；网格的顶点存在内部坐标（Y 朝上），换算见 doc.objects。
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..core import ops
from ..core.keymap import LEFTMOUSE, MIDDLEMOUSE, MOUSEMOVE, PRESS, RELEASE, RIGHTMOUSE, WHEEL, WHEELDOWN, WHEELUP
from ..core.ops import CANCELLED, FINISHED, RUNNING_MODAL, Operator
from ..core.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty
from ..doc.objects import (SceneObject, compose, display_to_internal_matrix, euler_from_matrix, rotation_matrix,
                           to_display, to_internal)
from .view3d_ops import view3d

AXIS_KEYS = {"X": 0, "Y": 1, "Z": 2}
DIGIT_KEYS = {**{str(i): str(i) for i in range(10)}, **{"NUMPAD_%d" % i: str(i) for i in range(10)},
              "PERIOD": ".", "NUMPAD_PERIOD": ".", "MINUS": "-", "NUMPAD_MINUS": "-"}
AXIS_COLORS = [(0.9, 0.24, 0.27), (0.48, 0.7, 0.24), (0.27, 0.48, 0.9)]
CONSTRAINT_ITEMS = [("NONE", "无", ""), ("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", ""),
                    ("YZ", "YZ 平面", "除了 X 都能动"), ("XZ", "XZ 平面", "除了 Y 都能动"), ("XY", "XY 平面", "除了 Z 都能动")]
ORIENT_ITEMS = [("GLOBAL", "全局", ""), ("LOCAL", "局部", ""), ("NORMAL", "法向", ""), ("VIEW", "视图", ""),
                ("CURSOR", "游标", "")]
ORIENT_LABELS = {"GLOBAL": "全局", "LOCAL": "局部", "NORMAL": "法向", "VIEW": "视图", "CURSOR": "游标"}


# ====================================================================== 小工具
def _project(ctx):
    return getattr(ctx.app, "project", None)


def _engine(ctx):
    host = getattr(ctx.app, "host", None)
    return host.engine if host is not None else None


def _history(ctx):
    return getattr(ctx.app, "history", None)


def _current(ctx) -> None:
    host = getattr(ctx.app, "host", None)
    if host is not None:
        host.make_current()


def object_mode(ctx) -> bool:
    return ctx.app.tool_settings.mode == "OBJECT" and _project(ctx) is not None


def edit_active(ctx) -> bool:
    """编辑模式且有编辑会话。"""
    engine = _engine(ctx)
    return ctx.app.tool_settings.mode == "EDIT" and engine is not None and getattr(engine, "edit", None) is not None


def edit_has_selection(ctx) -> bool:
    if not edit_active(ctx):
        return False
    mesh = _engine(ctx).edit.mesh
    return bool((mesh.vert_sel & ~mesh.vert_hide).any())


def _hint(ctx, text: str) -> None:
    wm = getattr(ctx, "wm", None)
    if wm is not None:
        wm.set_hint(text)


def _redraw(ctx, tag: str = "selection") -> None:
    engine = _engine(ctx)
    if engine is not None:
        engine.refresh_overlays()
    ctx.app.request_frame()
    ctx.app.notify(tag)


def _meshes(objects) -> list:
    return [o for o in objects if o.kind == "MESH"]


# ====================================================================== 选择（可撤销）
def selection_state(project) -> tuple:
    return (frozenset(o.uid for o in project.all_objects() if getattr(o, "select", False)),
            project.active_object_uid, project.active_set_uid)


def apply_selection(project, state) -> None:
    chosen, active, active_set = state
    for obj in project.all_objects():
        obj.select = obj.uid in chosen
    project.active_object_uid = active
    if active_set and project.texture_set(active_set) is not None and project.active_set_uid != active_set:
        project.active_set_uid = active_set
        project.changed.emit("active_set")
    project.selection_changed()


def record_selection(ctx, label: str, before: tuple) -> None:
    project = _project(ctx)
    after = selection_state(project)
    history = _history(ctx)
    if history is None or after == before:
        return

    def undo() -> None:
        apply_selection(project, before)
        _redraw(ctx)

    def redo() -> None:
        apply_selection(project, after)
        _redraw(ctx)

    history.push(label, undo, redo)


def set_active(project, obj) -> None:
    """设当前物体。模型的话当前纹理集跟着换成它的（和 Blender 当前材质跟着当前物体走一样）。"""
    project.active_object_uid = obj.uid if obj is not None else 0
    if obj is not None and obj.kind == "MESH" and obj.material_sets and project.active_set_uid not in obj.material_sets:
        ts = project.texture_set(obj.material_sets[0])
        if ts is not None:
            project.active_set_uid = ts.uid
            project.changed.emit("active_set")


def select_only(project, objs, active=None) -> None:
    chosen = {o.uid for o in objs}
    for obj in project.all_objects():
        obj.select = obj.uid in chosen
    target = active if active is not None else (objs[-1] if objs else None)
    if target is not None:
        set_active(project, target)
    project.selection_changed()


def apply_select_mode(project, found, mode: str) -> None:
    """按选择方式改选中状态：SET 替换、ADD 加选、SUB 减选、XOR 反转、AND 取交集。"""
    found_ids = {o.uid for o in found}
    for obj in project.all_objects():
        if not obj.visible:
            continue
        inside = obj.uid in found_ids
        if mode == "SET":
            obj.select = inside
        elif mode == "ADD":
            obj.select = obj.select or inside
        elif mode == "SUB":
            obj.select = obj.select and not inside
        elif mode == "XOR":
            obj.select = obj.select != inside
        elif mode == "AND":
            obj.select = obj.select and inside
    active = project.active_object
    if mode in ("SET", "ADD", "XOR") and found and (active is None or not active.select):
        chosen = next((o for o in found if o.select), None)
        if chosen is not None:
            set_active(project, chosen)
    project.selection_changed()


def objects_under(engine, view, x: float, y: float, radius_px: float = 12.0) -> list:
    """鼠标下的全部物体，从前往后：摄像机、灯光、空物体按屏幕距离，模型按视线打到的远近。"""
    project = engine.project
    if project is None:
        return []
    from ..bake.parts import ray_hit

    found = []
    camera = view.camera
    view_proj = camera.view_projection(view.aspect)
    for obj in project.scene_objects:
        if not obj.visible:
            continue
        p = to_internal(np.asarray(obj.transform.location, np.float64))
        clip = view_proj @ np.array([p[0], p[1], p[2], 1.0])
        if clip[3] <= 1e-9:
            continue
        sx = (clip[0] / clip[3] * 0.5 + 0.5) * view.width
        sy = (0.5 - clip[1] / clip[3] * 0.5) * view.height
        if math.hypot(sx - x, sy - y) <= radius_px:
            found.append((float(camera.depth_of(p)), obj))
    origin, direction = camera.ray(x, y, view.width, view.height)
    for obj in project.objects:
        if not obj.visible:
            continue
        lo = np.asarray(obj.data.bounds_min, np.float64)
        hi = np.asarray(obj.data.bounds_max, np.float64)
        if not _ray_box(origin, direction, lo, hi):
            continue
        tri, _u, _v, t = ray_hit(obj.data.positions, origin, direction)
        if tri >= 0:
            found.append((float(t), obj))
    found.sort(key=lambda item: item[0])
    return [obj for _d, obj in found]


def _ray_box(origin, direction, lo, hi) -> bool:
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(direction) < 1e-12, 1e-12, direction)
        t0 = (lo - origin) * inv
        t1 = (hi - origin) * inv
    near = np.max(np.minimum(t0, t1))
    far = np.min(np.maximum(t0, t1))
    return bool(far >= max(near, 0.0))


class _ObjectOp(Operator):
    @classmethod
    def poll(cls, ctx) -> bool:
        return object_mode(ctx)


class _SelectedOp(_ObjectOp):
    @classmethod
    def poll(cls, ctx) -> bool:
        return object_mode(ctx) and bool(_project(ctx).selected_objects())


class _ViewObjectOp(_ObjectOp):
    @classmethod
    def poll(cls, ctx) -> bool:
        return (object_mode(ctx) or edit_active(ctx)) and view3d(ctx) is not None


# ====================================================================== 点选
@ops.register
class View3DSelect(_ViewObjectOp):
    idname = "view3d.select"
    label = "选择"
    description = "点选物体：Shift 加选或减选，Alt 列出鼠标下的所有物体来选，点空白处取消全部选择"
    searchable = False
    deselect_all = BoolProperty("点空白处全不选", default=False)
    toggle = BoolProperty("切换", default=False)
    extend = BoolProperty("加选", default=False)
    deselect = BoolProperty("减选", default=False)
    enumerate = BoolProperty("列出", default=False)

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        project = _project(ctx)
        engine = _engine(ctx)
        if event is None or engine is None:
            return CANCELLED
        if edit_active(ctx):
            from .mesh_edit_ops import edit_click_select

            return edit_click_select(ctx, editor, event, toggle=bool(self.toggle), extend=bool(self.extend),
                                     deselect=bool(self.deselect), deselect_all=bool(self.deselect_all))
        _current(ctx)
        x, y = editor.pixel(event)
        if self.enumerate:
            candidates = objects_under(engine, editor.view, x, y)
            obj = _choose_object(candidates) if len(candidates) > 1 else (candidates[0] if candidates else None)
            if obj is None and candidates:
                return CANCELLED
        else:
            obj = engine.object_at(editor.view, x, y)
        before = selection_state(project)
        if obj is None:
            if not (self.deselect_all and not (self.toggle or self.extend or self.deselect)):
                return CANCELLED
            for item in project.all_objects():
                item.select = False
            project.selection_changed()
        elif self.toggle:
            if obj.select and project.active_object_uid == obj.uid:
                obj.select = False
            elif obj.select:
                set_active(project, obj)
            else:
                obj.select = True
                set_active(project, obj)
            project.selection_changed()
        elif self.extend:
            obj.select = True
            set_active(project, obj)
            project.selection_changed()
        elif self.deselect:
            obj.select = False
            project.selection_changed()
        else:
            select_only(project, [obj], obj)
        record_selection(ctx, "选择", before)
        _redraw(ctx)
        return FINISHED


def _choose_object(candidates):
    """Alt+点击：鼠标下有好几个物体时弹出列表让人挑。"""
    from PySide6.QtGui import QCursor
    from PySide6.QtWidgets import QMenu

    from ..ui import icons

    menu = QMenu()
    actions = {}
    for obj in candidates:
        action = menu.addAction(icons.icon(object_icon(obj)), obj.name)
        actions[action] = obj
    chosen = menu.exec(QCursor.pos())
    return actions.get(chosen)


def object_icon(obj) -> str:
    return {"MESH": "object.mesh", "CAMERA": "object.camera", "LIGHT": "object.light", "EMPTY": "object.empty"}.get(
        obj.kind, "object.mesh")


@ops.register
class ObjectSelectAll(_ObjectOp):
    idname = "object.select_all"
    label = "全选"
    description = "选择、取消选择或反选所有可见的物体"
    action = EnumProperty("动作", items=[("TOGGLE", "切换", ""), ("SELECT", "全选", ""), ("DESELECT", "全不选", ""),
                                       ("INVERT", "反选", "")], default="TOGGLE")

    def execute(self, ctx) -> str:
        project = _project(ctx)
        before = selection_state(project)
        visible = [o for o in project.all_objects() if o.visible]
        action = self.action
        if action == "TOGGLE":
            action = "DESELECT" if any(o.select for o in visible) else "SELECT"
        for obj in visible:
            obj.select = {"SELECT": True, "DESELECT": False}.get(action, not obj.select)
        project.selection_changed()
        record_selection(ctx, {"SELECT": "全选", "DESELECT": "全不选", "INVERT": "反选"}[action], before)
        _redraw(ctx)
        return FINISHED


@ops.register
class ObjectSelectByType(_ObjectOp):
    idname = "object.select_by_type"
    label = "按类型全选"
    description = "选择所有同一类型的物体"
    type = EnumProperty("类型", items=[("MESH", "网格", ""), ("CAMERA", "摄像机", ""), ("LIGHT", "灯光", ""),
                                     ("EMPTY", "空物体", "")], default="MESH")
    extend = BoolProperty("加选", default=False)

    def execute(self, ctx) -> str:
        project = _project(ctx)
        before = selection_state(project)
        found = [o for o in project.all_objects() if o.visible and o.kind == self.type]
        apply_select_mode(project, found, "ADD" if self.extend else "SET")
        record_selection(ctx, "按类型全选", before)
        _redraw(ctx)
        return FINISHED


@ops.register
class ObjectSelectLinked(_ObjectOp):
    idname = "object.select_linked"
    label = "选择关联"
    description = "选择和选中物体共用材质（纹理集）的物体"
    type = EnumProperty("关联", items=[("MATERIAL", "材质", "共用同一套纹理集")], default="MATERIAL")
    extend = BoolProperty("加选", default=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return object_mode(ctx) and any(o.kind == "MESH" for o in _project(ctx).selected_objects())

    def execute(self, ctx) -> str:
        project = _project(ctx)
        before = selection_state(project)
        sets = {uid for o in project.selected_objects() if o.kind == "MESH" for uid in o.material_sets}
        found = [o for o in project.objects if o.visible and sets.intersection(o.material_sets)]
        apply_select_mode(project, found, "ADD" if self.extend else "SET")
        record_selection(ctx, "选择关联", before)
        _redraw(ctx)
        return FINISHED


@ops.register
class ObjectSelectGrouped(_ObjectOp):
    idname = "object.select_grouped"
    label = "按组选择"
    description = "选择和当前物体同类的物体"
    type = EnumProperty("依据", items=[("TYPE", "类型", "同一种物体（网格、摄像机、灯光、空物体）"),
                                     ("LIGHT_TYPE", "灯光类型", "同一种灯光")], default="TYPE")
    extend = BoolProperty("加选", default=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        return object_mode(ctx) and _project(ctx).active_object is not None

    def execute(self, ctx) -> str:
        project = _project(ctx)
        active = project.active_object
        before = selection_state(project)
        if self.type == "LIGHT_TYPE":
            if active.kind != "LIGHT":
                return CANCELLED
            found = [o for o in project.scene_objects if o.visible and o.kind == "LIGHT"
                     and o.data.type == active.data.type]
        else:
            found = [o for o in project.all_objects() if o.visible and o.kind == active.kind]
        apply_select_mode(project, found, "ADD" if self.extend else "SET")
        record_selection(ctx, "按组选择", before)
        _redraw(ctx)
        return FINISHED


# ====================================================================== 框选、刷选、套索
SELECT_MODE_ITEMS = [("SET", "替换", "只选框里的"), ("ADD", "加选", ""), ("SUB", "减选", ""),
                     ("XOR", "反转", "框里的选中状态反过来"), ("AND", "交集", "只留框里本来就选中的")]


class _GestureOp(_ViewObjectOp):
    searchable = False
    mode = EnumProperty("方式", items=SELECT_MODE_ITEMS, default="SET")
    wait_for_input = BoolProperty("先等按下", default=False, description="按快捷键后先等鼠标按下再开始")

    def _start(self, ctx, event) -> None:
        self.editor = view3d(ctx)
        self.button = event.type if event is not None and event.type in (LEFTMOUSE, MIDDLEMOUSE, RIGHTMOUSE) else None
        self.started = not self.wait_for_input
        self.current = self._pixel(event)
        self.start = self.current
        self.before = selection_state(_project(ctx))

    def _pixel(self, event):
        editor = self.editor
        if event is not None and event.source is editor.widget:
            return editor.pixel(event)
        return editor.cursor_pixel()

    def _finish(self, ctx) -> None:
        self.editor.set_overlay_shape(None)
        _hint(ctx, "")

    def cancel(self, ctx) -> None:
        self._finish(ctx)


@ops.register
class View3DSelectBox(_GestureOp):
    idname = "view3d.select_box"
    label = "框选"
    description = "拖出一个框选择物体：Shift 加选，Ctrl 减选"

    def invoke(self, ctx, event) -> str:
        self._start(ctx, event)
        if self.started:
            self.editor.set_overlay_shape(("box", self.start, self.current))
        else:
            self.editor.set_overlay_shape(("cross", self.current))
        _hint(ctx, "拖出一个框：左键选择，中键取消选择；右键或 Esc 退出")
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            self.current = self._pixel(event)
            editor.set_overlay_shape(("box", self.start, self.current) if self.started else ("cross", self.current))
            return RUNNING_MODAL
        if not self.started and event.value == PRESS and event.type in (LEFTMOUSE, MIDDLEMOUSE):
            self.started = True
            self.button = event.type
            self.start = self.current = self._pixel(event)
            # 和 Blender 一样：按 B 之后左键是加选，中键是减选
            self.mode = "SUB" if (event.type == MIDDLEMOUSE or event.ctrl) else "ADD"
            editor.set_overlay_shape(("box", self.start, self.current))
            return RUNNING_MODAL
        if self.started and event.value == RELEASE and event.type == (self.button or event.type):
            self.current = self._pixel(event)
            self._finish(ctx)
            if max(abs(self.current[0] - self.start[0]), abs(self.current[1] - self.start[1])) < 1.0:
                return CANCELLED
            self._apply(ctx)
            return FINISHED
        if event.value == PRESS and event.type in (RIGHTMOUSE, "ESC"):
            self._finish(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def _apply(self, ctx) -> None:
        project = _project(ctx)
        engine = _engine(ctx)
        _current(ctx)
        (x0, y0), (x1, y1) = self.start, self.current
        if edit_active(ctx):
            from .mesh_edit_ops import edit_region_select

            lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
            edit_region_select(ctx, self.editor, lambda sx, sy: (sx >= lo_x) & (sx <= hi_x) & (sy >= lo_y) &
                               (sy <= hi_y), self.mode, "框选")
            return
        found = engine.objects_in_rect(self.editor.view, x0, y0, x1, y1)
        apply_select_mode(project, found, self.mode)
        record_selection(ctx, "框选", self.before)
        _redraw(ctx)


@ops.register
class View3DSelectCircle(_GestureOp):
    idname = "view3d.select_circle"
    label = "刷选"
    description = "用一个圆圈刷过物体来选择：左键选，中键或 Shift 取消选；滚轮改圆圈大小"
    radius = IntProperty("半径", default=25, min=1, max=2000, unit="px")
    wait_for_input = BoolProperty("先等按下", default=True)

    def invoke(self, ctx, event) -> str:
        self._start(ctx, event)
        self.painting = None          # None 没按着；True 选；False 取消选
        if not self.wait_for_input and event is not None:
            self.painting = not (self.mode == "SUB" or event.shift)
            if self.mode == "SET":
                if edit_active(ctx):
                    session = _engine(ctx).edit
                    self.edit_before = session.begin()
                    session.mesh.select_all("DESELECT")
                else:
                    apply_select_mode(_project(ctx), [], "SET")
            self._paint(ctx)
        self.editor.set_overlay_shape(("circle", self.current, float(self.radius)))
        _hint(ctx, "左键刷选，中键刷掉选择，滚轮改大小；右键、Esc 或回车结束")
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            self.current = self._pixel(event)
            editor.set_overlay_shape(("circle", self.current, float(self.radius)))
            if self.painting is not None:
                self._paint(ctx)
            return RUNNING_MODAL
        if event.type in (WHEELUP, WHEELDOWN) or event.value == WHEEL:
            bigger = event.type == WHEELDOWN or getattr(event, "wheel", 0.0) < 0
            self.radius = max(1, int(round(self.radius * (1.15 if bigger else 1.0 / 1.15))))
            editor.set_overlay_shape(("circle", self.current, float(self.radius)))
            return RUNNING_MODAL
        if event.value == PRESS and event.type in ("NUMPAD_PLUS", "EQUAL"):
            self.radius = int(self.radius * 1.15) + 1
            editor.set_overlay_shape(("circle", self.current, float(self.radius)))
            return RUNNING_MODAL
        if event.value == PRESS and event.type in ("NUMPAD_MINUS", "MINUS"):
            self.radius = max(1, int(self.radius / 1.15))
            editor.set_overlay_shape(("circle", self.current, float(self.radius)))
            return RUNNING_MODAL
        if event.value == PRESS and event.type in (LEFTMOUSE, MIDDLEMOUSE):
            self.painting = event.type == LEFTMOUSE and not event.shift and not event.ctrl
            self._paint(ctx)
            return RUNNING_MODAL
        if event.value == RELEASE and event.type in (LEFTMOUSE, MIDDLEMOUSE):
            self.painting = None
            if not self.wait_for_input:
                return self._end(ctx)
            return RUNNING_MODAL
        if event.value == PRESS and event.type in (RIGHTMOUSE, "ESC", "RET", "NUMPAD_ENTER"):
            return self._end(ctx)
        return RUNNING_MODAL

    def _paint(self, ctx) -> None:
        engine = _engine(ctx)
        project = _project(ctx)
        _current(ctx)
        x, y = self.current
        r = float(self.radius)
        if edit_active(ctx):
            from ..mesh import select as mesh_select
            from .mesh_edit_ops import _xray

            session = engine.edit
            if not hasattr(self, "edit_before"):
                self.edit_before = session.begin()
            found = mesh_select.elements_in(engine, self.editor.view, session.mesh,
                                            lambda sx, sy: (sx - x) ** 2 + (sy - y) ** 2 <= r * r, _xray(self.editor))
            mesh_select.apply_region(session.mesh, found, "ADD" if self.painting else "SUB")
            session.update_overlay()
            ctx.app.request_frame()
            return
        ids, (ox, oy) = engine.renderer.object_id_image(self.editor.view, x - r, y - r, x + r, y + r)
        found = []
        if ids.size:
            yy, xx = np.mgrid[0:ids.shape[0], 0:ids.shape[1]]
            inside = (xx + ox - x) ** 2 + (yy + oy - y) ** 2 <= r * r
            for pick in np.unique(ids[inside]):
                obj = engine.pick_objects.get(int(pick))
                if obj is not None:
                    found.append(obj)
        found += _scene_objects_in(engine, self.editor.view,
                                   lambda sx, sy: (sx - x) ** 2 + (sy - y) ** 2 <= r * r)
        if not found:
            return
        apply_select_mode(project, found, "ADD" if self.painting else "SUB")
        _redraw(ctx)

    def _end(self, ctx) -> str:
        self._finish(ctx)
        if edit_active(ctx):
            session = _engine(ctx).edit
            if hasattr(self, "edit_before"):
                session.push_selection("刷选", self.edit_before)
            return FINISHED
        record_selection(ctx, "刷选", self.before)
        return FINISHED


def _scene_objects_in(engine, view, test) -> list:
    out = []
    project = engine.project
    if project is None:
        return out
    view_proj = view.camera.view_projection(view.aspect)
    for obj in project.scene_objects:
        if not obj.visible:
            continue
        p = to_internal(np.asarray(obj.transform.location, np.float64))
        clip = view_proj @ np.array([p[0], p[1], p[2], 1.0])
        if clip[3] <= 1e-9:
            continue
        sx = (clip[0] / clip[3] * 0.5 + 0.5) * view.width
        sy = (0.5 - clip[1] / clip[3] * 0.5) * view.height
        if test(sx, sy):
            out.append(obj)
    return out


@ops.register
class View3DSelectLasso(_GestureOp):
    idname = "view3d.select_lasso"
    label = "套索选择"
    description = "按住拖出任意形状选择物体：Shift 加选，Ctrl 减选"

    def invoke(self, ctx, event) -> str:
        self._start(ctx, event)
        self.path = [self.start]
        self.editor.set_overlay_shape(("lasso", list(self.path)))
        _hint(ctx, "拖出一圈选择物体；Esc 取消")
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            point = self._pixel(event)
            last = self.path[-1]
            if abs(point[0] - last[0]) + abs(point[1] - last[1]) >= 2.0:
                self.path.append(point)
                self.editor.set_overlay_shape(("lasso", list(self.path)))
            return RUNNING_MODAL
        if event.value == RELEASE and event.type == (self.button or event.type):
            self._finish(ctx)
            if len(self.path) < 3:
                return CANCELLED
            self._apply(ctx)
            return FINISHED
        if event.value == PRESS and (event.type == "ESC" or (event.type == RIGHTMOUSE and self.button != RIGHTMOUSE)):
            self._finish(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def _apply(self, ctx) -> None:
        engine = _engine(ctx)
        project = _project(ctx)
        _current(ctx)
        path = np.asarray(self.path, np.float64)
        x0, y0 = path.min(axis=0)
        x1, y1 = path.max(axis=0)
        if edit_active(ctx):
            from .mesh_edit_ops import edit_region_select

            edit_region_select(ctx, self.editor, lambda sx, sy: points_in_polygon(sx, sy, path), self.mode,
                               "套索选择")
            return
        ids, (ox, oy) = engine.renderer.object_id_image(self.editor.view, x0, y0, x1, y1)
        found = []
        if ids.size:
            mask = polygon_mask(path - np.array([ox, oy]), ids.shape[1], ids.shape[0])
            for pick in np.unique(ids[mask]):
                obj = engine.pick_objects.get(int(pick))
                if obj is not None:
                    found.append(obj)
        found += _scene_objects_in(engine, self.editor.view, lambda sx, sy: point_in_polygon(sx, sy, path))
        apply_select_mode(project, found, self.mode)
        record_selection(ctx, "套索选择", self.before)
        _redraw(ctx)


def polygon_mask(path: np.ndarray, width: int, height: int) -> np.ndarray:
    """多边形（像素坐标）里面的像素：[height, width] 布尔数组。"""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QImage, QPainter, QPolygonF

    image = QImage(max(1, width), max(1, height), QImage.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    painter.setPen(Qt.NoPen)
    painter.setBrush(Qt.white)
    painter.drawPolygon(QPolygonF([QPointF(float(x) + 0.5, float(y) + 0.5) for x, y in path]))
    painter.end()
    stride = image.bytesPerLine()
    raw = np.frombuffer(image.constBits(), np.uint8, count=stride * image.height()).reshape(image.height(), stride)
    return raw[:height, :width] > 127


def points_in_polygon(xs: np.ndarray, ys: np.ndarray, path: np.ndarray) -> np.ndarray:
    """一组屏幕点是否在多边形里（奇偶规则）。"""
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    inside = np.zeros(xs.shape, bool)
    n = len(path)
    for i in range(n):
        x0, y0 = path[i]
        x1, y1 = path[(i + 1) % n]
        if y0 == y1:
            continue
        crosses = (y0 > ys) != (y1 > ys)
        cross_x = x0 + (ys - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (xs < cross_x)
    return inside


def point_in_polygon(x: float, y: float, path: np.ndarray) -> bool:
    inside = False
    n = len(path)
    for i in range(n):
        x0, y0 = path[i]
        x1, y1 = path[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < cross:
                inside = not inside
    return inside


@ops.register
class View3DSelectTweakMove(_ViewObjectOp):
    idname = "view3d.select_tweak_move"
    label = "拖动物体"
    description = "按住物体拖动就移动它（没选中的先选中）；在空白处拖动是框选"
    searchable = False

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        project = _project(ctx)
        engine = _engine(ctx)
        if event is None:
            return CANCELLED
        _current(ctx)
        x, y = editor.pixel(event)
        if edit_active(ctx):
            return self._edit_tweak(ctx, editor, engine, event, x, y)
        obj = engine.object_at(editor.view, x, y)
        if obj is None:
            # 交给框选（它自己进模态），这个操作本身就此结束
            ops.call("view3d.select_box", ctx, event, mode="SUB" if event.ctrl else ("ADD" if event.shift else "SET"))
            return FINISHED
        if not obj.select:
            before = selection_state(project)
            if event.shift:
                obj.select = True
                set_active(project, obj)
                project.selection_changed()
            else:
                select_only(project, [obj], obj)
            record_selection(ctx, "选择", before)
            _redraw(ctx)
        ops.call("transform.translate", ctx, event, release_confirm=True)
        return FINISHED

    def _edit_tweak(self, ctx, editor, engine, event, x, y) -> str:
        """编辑模式：按住点、边、面拖动就移动它（没选中的先选中）；空白处拖动是框选。"""
        from ..mesh import select as mesh_select
        from .mesh_edit_ops import _xray

        session = engine.edit
        mesh = session.mesh
        element = mesh_select.pick(engine, editor.view, mesh, x, y, _xray(editor))
        if element is None:
            ops.call("view3d.select_box", ctx, event, mode="SUB" if event.ctrl else ("ADD" if event.shift else "SET"))
            return FINISHED
        kind, index = element
        selected = {"VERT": mesh.vert_sel, "EDGE": mesh.edge_sel, "FACE": mesh.face_sel}[kind]
        if not selected[index]:
            before = session.begin()
            mesh_select.select_element(mesh, element, extend=bool(event.shift))
            session.push_selection("选择", before)
            _redraw(ctx)
        ops.call("transform.translate", ctx, event, release_confirm=True)
        return FINISHED


# ====================================================================== 变换（G / R / S）
def pivot_points(ctx, objects, pivot_mode: str) -> tuple[np.ndarray, bool]:
    """(轴心点, 是否各自绕自己的原点)。都是显示坐标。"""
    project = _project(ctx)
    origins = np.array([o.transform.matrix()[:3, 3] for o in objects], np.float64).reshape(-1, 3)
    if pivot_mode == "CURSOR":
        return np.asarray(project.cursor_location, np.float64), False
    if len(origins) == 0:
        return np.zeros(3), False
    if pivot_mode == "BOUNDING_BOX_CENTER":
        return (origins.min(axis=0) + origins.max(axis=0)) * 0.5, False
    if pivot_mode == "ACTIVE_ELEMENT":
        active = project.active_object
        if active is not None and active in objects:
            return active.transform.matrix()[:3, 3].copy(), False
    if pivot_mode == "INDIVIDUAL_ORIGINS":
        return origins.mean(axis=0), True
    return origins.mean(axis=0), False


def _basis_of(matrix) -> np.ndarray:
    m = np.asarray(matrix, np.float64)[:3, :3]
    cols = []
    for k in range(3):
        v = m[:, k]
        n = np.linalg.norm(v)
        cols.append(v / n if n > 1e-12 else np.eye(3)[k])
    return np.stack(cols, axis=1)


def orientation_basis(ctx, orient: str, obj=None, editor=None) -> np.ndarray:
    """坐标系的三个轴（显示坐标，按列排）。"""
    if orient == "NORMAL":
        if ctx is not None and edit_active(ctx):
            from .edit_transform import normal_basis

            mesh = _engine(ctx).edit.mesh
            return normal_basis(mesh, np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0])
        orient = "LOCAL"
    if orient == "LOCAL" and obj is not None:
        return _basis_of(obj.transform.matrix())
    if orient == "VIEW" and editor is not None and getattr(editor, "view", None) is not None:
        right, up, back = editor.view.camera.basis()
        return np.stack([to_display(right), to_display(up), to_display(back)], axis=1)
    return np.identity(3)


def _axis_rotation(axis, angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, np.float64) / max(np.linalg.norm(axis), 1e-12)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def _plane_hit(origin, direction, point, normal):
    denom = float(direction @ normal)
    if abs(denom) < 1e-9:
        return None
    t = float((point - origin) @ normal) / denom
    return origin + direction * t


def _axis_param(origin, direction, point, axis) -> float:
    """视线和轴线最近处，在轴线上的参数（相对 point）。"""
    w0 = point - origin
    a = float(axis @ axis)
    b = float(axis @ direction)
    c = float(direction @ direction)
    d = float(axis @ w0)
    e = float(direction @ w0)
    denom = a * c - b * b
    if abs(denom) < 1e-12:
        return 0.0
    return (b * e - c * d) / denom


def constraint_axes(constraint: str) -> list[int]:
    """约束 → 能动的轴：X → [0]，YZ → [1, 2]，NONE → []（自由）。"""
    if constraint in ("X", "Y", "Z"):
        return ["XYZ".index(constraint)]
    if constraint in ("YZ", "XZ", "XY"):
        return ["XYZ".index(c) for c in constraint]
    return []


def _axis_mask(constraint: str) -> np.ndarray:
    axes = constraint_axes(constraint)
    if not axes:
        return np.ones(3)
    mask = np.zeros(3)
    mask[axes] = 1.0
    return mask


def apply_object_matrices(ctx, changes: list, label: str, merge_previous: int = 0) -> None:
    """把一组 (物体, 原矩阵, 新矩阵) 写进物体（网格的顶点由引擎跟着变），记一步撤销。"""
    for obj, _start, new in changes:
        obj.transform.set_matrix(new)
    history = _history(ctx)
    _redraw(ctx, "transform")
    if history is None or not changes:
        return

    def undo() -> None:
        _current(ctx)
        for obj, start, _new in changes:
            obj.transform.set_matrix(start)
        _redraw(ctx, "transform")

    def redo() -> None:
        _current(ctx)
        for obj, _start, new in changes:
            obj.transform.set_matrix(new)
        _redraw(ctx, "transform")

    history.push(label, undo, redo)
    if merge_previous and hasattr(history, "merge_last"):
        history.merge_last(merge_previous + 1)


def _surface_hit(app, items, editor, pixel):
    """吸附到表面：鼠标视线打到的（没在移动的）模型表面上的点，显示坐标。"""
    from ..bake.parts import ray_hit

    moving = {obj.uid for obj, _m in items}
    view = editor.view
    origin, direction = view.camera.ray(pixel[0], pixel[1], view.width, view.height)
    best = None
    for obj in app.project.objects:
        if not obj.visible or obj.uid in moving:
            continue
        if not _ray_box(origin, direction, np.asarray(obj.data.bounds_min, np.float64),
                        np.asarray(obj.data.bounds_max, np.float64)):
            continue
        tri, _u, _v, t = ray_hit(obj.data.positions, origin, direction)
        if tri >= 0 and (best is None or t < best):
            best = t
    if best is None:
        return None
    return to_display(origin + direction * best)


class _TransformBase(Operator):
    """移动、旋转、缩放共用：能拖（invoke 进入模态），也能按属性直接做（execute，「调整上一步」也走这里）。"""

    kind = "TRANSLATE"
    redo = True
    release_confirm = BoolProperty("松开确认", default=False, hidden=True)
    merge_previous = IntProperty("并入前几步", default=0, hidden=True, description="复制后移动合成一步撤销")
    constraint = EnumProperty("约束", items=CONSTRAINT_ITEMS, default="NONE")
    orient_type = EnumProperty("坐标系", items=ORIENT_ITEMS, default="GLOBAL")

    @classmethod
    def poll(cls, ctx) -> bool:
        return (object_mode(ctx) and bool(_project(ctx).selected_objects())) or edit_has_selection(ctx)

    # ---------------------------------------------------------------- 直接做
    def execute(self, ctx) -> str:
        if edit_active(ctx):
            return self._execute_edit(ctx)
        project = _project(ctx)
        objects = project.selected_objects()
        if not objects:
            return CANCELLED
        settings = ctx.app.tool_settings
        editor = view3d(ctx)
        pivot, individual = pivot_points(ctx, objects, settings.transform_pivot_point)
        changes = []
        for obj in objects:
            start = obj.transform.matrix()
            delta = self.delta_from_props(ctx, obj, start[:3, 3] if individual else pivot, editor)
            changes.append((obj, start, delta @ start))
        _current(ctx)
        apply_object_matrices(ctx, changes, self.label)
        return FINISHED

    def delta_from_props(self, ctx, obj, pivot, editor) -> np.ndarray:
        raise NotImplementedError

    def _execute_edit(self, ctx) -> str:
        """编辑模式里按属性直接做（「调整上一步」也走这里）。"""
        from .edit_transform import apply_edit_delta, edit_pivot, proportional_weights, to_disp

        session = _engine(ctx).edit
        mesh = session.mesh
        sel = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
        if not len(sel):
            return CANCELLED
        self.edit = session
        self.edit_sel = sel
        before = session.begin()
        self.edit_disp = to_disp(mesh.verts)
        self.pivot, self.edit_islands = edit_pivot(self, ctx, mesh, self.edit_disp, sel)
        settings = ctx.app.tool_settings
        if settings.use_proportional_edit:
            self.prop_idx, self.prop_weight = proportional_weights(
                mesh, self.edit_disp, sel, settings.proportional_size, settings.proportional_edit_falloff,
                settings.use_proportional_connected)
        else:
            self.prop_idx, self.prop_weight = None, None
        delta = self.delta_from_props(ctx, session.obj, self.pivot, view3d(ctx))
        _current(ctx)
        apply_edit_delta(self, delta)
        session.push(self.label, before, topology=False)
        return FINISHED

    # ---------------------------------------------------------------- 拖动
    def invoke(self, ctx, event) -> str:
        project = _project(ctx)
        self.editor = view3d(ctx)
        self.engine = _engine(ctx)
        self.app = ctx.app
        self.edit = None
        if self.editor is None or self.engine is None:
            return self.execute(ctx)
        self.settings = ctx.app.tool_settings
        if edit_active(ctx):
            if not self._invoke_edit(ctx):
                return CANCELLED
        else:
            self.items = [(obj, obj.transform.matrix()) for obj in project.selected_objects()]
            if not self.items:
                return CANCELLED
            self._originals = {obj.uid: m.copy() for obj, m in self.items}
            self.pivot, self.individual = pivot_points(ctx, [o for o, _m in self.items],
                                                       self.settings.transform_pivot_point)
            selected = [o for o, _m in self.items]
            self.reference = project.active_object if project.active_object in selected else self.items[0][0]
        self.button = event.type if event is not None and event.type in (LEFTMOUSE, MIDDLEMOUSE, RIGHTMOUSE) else None
        mouse = self._pixel(event)
        self.start = tuple(mouse)
        self.current = tuple(mouse)
        self.axes: list[int] = []
        self.orient = "GLOBAL"
        self.orient_cycle = 0
        self.numeric = ""
        self.snap_invert = bool(event.ctrl) if event is not None and self.release_confirm else False
        self.precise = False
        self._precise_base = None
        self._last_angle = None
        self._angle_total = 0.0
        self._effective = None
        self.kind = type(self).kind
        if self.constraint != "NONE":
            # 调用时给了约束（比如挤出后沿法线拉出去）
            self.axes = constraint_axes(self.constraint)
            self.orient = self.orient_type
        self._update(ctx)
        return RUNNING_MODAL

    def _invoke_edit(self, ctx) -> bool:
        from .edit_transform import edit_pivot, normal_basis, to_disp

        session = self.engine.edit
        mesh = session.mesh
        sel = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
        if not len(sel):
            return False
        self.edit = session
        self.edit_sel = sel
        self.edit_before = session.begin()
        self.edit_verts0 = mesh.verts.copy()
        self.edit_disp = to_disp(mesh.verts)
        self.pivot, self.edit_islands = edit_pivot(self, ctx, mesh, self.edit_disp, sel)
        self.individual = False
        obj = session.obj
        self.items = [(obj, obj.transform.matrix())]
        self._originals = {obj.uid: self.items[0][1].copy()}
        self.reference = obj
        self.normal_basis = normal_basis(mesh, sel)
        self._update_weights()
        return True

    def _update_weights(self) -> None:
        """衰减编辑：受影响的点和权重（半径、衰减方式改了要重算）。"""
        from .edit_transform import proportional_weights

        settings = self.settings
        if self.edit is not None and settings.use_proportional_edit:
            self.prop_idx, self.prop_weight = proportional_weights(
                self.edit.mesh, self.edit_disp, self.edit_sel, settings.proportional_size,
                settings.proportional_edit_falloff, settings.use_proportional_connected)
        else:
            self.prop_idx, self.prop_weight = None, None

    def _pixel(self, event):
        editor = self.editor
        if event is not None and getattr(event, "source", None) is editor.widget:
            return editor.pixel(event)
        return editor.cursor_pixel()

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            self.current = self._pixel(event)
            self.snap_invert = bool(event.ctrl)
            self._set_precise(bool(event.shift))
            self._update(ctx)
            return RUNNING_MODAL
        if self.edit is not None and self.settings.use_proportional_edit and (
                event.type in (WHEELUP, WHEELDOWN) or event.value == WHEEL or
                (event.value == PRESS and event.type in ("PAGE_UP", "PAGE_DOWN"))):
            # 和 Blender 一样：滚轮、PageUp/PageDown 改衰减范围
            grow = event.type in (WHEELDOWN, "PAGE_UP") or (event.value == WHEEL and
                                                            getattr(event, "wheel", 0.0) < 0)
            size = float(self.settings.proportional_size) * (1.1 if grow else 1.0 / 1.1)
            self.settings.proportional_size = max(1e-5, size)
            self._update_weights()
            self._update(ctx)
            return RUNNING_MODAL
        if self.edit is not None and event.value == PRESS and event.type == "O" and not (event.ctrl or event.alt):
            self.settings.use_proportional_edit = not self.settings.use_proportional_edit
            self._update_weights()
            self._update(ctx)
            return RUNNING_MODAL
        if self.release_confirm and event.value == RELEASE and event.type == self.button:
            self._confirm(ctx)
            return FINISHED
        if event.value == PRESS:
            key = event.type
            if key in AXIS_KEYS and not event.ctrl and not event.alt:
                self._press_axis(AXIS_KEYS[key], bool(event.shift))
                self._update(ctx)
                return RUNNING_MODAL
            if key in ("G", "R", "S") and not (event.ctrl or event.alt or event.shift):
                if key == "G" and self.kind == "TRANSLATE" and self.edit is not None and not self.merge_previous:
                    # 编辑模式里连按两次 G：滑移边（只选了点时滑移点）
                    slide = "transform.edge_slide" if self.edit.mesh.edge_sel.any() else "transform.vert_slide"
                    self.cancel(ctx)
                    ops.call(slide, ctx, event)
                    return CANCELLED
                self._switch({"G": "TRANSLATE", "R": "ROTATE", "S": "RESIZE"}[key], ctx)
                return RUNNING_MODAL
            if key in DIGIT_KEYS:
                char = DIGIT_KEYS[key]
                if char == "-":
                    self.numeric = self.numeric[1:] if self.numeric.startswith("-") else "-" + self.numeric
                else:
                    self.numeric += char
                self._update(ctx)
                return RUNNING_MODAL
            if key == "BACK_SPACE":
                self.numeric = self.numeric[:-1]
                self._update(ctx)
                return RUNNING_MODAL
            if key == MIDDLEMOUSE:
                self._auto_axis()
                self._update(ctx)
                return RUNNING_MODAL
            if key in (LEFTMOUSE, "RET", "NUMPAD_ENTER", "SPACE"):
                self._confirm(ctx)
                return FINISHED
            if key in (RIGHTMOUSE, "ESC"):
                self.cancel(ctx)
                return CANCELLED
            if key in ("LEFT_CTRL", "RIGHT_CTRL"):
                self.snap_invert = True
                self._update(ctx)
            elif key in ("LEFT_SHIFT", "RIGHT_SHIFT"):
                self._set_precise(True)
                self._update(ctx)
            return RUNNING_MODAL
        if event.value == RELEASE:
            if event.type in ("LEFT_CTRL", "RIGHT_CTRL"):
                self.snap_invert = False
                self._update(ctx)
            elif event.type in ("LEFT_SHIFT", "RIGHT_SHIFT"):
                self._set_precise(False)
                self._update(ctx)
        return RUNNING_MODAL

    # ---- 约束 ----
    def _press_axis(self, axis: int, plane: bool) -> None:
        """第一次按：用设置里的坐标系；再按同一个：换成全局、局部里的另一个；第三次：取消约束（和 Blender 一样）。"""
        axes = [a for a in range(3) if a != axis] if plane else [axis]
        user = self.settings.transform_orientation
        alternate = "LOCAL" if user == "GLOBAL" else "GLOBAL"
        if axes == self.axes:
            self.orient_cycle += 1
            if self.orient_cycle == 1:
                self.orient = alternate
            else:
                self.axes, self.orient_cycle = [], 0
        else:
            self.axes, self.orient, self.orient_cycle = axes, user, 0
        self._precise_base = None

    def _auto_axis(self) -> None:
        """中键：约束到和鼠标移动方向最接近的轴。"""
        move = np.array([self.current[0] - self.start[0], self.current[1] - self.start[1]], np.float64)
        length = float(np.linalg.norm(move))
        if length < 2.0:
            return
        move /= length
        center = self._screen_of(self.pivot)
        basis = self._basis()
        best, best_dot = None, -1.0
        for k in range(3):
            tip = self._screen_of(self.pivot + basis[:, k] * self._world_per_pixel() * 50.0)
            d = tip - center
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                continue
            dot = abs(float(d @ move) / n)
            if dot > best_dot:
                best, best_dot = k, dot
        if best is not None:
            self.axes, self.orient, self.orient_cycle = [best], self.settings.transform_orientation, 0

    def _switch(self, kind: str, ctx) -> None:
        """拖动中按 G/R/S 换成别的变换：已经做的先算数，从现在的样子接着做。"""
        if kind == self.kind:
            return
        deltas = self._delta()
        if self.edit is not None:
            from .edit_transform import apply_edit_delta, edit_pivot, to_disp

            _current(ctx)
            apply_edit_delta(self, deltas[0])
            mesh = self.edit.mesh
            self.edit_disp = to_disp(mesh.verts)
            self.pivot, self.edit_islands = edit_pivot(self, ctx, mesh, self.edit_disp, self.edit_sel)
            self._update_weights()
        else:
            self.items = [(obj, d @ start) for (obj, start), d in zip(self.items, deltas)]
            self.pivot = self._items_pivot()
        self.kind = kind
        self.start = tuple(self.current)
        self.axes, self.orient_cycle, self.numeric = [], 0, ""
        self._precise_base = None
        self._last_angle = None
        self._angle_total = 0.0
        self._update(ctx)

    def _items_pivot(self) -> np.ndarray:
        origins = np.array([m[:3, 3] for _o, m in self.items], np.float64)
        mode = self.settings.transform_pivot_point
        if mode == "CURSOR":
            return np.asarray(self.app.project.cursor_location, np.float64)
        if mode == "BOUNDING_BOX_CENTER":
            return (origins.min(axis=0) + origins.max(axis=0)) * 0.5
        if mode == "ACTIVE_ELEMENT":
            for obj, m in self.items:
                if obj is self.reference:
                    return m[:3, 3].copy()
        return origins.mean(axis=0)

    def _set_precise(self, value: bool) -> None:
        if value != self.precise:
            self.precise = value
            self._precise_base = None      # 下次计算时以当前位置为基准，避免跳动

    def _snapping(self) -> bool:
        return bool(self.settings.use_snap) != bool(self.snap_invert)

    # ---- 几何 ----
    def _basis(self, obj=None) -> np.ndarray:
        orient = self.orient if self.axes else "GLOBAL"
        if orient == "NORMAL":
            if self.edit is not None:
                return self.normal_basis
            orient = "LOCAL"
        if orient == "LOCAL":
            target = obj if obj is not None else self.reference
            for item, m in self.items:
                if item is target:
                    return _basis_of(m)
            return _basis_of(self.items[0][1])
        return orientation_basis(None, orient, None, self.editor)

    def _ray(self, pixel):
        view = self.editor.view
        origin, direction = view.camera.ray(pixel[0], pixel[1], view.width, view.height)
        return to_display(origin), to_display(direction)

    def _screen_of(self, point_display) -> np.ndarray:
        view = self.editor.view
        p = to_internal(np.asarray(point_display, np.float64))
        clip = view.camera.view_projection(view.aspect) @ np.array([p[0], p[1], p[2], 1.0])
        w = clip[3] if abs(clip[3]) > 1e-12 else 1e-12
        return np.array([(clip[0] / w * 0.5 + 0.5) * view.width, (0.5 - clip[1] / w * 0.5) * view.height])

    def _world_per_pixel(self) -> float:
        view = self.editor.view
        camera = view.camera
        return camera.world_per_pixel(camera.depth_of(to_internal(self.pivot)), view.height, view.aspect)

    def _value(self) -> float | None:
        text = self.numeric
        if not text or text in ("-", ".", "-."):
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _precise_scale(self, raw):
        """Shift 精细：从按下 Shift 那一刻起，后面的变化量乘以精细倍率。"""
        if not self.precise:
            self._precise_base = None
            return raw
        if self._precise_base is None:
            self._precise_base = (raw, self._effective if self._effective is not None else raw)
        raw0, eff0 = self._precise_base
        return eff0 + (raw - raw0) * float(self.settings.precision_factor)

    # ---- 每帧 ----
    def _delta(self) -> list:
        """每个物体各自的变换矩阵（显示坐标）。"""
        if self.kind == "TRANSLATE":
            move = self._translate_vector()
            out = []
            for obj, _start in self.items:
                world = move
                if self.axes and self.orient == "LOCAL" and len(self.items) > 1:
                    world = self._basis(obj) @ (self._basis().T @ move)
                m = np.identity(4)
                m[:3, 3] = world
                out.append(m)
            return out
        if self.kind == "ROTATE":
            angle, axis_local = self._rotate_angle_axis()
            out = []
            for obj, start in self.items:
                axis = self._basis(obj) @ axis_local if self.axes else axis_local
                r = _axis_rotation(axis, math.radians(angle))
                center = start[:3, 3] if self.individual else self.pivot
                m = np.identity(4)
                m[:3, :3] = r
                m[:3, 3] = center - r @ center
                out.append(m)
            return out
        factors = self._resize_factors()
        out = []
        for obj, start in self.items:
            basis = self._basis(obj) if self.axes else np.identity(3)
            s = basis @ np.diag(factors) @ basis.T
            center = start[:3, 3] if self.individual else self.pivot
            m = np.identity(4)
            m[:3, :3] = s
            m[:3, 3] = center - s @ center
            out.append(m)
        return out

    def _translate_vector(self) -> np.ndarray:
        value = self._value()
        basis = self._basis()
        if value is not None:
            axis = self.axes[0] if self.axes else 0
            self.value_vector = np.eye(3)[axis] * value
            return basis[:, axis] * value
        o0, d0 = self._ray(self.start)
        o1, d1 = self._ray(self.current)
        pivot = self.pivot
        if not self.axes:
            normal = -d0
            h0, h1 = _plane_hit(o0, d0, pivot, normal), _plane_hit(o1, d1, pivot, normal)
            raw = h1 - h0 if h0 is not None and h1 is not None else np.zeros(3)
        elif len(self.axes) == 1:
            axis = basis[:, self.axes[0]]
            raw = axis * (_axis_param(o1, d1, pivot, axis) - _axis_param(o0, d0, pivot, axis))
        else:
            normal = basis[:, [a for a in range(3) if a not in self.axes][0]]
            h0, h1 = _plane_hit(o0, d0, pivot, normal), _plane_hit(o1, d1, pivot, normal)
            raw = h1 - h0 if h0 is not None and h1 is not None else np.zeros(3)
        move = self._precise_scale(raw)
        self._effective = move
        if self._snapping():
            move = self._snap_translate(move, basis)
        self.value_vector = basis.T @ move
        return move

    def _snap_translate(self, move: np.ndarray, basis: np.ndarray) -> np.ndarray:
        settings = self.settings
        step = float(settings.snap_translate) * (float(settings.precision_factor) if self.precise else 1.0)
        mode = settings.snap_elements
        mask = _axis_mask("".join("XYZ"[a] for a in self.axes)) if self.axes else np.ones(3)
        if mode == "SURFACE":
            hit = _surface_hit(self.app, self.items, self.editor, self.current)
            if hit is not None:
                return basis @ ((basis.T @ (hit - self.pivot)) * mask)
        local = basis.T @ move
        if mode == "GRID":
            origin = basis.T @ self.pivot
            local = np.round((origin + local) / step) * step - origin
        else:
            local = np.round(local / step) * step
        return basis @ (local * mask)

    def _rotate_angle_axis(self) -> tuple[float, np.ndarray]:
        view = self.editor.view
        center = self._screen_of(self.pivot)
        a1 = math.atan2(-(self.current[1] - center[1]), self.current[0] - center[0])
        if self._last_angle is None:
            self._last_angle = math.atan2(-(self.start[1] - center[1]), self.start[0] - center[0])
            self._angle_total = 0.0
        step = a1 - self._last_angle
        step = (step + math.pi) % (2.0 * math.pi) - math.pi        # 绕过 ±180° 时不跳
        self._angle_total += step
        self._last_angle = a1
        raw = math.degrees(self._angle_total)
        toward_viewer = to_display(view.camera.basis()[2])
        if self.axes:
            axis_index = self.axes[0] if len(self.axes) == 1 else [a for a in range(3) if a not in self.axes][0]
            axis_local = np.eye(3)[axis_index]
            world_axis = self._basis() @ axis_local
            if float(world_axis @ toward_viewer) < 0:
                raw = -raw
        else:
            axis_local = toward_viewer
        value = self._value()
        if value is not None:
            angle = value
        else:
            angle = float(self._precise_scale(raw))
            self._effective = angle
            if self._snapping():
                step_deg = float(self.settings.snap_rotate) * \
                    (float(self.settings.precision_factor) if self.precise else 1.0)
                angle = round(angle / step_deg) * step_deg
        self.value_angle = angle
        self.value_axis = axis_local
        return angle, axis_local

    def _resize_factors(self) -> np.ndarray:
        value = self._value()
        if value is not None:
            factor = value
        else:
            center = self._screen_of(self.pivot)
            d0 = max(math.hypot(self.start[0] - center[0], self.start[1] - center[1]), 1.0)
            d1 = math.hypot(self.current[0] - center[0], self.current[1] - center[1])
            factor = float(self._precise_scale(d1 / d0))
            self._effective = factor
            if self._snapping():
                step = float(self.settings.snap_scale) * \
                    (float(self.settings.precision_factor) if self.precise else 1.0)
                factor = round(factor / step) * step
        factors = np.ones(3)
        if self.axes:
            factors[self.axes] = factor
        else:
            factors[:] = factor
        self.value_factors = factors
        return factors

    def _update(self, ctx) -> None:
        deltas = self._delta()
        if self.edit is not None:
            from .edit_transform import apply_edit_delta

            _current(ctx)
            apply_edit_delta(self, deltas[0])
            self.engine.set_transform_guide(self._guide())
            self.engine.refresh_overlays()
            shapes = []
            if self.kind in ("ROTATE", "RESIZE"):
                shapes.append(("line", tuple(self._screen_of(self.pivot)), tuple(self.current)))
            if self.settings.use_proportional_edit:
                center = self._screen_of(self.pivot)
                radius_px = float(self.settings.proportional_size) / max(self._world_per_pixel(), 1e-12)
                shapes.append(("circle", (float(center[0]), float(center[1])), radius_px))
            self.editor.set_overlay_shape(("multi", tuple(shapes)) if shapes else None)
            _hint(ctx, self._status())
            ctx.app.request_frame()
            return
        for (obj, start), delta in zip(self.items, deltas):
            new = delta @ start
            if obj.kind == "MESH":
                # 网格的变换值在确认前不动，视口按矩阵实时显示
                self.engine.set_display_matrix(obj, display_to_internal_matrix(
                    new @ np.linalg.inv(self._originals[obj.uid])))
            else:
                obj.transform.set_matrix(new)
        self.engine.set_transform_guide(self._guide())
        self.engine.refresh_overlays()
        if self.kind in ("ROTATE", "RESIZE"):
            self.editor.set_overlay_shape(("line", tuple(self._screen_of(self.pivot)), tuple(self.current)))
        else:
            self.editor.set_overlay_shape(None)
        _hint(ctx, self._status())
        ctx.app.request_frame()

    def _guide(self):
        if not self.axes:
            return None
        lines = []
        basis = self._basis()
        for a in self.axes:
            lines.append((to_internal(self.pivot), to_internal(basis[:, a]), AXIS_COLORS[a]))
        return lines

    def _axis_text(self) -> str:
        if not self.axes:
            return ""
        return "  沿%s %s" % (ORIENT_LABELS.get(self.orient, ""), "".join("XYZ"[a] for a in self.axes))

    def _status(self) -> str:
        typed = ("  输入：%s" % self.numeric) if self.numeric else ""
        snap = "  吸附" if self._snapping() else ""
        tail = "    左键确认 · 右键取消 · X/Y/Z 约束（Shift 平面）· Ctrl 吸附 · Shift 精细 · G/R/S 切换"
        if self.kind == "TRANSLATE":
            v = getattr(self, "value_vector", np.zeros(3))
            return "移动  X %.4g  Y %.4g  Z %.4g%s%s%s%s" % (v[0], v[1], v[2], self._axis_text(), typed, snap, tail)
        if self.kind == "ROTATE":
            return "旋转  %.2f°%s%s%s%s" % (getattr(self, "value_angle", 0.0), self._axis_text(), typed, snap, tail)
        f = getattr(self, "value_factors", np.ones(3))
        return "缩放  %.4g  %.4g  %.4g%s%s%s%s" % (f[0], f[1], f[2], self._axis_text(), typed, snap, tail)

    def _confirm(self, ctx) -> None:
        deltas = self._delta()
        if self.edit is not None:
            from .edit_transform import apply_edit_delta

            self._store_props()
            self._finish_ui(ctx)
            _current(ctx)
            apply_edit_delta(self, deltas[0])
            label = {"TRANSLATE": "移动", "ROTATE": "旋转", "RESIZE": "缩放"}[self.kind]
            self.edit.push(label, self.edit_before, topology=False)
            history = _history(ctx)
            if self.merge_previous and history is not None:
                history.merge_last(int(self.merge_previous) + 1)
            return
        changes = []
        for (obj, start), delta in zip(self.items, deltas):
            original = self._originals[obj.uid]
            new = delta @ start
            if obj.kind == "MESH":
                self.engine.set_display_matrix(obj, None)
            else:
                with obj.transform.changed.block():
                    obj.transform.set_matrix(original)
            changes.append((obj, original, new))
        self._store_props()
        self._finish_ui(ctx)
        label = {"TRANSLATE": "移动", "ROTATE": "旋转", "RESIZE": "缩放"}[self.kind]
        _current(ctx)
        apply_object_matrices(ctx, changes, label, int(self.merge_previous))

    def _store_props(self) -> None:
        """把拖出来的结果写回属性，「调整上一步」面板能接着改。"""
        if self.axes:
            if len(self.axes) == 1:
                self.constraint = "XYZ"[self.axes[0]]
            else:
                self.constraint = {frozenset((1, 2)): "YZ", frozenset((0, 2)): "XZ",
                                   frozenset((0, 1)): "XY"}[frozenset(self.axes)]
            self.orient_type = self.orient
        else:
            self.constraint = "NONE"
            self.orient_type = "GLOBAL"
        if self.kind != type(self).kind:
            self.redo_disabled = True       # 中途换过 G/R/S，属性说不清，就不提供调整
            return
        if self.kind == "TRANSLATE":
            self.value = tuple(float(v) for v in getattr(self, "value_vector", np.zeros(3)))
        elif self.kind == "ROTATE":
            self.value = float(getattr(self, "value_angle", 0.0))
            if not self.axes:
                self.orient_axis = "VIEW"
                self._view_axis = np.asarray(getattr(self, "value_axis", np.array([0.0, 0.0, 1.0])), np.float64)
        else:
            self.value = tuple(float(v) for v in getattr(self, "value_factors", np.ones(3)))

    def cancel(self, ctx) -> None:
        if self.edit is not None:
            _current(ctx)
            self.edit.mesh.verts = self.edit_verts0.copy()
            self.edit.mesh.geometry_version += 1
            self.edit.sync(topology=False)
            self._finish_ui(ctx)
            return
        for obj, _start in self.items:
            if obj.kind == "MESH":
                self.engine.set_display_matrix(obj, None)
            else:
                obj.transform.set_matrix(self._originals[obj.uid])
        self._finish_ui(ctx)

    def _finish_ui(self, ctx) -> None:
        self.engine.set_transform_guide(None)
        self.engine.refresh_overlays()
        self.editor.set_overlay_shape(None)
        _hint(ctx, "")
        ctx.app.request_frame()
        ctx.app.notify("transform")


@ops.register
class TransformTranslate(_TransformBase):
    idname = "transform.translate"
    label = "移动"
    description = "移动选中的物体：X/Y/Z 沿轴，Shift+轴在平面里，输入数字给精确距离，Ctrl 吸附"
    icon = "tool.move"
    kind = "TRANSLATE"
    value = FloatVectorProperty("移动", default=(0.0, 0.0, 0.0), size=3, precision=4, unit="m")

    def delta_from_props(self, ctx, obj, pivot, editor) -> np.ndarray:
        orient = self.orient_type
        basis = orientation_basis(ctx, orient, obj if orient in ("LOCAL", "NORMAL") else None, editor)
        local = np.asarray(self.value, np.float64) * _axis_mask(self.constraint)
        m = np.identity(4)
        m[:3, 3] = basis @ local
        return m


@ops.register
class TransformRotate(_TransformBase):
    idname = "transform.rotate"
    label = "旋转"
    description = "旋转选中的物体：X/Y/Z 绕轴，输入数字给精确角度，Ctrl 按步长吸附"
    icon = "tool.rotate"
    kind = "ROTATE"
    value = FloatProperty("角度", default=0.0, precision=2, unit="°")
    orient_axis = EnumProperty("绕", items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", ""), ("VIEW", "视线", "")],
                               default="Z")

    def delta_from_props(self, ctx, obj, pivot, editor) -> np.ndarray:
        axes = constraint_axes(self.constraint)
        if axes:
            axis_index = axes[0] if len(axes) == 1 else [a for a in range(3) if a not in axes][0]
            orient = self.orient_type
            axis = orientation_basis(ctx, orient, obj if orient in ("LOCAL", "NORMAL") else None,
                                     editor)[:, axis_index]
        elif self.orient_axis == "VIEW":
            axis = getattr(self, "_view_axis", None)
            if axis is None and editor is not None and getattr(editor, "view", None) is not None:
                axis = to_display(editor.view.camera.basis()[2])
            if axis is None:
                axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = np.eye(3)["XYZ".index(self.orient_axis)]
        r = _axis_rotation(axis, math.radians(float(self.value)))
        m = np.identity(4)
        m[:3, :3] = r
        m[:3, 3] = pivot - r @ pivot
        return m


@ops.register
class TransformResize(_TransformBase):
    idname = "transform.resize"
    label = "缩放"
    description = "缩放选中的物体：X/Y/Z 沿轴，输入数字给精确倍数，Ctrl 按步长吸附"
    icon = "tool.scale"
    kind = "RESIZE"
    value = FloatVectorProperty("缩放", default=(1.0, 1.0, 1.0), size=3, precision=4)

    def delta_from_props(self, ctx, obj, pivot, editor) -> np.ndarray:
        orient = self.orient_type if self.constraint != "NONE" else "GLOBAL"
        basis = orientation_basis(ctx, orient, obj if orient in ("LOCAL", "NORMAL") else None, editor)
        s = basis @ np.diag(np.asarray(self.value, np.float64)) @ basis.T
        m = np.identity(4)
        m[:3, :3] = s
        m[:3, 3] = pivot - s @ pivot
        return m


@ops.register
class TransformMirror(_SelectedOp):
    idname = "transform.mirror"
    label = "镜像"
    description = "把选中的物体（编辑模式里是选中的点）沿一个轴翻过来（按 X、Y、Z 选轴，回车或左键确认）"
    icon = "mirror"
    redo = True
    constraint = EnumProperty("轴", items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")], default="X")
    orient_type = EnumProperty("坐标系", items=ORIENT_ITEMS, default="GLOBAL")

    @classmethod
    def poll(cls, ctx) -> bool:
        return (object_mode(ctx) and bool(_project(ctx).selected_objects())) or edit_has_selection(ctx)

    def invoke(self, ctx, event) -> str:
        self.chosen = None
        self.orient_type = ctx.app.tool_settings.transform_orientation
        _hint(ctx, "镜像：按 X、Y、Z 选轴（再按一次换坐标系），左键或回车确认，右键或 Esc 取消")
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        if event.value != PRESS:
            return RUNNING_MODAL
        if event.type in AXIS_KEYS:
            axis = event.type
            if self.chosen == axis:
                self.orient_type = "LOCAL" if self.orient_type == "GLOBAL" else "GLOBAL"
            self.chosen = axis
            self.constraint = axis
            _hint(ctx, "镜像：沿%s %s，左键或回车确认" % (ORIENT_LABELS[self.orient_type], axis))
            return RUNNING_MODAL
        if event.type in (LEFTMOUSE, "RET", "NUMPAD_ENTER", "SPACE") and self.chosen:
            _hint(ctx, "")
            return self.execute(ctx)
        if event.type in (RIGHTMOUSE, "ESC"):
            _hint(ctx, "")
            return CANCELLED
        return RUNNING_MODAL

    def execute(self, ctx) -> str:
        if edit_active(ctx):
            return self._execute_edit(ctx)
        project = _project(ctx)
        objects = project.selected_objects()
        settings = ctx.app.tool_settings
        pivot, individual = pivot_points(ctx, objects, settings.transform_pivot_point)
        axis = "XYZ".index(self.constraint)
        changes = []
        for obj in objects:
            start = obj.transform.matrix()
            basis = orientation_basis(ctx, self.orient_type, obj, view3d(ctx))
            s = basis @ np.diag([-1.0 if k == axis else 1.0 for k in range(3)]) @ basis.T
            center = start[:3, 3] if individual else pivot
            m = np.identity(4)
            m[:3, :3] = s
            m[:3, 3] = center - s @ center
            changes.append((obj, start, m @ start))
        _current(ctx)
        apply_object_matrices(ctx, changes, "镜像")
        return FINISHED

    def _execute_edit(self, ctx) -> str:
        from .edit_transform import apply_edit_delta, edit_pivot, to_disp

        session = _engine(ctx).edit
        mesh = session.mesh
        sel = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
        if not len(sel):
            return CANCELLED
        self.edit, self.edit_sel = session, sel
        before = session.begin()
        self.edit_disp = to_disp(mesh.verts)
        self.pivot, self.edit_islands = edit_pivot(self, ctx, mesh, self.edit_disp, sel)
        self.prop_idx, self.prop_weight = None, None
        axis = "XYZ".index(self.constraint)
        basis = orientation_basis(ctx, self.orient_type, session.obj, view3d(ctx))
        s = basis @ np.diag([-1.0 if k == axis else 1.0 for k in range(3)]) @ basis.T
        m = np.identity(4)
        m[:3, :3] = s
        m[:3, 3] = self.pivot - s @ self.pivot
        _current(ctx)
        apply_edit_delta(self, m)
        session.push("镜像", before, topology=False)
        return FINISHED


# ====================================================================== 清除、应用变换
class _ClearTransform(_SelectedOp):
    field = "location"

    def execute(self, ctx) -> str:
        changes = []
        reset = {"location": (0.0, 0.0, 0.0), "rotation": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0)}[self.field]
        for obj in _project(ctx).selected_objects():
            start = obj.transform.matrix()
            values = {"location": obj.transform.location, "rotation": obj.transform.rotation,
                      "scale": obj.transform.scale}
            values[self.field] = reset
            changes.append((obj, start, compose(values["location"], values["rotation"], values["scale"])))
        _current(ctx)
        apply_object_matrices(ctx, changes, self.label)
        return FINISHED


@ops.register
class ObjectLocationClear(_ClearTransform):
    idname = "object.location_clear"
    label = "清除位置"
    description = "选中的物体回到世界原点"
    field = "location"


@ops.register
class ObjectRotationClear(_ClearTransform):
    idname = "object.rotation_clear"
    label = "清除旋转"
    description = "选中的物体转回不旋转的样子"
    field = "rotation"


@ops.register
class ObjectScaleClear(_ClearTransform):
    idname = "object.scale_clear"
    label = "清除缩放"
    description = "选中的物体缩放回 1"
    field = "scale"


def set_transform_keep_geometry(engine, obj, location=None, rotation=None, scale=None) -> None:
    """改物体的变换值，但模型的样子不动（应用变换、设置原点用）。"""
    with obj.transform.changed.block():
        obj.transform.set_values(location=location, rotation=rotation, scale=scale)
    if engine is not None and obj.kind == "MESH":
        engine.mark_transform_applied(obj)
    obj.transform.changed.emit("location")


def shift_geometry(engine, obj, offset_display) -> None:
    """模型整体挪一段（变换值不变）。"""
    location = np.asarray(obj.transform.location, np.float64)
    obj.transform.set_values(location=location + np.asarray(offset_display, np.float64))
    set_transform_keep_geometry(engine, obj, location=location)


@ops.register
class ObjectTransformApply(_SelectedOp):
    idname = "object.transform_apply"
    label = "应用变换"
    description = "把位置、旋转、缩放固定到物体上：样子不变，这几个值归零（缩放归一）"
    redo = True
    location = BoolProperty("位置", default=True)
    rotation = BoolProperty("旋转", default=True)
    scale = BoolProperty("缩放", default=True)

    def execute(self, ctx) -> str:
        engine = _engine(ctx)
        _current(ctx)
        changes = []
        for obj in _project(ctx).selected_objects():
            before = (obj.transform.location, obj.transform.rotation, obj.transform.scale, obj.empty_size
                      if obj.kind == "EMPTY" else None)
            if obj.kind != "MESH":
                # 摄像机、灯光没有模型可固定；空物体应用缩放时把缩放并进显示大小
                if self.scale and obj.kind == "EMPTY":
                    obj.empty_size = float(obj.empty_size) * float(max(abs(v) for v in obj.transform.scale))
                    with obj.transform.changed.block():
                        obj.transform.set_values(scale=(1.0, 1.0, 1.0))
                    obj.transform.changed.emit("scale")
                    changes.append((obj, before, (obj.transform.location, obj.transform.rotation,
                                                  obj.transform.scale, obj.empty_size)))
                continue
            values = {}
            if self.location:
                values["location"] = (0.0, 0.0, 0.0)
            if self.rotation:
                values["rotation"] = (0.0, 0.0, 0.0)
            if self.scale:
                values["scale"] = (1.0, 1.0, 1.0)
            set_transform_keep_geometry(engine, obj, **values)
            changes.append((obj, before, (obj.transform.location, obj.transform.rotation, obj.transform.scale, None)))
        history = _history(ctx)
        if history is not None and changes:
            def restore(index) -> None:
                _current(ctx)
                for item in changes:
                    obj = item[0]
                    loc, rot, scl, size = item[index]
                    if size is not None:
                        obj.empty_size = size
                    set_transform_keep_geometry(engine, obj, location=loc, rotation=rot, scale=scl)
                _redraw(ctx, "transform")

            history.push("应用变换", lambda: restore(1), lambda: restore(2))
        _redraw(ctx, "transform")
        return FINISHED


# ====================================================================== 添加
def new_texture_set(app, name: str):
    from ..doc.project import Layer, TextureSet

    paint = app.prefs.paint
    resolution = getattr(paint, "object_resolution", None) or paint.default_resolution
    ts = TextureSet(name=name, resolution=resolution)
    ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.6, 0.6, 0.62), fill_roughness=0.5))
    ts.add_layer(Layer(name="图层 1"))
    return ts


def unique_name(project, base: str) -> str:
    names = {o.name for o in project.all_objects()}
    if base not in names:
        return base
    index = 1
    while "%s.%03d" % (base, index) in names:
        index += 1
    return "%s.%03d" % (base, index)


def add_objects_with_undo(ctx, objs: list, texture_sets: list, label: str, select: bool = True) -> None:
    """把新物体（和它们新建的纹理集）放进场景，选中，记一步撤销。"""
    project = _project(ctx)
    engine = _engine(ctx)
    before = selection_state(project)

    def attach() -> None:
        _current(ctx)
        for ts in texture_sets:
            if ts not in project.texture_sets:
                project.add_texture_set(ts)
                if engine is not None:
                    engine.add_texture_set(ts)
        for obj in objs:
            if obj.kind == "MESH":
                if obj not in project.objects:
                    project.add_object(obj)
                if engine is not None:
                    engine.add_mesh_object(obj)
            elif obj not in project.scene_objects:
                project.add_scene_object(obj)
        if select:
            select_only(project, objs, objs[-1] if objs else None)
        _redraw(ctx, "objects")

    def detach() -> None:
        _current(ctx)
        for obj in objs:
            project.remove_object(obj)
            if obj.kind == "MESH" and engine is not None:
                engine.remove_mesh_object(obj)
        for ts in texture_sets:
            if ts in project.texture_sets:
                project.remove_texture_set(ts)
                if engine is not None:
                    engine.remove_texture_set(ts)
        apply_selection(project, before)
        _redraw(ctx, "objects")

    attach()
    history = _history(ctx)
    if history is not None:
        history.push(label, detach, attach)


ALIGN_ITEMS = [("WORLD", "世界", "和世界坐标轴对齐"), ("VIEW", "视图", "正面朝向当前视角")]


class _AddOp(_ObjectOp):
    redo = True
    align = EnumProperty("对齐", items=ALIGN_ITEMS, default="WORLD")
    location = FloatVectorProperty("位置", default=(0.0, 0.0, 0.0), size=3, precision=4, unit="m")
    rotation = FloatVectorProperty("旋转", default=(0.0, 0.0, 0.0), size=3, precision=2, unit="°")

    def invoke(self, ctx, event) -> str:
        project = _project(ctx)
        self.location = tuple(float(v) for v in project.cursor_location)
        if self.align == "VIEW":
            editor = view3d(ctx)
            if editor is not None:
                right, up, back = editor.view.camera.basis()
                m = np.stack([to_display(right), to_display(up), to_display(back)], axis=1)
                self.rotation = tuple(float(v) for v in euler_from_matrix(m))
        return self.execute(ctx)

    def _place(self, obj) -> None:
        obj.transform.location = tuple(float(v) for v in self.location)
        obj.transform.rotation = tuple(float(v) for v in self.rotation)


def primitive_object(ctx, op, kind: str, mesh):
    from ..doc.project import MeshObject
    from ..geometry.primitives import PRIMITIVES

    project = _project(ctx)
    label = PRIMITIVES[kind][0]
    name = unique_name(project, label)
    mesh.name = name
    m = np.identity(4)
    m[:3, :3] = rotation_matrix(op.rotation)
    m[:3, 3] = np.asarray(op.location, np.float64)
    internal = display_to_internal_matrix(m)
    positions = mesh.positions.astype(np.float64) @ internal[:3, :3].T + internal[:3, 3]
    normals = mesh.normals.astype(np.float64) @ internal[:3, :3].T
    mesh.positions = positions.astype(np.float32)
    mesh.normals = (normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)).astype(np.float32)
    mesh.bounds_min = mesh.positions.min(axis=0)
    mesh.bounds_max = mesh.positions.max(axis=0)
    ts = new_texture_set(ctx.app, name)
    obj = MeshObject(name, mesh, [ts.uid])
    obj.geometry_dirty = True
    op._place(obj)
    add_objects_with_undo(ctx, [obj], [ts], "添加%s" % label)
    return obj


def primitive_into_edit(ctx, op, kind: str, mesh) -> None:
    """编辑模式里加形体：放在 3D 游标处，接进正在编辑的网格，只选中新加的部分。"""
    from ..geometry.primitives import PRIMITIVES
    from ..mesh import topology as T
    from ..mesh.editmesh import from_mesh_data

    session = _engine(ctx).edit
    m = np.identity(4)
    m[:3, :3] = rotation_matrix(op.rotation)
    m[:3, 3] = np.asarray(op.location, np.float64)
    internal = display_to_internal_matrix(m)
    mesh.positions = (mesh.positions.astype(np.float64) @ internal[:3, :3].T + internal[:3, 3]).astype(np.float32)
    other = from_mesh_data(mesh)
    _current(ctx)
    before = session.begin()
    T.append(session.mesh, other)
    session.push("添加%s" % PRIMITIVES[kind][0], before)
    _redraw(ctx, "edit")


def _primitive_op(kind: str, idname: str, label: str, icon: str, props: dict, build):
    def execute(self, ctx) -> str:
        if edit_active(ctx):
            primitive_into_edit(ctx, self, kind, build(self))
        else:
            primitive_object(ctx, self, kind, build(self))
        return FINISHED

    def poll(cls, ctx) -> bool:
        return object_mode(ctx) or edit_active(ctx)

    attrs = {"idname": idname, "label": label, "description": "在 3D 游标处添加%s" % label, "icon": icon,
             "execute": execute, "poll": classmethod(poll), "__module__": __name__}
    attrs.update(props)
    return ops.register(type("MeshPrimitive_" + kind, (_AddOp,), attrs))


def _prims():
    from ..geometry import primitives

    return primitives


_SIZE = {"size": FloatProperty("尺寸", default=2.0, min=0.0001, max=100000.0, precision=3, unit="m")}
_primitive_op("PLANE", "mesh.primitive_plane_add", "平面", "prim.plane", dict(_SIZE),
              lambda op: _prims().plane(op.size))
_primitive_op("CUBE", "mesh.primitive_cube_add", "立方体", "prim.cube",
              {"size": FloatProperty("尺寸", default=2.0, min=0.0001, max=100000.0, precision=3, unit="m")},
              lambda op: _prims().cube(op.size))
_primitive_op("CIRCLE", "mesh.primitive_circle_add", "圆", "prim.circle",
              {"vertices": IntProperty("顶点", default=32, min=3, max=10000),
               "radius": FloatProperty("半径", default=1.0, min=0.0001, max=100000.0, precision=3, unit="m")},
              lambda op: _prims().circle(op.vertices, op.radius))
_primitive_op("UV_SPHERE", "mesh.primitive_uv_sphere_add", "经纬球", "prim.uv_sphere",
              {"segments": IntProperty("段数", default=32, min=3, max=1000),
               "ring_count": IntProperty("环数", default=16, min=3, max=1000),
               "radius": FloatProperty("半径", default=1.0, min=0.0001, max=100000.0, precision=3, unit="m")},
              lambda op: _prims().uv_sphere(op.segments, op.ring_count, op.radius))
_primitive_op("ICO_SPHERE", "mesh.primitive_ico_sphere_add", "棱角球", "prim.ico_sphere",
              {"subdivisions": IntProperty("细分", default=2, min=1, max=8),
               "radius": FloatProperty("半径", default=1.0, min=0.0001, max=100000.0, precision=3, unit="m")},
              lambda op: _prims().ico_sphere(op.subdivisions, op.radius))
_primitive_op("CYLINDER", "mesh.primitive_cylinder_add", "柱体", "prim.cylinder",
              {"vertices": IntProperty("顶点", default=32, min=3, max=10000),
               "radius": FloatProperty("半径", default=1.0, min=0.0001, max=100000.0, precision=3, unit="m"),
               "depth": FloatProperty("深度", default=2.0, min=0.0001, max=100000.0, precision=3, unit="m")},
              lambda op: _prims().cylinder(op.vertices, op.radius, op.depth))
_primitive_op("CONE", "mesh.primitive_cone_add", "锥体", "prim.cone",
              {"vertices": IntProperty("顶点", default=32, min=3, max=10000),
               "radius1": FloatProperty("底面半径", default=1.0, min=0.0, max=100000.0, precision=3, unit="m"),
               "radius2": FloatProperty("顶面半径", default=0.0, min=0.0, max=100000.0, precision=3, unit="m"),
               "depth": FloatProperty("深度", default=2.0, min=0.0001, max=100000.0, precision=3, unit="m")},
              lambda op: _prims().cone(op.vertices, op.radius1, op.radius2, op.depth))
_primitive_op("TORUS", "mesh.primitive_torus_add", "环体", "prim.torus",
              {"major_segments": IntProperty("主环段数", default=48, min=3, max=1000),
               "minor_segments": IntProperty("截面段数", default=12, min=3, max=1000),
               "major_radius": FloatProperty("主半径", default=1.0, min=0.0001, max=100000.0, precision=3, unit="m"),
               "minor_radius": FloatProperty("截面半径", default=0.25, min=0.0001, max=100000.0, precision=3,
                                             unit="m")},
              lambda op: _prims().torus(op.major_segments, op.minor_segments, op.major_radius, op.minor_radius))
_primitive_op("GRID", "mesh.primitive_grid_add", "栅格", "prim.grid",
              {"x_subdivisions": IntProperty("X 细分", default=10, min=1, max=10000),
               "y_subdivisions": IntProperty("Y 细分", default=10, min=1, max=10000),
               "size": FloatProperty("尺寸", default=2.0, min=0.0001, max=100000.0, precision=3, unit="m")},
              lambda op: _prims().grid(op.x_subdivisions, op.y_subdivisions, op.size))


@ops.register
class ObjectCameraAdd(_AddOp):
    idname = "object.camera_add"
    label = "摄像机"
    description = "在 3D 游标处添加一台摄像机"
    icon = "object.camera"

    def execute(self, ctx) -> str:
        project = _project(ctx)
        obj = SceneObject("CAMERA", unique_name(project, "摄像机"))
        self._place(obj)
        add_objects_with_undo(ctx, [obj], [], "添加摄像机")
        return FINISHED


@ops.register
class ObjectLightAdd(_AddOp):
    idname = "object.light_add"
    label = "灯光"
    description = "在 3D 游标处添加一盏灯"
    icon = "object.light"
    type = EnumProperty("类型", items=[("POINT", "点光", "", "light.point"), ("SUN", "日光", "", "light.sun"),
                                     ("SPOT", "聚光", "", "light.spot"), ("AREA", "面光", "", "light.area")],
                        default="POINT")

    def execute(self, ctx) -> str:
        project = _project(ctx)
        names = {"POINT": "点光", "SUN": "日光", "SPOT": "聚光", "AREA": "面光"}
        obj = SceneObject("LIGHT", unique_name(project, names[self.type]))
        obj.data.type = self.type
        self._place(obj)
        add_objects_with_undo(ctx, [obj], [], "添加%s" % names[self.type])
        return FINISHED


from ..doc.objects import EMPTY_DISPLAY_ITEMS as EMPTY_TYPES  # noqa: E402


@ops.register
class ObjectEmptyAdd(_AddOp):
    idname = "object.empty_add"
    label = "空物体"
    description = "在 3D 游标处添加一个空物体（只有线框，做参照、做轴心用）"
    icon = "object.empty"
    type = EnumProperty("显示为", items=EMPTY_TYPES, default="PLAIN_AXES")
    radius = FloatProperty("大小", default=1.0, min=0.0001, max=100000.0, precision=3, unit="m")

    def execute(self, ctx) -> str:
        project = _project(ctx)
        obj = SceneObject("EMPTY", unique_name(project, "空物体"))
        obj.empty_display_type = self.type
        obj.empty_size = float(self.radius)
        self._place(obj)
        add_objects_with_undo(ctx, [obj], [], "添加空物体")
        return FINISHED


# ====================================================================== 删除、复制、隐藏
@ops.register
class ObjectDelete(_SelectedOp):
    idname = "object.delete"
    label = "删除"
    description = "删除选中的物体（它们的材质留在工程里）"
    icon = "remove"
    confirm = BoolProperty("先确认", default=True)

    def invoke(self, ctx, event) -> str:
        if self.confirm:
            from PySide6.QtGui import QCursor
            from PySide6.QtWidgets import QMenu

            menu = QMenu()
            menu.addAction("删除 %d 个物体？" % len(_project(ctx).selected_objects())).setEnabled(False)
            action = menu.addAction("删除")
            chosen = menu.exec(QCursor.pos())
            if chosen is not action:
                return CANCELLED
        return self.execute(ctx)

    def execute(self, ctx) -> str:
        project = _project(ctx)
        engine = _engine(ctx)
        victims = list(project.selected_objects())
        if not victims:
            return CANCELLED
        before = selection_state(project)
        _current(ctx)
        places = {}
        for obj in victims:
            places[obj.uid] = project.remove_object(obj)
            if obj.kind == "MESH" and engine is not None:
                engine.remove_mesh_object(obj)
        project.selection_changed()

        def undo() -> None:
            _current(ctx)
            for obj in reversed(victims):
                index = places[obj.uid]
                if obj.kind == "MESH":
                    project.add_object(obj, index)
                    if engine is not None:
                        engine.add_mesh_object(obj)
                else:
                    project.add_scene_object(obj, index)
            apply_selection(project, before)
            _redraw(ctx, "objects")

        def redo() -> None:
            _current(ctx)
            for obj in victims:
                project.remove_object(obj)
                if obj.kind == "MESH" and engine is not None:
                    engine.remove_mesh_object(obj)
            project.selection_changed()
            _redraw(ctx, "objects")

        history = _history(ctx)
        if history is not None:
            history.push("删除", undo, redo)
        _redraw(ctx, "objects")
        self.report(ctx, "已删除 %d 个物体" % len(victims))
        return FINISHED


def duplicate_object(project, obj):
    from ..doc.project import MeshObject

    if obj.kind == "MESH":
        data = copy.copy(obj.data)
        for field in ("positions", "normals", "uvs", "material_ids"):
            setattr(data, field, np.array(getattr(obj.data, field), copy=True))
        sizes = getattr(obj.data, "polygon_sizes", None)
        if sizes is not None:
            data.polygon_sizes = np.array(sizes, copy=True)
        data.materials = list(obj.data.materials)
        dup = MeshObject(unique_name(project, obj.name), data, list(obj.material_sets))
        dup.transform.from_dict(obj.transform.to_dict())
        dup.geometry_dirty = True
    else:
        dup = SceneObject.from_dict(dict(obj.to_dict(), uid=0))
        dup.name = unique_name(project, obj.name)
    return dup


@ops.register
class ObjectDuplicateMove(_SelectedOp):
    idname = "object.duplicate_move"
    label = "复制物体"
    description = "复制选中的物体并跟着鼠标移动（复制出的模型和原来的共用材质）"
    icon = "duplicate"
    linked = BoolProperty("关联复制", default=False, hidden=True)

    def invoke(self, ctx, event) -> str:
        project = _project(ctx)
        copies = [duplicate_object(project, obj) for obj in project.selected_objects()]
        add_objects_with_undo(ctx, copies, [], "复制物体")
        if view3d(ctx) is not None and ctx.app.tool_settings.mode == "OBJECT":
            # 接着移动（移动自己进模态，确认后和「复制」合成一步撤销）
            ops.call("transform.translate", ctx, event, merge_previous=1)
        return FINISHED

    def execute(self, ctx) -> str:
        project = _project(ctx)
        copies = [duplicate_object(project, obj) for obj in project.selected_objects()]
        add_objects_with_undo(ctx, copies, [], "复制物体")
        return FINISHED


@ops.register
class ObjectHideViewSet(_ObjectOp):
    idname = "object.hide_view_set"
    label = "隐藏所选"
    description = "在视口里隐藏选中的物体"
    unselected = BoolProperty("隐藏没选中的", default=False)

    def execute(self, ctx) -> str:
        project = _project(ctx)
        targets = [o for o in project.all_objects() if o.visible and (bool(o.select) != bool(self.unselected))]
        if not targets:
            return CANCELLED
        set_visibility(ctx, targets, False, "隐藏未选" if self.unselected else "隐藏")
        return FINISHED


@ops.register
class ObjectHideViewClear(_ObjectOp):
    idname = "object.hide_view_clear"
    label = "显示隐藏的物体"
    description = "把隐藏的物体都显示出来并选中"
    select = BoolProperty("选中它们", default=True)

    def execute(self, ctx) -> str:
        project = _project(ctx)
        targets = [o for o in project.all_objects() if not o.visible]
        if not targets:
            return CANCELLED
        set_visibility(ctx, targets, True, "显示隐藏的物体", select=bool(self.select))
        return FINISHED


def set_visibility(ctx, targets: list, visible: bool, label: str, select: bool = False) -> None:
    project = _project(ctx)
    engine = _engine(ctx)
    before = selection_state(project)

    def apply(value: bool) -> None:
        _current(ctx)
        for obj in targets:
            obj.visible = value
            if not value:
                obj.select = False
            elif select:
                obj.select = True
        if engine is not None:
            engine._rebuild_scene()
        project.mark_dirty()
        project.changed.emit("objects")
        project.selection_changed()
        _redraw(ctx, "objects")

    apply(visible)
    after = selection_state(project)
    history = _history(ctx)
    if history is not None:
        history.push(label, lambda: (apply(not visible), apply_selection(project, before)),
                     lambda: (apply(visible), apply_selection(project, after)))


# ====================================================================== 3D 游标、吸附
def set_cursor(ctx, location, label: str = "") -> None:
    project = _project(ctx)
    before = tuple(project.cursor_location)
    after = tuple(float(v) for v in location)
    project.cursor_location = after
    _redraw(ctx, "cursor")
    history = _history(ctx)
    if label and history is not None and before != after:
        def put(value) -> None:
            project.cursor_location = value
            _redraw(ctx, "cursor")

        history.push(label, lambda: put(before), lambda: put(after))


@ops.register
class View3DCursor3D(Operator):
    idname = "view3d.cursor3d"
    label = "放置 3D 游标"
    description = "把 3D 游标放到鼠标下的表面上（新加的物体会放在这里）"
    searchable = False

    @classmethod
    def poll(cls, ctx) -> bool:
        return _project(ctx) is not None and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        project = _project(ctx)
        if event is None:
            return CANCELLED
        _current(ctx)
        x, y = editor.pixel(event)
        view = editor.view
        hit = view.surface_at(x, y)
        if hit is not None:
            point = np.asarray(hit[0], np.float64)
        else:
            origin, direction = view.camera.ray(x, y, view.width, view.height)
            current = to_internal(np.asarray(project.cursor_location, np.float64))
            forward = view.camera.forward
            depth = float((current - origin) @ forward)
            denom = float(direction @ forward)
            point = origin + direction * (depth / denom if abs(denom) > 1e-9 else 1.0)
        set_cursor(ctx, to_display(point), "放置 3D 游标")
        return FINISHED


class _SnapOp(Operator):
    @classmethod
    def poll(cls, ctx) -> bool:
        return _project(ctx) is not None


def _edit_selected_points(ctx):
    """编辑模式里选中的点（编号，显示坐标）。"""
    from .edit_transform import to_disp

    mesh = _engine(ctx).edit.mesh
    sel = np.nonzero(mesh.vert_sel & ~mesh.vert_hide)[0]
    return sel, to_disp(mesh.verts[sel])


def _edit_active_point(ctx):
    """编辑模式里当前元素的位置（显示坐标），没有当前元素时返回 None。"""
    from .edit_transform import to_disp

    mesh = _engine(ctx).edit.mesh
    if mesh.active is None:
        return None
    kind, index = mesh.active
    if kind == "VERT":
        return to_disp(mesh.verts[index])
    if kind == "EDGE":
        a, b = mesh.edges().edges[index]
        return to_disp((mesh.verts[a] + mesh.verts[b]) * 0.5)
    return to_disp(mesh.face_centers()[index])


def _edit_move_selected(ctx, label: str, compute) -> str:
    """编辑模式里把选中的点挪到 compute(显示坐标) 给的新位置，记一步撤销。"""
    from .edit_transform import to_int

    session = _engine(ctx).edit
    mesh = session.mesh
    sel, disp = _edit_selected_points(ctx)
    if not len(sel):
        return CANCELLED
    _current(ctx)
    before = session.begin()
    verts = mesh.verts.copy()
    verts[sel] = to_int(np.asarray(compute(disp), np.float64).reshape(-1, 3))
    mesh.verts = verts
    mesh.geometry_version += 1
    session.push(label, before, topology=False)
    _redraw(ctx, "edit")
    return FINISHED


def _grid_step(ctx) -> float:
    editor = view3d(ctx)
    overlay = getattr(getattr(editor, "view", None), "overlay", None)
    return float(getattr(overlay, "grid_scale", 1.0) or 1.0)


@ops.register
class View3DSnapCursorToCenter(_SnapOp):
    idname = "view3d.snap_cursor_to_center"
    label = "游标 → 世界原点"
    icon = "pivot.cursor"

    def execute(self, ctx) -> str:
        set_cursor(ctx, (0.0, 0.0, 0.0), self.label)
        return FINISHED


@ops.register
class View3DSnapCursorToGrid(_SnapOp):
    idname = "view3d.snap_cursor_to_grid"
    label = "游标 → 网格点"
    icon = "pivot.cursor"

    def execute(self, ctx) -> str:
        step = _grid_step(ctx)
        location = np.round(np.asarray(_project(ctx).cursor_location, np.float64) / step) * step
        set_cursor(ctx, location, self.label)
        return FINISHED


@ops.register
class View3DSnapCursorToSelected(_SnapOp):
    idname = "view3d.snap_cursor_to_selected"
    label = "游标 → 选中项"
    icon = "pivot.cursor"

    @classmethod
    def poll(cls, ctx) -> bool:
        if edit_active(ctx):
            return edit_has_selection(ctx)
        return _project(ctx) is not None and bool(_project(ctx).selected_objects())

    def execute(self, ctx) -> str:
        if edit_active(ctx):
            _sel, disp = _edit_selected_points(ctx)
            mode = ctx.app.tool_settings.transform_pivot_point
            center = (disp.min(axis=0) + disp.max(axis=0)) * 0.5 if mode == "BOUNDING_BOX_CENTER" else disp.mean(axis=0)
            set_cursor(ctx, center, self.label)
            return FINISHED
        objects = _project(ctx).selected_objects()
        mode = ctx.app.tool_settings.transform_pivot_point
        pivot, _individual = pivot_points(ctx, objects, mode if mode != "CURSOR" else "MEDIAN_POINT")
        set_cursor(ctx, pivot, self.label)
        return FINISHED


@ops.register
class View3DSnapCursorToActive(_SnapOp):
    idname = "view3d.snap_cursor_to_active"
    label = "游标 → 活动项"
    icon = "pivot.cursor"

    @classmethod
    def poll(cls, ctx) -> bool:
        if edit_active(ctx):
            return edit_has_selection(ctx) or _engine(ctx).edit.mesh.active is not None
        return _project(ctx) is not None and _project(ctx).active_object is not None

    def execute(self, ctx) -> str:
        if edit_active(ctx):
            point = _edit_active_point(ctx)
            if point is None:
                point = _edit_selected_points(ctx)[1].mean(axis=0)
            set_cursor(ctx, point, self.label)
            return FINISHED
        set_cursor(ctx, _project(ctx).active_object.transform.location, self.label)
        return FINISHED


@ops.register
class View3DSnapSelectedToCursor(_SelectedOp):
    idname = "view3d.snap_selected_to_cursor"
    label = "选中项 → 游标"
    description = "把选中的物体（编辑模式里是选中的点）移到 3D 游标处"
    icon = "pivot.cursor"
    use_offset = BoolProperty("保持相对位置", default=False, description="整体移过去，相互之间的位置不变")

    @classmethod
    def poll(cls, ctx) -> bool:
        return (object_mode(ctx) and bool(_project(ctx).selected_objects())) or edit_has_selection(ctx)

    def execute(self, ctx) -> str:
        project = _project(ctx)
        if edit_active(ctx):
            cursor = np.asarray(project.cursor_location, np.float64)
            if self.use_offset:
                return _edit_move_selected(ctx, self.label, lambda p: p + (cursor - p.mean(axis=0)))
            return _edit_move_selected(ctx, self.label, lambda p: np.repeat(cursor[None], len(p), axis=0))
        objects = project.selected_objects()
        cursor = np.asarray(project.cursor_location, np.float64)
        center, _ = pivot_points(ctx, objects, "MEDIAN_POINT")
        changes = []
        for obj in objects:
            start = obj.transform.matrix()
            new = start.copy()
            new[:3, 3] = start[:3, 3] + (cursor - center) if self.use_offset else cursor
            changes.append((obj, start, new))
        _current(ctx)
        apply_object_matrices(ctx, changes, "选中项 → 游标")
        return FINISHED


@ops.register
class View3DSnapSelectedToGrid(_SelectedOp):
    idname = "view3d.snap_selected_to_grid"
    label = "选中项 → 网格点"
    icon = "grid"

    @classmethod
    def poll(cls, ctx) -> bool:
        return (object_mode(ctx) and bool(_project(ctx).selected_objects())) or edit_has_selection(ctx)

    def execute(self, ctx) -> str:
        step = _grid_step(ctx)
        if edit_active(ctx):
            return _edit_move_selected(ctx, self.label, lambda p: np.round(p / step) * step)
        changes = []
        for obj in _project(ctx).selected_objects():
            start = obj.transform.matrix()
            new = start.copy()
            new[:3, 3] = np.round(start[:3, 3] / step) * step
            changes.append((obj, start, new))
        _current(ctx)
        apply_object_matrices(ctx, changes, self.label)
        return FINISHED


@ops.register
class View3DSnapSelectedToActive(_SelectedOp):
    idname = "view3d.snap_selected_to_active"
    label = "选中项 → 活动项"
    icon = "pivot.active"

    @classmethod
    def poll(cls, ctx) -> bool:
        project = _project(ctx)
        if edit_active(ctx):
            return edit_has_selection(ctx) and _engine(ctx).edit.mesh.active is not None
        return object_mode(ctx) and project.active_object is not None and len(project.selected_objects()) > 1

    def execute(self, ctx) -> str:
        project = _project(ctx)
        if edit_active(ctx):
            point = _edit_active_point(ctx)
            if point is None:
                return CANCELLED
            return _edit_move_selected(ctx, self.label, lambda p: np.repeat(point[None], len(p), axis=0))
        target = np.asarray(project.active_object.transform.location, np.float64)
        changes = []
        for obj in project.selected_objects():
            if obj is project.active_object:
                continue
            start = obj.transform.matrix()
            new = start.copy()
            new[:3, 3] = target
            changes.append((obj, start, new))
        _current(ctx)
        apply_object_matrices(ctx, changes, self.label)
        return FINISHED


@ops.register
class TextureSetActivate(Operator):
    idname = "project.texture_set_activate"
    label = "设为当前纹理集"
    description = "之后画的、图层编辑器里显示的都是这套贴图"
    searchable = False
    uid = IntProperty("纹理集", default=0)

    @classmethod
    def poll(cls, ctx) -> bool:
        return _project(ctx) is not None

    def execute(self, ctx) -> str:
        project = _project(ctx)
        ts = project.texture_set(int(self.uid))
        if ts is None:
            return CANCELLED
        project.set_active_set(ts)
        ctx.app.notify("layers")
        return FINISHED


# ====================================================================== 着色、原点、合并、关联
def smooth_normals(positions: np.ndarray) -> np.ndarray:
    from ..doc.meshio import _angle_weighted_normals, _weld_bits

    ids, count = _weld_bits(positions)
    return _angle_weighted_normals(np.ascontiguousarray(positions, np.float32), ids, count)


def flat_normals(positions: np.ndarray) -> np.ndarray:
    tris = np.asarray(positions, np.float64).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return np.repeat(n, 3, axis=0).astype(np.float32)


def angle_normals(positions: np.ndarray, angle_deg: float) -> np.ndarray:
    """按角度平滑：夹角小于 angle 的相邻面一起平滑，大于的保持锐利。"""
    from ..doc.meshio import _weld_bits
    from ..geometry.normals import angle_smooth

    ids, count = _weld_bits(positions)
    return angle_smooth(np.ascontiguousarray(positions, np.float32), ids.astype(np.int64), int(count),
                        math.cos(math.radians(float(angle_deg))))


class _ShadeOp(_ObjectOp):
    redo = True

    @classmethod
    def poll(cls, ctx) -> bool:
        return object_mode(ctx) and any(o.kind == "MESH" for o in _project(ctx).selected_objects())

    def normals_for(self, positions: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def execute(self, ctx) -> str:
        engine = _engine(ctx)
        _current(ctx)
        changes = []
        for obj in _meshes(_project(ctx).selected_objects()):
            before = obj.data.normals
            after = np.ascontiguousarray(self.normals_for(obj.data.positions), np.float32)
            changes.append((obj, before, after))

        def apply(index) -> None:
            _current(ctx)
            for item in changes:
                obj = item[0]
                obj.data.normals = item[index]
                obj.geometry_dirty = True
                mesh = engine.meshes.get(obj.uid) if engine is not None else None
                if mesh is not None:
                    mesh.update_vertices(obj.data)
            if engine is not None:
                for view in engine.views:
                    view.dirty = True
                    view.gbuf_valid = False
                engine._pending_work = True
            ctx.app.request_frame()

        apply(2)
        history = _history(ctx)
        if history is not None and changes:
            history.push(self.label, lambda: apply(1), lambda: apply(2))
        return FINISHED


@ops.register
class ObjectShadeSmooth(_ShadeOp):
    idname = "object.shade_smooth"
    label = "平滑着色"
    description = "按相邻面的平均朝向显示，表面看起来光滑"
    icon = "shade.smooth"

    def normals_for(self, positions):
        return smooth_normals(positions)


@ops.register
class ObjectShadeFlat(_ShadeOp):
    idname = "object.shade_flat"
    label = "平直着色"
    description = "每个面一个朝向，棱角分明"
    icon = "shade.flat"

    def normals_for(self, positions):
        return flat_normals(positions)


@ops.register
class ObjectShadeSmoothByAngle(_ShadeOp):
    idname = "object.shade_smooth_by_angle"
    label = "按角度平滑着色"
    description = "相邻面夹角小于这个角度的平滑，大于的保持锐利的棱"
    icon = "shade.smooth"
    angle = FloatProperty("角度", default=30.0, min=0.0, max=180.0, precision=1, unit="°")

    def normals_for(self, positions):
        return angle_normals(positions, self.angle)


@ops.register
class ObjectOriginSet(_ObjectOp):
    idname = "object.origin_set"
    label = "设置原点"
    description = "移动物体的原点（模型不动），或者把模型移到原点上"
    redo = True
    type = EnumProperty("方式", items=[("GEOMETRY_ORIGIN", "几何中心 → 原点", "模型移过来，中心对到原点上"),
                                     ("ORIGIN_GEOMETRY", "原点 → 几何中心", ""),
                                     ("ORIGIN_CURSOR", "原点 → 3D 游标", ""),
                                     ("ORIGIN_CENTER_OF_MASS", "原点 → 质心（表面）", "按表面面积算的重心")],
                        default="ORIGIN_GEOMETRY")
    center = EnumProperty("中心", items=[("MEDIAN", "中位点", "所有点的平均位置"),
                                       ("BOUNDS", "边界框中心", "")], default="MEDIAN")

    @classmethod
    def poll(cls, ctx) -> bool:
        return object_mode(ctx) and any(o.kind == "MESH" for o in _project(ctx).selected_objects())

    def _center_of(self, obj) -> np.ndarray:
        positions = np.asarray(obj.data.positions, np.float64)
        if self.type == "ORIGIN_CENTER_OF_MASS":
            tris = positions.reshape(-1, 3, 3)
            area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
            total = float(area.sum())
            centers = tris.mean(axis=1)
            internal = (centers * area[:, None]).sum(axis=0) / total if total > 0 else centers.mean(axis=0)
        elif self.center == "BOUNDS":
            internal = (np.asarray(obj.data.bounds_min, np.float64) + np.asarray(obj.data.bounds_max, np.float64)) * 0.5
        else:
            from ..sculpt.topology import weld

            internal = weld(obj.data.positions, None, None, tolerance=1e-6).vertices.astype(np.float64).mean(axis=0)
        return to_display(internal)

    def execute(self, ctx) -> str:
        engine = _engine(ctx)
        project = _project(ctx)
        _current(ctx)
        steps = []
        for obj in _meshes(project.selected_objects()):
            origin = np.asarray(obj.transform.location, np.float64)
            if self.type == "GEOMETRY_ORIGIN":
                offset = origin - self._center_of(obj)
                shift_geometry(engine, obj, offset)
                steps.append(("shift", obj, offset))
            else:
                target = np.asarray(project.cursor_location, np.float64) if self.type == "ORIGIN_CURSOR" \
                    else self._center_of(obj)
                set_transform_keep_geometry(engine, obj, location=target)
                steps.append(("origin", obj, (origin, target)))
        history = _history(ctx)
        if history is not None and steps:
            def run(forward: bool) -> None:
                _current(ctx)
                for kind, obj, payload in (steps if forward else list(reversed(steps))):
                    if kind == "shift":
                        shift_geometry(engine, obj, payload if forward else -payload)
                    else:
                        set_transform_keep_geometry(engine, obj, location=payload[1] if forward else payload[0])
                _redraw(ctx, "transform")

            history.push("设置原点", lambda: run(False), lambda: run(True))
        _redraw(ctx, "transform")
        return FINISHED


@ops.register
class ObjectJoin(_ObjectOp):
    idname = "object.join"
    label = "合并"
    description = "把选中的模型合并进当前模型（材质都保留）"
    icon = "join"

    @classmethod
    def poll(cls, ctx) -> bool:
        project = _project(ctx)
        if not object_mode(ctx):
            return False
        meshes = _meshes(project.selected_objects())
        active = project.active_object
        return len(meshes) >= 2 and active is not None and active.kind == "MESH" and active.select

    def execute(self, ctx) -> str:
        project = _project(ctx)
        engine = _engine(ctx)
        _current(ctx)
        target = project.active_object
        others = [o for o in _meshes(project.selected_objects()) if o is not target]
        old_data = target.data
        old_sets = list(target.material_sets)
        sets = list(target.material_sets)
        positions, normals, uvs = [old_data.positions], [old_data.normals], [old_data.uvs]
        mats = [np.asarray(old_data.material_ids, np.int32)]
        materials = list(old_data.materials)
        sizes = [getattr(old_data, "polygon_sizes", None)]
        for obj in others:
            remap = []
            for index, set_uid in enumerate(obj.material_sets):
                if set_uid in sets:
                    remap.append(sets.index(set_uid))
                else:
                    sets.append(set_uid)
                    materials.append(obj.data.materials[index] if index < len(obj.data.materials) else "材质")
                    remap.append(len(sets) - 1)
            if not remap:
                remap = [0]
            positions.append(obj.data.positions)
            normals.append(obj.data.normals)
            uvs.append(obj.data.uvs)
            ids = np.clip(np.asarray(obj.data.material_ids, np.int64), 0, len(remap) - 1)
            mats.append(np.asarray(remap, np.int32)[ids])
            sizes.append(getattr(obj.data, "polygon_sizes", None))
        new_data = copy.copy(old_data)
        new_data.positions = np.concatenate(positions).astype(np.float32)
        new_data.normals = np.concatenate(normals).astype(np.float32)
        new_data.uvs = np.concatenate(uvs).astype(np.float32)
        new_data.material_ids = np.concatenate(mats).astype(np.int32)
        new_data.materials = materials
        new_data.bounds_min = new_data.positions.min(axis=0)
        new_data.bounds_max = new_data.positions.max(axis=0)
        new_data.has_uvs = all(getattr(o.data, "has_uvs", True) for o in [target] + others)
        new_data.polygon_sizes = np.concatenate(sizes).astype(np.int32) if all(s is not None for s in sizes) else None
        places = {o.uid: project.objects.index(o) for o in others}
        before_sel = selection_state(project)

        def apply(joined: bool) -> None:
            _current(ctx)
            target.data = new_data if joined else old_data
            target.material_sets = sets if joined else old_sets
            target.geometry_dirty = True
            for obj in others:
                if joined and obj in project.objects:
                    project.remove_object(obj)
                    engine.remove_mesh_object(obj)
                elif not joined and obj not in project.objects:
                    project.add_object(obj, places[obj.uid])
                    engine.add_mesh_object(obj)
            engine.add_mesh_object(target)
            if joined:
                select_only(project, [target], target)
            else:
                apply_selection(project, before_sel)
            project.changed.emit("objects")
            _redraw(ctx, "objects")

        apply(True)
        history = _history(ctx)
        if history is not None:
            history.push("合并", lambda: apply(False), lambda: apply(True))
        self.report(ctx, "已合并 %d 个模型" % (len(others) + 1))
        return FINISHED


@ops.register
class ObjectMakeLinksData(_ObjectOp):
    idname = "object.make_links_data"
    label = "关联材质"
    description = "让选中的模型都用当前模型的材质（纹理集）"
    type = EnumProperty("关联", items=[("MATERIAL", "材质", "")], default="MATERIAL")

    @classmethod
    def poll(cls, ctx) -> bool:
        project = _project(ctx)
        active = project.active_object if project is not None else None
        return object_mode(ctx) and active is not None and active.kind == "MESH" and \
            len(_meshes(project.selected_objects())) > 1

    def execute(self, ctx) -> str:
        project = _project(ctx)
        engine = _engine(ctx)
        _current(ctx)
        source = project.active_object
        targets = [o for o in _meshes(project.selected_objects()) if o is not source]
        before = {o.uid: (list(o.material_sets), np.array(o.data.material_ids, copy=True), list(o.data.materials))
                  for o in targets}
        first = source.material_sets[0] if source.material_sets else 0

        def apply(linked: bool) -> None:
            _current(ctx)
            for obj in targets:
                if linked:
                    obj.material_sets = [first]
                    obj.data.material_ids = np.zeros(len(obj.data.material_ids), np.int32)
                    obj.data.materials = [source.data.materials[0] if source.data.materials else "材质"]
                else:
                    sets, ids, names = before[obj.uid]
                    obj.material_sets = list(sets)
                    obj.data.material_ids = np.array(ids, copy=True)
                    obj.data.materials = list(names)
                obj.geometry_dirty = True
                engine.add_mesh_object(obj)
            project.changed.emit("objects")
            _redraw(ctx, "objects")

        apply(True)
        history = _history(ctx)
        if history is not None:
            history.push("关联材质", lambda: apply(False), lambda: apply(True))
        return FINISHED
