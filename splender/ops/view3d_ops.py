"""3D 视口的导航与显示操作。键位对齐 Blender：中键旋转，Shift+中键平移，Ctrl+中键或滚轮缩放。"""
from __future__ import annotations

import numpy as np

from ..core import ops
from ..core.keymap import MIDDLEMOUSE, MOUSEMOVE, PRESS, RELEASE
from ..core.ops import CANCELLED, FINISHED, RUNNING_MODAL, Operator
from ..core.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from ..doc.project import CHANNEL_IDS


def view3d(ctx):
    """上下文里的 3D 视口编辑器；鼠标不在视口上时退回最近用过的那个。"""
    editor = ctx.editor
    if getattr(editor, "idname", "") == "VIEW_3D" and getattr(editor, "view", None) is not None:
        return editor
    editor = getattr(ctx.app, "active_view3d", None)
    if editor is not None and getattr(editor, "view", None) is not None:
        return editor
    return None


class _ViewOp(Operator):
    searchable = False

    @classmethod
    def poll(cls, ctx) -> bool:
        return view3d(ctx) is not None


def _point_under(editor, px: float, py: float) -> np.ndarray:
    """鼠标下的世界位置：有表面取表面，否则取过目标点、垂直视线的平面上的点。"""
    view = editor.view
    editor.app.host.make_current()
    hit = view.surface_at(px, py)
    if hit is not None:
        return hit[0]
    camera = view.camera
    origin, direction = camera.ray(px, py, view.width, view.height)
    forward = camera.forward
    denom = float(direction @ forward)
    if abs(denom) < 1e-9:
        return camera.target.copy()
    t = float((camera.target - origin) @ forward) / denom
    return origin + direction * t


class _DragOp(_ViewOp):
    """按住拖动的导航操作的共同部分。"""

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.button = event.type
        self.last = self.editor.pixel(event)
        self.begin(ctx, event)
        return RUNNING_MODAL

    def begin(self, ctx, event) -> None:
        pass

    def drag(self, ctx, dx: float, dy: float, event) -> None:
        raise NotImplementedError

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            x, y = self.editor.pixel(event) if event.source is self.editor.widget else self.last
            dx, dy = x - self.last[0], y - self.last[1]
            self.last = (x, y)
            if dx or dy:
                self.drag(ctx, dx, dy, event)
                self.editor.camera_changed()
            return RUNNING_MODAL
        if event.type == self.button and event.value == RELEASE:
            self.editor.sync_settings()
            return FINISHED
        if event.type == "ESC" and event.value == PRESS:
            return CANCELLED
        return RUNNING_MODAL


@ops.register
class View3DOrbit(_DragOp):
    idname = "view3d.orbit"
    label = "旋转视图"

    def begin(self, ctx, event) -> None:
        nav = ctx.app.prefs.navigation
        if self.editor.in_camera_view() and not self.editor.settings.lock_camera:
            self.editor.exit_camera_view(restore=False)
        self.pivot = _point_under(self.editor, *self.last) if nav.orbit_around_cursor else None
        camera = self.editor.view.camera
        if camera.axis_view and getattr(camera, "_auto_ortho", False):
            camera.ortho = False
            camera._auto_ortho = False

    def drag(self, ctx, dx, dy, event) -> None:
        nav = ctx.app.prefs.navigation
        camera = self.editor.view.camera
        if self.pivot is not None:
            camera.orbit_around(dx, dy, self.pivot, float(nav.orbit_sensitivity))
        else:
            camera.orbit(dx, dy, float(nav.orbit_sensitivity))


@ops.register
class View3DPan(_DragOp):
    idname = "view3d.pan"
    label = "平移视图"

    def begin(self, ctx, event) -> None:
        camera = self.editor.view.camera
        point = _point_under(self.editor, *self.last)
        self.depth = max(camera.depth_of(point), camera.distance * 0.05)

    def drag(self, ctx, dx, dy, event) -> None:
        view = self.editor.view
        editor = self.editor
        if editor.in_camera_view() and not editor.settings.lock_camera:
            px, py = editor.camera_pan
            editor.camera_pan = (px + 2.0 * dx / max(1, view.width), py - 2.0 * dy / max(1, view.height))
            return
        view.camera.pan(dx, dy, view.height, view.aspect, self.depth)


@ops.register
class View3DZoomDrag(_DragOp):
    idname = "view3d.zoom_drag"
    label = "拖动缩放视图"

    def begin(self, ctx, event) -> None:
        self.toward = _point_under(self.editor, *self.last) if ctx.app.prefs.navigation.zoom_to_mouse else None

    def drag(self, ctx, dx, dy, event) -> None:
        amount = -dy if not ctx.app.prefs.navigation.invert_zoom else dy
        editor = self.editor
        if editor.in_camera_view() and not editor.settings.lock_camera:
            editor.camera_zoom = float(np.clip(editor.camera_zoom * np.exp(amount * 0.01), 0.05, 20.0))
            return
        editor.view.camera.zoom(float(np.exp(amount * 0.01)), self.toward)


@ops.register
class View3DZoom(_ViewOp):
    idname = "view3d.zoom"
    label = "缩放视图"
    delta = FloatProperty("步数", default=1.0)

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        nav = ctx.app.prefs.navigation
        steps = float(self.delta)
        if event is not None and getattr(event, "wheel", 0.0):
            steps = float(event.wheel)
        if nav.invert_zoom:
            steps = -steps
        if editor.in_camera_view() and not editor.settings.lock_camera:
            editor.camera_zoom = float(np.clip(editor.camera_zoom * float(nav.zoom_speed) ** steps, 0.05, 20.0))
            editor.camera_changed()
            return FINISHED
        toward = None
        if nav.zoom_to_mouse and event is not None and event.source is editor.widget:
            toward = _point_under(editor, *editor.pixel(event))
        editor.view.camera.zoom(float(nav.zoom_speed) ** steps, toward)
        editor.camera_changed()
        editor.refresh_cursor()
        return FINISHED

    def execute(self, ctx) -> str:
        return self.invoke(ctx, None)


@ops.register
class View3DViewAll(_ViewOp):
    idname = "view3d.view_all"
    label = "查看全部"
    description = "把场景里的全部物体放进视口"
    searchable = True
    center = BoolProperty("游标回原点", default=False, description="同时把 3D 游标放回世界原点")

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        ctx.app.host.make_current()
        if editor.in_camera_view() and not editor.settings.lock_camera:
            editor.camera_zoom, editor.camera_pan = 1.0, (0.0, 0.0)      # 摄像机视图：画框放回中间
            editor.camera_changed()
            return FINISHED
        project = ctx.app.project
        if self.center and project is not None:
            project.cursor_location = (0.0, 0.0, 0.0)
        engine = ctx.app.engine
        objects = [o for o in project.all_objects() if o.visible] if project is not None else []
        local = editor.local_view["uids"] if editor.local_view is not None else None
        if local is not None:
            objects = [o for o in objects if o.uid in local]
        if not objects or not engine.frame_objects(editor.view, objects):
            engine.frame_view(editor.view)
        editor.camera_changed()
        return FINISHED


@ops.register
class View3DViewSelected(_ViewOp):
    idname = "view3d.view_selected"
    label = "查看所选"
    description = "把选中的物体放进视口（绘制、雕刻模式下是当前物体）"
    searchable = True

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        project = ctx.app.project
        ctx.app.host.make_current()
        if editor.in_camera_view() and not editor.settings.lock_camera:
            editor.exit_camera_view(restore=False)
        engine = ctx.app.engine
        objects = project.selected_objects() if project is not None else []
        if ctx.app.tool_settings.mode != "OBJECT" and project is not None and project.active_object is not None:
            objects = [project.active_object]
        if not objects or not engine.frame_objects(editor.view, objects):
            engine.frame_view(editor.view)
        editor.camera_changed()
        return FINISHED


@ops.register
class View3DViewAxis(_ViewOp):
    idname = "view3d.view_axis"
    label = "标准视图"
    searchable = True
    axis = EnumProperty("方向", items=[("FRONT", "前", ""), ("BACK", "后", ""), ("RIGHT", "右", ""), ("LEFT", "左", ""),
                                     ("TOP", "顶", ""), ("BOTTOM", "底", "")], default="FRONT")
    align_active = BoolProperty("对齐当前物体", default=False, description="按当前物体自己的朝向看")

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        if editor.in_camera_view() and not editor.settings.lock_camera:
            editor.exit_camera_view(restore=False)
        camera = editor.view.camera
        project = ctx.app.project
        active = project.active_object if project is not None else None
        if self.align_active and active is not None:
            _align_to_object(camera, active, self.axis)
            editor.camera_changed()
            return FINISHED
        camera.set_axis(self.axis)
        if ctx.app.prefs.navigation.auto_perspective and not camera.ortho:
            camera.ortho = True
            camera._auto_ortho = True
        editor.sync_settings()
        editor.camera_changed()
        return FINISHED


@ops.register
class View3DPerspOrtho(_ViewOp):
    idname = "view3d.view_persportho"
    label = "切换透视与正交"
    searchable = True

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        camera = editor.view.camera
        camera.ortho = not camera.ortho
        camera._auto_ortho = False
        editor.sync_settings()
        editor.camera_changed()
        return FINISHED


@ops.register
class View3DOrbitStep(_ViewOp):
    idname = "view3d.orbit_step"
    label = "按步旋转视图"
    yaw = FloatProperty("水平", default=0.0)
    pitch = FloatProperty("俯仰", default=0.0)

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        camera = editor.view.camera
        camera.yaw = (camera.yaw + self.yaw + 180.0) % 360.0 - 180.0
        camera.pitch = max(-89.999, min(89.999, camera.pitch + self.pitch))
        camera.axis_view = ""
        editor.camera_changed()
        return FINISHED


@ops.register
class View3DShadingCycle(_ViewOp):
    idname = "view3d.shading_cycle"
    label = "切换着色模式"
    description = "在实体和材质预览之间切换"
    searchable = True

    def execute(self, ctx) -> str:
        shading = view3d(ctx).shading
        shading.mode = "SOLID" if shading.mode != "SOLID" else "MATERIAL"
        return FINISHED


@ops.register
class View3DChannelCycle(_ViewOp):
    idname = "view3d.channel_cycle"
    label = "逐个查看通道"
    description = "依次只看基础色、金属度、粗糙度、高度，最后回到材质预览"
    searchable = True

    def execute(self, ctx) -> str:
        shading = view3d(ctx).shading
        if shading.mode != "CHANNEL":
            shading.channel = CHANNEL_IDS[0]
            shading.mode = "CHANNEL"
            return FINISHED
        index = CHANNEL_IDS.index(shading.channel) + 1
        if index >= len(CHANNEL_IDS):
            shading.mode = "MATERIAL"
        else:
            shading.channel = CHANNEL_IDS[index]
        return FINISHED


@ops.register
class View3DShadingSet(_ViewOp):
    idname = "view3d.shading_set"
    label = "设置着色模式"
    mode = EnumProperty("模式", items=[("WIREFRAME", "线框", ""), ("SOLID", "实体", ""), ("MATERIAL", "材质预览", ""),
                                     ("CHANNEL", "通道", ""), ("RENDERED", "渲染", "")], default="MATERIAL")

    def execute(self, ctx) -> str:
        view3d(ctx).shading.mode = self.mode
        return FINISHED


@ops.register
class View3DShadingToggleRendered(_ViewOp):
    idname = "view3d.shading_toggle_rendered"
    label = "切换渲染预览"
    description = "在视口里用渲染引擎实时出图，再按一次回到材质预览"
    searchable = True

    def execute(self, ctx) -> str:
        shading = view3d(ctx).shading
        shading.mode = "MATERIAL" if shading.mode == "RENDERED" else "RENDERED"
        return FINISHED


# ====================================================================== 视图：更多（照 Blender）
def _align_to_object(camera, obj, axis: str) -> None:
    """按物体自己的朝向摆标准视图：顶 = 从物体 +Z 往下看，前 = 从 -Y 看，右 = 从 +X 看（和 Blender 一样）。"""
    from ..doc.objects import local_to_internal_matrix

    m = local_to_internal_matrix(obj.transform.matrix())
    rot = m[:3, :3]
    x, y, z = [rot[:, k] / max(np.linalg.norm(rot[:, k]), 1e-12) for k in range(3)]
    table = {"TOP": (z, y), "BOTTOM": (-z, -y), "FRONT": (-y, z), "BACK": (y, z), "RIGHT": (x, z), "LEFT": (-x, z)}
    back, up = table[axis]
    camera.set_view(camera.target + back * camera.distance, back, up)


@ops.register
class View3DViewCenterCursor(_ViewOp):
    idname = "view3d.view_center_cursor"
    label = "视图中心对齐游标"
    description = "视图绕 3D 游标转，游标挪到屏幕中间"
    searchable = True

    def execute(self, ctx) -> str:
        from ..doc.objects import to_internal

        editor = view3d(ctx)
        project = ctx.app.project
        if project is None:
            return CANCELLED
        if editor.in_camera_view() and not editor.settings.lock_camera:
            editor.exit_camera_view(restore=False)
        editor.view.camera.target = to_internal(np.asarray(project.cursor_location, np.float64))
        editor.camera_changed()
        return FINISHED


@ops.register
class View3DViewCenterPick(_ViewOp):
    idname = "view3d.view_center_pick"
    label = "视图中心对齐鼠标处"
    description = "把鼠标下的表面点挪到屏幕中间，之后绕它转"

    def invoke(self, ctx, event) -> str:
        editor = view3d(ctx)
        if event is None:
            return CANCELLED
        ctx.app.host.make_current()
        hit = editor.view.surface_at(*editor.pixel(event))
        if hit is None:
            return CANCELLED
        camera = editor.view.camera
        if editor.in_camera_view() and not editor.settings.lock_camera:
            editor.exit_camera_view(restore=False)
        eye = camera.eye
        target = np.asarray(hit[0], np.float64)
        camera.distance = max(float(np.linalg.norm(eye - target)), 1e-6)
        right, up, back = camera.basis()
        camera.target = target
        camera.set_view(target + back * camera.distance, back, up)
        editor.camera_changed()
        return FINISHED


@ops.register
class View3DViewPan(_ViewOp):
    idname = "view3d.view_pan"
    label = "平移视图"
    direction = EnumProperty("方向", items=[("LEFT", "左", ""), ("RIGHT", "右", ""), ("UP", "上", ""),
                                          ("DOWN", "下", "")], default="LEFT")
    pixels = IntProperty("步长", default=48, min=1, max=2000, unit="px")

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        view = editor.view
        step = float(self.pixels)
        dx, dy = {"LEFT": (step, 0.0), "RIGHT": (-step, 0.0), "UP": (0.0, step), "DOWN": (0.0, -step)}[self.direction]
        view.camera.pan(dx, dy, view.height, view.aspect)
        editor.camera_changed()
        return FINISHED


@ops.register
class View3DViewRoll(_ViewOp):
    idname = "view3d.view_roll"
    label = "滚转视图"
    description = "绕视线转动视图"
    angle = FloatProperty("角度", default=15.0, min=-180.0, max=180.0, unit="°")

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        camera = editor.view.camera
        camera.roll = (camera.roll + float(self.angle) + 180.0) % 360.0 - 180.0
        camera.axis_view = ""
        editor.camera_changed()
        return FINISHED


@ops.register
class View3DViewAxisDrag(_ViewOp):
    """Alt+中键拖动：往哪边拖就转到哪个方向的标准视图（左右拖转 90°，上下拖到顶视图、底视图）。"""

    idname = "view3d.view_axis_drag"
    label = "拖到标准视图"

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.start = self.editor.pixel(event) if event is not None else self.editor.cursor_pixel()
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            x, y = self.editor.pixel(event) if event.source is self.editor.widget else self.editor.cursor_pixel()
            dx, dy = x - self.start[0], y - self.start[1]
            if max(abs(dx), abs(dy)) < 30:
                return RUNNING_MODAL
            camera = self.editor.view.camera
            yaw = round(camera.yaw / 90.0) * 90.0
            pitch = round(camera.pitch / 90.0) * 90.0
            if abs(dx) >= abs(dy):
                yaw = yaw - 90.0 if dx > 0 else yaw + 90.0
                pitch = 0.0 if abs(pitch) >= 89 else pitch
            else:
                pitch = max(-89.999, min(89.999, pitch + (89.999 if dy > 0 else -89.999)))
            camera.yaw = (yaw + 180.0) % 360.0 - 180.0
            camera.pitch = pitch
            camera.roll = 0.0
            names = {v: k for k, v in AXIS_VIEWS_ROUND.items()}
            camera.axis_view = names.get((round(camera.yaw) % 360, round(camera.pitch)), "")
            if ctx.app.prefs.navigation.auto_perspective and camera.axis_view:
                camera.ortho = True
                camera._auto_ortho = True
            self.start = (x, y)
            self.editor.sync_settings()
            self.editor.camera_changed()
            return RUNNING_MODAL
        if event.value == RELEASE and event.type == MIDDLEMOUSE:
            return FINISHED
        if event.value == PRESS and event.type in ("ESC", "RIGHTMOUSE"):
            return FINISHED
        return RUNNING_MODAL


AXIS_VIEWS_ROUND = {"FRONT": (0, 0), "BACK": (180, 0), "RIGHT": (90, 0), "LEFT": (270, 0), "TOP": (0, 90),
                    "BOTTOM": (0, -90)}


@ops.register
class View3DZoomBorder(_ViewOp):
    idname = "view3d.zoom_border"
    label = "框选放大"
    description = "拖一个框，把框里的部分放大到整个视口"
    searchable = True

    def invoke(self, ctx, event) -> str:
        self.editor = view3d(ctx)
        self.start = None
        self.current = self.editor.cursor_pixel()
        self.editor.set_overlay_shape(("cross", self.current))
        if ctx.wm is not None:
            ctx.wm.set_hint("拖一个框放大；右键或 Esc 取消")
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        editor = self.editor
        if event.type == MOUSEMOVE:
            self.current = editor.pixel(event) if event.source is editor.widget else editor.cursor_pixel()
            editor.set_overlay_shape(("box", self.start, self.current) if self.start else ("cross", self.current))
            return RUNNING_MODAL
        if event.value == PRESS and event.type == "LEFTMOUSE":
            self.start = self.current
            return RUNNING_MODAL
        if event.value == RELEASE and event.type == "LEFTMOUSE" and self.start is not None:
            self._finish(ctx)
            self._apply(ctx)
            return FINISHED
        if event.value == PRESS and event.type in ("ESC", "RIGHTMOUSE"):
            self._finish(ctx)
            return CANCELLED
        return RUNNING_MODAL

    def _finish(self, ctx) -> None:
        self.editor.set_overlay_shape(None)
        if ctx.wm is not None:
            ctx.wm.set_hint("")

    def _apply(self, ctx) -> None:
        editor = self.editor
        view = editor.view
        (x0, y0), (x1, y1) = self.start, self.current
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w < 4 or h < 4:
            return
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        if editor.in_camera_view() and not editor.settings.lock_camera:
            editor.exit_camera_view(restore=False)
        center = _point_under(editor, cx, cy)
        camera = view.camera
        factor = max(w / max(1, view.width), h / max(1, view.height))
        right, up, back = camera.basis()
        depth = max(camera.depth_of(center), 1e-6)
        camera.target = center
        camera.distance = max(depth * factor, 1e-6)
        camera.set_view(center + back * camera.distance, back, up)
        editor.camera_changed()

    def cancel(self, ctx) -> None:
        self._finish(ctx)


@ops.register
class View3DLocalView(_ViewOp):
    idname = "view3d.localview"
    label = "局部视图"
    description = "只看选中的物体（再按一次回到全部）"
    searchable = True

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        project = ctx.app.project
        objects = project.selected_objects() if project is not None else []
        if editor.local_view is None and not objects:
            self.report(ctx, "先选中要单独看的物体", "WARNING")
            return CANCELLED
        return FINISHED if editor.toggle_local_view(objects) else CANCELLED


@ops.register
class View3DLocalViewRemove(_ViewOp):
    idname = "view3d.localview_remove_from"
    label = "移出局部视图"
    description = "把选中的物体从局部视图里拿出去"

    @classmethod
    def poll(cls, ctx) -> bool:
        editor = view3d(ctx)
        return editor is not None and editor.local_view is not None

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        project = ctx.app.project
        for obj in project.selected_objects():
            editor.local_view["uids"].discard(obj.uid)
            obj.select = False
        editor.view.local_uids = set(editor.local_view["uids"])
        project.selection_changed()
        editor.camera_changed()
        return FINISHED


def _active_camera(project):
    camera = project.active_camera
    if camera is None:
        camera = next((o for o in project.scene_objects if o.kind == "CAMERA"), None)
        if camera is not None:
            project.active_camera_uid = camera.uid
    return camera


@ops.register
class View3DViewCamera(_ViewOp):
    idname = "view3d.view_camera"
    label = "摄像机视图"
    description = "从当前摄像机看（再按一次回到原来的视角）"
    searchable = True

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        project = ctx.app.project
        if editor.in_camera_view():
            editor.exit_camera_view(restore=True)
            return FINISHED
        camera = _active_camera(project) if project is not None else None
        if camera is None:
            self.report(ctx, "场景里没有摄像机：Shift+A 添加一台", "WARNING")
            return CANCELLED
        editor.enter_camera_view(camera)
        return FINISHED


@ops.register
class View3DObjectAsCamera(_ViewOp):
    idname = "view3d.object_as_camera"
    label = "设当前物体为摄像机"
    description = "把当前选中的摄像机设为出图用的摄像机，并从它看"
    searchable = True

    @classmethod
    def poll(cls, ctx) -> bool:
        project = ctx.app.project
        active = project.active_object if project is not None else None
        return view3d(ctx) is not None and active is not None and active.kind == "CAMERA"

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        project = ctx.app.project
        camera = project.active_object
        project.active_camera_uid = camera.uid
        project.mark_dirty()
        editor.enter_camera_view(camera)
        return FINISHED


@ops.register
class View3DCameraToView(_ViewOp):
    idname = "view3d.camera_to_view"
    label = "当前摄像机对齐视图"
    description = "把出图用的摄像机挪到现在的视角，并从它看"
    searchable = True

    def execute(self, ctx) -> str:
        from ..doc.objects import compose, decompose, euler_from_matrix, to_display

        editor = view3d(ctx)
        project = ctx.app.project
        camera_obj = _active_camera(project) if project is not None else None
        if camera_obj is None:
            self.report(ctx, "场景里没有摄像机：Shift+A 添加一台", "WARNING")
            return CANCELLED
        if editor.in_camera_view():
            return CANCELLED
        camera = editor.view.camera
        right, up, back = camera.basis()
        basis = np.stack([to_display(right), to_display(up), to_display(back)], axis=1)
        before = camera_obj.transform.matrix()
        _loc, _rot, scale = decompose(before)
        after = compose(to_display(camera.eye), euler_from_matrix(basis), scale)
        camera_obj.transform.set_matrix(after)
        history = ctx.app.history
        if history is not None:
            history.push("摄像机对齐视图", lambda: camera_obj.transform.set_matrix(before),
                         lambda: camera_obj.transform.set_matrix(after))
        editor.enter_camera_view(camera_obj)
        return FINISHED


@ops.register
class View3DToggleShading(_ViewOp):
    idname = "view3d.toggle_shading"
    label = "切换线框"
    description = "在线框和原来的着色之间来回切换"
    type = EnumProperty("着色", items=[("WIREFRAME", "线框", ""), ("SOLID", "实体", ""), ("MATERIAL", "材质预览", ""),
                                     ("RENDERED", "渲染", "")], default="WIREFRAME")

    def execute(self, ctx) -> str:
        editor = view3d(ctx)
        shading = editor.shading
        if shading.mode == self.type:
            shading.mode = getattr(editor, "_shading_before_toggle", "") or "SOLID"
        else:
            editor._shading_before_toggle = shading.mode
            shading.mode = self.type
        return FINISHED


@ops.register
class View3DToggleXray(_ViewOp):
    idname = "view3d.toggle_xray"
    label = "X 光"
    description = "模型半透明，能看到、选到后面的东西"
    searchable = True

    def execute(self, ctx) -> str:
        shading = view3d(ctx).shading
        shading.show_xray = not shading.show_xray
        return FINISHED


@ops.register
class View3DNavigate(_ViewOp):
    """漫游（照 Blender 的行走模式）：W/S/A/D 前后左右，Q/E 下上，鼠标转头，Shift 快、Alt 慢，滚轮调速度。
    左键或回车确认，右键或 Esc 回到原来的视角。"""

    idname = "view3d.navigate"
    label = "漫游"
    description = "像走路一样在场景里移动：W/S/A/D 前后左右，Q/E 下上，鼠标转头"
    searchable = True

    def invoke(self, ctx, event) -> str:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QCursor

        self.editor = view3d(ctx)
        if self.editor.in_camera_view() and not self.editor.settings.lock_camera:
            self.editor.exit_camera_view(restore=False)
        self.app = ctx.app
        self.camera = self.editor.view.camera
        self.saved = self.camera.state()
        self.camera.ortho = False
        self.keys: set[str] = set()
        self.speed = float(ctx.app.prefs.navigation.walk_speed)
        self.fast = self.slow = False
        widget = self.editor.widget
        self.center = widget.mapToGlobal(widget.rect().center())
        QCursor.setPos(self.center)
        widget.setCursor(Qt.BlankCursor)
        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.last = None
        self.timer.start()
        if ctx.wm is not None:
            ctx.wm.set_hint("漫游：W/S/A/D 前后左右 · Q/E 下上 · 鼠标转头 · Shift 快 · Alt 慢 · 滚轮调速度 · "
                            "左键确认 · 右键取消")
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        from PySide6.QtGui import QCursor

        if event.type == MOUSEMOVE:
            pos = QCursor.pos()
            dx, dy = pos.x() - self.center.x(), pos.y() - self.center.y()
            if dx or dy:
                sensitivity = float(ctx.app.prefs.navigation.walk_mouse_sensitivity) * 0.15
                self.camera.yaw = (self.camera.yaw - dx * sensitivity + 180.0) % 360.0 - 180.0
                self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch - dy * sensitivity))
                self._keep_eye()
                QCursor.setPos(self.center)
                self.editor.camera_changed()
            return RUNNING_MODAL
        if event.type in ("WHEELUPMOUSE", "WHEELDOWNMOUSE") or event.value == "WHEEL":
            up = event.type == "WHEELUPMOUSE" or getattr(event, "wheel", 0.0) > 0
            self.speed = float(np.clip(self.speed * (1.2 if up else 1.0 / 1.2), 0.01, 10000.0))
            return RUNNING_MODAL
        if event.type in ("LEFT_SHIFT", "RIGHT_SHIFT"):
            self.fast = event.value == PRESS
            return RUNNING_MODAL
        if event.type in ("LEFT_ALT", "RIGHT_ALT"):
            self.slow = event.value == PRESS
            return RUNNING_MODAL
        if event.value == PRESS and event.type in ("LEFTMOUSE", "RET", "NUMPAD_ENTER"):
            self._end(ctx)
            return FINISHED
        if event.value == PRESS and event.type in ("RIGHTMOUSE", "ESC"):
            self.camera.set_state(self.saved)
            self._end(ctx)
            self.editor.camera_changed()
            return CANCELLED
        if event.type in ("W", "S", "A", "D", "Q", "E", "UP_ARROW", "DOWN_ARROW", "LEFT_ARROW", "RIGHT_ARROW"):
            if event.value == PRESS:
                self.keys.add(event.type)
            elif event.value == RELEASE:
                self.keys.discard(event.type)
            return RUNNING_MODAL
        return RUNNING_MODAL

    def _keep_eye(self) -> None:
        """转头时眼睛不动（目标点绕着眼睛转）。"""
        eye = self._eye if getattr(self, "_eye", None) is not None else self.camera.eye
        self.camera.target = eye - self.camera.basis()[2] * self.camera.distance
        self._eye = eye

    def _tick(self) -> None:
        import time

        now = time.perf_counter()
        dt = 0.016 if self.last is None else min(0.1, now - self.last)
        self.last = now
        if not self.keys:
            self._eye = self.camera.eye
            return
        right, up, back = self.camera.basis()
        forward = -back
        flat = np.array([forward[0], 0.0, forward[2]])
        if np.linalg.norm(flat) > 1e-6:
            flat /= np.linalg.norm(flat)
        move = np.zeros(3)
        k = self.keys
        if k & {"W", "UP_ARROW"}:
            move += forward
        if k & {"S", "DOWN_ARROW"}:
            move -= forward
        if k & {"D", "RIGHT_ARROW"}:
            move += right
        if k & {"A", "LEFT_ARROW"}:
            move -= right
        if "E" in k:
            move += np.array([0.0, 1.0, 0.0])
        if "Q" in k:
            move -= np.array([0.0, 1.0, 0.0])
        if np.linalg.norm(move) < 1e-9:
            return
        nav = self.app.prefs.navigation
        speed = self.speed * (float(nav.walk_speed_factor) if self.fast else 1.0) / \
            (float(nav.walk_speed_factor) if self.slow else 1.0)
        step = move / np.linalg.norm(move) * speed * dt
        self.camera.target = self.camera.target + step
        self._eye = self.camera.eye
        self.editor.camera_changed()

    def _end(self, ctx) -> None:
        self.timer.stop()
        self.editor.widget.unsetCursor()
        if ctx.wm is not None:
            ctx.wm.set_hint("")

    def cancel(self, ctx) -> None:
        self.camera.set_state(self.saved)
        self._end(ctx)
