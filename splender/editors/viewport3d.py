"""3D 视口：看模型、绘制、导航。"""
from __future__ import annotations

import logging

from PySide6.QtCore import QEvent, QObject, QRectF, Qt
from PySide6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QLabel

from ..core import registry
from ..core.keymap import MOUSEMOVE
from ..core.props import BoolProperty, FloatProperty, PropertyGroup
from ..doc.project import ViewOverlay, ViewShading
from ..engine import ibl
from ..tools.paint_tools import EFFECT_TOOLS, STROKE_TOOLS
from ..ui import theme
from ..ui.editor import Editor

log = logging.getLogger("splender.editors.view3d")

HINT_PAINT = "左键 绘制 · Ctrl+左键 擦除 · 中键 旋转 · Shift+中键 平移 · 滚轮 缩放 · F 笔刷大小 · Shift+F 流量"
HINT_NAV = "中键 旋转 · Shift+中键 平移 · 滚轮 缩放 · Home 查看全部"
HINT_SCULPT = "左键 雕刻 · Ctrl 反向 · Shift 平滑 · F 大小 · Shift+F 力度 · V 标准 · C 黏土条 · G 抓取 · M 遮罩"
HINT_OBJECT = ("左键 选择 · 拖动 框选 · G 移动 · R 旋转 · S 缩放 · Shift+A 添加 · X 删除 · Shift+D 复制 · "
               "Shift+右键 放 3D 游标 · Ctrl+Tab 换模式")
HINT_EDIT = ("1 2 3 点边面 · 左键 选择 · Alt+左键 循环边 · G R S 变换 · E 挤出 · I 内插 · Ctrl+B 倒角 · "
             "Ctrl+R 环切 · K 切刀 · M 合并 · X 删除 · 右键 菜单 · Tab 退出")
#: 每种模式先查的键位表（和 Blender 一样，模式的表能盖掉通用的键）
MODE_KEYMAPS = {"OBJECT": ["Object Mode"], "SCULPT": ["Sculpt"], "PAINT": ["Paint"], "EDIT": ["Mesh"]}


class ViewSettings(PropertyGroup):
    """一个视口的相机设置。"""

    lens = FloatProperty("焦距", default=50.0, min=10.0, max=250.0, unit="mm", precision=0,
                         description="数值越大视角越窄，透视变形越小")
    ortho = BoolProperty("正交", default=False, description="关闭透视，远近一样大")
    lock_camera = BoolProperty("锁定摄像机到视图", default=False,
                               description="从摄像机看时，旋转、平移、缩放视图会直接移动摄像机本身")
    camera_fill = FloatProperty("摄像机画框大小", default=0.86, min=0.2, max=1.0, precision=2, subtype="FACTOR",
                                description="从摄像机看时，画框占视口的比例")
    passepartout = FloatProperty("画框外变暗", default=0.5, min=0.0, max=1.0, precision=2, subtype="FACTOR",
                                 description="从摄像机看时，画框外面压暗多少")


STROKE_OPS = {"paint.stroke", "sculpt.stroke"}


class _LeaveFilter(QObject):
    def __init__(self, editor) -> None:
        super().__init__(editor)
        self.editor = editor

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Leave:
            self.editor.clear_cursor()
        elif event.type() == QEvent.Enter:
            self.editor.on_enter()
        return False


@registry.register_editor
class Viewport3D(Editor):
    idname = "VIEW_3D"
    label = "3D 视口"
    icon = "editor.view3d"
    category = "通用"
    order = 10
    description = "查看模型并在表面上绘制"
    has_toolbar = True
    has_sidebar = True
    sidebar_default = False
    keymaps = ["Paint", "3D View"]

    def __init__(self, app, area=None) -> None:
        self.shading = ViewShading()
        self.overlay = ViewOverlay()
        self.settings = ViewSettings()
        self.view = None
        self.widget = None
        self.redo_panel = None
        self._mouse = None            # 最近一次鼠标位置（视口像素）
        self._pending_state = None
        self._shape = None            # 框选框、套索线、刷选圆这些临时的二维叠加（视口像素）
        self.camera_view_uid = 0      # 正在从哪台摄像机看（0 表示没有）
        self.camera_zoom = 1.0        # 从摄像机看时画框的缩放、平移（只影响这个视口）
        self.camera_pan = (0.0, 0.0)
        self.camera_frame = None      # 画框在视口里的位置 (中心 x, y, 半宽, 半高)，屏幕归一化坐标
        self._pre_camera = None       # 进摄像机视图前的视角，出来时恢复
        self.local_view = None        # 局部视图：{"uids": 物体 uid 集合, "camera": 进来前的视角}
        super().__init__(app, area)

    def active_keymaps(self) -> list[str]:
        mode = getattr(self.app.tool_settings, "mode", "PAINT")
        return MODE_KEYMAPS.get(mode, []) + ["Object Non-modal", "3D View"]

    # ------------------------------------------------------------------ 搭建
    def build_main(self):
        host = getattr(self.app, "host", None)
        if host is None:
            label = QLabel("这台电脑上没有可用的三维显示")
            label.setAlignment(Qt.AlignCenter)
            return label
        from ..ui.glhost import ViewportWidget

        host.make_current()
        self.view = host.engine.create_view(self.shading, self.overlay)
        self.view.before_render = self._before_render
        self.widget = ViewportWidget(host, self.view)
        self.widget.on_paint_overlay = self._paint_overlay
        self.overlay.changed.connect(self._on_overlay_changed)
        from ..ui.redo import RedoPanel
        from .nav_gizmo import NavGizmo

        self.redo_panel = RedoPanel(self, self.widget)
        self.nav_gizmo = NavGizmo(self)
        from .transform_gizmo import TransformGizmo

        self.transform_gizmo = TransformGizmo(self)
        self.install_input(self.widget)
        self._leave = _LeaveFilter(self)
        self.widget.installEventFilter(self._leave)
        self.shading.changed.connect(self._on_shading_changed)
        self.settings.changed.connect(self._on_settings_changed)
        self.settings.lens = float(self.app.prefs.navigation.lens)
        self.view.camera.lens = self.settings.lens
        if getattr(self.app, "active_view3d", None) is None:
            self.app.active_view3d = self
        return self.widget

    def dispose(self) -> None:
        if getattr(self.app, "active_view3d", None) is self:
            self.app.active_view3d = None
        if self.widget is not None:
            self.widget.dispose()
            self.widget = None
            self.view = None
        super().dispose()

    # ------------------------------------------------------------------ 标题栏
    def draw_header(self, layout, ctx) -> None:
        """和 Blender 一样：模式、视图和本模式的菜单在左，坐标系、轴心点、吸附在中，叠加层、着色在右。"""
        tools = self.app.tool_settings
        mode = tools.mode
        layout.prop(tools, "mode", text="")
        if mode == "EDIT":
            self._draw_select_mode(layout)
        layout.menu("VIEW3D_MT_view", text="视图")
        if mode == "OBJECT":
            layout.menu("VIEW3D_MT_select_object", text="选择")
            layout.menu("VIEW3D_MT_add", text="添加")
            layout.menu("VIEW3D_MT_object", text="物体")
        elif mode == "EDIT":
            layout.menu("VIEW3D_MT_select_edit_mesh", text="选择")
            layout.menu("VIEW3D_MT_mesh_add", text="添加")
            layout.menu("VIEW3D_MT_edit_mesh", text="网格")
            layout.menu("VIEW3D_MT_edit_mesh_vertices", text="点")
            layout.menu("VIEW3D_MT_edit_mesh_edges", text="边")
            layout.menu("VIEW3D_MT_edit_mesh_faces", text="面")
            layout.menu("VIEW3D_MT_uv_map", text="UV")
        elif mode == "SCULPT":
            layout.menu("VIEW3D_MT_sculpt", text="雕刻")
            layout.separator()
            sculpt = tools.sculpt
            layout.prop(sculpt, "size", text="直径")
            layout.prop(sculpt, "strength", text="力度", slider=True)
            layout.prop(sculpt, "hardness", text="硬度", slider=True)
            layout.prop(sculpt, "invert", toggle=True)
        else:
            layout.menu("VIEW3D_MT_paint", text="绘制")
            layout.menu("PAINT_MT_select", text="选择")
            layout.menu("LAYERS_MT_adjust", text="调整")
            layout.menu("LAYERS_MT_filter", text="滤镜")
            layout.separator()
            from ..ops.select_ops import draw_tool_header

            if draw_tool_header(layout, tools):
                pass
            elif tools.tool == "paint.fill":
                fill = tools.fill
                layout.prop(fill, "mode", text="")
                layout.prop(tools.brush, "color", text="")
                sub = layout.row(align=True)
                sub.enabled = fill.mode == "SIMILAR"
                sub.prop(fill, "tolerance", text="容差", slider=True)
                sub.prop(fill, "contiguous", toggle=True)
                layout.prop(fill, "opacity", text="不透明度", slider=True)
            elif tools.tool == "paint.gradient":
                gradient = tools.gradient
                layout.prop(gradient, "shape", text="")
                layout.prop(tools.brush, "color", text="")
                layout.prop(tools.brush, "secondary_color", text="")
                layout.prop(gradient, "opacity", text="不透明度", slider=True)
                layout.prop(gradient, "reverse", toggle=True)
            elif tools.tool == "paint.text":
                layout.prop(tools.text, "font", text="")
                layout.prop(tools.text, "size", text="字号")
                layout.prop(tools.brush, "color", text="")
                layout.prop(tools.text, "bold", toggle=True)
                layout.prop(tools.text, "italic", toggle=True)
            elif tools.tool == "paint.shape":
                layout.prop(tools.shape, "kind", text="")
                layout.prop(tools.brush, "color", text="")
                layout.prop(tools.shape, "fill", toggle=True)
                layout.prop(tools.shape, "stroke", text="描边")
            elif tools.tool in EFFECT_TOOLS:
                brush = tools.brush
                layout.prop(brush, "size", text="直径")
                layout.prop(brush, "hardness", text="硬度", slider=True)
                layout.prop(tools.effect, "strength", text="强度", slider=True)
                if tools.tool in ("paint.dodge", "paint.burn"):
                    layout.prop(tools.effect, "tone_range", text="")
                elif tools.tool == "paint.sponge":
                    layout.prop(tools.effect, "sponge_mode", text="")
            else:
                brush = tools.active_brush()
                if tools.tool != "paint.eraser":
                    layout.prop(brush, "color", text="")
                layout.prop(brush, "size", text="直径")
                layout.prop(brush, "flow", text="流量", slider=True)
                layout.prop(brush, "opacity", text="不透明度", slider=True)
                layout.prop(brush, "hardness", text="硬度", slider=True)
        layout.stretch()
        if mode in ("OBJECT", "EDIT"):
            layout.prop(tools, "transform_orientation", text="", icon_only=True)
            layout.prop(tools, "transform_pivot_point", text="", icon_only=True)
            row = layout.row(align=True)
            row.prop(tools, "use_snap", text="吸附", icon="snap", icon_only=True)
            row.popover("", "chevron.down", self._draw_snap)
            if mode == "EDIT":
                row = layout.row(align=True)
                row.prop(tools, "use_proportional_edit", text="衰减编辑", icon="proportional", icon_only=True)
                row.popover("", "chevron.down", self._draw_proportional)
        else:
            layout.label("对称", role="dim")
            row = layout.row(align=True)
            row.prop(tools.symmetry, "x", toggle=True)
            row.prop(tools.symmetry, "y", toggle=True)
            row.prop(tools.symmetry, "z", toggle=True)
        layout.separator()
        row = layout.row(align=True)
        row.prop(self.overlay, "show_overlays", text="叠加层", icon="overlays", icon_only=True)
        row.popover("", "chevron.down", self._draw_overlays)
        layout.prop(self.shading, "show_xray", text="X 光", icon="xray", icon_only=True)
        row = layout.row(align=True)
        row.prop(self.shading, "mode", text="", expand=True, icon_only=True)
        row.popover("", "chevron.down", self._draw_shading)

    def _draw_select_mode(self, layout) -> None:
        """点、边、面选择模式（Shift 点击同时开几种，Ctrl 点击按现在的选中扩展）。"""
        engine = getattr(self.app.host, "engine", None)
        session = getattr(engine, "edit", None)
        modes = set(session.mesh.select_mode) if session is not None else {"VERT"}
        row = layout.row(align=True)
        for kind, icon, tip in (("VERT", "select.vert", "点选择模式（1）"), ("EDGE", "select.edge", "边选择模式（2）"),
                                ("FACE", "select.face", "面选择模式（3）")):
            button = row.operator("mesh.select_mode", text="", icon=icon, type=kind)
            button.setCheckable(True)
            button.setChecked(kind in modes)
            button.setToolTip(tip + "\nShift 点击：同时用几种 · Ctrl 点击：按现在的选中扩展")

    def _draw_proportional(self, layout, ctx) -> None:
        tools = self.app.tool_settings
        layout.label("衰减编辑", role="title")
        layout.prop(tools, "use_proportional_edit")
        layout.prop(tools, "proportional_edit_falloff", expand=True)
        layout.prop(tools, "proportional_size")
        layout.prop(tools, "use_proportional_connected")

    def _draw_snap(self, layout, ctx) -> None:
        tools = self.app.tool_settings
        layout.label("吸附", role="title")
        layout.prop(tools, "use_snap")
        layout.prop(tools, "snap_elements", expand=True)
        layout.prop(tools, "snap_translate")
        layout.prop(tools, "snap_rotate")
        layout.prop(tools, "snap_scale")
        layout.separator()
        layout.prop(tools, "precision_factor")

    def _draw_overlays(self, layout, ctx) -> None:
        overlay = self.overlay
        layout.label("视口叠加层", role="title")
        layout.prop(overlay, "show_floor")
        row = layout.row(align=True)
        row.prop(overlay, "show_axis_x", toggle=True)
        row.prop(overlay, "show_axis_y", toggle=True)
        layout.prop(overlay, "grid_scale")
        layout.separator()
        layout.prop(overlay, "show_cursor")
        layout.prop(overlay, "show_extras")
        layout.prop(overlay, "show_outline")
        layout.prop(overlay, "show_object_origins")
        layout.prop(overlay, "show_object_origins_all")
        layout.separator()
        layout.prop(overlay, "show_gizmo")
        layout.prop(overlay, "show_stats")
        layout.prop(overlay, "show_wireframes")
        layout.prop(overlay, "wireframe_opacity", slider=True)

    def _draw_shading(self, layout, ctx) -> None:
        shading = self.shading
        layout.label("显示", role="title")
        if shading.mode == "WIREFRAME":
            layout.prop(self.overlay, "wireframe_opacity", slider=True)
            return
        if shading.mode in ("SOLID", "MATERIAL", "CHANNEL"):
            layout.prop(shading, "xray_alpha", slider=True)
        if shading.mode == "RENDERED":
            project = self.app.project
            if project is not None:
                layout.prop(project.render, "engine")
                settings = project.render.engine_settings()
                if settings is not None and hasattr(settings, "viewport_samples"):
                    layout.prop(settings, "viewport_samples")
                    layout.prop(settings, "viewport_scale", expand=True)
        if shading.mode in ("MATERIAL", "RENDERED"):
            layout.prop(shading, "hdri", text="环境")
            layout.prop(shading, "hdri_rotation")
            layout.prop(shading, "hdri_strength")
            layout.prop(shading, "background_opacity", slider=True)
            layout.prop(shading, "background_blur", slider=True)
            layout.prop(shading, "exposure")
            layout.prop(shading, "view_transform", expand=True)
            layout.prop(shading, "bump")
        elif shading.mode == "SOLID":
            layout.prop(shading, "solid_light", expand=True)
            if shading.solid_light == "STUDIO":
                layout.prop(shading, "studio_light")
            elif shading.solid_light == "MATCAP":
                layout.prop(shading, "matcap")
            layout.prop(shading, "solid_color", expand=True)
            layout.prop(shading, "bump")
        else:
            layout.prop(shading, "channel", expand=True)

    # ------------------------------------------------------------------ 事件
    def handle_event(self, event) -> bool:
        if self.view is None:
            return super().handle_event(event)
        self.app.active_view3d = self
        if event.type == MOUSEMOVE or event.source is self.widget:
            self._mouse = self.widget.to_pixels(event.x, event.y)
        wm = self.wm
        gizmo = getattr(self, "nav_gizmo", None)
        if gizmo is not None and event.source is self.widget and not (wm is not None and wm.has_modal()):
            # 右上角的导航小部件先接鼠标（和 Blender 一样，它在所有工具之上）
            if event.type == MOUSEMOVE and gizmo.on_move(event.x, event.y):
                return True
            if event.value == "PRESS" and gizmo.on_press(event):
                return True
            if event.value == "RELEASE" and gizmo.on_release(event):
                return True
        tgizmo = getattr(self, "transform_gizmo", None)
        if tgizmo is not None and event.source is self.widget and not (wm is not None and wm.has_modal()):
            # 移动、旋转、缩放工具的操纵杆：按在上面就是带约束的变换，别处照常（框选等）
            if event.type == MOUSEMOVE:
                tgizmo.on_move(event.x, event.y)
            elif event.value == "PRESS" and tgizmo.on_press(event):
                return True
        handled = super().handle_event(event)
        wm = self.wm
        if event.type == MOUSEMOVE:
            # 画笔、雕刻的一笔进行中，光标圈也跟着鼠标走（其它模态操作时不动）
            stroking = wm is not None and any(getattr(op, "idname", "") in STROKE_OPS for op in wm.modal_ops())
            if stroking or (not handled and not (wm is not None and wm.has_modal())):
                self.refresh_cursor()
        return handled

    def pixel(self, event) -> tuple[float, float]:
        """事件位置换算成视口像素。"""
        return self.widget.to_pixels(event.x, event.y)

    def cursor_pixel(self) -> tuple[float, float]:
        """鼠标现在的位置（视口像素），鼠标在视口外也照算。"""
        local = self.widget.mapFromGlobal(QCursor.pos())
        return self.widget.to_pixels(float(local.x()), float(local.y()))

    def set_overlay_shape(self, shape) -> None:
        """拖动时的临时形状（选框、椭圆、刷选圆、套索、虚线……，见 ui/overlay_shapes.py），None 清掉。坐标都是视口像素。"""
        if shape == self._shape:
            return
        self._shape = shape
        if self.widget is not None:
            self.widget.overlay.update()

    def on_enter(self) -> None:
        if self.wm is not None and self.app.prefs.interface.status_hints:
            tool = self.app.tool_settings.tool
            if self.app.tool_settings.mode == "OBJECT":
                hint = HINT_OBJECT
            elif self.app.tool_settings.mode == "EDIT":
                hint = HINT_EDIT
            else:
                hint = HINT_PAINT if tool.startswith("paint.") else (HINT_SCULPT if tool.startswith("sculpt.")
                                                                      else HINT_NAV)
            self.wm.set_hint(hint)

    def refresh_cursor(self) -> None:
        """按当前鼠标位置和笔刷设置更新笔刷光标。"""
        if self.view is None or self._mouse is None:
            return
        host = self.app.host
        host.make_current()
        tools = self.app.tool_settings
        # 只有笔刷类工具有笔刷圈（渐变、油漆桶、文字、形状这些点、拖的工具没有）
        brush = tools.active_brush() if (tools.tool in STROKE_TOOLS or tools.tool.startswith("sculpt.")) else None
        if not self.app.prefs.viewport.cursor_ring:
            brush = None
        host.engine.hover(self.view, self._mouse[0], self._mouse[1], brush)
        host.request_frame()

    def clear_cursor(self) -> None:
        self._mouse = None
        if self.view is not None and self.view.cursor is not None:
            self.view.cursor = None
            self.view.dirty = True
            self.app.host.request_frame()
        if self.wm is not None:
            self.wm.set_hint("")

    def camera_changed(self) -> None:
        if self.in_camera_view() and self.settings.lock_camera:
            self.write_camera_object()
        self.view.invalidate_camera()
        self.app.host.request_frame()

    # ------------------------------------------------------------------ 从摄像机看（小键盘 0）
    def camera_object(self):
        project = self.app.project
        if project is None or not self.camera_view_uid:
            return None
        obj = project.object_by_uid(self.camera_view_uid)
        return obj if obj is not None and obj.kind == "CAMERA" else None

    def in_camera_view(self) -> bool:
        return self.camera_object() is not None

    def enter_camera_view(self, obj) -> None:
        if self.view is None:
            return
        if not self.in_camera_view():
            self._pre_camera = self.view.camera.state()
        self.camera_view_uid = obj.uid
        self.view.camera_object_uid = obj.uid          # 叠加层不画正在用的这台摄像机
        self.camera_zoom = 1.0
        self.camera_pan = (0.0, 0.0)
        self._sync_camera_view()
        self.view.invalidate_camera()
        self.app.host.request_frame()
        self.refresh_header()

    def exit_camera_view(self, restore: bool = True) -> None:
        if not self.camera_view_uid:
            return
        self.camera_view_uid = 0
        self.view.camera_object_uid = 0
        self.camera_frame = None
        camera = self.view.camera
        camera.offset = (0.0, 0.0)
        camera.clip_override = None
        if restore and self._pre_camera is not None:
            camera.set_state(self._pre_camera)
        else:
            camera.lens = float(self.settings.lens)
            camera.ortho = bool(self.settings.ortho)
        self._pre_camera = None
        self.view.invalidate_camera()
        self.app.host.request_frame()
        self.refresh_header()

    def _before_render(self) -> None:
        if self.camera_view_uid:
            if self.camera_object() is None:
                self.exit_camera_view(restore=True)
            else:
                self._sync_camera_view()

    def _sync_camera_view(self) -> None:
        """视口相机对准摄像机物体：位置、朝向、焦距或正交大小、偏移、裁剪；画框按出图的宽高比放进视口。"""
        import math

        import numpy as np

        from ..doc.objects import local_to_internal_matrix
        from ..engine.camera import SENSOR

        obj = self.camera_object()
        view = self.view
        if obj is None or view is None or not view.width:
            return
        m = local_to_internal_matrix(obj.transform.matrix())
        rot = m[:3, :3]
        cols = [rot[:, k] / max(np.linalg.norm(rot[:, k]), 1e-12) for k in range(3)]
        eye = m[:3, 3]
        data = obj.data
        camera = view.camera
        camera.set_view(eye, cols[2], cols[1], distance=max(float(data.focus_distance), 0.1)
                        if data.use_dof else max(camera.distance, 0.1))
        width, height = self.app.project.render.size if self.app.project is not None else (1920, 1080)
        render_aspect = width / max(1.0, height)
        view_aspect = view.aspect
        fill = float(self.settings.camera_fill) * float(self.camera_zoom)
        ortho = data.projection == "ORTHO"
        half_long = float(data.ortho_scale) * 0.5 if ortho else float(data.sensor) * 0.5 / max(1e-6, float(data.lens))
        if render_aspect >= 1.0:
            fx, fy = half_long, half_long / render_aspect
        else:
            fx, fy = half_long * render_aspect, half_long
        if view_aspect >= 1.0:
            major = max(fx / fill, fy * view_aspect / fill)
            vx, vy = major, major / view_aspect
        else:
            major = max(fx / (view_aspect * fill), fy / fill)
            vx, vy = major * view_aspect, major
        camera.ortho = ortho
        if ortho:
            camera.lens = 50.0
            tan_y, _tan_x = camera.tan_half(view_aspect)
            camera.distance = vy / max(tan_y, 1e-9)
            camera.target = eye - cols[2] * camera.distance
        else:
            camera.lens = SENSOR / max(major, 1e-9)
        shift_x = float(data.shift_x) * 2.0 * half_long / vx
        shift_y = float(data.shift_y) * 2.0 * half_long / vy
        pan_x, pan_y = self.camera_pan
        camera.offset = (shift_x + pan_x, shift_y + pan_y)
        camera.clip_override = (float(data.clip_start), float(data.clip_end))
        camera.axis_view = ""
        # 画框在视口里：中心在镜头偏移处（摄像机的偏移不移动画框，只移动画面），大小按 fill
        self.camera_frame = (pan_x, pan_y, fx / vx, fy / vy)
        del math

    def write_camera_object(self) -> None:
        """「锁定摄像机到视图」：把视口现在的位置、朝向写回摄像机物体（保持它的缩放）。"""
        import numpy as np

        from ..doc.objects import compose, decompose, euler_from_matrix, to_display

        obj = self.camera_object()
        if obj is None:
            return
        right, up, back = self.view.camera.basis()
        basis = np.stack([to_display(right), to_display(up), to_display(back)], axis=1)
        _loc, _rot, scale = decompose(obj.transform.matrix())
        location = to_display(self.view.camera.eye)
        obj.transform.set_matrix(compose(location, euler_from_matrix(basis), scale))

    # ------------------------------------------------------------------ 局部视图（/）
    def toggle_local_view(self, objects) -> bool:
        if self.view is None:
            return False
        if self.local_view is not None:
            state = self.local_view.get("camera")
            self.local_view = None
            self.view.local_uids = None
            if state:
                self.view.camera.set_state(state)
            self.view.invalidate_camera()
            self.app.host.request_frame()
            return True
        objects = [o for o in objects if o.visible]
        if not objects:
            return False
        self.local_view = {"uids": {o.uid for o in objects}, "camera": self.view.camera.state()}
        self.view.local_uids = set(self.local_view["uids"])
        self.app.host.make_current()
        self.app.engine.frame_objects(self.view, objects)
        self.view.invalidate_camera()
        self.app.host.request_frame()
        return True

    # ------------------------------------------------------------------ 设置变化
    def _on_shading_changed(self, name: str) -> None:
        if self.view is not None:
            self.view.dirty = True
            self.app.host.request_frame()
        if name in ("mode", "solid_light"):
            self.refresh_header()

    def _on_overlay_changed(self, name: str) -> None:
        if self.view is not None:
            self.view.dirty = True
            self.app.host.request_frame()
        if name == "show_overlays":
            self.refresh_header()

    def _on_settings_changed(self, name: str) -> None:
        if self.view is None:
            return
        camera = self.view.camera
        camera.lens = float(self.settings.lens)
        camera.ortho = bool(self.settings.ortho)
        self.camera_changed()

    def sync_settings(self) -> None:
        """相机被操作改动后，把数值同步回设置面板。"""
        camera = self.view.camera
        with self.settings.changed.block():
            self.settings.ortho = camera.ortho
            self.settings.lens = camera.lens

    def refresh(self) -> None:
        super().refresh()
        if self.view is not None and self.view.dirty:
            self.app.host.request_frame()

    # ------------------------------------------------------------------ 叠加文字
    def _paint_overlay(self, widget) -> None:
        painter = QPainter(widget)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setFont(theme.font("font.small"))
        margin = theme.px(10)
        text = self.view.camera.label()
        if self.shading.mode == "RENDERED" and self.app.engine is not None:
            status = self.app.engine.rendered_status(self.view)
            if status:
                text += " · " + status
        if self.camera_frame is not None and self.in_camera_view():
            text = "摄像机视图 · %s" % self.camera_object().name
            self._paint_camera_frame(painter, widget)
        if self.local_view is not None:
            text += " · 局部"
        rect = QRectF(margin, margin, widget.width() - 2 * margin, theme.px(18))
        painter.setPen(QColor(0, 0, 0, 150))
        painter.drawText(rect.translated(1, 1), Qt.AlignLeft | Qt.AlignVCenter, text)
        painter.setPen(theme.qcolor("text.dim"))
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, text)
        lines = []
        if self.overlay.show_overlays and self.overlay.show_stats:
            lines.extend(self._scene_stats())
        if self.app.prefs.viewport.show_stats and self.app.engine is not None:
            stats = self.app.engine.perf.summary()
            lines.append("每帧 %.1f ms · 延迟 %.1f ms" % (stats["frame_ms_p50"], stats["latency_ms_p50"]))
        painter.setPen(theme.qcolor("text.faint"))
        for index, line in enumerate(lines):
            painter.drawText(rect.translated(0, theme.px(18) * (index + 1)), Qt.AlignLeft | Qt.AlignVCenter, line)
        if self._shape is not None:
            self._paint_shape(painter, widget)
        tgizmo = getattr(self, "transform_gizmo", None)
        if tgizmo is not None and not (self.wm is not None and self.wm.has_modal()):
            tgizmo.paint(painter)
        gizmo = getattr(self, "nav_gizmo", None)
        if gizmo is not None:
            gizmo.paint(painter)
        painter.end()

    def _paint_camera_frame(self, painter: QPainter, widget) -> None:
        cx, cy, hx, hy = self.camera_frame
        w, h = widget.width(), widget.height()
        left = (cx - hx + 1.0) * 0.5 * w
        right = (cx + hx + 1.0) * 0.5 * w
        top = (1.0 - (cy + hy)) * 0.5 * h
        bottom = (1.0 - (cy - hy)) * 0.5 * h
        frame = QRectF(left, top, right - left, bottom - top)
        dark = QColor(0, 0, 0, int(255 * float(self.settings.passepartout)))
        outside = QPainterPath()
        outside.addRect(QRectF(0, 0, w, h))
        inner = QPainterPath()
        inner.addRect(frame)
        painter.fillPath(outside.subtracted(inner), dark)
        painter.setPen(QPen(QColor(0, 0, 0, 220), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(frame.adjusted(-1, -1, 1, 1))
        painter.setPen(QPen(QColor(230, 230, 230, 200), 1.0, Qt.DashLine))
        painter.drawRect(frame)

    def _scene_stats(self) -> list[str]:
        project = self.app.project
        if project is None:
            return []
        session = getattr(self.app.engine, "edit", None) if self.app.engine is not None else None
        if session is not None:
            # 编辑模式：选中的点、边、面 / 总数（和 Blender 一样）
            mesh = session.mesh
            shown_e = len(mesh.edge_sel)
            return [session.obj.name,
                    "点 %s / %s" % (format(int(mesh.vert_sel.sum()), ","), format(mesh.vert_count, ",")),
                    "边 %s / %s" % (format(int(mesh.edge_sel.sum()), ","), format(shown_e, ",")),
                    "面 %s / %s" % (format(int(mesh.face_sel.sum()), ","), format(mesh.face_count, ","))]
        objects = [o for o in project.all_objects() if o.visible]
        meshes = [o for o in objects if o.kind == "MESH"]
        tris = sum(int(o.data.triangle_count) for o in meshes)
        chosen = [o for o in objects if o.select]
        out = ["物体 %d / %d" % (len(chosen), len(objects)), "三角面 %s" % format(tris, ",")]
        active = project.active_object
        if active is not None:
            out.insert(0, active.name)
        return out

    def _paint_shape(self, painter: QPainter, widget) -> None:
        """拖动时的临时形状（视口像素 → 控件坐标），见 ui/overlay_shapes.py。"""
        from ..ui.overlay_shapes import paint_shape

        ratio = widget.width() and (self.view.width / max(1, widget.width())) or 1.0
        paint_shape(painter, self._shape, 1.0 / max(ratio, 1e-6), widget.width(), widget.height())

    # ------------------------------------------------------------------ 状态
    def save_state(self) -> dict:
        state = super().save_state()
        if self.view is not None:
            state["camera"] = self.view.camera.state()
        state["shading"] = self.shading.to_dict()
        state["overlay"] = self.overlay.to_dict()
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        state = state or {}
        self.shading.from_dict(state.get("shading"))
        self.overlay.from_dict(state.get("overlay"))
        names = ibl.list_hdris()
        if names and self.shading.hdri not in names:
            self.shading.hdri = "forest" if "forest" in names else names[0]
        if self.view is not None and state.get("camera"):
            self.view.camera.set_state(state["camera"])
            self.sync_settings()
            self.view.invalidate_camera()
