"""UV / 图像编辑器：把贴图按 UV 铺开看，叠加 UV 线框。"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QLabel

from ..core import ops, registry
from ..core.keymap import MOUSEMOVE, PRESS, RELEASE
from ..core.ops import CANCELLED, FINISHED, RUNNING_MODAL, Operator
from ..core.props import BoolProperty, EnumProperty, FloatProperty, PropertyGroup
from ..doc.project import CHANNEL_IDS, CHANNEL_LABELS, CHANNELS
from ..ui import theme
from ..tools.paint_tools import STROKE_TOOLS
from ..ui.editor import Editor


class ImageSettings(PropertyGroup):
    channel = EnumProperty("通道", items=[(c[0], c[1], "") for c in CHANNELS], default="basecolor",
                           description="查看哪个通道")
    show_wire = BoolProperty("UV 线框", default=True, description="在贴图上叠加模型的 UV 线框")
    wire_opacity = FloatProperty("线框不透明度", default=0.35, min=0.0, max=1.0, precision=2, subtype="FACTOR",
                                 description="UV 线框有多明显。缩小到线挤在一起时会再自动变淡")


@registry.register_editor
class ImageEditor(Editor):
    idname = "IMAGE_EDITOR"
    label = "UV / 图像"
    icon = "editor.image"
    category = "通用"
    order = 15
    description = "把贴图按 UV 铺开查看"
    keymaps = ["Image"]
    has_toolbar = True

    def __init__(self, app, area=None) -> None:
        self.settings = ImageSettings()
        self.view = None
        self.widget = None
        self._mouse = None            # 最近一次鼠标位置（控件坐标），画笔刷圈用
        self._shape = None            # 拖动工具的临时叠加（视图像素），比如渐变的那条线
        super().__init__(app, area)

    def set_overlay_shape(self, shape) -> None:
        """拖动时的临时形状（选框、椭圆、刷选圆、套索、虚线……，见 ui/overlay_shapes.py），None 清掉。坐标是视图像素。"""
        if shape == self._shape:
            return
        self._shape = shape
        if self.widget is not None:
            self.widget.update()

    def build_main(self):
        host = getattr(self.app, "host", None)
        if host is None:
            label = QLabel("这台电脑上没有可用的三维显示")
            label.setAlignment(Qt.AlignCenter)
            return label
        from ..ui.glhost import ViewportWidget

        host.make_current()
        self.view = host.engine.create_view2d()
        self.widget = ViewportWidget(host, self.view)
        self.widget.on_paint_overlay = self._paint_overlay
        self.install_input(self.widget)
        from .viewport3d import _LeaveFilter
        self._leave_filter = _LeaveFilter(self)
        self.widget.installEventFilter(self._leave_filter)
        self.settings.changed.connect(self._on_settings)
        return self.widget

    def dispose(self) -> None:
        if self.widget is not None:
            self.widget.dispose()
            self.widget = None
            self.view = None
        super().dispose()

    def draw_header(self, layout, ctx) -> None:
        """和 Substance Painter 的二维视图一样：通道是一个紧凑的下拉框（C 轮换）；UV 线框收在叠加层的弹出面板里。"""
        layout.prop(self.settings, "channel", text="")
        layout.operator("image.view_all", text="", icon="focus")
        layout.menu("PAINT_MT_select", text="选择")
        layout.menu("LAYERS_MT_adjust", text="调整")
        layout.menu("LAYERS_MT_filter", text="滤镜")
        from ..ops.select_ops import draw_tool_header

        draw_tool_header(layout, ctx.app.tool_settings)
        layout.stretch()
        ts = ctx.texture_set
        if ts is not None:
            layout.label("%s · %d × %d" % (ts.name, ts.size, ts.size), role="dim")
        row = layout.row(align=True)
        row.prop(self.settings, "show_wire", text="UV 线框", icon="overlays", icon_only=True)
        row.popover("", "chevron.down", self._draw_overlays)

    def _draw_overlays(self, layout, ctx) -> None:
        layout.label("叠加层", role="title")
        layout.prop(self.settings, "show_wire")
        layout.prop(self.settings, "wire_opacity", slider=True)

    def _on_settings(self, name: str) -> None:
        if self.view is None:
            return
        self.view.channel = self.settings.channel
        self.view.show_wire = bool(self.settings.show_wire)
        self.view.wire_opacity = float(self.settings.wire_opacity)
        self.view.dirty = True
        self.app.host.request_frame()

    def pixel(self, event) -> tuple[float, float]:
        return self.widget.to_pixels(event.x, event.y)

    def handle_event(self, event) -> bool:
        if self.widget is not None and (event.type == MOUSEMOVE or event.source is self.widget):
            self._mouse = (float(event.x), float(event.y))
            if self.app.tool_settings.tool.startswith("paint."):
                self.widget.update()
        return super().handle_event(event)

    def clear_cursor(self) -> None:
        self._mouse = None
        if self.widget is not None:
            self.widget.update()

    def on_enter(self) -> None:
        painting = self.app.tool_settings.tool.startswith("paint.")
        if self.wm is not None and self.app.prefs.interface.status_hints and painting:
            self.wm.set_hint("左键在 UV 上画 · Ctrl 左键擦除 · 中键平移 · 滚轮缩放 · [ ] 调笔刷大小")

    def _uv_world(self, ts) -> float:
        """这套贴图每单位 UV 平均多长的表面（按场景尺寸画笔刷圈用），模型不变就不重算。"""
        project = self.app.project
        if ts is None or project is None:
            return 0.0
        key = (ts.uid, id(project), tuple(id(o.data) for o in project.objects))
        cached = getattr(self, "_uv_world_cache", None)
        if cached is None or cached[0] != key:
            from ..engine.export_normal import surface_scale
            cached = (key, surface_scale(project, ts)[0])
            self._uv_world_cache = cached
        return cached[1]

    def view_changed(self) -> None:
        self.view.dirty = True
        self.app.host.request_frame()

    def refresh(self) -> None:
        super().refresh()
        if self.view is not None:
            self.view.dirty = True
            self.app.host.request_frame()

    def _paint_overlay(self, widget) -> None:
        if self.view is None:
            return
        painter = QPainter(widget)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.setFont(theme.font("font.small"))
        margin = theme.px(10)
        ts = self.app.project.active_texture_set if self.app.project is not None else None
        scale = (self.view.zoom / ts.size * 100.0) if ts is not None else 0.0
        text = "%s · %.0f%%" % (CHANNEL_LABELS.get(self.view.channel, ""), scale)
        rect = QRectF(margin, margin, widget.width() - 2 * margin, theme.px(18))
        painter.setPen(QColor(0, 0, 0, 150))
        painter.drawText(rect.translated(1, 1), Qt.AlignLeft | Qt.AlignVCenter, text)
        painter.setPen(theme.qcolor("text.dim"))
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter, text)
        if self._shape is not None:
            from ..ui.overlay_shapes import paint_shape

            paint_shape(painter, self._shape, 1.0 / max(widget.devicePixelRatioF(), 1e-6), widget.width(),
                        widget.height())
        tools = self.app.tool_settings
        if tools.tool in ("paint.clone", "paint.heal") and tools.effect.has_source:
            ratio = max(widget.devicePixelRatioF(), 1e-6)
            view = self.view
            sx = ((tools.effect.source_u - view.center[0]) * view.zoom + view.width * 0.5) / ratio
            sy = (view.height * 0.5 - (tools.effect.source_v - view.center[1]) * view.zoom) / ratio
            painter.setRenderHint(QPainter.Antialiasing, True)
            for color, width in ((QColor(0, 0, 0, 170), 3.0), (QColor(255, 255, 255, 235), 1.2)):
                painter.setPen(QPen(color, width))
                painter.drawLine(QPointF(sx - 9, sy), QPointF(sx + 9, sy))
                painter.drawLine(QPointF(sx, sy - 9), QPointF(sx, sy + 9))
        if (self._mouse is not None and tools.tool in STROKE_TOOLS and self.app.prefs.viewport.cursor_ring):
            brush = tools.active_brush()
            ratio = max(widget.devicePixelRatioF(), 1e-6)
            if brush.size_unit == "SCENE":
                per_uv = self._uv_world(ts)
                radius = brush.scene_size * 0.5 / max(per_uv, 1e-12) * self.view.zoom / ratio if per_uv else 0.0
            else:
                radius = max(1.0, brush.size) * 0.5 / ratio
            if radius > 0.5:
                painter.setRenderHint(QPainter.Antialiasing, True)
                x, y = self._mouse
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QColor(0, 0, 0, 140))
                painter.drawEllipse(QRectF(x - radius - 1, y - radius - 1, 2 * radius + 2, 2 * radius + 2))
                painter.setPen(QColor(255, 255, 255, 220))
                painter.drawEllipse(QRectF(x - radius, y - radius, 2 * radius, 2 * radius))
        painter.end()

    def save_state(self) -> dict:
        state = super().save_state()
        if self.view is not None:
            state["view"] = self.view.state()
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        data = (state or {}).get("view")
        if self.view is not None and data:
            self.view.set_state(data)
            with self.settings.changed.block():
                self.settings.channel = self.view.channel
                self.settings.show_wire = self.view.show_wire
                self.settings.wire_opacity = self.view.wire_opacity


def _image(ctx):
    editor = ctx.editor
    return editor if getattr(editor, "idname", "") == "IMAGE_EDITOR" and getattr(editor, "view", None) is not None else None


class _ImageOp(Operator):
    searchable = False

    @classmethod
    def poll(cls, ctx) -> bool:
        return _image(ctx) is not None


@ops.register
class ImagePan(_ImageOp):
    idname = "image.pan"
    label = "平移图像"

    def invoke(self, ctx, event) -> str:
        self.editor = _image(ctx)
        self.button = event.type
        self.last = self.editor.pixel(event)
        return RUNNING_MODAL

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE:
            if event.source is self.editor.widget:
                x, y = self.editor.pixel(event)
                self.editor.view.pan(x - self.last[0], y - self.last[1])
                self.last = (x, y)
                self.editor.view_changed()
            return RUNNING_MODAL
        if event.type == self.button and event.value == RELEASE:
            return FINISHED
        if event.type == "ESC" and event.value == PRESS:
            return CANCELLED
        return RUNNING_MODAL


@ops.register
class ImageZoomDrag(ImagePan):
    idname = "image.zoom_drag"
    label = "拖动缩放图像"

    def invoke(self, ctx, event) -> str:
        result = super().invoke(ctx, event)
        self.anchor = self.last
        return result

    def modal(self, ctx, event) -> str:
        if event.type == MOUSEMOVE and event.source is self.editor.widget:
            x, y = self.editor.pixel(event)
            self.editor.view.zoom_at(2.0 ** (-(y - self.last[1]) * 0.01), *self.anchor)
            self.last = (x, y)
            self.editor.view_changed()
            return RUNNING_MODAL
        return super().modal(ctx, event)


@ops.register
class ImageZoom(_ImageOp):
    idname = "image.zoom"
    label = "缩放图像"
    delta = FloatProperty("步数", default=1.0)

    def invoke(self, ctx, event) -> str:
        editor = _image(ctx)
        steps = float(event.wheel) if event is not None and getattr(event, "wheel", 0.0) else float(self.delta)
        nav = ctx.app.prefs.navigation
        if nav.invert_zoom:
            steps = -steps
        x, y = editor.pixel(event) if event is not None else (editor.view.width / 2, editor.view.height / 2)
        editor.view.zoom_at(float(nav.zoom_speed) ** steps, x, y)
        editor.view_changed()
        return FINISHED


@ops.register
class ImageViewAll(_ImageOp):
    idname = "image.view_all"
    label = "查看全部"
    description = "把整张贴图放进视图"

    def execute(self, ctx) -> str:
        editor = _image(ctx)
        editor.view.fit()
        editor.view_changed()
        return FINISHED


@ops.register
class ImageChannelCycle(_ImageOp):
    idname = "image.channel_cycle"
    label = "切换查看的通道"

    def execute(self, ctx) -> str:
        editor = _image(ctx)
        index = (CHANNEL_IDS.index(editor.settings.channel) + 1) % len(CHANNEL_IDS)
        editor.settings.channel = CHANNEL_IDS[index]
        return FINISHED
