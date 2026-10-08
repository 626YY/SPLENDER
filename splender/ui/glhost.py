"""引擎宿主：引擎用的显卡上下文、帧调度，以及把引擎画面贴到屏幕上的视口控件。

所有三维渲染都在引擎自己的上下文里完成，画进每个视口各自的贴图；
界面上的视口控件只做一件事：把这张贴图贴出来。各上下文共享资源，所以不需要拷贝。
"""
from __future__ import annotations

import logging
import time
from collections import deque

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from ..core.signals import Signal
from ..paths import ensure_vendor_path

log = logging.getLogger("splender.ui.glhost")


def configure_surface_format(vsync: bool = True) -> None:
    """在创建 QApplication 之前调用。"""
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(4, 5)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapInterval(1 if vsync else 0)
    QSurfaceFormat.setDefaultFormat(fmt)


class EngineHost(QObject):
    """拥有引擎和它的显卡上下文。界面通过 request_frame() 请求推进一帧。"""

    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self.context = QOpenGLContext()
        self.context.setFormat(QSurfaceFormat.defaultFormat())
        share = QOpenGLContext.globalShareContext()
        if share is not None:
            self.context.setShareContext(share)
        if not self.context.create():
            raise RuntimeError("无法创建 OpenGL 上下文")
        self.surface = QOffscreenSurface()
        self.surface.setFormat(self.context.format())
        self.surface.create()
        self.make_current()
        ensure_vendor_path()
        import moderngl

        self.mgl = moderngl.create_context()
        version = self.mgl.version_code
        if version < 430:
            raise RuntimeError("显卡驱动只支持 OpenGL %d.%d，需要 4.3 以上" % (version // 100, version % 100 // 10))
        from ..engine.engine import Engine

        self.engine = Engine(self.mgl, app.prefs, app.history)
        self.engine._make_current = self.make_current
        self.widgets: list["ViewportWidget"] = []
        self._scheduled = False
        self._closing = False
        #: 每帧结束后发出（给性能面板等）
        self.frame_done = Signal()
        self.present_ms = deque(maxlen=240)      # 输入事件到画面交给系统显示的间隔
        self._pending_latency: list[float] = []

    def make_current(self) -> None:
        self.context.makeCurrent(self.surface)

    # ---- 帧调度 ----
    def request_frame(self) -> None:
        if not self._scheduled and not self._closing:
            self._scheduled = True
            QTimer.singleShot(0, self.tick)

    def tick(self) -> None:
        self._scheduled = False
        if self._closing:
            return
        self.make_current()
        try:
            rendered = self.engine.frame()
        except Exception:  # noqa: BLE001
            log.exception("引擎推进一帧时出错")
            rendered = []
        rendering = False
        if getattr(self.app, "render_job", None) is not None:
            from ..render.session import step_render

            budget = float(getattr(self.app.prefs.viewport, "render_budget_ms", 30.0))
            if self.engine.stroke is not None:
                budget = min(budget, 4.0)
            rendering = step_render(self.app, budget)
        for widget in list(self.widgets):
            if widget.view in rendered:
                widget.present_event_time = self._take_event_time()
                widget.update()
        self.frame_done.emit()
        if self.engine.needs_frame or rendering:
            if rendered:
                self.request_frame()
            elif not self._scheduled:
                self._scheduled = True
                QTimer.singleShot(2, self.tick)       # 只有后台工作时不空转

    def note_input(self, when: float) -> None:
        """记下一次输入事件的到达时间，用于统计到画面呈现的延迟。"""
        self._pending_latency.append(when)

    def _take_event_time(self) -> float:
        if not self._pending_latency:
            return 0.0
        oldest = self._pending_latency[0]
        self._pending_latency.clear()
        return oldest

    def run_now(self, fn):
        """在引擎上下文里立刻执行一个函数（载入工程、导出等一次性操作）。"""
        self.make_current()
        return fn()

    def shutdown(self) -> None:
        self._closing = True
        self.make_current()
        try:
            self.engine.release()
        except Exception:  # noqa: BLE001
            log.exception("引擎关闭时出错")


_BLIT_VERT = """#version 330
in vec2 in_p; out vec2 v_uv;
void main(){ v_uv = in_p * 0.5 + 0.5; gl_Position = vec4(in_p, 0.0, 1.0); }
"""
_BLIT_FRAG = """#version 330
in vec2 v_uv; out vec4 o_color; uniform sampler2D u_tex;
void main(){ o_color = vec4(texture(u_tex, v_uv).rgb, 1.0); }
"""


class ViewportOverlay(QWidget):
    """叠在视口画面上的透明层，画文字提示。不接收鼠标和焦点，输入直接落到下面的视口。"""

    def __init__(self, viewport: "ViewportWidget") -> None:
        super().__init__(viewport)
        self.viewport = viewport
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.NoFocus)

    def paintEvent(self, event) -> None:  # noqa: N802
        callback = self.viewport.on_paint_overlay
        if callback is None:
            return
        try:
            callback(self)
        except Exception:  # noqa: BLE001
            log.exception("绘制视口叠加内容时出错")


class ViewportWidget(QOpenGLWidget):
    """把引擎的一个视口画面贴到屏幕上。输入事件由所属编辑器的 install_input 接管。"""

    def __init__(self, host: EngineHost, view, parent=None) -> None:
        super().__init__(parent)
        self.host = host
        self.view = view
        self._ctx = None
        self._program = None
        self._vao = None
        self._wrapped = None
        self._wrapped_key = None
        self.present_event_time = 0.0
        self.on_paint_overlay = None          # 可选：在叠加层上用 QPainter 画文字的回调，参数是叠加层控件
        self.setMinimumSize(64, 64)
        self.overlay = ViewportOverlay(self)
        host.widgets.append(self)
        self.frameSwapped.connect(self._on_swapped)

    # ---- 尺寸换算 ----
    def pixel_ratio(self) -> float:
        return float(self.devicePixelRatioF())

    def to_pixels(self, x: float, y: float) -> tuple[float, float]:
        ratio = self.pixel_ratio()
        return x * ratio, y * ratio

    # ---- Qt 回调 ----
    def initializeGL(self) -> None:  # noqa: N802
        import moderngl
        import numpy as np

        self._ctx = moderngl.create_context()
        self._program = self._ctx.program(vertex_shader=_BLIT_VERT, fragment_shader=_BLIT_FRAG)
        buffer = self._ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes())
        self._vao = self._ctx.vertex_array(self._program, [(buffer, "2f", "in_p")])
        self._wrapped = None
        self._wrapped_key = None

    def resizeGL(self, width: int, height: int) -> None:  # noqa: N802
        ratio = self.pixel_ratio()
        pw, ph = max(8, int(round(width * ratio))), max(8, int(round(height * ratio)))
        self.host.make_current()
        self.view.resize(pw, ph)
        self.makeCurrent()
        self.host.request_frame()

    def paintGL(self) -> None:  # noqa: N802
        view = self.view
        if self._ctx is None or view.color_tex is None:
            return
        import moderngl

        key = (view.color_tex.glo, view.width, view.height)
        if key != self._wrapped_key:
            self._wrapped = self._ctx.external_texture(view.color_tex.glo, (view.width, view.height), 4, 0, "f1")
            self._wrapped_key = key
        fbo = self._ctx.detect_framebuffer(self.defaultFramebufferObject())
        fbo.use()
        ratio = self.pixel_ratio()
        self._ctx.viewport = (0, 0, int(round(self.width() * ratio)), int(round(self.height() * ratio)))
        self._ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self._wrapped.use(0)
        self._program["u_tex"] = 0
        self._vao.render(moderngl.TRIANGLES)
        if self.on_paint_overlay is not None:
            self.overlay.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())

    def _on_swapped(self) -> None:
        if self.present_event_time:
            self.host.present_ms.append((time.perf_counter() - self.present_event_time) * 1000.0)
            self.present_event_time = 0.0

    def dispose(self) -> None:
        if self in self.host.widgets:
            self.host.widgets.remove(self)
        self.host.make_current()
        self.host.engine.destroy_view(self.view)
