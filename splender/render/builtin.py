"""内置渲染引擎：工作台、实时、路径追踪。"""
from __future__ import annotations

import logging
import time

import moderngl
import numpy as np

from ..core.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PropertyGroup
from . import engines
from .engines import RenderEngine, RenderJob, RenderRequest, ViewportRender
from .props import TEXTURE_SIZES
from .scene import RenderCamera, RenderSettings

log = logging.getLogger("splender.render")


# ====================================================================== 视口渲染器出图（工作台、实时）
class RasterJob(RenderJob):
    """用视口的渲染器在离屏画面上出图：画面和视口一模一样，只是分辨率按渲染设置。"""

    MAX_FRAMES = 900

    def __init__(self, request: RenderRequest, mode: str, supersample: int = 1) -> None:
        super().__init__(request)
        from ..doc.project import ViewShading

        app = request.app_engine
        source = request.view
        props = request.render_props
        self.supersample = max(1, int(supersample))
        width, height = request.settings.width, request.settings.height
        shading = ViewShading()
        if request.shading is not None:
            shading.from_dict(request.shading.to_dict())
        shading.mode = mode
        if props is not None:
            shading.view_transform = props.view_transform
            shading.exposure = props.exposure
            if mode == "MATERIAL":                     # 出图时环境作为背景（透明背景除外），和路径追踪一致
                shading.background_opacity = 0.0 if props.transparent else 1.0
                shading.background_blur = 0.0
        self.view = app.create_view(shading)
        self.view.resize(width * self.supersample, height * self.supersample)
        if source is not None:
            self.view.camera.set_state(source.camera.state())
            self.view.camera.scene_radius = source.camera.scene_radius
        self.view.invalidate_camera()
        self.frames = 0
        self._image = None
        self.status = "准备画面"

    def step(self, budget_ms: float) -> None:
        if self.done:
            return
        if self.cancelled:
            self.release()
            self.finish("已取消")
            return
        app = self.request.app_engine
        started = time.perf_counter()
        while (time.perf_counter() - started) * 1000.0 < budget_ms:
            app.frame()
            self.frames += 1
            if self.frames >= 3 and not app.needs_render:
                break
        self.progress = min(0.95, self.frames / 60.0)
        if (self.frames >= 3 and not app.needs_render) or self.frames >= self.MAX_FRAMES:
            image = app.renderer.read_pixels(self.view)
            ss = self.supersample
            if ss > 1:
                h, w = image.shape[0] // ss, image.shape[1] // ss
                image = image[:h * ss, :w * ss].reshape(h, ss, w, ss, -1).mean(axis=(1, 3)).round().astype(np.uint8)
            if image.shape[2] == 3:
                image = np.concatenate([image, np.full(image.shape[:2] + (1,), 255, np.uint8)], axis=2)
            image[:, :, 3] = 255
            self._image = np.ascontiguousarray(image)
            self.progress = 1.0
            self.status = "完成"
            self.release()
            self.finish()

    def preview(self) -> np.ndarray | None:
        return self._image

    def release(self) -> None:
        if self.view is not None:
            self.request.app_engine.destroy_view(self.view)
            self.view = None


class RealtimeSettings(PropertyGroup):
    supersample = EnumProperty("超采样", items=[("1", "关", ""), ("2", "2×2", ""), ("3", "3×3", "")], default="2",
                               description="先按几倍分辨率画再缩小，边缘更干净")


@engines.register_engine
class WorkbenchEngine(RenderEngine):
    idname = "WORKBENCH"
    label = "工作台"
    description = "实体着色，看造型"
    icon = "shading.solid"
    order = 10
    needs_scene = False

    @classmethod
    def create_job(cls, request: RenderRequest) -> RenderJob:
        return RasterJob(request, "SOLID", 2)


@engines.register_engine
class RealtimeEngine(RenderEngine):
    idname = "REALTIME"
    label = "实时"
    description = "和视口材质预览相同的实时渲染，秒出图"
    icon = "shading.material"
    order = 20
    needs_scene = False
    settings_class = RealtimeSettings

    @classmethod
    def create_job(cls, request: RenderRequest) -> RenderJob:
        settings = request.engine_settings
        return RasterJob(request, "MATERIAL", int(settings.supersample) if settings is not None else 2)


# ====================================================================== 路径追踪
class PathTraceSettings(PropertyGroup):
    samples = IntProperty("出图采样数", default=256, min=1, max=65536,
                          description="每个像素采样多少次。越多越干净，也越慢")
    viewport_samples = IntProperty("视口采样数", default=1024, min=1, max=65536,
                                   description="视口里渐进渲染到多少次采样后停下")
    max_bounces = IntProperty("最多反弹", default=6, min=0, max=64, description="光线最多弹几次")
    clamp_indirect = FloatProperty("间接光上限", default=10.0, min=0.0, max=10000.0, precision=1,
                                   description="压掉过亮的噪点。0 表示不限")
    time_limit = FloatProperty("时间上限", default=0.0, min=0.0, max=86400.0, unit="s", precision=0,
                               description="出图最多渲染多久。0 表示不限")
    seed = IntProperty("随机种子", default=0, min=0, max=1 << 30)
    viewport_scale = EnumProperty("视口渲染精度", items=[("0.25", "1/4", ""), ("0.5", "1/2", ""), ("1.0", "全", "")],
                                  default="0.5", description="视口里按多大比例渲染。越小越快")
    viewport_texture_size = EnumProperty("视口贴图精度", items=TEXTURE_SIZES, default="2048",
                                         description="视口渲染时图层合成成多大的贴图")
    viewport_budget_ms = FloatProperty("视口每帧用时", default=12.0, min=1.0, max=200.0, unit="ms", precision=0,
                                       description="视口渲染每帧最多花多少时间")
    denoise = BoolProperty("降噪", default=True, description="出图完成后去掉噪点，少量采样也能得到干净的图")
    denoise_viewport = BoolProperty("视口降噪", default=True,
                                    description="视口渲染时，采样数到 1、4、16、64……时各降噪一次")
    denoise_device = EnumProperty("降噪设备", items=[("AUTO", "自动", "有显卡用显卡，否则用处理器"),
                                                    ("GPU", "显卡", "用 CUDA 显卡"),
                                                    ("CPU", "处理器", "用处理器")], default="AUTO")


class PathTraceJob(RenderJob):
    PREVIEW_EVERY = 0.4

    def __init__(self, request: RenderRequest) -> None:
        super().__init__(request)
        from .pathtracer import PathTracer

        self.tracer = PathTracer(request.app_engine.ctx)
        self.tracer.set_scene(request.scene)
        self._preview = None
        self._result = None
        self._last = 0.0
        self.status = "开始渲染"

    def step(self, budget_ms: float) -> None:
        if self.done:
            return
        if self.cancelled:
            self._snapshot()
            self.release()
            self.finish("已取消")
            return
        tracer = self.tracer
        tracer.render_pass(budget_ms)
        settings = tracer.settings
        self.progress = min(1.0, tracer.samples_done / max(1, int(settings.samples)))
        self.status = "采样 %d / %d" % (tracer.samples_done, int(settings.samples))
        if tracer.finished:
            es = self.request.engine_settings
            note = ""
            if es is None or bool(getattr(es, "denoise", True)):
                from .denoise import available

                ok, reason = available()
                if ok:
                    try:
                        stats = tracer.denoise_now(str(getattr(es, "denoise_device", "AUTO")))
                        note = "，已降噪（%s，%.0f 毫秒）" % ("显卡" if stats["device"] == "CUDA" else "处理器", stats["ms"])
                    except Exception as exc:  # noqa: BLE001
                        log.exception("出图降噪失败")
                        note = "，降噪没有成功：%s" % exc
                else:
                    note = "，%s" % reason
            self._snapshot()
            self._result = tracer.read_image(linear=True)
            self.progress = 1.0
            self.status = "完成：%d 次采样%s" % (tracer.samples_done, note)
            self.release()
            self.finish()

    def _snapshot(self) -> None:
        if self.tracer is not None:
            self._preview = self.tracer.read_image()
            self._last = time.perf_counter()

    def preview(self) -> np.ndarray | None:
        if self.tracer is not None and time.perf_counter() - self._last > self.PREVIEW_EVERY:
            self._snapshot()
        return self._preview

    def result(self) -> np.ndarray | None:
        return self._result

    def release(self) -> None:
        if self.tracer is not None:
            self.tracer.release()
            self.tracer = None


_BLIT_VERT = """#version 430
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
_BLIT_FRAG = """#version 430
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 o_color;
void main() { o_color = vec4(texture(u_tex, v_uv).rgb, 1.0); }
"""


class PathTraceViewport(ViewportRender):
    """视口里的渐进路径追踪。场景（含合成贴图）只在模型或图层内容变了时重建；转视角只换相机。"""

    def __init__(self, request: RenderRequest) -> None:
        from .pathtracer import PathTracer

        ctx = request.app_engine.ctx
        self.ctx = ctx
        self.tracer = PathTracer(ctx)
        self.ready = False
        self._program = ctx.program(vertex_shader=_BLIT_VERT, fragment_shader=_BLIT_FRAG)
        self._vao = ctx.vertex_array(self._program, [])
        self._status = ""
        self._denoise = True
        self._denoise_device = "AUTO"
        self._next_denoise = 1

    def _settings(self, request: RenderRequest) -> RenderSettings:
        view = request.view
        es = request.engine_settings
        shading = request.shading
        scale = float(es.viewport_scale) if es is not None else 0.5
        return RenderSettings(width=max(16, int(view.width * scale)), height=max(16, int(view.height * scale)),
                              samples=int(es.viewport_samples) if es is not None else 1024,
                              max_bounces=int(es.max_bounces) if es is not None else 6,
                              clamp_indirect=float(es.clamp_indirect) if es is not None else 10.0,
                              exposure=float(shading.exposure), view_transform=str(shading.view_transform),
                              texture_size=int(es.viewport_texture_size) if es is not None else 2048,
                              seed=int(es.seed) if es is not None else 0)

    #: 采样数到这些档位时降噪一次（最后一次在全部采样做完时）
    DENOISE_STEPS = (1, 4, 16, 64, 256, 1024, 4096, 16384)

    def sync(self, request: RenderRequest, changes: set[str]) -> None:
        from .build import _environment, build_scene

        es = request.engine_settings
        self._denoise = bool(getattr(es, "denoise_viewport", True)) if es is not None else True
        self._denoise_device = str(getattr(es, "denoise_device", "AUTO")) if es is not None else "AUTO"
        self._next_denoise = 1
        view = request.view
        settings = self._settings(request)
        if not self.ready or changes & {"content", "objects"}:
            started = time.perf_counter()
            scene = build_scene(request.app_engine, request.app_engine.project, view.camera, settings,
                                request.shading, aspect=view.aspect)
            self.tracer.set_scene(scene)
            self.ready = True
            log.info("视口渲染场景重建，用时 %.2f 秒", time.perf_counter() - started)
            return
        if changes & {"size", "settings"}:
            self.tracer.settings = settings
            self.tracer.update_settings(settings)
        if "environment" in changes:
            self.tracer.scene.environment = _environment(request.shading)
            self.tracer.update_environment(self.tracer.scene)
        if "camera" in changes or changes & {"size"}:
            self.tracer.update_camera(RenderCamera.from_view_camera(view.camera, view.aspect))
        if "display" in changes:
            self.tracer.settings.exposure = float(request.shading.exposure)
            self.tracer.settings.view_transform = str(request.shading.view_transform)

    def draw(self, view, budget_ms: float) -> bool:
        tracer = self.tracer
        if not self.ready:
            return False
        if not tracer.finished:
            tracer.render_pass(budget_ms)
        busy = False
        if self._denoise:
            from .denoise import available

            if tracer.denoise_poll():
                pass
            if not tracer.denoising and available()[0]:
                total = max(1, int(tracer.settings.samples))
                done = tracer.samples_done
                if done >= min(self._next_denoise, total) and tracer.denoised_samples < done:
                    tracer.denoise_start(self._denoise_device)
                    self._next_denoise = next((s for s in self.DENOISE_STEPS if s > done), total)
            busy = tracer.denoising or (tracer.finished and tracer.denoised_samples < tracer.samples_done)
        tracer.resolve()
        target = view.msaa_fbo or view.fbo
        view.fbo.use()
        self.ctx.viewport = (0, 0, view.width, view.height)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
        tex = tracer.display_texture
        tex.use(0)
        self._program["u_tex"] = 0
        self._vao.render(moderngl.TRIANGLES, vertices=3)
        del target
        self._status = "路径追踪 · %d / %d%s" % (tracer.samples_done, int(tracer.settings.samples),
                                                  " · 已降噪" if tracer.denoised_samples else "")
        return not tracer.finished or busy

    def status(self) -> str:
        return self._status

    def release(self) -> None:
        self.tracer.release()
        self._vao.release()
        self._program.release()


@engines.register_engine
class PathTraceEngine(RenderEngine):
    idname = "PATHTRACE"
    label = "路径追踪"
    description = "物理正确的光线追踪，视口里渐进预览"
    icon = "render"
    order = 30
    supports_viewport = True
    settings_class = PathTraceSettings

    @classmethod
    def create_job(cls, request: RenderRequest) -> RenderJob:
        return PathTraceJob(request)

    @classmethod
    def create_viewport(cls, request: RenderRequest) -> ViewportRender | None:
        return PathTraceViewport(request)
