"""出图（F12）：准备场景、交给当前引擎、每帧推进、结果交给「渲染结果」编辑器。"""
from __future__ import annotations

import logging

from . import engines
from .engines import RenderJob, RenderRequest
from .scene import RenderSettings

log = logging.getLogger("splender.render")

#: 引擎设置里这些名字会带进 RenderSettings
_SHARED = ("samples", "max_bounces", "clamp_indirect", "time_limit", "seed")


def make_settings(props, engine_settings) -> RenderSettings:
    width, height = props.size
    settings = RenderSettings(width=width, height=height, exposure=float(props.exposure),
                              view_transform=str(props.view_transform), transparent=bool(props.transparent),
                              texture_size=int(props.texture_size))
    if engine_settings is not None:
        for name in _SHARED:
            if hasattr(engine_settings, name):
                setattr(settings, name, type(getattr(settings, name))(getattr(engine_settings, name)))
    return settings


def start_render(app, view_editor=None) -> RenderJob:
    """用当前视口的视角出图。失败时抛出带中文说明的异常。"""
    host = app.host
    if host is None or app.project is None:
        raise RuntimeError("没有可用的三维显示")
    host.make_current()
    engine = host.engine
    project = app.project
    props = project.render
    cls = engines.engine(props.engine) or engines.engine("PATHTRACE")
    ok, reason = cls.available()
    if not ok:
        raise RuntimeError("「%s」现在不能用：%s" % (cls.label, reason))
    view_editor = view_editor or getattr(app, "active_view3d", None)
    if view_editor is None or getattr(view_editor, "view", None) is None:
        raise RuntimeError("先打开一个 3D 视口，出图用它的视角")
    view = view_editor.view
    engine_settings = props.engine_settings(cls.idname)
    settings = make_settings(props, engine_settings)
    cancel_render(app)
    scene = None
    if cls.needs_scene:
        from .build import build_scene

        scene = build_scene(engine, project, view.camera, settings, view_editor.shading,
                            camera_object=project.active_camera)
    request = RenderRequest(scene=scene, settings=settings, engine_settings=engine_settings, app_engine=engine,
                            view=view, shading=view_editor.shading, render_props=props)
    job = cls.create_job(request)
    job.engine_label = cls.label
    app.render_job = job
    log.info("开始出图：%s，%d×%d", cls.label, settings.width, settings.height)
    host.request_frame()
    return job


def step_render(app, budget_ms: float) -> bool:
    """每帧调用。返回渲染是否还在进行。"""
    job = getattr(app, "render_job", None)
    if job is None or job.done:
        return False
    try:
        job.step(budget_ms)
    except Exception as error:  # noqa: BLE001
        log.exception("出图出错")
        try:
            job.release()
        except Exception:  # noqa: BLE001
            pass
        job.finish("出错：%s" % error)
    if job.done:
        log.info("出图结束：%s，用时 %.1f 秒%s", getattr(job, "engine_label", ""), job.elapsed,
                 ("，" + job.error) if job.error else "")
    return not job.done


def cancel_render(app) -> None:
    job = getattr(app, "render_job", None)
    if job is not None and not job.done:
        job.cancel()
        if app.host is not None:
            app.host.make_current()
            step_render(app, 1.0)
