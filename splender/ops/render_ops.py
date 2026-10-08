"""出图相关的操作：渲染（F12）、停止、看结果（F11）、保存图片。"""
from __future__ import annotations

from ..core import ops
from ..core.ops import CANCELLED, FINISHED, Operator


def _open_result_window(ctx) -> None:
    app = ctx.app
    window = getattr(app, "render_window", None)
    try:
        if window is not None and window.isVisible():
            window.raise_()
            return
    except RuntimeError:
        window = None
    from ..ui.window import PopoutWindow

    window = PopoutWindow(app, ctx.wm, {"type": "area", "editor": "RENDER_RESULT", "state": {}}, title="渲染结果")
    window.resize(1100, 720)
    window.show()
    app.render_window = window


@ops.register
class RenderRender(Operator):
    idname = "render.render"
    label = "渲染"
    description = "用当前视口的视角和「渲染」设置出图"
    icon = "render"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.app.project is not None and getattr(ctx.app, "host", None) is not None

    def execute(self, ctx) -> str:
        from ..render.session import start_render

        view_editor = ctx.editor if getattr(ctx.editor, "idname", "") == "VIEW_3D" else None
        try:
            start_render(ctx.app, view_editor)
        except Exception as error:  # noqa: BLE001
            self.report(ctx, str(error), "WARNING")
            return CANCELLED
        if not getattr(ctx.app, "headless_render", False):
            _open_result_window(ctx)
        return FINISHED


@ops.register
class RenderCancel(Operator):
    idname = "render.cancel"
    label = "停止渲染"

    @classmethod
    def poll(cls, ctx) -> bool:
        job = getattr(ctx.app, "render_job", None)
        return job is not None and not job.done

    def execute(self, ctx) -> str:
        from ..render.session import cancel_render

        cancel_render(ctx.app)
        return FINISHED


@ops.register
class RenderViewResult(Operator):
    idname = "render.view_result"
    label = "查看渲染结果"
    description = "打开渲染结果窗口"

    def execute(self, ctx) -> str:
        _open_result_window(ctx)
        return FINISHED


@ops.register
class RenderSaveImage(Operator):
    idname = "render.save_image"
    label = "保存渲染图片"
    description = "存成 PNG（8 位，可带透明）或 EXR（32 位浮点，保留高光细节）"
    icon = "save"

    @classmethod
    def poll(cls, ctx) -> bool:
        job = getattr(ctx.app, "render_job", None)
        return job is not None and job.preview() is not None

    def invoke(self, ctx, event) -> str:
        from PySide6.QtWidgets import QFileDialog

        app = ctx.app
        name = (app.project.name if app.project is not None else "渲染") + "_渲染.png"
        folder = app.prefs.files.last_folder if hasattr(app.prefs.files, "last_folder") else ""
        path, _filter = QFileDialog.getSaveFileName(None, "保存渲染图片", str(folder or "") + "/" + name,
                                                    "PNG 图片 (*.png);;OpenEXR 图片 (*.exr)")
        if not path:
            return CANCELLED
        return self._save(ctx, path)

    def execute(self, ctx) -> str:
        path = getattr(self, "path", "")
        return self._save(ctx, path) if path else CANCELLED

    def _save(self, ctx, path: str) -> str:
        from ..render.output import save_image

        job = ctx.app.render_job
        try:
            written = save_image(path, job.preview(), job.result())
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "保存失败：%s" % error, "ERROR")
            return CANCELLED
        self.report(ctx, "已保存：%s" % written)
        return FINISHED
