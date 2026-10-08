"""烘焙模型贴图、停止烘焙。"""
from __future__ import annotations

from ..core import ops
from ..core.ops import CANCELLED, FINISHED, Operator


def _engine(ctx):
    host = getattr(ctx.app, "host", None)
    if host is None:
        return None
    host.make_current()
    return host.engine


@ops.register
class BakeMeshMaps(Operator):
    idname = "bake.mesh_maps"
    label = "烘焙模型贴图"
    description = "按模型的形状算出遮蔽、曲率、厚度、法线和位置贴图。生成器蒙版和智能材质靠它们决定盖住哪里"
    icon = "bake"

    @classmethod
    def poll(cls, ctx) -> bool:
        app = ctx.app
        return (ctx.texture_set is not None and getattr(app, "host", None) is not None
                and app.tool_settings.mode in ("PAINT", "OBJECT"))

    def execute(self, ctx) -> str:
        engine = _engine(ctx)
        if engine is None:
            return CANCELLED
        try:
            engine.bake_start(ctx.texture_set)
        except ValueError as exc:
            self.report(ctx, str(exc), "WARNING")
            return CANCELLED
        ctx.app.notify("meshmaps")
        ctx.app.request_frame()
        return FINISHED


@ops.register
class BakeCancel(Operator):
    idname = "bake.cancel"
    label = "停止烘焙"
    description = "停下正在进行的烘焙，已有的模型贴图不变"
    icon = "stop"

    @classmethod
    def poll(cls, ctx) -> bool:
        engine = getattr(ctx.app, "engine", None)
        return engine is not None and engine.bake is not None

    def execute(self, ctx) -> str:
        engine = _engine(ctx)
        if engine is None:
            return CANCELLED
        engine.bake_cancel()
        ctx.app.notify("meshmaps")
        return FINISHED
