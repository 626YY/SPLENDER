"""纹理集：改分辨率（已经画的内容按新尺寸重新采样，可以撤销）。"""
from __future__ import annotations

from ..core import ops, registry
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import EnumProperty, IntProperty
from ..doc.project import RESOLUTION_ITEMS


def _texture_set(ctx, uid: int):
    project = getattr(ctx.app, "project", None)
    if project is None:
        return None
    return project.texture_set(int(uid)) if uid else project.active_texture_set


@ops.register
class TextureSetResize(Operator):
    idname = "texture_set.resize"
    label = "改分辨率"
    description = "整套贴图换成新的大小，已经画的内容按新尺寸重新采样"
    redo = False
    resolution = EnumProperty("分辨率", items=RESOLUTION_ITEMS, default="4096")
    uid = IntProperty("纹理集", default=0, hidden=True, description="0 表示当前纹理集")

    @classmethod
    def poll(cls, ctx) -> bool:
        return getattr(ctx.app, "engine", None) is not None and _texture_set(ctx, 0) is not None

    def execute(self, ctx) -> str:
        from ..ui.busy import busy_cursor

        ts = _texture_set(ctx, self.uid)
        app = ctx.app
        engine = app.engine
        if ts is None or ts.resolution == self.resolution:
            return CANCELLED
        old_res, new_res = ts.resolution, str(self.resolution)
        host = getattr(app, "host", None)
        if host is not None:
            host.make_current()
        with busy_cursor():
            pairs = engine.resize_texture_set(ts, int(new_res))
        state = {"new": True}

        def apply(resolution: str, use_new: bool) -> None:
            if host is not None:
                host.make_current()
            ts.resolution = resolution
            engine.apply_texture_set_size(ts, pairs, use_new)
            state["new"] = use_new
            app.request_frame()
            app.notify("layers")

        apply(new_res, True)
        history = getattr(app, "history", None)
        if history is not None:
            nbytes = sum(new.page_count() for _old, new in pairs.values() if new is not None) * 65536
            history.push("改分辨率", lambda: apply(old_res, False), lambda: apply(new_res, True), nbytes,
                         lambda: engine.free_texture_set_stores(pairs, keep_new=state["new"]))
        labels = dict((k, v) for k, v, _d in RESOLUTION_ITEMS)
        self.report(ctx, "「%s」换成 %s（%d × %d）" % (ts.name, labels.get(new_res, new_res), ts.size, ts.size))
        return FINISHED


@registry.register_menu
class TextureSetResolutionMenu(registry.Menu):
    idname = "TEXTURE_SET_MT_resolution"
    label = "分辨率"

    def draw(self, layout, ctx) -> None:
        ts = _texture_set(ctx, 0)
        for value, text, _desc in RESOLUTION_ITEMS:
            mark = "  ✓" if ts is not None and ts.resolution == value else ""
            layout.operator("texture_set.resize", text="%s（%s × %s）%s" % (text, value, value, mark), resolution=value)
