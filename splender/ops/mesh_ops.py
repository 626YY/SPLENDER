"""模型相关的操作：自动展开 UV。"""
from __future__ import annotations

import logging

import numpy as np

from ..core import ops
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import BoolProperty

log = logging.getLogger("splender.mesh")


def object_texture_size(app, obj) -> int:
    sizes = [ts.size for uid in obj.material_sets for ts in app.project.texture_sets if ts.uid == uid]
    return max(sizes) if sizes else 4096


def unwrap_object(app, obj, old: dict | None = None) -> dict:
    """按工具设置里的展开参数给物体生成新 UV，重建绘制几何，并把画在旧 UV 上的内容搬到新 UV。
    old：旧几何 {"positions", "uvs", "material_ids"}（逐角点）；不给时用物体现在的（有 UV 时）。
    返回统计，其中 "swaps" 是搬运时换下来的图层像素（调用方记撤销）。"""
    from ..geometry.unwrap import smart_unwrap
    from ..sculpt.topology import weld

    settings = app.tool_settings.unwrap
    data = obj.data
    if old is None and getattr(data, "has_uvs", True):
        old = {"positions": data.positions.copy(), "uvs": data.uvs.copy(), "material_ids": data.material_ids.copy()}
    mesh = weld(data.positions, None, data.material_ids, drop_degenerate=False)
    size = object_texture_size(app, obj)
    skirt = float(app.prefs.paint.dilate_texels) * size / 16384.0 if hasattr(app.prefs, "paint") else 8.0
    margin = max(float(settings.margin), 2.0 * skirt + 2.0)
    uvs, stats = smart_unwrap(mesh.vertices, mesh.triangles, mesh.material_ids, angle_limit=float(settings.angle_limit),
                              min_chart_faces=int(settings.min_chart), margin_texels=margin, resolution=size,
                              allow_rotate=bool(settings.rotate))
    data.uvs = np.ascontiguousarray(uvs, np.float32)
    data.has_uvs = True
    obj.geometry_dirty = True
    engine = app.engine
    stats["swaps"] = []
    if engine is not None:
        app.host.make_current()
        engine.rebuild_object(obj)
        if old is not None and bool(getattr(settings, "transfer", True)):
            try:
                stats["swaps"] = engine.transfer_layers(obj, old)
            except Exception:  # noqa: BLE001
                log.exception("搬运贴图出错")
                app.report("原来画的内容没能搬到新 UV 上，可以撤销恢复", "WARNING")
    stats["texture_size"] = size
    stats["margin"] = margin
    log.info("展开 UV：%s，%d 块，覆盖 %.0f%%，%.2f 秒", obj.name, stats["charts"], stats["coverage"] * 100.0,
             stats["unwrap_ms"] / 1000.0)
    return stats


@ops.register
class MeshUvUnwrap(Operator):
    idname = "mesh.uv_unwrap"
    label = "自动展开 UV"
    description = "按朝向把模型分成几块，平铺到贴图上。已经画好的内容会跟着搬到新 UV 上"
    icon = "uv_unwrap"
    only_missing = BoolProperty("只展开没有 UV 的模型", default=True)

    @classmethod
    def poll(cls, ctx) -> bool:
        app = ctx.app
        return (app.project is not None and bool(app.project.objects) and app.tool_settings.mode == "PAINT"
                and getattr(app, "host", None) is not None)

    def execute(self, ctx) -> str:
        from ..sculpt.session import apply_snapshot, snapshot_mesh
        from ..ui.busy import busy_cursor

        app = ctx.app
        targets = [o for o in app.project.objects
                   if o.visible and (not self.only_missing or not getattr(o.data, "has_uvs", True))]
        if not targets:
            self.report(ctx, "模型都已经有 UV" if self.only_missing else "没有可以展开的模型")
            return CANCELLED
        changes = []
        moved = 0
        with busy_cursor():
            for obj in targets:
                before = snapshot_mesh(obj.data)
                stats = unwrap_object(app, obj)
                moved += sum(len(pairs) for _ts, pairs in stats["swaps"])
                changes.append((obj, before, snapshot_mesh(obj.data), stats))
        state = {"applied": True}

        def restore(which: int) -> None:
            if app.engine is None:
                return
            app.host.make_current()
            for obj, before, after, stats in changes:
                apply_snapshot(obj.data, before if which == 0 else after)
                obj.geometry_dirty = True
                app.engine.rebuild_object(obj)
                if stats["swaps"]:
                    app.engine.swap_layer_stores(stats["swaps"], use_new=which == 1)
            state["applied"] = which == 1
            app.request_frame()

        def dispose() -> None:
            if app.engine is None:
                return
            for _obj, _before, _after, stats in changes:
                if stats["swaps"]:
                    app.engine.free_layer_stores(stats["swaps"], keep_new=state["applied"])

        nbytes = sum(before["uvs"].nbytes * 2 for _o, before, _a, _s in changes)
        app.history.push("展开 UV", lambda: restore(0), lambda: restore(1), nbytes, dispose)
        app.project.mark_dirty()
        charts = sum(s["charts"] for *_x, s in changes)
        text = "已展开 %d 个模型的 UV：%d 块" % (len(changes), charts)
        if moved:
            text += "，原来画的 %d 个图层已跟着搬到新 UV 上" % moved
        self.report(ctx, text)
        app.request_frame()
        return FINISHED
