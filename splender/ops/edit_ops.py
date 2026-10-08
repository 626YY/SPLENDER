"""编辑操作：撤销、重做。"""
from __future__ import annotations

from ..core import ops
from ..core.ops import CANCELLED, FINISHED, Operator


@ops.register
class EditUndo(Operator):
    idname = "ed.undo"
    label = "撤销"
    icon = "undo"

    @classmethod
    def poll(cls, ctx) -> bool:
        if ctx.history is None:
            return False
        engine = ctx.engine
        return ctx.history.can_undo or (engine is not None and (engine.stroke is not None or bool(engine.merges)))

    def execute(self, ctx) -> str:
        engine = ctx.engine
        if engine is not None and engine.stroke is not None:
            ctx.app.host.make_current()
            engine.stroke_cancel()
            ctx.app.request_frame()
            return FINISHED
        if engine is not None and engine.merges:
            ctx.app.host.make_current()
            engine.complete_strokes()
        label = ctx.history.undo_label
        if not ctx.history.undo():
            return CANCELLED
        self.report(ctx, "撤销：%s" % label)
        ctx.app.request_frame()
        return FINISHED


@ops.register
class EditRedo(Operator):
    idname = "ed.redo"
    label = "重做"
    icon = "redo"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.history is not None and ctx.history.can_redo

    def execute(self, ctx) -> str:
        engine = ctx.engine
        if engine is not None and engine.merges:
            ctx.app.host.make_current()
            engine.complete_strokes()
        label = ctx.history.redo_label
        if not ctx.history.redo():
            return CANCELLED
        self.report(ctx, "重做：%s" % label)
        ctx.app.request_frame()
        return FINISHED
