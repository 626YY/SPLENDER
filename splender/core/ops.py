"""操作系统：菜单、按钮、快捷键、搜索调用的都是同一个操作。

定义::

    @register
    class LayerAdd(Operator):
        idname = "layer.add_paint"
        label = "新建绘制图层"
        undo = True

        @classmethod
        def poll(cls, ctx):
            return ctx.texture_set is not None

        def execute(self, ctx):
            ...
            return FINISHED

调用::

    ops.call("layer.add_paint", ctx)
"""
from __future__ import annotations

import logging
from typing import Any

from .props import PropertyGroup
from .signals import Signal

log = logging.getLogger("splender.ops")

FINISHED = "FINISHED"
CANCELLED = "CANCELLED"
RUNNING_MODAL = "RUNNING_MODAL"
PASS_THROUGH = "PASS_THROUGH"

_REGISTRY: dict[str, type["Operator"]] = {}

#: 每次操作执行完发出 (idname, 结果)
executed = Signal()
#: 操作做完（模态的在结束时）发出 (操作实例, 上下文, 结果)。「调整上一步」「重复上一步」靠它
finished = Signal()


class Operator(PropertyGroup):
    """操作基类。属性直接声明在类上，调用时以关键字参数传入。"""

    idname = ""
    label = ""
    description = ""
    icon = ""
    undo = False          # 执行后是否计入撤销历史（由操作自己向 History 推入步骤）
    searchable = True     # 是否出现在操作搜索里

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return True

    def execute(self, ctx: Any) -> str:
        return FINISHED

    def invoke(self, ctx: Any, event: Any) -> str:
        return self.execute(ctx)

    def modal(self, ctx: Any, event: Any) -> str:
        return FINISHED

    def cancel(self, ctx: Any) -> None:
        """模态操作被外部打断时调用。"""

    def report(self, ctx: Any, text: str, level: str = "INFO") -> None:
        log.log(logging.WARNING if level in ("WARNING", "ERROR") else logging.INFO, "%s: %s", self.label or self.idname, text)
        wm = getattr(ctx, "wm", None)
        if wm is not None:
            wm.report(text, level)


def register(cls: type[Operator]) -> type[Operator]:
    if not cls.idname:
        raise ValueError("操作必须有 idname")
    _REGISTRY[cls.idname] = cls
    return cls


def unregister(idname: str) -> None:
    _REGISTRY.pop(idname, None)


def get(idname: str) -> type[Operator] | None:
    return _REGISTRY.get(idname)


def all_operators() -> list[type[Operator]]:
    return sorted(_REGISTRY.values(), key=lambda c: c.idname)


def poll(idname: str, ctx: Any) -> bool:
    cls = _REGISTRY.get(idname)
    if cls is None:
        return False
    try:
        return bool(cls.poll(ctx))
    except Exception:  # noqa: BLE001
        log.exception("操作 %s 的可用条件判断出错", idname)
        return False


def call(idname: str, ctx: Any, event: Any = None, *, invoke: bool = True, **props: Any) -> str:
    """执行一个操作。返回 FINISHED / CANCELLED / RUNNING_MODAL / PASS_THROUGH。"""
    cls = _REGISTRY.get(idname)
    if cls is None:
        log.warning("没有名为 %s 的操作", idname)
        return CANCELLED
    if not poll(idname, ctx):
        return CANCELLED
    op = cls()
    for name, value in props.items():
        if name in cls.properties():
            setattr(op, name, value)
    try:
        result = op.invoke(ctx, event) if invoke else op.execute(ctx)
    except Exception as error:  # noqa: BLE001
        log.exception("操作 %s 执行出错", idname)
        op.report(ctx, "没有完成：%s" % error, "ERROR")
        return CANCELLED
    result = result or FINISHED
    if isinstance(result, (set, frozenset)):
        result = next(iter(result)) if result else FINISHED
    if result == RUNNING_MODAL:
        wm = getattr(ctx, "wm", None)
        if wm is not None:
            wm.modal_push(op, ctx)
    executed.emit(idname, result)
    if result != RUNNING_MODAL:
        finished.emit(op, ctx, result)
    return result
