"""撤销历史。像素步骤和数据步骤都以「可撤销、可重做的一步」进入同一条历史。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from .signals import Signal

log = logging.getLogger("splender.history")


@dataclass
class Step:
    label: str
    undo: Callable[[], None]
    redo: Callable[[], None]
    nbytes: int = 0
    dispose: Callable[[], None] | None = None   # 步骤被丢弃时释放它占的资源


class History:
    def __init__(self, max_steps: int = 64, max_bytes: int = 2 * 1024 ** 3) -> None:
        self.max_steps = max_steps
        self.max_bytes = max_bytes
        self.steps: list[Step] = []
        self.index = 0                      # 已执行步骤的数量；steps[index:] 是可重做的
        self.changed = Signal()
        self.barrier: Callable[[], None] | None = None   # 改动历史之前先调用（让进行中的后台工作收尾）

    # ---- 查询 ----
    @property
    def can_undo(self) -> bool:
        return self.index > 0

    @property
    def can_redo(self) -> bool:
        return self.index < len(self.steps)

    @property
    def undo_label(self) -> str:
        return self.steps[self.index - 1].label if self.can_undo else ""

    @property
    def redo_label(self) -> str:
        return self.steps[self.index].label if self.can_redo else ""

    @property
    def nbytes(self) -> int:
        return sum(step.nbytes for step in self.steps)

    # ---- 修改 ----
    def push(self, label: str, undo: Callable[[], None], redo: Callable[[], None], nbytes: int = 0,
             dispose: Callable[[], None] | None = None) -> None:
        """记录一步已经完成的操作。重做分支被丢弃。"""
        self._sync()
        for step in self.steps[self.index:]:
            self._dispose(step)
        del self.steps[self.index:]
        self.steps.append(Step(label, undo, redo, nbytes, dispose))
        self.index = len(self.steps)
        self._trim()
        self.changed.emit()

    def merge_last(self, count: int, label: str | None = None) -> bool:
        """把最近的 count 步合成一步（撤销时倒着撤，重做时顺着做）。比如「复制」和接着的「移动」，
        名字默认用第一步的。"""
        if count < 2 or self.index < count or self.index != len(self.steps):
            return False
        parts = self.steps[self.index - count:self.index]

        def undo() -> None:
            for step in reversed(parts):
                step.undo()

        def redo() -> None:
            for step in parts:
                step.redo()

        def dispose() -> None:
            for step in parts:
                self._dispose(step)

        merged = Step(label or parts[0].label, undo, redo, sum(step.nbytes for step in parts), dispose)
        merged.parts = parts
        tags = [getattr(step, "tag", None) for step in parts]
        if tags[0] is not None and all(tag is tags[0] for tag in tags):
            merged.tag = tags[0]
        del self.steps[self.index - count:self.index]
        self.steps.append(merged)
        self.index = len(self.steps)
        self.changed.emit()
        return True

    def split_last(self) -> int:
        """把最后一步（合成的）拆回原来的几步，返回几步；不是合成的返回 0。"""
        if not self.steps or self.index != len(self.steps):
            return 0
        parts = getattr(self.steps[-1], "parts", None)
        if not parts:
            return 0
        self.steps[-1:] = list(parts)
        self.index = len(self.steps)
        return len(parts)

    def _sync(self) -> None:
        if self.barrier is not None:
            self.barrier()

    def undo(self) -> bool:
        self._sync()
        if not self.can_undo:
            return False
        step = self.steps[self.index - 1]
        step.undo()
        self.index -= 1
        self.changed.emit()
        return True

    def redo(self) -> bool:
        self._sync()
        if not self.can_redo:
            return False
        step = self.steps[self.index]
        step.redo()
        self.index += 1
        self.changed.emit()
        return True

    def drop_oldest(self) -> bool:
        """丢掉最早的一步（内存紧张时用）。至少保留最近一步。"""
        if len(self.steps) <= 1 or self.index <= 1:
            return False
        self._dispose(self.steps.pop(0))
        self.index -= 1
        self.changed.emit()
        return True

    def remove_tagged(self, tag) -> int:
        """拿掉带某个标签的步骤（例如退出雕刻模式时，雕刻模式里的笔划步骤失效）。返回拿掉几步。"""
        kept: list[Step] = []
        index = self.index
        removed = 0
        for position, step in enumerate(self.steps):
            if getattr(step, "tag", None) is tag:
                self._dispose(step)
                removed += 1
                if position < self.index:
                    index -= 1
            else:
                kept.append(step)
        if removed:
            self.steps = kept
            self.index = max(0, min(index, len(kept)))
            self.changed.emit()
        return removed

    def clear(self) -> None:
        for step in self.steps:
            self._dispose(step)
        self.steps.clear()
        self.index = 0
        self.changed.emit()

    def set_limits(self, max_steps: int, max_bytes: int) -> None:
        self.max_steps = max_steps
        self.max_bytes = max_bytes
        self._trim()

    def _trim(self) -> None:
        while self.steps and (len(self.steps) > self.max_steps or self.nbytes > self.max_bytes) and self.index > 1:
            self._dispose(self.steps.pop(0))
            self.index -= 1

    @staticmethod
    def _dispose(step: Step) -> None:
        if step.dispose is not None:
            try:
                step.dispose()
            except Exception:  # noqa: BLE001
                log.exception("释放撤销步骤资源时出错")
