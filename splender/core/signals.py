"""轻量信号：不依赖 Qt，引擎和文档层都能用。"""
from __future__ import annotations

import weakref
from typing import Any, Callable


class Signal:
    """可连接多个回调的信号。绑定方法以弱引用保存，对象销毁后自动断开。"""

    __slots__ = ("_slots", "_blocked")

    def __init__(self) -> None:
        self._slots: list[Any] = []
        self._blocked = 0

    def connect(self, slot: Callable[..., Any]) -> Callable[..., Any]:
        ref: Any
        if hasattr(slot, "__self__") and hasattr(slot, "__func__"):
            ref = weakref.WeakMethod(slot)
        else:
            ref = slot
        self._slots.append(ref)
        return slot

    def disconnect(self, slot: Callable[..., Any]) -> None:
        keep = []
        for ref in self._slots:
            target = ref() if isinstance(ref, weakref.WeakMethod) else ref
            if target is None or target == slot:
                continue
            keep.append(ref)
        self._slots = keep

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if self._blocked:
            return
        dead = False
        for ref in list(self._slots):
            if isinstance(ref, weakref.WeakMethod):
                target = ref()
                if target is None:
                    dead = True
                    continue
            else:
                target = ref
            target(*args, **kwargs)
        if dead:
            self._slots = [r for r in self._slots if not (isinstance(r, weakref.WeakMethod) and r() is None)]

    def block(self) -> "_Blocker":
        return _Blocker(self)


class _Blocker:
    def __init__(self, signal: Signal) -> None:
        self.signal = signal

    def __enter__(self) -> None:
        self.signal._blocked += 1

    def __exit__(self, *exc: Any) -> None:
        self.signal._blocked -= 1
