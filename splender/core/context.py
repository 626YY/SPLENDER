"""上下文：操作和面板从这里知道「现在作用在哪」。

快捷键作用在鼠标所在的区域上，所以上下文由窗口管理器按鼠标位置现场组装。
"""
from __future__ import annotations

from typing import Any


class Context:
    """一次调用的上下文快照。字段可能为 None，操作的 poll 负责判断。"""

    __slots__ = ("app", "wm", "window", "screen", "area", "editor", "region", "event", "extra")

    def __init__(self, app: Any = None, wm: Any = None, window: Any = None, screen: Any = None, area: Any = None,
                 editor: Any = None, region: str = "WINDOW", event: Any = None, **extra: Any) -> None:
        self.app = app
        self.wm = wm
        self.window = window
        self.screen = screen
        self.area = area
        self.editor = editor
        self.region = region
        self.event = event
        self.extra = extra

    # ---- 派生数据：都从应用对象取，保持单一真值 ----
    @property
    def prefs(self) -> Any:
        return getattr(self.app, "prefs", None)

    @property
    def project(self) -> Any:
        return getattr(self.app, "project", None)

    @property
    def engine(self) -> Any:
        return getattr(self.app, "engine", None)

    @property
    def history(self) -> Any:
        return getattr(self.app, "history", None)

    @property
    def tool_settings(self) -> Any:
        return getattr(self.app, "tool_settings", None)

    @property
    def keyconfig(self) -> Any:
        return getattr(self.app, "keyconfig", None)

    @property
    def texture_set(self) -> Any:
        project = self.project
        return project.active_texture_set if project is not None else None

    @property
    def layer(self) -> Any:
        ts = self.texture_set
        return ts.active_layer if ts is not None else None

    @property
    def space_type(self) -> str:
        return getattr(self.editor, "idname", "") if self.editor is not None else ""

    def copy(self, **changes: Any) -> "Context":
        data = {name: getattr(self, name) for name in ("app", "wm", "window", "screen", "area", "editor", "region", "event")}
        extra = dict(self.extra)
        for key, value in changes.items():
            if key in data:
                data[key] = value
            else:
                extra[key] = value
        return Context(**data, **extra)

    def __getattr__(self, name: str) -> Any:
        extra = object.__getattribute__(self, "extra")
        if name in extra:
            return extra[name]
        raise AttributeError(name)
