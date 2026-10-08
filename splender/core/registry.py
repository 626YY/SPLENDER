"""注册表：编辑器、面板、工具、菜单。插件用的也是这里的接口。"""
from __future__ import annotations

from typing import Any

from .signals import Signal

_EDITORS: dict[str, type] = {}
_PANELS: list[type] = []
_TOOLS: dict[str, list[type]] = {}
_MENUS: dict[str, type] = {}

changed = Signal()


# ---- 编辑器 ----
def register_editor(cls: type) -> type:
    """注册一种编辑器。类需有 idname / label / icon / category。"""
    _EDITORS[cls.idname] = cls
    changed.emit("editors")
    return cls


def unregister_editor(idname: str) -> None:
    if _EDITORS.pop(idname, None) is not None:
        changed.emit("editors")


def editors() -> list[type]:
    return sorted(_EDITORS.values(), key=lambda c: (getattr(c, "order", 100), c.label))


def editor(idname: str) -> type | None:
    return _EDITORS.get(idname)


# ---- 面板 ----
class Panel:
    """面板：声明属于哪个编辑器、哪个区、哪个标签页；draw 里用布局语句排内容。"""

    idname = ""
    label = ""
    space = ""            # 编辑器 idname
    region = "SIDEBAR"    # SIDEBAR / MAIN
    category = ""         # 标签页名；属性编辑器里是标签的标识
    order = 100
    default_closed = False
    icon = ""

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return True

    def draw(self, layout: Any, ctx: Any) -> None:
        raise NotImplementedError

    def draw_header(self, layout: Any, ctx: Any) -> None:
        """标题行右侧的小控件，可不实现。"""


def register_panel(cls: type) -> type:
    if cls not in _PANELS:
        _PANELS.append(cls)
        changed.emit("panels")
    return cls


def unregister_panel(cls: type) -> None:
    if cls in _PANELS:
        _PANELS.remove(cls)
        changed.emit("panels")


def panels(space: str, region: str = "SIDEBAR", category: str | None = None) -> list[type]:
    found = [p for p in _PANELS if p.space == space and p.region == region and (category is None or p.category == category)]
    return sorted(found, key=lambda p: (p.order, p.label))


def panel_categories(space: str, region: str = "SIDEBAR") -> list[str]:
    seen: list[str] = []
    for panel in sorted((p for p in _PANELS if p.space == space and p.region == region), key=lambda p: p.order):
        if panel.category not in seen:
            seen.append(panel.category)
    return seen


# ---- 工具 ----
class Tool:
    """工具栏里的工具。keymap 指定它生效时额外启用的键位表名。"""

    idname = ""
    label = ""
    icon = ""
    space = "VIEW_3D"
    description = ""
    keymap = ""
    cursor = "CROSS"
    order = 100

    @classmethod
    def poll(cls, ctx: Any) -> bool:
        return True

    @classmethod
    def draw_settings(cls, layout: Any, ctx: Any) -> None:
        """标题栏里的工具设置条。"""


def register_tool(cls: type) -> type:
    bucket = _TOOLS.setdefault(cls.space, [])
    if cls not in bucket:
        bucket.append(cls)
        bucket.sort(key=lambda t: t.order)
        changed.emit("tools")
    return cls


def unregister_tool(cls: type) -> None:
    bucket = _TOOLS.get(cls.space, [])
    if cls in bucket:
        bucket.remove(cls)
        changed.emit("tools")


def tools(space: str) -> list[type]:
    return list(_TOOLS.get(space, []))


def tool(space: str, idname: str) -> type | None:
    for item in _TOOLS.get(space, []):
        if item.idname == idname:
            return item
    return None


# ---- 菜单 ----
class Menu:
    """菜单：draw 里用 layout.operator / layout.separator / layout.menu 排内容。"""

    idname = ""
    label = ""

    def draw(self, layout: Any, ctx: Any) -> None:
        raise NotImplementedError


def register_menu(cls: type) -> type:
    _MENUS[cls.idname] = cls
    return cls


def unregister_menu(idname: str) -> None:
    _MENUS.pop(idname, None)


def menu(idname: str) -> type | None:
    return _MENUS.get(idname)


# ---- 往已有菜单里加内容（插件用，相当于 Blender 的 Menu.append）----
_MENU_EXTRA: dict[str, list] = {}


def menu_append(idname: str, draw_fn) -> None:
    """在菜单 idname 末尾追加内容。draw_fn(layout, ctx)。"""
    items = _MENU_EXTRA.setdefault(idname, [])
    if draw_fn not in items:
        items.append(draw_fn)


def menu_prepend(idname: str, draw_fn) -> None:
    items = _MENU_EXTRA.setdefault(idname, [])
    if draw_fn not in items:
        items.insert(0, draw_fn)
        draw_fn._splender_prepend = True


def menu_remove(idname: str, draw_fn) -> None:
    items = _MENU_EXTRA.get(idname, [])
    if draw_fn in items:
        items.remove(draw_fn)


def menu_extras(idname: str) -> list:
    return list(_MENU_EXTRA.get(idname, []))
