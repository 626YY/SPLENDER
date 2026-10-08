"""工作区：默认布局（绘制、材质、UV）和存进偏好设置的读写。

偏好里的 prefs.workspaces 是一段 JSON：

    {"version": 1, "active": 0, "items": [{"name": "绘制", "layout": {...}}, ...]}

layout 是 Screen.to_dict() 的格式（docs/ARCHITECTURE.md 3.6）。读的时候坏数据不会让程序崩：
整段读不出来就用默认工作区；某个工作区的布局坏了，默认名字的换回默认布局，其它的换成单个 3D 视口。
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable

log = logging.getLogger("splender.ui.workspaces")

FORMAT_VERSION = 1


def area(editor: str, **state: Any) -> dict:
    return {"type": "area", "editor": editor, "state": dict(state)}


def split(orientation: str, ratio: float, a: dict, b: dict) -> dict:
    return {"type": "split", "orientation": orientation, "ratio": ratio, "a": a, "b": b}


#: 默认工作区的布局。H 左右排列，V 上下排列，ratio 是左（上）块的比例。
DEFAULT_LAYOUTS: dict[str, dict] = {
    # 和 Blender 的「布局」一样：大视口（物体模式），右上大纲视图，右下属性，视口下面一条信息
    "布局": split("H", 0.78,
                split("V", 0.84, area("VIEW_3D"), area("INFO")),
                split("V", 0.38, area("OUTLINER"), area("PROPERTIES", tab="OBJECT"))),
    # 大视口在左；右边一列上面图层、下面属性；视口下面一条资源库
    "绘制": split("H", 0.74,
                split("V", 0.78, area("VIEW_3D"), area("ASSETS")),
                split("V", 0.46, area("LAYERS"), area("PROPERTIES"))),
    # 和 Blender 的「着色」一样：上面材质预览的视口，下面着色器编辑器；右边图层和属性
    "材质": split("H", 0.74,
                split("V", 0.55, area("VIEW_3D"), area("SHADER_EDITOR")),
                split("V", 0.42, area("LAYERS"), area("PROPERTIES"))),
    # 图像编辑器和视口左右并排，右边图层
    "UV": split("H", 0.80,
              split("H", 0.50, area("IMAGE_EDITOR"), area("VIEW_3D")),
              area("LAYERS")),
    # 大视口加属性：雕刻
    "雕刻": split("H", 0.80, area("VIEW_3D"), area("PROPERTIES")),
    # 上面视口看效果，下面节点编辑器；右边图层和属性
    "节点": split("H", 0.74,
                split("V", 0.42, area("VIEW_3D"), area("NODE_EDITOR")),
                split("V", 0.42, area("LAYERS"), area("PROPERTIES"))),
}
DEFAULT_ORDER: tuple[str, ...] = ("布局", "绘制", "雕刻", "材质", "节点", "UV")
#: 切到这些工作区时自动换成对应的模式（和 Blender 一样，工作区只是一套布局，模式在哪都能换）
WORKSPACE_MODES = {"布局": "OBJECT", "绘制": "PAINT", "材质": "PAINT", "节点": "PAINT", "UV": "PAINT",
                   "雕刻": "SCULPT"}
#: 第一版就有的默认工作区（用来判断哪些是后来新增的）
_ORIGINAL_DEFAULTS = ("绘制", "材质", "UV")
#: 默认布局改过的工作区：用户没动过（还是旧的默认样子）就换成新的
_OLD_DEFAULTS = {"材质": split("H", 0.72, split("V", 0.80, area("VIEW_3D"), area("INFO")), area("PROPERTIES"))}


def _shape(layout: dict):
    """布局的骨架（编辑器种类和拆分方向），不管比例和各区的状态。"""
    if not isinstance(layout, dict):
        return None
    if layout.get("type") == "area":
        return ("area", layout.get("editor"))
    return ("split", layout.get("orientation"), _shape(layout.get("a")), _shape(layout.get("b")))


@dataclass
class Workspace:
    """一个工作区：名字、布局，以及显示过之后对应的 Screen 控件。"""

    name: str
    layout: dict
    screen: Any = None

    def snapshot(self) -> dict:
        """当前布局：建过 Screen 的从 Screen 取，没建过的用保存的那份。"""
        if self.screen is not None:
            try:
                data = self.screen.to_dict()
                if is_valid_layout(data):
                    self.layout = data
                    return copy.deepcopy(data)
            except Exception:  # noqa: BLE001
                log.exception("读取工作区 %s 的布局出错", self.name)
        return copy.deepcopy(self.layout)


def default_layout(name: str) -> dict | None:
    layout = DEFAULT_LAYOUTS.get(name)
    return copy.deepcopy(layout) if layout is not None else None


def fallback_layout() -> dict:
    return area("VIEW_3D")


def default_workspaces() -> list[Workspace]:
    return [Workspace(name, copy.deepcopy(DEFAULT_LAYOUTS[name])) for name in DEFAULT_ORDER]


def unique_name(base: str, existing: Iterable[str]) -> str:
    """名字重了就在后面加数字。"""
    taken = set(existing)
    base = (base or "").strip() or "工作区"
    if base not in taken:
        return base
    number = 2
    while "%s %d" % (base, number) in taken:
        number += 1
    return "%s %d" % (base, number)


def is_valid_layout(data: Any, depth: int = 0) -> bool:
    """布局字典结构是否完整。"""
    if not isinstance(data, dict) or depth > 64:
        return False
    kind = data.get("type")
    if kind == "area":
        return isinstance(data.get("editor"), str) and isinstance(data.get("state", {}), dict)
    if kind == "split":
        ratio = data.get("ratio")
        return (data.get("orientation") in ("H", "V") and isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
                and 0.0 <= float(ratio) <= 1.0
                and is_valid_layout(data.get("a"), depth + 1) and is_valid_layout(data.get("b"), depth + 1))
    return False


def dumps(items: Iterable[Workspace], active: int) -> str:
    entries = [{"name": ws.name, "layout": ws.snapshot()} for ws in items]
    return json.dumps({"version": FORMAT_VERSION, "active": int(active), "items": entries,
                       "defaults_seen": list(DEFAULT_ORDER)}, ensure_ascii=False)


def loads(text: str) -> tuple[list[Workspace], int]:
    """解析 prefs.workspaces。读不出来时返回默认工作区。"""
    if not text or not str(text).strip():
        return default_workspaces(), 0
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        log.warning("工作区布局读取失败，使用默认工作区")
        return default_workspaces(), 0
    raw = data.get("items") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return default_workspaces(), 0
    items: list[Workspace] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip() or "工作区"
        name = unique_name(name, (ws.name for ws in items))
        layout = entry.get("layout")
        if not is_valid_layout(layout):
            log.warning("工作区 %s 的布局损坏，换成默认布局", name)
            layout = default_layout(name) or fallback_layout()
        elif name in _OLD_DEFAULTS and _shape(layout) == _shape(_OLD_DEFAULTS[name]):
            layout = default_layout(name)
        items.append(Workspace(name, copy.deepcopy(layout)))
    if not items:
        return default_workspaces(), 0
    active = data.get("active", 0)
    names = [ws.name for ws in items]
    if isinstance(active, str):
        active = names.index(active) if active in names else 0
    try:
        active = int(active)
    except (TypeError, ValueError):
        active = 0
    active_name = names[max(0, min(len(names) - 1, active))] if names else ""
    # 新版本新增的默认工作区：用户第一次见到时补进去（之后删掉就不再补）；原来停在哪个工作区还停在哪个
    seen = data.get("defaults_seen") or list(_ORIGINAL_DEFAULTS)
    for position, name in enumerate(DEFAULT_ORDER):
        if name not in seen and name not in names:
            items.insert(min(position, len(items)), Workspace(name, copy.deepcopy(DEFAULT_LAYOUTS[name])))
            names = [ws.name for ws in items]
    if active_name in names:
        active = names.index(active_name)
    return items, max(0, min(len(items) - 1, active))


def load(prefs: Any) -> tuple[list[Workspace], int]:
    """从偏好设置读出工作区列表和当前工作区序号。"""
    text = getattr(prefs, "workspaces", "") if prefs is not None else ""
    return loads(text if isinstance(text, str) else "")


def save(prefs: Any, items: Iterable[Workspace], active: int) -> str:
    """把工作区写进偏好设置（只写内存里的偏好对象，落盘由偏好设置统一负责）。"""
    text = dumps(items, active)
    if prefs is not None:
        try:
            prefs.workspaces = text
        except Exception:  # noqa: BLE001
            log.exception("工作区布局写入偏好设置失败")
    return text
