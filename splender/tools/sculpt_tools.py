"""3D 视口工具栏里的雕刻工具（雕刻模式时显示）。"""
from __future__ import annotations

from ..core import registry
from ..sculpt.engine import BRUSHES

ICONS = {"DRAW": "circle-dot", "CLAY": "layers", "CLAY_STRIPS": "rows-2", "SMOOTH": "waves", "GRAB": "hand",
         "INFLATE": "circle-plus", "PINCH": "minimize-2", "FLATTEN": "minus", "CREASE": "pen-tool",
         "MASK": "square-dashed"}
SHORTCUTS = {"DRAW": "X", "CLAY": "C", "SMOOTH": "S", "GRAB": "G", "INFLATE": "I", "PINCH": "P", "FLATTEN": "T",
             "CREASE": "Shift+C", "MASK": "M"}


def _make_tool(index: int, name: str, label: str, description: str) -> type:
    attrs = {
        "idname": "sculpt." + name.lower(),
        "label": label,
        "icon": ICONS.get(name, "sculpt"),
        "space": "VIEW_3D",
        "description": description + "。按住 Ctrl 反向，按住 Shift 临时平滑",
        "keymap": "Sculpt",
        "mode": "SCULPT",
        "order": 100 + index,
    }
    return type("SculptTool_" + name, (registry.Tool,), attrs)


SCULPT_TOOLS = []
for _index, (_name, _label, _desc) in enumerate(BRUSHES):
    SCULPT_TOOLS.append(registry.register_tool(_make_tool(_index, _name, _label, _desc)))
