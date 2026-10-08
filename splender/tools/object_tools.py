"""3D 视口工具栏里物体模式的工具（照 Blender）：调整、框选、刷选、套索、游标。W 在几种选择工具之间轮换。"""
from __future__ import annotations

from ..core import registry


def _tool(idname: str, label: str, icon: str, keymap: str, description: str, order: int, group: str = ""):
    attrs = {"idname": idname, "label": label, "icon": icon, "space": "VIEW_3D", "keymap": keymap,
             "description": description, "mode": "OBJECT", "order": order, "group": group}
    return registry.register_tool(type("ObjectTool_" + idname.split(".")[-1], (registry.Tool,), attrs))


TweakTool = _tool("object.tweak", "调整", "tool.tweak", "3D View Tool: Tweak",
                  "点一下选中物体，按住物体拖动就移动它", 1, "select")
SelectBoxTool = _tool("object.select_box", "框选", "tool.select_box", "3D View Tool: Select Box",
                      "点一下选中物体，拖动画框选择。Shift 加选，Ctrl 减选", 2, "select")
SelectCircleTool = _tool("object.select_circle", "刷选", "tool.select_circle", "3D View Tool: Select Circle",
                         "按住拖动，圆圈刷过的物体都选中。Shift 减选", 3, "select")
SelectLassoTool = _tool("object.select_lasso", "套索", "tool.select_lasso", "3D View Tool: Select Lasso",
                        "按住画一圈，圈里的物体都选中。Shift 加选，Ctrl 减选", 4, "select")
CursorTool = _tool("object.cursor", "游标", "tool.cursor", "3D View Tool: Cursor",
                   "点一下把 3D 游标放到表面上，新加的物体放在这里", 10)
MoveTool = _tool("object.move", "移动", "tool.move", "3D View Tool: Move",
                 "拖操纵杆的箭头沿轴移动、拖小方块在平面里移动、拖中间的圈随便移动；别处拖动是框选", 11)
RotateTool = _tool("object.rotate", "旋转", "tool.rotate", "3D View Tool: Rotate",
                   "拖操纵杆的彩色环绕轴旋转、拖外面的白环绕视线旋转；别处拖动是框选", 12)
ScaleTool = _tool("object.scale", "缩放", "tool.scale", "3D View Tool: Scale",
                  "拖操纵杆的轴沿轴缩放、拖小方块在平面里缩放、拖中间的圈整体缩放；别处拖动是框选", 13)
