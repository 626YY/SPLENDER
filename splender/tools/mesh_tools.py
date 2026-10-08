"""3D 视口工具栏里编辑模式的工具（照 Blender）：调整、框选、刷选、套索、游标，挤出、内插、倒角、环切、切刀、
法向缩放、平滑。W 在几种选择工具之间轮换。拖动类工具在选中的部分上按住拖动才生效，在别处拖动是框选。"""
from __future__ import annotations

from ..core import registry


def _tool(idname: str, label: str, icon: str, keymap: str, description: str, order: int, group: str = ""):
    attrs = {"idname": idname, "label": label, "icon": icon, "space": "VIEW_3D", "keymap": keymap,
             "description": description, "mode": "EDIT", "order": order, "group": group}
    return registry.register_tool(type("MeshTool_" + idname.split(".")[-1], (registry.Tool,), attrs))


TweakTool = _tool("mesh.tweak", "调整", "tool.tweak", "3D View Tool: Tweak",
                  "点一下选中点、边、面，按住拖动就移动它", 1, "select")
SelectBoxTool = _tool("mesh.select_box", "框选", "tool.select_box", "3D View Tool: Select Box",
                      "点一下选中，拖动画框选择。Shift 加选，Ctrl 减选", 2, "select")
SelectCircleTool = _tool("mesh.select_circle", "刷选", "tool.select_circle", "3D View Tool: Select Circle",
                         "按住拖动，圆圈刷过的都选中。Shift 减选", 3, "select")
SelectLassoTool = _tool("mesh.select_lasso", "套索", "tool.select_lasso", "3D View Tool: Select Lasso",
                        "按住画一圈，圈里的都选中。Shift 加选，Ctrl 减选", 4, "select")
CursorTool = _tool("mesh.cursor", "游标", "tool.cursor", "3D View Tool: Cursor",
                   "点一下把 3D 游标放到表面上", 10)
MoveTool = _tool("mesh.move", "移动", "tool.move", "3D View Tool: Move",
                 "拖操纵杆的箭头沿轴移动、拖小方块在平面里移动、拖中间的圈随便移动；别处拖动是框选", 11)
RotateTool = _tool("mesh.rotate", "旋转", "tool.rotate", "3D View Tool: Rotate",
                   "拖操纵杆的彩色环绕轴旋转、拖外面的白环绕视线旋转；别处拖动是框选", 12)
ScaleTool = _tool("mesh.scale", "缩放", "tool.scale", "3D View Tool: Scale",
                  "拖操纵杆的轴沿轴缩放、拖小方块在平面里缩放、拖中间的圈整体缩放；别处拖动是框选", 13)
ExtrudeTool = _tool("mesh.tool_extrude", "挤出", "tool.extrude", "3D View Tool: Edit Mesh, Extrude Region",
                    "在选中的部分上按住拖动：挤出去（没选面时挤出边或点）", 20)
InsetTool = _tool("mesh.tool_inset", "内插面", "tool.inset", "3D View Tool: Edit Mesh, Inset Faces",
                  "在选中的面上按住拖动：往里缩出一圈，按住 Ctrl 调深度", 21)
BevelTool = _tool("mesh.tool_bevel", "倒角", "tool.bevel", "3D View Tool: Edit Mesh, Bevel",
                  "在选中的边上按住拖动：倒成斜面，滚轮加段数", 22)
LoopCutTool = _tool("mesh.tool_loopcut", "环切", "tool.loopcut", "3D View Tool: Edit Mesh, Loop Cut",
                    "指着一条边按下：沿它那一圈切一刀，拖动滑动位置，松开确认", 23)
KnifeTool = _tool("mesh.tool_knife", "切刀", "tool.knife", "3D View Tool: Edit Mesh, Knife",
                  "在面上点出一条线，沿线切开；回车确认", 24)
ShrinkFattenTool = _tool("mesh.tool_shrink_fatten", "法向缩放", "tool.shrink_fatten",
                         "3D View Tool: Edit Mesh, Shrink/Fatten", "在选中的部分上按住拖动：沿各自的法线往外推或往里收", 30)
SmoothTool = _tool("mesh.tool_smooth", "平滑", "tool.smooth", "3D View Tool: Edit Mesh, Smooth",
                   "在选中的部分上按住左右拖动：往周围的平均位置靠，越往右越平滑", 31)
