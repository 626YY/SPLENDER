"""绘制工具：三维视口（绘制模式）和 UV 视图的工具栏各一份。"""
from __future__ import annotations

from ..core import registry

#: 效果笔刷（照 Photoshop）：工具 → 效果（engine/filters.py 的 EFFECTS）
EFFECT_TOOLS = {"paint.dodge": "DODGE", "paint.burn": "BURN", "paint.sponge": "SPONGE", "paint.blur": "BLUR",
                "paint.sharpen": "SHARPEN", "paint.smudge": "SMUDGE", "paint.clone": "CLONE", "paint.heal": "HEAL"}
EFFECT_LABELS = {"DODGE": "减淡", "BURN": "加深", "SPONGE": "海绵", "BLUR": "模糊", "SHARPEN": "锐化",
                 "SMUDGE": "涂抹", "CLONE": "仿制图章", "HEAL": "修复画笔"}
#: 按住 Ctrl 时换成的效果（和 Photoshop 按住 Alt 一样：减淡、加深互换，模糊、锐化互换）
EFFECT_INVERSE = {"DODGE": "BURN", "BURN": "DODGE", "BLUR": "SHARPEN", "SHARPEN": "BLUR"}
#: 用笔刷画的工具（属性编辑器里的笔刷面板只在这些工具下显示）
STROKE_TOOLS = ("paint.brush", "paint.eraser") + tuple(EFFECT_TOOLS)
#: 选区工具（照 Photoshop）
SELECT_TOOLS = ("paint.select_rect", "paint.select_ellipse", "paint.select_lasso", "paint.select_poly",
                "paint.select_quick", "paint.select_wand")

_TOOLS = [
    # 名字、标题、图标、键位表、说明、顺序
    ("paint.select_rect", "矩形选框", "tool.select_rect", "Paint Tool: Marquee Rect",
     "拖出矩形选区。拖之前按住 Shift 添加、Alt 减去、Shift+Alt 交叉；拖的时候按 Shift 是正方形、Alt 从中心拉；"
     "点一下取消选择。有选区时画笔、滤镜、填充只改选区里面", 1),
    ("paint.select_ellipse", "椭圆选框", "tool.select_ellipse", "Paint Tool: Marquee Ellipse",
     "拖出椭圆选区。拖之前按住 Shift 添加、Alt 减去；拖的时候按 Shift 是正圆、Alt 从中心拉", 2),
    ("paint.select_lasso", "套索", "tool.select_lasso", "Paint Tool: Lasso",
     "按住拖，圈出一块随手画的选区。拖之前按住 Shift 添加、Alt 减去", 3),
    ("paint.select_poly", "多边形套索", "tool.select_poly", "Paint Tool: Polygon Lasso",
     "一下一下点出多边形的顶点：点回第一个点、双击或按回车闭合；退格去掉上一个点，Esc 放弃", 4),
    ("paint.select_quick", "快速选择", "tool.select_quick", "Paint Tool: Quick Select",
     "像画画一样拖：碰到的地方连同颜色相近、连成一片的一起选上。按住 Alt 减去", 5),
    ("paint.select_wand", "魔棒", "tool.select_wand", "Paint Tool: Wand",
     "点一下，选上颜色相近的一片。拖之前按住 Shift 添加、Alt 减去", 6),
    ("paint.brush", "画笔", "tool.brush", "Paint", "在模型表面绘制。按住 Ctrl 临时变成擦除", 10),
    ("paint.eraser", "橡皮", "tool.eraser", "Paint", "擦掉当前图层上的内容。画蒙版时把蒙版擦成隐藏", 20),
    ("paint.gradient", "渐变", "tool.gradient", "Paint Tool: Gradient",
     "拖一条线，整层按这条线铺上渐变（笔刷颜色到备用颜色，或自定色带）。按住 Shift 角度按 45° 吸附", 30),
    ("paint.fill", "油漆桶", "tool.fill", "Paint Tool: Fill",
     "点一下，把一片地方填上笔刷的颜色和材质：相近颜色、UV 岛、网格部件或整层", 40),
    ("paint.dodge", "减淡", "tool.dodge", "Paint", "划过的地方变亮。按住 Ctrl 临时换成加深", 50),
    ("paint.burn", "加深", "tool.burn", "Paint", "划过的地方变暗。按住 Ctrl 临时换成减淡", 51),
    ("paint.sponge", "海绵", "tool.sponge", "Paint", "划过的地方颜色变灰（或变艳）。按住 Ctrl 反过来", 52),
    ("paint.blur", "模糊", "tool.blur", "Paint", "划过的地方变糊。按住 Ctrl 临时换成锐化", 60),
    ("paint.sharpen", "锐化", "tool.sharpen", "Paint", "划过的地方细节更清楚。按住 Ctrl 临时换成模糊", 61),
    ("paint.smudge", "涂抹", "tool.smudge", "Paint", "像用手指抹没干的颜料，把颜色往拖的方向带", 62),
    ("paint.clone", "仿制图章", "tool.clone", "Paint Tool: Clone",
     "按住 Alt 点一下定下来源，再划：把来源那里的样子照搬过来", 70),
    ("paint.heal", "修复画笔", "tool.heal", "Paint Tool: Clone",
     "按住 Alt 点一下定下来源，再划：照搬来源的纹理，颜色明暗跟着周围走，用来去瑕疵", 71),
    ("paint.text", "文字", "tool.text", "Paint Tool: Text", "点一下，在那里写字。做完在「调整上一步」里还能改字、字号、旋转", 80),
    ("paint.shape", "形状", "tool.shape", "Paint Tool: Shape",
     "拖出矩形、椭圆、多边形或直线。按住 Shift 等比，按住 Alt 从中心往外拉", 81),
]


def _register(space: str, idname: str, label: str, icon: str, keymap: str, description: str, order: int,
              mode) -> type:
    attrs = {"idname": idname, "label": label, "icon": icon, "space": space, "keymap": keymap,
             "description": description, "order": order}
    if mode is not None:
        attrs["mode"] = mode
    name = ("UV" if space == "IMAGE_EDITOR" else "") + "Tool_" + idname.split(".")[-1].title()
    return registry.register_tool(type(name, (registry.Tool,), attrs))


for _idname, _label, _icon, _keymap, _desc, _order in _TOOLS:
    _register("VIEW_3D", _idname, _label, _icon, _keymap, _desc, _order, "PAINT")
    # UV 视图：画笔、橡皮的左键在「Image」键位表里，别的工具用自己的表
    _register("IMAGE_EDITOR", _idname, _label, _icon, "" if _keymap == "Paint" else _keymap, _desc, _order, None)
