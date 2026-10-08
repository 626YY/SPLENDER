"""键位表。

事件按上下文分层匹配：当前工具 → 鼠标所在编辑器 → 屏幕 → 窗口。
默认预设对齐 Blender 4.3.2（见 docs/planning-drafts/2026-10-04/notes-blender.md）。
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("splender.keymap")

# 事件值
PRESS = "PRESS"
RELEASE = "RELEASE"
MOVE = "MOVE"
WHEEL = "WHEEL"
DOUBLE = "DOUBLE"
CLICK = "CLICK"              # 按下后没怎么动就松开（和 Blender 一样，按下没有操作接的时候才会产生）
CLICK_DRAG = "CLICK_DRAG"    # 按下后拖动超过阈值；事件坐标是按下的位置

# 鼠标键名
LEFTMOUSE = "LEFTMOUSE"
MIDDLEMOUSE = "MIDDLEMOUSE"
RIGHTMOUSE = "RIGHTMOUSE"
MOUSEMOVE = "MOUSEMOVE"
WHEELUP = "WHEELUPMOUSE"
WHEELDOWN = "WHEELDOWNMOUSE"


@dataclass
class Event:
    """与界面库无关的输入事件。坐标是区域内的逻辑像素。"""

    type: str = ""
    value: str = PRESS
    ctrl: bool = False
    shift: bool = False
    alt: bool = False
    x: float = 0.0
    y: float = 0.0
    prev_x: float = 0.0
    prev_y: float = 0.0
    pressure: float = 1.0
    tilt_x: float = 0.0
    tilt_y: float = 0.0
    is_tablet: bool = False
    is_eraser: bool = False
    wheel: float = 0.0
    is_repeat: bool = False
    text: str = ""
    time: float = field(default_factory=time.perf_counter)
    source: Any = None

    @property
    def modifiers(self) -> tuple[bool, bool, bool]:
        return (self.ctrl, self.shift, self.alt)


@dataclass
class KeyMapItem:
    op: str
    type: str
    value: str = PRESS
    ctrl: bool = False
    shift: bool = False
    alt: bool = False
    any_mod: bool = False
    repeat: bool = False
    props: dict = field(default_factory=dict)
    active: bool = True
    default: tuple | None = field(default=None, compare=False, repr=False)   # 出厂时的 KEY_FIELDS（改键用）

    def matches(self, event: Event) -> bool:
        if not self.active or event.type != self.type or event.value != self.value:
            return False
        if event.is_repeat and not self.repeat:
            return False
        if self.any_mod:
            return True
        return (event.ctrl, event.shift, event.alt) == (self.ctrl, self.shift, self.alt)

    def shortcut_text(self) -> str:
        parts = []
        if self.ctrl:
            parts.append("Ctrl")
        if self.shift:
            parts.append("Shift")
        if self.alt:
            parts.append("Alt")
        parts.append(KEY_LABELS.get(self.type, self.type.title() if len(self.type) > 1 else self.type))
        text = "+".join(parts)
        if self.value == CLICK_DRAG:
            text += "拖动"
        elif self.value == DOUBLE:
            text += "双击"
        return text

    def to_dict(self) -> dict:
        return {"op": self.op, "type": self.type, "value": self.value, "ctrl": self.ctrl, "shift": self.shift,
                "alt": self.alt, "any_mod": self.any_mod, "repeat": self.repeat, "props": dict(self.props),
                "active": self.active}

    @classmethod
    def from_dict(cls, data: dict) -> "KeyMapItem":
        return cls(**{k: data[k] for k in ("op", "type", "value", "ctrl", "shift", "alt", "any_mod", "repeat",
                                           "props", "active") if k in data})


@dataclass
class KeyMap:
    """一张键位表。space 为空表示全局；否则是编辑器的 idname。"""

    name: str
    space: str = ""
    items: list[KeyMapItem] = field(default_factory=list)

    def add(self, op: str, type: str, value: str = PRESS, **kw: Any) -> KeyMapItem:  # noqa: A002
        props = kw.pop("props", {})
        item = KeyMapItem(op, type, value, props=props, **kw)
        self.items.append(item)
        return item

    def find(self, event: Event) -> KeyMapItem | None:
        for item in self.items:
            if item.matches(event):
                return item
        return None


class KeyConfig:
    """全部键位表。"""

    def __init__(self, name: str = "Blender") -> None:
        self.name = name
        self.keymaps: dict[str, KeyMap] = {}

    def keymap(self, name: str, space: str = "") -> KeyMap:
        km = self.keymaps.get(name)
        if km is None:
            km = KeyMap(name, space)
            self.keymaps[name] = km
        return km

    def lookup(self, event: Event, names: list[str]) -> KeyMapItem | None:
        """按给定顺序在多张键位表里找第一个匹配项。"""
        for name in names:
            km = self.keymaps.get(name)
            if km is None:
                continue
            item = km.find(event)
            if item is not None:
                return item
        return None

    def shortcut_for(self, op: str, props: dict | None = None) -> str:
        """给菜单显示用：某个操作当前绑定的快捷键文字。"""
        for km in self.keymaps.values():
            for item in km.items:
                if item.op == op and item.active and (props is None or item.props == props):
                    if item.type in (MOUSEMOVE,):
                        continue
                    return item.shortcut_text()
        return ""

    def to_dict(self) -> dict:
        return {name: {"space": km.space, "items": [i.to_dict() for i in km.items]} for name, km in self.keymaps.items()}

    def from_dict(self, data: dict) -> None:
        for name, entry in (data or {}).items():
            km = self.keymap(name, entry.get("space", ""))
            km.items = [KeyMapItem.from_dict(i) for i in entry.get("items", [])]


# ====================================================================== 改键
#: 一条键位里用户能改的部分
KEY_FIELDS = ("type", "value", "ctrl", "shift", "alt", "any_mod", "active")
#: 键位表的中文名（偏好设置里分组显示用）
KEYMAP_TITLES = {
    "Window": "全局", "Screen": "窗口布局", "3D View": "3D 视口", "Object Mode": "物体模式",
    "Object Non-modal": "物体模式（通用）", "Mesh": "编辑模式", "Paint": "绘制", "Sculpt": "雕刻",
    "Image": "UV / 图像", "Shader Editor": "材质节点", "Node": "节点纹理", "Layers": "图层", "Outliner": "大纲",
}
_TOOL_TITLES = {
    "Select Box": "框选", "Select Circle": "刷选", "Select Lasso": "套索选择", "Tweak": "点选拖动", "Cursor": "游标",
    "Edit Mesh, Extrude Region": "挤出", "Edit Mesh, Inset Faces": "内插面", "Edit Mesh, Bevel": "倒角",
    "Edit Mesh, Shrink/Fatten": "沿法线缩放", "Edit Mesh, Smooth": "平滑顶点", "Edit Mesh, Loop Cut": "环切",
    "Edit Mesh, Knife": "切刀", "Move": "移动", "Rotate": "旋转", "Scale": "缩放", "Gradient": "渐变",
    "Fill": "油漆桶", "Clone": "仿制图章", "Text": "文字", "Marquee Rect": "矩形选框", "Marquee Ellipse": "椭圆选框",
    "Lasso": "套索", "Polygon Lasso": "多边形套索", "Quick Select": "快速选择", "Wand": "魔棒", "Shape": "形状",
}
#: 全局的键位表：别的表里同样的键会先起作用
GLOBAL_KEYMAPS = ("Window", "Screen")


def keymap_title(name: str) -> str:
    if name in KEYMAP_TITLES:
        return KEYMAP_TITLES[name]
    for prefix, label in (("3D View Tool: ", "3D 视口工具："), ("Paint Tool: ", "绘制工具：")):
        if name.startswith(prefix):
            tool = name[len(prefix):]
            return label + _TOOL_TITLES.get(tool, tool)
    return name


def key_state(item: KeyMapItem) -> tuple:
    return tuple(getattr(item, name) for name in KEY_FIELDS)


def snapshot_defaults(kc: "KeyConfig") -> int:
    """给还没记出厂值的键位记下现在的样子（程序刚建好键位表、插件刚加了键位表时调用）。返回记了几条。"""
    count = 0
    for km in kc.keymaps.values():
        for item in km.items:
            if item.default is None:
                item.default = key_state(item)
                count += 1
    return count


def _identity(name: str, item: KeyMapItem, default: tuple | None = None) -> tuple:
    """认一条出厂键位：键位表、操作、参数、出厂的键。默认键位以后增删、换顺序也认得出来。"""
    props = json.dumps(item.props, sort_keys=True, ensure_ascii=False, default=str)
    return name, item.op, props, tuple(default if default is not None else item.default)


def is_modified(item: KeyMapItem) -> bool:
    return item.default is not None and key_state(item) != tuple(item.default)


def overrides_from(kc: "KeyConfig") -> dict:
    """和出厂键位不一样的那些，存进偏好设置（keymap_overrides）。"""
    changes = []
    for name, km in kc.keymaps.items():
        seen: Counter = Counter()
        for item in km.items:
            if item.default is None:
                continue
            ident = _identity(name, item)
            occurrence = seen[ident]
            seen[ident] += 1
            if is_modified(item):
                changes.append({"keymap": name, "op": item.op, "props": dict(item.props),
                                "default": dict(zip(KEY_FIELDS, item.default)), "occurrence": occurrence,
                                "set": dict(zip(KEY_FIELDS, key_state(item)))})
    return {"version": 1, "changes": changes}


def apply_overrides(kc: "KeyConfig", data: Any) -> tuple[int, int]:
    """把存下的改动用到键位表上（可以重复调用）。返回 (用上的, 找不到出厂键位的)。"""
    if isinstance(data, str):
        try:
            data = json.loads(data or "{}")
        except ValueError:
            log.warning("键位改动读不出来，按出厂键位")
            return 0, 0
    snapshot_defaults(kc)
    changes = (data or {}).get("changes", []) if isinstance(data, dict) else []
    index: dict[tuple, list[KeyMapItem]] = {}
    for name, km in kc.keymaps.items():
        for item in km.items:
            index.setdefault(_identity(name, item), []).append(item)
    applied = missing = 0
    for change in changes:
        try:
            default = tuple(change["default"][name] for name in KEY_FIELDS)
            probe = KeyMapItem(change["op"], default[0], props=dict(change.get("props") or {}))
            items = index.get(_identity(change["keymap"], probe, default), [])
            occurrence = int(change.get("occurrence", 0))
            if occurrence >= len(items):
                missing += 1
                continue
            item = items[occurrence]
            for name in KEY_FIELDS:
                if name in change["set"]:
                    setattr(item, name, change["set"][name])
            applied += 1
        except (KeyError, TypeError, ValueError):
            missing += 1
    if missing:
        log.info("有 %d 条键位改动对应的出厂键位已经不在了，没有用上", missing)
    return applied, missing


def reset_item(item: KeyMapItem) -> None:
    if item.default is not None:
        for name, value in zip(KEY_FIELDS, item.default):
            setattr(item, name, value)


def reset_all(kc: "KeyConfig") -> None:
    for km in kc.keymaps.values():
        for item in km.items:
            reset_item(item)


def _same_keys(a: KeyMapItem, b: KeyMapItem) -> bool:
    if a.type != b.type or a.value != b.value:
        return False
    return a.any_mod or b.any_mod or (a.ctrl, a.shift, a.alt) == (b.ctrl, b.shift, b.alt)


def conflicts(kc: "KeyConfig", name: str, item: KeyMapItem) -> list[tuple[str, KeyMapItem]]:
    """和这条键位用同一个键、会互相挡住的其他键位：同一张表里的；全局表和各编辑器表之间的。"""
    if not item.active or not item.type:
        return []
    found = []
    for other_name, km in kc.keymaps.items():
        if other_name != name and name not in GLOBAL_KEYMAPS and other_name not in GLOBAL_KEYMAPS:
            continue
        for other in km.items:
            if other is item or not other.active:
                continue
            if _same_keys(item, other) and not (other.op == item.op and other.props == item.props):
                found.append((other_name, other))
    return found


KEY_LABELS = {
    LEFTMOUSE: "左键", MIDDLEMOUSE: "中键", RIGHTMOUSE: "右键", WHEELUP: "滚轮上", WHEELDOWN: "滚轮下",
    "SPACE": "空格", "ESC": "Esc", "RET": "回车", "TAB": "Tab", "DEL": "Del", "BACK_SPACE": "退格",
    "HOME": "Home", "END": "End", "PAGE_UP": "PgUp", "PAGE_DOWN": "PgDn",
    "LEFT_BRACKET": "[", "RIGHT_BRACKET": "]", "PERIOD": ".", "COMMA": ",", "SLASH": "/", "MINUS": "-",
    "EQUAL": "=", "ACCENT_GRAVE": "`", "SEMI_COLON": ";", "QUOTE": "'", "BACK_SLASH": "\\",
    "LEFT_ARROW": "←", "RIGHT_ARROW": "→", "UP_ARROW": "↑", "DOWN_ARROW": "↓",
    "NUMPAD_0": "小键盘0", "NUMPAD_1": "小键盘1", "NUMPAD_2": "小键盘2", "NUMPAD_3": "小键盘3",
    "NUMPAD_4": "小键盘4", "NUMPAD_5": "小键盘5", "NUMPAD_6": "小键盘6", "NUMPAD_7": "小键盘7",
    "NUMPAD_8": "小键盘8", "NUMPAD_9": "小键盘9", "NUMPAD_PERIOD": "小键盘.", "NUMPAD_PLUS": "小键盘+",
    "NUMPAD_MINUS": "小键盘-", "NUMPAD_ENTER": "小键盘回车", "NUMPAD_SLASH": "小键盘/", "NUMPAD_ASTERIX": "小键盘*",
}


def _selection_keys(km) -> None:
    """选区（照 Photoshop）：M 选框、L 套索、W 快速选择和魔棒；全选、取消、重新选择、反选、羽化；
    有选区时 Delete、退格清掉选区里的，Ctrl+Shift+J 通过剪切的图层（Ctrl+J 通过拷贝在复制图层里）。"""
    km.add("paint.tool_cycle", "M", props={"tools": "paint.select_rect,paint.select_ellipse"})
    km.add("paint.tool_cycle", "L", props={"tools": "paint.select_lasso,paint.select_poly"})
    km.add("paint.tool_cycle", "W", props={"tools": "paint.select_quick,paint.select_wand"})
    km.add("select.all", "A", ctrl=True)
    km.add("select.none", "D", ctrl=True)
    km.add("select.reselect", "D", ctrl=True, shift=True)
    km.add("select.invert", "I", ctrl=True, shift=True)
    km.add("select.modify", "F6", shift=True, props={"kind": "FEATHER"})
    km.add("layer.clear_selection", "DEL")
    km.add("layer.clear_selection", "BACK_SPACE")
    km.add("layer.cut_selection", "J", ctrl=True, shift=True)


def default_keyconfig() -> KeyConfig:
    """默认键位，照 Blender 4.3 默认预设（左键选择）。操作名对应 splender/ops 里注册的操作。

    3D 视口按模式换键位表（和 Blender 一样，模式的表排在通用表前面，能盖掉通用的键）：
    物体模式 "Object Mode"、雕刻 "Sculpt"、纹理绘制 "Paint"，再往后是 "Object Non-modal"、"3D View"。
    """
    kc = KeyConfig("Blender")

    win = kc.keymap("Window")
    win.add("wm.new", "N", ctrl=True)
    win.add("wm.call_menu", "O", ctrl=True, shift=True, props={"name": "TOPBAR_MT_recent"})
    win.add("wm.open", "O", ctrl=True)
    win.add("wm.save", "S", ctrl=True)
    win.add("wm.save_as", "S", ctrl=True, shift=True)
    win.add("wm.save_incremental", "S", ctrl=True, alt=True)
    win.add("wm.quit", "Q", ctrl=True)
    win.add("wm.import_mesh", "I", ctrl=True)
    win.add("wm.export_textures", "E", ctrl=True, shift=True)
    win.add("ed.undo", "Z", ctrl=True, repeat=True)
    win.add("ed.redo", "Z", ctrl=True, shift=True, repeat=True)
    win.add("ed.redo", "Y", ctrl=True, repeat=True)
    win.add("wm.rename_active", "F2")
    win.add("wm.search", "F3")
    win.add("wm.call_menu", "F4", props={"name": "TOPBAR_MT_file_context"})
    win.add("wm.toolbar", "SPACE", shift=True)
    win.add("wm.preferences", "COMMA", ctrl=True)
    win.add("render.render", "F12")
    win.add("render.view_result", "F11")
    for key, space in (("F3", "NODE_EDITOR"), ("F5", "VIEW_3D"), ("F7", "PROPERTIES"), ("F9", "OUTLINER"),
                       ("F10", "IMAGE_EDITOR")):
        win.add("screen.space_type_set_or_cycle", key, shift=True, props={"space_type": space})

    screen = kc.keymap("Screen")
    screen.add("screen.area_maximize", "SPACE", ctrl=True)
    screen.add("screen.workspace_cycle", "PAGE_DOWN", ctrl=True, props={"direction": 1})
    screen.add("screen.workspace_cycle", "PAGE_UP", ctrl=True, props={"direction": -1})
    screen.add("screen.repeat_last", "R", shift=True)
    screen.add("screen.redo_last", "F9")
    screen.add("screen.region_toggle", "T", props={"region": "TOOLBAR"})
    screen.add("screen.region_toggle", "N", props={"region": "SIDEBAR"})

    # ---------------------------------------------------------------- 3D 视口（所有模式通用）
    v3d = kc.keymap("3D View", "VIEW_3D")
    v3d.add("view3d.cursor3d", RIGHTMOUSE, shift=True)
    v3d.add("view3d.localview", "NUMPAD_SLASH")
    v3d.add("view3d.localview", "SLASH")
    v3d.add("view3d.orbit", MIDDLEMOUSE)
    v3d.add("view3d.pan", MIDDLEMOUSE, shift=True)
    v3d.add("view3d.zoom_drag", MIDDLEMOUSE, ctrl=True)
    v3d.add("view3d.zoom_drag", MIDDLEMOUSE, ctrl=True, shift=True)
    v3d.add("view3d.view_selected", "NUMPAD_PERIOD")
    v3d.add("view3d.view_selected", "NUMPAD_PERIOD", ctrl=True)
    v3d.add("view3d.zoom", "NUMPAD_PLUS", repeat=True, props={"delta": 1})
    v3d.add("view3d.zoom", "NUMPAD_MINUS", repeat=True, props={"delta": -1})
    v3d.add("view3d.zoom", "EQUAL", ctrl=True, repeat=True, props={"delta": 1})
    v3d.add("view3d.zoom", "MINUS", ctrl=True, repeat=True, props={"delta": -1})
    v3d.add("view3d.zoom", WHEELUP, value=WHEEL, props={"delta": 1})
    v3d.add("view3d.zoom", WHEELDOWN, value=WHEEL, props={"delta": -1})
    v3d.add("view3d.view_all", "HOME")
    v3d.add("view3d.view_all", "HOME", ctrl=True)
    v3d.add("view3d.view_all", "C", shift=True, props={"center": True})
    v3d.add("wm.call_menu_pie", "ACCENT_GRAVE", props={"name": "VIEW3D_MT_view_pie"})
    v3d.add("view3d.navigate", "ACCENT_GRAVE", shift=True)
    v3d.add("view3d.view_camera", "NUMPAD_0")
    v3d.add("view3d.view_axis", "NUMPAD_1", props={"axis": "FRONT"})
    v3d.add("view3d.view_axis", "NUMPAD_3", props={"axis": "RIGHT"})
    v3d.add("view3d.view_axis", "NUMPAD_7", props={"axis": "TOP"})
    v3d.add("view3d.view_axis", "NUMPAD_1", ctrl=True, props={"axis": "BACK"})
    v3d.add("view3d.view_axis", "NUMPAD_3", ctrl=True, props={"axis": "LEFT"})
    v3d.add("view3d.view_axis", "NUMPAD_7", ctrl=True, props={"axis": "BOTTOM"})
    v3d.add("view3d.view_axis", "NUMPAD_1", shift=True, props={"axis": "FRONT", "align_active": True})
    v3d.add("view3d.view_axis", "NUMPAD_3", shift=True, props={"axis": "RIGHT", "align_active": True})
    v3d.add("view3d.view_axis", "NUMPAD_7", shift=True, props={"axis": "TOP", "align_active": True})
    v3d.add("view3d.view_axis", "NUMPAD_1", ctrl=True, shift=True, props={"axis": "BACK", "align_active": True})
    v3d.add("view3d.view_axis", "NUMPAD_3", ctrl=True, shift=True, props={"axis": "LEFT", "align_active": True})
    v3d.add("view3d.view_axis", "NUMPAD_7", ctrl=True, shift=True, props={"axis": "BOTTOM", "align_active": True})
    v3d.add("view3d.view_persportho", "NUMPAD_5")
    v3d.add("view3d.orbit_step", "NUMPAD_4", repeat=True, props={"yaw": -15.0})
    v3d.add("view3d.orbit_step", "NUMPAD_6", repeat=True, props={"yaw": 15.0})
    v3d.add("view3d.orbit_step", "NUMPAD_8", repeat=True, props={"pitch": -15.0})
    v3d.add("view3d.orbit_step", "NUMPAD_2", repeat=True, props={"pitch": 15.0})
    v3d.add("view3d.orbit_step", "NUMPAD_9", props={"yaw": 180.0})
    v3d.add("view3d.view_pan", "NUMPAD_4", ctrl=True, repeat=True, props={"direction": "LEFT"})
    v3d.add("view3d.view_pan", "NUMPAD_6", ctrl=True, repeat=True, props={"direction": "RIGHT"})
    v3d.add("view3d.view_pan", "NUMPAD_8", ctrl=True, repeat=True, props={"direction": "UP"})
    v3d.add("view3d.view_pan", "NUMPAD_2", ctrl=True, repeat=True, props={"direction": "DOWN"})
    v3d.add("view3d.view_roll", "NUMPAD_4", shift=True, repeat=True, props={"angle": -15.0})
    v3d.add("view3d.view_roll", "NUMPAD_6", shift=True, repeat=True, props={"angle": 15.0})
    v3d.add("view3d.view_axis_drag", MIDDLEMOUSE, value=CLICK_DRAG, alt=True)
    v3d.add("view3d.view_center_pick", MIDDLEMOUSE, value=CLICK, alt=True)
    v3d.add("view3d.select", LEFTMOUSE, value=CLICK, props={"deselect_all": True})
    v3d.add("view3d.select", LEFTMOUSE, value=CLICK, shift=True, props={"toggle": True})
    v3d.add("view3d.select", LEFTMOUSE, value=CLICK, ctrl=True, props={"deselect_all": True})
    v3d.add("view3d.select", LEFTMOUSE, value=CLICK, alt=True, props={"enumerate": True})
    v3d.add("view3d.select", LEFTMOUSE, value=CLICK, shift=True, alt=True, props={"toggle": True, "enumerate": True})
    v3d.add("view3d.select_box", "B", props={"wait_for_input": True})
    v3d.add("view3d.select_lasso", RIGHTMOUSE, value=CLICK_DRAG, ctrl=True, props={"mode": "ADD"})
    v3d.add("view3d.select_lasso", RIGHTMOUSE, value=CLICK_DRAG, ctrl=True, shift=True, props={"mode": "SUB"})
    v3d.add("view3d.select_circle", "C")
    v3d.add("view3d.zoom_border", "B", shift=True)
    v3d.add("view3d.camera_to_view", "NUMPAD_0", ctrl=True, alt=True)
    v3d.add("view3d.object_as_camera", "NUMPAD_0", ctrl=True)
    v3d.add("wm.context_toggle", "TAB", shift=True, props={"data_path": "tool_settings.use_snap"})
    v3d.add("wm.call_menu_pie", "S", shift=True, props={"name": "VIEW3D_MT_snap_pie"})
    v3d.add("wm.call_menu_pie", "PERIOD", props={"name": "VIEW3D_MT_pivot_pie"})
    v3d.add("wm.call_menu_pie", "COMMA", props={"name": "VIEW3D_MT_orientations_pie"})
    v3d.add("wm.call_menu_pie", "Z", props={"name": "VIEW3D_MT_shading_pie"})
    v3d.add("view3d.toggle_shading", "Z", shift=True, props={"type": "WIREFRAME"})
    v3d.add("view3d.toggle_xray", "Z", alt=True)
    v3d.add("wm.context_toggle", "Z", shift=True, alt=True, props={"data_path": "space_data.overlay.show_overlays"})
    v3d.add("wm.context_toggle", "ACCENT_GRAVE", ctrl=True, props={"data_path": "space_data.overlay.show_gizmo"})
    v3d.add("wm.tool_set_cycle", "W", props={"group": "select"})

    # ---------------------------------------------------------------- 物体模式
    om = kc.keymap("Object Mode", "VIEW_3D")
    om.add("object.select_all", "A", props={"action": "SELECT"})
    om.add("object.select_all", "A", alt=True, props={"action": "DESELECT"})
    om.add("object.select_all", "I", ctrl=True, props={"action": "INVERT"})
    om.add("object.select_linked", "L", shift=True)
    om.add("wm.call_menu", "G", shift=True, props={"name": "VIEW3D_MT_select_grouped"})
    om.add("transform.translate", "G")
    om.add("transform.rotate", "R")
    om.add("transform.resize", "S")
    om.add("transform.mirror", "M", ctrl=True)
    om.add("object.location_clear", "G", alt=True)
    om.add("object.rotation_clear", "R", alt=True)
    om.add("object.scale_clear", "S", alt=True)
    om.add("object.delete", "X")
    om.add("object.delete", "X", shift=True)
    om.add("object.delete", "DEL", props={"confirm": False})
    om.add("object.delete", "DEL", shift=True, props={"confirm": False})
    om.add("wm.call_menu", "A", shift=True, props={"name": "VIEW3D_MT_add"})
    om.add("wm.call_menu", "A", ctrl=True, props={"name": "VIEW3D_MT_object_apply"})
    om.add("wm.call_menu", "L", ctrl=True, props={"name": "VIEW3D_MT_make_links"})
    om.add("object.duplicate_move", "D", shift=True)
    om.add("object.duplicate_move", "D", alt=True, props={"linked": True})
    om.add("object.join", "J", ctrl=True)
    om.add("object.hide_view_clear", "H", alt=True)
    om.add("object.hide_view_set", "H")
    om.add("object.hide_view_set", "H", shift=True, props={"unselected": True})
    om.add("wm.call_menu", RIGHTMOUSE, props={"name": "VIEW3D_MT_object_context_menu"})
    om.add("wm.call_menu", "APP", props={"name": "VIEW3D_MT_object_context_menu"})

    nonmodal = kc.keymap("Object Non-modal", "VIEW_3D")
    nonmodal.add("object.mode_set", "TAB", props={"mode": "EDIT", "toggle": True})
    nonmodal.add("view3d.object_mode_pie_or_toggle", "TAB", ctrl=True)

    # ---------------------------------------------------------------- 编辑模式（网格）
    me = kc.keymap("Mesh", "VIEW_3D")
    me.add("transform.translate", "G")
    me.add("transform.rotate", "R")
    me.add("transform.resize", "S")
    me.add("transform.translate", LEFTMOUSE, value=CLICK_DRAG, props={"release_confirm": True})
    me.add("transform.mirror", "M", ctrl=True)
    me.add("transform.tosphere", "S", shift=True, alt=True)
    me.add("transform.shear", "S", ctrl=True, shift=True, alt=True)
    me.add("mesh.loopcut_slide", "R", ctrl=True)
    me.add("mesh.inset", "I")
    me.add("mesh.bevel", "B", ctrl=True, props={"affect": "EDGES"})
    me.add("mesh.shrink_fatten", "S", alt=True)
    me.add("mesh.bevel", "B", ctrl=True, shift=True, props={"affect": "VERTICES"})
    for key, kind in (("1", "VERT"), ("2", "EDGE"), ("3", "FACE")):
        me.add("mesh.select_mode", key, props={"type": kind})
        me.add("mesh.select_mode", key, shift=True, props={"type": kind, "use_extend": True})
        me.add("mesh.select_mode", key, ctrl=True, props={"type": kind, "use_expand": True})
        me.add("mesh.select_mode", key, ctrl=True, shift=True,
               props={"type": kind, "use_extend": True, "use_expand": True})
    me.add("mesh.loop_select", LEFTMOUSE, value=CLICK, alt=True)
    me.add("mesh.loop_select", LEFTMOUSE, value=CLICK, shift=True, alt=True, props={"toggle": True})
    me.add("mesh.edgering_select", LEFTMOUSE, value=CLICK, ctrl=True, alt=True)
    me.add("mesh.edgering_select", LEFTMOUSE, value=CLICK, ctrl=True, shift=True, alt=True, props={"toggle": True})
    me.add("mesh.shortest_path_pick", LEFTMOUSE, value=CLICK, ctrl=True)
    me.add("mesh.shortest_path_pick", LEFTMOUSE, value=CLICK, ctrl=True, shift=True)
    me.add("mesh.select_all", "A", props={"action": "SELECT"})
    me.add("mesh.select_all", "A", alt=True, props={"action": "DESELECT"})
    me.add("mesh.select_all", "I", ctrl=True, props={"action": "INVERT"})
    me.add("mesh.select_more", "NUMPAD_PLUS", ctrl=True, repeat=True)
    me.add("mesh.select_less", "NUMPAD_MINUS", ctrl=True, repeat=True)
    me.add("mesh.select_linked", "L", ctrl=True)
    me.add("mesh.select_linked_pick", "L", props={"deselect": False})
    me.add("mesh.select_linked_pick", "L", shift=True, props={"deselect": True})
    me.add("mesh.select_mirror", "M", ctrl=True, shift=True)
    me.add("wm.call_menu", "G", shift=True, props={"name": "VIEW3D_MT_edit_mesh_select_similar"})
    me.add("mesh.reveal", "H", alt=True)
    me.add("mesh.hide", "H", props={"unselected": False})
    me.add("mesh.hide", "H", shift=True, props={"unselected": True})
    me.add("mesh.normals_make_consistent", "N", shift=True, props={"inside": False})
    me.add("mesh.normals_make_consistent", "N", ctrl=True, shift=True, props={"inside": True})
    me.add("view3d.edit_mesh_extrude_move_normal", "E")
    me.add("wm.call_menu", "E", alt=True, props={"name": "VIEW3D_MT_edit_mesh_extrude"})
    me.add("mesh.fill", "F", alt=True)
    me.add("mesh.quads_convert_to_tris", "T", ctrl=True)
    me.add("mesh.quads_convert_to_tris", "T", ctrl=True, shift=True)
    me.add("mesh.tris_convert_to_quads", "J", alt=True)
    me.add("mesh.rip_move", "V", props={"use_fill": False})
    me.add("mesh.rip_move", "V", alt=True, props={"use_fill": True})
    me.add("wm.call_menu", "M", props={"name": "VIEW3D_MT_edit_mesh_merge"})
    me.add("wm.call_menu", "M", alt=True, props={"name": "VIEW3D_MT_edit_mesh_split"})
    me.add("mesh.edge_face_add", "F")
    me.add("mesh.duplicate_move", "D", shift=True)
    me.add("wm.call_menu", "A", shift=True, props={"name": "VIEW3D_MT_mesh_add"})
    me.add("wm.call_menu", "P", props={"name": "VIEW3D_MT_edit_mesh_separate"})
    me.add("mesh.split", "Y")
    me.add("mesh.vert_connect_path", "J")
    me.add("transform.vert_slide", "V", shift=True)
    me.add("mesh.dupli_extrude_cursor", RIGHTMOUSE, value=CLICK, ctrl=True)
    me.add("mesh.dupli_extrude_cursor", RIGHTMOUSE, value=CLICK, ctrl=True, shift=True)
    me.add("wm.call_menu", "X", props={"name": "VIEW3D_MT_edit_mesh_delete"})
    me.add("wm.call_menu", "DEL", props={"name": "VIEW3D_MT_edit_mesh_delete"})
    me.add("mesh.dissolve_mode", "X", ctrl=True)
    me.add("mesh.dissolve_mode", "DEL", ctrl=True)
    me.add("mesh.knife_tool", "K", props={"use_occlude_geometry": True, "only_selected": False})
    me.add("mesh.knife_tool", "K", shift=True, props={"use_occlude_geometry": False, "only_selected": True})
    me.add("wm.call_menu", "F", ctrl=True, props={"name": "VIEW3D_MT_edit_mesh_faces"})
    me.add("wm.call_menu", "E", ctrl=True, props={"name": "VIEW3D_MT_edit_mesh_edges"})
    me.add("wm.call_menu", "V", ctrl=True, props={"name": "VIEW3D_MT_edit_mesh_vertices"})
    me.add("wm.call_menu", "U", props={"name": "VIEW3D_MT_uv_map"})
    me.add("wm.call_menu", "N", alt=True, props={"name": "VIEW3D_MT_edit_mesh_normals"})
    me.add("wm.call_menu_pie", "O", shift=True, props={"name": "VIEW3D_MT_proportional_editing_falloff_pie"})
    me.add("wm.context_toggle", "O", props={"data_path": "tool_settings.use_proportional_edit"})
    me.add("wm.context_toggle", "O", alt=True, props={"data_path": "tool_settings.use_proportional_connected"})
    me.add("wm.call_menu", RIGHTMOUSE, props={"name": "VIEW3D_MT_edit_mesh_context_menu"})
    me.add("wm.call_menu", "APP", props={"name": "VIEW3D_MT_edit_mesh_context_menu"})

    # ---------------------------------------------------------------- 物体模式的工具
    tool = kc.keymap("3D View Tool: Select Box", "VIEW_3D")
    tool.add("view3d.select_box", LEFTMOUSE, value=CLICK_DRAG, props={"mode": "SET"})
    tool.add("view3d.select_box", LEFTMOUSE, value=CLICK_DRAG, shift=True, props={"mode": "ADD"})
    tool.add("view3d.select_box", LEFTMOUSE, value=CLICK_DRAG, ctrl=True, props={"mode": "SUB"})
    tool = kc.keymap("3D View Tool: Select Circle", "VIEW_3D")
    tool.add("view3d.select_circle", LEFTMOUSE, props={"mode": "SET", "wait_for_input": False})
    tool.add("view3d.select_circle", LEFTMOUSE, shift=True, props={"mode": "ADD", "wait_for_input": False})
    tool.add("view3d.select_circle", LEFTMOUSE, ctrl=True, props={"mode": "SUB", "wait_for_input": False})
    tool = kc.keymap("3D View Tool: Select Lasso", "VIEW_3D")
    tool.add("view3d.select_lasso", LEFTMOUSE, value=CLICK_DRAG, props={"mode": "SET"})
    tool.add("view3d.select_lasso", LEFTMOUSE, value=CLICK_DRAG, shift=True, props={"mode": "ADD"})
    tool.add("view3d.select_lasso", LEFTMOUSE, value=CLICK_DRAG, ctrl=True, props={"mode": "SUB"})
    tool = kc.keymap("3D View Tool: Tweak", "VIEW_3D")
    tool.add("view3d.select_tweak_move", LEFTMOUSE, value=CLICK_DRAG)
    tool = kc.keymap("3D View Tool: Cursor", "VIEW_3D")
    tool.add("view3d.cursor3d", LEFTMOUSE)
    # 编辑模式的工具：在选中的部分上拖动用工具，别处拖动是框选
    for name, action in (("Extrude Region", "EXTRUDE"), ("Inset Faces", "INSET"), ("Bevel", "BEVEL"),
                         ("Shrink/Fatten", "SHRINK_FATTEN"), ("Smooth", "SMOOTH")):
        tool = kc.keymap("3D View Tool: Edit Mesh, " + name, "VIEW_3D")
        tool.add("mesh.tool_drag", LEFTMOUSE, value=CLICK_DRAG, props={"action": action})
        tool.add("mesh.tool_drag", LEFTMOUSE, value=CLICK_DRAG, shift=True, props={"action": action})
        tool.add("mesh.tool_drag", LEFTMOUSE, value=CLICK_DRAG, ctrl=True, props={"action": action})
    tool = kc.keymap("3D View Tool: Edit Mesh, Loop Cut", "VIEW_3D")
    tool.add("mesh.loopcut_slide", LEFTMOUSE, props={"release_confirm": True})
    tool = kc.keymap("3D View Tool: Edit Mesh, Knife", "VIEW_3D")
    tool.add("mesh.knife_tool", LEFTMOUSE, props={"use_occlude_geometry": True, "only_selected": False})
    for name in ("Move", "Rotate", "Scale"):
        tool = kc.keymap("3D View Tool: " + name, "VIEW_3D")
        tool.add("view3d.select_box", LEFTMOUSE, value=CLICK_DRAG, props={"mode": "SET"})
        tool.add("view3d.select_box", LEFTMOUSE, value=CLICK_DRAG, shift=True, props={"mode": "ADD"})
        tool.add("view3d.select_box", LEFTMOUSE, value=CLICK_DRAG, ctrl=True, props={"mode": "SUB"})

    # ---------------------------------------------------------------- 纹理绘制
    paint = kc.keymap("Paint", "VIEW_3D")
    paint.add("paint.stroke", LEFTMOUSE)
    paint.add("paint.stroke", LEFTMOUSE, ctrl=True, props={"invert": True})
    paint.add("paint.swap_colors", "X")
    paint.add("paint.sample_color", "X", shift=True)
    paint.add("paint.brush_scale", "LEFT_BRACKET", repeat=True, props={"factor": 0.9})
    paint.add("paint.brush_scale", "RIGHT_BRACKET", repeat=True, props={"factor": 1.0 / 0.9})
    paint.add("paint.brush_resize", "F", props={"target": "size"})
    paint.add("paint.brush_resize", "F", shift=True, props={"target": "flow"})
    paint.add("paint.brush_resize", "F", ctrl=True, props={"target": "hardness"})
    paint.add("paint.tool_set", "B", props={"tool": "paint.brush"})
    paint.add("paint.tool_set", "E", props={"tool": "paint.eraser"})
    paint.add("paint.tool_cycle", "G", props={"tools": "paint.gradient,paint.fill"})
    paint.add("paint.tool_cycle", "O", props={"tools": "paint.dodge,paint.burn,paint.sponge"})
    paint.add("paint.tool_cycle", "R", props={"tools": "paint.blur,paint.sharpen,paint.smudge"})
    paint.add("paint.tool_set", "S", props={"tool": "paint.clone"})
    paint.add("paint.tool_set", "J", props={"tool": "paint.heal"})
    paint.add("paint.tool_set", "U", props={"tool": "paint.shape"})
    _selection_keys(paint)
    paint.add("paint.context_menu", RIGHTMOUSE)
    paint.add("paint.context_menu", "APP")
    paint.add("paint.toggle_stabilizer", "S", shift=True)
    # Photoshop 的习惯（和 Blender 不冲突的键）
    paint.add("paint.default_colors", "D")
    paint.add("paint.sample_color", "I")
    paint.add("paint.sample_color", LEFTMOUSE, alt=True)
    for digit in range(10):
        value = (digit or 10) / 10.0
        paint.add("paint.brush_set_value", str(digit), props={"target": "opacity", "value": value})
        paint.add("paint.brush_set_value", str(digit), shift=True, props={"target": "flow", "value": value})
    paint.add("layer.add_paint", "N", ctrl=True, shift=True)
    paint.add("layer.duplicate", "J", ctrl=True)
    paint.add("layer.group", "G", ctrl=True)
    paint.add("layer.add_adjust", "L", ctrl=True, props={"adjust": "LEVELS"})
    paint.add("layer.add_adjust", "U", ctrl=True, props={"adjust": "HSV"})
    paint.add("layer.add_adjust", "I", ctrl=True, props={"adjust": "INVERT"})
    paint.add("layer.add_adjust", "M", ctrl=True, props={"adjust": "CURVES"})
    paint.add("layer.add_adjust", "B", ctrl=True, props={"adjust": "COLOR_BALANCE"})
    paint.add("layer.add_adjust", "B", ctrl=True, alt=True, shift=True, props={"adjust": "BLACK_WHITE"})
    paint.add("layer.adjust_auto", "L", ctrl=True, shift=True, props={"mode": "TONE"})
    paint.add("layer.adjust_auto", "L", ctrl=True, alt=True, shift=True, props={"mode": "CONTRAST"})
    paint.add("layer.adjust_auto", "B", ctrl=True, shift=True, props={"mode": "COLOR"})
    paint.add("layer.add_desaturate", "U", ctrl=True, shift=True)
    paint.add("layer.fill_color", "BACK_SPACE", alt=True, props={"which": "PRIMARY"})
    paint.add("layer.fill_color", "BACK_SPACE", ctrl=True, props={"which": "SECONDARY"})

    # 绘制模式里不是笔刷的工具（三维视口和 UV 视图共用）
    gradient = kc.keymap("Paint Tool: Gradient", "VIEW_3D")
    gradient.add("paint.gradient", LEFTMOUSE)
    fill = kc.keymap("Paint Tool: Fill", "VIEW_3D")
    fill.add("paint.fill", LEFTMOUSE)
    clone = kc.keymap("Paint Tool: Clone", "VIEW_3D")          # 仿制图章、修复画笔：Alt 点一下定来源
    clone.add("paint.clone_source", LEFTMOUSE, alt=True)
    text = kc.keymap("Paint Tool: Text", "VIEW_3D")
    text.add("paint.text", LEFTMOUSE)
    # 选区工具：拖之前按住的 Shift、Alt 由操作自己看（添加、减去、交叉）
    for name, op, props in (("Marquee Rect", "select.marquee", {"shape": "RECT"}),
                            ("Marquee Ellipse", "select.marquee", {"shape": "ELLIPSE"}),
                            ("Lasso", "select.lasso", {}), ("Polygon Lasso", "select.polygon", {}),
                            ("Quick Select", "select.quick", {}), ("Wand", "select.wand", {})):
        kc.keymap("Paint Tool: " + name, "VIEW_3D").add(op, LEFTMOUSE, any_mod=True, props=props)
    shape = kc.keymap("Paint Tool: Shape", "VIEW_3D")
    shape.add("paint.shape", LEFTMOUSE)
    shape.add("paint.shape", LEFTMOUSE, shift=True)
    shape.add("paint.shape", LEFTMOUSE, alt=True)
    shape.add("paint.shape", LEFTMOUSE, shift=True, alt=True)

    # ---------------------------------------------------------------- 雕刻
    sculpt = kc.keymap("Sculpt", "VIEW_3D")
    sculpt.add("sculpt.stroke", LEFTMOUSE)
    sculpt.add("sculpt.stroke", LEFTMOUSE, ctrl=True, props={"invert": True})
    sculpt.add("sculpt.stroke", LEFTMOUSE, shift=True, props={"smooth": True})
    sculpt.add("sculpt.brush_resize", "F", props={"target": "size"})
    sculpt.add("sculpt.brush_resize", "F", shift=True, props={"target": "strength"})
    sculpt.add("sculpt.brush_resize", "F", ctrl=True, props={"target": "hardness"})
    sculpt.add("paint.brush_scale", "LEFT_BRACKET", repeat=True, props={"factor": 0.9})
    sculpt.add("paint.brush_scale", "RIGHT_BRACKET", repeat=True, props={"factor": 1.0 / 0.9})
    for key, tool_name, mods in (("V", "sculpt.draw", {}), ("S", "sculpt.smooth", {}), ("P", "sculpt.pinch", {}),
                                 ("I", "sculpt.inflate", {}), ("G", "sculpt.grab", {}),
                                 ("T", "sculpt.flatten", {"shift": True}), ("C", "sculpt.clay_strips", {}),
                                 ("C", "sculpt.crease", {"shift": True}), ("M", "sculpt.mask", {})):
        sculpt.add("paint.tool_set", key, props={"tool": tool_name}, **mods)
    sculpt.add("sculpt.mask_clear", "M", alt=True)
    sculpt.add("sculpt.mask_invert", "I", ctrl=True)
    sculpt.add("paint.mask_box_gesture", "B")
    sculpt.add("paint.mask_lasso_gesture", RIGHTMOUSE, value=CLICK_DRAG, ctrl=True, props={"value": 0.0})
    sculpt.add("paint.mask_lasso_gesture", RIGHTMOUSE, value=CLICK_DRAG, ctrl=True, shift=True, props={"value": 1.0})
    sculpt.add("wm.call_menu_pie", "A", props={"name": "VIEW3D_MT_sculpt_mask_edit_pie"})
    sculpt.add("sculpt.voxel_remesh", "R", ctrl=True)
    sculpt.add("sculpt.subdivide", "D", ctrl=True)
    sculpt.add("paint.context_menu", RIGHTMOUSE)
    sculpt.add("paint.context_menu", "APP")

    # ---------------------------------------------------------------- 图像编辑器（UV 视图）
    img = kc.keymap("Image", "IMAGE_EDITOR")
    img.add("image.pan", MIDDLEMOUSE)
    img.add("image.zoom", WHEELUP, value=WHEEL, props={"delta": 1})
    img.add("image.zoom", WHEELDOWN, value=WHEEL, props={"delta": -1})
    img.add("image.zoom", "NUMPAD_PLUS", repeat=True, props={"delta": 1})
    img.add("image.zoom", "NUMPAD_MINUS", repeat=True, props={"delta": -1})
    img.add("image.zoom_drag", MIDDLEMOUSE, ctrl=True)
    img.add("image.view_all", "HOME")
    img.add("image.channel_cycle", "C")
    img.add("paint.stroke", LEFTMOUSE)
    img.add("paint.stroke", LEFTMOUSE, ctrl=True, props={"invert": True})
    img.add("paint.brush_scale", "LEFT_BRACKET", repeat=True, props={"factor": 0.9})
    img.add("paint.brush_scale", "RIGHT_BRACKET", repeat=True, props={"factor": 1.0 / 0.9})
    img.add("paint.swap_colors", "X")
    img.add("paint.tool_set", "B", props={"tool": "paint.brush"})
    img.add("paint.tool_set", "E", props={"tool": "paint.eraser"})
    img.add("paint.tool_cycle", "G", props={"tools": "paint.gradient,paint.fill"})
    img.add("paint.tool_cycle", "O", props={"tools": "paint.dodge,paint.burn,paint.sponge"})
    img.add("paint.tool_cycle", "R", props={"tools": "paint.blur,paint.sharpen,paint.smudge"})
    img.add("paint.tool_set", "S", props={"tool": "paint.clone"})
    img.add("paint.tool_set", "J", props={"tool": "paint.heal"})
    img.add("paint.tool_set", "U", props={"tool": "paint.shape"})
    _selection_keys(img)
    # 按比例看（照 Blender 的图像编辑器）、框选放大
    for key, ratio in (("NUMPAD_1", 1.0), ("NUMPAD_2", 0.5), ("NUMPAD_4", 0.25), ("NUMPAD_8", 0.125)):
        img.add("image.view_zoom_ratio", key, props={"ratio": ratio})
    for key, ratio in (("NUMPAD_2", 2.0), ("NUMPAD_4", 4.0), ("NUMPAD_8", 8.0)):
        img.add("image.view_zoom_ratio", key, ctrl=True, props={"ratio": ratio})
    img.add("image.zoom_border", "B", shift=True)
    # Photoshop 的习惯
    img.add("image.view_zoom_ratio", "1", ctrl=True, props={"ratio": 1.0})
    img.add("image.view_all", "0", ctrl=True)
    img.add("paint.default_colors", "D")
    img.add("paint.toggle_stabilizer", "S", shift=True)
    for digit in range(10):
        value = (digit or 10) / 10.0
        img.add("paint.brush_set_value", str(digit), props={"target": "opacity", "value": value})
        img.add("paint.brush_set_value", str(digit), shift=True, props={"target": "flow", "value": value})
    img.add("layer.add_paint", "N", ctrl=True, shift=True)
    img.add("layer.duplicate", "J", ctrl=True)
    img.add("layer.fill_color", "BACK_SPACE", alt=True, props={"which": "PRIMARY"})
    img.add("layer.fill_color", "BACK_SPACE", ctrl=True, props={"which": "SECONDARY"})

    # ---------------------------------------------------------------- 着色器编辑器（照 Blender 节点编辑器）
    sh = kc.keymap("Shader Editor", "SHADER_EDITOR")
    sh.add("wm.call_menu", "A", shift=True, props={"name": "SHADER_MT_add"})
    sh.add("shader.delete", "X")
    sh.add("shader.delete", "DEL")
    sh.add("shader.delete_reconnect", "X", ctrl=True)
    sh.add("shader.duplicate_move", "D", shift=True)
    sh.add("shader.clipboard_copy", "C", ctrl=True)
    sh.add("shader.clipboard_paste", "V", ctrl=True)
    sh.add("shader.mute_toggle", "M")
    sh.add("shader.hide_toggle", "H")
    sh.add("shader.link_make", "F")
    sh.add("shader.select_all", "A", props={"action": "SELECT"})
    sh.add("shader.select_all", "A", alt=True, props={"action": "DESELECT"})
    sh.add("shader.select_all", "I", ctrl=True, props={"action": "INVERT"})
    sh.add("shader.view_all", "HOME")
    sh.add("shader.view_selected", "NUMPAD_PERIOD")
    sh.add("shader.translate", "G")
    sh.add("shader.rename", "F2")
    sh.add("wm.call_menu", "APP", props={"name": "SHADER_MT_context_menu"})

    # ---------------------------------------------------------------- 节点编辑器
    node = kc.keymap("Node", "NODE_EDITOR")
    node.add("node.delete", "X")
    node.add("node.delete", "DEL")
    node.add("node.add_menu", "A", shift=True)
    node.add("node.duplicate", "D", ctrl=True)
    node.add("node.duplicate", "D", shift=True)
    node.add("node.view_all", "HOME")

    # ---------------------------------------------------------------- 图层
    layers = kc.keymap("Layers", "LAYERS")
    layers.add("layer.delete", "DEL")
    layers.add("layer.delete", "X")
    layers.add("layer.duplicate", "D", shift=True)
    layers.add("layer.add_paint", "N", ctrl=True, shift=True)
    layers.add("layer.group", "G", ctrl=True)
    layers.add("layer.duplicate", "J", ctrl=True)
    layers.add("layer.add_adjust", "L", ctrl=True, props={"adjust": "LEVELS"})
    layers.add("layer.add_adjust", "U", ctrl=True, props={"adjust": "HSV"})
    layers.add("layer.add_adjust", "I", ctrl=True, props={"adjust": "INVERT"})
    layers.add("layer.add_adjust", "M", ctrl=True, props={"adjust": "CURVES"})
    layers.add("layer.add_adjust", "B", ctrl=True, props={"adjust": "COLOR_BALANCE"})
    layers.add("layer.add_adjust", "B", ctrl=True, alt=True, shift=True, props={"adjust": "BLACK_WHITE"})
    layers.add("layer.adjust_auto", "L", ctrl=True, shift=True, props={"mode": "TONE"})
    layers.add("layer.adjust_auto", "L", ctrl=True, alt=True, shift=True, props={"mode": "CONTRAST"})
    layers.add("layer.adjust_auto", "B", ctrl=True, shift=True, props={"mode": "COLOR"})
    layers.add("layer.add_desaturate", "U", ctrl=True, shift=True)
    layers.add("layer.fill_color", "BACK_SPACE", alt=True, props={"which": "PRIMARY"})
    layers.add("layer.fill_color", "BACK_SPACE", ctrl=True, props={"which": "SECONDARY"})

    # ---------------------------------------------------------------- 大纲视图
    outliner = kc.keymap("Outliner", "OUTLINER")
    outliner.add("outliner.select_all", "A", props={"action": "SELECT"})
    outliner.add("outliner.select_all", "A", alt=True, props={"action": "DESELECT"})
    outliner.add("outliner.select_all", "I", ctrl=True, props={"action": "INVERT"})
    outliner.add("outliner.delete", "X")
    outliner.add("outliner.delete", "DEL")
    outliner.add("outliner.hide", "H")
    outliner.add("outliner.unhide_all", "H", alt=True)
    outliner.add("outliner.show_active", "NUMPAD_PERIOD")
    outliner.add("outliner.show_active", "PERIOD")
    outliner.add("outliner.rename", "F2")
    outliner.add("object.duplicate_move", "D", shift=True)
    return kc
