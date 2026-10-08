"""偏好设置：界面由偏好里声明的属性自动生成，新增一个属性就自动多一项。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ..core import ops, registry
from ..core.props import PointerProperty
from ..ui import theme
from ..ui.editor import Editor
from ..ui.layout import UILayout
from ..ui.widgets import TabStrip

SECTION_ICONS = {"interface": "window", "navigation": "perspective", "memory": "memory", "paint": "tool.brush",
                 "viewport": "editor.view3d", "files": "folder", "keymap": "keyboard", "addons": "addon"}
NOTES = {
    "memory": "预算留 0 表示按这台电脑的显卡和内存自动决定。改动在下次启动时生效。",
    "viewport": "抗锯齿、各向异性过滤和垂直同步的改动在下次启动时生效。",
}


@registry.register_editor
class PreferencesEditor(Editor):
    idname = "PREFERENCES"
    label = "偏好设置"
    icon = "editor.prefs"
    category = "通用"
    order = 90
    description = "缓存预算、导航习惯、界面缩放、键位"

    def __init__(self, app, area=None) -> None:
        self._section = "interface"
        self._layout = None
        super().__init__(app, area)

    def sections(self) -> list[tuple]:
        prefs = self.app.prefs
        items = []
        for name, prop in type(prefs).properties().items():
            if isinstance(prop, PointerProperty):
                items.append((name, prop.label, "", SECTION_ICONS.get(name, "settings")))
        items.append(("addons", "插件", "", "addon"))
        items.append(("keymap", "键位", "", "keyboard"))
        return items

    def build_main(self):
        root = QWidget()
        box = QHBoxLayout(root)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.tabs = TabStrip(self.sections(), vertical=True)
        self.tabs.setCurrent(self._section)
        self.tabs.currentChanged.connect(self._on_section)
        box.addWidget(self.tabs)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.page = QWidget()
        self.page.setContentsMargins(theme.px(16), theme.px(12), theme.px(16), theme.px(12))
        self.scroll.setWidget(self.page)
        box.addWidget(self.scroll, 1)
        self._built = None
        return root

    def draw_header(self, layout, ctx) -> None:
        layout.label("改动立即保存", role="faint")
        layout.stretch()

    def _on_section(self, ident) -> None:
        self._section = ident
        self._rebuild()

    def _rebuild(self) -> None:
        if self._built == self._section:
            return
        self._built = self._section
        if self._layout is not None:
            self._layout.clear()
        else:
            self._layout = UILayout(self.page, self.context(), "COLUMN")
        layout = self._layout
        layout.ctx = self.context()
        layout.use_property_split = True
        if self._section == "keymap":
            self._draw_keymap(layout)
            return
        if self._section == "addons":
            self._draw_addons(layout)
            return
        group = getattr(self.app.prefs, self._section)
        layout.label(type(self.app.prefs).prop(self._section).label, role="title")
        note = NOTES.get(self._section)
        if note:
            layout.label(note, role="dim")
        layout.separator()
        for name, prop in type(group).properties().items():
            if prop.hidden:
                continue
            layout.prop(group, name)

    def _draw_addons(self, layout) -> None:
        from ..core import addons

        layout.label("插件", role="title")
        layout.label("功能可以做成插件：启用时加进程序，停用时拿掉。内置插件随程序提供，也可以安装别人写的。", role="dim")
        row = layout.row(align=True)
        row.use_property_split = False
        row.operator("preferences.addon_install", icon="import")
        row.operator("preferences.addon_refresh", icon="reset")
        row.operator("preferences.addon_open_folder", icon="folder")
        layout.separator()
        category = None
        for addon in addons.addons():
            if addon.category != category:
                category = addon.category
                layout.label(category, role="title")
            row = layout.row(align=True)
            row.use_property_split = False
            text = "%s  %s" % ("☑" if addon.enabled else "☐", addon.name)
            row.operator("preferences.addon_toggle", text=text, module=addon.module)
            meta = "，".join(v for v in (addon.version_text and "版本 " + addon.version_text,
                                          addon.info.get("author") and "作者 " + str(addon.info.get("author")),
                                          "内置" if addon.builtin else "用户安装") if v)
            row.label(meta, role="dim")
            if not addon.builtin:
                row.operator("preferences.addon_remove", text="", icon="remove", module=addon.module)
            description = addon.info.get("description")
            if description:
                layout.label(str(description), role="dim")
            if addon.error:
                layout.label(addon.error, icon="warning")
            if addon.enabled and addon.preferences is not None:
                box = layout.column()
                box.use_property_split = True
                for name, prop in type(addon.preferences).properties().items():
                    if not prop.hidden:
                        box.prop(addon.preferences, name)
            layout.separator()

    def _draw_keymap(self, layout) -> None:
        layout.label("键位", role="title")
        layout.label("默认键位对齐 Blender：中键旋转，Shift+中键平移，滚轮缩放。", role="dim")
        layout.separator()
        kc = self.app.keyconfig
        titles = {"Window": "全局", "Screen": "窗口布局", "3D View": "3D 视口", "Paint": "绘制", "Image": "UV / 图像",
                  "Layers": "图层"}
        for name, km in kc.keymaps.items():
            layout.label(titles.get(name, name), role="title")
            for item in km.items:
                cls = ops.get(item.op)
                if cls is None:
                    continue
                text = cls.label
                if item.props:
                    text += "（%s）" % "，".join(str(v) for v in item.props.values())
                row = layout.row(heading=text)
                row.label(item.shortcut_text(), role="dim")
            layout.separator()

    def refresh(self) -> None:
        super().refresh()
        if self._section == "addons":
            self._built = None                 # 插件状态可能变了：插件页每次重画
        self._rebuild()

    def save_state(self) -> dict:
        state = super().save_state()
        state["section"] = self._section
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        section = (state or {}).get("section", self._section)
        if section in [item[0] for item in self.sections()]:
            self._section = section
            self.tabs.setCurrent(section)
