"""界面控件展示：在离屏平台搭出全部控件的各种状态，按界面缩放 1.0 和 1.5 各截一张图。

用法：python tools/ui_gallery.py [输出目录]
默认输出到 docs/evidence/ui-kit/。强制使用离屏平台，不会在桌面上弹出任何窗口。

截图内容：
- ui_kit_scale1.0.png / ui_kit_scale1.5.png：控件状态表（常态、悬停、按下、选中、键盘焦点、禁用）、
  面板宿主（笔刷、图层、视口显示、收起的面板）、嵌入的取色器、标签条、文字层级、标题栏写法。
- colorpicker_popover_scale*.png：色块点开的取色浮层。
- popover_scale*.png：标题栏按钮点开的属性浮层。
- menu_scale*.png：由 registry.Menu 生成的菜单（带快捷键、置灰项、子菜单）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"     # 只在离屏平台上画，绝不弹窗
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFontDatabase, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QSizePolicy, QVBoxLayout,  # noqa: E402
                               QWidget)

from splender.core import ops, registry  # noqa: E402
from splender.core.context import Context  # noqa: E402
from splender.core.history import History  # noqa: E402
from splender.core.keymap import default_keyconfig  # noqa: E402
from splender.doc.project import BLEND_ITEMS, Brush, Layer, ToolSettings, ViewShading  # noqa: E402

FONT_FILES = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/segoeui.ttf")


def load_fonts() -> None:
    """离屏平台默认没有字体，中文会变方块。这里只在展示脚本里加载系统字体。"""
    for path in FONT_FILES:
        if Path(path).is_file():
            QFontDatabase.addApplicationFont(path)


# ---------------------------------------------------------------------------
# 演示用的应用对象、操作、菜单、面板
# ---------------------------------------------------------------------------
class DemoApp:
    def __init__(self) -> None:
        self.prefs = None
        self.keyconfig = default_keyconfig()
        self.project = None
        self.history = History()
        self.tool_settings = ToolSettings()
        self.engine = None
        self.wm = None


class DemoData:
    brush: Brush
    layer: Layer
    shading: ViewShading


DATA = DemoData()


@ops.register
class DemoAddLayer(ops.Operator):
    idname = "layer.add_paint"
    label = "新建绘制图层"
    description = "在当前图层上方新建一个空的绘制图层"
    icon = "add"


@ops.register
class DemoRemoveLayer(ops.Operator):
    idname = "layer.delete"
    label = "删除图层"
    description = "删除当前图层"
    icon = "remove"


@ops.register
class DemoBake(ops.Operator):
    idname = "gallery.bake"
    label = "烘焙网格贴图"
    description = "从模型烘焙曲率、环境光遮蔽等贴图"
    icon = "zap"

    @classmethod
    def poll(cls, ctx):
        return False          # 演示置灰


@ops.register
class DemoSave(ops.Operator):
    idname = "wm.save"
    label = "保存"
    description = "保存工程"
    icon = "save"


@ops.register
class DemoUndo(ops.Operator):
    idname = "ed.undo"
    label = "撤销"
    description = "撤销上一步"
    icon = "undo"


@ops.register
class DemoRedo(ops.Operator):
    idname = "ed.redo"
    label = "重做"
    description = "重做撤销的一步"
    icon = "redo"

    @classmethod
    def poll(cls, ctx):
        return False


@registry.register_menu
class DemoRecentMenu(registry.Menu):
    idname = "GALLERY_MT_recent"
    label = "最近的工程"

    def draw(self, layout, ctx):
        layout.label("角色_头部.splender")
        layout.label("道具_剑.splender")


@registry.register_menu
class DemoFileMenu(registry.Menu):
    idname = "GALLERY_MT_file"
    label = "文件"

    def draw(self, layout, ctx):
        layout.operator("wm.save")
        layout.operator("ed.undo")
        layout.operator("ed.redo")
        layout.separator()
        layout.menu("GALLERY_MT_recent")
        layout.operator("gallery.bake")
        layout.separator()
        layout.prop(DATA.shading, "show_wireframe")
        layout.prop(DATA.shading, "view_transform")


class BrushPanel(registry.Panel):
    idname = "GALLERY_PT_brush"
    label = "笔刷"
    space = "GALLERY"
    order = 1

    def draw(self, layout, ctx):
        brush = DATA.brush
        row = layout.row(align=True)
        row.prop(brush, "size")
        row.prop(brush, "pressure_size", text="", icon="pressure", icon_only=True)
        layout.prop(brush, "size_unit", expand=True)
        col = layout.column(align=True)
        col.prop(brush, "flow")
        col.prop(brush, "opacity")
        col.prop(brush, "hardness")
        layout.prop(brush, "spacing")
        layout.prop(brush, "color")
        layout.prop(brush, "secondary_color")
        row = layout.row(align=True, heading="通道")
        row.prop(brush, "use_basecolor", toggle=True)
        row.prop(brush, "use_metallic", toggle=True)
        row.prop(brush, "use_roughness", toggle=True)
        row.prop(brush, "use_height", toggle=True)
        layout.prop(brush, "normal_falloff")
        layout.prop(brush, "backface_cull")
        layout.prop(brush, "pressure_flow")


class LayerPanel(registry.Panel):
    idname = "GALLERY_PT_layer"
    label = "图层"
    space = "GALLERY"
    order = 2
    icon = "layers"

    def draw(self, layout, ctx):
        layer = DATA.layer
        layout.prop(layer, "name")
        layout.prop(layer, "kind", expand=True)
        layout.prop(layer, "opacity")
        layout.prop(layer, "blend_basecolor")
        layout.prop(layer, "fill_color")
        layout.prop(layer, "visible")
        sub = layout.column()
        sub.enabled = layer.kind == "FILL"
        sub.prop(layer, "fill_roughness")
        row = layout.row(align=True)
        row.operator("layer.add_paint")
        row.operator("layer.delete", text="", icon="remove")
        layout.operator("gallery.bake")

    def draw_header(self, layout, ctx):
        layout.prop(DATA.layer, "locked", text="", icon="locked", icon_only=True)


class ShadingPanel(registry.Panel):
    idname = "GALLERY_PT_shading"
    label = "视口显示"
    space = "GALLERY"
    order = 3
    icon = "environment"

    def draw(self, layout, ctx):
        shading = DATA.shading
        layout.prop(shading, "mode")
        layout.prop(shading, "channel")
        layout.prop(shading, "hdri_rotation")
        layout.prop(shading, "hdri_strength")
        layout.prop(shading, "exposure")
        layout.prop(shading, "view_transform", expand=True)
        box = layout.box()
        box.label("网格与线框", role="dim")
        box.prop(shading, "show_wireframe")
        box.prop(shading, "show_grid")


class ClosedPanel(registry.Panel):
    idname = "GALLERY_PT_closed"
    label = "收起的面板"
    space = "GALLERY"
    order = 4
    default_closed = True

    def draw(self, layout, ctx):
        layout.label("收起时不显示内容")


class HiddenPanel(registry.Panel):
    """poll 为假：不显示。"""

    idname = "GALLERY_PT_hidden"
    label = "不该出现"
    space = "GALLERY"
    order = 5

    @classmethod
    def poll(cls, ctx):
        return False

    def draw(self, layout, ctx):
        layout.label("不该出现")


for _panel in (BrushPanel, LayerPanel, ShadingPanel, ClosedPanel, HiddenPanel):
    registry.register_panel(_panel)


# ---------------------------------------------------------------------------
# 展示窗口
# ---------------------------------------------------------------------------
def _fake_hover(widget: QWidget) -> None:
    widget.setAttribute(Qt.WA_UnderMouse, True)


def _fake_focus(widget: QWidget) -> None:
    widget._kbd_focus = True
    widget.hasFocus = lambda: True        # 只影响 Python 侧绘制判断，用来同时展示多个焦点态


class Area(QWidget):
    """模拟一个编辑器区域：上面是标题栏，下面是编辑器主体。"""

    def __init__(self, title: str, body: QWidget, header_fill=None) -> None:
        super().__init__()
        from splender.ui import theme
        from splender.ui.widgets import Label

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = QWidget()
        self.header.setFixedHeight(theme.size("header"))
        hl = QHBoxLayout(self.header)
        pad = theme.size("pad")
        hl.setContentsMargins(pad, 0, theme.size("gap"), 0)
        hl.setSpacing(theme.size("gap"))
        if header_fill is None:
            hl.addWidget(Label(title, role="dim"))
            hl.addStretch(1)
        else:
            header_fill(self.header)
        lay.addWidget(self.header)
        lay.addWidget(body, 1)

    def paintEvent(self, event):
        from splender.ui import theme
        p = QPainter(self)
        p.fillRect(self.rect(), theme.qcolor("bg.area"))
        p.fillRect(QRectF(0, 0, self.width(), self.header.height()), theme.qcolor("bg.header"))
        p.end()


class Gallery(QWidget):
    def paintEvent(self, event):
        from splender.ui import theme
        p = QPainter(self)
        p.fillRect(self.rect(), theme.qcolor("bg.window"))
        p.end()


def build_states(ctx) -> QWidget:
    from splender.ui import theme
    from splender.ui.widgets import (Button, CheckBox, ColorButton, EnumField, IconButton, Label, NumberField,
                                     SearchField, SegmentedControl, TextField, Toggle)

    body = QWidget()
    grid = QGridLayout(body)
    pad = theme.size("pad")
    grid.setContentsMargins(pad * 2, pad * 2, pad * 2, pad * 2)
    grid.setHorizontalSpacing(theme.size("pad"))
    grid.setVerticalSpacing(theme.size("pad"))
    states = ["常态", "悬停", "按下", "选中 / 开启", "键盘焦点", "禁用"]
    for col, name in enumerate(states):
        grid.addWidget(Label(name, role="faint"), 0, col + 1)

    def dash() -> QWidget:
        return Label("—", role="faint", align=Qt.AlignHCenter)

    rows = [
        ("数值框", lambda: NumberField(60, 1, 4000, 1, 600, None, 0, "px", "直径")),
        ("滑杆", lambda: NumberField(0.62, 0, 1, None, None, None, 2, "", "流量", slider=True)),
        ("按钮", lambda: Button("新建图层")),
        ("强调按钮", lambda: Button("导出贴图", role="primary")),
        ("开关", lambda: Toggle("对称", "symmetry")),
        ("图标按钮", lambda: IconButton("visible", "显示或隐藏", checkable=True)),
        ("勾选框", lambda: CheckBox("压感控制")),
        ("下拉", lambda: EnumField(BLEND_ITEMS)),
        ("分段按钮", lambda: SegmentedControl([("SOLID", "实体", "", "shade_solid"), ("MATERIAL", "材质", "", "shade_material")])),
        ("色块", lambda: ColorButton((0.80, 0.30, 0.20))),
        ("文本框", lambda: TextField("图层 1")),
        ("搜索框", lambda: SearchField("", "搜索操作")),
    ]
    for r, (name, make) in enumerate(rows, start=1):
        grid.addWidget(Label(name, role="dim"), r, 0)
        for c, state in enumerate(states, start=1):
            w = make()
            applicable = True
            if state == "悬停":
                _fake_hover(w)
                if isinstance(w, NumberField):
                    w._hover_zone = 1
                if isinstance(w, SegmentedControl):
                    w._hover_index = 1
            elif state == "按下":
                if hasattr(w, "setDown") and not isinstance(w, CheckBox):
                    w.setDown(True)
                    _fake_hover(w)
                elif isinstance(w, CheckBox):
                    w.setDown(True)
                    _fake_hover(w)
                elif isinstance(w, NumberField):
                    w._dragging = True
                    w._pressed = True
                elif isinstance(w, EnumField):
                    w._menu = True
                elif isinstance(w, SegmentedControl):
                    w._pressed_index = 1
                    w._hover_index = 1
                    _fake_hover(w)
                else:
                    applicable = False
            elif state == "选中 / 开启":
                if isinstance(w, (Toggle, IconButton, CheckBox)):
                    w.setChecked(True)
                elif isinstance(w, SegmentedControl):
                    w.setCurrent("MATERIAL")
                else:
                    applicable = False
            elif state == "键盘焦点":
                if isinstance(w, TextField):
                    w.setStyleSheet("QLineEdit { border-color: %s; }" % theme.color("line.focus"))
                else:
                    _fake_focus(w)
            elif state == "禁用":
                w.setEnabled(False)
            if not applicable:
                w.deleteLater()
                w = dash()
            w.setMinimumWidth(theme.px(104))
            grid.addWidget(w, r, c)
    # 文字编辑中的数值框
    r = len(rows) + 1
    grid.addWidget(Label("数值框输入中", role="dim"), r, 0)
    editing = NumberField(0.8, 0, 1, None, None, None, 2, "", "不透明度", slider=True)
    editing.setMinimumWidth(theme.px(104))
    grid.addWidget(editing, r, 1)
    body._editing_field = editing
    # 对齐成组
    r += 1
    grid.addWidget(Label("对齐成组", role="dim"), r, 0)
    group = QWidget()
    gl = QHBoxLayout(group)
    gl.setContentsMargins(0, 0, 0, 0)
    gl.setSpacing(int(max(1, int(theme.scale()))))
    from splender.ui.widgets import CORNERS_LEFT, CORNERS_RIGHT
    for i, comp in enumerate(("X", "Y", "Z")):
        f = NumberField((0.0, 1.25, -0.5)[i], label=comp, precision=3)
        f.set_corners(CORNERS_LEFT if i == 0 else CORNERS_RIGHT if i == 2 else 0)
        gl.addWidget(f)
    grid.addWidget(group, r, 1, 1, 3)
    toggles = QWidget()
    tl = QHBoxLayout(toggles)
    tl.setContentsMargins(0, 0, 0, 0)
    tl.setSpacing(int(max(1, int(theme.scale()))))
    for i, (txt, on) in enumerate((("X", True), ("Y", False), ("Z", False))):
        t = Toggle(txt)
        t.setChecked(on)
        t.set_corners(CORNERS_LEFT if i == 0 else CORNERS_RIGHT if i == 2 else 0)
        tl.addWidget(t)
    grid.addWidget(toggles, r, 4, 1, 3)
    grid.setRowStretch(r + 1, 1)
    grid.setColumnStretch(7, 1)
    return body


def build_panels(ctx):
    from splender.ui import theme
    from splender.ui.panels import PanelHost

    host = PanelHost(lambda: ctx, "GALLERY", "SIDEBAR")
    host.setMinimumWidth(theme.px(300))
    host.setMaximumWidth(theme.px(300))
    return host


def build_header(ctx):
    from splender.ui.layout import UILayout
    from splender.ui.widgets import IconButton

    def fill(header: QWidget) -> None:
        holder = QWidget()
        header.layout().addWidget(holder, 1)
        layout = UILayout(holder, ctx, "ROW")
        editor_button = IconButton("editor.properties", "编辑器类型")
        editor_button.set_menu_indicator(True)
        layout.widget(editor_button)
        layout.prop(DATA.shading, "mode", text="", expand=True, icon_only=True)
        layout.separator()
        layout.prop(DATA.shading, "channel", text="")
        layout.stretch()

        def draw_popover(lay, c):
            lay.prop(DATA.shading, "hdri_strength")
            lay.prop(DATA.shading, "hdri_rotation")
            lay.prop(DATA.shading, "background_opacity")
            lay.prop(DATA.shading, "view_transform")
            lay.prop(DATA.shading, "bump")

        layout.popover("显示", "environment", draw_popover)
        layout.menu("GALLERY_MT_file", text="文件")
        layout.prop(DATA.shading, "show_wireframe", text="", icon="wireframe", icon_only=True)
        header._ui_layout = layout

    return fill


def build_extras(ctx) -> QWidget:
    from splender.ui import theme
    from splender.ui.widgets import ColorPicker, Label, Separator, TabStrip

    body = QWidget()
    lay = QVBoxLayout(body)
    pad = theme.size("pad")
    lay.setContentsMargins(pad * 2, pad * 2, pad * 2, pad * 2)
    lay.setSpacing(pad)
    lay.addWidget(Label("取色器（嵌入面板）", role="title"))
    picker = ColorPicker((0.29, 0.55, 0.96))
    picker.setMaximumWidth(theme.px(240))
    lay.addWidget(picker)
    lay.addWidget(Separator())
    lay.addWidget(Label("标签条", role="title"))
    tabs = TabStrip([("PAINT", "绘制"), ("MATERIAL", "材质"), ("UV", "UV"), ("BAKE", "烘焙")])
    tabs.setCurrent("PAINT")
    tabs._hover = 2
    _fake_hover(tabs)
    lay.addWidget(tabs)
    row = QHBoxLayout()
    row.setSpacing(pad * 2)
    vtabs = TabStrip([("TOOL", "工具", "当前工具", "tool.brush"), ("LAYER", "图层", "", "layers"),
                      ("VIEW", "视图", "", "environment"), ("SETTINGS", "设置", "", "settings")],
                     vertical=True, icon_only=True)
    vtabs.setCurrent("LAYER")
    vtext = TabStrip([("BRUSH", "笔刷"), ("LAYER", "图层"), ("VIEW", "视图"), ("HDRI", "HDRI")], vertical=True)
    vtext.setCurrent("BRUSH")
    row.addWidget(vtabs, 0, Qt.AlignTop)
    row.addWidget(vtext, 0, Qt.AlignTop)
    texts = QVBoxLayout()
    texts.setSpacing(0)
    texts.addWidget(Label("小标题", role="title"))
    texts.addWidget(Label("正文：笔刷直径"))
    texts.addWidget(Label("次要：右对齐的属性名", role="dim"))
    texts.addWidget(Label("更淡的小字说明", role="faint"))
    texts.addWidget(Label("带图标的提示", icon="info", role="dim"))
    disabled = Label("禁用的文字")
    disabled.setEnabled(False)
    texts.addWidget(disabled)
    texts.addStretch(1)
    row.addLayout(texts, 1)
    lay.addLayout(row)
    lay.addStretch(1)
    body.setMinimumWidth(theme.px(272))
    return body


def build_gallery(ctx) -> Gallery:
    from splender.ui import theme

    gallery = Gallery()
    lay = QHBoxLayout(gallery)
    gap = theme.size("splitter")
    lay.setContentsMargins(gap, gap, gap, gap)
    lay.setSpacing(gap)
    states = Area("控件状态", build_states(ctx))
    panels = Area("属性", build_panels(ctx), build_header(ctx))
    extras = Area("取色与标签", build_extras(ctx))
    lay.addWidget(states, 1)
    lay.addWidget(panels, 0)
    lay.addWidget(extras, 0)
    gallery._states = states
    gallery._panels = panels
    return gallery


def snapshot(widget: QWidget, path: Path, background: str | None = None) -> None:
    image = widget.grab().toImage()
    if background is not None:
        from splender.ui import theme
        flat = QImage(image.size(), QImage.Format_ARGB32)
        flat.setDevicePixelRatio(image.devicePixelRatio())
        flat.fill(theme.qcolor(background))
        p = QPainter(flat)
        p.drawImage(0, 0, image)
        p.end()
        image = flat
    image.save(str(path))
    print("保存", path)


def run(out_dir: Path) -> int:
    app = QApplication.instance() or QApplication([])
    load_fonts()
    from splender.ui import theme
    from splender.ui.widgets import ColorButton, remember_color

    out_dir.mkdir(parents=True, exist_ok=True)
    for color in ((0.95, 0.75, 0.2), (0.2, 0.6, 0.35), (0.15, 0.15, 0.17), (0.85, 0.85, 0.82), (0.6, 0.2, 0.55)):
        remember_color(color)
    for scale in (1.0, 1.5):
        theme.apply(app, scale)
        DATA.brush = Brush()
        DATA.layer = Layer(name="底漆")
        DATA.shading = ViewShading()
        ctx = Context(app=DemoApp())
        gallery = build_gallery(ctx)
        gallery.resize(theme.px(1320), theme.px(900))
        gallery.show()
        gallery.activateWindow()
        app.processEvents()
        editing = gallery.findChild(QWidget, "")  # noqa: F841
        field = gallery._states.findChildren(QWidget)
        for widget in field:
            if getattr(widget, "_editing_field", None) is not None:
                widget._editing_field.begin_text_edit()
        app.processEvents()
        tag = "%.1f" % scale
        snapshot(gallery, out_dir / ("ui_kit_scale%s.png" % tag))

        # 取色浮层
        button = ColorButton((0.29, 0.55, 0.96), gallery)
        button.resize(theme.px(120), theme.size("widget"))
        button.move(theme.px(40), theme.px(40))
        button.show()
        pop = button.openPicker()
        app.processEvents()
        snapshot(pop, out_dir / ("colorpicker_popover_scale%s.png" % tag), "bg.area")
        pop.close()
        app.processEvents()

        # 标题栏按钮点开的属性浮层
        header_layout = gallery._panels.header._ui_layout
        popover_button = None
        from splender.ui.widgets import Button
        for child in gallery._panels.header.findChildren(Button):
            if child.text() == "显示":
                popover_button = child
        if popover_button is not None:
            popover_button.click()
            app.processEvents()
            pop = getattr(popover_button, "_splender_popover", None)
            if pop is not None:
                snapshot(pop, out_dir / ("popover_scale%s.png" % tag), "bg.area")
                pop.close()
        del header_layout

        # 菜单
        from splender.ui.widgets import build_menu
        menu = build_menu("GALLERY_MT_file", ctx, gallery)
        menu.popup(gallery.mapToGlobal(gallery.rect().center()))
        app.processEvents()
        snapshot(menu, out_dir / ("menu_scale%s.png" % tag), "bg.area")
        menu.close()

        gallery.close()
        gallery.deleteLater()
        app.processEvents()
    theme.apply(app, 1.0)
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "evidence" / "ui-kit"
    raise SystemExit(run(target))
