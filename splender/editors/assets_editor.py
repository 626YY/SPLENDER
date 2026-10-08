"""资源库：环境、色板、笔刷预设。点一下就用。"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QScrollArea, QStackedWidget, QVBoxLayout, QWidget

from ..core import registry
from ..engine import ibl
from ..ui import icons, theme
from ..ui.editor import Editor

TABS = [("HDRI", "环境", "照亮模型的环境图"), ("PALETTE", "色板", "常用颜色"), ("BRUSH", "笔刷", "笔刷预设"),
        ("SMART", "智能材质", "按模型形状自动落位的成套材质")]

#: 笔刷预设：名字、说明、要写进笔刷的数值
BRUSH_PRESETS = [
    ("硬边圆", "边缘清晰，适合铺色块", {"hardness": 0.95, "flow": 1.0, "opacity": 1.0, "spacing": 0.08}),
    ("柔边圆", "边缘柔和，适合过渡", {"hardness": 0.35, "flow": 1.0, "opacity": 1.0, "spacing": 0.08}),
    ("喷枪", "一点点叠上去", {"hardness": 0.0, "flow": 0.12, "opacity": 1.0, "spacing": 0.05}),
    ("勾线", "细而实，压感控制粗细", {"hardness": 0.9, "flow": 1.0, "opacity": 1.0, "spacing": 0.04, "size": 8.0,
                            "pressure_size": True}),
    ("薄涂", "半透明，一笔之内不叠深", {"hardness": 0.6, "flow": 1.0, "opacity": 0.35, "spacing": 0.08}),
    ("大面积", "很大的软笔，快速铺底", {"hardness": 0.2, "flow": 1.0, "opacity": 1.0, "spacing": 0.1, "size": 300.0}),
]


class _Grid(QWidget):
    """图标加文字的网格。子类给出条目并处理点击。"""

    cell = (112, 84)

    def __init__(self, editor) -> None:
        super().__init__()
        self.editor = editor
        self.setMouseTracking(True)
        self._hover = -1

    def items(self) -> list:
        return []

    def columns(self) -> int:
        return max(1, (self.width() - theme.px(8)) // theme.px(self.cell[0]))

    def cell_rect(self, index: int) -> QRect:
        cols = self.columns()
        w, h = theme.px(self.cell[0]), theme.px(self.cell[1])
        return QRect(theme.px(6) + (index % cols) * w, theme.px(6) + (index // cols) * h, w - theme.px(6), h - theme.px(6))

    def index_at(self, pos: QPoint) -> int:
        for index in range(len(self.items())):
            if self.cell_rect(index).contains(pos):
                return index
        return -1

    def sizeHint(self) -> QSize:  # noqa: N802
        count = len(self.items())
        rows = (count + self.columns() - 1) // self.columns()
        return QSize(theme.px(240), rows * theme.px(self.cell[1]) + theme.px(12))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(theme.px(120), self.sizeHint().height())

    def resizeEvent(self, event) -> None:  # noqa: N802
        self.updateGeometry()

    def is_current(self, item) -> bool:
        return False

    def draw_thumb(self, painter: QPainter, rect: QRectF, item) -> None:
        pass

    def label_of(self, item) -> str:
        return ""

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setFont(theme.font("font.small"))
        radius = theme.size("radius")
        for index, item in enumerate(self.items()):
            rect = QRectF(self.cell_rect(index))
            current = self.is_current(item)
            if current or index == self._hover:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.qcolor("accent.soft" if current else "bg.row_hover"))
                painter.drawRoundedRect(rect, radius, radius)
            label_h = theme.px(20)
            thumb = rect.adjusted(theme.px(5), theme.px(5), -theme.px(5), -label_h)
            self.draw_thumb(painter, thumb, item)
            painter.setPen(theme.qcolor("text" if current else "text.dim"))
            text_rect = QRectF(rect.left(), rect.bottom() - label_h, rect.width(), label_h)
            text = painter.fontMetrics().elidedText(self.label_of(item), Qt.ElideRight, int(rect.width()) - theme.px(6))
            painter.drawText(text_rect, Qt.AlignCenter, text)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        hover = self.index_at(event.position().toPoint())
        if hover != self._hover:
            self._hover = hover
            self.update()
            self.setToolTip(self.tip_of(self.items()[hover]) if hover >= 0 else "")

    def tip_of(self, item) -> str:
        return self.label_of(item)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = -1
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        index = self.index_at(event.position().toPoint())
        if index < 0:
            self.pressed_empty(event)
            return
        if event.button() == Qt.LeftButton:
            self.activate(self.items()[index])
        elif event.button() == Qt.RightButton:
            self.context(self.items()[index], event.globalPosition().toPoint())
        self.update()

    def activate(self, item) -> None:
        pass

    def context(self, item, pos: QPoint) -> None:
        pass

    def pressed_empty(self, event) -> None:
        pass


class HdriGrid(_Grid):
    cell = (124, 92)

    def __init__(self, editor) -> None:
        super().__init__(editor)
        self._pixmaps: dict[str, QPixmap] = {}

    def items(self) -> list:
        return ibl.list_hdris()

    def _view(self):
        return getattr(self.editor.app, "active_view3d", None)

    def is_current(self, item) -> bool:
        view = self._view()
        return view is not None and view.shading.hdri == item

    def label_of(self, item) -> str:
        return ibl.label(item, "hdri")

    def draw_thumb(self, painter, rect, item) -> None:
        pm = self._pixmaps.get(item)
        if pm is None:
            pm = QPixmap(str(ibl.thumbnail(item)))
            self._pixmaps[item] = pm
        if not pm.isNull():
            target = QRectF(rect)
            ratio = pm.width() / max(1, pm.height())
            if target.width() / max(1.0, target.height()) > ratio:
                w = target.height() * ratio
                target = QRectF(target.center().x() - w / 2, target.top(), w, target.height())
            else:
                h = target.width() / ratio
                target = QRectF(target.left(), target.center().y() - h / 2, target.width(), h)
            painter.drawPixmap(target, pm, QRectF(pm.rect()))

    def activate(self, item) -> None:
        view = self._view()
        if view is None:
            return
        view.shading.hdri = item
        if view.shading.mode != "MATERIAL":
            view.shading.mode = "MATERIAL"
        self.editor.app.notify("shading")


class PaletteGrid(_Grid):
    cell = (44, 44)

    def items(self) -> list:
        return list(self.editor.app.palette) + ["+"]

    def is_current(self, item) -> bool:
        if item == "+":
            return False
        color = self.editor.app.tool_settings.brush.color
        return all(abs(a - b) < 0.004 for a, b in zip(color, item))

    def label_of(self, item) -> str:
        return ""

    def tip_of(self, item) -> str:
        if item == "+":
            return "把当前笔刷颜色加进色板"
        return "#%02X%02X%02X  右键删除" % tuple(int(round(v * 255)) for v in item)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        for index, item in enumerate(self.items()):
            rect = QRectF(self.cell_rect(index)).adjusted(2, 2, -2, -2)
            if item == "+":
                painter.setPen(QPen(theme.qcolor("line.soft"), 1, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(rect, theme.px(4), theme.px(4))
                side = theme.size("icon")
                pm = icons.pixmap("add", theme.color("text.dim"), side, self.devicePixelRatioF())
                painter.drawPixmap(rect.center().toPoint() - QPoint(side // 2, side // 2), pm)
                continue
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor.fromRgbF(*item))
            painter.drawRoundedRect(rect, theme.px(4), theme.px(4))
            if self.is_current(item) or index == self._hover:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(theme.qcolor("accent" if self.is_current(item) else "text.dim"), theme.px(2)))
                painter.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), theme.px(5), theme.px(5))

    def activate(self, item) -> None:
        app = self.editor.app
        if item == "+":
            color = tuple(app.tool_settings.brush.color)
            if color not in app.palette:
                app.palette.append(color)
                app.save_palette()
                self.updateGeometry()
            return
        app.tool_settings.brush.color = item

    def context(self, item, pos) -> None:
        app = self.editor.app
        if item != "+" and item in app.palette:
            app.palette.remove(item)
            app.save_palette()
            self.updateGeometry()


class BrushGrid(_Grid):
    cell = (96, 84)

    def items(self) -> list:
        return BRUSH_PRESETS

    def label_of(self, item) -> str:
        return item[0]

    def tip_of(self, item) -> str:
        return item[1]

    def draw_thumb(self, painter, rect, item) -> None:
        values = item[2]
        hardness = values.get("hardness", 0.8)
        strength = min(1.0, values.get("flow", 1.0) * values.get("opacity", 1.0) + 0.25)
        side = min(rect.width(), rect.height()) * (0.35 if values.get("size", 60) < 20 else 0.9)
        center = rect.center()
        from PySide6.QtGui import QRadialGradient

        gradient = QRadialGradient(center, side / 2)
        tone = theme.qcolor("text")
        solid = QColor(tone)
        solid.setAlphaF(strength)
        clear = QColor(tone)
        clear.setAlphaF(0.0)
        gradient.setColorAt(0.0, solid)
        gradient.setColorAt(max(0.01, min(0.99, hardness)), solid)
        gradient.setColorAt(1.0, clear)
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(center, side / 2, side / 2)

    def activate(self, item) -> None:
        brush = self.editor.app.tool_settings.active_brush()
        for name, value in item[2].items():
            setattr(brush, name, value)
        self.editor.app.report("笔刷：%s" % item[0])


class SmartGrid(_Grid):
    """智能材质：缩略图是 resources/smart/<id>.png（tools/render_smart_thumbs.py 渲染的材质球）；
    没有图时画一个示意的小球：底色打底，边缘、凹陷、朝向的颜色画在对应的位置上。"""

    cell = (104, 104)

    def __init__(self, editor) -> None:
        super().__init__(editor)
        self._pixmaps: dict[str, QPixmap | None] = {}

    def items(self) -> list:
        from ..bake.smart import SMART_MATERIALS

        return SMART_MATERIALS

    def label_of(self, item) -> str:
        return item["name"]

    def tip_of(self, item) -> str:
        return "%s\n点一下加到当前纹理集" % item["description"]

    def draw_thumb(self, painter, rect, item) -> None:
        from PySide6.QtGui import QPainterPath, QRadialGradient

        from ..paths import resource

        ident = item["id"]
        if ident not in self._pixmaps:
            path = resource("smart", ident + ".png")
            pixmap = QPixmap(str(path)) if path.is_file() else None
            self._pixmaps[ident] = pixmap if pixmap is not None and not pixmap.isNull() else None
        pixmap = self._pixmaps[ident]
        if pixmap is not None:
            side = min(rect.width(), rect.height())
            target = QRectF(rect.center().x() - side / 2, rect.center().y() - side / 2, side, side)
            painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))
            return

        side = min(rect.width(), rect.height()) * 0.92
        center = rect.center()
        r = side / 2
        ball = QPainterPath()
        ball.addEllipse(center, r, r)
        painter.save()
        painter.setClipPath(ball)
        layers = item["layers"]
        base = layers[0]["fill_color"]
        painter.fillRect(rect, QColor.fromRgbF(*base))
        for layer in layers[1:]:
            color = QColor.fromRgbF(*layer["fill_color"])
            color.setAlphaF(float(layer.get("opacity", 1.0)) * 0.95)
            kind = layer.get("mask_generator", "NONE")
            painter.setPen(Qt.NoPen)
            if kind == "EDGES":
                pen = QPen(color, r * 0.22)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                if layer.get("gen_invert"):
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(color)
                    painter.drawEllipse(center, r * 0.82, r * 0.82)
                else:
                    painter.drawEllipse(center, r * 0.95, r * 0.95)
            elif kind in ("CAVITY", "OCCLUSION", "NOISE", "THIN"):
                painter.setBrush(color)
                seed = sum(ord(ch) for ch in layer["name"])
                for k in range(9):
                    a = (seed * 37 + k * 71) % 360
                    d = ((seed * 13 + k * 29) % 70) / 100.0 * r
                    s = (0.10 + ((seed + k * 17) % 9) / 60.0) * r
                    from math import cos, radians, sin

                    painter.drawEllipse(center.x() + cos(radians(a)) * d - s, center.y() + sin(radians(a)) * d - s,
                                        s * 2, s * 2)
            elif kind == "FACING":
                painter.setBrush(color)
                painter.drawEllipse(center.x() - r, center.y() - r * 1.45, r * 2, r * 1.25)
            elif kind == "GRADIENT":
                painter.setBrush(color)
                down = layer.get("gen_direction", "UP") == "DOWN"
                top = center.y() + (r * 0.25 if down else -r)
                painter.drawRect(QRectF(center.x() - r, top, r * 2, r * 0.75))
        shade = QRadialGradient(center.x() - r * 0.35, center.y() - r * 0.4, r * 1.6)
        shade.setColorAt(0.0, QColor(255, 255, 255, 70))
        shade.setColorAt(0.45, QColor(255, 255, 255, 0))
        shade.setColorAt(1.0, QColor(0, 0, 0, 120))
        painter.setPen(Qt.NoPen)
        painter.setBrush(shade)
        painter.drawEllipse(center, r, r)
        painter.restore()

    def activate(self, item) -> None:
        from ..core import ops as core_ops

        core_ops.call("layer.add_smart_material", self.editor.context(), material=item["id"])


@registry.register_editor
class AssetsEditor(Editor):
    idname = "ASSETS"
    label = "资源库"
    icon = "editor.assets"
    category = "通用"
    order = 40
    description = "环境、色板、笔刷预设和智能材质"

    def __init__(self, app, area=None) -> None:
        self._tab = "BRUSH"
        super().__init__(app, area)

    def build_main(self):
        root = QWidget()
        box = QVBoxLayout(root)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.stack = QStackedWidget()
        self.grids = {"HDRI": HdriGrid(self), "PALETTE": PaletteGrid(self), "BRUSH": BrushGrid(self),
                      "SMART": SmartGrid(self)}
        for ident, _label, _tip in TABS:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(self.grids[ident])
            self.stack.addWidget(scroll)
        box.addWidget(self.stack, 1)
        self._show(self._tab)
        return root

    def _show(self, ident: str) -> None:
        self._tab = ident
        self.stack.setCurrentIndex([t[0] for t in TABS].index(ident))

    def draw_header(self, layout, ctx) -> None:
        from ..ui.widgets import TabStrip

        tabs = TabStrip(TABS)
        tabs.setCurrent(self._tab)
        tabs.currentChanged.connect(self._show)
        layout.widget(tabs)
        layout.stretch()

    def refresh(self) -> None:
        super().refresh()
        for grid in self.grids.values():
            grid.update()

    def save_state(self) -> dict:
        state = super().save_state()
        state["tab"] = self._tab
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        tab = (state or {}).get("tab", self._tab)
        if tab in self.grids:
            self._show(tab)
