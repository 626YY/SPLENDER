"""图层编辑器：图层栈、蒙版、显隐、排序。最上面的图层排在最上面。"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QLineEdit, QScrollArea, QVBoxLayout, QWidget

from ..core import ops, registry
from ..ui import icons, theme
from ..ui.editor import Editor
from ..ui.layout import UILayout
from ..ui.widgets import build_menu


class LayerList(QWidget):
    """图层列表。自绘：眼睛、类型图标、名字、蒙版、锁。支持拖动排序和就地改名。"""

    def __init__(self, editor: "LayersEditor") -> None:
        super().__init__()
        self.editor = editor
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self._hover = -1
        self._press = None            # (行, 位置, 区域)
        self._drag_row = -1
        self._drop_index = -1
        self._drop_target = None          # (所属文件夹 uid, 列表下标, 指示：("into", 行) 或 ("line", 行界, 层级))
        self._rename: QLineEdit | None = None

    # ---- 数据 ----
    def texture_set(self):
        return self.editor.app.project.active_texture_set if self.editor.app.project is not None else None

    def rows(self) -> list:
        """从上往下显示的行：折叠的文件夹里的图层不显示。"""
        ts = self.texture_set()
        if ts is None:
            return []
        collapsed = self.editor.collapsed
        found = []
        for layer in reversed(ts.layers):
            parent = ts.parent_of(layer)
            hidden = False
            while parent is not None:
                if parent.uid in collapsed:
                    hidden = True
                    break
                parent = ts.parent_of(parent)
            if not hidden:
                found.append(layer)
        return found

    def indent(self) -> int:
        return theme.px(14)

    # ---- 几何 ----
    def row_height(self) -> int:
        return theme.px(34)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(theme.px(220), max(self.row_height() * len(self.rows()) + theme.px(8), theme.px(60)))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(theme.px(160), self.row_height() * max(1, len(self.rows())) + theme.px(8))

    def row_rect(self, row: int) -> QRect:
        pad = theme.px(4)
        return QRect(pad, pad + row * self.row_height(), self.width() - 2 * pad, self.row_height() - theme.px(2))

    def row_at(self, pos: QPoint) -> int:
        row = (pos.y() - theme.px(4)) // self.row_height()
        return int(row) if 0 <= row < len(self.rows()) else -1

    def zones(self, rect: QRect, layer) -> dict:
        h = rect.height()
        eye = QRect(rect.left() + theme.px(2), rect.top(), theme.px(26), h)
        ts = self.texture_set()
        depth = ts.depth(layer) if ts is not None else 0
        arrow = QRect(eye.right() + theme.px(1) + depth * self.indent(), rect.top(), theme.px(14), h)
        thumb_side = h - theme.px(8)
        thumb = QRect(arrow.right() + theme.px(2), rect.top() + theme.px(4), thumb_side, thumb_side)
        right = rect.right() - theme.px(4)
        lock = QRect(right - theme.px(20), rect.top(), theme.px(20), h)
        mask = QRect(0, 0, 0, 0)
        name_right = lock.left() - theme.px(2)
        if layer.has_mask:
            mask = QRect(name_right - thumb_side, rect.top() + theme.px(4), thumb_side, thumb_side)
            name_right = mask.left() - theme.px(6)
        name = QRect(thumb.right() + theme.px(8), rect.top(), max(10, name_right - thumb.right() - theme.px(8)), h)
        return {"eye": eye, "arrow": arrow, "thumb": thumb, "name": name, "mask": mask, "lock": lock}

    # ---- 绘制 ----
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), theme.qcolor("bg.panel"))
        ts = self.texture_set()
        rows = self.rows()
        if ts is None or not rows:
            painter.setPen(theme.qcolor("text.faint"))
            painter.setFont(theme.font())
            painter.drawText(self.rect(), Qt.AlignCenter, "还没有图层" if ts is not None else "没有可绘制的模型")
            return
        radius = theme.size("radius")
        active = ts.active_layer
        for row, layer in enumerate(rows):
            rect = self.row_rect(row)
            is_active = layer is active
            if is_active:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.qcolor("accent.soft"))
                painter.drawRoundedRect(QRectF(rect), radius, radius)
            elif row == self._hover:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.qcolor("bg.row_hover"))
                painter.drawRoundedRect(QRectF(rect), radius, radius)
            z = self.zones(rect, layer)
            dim = not layer.visible
            text_color = "text.disabled" if dim else "text"
            icon_side = theme.size("icon")
            eye_icon = icons.pixmap("visible" if layer.visible else "hidden",
                                    theme.color("text.dim" if layer.visible else "text.faint"), icon_side,
                                    self.devicePixelRatioF())
            painter.drawPixmap(z["eye"].center() - QPoint(icon_side // 2, icon_side // 2), eye_icon)
            if layer.kind == "FOLDER":
                expanded = layer.uid not in self.editor.collapsed
                pm = icons.pixmap("chevron.down" if expanded else "chevron.right", theme.color("text.dim"),
                                  icon_side, self.devicePixelRatioF())
                painter.drawPixmap(z["arrow"].center() - QPoint(icon_side // 2, icon_side // 2), pm)
            # 内容缩略块：填充层显示它的颜色，其他显示类型图标
            thumb = QRectF(z["thumb"])
            painter.setPen(Qt.NoPen)
            if layer.kind == "FILL":
                r, g, b = layer.fill_color
                painter.setBrush(QColor.fromRgbF(r, g, b))
                painter.drawRoundedRect(thumb, theme.px(3), theme.px(3))
            else:
                painter.setBrush(theme.qcolor("bg.field"))
                painter.drawRoundedRect(thumb, theme.px(3), theme.px(3))
                kind_icon = {"FOLDER": "layer.folder", "ADJUST": "adjust"}.get(layer.kind, "layer.paint")
                pm = icons.pixmap(kind_icon, theme.color("text.dim"), icon_side, self.devicePixelRatioF())
                painter.drawPixmap(thumb.center().toPoint() - QPoint(icon_side // 2, icon_side // 2), pm)
            content_target = is_active and ts.paint_target == "CONTENT"
            if content_target:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(theme.qcolor("accent"), theme.px(2)))
                painter.drawRoundedRect(thumb.adjusted(-1, -1, 1, 1), theme.px(4), theme.px(4))
            if layer.has_mask:
                mask = QRectF(z["mask"])
                painter.setPen(Qt.NoPen)
                value = int(255 * layer.mask_default)
                painter.setBrush(QColor(value, value, value) if layer.mask_enabled else theme.qcolor("bg.widget"))
                painter.drawRoundedRect(mask, theme.px(3), theme.px(3))
                pm = icons.pixmap("layer.mask", theme.color("accent" if layer.mask_enabled else "text.faint"),
                                  icon_side, self.devicePixelRatioF())
                painter.drawPixmap(mask.center().toPoint() - QPoint(icon_side // 2, icon_side // 2), pm)
                if is_active and ts.paint_target == "MASK":
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(theme.qcolor("accent"), theme.px(2)))
                    painter.drawRoundedRect(mask.adjusted(-1, -1, 1, 1), theme.px(4), theme.px(4))
            painter.setFont(theme.font())
            painter.setPen(theme.qcolor(text_color))
            metrics = painter.fontMetrics()
            name = metrics.elidedText(layer.name, Qt.ElideRight, z["name"].width())
            painter.drawText(z["name"], Qt.AlignVCenter | Qt.AlignLeft, name)
            if layer.locked:
                pm = icons.pixmap("locked", theme.color("text.dim"), icon_side, self.devicePixelRatioF())
                painter.drawPixmap(z["lock"].center() - QPoint(icon_side // 2, icon_side // 2), pm)
        if self._drag_row >= 0 and self._drop_target is not None:
            hint = self._drop_target[2]
            painter.setPen(QPen(theme.qcolor("accent"), theme.px(2)))
            if hint[0] == "into":
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(QRectF(self.row_rect(hint[1])).adjusted(1, 1, -1, -1), radius, radius)
            else:
                y = theme.px(4) + hint[1] * self.row_height() - 1
                x = theme.px(34) + hint[2] * self.indent()
                painter.drawLine(x, y, self.width() - theme.px(6), y)

    # ---- 鼠标 ----
    def _zone_at(self, row: int, pos: QPoint) -> str:
        layer = self.rows()[row]
        z = self.zones(self.row_rect(row), layer)
        for name in ("eye", "mask", "lock"):
            if z[name].contains(pos):
                return name
        if layer.kind == "FOLDER" and z["arrow"].contains(pos):
            return "arrow"
        return "name"

    def _target_at(self, pos: QPoint):
        """拖到这里时放到哪：(所属文件夹 uid, 列表下标, 指示)。拖到文件夹行中间就放进它，放在最上面。"""
        ts = self.texture_set()
        rows = self.rows()
        if ts is None or not rows or self._drag_row < 0:
            return None
        dragged = rows[self._drag_row]
        height = self.row_height()
        y = pos.y() - theme.px(4)
        row = int(y // height)
        frac = (y - row * height) / float(height)
        if 0 <= row < len(rows):
            target = rows[row]
            if (target.kind == "FOLDER" and 0.25 < frac < 0.75 and target is not dragged
                    and not ts.is_ancestor(dragged, target)):
                return target.uid, ts.index_of(target), ("into", row)
        line = max(0, min(len(rows), row + (1 if frac >= 0.5 else 0)))
        above = rows[line - 1] if line > 0 else None
        if above is None:
            return 0, len(ts.layers), ("line", line, 0)
        if above.kind == "FOLDER" and above.uid not in self.editor.collapsed and above is not dragged:
            return above.uid, ts.index_of(above), ("line", line, ts.depth(above) + 1)
        start, _end = ts.subtree_range(above)
        return above.parent_uid, start, ("line", line, ts.depth(above))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        if self._press is not None and event.buttons() & Qt.LeftButton:
            row, start, zone = self._press
            if zone == "name" and (pos - start).manhattanLength() > theme.px(6):
                self._drag_row = row
            if self._drag_row >= 0:
                self._drop_target = self._target_at(pos)
                self._drop_index = 0 if self._drop_target is not None else -1
                self.update()
                return
        hover = self.row_at(pos)
        if hover != self._hover:
            self._hover = hover
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = -1
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        row = self.row_at(pos)
        if row < 0:
            return
        layer = self.rows()[row]
        ctx = self.editor.context()
        zone = self._zone_at(row, pos)
        if event.button() == Qt.RightButton:
            ops.call("layer.select", ctx, uid=layer.uid)
            menu = build_menu("LAYERS_MT_context", self.editor.context(), self)
            menu.popup(event.globalPosition().toPoint())
            return
        if event.button() != Qt.LeftButton:
            return
        if zone == "arrow":
            collapsed = self.editor.collapsed
            if layer.uid in collapsed:
                collapsed.discard(layer.uid)
            else:
                collapsed.add(layer.uid)
            self.updateGeometry()
            self.update()
            return
        if zone == "eye":
            ops.call("layer.toggle_visible", ctx, uid=layer.uid)
        elif zone == "lock":
            ops.call("layer.toggle_lock", ctx, uid=layer.uid)
        elif zone == "mask":
            ops.call("layer.select", ctx, uid=layer.uid, target="MASK")
        else:
            ops.call("layer.select", ctx, uid=layer.uid, target="CONTENT")
            self._press = (row, pos, zone)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_row >= 0 and self._drop_target is not None:
            layer = self.rows()[self._drag_row]
            parent_uid, index, _hint = self._drop_target
            ops.call("layer.move_to", self.editor.context(), uid=layer.uid, index=index, parent=parent_uid)
            if parent_uid:
                self.editor.collapsed.discard(parent_uid)
        self._press = None
        self._drag_row = -1
        self._drop_index = -1
        self._drop_target = None
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        row = self.row_at(pos)
        if row >= 0 and self._zone_at(row, pos) == "name":
            self.start_rename(row)

    def start_rename(self, row: int) -> None:
        layer = self.rows()[row]
        rect = self.zones(self.row_rect(row), layer)["name"]
        edit = QLineEdit(layer.name, self)
        edit.setGeometry(rect.adjusted(-theme.px(4), theme.px(4), 0, -theme.px(4)))
        edit.selectAll()
        edit.show()
        edit.setFocus()
        self._rename = edit

        def commit() -> None:
            text = edit.text().strip()
            if text and text != layer.name:
                old = layer.name
                layer.name = text
                history = self.editor.app.history
                history.push("重命名图层", lambda: setattr(layer, "name", old), lambda: setattr(layer, "name", text))
            edit.deleteLater()
            self._rename = None
            self.update()

        edit.editingFinished.connect(commit)


@registry.register_editor
class LayersEditor(Editor):
    idname = "LAYERS"
    label = "图层"
    icon = "editor.layers"
    category = "绘制"
    order = 20
    description = "图层栈、蒙版和混合"
    keymaps = ["Layers"]

    def __init__(self, app, area=None) -> None:
        self.collapsed: set[int] = set()      # 折叠起来的文件夹（界面状态，不进撤销）
        super().__init__(app, area)

    def save_state(self) -> dict:
        state = super().save_state()
        state["collapsed"] = sorted(self.collapsed)
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        self.collapsed = set(int(v) for v in (state or {}).get("collapsed", []))

    def build_main(self):
        root = QWidget()
        box = QVBoxLayout(root)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.strip = QWidget()
        self.strip.setContentsMargins(theme.size("pad"), theme.size("gap"), theme.size("pad"), theme.size("gap"))
        box.addWidget(self.strip)
        self.list = LayerList(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.list)
        box.addWidget(scroll, 1)
        self._strip_layout = None
        self._strip_key = None
        return root

    def draw_header(self, layout, ctx) -> None:
        project = self.app.project
        layout.menu("LAYERS_MT_add", text="新建", icon="add")
        layout.menu("LAYERS_MT_filter", text="滤镜")
        layout.operator("layer.duplicate", text="", icon="duplicate")
        layout.operator("layer.move", text="", icon="up", direction="UP")
        layout.operator("layer.move", text="", icon="down", direction="DOWN")
        layout.operator("layer.delete", text="", icon="remove")
        layout.stretch()
        if project is not None and len(project.texture_sets) > 1:
            layout.label(project.active_texture_set.name, role="dim")

    def _rebuild_strip(self) -> None:
        """列表上方的一条：当前图层的不透明度和基础色混合模式。"""
        ts = self.app.project.active_texture_set if self.app.project is not None else None
        layer = ts.active_layer if ts is not None else None
        key = (layer.uid if layer is not None else 0, ts.uid if ts is not None else 0)
        if key == self._strip_key:
            return
        self._strip_key = key
        if self._strip_layout is not None:
            self._strip_layout.clear()
        else:
            self._strip_layout = UILayout(self.strip, self.context(), "ROW")
        layout = self._strip_layout
        layout.ctx = self.context()
        if layer is None:
            layout.label("没有选中的图层", role="faint")
            return
        layout.prop(layer, "blend_basecolor", text="")
        layout.prop(layer, "opacity", text="不透明度", slider=True)

    def refresh(self) -> None:
        super().refresh()
        self._rebuild_strip()
        self.list.updateGeometry()
        self.list.update()
