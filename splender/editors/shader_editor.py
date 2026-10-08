"""着色器编辑器（照 Blender）：当前纹理集的材质节点图。

- 节点上直接显示参数：没接线的输入是数值框、颜色块、XYZ 三格，下拉、开关、色带也都在节点里，改了立刻生效；
- 插口按类型着色（灰：数值，黄：颜色，紫：矢量），从输出口拖到输入口连线，拖接着线的输入口拔下来改接或断开；
- 左键点选（Shift 加减选）、拖动节点挪位置、空白处拖动框选；Ctrl+右键拖动割断连线；
- Shift+A 加节点（加完跟着鼠标走，点一下放下），X 删除、Ctrl+X 删除并接通、Shift+D 复制、M 静音、H 折叠、
  F 连接选中的节点、G 移动、A 全选、Home 看全部、小键盘 . 看选中；右键是节点菜单；滚轮缩放，中键平移。
"""
from __future__ import annotations

import logging
import math

import shiboken6
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsProxyWidget, QGraphicsScene, QGraphicsView, QHBoxLayout,
                               QLabel, QSizePolicy, QVBoxLayout, QWidget)

from ..core import registry
from ..shading import nodes as shader_nodes
from ..ui import theme
from ..ui.editor import Editor

log = logging.getLogger("splender.shading")

HEADER = 22          # 逻辑像素
PAD = 6
SOCKET_R = 5.0


def _px(v: float) -> float:
    return float(theme.px(v))


# ====================================================================== 节点上的一行文字（插口名）
class _RowLabel(QLabel):
    def __init__(self, text: str, right: bool, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setAlignment((Qt.AlignRight if right else Qt.AlignLeft) | Qt.AlignVCenter)
        self.setFixedHeight(theme.size("widget"))
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background: transparent; color: %s;" % theme.color("text"))


class _Body(QWidget):
    """节点的内容（输出名、参数、输入）。背景透明，节点的底色由节点方块画。"""

    def __init__(self) -> None:
        super().__init__()
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.setSpacing(theme.px(2))


# ====================================================================== 节点方块
class ShaderNodeItem(QGraphicsItem):
    def __init__(self, editor: "ShaderEditor", node) -> None:
        super().__init__()
        self.editor = editor
        self.node = node
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setPos(QPointF(*node.location))
        self.setZValue(1.0)
        self.proxy: QGraphicsProxyWidget | None = None
        self.body: _Body | None = None
        self.layouts: list = []
        self.inputs: dict[str, QPointF] = {}
        self.outputs: dict[str, QPointF] = {}
        self._height = _px(HEADER)
        self._signature = None
        self.rebuild()

    # ---- 尺寸 ----
    def width(self) -> float:
        return _px(self.node.type.width)

    def height(self) -> float:
        return self._height

    def boundingRect(self) -> QRectF:  # noqa: N802
        r = _px(SOCKET_R) + 2.0
        return QRectF(-r, -2.0, self.width() + 2 * r, self._height + 4.0)

    def signature(self) -> tuple:
        graph = self.editor.graph()
        linked = tuple(sorted(name for name in self.node.input_names()
                              if graph is not None and graph.input_link(self.node.uid, name) is not None))
        return linked, self.node.hide, self.node.mute

    # ---- 内容 ----
    def rebuild(self) -> None:
        """按现在的连线重排节点内容（接了线的输入只显示名字）。"""
        self.prepareGeometryChange()
        self._signature = self.signature()
        if self.proxy is not None:
            self.proxy.setWidget(None)
            if self.proxy.scene() is not None:
                self.proxy.scene().removeItem(self.proxy)
            self.proxy = None
        if self.body is not None:
            self.body.deleteLater()
            self.body = None
        self.layouts = []
        self.inputs, self.outputs = {}, {}
        node = self.node
        node_type = node.type
        width = self.width()
        header = _px(HEADER)
        graph = self.editor.graph()
        if node.hide:
            # 折叠：只剩标题，接着线的插口排在标题两边
            linked_in = [n for n in node.input_names() if graph is not None and graph.input_link(node.uid, n)]
            linked_out = [n for n in node.output_names() if graph is not None and graph.output_links(node.uid, n)]
            for k, name in enumerate(linked_in):
                self.inputs[name] = QPointF(0.0, header * 0.5 + (k - (len(linked_in) - 1) / 2.0) * _px(6))
            for k, name in enumerate(linked_out):
                self.outputs[name] = QPointF(width, header * 0.5 + (k - (len(linked_out) - 1) / 2.0) * _px(6))
            self._height = header
            self.update()
            return
        if node.type_id == "REROUTE":
            self.inputs["input"] = QPointF(0.0, header * 0.5)
            self.outputs["output"] = QPointF(width, header * 0.5)
            self._height = header
            self.update()
            return
        from ..ui.layout import UILayout

        body = _Body()
        inner = width - 2 * _px(PAD)
        body.setFixedWidth(int(inner))
        rows: list[tuple[str, str, QWidget]] = []           # (输入/输出, 名字, 这一行的控件)
        ctx = self.editor.context()
        for name, label, _kind in node_type.outputs:
            row = _RowLabel(label, True)
            body.box.addWidget(row)
            rows.append(("out", name, row))
        if node_type.params:
            holder = QWidget()
            holder.setStyleSheet("background: transparent;")
            layout = UILayout(holder, ctx)
            layout.use_property_split = False
            for name, prop in node_type.params.items():
                kind = getattr(prop, "kind", "")
                if kind == "ENUM":
                    layout.prop(node.values, name, text="")
                elif kind == "FLOAT":
                    layout.prop(node.values, name, slider=getattr(prop, "subtype", "") == "FACTOR")
                else:
                    layout.prop(node.values, name, text="" if kind == "RAMP" else None)
            self.layouts.append(layout)
            body.box.addWidget(holder)
        for name, label, kind, _default, opts in node_type.inputs:
            linked = graph is not None and graph.input_link(node.uid, name) is not None
            if linked or opts.get("implicit") or kind == "ANY":
                row = _RowLabel(label, False)
                body.box.addWidget(row)
                rows.append(("in", name, row))
                continue
            holder = QWidget()
            holder.setStyleSheet("background: transparent;")
            layout = UILayout(holder, ctx)
            layout.use_property_split = False
            if kind == "FLOAT":
                layout.prop(node.values, "in_" + name, text=label,
                            slider=opts.get("subtype") == "FACTOR")
            elif kind == "COLOR":
                row_layout = layout.row(align=True)
                row_layout.label(label)
                row_layout.prop(node.values, "in_" + name, text="")
            else:
                layout.label(label)
                column = layout.column(align=True)
                column.prop(node.values, "in_" + name, text="")
            self.layouts.append(layout)
            body.box.addWidget(holder)
            rows.append(("in", name, holder))
        body.adjustSize()
        body.box.activate()
        self.body = body
        self.proxy = QGraphicsProxyWidget(self)
        self.proxy.setWidget(body)
        self.proxy.setPos(_px(PAD), header + _px(3))
        self._height = header + _px(3) + body.sizeHint().height() + _px(PAD)
        # 插口位置：每一行的中间（矢量输入对齐它的名字那一行）
        body.resize(body.sizeHint())
        body.box.activate()
        for side, name, widget in rows:
            geo = widget.geometry()
            y = header + _px(3) + (geo.top() + min(geo.height(), theme.size("widget")) * 0.5)
            if side == "out":
                self.outputs[name] = QPointF(width, y)
            else:
                self.inputs[name] = QPointF(0.0, y)
        self.update()

    def interactive_at(self, scene_pos: QPointF) -> bool:
        """这一点是不是落在节点里的数值框、下拉、色带这类控件上（是的话鼠标交给控件）。"""
        if self.proxy is None or self.body is None:
            return False
        local = self.proxy.mapFromScene(scene_pos)
        child = self.body.childAt(local.toPoint())
        while child is not None and child is not self.body:
            if isinstance(child, _RowLabel):
                return False
            if child.metaObject().className() not in ("QWidget", "_Block"):
                return True
            child = child.parentWidget()
        return False

    # ---- 画 ----
    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        node = self.node
        node_type = node.type
        width, height = self.width(), self._height
        header = _px(HEADER)
        radius = _px(5)
        rect = QRectF(0, 0, width, height)
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillPath(path, QColor(48, 48, 48, 240) if node.type_id != "REROUTE" else QColor(0, 0, 0, 0))
        if node.type_id != "REROUTE":
            head = QPainterPath()
            head.addRoundedRect(QRectF(0, 0, width, header), radius, radius)
            if not node.hide:
                head.addRect(QRectF(0, header * 0.5, width, header * 0.5))
            color = QColor(shader_nodes.CATEGORY_COLORS.get(node_type.category, "#444444"))
            if node.mute:
                color = QColor(90, 90, 90)
            painter.fillPath(head.simplified(), color)
            painter.setPen(QColor(235, 235, 235) if not node.mute else QColor(160, 160, 160))
            font = theme.font()
            painter.setFont(font)
            arrow = "▸" if node.hide else "▾"
            painter.drawText(QRectF(_px(6), 0, width - _px(12), header), Qt.AlignVCenter | Qt.AlignLeft,
                             "%s  %s" % (arrow, node.label))
            if node.mute:
                painter.setPen(QPen(QColor(200, 60, 60, 200), 2.0))
                painter.drawLine(QPointF(_px(4), header - 2), QPointF(width - _px(4), header - 2))
        # 选中描边：当前节点白、其余选中橙
        graph = self.editor.graph()
        if self.isSelected():
            active = graph is not None and graph.active_uid == node.uid
            painter.setPen(QPen(QColor(255, 255, 255) if active else QColor(237, 87, 0), 1.6))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        elif node.type_id != "REROUTE":
            painter.setPen(QPen(QColor(20, 20, 20), 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        # 插口
        r = _px(SOCKET_R)
        for name, point in self.outputs.items():
            self._socket(painter, point, node.output_type(name), r)
        for name, point in self.inputs.items():
            kind = node.input_type(name)
            if kind == "ANY" and graph is not None:
                link = graph.input_link(node.uid, name)
                src = graph.nodes.get(link[0]) if link else None
                kind = src.output_type(link[1]) if src is not None else "FLOAT"
            self._socket(painter, point, kind, r)

    @staticmethod
    def _socket(painter: QPainter, center: QPointF, kind: str, r: float) -> None:
        painter.setPen(QPen(QColor(15, 15, 15), 1.0))
        painter.setBrush(QColor(shader_nodes.SOCKET_COLORS.get(kind, "#a1a1a1")))
        if kind == "VECTOR":
            # 矢量插口是菱形（和 Blender 一样）
            path = QPainterPath(center + QPointF(0, -r))
            path.lineTo(center + QPointF(r, 0))
            path.lineTo(center + QPointF(0, r))
            path.lineTo(center + QPointF(-r, 0))
            path.closeSubpath()
            painter.drawPath(path)
        else:
            painter.drawEllipse(center, r, r)


# ====================================================================== 场景：网格背景、连线
class ShaderScene(QGraphicsScene):
    def __init__(self, editor: "ShaderEditor") -> None:
        super().__init__()
        self.editor = editor
        self.drag_line: tuple | None = None          # (起点, 终点, 颜色)
        self.cut_line: list | None = None             # 割线（场景坐标点列）
        self.setSceneRect(-200000, -200000, 400000, 400000)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.fillRect(rect, QColor(29, 29, 29))
        view = self.editor.view
        scale = view.transform().m11() if view is not None else 1.0
        for step, color in ((20.0, QColor(38, 38, 38)), (100.0, QColor(46, 46, 46))):
            if step * scale < 8:
                continue
            painter.setPen(QPen(color, 0))
            x = math.floor(rect.left() / step) * step
            while x < rect.right():
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                x += step
            y = math.floor(rect.top() / step) * step
            while y < rect.bottom():
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
                y += step
        painter.setRenderHint(QPainter.Antialiasing, True)
        graph = self.editor.graph()
        if graph is None:
            return
        for link in graph.links:
            curve = self.editor.link_curve(link)
            if curve is None:
                continue
            a, b, color, muted = curve
            self._curve(painter, a, b, QColor(color) if not muted else QColor(150, 60, 60), 2.0)
        if self.drag_line is not None:
            start, end, color = self.drag_line
            self._curve(painter, start, end, QColor(color), 1.6, dashed=True)
        if self.cut_line:
            pen = QPen(QColor(255, 80, 80), 1.5)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            for p, q in zip(self.cut_line, self.cut_line[1:]):
                painter.drawLine(p, q)

    @staticmethod
    def bezier(a: QPointF, b: QPointF) -> QPainterPath:
        dx = max(30.0, abs(b.x() - a.x()) * 0.5)
        path = QPainterPath(a)
        path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
        return path

    @classmethod
    def _curve(cls, painter: QPainter, a: QPointF, b: QPointF, color: QColor, width: float,
               dashed: bool = False) -> None:
        painter.setPen(QPen(QColor(10, 10, 10, 160), width + 2.0))
        painter.setBrush(Qt.NoBrush)
        path = cls.bezier(a, b)
        if not dashed:
            painter.drawPath(path)
        pen = QPen(color, width)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.drawPath(path)


# ====================================================================== 视图：鼠标
class ShaderView(QGraphicsView):
    def __init__(self, editor: "ShaderEditor", scene: ShaderScene) -> None:
        super().__init__(scene)
        self.editor = editor
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self._pan = None
        self._link: dict | None = None
        self._move: dict | None = None
        self._band = False
        self._cut: list | None = None
        self._passthrough = False
        self.grabbing: dict | None = None             # G 移动、加节点后跟着鼠标走

    # ---- 命中 ----
    def hit(self, scene_pos: QPointF):
        """插口优先，其次节点。返回 (种类, 节点方块, 插口名) 或 None。"""
        best = None
        tolerance = _px(SOCKET_R) + 5.0
        for item in sorted(self.editor.items.values(), key=lambda it: -it.zValue()):
            p = item.mapFromScene(scene_pos)
            for name, point in item.outputs.items():
                d = p - point
                if abs(d.x()) <= tolerance and abs(d.y()) <= tolerance:
                    return ("output", item, name)
            for name, point in item.inputs.items():
                d = p - point
                if abs(d.x()) <= tolerance and abs(d.y()) <= tolerance:
                    return ("input", item, name)
            if best is None and QRectF(0, 0, item.width(), item.height()).contains(p):
                best = ("node", item, "")
        return best

    def scene_pos(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        editor = self.editor
        graph = editor.graph()
        pos = self.scene_pos(event)
        self._passthrough = False
        if self.grabbing is not None:
            if event.button() == Qt.LeftButton:
                editor.end_grab(True)
            elif event.button() == Qt.RightButton:
                editor.end_grab(False)
            return
        if event.button() == Qt.MiddleButton:
            self._pan = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if graph is None:
            return
        if event.button() == Qt.RightButton:
            if event.modifiers() & Qt.ControlModifier:
                self._cut = [pos]
                return
            # 右键：节点菜单（和 Blender 一样，点在节点上先选中它）
            hit = self.hit(pos)
            if hit is not None and hit[0] == "node" and not hit[1].isSelected():
                editor.select_only([hit[1].node.uid])
            from ..core import ops

            ops.call("wm.call_menu", editor.context(), name="SHADER_MT_context_menu")
            return
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        hit = self.hit(pos)
        if hit is not None and hit[0] == "node" and hit[1].interactive_at(pos):
            # 点在节点里的控件上：交给控件
            self._passthrough = True
            graph.set_active(hit[1].node.uid)
            super().mousePressEvent(event)
            return
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if hit is None:
            if not shift:
                self.scene().clearSelection()
            self._band = True
            self.setDragMode(QGraphicsView.RubberBandDrag)
            super().mousePressEvent(event)
            return
        kind, item, name = hit
        if kind == "output":
            self._link = {"source": (item.node.uid, name), "reverse": False, "before": graph.to_dict(),
                          "start": item.mapToScene(item.outputs[name]),
                          "color": shader_nodes.SOCKET_COLORS.get(item.node.output_type(name), "#a1a1a1")}
            return
        if kind == "input":
            before = graph.to_dict()
            src = graph.input_link(item.node.uid, name)
            if src is not None and src[0] in editor.items:
                # 把线拔下来：拿着上游接着拖，放到别的输入口就改接，放到空白处就是断开
                graph.unlink(item.node.uid, name)
                source = editor.items[src[0]]
                self._link = {"source": src, "reverse": False, "before": before,
                              "start": source.mapToScene(source.outputs.get(src[1], QPointF())),
                              "color": shader_nodes.SOCKET_COLORS.get(source.node.output_type(src[1]), "#a1a1a1")}
            else:
                self._link = {"target": (item.node.uid, name), "reverse": True, "before": before,
                              "start": item.mapToScene(item.inputs[name]),
                              "color": shader_nodes.SOCKET_COLORS.get(item.node.input_type(name), "#a1a1a1")}
            return
        uid = item.node.uid
        if shift:
            if item.isSelected() and graph.active_uid == uid:
                item.setSelected(False)
            else:
                item.setSelected(True)
                graph.set_active(uid)
        else:
            if not item.isSelected():
                editor.select_only([uid])
            graph.set_active(uid)
        selected = [it for it in editor.items.values() if it.isSelected()] or [item]
        for it in selected:
            it.setZValue(2.0)
        self._move = {"start": pos, "items": {it: QPointF(it.pos()) for it in selected}, "moved": False,
                      "before": graph.to_dict()}
        self.viewport().update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._passthrough:
            super().mouseMoveEvent(event)
            return
        if self._pan is not None:
            delta = event.position() - self._pan
            self._pan = event.position()
            scale = self.transform().m11() or 1.0
            center = self.mapToScene(self.viewport().rect().center())
            self.centerOn(center - QPointF(delta.x(), delta.y()) / scale)
            return
        pos = self.scene_pos(event)
        if self.grabbing is not None:
            self.editor.update_grab(pos)
            return
        if self._cut is not None:
            self._cut.append(pos)
            self.scene().cut_line = list(self._cut)
            self.viewport().update()
            return
        if self._link is not None:
            self.scene().drag_line = ((pos, self._link["start"], self._link["color"]) if self._link["reverse"]
                                      else (self._link["start"], pos, self._link["color"]))
            self.viewport().update()
            return
        if self._move is not None:
            delta = pos - self._move["start"]
            if abs(delta.x()) + abs(delta.y()) > 0.5:
                self._move["moved"] = True
            for item, origin in self._move["items"].items():
                item.setPos(origin + delta)
            self.viewport().update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        editor = self.editor
        graph = editor.graph()
        if self._passthrough:
            self._passthrough = False
            super().mouseReleaseEvent(event)
            return
        if self._pan is not None and event.button() == Qt.MiddleButton:
            self._pan = None
            self.unsetCursor()
            return
        if self._cut is not None and event.button() == Qt.RightButton:
            points, self._cut = self._cut, None
            self.scene().cut_line = None
            if graph is not None and len(points) > 1:
                editor.cut_links(points)
            self.viewport().update()
            return
        if self._band:
            super().mouseReleaseEvent(event)
            self.setDragMode(QGraphicsView.NoDrag)
            self._band = False
            editor.on_selection()
            return
        pos = self.scene_pos(event)
        if self._link is not None and graph is not None:
            link, self._link = self._link, None
            self.scene().drag_line = None
            hit = self.hit(pos)
            if not link["reverse"] and hit is not None and hit[0] == "input":
                if not graph.link(link["source"][0], link["source"][1], hit[1].node.uid, hit[2]):
                    editor.app.report("这样连会绕成一个圈，接不上", "WARNING")
            elif link["reverse"] and hit is not None and hit[0] == "output":
                if not graph.link(hit[1].node.uid, hit[2], link["target"][0], link["target"][1]):
                    editor.app.report("这样连会绕成一个圈，接不上", "WARNING")
            editor.record("连线", link["before"])
            self.viewport().update()
            return
        if self._move is not None and graph is not None:
            move, self._move = self._move, None
            for item in move["items"]:
                item.setZValue(1.0)
            if move["moved"]:
                for item in move["items"]:
                    graph.move_node(item.node.uid, (item.pos().x(), item.pos().y()))
                editor.record("移动节点", move["before"])
            editor.on_selection()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._passthrough:
            super().wheelEvent(event)
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 1.15 ** steps
        scale = self.transform().m11()
        factor = max(0.2 / scale, min(3.0 / scale, factor))
        self.scale(factor, factor)


# ====================================================================== 编辑器
@registry.register_editor
class ShaderEditor(Editor):
    idname = "SHADER_EDITOR"
    label = "着色器编辑器"
    icon = "editor.shader"
    category = "通用"
    order = 15
    description = "当前纹理集的材质节点：图层堆栈是其中一个节点，接上纹理、颜色渐变、运算再输出"
    keymaps = ["Shader Editor"]
    has_sidebar = True

    def __init__(self, app, area=None) -> None:
        self.items: dict[int, ShaderNodeItem] = {}
        self.view: ShaderView | None = None
        self.scene: ShaderScene | None = None
        self._bound_graph = None
        self._bound_project = None
        self._framed_graph = None
        self.clipboard: dict | None = None
        super().__init__(app, area)

    # ---- 主区域 ----
    def build_main(self) -> QWidget:
        self.scene = ShaderScene(self)
        self.view = ShaderView(self, self.scene)
        self.scene.selectionChanged.connect(self.on_selection)
        QTimer.singleShot(0, self._sync)
        return self.view

    def texture_set(self):
        project = self.app.project
        return project.active_texture_set if project is not None else None

    def graph(self):
        ts = self.texture_set()
        return getattr(ts, "material", None) if ts is not None else None

    # ---- 和图同步 ----
    def _bind(self, graph) -> None:
        if graph is self._bound_graph:
            return
        old = self._bound_graph
        if old is not None:
            try:
                old.changed.disconnect(self._on_graph_changed)
            except Exception:  # noqa: BLE001
                pass
        self._bound_graph = graph
        if graph is not None:
            graph.changed.connect(self._on_graph_changed)

    def _sync(self) -> None:
        if self.scene is None or not shiboken6.isValid(self.scene):
            return
        graph = self.graph()
        self._bind(graph)
        current = set(graph.nodes) if graph is not None else set()
        for uid in list(self.items):
            if uid not in current:
                item = self.items.pop(uid)
                self.scene.removeItem(item)
        if graph is not None:
            for uid, node in graph.nodes.items():
                item = self.items.get(uid)
                if item is None or item.node is not node:
                    if item is not None:
                        self.scene.removeItem(item)
                    item = ShaderNodeItem(self, node)
                    self.scene.addItem(item)
                    self.items[uid] = item
                else:
                    if item.signature() != item._signature:
                        item.rebuild()
                    item.setPos(QPointF(*node.location))
        if graph is not None and graph is not self._framed_graph and self.view is not None:
            self._framed_graph = graph
            QTimer.singleShot(0, self.frame_all)
        self.scene.update()

    def _on_graph_changed(self, kind: str, node) -> None:
        if kind in ("structure", "layout"):
            self._sync()
        elif kind == "active" and self.scene is not None:
            self.scene.update()
            if self.sidebar is not None and self.sidebar_visible:
                self.sidebar.refresh()

    def refresh(self) -> None:
        super().refresh()
        if self.app.project is not self._bound_project:
            self._bound_project = self.app.project
        self._sync()

    # ---- 选择 ----
    def selected_uids(self) -> list[int]:
        return [uid for uid, item in self.items.items() if item.isSelected()]

    def select_only(self, uids) -> None:
        if self.scene is None:
            return
        wanted = set(uids)
        blocked = self.scene.blockSignals(True)
        for uid, item in self.items.items():
            item.setSelected(uid in wanted)
        self.scene.blockSignals(blocked)
        self.on_selection()

    def on_selection(self) -> None:
        graph = self.graph()
        if graph is None:
            return
        chosen = self.selected_uids()
        if chosen and graph.active_uid not in chosen:
            graph.set_active(chosen[-1])
        if self.scene is not None:
            self.scene.update()

    # ---- 视图 ----
    def frame_all(self, only_selected: bool = False) -> None:
        if self.view is None or not self.items:
            return
        items = [it for it in self.items.values() if it.isSelected()] if only_selected else list(self.items.values())
        if not items:
            items = list(self.items.values())
        rect = QRectF()
        for item in items:
            rect = rect.united(item.mapRectToScene(item.boundingRect()))
        rect.adjust(-60, -60, 60, 60)
        self.view.fitInView(rect, Qt.KeepAspectRatio)
        scale = self.view.transform().m11()
        if scale > 1.0:
            self.view.scale(1.0 / scale, 1.0 / scale)

    def cursor_scene_pos(self) -> QPointF:
        if self.view is None:
            return QPointF()
        local = self.view.mapFromGlobal(QCursor.pos())
        if not self.view.rect().contains(local):
            local = self.view.viewport().rect().center()
        return self.view.mapToScene(local)

    # ---- 撤销 ----
    def record(self, label: str, before: dict) -> None:
        graph = self.graph()
        history = getattr(self.app, "history", None)
        if graph is None or history is None:
            return
        after = graph.to_dict()
        if after == before:
            return
        target = graph

        def undo() -> None:
            target.load(before)

        def redo() -> None:
            target.load(after)

        history.push(label, undo, redo)

    # ---- 连线 ----
    def link_curve(self, link):
        a, sa, b, sb = link
        ia, ib = self.items.get(a), self.items.get(b)
        if ia is None or ib is None or sa not in ia.outputs or sb not in ib.inputs:
            return None
        color = shader_nodes.SOCKET_COLORS.get(ia.node.output_type(sa), "#a1a1a1")
        return ia.mapToScene(ia.outputs[sa]), ib.mapToScene(ib.inputs[sb]), color, ia.node.mute or ib.node.mute

    def cut_links(self, points) -> None:
        """Ctrl+右键拖过的线都割断。"""
        graph = self.graph()
        if graph is None:
            return
        cut = []
        for link in graph.links:
            curve = self.link_curve(link)
            if curve is None:
                continue
            path = ShaderScene.bezier(curve[0], curve[1])
            samples = [path.pointAtPercent(t / 24.0) for t in range(25)]
            if _polyline_cross(samples, points):
                cut.append(link)
        if cut:
            before = graph.to_dict()
            graph.remove_links(cut)
            self.record("割断连线", before)

    # ---- 跟着鼠标走（G、加节点、复制）----
    def start_grab(self, uids, label: str, before: dict | None = None) -> None:
        if self.view is None:
            return
        graph = self.graph()
        items = [self.items[uid] for uid in uids if uid in self.items]
        if not items or graph is None:
            return
        self.view.grabbing = {"start": self.cursor_scene_pos(), "items": {it: QPointF(it.pos()) for it in items},
                          "label": label, "before": before if before is not None else graph.to_dict()}
        self.view.setFocus()

    def update_grab(self, pos: QPointF) -> None:
        grab = self.view.grabbing
        delta = pos - grab["start"]
        for item, origin in grab["items"].items():
            item.setPos(origin + delta)
        self.view.viewport().update()

    def end_grab(self, confirm: bool) -> None:
        grab, self.view.grabbing = self.view.grabbing, None
        graph = self.graph()
        if grab is None or graph is None:
            return
        if confirm:
            for item in grab["items"]:
                graph.move_node(item.node.uid, (item.pos().x(), item.pos().y()))
            self.record(grab["label"], grab["before"])
        else:
            if grab["label"] == "移动节点":
                for item, origin in grab["items"].items():
                    item.setPos(origin)
            else:
                # 加节点、复制时取消：连同新节点一起撤回
                graph.load(grab["before"])
        self.view.viewport().update()

    def handle_event(self, event) -> bool:
        if self.view is not None and self.view.grabbing is not None and event.value == "PRESS":
            if event.type in ("RET", "NUMPAD_ENTER", "SPACE"):
                self.end_grab(True)
                return True
            if event.type == "ESC":
                self.end_grab(False)
                return True
        return super().handle_event(event)

    # ---- 标题栏 ----
    def draw_header(self, layout, ctx) -> None:
        project = self.app.project
        ts = self.texture_set()
        layout.menu("SHADER_MT_view", text="视图")
        layout.menu("SHADER_MT_select", text="选择")
        layout.menu("SHADER_MT_add", text="添加")
        layout.menu("SHADER_MT_node", text="节点")
        layout.separator()
        if project is not None and project.texture_sets:
            layout.menu("SHADER_MT_texture_sets", text="材质：%s" % (ts.name if ts is not None else "无"),
                        icon="texture_set")
        if ts is not None:
            if ts.material is None:
                layout.operator("shader.use_nodes", text="使用节点", icon="editor.shader")
            else:
                layout.operator("shader.use_nodes", text="停用节点", icon="editor.shader", enable=False)
        layout.stretch()

    def save_state(self) -> dict:
        state = super().save_state()
        if self.view is not None:
            center = self.view.mapToScene(self.view.viewport().rect().center())
            state["center"] = [center.x(), center.y()]
            state["zoom"] = self.view.transform().m11()
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x())

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def _polyline_cross(a: list, b: list) -> bool:
    for p1, p2 in zip(a, a[1:]):
        for q1, q2 in zip(b, b[1:]):
            if _segments_cross(p1, p2, q1, q2):
                return True
    return False


# ====================================================================== 侧栏：当前节点
def _editor(ctx):
    editor = getattr(ctx, "editor", None)
    return editor if isinstance(editor, ShaderEditor) else None


@registry.register_panel
class ShaderNodePanel(registry.Panel):
    idname = "SHADER_PT_active_node"
    label = "节点"
    space = "SHADER_EDITOR"
    region = "SIDEBAR"

    @classmethod
    def poll(cls, ctx) -> bool:
        editor = _editor(ctx)
        graph = editor.graph() if editor is not None else None
        return graph is not None and graph.active is not None

    def draw(self, layout, ctx) -> None:
        graph = _editor(ctx).graph()
        node = graph.active
        layout.label(node.type.label, role="title")
        if node.type.description:
            layout.label(node.type.description, role="dim")
        for name, prop in type(node.values).properties().items():
            layout.prop(node.values, name, slider=getattr(prop, "subtype", "") == "FACTOR")
