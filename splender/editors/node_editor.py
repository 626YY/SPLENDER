"""节点编辑器：把节点图画成方块和连线。

- 左键从输出口拖到输入口连线；拖一个接着线的输入口能把线拔下来改接别处，拖到空白处就是断开；
- 左键拖节点挪位置，拖空白处框选，Ctrl 点选加减选择；
- 右键空白处或 Shift+A 加节点，右键节点删除、复制；X / Delete 删除，Ctrl+D 复制，Home 看全部；
- 滚轮缩放，中键拖动平移；侧栏是当前节点的参数和大预览。
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene, QGraphicsView, QMenu, QWidget

from ..core import ops, registry
from ..nodes import library
from ..ui import theme
from ..ui.editor import Editor

log = logging.getLogger("splender.nodes")

NODE_W = 150
HEADER_H = 22
THUMB = 104
ROW_H = 18
SOCKET_R = 5.5
CATEGORY_COLORS = {"GENERATE": "#3e6b4c", "ADJUST": "#5b4d86", "FILTER": "#2d6380", "OUTPUT": "#8a5228"}
KIND_COLORS = {"gray": "#a8a8ad", "color": "#e0b341"}


def _qimage(array) -> QImage:
    """(h, w, 4) uint8 → QImage（复制一份，QImage 不持有 numpy 内存）。"""
    height, width = array.shape[:2]
    return QImage(array.tobytes(), width, height, width * 4, QImage.Format_RGBA8888).copy()


# ====================================================================== 节点方块
class NodeItem(QGraphicsItem):
    def __init__(self, editor: "NodeEditor", node) -> None:
        super().__init__()
        self.editor = editor
        self.node = node
        self.image: QImage | None = None
        self.sig = ""
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(1.0)
        self.setPos(QPointF(*node.location))

    @property
    def inputs(self) -> tuple:
        return self.node.type.inputs

    def has_output(self) -> bool:
        return self.node.type.output_kind != "none"

    def output_kind(self) -> str:
        kind = self.node.type.output_kind
        if kind == "input":
            graph = self.editor.graph()
            src = graph.input_link(self.node.uid, 0) if graph is not None else None
            item = self.editor.items.get(src) if src is not None else None
            return item.output_kind() if item is not None and item is not self else "gray"
        return kind if kind in KIND_COLORS else "gray"

    def height(self) -> float:
        rows = len(self.inputs)
        return HEADER_H + THUMB + 8 + rows * ROW_H + (6 if rows else 0)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-SOCKET_R - 3, -3, NODE_W + 2 * SOCKET_R + 6, self.height() + 6)

    def input_point(self, index: int) -> QPointF:
        return QPointF(0.0, HEADER_H + THUMB + 8 + index * ROW_H + ROW_H * 0.5)

    def output_point(self) -> QPointF:
        return QPointF(float(NODE_W), HEADER_H + 4 + THUMB * 0.5)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ARG002
        graph = self.editor.graph()
        active = graph is not None and graph.active_uid == self.node.uid
        painter.setRenderHint(QPainter.Antialiasing, True)
        body = QRectF(0, 0, NODE_W, self.height())
        path = QPainterPath()
        path.addRoundedRect(body, 6, 6)
        painter.fillPath(path, theme.qcolor("bg.panel"))
        painter.save()
        painter.setClipRect(QRectF(0, 0, NODE_W, HEADER_H))
        painter.fillPath(path, QColor(CATEGORY_COLORS.get(self.node.type.category, "#555")))
        painter.restore()
        painter.setPen(theme.qcolor("text"))
        painter.setFont(theme.font("font.small", bold=True))
        painter.drawText(QRectF(8, 0, NODE_W - 16, HEADER_H), Qt.AlignVCenter | Qt.AlignLeft, self.node.label)
        thumb = QRectF((NODE_W - THUMB) / 2, HEADER_H + 4, THUMB, THUMB)
        if self.image is not None:
            painter.drawImage(thumb, self.image)
        else:
            painter.fillRect(thumb, theme.qcolor("bg.field"))
        painter.setFont(theme.font("font.small"))
        for index, (_name, label, _default, gray) in enumerate(self.inputs):
            y = HEADER_H + THUMB + 8 + index * ROW_H
            painter.setPen(theme.qcolor("text.dim"))
            painter.drawText(QRectF(12, y, NODE_W - 16, ROW_H), Qt.AlignVCenter | Qt.AlignLeft, label)
            linked = graph is not None and graph.input_link(self.node.uid, index) is not None
            self._socket(painter, self.input_point(index), KIND_COLORS["gray" if gray else "color"], linked)
        if self.has_output():
            self._socket(painter, self.output_point(), KIND_COLORS[self.output_kind()], True)
        if active or self.isSelected():
            painter.setPen(QPen(theme.qcolor("accent") if active else theme.qcolor("accent.hover"),
                                2.0 if active else 1.4))
        else:
            painter.setPen(QPen(theme.qcolor("line.soft"), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    @staticmethod
    def _socket(painter: QPainter, center: QPointF, color: str, filled: bool) -> None:
        painter.setPen(QPen(QColor(color), 1.6))
        painter.setBrush(QColor(color) if filled else theme.qcolor("bg.panel"))
        painter.drawEllipse(center, SOCKET_R, SOCKET_R)


# ====================================================================== 场景：网格、连线
class NodeScene(QGraphicsScene):
    def __init__(self, editor: "NodeEditor") -> None:
        super().__init__()
        self.editor = editor
        self.drag_line: tuple | None = None          # (起点, 终点, 颜色)
        self.setSceneRect(-200000, -200000, 400000, 400000)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.fillRect(rect, theme.qcolor("bg.area"))
        view = self.editor.view
        scale = view.transform().m11() if view is not None else 1.0
        for step, color in ((20.0, theme.qcolor("bg.row_alt")), (100.0, theme.qcolor("line.soft", 0.35))):
            if step * scale < 8:
                continue
            painter.setPen(QPen(color, 0))
            x = int(rect.left() // step) * step
            while x < rect.right():
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
                x += step
            y = int(rect.top() // step) * step
            while y < rect.bottom():
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
                y += step
        painter.setRenderHint(QPainter.Antialiasing, True)
        graph = self.editor.graph()
        if graph is not None:
            items = self.editor.items
            for src, dst, slot in graph.links:
                a, b = items.get(src), items.get(dst)
                if a is None or b is None or slot >= len(b.inputs):
                    continue
                gray = b.inputs[slot][3]
                color = KIND_COLORS["gray" if gray else a.output_kind()]
                self._curve(painter, a.mapToScene(a.output_point()), b.mapToScene(b.input_point(slot)), color, 2.0)
        if self.drag_line is not None:
            start, end, color = self.drag_line
            self._curve(painter, start, end, color, 1.6, dashed=True)

    @staticmethod
    def _curve(painter: QPainter, a: QPointF, b: QPointF, color: str, width: float, dashed: bool = False) -> None:
        dx = max(40.0, abs(b.x() - a.x()) * 0.5)
        path = QPainterPath(a)
        path.cubicTo(QPointF(a.x() + dx, a.y()), QPointF(b.x() - dx, b.y()), b)
        pen = QPen(QColor(color), width)
        if dashed:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)


# ====================================================================== 视图：鼠标交互
class NodeView(QGraphicsView):
    def __init__(self, editor: "NodeEditor", scene: NodeScene) -> None:
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

    # ---- 命中 ----
    def hit(self, scene_pos: QPointF):
        """插口优先，其次节点。返回 (种类, 节点方块, 输入序号) 或 None。"""
        best = None
        for item in reversed(list(self.editor.items.values())):
            p = item.mapFromScene(scene_pos)
            if item.has_output():
                d = p - item.output_point()
                if abs(d.x()) < 12 and abs(d.y()) < 12:
                    return ("output", item, -1)
            for index in range(len(item.inputs)):
                d = p - item.input_point(index)
                if -12 < d.x() < 40 and abs(d.y()) < ROW_H * 0.5:
                    return ("input", item, index)
            if best is None and QRectF(0, 0, NODE_W, item.height()).contains(p):
                best = ("node", item, -1)
        return best

    def _scene_pos(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    # ---- 按下 ----
    def mousePressEvent(self, event) -> None:  # noqa: N802
        editor = self.editor
        graph = editor.graph()
        pos = self._scene_pos(event)
        if event.button() == Qt.MiddleButton:
            self._pan = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if graph is None:
            return
        if event.button() == Qt.RightButton:
            hit = self.hit(pos)
            if hit is not None and hit[0] == "node":
                item = hit[1]
                if not item.isSelected():
                    editor.select_only([item.node.uid])
                graph.set_active(item.node.uid)
                editor.open_node_menu(event.globalPosition().toPoint())
            else:
                editor.open_add_menu(pos, event.globalPosition().toPoint())
            return
        if event.button() != Qt.LeftButton:
            return
        hit = self.hit(pos)
        ctrl = bool(event.modifiers() & Qt.ControlModifier)
        if hit is None:
            if not ctrl:
                self.scene().clearSelection()
            self._band = True
            self.setDragMode(QGraphicsView.RubberBandDrag)
            super().mousePressEvent(event)
            return
        kind, item, index = hit
        if kind == "output":
            self._link = {"source": item.node.uid, "reverse": False, "before": graph.to_dict(),
                          "start": item.mapToScene(item.output_point()), "color": KIND_COLORS[item.output_kind()]}
            return
        if kind == "input":
            before = graph.to_dict()
            src = graph.input_link(item.node.uid, index)
            gray = item.inputs[index][3]
            if src is not None and src in editor.items:
                # 把线拔下来：拿着上游接着拖，放到别的输入口就改接，放到空白处就是断开
                graph.unlink(item.node.uid, index)
                source = editor.items[src]
                self._link = {"source": src, "reverse": False, "before": before,
                              "start": source.mapToScene(source.output_point()),
                              "color": KIND_COLORS["gray" if gray else source.output_kind()]}
            else:
                self._link = {"target": (item.node.uid, index), "reverse": True, "before": before,
                              "start": item.mapToScene(item.input_point(index)),
                              "color": KIND_COLORS["gray" if gray else "color"]}
            return
        uid = item.node.uid
        if ctrl:
            item.setSelected(not item.isSelected())
        elif not item.isSelected():
            editor.select_only([uid])
        graph.set_active(uid)
        selected = [it for it in editor.items.values() if it.isSelected()] or [item]
        self._move = {"start": pos, "items": {it: QPointF(it.pos()) for it in selected}, "moved": False}

    # ---- 移动 ----
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._pan is not None:
            delta = event.position() - self._pan
            self._pan = event.position()
            scale = self.transform().m11() or 1.0
            center = self.mapToScene(self.viewport().rect().center())
            self.centerOn(center - QPointF(delta.x(), delta.y()) / scale)
            return
        pos = self._scene_pos(event)
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
                item.node.location = (item.pos().x(), item.pos().y())
            self.viewport().update()
            return
        super().mouseMoveEvent(event)

    # ---- 松开 ----
    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        editor = self.editor
        graph = editor.graph()
        if self._pan is not None and event.button() == Qt.MiddleButton:
            self._pan = None
            self.unsetCursor()
            return
        if self._band:
            super().mouseReleaseEvent(event)
            self.setDragMode(QGraphicsView.NoDrag)
            self._band = False
            return
        pos = self._scene_pos(event)
        if self._link is not None and graph is not None:
            link, self._link = self._link, None
            self.scene().drag_line = None
            hit = self.hit(pos)
            refused = False
            if not link["reverse"] and hit is not None and hit[0] == "input":
                refused = not graph.link(link["source"], hit[1].node.uid, hit[2])
            elif link["reverse"] and hit is not None and hit[0] == "output":
                refused = not graph.link(hit[1].node.uid, link["target"][0], link["target"][1])
            if refused:
                editor.app.report("这样连会绕成一个圈，或者接不上", "WARNING")
            editor.record("连线", graph, link["before"])
            self.viewport().update()
            return
        if self._move is not None and graph is not None:
            move, self._move = self._move, None
            if move["moved"]:
                before = graph.to_dict()
                for item, origin in move["items"].items():
                    before_node = next((n for n in before["nodes"] if n["uid"] == item.node.uid), None)
                    if before_node is not None:
                        before_node["location"] = [origin.x(), origin.y()]
                for item in move["items"]:
                    graph.move_node(item.node.uid, (item.pos().x(), item.pos().y()))
                editor.record("移动节点", graph, before)
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        hit = self.hit(self._scene_pos(event))
        graph = self.editor.graph()
        if hit is not None and hit[0] == "node" and graph is not None:
            graph.set_active(hit[1].node.uid)
            self.editor.show_sidebar()

    def wheelEvent(self, event) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 1.15 ** steps
        scale = self.transform().m11()
        factor = max(0.15 / scale, min(3.0 / scale, factor))
        self.scale(factor, factor)


# ====================================================================== 编辑器
@registry.register_editor
class NodeEditor(Editor):
    idname = "NODE_EDITOR"
    label = "节点"
    icon = "editor.nodes"
    category = "通用"
    order = 16
    description = "用节点拼出程序化材质：噪波、砖块、细胞、划痕……可以平铺到填充层上"
    keymaps = ["Node"]
    has_sidebar = True

    def __init__(self, app, area=None) -> None:
        self.graph_uid = 0
        self.items: dict[int, NodeItem] = {}
        self.view: NodeView | None = None
        self.scene: NodeScene | None = None
        self._bound_graph = None
        self._bound_project = None
        self._framed = False
        self.preview: QImage | None = None
        self._preview_sig = ""
        super().__init__(app, area)

    # ---- 主区域 ----
    def build_main(self) -> QWidget:
        self.scene = NodeScene(self)
        self.view = NodeView(self, self.scene)
        self.timer = QTimer(self.view)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        QTimer.singleShot(0, self._sync)
        return self.view

    def graph(self):
        project = self.app.project
        if project is None:
            return None
        found = project.graph(self.graph_uid) if self.graph_uid else None
        return found or project.active_graph

    def show_graph(self, uid: int) -> None:
        self.graph_uid = uid
        self._sync()
        self.frame_all()
        self.refresh_header()

    def show_sidebar(self) -> None:
        if not self.sidebar_visible:
            self.sidebar_visible = True
        if self.sidebar is not None:
            self.sidebar.refresh()

    # ---- 和图同步 ----
    def _bind(self, graph) -> None:
        if graph is self._bound_graph:
            return
        if self._bound_graph is not None:
            try:
                self._bound_graph.changed.disconnect(self._on_graph_changed)
            except Exception:  # noqa: BLE001
                pass
        self._bound_graph = graph
        if graph is not None:
            graph.changed.connect(self._on_graph_changed)
        self._framed = False

    def _sync(self) -> None:
        """方块和图里的节点对齐：删掉多的、补上缺的、挪到节点的位置。"""
        if self.scene is None:
            return
        graph = self.graph()
        self._bind(graph)
        nodes = graph.nodes if graph is not None else {}
        for uid in [uid for uid in self.items if uid not in nodes or self.items[uid].node is not nodes[uid]]:
            item = self.items.pop(uid)
            self.scene.removeItem(item)
        for uid, node in nodes.items():
            item = self.items.get(uid)
            if item is None:
                item = NodeItem(self, node)
                self.items[uid] = item
                self.scene.addItem(item)
            elif (item.pos().x(), item.pos().y()) != node.location:
                item.setPos(QPointF(*node.location))
        if graph is not None and not self._framed and self.view is not None and self.view.width() > 50:
            self._framed = True
            self.frame_all()
        self.scene.update()

    def _on_graph_changed(self, kind: str, node) -> None:
        if kind in ("structure", "layout"):
            self._sync()
        if kind in ("active", "structure", "settings"):
            if self.sidebar is not None and self.sidebar_visible:
                self.sidebar.refresh()
            self.refresh_header()
        if self.scene is not None:
            self.scene.update()

    def _tick(self) -> None:
        """隔一会儿看看：工程换了没有；哪些缩略图过期了（每次最多重画几张，不卡界面）。"""
        project = self.app.project
        if project is not self._bound_project:
            self._bound_project = project
            self.graph_uid = 0
            self._sync()
            self.refresh_header()
        graph = self.graph()
        if graph is not self._bound_graph:
            self._sync()
            self.refresh_header()
        if graph is None or self.view is None or not self.view.isVisible():
            return
        engine = self.app.engine
        host = getattr(self.app, "host", None)
        if engine is None or host is None:
            return
        evaluator = engine.node_graphs
        memo: dict = {}
        budget = 6
        try:
            host.make_current()
            for uid, item in self.items.items():
                if budget <= 0:
                    break
                sig = evaluator._signature(graph, uid, memo)
                if sig == item.sig:
                    continue
                data = evaluator.thumbnail(graph, uid, THUMB)
                item.image = _qimage(data) if data is not None else None
                item.sig = sig
                item.update()
                budget -= 1
            active = graph.active
            if active is not None:
                sig = evaluator._signature(graph, active.uid, memo) + "|big"
                if sig != self._preview_sig:
                    data = evaluator.thumbnail(graph, active.uid, 256)
                    self.preview = _qimage(data) if data is not None else None
                    self._preview_sig = sig
                    for widget in self.findChildren(PreviewWidget):
                        widget.update()
        except Exception:  # noqa: BLE001
            log.exception("节点缩略图出错")

    # ---- 选择、位置 ----
    def selected_uids(self) -> list[int]:
        return [uid for uid, item in self.items.items() if item.isSelected()]

    def select_only(self, uids) -> None:
        if self.scene is None:
            return
        self.scene.clearSelection()
        for uid in uids:
            item = self.items.get(uid)
            if item is not None:
                item.setSelected(True)
        graph = self.graph()
        if graph is not None and uids:
            graph.set_active(list(uids)[-1])

    def view_center(self) -> tuple:
        if self.view is None:
            return (0.0, 0.0)
        center = self.view.mapToScene(self.view.viewport().rect().center())
        return (center.x() - NODE_W * 0.5, center.y() - 60.0)

    def cursor_scene_pos(self) -> QPointF | None:
        if self.view is None:
            return None
        local = self.view.viewport().mapFromGlobal(QCursor.pos())
        if not self.view.viewport().rect().contains(local):
            return None
        return self.view.mapToScene(local)

    def frame_all(self) -> None:
        if self.view is None or not self.items:
            return
        rect = QRectF()
        for item in self.items.values():
            rect = rect.united(item.sceneBoundingRect())
        rect = rect.adjusted(-60, -60, 60, 60)
        self.view.fitInView(rect, Qt.KeepAspectRatio)
        scale = self.view.transform().m11()
        if scale > 1.0:
            self.view.scale(1.0 / scale, 1.0 / scale)
        self.view.centerOn(rect.center())

    def record(self, label: str, graph, before: dict) -> None:
        from ..nodes.graph import record

        record(getattr(self.app, "history", None), label, graph, before)

    # ---- 菜单 ----
    def open_add_menu(self, scene_pos: QPointF | None = None, global_pos=None) -> None:
        graph = self.graph()
        if graph is None or self.view is None:
            return
        if scene_pos is None:
            scene_pos = self.cursor_scene_pos()
        if scene_pos is None:
            center = self.view_center()
            scene_pos = QPointF(center[0] + NODE_W * 0.5, center[1] + 20.0)
        menu = QMenu(self.view)
        for key, label in library.CATEGORIES:
            sub = menu.addMenu(label)
            for node_type in library.all_types():
                if node_type.category != key:
                    continue
                action = sub.addAction(node_type.label)
                action.setToolTip(node_type.description)
                action.triggered.connect(lambda _checked=False, t=node_type.id: self.add_node_at(t, scene_pos))
        menu.exec(global_pos or QCursor.pos())

    def add_node_at(self, type_id: str, scene_pos: QPointF):
        graph = self.graph()
        if graph is None:
            return None
        before = graph.to_dict()
        node = graph.add_node(type_id, (scene_pos.x() - NODE_W * 0.5, scene_pos.y() - 20.0))
        self.record("添加节点「%s」" % node.type.label, graph, before)
        self.select_only([node.uid])
        return node

    def open_node_menu(self, global_pos) -> None:
        menu = QMenu(self.view)
        ctx = self.context()
        for idname, text in (("node.duplicate", "复制"), ("node.delete", "删除")):
            action = menu.addAction(text)
            action.triggered.connect(lambda _checked=False, op=idname: ops.call(op, self.context()))
            action.setEnabled(ops.poll(idname, ctx))
        menu.exec(global_pos)

    # ---- 标题栏 ----
    def draw_header(self, layout, ctx) -> None:
        graph = self.graph()
        layout.menu("NODE_MT_graphs", text=graph.name if graph is not None else "没有节点图", icon="editor.nodes")
        layout.operator("node.graph_new", text="", icon="add")
        if graph is not None:
            layout.prop(graph.settings, "resolution", text="")
            layout.operator("node.add_menu", text="添加", icon="add")
        layout.stretch()
        if graph is not None:
            layout.operator("layer.add_graph_fill", text="用作材质层", icon="layer.fill")
        layout.operator("node.view_all", text="", icon="focus")

    def refresh(self) -> None:
        super().refresh()
        if self.graph() is not self._bound_graph:
            self._sync()

    # ---- 状态 ----
    def save_state(self) -> dict:
        state = super().save_state()
        state["graph"] = int(self.graph_uid)
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        if isinstance(state, dict):
            self.graph_uid = int(state.get("graph", 0) or 0)

    def dispose(self) -> None:
        if getattr(self, "timer", None) is not None:
            self.timer.stop()
        self._bind(None)
        super().dispose()


class PreviewWidget(QWidget):
    """侧栏里当前节点的大预览。"""

    def __init__(self, editor: NodeEditor) -> None:
        super().__init__()
        self.editor = editor
        self.setMinimumHeight(theme.px(200))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        side = min(self.width(), self.height())
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        image = self.editor.preview
        if image is None:
            painter.fillRect(rect, theme.qcolor("bg.field"))
        else:
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            painter.drawImage(rect, image)
        painter.end()

    def sizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize
        return QSize(theme.px(220), theme.px(220))


# ====================================================================== 侧栏面板
def _editor(ctx):
    editor = ctx.editor
    return editor if getattr(editor, "idname", "") == "NODE_EDITOR" else None


@registry.register_panel
class NodeParamsPanel(registry.Panel):
    idname = "NODE_PT_params"
    label = "参数"
    space = "NODE_EDITOR"
    category = "节点"
    order = 10

    @classmethod
    def poll(cls, ctx) -> bool:
        editor = _editor(ctx)
        graph = editor.graph() if editor is not None else None
        return graph is not None and graph.active is not None

    def draw(self, layout, ctx) -> None:
        node = _editor(ctx).graph().active
        layout.label(node.type.label, icon="editor.nodes")
        layout.label(node.type.description, role="dim")
        if node.type_id == "OUTPUT":
            col = layout.column()
            col.use_property_split = False
            col.operator("layer.add_graph_fill", text="用作材质层", icon="layer.fill", role="primary")
            return
        for name, prop in node.params.properties().items():
            layout.prop(node.params, name, slider=getattr(prop, "subtype", "") == "FACTOR")


@registry.register_panel
class NodePreviewPanel(registry.Panel):
    idname = "NODE_PT_preview"
    label = "预览"
    space = "NODE_EDITOR"
    category = "节点"
    order = 20

    @classmethod
    def poll(cls, ctx) -> bool:
        return NodeParamsPanel.poll(ctx)

    def draw(self, layout, ctx) -> None:
        layout.widget(PreviewWidget(_editor(ctx)))


@registry.register_panel
class NodeGraphPanel(registry.Panel):
    idname = "NODE_PT_graph"
    label = "节点图"
    space = "NODE_EDITOR"
    category = "图"
    order = 10

    @classmethod
    def poll(cls, ctx) -> bool:
        return _editor(ctx) is not None

    def draw(self, layout, ctx) -> None:
        graph = _editor(ctx).graph()
        if graph is None:
            layout.label("还没有节点图", role="dim")
            col = layout.column()
            col.use_property_split = False
            col.operator("node.graph_new", icon="add", role="primary")
            return
        layout.prop(graph.settings, "name")
        layout.prop(graph.settings, "resolution")
        row = layout.row(heading="节点")
        row.label("%d 个，%d 根连线" % (len(graph.nodes), len(graph.links)), role="dim")
        if graph.output_node() is None:
            layout.label("没有「材质输出」节点，用作材质时没有内容", icon="warning")
        col = layout.column()
        col.use_property_split = False
        col.operator("layer.add_graph_fill", text="用作材质层", icon="layer.fill", role="primary")
        col.operator("node.graph_new", icon="add")
        col.operator("node.graph_delete", icon="remove")
