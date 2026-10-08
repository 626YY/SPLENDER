"""整机自检：节点编辑器。切到「节点」工作区，新建节点图，等缩略图出来；加一个砖块节点，
用鼠标事件从它的输出口拖线到材质输出的「基础色」；用作材质层，视口里应该出现砖纹；改参数视口跟着变；撤销连线。
变量 app 和 d 由自检框架提供。"""
import json
import logging
import time

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

errors = []


class _Catch(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())


logging.getLogger("splender").addHandler(_Catch())
results = {}
d.settle()
d.out.mkdir(parents=True, exist_ok=True)

assert app.window.set_workspace("节点"), "应该有「节点」工作区"
d.settle()
d.wait(300)
editor = d.editor("NODE_EDITOR")
assert editor is not None, "「节点」工作区里应该有节点编辑器"
assert d.call("node.graph_new", editor) == "FINISHED"
d.settle()
graph = editor.graph()
assert graph is not None and len(graph.nodes) == 3, "新图带一个小例子"
t0 = time.perf_counter()
while time.perf_counter() - t0 < 10 and any(item.image is None for item in editor.items.values()
                                             if item.node.type_id != "OUTPUT"):
    d.wait(100)
results["thumbnails"] = sum(1 for item in editor.items.values() if item.image is not None)
assert results["thumbnails"] >= 2, results

# 加砖块节点，用鼠标事件连线：砖块输出 → 材质输出的基础色
output = graph.output_node()
bricks = editor.add_node_at("BRICKS", QPointF(output.location[0] - 260, output.location[1] + 240))
d.settle()
editor.frame_all()
d.settle()
view = editor.view


def send(kind, scene_point, button=Qt.LeftButton, buttons=Qt.LeftButton):
    local = view.mapFromScene(scene_point)
    pos = QPointF(local)
    glob = QPointF(view.viewport().mapToGlobal(local))
    event = QMouseEvent(kind, pos, glob, button, buttons, Qt.NoModifier)
    QApplication.sendEvent(view.viewport(), event)


src_item = editor.items[bricks.uid]
dst_item = editor.items[output.uid]
a = src_item.mapToScene(src_item.output_point())
b = dst_item.mapToScene(dst_item.input_point(0))
before_links = len(graph.links)
send(QEvent.MouseButtonPress, a)
for t in np.linspace(0.0, 1.0, 8):
    send(QEvent.MouseMove, a + (b - a) * float(t), Qt.NoButton, Qt.LeftButton)
send(QEvent.MouseButtonRelease, b, Qt.LeftButton, Qt.NoButton)
d.settle()
results["linked"] = graph.input_link(output.uid, 0) == bricks.uid
assert results["linked"], "拖线之后，砖块应该接到材质输出的基础色"

# 用作材质层
view3d = d.editor("VIEW_3D")
before = app.engine.renderer.read_pixels(view3d.view).astype(int)
assert d.call("layer.add_graph_fill", editor) == "FINISHED"
d.settle()
d.wait(300)
d.settle()
after = app.engine.renderer.read_pixels(view3d.view).astype(int)
results["view_changed"] = int((np.abs(after - before).max(axis=2) > 20).sum())
assert results["view_changed"] > 2000, results
d.shot("nodes_01_workspace.png")

# 改参数：视口跟着变
bricks.params.columns = 9
d.settle()
d.wait(200)
d.settle()
changed = app.engine.renderer.read_pixels(view3d.view).astype(int)
results["param_changed"] = int((np.abs(changed - after).max(axis=2) > 20).sum())
assert results["param_changed"] > 500, results

# 侧栏：当前节点的参数
graph.set_active(bricks.uid)
editor.show_sidebar()
d.settle()
d.wait(300)
d.shot("nodes_02_sidebar.png")

# 撤销到连线之前
for _ in range(8):
    if graph.input_link(output.uid, 0) != bricks.uid:
        break
    d.call("ed.undo")
    d.settle()
results["undo_unlinked"] = graph.input_link(output.uid, 0) != bricks.uid
assert results["undo_unlinked"], results
(d.out / "nodes_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
