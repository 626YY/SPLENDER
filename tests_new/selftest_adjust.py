"""整机自检：调整层（照 Photoshop）。Ctrl+M 加曲线层 → 在属性面板里拖曲线（拖动期间面板不重画，松开后重画、
当前控制点还在）→ Delete 删点 → 每种调整的参数面板截图 → 预设和撤销 → Ctrl+B、Ctrl+Alt+Shift+B →
调整层菜单的分组。变量 app 和 d 由自检框架提供。"""
import json
import logging
import math

import numpy as np
import shiboken6
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from splender.doc.project import ADJUST_ITEMS
from splender.ops.adjust_ops import presets
from splender.ui.curve_widget import CurveWidget
from splender.ui.ramp_widget import ColorRampWidget
from splender.ui.widgets import build_menu

errors = []


class _Catch(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())


logging.getLogger("splender").addHandler(_Catch())
app.window.set_workspace("绘制")
results = {}
d.settle()
d.out.mkdir(parents=True, exist_ok=True)
view = d.editor("VIEW_3D")
ts = app.project.active_texture_set
w, h = view.widget.width(), view.widget.height()
cx, cy = w / 2, h / 2


def pixels():
    app.host.make_current()
    return app.engine.renderer.read_pixels(view.view)[:, :, :3].astype(np.float32)


paint_layer = next(layer for layer in ts.layers if layer.kind == "PAINT")
ts.set_active(paint_layer)
app.tool_settings.brush.color = (0.8, 0.25, 0.2)
app.tool_settings.brush.size = 80
d.stroke(view, [(cx - 130 + i * 9, cy + 20 * math.sin(i * 0.4)) for i in range(30)])
d.settle()
before_image = pixels()

props = d.editor("PROPERTIES")
assert props is not None
props.tabs.setCurrent("LAYER")
props._on_tab("LAYER")
d.settle()

# ---- Ctrl+M：曲线调整层
d.move(view, cx, cy)
d.key(view, "M", ctrl=True)
d.settle()
adjust = ts.active_layer
results["ctrl_m"] = [adjust.kind, adjust.adjust_type, adjust.name]
assert adjust.kind == "ADJUST" and adjust.adjust_type == "CURVES", results


def panel_widgets(cls):
    return [wd for wd in props.panels._content.findChildren(cls) if wd.isVisible()]


def mouse(widget, kind, pos, buttons=Qt.LeftButton, button=Qt.LeftButton):
    event = QMouseEvent(kind, QPointF(pos), QPointF(widget.mapToGlobal(pos.toPoint())), button, buttons,
                        Qt.NoModifier)
    QApplication.sendEvent(widget, event)


curves = panel_widgets(CurveWidget)
assert len(curves) == 1, "曲线层的属性面板里应有一个曲线控件"
widget = curves[0]
d.shot("adjust_00_curves_panel.png", widget=props)
steps_before = len(app.history.steps)
start = widget._to_screen(0.5, 0.5)
end = widget._to_screen(0.5, 0.75)
mouse(widget, QEvent.MouseButtonPress, start)
for t in np.linspace(0.0, 1.0, 6):
    mouse(widget, QEvent.MouseMove, start + (end - start) * float(t), button=Qt.NoButton)
    d.wait(15)
results["alive_during_drag"] = shiboken6.isValid(widget)
assert results["alive_during_drag"], "拖动期间面板不该重画把曲线控件删掉"
mouse(widget, QEvent.MouseButtonRelease, end, buttons=Qt.NoButton)
d.settle()
results["curve_after_drag"] = [list(p) for p in adjust.adj_curve]
assert len(adjust.adj_curve) == 3 and abs(adjust.adj_curve[1][1] - 0.75) < 0.02, results
results["history_after_drag"] = [step.label for step in app.history.steps[steps_before:]]
assert len(app.history.steps) == steps_before + 1, results
after_image = pixels()
results["brightness"] = [float(before_image.mean()), float(after_image.mean())]
assert after_image.mean() > before_image.mean() + 3.0, results
d.shot("adjust_01_curve_dragged.png", widget=props)
d.shot("adjust_02_curve_view.png", widget=view.widget)
# 面板已经重画：新的曲线控件接着选中刚才的点
curves = panel_widgets(CurveWidget)
assert len(curves) == 1
new_widget = curves[0]
results["rebuilt"] = new_widget is not widget
results["active_after_rebuild"] = new_widget.active
assert new_widget.active == 1, results
QApplication.sendEvent(new_widget, QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
d.settle()
results["curve_after_delete"] = [list(p) for p in adjust.adj_curve]
assert len(adjust.adj_curve) == 2, results
d.call("ed.undo")
d.settle()
assert len(adjust.adj_curve) == 3, "撤销删点"

# ---- 分通道曲线：切到红通道，面板换成红曲线
adjust.adj_curve_channel = "R"
d.settle()
curves = panel_widgets(CurveWidget)
assert len(curves) == 1 and curves[0].channel == "R", [c.channel for c in curves]

# ---- 每种调整的参数面板
shots = {}
for index, (kind, label, _desc) in enumerate(ADJUST_ITEMS):
    adjust.adjust_type = kind
    d.settle()
    name = "adjust_%02d_%s.png" % (10 + index, kind.lower())
    d.shot(name, widget=props)
    shots[kind] = name
    if kind == "GRADIENT_MAP":
        assert len(panel_widgets(ColorRampWidget)) == 1, "渐变映射要有色带"
    if kind == "CURVES":
        assert len(panel_widgets(CurveWidget)) == 1
results["panel_shots"] = shots

# ---- 预设：照片滤镜换成冷却滤镜，撤销回来；「默认值」恢复
adjust.adjust_type = "PHOTO_FILTER"
d.settle()
items = presets("PHOTO_FILTER")
cool = next(i for i, (name, _v) in enumerate(items) if name.startswith("冷却滤镜 (80)"))
assert d.call("layer.adjust_preset", props, index=cool) == "FINISHED"
results["cool_filter"] = list(adjust.adj_filter_color)
assert abs(adjust.adj_filter_color[2] - 1.0) < 1e-6 and adjust.adj_filter_color[0] < 0.01, results
d.call("ed.undo")
assert abs(adjust.adj_filter_color[0] - 236 / 255.0) < 1e-6, list(adjust.adj_filter_color)
adjust.adj_filter_density = 0.8
assert d.call("layer.adjust_preset", props, index=0) == "FINISHED"
assert abs(adjust.adj_filter_density - 0.25) < 1e-6
menu = build_menu("LAYERS_MT_adjust_presets", props.context())
results["preset_menu"] = [a.text().split(chr(9))[0] for a in menu.actions() if a.text()]
assert results["preset_menu"][0] == "默认值" and len(results["preset_menu"]) == len(items), results
menu.deleteLater()
# 渐变映射的「笔刷颜色到备用颜色」
adjust.adjust_type = "GRADIENT_MAP"
app.tool_settings.brush.secondary_color = (0.1, 0.2, 0.9)
gradients = presets("GRADIENT_MAP")
brush_index = next(i for i, (name, _v) in enumerate(gradients) if name == "笔刷颜色到备用颜色")
assert d.call("layer.adjust_preset", props, index=brush_index) == "FINISHED"
stops = adjust.adj_ramp[1]
assert abs(stops[-1][3] - 0.9) < 1e-6 and abs(stops[0][1] - 0.8) < 1e-6, stops
d.settle()
d.shot("adjust_03_gradient_view.png", widget=view.widget)

# ---- Ctrl+B 色彩平衡、Ctrl+Alt+Shift+B 黑白
d.move(view, cx, cy)
d.key(view, "B", ctrl=True)
d.settle()
assert ts.active_layer.adjust_type == "COLOR_BALANCE", ts.active_layer.adjust_type
d.key(view, "B", ctrl=True, alt=True, shift=True)
d.settle()
assert ts.active_layer.adjust_type == "BLACK_WHITE", ts.active_layer.adjust_type
gray = pixels()
spread = float(np.abs(gray[:, :, 0] - gray[:, :, 2]).mean())
results["bw_channel_spread"] = spread
d.shot("adjust_04_black_white_view.png", widget=view.widget)

# ---- 调整层菜单：十五种，按 Photoshop 分三组
menu = build_menu("LAYERS_MT_adjust", props.context())
texts = [a.text().split(chr(9))[0] for a in menu.actions() if a.text()]
separators = [a for a in menu.actions() if a.isSeparator()]
results["adjust_menu"] = texts
assert len(texts) == len(ADJUST_ITEMS) and len(separators) == 2, (texts, len(separators))
menu.deleteLater()

# ---- Ctrl+Shift+L 自动色调：在上面加一个色阶层；面板里拖色阶的黑三角
from splender.ui.levels_widget import LevelsWidget  # noqa: E402

d.move(view, cx, cy)
count = len(ts.layers)
d.key(view, "L", ctrl=True, shift=True)
d.settle()
auto = ts.active_layer
results["auto_tone"] = [auto.name, auto.adjust_type, auto.adj_lv_r_in_black, auto.adj_lv_r_in_white]
assert len(ts.layers) == count + 1 and auto.adjust_type == "LEVELS" and auto.name.startswith("自动色调"), results
level_widgets = panel_widgets(LevelsWidget)
assert len(level_widgets) == 1, "色阶层的面板里应有色阶控件"
lw = level_widgets[0]
results["levels_hist_sum"] = int(lw.counts.sum()) if lw.counts is not None else 0
assert results["levels_hist_sum"] > 0, results
steps_before = len(app.history.steps)
start = lw.marker_positions()["in_black"]
mouse(lw, QEvent.MouseButtonPress, start)
for t in np.linspace(0.0, 1.0, 5):
    mouse(lw, QEvent.MouseMove, start + QPointF(60.0 * float(t), 0.0), button=Qt.NoButton)
mouse(lw, QEvent.MouseButtonRelease, start + QPointF(60.0, 0.0), buttons=Qt.NoButton)
d.settle()
results["levels_drag"] = [auto.adj_in_black, [s.label for s in app.history.steps[steps_before:]]]
assert auto.adj_in_black > 0.05 and len(app.history.steps) == steps_before + 1, results
assert app.history.steps[-1].label == "色阶", results
d.shot("adjust_05_levels_panel.png", widget=props)
d.call("ed.undo")
assert auto.adj_in_black == 0.0, auto.adj_in_black
auto.adjust_type = "CURVES"
d.settle()
assert panel_widgets(CurveWidget)[0].counts is not None, "曲线背后要有直方图"
d.shot("adjust_06_curves_histogram.png", widget=props)

results["errors"] = errors
(d.out / "adjust_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
