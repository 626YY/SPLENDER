"""实际使用测：默认工程（16K 贴图）里像真人一样连着用 F4/F5/F6 的东西，量每一步多久、画的时候每帧多久。
不是通过/不通过的自检，是找哪里慢、哪里别扭。结果写到 docs/evidence/selftest/usage_paint_tools.json。
用法：python -m splender --selftest tests_new/usage_paint_tools.py。变量 app 和 d 由自检框架提供。"""
import json
import math
import time

import numpy as np

from splender.core.keymap import LEFTMOUSE, PRESS, RELEASE

report = {"steps": []}


def step(name, started, **extra):
    entry = {"name": name, "seconds": round(time.perf_counter() - started, 3)}
    entry.update(extra)
    report["steps"].append(entry)
    return entry


def frames_during(fn):
    """fn 里每次 d.wait 之间引擎一帧多久：返回每帧毫秒数的 p50、p95、最大。"""
    perf = app.engine.perf
    perf.frame_ms.clear() if hasattr(perf.frame_ms, "clear") else None
    fn()
    values = list(perf.frame_ms)
    if not values:
        return {}
    return {"frame_p50": round(float(np.percentile(values, 50)), 1), "frame_p95": round(float(np.percentile(values, 95)), 1),
            "frame_max": round(float(max(values)), 1), "frames": len(values)}


def real_stroke(editor, points, step_ms=8.0):
    """按真人鼠标速度（约 8 毫秒一个点）划一笔。"""
    d.stroke(editor, points, step_ms=step_ms)


app.window.set_workspace("绘制")
d.settle()
view = d.editor("VIEW_3D")
ts = app.project.active_texture_set
report["texture_size"] = ts.size
w, h = view.widget.width(), view.widget.height()
cx, cy = w / 2, h / 2
tools = app.tool_settings
brush = tools.brush
brush.size = 40
brush.color = (0.8, 0.2, 0.15)
line = [(cx - 160 + i * 4, cy + 30 * math.sin(i * 0.05)) for i in range(80)]

# ---- 普通画笔（对照）
started = time.perf_counter()
stats = frames_during(lambda: (real_stroke(view, line), d.settle()))
step("画笔一笔", started, **stats)

# ---- 调整层：曲线，改一下
started = time.perf_counter()
assert d.call("layer.add_adjust", adjust="CURVES") == "FINISHED"
adjust = ts.active_layer
adjust.adj_curve = ((0.0, 0.0), (0.5, 0.7), (1.0, 1.0))
d.settle()
step("加曲线调整层并改曲线", started)
started = time.perf_counter()
for k in range(10):
    adjust.adj_curve = ((0.0, 0.0), (0.5, 0.55 + 0.02 * k), (1.0, 1.0))
    d.settle()
step("拖曲线 10 次（每次等画面更新）", started)
ts.set_active(next(layer for layer in ts.layers if layer.kind == "PAINT"))

# ---- 滤镜：高斯模糊（画过的那几页）
started = time.perf_counter()
assert d.call("layer.filter_gaussian_blur", view, radius=8.0) == "FINISHED"
d.settle()
step("高斯模糊 8", started, filter=dict(app.engine.last_filter_stats))

# ---- 渐变：模型上第一次拉（要不要先烘焙）
tools.tool = "paint.gradient"
d.settle()
had_maps = app.engine.meshmap(ts.uid) is not None
index = app.history.index
started = time.perf_counter()
d.move(view, cx - 150, cy)
d.event(view, LEFTMOUSE, PRESS, cx - 150, cy)
for t in np.linspace(0, 1, 10):
    d.move(view, cx - 150 + 300 * t, cy)
d.event(view, LEFTMOUSE, RELEASE, cx + 150, cy)
bake_started = app.engine.bake is not None
d.settle()
first = step("渐变（第一次）", started, had_maps=had_maps, bake_started=bake_started)
for _ in range(600):
    if app.engine.bake is None and app.history.index > index:
        break
    d.wait(50)
d.settle()
first["applied_after"] = round(time.perf_counter() - started, 2)
first["history"] = [s.label for s in app.history.steps[:app.history.index]][-1:]
started = time.perf_counter()
d.move(view, cx - 150, cy)
d.event(view, LEFTMOUSE, PRESS, cx - 150, cy)
for t in np.linspace(0, 1, 10):
    d.move(view, cx - 150 + 300 * t, cy)
d.event(view, LEFTMOUSE, RELEASE, cx + 150, cy)
d.settle()
step("渐变（模型贴图已有）", started, filter=dict(app.engine.last_filter_stats))

# ---- 油漆桶：相近颜色
tools.tool = "paint.fill"
tools.fill.mode = "SIMILAR"
brush.color = (0.1, 0.7, 0.3)
d.settle()
started = time.perf_counter()
d.move(view, cx, cy)
d.event(view, LEFTMOUSE, PRESS, cx, cy)
d.event(view, LEFTMOUSE, RELEASE, cx, cy)
d.settle()
step("油漆桶 相近颜色", started, filter=dict(app.engine.last_filter_stats))
tools.fill.mode = "ISLAND"
started = time.perf_counter()
d.event(view, LEFTMOUSE, PRESS, cx, cy)
d.event(view, LEFTMOUSE, RELEASE, cx, cy)
d.settle()
step("油漆桶 UV 岛", started, filter=dict(app.engine.last_filter_stats))

# ---- 效果笔刷：在模型上划
for tool in ("paint.dodge", "paint.blur", "paint.smudge", "paint.sharpen", "paint.sponge"):
    tools.tool = tool
    d.settle()
    started = time.perf_counter()
    stats = frames_during(lambda: (real_stroke(view, line), d.settle()))
    step("效果笔刷 " + tool, started, **stats)

tools.tool = "paint.clone"
d.settle()
d.move(view, cx - 100, cy - 60)
d.event(view, LEFTMOUSE, PRESS, cx - 100, cy - 60, alt=True)
d.event(view, LEFTMOUSE, RELEASE, cx - 100, cy - 60, alt=True)
started = time.perf_counter()
stats = frames_during(lambda: (real_stroke(view, [(x, y + 80) for x, y in line]), d.settle()))
step("仿制图章一笔", started, **stats)
tools.tool = "paint.heal"
started = time.perf_counter()
stats = frames_during(lambda: (real_stroke(view, [(x, y + 120) for x, y in line]), d.settle()))
step("修复画笔一笔", started, **stats)

# ---- 文字、形状：默认设置，真实点击、拖动（问字的框替成直接给字）
from PySide6.QtWidgets import QInputDialog  # noqa: E402

asked = QInputDialog.getMultiLineText
QInputDialog.getMultiLineText = staticmethod(lambda *args, **kwargs: ("SPLENDER 2026", True))
try:
    tools.tool = "paint.text"
    brush.color = (1.0, 1.0, 1.0)
    d.move(view, cx, cy - 90)
    d.settle()
    started = time.perf_counter()
    d.event(view, LEFTMOUSE, PRESS, cx, cy - 90)
    d.event(view, LEFTMOUSE, RELEASE, cx, cy - 90)
    d.settle()
    step("文字（默认字号，点一下）", started, size=tools.text.size, history=[s.label for s in app.history.steps][-1:],
         filter=dict(app.engine.last_filter_stats))
finally:
    QInputDialog.getMultiLineText = asked
d.shot("usage_text_default.png", widget=view.widget)
tools.tool = "paint.shape"
brush.color = (0.95, 0.75, 0.1)
for kind, (x0, y0, x1, y1) in (("ELLIPSE", (cx - 60, cy + 40, cx + 60, cy + 120)),
                               ("RECT", (cx - 240, cy - 40, cx + 240, cy + 10))):
    tools.shape.kind = kind
    d.move(view, x0, y0)
    d.settle()
    started = time.perf_counter()
    d.event(view, LEFTMOUSE, PRESS, x0, y0)
    for t in np.linspace(0, 1, 8):
        d.move(view, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    d.event(view, LEFTMOUSE, RELEASE, x1, y1)
    d.settle()
    step("形状 " + kind, started, history=[s.label for s in app.history.steps][-1:],
         filter=dict(app.engine.last_filter_stats))
d.shot("usage_shapes.png", widget=view.widget)

report["perf_summary"] = app.engine.perf.summary()
(d.out / "usage_paint_tools.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")
d.shot("usage_final.png", widget=view.widget)
