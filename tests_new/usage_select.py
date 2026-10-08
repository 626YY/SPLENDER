"""实际使用测：默认工程（16K 贴图）里像真人一样用选区，量每一步多久、有选区时画画每帧多久、边线流动时每帧多久。
不是通过/不通过的自检，是找哪里慢。结果写到 docs/evidence/selftest/usage_select.json。
用法：python -m splender --selftest tests_new/usage_select.py。变量 app 和 d 由自检框架提供。"""
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
    perf = app.engine.perf
    perf.frame_ms.clear() if hasattr(perf.frame_ms, "clear") else None
    fn()
    values = list(perf.frame_ms)
    if not values:
        return {}
    return {"frame_p50": round(float(np.percentile(values, 50)), 1), "frame_p95": round(float(np.percentile(values, 95)), 1),
            "frame_max": round(float(max(values)), 1), "frames": len(values)}


def drag(editor, start, end, steps=10, **kw):
    d.move(editor, *start)
    d.event(editor, LEFTMOUSE, PRESS, *start, **kw)
    for t in np.linspace(0.0, 1.0, steps):
        d.move(editor, start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t, **kw)
    d.event(editor, LEFTMOUSE, RELEASE, *end, **kw)
    d.settle()


app.window.set_workspace("绘制")
d.settle()
view = d.editor("VIEW_3D")
ts = app.project.active_texture_set
report["texture_size"] = ts.size
engine = app.engine
selection = engine.selection(ts)
w, h = view.widget.width(), view.widget.height()
cx, cy = w / 2, h / 2
tools = app.tool_settings
tools.brush.size = 40
tools.brush.color = (0.8, 0.2, 0.15)
line = [(cx - 160 + i * 4, cy + 30 * math.sin(i * 0.05)) for i in range(80)]

# ---- 三维视口：矩形选框（第一次要先烘焙模型贴图）
tools.tool = "paint.select_rect"
d.settle()
index = app.history.index
started = time.perf_counter()
drag(view, (cx - 120, cy - 90), (cx + 120, cy + 90))
for _ in range(600):
    if engine.bake is None and app.history.index > index:
        break
    d.wait(50)
d.settle()
step("矩形选框（第一次，含烘焙）", started, filter=dict(getattr(engine, "last_filter_stats", None) or {}))
started = time.perf_counter()
drag(view, (cx - 60, cy - 140), (cx + 160, cy + 40), shift=True)
step("矩形选框 添加", started, filter=dict(getattr(engine, "last_filter_stats", None) or {}))

# ---- 套索
tools.tool = "paint.select_lasso"
d.settle()
ring = [(cx + 150 * math.cos(a), cy + 110 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 60)]
started = time.perf_counter()
d.stroke(view, ring)
d.settle()
step("套索", started)

# ---- 有选区时画画：每帧多久（对照：没有选区）
tools.tool = "paint.brush"
d.settle()
started = time.perf_counter()
stats = frames_during(lambda: (d.stroke(view, line, step_ms=8.0), d.settle()))
step("有选区时画笔一笔", started, **stats)
assert d.key(view, "D", ctrl=True)
d.settle()
started = time.perf_counter()
stats = frames_during(lambda: (d.stroke(view, [(x, y + 60) for x, y in line], step_ms=8.0), d.settle()))
step("没有选区时画笔一笔", started, **stats)
assert d.key(view, "D", ctrl=True, shift=True)
d.settle()

# ---- 菜单里的：反选、全选、羽化、扩展、收缩、平滑、边界
for name, call in (("反选", lambda: d.call("select.invert", view)),
                   ("反选回来", lambda: d.call("select.invert", view)),
                   ("羽化 16", lambda: d.call("select.modify", view, kind="FEATHER", amount=16.0)),
                   ("扩展 32", lambda: d.call("select.modify", view, kind="EXPAND", amount=32.0)),
                   ("收缩 32", lambda: d.call("select.modify", view, kind="CONTRACT", amount=32.0)),
                   ("平滑 8", lambda: d.call("select.modify", view, kind="SMOOTH", amount=8.0)),
                   ("边界 24", lambda: d.call("select.modify", view, kind="BORDER", amount=24.0)),
                   ("撤销", lambda: app.history.undo()),
                   ("重做", lambda: app.history.redo()),
                   ("全选", lambda: d.call("select.all", view)),
                   ("反选全选（空）", lambda: d.call("select.invert", view))):
    started = time.perf_counter()
    result = call()
    d.settle()
    step(name, started, result=result, filter=dict(getattr(engine, "last_filter_stats", None) or {}),
         pages=len(selection.cells()))

# ---- 魔棒（相近颜色，整张贴图）
tools.tool = "paint.select_wand"
d.settle()
started = time.perf_counter()
d.move(view, cx, cy)
d.event(view, LEFTMOUSE, PRESS, cx, cy)
d.event(view, LEFTMOUSE, RELEASE, cx, cy)
d.settle()
step("魔棒", started, filter=dict(getattr(engine, "last_filter_stats", None) or {}))

# ---- 快速选择：拖一段
tools.tool = "paint.select_quick"
d.settle()
assert d.key(view, "D", ctrl=True)
d.settle()
started = time.perf_counter()
stats = frames_during(lambda: (d.stroke(view, [(cx - 100 + i * 5, cy - 50) for i in range(40)], step_ms=10.0), d.settle()))
step("快速选择一笔", started, **stats)

# ---- 有选区时滤镜
tools.tool = "paint.select_rect"
d.settle()
drag(view, (cx - 120, cy - 90), (cx + 120, cy + 90))
started = time.perf_counter()
assert d.call("layer.filter_gaussian_blur", view, radius=8.0) == "FINISHED"
d.settle()
step("有选区时高斯模糊 8", started, filter=dict(getattr(engine, "last_filter_stats", None) or {}))

# ---- 边线流动：每帧多久
app.prefs.viewport.selection_animate = True
app._update_ants_timer()
started = time.perf_counter()
perf = engine.perf
perf.frame_ms.clear() if hasattr(perf.frame_ms, "clear") else None
end = time.perf_counter() + 2.0
while time.perf_counter() < end:
    d.wait(20)
values = list(perf.frame_ms)
step("边线流动 2 秒", started, frames=len(values),
     frame_p50=round(float(np.percentile(values, 50)), 1) if values else None,
     frame_p95=round(float(np.percentile(values, 95)), 1) if values else None)
d.shot("usage_select_3d.png", widget=view.widget)

report["perf_summary"] = engine.perf.summary()
(d.out / "usage_select.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
