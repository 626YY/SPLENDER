"""实际使用测：默认 16K 工程，两层铺满（约 1.1 万页、每页都不是纯色），量自动保存有多慢、自动保存进行中画画每帧多久、
补写一笔要多久、有工程文件时只写改动要多久、恢复要多久。不是通过/不通过的自检，是找哪里慢。
结果写到 docs/evidence/selftest/usage_autosave.json。
用法：python -m splender --selftest tests_new/usage_autosave.py。变量 app 和 d 由自检框架提供。"""
import json
import math
import os
import shutil
import tempfile
import time

import numpy as np

from splender.core.keymap import LEFTMOUSE, PRESS, RELEASE
from splender.doc.project import PLANE_COLOR
from splender.engine.autosave import restore

report = {"steps": []}


def step(name, started, **extra):
    entry = {"name": name, "seconds": round(time.perf_counter() - started, 3)}
    entry.update(extra)
    report["steps"].append(entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)
    return entry


def frame_stats(values):
    if not values:
        return {}
    return {"frame_p50": round(float(np.percentile(values, 50)), 2), "frame_p95": round(float(np.percentile(values, 95)), 2),
            "frame_max": round(float(max(values)), 1), "frames": len(values)}


_frame_log = None


def _timed_frame(original):
    def frame():
        painting = app.engine.stroke is not None
        started = time.perf_counter()
        try:
            return original()
        finally:
            if _frame_log is not None:
                _frame_log.append(((time.perf_counter() - started) * 1000.0, painting))
    return frame


def frames_during(fn, painting_only=False):
    """fn 运行期间每一帧引擎用了多久（自己记，不受性能统计只留最近 240 帧的限制）。painting_only：只算笔划进行中的帧。"""
    global _frame_log
    _frame_log = []
    try:
        fn()
    finally:
        values, _frame_log = _frame_log, None
    return frame_stats([ms for ms, painting in values if painting or not painting_only])


def file_mb(path):
    return round(sum(os.path.getsize(path + s) for s in ("", "-wal") if os.path.exists(path + s)) / 1048576.0, 2)


def page_count():
    return sum(store.page_count() for store in app.engine.layers.stores.values())


def wait_autosave(limit=600.0):
    started = time.perf_counter()
    while saver.active and time.perf_counter() - started < limit:
        app.request_frame()
        d.wait(10)
    assert not saver.active, "自动保存没做完"
    return dict(saver.last_result or {})


def strokes(count=3, dy=0.0):
    """在模型上划几笔（起点要落在模型上，不然这一笔不算）。每一笔都要记进撤销历史。"""
    for k in range(count):
        y0 = cy - 70 + 50 * k + dy
        before = len(_pushes)
        d.stroke(view, [(cx - 150 + i * 3.75, y0 + 20 * math.sin(i * 0.08)) for i in range(81)], step_ms=8.0)
        deadline = time.perf_counter() + 10
        while len(_pushes) == before and time.perf_counter() < deadline:
            d.wait(20)                  # 抬笔后合并进图层、记一步撤销
        assert len(_pushes) > before, "这一笔没画上"
        d.wait(120)                     # 两笔之间停一下（像真人）


_pushes = []
app.history.changed.connect(lambda: _pushes.append(app.history.undo_label))
app.window.set_workspace("绘制")
app.prefs.files.autosave = False          # 计时器别插手，这里手动开始每一次
d.settle()
view = d.editor("VIEW_3D")
engine = app.engine
engine.frame = _timed_frame(engine.frame)
saver = engine.autosave
ts = app.project.active_texture_set
report["texture_size"] = ts.size
w, h = view.widget.width(), view.widget.height()
cx, cy = w / 2, h / 2
tools = app.tool_settings
brush = tools.brush
brush.size = 40

# ---- 准备：两层铺满，每层加杂色（每页都不是纯色，最费的情况）
started = time.perf_counter()
for index, color in enumerate(((0.55, 0.35, 0.25), (0.25, 0.45, 0.6))):
    if index:
        assert d.call("layer.add_paint") == "FINISHED"
    tools.tool = "paint.fill"
    tools.fill.mode = "LAYER"
    brush.color = color
    d.settle()
    d.move(view, cx, cy)
    d.event(view, LEFTMOUSE, PRESS, cx, cy)
    d.event(view, LEFTMOUSE, RELEASE, cx, cy)
    d.settle()
    layer = ts.active_layer
    engine.apply_filter(ts, layer, "ADD_NOISE", {"amount": 0.25, "seed": index + 1}, {PLANE_COLOR: (True,)}, "杂色")
    d.settle(30000)
tools.tool = "paint.brush"
brush.color = (0.9, 0.85, 0.2)
d.settle()
app.project.mark_dirty(True)
# 让后台备份把显存里的页都读回内存（实际用的时候，画了一阵之后就是这样）
deadline = time.perf_counter() + 120
while engine.cache.stats().get("unbacked", 0) > 0 and time.perf_counter() < deadline:
    app.request_frame()
    d.wait(20)
step("准备：两层铺满加杂色", started, pages=page_count(), unbacked=engine.cache.stats().get("unbacked"))

# ---- 对照：没有自动保存时画三笔
base_stats = frames_during(lambda: (strokes(3), d.settle()), painting_only=True)
step("对照：画三笔（没有自动保存）", time.perf_counter(), **base_stats)

# ---- 整份自动保存（没存过的工程，全部页都要写），同时接着画三笔
started = time.perf_counter()
assert app.autosave_now()
during = frames_during(lambda: strokes(3, dy=10), painting_only=True)
still_running = saver.active
result = wait_autosave()
step("整份自动保存，期间画三笔", started, painting=during, still_running_after_strokes=still_running,
     pages=result.get("pages"), autosave_seconds=round(result.get("seconds", 0.0), 2), file_mb=file_mb(saver.path),
     pump_p95_ms=round(result.get("pump_p95_ms", 0.0), 2), pump_max_ms=round(result.get("pump_max_ms", 0.0), 2))
report["p95_increase_ms"] = round(during.get("frame_p95", 0) - base_stats.get("frame_p95", 0), 2)

# ---- 停手时整份自动保存的每帧开销（不画画，只看界面帧）
app.project.mark_dirty(True)
first_path = saver.path
saver.reset()
saver.writer.sync()
started = time.perf_counter()
assert app.autosave_now()
idle_frames = frames_during(lambda: wait_autosave())
result = dict(saver.last_result)
step("整份自动保存（停手时）", started, pages=result.get("pages"), autosave_seconds=round(result.get("seconds", 0.0), 2),
     pump_p95_ms=round(result.get("pump_p95_ms", 0.0), 2), pump_max_ms=round(result.get("pump_max_ms", 0.0), 2),
     **idle_frames)

# ---- 再画一笔：只补写这一笔
strokes(1, dy=40)
d.settle()
started = time.perf_counter()
assert app.autosave_now()
result = wait_autosave()
step("补写一笔", started, pages=result.get("pages"), autosave_seconds=round(result.get("seconds", 0.0), 2))

# ---- 有工程文件：存盘后画一笔，自动保存只写改动
work = tempfile.mkdtemp(prefix="splender_usage_autosave_")
started = time.perf_counter()
assert d.call("wm.save_as", filepath=os.path.join(work, "大工程.splender")) == "FINISHED"
step("另存为（对照）", started, file_mb=round(os.path.getsize(os.path.join(work, "大工程.splender")) / 1048576.0, 1))
strokes(1, dy=-30)
d.settle()
started = time.perf_counter()
assert app.autosave_now()
result = wait_autosave()
step("有工程文件，画一笔后自动保存", started, pages=result.get("pages"),
     autosave_seconds=round(result.get("seconds", 0.0), 2), file_mb=file_mb(saver.path))

# ---- 恢复：原工程复制一份再盖上改动，打开
saver.writer.sync()
target = os.path.join(work, "恢复.splender")
started = time.perf_counter()
info = restore(saver.path, target)
step("还原成工程文件", started, pages=info["pages"], warnings=info["warnings"])
started = time.perf_counter()
app.open_project(target)
d.settle(60000)
step("打开恢复出来的工程", started)

report["first_path_deleted"] = not os.path.exists(first_path or "")
(d.out / "usage_autosave.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
app.new_project()
d.settle()
shutil.rmtree(work, ignore_errors=True)
