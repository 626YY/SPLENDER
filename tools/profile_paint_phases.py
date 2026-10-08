"""绘制帧按阶段串行计时（每阶段前后等显卡做完）。用法：python tools/profile_paint_phases.py [笔刷直径]"""
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import bench_paint as bp  # noqa: E402

size = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
ctx = bp.moderngl.create_standalone_context(require=430)
engine = bp.Engine(ctx, bp.Preferences(), bp.History())
project, ts, view, mesh = bp.make_scene(engine)
tools = bp.ToolSettings()
brush = tools.brush
brush.use_basecolor = brush.use_metallic = brush.use_roughness = brush.use_height = True
brush.size = size
sphere_px = min(view.width, view.height) * 0.26
spent = defaultdict(list)
active = {"on": False}


def wrap(obj, name, label):
    fn = getattr(obj, name)

    def timed(*a, **k):
        if not active["on"]:
            return fn(*a, **k)
        ctx.finish()
        t0 = time.perf_counter()
        result = fn(*a, **k)
        t1 = time.perf_counter()
        ctx.finish()
        spent[label + " CPU"].append((t1 - t0) * 1000)
        spent[label + " 显卡"].append((time.perf_counter() - t1) * 1000)
        return result
    setattr(obj, name, timed)


wrap(engine.renderer, "read_feedback", "需求图")
wrap(engine, "_collect_dabs", "取笔触")
wrap(engine.cache, "begin", "收货上传")
wrap(engine.stamper, "stamp", "盖章")
wrap(engine.stamper, "update_mips", "笔划缩小")
wrap(engine, "_prefetch_layer", "预取")
wrap(engine, "_update_display", "合成")
wrap(engine.renderer, "render", "渲染")
wrap(engine.renderer, "render_feedback", "需求渲染")
wrap(engine.cache, "maintain", "换出备份")
for rep in range(2):
    path = bp.wave_path(view, sphere_px, 2.5, 90, phase=rep * 0.9)
    x, y = path[0]
    engine.stroke_begin(view, x, y, 1.0, time.perf_counter(), tools)
    spent.clear()
    totals = []
    index = 1
    while index < len(path):
        for _ in range(3):
            if index < len(path):
                engine.stroke_move(*path[index], 1.0, time.perf_counter())
                index += 1
        active["on"] = True
        t0 = time.perf_counter()
        engine.frame()
        ctx.finish()
        totals.append((time.perf_counter() - t0) * 1000)
        active["on"] = False
    engine.stroke_end()
    engine.settle(200)
    for _ in range(10):
        engine.frame()
    print("第 %d 笔（%.0fpx）串行帧中位数 %.2f ms" % (rep, size, np.median(totals)))
    print("   " + "，".join("%s %.2f" % (k, np.median(v)) for k, v in spent.items() if np.median(v) >= 0.05))
engine.release()
