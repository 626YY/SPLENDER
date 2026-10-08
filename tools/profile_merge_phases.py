"""抬笔帧按阶段串行计时：每个阶段前后都等显卡做完，看时间花在哪一段。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import bench_paint as bp  # noqa: E402

ctx = bp.moderngl.create_standalone_context(require=430)
engine = bp.Engine(ctx, bp.Preferences(), bp.History())
project, ts, view, mesh = bp.make_scene(engine)
tools = bp.ToolSettings()
brush = tools.brush
brush.use_basecolor = brush.use_metallic = brush.use_roughness = brush.use_height = True
brush.size = 12.0
sphere_px = min(view.width, view.height) * 0.26
phases = []
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
        t2 = time.perf_counter()
        phases.append("%s CPU %.2f+显卡 %.2f" % (label, (t1 - t0) * 1000, (t2 - t1) * 1000))
        return result
    setattr(obj, name, timed)


wrap(engine.cache, "begin", "收货上传")
wrap(engine, "_advance_merges", "合并")
wrap(engine, "_update_display", "合成")
wrap(engine.cache, "maintain", "换出备份")
wrap(engine.renderer, "read_feedback", "读需求图")
for rep in range(3):
    path = bp.wave_path(view, sphere_px, 2.5, 120, phase=rep * 0.7)
    x, y = path[0]
    engine.stroke_begin(view, x, y, 1.0, time.perf_counter(), tools)
    for px, py in path[1:]:
        engine.stroke_move(px, py, 1.0, time.perf_counter())
        engine.frame()
    ctx.finish()
    engine.stroke_end()
    phases.clear()
    before = {f: len(pl.arrays) for f, pl in engine.pools.pools.items()}
    active["on"] = True
    t0 = time.perf_counter()
    engine.frame()
    ctx.finish()
    active["on"] = False
    after = {f: len(pl.arrays) for f, pl in engine.pools.pools.items()}
    print("抬笔帧 %.1f ms：%s；页池数组 %s -> %s" % ((time.perf_counter() - t0) * 1000, "；".join(phases), before, after))
    engine.settle(60)
    for _ in range(10):          # 停手：程序在空闲帧里做后台工作
        engine.frame()
engine.release()
