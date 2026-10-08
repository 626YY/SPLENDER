"""抬笔帧里每个显卡操作各花多少显卡时间（每个操作单独计时）。"""
import sys
import time
from collections import defaultdict
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
query = ctx.query(time=True)
spent = defaultdict(float)
calls = defaultdict(int)
active = {"on": False}


def wrap(obj, name, label):
    fn = getattr(obj, name)

    def timed(*a, **k):
        if not active["on"]:
            return fn(*a, **k)
        ctx.finish()
        with query:
            result = fn(*a, **k)
        ctx.finish()
        spent[label] += query.elapsed / 1e6
        calls[label] += 1
        return result
    setattr(obj, name, timed)


ops = engine.ops
for name in ("apply_stroke", "downsample", "check_uniform", "pack", "fill_packed", "clear"):
    wrap(ops, name, name)
wrap(engine.stamper, "stamp", "stamp")
wrap(engine.stamper, "update_mips", "stroke_mips")
wrap(engine.renderer, "render", "render")
wrap(engine.renderer, "render_feedback", "feedback")
wrap(engine.compositor, "compose", "compose")
for rep in range(3):
    path = bp.wave_path(view, sphere_px, 2.5, 120, phase=rep * 0.7)
    x, y = path[0]
    engine.stroke_begin(view, x, y, 1.0, time.perf_counter(), tools)
    for px, py in path[1:]:
        engine.stroke_move(px, py, 1.0, time.perf_counter())
        engine.frame()
    ctx.finish()
    engine.stroke_end()
    spent.clear()
    calls.clear()
    active["on"] = True
    t0 = time.perf_counter()
    with query:
        pass
    engine.frame()
    ctx.finish()
    active["on"] = False
    print("抬笔帧（逐项计时后）%.1f ms：" % ((time.perf_counter() - t0) * 1000),
          ", ".join("%s %d 次 %.2f ms" % (k, calls[k], v) for k, v in sorted(spent.items(), key=lambda kv: -kv[1])))
    engine.settle(60)
engine.release()
