"""剖析小笔刷抬笔那一帧的开销。"""
import cProfile
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import bench_paint as bp  # noqa: E402

ctx = bp.moderngl.create_standalone_context(require=430)
prefs = bp.Preferences()
history = bp.History()
engine = bp.Engine(ctx, prefs, history)
project, ts, view, mesh = bp.make_scene(engine)
tools = bp.ToolSettings()
brush = tools.brush
brush.use_basecolor = brush.use_metallic = brush.use_roughness = brush.use_height = True
brush.size = 12.0
sphere_px = min(view.width, view.height) * 0.26
for rep in range(3):
    path = bp.wave_path(view, sphere_px, 2.5, 120, phase=rep * 0.7)
    x, y = path[0]
    engine.stroke_begin(view, x, y, 1.0, time.perf_counter(), tools)
    for px, py in path[1:]:
        engine.stroke_move(px, py, 1.0, time.perf_counter())
        engine.frame()
    engine.ctx.finish()
    engine.stroke_end()
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    engine.frame()
    engine.ctx.finish()
    profiler.disable()
    print("第 %d 次抬笔帧 %.2f ms，剩余合并 %d" % (rep, (time.perf_counter() - t0) * 1000, len(engine.merges)))
    engine.settle(60)
    for _ in range(10):
        engine.frame()
pstats.Stats(profiler).sort_stats("tottime").print_stats(25)
engine.release()
