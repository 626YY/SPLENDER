"""找出绘制时每帧的开销在哪：缓存压力场景下用 cProfile 统计 engine.frame。"""
import cProfile
import math
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import bench_paint as bp  # noqa: E402
from splender.core.gcpolicy import POLICY  # noqa: E402

POLICY.install()
ctx = bp.moderngl.create_standalone_context(require=430)
prefs = bp.Preferences()
history = bp.History(int(prefs.memory.undo_steps), int(prefs.memory.undo_mb) << 20)
engine = bp.Engine(ctx, prefs, history)
project, ts, view, mesh = bp.make_scene(engine)
tools = bp.ToolSettings()
brush = tools.brush
brush.use_basecolor = brush.use_metallic = brush.use_roughness = brush.use_height = True
brush.height = 0.03
sphere_px = min(view.width, view.height) * 0.26
brush.size = 150.0
# 先撑满缓存（不统计）
for index in range(14):
    view.camera.yaw = (index * 47.0) % 360.0 - 180.0
    view.camera.pitch = 50.0 * math.sin(index * 0.7)
    view.invalidate_camera()
    engine.settle(60)
    path = bp.wave_path(view, sphere_px, 3.0, 150, phase=index * 0.9, amplitude=0.7)
    bp.stroke(engine, view, tools, path)
print("撑满后:", {k: engine.stats()[k] for k in ("layer_pages", "used_mb", "ram_mb", "unbacked")})
profiler = cProfile.Profile()
phase = {"service": [], "advance": [], "merge": [], "compose": [], "render": []}
original = {}


def wrap(name, obj, attr):
    fn = getattr(obj, attr)

    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            phase[name].append((time.perf_counter() - t0) * 1000)
    setattr(obj, attr, timed)


wrap("service", engine.cache, "service")
wrap("advance", engine, "_advance_stroke")
wrap("merge", engine, "_advance_merges")
wrap("compose", engine, "_update_display")
wrap("render", engine.renderer, "render")
frames = []
profiler.enable()
for index in range(14, 22):
    view.camera.yaw = (index * 47.0) % 360.0 - 180.0
    view.camera.pitch = 50.0 * math.sin(index * 0.7)
    view.invalidate_camera()
    engine.settle(60)
    for k in phase:
        phase[k].clear()
    path = bp.wave_path(view, sphere_px, 3.0, 150, phase=index * 0.9, amplitude=0.7)
    f, mf, total, ok = bp.stroke(engine, view, tools, path, wait_merge=True)
    frames.extend(f)
    print("第 %d 笔：绘制帧 p50 %.2f p95 %.2f；各阶段中位数 %s" % (
        index, bp.np.percentile(f, 50), bp.np.percentile(f, 95),
        {k: round(float(bp.np.median(v)), 2) for k, v in phase.items() if v}))
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("tottime").print_stats(28)
engine.release()
