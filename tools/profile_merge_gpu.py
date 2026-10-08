"""抬笔帧的显卡耗时与调度、屏障次数。"""
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
counts = {"run": 0, "barrier": 0}
import moderngl  # noqa: E402
orig_run = moderngl.ComputeShader.run
orig_barrier = moderngl.Context.memory_barrier


def run(self, *a, **k):
    counts["run"] += 1
    return orig_run(self, *a, **k)


NARROW = "--narrow" in sys.argv


def barrier(self, *a, **k):
    counts["barrier"] += 1
    if NARROW:
        return orig_barrier(self, 0x2028)       # 只等纹理读取、图像读写、存储缓冲
    return orig_barrier(self, *a, **k)


moderngl.ComputeShader.run = run
moderngl.Context.memory_barrier = barrier
query = ctx.query(time=True)
for rep in range(4):
    path = bp.wave_path(view, sphere_px, 2.5, 120, phase=rep * 0.7)
    x, y = path[0]
    engine.stroke_begin(view, x, y, 1.0, time.perf_counter(), tools)
    for px, py in path[1:]:
        engine.stroke_move(px, py, 1.0, time.perf_counter())
        engine.frame()
    ctx.finish()
    engine.stroke_end()
    counts["run"] = counts["barrier"] = 0
    t0 = time.perf_counter()
    with query:
        engine.frame()
    cpu = (time.perf_counter() - t0) * 1000
    ctx.finish()
    total = (time.perf_counter() - t0) * 1000
    print("抬笔帧：CPU %.2f ms，含等待 %.2f ms，显卡 %.2f ms，调度 %d 次，屏障 %d 次" % (
        cpu, total, query.elapsed / 1e6, counts["run"], counts["barrier"]))
    engine.settle(60)
engine.release()
