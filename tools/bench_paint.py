"""流畅度跑分：16K、四通道同时绘制，从小笔刷到巨型笔刷，再到撑满缓存后的持续绘制。

用隐藏的显卡上下文，不开窗口。结果写到 docs/evidence/bench/paint_bench.json。
用法：python tools/bench_paint.py [--quick]
"""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet, ToolSettings, ViewShading  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402

OUT = ROOT / "docs" / "evidence" / "bench"
QUICK = "--quick" in sys.argv


def percentiles(values):
    a = np.asarray(values, float)
    if len(a) == 0:
        return {}
    return {"n": int(len(a)), "p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2), "max": round(float(a.max()), 2)}


def make_scene(engine, resolution="16384", triangles_segments=256):
    project = Project("跑分")
    ts = project.add_texture_set(TextureSet(name="材质", resolution=resolution))
    ts.add_layer(Layer(name="底色", kind="FILL"))
    ts.add_layer(Layer(name="绘制"))
    mesh = make_test_mesh("sphere", segments=triangles_segments, rings=triangles_segments // 2)
    project.add_object(MeshObject("球", mesh, [ts.uid]))
    engine.set_project(project)
    view = engine.create_view(ViewShading())
    view.resize(1920, 1080)
    engine.frame_view(view)
    engine.settle()
    return project, ts, view, mesh


def timed_frame(engine):
    """推进一帧并等显卡做完，返回毫秒（含显卡时间，比实际感受更严）。"""
    t0 = time.perf_counter()
    engine.frame()
    engine.ctx.finish()
    return (time.perf_counter() - t0) * 1000.0


def stroke(engine, view, tools, path, samples_per_frame=3, pressure=1.0, wait_merge=True):
    """沿 path（视口像素）画一笔。返回 (绘制中每帧耗时, 抬笔后合并期间每帧耗时, 抬笔到合并完成的毫秒, 是否落上)。"""
    frames = []
    x, y = path[0]
    ok = engine.stroke_begin(view, x, y, pressure, time.perf_counter(), tools)
    if not ok:
        return [], [], 0.0, False
    index = 1
    while index < len(path):
        for _ in range(samples_per_frame):
            if index >= len(path):
                break
            x, y = path[index]
            engine.stroke_move(x, y, pressure, time.perf_counter())
            index += 1
        frames.append(timed_frame(engine))
    engine.stroke_end()
    merge_frames = []
    started = time.perf_counter()
    merge_total = 0.0
    if wait_merge:
        for _ in range(2000):
            merge_frames.append(timed_frame(engine))
            if engine.stroke is None and not engine.merges:
                break
        merge_total = (time.perf_counter() - started) * 1000.0
        engine.settle(120)
    else:
        merge_frames.append(timed_frame(engine))
    return frames, merge_frames, merge_total, True


def wave_path(view, radius_px, turns, count, phase=0.0, amplitude=0.55):
    cx, cy = view.width / 2, view.height / 2
    points = []
    for i in range(count):
        t = i / (count - 1)
        angle = phase + t * turns * 2 * math.pi
        r = radius_px * (0.15 + amplitude * t)
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle) * 0.9))
    return points


GC_PAUSES = {0: [], 1: [], 2: []}
_gc_started = {}


_clock = time.perf_counter


def _gc_probe(phase, info):
    if phase == "start":
        _gc_started["t"] = _clock()
    elif "t" in _gc_started:
        GC_PAUSES[info["generation"]].append((_clock() - _gc_started["t"]) * 1000.0)


def main():
    import gc
    from splender.core.gcpolicy import POLICY
    gc.callbacks.append(_gc_probe)
    POLICY.install()            # 和程序里一样：全量回收只在停手时做
    ctx = moderngl.create_standalone_context(require=430)
    prefs = Preferences()
    for arg in sys.argv[1:]:                    # 例如 --set memory.backup_mb_painting=0
        if arg.startswith("--set") and "=" in arg:
            name, value = arg.split(" ", 1)[-1].split("=", 1) if " " in arg else arg[len("--set="):].split("=", 1)
            group, prop = name.split(".")
            current = getattr(getattr(prefs, group), prop)
            setattr(getattr(prefs, group), prop, type(current)(float(value)) if not isinstance(current, str) else value)
            print("设置", name, "=", getattr(getattr(prefs, group), prop))
    settings = [a[len("--set="):] for a in sys.argv[1:] if a.startswith("--set=")]
    history = History(int(prefs.memory.undo_steps), int(prefs.memory.undo_mb) << 20)
    engine = Engine(ctx, prefs, history)
    project, ts, view, mesh = make_scene(engine)
    tools = ToolSettings()
    brush = tools.brush
    brush.use_basecolor = brush.use_metallic = brush.use_roughness = brush.use_height = True
    brush.height = 0.03
    sphere_px = min(view.width, view.height) * 0.26          # 球在屏幕上的半径（约）
    report = {"gpu": ctx.info["GL_RENDERER"], "resolution": ts.size, "triangles": mesh.triangle_count,
              "viewport": [view.width, view.height], "channels": 4, "budgets": engine.budgets, "cases": [],
              "settings": settings}

    idle_gc = []
    cases = [("细笔 12px", 12, 260), ("常用 40px", 40, 260), ("中笔 120px", 120, 200), ("大笔 300px", 300, 140),
             ("巨笔 600px", 600, 90)]
    if QUICK:
        cases = cases[:3]
    for name, size, count in cases:
        brush.size = float(size)
        brush.color = tuple(np.random.default_rng(size).uniform(0.1, 0.9, 3))
        path = wave_path(view, sphere_px, 2.5, count, phase=size * 0.01)
        frames, merge_frames, merge_total, ok = stroke(engine, view, tools, path)
        stats = engine.stats()
        entry = {"name": name, "brush_px": size, "ok": ok, "frame_ms": percentiles(frames),
                 "merge_frame_ms": percentiles(merge_frames), "merge_total_ms": round(merge_total, 1),
                 "merge_frames": len(merge_frames), "stamp_jobs": engine.stamper.stat_jobs,
                 "layer_pages": stats["layer_pages"], "pool_used_mb": stats["used_mb"],
                 "blocking_evictions": stats["blocking_evictions"]}
        report["cases"].append(entry)
        print(entry, flush=True)
        idle_gc.append(round(POLICY.collect_if_due(), 1))        # 两笔之间停手：整理内存（不计入帧时间）

    # 撑满缓存：换着视角不停地画，直到图层页面总量远超显存预算
    rounds = 6 if QUICK else 36
    brush.size = 150.0
    all_frames, merge_frames, merges, nav_frames = [], [], [], []
    started = time.perf_counter()
    for index in range(rounds):
        view.camera.yaw = (index * 47.0) % 360.0 - 180.0
        view.camera.pitch = 50.0 * math.sin(index * 0.7)
        view.invalidate_camera()
        for _ in range(60):
            nav_frames.append(timed_frame(engine))
            if not engine.needs_render:
                break
        brush.color = tuple(np.random.default_rng(1000 + index).uniform(0.1, 0.9, 3))
        path = wave_path(view, sphere_px, 3.0, 150, phase=index * 0.9, amplitude=0.7)
        frames, mframes, merge_total, ok = stroke(engine, view, tools, path)
        all_frames.extend(frames)
        merge_frames.extend(mframes)
        merges.append(merge_total)
    stats = engine.stats()
    pressure = {"name": "撑满缓存后持续绘制", "strokes": rounds, "seconds": round(time.perf_counter() - started, 1),
                "frame_ms": percentiles(all_frames), "merge_frame_ms": percentiles(merge_frames),
                "merge_total_ms": percentiles(merges), "view_change_frame_ms": percentiles(nav_frames),
                "backed_up": stats["backed_up"], "spilled": stats["spilled"], "scratch_mb": round(stats["scratch_mb"]),
                "layer_pages": stats["layer_pages"], "pool_used_mb": stats["used_mb"], "pool_budget_mb": stats["budget_mb"],
                "ram_cache_mb": round(stats["ram_mb"]), "evicted": stats["evicted"], "uploaded": stats["uploaded"],
                "blocking_evictions": stats["blocking_evictions"], "sync_disk_reads": stats["sync_disk_reads"],
                "undo_steps": len(history.steps), "undo_mb": history.nbytes >> 20}
    report["pressure"] = pressure
    print(pressure, flush=True)
    idle_gc.append(round(POLICY.collect_if_due(force=True), 1))

    # 连续快速落笔：不等上一笔合并完就落下一笔（排线、快速铺色）
    brush.size = 260.0
    rapid_frames = []
    pending_max = 0
    started = time.perf_counter()
    for index in range(12 if not QUICK else 4):
        brush.color = tuple(np.random.default_rng(2000 + index).uniform(0.1, 0.9, 3))
        path = wave_path(view, sphere_px, 1.2, 40, phase=index * 0.5, amplitude=0.6)
        frames, mframes, _total, ok = stroke(engine, view, tools, path, wait_merge=False)
        rapid_frames.extend(frames + mframes)
        pending_max = max(pending_max, len(engine.merges))
    tail = []
    for _ in range(2000):
        tail.append(timed_frame(engine))
        if not engine.merges:
            break
    stats = engine.stats()
    rapid = {"name": "连续快速落笔（260px，不等合并）", "frame_ms": percentiles(rapid_frames),
             "tail_frame_ms": percentiles(tail), "pending_max": pending_max,
             "seconds": round(time.perf_counter() - started, 1), "blocking_evictions": stats["blocking_evictions"]}
    report["rapid"] = rapid
    print(rapid, flush=True)

    # 撤销重做的耗时
    t0 = time.perf_counter()
    history.undo()
    undo_frames = engine.settle(600)
    undo_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    history.redo()
    redo_frames = engine.settle(600)
    redo_ms = (time.perf_counter() - t0) * 1000.0
    print("撤销后 %d 帧画面到位，重做后 %d 帧" % (undo_frames, redo_frames))
    report["idle_gc_ms"] = idle_gc
    print("停手时整理内存用时:", idle_gc)
    report["gc_pauses_ms"] = {str(g): percentiles(v) for g, v in GC_PAUSES.items()}
    print("垃圾回收停顿:", report["gc_pauses_ms"])
    import gc as _gc
    print("被跟踪的对象数:", len(_gc.get_objects()))
    report["undo_ms"] = round(undo_ms, 1)
    report["redo_ms"] = round(redo_ms, 1)
    print("撤销 %.1f ms，重做 %.1f ms" % (undo_ms, redo_ms))
    OUT.mkdir(parents=True, exist_ok=True)
    name = "paint_bench%s.json" % ("" if not settings else "_" + "_".join(s.replace("=", "-") for s in settings))
    (OUT / name).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    from PIL import Image

    Image.fromarray(engine.renderer.read_pixels(view)[:, :, :3]).save(OUT / "paint_bench_final.png")
    engine.release()
    import gc as _gc2
    _gc2.callbacks.remove(_gc_probe)


if __name__ == "__main__":
    main()
