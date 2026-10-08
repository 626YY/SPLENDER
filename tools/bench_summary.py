"""把 paint_bench.json 压成一张对照表。用法：python tools/bench_summary.py [json 路径]"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "docs/evidence/bench/paint_bench.json"
report = json.loads(path.read_text(encoding="utf-8"))


def fmt(d):
    return "p50 %5.2f  p95 %5.2f  p99 %5.2f  max %6.2f" % (d["p50"], d["p95"], d["p99"], d["max"]) if d else "-"


print("设置:", report.get("settings", {}))
for case in report["cases"]:
    print("%-10s 绘制帧 %s | 合并帧 %s | 合并共 %6.1f ms | 阻塞 %d" % (
        case["name"], fmt(case["frame_ms"]), fmt(case.get("merge_frame_ms")), case.get("merge_total_ms", 0),
        case["blocking_evictions"]))
p = report["pressure"]
print("缓存压力   绘制帧 %s | 合并帧 %s | 合并共 p50 %.0f max %.0f ms | 阻塞 %d | 撤销步数 %d" % (
    fmt(p["frame_ms"]), fmt(p["merge_frame_ms"]), p["merge_total_ms"]["p50"], p["merge_total_ms"]["max"],
    p["blocking_evictions"], p["undo_steps"]))
print("           转视角帧 %s" % fmt(p["view_change_frame_ms"]))
r = report.get("rapid")
if r:
    print("连续落笔   帧 %s | 收尾帧 %s | 阻塞 %d" % (fmt(r["frame_ms"]), fmt(r["tail_frame_ms"]), r["blocking_evictions"]))
print("撤销 %.1f ms，重做 %.1f ms；停手整理内存 %s" % (report["undo_ms"], report["redo_ms"], report.get("idle_gc_ms")))
