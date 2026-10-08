"""整机自检：模型贴图烘焙、生成器蒙版、智能材质。
先把预览球细分、雕几道纹路，再烘焙、加智能材质（撤销重做）、调生成器、看面板；最后再雕一笔，回到绘制时应自动重烘。
变量 app 和 d 由自检框架提供。"""
import json
import logging
import math
import time

import numpy as np

errors = []


class _Catch(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())


logging.getLogger("splender").addHandler(_Catch())
app.window.set_workspace("绘制")       # 这个自检从绘制工作区（纹理绘制模式）开始
results = {}
d.settle()
d.out.mkdir(parents=True, exist_ok=True)


def wait_bake(limit=180.0):
    t0 = time.perf_counter()
    while app.engine.bake is not None and time.perf_counter() - t0 < limit:
        d.wait(50)
    d.settle()
    return time.perf_counter() - t0


def center_of(view):
    return view.widget.width() / 2, view.widget.height() / 2


def match_camera(source, target):
    """让绘制视口和雕刻视口从同一个角度看（截图里能看到雕出来的纹路）。"""
    a, b = source.view.camera, target.view.camera
    b.target = a.target.copy()
    b.distance, b.yaw, b.pitch, b.lens, b.ortho = a.distance, a.yaw, a.pitch, a.lens, a.ortho
    target.view.invalidate_camera()
    target.view.dirty = True


def wave(cx, cy, oy, amp=22, n=44, x0=-170, x1=170):
    return [(cx + x0 + (x1 - x0) * i / (n - 1), cy + oy + amp * math.sin(i * 0.42)) for i in range(n)]


# 1. 细分两级，雕几道纹路
app.window.set_workspace("雕刻")
d.settle()
d.wait(300)
sview = d.editor("VIEW_3D")
d.call("sculpt.subdivide", levels=2)
d.settle()
cx, cy = center_of(sview)
app.tool_settings.sculpt.size = 55
app.tool_settings.sculpt.strength = 0.85
for tool, oy in (("sculpt.crease", -100), ("sculpt.draw", -40), ("sculpt.crease", 20), ("sculpt.clay_strips", 80)):
    app.tool_settings.tool = tool
    d.settle()
    d.stroke(sview, wave(cx, cy, oy))
    d.settle()
d.shot("bake_00_sculpted.png")
results["history"] = [step.label for step in app.history.steps]
results["sculpt_view"] = [sview.widget.width(), sview.widget.height()]
app.window.set_workspace("绘制")
d.settle()
match_camera(sview, d.editor("VIEW_3D"))
d.settle()
ts = app.project.active_texture_set
results["triangles"] = app.project.objects[0].data.triangle_count
assert app.engine.meshmap(ts.uid) is None

# 2. 烘焙
ts.meshmap.resolution = "2048"
assert d.call("bake.mesh_maps") == "FINISHED"
results["bake_seconds"] = round(wait_bake(), 2)
maps = app.engine.meshmap(ts.uid)
assert maps is not None, "烘焙后应有模型贴图"
results["bake_stats"] = maps.meta["stats"]
a = maps.read_arrays()["a"]
covered = a[:, :, 3] > 250
results["ao_mean"] = round(float(a[covered, 0].mean() / 255.0), 3)
results["ao_p5"] = round(float(np.percentile(a[covered, 0], 5) / 255.0), 3)
results["curvature_p5_p95"] = [round(float(np.percentile(a[covered, 1], q) / 255.0), 3) for q in (5, 95)]
results["ao_dark_fraction"] = round(float((a[covered, 0] < 250).mean()), 4)
results["ao_min"] = round(float(a[covered, 0].min() / 255.0), 3)
results["concave_fraction"] = round(float((a[covered, 1] < 115).mean()), 4)
d.shot("bake_01_after_bake.png")
(d.out / "bake_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert results["ao_min"] < 0.97, "雕出的沟里应该更暗：%s" % results
assert results["concave_fraction"] > 0.002, "雕出的沟应该是凹的：%s" % results
props = d.editor("PROPERTIES")
if props is not None:
    props.tabs.setCurrent("SET")
    props._on_tab("SET")
    d.settle()
    d.shot("bake_02_meshmaps_panel.png")

# 3. 智能材质：加上、截图、撤销
before = len(ts.layers)
results["smart"] = {}
for material, shot in (("bronze", "bake_03_bronze.png"), ("worn_paint", "bake_04_worn_paint.png"),
                       ("rust_iron", "bake_05_rust_iron.png"), ("lacquer_gold", "bake_06_lacquer_gold.png")):
    assert d.call("layer.add_smart_material", material=material) == "FINISHED", material
    d.settle()
    d.shot(shot)
    results["smart"][material] = len(ts.layers) - before
    d.call("ed.undo")
    d.settle()
    assert len(ts.layers) == before, "撤销应去掉整组图层"
d.call("ed.redo")
d.settle()
assert len(ts.layers) == before + results["smart"]["lacquer_gold"], "重做应加回整组图层"

# 4. 图层页的生成器面板；调范围
if props is not None:
    props.tabs.setCurrent("LAYER")
    props._on_tab("LAYER")
gold = next(layer for layer in ts.layers if layer.name.endswith("描金"))
ts.set_active(gold)
d.settle()
d.shot("bake_07_generator_panel.png")
gold.gen_range = 0.6
d.settle()
d.shot("bake_08_more_gold.png")
gold.gen_range = 0.3
d.settle()

# 5. 再雕一笔：回到绘制时自动重烘
app.window.set_workspace("雕刻")
d.settle()
d.wait(200)
sview = d.editor("VIEW_3D")
cx, cy = center_of(sview)
app.tool_settings.tool = "sculpt.inflate"
d.settle()
d.stroke(sview, [(cx - 40 + i * 4, cy + 130) for i in range(20)])
d.settle()
app.window.set_workspace("绘制")
d.settle(1000)
results["auto_rebake_started"] = app.engine.bake is not None or not app.engine.meshmap(ts.uid).meta.get("stale")
wait_bake()
assert app.engine.meshmap(ts.uid) is not None and not app.engine.meshmap(ts.uid).meta.get("stale"), "应已自动重烘"
d.shot("bake_09_after_rebake.png")
results["errors"] = errors
(d.out / "bake_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert results["auto_rebake_started"]
assert not errors, errors
