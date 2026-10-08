"""整机自检：同一个工程里雕高模、烘到低模。复制模型 → 雕刻复制出来的那个（细分、几笔）→ 藏起来 →
烘焙面板里「高模（场景里的）」选它 → 烘焙：法线贴图有雕刻的起伏，和把高模导出成文件再烘的结果一致；
再试「导入到当前场景」和撤销。变量 app 和 d 由自检框架提供。"""
import json
import logging
import math
import os
import shutil
import tempfile
import time

import numpy as np

errors = []


class _Catch(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())


logging.getLogger("splender").addHandler(_Catch())
from splender.core import ops  # noqa: E402

results = {}
d.out.mkdir(parents=True, exist_ok=True)
tmp = tempfile.mkdtemp(prefix="splender_highpoly_")
project = app.project
engine = app.engine


def bake_now(ts, limit=120.0):
    assert d.call("bake.mesh_maps") == "FINISHED"
    started = time.perf_counter()
    while engine.bake is not None and time.perf_counter() - started < limit:
        app.request_frame()
        d.wait(20)
    assert engine.bake is None, "烘焙超时"
    d.settle()
    maps = engine.meshmap(ts.uid)
    assert maps is not None
    return maps, time.perf_counter() - started


try:
    app.window.set_workspace("布局")
    d.settle()
    low = project.objects[0]
    ts = project.texture_set(low.material_sets[0])
    for obj in project.objects:
        obj.select = obj is low
    project.set_active_object(low)
    # 1. 复制（和原来的共用一套贴图），雕刻复制出来的那个
    ctx = app.wm.context(use_mouse=False)
    assert ops.call("object.duplicate_move", ctx, invoke=False) == "FINISHED"
    high = next(obj for obj in project.objects if obj is not low)
    assert high.material_sets == low.material_sets
    project.set_active_object(high)
    app.window.set_workspace("雕刻")
    d.settle()
    assert app.engine.sculpt is not None and app.engine.sculpt.obj is high
    d.call("sculpt.subdivide")
    d.settle()
    view = d.editor("VIEW_3D")
    cx, cy = view.widget.width() / 2, view.widget.height() / 2
    app.tool_settings.sculpt.size = 70
    for k, tool in enumerate(("sculpt.draw", "sculpt.clay", "sculpt.draw")):
        app.tool_settings.tool = tool
        d.settle()
        d.stroke(view, [(cx - 140 + 280 * i / 39, cy - 80 + 70 * k + 25 * math.sin(i * 0.35)) for i in range(40)])
        d.settle()
    app.window.set_workspace("绘制")
    d.settle()
    results["high_triangles"] = high.data.triangle_count
    results["low_triangles"] = low.data.triangle_count
    assert high.data.triangle_count > low.data.triangle_count
    # 2. 高模藏起来，回到低模；烘焙面板里选场景里的高模
    high.visible = False
    project.set_active_object(low)
    for obj in project.objects:
        obj.select = obj is low
    d.settle()
    ts.meshmap.resolution = "1024"
    ts.meshmap.cage_front = 0.05
    ts.meshmap.cage_back = 0.05
    items = [ident for ident, *_rest in type(ts.meshmap).prop("high_poly_object").items(ts.meshmap)]
    assert str(high.uid) in items, items
    ts.meshmap.high_poly_object = str(high.uid)
    maps, seconds = bake_now(ts)
    results["bake_seconds"] = round(seconds, 2)
    assert maps.has_normal and maps.meta["high_poly"] == "场景：%s" % high.name, maps.meta.get("high_poly")
    arrays = maps.read_arrays()
    covered = arrays["a"][:, :, 3] > 250
    scene_t = arrays["t"][:, :, :3].astype(np.float64).copy()
    tn = scene_t / 255.0 * 2.0 - 1.0
    tilt = np.degrees(np.arccos(np.clip(tn[covered, 2] / np.linalg.norm(tn[covered], axis=1), -1, 1)))
    results["tilt_p99"] = round(float(np.percentile(tilt, 99)), 2)
    results["tilted_fraction"] = round(float((tilt > 5).mean()), 4)
    assert results["tilt_p99"] > 8.0, "雕刻的起伏应该烘进法线贴图"
    # 3. 同一个高模导出成文件再烘，结果一致
    high_path = os.path.join(tmp, "高模.obj")
    app.export_meshes(high_path, [high], with_textures=False)
    ts.meshmap.high_poly_object = "0"
    ts.meshmap.high_poly = high_path
    maps2, _seconds = bake_now(ts)
    file_t = maps2.read_arrays()["t"][:, :, :3].astype(np.float64)
    a = scene_t[covered].reshape(-1) - scene_t[covered].mean()
    b = file_t[covered].reshape(-1) - file_t[covered].mean()
    results["corr_with_file"] = round(float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())), 5)
    assert results["corr_with_file"] >= 0.98, results["corr_with_file"]
    ts.meshmap.high_poly = ""
    ts.meshmap.high_poly_object = str(high.uid)
    # 烘焙面板截图
    props = d.editor("PROPERTIES")
    if props is not None:
        props.tabs.setCurrent("SET")
        props._on_tab("SET")
        d.settle()
        d.shot("highpoly_01_bake_panel.png")
        # 选 16K：面板上显示大约占多少显存（只看，不烘）
        ts.meshmap.resolution = "16384"
        app.notify("layers")
        d.settle()
        from PySide6.QtWidgets import QScrollArea  # noqa: E402

        from splender.ui.widgets import Label  # noqa: E402

        root = getattr(props, "widget", props)
        found = [w for w in root.findChildren(Label) if w.text().startswith(("3.0 GB", "3.1 GB")) or "GB" in w.text()]
        results["vram_label"] = found[0].text() if found else ""
        assert found and "3.0 GB" in results["vram_label"], results["vram_label"]
        for area in root.findChildren(QScrollArea):
            area.ensureWidgetVisible(found[0], 0, 120)
        d.settle()
        d.shot("highpoly_03_bake_16k.png")
        ts.meshmap.resolution = "1024"
    # 4. 导入到当前场景，撤销
    objects_before = len(project.objects)
    sets_before = len(project.texture_sets)
    assert d.call("wm.import_to_scene", filepath=high_path) == "FINISHED"
    d.settle()
    results["imported_objects"] = len(project.objects) - objects_before
    results["imported_sets"] = len(project.texture_sets) - sets_before
    assert results["imported_objects"] == 1 and results["imported_sets"] == 1
    imported = project.objects[-1]
    assert imported.name.startswith("高模") and imported.data.triangle_count == high.data.triangle_count
    assert d.call("ed.undo") == "FINISHED"
    d.settle()
    assert len(project.objects) == objects_before and len(project.texture_sets) == sets_before, "撤销导入"
    d.shot("highpoly_02_after_undo.png")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
(d.out / "highpoly_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
