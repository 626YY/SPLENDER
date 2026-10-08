"""整机自检：滤镜（照 Photoshop 的滤镜菜单）。画一笔 → 菜单里的分组 → 高斯模糊 → 在「调整上一步」里加大半径 →
撤销回到原样 → Shift+R 再来一次 → 蒙版上的模糊 → 空图层上铺云彩。变量 app 和 d 由自检框架提供。"""
import json
import logging
import math

import numpy as np

from splender.doc.project import PLANE_COLOR, PLANE_FORMAT, PLANE_MASK
from splender.engine.pagepool import PAGE
from splender.ui.widgets import build_menu

errors = []


class _Catch(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())


logging.getLogger("splender").addHandler(_Catch())
app.window.set_workspace("绘制")
results = {}
d.settle()
d.out.mkdir(parents=True, exist_ok=True)
view = d.editor("VIEW_3D")
ts = app.project.active_texture_set
w, h = view.widget.width(), view.widget.height()
cx, cy = w / 2, h / 2


def plane_values(layer, plane, limit=None):
    """图层这种页第 0 级的纹素（只看有页的格，limit 给出时只取前几页），(n, 通道) 浮点。"""
    app.host.make_current()
    store = app.engine.layers.stores.get(layer.uid)
    if store is None or plane not in store.planes:
        return np.zeros((0, 4), np.float32)
    fmt = PLANE_FORMAT[plane]
    channels = {"rgba8": 4, "rg16f": 2, "r8": 1}[fmt]
    dtype = np.float16 if fmt == "rg16f" else np.uint8
    parts = []
    for page in list(store.planes[plane].levels[0].pages.values())[:limit]:
        raw = np.frombuffer(app.engine.cache.page_bytes(page), dtype).reshape(-1, channels).astype(np.float32)
        parts.append(raw / (1.0 if fmt == "rg16f" else 255.0))
    return np.concatenate(parts) if parts else np.zeros((0, channels), np.float32)


def soft_texels(layer) -> int:
    """覆盖度在 0 和 1 之间的纹素数（越糊越多）。"""
    values = plane_values(layer, PLANE_COLOR)
    return int(((values[:, 3] > 0.03) & (values[:, 3] < 0.97)).sum())


paint = next(layer for layer in ts.layers if layer.kind == "PAINT")
ts.set_active(paint)
app.tool_settings.brush.color = (0.9, 0.15, 0.1)
app.tool_settings.brush.size = 60
app.tool_settings.brush.hardness = 1.0
d.stroke(view, [(cx - 120 + i * 8, cy + 25 * math.sin(i * 0.3)) for i in range(30)])
d.settle()
soft0 = soft_texels(paint)
results["soft_before"] = soft0

# ---- 菜单：照 Photoshop 分组
menu = build_menu("LAYERS_MT_filter", view.context())
groups = [a.text().split(chr(9))[0] for a in menu.actions() if a.text()]
menu.deleteLater()
results["groups"] = groups
assert groups == ["模糊", "锐化", "杂色", "风格化", "像素化", "渲染", "其它"], groups
blur_menu = build_menu("LAYERS_MT_filter_blur", view.context())
results["blur_items"] = [a.text().split(chr(9))[0] for a in blur_menu.actions() if a.text()]
blur_menu.deleteLater()
assert results["blur_items"] == ["高斯模糊", "动感模糊", "径向模糊", "表面模糊"], results

# ---- 高斯模糊，然后在「调整上一步」里加大半径
steps = len(app.history.steps)
assert d.call("layer.filter_gaussian_blur", view, radius=4.0) == "FINISHED"
d.settle()
soft1 = soft_texels(paint)
results["soft_r4"] = soft1
assert soft1 > soft0 * 1.5 + 100, results
assert len(app.history.steps) == steps + 1 and app.history.steps[-1].label == "高斯模糊"
panel = view.redo_panel
results["redo_panel"] = bool(panel.isVisible())
assert panel.isVisible(), "滤镜做完左下角应该出现调整面板"
d.shot("filters_00_blur.png", widget=view.widget)
panel._op.radius = 16.0
d.wait(50)
d.settle()
soft2 = soft_texels(paint)
results["soft_r16"] = soft2
assert soft2 > soft1 * 1.3, results
assert len(app.history.steps) == steps + 1, "调整只是重做那一步"
d.shot("filters_01_blur_adjusted.png", widget=view.widget)
# ---- 撤销回到原样，重做回来；Shift+R 再做一次
d.call("ed.undo")
d.settle()
results["soft_undo"] = soft_texels(paint)
assert abs(results["soft_undo"] - soft0) <= max(5, soft0 * 0.01), results
d.call("ed.redo")
d.settle()
d.move(view, cx, cy)
d.key(view, "R", shift=True)
d.settle()
results["history_after_repeat"] = [s.label for s in app.history.steps[-3:]]
assert app.history.steps[-1].label == "高斯模糊" and len(app.history.steps) == steps + 2, results

# ---- 杂色、马赛克、浮雕都能在真窗口里跑
for idname, props in (("layer.filter_add_noise", {"amount": 0.2}), ("layer.filter_mosaic", {"cell": 12.0}),
                      ("layer.filter_emboss", {}), ("layer.filter_unsharp_mask", {"amount": 1.5, "radius": 2.0})):
    assert d.call(idname, view, **props) == "FINISHED", idname
    d.settle()
d.shot("filters_02_stack.png", widget=view.widget)

# ---- 蒙版：画蒙版时滤镜改蒙版
assert d.call("layer.mask_add", fill="WHITE") == "FINISHED"
ts.paint_target = "MASK"
app.tool_settings.brush.mask_value = 0.0
d.stroke(view, [(cx - 60 + i * 6, cy) for i in range(20)])
d.settle()
mask_before = plane_values(paint, PLANE_MASK)
assert d.call("layer.filter_gaussian_blur", view, radius=10.0) == "FINISHED"
d.settle()
mask_after = plane_values(paint, PLANE_MASK)
mid = lambda v: int(((v[:, 0] > 0.05) & (v[:, 0] < 0.95)).sum())  # noqa: E731
results["mask_soft"] = [mid(mask_before), mid(mask_after)]
assert mid(mask_after) > mid(mask_before) * 1.5 + 50, results
ts.paint_target = "CONTENT"

# ---- 新的空图层上铺云彩
assert d.call("layer.add_paint") == "FINISHED"
clouds = ts.active_layer
app.tool_settings.brush.color = (0.1, 0.2, 0.6)
app.tool_settings.brush.secondary_color = (0.95, 0.9, 0.8)
import time  # noqa: E402

started = time.perf_counter()
assert d.call("layer.filter_clouds", view, scale=128.0) == "FINISHED"
results["clouds_seconds"] = [round(time.perf_counter() - started, 2), dict(app.engine.last_filter_stats)]
d.settle()
store = app.engine.layers.stores[clouds.uid]
values = plane_values(clouds, PLANE_COLOR, limit=48)
results["clouds"] = [len(store.planes[PLANE_COLOR].levels[0].pages), float(values[:, 3].min()),
                     float(values[:, 0].std())]
assert results["clouds"][0] == (ts.size // PAGE) ** 2 and values[:, 3].min() > 0.99, results
assert values[:, 0].std() > 0.02, results
d.shot("filters_03_clouds.png", widget=view.widget)

results["errors"] = errors
(d.out / "filters_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
