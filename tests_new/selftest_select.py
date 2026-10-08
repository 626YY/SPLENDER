"""整机自检：选区（照 Photoshop）。UV 视图里拖矩形选框 → 选区对、边线画出来了 → 跨过选区边画一笔，只画在里面 →
Ctrl+D 取消 → 套索、多边形套索（点回第一个点闭合）→ 三维视口里拖椭圆选框（投到模型上，背面不选）→ 反选 →
Alt+Backspace 填色（新的填充层自动带上选区蒙版）→ 魔棒、快速选择 → 羽化 → 有选区时 Ctrl+J 只复制选区里的、
Delete 清掉选区里的 → 撤销重做。变量 app 和 d 由自检框架提供。"""
import json
import logging

import numpy as np

from splender.core.keymap import LEFTMOUSE, PRESS, RELEASE
from splender.doc.project import PLANE_COLOR, PLANE_MASK
from splender.engine.pagepool import PAGE

errors = []


class _Catch(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())


logging.getLogger("splender").addHandler(_Catch())
app.window.set_workspace("UV")
results = {}
d.settle()
d.out.mkdir(parents=True, exist_ok=True)
image = d.editor("IMAGE_EDITOR")
view3d = d.editor("VIEW_3D")
assert image is not None and view3d is not None
ts = app.project.active_texture_set
paint = next(layer for layer in ts.layers if layer.kind == "PAINT")
ts.set_active(paint)
brush = app.tool_settings.brush
engine = app.engine
selection = engine.selection(ts)


def plane_texel(store_plane, u, v, default, channels, dtype=np.uint8):
    app.host.make_current()
    if store_plane is None:
        return np.full(channels, default)
    x, y = int(u * ts.size), int(v * ts.size)
    page = store_plane.levels[0].pages.get((x // PAGE, y // PAGE))
    if page is None:
        return np.full(channels, default)
    raw = np.frombuffer(engine.cache.page_bytes(page), dtype).reshape(PAGE, PAGE, channels)
    return raw[y % PAGE, x % PAGE].astype(np.float64) / 255.0


def sel_at(u, v):
    return float(plane_texel(selection.plane(), u, v, selection.default, 1)[0]) if selection.active else 1.0


def texel(layer, u, v):
    store = engine.layers.stores.get(layer.uid)
    plane = store.planes.get(PLANE_COLOR) if store is not None else None
    return plane_texel(plane, u, v, 0.0, 4)


def uv_pixel(u, v):
    view = image.view
    return ((u - view.center[0]) * view.zoom + view.width * 0.5, view.height * 0.5 - (v - view.center[1]) * view.zoom)


def drag(editor, start, end, steps=8, **kw):
    d.move(editor, *start)
    d.event(editor, LEFTMOUSE, PRESS, *start, **kw)
    for t in np.linspace(0.0, 1.0, steps):
        d.move(editor, start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t, **kw)
    d.event(editor, LEFTMOUSE, RELEASE, *end, **kw)
    d.settle()


def click(editor, x, y, **kw):
    d.move(editor, x, y, **kw)
    d.event(editor, LEFTMOUSE, PRESS, x, y, **kw)
    d.event(editor, LEFTMOUSE, RELEASE, x, y, **kw)
    d.settle()


def labels():
    return [s.label for s in app.history.steps[:app.history.index]]


def render2d():
    d.settle()
    app.host.make_current()
    return engine.renderer.read_pixels(image.view)[:, :, :3].astype(np.float32)


# ---- 菜单和工具都在
from splender.core import registry  # noqa: E402

assert registry.menu("PAINT_MT_select") is not None
image.toolbar_visible = True
d.settle()
results["uv_toolbar_select"] = [t for t in getattr(image, "toolbar")._buttons if t.startswith("paint.select")]
assert len(results["uv_toolbar_select"]) == 6, results

# ---- UV 视图里拖矩形选框
app.tool_settings.tool = "paint.select_rect"
app.prefs.viewport.selection_animate = False        # 截图对比时虚线别动
d.settle()
before_ants = render2d()
index = app.history.index
drag(image, uv_pixel(0.2, 0.8), uv_pixel(0.5, 0.5))
assert app.history.index == index + 1 and labels()[-1] == "选框", labels()
assert selection.active
results["rect_values"] = [round(sel_at(0.35, 0.65), 3), round(sel_at(0.1, 0.1), 3), round(sel_at(0.6, 0.65), 3)]
assert results["rect_values"][0] > 0.99 and results["rect_values"][1] < 0.01 and results["rect_values"][2] < 0.01, results
after_ants = render2d()
# 边线：选区边上（矩形的左边 u = 0.2）一圈像素变了，离边远的地方没变
ex, ey = uv_pixel(0.2, 0.65)
edge = np.abs(after_ants[int(ey) - 20:int(ey) + 20, int(ex) - 3:int(ex) + 4] - before_ants[int(ey) - 20:int(ey) + 20,
                                                                                         int(ex) - 3:int(ex) + 4])
tx, ty = uv_pixel(0.35, 0.8)                          # 上边（横着的边）也要有
top = np.abs(after_ants[int(ty) - 3:int(ty) + 4, int(tx) - 20:int(tx) + 20] - before_ants[int(ty) - 3:int(ty) + 4,
                                                                                         int(tx) - 20:int(tx) + 20])
fx, fy = uv_pixel(0.35, 0.65)
far = np.abs(after_ants[int(fy) - 5:int(fy) + 5, int(fx) - 5:int(fx) + 5] - before_ants[int(fy) - 5:int(fy) + 5,
                                                                                       int(fx) - 5:int(fx) + 5])
results["ants"] = [float(edge.sum(axis=2).max()), float(top.sum(axis=2).max()), float(far.sum(axis=2).max())]
assert results["ants"][0] > 100 and results["ants"][1] > 100 and results["ants"][2] < 10, results
d.shot("select_00_rect_uv.png", widget=image.widget)

# ---- 跨过选区的边画一笔：只画在选区里
app.tool_settings.tool = "paint.brush"
brush.color = (0.9, 0.1, 0.1)
brush.size = 30
d.settle()
before_in, before_out = texel(paint, 0.4, 0.65), texel(paint, 0.6, 0.65)
start, end = uv_pixel(0.3, 0.65), uv_pixel(0.7, 0.65)
d.stroke(image, [(start[0] + (end[0] - start[0]) * t, start[1]) for t in np.linspace(0, 1, 30)])
d.settle()
engine.complete_strokes(include_active=True)
inside, outside = texel(paint, 0.4, 0.65), texel(paint, 0.6, 0.65)
results["stroke"] = [list(np.round(inside, 3)), list(np.round(outside, 3))]
assert np.allclose(inside[:3], (0.9, 0.1, 0.1), atol=0.03), results
assert np.allclose(outside, before_out, atol=0.01), results
d.shot("select_01_stroke_inside.png", widget=image.widget)

# ---- Ctrl+D 取消选择、Ctrl+Shift+D 找回来
d.move(image, *uv_pixel(0.5, 0.5))
assert d.key(image, "D", ctrl=True)
d.settle()
assert not selection.active
assert d.key(image, "D", ctrl=True, shift=True)
d.settle()
assert selection.active
assert d.key(image, "D", ctrl=True)
d.settle()
# 没有选区时哪里都能画：在选区外面（v = 0.35）横着画一道绿的，后面 Ctrl+J 时它不该跟着复制
brush.color = (0.1, 0.8, 0.2)
start, end = uv_pixel(0.1, 0.35), uv_pixel(0.9, 0.35)
d.stroke(image, [(start[0] + (end[0] - start[0]) * t, start[1]) for t in np.linspace(0, 1, 30)])
d.settle()
engine.complete_strokes(include_active=True)
assert texel(paint, 0.3, 0.35)[3] > 0.99, "没有选区时画上了"

# ---- 套索：画一个三角形
app.tool_settings.tool = "paint.select_lasso"
d.settle()
tri = [uv_pixel(0.1, 0.1), uv_pixel(0.4, 0.1), uv_pixel(0.25, 0.4)]
path = []
for a, b in zip(tri, tri[1:] + tri[:1]):
    path += [(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) for t in np.linspace(0, 1, 12)]
d.stroke(image, path)
d.settle()
assert labels()[-1] == "套索", labels()
results["lasso"] = [round(sel_at(0.25, 0.2), 3), round(sel_at(0.12, 0.38), 3)]
assert results["lasso"][0] > 0.99 and results["lasso"][1] < 0.01, results

# ---- 多边形套索：按住 Shift（添加）点出一个四边形，点回第一个点闭合
app.tool_settings.tool = "paint.select_poly"
d.settle()
quad = [uv_pixel(0.6, 0.6), uv_pixel(0.9, 0.6), uv_pixel(0.9, 0.9), uv_pixel(0.6, 0.9)]
click(image, *quad[0], shift=True)
for x, y in quad[1:]:
    click(image, x, y)
assert app.wm.modal_ops(), "多边形套索还在等点"
click(image, quad[0][0] + 2, quad[0][1] - 2)
assert labels()[-1] == "多边形套索", labels()
results["polygon"] = [round(sel_at(0.75, 0.75), 3), round(sel_at(0.25, 0.2), 3)]
assert results["polygon"][0] > 0.99 and results["polygon"][1] > 0.99, "添加：原来的三角形还在"
d.shot("select_02_lasso_polygon.png", widget=image.widget)

# ---- 三维视口：拖椭圆选框，投到模型上；背面不选
app.window.set_workspace("UV")
app.tool_settings.tool = "paint.select_ellipse"
d.settle()
w, h = view3d.view.width, view3d.view.height
steps = app.history.index
drag(view3d, (w * 0.3, h * 0.3), (w * 0.7, h * 0.7))
for _ in range(600):                                  # 第一次要先烘焙模型贴图，好了自动选上
    if engine.bake is None and app.history.index > steps:
        break
    d.wait(50)
d.settle()
assert labels()[-1] == "选框", labels()
app.host.make_current()
hit = engine.uv_hit(view3d.view, w * 0.5, h * 0.5)
assert hit is not None
front = sel_at(*hit[1])
camera = view3d.view.camera
camera.yaw += 180.0
view3d.view.invalidate_camera()
d.settle()
back_hit = engine.uv_hit(view3d.view, w * 0.5, h * 0.5)
camera.yaw -= 180.0
view3d.view.invalidate_camera()
d.settle()
back = sel_at(*back_hit[1]) if back_hit is not None else 0.0
results["ellipse3d"] = [round(front, 3), round(back, 3)]
assert front > 0.99 and back < 0.01, results
d.shot("select_03_ellipse_3d.png", widget=view3d.widget)

# ---- 反选，再 Alt+Backspace 填色：新的填充层自带选区蒙版
assert d.key(view3d, "I", ctrl=True, shift=True)
d.settle()
assert labels()[-1] == "反选", labels()
results["inverted"] = round(sel_at(*hit[1]), 3)
assert results["inverted"] < 0.01, results
layers_before = len(ts.layers)
assert d.key(view3d, "BACK_SPACE", alt=True)
d.settle()
filled = ts.active_layer
assert len(ts.layers) == layers_before + 1 and filled.kind == "FILL" and filled.has_mask, [l.name for l in ts.layers]
store = engine.layers.stores.get(filled.uid)
mask_plane = store.planes.get(PLANE_MASK) if store is not None else None
results["fill_mask"] = [float(plane_texel(mask_plane, *hit[1], filled.mask_default, 1)[0]),
                        float(plane_texel(mask_plane, *back_hit[1], filled.mask_default, 1)[0]) if back_hit else None]
assert results["fill_mask"][0] < 0.01, results
assert labels()[-1] == "用颜色填充", labels()
app.history.undo()
d.settle()
assert len(ts.layers) == layers_before, "一步撤销掉填充层和它的蒙版"
ts.set_active(paint)

# ---- 魔棒：点到模型上，选上颜色相近的一片
app.tool_settings.tool = "paint.select_wand"
app.tool_settings.selection.mode = "SET"
d.settle()
click(view3d, w * 0.5, h * 0.5)
assert labels()[-1] == "魔棒", labels()
results["wand_center"] = round(sel_at(*hit[1]), 3)
assert results["wand_center"] > 0.99, results

# ---- 快速选择：在 UV 视图里拖
assert d.key(image, "D", ctrl=True)
app.tool_settings.tool = "paint.select_quick"
app.tool_settings.selection.quick_size = 40
d.settle()
start, end = uv_pixel(0.3, 0.3), uv_pixel(0.45, 0.3)
d.stroke(image, [(start[0] + (end[0] - start[0]) * t, start[1]) for t in np.linspace(0, 1, 15)])
d.settle()
assert labels()[-1] == "快速选择", labels()
results["quick"] = round(sel_at(0.38, 0.3), 3)
assert results["quick"] > 0.99, results
d.shot("select_04_quick.png", widget=image.widget)

# ---- 羽化（Shift+F6）：边上变成半选
assert d.key(image, "D", ctrl=True)
app.tool_settings.tool = "paint.select_rect"
d.settle()
drag(image, uv_pixel(0.2, 0.8), uv_pixel(0.5, 0.5))
assert d.call("select.modify", image, kind="FEATHER", amount=float(ts.size) * 0.02) == "FINISHED"
d.settle()
results["feather_edge"] = round(sel_at(0.2, 0.65), 3)
assert 0.15 < results["feather_edge"] < 0.85, results

# ---- 有选区时 Ctrl+J 只复制选区里的，Delete 清掉选区里的
app.history.undo()
d.settle()
count = len(ts.layers)
assert d.key(image, "J", ctrl=True)
d.settle()
copy = ts.active_layer
assert len(ts.layers) == count + 1 and labels()[-1] == "通过拷贝的图层", labels()
results["copy"] = [list(np.round(texel(copy, 0.4, 0.65), 3)), list(np.round(texel(copy, 0.3, 0.35), 3))]
assert results["copy"][0][3] > 0.99, "选区里的红的复制了"
assert results["copy"][1][3] < 0.01 and texel(paint, 0.3, 0.35)[3] > 0.99, "选区外的绿的没复制"
app.history.undo()
d.settle()
ts.set_active(paint)
assert d.key(image, "DEL")
d.settle()
assert labels()[-1] == "清除选区里的内容", labels()
results["cleared"] = [round(float(texel(paint, 0.4, 0.65)[3]), 3), round(float(texel(paint, 0.6, 0.65)[3]), 3)]
assert results["cleared"][0] < 0.01, results
app.history.undo()
d.settle()
assert texel(paint, 0.4, 0.65)[3] > 0.99, "撤销后回来了"

results["history"] = labels()[-12:]
results["errors"] = errors
(d.out / "select_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                           encoding="utf-8")
assert not errors, errors
