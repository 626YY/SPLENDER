"""整机自检：绘制模式里不是笔刷的工具（照 Photoshop）。UV 视图里有工具栏 → 渐变工具在 UV 视图里拉一条线 →
颜色按位置过渡、撤销 → 三维视口里横着拉一条线（还没有模型贴图：先烘焙，烘好了自动铺上），模型左边是笔刷颜色、
右边是备用颜色 → 油漆桶、减淡、仿制 → 文字、形状在 UV 视图里盖 → 三维视口里点一下写字：字正对屏幕、按屏幕像素的字号、
背面不写 → 三维视口里拖一个矩形，正好盖在拖出的那块。变量 app 和 d 由自检框架提供。"""
import json
import logging

import numpy as np

from splender.core.keymap import LEFTMOUSE, PRESS, RELEASE
from splender.doc.project import PLANE_COLOR
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
brush.color = (0.9, 0.1, 0.1)
brush.secondary_color = (0.1, 0.2, 0.9)


def texel(layer, u, v):
    """图层颜色页在 (u, v) 处的值 (r, g, b, 覆盖度)，0..1。"""
    app.host.make_current()
    store = app.engine.layers.stores.get(layer.uid)
    if store is None or PLANE_COLOR not in store.planes:
        return np.zeros(4)
    x, y = int(u * ts.size), int(v * ts.size)
    page = store.planes[PLANE_COLOR].levels[0].pages.get((x // PAGE, y // PAGE))
    if page is None:
        return np.zeros(4)
    raw = np.frombuffer(app.engine.cache.page_bytes(page), np.uint8).reshape(PAGE, PAGE, 4)
    return raw[y % PAGE, x % PAGE].astype(np.float64) / 255.0


def uv_pixel(u, v):
    view = image.view
    return ((u - view.center[0]) * view.zoom + view.width * 0.5, view.height * 0.5 - (v - view.center[1]) * view.zoom)


def drag(editor, start, end, steps=8):
    d.move(editor, *start)
    d.event(editor, LEFTMOUSE, PRESS, *start)
    for t in np.linspace(0.0, 1.0, steps):
        d.move(editor, start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t)
    d.event(editor, LEFTMOUSE, RELEASE, *end)
    d.settle()


# ---- UV 视图有工具栏，渐变工具在里面
app.tool_settings.tool = "paint.gradient"
d.settle()
image.toolbar_visible = True
toolbar = getattr(image, "toolbar", None)
results["uv_toolbar"] = sorted(toolbar._buttons) if toolbar is not None else None
assert toolbar is not None and "paint.gradient" in toolbar._buttons, results

# ---- UV 视图里拉渐变
steps = len(app.history.steps)
drag(image, uv_pixel(0.2, 0.5), uv_pixel(0.8, 0.5))
results["uv_history"] = [s.label for s in app.history.steps[steps:]]
assert len(app.history.steps) == steps + 1 and app.history.steps[-1].label == "渐变", results
left, mid, right = texel(paint, 0.1, 0.3), texel(paint, 0.5, 0.7), texel(paint, 0.9, 0.5)
results["uv_texels"] = [list(np.round(c, 3)) for c in (left, mid, right)]
assert np.allclose(left[:3], (0.9, 0.1, 0.1), atol=0.02) and np.allclose(right[:3], (0.1, 0.2, 0.9), atol=0.02), results
assert np.allclose(mid[:3], (0.5, 0.15, 0.5), atol=0.03) and left[3] > 0.99, results
d.shot("tools_00_uv_gradient.png")
d.call("ed.undo")
d.settle()
results["after_undo"] = list(np.round(texel(paint, 0.9, 0.5), 3))

# ---- 三维视口里拉渐变：还没有模型贴图时先烘焙，烘好了自动铺上
for _ in range(600):
    if app.engine.bake is None:
        break
    d.wait(50)
app.host.make_current()
app.engine.set_meshmap(ts.uid, None)          # 从没烘焙过的样子
w, h = view3d.view.width, view3d.view.height
d.settle()
app.host.make_current()
before = app.engine.renderer.read_pixels(view3d.view)[:, :, :3].astype(np.float32)
steps = app.history.index          # 撤销过一步：新的一步会顶掉可重做的那步，所以按位置数
d.move(view3d, w * 0.3, h * 0.5)
d.event(view3d, LEFTMOUSE, PRESS, w * 0.3, h * 0.5)
results["view3d_modal"] = [getattr(op, "idname", "") for op in app.wm.modal_ops()]
for t in np.linspace(0.0, 1.0, 8):
    d.move(view3d, w * (0.3 + 0.4 * t), h * 0.5)
d.event(view3d, LEFTMOUSE, RELEASE, w * 0.7, h * 0.5)
results["view3d_waiting"] = [app.engine.bake is not None, app.history.index - steps]
assert app.engine.bake is not None and app.history.index == steps, results
for _ in range(600):
    if app.engine.bake is None and app.history.index > steps:
        break
    d.wait(50)
d.settle()
assert app.engine.meshmap(ts.uid) is not None, "模型贴图没烘焙出来"
results["view3d_history"] = [s.label for s in app.history.steps[:app.history.index]][-2:]
assert app.history.index == steps + 1 and app.history.steps[app.history.index - 1].label == "渐变", results
app.host.make_current()
after = app.engine.renderer.read_pixels(view3d.view)[:, :, :3].astype(np.float32)
changed = np.abs(after - before).sum(axis=2) > 30
ys, xs = np.nonzero(changed)
assert len(xs) > 500, "画面应该变了"
lo, hi = np.percentile(xs, 10), np.percentile(xs, 90)
left_px = after[changed & (np.arange(after.shape[1])[None, :] < lo)]
right_px = after[changed & (np.arange(after.shape[1])[None, :] > hi)]
results["view3d_left_right"] = [list(np.round(left_px.mean(axis=0), 1)), list(np.round(right_px.mean(axis=0), 1))]
assert left_px[:, 0].mean() > left_px[:, 2].mean() + 20, results
assert right_px[:, 2].mean() > right_px[:, 0].mean() + 20, results
d.shot("tools_01_view3d_gradient.png", widget=view3d.widget)

# ---- 油漆桶：UV 视图里点一块 UV 岛，整块填上
from splender.engine import fill as fill_mod  # noqa: E402

app.tool_settings.tool = "paint.fill"
app.tool_settings.fill.mode = "ISLAND"
brush.color = (0.1, 0.8, 0.2)
obj, tris = fill_mod.triangles_of_set(app.project, ts)[0]
tri = int(tris[len(tris) // 2])
cu, cv = np.asarray(obj.data.uvs, np.float64).reshape(-1, 3, 2)[tri].mean(axis=0)
index = app.history.index
x, y = uv_pixel(cu, cv)
d.move(image, x, y)
d.event(image, LEFTMOUSE, PRESS, x, y)
d.event(image, LEFTMOUSE, RELEASE, x, y)
d.settle()
assert app.history.index == index + 1 and app.history.steps[index].label == "油漆桶", [s.label for s in app.history.steps]
results["island_texel"] = list(np.round(texel(paint, cu, cv), 3))
assert np.allclose(results["island_texel"][:3], (0.1, 0.8, 0.2), atol=0.02), results
app.host.make_current()
island = fill_mod.triangle_mask(app.engine.ctx, fill_mod.group_triangles(obj, tri, "ISLAND"), 512, dilate=6)
outside = np.argwhere(island < 0.5)
if len(outside):
    oy, ox = outside[len(outside) // 2]
    results["outside_texel"] = list(np.round(texel(paint, (ox + 0.5) / 512, (oy + 0.5) / 512), 3))
    assert not np.allclose(results["outside_texel"][:3], (0.1, 0.8, 0.2), atol=0.05), results
d.shot("tools_02_fill_island.png")
d.call("ed.undo")
d.settle()

# ---- 油漆桶：三维视口里点模型，只填颜色相近的那一条（渐变层上）
app.tool_settings.fill.mode = "SIMILAR"
app.tool_settings.fill.tolerance = 0.06
brush.color = (0.95, 0.85, 0.1)
app.host.make_current()
before = app.engine.renderer.read_pixels(view3d.view)[:, :, :3].astype(np.float32)
index = app.history.index
d.move(view3d, w * 0.5, h * 0.5)
d.event(view3d, LEFTMOUSE, PRESS, w * 0.5, h * 0.5)
d.event(view3d, LEFTMOUSE, RELEASE, w * 0.5, h * 0.5)
d.settle()
assert app.history.index == index + 1, [s.label for s in app.history.steps]
app.host.make_current()
after = app.engine.renderer.read_pixels(view3d.view)[:, :, :3].astype(np.float32)
sphere = np.abs(before - before[5, 5]).sum(axis=2) > 25          # 和左上角背景不一样的就是模型
yellow = (after[:, :, 0] > after[:, :, 2] + 60) & (after[:, :, 1] > after[:, :, 2] + 40)
results["similar_fraction"] = float(yellow.sum()) / max(1.0, float(sphere.sum()))
assert 0.03 < results["similar_fraction"] < 0.6, results
d.shot("tools_03_fill_similar.png", widget=view3d.widget)

# ---- 效果笔刷：三维视口里用减淡划一笔，划过的地方变亮
app.tool_settings.tool = "paint.dodge"
app.tool_settings.effect.strength = 1.0
app.tool_settings.brush.size = 50
d.settle()
app.host.make_current()
before = app.engine.renderer.read_pixels(view3d.view)[:, :, :3].astype(np.float32)
index = app.history.index
d.stroke(view3d, [(w * 0.35 + i * w * 0.015, h * 0.42) for i in range(20)])
d.settle()
assert app.history.index == index + 1 and app.history.steps[index].label == "减淡", [s.label for s in app.history.steps]
app.host.make_current()
after = app.engine.renderer.read_pixels(view3d.view)[:, :, :3].astype(np.float32)
row = int(h * 0.42)
results["dodge_gain"] = float((after[row - 3:row + 3, int(w * 0.4):int(w * 0.6)] -
                               before[row - 3:row + 3, int(w * 0.4):int(w * 0.6)]).mean())
assert results["dodge_gain"] > 8.0, results
d.shot("tools_04_dodge.png", widget=view3d.widget)

# ---- 仿制图章：UV 视图里 Alt 点一下定来源，再划
app.tool_settings.tool = "paint.clone"
d.settle()
sx, sy = uv_pixel(0.2, 0.5)
d.move(image, sx, sy)
d.event(image, LEFTMOUSE, PRESS, sx, sy, alt=True)
d.event(image, LEFTMOUSE, RELEASE, sx, sy, alt=True)
d.settle()
effect = app.tool_settings.effect
results["clone_source"] = [effect.has_source, round(effect.source_u, 3), round(effect.source_v, 3)]
assert effect.has_source and abs(effect.source_u - 0.2) < 0.01, results
source_color = texel(paint, 0.2, 0.5)
index = app.history.index
x0, y0 = uv_pixel(0.7, 0.5)
x1, y1 = uv_pixel(0.8, 0.5)
d.stroke(image, [(x0 + (x1 - x0) * i / 15, y0) for i in range(16)])
d.settle()
assert app.history.index == index + 1 and app.history.steps[index].label == "仿制图章", results
cloned = texel(paint, 0.72, 0.5)
results["clone"] = [list(np.round(source_color, 3)), list(np.round(cloned, 3))]
assert np.abs(cloned[:3] - texel(paint, 0.22, 0.5)[:3]).max() < 0.08, results
d.shot("tools_05_clone.png")

# ---- 文字：直接给字（点击时会弹框问字，这里不弹）
app.tool_settings.tool = "paint.text"
brush.color = (1.0, 1.0, 1.0)
d.settle()
index = app.history.index
assert d.call("paint.text", image, text="SPLENDER", size=900.0, uv=(0.5, 0.75), bold=True) == "FINISHED"
d.settle()
assert app.history.index == index + 1 and app.history.steps[index].label == "文字"
# 字的范围里有白的、也有没盖到的（字之间的空隙），范围外不动
samples = [texel(paint, 0.3 + 0.4 * k / 60.0, 0.75) for k in range(61)]
whites = sum(1 for c in samples if c[0] > 0.95 and c[1] > 0.95 and c[2] > 0.95)
results["text_whites"] = whites
assert 5 < whites < 58, results
d.shot("tools_06_text.png")

# ---- 形状：UV 视图里拖一个椭圆
app.tool_settings.tool = "paint.shape"
app.tool_settings.shape.kind = "ELLIPSE"
app.tool_settings.shape.fill = True
brush.color = (0.95, 0.5, 0.0)
d.settle()
index = app.history.index
a = uv_pixel(0.2, 0.1)
b = uv_pixel(0.4, 0.3)
drag(image, a, b)
assert app.history.index == index + 1 and app.history.steps[index].label == "形状", [s.label for s in app.history.steps]
center = texel(paint, 0.3, 0.2)
corner = texel(paint, 0.205, 0.105)
results["shape"] = [list(np.round(center, 3)), list(np.round(corner, 3))]
assert np.allclose(center[:3], (0.95, 0.5, 0.0), atol=0.02), results
assert not np.allclose(corner[:3], (0.95, 0.5, 0.0), atol=0.05), "椭圆的角上不该盖到"
d.shot("tools_07_shape.png")


def render3d():
    d.settle()
    app.host.make_current()
    return app.engine.renderer.read_pixels(view3d.view)[:, :, :3].astype(np.float32)


def turn(degrees):
    view3d.view.camera.yaw += degrees
    view3d.view.invalidate_camera()


# ---- 文字：三维视口里点一下（会弹框问字，这里替成直接给字）。字正对屏幕、按屏幕像素的字号，只写在看得见的那面
from PySide6.QtWidgets import QInputDialog  # noqa: E402

asked = QInputDialog.getMultiLineText
QInputDialog.getMultiLineText = staticmethod(lambda *args, **kwargs: ("SPLENDER", True))
yaw = view3d.view.camera.yaw
try:
    app.tool_settings.tool = "paint.text"
    app.tool_settings.text.size = 32.0
    app.tool_settings.text.bold = True
    brush.color = (0.05, 0.95, 0.95)
    d.move(view3d, w * 0.5, h * 0.5)               # 文字工具没有笔刷圈：先把鼠标挪过去，前后只差字
    turn(180.0)
    back_before = render3d()
    turn(-180.0)
    front_before = render3d()
    index = app.history.index
    d.event(view3d, LEFTMOUSE, PRESS, w * 0.5, h * 0.5)
    d.event(view3d, LEFTMOUSE, RELEASE, w * 0.5, h * 0.5)
    front_after = render3d()
    assert app.history.index == index + 1 and app.history.steps[index].label == "文字", \
        [s.label for s in app.history.steps]
    ys, xs = np.nonzero(np.abs(front_after - front_before).sum(axis=2) > 40)
    results["text3d_box"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())] if len(xs) else None
    assert len(xs) > 200, results
    assert abs(float(xs.mean()) - w * 0.5) < w * 0.06 and abs(float(ys.mean()) - h * 0.5) < h * 0.06, results
    assert (xs.max() - xs.min()) > 2.5 * (ys.max() - ys.min()), "横着写（正对屏幕）"
    assert 15 < ys.max() - ys.min() < 60, "字高按屏幕像素"
    d.shot("tools_08_text3d.png", widget=view3d.widget)
    turn(180.0)
    back_after = render3d()
    results["text3d_back_changed"] = int((np.abs(back_after - back_before).sum(axis=2) > 40).sum())
    assert results["text3d_back_changed"] < 30, "背面不该写上"
    turn(-180.0)
finally:
    QInputDialog.getMultiLineText = asked
    view3d.view.camera.yaw = yaw
    view3d.view.invalidate_camera()

# ---- 形状：三维视口里拖一个矩形，正好盖在拖出的那块（不用两个角都点在模型上）
app.tool_settings.tool = "paint.shape"
app.tool_settings.shape.kind = "RECT"
brush.color = (0.9, 0.1, 0.9)
a, b = (w * 0.40, h * 0.28), (w * 0.60, h * 0.40)
d.move(view3d, *a)
before = render3d()
sphere = np.abs(before - before[5, 5]).sum(axis=2) > 25
index = app.history.index
drag(view3d, a, b)
after = render3d()
assert app.history.index == index + 1 and app.history.steps[index].label == "形状", [s.label for s in app.history.steps]
changed = np.abs(after - before).sum(axis=2) > 40
ys, xs = np.nonzero(changed)
results["shape3d_box"] = [int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())] if len(xs) else None
assert len(xs) > 200, results
assert xs.min() >= a[0] - 4 and xs.max() <= b[0] + 4 and ys.min() >= a[1] - 4 and ys.max() <= b[1] + 4, results
box = (slice(int(a[1]) + 3, int(b[1]) - 3), slice(int(a[0]) + 3, int(b[0]) - 3))
results["shape3d_cover"] = float(changed[box][sphere[box]].mean())
# 记下的表面位置要原样（负的坐标不能被压成 0）
app.host.make_current()
_token, captured = app.engine.screen_capture
seen = np.frombuffer(captured.read(), np.float32).reshape(captured.size[1], captured.size[0], 4)
seen = seen[seen[..., 3] > 0.5][:, :3]
results["capture_range"] = [float(seen.min()), float(seen.max())]
assert results["capture_range"][0] < -0.2, results
assert results["shape3d_cover"] > 0.97, results
d.shot("tools_09_shape3d.png", widget=view3d.widget)

results["errors"] = errors
(d.out / "tools_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
assert not errors, errors
