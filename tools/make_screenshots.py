"""给 README 拍截图，存到 docs/screenshots（窗口开在屏幕外，不抢前台）。
依次：雕刻、烘焙后加智能材质、材质节点、节点纹理、UV 视图里的选区和调整层、视口路径追踪。

用法：python -m splender --selftest tools/make_screenshots.py。变量 app 和 d 由自检框架提供。
"""
import math
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = (1600, 1000)


def shot(name: str) -> None:
    d.settle()
    d.wait(200)
    d.settle()
    app.window.grab().save(str(OUT / name))


def wait_bake(limit: float = 240.0) -> None:
    started = time.perf_counter()
    while app.engine.bake is not None and time.perf_counter() - started < limit:
        d.wait(50)
    d.settle()


def wave(cx, cy, oy, amp=22, n=44, x0=-170, x1=170):
    return [(cx + x0 + (x1 - x0) * i / (n - 1), cy + oy + amp * math.sin(i * 0.42)) for i in range(n)]


def match_camera(source, target) -> None:
    a, b = source.view.camera, target.view.camera
    b.target = a.target.copy()
    b.distance, b.yaw, b.pitch, b.lens, b.ortho = a.distance, a.yaw, a.pitch, a.lens, a.ortho
    target.view.invalidate_camera()
    target.view.dirty = True


app.window.resize(*SIZE)
app.headless_render = True
app.prefs.viewport.selection_animate = False

# 1. 雕刻：细分两级，雕几道纹路
app.window.set_workspace("雕刻")
d.settle()
d.wait(300)
sview = d.editor("VIEW_3D")
d.call("sculpt.subdivide", levels=2)
d.settle()
cx, cy = sview.widget.width() / 2, sview.widget.height() / 2
app.tool_settings.sculpt.size = 55
app.tool_settings.sculpt.strength = 0.85
for tool, oy in (("sculpt.crease", -100), ("sculpt.draw", -40), ("sculpt.crease", 20), ("sculpt.clay_strips", 80)):
    app.tool_settings.tool = tool
    d.settle()
    d.stroke(sview, wave(cx, cy, oy))
    d.settle()
app.tool_settings.tool = "sculpt.draw"
d.move(sview, cx + 30, cy - 150)                    # 笔刷圈放在球的上面
shot("01_sculpt.png")

# 2. 绘制：烘焙模型贴图，加智能材质「锈铁」
app.window.set_workspace("绘制")
d.settle()
pview = d.editor("VIEW_3D")
match_camera(sview, pview)
ts = app.project.active_texture_set
ts.meshmap.resolution = "2048"
d.call("bake.mesh_maps")
wait_bake()
d.call("layer.add_smart_material", material="rust_iron")
d.settle()
app.tool_settings.tool = "paint.brush"
shot("02_paint.png")

# 3. 材质：着色器节点，图层的颜色乘上按遮蔽取的颜色渐变
app.window.set_workspace("材质")
d.settle()
shader = d.editor("SHADER_EDITOR")
if ts.material is None:
    d.call("shader.use_nodes", shader)
    d.settle()
graph = ts.material
stack = next(n for n in graph.nodes.values() if n.type_id == "LAYER_STACK")
out = graph.output_node()
x0, y0 = stack.location
out.location = (x0 + 820, y0)
maps = graph.add_node("MESH_MAPS", (x0, y0 + 230))
ramp = graph.add_node("VALTORGB", (x0 + 250, y0 + 230))
ramp.values.ramp = ("LINEAR", ((0.0, 0.25, 0.16, 0.12, 1.0), (0.85, 1.0, 1.0, 1.0, 1.0)))
mix = graph.add_node("MIX_RGB", (x0 + 560, y0 + 40))
mix.values.blend_type = "MULTIPLY"
mix.values.fac = 1.0
graph.link(maps.uid, "ao", ramp.uid, "fac")
graph.link(stack.uid, "base_color", mix.uid, "a")
graph.link(ramp.uid, "color", mix.uid, "b")
graph.link(mix.uid, "result", out.uid, "base_color")
d.settle()
shader.frame_all()
shot("03_material_nodes.png")

# 4. 节点：程序化纹理（类 Substance Designer），砖块接到材质输出，用作材质层
app.window.set_workspace("节点")
d.settle()
d.wait(300)
editor = d.editor("NODE_EDITOR")
d.call("node.graph_new", editor)
d.settle()
tex_graph = editor.graph()
output = tex_graph.output_node()
from PySide6.QtCore import QPointF  # noqa: E402

bricks = editor.add_node_at("BRICKS", QPointF(output.location[0] - 260, output.location[1] + 240))
tex_graph.link(bricks.uid, output.uid, 0)
d.settle()
started = time.perf_counter()
while time.perf_counter() - started < 10 and any(item.image is None for item in editor.items.values()
                                                 if item.node.type_id != "OUTPUT"):
    d.wait(100)
editor.frame_all()
d.call("layer.add_graph_fill", editor)
d.settle()
d.wait(400)
shot("04_node_textures.png")
d.call("ed.undo")
d.settle()

# 5. 渲染：视口路径追踪
app.window.set_workspace("绘制")
d.settle()
app.project.render.engine = "PATHTRACE"
rview = d.editor("VIEW_3D")
app.tool_settings.tool = "paint.brush"
rview.shading.mode = "RENDERED"
d.settle()
d.wait(6000)
shot("05_render_pathtrace.png")
rview.shading.mode = "MATERIAL"
d.settle()

# 6. 选区：在三维视口里拖椭圆选框（投到模型上），加色相饱和度调整层（自动只调选区里）；UV 视图里同一块
app.window.set_workspace("UV")
d.settle()
from splender.core.keymap import LEFTMOUSE, PRESS, RELEASE  # noqa: E402

v3d = d.editor("VIEW_3D")
app.tool_settings.tool = "paint.select_ellipse"
d.settle()
w, h = v3d.widget.width(), v3d.widget.height()
a, b = (w * 0.30, h * 0.30), (w * 0.62, h * 0.62)
d.move(v3d, *a)
d.event(v3d, LEFTMOUSE, PRESS, *a)
for t in (0.25, 0.5, 0.75, 1.0):
    d.move(v3d, a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
d.event(v3d, LEFTMOUSE, RELEASE, *b)
d.settle()
d.call("layer.add_adjust", v3d, adjust="HSV")
d.settle()
layer = ts.active_layer
layer.adj_hue = 140.0
layer.adj_saturation = 0.45
d.move(v3d, w * 0.85, h * 0.85)
shot("06_selection_adjust.png")
