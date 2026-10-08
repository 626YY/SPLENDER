"""整机自检：在 UV 编辑器里画。切到 UV 工作区，用鼠标事件画一笔（走真实的键位和绘制操作），
三维视口里应该看得到，撤销能撤掉。变量 app 和 d 由自检框架提供。"""
import json
import logging

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
view3d = d.editor("VIEW_3D")
before = app.engine.renderer.read_pixels(view3d.view).astype(int)

app.window.set_workspace("UV")
d.settle()
d.wait(300)
image = d.editor("IMAGE_EDITOR")
assert image is not None, "UV 工作区里应该有 UV 编辑器"
app.tool_settings.tool = "paint.brush"
app.tool_settings.brush.size = 30
app.tool_settings.brush.color = (0.1, 0.35, 0.95)
w, h = image.widget.width(), image.widget.height()
points = [(w * 0.25 + w * 0.5 * i / 40, h * 0.5 + h * 0.08 * np.sin(i * 0.35)) for i in range(41)]
undo_before = app.history.can_undo
d.stroke(image, points)
d.settle()
results["can_undo"] = app.history.can_undo
d.shot("uvpaint_01_uv_editor.png")

app.window.set_workspace("绘制")
d.settle()
d.wait(200)
view3d = d.editor("VIEW_3D")
after = app.engine.renderer.read_pixels(view3d.view).astype(int)
blue = (after[:, :, 2] - after[:, :, 0]) > 60
results["blue_pixels_3d"] = int(blue.sum())
d.shot("uvpaint_02_view3d.png")
assert results["blue_pixels_3d"] > 300, results
d.call("ed.undo")
d.settle()
undone = app.engine.renderer.read_pixels(view3d.view).astype(int)
results["undo_max_diff"] = int(np.abs(undone - before).max())
assert results["undo_max_diff"] < 6, results
(d.out / "uvpaint_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
