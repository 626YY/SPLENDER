"""整机自检：导出贴图。画一道带高度的笔划，按纹理集的导出设置导出（勾上环境遮蔽、改文件名规则和尺寸），
检查导出的文件和法线，再看导出面板。变量 app 和 d 由自检框架提供。"""
import json
import logging
import math
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

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

# 1. 画一道带高度的笔划
view = d.editor("VIEW_3D")
cx, cy = view.widget.width() / 2, view.widget.height() / 2
app.tool_settings.brush.use_height = True
d.stroke(view, [(cx - 150 + 300 * i / 40, cy + 25 * math.sin(i * 0.4)) for i in range(41)])
d.settle()
ts = app.project.active_texture_set

# 2. 按导出设置导出
ts.export.ao = True
ts.export.name_pattern = "{纹理集}-{通道}"
ts.export.size = "1024"
folder = tempfile.mkdtemp(prefix="splender_export_")
try:
    assert d.call("wm.export_textures", directory=folder) == "FINISHED"
    names = sorted(os.listdir(folder))
    results["files"] = names
    expected = sorted("%s-%s.png" % (ts.name, c) for c in ("basecolor", "metallic", "roughness", "height", "normal", "ao"))
    assert names == expected, names
    normal = np.asarray(Image.open(os.path.join(folder, "%s-normal.png" % ts.name)))
    assert normal.shape == (1024, 1024, 3), normal.shape
    tilt = np.abs(normal[:, :, :2].astype(int) - 128)
    results["normal_tilt_max"] = int(tilt.max())
    results["normal_tilted_pixels"] = int((tilt.max(axis=2) > 8).sum())
    assert results["normal_tilt_max"] > 10, "笔划处的法线应该倾斜"
    flat = np.abs(normal.astype(int) - np.array([128, 128, 255])).max(axis=2) <= 2
    results["normal_flat_fraction"] = round(float(flat.mean()), 3)
    assert flat.mean() > 0.5, "没画的地方法线应该是平的"
    def as_rgb(channel):
        image = np.asarray(Image.open(os.path.join(folder, "%s-%s.png" % (ts.name, channel))))
        if image.dtype != np.uint8:                   # 16 位高度：中灰是零，放大一点看得清
            image = np.clip((image.astype(np.float64) / 65535.0 - 0.5) * 8.0 + 0.5, 0, 1) * 255
        image = image.astype(np.uint8)
        return np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image[:, :, :3]

    preview = np.concatenate([as_rgb(c) for c in ("basecolor", "height", "normal")], axis=1)
    Image.fromarray(preview).resize((1536, 512)).save(d.out / "export_02_files.png")
    # 只导法线、16 位
    ts.export.normal_depth = "16"
    for name in names:
        os.remove(os.path.join(folder, name))
    written = app.export_textures(folder, channels=["normal"])
    with open(written[0], "rb") as handle:
        header = handle.read(29)
    results["normal16_bit_depth"] = header[24]
    assert header[24] == 16, "16 位法线"
finally:
    shutil.rmtree(folder, ignore_errors=True)

# 3. 导出面板
props = d.editor("PROPERTIES")
if props is not None:
    props.tabs.setCurrent("SET")
    props._on_tab("SET")
    d.settle()
    try:
        from PySide6.QtWidgets import QScrollArea
        for area in getattr(props, "widget", props).findChildren(QScrollArea):
            bar = area.verticalScrollBar()
            bar.setValue(bar.maximum())
    except Exception as error:  # noqa: BLE001
        results["scroll_error"] = str(error)
    d.settle()
    d.shot("export_01_panel.png")
(d.out / "export_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
