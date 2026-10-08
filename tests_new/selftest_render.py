"""整机自检：渲染。视口「渲染」着色模式、三个内置引擎和 Arnold 插件出图、插件列表。变量 app 和 d 由自检框架提供。"""
import json
import time

from PIL import Image

from splender.core import addons
from splender.render import engines

results = {}
app.headless_render = True               # 自检里不弹渲染结果窗口
d.settle()
view = d.editor("VIEW_3D")
assert view is not None and view.view is not None, "没有 3D 视口"

# 插件
results["addons"] = [(a.name, a.enabled, a.error) for a in addons.addons()]
results["engines"] = [cls.idname for cls in engines.engines()]
assert "PATHTRACE" in results["engines"] and "REALTIME" in results["engines"]

# 视口渲染模式：渐进路径追踪
view.shading.mode = "RENDERED"
started = time.perf_counter()
d.wait(2500)
results["viewport_status"] = app.engine.rendered_status(view.view)
d.shot("render_01_viewport_rendered.png")
results["viewport_seconds"] = round(time.perf_counter() - started, 1)
view.shading.mode = "MATERIAL"
d.settle()

props = app.project.render
props.resolution_x = 640
props.resolution_y = 360
props.texture_size = "2048"


def render_with(engine_id: str, shot: str, timeout: float = 240.0) -> dict:
    props.engine = engine_id
    settings = props.engine_settings(engine_id)
    if settings is not None and hasattr(settings, "samples"):
        settings.samples = 48
    if settings is not None and hasattr(settings, "camera_aa"):
        settings.camera_aa = 3
    d.call("render.render", view)
    job = app.render_job
    assert job is not None, "%s 没有开始出图" % engine_id
    start = time.perf_counter()
    while not job.done and time.perf_counter() - start < timeout:
        d.wait(100)
    image = job.preview()
    info = {"done": job.done, "error": job.error, "status": job.status, "seconds": round(job.elapsed, 2)}
    if image is not None:
        Image.fromarray(image[:, :, :3]).save(d.out / shot)
        info["size"] = [int(image.shape[1]), int(image.shape[0])]
        info["mean"] = round(float(image[:, :, :3].mean()), 1)
    return info


d.out.mkdir(parents=True, exist_ok=True)
results["pathtrace"] = render_with("PATHTRACE", "render_02_pathtrace.png")
results["realtime"] = render_with("REALTIME", "render_03_realtime.png")
results["workbench"] = render_with("WORKBENCH", "render_04_workbench.png")
if engines.engine("ARNOLD") is not None and engines.engine("ARNOLD").available()[0]:
    arnold_addon = addons.get("splender.addons.render_arnold")
    if arnold_addon is not None and arnold_addon.preferences is not None:
        arnold_addon.preferences.skip_license_check = "NEVER"    # 自检不等授权检查
    results["arnold"] = render_with("ARNOLD", "render_05_arnold.png", timeout=600.0)
props.engine = "PATHTRACE"
(d.out / "render_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
for key in ("pathtrace", "realtime", "workbench", "arnold"):
    if key in results:
        assert results[key]["done"] and not results[key]["error"], "%s 出图失败：%s" % (key, results[key])
