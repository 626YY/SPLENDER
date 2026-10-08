"""对照标定：同一场景用 SPLENDER 的路径追踪和 Arnold 各渲一张，找出环境图朝向差的转角、检查贴图方向。

用法：python tools/arnold_calibrate.py      结果写到 docs/evidence/render/arnold_*.png
"""
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests_new"))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from splender.addons import render_arnold as ra  # noqa: E402
from splender.engine import ibl  # noqa: E402
from splender.render.engines import RenderRequest  # noqa: E402
from splender.render.pathtracer import PathTracer  # noqa: E402
from splender.render.scene import RenderEnvironment, RenderMaterial, RenderScene, RenderSettings, RenderTexture  # noqa: E402
from test_pathtracer import look_at, sphere_mesh  # noqa: E402

OUT = ROOT / "docs" / "evidence" / "render"


def make_scene(width, height):
    size = 512
    tex = np.zeros((size, size, 4), np.uint8)
    tex[:, :, 3] = 255
    tex[:, : size // 2, 0] = 230              # u < 0.5 红
    tex[:, size // 2:, 2] = 230               # u > 0.5 蓝
    tex[int(size * 0.8):, :, 1] = 230         # v > 0.8（数组第 0 行是 v = 0，所以是靠后的行）绿
    material = RenderMaterial(base_color_tex=RenderTexture(tex, srgb=True), roughness=0.6, metallic=0.0)
    hdri = ibl.list_hdris()[0]
    env = RenderEnvironment(image=ibl.load_hdri(hdri), rotation=0.0, strength=1.0, background=True)
    settings = RenderSettings(width=width, height=height, samples=64, max_bounces=3, view_transform="STANDARD")
    camera = look_at((2.6, 0.9, 2.2), (0, 0, 0), math.tan(math.radians(24)))
    return RenderScene(meshes=[sphere_mesh(64, 32, radius=0.7)], materials=[material], environment=env,
                       camera=camera, settings=settings)


def arnold_render(scene, align):
    ra.SKY_ALIGN = align
    settings = ra.ArnoldSettings()
    settings.camera_aa = 3
    settings.progressive = False
    request = RenderRequest(scene=scene, settings=scene.settings, engine_settings=settings)
    job = ra.ArnoldJob(request)
    while not job.done:
        time.sleep(0.05)
    if job.error:
        raise RuntimeError(job.error)
    return job.preview()


def main():
    width, height = 320, 200
    scene = make_scene(width, height)
    ctx = moderngl.create_standalone_context(require=430)
    tracer = PathTracer(ctx)
    tracer.set_scene(scene)
    while not tracer.finished:
        tracer.render_pass(200)
    ours = tracer.read_image()
    Image.fromarray(ours[:, :, :3]).save(OUT / "arnold_cal_splender.png")
    ra._STATE["license"] = False                 # 标定只看画面，跳过授权检查
    best = None
    background = np.ones((height, width), bool)
    background[40:160, 100:220] = False          # 去掉中间的球，只比背景
    for degrees in (0, 90, 180, 270):
        image = arnold_render(scene, math.radians(degrees))
        a = image[:, :, :3].astype(np.float64)[background]
        b = ours[:, :, :3].astype(np.float64)[background]
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        print("转角 %3d°：背景相关系数 %.3f" % (degrees, corr))
        Image.fromarray(image[:, :, :3]).save(OUT / ("arnold_cal_%d.png" % degrees))
        if best is None or corr > best[0]:
            best = (corr, degrees, image)
    corr, degrees, image = best
    print("最佳转角 %d°（相关系数 %.3f）" % (degrees, corr))
    # 贴图方向：比较球上红、蓝、绿三块区域的位置
    sphere = (slice(50, 150), slice(110, 210))
    for name, img in (("SPLENDER", ours), ("Arnold", image)):
        patch = img[sphere][:, :, :3].astype(int)
        red = patch[:, :, 0] - patch[:, :, 2] > 60
        blue = patch[:, :, 2] - patch[:, :, 0] > 60
        green = patch[:, :, 1] - np.maximum(patch[:, :, 0], patch[:, :, 2]) > 40
        def center(mask):
            ys, xs = np.nonzero(mask)
            return (round(float(xs.mean()), 1), round(float(ys.mean()), 1)) if len(xs) else None
        print("%s：红色中心 %s，蓝色中心 %s，绿色中心 %s" % (name, center(red), center(blue), center(green)))
    tracer.release()


if __name__ == "__main__":
    main()
