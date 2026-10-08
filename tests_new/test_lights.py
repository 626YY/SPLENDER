"""灯光和摄像机物体出图：路径追踪里点光、聚光、日光、面光都照得亮、有阴影、按距离衰减；聚光锥外是黑的；
从摄像机物体出图时视角、偏移和 Blender 一样。没有环境光（纯黑），只靠灯。隐藏的显卡上下文，不开窗口。"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from splender.doc.objects import SceneObject, to_internal  # noqa: E402
from splender.geometry import primitives  # noqa: E402
from splender.render.pathtracer import PathTracer  # noqa: E402
from splender.render.scene import (RenderCamera, RenderEnvironment, RenderMaterial, RenderMesh, RenderScene,  # noqa: E402
                                   RenderSettings, light_from_object)

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/render")
OUT.mkdir(parents=True, exist_ok=True)


def mesh_of(data, offset=(0.0, 0.0, 0.0)):
    positions = data.positions + to_internal(np.asarray(offset, np.float64)).astype(np.float32)
    return RenderMesh(data.name, positions.astype(np.float32), data.normals.astype(np.float32),
                      data.uvs.astype(np.float32), np.zeros(data.triangle_count, np.int32))


def light(kind, location, rotation=(0.0, 0.0, 0.0), **values):
    obj = SceneObject("LIGHT")
    obj.data.type = kind
    obj.transform.location = location
    obj.transform.rotation = rotation
    for name, value in values.items():
        setattr(obj.data, name, value)
    return light_from_object(obj)


class LightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)

    def render(self, lights, camera=None, name="", samples=48, size=(160, 120)):
        scene = RenderScene(settings=RenderSettings(width=size[0], height=size[1], samples=samples, max_bounces=2,
                                                    view_transform="STANDARD"))
        scene.environment = RenderEnvironment(image=None, color=(0.0, 0.0, 0.0), background=True,
                                              background_color=(0.0, 0.0, 0.0))
        scene.materials = [RenderMaterial(base_color=(0.8, 0.8, 0.8), roughness=0.9)]
        floor = primitives.plane(20.0)
        block = primitives.cube(1.0)
        scene.meshes = [mesh_of(floor), mesh_of(block, (0.0, 0.0, 0.5))]
        scene.lights = lights
        if camera is None:
            eye = to_internal(np.array([0.0, -7.0, 6.0]))
            target = to_internal(np.array([0.0, 0.0, 0.0]))
            forward = (target - eye) / np.linalg.norm(target - eye)
            right = np.cross(forward, [0.0, 1.0, 0.0])
            right /= np.linalg.norm(right)
            camera = RenderCamera(eye=eye, forward=forward, up=np.cross(right, forward),
                                  tan_half_y=math.tan(math.radians(30.0)))
        scene.camera = camera
        tracer = PathTracer(self.ctx)
        self.addCleanup(tracer.release)
        tracer.set_scene(scene)
        while not tracer.finished:
            tracer.render_pass(200.0)
        image = tracer.read_image(linear=True)[:, :, :3]
        if name:
            Image.fromarray(np.clip(np.sqrt(np.clip(image, 0, 1)) * 255, 0, 255).astype(np.uint8)).save(OUT / name)
        return image

    def test_point_spot_sun_area(self):
        dark = self.render([], name="light_00_none.png")
        self.assertLess(float(dark.mean()), 1e-4)                  # 没灯没环境：全黑
        point = self.render([light("POINT", (2.0, -1.0, 3.0), power=1000.0)], name="light_01_point.png")
        self.assertGreater(float(point.mean()), 0.02)
        far = self.render([light("POINT", (4.0, -2.0, 6.0), power=1000.0)])
        self.assertGreater(point.mean(), far.mean() * 2.0)          # 远两倍暗很多
        # 阴影：方块挡住的那一侧地面更暗（灯在 +X 上方，阴影落在 -X）
        h, w = point.shape[:2]
        left = point[int(h * 0.55):int(h * 0.7), int(w * 0.2):int(w * 0.35)].mean()
        right = point[int(h * 0.55):int(h * 0.7), int(w * 0.65):int(w * 0.8)].mean()
        self.assertGreater(right, left * 1.3)
        spot = self.render([light("SPOT", (0.0, 0.0, 6.0), power=3000.0, spot_size=20.0, spot_blend=0.0)],
                           name="light_02_spot.png")
        corner = spot[:int(h * 0.15), :int(w * 0.15)].mean()
        center = spot[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)].mean()
        self.assertGreater(center, 0.01)
        self.assertLess(corner, center * 0.05)                    # 锥外是黑的
        sun = self.render([light("SUN", (0.0, 0.0, 10.0), rotation=(30.0, 0.0, 0.0), strength=3.0)],
                          name="light_03_sun.png")
        self.assertGreater(float(sun.mean()), 0.05)
        area = self.render([light("AREA", (0.0, 0.0, 3.0), power=500.0, size=2.0)], name="light_04_area.png")
        self.assertGreater(float(area.mean()), 0.01)
        # 面光只朝一面发光：翻过来朝上照，地面就黑了
        up = self.render([light("AREA", (0.0, 0.0, 3.0), rotation=(180.0, 0.0, 0.0), power=500.0, size=2.0)])
        self.assertLess(float(up.mean()), float(area.mean()) * 0.1)

    def test_camera_object(self):
        cam = SceneObject("CAMERA")
        cam.transform.location = (0.0, -8.0, 3.0)
        cam.transform.rotation = (75.0, 0.0, 0.0)
        cam.data.lens = 35.0
        render_cam = RenderCamera.from_camera_object(cam, 160 / 120)
        np.testing.assert_allclose(render_cam.eye, to_internal(np.array([0.0, -8.0, 3.0])), atol=1e-9)
        # 局部 -Z 是视线：转 75° 后大致朝 +Y 往下看
        forward_display = np.array([render_cam.forward[0], -render_cam.forward[2], render_cam.forward[1]])
        self.assertGreater(forward_display[1], 0.9)
        self.assertLess(forward_display[2], 0.0)
        half_long = 36.0 * 0.5 / 35.0
        self.assertAlmostEqual(render_cam.tan_half_y, half_long / (160 / 120), places=9)
        image = self.render([light("SUN", (0.0, 0.0, 10.0), rotation=(20.0, 10.0, 0.0), strength=3.0)],
                            camera=render_cam, name="light_05_camera.png")
        self.assertGreater(float(image.mean()), 0.03)
        # 镜头偏移：画面整体移动，方块跟着往另一边走
        cam.data.shift_x = 0.2
        shifted = RenderCamera.from_camera_object(cam, 160 / 120)
        _o0, d0 = render_cam.ray(80, 60, 160, 120)
        _o1, d1 = shifted.ray(80, 60, 160, 120)
        self.assertGreater(float(d1 @ render_cam.right), float(d0 @ render_cam.right) + 0.1)


def arnold_found() -> bool:
    try:
        from splender.addons import render_arnold as ra
        return ra.find_arnold() is not None
    except Exception:  # noqa: BLE001
        return False


class ArnoldLightTest(unittest.TestCase):
    """Arnold 也用场景里的灯：只有一盏面光时地面亮、天空黑。"""

    @unittest.skipUnless(arnold_found(), "这台电脑上没有 Arnold")
    def test_arnold_lights(self):
        import time

        from splender.addons import render_arnold as ra
        from splender.render.engines import RenderRequest

        ra._STATE["license"] = False
        settings = RenderSettings(width=160, height=120, samples=16, view_transform="STANDARD")
        scene = RenderScene(settings=settings)
        scene.environment = RenderEnvironment(image=None, color=(0.0, 0.0, 0.0), strength=0.0)
        scene.materials = [RenderMaterial(base_color=(0.8, 0.8, 0.8), roughness=0.9)]
        scene.meshes = [mesh_of(primitives.plane(20.0)), mesh_of(primitives.cube(1.0), (0.0, 0.0, 0.5))]
        eye = to_internal(np.array([0.0, -7.0, 6.0]))
        target = np.zeros(3)
        forward = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        scene.camera = RenderCamera(eye=eye, forward=forward, up=np.cross(right, forward),
                                    tan_half_y=math.tan(math.radians(30.0)))
        results = {}
        for name, lights in (("none", []), ("area", [light("AREA", (0.0, 0.0, 3.0), power=500.0, size=2.0)]),
                             ("point", [light("POINT", (2.0, -1.0, 3.0), power=1000.0)]),
                             ("spot", [light("SPOT", (0.0, 0.0, 6.0), power=3000.0, spot_size=30.0)]),
                             ("sun", [light("SUN", (0.0, 0.0, 10.0), rotation=(30.0, 0.0, 0.0), strength=3.0)])):
            scene.lights = lights
            es = ra.ArnoldSettings()
            es.camera_aa = 2
            es.progressive = False
            job = ra.ArnoldJob(RenderRequest(scene=scene, settings=settings, engine_settings=es))
            while not job.done:
                time.sleep(0.05)
            if job.error:
                raise RuntimeError(job.error)
            image = np.asarray(job.preview(), np.float32)[:, :, :3] / 255.0
            Image.fromarray((image * 255).astype(np.uint8)).save(OUT / ("light_ai_%s.png" % name))
            results[name] = float(image.mean())
        self.assertLess(results["none"], 0.02)
        for name in ("area", "point", "spot", "sun"):
            self.assertGreater(results[name], 0.03, results)


if __name__ == "__main__":
    unittest.main()
