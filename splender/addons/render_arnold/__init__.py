"""Arnold 渲染器插件：用本机安装的 Autodesk Arnold 出图。

随 Maya、3ds Max、Cinema 4D 安装的 Arnold，或单独下载的 Arnold SDK，只要有 bin/ai.dll 和 Python 接口都能用。
场景按 SPLENDER 的渲染快照转换：模型转 polymesh，纹理集转 standard_surface（基础色、金属度、粗糙度贴图，高度贴图做凹凸），
环境图转 skydome_light，视角转 persp_camera。没有 Arnold 授权时成图带水印。
"""
from __future__ import annotations

import ctypes
import glob
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from ...core.props import BoolProperty, EnumProperty, IntProperty, PropertyGroup, StringProperty
from ...render import engines
from ...render.engines import RenderEngine, ThreadedRenderJob

ADDON_INFO = {
    "name": "Arnold 渲染器",
    "description": "用本机安装的 Autodesk Arnold 出图。没有 Arnold 授权时成图带水印",
    "author": "SPLENDER",
    "version": (1, 0, 0),
    "category": "渲染",
    "default_enabled": True,
}

log = logging.getLogger("splender.arnold")
_LOCK = threading.Lock()          # Arnold 一个进程同时只能有一次 AiBegin
_STATE = {"module": None, "root": None, "license": None}


class ArnoldPreferences(PropertyGroup):
    install_dir = StringProperty("Arnold 位置", default="", subtype="DIR_PATH",
                                 description="装有 bin/ai.dll 的 Arnold 文件夹。留空时自动查找本机安装的 Arnold")
    skip_license_check = EnumProperty("授权检查", items=[
        ("AUTO", "自动", "第一次检查后记住结果：没有授权就直接带水印渲染，不再每次等待"),
        ("ALWAYS", "每次都检查", "每次出图前检查授权（约十几秒）"),
        ("NEVER", "不检查", "直接带水印渲染")], default="AUTO")


PREFERENCES = ArnoldPreferences


class ArnoldSettings(PropertyGroup):
    device = EnumProperty("设备", items=[("CPU", "处理器", ""), ("GPU", "显卡", "第一次用显卡渲染要先编译着色器，会等几分钟")],
                          default="CPU")
    camera_aa = IntProperty("抗锯齿采样", default=4, min=1, max=16,
                            description="每个像素的相机采样数是它的平方。越大越干净")
    diffuse_samples = IntProperty("漫反射采样", default=2, min=0, max=10)
    specular_samples = IntProperty("高光采样", default=2, min=0, max=10)
    diffuse_depth = IntProperty("漫反射反弹", default=1, min=0, max=16)
    specular_depth = IntProperty("高光反弹", default=1, min=0, max=16)
    progressive = BoolProperty("先出粗图再细化", default=True, description="先渲一遍低采样的预览，再出最终图")
    threads = IntProperty("线程数", default=0, min=-64, max=256, description="0 用全部核心，负数表示留出几个核心")


# ====================================================================== 找到并载入 Arnold
def _version_of(root: Path) -> tuple:
    header = root / "include" / "arnold" / "ai_version.h"
    try:
        text = header.read_text(encoding="utf-8", errors="ignore")
        parts = [int(re.search(r"define AI_VERSION_%s_NUM\s+(\d+)" % key, text).group(1))
                 for key in ("ARCH", "MAJOR", "MINOR")]
        return tuple(parts)
    except Exception:  # noqa: BLE001
        digits = re.findall(r"\d+", root.name)
        return tuple(int(d) for d in digits) or (0,)


def _python_dir(root: Path) -> Path | None:
    for sub in ("python", "scripts"):
        if (root / sub / "arnold" / "__init__.py").is_file():
            return root / sub
    return None


def candidates(preferred: str = "") -> list[Path]:
    found: list[Path] = []
    roots = [preferred, os.environ.get("ARNOLD_PATH", ""), os.environ.get("ARNOLD_ROOT", "")]
    patterns = ["C:/Program Files/Autodesk/Arnold/*", "C:/Program Files/Autodesk/Arnold-*", "C:/solidangle/*/*",
                "C:/solidangle/*", "C:/Program Files/Maxon/*/plugins/C4DtoA", "C:/Program Files/Arnold*"]
    for pattern in patterns:
        roots.extend(glob.glob(pattern))
    for raw in roots:
        if not raw:
            continue
        root = Path(raw)
        if (root / "bin" / "ai.dll").is_file() and _python_dir(root) is not None and root not in found:
            found.append(root)
    return found


def find_arnold(preferred: str = "") -> Path | None:
    found = candidates(preferred)
    if preferred and found and found[0] == Path(preferred):
        return found[0]
    return max(found, key=_version_of) if found else None


def _preferences() -> ArnoldPreferences | None:
    from ...core import addons

    addon = addons.get(__name__)
    return addon.preferences if addon is not None else None


def load_arnold():
    """载入 Arnold 的 Python 接口（一个进程只载入一次）。"""
    if _STATE["module"] is not None:
        return _STATE["module"]
    prefs = _preferences()
    root = find_arnold(prefs.install_dir if prefs is not None else "")
    if root is None:
        raise RuntimeError("这台电脑上没有找到 Arnold")
    bin_dir = root / "bin"
    os.add_dll_directory(str(bin_dir))
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(_python_dir(root)))
    import arnold  # noqa: PLC0415

    _STATE["module"] = arnold
    _STATE["root"] = root
    log.info("已载入 Arnold %s（%s）", arnold.AiGetVersionString(), root)
    return arnold


# ====================================================================== 场景转换
def _array(arnold, values: np.ndarray, ai_type: int, per_element: int):
    data = np.ascontiguousarray(values)
    count = len(data) // per_element if data.ndim == 1 else len(data)
    return arnold.AiArrayConvert(count, 1, ai_type, data.ctypes.data_as(ctypes.c_void_p))


def _write_png(path: Path, data: np.ndarray) -> str:
    from PIL import Image

    image = np.ascontiguousarray(data[::-1])            # 快照第 0 行是 v = 0；图片第 0 行是顶
    mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[image.shape[2] if image.ndim == 3 else 1]
    if image.ndim == 3 and image.shape[2] == 2:
        image = np.concatenate([image, np.zeros(image.shape[:2] + (1,), image.dtype)], axis=2)
        mode = "RGB"
    Image.fromarray(image if image.ndim == 3 else image, mode).save(str(path))
    return str(path).replace("\\", "/")


def _write_exr(path: Path, data: np.ndarray, flip: bool = True) -> str:
    from ...render.output import save_exr

    image = np.asarray(data, np.float32)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if flip:
        image = image[::-1]
    save_exr(np.ascontiguousarray(image), path)
    return str(path).replace("\\", "/")


def _camera_matrix(arnold, camera):
    right = camera.right
    up = camera.up
    back = -np.asarray(camera.forward, np.float64)
    eye = np.asarray(camera.eye, np.float64)
    return arnold.AtMatrix(*(float(v) for v in (*right, 0.0, *up, 0.0, *back, 0.0, *eye, 1.0)))


def build_universe(arnold, scene, settings, es, folder: Path, skip_license: bool):
    universe = arnold.AiUniverse()
    options = arnold.AiUniverseGetOptions(universe)
    width, height = settings.width, settings.height
    arnold.AiNodeSetInt(options, "xres", int(width))
    arnold.AiNodeSetInt(options, "yres", int(height))
    arnold.AiNodeSetInt(options, "AA_samples", int(es.camera_aa))
    arnold.AiNodeSetInt(options, "GI_diffuse_samples", int(es.diffuse_samples))
    arnold.AiNodeSetInt(options, "GI_specular_samples", int(es.specular_samples))
    arnold.AiNodeSetInt(options, "GI_diffuse_depth", int(es.diffuse_depth))
    arnold.AiNodeSetInt(options, "GI_specular_depth", int(es.specular_depth))
    arnold.AiNodeSetInt(options, "threads", int(es.threads))
    arnold.AiNodeSetStr(options, "render_device", str(es.device))
    arnold.AiNodeSetBool(options, "skip_license_check", bool(skip_license))
    arnold.AiNodeSetBool(options, "abort_on_license_fail", False)
    arnold.AiNodeSetBool(options, "texture_automip", True)
    for name in ("texture_auto_generate_tx",):
        entry = arnold.AiNodeGetNodeEntry(options)
        if arnold.AiNodeEntryLookUpParameter(entry, name):
            arnold.AiNodeSetBool(options, name, False)

    # 相机
    cam = scene.camera
    aspect = width / max(1, height)
    if cam.ortho:
        camera = arnold.AiNode(universe, "ortho_camera", "camera")
        half_w = cam.ortho_half_height * aspect
        arnold.AiNodeSetVec2(camera, "screen_window_min", -half_w, -cam.ortho_half_height)
        arnold.AiNodeSetVec2(camera, "screen_window_max", half_w, cam.ortho_half_height)
    else:
        camera = arnold.AiNode(universe, "persp_camera", "camera")
        arnold.AiNodeSetFlt(camera, "fov", math.degrees(2.0 * math.atan(cam.tan_half_y * aspect)))
    shift = getattr(cam, "shift", (0.0, 0.0))
    if not cam.ortho and any(abs(float(v)) > 1e-9 for v in shift):
        # 镜头偏移：屏幕窗口按偏移平移（单位是半个画面宽）
        half_w = cam.tan_half_y * aspect
        sx, sy = float(shift[0]) / half_w, float(shift[1]) / half_w
        arnold.AiNodeSetVec2(camera, "screen_window_min", -1.0 + sx, -1.0 / aspect + sy)
        arnold.AiNodeSetVec2(camera, "screen_window_max", 1.0 + sx, 1.0 / aspect + sy)
    arnold.AiNodeSetMatrix(camera, "matrix", _camera_matrix(arnold, cam))
    arnold.AiNodeSetFlt(camera, "near_clip", float(cam.clip_near))
    arnold.AiNodeSetFlt(camera, "far_clip", float(cam.clip_far))
    arnold.AiNodeSetPtr(options, "camera", camera)

    # 材质
    shaders = []
    for index, material in enumerate(scene.materials):
        surface = arnold.AiNode(universe, "standard_surface", "material_%d" % index)
        arnold.AiNodeSetFlt(surface, "base", 1.0)
        arnold.AiNodeSetFlt(surface, "specular", 1.0)
        color = np.asarray(material.base_color, np.float64)
        linear = np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)
        arnold.AiNodeSetRGB(surface, "base_color", *(float(v) for v in linear))
        arnold.AiNodeSetFlt(surface, "metalness", float(material.metallic))
        arnold.AiNodeSetFlt(surface, "specular_roughness", float(material.roughness))
        if material.base_color_tex is not None:
            image = arnold.AiNode(universe, "image", "base_%d" % index)
            arnold.AiNodeSetStr(image, "filename", _write_png(folder / ("base_%d.png" % index),
                                                              material.base_color_tex.data[:, :, :3]))
            arnold.AiNodeSetStr(image, "color_space", "sRGB")
            arnold.AiNodeLink(image, "base_color", surface)
        if material.mr_tex is not None:
            image = arnold.AiNode(universe, "image", "mr_%d" % index)
            arnold.AiNodeSetStr(image, "filename", _write_png(folder / ("mr_%d.png" % index), material.mr_tex.data))
            arnold.AiNodeSetStr(image, "color_space", "Raw")
            arnold.AiNodeLinkOutput(image, "r", surface, "metalness")
            arnold.AiNodeLinkOutput(image, "g", surface, "specular_roughness")
        normal_map = None
        if getattr(material, "normal_tex", None) is not None:
            # 从高模烘的切线空间法线：T 沿 dP/du、B 沿 dP/dv，和 Arnold 的切线空间一致
            image = arnold.AiNode(universe, "image", "nmap_%d" % index)
            arnold.AiNodeSetStr(image, "filename", _write_png(folder / ("nmap_%d.png" % index),
                                                              material.normal_tex.data[:, :, :3]))
            arnold.AiNodeSetStr(image, "color_space", "Raw")
            normal_map = arnold.AiNode(universe, "normal_map", "normal_map_%d" % index)
            arnold.AiNodeLink(image, "input", normal_map)
        if material.height_tex is not None and material.height_scale > 0.0 and \
                float(np.ptp(material.height_tex.data)) > 1e-6:
            image = arnold.AiNode(universe, "image", "height_%d" % index)
            arnold.AiNodeSetStr(image, "filename", _write_exr(folder / ("height_%d.exr" % index),
                                                              material.height_tex.data))
            arnold.AiNodeSetStr(image, "color_space", "Raw")
            bump = arnold.AiNode(universe, "bump2d", "bump_%d" % index)
            arnold.AiNodeLinkOutput(image, "r", bump, "bump_map")
            arnold.AiNodeSetFlt(bump, "bump_height", float(material.height_scale))
            if normal_map is not None:
                arnold.AiNodeLink(normal_map, "normal", bump)      # 高度凹凸叠在法线贴图上
            arnold.AiNodeLink(bump, "normal", surface)
        elif normal_map is not None:
            arnold.AiNodeLink(normal_map, "normal", surface)
        shaders.append(surface)

    # 模型
    for index, mesh in enumerate(scene.meshes):
        node = arnold.AiNode(universe, "polymesh", "mesh_%d" % index)
        tri_count = mesh.triangle_count
        corners = tri_count * 3
        arnold.AiNodeSetArray(node, "nsides", _array(arnold, np.full(tri_count, 3, np.uint32), arnold.AI_TYPE_UINT, 1))
        indices = np.arange(corners, dtype=np.uint32)
        arnold.AiNodeSetArray(node, "vidxs", _array(arnold, indices, arnold.AI_TYPE_UINT, 1))
        arnold.AiNodeSetArray(node, "vlist", _array(arnold, np.asarray(mesh.positions, np.float32),
                                                    arnold.AI_TYPE_VECTOR, 3))
        arnold.AiNodeSetArray(node, "nidxs", _array(arnold, indices, arnold.AI_TYPE_UINT, 1))
        arnold.AiNodeSetArray(node, "nlist", _array(arnold, np.asarray(mesh.normals, np.float32),
                                                    arnold.AI_TYPE_VECTOR, 3))
        arnold.AiNodeSetArray(node, "uvidxs", _array(arnold, indices, arnold.AI_TYPE_UINT, 1))
        arnold.AiNodeSetArray(node, "uvlist", _array(arnold, np.asarray(mesh.uvs, np.float32),
                                                     arnold.AI_TYPE_VECTOR2, 2))
        arnold.AiNodeSetBool(node, "smoothing", True)
        transform = np.asarray(mesh.transform, np.float64).T.ravel()      # Arnold 用行向量，平移在最后一行
        arnold.AiNodeSetMatrix(node, "matrix", arnold.AtMatrix(*(float(v) for v in transform)))
        ids = np.clip(np.asarray(mesh.material_ids, np.int64), 0, max(0, len(shaders) - 1))
        used = sorted(set(ids.tolist())) if len(ids) else [0]
        if len(used) == 1:
            arnold.AiNodeSetPtr(node, "shader", shaders[used[0]])
        else:
            lookup = {m: i for i, m in enumerate(used)}
            arnold.AiNodeSetArray(node, "shidxs", _array(arnold, np.asarray([lookup[m] for m in ids], np.uint8),
                                                         arnold.AI_TYPE_BYTE, 1))
            shader_array = arnold.AiArrayAllocate(len(used), 1, arnold.AI_TYPE_NODE)
            for slot, material_index in enumerate(used):
                arnold.AiArraySetPtr(shader_array, slot, shaders[material_index])
            arnold.AiNodeSetArray(node, "shader", shader_array)

    # 环境
    env = scene.environment
    sky = arnold.AiNode(universe, "skydome_light", "sky")
    arnold.AiNodeSetFlt(sky, "intensity", float(env.strength))
    arnold.AiNodeSetStr(sky, "format", "latlong")
    arnold.AiNodeSetInt(sky, "resolution", 2048)
    if env.image is not None:
        image = arnold.AiNode(universe, "image", "sky_image")
        arnold.AiNodeSetStr(image, "filename", _write_exr(folder / "sky.exr", env.image, flip=False))
        arnold.AiNodeSetStr(image, "color_space", "linear")
        arnold.AiNodeLink(image, "color", sky)
    else:
        arnold.AiNodeSetRGB(sky, "color", *(float(v) for v in env.color))
    # SPLENDER 的环境图方向约定和 Arnold 的经纬图差一个绕 Y 轴的转角；再叠加用户设的旋转
    angle = SKY_ALIGN + float(env.rotation)
    c, s = math.cos(angle), math.sin(angle)
    arnold.AiNodeSetMatrix(sky, "matrix", arnold.AtMatrix(c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1))
    if not env.background or settings.transparent:
        arnold.AiNodeSetFlt(sky, "camera", 0.0)                    # 相机看不到，照明照常

    # 灯光（和 Blender 的数值一样：点光、聚光、面光按瓦，日光按瓦每平方米）
    for index, light in enumerate(getattr(scene, "lights", []) or []):
        _arnold_light(arnold, universe, index, light)

    # 输出
    driver = arnold.AiNode(universe, "driver_exr", "driver")
    output_path = str(folder / "beauty.exr").replace("\\", "/")
    arnold.AiNodeSetStr(driver, "filename", output_path)
    arnold.AiNodeSetBool(driver, "half_precision", False)
    image_filter = arnold.AiNode(universe, "gaussian_filter", "filter")
    arnold.AiNodeSetFlt(image_filter, "width", 2.0)
    outputs = arnold.AiArrayAllocate(1, 1, arnold.AI_TYPE_STRING)
    arnold.AiArraySetStr(outputs, 0, "RGBA RGBA filter driver")
    arnold.AiNodeSetArray(options, "outputs", outputs)
    return universe, options, output_path


def _arnold_light(arnold, universe, index: int, light) -> None:
    kind = str(light.type)
    direction = np.asarray(light.direction, np.float64)
    if kind == "SUN":
        node = arnold.AiNode(universe, "distant_light", "light_%d" % index)
        arnold.AiNodeSetFlt(node, "intensity", float(light.radiance))
        arnold.AiNodeSetFlt(node, "angle", math.degrees(float(light.angle)))
    elif kind == "SPOT":
        node = arnold.AiNode(universe, "spot_light", "light_%d" % index)
        arnold.AiNodeSetFlt(node, "intensity", float(light.radiance))
        arnold.AiNodeSetFlt(node, "cone_angle", math.degrees(float(light.spot_size)))
        arnold.AiNodeSetFlt(node, "penumbra_angle", math.degrees(float(light.spot_size)) * float(light.spot_blend) * 0.5)
        arnold.AiNodeSetFlt(node, "radius", float(light.radius))
    elif kind == "AREA":
        node = arnold.AiNode(universe, "quad_light", "light_%d" % index)
        su, sv = (float(v) * 0.5 for v in light.size)
        # 顶点顺时针排（从 +Z 看）：Arnold 的面光朝顶点顺序的法线一侧发光，这样朝局部 -Z 照
        verts = np.array([[-su, -sv, 0.0], [-su, sv, 0.0], [su, sv, 0.0], [su, -sv, 0.0]], np.float32)
        arnold.AiNodeSetArray(node, "vertices", _array(arnold, verts, arnold.AI_TYPE_VECTOR, 3))
        arnold.AiNodeSetFlt(node, "intensity", float(light.radiance))
    else:
        node = arnold.AiNode(universe, "point_light", "light_%d" % index)
        arnold.AiNodeSetFlt(node, "intensity", float(light.radiance))
        arnold.AiNodeSetFlt(node, "radius", float(light.radius))
    arnold.AiNodeSetRGB(node, "color", *(float(c) for c in light.color))
    if kind == "AREA":
        # 面光的 intensity 直接当面上的辐亮度用；点光、聚光、日光用默认的归一化（数值就是发光强度、照度，
        # 和半径、角直径无关）
        arnold.AiNodeSetBool(node, "normalize", False)
    arnold.AiNodeSetBool(node, "cast_shadows", bool(light.shadow))
    # 灯的局部 -Z 是照射方向；面光的局部 X、Y 是两条边
    back = -direction / max(np.linalg.norm(direction), 1e-12)
    u = np.asarray(light.axis_u, np.float64)
    u = u - back * float(u @ back)
    if np.linalg.norm(u) < 1e-9:
        u = np.cross([0.0, 1.0, 0.0], back) if abs(back[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(back, u)
    p = np.asarray(light.position, np.float64)
    arnold.AiNodeSetMatrix(node, "matrix", arnold.AtMatrix(*(float(x) for x in (*u, 0.0, *v, 0.0, *back, 0.0,
                                                                                    *p, 1.0))))


#: SPLENDER 环境图约定（u = 0.5 + atan2(z, x) / 2π）转到 Arnold 经纬图需要的绕 Y 转角（弧度）。
#: 由 tools/arnold_calibrate.py 对照渲染标定：转 90° 时背景相关系数 0.991，用户旋转同号叠加时 0.992
SKY_ALIGN = math.pi / 2.0


def read_exr(path: str) -> np.ndarray:
    from ...paths import ensure_vendor_path

    ensure_vendor_path()
    import OpenEXR

    with OpenEXR.File(path) as handle:
        channels = handle.channels()
        if "RGBA" in channels:
            data = np.asarray(channels["RGBA"].pixels, np.float32)
        elif "RGB" in channels:
            rgb = np.asarray(channels["RGB"].pixels, np.float32)
            data = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], axis=2)
        else:
            planes = [np.asarray(channels[name].pixels, np.float32) for name in ("R", "G", "B")]
            alpha = np.asarray(channels["A"].pixels, np.float32) if "A" in channels else np.ones_like(planes[0])
            data = np.stack(planes + [alpha], axis=2)
    return np.ascontiguousarray(data)


class ArnoldJob(ThreadedRenderJob):
    def __init__(self, request) -> None:
        self._rs = None
        self._arnold = None
        super().__init__(request)

    def cancel(self) -> None:
        super().cancel()
        if self._rs is not None and self._arnold is not None:
            try:
                self._arnold.AiRenderAbort(self._rs, self._arnold.AI_NON_BLOCKING)
            except Exception:  # noqa: BLE001
                pass

    def _show(self, path: str, final: bool) -> None:
        from ...engine import ibl

        linear = read_exr(path)
        settings = self.request.settings
        srgb = ibl.to_uint8(ibl.tonemap(linear[:, :, :3], str(settings.view_transform).upper(), settings.exposure))
        alpha = ibl.to_uint8(np.clip(linear[:, :, 3:4], 0.0, 1.0))
        self.set_preview(np.ascontiguousarray(np.concatenate([srgb, alpha], axis=2)))
        if final:
            self.set_result(linear)

    def run(self) -> None:
        self.status = "载入 Arnold"
        arnold = load_arnold()
        self._arnold = arnold
        request = self.request
        es = request.engine_settings or ArnoldSettings()
        prefs = _preferences()
        mode = prefs.skip_license_check if prefs is not None else "AUTO"
        skip = mode == "NEVER" or (mode == "AUTO" and _STATE["license"] is False)
        folder = Path(tempfile.mkdtemp(prefix="splender_arnold_"))
        universe = None
        with _LOCK:
            arnold.AiBegin(arnold.AI_SESSION_BATCH)
            try:
                arnold.AiMsgSetConsoleFlags(None, arnold.AI_LOG_NONE)
                arnold.AiMsgSetLogFileName(str(folder / "arnold.log"))
                arnold.AiMsgSetLogFileFlags(None, arnold.AI_LOG_ALL)
                if not skip and (mode == "ALWAYS" or _STATE["license"] is None):
                    self.status = "检查 Arnold 授权"
                    try:
                        _STATE["license"] = bool(arnold.AiLicenseIsAvailable(None))
                    except TypeError:
                        _STATE["license"] = bool(arnold.AiLicenseIsAvailable())
                    skip = not _STATE["license"]          # 没有授权：直接带水印渲染，不再让 Arnold 自己再等一遍
                self.status = "转换场景"
                universe, options, output = build_universe(arnold, request.scene, request.settings, es, folder, skip)
                self._rs = arnold.AiRenderSession(universe, arnold.AI_SESSION_BATCH)
                passes = [-3, int(es.camera_aa)] if es.progressive and int(es.camera_aa) > 1 else [int(es.camera_aa)]
                for number, aa in enumerate(passes):
                    if self.cancelled:
                        break
                    arnold.AiNodeSetInt(options, "AA_samples", aa)
                    final = number == len(passes) - 1
                    self.status = "Arnold 渲染中%s" % ("（预览）" if not final else "")
                    started = time.perf_counter()
                    code = arnold.AiRender(self._rs, arnold.AI_RENDER_MODE_CAMERA)
                    if self.cancelled:
                        break
                    if int(getattr(code, "value", code)) != 0:
                        raise RuntimeError("Arnold 渲染失败（错误码 %s），日志：%s" % (code, folder / "arnold.log"))
                    self._show(output, final)
                    self.progress = (number + 1) / len(passes)
                    log.info("Arnold 第 %d 遍（采样 %d）用时 %.1f 秒", number + 1, aa, time.perf_counter() - started)
                licensed = _STATE["license"]
                self.status = "完成" + ("（未授权，带水印）" if licensed is False or skip else "")
            finally:
                rs, self._rs = self._rs, None
                if rs is not None:
                    arnold.AiRenderSessionDestroy(rs)
                if universe is not None:
                    try:
                        arnold.AiUniverseDestroy(universe)
                    except Exception:  # noqa: BLE001
                        pass
                arnold.AiEnd()


class ArnoldEngine(RenderEngine):
    idname = "ARNOLD"
    label = "Arnold"
    description = "Autodesk Arnold：影视级渲染器，用本机安装的 Arnold"
    icon = "render"
    order = 50
    settings_class = ArnoldSettings
    supports_viewport = False

    @classmethod
    def available(cls) -> tuple[bool, str]:
        prefs = _preferences()
        if find_arnold(prefs.install_dir if prefs is not None else "") is None:
            return False, "这台电脑上没有找到 Arnold，可以在插件设置里指定位置"
        return True, ""

    @classmethod
    def create_job(cls, request):
        return ArnoldJob(request)


def register() -> None:
    engines.register_engine(ArnoldEngine)


def unregister() -> None:
    engines.unregister_engine(ArnoldEngine.idname)
