"""从程序当前的工程生成渲染快照。

要在引擎的显卡上下文里调用：图层要先合成成普通贴图（基础色、金属度粗糙度、高度）。
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from ..engine import glx
from ..engine.pagepool import PAGE
from .scene import (RenderCamera, RenderEnvironment, RenderMaterial, RenderMesh, RenderScene, RenderSettings,
                    RenderTexture, light_from_object)

log = logging.getLogger("splender.render")


def composite_textures(engine, ts, size: int, progress=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把纹理集的图层合成成 size×size 的三张贴图：基础色 (S,S,4) uint8 sRGB、金属度粗糙度 (S,S,2) uint8、
    高度 (S,S) float32。第 0 行是 v = 0。size 会按纹理集分辨率取最接近的一级。"""
    engine.complete_strokes(include_active=True)
    state = engine.sets[ts.uid]
    display = state.display
    full = display.size
    size = int(min(max(PAGE, size), full))
    level = max(0, min(display.levels - 1, int(round(math.log2(full / size))))) if size < full else 0
    grid = display.level_size[level]
    size = grid * PAGE
    ctx = engine.ctx
    width = grid * PAGE
    strip_color = ctx.texture((width, PAGE), 4, dtype="f1")
    strip_mrao = ctx.texture((width, PAGE), 4, dtype="f1")
    strip_height = ctx.texture((width, PAGE), 1, dtype="f2")
    fbo = ctx.framebuffer([strip_color, strip_mrao, strip_height])
    base = np.empty((size, size, 4), np.uint8)
    mr = np.empty((size, size, 2), np.uint8)
    height = np.empty((size, size), np.float32)
    compositor = engine.compositor
    try:
        engine._prepare_composite(state, with_stroke=False)
        tx = np.arange(grid, dtype=np.int64)
        for row in range(grid):
            ty = np.full(grid, row, np.int64)
            for _attempt in range(4000):
                table, ready = engine._build_table(state, level, tx, ty, include_stroke=False)
                if ready.all():
                    break
                glx.flush()
                engine.cache.service(engine.cache.frame + 1, budget_ms=50.0)
                time.sleep(0.001)
            else:
                raise RuntimeError("合成渲染贴图时页面一直没有准备好")
            compositor.compose_to(fbo, (width, PAGE), (0, row), tx, ty, table, srgb_encode=True, level_texels=width)
            rows = slice(row * PAGE, (row + 1) * PAGE)
            color = np.frombuffer(fbo.read(components=4, attachment=0, alignment=1), np.uint8)
            base[rows] = color.reshape(PAGE, width, 4)
            mrao = np.frombuffer(fbo.read(components=4, attachment=1, alignment=1), np.uint8).reshape(PAGE, width, 4)
            mr[rows] = mrao[:, :, :2]
            h = np.frombuffer(fbo.read(components=1, attachment=2, dtype="f2", alignment=1), np.float16)
            height[rows] = h.reshape(PAGE, width).astype(np.float32)
            if progress is not None:
                progress((row + 1) / grid)
    finally:
        fbo.release()
        strip_color.release()
        strip_mrao.release()
        strip_height.release()
    return base, mr, height


def _environment(shading) -> RenderEnvironment:
    from ..engine import ibl

    image = None
    try:
        image = ibl.load_hdri(shading.hdri)
    except Exception:  # noqa: BLE001
        log.exception("读取环境图失败，渲染改用纯色环境")
    return RenderEnvironment(image=image, rotation=math.radians(float(shading.hdri_rotation)),
                             strength=float(shading.hdri_strength), background=True)


def build_scene(engine, project, camera, settings: RenderSettings, shading, aspect: float | None = None,
                progress=None, camera_object=None) -> RenderScene:
    """camera 是视口相机（engine/camera.py）；camera_object 给了就从这台摄像机出图（和 Blender 的 F12 一样）；
    shading 提供环境图、旋转、强度。场景里看得见的灯光都放进去。"""
    started = time.perf_counter()
    scene = RenderScene(settings=settings)
    scene.environment = _environment(shading)
    aspect = aspect or settings.width / max(1, settings.height)
    if camera_object is not None:
        scene.camera = RenderCamera.from_camera_object(camera_object, aspect)
    else:
        scene.camera = RenderCamera.from_view_camera(camera, aspect)
    scene.lights = [light_from_object(obj) for obj in getattr(project, "scene_objects", [])
                    if obj.kind == "LIGHT" and obj.visible]
    material_of_set: dict[int, int] = {}
    sets = list(project.texture_sets)
    radius_of_set: dict[int, float] = {}
    for obj in project.objects:
        data = obj.data
        if not getattr(obj, "visible", True) or data is None:
            continue
        lo = np.asarray(data.positions, np.float32).min(axis=0)
        hi = np.asarray(data.positions, np.float32).max(axis=0)
        radius = float(np.linalg.norm(hi - lo) * 0.5) or 1.0
        for set_uid in obj.material_sets:
            radius_of_set[set_uid] = max(radius_of_set.get(set_uid, 0.0), radius)
    for index, ts in enumerate(sets):
        if progress is not None:
            progress(index / max(1, len(sets)))
        base, mr, height = composite_textures(engine, ts, int(settings.texture_size))
        material = RenderMaterial(name=ts.name, base_color=tuple(ts.base_color), metallic=float(ts.base_metallic),
                                  roughness=float(ts.base_roughness), base_color_tex=RenderTexture(base, srgb=True),
                                  mr_tex=RenderTexture(mr), height_tex=RenderTexture(height),
                                  height_scale=0.05 * radius_of_set.get(ts.uid, 1.0) * float(ts.height_scale))
        maps = engine.meshmap(ts.uid) if hasattr(engine, "meshmap") else None
        if maps is not None and maps.has_normal:
            material.normal_tex = RenderTexture(np.ascontiguousarray(maps.read_arrays()["t"][:, :, :3]))
        material_of_set[ts.uid] = len(scene.materials)
        scene.materials.append(material)
    if not scene.materials:
        scene.materials.append(RenderMaterial())
    for obj in project.objects:
        data = obj.data
        if not getattr(obj, "visible", True) or data is None:
            continue
        ids = np.asarray(data.material_ids, np.int32)
        remap = np.zeros(max(1, int(ids.max()) + 1 if len(ids) else 1), np.int32)
        for slot in range(len(remap)):
            set_uid = obj.material_sets[slot] if slot < len(obj.material_sets) else 0
            remap[slot] = material_of_set.get(set_uid, 0)
        scene.meshes.append(RenderMesh(obj.name, np.asarray(data.positions, np.float32),
                                       np.asarray(data.normals, np.float32), np.asarray(data.uvs, np.float32),
                                       remap[np.clip(ids, 0, len(remap) - 1)]))
    log.info("渲染场景准备好：%d 个模型、%d 个材质、%d 个三角形，贴图 %d，用时 %.2f 秒", len(scene.meshes),
             len(scene.materials), scene.triangle_count, int(settings.texture_size), time.perf_counter() - started)
    return scene
