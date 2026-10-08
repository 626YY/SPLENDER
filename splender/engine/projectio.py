"""工程的保存、打开，以及贴图导出。都需要引擎的显卡上下文是当前上下文。"""
from __future__ import annotations

import io
import logging
import os
import time

import numpy as np

from ..doc.meshio import MeshData
from ..doc.project import (EXPORT_LABELS, MeshObject, PLANE_FORMAT, Project, TextureSet, reserve_uid)
from ..doc.storage import ProjectFile
from . import glx
from .cache import Page
from .display import LAYER_STRIDE, MAX_LAYERS
from .layerstore import LayerPixels
from .pagepool import PAGE

log = logging.getLogger("splender.engine.io")

FORMAT_VERSION = 1


# ---------------------------------------------------------------- 模型数据
def mesh_to_bytes(mesh: MeshData) -> bytes:
    buffer = io.BytesIO()
    extra = {}
    sizes = getattr(mesh, "polygon_sizes", None)
    if sizes is not None:
        extra["polygon_sizes"] = np.asarray(sizes, np.int32)
    loose = getattr(mesh, "loose_edges", None)
    if loose is not None and len(loose):
        extra["loose_edges"] = np.asarray(loose, np.float32)
    np.savez_compressed(buffer, positions=mesh.positions, normals=mesh.normals, uvs=mesh.uvs,
                        material_ids=mesh.material_ids, materials=np.array(mesh.materials, dtype=object),
                        name=np.array(mesh.name), has_uvs=np.array(bool(getattr(mesh, "has_uvs", True))), **extra)
    return buffer.getvalue()


def mesh_from_bytes(data: bytes) -> MeshData:
    with np.load(io.BytesIO(data), allow_pickle=True) as z:
        positions = np.ascontiguousarray(z["positions"], np.float32)
        mesh = MeshData(name=str(z["name"]), positions=positions,
                        normals=np.ascontiguousarray(z["normals"], np.float32),
                        uvs=np.ascontiguousarray(z["uvs"], np.float32),
                        material_ids=np.ascontiguousarray(z["material_ids"], np.int32),
                        materials=[str(m) for m in z["materials"].tolist()],
                        bounds_min=positions.min(axis=0) if len(positions) else np.zeros(3, np.float32),
                        bounds_max=positions.max(axis=0) if len(positions) else np.zeros(3, np.float32))
        mesh.has_uvs = bool(z["has_uvs"]) if "has_uvs" in z else True
        mesh.polygon_sizes = np.asarray(z["polygon_sizes"], np.int32) if "polygon_sizes" in z else None
        mesh.loose_edges = np.asarray(z["loose_edges"], np.float32) if "loose_edges" in z else None
    return mesh


# ---------------------------------------------------------------- 保存
def save_project(engine, project: Project, path: str, extra: dict | None = None, progress=None) -> dict:
    """把工程写进 path。已经打开的同一个文件只写改过的页；换了路径就整份写。"""
    started = time.perf_counter()
    cache = engine.cache
    storage: ProjectFile | None = cache.storage
    same_file = storage is not None and os.path.abspath(str(getattr(storage, "path", ""))) == os.path.abspath(path)
    if not same_file:
        if storage is not None:
            # 另存为：先把现有文件一致地复制过去，再在副本上写增量
            storage.save_copy(path)
            storage.close()
            storage = ProjectFile(path)
        else:
            if os.path.exists(path):
                os.remove(path)
            storage = ProjectFile(path, create=True)
        cache.storage = storage

    assets = set(storage.list_assets())
    used = set()
    for obj in project.objects:
        name = "mesh/%d.npz" % obj.uid
        used.add(name)
        if name not in assets or getattr(obj, "geometry_dirty", False):
            storage.write_asset(name, mesh_to_bytes(obj.data))
            obj.geometry_dirty = False
    set_uids = {ts.uid for ts in project.texture_sets}
    for name in assets:
        if name.startswith("mesh/") and name not in used:
            storage.delete_asset(name)
        elif name.startswith("meshmaps/"):
            try:
                uid = int(name.split("/", 1)[1].split(".", 1)[0])
            except ValueError:
                continue
            if uid not in set_uids and uid not in getattr(engine, "removed_sets", {}):
                storage.delete_asset(name)
    for ts in project.texture_sets:
        name = "meshmaps/%d.bin" % ts.uid
        maps = engine.meshmaps.get(ts.uid)
        if maps is None:
            if name in assets:
                storage.delete_asset(name)
        elif not maps.saved or name not in assets or not same_file:
            storage.write_asset(name, maps.to_bytes())
            maps.saved = True

    items: list[tuple[tuple, Page]] = []
    keep: set[tuple] = set()
    owner: dict[tuple, Page] = {}
    parked = [state.ts for state in getattr(engine, "removed_sets", {}).values()]
    for ts in list(project.texture_sets) + parked:
        for layer in ts.layers:
            store = engine.layers.stores.get(layer.uid)
            if store is None:
                continue
            for plane, mip, tx, ty, page in store.pages():
                key = (layer.uid, plane, mip, tx, ty)
                keep.add(key)
                owner[key] = page
                if page.dirty or page.disk_key != key:
                    items.append((key, page))
    # 撤销历史里的旧页如果只剩文件里那一份，写新页之前先把它读进内存，免得被覆盖后丢失
    writing = {key for key, _page in items}
    for page in list(_all_history_pages(engine)):
        if page.disk_key is not None and page.disk_key in writing and owner.get(page.disk_key) is not page:
            if page.blob is None and page.scratch is None and page.slot < 0:
                data = storage.read_page(page.disk_key)
                if data is not None:
                    cache.adopt_bytes(page, data)
            cache.forget_disk(page)
    delete = [key for key in storage.page_keys() if key not in keep]
    total = len(items)

    def generate():
        chunk = 48
        for start in range(0, total, chunk):
            if progress is not None:
                progress(start / max(1, total))
            part = items[start:start + chunk]
            for (key, _page), data in zip(part, cache.fetch_bytes([page for _key, page in part])):
                yield key, data

    storage.write_pages(generate(), delete=delete)
    for key, page in items:
        cache.mark_saved(page, key)
    meta = {}
    try:
        meta = dict(storage.read_meta() or {})      # 保留文件里已有的其他内容（界面布局等）
    except Exception:  # noqa: BLE001
        meta = {}
    meta.update({"format": FORMAT_VERSION, "project": project.to_dict()})
    if extra:
        meta.update(extra)
    storage.write_meta(meta)
    project.path = path
    project.name = os.path.splitext(os.path.basename(path))[0]
    project.mark_dirty(False)
    project.changed.emit("path")
    elapsed = time.perf_counter() - started
    log.info("已保存 %s：写入 %d 页，删除 %d 页，用时 %.2f 秒", path, total, len(delete), elapsed)
    return {"pages": total, "deleted": len(delete), "seconds": elapsed}


def _all_history_pages(engine):
    history = engine.history
    if history is None:
        return
    for step in history.steps:
        records = getattr(step, "records", None)
        if not records:
            continue
        yield from LayerPixels.record_pages(records)


# ---------------------------------------------------------------- 打开
def load_project(engine, path: str) -> tuple[Project, dict]:
    """打开工程文件。贴图页面不在这里读，之后看到哪读到哪。"""
    storage = ProjectFile(path)
    meta = storage.read_meta()
    data = meta.get("project", {})
    project = Project(os.path.splitext(os.path.basename(path))[0])
    project.path = path
    for item in data.get("texture_sets", []):
        project.add_texture_set(TextureSet.from_saved(item))
    project.active_set_uid = int(data.get("active_set", 0)) or (project.texture_sets[0].uid if project.texture_sets else 0)
    project.render.from_saved(data.get("render"))
    from ..nodes.graph import NodeGraph
    for item in data.get("graphs", []):
        try:
            project.add_graph(NodeGraph.from_dict(item))
        except Exception:  # noqa: BLE001  一张坏图不该让整个工程打不开
            log.exception("读取节点图出错：%s", item.get("name") if isinstance(item, dict) else item)
    project.active_graph_uid = int(data.get("active_graph", 0) or 0)
    for item in data.get("objects", []):
        raw = storage.read_asset("mesh/%d.npz" % int(item["uid"]))
        if raw is None:
            log.warning("工程里缺少模型数据：%s", item.get("name"))
            continue
        obj = MeshObject(item.get("name", "模型"), mesh_from_bytes(raw), [int(v) for v in item.get("material_sets", [])])
        obj.uid = int(item["uid"])
        obj.visible = bool(item.get("visible", True))
        obj.select = bool(item.get("select", False))
        obj.transform.from_dict(item.get("transform") or {})
        reserve_uid(obj.uid)
        project.objects.append(obj)
    from ..doc.objects import SceneObject
    for item in data.get("scene_objects", []):
        try:
            project.scene_objects.append(SceneObject.from_dict(item))
        except Exception:  # noqa: BLE001
            log.exception("读取物体出错：%s", item.get("name") if isinstance(item, dict) else item)
    project.active_object_uid = int(data.get("active_object", 0) or 0)
    project.active_camera_uid = int(data.get("active_camera", 0) or 0)
    cursor = data.get("cursor")
    if isinstance(cursor, (list, tuple)) and len(cursor) == 3:
        project.cursor_location = tuple(float(v) for v in cursor)
    old_storage = engine.cache.storage
    engine.set_project(project)
    engine.cache.storage = storage
    for ts in project.texture_sets:
        raw = storage.read_asset("meshmaps/%d.bin" % ts.uid)
        if raw is None:
            continue
        try:
            from ..bake.baker import MeshMapSet

            engine.set_meshmap(ts.uid, MeshMapSet.from_bytes(engine.ctx, raw))
        except Exception:  # noqa: BLE001
            log.exception("读取模型贴图出错：%s", ts.name)
    if old_storage is not None and old_storage is not storage:
        try:
            old_storage.close()
        except Exception:  # noqa: BLE001
            log.exception("关闭上一个工程文件时出错")
    layer_grid = {layer.uid: ts.size // PAGE for ts in project.texture_sets for layer in ts.layers}
    count = 0
    for key in storage.page_keys():
        layer_uid, plane, mip, tx, ty = key
        grid = layer_grid.get(layer_uid)
        if grid is None or plane not in PLANE_FORMAT:
            continue
        store = engine.layers.store(layer_uid, grid).plane(plane)
        if mip >= len(store.levels):
            continue
        page = Page(store.fmt)
        page.disk_key = key
        page.dirty = False
        engine.cache.set_page(store.levels[mip], tx, ty, page)
        count += 1
    for state in engine.sets.values():
        state.display.mark_all_dirty()
    project.dirty = False
    log.info("已打开 %s：%d 页（按需读取）", path, count)
    return project, meta


# ---------------------------------------------------------------- 导出
EXPORT_CHANNELS = {
    # 通道: (显示贴图序号, 分量下标列表, 位深)；法线另有专门的写法（export_normal）
    "basecolor": (0, (0, 1, 2), 8),
    "metallic": (1, (0,), 8),
    "roughness": (1, (1,), 8),
    "height": (2, (0,), 16),
    "ao": (1, (2,), 8),
    "orm": (1, (2, 1, 0), 8),         # 打包：R 环境遮蔽、G 粗糙度、B 金属度（glTF、Unreal 的用法）
}


def export_file_name(pattern: str, project: str, texture_set: str, channel: str) -> str:
    """按命名规则得出导出文件名（带 .png），去掉文件名里不能用的字符。"""
    text = pattern or "{工程}_{纹理集}_{通道}"
    for keys, value in ((("{工程}", "{project}"), project), (("{纹理集}", "{set}"), texture_set),
                        (("{通道}", "{channel}"), channel)):
        for key in keys:
            text = text.replace(key, value)
    bad = set('/:*?"<>|' + chr(92))
    text = "".join("_" if (ch in bad or ord(ch) < 32) else ch for ch in text).strip(" .")
    return (text or channel) + ".png"


def compose_row(engine, state, level: int, row: int, grid: int, fbo, width: int, srgb_encode: bool) -> None:
    """把这一级的第 row 行格子合成进 fbo（三个附件：颜色、金属粗糙遮蔽、高度）。页面没进显存就等它进来。"""
    tx = np.arange(grid, dtype=np.int64)
    ty = np.full(grid, row, np.int64)
    for _attempt in range(2000):
        table, ready = engine._build_table(state, level, tx, ty, include_stroke=False)
        if ready.all():
            break
        glx.flush()
        engine.cache.service(engine.cache.frame + 1, budget_ms=50.0)
        time.sleep(0.001)
    else:
        raise RuntimeError("导出时页面一直没有准备好")
    engine.compositor.compose_to(fbo, (width, PAGE), (0, row), tx, ty, table, srgb_encode=srgb_encode,
                                 level_texels=width)


def composite_image(engine, ts: TextureSet, size: int = 256, below=None, dtype=np.float32) -> np.ndarray:
    """合成结果的小图：sRGB 基础色，(边长, 边长, 3) 的 0..1 浮点（边长取不超过 size 的那一级）；dtype 给 np.uint8 时是 0..255。
    below：只看这一层下面的结果（调整层的直方图、自动色阶用）；它在文件夹里时按去掉它的整体结果算。"""
    state = engine.sets[ts.uid]
    display = state.display
    full = display.size
    level = max(0, min(display.levels - 1, int(np.ceil(np.log2(full / max(1, size)))))) if size < full else 0
    grid = display.level_size[level]
    width = grid * PAGE
    ctx = engine.ctx
    strip_color = ctx.texture((width, PAGE), 4, dtype="f1")
    strip_mrao = ctx.texture((width, PAGE), 4, dtype="f1")
    strip_height = ctx.texture((width, PAGE), 1, dtype="f2")
    fbo = ctx.framebuffer([strip_color, strip_mrao, strip_height])
    as_bytes = np.dtype(dtype) == np.uint8
    out = np.zeros((width, width, 3), np.uint8 if as_bytes else np.float32)
    try:
        engine._prepare_composite(state, with_stroke=False)
        if below is not None and below in ts.layers:
            params = state.params.copy()
            index = ts.layers.index(below)
            count = min(len(ts.layers), MAX_LAYERS)
            if ts.depth(below) == 0:
                count = min(index, MAX_LAYERS)
            elif index < MAX_LAYERS:
                params["opacity"][index][0] = 0.0
            engine.compositor.set_layers(params, count, ts.base_color, (ts.base_metallic, ts.base_roughness))
        for row in range(grid):
            compose_row(engine, state, level, row, grid, fbo, width, srgb_encode=True)
            raw = np.frombuffer(fbo.read(components=4, attachment=0, dtype="f1", alignment=1), np.uint8)
            rgb = raw.reshape(PAGE, width, 4)[:, :, :3]
            out[row * PAGE:(row + 1) * PAGE] = rgb if as_bytes else rgb * np.float32(1.0 / 255.0)
    finally:
        for item in (fbo, strip_color, strip_mrao, strip_height):
            item.release()
    return out


def export_channel(engine, ts: TextureSet, channel: str, path: str, progress=None, size: int | None = None) -> dict:
    """把一个通道导出成 PNG。逐行合成、逐行写出，不把整张图放进内存。"""
    from .export_png import PngStreamWriter

    if channel in ("normal", "normal_gl"):            # normal_gl：一律 OpenGL 格式（glTF 用）
        from .export_normal import export_normal
        return export_normal(engine, ts, path, progress, size, opengl=channel == "normal_gl")
    started = time.perf_counter()
    state = engine.sets[ts.uid]
    display = state.display
    full = display.size
    size = int(size or full)
    level = max(0, min(display.levels - 1, int(round(np.log2(full / size))))) if size < full else 0
    size = full >> level
    grid = display.level_size[level]
    target, comps, depth = EXPORT_CHANNELS[channel]
    ctx = engine.ctx
    width = grid * PAGE
    strip_color = ctx.texture((width, PAGE), 4, dtype="f1")
    strip_mrao = ctx.texture((width, PAGE), 4, dtype="f1")
    strip_height = ctx.texture((width, PAGE), 1, dtype="f2")
    fbo = ctx.framebuffer([strip_color, strip_mrao, strip_height])
    writer = PngStreamWriter(path, size, size, len(comps), bit_depth=depth)
    try:
        engine._prepare_composite(state, with_stroke=False, ao_mix=1.0 if channel in ("ao", "orm") else None)
        for row in range(grid - 1, -1, -1):           # 图片第一行对应 v=1
            compose_row(engine, state, level, row, grid, fbo, width, srgb_encode=(channel == "basecolor"))
            if target == 2:
                raw = np.frombuffer(fbo.read(components=1, attachment=2, dtype="f2", alignment=1), np.float16)
                rows = raw.reshape(PAGE, width, 1)[::-1].astype(np.float32)
                data = np.clip(np.rint((rows * 0.5 + 0.5) * 65535.0), 0, 65535).astype(np.uint16)
            else:
                raw = np.frombuffer(fbo.read(components=4, attachment=target, alignment=1), np.uint8)
                data = raw.reshape(PAGE, width, 4)[::-1][:, :, list(comps)]
            writer.write_rows(np.ascontiguousarray(data))
            if progress is not None:
                progress((grid - row) / grid)
        writer.close()
    except Exception:
        writer.abort()
        raise
    finally:
        fbo.release()
        strip_color.release()
        strip_mrao.release()
        strip_height.release()
    elapsed = time.perf_counter() - started
    log.info("已导出 %s（%s，%d×%d），用时 %.2f 秒", path, EXPORT_LABELS.get(channel, channel), size, size, elapsed)
    return {"path": path, "size": size, "seconds": elapsed}
