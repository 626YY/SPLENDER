"""雕刻会话：进入雕刻模式时把物体的网格交给显卡雕刻引擎；退出（或存盘）时把形状写回物体，绘制引擎重建几何。

雕刻模式里的每一笔是一步撤销（按块恢复）。退出雕刻模式时，这些步骤合成一步「雕刻」：记下进入前后的整个形状，
回到绘制模式后也能撤销整段雕刻。
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque

import numpy as np

from . import shaders as S
from .engine import BRUSH_INDEX, DAB_DTYPE, SculptEngine
from .remesh import grid_for, voxel_remesh
from .topology import SculptMesh, subdivide, unweld, weld

log = logging.getLogger("splender.sculpt")

SOLID_MODES = {"STUDIO": 0, "MATCAP": 1, "FLAT": 2}


def snapshot_mesh(data) -> dict:
    """物体形状的一份拷贝（撤销用）：连多边形大小、散边、材质名一起。"""
    sizes = getattr(data, "polygon_sizes", None)
    loose = getattr(data, "loose_edges", None)
    return {"positions": data.positions.copy(), "normals": data.normals.copy(), "uvs": data.uvs.copy(),
            "material_ids": data.material_ids.copy(), "has_uvs": getattr(data, "has_uvs", True),
            "polygon_sizes": None if sizes is None else np.array(sizes, copy=True),
            "loose_edges": None if loose is None else np.array(loose, copy=True),
            "materials": list(getattr(data, "materials", []))}


def apply_snapshot(data, snap: dict) -> None:
    data.positions = snap["positions"].copy()
    data.normals = snap["normals"].copy()
    data.uvs = snap["uvs"].copy()
    data.material_ids = snap["material_ids"].copy()
    data.has_uvs = snap["has_uvs"]
    if "polygon_sizes" in snap:
        data.polygon_sizes = None if snap["polygon_sizes"] is None else np.array(snap["polygon_sizes"], copy=True)
        data.loose_edges = None if snap["loose_edges"] is None else np.array(snap["loose_edges"], copy=True)
        if snap.get("materials"):
            data.materials = list(snap["materials"])
    data.bounds_min = data.positions.min(axis=0) if len(data.positions) else np.zeros(3, np.float32)
    data.bounds_max = data.positions.max(axis=0) if len(data.positions) else np.zeros(3, np.float32)


class SculptStroke:
    """进行中的一笔：屏幕上的采样按间距变成笔触，再从几何缓冲取出表面位置和法线。"""

    def __init__(self, session: "SculptSession", view, brush: int, settings: dict, symmetry: tuple) -> None:
        self.session = session
        self.view = view
        self.brush = brush
        self.settings = settings
        self.symmetry = symmetry
        self.samples: deque = deque()
        self.last = None
        self.residual = 0.0
        self.ended = False
        self.grab_origin = None
        self.dab_count = 0

    def add(self, x: float, y: float, pressure: float) -> None:
        self.samples.append((float(x), float(y), float(pressure)))

    def _diameter(self, pressure: float) -> float:
        size = self.settings["size"]
        if self.settings["pressure_size"]:
            size *= max(0.05, pressure)
        return max(1.0, size)

    def take_dabs(self) -> np.ndarray:
        points = []
        while self.samples:
            x, y, p = self.samples.popleft()
            if self.last is None:
                points.append((x, y, p))
                self.last = (x, y, p)
                continue
            lx, ly, lp = self.last
            seg = math.hypot(x - lx, y - ly)
            if seg < 1e-6:
                continue
            step = max(1.0, self.settings["spacing"] * self._diameter((lp + p) * 0.5))
            travel = self.residual + seg
            count = int(travel // step)
            for k in range(1, count + 1):
                t = (k * step - self.residual) / seg
                points.append((lx + (x - lx) * t, ly + (y - ly) * t, lp + (p - lp) * t))
            self.residual = travel - count * step
            self.last = (x, y, p)
        if not points:
            return np.empty(0, DAB_DTYPE)
        return self.make_dabs(points)

    def make_dabs(self, points) -> np.ndarray:
        view = self.view
        pts = np.asarray(points, np.float64)
        xs = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, view.width - 1)
        ys = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, view.height - 1)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        pos, nrm = view.read_surface(x0, y0, x1, y1)
        p = pos[ys - y0, xs - x0]
        n = nrm[ys - y0, xs - x0]
        valid = p[:, 3] > 0.5
        if not valid.any():
            return np.empty(0, DAB_DTYPE)
        p, n, pressure = p[valid, :3].astype(np.float64), n[valid].astype(np.float64), pts[valid, 2]
        camera = view.camera
        depth = (p - camera.eye) @ camera.forward
        tan_y, _ = camera.tan_half(view.aspect)
        span = np.full(len(p), camera.distance) if camera.ortho else np.maximum(depth, 1e-9)
        per_pixel = 2.0 * span * tan_y / max(1, view.height)
        diameter = np.full(len(p), self.settings["size"])
        if self.settings["pressure_size"]:
            diameter = diameter * np.maximum(0.05, pressure)
        radius = np.maximum(diameter, 1.0) * 0.5 * per_pixel
        strength = np.full(len(p), self.settings["strength"])
        if self.settings["pressure_strength"]:
            strength = strength * pressure
        signs = [(1.0, 1.0, 1.0)]
        sx, sy, sz = self.symmetry
        for fx in ((1.0, -1.0) if sx else (1.0,)):
            for fy in ((1.0, -1.0) if sy else (1.0,)):
                for fz in ((1.0, -1.0) if sz else (1.0,)):
                    if (fx, fy, fz) != (1.0, 1.0, 1.0):
                        signs.append((fx, fy, fz))
        count = len(p)
        dabs = np.zeros(count * len(signs), DAB_DTYPE)
        for index, sign in enumerate(signs):
            block = dabs[index * count:(index + 1) * count]
            mirror = np.asarray(sign)
            block["center_radius"][:, :3] = p * mirror
            block["center_radius"][:, 3] = radius
            block["normal_strength"][:, :3] = n * mirror
            block["normal_strength"][:, 3] = strength
            block["params"][:, 0] = self.settings["hardness"]
            block["params"][:, 1] = self.settings["sign"]
        self.dab_count += len(dabs)
        return dabs


class SculptSession:
    def __init__(self, engine, obj) -> None:
        started = time.perf_counter()
        self.version = 0
        self._changed = False
        self.engine = engine
        self.obj = obj
        self.ctx = engine.ctx
        self.sculptor = SculptEngine(engine.ctx)
        data = obj.data
        self.entry_snapshot = snapshot_mesh(data)
        self.base = weld(data.positions, data.uvs if getattr(data, "has_uvs", True) else None, data.material_ids)
        self.sculptor.load(self.base, arena_fraction=self._backup_fraction())
        self.tag = object()
        self.stroke: SculptStroke | None = None
        self.changed = False
        self.ever_changed = False
        self.topology_version = 0
        self.uv_source = None      # 第一次重构前带 UV 的形状：回到绘制时据此把画好的内容搬到新 UV
        self._mesh_gpu = engine.meshes.get(obj.uid)
        renderer = engine.renderer
        if self._mesh_gpu is not None:
            renderer.hidden.add(id(self._mesh_gpu))
        renderer.extra_draws.append(self._draw)
        self._invalidate()
        log.info("进入雕刻：%s，%d 个三角形，用时 %.2f 秒", obj.name, self.base.triangle_count,
                 time.perf_counter() - started)

    def _backup_fraction(self) -> float:
        prefs = getattr(self.engine, "prefs", None)
        sculpt = getattr(prefs, "sculpt", None) if prefs is not None else None
        return float(getattr(sculpt, "undo_backup", 1.0)) if sculpt is not None else 1.0

    # ------------------------------------------------------------------ 绘制挂钩
    def _draw(self, renderer, view, pass_name: str) -> None:
        camera = view.camera
        view_proj = camera.view_projection(view.aspect)
        if pass_name == "gbuffer":
            self.sculptor.draw_gbuffer(view_proj)
            return
        shading = view.shading
        solid = SOLID_MODES.get(getattr(shading, "solid_light", "MATCAP"), 1)
        pixel_world = camera.world_per_pixel(camera.distance, view.height, view.aspect)
        self.sculptor.draw(view_proj, camera.view_matrix(), camera.eye, solid=solid, tint=(0.8, 0.8, 0.8),
                           matcap_tex=renderer._matcap_tex, studio=renderer._studio_data, cursor=view.cursor,
                           pixel_world=pixel_world)

    def _invalidate(self) -> None:
        for view in self.engine.views:
            view.gbuf_valid = False
            view.dirty = True
        self.engine._pending_work = True

    # ------------------------------------------------------------------ 笔划
    def stroke_begin(self, view, x: float, y: float, pressure: float, brush_name: str, settings, symmetry,
                     invert: bool = False) -> bool:
        if self.stroke is not None:
            self.stroke_end()
        hit = view.surface_at(x, y)
        if hit is None:
            return False
        brush = BRUSH_INDEX[brush_name]
        snapshot = {"size": float(settings.size), "strength": float(settings.strength),
                    "hardness": float(settings.hardness), "spacing": float(settings.spacing),
                    "pressure_size": bool(settings.pressure_size), "pressure_strength": bool(settings.pressure_strength),
                    "sign": -1.0 if (bool(settings.invert) != bool(invert)) else 1.0}
        if brush == S.SMOOTH:
            snapshot["strength"] = float(settings.smooth_strength)
        self.sculptor.begin_stroke()
        self.stroke = SculptStroke(self, view, brush, snapshot, symmetry)
        if brush == S.GRAB:
            dabs = self.stroke.make_dabs([(x, y, pressure)])
            if len(dabs) == 0:
                self.stroke = None
                return False
            self.sculptor.grab_begin(dabs)
            self.stroke.grab_origin = (np.asarray(hit[0], np.float64), x, y)
        else:
            self.stroke.add(x, y, pressure)
        return True

    def stroke_move(self, x: float, y: float, pressure: float) -> None:
        stroke = self.stroke
        if stroke is None:
            return
        if stroke.brush == S.GRAB:
            origin, _ox, _oy = stroke.grab_origin
            view = stroke.view
            ray_origin, direction = view.camera.ray(x, y, view.width, view.height)
            forward = view.camera.forward
            denom = float(direction @ forward)
            if abs(denom) < 1e-9:
                return
            t = float((origin - ray_origin) @ forward) / denom
            delta = ray_origin + direction * t - origin
            self.sculptor.grab_move(delta)
            self.changed = True
            self.ever_changed = True
            self._invalidate()
        else:
            stroke.add(x, y, pressure)
            self.engine._pending_work = True

    def advance(self) -> bool:
        """每帧调用：把攒下的笔触交给显卡。返回这一帧有没有改动形状。"""
        stroke = self.stroke
        if stroke is None or stroke.brush == S.GRAB:
            return False
        dabs = stroke.take_dabs()
        if len(dabs) == 0:
            return False
        self.sculptor.apply_dabs(dabs, stroke.brush)
        self.changed = True
        self.ever_changed = True
        self._invalidate()
        return True

    def stroke_end(self) -> None:
        stroke = self.stroke
        if stroke is None:
            return
        self.advance()
        self.stroke = None
        record = self.sculptor.end_stroke()
        if record is None:
            return
        history = self.engine.history
        if history is None:
            return
        sculptor = self.sculptor
        session = self

        def undo() -> None:
            if session.sculptor is sculptor and sculptor.mesh is not None:
                session.engine.ctx_make_current()
                sculptor.restore(record["clusters"], record["before"])
                session._invalidate()

        def redo() -> None:
            if session.sculptor is sculptor and sculptor.mesh is not None:
                session.engine.ctx_make_current()
                sculptor.restore(record["clusters"], record["after"])
                session._invalidate()

        label = {S.MASK: "遮罩"}.get(stroke.brush, "雕刻")
        nbytes = record["before"].nbytes + record["after"].nbytes
        history.push(label, undo, redo, nbytes)
        history.steps[-1].tag = self.tag

    def apply_once(self, dabs, brush: int) -> None:
        """不经过鼠标的一笔（例如清除遮罩）：直接作用并记一步撤销。"""
        self.sculptor.begin_stroke()
        self.sculptor.apply_dabs(dabs, brush)
        self.changed = True
        self.ever_changed = True
        self.stroke = SculptStroke(self, None, brush, {}, (False, False, False))
        self.stroke_end()
        self._invalidate()

    # ------------------------------------------------------------------ 遮罩整体编辑（反转、填满、柔化、扩大…）
    def _read_mask(self) -> np.ndarray:
        """当前遮罩（按排序后的顶点顺序）。"""
        sculptor = self.sculptor
        v = sculptor.vertex_count
        pos = np.frombuffer(sculptor.buffers["pos"].read(size=v * 16), np.float32).reshape(v, 4)
        return pos[:, 3].copy()

    def _write_mask(self, mask: np.ndarray) -> None:
        """只改遮罩，不动位置（位置用现在显卡上的，撤销时不会把之后雕的形状也退回去）。"""
        sculptor = self.sculptor
        v = sculptor.vertex_count
        pos = np.frombuffer(sculptor.buffers["pos"].read(size=v * 16), np.float32).reshape(v, 4).copy()
        pos[:, 3] = mask
        sculptor.buffers["pos"].write(np.ascontiguousarray(pos).tobytes())
        sculptor.version += 1
        self._invalidate()

    def neighbor_edges(self) -> np.ndarray:
        """排序后网格的边（两个方向都有），遮罩扩大、柔化用。随拓扑缓存。"""
        key = (id(self.sculptor.mesh), self.topology_version)
        cached = getattr(self, "_edges_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        tris = np.asarray(self.sculptor.mesh.triangles, np.int64)
        a = tris[:, [0, 1, 2]].reshape(-1)
        b = tris[:, [1, 2, 0]].reshape(-1)
        edges = np.unique(np.stack([np.concatenate([a, b]), np.concatenate([b, a])], axis=1), axis=0)
        self._edges_cache = (key, edges)
        return edges

    def edit_mask(self, fn, label: str) -> bool:
        """mask → fn(mask)，整体改遮罩，记一步撤销。没变化返回 False。"""
        before = self._read_mask()
        after = np.clip(np.asarray(fn(before), np.float32), 0.0, 1.0)
        if np.allclose(before, after, atol=1e-6):
            return False
        self._write_mask(after)
        self.changed = True
        self.ever_changed = True
        history = self.engine.history
        if history is None:
            return True
        sculptor = self.sculptor
        session = self

        def undo() -> None:
            if session.sculptor is sculptor and sculptor.mesh is not None:
                session.engine.ctx_make_current()
                session._write_mask(before)

        def redo() -> None:
            if session.sculptor is sculptor and sculptor.mesh is not None:
                session.engine.ctx_make_current()
                session._write_mask(after)

        history.push(label, undo, redo, before.nbytes * 2)
        history.steps[-1].tag = self.tag
        return True

    def stroke_cancel(self) -> None:
        if self.stroke is None:
            return
        self.stroke = None
        record = self.sculptor.end_stroke()
        if record is not None:
            self.sculptor.restore(record["clusters"], record["before"])
            self._invalidate()

    # ------------------------------------------------------------------ 拓扑
    def current_mesh(self) -> SculptMesh:
        positions, _normals, mask = self.sculptor.read_vertices()
        mesh = SculptMesh(positions, self.base.triangles, self.base.corner_uvs, self.base.material_ids)
        mesh.mask = mask
        return mesh

    def set_mesh(self, mesh: SculptMesh, label: str) -> None:
        """换成新拓扑（细分、重构），记一步撤销。"""
        before = self.current_mesh()
        self._load(mesh)
        history = self.engine.history
        if history is None:
            return

        def undo() -> None:
            self.engine.ctx_make_current()
            self._load(before)

        def redo() -> None:
            self.engine.ctx_make_current()
            self._load(mesh)

        history.push(label, undo, redo, before.vertices.nbytes * 3 + mesh.vertices.nbytes * 3)
        history.steps[-1].tag = self.tag

    def _load(self, mesh: SculptMesh) -> None:
        self.base = SculptMesh(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32),
                               mesh.corner_uvs, mesh.material_ids)
        if getattr(mesh, "mask", None) is not None:
            self.base.mask = mesh.mask
        self.sculptor.load(self.base, arena_fraction=self._backup_fraction())
        self.topology_version += 1
        self.changed = True
        self._invalidate()

    def subdivide(self, levels: int = 1, smooth: bool = True) -> None:
        current = self.current_mesh()
        result = subdivide(current, levels, smooth)
        if getattr(current, "mask", None) is not None:
            # 新的边点遮罩取两端的平均：Loop 细分里新顶点排在原顶点之后，这里简单地把原遮罩带过去，新点置 0
            mask = np.zeros(result.vertex_count, np.float32)
            mask[:current.vertex_count] = current.mask
            result.mask = mask
        self.set_mesh(result, "细分")

    def surface_area(self) -> float:
        """当前拓扑的表面积（按最近一次换拓扑时的形状算，估计重构面数用）。"""
        cached = getattr(self, "_area", None)
        if cached is not None and cached[0] == self.topology_version:
            return cached[1]
        v = np.asarray(self.base.vertices, np.float64)
        t = np.asarray(self.base.triangles, np.int64)
        area = float(np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1).sum() * 0.5)
        self._area = (self.topology_version, area)
        return area

    def remesh_estimate(self, resolution: int) -> dict:
        """重构前的预估：体素大小和大约多少个三角形（Surface Nets 每个表面格子一个顶点，三角形约为面积/体素² 的 3 倍）。"""
        grid = grid_for(self.base, resolution)
        h = float(grid["voxel"])
        return {"voxel": h, "triangles": int(3.0 * self.surface_area() / (h * h)), "resolution": grid["resolution"]}

    def voxel_remesh(self, resolution: int, relax: int = 2, close_holes: bool = True, keep_mask: bool = True,
                     min_island: int = 64) -> dict:
        """体素重构：换成均匀的新网格（没有 UV），记一步撤销。返回统计。"""
        current = self.current_mesh()
        if self.uv_source is None and current.corner_uvs is not None:
            positions, _normals, uvs, mats = unweld(current)
            self.uv_source = {"positions": positions, "uvs": uvs, "material_ids": mats}
        mask = current.mask if keep_mask else None
        if mask is not None and not np.any(mask > 0.0):
            mask = None
        result = voxel_remesh(current, int(resolution), int(relax), mask=mask, close_holes=bool(close_holes),
                              min_island=int(min_island))
        self.set_mesh(result, "体素重构")
        return dict(result.stats, triangles=result.triangle_count, vertices=result.vertex_count)


    # 「有没有没写回物体的改动」；每次标成 True 都换一个新版本号（自动保存据此判断形状变没变）
    @property
    def changed(self) -> bool:
        return self._changed

    @changed.setter
    def changed(self, value: bool) -> None:
        self._changed = bool(value)
        if value:
            self.version += 1

    # ------------------------------------------------------------------ 写回与退出
    def save_snapshot(self):
        """存盘用的当前形状，不写回物体、不重建显示（自动保存用，不打断雕刻）。

        显卡读回在这里做；拆成每个角的数据交给返回的函数（不碰显卡，可以在别的线程里跑），它返回 MeshData。"""
        from ..doc.meshio import MeshData

        positions, normals, _mask = self.sculptor.read_vertices()
        base = self.base
        source = self.obj.data
        retopo = bool(self.topology_version)
        name = getattr(source, "name", "")
        materials = list(getattr(source, "materials", []))
        sizes = None if retopo else getattr(source, "polygon_sizes", None)
        loose = getattr(source, "loose_edges", None)

        def build():
            mesh = SculptMesh(positions, base.triangles, base.corner_uvs, base.material_ids)
            corner_pos, corner_nrm, corner_uv, mats = unweld(mesh, normals)
            data = MeshData(name=name, positions=corner_pos, normals=corner_nrm, uvs=corner_uv, material_ids=mats,
                            materials=materials,
                            bounds_min=corner_pos.min(axis=0) if len(corner_pos) else np.zeros(3, np.float32),
                            bounds_max=corner_pos.max(axis=0) if len(corner_pos) else np.zeros(3, np.float32))
            data.has_uvs = base.corner_uvs is not None
            data.polygon_sizes = sizes
            data.loose_edges = loose
            return data

        return build

    def sync_to_object(self) -> bool:
        """把雕刻后的形状写回物体，绘制引擎重建几何。返回是否有改动。"""
        if not self.changed:
            return False
        positions, normals, _mask = self.sculptor.read_vertices()
        mesh = SculptMesh(positions, self.base.triangles, self.base.corner_uvs, self.base.material_ids)
        corner_pos, corner_nrm, corner_uv, mats = unweld(mesh, normals)
        data = self.obj.data
        data.positions = corner_pos
        data.normals = corner_nrm
        data.uvs = corner_uv
        data.has_uvs = self.base.corner_uvs is not None
        data.material_ids = mats
        data.bounds_min = corner_pos.min(axis=0)
        data.bounds_max = corner_pos.max(axis=0)
        if self.topology_version:
            data.polygon_sizes = None            # 细分、重构后全是三角形
        self.obj.geometry_dirty = True
        self.changed = False
        self.engine.rebuild_object(self.obj, keep_hidden=True)
        self._mesh_gpu = self.engine.meshes.get(self.obj.uid)
        if self._mesh_gpu is not None:
            self.engine.renderer.hidden.add(id(self._mesh_gpu))
        return True

    def close(self) -> dict | None:
        """退出雕刻模式：写回形状、拿掉挂钩、释放显存。返回进入和退出时的形状（有改动时），供记一步撤销。"""
        changed = self.sync_to_object()
        renderer = self.engine.renderer
        if self._draw in renderer.extra_draws:
            renderer.extra_draws.remove(self._draw)
        if self._mesh_gpu is not None:
            renderer.hidden.discard(id(self._mesh_gpu))
        self.sculptor.release()
        self.sculptor = None
        self._invalidate()
        history = self.engine.history
        if history is not None:
            history.remove_tagged(self.tag)
        if not (changed or self.ever_changed or self.topology_version):
            return None
        return {"before": self.entry_snapshot, "after": snapshot_mesh(self.obj.data), "uv_source": self.uv_source}
