"""引擎门面：界面和工具只和这个类打交道。

每帧 frame()：收回换页结果 → 读页面需求 → 处理笔划 → 推进抬笔后的合并 → 合成看得见的脏页 → 渲染变了的视口。

抬笔后笔划进入合并队列，每帧限时推进；合并完成前它仍作为叠加层显示，所以抬笔时画面不卡，
也可以马上落下一笔（最多同时叠加 MAX_OVERLAYS 笔）。撤销、重做和其他改动历史的操作之前，
未完成的合并会先同步做完。
所有方法都要求引擎的显卡上下文是当前上下文（撤销重做的回调除外，它们不碰显卡）。
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque

import moderngl
import numpy as np

from ..doc.project import (DIRECTION_VECTOR, GENERATOR_CODE, LAYER_KIND_CODE, MAX_FOLDER_DEPTH,
                           PLANE_COLOR, PLANE_HEIGHT, PLANE_MASK, PLANE_MR, Layer, Project, TextureSet)
from . import glx
from .adjust import LUT_SIZE
from .adjust import pack as adjust_params
from .cache import PageCache
from .display import (LAYER_STRIDE, MAX_GRAPHS, MAX_LAYERS, MAX_OVERLAYS, MAX_PART_IDS, PARAM_DTYPE, Compositor,
                      DisplaySet)
from .layerstore import LayerPixels, MergeJob, StrokeParams
from .meshgpu import MeshGPU, SetGPU
from .pageops import PageOps
from .pagepool import PoolSet
from .renderer import Renderer, View
from .stamp import DAB_DTYPE, Stamper, StrokeBuffer
from .view2d import Renderer2D, View2D

log = logging.getLogger("splender.engine")

BLEND_CODE = {"NORMAL": 0, "MULTIPLY": 1, "SCREEN": 2, "OVERLAY": 3, "ADD": 4, "SUBTRACT": 5, "DARKEN": 6, "LIGHTEN": 7}


class Perf:
    """给性能面板用的滚动统计。"""

    def __init__(self, size: int = 240) -> None:
        self.frame_ms = deque(maxlen=size)
        self.stamp_ms = deque(maxlen=size)
        self.compose_ms = deque(maxlen=size)
        self.render_ms = deque(maxlen=size)
        self.latency_ms = deque(maxlen=size)      # 输入事件到达 → 含该输入的画面渲染完成
        self.dabs = deque(maxlen=size)
        self.pages_composed = deque(maxlen=size)
        self.merge_ms = deque(maxlen=64)            # 抬笔到合并完成（不阻塞画面，分帧进行）
        self.merge_frame_ms = deque(maxlen=size)    # 每帧花在合并上的时间
        self.last_event_time = 0.0

    @staticmethod
    def _p(values, q) -> float:
        return float(np.percentile(np.fromiter(values, float), q)) if values else 0.0

    def summary(self) -> dict:
        return {
            "frame_ms_p50": self._p(self.frame_ms, 50), "frame_ms_p95": self._p(self.frame_ms, 95),
            "stamp_ms_p95": self._p(self.stamp_ms, 95), "compose_ms_p95": self._p(self.compose_ms, 95),
            "render_ms_p95": self._p(self.render_ms, 95),
            "latency_ms_p50": self._p(self.latency_ms, 50), "latency_ms_p95": self._p(self.latency_ms, 95),
            "dabs_per_frame": self._p(self.dabs, 50), "pages_per_frame": self._p(self.pages_composed, 50),
            "merge_ms_last": float(self.merge_ms[-1]) if self.merge_ms else 0.0,
            "merge_frame_ms_p95": self._p(self.merge_frame_ms, 95),
        }


class SetState:
    """一个纹理集在引擎里的状态。"""

    def __init__(self, ts: TextureSet, display: DisplaySet, index: int) -> None:
        self.ts = ts
        self.display = display
        self.index = index
        self.set_gpus: list[SetGPU] = []
        self.params = np.zeros(MAX_LAYERS, PARAM_DTYPE)
        self.params_dirty = True
        self.luts = None                         # 调整层的查找表，和 params 一起重建
        self.luts_version = 0
        self.hidden_rows: set = set()            # 自己或上级文件夹被隐藏的图层（按序号）
        self.selection = None                    # 选区（engine/selection.py），第一次用时建
        self.visible = [np.zeros((g, g), bool) for g in display.level_size]
        for level in range(display.base_level, display.levels):
            self.visible[level][:] = True


class ActiveStroke:
    """进行中的一笔。"""

    def __init__(self, engine: "Engine", view: View, state: SetState, layer: Layer, brush: dict, params: StrokeParams,
                 symmetry: tuple) -> None:
        self.engine = engine
        self.view = view
        self.state = state
        self.layer = layer
        self.brush = brush
        self.params = params
        self.symmetry = symmetry
        self.buffer = StrokeBuffer(engine.ctx, state.display.grid)
        self.samples: deque = deque()
        self.last = None
        self.residual = 0.0
        self.smooth_xy = None
        self.dab_count = 0
        self.ended = False
        self.oldest_event = 0.0
        self.ended_at = 0.0
        self.merge: MergeJob | None = None
        self.seen = np.zeros(state.display.grid * state.display.grid, bool)   # 已经预取过图层页的格子
        self.effect_cells: dict = {}  # 效果笔刷改过的格：(页表, 级, x, y) → [页表, 级, x, y, 最早的旧页, 最新的新页]
        self.smudge_uv = None         # 涂抹：上一帧鼠标在贴图上的位置
        self.uv_world = 1.0           # UV 视图里按场景尺寸画时：每单位 UV 平均多长的表面

    def add(self, x: float, y: float, pressure: float, when: float) -> None:
        self.samples.append((float(x), float(y), float(pressure), float(when)))

    # ---- 效果笔刷：一笔里每帧改的页合成一份撤销记录 ----
    def absorb_effect(self, records: list, cache) -> None:
        for group in records:
            for x, y, old, new in group.cells():
                key = (id(group.level), x, y)
                entry = self.effect_cells.get(key)
                if entry is None:
                    self.effect_cells[key] = [group.level, group.mip, x, y, old, new]
                else:
                    cache.free_page(entry[5])       # 前一帧的结果已经被这一帧换掉，没人再用
                    entry[5] = new

    def effect_records(self) -> list:
        from .layerstore import RecordGroup

        groups: dict = {}
        for level, mip, x, y, old, new in self.effect_cells.values():
            groups.setdefault(id(level), (level, mip, [], [], [], []))
            _level, _mip, xs, ys, olds, news = groups[id(level)]
            xs.append(x)
            ys.append(y)
            olds.append(old)
            news.append(new)
        return [RecordGroup(level, mip, xs, ys, olds, news) for level, mip, xs, ys, olds, news in groups.values()]

    def _diameter_px(self, pressure: float) -> float:
        size = self.brush["size"]
        if self.brush["pressure_size"]:
            size *= max(0.05, pressure)
        return max(1.0, size)

    def take_dabs(self) -> np.ndarray:
        """把攒下的输入采样变成笔触（屏幕上的点），再从几何缓冲取出它们在模型表面的位置。"""
        brush = self.brush
        points: list[tuple[float, float, float]] = []
        smooth = brush["smooth"]
        curve = brush["pressure_curve"]
        if self.samples:
            self.oldest_event = self.samples[0][3]
        while self.samples:
            x, y, pressure, _when = self.samples.popleft()
            pressure = max(0.0, min(1.0, pressure)) ** curve
            if self.smooth_xy is None or smooth <= 0.0:
                self.smooth_xy = (x, y)
            else:
                sx, sy = self.smooth_xy
                self.smooth_xy = (sx + (x - sx) * (1.0 - smooth), sy + (y - sy) * (1.0 - smooth))
            x, y = self.smooth_xy
            if self.last is None:
                points.append((x, y, pressure))
                self.last = (x, y, pressure)
                self.residual = 0.0
                continue
            lx, ly, lp = self.last
            seg = math.hypot(x - lx, y - ly)
            if seg <= 1e-6:
                self.last = (x, y, pressure)
                continue
            step = max(0.75, brush["spacing"] * self._diameter_px((lp + pressure) * 0.5))
            travel = self.residual + seg
            count = int(travel // step)
            for k in range(1, count + 1):
                t = (k * step - self.residual) / seg
                points.append((lx + (x - lx) * t, ly + (y - ly) * t, lp + (pressure - lp) * t))
            self.residual = travel - count * step
            self.last = (x, y, pressure)
        if not points:
            return np.empty(0, DAB_DTYPE)
        pts = np.asarray(points, np.float64)
        if getattr(self.view, "is_2d", False):
            return self._uv_dabs(pts)
        view = self.view
        xs = np.clip(np.rint(pts[:, 0]).astype(np.int64), 0, view.width - 1)
        ys = np.clip(np.rint(pts[:, 1]).astype(np.int64), 0, view.height - 1)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        pos, nrm = view.read_surface(x0, y0, x1, y1)
        p = pos[ys - y0, xs - x0]
        n = nrm[ys - y0, xs - x0]
        valid = (p[:, 3] > 0.5) & (np.rint(p[:, 3]).astype(np.int64) - 1 == self.state.index)
        valid &= (pts[:, 0] >= 0) & (pts[:, 0] < view.width) & (pts[:, 1] >= 0) & (pts[:, 1] < view.height)
        if not valid.any():
            return np.empty(0, DAB_DTYPE)
        p, n, pressures = p[valid, :3].astype(np.float64), n[valid].astype(np.float64), pts[valid, 2]
        camera = view.camera
        if brush["size_unit"] == "SCENE":
            radius = np.full(len(p), brush["scene_size"] * 0.5)
            if brush["pressure_size"]:
                radius = radius * np.maximum(0.05, pressures)
        else:
            depth = (p - camera.eye) @ camera.forward
            tan_y, _ = camera.tan_half(view.aspect)
            span = np.full(len(p), camera.distance) if camera.ortho else np.maximum(depth, 1e-9)
            per_pixel = 2.0 * span * tan_y / max(1, view.height)
            diameter = np.full(len(p), brush["size"])
            if brush["pressure_size"]:
                diameter = diameter * np.maximum(0.05, pressures)
            radius = np.maximum(diameter, 1.0) * 0.5 * per_pixel
        flow = np.full(len(p), brush["flow"])
        if brush["pressure_flow"]:
            flow = flow * pressures
        limit = math.cos(math.radians(brush["normal_falloff"])) if brush["backface_cull"] else -2.0
        signs = [(1.0, 1.0, 1.0)]
        sx, sy, sz = self.symmetry
        for flip_x in ((1.0, -1.0) if sx else (1.0,)):
            for flip_y in ((1.0, -1.0) if sy else (1.0,)):
                for flip_z in ((1.0, -1.0) if sz else (1.0,)):
                    if (flip_x, flip_y, flip_z) != (1.0, 1.0, 1.0):
                        signs.append((flip_x, flip_y, flip_z))
        dabs = np.zeros(len(p) * len(signs), DAB_DTYPE)
        count = len(p)
        for index, sign in enumerate(signs):
            block = dabs[index * count:(index + 1) * count]
            sign = np.asarray(sign)
            block["center_radius"][:, :3] = p * sign
            block["center_radius"][:, 3] = radius
            block["normal_hardness"][:, :3] = n * sign
            block["normal_hardness"][:, 3] = brush["hardness"]
            block["params"][:, 0] = flow
            block["params"][:, 1] = limit
        self.dab_count += len(dabs)
        return dabs

    def _uv_dabs(self, pts: np.ndarray) -> np.ndarray:
        """UV 视图里的笔触：中心是 UV，半径按屏幕像素（或场景尺寸）换算成 UV。"""
        view = self.view
        brush = self.brush
        pressures = pts[:, 2]
        uv = np.empty((len(pts), 2))
        uv[:, 0] = view.center[0] + (pts[:, 0] - view.width * 0.5) / view.zoom
        uv[:, 1] = view.center[1] + (view.height * 0.5 - pts[:, 1]) / view.zoom
        if brush["size_unit"] == "SCENE":
            radius = np.full(len(pts), brush["scene_size"] * 0.5 / max(self.uv_world, 1e-12))
            if brush["pressure_size"]:
                radius = radius * np.maximum(0.05, pressures)
        else:
            diameter = np.full(len(pts), brush["size"])
            if brush["pressure_size"]:
                diameter = diameter * np.maximum(0.05, pressures)
            radius = np.maximum(diameter, 1.0) * 0.5 / view.zoom
        flow = np.full(len(pts), brush["flow"])
        if brush["pressure_flow"]:
            flow = flow * pressures
        dabs = np.zeros(len(pts), DAB_DTYPE)
        dabs["center_radius"][:, :2] = uv
        dabs["center_radius"][:, 3] = radius
        dabs["normal_hardness"][:, 2] = 1.0
        dabs["normal_hardness"][:, 3] = brush["hardness"]
        dabs["params"][:, 0] = flow
        dabs["params"][:, 1] = -2.0
        dabs["params"][:, 2] = 1.0
        self.dab_count += len(dabs)
        return dabs


class Engine:
    def __init__(self, ctx, prefs=None, history=None) -> None:
        self.ctx = ctx
        self.prefs = prefs
        self.history = history
        self.caps = glx.Caps()
        import psutil
        ram_total = int(psutil.virtual_memory().total >> 20)
        if prefs is not None:
            budgets = prefs.resolved_budgets(self.caps.vram_total_mb, ram_total)
            samples = int(prefs.viewport.msaa)
            anisotropy = float(prefs.viewport.anisotropy)
            divisor = int(prefs.viewport.feedback_divisor)
            watermark = float(prefs.memory.free_watermark)
            self.evict_budget_ms = float(prefs.memory.evict_budget_ms)
            backup_active = float(prefs.memory.backup_mb_painting)
            backup_idle = float(prefs.memory.backup_mb_idle)
            scratch_dir = str(prefs.memory.scratch_dir or "") or None
            self.merge_budget_ms = float(prefs.paint.merge_budget_ms)
            self.merge_budget_painting_ms = float(prefs.paint.merge_budget_painting_ms)
            self.merge_pages = int(prefs.paint.merge_pages_per_frame)
            self.upload_budget_painting_ms = float(prefs.memory.upload_budget_painting_ms)
        else:
            budgets = {"vram_pool_mb": 2048, "display_cache_mb": 1024, "ram_cache_mb": 4096, "undo_mb": 2048}
            samples, anisotropy, divisor, watermark = 4, 16.0, 8, 0.08
            self.evict_budget_ms = 4.0
            backup_active, backup_idle, scratch_dir = 8.0, 128.0, None
            self.merge_budget_ms = 6.0
            self.merge_budget_painting_ms = 2.5
            self.merge_pages = 1024
            self.upload_budget_painting_ms = 1.5
        self.budgets = budgets
        self.pools = PoolSet(ctx, budgets["vram_pool_mb"] << 20)
        self.cache = PageCache(ctx, self.pools, ram_budget=budgets["ram_cache_mb"] << 20, caps=self.caps,
                               watermark=watermark, scratch_dir=scratch_dir, backup_mb_active=backup_active,
                               backup_mb_idle=backup_idle)
        self.ops = PageOps(ctx, self.pools)
        self.cache.packer = self.ops
        for fmt in ("rgba8", "rg16f"):                # 颜色页、笔划页一定会用到：启动时先建好
            self.pools[fmt].warm_up()
        self.stamper = Stamper(ctx, self.pools, self.cache, self.ops)
        self.layers = LayerPixels(self.cache, self.ops)
        self.compositor = Compositor(ctx, self.pools, self.cache)
        self.renderer = Renderer(ctx, samples=samples, anisotropy=anisotropy, feedback_divisor=divisor)
        self.renderer2d = Renderer2D(ctx, self.renderer)
        self.renderer2d.prefs = prefs
        from .selection import AntsPass

        self.ants = AntsPass(ctx)
        self.renderer.extra_draws.append(self._draw_selection)
        from .overlays import Overlays
        self.overlays = Overlays(ctx)                  # 地面网格、3D 游标、摄像机和灯光的线框
        self.renderer.overlays = self.overlays
        self.pick_objects: dict[int, object] = {}      # 点选编号 → 模型物体
        self.removed_sets: dict[int, SetState] = {}    # 撤销时暂时拿掉的纹理集（重做时原样放回）
        from ..nodes.evaluate import GraphEvaluator
        self.node_graphs = GraphEvaluator(ctx)        # 节点图求值（结果按签名缓存）
        self.node_graph_uids: set = set()
        self.display_budget = budgets["display_cache_mb"] << 20
        self.anisotropy = anisotropy
        self.views: list[View] = []
        self.sets: dict[int, SetState] = {}
        self.meshes: dict[int, MeshGPU] = {}
        self.project: Project | None = None
        self.frame_index = 0
        self.stroke: ActiveStroke | None = None
        self.merges: deque[ActiveStroke] = deque()     # 抬笔后等着并入图层的笔划，按先后顺序
        self._last_render = 0
        self.content_version = 0                        # 图层像素或图层设置每变一次加一（渲染预览据此重建）
        self._content_changed_at = 0.0
        self._rendered: dict[int, list] = {}            # 「渲染」着色模式的视口：id(view) -> [预览器, 引擎名, 上次的状态, 是否还在收敛]
        self.sculpt = None                              # 雕刻会话（雕刻模式时）
        self.edit = None                                # 编辑模式会话（mesh.session.EditSession）
        self.meshmaps: dict = {}                        # 纹理集 uid -> MeshMapSet（烘焙出的模型贴图）
        self.renderer.meshmaps = self.meshmaps
        self.bake = None                                # 进行中的烘焙任务
        self.bake_finished: list = []                   # 烘焙结束（完成、出错、取消）时调用 fn(job)
        self.selection_changed: list = []               # 选区变了（改、撤销、重做）时调用 fn(纹理集)
        viewport_prefs = getattr(prefs, "viewport", None)
        self.bake_budget_ms = float(getattr(viewport_prefs, "bake_budget_ms", 12.0))
        self._make_current = None                       # 宿主设置：让引擎的显卡上下文成为当前上下文
        self._collect_ms = 0.0
        self._completing = False
        self.stroke_reject = ""              # 上一次落笔没成功的原因，界面据此提示
        self.perf = Perf()
        self.compose_budget = 1024           # 每帧最多合成多少页
        self._visibility_dirty = True
        self._pending_work = True
        if history is not None:
            history.barrier = self.complete_strokes
        log.info("引擎就绪：%s，显存 %d MB，页池预算 %d MB，显示缓存 %d MB，内存缓存 %d MB，稀疏贴图%s",
                 ctx.info.get("GL_RENDERER", "?"), self.caps.vram_total_mb, budgets["vram_pool_mb"],
                 budgets["display_cache_mb"], budgets["ram_cache_mb"], "可用" if self.caps.sparse else "不可用")

    # ================================================================== 文档
    def set_project(self, project: Project | None) -> None:
        saver = self.__dict__.get("_autosave")
        if saver is not None:
            saver.reset()                    # 换工程：上一个工程的恢复文件作废
        self._clear_scene()
        self.project = project
        if project is None:
            return
        project.graphs_changed.connect(self._on_graph_changed)
        self.node_graph_uids = {graph.uid for graph in project.graphs}
        project.changed.connect(self._on_project_changed)
        self.overlays.project = project
        for obj in project.all_objects():
            self.watch_object(obj)
        self.update_selection()
        for index, ts in enumerate(project.texture_sets):
            self._add_set(ts, index)
        for obj in project.objects:
            self._add_object(obj)
        self._rebuild_scene()
        for view in self.views:
            self.frame_view(view)

    def _touch_content(self) -> None:
        self.content_version += 1
        self._content_changed_at = time.perf_counter()

    def ctx_make_current(self) -> None:
        if self._make_current is not None:
            self._make_current()

    # ================================================================== 雕刻模式
    def enter_sculpt(self, obj):
        from ..sculpt.session import SculptSession

        if self.sculpt is not None:
            if self.sculpt.obj is obj:
                return self.sculpt
            self.exit_sculpt()
        self.complete_strokes(include_active=True)
        self.sculpt = SculptSession(self, obj)
        return self.sculpt

    def exit_sculpt(self) -> dict | None:
        session = self.sculpt
        if session is None:
            return None
        session.stroke_end()
        self.sculpt = None
        return session.close()

    # ================================================================== 编辑模式
    def enter_edit(self, obj):
        from ..mesh.session import EditSession

        if self.edit is not None:
            if self.edit.obj is obj:
                return self.edit
            self.exit_edit()
        self.complete_strokes(include_active=True)
        if self.bake is not None:
            self.bake_cancel()
        self.edit = EditSession(self, obj)
        self.update_selection()
        return self.edit

    def exit_edit(self) -> dict | None:
        """退出编辑模式：三角化写回物体，重建它的绘制数据。返回进入、退出时的形状（有改动时）。"""
        session = self.edit
        if session is None:
            return None
        self.edit = None
        result = session.close()
        if result and result.get("changed"):
            self.rebuild_object(session.obj)
        self.update_selection()
        return result

    def edit_mesh_changed(self, obj, topology: bool) -> None:
        """编辑模式里形状变了：只更新显示用的显卡数据（绘制用的分块几何退出时再建）。"""
        mesh = self.meshes.get(obj.uid)
        if topology or mesh is None or mesh.corner_count != len(obj.data.positions):
            if mesh is not None:
                for state in list(self.sets.values()):
                    keep = []
                    for set_gpu in state.set_gpus:
                        if set_gpu.mesh is mesh:
                            self.stamper.forget(set_gpu)
                            set_gpu.release()
                        else:
                            keep.append(set_gpu)
                    state.set_gpus = keep
                self.renderer.forget_mesh(mesh)
                self.renderer2d.forget_mesh(mesh)
                self.ants.forget_mesh(mesh)
                mesh.release()
            self.meshes[obj.uid] = MeshGPU(self.ctx, obj.data)
            self._rebuild_scene()
        else:
            mesh.update_vertices(obj.data)
        for set_uid in getattr(obj, "material_sets", []):
            maps = self.meshmaps.get(set_uid)
            if maps is not None:
                maps.meta["stale"] = True
        self._touch_content()
        self.update_scene_extent()
        for view in self.views:
            view.dirty = True
            view.gbuf_valid = False
        self._pending_work = True

    def rebuild_object(self, obj, keep_hidden: bool = False) -> None:
        """物体的形状变了：重建它的显卡数据和绘制用的分块几何。贴图页面不受影响（UV 不变）。
        正在烘焙就取消；这个物体用到的模型贴图标成过期。"""
        if self.bake is not None:
            self.bake_cancel()
        for set_uid in getattr(obj, "material_sets", []):
            maps = self.meshmaps.get(set_uid)
            if maps is not None:
                maps.meta["stale"] = True
        old = self.meshes.pop(obj.uid, None)
        if old is not None:
            for state in self.sets.values():
                keep = []
                for set_gpu in state.set_gpus:
                    if set_gpu.mesh is old:
                        self.stamper.forget(set_gpu)
                        set_gpu.release()
                    else:
                        keep.append(set_gpu)
                state.set_gpus = keep
            self.renderer.forget_mesh(old)
            self.renderer2d.forget_mesh(old)
            self.ants.forget_mesh(old)
            self.renderer.hidden.discard(id(old))
            old.release()
        self._add_object(obj)
        self._rebuild_scene()
        self._touch_content()
        for state in self.sets.values():
            state.display.mark_all_dirty()
        for view in self.views:
            view.dirty = True
            view.gbuf_valid = False
        self._pending_work = True

    def _clear_scene(self) -> None:
        self.bake_cancel()
        if self.project is not None:
            try:
                self.project.graphs_changed.disconnect(self._on_graph_changed)
            except Exception:  # noqa: BLE001
                pass
            try:
                self.project.changed.disconnect(self._on_project_changed)
            except Exception:  # noqa: BLE001
                pass
        self.overlays.project = None
        for state in self.removed_sets.values():
            for set_gpu in state.set_gpus:
                self.stamper.forget(set_gpu)
                set_gpu.release()
            self.renderer.forget_display(state.display)
            state.display.release()
        self.removed_sets.clear()
        for uid in list(self.node_graph_uids):
            self.node_graphs.forget(uid)
        self.node_graph_uids = set()
        for maps in self.meshmaps.values():
            maps.release()
        self.meshmaps.clear()
        if self.sculpt is not None:
            try:
                self.sculpt.close()
            except Exception:  # noqa: BLE001
                log.exception("关闭雕刻会话时出错")
            self.sculpt = None
        if self.edit is not None:
            try:
                self.edit.close()
            except Exception:  # noqa: BLE001
                log.exception("关闭编辑模式时出错")
            self.edit = None
        self._touch_content()
        self.stroke_cancel()
        while self.merges:
            stroke = self.merges.popleft()
            if stroke.merge is not None:
                stroke.merge.cancel()
            stroke.buffer.release(self.cache)
        for state in self.sets.values():
            for set_gpu in state.set_gpus:
                self.stamper.forget(set_gpu)
                set_gpu.release()
            self.renderer.forget_display(state.display)
            state.display.release()
            if getattr(state, "idsel_tex", None) is not None:
                state.idsel_tex.release()
                state.idsel_tex = None
            if state.selection is not None:
                state.selection.release()
                state.selection = None
            for layer in state.ts.layers:
                self.layers.drop_layer(layer.uid)
        self.sets.clear()
        for mesh in self.meshes.values():
            self.renderer.forget_mesh(mesh)
            self.renderer2d.forget_mesh(mesh)
            self.ants.forget_mesh(mesh)
            mesh.release()
        self.meshes.clear()
        self.renderer.scene = []

    def _add_set(self, ts: TextureSet, index: int) -> SetState:
        display = DisplaySet(self.ctx, self.caps, ts.size, self.anisotropy)
        state = SetState(ts, display, index)
        self.sets[ts.uid] = state
        ts.layers_changed.connect(lambda kind, layer, uid=ts.uid: self._on_layers_changed(uid, kind, layer))
        return state

    def _add_object(self, obj) -> None:
        mesh = MeshGPU(self.ctx, obj.data)
        self.meshes[obj.uid] = mesh
        if not getattr(obj.data, "has_uvs", True):
            return
        for material, set_uid in enumerate(obj.material_sets):
            state = self.sets.get(set_uid)
            if state is None or mesh.material_count[material] == 0:
                continue
            self._add_set_gpu(state, obj, mesh, material)

    def _add_set_gpu(self, state: SetState, obj, mesh, material: int) -> None:
        from .meshprep import build_set_geometry

        width = float(self.prefs.paint.dilate_texels) if self.prefs is not None else 32.0
        width = max(2.0, width * state.ts.size / 16384.0)
        geometry = build_set_geometry(obj.data, material, state.ts.size, skirt_texels=width)
        state.set_gpus.append(SetGPU(self.ctx, mesh, geometry))

    # ================================================================== 纹理集换分辨率
    def resize_texture_set(self, ts, new_size: int) -> dict:
        """按新边长重新采样这套贴图每个图层的像素（还没换上去）。返回 {图层 uid: (旧像素, 新像素)}，
        交给 apply_texture_set_size 换上、撤销时换回。"""
        from .layerstore import LayerStore
        from .resize import TextureResize

        self.complete_strokes(include_active=True)
        if self.bake is not None:
            self.bake_cancel()
        new_stores = TextureResize(self, ts, int(new_size)).run()
        pairs = {}
        for layer in ts.layers:
            old = self.layers.stores.get(layer.uid)
            new = new_stores.get(layer.uid)
            if old is not None or new is not None:
                pairs[layer.uid] = (old, new if new is not None else LayerStore(int(new_size) // 256))
        return pairs

    def apply_texture_set_size(self, ts, pairs: dict, use_new: bool) -> None:
        """ts.resolution 已经改好：换显示贴图和分块几何，图层像素换成新的（use_new）或旧的。"""
        self.complete_strokes(include_active=True)
        self.drop_selection(ts)
        for uid, (previous, new) in pairs.items():
            self.layers.detach_layer(uid)
            target = new if use_new else previous
            if target is not None:
                self.layers.attach_layer(uid, target)
        state = self.sets.get(ts.uid) or self.removed_sets.get(ts.uid)
        if state is not None and state.display.size != ts.size:
            old_display = state.display
            state.display = DisplaySet(self.ctx, self.caps, ts.size, self.anisotropy)
            old_display.release()
            for set_gpu in state.set_gpus:
                self.stamper.forget(set_gpu)
                set_gpu.release()
            state.set_gpus = []
            if self.project is not None:
                for obj in self.project.objects:
                    mesh = self.meshes.get(obj.uid)
                    if mesh is None or not getattr(obj.data, "has_uvs", True):
                        continue
                    for material, set_uid in enumerate(obj.material_sets):
                        if set_uid == ts.uid and mesh.material_count[material] > 0:
                            self._add_set_gpu(state, obj, mesh, material)
            state.params_dirty = True
            state.material_dirty = True
        self._rebuild_scene()
        self._after_store_swap()

    def apply_filter(self, ts, layer, name: str, params: dict, planes: dict, label: str) -> int:
        """对图层的几种页做一次滤镜（engine/filters.py），记一步撤销。返回改了多少格（第 0 级）。"""
        from .filters import LayerFilter

        self.complete_strokes(include_active=True)
        state = self.sets.get(ts.uid)
        if state is None:
            return 0
        runner = LayerFilter(self, ts, layer, name, params, planes)
        records = runner.run()
        self.last_filter_stats = dict(runner.stats)
        if not records:
            return 0
        self._touch_content()
        for mip, tiles in LayerPixels.affected(records).items():
            state.display.mark_dirty(mip, tiles)
        for view in self.views:
            view.dirty = True
        self._pending_work = True
        if self.project is not None:
            self.project.mark_dirty()
        self._push_pixel_step(label, state, records)
        return int(runner.stats.get("pages", 0))

    def free_texture_set_stores(self, pairs: dict, keep_new: bool) -> None:
        """换分辨率的撤销步骤被丢弃：释放用不到的那一份像素。"""
        for _uid, (previous, new) in pairs.items():
            self.layers.free_store(previous if keep_new else new)

    def _rebuild_scene(self) -> None:
        scene = []
        self.pick_objects = {}
        if self.project is not None:
            for index, obj in enumerate(self.project.objects):
                mesh = self.meshes.get(obj.uid)
                if mesh is not None:
                    mesh.pick_id = index + 1
                    mesh.object_uid = obj.uid
                    self.pick_objects[index + 1] = obj
                if mesh is None or not obj.visible:
                    continue
                for material in range(len(mesh.material_ibo)):
                    set_uid = obj.material_sets[material] if material < len(obj.material_sets) else 0
                    state = self.sets.get(set_uid)
                    if state is not None and getattr(obj.data, "has_uvs", True):
                        scene.append((mesh, material, state.display, state.ts, state.index))
                    else:
                        scene.append((mesh, material, None, None, -1))
        self.renderer.scene = scene
        self.update_scene_extent()
        for view in self.views:
            view.invalidate_camera()
        self._visibility_dirty = True
        self.update_selection()

    def update_scene_extent(self) -> None:
        """场景包围球（模型、摄像机、灯光都算）交给各视口的相机，远近裁剪按它算。"""
        center, radius = self.scene_bounds(include_objects=True)
        for view in self.views:
            camera = view.camera
            camera.scene_center = np.asarray(center, np.float64)
            camera.scene_radius = max(float(radius), 1e-6)

    def scene_bounds(self, include_objects: bool = False, only=None) -> tuple[np.ndarray, float]:
        """场景（或 only 里这些物体）的包围球：(中心, 半径)，内部坐标。"""
        from ..doc.objects import to_internal

        points = []
        project = self.project
        if only is not None:
            objects = list(only)
        elif project is not None:
            objects = [o for o in project.all_objects() if o.visible] if include_objects else \
                [o for o in project.objects if o.visible]
        else:
            objects = []
        for obj in objects:
            if obj.kind == "MESH":
                mesh = self.meshes.get(obj.uid)
                if mesh is not None:
                    points.append(np.asarray(mesh.bounds_min, np.float64))
                    points.append(np.asarray(mesh.bounds_max, np.float64))
            else:
                p = to_internal(np.asarray(obj.transform.location, np.float64))
                size = 0.5
                points.append(p - size)
                points.append(p + size)
        if not points and only is None and self.meshes:
            for mesh in self.meshes.values():
                points.append(np.asarray(mesh.bounds_min, np.float64))
                points.append(np.asarray(mesh.bounds_max, np.float64))
        if not points:
            return np.zeros(3), 1.0
        lo = np.min(points, axis=0)
        hi = np.max(points, axis=0)
        return (lo + hi) * 0.5, float(np.linalg.norm(hi - lo) * 0.5) or 1.0

    def _on_layers_changed(self, set_uid: int, kind: str, layer) -> None:
        state = self.sets.get(set_uid)
        if state is None or kind == "active":
            return
        if kind in ("material", "material_values"):
            # 材质节点：结构变了重新编译，数值变了只换参数
            if kind == "material":
                state.material_dirty = True
            state.material_values_dirty = True
            state.display.mark_all_dirty()
            self._touch_content()
            self._pending_work = True
            for view in self.views:
                view.dirty = True
            return
        state.params_dirty = True
        state.display.mark_all_dirty()
        self._pending_work = True
        self._touch_content()
        for view in self.views:
            view.dirty = True

    # ================================================================== 视图
    # ================================================================== 物体（选择、变换、增删）
    def _on_project_changed(self, kind: str = "") -> None:
        if kind == "objects" and self.project is not None:
            for obj in self.project.all_objects():
                self.watch_object(obj)
        if kind in ("selection", "objects"):
            self.update_selection()
            self.refresh_overlays()

    def update_selection(self) -> None:
        """选中状态同步到描边：模型的点选编号 → 0.5 选中、1.0 当前物体。"""
        states = {}
        project = self.project
        if project is not None and getattr(self, "edit", None) is None:     # 编辑模式里不画物体描边（和 Blender 一样）
            active = project.active_object_uid
            for pick_id, obj in self.pick_objects.items():
                if obj.visible and getattr(obj, "select", False):
                    states[pick_id] = 1.0 if obj.uid == active else 0.5
        if states != self.renderer.selection_states:
            self.renderer.selection_states = states
            for view in self.views:
                view.dirty = True
            self._pending_work = True

    def object_at(self, view: View, x: float, y: float, radius_px: float = 12.0):
        """鼠标下的物体：先看摄像机、灯光、空物体（屏幕上离得够近），再看模型（几何缓冲里的编号）。"""
        from ..doc.objects import display_to_internal_matrix

        project = self.project
        if project is None:
            return None
        camera = view.camera
        view_proj = camera.view_projection(view.aspect)
        best, best_d = None, radius_px
        for obj in project.scene_objects:
            if not obj.visible:
                continue
            p = display_to_internal_matrix(obj.transform.matrix())[:3, 3]
            clip = view_proj @ np.array([p[0], p[1], p[2], 1.0])
            if clip[3] <= 1e-9:
                continue
            sx = (clip[0] / clip[3] * 0.5 + 0.5) * view.width
            sy = (0.5 - clip[1] / clip[3] * 0.5) * view.height
            d = math.hypot(sx - x, sy - y)
            if d < best_d:
                best, best_d = obj, d
        if best is not None:
            return best
        pick = self.renderer.object_id_at(view, x, y)
        return self.pick_objects.get(pick)

    def objects_in_rect(self, view: View, x0: float, y0: float, x1: float, y1: float) -> list:
        from ..doc.objects import display_to_internal_matrix

        project = self.project
        if project is None:
            return []
        found = [self.pick_objects[i] for i in self.renderer.object_ids_in(view, int(x0), int(y0), int(x1), int(y1))
                 if i in self.pick_objects]
        lo_x, hi_x, lo_y, hi_y = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        view_proj = view.camera.view_projection(view.aspect)
        for obj in project.scene_objects:
            if not obj.visible:
                continue
            p = display_to_internal_matrix(obj.transform.matrix())[:3, 3]
            clip = view_proj @ np.array([p[0], p[1], p[2], 1.0])
            if clip[3] <= 1e-9:
                continue
            sx = (clip[0] / clip[3] * 0.5 + 0.5) * view.width
            sy = (0.5 - clip[1] / clip[3] * 0.5) * view.height
            if lo_x <= sx <= hi_x and lo_y <= sy <= hi_y:
                found.append(obj)
        return found

    def set_display_matrix(self, obj, matrix) -> None:
        """拖动变换时：模型按这个内部矩阵实时显示（None 恢复原样）。"""
        mesh = self.meshes.get(obj.uid)
        if mesh is not None:
            mesh.model = None if matrix is None else np.asarray(matrix, np.float64)
        for view in self.views:
            view.dirty = True
            view.gbuf_valid = False
        self._pending_work = True

    def transform_object(self, obj, delta) -> None:
        """把内部坐标的变换矩阵 delta 写进模型的顶点（松手、撤销、重做时调用），重建它的显卡数据。
        贴图内容不用重新合成（UV 没变）；模型贴图（遮蔽、位置等）标成过期。"""
        data = obj.data
        obj.geometry_dirty = True          # 顶点变了，保存时要重写模型数据
        delta = np.asarray(delta, np.float64)
        rot = delta[:3, :3]
        offset = delta[:3, 3].astype(np.float32)
        if np.allclose(rot, np.identity(3), atol=1e-12):
            # 只是平移（最常见）：顶点加偏移，法线不变
            data.positions = data.positions + offset
            data.bounds_min = np.asarray(data.bounds_min, np.float32) + offset
            data.bounds_max = np.asarray(data.bounds_max, np.float32) + offset
        else:
            r32 = rot.astype(np.float32)
            data.positions = (data.positions @ r32.T + offset).astype(np.float32)
            n = data.normals @ np.linalg.inv(rot).astype(np.float32)
            data.normals = (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)).astype(np.float32)
            if np.linalg.det(rot) < 0:
                # 镜像（负缩放）会把三角形翻面：每个三角形的后两个角对调，正面还是朝外
                for name in ("positions", "normals", "uvs"):
                    corners = getattr(data, name).reshape(-1, 3, getattr(data, name).shape[1])
                    corners[:, [1, 2]] = corners[:, [2, 1]]
            if len(data.positions):
                data.bounds_min = data.positions.min(axis=0)
                data.bounds_max = data.positions.max(axis=0)
        mesh = self.meshes.get(obj.uid)
        if mesh is None:
            return
        if self.bake is not None:
            self.bake_cancel()
        for set_uid in getattr(obj, "material_sets", []):
            maps = self.meshmaps.get(set_uid)
            if maps is not None:
                maps.meta["stale"] = True
        mesh.update_vertices(data)
        mesh.model = None
        for state in list(self.sets.values()) + list(self.removed_sets.values()):
            for set_gpu in state.set_gpus:
                if set_gpu.mesh is mesh:
                    set_gpu.apply_transform(delta)
        self._touch_content()
        self.update_scene_extent()
        for view in self.views:
            view.dirty = True
            view.gbuf_valid = False
        self._pending_work = True

    def set_transform_guide(self, lines) -> None:
        """拖动变换时的约束轴线：[(内部坐标的点, 方向, 颜色), ...]，None 不画。"""
        self.overlays.guide = lines or None
        for view in self.views:
            view.dirty = True
        self._pending_work = True

    def refresh_overlays(self) -> None:
        """摄像机、灯光、游标这些叠加层的内容变了：重画各视口。"""
        for view in self.views:
            view.dirty = True
        self._pending_work = True

    def watch_object(self, obj) -> None:
        """跟着物体走：网格的变换值一改，就把差量补到顶点上（面板输入、拖动松手、撤销都走这里）；
        显示开关改了重排场景；摄像机、灯光、空物体改了重画叠加层。"""
        if getattr(obj, "_watched_by", None) is self:
            return
        obj._watched_by = self
        if obj.kind == "MESH":
            obj._applied = obj.transform.matrix()
            obj.transform.changed.connect(lambda _name, target=obj: self._on_object_transform(target))
        else:
            obj.transform.changed.connect(lambda _name: self.refresh_overlays())
            data = getattr(obj, "data", None)
            if data is not None and hasattr(data, "changed"):
                data.changed.connect(lambda _name: self.refresh_overlays())
        changed = getattr(obj, "changed", None)
        if changed is not None:
            changed.connect(lambda name, target=obj: self._on_object_prop(target, name))

    def _on_object_prop(self, obj, name: str) -> None:
        if name == "visible":
            if obj.kind == "MESH":
                self._rebuild_scene()
            if not obj.visible and obj.select:
                obj.select = False
            self.update_selection()
            self.refresh_overlays()
            for view in self.views:
                view.gbuf_valid = False
        elif name in ("empty_display_type", "empty_size", "name"):
            self.refresh_overlays()

    def _on_object_transform(self, obj) -> None:
        from ..doc.objects import display_to_internal_matrix

        if self.project is None or obj not in self.project.objects:
            return
        new = obj.transform.matrix()
        old = getattr(obj, "_applied", None)
        obj._applied = new
        if old is None:
            return
        delta = new @ np.linalg.inv(old)
        if np.allclose(delta, np.identity(4), atol=1e-12):
            return
        self.transform_object(obj, display_to_internal_matrix(delta))

    def mark_transform_applied(self, obj) -> None:
        """「应用变换」：变换值改成新值，但顶点不动（调用方先改 obj.transform，再调这个）。"""
        obj._applied = obj.transform.matrix()

    def _rebuild_geometry(self, obj) -> None:
        if self.bake is not None:
            self.bake_cancel()
        for set_uid in getattr(obj, "material_sets", []):
            maps = self.meshmaps.get(set_uid)
            if maps is not None:
                maps.meta["stale"] = True
        self._drop_mesh(obj)
        self._add_object(obj)
        self._rebuild_scene()
        self._touch_content()
        for view in self.views:
            view.dirty = True
            view.gbuf_valid = False
        self._pending_work = True

    def _drop_mesh(self, obj) -> None:
        old = self.meshes.pop(obj.uid, None)
        if old is None:
            return
        for state in list(self.sets.values()) + list(self.removed_sets.values()):
            keep = []
            for set_gpu in state.set_gpus:
                if set_gpu.mesh is old:
                    self.stamper.forget(set_gpu)
                    set_gpu.release()
                else:
                    keep.append(set_gpu)
            state.set_gpus = keep
        self.renderer.forget_mesh(old)
        self.renderer2d.forget_mesh(old)
        self.ants.forget_mesh(old)
        self.renderer.hidden.discard(id(old))
        old.release()

    def add_texture_set(self, ts) -> None:
        """运行中加一套纹理集（新建物体、撤销删除时）。以前拿掉的同一套原样放回。"""
        if ts.uid in self.sets:
            return
        state = self.removed_sets.pop(ts.uid, None)
        if state is not None:
            self.sets[ts.uid] = state
            state.index = len(self.sets) - 1
            state.params_dirty = True
            state.display.mark_all_dirty()
        else:
            self._add_set(ts, len(self.sets))
        self._reindex_sets()
        self._pending_work = True

    def remove_texture_set(self, ts) -> None:
        """运行中拿掉一套纹理集：显示数据和图层像素留着（撤销、重做时还要用），直到工程关闭。"""
        state = self.sets.pop(ts.uid, None)
        if state is None:
            return
        self.removed_sets[ts.uid] = state
        self._reindex_sets()
        self._rebuild_scene()
        self._pending_work = True

    def _reindex_sets(self) -> None:
        if self.project is None:
            return
        for index, ts in enumerate(self.project.texture_sets):
            state = self.sets.get(ts.uid)
            if state is not None:
                state.index = index

    def add_mesh_object(self, obj) -> None:
        """运行中加一个模型（工程里已经加好了）：建显卡数据，放进场景。"""
        self.watch_object(obj)
        self._drop_mesh(obj)
        self._add_object(obj)
        self._rebuild_scene()
        self._visibility_dirty = True
        for view in self.views:
            view.dirty = True
            view.gbuf_valid = False
        self._pending_work = True

    def remove_mesh_object(self, obj) -> None:
        self._drop_mesh(obj)
        self._rebuild_scene()
        for view in self.views:
            view.dirty = True
            view.gbuf_valid = False
        self._pending_work = True

    def create_view(self, shading, overlay=None) -> View:
        view = View(self.renderer, shading)
        center, radius = self.scene_bounds(include_objects=True)
        view.camera.scene_center = np.asarray(center, np.float64)
        view.camera.scene_radius = max(float(radius), 1e-6)
        if overlay is None:
            from ..doc.project import ViewOverlay
            overlay = ViewOverlay()
        view.overlay = overlay
        center, radius = self.scene_bounds()
        view.camera.scene_radius = radius
        view.camera.frame(center, radius)
        self.views.append(view)
        return view

    def create_view2d(self) -> View2D:
        view = View2D(self.renderer2d)
        self.views.append(view)
        return view

    # ================================================================== 选区
    def selection(self, ts):
        """这套贴图的选区（没有就建一个空的）。"""
        from .selection import Selection

        state = self.sets.get(ts.uid)
        if state is None:
            raise ValueError("没有这套贴图")
        if state.selection is None:
            state.selection = Selection(self, ts)
        return state.selection

    def active_selection(self, ts):
        """这套贴图有选区时返回它，否则 None。"""
        state = self.sets.get(getattr(ts, "uid", None))
        selection = getattr(state, "selection", None)
        return selection if selection is not None and selection.active else None

    def drop_selection(self, ts) -> None:
        """丢掉选区（换分辨率、换 UV 之后原来的选区对不上了）。"""
        state = self.sets.get(ts.uid) or self.removed_sets.get(ts.uid)
        if state is not None and state.selection is not None:
            state.selection.release()
            state.selection = None
            for fn in list(self.selection_changed):
                fn(ts)

    def _paint_selection(self, stroke):
        """画这一笔时要按哪个选区限制（没有、或者画的就是选区自己时 None）。"""
        selection = self.active_selection(stroke.state.ts)
        return selection if selection is not None and selection.applies_to(stroke.layer) else None

    def _draw_selection(self, renderer, view, phase: str) -> None:
        """三维视口：画完模型以后画选区边线。"""
        if phase != "color" or not getattr(getattr(self.prefs, "viewport", None), "selection_ants", True):
            return
        overlay = getattr(view, "overlay", None)
        if overlay is not None and not overlay.show_overlays:
            return
        self.ants.draw(self, renderer, view)

    def view2d_state(self, view: View2D):
        """平面视图当前显示的纹理集。"""
        state = self.sets.get(view.set_uid)
        if state is None and self.project is not None:
            ts = self.project.active_texture_set
            state = self.sets.get(ts.uid) if ts is not None else None
        return state

    def destroy_view(self, view: View) -> None:
        entry = self._rendered.pop(id(view), None)
        if entry is not None:
            entry[0].release()
        if view in self.views:
            self.views.remove(view)
        view.release()
        self._visibility_dirty = True

    def frame_view(self, view: View) -> None:
        if view.is_2d:
            view.fit()
            return
        center, radius = self.scene_bounds()
        view.camera.frame(center, radius, view.aspect if view.height else 1.5)
        center_all, radius_all = self.scene_bounds(include_objects=True)
        view.camera.scene_center = np.asarray(center_all, np.float64)
        view.camera.scene_radius = max(float(radius_all), 1e-6)
        view.invalidate_camera()

    def frame_objects(self, view: View, objects) -> bool:
        """把这些物体放进视口（「查看所选」）。没有可看的返回 False。"""
        objects = [o for o in objects if o.visible]
        if not objects or view.is_2d:
            return False
        center, radius = self.scene_bounds(only=objects)
        view.camera.frame(center, max(radius, 0.05), view.aspect if view.height else 1.5)
        view.invalidate_camera()
        return True

    # ================================================================== 拾取与光标
    def brush_radius_world(self, view: View, position, brush) -> float:
        if brush.size_unit == "SCENE":
            return float(brush.scene_size) * 0.5
        camera = view.camera
        return float(brush.size) * 0.5 * camera.world_per_pixel(camera.depth_of(position), view.height, view.aspect)

    def hover(self, view: View, x: float, y: float, brush=None) -> bool:
        """更新笔刷光标。返回鼠标下是否有表面。"""
        hit = view.surface_at(x, y)
        previous = view.cursor
        if hit is None or brush is None:
            view.cursor = None
        else:
            position, normal, _set_index = hit
            view.cursor = (position, normal, self.brush_radius_world(view, position, brush), float(brush.hardness))
        if (previous is None) != (view.cursor is None) or view.cursor is not None:
            view.dirty = True
        return hit is not None

    def pick(self, view: View, x: float, y: float):
        return view.surface_at(x, y)

    def uv_hit(self, view: View, x: float, y: float):
        """鼠标下的表面：(纹理集, (u, v), 物体, 三角形序号) 或 None（按模型数据算射线，不用读显卡）。"""
        from ..bake.parts import ray_hit

        if self.project is None:
            return None
        origin, direction = view.camera.ray(x, y, view.width, view.height)
        best = None
        for obj in self.project.objects:
            if not obj.visible or not getattr(obj.data, "has_uvs", True):
                continue
            tri, u, v, t = ray_hit(obj.data.positions, origin, direction)
            if tri >= 0 and (best is None or t < best[0]):
                best = (t, obj, tri, u, v)
        if best is None:
            return None
        _t, obj, tri, u, v = best
        mats = np.asarray(obj.data.material_ids)
        material = int(mats[tri]) if len(mats) else 0
        if material >= len(obj.material_sets):
            return None
        ts = self.project.texture_set(obj.material_sets[material])
        if ts is None:
            return None
        corners = np.asarray(obj.data.uvs, np.float64).reshape(-1, 3, 2)[tri]
        uv = corners[0] * (1.0 - u - v) + corners[1] * u + corners[2] * v
        return ts, (float(uv[0]), float(uv[1])), obj, int(tri)

    def part_at(self, view: View, x: float, y: float):
        """鼠标下是哪套贴图的哪个部件：(纹理集, 部件号) 或 None。部件号按烘焙的 ID 贴图，-1 表示还没分部件。"""
        from ..bake.parts import ray_hit

        if self.project is None:
            return None
        origin, direction = view.camera.ray(x, y, view.width, view.height)
        best = None
        for obj in self.project.objects:
            if not obj.visible:
                continue
            tri, u, v, t = ray_hit(obj.data.positions, origin, direction)
            if tri >= 0 and (best is None or t < best[0]):
                best = (t, obj, tri, u, v)
        if best is None:
            return None
        _t, obj, tri, u, v = best
        mats = np.asarray(obj.data.material_ids)
        material = int(mats[tri]) if len(mats) else 0
        if material >= len(obj.material_sets):
            return None
        ts = self.project.texture_set(obj.material_sets[material])
        if ts is None:
            return None
        maps = self.meshmaps.get(ts.uid)
        if maps is None or "i" not in maps.textures or not maps.part_count:
            return ts, -1
        corners = np.asarray(obj.data.uvs, np.float64).reshape(-1, 3, 2)[tri]
        uv = corners[0] * (1.0 - u - v) + corners[1] * u + corners[2] * v
        return ts, maps.part_at_uv(float(uv[0]), float(uv[1]))

    def sample_color(self, view: View, x: float, y: float):
        """取鼠标下画面的颜色（sRGB，0..1）。"""
        if not (0 <= x < view.width and 0 <= y < view.height) or view.fbo is None:
            return None
        raw = view.fbo.read(viewport=(int(x), view.height - 1 - int(y), 1, 1), components=3, alignment=1)
        return tuple(v / 255.0 for v in raw[:3])

    # ================================================================== 笔划
    def stroke_begin(self, view: View, x: float, y: float, pressure: float, when: float, tool_settings,
                     erase: bool = False) -> bool:
        if self.stroke is not None:
            self._end_active()
        while len(self.merges) >= MAX_OVERLAYS:      # 叠加层用满了：最早的一笔同步做完
            self._complete_one()
        self.stroke_reject = ""
        if self.project is None:
            return False
        if getattr(view, "is_2d", False):
            state = self.view2d_state(view)
            if state is None:
                self.stroke_reject = "这里没有可以画的贴图"
                return False
        else:
            hit = view.surface_at(x, y)
            if hit is None:
                return False
            set_index = hit[2]
            state = next((s for s in self.sets.values() if s.index == set_index), None)
            if state is None:
                self.stroke_reject = "这个模型没有 UV，不能绘制。可以在属性的「模型」页自动展开 UV"
                return False
        ts = state.ts
        layer = ts.active_layer
        if layer is None:
            self.stroke_reject = "先新建一个绘制图层"
            return False
        if layer.locked:
            self.stroke_reject = "图层「%s」已锁定" % layer.name
            return False
        if not layer.visible:
            self.stroke_reject = "图层「%s」已隐藏" % layer.name
            return False
        target_mask = ts.paint_target == "MASK" and layer.has_mask
        if layer.kind != "PAINT" and not target_mask:
            reasons = {"FILL": "「%s」是填充层，不能直接画。可以给它加蒙版，或新建绘制图层",
                       "FOLDER": "「%s」是文件夹，不能直接画。可以给它加蒙版，或者选中里面的绘制图层",
                       "ADJUST": "「%s」是调整层，不能直接画。可以给它加蒙版，控制调整落在哪里"}
            self.stroke_reject = reasons.get(layer.kind, reasons["FILL"]) % layer.name
            return False
        if self.project.active_set_uid != ts.uid:
            self.project.set_active_set(ts)
        from ..tools.paint_tools import EFFECT_INVERSE, EFFECT_TOOLS

        effect = EFFECT_TOOLS.get(tool_settings.tool, "")
        invert_effect = False
        if effect and erase and tool_settings.tool != "paint.eraser":
            # 效果笔刷按住 Ctrl：换成相反的效果，而不是擦
            invert_effect = True
            effect = EFFECT_INVERSE.get(effect, effect)
            erase = False
        brush = tool_settings.eraser if erase else tool_settings.brush
        params = StrokeParams(color=brush.color, metallic=brush.metallic, roughness=brush.roughness,
                              height=brush.height, mask_value=0.0 if (erase and target_mask) else brush.mask_value,
                              use_basecolor=brush.use_basecolor, use_metallic=brush.use_metallic,
                              use_roughness=brush.use_roughness, use_height=brush.use_height, opacity=brush.opacity,
                              erase=erase and not target_mask, target_mask=target_mask,
                              mask_default=float(layer.mask_default))
        if erase and not target_mask:
            params.use_basecolor = params.use_metallic = params.use_roughness = params.use_height = True
        if effect:
            values = self._effect_values(effect, tool_settings.effect, view, x, y, ts)
            if values is None:
                return False
            if effect == "SPONGE" and invert_effect:
                values["saturate"] = not values.get("saturate", False)
            params.effect = effect
            params.effect_values = values
            if effect == "SPONGE" and not target_mask:
                params.use_metallic = params.use_roughness = params.use_height = False
        if not params.planes():
            self.stroke_reject = "笔刷没有启用任何通道"
            return False
        curve = float(self.prefs.paint.pressure_curve) if self.prefs is not None else 1.0
        snapshot = {"size": float(brush.size), "size_unit": brush.size_unit, "scene_size": float(brush.scene_size),
                    "flow": float(brush.flow), "hardness": float(brush.hardness), "spacing": float(brush.spacing),
                    "smooth": float(brush.smooth), "pressure_size": bool(brush.pressure_size),
                    "pressure_flow": bool(brush.pressure_flow), "backface_cull": bool(brush.backface_cull),
                    "normal_falloff": float(brush.normal_falloff), "pressure_curve": curve}
        sym = tool_settings.symmetry
        self.stroke = ActiveStroke(self, view, state, layer, snapshot, params, (sym.x, sym.y, sym.z))
        if getattr(view, "is_2d", False) and brush.size_unit == "SCENE":
            from .export_normal import surface_scale
            self.stroke.uv_world = surface_scale(self.project, ts)[0]
        self.stroke.add(x, y, pressure, when)
        self._pending_work = True
        return True

    def surface_uv(self, view: View, x: float, y: float):
        """鼠标下的贴图位置 (u, v)：UV 视图直接换算，三维视口打一条射线。没打到时 None。"""
        if getattr(view, "is_2d", False):
            u, v = view.pixel_to_uv(x, y)
            return float(u), float(v)
        hit = self.uv_hit(view, x, y)
        return hit[1] if hit is not None else None

    def _effect_values(self, effect: str, settings, view: View, x: float, y: float, ts):
        """效果笔刷这一笔的参数（不能开始时设 stroke_reject 并返回 None）。"""
        from ..doc.project import TONE_RANGE_CODE

        values = {"strength": float(settings.strength)}
        if effect in ("DODGE", "BURN"):
            values.update(range=TONE_RANGE_CODE.get(settings.tone_range, 1), protect=bool(settings.protect_tones))
        elif effect == "SPONGE":
            values.update(saturate=settings.sponge_mode == "SATURATE", vibrance=bool(settings.vibrance))
        elif effect == "BLUR":
            values["radius"] = float(settings.blur_radius)
        elif effect == "SHARPEN":
            values["amount"] = float(settings.sharpen_amount)
        elif effect in ("CLONE", "HEAL"):
            if not settings.has_source:
                self.stroke_reject = "先按住 Alt 点一下，定下从哪里仿制"
                return None
            start = self.surface_uv(view, x, y)
            if start is None:
                self.stroke_reject = "没点到模型"
                return None
            if not (settings.clone_aligned and settings.offset_set):
                settings.offset_u = settings.source_u - start[0]
                settings.offset_v = settings.source_v - start[1]
                settings.offset_set = True
            values["offset"] = (settings.offset_u * ts.size, settings.offset_v * ts.size)
            if effect == "HEAL":
                values["radius"] = max(4.0, float(settings.blur_radius) * 3.0)
        return values

    def stroke_move(self, x: float, y: float, pressure: float, when: float) -> None:
        if self.stroke is not None and not self.stroke.ended:
            self.stroke.add(x, y, pressure, when)
            self._pending_work = True

    def stroke_end(self) -> None:
        if self.stroke is not None:
            self.stroke.ended = True
            self._pending_work = True

    def stroke_cancel(self) -> None:
        stroke = self.stroke
        if stroke is None:
            return
        self.stroke = None
        if stroke.params.effect:
            # 效果笔刷已经改了图层：放回原来的页
            records = stroke.effect_records()
            self.layers.apply_records(records, undo=True)
            self.layers.dispose_records(records, applied=False)
            for mip, affected in LayerPixels.affected(records).items():
                stroke.state.display.mark_dirty(mip, affected)
            self._touch_content()
        display = stroke.state.display
        for mip, level in enumerate(stroke.buffer.store.levels):
            if level.pages:
                tiles = np.fromiter((ty * level.size + tx for (tx, ty) in level.pages), np.int64, len(level.pages))
                display.mark_dirty(mip, tiles)
        stroke.buffer.release(self.cache)
        for view in self.views:
            view.dirty = True
        self._pending_work = True

    def _collect_dabs(self) -> np.ndarray:
        """把攒下的输入变成笔触（要读几何缓冲，放在一帧最前面做）。"""
        started = time.perf_counter()
        dabs = self.stroke.take_dabs()
        self._collect_ms = (time.perf_counter() - started) * 1000.0
        return dabs

    def _advance_effect(self, stroke: ActiveStroke, dabs: np.ndarray) -> None:
        """效果笔刷：这一帧的笔触画到一张临时笔划页上当遮罩，马上按效果改图层（来回划越改越多）。"""
        from .filters import LayerFilter
        from .stamp import StrokeBuffer

        params = stroke.params
        grid = stroke.state.display.grid
        values = dict(params.effect_values)
        if params.effect == "SMUDGE":
            # 涂抹：把上一帧鼠标下的颜色拖到这一帧的位置
            last = stroke.last
            uv = self.surface_uv(stroke.view, last[0], last[1]) if last is not None else None
            previous, stroke.smudge_uv = stroke.smudge_uv, uv
            if uv is None or previous is None:
                return
            size = stroke.state.ts.size
            values["offset"] = ((previous[0] - uv[0]) * size, (previous[1] - uv[1]) * size)
        frame = StrokeBuffer(self.ctx, grid)
        selection = self._paint_selection(stroke)
        values["ignore_selection"] = True            # 这一帧的笔触已经只盖在选区里，滤镜不用再按选区过渡
        try:
            dilate = bool(self.prefs.paint.stroke_dilate) if self.prefs is not None else True
            touched = []
            for set_gpu in stroke.state.set_gpus:
                tiles = self.stamper.stamp(set_gpu, frame, dabs, dilate, selection)
                if len(tiles):
                    touched.append(tiles)
            if not touched:
                return
            tiles = np.unique(np.concatenate(touched))
            values["cells"] = [(int(t % grid), int(t // grid)) for t in tiles]
            values["mask_store"] = frame.store
            planes = {}
            if params.target_mask:
                planes[PLANE_MASK] = (True,)
            else:
                if params.use_basecolor:
                    planes[PLANE_COLOR] = (True,)
                if params.use_metallic or params.use_roughness:
                    planes[PLANE_MR] = (bool(params.use_metallic), bool(params.use_roughness))
                if params.use_height:
                    planes[PLANE_HEIGHT] = (True,)
            store = self.layers.stores.get(stroke.layer.uid)
            if params.effect not in ("CLONE",) and (store is None or not any(p in store.planes for p in planes)):
                return                                  # 图层上还没有内容，改不出什么
            records = LayerFilter(self, stroke.state.ts, stroke.layer, params.effect, values, planes).run()
            if records:
                stroke.absorb_effect(records, self.cache)
                display = stroke.state.display
                for mip, affected in LayerPixels.affected(records).items():
                    display.mark_dirty(mip, affected)
                self._touch_content()
                for view in self.views:
                    view.dirty = True
        finally:
            frame.release(self.cache)

    def _advance_stroke(self, dabs: np.ndarray) -> None:
        stroke = self.stroke
        started = time.perf_counter() - self._collect_ms / 1000.0
        if len(dabs) and stroke.params.effect:
            self._advance_effect(stroke, dabs)
            self.perf.last_event_time = stroke.oldest_event
        elif len(dabs):
            dilate = bool(self.prefs.paint.stroke_dilate) if self.prefs is not None else True
            display = stroke.state.display
            selection = self._paint_selection(stroke)
            for set_gpu in stroke.state.set_gpus:
                tiles = self.stamper.stamp(set_gpu, stroke.buffer, dabs, dilate, selection)
                if len(tiles):
                    for mip, affected in enumerate(self.stamper.update_mips(stroke.buffer, tiles)):
                        display.mark_dirty(mip, affected)
                    fresh = tiles[~stroke.seen[tiles]]
                    if len(fresh):
                        stroke.seen[fresh] = True
                        self._prefetch_layer(stroke, fresh)
            self.perf.last_event_time = stroke.oldest_event
        self.perf.dabs.append(len(dabs))
        self.perf.stamp_ms.append((time.perf_counter() - started) * 1000.0)
        if stroke.ended and not stroke.samples:
            self._end_active()

    def _prefetch_layer(self, stroke: ActiveStroke, tiles: np.ndarray) -> None:
        """笔划刚碰到的格子：提前把图层在这些位置（各级）的旧页准备进显存，抬笔合并时就不用等。"""
        store = self.layers.stores.get(stroke.layer.uid)
        if store is None:
            return
        want = []
        frame = self.frame_index
        planes = [store.planes.get(plane) for plane in stroke.params.planes()]
        planes = [p for p in planes if p is not None]
        if not planes:
            return
        size = stroke.state.display.grid
        xs, ys = tiles % size, tiles // size
        for mip in range(len(planes[0].levels)):
            for pstore in planes:
                level = pstore.levels[mip]
                exists = level.exists[ys, xs]
                if not exists.any():
                    continue
                slots = level.slots[ys, xs]
                held = exists & (slots >= 0)
                if held.any():
                    self.pools[pstore.fmt].touch(slots[held], frame)
                for j in np.flatnonzero(exists & (slots < 0)):
                    want.append(level.pages[(int(xs[j]), int(ys[j]))])
            if size <= 1:
                break
            parents = np.unique((ys >> 1) * (size >> 1) + (xs >> 1))
            size >>= 1
            xs, ys = parents % size, parents // size
        if want:
            self.cache.request(want)

    def _end_active(self) -> None:
        """当前这一笔结束：交给合并队列（效果笔刷已经改好了，直接记一步撤销）。"""
        stroke = self.stroke
        if stroke is None:
            return
        self.stroke = None
        if stroke.params.effect:
            from ..tools.paint_tools import EFFECT_LABELS

            stroke.ended = True
            records = stroke.effect_records()
            if records:
                self._push_pixel_step(EFFECT_LABELS.get(stroke.params.effect, "效果笔刷"), stroke.state, records)
                if self.project is not None:
                    self.project.mark_dirty()
            stroke.buffer.release(self.cache)
            self._pending_work = True
            return
        stroke.ended = True
        stroke.ended_at = time.perf_counter()
        self.merges.append(stroke)
        self._pending_work = True

    def _advance_merges(self, budget_ms: float) -> None:
        started = time.perf_counter()
        while self.merges:
            left = budget_ms - (time.perf_counter() - started) * 1000.0
            if left <= 0.0:
                break
            stroke = self.merges[0]
            if stroke.merge is None:
                stroke.merge = self.layers.merge_job(stroke.layer.uid, stroke.state.display.grid, stroke.buffer,
                                                     stroke.params)
            if not stroke.merge.step(left, self.merge_pages):
                break
            self._commit_merge(stroke)
        self.perf.merge_frame_ms.append((time.perf_counter() - started) * 1000.0)
        self._pending_work = True

    def _complete_one(self) -> None:
        """把最早的一笔立刻并完（同步）。"""
        stroke = self.merges[0]
        if stroke.merge is None:
            stroke.merge = self.layers.merge_job(stroke.layer.uid, stroke.state.display.grid, stroke.buffer,
                                                 stroke.params)
        stroke.merge.step(block=True)
        self._commit_merge(stroke)

    def _commit_merge(self, stroke: ActiveStroke) -> None:
        """合并做完：页表一次换成新页，记一步撤销。显示贴图已经是叠加后的样子，不用重新合成。"""
        if self.merges and self.merges[0] is stroke:
            self.merges.popleft()
        records = stroke.merge.commit()
        stroke.buffer.release(self.cache)
        self._touch_content()
        if records and stroke.layer in stroke.state.ts.layers:
            was = self._completing
            self._completing = True
            try:
                self._push_pixel_step("橡皮" if stroke.params.erase else "笔划", stroke.state, records)
            finally:
                self._completing = was
            if self.project is not None:
                self.project.mark_dirty()
        self.perf.merge_ms.append((time.perf_counter() - stroke.ended_at) * 1000.0)
        self._pending_work = True

    def complete_strokes(self, include_active: bool = False) -> None:
        """把等着合并的笔划全部同步做完（撤销、存盘、导出、改图层结构之前调用）。"""
        if self._completing:
            return
        self._completing = True
        try:
            if include_active and self.stroke is not None:
                self._end_active()
            while self.merges:
                self._complete_one()
        finally:
            self._completing = False

    def _push_pixel_step(self, label: str, state: SetState, records: list) -> None:
        if self.history is None:
            return
        flag = {"applied": True}
        affected = LayerPixels.affected(records)

        def mark() -> None:
            self._touch_content()
            for mip, tiles in affected.items():
                state.display.mark_dirty(mip, tiles)
            for view in self.views:
                view.dirty = True
            self._pending_work = True
            if self.project is not None:
                self.project.mark_dirty()

        def undo() -> None:
            self.layers.apply_records(records, undo=True)
            flag["applied"] = False
            mark()

        def redo() -> None:
            self.layers.apply_records(records, undo=False)
            flag["applied"] = True
            mark()

        def dispose() -> None:
            self.layers.dispose_records(records, flag["applied"])

        self.history.push(label, undo, redo, self.layers.records_bytes(records), dispose)
        self.history.steps[-1].records = records

    # ================================================================== 显示维护
    def _layer_params(self, state: SetState) -> None:
        ts = state.ts
        params = np.zeros(MAX_LAYERS, PARAM_DTYPE)
        layers = ts.layers[:MAX_LAYERS]
        by_uid = {layer.uid: layer for layer in layers}

        def shown(layer) -> bool:
            """自己和所有上级文件夹都可见。"""
            for _ in range(64):
                if not layer.visible:
                    return False
                if not layer.parent_uid:
                    return True
                layer = by_uid.get(layer.parent_uid)
                if layer is None:
                    return True
            return True

        state.hidden_rows = set()
        selection = None
        luts = []
        lut_rows = 0
        state.graph_slots = []
        state.material_values_dirty = True          # 节点图的贴图槽重新排过
        for index, layer in enumerate(layers):
            row = params[index]
            visible = shown(layer)
            if not visible:
                state.hidden_rows.add(index)
            row["fill_color"] = (*layer.fill_color, layer.opacity_height)
            row["fill_mrh"] = (layer.fill_metallic, layer.fill_roughness, layer.fill_height, layer.mask_default)
            row["blend"] = (BLEND_CODE[layer.blend_basecolor], BLEND_CODE[layer.blend_metallic],
                            BLEND_CODE[layer.blend_roughness], BLEND_CODE[layer.blend_height])
            row["opacity"] = (layer.opacity if visible else 0.0, layer.opacity_basecolor, layer.opacity_metallic,
                              layer.opacity_roughness)
            use = (int(layer.use_basecolor) | int(layer.use_metallic) << 1 | int(layer.use_roughness) << 2
                   | int(layer.use_height) << 3)
            code = GENERATOR_CODE.get(layer.mask_generator, 0)
            kind = LAYER_KIND_CODE.get(layer.kind, 0)
            row["flags"] = (kind, use, int(layer.has_mask and layer.mask_enabled), code)
            graph_slot = self._graph_slot(state, layer) if kind == 1 else 0
            row["flags2"] = (min(ts.depth(layer), MAX_FOLDER_DEPTH), graph_slot, -1, 0)
            if graph_slot:
                row["graph"] = (max(1e-3, float(layer.graph_tiling)), float(layer.graph_offset_u),
                                float(layer.graph_offset_v), float(layer.graph_height))
            if kind == 3:
                row["adj"], row["adj2"], row["adj3"], row["adj4"], table = adjust_params(layer)
                if table is not None:
                    row["flags2"][2] = lut_rows
                    luts.append(table)
                    lut_rows += len(table) // LUT_SIZE
            if code:
                maps = self.meshmaps.get(ts.uid)
                lo, hi = maps.bounds if maps is not None else (np.zeros(3), np.ones(3))
                diag = float(np.linalg.norm(np.asarray(hi, np.float64) - np.asarray(lo, np.float64))) or 1.0
                frequency = 1.0 / max(1e-6, float(layer.gen_noise_size) * diag)
                row["gen"] = (1.0 - layer.gen_range, layer.gen_softness * 0.5, layer.gen_noise, frequency)
                row["gen2"] = (*DIRECTION_VECTOR.get(layer.gen_direction, (0.0, 1.0, 0.0)),
                               1.0 if layer.gen_invert else 0.0)
                offset = np.random.default_rng(int(layer.gen_seed) + 7).random(3) * 97.0
                row["gen3"] = (*offset, 0.0)
            if layer.mask_generator == "PARTS":
                from ..bake.parts import parse_ids

                if selection is None:
                    selection = np.zeros((MAX_LAYERS, MAX_PART_IDS), np.uint8)
                ids = np.asarray([i for i in parse_ids(layer.gen_parts) if i < MAX_PART_IDS], np.int64)
                selection[index, ids] = 255
        state.idsel = selection
        state.idsel_dirty = True
        state.luts = np.concatenate(luts) if luts else None
        state.luts_version = getattr(state, "luts_version", 0) + 1
        state.params = params
        state.params_dirty = False

    def _graph_slot(self, state: SetState, layer) -> int:
        """填充层用的节点图在这套贴图里的序号 + 1（0 表示不用节点图、图没有材质输出或者放不下）。"""
        uid = int(layer.fill_graph or 0)
        graph = self.project.graph(uid) if (uid and self.project is not None) else None
        if graph is None or graph.output_node() is None:
            return 0
        if uid not in state.graph_slots:
            if len(state.graph_slots) >= MAX_GRAPHS:
                return 0
            state.graph_slots.append(uid)
        return state.graph_slots.index(uid) + 1

    def _graph_textures(self, state: SetState) -> list:
        out = []
        for uid in getattr(state, "graph_slots", []):
            graph = self.project.graph(uid) if self.project is not None else None
            texture = None
            if graph is not None:
                try:
                    texture = self.node_graphs.output(graph)
                except Exception:  # noqa: BLE001
                    log.exception("节点图「%s」求值出错", graph.name)
            out.append(texture)
        return out

    def _on_graph_changed(self, graph, kind: str) -> None:
        """节点图变了：用到它的纹理集重新合成（增删图时每套都重新整理图层参数）。"""
        if kind in ("active", "layout"):
            return
        for state in self.sets.values():
            if graph is None or kind == "list":
                state.params_dirty = True
                state.display.mark_all_dirty()
            elif graph.uid in getattr(state, "graph_slots", []):
                if kind == "structure":
                    state.params_dirty = True
                state.display.mark_all_dirty()
        if graph is None and self.project is not None:
            alive = {g.uid for g in self.project.graphs}
            for uid in list(self.node_graph_uids):
                if uid not in alive:
                    self.node_graphs.forget(uid)
            self.node_graph_uids = set(alive)
        elif graph is not None:
            self.node_graph_uids.add(graph.uid)
            self.node_graphs.forget(graph.uid, keep=set(graph.nodes))
        self._touch_content()
        for view in self.views:
            view.dirty = True
        self._pending_work = True

    def _selection_texture(self, state: SetState):
        """「部件」生成器的选择表，按需建、改了才传。"""
        selection = getattr(state, "idsel", None)
        if selection is None:
            return None
        texture = getattr(state, "idsel_tex", None)
        if texture is None:
            texture = self.ctx.texture((MAX_PART_IDS, MAX_LAYERS), 1, dtype="f1")
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            state.idsel_tex = texture
            state.idsel_dirty = True
        if getattr(state, "idsel_dirty", True):
            texture.write(np.ascontiguousarray(selection).tobytes())
            state.idsel_dirty = False
        return texture

    def _overlays(self, state: SetState) -> list[ActiveStroke]:
        """这个纹理集上要叠加显示的笔划：还在合并的，加上正在画的，按先后顺序。"""
        found = [s for s in self.merges if s.state is state]
        if self.stroke is not None and self.stroke.state is state and not self.stroke.params.effect:
            found.append(self.stroke)
        layers = state.ts.layers
        return [s for s in found if s.layer in layers and layers.index(s.layer) < MAX_LAYERS][-MAX_OVERLAYS:]

    def _material(self, state: SetState):
        """这套贴图的材质节点：结构变了重新编译，数值变了重新填参数。没有材质节点时返回 None。"""
        from ..shading.compile import compile_graph

        graph = getattr(state.ts, "material", None)
        if getattr(state, "material_dirty", True) or getattr(state, "material_graph", None) is not graph:
            try:
                state.material_compiled = compile_graph(graph)
            except Exception:  # noqa: BLE001
                log.exception("材质节点编译出错")
                state.material_compiled = None
            state.material_graph = graph
            state.material_dirty = False
            state.material_values_dirty = True
        compiled = state.material_compiled
        if compiled is not None and getattr(state, "material_values_dirty", True):
            state.material_values = compiled.values(graph, lambda uid: self._material_graph_slot(state, uid))
            state.material_values_dirty = False
        return compiled

    def _material_graph_slot(self, state: SetState, graph_uid: int) -> int:
        """材质节点里「节点图纹理」用的节点图占哪个贴图槽（和填充层用的共用这几个槽）。"""
        graph = self.project.graph(graph_uid) if self.project is not None else None
        if graph is None or graph.output_node() is None:
            return -1
        slots = state.graph_slots
        if graph_uid not in slots:
            if len(slots) >= MAX_GRAPHS:
                return -1
            slots.append(graph_uid)
        return slots.index(graph_uid)

    def _prepare_composite(self, state: SetState, with_stroke: bool = True, ao_mix: float | None = None) -> None:
        """ao_mix：环境遮蔽叠多少（None 按纹理集的显示设置）。"""
        if state.params_dirty:
            self._layer_params(state)
        compiled = self._material(state)
        self.compositor.set_material(compiled, getattr(state, "material_values", None))
        self.compositor.set_layers(state.params, min(len(state.ts.layers), MAX_LAYERS), state.ts.base_color,
                                   (state.ts.base_metallic, state.ts.base_roughness))
        self.compositor.set_luts(getattr(state, "luts", None), (id(state), getattr(state, "luts_version", 0)))
        settings = state.ts.meshmap
        if ao_mix is None:
            ao_mix = float(settings.ao_strength) if settings.show_ao else 0.0
        self.compositor.set_maps(self.meshmaps.get(state.ts.uid), ao_mix)
        self.compositor.set_selection(self._selection_texture(state))
        self.compositor.set_graphs(self._graph_textures(state))
        overlays = []
        if with_stroke:
            for stroke in self._overlays(state):
                p = stroke.params
                overlays.append((state.ts.layers.index(stroke.layer), p.color, p.values,
                                 (int(p.use_basecolor), int(p.use_metallic), int(p.use_roughness), int(p.use_height)),
                                 p.opacity, p.erase, p.target_mask))
        self.compositor.set_overlays(overlays)

    def _build_table(self, state: SetState, level: int, tx: np.ndarray, ty: np.ndarray, include_stroke: bool = True):
        count = len(tx)
        table = np.full((count, LAYER_STRIDE, 4), -1, np.int32)
        ready = np.ones(count, bool)
        wanted = []
        pools = self.pools
        frame = self.frame_index
        if state.params_dirty:
            self._layer_params(state)
        hidden = getattr(state, "hidden_rows", set())
        for index, layer in enumerate(state.ts.layers[:MAX_LAYERS]):
            store = self.layers.stores.get(layer.uid)
            if store is None or index in hidden:
                continue
            columns = []
            if layer.kind == "PAINT":
                columns = [(0, PLANE_COLOR), (1, PLANE_MR), (2, PLANE_HEIGHT)]
            if layer.has_mask and layer.mask_enabled:
                columns.append((3, PLANE_MASK))
            for column, plane in columns:
                pstore = store.planes.get(plane)
                if pstore is None:
                    continue
                grid = pstore.levels[level]
                slots = grid.slots[ty, tx]
                table[:, index, column] = slots
                missing = grid.exists[ty, tx] & (slots < 0)
                if missing.any():
                    ready &= ~missing
                    for j in np.nonzero(missing)[0]:
                        wanted.append(grid.pages[(int(tx[j]), int(ty[j]))])
                held = slots[slots >= 0]
                if len(held):
                    pools[pstore.fmt].touch(held, frame)
        if include_stroke:
            for k, stroke in enumerate(self._overlays(state)):
                table[:, MAX_LAYERS, k] = stroke.buffer.store.levels[level].slots[ty, tx]
        if wanted:
            self.cache.request(wanted)
        return table, ready

    def _update_visibility(self) -> None:
        self._visibility_dirty = False
        for state in self.sets.values():
            display = state.display
            visible = [np.zeros((g, g), bool) for g in display.level_size]
            for view in self.views:
                requests = getattr(view, "requests", None)
                if requests is None or len(requests) == 0:
                    continue
                rows = requests[requests[:, 3] == state.index + 1]
                for level in np.unique(rows[:, 2]):
                    if level >= display.levels:
                        continue
                    part = rows[rows[:, 2] == level]
                    g = display.level_size[level]
                    visible[level][np.minimum(part[:, 1], g - 1), np.minimum(part[:, 0], g - 1)] = True
            for level in range(display.levels - 1):          # 上一级也要在（三线性过滤和兜底）
                g = display.level_size[level]
                if g >= 2:
                    visible[level + 1] |= visible[level].reshape(g // 2, 2, g // 2, 2).any(axis=(1, 3))
            for level in range(display.levels):              # 向外扩一圈（过滤会采到相邻页）
                v = visible[level]
                if v.any() and v.shape[0] > 1:
                    grown = v.copy()
                    grown[1:, :] |= v[:-1, :]
                    grown[:-1, :] |= v[1:, :]
                    grown[:, 1:] |= grown[:, :-1].copy()
                    grown[:, :-1] |= grown[:, 1:].copy()
                    visible[level] = grown
            for level in range(display.base_level, display.levels):
                visible[level][:] = True
            state.visible = visible
            for level in range(display.levels):
                display.needed[level][visible[level]] = self.frame_index

    def _update_display(self, state: SetState, budget: int) -> int:
        display = state.display
        composed = 0
        prepared = False
        pending = False
        for level in range(display.levels - 1, -1, -1):
            visible = state.visible[level]
            lack = visible & ~display.committed[level]
            if lack.any():
                tiles = np.flatnonzero(lack)
                display.commit(level, tiles[:512])
                pending = True
            todo = visible & display.committed[level] & display.stale[level]
            if not todo.any():
                continue
            tiles = np.flatnonzero(todo)
            if composed >= budget:
                pending = True
                continue
            if len(tiles) > budget - composed:
                tiles = tiles[: budget - composed]
                pending = True
            g = display.level_size[level]
            ty, tx = np.divmod(tiles, g)
            table, ready = self._build_table(state, level, tx, ty)
            if not ready.all():
                pending = True
                tx, ty, table = tx[ready], ty[ready], table[ready]
            if len(tx) == 0:
                continue
            if not prepared:
                self._prepare_composite(state)
                prepared = True
            self.compositor.compose(display, level, tx, ty, table)
            composed += len(tx)
        if composed:
            display.update_clamp()
            for view in self.views:
                view.dirty = True
        # 显示缓存超预算：放掉最久没看的页
        if display.sparse and display.committed_bytes > self.display_budget:
            self._trim_display(state)
        if pending:
            self._pending_work = True
        return composed

    def _trim_display(self, state: SetState) -> None:
        display = state.display
        excess = display.committed_bytes - int(self.display_budget * 0.9)
        for level in range(display.base_level):
            if excess <= 0:
                break
            candidates = display.committed[level] & ~state.visible[level]
            if not candidates.any():
                continue
            tiles = np.flatnonzero(candidates)
            age = display.needed[level].ravel()[tiles]
            order = tiles[np.argsort(age)]
            take = order[: max(1, excess // display.page_bytes)]
            display.decommit(level, take)
            excess -= len(take) * display.page_bytes
        display.update_clamp()

    # ================================================================== 每帧
    def frame(self) -> list[View]:
        """推进一帧。返回这一帧重新渲染过的视口。"""
        started = time.perf_counter()
        # 和页面缓存的帧号对齐（同步的大操作会让缓存的帧号先往前走），两边一起只往前
        self.frame_index = max(self.frame_index + 1, self.cache.frame + 1)
        self._pending_work = False
        # 1. 要等显卡结果的读取放在最前面：这时显卡上还没有本帧的新工作，几乎不用等
        for view in self.views:
            if view.is_2d:
                state = self.view2d_state(view)
                requests = view.compute_requests(state.display, state.index) if state is not None else None
            else:
                requests = self.renderer.read_feedback(view)
            if requests is not None:
                previous = getattr(view, "requests", None)
                if previous is None or len(previous) != len(requests) or not np.array_equal(previous, requests):
                    view.requests = requests
                    self._visibility_dirty = True
        dabs = self._collect_dabs() if self.stroke is not None else None
        # 2. 收回换页结果，上传到货的页（正在画时少花点时间，先保证笔迹跟手）
        painting = self.stroke is not None
        self.cache.begin(self.frame_index, self.upload_budget_painting_ms if painting else self.evict_budget_ms)
        if self._visibility_dirty:
            self._update_visibility()
        # 3. 笔划、合并
        if self.stroke is not None:
            self._advance_stroke(dabs)
        if self.sculpt is not None and self.sculpt.stroke is not None:
            self.sculpt.advance()
        if self.merges:
            self._advance_merges(self.merge_budget_painting_ms if self.stroke is not None else self.merge_budget_ms)
        if self.bake is not None:
            self._step_bake(min(self.bake_budget_ms, 3.0) if (painting or self.sculpt is not None)
                            else self.bake_budget_ms)
        compose_started = time.perf_counter()
        composed = 0
        for state in self.sets.values():
            composed += self._update_display(state, self.compose_budget - composed)
        self.perf.pages_composed.append(composed)
        self.perf.compose_ms.append((time.perf_counter() - compose_started) * 1000.0)
        render_started = time.perf_counter()
        rendered = []
        for view in self.views:
            if not view.is_2d and view.fbo is not None and getattr(view.shading, "mode", "") == "RENDERED":
                if self._draw_rendered(view):
                    view.frame_rendered = self.frame_index
                    rendered.append(view)
                    self._last_render = self.frame_index
                continue
            if view.dirty and view.fbo is not None:
                if view.is_2d:
                    self.renderer2d.render(view, self.view2d_state(view), self.renderer.scene)
                else:
                    self.renderer.render(view)
                    self.renderer.render_feedback(view)
                view.frame_rendered = self.frame_index
                rendered.append(view)
                self._last_render = self.frame_index
        if rendered:
            self._pending_work = True          # 下一帧要读这次的页面需求
        self.perf.render_ms.append((time.perf_counter() - render_started) * 1000.0)
        # 5. 保持显存空闲水位、后台备份：都会让显卡读回，排在本帧渲染之后
        idle = self.stroke is None and not self.merges and self.frame_index - self._last_render > 5
        self.cache.maintain(idle)
        # 6. 自动保存：笔划进行中不动（写盘线程也暂停），别的时候每帧花一点时间取页面数据交给写盘线程
        saver = self.__dict__.get("_autosave")
        if saver is not None and saver.job is not None:
            stroking = self.stroke is not None or (self.sculpt is not None and self.sculpt.stroke is not None)
            saver.set_paused(stroking)
            if not stroking:
                saver.pump(self.autosave_budget_ms)
        glx.flush()
        now = time.perf_counter()
        self.perf.frame_ms.append((now - started) * 1000.0)
        if self.perf.last_event_time:
            self.perf.latency_ms.append((now - self.perf.last_event_time) * 1000.0)
            self.perf.last_event_time = 0.0
        return rendered

    # ================================================================== 模型贴图（烘焙）
    def bake_start(self, ts: TextureSet):
        """开始烘焙这套贴图的模型贴图（按 ts.meshmap 的设置），之后每帧推进。返回任务。"""
        from ..bake.baker import BakeJob

        self.bake_cancel()
        self.complete_strokes(include_active=True)
        job = BakeJob(self, ts, ts.meshmap)
        self.bake = job
        self._pending_work = True
        log.info("开始烘焙模型贴图：%s，%s²，遮蔽采样 %d", ts.name, ts.meshmap.resolution, ts.meshmap.ao_samples)
        return job

    def bake_cancel(self) -> None:
        job = self.bake
        if job is None:
            return
        job.cancel()
        job.step(0.0)
        self.bake = None
        for fn in list(self.bake_finished):
            fn(job)

    def _step_bake(self, budget_ms: float) -> None:
        job = self.bake
        self._pending_work = True
        if not job.step(budget_ms):
            return
        self.bake = None
        if job.result is not None:
            self.set_meshmap(job.ts.uid, job.result)
        for fn in list(self.bake_finished):
            fn(job)

    def set_meshmap(self, set_uid: int, maps) -> None:
        """换上（或去掉）一套纹理集的模型贴图，生成器和遮蔽显示都会重新合成。"""
        old = self.meshmaps.pop(set_uid, None)
        if old is not None and old is not maps:
            old.release()
        if maps is not None:
            self.meshmaps[set_uid] = maps
        state = self.sets.get(set_uid)
        if state is not None:
            state.params_dirty = True
            state.display.mark_all_dirty()
        self._touch_content()
        for view in self.views:
            view.dirty = True
        self._pending_work = True

    def meshmap(self, set_uid: int):
        return self.meshmaps.get(set_uid)

    # ================================================================== 贴图搬运（换了 UV 之后）
    def transfer_layers(self, obj, old: dict, distance: float | None = None) -> list:
        """物体的 UV 换过了：把它用到的每套纹理集的图层像素从旧 UV 搬到新 UV。
        old：旧几何 {"positions" (3T,3), "uvs" (3T,2), "material_ids" (T,)}，物体的 MeshGPU 必须已经重建。
        返回 [(纹理集, {图层 uid: (旧 LayerStore, 新 LayerStore)})]，新的已经换上；调用方记撤销。"""
        from .transfer import TextureTransfer

        self.complete_strokes(include_active=True)
        done = []
        project = self.project
        for set_uid in dict.fromkeys(obj.material_sets):
            if project is not None and project.texture_set(set_uid) is not None:
                self.drop_selection(project.texture_set(set_uid))
            state = self.sets.get(set_uid)
            if state is None:
                continue
            tris, uvs = [], []
            for other in project.objects:
                materials = [m for m, uid in enumerate(other.material_sets) if uid == set_uid]
                if not materials:
                    continue
                if other is obj:
                    positions, corner_uvs, mats = old["positions"], old["uvs"], old["material_ids"]
                else:
                    if not other.visible or not getattr(other.data, "has_uvs", True):
                        continue
                    positions, corner_uvs, mats = other.data.positions, other.data.uvs, other.data.material_ids
                keep = np.isin(np.asarray(mats), materials)
                tris.append(np.asarray(positions, np.float32).reshape(-1, 3, 3)[keep])
                uvs.append(np.asarray(corner_uvs, np.float32).reshape(-1, 3, 2)[keep])
            if not tris:
                continue
            new_sets = list(state.set_gpus)
            # 形状没变的模型（只换了 UV，或者根本没动）：旧 UV 直接插值，不用发射线
            known = {}
            for set_gpu in new_sets:
                owner = next((o for o in project.objects if self.meshes.get(o.uid) is set_gpu.mesh), None)
                if owner is None:
                    continue
                if owner is obj:
                    same = (np.shape(old["positions"]) == np.shape(owner.data.positions)
                            and np.array_equal(old["positions"], owner.data.positions))
                    if same:
                        known[id(set_gpu)] = old["uvs"]
                else:
                    known[id(set_gpu)] = owner.data.uvs
            job = TextureTransfer(self, state.ts, np.concatenate(tris), np.concatenate(uvs), new_sets, distance, known)
            stores = job.run()
            if not stores:
                continue
            swaps = {}
            for uid, store in stores.items():
                previous = self.layers.detach_layer(uid)
                self.layers.attach_layer(uid, store)
                swaps[uid] = (previous, store)
            done.append((state.ts, swaps))
        self._after_store_swap()
        return done

    def swap_layer_stores(self, swaps: list, use_new: bool) -> None:
        """撤销、重做贴图搬运：把每个图层的像素换成旧的或新的。"""
        self.complete_strokes(include_active=True)
        for _ts, pairs in swaps:
            for uid, (previous, new) in pairs.items():
                self.layers.detach_layer(uid)
                target = new if use_new else previous
                if target is not None:
                    self.layers.attach_layer(uid, target)
        self._after_store_swap()

    def free_layer_stores(self, swaps: list, keep_new: bool) -> None:
        """撤销步骤被丢弃：用着新像素就释放旧的，反之释放新的。"""
        for _ts, pairs in swaps:
            for _uid, (previous, new) in pairs.items():
                self.layers.free_store(previous if keep_new else new)

    def _after_store_swap(self) -> None:
        for state in self.sets.values():
            state.display.mark_all_dirty()
        self._touch_content()
        for view in self.views:
            view.dirty = True
        self._pending_work = True

    @property
    def needs_frame(self) -> bool:
        if self._pending_work or self.cache.busy or self.merges or self.bake is not None:
            return True
        saver = self.__dict__.get("_autosave")
        if saver is not None and saver.job is not None:
            return True
        if self.stroke is not None and (self.stroke.samples or self.stroke.ended):
            return True
        if any(view.dirty or view.feedback_pending for view in self.views):
            return True
        if any(entry[3] for entry in self._rendered.values()):
            return True
        return self.cache.has_background_work()

    # ================================================================== 「渲染」着色模式
    def _render_request(self, view):
        from ..render import engines
        from ..render.engines import RenderRequest

        props = getattr(self.project, "render", None)
        idname = getattr(props, "engine", "PATHTRACE")
        cls = engines.engine(idname)
        if cls is None or not cls.supports_viewport or not cls.available()[0]:
            idname = "PATHTRACE"
            cls = engines.engine(idname)
        settings = props.engine_settings(idname) if props is not None else None
        return cls, RenderRequest(app_engine=self, view=view, shading=view.shading, engine_settings=settings,
                                  render_props=props)

    def _draw_rendered(self, view) -> bool:
        """推进一个「渲染」着色模式的视口。返回这一帧有没有画。"""
        if self.project is None:
            return False
        cls, request = self._render_request(view)
        if cls is None:
            return False
        shading = view.shading
        settings = request.engine_settings
        camera_state = view.camera.state()
        key = {
            "camera": str(sorted(camera_state.items())) if isinstance(camera_state, dict) else str(camera_state),
            "size": (view.width, view.height),
            "environment": (shading.hdri, round(float(shading.hdri_rotation), 4), round(float(shading.hdri_strength), 4)),
            "display": (round(float(shading.exposure), 4), shading.view_transform),
            "settings": (cls.idname, str(sorted(settings.to_dict().items())) if settings is not None else ""),
            "objects": tuple(o.uid for o in self.project.objects),
            "content": self.content_version,
        }
        entry = self._rendered.get(id(view))
        if entry is None or entry[1] != cls.idname:
            if entry is not None:
                entry[0].release()
            try:
                preview = cls.create_viewport(request)
            except Exception:  # noqa: BLE001
                log.exception("创建视口渲染失败")
                return False
            if preview is None:
                return False
            entry = [preview, cls.idname, None, True]
            self._rendered[id(view)] = entry
        previous = entry[2]
        changes = set(key) if previous is None else {name for name in key if key[name] != previous.get(name)}
        if "content" in changes and previous is not None:
            # 正在画或刚改完：先不重建，等停下来
            if self.stroke is not None or self.merges or time.perf_counter() - self._content_changed_at < 0.35:
                changes.discard("content")
                key["content"] = previous["content"]
                self._pending_work = True
        try:
            if changes:
                entry[0].sync(request, changes)
            busy = entry[0].draw(view, float(getattr(settings, "viewport_budget_ms", 12.0) or 12.0))
        except Exception:  # noqa: BLE001
            log.exception("视口渲染出错，改回材质预览")
            entry[0].release()
            self._rendered.pop(id(view), None)
            shading.mode = "MATERIAL"
            return False
        entry[2] = key
        entry[3] = bool(busy)
        view.dirty = False
        view.serial += 1
        return True

    def rendered_status(self, view) -> str:
        entry = self._rendered.get(id(view))
        return entry[0].status() if entry is not None else ""

    @property
    def needs_render(self) -> bool:
        """有没有会改变画面的待办（区别于只是后台备份）。"""
        if self._pending_work or self.merges or (self.stroke is not None and (self.stroke.samples or self.stroke.ended)):
            return True
        if self.bake is not None:
            return True
        if self.sculpt is not None and self.sculpt.stroke is not None and self.sculpt.stroke.samples:
            return True
        if any(entry[3] for entry in self._rendered.values()):
            return True
        return any(view.dirty or view.feedback_pending for view in self.views)

    def settle(self, max_frames: int = 600) -> int:
        """一直推进到画面不再变化（测试、导出、截图前用；不等后台备份）。返回用了多少帧。"""
        for count in range(max_frames):
            self.frame()
            if not self.needs_render:
                return count + 1
        return max_frames

    # ================================================================== 存取与导出
    @property
    def autosave(self):
        """自动保存（第一次用到时建）。"""
        saver = self.__dict__.get("_autosave")
        if saver is None:
            from .autosave import Autosaver

            saver = self._autosave = Autosaver(self)
        return saver

    @property
    def autosave_budget_ms(self) -> float:
        files = getattr(self.prefs, "files", None)
        return float(getattr(files, "autosave_budget_ms", 1.5))

    def save_project(self, path: str, extra: dict | None = None, progress=None) -> dict:
        saver = self.__dict__.get("_autosave")
        if saver is not None:
            saver.cancel()
        info = self._save_project(path, extra, progress)
        if saver is not None:
            saver.reset()                    # 工程文件已经是最新的：恢复文件作废
        return info

    def _save_project(self, path: str, extra: dict | None = None, progress=None) -> dict:
        from .projectio import save_project
        self.complete_strokes(include_active=True)
        if self.sculpt is not None:
            self.sculpt.stroke_end()
            self.sculpt.sync_to_object()
        session = self.edit
        if session is None or not session.changed:
            return save_project(self, self.project, path, extra, progress)
        # 编辑模式里存盘：写进去的是现在编辑着的完整网格
        display = session.obj.data
        session.obj.data = session.full_data()
        session.obj.geometry_dirty = True
        try:
            return save_project(self, self.project, path, extra, progress)
        finally:
            session.obj.data = display

    def load_project(self, path: str):
        from .projectio import load_project
        return load_project(self, path)

    def export_channel(self, ts: TextureSet, channel: str, path: str, progress=None, size: int | None = None) -> dict:
        from .projectio import export_channel
        self.complete_strokes(include_active=True)
        return export_channel(self, ts, channel, path, progress, size)

    def close_storage(self) -> None:
        storage = self.cache.storage
        self.cache.storage = None
        if storage is not None:
            self.cache.wait_idle(5.0)
            try:
                storage.close()
            except Exception:  # noqa: BLE001
                log.exception("关闭工程文件时出错")

    # ================================================================== 图层像素的整层操作
    def detach_layer_pixels(self, layer_uid: int):
        """删除图层时取下它的像素，交给撤销步骤保管。"""
        return self.layers.detach_layer(layer_uid)

    def attach_layer_pixels(self, layer_uid: int, store) -> None:
        if store is not None:
            self.layers.attach_layer(layer_uid, store)

    def free_layer_pixels(self, store) -> None:
        self.layers.free_store(store)

    def duplicate_layer_pixels(self, source_uid: int, target_uid: int) -> None:
        self.layers.duplicate_layer(source_uid, target_uid)

    # ================================================================== 统计
    def stats(self) -> dict:
        info = self.cache.stats()
        info["display_mb"] = sum(s.display.committed_bytes for s in self.sets.values()) / 1048576.0
        info["display_budget_mb"] = self.display_budget >> 20
        info["layer_pages"] = sum(store.page_count() for store in self.layers.stores.values())
        info["merges_pending"] = len(self.merges)
        info["frame"] = self.frame_index
        vram_total, vram_free = glx.vram_info_mb()
        info["vram_total_mb"] = vram_total or self.caps.vram_total_mb
        info["vram_free_mb"] = vram_free
        info.update(self.perf.summary())
        return info

    def release(self) -> None:
        saver = self.__dict__.pop("_autosave", None)
        if saver is not None:
            saver.shutdown()
        self._clear_scene()
        for view in list(self.views):
            view.release()
        self.views.clear()
        self.cache.release()
        self.renderer2d.release()
        self.ants.release()
        self.renderer.release()
        self.compositor.release()
        for program in self.__dict__.get("filter_programs", {}).values():
            program.release()
        for program in self.__dict__.get("bake_programs", {}).values():
            program.release()
        for texture in self.__dict__.get("filter_temps", {}).values():
            texture.release()
        capture = self.__dict__.pop("screen_capture", None)
        if capture is not None:
            capture[1].release()
        for tables in self.__dict__.get("filter_tables", {}).values():
            for buffer in tables:
                buffer.release()
        self.node_graphs.release()
        self.overlays.release()
        self.stamper.release()
        self.ops.release()
        self.pools.release()
