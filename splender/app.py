"""应用对象与启动流程。

Application 持有全程序共用的东西：偏好、键位、工程、撤销历史、工具设置、窗口管理器、引擎宿主。
界面和操作通过 ctx.app 拿到它。
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import time

from . import APP_NAME, __version__
from . import log as applog
from .core.history import History
from .core.keymap import default_keyconfig
from .core.prefs import Preferences
from .doc.project import Layer, MeshObject, Project, TextureSet, ToolSettings
from .paths import ensure_vendor_path, resource, user_dir

log = logging.getLogger("splender.app")

_instance: "Application | None" = None


def instance() -> "Application | None":
    return _instance


class Application:
    def __init__(self) -> None:
        global _instance
        _instance = self
        self.prefs = Preferences()
        self.prefs_path = user_dir() / "preferences.json"
        self.prefs.load(self.prefs_path)
        self.keyconfig = default_keyconfig()
        self.history = History(int(self.prefs.memory.undo_steps), int(self.prefs.memory.undo_mb) << 20)
        self.tool_settings = ToolSettings()
        self.tool_settings.eraser.size = 60.0
        self.previous_mode = "OBJECT"             # 上一个模式（Tab 来回切换用）
        self._current_mode = self.tool_settings.mode
        self.project = Project()
        self.wm = None
        self.host = None
        self.window = None
        self.qt = None
        self.active_view3d = None
        self.render_job = None                 # 当前（或最近一次）出图
        self.render_window = None
        self._prefs_timer = None
        self._watched: list = []
        self.palette = self._load_palette()
        self._bake_queue: list = []            # 等着烘焙模型贴图的纹理集

    # ------------------------------------------------------------------ 便捷访问
    @property
    def engine(self):
        return self.host.engine if self.host is not None else None

    def request_frame(self) -> None:
        if self.host is not None:
            self.host.request_frame()

    def notify(self, tag: str = "") -> None:
        if self.wm is not None:
            self.wm.notify(tag)

    def report(self, text: str, level: str = "INFO") -> None:
        if self.wm is not None:
            self.wm.report(text, level)
        else:
            log.info(text)

    # ------------------------------------------------------------------ 启动
    def start(self, qt, *, with_engine: bool = True) -> None:
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QIcon

        from .ui import theme
        from .ui.wm import WindowManager

        self.qt = qt
        theme.apply(qt, float(self.prefs.interface.ui_scale))
        icon_path = resource("logo", "splender.ico")
        if icon_path.is_file():
            qt.setWindowIcon(QIcon(str(icon_path)))
        self._register_everything()
        self.wm = WindowManager(self)
        if with_engine:
            from .ui.glhost import EngineHost

            self.host = EngineHost(self)
            self.host.frame_done.connect(self._on_frame_done)
            self.host.engine.bake_finished.append(self._on_bake_finished)
            self.host.engine.selection_changed.append(self._on_selection_changed)
        from .ui.window import MainWindow

        self.window = MainWindow(self, self.wm)
        if hasattr(self.window, "workspaceChanged"):
            self.window.workspaceChanged.connect(self._on_workspace_changed)
        self.prefs.changed.connect(self._on_prefs_changed)
        self.history.changed.connect(self._on_history_changed)
        self.tool_settings.changed.connect(self._on_tool_changed)
        if hasattr(self.window, "active_workspace_index"):
            # 启动时停在哪个工作区，就用它的模式（比如「布局」是物体模式）
            self._on_workspace_changed(self.window.active_workspace_index())
        self._prefs_timer = QTimer()
        self._prefs_timer.setSingleShot(True)
        self._prefs_timer.setInterval(600)
        self._prefs_timer.timeout.connect(self.save_prefs)
        self._stats_timer = QTimer()
        self._stats_timer.setInterval(500)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start()
        # 选区边线流动：有选区时按设置的帧率重画视口
        self._ants_timer = QTimer()
        self._ants_timer.timeout.connect(self._ants_tick)
        from .core.gcpolicy import POLICY

        POLICY.install()
        from .core import addons

        addons.startup(self.prefs)

    # ---- 选区 ----
    def _on_selection_changed(self, ts) -> None:
        # 只重画视口：界面（菜单能不能点、面板）由操作做完时统一刷新，快速选择拖的时候不用每一下都整个界面重建
        self.request_frame()
        self._update_ants_timer()

    def _any_selection(self) -> bool:
        engine = self.engine
        return engine is not None and any(getattr(state, "selection", None) is not None and state.selection.active
                                          for state in engine.sets.values())

    def _update_ants_timer(self) -> None:
        timer = getattr(self, "_ants_timer", None)
        if timer is None:
            return
        viewport = self.prefs.viewport
        if self._any_selection() and viewport.selection_ants and viewport.selection_animate:
            timer.setInterval(max(16, int(1000 / max(1, int(viewport.selection_fps)))))
            if not timer.isActive():
                timer.start()
        elif timer.isActive():
            timer.stop()

    def _ants_tick(self) -> None:
        """虚线往前走一点：重画显示着有选区的贴图的视口（三维视口在出图模式时不动，免得打断渲染）。"""
        engine = self.engine
        if engine is None or not self._any_selection():
            self._update_ants_timer()
            return
        for view in engine.views:
            shading = getattr(view, "shading", None)
            if getattr(shading, "mode", "") == "RENDERED":
                continue
            view.dirty = True
        self.request_frame()

    def _idle_housekeeping(self) -> None:
        """用户停手、画面也静止时整理内存（全量垃圾回收会停顿一下，所以挑这个时候）。"""
        from .core.gcpolicy import POLICY

        if self.wm is None:
            return
        idle = time.perf_counter() - getattr(self.wm, "last_input", 0.0)
        if idle < float(self.prefs.memory.gc_idle_seconds):
            return
        engine = self.engine
        if engine is not None and engine.needs_render:
            return
        POLICY.collect_if_due()

    def _register_everything(self) -> None:
        """导入即注册：编辑器、操作、工具、菜单、面板。"""
        from . import editors  # noqa: F401
        from .ops import (addon_ops, bake_ops, edit_ops, file_ops, layer_ops, mesh_ops, node_ops,  # noqa: F401
                          object_ops, paint_ops, render_ops, screen_ops, sculpt_ops, view3d_ops, wm_ops)
        from .ops import adjust_ops, filter_ops, mesh_edit_ops, mesh_edit_tools, ps_ops, shader_ops  # noqa: F401
        from .ops import select_ops, texture_set_ops, tool_ops  # noqa: F401
        from .shading import nodes as shader_nodes

        shader_nodes.set_project_getter(lambda: self.project)
        from .sculpt import panels as sculpt_panels
        from .tools import mesh_tools, object_tools, sculpt_tools  # noqa: F401

        sculpt_panels.mark_paint_panels()
        from .render import builtin, panels  # noqa: F401  内置渲染引擎、渲染设置面板
        from .tools import paint_tools  # noqa: F401
        from .ui import menus, menus_edit_mesh, menus_shader, menus_view3d  # noqa: F401
        from .ui import redo

        redo.controller(self)

    # ------------------------------------------------------------------ 信号
    def _on_prefs_changed(self, name: str) -> None:
        if self._prefs_timer is not None:
            self._prefs_timer.start()
        if name == "interface.ui_scale" and self.qt is not None:
            from .ui import theme

            theme.apply(self.qt, float(self.prefs.interface.ui_scale))
        if name.startswith("memory.undo"):
            self.history.set_limits(int(self.prefs.memory.undo_steps), int(self.prefs.memory.undo_mb) << 20)
        if name.startswith("navigation") or name.startswith("viewport"):
            self.request_frame()
        if name.startswith("viewport.selection"):
            if self.engine is not None:
                for view in self.engine.views:
                    view.dirty = True
            self._update_ants_timer()

    def _on_history_changed(self) -> None:
        self.notify("history")
        self.request_frame()

    def _on_tool_changed(self, name: str) -> None:
        if name == "mode":
            mode = self.tool_settings.mode
            if mode != self._current_mode:
                self.previous_mode, self._current_mode = self._current_mode, mode
                from .ui import redo as redo_ui

                redo_ui.controller(self).clear()          # 换了模式，上一步的参数不再能调
            self._apply_mode()
        if name in ("tool", "mode"):
            self.notify("tool")
        view = self.active_view3d
        if view is not None and hasattr(view, "refresh_cursor"):
            view.refresh_cursor()

    def _on_render_changed(self, name: str) -> None:
        self.notify("render")
        if self.project is not None:
            self.project.mark_dirty()
        self.request_frame()

    def _apply_mode(self) -> None:
        """工具设置里的模式变了：进出雕刻模式（把模型交给雕刻引擎，或者把形状写回）。"""
        ts = self.tool_settings
        engine = self.engine
        if engine is None or self.project is None:
            return
        self.host.make_current()
        if ts.mode != "EDIT" and engine.edit is not None:
            obj = engine.edit.obj
            result = engine.exit_edit()
            if result and result.get("changed"):
                self._push_geometry_step(obj, result["before"], result["after"], label="编辑网格")
                self._rebake_after_shape_change(obj)
        if ts.mode == "EDIT":
            if engine.sculpt is not None:
                obj = engine.sculpt.obj
                result = engine.exit_sculpt()
                if result is not None:
                    self._auto_unwrap(obj, result)
                    self._push_geometry_step(obj, result["before"], result["after"], result.get("swaps"))
            active = self.project.active_object
            obj = active if (active is not None and active.kind == "MESH" and active.visible) else None
            if obj is None:
                with ts.changed.block():
                    ts.mode = self.previous_mode if self.previous_mode != "EDIT" else "OBJECT"
                self._current_mode = ts.mode
                self.report("先选一个模型：编辑模式编辑的是当前物体", "WARNING")
                self.notify("mode")
                return
            if engine.edit is None or engine.edit.obj is not obj:
                try:
                    engine.enter_edit(obj)
                except Exception as exc:  # noqa: BLE001
                    log.exception("进入编辑模式失败")
                    with ts.changed.block():
                        ts.mode = "OBJECT"
                    self._current_mode = ts.mode
                    self.report("进入编辑模式失败：%s" % exc, "WARNING")
                    self.notify("mode")
                    return
            if not ts.tool.startswith("mesh."):
                ts.tool = "mesh.select_box"
            engine.refresh_overlays()
            self.notify("mode")
            self.request_frame()
            return
        if ts.mode == "SCULPT":
            if engine.sculpt is None:
                # 和 Blender 一样雕刻当前物体；当前物体不是模型时用第一个看得见的模型
                active = self.project.active_object
                obj = active if (active is not None and active.kind == "MESH" and active.visible) else \
                    next((o for o in self.project.objects if o.visible), None)
                error = "工程里没有可以雕刻的模型" if obj is None else ""
                if obj is not None:
                    try:
                        engine.enter_sculpt(obj)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("进入雕刻模式失败")
                        error = "进入雕刻模式失败：%s" % exc
                if error:
                    with ts.changed.block():
                        ts.mode = self.previous_mode if self.previous_mode != "SCULPT" else "OBJECT"
                    self._current_mode = ts.mode
                    self.report(error, "WARNING")
                    self.notify("mode")
                    return
            if not ts.tool.startswith("sculpt."):
                ts.tool = "sculpt.draw"
        else:
            if engine.sculpt is not None:
                obj = engine.sculpt.obj
                result = engine.exit_sculpt()
                if result is not None:
                    self._auto_unwrap(obj, result)
                    self._push_geometry_step(obj, result["before"], result["after"], result.get("swaps"))
                    self._rebake_after_shape_change(obj)
            if ts.mode == "OBJECT":
                if not ts.tool.startswith("object."):
                    ts.tool = "object.select_box"
            elif ts.tool.startswith(("sculpt.", "object.", "mesh.")):
                ts.tool = "paint.brush"
        engine.refresh_overlays()
        self.notify("mode")
        self.request_frame()

    def _auto_unwrap(self, obj, result: dict) -> None:
        """重构后的模型没有 UV：按偏好设置自动展开，并把展开后的状态算进这一步「雕刻」。"""
        sculpt_prefs = getattr(self.prefs, "sculpt", None)
        if getattr(obj.data, "has_uvs", True) or sculpt_prefs is None or not sculpt_prefs.auto_unwrap:
            return
        from .ops.mesh_ops import unwrap_object
        from .sculpt.session import snapshot_mesh
        from .ui.busy import busy_cursor

        old = result.get("uv_source")
        before = result.get("before") or {}
        if old is None and before.get("has_uvs"):
            old = {"positions": before["positions"], "uvs": before["uvs"], "material_ids": before["material_ids"]}
        try:
            with busy_cursor():
                stats = unwrap_object(self, obj, old)
        except Exception as exc:  # noqa: BLE001
            log.exception("自动展开 UV 失败")
            self.report("自动展开 UV 没有成功：%s" % exc, "WARNING")
            return
        result["after"] = snapshot_mesh(obj.data)
        result["swaps"] = stats.get("swaps") or []
        moved = sum(len(pairs) for _ts, pairs in result["swaps"])
        self.report("已自动展开 UV：%d 块%s" % (stats["charts"], "，画好的 %d 个图层已跟着搬过去" % moved if moved else ""))

    def _push_geometry_step(self, obj, before: dict, after: dict, swaps: list | None = None,
                            label: str = "雕刻") -> None:
        """退出雕刻模式：整段雕刻合成一步撤销（绘制模式下也能撤销形状的改动）。
        swaps：换 UV 时搬运过的图层像素，跟着形状一起撤销、重做。"""
        from .sculpt.session import apply_snapshot

        state = {"applied": True}

        def restore(snap, use_new: bool) -> None:
            if self.engine is None:
                return
            self.host.make_current()
            apply_snapshot(obj.data, snap)
            obj.geometry_dirty = True
            self.engine.rebuild_object(obj)
            if swaps:
                self.engine.swap_layer_stores(swaps, use_new=use_new)
            state["applied"] = use_new
            self.request_frame()

        def dispose() -> None:
            if swaps and self.engine is not None:
                self.engine.free_layer_stores(swaps, keep_new=state["applied"])

        nbytes = sum(v.nbytes for v in before.values() if hasattr(v, "nbytes")) * 2
        self.history.push(label, lambda: restore(before, False), lambda: restore(after, True), nbytes, dispose)

    def _on_workspace_changed(self, index: int) -> None:
        from .ui.workspaces import WORKSPACE_MODES

        window = self.window
        ws = window.active_workspace() if window is not None and hasattr(window, "active_workspace") else None
        mode = WORKSPACE_MODES.get(getattr(ws, "name", ""))
        if mode is not None and mode != self.tool_settings.mode:
            self.tool_settings.mode = mode

    def _on_project_changed(self, kind: str) -> None:
        self.notify("project")
        if kind in ("objects", "sets", "loaded"):
            self.request_frame()

    def _on_layers_changed(self, kind: str, layer) -> None:
        self.notify("layers")
        self.request_frame()
        if kind == "material" and self.project is not None and self.engine is not None:
            # 材质节点用到了模型贴图（按表面算的纹理、遮蔽、曲率……）：还没烘焙过就去烘焙
            from .shading.compile import compile_graph

            for ts in self.project.texture_sets:
                if ts.material is None or self.engine.meshmap(ts.uid) is not None:
                    continue
                try:
                    compiled = compile_graph(ts.material)
                except Exception:  # noqa: BLE001
                    continue
                if compiled is not None and compiled.needs_maps:
                    self.ensure_meshmaps(ts)
        if kind in ("props", "structure") and layer is not None and getattr(layer, "mask_generator", "NONE") != "NONE":
            project = self.project
            ts = next((t for t in project.texture_sets if layer in t.layers), None) if project is not None else None
            if ts is not None:
                self.ensure_meshmaps(ts)

    def _on_frame_done(self) -> None:
        pass

    # ------------------------------------------------------------------ 模型贴图
    def ensure_meshmaps(self, ts) -> None:
        """这套贴图要用生成器了：还没有模型贴图、也没在烘焙，就开始烘焙。"""
        engine = self.engine
        if engine is None or ts is None or engine.meshmap(ts.uid) is not None:
            return
        if engine.bake is not None:
            if engine.bake.ts is not ts and ts not in self._bake_queue:
                self._bake_queue.append(ts)
            return
        self._start_bake(ts)

    def _start_bake(self, ts) -> None:
        try:
            self.host.make_current()
            self.engine.bake_start(ts)
            self.report("正在烘焙「%s」的模型贴图" % ts.name)
        except ValueError as exc:
            self.report(str(exc), "WARNING")
        self.notify("meshmaps")
        self.request_frame()

    def _rebake_after_shape_change(self, obj) -> None:
        """形状改过（雕刻、重构）：按设置把这个模型用到的、已经烘焙过的模型贴图重新烘焙。"""
        engine = self.engine
        if engine is None or self.project is None:
            return
        for set_uid in obj.material_sets:
            ts = self.project.texture_set(set_uid)
            if ts is None or engine.meshmap(set_uid) is None or not ts.meshmap.auto_rebake:
                continue
            if engine.bake is None:
                self._start_bake(ts)
            elif ts not in self._bake_queue:
                self._bake_queue.append(ts)

    def _on_bake_finished(self, job) -> None:
        if self.wm is not None:
            self.wm.set_progress("", None)
        if job.error:
            if job.cancelled:
                self.report("已停止烘焙")
            else:
                self.report("模型贴图没有烘焙成功：%s" % job.error, "WARNING")
        else:
            self.report("「%s」的模型贴图烘焙好了：%d²，用时 %.1f 秒" % (job.ts.name, job.settings["size"], job.elapsed))
            if self.project is not None:
                self.project.mark_dirty()
        self.notify("meshmaps")
        self.request_frame()
        while self._bake_queue and self.engine is not None and self.engine.bake is None:
            ts = self._bake_queue.pop(0)
            if self.project is not None and ts in self.project.texture_sets:
                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, lambda target=ts: self._start_bake(target))
                break

    def _update_stats(self) -> None:
        self._idle_housekeeping()
        engine = self.engine
        if engine is None or self.wm is None:
            return
        job = engine.bake
        if job is not None:
            self.wm.set_progress("烘焙模型贴图 · %s" % job.status, job.progress)
            self.notify("meshmaps")
        try:
            cache = engine.cache.stats()
            display = sum(s.display.committed_bytes for s in engine.sets.values()) / 1048576.0
            text = "显存 %d / %d MB · 内存 %d MB" % (cache["used_mb"] + int(display),
                                                  cache["budget_mb"] + (engine.display_budget >> 20), int(cache["ram_mb"]))
            self.wm.set_stats(text)
        except Exception:  # noqa: BLE001
            log.exception("更新状态栏统计时出错")

    # ------------------------------------------------------------------ 工程
    def _watch_project(self, project: Project) -> None:
        project.changed.connect(self._on_project_changed)
        project.graphs_changed.connect(self._on_graphs_changed)
        project.render.changed.connect(self._on_render_changed)
        for ts in project.texture_sets:
            ts.layers_changed.connect(self._on_layers_changed)

    def _on_graphs_changed(self, graph, kind: str) -> None:
        """节点图变了：视口重画；图的增删、换当前图、连线变化时刷新界面（拖参数时只重画，不重建界面）。"""
        self.request_frame()
        if kind in ("list", "active", "structure", "settings") or graph is None:
            self.notify("graphs")

    def set_project(self, project: Project, *, loaded: bool = False) -> None:
        self.history.clear()
        self.project = project
        self._watch_project(project)
        if self.host is not None and not loaded:
            self.host.make_current()
            self.engine.close_storage()
            self.engine.set_project(project)
        if self.window is not None and hasattr(self.window, "_hook_project"):
            try:
                self.window._hook_project()
            except Exception:  # noqa: BLE001
                log.exception("主窗口挂接工程时出错")
        self.notify("project")
        self.notify("layers")
        self.request_frame()

    def new_project(self, mesh=None, name: str = "未命名", resolution: str | None = None) -> Project:
        from .doc.meshio import make_test_mesh

        if mesh is None:
            mesh = make_test_mesh("sphere", segments=96, rings=48)
            mesh.name = "预览球"
        resolution = resolution or self.prefs.paint.default_resolution
        project = Project(name)
        set_uids = []
        materials = list(mesh.materials) or ["材质"]
        for index, material in enumerate(materials):
            ts = TextureSet(name=material or ("材质 %d" % (index + 1)), resolution=resolution)
            ts.add_layer(Layer(name="底色", kind="FILL", fill_color=(0.60, 0.60, 0.62), fill_roughness=0.5,
                               fill_metallic=0.0))
            ts.add_layer(Layer(name="图层 1"))
            project.add_texture_set(ts)
            set_uids.append(ts.uid)
        project.add_object(MeshObject(mesh.name or name, mesh, set_uids))
        project.dirty = False
        self.set_project(project)
        for warning in getattr(mesh, "warnings", []) or []:
            self.report(str(warning), "WARNING")
        return project

    def import_mesh(self, path: str, resolution: str | None = None) -> Project:
        from .doc.meshio import load_mesh

        started = time.perf_counter()
        mesh = load_mesh(path)
        project = self.new_project(mesh, name=os.path.splitext(os.path.basename(path))[0], resolution=resolution)
        project.mark_dirty(True)
        self.prefs.files.last_dir = os.path.dirname(path)
        self.report("已导入 %s：%s 个三角形，用时 %.1f 秒" % (os.path.basename(path), format(mesh.triangle_count, ","),
                                                   time.perf_counter() - started))
        return project

    def open_project(self, path: str) -> Project:
        self.history.clear()
        self.host.make_current()
        project, meta = self.engine.load_project(path)
        self.set_project(project, loaded=True)
        self.prefs.add_recent(path)
        self.prefs.files.last_dir = os.path.dirname(path)
        ui = meta.get("ui") if isinstance(meta, dict) else None
        if ui and self.prefs.interface.load_ui and self.window is not None:
            self._apply_ui_state(ui)
        self.report("已打开 %s" % os.path.basename(path))
        return project

    def save_project(self, path: str | None = None) -> bool:
        path = path or self.project.path
        if not path:
            return False
        self.host.make_current()
        extra = {"ui": self._ui_state()} if self.prefs.files.save_layout_in_project else {}
        info = self.engine.save_project(path, extra)
        self.prefs.add_recent(path)
        self.prefs.files.last_dir = os.path.dirname(path)
        self.report("已保存 %s（%d 页，%.1f 秒）" % (os.path.basename(path), info["pages"], info["seconds"]))
        self.notify("project")
        return True

    def export_textures(self, folder: str, channels: list[str] | None = None, size: int | None = None) -> list[str]:
        """导出每套贴图：通道、尺寸、文件名按各纹理集的导出设置；channels、size 给了就用给的。"""
        from .doc.project import EXPORT_IDS
        from .engine.projectio import export_file_name

        self.host.make_current()
        written = []
        used: set[str] = set()
        for ts in self.project.texture_sets:
            settings = ts.export
            chosen = channels or [c for c in EXPORT_IDS if getattr(settings, c)]
            set_size = size or int(settings.size) or None
            for channel in chosen:
                name = export_file_name(settings.name_pattern, self.project.name, ts.name, channel)
                if name.lower() in used:                  # 命名规则里没有区分的部分：补上纹理集和通道
                    name = export_file_name("{工程}_{纹理集}_{通道}", self.project.name, ts.name, channel)
                used.add(name.lower())
                target = os.path.join(folder, name)
                self.wm.set_progress("正在导出 %s" % name, 0.0)
                self.qt.processEvents()
                self.engine.export_channel(ts, channel, target, size=set_size)
                written.append(target)
        self.wm.set_progress("", None)
        self.report("已导出 %d 张贴图到 %s" % (len(written), folder))
        return written

    def _ui_state(self) -> dict:
        try:
            return {"workspaces": self.window.store_workspaces()}
        except Exception:  # noqa: BLE001
            return {}

    def _apply_ui_state(self, ui: dict) -> None:
        text = ui.get("workspaces")
        if text:
            self.prefs.workspaces = text

    # ------------------------------------------------------------------ 偏好与色板
    def save_prefs(self) -> None:
        try:
            if self.window is not None:
                self.window.store_workspaces()
        except Exception:  # noqa: BLE001
            pass
        self.prefs.save(self.prefs_path)

    def _load_palette(self) -> list[tuple]:
        path = user_dir() / "palette.json"
        try:
            return [tuple(c) for c in json.loads(path.read_text(encoding="utf-8"))]
        except Exception:  # noqa: BLE001
            return [(0.80, 0.30, 0.20), (0.92, 0.72, 0.18), (0.25, 0.60, 0.35), (0.18, 0.45, 0.80), (0.55, 0.30, 0.75),
                    (0.95, 0.95, 0.95), (0.50, 0.50, 0.50), (0.08, 0.08, 0.08), (0.75, 0.55, 0.40), (0.35, 0.22, 0.15)]

    def save_palette(self) -> None:
        try:
            (user_dir() / "palette.json").write_text(json.dumps([list(c) for c in self.palette]), encoding="utf-8")
        except Exception:  # noqa: BLE001
            log.exception("色板保存失败")

    # ------------------------------------------------------------------ 退出
    def shutdown(self) -> None:
        from .core import addons
        from .render.session import cancel_render

        try:
            cancel_render(self)
        except Exception:  # noqa: BLE001
            log.exception("停止渲染时出错")
        try:
            addons.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("停用插件时出错")
        self.save_prefs()
        if self.wm is not None:
            self.wm.shutdown()
        if self.host is not None:
            self.host.make_current()
            self.engine.close_storage()
            self.host.shutdown()


def _set_app_id() -> None:
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Splender.App")
    except Exception:  # noqa: BLE001
        pass


def create_qt(argv: list[str], vsync: bool = True):
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtWidgets import QApplication

    from .ui.glhost import configure_surface_format

    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    configure_surface_format(vsync)
    qt = QApplication.instance() or QApplication([APP_NAME] + list(argv))
    qt.setApplicationName(APP_NAME)
    qt.setApplicationDisplayName(APP_NAME)
    qt.setApplicationVersion(__version__)
    return qt


def run(argv: list[str]) -> int:
    applog.setup()
    ensure_vendor_path()
    _set_app_id()
    selftest = None
    if "--selftest" in argv:
        index = argv.index("--selftest")
        selftest = argv[index + 1] if index + 1 < len(argv) else ""
        del argv[index:index + 2]
    project_path = next((a for a in argv if not a.startswith("-") and a.lower().endswith(".splender")), None)
    app = Application()
    qt = create_qt(argv, vsync=bool(app.prefs.viewport.vsync))
    splash = None
    if selftest is None and app.prefs.interface.show_splash:
        from .ui.splash import show_splash

        splash = show_splash(qt)
    try:
        app.start(qt)
    except Exception as error:  # noqa: BLE001
        log.exception("启动失败")
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.critical(None, APP_NAME, "启动失败：%s\n\n详细信息在日志里：%s" % (error, user_dir() / "logs"))
        return 1
    window = app.window
    window.resize(1600, 1000)
    if selftest is not None:
        from .selftest import prepare_window

        prepare_window(window)
    else:
        window.showMaximized()
    qt.processEvents()
    try:
        if project_path and os.path.isfile(project_path):
            app.open_project(project_path)
        else:
            app.new_project()
    except Exception as error:  # noqa: BLE001
        log.exception("载入工程失败")
        app.new_project()
        app.report("没有打开：%s" % error, "ERROR")
    if splash is not None:
        splash.finish(window)
    if selftest is not None:
        from .selftest import run_script

        run_script(app, selftest)
    code = qt.exec()
    app.shutdown()
    return int(code)
