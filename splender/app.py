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
        self._apply_keymap_overrides()
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
        self.selftest = False                  # 自检模式：不弹恢复、出错提示这类要人回答的窗口
        self.session = None                    # 会话锁（core.recovery.Session）
        self.last_restored: str | None = None  # 最近一次从自动保存恢复出来的工程文件
        self._autosave_due = 0.0               # 下一次自动保存的时间（time.monotonic）
        self._autosave_wait_start: float | None = None
        self._autosave_failed = False
        self._error_messages: set[str] = set()
        self.wintab = None                     # WinTab 数位板接口（ui.wintab.WinTab），偏好里选了才开

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
        self._apply_keymap_overrides()                 # 插件加的键位表也用上改过的键位
        self._start_session()
        applog.on_unhandled = self._on_unhandled_error
        if self.host is not None:
            self.engine.autosave.on_finished.append(self._on_autosave_finished)
        self._autosave_timer = QTimer()
        self._autosave_timer.setInterval(1000)
        self._autosave_timer.timeout.connect(self._autosave_tick)
        self._autosave_timer.start()
        self._schedule_autosave()
        self._update_tablet()
        try:
            qt.applicationStateChanged.connect(self._on_app_state)
        except Exception:  # noqa: BLE001
            pass

    # ---- 数位板 ----
    def _update_tablet(self) -> None:
        """按偏好设置开、关 WinTab（改了立即生效）。"""
        mode = getattr(self.prefs.paint, "tablet_api", "AUTO")
        # 自检时不开：新开的 WinTab 上下文会排到最前面，可能抢走正在别的软件里画画的人的笔
        if mode == "WININK" or sys.platform != "win32" or self.window is None or self.selftest:
            if self.wintab is not None:
                self.wintab.close()
                self.wintab = None
            return
        if self.wintab is not None and self.wintab.ok:
            return
        from .ui.wintab import WinTab

        tablet = WinTab()
        if tablet.open(int(self.window.winId())):
            self.wintab = tablet
        else:
            self.wintab = None
            log.info("WinTab 没开：%s", tablet.reason)
            if mode == "WINTAB":
                self.report("WinTab 没开：%s" % tablet.reason, "WARNING")

    def tablet_mode(self) -> str:
        return str(getattr(self.prefs.paint, "tablet_api", "AUTO"))

    def _on_app_state(self, state) -> None:
        from PySide6.QtCore import Qt

        if state == Qt.ApplicationActive and self.wintab is not None:
            self.wintab.to_front()

    # ---- 改键 ----
    def _apply_keymap_overrides(self) -> None:
        from .core.keymap import apply_overrides

        try:
            apply_overrides(self.keyconfig, self.prefs.keymap_overrides)
        except Exception:  # noqa: BLE001
            log.exception("用上改过的键位时出错，按出厂键位")

    def save_keymap_overrides(self) -> None:
        """键位改过了：把和出厂不一样的那些存进偏好设置（随偏好设置一起写盘），菜单上的快捷键文字跟着变。"""
        from .core.keymap import overrides_from

        self.prefs.keymap_overrides = json.dumps(overrides_from(self.keyconfig), ensure_ascii=False)
        self.notify("keymap")

    # ---- 会话锁、自动保存、崩溃恢复 ----
    def _start_session(self) -> None:
        from .core.recovery import Session
        from .paths import autosave_dir, log_dir

        try:
            self.session = Session(autosave_dir(self.prefs.files.autosave_dir), log_dir())
            self.session.create()
            if self.session.crash_path is not None:
                applog.enable_crash_file(self.session.crash_path)
        except Exception:  # noqa: BLE001
            log.exception("建立会话锁失败，自动保存放进临时目录")
            self.session = None

    def _schedule_autosave(self) -> None:
        self._autosave_due = time.monotonic() + max(0.25, float(self.prefs.files.autosave_minutes)) * 60.0
        self._autosave_wait_start = None

    def _autosave_busy(self) -> bool:
        engine = self.engine
        if engine is None:
            return True
        if engine.stroke is not None or engine.merges:
            return True
        sculpt = engine.sculpt
        if sculpt is not None and sculpt.stroke is not None:
            return True
        return self.wm is not None and self.wm.has_modal()

    def _autosave_tick(self) -> None:
        """每秒看一眼：到时间了、你停手了（或推迟太久了）、没有笔划在进行，就开始一次自动保存。"""
        files = self.prefs.files
        engine = self.engine
        if engine is None or not files.autosave or engine.autosave.active:
            return
        now = time.monotonic()
        if now < self._autosave_due:
            return
        if self._autosave_wait_start is None:
            self._autosave_wait_start = now
        if self._autosave_busy():
            return
        idle = time.perf_counter() - getattr(self.wm, "last_input", 0.0)
        if idle < float(files.autosave_idle) and now - self._autosave_wait_start < float(files.autosave_max_wait):
            return
        self._schedule_autosave()
        self.autosave_now()

    def autosave_now(self, wait: bool = False) -> bool:
        """马上开始一次自动保存（没有要存的就不开始）。wait 为真时一直推进到做完。返回是否开始了。"""
        engine = self.engine
        if engine is None:
            return False
        from . import __version__

        files = self.prefs.files
        saver = engine.autosave
        if self.session is not None:
            saver.new_path = self.session.new_recovery_path
        saver.level = 3
        saver.threads = int(files.autosave_threads)
        extra = {"ui": self._ui_state()} if files.save_layout_in_project else {}
        info = {"session": self.session.id if self.session is not None else "", "app": __version__}
        self.host.make_current()
        try:
            started = saver.start(self.project, extra=extra, info=info)
        except Exception as error:  # noqa: BLE001
            log.exception("自动保存没能开始")
            self._on_autosave_finished({"error": str(error)})
            return False
        if started:
            self.request_frame()
            if wait:
                saver.wait()
        return started

    def _on_autosave_finished(self, result: dict) -> None:
        error = result.get("error")
        if error:
            if not self._autosave_failed:
                self.report("自动保存失败：%s（到时间会再试）" % error, "WARNING")
            self._autosave_failed = True
            return
        if self._autosave_failed:
            self.report("自动保存恢复正常")
        self._autosave_failed = False

    def autosave_scan(self) -> tuple[list, list]:
        """找可以恢复的自动保存：(列表, 刚发现的没正常退出的会话)。"""
        from .core.recovery import scan
        from .paths import autosave_dir

        files = self.prefs.files
        return scan(autosave_dir(files.autosave_dir), self.session, int(files.autosave_keep))

    def restore_autosave(self, path: str) -> str:
        """把一份自动保存还原成工程文件并打开，返回新工程文件的路径。"""
        from .core.recovery import read_recoverable, remove_quiet, restore_target
        from .engine.autosave import restore

        item = read_recoverable(path)
        if item is None:
            raise RuntimeError("这份自动保存没有写完，不能恢复")
        target = restore_target(item, self.prefs.files.restore_dir or None)
        info = restore(path, target)
        self.open_project(target)
        self.last_restored = target
        remove_quiet(path)
        for warning in info["warnings"]:
            self.report(warning, "WARNING")
        self.report("已恢复到 %s" % target)
        return target

    def check_recovery_on_start(self) -> None:
        """启动后：上次没正常退出、留下了自动保存，就问要不要恢复；只有崩溃报告时在状态栏提一句。"""
        try:
            items, ended = self.autosave_scan()
        except Exception:  # noqa: BLE001
            log.exception("查找自动保存时出错")
            return
        crashes = [item["crash"] for item in ended if item.get("crash")]
        if crashes:
            log.warning("上次程序崩溃了，崩溃报告：%s", "、".join(crashes))
        if items and not self.selftest:
            from .ui.recovery_dialog import RecoveryDialog

            RecoveryDialog(self, items, startup=True).exec()
        elif crashes:
            self.report("上次程序意外退出了，崩溃报告在日志文件夹里（帮助 → 打开日志文件夹）", "WARNING")
        elif ended:
            self.report("上次程序没有正常退出", "WARNING")

    def _on_unhandled_error(self, kind, value, tb) -> None:
        text = str(value) or kind.__name__
        if self.wm is not None:
            self.wm.report("出错了：%s（详细信息在日志里）" % text, "ERROR")
        if self.selftest or not self.prefs.interface.error_dialog:
            return
        window = self.window
        if window is None or not window.isVisible() or text in self._error_messages or len(self._error_messages) >= 3:
            return
        self._error_messages.add(text)
        from .ui.recovery_dialog import show_error

        show_error(self, kind, value, tb)

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
        if name == "paint.tablet_api":
            self._update_tablet()
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
        affected = list(obj.material_sets)
        # 拿这个模型当高模的纹理集也要重烘
        affected += [ts.uid for ts in self.project.texture_sets if str(ts.meshmap.high_poly_object) == str(obj.uid)]
        for set_uid in dict.fromkeys(affected):
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

    def import_into_scene(self, path: str):
        """把模型文件里的模型加进当前场景：每个材质一套新贴图，记一步撤销。返回新模型。"""
        from .doc.meshio import load_mesh
        from .ops.object_ops import add_objects_with_undo, new_texture_set, unique_name

        started = time.perf_counter()
        mesh = load_mesh(path)
        project = self.project
        name = unique_name(project, os.path.splitext(os.path.basename(path))[0] or "模型")
        mesh.name = name
        taken = {ts.name for ts in project.texture_sets}
        sets = []
        for index, material in enumerate(list(mesh.materials) or ["材质"]):
            label = material or "%s 材质 %d" % (name, index + 1)
            unique = label
            number = 2
            while unique in taken:
                unique = "%s %d" % (label, number)
                number += 1
            taken.add(unique)
            sets.append(new_texture_set(self, unique))
        obj = MeshObject(name, mesh, [ts.uid for ts in sets])
        obj.geometry_dirty = True
        self.host.make_current()
        add_objects_with_undo(self.wm.context(use_mouse=False), [obj], sets, "导入 %s" % os.path.basename(path))
        self.prefs.files.last_dir = os.path.dirname(path)
        for warning in getattr(mesh, "warnings", []) or []:
            self.report(str(warning), "WARNING")
        if not getattr(mesh, "has_uvs", True):
            self.report("「%s」没有 UV，画之前先展开（UV → 智能投射）" % name, "WARNING")
        self.report("已导入到场景：%s，%s 个三角形，用时 %.1f 秒" % (name, format(mesh.triangle_count, ","),
                                                       time.perf_counter() - started))
        return obj

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
        files: dict[int, dict] = {}
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
                files.setdefault(ts.uid, {})[channel] = name
        self.wm.set_progress("", None)
        self.report("已导出 %d 张贴图到 %s" % (len(written), folder))
        # 勾了「连模型一起导出」的纹理集：用到它们的模型按各自选的格式导出到同一个文件夹
        by_format: dict[str, set] = {}
        for ts in self.project.texture_sets:
            if ts.export.with_mesh:
                by_format.setdefault(ts.export.mesh_format, set()).add(ts.uid)
        for fmt, uids in by_format.items():
            objects = [obj for obj in self.project.objects if obj.visible and uids & set(obj.material_sets)]
            if not objects:
                continue
            target = os.path.join(folder, export_file_name("{工程}", self.project.name, "", "")[:-4]
                                  + (".glb" if fmt == "GLB" else ".obj"))
            info = self.export_meshes(target, objects, with_textures=True, size=size, texture_files=files)
            written.append(info["path"])
        return written

    def _export_mesh_data(self, obj):
        """导出用的模型数据：雕刻中的先写回物体，编辑模式里的用完整网格（含隐藏的面）。"""
        engine = self.engine
        sculpt = getattr(engine, "sculpt", None) if engine is not None else None
        if sculpt is not None and sculpt.obj is obj:
            sculpt.sync_to_object()
        edit = getattr(engine, "edit", None) if engine is not None else None
        if edit is not None and edit.obj is obj and edit.changed:
            return edit.full_data()
        return obj.data

    def export_meshes(self, path: str, objects: list | None = None, *, with_textures: bool = True,
                      size: int | None = None, scale: float = 1.0, texture_files: dict | None = None) -> dict:
        """把模型导出成 .glb 或 .obj（按扩展名）。objects 默认是全部可见的模型。

        with_textures：glb 把基础色、法线、遮蔽粗糙度金属度打包贴图装进文件；obj 在旁边写 PNG，材质文件引用它们。
        texture_files：{纹理集 uid: {通道: 文件名}}，obj 直接引用这些已经导出的贴图（不再另外导出）。"""
        import shutil
        import tempfile

        from .doc.meshexport import export_glb, export_obj
        from .engine.projectio import export_file_name

        fmt = os.path.splitext(path)[1].lower()
        if fmt not in (".glb", ".obj"):
            raise ValueError("只支持 .glb 和 .obj：%s" % path)
        project = self.project
        if objects is None:
            objects = [obj for obj in project.objects if obj.visible]
        if not objects:
            raise ValueError("没有可以导出的模型")
        self.host.make_current()
        engine = self.engine
        engine.complete_strokes(include_active=True)
        sets = {ts.uid: ts for ts in project.texture_sets}
        labels: dict[int, str] = {}
        used: set[str] = set()
        for obj in objects:
            for uid in obj.material_sets:
                if uid in labels or uid not in sets:
                    continue
                label = sets[uid].name or "材质"
                number = 2
                while label.lower() in used:
                    label = "%s %d" % (sets[uid].name, number)
                    number += 1
                used.add(label.lower())
                labels[uid] = label
        items = []
        for obj in objects:
            data = self._export_mesh_data(obj)
            names = [labels.get(uid, "材质") for uid in obj.material_sets] or ["材质"]
            items.append((obj.name, data, names))
        folder = os.path.dirname(os.path.abspath(path))
        info: dict = {}
        if fmt == ".glb":
            materials = {}
            temp = tempfile.mkdtemp(prefix="splender_glb_") if with_textures else None
            try:
                for uid, label in labels.items():
                    ts = sets[uid]
                    entry = {"color": tuple(ts.base_color), "metallic": float(ts.base_metallic),
                             "roughness": float(ts.base_roughness)}
                    if temp is not None and uid in engine.sets:
                        set_size = size or int(ts.export.size) or None
                        for channel, source in (("basecolor", "basecolor"), ("orm", "orm"), ("normal", "normal_gl")):
                            self.wm.set_progress("正在准备「%s」的%s贴图" % (label, {"basecolor": "基础色", "orm": "遮蔽粗糙度金属度",
                                                                              "normal": "法线"}[channel]), 0.0)
                            self.qt.processEvents()
                            target = os.path.join(temp, "%d_%s.png" % (uid, channel))
                            engine.export_channel(ts, source, target, size=set_size)
                            with open(target, "rb") as handle:
                                entry[channel] = handle.read()
                    materials[label] = entry
                self.wm.set_progress("正在写 %s" % os.path.basename(path), 0.9)
                self.qt.processEvents()
                info = export_glb(path, items, materials=materials, scale=scale)
            finally:
                if temp is not None:
                    shutil.rmtree(temp, ignore_errors=True)
        else:
            textures = {}
            if with_textures:
                for uid, label in labels.items():
                    ts = sets[uid]
                    given = (texture_files or {}).get(uid)
                    if given is not None:
                        textures[label] = dict(given)
                        continue
                    if uid not in engine.sets:
                        continue
                    files = {}
                    set_size = size or int(ts.export.size) or None
                    for channel in ("basecolor", "metallic", "roughness", "normal"):
                        name = export_file_name(ts.export.name_pattern, project.name, label, channel)
                        self.wm.set_progress("正在导出 %s" % name, 0.0)
                        self.qt.processEvents()
                        engine.export_channel(ts, channel, os.path.join(folder, name), size=set_size)
                        files[channel] = name
                    textures[label] = files
            self.wm.set_progress("正在写 %s" % os.path.basename(path), 0.9)
            self.qt.processEvents()
            info = export_obj(path, items, textures=textures, scale=scale)
        self.wm.set_progress("", None)
        self.prefs.files.export_dir = folder
        self.report("已导出模型 %s：%s 个三角形，用时 %.1f 秒" % (os.path.basename(path), format(info.get("triangles", 0), ","),
                                                     float(info.get("seconds", 0.0))))
        return info

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
        timer = getattr(self, "_autosave_timer", None)
        if timer is not None:
            timer.stop()
        if self.wm is not None:
            self.wm.shutdown()
        if self.host is not None:
            self.host.make_current()
            try:
                self.engine.autosave.shutdown(delete=True)     # 正常退出：恢复文件用不着了
            except Exception:  # noqa: BLE001
                log.exception("关闭自动保存时出错")
            self.engine.close_storage()
            self.host.shutdown()
        if self.wintab is not None:
            self.wintab.close()
            self.wintab = None
        applog.on_unhandled = None
        if self.session is not None:
            self.session.close()
        applog.disable_crash_file(delete_if_empty=True)


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
    _install_qt_chinese(qt)
    return qt


def _install_qt_chinese(qt) -> None:
    """Qt 自带的窗口（消息框的「显示详细信息」、文件对话框等）用中文。"""
    from PySide6.QtCore import QLibraryInfo, QTranslator

    if getattr(qt, "_splender_translator", None) is not None:
        return
    folders = [QLibraryInfo.path(QLibraryInfo.TranslationsPath)]
    try:
        import PySide6

        folders.append(os.path.join(os.path.dirname(PySide6.__file__), "translations"))
    except Exception:  # noqa: BLE001
        pass
    translator = QTranslator(qt)
    for folder in folders:
        if folder and translator.load("qtbase_zh_CN", folder):
            qt.installTranslator(translator)
            qt._splender_translator = translator
            return
    log.info("没找到 Qt 自带窗口的中文翻译，按钮等会显示英文")


def _warm_up_numba() -> None:
    """后台先把显示模型要用的 numba 函数编译好（有缓存时只是读进来），和建界面、建显卡上下文同时进行。
    第一次启动（还没有编译缓存）能少等几秒；主线程用到时如果还在编，会等编译锁，不会重复编。"""
    import threading

    def work() -> None:
        try:
            from .doc.meshio import make_test_mesh
            from .engine.meshprep import build_set_geometry

            build_set_geometry(make_test_mesh("sphere", segments=8, rings=4), 0, 256, skirt_texels=2.0)
        except Exception:  # noqa: BLE001
            log.debug("预先编译没做完", exc_info=True)

    threading.Thread(target=work, name="splender-warmup", daemon=True).start()


def run(argv: list[str]) -> int:
    applog.setup()
    ensure_vendor_path()
    _warm_up_numba()
    _set_app_id()
    selftest = None
    if "--selftest" in argv:
        index = argv.index("--selftest")
        selftest = argv[index + 1] if index + 1 < len(argv) else ""
        del argv[index:index + 2]
    project_path = next((a for a in argv if not a.startswith("-") and a.lower().endswith(".splender")), None)
    app = Application()
    app.selftest = selftest is not None
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
    if selftest is None:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(400, app.check_recovery_on_start)
    if selftest is not None:
        from .selftest import run_script

        run_script(app, selftest)
    code = qt.exec()
    app.shutdown()
    return int(code)
