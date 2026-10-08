"""诊断信息：版本、系统、显卡、预算、工程概况、自动保存状态、最近的日志。反馈问题时整段复制给开发者。"""
from __future__ import annotations

import os
import platform
import sys
import time


def _safe(func, default="（取不到）"):
    try:
        value = func()
    except Exception:  # noqa: BLE001
        return default
    return default if value in (None, "") else value


def collect(app=None, log_lines: int = 80) -> str:
    from .. import __version__
    from .. import log as applog
    from ..paths import ROOT_DIR, user_dir

    lines = ["SPLENDER %s（%s）" % (__version__, "便携包" if (ROOT_DIR / "runtime").is_dir() else "源码"),
             "时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "系统：%s" % _safe(platform.platform),
             "Python：%s" % sys.version.split()[0]]
    try:
        import PySide6
        from PySide6.QtCore import qVersion

        lines.append("Qt：%s（PySide6 %s）" % (qVersion(), PySide6.__version__))
    except Exception:  # noqa: BLE001
        pass
    lines.append("处理器：%s 个逻辑核" % (os.cpu_count() or "?"))
    try:
        import psutil

        memory = psutil.virtual_memory()
        lines.append("内存：共 %.1f GB，可用 %.1f GB" % (memory.total / 2 ** 30, memory.available / 2 ** 30))
    except Exception:  # noqa: BLE001
        pass
    engine = getattr(app, "engine", None) if app is not None else None
    if engine is not None:
        info = _safe(lambda: dict(engine.ctx.info), {})
        lines.append("显卡：%s" % info.get("GL_RENDERER", "?"))
        lines.append("驱动：%s，%s" % (info.get("GL_VENDOR", "?"), info.get("GL_VERSION", "?")))
        stats = _safe(engine.stats, {})
        if stats:
            lines.append("显存：共 %s MB，空闲 %s MB；图层显存池 %s，内存缓存 %.0f / %s MB" % (
                stats.get("vram_total_mb", "?"), stats.get("vram_free_mb", "?"),
                _pool_text(stats), float(stats.get("ram_mb", 0.0)), stats.get("ram_budget_mb", "?")))
            lines.append("图层页：%s，显示缓存 %.0f / %s MB" % (stats.get("layer_pages", "?"), float(stats.get("display_mb", 0.0)),
                                                       stats.get("display_budget_mb", "?")))
    if app is not None and getattr(app, "prefs", None) is not None:
        tablet = getattr(app, "wintab", None)
        state = "WinTab 开着（收到 %d 个包）" % tablet.packets if tablet is not None and tablet.ok else "WinTab 没开"
        lines.append("数位板：接口 %s，%s" % (getattr(app.prefs.paint, "tablet_api", "?"), state))
    project = getattr(app, "project", None) if app is not None else None
    if project is not None:
        lines.append("工程：%s%s（%s）" % (project.name, " *" if project.dirty else "", project.path or "没存过"))
        for ts in project.texture_sets:
            lines.append("  纹理集「%s」：%s，%d 层" % (ts.name, ts.resolution, len(ts.layers)))
        for obj in project.objects:
            count = _safe(lambda obj=obj: obj.data.triangle_count, "?")
            lines.append("  模型「%s」：%s 个三角形" % (obj.name, count))
    prefs = getattr(app, "prefs", None) if app is not None else None
    if prefs is not None:
        files = prefs.files
        state = "开，每 %.2g 分钟" % float(files.autosave_minutes) if files.autosave else "关"
        saver = getattr(engine, "__dict__", {}).get("_autosave") if engine is not None else None
        if saver is not None and saver.last_result:
            state += "；上次写了 %d 页，用时 %.1f 秒" % (saver.last_result.get("pages", 0),
                                                 saver.last_result.get("seconds", 0.0))
        if saver is not None and saver.last_error:
            state += "；上次出错：%s" % saver.last_error
        lines.append("自动保存：%s" % state)
    lines.append("设置目录：%s" % _safe(user_dir))
    records = list(applog.records)[-max(0, int(log_lines)):]
    if records:
        lines.append("")
        lines.append("最近的日志：")
        for when, level, text in records:
            lines.append("%s %s %s" % (when, level, text))
    return "\n".join(lines)


def _pool_text(stats: dict) -> str:
    """图层显存池：在用 / 已分配 / 预算。"""
    if "budget_mb" not in stats:
        return "?"
    return "在用 %s / 已分配 %s / 预算 %s MB" % (stats.get("used_mb", "?"), stats.get("allocated_mb", "?"),
                                          stats.get("budget_mb", "?"))


def copy_to_clipboard(app=None) -> str:
    """收集诊断信息并放进剪贴板，返回那段文字。"""
    text = collect(app)
    try:
        from PySide6.QtGui import QGuiApplication

        QGuiApplication.clipboard().setText(text)
    except Exception:  # noqa: BLE001
        pass
    return text
