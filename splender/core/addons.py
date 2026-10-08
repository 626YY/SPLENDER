"""插件：和 Blender 一样，功能可以做成插件，启用时注册、停用时注销。

一个插件是一个 Python 包（带 __init__.py 的文件夹）或单个 .py 文件，里面写：

    ADDON_INFO = {
        "name": "显示名", "description": "一句话说明", "author": "作者", "version": (1, 0, 0),
        "category": "渲染", "default_enabled": False,
    }
    def register(): ...      # 注册操作、面板、菜单条目、渲染引擎等
    def unregister(): ...    # 注销 register() 里注册过的全部内容

可选：
    PREFERENCES = PropertyGroup 子类   # 插件自己的设置，显示在偏好设置的插件页，跟着偏好一起保存

内置插件在 splender/addons/ 下，用户插件在用户目录的 addons/ 里。
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import logging
import shutil
import sys
import types
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..paths import user_dir
from .signals import Signal

log = logging.getLogger("splender.addons")

BUILTIN_PACKAGE = "splender.addons"
USER_PACKAGE = "splender_user_addons"
BUILTIN_DIR = Path(__file__).resolve().parent.parent / "addons"

#: 启用状态或插件列表变化时发出
changed = Signal()


@dataclass
class Addon:
    module: str                 # 导入名
    path: Path                  # 包目录或 .py 文件
    builtin: bool
    info: dict = field(default_factory=dict)
    enabled: bool = False
    error: str = ""
    instance: Any = None        # 导入后的模块
    preferences: Any = None     # PREFERENCES 的实例

    @property
    def name(self) -> str:
        return str(self.info.get("name") or self.module.rsplit(".", 1)[-1])

    @property
    def category(self) -> str:
        return str(self.info.get("category") or "其他")

    @property
    def version_text(self) -> str:
        version = self.info.get("version")
        if isinstance(version, (tuple, list)):
            return ".".join(str(v) for v in version)
        return str(version or "")


_ADDONS: dict[str, Addon] = {}
_state = {"prefs": None}


def user_addon_dir() -> Path:
    folder = user_dir() / "addons"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ------------------------------------------------------------------ 发现
def _source_of(path: Path) -> Path:
    return path / "__init__.py" if path.is_dir() else path


def read_info(path: Path) -> dict:
    """不导入模块，直接从源码里读出 ADDON_INFO 字面量（列表里显示用，坏插件不影响启动）。"""
    try:
        tree = ast.parse(_source_of(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ADDON_INFO" for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
                return value if isinstance(value, dict) else {}
            except Exception:  # noqa: BLE001
                return {}
    return {}


def _candidates(folder: Path):
    if not folder.is_dir():
        return
    for item in sorted(folder.iterdir()):
        if item.name.startswith(("_", ".")):
            continue
        if item.is_dir() and (item / "__init__.py").is_file():
            yield item.name, item
        elif item.is_file() and item.suffix == ".py":
            yield item.stem, item


def discover() -> list[Addon]:
    """重新扫描内置和用户插件。已启用的保持原状。"""
    found: dict[str, Addon] = {}
    for stem, path in _candidates(BUILTIN_DIR):
        module = "%s.%s" % (BUILTIN_PACKAGE, stem)
        found[module] = _ADDONS.get(module) or Addon(module, path, True)
        found[module].info = read_info(path)
    for stem, path in _candidates(user_addon_dir()):
        module = "%s.%s" % (USER_PACKAGE, stem)
        found[module] = _ADDONS.get(module) or Addon(module, path, False)
        found[module].info = read_info(path)
    for module, addon in list(_ADDONS.items()):
        if module not in found and addon.enabled:
            found[module] = addon             # 文件被删了但还在运行：留着，停用后消失
    _ADDONS.clear()
    _ADDONS.update(found)
    changed.emit()
    return addons()


def addons() -> list[Addon]:
    return sorted(_ADDONS.values(), key=lambda a: (a.category, a.name))


def get(module: str) -> Addon | None:
    return _ADDONS.get(module)


# ------------------------------------------------------------------ 导入
def _import(addon: Addon):
    if addon.builtin:
        return importlib.import_module(addon.module)
    if USER_PACKAGE not in sys.modules:
        package = types.ModuleType(USER_PACKAGE)
        package.__path__ = [str(user_addon_dir())]
        sys.modules[USER_PACKAGE] = package
    if addon.module in sys.modules:
        return sys.modules[addon.module]
    source = _source_of(addon.path)
    search = [str(addon.path)] if addon.path.is_dir() else None
    spec = importlib.util.spec_from_file_location(addon.module, source, submodule_search_locations=search)
    module = importlib.util.module_from_spec(spec)
    sys.modules[addon.module] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(addon.module, None)
        raise
    return module


# ------------------------------------------------------------------ 状态存取（偏好里的一段 JSON）
def _load_state() -> dict:
    prefs = _state["prefs"]
    try:
        data = json.loads(getattr(prefs, "addons", "") or "{}") if prefs is not None else {}
    except Exception:  # noqa: BLE001
        data = {}
    data.setdefault("enabled", None)
    data.setdefault("settings", {})
    return data


def _save_state() -> None:
    prefs = _state["prefs"]
    if prefs is None:
        return
    data = _load_state()
    data["enabled"] = sorted(a.module for a in _ADDONS.values() if a.enabled)
    for addon in _ADDONS.values():
        if addon.preferences is not None:
            data["settings"][addon.module] = addon.preferences.to_dict()
    text = json.dumps(data, ensure_ascii=False)
    if getattr(prefs, "addons", None) != text:
        prefs.addons = text


# ------------------------------------------------------------------ 启用与停用
def enable(module: str, save: bool = True) -> bool:
    addon = _ADDONS.get(module)
    if addon is None:
        return False
    if addon.enabled:
        return True
    try:
        instance = _import(addon)
        addon.instance = instance
        prefs_cls = getattr(instance, "PREFERENCES", None)
        if prefs_cls is not None and addon.preferences is None:
            addon.preferences = prefs_cls()
            stored = _load_state()["settings"].get(module)
            if stored:
                addon.preferences.from_dict(stored)
            addon.preferences.changed.connect(lambda _name, m=module: _on_settings_changed(m))
        register = getattr(instance, "register", None)
        if register is not None:
            register()
        addon.enabled = True
        addon.error = ""
        log.info("已启用插件：%s", addon.name)
    except Exception as error:  # noqa: BLE001
        addon.error = "%s: %s" % (type(error).__name__, error)
        log.exception("启用插件 %s 失败", addon.name)
        try:
            if addon.instance is not None and hasattr(addon.instance, "unregister"):
                addon.instance.unregister()
        except Exception:  # noqa: BLE001
            pass
        addon.enabled = False
    if save:
        _save_state()
    changed.emit()
    return addon.enabled


def disable(module: str, save: bool = True) -> None:
    addon = _ADDONS.get(module)
    if addon is None or not addon.enabled:
        return
    try:
        unregister = getattr(addon.instance, "unregister", None)
        if unregister is not None:
            unregister()
        log.info("已停用插件：%s", addon.name)
    except Exception as error:  # noqa: BLE001
        addon.error = "%s: %s" % (type(error).__name__, error)
        log.exception("停用插件 %s 时出错", addon.name)
    addon.enabled = False
    if save:
        _save_state()
    changed.emit()


def _on_settings_changed(module: str) -> None:
    _save_state()


def startup(prefs) -> None:
    """程序启动时调用：扫描插件，按偏好里的记录启用；第一次运行时启用标了 default_enabled 的插件。"""
    _state["prefs"] = prefs
    discover()
    enabled = _load_state()["enabled"]
    if enabled is None:
        enabled = [a.module for a in _ADDONS.values() if a.info.get("default_enabled")]
    for module in enabled:
        if module in _ADDONS:
            enable(module, save=False)
    _save_state()


def shutdown() -> None:
    _save_state()
    for addon in list(_ADDONS.values()):
        if addon.enabled:
            disable(addon.module, save=False)


# ------------------------------------------------------------------ 安装与移除
def install(source: str | Path) -> list[str]:
    """从文件夹、.py 文件或 zip 安装到用户插件目录，返回装好的插件导入名。"""
    source = Path(source)
    target = user_addon_dir()
    names: list[str] = []
    if source.is_dir():
        if not (source / "__init__.py").is_file():
            raise ValueError("文件夹里没有 __init__.py，不是插件")
        dest = target / source.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
        names.append(source.name)
    elif source.suffix == ".py":
        shutil.copy2(source, target / source.name)
        names.append(source.stem)
    elif source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            base = target.resolve()
            for name in archive.namelist():
                resolved = (target / name).resolve()
                if base != resolved and base not in resolved.parents:
                    raise ValueError("压缩包里有不安全的路径：%s" % name)
            roots = {Path(n).parts[0] for n in archive.namelist() if n and not n.startswith("__MACOSX")}
            archive.extractall(target)
        names.extend(Path(r).stem for r in roots)
    else:
        raise ValueError("只能安装文件夹、.py 或 .zip")
    discover()
    return ["%s.%s" % (USER_PACKAGE, n) for n in names if "%s.%s" % (USER_PACKAGE, n) in _ADDONS]


def remove(module: str) -> None:
    """删除一个用户插件（内置插件只能停用）。"""
    addon = _ADDONS.get(module)
    if addon is None or addon.builtin:
        return
    disable(module)
    if addon.path.is_dir():
        shutil.rmtree(addon.path, ignore_errors=True)
    elif addon.path.exists():
        addon.path.unlink()
    sys.modules.pop(module, None)
    _ADDONS.pop(module, None)
    _save_state()
    changed.emit()
