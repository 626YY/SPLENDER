"""程序用到的目录：资源、用户配置、日志、暂存。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
VENDOR_DIR = ROOT_DIR / "runtime" / "site-packages"


def ensure_vendor_path() -> None:
    """把随程序携带的第三方库目录加入搜索路径。"""
    path = str(VENDOR_DIR)
    if VENDOR_DIR.is_dir() and path not in sys.path:
        sys.path.insert(0, path)


def user_dir() -> Path:
    base = os.environ.get("SPLENDER_USER_DIR")
    if base:
        root = Path(base)
    else:
        root = Path(os.environ.get("APPDATA", str(Path.home()))) / "SPLENDER"
    root.mkdir(parents=True, exist_ok=True)
    return root


def log_dir() -> Path:
    path = user_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def autosave_dir(custom: str = "") -> Path:
    """自动保存（恢复文件、会话锁）放哪：给了就用给的，否则是用户设置目录里的 autosave。"""
    path = Path(custom) if custom else user_dir() / "autosave"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scratch_dir() -> Path:
    path = user_dir() / "scratch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resource(*parts: str) -> Path:
    return RESOURCE_DIR.joinpath(*parts)
