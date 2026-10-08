"""程序入口：python -m splender [工程文件]

--version           打印版本后退出
--selfcheck <文件>   不开窗口，把环境检查结果写进文件后退出（给启动器和安装检查用）
"""
from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    from . import __version__

    if "--version" in args:
        if sys.stdout is not None:
            sys.stdout.write(__version__ + "\n")
        return 0
    if "--selfcheck" in args:
        index = args.index("--selfcheck")
        target = args[index + 1] if index + 1 < len(args) else ""
        report = {"version": __version__, "python": sys.version.split()[0], "executable": sys.executable, "ok": True}
        try:
            from .paths import ensure_vendor_path
            ensure_vendor_path()
            import moderngl  # noqa: F401
            import numpy  # noqa: F401
            import PySide6  # noqa: F401
            report["pyside"] = PySide6.__version__
        except Exception as error:  # noqa: BLE001
            report["ok"] = False
            report["error"] = repr(error)
        if target:
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False)
        return 0 if report["ok"] else 2
    if "--selftest" in args:
        isolate_selftest_user_dir()
    configure_numba_cache()
    from .app import run
    return run(args)


def configure_numba_cache() -> None:
    """numba 编译好的函数存在用户设置目录里（默认存在源码旁边：程序装在只读目录时存不进去，每次启动都要重新编译）。
    要在第一次 import numba 之前设好。外面已经指定了就不动。"""
    import os

    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    try:
        from .paths import user_dir

        path = user_dir() / "numba-cache"
        path.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(path)
    except OSError:
        pass


def isolate_selftest_user_dir() -> None:
    """自检用一个全新的临时设置目录（除非外面指定了 SPLENDER_USER_DIR），绝不碰用户自己的设置、布局和最近文件。"""
    import os
    import tempfile

    if not os.environ.get("SPLENDER_USER_DIR"):
        os.environ["SPLENDER_USER_DIR"] = tempfile.mkdtemp(prefix="splender_selftest_")
    if not os.environ.get("NUMBA_CACHE_DIR"):
        # 临时设置目录每次都是新的：编译缓存放在源码目录的 .cache 里，自检不用每次重新编译
        from pathlib import Path

        cache = Path(__file__).resolve().parents[1] / ".cache" / "numba"
        try:
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["NUMBA_CACHE_DIR"] = str(cache)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
