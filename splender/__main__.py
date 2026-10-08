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
    from .app import run
    return run(args)


def isolate_selftest_user_dir() -> None:
    """自检用一个全新的临时设置目录（除非外面指定了 SPLENDER_USER_DIR），绝不碰用户自己的设置、布局和最近文件。"""
    import os
    import tempfile

    if not os.environ.get("SPLENDER_USER_DIR"):
        os.environ["SPLENDER_USER_DIR"] = tempfile.mkdtemp(prefix="splender_selftest_")


if __name__ == "__main__":
    raise SystemExit(main())
