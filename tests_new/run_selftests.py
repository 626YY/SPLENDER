"""依次跑全部整机自检（真实窗口放在屏幕外、不抢焦点），最后汇总。用法：python tests_new/run_selftests.py [名字…]

每个自检单独一个进程；用临时的用户目录（不碰真实的偏好设置）。失败时打印 error.txt 和日志末尾。"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVIDENCE = ROOT / "docs" / "evidence" / "selftest"
NAMES = ["basic", "window", "objects", "edit", "gizmo", "shader", "render", "sculpt", "bake", "folders", "export", "parts", "uvpaint", "nodes", "adjust", "filters", "tools", "select"]


def _print(text: str) -> None:
    """控制台的编码印不出来的字换成问号，别让打印本身把汇总弄崩。"""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, "replace").decode(encoding, "replace"), flush=True)


def main(argv: list[str]) -> int:
    names = argv or NAMES
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("SPLENDER_USER_DIR", None)
    failed = []
    started = time.perf_counter()
    for name in names:
        script = HERE / ("selftest_%s.py" % name)
        error = EVIDENCE / "error.txt"
        if error.exists():
            error.unlink()
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", "splender", "--selftest", str(script)], cwd=str(ROOT), env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=900)
        ok = proc.returncode == 0 and not error.exists()
        print("%-10s %s  %.1f 秒" % (name, "通过" if ok else "失败", time.perf_counter() - t0), flush=True)
        if not ok:
            failed.append(name)
            if error.exists():
                _print(error.read_text(encoding="utf-8", errors="replace")[-3000:])
            _print((proc.stdout + proc.stderr)[-2000:])
    print("全部用时 %.1f 秒；%s" % (time.perf_counter() - started, "全部通过" if not failed else "失败：" + "、".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
