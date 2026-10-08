"""依次运行全部测试文件（各自独立进程，互不影响），最后汇总。用法：python tests_new/run_all.py"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["test_storage.py", "test_meshio.py", "test_modules.py", "test_ui_offscreen.py", "test_engine_cache.py", "test_pathtracer.py", "test_sculpt.py",
         "test_geometry.py", "test_bake.py", "test_normal_map.py", "test_parts.py", "test_nodes.py", "test_transfer.py", "test_folders.py", "test_engine_paint.py", "test_uv_paint.py", "test_engine_io.py",
         "test_objects.py", "test_lights.py", "test_editmode.py", "test_material_nodes.py", "test_resize.py", "test_keymap.py",
         "test_adjust.py", "test_filters.py",
         "test_effect_brushes.py", "test_selection.py", "test_recovery.py", "test_autosave.py", "test_meshexport.py", "test_wintab.py"]


def main() -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    started = time.perf_counter()
    for name in FILES:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(HERE / name)], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env, cwd=str(HERE.parent),
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        output = proc.stdout + proc.stderr
        ran = re.search(r"Ran (\d+) tests?", output)
        ok = proc.returncode == 0
        print("%-24s %s  %s 项  %.1f 秒" % (name, "通过" if ok else "失败", ran.group(1) if ran else "?",
                                         time.perf_counter() - t0), flush=True)
        if not ok:
            failed.append(name)
            print(output[-3000:])
    print("全部用时 %.1f 秒；%s" % (time.perf_counter() - started, "全部通过" if not failed else "失败：" + "、".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
