"""实际使用测：启动要多久。连着启动两次（第一次编译缓存是空的，第二次用第一次留下的），量从进程开始到：
「窗口出来、默认 16K 工程建好」和「画完第一笔」。结果写到 docs/evidence/selftest/usage_startup.json。

直接运行（python tests_new/usage_startup.py [python 路径]）时是总指挥；给了 python 路径（比如便携包里的
runtime\\python\\python.exe）就用它启动。"""
import json
import os
import sys
import time

if "app" not in globals():
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    OUT = ROOT / "docs" / "evidence" / "selftest"

    def main(argv: list[str]) -> int:
        python = argv[0] if argv else sys.executable
        cwd = str(Path(python).resolve().parents[2]) if argv else str(ROOT)     # 便携包：runtime\\python\\python.exe 上两级
        user = tempfile.mkdtemp(prefix="splender_startup_")
        cache = tempfile.mkdtemp(prefix="splender_numba_")
        report = {"python": python, "runs": []}
        try:
            for index in (1, 2):
                env = dict(os.environ, PYTHONIOENCODING="utf-8", SPLENDER_USER_DIR=user, NUMBA_CACHE_DIR=cache,
                           SPLENDER_STARTUP_RUN=str(index))
                started = time.time()
                proc = subprocess.run([python, "-m", "splender", "--selftest", str(Path(__file__).resolve())], cwd=cwd,
                                      env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=900)
                result_path = Path(user, "startup-%d.json" % index)
                entry = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
                entry.update(run=index, exit=proc.returncode, wall=round(time.time() - started, 2),
                             cache_files=sum(1 for _ in Path(cache).rglob("*.nbi")))
                if proc.returncode != 0:
                    entry["output"] = (proc.stdout + proc.stderr)[-1500:]
                report["runs"].append(entry)
                print(json.dumps({k: v for k, v in entry.items() if k != "output"}, ensure_ascii=False), flush=True)
        finally:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "usage_startup.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            shutil.rmtree(user, ignore_errors=True)
            shutil.rmtree(cache, ignore_errors=True)
        return 0 if all(r.get("exit") == 0 for r in report["runs"]) else 1

    if __name__ == "__main__":
        raise SystemExit(main(sys.argv[1:]))
else:
    import math

    import psutil

    from splender.paths import user_dir

    created = psutil.Process(os.getpid()).create_time()
    result = {"ready": round(time.time() - created, 2)}
    app.window.set_workspace("绘制")
    d.settle()
    view = d.editor("VIEW_3D")
    cx, cy = view.widget.width() / 2, view.widget.height() / 2
    d.stroke(view, [(cx - 150 + 300 * i / 40, cy + 15 * math.sin(i * 0.3)) for i in range(41)])
    d.settle()
    result["first_stroke"] = round(time.time() - created, 2)
    result["texture"] = app.project.active_texture_set.size
    with open(os.path.join(user_dir(), "startup-%s.json" % os.environ.get("SPLENDER_STARTUP_RUN", "0")), "w",
              encoding="utf-8") as handle:
        json.dump(result, handle)
