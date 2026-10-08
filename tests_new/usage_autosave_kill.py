"""实际使用测：强杀进程后重启，能不能、多快恢复到最近一次自动保存。

直接运行（python tests_new/usage_autosave_kill.py）时是总指挥：
1. 用一个临时设置目录启动 SPLENDER（自检模式，窗口在屏幕外），默认 16K 工程里铺满一层、加杂色、再画几笔，
   自动保存，记下当时的画面，然后原地等着；
2. 总指挥像崩溃一样直接结束这个进程（不走正常退出）；
3. 再启动一次 SPLENDER（同一个设置目录），找自动保存、恢复成工程文件、打开，对比画面，量从启动到恢复完成多久。
结果写到 docs/evidence/selftest/usage_autosave_kill.json。

被自检框架执行时（有 app 和 d），按环境变量 SPLENDER_KILL_PHASE 做「干活」或「检查」那一段。"""
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

    def launch(phase: str, user: str) -> subprocess.Popen:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", SPLENDER_USER_DIR=user, SPLENDER_KILL_PHASE=phase)
        return subprocess.Popen([sys.executable, "-m", "splender", "--selftest", str(Path(__file__).resolve())],
                                cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def main() -> int:
        user = tempfile.mkdtemp(prefix="splender_kill_")
        ready = Path(user, "ready.json")
        report = {}
        try:
            worker = launch("work", user)
            deadline = time.time() + 600
            while not ready.exists() and worker.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if not ready.exists():
                print("干活的进程没有准备好（退出码 %s）" % worker.poll())
                worker.kill()
                return 1
            report["work"] = json.loads(ready.read_text(encoding="utf-8"))
            time.sleep(0.5)
            worker.kill()                       # 像崩溃一样直接结束（TerminateProcess），不走正常退出
            worker.wait(30)
            report["killed_exit_code"] = worker.returncode
            started = time.time()
            checker = launch("check", user)
            checker.wait(600)
            report["restart_to_restored_seconds"] = None
            result = Path(user, "checked.json")
            if result.exists():
                report["check"] = json.loads(result.read_text(encoding="utf-8"))
                report["restart_to_restored_seconds"] = round(report["check"]["restored_at"] - started, 2)
            report["checker_exit_code"] = checker.returncode
        finally:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "usage_autosave_kill.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=1))
            shutil.rmtree(user, ignore_errors=True)
        ok = report.get("checker_exit_code") == 0 and report.get("check", {}).get("ok")
        return 0 if ok else 1

    if __name__ == "__main__":
        raise SystemExit(main())
else:
    import math

    import numpy as np

    from splender.core.keymap import LEFTMOUSE, PRESS, RELEASE
    from splender.doc.project import PLANE_COLOR
    from splender.engine.projectio import composite_image
    from splender.paths import user_dir

    phase = os.environ.get("SPLENDER_KILL_PHASE", "")
    folder = user_dir()
    app.window.set_workspace("绘制")
    app.prefs.files.autosave = False
    d.settle()
    engine = app.engine
    view = d.editor("VIEW_3D")
    cx, cy = view.widget.width() / 2, view.widget.height() / 2
    if phase == "work":
        tools = app.tool_settings
        tools.tool = "paint.fill"
        tools.fill.mode = "LAYER"
        tools.brush.color = (0.45, 0.4, 0.3)
        d.settle()
        d.move(view, cx, cy)
        d.event(view, LEFTMOUSE, PRESS, cx, cy)
        d.event(view, LEFTMOUSE, RELEASE, cx, cy)
        d.settle()
        ts = app.project.active_texture_set
        engine.apply_filter(ts, ts.active_layer, "ADD_NOISE", {"amount": 0.2, "seed": 7}, {PLANE_COLOR: (True,)}, "杂色")
        tools.tool = "paint.brush"
        tools.brush.color = (0.9, 0.2, 0.1)
        d.settle()
        for k in range(3):
            d.stroke(view, [(cx - 150 + i * 3.75, cy - 60 + 50 * k + 20 * math.sin(i * 0.1)) for i in range(81)])
            d.settle()
        app.project.mark_dirty(True)
        started = time.perf_counter()
        assert app.autosave_now(wait=True)
        result = dict(engine.autosave.last_result)
        np.save(os.path.join(folder, "reference.npy"), composite_image(engine, ts, 512, dtype=np.uint8))
        info = {"pages": result["pages"], "autosave_seconds": round(time.perf_counter() - started, 2),
                "layers": [layer.name for layer in ts.layers], "session": app.session.id}
        tmp = os.path.join(folder, "ready.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(info, handle, ensure_ascii=False)
        os.replace(tmp, os.path.join(folder, "ready.json"))
        while True:                       # 等着被结束
            d.wait(200)
    elif phase == "check":
        import psutil

        launched = psutil.Process(os.getpid()).create_time()
        app.prefs.files.restore_dir = str(folder)      # 恢复出来的放在临时设置目录里，别落到用户的文档文件夹
        items, ended = app.autosave_scan()
        out = {"found": [item.name for item in items], "ended": [e["session"] for e in ended],
               "app_ready_seconds": round(time.time() - launched, 2)}
        assert len(items) == 1, out
        started = time.perf_counter()
        target = app.restore_autosave(items[0].path)
        d.settle(60000)
        out["restore_seconds"] = round(time.perf_counter() - started, 2)
        out["restored_at"] = time.time()
        ts = app.project.active_texture_set
        reference = np.load(os.path.join(folder, "reference.npy")).astype(np.int16)
        image = composite_image(engine, ts, 512, dtype=np.uint8).astype(np.int16)
        out["max_diff"] = int(np.abs(image - reference).max())
        out["layers"] = [layer.name for layer in ts.layers]
        out["target"] = os.path.basename(target)
        out["ok"] = out["max_diff"] <= 1
        with open(os.path.join(folder, "checked.json"), "w", encoding="utf-8") as handle:
            json.dump(out, handle, ensure_ascii=False)
        assert out["ok"], out
