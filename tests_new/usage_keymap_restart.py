"""实际使用测：在偏好设置的键位页里改 20 条键位（真的按键录进去，再停用一条），退出；重新启动后它们都还在：
新的键能触发那个操作、旧的键不再触发，菜单上显示新的快捷键。

直接运行（python tests_new/usage_keymap_restart.py）时是总指挥：用一个临时设置目录先后启动两次 SPLENDER（自检模式，
窗口在屏幕外）。结果写到 docs/evidence/selftest/usage_keymap_restart.json。"""
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

    def run(phase: str, user: str) -> int:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", SPLENDER_USER_DIR=user, SPLENDER_KEYMAP_PHASE=phase)
        proc = subprocess.run([sys.executable, "-m", "splender", "--selftest", str(Path(__file__).resolve())],
                              cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=600)
        if proc.returncode != 0:
            print((proc.stdout + proc.stderr)[-3000:])
        return proc.returncode

    def main() -> int:
        user = tempfile.mkdtemp(prefix="splender_keymap_")
        report = {}
        try:
            report["change_exit"] = run("change", user)
            report["check_exit"] = run("check", user)
            for name in ("changed.json", "checked.json"):
                path = Path(user, name)
                if path.exists():
                    report[name.split(".")[0]] = json.loads(path.read_text(encoding="utf-8"))
        finally:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "usage_keymap_restart.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                           encoding="utf-8")
            shutil.rmtree(user, ignore_errors=True)
        checked = report.get("checked", {})
        print(json.dumps({k: v for k, v in checked.items() if k != "items"}, ensure_ascii=False))
        ok = report.get("change_exit") == 0 and report.get("check_exit") == 0 and checked.get("ok")
        return 0 if ok else 1

    if __name__ == "__main__":
        raise SystemExit(main())
else:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication

    from splender.core.keymap import PRESS, Event, is_modified
    from splender.paths import user_dir

    phase = os.environ.get("SPLENDER_KEYMAP_PHASE", "")
    folder = user_dir()
    kc = app.keyconfig
    KEYS = [(Qt.Key_F1 + n, "F%d" % (n + 1)) for n in range(12)] + [(getattr(Qt, "Key_" + c), c) for c in "QWERTYUI"]
    MODS = Qt.ControlModifier | Qt.ShiftModifier | Qt.AltModifier

    if phase == "change":
        from splender.selftest import prepare_window
        from splender.ui import theme
        from splender.ui.window import PopoutWindow

        window = PopoutWindow(app, app.wm, {"type": "area", "editor": "PREFERENCES", "state": {}}, "偏好设置")
        window.resize(theme.px(900), theme.px(760))
        prepare_window(window)
        d.wait(100)
        editor = next(a.editor for a in window.findChildren(type(app.window.current_screen().areas()[0]))
                      if getattr(a, "editor", None) is not None and a.editor.idname == "PREFERENCES")
        editor.tabs.setCurrent("keymap")
        editor._on_section("keymap")
        d.wait(200)
        widget = editor.keymap_widget
        # 挑 20 条：每张表的第一条键盘键位，不够再从大表里拿
        rows, used = [], set()
        for row in widget.rows:
            name, item = row.data(0, Qt.UserRole)
            if name in used or item.type.endswith("MOUSE") or item.value != PRESS:
                continue
            used.add(name)
            rows.append(row)
        for row in widget.rows:
            if len(rows) >= 20:
                break
            _name, item = row.data(0, Qt.UserRole)
            if row not in rows and not item.type.endswith("MOUSE") and item.value == PRESS:
                rows.append(row)
        rows = rows[:20]
        expected = []
        for row, (qt_key, key_name) in zip(rows, KEYS):
            name, item = row.data(0, Qt.UserRole)
            old = (item.type, item.ctrl, item.shift, item.alt)
            widget.start_recording(row)
            QApplication.sendEvent(widget.tree, QKeyEvent(QEvent.KeyPress, int(qt_key), MODS))
            QApplication.sendEvent(widget.tree, QKeyEvent(QEvent.KeyRelease, int(qt_key), MODS))
            assert (item.type, item.ctrl, item.shift, item.alt) == (key_name, True, True, True), \
                (item.op, item.type, item.ctrl, item.shift, item.alt)
            expected.append({"keymap": name, "op": item.op, "props": item.props, "old": old,
                             "new": [key_name, True, True, True], "text": item.shortcut_text(), "active": True})
        # 再停用一条（勾掉前面的框）
        off = next(r for r in widget.rows if r not in rows and r.data(0, Qt.UserRole)[1].value == PRESS
                   and not r.data(0, Qt.UserRole)[1].type.endswith("MOUSE"))
        off.setCheckState(0, Qt.Unchecked)
        name, item = off.data(0, Qt.UserRole)
        assert not item.active
        expected.append({"keymap": name, "op": item.op, "props": item.props,
                         "old": (item.type, item.ctrl, item.shift, item.alt),
                         "new": [item.type, item.ctrl, item.shift, item.alt], "text": "", "active": False})
        widget.only_modified.setChecked(True)
        d.wait(150)
        d.shot("keymap_01_modified.png", widget=window)
        modified = sum(1 for km in kc.keymaps.values() for i in km.items if is_modified(i))
        assert modified == 21, modified
        window.close()
        with open(os.path.join(folder, "changed.json"), "w", encoding="utf-8") as handle:
            json.dump({"items": expected, "modified": modified}, handle, ensure_ascii=False)
        # 自检脚本跑完程序正常退出，退出时把偏好设置写盘
    elif phase == "check":
        expected = json.load(open(os.path.join(folder, "changed.json"), encoding="utf-8"))["items"]
        results = []
        for entry in expected:
            km = kc.keymaps[entry["keymap"]]
            names = [entry["keymap"]]
            matches = [i for i in km.items if i.op == entry["op"] and i.props == entry["props"]]
            new = entry["new"]
            item = next((i for i in matches if [i.type, i.ctrl, i.shift, i.alt] == new), None)
            ok = item is not None and item.active == entry["active"]
            fires = old_fires = None
            if ok and entry["active"]:
                event = Event(type=new[0], value=PRESS, ctrl=new[1], shift=new[2], alt=new[3])
                found = kc.lookup(event, names)
                fires = found is item
                old = entry["old"]
                old_event = Event(type=old[0], value=PRESS, ctrl=old[1], shift=old[2], alt=old[3])
                found_old = kc.lookup(old_event, names)
                old_fires = found_old is not None and found_old.op == entry["op"] and found_old.props == entry["props"]
                ok = fires and not old_fires and kc.shortcut_for(entry["op"], entry["props"] or None) != ""
            elif ok:
                old = entry["old"]
                found = kc.lookup(Event(type=old[0], value=PRESS, ctrl=old[1], shift=old[2], alt=old[3]), names)
                ok = found is not item
            results.append({"op": entry["op"], "keymap": entry["keymap"], "ok": bool(ok), "fires": fires,
                            "old_fires": old_fires})
        report = {"count": len(results), "passed": sum(r["ok"] for r in results),
                  "ok": all(r["ok"] for r in results) and len(results) == 21, "items": results}
        with open(os.path.join(folder, "checked.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False)
        assert report["ok"], [r for r in results if not r["ok"]]
