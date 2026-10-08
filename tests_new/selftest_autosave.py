"""整机自检：自动保存与崩溃恢复。

画几笔 → 按计时器的流程（到时间、停手）自动保存，靠每帧推进写完 → 会话锁和恢复文件都在；
再画一笔 → 只补写新改的页 → 把恢复文件当成「上次崩溃留下的」（死掉的进程的会话锁）→ 启动检查能找到它 →
恢复窗口截图 → 用「恢复这份自动保存」还原成工程文件并打开 → 画面和当时一样；诊断信息齐全。
变量 app 和 d 由自检框架提供。"""
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

errors = []


class _Catch(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append(record.getMessage())


logging.getLogger("splender").addHandler(_Catch())
from splender.core import diagnostics, recovery  # noqa: E402
from splender.engine.projectio import composite_image  # noqa: E402
from splender.selftest import prepare_window  # noqa: E402
from splender.ui.recovery_dialog import RecoveryDialog  # noqa: E402

app.window.set_workspace("绘制")
results = {}
d.settle()
d.out.mkdir(parents=True, exist_ok=True)
engine = app.engine
saver = engine.autosave
session = app.session
assert session is not None and session.lock_path.is_file(), "启动时建会话锁"
results["session"] = session.id

view = d.editor("VIEW_3D")
cx, cy = view.widget.width() / 2, view.widget.height() / 2


def paint(dy, color):
    app.tool_settings.brush.color = color
    d.stroke(view, [(cx - 160 + 320 * i / 40, cy + dy + 20 * math.sin(i * 0.3)) for i in range(41)])
    d.settle()


paint(-60, (0.85, 0.25, 0.15))
paint(40, (0.15, 0.55, 0.85))
assert app.project.dirty

# 1. 按计时器的流程：到时间、停手够久 → 开始；每帧推进写完（不调用 wait）
app._autosave_due = 0.0
app.wm.last_input = time.perf_counter() - 60.0
app._autosave_tick()
assert saver.active, "到时间又停手了，应该开始自动保存"
deadline = time.perf_counter() + 60
while saver.active and time.perf_counter() < deadline:
    app.request_frame()
    d.wait(15)
assert not saver.active, "自动保存没做完"
first = dict(saver.last_result)
results["first"] = {k: v for k, v in first.items() if k != "path"}
path = saver.path
assert path and os.path.isfile(path) and session.owns(path), path
assert app._autosave_due > time.monotonic(), "存完排好下一次"

# 正在画的时候不开始（等那一笔画完）
app._autosave_due = 0.0
engine.stroke = object()
try:
    app._autosave_tick()
    assert not saver.active, "笔划进行中不开始自动保存"
finally:
    engine.stroke = None
# 下面都手动开始：计时器别插手（不然它会在画画的间隙自己先存掉）
app.prefs.files.autosave = False
app._schedule_autosave()

# 2. 再画一笔：只补写新改的页
paint(0, (0.95, 0.85, 0.2))
assert app.autosave_now(wait=True)
second = dict(saver.last_result)
results["second"] = {k: v for k, v in second.items() if k != "path"}
assert 0 < second["pages"] <= first["pages_total"] + 64, second
assert not app.autosave_now(wait=True), "没有新改动时不存"
ts = app.project.active_texture_set
reference = composite_image(engine, ts, 512).astype(np.float32)
layer_names = [layer.name for layer in ts.layers]

# 3. 当成上次崩溃留下的：换成一个已经退出的进程的会话
proc = subprocess.Popen([sys.executable, "-c", "pass"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
started = recovery.process_started(proc.pid)
proc.wait()
dead = "20260101-000000-%d" % proc.pid
saver.writer.sync()
copy_path = os.path.join(os.path.dirname(path), dead + "-1" + recovery.RECOVER_SUFFIX)
for side in ("", "-wal", "-shm", "-journal"):    # 崩溃时日志文件也留在旁边，一起拷
    if os.path.exists(path + side):
        shutil.copyfile(path + side, copy_path + side)
crash = os.path.join(os.path.dirname(path), "crash-%s.txt" % dead)
with open(crash, "w", encoding="utf-8") as handle:
    handle.write("Fatal Python error: 模拟的崩溃\n")
with open(os.path.join(os.path.dirname(path), dead + recovery.LOCK_SUFFIX), "w", encoding="utf-8") as handle:
    json.dump({"pid": proc.pid, "started": started, "version": "test", "crash": crash}, handle)
items, ended = app.autosave_scan()
results["found"] = [item.name for item in items]
assert [item.path for item in items] == [copy_path], [item.path for item in items]
assert [e["session"] for e in ended] == [dead]
assert items[0].crash == crash

# 恢复窗口
dialog = RecoveryDialog(app, items, startup=True)
dialog.resize(720, 360)
prepare_window(dialog)
d.wait(150)
d.shot("autosave_01_dialog.png", widget=dialog)
dialog.close()

# 4. 恢复：先存一下当前工程（免得问要不要保存），再从恢复文件还原
import tempfile  # noqa: E402

saved_dir = tempfile.mkdtemp(prefix="splender_autosave_work_")
app.prefs.files.restore_dir = saved_dir
assert d.call("wm.save_as", filepath=os.path.join(saved_dir, "当前.splender")) == "FINISHED"
saver.writer.sync()                               # 删文件在写盘线程里做
assert saver.path is None and not os.path.exists(path), "存盘后自己的恢复文件作废"
assert d.call("wm.restore_autosave", filepath=copy_path) == "FINISHED"
d.settle()
restored = app.last_restored
results["restored"] = os.path.basename(restored)
assert restored and os.path.dirname(restored) == saved_dir and app.project.path == restored
assert not os.path.exists(copy_path), "恢复后那份自动保存删掉"
ts2 = app.project.active_texture_set
assert [layer.name for layer in ts2.layers] == layer_names
image = composite_image(engine, ts2, 512).astype(np.float32)
diff = float(np.abs(image - reference).max())
results["restore_max_diff"] = round(diff, 5)
assert diff <= 1.5 / 255.0, diff
d.shot("autosave_02_restored.png")

# 出错提示窗口、偏好设置里的「文件」一页（屏幕外显示后截图，不弹出来）
from splender.ui.recovery_dialog import error_box  # noqa: E402

try:
    raise RuntimeError("模拟的内部错误")
except RuntimeError as error:
    box, _copy, _folder = error_box(app, type(error), error, error.__traceback__)
prepare_window(box)
d.wait(150)
d.shot("autosave_03_error.png", widget=box)
box.close()
from splender.ui import theme  # noqa: E402
from splender.ui.window import PopoutWindow  # noqa: E402

prefs_window = PopoutWindow(app, app.wm, {"type": "area", "editor": "PREFERENCES", "state": {}}, "偏好设置")
prefs_window.resize(theme.px(760), theme.px(700))
prepare_window(prefs_window)
d.wait(100)
editors = [a.editor for a in prefs_window.findChildren(type(app.window.current_screen().areas()[0]))
           if getattr(a, "editor", None) is not None and a.editor.idname == "PREFERENCES"]
assert editors, "偏好设置编辑器没建出来"
editors[0].tabs.setCurrent("files")
editors[0]._on_section("files")
d.wait(200)
d.shot("autosave_04_prefs_files.png", widget=prefs_window)
prefs_window.close()

# 5. 诊断信息
text = diagnostics.collect(app)
for word in ("SPLENDER", "系统：", "显卡：", "工程：", "自动保存：", "最近的日志："):
    assert word in text, word
results["diagnostics_lines"] = text.count("\n") + 1
(d.out / "autosave_diagnostics.txt").write_text(text, encoding="utf-8")
from splender.core import ops  # noqa: E402

for idname in ("wm.recover_autosave", "wm.restore_autosave", "wm.copy_diagnostics"):
    assert ops.get(idname) is not None, idname
assert d.call("wm.copy_diagnostics") == "FINISHED"
(d.out / "autosave_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
assert not errors, errors
