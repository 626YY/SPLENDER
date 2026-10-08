"""整机自检：数位板接口（用假的 WinTab，不碰真的驱动）。真的 Qt 鼠标事件、数位板事件经过视口的输入转发：
- 笔在板子上、包是刚来的：鼠标事件补上压感、橡皮端；笔离开后不补；
- 「WinTab」模式下不用 Windows Ink 的笔事件（让 Qt 合成鼠标事件）；「自动」模式下直接用；
- 切到「Windows Ink」时把 WinTab 关掉。
变量 app 和 d 由自检框架提供。"""
import json
import time

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QInputDevice, QMouseEvent, QPointingDevice, QTabletEvent
from PySide6.QtWidgets import QApplication

from splender.ui import wintab as W

results = {}


class FakeTablet:
    ok = True

    def __init__(self):
        self.pen = W.PenState()
        self.packets = 0
        self.closed = False

    def poll(self):
        self.packets += 1
        return self.pen

    def to_front(self):
        pass

    def close(self):
        self.ok = False
        self.closed = True


app.window.set_workspace("绘制")
d.settle()
view = d.editor("VIEW_3D")
widget = view.widget
captured = []
original = view.handle_event


def spy(event):
    captured.append({"type": event.type, "value": event.value, "tablet": event.is_tablet,
                     "pressure": round(event.pressure, 3), "eraser": event.is_eraser})
    return original(event)


view.handle_event = spy
cx, cy = widget.width() / 2, widget.height() / 2


def mouse(kind, x, y, button=Qt.NoButton, buttons=Qt.NoButton):
    pos = QPointF(x, y)
    QApplication.sendEvent(widget, QMouseEvent(kind, pos, widget.mapToGlobal(pos), button, buttons, Qt.NoModifier))


def tablet(kind, x, y, pressure):
    pos = QPointF(x, y)
    device = QPointingDevice("测试笔", 4242, QInputDevice.DeviceType.Stylus, QPointingDevice.PointerType.Pen,
                             QInputDevice.Capability.Position | QInputDevice.Capability.Pressure, 1, 3)
    button = Qt.LeftButton if kind != QEvent.TabletMove else Qt.NoButton
    buttons = Qt.LeftButton if kind != QEvent.TabletRelease else Qt.NoButton
    event = QTabletEvent(kind, device, pos, QPointF(widget.mapToGlobal(pos)), pressure, 0.0, 0.0, 0.0, 0.0, 0.0,
                         Qt.NoModifier, button, buttons)
    QApplication.sendEvent(widget, event)
    return event


# 1. 「自动」+ WinTab：笔在板子上、包刚来 → 鼠标事件补压感、橡皮端
app.prefs.paint.tablet_api = "AUTO"
fake = FakeTablet()
app.wintab = fake
fake.pen.near = True
fake.pen.pressure = 0.35
fake.pen.eraser = True
fake.pen.received = time.perf_counter()
mouse(QEvent.MouseMove, cx, cy)
results["near"] = captured[-1]
assert captured[-1]["tablet"] and captured[-1]["pressure"] == 0.35 and captured[-1]["eraser"], captured[-1]
# 拿笔画一笔：按下、移动、抬起都带压感，画上了
fake.pen.eraser = False
before = app.history.index
for step in range(25):
    fake.pen.pressure = 0.2 + 0.03 * step
    fake.pen.received = time.perf_counter()
    if step == 0:
        mouse(QEvent.MouseButtonPress, cx - 100, cy, Qt.LeftButton, Qt.LeftButton)
    mouse(QEvent.MouseMove, cx - 100 + step * 8, cy, Qt.NoButton, Qt.LeftButton)
    d.wait(8)
mouse(QEvent.MouseButtonRelease, cx + 100, cy, Qt.LeftButton, Qt.NoButton)
d.settle()
pressures = [c["pressure"] for c in captured if c["type"] == "MOUSEMOVE" and c["tablet"]]
results["stroke_pressures"] = [pressures[0], pressures[-1]] if pressures else []
results["stroke_recorded"] = app.history.index > before or app.history.undo_label == "笔划"
assert results["stroke_recorded"], "笔划应该画上了"
assert pressures and pressures[-1] > pressures[0]
# 2. 笔离开 → 不补
fake.pen.near = False
mouse(QEvent.MouseMove, cx + 10, cy)
results["away"] = captured[-1]
assert not captured[-1]["tablet"] and captured[-1]["pressure"] == 1.0
# 3. 「WinTab」模式：Windows Ink 的笔事件不用
app.prefs.paint.tablet_api = "WINTAB"
d.wait(50)
app.wintab = fake = FakeTablet()
count = len(captured)
event = tablet(QEvent.TabletMove, cx, cy, 0.8)
results["wintab_mode_tablet_used"] = len(captured) > count
assert len(captured) == count and not event.isAccepted(), "WinTab 模式下不用 Windows Ink 的笔事件"
# 4. 「自动」模式：Windows Ink 的笔事件直接用
app.prefs.paint.tablet_api = "AUTO"
d.wait(50)
count = len(captured)
tablet(QEvent.TabletMove, cx, cy, 0.8)
assert len(captured) == count + 1 and captured[-1]["tablet"] and captured[-1]["pressure"] == 0.8, captured[-1]
results["auto_mode_tablet"] = captured[-1]
# 5. 切到「Windows Ink」：WinTab 关掉
app.wintab = fake = FakeTablet()
app.prefs.paint.tablet_api = "WININK"
d.wait(50)
assert fake.closed and app.wintab is None
results["wink_closes_wintab"] = True
view.handle_event = original
app.prefs.paint.tablet_api = "AUTO"
(d.out / "tablet_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
