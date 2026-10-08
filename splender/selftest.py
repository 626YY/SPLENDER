"""自动化自检：用真实窗口和真实显卡跑一遍脚本，但窗口放在屏幕外、不激活，不打扰正在使用电脑的人。

用法：python -m splender --selftest 脚本.py
脚本里可以直接用变量 app 和 d（Driver）。脚本跑完自动退出。
"""
from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer, Qt

from .core.keymap import LEFTMOUSE, MOUSEMOVE, MOVE, PRESS, RELEASE, Event

log = logging.getLogger("splender.selftest")

OFFSCREEN_POS = (-12000, -12000)


def prepare_window(window) -> None:
    """显示窗口但不抢焦点，并放到所有显示器之外。"""
    window.setAttribute(Qt.WA_ShowWithoutActivating, True)
    window.setWindowFlag(Qt.WindowDoesNotAcceptFocus, True)
    window.setWindowFlag(Qt.Tool, True)
    window.move(*OFFSCREEN_POS)
    window.show()
    window.move(*OFFSCREEN_POS)


class Driver:
    def __init__(self, app) -> None:
        self.app = app
        self.out = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "selftest"
        self.results: dict = {}

    # ---- 等待 ----
    def wait(self, ms: float) -> None:
        loop = QEventLoop()
        QTimer.singleShot(int(ms), loop.quit)
        loop.exec()

    def settle(self, max_ms: float = 8000) -> None:
        """等引擎和界面都没有待办。"""
        deadline = time.perf_counter() + max_ms / 1000.0
        self.wait(30)
        while time.perf_counter() < deadline:
            engine = self.app.engine
            busy = engine is not None and engine.needs_frame
            if not busy:
                self.wait(40)
                if not (engine is not None and engine.needs_frame):
                    return
            else:
                self.app.request_frame()
                self.wait(10)

    # ---- 取界面对象 ----
    def editor(self, idname: str, index: int = 0):
        found = [a.editor for a in self.app.window.current_screen().areas() if a.editor is not None
                 and a.editor.idname == idname]
        return found[index] if index < len(found) else None

    def screen(self):
        return self.app.window.current_screen()

    # ---- 截图 ----
    def shot(self, name: str, widget=None) -> str:
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / name
        (widget or self.app.window).grab().save(str(path))
        return str(path)

    # ---- 输入（直接走编辑器的事件入口，不向系统注入键鼠）----
    def event(self, editor, type_: str, value: str, x: float = 0.0, y: float = 0.0, **kw) -> bool:
        source = getattr(editor, "widget", None) or getattr(editor, "main", None)
        event = Event(type=type_, value=value, x=x, y=y, prev_x=x, prev_y=y, source=source, **kw)
        return bool(editor.handle_event(event))

    def key(self, editor, name: str, **kw) -> bool:
        x, y = getattr(self, "_last", (0.0, 0.0))
        return self.event(editor, name, PRESS, x, y, **kw)

    def move(self, editor, x: float, y: float, **kw) -> None:
        self._last = (x, y)
        self.event(editor, MOUSEMOVE, MOVE, x, y, **kw)

    def stroke(self, editor, points, button: str = LEFTMOUSE, step_ms: float = 6.0, pressure=None, **kw) -> None:
        """按下、沿 points 移动、抬起。points 是 [(x, y), ...]（编辑器主区域里的逻辑像素）。"""
        x, y = points[0]
        extra = dict(kw)
        if pressure is not None:
            extra.update(is_tablet=True, pressure=pressure if not callable(pressure) else pressure(0.0))
        self.move(editor, x, y)
        self.event(editor, button, PRESS, x, y, **extra)
        count = max(1, len(points) - 1)
        for index, (x, y) in enumerate(points[1:], 1):
            if pressure is not None and callable(pressure):
                extra["pressure"] = pressure(index / count)
            self._last = (x, y)
            self.event(editor, MOUSEMOVE, MOVE, x, y, **extra)
            self.wait(step_ms)
        self.event(editor, button, RELEASE, x, y, **extra)

    def call(self, idname: str, editor=None, **props) -> str:
        from .core import ops

        ctx = editor.context() if editor is not None else self.app.wm.context(use_mouse=False)
        return ops.call(idname, ctx, **props)


def run_script(app, path: str) -> None:
    driver = Driver(app)

    def start() -> None:
        code = 0
        try:
            source = Path(path).read_text(encoding="utf-8")
            exec(compile(source, path, "exec"), {"app": app, "d": driver, "__file__": path})  # noqa: S102
        except SystemExit:
            pass
        except Exception:  # noqa: BLE001
            code = 1
            text = traceback.format_exc()
            log.error("自检脚本出错：\n%s", text)
            try:
                driver.out.mkdir(parents=True, exist_ok=True)
                (driver.out / "error.txt").write_text(text, encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        app.project.dirty = False
        app.qt.exit(code)

    QTimer.singleShot(400, start)
