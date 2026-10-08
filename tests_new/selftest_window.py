"""整机自检：主窗口没有系统标题栏。客户区铺满整个窗口；顶栏空白处是标题栏（能拖动、双击最大化），
菜单、标签、窗口按钮是内容；四边和四角能拖动改大小。变量 app 和 d 由自检框架提供。"""
import ctypes
import json
import sys
from ctypes import wintypes

results = {}
d.settle()
win = app.window
d.out.mkdir(parents=True, exist_ok=True)
if sys.platform == "win32":
    assert win.frame is not None and win.frame.active, "Windows 上应该去掉系统标题栏"
    hwnd = int(win.winId())
    user32 = ctypes.windll.user32
    wr, cr = wintypes.RECT(), wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(wr))
    user32.GetClientRect(hwnd, ctypes.byref(cr))
    results["window_size"] = [wr.right - wr.left, wr.bottom - wr.top]
    results["client_size"] = [cr.right - cr.left, cr.bottom - cr.top]
    assert results["window_size"] == results["client_size"], results
    ratio = win.devicePixelRatioF()

    def hit(widget, x, y):
        point = widget.mapTo(win, widget.rect().topLeft()) if widget is not win else None
        gx = wr.left + int(round(((point.x() if point else 0) + x) * ratio))
        gy = wr.top + int(round(((point.y() if point else 0) + y) * ratio))
        lparam = (gy & 0xFFFF) << 16 | (gx & 0xFFFF)
        user32.SendMessageW.restype = ctypes.c_ssize_t
        return int(user32.SendMessageW(hwnd, 0x0084, 0, lparam))

    bar = win.topbar
    title = bar.title
    results["caption_empty"] = hit(bar, bar.tabs.geometry().right() + 30, bar.height() // 2)
    results["caption_title"] = hit(title, title.width() // 2, title.height() // 2)
    first = bar.menus.actions()[0]
    rect = bar.menus.actionGeometry(first)
    results["menu_item"] = hit(bar.menus, rect.center().x(), rect.center().y())
    results["tab"] = hit(bar.tabs, bar.tabs.tab_rect(0).center().x(), bar.tabs.tab_rect(0).center().y())
    close = bar.buttons[-1]
    results["close_button"] = hit(close, close.width() // 2, close.height() // 2)
    results["left_edge"] = hit(win, 2, win.height() // 2)
    results["bottom_right"] = hit(win, win.width() - 2, win.height() - 2)
    results["top_edge"] = hit(win, win.width() // 2, 1)
    results["content"] = hit(win, win.width() // 2, win.height() // 2)
    expect = {"caption_empty": 2, "caption_title": 2, "menu_item": 1, "tab": 1, "close_button": 1, "left_edge": 10,
              "bottom_right": 17, "top_edge": 12, "content": 1}
    wrong = {k: (results[k], v) for k, v in expect.items() if results[k] != v}
    assert not wrong, wrong
    bar.set_maximized(True)
    d.settle()
    d.shot("window_01_topbar_maximized_icon.png", widget=bar)
    bar.set_maximized(False)
d.shot("window_00_full.png")
(d.out / "window_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
