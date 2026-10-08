"""Windows 上去掉系统标题栏：窗口内容铺满整个窗口，顶栏自己画最小化、最大化、关闭。

做法：窗口设成无边框（Qt 认为没有边框，几何计算一致），再把系统的标题栏、可调边框样式加回去，
让 Windows 照样提供贴边分屏、Win+方向键、最大化动画和阴影；WM_NCCALCSIZE 返回 0 让客户区盖住整个窗口
（系统不再画标题栏和边框）；WM_NCHITTEST 按鼠标位置告诉系统哪里是标题栏（可拖动、双击最大化）、哪里是可拖动改大小的边。
其它系统上什么都不做，用系统自己的窗口边框。
"""
from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Any, Callable

from PySide6.QtCore import QPoint, Qt

log = logging.getLogger("splender.ui.frameless")

ENABLED = sys.platform == "win32"

WM_NCCALCSIZE = 0x0083
WM_NCHITTEST = 0x0084
GWL_STYLE = -16
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_SYSMENU = 0x00080000
SM_CXFRAME, SM_CYFRAME, SM_CXPADDEDBORDER = 32, 33, 92
HT = {"client": 1, "caption": 2, "left": 10, "right": 11, "top": 12, "topleft": 13, "topright": 14, "bottom": 15,
      "bottomleft": 16, "bottomright": 17}


class _MARGINS(ctypes.Structure):
    _fields_ = [("left", ctypes.c_int), ("right", ctypes.c_int), ("top", ctypes.c_int), ("bottom", ctypes.c_int)]


class _NCCALCSIZE_PARAMS(ctypes.Structure):
    _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]


def _user32():
    return ctypes.windll.user32


class WinFrame:
    """装在一个顶层窗口上。hit_test(本地逻辑坐标) 返回 "caption"（可拖动的标题栏）或 "client"。"""

    def __init__(self, window: Any, hit_test: Callable[[QPoint], str], border: int = 6) -> None:
        self.window = window
        self.hit_test = hit_test
        self.border = border
        self.active = False
        if ENABLED:
            window.setWindowFlags(window.windowFlags() | Qt.FramelessWindowHint)

    def attach(self) -> None:
        """窗口有了原生句柄之后调用（第一次显示时）：把系统的标题栏、边框样式加回去，打开阴影。"""
        if not ENABLED or self.active:
            return
        try:
            hwnd = int(self.window.winId())
            user32 = _user32()
            user32.GetWindowLongW.restype = ctypes.c_long
            style = user32.GetWindowLongW(hwnd, GWL_STYLE)
            user32.SetWindowLongW(hwnd, GWL_STYLE,
                                  style | WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU)
            margins = _MARGINS(-1, -1, -1, -1)
            ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins))
            # 样式变了要让系统重新算一次非客户区
            user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)
            self.active = True
        except Exception:  # noqa: BLE001
            log.exception("去掉系统标题栏失败，用系统边框")

    def native_event(self, event_type: Any, message: Any):
        """MainWindow.nativeEvent 转过来。处理了返回 (True, 结果)，没处理返回 None。"""
        if not self.active:
            return None
        try:
            import shiboken6
            if not shiboken6.isValid(self.window):
                self.active = False
                return None
            if bytes(event_type) not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
                return None
            return self._handle(wintypes.MSG.from_address(int(message)))
        except Exception:  # noqa: BLE001  系统消息里出错不能让程序崩
            return None

    def _handle(self, msg):
        if msg.message == WM_NCCALCSIZE:
            if msg.wParam and self.window.isMaximized():
                # 最大化时系统把窗口往外推出一圈边框，把客户区收回到屏幕里
                params = _NCCALCSIZE_PARAMS.from_address(msg.lParam)
                user32 = _user32()
                pad = user32.GetSystemMetrics(SM_CXPADDEDBORDER)
                fx = user32.GetSystemMetrics(SM_CXFRAME) + pad
                fy = user32.GetSystemMetrics(SM_CYFRAME) + pad
                rect = params.rgrc[0]
                rect.left += fx
                rect.top += fy
                rect.right -= fx
                rect.bottom -= fy
                params.rgrc[0] = rect
            return True, 0
        if msg.message == WM_NCHITTEST:
            x = ctypes.c_short(msg.lParam & 0xFFFF).value
            y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
            return True, HT[self.region(x, y)]
        return None

    def region(self, x: int, y: int) -> str:
        """屏幕物理坐标 (x, y) 属于窗口的哪一块。"""
        window = self.window
        ratio = window.devicePixelRatioF() or 1.0
        rect = wintypes.RECT()
        _user32().GetWindowRect(int(window.winId()), ctypes.byref(rect))
        lx = (x - rect.left) / ratio
        ly = (y - rect.top) / ratio
        width = (rect.right - rect.left) / ratio
        height = (rect.bottom - rect.top) / ratio
        if not window.isMaximized() and not window.isFullScreen():
            b = self.border
            left, right = lx < b, lx >= width - b
            top, bottom = ly < b, ly >= height - b
            if top and left:
                return "topleft"
            if top and right:
                return "topright"
            if bottom and left:
                return "bottomleft"
            if bottom and right:
                return "bottomright"
            if left:
                return "left"
            if right:
                return "right"
            if top:
                return "top"
            if bottom:
                return "bottom"
        try:
            return self.hit_test(QPoint(int(lx), int(ly)))
        except Exception:  # noqa: BLE001
            return "client"
