"""WinTab 数位板接口（只在 Windows 上）。

Qt 6 在 Windows 上只走 Windows Ink（WM_POINTER）。数位板驱动里关掉「Windows Ink」（不少人为了别的软件这样设）或者
老驱动只有 WinTab 时，Qt 收不到数位板事件，笔只是一只鼠标、没有压感。这里直接用驱动带的 wintab32.dll：
在主窗口上开一个系统上下文，每来一个鼠标事件就把 WinTab 队列里的包取空，拿最新那个包的压感、倾斜、是不是橡皮端
补到这个鼠标事件上。笔还照常控制系统光标（CXO_SYSTEM），位置、按键仍然来自鼠标事件。

偏好设置「数位板接口」：
- AUTO：有 Windows Ink 的数位板事件就用它；只有鼠标事件、但 WinTab 那边笔就在板上时，从 WinTab 补压感。
- WINTAB：一律从 WinTab 取（Qt 的数位板事件不用，让它合成鼠标事件）。
- WININK：只用 Windows Ink，不开 WinTab。
"""
from __future__ import annotations

import ctypes
import logging
import math
import os
import time
from ctypes import wintypes

log = logging.getLogger("splender.wintab")

# wintab.h
WTI_DEFSYSCTX = 4
WTI_DEVICES = 100
DVC_NPRESSURE = 15
DVC_ORIENTATION = 17
PK_STATUS = 0x0002
PK_TIME = 0x0004
PK_CURSOR = 0x0020
PK_BUTTONS = 0x0040
PK_NORMAL_PRESSURE = 0x0400
PK_ORIENTATION = 0x1000
PACKET_DATA = PK_STATUS | PK_TIME | PK_CURSOR | PK_BUTTONS | PK_NORMAL_PRESSURE | PK_ORIENTATION
CXO_SYSTEM = 0x0001
CXO_MESSAGES = 0x0004
TPS_PROXIMITY = 0x0001          # 置位表示笔离开了感应范围
TPS_INVERT = 0x0010             # 笔倒过来（橡皮端）
LCNAMELEN = 40

#: 包超过这么久没来就不用它补压感（笔离开了、或者是真鼠标在动）
FRESH_SECONDS = 0.25
QUEUE_SIZE = 128


class LOGCONTEXTW(ctypes.Structure):
    _fields_ = [("lcName", wintypes.WCHAR * LCNAMELEN), ("lcOptions", wintypes.UINT), ("lcStatus", wintypes.UINT),
                ("lcLocks", wintypes.UINT), ("lcMsgBase", wintypes.UINT), ("lcDevice", wintypes.UINT),
                ("lcPktRate", wintypes.UINT), ("lcPktData", wintypes.DWORD), ("lcPktMode", wintypes.DWORD),
                ("lcMoveMask", wintypes.DWORD), ("lcBtnDnMask", wintypes.DWORD), ("lcBtnUpMask", wintypes.DWORD),
                ("lcInOrgX", wintypes.LONG), ("lcInOrgY", wintypes.LONG), ("lcInOrgZ", wintypes.LONG),
                ("lcInExtX", wintypes.LONG), ("lcInExtY", wintypes.LONG), ("lcInExtZ", wintypes.LONG),
                ("lcOutOrgX", wintypes.LONG), ("lcOutOrgY", wintypes.LONG), ("lcOutOrgZ", wintypes.LONG),
                ("lcOutExtX", wintypes.LONG), ("lcOutExtY", wintypes.LONG), ("lcOutExtZ", wintypes.LONG),
                ("lcSensX", wintypes.DWORD), ("lcSensY", wintypes.DWORD), ("lcSensZ", wintypes.DWORD),
                ("lcSysMode", wintypes.BOOL), ("lcSysOrgX", ctypes.c_int), ("lcSysOrgY", ctypes.c_int),
                ("lcSysExtX", ctypes.c_int), ("lcSysExtY", ctypes.c_int), ("lcSysSensX", wintypes.DWORD),
                ("lcSysSensY", wintypes.DWORD)]


class AXIS(ctypes.Structure):
    _fields_ = [("axMin", wintypes.LONG), ("axMax", wintypes.LONG), ("axUnits", wintypes.UINT),
                ("axResolution", wintypes.DWORD)]


class PACKET(ctypes.Structure):
    """按 PACKET_DATA 里各位从低到高排的包。"""
    _fields_ = [("status", wintypes.UINT), ("time", wintypes.DWORD), ("cursor", wintypes.UINT),
                ("buttons", wintypes.DWORD), ("pressure", wintypes.UINT), ("azimuth", ctypes.c_int),
                ("altitude", ctypes.c_int), ("twist", ctypes.c_int)]


class PenState:
    """最新一个包换算出来的笔的状态。"""

    __slots__ = ("pressure", "tilt_x", "tilt_y", "eraser", "near", "buttons", "received")

    def __init__(self) -> None:
        self.pressure = 0.0
        self.tilt_x = 0.0
        self.tilt_y = 0.0
        self.eraser = False
        self.near = False
        self.buttons = 0
        self.received = 0.0


def pen_from_packet(packet, pressure_range: tuple[int, int], altitude_max: int, state: PenState | None = None,
                    now: float | None = None) -> PenState:
    """一个包换算成笔的状态：压感 0..1、倾斜（度，和 Qt 的 xTilt/yTilt 同向）、橡皮端、在不在板子上。"""
    state = state or PenState()
    low, high = pressure_range
    span = max(1, int(high) - int(low))
    state.pressure = min(1.0, max(0.0, (int(packet.pressure) - int(low)) / span))
    state.eraser = bool(int(packet.status) & TPS_INVERT)
    state.near = not (int(packet.status) & TPS_PROXIMITY)
    state.buttons = int(packet.buttons)
    if altitude_max > 0:
        altitude = abs(int(packet.altitude)) / float(altitude_max) * 90.0      # 90 度是竖直
        azimuth = math.radians(int(packet.azimuth) / 10.0)                       # 十分之一度，从正上方顺时针
        lean = max(0.0, 90.0 - altitude)
        state.tilt_x = lean * math.sin(azimuth)
        state.tilt_y = -lean * math.cos(azimuth)
    else:
        state.tilt_x = state.tilt_y = 0.0
    state.received = time.perf_counter() if now is None else now
    return state


def annotate(event, pen: PenState | None, now: float | None = None) -> bool:
    """笔就在板子上、包是刚来的：把压感、倾斜、橡皮端补到这个鼠标事件上。返回补没补。"""
    if pen is None or not pen.near:
        return False
    now = time.perf_counter() if now is None else now
    if now - pen.received > FRESH_SECONDS:
        return False
    event.is_tablet = True
    event.pressure = pen.pressure
    event.tilt_x = pen.tilt_x
    event.tilt_y = pen.tilt_y
    event.is_eraser = pen.eraser
    return True


class WinTab:
    """一个打开着的 WinTab 上下文。open() 失败（没装驱动、没接数位板）时 ok 为假，别的方法什么都不做。"""

    def __init__(self) -> None:
        self.ok = False
        self.reason = ""
        self._dll = None
        self._ctx = None
        self._buffer = (PACKET * QUEUE_SIZE)()
        self.pressure_range = (0, 1023)
        self.altitude_max = 0
        self.pen = PenState()
        self.packets = 0

    def open(self, hwnd: int) -> bool:
        if os.name != "nt":
            self.reason = "只在 Windows 上有"
            return False
        try:
            dll = ctypes.WinDLL("wintab32.dll")
        except OSError:
            self.reason = "没有装数位板驱动的 WinTab（wintab32.dll）"
            return False
        dll.WTInfoW.argtypes = [wintypes.UINT, wintypes.UINT, ctypes.c_void_p]
        dll.WTInfoW.restype = wintypes.UINT
        dll.WTOpenW.argtypes = [wintypes.HWND, ctypes.POINTER(LOGCONTEXTW), wintypes.BOOL]
        dll.WTOpenW.restype = ctypes.c_void_p
        dll.WTClose.argtypes = [ctypes.c_void_p]
        dll.WTClose.restype = wintypes.BOOL
        dll.WTPacketsGet.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        dll.WTPacketsGet.restype = ctypes.c_int
        dll.WTQueueSizeSet.argtypes = [ctypes.c_void_p, ctypes.c_int]
        dll.WTQueueSizeSet.restype = wintypes.BOOL
        dll.WTOverlap.argtypes = [ctypes.c_void_p, wintypes.BOOL]
        dll.WTOverlap.restype = wintypes.BOOL
        try:
            if not dll.WTInfoW(0, 0, None):
                self.reason = "WinTab 没有响应（数位板服务没开？）"
                return False
            axis = AXIS()
            if dll.WTInfoW(WTI_DEVICES, DVC_NPRESSURE, ctypes.byref(axis)):
                self.pressure_range = (int(axis.axMin), int(axis.axMax) or 1023)
            orientation = (AXIS * 3)()
            if dll.WTInfoW(WTI_DEVICES, DVC_ORIENTATION, ctypes.byref(orientation)):
                self.altitude_max = int(orientation[1].axMax)
            context = LOGCONTEXTW()
            if not dll.WTInfoW(WTI_DEFSYSCTX, 0, ctypes.byref(context)):
                self.reason = "取不到 WinTab 的系统上下文"
                return False
            context.lcName = "SPLENDER"
            context.lcOptions = (context.lcOptions | CXO_SYSTEM) & ~CXO_MESSAGES     # 笔照常管光标；不发消息，自己取包
            context.lcPktData = PACKET_DATA
            context.lcPktMode = 0
            context.lcMoveMask = PACKET_DATA
            context.lcBtnUpMask = context.lcBtnDnMask
            handle = dll.WTOpenW(int(hwnd), ctypes.byref(context), True)
            if not handle:
                self.reason = "WinTab 打不开（数位板没接上？）"
                return False
            size = QUEUE_SIZE
            while size >= 8 and not dll.WTQueueSizeSet(handle, size):
                size //= 2
            self._dll, self._ctx = dll, handle
        except OSError as error:
            self.reason = "WinTab 出错：%s" % error
            return False
        self.ok = True
        log.info("WinTab 已打开：压感 %d..%d，倾斜%s", self.pressure_range[0], self.pressure_range[1],
                 "有" if self.altitude_max else "没有")
        return True

    def poll(self) -> PenState | None:
        """把队列里的包取空，返回最新的笔状态（这次没有新包就是上次的）。"""
        if not self.ok:
            return None
        try:
            count = self._dll.WTPacketsGet(self._ctx, QUEUE_SIZE, ctypes.byref(self._buffer))
        except OSError:
            return self.pen
        if count > 0:
            self.packets += count
            pen_from_packet(self._buffer[count - 1], self.pressure_range, self.altitude_max, self.pen)
        return self.pen

    def to_front(self) -> None:
        """窗口激活时把上下文放到最上面（多个程序都开了 WinTab 时包才会给我们）。"""
        if self.ok:
            try:
                self._dll.WTOverlap(self._ctx, True)
            except OSError:
                pass

    def close(self) -> None:
        if self._ctx is not None and self._dll is not None:
            try:
                self._dll.WTClose(self._ctx)
            except OSError:
                pass
        self._ctx = None
        self.ok = False
