"""WinTab：包和上下文的内存布局对得上 wintab.h；压感、倾斜、橡皮端、在不在板子上换算对；只用刚来的包补鼠标事件；
装了驱动的机器上真的打开一次（没接数位板时打不开也不能出错）。"""
import ctypes
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
from splender.core.keymap import Event  # noqa: E402
from splender.ui import wintab as W  # noqa: E402


def packet(pressure=0, status=0, azimuth=0, altitude=900, buttons=0):
    p = W.PACKET()
    p.pressure, p.status, p.azimuth, p.altitude, p.buttons = pressure, status, azimuth, altitude, buttons
    return p


class WinTabTest(unittest.TestCase):
    def test_layouts_match_wintab_h(self):
        self.assertEqual(ctypes.sizeof(W.LOGCONTEXTW), 212)
        self.assertEqual(ctypes.sizeof(W.AXIS), 16)
        self.assertEqual(ctypes.sizeof(W.PACKET), 32)      # 状态、时间、笔、按键、压感各 4 字节，朝向 3 个 int

    def test_pen_from_packet(self):
        pen = W.pen_from_packet(packet(pressure=4096), (0, 8191), 900)
        self.assertAlmostEqual(pen.pressure, 4096 / 8191, places=4)
        self.assertTrue(pen.near)
        self.assertFalse(pen.eraser)
        self.assertAlmostEqual(pen.tilt_x, 0.0)
        self.assertAlmostEqual(pen.tilt_y, 0.0)
        pen = W.pen_from_packet(packet(pressure=9999, status=W.TPS_INVERT, azimuth=900, altitude=450), (0, 8191), 900)
        self.assertEqual(pen.pressure, 1.0)
        self.assertTrue(pen.eraser)
        self.assertAlmostEqual(pen.tilt_x, 45.0, places=3)     # 朝右倒 45 度
        self.assertAlmostEqual(pen.tilt_y, 0.0, places=3)
        pen = W.pen_from_packet(packet(status=W.TPS_PROXIMITY), (0, 1023), 0)
        self.assertFalse(pen.near)

    def test_annotate_only_fresh_and_near(self):
        pen = W.pen_from_packet(packet(pressure=300), (0, 1000), 900, now=100.0)
        event = Event(type="MOUSEMOVE")
        self.assertTrue(W.annotate(event, pen, now=100.1))
        self.assertTrue(event.is_tablet)
        self.assertAlmostEqual(event.pressure, 0.3)
        stale = Event(type="MOUSEMOVE")
        self.assertFalse(W.annotate(stale, pen, now=100.0 + W.FRESH_SECONDS + 0.01))
        self.assertFalse(stale.is_tablet)
        self.assertEqual(stale.pressure, 1.0)
        pen.near = False
        self.assertFalse(W.annotate(Event(type="MOUSEMOVE"), pen, now=100.1))
        self.assertFalse(W.annotate(Event(type="MOUSEMOVE"), None))

    @unittest.skipUnless(os.name == "nt" and os.environ.get("SPLENDER_TEST_WINTAB") == "1",
                         "要真的打开 WinTab 时设 SPLENDER_TEST_WINTAB=1（新开的上下文会排到最前面，别的软件里正在画的笔会被抢）")
    def test_open_with_real_driver(self):
        tablet = W.WinTab()
        hwnd = ctypes.windll.user32.GetDesktopWindow()
        started = time.perf_counter()
        ok = tablet.open(hwnd)
        print("\nWinTab：%s（%s），用时 %.0f ms" % ("打开了" if ok else "没打开", tablet.reason or "压感 %r" %
                                               (tablet.pressure_range,), (time.perf_counter() - started) * 1000))
        self.assertEqual(ok, tablet.ok)
        if ok:
            tablet.poll()
            tablet.to_front()
        tablet.close()
        self.assertFalse(tablet.ok)
        self.assertIsNone(tablet.poll())


if __name__ == "__main__":
    unittest.main()
