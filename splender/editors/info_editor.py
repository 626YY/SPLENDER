"""信息：性能面板和操作记录。画的时候可以直接看到延迟和显存。"""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget

from .. import log as applog
from ..core import registry
from ..ui import theme
from ..ui.editor import Editor


class PerfPanel(QWidget):
    """几行数字加一条每帧耗时的柱状图。"""

    def __init__(self, editor) -> None:
        super().__init__()
        self.editor = editor
        self.lines: list[tuple[str, str, str]] = []
        self.history = deque(maxlen=180)
        self.setMinimumHeight(theme.px(150))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(theme.px(320), theme.px(206))

    def update_stats(self) -> None:
        app = self.editor.app
        engine = app.engine
        if engine is None:
            self.lines = [("三维显示", "不可用", "warn")]
            self.update()
            return
        s = engine.stats()
        host = app.host
        present = sorted(host.present_ms)
        if present:
            present_p50 = present[len(present) // 2]
            present_p95 = present[min(len(present) - 1, int(len(present) * 0.95))]
        else:                                   # 还没交给屏幕显示过：用引擎算好画面的时间
            present_p50, present_p95 = s["latency_ms_p50"], s["latency_ms_p95"]

        def grade(value: float, good: float, bad: float) -> str:
            return "ok" if value <= good else ("warn" if value <= bad else "error")

        if present_p95 > 0.0:
            latency = ("落笔到显示", "%.0f ms（95%% 的情况在 %.0f ms 以内）" % (present_p50, present_p95),
                       grade(present_p95, 34, 60))
        else:
            latency = ("落笔到显示", "还没有落笔", "")
        merge = ("抬笔后并入图层", "%.0f ms，期间画面照常" % s["merge_ms_last"] if s["merge_ms_last"] else "还没有落笔", "")
        if s.get("merges_pending"):
            merge = ("抬笔后并入图层", "正在进行（%d 笔）" % s["merges_pending"], "")
        self.lines = [
            latency,
            merge,
            ("每帧引擎耗时", "%.1f ms（95%%：%.1f ms）" % (s["frame_ms_p50"], s["frame_ms_p95"]), grade(s["frame_ms_p95"], 8, 16)),
            ("盖章 / 合成 / 渲染", "%.1f / %.1f / %.1f ms" % (s["stamp_ms_p95"], s["compose_ms_p95"], s["render_ms_p95"]), ""),
            ("图层显存", "%d / %d MB，共 %s 页" % (s["used_mb"], s["budget_mb"], format(s["layer_pages"], ",")),
             grade(s["used_mb"] / max(1, s["budget_mb"]), 0.85, 0.97)),
            ("显示缓存", "%d / %d MB" % (s["display_mb"], s["display_budget_mb"]), ""),
            ("内存缓存", "%d / %d MB" % (s["ram_mb"], s["ram_budget_mb"]), ""),
            ("换出 / 换入 / 读盘", "%s / %s / %s 页" % (format(s["evicted"], ","), format(s["uploaded"], ","),
                                                format(s["disk_reads"], ",")), ""),
            ("被迫等待换页", "%d 次" % (s["blocking_evictions"] + s["sync_disk_reads"]),
             "ok" if s["blocking_evictions"] + s["sync_disk_reads"] == 0 else "warn"),
        ]
        frames = list(engine.perf.frame_ms)
        if frames:
            self.history.extend(frames[-6:])
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pad = theme.size("pad")
        line_h = theme.px(19)
        label_w = theme.px(128)
        painter.setFont(theme.font())
        y = pad
        for label, value, level in self.lines:
            painter.setPen(theme.qcolor("text.dim"))
            painter.drawText(QRectF(pad, y, label_w, line_h), Qt.AlignVCenter | Qt.AlignLeft, label)
            painter.setPen(theme.qcolor(level if level in ("ok", "warn", "error") else "text"))
            painter.drawText(QRectF(pad + label_w, y, self.width() - label_w - 2 * pad, line_h),
                             Qt.AlignVCenter | Qt.AlignLeft, value)
            y += line_h
        # 每帧耗时柱状图：一格 16.7ms 的参考线
        chart = QRectF(pad, y + theme.px(4), self.width() - 2 * pad, max(theme.px(20), self.height() - y - pad))
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.qcolor("bg.field"))
        painter.drawRoundedRect(chart, theme.px(3), theme.px(3))
        if self.history:
            top = 33.4
            values = list(self.history)
            step = chart.width() / max(len(values), 60)
            for index, value in enumerate(values):
                h = min(1.0, value / top) * (chart.height() - 2)
                color = "ok" if value <= 8 else ("warn" if value <= 16.7 else "error")
                painter.setBrush(theme.qcolor(color, 0.85))
                painter.drawRect(QRectF(chart.left() + index * step, chart.bottom() - 1 - h, max(1.0, step - 1), h))
            ref = chart.bottom() - 1 - (16.7 / top) * (chart.height() - 2)
            painter.setPen(QPen(theme.qcolor("text.faint"), 1, Qt.DashLine))
            painter.drawLine(chart.left(), ref, chart.right(), ref)


@registry.register_editor
class InfoEditor(Editor):
    idname = "INFO"
    label = "信息"
    icon = "editor.info"
    category = "通用"
    order = 50
    description = "性能数字和操作记录"

    def build_main(self):
        root = QWidget()
        box = QVBoxLayout(root)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.perf = PerfPanel(self)
        box.addWidget(self.perf)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1500)
        self.log_view.setFrameShape(QPlainTextEdit.NoFrame)
        self.log_view.setFont(theme.font("font.small"))
        box.addWidget(self.log_view, 1)
        self._seen = 0
        self._timer = QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._tick)
        return root

    def draw_header(self, layout, ctx) -> None:
        layout.prop(ctx.prefs.viewport, "show_stats", text="在视口里显示", toggle=True)
        layout.stretch()

    def on_show(self) -> None:
        self._timer.start()
        self._tick()

    def on_hide(self) -> None:
        self._timer.stop()

    def _tick(self) -> None:
        self.perf.update_stats()
        records = applog.records
        total = len(records)
        if total < self._seen:
            self._seen = 0
        if total > self._seen:
            fresh = list(records)[self._seen:]
            self._seen = total
            self.log_view.appendPlainText("\n".join("%s  %s" % (when, text) if level == "INFO" else
                                                    "%s  [%s] %s" % (when, {"WARNING": "注意", "ERROR": "出错"}.get(level, level), text)
                                                    for when, level, text in fresh))
