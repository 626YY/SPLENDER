"""渲染结果：显示 F12 出图的进度和画面，可以保存。"""
from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QWidget

from ..core import registry
from ..ui import theme
from ..ui.editor import Editor


class _ResultView(QWidget):
    """按比例缩放显示画面；透明部分画棋盘格。"""

    def __init__(self, editor: "RenderResultEditor") -> None:
        super().__init__()
        self.editor = editor
        self.image: QImage | None = None
        self.setMinimumSize(120, 90)

    def set_array(self, array) -> None:
        if array is None:
            self.image = None
        else:
            height, width = array.shape[:2]
            self._data = array.copy()                   # QImage 不复制数据，留一份引用
            self.image = QImage(self._data.data, width, height, width * 4, QImage.Format_RGBA8888)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.qcolor("bg.area"))
        job = self.editor.app.render_job
        if self.image is None:
            painter.setPen(theme.qcolor("text.dim"))
            text = "按 F12 用当前视口的视角出图" if job is None else (job.status or "准备中")
            painter.drawText(self.rect(), Qt.AlignCenter, text)
            painter.end()
            return
        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih, 1.0 if self.editor.fit_small else 1e9)
        w, h = iw * scale, ih * scale
        target = QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        cell = theme.px(8)
        painter.save()
        painter.setClipRect(target)
        light, dark = QColor(70, 70, 74), QColor(52, 52, 56)
        y = target.top()
        row = 0
        while y < target.bottom():
            x = target.left()
            col = row % 2
            while x < target.right():
                painter.fillRect(QRectF(x, y, cell, cell), light if col % 2 == 0 else dark)
                x += cell
                col += 1
            y += cell
            row += 1
        painter.restore()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(target, self.image)
        painter.end()


@registry.register_editor
class RenderResultEditor(Editor):
    idname = "RENDER_RESULT"
    label = "渲染结果"
    icon = "render"
    category = "通用"
    order = 18
    description = "F12 出图的进度和画面"

    def __init__(self, app, area=None) -> None:
        self.fit_small = True
        self._shown_job = None
        super().__init__(app, area)

    def build_main(self):
        self.view = _ResultView(self)
        self.timer = QTimer(self.view)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        return self.view

    def _poll(self) -> None:
        job = self.app.render_job
        if job is None:
            return
        array = job.preview()
        if array is not None:
            self.view.set_array(array)
        if job is not self._shown_job or not job.done:
            self._shown_job = job
            self.refresh_header()
        elif job.done and getattr(self, "_final_shown", None) is not job:
            self._final_shown = job
            self.refresh_header()

    def draw_header(self, layout, ctx) -> None:
        job = self.app.render_job
        if job is None:
            layout.label("还没有渲染", role="dim")
        else:
            if job.done:
                text = job.error or ("完成，用时 %.1f 秒" % job.elapsed)
            else:
                text = "%s · %.0f%% · %.1f 秒" % (job.status or "渲染中", job.progress * 100.0, job.elapsed)
            layout.label(text, role="dim" if job.done and not job.error else "")
        layout.stretch()
        layout.operator("render.render", text="渲染", icon="render")
        layout.operator("render.cancel", text="停止", icon="close")
        layout.operator("render.save_image", text="保存图片", icon="save")

    def dispose(self) -> None:
        if getattr(self, "timer", None) is not None:
            self.timer.stop()
        super().dispose()
