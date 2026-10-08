"""「调整上一步」和「重复上一步」（照 Blender 视口左下角的 Adjust Last Operation 面板、Shift+R）。

能调整的操作（类上 redo = True，比如添加立方体、移动、设置原点）做完后，视口左下角出现一个小面板，列出它的参数；
改任何一个参数，就撤销上一步、用新参数重做一次。之后做了别的事（撤销历史变了），面板就收起来。F9 在鼠标处弹出同样的内容。
"""
from __future__ import annotations

import logging
from typing import Any

import shiboken6
from PySide6.QtCore import QPoint, QRectF, Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..core import ops
from ..core.signals import Signal
from . import icons, theme

log = logging.getLogger("splender.ui.redo")

_CONTROLLERS: dict[int, "RedoController"] = {}


def controller(app) -> "RedoController":
    found = _CONTROLLERS.get(id(app))
    if found is None or found.app is not app:
        found = RedoController(app)
        _CONTROLLERS[id(app)] = found
    return found


def draw_operator_props(layout, op) -> None:
    """操作的可调参数（不含隐藏的）。操作可以自己定义 draw_redo(layout) 排版。"""
    layout.use_property_split = True
    custom = getattr(op, "draw_redo", None)
    if callable(custom):
        custom(layout)
        return
    for name, prop in type(op).properties().items():
        if getattr(prop, "hidden", False):
            continue
        layout.prop(op, name)


class RedoController:
    def __init__(self, app) -> None:
        self.app = app
        self.last_op = None
        self.last_ctx = None
        self.last_index = -1
        self.repeat_op = None          # Shift+R 重复的那个操作：撤销、重做之后也还在（调整面板会收起）
        self.changed = Signal()
        self._adjusting = False
        self._scheduled = False
        ops.finished.connect(self._on_finished)
        history = getattr(app, "history", None)
        if history is not None:
            history.changed.connect(self._on_history)

    # ---- 记录 ----
    def _on_finished(self, op, ctx, result) -> None:
        if self._adjusting or result != ops.FINISHED:
            return
        if not getattr(op, "redo", False) or getattr(op, "redo_disabled", False):
            return
        history = getattr(self.app, "history", None)
        if history is None or not history.can_undo:
            return
        self._detach()
        self.last_op = op
        self.repeat_op = op
        self.last_ctx = ctx
        self.last_index = history.index
        op.changed.connect(self._on_prop_changed)
        self.changed.emit()

    def _detach(self) -> None:
        if self.last_op is not None:
            try:
                self.last_op.changed.disconnect(self._on_prop_changed)
            except Exception:  # noqa: BLE001
                pass

    def _on_history(self) -> None:
        if self._adjusting or self.last_op is None:
            return
        history = self.app.history
        if history.index != self.last_index:
            self.clear()

    def clear(self) -> None:
        self._detach()
        self.last_op = None
        self.last_ctx = None
        self.last_index = -1
        self.changed.emit()

    # ---- 调整 ----
    def can_adjust(self) -> bool:
        history = getattr(self.app, "history", None)
        return (self.last_op is not None and history is not None and history.can_undo
                and history.index == self.last_index)

    def _on_prop_changed(self, _name: str = "") -> None:
        if self._adjusting or self._scheduled:
            return
        self._scheduled = True
        QTimer.singleShot(0, self.adjust)

    def adjust(self) -> bool:
        """撤销上一步，用面板上的新参数重做。"""
        self._scheduled = False
        if not self.can_adjust():
            return False
        history = self.app.history
        op, ctx = self.last_op, self.last_ctx
        self._adjusting = True
        try:
            # 「复制」「挤出」后接着的移动：只重做移动，前面那步留着
            merged = history.split_last() if int(getattr(op, "merge_previous", 0) or 0) else 0
            history.undo()
            try:
                host = getattr(self.app, "host", None)
                if host is not None:
                    host.make_current()
                result = op.execute(ctx)
            except Exception as error:  # noqa: BLE001
                log.exception("调整上一步时出错")
                self.app.report("没能按新参数重做：%s" % error, "ERROR")
                result = ops.CANCELLED
            if not (result == ops.FINISHED and history.can_undo):
                history.redo()
            if merged:
                history.merge_last(merged)
            self.last_index = history.index
        finally:
            self._adjusting = False
        self.app.request_frame()
        self.app.notify("redo")
        return True

    def repeat(self, ctx) -> str:
        """用同样的参数再做一次上一个操作。"""
        op = self.repeat_op
        if op is None:
            return ops.CANCELLED
        props = {name: getattr(op, name) for name in type(op).properties()}
        return ops.call(op.idname, ctx, invoke=False, **props)


class RedoPanel(QWidget):
    """视口左下角的「调整上一步」面板：标题一行（点一下展开或收起），下面是参数。"""

    expanded = True       # 所有视口共用展开状态（和 Blender 一样记住上次）

    def __init__(self, editor, parent: QWidget) -> None:
        super().__init__(parent)
        self.editor = editor
        self.setAttribute(Qt.WA_StyledBackground, False)
        self._box = QVBoxLayout(self)
        pad = theme.px(6)
        self._box.setContentsMargins(pad, theme.px(26), pad, pad)
        self._box.setSpacing(0)
        self._body = QWidget(self)
        self._box.addWidget(self._body)
        self._layout = None
        self._op = None
        self.hide()
        self.ctl = controller(editor.app)
        self.ctl.changed.connect(self.rebuild)

    def header_rect(self) -> QRectF:
        return QRectF(0, 0, self.width(), theme.px(26))

    def rebuild(self) -> None:
        if not shiboken6.isValid(self):
            return
        op = self.ctl.last_op
        mine = op is not None and self._belongs_here()
        if not mine:
            self._op = None
            self.hide()
            return
        if op is not self._op:
            self._op = op
            from .layout import UILayout

            if self._layout is not None:
                self._layout.clear()
            else:
                self._layout = UILayout(self._body, self.editor.context())
            self._layout.ctx = self.editor.context()
            try:
                draw_operator_props(self._layout, op)
            except Exception:  # noqa: BLE001
                log.exception("调整面板生成出错")
        self._body.setVisible(RedoPanel.expanded)
        self.place()
        self.show()
        self.raise_()
        # 旧控件要等这一轮事件结束才真正删掉，再量一次高度
        QTimer.singleShot(0, self._replace)

    def _replace(self) -> None:
        if shiboken6.isValid(self) and self.isVisible():
            self.place()

    def _belongs_here(self) -> bool:
        ctx = self.ctl.last_ctx
        editor = getattr(ctx, "editor", None) if ctx is not None else None
        if editor is self.editor:
            return True
        active = getattr(self.editor.app, "active_view3d", None)
        return (editor is None or getattr(editor, "idname", "") != "VIEW_3D") and active is self.editor

    def place(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        width = theme.px(270)
        self.setFixedWidth(width)
        # 换了参数多少不一样的操作：先放开上次定死的高度，让布局重新算
        self.setMinimumHeight(0)
        self.setMaximumHeight(16777215)
        inner = self._body.layout()
        if inner is not None:
            inner.invalidate()
            inner.activate()
        self._box.invalidate()
        self._box.activate()
        height = self._box.sizeHint().height() if RedoPanel.expanded else theme.px(26) + theme.px(4)
        self.setFixedHeight(min(height, max(theme.px(40), parent.height() - theme.px(20))))
        self.move(theme.px(10), parent.height() - self.height() - theme.px(10))

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = float(theme.size("radius")) + 1.0
        painter.setPen(theme.qcolor("line"))
        painter.setBrush(theme.qcolor("bg.panel", 0.94))
        painter.drawRoundedRect(rect, radius, radius)
        header = self.header_rect()
        icon = icons.pixmap("chevron.down" if RedoPanel.expanded else "chevron.right", theme.color("text.dim"),
                            theme.px(12), self.devicePixelRatioF())
        painter.drawPixmap(QPoint(theme.px(8), int(header.center().y()) - theme.px(6)), icon)
        painter.setPen(theme.qcolor("text"))
        painter.setFont(theme.font())
        label = getattr(self._op, "label", "") or getattr(self._op, "idname", "")
        painter.drawText(header.adjusted(theme.px(26), 0, -theme.px(8), 0), Qt.AlignVCenter | Qt.AlignLeft, label)
        painter.end()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if self.header_rect().contains(event.position()):
            RedoPanel.expanded = not RedoPanel.expanded
            self._body.setVisible(RedoPanel.expanded)
            self.place()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)


def open_popover(ctx) -> None:
    """F9：在鼠标处弹出上一步的参数。"""
    from .layout import _open_floating_popover

    ctl = controller(ctx.app)
    op = ctl.last_op
    if op is None:
        return

    def draw(layout, _ctx) -> None:
        layout.label(op.label or op.idname, role="title")
        draw_operator_props(layout, op)

    _open_floating_popover(draw, ctx)
