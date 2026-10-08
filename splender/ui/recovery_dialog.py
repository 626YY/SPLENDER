"""恢复自动保存的窗口、出错提示窗口。"""
from __future__ import annotations

import os
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from . import theme


def _describe(item) -> tuple[str, str, list[str], str]:
    """列表里一行：标题、说明、要注意的事、悬停提示。"""
    from ..core.recovery import describe_age

    when = time.strftime("%m-%d %H:%M", time.localtime(item.saved_at)) if item.saved_at else "时间不明"
    title = "%s　·　%s（%s）" % (item.name, when, describe_age(time.time() - item.saved_at))
    detail = "　·　".join([os.path.basename(item.base) if item.base else "没存过的工程",
                          "%d 页改动" % item.pages, "%.1f MB" % (item.size / 1048576.0)])
    notes = []
    if item.base and not item.base_exists:
        notes.append("原工程文件已经不在了，只能恢复改过的部分")
    elif item.base_changed:
        notes.append("原工程文件在那之后又被改过")
    if item.crash:
        notes.append("那次是崩溃退出的")
    tip = [item.path]
    if item.base:
        tip.insert(0, "原工程：%s" % item.base)
    if item.crash:
        tip.append("崩溃报告：%s" % item.crash)
    return title, detail, notes, "\n".join(tip)


class _Row(QWidget):
    """列表里的一行：标题粗一点，说明淡一点，要注意的事用提醒色。"""

    def __init__(self, item) -> None:
        super().__init__()
        title, detail, notes, tip = _describe(item)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.px(10), theme.px(7), theme.px(10), theme.px(7))
        layout.setSpacing(theme.px(2))
        head = QLabel(title)
        head.setFont(theme.font("font", bold=True))
        layout.addWidget(head)
        body = QLabel(detail)
        body.setProperty("role", "dim")
        layout.addWidget(body)
        for note in notes:
            warn = QLabel(note)
            warn.setStyleSheet("color: %s;" % theme.color("warn"))
            layout.addWidget(warn)
        self.setToolTip(tip)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)


class RecoveryDialog(QDialog):
    """列出可以恢复的自动保存。点「恢复」把选中的那份还原成工程文件并打开。"""

    def __init__(self, app, items: list, startup: bool = False, parent=None) -> None:
        super().__init__(parent or getattr(app, "window", None))
        self.app = app
        self.items = list(items)
        self.setWindowTitle("恢复自动保存")
        self.setMinimumWidth(theme.px(560))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.px(16), theme.px(14), theme.px(16), theme.px(12))
        layout.setSpacing(theme.px(10))
        heading = QLabel("上次 SPLENDER 没有正常关闭。下面是留下的自动保存，选一份恢复："
                         if startup else "程序意外退出时留下的自动保存。选一份恢复：")
        heading.setWordWrap(True)
        heading.setProperty("role", "title")
        layout.addWidget(heading)
        note = QLabel("恢复出来的是一个新的工程文件，原来的工程文件不会被改动。")
        note.setProperty("role", "dim")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.list = QListWidget()
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setStyleSheet(
            "QListWidget { background: %s; border: 1px solid %s; border-radius: %dpx; outline: none; }"
            "QListWidget::item { border-bottom: 1px solid %s; }"
            "QListWidget::item:selected { background: %s; }"
            "QListWidget::item:hover:!selected { background: %s; }"
            % (theme.color("bg.field"), theme.color("line"), theme.px(3), theme.color("bg.panel"),
               theme.color("accent.soft"), theme.color("bg.row_hover")))
        self.list.itemDoubleClicked.connect(lambda _item: self._restore())
        self.list.currentRowChanged.connect(lambda _row: self._update_buttons())
        layout.addWidget(self.list, 1)
        self.status = QLabel("")
        self.status.setProperty("role", "dim")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        row = QHBoxLayout()
        row.setSpacing(theme.px(8))
        self.folder_button = QPushButton("打开所在文件夹")
        self.folder_button.clicked.connect(self._open_folder)
        row.addWidget(self.folder_button)
        self.delete_button = QPushButton("删除")
        self.delete_button.clicked.connect(self._delete)
        row.addWidget(self.delete_button)
        row.addStretch(1)
        self.close_button = QPushButton("以后再说" if startup else "关闭")
        self.close_button.clicked.connect(self.reject)
        row.addWidget(self.close_button)
        self.restore_button = QPushButton("恢复")
        self.restore_button.setProperty("role", "primary")
        self.restore_button.setDefault(True)
        self.restore_button.clicked.connect(self._restore)
        row.addWidget(self.restore_button)
        for button in (self.folder_button, self.delete_button, self.close_button, self.restore_button):
            button.setMinimumHeight(theme.px(26))
        layout.addLayout(row)
        self.restored_path: str | None = None
        self._fill()

    def _fill(self) -> None:
        self.list.clear()
        for item in self.items:
            row = _Row(item)
            entry = QListWidgetItem()
            entry.setData(Qt.UserRole, item.path)
            entry.setToolTip(row.toolTip())
            entry.setSizeHint(row.sizeHint())
            self.list.addItem(entry)
            self.list.setItemWidget(entry, row)
        if self.items:
            self.list.setCurrentRow(0)
        else:
            self.status.setText("没有可以恢复的自动保存。")
        self.list.setMinimumHeight(theme.px(70) * max(1, min(4, len(self.items))))
        self._update_buttons()

    def _current(self):
        row = self.list.currentRow()
        return self.items[row] if 0 <= row < len(self.items) else None

    def _update_buttons(self) -> None:
        has = self._current() is not None
        self.restore_button.setEnabled(has)
        self.delete_button.setEnabled(has)

    def _restore(self) -> None:
        item = self._current()
        if item is None:
            return
        from ..core import ops
        from ..core.ops import FINISHED

        ctx = self.app.wm.context() if self.app.wm is not None else None
        result = ops.call("wm.restore_autosave", ctx, filepath=item.path) if ctx is not None else None
        if result == FINISHED:
            self.restored_path = getattr(self.app, "last_restored", None)
            self.accept()
        else:
            self.status.setText("没有恢复（详细原因在状态栏和日志里）。")

    def _delete(self) -> None:
        item = self._current()
        if item is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("删除自动保存")
        box.setIcon(QMessageBox.Warning)
        box.setText("删掉「%s」的这份自动保存？删掉后就不能恢复了。" % item.name)
        delete = box.addButton("删除", QMessageBox.DestructiveRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is not delete:
            return
        from ..core.recovery import remove_quiet

        remove_quiet(item.path)
        self.items.remove(item)
        self._fill()
        if not self.items:
            self.accept()

    def _open_folder(self) -> None:
        item = self._current()
        folder = os.path.dirname(item.path) if item is not None else None
        if folder is None:
            from ..paths import autosave_dir

            folder = str(autosave_dir(self.app.prefs.files.autosave_dir))
        try:
            os.startfile(folder)  # noqa: S606
        except OSError:
            pass


def error_box(app, kind, value, tb) -> tuple:
    """程序内部出错的提示窗口（还没显示）：说明、可展开的细节、「复制诊断信息」「打开日志文件夹」。
    返回 (窗口, 复制按钮, 文件夹按钮)。"""
    import traceback

    box = QMessageBox(getattr(app, "window", None))
    box.setWindowTitle("SPLENDER 出错了")
    box.setIcon(QMessageBox.Warning)
    message = str(value) or kind.__name__
    box.setText("程序内部出错：%s" % message)
    box.setInformativeText("这个错误已经记进日志，程序可以继续用。建议先「另存为」一份工程。\n"
                           "反馈问题时点「复制诊断信息」，把复制的内容一起发过来。")
    box.setDetailedText("".join(traceback.format_exception(kind, value, tb)))
    copy = box.addButton("复制诊断信息", QMessageBox.ActionRole)
    folder = box.addButton("打开日志文件夹", QMessageBox.ActionRole)
    box.addButton("关闭", QMessageBox.AcceptRole)
    return box, copy, folder


def show_error(app, kind, value, tb) -> None:
    """弹出出错提示，按用户点的按钮复制诊断信息或打开日志文件夹。"""
    from ..core import diagnostics
    from ..paths import log_dir

    box, copy, folder = error_box(app, kind, value, tb)
    box.exec()
    clicked = box.clickedButton()
    if clicked is copy:
        diagnostics.copy_to_clipboard(app)
        if hasattr(app, "report"):
            app.report("诊断信息已复制到剪贴板")
    elif clicked is folder:
        try:
            os.startfile(str(log_dir()))  # noqa: S606
        except OSError:
            pass
