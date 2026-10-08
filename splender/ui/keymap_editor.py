"""偏好设置里的键位页：按键位表分组列出全部键位，能搜索、改键（点快捷键那一格，再按新的键或鼠标键）、停用、
恢复出厂，同一个键会互相挡住时用橙色标出来。改动立即生效，存进偏好设置（只存和出厂不一样的）。"""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QBrush, QFont
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import keymap as K
from ..core import ops
from . import theme
from .editor import key_event, modifier_state, mouse_button_name

ROLE = Qt.UserRole
_MODIFIER_KEYS = {int(Qt.Key_Shift), int(Qt.Key_Control), int(Qt.Key_Alt), int(Qt.Key_Meta), int(Qt.Key_AltGr)}
_MOUSE_ONLY = (K.CLICK, K.CLICK_DRAG, K.DOUBLE, K.WHEEL)
RESET_TEXT = "恢复"


def _prop_text(cls, key: str, value) -> str:
    """一个操作参数的显示：菜单写菜单名、选项写选项名、开关写开关名（关着的不写）、工具写工具名。"""
    from ..core import registry
    from ..core.props import BoolProperty, EnumProperty

    prop = cls.properties().get(key) if cls is not None else None
    if isinstance(value, str):
        menu = registry.menu(value)
        if menu is not None:
            return str(getattr(menu, "label", value) or value)
    if isinstance(prop, EnumProperty):
        return prop.label_of(value)
    if isinstance(prop, BoolProperty) or isinstance(value, bool):
        return (prop.label if prop is not None else key) if value else ""
    if isinstance(value, str) and "," in value:                 # 一串工具（轮换）
        return "、".join(_tool_name(v) for v in value.split(","))
    if isinstance(value, str):
        return _tool_name(value)
    if isinstance(value, (int, float)) and prop is not None and prop.label:
        return "%s %s" % (prop.label, "%.3g" % value)
    return str(value)


def _tool_name(ident: str) -> str:
    from ..core import registry

    for space in ("VIEW_3D", "IMAGE_EDITOR"):
        tool = registry.tool(space, ident)
        if tool is not None:
            return str(getattr(tool, "label", ident) or ident)
    return ident


def item_label(item: K.KeyMapItem) -> str:
    cls = ops.get(item.op)
    text = cls.label if cls is not None else item.op
    if item.props:
        parts = [part for part in (_prop_text(cls, k, v) for k, v in item.props.items()) if part]
        if parts:
            text += "（%s）" % "，".join(parts)
    return text


def set_keys(item: K.KeyMapItem, type_: str, ctrl: bool = False, shift: bool = False, alt: bool = False) -> None:
    """把一条键位改成这个键（按下的方式按新键能产生的来：键盘没有拖动、单击、双击，滚轮只有滚动）。"""
    if type_ in (K.WHEELUP, K.WHEELDOWN):
        value = K.WHEEL
    elif type_.endswith("MOUSE"):
        value = K.PRESS if item.value == K.WHEEL else item.value
    else:
        value = K.PRESS if item.value in _MOUSE_ONLY else item.value
    item.type, item.value = type_, value
    item.ctrl, item.shift, item.alt = bool(ctrl), bool(shift), bool(alt)
    item.any_mod = False if (ctrl or shift or alt) else item.any_mod


class KeymapWidget(QWidget):
    """键位列表。changed 在每次改动之后发出。"""

    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app
        self._recording: QTreeWidgetItem | None = None
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.px(8))
        top = QHBoxLayout()
        top.setSpacing(theme.px(8))
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索操作或快捷键，比如「保存」「Ctrl+S」「雕刻」")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _text: self._apply_filter())
        top.addWidget(self.search, 1)
        self.only_modified = QCheckBox("只看改过的")
        self.only_modified.toggled.connect(lambda _on: self._apply_filter())
        top.addWidget(self.only_modified)
        box.addLayout(top)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["操作", "快捷键", ""])
        self.tree.setUniformRowHeights(True)
        self.tree.setAlternatingRowColors(False)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Interactive)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self.tree.setColumnWidth(1, theme.px(200))
        self.tree.setColumnWidth(2, theme.px(56))
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.installEventFilter(self)
        self.tree.viewport().installEventFilter(self)
        box.addWidget(self.tree, 1)
        self.status = QLabel("点快捷键那一格，再按新的键（可以带 Ctrl、Shift、Alt；鼠标键、滚轮也行）。勾掉前面的框是停用。")
        self.status.setWordWrap(True)
        self.status.setProperty("role", "dim")
        box.addWidget(self.status)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.reset_all_button = QPushButton("全部恢复出厂")
        self.reset_all_button.clicked.connect(self._reset_all)
        bottom.addWidget(self.reset_all_button)
        box.addLayout(bottom)
        self.rows: list[QTreeWidgetItem] = []
        self.fill()

    # ---------------------------------------------------------------- 列表
    def fill(self) -> None:
        self._recording = None
        self.tree.blockSignals(True)
        self.tree.clear()
        self.rows = []
        bold = QFont(self.tree.font())
        bold.setBold(True)
        for name, km in self.app.keyconfig.keymaps.items():
            group = QTreeWidgetItem([K.keymap_title(name), "", ""])
            group.setFont(0, bold)
            group.setFlags(Qt.ItemIsEnabled)
            for item in km.items:
                if ops.get(item.op) is None:
                    continue
                row = QTreeWidgetItem([item_label(item), "", ""])
                row.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                row.setData(0, ROLE, (name, item))
                group.addChild(row)
                self.rows.append(row)
                self._show(row)
            self.tree.addTopLevelItem(group)
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self._refresh_conflicts()
        self._apply_filter()

    def _show(self, row: QTreeWidgetItem) -> None:
        name, item = row.data(0, ROLE)
        self.tree.blockSignals(True)
        row.setCheckState(0, Qt.Checked if item.active else Qt.Unchecked)
        row.setText(1, item.shortcut_text() if item.type else "（没有）")
        modified = K.is_modified(item)
        row.setText(2, RESET_TEXT if modified else "")
        row.setToolTip(2, "恢复成出厂的「%s」" % _default_text(item) if modified else "")
        row.setForeground(2, QBrush(theme.qcolor("accent")))
        self.tree.blockSignals(False)

    def _refresh_conflicts(self) -> None:
        kc = self.app.keyconfig
        warn = QBrush(theme.qcolor("warn"))
        normal = QBrush(theme.qcolor("text"))
        dim = QBrush(theme.qcolor("text.dim"))
        for row in self.rows:
            name, item = row.data(0, ROLE)
            found = K.conflicts(kc, name, item)
            row.setForeground(1, warn if found else (normal if item.active else dim))
            row.setForeground(0, normal if item.active else dim)
            row.setToolTip(1, _conflict_text(name, item, found) if found else "点一下，再按新的快捷键（Esc 取消）")

    def _apply_filter(self) -> None:
        text = self.search.text().strip().lower()
        only = self.only_modified.isChecked()
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            title = group.text(0).lower()
            visible = 0
            for child_index in range(group.childCount()):
                row = group.child(child_index)
                _name, item = row.data(0, ROLE)
                hay = (row.text(0) + " " + item.shortcut_text() + " " + title + " " + item.op).lower()
                show = (not text or text in hay) and (not only or K.is_modified(item))
                row.setHidden(not show)
                visible += int(show)
            group.setHidden(visible == 0)

    # ---------------------------------------------------------------- 改
    def _on_clicked(self, row: QTreeWidgetItem, column: int) -> None:
        data = row.data(0, ROLE)
        if data is None or self._recording is not None:
            return
        _name, item = data
        if column == 1:
            self.start_recording(row)
        elif column == 2 and K.is_modified(item):
            K.reset_item(item)
            self._after_change(row, "「%s」恢复成出厂的 %s" % (item_label(item), item.shortcut_text()))

    def _on_item_changed(self, row: QTreeWidgetItem, column: int) -> None:
        data = row.data(0, ROLE)
        if data is None or column != 0:
            return
        _name, item = data
        active = row.checkState(0) == Qt.Checked
        if active != item.active:
            item.active = active
            self._after_change(row, "「%s」%s" % (item_label(item), "启用了" if active else "停用了"))

    def start_recording(self, row: QTreeWidgetItem) -> None:
        self._recording = row
        self.tree.blockSignals(True)
        row.setText(1, "按下新的快捷键…")
        self.tree.blockSignals(False)
        self.tree.setFocus(Qt.OtherFocusReason)
        self.status.setText("按下新的快捷键（可以带 Ctrl、Shift、Alt；鼠标键在列表里点、滚轮在列表里滚）。Esc 取消。")

    def cancel_recording(self) -> None:
        row, self._recording = self._recording, None
        if row is not None:
            self._show(row)
            self.status.setText("没有改。")

    def record(self, type_: str, ctrl: bool = False, shift: bool = False, alt: bool = False) -> None:
        """把正在录的那条改成这个键（界面上按键、测试都走这里）。"""
        row, self._recording = self._recording, None
        if row is None:
            return
        name, item = row.data(0, ROLE)
        set_keys(item, type_, ctrl, shift, alt)
        found = K.conflicts(self.app.keyconfig, name, item)
        message = "「%s」改成了 %s。" % (item_label(item), item.shortcut_text())
        if found:
            message += _conflict_text(name, item, found)
        self._after_change(row, message)

    def _after_change(self, row: QTreeWidgetItem, message: str) -> None:
        self._show(row)
        self._refresh_conflicts()
        self._apply_filter()
        self.status.setText(message)
        self.app.save_keymap_overrides()
        self.changed.emit()

    def _reset_all(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("恢复出厂键位")
        box.setIcon(QMessageBox.Question)
        box.setText("把全部键位恢复成出厂的样子？改过的都会丢掉。")
        yes = box.addButton("全部恢复", QMessageBox.AcceptRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is yes:
            self.reset_all()

    def reset_all(self) -> None:
        K.reset_all(self.app.keyconfig)
        self.app.save_keymap_overrides()
        self.fill()
        self.status.setText("全部键位恢复成出厂的了。")
        self.changed.emit()

    # ---------------------------------------------------------------- 录键
    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if self._recording is None:
            return False
        kind = event.type()
        if kind == QEvent.KeyPress:
            if int(event.key()) in _MODIFIER_KEYS:
                return True                                  # 等组合键里的主键
            if event.key() == Qt.Key_Escape and not (event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier |
                                                                           Qt.AltModifier)):
                self.cancel_recording()
                return True
            translated = key_event(event)
            if translated is not None:
                self.record(translated.type, translated.ctrl, translated.shift, translated.alt)
            return True
        if kind == QEvent.KeyRelease:
            return True
        if obj is self.tree.viewport() and kind in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick):
            name = mouse_button_name(event.button())
            if name:
                self.record(name, *modifier_state(event.modifiers()))
            return True
        if obj is self.tree.viewport() and kind == QEvent.MouseButtonRelease:
            return True
        if obj is self.tree.viewport() and kind == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta:
                self.record(K.WHEELUP if delta > 0 else K.WHEELDOWN, *modifier_state(event.modifiers()))
            return True
        return False


def _default_text(item: K.KeyMapItem) -> str:
    if item.default is None:
        return item.shortcut_text()
    probe = K.KeyMapItem(item.op, item.default[0])
    for name, value in zip(K.KEY_FIELDS, item.default):
        setattr(probe, name, value)
    return probe.shortcut_text() + ("" if probe.active else "（停用）")


def _conflict_text(name: str, item: K.KeyMapItem, found: list) -> str:
    parts = []
    for other_name, other in found[:4]:
        where = K.keymap_title(other_name)
        if other_name == name:
            parts.append("同一处的「%s」" % item_label(other))
        elif name in K.GLOBAL_KEYMAPS:
            parts.append("「%s」里会先触发「%s」" % (where, item_label(other)))
        else:
            parts.append("会盖过%s的「%s」" % (where, item_label(other)))
    more = "等 %d 条" % len(found) if len(found) > 4 else ""
    return "同一个键还有：" + "；".join(parts) + more + "。同一处只有先找到的那条起作用。"
