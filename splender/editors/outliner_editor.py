"""大纲视图（照 Blender）：场景里的全部物体排成一棵树，模型下面挂着它的材质（纹理集）。

点一下选中（Ctrl 加选或减选，Shift 连选一段），双击或 F2 改名，眼睛开关在视口里显示或隐藏，
右键是物体菜单；A 全选、Alt+A 全不选、X 删除、H 隐藏、Alt+H 全部显示、. 在视口里找到当前物体。
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QLineEdit, QScrollArea, QVBoxLayout, QWidget

from ..core import ops, registry
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import BoolProperty, EnumProperty, IntProperty, StringProperty
from ..ui import icons, theme
from ..ui.editor import Editor
from ..ui.widgets import build_menu

KIND_ICONS = {"MESH": "object.mesh", "CAMERA": "object.camera", "LIGHT": "object.light", "EMPTY": "object.empty"}
KIND_ORDER = {"MESH": 0, "CAMERA": 1, "LIGHT": 2, "EMPTY": 3}


class _Row:
    __slots__ = ("kind", "obj", "ts", "depth", "label", "icon")

    def __init__(self, kind: str, obj=None, ts=None, depth: int = 0, label: str = "", icon: str = "") -> None:
        self.kind = kind            # "SCENE" 场景集合 / "OBJECT" 物体 / "MATERIAL" 材质
        self.obj = obj
        self.ts = ts
        self.depth = depth
        self.label = label
        self.icon = icon


class OutlinerTree(QWidget):
    def __init__(self, editor: "OutlinerEditor") -> None:
        super().__init__()
        self.editor = editor
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self._hover = -1
        self._anchor = None              # Shift 连选的起点
        self._rename: QLineEdit | None = None

    # ---- 数据 ----
    def project(self):
        return self.editor.app.project

    def rows(self) -> list[_Row]:
        project = self.project()
        if project is None:
            return []
        out = [_Row("SCENE", depth=0, label="场景集合", icon="collection")]
        if "scene" in self.editor.collapsed:
            return out
        objects = sorted(project.all_objects(), key=lambda o: (KIND_ORDER.get(o.kind, 9), o.name.lower()))
        if self.editor.sort_mode == "ORDER":
            objects = project.all_objects()
        query = self.editor.filter_text.strip().lower()
        for obj in objects:
            if query and query not in obj.name.lower():
                continue
            if self.editor.filter_kind != "ALL" and obj.kind != self.editor.filter_kind:
                continue
            out.append(_Row("OBJECT", obj=obj, depth=1, label=obj.name, icon=KIND_ICONS.get(obj.kind, "object.mesh")))
            if obj.kind == "MESH" and obj.uid not in self.editor.collapsed:
                for set_uid in obj.material_sets:
                    ts = project.texture_set(set_uid)
                    if ts is not None:
                        out.append(_Row("MATERIAL", obj=obj, ts=ts, depth=2, label=ts.name, icon="material"))
        return out

    # ---- 几何 ----
    def row_height(self) -> int:
        return theme.px(24)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(theme.px(220), self.row_height() * len(self.rows()) + theme.px(8))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(theme.px(140), self.row_height() * max(1, len(self.rows())) + theme.px(8))

    def row_rect(self, index: int) -> QRect:
        pad = theme.px(4)
        return QRect(pad, pad + index * self.row_height(), self.width() - 2 * pad, self.row_height())

    def row_at(self, pos: QPoint) -> int:
        index = (pos.y() - theme.px(4)) // self.row_height()
        return int(index) if 0 <= index < len(self.rows()) else -1

    def zones(self, rect: QRect, row: _Row) -> dict:
        h = rect.height()
        indent = theme.px(16)
        arrow = QRect(rect.left() + row.depth * indent, rect.top(), theme.px(16), h)
        icon = QRect(arrow.right() + theme.px(2), rect.top(), theme.px(20), h)
        eye = QRect(rect.right() - theme.px(26), rect.top(), theme.px(24), h)
        name = QRect(icon.right() + theme.px(4), rect.top(), max(10, eye.left() - icon.right() - theme.px(8)), h)
        return {"arrow": arrow, "icon": icon, "name": name, "eye": eye}

    def _expandable(self, row: _Row) -> bool:
        return row.kind == "SCENE" or (row.kind == "OBJECT" and row.obj.kind == "MESH" and bool(row.obj.material_sets))

    def _expanded(self, row: _Row) -> bool:
        key = "scene" if row.kind == "SCENE" else row.obj.uid
        return key not in self.editor.collapsed

    # ---- 画 ----
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), theme.qcolor("bg.panel"))
        project = self.project()
        rows = self.rows()
        if project is None:
            painter.setPen(theme.qcolor("text.faint"))
            painter.drawText(self.rect(), Qt.AlignCenter, "没有工程")
            return
        dpr = self.devicePixelRatioF()
        side = theme.size("icon")
        radius = theme.size("radius")
        active_uid = project.active_object_uid
        active_set = project.active_set_uid
        painter.setFont(theme.font())
        metrics = painter.fontMetrics()
        for index, row in enumerate(rows):
            rect = self.row_rect(index)
            selected = row.kind == "OBJECT" and row.obj.select
            active = row.kind == "OBJECT" and row.obj.uid == active_uid
            if index % 2 == 1:
                painter.fillRect(rect, theme.qcolor("bg.row_alt"))
            if selected or active:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.qcolor("accent.soft" if active else "accent.softer"))
                painter.drawRoundedRect(QRectF(rect), radius, radius)
            elif index == self._hover:
                painter.setPen(Qt.NoPen)
                painter.setBrush(theme.qcolor("bg.row_hover"))
                painter.drawRoundedRect(QRectF(rect), radius, radius)
            z = self.zones(rect, row)
            if self._expandable(row):
                pm = icons.pixmap("chevron.down" if self._expanded(row) else "chevron.right", theme.color("text.dim"),
                                  theme.px(12), dpr)
                painter.drawPixmap(z["arrow"].center() - QPoint(theme.px(6), theme.px(6)), pm)
            visible = row.obj.visible if row.obj is not None else True
            if row.kind == "OBJECT":
                color = "accent" if active else ("text" if visible else "text.faint")
            elif row.kind == "MATERIAL":
                color = "accent" if row.ts.uid == active_set else "text.dim"
            else:
                color = "text.dim"
            pm = icons.pixmap(row.icon, theme.color(color), side, dpr)
            painter.drawPixmap(z["icon"].center() - QPoint(side // 2, side // 2), pm)
            painter.setPen(theme.qcolor("text" if visible else "text.disabled"))
            font = theme.font()
            font.setBold(active)
            painter.setFont(font)
            text = metrics.elidedText(row.label, Qt.ElideRight, z["name"].width())
            painter.drawText(z["name"], Qt.AlignVCenter | Qt.AlignLeft, text)
            if row.kind == "OBJECT":
                pm = icons.pixmap("visible" if visible else "hidden",
                                  theme.color("text.dim" if visible else "text.faint"), side, dpr)
                painter.drawPixmap(z["eye"].center() - QPoint(side // 2, side // 2), pm)
        painter.end()

    # ---- 鼠标 ----
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        row = self.row_at(event.position().toPoint())
        if row != self._hover:
            self._hover = row
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = -1
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        index = self.row_at(pos)
        rows = self.rows()
        if event.button() == Qt.RightButton:
            if index >= 0 and rows[index].kind == "OBJECT" and not rows[index].obj.select:
                self._click(rows, index, Qt.NoModifier)
            self.editor.open_context_menu(event.globalPosition().toPoint())
            return
        if event.button() != Qt.LeftButton:
            return
        if index < 0:
            ops.call("outliner.select_all", self.editor.context(), action="DESELECT")
            return
        row = rows[index]
        z = self.zones(self.row_rect(index), row)
        if self._expandable(row) and z["arrow"].contains(pos):
            key = "scene" if row.kind == "SCENE" else row.obj.uid
            self.editor.collapsed.symmetric_difference_update({key})
            self.updateGeometry()
            self.update()
            return
        if row.kind == "OBJECT" and z["eye"].contains(pos):
            self.editor.toggle_visibility(row.obj, solo=bool(event.modifiers() & Qt.ControlModifier))
            return
        self._click(rows, index, event.modifiers())

    def _click(self, rows, index: int, modifiers) -> None:
        row = rows[index]
        ctx = self.editor.context()
        if row.kind == "SCENE":
            return
        if row.kind == "MATERIAL":
            project = self.project()
            project.set_active_set(row.ts)
            ops.call("outliner.item_activate", ctx, uid=row.obj.uid, extend=False)
            self.update()
            return
        if modifiers & Qt.ShiftModifier and self._anchor is not None:
            uids = [r.obj.uid for r in rows if r.kind == "OBJECT"]
            if self._anchor in uids:
                a, b = sorted((uids.index(self._anchor), uids.index(row.obj.uid)))
                ops.call("outliner.item_activate", ctx, uid=row.obj.uid, extend=bool(modifiers & Qt.ControlModifier),
                         range_uids=",".join(str(u) for u in uids[a:b + 1]))
                return
        ops.call("outliner.item_activate", ctx, uid=row.obj.uid, extend=bool(modifiers & Qt.ControlModifier))
        self._anchor = row.obj.uid

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        pos = event.position().toPoint()
        index = self.row_at(pos)
        if index < 0:
            return
        row = self.rows()[index]
        if row.kind in ("OBJECT", "MATERIAL") and self.zones(self.row_rect(index), row)["name"].contains(pos):
            self.start_rename(index)

    def start_rename(self, index: int) -> None:
        rows = self.rows()
        if not 0 <= index < len(rows):
            return
        row = rows[index]
        target = row.obj if row.kind == "OBJECT" else row.ts
        if target is None:
            return
        rect = self.zones(self.row_rect(index), row)["name"]
        edit = QLineEdit(target.name, self)
        edit.setGeometry(rect.adjusted(-theme.px(4), theme.px(1), 0, -theme.px(1)))
        edit.selectAll()
        edit.show()
        edit.setFocus()
        self._rename = edit

        def commit() -> None:
            if self._rename is not edit:
                return
            self._rename = None
            text = edit.text().strip()
            edit.deleteLater()
            if text and text != target.name:
                rename_with_undo(self.editor.app, target, text)
            self.update()

        edit.editingFinished.connect(commit)

    def rename_active(self) -> bool:
        project = self.project()
        if project is None:
            return False
        for index, row in enumerate(self.rows()):
            if row.kind == "OBJECT" and row.obj.uid == project.active_object_uid:
                self.start_rename(index)
                return True
        return False


def rename_with_undo(app, target, text: str) -> None:
    """改物体或纹理集的名字，记一步撤销。"""
    old = target.name

    def put(value: str) -> None:
        target.name = value
        if app.project is not None:
            app.project.mark_dirty()
            app.project.changed.emit("objects")
        app.notify("objects")

    put(text)
    if app.history is not None:
        app.history.push("重命名", lambda: put(old), lambda: put(text))


@registry.register_editor
class OutlinerEditor(Editor):
    idname = "OUTLINER"
    label = "大纲视图"
    icon = "editor.outliner"
    category = "通用"
    order = 25
    description = "场景里的全部物体：选择、显示隐藏、改名"
    keymaps = ["Outliner"]

    def __init__(self, app, area=None) -> None:
        self.collapsed: set = set()
        self.filter_text = ""
        self.filter_kind = "ALL"
        self.sort_mode = "ALPHA"
        super().__init__(app, area)

    def build_main(self):
        root = QWidget()
        box = QVBoxLayout(root)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        self.tree = OutlinerTree(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.tree)
        box.addWidget(scroll, 1)
        return root

    def draw_header(self, layout, ctx) -> None:
        layout.popover("", "filter", self._draw_filter)
        layout.stretch()
        project = self.app.project
        if project is not None:
            count = len(project.all_objects())
            chosen = len(project.selected_objects())
            layout.label("%d / %d" % (chosen, count), role="dim")

    def _draw_filter(self, layout, ctx) -> None:
        layout.label("筛选", role="title")
        layout.operator("outliner.filter_set", text="全部", kind="ALL")
        layout.operator("outliner.filter_set", text="只看网格", kind="MESH")
        layout.operator("outliner.filter_set", text="只看摄像机", kind="CAMERA")
        layout.operator("outliner.filter_set", text="只看灯光", kind="LIGHT")
        layout.operator("outliner.filter_set", text="只看空物体", kind="EMPTY")
        layout.separator()
        layout.operator("outliner.sort_set", text="按名字排", mode="ALPHA")
        layout.operator("outliner.sort_set", text="按添加顺序排", mode="ORDER")

    def toggle_visibility(self, obj, solo: bool = False) -> None:
        ctx = self.context()
        ops.call("outliner.toggle_visible", ctx, uid=obj.uid, solo=solo)

    def open_context_menu(self, global_pos) -> None:
        menu = build_menu("OUTLINER_MT_object", self.context(), self)
        menu.popup(global_pos)
        self._menu = menu

    def refresh(self) -> None:
        super().refresh()
        self.tree.updateGeometry()
        self.tree.update()

    def save_state(self) -> dict:
        state = super().save_state()
        state["collapsed"] = [str(v) for v in self.collapsed]
        state["filter_kind"] = self.filter_kind
        state["sort_mode"] = self.sort_mode
        return state

    def load_state(self, state: dict) -> None:
        super().load_state(state)
        state = state or {}
        collapsed = set()
        for value in state.get("collapsed", []):
            collapsed.add(int(value) if str(value).isdigit() else str(value))
        self.collapsed = collapsed
        self.filter_kind = state.get("filter_kind", "ALL")
        self.sort_mode = state.get("sort_mode", "ALPHA")


# ====================================================================== 大纲视图的操作
def _outliner(ctx):
    editor = ctx.editor
    if getattr(editor, "idname", "") == "OUTLINER":
        return editor
    return None


class _OutlinerOp(Operator):
    searchable = False

    @classmethod
    def poll(cls, ctx) -> bool:
        return getattr(ctx.app, "project", None) is not None


@ops.register
class OutlinerItemActivate(_OutlinerOp):
    idname = "outliner.item_activate"
    label = "在大纲里选择"
    uid = IntProperty("物体", default=0)
    extend = BoolProperty("加选", default=False)
    range_uids = StringProperty("连选", default="")

    def execute(self, ctx) -> str:
        from ..ops.object_ops import record_selection, select_only, selection_state, set_active

        project = ctx.app.project
        obj = project.object_by_uid(int(self.uid))
        if obj is None:
            return CANCELLED
        before = selection_state(project)
        if self.range_uids:
            uids = {int(v) for v in self.range_uids.split(",") if v.strip().isdigit()}
            chosen = [o for o in project.all_objects() if o.uid in uids]
            if self.extend:
                for item in chosen:
                    item.select = True
                set_active(project, obj)
                project.selection_changed()
            else:
                select_only(project, chosen, obj)
        elif self.extend:
            if obj.select and project.active_object_uid == obj.uid:
                obj.select = False
            else:
                obj.select = True
                set_active(project, obj)
            project.selection_changed()
        else:
            select_only(project, [obj], obj)
        record_selection(ctx, "选择", before)
        ctx.app.request_frame()
        ctx.app.notify("selection")
        return FINISHED


@ops.register
class OutlinerSelectAll(_OutlinerOp):
    idname = "outliner.select_all"
    label = "全选"
    action = EnumProperty("动作", items=[("TOGGLE", "切换", ""), ("SELECT", "全选", ""), ("DESELECT", "全不选", ""),
                                       ("INVERT", "反选", "")], default="TOGGLE")

    def execute(self, ctx) -> str:
        from ..ops.object_ops import record_selection, selection_state

        project = ctx.app.project
        before = selection_state(project)
        objects = project.all_objects()
        action = self.action
        if action == "TOGGLE":
            action = "DESELECT" if any(o.select for o in objects) else "SELECT"
        for obj in objects:
            obj.select = {"SELECT": True, "DESELECT": False}.get(action, not obj.select)
        project.selection_changed()
        record_selection(ctx, "全选", before)
        ctx.app.request_frame()
        ctx.app.notify("selection")
        return FINISHED


@ops.register
class OutlinerToggleVisible(_OutlinerOp):
    idname = "outliner.toggle_visible"
    label = "显示或隐藏"
    uid = IntProperty("物体", default=0)
    solo = BoolProperty("只看它", default=False, description="Ctrl+点眼睛：只显示这一个")

    def execute(self, ctx) -> str:
        from ..ops.object_ops import set_visibility

        project = ctx.app.project
        obj = project.object_by_uid(int(self.uid))
        if obj is None:
            return CANCELLED
        if self.solo:
            others = [o for o in project.all_objects() if o is not obj and o.visible]
            if others:
                set_visibility(ctx, others, False, "只显示这一个")
            if not obj.visible:
                set_visibility(ctx, [obj], True, "显示")
            return FINISHED
        set_visibility(ctx, [obj], not obj.visible, "隐藏" if obj.visible else "显示")
        return FINISHED


@ops.register
class OutlinerDelete(_OutlinerOp):
    idname = "outliner.delete"
    label = "删除"
    icon = "remove"

    @classmethod
    def poll(cls, ctx) -> bool:
        project = getattr(ctx.app, "project", None)
        return project is not None and bool(project.selected_objects())

    def execute(self, ctx) -> str:
        from ..ops import object_ops

        return object_ops.ObjectDelete.execute(object_ops.ObjectDelete(), ctx)


@ops.register
class OutlinerHide(_OutlinerOp):
    idname = "outliner.hide"
    label = "隐藏所选"

    def execute(self, ctx) -> str:
        from ..ops.object_ops import set_visibility

        project = ctx.app.project
        targets = [o for o in project.all_objects() if o.select and o.visible]
        if not targets:
            return CANCELLED
        set_visibility(ctx, targets, False, "隐藏")
        return FINISHED


@ops.register
class OutlinerUnhideAll(_OutlinerOp):
    idname = "outliner.unhide_all"
    label = "全部显示"

    def execute(self, ctx) -> str:
        from ..ops.object_ops import set_visibility

        project = ctx.app.project
        targets = [o for o in project.all_objects() if not o.visible]
        if not targets:
            return CANCELLED
        set_visibility(ctx, targets, True, "全部显示")
        return FINISHED


@ops.register
class OutlinerShowActive(_OutlinerOp):
    idname = "outliner.show_active"
    label = "在视口里查看"
    description = "把视口对准当前物体"

    def execute(self, ctx) -> str:
        project = ctx.app.project
        engine = getattr(ctx.app, "engine", None)
        view_editor = getattr(ctx.app, "active_view3d", None)
        obj = project.active_object
        if obj is None or engine is None or view_editor is None or view_editor.view is None:
            return CANCELLED
        ctx.app.host.make_current()
        engine.frame_objects(view_editor.view, [obj])
        view_editor.camera_changed()
        return FINISHED


@ops.register
class OutlinerRename(_OutlinerOp):
    idname = "outliner.rename"
    label = "重命名"

    def execute(self, ctx) -> str:
        editor = _outliner(ctx)
        if editor is None:
            return CANCELLED
        return FINISHED if editor.tree.rename_active() else CANCELLED


@ops.register
class OutlinerFilterSet(_OutlinerOp):
    idname = "outliner.filter_set"
    label = "筛选"
    kind = EnumProperty("类型", items=[("ALL", "全部", ""), ("MESH", "网格", ""), ("CAMERA", "摄像机", ""),
                                     ("LIGHT", "灯光", ""), ("EMPTY", "空物体", "")], default="ALL")

    def execute(self, ctx) -> str:
        editor = _outliner(ctx)
        if editor is None:
            return CANCELLED
        editor.filter_kind = self.kind
        editor.refresh()
        return FINISHED


@ops.register
class OutlinerSortSet(_OutlinerOp):
    idname = "outliner.sort_set"
    label = "排序"
    mode = EnumProperty("排序", items=[("ALPHA", "按名字", ""), ("ORDER", "按添加顺序", "")], default="ALPHA")

    def execute(self, ctx) -> str:
        editor = _outliner(ctx)
        if editor is None:
            return CANCELLED
        editor.sort_mode = self.mode
        editor.refresh()
        return FINISHED


@registry.register_menu
class OutlinerObjectMenu(registry.Menu):
    idname = "OUTLINER_MT_object"
    label = "物体"

    def draw(self, layout, ctx) -> None:
        layout.operator("outliner.select_all", text="全选", action="SELECT")
        layout.operator("outliner.select_all", text="全不选", action="DESELECT")
        layout.separator()
        layout.operator("outliner.rename", text="重命名", icon="pencil")
        layout.operator("object.duplicate_move", text="复制", icon="duplicate")
        layout.operator("outliner.delete", text="删除", icon="remove")
        layout.separator()
        layout.operator("outliner.hide", text="隐藏所选", icon="hidden")
        layout.operator("outliner.unhide_all", text="全部显示", icon="visible")
        layout.separator()
        layout.operator("outliner.show_active", text="在视口里查看", icon="focus")
