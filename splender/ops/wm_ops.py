"""窗口级的通用操作（照 Blender 的 wm.*、screen.*）：弹菜单、弹饼菜单、开关一个设置、换工具、改名、
换编辑器、切换模式、重复上一步、调整上一步。"""
from __future__ import annotations

import logging

from ..core import ops, registry
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import BoolProperty, EnumProperty, StringProperty

log = logging.getLogger("splender.ops.wm")

_open_menus: set = set()


def _cursor_pos():
    from PySide6.QtGui import QCursor

    return QCursor.pos()


def resolve_data_path(ctx, path: str):
    """「tool_settings.use_snap」「space_data.overlay.show_overlays」这种路径 → (对象, 属性名)。找不到返回 (None, "")。"""
    parts = [p for p in str(path or "").split(".") if p]
    if len(parts) < 2:
        return None, ""
    head = parts[0]
    roots = {
        "tool_settings": lambda: getattr(ctx.app, "tool_settings", None),
        "space_data": lambda: ctx.editor if ctx.editor is not None else getattr(ctx.app, "active_view3d", None),
        "preferences": lambda: getattr(ctx.app, "prefs", None),
        "scene": lambda: getattr(ctx.app, "project", None),
    }
    getter = roots.get(head)
    target = getter() if getter is not None else None
    for name in parts[1:-1]:
        if target is None:
            return None, ""
        target = getattr(target, name, None)
    if target is None or not hasattr(target, parts[-1]):
        return None, ""
    return target, parts[-1]


@ops.register
class WMCallMenu(Operator):
    idname = "wm.call_menu"
    label = "弹出菜单"
    searchable = False
    name = StringProperty("菜单", default="")

    @classmethod
    def poll(cls, ctx) -> bool:
        return True

    def invoke(self, ctx, event) -> str:
        from ..ui.widgets import build_menu

        if registry.menu(self.name) is None:
            return CANCELLED
        editor = ctx.editor
        if editor is not None and hasattr(editor, "cancel_press"):
            editor.cancel_press()
        menu = build_menu(self.name, ctx)
        _open_menus.add(menu)
        menu.aboutToHide.connect(lambda m=menu: _open_menus.discard(m))
        menu.popup(_cursor_pos())
        return FINISHED

    def execute(self, ctx) -> str:
        return self.invoke(ctx, None)


@ops.register
class WMCallMenuPie(Operator):
    idname = "wm.call_menu_pie"
    label = "弹出饼菜单"
    searchable = False
    name = StringProperty("菜单", default="")

    @classmethod
    def poll(cls, ctx) -> bool:
        return True

    def invoke(self, ctx, event) -> str:
        from ..ui.pie import qt_key_of, show_pie

        if registry.menu(self.name) is None:
            return CANCELLED
        hold = qt_key_of(event.type) if event is not None and getattr(event, "type", "") else None
        pie = show_pie(self.name, ctx, hold, _cursor_pos())
        if pie is not None:
            _open_menus.add(pie)
            pie.destroyed.connect(lambda *_a, p=pie: _open_menus.discard(p))
        return FINISHED if pie is not None else CANCELLED

    def execute(self, ctx) -> str:
        return self.invoke(ctx, None)


@ops.register
class WMContextToggle(Operator):
    idname = "wm.context_toggle"
    label = "开关"
    searchable = False
    data_path = StringProperty("路径", default="")

    @classmethod
    def poll(cls, ctx) -> bool:
        return True

    def execute(self, ctx) -> str:
        target, attr = resolve_data_path(ctx, self.data_path)
        if target is None:
            return CANCELLED
        value = not bool(getattr(target, attr))
        setattr(target, attr, value)
        prop = None
        props = getattr(type(target), "properties", None)
        if callable(props):
            prop = props().get(attr)
        label = getattr(prop, "label", attr) if prop is not None else attr
        self.report(ctx, "%s：%s" % (label, "开" if value else "关"))
        ctx.app.request_frame()
        ctx.app.notify("settings")
        return FINISHED


@ops.register
class WMContextSetEnum(Operator):
    idname = "wm.context_set_enum"
    label = "设置"
    searchable = False
    data_path = StringProperty("路径", default="")
    value = StringProperty("值", default="")

    @classmethod
    def poll(cls, ctx) -> bool:
        return True

    def execute(self, ctx) -> str:
        target, attr = resolve_data_path(ctx, self.data_path)
        if target is None:
            return CANCELLED
        try:
            setattr(target, attr, self.value)
        except Exception:  # noqa: BLE001
            log.exception("设置 %s 失败", self.data_path)
            return CANCELLED
        ctx.app.request_frame()
        ctx.app.notify("settings")
        return FINISHED


def _mode_tools(app, space: str = "VIEW_3D") -> list:
    mode = app.tool_settings.mode
    return [tool for tool in registry.tools(space) if getattr(tool, "mode", mode) == mode]


@ops.register
class WMToolSetCycle(Operator):
    idname = "wm.tool_set_cycle"
    label = "轮换工具"
    description = "在同一组工具之间轮换（比如几种选择工具）"
    searchable = False
    group = StringProperty("组", default="select")

    @classmethod
    def poll(cls, ctx) -> bool:
        return any(getattr(t, "group", "") for t in _mode_tools(ctx.app))

    def execute(self, ctx) -> str:
        tools = [t for t in _mode_tools(ctx.app) if getattr(t, "group", "") == self.group]
        if not tools:
            return CANCELLED
        names = [t.idname for t in tools]
        current = ctx.app.tool_settings.tool
        index = (names.index(current) + 1) % len(names) if current in names else 0
        ctx.app.tool_settings.tool = names[index]
        self.report(ctx, "工具：%s" % tools[index].label)
        return FINISHED


@ops.register
class WMToolbar(Operator):
    idname = "wm.toolbar"
    label = "工具"
    description = "在鼠标处列出当前模式的工具"
    searchable = False

    @classmethod
    def poll(cls, ctx) -> bool:
        return bool(_mode_tools(ctx.app))

    def invoke(self, ctx, event) -> str:
        from PySide6.QtWidgets import QMenu

        from ..ui import icons
        from ..ui.editor import _tool_shortcut

        menu = QMenu()
        current = ctx.app.tool_settings.tool
        for tool in _mode_tools(ctx.app):
            shortcut = _tool_shortcut(ctx.app, tool.idname)
            action = menu.addAction(icons.icon(tool.icon or "tool.brush"),
                                    tool.label + ("\t" + shortcut if shortcut else ""))
            action.setCheckable(True)
            action.setChecked(tool.idname == current)
            action.triggered.connect(lambda _c=False, name=tool.idname: setattr(ctx.app.tool_settings, "tool", name))
        _open_menus.add(menu)
        menu.aboutToHide.connect(lambda m=menu: _open_menus.discard(m))
        menu.popup(_cursor_pos())
        return FINISHED

    def execute(self, ctx) -> str:
        return self.invoke(ctx, None)


@ops.register
class WMRenameActive(Operator):
    idname = "wm.rename_active"
    label = "重命名当前项"
    description = "给当前物体（或图层编辑器里的当前图层）改名"

    @classmethod
    def poll(cls, ctx) -> bool:
        project = getattr(ctx.app, "project", None)
        if project is None:
            return False
        if getattr(ctx.editor, "idname", "") == "LAYERS":
            return ctx.layer is not None
        return project.active_object is not None

    def invoke(self, ctx, event) -> str:
        from PySide6.QtWidgets import QInputDialog

        editor = ctx.editor
        if getattr(editor, "idname", "") == "OUTLINER":
            return FINISHED if editor.tree.rename_active() else CANCELLED
        if getattr(editor, "idname", "") == "LAYERS":
            layer = ctx.layer
            rows = editor.list.rows()
            if layer in rows:
                editor.list.start_rename(rows.index(layer))
                return FINISHED
            return CANCELLED
        obj = ctx.app.project.active_object
        text, ok = QInputDialog.getText(None, "重命名", "名字", text=obj.name)
        if not ok or not text.strip() or text.strip() == obj.name:
            return CANCELLED
        from ..editors.outliner_editor import rename_with_undo

        rename_with_undo(ctx.app, obj, text.strip())
        return FINISHED


SPACE_ITEMS = [("VIEW_3D", "3D 视口", ""), ("PROPERTIES", "属性", ""), ("OUTLINER", "大纲视图", ""),
               ("IMAGE_EDITOR", "图像编辑器", ""), ("NODE_EDITOR", "节点编辑器", ""), ("LAYERS", "图层", "")]


@ops.register
class ScreenSpaceTypeSetOrCycle(Operator):
    idname = "screen.space_type_set_or_cycle"
    label = "换编辑器"
    description = "把鼠标所在的区域换成另一种编辑器"
    searchable = False
    space_type = EnumProperty("编辑器", items=SPACE_ITEMS, default="VIEW_3D")

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.area is not None and hasattr(ctx.area, "set_editor")

    def execute(self, ctx) -> str:
        if registry.editor(self.space_type) is None:
            return CANCELLED
        if ctx.area.editor_idname() == self.space_type:
            return CANCELLED
        ctx.area.set_editor(self.space_type)
        return FINISHED


# ---------------------------------------------------------------- 模式
def available_modes(app) -> list[tuple]:
    from ..doc.project import ToolSettings

    prop = ToolSettings.prop("mode")
    return list(prop.items(app.tool_settings))


@ops.register
class ObjectModeSet(Operator):
    idname = "object.mode_set"
    label = "切换模式"
    description = "切换 3D 视口的模式：物体、雕刻、纹理绘制"
    mode = EnumProperty("模式", items=[("OBJECT", "物体模式", ""), ("EDIT", "编辑模式", ""), ("SCULPT", "雕刻模式", ""),
                                     ("PAINT", "纹理绘制", "")], default="OBJECT")
    toggle = BoolProperty("来回切换", default=False, description="已经在这个模式时回到上一个模式")

    @classmethod
    def poll(cls, ctx) -> bool:
        return ctx.app.project is not None and getattr(ctx.app, "host", None) is not None

    def execute(self, ctx) -> str:
        ts = ctx.app.tool_settings
        modes = [item[0] for item in available_modes(ctx.app)]
        mode = self.mode
        if mode not in modes:
            return CANCELLED
        if self.toggle and ts.mode == mode:
            mode = getattr(ctx.app, "previous_mode", "OBJECT") or "OBJECT"
            if mode == ts.mode or mode not in modes:
                mode = "OBJECT"
        if mode == ts.mode:
            return CANCELLED
        ts.mode = mode
        return FINISHED


@ops.register
class ObjectModeToggle(Operator):
    idname = "object.mode_toggle"
    label = "切回上一个模式"
    description = "在当前模式和上一个模式之间来回切换"

    @classmethod
    def poll(cls, ctx) -> bool:
        return ObjectModeSet.poll(ctx)

    def execute(self, ctx) -> str:
        ts = ctx.app.tool_settings
        previous = getattr(ctx.app, "previous_mode", "") or ("OBJECT" if ts.mode != "OBJECT" else "PAINT")
        if previous == ts.mode:
            previous = "OBJECT" if ts.mode != "OBJECT" else "PAINT"
        ts.mode = previous
        return FINISHED


@ops.register
class View3DObjectModePie(Operator):
    idname = "view3d.object_mode_pie_or_toggle"
    label = "模式饼菜单"
    description = "在鼠标处弹出模式饼菜单"
    searchable = False

    @classmethod
    def poll(cls, ctx) -> bool:
        return ObjectModeSet.poll(ctx)

    def invoke(self, ctx, event) -> str:
        return ops.call("wm.call_menu_pie", ctx, event, name="VIEW3D_MT_object_mode_pie")

    def execute(self, ctx) -> str:
        return self.invoke(ctx, None)


# ---------------------------------------------------------------- 重复上一步、调整上一步
@ops.register
class ScreenRepeatLast(Operator):
    idname = "screen.repeat_last"
    label = "重复上一步"
    description = "用同样的设置再做一次上一个操作"

    @classmethod
    def poll(cls, ctx) -> bool:
        from ..ui import redo

        return redo.controller(ctx.app).repeat_op is not None

    def execute(self, ctx) -> str:
        from ..ui import redo

        return redo.controller(ctx.app).repeat(ctx)


@ops.register
class ScreenRedoLast(Operator):
    idname = "screen.redo_last"
    label = "调整上一步"
    description = "改上一个操作的参数（比如刚加的立方体的尺寸），结果马上更新"

    @classmethod
    def poll(cls, ctx) -> bool:
        from ..ui import redo

        return redo.controller(ctx.app).can_adjust()

    def invoke(self, ctx, event) -> str:
        from ..ui import redo

        redo.open_popover(ctx)
        return FINISHED

    def execute(self, ctx) -> str:
        return self.invoke(ctx, None)
