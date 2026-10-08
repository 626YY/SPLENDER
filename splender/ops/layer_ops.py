"""图层操作。结构性的改动都进撤销历史；被删除图层的像素由撤销步骤保管，步骤被丢弃时才真正释放。"""
from __future__ import annotations

from ..core import ops
from ..core.keymap import LEFTMOUSE, PRESS, RIGHTMOUSE
from ..core.ops import CANCELLED, FINISHED, PASS_THROUGH, RUNNING_MODAL, Operator
from ..core.props import EnumProperty, IntProperty, StringProperty
from ..doc.project import ADJUST_ITEMS, MAX_FOLDER_DEPTH, PLANE_MASK, Layer
from .view3d_ops import view3d


def _ts(ctx):
    return ctx.texture_set


def _target(ctx, uid: int):
    ts = _ts(ctx)
    if ts is None:
        return None
    return ts.layer(uid) if uid else ts.active_layer


def _engine(ctx):
    host = getattr(ctx.app, "host", None)
    if host is None:
        return None
    host.make_current()
    return host.engine


class _LayerOp(Operator):
    @classmethod
    def poll(cls, ctx) -> bool:
        return _ts(ctx) is not None


class _ActiveLayerOp(Operator):
    @classmethod
    def poll(cls, ctx) -> bool:
        ts = _ts(ctx)
        return ts is not None and ts.active_layer is not None


def _record(ctx, label: str, ts, before: tuple, added=(), removed=()) -> None:
    """结构已经改好（before 是改之前的 ts.structure()），记一步撤销。

    removed：这一步拿掉的图层，像素在这里从引擎取下、由这一步保管；
    added：这一步新加的图层，撤销时把它们的像素取下保管，重做时放回。
    步骤被丢弃时，释放不再用得到的那一份。"""
    after = ts.structure()
    held = {"removed": {}, "added": {}, "applied": True}
    engine = _engine(ctx)
    if engine is not None:
        for layer in removed:
            held["removed"][layer.uid] = engine.detach_layer_pixels(layer.uid)

    def undo() -> None:
        eng = _engine(ctx)
        if eng is not None:
            for layer in added:
                held["added"][layer.uid] = eng.detach_layer_pixels(layer.uid)
            for uid, store in held["removed"].items():
                if store is not None:
                    eng.attach_layer_pixels(uid, store)
            held["removed"] = {}
        ts.set_structure(before)
        held["applied"] = False

    def redo() -> None:
        eng = _engine(ctx)
        if eng is not None:
            for layer in removed:
                held["removed"][layer.uid] = eng.detach_layer_pixels(layer.uid)
            for uid, store in held["added"].items():
                if store is not None:
                    eng.attach_layer_pixels(uid, store)
            held["added"] = {}
        ts.set_structure(after)
        held["applied"] = True

    def dispose() -> None:
        eng = _engine(ctx)
        if eng is None:
            return
        for store in (held["removed"] if held["applied"] else held["added"]).values():
            if store is not None:
                eng.free_layer_pixels(store)

    ctx.history.push(label, undo, redo, 0, dispose)


def _room(ts, count: int) -> bool:
    from ..engine.display import MAX_LAYERS

    return len(ts.layers) + count <= MAX_LAYERS


def _subtree_depth(ts, layer) -> int:
    """layer 的子树里最深的一层比 layer 深几级。"""
    base = ts.depth(layer)
    return max((ts.depth(item) - base for item in ts.subtree(layer)), default=0)


def _new_depth_ok(ts, layer, parent_uid: int) -> bool:
    parent = ts.layer(parent_uid) if parent_uid else None
    depth = ts.depth(parent) + 1 if parent is not None else 0
    return depth + _subtree_depth(ts, layer) <= MAX_FOLDER_DEPTH


def _stop_strokes(ctx) -> None:
    engine = _engine(ctx)
    if engine is not None:
        engine.stroke_cancel()


@ops.register
class LayerAddPaint(_LayerOp):
    idname = "layer.add_paint"
    label = "新建绘制图层"
    description = "在当前图层上方新建一个可以用笔刷画的图层"
    icon = "layer.paint"

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        if not _room(ts, 1):
            self.report(ctx, "一套贴图的图层已经满了", "WARNING")
            return CANCELLED
        before = ts.structure()
        layer = ts.add_layer(Layer(name=ts.unique_name("图层 %d" % (len(ts.layers) + 1))))
        ts.paint_target = "CONTENT"
        _record(ctx, self.label, ts, before, added=[layer])
        return FINISHED


@ops.register
class LayerAddFill(_LayerOp):
    idname = "layer.add_fill"
    label = "新建填充图层"
    description = "新建一个整层统一数值的图层，颜色取当前笔刷颜色"
    icon = "layer.fill"

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        if not _room(ts, 1):
            self.report(ctx, "一套贴图的图层已经满了", "WARNING")
            return CANCELLED
        brush = ctx.app.tool_settings.brush
        before = ts.structure()
        layer = Layer(name=ts.unique_name("填充 %d" % (len(ts.layers) + 1)), kind="FILL", fill_color=brush.color,
                      fill_metallic=brush.metallic, fill_roughness=brush.roughness)
        ts.add_layer(layer)
        _record(ctx, self.label, ts, before, added=[layer])
        from .select_ops import mask_new_layer

        mask_new_layer(ctx, layer, self.label)          # 有选区时只填选区里（照 Photoshop）
        return FINISHED


@ops.register
class LayerAddFolder(_LayerOp):
    idname = "layer.add_folder"
    label = "新建文件夹"
    description = "新建一个空文件夹。把图层拖进去，整组一起调不透明度、混合方式和蒙版"
    icon = "layer.folder"

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        if not _room(ts, 1):
            self.report(ctx, "一套贴图的图层已经满了", "WARNING")
            return CANCELLED
        before = ts.structure()
        folder = Layer(name=ts.unique_name("文件夹"), kind="FOLDER")
        ts.add_layer(folder)
        if ts.depth(folder) >= MAX_FOLDER_DEPTH:
            ts.set_structure(before)
            self.report(ctx, "文件夹最多套 %d 层" % MAX_FOLDER_DEPTH, "WARNING")
            return CANCELLED
        _record(ctx, self.label, ts, before, added=[folder])
        return FINISHED


@ops.register
class LayerAddAdjust(_LayerOp):
    idname = "layer.add_adjust"
    label = "新建调整层"
    description = "调整它下面（同一个文件夹里）已经合好的结果：曲线、色阶、色相饱和度、色彩平衡、渐变映射等"
    icon = "adjust"
    adjust = EnumProperty("调整", items=ADJUST_ITEMS, default="HSV")

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        if not _room(ts, 1):
            self.report(ctx, "一套贴图的图层已经满了", "WARNING")
            return CANCELLED
        before = ts.structure()
        label = next(item[1] for item in ADJUST_ITEMS if item[0] == self.adjust)
        # 和 Photoshop 一样先只改颜色；要连金属度、粗糙度、高度一起改，在「作用于」里打开
        layer = Layer(name=ts.unique_name(label), kind="ADJUST", adjust_type=self.adjust, use_metallic=False,
                      use_roughness=False, use_height=False)
        ts.add_layer(layer)
        _record(ctx, "新建调整层「%s」" % label, ts, before, added=[layer])
        from .select_ops import mask_new_layer

        mask_new_layer(ctx, layer, "新建调整层「%s」" % label)    # 有选区时只调选区里（照 Photoshop）
        return FINISHED


def _set_parts(ctx, layer, label: str, before: str, after: str) -> None:
    """改「部件」生成器选中的部件，记一步撤销。"""
    layer.gen_parts = after
    history = getattr(ctx.app, "history", None)
    if history is not None and before != after:
        def undo(layer=layer, value=before) -> None:
            layer.gen_parts = value

        def redo(layer=layer, value=after) -> None:
            layer.gen_parts = value

        history.push(label, undo, redo)


@ops.register
class LayerPickParts(Operator):
    idname = "layer.pick_parts"
    label = "点选部件"
    description = "在视口里点模型的部件：点一下选中，再点一下取消。右键、回车或 Esc 结束"
    icon = "generator"

    @classmethod
    def poll(cls, ctx) -> bool:
        ts = ctx.texture_set
        return ts is not None and ts.active_layer is not None and view3d(ctx) is not None

    def invoke(self, ctx, event) -> str:
        ts = ctx.texture_set
        self.layer = ts.active_layer
        self.ts = ts
        self.before_generator = self.layer.mask_generator
        self.before = self.layer.gen_parts
        if self.layer.mask_generator != "PARTS":
            self.layer.mask_generator = "PARTS"
        engine = ctx.app.engine
        maps = engine.meshmap(ts.uid) if engine is not None else None
        if maps is None or not maps.part_count:
            ensure = getattr(ctx.app, "ensure_meshmaps", None)
            if ensure is not None:
                ensure(ts)
        self._hint(ctx)
        return RUNNING_MODAL

    def _hint(self, ctx) -> None:
        from ..bake.parts import parse_ids

        if ctx.wm is not None:
            count = len(parse_ids(self.layer.gen_parts))
            ctx.wm.set_hint("点选部件：已选 %d 个。左键点模型选中或取消，右键、回车或 Esc 结束" % count)

    def modal(self, ctx, event) -> str:
        from ..bake.parts import format_ids, parse_ids

        if event.value == PRESS and event.type == LEFTMOUSE:
            editor = view3d(ctx)
            if editor is None or event.source is not editor.widget:
                self._finish(ctx)
                return FINISHED
            host = ctx.app.host
            host.make_current()
            x, y = editor.pixel(event)
            found = host.engine.part_at(editor.view, x, y)
            if found is None:
                return RUNNING_MODAL
            ts, part = found
            if ts is not self.ts:
                self.report(ctx, "点到的是另一套贴图「%s」的部件" % ts.name, "WARNING")
                return RUNNING_MODAL
            if part < 0:
                self.report(ctx, "还没有部件：烘焙好模型贴图之后再点", "WARNING")
                return RUNNING_MODAL
            ids = set(parse_ids(self.layer.gen_parts))
            ids.symmetric_difference_update({part})
            self.layer.gen_parts = format_ids(ids)
            host.request_frame()
            self._hint(ctx)
            return RUNNING_MODAL
        if event.value == PRESS and event.type in (RIGHTMOUSE, "ESC", "RET", "NUMPAD_ENTER"):
            self._finish(ctx)
            return FINISHED
        return PASS_THROUGH if event.type not in (LEFTMOUSE, RIGHTMOUSE) else RUNNING_MODAL

    def _finish(self, ctx) -> None:
        if ctx.wm is not None:
            ctx.wm.set_hint("")
        after = self.layer.gen_parts
        layer = self.layer
        if after == self.before and layer.mask_generator == self.before_generator:
            return
        history = getattr(ctx.app, "history", None)
        if history is not None:
            before_values = (self.before_generator, self.before)
            after_values = (layer.mask_generator, after)

            def undo(layer=layer, values=before_values) -> None:
                layer.mask_generator, layer.gen_parts = values

            def redo(layer=layer, values=after_values) -> None:
                layer.mask_generator, layer.gen_parts = values

            history.push("点选部件", undo, redo)

    def cancel(self, ctx) -> None:
        self._finish(ctx)


@ops.register
class LayerClearParts(_LayerOp):
    idname = "layer.clear_parts"
    label = "清空部件"
    description = "「部件」生成器一个部件都不选"
    icon = "remove"

    @classmethod
    def poll(cls, ctx) -> bool:
        ts = ctx.texture_set
        return ts is not None and ts.active_layer is not None and bool(ts.active_layer.gen_parts)

    def execute(self, ctx) -> str:
        layer = ctx.texture_set.active_layer
        _set_parts(ctx, layer, "清空部件", layer.gen_parts, "")
        return FINISHED


@ops.register
class LayerApplySmartMask(_LayerOp):
    idname = "layer.apply_smart_mask"
    label = "套用智能遮罩"
    description = "把一套生成器设置套到当前图层上，蒙版按模型的形状落在边缘、凹缝、朝上的面等地方"
    icon = "generator"
    mask = StringProperty("遮罩", default="")

    @classmethod
    def poll(cls, ctx) -> bool:
        ts = ctx.texture_set
        return ts is not None and ts.active_layer is not None

    def execute(self, ctx) -> str:
        from ..bake import smart

        preset = smart.get_mask(self.mask)
        ts = _ts(ctx)
        layer = ts.active_layer if ts is not None else None
        if preset is None or layer is None:
            self.report(ctx, "没有这个智能遮罩" if preset is None else "先选一个图层", "WARNING")
            return CANCELLED
        values = dict(preset["values"])
        before = {name: getattr(layer, name) for name in values}
        for name, value in values.items():
            setattr(layer, name, value)
        history = getattr(ctx.app, "history", None)
        if history is not None:
            def undo(layer=layer, before=before) -> None:
                for name, value in before.items():
                    setattr(layer, name, value)

            def redo(layer=layer, values=values) -> None:
                for name, value in values.items():
                    setattr(layer, name, value)

            history.push("智能遮罩「%s」" % preset["name"], undo, redo)
        ensure = getattr(ctx.app, "ensure_meshmaps", None)
        if ensure is not None:
            ensure(ts)
        self.report(ctx, "已给「%s」套上智能遮罩「%s」" % (layer.name, preset["name"]))
        return FINISHED


@ops.register
class LayerAddSmartMaterial(_LayerOp):
    idname = "layer.add_smart_material"
    label = "添加智能材质"
    description = "加一个文件夹，里面是几层带生成器蒙版的填充层，按模型的形状自动落在边缘、凹陷、朝上的面等地方"
    icon = "smart_material"
    material = StringProperty("材质", default="")

    def execute(self, ctx) -> str:
        from ..bake import smart

        preset = smart.get(self.material)
        ts = _ts(ctx)
        if preset is None:
            self.report(ctx, "没有这个智能材质", "WARNING")
            return CANCELLED
        if not _room(ts, len(preset["layers"]) + 1):
            self.report(ctx, "一套贴图的图层放不下这个智能材质了", "WARNING")
            return CANCELLED
        before = ts.structure()
        folder = Layer(name=ts.unique_name(preset["name"]), kind="FOLDER")
        ts.add_layer(folder)
        if ts.depth(folder) >= MAX_FOLDER_DEPTH:
            ts.set_structure(before)
            self.report(ctx, "文件夹最多套 %d 层，这里放不下智能材质" % MAX_FOLDER_DEPTH, "WARNING")
            return CANCELLED
        added = [folder]
        for values in preset["layers"]:
            values = dict(values)
            layer = Layer(name=values.pop("name"), **values)
            layer.parent_uid = folder.uid
            ts.add_layer(layer, ts.index_of(folder))
            added.append(layer)
        ts.set_active(folder)
        _record(ctx, "智能材质「%s」" % preset["name"], ts, before, added=added)
        ensure = getattr(ctx.app, "ensure_meshmaps", None)
        if ensure is not None:
            ensure(ts)
        self.report(ctx, "已添加智能材质「%s」" % preset["name"])
        return FINISHED


@ops.register
class LayerDelete(_ActiveLayerOp):
    idname = "layer.delete"
    label = "删除图层"
    description = "删除当前图层；文件夹连同里面的图层一起删除"
    icon = "remove"
    uid = IntProperty("图层", default=0)

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        layer = _target(ctx, self.uid)
        if layer is None:
            return CANCELLED
        _stop_strokes(ctx)
        before = ts.structure()
        start, end = ts.subtree_range(layer)
        block = ts.layers[start:end]
        del ts.layers[start:end]
        if ts.active_layer is None and ts.layers:
            ts.active_layer_uid = ts.layers[min(start, len(ts.layers) - 1)].uid
        ts.layers_changed.emit("structure", layer)
        _record(ctx, "删除文件夹" if layer.kind == "FOLDER" else self.label, ts, before, removed=block)
        return FINISHED


@ops.register
class LayerDuplicate(_ActiveLayerOp):
    idname = "layer.duplicate"
    label = "复制图层"
    description = "复制当前图层；文件夹连同里面的图层一起复制。有选区时只复制选区里的内容（通过拷贝的图层）"
    icon = "duplicate"
    uid = IntProperty("图层", default=0)

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        source = _target(ctx, self.uid)
        if source is None:
            return CANCELLED
        block = ts.subtree(source)
        if not _room(ts, len(block)):
            self.report(ctx, "一套贴图的图层放不下了", "WARNING")
            return CANCELLED
        before = ts.structure()
        engine = _engine(ctx)
        mapping = {}
        copies = []
        for item in block:
            data = item.to_dict()
            data.pop("uid", None)
            copy = Layer()
            copy.from_dict(data)
            mapping[item.uid] = copy.uid
            copies.append(copy)
        for item, copy in zip(block, copies):
            copy.parent_uid = mapping.get(item.parent_uid, item.parent_uid)
            if engine is not None:
                engine.duplicate_layer_pixels(item.uid, copy.uid)
        root = copies[-1]
        root.name = ts.unique_name(source.name + " 副本")
        insert = ts.index_of(source) + 1
        ts.layers[insert:insert] = copies
        for copy in copies:
            ts._watch(copy)
        ts.active_layer_uid = root.uid
        ts.layers_changed.emit("structure", root)
        _record(ctx, self.label, ts, before, added=copies)
        if len(copies) == 1 and engine is not None:
            from .select_ops import keep_selected

            keep_selected(ctx, root, "通过拷贝的图层")
        return FINISHED


@ops.register
class LayerMove(_ActiveLayerOp):
    idname = "layer.move"
    label = "移动图层"
    description = "和同一个文件夹里相邻的图层换位置"
    direction = EnumProperty("方向", items=[("UP", "上移", ""), ("DOWN", "下移", "")], default="UP")
    uid = IntProperty("图层", default=0)

    @classmethod
    def poll(cls, ctx) -> bool:
        ts = _ts(ctx)
        return ts is not None and ts.active_layer is not None and len(ts.layers) > 1

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        layer = _target(ctx, self.uid)
        if layer is None:
            return CANCELLED
        siblings = [item for item in ts.layers if item.parent_uid == layer.parent_uid]
        position = siblings.index(layer)
        step = 1 if self.direction == "UP" else -1
        if not 0 <= position + step < len(siblings):
            return CANCELLED
        neighbor = siblings[position + step]
        start, end = ts.subtree_range(neighbor)
        before = ts.structure()
        ts.move_subtree(layer, layer.parent_uid, end if step > 0 else start)
        _record(ctx, "上移图层" if step > 0 else "下移图层", ts, before)
        return FINISHED


@ops.register
class LayerMoveTo(_LayerOp):
    idname = "layer.move_to"
    label = "调整图层位置"
    searchable = False
    uid = IntProperty("图层", default=0)
    index = IntProperty("位置", default=0, description="放到列表的这个下标处（按挪走之前算）")
    parent = IntProperty("文件夹", default=-1, description="放进哪个文件夹，0 是最外层，-1 表示不变")

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        layer = ts.layer(self.uid)
        if layer is None:
            return CANCELLED
        parent_uid = layer.parent_uid if self.parent < 0 else int(self.parent)
        parent = ts.layer(parent_uid) if parent_uid else None
        if parent_uid and (parent is None or parent.kind != "FOLDER" or parent is layer
                           or ts.is_ancestor(layer, parent)):
            return CANCELLED
        if not _new_depth_ok(ts, layer, parent_uid):
            self.report(ctx, "文件夹最多套 %d 层" % MAX_FOLDER_DEPTH, "WARNING")
            return CANCELLED
        index = max(0, min(len(ts.layers), int(self.index)))
        before = ts.structure()
        ts.move_subtree(layer, parent_uid, index)
        if ts.structure()[:2] == before[:2]:
            return CANCELLED
        _record(ctx, self.label, ts, before)
        return FINISHED


@ops.register
class LayerGroup(_ActiveLayerOp):
    idname = "layer.group"
    label = "放进新文件夹"
    description = "新建一个文件夹，把当前图层放进去"
    icon = "layer.folder"

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        layer = ts.active_layer
        if not _room(ts, 1):
            self.report(ctx, "一套贴图的图层已经满了", "WARNING")
            return CANCELLED
        parent = ts.parent_of(layer)
        if (ts.depth(parent) + 1 if parent is not None else 0) + 1 + _subtree_depth(ts, layer) > MAX_FOLDER_DEPTH:
            self.report(ctx, "文件夹最多套 %d 层" % MAX_FOLDER_DEPTH, "WARNING")
            return CANCELLED
        before = ts.structure()
        folder = Layer(name=ts.unique_name("文件夹"), kind="FOLDER", parent_uid=layer.parent_uid)
        index = ts.index_of(layer) + 1
        ts.layers.insert(index, folder)
        ts._watch(folder)
        layer.parent_uid = folder.uid
        ts.active_layer_uid = folder.uid
        ts.layers_changed.emit("structure", folder)
        _record(ctx, self.label, ts, before, added=[folder])
        return FINISHED


@ops.register
class LayerUngroup(_ActiveLayerOp):
    idname = "layer.ungroup"
    label = "解散文件夹"
    description = "把文件夹里的图层放到外面一层，删掉这个文件夹（文件夹的蒙版和设置一起去掉）"
    icon = "layer.folder"

    @classmethod
    def poll(cls, ctx) -> bool:
        ts = _ts(ctx)
        layer = ts.active_layer if ts is not None else None
        return layer is not None and layer.kind == "FOLDER"

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        folder = ts.active_layer
        _stop_strokes(ctx)
        before = ts.structure()
        for item in ts.children(folder):
            item.parent_uid = folder.parent_uid
        index = ts.index_of(folder)
        ts.layers.pop(index)
        if ts.layers:
            ts.active_layer_uid = ts.layers[max(0, index - 1)].uid
        ts.layers_changed.emit("structure", folder)
        _record(ctx, self.label, ts, before, removed=[folder])
        return FINISHED


@ops.register
class LayerSelect(_LayerOp):
    idname = "layer.select"
    label = "选择图层"
    searchable = False
    uid = IntProperty("图层", default=0)
    target = EnumProperty("目标", items=[("KEEP", "保持", ""), ("CONTENT", "内容", ""), ("MASK", "蒙版", "")],
                          default="KEEP")

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        layer = ts.layer(self.uid)
        if layer is None:
            return CANCELLED
        ts.set_active(layer)
        if self.target != "KEEP":
            ts.paint_target = self.target if (self.target == "CONTENT" or layer.has_mask) else "CONTENT"
        elif ts.paint_target == "MASK" and not layer.has_mask:
            ts.paint_target = "CONTENT"
        return FINISHED


@ops.register
class LayerToggleVisible(_LayerOp):
    idname = "layer.toggle_visible"
    label = "显示或隐藏图层"
    searchable = False
    uid = IntProperty("图层", default=0)

    def execute(self, ctx) -> str:
        layer = _target(ctx, self.uid)
        if layer is None:
            return CANCELLED
        value = not layer.visible
        layer.visible = value
        ctx.history.push("显示图层" if value else "隐藏图层", lambda: setattr(layer, "visible", not value),
                         lambda: setattr(layer, "visible", value))
        return FINISHED


@ops.register
class LayerToggleLock(_LayerOp):
    idname = "layer.toggle_lock"
    label = "锁定或解锁图层"
    searchable = False
    uid = IntProperty("图层", default=0)

    def execute(self, ctx) -> str:
        layer = _target(ctx, self.uid)
        if layer is None:
            return CANCELLED
        layer.locked = not layer.locked
        return FINISHED


@ops.register
class LayerMaskAdd(_ActiveLayerOp):
    idname = "layer.mask_add"
    label = "添加蒙版"
    description = "给当前图层加一张蒙版。白色蒙版先全部显示，黑色蒙版先全部隐藏"
    icon = "layer.mask"
    fill = EnumProperty("初始", items=[("WHITE", "白色（全部显示）", ""), ("BLACK", "黑色（全部隐藏）", "")],
                        default="WHITE")

    @classmethod
    def poll(cls, ctx) -> bool:
        ts = _ts(ctx)
        layer = ts.active_layer if ts is not None else None
        return layer is not None and not layer.has_mask

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        layer = ts.active_layer
        value = 1.0 if self.fill == "WHITE" else 0.0
        previous = (layer.mask_default, ts.paint_target)

        def apply() -> None:
            layer.mask_default = value
            layer.has_mask = True
            layer.mask_enabled = True
            ts.paint_target = "MASK"

        def undo() -> None:
            layer.has_mask = False
            layer.mask_default, ts.paint_target = previous[0], "CONTENT"

        apply()
        if self.fill == "BLACK":
            ctx.app.tool_settings.brush.mask_value = 1.0
        ctx.history.push(self.label, undo, apply)
        return FINISHED


@ops.register
class LayerMaskRemove(_ActiveLayerOp):
    idname = "layer.mask_remove"
    label = "删除蒙版"
    icon = "remove"

    @classmethod
    def poll(cls, ctx) -> bool:
        ts = _ts(ctx)
        layer = ts.active_layer if ts is not None else None
        return layer is not None and layer.has_mask

    def execute(self, ctx) -> str:
        ts = _ts(ctx)
        layer = ts.active_layer
        engine = _engine(ctx)
        state = {"plane": None, "removed": True}

        def take() -> None:
            eng = _engine(ctx)
            store = eng.layers.stores.get(layer.uid) if eng is not None else None
            state["plane"] = store.planes.pop(PLANE_MASK, None) if store is not None else None
            layer.has_mask = False
            ts.paint_target = "CONTENT"
            state["removed"] = True

        def give() -> None:
            eng = _engine(ctx)
            if eng is not None and state["plane"] is not None:
                eng.layers.store(layer.uid, ts.size // 256).planes[PLANE_MASK] = state["plane"]
            state["plane"] = None
            layer.has_mask = True
            state["removed"] = False

        def dispose() -> None:
            plane = state["plane"]
            eng = _engine(ctx)
            if state["removed"] and plane is not None and eng is not None:
                for level in plane.levels:
                    for page in list(level.pages.values()):
                        eng.cache.free_page(page)

        if engine is not None:
            engine.stroke_cancel()
        take()
        ctx.history.push(self.label, give, take, 0, dispose)
        return FINISHED
