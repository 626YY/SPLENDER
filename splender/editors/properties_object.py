"""属性编辑器里「物体」和「数据」两页（照 Blender）：当前物体的名字、显示、变换；摄像机、灯光、空物体的参数。"""
from __future__ import annotations

from ..core import registry

SPACE = "PROPERTIES"


def _active(ctx):
    project = getattr(ctx.app, "project", None)
    return project.active_object if project is not None else None


class _Panel(registry.Panel):
    space = SPACE
    region = "MAIN"


# ====================================================================== 物体
@registry.register_panel
class ObjectPanel(_Panel):
    idname = "PROPERTIES_PT_object"
    label = "物体"
    category = "OBJECT"
    order = 1
    icon = "mode.object"

    def draw(self, layout, ctx) -> None:
        obj = _active(ctx)
        if obj is None:
            layout.label("没有当前物体：在视口或大纲视图里点一个", role="dim")
            return
        layout.prop(obj, "name")
        layout.prop(obj, "visible")
        kind = {"MESH": "网格", "CAMERA": "摄像机", "LIGHT": "灯光", "EMPTY": "空物体"}.get(obj.kind, obj.kind)
        layout.label("类型：%s" % kind, role="dim")
        if obj.kind == "MESH":
            layout.label("%s 个三角形" % format(int(obj.data.triangle_count), ","), role="dim")


@registry.register_panel
class ObjectTransformPanel(_Panel):
    idname = "PROPERTIES_PT_object_transform"
    label = "变换"
    category = "OBJECT"
    order = 2
    icon = "tool.move"

    @classmethod
    def poll(cls, ctx) -> bool:
        return _active(ctx) is not None

    def draw(self, layout, ctx) -> None:
        obj = _active(ctx)
        transform = obj.transform
        layout.prop(transform, "location")
        layout.prop(transform, "rotation")
        layout.prop(transform, "scale")
        row = layout.row(align=True)
        row.use_property_split = False
        row.menu("VIEW3D_MT_object_apply", text="应用")
        row.menu("VIEW3D_MT_object_clear", text="清除")
        row.menu("VIEW3D_MT_origin_set", text="原点")


@registry.register_panel
class ObjectMaterialsPanel(_Panel):
    idname = "PROPERTIES_PT_object_materials"
    label = "材质"
    category = "OBJECT"
    order = 3
    icon = "material"

    @classmethod
    def poll(cls, ctx) -> bool:
        obj = _active(ctx)
        return obj is not None and obj.kind == "MESH"

    def draw(self, layout, ctx) -> None:
        obj = _active(ctx)
        project = ctx.app.project
        for index, set_uid in enumerate(obj.material_sets):
            ts = project.texture_set(set_uid)
            if ts is None:
                continue
            name = obj.data.materials[index] if index < len(obj.data.materials) else ""
            row = layout.row(heading=name or ("材质 %d" % (index + 1)))
            row.operator("project.texture_set_activate", text=ts.name, icon="texture_set", uid=ts.uid)


# ====================================================================== 数据：摄像机
@registry.register_panel
class CameraLensPanel(_Panel):
    idname = "PROPERTIES_PT_camera_lens"
    label = "镜头"
    category = "MESH"
    order = 0
    icon = "object.camera"

    @classmethod
    def poll(cls, ctx) -> bool:
        obj = _active(ctx)
        return obj is not None and obj.kind == "CAMERA"

    def draw(self, layout, ctx) -> None:
        data = _active(ctx).data
        layout.prop(data, "projection", expand=True)
        if data.projection == "ORTHO":
            layout.prop(data, "ortho_scale")
        else:
            layout.prop(data, "lens")
        layout.prop(data, "sensor")
        layout.separator()
        layout.prop(data, "shift_x")
        layout.prop(data, "shift_y")
        layout.separator()
        layout.prop(data, "clip_start")
        layout.prop(data, "clip_end")
        layout.separator()
        layout.operator("view3d.object_as_camera", text="设为当前摄像机并从它看", icon="object.camera")


@registry.register_panel
class CameraDofPanel(_Panel):
    idname = "PROPERTIES_PT_camera_dof"
    label = "景深"
    category = "MESH"
    order = 0.5
    icon = "focus"

    @classmethod
    def poll(cls, ctx) -> bool:
        obj = _active(ctx)
        return obj is not None and obj.kind == "CAMERA"

    def draw(self, layout, ctx) -> None:
        data = _active(ctx).data
        layout.prop(data, "use_dof")
        col = layout.column()
        col.enabled = bool(data.use_dof)
        col.prop(data, "focus_distance")
        col.prop(data, "fstop")


@registry.register_panel
class CameraDisplayPanel(_Panel):
    idname = "PROPERTIES_PT_camera_display"
    label = "视口显示"
    category = "MESH"
    order = 0.7
    icon = "visible"
    default_closed = True

    @classmethod
    def poll(cls, ctx) -> bool:
        obj = _active(ctx)
        return obj is not None and obj.kind == "CAMERA"

    def draw(self, layout, ctx) -> None:
        layout.prop(_active(ctx).data, "display_size")


# ====================================================================== 数据：灯光
@registry.register_panel
class LightPanel(_Panel):
    idname = "PROPERTIES_PT_light"
    label = "灯光"
    category = "MESH"
    order = 0
    icon = "object.light"

    @classmethod
    def poll(cls, ctx) -> bool:
        obj = _active(ctx)
        return obj is not None and obj.kind == "LIGHT"

    def draw(self, layout, ctx) -> None:
        data = _active(ctx).data
        layout.prop(data, "type", expand=True)
        layout.prop(data, "color")
        if data.type == "SUN":
            layout.prop(data, "strength")
            layout.prop(data, "angle")
        else:
            layout.prop(data, "power")
            if data.type == "AREA":
                layout.prop(data, "size")
            else:
                layout.prop(data, "radius")
        if data.type == "SPOT":
            layout.separator()
            layout.prop(data, "spot_size")
            layout.prop(data, "spot_blend", slider=True)
        layout.separator()
        layout.prop(data, "use_shadow")


# ====================================================================== 数据：空物体
@registry.register_panel
class EmptyPanel(_Panel):
    idname = "PROPERTIES_PT_empty"
    label = "空物体"
    category = "MESH"
    order = 0
    icon = "object.empty"

    @classmethod
    def poll(cls, ctx) -> bool:
        obj = _active(ctx)
        return obj is not None and obj.kind == "EMPTY"

    def draw(self, layout, ctx) -> None:
        obj = _active(ctx)
        layout.prop(obj, "empty_display_type")
        layout.prop(obj, "empty_size")
