"""3D 视口的菜单和饼菜单（照 Blender 4.3）：视图、选择、添加、物体、右键菜单，
`（视图饼）、Z（着色饼）、Shift+S（吸附饼）、.（轴心点饼）、,（坐标系饼）、Ctrl+Tab（模式饼）。"""
from __future__ import annotations

from ..core import registry
from ..doc.objects import D  # noqa: F401  确保物体模块已加载


def _menu(cls):
    return registry.register_menu(cls)


# ====================================================================== 视图
@_menu
class View3DViewMenu(registry.Menu):
    idname = "VIEW3D_MT_view"
    label = "视图"

    def draw(self, layout, ctx) -> None:
        layout.operator("screen.region_toggle", text="工具栏", region="TOOLBAR")
        layout.operator("screen.region_toggle", text="侧栏", region="SIDEBAR")
        layout.separator()
        layout.operator("screen.redo_last", text="调整上一步")
        layout.separator()
        layout.operator("view3d.view_selected", text="查看所选", icon="focus")
        layout.operator("view3d.view_all", text="查看全部")
        layout.operator("view3d.view_all", text="查看全部并重置游标", center=True)
        layout.operator("view3d.view_center_cursor", text="视图中心对齐游标")
        layout.operator("view3d.zoom_border", text="框选放大")
        layout.separator()
        layout.operator("view3d.view_camera", text="摄像机视图", icon="object.camera")
        layout.menu("VIEW3D_MT_view_viewpoint", text="视角")
        layout.menu("VIEW3D_MT_view_navigation", text="导航")
        layout.menu("VIEW3D_MT_view_align", text="对齐视图")
        layout.separator()
        layout.operator("view3d.view_persportho", text="透视 / 正交", icon="perspective")
        layout.operator("view3d.localview", text="局部视图")
        layout.separator()
        layout.operator("screen.area_maximize", text="最大化区域", icon="maximize")


@_menu
class View3DViewpointMenu(registry.Menu):
    idname = "VIEW3D_MT_view_viewpoint"
    label = "视角"

    def draw(self, layout, ctx) -> None:
        layout.operator("view3d.view_camera", text="摄像机")
        layout.separator()
        for axis, text in (("TOP", "顶视图"), ("BOTTOM", "底视图"), ("FRONT", "前视图"), ("BACK", "后视图"),
                           ("RIGHT", "右视图"), ("LEFT", "左视图")):
            layout.operator("view3d.view_axis", text=text, axis=axis)


@_menu
class View3DNavigationMenu(registry.Menu):
    idname = "VIEW3D_MT_view_navigation"
    label = "导航"

    def draw(self, layout, ctx) -> None:
        layout.operator("view3d.orbit_step", text="向左转", yaw=-15.0)
        layout.operator("view3d.orbit_step", text="向右转", yaw=15.0)
        layout.operator("view3d.orbit_step", text="向上转", pitch=-15.0)
        layout.operator("view3d.orbit_step", text="向下转", pitch=15.0)
        layout.operator("view3d.orbit_step", text="转到对面", yaw=180.0)
        layout.separator()
        layout.operator("view3d.view_roll", text="向左滚", angle=-15.0)
        layout.operator("view3d.view_roll", text="向右滚", angle=15.0)
        layout.separator()
        layout.operator("view3d.view_pan", text="左移", direction="LEFT")
        layout.operator("view3d.view_pan", text="右移", direction="RIGHT")
        layout.operator("view3d.view_pan", text="上移", direction="UP")
        layout.operator("view3d.view_pan", text="下移", direction="DOWN")
        layout.separator()
        layout.operator("view3d.zoom", text="拉近", delta=1)
        layout.operator("view3d.zoom", text="推远", delta=-1)
        layout.separator()
        layout.operator("view3d.navigate", text="飞行 / 行走漫游")


@_menu
class View3DAlignMenu(registry.Menu):
    idname = "VIEW3D_MT_view_align"
    label = "对齐视图"

    def draw(self, layout, ctx) -> None:
        layout.operator("view3d.camera_to_view", text="当前摄像机对齐视图")
        layout.operator("view3d.object_as_camera", text="设当前物体为摄像机")
        layout.separator()
        layout.operator("view3d.view_center_cursor", text="视图中心对齐游标")
        layout.operator("view3d.view_selected", text="视图中心对齐所选")
        layout.separator()
        for axis, text in (("TOP", "对齐当前物体：顶"), ("FRONT", "对齐当前物体：前"), ("RIGHT", "对齐当前物体：右")):
            layout.operator("view3d.view_axis", text=text, axis=axis, align_active=True)


# ====================================================================== 选择
@_menu
class View3DSelectObjectMenu(registry.Menu):
    idname = "VIEW3D_MT_select_object"
    label = "选择"

    def draw(self, layout, ctx) -> None:
        layout.operator("object.select_all", text="全选", action="SELECT")
        layout.operator("object.select_all", text="全不选", action="DESELECT")
        layout.operator("object.select_all", text="反选", action="INVERT")
        layout.separator()
        layout.operator("view3d.select_box", text="框选", wait_for_input=True)
        layout.operator("view3d.select_circle", text="刷选")
        layout.separator()
        layout.menu("VIEW3D_MT_select_by_type", text="按类型全选")
        layout.operator("object.select_linked", text="选择关联（共用材质）")
        layout.menu("VIEW3D_MT_select_grouped", text="按组选择")


@_menu
class View3DSelectByTypeMenu(registry.Menu):
    idname = "VIEW3D_MT_select_by_type"
    label = "按类型全选"

    def draw(self, layout, ctx) -> None:
        for kind, text, icon in (("MESH", "网格", "object.mesh"), ("CAMERA", "摄像机", "object.camera"),
                                 ("LIGHT", "灯光", "object.light"), ("EMPTY", "空物体", "object.empty")):
            layout.operator("object.select_by_type", text=text, icon=icon, type=kind)


@_menu
class View3DSelectGroupedMenu(registry.Menu):
    idname = "VIEW3D_MT_select_grouped"
    label = "按组选择"

    def draw(self, layout, ctx) -> None:
        layout.operator("object.select_grouped", text="类型", type="TYPE")
        layout.operator("object.select_grouped", text="灯光类型", type="LIGHT_TYPE")


# ====================================================================== 添加
@_menu
class View3DAddMenu(registry.Menu):
    idname = "VIEW3D_MT_add"
    label = "添加"

    def draw(self, layout, ctx) -> None:
        layout.menu("VIEW3D_MT_mesh_add", text="网格", icon="object.mesh")
        layout.separator()
        layout.menu("VIEW3D_MT_empty_add", text="空物体", icon="object.empty")
        layout.separator()
        layout.operator("object.camera_add", text="摄像机", icon="object.camera")
        layout.menu("VIEW3D_MT_light_add", text="灯光", icon="object.light")


@_menu
class View3DMeshAddMenu(registry.Menu):
    idname = "VIEW3D_MT_mesh_add"
    label = "网格"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.primitive_plane_add", text="平面", icon="prim.plane")
        layout.operator("mesh.primitive_cube_add", text="立方体", icon="prim.cube")
        layout.operator("mesh.primitive_circle_add", text="圆", icon="prim.circle")
        layout.operator("mesh.primitive_uv_sphere_add", text="经纬球", icon="prim.uv_sphere")
        layout.operator("mesh.primitive_ico_sphere_add", text="棱角球", icon="prim.ico_sphere")
        layout.operator("mesh.primitive_cylinder_add", text="柱体", icon="prim.cylinder")
        layout.operator("mesh.primitive_cone_add", text="锥体", icon="prim.cone")
        layout.operator("mesh.primitive_torus_add", text="环体", icon="prim.torus")
        layout.separator()
        layout.operator("mesh.primitive_grid_add", text="栅格", icon="prim.grid")


@_menu
class View3DLightAddMenu(registry.Menu):
    idname = "VIEW3D_MT_light_add"
    label = "灯光"

    def draw(self, layout, ctx) -> None:
        for kind, text, icon in (("POINT", "点光", "light.point"), ("SUN", "日光", "light.sun"),
                                 ("SPOT", "聚光", "light.spot"), ("AREA", "面光", "light.area")):
            layout.operator("object.light_add", text=text, icon=icon, type=kind)


@_menu
class View3DEmptyAddMenu(registry.Menu):
    idname = "VIEW3D_MT_empty_add"
    label = "空物体"

    def draw(self, layout, ctx) -> None:
        from ..ops.object_ops import EMPTY_TYPES

        for ident, text, _desc, icon in EMPTY_TYPES:
            layout.operator("object.empty_add", text=text, icon=icon, type=ident)


# ====================================================================== 物体
@_menu
class View3DObjectMenu(registry.Menu):
    idname = "VIEW3D_MT_object"
    label = "物体"

    def draw(self, layout, ctx) -> None:
        layout.menu("VIEW3D_MT_transform_object", text="变换")
        layout.menu("VIEW3D_MT_origin_set", text="设置原点")
        layout.menu("VIEW3D_MT_mirror", text="镜像")
        layout.menu("VIEW3D_MT_object_clear", text="清除")
        layout.menu("VIEW3D_MT_object_apply", text="应用")
        layout.menu("VIEW3D_MT_snap", text="吸附")
        layout.separator()
        layout.operator("object.duplicate_move", text="复制物体", icon="duplicate")
        layout.operator("object.join", text="合并", icon="join")
        layout.separator()
        layout.menu("VIEW3D_MT_make_links", text="关联 / 传递数据")
        layout.separator()
        layout.operator("object.shade_smooth", text="平滑着色")
        layout.operator("object.shade_smooth_by_angle", text="按角度平滑着色")
        layout.operator("object.shade_flat", text="平直着色")
        layout.separator()
        layout.menu("VIEW3D_MT_object_showhide", text="显示 / 隐藏")
        layout.separator()
        layout.operator("object.delete", text="删除", icon="remove", confirm=False)


@_menu
class View3DTransformObjectMenu(registry.Menu):
    idname = "VIEW3D_MT_transform_object"
    label = "变换"

    def draw(self, layout, ctx) -> None:
        layout.operator("transform.translate", text="移动", icon="tool.move")
        layout.operator("transform.rotate", text="旋转", icon="tool.rotate")
        layout.operator("transform.resize", text="缩放", icon="tool.scale")
        layout.separator()
        layout.operator("transform.mirror", text="镜像", icon="mirror")


@_menu
class View3DOriginSetMenu(registry.Menu):
    idname = "VIEW3D_MT_origin_set"
    label = "设置原点"

    def draw(self, layout, ctx) -> None:
        layout.operator("object.origin_set", text="几何中心 → 原点", type="GEOMETRY_ORIGIN")
        layout.operator("object.origin_set", text="原点 → 几何中心", type="ORIGIN_GEOMETRY")
        layout.operator("object.origin_set", text="原点 → 3D 游标", type="ORIGIN_CURSOR")
        layout.operator("object.origin_set", text="原点 → 质心（表面）", type="ORIGIN_CENTER_OF_MASS")


@_menu
class View3DMirrorMenu(registry.Menu):
    idname = "VIEW3D_MT_mirror"
    label = "镜像"

    def draw(self, layout, ctx) -> None:
        layout.operator("transform.mirror", text="交互镜像")
        layout.separator()
        for orient, word in (("GLOBAL", "全局"), ("LOCAL", "局部")):
            for axis in "XYZ":
                layout.operator("transform.mirror", text="%s %s" % (axis, word), constraint=axis, orient_type=orient)
            layout.separator()


@_menu
class View3DObjectClearMenu(registry.Menu):
    idname = "VIEW3D_MT_object_clear"
    label = "清除"

    def draw(self, layout, ctx) -> None:
        layout.operator("object.location_clear", text="位置")
        layout.operator("object.rotation_clear", text="旋转")
        layout.operator("object.scale_clear", text="缩放")


@_menu
class View3DObjectApplyMenu(registry.Menu):
    idname = "VIEW3D_MT_object_apply"
    label = "应用"

    def draw(self, layout, ctx) -> None:
        layout.operator("object.transform_apply", text="位置", location=True, rotation=False, scale=False)
        layout.operator("object.transform_apply", text="旋转", location=False, rotation=True, scale=False)
        layout.operator("object.transform_apply", text="缩放", location=False, rotation=False, scale=True)
        layout.operator("object.transform_apply", text="全部变换", location=True, rotation=True, scale=True)
        layout.operator("object.transform_apply", text="旋转和缩放", location=False, rotation=True, scale=True)


@_menu
class View3DSnapMenu(registry.Menu):
    idname = "VIEW3D_MT_snap"
    label = "吸附"

    def draw(self, layout, ctx) -> None:
        layout.operator("view3d.snap_selected_to_grid", text="选中项 → 网格点")
        layout.operator("view3d.snap_selected_to_cursor", text="选中项 → 游标", use_offset=False)
        layout.operator("view3d.snap_selected_to_cursor", text="选中项 → 游标（保持偏移）", use_offset=True)
        layout.operator("view3d.snap_selected_to_active", text="选中项 → 活动项")
        layout.separator()
        layout.operator("view3d.snap_cursor_to_selected", text="游标 → 选中项")
        layout.operator("view3d.snap_cursor_to_center", text="游标 → 世界原点")
        layout.operator("view3d.snap_cursor_to_grid", text="游标 → 网格点")
        layout.operator("view3d.snap_cursor_to_active", text="游标 → 活动项")


@_menu
class View3DMakeLinksMenu(registry.Menu):
    idname = "VIEW3D_MT_make_links"
    label = "关联 / 传递数据"

    def draw(self, layout, ctx) -> None:
        layout.operator("object.make_links_data", text="关联材质", type="MATERIAL")


@_menu
class View3DShowHideMenu(registry.Menu):
    idname = "VIEW3D_MT_object_showhide"
    label = "显示 / 隐藏"

    def draw(self, layout, ctx) -> None:
        layout.operator("object.hide_view_clear", text="显示隐藏的物体")
        layout.operator("object.hide_view_set", text="隐藏所选", unselected=False)
        layout.operator("object.hide_view_set", text="隐藏未选", unselected=True)


@_menu
class View3DObjectContextMenu(registry.Menu):
    """物体模式的右键菜单（照 Blender：按当前物体的类型给常用的几项）。"""

    idname = "VIEW3D_MT_object_context_menu"
    label = "物体"

    def draw(self, layout, ctx) -> None:
        project = getattr(ctx.app, "project", None)
        active = project.active_object if project is not None else None
        kind = getattr(active, "kind", "MESH")
        if kind == "MESH":
            layout.operator("object.shade_smooth", text="平滑着色")
            layout.operator("object.shade_smooth_by_angle", text="按角度平滑着色")
            layout.operator("object.shade_flat", text="平直着色")
            layout.separator()
            layout.menu("VIEW3D_MT_origin_set", text="设置原点")
        elif kind == "CAMERA":
            layout.operator("view3d.object_as_camera", text="设为当前摄像机并从它看", icon="object.camera")
            layout.separator()
        elif kind == "LIGHT":
            layout.label(active.name, icon="object.light")
            layout.separator()
        layout.separator()
        layout.operator("object.duplicate_move", text="复制物体", icon="duplicate")
        layout.operator("object.join", text="合并", icon="join")
        layout.separator()
        layout.operator("wm.rename_active", text="重命名", icon="pencil")
        layout.separator()
        layout.menu("VIEW3D_MT_mirror", text="镜像")
        layout.menu("VIEW3D_MT_snap", text="吸附")
        layout.separator()
        layout.menu("VIEW3D_MT_object_showhide", text="显示 / 隐藏")
        layout.operator("object.delete", text="删除", icon="remove", confirm=False)


# ====================================================================== 饼菜单（条目顺序：左、右、下、上、左上、右上、左下、右下）
@_menu
class View3DViewPie(registry.Menu):
    idname = "VIEW3D_MT_view_pie"
    label = "视图"

    def draw(self, layout, ctx) -> None:
        layout.operator("view3d.view_axis", text="左视图", axis="LEFT")
        layout.operator("view3d.view_axis", text="右视图", axis="RIGHT")
        layout.operator("view3d.view_axis", text="底视图", axis="BOTTOM")
        layout.operator("view3d.view_axis", text="顶视图", axis="TOP")
        layout.operator("view3d.view_axis", text="前视图", axis="FRONT")
        layout.operator("view3d.view_axis", text="后视图", axis="BACK")
        layout.operator("view3d.view_camera", text="摄像机视图")
        layout.operator("view3d.view_selected", text="查看所选")


@_menu
class View3DShadingPie(registry.Menu):
    idname = "VIEW3D_MT_shading_pie"
    label = "着色"

    def draw(self, layout, ctx) -> None:
        layout.operator("view3d.shading_set", text="线框", mode="WIREFRAME")
        layout.operator("view3d.shading_set", text="实体", mode="SOLID")
        layout.operator("view3d.shading_set", text="材质预览", mode="MATERIAL")
        layout.operator("view3d.shading_set", text="渲染", mode="RENDERED")
        layout.operator("view3d.toggle_xray", text="X 光")
        layout.operator("wm.context_toggle", text="叠加层", data_path="space_data.overlay.show_overlays")
        layout.operator("view3d.shading_set", text="通道", mode="CHANNEL")
        layout.operator("view3d.channel_cycle", text="下一个通道")


@_menu
class View3DSnapPie(registry.Menu):
    idname = "VIEW3D_MT_snap_pie"
    label = "吸附"

    def draw(self, layout, ctx) -> None:
        layout.operator("view3d.snap_cursor_to_grid", text="游标 → 网格点")
        layout.operator("view3d.snap_selected_to_grid", text="选中项 → 网格点")
        layout.operator("view3d.snap_cursor_to_selected", text="游标 → 选中项")
        layout.operator("view3d.snap_selected_to_cursor", text="选中项 → 游标", use_offset=False)
        layout.operator("view3d.snap_selected_to_cursor", text="选中项 → 游标（保持偏移）", use_offset=True)
        layout.operator("view3d.snap_selected_to_active", text="选中项 → 活动项")
        layout.operator("view3d.snap_cursor_to_center", text="游标 → 世界原点")
        layout.operator("view3d.snap_cursor_to_active", text="游标 → 活动项")


@_menu
class View3DPivotPie(registry.Menu):
    idname = "VIEW3D_MT_pivot_pie"
    label = "轴心点"

    def draw(self, layout, ctx) -> None:
        for value, text in (("BOUNDING_BOX_CENTER", "边界框中心"), ("CURSOR", "3D 游标"),
                            ("INDIVIDUAL_ORIGINS", "各自的原点"), ("MEDIAN_POINT", "质心点"),
                            ("ACTIVE_ELEMENT", "活动元素")):
            layout.operator("wm.context_set_enum", text=text, data_path="tool_settings.transform_pivot_point",
                            value=value)


@_menu
class View3DOrientationsPie(registry.Menu):
    idname = "VIEW3D_MT_orientations_pie"
    label = "坐标系"

    def draw(self, layout, ctx) -> None:
        for value, text in (("GLOBAL", "全局"), ("LOCAL", "局部"), ("NORMAL", "法向"), ("VIEW", "视图"),
                            ("CURSOR", "游标")):
            layout.operator("wm.context_set_enum", text=text, data_path="tool_settings.transform_orientation",
                            value=value)


@_menu
class View3DObjectModePie(registry.Menu):
    idname = "VIEW3D_MT_object_mode_pie"
    label = "模式"

    def draw(self, layout, ctx) -> None:
        from ..ops.wm_ops import available_modes

        for ident, text, _desc, icon in available_modes(ctx.app):
            layout.operator("object.mode_set", text=text, icon=icon, mode=ident)


@_menu
class View3DSculptMenu(registry.Menu):
    idname = "VIEW3D_MT_sculpt"
    label = "雕刻"

    def draw(self, layout, ctx) -> None:
        layout.operator("sculpt.subdivide", text="细分", icon="grid")
        layout.operator("sculpt.voxel_remesh", text="体素重构", icon="remesh")
        layout.separator()
        layout.operator("sculpt.mask_clear", text="清除遮罩")
        layout.operator("sculpt.mask_invert", text="反转遮罩")
        layout.operator("sculpt.mask_fill", text="填满遮罩")


@_menu
class View3DSculptMaskPie(registry.Menu):
    idname = "VIEW3D_MT_sculpt_mask_edit_pie"
    label = "遮罩"

    def draw(self, layout, ctx) -> None:
        layout.operator("sculpt.mask_invert", text="反转遮罩")
        layout.operator("sculpt.mask_clear", text="清除遮罩")
        layout.operator("sculpt.mask_fill", text="填满遮罩")
        layout.operator("sculpt.mask_smooth", text="柔化遮罩")
        layout.operator("sculpt.mask_sharpen", text="锐化遮罩")
        layout.operator("sculpt.mask_grow", text="扩大遮罩")
        layout.operator("sculpt.mask_shrink", text="缩小遮罩")
        layout.operator("sculpt.mask_contrast", text="增加遮罩对比度")


@_menu
class View3DPaintMenu(registry.Menu):
    idname = "VIEW3D_MT_paint"
    label = "纹理绘制"

    def draw(self, layout, ctx) -> None:
        layout.operator("paint.swap_colors", text="交换颜色")
        layout.operator("paint.sample_color", text="吸取颜色")
        layout.separator()
        layout.operator("paint.tool_set", text="画笔", icon="tool.brush", tool="paint.brush")
        layout.operator("paint.tool_set", text="橡皮", icon="tool.eraser", tool="paint.eraser")
