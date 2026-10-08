"""编辑模式的菜单（照 Blender 4.3）：标题栏的选择、添加、网格、点、边、面、UV，右键菜单，
X 删除、M 合并、Alt+M 拆分、P 分离、Alt+E 挤出、Shift+G 选择相似、Alt+N 法线、Shift+O 衰减方式饼。"""
from __future__ import annotations

from ..core import registry


def _menu(cls):
    return registry.register_menu(cls)


def _modes(ctx) -> set:
    engine = getattr(getattr(ctx.app, "host", None), "engine", None)
    session = getattr(engine, "edit", None)
    return set(session.mesh.select_mode) if session is not None else {"VERT"}


# ====================================================================== 选择
@_menu
class SelectEditMeshMenu(registry.Menu):
    idname = "VIEW3D_MT_select_edit_mesh"
    label = "选择"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.select_all", text="全选", action="SELECT")
        layout.operator("mesh.select_all", text="全不选", action="DESELECT")
        layout.operator("mesh.select_all", text="反选", action="INVERT")
        layout.separator()
        layout.operator("view3d.select_box", text="框选", wait_for_input=True)
        layout.operator("view3d.select_circle", text="刷选")
        layout.separator()
        layout.operator("mesh.select_random", text="随机选择")
        layout.separator()
        layout.operator("mesh.select_more", text="扩大选区")
        layout.operator("mesh.select_less", text="缩小选区")
        layout.separator()
        layout.operator("mesh.select_linked", text="选择相连")
        layout.menu("VIEW3D_MT_edit_mesh_select_similar", text="选择相似")
        layout.menu("VIEW3D_MT_edit_mesh_select_loops", text="选择循环")
        layout.separator()
        layout.operator("mesh.select_non_manifold", text="选择非流形")
        layout.operator("mesh.select_mirror", text="镜像选择")


@_menu
class SelectSimilarMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_select_similar"
    label = "选择相似"

    def draw(self, layout, ctx) -> None:
        modes = _modes(ctx)
        if "FACE" in modes:
            layout.operator("mesh.select_similar", text="朝向", type="NORMAL")
            layout.operator("mesh.select_similar", text="面积", type="AREA")
            layout.operator("mesh.select_similar", text="材质", type="MATERIAL")
            layout.operator("mesh.select_similar", text="边数", type="SIDES")
        if "EDGE" in modes:
            layout.operator("mesh.select_similar", text="边长", type="LENGTH")
        if "VERT" in modes:
            layout.operator("mesh.select_similar", text="连着的面数", type="FACES")


@_menu
class SelectLoopsMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_select_loops"
    label = "选择循环"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.loop_multi_select", text="循环边", ring=False)
        layout.operator("mesh.loop_multi_select", text="并排边", ring=True)


# ====================================================================== 网格
@_menu
class EditMeshMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh"
    label = "网格"

    def draw(self, layout, ctx) -> None:
        layout.menu("VIEW3D_MT_edit_mesh_transform", text="变换")
        layout.menu("VIEW3D_MT_mirror_edit", text="镜像")
        layout.menu("VIEW3D_MT_snap", text="吸附")
        layout.separator()
        layout.operator("mesh.duplicate_move", text="复制", icon="duplicate")
        layout.menu("VIEW3D_MT_edit_mesh_extrude", text="挤出")
        layout.separator()
        layout.menu("VIEW3D_MT_edit_mesh_merge", text="合并")
        layout.menu("VIEW3D_MT_edit_mesh_split", text="拆分")
        layout.menu("VIEW3D_MT_edit_mesh_separate", text="分离")
        layout.separator()
        layout.operator("mesh.knife_tool", text="切刀", icon="tool.knife")
        layout.separator()
        layout.menu("VIEW3D_MT_edit_mesh_normals", text="法线")
        layout.menu("VIEW3D_MT_edit_mesh_clean", text="清理")
        layout.separator()
        layout.menu("VIEW3D_MT_edit_mesh_showhide", text="显示 / 隐藏")
        layout.menu("VIEW3D_MT_edit_mesh_delete", text="删除")


@_menu
class EditMeshTransformMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_transform"
    label = "变换"

    def draw(self, layout, ctx) -> None:
        layout.operator("transform.translate", text="移动", icon="tool.move")
        layout.operator("transform.rotate", text="旋转", icon="tool.rotate")
        layout.operator("transform.resize", text="缩放", icon="tool.scale")
        layout.separator()
        layout.operator("transform.tosphere", text="球形化")
        layout.operator("transform.shear", text="切变")
        layout.operator("mesh.shrink_fatten", text="法向缩放")
        layout.separator()
        layout.operator("transform.edge_slide", text="滑移边")
        layout.operator("transform.vert_slide", text="滑移顶点")


@_menu
class EditMirrorMenu(registry.Menu):
    idname = "VIEW3D_MT_mirror_edit"
    label = "镜像"

    def draw(self, layout, ctx) -> None:
        layout.operator("transform.mirror", text="交互镜像")
        layout.separator()
        for axis in ("X", "Y", "Z"):
            layout.operator("transform.mirror", text="%s 全局" % axis, constraint=axis, orient_type="GLOBAL")
        for axis in ("X", "Y", "Z"):
            layout.operator("transform.mirror", text="%s 局部" % axis, constraint=axis, orient_type="LOCAL")


@_menu
class ExtrudeMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_extrude"
    label = "挤出"

    def draw(self, layout, ctx) -> None:
        modes = _modes(ctx)
        if "FACE" in modes or "VERT" in modes:
            layout.operator("view3d.edit_mesh_extrude_move_normal", text="挤出面", icon="tool.extrude")
            layout.operator("view3d.edit_mesh_extrude_move_shrink_fatten", text="沿法线挤出面")
            layout.operator("mesh.extrude_faces_move", text="挤出各个面")
        layout.operator("mesh.extrude_edges_move", text="只挤出边")
        layout.operator("mesh.extrude_vertices_move", text="只挤出点")


@_menu
class MergeMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_merge"
    label = "合并"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.merge", text="到中心", type="CENTER")
        layout.operator("mesh.merge", text="到游标", type="CURSOR")
        layout.operator("mesh.merge", text="塌陷", type="COLLAPSE")
        layout.operator("mesh.merge", text="到第一个选中的", type="FIRST")
        layout.operator("mesh.merge", text="到最后一个选中的", type="LAST")
        layout.separator()
        layout.operator("mesh.remove_doubles", text="按距离合并")


@_menu
class SplitMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_split"
    label = "拆分"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.split", text="选中项")
        layout.separator()
        layout.operator("mesh.edge_split", text="按边拆分面", type="EDGE")
        layout.operator("mesh.edge_split", text="按点拆分面和边", type="VERT")


@_menu
class SeparateMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_separate"
    label = "分离"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.separate", text="选中项", type="SELECTED")
        layout.operator("mesh.separate", text="按材质", type="MATERIAL")
        layout.operator("mesh.separate", text="按松散块", type="LOOSE")


@_menu
class NormalsMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_normals"
    label = "法线"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.flip_normals", text="翻转")
        layout.operator("mesh.normals_make_consistent", text="重新计算外侧", inside=False)
        layout.operator("mesh.normals_make_consistent", text="重新计算内侧", inside=True)
        layout.separator()
        layout.operator("mesh.faces_shade", text="平滑着色", smooth=True)
        layout.operator("mesh.faces_shade", text="平直着色", smooth=False)


@_menu
class CleanMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_clean"
    label = "清理"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.remove_doubles", text="按距离合并")
        layout.operator("mesh.fill", text="补洞")
        layout.operator("mesh.select_non_manifold", text="选择非流形")
        layout.separator()
        layout.operator("mesh.tris_convert_to_quads", text="三角面转四边面")


@_menu
class ShowHideMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_showhide"
    label = "显示 / 隐藏"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.reveal", text="显示隐藏的部分")
        layout.operator("mesh.hide", text="隐藏选中项", unselected=False)
        layout.operator("mesh.hide", text="隐藏没选中的", unselected=True)


@_menu
class DeleteMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_delete"
    label = "删除"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.delete", text="点", type="VERT")
        layout.operator("mesh.delete", text="边", type="EDGE")
        layout.operator("mesh.delete", text="面", type="FACE")
        layout.operator("mesh.delete", text="只删边和面", type="EDGE_FACE")
        layout.operator("mesh.delete", text="只删面", type="ONLY_FACE")
        layout.separator()
        layout.operator("mesh.dissolve_verts", text="融并点")
        layout.operator("mesh.dissolve_edges", text="融并边")
        layout.operator("mesh.dissolve_faces", text="融并面")
        layout.separator()
        layout.operator("mesh.merge", text="塌陷边和面", type="COLLAPSE")


# ====================================================================== 点、边、面
@_menu
class VerticesMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_vertices"
    label = "点"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.extrude_vertices_move", text="只挤出点")
        layout.operator("mesh.bevel", text="倒点", affect="VERTICES")
        layout.separator()
        layout.operator("mesh.edge_face_add", text="由点创建边或面")
        layout.operator("mesh.vert_connect_path", text="连接顶点路径")
        layout.operator("mesh.vert_connect", text="连接顶点对")
        layout.separator()
        layout.operator("mesh.rip_move", text="撕开", use_fill=False)
        layout.operator("mesh.rip_move", text="撕开并补缝", use_fill=True)
        layout.operator("transform.vert_slide", text="滑移顶点")
        layout.operator("mesh.vertices_smooth", text="平滑顶点")
        layout.separator()
        layout.menu("VIEW3D_MT_edit_mesh_merge", text="合并")
        layout.menu("VIEW3D_MT_edit_mesh_split", text="拆分")
        layout.menu("VIEW3D_MT_edit_mesh_separate", text="分离")


@_menu
class EdgesMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_edges"
    label = "边"

    def draw(self, layout, ctx) -> None:
        layout.operator("mesh.extrude_edges_move", text="只挤出边")
        layout.operator("mesh.bevel", text="倒角", affect="EDGES", icon="tool.bevel")
        layout.operator("mesh.bridge_edge_loops", text="桥接循环边")
        layout.separator()
        layout.operator("mesh.subdivide", text="细分")
        layout.operator("mesh.loopcut_slide", text="环切", icon="tool.loopcut")
        layout.separator()
        layout.operator("mesh.edge_rotate", text="顺时针旋转边", use_ccw=False)
        layout.operator("mesh.edge_rotate", text="逆时针旋转边", use_ccw=True)
        layout.separator()
        layout.operator("transform.edge_slide", text="滑移边")
        layout.operator("mesh.edge_split", text="按边拆分", type="EDGE")
        layout.operator("mesh.dissolve_edges", text="融并边")


@_menu
class FacesMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_faces"
    label = "面"

    def draw(self, layout, ctx) -> None:
        layout.operator("view3d.edit_mesh_extrude_move_normal", text="挤出面", icon="tool.extrude")
        layout.operator("view3d.edit_mesh_extrude_move_shrink_fatten", text="沿法线挤出面")
        layout.operator("mesh.extrude_faces_move", text="挤出各个面")
        layout.separator()
        layout.operator("mesh.inset", text="内插面", icon="tool.inset")
        layout.operator("mesh.poke", text="戳面")
        layout.separator()
        layout.operator("mesh.edge_face_add", text="填充")
        layout.operator("mesh.fill", text="补洞")
        layout.separator()
        layout.operator("mesh.quads_convert_to_tris", text="三角化面")
        layout.operator("mesh.tris_convert_to_quads", text="三角面转四边面")
        layout.operator("mesh.dissolve_faces", text="融并面")
        layout.separator()
        layout.operator("mesh.faces_shade", text="平滑着色", smooth=True)
        layout.operator("mesh.faces_shade", text="平直着色", smooth=False)


@_menu
class UVMapMenu(registry.Menu):
    idname = "VIEW3D_MT_uv_map"
    label = "UV"

    def draw(self, layout, ctx) -> None:
        layout.operator("uv.smart_project", text="智能 UV 投射")
        layout.operator("uv.project_from_view", text="从视角投射")
        layout.separator()
        layout.operator("uv.reset", text="重置 UV")


# ====================================================================== 右键菜单（按选择模式给常用的几项）
@_menu
class EditMeshContextMenu(registry.Menu):
    idname = "VIEW3D_MT_edit_mesh_context_menu"
    label = "网格"

    def draw(self, layout, ctx) -> None:
        modes = _modes(ctx)
        if "FACE" in modes and len(modes) == 1:
            layout.label("面", role="dim")
            layout.operator("mesh.subdivide", text="细分")
            layout.separator()
            layout.operator("view3d.edit_mesh_extrude_move_normal", text="挤出面")
            layout.operator("mesh.extrude_faces_move", text="挤出各个面")
            layout.operator("mesh.inset", text="内插面")
            layout.operator("mesh.poke", text="戳面")
            layout.separator()
            layout.operator("mesh.faces_shade", text="平滑着色", smooth=True)
            layout.operator("mesh.faces_shade", text="平直着色", smooth=False)
            layout.separator()
            layout.operator("mesh.flip_normals", text="翻转法线")
            layout.operator("mesh.quads_convert_to_tris", text="三角化面")
        elif "EDGE" in modes and "VERT" not in modes:
            layout.label("边", role="dim")
            layout.operator("mesh.subdivide", text="细分")
            layout.operator("mesh.loopcut_slide", text="环切")
            layout.operator("mesh.bridge_edge_loops", text="桥接循环边")
            layout.separator()
            layout.operator("mesh.extrude_edges_move", text="只挤出边")
            layout.operator("mesh.bevel", text="倒角", affect="EDGES")
            layout.separator()
            layout.operator("mesh.edge_rotate", text="顺时针旋转边", use_ccw=False)
            layout.operator("mesh.edge_rotate", text="逆时针旋转边", use_ccw=True)
            layout.separator()
            layout.operator("transform.edge_slide", text="滑移边")
            layout.operator("mesh.edge_split", text="按边拆分", type="EDGE")
        else:
            layout.label("点", role="dim")
            layout.operator("mesh.subdivide", text="细分")
            layout.separator()
            layout.operator("mesh.extrude_vertices_move", text="只挤出点")
            layout.operator("mesh.bevel", text="倒点", affect="VERTICES")
            layout.separator()
            layout.operator("mesh.edge_face_add", text="由点创建边或面")
            layout.operator("mesh.vert_connect_path", text="连接顶点路径")
            layout.separator()
            layout.operator("mesh.rip_move", text="撕开", use_fill=False)
            layout.operator("transform.vert_slide", text="滑移顶点")
            layout.operator("mesh.vertices_smooth", text="平滑顶点")
        layout.separator()
        layout.menu("VIEW3D_MT_mirror_edit", text="镜像")
        layout.menu("VIEW3D_MT_snap", text="吸附")
        layout.separator()
        layout.menu("VIEW3D_MT_edit_mesh_merge", text="合并")
        layout.menu("VIEW3D_MT_edit_mesh_split", text="拆分")
        layout.menu("VIEW3D_MT_edit_mesh_separate", text="分离")
        layout.separator()
        layout.operator("mesh.dissolve_mode", text="融并")
        layout.menu("VIEW3D_MT_edit_mesh_delete", text="删除")


# ====================================================================== 饼菜单
@_menu
class ProportionalFalloffPie(registry.Menu):
    idname = "VIEW3D_MT_proportional_editing_falloff_pie"
    label = "衰减方式"

    def draw(self, layout, ctx) -> None:
        for value, text in (("SMOOTH", "平滑"), ("SPHERE", "球状"), ("ROOT", "根凸"), ("INVERSE_SQUARE", "平方倒数"),
                            ("SHARP", "锐利"), ("LINEAR", "线性"), ("CONSTANT", "常量"), ("RANDOM", "随机")):
            layout.operator("wm.context_set_enum", text=text, data_path="tool_settings.proportional_edit_falloff",
                            value=value)
