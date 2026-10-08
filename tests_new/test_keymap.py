"""键位表自检：每一条都指向存在的操作、属性名和取值都对、弹出的菜单都存在、切换的工具都存在、
同一张表里没有两条一模一样的按键（后一条永远轮不到）。"""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
from PySide6.QtWidgets import QApplication  # noqa: E402

QT = QApplication.instance() or QApplication(sys.argv)

from splender import editors  # noqa: E402,F401
from splender.core import ops, registry  # noqa: E402
from splender.core.keymap import default_keyconfig  # noqa: E402
from splender.ops import (addon_ops, adjust_ops, bake_ops, edit_ops, file_ops, filter_ops, layer_ops,  # noqa: E402,F401
                          mesh_edit_ops, mesh_edit_tools, mesh_ops, node_ops, object_ops, paint_ops, ps_ops, render_ops,
                          screen_ops, sculpt_ops, select_ops, shader_ops, texture_set_ops, tool_ops, view3d_ops,
                          wm_ops)
from splender.tools import mesh_tools, object_tools, paint_tools, sculpt_tools  # noqa: E402,F401
from splender.ui import menus, menus_edit_mesh, menus_shader, menus_view3d  # noqa: E402,F401


class KeymapTest(unittest.TestCase):
    def setUp(self):
        self.kc = default_keyconfig()

    def test_operators_and_props_exist(self):
        problems = []
        for name, km in self.kc.keymaps.items():
            for item in km.items:
                cls = ops.get(item.op)
                if cls is None:
                    problems.append("%s：%s 没有这个操作" % (name, item.op))
                    continue
                props = cls.properties()
                for key, value in dict(item.props).items():
                    prop = props.get(key)
                    if prop is None:
                        problems.append("%s：%s 没有属性 %s" % (name, item.op, key))
                        continue
                    if getattr(prop, "kind", "") == "ENUM" and not callable(prop._items):
                        idents = [i[0] for i in prop.items()]
                        if value not in idents:
                            problems.append("%s：%s.%s 没有 %r 这一项" % (name, item.op, key, value))
        self.assertEqual(problems, [], "\n".join(problems))

    def test_menus_and_tools_exist(self):
        problems = []
        tool_ids = {t.idname for space in ("VIEW_3D", "IMAGE_EDITOR") for t in registry.tools(space)}
        for name, km in self.kc.keymaps.items():
            for item in km.items:
                props = dict(item.props)
                if item.op in ("wm.call_menu", "wm.call_menu_pie") and registry.menu(props.get("name", "")) is None:
                    problems.append("%s：没有菜单 %s" % (name, props.get("name")))
                if item.op == "paint.tool_set" and props.get("tool") not in tool_ids:
                    problems.append("%s：没有工具 %s" % (name, props.get("tool")))
                if item.op in ("wm.context_toggle", "wm.context_set_enum"):
                    root = str(props.get("data_path", "")).split(".")[0]
                    if root not in ("tool_settings", "space_data", "preferences", "scene"):
                        problems.append("%s：数据路径 %s 找不到" % (name, props.get("data_path")))
        # 工具栏里每个工具用到的键位表都存在
        for space in ("VIEW_3D", "IMAGE_EDITOR"):
            for tool in registry.tools(space):
                if tool.keymap and tool.keymap not in self.kc.keymaps:
                    problems.append("工具 %s 的键位表 %s 不存在" % (tool.idname, tool.keymap))
        self.assertEqual(problems, [], "\n".join(problems))

    def test_no_dead_duplicates(self):
        problems = []
        for name, km in self.kc.keymaps.items():
            seen = {}
            for item in km.items:
                key = (item.type, item.value, item.ctrl, item.shift, item.alt, item.any_mod)
                if key in seen:
                    problems.append("%s：%s 和 %s 用了同一个键 %s" % (name, seen[key], item.op, item.label()
                                                             if hasattr(item, "label") else key))
                else:
                    seen[key] = item.op
        self.assertEqual(problems, [], "\n".join(problems))

    def test_edit_mode_blender_keys(self):
        """编辑模式里 Blender 的常用键都在。"""
        mesh = self.kc.keymaps["Mesh"]
        have = {(i.type, i.ctrl, i.shift, i.alt): i.op for i in mesh.items if i.value in ("PRESS", "CLICK")}
        expect = {("E", False, False, False): "view3d.edit_mesh_extrude_move_normal",
                  ("I", False, False, False): "mesh.inset",
                  ("B", True, False, False): "mesh.bevel",
                  ("R", True, False, False): "mesh.loopcut_slide",
                  ("K", False, False, False): "mesh.knife_tool",
                  ("J", False, False, False): "mesh.vert_connect_path",
                  ("V", False, False, False): "mesh.rip_move",
                  ("F", False, False, False): "mesh.edge_face_add",
                  ("1", False, False, False): "mesh.select_mode",
                  ("G", False, False, False): "transform.translate"}
        for key, op in expect.items():
            self.assertEqual(have.get(key), op, key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
