"""图层文件夹和调整层：合成结果和逐层手算的参考值一致。没有文件夹时和原来的混合公式一模一样。
全用填充层（每个纹素结果相同），导出 256² 基础色后取中间一点比较。隐藏上下文，不开窗口。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet  # noqa: E402
from splender.engine import projectio  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402


def to_lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def blend(b, s, mode):
    return {"NORMAL": s, "MULTIPLY": b * s, "SCREEN": 1 - (1 - b) * (1 - s), "ADD": b + s, "SUBTRACT": b - s,
            "DARKEN": np.minimum(b, s), "LIGHTEN": np.maximum(b, s),
            "OVERLAY": np.where(b < 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))}[mode]


def comp(back, back_a, s, a, mode):
    """W3C 合成：返回 (颜色, 覆盖度)。"""
    mixed = (1 - back_a) * s + back_a * blend(back, s, mode)
    ao = a + back_a * (1 - a)
    color = (a * mixed + back_a * (1 - a) * back) / ao if ao > 0 else np.zeros_like(back)
    return color, ao


class FolderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = Path(tempfile.mkdtemp())

    def run_stack(self, layers, base=(0.5, 0.5, 0.5)):
        project = Project("文件夹")
        ts = project.add_texture_set(TextureSet(name="材质", resolution="512", base_color=base))
        for layer in layers:
            ts.layers.append(layer)
            ts._watch(layer)
        mesh = make_test_mesh("plane")
        project.add_object(MeshObject(mesh.name, mesh, [ts.uid]))
        engine = Engine(self.ctx, Preferences(), History())
        try:
            engine.set_project(project)
            engine.settle()
            path = self.tmp / "out.png"
            projectio.export_channel(engine, ts, "basecolor", str(path), size=256)
            image = np.asarray(Image.open(path).convert("RGB")).astype(np.float64) / 255.0
            return image[128, 128]
        finally:
            engine.release()

    def assertColor(self, got, expected, tol=2.5 / 255.0):
        self.assertTrue(np.all(np.abs(np.asarray(got) - np.asarray(expected)) <= tol), "得到 %s，应为 %s" % (got, expected))

    def test_flat_stack_matches_old_formula(self):
        """没有文件夹：和原来的 mix(底, 混合(底, 色), 不透明度) 完全一样。"""
        stack = [("NORMAL", (0.8, 0.2, 0.1), 0.7), ("MULTIPLY", (0.5, 0.9, 0.9), 1.0), ("SCREEN", (0.2, 0.3, 0.6), 0.5),
                 ("OVERLAY", (0.7, 0.7, 0.2), 0.8), ("ADD", (0.1, 0.1, 0.1), 0.6)]
        layers = [Layer(name=m, kind="FILL", fill_color=c, opacity=o, blend_basecolor=m) for m, c, o in stack]
        got = self.run_stack(layers)
        bc = to_lin((0.5, 0.5, 0.5))
        for mode, color, opacity in stack:
            bc = bc + (blend(bc, to_lin(color), mode) - bc) * opacity
        self.assertColor(got, to_srgb(bc))

    def test_folder_opacity_and_blend(self):
        folder = Layer(name="组", kind="FOLDER", opacity=0.5)
        red = Layer(name="红", kind="FILL", fill_color=(0.9, 0.1, 0.1), parent_uid=folder.uid)
        blue = Layer(name="蓝", kind="FILL", fill_color=(0.3, 0.4, 0.9), blend_basecolor="MULTIPLY", opacity=0.8,
                     parent_uid=folder.uid)
        got = self.run_stack([red, blue, folder])
        group, group_a = comp(np.zeros(3), 0.0, to_lin((0.9, 0.1, 0.1)), 1.0, "NORMAL")
        group, group_a = comp(group, group_a, to_lin((0.3, 0.4, 0.9)), 0.8, "MULTIPLY")
        base = to_lin((0.5, 0.5, 0.5))
        result, _ = comp(base, 1.0, group, group_a * 0.5, "NORMAL")
        self.assertColor(got, to_srgb(result))
        # 文件夹之外的同样两层（不透明度直接乘）结果不同：证明是整组合成后再叠
        flat = self.run_stack([Layer(name="红", kind="FILL", fill_color=(0.9, 0.1, 0.1), opacity=0.5),
                               Layer(name="蓝", kind="FILL", fill_color=(0.3, 0.4, 0.9), blend_basecolor="MULTIPLY",
                                     opacity=0.4)])
        self.assertGreater(float(np.abs(flat - got).max()), 0.02)

    def test_folder_blend_mode(self):
        folder = Layer(name="组", kind="FOLDER", blend_basecolor="SCREEN")
        inner = Layer(name="里", kind="FILL", fill_color=(0.2, 0.6, 0.3), parent_uid=folder.uid)
        got = self.run_stack([inner, folder])
        base = to_lin((0.5, 0.5, 0.5))
        self.assertColor(got, to_srgb(blend(base, to_lin((0.2, 0.6, 0.3)), "SCREEN")))

    def test_hidden_folder_and_mask(self):
        folder = Layer(name="组", kind="FOLDER", visible=False)
        inner = Layer(name="里", kind="FILL", fill_color=(1.0, 0.0, 0.0), parent_uid=folder.uid)
        self.assertColor(self.run_stack([inner, folder]), (0.5, 0.5, 0.5))
        folder = Layer(name="组", kind="FOLDER", has_mask=True, mask_default=0.0)
        inner = Layer(name="里", kind="FILL", fill_color=(1.0, 0.0, 0.0), parent_uid=folder.uid)
        self.assertColor(self.run_stack([inner, folder]), (0.5, 0.5, 0.5))
        folder = Layer(name="组", kind="FOLDER", has_mask=True, mask_default=1.0)
        inner = Layer(name="里", kind="FILL", fill_color=(1.0, 0.0, 0.0), parent_uid=folder.uid)
        self.assertColor(self.run_stack([inner, folder]), (1.0, 0.0, 0.0))

    def test_nested_and_empty_folders(self):
        outer = Layer(name="外", kind="FOLDER", opacity=0.5)
        inner = Layer(name="内", kind="FOLDER", parent_uid=outer.uid, opacity=0.5)
        empty = Layer(name="空", kind="FOLDER", parent_uid=outer.uid)
        red = Layer(name="红", kind="FILL", fill_color=(1.0, 0.0, 0.0), parent_uid=inner.uid)
        got = self.run_stack([red, inner, empty, outer])
        base = to_lin((0.5, 0.5, 0.5))
        g_inner, a_inner = comp(np.zeros(3), 0.0, to_lin((1, 0, 0)), 1.0, "NORMAL")
        g_outer, a_outer = comp(np.zeros(3), 0.0, g_inner, a_inner * 0.5, "NORMAL")
        result, _ = comp(base, 1.0, g_outer, a_outer * 0.5, "NORMAL")
        self.assertColor(got, to_srgb(result))

    def test_adjust_layers(self):
        base = (0.6, 0.3, 0.2)
        got = self.run_stack([Layer(name="反相", kind="ADJUST", adjust_type="INVERT")], base=base)
        self.assertColor(got, 1.0 - np.asarray(base))
        got = self.run_stack([Layer(name="去色", kind="ADJUST", adjust_type="HSV", adj_saturation=-1.0)], base=base)
        self.assertColor(got, (0.6, 0.6, 0.6))
        got = self.run_stack([Layer(name="色阶", kind="ADJUST", adjust_type="LEVELS", adj_in_white=0.8)], base=base)
        self.assertColor(got, np.clip(np.asarray(base) / 0.8, 0, 1))
        got = self.run_stack([Layer(name="阈值", kind="ADJUST", adjust_type="THRESHOLD", adj_threshold=0.3)], base=base)
        self.assertColor(got, (1.0, 1.0, 1.0))
        # 一半强度
        got = self.run_stack([Layer(name="反相", kind="ADJUST", adjust_type="INVERT", opacity=0.5)], base=base)
        self.assertColor(got, to_srgb((to_lin(base) + to_lin(1 - np.asarray(base))) * 0.5))

    def test_adjust_inside_folder_only(self):
        """文件夹里的调整层只改文件夹里合出来的东西。"""
        folder = Layer(name="组", kind="FOLDER")
        red = Layer(name="红", kind="FILL", fill_color=(0.8, 0.2, 0.2), opacity=0.5, parent_uid=folder.uid)
        invert = Layer(name="反相", kind="ADJUST", adjust_type="INVERT", parent_uid=folder.uid)
        got = self.run_stack([red, invert, folder], base=(0.5, 0.5, 0.5))
        group = to_lin((0.8, 0.2, 0.2))
        group = to_lin(1 - to_srgb(group))
        result, _ = comp(to_lin((0.5, 0.5, 0.5)), 1.0, group, 0.5, "NORMAL")
        self.assertColor(got, to_srgb(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
