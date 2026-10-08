"""调整层（照 Photoshop 的调整）：每种调整导出的结果和 numpy 照着同样的算法手算的一致。
一个隐藏上下文、一个引擎，改底色和调整参数后导出 256² 取中间一点比较。不开窗口。"""
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

from splender.core import curve, ramp  # noqa: E402
from splender.core.history import History  # noqa: E402
from splender.core.prefs import Preferences  # noqa: E402
from splender.doc.meshio import make_test_mesh  # noqa: E402
from splender.doc.project import Layer, MeshObject, Project, TextureSet  # noqa: E402
from splender.engine import projectio  # noqa: E402
from splender.engine.adjust import TYPE_PROPS  # noqa: E402
from splender.engine.engine import Engine  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])


def luma(c):
    return float(np.dot(c, LUMA))


def clip_color(c):
    c = np.asarray(c, np.float64)
    l, n, x = luma(c), c.min(), c.max()
    if n < 0.0:
        c = l + (c - l) * (l / max(l - n, 1e-6))
    if x > 1.0:
        c = l + (c - l) * ((1.0 - l) / max(x - l, 1e-6))
    return np.clip(c, 0.0, 1.0)


def set_lum(c, l):
    c = np.asarray(c, np.float64)
    return clip_color(c + (l - luma(c)))


def to_lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def cb_map(v, l, sh, mi, hi):
    sh *= np.clip((l - 0.333) / -0.25 + 0.5, 0, 1) * 0.7
    mi *= np.clip((l - 0.333) / 0.25 + 0.5, 0, 1) * np.clip((l + 0.333 - 1.0) / -0.25 + 0.5, 0, 1) * 0.7
    hi *= np.clip((l + 0.333 - 1.0) / 0.25 + 0.5, 0, 1) * 0.7
    return float(np.clip(v + sh + mi + hi, 0, 1))


def bw_gray(c, weights):
    """weights：红、黄、绿、青、蓝、洋红。"""
    red, yellow, green, cyan, blue, magenta = weights
    r, g, b = c
    mn, mx = min(c), max(c)
    md = r + g + b - mn - mx
    if r >= g and r >= b:
        wp, ws = red, (yellow if g >= b else magenta)
    elif g >= b:
        wp, ws = green, (yellow if r >= b else cyan)
    else:
        wp, ws = blue, (magenta if r >= g else cyan)
    return float(np.clip(mn + (md - mn) * ws + (mx - md) * wp, 0, 1))


def selective(c, table, relative):
    """table：九类颜色（红黄绿青蓝洋红白中性黑）的 (青, 洋红, 黄, 黑)。"""
    c = np.asarray(c, np.float64)
    r, g, b = c
    mn, mx = c.min(), c.max()
    md = c.sum() - mn - mx
    w = [mx - md if r >= mx else 0, md - mn if b <= mn else 0, mx - md if g >= mx else 0,
         md - mn if r <= mn else 0, mx - md if b >= mx else 0, md - mn if g <= mn else 0,
         max(mn - 0.5, 0) * 2, max(1 - (abs(mx - 0.5) + abs(mn - 0.5)), 0), max(0.5 - mx, 0) * 2]
    k0 = 1 - mx
    cmy0 = (1 - c - k0) / (1 - k0) if k0 < 0.9999 else np.zeros(3)
    delta = np.zeros(3)
    for k in range(9):
        if w[k] <= 0:
            continue
        a = np.asarray(table[k], np.float64)
        cmy = np.clip(cmy0 * (1 + a[:3]) if relative else cmy0 + a[:3], 0, 1)
        kk = np.clip(k0 * (1 + a[3]) if relative else k0 + a[3], 0, 1)
        delta += ((1 - cmy) * (1 - kk) - c) * w[k]
    return np.clip(c + delta, 0, 1)


class AdjustTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.tmp = Path(tempfile.mkdtemp())
        project = Project("调整")
        cls.ts = project.add_texture_set(TextureSet(name="材质", resolution="512", base_color=(0.5, 0.5, 0.5)))
        cls.layer = Layer(name="调整", kind="ADJUST", adjust_type="CURVES")
        cls.ts.layers.append(cls.layer)
        cls.ts._watch(cls.layer)
        mesh = make_test_mesh("plane")
        project.add_object(MeshObject(mesh.name, mesh, [cls.ts.uid]))
        cls.engine = Engine(cls.ctx, Preferences(), History())
        cls.engine.set_project(project)
        cls.engine.settle()

    @classmethod
    def tearDownClass(cls):
        cls.engine.release()

    def setUp(self):
        layer = self.layer
        for name in layer.properties():
            if name.startswith("adj"):
                layer.reset(name)
        layer.kind = "ADJUST"
        layer.opacity = 1.0
        for name in ("use_basecolor", "use_metallic", "use_roughness"):
            setattr(layer, name, True)

    def adjust(self, base, kind, channel="basecolor", **values):
        ts = self.ts
        ts.base_color = tuple(base)
        self.layer.adjust_type = kind
        for name, value in values.items():
            setattr(self.layer, name, value)
        self.engine.settle()
        path = self.tmp / ("out_%s.png" % channel)
        projectio.export_channel(self.engine, ts, channel, str(path), size=256)
        image = np.asarray(Image.open(path)).astype(np.float64) / 255.0
        return image[128, 128][:3] if image.ndim == 3 else image[128, 128]

    def assertColor(self, got, expected, tol=2.5 / 255.0):
        got, expected = np.asarray(got, np.float64), np.asarray(expected, np.float64)
        self.assertTrue(np.all(np.abs(got - expected) <= tol), "得到 %s，应为 %s" % (got, expected))

    def test_type_props_cover_everything(self):
        """「默认值」预设要恢复的属性都真实存在；每种调整都有一项。"""
        from splender.doc.project import ADJUST_ITEMS

        names = set(Layer.properties())
        self.assertEqual(set(TYPE_PROPS), {item[0] for item in ADJUST_ITEMS})
        for kind, props in TYPE_PROPS.items():
            for name in props:
                self.assertIn(name, names, kind)

    def test_curves(self):
        base = np.array([0.6, 0.3, 0.2])
        master = ((0.0, 0.0), (0.5, 0.7), (1.0, 1.0))
        red = ((0.0, 0.0), (1.0, 0.5))
        got = self.adjust(base, "CURVES", adj_curve=master, adj_curve_r=red)
        expected = [curve.evaluate(master, curve.evaluate(red, base[0])), curve.evaluate(master, base[1]),
                     curve.evaluate(master, base[2])]
        self.assertColor(got, expected)

    def test_curves_on_values(self):
        """数值贴图只过 RGB 总曲线。"""
        self.ts.base_roughness = 0.4
        master = ((0.0, 0.0), (0.4, 0.8), (1.0, 1.0))
        got = self.adjust((0.5, 0.5, 0.5), "CURVES", channel="roughness", adj_curve=master,
                          adj_curve_r=((0.0, 1.0), (1.0, 0.0)))
        self.assertAlmostEqual(float(got), float(curve.evaluate(master, 0.4)), delta=2.5 / 255.0)

    def test_gradient_map(self):
        base = np.array([0.6, 0.3, 0.2])
        value = ("LINEAR", ((0.0, 0.1, 0.0, 0.4, 1.0), (1.0, 1.0, 0.8, 0.2, 1.0)))
        got = self.adjust(base, "GRADIENT_MAP", adj_ramp=value)
        self.assertColor(got, ramp.evaluate(value, luma(base))[:3])
        got = self.adjust(base, "GRADIENT_MAP", adj_ramp=value, adj_ramp_reverse=True)
        self.assertColor(got, ramp.evaluate(value, 1.0 - luma(base))[:3], tol=3.5 / 255.0)
        # 色标半透明：只改一半
        half = ("LINEAR", ((0.0, 1.0, 0.0, 0.0, 0.5), (1.0, 1.0, 0.0, 0.0, 0.5)))
        got = self.adjust(base, "GRADIENT_MAP", adj_ramp=half, adj_ramp_reverse=False)
        self.assertColor(got, base * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5)

    def test_color_balance(self):
        base = np.array([0.55, 0.45, 0.35])
        l = (base.max() + base.min()) / 2
        got = self.adjust(base, "COLOR_BALANCE", adj_cb_midtones_r=0.5, adj_cb_shadows_b=0.3, adj_cb_preserve=False)
        expected = np.array([cb_map(base[0], l, 0.0, 0.5, 0.0), cb_map(base[1], l, 0.0, 0.0, 0.0),
                             cb_map(base[2], l, 0.3, 0.0, 0.0)])
        self.assertColor(got, expected)
        got = self.adjust(base, "COLOR_BALANCE", adj_cb_preserve=True)
        self.assertColor(got, set_lum(expected, luma(base)))

    def test_posterize(self):
        got = self.adjust((0.6, 0.3, 0.1), "POSTERIZE", adj_posterize=4)
        self.assertColor(got, (2 / 3, 1 / 3, 0.0))

    def test_vibrance(self):
        base = np.array([0.6, 0.45, 0.4])
        for vib, sat in ((0.8, 0.0), (-0.5, 0.3)):
            got = self.adjust(base, "VIBRANCE", adj_vibrance=vib, adj_vib_saturation=sat)
            l = luma(base)
            s = base.max() - base.min()
            self.assertColor(got, clip_color(l + (base - l) * ((1 + vib * (1 - s)) * (1 + sat))))
        # 灰色不受影响
        self.assertColor(self.adjust((0.4, 0.4, 0.4), "VIBRANCE", adj_vibrance=1.0), (0.4, 0.4, 0.4))

    def test_photo_filter(self):
        base = np.array([0.5, 0.5, 0.5])
        color = np.array([236, 138, 0]) / 255.0
        got = self.adjust(base, "PHOTO_FILTER")
        mixed = base + (base * color - base) * 0.25
        self.assertColor(got, set_lum(mixed, luma(base)))
        got = self.adjust(base, "PHOTO_FILTER", adj_filter_preserve=False, adj_filter_density=0.6)
        self.assertColor(got, base + (base * color - base) * 0.6)

    def test_exposure(self):
        base = np.array([0.4, 0.3, 0.2])
        got = self.adjust(base, "EXPOSURE", adj_exposure=1.0)
        self.assertColor(got, to_srgb(to_lin(base) * 2.0))
        got = self.adjust(base, "EXPOSURE", adj_exposure=-1.0, adj_exp_offset=0.02, adj_exp_gamma=1.5)
        self.assertColor(got, to_srgb(np.maximum(to_lin(base) * 0.5 + 0.02, 0.0) ** (1 / 1.5)))

    def test_exposure_on_values(self):
        self.ts.base_metallic = 0.3
        got = self.adjust((0.5, 0.5, 0.5), "EXPOSURE", channel="metallic", adj_exposure=1.0)
        self.assertAlmostEqual(float(got), 0.6, delta=2.5 / 255.0)

    def test_black_white(self):
        base = np.array([0.6, 0.3, 0.2])
        got = self.adjust(base, "BLACK_WHITE")
        gray = bw_gray(base, (0.4, 0.6, 0.4, 0.6, 0.2, 0.8))
        self.assertColor(got, (gray, gray, gray))
        weights = (1.2, 0.3, -0.5, 0.6, 0.9, 0.1)
        got = self.adjust(base, "BLACK_WHITE", adj_bw_red=1.2, adj_bw_yellow=0.3, adj_bw_green=-0.5, adj_bw_cyan=0.6,
                          adj_bw_blue=0.9, adj_bw_magenta=0.1, adj_bw_tint=True)
        gray = bw_gray(base, weights)
        self.assertColor(got, set_lum((0.6, 0.54, 0.4), gray))
        # 纯色各自按自己那一项
        self.assertColor(self.adjust((1.0, 1.0, 0.0), "BLACK_WHITE", adj_bw_yellow=0.25, adj_bw_tint=False),
                         (0.25, 0.25, 0.25))

    def test_channel_mixer(self):
        base = np.array([0.6, 0.3, 0.2])
        got = self.adjust(base, "CHANNEL_MIXER", adj_mix_rr=0.0, adj_mix_rb=1.0, adj_mix_br=1.0, adj_mix_bb=0.0)
        self.assertColor(got, (0.2, 0.3, 0.6))
        got = self.adjust(base, "CHANNEL_MIXER", adj_mix_mono=True, adj_mix_rr=0.4, adj_mix_rg=0.4, adj_mix_rb=0.2,
                          adj_mix_rc=0.05)
        gray = 0.6 * 0.4 + 0.3 * 0.4 + 0.2 * 0.2 + 0.05
        self.assertColor(got, (gray, gray, gray))

    def test_selective_color(self):
        base = np.array([0.8, 0.3, 0.2])
        table = np.zeros((9, 4))
        table[0] = (-0.5, 0.2, 0.0, 0.1)
        table[7] = (0.0, 0.0, 0.3, 0.2)
        values = {"adj_sel_reds_c": -0.5, "adj_sel_reds_m": 0.2, "adj_sel_reds_k": 0.1, "adj_sel_neutrals_y": 0.3,
                  "adj_sel_neutrals_k": 0.2}
        got = self.adjust(base, "SELECTIVE_COLOR", **values)
        self.assertColor(got, selective(base, table, True))
        got = self.adjust(base, "SELECTIVE_COLOR", adj_sel_method="ABSOLUTE", **values)
        self.assertColor(got, selective(base, table, False))

    def test_colorize(self):
        base = np.array([0.6, 0.3, 0.2])
        got = self.adjust(base, "HSV", adj_colorize=True, adj_colorize_hue=200.0, adj_colorize_sat=0.5)
        import colorsys

        tint = np.array(colorsys.hsv_to_rgb(200 / 360.0, 0.5, 1.0))
        self.assertColor(got, set_lum(tint, luma(base)))

    def test_levels_per_channel(self):
        """分通道色阶：先过自己通道的，再过 RGB 的；数值贴图只过 RGB 的。"""
        from splender.engine.adjust import levels

        base = np.array([0.6, 0.3, 0.2])
        got = self.adjust(base, "LEVELS", adj_in_white=0.8, adj_lv_r_gamma=2.0, adj_lv_b_out_white=0.5)
        expected = [levels(levels(base[0], 0, 1, 2.0, 0, 1), 0, 0.8, 1, 0, 1),
                    levels(base[1], 0, 0.8, 1, 0, 1),
                    levels(levels(base[2], 0, 1, 1, 0, 0.5), 0, 0.8, 1, 0, 1)]
        self.assertColor(got, np.asarray(expected, np.float64).reshape(3))
        self.ts.base_roughness = 0.4
        got = self.adjust(base, "LEVELS", channel="roughness", adj_in_white=0.8, adj_lv_r_gamma=2.0)
        self.assertAlmostEqual(float(got), 0.5, delta=2.5 / 255.0)

    def test_hue_ranges(self):
        """色相饱和度只改某一类颜色：红色范围转色相，绿色不受影响，灰色不受影响。"""
        import colorsys

        red = np.array([0.8, 0.2, 0.2])
        got = self.adjust(red, "HSV", adj_hsv_reds_hue=120.0)
        h, s, v = colorsys.rgb_to_hsv(*red)
        self.assertColor(got, colorsys.hsv_to_rgb((h + 120 / 360.0) % 1.0, s, v))
        green = np.array([0.2, 0.7, 0.25])
        self.assertColor(self.adjust(green, "HSV", adj_hsv_reds_hue=120.0), green)
        gray = np.array([0.5, 0.5, 0.5])
        self.assertColor(self.adjust(gray, "HSV", adj_hsv_reds_light=-0.8), gray)
        # 介于红和黄之间（30°）：两类各占一半
        orange = np.array(colorsys.hsv_to_rgb(30 / 360.0, 0.7, 0.8))
        got = self.adjust(orange, "HSV", adj_hsv_reds_hue=0.0, adj_hsv_reds_light=0.0, adj_hsv_reds_sat=-1.0)
        self.assertColor(got, colorsys.hsv_to_rgb(30 / 360.0, 0.7 * 0.5, 0.8))

    def test_composite_below_layer(self):
        """调整层的直方图看它下面的结果：反相层下面还是原色。"""
        from splender.engine.adjust import histogram
        from splender.engine.projectio import composite_image

        self.adjust((0.6, 0.6, 0.6), "INVERT")
        full = composite_image(self.engine, self.ts, 256)
        below = composite_image(self.engine, self.ts, 256, below=self.layer)
        self.assertEqual(full.shape, (256, 256, 3))
        self.assertAlmostEqual(float(full.mean()), 0.4, delta=2.5 / 255.0)
        self.assertAlmostEqual(float(below.mean()), 0.6, delta=2.5 / 255.0)
        counts = histogram(below)
        self.assertEqual(int(counts[3].sum()), 256 * 256)
        self.assertEqual(int(np.argmax(counts[0])), 153)

    def test_auto_levels(self):
        from splender.engine.adjust import auto_levels, levels

        rng = np.random.default_rng(3)
        image = np.stack([rng.uniform(0.2, 0.7, 20000), rng.uniform(0.1, 0.9, 20000),
                          rng.uniform(0.3, 0.6, 20000)], axis=1)
        tone = auto_levels(image, "TONE", 0.0)
        self.assertAlmostEqual(tone["adj_lv_r_in_black"], image[:, 0].min(), places=6)
        self.assertAlmostEqual(tone["adj_lv_b_in_white"], image[:, 2].max(), places=6)
        contrast = auto_levels(image, "CONTRAST", 0.0)
        self.assertAlmostEqual(contrast["adj_in_black"], image.min(), places=6)
        self.assertAlmostEqual(contrast["adj_in_white"], image.max(), places=6)
        # 自动颜色：中间调各通道平均值拉到一起
        tinted = np.clip(np.stack([image[:, 1] * 0.9 + 0.08, image[:, 1], image[:, 1] * 0.8], axis=1), 0, 1)
        color = auto_levels(tinted, "COLOR", 0.001)
        out = np.stack([levels(tinted[:, k], color["adj_lv_%s_in_black" % c], color["adj_lv_%s_in_white" % c],
                               color.get("adj_lv_%s_gamma" % c, 1.0), 0, 1) for k, c in enumerate("rgb")], axis=1)
        lum = out @ np.array([0.2126, 0.7152, 0.0722])
        mid = out[(lum > 0.2) & (lum < 0.8)].mean(axis=0)
        self.assertLess(float(mid.max() - mid.min()), 0.03, mid)
        # 几乎没有起伏：不拉
        flat = np.full((100, 3), 0.5)
        self.assertEqual(auto_levels(flat, "CONTRAST"), {"adj_in_black": 0.0, "adj_in_white": 1.0})

    def test_color_lookup_file(self):
        """导入的 .cube：三维按三线性插值，一维各通道各查各的；输入范围不是 0..1 也对。"""
        from splender.engine import cube

        size = 5
        g = cube.grid(size)
        data = np.stack([g[:, 2], g[:, 1] ** 0.5, g[:, 0]], axis=1)
        lines = ['TITLE "swap"', "LUT_3D_SIZE 5"] + ["%.6f %.6f %.6f" % tuple(r) for r in data]
        found = cube.parse_cube("\n".join(lines))
        self.assertEqual((found["dims"], found["size"], found["title"]), (3, 5, "swap"))
        base = np.array([0.6, 0.3, 0.2])
        got = self.adjust(base, "COLOR_LOOKUP", adj_lut_look="FILE", adj_lut_data=cube.encode(found))
        self.assertColor(got, cube.apply(found, base[None])[0], tol=3.0 / 255.0)
        self.assertColor(got, (0.2, 0.3 ** 0.5, 0.6), tol=0.03)
        one = cube.parse_cube("\n".join(["LUT_1D_SIZE 3", "DOMAIN_MIN 0 0 0", "DOMAIN_MAX 2 2 2", "0 0 0",
                                         "0.2 0.5 0.8", "1 1 1"]))
        got = self.adjust(base, "COLOR_LOOKUP", adj_lut_data=cube.encode(one))
        self.assertColor(got, cube.apply(one, base[None])[0], tol=3.0 / 255.0)
        for bad in (["0 0 0"], ["LUT_3D_SIZE 2", "0 0 0"], ["LUT_3D_SIZE 99"]):
            with self.assertRaises(cube.CubeError):
                cube.parse_cube("\n".join(bad))

    def test_color_lookup_builtin(self):
        from splender.doc.project import LOOK_ITEMS
        from splender.engine import cube

        base = np.array([0.6, 0.3, 0.2])
        for look, _label, _desc in LOOK_ITEMS:
            if look == "FILE":
                continue
            got = self.adjust(base, "COLOR_LOOKUP", adj_lut_look=look)
            expected = cube.apply(cube.builtin(look), base[None])[0]
            self.assertColor(got, expected, tol=3.0 / 255.0)
            self.assertGreater(float(np.abs(expected - base).max()), 0.01, look)

    def test_cube_round_trip(self):
        from splender.engine import cube

        found = cube.builtin("TEAL_ORANGE", 9)
        path = self.tmp / "look.cube"
        cube.write_cube(str(path), found["data"], 9, "青橙")
        again = cube.parse_cube(path.read_text(encoding="utf-8"))
        self.assertEqual(again["size"], 9)
        self.assertTrue(np.allclose(again["data"], found["data"], atol=1e-5))
        decoded = cube.decode(cube.encode(again))
        self.assertTrue(np.allclose(decoded["data"], found["data"], atol=2e-3))

    def test_two_tables_stacked(self):
        """两层都用查找表（曲线、渐变映射）：各用各的那一段。"""
        second = Layer(name="渐变", kind="ADJUST", adjust_type="GRADIENT_MAP",
                       adj_ramp=("LINEAR", ((0.0, 0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 1.0, 0.0, 1.0))))
        self.ts.add_layer(second)
        try:
            base = np.array([0.6, 0.3, 0.2])
            master = ((0.0, 0.0), (1.0, 0.5))
            got = self.adjust(base, "CURVES", adj_curve=master)
            first = np.array([curve.evaluate(master, v) for v in base])
            self.assertColor(got, ramp.evaluate(second.adj_ramp, luma(first))[:3])
        finally:
            self.ts.remove_layer(second)

    def test_random_parity_with_cpu(self):
        """每种调整随机取参数：着色器的结果和 CPU 实现（engine/adjust.apply_color、scalar_function）一样。
        阈值、色调分离是台阶，单独测过，这里不随机。"""
        from splender.core.props import (BoolProperty, ColorProperty, CurveProperty, EnumProperty, FloatProperty,
                                         IntProperty, RampProperty)
        from splender.doc.project import ADJUST_ITEMS
        from splender.engine.adjust import apply_color, scalar_function

        rng = np.random.default_rng(11)
        colors = [np.array([0.62, 0.31, 0.18]), np.array([0.2, 0.55, 0.7]), np.array([0.47, 0.45, 0.41])]
        props = Layer.properties()
        for kind, _label, _desc in ADJUST_ITEMS:
            if kind in ("THRESHOLD", "POSTERIZE"):
                continue
            for trial in range(2):
                values = {}
                for name in TYPE_PROPS[kind]:
                    prop = props[name]
                    if isinstance(prop, BoolProperty):
                        values[name] = bool(rng.random() < 0.5)
                    elif isinstance(prop, ColorProperty):
                        values[name] = tuple(float(v) for v in rng.uniform(0.1, 0.9, 3))
                    elif isinstance(prop, FloatProperty):
                        lo, hi = max(prop.soft_min, -1.0), min(prop.soft_max, 2.0)
                        if name.endswith("gamma"):
                            lo, hi = 0.5, 2.0
                        values[name] = float(rng.uniform(lo, hi) * 0.5 + (lo + hi) * 0.25)
                    elif isinstance(prop, IntProperty):
                        values[name] = int(rng.integers(prop.min, min(prop.max, 16) + 1))
                    elif isinstance(prop, CurveProperty):
                        xs = np.sort(rng.uniform(0.1, 0.9, 2))
                        values[name] = ((0.0, float(rng.uniform(0, 0.2))), (float(xs[0]), float(rng.uniform(0.2, 0.5))),
                                        (float(xs[1]), float(rng.uniform(0.5, 0.8))), (1.0, float(rng.uniform(0.8, 1.0))))
                    elif isinstance(prop, RampProperty):
                        stops = tuple((float(p), *(float(v) for v in rng.uniform(0, 1, 3)), float(rng.uniform(0.3, 1)))
                                      for p in sorted(rng.uniform(0, 1, 3)))
                        values[name] = (["LINEAR", "EASE", "CARDINAL", "B_SPLINE"][trial * 2 % 4], stops)
                    elif isinstance(prop, EnumProperty) and name == "adj_lut_look":
                        values[name] = ["TEAL_ORANGE", "CROSS", "NIGHT", "BLEACH"][int(rng.integers(0, 4))]
                    elif isinstance(prop, EnumProperty):
                        keys = [item[0] for item in prop.items()]
                        values[name] = keys[int(rng.integers(0, len(keys)))]
                if kind == "HSV" and trial == 0:
                    values["adj_colorize"] = False
                for name in TYPE_PROPS[kind]:
                    self.layer.reset(name)
                for base in colors:
                    got = self.adjust(base, kind, **values)
                    expected = apply_color(self.layer, base[None])[0]
                    self.assertTrue(np.all(np.abs(got - expected) <= 3.5 / 255.0),
                                    "%s 第 %d 组：得到 %s，应为 %s（%s）" % (kind, trial, got, expected, values))
                self.ts.base_roughness = 0.37
                got = self.adjust(colors[0], kind, channel="roughness")
                self.assertAlmostEqual(float(got), float(scalar_function(self.layer, 0.37)), delta=2.5 / 255.0,
                                       msg="%s 的粗糙度" % kind)

    def test_non_adjust_layers_do_not_save_adjust_values(self):
        paint = Layer(name="画", kind="PAINT")
        self.assertFalse(any(key.startswith("adj") for key in paint.to_dict()))
        self.layer.adj_sel_reds_c = 0.3
        data = self.layer.to_dict()
        self.assertEqual(data["adj_sel_reds_c"], 0.3)
        copy = Layer.from_saved(dict(data, uid=0))
        self.assertEqual(copy.adj_sel_reds_c, 0.3)
        self.assertEqual(copy.adj_curve, self.layer.adj_curve)


if __name__ == "__main__":
    unittest.main(verbosity=2)
