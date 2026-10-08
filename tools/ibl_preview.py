"""环境光与 PBR 着色的验证图。全部在不可见的显卡上下文里渲染，输出到 docs/evidence/ibl/。

    python tools/ibl_preview.py            # 全部
    python tools/ibl_preview.py grid sun   # 只出某几类：grid sun studio matcap levels furnace tonemap blur timing

画面里的球用光线求交直接算，着色走 splender.engine.ibl 的 GLSL（和视口用的是同一份代码）。
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()

import moderngl  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from splender.engine import ibl  # noqa: E402

OUT = ROOT / "docs" / "evidence" / "ibl"
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
BLENDER_CM = Path("C:/Program Files/Blender Foundation/Blender 4.3/4.3/datafiles/colormanagement")
TRANSFORM_NAMES = {0: "标准", 1: "Filmic", 2: "AgX"}

SCENE_FS = "#version 430\n" + ibl.GLSL_ENV + ibl.GLSL_PBR + ibl.GLSL_TONEMAP + ibl.GLSL_STUDIO + ibl.GLSL_MATCAP + """
uniform vec2 u_res;
uniform vec3 u_cam_pos;
uniform mat3 u_cam_rot;        // 列：右、上、后（相机看向 -后）
uniform float u_tan_half;
uniform int u_ortho;
uniform float u_ortho_half;
uniform int u_count;
uniform vec4 u_spheres[25];    // xyz 球心，w 半径
uniform vec4 u_albedo[25];     // rgb 线性基础色，a 金属度
uniform float u_rough[25];
uniform int u_mode;            // 0 PBR，1 工作室光，2 Matcap，3 全景展开
uniform int u_transform;
uniform float u_exposure;
uniform float u_blur;
uniform float u_bg_mix;
uniform vec3 u_bg_color;
uniform int u_linear_out;
out vec4 frag;

void main() {
    vec2 pix = gl_FragCoord.xy;
    vec2 ndc = pix / u_res * 2.0 - 1.0;
    float aspect = u_res.x / u_res.y;
    vec3 ro;
    vec3 rd;
    if (u_ortho == 1) {
        ro = u_cam_pos + u_cam_rot * vec3(ndc.x * aspect * u_ortho_half, ndc.y * u_ortho_half, 0.0);
        rd = -u_cam_rot[2];
    } else {
        ro = u_cam_pos;
        rd = normalize(u_cam_rot * vec3(ndc.x * aspect * u_tan_half, ndc.y * u_tan_half, -1.0));
    }
    vec3 color;
    if (u_mode == 3) {
        color = env_background(env_uv_to_dir(pix / u_res), u_blur);
    } else {
        vec3 bg = mix(u_bg_color, env_background(rd, u_blur), u_bg_mix);
        float tbest = 1e30;
        int hit = -1;
        for (int i = 0; i < u_count; ++i) {
            vec3 oc = ro - u_spheres[i].xyz;
            float b = dot(oc, rd);
            float c = dot(oc, oc) - u_spheres[i].w * u_spheres[i].w;
            float disc = b * b - c;
            if (disc > 0.0) {
                float t = -b - sqrt(disc);
                if (t > 0.0 && t < tbest) {
                    tbest = t;
                    hit = i;
                }
            }
        }
        if (hit < 0) {
            color = bg;
        } else {
            vec3 p = ro + rd * tbest;
            vec3 n = normalize(p - u_spheres[hit].xyz);
            vec3 v = -rd;
            vec3 base = u_albedo[hit].rgb;
            if (u_mode == 0) {
                color = shade_pbr(n, v, base, u_albedo[hit].a, u_rough[hit], 1.0);
            } else {
                mat3 to_view = transpose(u_cam_rot);
                vec3 nv = to_view * n;
                vec3 vv = to_view * v;
                color = u_mode == 1 ? shade_studio(nv, vv, base) : shade_matcap(nv, vv, base);
            }
        }
    }
    frag = u_linear_out == 1 ? vec4(color, 1.0) : vec4(tonemap(color, u_transform, u_exposure), 1.0);
}
"""


# ----------------------------------------------------------------------
# 渲染
# ----------------------------------------------------------------------

def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    return np.stack([right, upv, -fwd], axis=1)      # 列：右、上、后


class Renderer:
    def __init__(self) -> None:
        self.ctx = moderngl.create_standalone_context(require=430)
        self.env = ibl.Environment(self.ctx)
        self.prog = self.ctx.program(vertex_shader=ibl._VS_FULLSCREEN, fragment_shader=SCENE_FS)
        self.vao = self.ctx.vertex_array(self.prog, [])
        self.matcap_tex = None
        self.current_hdri = None

    def set_hdri(self, name: str) -> float:
        if self.current_hdri == name:
            return 0.0
        t = time.perf_counter()
        self.env.set_hdri(name)
        self.current_hdri = name
        return (time.perf_counter() - t) * 1000.0

    def set_matcap(self, name: str) -> None:
        if self.matcap_tex is not None:
            self.matcap_tex.release()
        self.matcap_tex = ibl.create_matcap_texture(self.ctx, ibl.load_matcap(name))

    def render(self, size, *, cam_pos=(0, 0, 10), cam_rot=None, fov_deg=30.0, ortho_half=None, spheres=(),
               mode=0, transform=2, exposure=0.0, blur=0.0, bg_mix=1.0, bg_color=(0.05, 0.05, 0.05),
               linear=False, ss=2, rotation=0.0, strength=1.0, studio=None) -> np.ndarray:
        """返回 (H, W, 3) float32，行序自上而下。linear=True 时是线性亮度，否则是 sRGB 编码值。"""
        w, h = size
        rw, rh = w * ss, h * ss
        tex = self.ctx.texture((rw, rh), 4, dtype="f4")
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        p = self.prog
        p["u_res"].value = (float(rw), float(rh))
        rot = look_at(cam_pos, (0, 0, 0)) if cam_rot is None else np.asarray(cam_rot, float)
        set_ = lambda name, value: ibl._set_uniform(p, name, value)  # noqa: E731
        set_("u_cam_pos", tuple(float(x) for x in cam_pos))
        p["u_cam_rot"].write(np.asarray(rot, "f4").T.copy().tobytes())
        set_("u_tan_half", math.tan(math.radians(fov_deg) * 0.5))
        set_("u_ortho", 1 if ortho_half else 0)
        set_("u_ortho_half", float(ortho_half or 1.0))
        sph = np.zeros((25, 4), "f4")
        alb = np.zeros((25, 4), "f4")
        rough = np.zeros(25, "f4")
        for i, s in enumerate(spheres):
            sph[i] = (*s["center"], s.get("radius", 1.0))
            alb[i] = (*s["base"], s.get("metallic", 0.0))
            rough[i] = s.get("roughness", 0.5)
        set_("u_count", len(spheres))
        ibl._write_uniform(p, "u_spheres", sph)
        ibl._write_uniform(p, "u_albedo", alb)
        ibl._write_uniform(p, "u_rough", rough)
        set_("u_mode", mode)
        set_("u_transform", transform)
        set_("u_exposure", float(exposure))
        set_("u_blur", float(blur))
        set_("u_bg_mix", float(bg_mix))
        set_("u_bg_color", tuple(float(x) for x in bg_color))
        set_("u_linear_out", 1 if linear else 0)
        unit = self.env.bind(p, 0, rotation=rotation, strength=strength)
        if self.matcap_tex is not None:
            ibl.bind_matcap(p, self.matcap_tex, unit)
        else:
            set_("u_matcap", unit)
            self.env.bg_texture.use(location=unit)
        if studio is not None:
            ibl.bind_studio(p, studio)
        with self.ctx.scope(fbo, enable_only=moderngl.NOTHING):
            self.vao.render(mode=moderngl.TRIANGLES, vertices=3)
        data = np.frombuffer(tex.read(), "f4").reshape(rh, rw, 4)[..., :3]
        fbo.release()
        tex.release()
        img = np.flipud(data)
        if ss > 1:
            img = img.reshape(h, ss, w, ss, 3).mean(axis=(1, 3))
        return np.ascontiguousarray(img, dtype=np.float32)

    def project(self, point, cam_pos, cam_rot, fov_deg, size) -> np.ndarray:
        """世界坐标点 → 像素坐标 (x, y)，y 自上而下。"""
        w, h = size
        local = np.asarray(cam_rot).T @ (np.asarray(point, float) - np.asarray(cam_pos, float))
        t = math.tan(math.radians(fov_deg) * 0.5)
        ndc_x = local[0] / (-local[2]) / (t * w / h)
        ndc_y = local[1] / (-local[2]) / t
        return np.array([(ndc_x + 1) * 0.5 * w, (1 - ndc_y) * 0.5 * h])


# ----------------------------------------------------------------------
# 图像输出与标注
# ----------------------------------------------------------------------

_FONTS: dict[int, ImageFont.FreeTypeFont] = {}


def font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.truetype(FONT_PATH, size)
        except OSError:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def to_image(srgb: np.ndarray, seed: int = 0) -> Image.Image:
    """sRGB 浮点 → 8 位，加三角分布抖动。"""
    rng = np.random.default_rng(seed)
    noise = (rng.random(srgb.shape) + rng.random(srgb.shape) - 1.0) / 255.0
    return Image.fromarray(ibl.to_uint8(np.clip(srgb + noise, 0, 1)), "RGB")


def text(draw: ImageDraw.ImageDraw, xy, content: str, size=20, fill=(235, 235, 235), anchor="la", shadow=True):
    f = font(size)
    if shadow:
        draw.text((xy[0] + 1, xy[1] + 1), content, font=f, fill=(0, 0, 0), anchor=anchor)
    draw.text(xy, content, font=f, fill=fill, anchor=anchor)


def band(img: Image.Image, box, alpha=150):
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(box, fill=(18, 18, 18, alpha))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


# ----------------------------------------------------------------------
# a. 材质球阵列
# ----------------------------------------------------------------------

ROUGHNESS = [0.05 + i * 0.95 / 4 for i in range(5)]
METALLIC = [i / 4 for i in range(5)]
COLORED = tuple(float(x) for x in ibl.srgb_decode(np.array([0.78, 0.47, 0.30])))   # 中等饱和的赤陶色
WHITE = (0.8, 0.8, 0.8)


def grid_spheres(base) -> list[dict]:
    out = []
    for j, m in enumerate(METALLIC):
        for i, r in enumerate(ROUGHNESS):
            out.append({"center": ((i - 2) * 2.3, (2 - j) * 2.3 - 0.35, 0.0), "radius": 1.0,
                        "base": base, "metallic": m, "roughness": r})
    return out


def render_grid(rd: Renderer, hdri: str, base, base_name: str, transform: int, blur: float, path: Path) -> None:
    size = (1600, 1000)
    cam = (0.0, 0.0, 24.0)
    img = rd.render(size, cam_pos=cam, fov_deg=30.0, spheres=grid_spheres(base), transform=transform, blur=blur)
    im = band(to_image(img), (0, 0, size[0], 58))
    d = ImageDraw.Draw(im)
    text(d, (16, 14), f"环境 {hdri}（{ibl.label(hdri)}）· 显示变换 {TRANSFORM_NAMES[transform]} · 背景模糊 {blur:g}"
                      f" · 基础色 {base_name}", 22)
    rot = look_at(cam, (0, 0, 0))
    for i, r in enumerate(ROUGHNESS):
        x, _ = rd.project(((i - 2) * 2.3, 0, 0), cam, rot, 30.0, size)
        text(d, (x, 50), f"粗糙度 {r:.2f}", 18, anchor="mb")
    for j, m in enumerate(METALLIC):
        _, y = rd.project((0, (2 - j) * 2.3 - 0.35, 0), cam, rot, 30.0, size)
        x0, _ = rd.project((-2 * 2.3 - 1.25, 0, 0), cam, rot, 30.0, size)
        text(d, (x0, y), f"金属度 {m:.2f}", 18, anchor="rm")
    im.save(path)


def do_grid(rd: Renderer) -> list[Path]:
    paths = []
    for hdri in ("forest", "studio"):
        rd.set_hdri(hdri)
        for transform in (0, 1, 2):
            for blur in (0.0, 0.5):
                p = OUT / f"grid_{hdri}_{['standard', 'filmic', 'agx'][transform]}_blur{int(blur * 10)}_color.png"
                render_grid(rd, hdri, COLORED, "赤陶色", transform, blur, p)
                paths.append(p)
        for blur in (0.0, 0.5):
            p = OUT / f"grid_{hdri}_agx_blur{int(blur * 10)}_white.png"
            render_grid(rd, hdri, WHITE, "白色", 2, blur, p)
            paths.append(p)
    return paths


# ----------------------------------------------------------------------
# b. 太阳方向
# ----------------------------------------------------------------------

def highlight_point(center, radius, eye, sun, iterations=20) -> np.ndarray:
    """镜面球上反射太阳的点：法线 = normalize(太阳方向 + 视线方向)，迭代处理透视。"""
    c = np.asarray(center, float)
    p = c + radius * np.asarray(sun)
    for _ in range(iterations):
        v = np.asarray(eye, float) - p
        v /= np.linalg.norm(v)
        n = sun + v
        n /= np.linalg.norm(n)
        p = c + radius * n
    return p


def sun_view(rd: Renderer, sun: np.ndarray, yaw_offset_deg: float, pitch_deg: float, size=(800, 560)):
    """相机朝向 = 太阳水平方位 + 偏角；两个球放在相机前方。返回 (图, 相机信息, 球列表)。"""
    sh = np.array([sun[0], 0.0, sun[2]])
    sh /= np.linalg.norm(sh)
    yaw = math.radians(yaw_offset_deg)
    fwd_h = np.array([sh[0] * math.cos(yaw) - sh[2] * math.sin(yaw), 0.0, sh[0] * math.sin(yaw) + sh[2] * math.cos(yaw)])
    pitch = math.radians(pitch_deg)
    fwd = fwd_h * math.cos(pitch) + np.array([0, 1, 0]) * math.sin(pitch)
    eye = np.array([0.0, 0.0, 0.0])
    center = eye + fwd_h * 9.0 + np.array([0, -0.9, 0])
    right = np.cross(fwd_h, [0, 1, 0])
    right /= np.linalg.norm(right)
    spheres = [
        {"center": tuple(center - right * 1.35), "radius": 1.15, "base": (0.95, 0.95, 0.95), "metallic": 1.0,
         "roughness": 0.02},
        {"center": tuple(center + right * 1.35), "radius": 1.15, "base": (0.8, 0.8, 0.8), "metallic": 0.0,
         "roughness": 0.5},
    ]
    rot = look_at(eye, eye + fwd)
    fov = 46.0
    img = rd.render(size, cam_pos=tuple(eye), cam_rot=rot, fov_deg=fov, spheres=spheres, transform=2, blur=0.0)
    lin = rd.render(size, cam_pos=tuple(eye), cam_rot=rot, fov_deg=fov, spheres=spheres, linear=True, ss=1)
    return img, lin, (eye, rot, fov), spheres


def do_sun(rd: Renderer, hdri: str) -> tuple[Path, dict]:
    rd.set_hdri(hdri)
    src = ibl.load_hdri(hdri)
    sun = ibl.brightest_direction(src)
    pano_size = (1600, 800)
    pano = rd.render(pano_size, mode=3, transform=2, ss=1)
    view_a, lin_a, cam_a, sph_a = sun_view(rd, sun, 0.0, max(math.degrees(math.asin(sun[1])) - 6.0, 0.0))
    view_b, lin_b, cam_b, sph_b = sun_view(rd, sun, 150.0, -4.0)

    # 实测：背对太阳那张图里，镜面球上最亮的像素
    eye, rot, fov = cam_b
    chrome = sph_b[0]
    pred = highlight_point(chrome["center"], chrome["radius"], eye, sun)
    pred_px = rd.project(pred, eye, rot, fov, (800, 560))
    c_px = rd.project(chrome["center"], eye, rot, fov, (800, 560))
    edge_px = rd.project(np.asarray(chrome["center"]) + rot[:, 0] * chrome["radius"], eye, rot, fov, (800, 560))
    rad_px = np.linalg.norm(edge_px - c_px)
    yy, xx = np.mgrid[0:560, 0:800]
    mask = (xx + 0.5 - c_px[0]) ** 2 + (yy + 0.5 - c_px[1]) ** 2 < (rad_px * 0.97) ** 2
    lum = lin_b @ np.array([0.2126, 0.7152, 0.0722])
    lum = np.where(mask, lum, -1.0)
    my, mx = np.unravel_index(int(np.argmax(lum)), lum.shape)
    err = float(np.hypot(mx + 0.5 - pred_px[0], my + 0.5 - pred_px[1]))

    # 漫反射球亮面朝向：亮度加权质心相对球心的方向，和太阳在屏幕上的方向比较
    diff = sph_b[1]
    d_px = rd.project(diff["center"], eye, rot, fov, (800, 560))
    dmask = (xx + 0.5 - d_px[0]) ** 2 + (yy + 0.5 - d_px[1]) ** 2 < (rad_px * 0.95) ** 2
    dl = np.where(dmask, lin_b @ np.array([0.2126, 0.7152, 0.0722]), 0.0)
    cx = (dl * (xx + 0.5)).sum() / dl.sum() - d_px[0]
    cy = (dl * (yy + 0.5)).sum() / dl.sum() - d_px[1]
    sun_local = rot.T @ sun                       # 相机空间：x 右，y 上，z 后
    sun_screen = np.array([sun_local[0], -sun_local[1]])
    lit_angle = math.degrees(math.atan2(cy, cx))
    sun_angle = math.degrees(math.atan2(sun_screen[1], sun_screen[0]))
    lit_err = abs((lit_angle - sun_angle + 180) % 360 - 180)

    # 拼图
    canvas = Image.new("RGB", (1600, 800 + 560 + 60), (24, 24, 24))
    pim = to_image(pano)
    d = ImageDraw.Draw(pim)
    sx, sy = ibl.direction_to_pixel(sun, *pano_size)
    d.ellipse((sx - 14, sy - 14, sx + 14, sy + 14), outline=(255, 60, 60), width=3)
    text(d, (sx + 18, sy - 12), "检测到的太阳", 20, fill=(255, 120, 120))
    for u, name in ((0.0, "−X"), (0.25, "−Z（默认相机正前方）"), (0.5, "+X"), (0.75, "+Z"), (1.0, "−X")):
        x = min(max(u * pano_size[0], 2), pano_size[0] - 2)
        d.line((x, 0, x, pano_size[1]), fill=(255, 255, 255), width=1)
        text(d, (x + (6 if u < 1 else -6), pano_size[1] - 30), name, 18, anchor="la" if u < 1 else "ra")
    d.line((0, pano_size[1] / 2, pano_size[0], pano_size[1] / 2), fill=(160, 160, 160), width=1)
    text(d, (8, 8), f"{hdri} 等距柱状展开（GLSL env_background 逐像素采样）· 太阳方向 "
                    f"({sun[0]:+.3f}, {sun[1]:+.3f}, {sun[2]:+.3f})，仰角 {math.degrees(math.asin(sun[1])):.1f}°", 20)
    for cam, color, label_ in ((cam_a, (90, 200, 255), "A"), (cam_b, (255, 210, 80), "B")):
        fwd = -cam[1][:, 2]
        fx, fy = ibl.direction_to_pixel(fwd, *pano_size)
        d.rectangle((fx - 10, fy - 10, fx + 10, fy + 10), outline=color, width=3)
        text(d, (fx + 14, fy - 10), f"相机 {label_} 朝向", 18, fill=color)
    canvas.paste(pim, (0, 0))
    for k, (view, title) in enumerate(((view_a, "A：迎着太阳看（太阳在画面上方，两个球背光）"),
                                       (view_b, "B：背对太阳（太阳在身后左上方）"))):
        vim = to_image(view, seed=k + 1)
        dv = ImageDraw.Draw(vim)
        text(dv, (10, 8), title, 19)
        if k == 1:
            dv.ellipse((pred_px[0] - 9, pred_px[1] - 9, pred_px[0] + 9, pred_px[1] + 9), outline=(0, 255, 120), width=2)
            dv.line((mx - 12, my, mx + 12, my), fill=(255, 60, 60), width=2)
            dv.line((mx, my - 12, mx, my + 12), fill=(255, 60, 60), width=2)
            text(dv, (10, 34), f"绿圈 = 按太阳方向算出的镜面高光位置，红十字 = 渲染结果最亮像素，相差 {err:.1f} 像素", 17)
            text(dv, (10, 58), f"白球亮面朝向 {lit_angle:.0f}°，太阳在屏幕上的方向 {sun_angle:.0f}°，相差 {lit_err:.0f}°", 17)
        canvas.paste(vim, (k * 800, 800))
    dc = ImageDraw.Draw(canvas)
    text(dc, (10, 800 + 560 + 18), "左右两张都是 AgX、背景不模糊。镜面球（左）金属度 1、粗糙度 0.02；白球（右）金属度 0、粗糙度 0.5。", 19)
    path = OUT / f"sun_{hdri}.png"
    canvas.save(path)
    info = {"hdri": hdri, "sun_direction": sun.round(4).tolist(), "highlight_error_px": round(err, 2),
            "chrome_radius_px": round(float(rad_px), 1), "diffuse_lit_angle_error_deg": round(lit_err, 1)}
    return path, info


# ----------------------------------------------------------------------
# c. 工作室光与 Matcap
# ----------------------------------------------------------------------

def single_sphere(rd: Renderer, mode: int, size=512, studio=None) -> np.ndarray:
    sphere = [{"center": (0.0, 0.0, 0.0), "radius": 1.0, "base": (0.8, 0.8, 0.8), "metallic": 0.0, "roughness": 0.4}]
    return rd.render((size, size), cam_pos=(0, 0, 10), ortho_half=1.08, spheres=sphere, mode=mode, transform=0,
                     bg_mix=0.0, bg_color=ibl.srgb_decode(np.array([0.24, 0.24, 0.24])), studio=studio, ss=3)


def do_studio(rd: Renderer) -> tuple[list[Path], dict]:
    paths, diffs = [], {}
    tiles = []
    for name in ibl.list_studio_lights():
        data = ibl.load_studio_light(name)
        img = single_sphere(rd, 1, studio=data)
        # 与 numpy 参考实现对比（同一算法两份实现，验证 GLSL）
        lin = rd.render((256, 256), cam_pos=(0, 0, 10), ortho_half=1.08, spheres=[{"center": (0, 0, 0), "radius": 1.0,
                        "base": (0.8, 0.8, 0.8), "metallic": 0.0, "roughness": 0.4}], mode=1, linear=True, ss=1,
                        studio=data, bg_mix=0.0, bg_color=(0, 0, 0))
        c = ((np.arange(256) + 0.5) / 256 * 2 - 1) * 1.08
        x, y = np.meshgrid(c, -c)
        inside = x * x + y * y < 0.97
        n = np.stack([x, y, np.sqrt(np.clip(1 - x * x - y * y, 0, 1))], -1)
        ref = ibl.shade_studio_np(n, np.array([0, 0, 1.0]), (0.8, 0.8, 0.8), data)
        diffs[name] = float(np.abs(lin - ref)[inside].max())
        im = to_image(img)
        d = ImageDraw.Draw(im)
        text(d, (12, 10), f"工作室光 {name}（{ibl.label(name, 'studio')}）", 22)
        p = OUT / f"studio_{name}.png"
        im.save(p)
        paths.append(p)
        tiles.append(im)
    sheet = Image.new("RGB", (len(tiles) * 512, 512))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * 512, 0))
    sheet.save(OUT / "studio_all.png")
    paths.append(OUT / "studio_all.png")
    return paths, diffs


MATCAPS = ["basic_1", "basic_2", "clay_studio", "clay_brown", "metal_shiny", "metal_carpaint", "jade", "skin",
           "toon", "check_normal+y", "reflection_check_horizontal", "pearl"]


def do_matcap(rd: Renderer) -> list[Path]:
    paths, tiles = [], []
    for name in MATCAPS:
        rd.set_matcap(name)
        img = single_sphere(rd, 2, size=384)
        im = to_image(img)
        d = ImageDraw.Draw(im)
        text(d, (10, 8), f"Matcap {name}", 18)
        p = OUT / f"matcap_{name.replace('+', 'p')}.png"
        im.save(p)
        paths.append(p)
        tiles.append(im)
    cols = 6
    sheet = Image.new("RGB", (cols * 384, math.ceil(len(tiles) / cols) * 384))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 384, (i // cols) * 384))
    sheet.save(OUT / "matcap_all.png")
    paths.append(OUT / "matcap_all.png")
    if rd.matcap_tex is not None:
        rd.matcap_tex.release()
        rd.matcap_tex = None
    return paths


# ----------------------------------------------------------------------
# 其他检查图
# ----------------------------------------------------------------------

def do_levels(rd: Renderer, hdri: str) -> Path:
    """预过滤各级展开图（放大到同一尺寸，AgX），直接看有没有亮斑、噪点、接缝。"""
    rd.set_hdri(hdri)
    env = rd.env
    rows = []
    for lv in range(env.levels):
        a = env.read_specular_level(lv)
        big = np.asarray(Image.fromarray(ibl.to_uint8(ibl.tonemap(a, 2))).resize((800, 400), Image.BILINEAR))
        im = Image.fromarray(big)
        d = ImageDraw.Draw(im)
        text(d, (8, 6), f"第 {lv} 级 {a.shape[1]}×{a.shape[0]} · 粗糙度 {env.level_roughness[lv]:.3f}", 18)
        rows.append(im)
    sheet = Image.new("RGB", (1600, 400 * math.ceil(len(rows) / 2)), (20, 20, 20))
    for i, im in enumerate(rows):
        sheet.paste(im, ((i % 2) * 800, (i // 2) * 400))
    path = OUT / f"levels_{hdri}.png"
    sheet.save(path)
    return path


def do_blur(rd: Renderer, hdri: str) -> Path:
    """背景模糊 0 / 0.1 / 0.25 / 0.5 / 0.75 / 1。"""
    rd.set_hdri(hdri)
    tiles = []
    for b in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        img = rd.render((800, 450), cam_pos=(0, 0, 0.01), cam_rot=look_at((0, 0, 0), (0, 0.15, -1)), fov_deg=70,
                        spheres=[], blur=b)
        im = to_image(img)
        text(ImageDraw.Draw(im), (10, 8), f"背景模糊 {b:g}", 20)
        tiles.append(im)
    sheet = Image.new("RGB", (1600, 1350))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 2) * 800, (i // 2) * 450))
    path = OUT / f"blur_{hdri}.png"
    sheet.save(path)
    return path


def do_furnace(rd: Renderer) -> tuple[Path, dict]:
    """白炉测试：常数 1 的环境里，白色（1.0）材质球应当和背景融为一体。"""
    rd.env.set_image(np.ones((64, 128, 3), np.float32))
    rd.current_hdri = None
    spheres = grid_spheres((1.0, 1.0, 1.0))
    lin = rd.render((1600, 1000), cam_pos=(0, 0, 24.0), fov_deg=30.0, spheres=spheres, linear=True, ss=1)
    dev = np.abs(lin - 1.0)
    stats = {"max_abs_dev": float(dev.max()), "mean_abs_dev": float(dev.mean()),
             "p99_abs_dev": float(np.percentile(dev, 99))}
    # 放大偏差显示：灰 = 1.0，越亮/越暗代表多出/少了能量（×10）
    vis = np.clip(0.5 + (lin - 1.0) * 10.0, 0, 1)
    im = to_image(ibl.srgb_encode(ibl.srgb_decode(vis)))
    d = ImageDraw.Draw(im)
    text(d, (14, 12), f"白炉测试：环境亮度恒为 1，球基础色 1.0。偏差放大 10 倍显示（中灰 = 无偏差）。"
                      f"最大偏差 {stats['max_abs_dev'] * 100:.2f}%，平均 {stats['mean_abs_dev'] * 100:.3f}%", 20)
    path = OUT / "furnace.png"
    im.save(path)
    return path, stats


def do_timing(rd: Renderer) -> dict:
    out = {}
    for name in ibl.list_hdris():
        t = time.perf_counter()
        rgb = ibl.load_hdri(name)
        t_load = (time.perf_counter() - t) * 1000
        rd.env.set_image(rgb)
        rd.ctx.finish()
        out[name] = {"load_ms": round(t_load, 1), **{k: round(v, 1) for k, v in rd.env.stats.items()}}
    rd.current_hdri = None
    return out


# ----------------------------------------------------------------------
# 显示变换与 Blender 实际输出对比（需要本机装有 Blender 4.3）
# ----------------------------------------------------------------------

def _read_cube(path: Path) -> np.ndarray:
    size, rows = None, []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("LUT_3D_SIZE"):
            size = int(line.split()[1])
        elif line and (line[0].isdigit() or line[0] in "-."):
            rows.append([float(v) for v in line.split()])
    return np.array(rows).reshape(size, size, size, 3)


def _read_spi1d(path: Path) -> np.ndarray:
    vals, started = [], False
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            started = True
        elif line.startswith("}"):
            break
        elif started and line:
            vals.append(float(line.split()[0]))
    return np.array(vals)


def _tetra(lut: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = lut.shape[0]
    p = np.clip(x, 0, 1) * (n - 1)
    i = np.minimum(np.floor(p).astype(int), n - 2)
    f = p - i
    r, g, b = i[:, 0], i[:, 1], i[:, 2]
    fr, fg, fb = f[:, :1], f[:, 1:2], f[:, 2:3]

    def at(dr, dg, db):
        return lut[b + db, g + dg, r + dr]

    c000, c111 = at(0, 0, 0), at(1, 1, 1)
    c100, c010, c001 = at(1, 0, 0), at(0, 1, 0), at(0, 0, 1)
    c110, c101, c011 = at(1, 1, 0), at(1, 0, 1), at(0, 1, 1)
    out = np.zeros_like(c000)
    cases = [
        ((fr >= fg) & (fg >= fb), lambda: c000 + fr * (c100 - c000) + fg * (c110 - c100) + fb * (c111 - c110)),
        ((fr >= fb) & (fb > fg), lambda: c000 + fr * (c100 - c000) + fb * (c101 - c100) + fg * (c111 - c101)),
        ((fb > fr) & (fr >= fg), lambda: c000 + fb * (c001 - c000) + fr * (c101 - c001) + fg * (c111 - c101)),
        ((fg > fr) & (fr >= fb), lambda: c000 + fg * (c010 - c000) + fr * (c110 - c010) + fb * (c111 - c110)),
        ((fg >= fb) & (fb > fr), lambda: c000 + fg * (c010 - c000) + fb * (c011 - c010) + fr * (c111 - c011)),
        ((fb > fg) & (fg > fr), lambda: c000 + fb * (c001 - c000) + fg * (c011 - c001) + fr * (c111 - c011)),
    ]
    for cond, fn in cases:
        m = cond[:, 0]
        out[m] = fn()[m]
    return out


class BlenderReference:
    """按 Blender 4.3 config.ocio 复现 AgX Base sRGB 与 Filmic sRGB（只用于对比）。"""

    def __init__(self) -> None:
        xyz_to_709 = np.array([[3.2404373564920070, -1.5371305409000549, -0.4985288240756933],
                               [-0.9692674982687597, 1.8760136862977035, 0.0415560804587997],
                               [0.0556434463874256, -0.2040259700872272, 1.0572254813610864]])
        eg_to_xyz = np.array([[0.7053968501, 0.1640413283, 0.08101774865],
                              [0.2801307241, 0.8202066415, -0.1003373656],
                              [-0.1037815116, -0.07290725703, 1.265746519]])
        self.m709_to_eg = np.linalg.inv(eg_to_xyz) @ np.linalg.inv(xyz_to_709)
        self.agx = _read_cube(BLENDER_CM / "luts" / "AgX_Base_sRGB.cube")
        self.desat = _read_cube(BLENDER_CM / "filmic" / "filmic_desat_33.cube")
        self.curve = _read_spi1d(BLENDER_CM / "filmic" / "filmic_to_0-70_1-03.spi1d")

    def agx_srgb(self, c: np.ndarray) -> np.ndarray:
        eg = c.reshape(-1, 3) @ self.m709_to_eg.T
        lo, hi = -12.47393, 12.5260688117
        log = (np.log2(np.maximum(eg, 2.0 ** lo)) - lo) / (hi - lo)
        return ibl.srgb_encode(np.power(np.maximum(_tetra(self.agx, log), 0), 2.4)).reshape(c.shape)

    def filmic_srgb(self, c: np.ndarray) -> np.ndarray:
        lo, hi = -12.473931188, 12.526068812
        log = (np.log2(np.maximum(c.reshape(-1, 3), 2.0 ** lo)) - lo) / (hi - lo)
        d = _tetra(self.desat, np.clip(log, 0, 1)) / 0.66
        n = self.curve.shape[0]
        out = np.interp(np.clip(d, 0, 1) * (n - 1), np.arange(n), self.curve)
        return out.reshape(c.shape)


def _lab(srgb: np.ndarray) -> np.ndarray:
    lin = ibl.srgb_decode(srgb).reshape(-1, 3)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    t = (lin @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], -1)


def do_tonemap(rd: Renderer) -> tuple[Path | None, dict]:
    if not (BLENDER_CM / "luts" / "AgX_Base_sRGB.cube").is_file():
        return None, {"skipped": "本机没有 Blender 4.3 的色彩管理文件"}
    ref = BlenderReference()
    import colorsys
    evs = np.arange(-6, 10.01, 0.25)
    hues = list(range(0, 360, 30))
    rows = []
    for hdeg in hues:
        r, g, b = colorsys.hsv_to_rgb(hdeg / 360, 0.85, 1.0)
        rgb = np.array([r, g, b]) / (np.array([r, g, b]) @ [0.2126, 0.7152, 0.0722])
        rows.append(rgb[None, :] * (0.18 * 2.0 ** evs)[:, None])
    rows.append(np.ones((len(evs), 3)) * (0.18 * 2.0 ** evs)[:, None])
    data = np.stack(rows)                                   # (行, 列, 3)
    cell = 22
    stats = {}
    panels = []
    for tid, fn in ((2, ref.agx_srgb), (1, ref.filmic_srgb)):
        mine = ibl.tonemap(data, tid)
        theirs = fn(data)
        de = np.linalg.norm(_lab(mine) - _lab(theirs), axis=-1).reshape(data.shape[:2])
        stats[TRANSFORM_NAMES[tid]] = {"mean_dE": round(float(de.mean()), 2),
                                       "p95_dE": round(float(np.percentile(de, 95)), 2),
                                       "max_dE": round(float(de.max()), 2)}
        heat = np.clip(de / 10.0, 0, 1)
        heat_rgb = np.stack([heat, 0.25 * (1 - heat), 0.4 * (1 - heat)], -1)
        blocks = [("Blender 4.3", theirs), ("SPLENDER", mine), ("色差 ΔE（满格 = 10）", heat_rgb)]
        h_rows = data.shape[0]
        panel = Image.new("RGB", (data.shape[1] * cell + 170, 3 * (h_rows * cell + 34) + 40), (24, 24, 24))
        d = ImageDraw.Draw(panel)
        text(d, (8, 8), f"{TRANSFORM_NAMES[tid]}：横向曝光 −6 … +10 档（中灰 0.18 为 0 档），纵向 12 个色相 + 灰。"
                        f"平均 ΔE {stats[TRANSFORM_NAMES[tid]]['mean_dE']}，95% 分位 {stats[TRANSFORM_NAMES[tid]]['p95_dE']}", 18)
        for k, (name, img) in enumerate(blocks):
            y0 = 40 + k * (h_rows * cell + 34)
            text(d, (8, y0 + 4), name, 17)
            tile = Image.fromarray(ibl.to_uint8(img)).resize((data.shape[1] * cell, h_rows * cell), Image.NEAREST)
            panel.paste(tile, (160, y0 + 28))
        panels.append(panel)
    sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), (24, 24, 24))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height
    path = OUT / "tonemap_vs_blender.png"
    sheet.save(path)
    return path, stats


# ----------------------------------------------------------------------

def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    which = set(argv) or {"grid", "sun", "studio", "matcap", "levels", "furnace", "tonemap", "blur", "timing"}
    rd = Renderer()
    report: dict = {"gpu": rd.ctx.info["GL_RENDERER"]}
    t0 = time.perf_counter()
    if "timing" in which:
        report["set_image_timing_ms"] = do_timing(rd)
        print("换环境图耗时（毫秒）：", json.dumps(report["set_image_timing_ms"], ensure_ascii=False))
    if "grid" in which:
        for p in do_grid(rd):
            print("  ", p.relative_to(ROOT))
    if "sun" in which:
        report["sun"] = []
        for hdri in ("sunrise", "sunset"):
            p, info = do_sun(rd, hdri)
            report["sun"].append(info)
            print("  ", p.relative_to(ROOT), info)
    if "studio" in which:
        paths, diffs = do_studio(rd)
        report["studio_glsl_vs_numpy_max_diff"] = diffs
        print("   工作室光 GLSL 与 numpy 最大差：", diffs)
    if "matcap" in which:
        do_matcap(rd)
    if "levels" in which:
        for h in ("sunrise", "forest"):
            print("  ", do_levels(rd, h).relative_to(ROOT))
    if "blur" in which:
        print("  ", do_blur(rd, "forest").relative_to(ROOT))
    if "furnace" in which:
        p, st = do_furnace(rd)
        report["furnace"] = st
        print("  ", p.relative_to(ROOT), st)
    if "tonemap" in which:
        p, st = do_tonemap(rd)
        report["tonemap_vs_blender"] = st
        print("   显示变换与 Blender 对比：", st)
    report["total_s"] = round(time.perf_counter() - t0, 1)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    rd.env.release()
    rd.ctx.release()
    print(f"完成，用时 {report['total_s']} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
