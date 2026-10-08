"""路径追踪渲染器（相当于 Blender 的 Cycles）：显卡计算着色器，渐进累积。

用法：
    tracer = PathTracer(ctx)
    tracer.set_scene(scene)            # RenderScene，见 splender/render/scene.py
    while not tracer.finished:
        tracer.render_pass(budget_ms=12)
    tracer.resolve()
    image = tracer.read_image()        # (高, 宽, 4) uint8 sRGB，第 0 行是画面顶行

着色算法改编自 GLSL-PathTracer（MIT 许可），见 shaders.py 的说明。
"""
from __future__ import annotations

import math
import time

import moderngl
import numpy as np

from ...engine import glx
from ..scene import RenderCamera, RenderScene, RenderSettings
from .bvh import build_bvh
from .shaders import CLEAR, RESOLVE, STACK, TRACE

TRANSFORMS = {"STANDARD": 0, "FILMIC": 1, "AGX": 2}
MAX_MATERIALS = 64


def _fit(data: np.ndarray, size: int) -> np.ndarray:
    """把贴图缩放到 size×size（行列都按 v=0 在第 0 行的约定）。"""
    if data.shape[0] == size and data.shape[1] == size:
        return data
    from PIL import Image

    if data.dtype == np.uint8:
        channels = data.shape[2] if data.ndim == 3 else 1
        mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[channels]
        image = Image.fromarray(data if channels > 1 else data.reshape(data.shape[:2]), mode)
        out = np.asarray(image.resize((size, size), Image.BILINEAR))
        return out.reshape(size, size, channels)
    planes = data.reshape(data.shape[0], data.shape[1], -1).astype(np.float32)
    result = np.empty((size, size, planes.shape[2]), np.float32)
    for c in range(planes.shape[2]):
        result[:, :, c] = np.asarray(Image.fromarray(planes[:, :, c], "F").resize((size, size), Image.BILINEAR))
    return result


class PathTracer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._trace = ctx.compute_shader(TRACE)
        self._resolve = ctx.compute_shader(RESOLVE)
        self._clear = ctx.compute_shader(CLEAR)
        self.settings = RenderSettings()
        self.camera: RenderCamera | None = None
        self.scene: RenderScene | None = None
        self._buffers: dict[str, moderngl.Buffer] = {}
        self._textures: dict[str, object] = {}
        self._accum = None
        self._display = None
        self.samples_done = 0
        self._row = 0
        self._rows_per_band = 64
        self._started = 0.0
        self._render_seconds = 0.0
        self._eps = 1e-4
        self._texel = 1.0 / 4096.0
        self._env_size = (1, 1)
        self._env_sum = 0.0
        self._has_env = False
        self._stats = {"triangles": 0, "nodes": 0, "bvh_ms": 0.0, "bvh_depth": 0, "upload_ms": 0.0}

    # ------------------------------------------------------------------ 场景
    def set_scene(self, scene: RenderScene) -> None:
        started = time.perf_counter()
        self.scene = scene
        positions, normals, uvs, mats = [], [], [], []
        for mesh in scene.meshes:
            p = np.asarray(mesh.positions, np.float32).reshape(-1, 3)
            n = np.asarray(mesh.normals, np.float32).reshape(-1, 3)
            m = np.asarray(mesh.transform, np.float64)
            if not np.allclose(m, np.identity(4)):
                p = (np.c_[p, np.ones(len(p))] @ m.T)[:, :3].astype(np.float32)
                normal_m = np.linalg.inv(m[:3, :3]).T
                n = n @ normal_m.T
                n = (n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)).astype(np.float32)
            positions.append(p)
            normals.append(n)
            uvs.append(np.asarray(mesh.uvs, np.float32).reshape(-1, 2))
            mats.append(np.clip(np.asarray(mesh.material_ids, np.int32), 0, max(0, len(scene.materials) - 1)))
        if positions:
            pos = np.concatenate(positions)
            nrm = np.concatenate(normals)
            uv = np.concatenate(uvs)
            mat = np.concatenate(mats)
        else:
            pos = np.zeros((0, 3), np.float32)
            nrm = np.zeros((0, 3), np.float32)
            uv = np.zeros((0, 2), np.float32)
            mat = np.zeros(0, np.int32)
        v0, v1, v2 = pos[0::3], pos[1::3], pos[2::3]
        bvh = build_bvh(v0, v1, v2)
        order = bvh["order"]
        self.order = order                       # 排列后的第 i 个三角形是原来的第 order[i] 个
        tri_count = len(order)
        tri_pos = np.zeros((tri_count, 3, 4), np.float32)
        tri_pos[:, 0, :3] = v0[order]
        tri_pos[:, 1, :3] = v1[order] - v0[order]
        tri_pos[:, 2, :3] = v2[order] - v0[order]
        n0, n1, n2 = nrm[0::3][order], nrm[1::3][order], nrm[2::3][order]
        t0, t1, t2 = uv[0::3][order], uv[1::3][order], uv[2::3][order]
        attr = np.zeros((tri_count, 4, 4), np.float32)
        attr[:, 0, :3] = n0
        attr[:, 0, 3] = t0[:, 0]
        attr[:, 1, :3] = n1
        attr[:, 1, 3] = t0[:, 1]
        attr[:, 2, :3] = n2
        attr[:, 2, 3] = t1[:, 0]
        attr[:, 3, 0] = t1[:, 1]
        attr[:, 3, 1] = t2[:, 0]
        attr[:, 3, 2] = t2[:, 1]
        attr[:, 3, 3] = mat[order].astype(np.float32) if tri_count else 0.0
        if tri_count:
            lo = pos.min(axis=0)
            hi = pos.max(axis=0)
            self._eps = float(max(np.linalg.norm(hi - lo), 1e-6) * 2e-5)
        if bvh["depth"] >= STACK:
            raise RuntimeError("模型的 BVH 太深（%d 层），超过渲染器的上限 %d" % (bvh["depth"], STACK))
        self._upload("nodes", bvh["nodes"])
        self._upload("tri_pos", tri_pos if tri_count else np.zeros((1, 3, 4), np.float32))
        self._upload("tri_attr", attr if tri_count else np.zeros((1, 4, 4), np.float32))
        self._set_materials(scene)
        self._set_environment(scene)
        self._set_lights(scene)
        self._stats.update(triangles=int(tri_count), nodes=int(len(bvh["nodes"])), bvh_ms=round(bvh["ms"], 1),
                           bvh_depth=int(bvh["depth"]))
        self.settings = scene.settings
        self._ensure_targets()
        if scene.camera is not None:
            self.camera = scene.camera
        self._stats["upload_ms"] = round((time.perf_counter() - started) * 1000.0 - bvh["ms"], 1)
        self.reset()

    def _upload(self, name: str, array: np.ndarray) -> None:
        data = np.ascontiguousarray(array).tobytes()
        old = self._buffers.get(name)
        if old is not None and old.size == len(data):
            old.write(data)
            return
        if old is not None:
            old.release()
        self._buffers[name] = self.ctx.buffer(data if data else bytes(16))

    def _texture_array(self, name: str, layers: list, components: int, dtype: str) -> None:
        old = self._textures.pop(name, None)
        if old is not None:
            old.release()
        if not layers:
            data = np.zeros((1, 1, components), np.float32 if dtype == "f4" else np.uint8)
            tex = self.ctx.texture_array((1, 1, 1), components, data.tobytes(), dtype=dtype)
        else:
            size = layers[0].shape[0]
            stack = np.ascontiguousarray(np.stack(layers))
            tex = self.ctx.texture_array((size, size, len(layers)), components, stack.tobytes(), dtype=dtype)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        self._textures[name] = tex

    def _set_materials(self, scene: RenderScene) -> None:
        materials = scene.materials[:MAX_MATERIALS] or []
        size = 0
        for m in materials:
            for tex in (m.base_color_tex, m.mr_tex, m.height_tex, getattr(m, "normal_tex", None)):
                if tex is not None:
                    size = max(size, tex.size[0], tex.size[1])
        size = min(size, max(16, int(scene.settings.texture_size))) if size else 0
        base_layers, mr_layers, height_layers, nmap_layers = [], [], [], []
        table = np.zeros(max(1, len(materials)), dtype=[("base", "f4", 4), ("params", "f4", 4), ("tex", "i4", 4)])
        for index, m in enumerate(materials):
            color = np.asarray(m.base_color, np.float64)
            linear = np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)
            table[index]["base"] = (*linear, 1.0)
            table[index]["params"] = (float(m.metallic), float(m.roughness), float(m.height_scale), 0.0)
            tex = [-1, -1, -1, -1]
            if m.base_color_tex is not None:
                data = np.asarray(m.base_color_tex.data)
                if data.dtype != np.uint8:
                    data = np.clip(np.rint(data * 255.0), 0, 255).astype(np.uint8)
                if data.ndim == 2:
                    data = data[:, :, None]
                if data.shape[2] == 3:
                    data = np.concatenate([data, np.full(data.shape[:2] + (1,), 255, np.uint8)], axis=2)
                tex[0] = len(base_layers)
                base_layers.append(_fit(data[:, :, :4], size))
            if m.mr_tex is not None:
                data = np.asarray(m.mr_tex.data)
                if data.dtype != np.uint8:
                    data = np.clip(np.rint(data * 255.0), 0, 255).astype(np.uint8)
                if data.ndim == 2:
                    data = np.stack([data, data], axis=2)
                tex[1] = len(mr_layers)
                mr_layers.append(_fit(np.ascontiguousarray(data[:, :, :2]), size))
            if m.height_tex is not None:
                data = np.asarray(m.height_tex.data, np.float32)
                if data.ndim == 3:
                    data = data[:, :, 0]
                tex[2] = len(height_layers)
                height_layers.append(_fit(data[:, :, None], size).reshape(size, size, 1))
            if getattr(m, "normal_tex", None) is not None:
                data = np.asarray(m.normal_tex.data, np.uint8)[:, :, :3]
                data = np.concatenate([data, np.full(data.shape[:2] + (1,), 255, np.uint8)], axis=2)
                tex[3] = len(nmap_layers)
                nmap_layers.append(_fit(np.ascontiguousarray(data), size))
            table[index]["tex"] = tex
        self._upload("mats", table)
        self._texture_array("base", base_layers, 4, "f1")
        self._texture_array("mr", mr_layers, 2, "f1")
        self._texture_array("height", height_layers, 1, "f4")
        self._texture_array("nmap", nmap_layers, 4, "f1")
        self._texel = 1.0 / float(size) if size else 1.0 / 4096.0

    def _set_lights(self, scene: RenderScene) -> None:
        lights = list(getattr(scene, "lights", []) or [])
        kinds = {"POINT": 0, "SUN": 1, "SPOT": 2, "AREA": 3}
        data = np.zeros((max(1, len(lights)), 6, 4), np.float32)
        for i, light in enumerate(lights):
            kind = kinds.get(str(light.type), 0)
            data[i, 0, :3] = light.position
            data[i, 0, 3] = kind
            data[i, 1, :3] = light.direction
            data[i, 1, 3] = light.angle * 0.5 if kind == 1 else light.radius
            data[i, 2, :3] = np.asarray(light.color, np.float32) * float(light.radiance)
            data[i, 2, 3] = 1.0 if light.shadow else 0.0
            cos_half = math.cos(light.spot_size * 0.5)
            data[i, 3] = (cos_half, (1.0 - cos_half) * float(light.spot_blend), light.size[0], light.size[1])
            data[i, 4, :3] = light.axis_u
            data[i, 5, :3] = light.axis_v
        self._upload("lights", data)
        self._light_count = len(lights)

    def _set_environment(self, scene: RenderScene) -> None:
        env = scene.environment
        old = self._textures.pop("env", None)
        if old is not None:
            old.release()
        image = env.image
        if image is None or image.size == 0:
            self._has_env = False
            tex = self.ctx.texture((1, 1), 3, np.zeros(3, np.float32).tobytes(), dtype="f4")
            self._env_size = (1, 1)
            self._env_sum = 1.0
            self._upload("env_cdf", np.zeros(4, np.float32))
        else:
            image = np.ascontiguousarray(np.asarray(image, np.float32)[:, :, :3])
            height, width = image.shape[:2]
            tex = self.ctx.texture((width, height), 3, image.tobytes(), dtype="f4")
            rows = (np.arange(height) + 0.5) / height
            elevation = ((1.0 - rows) - 0.5) * math.pi
            weight = (image @ np.array([0.2126, 0.7152, 0.0722], np.float32)) * np.cos(elevation)[:, None]
            weight = np.maximum(weight, 0.0).astype(np.float64)
            row_sum = weight.sum(axis=1)
            total = float(row_sum.sum())
            if total <= 0.0:
                weight[:] = 1.0
                row_sum = weight.sum(axis=1)
                total = float(row_sum.sum())
            conditional = np.cumsum(weight, axis=1) / np.maximum(row_sum[:, None], 1e-30)
            conditional[row_sum <= 0.0] = (np.arange(1, width + 1) / width)[None, :]
            marginal = np.cumsum(row_sum) / total
            self._upload("env_cdf", np.concatenate([marginal, conditional.ravel()]).astype(np.float32))
            self._has_env = True
            self._env_size = (width, height)
            self._env_sum = total
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = False
        self._textures["env"] = tex

    # ------------------------------------------------------------------ 设置
    def update_camera(self, camera: RenderCamera) -> None:
        self.camera = camera
        self.reset()

    def update_settings(self, settings: RenderSettings) -> None:
        self.settings = settings
        self._ensure_targets()
        self.reset()

    def update_environment(self, scene: RenderScene) -> None:
        """只换环境（旋转、强度、换图）。"""
        if self.scene is not None:
            self.scene.environment = scene.environment
        self._set_environment(scene)
        self.reset()

    def _ensure_targets(self) -> None:
        width = max(1, int(self.settings.width))
        height = max(1, int(self.settings.height))
        if self._accum is not None and self._accum.size == (width, height):
            return
        for tex in (self._accum, self._display, getattr(self, "_aov_albedo", None), getattr(self, "_aov_normal", None),
                    getattr(self, "_denoised", None)):
            if tex is not None:
                tex.release()
        self._accum = self.ctx.texture((width, height), 4, dtype="f4")
        self._aov_albedo = self.ctx.texture((width, height), 4, dtype="f4")
        self._aov_normal = self.ctx.texture((width, height), 4, dtype="f4")
        self._denoised = self.ctx.texture((width, height), 4, dtype="f4")
        self._display = self.ctx.texture((width, height), 4, dtype="f1")
        self._display.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.denoised_samples = 0

    def reset(self) -> None:
        self._ensure_targets()
        width, height = self._accum.size
        self._clear["u_size"] = (width, height)
        self._accum.bind_to_image(0, read=False, write=True)
        self._aov_albedo.bind_to_image(2, read=False, write=True)
        self._aov_normal.bind_to_image(3, read=False, write=True)
        self._clear.run((width + 7) // 8, (height + 7) // 8, 1)
        self.ctx.memory_barrier()
        self.samples_done = 0
        self.denoised_samples = 0           # 降噪结果对应的采样数，0 表示没有（换视角、改场景后作废）
        self._denoise_task = None
        self._row = 0
        self._render_seconds = 0.0
        self._started = time.perf_counter()

    # ------------------------------------------------------------------ 渲染
    @property
    def finished(self) -> bool:
        if self.samples_done >= max(1, int(self.settings.samples)):
            return True
        limit = float(getattr(self.settings, "time_limit", 0.0) or 0.0)
        return limit > 0.0 and self._render_seconds >= limit

    def _set(self, name: str, value) -> None:
        program = self._trace
        if name in program:
            program[name] = value

    def _bind(self) -> None:
        settings = self.settings
        camera = self.camera
        width, height = self._accum.size
        for name, binding in (("nodes", 1), ("tri_pos", 2), ("tri_attr", 3), ("mats", 4), ("env_cdf", 5)):
            self._buffers[name].bind_to_storage_buffer(binding)
        if "lights" not in self._buffers:
            self._upload("lights", np.zeros((1, 6, 4), np.float32))
            self._light_count = 0
        self._buffers["lights"].bind_to_storage_buffer(6)
        self._set("u_light_count", int(getattr(self, "_light_count", 0)))
        self._set("u_shift", tuple(float(v) for v in getattr(camera, "shift", (0.0, 0.0))))
        for unit, (name, uniform) in enumerate((("base", "u_base_tex"), ("mr", "u_mr_tex"), ("height", "u_height_tex"),
                                                ("env", "u_env"), ("nmap", "u_nmap_tex"))):
            self._textures[name].use(unit)
            self._set(uniform, unit)
        env = self.scene.environment
        self._set("u_env_size", tuple(int(v) for v in self._env_size))
        self._set("u_has_env", int(self._has_env))
        self._set("u_env_sum", float(self._env_sum))
        self._set("u_env_color", tuple(float(v) for v in env.color))
        self._set("u_env_rotation", float(env.rotation))
        self._set("u_env_strength", float(env.strength))
        self._set("u_background", int(env.background))
        self._set("u_transparent", int(settings.transparent))
        self._set("u_background_color", tuple(float(v) for v in env.background_color))
        self._set("u_eye", tuple(float(v) for v in camera.eye))
        self._set("u_forward", tuple(float(v) for v in camera.forward))
        self._set("u_right", tuple(float(v) for v in camera.right))
        self._set("u_up", tuple(float(v) for v in camera.up))
        self._set("u_tan_half_y", float(camera.tan_half_y))
        self._set("u_aspect", float(width) / float(max(1, height)))
        self._set("u_ortho", int(camera.ortho))
        self._set("u_ortho_half_h", float(camera.ortho_half_height))
        self._set("u_size", (width, height))
        self._set("u_max_bounces", int(max(0, settings.max_bounces)))
        self._set("u_clamp", float(settings.clamp_indirect))
        self._set("u_eps", float(self._eps))
        self._set("u_texel", float(self._texel))
        self._set("u_seed", int(settings.seed) & 0xFFFFFFFF)
        self._set("u_debug", int(getattr(settings, "debug_mode", 0)))
        self._set("u_aov", 1)
        self._accum.bind_to_image(0, read=True, write=True)
        self._aov_albedo.bind_to_image(2, read=True, write=True)
        self._aov_normal.bind_to_image(3, read=True, write=True)

    def render_pass(self, budget_ms: float = 12.0) -> int:
        """推进渲染，尽量用满 budget_ms。画面分成横条，每条一次显卡调度。返回已完成的每像素采样数。"""
        if self.scene is None or self.camera is None or self.finished:
            return self.samples_done
        self._bind()
        width, height = self._accum.size
        spent = 0.0
        while not self.finished:
            rows = max(8, min(height - self._row, self._rows_per_band))
            self._set("u_offset", (0, self._row))
            self._set("u_sample", int(self.samples_done))
            t0 = time.perf_counter()
            self._trace.run((width + 7) // 8, (rows + 7) // 8, 1)
            fence = glx.Fence()
            while not fence.ready(wait_ns=2_000_000):
                pass
            fence.release()
            elapsed = (time.perf_counter() - t0) * 1000.0
            spent += elapsed
            self._render_seconds += elapsed / 1000.0
            # 调整每条的行数：单次调度控制在 20 ms 左右
            per_row = elapsed / max(1, rows)
            target = min(20.0, max(2.0, budget_ms * 0.5))
            self._rows_per_band = int(max(8, min(height, target / max(per_row, 1e-4))))
            self._row += rows
            if self._row >= height:
                self._row = 0
                self.samples_done += 1
            if spent + per_row * self._rows_per_band > budget_ms:
                break
        self.ctx.memory_barrier()
        return self.samples_done

    def resolve(self) -> None:
        if self._accum is None:
            return
        width, height = self._accum.size
        program = self._resolve
        program["u_size"] = (width, height)
        program["u_transform"] = TRANSFORMS.get(str(self.settings.view_transform).upper(), 1)
        program["u_exposure"] = float(self.settings.exposure)
        if self.denoised_samples:
            program["u_inv_spp"] = 1.0
            self._denoised.bind_to_image(0, read=True, write=False)
        else:
            program["u_inv_spp"] = 1.0 / max(1, self.samples_done)
            self._accum.bind_to_image(0, read=True, write=False)
        self._display.bind_to_image(1, read=False, write=True)
        program.run((width + 7) // 8, (height + 7) // 8, 1)
        self.ctx.memory_barrier()

    @property
    def display_texture(self):
        return self._display

    # ------------------------------------------------------------------ 降噪
    def _denoise_inputs(self):
        width, height = self._accum.size
        spp = float(max(1, self.samples_done))
        color = np.frombuffer(self._accum.read(), np.float32).reshape(height, width, 4) / spp
        albedo = np.frombuffer(self._aov_albedo.read(), np.float32).reshape(height, width, 4)[:, :, :3] / spp
        normal = np.frombuffer(self._aov_normal.read(), np.float32).reshape(height, width, 4)[:, :, :3] / spp
        return color, albedo, normal

    def _store_denoised(self, rgb: np.ndarray, alpha: np.ndarray, samples: int) -> None:
        height, width = alpha.shape
        data = np.empty((height, width, 4), np.float32)
        data[:, :, :3] = rgb
        data[:, :, 3] = alpha
        self._denoised.write(data.tobytes())
        self.denoised_samples = int(samples)

    def denoise_now(self, prefer: str = "AUTO") -> dict:
        """同步降噪当前结果（出图结束时用）。之后 resolve() 和 read_image() 用降噪后的图。"""
        from ..denoise import denoise

        color, albedo, normal = self._denoise_inputs()
        rgb, stats = denoise(color[:, :, :3], albedo, normal, prefer)
        self._store_denoised(rgb, color[:, :, 3], self.samples_done)
        return stats

    def denoise_start(self, prefer: str = "AUTO") -> bool:
        """后台降噪（视口用）：读出当前结果交给线程，之后每帧 denoise_poll()。已有任务在跑时不重复开。"""
        from ..denoise import DenoiseTask

        if getattr(self, "_denoise_task", None) is not None:
            return False
        color, albedo, normal = self._denoise_inputs()
        task = DenoiseTask(color[:, :, :3].copy(), albedo, normal, prefer, (self.samples_done, color[:, :, 3].copy()))
        self._denoise_task = task
        return True

    def denoise_poll(self) -> bool:
        """后台降噪做完了就换上，返回是否换了新结果。"""
        task = getattr(self, "_denoise_task", None)
        if task is None or not task.done:
            return False
        self._denoise_task = None
        if task.result is None:
            return False
        samples, alpha = task.tag
        self._store_denoised(task.result, alpha, samples)
        return True

    @property
    def denoising(self) -> bool:
        return getattr(self, "_denoise_task", None) is not None

    def read_image(self, linear: bool = False) -> np.ndarray:
        width, height = self._accum.size
        if linear:
            if self.denoised_samples:
                return np.frombuffer(self._denoised.read(), np.float32).reshape(height, width, 4).copy()
            data = np.frombuffer(self._accum.read(), np.float32).reshape(height, width, 4)
            return data / max(1, self.samples_done)
        self.resolve()
        data = np.frombuffer(self._display.read(), np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(data[::-1])

    def stats(self) -> dict:
        info = dict(self._stats)
        info["samples"] = self.samples_done
        info["seconds"] = round(self._render_seconds, 3)
        if self._render_seconds > 0 and self._accum is not None:
            width, height = self._accum.size
            info["spp_per_second"] = round(self.samples_done / self._render_seconds, 2)
            info["mpix_samples_per_second"] = round(self.samples_done * width * height / self._render_seconds / 1e6, 1)
        return info

    def release(self) -> None:
        for buffer in self._buffers.values():
            buffer.release()
        self._buffers.clear()
        for tex in self._textures.values():
            tex.release()
        self._textures.clear()
        for tex in (self._accum, self._display, getattr(self, "_aov_albedo", None), getattr(self, "_aov_normal", None),
                    getattr(self, "_denoised", None)):
            if tex is not None:
                tex.release()
        self._accum = self._display = self._aov_albedo = self._aov_normal = self._denoised = None
        for program in (self._trace, self._resolve, self._clear):
            program.release()
