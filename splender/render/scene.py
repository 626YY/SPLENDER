"""渲染用的场景快照。

渲染引擎只读这里的数据，不碰界面、图层引擎和撤销历史，所以可以放到后台线程或别的进程里渲染。

约定（和视口一致）：
- 世界坐标 Y 轴向上，右手系，长度单位与模型相同。
- 模型是逐角点排列：positions/normals/uvs 每 3 行是一个三角形，material_ids 每个三角形一个。
- UV 原点在左下角，贴图数组第 0 行是 v = 0 那一行（与显卡贴图一致；存成图片时要上下翻转）。
- 环境贴图是等距柱状，方向约定见 engine/ibl.py 的 direction_to_uv：
  u = 0.5 + atan2(z, x) / 2π，v = 0.5 + asin(y) / π，v = 1 是天顶；数组第 0 行是 v = 1（天顶）那一行，
  和普通图片一样自上而下。rotation 是绕 +Y 的弧度：世界方向先转 -rotation 再查表。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RenderCamera:
    eye: np.ndarray                      # (3,) 相机位置
    forward: np.ndarray                  # (3,) 视线方向（单位向量）
    up: np.ndarray                       # (3,) 相机上方（单位向量，和 forward 垂直）
    tan_half_y: float                    # 竖直方向半视角的正切
    ortho: bool = False
    ortho_half_height: float = 1.0       # 正交时画面上下边到中心的世界距离
    clip_near: float = 0.001
    clip_far: float = 1.0e6
    shift: tuple = (0.0, 0.0)            # 镜头偏移：透视时是视角正切的偏移，正交时是世界距离

    @property
    def right(self) -> np.ndarray:
        r = np.cross(self.forward, self.up)
        return r / max(1e-12, float(np.linalg.norm(r)))

    def ray(self, x: float, y: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """像素 (x, y)（原点左上，像素中心传 i + 0.5）的射线 (起点, 单位方向)。渲染器的着色器要与此一致。"""
        aspect = width / max(1, height)
        nx = (x / width) * 2.0 - 1.0
        ny = 1.0 - (y / height) * 2.0
        sx, sy = self.shift
        if self.ortho:
            origin = (self.eye + self.right * (nx * self.ortho_half_height * aspect + sx)
                      + self.up * (ny * self.ortho_half_height + sy))
            return origin, self.forward.copy()
        d = self.forward + self.right * (nx * self.tan_half_y * aspect + sx) + self.up * (ny * self.tan_half_y + sy)
        return self.eye.copy(), d / np.linalg.norm(d)

    @staticmethod
    def from_camera_object(obj, aspect: float) -> "RenderCamera":
        """从场景里的摄像机物体生成（和 Blender 一样：传感器宽度按画面的长边算，偏移按长边的比例）。"""
        from ..doc.objects import D

        m = np.asarray(obj.transform.matrix(), np.float64)
        rot = m[:3, :3]
        axes = [rot[:, k] / max(np.linalg.norm(rot[:, k]), 1e-12) for k in range(3)]
        to_internal = D.T
        forward = to_internal @ (-axes[2])
        up = to_internal @ axes[1]
        eye = to_internal @ m[:3, 3]
        data = obj.data
        if data.projection == "ORTHO":
            half_long = float(data.ortho_scale) * 0.5
        else:
            half_long = float(data.sensor) * 0.5 / max(1e-6, float(data.lens))
        half_y = half_long / aspect if aspect >= 1.0 else half_long
        shift = (float(data.shift_x) * 2.0 * half_long, float(data.shift_y) * 2.0 * half_long)
        return RenderCamera(eye=eye, forward=forward / np.linalg.norm(forward), up=up / np.linalg.norm(up),
                            tan_half_y=float(half_y), ortho=data.projection == "ORTHO",
                            ortho_half_height=float(half_y), clip_near=float(max(data.clip_start, 1e-6)),
                            clip_far=float(data.clip_end), shift=shift)

    @staticmethod
    def from_view_camera(camera, aspect: float) -> "RenderCamera":
        """从视口相机（engine/camera.py 的 Camera）生成，画面与视口一致。"""
        right, up, back = camera.basis()
        tan_y, _tan_x = camera.tan_half(aspect)
        near, far = camera.clip_range()
        ox, oy = getattr(camera, "offset", (0.0, 0.0))
        half_y = float(camera.distance * tan_y) if camera.ortho else float(tan_y)
        return RenderCamera(eye=np.asarray(camera.eye, np.float64), forward=-np.asarray(back, np.float64),
                            up=np.asarray(up, np.float64), tan_half_y=float(tan_y), ortho=bool(camera.ortho),
                            ortho_half_height=float(camera.distance * tan_y), clip_near=float(max(near, 1e-6)),
                            clip_far=float(far), shift=(float(ox) * half_y * aspect, float(oy) * half_y))


@dataclass
class RenderTexture:
    """一张贴图。data 的形状 (高, 宽, 通道)，第 0 行是 v = 0。dtype 为 uint8（0..255）、float16 或 float32。

    srgb 为真表示 data 按 sRGB 编码（基础色）；否则是线性数值（金属度、粗糙度、高度）。
    """
    data: np.ndarray
    srgb: bool = False

    @property
    def size(self) -> tuple[int, int]:
        return int(self.data.shape[1]), int(self.data.shape[0])


@dataclass
class RenderMaterial:
    """金属度/粗糙度工作流的材质。有贴图时用贴图，没有时用常数。"""
    name: str = "材质"
    base_color: tuple = (0.8, 0.8, 0.8)          # sRGB
    metallic: float = 0.0
    roughness: float = 0.5
    base_color_tex: RenderTexture | None = None  # RGBA，sRGB 编码，A 不用
    mr_tex: RenderTexture | None = None          # R = 金属度，G = 粗糙度（线性）
    height_tex: RenderTexture | None = None      # R = 高度（线性，可为负）
    height_scale: float = 1.0                    # 高度贴图的数值乘以它，单位是世界长度
    normal_tex: RenderTexture | None = None      # 切线空间法线（RGB，×0.5+0.5），从高模烘的


@dataclass
class RenderMesh:
    name: str
    positions: np.ndarray                         # (3T, 3) float32
    normals: np.ndarray                           # (3T, 3) float32，单位向量
    uvs: np.ndarray                               # (3T, 2) float32
    material_ids: np.ndarray                      # (T,) int32，下标指向 RenderScene.materials
    transform: np.ndarray = field(default_factory=lambda: np.identity(4, np.float32))

    @property
    def triangle_count(self) -> int:
        return int(len(self.positions) // 3)


@dataclass
class RenderEnvironment:
    image: np.ndarray | None = None               # (高, 宽, 3) float32 线性，等距柱状；None 时用纯色
    color: tuple = (0.05, 0.05, 0.05)             # 没有图时的环境颜色（线性）
    rotation: float = 0.0                         # 绕 +Y 的弧度
    strength: float = 1.0
    background: bool = True                       # 相机直接看到环境（否则背景透明或用 background_color）
    background_color: tuple = (0.05, 0.05, 0.05)


@dataclass
class RenderSettings:
    width: int = 1920
    height: int = 1080
    samples: int = 128                            # 每像素采样数（路径追踪）
    max_bounces: int = 6
    clamp_indirect: float = 10.0                  # 间接光单样本亮度上限，压萤火虫；0 表示不限
    exposure: float = 0.0                         # 曝光（档）
    view_transform: str = "FILMIC"                # STANDARD / FILMIC / AGX
    transparent: bool = False                     # 背景透明
    seed: int = 0
    time_limit: float = 0.0                       # 秒，0 表示不限
    texture_size: int = 4096                      # 渲染时图层合成贴图的边长


@dataclass
class RenderLight:
    """一盏灯（世界坐标，Y 朝上）。和 Blender 一样：点光、聚光、面光用功率（瓦），日光用强度（瓦/平方米）。

    radiance 是渲染器直接用的数值：点光、聚光是发光强度（每球面度），日光是照度，面光是面上的辐亮度。"""
    type: str                                    # POINT / SUN / SPOT / AREA
    position: np.ndarray                         # (3,)
    direction: np.ndarray                        # (3,) 灯朝向（日光、聚光、面光照射的方向）
    color: tuple = (1.0, 1.0, 1.0)               # 线性
    radiance: float = 1.0
    radius: float = 0.0                          # 点光、聚光的半径（软阴影）
    angle: float = 0.0                           # 日光的角直径（弧度）
    spot_size: float = math.radians(45.0)        # 聚光锥的全角（弧度）
    spot_blend: float = 0.15
    size: tuple = (1.0, 1.0)                     # 面光的两边长
    axis_u: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    axis_v: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    shadow: bool = True
    power: float = 0.0                           # 原始数值（外部渲染器用）
    strength: float = 0.0


def light_from_object(obj) -> RenderLight:
    """场景里的灯光物体 → 渲染用的灯（灯照向它的局部 -Z，和 Blender 一样）。"""
    from ..doc.objects import D

    m = np.asarray(obj.transform.matrix(), np.float64)
    rot = m[:3, :3]
    scale = np.linalg.norm(rot, axis=0)
    axes = [rot[:, k] / max(scale[k], 1e-12) for k in range(3)]
    to_internal = D.T
    data = obj.data
    direction = to_internal @ (-axes[2])
    color = tuple(float(c) for c in data.color)
    kind = str(data.type)
    if kind == "SUN":
        radiance = float(data.strength)
    elif kind == "AREA":
        side = float(data.size)
        area = max(side * scale[0] * side * scale[1], 1e-8)
        radiance = float(data.power) / (math.pi * area)
    else:
        radiance = float(data.power) / (4.0 * math.pi)
    return RenderLight(type=kind, position=to_internal @ m[:3, 3], direction=direction / np.linalg.norm(direction),
                       color=color, radiance=radiance, radius=float(data.radius),
                       angle=math.radians(float(data.angle)), spot_size=math.radians(float(data.spot_size)),
                       spot_blend=float(data.spot_blend),
                       size=(float(data.size) * float(scale[0]), float(data.size) * float(scale[1])),
                       axis_u=to_internal @ axes[0], axis_v=to_internal @ axes[1], shadow=bool(data.use_shadow),
                       power=float(data.power), strength=float(data.strength))


@dataclass
class RenderScene:
    meshes: list[RenderMesh] = field(default_factory=list)
    materials: list[RenderMaterial] = field(default_factory=list)
    environment: RenderEnvironment = field(default_factory=RenderEnvironment)
    camera: RenderCamera | None = None
    settings: RenderSettings = field(default_factory=RenderSettings)
    lights: list[RenderLight] = field(default_factory=list)

    @property
    def triangle_count(self) -> int:
        return sum(m.triangle_count for m in self.meshes)


def fov_to_tan_half(fov_y_degrees: float) -> float:
    return math.tan(math.radians(fov_y_degrees) * 0.5)
