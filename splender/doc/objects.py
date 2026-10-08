"""场景物体（和 Blender 一样）：每个物体有位置、旋转（XYZ 欧拉角，度）、缩放，能选中、隐藏。

网格物体的顶点存成世界坐标：变换改了以后，按「新矩阵 × 旧矩阵的逆」把顶点变过去（拖动的过程中视口用矩阵实时显示，
松手才写进数据）。面板上看到、改的位置、旋转、缩放就是这个物体当前的变换；「应用变换」只把这几个数清零、顶点不动。
摄像机、灯光、空物体没有网格，只有变换和自己的参数。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.props import (BoolProperty, ColorProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                          PropertyGroup, StringProperty)

EMPTY_DISPLAY_ITEMS = [("PLAIN_AXES", "坐标轴", "", "empty.axes"), ("ARROWS", "箭头", "", "empty.arrows"),
                       ("SINGLE_ARROW", "单箭头", "", "empty.arrow"), ("CIRCLE", "圆", "", "empty.circle"),
                       ("CUBE", "立方体", "", "empty.cube"), ("SPHERE", "球", "", "empty.sphere"),
                       ("CONE", "锥", "", "empty.cone")]


class ObjectTransform(PropertyGroup):
    undoable = True
    location = FloatVectorProperty("位置", default=(0.0, 0.0, 0.0), size=3, precision=3, unit="m")
    rotation = FloatVectorProperty("旋转", default=(0.0, 0.0, 0.0), size=3, precision=1, unit="°")
    scale = FloatVectorProperty("缩放", default=(1.0, 1.0, 1.0), size=3, precision=3)

    def matrix(self) -> np.ndarray:
        return compose(self.location, self.rotation, self.scale)

    def set_matrix(self, matrix: np.ndarray) -> None:
        """三个值一起改，只算一次变化（顶点只跟着变一次）。"""
        location, rotation, scale = decompose(matrix)
        with self.changed.block():
            self.location = tuple(float(v) for v in location)
            self.rotation = tuple(float(v) for v in rotation)
            self.scale = tuple(float(v) for v in scale)
        for name in ("location", "rotation", "scale"):
            self.changed.emit(name)

    def set_values(self, location=None, rotation=None, scale=None) -> None:
        with self.changed.block():
            if location is not None:
                self.location = tuple(float(v) for v in location)
            if rotation is not None:
                self.rotation = tuple(float(v) for v in rotation)
            if scale is not None:
                self.scale = tuple(float(v) for v in scale)
        for name in ("location", "rotation", "scale"):
            self.changed.emit(name)


# ====================================================================== 矩阵
def rotation_matrix(degrees) -> np.ndarray:
    """XYZ 欧拉角（度，先绕 X、再 Y、再 Z，和 Blender 默认的 XYZ 相同）→ 3×3。"""
    rx, ry, rz = (math.radians(float(v)) for v in degrees)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    return mz @ my @ mx


def euler_from_matrix(r: np.ndarray) -> tuple[float, float, float]:
    """3×3 旋转 → XYZ 欧拉角（度）。"""
    sy = -float(r[2, 0])
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(sy) < 0.999999:
        rx = math.atan2(float(r[2, 1]), float(r[2, 2]))
        rz = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        rx = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def compose(location, rotation, scale) -> np.ndarray:
    m = np.identity(4)
    m[:3, :3] = rotation_matrix(rotation) @ np.diag([float(v) for v in scale])
    m[:3, 3] = [float(v) for v in location]
    return m


def decompose(matrix: np.ndarray) -> tuple:
    m = np.asarray(matrix, np.float64)
    location = m[:3, 3].copy()
    basis = m[:3, :3]
    scale = np.linalg.norm(basis, axis=0)
    if np.linalg.det(basis) < 0:
        scale[0] = -scale[0]
    rot = basis / np.where(np.abs(scale) > 1e-12, scale, 1.0)
    return location, euler_from_matrix(rot), scale


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, np.float64)
    return p @ matrix[:3, :3].T + matrix[:3, 3]


def transform_normals(matrix: np.ndarray, normals: np.ndarray) -> np.ndarray:
    n = np.asarray(normals, np.float64) @ np.linalg.inv(matrix[:3, :3])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(length, 1e-30)


# ====================================================================== 摄像机、灯光
class CameraData(PropertyGroup):
    undoable = True
    projection = EnumProperty("投影", items=[("PERSP", "透视", ""), ("ORTHO", "正交", "")], default="PERSP")
    lens = FloatProperty("焦距", default=50.0, min=1.0, max=5000.0, soft_max=300.0, precision=1, unit="mm")
    sensor = FloatProperty("传感器宽度", default=36.0, min=1.0, max=100.0, precision=1, unit="mm")
    ortho_scale = FloatProperty("正交大小", default=6.0, min=0.001, max=100000.0, precision=3)
    clip_start = FloatProperty("近裁剪", default=0.1, min=0.0001, max=1000.0, precision=4, unit="m")
    clip_end = FloatProperty("远裁剪", default=1000.0, min=0.01, max=1000000.0, precision=1, unit="m")
    shift_x = FloatProperty("横向偏移", default=0.0, min=-10.0, max=10.0, precision=3)
    shift_y = FloatProperty("纵向偏移", default=0.0, min=-10.0, max=10.0, precision=3)
    use_dof = BoolProperty("景深", default=False)
    focus_distance = FloatProperty("对焦距离", default=10.0, min=0.0, max=100000.0, precision=3, unit="m")
    fstop = FloatProperty("光圈 F 值", default=2.8, min=0.1, max=128.0, precision=1,
                          description="越小背景越虚")
    display_size = FloatProperty("显示大小", default=1.0, min=0.01, max=1000.0, precision=2)


class LightData(PropertyGroup):
    undoable = True
    type = EnumProperty("类型", items=[("POINT", "点光", "从一点向四周发光"), ("SUN", "日光", "无限远的平行光"),
                                     ("SPOT", "聚光", "锥形光"), ("AREA", "面光", "一块发光的面")], default="POINT")
    color = ColorProperty("颜色", default=(1.0, 1.0, 1.0))
    power = FloatProperty("功率", default=1000.0, min=0.0, max=1e9, soft_max=10000.0, precision=1, unit="W",
                          description="点光、聚光、面光的总功率")
    strength = FloatProperty("强度", default=3.0, min=0.0, max=1e6, soft_max=20.0, precision=2,
                             description="日光的强度")
    radius = FloatProperty("半径", default=0.1, min=0.0, max=1000.0, precision=3, unit="m",
                           description="越大阴影边缘越软")
    angle = FloatProperty("角直径", default=0.526, min=0.0, max=180.0, precision=3, unit="°",
                          description="日光的角直径，越大阴影越软")
    spot_size = FloatProperty("光束角", default=45.0, min=1.0, max=180.0, precision=1, unit="°")
    spot_blend = FloatProperty("边缘柔和", default=0.15, min=0.0, max=1.0, precision=2, subtype="FACTOR")
    size = FloatProperty("大小", default=1.0, min=0.0, max=10000.0, precision=3, unit="m")
    use_shadow = BoolProperty("投射阴影", default=True)


class SceneObject(PropertyGroup):
    """摄像机、灯光、空物体。kind：CAMERA / LIGHT / EMPTY。"""

    KINDS = ("CAMERA", "LIGHT", "EMPTY")
    undoable = True
    name = StringProperty("名称", default="物体")
    visible = BoolProperty("在视口中显示", default=True)
    select = BoolProperty("选中", default=False, hidden=True)
    empty_display_type = EnumProperty("显示为", items=EMPTY_DISPLAY_ITEMS, default="PLAIN_AXES",
                                      description="空物体在视口里画成什么形状")
    empty_size = FloatProperty("显示大小", default=1.0, min=0.0001, max=1000000.0, precision=3, unit="m")

    def __init__(self, kind: str, name: str = "", uid: int | None = None) -> None:
        from .project import new_uid

        super().__init__()
        if kind not in self.KINDS:
            raise ValueError("不认识的物体类型 %s" % kind)
        self.uid = uid if uid is not None else new_uid()
        self.kind = kind
        self.name = name or {"CAMERA": "摄像机", "LIGHT": "灯光", "EMPTY": "空物体"}[kind]
        self.transform = ObjectTransform()
        self.data: Any = CameraData() if kind == "CAMERA" else (LightData() if kind == "LIGHT" else None)

    def to_dict(self) -> dict:
        out = {"uid": self.uid, "kind": self.kind, "name": self.name, "visible": self.visible, "select": self.select,
               "transform": self.transform.to_dict(), "empty_size": self.empty_size,
               "empty_display_type": self.empty_display_type}
        if self.data is not None:
            out["data"] = self.data.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "SceneObject":
        from .project import reserve_uid

        uid = int(data.get("uid", 0)) or None
        if uid is not None:
            reserve_uid(uid)
        obj = cls(str(data.get("kind", "EMPTY")), str(data.get("name", "")), uid)
        obj.visible = bool(data.get("visible", True))
        obj.select = bool(data.get("select", False))
        obj.transform.from_dict(data.get("transform") or {})
        obj.empty_size = float(data.get("empty_size", 1.0))
        obj.empty_display_type = str(data.get("empty_display_type", "PLAIN_AXES"))
        if obj.data is not None:
            obj.data.from_dict(data.get("data") or {})
        return obj


def camera_fov(data: CameraData, aspect: float) -> float:
    """摄像机的竖直视角（弧度），传感器宽度按画面的长边算（和 Blender 的「自动」一样）。"""
    horizontal = 2.0 * math.atan(float(data.sensor) * 0.5 / max(1e-6, float(data.lens)))
    if aspect >= 1.0:
        return 2.0 * math.atan(math.tan(horizontal * 0.5) / aspect)
    return horizontal



# ====================================================================== 显示坐标（Blender，Z 朝上）↔ 内部坐标（Y 朝上）
#: 显示 = D · 内部。内部 +Y（上）是 Blender 的 +Z，内部 +Z（朝向前视图的观察者）是 Blender 的 -Y
D = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def to_display(vector) -> np.ndarray:
    return D @ np.asarray(vector, np.float64)


def to_internal(vector) -> np.ndarray:
    return D.T @ np.asarray(vector, np.float64)


def display_to_internal_matrix(matrix: np.ndarray) -> np.ndarray:
    """显示坐标里的 4×4 变换 → 内部坐标里的同一个变换。"""
    t = np.identity(4)
    t[:3, :3] = D
    return t.T @ np.asarray(matrix, np.float64) @ t


def local_to_internal_matrix(matrix: np.ndarray) -> np.ndarray:
    """物体矩阵（物体自己的局部坐标 → 世界显示坐标，和 Blender 一样）→ 局部坐标 → 世界内部坐标。
    摄像机、灯光、空物体的线框和朝向都按 Blender 的局部轴定义（摄像机看局部 -Z、上方是局部 +Y）。"""
    t = np.identity(4)
    t[:3, :3] = D
    return t.T @ np.asarray(matrix, np.float64)


def internal_to_display_matrix(matrix: np.ndarray) -> np.ndarray:
    t = np.identity(4)
    t[:3, :3] = D
    return t @ np.asarray(matrix, np.float64) @ t.T
