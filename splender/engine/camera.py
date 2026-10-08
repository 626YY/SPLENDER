"""视口相机：绕目标点的轨道相机，透视与正交可切换。世界坐标 Y 轴向上。"""
from __future__ import annotations

import math

import numpy as np

SENSOR = 36.0   # 与 Blender 视口一致：焦距 50 时长边视角约 71.5°

AXIS_VIEWS = {
    "FRONT": (0.0, 0.0), "BACK": (180.0, 0.0), "RIGHT": (90.0, 0.0), "LEFT": (-90.0, 0.0),
    "TOP": (0.0, 89.999), "BOTTOM": (0.0, -89.999),
}
AXIS_LABELS = {"FRONT": "前", "BACK": "后", "RIGHT": "右", "LEFT": "左", "TOP": "顶", "BOTTOM": "底"}


def perspective(tan_half_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / tan_half_y
    m = np.zeros((4, 4), np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def orthographic(half_h: float, aspect: float, near: float, far: float) -> np.ndarray:
    m = np.identity(4, np.float64)
    m[0, 0] = 1.0 / (half_h * aspect)
    m[1, 1] = 1.0 / half_h
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


class Camera:
    def __init__(self) -> None:
        self.target = np.zeros(3, np.float64)
        self.distance = 3.0
        self.yaw = 35.0          # 度，绕 Y 轴
        self.pitch = 20.0        # 度，抬头为正
        self.lens = 50.0
        self.ortho = False
        self.scene_radius = 1.0
        self.scene_center = np.zeros(3, np.float64)   # 场景包围球（远近裁剪按它算，物体移远了也看得见）
        self.axis_view = ""      # 处于标准视图时的名字，旋转后清空
        self.roll = 0.0          # 度，绕视线滚转
        self.offset = (0.0, 0.0)  # 镜头偏移（屏幕归一化坐标），摄像机视图的「偏移」用
        self.clip_override = None  # (近, 远)：从摄像机看时用摄像机自己的裁剪距离

    # ---- 基向量 ----
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (right, up, back)：相机坐标系的三个轴在世界里的方向，back 指向相机后方。"""
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        back = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
        right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
        up = np.cross(back, right)
        up = up / np.linalg.norm(up)
        if self.roll:
            r = math.radians(self.roll)
            c, s = math.cos(r), math.sin(r)
            right, up = right * c + up * s, up * c - right * s
        return right, up, back

    @property
    def eye(self) -> np.ndarray:
        return self.target + self.basis()[2] * self.distance

    @property
    def forward(self) -> np.ndarray:
        return -self.basis()[2]

    # ---- 矩阵 ----
    def view_matrix(self) -> np.ndarray:
        right, up, back = self.basis()
        eye = self.target + back * self.distance
        m = np.identity(4, np.float64)
        m[0, :3], m[1, :3], m[2, :3] = right, up, back
        m[0, 3], m[1, 3], m[2, 3] = -right @ eye, -up @ eye, -back @ eye
        return m

    def tan_half(self, aspect: float) -> tuple[float, float]:
        """(竖直方向半视角正切, 水平方向半视角正切)。焦距对应长边。"""
        major = SENSOR / max(1.0, self.lens)
        if aspect >= 1.0:
            return major / aspect, major
        return major, major * aspect

    def clip_range(self) -> tuple[float, float]:
        if self.clip_override is not None and not self.ortho:
            near, far = self.clip_override
            return max(float(near), 1e-6), max(float(far), float(near) * 2.0)
        radius = max(self.scene_radius, 1e-6)
        reach = radius + float(np.linalg.norm(self.target - self.scene_center))
        near = max(radius * 1e-6, self.distance * 0.005)
        far = self.distance + reach * 3.0 + near * 100.0
        if self.ortho:
            near = -(reach * 3.0 + self.distance)
        return near, far

    def projection_matrix(self, aspect: float) -> np.ndarray:
        near, far = self.clip_range()
        tan_y, _tan_x = self.tan_half(aspect)
        ox, oy = self.offset
        if self.ortho:
            m = orthographic(self.distance * tan_y, aspect, near, far)
            m[0, 3] += ox
            m[1, 3] += oy
            return m
        m = perspective(tan_y, aspect, near, far)
        m[0, 2] -= ox
        m[1, 2] -= oy
        return m

    def view_projection(self, aspect: float) -> np.ndarray:
        return self.projection_matrix(aspect) @ self.view_matrix()

    # ---- 度量 ----
    def world_per_pixel(self, depth: float, viewport_h: float, aspect: float) -> float:
        """视线深度 depth 处，一个屏幕像素对应多长的世界距离。"""
        tan_y, _ = self.tan_half(aspect)
        span = self.distance if self.ortho else max(depth, 1e-9)
        return 2.0 * span * tan_y / max(1.0, viewport_h)

    def depth_of(self, point: np.ndarray) -> float:
        return float((np.asarray(point, np.float64) - self.eye) @ self.forward)

    def ray(self, x: float, y: float, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
        """屏幕像素 (x, y)（原点左上）对应的世界射线 (起点, 方向)。"""
        aspect = width / max(1.0, height)
        tan_y, tan_x = self.tan_half(aspect)
        nx = (x / max(1.0, width)) * 2.0 - 1.0 - self.offset[0]
        ny = 1.0 - (y / max(1.0, height)) * 2.0 - self.offset[1]
        right, up, back = self.basis()
        if self.ortho:
            origin = self.eye + right * (nx * tan_x * self.distance) + up * (ny * tan_y * self.distance)
            return origin, -back
        direction = right * (nx * tan_x) + up * (ny * tan_y) - back
        return self.eye, direction / np.linalg.norm(direction)

    # ---- 操作 ----
    def orbit(self, dx: float, dy: float, sensitivity: float = 0.4) -> None:
        self.yaw = (self.yaw - dx * sensitivity + 180.0) % 360.0 - 180.0
        self.pitch = max(-89.999, min(89.999, self.pitch + dy * sensitivity))
        self.axis_view = ""

    def orbit_around(self, dx: float, dy: float, pivot: np.ndarray, sensitivity: float = 0.4) -> None:
        """绕任意点旋转：保持该点在屏幕上的位置不动。"""
        right, up, back = self.basis()
        offset = np.asarray(pivot, np.float64) - self.target
        local = np.array([offset @ right, offset @ up, offset @ back])
        self.orbit(dx, dy, sensitivity)
        right, up, back = self.basis()
        self.target = np.asarray(pivot, np.float64) - (right * local[0] + up * local[1] + back * local[2])

    def pan(self, dx: float, dy: float, viewport_h: float, aspect: float, depth: float | None = None) -> None:
        scale = self.world_per_pixel(self.distance if depth is None else depth, viewport_h, aspect)
        right, up, _ = self.basis()
        self.target = self.target - right * (dx * scale) + up * (dy * scale)

    def zoom(self, factor: float, toward: np.ndarray | None = None) -> None:
        """factor>1 拉近。toward 是世界里的一个点，缩放后它在屏幕上的位置保持不动。"""
        factor = max(1e-3, factor)
        new_distance = min(max(self.distance / factor, self.scene_radius * 1e-4), self.scene_radius * 200.0)
        ratio = new_distance / self.distance
        if toward is not None:
            toward = np.asarray(toward, np.float64)
            if self.ortho:
                right, up, _ = self.basis()
                offset = toward - self.target
                planar = right * (offset @ right) + up * (offset @ up)
                self.target = self.target + planar * (1.0 - ratio)
            else:
                eye = self.eye
                new_eye = toward + (eye - toward) * ratio
                self.target = new_eye - self.basis()[2] * new_distance
        self.distance = new_distance

    def frame(self, center: np.ndarray, radius: float, aspect: float = 1.5) -> None:
        self.target = np.asarray(center, np.float64).copy()
        tan_y, tan_x = self.tan_half(aspect)
        self.distance = max(radius, 1e-6) / max(1e-6, min(tan_y, tan_x)) * 1.1

    def set_axis(self, name: str) -> None:
        self.yaw, self.pitch = AXIS_VIEWS[name]
        self.roll = 0.0
        self.axis_view = name

    def set_view(self, eye, back, up, distance: float | None = None) -> None:
        """按眼睛位置、朝后方向（视线反方向）和朝上方向摆相机（内部坐标）。目标点在眼前 distance 处。"""
        back = np.asarray(back, np.float64)
        back = back / max(np.linalg.norm(back), 1e-12)
        up = np.asarray(up, np.float64)
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(back[1])))))
        if abs(back[1]) > 0.999999:
            # 正对上下：方位角由朝上方向决定
            yaw = math.degrees(math.atan2(-float(up[0]), -float(up[2]))) if back[1] > 0 else \
                math.degrees(math.atan2(float(up[0]), float(up[2])))
        else:
            yaw = math.degrees(math.atan2(float(back[0]), float(back[2])))
        self.yaw, self.pitch, self.roll = yaw, pitch, 0.0
        right0, up0, _back0 = self.basis()
        self.roll = math.degrees(math.atan2(float(np.cross(up0, up) @ back), float(up0 @ up)))
        if distance is not None:
            self.distance = max(float(distance), 1e-6)
        self.target = np.asarray(eye, np.float64) - back * self.distance
        self.axis_view = ""

    def label(self) -> str:
        kind = "正交" if self.ortho else "透视"
        return (AXIS_LABELS[self.axis_view] + "视图 · " + kind) if self.axis_view else ("用户视图 · " + kind)

    def state(self) -> dict:
        return {"target": [float(v) for v in self.target], "distance": float(self.distance), "yaw": self.yaw,
                "pitch": self.pitch, "roll": self.roll, "lens": self.lens, "ortho": self.ortho}

    def set_state(self, data: dict) -> None:
        if not data:
            return
        self.target = np.asarray(data.get("target", self.target), np.float64)
        self.distance = float(data.get("distance", self.distance))
        self.yaw = float(data.get("yaw", self.yaw))
        self.pitch = float(data.get("pitch", self.pitch))
        self.roll = float(data.get("roll", 0.0))
        self.lens = float(data.get("lens", self.lens))
        self.ortho = bool(data.get("ortho", self.ortho))

    def copy_from(self, other: "Camera") -> None:
        self.set_state(other.state())
        self.scene_radius = other.scene_radius
        self.scene_center = np.asarray(other.scene_center, np.float64).copy()
        self.axis_view = other.axis_view
