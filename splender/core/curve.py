"""曲线（照 Photoshop / Blender 的曲线调整）：一串控制点 (输入, 输出)，按输入排好，至少两个点；
点和点之间用保单调的三次插值（Fritsch–Carlson），不会冲过头；两头以外取端点的值。

值的形式（可比较、可撤销）：((x, y), ...)，x、y 都在 0..1。
"""
from __future__ import annotations

import numpy as np

DEFAULT = ((0.0, 0.0), (1.0, 1.0))
MAX_POINTS = 32


def normalize(value) -> tuple:
    try:
        points = []
        for item in value:
            x, y = float(item[0]), float(item[1])
            points.append((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))))
    except (TypeError, ValueError, IndexError):
        return DEFAULT
    if len(points) < 2:
        return DEFAULT
    points.sort(key=lambda p: p[0])
    # 输入一样的点只留一个
    out = [points[0]]
    for p in points[1:]:
        if p[0] - out[-1][0] > 1e-6:
            out.append(p)
    if len(out) < 2:
        return DEFAULT
    return tuple(out[:MAX_POINTS])


def _slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h = np.diff(xs)
    delta = np.diff(ys) / np.maximum(h, 1e-12)
    n = len(xs)
    m = np.zeros(n)
    if n == 2:
        m[:] = delta[0]
        return m
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    return m


def evaluate(value, t) -> np.ndarray:
    points = normalize(value)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    t = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    flat = t.reshape(-1)
    m = _slopes(xs, ys)
    i = np.clip(np.searchsorted(xs, flat, side="right") - 1, 0, len(xs) - 2)
    x0, x1 = xs[i], xs[i + 1]
    h = np.maximum(x1 - x0, 1e-12)
    s = np.clip((flat - x0) / h, 0.0, 1.0)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    out = h00 * ys[i] + h10 * h * m[i] + h01 * ys[i + 1] + h11 * h * m[i + 1]
    out = np.where(flat <= xs[0], ys[0], out)
    out = np.where(flat >= xs[-1], ys[-1], out)
    return np.clip(out, 0.0, 1.0).reshape(t.shape)


def lut(value, size: int = 256) -> np.ndarray:
    return evaluate(value, np.linspace(0.0, 1.0, size))


def to_json(value) -> list:
    return [list(p) for p in normalize(value)]
