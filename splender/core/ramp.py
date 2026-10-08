"""色带（照 Blender 的颜色渐变）：一串色标（位置 0..1、RGBA），按插值方式在色标之间取色。

值的形式（可比较、可撤销）：(插值方式, ((位置, r, g, b, a), ...))，色标按位置排好。颜色和其它颜色属性一样按 sRGB 存。
着色器里：p_名字_n、p_名字_mode、p_名字_pos[MAX_STOPS]、p_名字_col[MAX_STOPS]，函数 ramp_名字(t)。
"""
from __future__ import annotations

import numpy as np

MAX_STOPS = 32
INTERPOLATIONS = [("EASE", "缓入缓出", "色标之间平滑过渡，两头慢"), ("CARDINAL", "基数样条", "穿过每个色标的平滑曲线"),
                  ("LINEAR", "线性", "色标之间直线过渡"), ("B_SPLINE", "B 样条", "更柔和的平滑曲线（不一定穿过色标）"),
                  ("CONSTANT", "常值", "到下一个色标之前保持同一种颜色")]
MODE_INDEX = {key: i for i, (key, _l, _d) in enumerate(INTERPOLATIONS)}
DEFAULT = ("LINEAR", ((0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0)))


def normalize(value) -> tuple:
    """任何像样的输入 → 规范的色带值（色标按位置排、分量夹到 0..1、至少一个色标、最多 MAX_STOPS 个）。"""
    try:
        if isinstance(value, dict):
            mode, stops = value.get("interpolation", "LINEAR"), value.get("stops", ())
        else:
            mode, stops = value
        mode = mode if mode in MODE_INDEX else "LINEAR"
        out = []
        for stop in stops:
            items = [float(v) for v in stop]
            if len(items) == 4:
                items.append(1.0)
            if len(items) < 5:
                continue
            out.append(tuple(float(np.clip(v, 0.0, 1.0)) for v in items[:5]))
        if not out:
            return DEFAULT
        out.sort(key=lambda s: s[0])
        return mode, tuple(out[:MAX_STOPS])
    except (TypeError, ValueError):
        return DEFAULT


def evaluate(value, t) -> np.ndarray:
    """在 t（标量或数组，0..1）处取色，返回 (..., 4) 的 RGBA。和着色器里的算法一样。"""
    mode, stops = normalize(value)
    t = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    pos = np.array([s[0] for s in stops])
    col = np.array([s[1:] for s in stops])
    n = len(stops)
    flat = t.reshape(-1)
    out = np.empty((len(flat), 4))
    if n == 1:
        out[:] = col[0]
        return out.reshape(t.shape + (4,))
    i = np.clip(np.searchsorted(pos, flat, side="right") - 1, 0, n - 2)
    a, b = pos[i], pos[i + 1]
    f = np.where(b - a > 1e-9, (flat - a) / np.maximum(b - a, 1e-9), 0.0)
    f = np.clip(f, 0.0, 1.0)
    c1, c2 = col[i], col[i + 1]
    if mode == "CONSTANT":
        out = np.where((f >= 1.0)[:, None], c2, c1)
    elif mode in ("CARDINAL", "B_SPLINE"):
        c0 = col[np.maximum(i - 1, 0)]
        c3 = col[np.minimum(i + 2, n - 1)]
        f1 = f[:, None]
        f2, f3 = f1 * f1, f1 * f1 * f1
        if mode == "CARDINAL":
            out = 0.5 * (2.0 * c1 + (-c0 + c2) * f1 + (2.0 * c0 - 5.0 * c1 + 4.0 * c2 - c3) * f2 +
                         (-c0 + 3.0 * c1 - 3.0 * c2 + c3) * f3)
        else:
            out = ((-c0 + 3.0 * c1 - 3.0 * c2 + c3) * f3 + (3.0 * c0 - 6.0 * c1 + 3.0 * c2) * f2 +
                   (-3.0 * c0 + 3.0 * c2) * f1 + (c0 + 4.0 * c1 + c2)) / 6.0
        out = np.clip(out, 0.0, 1.0)
    else:
        if mode == "EASE":
            f = f * f * (3.0 - 2.0 * f)
        out = c1 + (c2 - c1) * f[:, None]
    # 第一个色标之前、最后一个之后保持端点颜色
    out = np.where((flat <= pos[0])[:, None], col[0], out)
    out = np.where((flat >= pos[-1])[:, None], col[-1], out)
    return out.reshape(t.shape + (4,))


def lut(value, size: int = 256) -> np.ndarray:
    """查找表：(size, 4) float32。"""
    return evaluate(value, np.linspace(0.0, 1.0, size)).astype(np.float32)


def add_stop(value, position: float | None = None, active: int = 0) -> tuple[tuple, int]:
    """加一个色标（不给位置时加在当前色标和下一个之间），颜色取那里的颜色。返回 (新值, 新色标的序号)。"""
    mode, stops = normalize(value)
    if len(stops) >= MAX_STOPS:
        return (mode, stops), active
    if position is None:
        k = min(max(active, 0), len(stops) - 1)
        nxt = stops[k + 1][0] if k + 1 < len(stops) else 1.0
        position = (stops[k][0] + nxt) * 0.5 if k + 1 < len(stops) or stops[k][0] < 1.0 else stops[k][0] * 0.5
    position = float(np.clip(position, 0.0, 1.0))
    color = tuple(float(v) for v in evaluate((mode, stops), position))
    new = sorted(stops + ((position,) + color,), key=lambda s: s[0])
    index = next(i for i, s in enumerate(new) if s[0] == position and s[1:] == color)
    return (mode, tuple(new)), index


def remove_stop(value, index: int) -> tuple[tuple, int]:
    mode, stops = normalize(value)
    if len(stops) <= 1 or not (0 <= index < len(stops)):
        return (mode, stops), index
    new = stops[:index] + stops[index + 1:]
    return (mode, new), min(index, len(new) - 1)


def move_stop(value, index: int, position: float) -> tuple[tuple, int]:
    """移动色标；越过别的色标时顺序跟着变，返回它的新序号。"""
    mode, stops = normalize(value)
    if not (0 <= index < len(stops)):
        return (mode, stops), index
    moved = (float(np.clip(position, 0.0, 1.0)),) + stops[index][1:]
    rest = list(stops[:index] + stops[index + 1:])
    new_index = sum(1 for s in rest if s[0] <= moved[0])
    rest.insert(new_index, moved)
    return (mode, tuple(rest)), new_index


def set_color(value, index: int, color) -> tuple:
    mode, stops = normalize(value)
    if not (0 <= index < len(stops)):
        return mode, stops
    rgba = tuple(float(np.clip(v, 0.0, 1.0)) for v in (list(color) + [1.0])[:4])
    new = list(stops)
    new[index] = (stops[index][0],) + rgba
    return mode, tuple(new)


def flip(value) -> tuple:
    mode, stops = normalize(value)
    return mode, tuple(sorted(((1.0 - s[0],) + s[1:] for s in stops), key=lambda s: s[0]))


def to_json(value) -> dict:
    mode, stops = normalize(value)
    return {"interpolation": mode, "stops": [list(s) for s in stops]}


# ---------------------------------------------------------------------- 着色器
def uniform_lines(name: str) -> str:
    return ("uniform int p_%s_n; uniform int p_%s_mode; uniform float p_%s_pos[%d]; uniform vec4 p_%s_col[%d];\n"
            % (name, name, name, MAX_STOPS, name, MAX_STOPS))


def glsl_function(name: str) -> str:
    """vec4 ramp_名字(float t)：和 evaluate 一样的算法。"""
    p = "p_" + name
    return f"""
vec4 ramp_{name}(float t) {{
  int n = {p}_n;
  if (n <= 0) return vec4(0.0);
  t = clamp(t, 0.0, 1.0);
  if (n == 1 || t <= {p}_pos[0]) return {p}_col[0];
  if (t >= {p}_pos[n - 1]) return {p}_col[n - 1];
  int i = 0;
  for (int k = 0; k < {MAX_STOPS - 1}; k++) {{
    if (k + 1 >= n) break;
    if (t < {p}_pos[k + 1]) {{ i = k; break; }}
    i = k;
  }}
  i = min(i, n - 2);
  float a = {p}_pos[i], b = {p}_pos[i + 1];
  float f = b - a > 1e-9 ? clamp((t - a) / (b - a), 0.0, 1.0) : 0.0;
  vec4 c1 = {p}_col[i], c2 = {p}_col[i + 1];
  int mode = {p}_mode;
  if (mode == 4) return f >= 1.0 ? c2 : c1;
  if (mode == 1 || mode == 3) {{
    vec4 c0 = {p}_col[max(i - 1, 0)], c3 = {p}_col[min(i + 2, n - 1)];
    float f2 = f * f, f3 = f2 * f;
    vec4 r = mode == 1
      ? 0.5 * (2.0 * c1 + (-c0 + c2) * f + (2.0 * c0 - 5.0 * c1 + 4.0 * c2 - c3) * f2 + (-c0 + 3.0 * c1 - 3.0 * c2 + c3) * f3)
      : ((-c0 + 3.0 * c1 - 3.0 * c2 + c3) * f3 + (3.0 * c0 - 6.0 * c1 + 3.0 * c2) * f2 + (-3.0 * c0 + 3.0 * c2) * f
         + (c0 + 4.0 * c1 + c2)) / 6.0;
    return clamp(r, 0.0, 1.0);
  }}
  if (mode == 0) f = f * f * (3.0 - 2.0 * f);
  return mix(c1, c2, f);
}}
"""


def set_uniforms(program, name: str, value) -> None:
    """把色带写进 program 的 p_名字_* uniform（没有的就跳过）。"""
    mode, stops = normalize(value)
    p = "p_" + name
    pos = np.zeros(MAX_STOPS, np.float32)
    col = np.zeros((MAX_STOPS, 4), np.float32)
    for k, stop in enumerate(stops):
        pos[k] = stop[0]
        col[k] = stop[1:]
    if p + "_n" in program:
        program[p + "_n"] = len(stops)
    if p + "_mode" in program:
        program[p + "_mode"] = MODE_INDEX[mode]
    # 驱动可能把没用满的数组报得短一些：按它报的长度写
    if p + "_pos" in program:
        uniform = program[p + "_pos"]
        uniform.write(pos[:max(1, int(getattr(uniform, "array_length", MAX_STOPS)))].tobytes())
    if p + "_col" in program:
        uniform = program[p + "_col"]
        uniform.write(col[:max(1, int(getattr(uniform, "array_length", MAX_STOPS)))].tobytes())
