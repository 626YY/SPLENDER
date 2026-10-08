"""节点着色器的公共部分：哈希、可平铺的噪波和图案、取输入的函数。

约定：
- 节点在 [0,1)² 的 UV 上求值，所有生成器都按整数周期重复，贴到模型上可以无缝平铺；
- 每个节点最多 4 个输入（u_in0..u_in3），没接线时用默认值 u_def0..u_def3；
- 灰度节点把数值写进 RGB 三个分量（A=1），颜色节点写 sRGB 颜色；
- 参数按名字成为 uniform：浮点 → float p_名字，整数、开关、枚举 → int p_名字，颜色 → vec3 p_名字。
"""

HEADER = """#version 430
uniform vec2 u_size;                 // 输出边长（像素）
uniform sampler2D u_in0; uniform sampler2D u_in1; uniform sampler2D u_in2; uniform sampler2D u_in3;
uniform ivec4 u_linked;              // 每个输入接没接线
uniform vec4 u_def0; uniform vec4 u_def1; uniform vec4 u_def2; uniform vec4 u_def3;
"""

LIBRARY = """
#define PI 3.14159265358979
#define TAU 6.28318530717959

float lum(vec4 c) { return dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)); }
vec4 gray(float v) { return vec4(v, v, v, 1.0); }

vec4 IN0(vec2 uv) { return u_linked.x == 1 ? textureLod(u_in0, uv, 0.0) : u_def0; }
vec4 IN1(vec2 uv) { return u_linked.y == 1 ? textureLod(u_in1, uv, 0.0) : u_def1; }
vec4 IN2(vec2 uv) { return u_linked.z == 1 ? textureLod(u_in2, uv, 0.0) : u_def2; }
vec4 IN3(vec2 uv) { return u_linked.w == 1 ? textureLod(u_in3, uv, 0.0) : u_def3; }
vec4 IN0L(vec2 uv, float lod) { return u_linked.x == 1 ? textureLod(u_in0, uv, lod) : u_def0; }
vec4 IN1L(vec2 uv, float lod) { return u_linked.y == 1 ? textureLod(u_in1, uv, lod) : u_def1; }
float G0(vec2 uv) { return lum(IN0(uv)); }
float G1(vec2 uv) { return lum(IN1(uv)); }
float G2(vec2 uv) { return lum(IN2(uv)); }
float G3(vec2 uv) { return lum(IN3(uv)); }

// ---- 哈希 ----
uvec3 pcg3d(uvec3 v) {
  v = v * 1664525u + 1013904223u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
  return v;
}
vec3 hash3(ivec2 p, int seed) { return vec3(pcg3d(uvec3(ivec3(p, seed) + ivec3(4096)))) * (1.0 / 4294967296.0); }
// GLSL 里负数取余没有定义，负数单独处理
int wrapi(int a, int n) { return n <= 0 ? a : (a >= 0 ? a % n : (n - 1) - ((-a - 1) % n)); }
ivec2 wrap2(ivec2 p, ivec2 n) { return ivec2(wrapi(p.x, n.x), wrapi(p.y, n.y)); }

// ---- 可平铺的梯度噪波：x 的整数部分按 period 重复 ----
vec2 grad2(ivec2 p, int seed) { float a = hash3(p, seed).x * TAU; return vec2(cos(a), sin(a)); }
float gnoise(vec2 x, ivec2 period, int seed) {
  vec2 i = floor(x);
  vec2 f = x - i;
  ivec2 c = ivec2(i);
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  float n00 = dot(grad2(wrap2(c, period), seed), f);
  float n10 = dot(grad2(wrap2(c + ivec2(1, 0), period), seed), f - vec2(1.0, 0.0));
  float n01 = dot(grad2(wrap2(c + ivec2(0, 1), period), seed), f - vec2(0.0, 1.0));
  float n11 = dot(grad2(wrap2(c + ivec2(1, 1), period), seed), f - vec2(1.0, 1.0));
  return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}
// 分形噪波。kind：0 平滑，1 云絮（取绝对值），2 山脊
float fbm(vec2 uv, int scale, int octaves, float persistence, int kind, int seed) {
  float sum = 0.0, amp = 1.0, norm = 0.0;
  int s = max(scale, 1);
  for (int o = 0; o < 12; o++) {
    if (o >= octaves) break;
    float n = gnoise(uv * float(s), ivec2(s), seed + o * 131);
    if (kind == 1) n = abs(n) * 2.0 - 0.5;
    else if (kind == 2) n = 0.5 - abs(n) * 2.0;
    sum += amp * n;
    norm += amp;
    amp *= persistence;
    s *= 2;
  }
  return clamp(sum / max(norm, 1e-6) * 0.9 + 0.5, 0.0, 1.0);
}

// ---- 可平铺的细胞（沃罗诺伊）：返回 (最近距离, 次近距离, 所在细胞的随机数, 0) ----
vec4 voronoi(vec2 uv, int scale, float jitter, int seed) {
  vec2 x = uv * float(scale);
  vec2 i = floor(x);
  vec2 f = x - i;
  float f1 = 8.0, f2 = 8.0, cell = 0.0;
  for (int y = -2; y <= 2; y++) {
    for (int xx = -2; xx <= 2; xx++) {
      ivec2 c = ivec2(i) + ivec2(xx, y);
      vec3 h = hash3(wrap2(c, ivec2(scale)), seed);
      vec2 p = vec2(float(xx), float(y)) + 0.5 + (h.xy - 0.5) * jitter;
      float d = length(p - f);
      if (d < f1) { f2 = f1; f1 = d; cell = h.z; }
      else if (d < f2) { f2 = d; }
    }
  }
  return vec4(f1, f2, cell, 0.0);
}

// ---- 砖块：返回 (带倒角的高度, 每块砖的随机数, 第二个随机数, 到缝的距离) ----
vec4 bricks(vec2 uv, int cols, int rows, float offset, float gap, float bevel, int seed) {
  vec2 g = uv * vec2(float(cols), float(rows));
  float row = floor(g.y);
  g.x += offset * mod(row, 2.0);
  vec2 cell = floor(g);
  vec2 f = g - cell;
  vec2 size = vec2(1.0 / float(cols), 1.0 / float(rows));
  vec2 d = min(f, 1.0 - f) * size;
  float edge = min(d.x, d.y);
  float unit = min(size.x, size.y);
  float half_gap = gap * 0.5 * unit;
  float bev = max(bevel * unit * 0.5, 1e-6);
  float h = clamp((edge - half_gap) / bev, 0.0, 1.0);
  h = h * h * (3.0 - 2.0 * h);
  vec3 r = hash3(wrap2(ivec2(cell), ivec2(cols, rows)), seed);
  return vec4(h, r.x, r.y, edge);
}

vec2 rotate2(vec2 p, float degrees) {
  float a = radians(degrees);
  float c = cos(a), s = sin(a);
  return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}
"""


def uniform_lines(params_class) -> str:
    """参数类 → uniform 声明。"""
    lines = []
    for name, prop in params_class.properties().items():
        kind = getattr(prop, "kind", "")
        if kind == "FLOAT":
            lines.append("uniform float p_%s;" % name)
        elif kind in ("INT", "BOOL", "ENUM"):
            lines.append("uniform int p_%s;" % name)
        elif kind == "COLOR":
            lines.append("uniform vec%d p_%s;" % (getattr(prop, "size", 3), name))
        elif kind == "RAMP":
            from ..core import ramp

            lines.append(ramp.uniform_lines(name) + ramp.glsl_function(name))
    return "\n".join(lines) + "\n"
