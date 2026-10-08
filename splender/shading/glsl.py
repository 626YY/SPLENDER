"""材质节点的着色器函数库：放进合成器着色器里（名字都以 mg_ 开头，和合成器已有的函数分开）。

用到合成器里已有的：g_mp（位置，内部坐标）、g_mn（法线）、g_uv、g_ma（遮蔽、曲率、厚度）、g_id、u_bmin/u_bmax、
gnoise(vec3)、rgb2hsv / hsv2rgb、srgb_to_linear / linear_to_srgb、graph_tex、u_graph_res、u_level_texels。
材质参数在存储缓冲 mg[]（每个值一个 vec4）；色带一块 41 个 vec4：(个数, 方式)、8 个装位置、32 个装颜色。
"""

LIBRARY = """
layout(std430, binding=3) readonly buffer MaterialValues { vec4 mg[]; };
float mg_lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
// 坐标都换成 Blender 的 Z 朝上；模型贴图还没烘焙好时先按 UV 算（烘焙好了自动换成按表面算）
vec3 mg_position() { return u_has_maps == 1 ? vec3(g_mp.x, -g_mp.z, g_mp.y) : vec3(g_uv, 0.0); }
vec3 mg_normal() { return vec3(g_mn.x, -g_mn.z, g_mn.y); }
vec3 mg_generated() {
  if (u_has_maps == 0) return vec3(g_uv, 0.0);
  vec3 g = (g_mp - u_bmin) / max(u_bmax - u_bmin, vec3(1e-6));
  return vec3(g.x, 1.0 - g.z, g.y);
}
float mg_fp(vec3 p) { return max(length(dFdx(p)), length(dFdy(p))); }
int mg_imod(int a, int n) { return n <= 0 ? 0 : (a >= 0 ? a % n : (n - 1) - ((-a - 1) % n)); }
vec3 mg_hash3(vec3 p) {
  return fract(sin(vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)),
                        dot(p, vec3(113.5, 271.9, 124.6)))) * 43758.5453123);
}
float mg_hash1(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

// ---- 色带（和 core/ramp.py 一样的算法）----
vec4 mg_ramp(int base, float t) {
  int n = int(mg[base].x + 0.5);
  int mode = int(mg[base].y + 0.5);
  if (n <= 0) return vec4(0.0);
  t = clamp(t, 0.0, 1.0);
  float p0 = mg[base + 1].x;
  if (n == 1 || t <= p0) return mg[base + 9];
  float plast = mg[base + 1 + (n - 1) / 4][(n - 1) % 4];
  if (t >= plast) return mg[base + 9 + n - 1];
  int i = 0;
  for (int k = 0; k < 31; k++) {
    if (k + 1 >= n) break;
    float pk1 = mg[base + 1 + (k + 1) / 4][(k + 1) % 4];
    if (t < pk1) { i = k; break; }
    i = k;
  }
  i = min(i, n - 2);
  float a = mg[base + 1 + i / 4][i % 4];
  float b = mg[base + 1 + (i + 1) / 4][(i + 1) % 4];
  float f = b - a > 1e-9 ? clamp((t - a) / (b - a), 0.0, 1.0) : 0.0;
  vec4 c1 = mg[base + 9 + i], c2 = mg[base + 9 + i + 1];
  if (mode == 4) return f >= 1.0 ? c2 : c1;
  if (mode == 1 || mode == 3) {
    vec4 c0 = mg[base + 9 + max(i - 1, 0)], c3 = mg[base + 9 + min(i + 2, n - 1)];
    float f2 = f * f, f3 = f2 * f;
    vec4 r = mode == 1
      ? 0.5 * (2.0 * c1 + (-c0 + c2) * f + (2.0 * c0 - 5.0 * c1 + 4.0 * c2 - c3) * f2 + (-c0 + 3.0 * c1 - 3.0 * c2 + c3) * f3)
      : ((-c0 + 3.0 * c1 - 3.0 * c2 + c3) * f3 + (3.0 * c0 - 6.0 * c1 + 3.0 * c2) * f2 + (-3.0 * c0 + 3.0 * c2) * f
         + (c0 + 4.0 * c1 + c2)) / 6.0;
    return clamp(r, 0.0, 1.0);
  }
  if (mode == 0) f = f * f * (3.0 - 2.0 * f);
  return mix(c1, c2, f);
}

// ---- 噪波（分形，照 Blender：细节是倍频数，可以带小数；比纹素还细的倍频淡出）----
float mg_snoise(vec3 p) { return clamp(gnoise(p) * 1.4, -1.0, 1.0); }
float mg_fbm(vec3 p, float detail, float rough, float fp) {
  float octaves = clamp(detail, 0.0, 15.0);
  int n = int(floor(octaves));
  float fscale = 1.0, amp = 1.0, maxamp = 0.0, sum = 0.0;
  for (int i = 0; i <= 15; i++) {
    if (i > n) break;
    float fade = clamp(1.5 - fp * fscale * 2.0, 0.0, 1.0);
    sum += mg_snoise(p * fscale) * amp * fade;
    maxamp += amp;
    amp *= clamp(rough, 0.0, 1.0);
    fscale *= 2.0;
  }
  float result = 0.5 * sum / maxamp + 0.5;
  float rmd = octaves - float(n);
  if (rmd > 0.0) {
    float fade = clamp(1.5 - fp * fscale * 2.0, 0.0, 1.0);
    float sum2 = sum + mg_snoise(p * fscale) * amp * fade;
    result = mix(result, 0.5 * sum2 / (maxamp + amp) + 0.5, rmd);
  }
  return result;
}
void mg_tex_noise(vec3 p, float scale, float detail, float rough, float distortion, out float fac, out vec3 color) {
  p *= scale;
  float fp = mg_fp(p);
  if (distortion != 0.0) {
    p += vec3(mg_snoise(p + vec3(13.5, 2.1, -4.4)), mg_snoise(p + vec3(-3.7, 8.1, 2.3)),
              mg_snoise(p + vec3(5.2, -11.7, 7.9))) * distortion;
  }
  fac = mg_fbm(p, detail, rough, fp);
  color = vec3(fac, mg_fbm(p + vec3(31.4, 17.1, 5.3), detail, rough, fp),
               mg_fbm(p + vec3(-9.6, 23.3, 41.7), detail, rough, fp));
}

// ---- 沃罗诺伊（细胞）----
float mg_dist(vec3 d, int metric) {
  if (metric == 1) return abs(d.x) + abs(d.y) + abs(d.z);
  if (metric == 2) return max(abs(d.x), max(abs(d.y), abs(d.z)));
  return length(d);
}
void mg_tex_voronoi(vec3 p, float scale, float randomness, int metric, int feature,
                    out float dist, out vec3 color, out vec3 pos) {
  p *= scale;
  vec3 cell = floor(p);
  vec3 f = p - cell;
  float d1 = 1e9, d2 = 1e9;
  vec3 c1 = vec3(0.0), o1 = vec3(0.0);
  for (int z = -1; z <= 1; z++)
  for (int y = -1; y <= 1; y++)
  for (int x = -1; x <= 1; x++) {
    vec3 g = vec3(float(x), float(y), float(z));
    vec3 o = mg_hash3(cell + g) * clamp(randomness, 0.0, 1.0);
    float d = mg_dist(g + o - f, metric);
    if (d < d1) { d2 = d1; d1 = d; c1 = cell + g; o1 = g + o; }
    else if (d < d2) { d2 = d; }
  }
  dist = feature == 1 ? d2 : (feature == 2 ? d2 - d1 : d1);
  color = mg_hash3(c1 + vec3(7.0, 3.0, 11.0));
  pos = (cell + o1) / max(scale, 1e-6);
}

// ---- 波浪 ----
float mg_tex_wave(vec3 p, float scale, float distortion, float detail, float dscale, float drough, float phase,
                  int type, int dir, int profile) {
  p = (p * scale + 0.000001) * 0.999999;
  float n;
  if (type == 0) {
    if (dir == 0) n = p.x * 20.0;
    else if (dir == 1) n = p.y * 20.0;
    else if (dir == 2) n = p.z * 20.0;
    else n = (p.x + p.y + p.z) * 10.0;
  } else {
    vec3 rp = p;
    if (dir == 0) rp *= vec3(0.0, 1.0, 1.0);
    else if (dir == 1) rp *= vec3(1.0, 0.0, 1.0);
    else if (dir == 2) rp *= vec3(1.0, 1.0, 0.0);
    n = length(rp) * 20.0;
  }
  n += phase;
  if (distortion != 0.0) {
    vec3 q = p * dscale;
    n += distortion * (mg_fbm(q, detail, drough, mg_fp(q)) * 2.0 - 1.0);
  }
  if (profile == 0) return 0.5 + 0.5 * sin(n - 1.5707963);
  if (profile == 1) { n /= 6.2831853; return n - floor(n); }
  n /= 6.2831853;
  return abs(n - floor(n + 0.5)) * 2.0;
}

// ---- 渐变 ----
float mg_tex_gradient(vec3 p, int type) {
  float x = p.x, y = p.y, z = p.z;
  if (type == 0) return clamp(x, 0.0, 1.0);
  if (type == 1) { float r = max(x, 0.0); return clamp(r * r, 0.0, 1.0); }
  if (type == 2) { float r = clamp(x, 0.0, 1.0); float t = r * r; return 3.0 * t - 2.0 * t * r; }
  if (type == 3) return clamp((x + y) * 0.5, 0.0, 1.0);
  if (type == 6) return clamp(atan(y, x) / 6.2831853 + 0.5, 0.0, 1.0);
  float r = max(0.999999 - sqrt(x * x + y * y + z * z), 0.0);
  if (type == 5) return clamp(r * r, 0.0, 1.0);
  return clamp(r, 0.0, 1.0);
}

// ---- 棋盘格、砖块 ----
float mg_tex_checker(vec3 p, float scale) {
  p = (p * scale + 0.000001) * 0.999999;
  int xi = int(abs(floor(p.x))), yi = int(abs(floor(p.y))), zi = int(abs(floor(p.z)));
  bool a = (xi % 2) == (yi % 2);
  bool b = (zi % 2) == 1;
  return a == b ? 1.0 : 0.0;
}
vec2 mg_tex_brick(vec3 p, float mortar_size, float mortar_smooth, float bias, float brick_width, float row_height,
                  float offset_amount, int offset_frequency, float squash_amount, int squash_frequency) {
  float rownum = floor(p.y / row_height);
  int row = int(rownum);
  float offset = 0.0;
  if (offset_frequency != 0 && squash_frequency != 0) {
    brick_width *= mg_imod(row, squash_frequency) != 0 ? 1.0 : squash_amount;
    offset = mg_imod(row, offset_frequency) != 0 ? 0.0 : brick_width * offset_amount;
  }
  float bricknum = floor((p.x + offset) / brick_width);
  float x = (p.x + offset) - brick_width * bricknum;
  float y = p.y - row_height * rownum;
  float tint = clamp(mg_hash1(vec2(rownum, bricknum)) + bias, 0.0, 1.0);
  float min_dist = min(min(x, y), min(brick_width - x, row_height - y));
  float mortar;
  if (min_dist >= mortar_size) mortar = 0.0;
  else if (mortar_smooth == 0.0) mortar = 1.0;
  else mortar = smoothstep(0.0, mortar_smooth, 1.0 - min_dist / mortar_size);
  return vec2(tint, mortar);
}

// ---- 颜色混合（照 Blender 混合颜色节点的公式）----
vec3 mg_mix(int mode, float fac, vec3 a, vec3 b) {
  float facm = 1.0 - fac;
  if (mode == 0) return mix(a, b, fac);
  if (mode == 1) return mix(a, min(a, b), fac);
  if (mode == 2) return a * (facm + fac * b);
  if (mode == 3) {
    vec3 tmp = facm + fac * b;
    return vec3(tmp.x <= 0.0 ? 0.0 : clamp(1.0 - (1.0 - a.x) / tmp.x, 0.0, 1.0),
                tmp.y <= 0.0 ? 0.0 : clamp(1.0 - (1.0 - a.y) / tmp.y, 0.0, 1.0),
                tmp.z <= 0.0 ? 0.0 : clamp(1.0 - (1.0 - a.z) / tmp.z, 0.0, 1.0));
  }
  if (mode == 4) return mix(a, max(a, b), fac);
  if (mode == 5) return 1.0 - (facm + fac * (1.0 - b)) * (1.0 - a);
  if (mode == 6) {
    vec3 r = a;
    for (int k = 0; k < 3; k++) {
      if (a[k] != 0.0) {
        float tmp = 1.0 - fac * b[k];
        r[k] = tmp <= 0.0 ? 1.0 : min(a[k] / tmp, 1.0);
      }
    }
    return r;
  }
  if (mode == 7) return a + b * fac;
  if (mode == 8) {
    vec3 r;
    for (int k = 0; k < 3; k++)
      r[k] = a[k] < 0.5 ? a[k] * (facm + 2.0 * fac * b[k]) : 1.0 - (facm + 2.0 * fac * (1.0 - b[k])) * (1.0 - a[k]);
    return r;
  }
  if (mode == 9) {
    vec3 scr = 1.0 - (1.0 - b) * (1.0 - a);
    return facm * a + fac * ((1.0 - a) * b * a + a * scr);
  }
  if (mode == 10) return a + fac * (2.0 * b - 1.0);
  if (mode == 11) return mix(a, abs(a - b), fac);
  if (mode == 12) return max(mix(a, a + b - 2.0 * a * b, fac), 0.0);
  if (mode == 13) return a - b * fac;
  if (mode == 14) return mix(a, vec3(b.x != 0.0 ? a.x / b.x : 0.0, b.y != 0.0 ? a.y / b.y : 0.0,
                                     b.z != 0.0 ? a.z / b.z : 0.0), fac);
  vec3 ha = rgb2hsv(max(a, 0.0));
  vec3 hb = rgb2hsv(max(b, 0.0));
  if (mode == 15) {
    if (hb.y == 0.0) return a;
    return mix(a, hsv2rgb(vec3(hb.x, ha.y, ha.z)), fac);
  }
  if (mode == 16) {
    if (ha.y == 0.0) return a;
    return mix(a, hsv2rgb(vec3(ha.x, mix(ha.y, hb.y, fac), ha.z)), 1.0);
  }
  if (mode == 17) {
    if (hb.y == 0.0) return a;
    return mix(a, hsv2rgb(vec3(hb.x, hb.y, ha.z)), fac);
  }
  return mix(a, hsv2rgb(vec3(ha.x, ha.y, hb.z)), fac);
}

// ---- 运算（照 Blender 运算节点）----
float mg_safe_div(float a, float b) { return b != 0.0 ? a / b : 0.0; }
float mg_smoothmin(float a, float b, float c) {
  if (c == 0.0) return min(a, b);
  float h = max(c - abs(a - b), 0.0) / c;
  return min(a, b) - h * h * h * c * (1.0 / 6.0);
}
float mg_wrap(float v, float lo, float hi) { float r = hi - lo; return r != 0.0 ? v - r * floor((v - lo) / r) : lo; }
float mg_math(int op, float a, float b, float c) {
  switch (op) {
    case 0: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    case 3: return mg_safe_div(a, b);
    case 4: return a * b + c;
    case 5: return (a < 0.0 && b != floor(b)) ? 0.0 : pow(abs(a), b) * ((a < 0.0 && mod(b, 2.0) == 1.0) ? -1.0 : 1.0);
    case 6: return (a > 0.0 && b > 0.0 && b != 1.0) ? log(a) / log(b) : 0.0;
    case 7: return a > 0.0 ? sqrt(a) : 0.0;
    case 8: return a > 0.0 ? inversesqrt(a) : 0.0;
    case 9: return abs(a);
    case 10: return exp(a);
    case 11: return min(a, b);
    case 12: return max(a, b);
    case 13: return a < b ? 1.0 : 0.0;
    case 14: return a > b ? 1.0 : 0.0;
    case 15: return sign(a);
    case 16: return abs(a - b) <= max(c, 1e-5) ? 1.0 : 0.0;
    case 17: return mg_smoothmin(a, b, c);
    case 18: return -mg_smoothmin(-a, -b, c);
    case 19: return floor(a + 0.5);
    case 20: return floor(a);
    case 21: return ceil(a);
    case 22: return trunc(a);
    case 23: return fract(a);
    case 24: return b != 0.0 ? a - b * trunc(a / b) : 0.0;
    case 25: return b != 0.0 ? a - b * floor(a / b) : 0.0;
    case 26: return mg_wrap(a, c, b);
    case 27: return b != 0.0 ? floor(a / b) * b : 0.0;
    case 28: return b != 0.0 ? abs(fract((a - b) / (b * 2.0)) * b * 2.0 - b) : 0.0;
    case 29: return sin(a);
    case 30: return cos(a);
    case 31: return tan(a);
    case 32: return asin(clamp(a, -1.0, 1.0));
    case 33: return acos(clamp(a, -1.0, 1.0));
    case 34: return atan(a);
    case 35: return atan(a, b);
    case 36: return sinh(a);
    case 37: return cosh(a);
    case 38: return tanh(a);
    case 39: return radians(a);
    case 40: return degrees(a);
  }
  return a;
}
void mg_vmath(int op, vec3 a, vec3 b, float s, out vec3 v, out float f) {
  v = vec3(0.0);
  f = 0.0;
  if (op == 0) v = a + b;
  else if (op == 1) v = a - b;
  else if (op == 2) v = a * b;
  else if (op == 3) v = vec3(mg_safe_div(a.x, b.x), mg_safe_div(a.y, b.y), mg_safe_div(a.z, b.z));
  else if (op == 4) v = cross(a, b);
  else if (op == 5) { float d = dot(b, b); v = d != 0.0 ? b * (dot(a, b) / d) : vec3(0.0); }
  else if (op == 6) { vec3 n = length(b) > 0.0 ? normalize(b) : vec3(0.0); v = reflect(a, n); }
  else if (op == 7) f = dot(a, b);
  else if (op == 8) f = distance(a, b);
  else if (op == 9) f = length(a);
  else if (op == 10) v = a * s;
  else if (op == 11) v = length(a) > 0.0 ? normalize(a) : vec3(0.0);
  else if (op == 12) v = abs(a);
  else if (op == 13) v = min(a, b);
  else if (op == 14) v = max(a, b);
  else if (op == 15) v = floor(a);
  else if (op == 16) v = ceil(a);
  else if (op == 17) v = fract(a);
  else if (op == 18) v = vec3(b.x != 0.0 ? mod(a.x, b.x) : 0.0, b.y != 0.0 ? mod(a.y, b.y) : 0.0,
                              b.z != 0.0 ? mod(a.z, b.z) : 0.0);
  else if (op == 19) v = vec3(b.x != 0.0 ? floor(a.x / b.x) * b.x : 0.0, b.y != 0.0 ? floor(a.y / b.y) * b.y : 0.0,
                              b.z != 0.0 ? floor(a.z / b.z) * b.z : 0.0);
  else if (op == 20) v = sin(a);
  else if (op == 21) v = cos(a);
  else if (op == 22) v = tan(a);
}
float mg_map_range(int mode, float v, float fmin, float fmax, float tmin, float tmax, float steps, bool clamp_it) {
  float range = fmax - fmin;
  float f = range != 0.0 ? (v - fmin) / range : 0.0;
  if (mode == 1) f = steps > 0.0 ? floor(f * (steps + 1.0)) / steps : 0.0;
  else if (mode == 2) { float t = clamp(f, 0.0, 1.0); f = t * t * (3.0 - 2.0 * t); }
  else if (mode == 3) { float t = clamp(f, 0.0, 1.0); f = t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
  float r = tmin + f * (tmax - tmin);
  if (clamp_it && mode < 2) r = tmin <= tmax ? clamp(r, tmin, tmax) : clamp(r, tmax, tmin);
  return r;
}
mat3 mg_euler(vec3 r) {
  float cx = cos(r.x), sx = sin(r.x), cy = cos(r.y), sy = sin(r.y), cz = cos(r.z), sz = sin(r.z);
  mat3 rx = mat3(1.0, 0.0, 0.0, 0.0, cx, sx, 0.0, -sx, cx);
  mat3 ry = mat3(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy);
  mat3 rz = mat3(cz, sz, 0.0, -sz, cz, 0.0, 0.0, 0.0, 1.0);
  return rz * ry * rx;
}
vec3 mg_mapping(int type, vec3 v, vec3 loc, vec3 rot, vec3 scale) {
  mat3 r = mg_euler(radians(rot));
  if (type == 0) return r * (v * scale) + loc;
  if (type == 1) return (transpose(r) * (v - loc)) / vec3(scale.x != 0.0 ? scale.x : 1e-9, scale.y != 0.0 ? scale.y : 1e-9,
                                                          scale.z != 0.0 ? scale.z : 1e-9);
  if (type == 2) return r * (v * scale);
  vec3 n = r * (v / vec3(scale.x != 0.0 ? scale.x : 1e-9, scale.y != 0.0 ? scale.y : 1e-9, scale.z != 0.0 ? scale.z : 1e-9));
  return length(n) > 0.0 ? normalize(n) : n;
}
vec4 mg_graph(int slot, vec3 uvw, float tiling, int layer) {
  if (slot < 0) return vec4(0.0);
  vec2 uv = uvw.xy * tiling;
  float lod = log2(max(u_graph_res[slot] * tiling / u_level_texels, 1e-6));
  return graph_tex(slot, vec3(uv, float(layer)), lod);
}
"""
