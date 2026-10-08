"""路径追踪的计算着色器。

材质模型（漫反射 + GGX）、可见法线采样、环境光重要性采样和多重重要性采样的写法，改编自
GLSL-PathTracer（https://github.com/knightcrawler25/GLSL-PathTracer ，MIT 许可，Copyright (c) 2019 Asif Ali），
按 SPLENDER 的数据布局（存储缓冲里的 BVH 和三角形、贴图数组、环境图约定）改写成计算着色器。
"""
from ...engine.ibl import GLSL_TONEMAP

STACK = 64

# 求交（BVH 遍历 + 三角形求交）：路径追踪和烘焙共用。用到 nodes、tri_pos 两个存储缓冲和 BIG 常量。
BVH_GLSL = """// ---------------------------------------------------------------- 求交
vec3 safe_inv(vec3 d) {
    vec3 s = vec3(d.x >= 0.0 ? 1.0 : -1.0, d.y >= 0.0 ? 1.0 : -1.0, d.z >= 0.0 ? 1.0 : -1.0);
    vec3 a = max(abs(d), vec3(1e-20));
    return s / a;
}

float box_hit(vec3 bmin, vec3 bmax, vec3 o, vec3 inv, float tmax) {
    vec3 t0 = (bmin - o) * inv;
    vec3 t1 = (bmax - o) * inv;
    vec3 tn3 = min(t0, t1);
    vec3 tf3 = max(t0, t1);
    float tn = max(max(tn3.x, tn3.y), max(tn3.z, 0.0));
    float tf = min(min(tf3.x, tf3.y), min(tf3.z, tmax));
    return tn <= tf ? tn : BIG;
}

// 最近交点（any_hit 为真时找到任何交点就返回）。tri 是排列后的三角形下标，bc 是重心坐标 (u, v)
bool trace(vec3 o, vec3 d, float tmax, bool any_hit, out int tri, out vec2 bc, out float t_hit) {
    vec3 inv = safe_inv(d);
    tri = -1;
    bc = vec2(0.0);
    t_hit = tmax;
    if (box_hit(nodes[0].bmin, nodes[0].bmax, o, inv, tmax) >= BIG)
        return false;
    int stack_node[""" + str(STACK) + """];
    float stack_t[""" + str(STACK) + """];
    int sp = 0;
    int node = 0;
    for (int guard = 0; guard < 1000000; guard++) {
        Node nd = nodes[node];
        if (nd.count > 0) {
            for (int i = 0; i < nd.count; i++) {
                int k = (nd.first + i) * 3;
                vec3 v0 = tri_pos[k].xyz;
                vec3 e1 = tri_pos[k + 1].xyz;
                vec3 e2 = tri_pos[k + 2].xyz;
                vec3 p = cross(d, e2);
                float det = dot(e1, p);
                if (abs(det) < 1e-24) continue;
                float inv_det = 1.0 / det;
                vec3 s = o - v0;
                float u = dot(s, p) * inv_det;
                if (u < 0.0 || u > 1.0) continue;
                vec3 q = cross(s, e1);
                float v = dot(d, q) * inv_det;
                if (v < 0.0 || u + v > 1.0) continue;
                float t = dot(e2, q) * inv_det;
                if (t > 0.0 && t < t_hit) {
                    t_hit = t;
                    tri = nd.first + i;
                    bc = vec2(u, v);
                    if (any_hit) return true;
                }
            }
        } else {
            int l = nd.first;
            int r = nd.first + 1;
            float tl = box_hit(nodes[l].bmin, nodes[l].bmax, o, inv, t_hit);
            float tr = box_hit(nodes[r].bmin, nodes[r].bmax, o, inv, t_hit);
            if (tl < BIG && tr < BIG) {
                if (tl > tr) { int ti = l; l = r; r = ti; float tt = tl; tl = tr; tr = tt; }
                if (sp < """ + str(STACK) + """) { stack_node[sp] = r; stack_t[sp] = tr; sp++; }
                node = l;
                continue;
            } else if (tl < BIG) { node = l; continue; }
            else if (tr < BIG) { node = r; continue; }
        }
        bool found = false;
        while (sp > 0) {
            sp--;
            if (stack_t[sp] < t_hit) { node = stack_node[sp]; found = true; break; }
        }
        if (!found) break;
    }
    return tri >= 0;
}

"""

TRACE = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba32f, binding = 0) uniform image2D u_accum;
// 降噪用的辅助图：第一次击中处的反照率和法线，逐次累加
layout(rgba32f, binding = 2) uniform image2D u_aov_albedo;
layout(rgba32f, binding = 3) uniform image2D u_aov_normal;
uniform int u_aov;

struct Node { vec3 bmin; int first; vec3 bmax; int count; };
layout(std430, binding = 1) readonly buffer Nodes { Node nodes[]; };
layout(std430, binding = 2) readonly buffer TriPos { vec4 tri_pos[]; };     // 每个三角形 3 项：v0、e1、e2
layout(std430, binding = 3) readonly buffer TriAttr { vec4 tri_attr[]; };   // 每个三角形 4 项：法线、UV、材质
struct Mat { vec4 base; vec4 params; ivec4 tex; };                          // params：金属度、粗糙度、高度倍数
layout(std430, binding = 4) readonly buffer Mats { Mat mats[]; };
layout(std430, binding = 5) readonly buffer EnvCdf { float env_cdf[]; };    // 前 H 个：行的累积分布；之后 H*W 个：行内累积分布
// 灯光：a = 位置 + 类型（0 点光 1 日光 2 聚光 3 面光），b = 照射方向 + 半径（日光是半角），
// c = 颜色×强度 + 投影开关，d = 聚光锥半角余弦、边缘柔和宽度、面光两边长，e/f = 面光的两条边方向
struct Light { vec4 a; vec4 b; vec4 c; vec4 d; vec4 e; vec4 f; };
layout(std430, binding = 6) readonly buffer Lights { Light lights[]; };
uniform int u_light_count;
uniform vec2 u_shift;

uniform sampler2DArray u_base_tex;
uniform sampler2DArray u_mr_tex;
uniform sampler2DArray u_height_tex;
uniform sampler2DArray u_nmap_tex;
uniform sampler2D u_env;

uniform ivec2 u_env_size;
uniform int u_has_env;
uniform float u_env_sum;
uniform vec3 u_env_color;
uniform float u_env_rotation;
uniform float u_env_strength;
uniform int u_background;
uniform int u_transparent;
uniform vec3 u_background_color;

uniform vec3 u_eye;
uniform vec3 u_forward;
uniform vec3 u_right;
uniform vec3 u_up;
uniform float u_tan_half_y;
uniform float u_aspect;
uniform int u_ortho;
uniform float u_ortho_half_h;

uniform ivec2 u_size;
uniform ivec2 u_offset;
uniform int u_sample;
uniform uint u_seed;
uniform int u_max_bounces;
uniform float u_clamp;
uniform float u_eps;
uniform float u_texel;
uniform int u_debug;          // 1：纯漫反射（白炉测试用）；2：只写主射线的交点距离和三角形（检查 BVH）

#define PI 3.14159265358979323
#define INV_PI 0.31830988618379067
#define TWO_PI 6.28318530717958648
#define BIG 1e30

// ---------------------------------------------------------------- 随机数（PCG）
uint g_rng;
uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rnd() {
    g_rng = pcg(g_rng);
    return float(g_rng) * (1.0 / 4294967296.0);
}

float lum(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

""" + BVH_GLSL + """// ---------------------------------------------------------------- 交点信息
struct Hit { vec3 p; vec3 ng; vec3 n; vec2 uv; int mat; vec3 dpdu; vec3 dpdv; };

void onb(vec3 n, out vec3 t, out vec3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = vec3(c, s + n.y * n.y * a, -n.y);
}

Hit fetch_hit(int tri, vec2 bc, vec3 o, vec3 d, float t) {
    Hit h;
    int k = tri * 3;
    vec3 e1 = tri_pos[k + 1].xyz;
    vec3 e2 = tri_pos[k + 2].xyz;
    vec4 a0 = tri_attr[tri * 4];
    vec4 a1 = tri_attr[tri * 4 + 1];
    vec4 a2 = tri_attr[tri * 4 + 2];
    vec4 a3 = tri_attr[tri * 4 + 3];
    float w = 1.0 - bc.x - bc.y;
    h.p = o + d * t;
    h.ng = normalize(cross(e1, e2));
    vec3 n = a0.xyz * w + a1.xyz * bc.x + a2.xyz * bc.y;
    h.n = dot(n, n) > 1e-12 ? normalize(n) : h.ng;
    vec2 uv0 = vec2(a0.w, a1.w);
    vec2 uv1 = vec2(a2.w, a3.x);
    vec2 uv2 = a3.yz;
    h.uv = uv0 * w + uv1 * bc.x + uv2 * bc.y;
    h.mat = int(a3.w + 0.5);
    vec2 duv1 = uv1 - uv0;
    vec2 duv2 = uv2 - uv0;
    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) > 1e-20) {
        float r = 1.0 / det;
        h.dpdu = (e1 * duv2.y - e2 * duv1.y) * r;
        h.dpdv = (e2 * duv1.x - e1 * duv2.x) * r;
    } else {
        onb(h.n, h.dpdu, h.dpdv);
    }
    return h;
}

void material_at(inout Hit h, out vec3 base, out float metal, out float rough) {
    Mat m = mats[h.mat];
    base = m.base.rgb;
    metal = m.params.x;
    rough = m.params.y;
    if (m.tex.x >= 0)
        base = srgb_to_linear(textureLod(u_base_tex, vec3(h.uv, float(m.tex.x)), 0.0).rgb);
    if (m.tex.y >= 0) {
        vec2 mr = textureLod(u_mr_tex, vec3(h.uv, float(m.tex.y)), 0.0).rg;
        metal = mr.x;
        rough = mr.y;
    }
    if (m.tex.w >= 0) {
        // 法线贴图：和烘焙器同一个切线空间约定（T 是 dP/du 去掉法线分量，B = N×T 和 dP/dv 同侧）
        vec3 tn = textureLod(u_nmap_tex, vec3(h.uv, float(m.tex.w)), 0.0).xyz * 2.0 - 1.0;
        vec3 n = h.n;
        vec3 t = h.dpdu - n * dot(n, h.dpdu);
        float lt = length(t);
        if (lt > 1e-20) {
            t /= lt;
            vec3 b = cross(n, t);
            if (dot(b, h.dpdv) < 0.0) b = -b;
            h.n = normalize(tn.x * t + tn.y * b + tn.z * n);
        }
    }
    if (m.tex.z >= 0 && m.params.z != 0.0) {
        float layer = float(m.tex.z);
        float h0 = textureLod(u_height_tex, vec3(h.uv, layer), 0.0).r;
        float hu = textureLod(u_height_tex, vec3(h.uv + vec2(u_texel, 0.0), layer), 0.0).r;
        float hv = textureLod(u_height_tex, vec3(h.uv + vec2(0.0, u_texel), layer), 0.0).r;
        float dhdu = (hu - h0) / u_texel * m.params.z;
        float dhdv = (hv - h0) / u_texel * m.params.z;
        vec3 n = h.n;
        vec3 tu = h.dpdu - n * dot(n, h.dpdu);
        vec3 tv = h.dpdv - n * dot(n, h.dpdv);
        vec3 nb = cross(tu + n * dhdu, tv + n * dhdv);
        if (dot(nb, nb) > 1e-20) {
            nb = normalize(nb);
            h.n = dot(nb, n) < 0.0 ? -nb : nb;
        }
    }
    metal = clamp(metal, 0.0, 1.0);
    rough = clamp(rough, 0.0, 1.0);
}

// ---------------------------------------------------------------- 材质：漫反射 + GGX
float d_ggx(float nh, float a2) {
    float d = nh * nh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
float g1_smith(float nx, float a2) {
    return 2.0 * nx / (nx + sqrt(a2 + (1.0 - a2) * nx * nx));
}
vec3 sample_vndf(vec3 v, float a, float r1, float r2) {
    vec3 vh = normalize(vec3(a * v.x, a * v.y, v.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    vec3 t1 = lensq > 0.0 ? vec3(-vh.y, vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 t2 = cross(vh, t1);
    float r = sqrt(r1);
    float phi = TWO_PI * r2;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * sqrt(max(0.0, 1.0 - p1 * p1)) + s * p2;
    vec3 nh = p1 * t1 + p2 * t2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(vec3(a * nh.x, a * nh.y, max(0.0, nh.z)));
}
float spec_prob(vec3 base, float metal, float nv) {
    if (u_debug == 1) return 0.0;
    vec3 f0 = mix(vec3(0.04), base, metal);
    float sw = pow(clamp(1.0 - nv, 0.0, 1.0), 5.0);
    float ws = lum(mix(f0, vec3(1.0), sw));
    float wd = lum(base) * (1.0 - metal);
    return clamp(ws / max(ws + wd, 1e-6), 0.0, 1.0);
}
// 返回 f·cosθ 和合成的概率密度（立体角）
vec3 bsdf_eval(vec3 base, float metal, float rough, vec3 n, vec3 v, vec3 l, out float pdf) {
    pdf = 0.0;
    float nl = dot(n, l);
    float nv = dot(n, v);
    if (nl <= 0.0 || nv <= 0.0) return vec3(0.0);
    if (u_debug == 1) {
        pdf = nl * INV_PI;
        return base * INV_PI * nl;
    }
    vec3 hv = normalize(v + l);
    float nh = max(dot(n, hv), 0.0);
    float vh = max(dot(v, hv), 0.0);
    float a = max(rough * rough, 0.0015);
    float a2 = a * a;
    vec3 f0 = mix(vec3(0.04), base, metal);
    vec3 fr = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    float d = d_ggx(nh, a2);
    float gv = g1_smith(nv, a2);
    float gl = g1_smith(nl, a2);
    vec3 spec = fr * (d * gv * gl / (4.0 * nv * nl));
    vec3 diff = (1.0 - metal) * (vec3(1.0) - fr) * base * INV_PI;
    float ps = spec_prob(base, metal, nv);
    pdf = ps * (gv * d / (4.0 * nv)) + (1.0 - ps) * (nl * INV_PI);
    return (diff + spec) * nl;
}
vec3 bsdf_sample(vec3 base, float metal, float rough, vec3 n, vec3 v, out vec3 l, out float pdf) {
    float nv = dot(n, v);
    vec3 t, b;
    onb(n, t, b);
    if (rnd() < spec_prob(base, metal, nv)) {
        float a = max(rough * rough, 0.0015);
        vec3 vl = vec3(dot(v, t), dot(v, b), dot(v, n));
        vec3 hl = sample_vndf(vl, a, rnd(), rnd());
        vec3 hw = t * hl.x + b * hl.y + n * hl.z;
        l = reflect(-v, hw);
    } else {
        float r1 = rnd();
        float r2 = rnd();
        float r = sqrt(r1);
        float phi = TWO_PI * r2;
        l = t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - r1));
    }
    return bsdf_eval(base, metal, rough, n, v, l, pdf);
}

// ---------------------------------------------------------------- 环境光
vec3 env_local(vec3 d) {
    float c = cos(u_env_rotation);
    float s = sin(u_env_rotation);
    return vec3(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}
vec3 env_world(vec3 l) {
    float c = cos(u_env_rotation);
    float s = sin(u_env_rotation);
    return vec3(c * l.x + s * l.z, l.y, -s * l.x + c * l.z);
}
vec2 dir_to_uv(vec3 d) {
    return vec2(0.5 + atan(d.z, d.x) * (0.5 / PI), 0.5 + asin(clamp(d.y, -1.0, 1.0)) * (1.0 / PI));
}
vec3 uv_to_dir(vec2 uv) {
    float phi = (uv.x - 0.5) * TWO_PI;
    float th = (uv.y - 0.5) * PI;
    return vec3(cos(th) * cos(phi), sin(th), cos(th) * sin(phi));
}
float row_weight(int y) {
    float v = 1.0 - (float(y) + 0.5) / float(u_env_size.y);
    return cos((v - 0.5) * PI);
}
vec3 env_eval(vec3 d) {
    if (u_has_env == 0) return u_env_color * u_env_strength;
    vec2 uv = dir_to_uv(env_local(d));
    return textureLod(u_env, vec2(uv.x, 1.0 - uv.y), 0.0).rgb * u_env_strength;
}
float env_pdf(vec3 d) {
    if (u_has_env == 0) return 1.0 / (4.0 * PI);
    vec2 uv = dir_to_uv(env_local(d));
    int w = u_env_size.x;
    int hgt = u_env_size.y;
    int x = clamp(int(uv.x * float(w)), 0, w - 1);
    int y = clamp(int((1.0 - uv.y) * float(hgt)), 0, hgt - 1);
    float p = lum(texelFetch(u_env, ivec2(x, y), 0).rgb) * row_weight(y) / max(u_env_sum, 1e-20);
    float ce = max(cos((uv.y - 0.5) * PI), 1e-6);
    return p * float(w * hgt) / (2.0 * PI * PI * ce);
}
vec3 env_sample(out vec3 dir, out float pdf) {
    if (u_has_env == 0) {
        float z = 1.0 - 2.0 * rnd();
        float r = sqrt(max(0.0, 1.0 - z * z));
        float phi = TWO_PI * rnd();
        dir = vec3(r * cos(phi), z, r * sin(phi));
        pdf = 1.0 / (4.0 * PI);
        return u_env_color * u_env_strength;
    }
    int w = u_env_size.x;
    int hgt = u_env_size.y;
    float r1 = rnd();
    int lo = 0;
    int hi = hgt - 1;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (r1 < env_cdf[mid]) hi = mid; else lo = mid + 1; }
    int y = lo;
    float r2 = rnd();
    int base = hgt + y * w;
    lo = 0;
    hi = w - 1;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (r2 < env_cdf[base + mid]) hi = mid; else lo = mid + 1; }
    int x = lo;
    vec2 uv = vec2((float(x) + rnd()) / float(w), 1.0 - (float(y) + rnd()) / float(hgt));
    dir = env_world(uv_to_dir(uv));
    vec3 c = texelFetch(u_env, ivec2(x, y), 0).rgb;
    float p = lum(c) * row_weight(y) / max(u_env_sum, 1e-20);
    float ce = max(cos((uv.y - 0.5) * PI), 1e-6);
    pdf = p * float(w * hgt) / (2.0 * PI * PI * ce);
    return c * u_env_strength;
}

// ---------------------------------------------------------------- 灯光（只做直接光采样：灯不在场景几何里，反射光线打不到它们）
// 返回到灯上采样点的方向 l、距离 dist，和这个方向来的「光量」（已含距离衰减、面光的立体角换算）
vec3 light_sample(int i, vec3 p, out vec3 l, out float dist) {
    Light lt = lights[i];
    int kind = int(lt.a.w + 0.5);
    vec3 radiance = lt.c.rgb;
    if (kind == 1) {                                      // 日光：在角直径的锥里取方向
        vec3 axis = -lt.b.xyz;
        float half_angle = lt.b.w;
        vec3 t;
        vec3 bt;
        onb(axis, t, bt);
        float ct = 1.0 - rnd() * (1.0 - cos(half_angle));
        float st = sqrt(max(0.0, 1.0 - ct * ct));
        float phi = TWO_PI * rnd();
        l = normalize(axis * ct + (t * cos(phi) + bt * sin(phi)) * st);
        dist = BIG;
        return radiance;
    }
    if (kind == 3) {                                      // 面光：在矩形上取点，单面朝照射方向发光
        vec3 q = lt.a.xyz + lt.e.xyz * ((rnd() - 0.5) * lt.d.z) + lt.f.xyz * ((rnd() - 0.5) * lt.d.w);
        vec3 to = q - p;
        float d2 = max(dot(to, to), 1e-12);
        dist = sqrt(d2);
        l = to / dist;
        float cos_l = dot(-l, lt.b.xyz);
        if (cos_l <= 0.0) return vec3(0.0);
        float area = lt.d.z * lt.d.w;
        return radiance * cos_l * area / d2;
    }
    // 点光、聚光：有半径时在朝向着色点的圆盘上取点（软阴影）
    vec3 center = lt.a.xyz;
    float radius = lt.b.w;
    vec3 q = center;
    if (radius > 0.0) {
        vec3 axis = normalize(p - center);
        vec3 t;
        vec3 bt;
        onb(axis, t, bt);
        float r = radius * sqrt(rnd());
        float phi = TWO_PI * rnd();
        q = center + (t * cos(phi) + bt * sin(phi)) * r;
    }
    vec3 to = q - p;
    float d2 = max(dot(to, to), 1e-12);
    dist = sqrt(d2);
    l = to / dist;
    vec3 li = radiance / d2;
    if (kind == 2) {                                      // 聚光：锥外没有光，边缘按柔和宽度过渡
        float c = dot(-l, lt.b.xyz);
        float cos_half = lt.d.x;
        float smooth_w = lt.d.y;
        float mask = smooth_w > 0.0 ? smoothstep(0.0, 1.0, (c - cos_half) / smooth_w) : (c > cos_half ? 1.0 : 0.0);
        li *= mask;
    }
    return li;
}

float power_heuristic(float a, float b) {
    float a2 = a * a;
    float b2 = b * b;
    return a2 + b2 > 0.0 ? a2 / (a2 + b2) : 0.0;
}

vec3 clamp_light(vec3 c, int depth) {
    if (depth == 0 || u_clamp <= 0.0) return c;
    float l = lum(c);
    return l > u_clamp ? c * (u_clamp / l) : c;
}

// ---------------------------------------------------------------- 主循环
void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy) + u_offset;
    if (pix.x >= u_size.x || pix.y >= u_size.y) return;
    g_rng = pcg(uint(pix.y * u_size.x + pix.x) ^ pcg(uint(u_sample) * 9781u + u_seed * 6271u));
    vec2 jitter = u_debug == 2 ? vec2(0.5) : vec2(rnd(), rnd());
    float nx = ((float(pix.x) + jitter.x) / float(u_size.x)) * 2.0 - 1.0;
    float ny = 1.0 - ((float(pix.y) + jitter.y) / float(u_size.y)) * 2.0;
    vec3 o;
    vec3 d;
    if (u_ortho == 1) {
        o = u_eye + u_right * (nx * u_ortho_half_h * u_aspect + u_shift.x) + u_up * (ny * u_ortho_half_h + u_shift.y);
        d = u_forward;
    } else {
        o = u_eye;
        d = normalize(u_forward + u_right * (nx * u_tan_half_y * u_aspect + u_shift.x)
                      + u_up * (ny * u_tan_half_y + u_shift.y));
    }
    if (u_debug == 2) {
        int dtri;
        vec2 dbc;
        float dt;
        bool hit = trace(o, d, BIG, false, dtri, dbc, dt);
        imageStore(u_accum, pix, hit ? vec4(dt, float(dtri), dbc) : vec4(-1.0));
        return;
    }
    vec3 radiance = vec3(0.0);
    vec3 throughput = vec3(1.0);
    float alpha = 1.0;
    float last_pdf = 0.0;
    vec3 aov_albedo = vec3(0.0);
    vec3 aov_normal = vec3(0.0);
    for (int depth = 0; depth <= u_max_bounces; depth++) {
        int tri;
        vec2 bc;
        float t;
        if (!trace(o, d, BIG, false, tri, bc, t)) {
            if (depth == 0) {
                if (u_transparent == 1) alpha = 0.0;
                else if (u_background == 0) radiance += u_background_color;
                else radiance += env_eval(d);
                aov_albedo = clamp(radiance, 0.0, 1.0);
                break;
            }
            float w = power_heuristic(last_pdf, env_pdf(d));
            radiance += clamp_light(throughput * env_eval(d) * w, depth);
            break;
        }
        Hit h = fetch_hit(tri, bc, o, d, t);
        vec3 v = -d;
        if (dot(h.ng, v) < 0.0) { h.ng = -h.ng; h.n = -h.n; }     // 双面
        vec3 shading_n = h.n;
        vec3 base;
        float metal;
        float rough;
        material_at(h, base, metal, rough);
        if (dot(h.n, v) <= 0.0) h.n = dot(shading_n, v) > 0.0 ? shading_n : h.ng;
        if (depth == 0) {
            aov_albedo = base;
            aov_normal = h.n;
        }
        vec3 spawn = h.p + h.ng * u_eps;
        // 直接光：对环境重要性采样
        {
            vec3 l;
            float pdf_l;
            vec3 le = env_sample(l, pdf_l);
            if (pdf_l > 0.0 && dot(h.ng, l) > 0.0) {
                float pdf_b;
                vec3 f = bsdf_eval(base, metal, rough, h.n, v, l, pdf_b);
                if (pdf_b > 0.0 && max(f.x, max(f.y, f.z)) > 0.0) {
                    int st;
                    vec2 sb;
                    float sh;
                    if (!trace(spawn, l, BIG, true, st, sb, sh)) {
                        float w = power_heuristic(pdf_l, pdf_b);
                        radiance += clamp_light(throughput * f * le * (w / pdf_l), depth);
                    }
                }
            }
        }
        // 直接光：场景里的灯
        for (int li = 0; li < u_light_count; li++) {
            vec3 l;
            float dist;
            vec3 le = light_sample(li, h.p, l, dist);
            if (max(le.x, max(le.y, le.z)) <= 0.0 || dot(h.ng, l) <= 0.0) continue;
            float pdf_b;
            vec3 f = bsdf_eval(base, metal, rough, h.n, v, l, pdf_b);
            if (max(f.x, max(f.y, f.z)) <= 0.0) continue;
            if (lights[li].c.w > 0.5) {
                int st;
                vec2 sb;
                float sh;
                if (trace(spawn, l, dist * (1.0 - 1e-4) - u_eps, true, st, sb, sh)) continue;
            }
            radiance += clamp_light(throughput * f * le, depth);
        }
        if (depth == u_max_bounces) break;
        vec3 l;
        float pdf;
        vec3 f = bsdf_sample(base, metal, rough, h.n, v, l, pdf);
        if (pdf <= 0.0 || dot(l, h.ng) <= 0.0) break;
        throughput *= f / pdf;
        last_pdf = pdf;
        o = spawn;
        d = l;
        if (depth >= 3) {
            float q = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001, 0.95);
            if (rnd() > q) break;
            throughput /= q;
        }
    }
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = vec3(0.0);
    vec4 prev = imageLoad(u_accum, pix);
    imageStore(u_accum, pix, prev + vec4(radiance, alpha));
    if (u_aov == 1) {
        imageStore(u_aov_albedo, pix, imageLoad(u_aov_albedo, pix) + vec4(aov_albedo, 1.0));
        imageStore(u_aov_normal, pix, imageLoad(u_aov_normal, pix) + vec4(aov_normal, 1.0));
    }
}
"""

RESOLVE = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) readonly uniform image2D u_accum;
layout(rgba8, binding = 1) writeonly uniform image2D u_out;
uniform ivec2 u_size;
uniform float u_inv_spp;
uniform int u_transform;
uniform float u_exposure;
""" + GLSL_TONEMAP + """
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= u_size.x || p.y >= u_size.y) return;
    vec4 s = imageLoad(u_accum, p) * u_inv_spp;
    vec3 c = tonemap(max(s.rgb, vec3(0.0)), u_transform, u_exposure);
    imageStore(u_out, ivec2(p.x, u_size.y - 1 - p.y), vec4(c, clamp(s.a, 0.0, 1.0)));
}
"""

CLEAR = """#version 430
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba32f, binding = 0) writeonly uniform image2D u_accum;
layout(rgba32f, binding = 2) writeonly uniform image2D u_aov_albedo;
layout(rgba32f, binding = 3) writeonly uniform image2D u_aov_normal;
uniform ivec2 u_size;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= u_size.x || p.y >= u_size.y) return;
    imageStore(u_accum, p, vec4(0.0));
    imageStore(u_aov_albedo, p, vec4(0.0));
    imageStore(u_aov_normal, p, vec4(0.0));
}
"""
