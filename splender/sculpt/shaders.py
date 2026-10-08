"""雕刻引擎的着色器。

笔刷的形变公式（衰减曲线 3d⁴−4d³+1、标准、黏土、压平、捏合、折痕、膨胀）参考自 SculptGL
（https://github.com/stephomi/sculptgl ，MIT 许可，Copyright (c) 2019 Stéphane Ginier），改写成显卡计算着色器：
网格按空间顺序每 256 个顶点一块，先挑出被笔触碰到的块，再对这些块执行笔刷、重算法线和包围盒。
"""

CLUSTER = 256

#: 笔刷种类编号（和 engine.BRUSHES 的顺序一致）
DRAW, CLAY, CLAY_STRIPS, SMOOTH, GRAB, INFLATE, PINCH, FLATTEN, CREASE, MASK = range(10)

_COMMON = """#version 430
#extension GL_NV_shader_atomic_float : enable
layout(std430, binding = 0) buffer Pos { vec4 pos[]; };              // xyz 位置，w 遮罩（1 = 冻结）
layout(std430, binding = 1) buffer Nrm { vec4 nrm[]; };
layout(std430, binding = 2) readonly buffer Tris { int tris[]; };
layout(std430, binding = 3) readonly buffer VfOff { int vf_off[]; };
layout(std430, binding = 4) readonly buffer VfIdx { int vf_idx[]; };
layout(std430, binding = 5) buffer Bounds { vec4 bounds[]; };        // 每块：中心 xyz、半径
layout(std430, binding = 6) buffer Work { int work_ids[]; };
layout(std430, binding = 7) buffer Meta { uint work_count; uint arena_count; uint arena_capacity; uint touched_count;
                                         uint indirect[4]; int touched[]; };
struct Dab { vec4 center_radius; vec4 normal_strength; vec4 params; };   // params：硬度、方向符号、保留、保留
layout(std430, binding = 8) readonly buffer Dabs { Dab dabs[]; };
layout(std430, binding = 9) buffer Cow { int cow[]; };               // 每块：本笔备份在 arena 的位置 + 1，0 表示还没备份
layout(std430, binding = 10) buffer Fresh { int fresh[]; };          // 每块：这一帧刚分到备份位置，要先拷贝原值
layout(std430, binding = 11) buffer Arena { vec4 arena[]; };
layout(std430, binding = 12) buffer Plane { float plane_acc[]; };    // 每个笔触 8 个：法线和、中心和（相对）、权重
layout(std430, binding = 13) buffer Scratch { vec4 scratch[]; };
layout(std430, binding = 14) buffer GrabW { float grab_w[]; };
uniform int u_vertex_count;
uniform int u_cluster_count;
uniform int u_dab_count;
uniform int u_type;
uniform int u_all;          // 1：处理全部块（载入、撤销后），0：只处理工作列表里的块

int cluster_of_group() { return u_all == 1 ? int(gl_WorkGroupID.x) : work_ids[gl_WorkGroupID.x]; }

float falloff(float d, float hard) {
    if (d >= 1.0) return 0.0;
    float x = hard < 0.999 ? clamp((d - hard) / (1.0 - hard), 0.0, 1.0) : 0.0;
    float x2 = x * x;
    return 3.0 * x2 * x2 - 4.0 * x2 * x + 1.0;
}
"""

CULL = _COMMON + """
layout(local_size_x = 64) in;
uniform float u_margin;
uniform int u_backup;
void main() {
    int c = int(gl_GlobalInvocationID.x);
    if (c >= u_cluster_count) return;
    vec4 b = bounds[c];
    bool hit = false;
    for (int i = 0; i < u_dab_count; i++) {
        vec4 cr = dabs[i].center_radius;
        if (distance(b.xyz, cr.xyz) <= b.w + cr.w + u_margin) { hit = true; break; }
    }
    if (!hit) return;
    uint slot = atomicAdd(work_count, 1u);
    work_ids[slot] = c;
    if (u_backup == 1 && cow[c] == 0) {
        uint a = atomicAdd(arena_count, 1u);
        if (a < arena_capacity) {
            cow[c] = int(a) + 1;
            fresh[c] = 1;
            uint t = atomicAdd(touched_count, 1u);
            touched[t] = c;
        }
    }
}
"""

ARGS = _COMMON + """
layout(local_size_x = 1) in;
void main() { indirect[0] = work_count; indirect[1] = 1u; indirect[2] = 1u; }
"""

PLANE = _COMMON + """
layout(local_size_x = 256) in;
shared vec4 s_n[256];
shared vec4 s_c[256];
void main() {
    int c = cluster_of_group();
    int local = int(gl_LocalInvocationID.x);
    int v = c * 256 + local;
    bool valid = v < u_vertex_count;
    vec3 p = valid ? pos[v].xyz : vec3(0.0);
    vec3 n = valid ? nrm[v].xyz : vec3(0.0);
    for (int i = 0; i < u_dab_count; i++) {
        vec4 cr = dabs[i].center_radius;
        vec3 dn = dabs[i].normal_strength.xyz;
        float w = 0.0;
        if (valid && dot(n, dn) > 0.0) w = falloff(distance(p, cr.xyz) / cr.w, 0.0);
        s_n[local] = vec4(n * w, w);
        s_c[local] = vec4((p - cr.xyz) / cr.w * w, w > 0.0 ? 1.0 : 0.0);
        barrier();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (local < stride) { s_n[local] += s_n[local + stride]; s_c[local] += s_c[local + stride]; }
            barrier();
        }
        if (local == 0 && s_n[0].w > 0.0) {
            atomicAdd(plane_acc[i * 8 + 0], s_n[0].x);
            atomicAdd(plane_acc[i * 8 + 1], s_n[0].y);
            atomicAdd(plane_acc[i * 8 + 2], s_n[0].z);
            atomicAdd(plane_acc[i * 8 + 3], s_n[0].w);
            atomicAdd(plane_acc[i * 8 + 4], s_c[0].x);
            atomicAdd(plane_acc[i * 8 + 5], s_c[0].y);
            atomicAdd(plane_acc[i * 8 + 6], s_c[0].z);
            atomicAdd(plane_acc[i * 8 + 7], s_c[0].w);
        }
        barrier();
    }
}
"""

_PLANE_OF = """
void plane_of(int i, out vec3 an, out vec3 ac) {
    vec4 cr = dabs[i].center_radius;
    float w = plane_acc[i * 8 + 3];
    vec3 ns = vec3(plane_acc[i * 8 + 0], plane_acc[i * 8 + 1], plane_acc[i * 8 + 2]);
    if (w > 1e-6 && dot(ns, ns) > 1e-12) {
        an = normalize(ns);
        ac = cr.xyz + vec3(plane_acc[i * 8 + 4], plane_acc[i * 8 + 5], plane_acc[i * 8 + 6]) / w * cr.w;
    } else {
        an = normalize(dabs[i].normal_strength.xyz);
        ac = cr.xyz;
    }
}
"""

APPLY = _COMMON + _PLANE_OF + """
layout(local_size_x = 256) in;
void main() {
    int c = cluster_of_group();
    int local = int(gl_LocalInvocationID.x);
    int v = c * 256 + local;
    if (v >= u_vertex_count) return;
    vec4 P = pos[v];
    if (fresh[c] == 1) arena[(cow[c] - 1) * 256 + local] = P;
    vec3 p = P.xyz;
    float mask = P.w;
    vec3 n = nrm[v].xyz;
    for (int i = 0; i < u_dab_count; i++) {
        vec4 cr = dabs[i].center_radius;
        float r = cr.w;
        float dist = distance(p, cr.xyz) / r;
        if (dist >= 1.0) continue;
        float f = falloff(dist, dabs[i].params.x) * dabs[i].normal_strength.w;
        float sgn = dabs[i].params.y;
        if (u_type == """ + str(MASK) + """) { mask = clamp(mask + f * sgn, 0.0, 1.0); continue; }
        f *= 1.0 - mask;
        if (f <= 0.0) continue;
        vec3 an;
        vec3 ac;
        plane_of(i, an, ac);
        if (u_type == """ + str(DRAW) + """) {
            p += an * (f * r * 0.1 * sgn);
        } else if (u_type == """ + str(INFLATE) + """) {
            p += normalize(n) * (f * r * 0.1 * sgn);
        } else if (u_type == """ + str(CLAY) + """ || u_type == """ + str(CLAY_STRIPS) + """) {
            vec3 pp = ac + an * (r * 0.1 * sgn);
            float dp = dot(p - pp, an);
            if (dp * sgn < 0.0) p -= an * (dp * f);
        } else if (u_type == """ + str(FLATTEN) + """) {
            float dp = dot(p - ac, an);
            p -= an * (dp * f * (sgn > 0.0 ? 1.0 : -1.0));
        } else if (u_type == """ + str(PINCH) + """) {
            vec3 to = cr.xyz - p;
            to -= an * dot(to, an);
            p += to * (f * 0.1 * sgn);
        } else if (u_type == """ + str(CREASE) + """) {
            vec3 to = cr.xyz - p;
            p += to * (f * 0.07) - an * (pow(f, 5.0) * r * 0.07 * sgn);
        }
    }
    pos[v] = vec4(p, mask);
}
"""

SMOOTH_A = _COMMON + """
layout(local_size_x = 256) in;
void main() {
    int c = cluster_of_group();
    int local = int(gl_LocalInvocationID.x);
    int v = c * 256 + local;
    if (v >= u_vertex_count) return;
    vec4 P = pos[v];
    if (fresh[c] == 1) arena[(cow[c] - 1) * 256 + local] = P;
    float amount = 0.0;
    for (int i = 0; i < u_dab_count; i++) {
        vec4 cr = dabs[i].center_radius;
        float f = falloff(distance(P.xyz, cr.xyz) / cr.w, dabs[i].params.x) * dabs[i].normal_strength.w;
        amount = 1.0 - (1.0 - amount) * (1.0 - clamp(f, 0.0, 1.0));
    }
    amount *= 1.0 - P.w;
    vec3 result = P.xyz;
    if (amount > 0.0) {
        vec3 sum = vec3(0.0);
        float count = 0.0;
        for (int k = vf_off[v]; k < vf_off[v + 1]; k++) {
            int t = vf_idx[k] * 3;
            sum += pos[tris[t]].xyz + pos[tris[t + 1]].xyz + pos[tris[t + 2]].xyz;
            count += 3.0;
        }
        if (count > 0.0) result = mix(P.xyz, sum / count, amount);
    }
    scratch[v] = vec4(result, P.w);
}
"""

SMOOTH_B = _COMMON + """
layout(local_size_x = 256) in;
void main() {
    int c = cluster_of_group();
    int v = c * 256 + int(gl_LocalInvocationID.x);
    if (v >= u_vertex_count) return;
    pos[v] = scratch[v];
}
"""

GRAB_INIT = _COMMON + """
layout(local_size_x = 256) in;
void main() {
    int c = cluster_of_group();
    int local = int(gl_LocalInvocationID.x);
    int v = c * 256 + local;
    if (v >= u_vertex_count) return;
    vec4 P = pos[v];
    if (fresh[c] == 1) arena[(cow[c] - 1) * 256 + local] = P;
    vec4 cr = dabs[0].center_radius;
    float f = falloff(distance(P.xyz, cr.xyz) / cr.w, dabs[0].params.x) * dabs[0].normal_strength.w;
    grab_w[v] = f * (1.0 - P.w);
}
"""

GRAB = _COMMON + """
layout(local_size_x = 256) in;
uniform vec3 u_delta;
void main() {
    int c = cluster_of_group();
    int local = int(gl_LocalInvocationID.x);
    int v = c * 256 + local;
    if (v >= u_vertex_count || cow[c] == 0) return;
    vec4 orig = arena[(cow[c] - 1) * 256 + local];
    pos[v] = vec4(orig.xyz + u_delta * grab_w[v], orig.w);
}
"""

NORMALS = _COMMON + """
layout(local_size_x = 256) in;
void main() {
    int c = cluster_of_group();
    int v = c * 256 + int(gl_LocalInvocationID.x);
    if (v >= u_vertex_count) return;
    vec3 acc = vec3(0.0);
    for (int k = vf_off[v]; k < vf_off[v + 1]; k++) {
        int t = vf_idx[k] * 3;
        vec3 a = pos[tris[t]].xyz;
        acc += cross(pos[tris[t + 1]].xyz - a, pos[tris[t + 2]].xyz - a);
    }
    float len2 = dot(acc, acc);
    nrm[v] = vec4(len2 > 1e-30 ? acc * inversesqrt(len2) : vec3(0.0, 1.0, 0.0), 0.0);
}
"""

BOUNDS = _COMMON + """
layout(local_size_x = 256) in;
shared vec3 s_min[256];
shared vec3 s_max[256];
void main() {
    int c = cluster_of_group();
    int local = int(gl_LocalInvocationID.x);
    int v = c * 256 + local;
    vec3 p = v < u_vertex_count ? pos[v].xyz : pos[min(c * 256, u_vertex_count - 1)].xyz;
    s_min[local] = p;
    s_max[local] = p;
    barrier();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (local < stride) {
            s_min[local] = min(s_min[local], s_min[local + stride]);
            s_max[local] = max(s_max[local], s_max[local + stride]);
        }
        barrier();
    }
    if (local == 0) {
        vec3 center = (s_min[0] + s_max[0]) * 0.5;
        bounds[c] = vec4(center, length(s_max[0] - s_min[0]) * 0.5);
        fresh[c] = 0;
    }
}
"""

GATHER = _COMMON + """
layout(local_size_x = 256) in;
void main() {
    int c = touched[gl_WorkGroupID.x];
    int v = c * 256 + int(gl_LocalInvocationID.x);
    if (v >= u_vertex_count) return;
    scratch[(cow[c] - 1) * 256 + int(gl_LocalInvocationID.x)] = pos[v];
}
"""

SCATTER = _COMMON + """
layout(local_size_x = 256) in;
void main() {
    int c = work_ids[gl_WorkGroupID.x];
    int v = c * 256 + int(gl_LocalInvocationID.x);
    if (v >= u_vertex_count) return;
    pos[v] = scratch[int(gl_WorkGroupID.x) * 256 + int(gl_LocalInvocationID.x)];
}
"""

RENDER_VERT = """#version 430
in vec4 in_pos;
in vec4 in_nrm;
uniform mat4 u_view_proj;
out vec3 v_pos;
out vec3 v_nrm;
out float v_mask;
void main() {
    v_pos = in_pos.xyz;
    v_nrm = in_nrm.xyz;
    v_mask = in_pos.w;
    gl_Position = u_view_proj * vec4(in_pos.xyz, 1.0);
}
"""

RENDER_FRAG_HEAD = """#version 430
in vec3 v_pos;
in vec3 v_nrm;
in float v_mask;
out vec4 o_color;
uniform vec3 u_eye;
uniform mat4 u_view;
uniform int u_solid;          // 0 工作室光，1 Matcap，2 平涂
uniform vec3 u_tint;
uniform float u_mask_dim;
uniform vec4 u_cursor;
uniform float u_cursor_hardness;
uniform int u_cursor_on;
uniform float u_pixel_world;
uniform int u_wire;
"""

RENDER_FRAG_BODY = """
vec3 to_display(vec3 c){ return mix(1.055 * pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.4)) - 0.055, c * 12.92, lessThanEqual(c, vec3(0.0031308))); }
void main() {
    vec3 n = normalize(v_nrm);
    if (!gl_FrontFacing) n = -n;
    vec3 v = normalize(u_eye - v_pos);
    vec3 nv = normalize(mat3(u_view) * n);
    vec3 vv = normalize(mat3(u_view) * v);
    vec3 color;
    if (u_solid == 0) color = to_display(shade_studio(nv, vv, u_tint));
    else if (u_solid == 1) color = to_display(shade_matcap(nv, vv, u_tint));
    else color = to_display(u_tint);
    color *= mix(1.0, u_mask_dim, clamp(v_mask, 0.0, 1.0));
    if (u_cursor_on == 1) {
        float d = length(v_pos - u_cursor.xyz);
        float w = max(fwidth(d), u_pixel_world * 0.5) * 1.3;
        float ring = 1.0 - smoothstep(0.0, w, abs(d - u_cursor.w));
        float inner = (1.0 - smoothstep(0.0, w * 0.8, abs(d - u_cursor.w * u_cursor_hardness))) * 0.35;
        float halo = 1.0 - smoothstep(w, w * 2.2, abs(d - u_cursor.w));
        color = mix(color, vec3(0.0), halo * 0.35);
        color = mix(color, vec3(1.0), max(ring, inner) * 0.9);
    }
    o_color = vec4(color, 1.0);
}
"""

GBUF_FRAG = """#version 430
in vec3 v_pos;
in vec3 v_nrm;
in float v_mask;
layout(location = 0) out vec4 o_pos;
layout(location = 1) out vec4 o_nrm;
void main() {
    vec3 n = normalize(v_nrm);
    if (!gl_FrontFacing) n = -n;
    o_pos = vec4(v_pos, 1.0);
    o_nrm = vec4(n, 1.0);
}
"""
