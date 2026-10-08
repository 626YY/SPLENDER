"""模型读取与测试模型。

读入 OBJ、glTF（.gltf 配外部 .bin 或 data: URI）、GLB，统一成 MeshData：
三角形逐角点存放、不共享顶点，三角形 k 的三个角是 3k、3k+1、3k+2。
坐标按文件原样使用（Y 轴向上的右手系）。UV 原点在左下角，glTF 的 v 读入时翻成 1 - v。

OBJ 的逐字节解析用 numba 编译（cache=True）。大文件按行边界切块，多线程并行解析：
每块先数一遍各类语句的个数，算出全局偏移后再填一遍，负索引按全局计数换算。
"""
from __future__ import annotations

import base64
import json
import math
import os
import struct
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit

__all__ = ["MeshData", "MeshImportError", "load_mesh", "make_test_mesh", "DEFAULT_MATERIAL", "TEST_MESH_KINDS"]

#: 没有指定材质的三角形归到这个材质名下。
DEFAULT_MATERIAL = "默认材质"

#: OBJ 解析时每块的字节数。小于它的文件不切块。
OBJ_CHUNK_BYTES = 8 << 20


class MeshImportError(ValueError):
    """模型文件读不了。消息是给用户看的一句话。"""


@dataclass
class MeshData:
    """一个导入的模型。三角形逐角点存放，三角形 k 的三个角是 3k、3k+1、3k+2。"""

    name: str
    positions: np.ndarray     # (T*3, 3) float32，逐角点，不共享
    normals: np.ndarray       # (T*3, 3) float32，单位长度
    uvs: np.ndarray           # (T*3, 2) float32，原点在左下角
    material_ids: np.ndarray  # (T,) int32，指向 materials
    materials: list[str]
    bounds_min: np.ndarray    # (3,) float32
    bounds_max: np.ndarray    # (3,) float32
    source_path: str = ""
    has_uvs: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def triangle_count(self) -> int:
        return int(self.material_ids.shape[0])


# =====================================================================================
# 入口
# =====================================================================================

_NORMAL_MODES = ("auto", "smooth", "flat")


def load_mesh(path, *, missing_normals: str = "auto", threads: int = 0) -> MeshData:
    """按扩展名读取模型：.obj、.gltf、.glb。

    missing_normals：文件缺法线时怎么补。
        "auto"   OBJ 按平滑补，glTF 按平直补（glTF 规范的要求）；
        "smooth" 按位置焊接顶点，面法线按顶点处的夹角加权平均；
        "flat"   直接用面法线。
    threads：OBJ 解析用的线程数，0 表示自动。

    读不了时抛 MeshImportError，消息可以直接给用户看。
    """
    if missing_normals not in _NORMAL_MODES:
        raise ValueError(f"missing_normals 只能是 {_NORMAL_MODES} 之一")
    p = Path(path)
    if not p.is_file():
        raise MeshImportError(f"找不到文件：{p}")
    ext = p.suffix.lower()
    if ext == ".obj":
        mode = "smooth" if missing_normals == "auto" else missing_normals
        return _load_obj(p, mode, threads=threads)
    if ext in (".gltf", ".glb"):
        mode = "flat" if missing_normals == "auto" else missing_normals
        return _load_gltf(p, mode)
    raise MeshImportError(f"不支持的模型格式：{p.suffix or '没有扩展名'}。可以读 .obj、.gltf、.glb")


def _workers(threads: int) -> int:
    if threads and threads > 0:
        return int(threads)
    return max(1, min(8, os.cpu_count() or 1))


def _run_jobs(jobs, workers: int):
    """依次或用线程池执行一组无参函数，按顺序返回结果。numba 函数带 nogil，线程能真正并行。"""
    if workers <= 1 or len(jobs) <= 1:
        return [job() for job in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(lambda job: job(), jobs))


# =====================================================================================
# OBJ：numba 逐字节解析
# =====================================================================================

# 语句种类
_K_OTHER = 0
_K_V = 1
_K_VT = 2
_K_VN = 3
_K_F = 4
_K_USEMTL = 5

# 面角点索引的特殊值
_IDX_MISSING = -1   # 没写（比如 v//vn 里的 vt）
_IDX_INVALID = -2   # 写了但无效（0、解析不了、负索引越过文件开头）

_POW10_MIN = -330
_POW10 = np.array([float(f"1e{k}") for k in range(_POW10_MIN, 309)], dtype=np.float64)


@njit(cache=True, nogil=True)
def _is_stop(data, i, end):
    """i 处是不是一个词的结尾：空白、换行、注释符，或者续行用的「反斜杠 + 换行」。"""
    ch = data[i]
    if ch == 32 or ch == 9 or ch == 10 or ch == 13 or ch == 35:
        return True
    if ch == 92:
        j = i + 1
        if j < end and data[j] == 13:
            j += 1
        return j >= end or data[j] == 10
    return False


@njit(cache=True, nogil=True)
def _skip_blank(data, i, end):
    """跳过空格、制表符和续行（反斜杠紧跟换行）。"""
    while i < end:
        ch = data[i]
        if ch == 32 or ch == 9:
            i += 1
        elif ch == 92:
            j = i + 1
            if j < end and data[j] == 13:
                j += 1
            if j >= end:
                i = j
            elif data[j] == 10:
                i = j + 1
            else:
                break
        else:
            break
    return i


@njit(cache=True, nogil=True)
def _next_line(data, i, end):
    """跳到下一行开头。\\r 和 \\n 都算行尾（\\r\\n 会多出一个空行，无害）。"""
    while i < end:
        ch = data[i]
        i += 1
        if ch == 10 or ch == 13:
            return i
    return i


@njit(cache=True, nogil=True)
def _line_start_after(data, i, end):
    """从 i 往后找下一行的开头，跳过续行的换行。用来给大文件切块。"""
    while i < end:
        if data[i] == 10:
            j = i - 1
            if j >= 0 and data[j] == 13:
                j -= 1
            if not (j >= 0 and data[j] == 92):
                return i + 1
        i += 1
    return end


@njit(cache=True, nogil=True)
def _keyword(data, i, end):
    """读行首关键字，返回 (种类, 关键字之后的位置)。"""
    j = i
    while j < end and not _is_stop(data, j, end):
        j += 1
    n = j - i
    kind = _K_OTHER
    c0 = data[i]
    if n == 1:
        if c0 == 118:            # v
            kind = _K_V
        elif c0 == 102:          # f
            kind = _K_F
    elif n == 2 and c0 == 118:
        c1 = data[i + 1]
        if c1 == 116:            # vt
            kind = _K_VT
        elif c1 == 110:          # vn
            kind = _K_VN
    elif (n == 6 and c0 == 117 and data[i + 1] == 115 and data[i + 2] == 101 and data[i + 3] == 109
          and data[i + 4] == 116 and data[i + 5] == 108):   # usemtl
        kind = _K_USEMTL
    return kind, j


@njit(cache=True, nogil=True)
def _skip_tokens(data, i, end):
    """跳过本条语句剩下的词，返回 (词数, 语句结尾位置)。"""
    n = 0
    while True:
        i = _skip_blank(data, i, end)
        if i >= end:
            break
        ch = data[i]
        if ch == 10 or ch == 13 or ch == 35:
            break
        n += 1
        while i < end and not _is_stop(data, i, end):
            i += 1
    return n, i


@njit(cache=True, nogil=True)
def _parse_float(data, i, end):
    """从 i 解析一个十进制浮点数，返回 (值, 结束位置, 是否读到数字)。

    尾数最多取 18 位有效数字，再乘 10 的幂。结果最终存成 float32，这个精度绰绰有余。
    """
    neg = False
    if i < end:
        ch = data[i]
        if ch == 45:
            neg = True
            i += 1
        elif ch == 43:
            i += 1
    mant = 0
    nd = 0
    exp10 = 0
    seen = False
    while i < end:
        ch = data[i]
        if 48 <= ch <= 57:
            seen = True
            if nd < 18:
                mant = mant * 10 + (ch - 48)
                if mant != 0:
                    nd += 1
            else:
                exp10 += 1
            i += 1
        else:
            break
    if i < end and data[i] == 46:
        i += 1
        while i < end:
            ch = data[i]
            if 48 <= ch <= 57:
                seen = True
                if nd < 18:
                    mant = mant * 10 + (ch - 48)
                    if mant != 0:
                        nd += 1
                    exp10 -= 1
                i += 1
            else:
                break
    if not seen:
        return np.nan, i, False
    if i < end and (data[i] == 101 or data[i] == 69):
        j = i + 1
        eneg = False
        if j < end and (data[j] == 45 or data[j] == 43):
            eneg = data[j] == 45
            j += 1
        if j < end and 48 <= data[j] <= 57:
            e = 0
            while j < end and 48 <= data[j] <= 57:
                if e < 100000:
                    e = e * 10 + (data[j] - 48)
                j += 1
            if eneg:
                exp10 -= e
            else:
                exp10 += e
            i = j
    val = float(mant)
    if mant != 0:
        if exp10 > 0:
            if exp10 > 308:
                val = np.inf
            else:
                val = val * _POW10[exp10 - _POW10_MIN]
        elif exp10 < 0:
            if exp10 >= -22:
                val = val / _POW10[-exp10 - _POW10_MIN]
            elif exp10 >= _POW10_MIN:
                val = val * _POW10[exp10 - _POW10_MIN]
            else:
                val = 0.0
    if neg:
        val = -val
    return val, i, True


@njit(cache=True, nogil=True)
def _parse_int(data, i, end):
    """解析一个可带符号的整数，返回 (值, 结束位置, 是否读到数字)。"""
    neg = False
    if i < end and (data[i] == 45 or data[i] == 43):
        neg = data[i] == 45
        i += 1
    val = 0
    nd = 0
    while i < end:
        ch = data[i]
        if 48 <= ch <= 57:
            if val < 100000000000000:
                val = val * 10 + (ch - 48)
            nd += 1
            i += 1
        else:
            break
    if neg:
        val = -val
    return val, i, nd > 0


@njit(cache=True, nogil=True)
def _resolve(idx, ok, count):
    """OBJ 索引（从 1 开始，负数从当前末尾倒数）换成从 0 开始的绝对索引。
    正索引的越界在全部读完后再查（允许引用后面才定义的顶点）。"""
    if not ok or idx == 0:
        return _IDX_INVALID
    if idx > 0:
        return idx - 1
    r = count + idx
    if r < 0:
        return _IDX_INVALID
    return r


@njit(cache=True, nogil=True)
def _read_floats(data, i, end, out, row, ncol, need):
    """把本条语句的数值读进 out[row, :ncol]。前 need 个缺了记 NaN，其余缺了记 0，多余的忽略。"""
    for d in range(ncol):
        i = _skip_blank(data, i, end)
        if i >= end or data[i] == 10 or data[i] == 13 or data[i] == 35:
            if d < need:
                out[row, d] = np.nan
            else:
                out[row, d] = 0.0
            continue
        val, j, ok = _parse_float(data, i, end)
        if ok:
            out[row, d] = val
        else:
            out[row, d] = np.nan
        while j < end and not _is_stop(data, j, end):
            j += 1
        i = j
    n, i = _skip_tokens(data, i, end)
    return i


@njit(cache=True, nogil=True)
def _read_corner(data, i, end, nv, nvt, nvn, cv, ct, cn, c):
    """读一个面角点：v、v/vt、v//vn、v/vt/vn。写进 cv/ct/cn[c]，返回词尾位置。"""
    raw, j, ok = _parse_int(data, i, end)
    cv[c] = _resolve(raw, ok, nv)
    ct[c] = _IDX_MISSING
    cn[c] = _IDX_MISSING
    if j < end and data[j] == 47:
        j += 1
        if j < end and data[j] != 47 and not _is_stop(data, j, end):
            raw, j, ok = _parse_int(data, j, end)
            ct[c] = _resolve(raw, ok, nvt)
        if j < end and data[j] == 47:
            j += 1
            if j < end and not _is_stop(data, j, end):
                raw, j, ok = _parse_int(data, j, end)
                cn[c] = _resolve(raw, ok, nvn)
    elif j < end and not _is_stop(data, j, end):
        # 数字后面跟着别的字符：这个角点无效
        cv[c] = _IDX_INVALID
    while j < end and not _is_stop(data, j, end):
        j += 1
    return j


@njit(cache=True, nogil=True)
def _obj_count(data, start, end):
    """第一遍：数 [start, end) 里的 v、vt、vn、面、面角点、usemtl 个数。"""
    counts = np.zeros(6, np.int64)
    i = start
    while i < end:
        i = _skip_blank(data, i, end)
        if i >= end:
            break
        ch = data[i]
        if ch == 10 or ch == 13:
            i += 1
            continue
        if ch == 35:
            i = _next_line(data, i, end)
            continue
        kind, i = _keyword(data, i, end)
        if kind == _K_V:
            counts[0] += 1
            n, i = _skip_tokens(data, i, end)
        elif kind == _K_VT:
            counts[1] += 1
            n, i = _skip_tokens(data, i, end)
        elif kind == _K_VN:
            counts[2] += 1
            n, i = _skip_tokens(data, i, end)
        elif kind == _K_F:
            counts[3] += 1
            n, i = _skip_tokens(data, i, end)
            counts[4] += n
        elif kind == _K_USEMTL:
            counts[5] += 1
        i = _next_line(data, i, end)
    return counts


@njit(cache=True, nogil=True)
def _obj_fill(data, start, end, base, vpos, vtex, vnrm, face_first, face_n, cv, ct, cn, mtl_span, mtl_face):
    """第二遍：把 [start, end) 的内容填进全局数组。base 是这一块在各数组里的起点。"""
    iv = base[0]
    it = base[1]
    jn = base[2]
    f = base[3]
    c = base[4]
    m = base[5]
    i = start
    while i < end:
        i = _skip_blank(data, i, end)
        if i >= end:
            break
        ch = data[i]
        if ch == 10 or ch == 13:
            i += 1
            continue
        if ch == 35:
            i = _next_line(data, i, end)
            continue
        kind, i = _keyword(data, i, end)
        if kind == _K_V:
            i = _read_floats(data, i, end, vpos, iv, 3, 3)
            iv += 1
        elif kind == _K_VT:
            i = _read_floats(data, i, end, vtex, it, 2, 1)
            it += 1
        elif kind == _K_VN:
            i = _read_floats(data, i, end, vnrm, jn, 3, 3)
            jn += 1
        elif kind == _K_F:
            face_first[f] = c
            n = 0
            while True:
                i = _skip_blank(data, i, end)
                if i >= end:
                    break
                ch = data[i]
                if ch == 10 or ch == 13 or ch == 35:
                    break
                i = _read_corner(data, i, end, iv, it, jn, cv, ct, cn, c)
                c += 1
                n += 1
            face_n[f] = n
            f += 1
        elif kind == _K_USEMTL:
            i = _skip_blank(data, i, end)
            s = i
            while i < end and data[i] != 10 and data[i] != 13 and data[i] != 35:
                i += 1
            e = i
            while e > s and (data[e - 1] == 32 or data[e - 1] == 9):
                e -= 1
            mtl_span[m, 0] = s
            mtl_span[m, 1] = e
            mtl_face[m] = f
            m += 1
        i = _next_line(data, i, end)


@njit(cache=True, nogil=True)
def _obj_triangulate(face_first, face_n, cv, ct, cn, nv, nvt, nvn):
    """检查每个面的索引，把多边形按扇形拆成三角形。

    返回 (三角形的 v 索引, vt 索引, vn 索引, 来源面号, 索引越界的面数, 顶点不足 3 个的面数)。
    vt、vn 没写时为 -1。
    """
    nf = face_first.shape[0]
    ok = np.zeros(nf, np.bool_)
    ntri = 0
    bad = 0
    short = 0
    for f in range(nf):
        n = face_n[f]
        if n < 3:
            short += 1
            continue
        s = face_first[f]
        good = True
        for c in range(s, s + n):
            v = cv[c]
            if v < 0 or v >= nv:
                good = False
                break
            t = ct[c]
            if t != _IDX_MISSING and (t < 0 or t >= nvt):
                good = False
                break
            q = cn[c]
            if q != _IDX_MISSING and (q < 0 or q >= nvn):
                good = False
                break
        if good:
            ok[f] = True
            ntri += n - 2
        else:
            bad += 1
    tv = np.empty((ntri, 3), np.int64)
    tt = np.empty((ntri, 3), np.int64)
    tn = np.empty((ntri, 3), np.int64)
    tf = np.empty(ntri, np.int64)
    k = 0
    for f in range(nf):
        if not ok[f]:
            continue
        s = face_first[f]
        n = face_n[f]
        for j in range(1, n - 1):
            a = s
            b = s + j
            c = s + j + 1
            tv[k, 0] = cv[a]
            tv[k, 1] = cv[b]
            tv[k, 2] = cv[c]
            tt[k, 0] = ct[a]
            tt[k, 1] = ct[b]
            tt[k, 2] = ct[c]
            tn[k, 0] = cn[a]
            tn[k, 1] = cn[b]
            tn[k, 2] = cn[c]
            tf[k] = f
            k += 1
    return tv, tt, tn, tf, bad, short


def _decode_name(raw: bytes) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace").strip()


def _load_obj(path, missing_normals: str = "smooth", *, threads: int = 0,
              chunk_bytes: int = OBJ_CHUNK_BYTES) -> MeshData:
    p = Path(path)
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError as exc:
        raise MeshImportError(f"读取文件失败：{p}（{exc}）") from exc
    return _obj_from_bytes(data, p.stem, str(p), missing_normals, threads=threads, chunk_bytes=chunk_bytes)


def _obj_from_bytes(data: np.ndarray, name: str, source: str, missing_normals: str = "smooth", *,
                    threads: int = 0, chunk_bytes: int = OBJ_CHUNK_BYTES) -> MeshData:
    """解析内存里的 OBJ 字节（uint8 数组）。"""
    data = np.asarray(data, dtype=np.uint8)
    if not data.flags.c_contiguous or not data.flags.writeable:
        data = data.copy()
    end = int(data.shape[0])
    start = 3 if end >= 3 and data[0] == 0xEF and data[1] == 0xBB and data[2] == 0xBF else 0

    # 按行边界切块
    chunk_bytes = max(1024, int(chunk_bytes))
    spans = []
    s = start
    while s < end:
        e = end if s + chunk_bytes >= end else int(_line_start_after(data, s + chunk_bytes, end))
        spans.append((s, e))
        s = e
    if not spans:
        spans.append((start, end))
    workers = _workers(threads)

    counts = np.stack(_run_jobs([lambda a=a, b=b: _obj_count(data, a, b) for a, b in spans], workers))
    bases = np.zeros_like(counts)
    if len(spans) > 1:
        bases[1:] = np.cumsum(counts, axis=0)[:-1]
    nv, nvt, nvn, nf, nc, nm = (int(x) for x in counts.sum(axis=0))

    vpos = np.empty((nv, 3), np.float32)
    vtex = np.empty((nvt, 2), np.float32)
    vnrm = np.empty((nvn, 3), np.float32)
    face_first = np.empty(nf, np.int64)
    face_n = np.empty(nf, np.int64)
    cv = np.empty(nc, np.int64)
    ct = np.empty(nc, np.int64)
    cn = np.empty(nc, np.int64)
    mtl_span = np.empty((nm, 2), np.int64)
    mtl_face = np.empty(nm, np.int64)
    _run_jobs([lambda a=a, b=b, base=base: _obj_fill(data, a, b, base, vpos, vtex, vnrm, face_first, face_n,
                                                     cv, ct, cn, mtl_span, mtl_face)
               for (a, b), base in zip(spans, bases)], workers)

    tv, tt, tn, tf, n_bad, n_short = _obj_triangulate(face_first, face_n, cv, ct, cn, nv, nvt, nvn)
    warnings: list[str] = []
    if nf == 0:
        raise MeshImportError("文件里没有面（f 语句），不是可用的模型")
    if n_bad:
        warnings.append(f"跳过 {n_bad} 个引用了不存在的顶点、UV 或法线的面")
    if n_short:
        warnings.append(f"跳过 {n_short} 个顶点数不足 3 个的面")

    # 材质：没有 usemtl 的面归默认材质；同名的 usemtl 是同一个材质
    order: list[str] = []
    key_of: dict[str, int] = {}
    occ_key = np.empty(nm, np.int64)
    for m in range(nm):
        a, b = int(mtl_span[m, 0]), int(mtl_span[m, 1])
        nm_text = _decode_name(data[a:b].tobytes()) or DEFAULT_MATERIAL
        if nm_text not in key_of:
            key_of[nm_text] = len(order)
            order.append(nm_text)
        occ_key[m] = key_of[nm_text]
    default_key = key_of.get(DEFAULT_MATERIAL, -1)
    mat_names = {k: n for k, n in enumerate(order)}
    mat_names.setdefault(default_key, DEFAULT_MATERIAL)
    if nm:
        occ = np.searchsorted(mtl_face, tf, side="right") - 1
        tri_key = np.where(occ >= 0, occ_key[np.maximum(occ, 0)], default_key)
    else:
        tri_key = np.full(tf.shape[0], default_key, np.int64)

    # 组装逐角点数组
    T = tv.shape[0]
    pos = vpos[tv]                                    # (T, 3, 3)
    uv = np.zeros((T, 3, 2), np.float32)
    uv_ok = tt >= 0
    if nvt:
        uv[uv_ok] = vtex[tt[uv_ok]]
    nrm = np.zeros((T, 3, 3), np.float32)
    nrm_ok = tn >= 0
    if nvn:
        nrm[nrm_ok] = vnrm[tn[nrm_ok]]
    return _finalize(name, source, pos, uv, uv_ok, nrm, nrm_ok, tri_key, mat_names, warnings,
                     missing_normals, vertex_ids=tv, vertex_pos=vpos, file_has_uvs=nvt > 0)


# =====================================================================================
# 公共收尾：剔除坏三角形、补法线、整理材质
# =====================================================================================

def _weld_bits(points: np.ndarray):
    """按坐标的二进制位完全相同来焊接点（-0 与 +0 视为相同）。返回 (每个点的组号, 组数)。"""
    p = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3) + np.float32(0.0)
    n = p.shape[0]
    if n == 0:
        return np.zeros(0, np.int64), 0
    bits = p.view(np.uint32)
    k1 = (bits[:, 0].astype(np.uint64) << np.uint64(32)) | bits[:, 1].astype(np.uint64)
    k2 = bits[:, 2]
    order = np.lexsort((k2, k1))
    s1 = k1[order]
    s2 = k2[order]
    new = np.empty(n, np.bool_)
    new[0] = True
    new[1:] = (s1[1:] != s1[:-1]) | (s2[1:] != s2[:-1])
    gid = np.cumsum(new) - 1
    ids = np.empty(n, np.int64)
    ids[order] = gid
    return ids, int(gid[-1]) + 1


@njit(cache=True, nogil=True)
def _angle_weighted_normals(pos, weld, nweld):
    """平滑法线：每个面的单位法线乘以它在该顶点处的夹角，按焊接后的顶点累加再归一化。

    pos 是 (T*3, 3) 逐角点坐标，weld 是每个角点的焊接组号。返回 (T*3, 3) float32 单位向量。
    累加结果接近零（比如两面背靠背）的角点退回用面法线。
    """
    T = pos.shape[0] // 3
    acc = np.zeros((nweld, 3))
    fn = np.zeros((T, 3))
    for k in range(T):
        b = 3 * k
        x0 = float(pos[b, 0])
        y0 = float(pos[b, 1])
        z0 = float(pos[b, 2])
        x1 = float(pos[b + 1, 0])
        y1 = float(pos[b + 1, 1])
        z1 = float(pos[b + 1, 2])
        x2 = float(pos[b + 2, 0])
        y2 = float(pos[b + 2, 1])
        z2 = float(pos[b + 2, 2])
        ax = x1 - x0
        ay = y1 - y0
        az = z1 - z0
        bx = x2 - x0
        by = y2 - y0
        bz = z2 - z0
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        L = math.sqrt(cx * cx + cy * cy + cz * cz)
        if not (L > 0.0):
            continue
        nx = cx / L
        ny = cy / L
        nz = cz / L
        fn[k, 0] = nx
        fn[k, 1] = ny
        fn[k, 2] = nz
        # 三个角处的夹角：|叉积| 对三个角都等于 L，只有点积不同
        d0 = ax * bx + ay * by + az * bz
        d1 = (x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1) + (z0 - z1) * (z2 - z1)
        d2 = (x0 - x2) * (x1 - x2) + (y0 - y2) * (y1 - y2) + (z0 - z2) * (z1 - z2)
        a0 = math.atan2(L, d0)
        a1 = math.atan2(L, d1)
        a2 = math.atan2(L, d2)
        w = weld[b]
        acc[w, 0] += a0 * nx
        acc[w, 1] += a0 * ny
        acc[w, 2] += a0 * nz
        w = weld[b + 1]
        acc[w, 0] += a1 * nx
        acc[w, 1] += a1 * ny
        acc[w, 2] += a1 * nz
        w = weld[b + 2]
        acc[w, 0] += a2 * nx
        acc[w, 1] += a2 * ny
        acc[w, 2] += a2 * nz
    out = np.empty((3 * T, 3), np.float32)
    for k in range(T):
        for c in range(3):
            w = weld[3 * k + c]
            vx = acc[w, 0]
            vy = acc[w, 1]
            vz = acc[w, 2]
            L = math.sqrt(vx * vx + vy * vy + vz * vz)
            if L > 1e-12:
                out[3 * k + c, 0] = vx / L
                out[3 * k + c, 1] = vy / L
                out[3 * k + c, 2] = vz / L
            elif fn[k, 0] != 0.0 or fn[k, 1] != 0.0 or fn[k, 2] != 0.0:
                out[3 * k + c, 0] = fn[k, 0]
                out[3 * k + c, 1] = fn[k, 1]
                out[3 * k + c, 2] = fn[k, 2]
            else:
                out[3 * k + c, 0] = 0.0
                out[3 * k + c, 1] = 1.0
                out[3 * k + c, 2] = 0.0
    return out


def _unique_names(names: list[str]) -> list[str]:
    """重名的后面加 .001、.002……"""
    seen: set[str] = set()
    out = []
    for nm in names:
        cand = nm
        k = 0
        while cand in seen:
            k += 1
            cand = f"{nm}.{k:03d}"
        seen.add(cand)
        out.append(cand)
    return out


def _finalize(name, source, pos, uv, uv_ok, nrm, nrm_ok, tri_key, mat_names, warnings, normal_mode, *,
              vertex_ids=None, vertex_pos=None, file_has_uvs=True) -> MeshData:
    """把三角形汤整理成 MeshData。

    pos (T,3,3)、uv (T,3,2)、nrm (T,3,3) 为 float32；uv_ok、nrm_ok (T,3) 表示该角点文件里有没有给值。
    tri_key (T,) 是材质键，mat_names 是 {键: 名字}。材质按第一次被三角形用到的先后编号，没用到的去掉。
    vertex_ids/vertex_pos：OBJ 的顶点表，平滑法线按它焊接更快；为 None 时按角点坐标焊接。
    """
    p64 = pos.astype(np.float64)
    finite = np.isfinite(p64).all(axis=(1, 2))
    cr = np.cross(p64[:, 1] - p64[:, 0], p64[:, 2] - p64[:, 0])
    nondeg = (cr != 0.0).any(axis=1)
    keep = finite & nondeg
    n_nonfinite = int((~finite).sum())
    n_degen = int((finite & ~nondeg).sum())
    if n_nonfinite:
        warnings.append(f"跳过 {n_nonfinite} 个坐标不是有效数值的三角形")
    if n_degen:
        warnings.append(f"跳过 {n_degen} 个面积为零的三角形")
    if not keep.all():
        pos = pos[keep]
        uv = uv[keep]
        uv_ok = uv_ok[keep]
        nrm = nrm[keep]
        nrm_ok = nrm_ok[keep]
        tri_key = tri_key[keep]
        cr = cr[keep]
        if vertex_ids is not None:
            vertex_ids = vertex_ids[keep]
    T = int(pos.shape[0])
    if T == 0:
        detail = "；".join(warnings)
        raise MeshImportError("文件里没有可用的三角形" + (f"（{detail}）" if detail else ""))

    # UV
    uv_finite = np.isfinite(uv).all(axis=2)
    if not uv_finite.all():
        uv[~uv_finite] = 0.0
        uv_ok = uv_ok & uv_finite
    has_uvs = bool(uv_ok.any())
    if not has_uvs:
        uv[:] = 0.0
        if file_has_uvs:
            warnings.append("模型的面没有引用 UV 坐标，无法在贴图上绘制")
        else:
            warnings.append("模型没有 UV 坐标，无法在贴图上绘制")
    else:
        lacking = int((~uv_ok).any(axis=1).sum())
        if lacking:
            warnings.append(f"{lacking} 个三角形缺少 UV，按 (0, 0) 处理")

    # 法线
    n64 = nrm.astype(np.float64)
    length = np.sqrt((n64 * n64).sum(axis=2))
    good = nrm_ok & np.isfinite(length) & (length > 1e-12)
    out = np.zeros((T, 3, 3), np.float32)
    if good.any():
        out[good] = (n64[good] / length[good][:, None]).astype(np.float32)
    need = ~good
    n_need = int(need.sum())
    if n_need:
        if normal_mode == "flat":
            fl = cr / np.sqrt((cr * cr).sum(axis=1))[:, None]
            comp = np.repeat(fl[:, None, :], 3, axis=1).astype(np.float32)
        else:
            if vertex_ids is not None and vertex_pos is not None:
                ids, nweld = _weld_bits(vertex_pos)
                weld = ids[vertex_ids.reshape(-1)]
            else:
                weld, nweld = _weld_bits(pos.reshape(-1, 3))
            comp = _angle_weighted_normals(np.ascontiguousarray(pos.reshape(-1, 3)), weld, nweld).reshape(T, 3, 3)
        out[need] = comp[need]
        how = "平直" if normal_mode == "flat" else "平滑"
        if n_need == 3 * T:
            warnings.append(f"文件没有法线，已按{how}方式计算")
        else:
            warnings.append(f"{n_need} 个角点的法线缺失或无效，已按{how}方式重新计算")

    # 材质：按第一次被用到的先后编号，没用到的自然不出现
    ukeys, first = np.unique(tri_key, return_index=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(order.shape[0], np.int64)
    rank[order] = np.arange(order.shape[0])
    material_ids = rank[np.searchsorted(ukeys, tri_key)].astype(np.int32)
    names = _unique_names([mat_names.get(int(k), f"材质{int(k)}") for k in ukeys[order]])

    positions = np.ascontiguousarray(pos.reshape(-1, 3), dtype=np.float32)
    bmin = positions.min(axis=0).astype(np.float32)
    bmax = positions.max(axis=0).astype(np.float32)
    return MeshData(name=name, positions=positions,
                    normals=np.ascontiguousarray(out.reshape(-1, 3)),
                    uvs=np.ascontiguousarray(uv.reshape(-1, 2), dtype=np.float32),
                    material_ids=np.ascontiguousarray(material_ids),
                    materials=names, bounds_min=bmin, bounds_max=bmax, source_path=source,
                    has_uvs=has_uvs, warnings=warnings)


# =====================================================================================
# glTF / GLB
# =====================================================================================

_GLTF_COMPONENT = {5120: np.dtype("<i1"), 5121: np.dtype("<u1"), 5122: np.dtype("<i2"),
                   5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
_GLTF_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_GLB_MAGIC = 0x46546C67
_GLB_JSON = 0x4E4F534A
_GLB_BIN = 0x004E4942
_UNSUPPORTED_EXT = {
    "KHR_draco_mesh_compression": "模型用了 Draco 网格压缩，暂时读不了，请导出时关掉压缩",
    "EXT_meshopt_compression": "模型用了 meshopt 压缩，暂时读不了，请导出时关掉压缩",
    "KHR_meshopt_compression": "模型用了 meshopt 压缩，暂时读不了，请导出时关掉压缩",
}


class _Gltf:
    """glTF 文档加缓冲区的按需读取。"""

    def __init__(self, doc: dict, base_dir: Path, glb_bin: bytes | None) -> None:
        self.doc = doc
        self.base_dir = base_dir
        self.glb_bin = glb_bin
        self._buffers: dict[int, np.ndarray] = {}

    def _list(self, key: str) -> list:
        value = self.doc.get(key, [])
        return value if isinstance(value, list) else []

    def _item(self, key: str, index, what: str) -> dict:
        items = self._list(key)
        if not isinstance(index, int) or index < 0 or index >= len(items):
            raise MeshImportError(f"glTF 文件损坏：{what} {index} 不存在")
        return items[index]

    def buffer(self, index: int) -> np.ndarray:
        if index in self._buffers:
            return self._buffers[index]
        buf = self._item("buffers", index, "缓冲区")
        uri = buf.get("uri")
        if uri is None:
            if self.glb_bin is None:
                raise MeshImportError(f"glTF 文件损坏：缓冲区 {index} 没有数据")
            raw = self.glb_bin
        elif uri.startswith("data:"):
            header, _, payload = uri.partition(",")
            try:
                raw = base64.b64decode(payload) if ";base64" in header else urllib.parse.unquote_to_bytes(payload)
            except (ValueError, TypeError) as exc:
                raise MeshImportError(f"glTF 文件损坏：缓冲区 {index} 的内嵌数据解不开") from exc
        else:
            file = self.base_dir / urllib.parse.unquote(uri)
            if not file.is_file():
                raise MeshImportError(f"找不到 glTF 引用的文件：{urllib.parse.unquote(uri)}")
            raw = file.read_bytes()
        length = int(buf.get("byteLength", len(raw)))
        if len(raw) < length:
            raise MeshImportError(f"glTF 文件损坏：缓冲区 {index} 比声明的短")
        arr = np.frombuffer(raw, dtype=np.uint8)[:length]
        self._buffers[index] = arr
        return arr

    def view(self, index: int):
        bv = self._item("bufferViews", index, "缓冲视图")
        buf = self.buffer(int(bv.get("buffer", -1)))
        off = int(bv.get("byteOffset", 0))
        length = int(bv["byteLength"])
        if off < 0 or off + length > buf.shape[0]:
            raise MeshImportError(f"glTF 文件损坏：缓冲视图 {index} 超出缓冲区")
        stride = bv.get("byteStride")
        return buf[off:off + length], (int(stride) if stride else None)

    def _typed(self, view_index: int, byte_offset: int, count: int, dtype: np.dtype, ncomp: int, what: str):
        raw, stride = self.view(view_index)
        esz = dtype.itemsize * ncomp
        stride = stride or esz
        if count == 0:
            return np.zeros((0, ncomp), dtype)
        need = byte_offset + (count - 1) * stride + esz
        if byte_offset < 0 or need > raw.shape[0]:
            raise MeshImportError(f"glTF 文件损坏：{what}超出缓冲视图")
        if stride == esz:
            chunk = raw[byte_offset:byte_offset + count * esz].copy()
        else:
            rows = byte_offset + np.arange(count, dtype=np.int64)[:, None] * stride
            chunk = raw[rows + np.arange(esz, dtype=np.int64)[None, :]]
        return chunk.view(dtype).reshape(count, ncomp)

    def accessor(self, index: int):
        """读访问器，返回 (原始类型的 (count, 分量数) 数组, componentType, normalized)。支持稀疏访问器。"""
        acc = self._item("accessors", index, "访问器")
        ctype = acc.get("componentType")
        if ctype not in _GLTF_COMPONENT:
            raise MeshImportError(f"glTF 文件损坏：访问器 {index} 的分量类型 {ctype} 不认识")
        dtype = _GLTF_COMPONENT[ctype]
        kind = acc.get("type")
        if kind not in _GLTF_NCOMP:
            raise MeshImportError(f"glTF 文件损坏：访问器 {index} 的类型 {kind} 不认识")
        if kind.startswith("MAT"):
            raise MeshImportError(f"glTF 访问器 {index} 是矩阵类型，网格属性里不该出现")
        ncomp = _GLTF_NCOMP[kind]
        count = int(acc.get("count", 0))
        if "bufferView" in acc:
            arr = self._typed(int(acc["bufferView"]), int(acc.get("byteOffset", 0)), count, dtype, ncomp,
                              f"访问器 {index} ")
        else:
            arr = np.zeros((count, ncomp), dtype)
        sparse = acc.get("sparse")
        if sparse:
            try:
                scount = int(sparse["count"])
                sidx = sparse["indices"]
                sval = sparse["values"]
                ictype = sidx["componentType"]
                if ictype not in (5121, 5123, 5125):
                    raise MeshImportError(f"glTF 文件损坏：访问器 {index} 的稀疏索引类型不对")
                ind = self._typed(int(sidx["bufferView"]), int(sidx.get("byteOffset", 0)), scount,
                                  _GLTF_COMPONENT[ictype], 1, f"访问器 {index} 的稀疏索引")[:, 0].astype(np.int64)
                val = self._typed(int(sval["bufferView"]), int(sval.get("byteOffset", 0)), scount, dtype, ncomp,
                                  f"访问器 {index} 的稀疏数值")
            except KeyError as exc:
                raise MeshImportError(f"glTF 文件损坏：访问器 {index} 的稀疏数据缺字段 {exc}") from exc
            if scount and (ind.min() < 0 or ind.max() >= count):
                raise MeshImportError(f"glTF 文件损坏：访问器 {index} 的稀疏索引越界")
            arr = arr.copy()
            arr[ind] = val
        return arr, ctype, bool(acc.get("normalized", False))

    def accessor_float(self, index: int) -> np.ndarray:
        arr, ctype, normalized = self.accessor(index)
        if ctype == 5126:
            return arr.astype(np.float64)
        a = arr.astype(np.float64)
        if normalized:
            if ctype == 5121:
                return a / 255.0
            if ctype == 5123:
                return a / 65535.0
            if ctype == 5125:
                return a / 4294967295.0
            if ctype == 5120:
                return np.maximum(a / 127.0, -1.0)
            if ctype == 5122:
                return np.maximum(a / 32767.0, -1.0)
        return a


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _node_matrix(node: dict) -> np.ndarray:
    """节点的局部变换。matrix 是列主序；否则按 平移 × 旋转 × 缩放 组合。"""
    if "matrix" in node:
        m = np.asarray(node["matrix"], dtype=np.float64)
        if m.shape != (16,):
            raise MeshImportError("glTF 文件损坏：节点矩阵不是 16 个数")
        return m.reshape(4, 4).T.copy()
    out = np.eye(4)
    rot = _quat_matrix(node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    scale = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    out[:3, :3] = rot * scale[None, :]
    out[:3, 3] = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    return out


def _gltf_triangles(idx: np.ndarray, mode: int) -> np.ndarray:
    """按图元模式把索引序列变成三角形 (t, 3)。"""
    n = idx.shape[0]
    if mode == 4:
        return idx[:n - n % 3].reshape(-1, 3)
    if n < 3:
        return np.zeros((0, 3), np.int64)
    i = np.arange(n - 2)
    if mode == 5:      # 三角形带：p_i = {v_i, v_{i+1+i%2}, v_{i+2-i%2}}
        odd = (i % 2).astype(bool)
        return np.stack([idx[i], np.where(odd, idx[i + 2], idx[i + 1]), np.where(odd, idx[i + 1], idx[i + 2])], 1)
    # 三角形扇：p_i = {v_{i+1}, v_{i+2}, v_0}
    return np.stack([idx[i + 1], idx[i + 2], np.full(n - 2, idx[0])], 1)


def _read_gltf_document(path: Path):
    raw = path.read_bytes()
    if raw[:4] == b"glTF":
        if len(raw) < 12:
            raise MeshImportError("GLB 文件不完整")
        magic, version, length = struct.unpack_from("<III", raw, 0)
        if magic != _GLB_MAGIC:
            raise MeshImportError("不是有效的 GLB 文件")
        if version != 2:
            raise MeshImportError(f"只支持 glTF 2.0，这个文件是 {version}.x 版")
        length = min(length, len(raw))
        off = 12
        doc = None
        bin_chunk = None
        while off + 8 <= length:
            clen, ctype = struct.unpack_from("<II", raw, off)
            off += 8
            chunk = raw[off:off + clen]
            off += clen
            if ctype == _GLB_JSON and doc is None:
                try:
                    doc = json.loads(chunk.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise MeshImportError("GLB 文件里的 JSON 损坏") from exc
            elif ctype == _GLB_BIN and bin_chunk is None:
                bin_chunk = chunk
        if doc is None:
            raise MeshImportError("GLB 文件里没有 JSON 块")
        return doc, bin_chunk
    if path.suffix.lower() == ".glb":
        raise MeshImportError("不是有效的 GLB 文件")
    try:
        doc = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MeshImportError("glTF 文件的 JSON 损坏") from exc
    return doc, None


def _load_gltf(path: Path, missing_normals: str = "flat") -> MeshData:
    doc, glb_bin = _read_gltf_document(path)
    if not isinstance(doc, dict):
        raise MeshImportError("glTF 文件的 JSON 结构不对")
    version = str(doc.get("asset", {}).get("version", "2.0"))
    if not version.startswith("2"):
        raise MeshImportError(f"只支持 glTF 2.0，这个文件是 {version} 版")
    for ext in doc.get("extensionsRequired", []) or []:
        if ext in _UNSUPPORTED_EXT:
            raise MeshImportError(_UNSUPPORTED_EXT[ext])
    g = _Gltf(doc, path.parent, glb_bin)
    warnings: list[str] = []
    meshes = g._list("meshes")
    nodes = g._list("nodes")
    if not meshes:
        raise MeshImportError("glTF 文件里没有网格")

    # 场景遍历，收集 (网格号, 世界矩阵)
    instances: list[tuple[int, np.ndarray]] = []
    if nodes:
        scenes = g._list("scenes")
        if scenes:
            sidx = doc.get("scene", 0)
            if not isinstance(sidx, int) or sidx < 0 or sidx >= len(scenes):
                sidx = 0
            roots = list(scenes[sidx].get("nodes", []))
        else:
            children = {c for n in nodes for c in n.get("children", [])}
            roots = [i for i in range(len(nodes)) if i not in children]
        visited: set[int] = set()
        stack = [(r, np.eye(4)) for r in reversed(roots)]
        n_cycle = 0
        n_skin = 0
        while stack:
            ni, parent = stack.pop()
            if not isinstance(ni, int) or ni < 0 or ni >= len(nodes):
                raise MeshImportError(f"glTF 文件损坏：节点 {ni} 不存在")
            if ni in visited:
                n_cycle += 1
                continue
            visited.add(ni)
            node = nodes[ni]
            world = parent @ _node_matrix(node)
            if "mesh" in node:
                if "skin" in node:
                    # 蒙皮网格的顶点在绑定空间里，按规范忽略所在节点的变换
                    n_skin += 1
                    instances.append((int(node["mesh"]), np.eye(4)))
                else:
                    instances.append((int(node["mesh"]), world))
            for c in reversed(node.get("children", [])):
                stack.append((c, world))
        if n_cycle:
            warnings.append("节点层级里有重复引用，重复的部分已忽略")
        if n_skin:
            warnings.append("蒙皮网格按绑定姿态读入")
        if not instances:
            raise MeshImportError("glTF 场景里没有用到任何网格")
    else:
        instances = [(m, np.eye(4)) for m in range(len(meshes))]
        warnings.append("文件没有节点，网格按原始坐标读入")

    pos_parts, nrm_parts, nok_parts, uv_parts, uok_parts, key_parts = [], [], [], [], [], []
    n_bad_index = 0
    n_skip_mode = 0
    n_no_pos = 0
    n_bad_mat = 0
    any_uv = False
    materials = g._list("materials")
    for mesh_index, world in instances:
        if mesh_index < 0 or mesh_index >= len(meshes):
            raise MeshImportError(f"glTF 文件损坏：网格 {mesh_index} 不存在")
        m3 = world[:3, :3]
        det = float(np.linalg.det(m3))
        if det != 0.0 and np.isfinite(det):
            nmat = np.linalg.inv(m3).T
        else:
            nmat = None
        for prim in meshes[mesh_index].get("primitives", []):
            for ext in (prim.get("extensions") or {}):
                if ext in _UNSUPPORTED_EXT:
                    raise MeshImportError(_UNSUPPORTED_EXT[ext])
            mode = int(prim.get("mode", 4))
            if mode not in (4, 5, 6):
                n_skip_mode += 1
                continue
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                n_no_pos += 1
                continue
            P = g.accessor_float(attrs["POSITION"])
            if P.shape[1] < 3:
                raise MeshImportError("glTF 文件损坏：POSITION 不是三维向量")
            P = P[:, :3]
            nv = P.shape[0]
            N = None
            if "NORMAL" in attrs:
                N = g.accessor_float(attrs["NORMAL"])[:, :3]
                if N.shape[0] != nv:
                    raise MeshImportError("glTF 文件损坏：NORMAL 与 POSITION 的个数不同")
            UV = None
            if "TEXCOORD_0" in attrs:
                UV = g.accessor_float(attrs["TEXCOORD_0"])[:, :2].copy()
                if UV.shape[0] != nv:
                    raise MeshImportError("glTF 文件损坏：TEXCOORD_0 与 POSITION 的个数不同")
                UV[:, 1] = 1.0 - UV[:, 1]     # glTF 的 UV 原点在左上角
                any_uv = True
            if "indices" in prim:
                idx_arr, ictype, _ = g.accessor(int(prim["indices"]))
                if ictype not in (5121, 5123, 5125):
                    raise MeshImportError("glTF 文件损坏：索引必须是无符号整数")
                idx = idx_arr[:, 0].astype(np.int64)
            else:
                idx = np.arange(nv, dtype=np.int64)
            tri = _gltf_triangles(idx, mode)
            bad = ((tri < 0) | (tri >= nv)).any(axis=1)
            if bad.any():
                n_bad_index += int(bad.sum())
                tri = tri[~bad]
            if tri.shape[0] == 0:
                continue
            if det < 0.0:
                tri = tri[:, [0, 2, 1]]       # 镜像变换会翻转朝向，交换两个角保持逆时针为正面
            Pw = P @ m3.T + world[:3, 3]
            pos_parts.append(Pw[tri].astype(np.float32))
            if N is not None and nmat is not None:
                nrm_parts.append((N @ nmat.T)[tri].astype(np.float32))
                nok_parts.append(np.ones(tri.shape, bool))
            else:
                nrm_parts.append(np.zeros(tri.shape + (3,), np.float32))
                nok_parts.append(np.zeros(tri.shape, bool))
            if UV is not None:
                uv_parts.append(UV[tri].astype(np.float32))
                uok_parts.append(np.ones(tri.shape, bool))
            else:
                uv_parts.append(np.zeros(tri.shape + (2,), np.float32))
                uok_parts.append(np.zeros(tri.shape, bool))
            mi = prim.get("material", -1)
            if not isinstance(mi, int) or mi < -1 or mi >= len(materials):
                n_bad_mat += 1
                mi = -1
            key_parts.append(np.full(tri.shape[0], mi, np.int64))
    if n_skip_mode:
        warnings.append(f"跳过 {n_skip_mode} 组点或线图元")
    if n_no_pos:
        warnings.append(f"跳过 {n_no_pos} 组没有顶点坐标的图元")
    if n_bad_index:
        warnings.append(f"跳过 {n_bad_index} 个索引越界的三角形")
    if n_bad_mat:
        warnings.append(f"{n_bad_mat} 组图元引用了不存在的材质，归到默认材质")
    if not pos_parts:
        raise MeshImportError("glTF 文件里没有可用的三角形" + (f"（{'；'.join(warnings)}）" if warnings else ""))

    mat_names = {-1: DEFAULT_MATERIAL}
    for i, mat in enumerate(materials):
        nm_text = str(mat.get("name") or "").strip() if isinstance(mat, dict) else ""
        mat_names[i] = nm_text or f"材质{i}"
    return _finalize(path.stem, str(path), np.concatenate(pos_parts), np.concatenate(uv_parts),
                     np.concatenate(uok_parts), np.concatenate(nrm_parts), np.concatenate(nok_parts),
                     np.concatenate(key_parts), mat_names, warnings, missing_normals, file_has_uvs=any_uv)


# =====================================================================================
# 测试模型
# =====================================================================================

TEST_MESH_KINDS = ("sphere", "dense_sphere", "cube", "plane", "torus")


def _soup_mesh(name: str, P: np.ndarray, N: np.ndarray, UV: np.ndarray) -> MeshData:
    """由 (T,3,3) 坐标、法线和 (T,3,2) UV 组装 MeshData（单一默认材质）。"""
    T = P.shape[0]
    N = N / np.sqrt((N * N).sum(axis=-1, keepdims=True))
    positions = np.ascontiguousarray(P.reshape(-1, 3), dtype=np.float32)
    return MeshData(name=name, positions=positions,
                    normals=np.ascontiguousarray(N.reshape(-1, 3), dtype=np.float32),
                    uvs=np.ascontiguousarray(UV.reshape(-1, 2), dtype=np.float32),
                    material_ids=np.zeros(T, np.int32), materials=[DEFAULT_MATERIAL],
                    bounds_min=positions.min(axis=0).astype(np.float32),
                    bounds_max=positions.max(axis=0).astype(np.float32),
                    source_path="", has_uvs=True, warnings=[])


def _sphere(segments: int = 64, rings: int = 32, radius: float = 1.0, name: str = "sphere") -> MeshData:
    """经纬球。u 沿经度 0→1，v 从南极 0 到北极 1；两极各一圈三角形，极点的 u 取所在格的中间。"""
    s = int(segments)
    r = int(rings)
    if s < 3 or r < 2:
        raise ValueError("球至少要 3 段经度、2 圈纬度")
    radius = float(radius)
    u = np.arange(s + 1) / s
    v = np.arange(r + 1) / r
    phi = 2.0 * np.pi * u
    theta = np.pi * v
    rho = np.sin(theta)
    rho[0] = 0.0
    rho[-1] = 0.0
    y = -np.cos(theta)
    # 网格点 (i 纬度, j 经度)：x = ρcosφ，z = -ρsinφ，使 UV 逆时针 = 从外面看逆时针
    gp = np.stack([rho[:, None] * np.cos(phi)[None, :], np.broadcast_to(y[:, None], (r + 1, s + 1)),
                   -rho[:, None] * np.sin(phi)[None, :]], axis=-1)
    guv = np.stack(np.broadcast_arrays(u[None, :], v[:, None]), axis=-1)
    tris_p = []
    tris_uv = []
    j = np.arange(s)
    # 南极帽：(极点, (1,j+1), (1,j))
    pole_s = np.zeros((s, 3))
    pole_s[:, 1] = -1.0
    pole_uv = np.stack([(j + 0.5) / s, np.zeros(s)], axis=-1)
    tris_p.append(np.stack([pole_s, gp[1, j + 1], gp[1, j]], axis=1))
    tris_uv.append(np.stack([pole_uv, guv[1, j + 1], guv[1, j]], axis=1))
    # 中间各圈：四边形 A(i,j) B(i,j+1) C(i+1,j+1) D(i+1,j) 拆成 ABC、ACD
    if r > 2:
        ii, jj = np.meshgrid(np.arange(1, r - 1), j, indexing="ij")
        ii = ii.ravel()
        jj = jj.ravel()
        A, B, C, D = gp[ii, jj], gp[ii, jj + 1], gp[ii + 1, jj + 1], gp[ii + 1, jj]
        a, b, c, d = guv[ii, jj], guv[ii, jj + 1], guv[ii + 1, jj + 1], guv[ii + 1, jj]
        quad_p = np.stack([np.stack([A, B, C], 1), np.stack([A, C, D], 1)], axis=1).reshape(-1, 3, 3)
        quad_uv = np.stack([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], axis=1).reshape(-1, 3, 2)
        tris_p.append(quad_p)
        tris_uv.append(quad_uv)
    # 北极帽：((r-1,j), (r-1,j+1), 极点)
    pole_n = np.zeros((s, 3))
    pole_n[:, 1] = 1.0
    pole_uv_n = np.stack([(j + 0.5) / s, np.ones(s)], axis=-1)
    tris_p.append(np.stack([gp[r - 1, j], gp[r - 1, j + 1], pole_n], axis=1))
    tris_uv.append(np.stack([guv[r - 1, j], guv[r - 1, j + 1], pole_uv_n], axis=1))
    unit = np.concatenate(tris_p)
    return _soup_mesh(name, unit * radius, unit, np.concatenate(tris_uv))


def _dense_sphere(triangles: int = 1_000_000, radius: float = 1.0) -> MeshData:
    """三角形数不少于 triangles 的经纬球（经度段数取纬度圈数的两倍）。"""
    t = max(8, int(triangles))
    rings = max(2, math.ceil(0.5 + math.sqrt(0.25 + t / 4.0)))
    return _sphere(2 * rings, rings, radius, name="dense_sphere")


# 立方体六个面：(外法线, s 轴, t 轴)，满足 s × t = 法线
_CUBE_FACES = (
    ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
)


def cube_face_uv_rect(face: int, margin: float = 0.02) -> tuple[float, float, float, float]:
    """立方体第 face 个面在 UV 里占的正方形 (u0, v0, u1, v1)。三列两行排布，各面之间留 margin 的间隙。"""
    cw, ch = 1.0 / 3.0, 0.5
    q = min(cw, ch) - 2.0 * margin
    col, row = face % 3, face // 3
    u0 = col * cw + (cw - q) / 2.0
    v0 = row * ch + (ch - q) / 2.0
    return u0, v0, u0 + q, v0 + q


def _grid_patch(div: int):
    """单位正方形切成 div×div 格，每格两个三角形。返回 (T,3,2) 的局部坐标 0..1，逆时针。"""
    k = np.arange(div)
    ii, jj = np.meshgrid(k, k, indexing="ij")
    s0 = jj.ravel() / div
    t0 = ii.ravel() / div
    s1 = (jj.ravel() + 1) / div
    t1 = (ii.ravel() + 1) / div
    A = np.stack([s0, t0], -1)
    B = np.stack([s1, t0], -1)
    C = np.stack([s1, t1], -1)
    D = np.stack([s0, t1], -1)
    return np.stack([np.stack([A, B, C], 1), np.stack([A, C, D], 1)], axis=1).reshape(-1, 3, 2)


def _cube(size: float = 2.0, margin: float = 0.02, divisions: int = 1) -> MeshData:
    """立方体，每个面一个 UV 岛，三列两行互不重叠。法线是平直的面法线。"""
    div = max(1, int(divisions))
    margin = float(margin)
    if not 0.0 <= margin < 1.0 / 6.0:
        raise ValueError("margin 要在 0 到 1/6 之间")
    h = float(size) / 2.0
    local = _grid_patch(div)                      # (t,3,2)
    Ps, Ns, UVs = [], [], []
    for f, (n, sa, ta) in enumerate(_CUBE_FACES):
        n = np.asarray(n, float)
        sa = np.asarray(sa, float)
        ta = np.asarray(ta, float)
        sc = (local[..., 0] * 2.0 - 1.0) * h
        tc = (local[..., 1] * 2.0 - 1.0) * h
        P = n * h + sc[..., None] * sa + tc[..., None] * ta
        u0, v0, u1, v1 = cube_face_uv_rect(f, margin)
        UV = np.stack([u0 + local[..., 0] * (u1 - u0), v0 + local[..., 1] * (v1 - v0)], -1)
        Ps.append(P)
        Ns.append(np.broadcast_to(n, P.shape))
        UVs.append(UV)
    return _soup_mesh("cube", np.concatenate(Ps), np.concatenate(Ns), np.concatenate(UVs))


def _plane(size: float = 2.0, divisions: int = 1, margin: float = 0.0) -> MeshData:
    """XZ 平面上的正方形，朝 +Y。UV 铺满 [margin, 1-margin]。"""
    div = max(1, int(divisions))
    margin = float(margin)
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin 要在 0 到 0.5 之间")
    h = float(size) / 2.0
    local = _grid_patch(div)
    sc = (local[..., 0] * 2.0 - 1.0) * h
    tc = (local[..., 1] * 2.0 - 1.0) * h
    P = np.stack([sc, np.zeros_like(sc), -tc], -1)        # s 轴 = +X，t 轴 = -Z，s × t = +Y
    N = np.broadcast_to(np.array([0.0, 1.0, 0.0]), P.shape)
    UV = margin + local * (1.0 - 2.0 * margin)
    return _soup_mesh("plane", P, N, UV)


def _torus(major_radius: float = 1.0, minor_radius: float = 0.35, segments: int = 48, sides: int = 24) -> MeshData:
    """圆环，躺在 XZ 平面上。u 沿大圆，v 绕管子一圈。"""
    s = int(segments)
    n = int(sides)
    if s < 3 or n < 3:
        raise ValueError("圆环至少要 3 段、3 边")
    R = float(major_radius)
    r = float(minor_radius)
    if not 0.0 < r < R:
        raise ValueError("要求 0 < minor_radius < major_radius")
    u = np.arange(s + 1) / s
    v = np.arange(n + 1) / n
    phi = 2.0 * np.pi * u
    th = 2.0 * np.pi * v
    radial = np.stack([np.cos(phi), np.zeros_like(phi), -np.sin(phi)], -1)       # (s+1,3)
    up = np.array([0.0, 1.0, 0.0])
    nrm = np.cos(th)[:, None, None] * radial[None, :, :] + np.sin(th)[:, None, None] * up   # (n+1, s+1, 3)
    pos = R * radial[None, :, :] + r * nrm
    guv = np.stack(np.broadcast_arrays(u[None, :], v[:, None]), axis=-1)
    ii, jj = np.meshgrid(np.arange(n), np.arange(s), indexing="ij")
    ii = ii.ravel()
    jj = jj.ravel()

    def quad(g):
        A, B, C, D = g[ii, jj], g[ii, jj + 1], g[ii + 1, jj + 1], g[ii + 1, jj]
        return np.stack([np.stack([A, B, C], 1), np.stack([A, C, D], 1)], axis=1).reshape(-1, 3, g.shape[-1])

    return _soup_mesh("torus", quad(pos), quad(nrm), quad(guv))


def make_test_mesh(kind: str = "sphere", **kw) -> MeshData:
    """程序化测试模型，UV 都在 [0, 1] 内，三角形逆时针朝外。

    kind 与参数：
        "sphere"        segments=64, rings=32, radius=1.0（经纬展开，三角形数 = 2·segments·(rings-1)）
        "dense_sphere"  triangles=1_000_000, radius=1.0（三角形数不少于 triangles 的经纬球）
        "cube"          size=2.0, margin=0.02, divisions=1（六个面各占一块 UV，互不重叠、留间隙）
        "plane"         size=2.0, divisions=1, margin=0.0（XZ 平面，朝 +Y）
        "torus"         major_radius=1.0, minor_radius=0.35, segments=48, sides=24
    """
    builders = {"sphere": _sphere, "dense_sphere": _dense_sphere, "cube": _cube, "plane": _plane, "torus": _torus}
    if kind not in builders:
        raise ValueError(f"不认识的测试模型 {kind!r}，可选：{', '.join(TEST_MESH_KINDS)}")
    mesh = builders[kind](**kw)
    mesh.name = kind
    return mesh
