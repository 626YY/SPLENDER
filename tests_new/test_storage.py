"""工程文件存储 splender/doc/storage.py 的测试。

运行（工作目录 F:/SPLENDER）：
    python -m unittest tests_new/test_storage.py -v
加性能测试（写 4096 页、随机读 1000 页，结果写进 docs/evidence/storage/benchmark.json）：
    先设环境变量 SPLENDER_BENCH=1 再运行上面的命令。
强杀测试会启动本文件作为子进程（参数 --kill-child），子进程没有窗口。
"""
from __future__ import annotations

import gc
import json
import os
import platform
import random
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from splender.doc import storage  # noqa: E402
from splender.doc.storage import (FORMAT_VERSION, ProjectClosedError, ProjectCorruptError,  # noqa: E402
                                  ProjectFile, ProjectFileError, ProjectVersionError)

TILE = 256
CJK = re.compile("[\u4e00-\u9fff]")


# ---------------------------------------------------------------- 测试数据

def random_page(seed, nbytes) -> bytes:
    return np.random.default_rng(seed).bytes(nbytes)


def smooth_page(seed, bpp=4) -> np.ndarray:
    """256×256 的平滑渐变加几个柔边圆斑加少量噪点，形状 (256, 256, bpp) uint8。"""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:TILE, 0:TILE].astype(np.float32)
    img = np.empty((TILE, TILE, bpp), np.float32)
    for c in range(bpp):
        img[..., c] = 0.5 + 0.3 * np.sin(xs * rng.uniform(0.005, 0.02) + c) * np.cos(ys * rng.uniform(0.005, 0.02))
    for _ in range(4):
        cx, cy, r = rng.uniform(0, TILE), rng.uniform(0, TILE), rng.uniform(20, 120)
        a = np.clip(1 - np.hypot(xs - cx, ys - cy) / r, 0, 1) ** 1.5
        img += (rng.uniform(0, 1, bpp) - img) * a[..., None]
    v = img * 255 + rng.integers(-1, 2, img.shape) * (rng.random(img.shape) < 0.2)
    return np.clip(v + 0.5, 0, 255).astype(np.uint8)


def pattern_page(gen, index, nbytes=65536) -> bytes:
    """内容由 (代, 序号) 决定的页：前 8 字节是代和序号，后面是可压缩的周期数据。"""
    body = ((np.arange(nbytes - 8, dtype=np.uint32) * (2 * gen + 1) + index * 13) & 0xFF).astype(np.uint8)
    return struct.pack("<II", gen, index) + body.tobytes()


def kill_page(gen, index, nbytes) -> bytes:
    """强杀测试用：不可压缩的随机页，开头 8 字节是代和序号。"""
    return struct.pack("<II", gen, index) + np.random.default_rng([gen, index]).bytes(nbytes - 8)


def kill_keys(n):
    return [(7, 0, 0, i % 64, i // 64) for i in range(n)]


class PaintedCanvas:
    """「像真实绘制」的 16K 画布：低分辨率上算平滑渐变和柔边圆斑（含擦除），按页双线性放大，再加少量噪点。

    噪点：25% 的像素 RGB 加 ±1 的颗粒，1% 的像素加 ±12 以内的斑点。覆盖度通道大部分是 255，擦除处变低。
    """

    def __init__(self, size=16384, seed=1, low=2048, circles=400):
        import cv2  # 只有性能测试用到
        self._cv2 = cv2
        self.size = size
        self.scale = size / low
        rng = np.random.default_rng(seed)
        ys = (np.arange(low, dtype=np.float32)[:, None] + 0.5) * self.scale
        xs = (np.arange(low, dtype=np.float32)[None, :] + 0.5) * self.scale
        img = np.empty((low, low, 4), np.float32)
        img[..., 0] = 0.45 + 0.25 * np.sin(xs * 0.00031 + 0.7) + 0.15 * np.cos(ys * 0.00023)
        img[..., 1] = 0.40 + 0.20 * np.sin(ys * 0.00027 + 1.9) + 0.10 * np.cos((xs + ys) * 0.00017)
        img[..., 2] = 0.35 + 0.20 * np.cos(xs * 0.00019 - ys * 0.00011)
        img[..., 3] = 1.0
        for _ in range(circles):
            cx, cy = rng.uniform(0, size, 2)
            r = float(rng.uniform(80, 1600))
            hard = float(rng.uniform(0.0, 0.9))
            col = rng.uniform(0.05, 0.95, 3).astype(np.float32)
            erase = rng.random() < 0.12
            lx0, lx1 = max(0, int((cx - r) / self.scale) - 1), min(low, int((cx + r) / self.scale) + 2)
            ly0, ly1 = max(0, int((cy - r) / self.scale) - 1), min(low, int((cy + r) / self.scale) + 2)
            if lx0 >= lx1 or ly0 >= ly1:
                continue
            sub = img[ly0:ly1, lx0:lx1]
            d = np.sqrt((xs[:, lx0:lx1] - cx) ** 2 + (ys[ly0:ly1] - cy) ** 2) / r
            a = np.clip((1.0 - d) / max(1e-3, 1.0 - hard), 0.0, 1.0)
            a = a * a * (3 - 2 * a)
            if erase:
                sub[..., 3] *= 1.0 - a * 0.85
            else:
                sub[..., :3] += (col - sub[..., :3]) * a[..., None]
        self.img = img * 255.0
        self.noise = []
        for _ in range(16):
            grain = rng.integers(-1, 2, size=(TILE, TILE, 1)) * (rng.random((TILE, TILE, 1)) < 0.25)
            speck = rng.integers(-12, 13, size=(TILE, TILE, 1)) * (rng.random((TILE, TILE, 1)) < 0.01)
            n = np.zeros((TILE, TILE, 4), np.float32)
            n[..., :3] = grain + speck
            self.noise.append(n + 0.5)

    def page(self, tx, ty) -> np.ndarray:
        cv2 = self._cv2
        s = 1.0 / self.scale
        m = np.array([[s, 0, (tx * TILE + 0.5) * s - 0.5], [0, s, (ty * TILE + 0.5) * s - 0.5]], np.float32)
        p = cv2.warpAffine(self.img, m, (TILE, TILE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REPLICATE)
        p += self.noise[(tx * 7 + ty * 13) % len(self.noise)]
        np.clip(p, 0, 255, out=p)
        return p.astype(np.uint8)


def files_in(folder):
    return sorted(os.listdir(folder))


def raw_rows(path, sql, params=()):
    """绕过 ProjectFile 直接查文件（只读），用完即关。"""
    conn = sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def raw_exec(path, *statements):
    """绕过 ProjectFile 直接改文件，用来制造损坏和版本不符。"""
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        for sql in statements:
            if isinstance(sql, tuple):
                conn.execute(*sql)
            else:
                conn.execute(sql)
    finally:
        conn.close()


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="splender_storage_")
        self._open: list[ProjectFile] = []

    def tearDown(self):
        for pf in self._open:
            pf.close()
        gc.collect()
        shutil.rmtree(self.dir, ignore_errors=True)

    def path(self, name="p.splender", sub=None):
        folder = self.dir if sub is None else os.path.join(self.dir, sub)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, name)

    def open(self, path, create=False, **kw) -> ProjectFile:
        pf = ProjectFile(path, create=create, **kw)
        self._open.append(pf)
        return pf

    def assertChinese(self, exc):
        self.assertRegex(str(exc), CJK, f"错误说明应是中文：{exc}")


# ---------------------------------------------------------------- 功能测试

class RoundTripTests(TempDirCase):
    def test_pages_roundtrip_all_sizes_and_kinds(self):
        items = {}
        idx = 0
        for bpp in (1, 2, 4):
            n = TILE * TILE * bpp
            kinds = {
                "random": random_page(100 + bpp, n),
                "zeros": bytes(n),
                "const": bytes([0x5A]) * n,
                "smooth": smooth_page(bpp, bpp).tobytes(),
                "bytearray": bytearray(random_page(200 + bpp, n)),
            }
            for kind, data in kinds.items():
                items[(1000 + bpp, bpp, idx % 7, idx % 64, (idx * 5) % 64)] = data
                idx += 1
        # numpy 数组直接当数据：RGBA8、RG16F（float16）、R8
        arr_rgba = smooth_page(9, 4)
        arr_rg16f = np.random.default_rng(3).standard_normal((TILE, TILE, 2)).astype(np.float16)
        arr_r8 = smooth_page(10, 1)[..., 0]
        items[(5, 0, 0, 1, 1)] = arr_rgba
        items[(5, 2, 0, 1, 1)] = arr_rg16f
        items[(5, 3, 0, 1, 1)] = arr_r8
        items[(5, 3, 0, 2, 1)] = memoryview(random_page(11, TILE * TILE))
        items[(2 ** 40, 4, 6, 63, 63)] = bytes(TILE * TILE * 4)  # 大编号、最大坐标

        path = self.path()
        pf = self.open(path, create=True)
        pf.write_pages(items.items())

        def expect(v):
            return v.tobytes() if isinstance(v, (np.ndarray, memoryview)) else bytes(v)

        for k, v in items.items():
            self.assertEqual(pf.read_page(k), expect(v), k)
        got = pf.read_pages(list(items))
        self.assertEqual(set(got), set(items))
        for k, v in items.items():
            self.assertEqual(got[k], expect(v), k)
        self.assertEqual(pf.page_keys(), sorted(items))

        # 不可压缩的页原样存放，全零、全同值的页压缩后极小
        stored = {tuple(r[:5]): (r[5], r[6]) for r in
                  raw_rows(path, "SELECT layer_uid, plane, mip, tx, ty, codec, LENGTH(data) FROM pages")}
        for k, v in items.items():
            codec, size = stored[k]
            raw = expect(v)
            if raw == bytes(len(raw)) or raw == raw[:1] * len(raw):
                self.assertEqual(codec, storage.CODEC_ZSTD)
                self.assertLess(size, 64, f"全同值页压缩后应极小：{k} {size}")
            if isinstance(v, (bytes, bytearray)) and k[0] >= 1000 and len(set(raw[:4096])) > 200:
                self.assertEqual(codec, storage.CODEC_RAW, f"随机页应原样存放：{k}")
                self.assertEqual(size, len(raw))

        # 关闭重开后一致
        pf.close()
        pf2 = self.open(path)
        got = pf2.read_pages(items)
        for k, v in items.items():
            self.assertEqual(got[k], expect(v), k)

    def test_missing_keys_overwrite_and_key_types(self):
        pf = self.open(self.path(), create=True)
        a, b = random_page(1, 65536), random_page(2, 65536)
        pf.write_pages([((1, 0, 0, 0, 0), a)])
        self.assertIsNone(pf.read_page((1, 0, 0, 9, 9)))
        self.assertEqual(pf.read_pages([(1, 0, 0, 0, 0), (1, 0, 0, 9, 9)]), {(1, 0, 0, 0, 0): a})
        self.assertEqual(pf.read_pages([]), {})
        # 覆盖；同一次调用里重复的键以最后一个为准
        pf.write_pages([((1, 0, 0, 0, 0), b), ((1, 0, 0, 0, 1), a), ((1, 0, 0, 0, 1), b)])
        self.assertEqual(pf.read_page((1, 0, 0, 0, 0)), b)
        self.assertEqual(pf.read_page((1, 0, 0, 0, 1)), b)
        # numpy 整数作键可以，浮点不行
        key_np = tuple(np.array([1, 0, 0, 0, 0], dtype=np.int64))
        self.assertEqual(pf.read_page(key_np), b)
        pf.write_pages([((np.int32(3), np.uint8(1), 0, np.int64(2), 5), a)])
        self.assertEqual(pf.read_page((3, 1, 0, 2, 5)), a)
        with self.assertRaises(TypeError):
            pf.read_page((1.0, 0, 0, 0, 0))
        with self.assertRaises(TypeError):
            pf.write_pages([((1, 0, 0), a)])
        with self.assertRaises(TypeError):
            pf.write_pages([((1, 0, 0, 0, 7), "不是字节")])
        self.assertIsNone(pf.read_page((1, 0, 0, 0, 7)))

    def test_single_and_multi_thread_identical(self):
        items = [((1, p, 0, i % 8, i // 8), smooth_page(i, 4).tobytes() if i % 3 else random_page(i, 262144))
                 for p in range(2) for i in range(40)]
        results = []
        for threads in (1, 8):
            pf = self.open(self.path(f"t{threads}.splender"), create=True, threads=threads, batch_pages=7)
            pf.write_pages(items)
            results.append((pf.read_pages(k for k, _ in items), pf.stats()["compressed_bytes"]))
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(results[0][1], results[1][1])
        self.assertEqual(results[0][0], dict(items))


class MetaAssetTests(TempDirCase):
    def test_meta_roundtrip_and_replace(self):
        path = self.path()
        pf = self.open(path, create=True)
        self.assertEqual(pf.read_meta(), {})
        meta = {"名称": "测试工程", "resolution": 16384, "opacity": 0.75, "visible": True, "parent": None,
                "layers": [{"uid": 1, "name": "底色", "blend": "NORMAL"}, {"uid": 2, "name": "划痕"}],
                "nested": {"a": [1, 2, {"b": "ç"}]}}
        pf.write_meta(meta)
        self.assertEqual(pf.read_meta(), meta)
        pf.write_meta({"only": 1})
        self.assertEqual(pf.read_meta(), {"only": 1})
        with self.assertRaises(TypeError):
            pf.write_meta({1: "键不是字符串"})
        with self.assertRaises(TypeError):
            pf.write_meta({"bad": object()})
        self.assertEqual(pf.read_meta(), {"only": 1})
        pf.close()
        self.assertEqual(self.open(path).read_meta(), {"only": 1})

    def test_assets_including_50mb(self):
        path = self.path()
        pf = self.open(path, create=True)
        small = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        big = np.random.default_rng(42).bytes(50 * 1024 * 1024)
        t0 = time.perf_counter()
        pf.write_asset("model/原始.obj", small)
        pf.write_asset("model/big.glb", big)
        t1 = time.perf_counter()
        got = pf.read_asset("model/big.glb")
        t2 = time.perf_counter()
        self.assertEqual(got, big)
        self.assertEqual(pf.read_asset("model/原始.obj"), small)
        self.assertIsNone(pf.read_asset("没有这个"))
        self.assertEqual(pf.list_assets(), sorted(["model/原始.obj", "model/big.glb"]))
        pf.write_asset("model/原始.obj", memoryview(b"replaced"))
        self.assertEqual(pf.read_asset("model/原始.obj"), b"replaced")
        self.assertTrue(pf.delete_asset("model/原始.obj"))
        self.assertFalse(pf.delete_asset("model/原始.obj"))
        with self.assertRaises(TypeError):
            pf.write_asset("", b"x")
        st = pf.stats()
        self.assertEqual(st["asset_count"], 1)
        self.assertEqual(st["asset_bytes"], len(big))
        pf.close()
        pf2 = self.open(path)
        t3 = time.perf_counter()
        self.assertEqual(pf2.read_asset("model/big.glb"), big)
        t4 = time.perf_counter()
        self.assertEqual(pf2.list_assets(), ["model/big.glb"])
        print(f"\n  50 MiB 资源：写 {(t1 - t0) * 1000:.0f} ms，读 {(t2 - t1) * 1000:.0f} ms，"
              f"重开后读 {(t4 - t3) * 1000:.0f} ms")


class TransactionTests(TempDirCase):
    N = 200

    def keys(self):
        return [(1, 0, 0, i % 64, i // 64) for i in range(self.N)]

    def state(self, pf):
        return pf.read_pages(pf.page_keys()), pf.read_meta()

    def test_failure_midway_rolls_back_everything(self):
        path = self.path()
        pf = self.open(path, create=True, batch_pages=16)  # 小批量：失败前已有多批写进事务
        keys = self.keys()
        pf.write_pages((k, pattern_page(1, i)) for i, k in enumerate(keys))
        pf.write_meta({"gen": 1})
        before = self.state(pf)
        extra = (2, 0, 0, 0, 0)

        def failing():
            for i, k in enumerate(keys):
                if i == 150:
                    raise RuntimeError("故意中断")
                yield k, pattern_page(2, i)
            yield extra, b"x"

        with self.assertRaises(RuntimeError):
            pf.write_pages(failing(), delete=keys[:20])
        self.assertEqual(self.state(pf), before)

        # 第 100 项数据非法：一次在主线程分批时发现，一次在线程池里压缩时才发现
        class NotABuffer:
            nbytes = 65536

        for bad_data in (None, NotABuffer()):
            bad = [(k, pattern_page(3, i)) for i, k in enumerate(keys)]
            bad[100] = (keys[100], bad_data)
            with self.assertRaises(TypeError):
                pf.write_pages(bad, delete=keys[:5])
            self.assertEqual(self.state(pf), before)

        # 第 120 项键非法
        bad = [(k, pattern_page(3, i)) for i, k in enumerate(keys)]
        bad[120] = ((1, 0, 0.5, 0, 0), pattern_page(3, 120))
        with self.assertRaises(TypeError):
            pf.write_pages(bad, delete=keys[50:60])
        self.assertEqual(self.state(pf), before)

        # transaction()：页面和工程信息一起撤销
        with self.assertRaises(ValueError):
            with pf.transaction():
                pf.write_pages((k, pattern_page(4, i)) for i, k in enumerate(keys))
                pf.write_meta({"gen": 4})
                pf.write_asset("a", b"1")
                pf.delete_layer(1)
                raise ValueError("故意中断")
        self.assertEqual(self.state(pf), before)
        self.assertEqual(pf.list_assets(), [])

        # 关闭重开，仍是上一次提交的状态
        pf.close()
        pf = self.open(path)
        self.assertEqual(self.state(pf), before)

        # transaction() 成功则一起提交；块内单个写操作失败被捕获时，只撤销它自己
        with pf.transaction():
            pf.write_pages((k, pattern_page(5, i)) for i, k in enumerate(keys))
            try:
                pf.write_pages([((9, 0, 0, 0, 0), b"y"), ((9, 0, 0, 0, 1), None)])
            except TypeError:
                pass
            pf.write_meta({"gen": 5})
        pages, meta = self.state(pf)
        self.assertEqual(meta, {"gen": 5})
        self.assertEqual(set(pages), set(keys))
        self.assertTrue(all(pages[k] == pattern_page(5, i) for i, k in enumerate(keys)))


class ConcurrencyTests(TempDirCase):
    def test_background_reads_while_main_thread_writes(self):
        n_keys, gens = 96, 40
        keys = [(3, 0, 0, i % 16, i // 16) for i in range(n_keys)]
        path = self.path()
        pf = self.open(path, create=True, batch_pages=8)  # 每次写 12 批，读线程有机会落在事务中间
        pf.write_pages((k, pattern_page(0, i)) for i, k in enumerate(keys))
        index = {k: i for i, k in enumerate(keys)}
        stop = threading.Event()
        errors: list[str] = []
        counts = {"read_pages": 0, "read_page": 0, "copies": 0}
        seen_gens: set[int] = set()
        lock = threading.Lock()

        def check_page(k, data):
            g, i = struct.unpack_from("<II", data)
            if i != index[k] or data != pattern_page(g, i):
                raise AssertionError(f"页面 {k} 内容不完整或错位（代 {g}）")
            return g

        def reader(seed):
            rnd = random.Random(seed)
            last = 0
            try:
                while not stop.is_set():
                    got = pf.read_pages(keys)
                    if len(got) != n_keys:
                        raise AssertionError(f"read_pages 少了页：{len(got)}")
                    g_set = {check_page(k, v) for k, v in got.items()}
                    if len(g_set) != 1:
                        raise AssertionError(f"一次 read_pages 混入了不同提交的内容：{sorted(g_set)}")
                    g = g_set.pop()
                    if g < last:
                        raise AssertionError(f"读到的提交倒退了：{g} < {last}")
                    last = g
                    k = rnd.choice(keys)
                    g2 = check_page(k, pf.read_page(k))
                    if g2 < last:
                        raise AssertionError(f"read_page 读到的提交倒退了：{g2} < {last}")
                    last = g2
                    if rnd.random() < 0.2:
                        pf.stats()
                        if len(pf.page_keys()) != n_keys:
                            raise AssertionError("page_keys 数量不对")
                        pf.read_meta()
                    with lock:
                        counts["read_pages"] += 1
                        counts["read_page"] += 1
                        seen_gens.add(g)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"读线程 {seed}：{exc!r}")

        def copier():
            try:
                i = 0
                while not stop.is_set():
                    dst = self.path(f"copy{i % 2}.splender", sub="copies")
                    pf.save_copy(dst, progress=(lambda f: None) if i % 2 else None, step_pages=4)
                    with ProjectFile(dst) as cp:
                        got = cp.read_pages(keys)
                        g_set = {check_page(k, v) for k, v in got.items()}
                        if len(got) != n_keys or len(g_set) != 1:
                            raise AssertionError(f"另存的副本不是一致的快照：{len(got)} 页，代 {sorted(g_set)}")
                    i += 1
                    with lock:
                        counts["copies"] += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"另存线程：{exc!r}")

        threads = [threading.Thread(target=reader, args=(s,)) for s in range(4)]
        threads.append(threading.Thread(target=copier))
        for t in threads:
            t.start()
        try:
            for g in range(1, gens + 1):
                with pf.transaction():
                    pf.write_pages((k, pattern_page(g, i)) for i, k in enumerate(keys))
                    pf.write_meta({"gen": g})
                time.sleep(0.002)
        finally:
            time.sleep(0.05)
            stop.set()
            for t in threads:
                t.join(60)
        self.assertEqual(errors, [])
        self.assertGreater(counts["read_pages"], 20)
        self.assertGreater(counts["copies"], 1)
        self.assertGreater(len(seen_gens), 3, f"读线程看到的提交太少，说明没有和写交错：{sorted(seen_gens)}")
        final = pf.read_pages(keys)
        self.assertTrue(all(v == pattern_page(gens, index[k]) for k, v in final.items()))
        print(f"\n  并发：{counts['read_pages']} 次 read_pages、{counts['read_page']} 次 read_page、"
              f"{counts['copies']} 次另存，读到了 {len(seen_gens)} 个不同的提交")


class FileLifecycleTests(TempDirCase):
    def test_close_leaves_only_the_project_file(self):
        path = self.path()
        pf = self.open(path, create=True)
        self.assertEqual(pf.journal_mode, "wal")
        keys = [(1, 0, 0, i, 0) for i in range(50)]
        pf.write_pages((k, smooth_page(i).tobytes()) for i, k in enumerate(keys))
        pf.write_meta({"a": 1})
        pf.write_asset("m", b"123")
        self.assertIn("p.splender-wal", files_in(self.dir))  # 打开期间确实在用 WAL

        def bg():
            pf.read_pages(keys)
            pf.read_page(keys[3])
            pf.stats()

        ts = [threading.Thread(target=bg) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        copy_path = self.path("copy.splender", sub="out")
        pf.save_copy(copy_path)
        pf.delete_layer(1)
        pf.vacuum()
        pf.close()
        pf.close()  # 重复关闭无害
        self.assertEqual(files_in(self.dir), ["out", "p.splender"])
        self.assertEqual(files_in(os.path.dirname(copy_path)), ["copy.splender"])
        with ProjectFile(copy_path) as cp:
            self.assertEqual(len(cp.page_keys()), 50)
        self.assertEqual(files_in(os.path.dirname(copy_path)), ["copy.splender"])
        # 只打开不写，关闭后也不留东西
        with ProjectFile(path) as again:
            again.read_meta()
        self.assertEqual(files_in(self.dir), ["out", "p.splender"])

    def test_closed_file_raises(self):
        pf = self.open(self.path(), create=True)
        pf.write_pages([((1, 0, 0, 0, 0), b"abc")])
        pf.close()
        self.assertTrue(pf.closed)
        for call in (lambda: pf.read_page((1, 0, 0, 0, 0)), lambda: pf.read_pages([(1, 0, 0, 0, 0)]),
                     lambda: pf.write_pages([]), lambda: pf.read_meta(), lambda: pf.stats(),
                     lambda: pf.page_keys(), lambda: pf.delete_layer(1), lambda: pf.vacuum()):
            with self.assertRaises(ProjectClosedError) as cm:
                call()
            self.assertIsInstance(cm.exception, ProjectFileError)
            self.assertChinese(cm.exception)

    def test_create_semantics_and_stale_sidecars(self):
        path = self.path()
        # 同名的旧 -wal（来自别的文件）不能被当成新文件的日志
        with open(path + "-wal", "wb") as fh:
            fh.write(os.urandom(4096))
        pf = self.open(path, create=True)
        pf.write_pages([((1, 0, 0, 0, 0), b"new")])
        pf.close()
        self.assertEqual(files_in(self.dir), ["p.splender"])
        # create=True 打开已有工程：内容不动
        pf = self.open(path, create=True)
        self.assertEqual(pf.read_page((1, 0, 0, 0, 0)), b"new")
        pf.close()
        # 找不到文件
        with self.assertRaises(ProjectFileError) as cm:
            ProjectFile(self.path("nope.splender"))
        self.assertNotIsInstance(cm.exception, ProjectCorruptError)
        self.assertChinese(cm.exception)
        self.assertFalse(os.path.exists(self.path("nope.splender")))
        # 0 字节文件：create=False 报错，create=True 当新文件
        empty = self.path("empty.splender")
        open(empty, "wb").close()
        with self.assertRaises(ProjectCorruptError):
            ProjectFile(empty)
        with ProjectFile(empty, create=True) as pf:
            pf.write_meta({"x": 1})
        with ProjectFile(empty) as pf:
            self.assertEqual(pf.read_meta(), {"x": 1})


class DeleteTests(TempDirCase):
    def test_delete_and_delete_layer(self):
        pf = self.open(self.path(), create=True)
        items = {(layer, plane, 0, x, 0): pattern_page(layer, x, 1024)
                 for layer in (1, 2, 3) for plane in (0, 1) for x in range(5)}
        pf.write_pages(items)
        # 删除和写入同一次提交；两边都有的键最后是写入的内容
        both = (1, 0, 0, 4, 0)
        pf.write_pages([((1, 0, 0, 9, 9), b"new"), (both, b"rewritten")],
                       delete=[(1, 0, 0, 0, 0), (1, 1, 0, 0, 0), (8, 8, 8, 8, 8), both])
        self.assertIsNone(pf.read_page((1, 0, 0, 0, 0)))
        self.assertIsNone(pf.read_page((1, 1, 0, 0, 0)))
        self.assertEqual(pf.read_page((1, 0, 0, 9, 9)), b"new")
        self.assertEqual(pf.read_page(both), b"rewritten")
        pf.delete_pages([(1, 0, 0, 1, 0)])
        self.assertIsNone(pf.read_page((1, 0, 0, 1, 0)))
        self.assertEqual(pf.delete_layer(2), 10)
        self.assertEqual(pf.delete_layer(2), 0)
        self.assertEqual(pf.page_keys(2), [])
        self.assertEqual(len(pf.page_keys(3)), 10)
        self.assertTrue(all(k[0] in (1, 3) for k in pf.page_keys()))
        for k in pf.page_keys(3):
            self.assertEqual(pf.read_page(k), items[k])

    def test_stats_and_vacuum(self):
        path = self.path()
        pf = self.open(path, create=True)
        items = {(1, 0, 0, i % 32, i // 32): smooth_page(i).tobytes() for i in range(300)}
        pf.write_pages(items)
        st = pf.stats()
        self.assertEqual(st["page_count"], 300)
        self.assertEqual(st["raw_bytes"], 300 * TILE * TILE * 4)
        self.assertLess(st["compressed_bytes"], st["raw_bytes"])
        self.assertAlmostEqual(st["compression_ratio"], st["raw_bytes"] / st["compressed_bytes"])
        self.assertGreater(st["file_bytes"] + st["wal_bytes"], st["compressed_bytes"])
        pf.write_pages((), delete=[k for i, k in enumerate(items) if i % 3])
        pf.vacuum()
        st2 = pf.stats()
        self.assertEqual(st2["page_count"], 100)
        self.assertLess(st2["file_bytes"], st["file_bytes"] + st["wal_bytes"])
        self.assertEqual(st2["free_bytes"], 0)
        self.assertEqual(pf.check(), [])
        self.assertEqual(pf.check(full=True), [])
        for i, (k, v) in enumerate(items.items()):
            if i % 3 == 0:
                self.assertEqual(pf.read_page(k), v)


class SaveCopyTests(TempDirCase):
    def test_save_copy(self):
        src = self.path()
        pf = self.open(src, create=True)
        items = {(1, 0, 0, i, 0): smooth_page(i).tobytes() for i in range(30)}
        pf.write_pages(items)
        pf.write_meta({"name": "原件"})
        pf.write_asset("model.obj", b"obj data")
        dst = self.path("另存.splender", sub="out")
        # 目标已存在（另一个工程）时被替换
        with ProjectFile(dst, create=True) as old:
            old.write_meta({"name": "旧文件"})
            old.write_pages([((99, 0, 0, 0, 0), b"old")])
        steps = []
        pf.save_copy(dst, progress=steps.append, step_pages=8)
        self.assertGreater(len(steps), 2)
        self.assertEqual(steps[-1], 1.0)
        self.assertEqual(steps, sorted(steps))
        self.assertEqual(files_in(os.path.dirname(dst)), ["另存.splender"])
        with ProjectFile(dst) as cp:
            self.assertEqual(cp.read_meta(), {"name": "原件"})
            self.assertEqual(cp.read_pages(cp.page_keys()), items)
            self.assertEqual(cp.read_asset("model.obj"), b"obj data")
        # 原文件照常可用
        pf.write_pages([((1, 0, 0, 0, 0), b"changed")])
        self.assertEqual(pf.read_page((1, 0, 0, 0, 0)), b"changed")
        # 一步复制
        dst2 = self.path("b.splender", sub="out2")
        pf.save_copy(dst2)
        with ProjectFile(dst2) as cp:
            self.assertEqual(cp.read_page((1, 0, 0, 0, 0)), b"changed")
        self.assertEqual(files_in(os.path.dirname(dst2)), ["b.splender"])
        with self.assertRaises(ValueError):
            pf.save_copy(src)
        with self.assertRaises(ProjectFileError):
            pf.save_copy(os.path.join(self.dir, "不存在的文件夹", "x.splender"))
        # 目标正被打开时不能覆盖，且不留临时文件
        busy = self.path("busy.splender", sub="out3")
        holder = self.open(busy, create=True)
        holder.write_meta({"keep": True})
        with self.assertRaises(ProjectFileError):
            pf.save_copy(busy)
        self.assertEqual(holder.read_meta(), {"keep": True})
        self.assertFalse([f for f in files_in(os.path.dirname(busy)) if f.endswith(".tmp")])


class VersionAndCorruptionTests(TempDirCase):
    def make_project(self, name="p.splender", n=40):
        path = self.path(name)
        with ProjectFile(path, create=True) as pf:
            pf.write_pages(((1, 0, 0, i % 8, i // 8), smooth_page(i).tobytes()) for i in range(n))
            pf.write_meta({"name": "x"})
        return path

    def test_newer_version_is_rejected(self):
        path = self.make_project()
        raw_exec(path, f"PRAGMA user_version={FORMAT_VERSION + 1}")
        with self.assertRaises(ProjectVersionError) as cm:
            ProjectFile(path)
        e = cm.exception
        self.assertIsInstance(e, ProjectFileError)
        self.assertEqual(e.file_version, FORMAT_VERSION + 1)
        self.assertEqual(e.supported_version, FORMAT_VERSION)
        self.assertIn(str(FORMAT_VERSION + 1), str(e))
        self.assertChinese(e)
        self.assertEqual(files_in(self.dir), ["p.splender"])
        raw_exec(path, "PRAGMA user_version=0")
        with self.assertRaises(ProjectCorruptError):
            ProjectFile(path)
        raw_exec(path, f"PRAGMA user_version={FORMAT_VERSION}", "PRAGMA application_id=12345")
        with self.assertRaises(ProjectCorruptError):
            ProjectFile(path)

    def test_not_a_project(self):
        cases = {
            "random.splender": os.urandom(100_000),
            "text.splender": "这不是工程文件\nhello\n".encode("utf-8"),
            "short.splender": b"SQLite format 3\x00",
        }
        try:
            from PIL import Image
            import io
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, "PNG")
            cases["image.splender"] = buf.getvalue()
        except ImportError:
            pass
        for name, data in cases.items():
            p = self.path(name)
            with open(p, "wb") as fh:
                fh.write(data)
            for create in (False, True):
                with self.assertRaises(ProjectCorruptError, msg=name) as cm:
                    ProjectFile(p, create=create)
                self.assertChinese(cm.exception)
                with open(p, "rb") as fh:
                    self.assertEqual(fh.read(), data, f"打开失败不能改动原文件：{name}")
        # 别的程序的 SQLite 数据库：不认，也不覆盖
        other = self.path("other.splender")
        raw_exec(other, "CREATE TABLE foo (a INTEGER)", "INSERT INTO foo VALUES (1)")
        for create in (False, True):
            with self.assertRaises(ProjectCorruptError):
                ProjectFile(other, create=create)
        self.assertEqual(raw_rows(other, "SELECT a FROM foo"), [(1,)])
        # 少了数据表
        broken = self.make_project("broken.splender")
        raw_exec(broken, "DROP TABLE pages")
        with self.assertRaises(ProjectCorruptError):
            ProjectFile(broken)
        # 文件夹
        with self.assertRaises(ProjectFileError):
            ProjectFile(self.dir)
        leftovers = [f for f in files_in(self.dir) if f.endswith(("-wal", "-shm", "-journal"))]
        self.assertEqual(leftovers, [])

    def test_damaged_header_and_truncation(self):
        path = self.make_project("hdr.splender")
        with open(path, "r+b") as fh:
            fh.write(os.urandom(100))
        with self.assertRaises(ProjectCorruptError):
            ProjectFile(path)

        path = self.make_project("trunc.splender", n=300)
        size = os.path.getsize(path)
        with open(path, "r+b") as fh:
            fh.truncate(size // 2)
        corrupt_seen = False
        try:
            pf = self.open(path)
        except ProjectCorruptError:
            corrupt_seen = True
        else:
            for op in (lambda: pf.read_pages(pf.page_keys()),
                       lambda: [pf.read_page((1, 0, 0, i % 8, i // 8)) for i in range(300)],
                       lambda: pf.stats()):
                try:
                    op()
                except ProjectCorruptError as exc:
                    self.assertChinese(exc)
                    corrupt_seen = True
            self.assertTrue(pf.check(full=True), "截断的文件检查应报告问题")
        self.assertTrue(corrupt_seen, "截断一半的文件应在打开或读取时报损坏")

    def test_damaged_page_records(self):
        path = self.make_project("rec.splender")
        k1, k2, k3, k4 = (1, 0, 0, 0, 0), (1, 0, 0, 1, 0), (1, 0, 0, 2, 0), (1, 0, 0, 3, 0)
        blob = raw_rows(path, "SELECT data FROM pages WHERE layer_uid=1 AND tx=0 AND ty=0")[0][0]
        flipped = bytearray(blob)
        flipped[len(flipped) // 2] ^= 0xFF
        raw_exec(path,
                 ("UPDATE pages SET data=? WHERE layer_uid=1 AND tx=0 AND ty=0", (bytes(flipped),)),
                 "UPDATE pages SET codec=99 WHERE layer_uid=1 AND tx=1 AND ty=0",
                 "UPDATE pages SET raw_len=raw_len+1 WHERE layer_uid=1 AND tx=2 AND ty=0",
                 "UPDATE pages SET data='文字' WHERE layer_uid=1 AND tx=3 AND ty=0",
                 "UPDATE meta SET value='{不是 JSON' WHERE key='name'")
        pf = self.open(path)
        for k in (k1, k2, k3, k4):
            with self.assertRaises(ProjectCorruptError, msg=str(k)) as cm:
                pf.read_page(k)
            self.assertChinese(cm.exception)
        with self.assertRaises(ProjectCorruptError):
            pf.read_pages(pf.page_keys())
        with self.assertRaises(ProjectCorruptError):
            pf.read_meta()
        # 其他页不受影响，文件仍可写
        self.assertEqual(pf.read_page((1, 0, 0, 4, 0)), smooth_page(4).tobytes())
        pf.write_pages([(k1, b"fixed")])
        self.assertEqual(pf.read_page(k1), b"fixed")


# ---------------------------------------------------------------- 强杀测试

KILL_PAGES = 256
KILL_PAGE_BYTES = 65536


def kill_child_main(path: str) -> None:
    """子进程：不停地提交新的一代（页面加工程信息一个事务），每提交一代打印代号。"""
    pf = ProjectFile(path, threads=2, batch_pages=16)
    gen = int(pf.read_meta().get("gen", 0))
    print("ready", flush=True)
    keys = kill_keys(KILL_PAGES)
    while True:
        gen += 1
        print(f"b{gen}", flush=True)  # 开始写这一代
        with pf.transaction():
            pf.write_pages((k, kill_page(gen, i, KILL_PAGE_BYTES)) for i, k in enumerate(keys))
            pf.write_meta({"gen": gen})
        print(gen, flush=True)


class KillTests(TempDirCase):
    def test_kill_during_writes_keeps_consistent_state(self):
        path = self.path()
        keys = kill_keys(KILL_PAGES)
        with ProjectFile(path, create=True) as pf:
            with pf.transaction():
                pf.write_pages((k, kill_page(0, i, KILL_PAGE_BYTES)) for i, k in enumerate(keys))
                pf.write_meta({"gen": 0})
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        report = []
        mid_transaction_kills = 0
        for trial, delay in enumerate((0.0, 0.15, 0.4, 0.9, 1.6, 0.25, 0.6)):
            proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--kill-child", path],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    encoding="utf-8", creationflags=flags, env=env, cwd=str(ROOT))
            lines: list[str] = []
            ready = threading.Event()

            def pump(stream=proc.stdout):
                for line in stream:
                    lines.append(line.strip())
                    if line.strip() == "ready":
                        ready.set()

            th = threading.Thread(target=pump, daemon=True)
            th.start()
            if not ready.wait(60):
                proc.kill()
                self.fail("子进程没有就绪：" + proc.stderr.read())
            time.sleep(delay)
            if proc.poll() is not None:
                self.fail("子进程提前退出：" + proc.stderr.read())
            proc.kill()  # TerminateProcess：不做任何清理
            proc.wait(30)
            th.join(10)
            proc.stdout.close()
            proc.stderr.close()
            committed = [int(x) for x in lines if x.isdigit()]
            last_printed = committed[-1] if committed else None
            began = [int(x[1:]) for x in lines if x.startswith("b")]
            last_began = began[-1] if began else None
            had_wal = os.path.exists(path + "-wal")

            with ProjectFile(path) as pf:
                self.assertEqual(pf.check(full=True), [], f"第 {trial} 次强杀后文件结构损坏")
                meta_gen = pf.read_meta()["gen"]
                pages = pf.read_pages(keys)
                self.assertEqual(len(pages), KILL_PAGES)
                gens = {struct.unpack_from("<I", pages[k])[0] for k in keys}
                self.assertEqual(gens, {meta_gen}, f"第 {trial} 次强杀后页面混了几代：{sorted(gens)}，工程信息是 {meta_gen}")
                for i, k in enumerate(keys):
                    self.assertEqual(pages[k], kill_page(meta_gen, i, KILL_PAGE_BYTES))
                if last_printed is not None:
                    # 打印之前就已提交；最后一次提交可能刚完成还没来得及打印
                    self.assertIn(meta_gen, (last_printed, last_printed + 1))
            self.assertEqual(files_in(self.dir), ["p.splender"])
            # 开始写第 g 代却恢复出 g-1：说明是在事务中途被杀，半截写入被整体丢弃
            mid = last_began is not None and meta_gen == last_began - 1
            mid_transaction_kills += mid
            report.append(f"延迟 {delay}s：开始写到第 {last_began} 代，打印提交到第 {last_printed} 代，"
                          f"恢复出第 {meta_gen} 代，{'事务中途被杀' if mid else '提交后被杀'}，强杀后有 -wal：{had_wal}")
        print("\n  " + "\n  ".join(report))
        self.assertGreater(mid_transaction_kills, 0, "没有一次落在事务中途，测试没有覆盖到要测的情形")


# ---------------------------------------------------------------- 性能测试

@unittest.skipUnless(os.environ.get("SPLENDER_BENCH") == "1", "设 SPLENDER_BENCH=1 才跑性能测试")
class StorageBenchmark(unittest.TestCase):
    N = 4096

    def test_benchmark(self):
        import psutil
        proc = psutil.Process()
        out: dict = {
            "说明": "写入 4096 页 256×256 RGBA8（一整层 16K 颜色页），随机读 1000 页。临时文件在系统临时目录，测完删除。",
            "machine": {"cpu": platform.processor(), "logical_cpus": os.cpu_count(),
                        "temp_dir": tempfile.gettempdir()},
            "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
            "zstandard": __import__("zstandard").__version__,
        }
        t0 = time.perf_counter()
        canvas = PaintedCanvas()
        t1 = time.perf_counter()
        keys = [(1, 0, 0, i % 64, i // 64) for i in range(self.N)]
        pages = [canvas.page(k[3], k[4]).tobytes() for k in keys]
        t2 = time.perf_counter()
        raw_total = sum(len(p) for p in pages)
        out["data"] = {
            "内容": "平滑渐变 + 400 个柔边圆斑（12% 是擦除，降低覆盖度）+ 少量噪点（25% 像素 ±1 颗粒，1% 像素 ±12 斑点）",
            "pages": self.N, "page_bytes": TILE * TILE * 4, "raw_MiB": round(raw_total / 2**20, 1),
            "generate_seconds": round(t2 - t0, 2),
        }
        folder = tempfile.mkdtemp(prefix="splender_bench_")
        try:
            path = os.path.join(folder, "bench.splender")
            pf = ProjectFile(path, create=True)
            settings = {"level": pf.level, "threads": pf.threads, "batch_pages": 256, "page_size": 16384,
                        "synchronous": "FULL", "journal_mode": pf.journal_mode}

            peak = [proc.memory_info().rss]
            sampling = threading.Event()

            def sampler():
                while not sampling.is_set():
                    peak[0] = max(peak[0], proc.memory_info().rss)
                    time.sleep(0.005)

            base_rss = proc.memory_info().rss
            th = threading.Thread(target=sampler, daemon=True)
            th.start()
            w0 = time.perf_counter()
            pf.write_pages(zip(keys, pages))
            w1 = time.perf_counter()
            sampling.set()
            th.join()
            st = pf.stats()
            pf.close()
            file_bytes = os.path.getsize(path)
            out["settings"] = settings
            out["write"] = {
                "seconds": round(w1 - w0, 3),
                "raw_MiB_per_s": round(raw_total / 2**20 / (w1 - w0), 1),
                "pages_per_s": round(self.N / (w1 - w0), 1),
                "extra_peak_rss_MiB": round((peak[0] - base_rss) / 2**20, 1),
                "说明": "一次 write_pages 调用（一个事务），含压缩、写入、提交落盘和检查点。extra_peak_rss 是写入期间比开始时多占的内存峰值",
            }
            out["file"] = {
                "file_bytes": file_bytes, "file_MiB": round(file_bytes / 2**20, 1),
                "compressed_MiB": round(st["compressed_bytes"] / 2**20, 1),
                "compression_ratio": round(st["compression_ratio"], 2),
                "raw_over_file": round(raw_total / file_bytes, 2),
            }

            r0 = time.perf_counter()
            pf = ProjectFile(path)
            r1 = time.perf_counter()
            pk = pf.page_keys()
            r2 = time.perf_counter()
            pf.stats()
            r3 = time.perf_counter()
            rnd = random.Random(7)
            sample = rnd.sample(keys, 1000)
            b0 = time.perf_counter()
            got = pf.read_pages(sample)
            b1 = time.perf_counter()
            sample2 = rnd.sample(keys, 1000)
            l0 = time.perf_counter()
            for k in sample2:
                pf.read_page(k)
            l1 = time.perf_counter()
            self.assertEqual(len(got), 1000)
            index = {k: i for i, k in enumerate(keys)}
            for k in sample[:50]:
                self.assertEqual(got[k], pages[index[k]])
            self.assertEqual(len(pk), self.N)
            pf.close()
            out["read"] = {
                "open_ms": round((r1 - r0) * 1000, 1), "page_keys_ms": round((r2 - r1) * 1000, 1),
                "stats_ms": round((r3 - r2) * 1000, 1),
                "read_pages_1000_random_ms": round((b1 - b0) * 1000, 1),
                "read_pages_MiB_per_s": round(1000 * TILE * TILE * 4 / 2**20 / (b1 - b0), 1),
                "read_page_x1000_random_ms": round((l1 - l0) * 1000, 1),
                "说明": "关闭后重新打开再读（系统文件缓存是热的）。read_pages 一次取 1000 页并行解压；read_page 是逐页调用 1000 次",
            }

            # 对照：单线程压缩
            path1 = os.path.join(folder, "bench_1t.splender")
            with ProjectFile(path1, create=True, threads=1) as pf1:
                s0 = time.perf_counter()
                pf1.write_pages(zip(keys, pages))
                s1 = time.perf_counter()
            out["write_single_thread"] = {"seconds": round(s1 - s0, 3),
                                          "raw_MiB_per_s": round(raw_total / 2**20 / (s1 - s0), 1)}
            # 对照：全零页
            zero = bytes(TILE * TILE * 4)
            path0 = os.path.join(folder, "bench_zero.splender")
            with ProjectFile(path0, create=True) as pf0:
                z0 = time.perf_counter()
                pf0.write_pages((k, zero) for k in keys)
                z1 = time.perf_counter()
                zst = pf0.stats()
            out["zero_pages"] = {"seconds": round(z1 - z0, 3), "compressed_bytes": zst["compressed_bytes"],
                                 "file_bytes": os.path.getsize(path0)}
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        out["temp_files_removed"] = not os.path.exists(folder)
        evidence = ROOT / "docs" / "evidence" / "storage"
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "benchmark.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n" + json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--kill-child":
        kill_child_main(sys.argv[2])
    else:
        unittest.main()
