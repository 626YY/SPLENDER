"""页池与三级缓存的显卡测试。用隐藏上下文，不开窗口。"""
import sys
from pathlib import Path
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

from splender.engine import glx  # noqa: E402
from splender.engine.cache import LevelGrid, PageCache  # noqa: E402
from splender.engine.pagepool import PAGE, PoolSet  # noqa: E402


def pattern(seed: int, nbytes: int) -> bytes:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, 64, dtype=np.uint8)
    return np.resize(base, nbytes).tobytes()


class CacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = moderngl.create_standalone_context(require=430)
        cls.caps = glx.Caps()

    def make(self, budget_mb=300, ram_mb=1024):
        pools = PoolSet(self.ctx, budget_mb << 20)
        cache = PageCache(self.ctx, pools, ram_budget=ram_mb << 20, caps=self.caps)
        return pools, cache

    def test_caps(self):
        self.assertTrue(self.caps.sparse)
        self.assertTrue(self.caps.buffer_storage)
        self.assertTrue(self.caps.get_sub_image)
        self.assertGreater(self.caps.vram_total_mb, 1000)

    def test_roundtrip_and_async_eviction(self):
        pools, cache = self.make(budget_mb=300)           # 只够一张 rgba8 数组（1024 页）
        level = LevelGrid(64)
        nbytes = pools["rgba8"].page_bytes
        pages = []
        frame = 0
        for i in range(1500):
            page = cache.new_page("rgba8")
            pools["rgba8"].write(page.slot, pattern(i, nbytes))
            cache.set_page(level, i % 64, i // 64, page)
            pages.append(page)
            if i % 24 == 0:
                frame += 1
                glx.flush()
                cache.service(frame)
        for _ in range(200):
            frame += 1
            glx.flush()
            cache.service(frame)
            if not cache.busy:
                break
            time.sleep(0.002)
        stats = cache.stats()
        self.assertGreater(stats["evicted"], 400)
        resident = int((level.slots >= 0).sum())
        self.assertEqual(resident, sum(1 for p in pages if p.slot >= 0))
        self.assertLessEqual(resident, 1024)
        # 内容校验：换出的页从内存副本换回，逐字节一致
        evicted = [i for i, p in enumerate(pages) if p.slot < 0]
        self.assertTrue(evicted)
        for i in evicted[:40]:
            self.assertEqual(cache.page_bytes(pages[i]), pattern(i, nbytes))
        frame += 5
        cache.frame = frame
        cache.make_resident([pages[i] for i in evicted[:40]])
        for i in evicted[:40]:
            self.assertGreaterEqual(pages[i].slot, 0)
            self.assertEqual(pools["rgba8"].read_sync(pages[i].slot), pattern(i, nbytes))
            self.assertEqual(level.slots[i // 64, i % 64], pages[i].slot)
        print("\n缓存统计:", stats)
        cache.release()
        pools.release()

    def test_blocking_fallback_never_loses_data(self):
        pools, cache = self.make(budget_mb=300)
        nbytes = pools["rgba8"].page_bytes
        pages = []
        for i in range(1200):                             # 不调用 service，逼出同步兜底
            page = cache.new_page("rgba8")
            pools["rgba8"].write(page.slot, pattern(1000 + i, nbytes))
            pages.append(page)
            cache.frame += 1
        self.assertGreater(cache.stats()["blocking_evictions"], 0)
        for i in (0, 1, 50, 150, 1199):
            self.assertEqual(cache.page_bytes(pages[i]), pattern(1000 + i, nbytes))
        cache.release()
        pools.release()

    def test_spill_to_scratch_and_clone(self):
        """内存副本超出预算时写进暂存文件，读回逐字节一致；复制页面共用副本或用显卡拷贝。"""
        import tempfile
        pools = PoolSet(self.ctx, 300 << 20)
        folder = tempfile.mkdtemp(prefix="splender_scratch_")
        cache = PageCache(self.ctx, pools, ram_budget=8 << 20, caps=self.caps, scratch_dir=folder)
        nbytes = pools["rgba8"].page_bytes
        rng = np.random.default_rng(7)
        pages, datas = [], []
        frame = 0
        for i in range(1400):
            page = cache.new_page("rgba8")
            data = rng.integers(0, 255, nbytes, dtype=np.uint8).tobytes() if i % 3 == 0 else pattern(i, nbytes)
            pools["rgba8"].write(page.slot, data)
            pages.append(page)
            datas.append(data)
            if i % 16 == 0:
                frame += 1
                glx.flush()
                cache.service(frame, idle=True)
        for _ in range(400):
            frame += 1
            glx.flush()
            cache.service(frame, idle=True)
            if not cache.busy and not cache.has_background_work():
                break
            time.sleep(0.002)
        stats = cache.stats()
        print(chr(10) + "落盘统计:", {k: stats[k] for k in ("ram_mb", "scratch_mb", "spilled", "backed_up", "evicted")})
        self.assertGreater(stats["spilled"], 0)
        self.assertLess(stats["ram_mb"], 8 * 1.6)
        spilled = [i for i, p in enumerate(pages) if p.scratch is not None and p.blob is None]
        self.assertTrue(spilled)
        for i in spilled[:30] + [0, 699, 1399]:
            self.assertEqual(cache.page_bytes(pages[i]), datas[i])
        # 从暂存文件换回显存
        target = [pages[i] for i in spilled[:20] if pages[i].slot < 0]
        cache.request(target)
        for _ in range(200):
            frame += 1
            glx.flush()
            cache.service(frame)
            if all(p.slot >= 0 for p in target):
                break
            time.sleep(0.002)
        for page in target:
            self.assertGreaterEqual(page.slot, 0)
            self.assertEqual(pools["rgba8"].read_sync(page.slot), datas[pages.index(page)])
        # 复制：有副本的共用，只在显存的用显卡拷贝
        fresh = cache.new_page("rgba8")
        payload = pattern(4242, nbytes)
        pools["rgba8"].write(fresh.slot, payload)
        for source, expect in ((fresh, payload), (pages[spilled[0]], datas[spilled[0]]), (pages[5], datas[5])):
            copy = cache.clone_page(source)
            self.assertEqual(cache.page_bytes(copy), expect)
        scratch_path = cache.scratch.path
        cache.release()
        pools.release()
        import os
        self.assertFalse(os.path.exists(scratch_path), "暂存文件应在关闭时删除")

    def test_try_new_page_never_blocks(self):
        pools, cache = self.make(budget_mb=300)
        got = 0
        for _ in range(1100):
            page = cache.try_new_page("rgba8")
            if page is None:
                break
            got += 1
        self.assertGreater(got, 900)
        self.assertEqual(cache.stats()["blocking_evictions"], 0)
        cache.release()
        pools.release()

    def test_shader_pack_readback_matches(self):
        """用着色器打包读回的字节，与逐页读回完全一致（三种格式）。"""
        from splender.engine.pageops import PageOps
        pools, cache = self.make(budget_mb=900)
        ops = PageOps(self.ctx, pools)
        rng = np.random.default_rng(3)
        pages, datas = [], []
        for i, fmt in enumerate(["rgba8", "rg16f", "r8"] * 20):
            page = cache.new_page(fmt)
            nbytes = pools[fmt].page_bytes
            if fmt == "rg16f":
                values = rng.uniform(-4.0, 4.0, nbytes // 2).astype(np.float16)
                data = values.tobytes()
            else:
                data = rng.integers(0, 255, nbytes, dtype=np.uint8).tobytes()
            pools[fmt].write(page.slot, data)
            pages.append(page)
            datas.append(data)
        plain = cache.fetch_bytes(pages)
        cache.packer = ops
        t0 = time.perf_counter()
        packed = cache.fetch_bytes(pages)
        elapsed = (time.perf_counter() - t0) * 1000
        for i, (a, b, c) in enumerate(zip(plain, packed, datas)):
            self.assertEqual(a, c, "逐页读回不一致 %d" % i)
            self.assertEqual(b, c, "打包读回不一致 %d（%s）" % (i, pages[i].fmt))
        print(chr(10) + "着色器打包读回 60 页：%.2f ms" % elapsed)
        ops.release()
        cache.release()
        pools.release()

    def test_uniform_pages_skip_readback(self):
        """纯色页被认出来后只记一个值；换出不用读回，换入在显卡上填色，内容逐字节一致。"""
        from splender.engine.cache import CONST
        from splender.engine.pageops import PageOps
        pools, cache = self.make(budget_mb=900)
        ops = PageOps(self.ctx, pools)
        cache.packer = ops
        rng = np.random.default_rng(11)
        cases = []
        for fmt in ("rgba8", "rg16f", "r8"):
            nbytes = pools[fmt].page_bytes
            texel = {"rgba8": 4, "rg16f": 4, "r8": 1}[fmt]
            if fmt == "rg16f":
                value = np.array([0.73, -1.25], np.float16).tobytes()
            else:
                value = bytes(rng.integers(0, 255, texel, dtype=np.uint8).tolist())
            flat = value * (nbytes // texel)
            noisy = bytearray(flat)
            noisy[nbytes // 2] ^= 0x5A                     # 只差一个字节，也不能算纯色
            for data, uniform in ((flat, True), (bytes(noisy), False)):
                page = cache.new_page(fmt)
                pools[fmt].write(page.slot, data)
                cases.append((page, data, uniform))
        cache.note_new_pages([c[0] for c in cases])
        frame = 0
        for _ in range(50):
            frame += 1
            glx.flush()
            cache.service(frame)
            if not cache._checking and not cache._to_check:
                break
            time.sleep(0.002)
        for page, data, uniform in cases:
            if uniform:
                self.assertEqual(page.codec, CONST, "%s 纯色页应被认出" % page.fmt)
                self.assertLessEqual(len(page.blob), 4)
            else:
                self.assertIsNone(page.blob, "%s 非纯色页不应被当成纯色" % page.fmt)
        # 换出再换入：纯色页走显卡填色
        uniform_pages = [c for c in cases if c[2]]
        for page, _data, _u in uniform_pages:
            cache._drop_slot(page)
        cache.request([c[0] for c in uniform_pages])
        for _ in range(20):
            frame += 1
            cache.service(frame)
            if all(c[0].slot >= 0 for c in uniform_pages):
                break
        for page, data, _u in uniform_pages:
            self.assertGreaterEqual(page.slot, 0)
            self.assertEqual(pools[page.fmt].read_sync(page.slot), data, "%s 填色换入内容不一致" % page.fmt)
            self.assertEqual(cache.page_bytes(page), data)
        ops.release()
        cache.release()
        pools.release()

    def test_free_and_formats(self):
        pools, cache = self.make(budget_mb=700)
        a = cache.new_page("rg16f")
        b = cache.new_page("r8")
        self.assertEqual(pools["rg16f"].page_bytes, PAGE * PAGE * 4)
        self.assertEqual(pools["r8"].page_bytes, PAGE * PAGE)
        pools["r8"].write(b.slot, bytes([7]) * (PAGE * PAGE))
        self.assertEqual(pools["r8"].read_sync(b.slot)[:4], bytes([7, 7, 7, 7]))
        used = pools["rg16f"].used
        cache.free_page(a)
        self.assertEqual(pools["rg16f"].used, used - 1)
        cache.release()
        pools.release()


if __name__ == "__main__":
    unittest.main(verbosity=2)
