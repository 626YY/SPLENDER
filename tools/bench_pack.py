"""着色器打包读回的稳态速度（隐藏上下文）。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

from splender.engine import glx  # noqa: E402
from splender.engine.cache import PageCache  # noqa: E402
from splender.engine.pageops import PageOps  # noqa: E402
from splender.engine.pagepool import PoolSet  # noqa: E402

ctx = moderngl.create_standalone_context(require=430)
caps = glx.Caps()
pools = PoolSet(ctx, 2048 << 20)
cache = PageCache(ctx, pools, ram_budget=4096 << 20, caps=caps)
ops = PageOps(ctx, pools)
pages = []
data = np.random.default_rng(1).integers(0, 255, pools["rgba8"].page_bytes, dtype=np.uint8).tobytes()
for i in range(2048):
    page = cache.new_page("rgba8")
    pools["rgba8"].write(page.slot, data)
    pages.append(page)
ctx.finish()
buf = glx.PersistentBuffer(16 << 20)
page_bytes = pools["rgba8"].page_bytes
for mode in ("逐页 glGetTextureSubImage", "着色器打包"):
    cache.packer = ops if mode == "着色器打包" else None
    issue_ms, wait_ms = [], []
    for r in range(40):
        chunk = pages[(r * 64) % 1984:(r * 64) % 1984 + 64]
        placed = [(p, i * page_bytes) for i, p in enumerate(chunk)]
        t0 = time.perf_counter()
        cache._issue_reads(buf, placed)
        fence = glx.Fence()
        t1 = time.perf_counter()
        while not fence.ready(wait_ns=100_000_000):
            pass
        fence.release()
        t2 = time.perf_counter()
        if r >= 5:
            issue_ms.append((t1 - t0) * 1000)
            wait_ms.append((t2 - t1) * 1000)
    ok = bytes(buf.view(0, page_bytes)) == data
    print("%s：发命令 %.3f ms，等显卡 %.3f ms（64 页 16 MB），内容%s" % (mode, np.median(issue_ms), np.median(wait_ms),
                                                            "一致" if ok else "不一致"))
