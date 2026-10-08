"""逐个测页面着色器的显卡耗时（计时查询）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402

from splender.engine import glx  # noqa: E402
from splender.engine.cache import PageCache  # noqa: E402
from splender.engine.pageops import KIND_COLOR, KIND_HEIGHT, PageOps  # noqa: E402
from splender.engine.pagepool import PoolSet  # noqa: E402

ctx = moderngl.create_standalone_context(require=430)
pools = PoolSet(ctx, 2048 << 20)
cache = PageCache(ctx, pools, ram_budget=1 << 30, caps=glx.Caps())
ops = PageOps(ctx, pools)
N = 256
color = [cache.new_page("rgba8") for _ in range(N * 2)]
stroke = [cache.new_page("rg16f") for _ in range(N)]
height = [cache.new_page("rg16f") for _ in range(N)]
src = np.array([p.slot for p in color[:N]], np.int32)
dst = np.array([p.slot for p in color[N:]], np.int32)
ss = np.array([p.slot for p in stroke], np.int32)
hs = np.array([p.slot for p in height], np.int32)
ops.clear("rgba8", src, (0.2, 0.4, 0.6, 1.0))
ops.clear("rg16f", ss, (0.7, 0.0, 0.0, 0.0))
buf = glx.PersistentBuffer(N * 8 + 64)
ctx.finish()
query = ctx.query(time=True)


def gpu(label, fn, repeat=5):
    times = []
    for _ in range(repeat):
        with query:
            fn()
        ctx.finish()
        times.append(query.elapsed / 1e6)
    t = sorted(times)[len(times) // 2]
    print("%-28s %7.2f ms  每页 %6.1f µs" % (label, t, t * 1000 / N))


gpu("清零 rgba8", lambda: ops.clear("rgba8", dst))
gpu("并入笔划（颜色）", lambda: ops.apply_stroke(KIND_COLOR, dst, src, ss, color=(1, 0, 0), values=(0, 0, 0, 0),
                                                enable=(0, 0, 0, 0), opacity=1.0, erase=False))
children = np.stack([src[(np.arange(N) * 4 + q) % N] for q in range(4)], axis=1).astype(np.int32)
gpu("生成上一级（颜色）", lambda: ops.downsample(KIND_COLOR, dst, np.full(N, -1, np.int32), children))
gpu("生成上一级（高度）", lambda: ops.downsample(KIND_HEIGHT, hs, np.full(N, -1, np.int32),
                                               np.stack([ss] * 4, axis=1).astype(np.int32)))
gpu("纯色检测 rgba8", lambda: ops.check_uniform("rgba8", dst, np.arange(N) * 2, buf.handle))
gpu("打包读回 rgba8（16 MB 一批）", lambda: [ops.pack("rgba8", dst[i:i + 64], np.arange(64) * 262144, cache._buffers_tmp)
                                       for i in range(0, N, 64)] if False else None, 1)
big = glx.PersistentBuffer(16 << 20)
gpu("打包读回 rgba8 64 页", lambda: ops.pack("rgba8", dst[:64], np.arange(64) * 262144, big.handle))
gpu("填色 rgba8", lambda: ops.fill_packed("rgba8", dst, np.full(N, 0x80FF2010, np.uint32)))
