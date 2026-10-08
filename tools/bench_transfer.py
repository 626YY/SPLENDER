"""显存与内存之间搬页面的速度：几种读回、上传方式对比。用隐藏上下文。"""
import ctypes
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
import moderngl  # noqa: E402
import numpy as np  # noqa: E402
import lz4.block  # noqa: E402

from splender.engine import glx  # noqa: E402

ctx = moderngl.create_standalone_context(require=430)
PAGE = 256
LAYERS = 1024
tex = ctx.texture_array((PAGE, PAGE, LAYERS), 4, dtype="f1")
rng = np.random.default_rng(1)
page_bytes = PAGE * PAGE * 4
base = rng.integers(0, 255, page_bytes, dtype=np.uint8).tobytes()
for layer in range(0, LAYERS, 64):
    tex.write(base * 64, viewport=(0, 0, layer, PAGE, PAGE, 64), alignment=1)
ctx.finish()
batch = 64
buf = glx.PersistentBuffer(batch * page_bytes)


def readback_batch(first):
    for i in range(batch):
        glx.read_layer_into(buf, i * page_bytes, tex.glo, first + i, PAGE, PAGE, glx.GL_RGBA, glx.GL_UNSIGNED_BYTE,
                            page_bytes)
    fence = glx.Fence()
    while not fence.ready(wait_ns=100_000_000):
        pass
    fence.release()


for _ in range(3):
    readback_batch(0)
rounds = 20
t0 = time.perf_counter()
for r in range(rounds):
    readback_batch((r * batch) % (LAYERS - batch))
dt = time.perf_counter() - t0
print("读回 glGetTextureSubImage→常驻缓冲：%.2f GB/s（每批 %d 页 %.1f ms）" % (rounds * batch * page_bytes / dt / 1e9, batch,
                                                                    dt / rounds * 1000))

# 只发命令不等：CPU 侧开销
t0 = time.perf_counter()
for i in range(batch):
    glx.read_layer_into(buf, i * page_bytes, tex.glo, i, PAGE, PAGE, glx.GL_RGBA, glx.GL_UNSIGNED_BYTE, page_bytes)
issue = (time.perf_counter() - t0) * 1000
fence = glx.Fence()
t1 = time.perf_counter()
while not fence.ready(wait_ns=100_000_000):
    pass
wait = (time.perf_counter() - t1) * 1000
fence.release()
print("发起 64 页读回的 CPU 开销 %.2f ms，之后等显卡 %.2f ms" % (issue, wait))

# 上传：直接从内存写
datas = [base] * batch
t0 = time.perf_counter()
for r in range(rounds):
    for i in range(batch):
        tex.write(datas[i], viewport=(0, 0, (r * batch + i) % LAYERS, PAGE, PAGE, 1), alignment=1)
ctx.finish()
dt = time.perf_counter() - t0
print("上传 texture.write：%.2f GB/s（每页 %.3f ms）" % (rounds * batch * page_bytes / dt / 1e9, dt / rounds / batch * 1000))

# 上传：经常驻缓冲
up = glx.PersistentBuffer(batch * page_bytes)
GL_UNPACK = glx.GL_PIXEL_UNPACK_BUFFER
glTextureSubImage3D = None
print("glTextureSubImage3D 可用:", glTextureSubImage3D is not None)

# 解压与压缩速度
blob = lz4.block.compress(base, mode="fast", store_size=False)
t0 = time.perf_counter()
for _ in range(200):
    lz4.block.decompress(blob, uncompressed_size=page_bytes)
print("LZ4 解压（随机数据，%.0f%%）：%.2f ms/页" % (len(blob) / page_bytes * 100, (time.perf_counter() - t0) / 200 * 1000))
flat = bytes(page_bytes)
blob2 = lz4.block.compress(flat, mode="fast", store_size=False)
t0 = time.perf_counter()
for _ in range(200):
    lz4.block.compress(flat, mode="fast", store_size=False)
print("LZ4 压缩（纯色页，压到 %d 字节）：%.3f ms/页" % (len(blob2), (time.perf_counter() - t0) / 200 * 1000))
