"""显存页池。

一种格式一个池，池由若干张 256×256×N 的贴图数组组成。槽号 = 数组序号 * N + 层号。
所有池共用一个显存预算，按需增长。
"""
from __future__ import annotations

import logging

import moderngl
import numpy as np

from . import glx

log = logging.getLogger("splender.engine.pool")

PAGE = 256

FORMATS = {
    # 名字: (分量数, ModernGL dtype, 每纹素字节, GL 格式, GL 类型, GLSL 图像格式)
    "rgba8": (4, "f1", 4, glx.GL_RGBA, glx.GL_UNSIGNED_BYTE, "rgba8"),
    "rg16f": (2, "f2", 4, glx.GL_RG, glx.GL_HALF_FLOAT, "rg16f"),
    "r8": (1, "f1", 1, glx.GL_RED, glx.GL_UNSIGNED_BYTE, "r8"),
}
#: 每种格式最多几张数组（决定合成着色器要占多少个贴图单元）
MAX_ARRAYS = {"rgba8": 12, "rg16f": 6, "r8": 2}
NEVER = np.iinfo(np.int32).max


class PoolBudget:
    """所有页池共用的显存预算。"""

    def __init__(self, total_bytes: int) -> None:
        self.total = int(total_bytes)
        self.used = 0

    def can_take(self, nbytes: int) -> bool:
        return self.used + nbytes <= self.total


class PagePool:
    def __init__(self, ctx: moderngl.Context, fmt: str, budget: PoolBudget, layers: int = 1024) -> None:
        self.ctx = ctx
        self.fmt = fmt
        self.budget = budget
        self.layers = layers
        self.shift = int(layers).bit_length() - 1
        if (1 << self.shift) != layers:
            raise ValueError("每张数组的层数必须是 2 的幂")
        components, dtype, texel, gl_format, gl_type, glsl = FORMATS[fmt]
        self.components = components
        self.dtype = dtype
        self.gl_format = gl_format
        self.gl_type = gl_type
        self.glsl_format = glsl
        self.page_bytes = PAGE * PAGE * texel
        self.max_arrays = MAX_ARRAYS[fmt]
        self.arrays: list[moderngl.TextureArray] = []
        self.free: list[int] = []
        capacity = self.max_arrays * layers
        self.stamp = np.zeros(capacity, np.int32)          # 最近一次使用的帧号
        self.pins = np.zeros(capacity, np.int16)           # 钉住计数，>0 不可换出
        self.owner: list = [None] * capacity               # 占用该槽的 Page
        self.occupied = np.zeros(capacity, bool)
        self.clean = np.zeros(capacity, bool)              # 该槽的页在内存或磁盘上已有副本，换出时不用读回
        self.used = 0

    # ---- 容量 ----
    @property
    def capacity(self) -> int:
        return len(self.arrays) * self.layers

    @property
    def free_count(self) -> int:
        """现有数组里的空槽，加上预算内还能新增的槽。"""
        return len(self.free) + self.growable_slots()

    def growable_slots(self) -> int:
        array_bytes = self.page_bytes * self.layers
        room = (self.budget.total - self.budget.used) // array_bytes
        return int(max(0, min(room, self.max_arrays - len(self.arrays)))) * self.layers

    def _grow(self) -> bool:
        array_bytes = self.page_bytes * self.layers
        if len(self.arrays) >= self.max_arrays or not self.budget.can_take(array_bytes):
            return False
        tex = self.ctx.texture_array((PAGE, PAGE, self.layers), self.components, dtype=self.dtype)
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = False
        tex.repeat_y = False
        index = len(self.arrays)
        self.arrays.append(tex)
        self.budget.used += array_bytes
        base = index * self.layers
        self.free.extend(range(base + self.layers - 1, base - 1, -1))
        log.info("页池 %s 增加第 %d 张数组（%d MB）", self.fmt, index + 1, array_bytes >> 20)
        return True

    def warm_up(self) -> None:
        """启动时先建好第一张数组并清零。"""
        if self.capacity == 0 and self._grow():
            glx.clear_texture(self.arrays[-1].glo, self.gl_format, self.gl_type)

    def grow_ahead(self) -> bool:
        """空闲时提前加一张数组：空槽不到半张数组、预算里除了这张还留得出一张给别的格式时才加。"""
        if self.capacity == 0 or len(self.free) >= self.layers // 2:
            return False
        array_bytes = self.page_bytes * self.layers
        if self.budget.total - self.budget.used < 2 * array_bytes:
            return False
        if not self._grow():
            return False
        glx.clear_texture(self.arrays[-1].glo, self.gl_format, self.gl_type)
        return True

    # ---- 分配 ----
    def alloc(self) -> int:
        """取一个空槽；没有空槽且不能再增长时返回 -1。"""
        if not self.free and not self._grow():
            return -1
        slot = self.free.pop()
        self.used += 1
        return slot

    def alloc_many(self, count: int) -> list[int]:
        """一次取最多 count 个空槽（不够时在预算内增长）。"""
        taken: list[int] = []
        while len(taken) < count:
            if not self.free and not self._grow():
                break
            take = min(count - len(taken), len(self.free))
            taken.extend(self.free[-take:][::-1])
            del self.free[-take:]
        self.used += len(taken)
        return taken

    def assign(self, slot: int, page, frame: int = 0) -> None:
        self.owner[slot] = page
        self.occupied[slot] = True
        self.stamp[slot] = frame
        self.clean[slot] = (getattr(page, "blob", None) is not None or getattr(page, "disk_key", None) is not None
                            or getattr(page, "scratch", None) is not None)

    def release(self, slot: int) -> None:
        if slot < 0:
            return
        self.owner[slot] = None
        self.occupied[slot] = False
        self.clean[slot] = False
        self.pins[slot] = 0
        self.free.append(slot)
        self.used -= 1

    def array_of(self, slot: int) -> int:
        return slot >> self.shift

    def layer_of(self, slot: int) -> int:
        return slot & (self.layers - 1)

    def texture(self, slot: int) -> moderngl.TextureArray:
        return self.arrays[slot >> self.shift]

    # ---- 读写 ----
    def write(self, slot: int, data: bytes) -> None:
        self.arrays[slot >> self.shift].write(data, viewport=(0, 0, slot & (self.layers - 1), PAGE, PAGE, 1), alignment=1)

    def read_sync(self, slot: int) -> bytes:
        """同步读回一页。只用于少量、不在笔划路径上的场合。"""
        if glx.available("glGetTextureSubImage"):
            buf = (glx.ctypes.c_ubyte * self.page_bytes)()
            glx.fn("glPixelStorei")(glx.GL_PACK_ALIGNMENT, 1)
            glx.fn("glBindBuffer")(glx.GL_PIXEL_PACK_BUFFER, 0)
            glx.fn("glGetTextureSubImage")(self.arrays[slot >> self.shift].glo, 0, 0, 0, slot & (self.layers - 1),
                                           PAGE, PAGE, 1, self.gl_format, self.gl_type, self.page_bytes, buf)
            return bytes(buf)
        raise RuntimeError("当前驱动不支持按层读回")

    # ---- 换出候选 ----
    def victims(self, count: int, frame: int, min_age: int = 2, clean_only: bool = False) -> np.ndarray:
        """最久没用、没被钉住的 count 个槽。clean_only 时只要已有副本的（可以立刻释放）。"""
        capacity = self.capacity
        if capacity == 0 or count <= 0:
            return np.empty(0, np.int64)
        stamp = self.stamp[:capacity].astype(np.int64)
        blocked = (~self.occupied[:capacity]) | (self.pins[:capacity] > 0) | (stamp > frame - min_age)
        if clean_only:
            blocked |= ~self.clean[:capacity]
        stamp[blocked] = NEVER
        count = min(count, int((~blocked).sum()))
        if count <= 0:
            return np.empty(0, np.int64)
        picked = np.argpartition(stamp, count - 1)[:count]
        return picked[np.argsort(stamp[picked])]

    def unbacked(self, count: int) -> np.ndarray:
        """还没有副本、没被钉住的槽，最久没用的排在前面，最多 count 个。"""
        capacity = self.capacity
        if capacity == 0 or count <= 0:
            return np.empty(0, np.int64)
        found = np.flatnonzero(self.occupied[:capacity] & ~self.clean[:capacity] & (self.pins[:capacity] == 0))
        if len(found) > count:
            found = found[np.argpartition(self.stamp[found], count - 1)[:count]]
        return found[np.argsort(self.stamp[found], kind="stable")]

    def unbacked_count(self) -> int:
        capacity = self.capacity
        if capacity == 0:
            return 0
        return int((self.occupied[:capacity] & ~self.clean[:capacity] & (self.pins[:capacity] == 0)).sum())

    def touch(self, slots: np.ndarray, frame: int) -> None:
        self.stamp[slots] = frame

    def release_all(self) -> None:
        for tex in self.arrays:
            tex.release()
        self.budget.used -= self.page_bytes * self.layers * len(self.arrays)
        self.arrays.clear()
        self.free.clear()
        self.used = 0


class PoolSet:
    """全部页池。"""

    def __init__(self, ctx: moderngl.Context, budget_bytes: int) -> None:
        self.budget = PoolBudget(budget_bytes)
        layers = 2048 if budget_bytes > 6 * 1024 ** 3 else 1024
        self.layers = layers
        self.shift = layers.bit_length() - 1
        self.pools = {fmt: PagePool(ctx, fmt, self.budget, layers) for fmt in FORMATS}

    def __getitem__(self, fmt: str) -> PagePool:
        return self.pools[fmt]

    def stats(self) -> dict:
        return {
            "budget_mb": self.budget.total >> 20,
            "allocated_mb": self.budget.used >> 20,
            "used_mb": sum(p.used * p.page_bytes for p in self.pools.values()) >> 20,
            "pages": {fmt: p.used for fmt, p in self.pools.items()},
        }

    def release(self) -> None:
        for pool in self.pools.values():
            pool.release_all()


def glsl_fetch(fmt: str, count: int, shift: int) -> str:
    """生成从页池取纹素的 GLSL：uniform sampler2DArray u_<fmt>[count]; vec4 fetch_<fmt>(int slot, ivec2 p)。

    用 switch 而不是变量下标，符合规范，各家驱动都能编译。
    """
    mask = (1 << shift) - 1
    lines = ["uniform sampler2DArray u_%s[%d];" % (fmt, count),
             "vec4 fetch_%s(int slot, ivec2 p){" % fmt,
             "  ivec3 c = ivec3(p, slot & %d);" % mask,
             "  switch(slot >> %d){" % shift]
    for index in range(count):
        lines.append("    case %d: return texelFetch(u_%s[%d], c, 0);" % (index, fmt, index))
    lines += ["  }", "  return vec4(0.0);", "}"]
    return "\n".join(lines)


def bind_samplers(program, pools: PoolSet, fmt: str, first_unit: int) -> int:
    """把某种格式的全部数组绑到连续的贴图单元上，返回下一个可用单元。"""
    pool = pools[fmt]
    name = "u_" + fmt
    if name not in program and (name + "[0]") not in program:
        return first_unit
    units = []
    for index in range(pool.max_arrays):
        tex = pool.arrays[index] if index < len(pool.arrays) else (pool.arrays[0] if pool.arrays else None)
        unit = first_unit + index
        if tex is not None:
            tex.use(unit)
        units.append(unit)
    member = program[name] if name in program else program[name + "[0]"]
    member.value = units if pool.max_arrays > 1 else units[0]
    return first_unit + pool.max_arrays
