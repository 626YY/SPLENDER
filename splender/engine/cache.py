"""页面缓存：显存、内存、磁盘三级。

页面一旦写好就不再原地修改（改动产生新页），所以内存和磁盘里的副本永远不会过期：
换出一个已有副本的页只需要释放槽位。

为了让「腾位置」永远是瞬间的事，缓存在空闲时把还没有副本的页提前读回内存（后台备份）。
抬笔合并、换入旧页需要槽位时，直接释放已有副本的页，不用等显卡读回。

内存里的副本超过预算时，最早的那些挪到暂存文件。

每帧调用 service()：收回已完成的读回、把到货的页上传、保持空闲水位、后台备份、内存超预算时落盘。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor

import lz4.block
import numpy as np

from . import glx
from .pagepool import PAGE, PoolSet

log = logging.getLogger("splender.engine.cache")

RAW = 0
LZ4 = 1
CONST = 2            # 纯色页：副本只存一个纹素的字节
TEXEL_BYTES = {"rgba8": 4, "rg16f": 4, "r8": 1}
CHECK_PAGES = 4096   # 每块纯色检测结果缓冲能放多少页的结果


class Page:
    """一页数据。slot>=0 表示在显存；blob 是内存副本；scratch 是暂存文件里的副本；disk_key 表示工程文件里有这页。"""

    __slots__ = ("fmt", "slot", "blob", "codec", "disk_key", "scratch", "dirty", "home", "home_x", "home_y", "busy",
                 "dead", "want", "asave")

    def __init__(self, fmt: str) -> None:
        self.fmt = fmt
        self.slot = -1
        self.blob: bytes | None = None
        self.codec = RAW
        self.disk_key = None
        self.scratch = None          # (偏移, 长度, 编码)
        self.dirty = True            # 相对工程文件有没有保存
        self.home = None             # 它所在的页表（LevelGrid），位置是 home_x, home_y。不用元组，少给垃圾回收添对象
        self.home_x = 0
        self.home_y = 0
        self.busy = False            # 正在读回或读盘
        self.dead = False            # 已被丢弃（异步任务回来时据此放弃）
        self.want = False            # 已请求换入
        self.asave = None            # 自动保存记号：(恢复文件的标记, 页面键)，说明这页已经写进当前的恢复文件


class LevelGrid:
    """某个图层、某种页、某一级的页表。slots 给着色器任务用，-1 表示不在显存。"""

    __slots__ = ("size", "pages", "slots", "exists")

    def __init__(self, size: int) -> None:
        self.size = size
        self.pages: dict[tuple[int, int], Page] = {}
        self.slots = np.full((size, size), -1, np.int32)
        self.exists = np.zeros((size, size), bool)


class PlaneStore:
    """一个图层的一种页，各级页表。"""

    __slots__ = ("fmt", "levels")

    def __init__(self, fmt: str, grid: int) -> None:
        self.fmt = fmt
        self.levels: list[LevelGrid] = []
        size = grid
        while size >= 1:
            self.levels.append(LevelGrid(size))
            size //= 2

    def page_count(self) -> int:
        return sum(len(level.pages) for level in self.levels)


class ScratchFile:
    """内存放不下时的暂存文件。按 4 KB 对齐分配，释放出来的空位按大小复用。关闭时删除。"""

    BLOCK = 4096

    def __init__(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, "splender-scratch-%d-%d.bin" % (os.getpid(), int(time.time())))
        self._file = open(self.path, "w+b", buffering=0)
        self._lock = threading.Lock()
        self._end = 0
        self._holes: dict[int, list[int]] = {}
        self.used = 0

    def _blocks(self, nbytes: int) -> int:
        return (nbytes + self.BLOCK - 1) // self.BLOCK

    def allocate(self, nbytes: int) -> int:
        blocks = self._blocks(nbytes)
        self.used += blocks * self.BLOCK
        holes = self._holes.get(blocks)
        if holes:
            return holes.pop()
        offset = self._end
        self._end += blocks * self.BLOCK
        return offset

    def free(self, offset: int, nbytes: int) -> None:
        blocks = self._blocks(nbytes)
        self._holes.setdefault(blocks, []).append(offset)
        self.used -= blocks * self.BLOCK

    def write(self, offset: int, data: bytes) -> None:
        with self._lock:
            self._file.seek(offset)
            self._file.write(data)

    def read(self, offset: int, nbytes: int) -> bytes:
        with self._lock:
            self._file.seek(offset)
            return self._file.read(nbytes)

    @property
    def size(self) -> int:
        return self._end

    def close(self) -> None:
        try:
            self._file.close()
        except OSError:
            pass
        try:
            os.remove(self.path)
        except OSError:
            pass


class _Batch:
    __slots__ = ("buffer", "items", "fence", "futures", "drop")

    def __init__(self, buffer_index: int, drop: bool) -> None:
        self.buffer = buffer_index
        self.items: list[tuple[Page, int, int]] = []
        self.fence = None
        self.futures: list[tuple[list, Future]] = []     # (这一组的页, 压缩任务)：每组若干页一起压缩
        self.drop = drop                 # 读回完成后是否释放槽位（换出）；否则只是备份


COMPRESS_GROUP = 16                      # 读回后每多少页交给一个线程一起压缩


class PageCache:
    BATCH_BYTES = 16 * 1024 * 1024       # 每批读回的字节数
    RING = 8                             # 同时在途的读回批数
    KEEP_FOR_EVICTION = 2                # 后台备份至少给换出留几个读回缓冲
    MAX_LOADING = 192                    # 同时在后台读取、解压的页数上限

    def __init__(self, ctx, pools: PoolSet, *, ram_budget: int, caps: glx.Caps, watermark: float = 0.08,
                 threads: int | None = None, scratch_dir: str | None = None, backup_mb_active: float = 8.0,
                 backup_mb_idle: float = 128.0) -> None:
        self.ctx = ctx
        self.pools = pools
        self.caps = caps
        self.ram_budget = int(ram_budget)
        self.watermark = float(watermark)
        self.backup_bytes_active = int(backup_mb_active * 1048576)
        self.backup_bytes_idle = int(backup_mb_idle * 1048576)
        self.frame = 0
        self.storage = None                       # 工程文件（doc.storage.ProjectFile）
        self.ram_bytes = 0
        self.blob_count = 0
        self.workers = ThreadPoolExecutor(max_workers=threads or max(2, min(8, (os.cpu_count() or 4) - 2)),
                                          thread_name_prefix="splender-page")
        self.async_readback = bool(caps.buffer_storage and caps.get_sub_image)
        self._buffers: list[glx.PersistentBuffer | None] = [None] * self.RING
        self._free_buffers = list(range(self.RING))
        self._batches: list[_Batch] = []
        self._requested: deque = deque()                        # 等着发起后台读取的页
        self._loading: list[tuple[Page, Future]] = []           # 后台读取中，结果是 (原始字节, 可留作副本的压缩字节)
        self._upload_queue: deque = deque()                     # (页, 原始字节)，等待上传
        self._sync_buffer: glx.PersistentBuffer | None = None   # 同步批量读回用
        self._recompress: list[tuple[Page, bytes, Future]] = []  # 兜底读回留下的未压缩副本，后台补压缩
        self._spilling: list[tuple[Page, bytes, int, int, Future]] = []   # 正在写往暂存文件的副本
        self._spilling_ids: set[int] = set()
        self._blob_order: deque = deque()                       # 内存副本产生的先后，超预算时从最早的开始落盘
        self.scratch_dir = scratch_dir
        self.scratch: ScratchFile | None = None
        self._scratch_failed = False
        self.reserve: dict[str, int] = {}                       # 各格式额外要保持空闲的槽数（等着用槽的工作）
        self.packer = None                                      # PageOps：有它时读回用一次着色器调度完成
        self._to_check: list[Page] = []                         # 新生成、等着检查是不是纯色的页
        self._checking: list[tuple[list, int, object]] = []     # (这批页, 结果缓冲序号, Fence)
        self._checking_ids: set[int] = set()
        self._check_buffers: list[glx.PersistentBuffer | None] = [None] * 4
        self._check_free = list(range(4))
        self._fill_queue: deque = deque()                       # 纯色页换入：直接在显卡上填色
        self.stat_uniform = 0
        # 统计
        self.stat_evicted = 0
        self.stat_uploaded = 0
        self.stat_blocking_evictions = 0
        self.stat_disk_reads = 0
        self.stat_sync_disk_reads = 0
        self.stat_backed_up = 0
        self.stat_spilled = 0

    # ------------------------------------------------------------------ 页表
    def set_page(self, level: LevelGrid, tx: int, ty: int, page: Page | None) -> Page | None:
        """把页表的一格指向新页（或清空），返回原来的页。原来的页不会被释放。"""
        old = level.pages.get((tx, ty))
        if old is not None and old is not page:
            old.home = None
            if old.slot >= 0:
                self.pools[old.fmt].stamp[old.slot] = 0      # 换下来的页只有撤销会用到，优先换出
        if page is None:
            level.pages.pop((tx, ty), None)
            level.slots[ty, tx] = -1
            level.exists[ty, tx] = False
        else:
            level.pages[(tx, ty)] = page
            page.home = level
            page.home_x = tx
            page.home_y = ty
            level.slots[ty, tx] = page.slot
            level.exists[ty, tx] = True
            if page.slot >= 0:
                self.pools[page.fmt].stamp[page.slot] = self.frame
        return old

    @staticmethod
    def _sync_home(page: Page) -> None:
        level = page.home
        if level is not None:
            level.slots[page.home_y, page.home_x] = page.slot

    def _refresh_clean(self, page: Page) -> None:
        if page.slot >= 0:
            self.pools[page.fmt].clean[page.slot] = (page.blob is not None or page.scratch is not None
                                                     or page.disk_key is not None)

    # ------------------------------------------------------------------ 分配与释放
    def new_page(self, fmt: str, pinned: bool = False) -> Page:
        """新建一页并在显存里占一个槽，一定成功（必要时同步腾位置）。内容由调用者写入。"""
        page = Page(fmt)
        self._attach_slot(page, block=True)
        if pinned:
            self.pools[fmt].pins[page.slot] += 1
        return page

    def try_new_page(self, fmt: str, pinned: bool = False) -> Page | None:
        """新建一页；只用现成能腾出的槽，腾不出就返回 None（过一帧再试，后台会腾出位置）。"""
        page = Page(fmt)
        if not self._attach_slot(page, block=False):
            return None
        if pinned:
            self.pools[fmt].pins[page.slot] += 1
        return page

    def try_new_pages(self, fmt: str, count: int, block: bool = False) -> list[Page]:
        """一次新建多页。不阻塞时只用现成能腾出的槽，可能少于 count。"""
        pool = self.pools[fmt]
        slots = pool.alloc_many(count)
        if len(slots) < count:
            self._release_clean(fmt, max(64, count - len(slots)))
            slots += pool.alloc_many(count - len(slots))
        if block:
            while len(slots) < count:
                self._make_room(fmt, max(64, count - len(slots)))
                more = pool.alloc_many(count - len(slots))
                if not more:
                    raise MemoryError("显存页池已满，且没有可以换出的页")
                slots += more
        pages = [Page(fmt) for _ in slots]
        owner = pool.owner
        for page, slot in zip(pages, slots):
            page.slot = slot
            owner[slot] = page
        if slots:
            index = np.asarray(slots, np.int64)
            pool.occupied[index] = True
            pool.clean[index] = False
            pool.stamp[index] = self.frame
        return pages

    def pin_many(self, pages, amount: int = 1) -> None:
        """批量钉住（amount=-1 解钉）。同一页出现多次会按次数计。"""
        groups: dict[str, list[int]] = {}
        for page in pages:
            if page.slot >= 0:
                groups.setdefault(page.fmt, []).append(page.slot)
        for fmt, slots in groups.items():
            pins = self.pools[fmt].pins
            np.add.at(pins, np.asarray(slots, np.int64), amount)
            if amount < 0:
                np.maximum(pins, 0, out=pins)

    def free_slots(self, fmt: str) -> int:
        return self.pools[fmt].free_count

    def _attach_slot(self, page: Page, block: bool = True) -> bool:
        pool = self.pools[page.fmt]
        slot = pool.alloc()
        if slot < 0:
            self._release_clean(page.fmt, 64)
            slot = pool.alloc()
        if slot < 0:
            if not block:
                return False
            self._make_room(page.fmt, 64)
            slot = pool.alloc()
            if slot < 0:
                raise MemoryError("显存页池已满，且没有可以换出的页")
        pool.assign(slot, page, self.frame)
        page.slot = slot
        self._sync_home(page)
        return True

    def free_page(self, page: Page | None) -> None:
        """彻底丢弃一页（显存、内存、暂存文件里的副本都放掉）。"""
        if page is None or page.dead:
            return
        page.dead = True
        if page.slot >= 0 and not page.busy:
            self.pools[page.fmt].release(page.slot)
            page.slot = -1
        if page.blob is not None:
            self._drop_blob(page)
        if page.scratch is not None:
            if self.scratch is not None:
                self.scratch.free(page.scratch[0], page.scratch[1])
            page.scratch = None
        if page.home is not None:
            self._sync_home(page)
            page.home = None              # 断开页与页表的互相引用，整层丢弃时靠引用计数就能回收

    def pin(self, page: Page) -> None:
        if page.slot >= 0:
            self.pools[page.fmt].pins[page.slot] += 1

    def unpin(self, page: Page) -> None:
        if page.slot >= 0:
            pool = self.pools[page.fmt]
            pool.pins[page.slot] = max(0, pool.pins[page.slot] - 1)

    def touch(self, pages) -> None:
        for page in pages:
            if page is not None and page.slot >= 0:
                self.pools[page.fmt].stamp[page.slot] = self.frame

    def clone_page(self, page: Page) -> Page:
        """复制一页（复制图层用）。内存副本直接共用，只在显存里的页用显卡拷贝。"""
        copy = Page(page.fmt)
        if page.blob is not None:
            self._set_blob(copy, page.blob, page.codec)
        elif page.scratch is not None and self.scratch is not None:
            offset, nbytes, codec = page.scratch
            self._set_blob(copy, self.scratch.read(offset, nbytes), codec)
        elif page.slot >= 0:
            pool = self.pools[page.fmt]
            self._attach_slot(copy, block=True)
            glx.copy_layer(pool.texture(page.slot).glo, pool.layer_of(page.slot),
                           pool.texture(copy.slot).glo, pool.layer_of(copy.slot), PAGE, PAGE)
            self.note_new_pages([copy])
        elif page.disk_key is not None and self.storage is not None:
            data = self.storage.read_page(page.disk_key)
            if data is None:
                raise RuntimeError("工程文件里找不到页面 %r" % (page.disk_key,))
            self.adopt_bytes(copy, data)
        else:
            raise RuntimeError("页面数据丢失")
        return copy

    # ------------------------------------------------------------------ 内存副本
    def _set_blob(self, page: Page, data: bytes, codec: int) -> None:
        if page.blob is not None:
            self.ram_bytes -= len(page.blob)
        else:
            self.blob_count += 1
        page.blob = data
        page.codec = codec
        self.ram_bytes += len(data)
        self._blob_order.append(page)
        if page.slot >= 0:
            self.pools[page.fmt].clean[page.slot] = True
        if len(self._blob_order) > 2 * self.blob_count + 8192:
            self._compact_order()

    def _drop_blob(self, page: Page) -> None:
        if page.blob is None:
            return
        self.ram_bytes -= len(page.blob)
        self.blob_count -= 1
        page.blob = None
        self._refresh_clean(page)

    def _compact_order(self) -> None:
        seen: set[int] = set()
        kept: list[Page] = []
        for page in reversed(self._blob_order):
            if page.blob is None or page.dead or id(page) in seen:
                continue
            seen.add(id(page))
            kept.append(page)
        kept.reverse()
        self._blob_order = deque(kept)

    def adopt_bytes(self, page: Page, data: bytes) -> None:
        """给一页一份原始字节的内存副本（之后后台压缩）。"""
        self._set_blob(page, data, RAW)
        self._recompress.append((page, data, self.workers.submit(self._compress, data)))

    def mark_saved(self, page: Page, key) -> None:
        """这页已经写进工程文件。"""
        page.disk_key = key
        page.dirty = False
        self._refresh_clean(page)

    def forget_disk(self, page: Page) -> None:
        """工程文件里这页的位置要被别的数据覆盖，不能再当副本用。"""
        page.disk_key = None
        page.dirty = True
        self._refresh_clean(page)

    def average_blob_bytes(self, fmt: str = "rgba8") -> float:
        """内存副本的平均大小（估算撤销历史占用用）。"""
        if self.blob_count > 64:
            return self.ram_bytes / self.blob_count
        return self.pools[fmt].page_bytes * 0.35

    # ------------------------------------------------------------------ 换入
    def _decode(self, blob: bytes, codec: int, fmt: str) -> bytes:
        if codec == LZ4:
            return lz4.block.decompress(blob, uncompressed_size=self.pools[fmt].page_bytes)
        if codec == CONST:
            return blob * (self.pools[fmt].page_bytes // len(blob))
        return blob

    @staticmethod
    def _const_value(blob: bytes) -> int:
        return int.from_bytes(blob.ljust(4, bytes(1)), "little")

    def _read_raw(self, page: Page, count_sync: bool = True) -> bytes:
        """同步取一页的原始字节（不在显存时）：内存副本、暂存文件或工程文件。"""
        if page.blob is not None:
            return self._decode(page.blob, page.codec, page.fmt)
        if page.scratch is not None and self.scratch is not None:
            offset, nbytes, codec = page.scratch
            return self._decode(self.scratch.read(offset, nbytes), codec, page.fmt)
        if page.disk_key is not None and self.storage is not None:
            data = self.storage.read_page(page.disk_key)
            if data is None:
                raise RuntimeError("工程文件里找不到页面 %r" % (page.disk_key,))
            if count_sync:
                self.stat_sync_disk_reads += 1
            return data
        raise RuntimeError("页面数据丢失")

    def _write_slot(self, page: Page, raw: bytes) -> None:
        self.pools[page.fmt].write(page.slot, raw)
        page.want = False
        self.stat_uploaded += 1

    def make_resident(self, pages: list[Page]) -> None:
        """立刻把这些页放进显存（同步）。只在必须马上完成的场合用，平时用 request()。
        整页一个值的页攒起来，每种格式一次填完（刚分到槽的页这一帧不会被换出，等到最后再填是安全的）。"""
        consts: dict = {}
        for page in pages:
            if page is None or page.slot >= 0 or page.dead:
                continue
            if page.blob is not None and page.codec == CONST and self.packer is not None:
                self._attach_slot(page, block=True)
                slots, values = consts.setdefault(page.fmt, ([], []))
                slots.append(page.slot)
                values.append(self._const_value(page.blob))
                page.want = False
                self.stat_uploaded += 1
                continue
            raw = self._read_raw(page)
            self._attach_slot(page, block=True)
            self._write_slot(page, raw)
        for fmt, (slots, values) in consts.items():
            self.packer.fill_packed(fmt, slots, values)

    def try_resident(self, pages: list[Page]) -> bool:
        """不等待：在显存的标记为刚用过，不在的发起后台读取。全部已在显存时返回 True。"""
        ready = True
        missing = []
        for page in pages:
            if page is None or page.dead:
                continue
            if page.slot >= 0:
                self.pools[page.fmt].stamp[page.slot] = self.frame
            else:
                ready = False
                missing.append(page)
        if missing:
            self.request(missing, urgent=True)
        return ready

    def request(self, pages: list[Page], urgent: bool = False) -> None:
        """异步换入：解压、读暂存文件、读工程文件都在后台线程做，到货后在 service() 里上传。"""
        for page in pages:
            if page.slot >= 0 or page.want or page.dead:
                continue
            if page.blob is None and page.scratch is None and (page.disk_key is None or self.storage is None):
                continue
            page.want = True
            if urgent:
                self._requested.appendleft(page)
            else:
                self._requested.append(page)
        self._start_loads()

    def _start_loads(self) -> None:
        while self._requested and len(self._loading) + len(self._upload_queue) < self.MAX_LOADING:
            page = self._requested.popleft()
            if page.dead or page.slot >= 0:
                page.want = False
                continue
            if page.blob is not None and page.codec == CONST and self.packer is not None:
                self._fill_queue.append(page)
                continue
            if page.blob is not None:
                future = self.workers.submit(self._load_blob, page.blob, page.codec, page.fmt)
            elif page.scratch is not None and self.scratch is not None:
                offset, nbytes, codec = page.scratch
                future = self.workers.submit(self._load_scratch, offset, nbytes, codec, page.fmt)
            elif page.disk_key is not None and self.storage is not None:
                future = self.workers.submit(self._load_disk, page.disk_key)
                self.stat_disk_reads += 1
            else:
                page.want = False
                continue
            self._loading.append((page, future))

    def _load_blob(self, blob: bytes, codec: int, fmt: str):
        return self._decode(blob, codec, fmt), None

    def _load_scratch(self, offset: int, nbytes: int, codec: int, fmt: str):
        return self._decode(self.scratch.read(offset, nbytes), codec, fmt), None

    def _load_disk(self, key):
        raw = self.storage.read_page(key)
        if raw is None:
            return None, None
        return raw, lz4.block.compress(raw, mode="fast", store_size=False)

    @staticmethod
    def _compress(view) -> bytes:
        return lz4.block.compress(view, mode="fast", store_size=False)

    @staticmethod
    def _compress_many(views: list) -> list[bytes]:
        return [lz4.block.compress(view, mode="fast", store_size=False) for view in views]

    # ------------------------------------------------------------------ 换出
    def _drop_slot(self, page: Page) -> None:
        self.pools[page.fmt].release(page.slot)
        page.slot = -1
        self._sync_home(page)
        self.stat_evicted += 1

    def _release_clean(self, fmt: str, count: int, min_age: int = 1) -> int:
        """立刻释放最久没用、已有副本的页。返回释放了几个。"""
        pool = self.pools[fmt]
        released = 0
        for slot in pool.victims(count, self.frame, min_age=min_age, clean_only=True):
            page = pool.owner[int(slot)]
            if page is None or page.busy:
                continue
            self._drop_slot(page)
            released += 1
        return released

    def _make_room(self, fmt: str, count: int) -> None:
        """同步腾出槽位（兜底：没有已备份的页可放时才走到这里，要等显卡读回）。"""
        pool = self.pools[fmt]
        victims = pool.victims(count, self.frame, min_age=1)
        if len(victims) == 0:
            return
        self.stat_blocking_evictions += 1
        chosen = [pool.owner[int(slot)] for slot in victims]
        chosen = [page for page in chosen if page is not None and not page.busy]
        need = [page for page in chosen if not pool.clean[page.slot]]
        for page, data in zip(need, self.fetch_bytes(need)):
            self.adopt_bytes(page, data)
        for page in chosen:
            self._drop_slot(page)

    def _start_readback(self, pages: list[Page], drop: bool) -> int:
        """把这些页异步读回内存。drop 为真时读完释放槽位（换出），否则只是备份。返回放进这一批的页数。"""
        if not pages:
            return 0
        if not self.async_readback:
            for page, data in zip(pages, self.fetch_bytes(pages)):
                self.adopt_bytes(page, data)
                if drop:
                    self._drop_slot(page)
            return len(pages)
        if not self._free_buffers:
            return 0
        index = self._free_buffers.pop()
        buffer = self._buffers[index]
        if buffer is None:
            buffer = self._buffers[index] = glx.PersistentBuffer(self.BATCH_BYTES)
        batch = _Batch(index, drop)
        offset = 0
        placed = []
        for page in pages:
            pool = self.pools[page.fmt]
            if offset + pool.page_bytes > self.BATCH_BYTES:
                break
            pool.pins[page.slot] += 1
            page.busy = True
            batch.items.append((page, offset, pool.page_bytes))
            placed.append((page, offset))
            offset += pool.page_bytes
        self._issue_reads(buffer, placed)
        batch.fence = glx.Fence()
        self._batches.append(batch)
        return len(batch.items)

    def _issue_reads(self, buffer: glx.PersistentBuffer, placed: list) -> None:
        """发起把这些页读进 buffer 的显卡命令。placed：[(页, 缓冲内偏移)]。"""
        if self.packer is not None:
            groups: dict[str, tuple[list, list]] = {}
            for page, offset in placed:
                slots, offsets = groups.setdefault(page.fmt, ([], []))
                slots.append(page.slot)
                offsets.append(offset)
            for fmt, (slots, offsets) in groups.items():
                self.packer.pack(fmt, slots, offsets, buffer.handle)
            return
        for page, offset in placed:
            pool = self.pools[page.fmt]
            glx.read_layer_into(buffer, offset, pool.texture(page.slot).glo, pool.layer_of(page.slot), PAGE, PAGE,
                                pool.gl_format, pool.gl_type, pool.page_bytes)

    def evict_some(self, fmt: str, need: int) -> None:
        """换出 need 个槽：已有副本的立刻释放，其余的分批发起异步读回。"""
        pool = self.pools[fmt]
        victims = pool.victims(need, self.frame)
        limit = self.BATCH_BYTES // pool.page_bytes
        pending: list[Page] = []
        for slot in victims:
            page = pool.owner[int(slot)]
            if page is None or page.busy:
                continue
            if pool.clean[int(slot)]:
                self._drop_slot(page)
            else:
                pending.append(page)
        for start in range(0, len(pending), limit):
            if not self._free_buffers:
                break
            self._start_readback(pending[start:start + limit], drop=True)

    def _backup(self, budget_bytes: int) -> None:
        """后台备份：把还没有副本的页提前读回内存，以后腾位置时不用等。"""
        if budget_bytes <= 0 or not self.async_readback:
            return
        spare = len(self._free_buffers) - self.KEEP_FOR_EVICTION
        if spare <= 0 or (self.ram_bytes > self.ram_budget and not self._scratch_ready()):
            return
        for pool in self.pools.pools.values():
            if spare <= 0 or budget_bytes <= 0 or pool.capacity == 0:
                break
            per_batch = self.BATCH_BYTES // pool.page_bytes
            want = min(spare * per_batch, max(1, budget_bytes // pool.page_bytes))
            slots = pool.unbacked(want)
            pages = []
            for slot in slots:
                if pool.stamp[int(slot)] > self.frame - 2:
                    continue
                page = pool.owner[int(slot)]
                if page is not None and not page.busy and not page.dead and id(page) not in self._checking_ids:
                    pages.append(page)
            for start in range(0, len(pages), per_batch):
                if spare <= 0 or budget_bytes <= 0:
                    break
                count = self._start_readback(pages[start:start + per_batch], drop=False)
                spare -= 1
                budget_bytes -= count * pool.page_bytes

    def request_backup(self, pages) -> int:
        """尽快把这些只在显存里的页读回内存（自动保存要用，不等空闲时的后台备份）。返回这次发起读回的页数。"""
        todo: dict[str, list[Page]] = {}
        for page in pages:
            if page.slot >= 0 and page.blob is None and page.scratch is None and not page.busy and not page.dead:
                todo.setdefault(page.fmt, []).append(page)
        issued = 0
        for fmt, items in todo.items():
            if not self.async_readback:
                for page, data in zip(items, self.fetch_bytes(items)):
                    self.adopt_bytes(page, data)
                issued += len(items)
                continue
            per_batch = max(1, self.BATCH_BYTES // self.pools[fmt].page_bytes)
            start = 0
            while start < len(items) and len(self._free_buffers) > self.KEEP_FOR_EVICTION:
                count = self._start_readback(items[start:start + per_batch], drop=False)
                if count <= 0:
                    break
                issued += count
                start += count
        return issued

    # ------------------------------------------------------------------ 纯色检测
    def note_new_pages(self, pages) -> None:
        """刚在显卡上生成的页：排队检查是不是纯色。纯色页只记一个值，换出时不用读回。"""
        if self.packer is None or not self.async_readback:
            return
        for page in pages:
            if page is not None and page.blob is None:
                self._to_check.append(page)
                self._checking_ids.add(id(page))

    def _dispatch_checks(self) -> None:
        while self._to_check and self._check_free:
            chunk = []
            while self._to_check and len(chunk) < CHECK_PAGES:
                page = self._to_check.pop()
                if page.dead or page.slot < 0 or page.blob is not None:
                    self._checking_ids.discard(id(page))
                    continue
                chunk.append(page)
            if not chunk:
                break
            index = self._check_free.pop()
            buffer = self._check_buffers[index]
            if buffer is None:
                buffer = self._check_buffers[index] = glx.PersistentBuffer(CHECK_PAGES * 8)
            groups: dict[str, tuple[list, list]] = {}
            for position, page in enumerate(chunk):
                slots, offsets = groups.setdefault(page.fmt, ([], []))
                slots.append(page.slot)
                offsets.append(position * 2)
            for fmt, (slots, offsets) in groups.items():
                self.packer.check_uniform(fmt, slots, offsets, buffer.handle)
            self._checking.append((chunk, index, glx.Fence()))

    def _collect_checks(self) -> None:
        remaining = []
        for chunk, index, fence in self._checking:
            if not fence.ready():
                remaining.append((chunk, index, fence))
                continue
            fence.release()
            words = np.frombuffer(self._check_buffers[index].memory, np.uint32, count=2 * len(chunk)).copy()
            for page, flag, value in zip(chunk, words[0::2].tolist(), words[1::2].tolist()):
                self._checking_ids.discard(id(page))
                if flag and not page.dead and page.blob is None:
                    self._set_blob(page, int(value).to_bytes(4, "little")[:TEXEL_BYTES[page.fmt]], CONST)
                    self.stat_uniform += 1
            self._check_free.append(index)
        self._checking = remaining

    def _process_fills(self) -> None:
        groups: dict[str, tuple[list, list]] = {}
        queue = self._fill_queue
        while queue:
            page = queue[0]
            if page.dead or page.slot >= 0 or page.blob is None or page.codec != CONST:
                queue.popleft()
                page.want = False
                if not page.dead and page.slot < 0 and page.blob is not None:
                    self.request([page])
                continue
            if not self._attach_slot(page, block=False):
                break
            queue.popleft()
            slots, values = groups.setdefault(page.fmt, ([], []))
            slots.append(page.slot)
            values.append(self._const_value(page.blob))
            page.want = False
            self.stat_uploaded += 1
        for fmt, (slots, values) in groups.items():
            self.packer.fill_packed(fmt, slots, values)

    def has_background_work(self) -> bool:
        """还有没备份的页、且现在能备份。"""
        if not self.async_readback or (self.ram_bytes > self.ram_budget and not self._scratch_ready()):
            return False
        return any(pool.unbacked_count() > 0 for pool in self.pools.pools.values() if pool.capacity)

    # ------------------------------------------------------------------ 暂存文件
    def _scratch_ready(self) -> bool:
        if self.scratch is not None:
            return True
        if self._scratch_failed:
            return False
        directory = self.scratch_dir
        if not directory:
            from ..paths import user_dir
            directory = os.path.join(str(user_dir()), "scratch")
        try:
            self.scratch = ScratchFile(directory)
            log.info("内存缓存超出预算，开始使用暂存文件：%s", self.scratch.path)
        except OSError:
            log.exception("无法创建暂存文件，超出内存预算的页面只能留在内存里")
            self._scratch_failed = True
            return False
        return True

    def _spill(self) -> None:
        """内存副本超出预算：已有别处副本的直接放掉，其余最早的写进暂存文件。"""
        excess = self.ram_bytes - self.ram_budget
        if excess <= 0:
            return
        order = self._blob_order
        checked = 0
        while order and excess > 0 and len(self._spilling) < 64 and checked < 4096:
            page = order.popleft()
            checked += 1
            if page.dead or page.blob is None or id(page) in self._spilling_ids:
                continue
            if page.codec == CONST:
                continue
            if page.codec != LZ4:
                order.append(page)
                continue
            nbytes = len(page.blob)
            if page.scratch is not None or page.disk_key is not None:
                self._drop_blob(page)
                excess -= nbytes
                continue
            if not self._scratch_ready():
                order.appendleft(page)
                return
            offset = self.scratch.allocate(nbytes)
            future = self.workers.submit(self.scratch.write, offset, page.blob)
            self._spilling.append((page, page.blob, offset, nbytes, future))
            self._spilling_ids.add(id(page))
            excess -= nbytes

    def _finish_spills(self) -> None:
        waiting = []
        for item in self._spilling:
            page, blob, offset, nbytes, future = item
            if not future.done():
                waiting.append(item)
                continue
            self._spilling_ids.discard(id(page))
            try:
                future.result()
            except Exception:  # noqa: BLE001
                log.exception("写暂存文件失败")
                self.scratch.free(offset, nbytes)
                continue
            if page.dead or page.blob is not blob or page.scratch is not None:
                self.scratch.free(offset, nbytes)
                continue
            page.scratch = (offset, nbytes, LZ4)
            self._drop_blob(page)
            self.stat_spilled += 1
        self._spilling = waiting

    # ------------------------------------------------------------------ 每帧维护
    def service(self, frame: int, budget_ms: float = 4.0, idle: bool = False, background: bool = True) -> None:
        """一次做完每帧的全部维护（导出、测试用）。引擎在一帧里分两次调用 begin() 和 maintain()。"""
        self.begin(frame, budget_ms)
        self.maintain(idle, background)

    def begin(self, frame: int, budget_ms: float = 4.0) -> None:
        """一帧开始：收回读回和读取的结果，把到货的页上传（限时）。

        帧号只往前走：同步的大操作（整层滤镜、导出等）会自己往前推帧，之后引擎给的帧号可能落在后面；
        要是倒回去，那段时间用过的页会一直算「刚用过」，换不出去，下一个大操作就把页池挤满。"""
        if frame <= self.frame:
            frame = self.frame + 1
        self.frame = frame
        start = time.perf_counter()
        # 1. 读回完成的批次：压缩交给工作线程（每组若干页）
        for batch in self._batches:
            if not batch.futures and batch.fence.ready():
                batch.fence.release()
                buffer = self._buffers[batch.buffer]
                items = batch.items
                for first in range(0, len(items), COMPRESS_GROUP):
                    group = items[first:first + COMPRESS_GROUP]
                    views = [buffer.view(offset, nbytes) for _page, offset, nbytes in group]
                    batch.futures.append(([page for page, _o, _n in group],
                                          self.workers.submit(self._compress_many, views)))
        # 2. 压缩完成：记下副本；换出的批次释放槽位
        if self._batches:
            remaining = []
            for batch in self._batches:
                if not batch.futures or not all(f.done() for _p, f in batch.futures):
                    remaining.append(batch)
                    continue
                for group, future in batch.futures:
                    for page, blob in zip(group, future.result()):
                        pool = self.pools[page.fmt]
                        page.busy = False
                        if page.slot >= 0:
                            pool.pins[page.slot] = max(0, pool.pins[page.slot] - 1)
                        if page.dead:
                            if page.slot >= 0:
                                pool.release(page.slot)
                                page.slot = -1
                            continue
                        self._set_blob(page, blob, LZ4)
                        if not batch.drop:
                            self.stat_backed_up += 1
                        elif pool.pins[page.slot] == 0 and pool.stamp[page.slot] < frame - 1:
                            self._drop_slot(page)
                self._free_buffers.append(batch.buffer)
            self._batches = remaining
        # 3. 后台读取到货：排队上传
        if self._loading:
            remaining_loads = []
            for page, future in self._loading:
                if not future.done():
                    remaining_loads.append((page, future))
                    continue
                if page.dead:
                    page.want = False
                    continue
                try:
                    raw, blob = future.result()
                except Exception:  # noqa: BLE001
                    log.exception("读取页面失败")
                    page.want = False
                    continue
                if raw is None:
                    page.want = False
                    continue
                if blob is not None and page.blob is None:
                    self._set_blob(page, blob, LZ4)
                if page.slot >= 0:
                    page.want = False
                    continue
                self._upload_queue.append((page, raw))
            self._loading = remaining_loads
        # 纯色检测结果、纯色页换入（在显卡上填色，不花上传时间）
        if self._checking:
            self._collect_checks()
        if self._fill_queue:
            self._process_fills()
        # 4. 上传到货的页（限时；没有空槽就等下一帧）
        done = 0
        queue = self._upload_queue
        while queue:
            if done >= 8 and (time.perf_counter() - start) * 1000.0 > budget_ms:
                break
            page, raw = queue[0]
            if page.dead or page.slot >= 0:
                queue.popleft()
                page.want = False
                continue
            if not self._attach_slot(page, block=False):
                break
            queue.popleft()
            self._write_slot(page, raw)
            done += 1
        self._start_loads()
        # 5. 兜底读回留下的未压缩副本，压缩好了就换上
        if self._recompress:
            pending = []
            for page, raw, future in self._recompress:
                if not future.done():
                    pending.append((page, raw, future))
                elif not page.dead and page.blob is raw:
                    self._set_blob(page, future.result(), LZ4)
            self._recompress = pending
        if self._spilling:
            self._finish_spills()

    def maintain(self, idle: bool = False, background: bool = True) -> None:
        """一帧结束（画面已经交给显卡之后）：保持空闲水位、内存超预算时落盘、后台备份。

        这些都会让显卡做读回，排在本帧渲染之后，不耽误本帧画面。
        """
        if self._to_check:
            self._dispatch_checks()
        waiting = {}
        for page, _raw in self._upload_queue:
            waiting[page.fmt] = waiting.get(page.fmt, 0) + 1
        for fmt, pool in self.pools.pools.items():
            if pool.capacity == 0:
                continue
            target = max(32, int(self.watermark * pool.capacity)) + int(self.reserve.get(fmt, 0)) + waiting.get(fmt, 0)
            lack = target - len(pool.free) - pool.growable_slots()
            if lack > 0:
                self.evict_some(fmt, min(lack, 2048))
        if not background:
            return
        if idle:
            for pool in self.pools.pools.values():      # 每帧最多提前加一张数组
                if pool.grow_ahead():
                    break
        self._spill()
        self._backup(self.backup_bytes_idle if idle else self.backup_bytes_active)

    @property
    def busy(self) -> bool:
        return bool(self._batches or self._loading or self._upload_queue or self._recompress or self._spilling
                    or self._requested or self._fill_queue or self._checking or self._to_check)

    # ------------------------------------------------------------------ 保存
    def page_bytes(self, page: Page) -> bytes:
        """取一页的原始字节（同步）。"""
        return self.fetch_bytes([page])[0]

    def fetch_bytes(self, pages: list[Page]) -> list[bytes]:
        """同步取一批页的原始字节。只在显存里的那些走批量读回，一批等一次。保存、导出、兜底换出时用。"""
        out: list = [None] * len(pages)
        on_gpu: list[int] = []
        for index, page in enumerate(pages):
            if page.blob is not None:
                out[index] = self._decode(page.blob, page.codec, page.fmt)
            elif page.slot >= 0:
                on_gpu.append(index)
            else:
                out[index] = self._read_raw(page, count_sync=False)
        if not on_gpu:
            return out
        if not self.async_readback:
            for index in on_gpu:
                page = pages[index]
                out[index] = self.pools[page.fmt].read_sync(page.slot)
            return out
        if self._sync_buffer is None:
            self._sync_buffer = glx.PersistentBuffer(self.BATCH_BYTES)
        buffer = self._sync_buffer
        cursor = 0
        while cursor < len(on_gpu):
            offset = 0
            batch: list[tuple[int, int, int]] = []
            placed = []
            while cursor < len(on_gpu):
                index = on_gpu[cursor]
                page = pages[index]
                pool = self.pools[page.fmt]
                if offset + pool.page_bytes > self.BATCH_BYTES:
                    break
                placed.append((page, offset))
                batch.append((index, offset, pool.page_bytes))
                offset += pool.page_bytes
                cursor += 1
            self._issue_reads(buffer, placed)
            fence = glx.Fence()
            while not fence.ready(wait_ns=200_000_000):
                pass
            fence.release()
            for index, start, nbytes in batch:
                out[index] = bytes(buffer.view(start, nbytes))
        return out

    def wait_idle(self, timeout: float = 10.0) -> None:
        """等所有在途的读回、读取、落盘结束（退出、保存前用）。不再发起新的后台备份。"""
        deadline = time.perf_counter() + timeout
        while self.busy and time.perf_counter() < deadline:
            glx.flush()
            self.service(self.frame + 1, budget_ms=50.0, background=False)
            time.sleep(0.001)

    def stats(self) -> dict:
        info = self.pools.stats()
        info.update(ram_mb=self.ram_bytes / 1048576.0, ram_budget_mb=self.ram_budget >> 20, evicted=self.stat_evicted,
                    uploaded=self.stat_uploaded, blocking_evictions=self.stat_blocking_evictions,
                    disk_reads=self.stat_disk_reads, sync_disk_reads=self.stat_sync_disk_reads,
                    backed_up=self.stat_backed_up, spilled=self.stat_spilled, uniform=self.stat_uniform,
                    scratch_mb=(self.scratch.used / 1048576.0) if self.scratch is not None else 0.0,
                    unbacked=sum(p.unbacked_count() for p in self.pools.pools.values() if p.capacity),
                    readbacks_in_flight=len(self._batches), upload_queue=len(self._upload_queue),
                    loading=len(self._loading) + len(self._requested))
        return info

    def release(self) -> None:
        self.wait_idle(3.0)
        self.workers.shutdown(wait=True, cancel_futures=True)
        for buffer in self._buffers:
            if buffer is not None:
                buffer.release()
        self._buffers = [None] * self.RING
        if self._sync_buffer is not None:
            self._sync_buffer.release()
            self._sync_buffer = None
        for buffer in self._check_buffers:
            if buffer is not None:
                buffer.release()
        self._check_buffers = [None] * 4
        if self.scratch is not None:
            self.scratch.close()
            self.scratch = None
