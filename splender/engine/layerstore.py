"""图层的像素：每个图层若干种页的稀疏金字塔，以及把笔划并入图层、撤销重做。

页面写好后不再修改。并入笔划时为受影响的格子生成新页，页表指向新页，旧页留给撤销。
撤销就是把页表指回旧页。

并入笔划分几帧做（MergeJob）：做完之前页表不动，显示仍是「旧图层 + 笔划叠加」，
全部做完再一次换上新页（只改指针）。所以抬笔时画面不会卡。
"""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from ..doc.project import PLANE_COLOR, PLANE_FORMAT, PLANE_HEIGHT, PLANE_MASK, PLANE_MR
from .cache import LevelGrid, Page, PageCache, PlaneStore
from .pageops import KIND_COLOR, KIND_HEIGHT, KIND_MASK, KIND_MR, PageOps
from .stamp import StrokeBuffer

PLANE_KIND = {PLANE_COLOR: KIND_COLOR, PLANE_MR: KIND_MR, PLANE_HEIGHT: KIND_HEIGHT, PLANE_MASK: KIND_MASK}

class RecordGroup:
    """一组撤销记录：同一张页表（同一图层、同一种页、同一级）里若干格的旧页和新页。

    按组存而不是每格一个元组：对象少，垃圾回收扫描快。
    """

    __slots__ = ("level", "mip", "xs", "ys", "olds", "news")

    def __init__(self, level: LevelGrid, mip: int, xs, ys, olds: list, news: list) -> None:
        self.level = level
        self.mip = mip
        self.xs = np.asarray(xs, np.int32)
        self.ys = np.asarray(ys, np.int32)
        self.olds = olds
        self.news = news

    def __len__(self) -> int:
        return len(self.news)

    def cells(self):
        return zip(self.xs.tolist(), self.ys.tolist(), self.olds, self.news)


#: 一步撤销的全部记录
Record = RecordGroup


class LayerStore:
    """一个图层的全部像素。"""

    def __init__(self, grid: int) -> None:
        self.grid = grid
        self.planes: dict[int, PlaneStore] = {}

    def plane(self, plane: int, create: bool = True) -> PlaneStore | None:
        store = self.planes.get(plane)
        if store is None and create:
            store = self.planes[plane] = PlaneStore(PLANE_FORMAT[plane], self.grid)
        return store

    def pages(self):
        for plane, store in self.planes.items():
            for mip, level in enumerate(store.levels):
                for (tx, ty), page in level.pages.items():
                    yield plane, mip, tx, ty, page

    def page_count(self) -> int:
        return sum(store.page_count() for store in self.planes.values())


class StrokeParams:
    """并入笔划时用到的数值快照。"""

    def __init__(self, *, color=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.5, height=0.0, mask_value=1.0,
                 use_basecolor=True, use_metallic=True, use_roughness=True, use_height=False, opacity=1.0,
                 erase=False, target_mask=False, mask_default=1.0) -> None:
        self.color = tuple(color)
        self.metallic = metallic
        self.roughness = roughness
        self.height = height
        self.mask_value = mask_value
        self.use_basecolor = use_basecolor
        self.use_metallic = use_metallic
        self.use_roughness = use_roughness
        self.use_height = use_height
        self.opacity = opacity
        self.erase = erase
        self.target_mask = target_mask
        self.mask_default = mask_default
        self.effect = ""                 # 减淡、加深……（engine/filters.py 的 EFFECTS）；空串是普通的画
        self.effect_values: dict = {}

    def planes(self) -> list[int]:
        if self.target_mask:
            return [PLANE_MASK]
        found = []
        if self.use_basecolor:
            found.append(PLANE_COLOR)
        if self.use_metallic or self.use_roughness:
            found.append(PLANE_MR)
        if self.use_height:
            found.append(PLANE_HEIGHT)
        return found

    @property
    def values(self) -> tuple:
        return (self.metallic, self.roughness, self.height, self.mask_value)

    @property
    def enable(self) -> tuple:
        return (int(self.use_metallic), int(self.use_roughness), 0, 0)


class MergeJob:
    """把一笔并进一个图层，分几帧做完。

    做完之前页表不变；新页先记在 staged 里，commit() 时一次换上。
    需要的旧页不在显存时发起后台读取，等它们到了再做；新页放不下时也等下一帧，后台会腾出位置。
    """

    CHUNK = 256          # 第 0 级一次处理多少格；上面各级每格要读 4 个子页，一次处理四分之一
    STALL_FRAMES = 20    # 连续等这么多帧还拿不到资源，就改为同步完成这一小块，保证一定能做完

    def __init__(self, pixels: "LayerPixels", layer_uid: int, grid: int, stroke: StrokeBuffer,
                 params: StrokeParams) -> None:
        self.cache = pixels.cache
        self.ops = pixels.ops
        self.stroke = stroke
        self.params = params
        self.store = pixels.store(layer_uid, grid)
        self.records: list[Record] = []
        self.staged: dict[tuple[int, int], dict] = {}
        self.queue: deque = deque()
        self.stall = 0
        self.pages_done = 0
        self.mask_default = float(params.mask_default)      # 蒙版缺页的格按这个值算（生成上面各级时也要用）
        self.formats = sorted({PLANE_FORMAT[p] for p in params.planes()})
        tiles = stroke.touched_tiles()
        if len(tiles):
            tiles = tiles[stroke.store.levels[0].slots[tiles // grid, tiles % grid] >= 0]
        for plane in params.planes():
            selected = tiles
            if params.erase and len(tiles):
                exists = self.store.plane(plane).levels[0].exists
                selected = tiles[exists[tiles // grid, tiles % grid]]
            if len(selected):
                self.queue.append([plane, 0, np.asarray(selected, np.int64), 0])

    @property
    def done(self) -> bool:
        return not self.queue

    def remaining_tiles(self) -> int:
        return sum(len(item[2]) - item[3] for item in self.queue)

    def step(self, budget_ms: float = 4.0, max_pages: int = 1024, block: bool = False) -> bool:
        """推进一点。返回是否全部做完（做完后调用 commit）。block 为真时一次做完。"""
        started = time.perf_counter()
        processed = 0
        while self.queue:
            if not block and (processed >= max_pages or (time.perf_counter() - started) * 1000.0 >= budget_ms):
                return False
            item = self.queue[0]
            plane, mip, tiles, cursor = item
            chunk = self.CHUNK if mip == 0 else max(16, self.CHUNK // 4)
            if not block:
                chunk = max(1, min(chunk, max_pages - processed))
            forced = block or self.stall >= self.STALL_FRAMES
            if forced and not block:
                chunk = max(1, chunk // 4)
            count = self._process(plane, mip, tiles[cursor:cursor + chunk], forced)
            if count == 0:
                self.stall += 1
                return False
            self.stall = 0
            item[3] += count
            processed += count
            self.pages_done += count
            if item[3] >= len(tiles):
                self.queue.popleft()
                levels = self.store.plane(plane).levels
                if mip + 1 < len(levels):
                    size = levels[mip].size
                    parents = np.unique(((tiles // size) >> 1) * (size >> 1) + ((tiles % size) >> 1))
                    self.queue.appendleft([plane, mip + 1, parents, 0])
        self._clear_reserve()
        return True

    def _clear_reserve(self) -> None:
        for fmt in self.formats:
            self.cache.reserve[fmt] = 0

    def _process(self, plane: int, mip: int, chunk: np.ndarray, block: bool) -> int:
        cache = self.cache
        kind = PLANE_KIND[plane]
        fmt = PLANE_FORMAT[plane]
        here = self.store.plane(plane).levels[mip]
        size = here.size
        xs = (chunk % size).tolist()
        ys = (chunk // size).tolist()
        olds = [here.pages.get((x, y)) for x, y in zip(xs, ys)]
        kids: list[tuple[int, int, Page]] = []
        if mip == 0:
            stroke_slots = self.stroke.store.levels[0].slots[chunk // size, chunk % size].astype(np.int32)
        else:
            below = self.staged.get((plane, mip - 1), {})
            for i, (x, y) in enumerate(zip(xs, ys)):
                for q in range(4):
                    page = below.get((2 * x + (q & 1), 2 * y + (q >> 1)))
                    if page is not None:
                        kids.append((i, q, page))
        sources = [p for p in olds if p is not None] + [k[2] for k in kids]
        if block:
            cache.make_resident(sources)
        elif not cache.try_resident(sources):
            return 0
        cache.pin_many(sources)
        try:
            news = cache.try_new_pages(fmt, len(chunk), block=block)
            count = len(news)
            cache.reserve[fmt] = len(chunk) if count < len(chunk) else 0
            if count == 0:
                return 0
            dst = np.fromiter((p.slot for p in news), np.int32, count)
            old = np.fromiter((p.slot if p is not None else -1 for p in olds[:count]), np.int32, count)
            if mip == 0:
                p = self.params
                self.ops.apply_stroke(kind, dst, old, stroke_slots[:count], color=p.color, values=p.values,
                                      enable=p.enable, opacity=p.opacity, erase=p.erase, mask_default=p.mask_default)
            else:
                children = np.full((count, 4), -1, np.int32)
                for i, q, page in kids:
                    if i < count:
                        children[i, q] = page.slot
                self.ops.downsample(kind, dst, old, children, empty=self.mask_default)
            cache.note_new_pages(news)
            staged = self.staged.setdefault((plane, mip), {})
            for i in range(count):
                staged[(xs[i], ys[i])] = news[i]
            self.records.append(RecordGroup(here, mip, xs[:count], ys[:count], olds[:count], news))
            return count
        finally:
            cache.pin_many(sources, -1)

    def commit(self) -> list[Record]:
        """把页表换成新页。返回撤销记录。"""
        set_page = self.cache.set_page
        for group in self.records:
            level = group.level
            for x, y, _old, new in group.cells():
                set_page(level, x, y, new)
        self.staged.clear()
        self._clear_reserve()
        return self.records

    def cancel(self) -> None:
        """放弃：丢掉已经生成的新页，页表保持原样。"""
        for staged in self.staged.values():
            for page in staged.values():
                self.cache.free_page(page)
        self.staged.clear()
        self.records = []
        self.queue.clear()
        self._clear_reserve()


class LayerPixels:
    def __init__(self, cache: PageCache, ops: PageOps) -> None:
        self.cache = cache
        self.ops = ops
        self.stores: dict[int, LayerStore] = {}

    def store(self, layer_uid: int, grid: int) -> LayerStore:
        found = self.stores.get(layer_uid)
        if found is None:
            found = self.stores[layer_uid] = LayerStore(grid)
        return found

    # ------------------------------------------------------------------ 并入笔划
    def merge_job(self, layer_uid: int, grid: int, stroke: StrokeBuffer, params: StrokeParams) -> MergeJob:
        return MergeJob(self, layer_uid, grid, stroke, params)

    def merge_stroke(self, layer_uid: int, grid: int, stroke: StrokeBuffer, params: StrokeParams) -> list[Record]:
        """一次做完的并入（测试和必须立刻完成的场合用）。返回撤销记录。"""
        job = self.merge_job(layer_uid, grid, stroke, params)
        job.step(block=True)
        return job.commit()

    # ------------------------------------------------------------------ 撤销重做
    def apply_records(self, records: list[Record], undo: bool) -> None:
        set_page = self.cache.set_page
        for group in records:
            level = group.level
            for x, y, old, new in group.cells():
                set_page(level, x, y, old if undo else new)

    def dispose_records(self, records: list[Record], applied: bool) -> None:
        """步骤被丢弃：已生效的丢旧页，被撤销掉的丢新页。"""
        for group in records:
            for page in (group.olds if applied else group.news):
                self.cache.free_page(page)

    def records_bytes(self, records: list[Record]) -> int:
        """一步撤销大约占多少空间：旧页数量乘以内存副本的平均大小。"""
        return int(sum(len(group) for group in records) * self.cache.average_blob_bytes())

    @staticmethod
    def affected(records: list[Record]) -> dict[int, np.ndarray]:
        """记录涉及的格子：{级号: 格子编号数组}。"""
        found: dict[int, list] = {}
        for group in records:
            found.setdefault(group.mip, []).append(group.ys.astype(np.int64) * group.level.size + group.xs)
        return {mip: np.unique(np.concatenate(parts)) for mip, parts in found.items()}

    @staticmethod
    def record_pages(records: list[Record]):
        """记录里引用的全部页（旧页和新页）。"""
        for group in records:
            for page in group.olds:
                if page is not None:
                    yield page
            for page in group.news:
                if page is not None:                 # 去掉页的记录（选区）新页是空
                    yield page

    # ------------------------------------------------------------------ 整层操作
    def drop_layer(self, layer_uid: int) -> None:
        store = self.stores.pop(layer_uid, None)
        if store is None:
            return
        for _plane, _mip, _tx, _ty, page in list(store.pages()):
            self.cache.free_page(page)

    def detach_layer(self, layer_uid: int) -> LayerStore | None:
        """把图层的像素整体取下（删除图层时交给撤销步骤保管）。"""
        return self.stores.pop(layer_uid, None)

    def attach_layer(self, layer_uid: int, store: LayerStore) -> None:
        self.stores[layer_uid] = store

    def free_store(self, store: LayerStore | None) -> None:
        if store is None:
            return
        for _plane, _mip, _tx, _ty, page in list(store.pages()):
            self.cache.free_page(page)

    def duplicate_layer(self, source_uid: int, target_uid: int) -> None:
        """复制图层：内存副本直接共用（页面不可变），只在显存里的页用显卡拷贝。"""
        source = self.stores.get(source_uid)
        if source is None:
            return
        target = self.store(target_uid, source.grid)
        cache = self.cache
        for plane, mip, tx, ty, page in list(source.pages()):
            cache.set_page(target.plane(plane).levels[mip], tx, ty, cache.clone_page(page))
