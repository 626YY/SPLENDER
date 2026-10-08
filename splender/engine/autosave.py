"""自动保存：把没保存的改动定时另存一份「恢复文件」，程序意外退出后能从它恢复。

恢复文件和工程文件同样的格式（doc.storage.ProjectFile），放在自动保存目录里，不动工程文件本身：
- 工程有文件时只写相对那个文件改过的东西：改过的页、改过的模型和模型贴图、工程信息，另记下哪些页、哪些资源已经不要了；
- 还没存过的工程整份写。
同一个工程里恢复文件一直开着，每次只补写上次自动保存以后又变了的页。页面一旦写好就不再原地修改（改动产生新页），
所以「这一格还是上次写进去的那个页对象」就说明它没变（页上记着写进去时的编号，和恢复文件里这一格的编号对得上才算）。

取页面数据在主线程按帧分摊，每帧有时间预算：有内存副本（或暂存文件、工程文件里的副本）的直接取；
只在显存里的请缓存在后台读回，读回来了再取。解压、压缩、写盘都在写盘线程里做。
一次自动保存是恢复文件里的一个事务：中途崩溃时恢复文件停在上一次完整的状态。

恢复（restore）：把原工程文件完整复制一份，再把恢复文件里的页、资源、工程信息盖上去，得到一个普通的工程文件。
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from collections import deque
from functools import partial

import lz4.block

from ..core.recovery import file_stat, remove_quiet
from ..doc.storage import ProjectFile, ProjectFileError
from . import glx
from .cache import CONST, LZ4
from .projectio import FORMAT_VERSION, mesh_to_bytes

log = logging.getLogger("splender.engine.autosave")

RECOVER_FORMAT = 1
#: 主线程已经读出来、写盘线程还没写完的新字节上限（超过就等写盘线程赶上来）
MAX_INFLIGHT_BYTES = 96 << 20
#: 每批交给写盘线程的页数
BATCH_PAGES = 32
#: 恢复文件的日志模式（回滚日志：一次写几 GB 的大事务比 WAL 快好几倍）。读它的地方也要用同样的模式打开
RECOVER_JOURNAL = "TRUNCATE"


# ================================================================== 写盘线程
class _Writer(threading.Thread):
    """恢复文件只在这个线程里读写；命令按先后执行。一次自动保存 = begin … commit 之间的一个事务。"""

    def __init__(self) -> None:
        super().__init__(name="splender-autosave", daemon=True)
        self.commands: queue.Queue = queue.Queue()
        self.file: ProjectFile | None = None
        self.path: str | None = None
        self._tx = None
        self._gen = 0                  # 当前事务属于哪一次自动保存
        self._failed_gen = -1          # 出错的那一次，它后面的命令都跳过
        self.committed_gen = 0
        self.failed: tuple[int, BaseException] | None = None
        self._lock = threading.Lock()
        self.inflight = 0              # 主线程新读出来、还没写完的字节
        self.gate = threading.Event()  # 关上时（正在画笔划）写完手上这一批就停下等，不和绘制抢处理器
        self.gate.set()
        self.start()

    # ---- 主线程调用 ----
    def put(self, *command) -> None:
        self.commands.put(command)

    def add_inflight(self, nbytes: int) -> None:
        with self._lock:
            self.inflight += int(nbytes)

    def sync(self, timeout: float = 60.0) -> bool:
        """等前面的命令都做完。"""
        event = threading.Event()
        self.commands.put(("sync", event))
        return event.wait(timeout)

    def stop(self, timeout: float = 60.0) -> None:
        self.commands.put(None)
        self.join(timeout)

    # ---- 写盘线程 ----
    def run(self) -> None:
        _lower_priority()
        while True:
            command = self.commands.get()
            if command is None:
                self._abort()
                self._close_file(False)
                return
            kind = command[0]
            if kind == "sync":
                command[1].set()
                continue
            try:
                getattr(self, "_do_" + kind)(*command[1:])
            except BaseException as exc:  # noqa: BLE001
                log.exception("自动保存写盘出错")
                gen = self._gen
                self._abort()
                self._failed_gen = gen
                self.failed = (gen, exc)

    def _skip(self, gen: int) -> bool:
        return gen != self._gen or gen == self._failed_gen or self._tx is None

    def _do_open(self, path: str, level: int, threads: int) -> None:
        self._abort()
        self._close_file(False)
        remove_quiet(path)
        # 崩溃时没写完的那次照样整体作废（下次打开时按日志撤回）
        self.file = ProjectFile(path, create=True, level=level, threads=threads, synchronous="FULL",
                                journal=RECOVER_JOURNAL)
        self.path = path

    def _do_close(self, delete: bool) -> None:
        self._abort()
        self._close_file(delete)

    def _close_file(self, delete: bool) -> None:
        storage, path = self.file, self.path
        self.file = None
        self.path = None
        if storage is not None:
            try:
                storage.close()
            except Exception:  # noqa: BLE001
                log.exception("关闭恢复文件出错")
        if delete and path:
            remove_quiet(path)

    def _do_begin(self, gen: int) -> None:
        self._abort()
        self._gen = gen
        if self.file is None:
            raise ProjectFileError("恢复文件没有打开")
        tx = self.file.transaction()
        tx.__enter__()
        self._tx = tx

    def _do_pages(self, gen: int, items: list, nbytes: int, delete) -> None:
        try:
            if self._skip(gen):
                return
            self.gate.wait()
            self.file.write_pages(self._decoded(items), delete=delete)
        finally:
            with self._lock:
                self.inflight -= int(nbytes)

    @staticmethod
    def _decoded(items: list):
        for key, codec, data, nbytes in items:
            if codec == LZ4:
                yield key, lz4.block.decompress(data, uncompressed_size=nbytes)
            elif codec == CONST:
                yield key, data * (nbytes // len(data))
            else:
                yield key, data

    def _do_asset(self, gen: int, name: str, producer) -> None:
        if self._skip(gen):
            return
        data = producer() if callable(producer) else producer
        self.file.write_asset(name, data)

    def _do_drop_asset(self, gen: int, name: str) -> None:
        if not self._skip(gen):
            self.file.delete_asset(name)

    def _do_commit(self, gen: int, meta: dict) -> None:
        if self._skip(gen):
            return
        self.file.write_meta(meta)
        tx, self._tx = self._tx, None
        tx.__exit__(None, None, None)
        self.committed_gen = gen

    def _do_abort(self, gen: int) -> None:
        if gen == self._gen:
            self._abort()

    def _abort(self) -> None:
        tx, self._tx = self._tx, None
        if tx is not None:
            try:
                tx.__exit__(RuntimeError, RuntimeError("自动保存取消"), None)
            except Exception:  # noqa: BLE001
                log.exception("撤销没写完的自动保存时出错")


def _lower_priority() -> None:
    """写盘线程调低一档优先级：系统先照顾界面和绘制。"""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        kernel32.SetThreadPriority(kernel32.GetCurrentThread(), -1)     # THREAD_PRIORITY_BELOW_NORMAL
    except Exception:  # noqa: BLE001
        pass


# ================================================================== 主线程
class _Job:
    """一次进行中的自动保存。"""

    def __init__(self, gen: int, items: list, want: set, meta: dict, doc: str, assets: dict) -> None:
        self.gen = gen
        self.items: deque = deque(items)       # [(键, 页)] 还没取数据的
        self.waiting: list = []                # 只在显存里、等着读回的
        self.written: list = []                # [(页, 键, 编号)] 已经交给写盘线程的
        self.dropped: set = set()              # 中途不用再存的键（页被丢弃、又回到和工程文件一样）
        self.want = want                       # 这次之后恢复文件里应该有的键
        self.meta = meta
        self.doc = doc
        self.assets = assets                   # 这次之后恢复文件里的资源 {名字: 记号}
        self.batch: list = []
        self.batch_bytes = 0
        self.new_bytes = 0
        self.started = time.perf_counter()
        self.finishing = False
        self.pump_ms: list[float] = []          # 每帧在主线程花了多久（统计用）
        self.readbacks: list = []               # [(资源名, 模型贴图, 分步读回的生成器)]：读完才交给写盘线程


class Autosaver:
    """一个引擎一个。start() 在用户停手时由程序调用；之后每帧 pump()；做完回调 on_finished。"""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.writer: _Writer | None = None
        self.new_path = None                    # () -> 新恢复文件的路径；没给时放系统临时目录
        self.level = 3                          # zstd 压缩级别
        self.threads = 2                        # 压缩线程数
        self.path: str | None = None            # 当前恢复文件
        self.token: object | None = None        # 当前恢复文件的标记（页上的记号要对上它）
        self.rec_keys: dict[tuple, int] = {}    # 恢复文件里已有的页：键 -> 写入编号
        self.rec_assets: dict[str, tuple] = {}  # 恢复文件里已有的资源：名字 -> 记号
        self.last_doc: str | None = None        # 上次写进去的工程信息
        self.job: _Job | None = None
        self.last_result: dict | None = None
        self.last_error: str | None = None
        self.on_finished: list = []             # 回调 (结果字典)；出错时字典里有 "error"
        self._base = None                       # (工程文件路径, 页键集合, 资源名集合, (大小, 修改时间))
        self._pending_maps: list = []
        self._gen = 0
        self._serial = itertools.count(1)
        self._temp_index = 0

    # ---------------------------------------------------------------- 状态
    @property
    def active(self) -> bool:
        return self.job is not None

    def set_paused(self, paused: bool) -> None:
        """正在画笔划时暂停写盘（写完手上这一批就等），抬笔后接着写。引擎每帧调用。"""
        writer = self.writer
        if writer is None:
            return
        if paused:
            if writer.gate.is_set():
                writer.gate.clear()
        elif not writer.gate.is_set():
            writer.gate.set()

    def _ensure_writer(self) -> _Writer:
        if self.writer is None or not self.writer.is_alive():
            self.writer = _Writer()
        return self.writer

    def _fresh_path(self) -> str:
        if self.new_path is not None:
            return str(self.new_path())
        self._temp_index += 1
        return os.path.join(tempfile.gettempdir(),
                            "splender-%d-%d.splender-recover" % (os.getpid(), self._temp_index))

    # ---------------------------------------------------------------- 开始
    def start(self, project, extra: dict | None = None, info: dict | None = None) -> bool:
        """开始一次自动保存（主线程，用户停手、没有笔划和合并在进行时调用）。没有要存的就不开始，返回 False。

        extra：额外写进工程信息的内容（界面布局等）；info：额外写进恢复记录的内容（会话名等）。"""
        if self.job is not None or project is None:
            return False
        engine = self.engine
        storage = engine.cache.storage
        base_path = getattr(storage, "path", None) if storage is not None else None
        if self._base is None or self._base[0] != base_path:
            if self.path is not None:
                self._discard_file()                # 换了底（另存为等）：之前的差异没意义了
            if storage is not None:
                self._base = (base_path, frozenset(storage.page_keys()), frozenset(storage.list_assets()),
                              file_stat(base_path))
            else:
                self._base = (None, frozenset(), frozenset(), None)
        _path, base_keys, base_assets, base_stat = self._base
        if not project.dirty and self.path is None:
            return False
        token = self.token if self.token is not None else object()
        rec_keys = self.rec_keys if self.token is not None else {}
        # 页：和工程文件里一样的不存；恢复文件里已经是这个页的不再存
        items: list = []
        want: set = set()
        current: set = set()
        for ts in project.texture_sets:
            for layer in ts.layers:
                store = engine.layers.stores.get(layer.uid)
                if store is None:
                    continue
                uid = layer.uid
                for plane, mip, tx, ty, page in store.pages():
                    key = (uid, plane, mip, tx, ty)
                    current.add(key)
                    if not page.dirty and page.disk_key == key:
                        continue
                    want.add(key)
                    mark = page.asave
                    if mark is not None and mark[0] is token and mark[1] == key and rec_keys.get(key) == mark[2]:
                        continue
                    items.append((key, page))
        stale = [key for key in rec_keys if key not in want]
        assets, producers, drops, deleted_assets = self._plan_assets(project, base_assets)
        meta = {"format": FORMAT_VERSION, "project": project.to_dict()}
        if extra:
            meta.update(extra)
        doc = json.dumps(meta, sort_keys=True, ensure_ascii=False, default=str)
        if not items and not stale and not producers and not drops and not self._pending_maps and doc == self.last_doc:
            return False
        meta["recover"] = dict(info or {}, format=RECOVER_FORMAT, name=project.name, base=base_path,
                               base_stat=list(base_stat) if base_stat else None,
                               deleted=[list(key) for key in sorted(base_keys - current)] if base_keys else [],
                               deleted_assets=deleted_assets)
        writer = self._ensure_writer()
        if self.path is None:
            self.path = self._fresh_path()
            self.token = token
            self.rec_keys = {}
            self.rec_assets = {}
            writer.put("open", self.path, int(self.level), int(self.threads))
        self._gen += 1
        job = _Job(self._gen, items, want, meta, doc, assets)
        writer.put("begin", job.gen)
        if stale:
            writer.put("pages", job.gen, [], 0, stale)
        for name in drops:
            writer.put("drop_asset", job.gen, name)
        for name, producer in producers:
            writer.put("asset", job.gen, name, producer)
        job.readbacks = [(name, maps, maps.readback_steps()) for name, maps in self._pending_maps]
        self._pending_maps = []
        self.job = job
        engine._pending_work = True
        log.info("自动保存开始：%d 页要写，%d 页要删，%d 个资源", len(items), len(stale), len(producers))
        return True

    def _plan_assets(self, project, base_assets: frozenset):
        """模型、模型贴图：哪些要进恢复文件（相对工程文件改过的），其中哪些这次要（重新）写。"""
        engine = self.engine
        sculpt = getattr(engine, "sculpt", None)
        edit = getattr(engine, "edit", None)
        self._pending_maps = []
        wanted: dict[str, tuple] = {}
        producers: list = []
        for obj in project.objects:
            name = "mesh/%d.npz" % obj.uid
            live_sculpt = sculpt is not None and sculpt.obj is obj and sculpt.changed
            live_edit = edit is not None and edit.obj is obj and edit.changed
            if not (obj.geometry_dirty or live_sculpt or live_edit or name not in base_assets):
                continue
            mark = ("mesh", obj.geometry_version, sculpt.version if live_sculpt else 0,
                    edit.version if live_edit else 0)
            wanted[name] = mark
            if self.rec_assets.get(name) == mark:
                continue
            if live_sculpt:
                build = sculpt.save_snapshot()            # 显卡读回在主线程，拆数据和压缩在写盘线程
                producers.append((name, lambda build=build: mesh_to_bytes(build())))
            elif live_edit:
                producers.append((name, partial(mesh_to_bytes, edit.full_data())))
            else:
                producers.append((name, partial(mesh_to_bytes, copy.copy(obj.data))))   # 数组不会原地改，浅拷贝就定住了
        for ts in project.texture_sets:
            name = "meshmaps/%d.bin" % ts.uid
            maps = engine.meshmaps.get(ts.uid)
            if maps is None or (maps.saved and name in base_assets):
                continue
            mark = ("maps", maps.serial)
            wanted[name] = mark
            if self.rec_assets.get(name) == mark:
                continue
            if maps.has_arrays:
                producers.append((name, self._maps_producer(maps)))
            else:                                          # 还没读回：分帧读回，读完再写（16K 一次读回会卡一两秒）
                self._pending_maps.append((name, maps))
        drops = [name for name in self.rec_assets if name not in wanted]
        set_uids = {ts.uid for ts in project.texture_sets}
        deleted_assets = []
        for name in base_assets:
            if name.startswith("meshmaps/"):
                try:
                    uid = int(name.split("/", 1)[1].split(".", 1)[0])
                except ValueError:
                    continue
                if uid in set_uids and engine.meshmaps.get(uid) is None:
                    deleted_assets.append(name)                 # 工程文件里有、现在已经清掉的模型贴图
        return wanted, producers, drops, sorted(deleted_assets)

    # ---------------------------------------------------------------- 每帧推进
    def pump(self, budget_ms: float = 1.5) -> None:
        job = self.job
        if job is None:
            return
        started = time.perf_counter()
        try:
            self._pump(job, budget_ms)
        finally:
            job.pump_ms.append((time.perf_counter() - started) * 1000.0)

    def _pump(self, job: _Job, budget_ms: float) -> None:
        writer = self.writer
        if writer.failed is not None and writer.failed[0] == job.gen:
            self._fail(job, writer.failed[1])
            return
        if job.finishing:
            if writer.committed_gen >= job.gen:
                self._finish(job)
            return
        cache = self.engine.cache
        deadline = time.perf_counter() + max(0.05, float(budget_ms)) / 1000.0
        while job.readbacks and time.perf_counter() < deadline:
            name, maps, steps = job.readbacks[0]
            try:
                next(steps)
            except StopIteration:
                job.readbacks.pop(0)
                writer.put("asset", job.gen, name, self._maps_producer(maps))
        while time.perf_counter() < deadline and writer.inflight <= MAX_INFLIGHT_BYTES:
            if not job.items:
                if not job.waiting:
                    break
                ready = [(k, p) for k, p in job.waiting if not self._gpu_only(p)]
                if not ready:
                    cache.request_backup([p for _k, p in job.waiting])
                    self.engine._pending_work = True
                    break
                job.waiting = [(k, p) for k, p in job.waiting if self._gpu_only(p)]
                job.items.extend(ready)
                continue
            key, page = job.items.popleft()
            self._take(job, cache, key, page)
        self._flush(job)
        if not job.items and not job.waiting and not job.readbacks:
            if job.dropped:
                writer.put("pages", job.gen, [], 0, sorted(job.dropped))
                job.want.difference_update(job.dropped)
            recover = job.meta["recover"]
            recover["saved_at"] = time.time()
            recover["pages"] = len(job.want)
            writer.put("commit", job.gen, job.meta)
            job.finishing = True

    @staticmethod
    def _maps_producer(maps):
        """写盘线程里把模型贴图转成字节（CPU 副本已经齐了，不碰显卡）。"""
        clone = copy.copy(maps)
        clone.meta = dict(maps.meta)
        return clone.to_bytes

    @staticmethod
    def _gpu_only(page) -> bool:
        return (not page.dead and page.blob is None and page.scratch is None and page.disk_key is None
                and page.slot >= 0)

    def _current_page(self, key):
        uid, plane, mip, tx, ty = key
        store = self.engine.layers.stores.get(uid)
        planes = getattr(store, "planes", {}) if store is not None else {}
        level_store = planes.get(plane)
        if level_store is None or mip >= len(level_store.levels):
            return None
        return level_store.levels[mip].pages.get((tx, ty))

    def _take(self, job: _Job, cache, key, page) -> None:
        """取一页的数据交给写盘线程；只在显存里的放进等待队列。"""
        if page.dead:
            page = self._current_page(key)
            if page is None or (not page.dirty and page.disk_key == key):
                job.dropped.add(key)
                return
        nbytes = cache.pools[page.fmt].page_bytes
        fresh = 0
        if page.blob is not None:
            entry = (key, page.codec, page.blob, nbytes)
        elif page.scratch is not None and cache.scratch is not None:
            offset, length, codec = page.scratch
            data = cache.scratch.read(offset, length)
            entry = (key, codec, data, nbytes)
            fresh = len(data)
        elif page.disk_key is not None and cache.storage is not None:
            data = cache.storage.read_page(page.disk_key)
            if data is None:
                log.warning("工程文件里找不到页面，自动保存跳过：%r", key)
                job.dropped.add(key)
                return
            entry = (key, 0, data, nbytes)
            fresh = len(data)
        elif page.slot >= 0:
            job.waiting.append((key, page))
            return
        else:
            log.warning("页面数据丢失，自动保存跳过：%r", key)
            job.dropped.add(key)
            return
        job.batch.append(entry)
        job.batch_bytes += fresh
        job.new_bytes += fresh
        job.written.append((page, key, next(self._serial)))
        if len(job.batch) >= BATCH_PAGES:
            self._flush(job)

    def _flush(self, job: _Job) -> None:
        if not job.batch:
            return
        self.writer.add_inflight(job.batch_bytes)
        self.writer.put("pages", job.gen, job.batch, job.batch_bytes, ())
        job.batch = []
        job.batch_bytes = 0

    def _finish(self, job: _Job) -> None:
        token = self.token
        kept = {key: serial for key, serial in self.rec_keys.items() if key in job.want}
        for page, key, serial in job.written:
            if key in job.want:
                page.asave = (token, key, serial)
                kept[key] = serial
        self.rec_keys = kept
        self.rec_assets = dict(job.assets)
        self.last_doc = job.doc
        self.job = None
        self.last_error = None
        pumps = sorted(job.pump_ms)
        result = {"path": self.path, "pages": len(job.written), "pages_total": len(kept),
                  "seconds": time.perf_counter() - job.started, "read_mb": job.new_bytes / 1048576.0,
                  "frames": len(pumps), "pump_p95_ms": pumps[int(0.95 * (len(pumps) - 1))] if pumps else 0.0,
                  "pump_max_ms": pumps[-1] if pumps else 0.0}
        self.last_result = result
        log.info("自动保存完成：写 %d 页（恢复文件里共 %d 页），用时 %.2f 秒", result["pages"], result["pages_total"],
                 result["seconds"])
        self._notify(result)

    def _fail(self, job: _Job, error: BaseException) -> None:
        self.job = None
        self.writer.failed = None
        if self.writer.file is None:              # 恢复文件根本没打开（目录不能写等）：下次换个新文件重来
            self.path = None
            self.token = None
            self.rec_keys = {}
            self.rec_assets = {}
            self.last_doc = None
        self.last_error = str(error) or type(error).__name__
        self._notify({"error": self.last_error, "path": self.path})

    def _notify(self, result: dict) -> None:
        for callback in list(self.on_finished):
            try:
                callback(result)
            except Exception:  # noqa: BLE001
                log.exception("自动保存回调出错")

    # ---------------------------------------------------------------- 取消、重来、关闭
    def wait(self, timeout: float = 120.0, budget_ms: float = 50.0) -> dict | None:
        """把进行中的自动保存一直推进到做完（测试、退出前用）。需要显卡上下文是当前的。"""
        deadline = time.perf_counter() + timeout
        self.set_paused(False)
        while self.job is not None and time.perf_counter() < deadline:
            glx.flush()
            self.engine.cache.service(self.engine.frame_index, budget_ms=budget_ms, background=False)
            self.pump(budget_ms)
            if self.job is not None and self.job.finishing:
                time.sleep(0.002)
        return self.last_result if self.job is None else None

    def cancel(self) -> None:
        """放弃进行中的这次（恢复文件停在上一次完整的状态）。"""
        job = self.job
        if job is None:
            return
        self.job = None
        for _name, _maps, steps in job.readbacks:
            steps.close()                         # 放掉读回用的帧缓冲
        job.readbacks = []
        if self.writer is not None:
            self.writer.gate.set()
            self.writer.put("abort", job.gen)

    def _discard_file(self) -> None:
        if self.path is not None and self.writer is not None:
            self.writer.put("close", True)
        self.path = None
        self.token = None
        self.rec_keys = {}
        self.rec_assets = {}
        self.last_doc = None

    def reset(self) -> None:
        """工程换了、刚存过盘：之前的恢复文件作废并删掉，下次从头写。"""
        self.cancel()
        self._discard_file()
        self._base = None

    def shutdown(self, delete: bool = False, timeout: float = 60.0) -> None:
        """停掉写盘线程。delete 为真时删掉恢复文件（正常退出），否则留着已经写完的。"""
        self.cancel()
        writer = self.writer
        if writer is not None:
            writer.gate.set()
        if writer is not None and writer.is_alive():
            writer.put("close", bool(delete))
            writer.stop(timeout)
        self.writer = None
        if delete:
            self.path = None
            self.token = None
            self.rec_keys = {}
            self.rec_assets = {}
            self.last_doc = None
        self._base = None


# ================================================================== 恢复
def _checkpointed_copy(source: str, target: str, journal: str = "WAL") -> None:
    """把一个工程格式的文件完整复制成 target：先打开、关闭一次（日志里的内容并回主文件、没写完的撤回），再按普通文件复制。
    比逐页复制快得多（几 GB 的工程只要几秒）。先写临时文件再改名，失败时不留半成品。"""
    ProjectFile(source, journal=journal).close()
    folder = os.path.dirname(os.path.abspath(target))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(target) + ".", suffix=".tmp", dir=folder)
    os.close(fd)
    try:
        shutil.copyfile(source, tmp)
        remove_quiet(target)
        os.replace(tmp, target)
    except BaseException:
        remove_quiet(tmp)
        raise


def restore(recover_path: str, target: str) -> dict:
    """把一份恢复文件还原成普通工程文件 target。

    有原工程文件时：原工程完整复制一份，再盖上恢复文件里的改动（改过的页、资源、工程信息，删掉不要了的页）。
    没存过的工程：恢复文件本身就是完整的工程，复制一份、去掉恢复记录。
    返回 {"pages", "base", "warnings"}。target 已存在时覆盖。"""
    with ProjectFile(recover_path, journal=RECOVER_JOURNAL) as rec:
        meta = rec.read_meta()
    info = meta.get("recover")
    if not isinstance(info, dict) or "project" not in meta:
        raise ProjectFileError("这份自动保存没有写完，不能恢复：%s" % recover_path, recover_path)
    warnings: list[str] = []
    base = info.get("base") or None
    have_base = bool(base) and os.path.isfile(base)
    if base and not have_base:
        warnings.append("原工程文件找不到了（%s），只恢复出改过的部分" % base)
    elif have_base and info.get("base_stat") is not None and list(file_stat(base) or ()) != list(info["base_stat"]):
        warnings.append("原工程文件在自动保存之后又被改过，恢复出来的内容可能对不上")
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    count = 0
    try:
        if have_base:
            _checkpointed_copy(base, target)
            with ProjectFile(target) as dst, ProjectFile(recover_path, journal=RECOVER_JOURNAL) as rec:
                old_meta = dst.read_meta()
                with dst.transaction():
                    deleted = [tuple(int(v) for v in key) for key in info.get("deleted", [])]
                    if deleted:
                        dst.delete_pages(deleted)
                    for name in info.get("deleted_assets", []):
                        dst.delete_asset(name)
                    count = dst.copy_pages_from(rec)
                    for name in rec.list_assets():
                        data = rec.read_asset(name)
                        if data is not None:
                            dst.write_asset(name, data)
                    old_meta.update({key: value for key, value in meta.items() if key != "recover"})
                    dst.write_meta(old_meta)
        else:
            _checkpointed_copy(recover_path, target, RECOVER_JOURNAL)
            with ProjectFile(target) as dst:
                count = len(dst.page_keys())
                dst.write_meta({key: value for key, value in meta.items() if key != "recover"})
    except BaseException:
        remove_quiet(target)
        raise
    log.info("已从 %s 恢复到 %s：%d 页%s", recover_path, target, count, "；" + "；".join(warnings) if warnings else "")
    return {"pages": count, "base": base, "warnings": warnings}
