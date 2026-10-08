"""工程文件 `.splender`：一个 SQLite 文件，装工程信息、资源文件和稀疏的贴图页。

格式（版本 1）
--------------
- 文件头：SQLite 的 application_id 固定为 APPLICATION_ID（ASCII "SPLD"），user_version 是格式版本。
  打开时两者都要对上；格式版本比程序新时抛 ProjectVersionError。
- meta(key, value)：工程信息，一个键一行，value 是 JSON 文本。
- assets(name, data)：资源文件原样存放（例如导入的模型原文件）。单个资源上限是 SQLite 的单值上限（约 1e9 字节）。
- pages(layer_uid, plane, mip, tx, ty, codec, raw_len, data)：一个页一行，前五列是联合主键。
  codec 0 = 原样存放（压缩后反而更大时用），1 = zstd 帧（带内容长度和校验和）；raw_len 是解压后的字节数。
  页面内容对本模块只是字节，像素格式由调用方决定。

耐久性
------
WAL 日志 + synchronous=FULL：提交在返回前已经落盘。进程被杀或断电时，没提交完的事务整体作废，
文件停在上一次成功提交的状态，下次打开时 SQLite 自动恢复。正常 close() 会做检查点并删掉 -wal、-shm，
目录里只剩 .splender 一个文件。

线程
----
- 写操作（write_*、delete_*、vacuum、transaction）共用一个写连接，彼此串行，通常在主线程调用。
- 读操作（read_*、page_keys、list_assets、stats、check、save_copy）可以在任何线程调用，走独立的读连接池。
  WAL 下读到的永远是最近一次提交的完整状态，不会看到写到一半的内容，也不会被写操作挡住。
- 压缩和解压在线程池里并行，zstandard 计算时释放 GIL。
"""
from __future__ import annotations

import json
import logging
import operator
import os
import pathlib
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from contextlib import contextmanager
from typing import Any

import zstandard

log = logging.getLogger("splender.doc.storage")

PageKey = tuple[int, int, int, int, int]  # (layer_uid, plane, mip, tx, ty)

FORMAT_VERSION = 1
APPLICATION_ID = 0x53504C44  # ASCII "SPLD"

CODEC_RAW = 0
CODEC_ZSTD = 1
CODEC_NAMES = {CODEC_RAW: "原样", CODEC_ZSTD: "zstd"}

#: 单页原始长度的上限。正常页最大 256×256×4 = 256 KiB，这个上限只用来挡住损坏数据把内存撑爆。
MAX_PAGE_BYTES = 256 * 1024 * 1024
#: 检查点之后 -wal 文件最多保留的大小。
JOURNAL_SIZE_LIMIT = 64 * 1024 * 1024
#: 允许的同步级别。FULL 保证提交后断电不丢；NORMAL 断电可能丢最后一次提交，但文件不坏。
SYNCHRONOUS_LEVELS = ("NORMAL", "FULL", "EXTRA")

_SIDECARS = ("-wal", "-shm", "-journal")

_SCHEMA = (
    "CREATE TABLE meta (key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)",
    "CREATE TABLE assets (name TEXT PRIMARY KEY NOT NULL, data BLOB NOT NULL)",
    "CREATE TABLE pages (layer_uid INTEGER NOT NULL, plane INTEGER NOT NULL, mip INTEGER NOT NULL, "
    "tx INTEGER NOT NULL, ty INTEGER NOT NULL, codec INTEGER NOT NULL, raw_len INTEGER NOT NULL, "
    "data BLOB NOT NULL, PRIMARY KEY (layer_uid, plane, mip, tx, ty))",
)
_REQUIRED_COLUMNS = {
    "meta": {"key", "value"},
    "assets": {"name", "data"},
    "pages": {"layer_uid", "plane", "mip", "tx", "ty", "codec", "raw_len", "data"},
}
_TABLE_LABELS = {"meta": "工程信息", "assets": "资源", "pages": "页面"}

_KEY_WHERE = "layer_uid=? AND plane=? AND mip=? AND tx=? AND ty=?"
_SELECT_PAGE = f"SELECT codec, raw_len, data FROM pages WHERE {_KEY_WHERE}"
_DELETE_PAGE = f"DELETE FROM pages WHERE {_KEY_WHERE}"
_UPSERT_PAGE = ("INSERT OR REPLACE INTO pages (layer_uid, plane, mip, tx, ty, codec, raw_len, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)")

# SQLite 主错误码
_SQLITE_BUSY, _SQLITE_LOCKED, _SQLITE_NOMEM, _SQLITE_READONLY = 5, 6, 7, 8
_SQLITE_IOERR, _SQLITE_CORRUPT, _SQLITE_FULL, _SQLITE_CANTOPEN = 10, 11, 13, 14
_SQLITE_TOOBIG, _SQLITE_NOTADB = 18, 26


# ---------------------------------------------------------------- 异常

class ProjectFileError(Exception):
    """工程文件打不开、读不出或写不进。存储层的所有错误都是它或它的子类，str(e) 是给用户看的中文说明。"""

    def __init__(self, message: str, path: str | None = None):
        super().__init__(message)
        self.path = path


class ProjectCorruptError(ProjectFileError):
    """文件不是 SPLENDER 工程文件，或者内容已损坏。"""


class ProjectVersionError(ProjectFileError):
    """文件由更新版本的程序保存，当前程序读不了。"""

    def __init__(self, message: str, path: str | None = None, file_version: int = 0,
                 supported_version: int = FORMAT_VERSION):
        super().__init__(message, path)
        self.file_version = file_version
        self.supported_version = supported_version


class ProjectClosedError(ProjectFileError):
    """工程文件已经关闭，不能再读写。"""


# ---------------------------------------------------------------- 小工具

# 每个线程自己的 zstd 压缩器、解压器（它们不能跨线程同时用），以及「是不是本模块线程池的线程」标记
_TLS = threading.local()


def _compressor(level: int) -> zstandard.ZstdCompressor:
    cache = getattr(_TLS, "compressors", None)
    if cache is None:
        cache = _TLS.compressors = {}
    comp = cache.get(level)
    if comp is None:
        comp = cache[level] = zstandard.ZstdCompressor(level=level, write_checksum=True, write_content_size=True)
    return comp


def _decompressor() -> zstandard.ZstdDecompressor:
    dec = getattr(_TLS, "decompressor", None)
    if dec is None:
        dec = _TLS.decompressor = zstandard.ZstdDecompressor()
    return dec


def _mark_pool_thread() -> None:
    _TLS.in_pool = True


def _in_pool_thread() -> bool:
    return getattr(_TLS, "in_pool", False)


def _key(key: Any) -> PageKey:
    """把页面键规范成五个 Python 整数（接受 numpy 整数，拒绝浮点）。"""
    try:
        layer_uid, plane, mip, tx, ty = key
        return (operator.index(layer_uid), operator.index(plane), operator.index(mip),
                operator.index(tx), operator.index(ty))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"页面键应是 (layer_uid, plane, mip, tx, ty) 五个整数，收到的是 {key!r}") from exc


def _byte_view(data: Any) -> memoryview:
    """把 bytes、bytearray、memoryview、numpy 数组等统一看成一维字节视图（不复制，除非内存不连续）。"""
    try:
        mv = data if isinstance(data, memoryview) else memoryview(data)
    except TypeError as exc:
        raise TypeError(f"页面数据应是字节类对象（bytes、bytearray、numpy 数组等），收到的是 {type(data).__name__}") from exc
    if mv.ndim == 1 and mv.format in ("B", "b", "c") and mv.c_contiguous:
        return mv.cast("B") if mv.format != "B" else mv
    if not mv.c_contiguous:
        return memoryview(mv.tobytes())
    try:
        return mv.cast("B")
    except (TypeError, ValueError):  # 非本机字节序等格式不能直接转换，复制一份
        return memoryview(mv.tobytes())


def _nbytes(data: Any) -> int:
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    n = getattr(data, "nbytes", None)
    if isinstance(n, int):
        return n
    return _byte_view(data).nbytes


def _sidecar_paths(path: str) -> list[str]:
    return [path + s for s in _SIDECARS]


def _remove_quiet(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        log.warning("删不掉临时文件 %s", path, exc_info=True)


def _replace_file(src: str, dst: str, attempts: int = 10) -> None:
    """os.replace，Windows 上目标被杀毒或索引程序短暂占用时稍等重试。"""
    delay = 0.02
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


def _fsync_file(path: str) -> None:
    with open(path, "rb+") as fh:
        os.fsync(fh.fileno())


def _sqlite_uri(path: str, mode: str) -> str:
    return pathlib.Path(path).as_uri() + "?mode=" + mode


class _Done:
    """不进线程池时代替 Future：已经算好的结果。"""

    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value

    def cancel(self) -> bool:
        return False


# ---------------------------------------------------------------- 工程文件

class ProjectFile:
    """一个打开着的 `.splender` 工程文件。

    参数：
        path：文件路径。
        create：为 False 时文件必须已存在且是工程文件。为 True 时文件不存在（或是 0 字节的空文件）就新建；
            已存在的工程文件照常打开，内容不动；已存在但不是工程文件则报错，绝不覆盖。
        level：zstd 压缩级别，默认 3。越高越小越慢，负数是更快的快速档。
        threads：压缩、解压线程数，0 表示取 CPU 逻辑核数，1 表示不开线程池。
        batch_pages、batch_bytes：write_pages 每批压缩并写入的页数和原始字节上限，满一个就写一批，内存只占约两批。
        synchronous：SQLite 同步级别，NORMAL / FULL / EXTRA，默认 FULL（提交后断电不丢）。
        page_size：新建文件时的 SQLite 页大小（字节，512 到 65536 的 2 的幂），对已有文件无效。
        cache_mib：写连接的 SQLite 缓存大小（MiB）。读连接各用四分之一。
        timeout：等待文件锁的秒数（别的进程也打开了同一个文件时才会遇到）。
    """

    def __init__(self, path: str | os.PathLike, create: bool = False, *, level: int = 3, threads: int = 0,
                 batch_pages: int = 256, batch_bytes: int = 64 * 1024 * 1024, synchronous: str = "FULL",
                 page_size: int = 16384, cache_mib: int = 64, timeout: float = 30.0):
        self.path = os.path.abspath(os.fspath(path))
        self._level = max(-100, min(int(level), zstandard.MAX_COMPRESSION_LEVEL))
        threads = int(threads)
        self._threads = max(1, min(threads if threads > 0 else (os.cpu_count() or 4), 256))
        self._batch_pages = max(1, int(batch_pages))
        self._batch_bytes = max(1, int(batch_bytes))
        sync = str(synchronous).upper()
        if sync not in SYNCHRONOUS_LEVELS:
            raise ValueError(f"synchronous 只能是 {'、'.join(SYNCHRONOUS_LEVELS)} 之一，收到 {synchronous!r}")
        self._synchronous = sync
        page_size = int(page_size)
        if page_size < 512 or page_size > 65536 or page_size & (page_size - 1):
            raise ValueError(f"page_size 应是 512 到 65536 之间的 2 的幂，收到 {page_size}")
        self._page_size = page_size
        self._cache_kib = max(256, int(cache_mib) * 1024)
        self._timeout = max(0.0, float(timeout))
        self.journal_mode = ""

        self._write_lock = threading.RLock()  # 写连接、事务深度
        self._state = threading.Condition()   # 关闭标记、读连接池
        self._closed = False
        self._idle_readers: list[sqlite3.Connection] = []
        self._all_readers: set[sqlite3.Connection] = set()
        self._active_reads = 0
        self._tx_depth = 0
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()
        self._wconn: sqlite3.Connection | None = self._open_writer(bool(create))

    # ------------------------------------------------------------ 打开、关闭

    def _open_writer(self, create: bool) -> sqlite3.Connection:
        path = self.path
        if os.path.isdir(path):
            raise ProjectFileError(f"这是一个文件夹，不是工程文件：{path}", path)
        existed = os.path.exists(path)
        size_before = os.path.getsize(path) if existed else 0
        if not existed:
            if not create:
                raise ProjectFileError(f"找不到工程文件：{path}", path)
            folder = os.path.dirname(path)
            if not os.path.isdir(folder):
                raise ProjectFileError(f"要保存到的文件夹不存在：{folder}", path)
            # 同名的旧 -wal 会被 SQLite 当成新文件的日志重放进来，先清掉
            for side in _sidecar_paths(path):
                if os.path.exists(side):
                    try:
                        os.remove(side)
                    except OSError as exc:
                        raise ProjectFileError(f"旧的同名临时文件删不掉，不能在这里新建工程：{side}（{exc}）", path) from exc
        try:
            conn = sqlite3.connect(_sqlite_uri(path, "rw" if existed else "rwc"), uri=True,
                                   timeout=self._timeout, isolation_level=None, check_same_thread=False)
        except sqlite3.Error as exc:
            raise self._map_error(exc, "打开工程文件") from exc
        try:
            try:
                pages = conn.execute("PRAGMA page_count").fetchone()[0]
                if pages == 0:
                    # 只有不存在的文件或 0 字节的文件才当新文件初始化；别的「空库」不是我们的，不动它
                    if not create or size_before > 0:
                        raise ProjectCorruptError(f"文件是空的或不完整，不是 SPLENDER 工程文件：{path}", path)
                    self._create_schema(conn)
                else:
                    self._check_schema(conn)
                self._configure_writer(conn)
            except sqlite3.Error as exc:
                raise self._map_error(exc, "打开工程文件", opening=True) from exc
        except BaseException:
            conn.close()
            if not existed:  # 新建失败，不留半成品
                for p in [path, *_sidecar_paths(path)]:
                    _remove_quiet(p)
            raise
        return conn

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(f"PRAGMA page_size={self._page_size}")
        conn.execute("PRAGMA journal_mode=WAL").fetchall()
        conn.execute("BEGIN IMMEDIATE")
        try:
            for sql in _SCHEMA:
                conn.execute(sql)
            conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
            conn.execute(f"PRAGMA user_version={FORMAT_VERSION}")
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def _check_schema(self, conn: sqlite3.Connection) -> None:
        path = self.path
        app_id = conn.execute("PRAGMA application_id").fetchone()[0]
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if app_id != APPLICATION_ID:
            raise ProjectCorruptError(f"这不是 SPLENDER 工程文件，是别的程序的数据库：{path}", path)
        if version > FORMAT_VERSION:
            raise ProjectVersionError(
                f"这个工程文件是更新版本的 SPLENDER 保存的（文件格式版本 {version}），"
                f"当前程序只能打开格式版本 {FORMAT_VERSION} 及以下的文件。请升级 SPLENDER 后再打开：{path}",
                path, int(version), FORMAT_VERSION)
        if version < 1:
            raise ProjectCorruptError(f"工程文件的格式版本无效（{version}），文件已损坏：{path}", path)
        for table, needed in _REQUIRED_COLUMNS.items():
            have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not have:
                raise ProjectCorruptError(f"工程文件缺少「{_TABLE_LABELS[table]}」数据表，文件已损坏：{path}", path)
            missing = needed - have
            if missing:
                raise ProjectCorruptError(f"工程文件的「{_TABLE_LABELS[table]}」数据表缺少字段 "
                                          f"{', '.join(sorted(missing))}，文件已损坏：{path}", path)

    def _configure_writer(self, conn: sqlite3.Connection) -> None:
        mode = str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        self.journal_mode = mode
        if mode != "wal":
            log.warning("工程文件不能使用 WAL 日志（当前 %s），后台读取会在保存时等待：%s", mode, self.path)
        conn.execute(f"PRAGMA synchronous={self._synchronous}")
        conn.execute(f"PRAGMA cache_size=-{self._cache_kib}")
        conn.execute(f"PRAGMA journal_size_limit={JOURNAL_SIZE_LIMIT}")

    def _open_reader(self) -> sqlite3.Connection:
        conn = sqlite3.connect(_sqlite_uri(self.path, "rw"), uri=True, timeout=self._timeout,
                               isolation_level=None, check_same_thread=False)
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.execute(f"PRAGMA cache_size=-{max(256, self._cache_kib // 4)}")
        except BaseException:
            conn.close()
            raise
        return conn

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def level(self) -> int:
        return self._level

    @property
    def threads(self) -> int:
        return self._threads

    def close(self) -> None:
        """关闭文件。等正在进行的读写做完，做检查点并删掉 -wal、-shm。可以重复调用。"""
        with self._state:
            if self._closed:
                return
            self._closed = True
            while self._active_reads:
                self._state.wait()
            readers = list(self._all_readers)
            self._all_readers.clear()
            self._idle_readers.clear()
        for conn in readers:
            try:
                conn.close()
            except sqlite3.Error:
                log.warning("关闭读连接出错", exc_info=True)
        with self._write_lock:
            conn, self._wconn = self._wconn, None
            if conn is not None:
                try:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
                except sqlite3.Error:
                    log.warning("关闭前的检查点没做成，下次打开会自动恢复：%s", self.path, exc_info=True)
                finally:
                    conn.close()
        with self._pool_lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def __enter__(self) -> ProjectFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<ProjectFile {self.path!r}{' 已关闭' if self._closed else ''}>"

    # ------------------------------------------------------------ 内部：连接、事务、线程池、错误

    def _map_error(self, exc: BaseException, action: str, opening: bool = False) -> ProjectFileError:
        """把 sqlite3 的异常翻译成带中文说明的 ProjectFileError。"""
        path = self.path
        code = getattr(exc, "sqlite_errorcode", None)
        primary = code & 0xFF if isinstance(code, int) else None
        if primary == _SQLITE_NOTADB or (primary is None and "not a database" in str(exc)):
            return ProjectCorruptError(f"{action}失败：这不是 SPLENDER 工程文件，或者文件头已损坏：{path}", path)
        if primary == _SQLITE_CORRUPT or (primary is None and "malformed" in str(exc)):
            return ProjectCorruptError(f"{action}失败：工程文件已损坏（{exc}）：{path}", path)
        if primary == _SQLITE_FULL:
            return ProjectFileError(f"{action}失败：磁盘空间不足。文件保持上一次成功保存的状态：{path}", path)
        if primary == _SQLITE_READONLY:
            return ProjectFileError(f"{action}失败：工程文件或所在文件夹是只读的：{path}", path)
        if primary in (_SQLITE_BUSY, _SQLITE_LOCKED):
            return ProjectFileError(f"{action}失败：工程文件正被别的程序占用：{path}", path)
        if primary == _SQLITE_CANTOPEN:
            return ProjectFileError(f"{action}失败：打不开文件，请检查路径和权限：{path}", path)
        if primary == _SQLITE_IOERR:
            return ProjectFileError(f"{action}失败：读写磁盘出错（{exc}）：{path}", path)
        if primary == _SQLITE_NOMEM:
            return ProjectFileError(f"{action}失败：内存不足：{path}", path)
        if primary == _SQLITE_TOOBIG:
            return ProjectFileError(f"{action}失败：单项数据超过了工程文件的上限：{path}", path)
        if opening:
            return ProjectCorruptError(f"{action}失败：文件结构不对，可能不是 SPLENDER 工程文件或已损坏（{exc}）：{path}", path)
        return ProjectFileError(f"{action}失败：{exc}：{path}", path)

    def _writer(self) -> sqlite3.Connection:
        conn = self._wconn
        if self._closed or conn is None:
            raise ProjectClosedError(f"工程文件已经关闭：{self.path}", self.path)
        return conn

    @contextmanager
    def _write_tx(self, action: str) -> Iterator[sqlite3.Connection]:
        """写事务。最外层是 BEGIN IMMEDIATE … COMMIT，嵌套时用保存点。异常时撤销本层全部改动再抛出。"""
        with self._write_lock:
            conn = self._writer()
            depth = self._tx_depth
            try:
                conn.execute("BEGIN IMMEDIATE" if depth == 0 else f"SAVEPOINT sp{depth}")
            except sqlite3.Error as exc:
                raise self._map_error(exc, action) from exc
            self._tx_depth = depth + 1
            try:
                yield conn
            except BaseException as exc:
                self._tx_depth = depth
                self._rollback(conn, depth)
                if isinstance(exc, sqlite3.Error):
                    raise self._map_error(exc, action) from exc
                raise
            self._tx_depth = depth
            if self._wconn is not conn:
                raise ProjectClosedError(f"{action}没有完成：工程文件在事务中被关闭了：{self.path}", self.path)
            try:
                conn.execute("COMMIT" if depth == 0 else f"RELEASE sp{depth}")
            except sqlite3.Error as exc:
                self._rollback(conn, depth)
                raise self._map_error(exc, action) from exc

    @staticmethod
    def _rollback(conn: sqlite3.Connection, depth: int) -> None:
        try:
            if depth == 0:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
            else:
                conn.execute(f"ROLLBACK TO sp{depth}")
                conn.execute(f"RELEASE sp{depth}")
        except sqlite3.Error:
            log.error("撤销未完成的写入时出错", exc_info=True)

    @contextmanager
    def _reader(self, action: str) -> Iterator[sqlite3.Connection]:
        """借一个读连接。关闭会等所有借出的连接还回来。"""
        with self._state:
            if self._closed:
                raise ProjectClosedError(f"工程文件已经关闭：{self.path}", self.path)
            conn = self._idle_readers.pop() if self._idle_readers else None
            self._active_reads += 1
        broken = False
        try:
            if conn is None:
                try:
                    conn = self._open_reader()
                except sqlite3.Error as exc:
                    raise self._map_error(exc, action) from exc
                with self._state:
                    self._all_readers.add(conn)
            try:
                yield conn
            except sqlite3.Error as exc:
                broken = True
                raise self._map_error(exc, action) from exc
            except BaseException:
                broken = True
                raise
        finally:
            with self._state:
                self._active_reads -= 1
                if conn is not None:
                    if broken and conn.in_transaction:
                        try:
                            conn.execute("ROLLBACK")
                        except sqlite3.Error:
                            pass
                    if self._closed or (broken and conn.in_transaction):
                        self._all_readers.discard(conn)
                        try:
                            conn.close()
                        except sqlite3.Error:
                            pass
                    else:
                        self._idle_readers.append(conn)
                self._state.notify_all()

    def _pool_for_call(self) -> ThreadPoolExecutor | None:
        """本次调用用的线程池；单线程设置或已经身在池里时返回 None（就地计算，免得池子自己等自己）。"""
        if self._threads <= 1 or _in_pool_thread():
            return None
        pool = self._pool
        if pool is None:
            with self._pool_lock:
                if self._closed:
                    return None
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(self._threads, thread_name_prefix="splender-store",
                                                    initializer=_mark_pool_thread)
                pool = self._pool
        return pool

    def _split(self, items: list, pool: ThreadPoolExecutor | None) -> list[list]:
        if pool is None or len(items) <= 1:
            return [items] if items else []
        size = max(1, min(16, -(-len(items) // (self._threads * 2))))
        return [items[i:i + size] for i in range(0, len(items), size)]

    @staticmethod
    def _cancel_all(futures: list) -> None:
        real = [f for f in futures if not isinstance(f, _Done)]
        for f in real:
            f.cancel()
        if real:
            wait_futures(real)

    # ------------------------------------------------------------ 内部：编码

    def _encode_many(self, datas: list) -> list[tuple[int, int, Any]]:
        comp = _compressor(self._level)
        out = []
        for data in datas:
            mv = _byte_view(data)
            n = mv.nbytes
            if n > MAX_PAGE_BYTES:
                raise ValueError(f"单页数据 {n} 字节，超过了上限 {MAX_PAGE_BYTES} 字节")
            blob = comp.compress(mv)
            if len(blob) >= n:
                out.append((CODEC_RAW, n, mv))
            else:
                out.append((CODEC_ZSTD, n, blob))
        return out

    def _decode(self, key: PageKey, codec: Any, raw_len: Any, blob: Any) -> bytes:
        path = self.path
        if type(codec) is not int or type(raw_len) is not int or not isinstance(blob, bytes):
            raise ProjectCorruptError(f"页面 {key} 的记录格式不对，文件已损坏：{path}", path)
        if raw_len < 0 or raw_len > MAX_PAGE_BYTES:
            raise ProjectCorruptError(f"页面 {key} 记录的长度 {raw_len} 不合理，文件已损坏：{path}", path)
        if codec == CODEC_RAW:
            if len(blob) != raw_len:
                raise ProjectCorruptError(f"页面 {key} 的数据长度不对（{len(blob)}，应为 {raw_len}），文件已损坏：{path}", path)
            return blob
        if codec == CODEC_ZSTD:
            try:
                size = zstandard.frame_content_size(blob)
                if size != raw_len:
                    raise ProjectCorruptError(f"页面 {key} 的压缩数据和记录的长度对不上，文件已损坏：{path}", path)
                out = _decompressor().decompress(blob)
            except zstandard.ZstdError as exc:
                raise ProjectCorruptError(f"页面 {key} 的压缩数据已损坏（{exc}）：{path}", path) from exc
            if len(out) != raw_len:
                raise ProjectCorruptError(f"页面 {key} 解压后的长度不对，文件已损坏：{path}", path)
            return out
        raise ProjectCorruptError(f"页面 {key} 用了不认识的压缩方式（编号 {codec}），"
                                  f"可能是更新版本的程序写的，或者文件已损坏：{path}", path)

    def _decode_many(self, rows: list) -> list[tuple[PageKey, bytes]]:
        return [(row[0], self._decode(*row)) for row in rows]

    # ------------------------------------------------------------ 工程信息

    def read_meta(self) -> dict:
        """读全部工程信息，返回 {键: JSON 值}。"""
        with self._reader("读取工程信息") as conn:
            rows = conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        out = {}
        for key, value in rows:
            try:
                out[key] = json.loads(value)
            except (TypeError, ValueError) as exc:
                raise ProjectCorruptError(f"工程信息「{key}」不是有效的 JSON，文件已损坏：{self.path}", self.path) from exc
        return out

    def write_meta(self, meta: Mapping[str, Any]) -> None:
        """用 meta 整体替换工程信息（不在 meta 里的旧键会被删掉）。值必须能转成 JSON。"""
        rows = []
        for key, value in dict(meta).items():
            if not isinstance(key, str):
                raise TypeError(f"工程信息的键必须是字符串，收到 {key!r}")
            rows.append((key, json.dumps(value, ensure_ascii=False)))
        with self._write_tx("保存工程信息") as conn:
            conn.execute("DELETE FROM meta")
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", rows)

    # ------------------------------------------------------------ 资源

    def write_asset(self, name: str, data: Any) -> None:
        """存一个资源（同名覆盖）。data 是字节类对象，原样存放。"""
        if not isinstance(name, str) or not name:
            raise TypeError(f"资源名必须是非空字符串，收到 {name!r}")
        mv = _byte_view(data)
        with self._write_tx(f"保存资源「{name}」") as conn:
            limit = conn.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
            if mv.nbytes > limit:
                raise ValueError(f"资源「{name}」有 {mv.nbytes} 字节，超过了单个资源的上限 {limit} 字节")
            conn.execute("INSERT OR REPLACE INTO assets (name, data) VALUES (?, ?)", (name, mv))

    def read_asset(self, name: str) -> bytes | None:
        """读一个资源，没有返回 None。"""
        with self._reader(f"读取资源「{name}」") as conn:
            row = conn.execute("SELECT data FROM assets WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        data = row[0]
        if not isinstance(data, bytes):
            raise ProjectCorruptError(f"资源「{name}」的记录格式不对，文件已损坏：{self.path}", self.path)
        return data

    def list_assets(self) -> list[str]:
        """全部资源名，按名字排序。"""
        with self._reader("读取资源列表") as conn:
            return [row[0] for row in conn.execute("SELECT name FROM assets ORDER BY name")]

    def delete_asset(self, name: str) -> bool:
        """删除一个资源，返回是否真的删掉了。"""
        with self._write_tx(f"删除资源「{name}」") as conn:
            return conn.execute("DELETE FROM assets WHERE name=?", (name,)).rowcount > 0

    # ------------------------------------------------------------ 页面

    def page_keys(self, layer_uid: int | None = None) -> list[PageKey]:
        """全部页面键（按主键排序）。给了 layer_uid 就只列这一层的。"""
        with self._reader("读取页面列表") as conn:
            if layer_uid is None:
                rows = conn.execute("SELECT layer_uid, plane, mip, tx, ty FROM pages "
                                    "ORDER BY layer_uid, plane, mip, tx, ty").fetchall()
            else:
                rows = conn.execute("SELECT layer_uid, plane, mip, tx, ty FROM pages WHERE layer_uid=? "
                                    "ORDER BY plane, mip, tx, ty", (operator.index(layer_uid),)).fetchall()
        return rows

    def read_page(self, key: PageKey) -> bytes | None:
        """读一页，返回解压后的原始字节；没有这一页返回 None。任何线程都可以调用。"""
        k = _key(key)
        with self._reader("读取页面") as conn:
            row = conn.execute(_SELECT_PAGE, k).fetchone()
        if row is None:
            return None
        return self._decode(k, *row)

    def read_pages(self, keys: Iterable[PageKey]) -> dict[PageKey, bytes]:
        """一次读多页，返回 {键: 原始字节}，文件里没有的键不出现在结果里。

        所有页取自同一个已提交的状态。解压在线程池里并行。任何线程都可以调用。
        """
        wanted = list(dict.fromkeys(_key(k) for k in keys))
        if not wanted:
            return {}
        pool = self._pool_for_call()
        chunk_size = 8 if pool is not None else len(wanted)
        futures: list = []
        try:
            with self._reader("读取页面") as conn:
                conn.execute("BEGIN")
                try:
                    chunk = []
                    for k in wanted:
                        row = conn.execute(_SELECT_PAGE, k).fetchone()
                        if row is None:
                            continue
                        chunk.append((k, *row))
                        if len(chunk) >= chunk_size:
                            futures.append(pool.submit(self._decode_many, chunk) if pool is not None
                                           else _Done(self._decode_many(chunk)))
                            chunk = []
                    if chunk:
                        futures.append(pool.submit(self._decode_many, chunk) if pool is not None
                                       else _Done(self._decode_many(chunk)))
                finally:
                    if conn.in_transaction:
                        conn.execute("COMMIT")
                # 在借着读连接的时候把解压做完，关闭时会等它
                out: dict[PageKey, bytes] = {}
                for f in futures:
                    for k, data in f.result():
                        out[k] = data
                return out
        except BaseException:
            self._cancel_all(futures)
            raise

    def write_pages(self, items: Iterable[tuple[PageKey, Any]] | Mapping[PageKey, Any],
                    delete: Iterable[PageKey] = ()) -> None:
        """一个事务里写入并删除页面：要么全部生效，要么（出任何异常时）全部不生效。

        items 是 (键, 原始字节) 的可迭代对象，也可以是 {键: 原始字节}；可以是生成器，按批取用。
        字节数据在本调用返回前不能被改动。同一个键出现多次时以最后一次为准。
        先执行 delete 再写入，所以同时出现在两边的键最后是写入的内容。
        """
        if isinstance(items, Mapping):
            items = items.items()
        del_keys = [_key(k) for k in delete]
        with self._write_tx("保存页面") as conn:
            for i in range(0, len(del_keys), 4096):
                conn.executemany(_DELETE_PAGE, del_keys[i:i + 4096])
            self._write_batches(conn, items)

    def _batches(self, items: Iterable) -> Iterator[list]:
        batch: list = []
        nbytes = 0
        for item in items:
            try:
                key, data = item
            except (TypeError, ValueError) as exc:
                raise TypeError(f"write_pages 的每一项应是 (页面键, 字节数据)，收到 {type(item).__name__}") from exc
            batch.append((key, data))
            nbytes += _nbytes(data)
            if len(batch) >= self._batch_pages or nbytes >= self._batch_bytes:
                yield batch
                batch, nbytes = [], 0
        if batch:
            yield batch

    def _write_batches(self, conn: sqlite3.Connection, items: Iterable) -> None:
        """分批压缩、分批写入。第 k 批写进 SQLite 的同时，线程池在压缩第 k+1 批。"""
        pool = self._pool_for_call()
        pending: tuple[list, list] | None = None
        try:
            for batch in self._batches(items):
                keys = [_key(k) for k, _ in batch]
                futures = [pool.submit(self._encode_many, part) if pool is not None else _Done(self._encode_many(part))
                           for part in self._split([d for _, d in batch], pool)]
                if pending is not None:
                    self._insert_batch(conn, *pending)
                pending = (keys, futures)
            if pending is not None:
                self._insert_batch(conn, *pending)
                pending = None
        except BaseException:
            if pending is not None:
                self._cancel_all(pending[1])
            raise

    @staticmethod
    def _insert_batch(conn: sqlite3.Connection, keys: list, futures: list) -> None:
        rows = []
        it = iter(keys)
        for f in futures:
            for codec, raw_len, blob in f.result():
                rows.append((*next(it), codec, raw_len, blob))
        conn.executemany(_UPSERT_PAGE, rows)

    def delete_pages(self, keys: Iterable[PageKey]) -> None:
        """删除一批页面（一个事务）。"""
        self.write_pages((), delete=keys)

    def delete_layer(self, layer_uid: int) -> int:
        """删掉一个图层的全部页面，返回删掉的页数。"""
        uid = operator.index(layer_uid)
        with self._write_tx("删除图层页面") as conn:
            return conn.execute("DELETE FROM pages WHERE layer_uid=?", (uid,)).rowcount

    # ------------------------------------------------------------ 事务、统计、维护

    @contextmanager
    def transaction(self) -> Iterator[ProjectFile]:
        """把多次写操作合成一个事务，例如一次保存里的 write_pages 加 write_meta：
        块内全部成功才一起提交，块内抛出异常就全部撤销。可以嵌套。其他线程的写操作会等它结束。"""
        with self._write_tx("保存"):
            yield self

    def stats(self) -> dict[str, Any]:
        """统计：
        page_count 页数，compressed_bytes 页面压缩后总字节，raw_bytes 页面原始总字节，
        compression_ratio 压缩比（原始/压缩后），file_bytes 主文件大小，wal_bytes 日志文件大小，
        free_bytes 文件内可回收的空闲空间（vacuum 能收回），asset_count、asset_bytes 资源个数和总字节。"""
        with self._reader("统计") as conn:
            conn.execute("BEGIN")
            try:
                count, comp, raw = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(data)), 0), "
                                                "COALESCE(SUM(raw_len), 0) FROM pages").fetchone()
                n_assets, asset_bytes = conn.execute("SELECT COUNT(*), COALESCE(SUM(LENGTH(data)), 0) "
                                                     "FROM assets").fetchone()
                page_size = conn.execute("PRAGMA page_size").fetchone()[0]
                free_pages = conn.execute("PRAGMA freelist_count").fetchone()[0]
            finally:
                if conn.in_transaction:
                    conn.execute("COMMIT")

        def size(p: str) -> int:
            try:
                return os.path.getsize(p)
            except OSError:
                return 0

        return {
            "page_count": int(count),
            "compressed_bytes": int(comp),
            "raw_bytes": int(raw),
            "compression_ratio": (raw / comp) if comp else 0.0,
            "file_bytes": size(self.path),
            "wal_bytes": size(self.path + "-wal"),
            "free_bytes": int(free_pages) * int(page_size),
            "asset_count": int(n_assets),
            "asset_bytes": int(asset_bytes),
        }

    def check(self, full: bool = False) -> list[str]:
        """检查文件结构。没问题返回空列表，否则返回 SQLite 报告的问题。full=True 做完整检查（慢）。"""
        with self._reader("检查工程文件") as conn:
            rows = conn.execute("PRAGMA integrity_check" if full else "PRAGMA quick_check").fetchall()
        messages = [str(r[0]) for r in rows]
        return [] if messages == ["ok"] else messages

    def vacuum(self) -> None:
        """整理文件，收回删除内容留下的空间。耗时与文件大小成正比，需要约一倍文件大小的临时空间。"""
        with self._write_lock:
            conn = self._writer()
            if self._tx_depth:
                raise RuntimeError("事务进行中不能整理工程文件")
            try:
                conn.execute("VACUUM")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            except sqlite3.Error as exc:
                raise self._map_error(exc, "整理工程文件") from exc

    def save_copy(self, dst_path: str | os.PathLike,
                  progress: Callable[[float], None] | None = None, step_pages: int = 2048) -> None:
        """把最近一次提交的状态完整复制到 dst_path（另存为）。可以在后台线程调用，不挡写操作。

        先写到目标文件夹里的临时文件，落盘后再原子替换目标；失败时目标保持原样，不留临时文件。
        progress(0..1) 会在复制过程中被调用（在调用 save_copy 的线程里）。
        """
        dst = os.path.abspath(os.fspath(dst_path))
        if os.path.normcase(dst) == os.path.normcase(self.path) or (
                os.path.exists(dst) and os.path.samefile(dst, self.path)):
            raise ValueError(f"另存为的目标就是当前工程文件：{dst}")
        folder = os.path.dirname(dst)
        if not os.path.isdir(folder):
            raise ProjectFileError(f"要保存到的文件夹不存在：{folder}", dst)
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(dst) + ".", suffix=".tmp", dir=folder)
        os.close(fd)
        try:
            try:
                target = sqlite3.connect(tmp, isolation_level=None)
            except sqlite3.Error as exc:
                raise self._map_error(exc, "另存为") from exc
            try:
                with self._reader("另存为") as src:
                    if progress is None:
                        src.backup(target)  # 一步完成，本身就是一致的快照
                    else:
                        # 分步复制时先开读事务把快照定住，期间别的连接提交的写入不影响这份副本
                        src.execute("BEGIN")
                        try:
                            src.execute("SELECT COUNT(*) FROM meta").fetchall()

                            def report(_status, remaining, total):
                                progress(1.0 - remaining / total if total else 1.0)

                            src.backup(target, pages=max(1, int(step_pages)), progress=report)
                        finally:
                            if src.in_transaction:
                                src.execute("COMMIT")
                # 备份会带上源文件头里的 WAL 标记；副本改回普通日志模式，关闭后不留 -wal、-shm
                target.execute("PRAGMA journal_mode=DELETE").fetchall()
            except sqlite3.Error as exc:
                raise self._map_error(exc, "另存为") from exc
            finally:
                target.close()
            _fsync_file(tmp)
            for side in _sidecar_paths(dst):
                if os.path.exists(side):
                    try:
                        os.remove(side)
                    except OSError as exc:
                        raise ProjectFileError(f"目标工程正在被使用，不能覆盖：{dst}", dst) from exc
            try:
                _replace_file(tmp, dst)
            except OSError as exc:
                raise ProjectFileError(f"不能覆盖目标文件，它可能正被别的程序占用：{dst}（{exc}）", dst) from exc
            if progress is not None:
                progress(1.0)
        except BaseException:
            for p in [tmp, *_sidecar_paths(tmp)]:
                _remove_quiet(p)
            raise
