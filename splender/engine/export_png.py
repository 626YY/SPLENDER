"""流式 PNG 写出：按条带逐行写入，内存与整图大小无关，多线程压缩，输出标准 PNG。

做法
----
- 行按固定行数分块（默认每块约 4 MiB 原始数据）。块的划分只由图像宽度和参数决定，
  与 write_rows 每次给多少行、用几个线程都无关，所以同样的像素总是得到逐字节相同的文件。
- 每块在线程池里独立完成：行滤波（numpy 向量化）→ adler32 → 原始 deflate（负 wbits）。
  不是最后一块的以 Z_SYNC_FLUSH 收尾（字节对齐、不设结束标志，可以直接拼接），最后一块用 Z_FINISH。
  压缩结果切成若干 IDAT 块并算好 CRC。zlib、numpy 计算时都释放 GIL。
- 主线程按输入顺序写出：2 字节 zlib 头放在第一块之前；各块的 adler32 用 adler32_combine 合成整条数据流的
  校验值，接在最后一块之后。
- 先写同目录的临时文件，close() 成功后用 os.replace 原子替换目标；abort() 或出错时删掉临时文件，原有目标不动。

压缩选项（默认值来自 docs/evidence/export_png/benchmark.json 的评估）
--------------------------------------------------------------------
- filter：每行的滤波方式。"up"（默认）对贴图类内容比不滤波小 30%~45%，滤波本身每线程约 1.5 GiB/s，几乎不花时间。
  另有 "none"、"sub"、"avg"、"paeth"。
- compress_level：0 不压缩；1~3 快速档；4~9 细致档，越高越小越慢。
- strategy："auto"（默认）时 1~3 档用 zlib 的 RLE 策略（快，带噪点的贴图上反而最小，1~3 档结果相同），
  4~9 档用 FILTERED 策略（平滑内容和色块图明显更小，慢几倍）。也可以直接指定 "default"、"filtered"、"rle"、"huffman"。
"""
from __future__ import annotations

import os
import struct
import tempfile
import time
import weakref
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}  # 通道数 → PNG 颜色类型：灰度、灰度加透明、RGB、RGBA
FILTERS = {"none": 0, "sub": 1, "up": 2, "avg": 3, "paeth": 4}
STRATEGIES = {"default": zlib.Z_DEFAULT_STRATEGY, "filtered": zlib.Z_FILTERED, "rle": zlib.Z_RLE,
              "huffman": zlib.Z_HUFFMAN_ONLY}

DEFAULT_FILTER = "up"
DEFAULT_STRATEGY = "auto"
DEFAULT_BLOCK_BYTES = 4 * 1024 * 1024
DEFAULT_IDAT_BYTES = 1024 * 1024
MAX_DIMENSION = 2 ** 31 - 1  # PNG 规范上限

_ADLER_BASE = 65521
_IDAT_CRC = zlib.crc32(b"IDAT")
_FILTER_SUB_ROWS_BYTES = 256 * 1024  # avg、paeth 分小段算，限制临时数组大小


def adler32_combine(adler1: int, adler2: int, len2: int) -> int:
    """由 A、B 两段各自的 adler32 和 B 的长度，算出 A+B 的 adler32（与 zlib 的 adler32_combine 相同）。"""
    rem = len2 % _ADLER_BASE
    sum1 = adler1 & 0xFFFF
    sum2 = (rem * sum1) % _ADLER_BASE
    sum1 = (sum1 + (adler2 & 0xFFFF) + _ADLER_BASE - 1) % _ADLER_BASE
    sum2 = (sum2 + ((adler1 >> 16) & 0xFFFF) + ((adler2 >> 16) & 0xFFFF) + _ADLER_BASE - rem) % _ADLER_BASE
    return sum1 | (sum2 << 16)


def zlib_header(level: int) -> bytes:
    """2 字节 zlib 头：deflate、32K 窗口，FLEVEL 按压缩级别标注（只是提示，不影响解码）。"""
    cmf = 0x78
    flevel = 0 if level <= 1 else 1 if level <= 5 else 2 if level == 6 else 3
    flg = flevel << 6
    flg |= (31 - ((cmf << 8) | flg) % 31) % 31
    return bytes((cmf, flg))


def resolve_strategy(strategy: str, level: int) -> int:
    """把策略名换成 zlib 常数。"auto"：1~3 档用 RLE，4~9 档用 FILTERED（0 档不压缩，策略无所谓）。"""
    if strategy == "auto":
        return zlib.Z_RLE if level <= 3 else zlib.Z_FILTERED
    try:
        return STRATEGIES[strategy]
    except KeyError:
        raise ValueError(f"不认识的压缩策略 {strategy!r}，可选：auto、{'、'.join(STRATEGIES)}") from None


# ---------------------------------------------------------------- 行滤波（全部在原始字节上算，可向量化）

def _filter_none(cur, up, bpp, out):
    out[...] = cur


def _filter_sub(cur, up, bpp, out):
    out[:, :bpp] = cur[:, :bpp]
    np.subtract(cur[:, bpp:], cur[:, :-bpp], out=out[:, bpp:])


def _filter_up(cur, up, bpp, out):
    np.subtract(cur, up, out=out)


def _filter_avg(cur, up, bpp, out):
    np.subtract(cur[:, :bpp], up[:, :bpp] >> 1, out=out[:, :bpp])
    a = cur[:, :-bpp]
    b = up[:, bpp:]
    pred = (a & b) + ((a ^ b) >> 1)  # = (a + b) // 2，不会溢出
    np.subtract(cur[:, bpp:], pred, out=out[:, bpp:])


def _filter_paeth(cur, up, bpp, out):
    n, rb = cur.shape
    a = np.zeros((n, rb), np.int16)
    a[:, bpp:] = cur[:, :-bpp]
    b = np.empty((n, rb), np.int16)
    b[...] = up
    c = np.zeros((n, rb), np.int16)
    c[:, bpp:] = b[:, :-bpp]
    pa = np.abs(b - c)
    pb = np.abs(a - c)
    pc = np.abs(a + b - 2 * c)
    pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
    np.subtract(cur, pred.astype(np.uint8), out=out)


_FILTER_FUNCS = (_filter_none, _filter_sub, _filter_up, _filter_avg, _filter_paeth)


def _apply_filter(raw: np.ndarray, prev: np.ndarray | None, ftype: int, bpp: int, out: np.ndarray) -> None:
    """raw (n, rb) uint8 → out (n, rb)。prev 是本块上一行的原始字节（图像第一行时为 None，按全零处理）。"""
    func = _FILTER_FUNCS[ftype]
    n, rb = raw.shape
    if ftype == 0:
        func(raw, None, bpp, out)
        return
    if ftype == 1:
        func(raw, None, bpp, out)
        return
    up0 = (np.zeros((1, rb), np.uint8) if prev is None else prev.reshape(1, rb))
    if ftype == 2:
        func(raw[:1], up0, bpp, out[:1])
        if n > 1:
            func(raw[1:], raw[:-1], bpp, out[1:])
        return
    # avg、paeth 有临时数组，分小段算
    step = max(1, _FILTER_SUB_ROWS_BYTES // max(1, rb))
    func(raw[:1], up0, bpp, out[:1])
    for y in range(1, n, step):
        y1 = min(n, y + step)
        func(raw[y:y1], raw[y - 1:y1 - 1], bpp, out[y:y1])


def _encode_block(raw: np.ndarray, prev: np.ndarray | None, ftype: int, bpp: int, level: int, strategy: int,
                  header: bytes | None, last: bool, idat_bytes: int):
    """在线程池里处理一块：滤波、adler32、deflate、切 IDAT 并算 CRC。

    raw 是 (n, 行字节数) uint8（已是 PNG 字节序），或 (n, 宽×通道) 的本机字节序 uint16（这里换成大端）。
    返回 (IDAT 块列表 [(数据, crc)], 本块未压缩数据的 adler32, 本块未压缩数据的长度)。
    """
    if raw.dtype != np.uint8:
        raw = raw.astype(">u2").view(np.uint8)
    n, rb = raw.shape
    framed = np.empty((n, rb + 1), np.uint8)
    framed[:, 0] = ftype
    _apply_filter(raw, prev, ftype, bpp, framed[:, 1:])
    data = memoryview(framed).cast("B")
    adler = zlib.adler32(data)
    comp = zlib.compressobj(level, zlib.DEFLATED, -15, 8, strategy)
    parts = [comp.compress(data), comp.flush(zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH)]
    if header:
        parts.insert(0, header)
    body = memoryview(b"".join(parts))
    chunks = []
    for off in range(0, len(body), idat_bytes):
        part = body[off:off + idat_bytes]
        chunks.append((part, zlib.crc32(part, _IDAT_CRC)))
    return chunks, adler, len(data)


class _Done:
    """单线程时代替 Future。"""

    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value

    def done(self) -> bool:
        return True

    def cancel(self) -> bool:
        return False


def _discard_temp(fh, tmp_path: str) -> None:
    """写出没有正常结束（对象被回收或程序退出）时删掉临时文件。"""
    try:
        fh.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        os.remove(tmp_path)
    except OSError:
        pass


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


class PngStreamWriter:
    """逐条带写一张 PNG。

    参数：
        path：目标文件。写的过程中用同目录的临时文件，close() 成功才替换它。
        width、height：图像尺寸（像素）。
        channels：1 灰度，2 灰度加透明，3 RGB，4 RGBA。
        bit_depth：8 或 16。16 位时 write_rows 收 uint16，文件里按 PNG 规范存大端。
        compress_level：0~9，见模块说明。
        threads：压缩线程数，0 表示取 CPU 逻辑核数，1 表示不开线程（全在调用线程里算）。
        filter、strategy：见模块说明。
        block_bytes：每个压缩块的目标原始字节数（按整行取整，至少一行）。
        max_inflight：同时在途（排队、压缩中、等写出）的块数上限，0 表示线程数加 4。内存约为它乘以块大小的两到三倍。
        idat_bytes：每个 IDAT 块的最大数据长度。
        fsync：替换目标前是否把临时文件刷到磁盘（断电时不会得到半个文件）。

    用法：
        with PngStreamWriter(path, w, h, 3) as png:
            for strip in strips:          # 每条 (n, w, 3) uint8，自上而下
                png.write_rows(strip)
        # 块正常结束时自动 close()，出异常时自动 abort()
    """

    def __init__(self, path, width, height, channels, bit_depth=8, compress_level=3, threads=0, *,
                 filter: str = DEFAULT_FILTER, strategy: str = DEFAULT_STRATEGY,
                 block_bytes: int = DEFAULT_BLOCK_BYTES, max_inflight: int = 0,
                 idat_bytes: int = DEFAULT_IDAT_BYTES, fsync: bool = True):
        self.path = os.path.abspath(os.fspath(path))
        width, height, channels, bit_depth = int(width), int(height), int(channels), int(bit_depth)
        if not (1 <= width <= MAX_DIMENSION and 1 <= height <= MAX_DIMENSION):
            raise ValueError(f"PNG 的宽和高必须在 1 到 {MAX_DIMENSION} 之间，收到 {width}×{height}")
        if channels not in COLOR_TYPES:
            raise ValueError(f"通道数只能是 1、2、3、4，收到 {channels}")
        if bit_depth not in (8, 16):
            raise ValueError(f"位深只能是 8 或 16，收到 {bit_depth}")
        if filter not in FILTERS:
            raise ValueError(f"不认识的滤波方式 {filter!r}，可选：{'、'.join(FILTERS)}")
        self.width, self.height, self.channels, self.bit_depth = width, height, channels, bit_depth
        self.compress_level = max(0, min(9, int(compress_level)))
        self.filter = filter
        self.strategy = strategy
        self._zstrategy = resolve_strategy(strategy, self.compress_level)
        self._ftype = FILTERS[filter]
        threads = int(threads)
        self.threads = max(1, min(threads if threads > 0 else (os.cpu_count() or 4), 256))
        self._sample_bytes = bit_depth // 8
        self._bpp = channels * self._sample_bytes          # 每像素字节数（滤波用）
        self._row_values = width * channels                 # 每行样本数
        self.row_bytes = self._row_values * self._sample_bytes
        self.block_rows = max(1, min(height, int(block_bytes) // self.row_bytes))
        inflight = int(max_inflight)
        self._max_inflight = max(2, inflight if inflight > 0 else self.threads + 4)
        self._idat_bytes = max(64, min(int(idat_bytes), 2 ** 30))
        self._fsync = bool(fsync)
        self._rows_done = 0
        self._next_block = 0
        self._adler = 1  # 空数据的 adler32
        self._queue: deque = deque()  # (future, 块缓冲或 None, 是否最后一块)
        self._free: list[np.ndarray] = []
        self._buf: np.ndarray | None = None
        self._fill = 0
        self._prev_row: np.ndarray | None = None
        self._state = "open"  # open → closed / aborted / failed

        folder = os.path.dirname(self.path)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"要导出到的文件夹不存在：{folder}")
        fd, self._tmp = tempfile.mkstemp(prefix=os.path.basename(self.path) + ".", suffix=".tmp", dir=folder)
        self._fh = os.fdopen(fd, "wb")
        self._finalizer = weakref.finalize(self, _discard_temp, self._fh, self._tmp)
        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="splender-png") if self.threads > 1 else None
        try:
            self._fh.write(PNG_SIGNATURE)
            ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, COLOR_TYPES[channels], 0, 0, 0)
            self._write_chunk(b"IHDR", ihdr, zlib.crc32(ihdr, zlib.crc32(b"IHDR")))
        except BaseException:
            self.abort()
            raise

    # ------------------------------------------------------------ 公开接口

    @property
    def rows_written(self) -> int:
        """已经收下的行数。"""
        return self._rows_done

    @property
    def progress(self) -> float:
        """已收下的行占全部行的比例。"""
        return self._rows_done / self.height

    def write_rows(self, rows) -> None:
        """追加若干行，自上而下。rows 形状 (n, width, channels)（单通道也可以是 (n, width)），
        8 位时是 uint8，16 位时是 uint16。数据会被复制，调用返回后 rows 可以立刻复用。
        行数累计超过 height 时报 ValueError，这一批不收，已写入的不受影响。"""
        self._check_open()
        flat = self._validate(rows)
        n = flat.shape[0]
        if n == 0:
            return
        try:
            pos = 0
            while pos < n:
                if self._buf is None:
                    self._buf = self._take_buffer()
                    self._fill = 0
                k = min(n - pos, self.block_rows - self._fill)
                dst = self._buf[self._fill:self._fill + k]
                if self.bit_depth == 16:
                    dst.view(">u2")[...] = flat[pos:pos + k]  # 复制时顺便换成大端
                else:
                    dst[...] = flat[pos:pos + k]
                self._fill += k
                pos += k
                done = self._rows_done + pos
                if self._fill == self.block_rows or done == self.height:
                    buf, rows_in = self._buf, self._fill
                    self._buf = None
                    self._submit(buf[:rows_in], buf, last=(done == self.height))
            self._rows_done += n
        except BaseException:
            self._fail()
            raise

    def close(self) -> None:
        """写完：检查行数，等全部块压缩写出，写 IEND，落盘后原子替换目标文件。
        行数不对时报 ValueError，并且不留任何文件（原有目标不动）。重复调用无害。"""
        if self._state == "closed":
            return
        self._check_open()
        try:
            if self._rows_done != self.height:
                raise ValueError(f"PNG 应有 {self.height} 行，实际只写了 {self._rows_done} 行，没有生成文件")
            while self._queue:
                self._drain_one()
            self._write_chunk(b"IEND", b"", zlib.crc32(b"IEND"))
            self._fh.flush()
            if self._fsync:
                os.fsync(self._fh.fileno())
            self._fh.close()
            try:
                _replace_file(self._tmp, self.path)
            except PermissionError as exc:
                raise PermissionError(f"目标文件正被别的程序占用，无法替换：{self.path}") from exc
        except BaseException:
            self._fail()
            raise
        self._state = "closed"
        self._finalizer.detach()
        self._shutdown()

    def abort(self) -> None:
        """放弃写出：停掉压缩，删掉临时文件。原有的目标文件不动。可以重复调用。"""
        if self._state in ("closed", "aborted"):
            return
        self._state = "aborted"
        self._release()

    def __enter__(self) -> PngStreamWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    # ------------------------------------------------------------ 内部

    def _check_open(self) -> None:
        if self._state != "open":
            raise RuntimeError({"closed": "PNG 已经写完关闭了", "aborted": "PNG 写出已经放弃",
                                "failed": "PNG 写出中途出错，已经放弃"}[self._state])

    def _validate(self, rows) -> np.ndarray:
        arr = np.asarray(rows)
        if arr.ndim == 2 and self.channels == 1:
            arr = arr[:, :, None]
        if arr.ndim != 3 or arr.shape[1] != self.width or arr.shape[2] != self.channels:
            raise ValueError(f"行数据的形状应是 (行数, {self.width}, {self.channels})，收到 {arr.shape}")
        if self.bit_depth == 8 and arr.dtype != np.uint8:
            raise TypeError(f"8 位 PNG 要求 uint8 数据，收到 {arr.dtype}")
        if self.bit_depth == 16 and not (arr.dtype.kind == "u" and arr.dtype.itemsize == 2):
            raise TypeError(f"16 位 PNG 要求 uint16 数据，收到 {arr.dtype}")
        n = arr.shape[0]
        if self._rows_done + n > self.height:
            raise ValueError(f"行数超出：已写 {self._rows_done} 行，再写 {n} 行就超过了图像高度 {self.height}")
        return arr.reshape(n, self._row_values)

    def _take_buffer(self) -> np.ndarray:
        if self._free:
            return self._free.pop()
        return np.empty((self.block_rows, self.row_bytes), np.uint8)

    def _submit(self, raw: np.ndarray, owner: np.ndarray | None, last: bool) -> None:
        """提交一块。raw 是 PNG 字节序的 (n, 行字节数) uint8，或本机字节序的 (n, 宽×通道) uint16。"""
        prev = self._prev_row
        tail = raw[-1]
        if tail.dtype != np.uint8:
            tail = tail.astype(">u2").view(np.uint8)
        self._prev_row = tail.copy()
        header = zlib_header(self.compress_level) if self._next_block == 0 else None
        args = (raw, prev, self._ftype, self._bpp, self.compress_level, self._zstrategy, header, last,
                self._idat_bytes)
        future = self._pool.submit(_encode_block, *args) if self._pool is not None else _Done(_encode_block(*args))
        self._queue.append((future, owner, last))
        self._next_block += 1
        while len(self._queue) >= self._max_inflight:
            self._drain_one()
        while self._queue and self._queue[0][0].done():
            self._drain_one()

    def _drain_one(self) -> None:
        """按顺序取出最早的一块，写进文件。"""
        future, owner, last = self._queue[0]
        chunks, adler, nbytes = future.result()
        self._queue.popleft()
        self._adler = adler32_combine(self._adler, adler, nbytes)
        if last:
            trailer = struct.pack(">I", self._adler)
            part, crc = chunks[-1]
            chunks[-1] = (bytes(part) + trailer, zlib.crc32(trailer, crc))
        for part, crc in chunks:
            self._write_chunk(b"IDAT", part, crc)
        if owner is not None and len(self._free) < 4:
            self._free.append(owner)

    def _write_chunk(self, kind: bytes, data, crc: int) -> None:
        fh = self._fh
        fh.write(struct.pack(">I", len(data)))
        fh.write(kind)
        fh.write(data)
        fh.write(struct.pack(">I", crc & 0xFFFFFFFF))

    def _fail(self) -> None:
        if self._state == "open":
            self._state = "failed"
            self._release()

    def _release(self) -> None:
        """停掉线程池、丢掉在途的块、关闭并删除临时文件。"""
        for future, _, _ in self._queue:
            future.cancel()
        pending = [f for f, _, _ in self._queue if not isinstance(f, _Done)]
        if pending:
            wait_futures(pending)
        self._queue.clear()
        self._free.clear()
        self._buf = None
        self._shutdown()
        self._finalizer()  # 关闭并删除临时文件

    def _shutdown(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def write_png(path, array, **kw) -> None:
    """把整张图写成 PNG。array 形状 (H, W) 或 (H, W, C)，C 为 1~4；uint8 写 8 位，uint16 写 16 位。
    其余参数同 PngStreamWriter（compress_level、threads、filter、strategy 等）。
    直接按块读取 array，不整体复制。"""
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or not 1 <= arr.shape[2] <= 4:
        raise ValueError(f"图像数组的形状应是 (高, 宽) 或 (高, 宽, 1~4)，收到 {np.shape(array)}")
    if arr.dtype == np.uint8:
        depth = 8
    elif arr.dtype.kind == "u" and arr.dtype.itemsize == 2:
        depth = 16
    else:
        raise TypeError(f"图像数组应是 uint8 或 uint16，收到 {arr.dtype}")
    if kw.get("bit_depth", depth) != depth:
        raise ValueError(f"bit_depth={kw['bit_depth']} 和数组类型 {arr.dtype} 不一致")
    kw["bit_depth"] = depth
    height, width, channels = arr.shape
    writer = PngStreamWriter(path, width, height, channels, **kw)
    try:
        rows = writer.block_rows
        for y in range(0, height, rows):
            y1 = min(height, y + rows)
            block = arr[y:y1].reshape(y1 - y, width * channels)  # 连续数组是视图，不连续时只复制这一块
            if depth == 8:
                block = block.reshape(y1 - y, writer.row_bytes)
            writer._submit(block, None, last=(y1 == height))
            writer._rows_done = y1
        writer.close()
    except BaseException:
        writer.abort()
        raise
