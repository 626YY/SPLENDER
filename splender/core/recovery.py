"""会话锁与恢复文件的管理（不碰显卡、不碰界面）。

每个运行中的程序在自动保存目录里放一个会话锁 <会话>.lock（JSON：进程号、进程启动时间、版本），
它写的恢复文件都叫 <会话>-<序号>.splender-recover。正常退出时删掉锁和恢复文件。
下次启动时，锁对应的进程已经不在了，说明那次没有正常退出：它留下的恢复文件就是可以恢复的工程。

崩溃报告：会话开着时 faulthandler 往日志目录里的 crash-<会话>.txt 写（程序崩溃时才有内容），正常退出时空文件删掉。
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("splender.recovery")

RECOVER_SUFFIX = ".splender-recover"
LOCK_SUFFIX = ".lock"
_SIDECARS = ("-wal", "-shm", "-journal")


# ---------------------------------------------------------------- 进程
def process_started(pid: int | None = None) -> float:
    """进程的启动时间（Unix 时间戳）；默认是本进程。取不到时用现在的时间。"""
    try:
        import psutil

        return float(psutil.Process(os.getpid() if pid is None else int(pid)).create_time())
    except Exception:  # noqa: BLE001
        return time.time()


def process_alive(pid: int, started: float | None = None) -> bool:
    """这个进程还在不在。给了启动时间时还要对得上（进程号可能被后来的别的程序重用）。拿不准时当作还在。"""
    pid = int(pid)
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        import psutil
    except ImportError:
        return True
    try:
        proc = psutil.Process(pid)
        if started is not None and abs(proc.create_time() - float(started)) > 2.0:
            return False
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except Exception:  # noqa: BLE001  拒绝访问等：进程在，只是不让看
        return True


# ---------------------------------------------------------------- 文件
def remove_quiet(path: str | os.PathLike) -> None:
    for item in [str(path)] + [str(path) + side for side in _SIDECARS]:
        try:
            if os.path.exists(item):
                os.remove(item)
        except OSError:
            log.warning("删不掉：%s", item)


def file_stat(path: str | None) -> tuple[int, float] | None:
    """(字节数, 修改时间)；文件不在时 None。"""
    if not path:
        return None
    try:
        info = os.stat(path)
    except OSError:
        return None
    return int(info.st_size), round(float(info.st_mtime), 3)


def session_of(name: str) -> str:
    """从恢复文件名里取会话名（<会话>-<序号>.splender-recover）。"""
    stem = name[:-len(RECOVER_SUFFIX)] if name.endswith(RECOVER_SUFFIX) else name
    head, _sep, tail = stem.rpartition("-")
    return head if tail.isdigit() and head else stem


class Session:
    """本次运行的会话锁。"""

    def __init__(self, directory: str | os.PathLike, crash_dir: str | os.PathLike | None = None) -> None:
        self.directory = Path(directory)
        self.pid = os.getpid()
        self.started = process_started()
        self.id = "%s-%d" % (time.strftime("%Y%m%d-%H%M%S", time.localtime(self.started)), self.pid)
        self.lock_path = self.directory / (self.id + LOCK_SUFFIX)
        self.crash_path = Path(crash_dir) / ("crash-%s.txt" % self.id) if crash_dir is not None else None
        self._serial = 0
        self.created = False

    def create(self) -> None:
        from .. import __version__

        self.directory.mkdir(parents=True, exist_ok=True)
        info = {"pid": self.pid, "started": self.started, "version": __version__,
                "crash": str(self.crash_path) if self.crash_path is not None else ""}
        tmp = self.lock_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.lock_path)
        self.created = True

    def new_recovery_path(self) -> str:
        """这个会话的下一个恢复文件路径（文件名带会话名，别的程序据此知道它还有主人）。"""
        self._serial += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        return str(self.directory / ("%s-%d%s" % (self.id, self._serial, RECOVER_SUFFIX)))

    def owns(self, path: str | os.PathLike) -> bool:
        return session_of(Path(path).name) == self.id

    def close(self) -> None:
        """正常退出：删掉锁，删掉本会话留下的恢复文件。"""
        for path in self.directory.glob(self.id + "-*" + RECOVER_SUFFIX):
            remove_quiet(path)
        try:
            self.lock_path.unlink(missing_ok=True)
        except OSError:
            log.warning("会话锁删不掉：%s", self.lock_path)
        self.created = False


# ---------------------------------------------------------------- 找可以恢复的
@dataclass
class Recoverable:
    """一份可以恢复的自动保存。"""

    path: str
    name: str                      # 工程名
    saved_at: float                # 自动保存的时间
    base: str | None               # 它是相对哪个工程文件的改动（没存过的工程是 None）
    base_exists: bool
    base_changed: bool             # 原工程文件在那之后又被改过
    pages: int
    size: int                      # 恢复文件的字节数
    version: str                   # 写它的程序版本
    session: str
    crash: str | None = None       # 那次运行的崩溃报告（有内容时）
    extra: dict = field(default_factory=dict)


def read_recoverable(path: str | os.PathLike) -> Recoverable | None:
    """读一份恢复文件的概况。没有写完过一次（不完整）的返回 None。"""
    from ..doc.storage import ProjectFile
    from ..engine.autosave import RECOVER_JOURNAL

    path = str(path)
    try:
        storage = ProjectFile(path, journal=RECOVER_JOURNAL)
    except Exception:  # noqa: BLE001
        log.warning("恢复文件打不开：%s", path, exc_info=True)
        return None
    try:
        meta = storage.read_meta()
    except Exception:  # noqa: BLE001
        log.warning("恢复文件读不出：%s", path, exc_info=True)
        return None
    finally:
        storage.close()
    info = meta.get("recover")
    if not isinstance(info, dict) or "project" not in meta:
        return None
    base = info.get("base") or None
    stat = file_stat(base)
    recorded = info.get("base_stat")
    return Recoverable(path=path, name=str(info.get("name") or "未命名"), saved_at=float(info.get("saved_at") or 0.0),
                       base=base, base_exists=stat is not None,
                       base_changed=bool(base) and stat is not None and recorded is not None
                       and (list(stat) != list(recorded)),
                       pages=int(info.get("pages") or 0), size=_size(path), version=str(info.get("app") or ""),
                       session=session_of(os.path.basename(path)), extra=info)


def _size(path: str) -> int:
    total = 0
    for item in [path] + [path + side for side in _SIDECARS]:
        try:
            total += os.path.getsize(item)
        except OSError:
            pass
    return total


def scan(directory: str | os.PathLike, own: Session | None = None, keep: int = 5) -> tuple[list[Recoverable], list[dict]]:
    """找上次（或更早）没正常退出留下的自动保存。

    返回 (可以恢复的列表（新的在前）, 这次刚发现的没正常退出的会话 [{session, crash}])。
    已经不在的进程的锁会被删掉（所以「刚发现」只报一次）；没写完过的恢复文件删掉；超过 keep 份时删最早的。
    """
    directory = Path(directory)
    if not directory.is_dir():
        return [], []
    live: set[str] = set()
    ended: list[dict] = []
    for lock in directory.glob("*" + LOCK_SUFFIX):
        session = lock.name[:-len(LOCK_SUFFIX)]
        if own is not None and session == own.id:
            live.add(session)
            continue
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
        pid = int(info.get("pid") or 0)
        if pid and process_alive(pid, info.get("started")):
            live.add(session)
            continue
        crash = info.get("crash") or ""
        report = None
        if crash and os.path.isfile(crash):
            try:
                if os.path.getsize(crash) > 0:
                    report = crash
                else:
                    os.remove(crash)
            except OSError:
                pass
        ended.append({"session": session, "crash": report, "version": info.get("version", "")})
        try:
            lock.unlink()
        except OSError:
            log.warning("旧会话锁删不掉：%s", lock)
    crashes = {item["session"]: item["crash"] for item in ended}
    found: list[Recoverable] = []
    for path in directory.glob("*" + RECOVER_SUFFIX):
        session = session_of(path.name)
        if session in live:
            continue
        item = read_recoverable(path)
        if item is None:
            remove_quiet(path)
            continue
        item.crash = crashes.get(session)
        found.append(item)
    found.sort(key=lambda r: r.saved_at, reverse=True)
    for stale in found[max(1, int(keep)):]:
        log.info("自动保存超过 %d 份，删掉最早的：%s", keep, stale.path)
        remove_quiet(stale.path)
    return found[:max(1, int(keep))], ended


def restore_target(item: Recoverable, folder: str | None = None) -> str:
    """恢复出来的工程放哪：给了 folder 就放那里；否则放原工程旁边；没存过的工程放 folder 或用户的文档目录。"""
    stamp = time.strftime("%m-%d %H%M", time.localtime(item.saved_at or time.time()))
    candidates = []
    if folder:
        candidates.append(folder)
    if item.base:
        candidates.append(os.path.dirname(item.base))
    candidates.append(os.path.join(os.path.expanduser("~"), "Documents"))
    candidates.append(os.path.expanduser("~"))
    target_dir = next((c for c in candidates if c and os.path.isdir(c) and os.access(c, os.W_OK)), candidates[-1])
    stem = "%s（恢复 %s）" % (item.name, stamp)
    path = os.path.join(target_dir, stem + ".splender")
    number = 2
    while os.path.exists(path):
        path = os.path.join(target_dir, "%s %d.splender" % (stem, number))
        number += 1
    return path


def describe_age(seconds: float) -> str:
    """多久以前（界面上用）。"""
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return "刚才"
    if seconds < 3600:
        return "%d 分钟前" % round(seconds / 60)
    if seconds < 86400:
        return "%d 小时前" % round(seconds / 3600)
    return "%d 天前" % round(seconds / 86400)
