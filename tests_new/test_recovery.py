"""会话锁、找可以恢复的自动保存、清理、恢复到哪。不用显卡。"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path  # noqa: E402

ensure_vendor_path()
from splender.core import recovery  # noqa: E402
from splender.doc.storage import ProjectFile  # noqa: E402


def dead_pid() -> tuple[int, float]:
    """起一个马上就退出的进程，返回它的进程号和启动时间（那时它已经不在了）。"""
    proc = subprocess.Popen([sys.executable, "-c", "pass"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    started = recovery.process_started(proc.pid)
    proc.wait()
    return proc.pid, started


def write_recovery(path: str, name: str, saved_at: float, base: str | None = None, complete: bool = True) -> None:
    with ProjectFile(path, create=True) as storage:
        storage.write_pages({(1, 0, 0, 0, 0): b"\x01" * 64})
        if complete:
            storage.write_meta({"format": 1, "project": {"texture_sets": []},
                                "recover": {"name": name, "saved_at": saved_at, "base": base, "pages": 1,
                                            "app": "test"}})


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="splender_recovery_")
        self.logs = tempfile.mkdtemp(prefix="splender_logs_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.rmtree(self.logs, ignore_errors=True)

    def fake_dead_session(self, name: str, crash_text: str = "") -> str:
        pid, started = dead_pid()
        session = "20260101-000000-%d" % pid
        crash = os.path.join(self.logs, "crash-%s.txt" % session)
        with open(crash, "w", encoding="utf-8") as handle:
            handle.write(crash_text)
        Path(self.dir, session + recovery.LOCK_SUFFIX).write_text(
            json.dumps({"pid": pid, "started": started, "version": "test", "crash": crash}), encoding="utf-8")
        return session

    def test_process_alive(self):
        self.assertTrue(recovery.process_alive(os.getpid()))
        pid, started = dead_pid()
        self.assertFalse(recovery.process_alive(pid, started))
        self.assertFalse(recovery.process_alive(0))
        # 进程号对得上、启动时间对不上：是别的程序重用了这个号
        self.assertFalse(recovery.process_alive(os.getppid(), recovery.process_started(os.getppid()) - 3600))

    def test_session_lock_and_names(self):
        session = recovery.Session(self.dir, self.logs)
        session.create()
        self.assertTrue(session.lock_path.is_file())
        first = session.new_recovery_path()
        second = session.new_recovery_path()
        self.assertNotEqual(first, second)
        self.assertTrue(session.owns(first))
        self.assertEqual(recovery.session_of(os.path.basename(first)), session.id)
        write_recovery(first, "我的", time.time())
        items, ended = recovery.scan(self.dir, session)
        self.assertEqual(items, [], "活着的会话的恢复文件不算")
        self.assertEqual(ended, [])
        session.close()
        self.assertFalse(session.lock_path.exists())
        self.assertFalse(os.path.exists(first), "正常退出时删掉自己的恢复文件")

    def test_scan_dead_sessions(self):
        own = recovery.Session(self.dir, self.logs)
        own.create()
        crashed = self.fake_dead_session("崩", crash_text="Fatal Python error: Segmentation fault")
        killed = self.fake_dead_session("杀")
        base = os.path.join(self.dir, "原工程.splender")
        with ProjectFile(base, create=True) as storage:
            storage.write_meta({"format": 1})
        good = os.path.join(self.dir, crashed + "-1" + recovery.RECOVER_SUFFIX)
        write_recovery(good, "角色", time.time() - 120, base=base)
        broken = os.path.join(self.dir, killed + "-1" + recovery.RECOVER_SUFFIX)
        write_recovery(broken, "没写完", time.time(), complete=False)
        items, ended = recovery.scan(self.dir, own)
        self.assertEqual([item.name for item in items], ["角色"])
        item = items[0]
        self.assertEqual(item.base, base)
        self.assertTrue(item.base_exists)
        self.assertTrue(item.crash and item.crash.endswith("crash-%s.txt" % crashed))
        self.assertFalse(os.path.exists(broken), "没写完的恢复文件删掉")
        self.assertEqual(sorted(e["session"] for e in ended), sorted([crashed, killed]))
        self.assertFalse(os.path.exists(os.path.join(self.logs, "crash-%s.txt" % killed)), "空的崩溃报告删掉")
        for session in (crashed, killed):
            self.assertFalse(Path(self.dir, session + recovery.LOCK_SUFFIX).exists(), "死掉的会话锁删掉")
        self.assertTrue(own.lock_path.exists(), "自己的会话锁不动")
        # 再找一次：会话只报一次，恢复文件还在
        items2, ended2 = recovery.scan(self.dir, own)
        self.assertEqual([i.path for i in items2], [good])
        self.assertEqual(ended2, [])
        own.close()

    def test_keep_limit_and_target(self):
        session = self.fake_dead_session("多")
        paths = []
        for index in range(4):
            path = os.path.join(self.dir, "%s-%d%s" % (session, index + 1, recovery.RECOVER_SUFFIX))
            write_recovery(path, "第%d份" % index, time.time() - 1000 + index * 10)
            paths.append(path)
        items, _ended = recovery.scan(self.dir, None, keep=2)
        self.assertEqual([item.name for item in items], ["第3份", "第2份"], "新的在前，只留两份")
        self.assertFalse(os.path.exists(paths[0]))
        self.assertFalse(os.path.exists(paths[1]))
        # 恢复到哪：原工程旁边；重名时加编号
        item = items[0]
        item.base = os.path.join(self.dir, "原工程.splender")
        target = recovery.restore_target(item)
        self.assertEqual(os.path.dirname(target), self.dir)
        self.assertTrue(os.path.basename(target).startswith("第3份（恢复 "))
        Path(target).write_bytes(b"")
        self.assertNotEqual(recovery.restore_target(item), target)
        custom = os.path.join(self.dir, "恢复到这里")
        os.makedirs(custom)
        self.assertEqual(os.path.dirname(recovery.restore_target(item, custom)), custom)
        self.assertEqual(recovery.describe_age(30), "刚才")
        self.assertEqual(recovery.describe_age(600), "10 分钟前")


if __name__ == "__main__":
    unittest.main()
