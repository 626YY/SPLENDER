"""打一份解压就能用的 SPLENDER：自带 Python 3.11 和用到的库，拷到别的电脑（Windows 10/11 64 位）双击 SPLENDER.exe 就能用。

用法：python tools/make_package.py [--out 目录] [--name 名字] [--no-zip]

放进去的东西：
- 程序：SPLENDER.exe、README.md、splender、tools、tests_new、docs（不带 evidence）、runtime/site-packages（内置的几个库）。
- runtime/python：运行这个脚本的 Python 3.11 精简的一份（标准库去掉测试、IDLE、tkinter 等）。程序用到的库（PACKAGES）
  按发行包的文件清单整包拷进 Lib/site-packages，依赖按 Requires-Dist 自动带上（连同 .libs、dist-info）；
  PySide6 只留用到的 Qt 模块和插件。
- runtime/python/python311._pth：只认包里的路径，别的电脑上装的 Python、用户目录里的库都混不进来。
  启动器（tools/launcher/launcher.c）先找 runtime/python。
- 根目录的「使用说明.txt」：tools/package_readme.txt 填上名字和 docs/CHANGELOG.md 最上面那段（本次更新）。
不放：__pycache__、.devuser（开发时的用户目录）、legacy（旧原型）、docs/evidence（自检截图，跑测试会重新生成）。
"""
from __future__ import annotations

import argparse
import fnmatch
import importlib.metadata as metadata
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: 程序用到的第三方库（import 名）；它们依赖的发行包自动带上
PACKAGES = ("PySide6", "numpy", "numba", "scipy", "PIL", "cv2", "zstandard", "psutil", "lz4")
#: PySide6 留下的 Qt 模块（Qt6<名字>.dll、Qt<名字>.pyd/.pyi）；别的模块、网页引擎、QML、多媒体、设计器这些都不带
QT_MODULES = {"Core", "Gui", "Widgets", "Svg", "SvgWidgets", "OpenGL", "OpenGLWidgets", "Network", "Concurrent", "Xml",
              "PrintSupport"}
QT_PLUGINS = {"platforms", "styles", "imageformats", "iconengines", "generic", "platforminputcontexts"}
QT_DROP_DIRS = {"resources", "qml", "metatypes", "translations", "include", "typesystems", "glue", "doc", "lib", "scripts"}
#: 翻译目录里只留 Qt 自带窗口（消息框按钮、文件对话框等）的中文
QT_KEEP = ("PySide6/translations/qtbase_zh_CN.qm",)
QT_DROP_FILES = ("*.exe", "*.lib", "pyside6qml*", "av*.dll", "sw*.dll")
OTHER_DROP = ("cv2/opencv_videoio_ffmpeg*.dll",)          # OpenCV 的视频读写，用不到
STDLIB_DROP = {"test", "idlelib", "tkinter", "turtledemo", "ensurepip", "site-packages"}
DLLS_DROP = ("_tkinter*", "tcl*", "tk*")
PYTHON_FILES = ("python311.dll", "python3.dll", "python.exe", "pythonw.exe", "vcruntime140.dll", "vcruntime140_1.dll",
                "LICENSE.txt")
PROGRAM_DIRS = ("splender", "tools", "tests_new", "docs", "runtime/site-packages")
PROGRAM_FILES = ("SPLENDER.exe", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md")
SKIP = {"__pycache__", "evidence", ".devuser", "legacy", ".cache"}
#: pip 装库时记下的「从哪装的」（direct_url.json 里是本机的文件路径），对用包的人没用，不带
SKIP_FILES = {"direct_url.json"}
#: 给自己用的工具和内部文档，不放进包里
PRIVATE = {"tools/baidu_upload.py", "tools/export_open_source.py", "docs/ROADMAP.md"}
#: python311._pth：相对它所在的 runtime/python。..\.. 是程序目录，..\site-packages 是内置的几个库。
#: 不写 import site：写了会把这台电脑用户目录里的库（%APPDATA%\Python\Python311\site-packages）也加进来，
#: 那里的旧版本会盖掉包里的；带的这些库都不靠 .pth 文件
PTH = ("Lib", "DLLs", "Lib\\site-packages", "..\\site-packages", "..\\..")
VENDOR = ROOT / "runtime" / "site-packages"


def _skip(directory, names) -> set:
    here = Path(directory)
    rel = here.relative_to(ROOT) if here.is_relative_to(ROOT) else Path()
    return {name for name in names
            if name in SKIP or name in SKIP_FILES or name.endswith(".pyc") or (rel / name).as_posix() in PRIVATE}


def copy_program(dest: Path) -> None:
    dest.mkdir(parents=True)
    for name in PROGRAM_FILES:
        shutil.copy2(ROOT / name, dest / name)
    for name in PROGRAM_DIRS:
        shutil.copytree(ROOT / name, dest / name, ignore=_skip)


def copy_python(home: Path) -> None:
    """精简的 Python：解释器、DLLs、标准库，加上 ._pth。"""
    base = Path(sys.base_prefix)
    home.mkdir(parents=True)
    for name in PYTHON_FILES:
        if (base / name).is_file():
            shutil.copy2(base / name, home / name)
    (home / "DLLs").mkdir()
    for item in (base / "DLLs").iterdir():
        if item.is_file() and not any(fnmatch.fnmatch(item.name.lower(), pattern) for pattern in DLLS_DROP):
            shutil.copy2(item, home / "DLLs" / item.name)
    lib = base / "Lib"

    def ignore(directory, names) -> set:
        top = Path(directory) == lib
        return {name for name in names if name == "__pycache__" or name.endswith(".pyc") or (top and name in STDLIB_DROP)}

    shutil.copytree(lib, home / "Lib", ignore=ignore)
    (home / "Lib" / "site-packages").mkdir()
    (home / "python311._pth").write_text("\n".join(PTH) + "\n", encoding="utf-8")


def _requirement(text: str):
    """依赖说明 → 名字；可选的（extra）和这台电脑用不到的返回 None。没有 pip 时按名字取，带条件的只认 extra。"""
    try:
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        name, _, marker = text.partition(";")
        if "extra" in marker:
            return None
        found = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", name)
        return found.group(1) if found else None
    requirement = Requirement(text)
    if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
        return None
    return requirement.name


def _key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def distributions() -> list:
    """要带的发行包：PACKAGES 各自所属的，加上它们在这台电脑上用得到的依赖（可选的 extra 不带）；
    runtime/site-packages 里内置的库的依赖也算上（比如降噪库要的 cffi），内置的那些本身不重复带。"""
    owners = metadata.packages_distributions()
    queue = []
    for name in PACKAGES:
        found = owners.get(name)
        if not found:
            raise SystemExit("找不到 %s 属于哪个发行包，先装上它" % name)
        queue.extend(found)
    vendored = set()
    for dist in metadata.distributions(path=[str(VENDOR)]):
        vendored.add(_key(dist.metadata["Name"]))
        queue.extend(needed for needed in map(_requirement, dist.requires or []) if needed)
    seen: dict = {}
    while queue:
        name = queue.pop()
        key = _key(name)
        if key in seen or key in vendored:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        seen[key] = dist
        for text in dist.requires or []:
            needed = _requirement(text)
            if needed:
                queue.append(needed)
    return sorted(seen.values(), key=lambda d: d.metadata["Name"].lower())


def _dropped(rel: Path) -> bool:
    """PySide6 里用不到的部分、别的库里用不到的大文件。"""
    text = rel.as_posix()
    if any(fnmatch.fnmatch(text, pattern) for pattern in OTHER_DROP):
        return True
    parts = rel.parts
    if parts[0] != "PySide6" or len(parts) < 2:
        return False
    if text in QT_KEEP:
        return False
    if parts[1] in QT_DROP_DIRS:
        return True
    if parts[1] == "plugins":
        return len(parts) < 3 or parts[2] not in QT_PLUGINS
    if len(parts) > 2:
        return False
    name = parts[1]
    if any(fnmatch.fnmatch(name.lower(), pattern) for pattern in QT_DROP_FILES):
        return True
    match = re.match(r"Qt6(\w+)\.dll$", name) or re.match(r"Qt(\w+)\.(pyd|pyi)$", name)
    return bool(match) and match.group(1) not in QT_MODULES


def copy_distribution(dist, site: Path, stats: dict) -> None:
    """按发行包的文件清单拷进 site（清单里指到 site-packages 外面的，比如 Scripts 下的命令行工具，不拷）。"""
    files = dist.files
    if files is None:
        raise SystemExit("发行包 %s 没有文件清单（RECORD）" % dist.metadata["Name"])
    for item in files:
        if ".." in item.parts or "__pycache__" in item.parts or item.suffix == ".pyc" or item.name in SKIP_FILES:
            continue
        source = Path(dist.locate_file(item))
        if not source.is_file():
            continue
        rel = Path(*item.parts)
        size = source.stat().st_size
        if _dropped(rel):
            stats["dropped"] += size
            continue
        target = site / rel
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        stats["kept"] += size


def latest_changes() -> str:
    """docs/CHANGELOG.md 最上面那段（第一个二级标题到下一个二级标题）。"""
    text = (ROOT / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = re.split(r"^## ", text, flags=re.M)
    if len(sections) < 2:
        return "（没有更新记录）"
    title, _, body = sections[1].partition("\n")
    return (title.strip() + "\n" + body.strip()).strip()


def write_readme(folder: Path, name: str) -> Path:
    template = (ROOT / "tools" / "package_readme.txt").read_text(encoding="utf-8")
    target = folder / "使用说明.txt"
    # 带 BOM：老版本的记事本也认得是 UTF-8
    target.write_text(template.format(name=name, changes=latest_changes()), encoding="utf-8-sig")
    return target


def privacy_scan(folder: Path) -> list[str]:
    """包里不能有这台电脑的信息：用户名、用户目录（原样、URL 编码、UTF-16 都查）。返回有问题的文件。"""
    import os
    import urllib.parse

    home = os.path.expanduser("~")
    user = os.path.basename(home)
    needles = {home, home.replace("\\", "/"), urllib.parse.quote(home.replace("\\", "/"))}
    if len(user) >= 3 or any(ord(ch) > 127 for ch in user):      # 太短的英文名会误报，不单独查
        needles |= {user, urllib.parse.quote(user)}
    patterns = set()
    for needle in needles:
        if needle:
            patterns.add(needle.encode("utf-8"))
            patterns.add(needle.encode("utf-16-le"))
    found = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.stat().st_size > 8 << 20:
            continue
        data = path.read_bytes()
        if any(pattern in data for pattern in patterns):
            found.append(str(path.relative_to(folder)))
    return found


def make_zip(folder: Path, target: Path) -> None:
    """压成 zip，里面的根目录是 SPLENDER。"""
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(folder.name) / path.relative_to(folder)).as_posix())


def folder_size(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="打一份解压就能用的 SPLENDER")
    parser.add_argument("--out", default=str(ROOT.parent / "SPLENDER_dist"), help="放到哪个目录")
    parser.add_argument("--name", default="", help="包名（默认 SPLENDER-版本-日期）")
    parser.add_argument("--no-zip", action="store_true", help="只整理文件夹，不压缩")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 11) or sys.maxsize < 2 ** 32:
        raise SystemExit("要用 64 位 Python 3.11 运行（启动器只认 python311.dll）")
    sys.path.insert(0, str(ROOT))
    from splender import __version__

    name = args.name or "SPLENDER-%s-%s" % (__version__, time.strftime("%Y%m%d"))
    out = Path(args.out)
    work = out / name
    if work.exists():
        shutil.rmtree(work)
    folder = work / "SPLENDER"
    started = time.perf_counter()
    copy_program(folder)
    home = folder / "runtime" / "python"
    copy_python(home)
    stats = {"kept": 0, "dropped": 0}
    dists = distributions()
    for dist in dists:
        copy_distribution(dist, home / "Lib" / "site-packages", stats)
    readme = write_readme(folder, name)
    shutil.copy2(readme, out / ("%s-使用说明.txt" % name))
    print("发行包：%s" % "、".join("%s %s" % (d.metadata["Name"], d.version) for d in dists))
    print("库：带上 %.0f MB，去掉用不到的 %.0f MB" % (stats["kept"] / 2 ** 20, stats["dropped"] / 2 ** 20))
    print("文件夹 %s：%.0f MB" % (folder, folder_size(folder) / 2 ** 20))
    leaks = privacy_scan(folder)
    if leaks:
        print("包里有这台电脑的信息（用户名或用户目录），不压缩：")
        for item in leaks:
            print("  " + item)
        return 1
    if not args.no_zip:
        target = out / (name + ".zip")
        make_zip(folder, target)
        print("压缩包 %s：%.0f MB" % (target, target.stat().st_size / 2 ** 20))
    print("用时 %.0f 秒" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
