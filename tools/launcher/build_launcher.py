"""构建启动器 SPLENDER.exe。

用法：python tools/launcher/build_launcher.py [--out 路径] [--keep 目录] [--vcvars vcvars64.bat 路径]

需要 Visual Studio 2022 的「使用 C++ 的桌面开发」和 Windows SDK。先用 vcvars64.bat 取得
cl、rc、link 的环境，再按 x64、图形界面子系统、Unicode 编译 launcher.c 和 launcher.rc。
图标取 splender/resources/logo/splender.ico（没有时先运行 tools/make_logo.py），
版本号取 splender/__init__.py 的 __version__。

运行库用「静态 vcruntime + 系统 UCRT」（/MT 编译，链接时把 libucrt.lib 换成 ucrt.lib）：
不依赖 vcruntime140.dll，又和 python311.dll 共用同一份 ucrtbase.dll（原因见 launcher.c 开头）。
清单和 python.exe 的一致：asInvoker、声明支持到 Windows 10、长路径、通用控件 6。
不声明 DPI 感知，由 Qt 自己设置。
"""
from __future__ import annotations

import argparse
import locale
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = HERE / "launcher.c"
RESOURCE = HERE / "launcher.rc"
ICON_DIR = ROOT / "splender" / "resources" / "logo"
ICON = ICON_DIR / "splender.ico"
INIT_FILE = ROOT / "splender" / "__init__.py"
DEFAULT_OUT = ROOT / "SPLENDER.exe"

#: 子进程不弹控制台窗口（从图形界面里调用本脚本时也不会闪黑窗）
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

MANIFEST = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{e2011457-1546-43c5-a5fe-008deee3d3f0}"/>
      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"/>
      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"/>
      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"/>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>
    </windowsSettings>
  </application>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls"
                        version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
    </dependentAssembly>
  </dependency>
</assembly>
"""

COMPILE_FLAGS = (
    "/nologo", "/c", "/utf-8", "/W4", "/WX", "/sdl", "/O2", "/GS", "/guard:cf", "/MT",
    "/DUNICODE", "/D_UNICODE", "/DNDEBUG", "/D_WIN32_WINNT=0x0603", "/DWINVER=0x0603",
)
LINK_FLAGS = (
    "/nologo", "/MACHINE:X64", "/DYNAMICBASE", "/NXCOMPAT", "/HIGHENTROPYVA", "/GUARD:CF",
    "/OPT:REF", "/OPT:ICF", "/INCREMENTAL:NO", "/RELEASE",
    "/NODEFAULTLIB:libucrt.lib",
)
LIBS = ("ucrt.lib", "kernel32.lib", "user32.lib", "advapi32.lib", "shell32.lib")


class BuildError(RuntimeError):
    pass


# ---------------------------------------------------------------- 工具链

def find_vcvars(explicit: str | os.PathLike | None = None) -> Path:
    """找 vcvars64.bat：参数 > 环境变量 SPLENDER_VCVARS > vswhere > 常见安装位置。"""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("SPLENDER_VCVARS"):
        candidates.append(Path(os.environ["SPLENDER_VCVARS"]))
    x86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    vswhere = x86 / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        proc = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath", "-utf8"],
            capture_output=True, creationflags=NO_WINDOW)
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            if line.strip():
                candidates.append(Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
    for base in (Path(os.environ.get("ProgramFiles", r"C:\Program Files")), x86):
        candidates.extend(sorted(base.glob("Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvars64.bat"),
                                 reverse=True))
    for path in candidates:
        if path.is_file():
            return path
    raise BuildError("没有找到 vcvars64.bat。需要安装 Visual Studio 2022 的「使用 C++ 的桌面开发」，"
                     "或用 --vcvars / 环境变量 SPLENDER_VCVARS 指定它的位置。")


def msvc_env(vcvars: Path) -> dict[str, str]:
    """运行 vcvars64.bat，取回它设好的环境变量（cmd /u 让 set 输出 UTF-16，中文路径不乱码）。"""
    command = f'cmd.exe /d /u /s /c ""{vcvars}" >nul 2>&1 && set"'
    proc = subprocess.run(command, capture_output=True, creationflags=NO_WINDOW)
    env: dict[str, str] = {}
    for line in proc.stdout.decode("utf-16-le", "replace").splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            env[key] = value
    keys = {k.upper() for k in env}
    if proc.returncode != 0 or not {"INCLUDE", "LIB", "PATH"} <= keys:
        raise BuildError(f"vcvars64.bat 没有设好编译环境：{vcvars}")
    env["VSLANG"] = "1033"  # 编译器提示用英文，输出好解析
    return env


def tool(env: dict[str, str], name: str) -> str:
    """在编译环境的 PATH 里找工具的完整路径（CreateProcess 不按新环境的 PATH 找程序）。"""
    path = next((v for k, v in env.items() if k.upper() == "PATH"), "")
    found = shutil.which(name, path=path)
    if not found:
        raise BuildError(f"编译环境里没有 {name}")
    return found


def run(args: list[str], env: dict[str, str], cwd: Path) -> str:
    proc = subprocess.run(args, env=env, cwd=str(cwd), capture_output=True, creationflags=NO_WINDOW)
    encoding = locale.getpreferredencoding(False)
    output = (proc.stdout + proc.stderr).decode(encoding, "replace").strip()
    if proc.returncode != 0:
        raise BuildError(f"{Path(args[0]).name} 失败（返回 {proc.returncode}）：\n{output}")
    return output


# ---------------------------------------------------------------- 版本号

def read_version() -> str:
    text = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", text, re.M)
    if not match:
        raise BuildError(f"{INIT_FILE} 里没有 __version__")
    return match.group(1)


def version_header(version: str) -> str:
    lead = re.match(r"\d+(?:\.\d+)*", version)
    numbers = [min(int(x), 65535) for x in lead.group(0).split(".")][:4] if lead else []
    numbers += [0] * (4 - len(numbers))
    text = version.replace("\\", "").replace('"', "")
    return (
        "/* 由 build_launcher.py 生成 */\n"
        f"#define SPLENDER_VER_MAJOR {numbers[0]}\n"
        f"#define SPLENDER_VER_MINOR {numbers[1]}\n"
        f"#define SPLENDER_VER_PATCH {numbers[2]}\n"
        f"#define SPLENDER_VER_BUILD {numbers[3]}\n"
        f'#define SPLENDER_VER_STRING "{text}"\n'
    )


# ---------------------------------------------------------------- 构建

def install(built: Path, out: Path) -> None:
    """先写临时文件再替换，替换失败不会留下半个程序。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".new")
    shutil.copyfile(built, temp)
    try:
        os.replace(temp, out)
    except PermissionError as error:
        temp.unlink(missing_ok=True)
        raise BuildError(f"{out} 正在使用中，关掉程序后再构建") from error


def build(out: Path = DEFAULT_OUT, keep: Path | None = None, vcvars: str | None = None) -> Path:
    if not ICON.is_file():
        raise BuildError(f"缺少图标 {ICON}，先运行 python tools/make_logo.py")
    env = msvc_env(find_vcvars(vcvars))
    cl, rc, link = tool(env, "cl.exe"), tool(env, "rc.exe"), tool(env, "link.exe")
    work = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="splender-launcher-"))
    try:
        work.mkdir(parents=True, exist_ok=True)
        (work / "splender_version.h").write_text(version_header(read_version()), encoding="utf-8")
        manifest = work / "splender.manifest"
        manifest.write_text(MANIFEST, encoding="utf-8")
        obj, res, exe = work / "launcher.obj", work / "launcher.res", work / "SPLENDER.exe"
        run([cl, *COMPILE_FLAGS, f"/Fo{obj}", str(SOURCE)], env, work)
        run([rc, "/nologo", "/c65001", "/I", str(work), "/I", str(ICON_DIR), "/fo", str(res), str(RESOURCE)],
            env, work)
        run([link, *LINK_FLAGS, "/SUBSYSTEM:WINDOWS", "/ENTRY:wWinMainCRTStartup",
             "/MANIFEST:EMBED", "/MANIFESTUAC:NO", f"/MANIFESTINPUT:{manifest}",
             f"/OUT:{exe}", str(obj), str(res), *LIBS], env, work)
        install(exe, Path(out))
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)
    return Path(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="构建启动器 SPLENDER.exe")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出路径")
    parser.add_argument("--keep", type=Path, default=None, help="保留中间文件的目录")
    parser.add_argument("--vcvars", default=None, help="vcvars64.bat 的路径")
    args = parser.parse_args()
    try:
        out = build(args.out, args.keep, args.vcvars)
    except BuildError as error:
        print(error)
        return 1
    print(f"{out}  {out.stat().st_size} 字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
