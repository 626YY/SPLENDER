"""在桌面（可选：开始菜单）放一个 SPLENDER 快捷方式。用法：python tools/make_shortcut.py [--start-menu]"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "SPLENDER.exe"
ICON = ROOT / "splender" / "resources" / "logo" / "splender.ico"


def make(folder_expr: str) -> str:
    script = (
        "$dir = %s; "
        "$shell = New-Object -ComObject WScript.Shell; "
        "$lnk = $shell.CreateShortcut((Join-Path $dir 'SPLENDER.lnk')); "
        "$lnk.TargetPath = '%s'; $lnk.WorkingDirectory = '%s'; $lnk.IconLocation = '%s,0'; "
        "$lnk.Description = 'SPLENDER：雕刻、绘制、材质、渲染'; $lnk.Save(); "
        "Write-Output (Join-Path $dir 'SPLENDER.lnk')"
    ) % (folder_expr, EXE, ROOT, ICON)
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode != 0:
        raise SystemExit("创建快捷方式失败：" + result.stderr.strip())
    return result.stdout.strip()


def main() -> None:
    if not EXE.is_file():
        raise SystemExit("找不到 %s，先运行 tools/launcher/build_launcher.py" % EXE)
    print("已创建：", make("[Environment]::GetFolderPath('Desktop')"))
    if "--start-menu" in sys.argv:
        print("已创建：", make("[Environment]::GetFolderPath('Programs')"))


if __name__ == "__main__":
    main()
