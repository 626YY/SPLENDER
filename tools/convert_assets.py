"""把 Blender 发行包里的环境光资源转成 SPLENDER 自带格式。

来源（Blender 4.3）：datafiles/studiolights/
- world/*.exr   → splender/resources/hdri/<名>.npz（键 rgb，float16，H×W×3，线性，行序自上而下）
                  + <名>.png 缩略图（256×128，AgX 显示变换后的 sRGB）
- matcap/*.exr  → splender/resources/matcap/<名>.npz + <名>.png（128×128，标准 sRGB）
- studio/*.sl   → splender/resources/studio/<名>.json + <名>.png（128×128 灰球预览）
- world/license.txt、matcap/license.txt → hdri/LICENSE.txt、matcap/LICENSE.txt

用法::

    python tools/convert_assets.py [--source <studiolights 目录>] [--out <resources 目录>]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from splender.paths import RESOURCE_DIR, ensure_vendor_path  # noqa: E402

ensure_vendor_path()

from PIL import Image  # noqa: E402

from splender.engine import ibl  # noqa: E402

DEFAULT_SOURCE = Path("C:/Program Files/Blender Foundation/Blender 4.3/4.3/datafiles/studiolights")


def save_npz(path: Path, rgb: np.ndarray) -> None:
    """存成 float16，超出 fp16 范围的值夹到上限。先写临时文件再替换。"""
    data = np.minimum(ibl.sanitize(rgb), ibl.HALF_MAX).astype(np.float16)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, rgb=data)
    tmp.replace(path)


def save_png(path: Path, pixels: np.ndarray) -> None:
    mode = "RGBA" if pixels.shape[2] == 4 else "RGB"
    Image.fromarray(pixels, mode).save(path, optimize=True)


def convert_hdris(src: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for exr in sorted(src.glob("*.exr")):
        rgb = ibl.read_exr(exr)
        save_npz(dst / f"{exr.stem}.npz", rgb)
        save_png(dst / f"{exr.stem}.png", ibl.make_hdri_thumbnail(rgb, 256, 128, ibl.TRANSFORM_AGX))
        peak = float(rgb.max())
        print(f"  HDRI {exr.stem:10s} {rgb.shape[1]}×{rgb.shape[0]}  最亮 {peak:9.1f}  平均 {float(rgb.mean()):.3f}")
        names.append(exr.stem)
    return names


def convert_matcaps(src: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for exr in sorted(src.glob("*.exr")):
        rgb = ibl.read_exr(exr)
        save_npz(dst / f"{exr.stem}.npz", rgb)
        save_png(dst / f"{exr.stem}.png", ibl.make_matcap_thumbnail(rgb, 128))
        names.append(exr.stem)
    print(f"  Matcap {len(names)} 张：{', '.join(names)}")
    return names


def convert_studio(src: Path, dst: Path) -> list[str]:
    dst.mkdir(parents=True, exist_ok=True)
    names = []
    for sl in sorted(src.glob("*.sl")):
        data = ibl.parse_studio_sl(sl.read_text(encoding="utf-8", errors="replace"), sl.stem)
        data["source"] = f"Blender 4.3 datafiles/studiolights/studio/{sl.name}"
        (dst / f"{sl.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        save_png(dst / f"{sl.stem}.png", ibl.make_studio_thumbnail(data, 128))
        on = sum(1 for light in data["lights"] if light["enabled"])
        print(f"  工作室光 {sl.stem:8s} {on} 盏灯")
        names.append(sl.stem)
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="转换 Blender 自带的 HDRI、Matcap、工作室光")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Blender 的 studiolights 目录")
    parser.add_argument("--out", type=Path, default=RESOURCE_DIR, help="输出的 resources 目录")
    args = parser.parse_args(argv)
    src: Path = args.source
    out: Path = args.out
    if not (src / "world").is_dir():
        print(f"找不到来源目录：{src}")
        return 1
    t0 = time.perf_counter()
    print("转换 HDRI …")
    hdris = convert_hdris(src / "world", out / "hdri")
    shutil.copyfile(src / "world" / "license.txt", out / "hdri" / "LICENSE.txt")
    print("转换 Matcap …")
    matcaps = convert_matcaps(src / "matcap", out / "matcap")
    shutil.copyfile(src / "matcap" / "license.txt", out / "matcap" / "LICENSE.txt")
    print("转换工作室光 …")
    studios = convert_studio(src / "studio", out / "studio")
    print(f"完成：HDRI {len(hdris)}，Matcap {len(matcaps)}，工作室光 {len(studios)}，"
          f"用时 {time.perf_counter() - t0:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
