"""从解包的 lucide-static 里挑出程序要用的图标，复制到 splender/resources/icons。

用法：python tools/copy_icons.py <lucide 解包目录>/package
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "splender" / "resources" / "icons"
EXTRA = """
activity app-window aperture arrow-down arrow-left arrow-left-right arrow-right arrow-up axis-3d blend box boxes
brush camera check chevron-down chevron-left chevron-right chevron-up circle circle-alert circle-dashed circle-dot
circle-help columns-2 component contrast copy cpu crosshair droplet ellipsis ellipsis-vertical eraser external-link
eye eye-off file file-down file-plus file-up filter flip-horizontal-2 flip-vertical-2 focus folder folder-open
folder-plus fullscreen globe grid-3x3 grip-horizontal grip-vertical hand hard-drive history house image image-plus
import info keyboard lamp layers layers-2 layout-grid layout-panel-left library lightbulb link list list-tree lock
lock-open maximize maximize-2 memory-stick menu minimize-2 minus monitor mouse-pointer move move-3d paint-bucket
paintbrush palette panel-bottom panel-left panel-right panel-top pen-tool pencil pin pipette plus redo-2 rotate-3d
rotate-ccw rotate-cw rows-2 ruler save scan search settings settings-2 shapes sliders-horizontal sliders-vertical
sparkles square stamp sun swatch-book terminal trash-2 triangle-alert undo-2 upload wand-sparkles workflow x zap
zoom-in zoom-out circle-plus square-plus copy-plus package download refresh-cw play pause gauge timer scissors
spline waves mountain grid-2x2 square-dashed scan-line sun-moon moon type hash percent move-horizontal move-vertical
panels-top-left picture-in-picture-2 combine layers-3 square-stack shield star heart bookmark tag funnel
vector-square box-select mouse-pointer-2 lasso-select locate-fixed scale-3d ruler-dimension-line triangle video
cylinder cone torus hexagon magnet dot target view merge flip-horizontal scan-eye arrow-up-from-line radius
square-arrow-up squircle square-split-horizontal slice expand split lasso wand squares-unite squares-subtract squares-intersect
""".split()


def main() -> int:
    source = Path(sys.argv[1]) / "icons"
    text = (ROOT / "splender" / "ui" / "icons.py").read_text(encoding="utf-8")
    block = text[text.index("ALIASES = {"): text.index("}\n", text.index("ALIASES = {"))]
    wanted = set(re.findall(r'":\s*"([a-z0-9-]+)"', block)) | set(EXTRA)
    DEST.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in sorted(wanted):
        src = source / (name + ".svg")
        if src.is_file():
            shutil.copyfile(src, DEST / (name + ".svg"))
        else:
            missing.append(name)
    shutil.copyfile(Path(sys.argv[1]) / "LICENSE", DEST / "LICENSE-lucide.txt")
    print("copied", len(wanted) - len(missing), "missing", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
