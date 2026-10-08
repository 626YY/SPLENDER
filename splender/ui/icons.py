"""图标：线性 SVG，按主题色着色，按尺寸与屏幕倍率缓存。

图标文件在 resources/icons，来自 Lucide（ISC 许可）。用法：icon("brush")。
"""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from ..paths import resource
from . import theme

_svg_cache: dict[str, str] = {}
_pix_cache: dict[tuple, QPixmap] = {}
_icon_cache: dict[tuple, QIcon] = {}

#: 语义名到文件名的映射。界面代码只用语义名，换图标集时只改这里。
ALIASES = {
    # 编辑器
    "editor.view3d": "box", "editor.image": "image", "editor.layers": "layers", "editor.properties": "sliders-horizontal",
    "editor.assets": "library", "editor.info": "activity", "editor.prefs": "settings", "editor.nodes": "workflow",
    "editor.outliner": "list-tree", "editor.history": "history", "editor.shader": "blend",
    # 工具
    "tool.brush": "brush", "tool.eraser": "eraser", "tool.projection": "stamp", "tool.fill": "paint-bucket",
    "tool.smudge": "hand", "tool.picker": "pipette", "tool.clone": "stamp", "tool.select": "mouse-pointer",
    "tool.gradient": "blend", "tool.heal": "bandage", "tool.dodge": "sun-dim", "tool.burn": "moon",
    "tool.sponge": "droplet-off", "tool.blur": "droplets", "tool.sharpen": "sparkle", "tool.text": "type",
    "tool.shape": "shapes", "tool.select_rect": "square-dashed", "tool.select_ellipse": "circle-dashed",
    "tool.select_lasso": "lasso", "tool.select_poly": "lasso-select", "tool.select_wand": "wand",
    "tool.select_quick": "wand-sparkles", "sel.set": "square", "sel.add": "squares-unite",
    "sel.subtract": "squares-subtract", "sel.intersect": "squares-intersect",
    "tool.rectangle": "rectangle-horizontal", "tool.ellipse": "ellipse", "tool.line": "pen-line",
    # 着色模式
    "shade_solid": "circle", "shade_material": "globe", "shade_channel": "contrast", "shade_rendered": "aperture",
    "render": "aperture", "addon": "component", "sculpt": "mountain", "paint": "paintbrush",
    # 图层
    "layer.paint": "brush", "layer.fill": "paint-bucket", "layer.mask": "circle-dashed", "layer.folder": "folder",
    "visible": "eye", "hidden": "eye-off", "locked": "lock", "unlocked": "lock-open",
    # 通用
    "add": "plus", "remove": "trash-2", "duplicate": "copy", "up": "arrow-up", "down": "arrow-down",
    "close": "x", "check": "check", "search": "search", "undo": "undo-2", "redo": "redo-2", "save": "save",
    "open": "folder-open", "new": "file-plus", "import": "import", "export": "upload", "settings": "settings",
    "more": "ellipsis", "menu": "menu", "chevron.down": "chevron-down", "chevron.right": "chevron-right",
    "chevron.up": "chevron-up", "chevron.left": "chevron-left", "maximize": "maximize-2", "restore": "minimize-2",
    "split.h": "columns-2", "split.v": "rows-2", "popout": "external-link", "grip": "grip-vertical",
    "symmetry": "flip-horizontal-2", "environment": "sun", "camera": "camera", "info": "info",
    "recover": "history", "diagnostics": "copy",
    "warning": "triangle-alert", "error": "circle-alert", "help": "circle-help", "keyboard": "keyboard",
    "memory": "memory-stick", "gpu": "cpu", "disk": "hard-drive", "performance": "zap", "palette": "palette",
    "pin": "pin", "link": "link", "reset": "rotate-ccw", "home": "house", "grid": "grid-3x3", "list": "list",
    "pressure": "pen-tool", "swap": "arrow-left-right", "mesh": "box", "texture_set": "layers-2",
    "channel": "blend", "color": "droplet", "wireframe": "scan", "focus": "focus", "perspective": "move-3d",
    "workspace": "layout-panel-left", "window": "app-window", "file": "file", "folder": "folder",
    "brush_preset": "paintbrush", "hdri": "sun", "matcap": "circle-dot", "material": "shapes",
    "remesh": "boxes", "uv_unwrap": "wand-sparkles", "bake": "scan-line", "generator": "sparkles",
    "smart_material": "layers-3", "stop": "square", "adjust": "sliders-horizontal",
    # 模式
    "mode.object": "box", "mode.edit": "vector-square", "mode.sculpt": "mountain", "mode.paint": "paintbrush",
    # 物体模式的工具
    "tool.select_box": "box-select", "tool.tweak": "mouse-pointer-2", "tool.select_circle": "circle-dashed",
    "tool.select_lasso": "lasso-select", "tool.cursor": "locate-fixed", "tool.move": "move-3d",
    "tool.rotate": "rotate-3d", "tool.scale": "scale-3d", "tool.measure": "ruler-dimension-line",
    "tool.extrude": "square-arrow-up", "tool.inset": "square-dashed", "tool.bevel": "squircle",
    "tool.loopcut": "square-split-horizontal", "tool.knife": "slice", "tool.shrink_fatten": "expand",
    "tool.smooth": "waves", "split": "split",
    # 物体类型
    "object.mesh": "triangle", "object.camera": "video", "object.light": "lightbulb", "object.empty": "axis-3d",
    "collection": "folder", "material_slot": "circle-dot",
    # 基本体
    "prim.plane": "square", "prim.cube": "box", "prim.circle": "circle", "prim.uv_sphere": "globe",
    "prim.ico_sphere": "hexagon", "prim.cylinder": "cylinder", "prim.cone": "cone", "prim.torus": "torus",
    "prim.grid": "grid-3x3",
    "light.point": "lightbulb", "light.sun": "sun", "light.spot": "cone", "light.area": "square",
    "empty.axes": "axis-3d", "empty.arrows": "move-3d", "empty.arrow": "arrow-up", "empty.circle": "circle",
    "empty.cube": "box", "empty.sphere": "globe", "empty.cone": "cone",
    # 变换设置
    "snap": "magnet", "pivot.bbox": "square-dashed", "pivot.cursor": "locate-fixed",
    "pivot.individual": "dot", "pivot.median": "circle-dot", "pivot.active": "target",
    "orientation.global": "globe", "orientation.local": "box", "orientation.view": "view",
    "orientation.cursor": "locate-fixed", "orientation.normal": "arrow-up-from-line",
    "select.vert": "dot", "select.edge": "minus", "select.face": "square",
    "proportional": "radius",
    # 显示
    "overlays": "blend", "xray": "scan-eye", "gizmo": "move-3d", "shade.smooth": "circle", "shade.flat": "hexagon",
    "mirror": "flip-horizontal", "join": "merge", "origin": "dot", "apply": "check", "pencil": "pencil",
    "filter": "funnel",
}


def _source(name: str) -> str | None:
    name = ALIASES.get(name, name)
    if name in _svg_cache:
        return _svg_cache[name]
    path = resource("icons", name + ".svg")
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    _svg_cache[name] = text
    return text


def pixmap(name: str, color: str | None = None, size: int | None = None, dpr: float = 1.0) -> QPixmap:
    """按逻辑尺寸取一张着色后的图标位图。"""
    logical = size if size is not None else theme.size("icon")
    hexcolor = color or theme.color("text")
    key = (name, hexcolor, logical, round(dpr, 2))
    cached = _pix_cache.get(key)
    if cached is not None:
        return cached
    text = _source(name)
    side = max(1, int(round(logical * dpr)))
    image = QImage(side, side, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    if text is not None:
        text = text.replace("currentColor", hexcolor).replace('stroke-width="2"', 'stroke-width="1.75"')
        renderer = QSvgRenderer(QByteArray(text.encode("utf-8")))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        renderer.render(painter, QRectF(0, 0, side, side))
        painter.end()
    pm = QPixmap.fromImage(image)
    pm.setDevicePixelRatio(dpr)
    _pix_cache[key] = pm
    return pm


def icon(name: str, color: str | None = None, size: int | None = None) -> QIcon:
    """图标：自带普通、禁用、选中三种状态的颜色，并准备好 1x 和 2x。"""
    logical = size if size is not None else theme.size("icon")
    base = color or theme.color("text")
    key = (name, base, logical)
    cached = _icon_cache.get(key)
    if cached is not None:
        return cached
    result = QIcon()
    for dpr in (1.0, 1.5, 2.0):
        result.addPixmap(pixmap(name, base, logical, dpr), QIcon.Normal, QIcon.Off)
        result.addPixmap(pixmap(name, theme.color("text.disabled"), logical, dpr), QIcon.Disabled, QIcon.Off)
        result.addPixmap(pixmap(name, theme.color("text.on_accent"), logical, dpr), QIcon.Normal, QIcon.On)
    _icon_cache[key] = result
    return result


def exists(name: str) -> bool:
    return _source(name) is not None


def icon_size(kind: str = "icon") -> QSize:
    side = theme.size(kind)
    return QSize(side, side)


def clear_cache() -> None:
    _pix_cache.clear()
    _icon_cache.clear()


theme.changed.connect(clear_cache)
