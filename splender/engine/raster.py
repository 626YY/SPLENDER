"""用 Qt 把文字、形状画成一张 RGBA 图（不预乘、行序和贴图一样：第一行是 v 最小的那行），交给「贴图」写回方式盖到图层上。"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF

MAX_SIDE = 8192


def to_array(image: QImage) -> np.ndarray:
    """QImage → (h, w, 4) uint8，不预乘，上下翻过来（贴图的第一行在下面）。"""
    image = image.convertToFormat(QImage.Format_RGBA8888)
    w, h = image.width(), image.height()
    data = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes()).reshape(h, image.bytesPerLine())
    return np.ascontiguousarray(data[:, :w * 4].reshape(h, w, 4)[::-1])


def default_family() -> str:
    """没选字体时用的字体：先找能写中文的常见字体，再退到界面字体（空名字的字体放不大）。"""
    from PySide6.QtGui import QFontDatabase, QGuiApplication

    families = set(QFontDatabase.families())
    for name in ("Microsoft YaHei", "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC",
                 "SimHei", "Arial"):
        if name in families:
            return name
    app = QGuiApplication.instance()
    return app.font().family() if app is not None else "Arial"


def _canvas(width: int, height: int) -> QImage:
    image = QImage(max(1, int(width)), max(1, int(height)), QImage.Format_RGBA8888_Premultiplied)
    image.fill(Qt.transparent)
    return image


def _text_layout(text: str, family: str, size_px: float, bold: bool, italic: bool, rotation: float,
                 line_spacing: float) -> tuple:
    """排版：(字体, 字体度量, 各行, 各行宽, 不转时的宽高, 转过以后的外框宽高)。"""
    lines = (text or "").split("\n")
    font = QFont(family or default_family())
    font.setPixelSize(max(1, int(round(size_px))))
    font.setBold(bool(bold))
    font.setItalic(bool(italic))
    metrics = QFontMetricsF(font)
    step = metrics.height() * float(line_spacing)
    widths = [metrics.horizontalAdvance(line) for line in lines]
    width = max(widths + [1.0]) + metrics.averageCharWidth()
    height = step * (len(lines) - 1) + metrics.height()
    angle = math.radians(rotation)
    cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
    box = (width * cos + height * sin + 4.0, width * sin + height * cos + 4.0)
    return font, metrics, lines, widths, step, (width, height), box


def text_extent(text: str, family: str, size_px: float, *, bold: bool = False, italic: bool = False,
                rotation: float = 0.0, line_spacing: float = 1.2) -> tuple:
    """这段字画出来的图有多大 (宽, 高)，不受 MAX_SIDE 限制。"""
    return _text_layout(text, family, size_px, bold, italic, rotation, line_spacing)[-1]


def text_image(text: str, family: str, size_px: float, *, bold: bool = False, italic: bool = False,
               rotation: float = 0.0, align: str = "LEFT", line_spacing: float = 1.2) -> np.ndarray:
    """多行文字画成白色带透明的图（颜色在盖上去时给），返回 (h, w, 4)。size_px 是字高（像素）。"""
    font, metrics, lines, widths, step, (width, height), box = _text_layout(text, family, size_px, bold, italic,
                                                                            rotation, line_spacing)
    out_w = min(MAX_SIDE, int(math.ceil(box[0])))
    out_h = min(MAX_SIDE, int(math.ceil(box[1])))
    image = _canvas(out_w, out_h)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255, 255))
    painter.translate(out_w / 2.0, out_h / 2.0)
    painter.rotate(-rotation)
    painter.translate(-width / 2.0, -height / 2.0)
    for index, (line, line_width) in enumerate(zip(lines, widths)):
        x = {"CENTER": (width - line_width) / 2.0, "RIGHT": width - line_width}.get(align, 0.0)
        painter.drawText(QPointF(x, index * step + metrics.ascent()), line)
    painter.end()
    return to_array(image)


def shape_image(kind: str, width: float, height: float, *, fill: bool = True, stroke: float = 0.0,
                radius: float = 0.0, sides: int = 6) -> np.ndarray:
    """形状画成白色带透明的图：RECT 矩形、ROUNDED 圆角矩形、ELLIPSE 椭圆、POLYGON 多边形、LINE 直线（对角）。"""
    pad = max(2.0, stroke)
    w = min(MAX_SIDE, int(math.ceil(width + 2 * pad)))
    h = min(MAX_SIDE, int(math.ceil(height + 2 * pad)))
    image = _canvas(w, h)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    white = QColor(255, 255, 255, 255)
    rect = QRectF(pad, pad, max(1.0, width), max(1.0, height))
    path = QPainterPath()
    if kind == "ELLIPSE":
        path.addEllipse(rect)
    elif kind == "ROUNDED":
        r = min(radius, min(rect.width(), rect.height()) / 2.0)
        path.addRoundedRect(rect, r, r)
    elif kind == "POLYGON":
        n = max(3, int(sides))
        cx, cy = rect.center().x(), rect.center().y()
        poly = QPolygonF([QPointF(cx + rect.width() / 2.0 * math.cos(-math.pi / 2 + 2 * math.pi * k / n),
                                  cy + rect.height() / 2.0 * math.sin(-math.pi / 2 + 2 * math.pi * k / n))
                          for k in range(n)])
        path.addPolygon(poly)
        path.closeSubpath()
    elif kind == "LINE":
        path.moveTo(rect.topLeft())
        path.lineTo(rect.bottomRight())
        fill = False
        stroke = max(stroke, 1.0)
    else:
        path.addRect(rect)
    if fill:
        painter.fillPath(path, white)
    if stroke > 0:
        pen = QPen(white, float(stroke))
        pen.setJoinStyle(Qt.MiterJoin if kind in ("RECT", "POLYGON") else Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        painter.strokePath(path, pen)
    painter.end()
    return to_array(image)


def tint(alpha_image: np.ndarray, color) -> np.ndarray:
    """白色带透明的图换成某种颜色（透明度不变）。"""
    out = np.array(alpha_image, copy=True)
    out[..., :3] = np.clip(np.asarray(color[:3], np.float64) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


def mask_shape(kind: str, points, scale: float, *, antialias: bool = True, feather: float = 0.0,
               max_side: int = MAX_SIDE, max_pixels: int = 16 * 1024 * 1024) -> tuple:
    """选区的形状画成一张 8 位遮罩（255 是选上）。kind：RECT 矩形（points 是两个对角）、ELLIPSE 椭圆（两个对角是外框）、
    POLYGON 多边形（points 是顶点）。points 和 feather 的单位一样（屏幕像素或纹素），scale 是图上几个像素一个单位，
    太大时自动画粗一点。返回 (遮罩（第一行是 y 最小的那行）, (x0, y0, x1, y1) 遮罩盖住的范围，按原来的单位)。"""
    import cv2

    pts = np.asarray(points, np.float64).reshape(-1, 2)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = max(0.0, float(feather)) * 2.0 + 2.0
    lo, hi = lo - pad, hi + pad
    span = np.maximum(hi - lo, 1e-6)
    scale = float(scale)
    scale = min(scale, max_side / float(span.max()), math.sqrt(max_pixels / float(span[0] * span[1])))
    w = max(1, int(math.ceil(span[0] * scale)))
    h = max(1, int(math.ceil(span[1] * scale)))
    image = QImage(w, h, QImage.Format_Grayscale8)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, bool(antialias))
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(255, 255, 255))
    local = (pts - lo) * scale
    if kind == "ELLIPSE":
        (x0, y0), (x1, y1) = local.min(axis=0), local.max(axis=0)
        painter.drawEllipse(QRectF(x0, y0, x1 - x0, y1 - y0))
    elif kind == "RECT":
        (x0, y0), (x1, y1) = local.min(axis=0), local.max(axis=0)
        painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
    else:
        painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in local]))
    painter.end()
    bits = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes()).reshape(h, image.bytesPerLine())
    mask = bits[:, :w].copy()                # 一定要拷出来：QImage 一释放，指着它内存的数组就成了野指针
    if feather > 0:
        sigma = max(0.3, float(feather) * scale * 0.5)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return mask, (float(lo[0]), float(lo[1]), float(lo[0] + w / scale), float(lo[1] + h / scale))
