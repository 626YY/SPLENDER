"""视口上拖动时的临时形状（选框、椭圆选框、刷选圆、套索、虚线……），三维视口和 UV 视图共用。

形状是元组，坐标都是视口像素（scale 乘上去变成控件坐标）：
("box", 起点, 终点)、("ellipse", 起点, 终点)、("cross", 点)、("circle", 点, 半径)、("lasso", 点列)（闭合）、
("line", 起点, 终点)、("polyline", 点列, 颜色)（不闭合）、("points", 点列, 颜色[, 大小])、("multi", [形状…])。
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen


def paint_shape(painter: QPainter, shape, scale: float, width: float, height: float) -> None:
    if shape is None:
        return
    kind = shape[0]
    if kind == "multi":
        for item in shape[1]:
            paint_shape(painter, item, scale, width, height)
        return
    painter.setRenderHint(QPainter.Antialiasing, True)
    white = QPen(QColor(255, 255, 255, 230), 1.0)
    dark = QPen(QColor(0, 0, 0, 200), 1.0)
    dark.setStyle(Qt.DashLine)
    if kind in ("box", "ellipse"):
        (x0, y0), (x1, y1) = shape[1], shape[2]
        rect = QRectF(min(x0, x1) * scale, min(y0, y1) * scale, abs(x1 - x0) * scale, abs(y1 - y0) * scale)
        draw = painter.drawRect if kind == "box" else painter.drawEllipse
        painter.setBrush(QColor(255, 255, 255, 18))
        painter.setPen(white)
        draw(rect)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(dark)
        draw(rect)
    elif kind == "cross":
        x, y = shape[1][0] * scale, shape[1][1] * scale
        for pen in (white, dark):
            painter.setPen(pen)
            painter.drawLine(QPointF(0, y), QPointF(width, y))
            painter.drawLine(QPointF(x, 0), QPointF(x, height))
    elif kind == "circle":
        x, y = shape[1][0] * scale, shape[1][1] * scale
        r = shape[2] * scale
        painter.setBrush(Qt.NoBrush)
        for pen in (white, dark):
            painter.setPen(pen)
            painter.drawEllipse(QPointF(x, y), r, r)
    elif kind == "lasso":
        points = shape[1]
        if len(points) >= 2:
            path = QPainterPath(QPointF(points[0][0] * scale, points[0][1] * scale))
            for px, py in points[1:]:
                path.lineTo(px * scale, py * scale)
            path.closeSubpath()
            painter.setBrush(QColor(255, 255, 255, 18))
            painter.setPen(white)
            painter.drawPath(path)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(dark)
            painter.drawPath(path)
    elif kind == "line":
        (x0, y0), (x1, y1) = shape[1], shape[2]
        for pen in (white, dark):
            painter.setPen(pen)
            painter.drawLine(QPointF(x0 * scale, y0 * scale), QPointF(x1 * scale, y1 * scale))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 235))
        painter.drawEllipse(QPointF(x0 * scale, y0 * scale), 3.0, 3.0)
        painter.setBrush(Qt.NoBrush)
    elif kind == "polyline":
        points = shape[1]
        r, g, b = shape[2]
        if len(points) >= 2:
            path = QPainterPath(QPointF(points[0][0] * scale, points[0][1] * scale))
            for px, py in points[1:]:
                path.lineTo(px * scale, py * scale)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 0, 0, 160), 3.0))
            painter.drawPath(path)
            painter.setPen(QPen(QColor(int(r * 255), int(g * 255), int(b * 255), 240), 1.6))
            painter.drawPath(path)
    elif kind == "points":
        r, g, b = shape[2]
        size = float(shape[3]) if len(shape) > 3 else 5.0
        painter.setPen(QPen(QColor(0, 0, 0, 200), 1.0))
        painter.setBrush(QColor(int(r * 255), int(g * 255), int(b * 255), 240))
        for px, py in shape[1]:
            painter.drawRect(QRectF(px * scale - size / 2, py * scale - size / 2, size, size))
        painter.setBrush(Qt.NoBrush)
