"""主题：一张色板加一张尺寸表，生成整个界面的样式。

取色用 color("text")，取尺寸用 px(24)（会乘界面缩放）。自绘控件和样式表都从这里取值。
"""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

from ..core.signals import Signal

DARK = {
    # 背景，从深到浅
    "bg.window": "#18181a",          # 最底层，区域之间的缝
    "bg.area": "#27272a",            # 编辑器主体
    "bg.header": "#2e2e31",          # 区域标题栏、顶栏
    "bg.panel": "#2c2c2f",           # 面板内容
    "bg.panel_header": "#343437",    # 面板标题行
    "bg.widget": "#3b3b40",          # 按钮、数值框
    "bg.widget_hover": "#47474d",
    "bg.widget_pressed": "#2a2a2d",
    "bg.field": "#1f1f22",           # 文本输入、列表底
    "bg.menu": "#232326",
    "bg.tooltip": "#141416",
    "bg.viewport": "#353538",
    "bg.row_alt": "#2a2a2d",
    "bg.row_hover": "#35353a",
    # 线
    "line": "#151517",
    "line.soft": "#434349",
    "line.focus": "#6aa1ff",
    # 文字
    "text": "#dededf",
    "text.dim": "#a3a3a9",
    "text.faint": "#77777e",
    "text.disabled": "#5c5c63",
    "text.on_accent": "#ffffff",
    # 强调色
    "accent": "#4b8bf5",
    "accent.hover": "#629bf8",
    "accent.pressed": "#3a74d4",
    "accent.soft": "#2c446b",        # 选中行底色
    "accent.softer": "#283548",
    # 状态
    "ok": "#58b368",
    "warn": "#e0a43a",
    "error": "#e5544b",
    # 坐标轴
    "axis.x": "#e5484d",
    "axis.y": "#5bb450",
    "axis.z": "#4a7dff",
}

SIZES = {
    "font": 12,            # 正文字号
    "font.small": 11,
    "font.title": 13,
    "header": 28,          # 区域标题栏高度
    "topbar": 30,
    "statusbar": 24,
    "widget": 24,          # 控件高度
    "row": 24,             # 列表行高
    "radius": 4,
    "radius.small": 3,
    "gap": 4,              # 控件间距
    "pad": 8,              # 面板内边距
    "icon": 16,
    "icon.tool": 20,
    "toolbar": 40,         # 工具栏宽度
    "tool_button": 32,
    "splitter": 3,         # 区域分隔缝
    "sidebar": 280,
}

_colors = dict(DARK)
_scale = 1.0
changed = Signal()

FONT_FAMILIES = ["Microsoft YaHei UI", "Segoe UI", "PingFang SC", "Noto Sans CJK SC", "sans-serif"]


def color(name: str) -> str:
    return _colors[name]


def qcolor(name: str, alpha: float | None = None) -> QColor:
    c = QColor(_colors[name])
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


def scale() -> float:
    return _scale


def px(value: float) -> int:
    """逻辑像素乘界面缩放。"""
    return max(1, int(round(value * _scale)))


def size(name: str) -> int:
    return px(SIZES[name])


def font(kind: str = "font", bold: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPixelSize(size(kind))
    f.setHintingPreference(QFont.PreferFullHinting)
    if bold:
        f.setWeight(QFont.DemiBold)
    return f


def set_colors(overrides: dict) -> None:
    _colors.update(overrides)


def apply(app: QApplication, ui_scale: float = 1.0) -> None:
    """把主题应用到整个程序。缩放改变后再调用一次即可。"""
    global _scale
    _scale = max(0.5, min(3.0, float(ui_scale)))
    app.setStyle("Fusion")
    app.setFont(font())
    pal = QPalette()
    pal.setColor(QPalette.Window, qcolor("bg.area"))
    pal.setColor(QPalette.WindowText, qcolor("text"))
    pal.setColor(QPalette.Base, qcolor("bg.field"))
    pal.setColor(QPalette.AlternateBase, qcolor("bg.row_alt"))
    pal.setColor(QPalette.Text, qcolor("text"))
    pal.setColor(QPalette.Button, qcolor("bg.widget"))
    pal.setColor(QPalette.ButtonText, qcolor("text"))
    pal.setColor(QPalette.Highlight, qcolor("accent"))
    pal.setColor(QPalette.HighlightedText, qcolor("text.on_accent"))
    pal.setColor(QPalette.ToolTipBase, qcolor("bg.tooltip"))
    pal.setColor(QPalette.ToolTipText, qcolor("text"))
    pal.setColor(QPalette.PlaceholderText, qcolor("text.faint"))
    pal.setColor(QPalette.Disabled, QPalette.Text, qcolor("text.disabled"))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, qcolor("text.disabled"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, qcolor("text.disabled"))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet())
    changed.emit()


def stylesheet() -> str:
    c = _colors
    s = {k: size(k) for k in SIZES}
    r = s["radius"]
    return f"""
* {{ outline: none; }}
QWidget {{ color: {c['text']}; font-size: {s['font']}px; }}
QMainWindow, QDialog {{ background: {c['bg.window']}; }}
QToolTip {{ background: {c['bg.tooltip']}; color: {c['text']}; border: 1px solid {c['line.soft']};
    padding: {px(4)}px {px(7)}px; border-radius: {r}px; }}

QMenu {{ background: {c['bg.menu']}; border: 1px solid {c['line.soft']}; padding: {px(4)}px; border-radius: {px(6)}px; }}
QMenu::item {{ padding: {px(5)}px {px(26)}px {px(5)}px {px(10)}px; border-radius: {r}px; margin: 0 {px(1)}px; }}
QMenu::item:selected {{ background: {c['accent']}; color: {c['text.on_accent']}; }}
QMenu::item:disabled {{ color: {c['text.disabled']}; background: transparent; }}
QMenu::separator {{ height: 1px; background: {c['line.soft']}; margin: {px(4)}px {px(6)}px; }}
QMenu::icon {{ padding-left: {px(8)}px; }}
QMenu::right-arrow {{ width: {px(10)}px; height: {px(10)}px; }}

QScrollBar:vertical {{ background: transparent; width: {px(10)}px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {c['line.soft']}; min-height: {px(28)}px; border-radius: {px(3)}px; margin: {px(2)}px; }}
QScrollBar::handle:vertical:hover {{ background: {c['text.faint']}; }}
QScrollBar:horizontal {{ background: transparent; height: {px(10)}px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {c['line.soft']}; min-width: {px(28)}px; border-radius: {px(3)}px; margin: {px(2)}px; }}
QScrollBar::handle:horizontal:hover {{ background: {c['text.faint']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}

QLineEdit {{ background: {c['bg.field']}; border: 1px solid {c['line']}; border-radius: {r}px;
    padding: 0 {px(6)}px; min-height: {s['widget'] - 2}px; selection-background-color: {c['accent']}; }}
QLineEdit:focus {{ border-color: {c['line.focus']}; }}
QLineEdit:disabled {{ color: {c['text.disabled']}; }}
QPlainTextEdit, QTextEdit {{ background: {c['bg.field']}; border: 1px solid {c['line']}; border-radius: {r}px;
    selection-background-color: {c['accent']}; }}

QPushButton {{ background: {c['bg.widget']}; border: none; border-radius: {r}px; padding: 0 {px(10)}px;
    min-height: {s['widget']}px; }}
QPushButton:hover {{ background: {c['bg.widget_hover']}; }}
QPushButton:pressed {{ background: {c['bg.widget_pressed']}; }}
QPushButton:checked {{ background: {c['accent']}; color: {c['text.on_accent']}; }}
QPushButton:disabled {{ color: {c['text.disabled']}; background: {c['bg.panel_header']}; }}
QPushButton[role="primary"] {{ background: {c['accent']}; color: {c['text.on_accent']}; }}
QPushButton[role="primary"]:hover {{ background: {c['accent.hover']}; }}
QPushButton[role="primary"]:pressed {{ background: {c['accent.pressed']}; }}

QToolButton {{ background: transparent; border: none; border-radius: {r}px; padding: {px(3)}px; }}
QToolButton:hover {{ background: {c['bg.widget_hover']}; }}
QToolButton:pressed {{ background: {c['bg.widget_pressed']}; }}
QToolButton:checked {{ background: {c['accent']}; color: {c['text.on_accent']}; }}
QToolButton:disabled {{ color: {c['text.disabled']}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}

QComboBox {{ background: {c['bg.widget']}; border: none; border-radius: {r}px; padding: 0 {px(8)}px;
    min-height: {s['widget']}px; }}
QComboBox:hover {{ background: {c['bg.widget_hover']}; }}
QComboBox::drop-down {{ border: none; width: {px(18)}px; }}
QComboBox QAbstractItemView {{ background: {c['bg.menu']}; border: 1px solid {c['line.soft']};
    selection-background-color: {c['accent']}; outline: none; padding: {px(3)}px; }}

QCheckBox {{ spacing: {px(6)}px; }}
QCheckBox::indicator {{ width: {px(14)}px; height: {px(14)}px; border-radius: {px(3)}px;
    background: {c['bg.widget']}; border: 1px solid {c['line']}; }}
QCheckBox::indicator:hover {{ background: {c['bg.widget_hover']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}

QListView, QTreeView, QTableView {{ background: {c['bg.field']}; border: 1px solid {c['line']};
    border-radius: {r}px; alternate-background-color: {c['bg.row_alt']}; }}
QListView::item, QTreeView::item {{ min-height: {s['row']}px; padding: 0 {px(4)}px; }}
QListView::item:hover, QTreeView::item:hover {{ background: {c['bg.row_hover']}; }}
QListView::item:selected, QTreeView::item:selected {{ background: {c['accent.soft']}; color: {c['text']}; }}
QHeaderView::section {{ background: {c['bg.panel_header']}; border: none; padding: {px(4)}px {px(6)}px; color: {c['text.dim']}; }}

QTabBar::tab {{ background: transparent; color: {c['text.dim']}; padding: {px(5)}px {px(12)}px;
    border-top-left-radius: {r}px; border-top-right-radius: {r}px; }}
QTabBar::tab:hover {{ color: {c['text']}; background: {c['bg.widget']}; }}
QTabBar::tab:selected {{ color: {c['text']}; background: {c['bg.area']}; }}
QTabWidget::pane {{ border: none; }}

QStatusBar {{ background: {c['bg.header']}; color: {c['text.dim']}; border-top: 1px solid {c['line']}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: {c['bg.window']}; }}
QLabel[role="dim"] {{ color: {c['text.dim']}; }}
QLabel[role="faint"] {{ color: {c['text.faint']}; font-size: {s['font.small']}px; }}
QLabel[role="title"] {{ font-size: {s['font.title']}px; font-weight: 600; }}
QProgressBar {{ background: {c['bg.field']}; border: none; border-radius: {px(2)}px; height: {px(4)}px; text-align: center; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: {px(2)}px; }}
"""


def load_fonts() -> None:
    """预留：需要自带字体时在这里注册。"""
    QFontDatabase.families()
