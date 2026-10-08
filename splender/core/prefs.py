"""偏好设置。所有预算、默认值、行为开关都在这里，偏好编辑器自动生成界面。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .props import (BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, PropertyGroup,
                    StringProperty)

log = logging.getLogger("splender.prefs")


class InterfacePrefs(PropertyGroup):
    ui_scale = FloatProperty("界面缩放", default=1.0, min=0.75, max=2.0, step=0.05, precision=2,
                             description="整个界面的大小倍率")
    show_tooltips = BoolProperty("显示提示", default=True, description="鼠标停留时显示说明")
    show_splash = BoolProperty("显示启动画面", default=True)
    status_hints = BoolProperty("状态栏操作提示", default=True, description="在底部显示当前工具的按键说明")
    load_ui = BoolProperty("打开工程时载入界面布局", default=False,
                           description="打开时使用工程里保存的窗口布局")
    error_dialog = BoolProperty("出错时弹出提示", default=True,
                                description="程序内部出错时弹一个窗口说明，能复制诊断信息；关掉后只在状态栏和日志里提示")


class NavigationPrefs(PropertyGroup):
    orbit_method = EnumProperty("旋转方式", items=[
        ("TURNTABLE", "转盘", "绕竖直轴旋转，地平线保持水平"),
        ("TRACKBALL", "轨迹球", "自由旋转")], default="TURNTABLE")
    orbit_sensitivity = FloatProperty("旋转灵敏度", default=0.4, min=0.05, max=2.0, unit="°/px", precision=2)
    orbit_around_cursor = BoolProperty("绕鼠标下的表面旋转", default=False,
                                       description="以鼠标指着的表面位置作为旋转中心")
    zoom_to_mouse = BoolProperty("向鼠标位置缩放", default=True)
    invert_zoom = BoolProperty("反转缩放方向", default=False)
    zoom_speed = FloatProperty("滚轮缩放步长", default=1.15, min=1.02, max=2.0, precision=2)
    auto_perspective = BoolProperty("自动透视", default=True,
                                    description="切到正视图时用正交，旋转后回到透视")
    smooth_view = IntProperty("视图过渡时间", default=180, min=0, max=1000, unit="ms")
    emulate_3_button = BoolProperty("Alt+左键模拟中键", default=False, description="没有中键时使用")
    drag_threshold = IntProperty("拖动阈值（鼠标）", default=3, min=1, max=64, unit="px",
                                 description="按下后移动超过这个距离算拖动（比如框选），没超过就松开算单击")
    drag_threshold_tablet = IntProperty("拖动阈值（数位笔）", default=10, min=1, max=64, unit="px",
                                        description="用数位笔时的拖动阈值，笔尖容易抖，所以大一些")
    lens = FloatProperty("默认焦距", default=50.0, min=10.0, max=250.0, unit="mm", precision=0)
    walk_speed = FloatProperty("漫游速度", default=2.5, min=0.01, max=1000.0, unit="m/s", precision=2,
                               description="漫游（Shift+`）时走多快")
    walk_speed_factor = FloatProperty("漫游加速倍数", default=5.0, min=1.0, max=100.0, precision=1,
                                      description="漫游时按住 Shift 快几倍、按住 Alt 慢几倍")
    walk_mouse_sensitivity = FloatProperty("漫游转头灵敏度", default=1.0, min=0.05, max=10.0, precision=2)


class MemoryPrefs(PropertyGroup):
    vram_pool_mb = IntProperty("图层显存预算", default=0, min=0, max=65536, unit="MB",
                               description="图层页面在显存里最多占多少。0 表示按显卡自动决定")
    display_cache_mb = IntProperty("显示缓存预算", default=0, min=0, max=16384, unit="MB",
                                   description="屏幕上看得见的合成贴图最多占多少显存。0 表示自动")
    ram_cache_mb = IntProperty("内存缓存预算", default=0, min=0, max=262144, unit="MB",
                               description="换出显存的页面在内存里最多占多少。0 表示按内存自动决定")
    undo_mb = IntProperty("撤销预算", default=4096, min=64, max=65536, unit="MB",
                          description="撤销历史最多占多少空间（按压缩后的估算），超过后最早的步骤被丢弃")
    undo_steps = IntProperty("撤销步数上限", default=64, min=1, max=1000)
    free_watermark = FloatProperty("显存空闲水位", default=0.08, min=0.02, max=0.5, precision=2, subtype="FACTOR",
                                   description="显存页池保留的空闲比例，低于它时后台开始换出")
    evict_budget_ms = FloatProperty("每帧换入时间上限", default=4.0, min=0.5, max=30.0, unit="ms", precision=1,
                                    description="每帧最多花多少时间把需要的页面放回显存")
    upload_budget_painting_ms = FloatProperty("绘制时每帧换入时间上限", default=1.5, min=0.2, max=30.0, unit="ms",
                                              precision=1, description="正在落笔时，每帧最多花多少时间把页面放回显存。"
                                                                       "越小笔迹越跟手，周围贴图清晰得越慢")
    backup_mb_painting = FloatProperty("绘制时每帧后台备份量", default=4.0, min=0.0, max=256.0, unit="MB", precision=0,
                                       description="画画的时候，每帧最多把多少显存里的新页面备份到内存。"
                                                   "备份过的页面需要腾位置时可以立刻放掉")
    backup_mb_idle = FloatProperty("空闲时每帧后台备份量", default=128.0, min=0.0, max=1024.0, unit="MB", precision=0,
                                   description="没在画画、画面也没在动的时候，每帧最多备份多少")
    compress_ram = BoolProperty("压缩内存里的冷页面", default=True)
    gc_idle_seconds = FloatProperty("停手多久后整理内存", default=1.5, min=0.3, max=30.0, unit="s", precision=1,
                                    description="整理内存时程序会停顿一下，所以只在停止操作这么久、画面也静止之后进行")
    scratch_dir = StringProperty("暂存目录", default="", subtype="DIR_PATH",
                                 description="内存放不下时页面写到这里。留空使用默认位置")


class PaintPrefs(PropertyGroup):
    default_resolution = EnumProperty("新工程贴图分辨率", items=[
        ("1024", "1K", ""), ("2048", "2K", ""), ("4096", "4K", ""), ("8192", "8K", ""), ("16384", "16K", "")],
        default="16384")
    object_resolution = EnumProperty("新物体贴图分辨率", items=[
        ("512", "512", ""), ("1024", "1K", ""), ("2048", "2K", ""), ("4096", "4K", ""), ("8192", "8K", ""),
        ("16384", "16K", "")], default="4096", description="Shift+A 新加的模型自带的那套贴图有多大")
    max_brush_px = IntProperty("笔刷直径上限", default=8192, min=64, max=16384, unit="px",
                               description="按贴图纹素计的最大笔刷直径")
    prefetch_radius = FloatProperty("悬停预取范围", default=2.5, min=1.0, max=8.0, precision=1,
                                    description="鼠标悬停时提前准备笔刷周围多大范围的页面，单位是笔刷半径")
    tablet_api = EnumProperty("数位板接口", items=[
        ("AUTO", "自动", "有 Windows Ink 的笔事件就用它；驱动里关了 Windows Ink 时从 WinTab 取压感"),
        ("WINTAB", "WinTab", "压感、倾斜一律从 WinTab 取（老驱动、关了 Windows Ink 的时候用）"),
        ("WININK", "Windows Ink", "只用系统的笔事件，不开 WinTab")], default="AUTO",
        description="压感从哪里来。画画没有压感时换一个试试")
    pressure_curve = FloatProperty("压感曲线", default=1.0, min=0.2, max=5.0, precision=2,
                                   description="小于 1 更敏感，大于 1 更迟钝")
    stroke_dilate = BoolProperty("绘制时向 UV 岛外扩边", default=True,
                                 description="避免贴图在 UV 接缝处露出未绘制的边")
    merge_budget_ms = FloatProperty("抬笔合并每帧用时", default=4.0, min=0.5, max=50.0, unit="ms", precision=1,
                                    description="抬笔后把这一笔并进图层，每帧最多花多少时间。越小画面越稳，合并越慢完成")
    merge_budget_painting_ms = FloatProperty("边画边合并每帧用时", default=2.5, min=0.5, max=50.0, unit="ms",
                                             precision=1, description="上一笔还没并完就落下一笔时，合并每帧最多花多少时间")
    merge_pages_per_frame = IntProperty("抬笔合并每帧页数", default=1024, min=16, max=16384,
                                        description="抬笔合并每帧最多处理多少页（每页 256×256）")
    dilate_texels = IntProperty("扩边宽度", default=32, min=2, max=256, unit="px",
                                description="向 UV 岛外多画多宽，按 16K 贴图的纹素计，其他分辨率按比例缩放。"
                                            "越宽，缩小看时接缝越干净。导入模型时生效")


class SculptPrefs(PropertyGroup):
    max_triangles = IntProperty("面数上限", default=40_000_000, min=100_000, max=500_000_000,
                                description="细分等操作不会让模型超过这么多三角形（显存不够时调小）")
    undo_backup = FloatProperty("单笔可撤销范围", default=1.0, min=0.05, max=1.0, subtype="FACTOR", precision=2,
                                description="一笔最多能改动模型的多大比例还能撤销。调小省显存")
    auto_unwrap = BoolProperty("回到绘制时自动展开 UV", default=True,
                               description="重构过的模型没有 UV，回到绘制模式时自动展开，马上就能画")


class ViewportPrefs(PropertyGroup):
    msaa = EnumProperty("抗锯齿", items=[("1", "关闭", ""), ("2", "2x", ""), ("4", "4x", ""), ("8", "8x", "")],
                        default="4")
    anisotropy = EnumProperty("各向异性过滤", items=[("1", "关闭", ""), ("4", "4x", ""), ("8", "8x", ""),
                                                  ("16", "16x", "")], default="16")
    vsync = BoolProperty("垂直同步", default=True)
    feedback_divisor = IntProperty("页面需求采样精度", default=8, min=2, max=32,
                                   description="数值越小，贴图清晰度跟随镜头越及时，开销越高")
    show_stats = BoolProperty("视口显示统计", default=False)
    bake_budget_ms = FloatProperty("烘焙时每帧用时", default=12.0, min=1.0, max=200.0, unit="ms", precision=0,
                                   description="烘焙模型贴图时每帧给显卡多少时间。越大烘得越快，视口越不跟手")
    render_budget_ms = FloatProperty("出图时每帧用时", default=30.0, min=2.0, max=500.0, unit="ms", precision=0,
                                     description="出图（F12）时每帧最多花多少时间渲染。越大出图越快，界面越不跟手")
    cursor_ring = BoolProperty("笔刷光标贴合表面", default=True)
    selection_ants = BoolProperty("显示选区边线", default=True, description="有选区时在模型和 UV 视图上画出选区的边")
    selection_animate = BoolProperty("选区边线流动", default=True,
                                     description="选区边线的黑白虚线一直往前走（和 Photoshop 一样）")
    selection_speed = FloatProperty("流动速度", default=1.5, min=0.1, max=10.0, precision=1,
                                    description="虚线每秒往前走几段（一黑一白算一段）")
    selection_fps = IntProperty("流动帧率", default=12, min=2, max=60, description="虚线流动时每秒重画几次视口")
    selection_width = FloatProperty("边线宽度", default=1.5, min=0.5, max=6.0, precision=1, unit="px")
    selection_dash = FloatProperty("虚线段长", default=4.0, min=1.0, max=32.0, precision=0, unit="px",
                                   description="黑、白每一小段多长")


class FilePrefs(PropertyGroup):
    recent = StringProperty("最近工程", default="[]", hidden=True)
    last_dir = StringProperty("上次目录", default="", hidden=True)
    recent_count = IntProperty("最近工程数量", default=12, min=0, max=50)
    save_layout_in_project = BoolProperty("把界面布局存进工程", default=True)
    export_dir = StringProperty("默认导出目录", default="", subtype="DIR_PATH")
    autosave = BoolProperty("自动保存", default=True,
                            description="每隔一段时间把没保存的改动另存一份；程序意外退出后，下次启动时可以恢复")
    autosave_minutes = FloatProperty("自动保存间隔", default=2.0, min=0.25, max=120.0, step=0.25, precision=2,
                                     unit="min", description="距上次自动保存（或手动保存）多久后再存一次")
    autosave_idle = FloatProperty("等停手", default=1.0, min=0.0, max=30.0, unit="s", precision=1,
                                  description="到时间后等你停止操作这么久再开始存，不打断正在画的笔划")
    autosave_max_wait = FloatProperty("最多推迟", default=60.0, min=0.0, max=1800.0, unit="s", precision=0,
                                      description="一直没停手时最多推迟这么久就照常保存（正在画的那一笔仍会等它画完）")
    autosave_budget_ms = FloatProperty("每帧最多花", default=1.5, min=0.2, max=20.0, unit="ms", precision=1,
                                       description="自动保存在每一帧里最多占用的时间。越小越不影响操作，存完需要的时间越长")
    autosave_threads = IntProperty("压缩线程数", default=2, min=1, max=32,
                                   description="自动保存压缩数据用几个线程。多了存得快，但会和绘制抢处理器")
    autosave_dir = StringProperty("自动保存目录", default="", subtype="DIR_PATH",
                                  description="恢复文件放在哪；空着就放在用户设置目录的 autosave 文件夹里")
    autosave_keep = IntProperty("最多保留几份", default=5, min=1, max=100,
                                description="意外退出留下、还没处理的自动保存最多留几份，多了删掉最早的")
    restore_dir = StringProperty("恢复到", default="", subtype="DIR_PATH",
                                 description="恢复出来的工程放在哪；空着就放在原工程旁边（没存过的工程放在文档文件夹）")


class Preferences(PropertyGroup):
    interface = PointerProperty("界面", type=InterfacePrefs)
    navigation = PointerProperty("导航", type=NavigationPrefs)
    memory = PointerProperty("内存与显存", type=MemoryPrefs)
    paint = PointerProperty("绘制", type=PaintPrefs)
    sculpt = PointerProperty("雕刻", type=SculptPrefs)
    viewport = PointerProperty("视口", type=ViewportPrefs)
    files = PointerProperty("文件", type=FilePrefs)
    theme = StringProperty("主题", default="dark", hidden=True)
    keymap_overrides = StringProperty("键位改动", default="{}", hidden=True)
    workspaces = StringProperty("工作区布局", default="", hidden=True)
    addons = StringProperty("插件", default="", hidden=True)        # 启用了哪些插件、插件自己的设置（JSON）

    # ---- 存取 ----
    def load(self, path: Path) -> None:
        try:
            if path.is_file():
                self.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            log.exception("偏好设置读取失败，使用默认值")

    def save(self, path: Path) -> None:
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(path)
        except Exception:  # noqa: BLE001
            log.exception("偏好设置保存失败")

    # ---- 最近工程 ----
    def recent_files(self) -> list[str]:
        try:
            return [str(p) for p in json.loads(self.files.recent)]
        except Exception:  # noqa: BLE001
            return []

    def add_recent(self, path: str) -> None:
        items = [p for p in self.recent_files() if p != path]
        items.insert(0, path)
        self.files.recent = json.dumps(items[: max(0, self.files.recent_count)], ensure_ascii=False)

    # ---- 自动预算 ----
    def resolved_budgets(self, vram_total_mb: int, ram_total_mb: int) -> dict:
        """把「0=自动」换算成实际预算（MB）。"""
        mem = self.memory
        if mem.vram_pool_mb > 0:
            pool = mem.vram_pool_mb
        elif vram_total_mb <= 0:
            pool = 2048
        elif vram_total_mb <= 6144:
            pool = int(vram_total_mb * 0.30)
        elif vram_total_mb <= 8192:
            pool = 3072
        elif vram_total_mb <= 12288:
            pool = 6144
        else:
            pool = int(vram_total_mb * 0.55)
        if mem.display_cache_mb > 0:
            display = mem.display_cache_mb
        else:
            display = 1024 if vram_total_mb <= 8192 else (1536 if vram_total_mb <= 12288 else 3072)
        if mem.ram_cache_mb > 0:
            ram = mem.ram_cache_mb
        else:
            ram = max(2048, min(int(ram_total_mb * 0.38), 49152)) if ram_total_mb > 0 else 4096
        return {"vram_pool_mb": pool, "display_cache_mb": display, "ram_cache_mb": ram, "undo_mb": mem.undo_mb}
