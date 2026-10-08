"""工程的渲染设置（属性编辑器「渲染」页）。每个引擎自己的设置挂在这里，跟着工程一起保存。"""
from __future__ import annotations

from typing import Any

from ..core.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PropertyGroup
from . import engines

TEXTURE_SIZES = [("1024", "1K", ""), ("2048", "2K", ""), ("4096", "4K", ""), ("8192", "8K", ""), ("16384", "16K", "")]


class RenderProps(PropertyGroup):
    engine = EnumProperty("渲染引擎", items=engines.engine_items, default="PATHTRACE",
                          description="出图和视口「渲染」模式用哪个引擎")
    resolution_x = IntProperty("宽度", default=1920, min=16, max=16384, unit="px")
    resolution_y = IntProperty("高度", default=1080, min=16, max=16384, unit="px")
    resolution_percentage = IntProperty("缩放", default=100, min=1, max=400, unit="%",
                                        description="出图尺寸按这个比例缩放，试渲时调小")
    texture_size = EnumProperty("贴图精度", items=TEXTURE_SIZES, default="4096",
                                description="出图时图层合成成多大的贴图。越大细节越多，准备时间和显存也越多")
    transparent = BoolProperty("透明背景", default=False, description="背景透明，存 PNG 时带透明通道")
    view_transform = EnumProperty("显示变换", items=[("STANDARD", "标准", ""), ("FILMIC", "Filmic", ""),
                                                 ("AGX", "AgX", "")], default="FILMIC")
    exposure = FloatProperty("曝光", default=0.0, min=-10.0, max=10.0, soft_min=-4.0, soft_max=4.0, unit="EV",
                             precision=2)

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        self.__dict__["_engine_settings"] = {}
        self.__dict__["_saved_engine_settings"] = {}

    @property
    def size(self) -> tuple[int, int]:
        scale = max(1, int(self.resolution_percentage)) / 100.0
        return max(16, int(round(self.resolution_x * scale))), max(16, int(round(self.resolution_y * scale)))

    def engine_settings(self, idname: str | None = None):
        """某个引擎自己的设置（没有设置的引擎返回 None）。"""
        idname = idname or self.engine
        cache = self.__dict__["_engine_settings"]
        found = cache.get(idname)
        if found is None:
            cls = engines.engine(idname)
            settings_cls = getattr(cls, "settings_class", None) if cls is not None else None
            if settings_cls is None:
                return None
            found = settings_cls()
            stored = self.__dict__["_saved_engine_settings"].get(idname)
            if stored:
                found.from_dict(stored)
            found.changed.connect(lambda name, owner=self: owner.changed.emit("engine_settings"))
            cache[idname] = found
        return found

    def to_saved(self) -> dict:
        data = self.to_dict()
        saved = dict(self.__dict__["_saved_engine_settings"])
        for idname, settings in self.__dict__["_engine_settings"].items():
            saved[idname] = settings.to_dict()
        data["engines"] = saved
        return data

    def from_saved(self, data: dict | None) -> None:
        data = dict(data or {})
        saved = data.pop("engines", {}) or {}
        self.from_dict(data)
        self.__dict__["_saved_engine_settings"] = dict(saved)
        self.__dict__["_engine_settings"] = {}
