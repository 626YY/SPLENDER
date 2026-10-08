"""属性系统：每个可调的值声明一次，界面、存盘、默认值、范围都从这里来。

用法::

    class Brush(PropertyGroup):
        size = FloatProperty("直径", default=40, min=1, max=4096, unit="px")

    brush = Brush()
    brush.size = 80          # 自动夹紧范围，发出 changed("size")
"""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable

from .signals import Signal


class Property:
    """属性描述符基类。"""

    kind = "ANY"

    def __init__(self, label: str = "", *, default: Any = None, description: str = "", hidden: bool = False,
                 save: bool = True, subtype: str = "NONE", update: Callable[[Any, str], None] | None = None,
                 icon: str = "") -> None:
        self.label = label
        self.default = default
        self.description = description
        self.hidden = hidden
        self.save = save
        self.subtype = subtype
        self.update = update
        self.icon = icon
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        if not self.label:
            self.label = name

    def default_value(self, owner: Any = None) -> Any:
        return self.default

    def coerce(self, value: Any, owner: Any = None) -> Any:
        return value

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        values = obj.__dict__.setdefault("_values", {})
        if self.name not in values:
            values[self.name] = self.default_value(obj)
        return values[self.name]

    def __set__(self, obj: Any, value: Any) -> None:
        value = self.coerce(value, obj)
        values = obj.__dict__.setdefault("_values", {})
        old = values.get(self.name, _MISSING)
        if old is _MISSING:
            old = self.default_value(obj)
        if _same(old, value):
            values[self.name] = value
            return
        values[self.name] = value
        obj._notify(self.name)

    def to_json(self, value: Any) -> Any:
        return value

    def from_json(self, data: Any, owner: Any = None) -> Any:
        return self.coerce(data, owner)


_MISSING = object()


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001
        return a is b


class BoolProperty(Property):
    kind = "BOOL"

    def __init__(self, label: str = "", *, default: bool = False, **kw: Any) -> None:
        super().__init__(label, default=bool(default), **kw)

    def coerce(self, value: Any, owner: Any = None) -> bool:
        return bool(value)


class FloatProperty(Property):
    kind = "FLOAT"

    def __init__(self, label: str = "", *, default: float = 0.0, min: float = -math.inf, max: float = math.inf,
                 soft_min: float | None = None, soft_max: float | None = None, step: float | None = None,
                 precision: int = 2, unit: str = "", **kw: Any) -> None:
        super().__init__(label, default=float(default), **kw)
        self.min = float(min)
        self.max = float(max)
        self.soft_min = self.min if soft_min is None else float(soft_min)
        self.soft_max = self.max if soft_max is None else float(soft_max)
        self.step = step
        self.precision = precision
        self.unit = unit

    def coerce(self, value: Any, owner: Any = None) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(self.default)
        if not math.isfinite(number):
            number = float(self.default)
        return min(self.max, max(self.min, number))


class IntProperty(Property):
    kind = "INT"

    def __init__(self, label: str = "", *, default: int = 0, min: int = -(2 ** 31), max: int = 2 ** 31 - 1,
                 soft_min: int | None = None, soft_max: int | None = None, step: int = 1, unit: str = "",
                 **kw: Any) -> None:
        super().__init__(label, default=int(default), **kw)
        self.min = int(min)
        self.max = int(max)
        self.soft_min = self.min if soft_min is None else int(soft_min)
        self.soft_max = self.max if soft_max is None else int(soft_max)
        self.step = step
        self.unit = unit
        self.precision = 0

    def coerce(self, value: Any, owner: Any = None) -> int:
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError):
            number = int(self.default)
        return min(self.max, max(self.min, number))


class StringProperty(Property):
    kind = "STRING"

    def __init__(self, label: str = "", *, default: str = "", maxlen: int = 0, **kw: Any) -> None:
        super().__init__(label, default=str(default), **kw)
        self.maxlen = maxlen

    def coerce(self, value: Any, owner: Any = None) -> str:
        text = "" if value is None else str(value)
        return text[: self.maxlen] if self.maxlen else text


class EnumProperty(Property):
    """枚举。items 是 [(标识, 显示名, 说明[, 图标])]，也可以是返回这种列表的函数。"""

    kind = "ENUM"

    def __init__(self, label: str = "", *, items: Iterable[tuple] | Callable[[Any], Iterable[tuple]] = (),
                 default: str | None = None, **kw: Any) -> None:
        super().__init__(label, default=default, **kw)
        self._items = items

    def items(self, owner: Any = None) -> list[tuple]:
        raw = self._items(owner) if callable(self._items) else self._items
        out = []
        for item in raw:
            item = tuple(item)
            ident = item[0]
            text = item[1] if len(item) > 1 else str(ident)
            desc = item[2] if len(item) > 2 else ""
            icon = item[3] if len(item) > 3 else ""
            out.append((ident, text, desc, icon))
        return out

    def default_value(self, owner: Any = None) -> Any:
        if self.default is not None:
            return self.default
        items = self.items(owner)
        return items[0][0] if items else ""

    def coerce(self, value: Any, owner: Any = None) -> Any:
        idents = [item[0] for item in self.items(owner)]
        if value in idents or callable(self._items):
            return value
        return self.default_value(owner)

    def label_of(self, value: Any, owner: Any = None) -> str:
        for ident, text, _desc, _icon in self.items(owner):
            if ident == value:
                return text
        return str(value)


class FloatVectorProperty(Property):
    kind = "VECTOR"

    def __init__(self, label: str = "", *, default: Iterable[float] = (0.0, 0.0, 0.0), size: int | None = None,
                 min: float = -math.inf, max: float = math.inf, precision: int = 3, unit: str = "",
                 **kw: Any) -> None:
        default = tuple(float(v) for v in default)
        super().__init__(label, default=default, **kw)
        self.size = size or len(default)
        self.min = float(min)
        self.max = float(max)
        self.precision = precision
        self.unit = unit

    def coerce(self, value: Any, owner: Any = None) -> tuple:
        try:
            items = [float(v) for v in value]
        except (TypeError, ValueError):
            return tuple(self.default)
        items = (items + list(self.default))[: self.size]
        return tuple(min(self.max, max(self.min, v)) if math.isfinite(v) else d
                     for v, d in zip(items, self.default))

    def to_json(self, value: Any) -> list:
        return list(value)


class ColorProperty(FloatVectorProperty):
    """颜色，分量 0..1，按 sRGB 显示。size=3 无透明，size=4 带透明。"""

    kind = "COLOR"

    def __init__(self, label: str = "", *, default: Iterable[float] = (1.0, 1.0, 1.0), **kw: Any) -> None:
        kw.setdefault("min", 0.0)
        kw.setdefault("max", 1.0)
        super().__init__(label, default=default, **kw)


class RampProperty(Property):
    """色带（照 Blender 的颜色渐变）：值是 (插值方式, ((位置, r, g, b, a), ...))，见 core/ramp.py。"""

    kind = "RAMP"

    def __init__(self, label: str = "", *, default: Any = None, **kw: Any) -> None:
        from . import ramp

        super().__init__(label, default=ramp.normalize(default if default is not None else ramp.DEFAULT), **kw)

    def coerce(self, value: Any, owner: Any = None) -> tuple:
        from . import ramp

        return ramp.normalize(value)

    def to_json(self, value: Any) -> dict:
        from . import ramp

        return ramp.to_json(value)


class CurveProperty(Property):
    """曲线（照 Photoshop / Blender 的曲线调整）：值是 ((输入, 输出), ...)，见 core/curve.py。
    subtype 写通道（RGB / R / G / B），控件按它画颜色。"""

    kind = "CURVE"

    def __init__(self, label: str = "", *, default: Any = None, **kw: Any) -> None:
        from . import curve

        super().__init__(label, default=curve.normalize(default if default is not None else curve.DEFAULT), **kw)

    def coerce(self, value: Any, owner: Any = None) -> tuple:
        from . import curve

        return curve.normalize(value)

    def to_json(self, value: Any) -> list:
        from . import curve

        return curve.to_json(value)


class PointerProperty(Property):
    """嵌套的属性组。"""

    kind = "POINTER"

    def __init__(self, label: str = "", *, type: type, **kw: Any) -> None:  # noqa: A002
        super().__init__(label, default=None, **kw)
        self.group_type = type

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        values = obj.__dict__.setdefault("_values", {})
        group = values.get(self.name)
        if group is None:
            group = self.group_type()
            group._parent = (obj, self.name)
            values[self.name] = group
        return group

    def __set__(self, obj: Any, value: Any) -> None:
        if isinstance(value, dict):
            self.__get__(obj).from_dict(value)
        elif isinstance(value, PropertyGroup):
            self.__get__(obj).copy_from(value)
        else:
            raise TypeError("嵌套属性组只能用字典或同类型对象赋值")

    def to_json(self, value: Any) -> Any:
        return value.to_dict()

    def from_json(self, data: Any, owner: Any = None) -> Any:
        return data


class PropertyGroup:
    """一组属性。值变化时发出 changed(属性名)；嵌套组的变化以 "子组.属性" 向上传。"""

    _parent: tuple[Any, str] | None = None

    def __init__(self, **values: Any) -> None:
        self.__dict__.setdefault("_values", {})
        self.changed = Signal()
        self._parent = None
        for name, value in values.items():
            if name in self.properties():
                setattr(self, name, value)

    @classmethod
    def properties(cls) -> dict[str, Property]:
        cached = cls.__dict__.get("_props_cache")
        if cached is not None:
            return cached
        found: dict[str, Property] = {}
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, Property):
                    found[name] = value
        cls._props_cache = found
        return found

    @classmethod
    def prop(cls, name: str) -> Property:
        return cls.properties()[name]

    def _notify(self, name: str) -> None:
        prop = self.properties().get(name.split(".")[0])
        if prop is not None and prop.update is not None and "." not in name:
            prop.update(self, name)
        self.on_changed(name)
        changed = self.__dict__.get("changed")
        if changed is not None:
            changed.emit(name)
        parent = self.__dict__.get("_parent")
        if parent:
            owner, attr = parent
            owner._notify(attr + "." + name)

    def on_changed(self, name: str) -> None:
        """子类可覆盖：属性变化后的处理。"""

    def to_dict(self) -> dict:
        out = {}
        for name, prop in self.properties().items():
            if not prop.save:
                continue
            out[name] = prop.to_json(getattr(self, name))
        return out

    def from_dict(self, data: dict | None) -> None:
        if not data:
            return
        for name, prop in self.properties().items():
            if name not in data or not prop.save:
                continue
            try:
                if isinstance(prop, PointerProperty):
                    getattr(self, name).from_dict(data[name])
                else:
                    setattr(self, name, prop.from_json(data[name], self))
            except Exception:  # noqa: BLE001  单个坏值不该毁掉整份配置
                continue

    def copy_from(self, other: "PropertyGroup") -> None:
        self.from_dict(other.to_dict())

    def reset(self, name: str | None = None) -> None:
        names = [name] if name else list(self.properties())
        for item in names:
            prop = self.properties()[item]
            if isinstance(prop, PointerProperty):
                getattr(self, item).reset()
            else:
                setattr(self, item, prop.default_value(self))

    def is_default(self, name: str) -> bool:
        prop = self.properties()[name]
        return _same(getattr(self, name), prop.default_value(self))


def resolve_path(root: Any, path: str) -> tuple[Any, str]:
    """把 "brush.size" 解析成 (brush 对象, "size")。"""
    parts = path.split(".")
    target = root
    for part in parts[:-1]:
        target = getattr(target, part)
    return target, parts[-1]


def get_path(root: Any, path: str) -> Any:
    target, name = resolve_path(root, path)
    return getattr(target, name)


def set_path(root: Any, path: str, value: Any) -> None:
    target, name = resolve_path(root, path)
    setattr(target, name, value)
