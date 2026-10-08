"""渲染引擎注册表（相当于 Blender 的 RenderEngine）。

内置三个：工作台、实时、路径追踪（splender/render/builtin.py）。插件可以注册更多，例如 Arnold：

    class MyEngine(RenderEngine):
        idname = "MY_ENGINE"
        label = "我的引擎"
        settings_class = MySettings          # 可选：引擎自己的渲染设置，显示在属性编辑器的「渲染」页，跟着工程保存

        @classmethod
        def create_job(cls, request): ...    # 返回 RenderJob

    register_engine(MyEngine)                # 插件的 register() 里调用；unregister() 里调用 unregister_engine
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any

import numpy as np

from ..core.signals import Signal

log = logging.getLogger("splender.render")

#: 注册的引擎有变化时发出
changed = Signal()
_ENGINES: dict[str, type] = {}


class RenderRequest:
    """一次渲染需要的东西。

    scene：场景快照（RenderScene），需要它的引擎（needs_scene 为真）才会准备。
    engine_settings：这个引擎自己的设置（settings_class 的实例），没有时为 None。
    app_engine、view、shading：直接借用视口渲染器的引擎（实时、工作台）用。
    """

    def __init__(self, *, scene=None, settings=None, engine_settings=None, app_engine=None, view=None,
                 shading=None, render_props=None) -> None:
        self.scene = scene
        self.settings = settings
        self.engine_settings = engine_settings
        self.app_engine = app_engine
        self.view = view
        self.shading = shading
        self.render_props = render_props


class RenderJob:
    """一次最终渲染（F12）。step() 由程序每帧调用（在显卡上下文里）。"""

    def __init__(self, request: RenderRequest) -> None:
        self.request = request
        self.progress = 0.0
        self.done = False
        self.error = ""
        self.status = ""
        self.cancelled = False
        self.started = time.perf_counter()
        self.finished_at = 0.0

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.perf_counter()) - self.started

    def step(self, budget_ms: float) -> None:
        raise NotImplementedError

    def preview(self) -> np.ndarray | None:
        """当前画面：(高, 宽, 4) uint8，sRGB，第 0 行是顶行。"""
        return None

    def result(self) -> np.ndarray | None:
        """最终的线性 HDR 画面 (高, 宽, 4) float32；引擎给不出时为 None（这时用 preview）。"""
        return None

    def cancel(self) -> None:
        self.cancelled = True

    def finish(self, error: str = "") -> None:
        if error:
            self.error = error
        self.done = True
        self.finished_at = time.perf_counter()

    def release(self) -> None:
        """释放显卡资源（在显卡上下文里调用）。"""


class ThreadedRenderJob(RenderJob):
    """在后台线程里渲染（不用 SPLENDER 的显卡上下文，例如调用外部渲染器）。子类实现 run()。

    run() 里定期检查 self.cancelled，用 set_preview / set_result 交出画面，用 self.progress、self.status 报进度。
    """

    def __init__(self, request: RenderRequest) -> None:
        super().__init__(request)
        self._lock = threading.Lock()
        self._preview: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._thread = threading.Thread(target=self._main, name="splender-render", daemon=True)
        self._thread.start()

    def _main(self) -> None:
        try:
            self.run()
            self.finish()
        except Exception as error:  # noqa: BLE001
            log.error("渲染出错：%s\n%s", error, traceback.format_exc())
            self.finish("%s" % error)

    def run(self) -> None:
        raise NotImplementedError

    def set_preview(self, image: np.ndarray) -> None:
        with self._lock:
            self._preview = image

    def set_result(self, image: np.ndarray) -> None:
        with self._lock:
            self._result = image

    def step(self, budget_ms: float) -> None:
        pass

    def preview(self) -> np.ndarray | None:
        with self._lock:
            return self._preview

    def result(self) -> np.ndarray | None:
        with self._lock:
            return self._result


class ViewportRender:
    """在视口里实时预览（「渲染」着色模式）。sync() 每帧调用，跟上相机、环境和图层的变化；draw() 把画面画进视口。"""

    def sync(self, request: RenderRequest, changes: set[str]) -> None:
        raise NotImplementedError

    def draw(self, view, budget_ms: float) -> bool:
        """推进并把当前画面画进 view。返回是否还没收敛（还要继续画）。"""
        raise NotImplementedError

    def status(self) -> str:
        return ""

    def release(self) -> None:
        pass


class RenderEngine:
    idname = ""
    label = ""
    description = ""
    icon = "render"
    order = 100
    supports_viewport = False      # 能在视口的「渲染」着色模式里实时预览
    needs_scene = True             # 需要场景快照（会把图层合成成贴图）
    settings_class: type | None = None

    @classmethod
    def available(cls) -> tuple[bool, str]:
        """能不能用；不能用时返回原因，界面里显示给用户。"""
        return True, ""

    @classmethod
    def create_job(cls, request: RenderRequest) -> RenderJob:
        raise NotImplementedError

    @classmethod
    def create_viewport(cls, request: RenderRequest) -> ViewportRender | None:
        return None


# ------------------------------------------------------------------ 注册表
def register_engine(cls: type) -> type:
    _ENGINES[cls.idname] = cls
    changed.emit()
    return cls


def unregister_engine(idname: str) -> None:
    if _ENGINES.pop(idname, None) is not None:
        changed.emit()


def engines() -> list[type]:
    return sorted(_ENGINES.values(), key=lambda c: (c.order, c.label))


def engine(idname: str) -> type | None:
    return _ENGINES.get(idname)


def engine_items(_owner: Any = None) -> list[tuple]:
    """给下拉框用：[(idname, 名字, 说明)]。"""
    items = []
    for cls in engines():
        ok, reason = cls.available()
        items.append((cls.idname, cls.label, cls.description if ok else "%s（%s）" % (cls.description, reason)))
    return items or [("PATHTRACE", "路径追踪", "")]
