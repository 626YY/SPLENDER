"""降噪：Intel Open Image Denoise（经 pyoidn，MIT 许可的 Python 封装；OIDN 本身是 Apache 2.0）。

优先用显卡（CUDA），没有就用处理器。输入是线性 HDR 颜色，加上第一次击中处的反照率和法线（帮它分清纹理和噪点）。
同一时间只跑一个任务（设备不保证多线程同时用）；可以在后台线程里跑，降噪本身不占 Python 的全局锁。
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

log = logging.getLogger("splender.render.denoise")

_LOCK = threading.Lock()
_STATE: dict = {"device": None, "kind": "", "filters": {}, "error": ""}


def available() -> tuple[bool, str]:
    try:
        from ..paths import ensure_vendor_path

        ensure_vendor_path()
        import pyoidn  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, "没有找到降噪组件：%s" % exc
    return True, ""


def _device(prefer: str):
    """prefer：AUTO（有显卡用显卡）、GPU、CPU。设备建好后复用。"""
    import pyoidn

    wanted = "CPU" if prefer == "CPU" else ("CUDA" if prefer in ("AUTO", "GPU") else "CPU")
    if wanted == "CUDA":
        try:
            if not pyoidn.Device.is_cuda_available():
                wanted = "CPU"
        except Exception:  # noqa: BLE001
            wanted = "CPU"
    if _STATE["device"] is not None and _STATE["kind"] == wanted:
        return _STATE["device"], wanted
    _release_locked()
    kind = pyoidn.OIDN_DEVICE_TYPE_CUDA if wanted == "CUDA" else pyoidn.OIDN_DEVICE_TYPE_CPU
    device = pyoidn.Device(kind)
    device.commit()
    error = device.get_error()
    if error and wanted == "CUDA":
        log.warning("显卡降噪不可用（%s），改用处理器", error)
        device.release()
        device = pyoidn.Device(pyoidn.OIDN_DEVICE_TYPE_CPU)
        device.commit()
        wanted = "CPU"
    _STATE.update(device=device, kind=wanted, filters={})
    log.info("降噪设备：%s", "显卡（CUDA）" if wanted == "CUDA" else "处理器")
    return device, wanted


def _filter(device, kind: str, width: int, height: int):
    """每个尺寸一套滤镜和缓冲（显卡设备的图像必须放在它自己的缓冲里）。"""
    import pyoidn
    from pyoidn.buffer import Buffer

    key = (width, height)
    found = _STATE["filters"].get(key)
    if found is not None:
        return found
    if len(_STATE["filters"]) >= 3:
        for flt, buffers in _STATE["filters"].values():
            flt.release()
            for buf in buffers.values():
                buf.release()
        _STATE["filters"] = {}
    size = width * height * 3 * 4
    buffers = {name: Buffer(device, size) for name in ("color", "albedo", "normal", "output")}
    flt = pyoidn.Filter(device, "RT")
    for name, slot in (("color", pyoidn.OIDN_IMAGE_COLOR), ("albedo", pyoidn.OIDN_IMAGE_ALBEDO),
                       ("normal", pyoidn.OIDN_IMAGE_NORMAL), ("output", pyoidn.OIDN_IMAGE_OUTPUT)):
        flt.set_image(slot, buffers[name], pyoidn.OIDN_FORMAT_FLOAT3, width, height)
    flt.set_bool("hdr", True)
    flt.set_bool("cleanAux", False)
    flt.commit()
    error = device.get_error()
    if error:
        raise RuntimeError(error)
    _STATE["filters"][key] = (flt, buffers)
    return flt, buffers


def denoise(color: np.ndarray, albedo: np.ndarray | None = None, normal: np.ndarray | None = None,
            prefer: str = "AUTO") -> tuple[np.ndarray, dict]:
    """color/albedo/normal：(H, W, 3) float32，第 0 行是图像底部还是顶部无所谓（三者一致即可）。返回 (降噪结果, 统计)。"""
    started = time.perf_counter()
    height, width = color.shape[:2]
    color = np.ascontiguousarray(color[:, :, :3], np.float32)
    albedo = np.ascontiguousarray(albedo[:, :, :3] if albedo is not None else np.ones_like(color), np.float32)
    normal = np.ascontiguousarray(normal[:, :, :3] if normal is not None else np.zeros_like(color), np.float32)
    with _LOCK:
        device, kind = _device(prefer)
        flt, buffers = _filter(device, kind, width, height)
        size = color.nbytes
        buffers["color"].write(0, size, color)
        buffers["albedo"].write(0, size, albedo)
        buffers["normal"].write(0, size, normal)
        flt.execute()
        out = np.empty_like(color)
        buffers["output"].read(0, size, out)
        error = device.get_error()
    if error:
        raise RuntimeError("降噪出错：%s" % error)
    stats = {"device": kind, "ms": round((time.perf_counter() - started) * 1000.0, 1), "size": [width, height]}
    return out, stats


def _release_locked() -> None:
    for flt, buffers in _STATE["filters"].values():
        flt.release()
        for buf in buffers.values():
            buf.release()
    _STATE["filters"] = {}
    if _STATE["device"] is not None:
        _STATE["device"].release()
    _STATE["device"] = None
    _STATE["kind"] = ""


def release() -> None:
    with _LOCK:
        _release_locked()


class DenoiseTask:
    """后台降噪：主线程读出图像交给它，线程里降噪，主线程再取结果。"""

    def __init__(self, color, albedo, normal, prefer: str, tag) -> None:
        self.tag = tag
        self.result: np.ndarray | None = None
        self.stats: dict = {}
        self.error = ""
        self._thread = threading.Thread(target=self._run, args=(color, albedo, normal, prefer), daemon=True,
                                        name="denoise")
        self._thread.start()

    def _run(self, color, albedo, normal, prefer) -> None:
        try:
            self.result, self.stats = denoise(color, albedo, normal, prefer)
        except Exception as exc:  # noqa: BLE001
            log.exception("降噪出错")
            self.error = str(exc) or type(exc).__name__

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()
