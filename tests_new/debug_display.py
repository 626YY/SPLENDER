"""调试：落笔过程中把显示贴图各级读出来看。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splender.paths import ensure_vendor_path

ensure_vendor_path()
import ctypes

import moderngl
import numpy as np
from PIL import Image

from splender.core.history import History
from splender.core.prefs import Preferences
from splender.doc.project import ToolSettings, ViewShading
from splender.engine import glx
from splender.engine.engine import Engine
from tests_new.test_engine_paint import make_project

OUT = Path(__file__).resolve().parents[1].joinpath("docs/evidence/engine/debug")
OUT.mkdir(parents=True, exist_ok=True)
ctx = moderngl.create_standalone_context(require=430)
engine = Engine(ctx, Preferences(), History())
project, ts = make_project()
engine.set_project(project)
view = engine.create_view(ViewShading())
view.resize(1280, 800)
engine.frame_view(view)
engine.settle()
tools = ToolSettings()
tools.brush.size = 90
tools.brush.color = (0.85, 0.2, 0.15)
cx, cy = view.width / 2, view.height / 2
engine.stroke_begin(view, cx - 170, cy - 40, 1.0, time.perf_counter(), tools)
for i in range(1, 41):
    t = i / 40
    engine.stroke_move(cx - 170 + 340 * t, cy - 40 + 90 * np.sin(t * 6.283), 1.0, time.perf_counter())
    engine.frame()

state = next(iter(engine.sets.values()))
display = state.display
print("levels", display.levels, "base", display.base_level)
for level in range(display.levels):
    g = display.level_size[level]
    size = g * 256
    print("level", level, "grid", g, "visible", int(state.visible[level].sum()), "committed", int(display.committed[level].sum()),
          "valid", int(display.valid[level].sum()), "stale", int(display.stale[level].sum()),
          "stroke pages", len(engine.stroke.buffer.store.levels[level].pages))
    if size > 4096:
        continue
    glx.fn("glBindFramebuffer")(glx.GL_FRAMEBUFFER, display.fbo(level))
    glx.fn("glReadBuffer") if False else None
    buf = (ctypes.c_ubyte * (size * size * 4))()
    glx.fn("glPixelStorei")(glx.GL_PACK_ALIGNMENT, 1)
    glx.fn("glReadPixels")(0, 0, size, size, glx.GL_RGBA, glx.GL_UNSIGNED_BYTE, buf)
    image = np.frombuffer(buf, np.uint8).reshape(size, size, 4)[::-1, :, :3]
    Image.fromarray(image).resize((512, 512)).save(OUT / ("color_level%d.png" % level))
print("clamp unique", np.unique(display.clamp, return_counts=True))
print("requests", getattr(view, "requests", None)[:20] if getattr(view, "requests", None) is not None else None)
