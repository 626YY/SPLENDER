"""选区（照 Photoshop）。每套贴图一个：一个不在图层列表里的隐藏图层，只有蒙版页（r8），缺页的格按 default 算。

- active：有没有选区。没有选区时画笔、滤镜哪里都能改；取消选择只是关掉 active，页留着给「重新选择」。
- 全选：页全去掉、default = 1。反选：default 取反、已有的页逐个取反。新选区：页全去掉、default = 0，再把形状盖上去。
- 形状（矩形、椭圆、套索、魔棒、快速选择……）都是一张单通道遮罩：UV 视图里按 UV 矩形盖（IMAGE），
  三维视口里按屏幕矩形投到模型上（SCREEN_IMAGE）。添加 = 盖白，减去 = 盖黑，交叉 = 相乘（形状外归 0）。
- 羽化、扩展、收缩、平滑、边界：对选区的页跑滤镜（高斯模糊、最大值、最小值、中间值、边界）。
- default 变了（反选、交叉、边界）或者去掉过页之后，上面各级按第 0 级和新的 default 从头生成（缺页的象限按 default 填）。
- 每次改动记一步撤销：页的记录（按做的顺序；撤销时倒着放回）+ 改动前后的 (active, default)。
- 用的地方：笔刷盖章（engine/stamp.py）、滤镜写回（engine/filters.py）用 slots_for 拿每格的选区页槽号，乘上选区值。
- 显示：preview() 把选区某一级读回来做一张小贴图（r8），视口按它画流动的虚线。
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from ..doc.project import PLANE_MASK, Layer
from .layerstore import RecordGroup
from .pageops import KIND_MASK
from .pagepool import PAGE

PREVIEW = 2048            # 显示用的小贴图最大边长
#: 选区修改用的滤镜和它往外够多远（按半径的倍数，第 0 级纹素）
MODIFY = {"FEATHER": ("GAUSSIAN_BLUR", 3.0), "EXPAND": ("MAXIMUM", 1.0), "CONTRACT": ("MINIMUM", 1.0),
          "SMOOTH": ("MEDIAN", 1.0), "BORDER": ("SEL_BORDER", 3.0)}
MODIFY_LABELS = {"FEATHER": "羽化选区", "EXPAND": "扩展选区", "CONTRACT": "收缩选区", "SMOOTH": "平滑选区",
                 "BORDER": "选区边界"}


class Selection:
    def __init__(self, engine, ts) -> None:
        self.engine = engine
        self.ts = ts
        self.layer = Layer(name="选区", kind="PAINT")
        self.layer.mask_default = 0.0
        self.active = False
        self.version = 0                    # 每改一次加一，显示用的小贴图按它重做
        self._preview = None
        self._preview_version = -1
        self.preview_max = 0                # 小贴图里最大的值（0..255）
        self.emptied = False                # 最近一次改完什么都没选上（已经自动取消选择）

    # ================================================================== 查询
    @property
    def default(self) -> float:
        return float(self.layer.mask_default)

    @property
    def grid(self) -> int:
        return int(self.ts.size) // PAGE

    def plane(self):
        store = self.engine.layers.stores.get(self.layer.uid)
        return store.planes.get(PLANE_MASK) if store is not None else None

    def cells(self) -> set:
        plane = self.plane()
        return set(plane.levels[0].pages) if plane is not None else set()

    def applies_to(self, layer) -> bool:
        """改这个图层时要不要按选区限制（选区自己不受限制）。"""
        return self.active and getattr(layer, "uid", None) != self.layer.uid

    def can_reselect(self) -> bool:
        return not self.active and (self.default > 0.5 or bool(self.cells()))

    def slots_for(self, xs, ys) -> tuple[np.ndarray, list]:
        """第 0 级这些格的选区页槽号（没有页的 -1，按 default 算）。有页的先放进显存并钉住，用完交给 unpin。"""
        xs = np.asarray(xs, np.int64)
        ys = np.asarray(ys, np.int64)
        plane = self.plane()
        if plane is None:
            return np.full(len(xs), -1, np.int32), []
        level = plane.levels[0]
        pages = [level.pages.get((x, y)) for x, y in zip(xs.tolist(), ys.tolist())]
        needed = [page for page in pages if page is not None]
        if needed:
            cache = self.engine.cache
            cache.make_resident(needed)
            cache.pin_many(needed)
        return np.fromiter((page.slot if page is not None else -1 for page in pages), np.int32, len(pages)), needed

    def exists_at(self, xs, ys) -> np.ndarray:
        """第 0 级这些格有没有选区页。"""
        plane = self.plane()
        if plane is None:
            return np.zeros(len(xs), bool)
        return plane.levels[0].exists[np.asarray(ys, np.int64), np.asarray(xs, np.int64)].copy()

    def unpin(self, pages: list) -> None:
        if pages:
            self.engine.cache.pin_many(pages, -1)

    # ================================================================== 改选区（每个都记一步撤销）
    def select_all(self) -> bool:
        return self._change("全选", lambda: (self._drop(), (True, 1.0)))

    def deselect(self) -> bool:
        if not self.active:
            return False
        return self._change("取消选择", lambda: ([], (False, self.default)))

    def reselect(self) -> bool:
        if not self.can_reselect():
            return False
        return self._change("重新选择", lambda: ([], (True, self.default)))

    def invert(self) -> bool:
        if not self.active:
            return False

        def work():
            cells = sorted(self.cells(), key=lambda c: (c[1], c[0]))
            records = self._filter("SEL_INVERT", {"cells": cells}) if cells else []
            self.layer.mask_default = 1.0 - self.default
            records += self._rebuild_mips()
            return records, (True, self.default)

        return self._change("反选", work)

    def combine(self, mode: str, name: str, params: dict, label: str) -> bool:
        """把一张遮罩形状按 mode（SET 新选区、ADD 添加、SUBTRACT 减去、INTERSECT 交叉）合进选区。
        name 是 IMAGE（按 UV 矩形盖）或 SCREEN_IMAGE（按屏幕矩形投到模型上），params 是那个滤镜的参数，
        image 是单通道遮罩（0..255 或 0..1）。没有选区时都当新选区。"""
        if not self.active:
            mode = "SET"

        def work():
            records = []
            if mode == "SET":
                records += self._drop()
                self.layer.mask_default = 0.0
            p = dict(params)
            p["color2"] = (0.0, 0.0, 0.0, 1.0) if mode == "SUBTRACT" else (1.0, 1.0, 1.0, 1.0)
            if mode == "INTERSECT":
                # 相乘：形状碰不到的地方归 0。已有的页都要过一遍（乘 0），没有页的格靠 default 归 0
                p["combine"] = "MULTIPLY"
                p["extra_cells"] = sorted(self.cells())
            records += self._filter(name, p)
            if mode == "INTERSECT":
                self.layer.mask_default = 0.0
                records += self._rebuild_mips()
            return records, (True, self.default)

        return self._change(label, work)

    def modify(self, kind: str, amount: float) -> bool:
        """羽化（FEATHER）、扩展（EXPAND）、收缩（CONTRACT）、平滑（SMOOTH）、边界（BORDER）。amount 按纹素。"""
        if not self.active or kind not in MODIFY:
            return False
        name, reach = MODIFY[kind]
        amount = max(0.5, float(amount))

        def work():
            cells = self._reach(amount * reach + 2.0)
            if not cells:
                return None
            records = self._filter(name, {"radius": amount, "cells": cells})
            if kind == "BORDER":
                self.layer.mask_default = 0.0
                records += self._rebuild_mips()
            return records, (True, self.default)

        return self._change(MODIFY_LABELS[kind], work)

    def load_mask(self, layer) -> bool:
        """把图层的蒙版当选区：页原样共用（不改图层），缺页的格按图层蒙版的初始值。"""
        engine = self.engine
        store = engine.layers.stores.get(layer.uid)
        source = store.planes.get(PLANE_MASK) if store is not None else None

        def work():
            records = self._drop()
            if source is not None:
                records += self._clone_into(source, engine.layers.store(self.layer.uid, self.grid).plane(PLANE_MASK))
            return records, (True, float(layer.mask_default))

        return self._change("载入蒙版为选区", work)

    def to_mask(self, layer, label: str = "选区变蒙版") -> bool:
        """给图层加一个和选区一样的蒙版（图层原来没有蒙版时）。记一步撤销。"""
        if not self.active or layer.has_mask:
            return False
        engine = self.engine
        engine.complete_strokes(include_active=True)
        target = engine.layers.store(layer.uid, self.grid).plane(PLANE_MASK)
        records = self._drop_plane(target) + self._clone_into(self.plane(), target)
        before = (bool(layer.has_mask), float(layer.mask_default))
        after = (True, self.default)

        def state(values) -> None:
            layer.mask_default, layer.has_mask = values[1], values[0]
            self._refresh_layer()

        state(after)
        self._push(label, records, lambda: state(before), lambda: state(after))
        return True

    # ================================================================== 连着改好几下，只记一步（快速选择拖一笔）
    def begin_batch(self) -> None:
        self._batch = {"records": [], "before": None}

    def end_batch(self, label: str) -> bool:
        """把这一批改动记成一步撤销。一下都没改时返回 False。"""
        batch, self._batch = getattr(self, "_batch", None), None
        if not batch or batch["before"] is None:
            return False
        before, after = batch["before"], (self.active, self.default)
        self._push(label, batch["records"], lambda: self._set_state(*before), lambda: self._set_state(*after))
        return True

    def cancel_batch(self) -> None:
        """放弃这一批：页和状态都退回去，新生成的页放掉。"""
        batch, self._batch = getattr(self, "_batch", None), None
        if not batch or batch["before"] is None:
            return
        engine = self.engine
        engine.layers.apply_records(list(reversed(batch["records"])), undo=True)
        engine.layers.dispose_records(batch["records"], applied=False)
        self._set_state(*batch["before"])

    # ================================================================== 内部
    def _change(self, label: str, work) -> bool:
        """做一次改动、记一步撤销。work() 返回 (页的记录, (新 active, 新 default))，返回 None 表示什么都没改。
        在一批里（begin_batch）时先攒着，end_batch 时一起记。"""
        engine = self.engine
        engine.complete_strokes(include_active=True)
        before = (self.active, self.default)
        result = work()
        if result is None:
            self.layer.mask_default = before[1]
            return False
        records, after = result
        self.emptied = False
        if after[0] and after[1] < 0.5 and not self.cells():
            # 什么都没选上（比如全选后反选）：和 Photoshop 一样直接取消选择，免得画笔哪里都画不上
            after = (False, after[1])
            self.emptied = True
        self._set_state(*after)
        batch = getattr(self, "_batch", None)
        if batch is not None:
            if batch["before"] is None:
                batch["before"] = before
            batch["records"].extend(records)
            return True
        self._push(label, records, lambda: self._set_state(*before), lambda: self._set_state(*after))
        return True

    def _push(self, label: str, records: list, undo_state, redo_state) -> None:
        engine = self.engine
        history = engine.history
        if history is None:
            return
        flag = {"applied": True}

        def undo() -> None:
            engine.layers.apply_records(list(reversed(records)), undo=True)
            undo_state()
            flag["applied"] = False

        def redo() -> None:
            engine.layers.apply_records(records, undo=False)
            redo_state()
            flag["applied"] = True

        def dispose() -> None:
            engine.layers.dispose_records(records, flag["applied"])

        history.push(label, undo, redo, engine.layers.records_bytes(records), dispose)
        history.steps[-1].records = records

    def _set_state(self, active: bool, default: float) -> None:
        self.active = bool(active)
        self.layer.mask_default = float(default)
        self.version += 1
        engine = self.engine
        for view in engine.views:
            view.dirty = True
        engine._pending_work = True
        for fn in list(engine.selection_changed):
            fn(self.ts)

    def _refresh_layer(self) -> None:
        engine = self.engine
        state = engine.sets.get(self.ts.uid)
        if state is not None:
            state.params_dirty = True
            state.display.mark_all_dirty()
        engine._touch_content()
        for view in engine.views:
            view.dirty = True
        engine._pending_work = True

    def _filter(self, name: str, params: dict) -> list:
        """对选区的页跑一次滤镜（选区自己不受选区限制），返回撤销记录。"""
        from .filters import LayerFilter

        params = dict(params, ignore_selection=True)
        runner = LayerFilter(self.engine, self.ts, self.layer, name, params, {PLANE_MASK: (True,)})
        records = runner.run()
        self.engine.last_filter_stats = dict(runner.stats)
        return records

    def _reach(self, distance: float) -> list:
        """已有页的格往外扩 distance 个纹素够到的格（缺页的格按 default 算，default 是 1 时它们也会变）。"""
        grid = self.grid
        cells = self.cells()
        if not cells:
            return []
        steps = int(math.ceil(distance / PAGE))
        found = set()
        for x, y in cells:
            for dy in range(-steps, steps + 1):
                for dx in range(-steps, steps + 1):
                    if 0 <= x + dx < grid and 0 <= y + dy < grid:
                        found.add((x + dx, y + dy))
        return sorted(found, key=lambda c: (c[1], c[0]))

    def _drop(self) -> list:
        plane = self.plane()
        return self._drop_plane(plane) if plane is not None else []

    def _drop_plane(self, plane) -> list:
        """去掉一种页的全部页（各级），返回撤销记录。"""
        records = []
        set_page = self.engine.cache.set_page
        for mip, level in enumerate(plane.levels):
            items = list(level.pages.items())
            if not items:
                continue
            xs = [cell[0] for cell, _page in items]
            ys = [cell[1] for cell, _page in items]
            for x, y in zip(xs, ys):
                set_page(level, x, y, None)
            records.append(RecordGroup(level, mip, xs, ys, [page for _cell, page in items], [None] * len(items)))
        return records

    def _clone_into(self, source, target) -> list:
        """把 source 的页（各级）复制到 target（target 里原来没有页），返回撤销记录。"""
        if source is None:
            return []
        cache = self.engine.cache
        records = []
        for mip, level in enumerate(source.levels):
            items = list(level.pages.items())
            if not items or mip >= len(target.levels):
                continue
            there = target.levels[mip]
            news = [cache.clone_page(page) for _cell, page in items]
            xs = [cell[0] for cell, _page in items]
            ys = [cell[1] for cell, _page in items]
            olds = [there.pages.get((x, y)) for x, y in zip(xs, ys)]
            for x, y, page in zip(xs, ys, news):
                cache.set_page(there, x, y, page)
            records.append(RecordGroup(there, mip, xs, ys, olds, news))
        return records

    def _rebuild_mips(self) -> list:
        """按第 0 级和现在的 default 从头生成上面各级，返回撤销记录。"""
        plane = self.plane()
        if plane is None:
            return []
        engine = self.engine
        cache = engine.cache
        default = self.default
        records = []
        levels = plane.levels
        made = 0
        for mip in range(1, len(levels)):
            below, here = levels[mip - 1], levels[mip]
            wanted = sorted({(x >> 1, y >> 1) for (x, y) in below.pages}, key=lambda c: (c[1], c[0]))
            keep = set(wanted)
            stale = [(cell, page) for cell, page in here.pages.items() if cell not in keep]
            if stale:
                for (x, y), _page in stale:
                    cache.set_page(here, x, y, None)
                records.append(RecordGroup(here, mip, [c[0] for c, _p in stale], [c[1] for c, _p in stale],
                                           [p for _c, p in stale], [None] * len(stale)))
            for start in range(0, len(wanted), 256):
                chunk = wanted[start:start + 256]
                kids = []
                for i, (x, y) in enumerate(chunk):
                    for q in range(4):
                        page = below.pages.get((2 * x + (q & 1), 2 * y + (q >> 1)))
                        if page is not None:
                            kids.append((i, q, page))
                sources = [kid[2] for kid in kids]
                cache.make_resident(sources)
                cache.pin_many(sources)
                try:
                    children = np.full((len(chunk), 4), -1, np.int32)
                    for i, q, page in kids:
                        children[i, q] = page.slot
                    news = cache.try_new_pages("r8", len(chunk), block=True)
                    if len(news) < len(chunk):
                        raise MemoryError("显存页池放不下选区的新页")
                    dst = np.fromiter((page.slot for page in news), np.int32, len(news))
                    engine.ops.downsample(KIND_MASK, dst, np.full(len(news), -1, np.int32), children, empty=default)
                    cache.note_new_pages(news)
                    olds = [here.pages.get(cell) for cell in chunk]
                    for (x, y), page in zip(chunk, news):
                        cache.set_page(here, x, y, page)
                    records.append(RecordGroup(here, mip, [c[0] for c in chunk], [c[1] for c in chunk], olds, news))
                finally:
                    cache.pin_many(sources, -1)
                made += len(chunk)
                if made >= 256:
                    made = 0
                    cache.service(cache.frame + 1, budget_ms=2.0)
        return records

    # ================================================================== 显示
    def preview(self):
        """显示用的小贴图（r8，边长不超过 PREVIEW，行序和贴图一样）：选区在对应一级的样子，缺页的格按 default。
        没有选区时 None。"""
        if not self.active:
            return None
        if self._preview is not None and self._preview_version == self.version:
            return self._preview
        size = int(self.ts.size)
        mip = max(0, int(math.ceil(math.log2(max(1, size / PREVIEW))))) if size > PREVIEW else 0
        plane = self.plane()
        if plane is not None:
            mip = min(mip, len(plane.levels) - 1)
        side = max(1, size >> mip)
        image = np.full((side, side), int(round(self.default * 255.0)), np.uint8)
        if plane is not None:
            items = list(plane.levels[mip].pages.items())
            raws = self.engine.cache.fetch_bytes([page for _cell, page in items]) if items else []
            for ((x, y), _page), raw in zip(items, raws):
                tile = np.frombuffer(raw, np.uint8).reshape(PAGE, PAGE)
                image[y * PAGE:(y + 1) * PAGE, x * PAGE:(x + 1) * PAGE] = tile[:side - y * PAGE, :side - x * PAGE]
        self.preview_max = int(image.max()) if image.size else 0
        ctx = self.engine.ctx
        if self._preview is None or self._preview.size != (side, side):
            if self._preview is not None:
                self._preview.release()
            self._preview = ctx.texture((side, side), 1, dtype="f1")
            self._preview.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._preview.repeat_x = False
            self._preview.repeat_y = False
        self._preview.write(np.ascontiguousarray(image).tobytes())
        self._preview_version = self.version
        return self._preview

    def release(self) -> None:
        """丢掉选区（换工程、换分辨率时）：页和小贴图都放掉。"""
        self.engine.layers.drop_layer(self.layer.uid)
        if self._preview is not None:
            self._preview.release()
            self._preview = None
        self.active = False


# ====================================================================== 选区边线（流动的虚线）
#: 这个像素和左右上下各隔半个线宽（u_ants.x 像素）的地方，选区值有没有跨过 0.5（选上一半）——跨过了就是边。
#: 不用屏幕导数找边：选区的边比一个像素还陡、又正好落在两组 2×2 像素之间时导数是 0，边会丢。
#: 黑白相间，每段 u_ants.y 像素，按 u_ants.z 往前走
ANTS_GLSL = """
uniform vec4 u_ants;
vec4 ants(sampler2D sel, vec2 uv, vec2 frag) {
  vec2 ex = dFdx(uv) * (u_ants.x * 0.5);
  vec2 ey = dFdy(uv) * (u_ants.x * 0.5);
  bool here = texture(sel, uv).r >= 0.5;
  bool edge = (texture(sel, uv + ex).r >= 0.5) != here || (texture(sel, uv - ex).r >= 0.5) != here
           || (texture(sel, uv + ey).r >= 0.5) != here || (texture(sel, uv - ey).r >= 0.5) != here;
  float stripe = step(0.5, fract((frag.x + frag.y) / (2.0 * u_ants.y) - u_ants.z));
  return vec4(vec3(stripe), edge ? 1.0 : 0.0);
}
"""

_ANTS_VERT = """#version 430
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv;
uniform mat4 u_view_proj; uniform mat4 u_model; uniform mat3 u_nmat;
out vec2 v_uv;
void main(){
  vec4 wp = u_model * vec4(in_pos, 1.0);
  v_uv = in_uv;
  gl_Position = u_view_proj * wp;
}
"""

_ANTS_FRAG = """#version 430
in vec2 v_uv; out vec4 o_color;
uniform sampler2D u_sel;
""" + ANTS_GLSL + """
void main(){
  vec4 a = ants(u_sel, v_uv, gl_FragCoord.xy);
  if (a.a <= 0.003) discard;
  o_color = a;
}
"""


def ants_values(prefs) -> tuple:
    """u_ants：(线宽, 段长, 走到哪了, 0)，按偏好设置。"""
    import time

    viewport = getattr(prefs, "viewport", None)
    width = float(getattr(viewport, "selection_width", 1.5))
    dash = float(getattr(viewport, "selection_dash", 4.0))
    phase = 0.0
    if getattr(viewport, "selection_animate", True):
        phase = (time.perf_counter() * float(getattr(viewport, "selection_speed", 1.5))) % 1024.0
    return (width, dash, phase, 0.0)


class AntsPass:
    """三维视口：有选区的纹理集，模型再画一遍（贴着原来的深度），按选区小贴图画边线。"""

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_ANTS_VERT, fragment_shader=_ANTS_FRAG)
        self._vaos: dict = {}

    def _vao(self, mesh, material: int):
        from .meshgpu import mesh_layout

        ibo = mesh.material_ibo[material] if material < len(mesh.material_ibo) else None
        if ibo is None:
            return None
        key = (id(mesh), material)
        vao = self._vaos.get(key)
        if vao is None:
            vao = self.ctx.vertex_array(self.program, [mesh_layout(self.program, mesh.vbo)], index_buffer=ibo,
                                        index_element_size=4)
            self._vaos[key] = vao
        return vao

    def forget_mesh(self, mesh) -> None:
        for key in [k for k in self._vaos if k[0] == id(mesh)]:
            self._vaos.pop(key).release()

    def draw(self, engine, renderer, view) -> None:
        from .renderer import set_model

        targets = []
        for mesh, material, display, ts, _index in renderer.scene:
            if ts is None or display is None or id(mesh) in renderer.hidden or renderer._skip(view, mesh):
                continue
            state = engine.sets.get(ts.uid)
            selection = getattr(state, "selection", None)
            texture = selection.preview() if selection is not None else None
            if texture is not None:
                targets.append((mesh, material, texture))
        if not targets:
            return
        ctx = self.ctx
        program = self.program
        fbo = ctx.fbo
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.depth_func = "<="
        ctx.polygon_offset = (-1.0, -2.0)
        fbo.depth_mask = False
        try:
            program["u_view_proj"].write(view.camera.view_projection(view.aspect).astype("f4").T.tobytes())
            program["u_ants"] = ants_values(engine.prefs)
            program["u_sel"] = 0
            for mesh, material, texture in targets:
                vao = self._vao(mesh, material)
                if vao is None:
                    continue
                texture.use(0)
                set_model(program, mesh)
                vao.render(moderngl.TRIANGLES)
        finally:
            fbo.depth_mask = True
            ctx.polygon_offset = (0.0, 0.0)
            ctx.depth_func = "<"
            ctx.disable(moderngl.BLEND)

    def release(self) -> None:
        for vao in self._vaos.values():
            vao.release()
        self._vaos.clear()
        self.program.release()
