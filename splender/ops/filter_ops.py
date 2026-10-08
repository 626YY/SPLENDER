"""滤镜（照 Photoshop 的滤镜菜单）：直接改当前绘制层的像素；正在画蒙版时改蒙版。
做完可以在视口左下角的「调整上一步」里改参数（撤销后用新参数重做），Shift+R 再来一次。"""
from __future__ import annotations

from ..core import ops, registry
from ..core.ops import CANCELLED, FINISHED, Operator
from ..core.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from ..doc.project import PLANE_COLOR, PLANE_HEIGHT, PLANE_MASK, PLANE_MR


def _engine(ctx):
    host = getattr(ctx.app, "host", None)
    if host is None:
        return None
    host.make_current()
    return host.engine


class _FilterOp(Operator):
    redo = True
    filter_name = ""
    #: 滤镜自己的参数（属性名），按这个顺序传给引擎
    filter_props: tuple = ()
    #: 会不会在空白处生成内容（云彩）
    creates = False

    use_basecolor = BoolProperty("基础色", default=True, description="改不改这一层的基础色")
    use_metallic = BoolProperty("金属度", default=True, description="改不改这一层的金属度")
    use_roughness = BoolProperty("粗糙度", default=True, description="改不改这一层的粗糙度")
    use_height = BoolProperty("高度", default=True, description="改不改这一层的高度")

    @classmethod
    def poll(cls, ctx) -> bool:
        ts = getattr(ctx, "texture_set", None)
        layer = ts.active_layer if ts is not None else None
        if layer is None or layer.locked:
            return False
        return layer.kind == "PAINT" or (ts.paint_target == "MASK" and layer.has_mask)

    def _planes(self, ts, layer) -> dict:
        if ts.paint_target == "MASK" and layer.has_mask:
            return {PLANE_MASK: (True,)}
        planes = {}
        if self.use_basecolor:
            planes[PLANE_COLOR] = (True,)
        if self.use_metallic or self.use_roughness:
            planes[PLANE_MR] = (bool(self.use_metallic), bool(self.use_roughness))
        if self.use_height:
            planes[PLANE_HEIGHT] = (True,)
        return planes

    def params(self, ctx) -> dict:
        return {name: getattr(self, name) for name in self.filter_props}

    def draw_redo(self, layout) -> None:
        for name in self.filter_props:
            layout.prop(self, name)
        # 两个一排，窄的面板里字也放得下
        for first, second in (("use_basecolor", "use_metallic"), ("use_roughness", "use_height")):
            row = layout.row(align=True, heading="作用于" if first == "use_basecolor" else "")
            row.prop(self, first, toggle=True)
            row.prop(self, second, toggle=True)

    def execute(self, ctx) -> str:
        ts = ctx.texture_set
        layer = ts.active_layer
        engine = _engine(ctx)
        if engine is None:
            return CANCELLED
        planes = self._planes(ts, layer)
        if not planes:
            self.report(ctx, "没有选要改的通道", "WARNING")
            return CANCELLED
        store = engine.layers.stores.get(layer.uid)
        if not self.creates and (store is None or not any(p in store.planes and store.planes[p].page_count()
                                                           for p in planes)):
            self.report(ctx, "「%s」上还没有内容" % layer.name, "WARNING")
            return CANCELLED
        try:
            count = engine.apply_filter(ts, layer, self.filter_name, self.params(ctx), planes, self.label)
        except Exception as error:  # noqa: BLE001
            self.report(ctx, "滤镜没做成：%s" % error, "ERROR")
            return CANCELLED
        if count == 0:
            return CANCELLED
        ctx.app.notify("layers")
        return FINISHED


def _filter(idname: str, label: str, name: str, description: str, props: dict, creates: bool = False):
    attrs = {"idname": idname, "label": label, "description": description, "filter_name": name,
             "filter_props": tuple(props), "creates": creates, "icon": "filter"}
    attrs.update(props)
    return ops.register(type("Filter_" + name.title().replace("_", ""), (_FilterOp,), attrs))


class _CloudsOp(_FilterOp):
    def params(self, ctx) -> dict:
        values = super().params(ctx)
        brush = ctx.app.tool_settings.brush
        values["color1"] = (*brush.color[:3], 1.0)
        values["color2"] = (*brush.secondary_color[:3], 1.0)
        return values


GaussianBlur = _filter("layer.filter_gaussian_blur", "高斯模糊", "GAUSSIAN_BLUR", "按半径柔和地模糊", {
    "radius": FloatProperty("半径", default=2.0, min=0.1, max=250.0, soft_max=100.0, precision=1, unit="px",
                            description="模糊的范围，越大越糊"),
})
MotionBlur = _filter("layer.filter_motion_blur", "动感模糊", "MOTION_BLUR", "沿一个方向拖出残影，像快速移动", {
    "angle": FloatProperty("角度", default=0.0, min=-180.0, max=180.0, unit="°", precision=0,
                           description="朝哪个方向拖"),
    "distance": FloatProperty("距离", default=20.0, min=1.0, max=2000.0, soft_max=500.0, precision=0, unit="px",
                              description="拖多长"),
})
RadialBlur = _filter("layer.filter_radial_blur", "径向模糊", "RADIAL_BLUR", "绕一个中心旋转着糊，或朝中心缩放着糊", {
    "mode": EnumProperty("方法", items=[("SPIN", "旋转", "沿着绕中心的圆弧糊"), ("ZOOM", "缩放", "沿着指向中心的方向糊")],
                         default="SPIN"),
    "amount": FloatProperty("数量", default=10.0, min=1.0, max=100.0, precision=0,
                            description="旋转时是转过的角度，缩放时是朝中心拉近的百分比"),
    "center_u": FloatProperty("中心 U", default=0.5, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                              description="中心在贴图上的横向位置"),
    "center_v": FloatProperty("中心 V", default=0.5, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                              description="中心在贴图上的纵向位置"),
})
SurfaceBlur = _filter("layer.filter_surface_blur", "表面模糊", "SURFACE_BLUR", "把相近的颜色糊平，边缘保持清楚", {
    "radius": FloatProperty("半径", default=5.0, min=1.0, max=100.0, precision=0, unit="px", description="在多大范围里找相近的颜色"),
    "threshold": FloatProperty("阈值", default=0.1, min=0.01, max=1.0, subtype="FACTOR", precision=2,
                               description="差多少以内算相近，越大越糊"),
})
Sharpen = _filter("layer.filter_sharpen", "锐化", "SHARPEN", "让细节更清楚", {
    "amount": FloatProperty("数量", default=0.5, min=0.05, max=3.0, precision=2, description="锐化多强"),
})
UnsharpMask = _filter("layer.filter_unsharp_mask", "USM 锐化", "UNSHARP_MASK", "加强边缘两侧的反差，可以控制范围和阈值", {
    "amount": FloatProperty("数量", default=1.0, min=0.01, max=5.0, precision=2, description="锐化多强（1 是 100%）"),
    "radius": FloatProperty("半径", default=1.0, min=0.1, max=250.0, soft_max=50.0, precision=1, unit="px",
                            description="边缘两侧多宽的范围加强反差"),
    "threshold": FloatProperty("阈值", default=0.0, min=0.0, max=1.0, subtype="FACTOR", precision=3,
                               description="和周围差得比这小的地方不锐化（避免把噪点也锐化了）"),
})
HighPass = _filter("layer.filter_high_pass", "高反差保留", "HIGH_PASS", "只留下细节，大块的明暗变成中性灰", {
    "radius": FloatProperty("半径", default=10.0, min=0.1, max=250.0, soft_max=100.0, precision=1, unit="px",
                            description="比这更细的细节保留下来"),
})
AddNoise = _filter("layer.filter_add_noise", "添加杂色", "ADD_NOISE", "在画过的地方撒上随机的颗粒", {
    "amount": FloatProperty("数量", default=0.1, min=0.0, max=1.0, subtype="FACTOR", precision=3, description="颗粒多强"),
    "distribution": EnumProperty("分布", items=[("UNIFORM", "平均分布", "颗粒强弱平均"),
                                               ("GAUSSIAN", "高斯分布", "多数颗粒弱、少数很强，更像胶片")],
                                 default="UNIFORM"),
    "monochromatic": BoolProperty("单色", default=False, description="颗粒只有明暗、不带颜色"),
    "seed": IntProperty("随机种子", default=0, min=0, max=9999, description="换一个数，颗粒就换一种"),
})
Median = _filter("layer.filter_median", "中间值", "MEDIAN", "去掉零星的斑点：每处取周围排在中间的那个颜色", {
    "radius": FloatProperty("半径", default=2.0, min=1.0, max=64.0, precision=0, unit="px", description="在多大范围里取"),
})
Minimum = _filter("layer.filter_minimum", "最小值", "MINIMUM", "暗的地方往外扩、亮的地方收缩", {
    "radius": FloatProperty("半径", default=1.0, min=1.0, max=250.0, soft_max=50.0, precision=0, unit="px",
                            description="扩多宽"),
})
Maximum = _filter("layer.filter_maximum", "最大值", "MAXIMUM", "亮的地方往外扩、暗的地方收缩", {
    "radius": FloatProperty("半径", default=1.0, min=1.0, max=250.0, soft_max=50.0, precision=0, unit="px",
                            description="扩多宽"),
})
Emboss = _filter("layer.filter_emboss", "浮雕效果", "EMBOSS", "变成灰色的浮雕，边缘凸起", {
    "angle": FloatProperty("角度", default=135.0, min=-180.0, max=180.0, unit="°", precision=0, description="光从哪个方向来"),
    "height": FloatProperty("高度", default=3.0, min=1.0, max=100.0, precision=0, unit="px", description="凸起有多宽"),
    "amount": FloatProperty("数量", default=1.0, min=0.01, max=5.0, precision=2, description="凸起多明显"),
})
Mosaic = _filter("layer.filter_mosaic", "马赛克", "MOSAIC", "变成一格一格的色块", {
    "cell": FloatProperty("单元格大小", default=16.0, min=2.0, max=512.0, precision=0, unit="px", description="一格多大"),
})
Clouds = ops.register(type("Filter_Clouds", (_CloudsOp,), {
    "idname": "layer.filter_clouds", "label": "云彩", "filter_name": "CLOUDS", "creates": True, "icon": "filter",
    "description": "整层铺上笔刷颜色和备用颜色之间的云雾纹理",
    "filter_props": ("scale", "contrast", "height", "seed"),
    # 和 Photoshop 一样先只铺颜色；要连金属度、粗糙度、高度一起铺就在「作用于」里打开
    "use_metallic": BoolProperty("金属度", default=False, description="改不改这一层的金属度"),
    "use_roughness": BoolProperty("粗糙度", default=False, description="改不改这一层的粗糙度"),
    "use_height": BoolProperty("高度", default=False, description="改不改这一层的高度"),
    "scale": FloatProperty("大小", default=256.0, min=8.0, max=8192.0, soft_max=2048.0, precision=0, unit="px",
                           description="云团有多大"),
    "contrast": FloatProperty("对比度", default=1.0, min=0.1, max=5.0, precision=2, description="云和空隙分得多开"),
    "height": FloatProperty("高度幅度", default=0.5, min=0.0, max=1.0, subtype="FACTOR", precision=2,
                            description="改高度时起伏有多大"),
    "seed": IntProperty("随机种子", default=0, min=0, max=9999, description="换一个数，云就换一种"),
}))


# ---------------------------------------------------------------- 菜单（照 Photoshop 的滤镜菜单分组）
def _menu(idname: str, label: str, items: list):
    def draw(self, layout, ctx) -> None:
        for op in items:
            layout.operator(op.idname, text=op.label)

    return registry.register_menu(type("Menu_" + idname, (registry.Menu,), {"idname": idname, "label": label,
                                                                             "draw": draw}))


_menu("LAYERS_MT_filter_blur", "模糊", [GaussianBlur, MotionBlur, RadialBlur, SurfaceBlur])
_menu("LAYERS_MT_filter_sharpen", "锐化", [Sharpen, UnsharpMask])
_menu("LAYERS_MT_filter_noise", "杂色", [AddNoise, Median])
_menu("LAYERS_MT_filter_stylize", "风格化", [Emboss])
_menu("LAYERS_MT_filter_pixelate", "像素化", [Mosaic])
_menu("LAYERS_MT_filter_render", "渲染", [Clouds])
_menu("LAYERS_MT_filter_other", "其它", [HighPass, Minimum, Maximum])


@registry.register_menu
class FilterMenu(registry.Menu):
    idname = "LAYERS_MT_filter"
    label = "滤镜"

    def draw(self, layout, ctx) -> None:
        for idname in ("LAYERS_MT_filter_blur", "LAYERS_MT_filter_sharpen", "LAYERS_MT_filter_noise",
                       "LAYERS_MT_filter_stylize", "LAYERS_MT_filter_pixelate", "LAYERS_MT_filter_render",
                       "LAYERS_MT_filter_other"):
            layout.menu(idname)
