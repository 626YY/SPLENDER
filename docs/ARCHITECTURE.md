# SPLENDER 代码约定

给所有参与开发的人和 AI。产品方案见 `docs/SPLENDER_PLAN_v2.md`，先读它。本文是模块之间的接口约定，改接口要同步改这里。

## 0. 硬规则

1. **不许抢前台。** 用户可能正在用这台电脑。任何测试不得在桌面上弹出可见窗口，不得发送键鼠输入。
   - 界面测试：设环境变量 `QT_QPA_PLATFORM=offscreen`，用 `widget.grab().save(...)` 截图。
   - 显卡测试：用 `moderngl.create_standalone_context(require=430)`，它没有可见窗口。
   - 整机自检 `python -m splender --selftest tests_new/selftest_basic.py` 会开真实窗口，但放在屏幕外、不激活、不接收焦点，可以用。除此之外不要启动带真实窗口的程序。
2. **写代码用 Write / Edit 工具。** 本机 Bash 的 heredoc 会吃掉反斜杠，用它写代码会坏。
3. **只动自己负责的文件。** 不改别人负责的模块，需要别人模块改动时写进交付报告。旧原型整体放在 `legacy/prototype-2026-10-03/`，不要动，也不要 import。
4. **不装全局包。** 需要新依赖时装到 `runtime/site-packages`：`python -m pip install --target F:/SPLENDER/runtime/site-packages --no-deps <包>`，并在报告里写明。
5. **语言。** 界面文字、注释、文档用简体中文。标识符用英文。界面文字只说这东西是什么、能做什么，不写实现细节和免责声明。
6. **可调。** 数值、开关、行为选项一律声明成属性（见 2.1），不要写死。硬编码只留防崩溃的夹紧。
7. **坐标。** 世界坐标 Y 轴向上，右手系。UV 原点在左下角。贴图纹素 (x, y) = (u·分辨率, v·分辨率)。
8. **Python 3.11，PySide6 6.11，numpy 2.4，numba 0.67，scipy，zstandard，lz4，Pillow，OpenCV 都已装好。** ModernGL 5.12 和 OpenEXR 在 `runtime/site-packages`，import 前调用 `splender.paths.ensure_vendor_path()`。
9. **测试放 `tests_new/`**，文件名 `test_*.py`，用 unittest 写（本机没装 pytest）。`python tests_new/run_all.py` 跑全部。截图和测量结果放 `docs/evidence/<模块名>/`。
10. 运行命令时工作目录是 `F:/SPLENDER`，路径用正斜杠。控制台是 GBK 编码，打印中文前设 `PYTHONIOENCODING=utf-8`。

## 1. 目录

```
splender/
  __main__.py app.py paths.py log.py
  core/      props ops keymap context registry prefs history signals   （已完成，无界面依赖）
  doc/       project（已完成） meshio storage
  ui/        theme icons（已完成） widgets layout panels editor area window wm workspaces
  editors/   viewport3d image_editor layers_editor properties_editor assets_editor info_editor prefs_editor
  engine/    glx pagepool cache pageops stamp layerstore display renderer view2d camera meshgpu meshprep
             ibl projectio export_png engine
  ops/       各类操作
  tools/     绘制工具
  resources/ icons hdri matcap studio logo
runtime/site-packages/   随程序携带的第三方库
tools/                   构建与资源转换脚本
tests_new/               测试（run_all.py 跑全部）
legacy/                  旧原型，只读
docs/                    方案、约定、证据
```

## 2. 内核（已完成，直接用）

### 2.1 属性 `splender/core/props.py`

`PropertyGroup` 子类上声明 `FloatProperty / IntProperty / BoolProperty / EnumProperty / StringProperty / ColorProperty / FloatVectorProperty / PointerProperty`。赋值自动夹紧，变化时发 `group.changed(name)`，嵌套组向上发 `"子组.属性"`。`to_dict()/from_dict()` 存取，`reset(name)`，`is_default(name)`，`Group.prop(name)` 取描述符（含 label、description、min、max、soft_min、soft_max、step、precision、unit、subtype、items()）。

subtype 约定：`FACTOR`（0..1 滑杆）、`PIXEL`、`ANGLE`、`DIR_PATH`、`FILE_PATH`。

### 2.2 操作 `splender/core/ops.py`

`@ops.register class X(Operator)`：`idname`、`label`、`description`、`poll(ctx)`、`execute(ctx)`、`invoke(ctx, event)`、`modal(ctx, event)`。`ops.call(idname, ctx, event=None, **props)`，`ops.poll(idname, ctx)`。返回 `FINISHED / CANCELLED / RUNNING_MODAL / PASS_THROUGH`。返回 `RUNNING_MODAL` 时窗口管理器把后续事件先交给它的 `modal`。

### 2.3 键位 `splender/core/keymap.py`

`Event`（type、value、ctrl、shift、alt、x、y、pressure、is_tablet、wheel…），`KeyConfig.lookup(event, [键位表名…])`，`default_keyconfig()`。键名：字母 `"A"`，数字 `"0"`，`"F3"`，`LEFTMOUSE / MIDDLEMOUSE / RIGHTMOUSE / MOUSEMOVE / WHEELUPMOUSE / WHEELDOWNMOUSE`，`SPACE ESC RET TAB DEL HOME`，`NUMPAD_0..9 NUMPAD_PERIOD NUMPAD_PLUS NUMPAD_MINUS`，`LEFT_BRACKET RIGHT_BRACKET COMMA PERIOD`。

### 2.4 上下文、注册表、历史、偏好

- `core/context.py`：`Context(app, wm, window, screen, area, editor, region, event)`，派生 `prefs / project / engine / history / tool_settings / keyconfig / texture_set / layer / space_type`。
- `core/registry.py`：`register_editor / editors()`，`Panel` + `register_panel / panels(space, region, category)`，`Tool` + `register_tool / tools(space)`，`Menu` + `register_menu / menu(idname)`。
- `core/history.py`：`History.push(label, undo, redo, nbytes, dispose)`，`undo() / redo()`，`changed` 信号。
- `core/prefs.py`：`Preferences`，分组 `interface / navigation / memory / paint / viewport / files`。

### 2.5 文档 `splender/doc/project.py`

`Project`（`objects`、`texture_sets`、`active_texture_set`、`changed` 信号），`TextureSet`（`layers` 自下而上、`active_layer`、`add_layer / remove_layer / move_layer / set_active`、`layers_changed(kind, layer)` 信号），`Layer`，`Brush`，`ToolSettings`，`ViewShading`，`MeshObject`。

### 2.6 应用对象

`app`（`splender/app.py` 里的 `Application` 单例）持有：`prefs`、`keyconfig`、`project`、`history`、`tool_settings`、`engine`（可能为 None）、`wm`。界面和操作都通过 `ctx.app` 拿它们。测试里可以用一个只带这些属性的简单对象代替。

## 3. 界面库

### 3.1 主题与图标（已完成）

- `ui/theme.py`：`theme.color(name)`、`theme.qcolor(name, alpha)`、`theme.px(n)`（乘界面缩放）、`theme.size(name)`、`theme.font(kind, bold)`、`theme.apply(app, ui_scale)`、`theme.changed` 信号。色名和尺寸名见文件里的 `DARK`、`SIZES`。**不要在别处写十六进制颜色和像素常量。**
- `ui/icons.py`：`icons.icon(name, color=None, size=None) -> QIcon`，`icons.pixmap(name, color, size, dpr)`。name 用 `ALIASES` 里的语义名或 Lucide 文件名。缺图标时往 `tools/copy_icons.py` 的名单里加，重新运行。

### 3.2 控件 `ui/widgets.py`

全部跟随主题和界面缩放。

| 类 | 说明 |
|---|---|
| `NumberField(value=0, minimum, maximum, soft_min, soft_max, step, precision, unit="", label="", slider=False, integer=False)` | Blender 式数值框。横向拖动改值，Shift 精调，Ctrl 取整步；单击进入文字编辑，回车确认，Esc 取消；双击全选。`slider=True` 时按软范围画填充条。label 画在框内左侧。信号 `valueChanged(float)`、`editingStarted()`、`editingFinished()`。方法 `value()`、`setValue(v, emit=False)`、`setRange(...)`。 |
| `Toggle(text="", icon=None)` | 开关按钮，信号 `toggled(bool)`。 |
| `EnumField(items)` | 下拉。items 是 `[(id, 文字, 说明, 图标)]`。信号 `currentChanged(id)`，`setCurrent(id)`、`current()`。 |
| `SegmentedControl(items, icon_only=False)` | 一排互斥按钮。接口同上。 |
| `ColorButton(color)` | 色块，点开 `ColorPicker` 浮层。信号 `colorChanged(tuple)`，`setColor`、`color()`。颜色是 0..1 的 sRGB 三元组。 |
| `ColorPicker` | 色相环加明度饱和度方块，十六进制输入，RGB 与 HSV 数值，最近用过的颜色。可以直接嵌进面板。信号同上。 |
| `IconButton(icon, tooltip="", checkable=False, size=None)` | 纯图标按钮。 |
| `Foldout(title, icon=None, closed=False)` | 可折叠面板。`body` 是内容容器，`header_layout` 可在标题行右侧放小控件。信号 `toggled(bool)`。 |
| `TextField`、`SearchField` | 文本输入。 |
| `Popover(anchor_widget)` | 浮层，点外面关闭。`body` 是内容容器，`popup()` 显示。 |
| `Separator(vertical=False)`、`Label(text, role=None)` | role: `dim`、`faint`、`title`。 |
| `TabStrip(items, vertical=False, icon_only=False)` | 标签条。 |
| `build_menu(menu_cls_or_idname, ctx, parent) -> QMenu` | 把 `registry.Menu` 变成 QMenu，操作项右侧显示当前快捷键，不可用的置灰。 |

### 3.3 布局语句 `ui/layout.py`

面板、标题栏、菜单的内容都用它排。

```python
layout = UILayout(parent_widget, ctx, direction="COLUMN")   # 把自己装进 parent_widget
layout.row(align=False) / layout.column(align=False) / layout.box() -> UILayout
layout.label(text, icon=None, role=None)
layout.prop(data, name, text=None, icon=None, slider=None, expand=False, toggle=False, icon_only=False) -> QWidget
layout.operator(idname, text=None, icon=None, role=None, **props) -> QWidget
layout.menu(menu_idname, text=None, icon=None) -> QWidget
layout.popover(text, icon, draw_fn) -> QWidget          # draw_fn(layout, ctx) 填充浮层
layout.separator(factor=1.0)
layout.widget(w, stretch=0) -> w                        # 放进自定义控件
layout.stretch()
layout.enabled = False                                  # 置灰其后加入的内容
layout.use_property_split = True                        # 标签在左、控件在右；面板里默认开，标题栏里默认关
layout.clear()
```

`prop` 的规则：
- 按属性类型自动选控件：Float/Int 用 `NumberField`（`subtype="FACTOR"` 或 `slider=True` 时带填充条），Bool 用勾选框（`toggle=True` 用开关按钮），Enum 用下拉（`expand=True` 用分段按钮），Color 用色块，String 用文本框（`DIR_PATH / FILE_PATH` 带浏览按钮）。
- 双向绑定：控件改数据，数据的 `changed(name)` 改控件，不能互相死循环。控件销毁后不再响应。
- 提示文字取属性的 description。
- 右键菜单：恢复默认值、复制数值、粘贴数值。
- 撤销：如果 `getattr(data, "undoable", False)` 为真且 `ctx.history` 存在，一次编辑（拖动从开始到结束算一次）结束后向历史推一步，撤销时把值写回去。
- `operator`：按钮文字默认取操作的 label，提示取 description 加当前快捷键。创建时和每次 `wm.notify` 后重新判断 `poll`，不可用就置灰。

### 3.4 面板宿主 `ui/panels.py`

`PanelHost(ctx_provider, space, region, category=None)`：一个可滚动的控件，把 `registry.panels(space, region, category)` 里通过 `poll` 的面板依次画成 `Foldout`。`refresh()` 重新判断 poll 并重画。记住每个面板的折叠状态（`state() / set_state()`）。`ctx_provider` 是返回 `Context` 的函数。

### 3.5 编辑器基类 `ui/editor.py`

```python
class Editor(QWidget):
    idname = ""; label = ""; icon = ""; category = "通用"; order = 100
    has_toolbar = False          # 左侧工具栏（registry.tools(idname)）
    has_sidebar = False          # 右侧侧栏（registry.panels(idname, "SIDEBAR") 按 category 分标签）
    keymaps: list[str] = []      # 本编辑器启用的键位表名，靠前的优先

    def __init__(self, app, area): ...
    def build_main(self) -> QWidget          # 子类实现：主区域控件
    def draw_header(self, layout, ctx)       # 子类实现：标题栏里编辑器类型按钮右边的内容
    def context(self, event=None) -> Context
    def handle_event(self, event) -> bool    # 按 活动工具键位表 → keymaps → "Screen" → "Window" 查操作并执行
    def active_keymaps(self) -> list[str]    # 默认返回 keymaps；子类可把当前工具的键位表插到前面
    def save_state(self) -> dict ; def load_state(self, state)
    def refresh_header() ; def refresh()     # 数据变了重画
    def on_show() ; def on_hide()
    toolbar_visible / sidebar_visible        # 可读写，状态随 save_state 保存
```

基类负责搭出四块：标题栏在上，下面是 工具栏 | 主区域 | 侧栏。侧栏宽度可拖。工具栏按钮由注册的工具生成，点击后把 `app.tool_settings.tool` 设为该工具的 idname。

### 3.6 区域与屏幕 `ui/area.py`

- `AreaWidget`：一个区域。标题栏最左是编辑器类型按钮（图标加下拉，列出 `registry.editors()` 按 category 分组），之后是编辑器自己的标题栏内容。`set_editor(idname)`，`editor` 属性，`to_dict() / from_dict()`。换编辑器类型时保留各类型上次的状态。
- `Screen`：一个工作区的区域拆分树。
  - 序列化：`{"type":"split","orientation":"H"|"V","ratio":0.7,"a":{…},"b":{…}}` 或 `{"type":"area","editor":"VIEW_3D","state":{…}}`。H 表示左右排列。
  - `areas()`、`area_at(global_pos)`、`split_area(area, orientation, ratio=0.5) -> 新区域`、`join_areas(keep, remove)`、`swap_areas(a, b)`、`maximize(area) / restore()`、`find_editor(idname)`、`to_dict() / from_dict()`。
  - 交互：分隔缝可拖；区域四角是热区，往里拖是拆分（方向取拖动的主方向），往相邻区域拖是合并（两个区域必须共用一整条边），按住 Ctrl 拖到别的区域是交换；拖动中画出预览。标题栏右键菜单：垂直拆分、水平拆分、最大化、在新窗口打开、关闭区域。
  - 区域最小尺寸要有下限，不能拖没。

### 3.7 窗口与窗口管理 `ui/window.py`、`ui/wm.py`、`ui/workspaces.py`

- `MainWindow`：顶栏（程序菜单：文件、编辑、窗口、帮助，由 `registry.menu("TOPBAR_MT_file")` 等生成；工作区标签，可切换、双击改名、右键复制与删除、加号新建；右侧显示工程名）。中间是当前工作区的 `Screen`。底部状态栏（左：操作提示；中：后台任务进度；右：显存、内存占用文字，由 `set_stats(text)` 更新）。
- `WindowManager`：
  - `windows`，`main_window`，`active_screen()`。
  - 事件路由：程序级事件过滤器截获按键，找到鼠标所在区域，交给该编辑器的 `handle_event`。焦点在文本输入控件里时，不处理不带 Ctrl/Alt 的单键。
  - 模态操作栈：`modal_push(op, ctx)`，模态期间所有输入事件先给栈顶操作的 `modal(ctx, event)`；返回 `FINISHED / CANCELLED` 时出栈。
  - `report(text, level)`：状态栏显示几秒并写日志。
  - `notify(tag)`：数据变了，通知各编辑器 `refresh()`。同一轮事件循环内的多次通知合并成一次。
  - `context(area=None, event=None) -> Context`。
  - `popout(area)`：把区域复制到新的顶层窗口。
- `workspaces.py`：默认工作区（绘制、材质、UV）的布局树，和存进 `prefs.workspaces` 的读写。编辑器 idname：`VIEW_3D`、`IMAGE_EDITOR`、`LAYERS`、`PROPERTIES`、`ASSETS`、`INFO`、`PREFERENCES`。布局里引用了没注册的编辑器时，用一个显示「这种编辑器不可用」的空编辑器顶上，不能崩。
- 屏幕操作 `ops/screen_ops.py`：`screen.area_maximize`、`screen.area_split`（属性 orientation）、`screen.area_close`、`screen.area_popout`、`screen.region_toggle`（属性 region：TOOLBAR / SIDEBAR）、`screen.workspace_cycle`（属性 direction）、`screen.workspace_add`、`wm.search`（F3 操作搜索浮层）。

## 4. 模型数据

### 4.1 `splender/doc/meshio.py`

```python
@dataclass
class MeshData:
    name: str
    positions: np.ndarray     # (T*3, 3) float32，逐角点，不共享
    normals: np.ndarray       # (T*3, 3) float32，单位长度
    uvs: np.ndarray           # (T*3, 2) float32
    material_ids: np.ndarray  # (T,) int32
    materials: list[str]
    bounds_min: np.ndarray    # (3,) float32
    bounds_max: np.ndarray
    source_path: str = ""
    has_uvs: bool = True
    warnings: list[str] = field(default_factory=list)
    @property
    def triangle_count(self) -> int

def load_mesh(path) -> MeshData            # 按扩展名分发：.obj .gltf .glb
def make_test_mesh(kind, **kw) -> MeshData # "sphere" "cube" "plane" "torus"，带合理 UV，供测试和默认场景
```

要求：OBJ 一百万三角形在几秒内读完；多边形扇形三角化；负索引；`usemtl` 分材质；没有法线时算平滑法线（按面夹角加权）；没有 UV 时 `has_uvs=False` 并给出警告；坏面（退化、越界索引）跳过并计数进 warnings。文件坐标按 Y 轴向上直接用。

### 4.2 `splender/engine/meshprep.py`

```python
@dataclass
class SetGeometry:
    material_index: int
    resolution: int
    grid: int                    # resolution // 256
    indices: np.ndarray          # (M,) uint32，按格排序的角点索引（指向 MeshData 的逐角数组）
    tile_first: np.ndarray       # (grid*grid,) int32，格 ty*grid+tx 在 indices 里的起点
    tile_count: np.ndarray       # (grid*grid,) int32，索引个数
    tile_min: np.ndarray         # (grid*grid, 3) float32，该格内表面的三维包围盒；空格填 +inf
    tile_max: np.ndarray         # 空格填 -inf
    skirt_vertices: np.ndarray   # (K, 8) float32：u, v, px, py, pz, nx, ny, nz。三角形列表，已按格排序
    skirt_first: np.ndarray      # (grid*grid,) int32，顶点起点
    skirt_count: np.ndarray      # (grid*grid,) int32，顶点个数
    overlap_ratio: float         # UV 重叠面积占比的估计
    uv_out_of_range: bool

def build_set_geometry(mesh: MeshData, material_index: int, resolution: int, skirt_texels: float = 2.0) -> SetGeometry
```

- 一个三角形和哪些 256 纹素格相交，要做精确的三角形与矩形相交判断，不能只用 UV 包围盒。
- `tile_min / tile_max` 是「三角形落在该格内的那一部分」的三维包围盒的并集：把三角形在 UV 里裁到格子矩形，裁出的多边形顶点用重心坐标换回三维。
- 扩边：UV 空间里只属于一个三角形的边是边界边。每条边界边向岛外挤出 `skirt_texels / resolution` 宽的四边形，位置和法线取边上端点的值。四边形按它覆盖的格子分桶，跨格的在每个格里各放一份。
- UV 超出 [0,1] 时取小数部分并置 `uv_out_of_range`。
- 用 numba 加速，`cache=True`。一百万三角形、16K 分辨率要在 5 秒内。

## 5. 工程文件 `splender/doc/storage.py`

```python
PageKey = tuple[int, int, int, int, int]      # (layer_uid, plane, mip, tx, ty)

class ProjectFile:
    def __init__(self, path, create=False)
    def close()
    def read_meta() -> dict ; def write_meta(meta: dict)
    def write_asset(name: str, data: bytes) ; def read_asset(name) -> bytes | None ; def list_assets() -> list[str]
    def page_keys() -> list[PageKey]
    def read_page(key) -> bytes | None            # 解压后的原始字节
    def read_pages(keys) -> dict[PageKey, bytes]
    def write_pages(items, delete=())             # items: 可迭代的 (key, 原始字节)。一次事务，内部多线程压缩
    def delete_layer(layer_uid)
    def stats() -> dict
```

- `.splender` 是单个 SQLite 文件。写入用事务，断电不坏文件。关闭后目录里只剩这一个文件。
- 读页面可以在后台线程里调用。
- 压缩用 zstd，级别可调，默认 3。

## 6. 导出 `splender/engine/export_png.py`

```python
class PngStreamWriter:
    def __init__(self, path, width, height, channels, bit_depth=8, compress_level=3, threads=0)
    def write_rows(self, rows: np.ndarray)     # (n, width, channels)，uint8 或 uint16，自上而下
    def close()                                 # 先写临时文件，成功后原子替换
    def abort()
```

逐条带写，不把整张图放进内存。多线程压缩。16K RGB 8 位导出目标在 10 秒以内。

## 7. 环境光 `splender/engine/ibl.py` 与资源

- `tools/convert_assets.py`：把 Blender 发行包里的 CC0 资源转成程序自带格式，放进 `splender/resources/`：
  - `hdri/<名>.npz`（键 `rgb`，float16，形状 H×W×3，线性）加缩略图 `hdri/<名>.png`（256×128）。
  - `matcap/<名>.npz` 加缩略图 `matcap/<名>.png`（128×128）。
  - `studio/<名>.json`（工作室光的各盏灯）。
  - 许可说明文件一并复制。
- `ibl.list_hdris()`、`ibl.load_hdri(名或文件路径) -> float32 (H, W, 3)`（文件支持 `.hdr`、`.exr`、`.npz`），`ibl.thumbnail(名) -> Path`，matcap 和工作室光同理。
- `class Environment(ctx)`：`set_image(rgb)` 生成三样东西：背景贴图（等距柱状，带多级渐远）、高光预过滤贴图（等距柱状，每一级对应一个粗糙度，GGX 重要性采样）、漫反射球谐系数 `sh`（9×3）。`bind(program, unit_base)` 把它们绑给着色器。`release()`。
- 提供三段 GLSL 字符串，供渲染器拼进着色器：
  - `GLSL_ENV`：`vec3 env_diffuse(vec3 n)`、`vec3 env_specular(vec3 r, float roughness)`、`vec3 env_background(vec3 dir, float blur)`。uniform 名固定：`u_env_spec`、`u_env_bg`、`u_env_sh[9]`、`u_env_rotation`（绕 Y 轴，弧度）、`u_env_strength`、`u_env_levels`。
  - `GLSL_PBR`：`vec3 shade_pbr(vec3 n, vec3 v, vec3 base_color, float metallic, float roughness, float ao)`，金属度工作流，split-sum 近似。
  - `GLSL_TONEMAP`：`vec3 tonemap(vec3 linear_rgb, int transform, float exposure_ev)`，返回可直接显示的 sRGB 编码值。transform：0 标准，1 Filmic，2 AgX。

## 8. 引擎门面 `splender/engine/engine.py`

界面和工具只和 `Engine` 打交道。除撤销重做的回调外，所有方法都要求引擎的显卡上下文是当前上下文（先调用 `host.make_current()`）。

### 8.1 接口

| 接口 | 作用 |
|---|---|
| `set_project(project)` | 换工程：清空场景，按纹理集和模型重建 |
| `create_view(shading)`、`create_view2d()`、`destroy_view(view)`、`frame_view(view)` | 三维视口、UV 视图 |
| `hover(view, x, y, brush)`、`pick(view, x, y)`、`sample_color(view, x, y)` | 笔刷光标、拾取表面、取色 |
| `stroke_begin(view, x, y, pressure, when, tool_settings, erase)` | 落笔，返回是否成功；失败原因在 `stroke_reject`，界面直接显示 |
| `stroke_move(...)`、`stroke_end()`、`stroke_cancel()` | 笔划过程 |
| `complete_strokes(include_active=False)` | 把等着并入图层的笔划同步做完 |
| `frame()` | 推进一帧，返回重新渲染过的视口 |
| `needs_frame`、`needs_render` | 还有没有工作（含后台备份）；画面还会不会变 |
| `settle(max_frames)` | 推进到画面不再变化（测试、截图、导出前用） |
| `save_project`、`load_project`、`export_channel`、`close_storage` | 工程存取与导出（内部会先做完未完成的合并） |
| `detach_layer_pixels`、`attach_layer_pixels`、`free_layer_pixels`、`duplicate_layer_pixels` | 图层像素整体操作，给图层的撤销步骤用 |
| `stats()` | 性能面板的数据 |

### 8.2 一帧的顺序

1. 读页面需求图（异步读回，Fence 完成才取），把攒下的输入变成笔触。这两步要读显卡结果，放在最前面：这时显卡上还没有本帧的新工作，几乎不用等。
2. `cache.begin()`：收回读回、压缩、读盘的结果；上传到货的页，限时（正在画时 1.5 ms，平时 4 ms）；纯色页直接在显卡上填色。
3. 盖章、生成笔划的各级缩小页、预取笔划下面图层的旧页；推进合并队列，限时（4 ms，正在画下一笔时 2.5 ms）。
4. 合成看得见的脏页（每帧最多 1024 页），渲染变了的视口。
5. `cache.maintain()`：派发纯色检测、保持空闲水位、内存超预算时落盘、后台备份、空闲时提前给页池加数组。这些都会让显卡读回，排在本帧渲染之后。

### 8.3 抬笔合并

- 抬笔后笔划进入 `merges` 队列。`MergeJob`（`layerstore.py`）按格分块、每帧限时推进：新页先记在 `staged` 里，页表不动，显示仍是「旧图层 + 笔划叠加」。全部做完后 `commit()` 一次把页表指向新页，并记一步撤销。显示贴图不重新合成：叠加显示和并入后的结果相差不超过 2 个色阶。
- 合成器最多同时叠加 `MAX_OVERLAYS = 4` 笔（还在合并的几笔加正在画的一笔），超出时最早的一笔同步做完。
- `History.barrier = complete_strokes`：撤销、重做、记任何新步骤之前，先把队列做完，保证页表只被一方修改。
- 合并需要的旧页在落笔过程中就预取（`_prefetch_layer`），新页放不下时等下一帧，连续 20 帧拿不到资源才改为同步完成一小块。

### 8.4 页面缓存 `cache.py`

- 页面写好后不再修改，改动产生新页。副本有三种：`blob`（内存，LZ4 压缩或纯色值）、`scratch`（暂存文件）、`disk_key`（工程文件）。`pool.clean[slot]` 表示有副本，换出时直接释放槽位。
- 后台备份：把没有副本的页用一次计算着色器打包读回（`PageOps.pack`，64 页一次调度，主线程开销约 0.07 ms），每 16 页交给一个线程压缩。正在画时每帧 4 MB，空闲时每帧 128 MB。
- 纯色检测：合并生成的新页排队检查（两遍并行着色器）。纯色页只存一个纹素，换出不用读回，换入在显卡上填色。大笔刷内部的页几乎都是纯色。
- 换入：`request()` 排队，后台线程解压、读暂存文件或读工程文件，`begin()` 里限时上传。`try_resident()` 不等待；`make_resident()` 同步，只给必须马上完成的地方。
- 分配：`try_new_page(s)` 不等待，只释放已有副本的页；`new_page` 必须成功，实在没有已备份的页可放时才同步读回，计入 `blocking_evictions`（正常为 0）。
- 内存副本超过预算：已有别处副本的直接放掉，其余最早的写进暂存文件（关闭时删除）。
- `set_page` 换下来的旧页时间戳清零，优先换出；换上来的页记为刚用过。

### 8.5 撤销记录

一步撤销是若干 `RecordGroup`（同一张页表里若干格的旧页和新页，按块存，对象少）。占用按「页数 × 内存副本平均大小」估算，超过撤销预算时丢最早的步骤。

### 8.6 垃圾回收 `core/gcpolicy.py`

全量回收不自动触发（几十万个页面对象，一次要停一百多毫秒），应用在用户停手 `memory.gc_idle_seconds` 秒、画面也静止时再做。新代码少建长期存活的小对象，页面相关的类一律用 `__slots__`。

### 8.7 可调项

都在偏好设置里，界面自动生成：
- 内存：图层显存预算、显示缓存预算、内存缓存预算、撤销预算与步数、显存空闲水位、每帧换入时间上限（正在画时、平时）、后台备份量（正在画时、空闲时）、暂存目录、停手多久后整理内存。
- 绘制：抬笔合并每帧用时（平时、边画边合并）、每帧页数、扩边宽度、压感曲线、笔刷直径上限。

### 8.8 测量工具

- `tools/bench_paint.py`：16K 四通道流畅度跑分，结果写 `docs/evidence/bench/`。`--set=组.属性=值` 改偏好做对照，`tools/bench_summary.py` 打印对照表。
- `tools/profile_paint_phases.py`、`tools/profile_merge_phases.py`：按阶段拆开每帧的主线程和显卡时间。
- `tools/bench_pageops.py`、`tools/bench_pack.py`、`tools/bench_transfer.py`：单个着色器和传输方式的耗时。

## 9. 插件与注册 `splender/core/addons.py`、`splender/core/registry.py`

- 编辑器、面板、操作、工具、菜单、键位、渲染引擎都是注册制，插件用同样的接口往里加东西，停用时注销干净。
- 插件是一个 Python 包或单个文件，模块里有 `ADDON_INFO`（名字、作者、版本、分类、说明、`default_enabled`）和 `register()`、`unregister()`；可以带一个 `PREFERENCES` 属性组，设置页自动生成。
- 内置插件在 `splender/addons/`，用户插件在用户目录的 `addons/`。启用状态存在偏好设置里，启动时按状态加载，出错的插件不影响程序启动，错误显示在插件页。
- 菜单扩展：`registry.menu_append(menu_id, fn)`、`menu_prepend`、`menu_remove`。
- 面板可以写 `mode = "PAINT"` 或 `"SCULPT"`，只在对应模式显示。

## 10. 渲染 `splender/render/`

- `scene.py`：渲染只读一份场景快照 `RenderScene`（网格、合成好的贴图、相机、环境），不碰图层引擎，所以引擎可以在后台线程跑。
- `engines.py`：`RenderEngine` 基类（`idname`、`label`、`use_preview`、`use_final`、`settings_type`），`RenderJob`（逐帧推进）和 `ThreadedRenderJob`（后台线程），`ViewportRender`（视口里的渐进渲染）。
- 内置引擎：`WORKBENCH`、`REALTIME`（视口的 PBR 渲染器超采样）、`PATHTRACE`（`render/pathtracer/`：numba 建 SAH BVH，计算着色器求交和着色，环境重要性采样 + 多重重要性采样，按时间预算分块派发）。
- 插件引擎：`splender/addons/render_arnold/`。进程内加载本机 Arnold 的 Python 接口，材质转 standard_surface，环境贴图转 skydome（方向校准 `SKY_ALIGN = π/2`）。
- 出图设置 `RenderProps` 跟着工程保存，每个引擎的设置各占一组。F12 出图，结果在「渲染结果」编辑器，可存 PNG、EXR。

## 11. 雕刻 `splender/sculpt/`

- `topology.py`：索引网格 `SculptMesh`（顶点、三角形、逐角点 UV、材质），焊接与拆开、法线、邻接、Morton 空间排序、Loop 细分、非流形边拆分（`split_nonmanifold`）、默认底模。
- `engine.py` + `shaders.py`：显卡雕刻引擎。顶点按空间排好、每 256 个一块；一笔的流程是 剔除块 → 间接派发 → 求笔刷平面 → 作用 → 重算法线 → 更新包围盒，全部计算着色器。撤销按块写时复制。
- `session.py`：雕刻会话。进入时把物体的网格交给雕刻引擎，渲染器隐藏原网格、改画雕刻网格；笔划按间距变成笔触；换拓扑（细分、体素重构）各记一步撤销；退出时形状写回物体，整段合成一步「雕刻」。
- `remesh.py`：体素重构。8³ 的稀疏小块只覆盖表面附近；距离按块多线程算；内外用三个方向的射线绕数投票（共边的格点用一致的平局规则只算一次）；Surface Nets 出面；之后补洞、拆非流形边、去碎块、切向放松。
- 模式：`ToolSettings.mode` 是 `PAINT` 或 `SCULPT`，工具栏和「工具」页的面板按模式显示。雕刻工作区自动进入雕刻模式。

## 12. 几何工具 `splender/geometry/`

- `unwrap.py`：自动展开 UV。法线角度长块 → 小块并入邻块 → 沿平均法线投影 → 凸包求最小包围矩形 → 天际线排布（几种排序取最好）。不同材质分开排。所有循环都是 numba。
- 操作 `mesh.uv_unwrap`（属性编辑器「模型」页）。重构过的模型回到绘制模式时自动展开（`prefs.sculpt.auto_unwrap`），展开后的状态并进同一步「雕刻」撤销。

## 13. 烘焙与生成器 `splender/bake/`

- `curvature.py`：逐角点曲率。位置完全相同的角点按位模式哈希合成顶点，边曲率 `(n_j - n_i)·(p_j - p_i)/|p_j - p_i|²` 取平均，平滑后按面积加权 95% 分位数归一。函数都不开 numba 多线程（在后台线程里跑）。
- `baker.py`：`BakeJob`（后台线程建 BVH 和算曲率 → 每块最大 2048² 在 UV 空间光栅化 → 计算着色器发射线 → 写三张贴图 → 跳跃泛洪扩边 → mip）和 `MeshMapSet`（a：遮蔽/曲率/厚度/覆盖，n：法线，p：位置；`to_bytes` 用 zstd，存在工程资源 `meshmaps/<纹理集 uid>.bin`）。求交用 `render/pathtracer/shaders.py` 的 `BVH_GLSL`，和路径追踪同一份。
- 引擎：`engine.meshmaps`、`bake_start(ts)`、`bake_cancel()`、`set_meshmap()`；每帧 `_step_bake`（正在画或雕刻时最多 3 毫秒）；`rebuild_object` 会取消烘焙、把相关贴图标成过期。
- 合成器：`set_maps(maps, ao_mix)`；着色器每个纹素按 `u_level_texels` 换算 UV 采一次模型贴图；图层参数多了 `gen`、`gen2`、`gen3`，`flags.w` 是生成器编号（`doc.project.GENERATOR_CODE`）。`compose_to` 要传 `level_texels`。
- `smart.py`：智能材质是数据（每层是图层属性的值）。加材质的操作 `layer.add_smart_material`，整组一步撤销。
- 高模（`MeshMapSettings.high_poly`）：准备阶段读高模、建高模的 BVH；光栅化之后多一步 `_PROJECT`：从低模表面沿法线找高模，写切线空间法线（贴图 t），并把位置、法线换成高模上的，后面的遮蔽、厚度从高模表面发射线。切线空间约定：T 是 dP/du 去掉法线分量，B = N×T 翻到和 dP/dv 同侧；视口（`engine/renderer.py`）、路径追踪（`render/pathtracer`）、Arnold（`normal_map` 节点）都按这个约定解。
- 部件（贴图 i，R32F，-1 是空白）：准备阶段 `parts.triangle_parts` 按网格部件或 UV 岛给三角形编号（焊接位置、并查集），光栅化时作为顶点属性写进去；按高模部件时投影那一步写高模三角形的编号。部件号单独一趟计算着色器写（收尾着色器的图像单元已经用满）。
- `parts.py` 还有点选用的 `ray_hit`（numba 暴力求交）；引擎 `part_at(view, x, y)` 由相机射线找到三角形和 UV，再读 ID 贴图。

## 14. 贴图搬迁 `splender/engine/transfer.py`

重构或重新展开 UV 后把旧 UV 上的像素搬到新 UV（`engine.transfer_layers`）。新 UV 光栅化（主体加扩边，深度优先）得到每个纹素的位置、法线；`_MAP` 计算着色器沿 ±法线对旧模型的 BVH 求交，记录命中三角形和重心坐标，再按旧 UV 双线性取旧图层的页。每个工作组先在共享内存里合并包围盒，再写「要读哪些旧页」的位图，避免原子操作争用。UV 没变的三角形走直接对应的快路径。新的图层存储替换旧的，旧的由撤销步骤保管（`swap_layer_stores`）。

## 15. 图层文件夹和调整层

- 数据：`Layer.kind` 有 PAINT、FILL、FOLDER、ADJUST；`parent_uid` 是所在文件夹；列表自下而上、文件夹的内容紧挨在它下面（见 `TextureSet` 的注释）。结构改动用 `ts.structure()` / `set_structure()` 整体快照撤销（`ops/layer_ops._record`）。
- 合成：图层参数 `flags2.x` 是深度。有文件夹时着色器用累加器栈（深度 `MAX_FOLDER_DEPTH + 1`），没有时 `u_flat = 1` 走寄存器快路径。调整层在 sRGB 里改它下面已经叠好的结果。

## 16. 降噪 `splender/render/denoise.py`

Intel Open Image Denoise 2.5（`pyoidn`，放在 `runtime/site-packages`）。路径追踪多写反照率、法线两个辅助通道；出图结束降噪一次，视口在 `DENOISE_STEPS` 的采样数节点异步降噪（`DenoiseTask` 在后台线程）。设备可选显卡或处理器，没装时自动跳过。

## 17. 导出 `splender/engine/projectio.py`、`export_normal.py`

- 导出设置在 `TextureSet.export`（`ExportSettings`）：勾选通道、尺寸、文件名规则（`{工程}`、`{纹理集}`、`{通道}`）、法线的位深、强度、是否含高模细节。`app.export_textures` 按它导出，同名时自动补上纹理集和通道。
- `compose_row` 合成一行格子（页面没进显存就等）；`export_channel` 逐行合成、逐行写 PNG。
- 法线：高度逐行合成进三张轮换的条带（上、中、下），一趟着色器求坡度 `(-dh/du·s, -dh/dv·s, 1)`，再和烘焙法线按 whiteout 方式叠加；坡度系数 `s = 0.05·半径·高度强度 / (每纹素对应的表面长度)`，和视口的凹凸一致；DirectX 翻转绿色。

## 18. 智能遮罩、部件生成器

- `bake/smart.py` 的 `SMART_MASKS`：一套生成器设置，`layer.apply_smart_mask` 套到当前图层，一步撤销。
- 「部件」生成器（`GENERATOR_CODE["PARTS"] = 8`）：`Layer.gen_parts` 是选中的部件号（逗号分隔）。引擎为每套贴图建一张 `MAX_PART_IDS × MAX_LAYERS` 的 R8 选择表（`_selection_texture`，改了才传），合成器按 (部件号, 图层序号) 查表。点选是模态操作 `layer.pick_parts`。

## 19. UV 视图里画

笔触参数 `params.z = 1` 表示 UV 空间：中心是 UV、半径按 UV（屏幕像素 / 缩放换算）。`SetGPU.tiles_for_uv_dabs` 按 UV 圆的外接方框找有几何的格子；盖章着色器按纹素自己的 UV 量距离，不做朝向判断。其余（笔划页、合并、撤销、叠加显示）和三维视口完全一样。绘制操作 `paint.stroke` 在 UV 编辑器里也能用（`_paint_editor`）。

## 20. 节点系统 `splender/nodes/`

- `graph.py`：`NodeGraph`（节点字典、连线 `(上游, 下游, 输入序号)`、当前节点、`settings` 里的名字和分辨率），每个节点一个输出；不许成环。`changed(kind, node)`：structure、params、layout、active、settings。撤销用整图快照（`record`）。工程里 `project.graphs`、`project.active_graph_uid`，`graphs_changed` 信号。
- `library.py`：节点类型（显示名、分类、输入及默认值、参数属性组、`vec4 node(vec2 uv)` 着色器）。`glsl.py`：可平铺的哈希、梯度噪波、细胞、砖块；参数按 `p_名字` 自动成为 uniform。
- `evaluate.py`：`GraphEvaluator`，每个节点一张 RGBA16F 贴图（带多级、平铺取样），按签名（类型、参数、分辨率、输入的签名）缓存；材质输出用计算着色器写两层贴图数组（基础色 sRGB；金属度、粗糙度、高度）。注意：设多级过滤要在生成多级之后，否则贴图「不完整」，图像写入会被忽略；一个着色器里两种采样器不能指向同一个贴图单元。
- 合成：填充层 `fill_graph` 指向节点图时，`flags2.y` 是这套贴图里的节点图序号加一，`graph` 参数是 (平铺次数, 横移, 纵移, 高度强度)；最多 `MAX_GRAPHS = 6` 张（贴图单元只剩这么多）。引擎 `_on_graph_changed` 让用到这张图的纹理集重新合成。
- 界面：`editors/node_editor.py`（QGraphicsView；缩略图由定时器按签名补画，每次最多几张）、`ops/node_ops.py`、键位表 Node、工作区「节点」。

## 21. 材质节点 `splender/shading/`

- `nodes.py`：节点类型（照 Blender 着色器节点）：`inputs` (名字, 显示名, FLOAT/COLOR/VECTOR/ANY, 默认值, 选项)、`outputs`、`params`、
  `emit(cx)` 生成 GLSL。注册时为每种节点生成数值组（`in_插口名` 加参数），节点上的控件、侧栏、撤销都绑它。颜色插口按线性算，
  界面上的颜色和色带按 sRGB 显示，进着色器前换成线性。「节点图纹理」要列出工程里的节点图：app 启动时 `set_project_getter`。
- `graph.py`：`ShaderGraph`（节点、连线 (上游, 插口, 下游, 插口)），`changed`：structure（要重新编译：增删节点、连线、静音、下拉/开关/
  整数参数）、values（只换数值）、layout、active。撤销用整图快照 `to_dict` / `load`；`load` 沿用编号和类型对得上的节点对象
  （界面控件和数值撤销步骤还指着它们）。`TextureSet.material` 存进工程；`set_material` / `ensure_material`，变化以
  `layers_changed("material" / "material_values")` 通知引擎。
- `compile.py`：只编译材质输出用到的节点，生成 `material_graph(inout base, metal, rough, height)`；没接线的输入、数值参数、色带
  （41 个 vec4 一块）都在存储缓冲 `mg[]`（binding 3），`Compiled.values()` 现算。插口间自动转换（颜色→数值取亮度，矢量→数值取平均）。
  静音节点每个输出取第一个同类型输入。`needs_maps` 为真时 app 自动烘焙模型贴图；没烘焙前位置类坐标先按 UV 算。
- `glsl.py`：`mg_` 开头的函数库（分形噪波、沃罗诺伊、波浪、渐变、棋盘格、砖块、混合模式、运算、矢量运算、映射范围、欧拉旋转、色带），
  插进合成器着色器 `main` 前；`main` 在图层叠完后调用 `material_graph`。
- 合成器 `Compositor.set_material(compiled, values)`：按材质代码缓存最多 8 个着色器变体（编译失败的记下来，退回图层堆栈），
  引擎 `_prepare_composite` 第一步调用；节点图纹理和填充层共用 6 个节点图贴图槽（`_material_graph_slot`）。
- 色带：`core/ramp.py`（值、插值算法、GLSL）、`RampProperty`（kind RAMP）、`ui/ramp_widget.py`（色标拖动、Ctrl+点击加、插值方式、翻转）。
- 界面：`editors/shader_editor.py`（QGraphicsView，节点内容是 QGraphicsProxyWidget 里的真控件；点在控件上把鼠标交给控件，
  其余是节点交互）、`ops/shader_ops.py`、`ui/menus_shader.py`、键位表 Shader Editor。

## 22. 调整层 `splender/engine/adjust.py`、`engine/cube.py`

- 种类照 Photoshop 的调整菜单（`doc/project.py` 的 `ADJUST_ITEMS`，顺序就是菜单顺序，编号 `ADJUST_CODE` 生成着色器里的
  `ADJ_名字` 宏，存工程只存名字）。参数都是 `Layer` 的 `adj_` 属性；可选颜色（九类 × 青洋红黄黑）、分通道色阶、分颜色的色相饱和度
  按表在类定义后面加（`_add_generated_props`）。非调整层 `to_dict` 不存 `adj` 开头的属性。
- 打包 `adjust.pack(layer)` → 每层 adj/adj2/adj3/adj4 四组数（adj3.w 是种类）+ 查找表；查找表放存储缓冲 `luts[]`（binding 4，
  每段 `LUT_SIZE`=1024 行 vec4，图层参数 `flags2.z` 是起始段）。第一段是按通道的函数：RGB 三列给颜色（亮度对比度、色阶、曲线、
  曝光度、反相、色调分离全靠它），第四列给金属度、粗糙度、高度（每种都有；只对颜色有意义的几种取「当灰色改完的明暗」）。
  后面接着各自的数据：渐变色带、六类颜色、九类颜色、颜色查找表（三维按 红最快 排，一维重采样成一段）。
- 着色器里只留跨通道的颜色计算（`adjust3`）：色相饱和度（含分颜色、着色）、阈值、渐变映射、可选颜色、颜色查找（三线性）、
  色彩平衡、自然饱和度、照片滤镜、黑白、通道混合器；`adjust1` 只查表。**按通道的函数和数值贴图的行为一律在 CPU 上生成查找表**：
  着色器里每种调整会被内联很多份（两条合成路径 × 颜色和三个数值），把它们写进着色器时编译从 1 秒涨到 49 秒。
- `adjust.apply_color` / `scalar_function`：和着色器一样算法的 numpy 实现（生成查找表、测试对拍 `test_random_parity_with_cpu`）。
- 直方图：`projectio.composite_image(engine, ts, 256, below=层)` 合成一张小图（只看这一层下面：顶层的截断图层数，在文件夹里的
  把它不透明度临时设 0），`adjust.histogram` 分 256 档。属性面板按 `content_version` 缓存（`ops/adjust_ops.layer_histogram`）。
  自动色调 / 对比度 / 颜色（`auto_levels`，两头各剪 0.1%）改当前色阶或曲线层，否则新建色阶层。
- `engine/cube.py`：.cube 读写（一维、三维、DOMAIN_MIN/MAX），存进图层的是压缩的半精度数据文本（`encode` / `decode`，解过的缓存），
  内置风格按公式现生成 33³。
- 界面：`ui/curve_widget.py`（曲线，背后画直方图）、`ui/levels_widget.py`（色阶三角拖动，松开记一步撤销）、
  预设 `ops/adjust_ops.py`（`PRESETS`，第一项「默认值」按 `TYPE_PROPS` 恢复默认；面板标题右边的按钮）。
  色带、曲线控件拖动时 `begin_ui_edit`，面板等松开再重画；当前色标 / 控制点存在 `ui/widget_state.py`，焦点由 `PanelHost`
  按控件的 `focus_key` 交给重建出来的新控件。

## 23. 滤镜 `splender/engine/filters.py`、`ops/filter_ops.py`

- `LayerFilter(engine, ts, layer, 名字, 参数, {页种类: (改第一组, 改第二组)})`：按 4×4 格一块，先把这一块加四周一圈（核能够到的范围）
  从页里取到临时贴图（预乘覆盖度、rgba16f），在临时贴图上跑几遍（高斯横竖、最小最大横竖、中值、动感、径向、表面模糊），
  再逐格写成新页（写回方式：直接用、USM、高反差、杂色、马赛克、云彩、浮雕）。核很大时在粗一级算（高斯 σ/2^k ≤ 12、
  动感半长 ≤ 48、中值半径 ≤ 4……），写回第 0 级时双线性放大。径向模糊每块按离中心最近、最远的位移各定一级。
- 哪些格要重写：会往外扩的滤镜把有内容的格向外扩（最多 6 格）；径向模糊按每格沿圆弧 / 半径方向会跑到哪里算；云彩整张。
- 新页交给 `FilterJob`（借 MergeJob 的逐级生成多级、记录、提交），引擎 `apply_filter` 标脏并 `_push_pixel_step` 记一步撤销。
- ★长的同步任务每做一块就 `cache.service(cache.frame + 1)` 推进一帧：新生成的页要过一帧才能备份、换出，
  否则 16K 整层（三种页一万多页）会把显存页池挤满（`_make_room` 只挑「至少一帧前」的页）。
- 着色器按引擎缓存（`engine.filter_programs`，引擎释放时一起释放）：取图、写回各四种页一个，跑遍的一个通用。
- 操作：每种滤镜一个 `layer.filter_*`（`redo = True`：视口左下角改参数、Shift+R 重复；Shift+R 的目标撤销后也保留），
  菜单 `LAYERS_MT_filter` 按 Photoshop 分组；画蒙版时（`ts.paint_target == "MASK"`）改蒙版页。

## 24. Photoshop 的工具 `ops/tool_ops.py`、`tools/paint_tools.py`、`engine/fill.py`、`engine/raster.py`

- 工具在三维视口（绘制模式）和 UV 视图各登记一份（`tools/paint_tools.py` 的表）。画笔、橡皮、效果笔刷走笔划（`paint.stroke`，
  `STROKE_TOOLS`）；别的工具有自己的键位表 `Paint Tool: 名字`（UV 视图的 `tool_keymap` 按自己那份工具查到同一张表）。
- 渐变、油漆桶、文字、形状都落到滤镜框架（`apply_filter`）：渐变是写回方式 S_GRADIENT（UV 空间按纹素算；模型上按位置贴图
  投到屏幕，用拖线时的视图矩阵）。在模型上做要用模型贴图，还没烘焙时 `_after_bake` 先开始烘焙、挂一个 `bake_finished`
  回调，烘好了按同样的参数自动再做一次（换了纹理集就不做）。
- 油漆桶先在 CPU 上做一张遮罩（`engine/fill.py`：相近颜色用 numba 扫描线洪泛，合成图直接要 8 位的；UV 岛 / 部件三角形栅格化），
  只传单通道遮罩加一种颜色（S_IMAGE 的 p0.y = 1，颜色是 u_c1）。UV 视图里的文字、形状用 `engine/raster.py`（Qt）画成 RGBA，
  S_IMAGE 按 UV 矩形盖上去；字号、描边、圆角按屏幕像素，换算成纹素（`px_scale`）。图太大时按 `MAX_SIDE`、`MAX_PIXELS` 画粗一点再放大。
- 三维视口里的文字、形状是投影（S_SCREEN_IMAGE）：图按屏幕矩形放，每个纹素按位置贴图投到屏幕，落在图里才盖。
  点下去那一刻 `capture_view` 把视口几何缓冲的位置附件拷一份留在显卡上（帧缓冲之间 blit；直接拷进贴图会被压成 0..1 的 8 位），
  只留最近一份，操作记它的编号（`capture`），调整上一步时还用它。遮挡（`occlude`）：取这个像素和周围一圈看到的表面里最远的深度
  （深度行：透视用 w，正交用 z 行），纹素不比它远 `u_eps.x + u_eps.y × w` 就算看得见，贴着看、轮廓边上也不漏。
  图在点击处按纹素密度（`_texel_density`，按点到的三角形算）画细一点（≤ `MAX_DETAIL`）。
  哪些格：先按三角形粗选（投到屏幕和矩形有重叠的三角形，它们的 UV 范围外扩补边宽，差分数组标格），再用 `_PROBE_SHADER`
  在显卡上逐纹素试一遍，只留真会变的格（背面、字缝里的格不重写，少占页）。投影、遮挡的 GLSL 是 `_PROJECT_GLSL`，写回和探测共用。
- 不取周围的滤镜（贴图、渐变、杂色、减淡……）按 `WIDE_TILE` × `WIDE_TILE` 格一块做，取周围的按 `TILE`。
- 效果笔刷（减淡、加深、海绵、模糊、锐化、涂抹、仿制、修复）：`StrokeParams.effect`。每帧把这一帧的笔触画到一张临时笔划页，
  滤镜按它当遮罩（`mask_store`，第 3 个存储缓冲）只改这几格并马上换上新页；同一笔里每帧的撤销记录合成一份
  （`ActiveStroke.absorb_effect` 把中间的页释放掉），抬笔记一步，取消就把原来的页放回去。效果笔刷不画颜色叠加。
  仿制、涂抹靠取样偏移（`offset`，临时贴图从挪开的地方取）；修复画笔取三张（来源原样、来源低频、落笔处低频），
  结果 = 来源 +（落笔处低频 − 来源低频）。按住 Ctrl 换成相反的效果（减淡↔加深、模糊↔锐化、海绵加色↔去色）。
- 工具操作的 invoke 在没有事件时（脚本、自检）直接按给的参数 execute；渐变、形状、文字、油漆桶都有 `redo = True`。

## 25. 选区 `engine/selection.py`、`ops/select_ops.py`

- 每套贴图一个 `Selection`（挂在 `SetState.selection`，第一次用时建）：一个不在图层列表里的隐藏图层，只有蒙版页（r8），
  缺页的格按 `default` 算。`active` 为假等于没有选区；取消选择只关 `active`，页留着给「重新选择」。
  全选 = 页全去掉、default 1；反选 = default 取反、已有的页取反（SEL_INVERT）；新选区 = 页全去掉、default 0 再盖形状。
  default 变了或去掉过页之后，上面各级按第 0 级和新 default 从头生成（`_rebuild_mips`）。
- 形状都是单通道遮罩（`raster.mask_shape`：矩形、椭圆、多边形，可羽化、消除锯齿）：UV 视图按 UV 矩形盖（IMAGE），
  三维视口按屏幕矩形投到模型上（SCREEN_IMAGE，用 `capture_view` 判断遮挡，没有模型贴图时 `_after_bake` 先烘焙）。
  合并：添加 = 盖白，减去 = 盖黑，交叉 = 相乘（p0.z = 3，形状外归 0；已有的页都过一遍，default 归 0）。
  魔棒、快速选择用和油漆桶一样的相近颜色洪泛（8 位合成图）；快速选择每一下只在笔刷附近一块里找，`begin_batch` 攒成一步撤销。
- 修改：羽化、扩展、收缩、平滑、边界 = 高斯模糊、最大值、最小值、中间值、SEL_BORDER（模糊后取 0.5 附近一圈），
  目标格是已有页往外扩到够得着的范围（default 是 1 时缺页的格也会变）。
- 每次改动记一步撤销：页的记录（按做的顺序，撤销时倒着放回）+ 前后的 (active, default)。改完什么都没选上就自动取消选择。
- 限制绘制：笔刷盖章（`stamp.py`）把每格的选区页槽号放在任务第 7 个数，覆盖度往选区值累积（a += (选区 − a) × 这一下），
  结果是「整笔覆盖度 × 选区」，实时样子和合并都只在选区里，选区外的格不开笔划页；效果笔刷的帧遮罩已经乘过选区，
  滤镜那边 `ignore_selection`。滤镜写回（`filters.py`）有 `sel_table`（第 4 个存储缓冲）和 r8 取页，
  最后 `blend_by(原值, 新值, 选区值)`；SEL_KEEP、SEL_CLEAR 直接按选区值改覆盖度（通过拷贝、清除）。
- 显示：`Selection.preview()` 把选区某一级读成一张不超过 2048² 的小贴图；三维视口在 `renderer.extra_draws` 里把模型
  贴着深度再画一遍（`AntsPass`），UV 视图在自己的着色器里画。边线 = 这个像素和左右上下各隔半个线宽处的选区值跨没跨过 0.5
  （不用屏幕导数：边比一个像素还陡时导数会是 0）。流动靠 app 的定时器（偏好设置：显示、流动、速度、帧率、线宽、段长）。
- 选区改了只重画视口（`selection_changed`）；界面在操作做完时刷新一次。
- 蒙版缺页的格按初始值：生成上面各级（`PageOps.downsample` 的 `empty`）时也按它，以前填 0，白蒙版拉远看会发黑。

