# Surface API v1 — CPU 通用模型表面查询

此模块已实现可执行 CPU 能力，不依赖 sphere、Qt、OpenGL、engine 或 storage。不启动 GPU，不分配 16K 纹理，不打开 OBJ 引用的材质/纹理文件。后续由主开发接入渲染、异步任务和 UI。

模块的公共入口是 `surface_core/__init__.py` 中的 `__all__`；其余带下划线的方法不是兼容性承诺。首版拓扑为三角网格，求值阶段只允许改变坐标和已有法线表，不能改变连接关系。

## 最小接入

在仓库根目录运行：

```python
from surface_core import (
    load_obj, EvaluatedMesh, MeshTopology, MeshBVH,
    SurfaceObject, SurfaceScene, Ray,
)

result = load_obj("assets/surface-tests/multi_uv.obj",
                  mesh_id="paint-mesh", authoring_version=1)
authoring = result.mesh
topology = MeshTopology.build(authoring)
evaluated = EvaluatedMesh(authoring, evaluated_version=1)
bvh = MeshBVH(evaluated, topology)
obj = SurfaceObject("paint-object", bvh, transform_version=1)

hit = obj.raycast(Ray((0.2, 0.3, 1.0), (0, 0, -1)), cull="back")
if hit is not None and hit.uv is not None:
    # texel routing: (hit.texture_set, hit.chart_id, hit.uv)
    # 保留 primitive_id 和 barycentrics，用于后续表面运动。
    print(hit.object_id, hit.primitive_id, hit.uv)

crossing = topology.map_edge(0, 1, 0.25)
mirror = obj.mirrored_hit(hit, axis="x", offset=0.5, max_distance=1e-6)
blocked = SurfaceScene([obj]).occluded((0.2, 0.3, 1), (0.2, 0.3, -1))
```

完整可执行例程：`python -B -m surface_core.example`。输出可直接 JSON 序列化的命中、跨缝与镜像结果。该例程仅使用仓库内自制合成模型。

## 数据和版本契约

| 类型 / 字段 | 含义 |
| --- | --- |
| `MeshData.positions` | `float64[V,3]`，对象空间位置；导入不改变坐标轴、单位或绕序 |
| `triangles` | `int32[F,3]`，逐角位置索引，零起始 |
| `uvs / normals` | `float64[T,2] / float64[N,3]` 独立属性表 |
| `corner_uvs / corner_normals` | `int32[F,3]`，逐角属性索引；缺失为 `-1` |
| `texture_sets / texture_set_names` | 每三角形的整数集 ID 与名字；导入时按 `usemtl` 分组，默认空名字为 ID 0 |
| `source_faces / face_info` | 三角形到源多边形的映射；源 face 保留 object 名、group 名、smoothing group、源文件行号 |
| `material_libraries` | 原始 mtllib 名称字符串，仅供显示/后续受控解析 |
| `mesh_id / authoring_version` | 调用者分配的资产身份与作者数据版本 |
| `EvaluatedMesh` | 指向作者快照，保存独立位置快照、`evaluated_version` 和可选法线值 |
| `SurfaceObject` | 一个查询实例，含 `object_id`、BVH、4×4 仿射矩阵及独立 `transform_version` |

所有 MeshData、EvaluatedMesh、MeshTopology、MeshBVH、SurfaceObject 快照及其数组均不可变。输入数组会复制为不可写的 bytes backing；不变求值位置共享作者只读数组。查询临时状态保存在调用栈内，可让只读 worker 共享快照，但 Python 标量查询不承诺线程并行提速。

三个版本必须由调用者维护，不能在内容变化后重用同一身份/版本组合。它们不是内容哈希，也没有全局版本注册器。`mirrored_hit` 检查命中的 object/mesh/三个版本，拒绝过期命中；其他消费命中的上层代码应做同样检查。

变形更新示例：

```python
positions = authoring.positions.copy()
positions[:, 2] += 0.02
next_eval = EvaluatedMesh(authoring, positions, evaluated_version=2)
next_bvh = MeshBVH(next_eval, topology)
next_obj = SurfaceObject(obj.object_id, next_bvh,
                         obj.object_to_world, obj.transform_version)
# 主线程在任务仍对应当前版本时发布 next_obj；旧 obj 可继续读。
```

只改对象变换时复用 BVH，构造新的 SurfaceObject 并增加 transform_version。变形要重建 BVH；首版没有 refit。改变拓扑/UV/材质时创建新的 MeshData 和 MeshTopology，并增加 authoring_version。

提供新求值位置后，不默认沿用旧法线，以免置换后出现陈旧 shading normal。未提供 `normal_values` 时采用几何平面法线；提供时必须保持作者 normal table 的形状。保留的 smoothing group 不会自动生成平滑法线。

## OBJ 明确支持范围

`load_obj(path, limits=None, strict=True, mesh_id=None, authoring_version=0)` 和 `parse_obj(text_or_bytes, ...)` 返回 `ObjImport(mesh, warnings)`。读取本地单个 UTF-8 / UTF-8 BOM 文件，支持注释、CRLF、空行。

支持 `v x y z [1]`、`vt u v [0]`、`vn x y z`、`f v` / `v/vt` / `v//vn` / `v/vt/vn`，包括逐角混合缺失、正索引和相对当前已定义表的负索引。正索引的前向引用拒绝。零索引、越界、非有限数值、零法线、非法数字、错误角数等抛明确异常。UV 原值保留，不 clamp、不翻 V、不 wrap。

支持 `o/g/s/usemtl/mtllib` 基本元数据。一个 OBJ 返回一个 MeshData；`o` 标签不会自动生成多个 SurfaceObject。MTL 不解析，usemtl 仅是 texture-set 路由占位，不表示已载入真实贴图，也不代表 UDIM/多通道材质实现。

三角形保持原绕序。简单、平面的凹/凸多边形用确定性耳切三角化，保留 source face 映射，但不保留原始 n-gon 角循环以供无损导出。多边形洞、自交、自接触、严重退化、重复邻接点和非平面多边形拒绝；非平面判定为面内局部尺度归一化后离面误差大于 `1e-7`。极近共线多边形可能明确拒绝，应由作者软件先三角化。

退化/近退化的原始三角形保留并产生摘要 warning；拓扑和 BVH 排除满足 `|cross(e1,e2)| <= 1e-14 * max(|e1|²,|e2|²)` 的三角形。求值后重新判断几何退化，作者拓扑仍固定。浮点极端尺度/巨大动态范围不承诺精确几何谓词。

不支持曲线、NURBS、点/线元素、自由形状面、顶点色、非单位 homogeneous w、3D UV、续行语法、动画、骨骼、多个 UV 通道。默认 strict=True 对未知 record 抛 ObjParseError。strict=False 只跳过未知 record 并给摘要 warning，已知但非法的数据仍失败，不能用于掩盖导入问题。

## 查询与 SurfaceHit

`Ray(origin, direction, t_min=0, t_max=inf)` 使用世界坐标，方向内部归一化；区间两端包含。t/distance 始终是世界距离，对非均匀缩放也成立。零方向、非法区间和非有限向量拒绝。没有自动 self-hit 偏移，调用者应根据模型尺度设置 t_min。

| 方法 | 行为 |
| --- | --- |
| `obj.raycast(ray, cull="none")` | 最近 SurfaceHit 或 None |
| `obj.raycast_all(ray, cull="none", max_hits=64)` | 按距离、primitive ID 排序；超限抛 ResourceLimitError，不返回截断结果 |
| `obj.intersects(ray, cull="none")` | 首个合格命中即返回布尔值 |
| `scene.raycast(ray, cull="none")` | 多对象最近命中；等距按 object_id、primitive_id 稳定选择 |
| `scene.occluded(start, end, bias=1e-7, cull="none")` | 有限世界线段遮挡检测，两端各减去 bias |
| `obj.nearest_object(point, max_distance=...)` | 对象空间欧氏最近表面；有限搜索半径必须明确提供 |
| `obj.to_world_point / to_object_point` | 3D 点的仿射转换 |

cull 可为 none/back/front。默认双面查询，不能用 backface culling 替代遮挡。负 determinant 变换按真实世界绕序翻转几何法线与正背面判断。法线使用逆转置，再与世界几何法线同半球；不会为了朝向相机而自动翻面。

SurfaceHit 字段包括：object_id、mesh_id、authoring_version、evaluated_version、transform_version、primitive_id、source_face_id、barycentrics、uv、chart_id、texture_set、texture_set_name、position、object_position、normal、geometric_normal、tangent、bitangent、tangent_handedness、uv_frame_valid、distance、front_facing。

- primitive_id 稳定对应该 MeshData 三角形行；重心权重按该三角形三个角顺序。
- position/normal/geometric_normal/tangent/bitangent 均为世界空间；object_position 是求值对象位置。
- 任何一角缺 UV 时 uv=None、chart_id=-1；可以拾取但调用者不可直接进行 UV 写入。
- tangent/bitangent/normal 正交归一。bitangent = handedness × cross(normal,tangent)。镜像 UV 保留负 handedness。缺失/退化 UV 仍给稳定几何切线架，但 uv_frame_valid=False。
- 不存在 incident ray 的 nearest/symmetry 命中，distance/front_facing=None；不能把对象搜索半径误认为世界射线距离。
- raycast_all 按“每三角形命中”返回，共享边可能出现同位置双命中，重叠/薄片不会按距离容差合并。没有物体 inside/outside 分类或自动命中去重。

BVH 是 float64、按中心最长轴的中位数分裂、默认每叶 8 面；AABB 边界向外一 ULP，三角形使用共享边一致的剪切坐标边函数。实现未做全范围精确算术/watertight 认证。首版没有 SAH、GPU 查询、batch SIMD 接口或顶层场景 BVH；SurfaceScene 对少量对象线性遍历。

矩阵必须是有限仿射 4×4，最后一行精确 [0,0,0,1]。线性部分奇异值允许范围 [1e-12,1e12]，条件数最多 1e12，拒绝奇异/病态变换。

## 邻接、UV 岛与跨缝

`MeshTopology.build(mesh, uv_tolerance=1e-10)` 仅按共用**位置索引**建立真实连接，不以 UV 重叠连接，不对相同坐标但不同索引做自动焊接。

每三角形三条有向边定义为角 e → (e+1)%3。`neighbors[F,3]`：非负为唯一邻接面，-1 为边界，-2 为非流形多入射边，-3 为排除的退化面。`neighbor_edges` 给对侧边。`winding_conflicts` 标记同向共享边；不自动修正绕序。

`vertex_faces(v)` 为去重的相邻有效作者三角形。`edge_incidents(f,e)` 返回全部 halfedge ID（face*3+edge），可检查非流形情况。`chart_ids` 是通过同材质集、UV 端点在绝对容差内连续的流形边连接得到的岛；完整 UV 且有效的孤立面仍有独立 chart。UV 数值相同但索引不同可连续；不相连的重叠 UV 永不合并。

`topology.map_edge(primitive, edge, t)` 在 t∈[0,1] 时返回 EdgeTransition，含对侧面/边、参数、重心权重、两侧 UV、目标 chart/texture set、seam 与边反向标记。它映射同一个几何边上点，两侧 UV 可以截然不同。边界/非流形/退化返回 None，不能猜测穿过哪个面。部分缺 UV 时边端点有属性仍可给边 UV，但整个面命中仍遵循完整角属性要求。

这是供未来 surface liquify 使用的基础邻接与边界映射；不包含曲面测地线步进、切向向量跨边输运、刷子半径传播、UV 冲突写入策略或液化求解器。

## 对象空间对称

`obj.mirrored_hit(hit, axis="x", offset=0, max_distance=1e-5, normal_cos_min=None)` 将真实求值对象位置对 axis=offset 平面反射，再通过同一 BVH 找最近实际三角形，返回目标面自己的 UV/岛/材质集。不会做 1-u/1-v。

max_distance 为对象单位；超出半径返回 None。normal_cos_min 可选地检查镜像法线与最近候选法线夹角；若不合格返回 None，不去猜更远薄片。模型未必精确对称，重叠层/相交面仍可能歧义；等距按 primitive ID 选择。首版不建立拓扑对称配对，不保证同岛/同材质，也不自动排除中心平面重复笔触。

## 资源和异常

默认 ResourceLimits：64 MiB 文件、64 KiB 单行、250000 位置、750000 UV、750000 法线、500000 源 face、500000 三角形、每 face 256 角、4096 材质名、65536 个元数据 record、8 MiB 元数据文本，以及 512 MiB 估算工作预算。各个上限需要同时满足，不能把名义三角形上限理解为所有输入都能导入。

导入过程中在增长前检查数量，二进制 readline 限制单行。按 `256*V + 96*(UV+N) + 192*source_faces + 1536*triangles` 估算联合工作量，拓扑/BVH 构建也检查。此估算用于保守准入，**不是 OS 强制 RSS 上限**；Python/NumPy、调用者已有字符串/数组、多个并存版本和其他子系统须由宿主另计。三角形数量不能靠分配 16K CPU 图像来替代。本模块没有全画幅图像缓冲。

`ObjParseError` 携带 line，`ResourceLimitError` 表示可配置限额；二者继承 `SurfaceError(ValueError)`。`load_obj` 的路径/权限错误保留标准 OSError。没有局部成功网格返回；主开发应在 worker 捕获这些错误、取消发布，保留旧场景。

## 验证和实际基准

执行：

```text
python -B -m unittest discover -s tests -p "test_surface_*.py" -v
python -B -m surface_core.example
python -B -m surface_core.benchmark --grids 64 128 256 --rays 512 --output assets/surface-tests/benchmark.json
```

最终测试与环境记录见 `assets/surface-tests/validation.json`、`test-results.txt`；量级实测原始数据见 `benchmark.json`。测试仅导入 CPU 模块，涵盖错误 OBJ、资源边界、凹多边形、逐角属性、退化面、边邻接、seam、重叠 UV、非流形、前后薄片、凹面遮挡、独立 Möller–Trumbore 暴力参考对照、共享边、世界/对象变换、版本失效和对象空间镜像。合成资产均为本任务自制，没有读取用户作品。

最近量级基准 UTC：2026-10-03T08:22:25.880503+00:00。Windows 10、16 逻辑 CPU、Python 3.11.9、NumPy 2.4.4。

| 三角形 | OBJ 导入 s | 拓扑 s | BVH s | 命中 p50 / p95 ms | 进程峰值 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.357 | 0.175 | 0.042 | 0.277 / 0.336 | 39.3 |
| 32,768 | 1.462 | 0.707 | 0.176 | 0.295 / 0.498 | 52.1 |
| 131,072 | 6.029 | 2.769 | 0.754 | 0.307 / 0.376 | 110.7 |

最终自动测试：51 项，0 failures，0 errors，0 skipped；本次实测 0.323 秒。API 例程成功执行并保存 JSON 输出。

每规模单独启动 CPU 进程；512 条射线中 256 命中、256 在根 AABB 处 miss，16 次预热；单次 wall-clock 测量，没有独占机器保证。峰值工作集包括解释器、NumPy、自制 OBJ 文本生成和构建过程。它不是全系统 RAM，也不是纯网格数组占用。基准为平面网格，并非复杂凹模型最坏情形、16K 绘制、液化或雕刻吞吐保证。

## 依赖、来源与范围

生产模块仅使用 Python 标准库和宿主已有 NumPy；本次未添加或安装依赖。实测 Python 3.11.9 / NumPy 2.4.4。NumPy 核心为 BSD-3-Clause，当前安装包元数据完整表达式为 `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`，随包许可证应在分发时保留。基准可选读取已有 psutil 7.2.2（BSD-3-Clause）以记录进程内存；缺失时内存字段为 null，不影响查询模块。

导入器、耳切、BVH、三角形求交、最近点和测试均为本任务编写；没有复制 GPL 代码或打包 DCC 源码，没有外部下载的模型。通用几何算法名称不代表兼容任何 DCC 的全部行为。本次代码适用宿主项目原有授权安排，未擅自为项目新增许可证。

未实现完整液化、雕刻、置换求值器、FBX/glTF、MTL/纹理读取、UDIM、mesh 修复、相机解投影、导入 UI、GPU 渲染接线或数据持久化；这些是后续集成工作。



## Surface liquify v1（独立新增 CPU 数学基础）

新增入口为 `surface_core.liquify`，没有修改已有 `surface_core.__init__`、OBJ/import、query 或 topology 语义。实现的是**有限局部表面的纹理坐标搬运**，不改变顶点位置，不是完整 PS 液化，也不是任意曲率网格上的通用液化求解器。

新增文件：

- `liquify_geometry.py`：SurfaceRef、受限等距展开、重心定位、切向向量搬运、受影响 UV 包围盒。
- `liquify.py`：冻结源身份、推/拉参数、冻结区、对称、逆向原图采样地址、事务与参数历史。
- `liquify_demo.py`：自制双 UV 岛、近邻背面薄片与 60° 折面构造，以及实际 CPU 棋盘示例。
- `tests/test_surface_liquify.py`：液化专项验收。

### 接入与源版本

```python
from surface_core.liquify import (
    FrozenSource, LiquifySession, LiquifyState, SurfaceRef,
    Symmetry, dirty_regions,
)

# surface_object 是现有不可变 SurfaceObject。
# 此 token 必须绑定宿主真正持有的原始多通道像素快照。
source = FrozenSource(surface_object, material_revision="material-snapshot:42")
session = LiquifySession(source)

start = source.ref(start_hit)       # 检查 hit 的 object/mesh/三个版本
end = source.ref(end_hit)
before = session.state
session.begin()
session.push(start, end, radius=0.2, strength=1.0)
preview = session.state             # 不可变，可交给 worker
preview_epoch = session.epoch       # 发布前必须仍匹配会话 epoch
session.commit()                   # 或 cancel() 精确返回笔划前参数
regions = dirty_regions(before, session.state)
```

FrozenSource 强引用已有不可变几何快照，`stamp` 包含 object_id、mesh_id、authoring/evaluated/transform version 和 material_revision。SurfaceRef 只含 primitive_id 和 barycentrics，作用域是该 FrozenSource，不是跨模型全局引用。直接构造 SurfaceRef 的宿主必须保证来源一致；建议由 source.ref(hit) 创建。

**材质 token 不是像素副本。**本层不知道 engine/storage 的纹理存储，不能替宿主自动锁定原图。集成层必须为该 token 持有原始多通道只读快照/lease，直到相关参数、预览与历史都释放；sampler 对失效或不匹配 token 必须失败，不能悄悄改读当前合成结果。这是接入时的必要契约，而非已完成 GPU/存储接线的宣称。

相机从未进入字段或运算。只要对应同一物理表面点，换相机不会改变映射。对象变换版本属于身份契约：新 transform_version 需要宿主显式重新绑定/重建相应参数身份；不会自动 rebase。纹理、网格或求值版本变化同理，旧源仍由原快照解释。

### 场表示、推拉与切向搬运

每个 PushOperation 保存一个有界 DevelopedPatch、局部二维位移 delta、当时的 FreezeMask、位移梯度上界和限制标记。不保存绘制后的像素，也不保存 16K 位移图。

展开从笔触中心三角形的对象空间切平面开始，仅沿 MeshTopology 的唯一流形邻接展开。跨边保留边长和三角形度量，面内三角形 ID/重心坐标与局部二维坐标互换。每面的 tangent_frames 是二维场到对象空间切向量的映射；`to_chart_vector` / `to_object_vector` 可以检验跨折边方向搬运。展开与搬运不按 UV 距离寻路，不调用全局最近点去穿过薄片。

推拉的前向映射为：

```text
F(x) = x + delta * k(x)
k(x) = max(0, 1 - |x|² / radius²)² * (1 - freeze(x))
```

中心坐标为零，radius 和 delta 都使用对象单位。未冻结的中心恰好前进 delta；strength=-1 反向拉，strength 范围 [-1,1]。 `push(start,end,...)` 以同一局部表面展开中的手势位移生成 delta；`gesture(points,...)` 按相邻表面点生成多个参数，整段原子提交。过长或离开局部 patch 的手势须由宿主细分；不会把两个屏幕点直接做图像扭曲。

开发时保守拒绝：开放边界、非流形/退化/错误绕序、缺 UV、奇异 UV、材质集边界、过薄三角形、展开面积重叠或曲率环路不一致。碰到这些情况抛带 `code` 的 LiquifyError，不会返回“看似成功”的局部映射。笔刷圆外核为零；数值边界带按保守规则处理，包含与开放边界相切的情况也会拒绝。

**支持边界：局部可一致等距展开的面片。**平面、无曲率环路的局部折面和部分可展曲面适用。包含非零离散高斯曲率顶点的较大 patch 常会报 `curvature_holonomy`；缩小半径可能通过。这不等价于已支持完整球面或任意有机模型的大范围液化。展开是三角形闭包上的保守检查，也可能拒绝圆外部分引起的歧义。首版不用失真的近似展开掩盖此限制。

### 逆向采样与防折叠

目标点先逐笔划逆序回溯，**只组合坐标**：

```text
target -> inverse(last operation) -> ... -> inverse(first operation)
       -> original source primitive + barycentrics + UV
```

每个逆映射用有界不动点迭代求解 `q = target - delta*k(q)`。连续两次预览都查询同一个源 revision，不把上次预览再次采样，因此模块不引入重复重采样衰减。

径向核梯度上界为 `8 / (3*sqrt(3)*radius)`，再加冻结区的 Lipschitz 上界。默认 `|delta| * gradient_bound <= 0.35`，因此在已通过检查的局部展开内，映射是 identity 加 Lipschitz 小于 1 的扰动；逆向迭代收敛，局部奇异值界为 [1-L,1+L]。浮点实现仍使用明确容差，并非全范围精确几何认证。

超过位移上界时，默认缩短 delta 并设置 `displacement_limited`；`limit=False` 则拒绝。操作记录的 flags 必须供上层显示，不应把被缩短的位移展示成完整手势。多笔划的奇异值保守乘积超出配置界时，报 `composition_limit`。此界对分离笔划也保守，可能提前要求撤销/恢复；不会通过重采样当前图像作为新源来绕过它。

超过迭代次数报 `inverse_not_converged`，不返回部分或猜测地址。MappingResult 返回 inverse_stretch_bounds，供宿主识别放大范围；它不是完整 UV Jacobian/各向异性滤波 footprint。

### 冻结与对象空间对称

`add_freeze(center,radius,amount=1,inner_fraction=0.5)` 生成表面圆盘冻结区：内部平台固定值，外圈 cubic smoothstep 衰减到零。多个区域取 max，保持值在 [0,1]；平台 amount=1 精确阻止**后续笔触**改变该区域。过渡区同样计入防折叠上界。已有变形仍保留，各 PushOperation 捕获创建时的不可变 FreezeMask。clear_freeze 仅改变后续笔触的冻结状态，不追溯改变旧笔触。

这是最多有限个平滑圆盘组成的 mask，没有导入全分辨率任意栅格冻结图。freeze 参数本身参与 cancel/undo/redo。

`Symmetry(axis="x",offset=0,snap_distance=1e-7)` 在对象空间反射中心与位移向量，镜像中心必须落在实际模型表面，并通过法线兼容检查；不是翻转 UV。仅镜像定位会用既有最近表面查询，局部搬运不会使用它跳转表面。

- 轴上中心只生成一个操作，方向与镜像方向取平均；垂直对称平面的位移分量消除，并标记 `symmetry_axis_deduplicated`，纯法向手势另标记 `symmetry_axis_cancelled`。
- 轴外两个支持圆若可能重叠，报 `symmetry_overlap`；不以先左后右的叠加顺序冒充严格对称。可减小半径或使用轴上手势。
- 镜像位置不存在/最近面的法线不匹配会明确失败，不扩大搜寻半径猜测目标。snap_distance 使用对象单位，宿主应小于需要隔离的薄片间距。
- 冻结区优先于运动，左右不对称的冻结区可以造成左右不同结果。add_freeze 同样可传 symmetry，轴上会去重。

### 给 tile 生成器的稳定接口

| API | 契约 |
| --- | --- |
| `state.affected_regions()` | 有限 AffectedRegion 列表：primitive_id/chart_id/texture_set/uv_bounds；UV 界由笔刷局部方框裁剪三角形得到，保守覆盖有效圆 |
| `dirty_regions(before,after)` | 旧与新影响区的并集；取消、撤销、restore 时必须把旧预览范围也重新生成 |
| `state.map_point(SurfaceRef)` | MappingResult，含最终原图 SourceAddress、status、已应用操作数及逆向伸缩界 |
| `state.map_uv_points(primitive_id,uv_points)` | 有界批量 UV→目标重心坐标→原图地址；不在指定 UV 三角形内的点返回 None |
| `state.map_batch(refs)` | 最多 max_batch_points 个表面点；超限整批抛错，不返回部分结果 |
| `state.sample_channels(ref,names,sampler)` | 只做一次坐标映射，将**同一个** SourceAddress 交给每个 channel 的 sampler |

SourceAddress 包含冻结 source_stamp、源 primitive_id/barycentrics、对象位置、UV、chart_id、texture_set/name。它与 color/roughness/height 的值域、位深、颜色管理无关。高程在这里是纹理数据的坐标搬运，不改变几何。

宿主流程：

1. 捕获 `state`、`session.epoch` 和原始像素 lease。
2. 将 dirty_regions 的 UV 界转为现有 tile 调度任务；按该 primitive 的三角形覆盖生成小批目标 UV 像素中心。
3. 调用 map_uv_points，对每个非 None 结果，向固定 source_stamp 的原图取样。
4. 使用相同 SourceAddress 为各通道读取；由宿主实现源纹理跨 tile 读取、过滤、色彩空间和最终写入。
5. 发布前核对 epoch 和源版本；过期任务丢弃，不能写回新的参数状态。

这里不预分配全 16K 数组，不生成完整位移图，也不持有输出像素。UV 不 clamp、不 wrap、不 flip-V；现有存储的坐标约定仍由宿主负责。重叠 UV 的目标写冲突必须由宿主按明确 primitive/material 路由处理，本层不自动选择覆盖顺序。默认不跨不同 texture-set 的表面边搬运。

### 事务、恢复与限额

LiquifySession 的 state 始终是不可变参数快照。begin/commit 将多次 push 合为一次历史；cancel 返回完全相同的起始 state 对象。undo/redo 返回保存的原始参数对象，避免“施加反向笔刷”引入误差。restore 清除全部变形参数、保留当前冻结区，可撤销；不是局部恢复笔刷。

epoch 是单独的单调发布序号，取消/撤销即使回到相同参数对象也增加，防止旧异步结果误发布。会话控制器由单一拥有线程修改；worker 只读捕获的 state。

`state.parameters()` 返回 JSON 可序列化参数；`parameter_digest()` 为规范参数 JSON 的 SHA-256。`LiquifyState.from_parameters(source,record,limits=...)` 重新建立有界 patch，并严格匹配 source_stamp。`session.load_parameters(record)` 是可撤销的原子恢复。它不导入/重建原图，不自动适配修改过的网格。当前运行环境测试恢复后参数摘要和采样结果完全一致；不承诺不同浮点平台上的结果位级一致。

默认：每 patch 最多 256 面、64 操作、16 冻结圆盘、32 历史步、每 gesture 最多 256 点、每批最多 4096 点、逆解最多 64 次、参数工作估算 32 MiB。每个 patch 估算为 `4096 + 1024*faces` 字节，活跃参数及历史中的独立 patch 都计入准入。这个估算不是操作系统 RSS 强制上限，原有 mesh/BVH、原图 lease、输出 tile 与宿主其他子系统另计。多操作的伸缩预算可能比名义 64 操作上限先触发。

CPU 标量实现适合数学验证和小批离线/worker 查询；还不能据此宣称 16K tile 的逐 texel 液化达到交互性能。未来 GPU/向量化路径应遵循同一逆向映射、版本与源快照契约。

### 液化阶段实测证据

运行：

```text
python -B -m unittest discover -s tests -p "test_surface_liquify*.py" -v
python -B -m unittest discover -s tests -p "test_surface_*.py" -v
python -B -m surface_core.liquify_demo --output assets/surface-tests/liquify-demo.json --svg assets/surface-tests/liquify-checker.svg
```

最终全套 CPU 验收 **86 项通过**（液化专项 35 项 + 原 Surface API 51 项），0 failures / 0 errors / 0 skipped，用时 1.082 秒。原 Surface API 与原测试文件 SHA-256 均与上阶段一致。

自制 256 面（含独立背片）的示例：1 个推拉操作，覆盖 28 个前表面三角形；256 个标量逆向采样实测 36.578 ms，均值 0.143 ms/点。单次 wall-clock 测量，不代表一般网格、冻结区或多笔触性能。

记录：`assets/surface-tests/liquify-validation.json`、`liquify-test-results.txt`（最终输出末段）、`liquify-demo.json`、`liquify-checker.svg`。另附可直接导入的自制模型 `liquify-two-islands.obj` 和 `liquify-fold60.obj`。

棋盘 SVG 是自制表面坐标数据的 CPU 可视化，左为冻结原始棋盘，右为跨 UV 缝采样结果，红线标出几何上的 UV 缝；不是屏幕截图或屏幕空间扭曲。例程中的三通道读取同一个源 UV，背面薄片距离前面 0.0001 对象单位且保持 identity。

新增代码只依赖原有 Python/NumPy，未添加依赖、复制第三方/GPL 代码或读取用户作品。未改变 app/engine/tile_core/GPU/UI，也没有启动图形上下文。



## Liquify batch / shader packet v1（新增批量阶段）

本阶段新增 `liquify_packet.py`、`liquify_batch.py` 和 `liquify_batch_benchmark.py`。既有标量入口和返回类型保持不变；按 review 修复了 `liquify_geometry.py` 的不安全数值边界，具体见本节末尾。没有修改主 shader、app、engine、tile_core，也没有执行 GPU。

### 一次准备与复用

```python
import numpy as np
from surface_core.liquify_batch import CompiledLiquify, BatchLimits

# state 已由 LiquifySession 的 push/gesture 准备好全部局部展开。
compiled = CompiledLiquify(state, limits=BatchLimits(chunk_points=2048, face_block=32))

# tile_uv 是受影响 tile 内像素中心的 float64[N,2] UV；没有全图输入。
result = compiled.map_uv(primitive_id, np.asarray(tile_uv, dtype=np.float64))
valid = result.valid
original_uv = result.uv[valid]
original_sets = result.texture_sets[valid]
# 原图 sampler 仍必须按 result.source_stamp 绑定只读材质快照。
# 单点适配旧 SourceAddress：
address = compiled.address(result, index=0)  # outside-triangle => None
```

展开和邻接构建仍由笔触/参数准备阶段完成。CompiledLiquify 只复制已准备的 patch、预计算紧凑 UV 逆矩阵和打包参数；compile 和 map 阶段均不调用 DevelopedPatch.build 或构造 topology。相同不可变 state 的 compiled 对象可在多次预览和多个 tile 中复用，mask/笔触改变后编译新快照，旧快照仍可读。

`map_refs(ids[N], barycentrics[N,3])` 批量处理已有表面引用；`map_uv(primitive_id,uv[N,2])` 处理单个明确 UV 三角形的 tile 片段。UV 三角形外的元素 valid=False、primitive/chart/set=-1、UV/barycentric=NaN、applied_operations=0，禁止直接拿无效项采样。

BatchMapping 的列均不可写，包含 source_stamp、mapping_digest、input_offset、valid、primitive_ids、barycentrics、uv、chart_ids、texture_sets、applied_operations、inverse_stretch_bounds 和 scalar_fallback_count。通道仍共用这些坐标列，不为每个通道重新计算场。

### 分块与失败原子性

默认一次返回最多 65,536 点，结果数组预算 16 MiB；内部始终按 2,048 点分块，三角形测试按 32 面分块。不会创建整个 N×全部 mesh faces 的矩阵。

对于更长的调用者已有数组：

```python
for chunk in compiled.iter_uv(primitive_id, caller_owned_uv_ndarray):
    # chunk.input_offset 为原输入中的起点。
    stage_mapping_chunk(chunk)  # 宿主 staging，不应立即发布整个笔划
```

也有 `iter_refs(ids_ndarray,bary_ndarray)`。流式接口要求调用者已有 NumPy 数组，切片处理；调用者必须在查询期间保持输入不变，处理完一块后释放，才能维持输出内存上限。若把全部 chunk 放进列表，额外保留内存属于调用者。

map_uv/map_refs 只有整次成功才返回结果，后续块出错不会返回前面的部分结果。iter_uv/iter_refs 可以先产出成功块，后续块仍可能报错，因此整个 tile/笔划的原子发布由宿主 staging 和 epoch 校验负责。不要把“流式已产出一块”当作整笔成功。

每个采样点可以独立并行。每个点内部必须按操作逆序依次求解，不能把多个笔触同时相加或调换顺序。mask 使用同一三角形定位与原先的冻结区规则。

### 精度与参考回退

批量路径保持 float64、原有迭代上限与终止条件、UV/重心判定阈值及防折叠上界。普通路径使用 NumPy 的点批运算；freeze 的每次求值仍进行有界三角形定位，不把冻结区近似为屏幕圆。

在 rho²=1、冻结平台/外圈界、或 barycentric=-1e-10 判定附近（guard 1e-13），CPU 实现将相应点交还原标量参考，不放宽判定。UV 判定回退直接保留参考的完整结果。scalar_fallback_count 报告本结果中走参考路径的点数，不会把一次点的多个内部回退重复计数。整块遇到 outside_patch/inverse_not_converged 时，会由参考复核该块；参考仍失败就抛同一失败，不以 identity 掩盖。

批量与标量并不承诺不同平台下浮点值逐位相等；测试严格比较 valid/primitive/chart/texture-set/applied 数值身份，并比较 float64 坐标误差。未降为 float32 来获取速度。包导出若请求 precision="float32" 会明确拒绝。

### 资源准入

默认 BatchLimits：

| 项目 | 值 / 计算 |
| --- | --- |
| chunk_points / face_block | 2048 / 32 |
| 批临时工作保守估算 | 1 MiB + 2048×chunk_points + 128×chunk_points×face_block = 13 MiB |
| max_scratch_bytes | 32 MiB；初始化前检查 |
| 每输入点结果数组 | 57 字节（bool 1 + IDs/metadata 16 + barycentric 24 + UV 16） |
| 65,536 点结果 | 3,735,552 字节，约 3.56 MiB |
| 原子返回阶段复制峰值估算 | 2×结果数组字节；冻结 backing 期间输入/输出共存 |
| packet 常驻上限 | 32 MiB |
| packet 构造峰值估算 | 4×packet 字节 + 1 MiB；默认上限 128 MiB |

`compiled.resource_estimate(n)` 返回这些独立预算。它们不包括调用者输入、既有 mesh/BVH/标量 state、Python/NumPy 基线、原图像素 lease 或宿主输出 tile。对 NumPy 数组输入不复制整幅图；便利入口若收到列表，会转换该次有界输入（IDs+bary 最多约 32 字节/点，UV 约 16 字节/点），应另计。估算是准入控制而非 OS RSS 强制上限。

packet 按实际打包字节核对计数；不同 patch/多个 mask 会增长，但不会随图像宽高增长。结果列只随当前局部采样批增长。

### GPU 数据 ABI：已实现数据，尚未实现 shader

`compiled.packet` 为 ShaderPacket，`buffers()` 返回只读小端 `int32[]` 和 `float64[]` 两个平坦 word heap。manifest() 给出每张表所在 heap、**word offset**、shape、源身份、参数摘要、solver_iterations 与全局 inverse_stretch_bounds。

GPU 接线可使用两个只读 SSBO，所有矩阵按显式标量下标读，避免结构体 packing/GLSL 列主序误用。GLSL 对 SSBO 的 std430 和双精度类型有规范支持，但具体驱动编译、精度、速度及绑定能力必须由主 GPU 阶段验证。[GLSL 4.30 官方规范](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.30.pdf)、[ARB_gpu_shader_fp64 官方规范](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_gpu_shader_fp64.txt)。

```glsl
// 仅说明拟消费 ABI；本阶段未编译或运行此 shader。
layout(std430, binding = 0) readonly buffer IntHeap { int iw[]; };
layout(std430, binding = 1) readonly buffer DoubleHeap { double dw[]; };
```

CPU 表布局：

| 表 | 类型与每行 word 数 | 字段顺序 |
| --- | --- | --- |
| patch_meta | i32×4 | patch-row 起点、行数、0、0 |
| patch_values | f64×4 | radius、barycentric tolerance=1e-10、物理残差预算 radius×1e-8、0 |
| patch_faces | i32×4 | primitive_id、global face_meta row、-1、-1 |
| patch_xy | f64×8 | x0,y0,x1,y1,x2,y2,0,0 |
| patch_inverse | f64×4 | a00,a01,a10,a11；作用于 point−xy0 |
| patch_edges | f64×8 | 对象空间 e1.xyz、e2.xyz、0、0；用于物理重建残差 |
| face_meta | i32×4 | primitive_id、chart_id、texture_set、1 |
| face_uv | f64×8 | u0,v0,u1,v1,u2,v2,0,0 |
| face_uv_inverse | f64×8 | UV origin.xy、逆矩阵 a00,a01,a10,a11、0、0 |
| operation_meta | i32×4 | patch index、freeze_links 起点、freeze 个数、参数 flags |
| operation_values | f64×4 | delta.x、delta.y、Lipschitz 上界、迭代步差停止阈值 |
| freeze_meta | i32×4 | mask patch index、0、0、0 |
| freeze_values | f64×4 | radius、amount、inner_fraction、1/(1−inner_fraction) |
| freeze_links | i32×1 | 指向 freeze_meta/values 的 region index |

所有 primitive_id 列按对应 patch/全局 face 表升序排列，可二分查找；没有 GPU 指针。patch 坐标为对象单位、相对笔触中心的局部展开。shape 的最后一维连续；不要将上述 row-major 逆矩阵四个值直接当成 GLSL 默认列主序矩阵构造器。可显式算 `u=a00*dx+a01*dy; v=a10*dx+a11*dy`。

operation flags：1=displacement_limited、2=symmetry_axis_deduplicated、4=symmetry_axis_cancelled、8=symmetry_copy。这些只用于状态说明，对称已经在 CPU 准备为实际操作，shader 不应再次生成镜像笔触。

GPU 消费顺序：

1. 宿主按 affected/dirty regions 生成 tile 内目标 primitive 与像素中心 UV；优先只 dispatch 已打包的受影响面。
2. 从 face_uv_inverse 求目标 barycentric；不在三角形内则输出 MISS，不采样。
3. 逆序遍历 operation_meta。目标面不在该 patch、零 delta 或零 kernel 时保持原引用。
4. 按同一 kernel、freeze 表和固定迭代上限求 inverse。freeze 定位没有 preferred face，须选最小合格 primitive；最终 source 定位优先保留当前 destination primitive（若仍合格），否则选最小 primitive。
5. 重心截断/归一后检查 chart 残差，以及 `(b_clamped.yz−b_raw.yz) * [e1;e2]` 的对象空间残差，均不得超过 patch_values 的预算。
6. 由 face_uv 与源 barycentric 得到原图 UV；输出 tri/bary/UV/chart/texture-set 和状态，由所有通道复用一次坐标映射。
7. 材质集编号是该 mesh 的局部 texture_set，须由宿主映射到冻结 material revision 下的实际原图 page/texture 绑定。

用于 GPU 输出的建议状态为 MISS、VALID、MAPPED、NEEDS_REFERENCE、FAILED。CPU 的 guard 区域应在 shader 标为 NEEDS_REFERENCE，由宿主对相应局部点调用参考路径；不得默默放宽容差。迭代失败、超出 patch、物理残差超标则 FAILED，禁止发布该 tile 的部分映射。具体输出 SSBO 与 shader 尚未编写，CPU BatchMapping 的 57 字节/点列布局不能直接当成 GLSL struct ABI。

顶点拓扑、展开、邻接及方程参数均不需在每 texel 重建。编译器只打包操作实际引用的 patch/mask 面；可用 extra_primitives 额外加入需要 identity 查询的面。GPU 不能在包外猜测其他面；CPU 对未受影响面可以走原网格的 identity 地址路径。

需要实际 FP64 支持，并评估它在目标 GPU 上的代价。float32 的相对精度不能直接承诺本参考的 1e-10 重心阈值及 radius×1e-11 逆解要求；未来如要混合精度，必须单独建立误差证书与 GPU 对照验收，不能只替换 dtype。GLSL precise/FMA 行为也应纳入驱动验收，不能假设和 NumPy 逐位一致。

### 可交给 GPU 开发的 CPU 对照包

`assets/surface-tests/liquify-packet-manifest.json` 包含实际 offset/shape。`liquify-packet-reference.npz` 包含两个 heap、一个 destination primitive、256 个局部 UV 输入、valid/源 tri/bary/UV/chart/set/applied 的 CPU 预期输出。使用 `np.load(path, allow_pickle=False)` 读取；全部为数字数组。

这只证明数据可打包、CPU 消费与标量一致。文件内及本报告均明确标记未执行 GPU。后续必须实际编译 shader、检查目标驱动 FP64/布局、在这批输入和边界例上比对，再谈 GPU 性能。

### Review P2 数值边界修复

反例：三角形 (-1e9,0,0)、(1e9,0,0)、(0,1e9,0)，中心在底边中点，radius=1e-4。旧容差含面片尺度项，可能比 radius 更大，令开放边界检查跳过；随后极小负 barycentric 又被 clamp 回边，形成错误源点。

修复规则（标量、packet 编译、批量一致）：

- 在展开前按 `64*float64_epsilon*max(绝对对象坐标尺度,面片跨度,radius)` 估算误差，超过 `radius*1e-8` 即报 precision_limit。该反例直接拒绝。
- 邻接边访问使用 radius+数值容差的保守边界带，容差只相对 radius，不再以巨大面片尺度把有效笔刷边界抹掉；开放边界相切也保守拒绝。
- 即使拿到旧式 patch，locator 也会检查截断后的物理残差；反例的 (0,-1e-5) 源不能被当成底边命中。packet 编译独立复核 precision，拒绝旧式不安全 patch。
- 新增对称折面轴线、以及实际 world ray hit 经非均匀缩放/剪切变换后的对称回归，不以构造的对象坐标替代这些 gap 的测试。

这些是对不安全输入的明确拒绝，没有放宽通过范围获取速度。原有 Surface import/query 接口和文件未变；液化几何文件为本次 review 明确要求的修复。

### 本阶段验证与基准

执行：

```text
python -B -m unittest discover -s tests -p "test_surface_*.py" -v
python -B -m surface_core.liquify_batch_benchmark --sizes 1000 10000 64000 --cases plain freeze --output assets/surface-tests/liquify-batch-benchmark.json --fixture-dir assets/surface-tests
```

最终 CPU 全套 **111 项通过**，0 failures / 0 errors / 0 skipped；实测 2.275 秒。
完整日志：`assets/surface-tests/liquify-batch-test-results.txt`；验证与代码哈希：`liquify-batch-validation.json`。
原 Surface import/query/API 文件哈希均未变化；前阶段液化文件仅授权 review 修复的 `liquify_geometry.py` 改变。

基准 UTC：2026-10-03T09:21:45.881890+00:00，Python 3.11.9 / NumPy 2.4.4。

| 负载 | 输入点 | 有效并映射点 | 批量 ms | 标量 ms | 实测加速 | 批量阶段进程峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| plain | 1,000 | 609 | 4.066 | 95.717 | 23.54× | 37.60 |
| plain | 10,000 | 6,048 | 32.007 | 1000.680 | 31.26× | 40.33 |
| plain | 64,000 | 38,554 | 225.111 | 5972.034 | 26.53× | 44.75 |
| freeze | 1,000 | 609 | 39.271 | 670.680 | 17.08× | 38.19 |
| freeze | 10,000 | 6,048 | 399.047 | 6894.528 | 17.28× | 40.99 |
| freeze | 64,000 | 38,554 | 2211.035 | 43919.694 | 19.86× | 46.02 |

全部 150,000 个输入逐一比对，基准 scalar_fallback_count 均为 0。最大 UV / barycentric 绝对差分别为 1.11e-16 / 1.11e-16；离散身份与 valid/apply 计数全部一致。

本例 plain/freeze packet 实际分别为 9,056 / 10,212 字节（已包含物理残差边向量）；64,000 点结果列实际 3,648,000 字节。进程峰值包含解释器、mesh/BVH/state、输入和输出，不是临时工作数组单项用量；原始 JSON 分开记录 batch 后与完整 scalar 对比后的 RSS/峰值。
原始报告：`assets/surface-tests/liquify-batch-benchmark.json`。

基准在每个 case/size 的全新 CPU 进程运行；输入为真实对齐的 16K 目标 tile (38,34) 中、256×256 局部像素中心，包含落在指定三角形外的 MISS。freeze case 位于冻结的平滑过渡区。每个输入均与完整标量参考比较，不是抽样验证。单次 wall-clock 计时，256 次 batch 和 16 次 scalar 输入预热，没有独占主机保证。

这些数据不等于一整幅 16K 纹理处理，也不代表全部点都在三角形内或任意复杂面片/多笔划性能。当前 CPU freeze 64K 仍需秒级，不能宣称交互完成；用途是更快的 CPU 基础与未来 GPU 对照。所有原有局部可展、材质边界、防折叠、源快照 lease 与 epoch 发布限制继续适用。

