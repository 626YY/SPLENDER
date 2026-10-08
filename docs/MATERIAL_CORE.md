# 现有 tile_core 上的材质结构与原子事务

本文件是内部实现契约。CPU 参考和现有 `AsyncMaterialSession`／`MeshMaterialDevice`／主窗口复用同一文档、历史及保存协议，未建立另一套持久化引擎。实际 GPU/UI 验收及限制见 `CHECKPOINT_MATERIAL_GPU.md`。

## 数据结构

`TileKey(texture_set, layer, channel, tx, ty)` 仍是唯一像素地址；layer 是稳定身份，不是列表索引。`ChannelDescriptor` 继续定义格式、分辨率、tile 大小、语义、颜色空间与 registration。`MaterialState.layers` 从底到顶，`ChannelPolicy` 定义单位、默认值、范围及是否可编辑；`LayerChannel` 关联输出、blend 及可选 coverage。

结构限制为最多 8 层、16 个通道／策略／单层绑定，结构 JSON 最多 64 KiB。图层 opacity 必须有限且在 0–1；字段、关联、格式、registration 和 tile 归属严格验证。已有结构的单位／策略不能通过普通图层编辑悄悄重解释；后续需要单独显式转换操作。

`new_material()` 的默认结构：

| 通道 | 存储 | 默认值 | 单位／计算 |
| --- | --- | --- | --- |
| color | RGBA8 | 透明黑 | sRGB 颜色；alpha 线性 |
| height | R16F | 0 | normalized_model_unit，单层 −1…+1 |
| roughness | R16F | 0.5 | dimensionless，0–1 |
| height.coverage | R8 | 0 | dimensionless，0–1 |
| roughness.coverage | R8 | 0 | dimensionless，0–1 |
| mask | R8 | 1 | dimensionless，0–1 |

缺失 tile 采用声明默认值。零高度和未绘制不同：独立 coverage 决定下层是否透出。R16F 是 IEEE 半精度浮点，不是均匀间隔 16 位整数。输入拒绝 NaN/Inf、超范围数据及超出 half 最大有限值 65504 的数值。

颜色采用 straight-alpha 存储，合成参考先解码颜色到线性，执行 normal/multiply/screen 并使用 alpha、opacity 和 mask，再重新编码。`unmanaged` 旧工程保留数值域算法，不能以迁移为由改变观感。高度支持 normal/add，粗糙度支持 normal；两者没有颜色传递函数。mask 可同时作用于该层各绑定。

## 事务、历史与保存

`VersionLedger` 持有 `material` 和 `published_material`；dirty 包含像素、场景及结构。`stage_material()` 与 `stage_cpu_write()`／`stage_gpu_write()` 使用同一个 pending 事务。commit 验证最终 tile 集合与结构匹配，再发布唯一 root。像素未变且结构未变的操作不增加 revision、不清空 redo。

结构历史保存在 `MaterialHistory`，与旧像素三元历史迭代接口兼容；结构前后像计入历史及事务预算。删除层时其 tile 删除也必须在同一事务中提交。undo/redo 同时恢复像素和结构。现有单通道路径无结构时仍使用原历史形式。

`SaveJob` 冻结像素、场景、descriptor 与材质结构。`ProjectRepository` 沿用内容 blob、校验、临时文件、fsync 和原子清单替换。发布前检查磁盘基线和调用者身份；晚到旧任务不能覆盖新版本。历史本身不序列化，工程保存的是当前可重新编辑的层栈和像素。

schema 1 为旧像素工程，schema 2 带内嵌场景，schema 3 带材质及可空场景。只显式迁移到 schema 3；旧格式可继续读。迁移保留稳定 layer ID、全部 descriptor 及 blob 字节，不猜旧数据单位。未知语义但已知字节格式可标为只读；未知字段／格式／schema 会拒绝整个工程，不能丢掉不认识的数据后继续保存。

## 同一映射与冻结源

`StrokeMapping(registration, width, height, tile_size, tiles)` 由不可变 R8 footprint 构成，大小有 8 MiB 上限。`CpuMaterialSession.paint()` 对同一个 footprint 求颜色、高度、粗糙度及必要 coverage，然后一次提交；不为每个通道独立重新求映射。

`VersionLedger.freeze_source()` 最多同时保留两个 `MaterialSource`，内容包括只读 root、结构、scene、document/session、revision 与 parameter epoch。逻辑版本加入 `required_versions()`，因此历史淘汰和 CPU 驻留清理后仍可从 durable blob 读取原源。`release_source()` 后不再为该操作保留；未释放数量有界。

`validate_source(require_current=True)` 既校验 lease 身份，也检查当前文档／会话／revision/root/结构/场景/参数 epoch。映射参数或工具状态改变时，宿主必须调用 `invalidate_mapping()`；该 epoch 是暂态失效标记，不是内容编辑，不增加 dirty。切换文档后旧源自动因身份不符被拒绝。

逻辑 lease **不是 GPU 槽位 pin**。现有异步 session 的 `SourcePageWindows`：acquire 按冻结版本载入并固定真实页，dispatch 包围采样回调并记录完成 fence，poll 完成后才释放。默认最多2个窗口／32MiB且每格式最多四分之一池；源、窗口和generation逐次校验。未知提交结果保留pin并隔离，关闭必须排空。CPU字节页及真实GL采样生命周期已通过，见 `evidence/material-source-pages/gpu1`。多格式源经COW仍保持的实际GL检查见 `evidence/material-gpu/run3`；这不等于液化工具已完成。

## 多批计算、一次发布

`session.mapped_batch(source, planned_keys)` 一次预留全部目标的前后像预算，开始现有账本事务。`stage(changes)` 可以分多次提供块，但每个 planned key 必须恰好出现一次。重复、越界、格式错误、源过期或中途异常会 abort 并回收新句柄；未提交前 current root、history、redo 都不改变。`commit()` 需要全部 planned key 已到齐并再次检查 source。

调用者在 finally 中释放源 lease。无变化块仍必须声明到齐，但不会分配新内容版本。纯映射计算可分 chunk 执行；不可把每个 chunk 作为独立可见提交。GPU 实现也必须保留这一边界。

## CPU 参考与预算

`CpuMaterialSession` 只接收无在途 GPU/readback 的 quiescent 账本；它使用现有 SnapshotPool／ProjectRepository，把逐 tile 数组用于正确性 oracle。`persist()` 在该参考中同步写 blob，`trim_durable()` 丢可重载 CPU 副本。不要将其耗时描述成实时 GPU 编辑性能。

`allocation_plan()` 本身只返回资源方案；当前主窗口已由 `MeshMaterialDevice` 实际分配三个完整输出2 GiB、三个格式各512页224 MiB。还需加driver、Qt、mesh、atlas/PBO与其他应用占用，不能把显式纹理预算当总显存实测。64 MiB事务预算在三通道＋两个coverage同时修改时最多51个不同tile坐标；合成输入页及物理源窗口也保留余量。数值布局不支持时在替换前拒绝，显存分配仍可失败；突发上下文丢失恢复未实现。

文档另有65,536块／32MiB清单总量限制，由 `document_limits` 同时用于提交及保存。双层六个存储平面全16K共49,152块，容量测试已通过。提交前校验proposed root与结构，超限不改历史或redo；8层结构上限不取消文档总量上限。清单大小按最长64位序号预留，防止仅保存epoch增长导致超限。这个文档限制不同于GPU驻留、CPU像素池和单笔事务限制。

重复 CPU 验证，使用新的输出目录：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -X utf8 run_cpu_checks.py evidence/material-cpu/my-regression
python -m experiments.verify_material_cpu --output evidence/material-cpu/my-reference
```

`my-reference/project` 是无内嵌OBJ的CPU schema 3夹具，模型主窗口不能直接把它当模型工程打开。主窗口新建／保存的是带scene的受支持材质布局。最新结果和缺项见 `CHECKPOINT_MATERIAL_GPU.md`。
