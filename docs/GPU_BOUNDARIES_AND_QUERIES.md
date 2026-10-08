# 网格 GPU 边界补测与 timer query 生命周期

## 未完成传输时请求 resize／受控重建

固定证据 `evidence/mesh-boundaries/run3/result.json` 与同目录 `source/`；5 项检查通过，源码前后相同，2026-10-03 10:19:53 UTC 开始，共 66.641 秒。该快照早于后续 CPU 材质 lease 和 query 改动，不能当成这些新改动的 GPU 验证。

`experiments/verify_mesh_boundaries.py` 先完成一次确定 UV 的控制笔划、记录完整 16K 哈希并撤销。被测操作的 readback enqueue 后立即执行零超时 fence 查询；仅在返回 not-ready 时立即发出 resize 或受控 context rebuild 请求，中间没有纹理读取、query result、glFinish 或 sleep。请求后等待整个操作、回读／存储队列及重建完成，才再次读取全图哈希。

两种情形都在第一次尝试观察到 not-ready；resize 保持同一设备，rebuild 记录旧原生上下文销毁和新上下文创建，最终完整 16384² RGBA8 哈希均与控制组相同。重建实现先受控排空并保存再销毁；此结果不表示“可以在 DMA 进行中强行销毁设备”，也不覆盖突然驱动重置或掉电。

保留的 `run1`／`run2` 是测试脚本失败：前者用 resize 前后屏幕坐标映射到略不同 UV，后者在异步重建完成前错误地判定 idle。最终使用固定 UV 和连续稳定的完整队列／操作检查修正，未为让测试通过改写编辑器结果。

## 导入时 UI 停顿

自制 131,072 三角面曲面，真实窗口，目标 5ms Qt 心跳：

| 项目 | 实测 |
| --- | --- |
| 导入总墙钟 | 30.044 s |
| 后台 OBJ 解析 | 6,519.57 ms |
| 后台 topology/BVH | 4,000.80 ms |
| 后台宿主 UV 验证 | 18,878.49 ms |
| UI 线程 GPU 设备资源创建跨度 | 340.47 ms |
| 心跳间隔，3,088 样本 | p50 5.17 / p95 28.65 / p99 74.91 / max 373.27 ms |
| 进程 RSS 采样峰值 | 660,267,008 字节 |

导入期间确实发生数百毫秒 UI 延迟；后台线程不能消除 GIL 与资源创建停顿。心跳不是笔刷输入延迟或 GPU kernel 时间。导入性能仍需改进，不能称完全流畅。

## query 资源审计与修复

主 `mesh_app`／`async_app`／`async_ui_engine`／`mesh_bridge/device`／`gl_async_device` 路径未创建逐帧 GPU timer query，使用 CPU 与 Qt 回调时序。发现旧同步颜色、独立高度笔刷及高度显示按 tile／帧创建 ModernGL Query，不能依赖 Python 对象销毁来删除原生 GL query。

新增 `gpu_timing.OwnedElapsedQuery`，每个旧引擎／高度视图只生成一个原生 query，反复使用；release 时当前上下文内明确 `glDeleteQueries`，可重复 close，异常也结束活动区间。修改 `engine.py`、`height_engine.py`、`profile_sync_path.py` 和独立网格 benchmark 的计时所有权。`app.py` 的 swap 统计仅保留最近 4,096 项。

`tests/test_gpu_query_ownership.py` 的 3 项 CPU GL 模型检查通过：20,000 次区间只生成 1 个 ID，异常闭合、显式且仅一次删除、关闭后拒绝使用。其测试列入 `evidence/material-cpu/final-cpu2/manifest.json`。

**真实 GL 生命周期 smoke 已通过**：协调确认 GPU 时段后，`evidence/query-ownership/run1/result.json` 记录 4 字节 SSBO、2,000 次计时区间、1 个 query ID，计算计数等于 2,000，close 前 `glIsQuery` 为 true、close 后为 false，二次 close 安全。helper 源码 SHA-256 为 `991365ad4c2340e421a61a9207e98f979480efe5d8f5a8fed4e4505912054b19`。进程正常退出，GPU 已明确交回。测试没有分配 16K 图，也不代替整个旧同步编辑器的性能回归。

下次需要复测时先协调 GPU，再使用新目录：

```powershell
python -m experiments.verify_owned_query_gpu --gpu-slot granted --output evidence/query-ownership/my-run
```

`--gpu-slot granted` 只是显式记录外部协调，不自行申请／锁定 GPU。旧同步 query 的 result 读取会阻塞，这是开发计时路径，不应移入主异步交互循环。
