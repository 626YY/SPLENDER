# 第三方组件和素材

SPLENDER 自己的代码按 [GNU AGPL-3.0](LICENSE) 发布。程序用到、或者便携包里一起带上的第三方组件和素材如下，各自按原来的许可证使用。

## 运行时用到的库

| 组件 | 用途 | 许可证 |
|---|---|---|
| Python 3.11 | 运行环境（便携包里带一份） | PSF License |
| PySide6 / Shiboken6（Qt 6） | 界面 | LGPL-3.0（Qt 按 LGPL 使用，动态链接，可自行替换） |
| ModernGL、glcontext | OpenGL 4.3 调用 | MIT |
| NumPy | 数组计算 | BSD-3-Clause |
| Numba、llvmlite | CPU 上的加速（体素重构、洪泛等） | BSD-2-Clause |
| SciPy | 图像和几何计算 | BSD-3-Clause |
| OpenCV（opencv-python） | 图像处理 | Apache-2.0 |
| Pillow | 读写图片 | MIT-CMU（HPND） |
| zstandard、lz4 | 工程文件和页面压缩 | BSD-3-Clause |
| psutil | 内存和显存预算 | BSD-3-Clause |
| cffi、pycparser | 调用降噪库 | MIT、BSD-3-Clause |
| pyoidn（Intel Open Image Denoise） | 渲染降噪 | Apache-2.0 |
| OpenEXR | 读写 EXR | BSD-3-Clause |

## 自带的素材

| 素材 | 位置 | 许可证 |
|---|---|---|
| Lucide 图标 | `splender/resources/icons` | ISC（见 `LICENSE-lucide.txt`） |
| 环境 HDRI（Greg Zaal / Poly Haven） | `splender/resources/hdri` | CC0 |
| MatCap 材质球（Blender 社区） | `splender/resources/matcap` | CC0 / 公有领域 |
| 工作室光预设（由 Blender 自带的 .sl 预设转换） | `splender/resources/studio` | GPL-2.0-or-later |

显示变换里的 Filmic、AgX 曲线只用 Blender 的输出数值拟合，程序里不带 Blender 的查找表。
