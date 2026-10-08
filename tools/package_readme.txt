{name}

一、怎么用
1. 把整个 SPLENDER 文件夹解压到任意位置（路径里有中文也行），双击里面的 SPLENDER.exe。
2. 不用装 Python，也不用装别的库，都在包里。
3. 电脑要求：Windows 10 或 11（64 位）；显卡支持 OpenGL 4.3，近几年的 NVIDIA、AMD 独显或 Intel 核显装好驱动都行。
4. 有几样功能第一次用要先准备十几秒（比如体素重构、油漆桶），之后就快了。
5. 偏好设置、快捷键、最近打开的工程存在 %APPDATA%\SPLENDER。删掉这个文件夹就恢复默认。
6. Arnold 渲染要电脑上另外装有 Arnold；没装时这个插件自动停用，别的功能照常。

二、包里有什么
- SPLENDER.exe：双击启动。
- README.md：功能介绍和全部快捷键。
- splender：程序本身（源码）。
- tools：打包、跑分、生成图标这些工具脚本。
- tests_new：自动测试。
- docs：设计、架构文档和更新记录。
- runtime：自带的 Python 3.11 和用到的库。

三、在这台电脑上接着开发
- 用包里的 Python：runtime\python\python.exe
- 跑单元测试：runtime\python\python.exe tests_new\run_all.py
- 跑整机自检（窗口开在屏幕外，不抢前台）：runtime\python\python.exe tests_new\run_selftests.py
- 重新打一份：runtime\python\python.exe tools\make_package.py

四、本次更新
{changes}
