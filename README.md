# multi_media — 说明

这是用于对比传统 J-UNIWARD 与本文 J-UNIWARD-P 流程的实验脚本。

主要文件：
- `multi_media.py`：主脚本，编辑顶部变量后直接运行即可。可通过命令行或顶部变量控制要处理的图片数量和参数。
- `jpeg_toolbox.py`：对 `jpegio` 写入的兼容封装（回退到 PIL/Scipy）。
- `requirements.txt`：列出 Python 依赖。

快速开始

1. 激活虚拟环境（若已有项目虚拟环境）：

```bash
source .venv/bin/activate
```

2. 安装依赖：

```bash
.venv/bin/pip install -r requirements.txt
```

3. 运行脚本：

```bash
.venv/bin/python multi_media.py
```

控制处理图片数量（两种方式）

- 直接编辑文件顶部的 `NUM_IMAGES`（推荐）：打开 `multi_media.py`，把 `NUM_IMAGES = 100` 改为你想要的数字，保存后运行脚本。
  - 若 `NUM_IMAGES` 为 `None` 或 `0`，则脚本会处理数据集中的所有图片。
- 使用命令行参数 `-n/--num`：

```bash
.venv/bin/python multi_media.py -n 200
```

注意：脚本优先使用顶部 `NUM_IMAGES`（如果为正数），否则使用命令行 `-n` 参数。

设置图像质量与嵌入率

可以通过命令行覆盖默认参数：

```bash
.venv/bin/python multi_media.py --Qo 100 --Qc 95 --payload 0.4
```

或者修改 `multi_media.py` 顶部的 `Q_O`, `Q_C`, `PAYLOAD` 变量。

临时文件与调试

- 临时文件存放在 `./temp_stego`，脚本默认在每张图处理完成后删除中间文件。如果想保留中间结果以便调试，请编辑 `multi_media.py` 注释掉删除 `os.remove(...)` 的行。

关于 `jpegio` 和系统依赖

- `jpegio` 在某些系统上需要系统级 JPEG 开发库（libjpeg）支持。若 `pip install jpegio` 报错，请先安装系统依赖（Debian/Ubuntu 示例）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libjpeg-dev libpng-dev
```
然后再安装 Python 包：
```bash
.venv/bin/pip install jpegio
```

- `jpeg_toolbox.py` 已实现对 `jpegio.write` 的子进程回退和 PIL/Scipy 回退路径，遇到写入问题时脚本会尝试更稳健的方案。

性能建议

- 全量处理 BOSSbase（约 10000 张）会耗时且 I/O 密集，建议先用小样本（例如 `NUM_IMAGES = 100`）调试参数与逻辑。
- 若需要更快，可以考虑并行化处理（我可以为你添加基于 `concurrent.futures` 的批处理改造）。

输出保存

- 当前脚本只在终端打印平均 BER。如果需要将每张图的 BER 保存到 CSV，我可以帮你修改脚本并保存为 `results.csv`。

遇到问题怎么办

- 如果运行时报错（例如模块导入错误、`jpegio` 安装问题或权限错误），把终端完整错误粘贴到这里，我来帮你定位与修复。

需要我现在：
- 帮你把 `NUM_IMAGES` 改为其它数字并运行一次；
- 或者添加把结果保存到 CSV 的功能；
- 或者并行化处理加速？

请选择下一步或直接运行 `python multi_media.py`（在虚拟环境中）。
