# multi_media — 说明

这是用于对比传统 J-UNIWARD 与本文 J-UNIWARD-P 流程的实验脚本。

主要文件：
- `multi_media.py`：主脚本，编辑顶部变量后直接运行即可。可通过命令行或顶部变量控制要处理的图片数量和参数。
- `caca_algorithm.py`：独立 CACA 程序，包含 JPEG 域 J-UNIWARD 代价、纯 Python syndrome trellis 嵌入/提取、合法解空间优化。
- `jpeg_backend.py`：JPEG DCT 后端兼容层，优先使用 `jpegio`/`jpeglib`，若都不可用则使用纯 OpenCV/Numpy DCT 后端。
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

独立 CACA 程序

嵌入文本消息并生成中间 JPEG：

```bash
python caca_algorithm.py embed input.pgm --message "hello" --output caca_intermediate.jpg --key 1234
```

嵌入二进制文件：

```bash
python caca_algorithm.py embed input.pgm --message-file secret.bin --output caca_intermediate.jpg --key 1234
```

从中间 JPEG 提取消息：

```bash
python caca_algorithm.py extract caca_intermediate.jpg --key 1234 --output recovered.bin
```

默认参数：`Qo=95`，`Qc=80`，`payload=0.2 bpnzac`，`STC h=10`，小波滤波器为 `Daubechies 8`，平滑常数 `sigma=2^-50`，CACA 搜索截断范围为 `[-2, 2]`。

说明：该程序保证的是论文公式中的理想 DCT 信道，即 `round(I * Mo / Mc) == S`；真实社交平台 JPEG 重压缩包含空域舍入和像素裁剪，不能由这个约束直接保证 100% 提取。若某些系数在默认 `[-2, 2]` 截断范围内没有合法解，程序会对这些系数回退到完整合法解空间并在诊断信息中报告数量。

BER 与抗隐写分析评测

`evaluate_caca_ber.py` 用于对比原论文式逆推算法与 CACA 改进算法。默认使用 Telegram Bot API 作为真实社交平台信道：脚本把图片作为 `photo` 上传到 Telegram，再通过 `getFile` 下载平台返回的图片并计算 BER。这个流程适合远程 SSH 服务器，不需要桌面浏览器。

准备 Telegram 真实平台信道：

1. 在 Telegram 中通过 `@BotFather` 创建 bot，得到 bot token。
2. 给 bot 发一条消息，或者把 bot 加入测试群/频道。
3. 复制 `.env.example` 为 `.env`，填入 bot token 和 chat id：

```bash
cp .env.example .env
nano .env
```

`.env` 内容示例：

```bash
TELEGRAM_BOT_TOKEN=123456789:AA...完整 token
TELEGRAM_CHAT_ID=你的 chat id 或 @channelusername
```

注意：`TELEGRAM_BOT_TOKEN` 必须是 `@BotFather` 给出的完整 token，格式通常是 `数字:AA...`。不要只填写冒号后面的 `AA...` 部分，否则 Telegram API 会返回 `HTTP Error 404: Not Found`。

如果不知道 `chat id`，可以在给 bot 发消息后运行：

```bash
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getUpdates"
```

返回 JSON 中的 `message.chat.id` 就是可用的 `TELEGRAM_CHAT_ID`。

脚本默认读取当前目录的 `.env`。也可以用 `--env-file path/to/.env` 指定其它配置文件；命令行参数 `--telegram-token`、`--telegram-chat-id` 和系统环境变量会覆盖 `.env` 中的值。

使用已上传的 BOSSbase zip 直接跑真实平台 BER 测试：

```bash
python evaluate_caca_ber.py --dataset BOSSbase_1.01.zip --limit 10
```

脚本会自动解压 `BOSSbase_1.01.zip` 到 `caca_eval_results/datasets/`，并递归读取其中的 `.pgm` 图像。

固定实验脚本：20 张同图对比 + 自动分析报告

如果要严格按本实验要求测试“老算法和新算法都使用同一批 20 张 BOSSbase 图像，并经过真实 Telegram 上传下载信道”，运行：

```bash
python run_20_image_real_analysis.py
```

该脚本默认：
- 从 `BOSSbase_1.01.zip` 中选取固定排序的前 20 张图像；
- 每张图同时生成 `original` 和 `caca` 两个候选图；
- 逐张上传到 Telegram，再下载 Telegram 返回图像；
- 计算真实平台 BER、DCT mismatch 和抗隐写分析代理指标；
- 输出 `real_platform_20_results/real_platform_20_analysis.md` 分析报告。

主要输出：
- `selected_20_images.txt`：本次实验使用的 20 张图片路径。
- `upload_candidates/`：上传到 Telegram 的老算法/新算法候选图。
- `telegram_received/`：Telegram 平台返回的真实下载图。
- `real_platform_20_detail.csv`：逐图逐算法明细。
- `real_platform_20_summary.csv`：均值汇总。
- `real_platform_20_analysis.md`：自动生成的实验分析报告。

如果 Telegram 偶发连接重置，只想重试失败样本并合并完整统计：

```bash
python retry_failed_real_analysis.py
```

该脚本会读取当前 `real_platform_20_results/real_platform_20_detail.csv`，只补跑其中 `algorithm=error` 的图像，成功后替换错误行并重建 `summary.csv` 和 `analysis.md`。

如果要手动上传到其它真实平台，先生成待上传图像：

```bash
python evaluate_caca_ber.py --dataset BOSSbase_1.01.zip --limit 10 --channel write-only --output-dir caca_eval_results
```

把 `caca_eval_results/upload_candidates` 中的图片上传到平台并下载回来，保持文件名类似 `<base>_original.jpg`、`<base>_caca.jpg`，放入一个目录后运行：

```bash
python evaluate_caca_ber.py --dataset BOSSbase_1.01.zip --limit 10 --channel received --received-dir downloaded_images --output-dir caca_eval_results
```

输出：
- `detail_results.csv`：逐图逐算法的真实信道 BER、理想信道 BER、DCT 系数 mismatch、修改率等。
- `summary_results.csv`：原算法和 CACA 的均值对比。

抗隐写分析性能目前使用无需训练模型的代理指标：J-UNIWARD 总失真、DCT 修改率、DCT L1/L2、DCT 直方图偏移和空域残差能量偏移。数值越低通常表示统计扰动越小；若要报告真正分类器检测准确率，需要额外接入训练好的隐写分析器。

`--channel simulate` 仅保留作调试用途，不作为真实平台实验结果。

临时文件与调试

- 临时文件存放在 `./temp_stego`，脚本默认在每张图处理完成后删除中间文件。如果想保留中间结果以便调试，请编辑 `multi_media.py` 注释掉删除 `os.remove(...)` 的行。

关于 aarch64 / ARM 服务器依赖

本项目现在不强制依赖 `jpegio` 或 `jpeglib`。在 aarch64 上，这两个包都可能因为 C 扩展编译问题失败；脚本会自动退回到纯 OpenCV/Numpy 的 JPEG-like DCT 后端。因此推荐安装最小依赖：

```bash
conda create -n caca310 python=3.10 -y
conda activate caca310
conda install -c conda-forge numpy scipy pillow tqdm opencv -y
```

然后检查：

```bash
python -c "import cv2, numpy, scipy, PIL, tqdm; print('ok')"
```

如果你在 x86_64 机器上已有 `jpegio` 或能成功安装 `jpeglib`，脚本会自动优先使用它们；否则会走 OpenCV DCT 后端。普通安装可以直接运行：

```bash
pip install -r requirements.txt
```

- 不建议在 aarch64 上继续强行安装 `jpegio`/`jpeglib`；遇到 `-m64`、`jinclude.h`、`requirements.txt` 等构建错误时，直接使用上述最小依赖即可。
- OpenCV DCT 后端不是 bit-exact JPEG 系数读写器，但足够在 ARM 服务器上运行真实 Telegram 平台 BER 和统计扰动对比实验。

性能建议

- 全量处理 BOSSbase（约 10000 张）会耗时且 I/O 密集，建议先用小样本（例如 `NUM_IMAGES = 100`）调试参数与逻辑。
- 若需要更快，可以考虑并行化处理（我可以为你添加基于 `concurrent.futures` 的批处理改造）。

输出保存

- 当前脚本只在终端打印平均 BER。如果需要将每张图的 BER 保存到 CSV，我可以帮你修改脚本并保存为 `results.csv`。

遇到问题怎么办

- 如果运行时报错（例如模块导入错误、JPEG 后端依赖问题或权限错误），把终端完整错误粘贴到这里，我来帮你定位与修复。

需要我现在：
- 帮你把 `NUM_IMAGES` 改为其它数字并运行一次；
- 或者添加把结果保存到 CSV 的功能；
- 或者并行化处理加速？

请选择下一步或直接运行 `python multi_media.py`（在虚拟环境中）。
