import argparse
import csv
import os
import time

import cv2
import numpy as np

import evaluate_caca_ber as evaluator
import jpeg_backend as jio


DEFAULT_OUTPUT_DIR = "telegram_qf_estimation"
DEFAULT_DATASET = "BOSSbase_1.01.zip"

STD_CHROMA_QTABLE = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)

ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]


def quality_to_qtable(base_table, quality):
    quality = int(np.clip(quality, 1, 100))
    scale = 5000 / quality if quality < 50 else 200 - quality * 2
    table = np.floor((base_table * scale + 50) / 100)
    return np.clip(table, 1, 255).astype(np.int32)


def parse_jpeg_quant_tables(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError(f"not a JPEG file: {path}")

    tables = {}
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker in (0xD8, 0xD9):
            continue
        if marker == 0xDA:
            break
        if pos + 2 > len(data):
            break
        segment_len = int.from_bytes(data[pos:pos + 2], "big")
        pos += 2
        segment = data[pos:pos + segment_len - 2]
        pos += segment_len - 2

        if marker != 0xDB:
            continue
        cursor = 0
        while cursor < len(segment):
            info = segment[cursor]
            cursor += 1
            precision = info >> 4
            table_id = info & 0x0F
            if precision == 0:
                if cursor + 64 > len(segment):
                    raise ValueError(f"truncated DQT segment in {path}")
                values = np.frombuffer(segment[cursor:cursor + 64], dtype=np.uint8).astype(np.int32)
                cursor += 64
            elif precision == 1:
                if cursor + 128 > len(segment):
                    raise ValueError(f"truncated 16-bit DQT segment in {path}")
                values = np.frombuffer(segment[cursor:cursor + 128], dtype=">u2").astype(np.int32)
                cursor += 128
            else:
                raise ValueError(f"invalid DQT precision {precision} in {path}")
            natural = np.zeros(64, dtype=np.int32)
            for idx, zz in enumerate(ZIGZAG):
                natural[zz] = values[idx]
            tables[table_id] = natural.reshape(8, 8)
    if not tables:
        raise ValueError(f"no JPEG quantization tables found: {path}")
    return tables


def estimate_table_quality(table, base_table):
    table = table.astype(np.float64)
    best = None
    for qf in range(1, 101):
        candidate = quality_to_qtable(base_table, qf).astype(np.float64)
        mae = float(np.mean(np.abs(table - candidate)))
        rmse = float(np.sqrt(np.mean((table - candidate) ** 2)))
        score = (rmse, mae)
        if best is None or score < best["score"]:
            best = {"qf": qf, "mae": mae, "rmse": rmse, "score": score}
    return best


def resolve_telegram_config(args):
    env_values = evaluator.load_env_file(args.env_file)
    token = args.telegram_token or os.environ.get("TELEGRAM_BOT_TOKEN") or env_values.get("TELEGRAM_BOT_TOKEN")
    chat_id = args.telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID") or env_values.get("TELEGRAM_CHAT_ID")
    return evaluator.validate_telegram_config(token, chat_id)


def write_input_jpeg(image_path, output_path, quality):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read input image: {image_path}")
    if not cv2.imwrite(output_path, gray, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
        raise ValueError(f"cannot write upload JPEG: {output_path}")


def selected_images_from_current_20(path, limit):
    if not os.path.exists(path):
        return []
    images = []
    with open(path, "r") as f:
        for line in f:
            item = line.strip()
            if item and os.path.exists(item):
                images.append(item)
            if len(images) >= limit:
                break
    return images


def build_report(path, args, rows, summary):
    lines = [
        "# Telegram 等效 JPEG QF 估计实验",
        "",
        "## 实验设置",
        "",
        f"- 数据集：`{args.dataset}`",
        f"- 使用图像数：{len(rows)}",
        f"- 图像选择：当前 20 图列表的前 {args.limit} 张",
        f"- 上传接口：Telegram Bot API `sendPhoto`",
        f"- 下载接口：Telegram Bot API `getFile`",
        f"- 上传前 JPEG 质量：{args.upload_quality}",
        "",
        "## 估计结果",
        "",
        f"- Luma QF：均值 {summary['qf_y_mean']:.2f}，中位数 {summary['qf_y_median']:.2f}，众数 {summary['qf_y_mode']}，范围 [{summary['qf_y_min']}, {summary['qf_y_max']}]",
    ]
    if summary.get("qf_c_mean") is not None:
        lines.append(
            f"- Chroma QF：均值 {summary['qf_c_mean']:.2f}，中位数 {summary['qf_c_median']:.2f}，众数 {summary['qf_c_mode']}，范围 [{summary['qf_c_min']}, {summary['qf_c_max']}]"
        )
    lines.extend([
        "",
        "| 图像 | 返回尺寸 | 返回字节数 | Luma QF | Luma RMSE | Chroma QF | Chroma RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['received_width']}x{row['received_height']} | {row['received_bytes']} | "
            f"{row['qf_y']} | {row['rmse_y']:.4f} | {row['qf_c'] or ''} | {row['rmse_c'] if row['rmse_c'] != '' else ''} |"
        )
    lines.extend([
        "",
        "## 说明",
        "",
        "该 QF 是通过平台返回 JPEG 的 DQT 量化表与标准 JPEG QF=1..100 的量化表逐一匹配得到的经验等效值。",
        "社交平台可能按尺寸、上传方式和服务端策略自适应处理，因此这里报告的是本实验条件下的 Telegram `sendPhoto` 等效 QF，而不是 Telegram 的全局固定 QF。",
    ])
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def summarize(rows):
    qf_y = np.array([int(r["qf_y"]) for r in rows], dtype=np.int32)
    summary = {
        "qf_y_mean": float(np.mean(qf_y)),
        "qf_y_median": float(np.median(qf_y)),
        "qf_y_mode": int(np.bincount(qf_y).argmax()),
        "qf_y_min": int(np.min(qf_y)),
        "qf_y_max": int(np.max(qf_y)),
    }
    qf_c_values = [int(r["qf_c"]) for r in rows if r["qf_c"] != ""]
    if qf_c_values:
        qf_c = np.array(qf_c_values, dtype=np.int32)
        summary.update({
            "qf_c_mean": float(np.mean(qf_c)),
            "qf_c_median": float(np.median(qf_c)),
            "qf_c_mode": int(np.bincount(qf_c).argmax()),
            "qf_c_min": int(np.min(qf_c)),
            "qf_c_max": int(np.max(qf_c)),
        })
    else:
        summary["qf_c_mean"] = None
    return summary


def main():
    parser = argparse.ArgumentParser(description="Estimate Telegram sendPhoto equivalent JPEG QF from returned quantization tables.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--selected-list", default="real_platform_20_results_robust/selected_20_images.txt")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--upload-quality", type=int, default=95)
    parser.add_argument("--telegram-token")
    parser.add_argument("--telegram-chat-id")
    parser.add_argument("--telegram-timeout", type=int, default=120)
    parser.add_argument("--telegram-sleep", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    upload_dir = os.path.join(args.output_dir, "upload")
    received_dir = os.path.join(args.output_dir, "received")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(received_dir, exist_ok=True)

    token, chat_id = resolve_telegram_config(args)
    evaluator.telegram_preflight_check(token, chat_id, timeout=min(args.telegram_timeout, 30))

    images = selected_images_from_current_20(args.selected_list, args.limit)
    if len(images) < args.limit:
        dataset_dir = evaluator.prepare_dataset(args.dataset, args.output_dir)
        images = evaluator.discover_images(dataset_dir, args.limit)
    if len(images) < args.limit:
        raise ValueError(f"need {args.limit} images, found {len(images)}")

    rows = []
    for idx, image_path in enumerate(images[:args.limit], 1):
        base = os.path.splitext(os.path.basename(image_path))[0]
        upload_path = os.path.join(upload_dir, f"{base}_q{args.upload_quality}.jpg")
        received_path = os.path.join(received_dir, f"{base}_telegram.jpg")
        write_input_jpeg(image_path, upload_path, args.upload_quality)
        evaluator.telegram_send_photo_and_download(
            upload_path,
            received_path,
            token,
            chat_id,
            caption=f"qf-estimate {idx}/{args.limit} {base}",
            timeout=args.telegram_timeout,
        )
        time.sleep(max(0.0, args.telegram_sleep))

        tables = parse_jpeg_quant_tables(received_path)
        y_est = estimate_table_quality(tables[min(tables.keys())], jio.STD_LUMA_QTABLE)
        c_est = None
        if len(tables) >= 2:
            chroma_id = sorted(tables.keys())[1]
            c_est = estimate_table_quality(tables[chroma_id], STD_CHROMA_QTABLE)
        received = cv2.imread(received_path, cv2.IMREAD_UNCHANGED)
        if received is None:
            raise ValueError(f"cannot decode returned image: {received_path}")
        height, width = received.shape[:2]
        row = {
            "image": os.path.basename(image_path),
            "input_path": image_path,
            "upload_path": upload_path,
            "received_path": received_path,
            "received_width": width,
            "received_height": height,
            "received_bytes": os.path.getsize(received_path),
            "num_quant_tables": len(tables),
            "qf_y": y_est["qf"],
            "mae_y": y_est["mae"],
            "rmse_y": y_est["rmse"],
            "qf_c": c_est["qf"] if c_est else "",
            "mae_c": c_est["mae"] if c_est else "",
            "rmse_c": c_est["rmse"] if c_est else "",
        }
        rows.append(row)
        print(f"{idx}/{args.limit} {row['image']}: Y QF={row['qf_y']} RMSE={row['rmse_y']:.4f}")

    detail_csv = os.path.join(args.output_dir, "telegram_qf_detail.csv")
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_csv = os.path.join(args.output_dir, "telegram_qf_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    report_md = os.path.join(args.output_dir, "telegram_qf_report.md")
    build_report(report_md, args, rows, summary)
    print(f"\nDetail CSV: {detail_csv}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Report: {report_md}")


if __name__ == "__main__":
    main()
