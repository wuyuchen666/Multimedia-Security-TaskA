import argparse
import os
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

import caca_algorithm as caca
import evaluate_caca_ber as evaluator


DEFAULT_OUTPUT_DIR = "real_platform_20_results"
DEFAULT_DATASET = "BOSSbase_1.01.zip"
DEFAULT_NUM_IMAGES = 20


def resolve_telegram_config(env_file, token_arg, chat_id_arg):
    env_values = evaluator.load_env_file(env_file)
    token = token_arg or os.getenv("TELEGRAM_BOT_TOKEN") or env_values.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id_arg or os.getenv("TELEGRAM_CHAT_ID") or env_values.get("TELEGRAM_CHAT_ID")
    return evaluator.validate_telegram_config(token, chat_id)


def build_eval_args(args, token, chat_id):
    return SimpleNamespace(
        Qo=args.Qo,
        Qc=args.Qc,
        payload=args.payload,
        constraint_height=args.constraint_height,
        key=args.key,
        search_min=args.search_min,
        search_max=args.search_max,
        original_search_radius=args.original_search_radius,
        max_message_bytes=args.max_message_bytes,
        robust_repeat=args.robust_repeat,
        robust_min_abs=args.robust_min_abs,
        channel="telegram",
        telegram_token=token,
        telegram_chat_id=chat_id,
        telegram_timeout=args.telegram_timeout,
        telegram_sleep=args.telegram_sleep,
    )


def paired_rows(rows):
    pairs = {}
    for row in rows:
        if row.get("algorithm") not in evaluator.ALGORITHMS:
            continue
        pairs.setdefault(row["image"], {})[row["algorithm"]] = row
    return {image: pair for image, pair in pairs.items() if set(pair.keys()) == set(evaluator.ALGORITHMS)}


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def mean_metric(rows, algorithm, key):
    values = [safe_float(r[key]) for r in rows if r.get("algorithm") == algorithm]
    values = np.array(values, dtype=np.float64)
    return float(np.nanmean(values)) if values.size else np.nan


def relative_reduction(original, improved):
    if not np.isfinite(original) or not np.isfinite(improved) or original == 0:
        return np.nan
    return (original - improved) / original


def format_float(value, digits=6):
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_selected_images(path, images):
    with open(path, "w") as f:
        for image in images:
            f.write(image + "\n")


def build_analysis_report(path, args, images, rows, summary_rows, detail_csv, summary_csv, selected_path):
    ok_rows = [r for r in rows if r.get("algorithm") in evaluator.ALGORITHMS]
    errors = [r for r in rows if r.get("algorithm") == "error"]
    pairs = paired_rows(ok_rows)

    metrics = [
        ("real_ber", "真实平台 BER"),
        ("real_coeff_mismatch", "真实平台 DCT mismatch"),
        ("uniward_distortion", "J-UNIWARD 总失真"),
        ("dct_change_rate", "DCT 修改率"),
        ("dct_l1", "DCT L1"),
        ("dct_l2", "DCT L2"),
        ("dct_hist_l1", "DCT 直方图偏移"),
        ("residual_delta", "空域残差能量偏移"),
    ]

    lines = [
        "# 20 张 BOSSbase 图像真实平台 BER 与抗隐写分析对比报告",
        "",
        "## 实验设置",
        "",
        f"- 数据集：`{args.dataset}`",
        f"- 选图数量：{len(images)}",
        f"- 实际成功成对比较数量：{len(pairs)}",
        f"- 社交平台信道：Telegram Bot API `sendPhoto` 上传 + `getFile` 下载",
        f"- 原始质量因子 Qo：{args.Qo}",
        f"- 信道质量因子 Qc 参数：{args.Qc}",
        f"- Payload：{args.payload} bpnzac",
        f"- STC 约束高度 h：{args.constraint_height}",
        f"- CACA 搜索截断范围：[{args.search_min}, {args.search_max}]",
        f"- CACA 鲁棒重复次数：{args.robust_repeat}",
        f"- CACA 鲁棒最小系数幅值：{args.robust_min_abs}",
        f"- 明细 CSV：`{detail_csv}`",
        f"- 汇总 CSV：`{summary_csv}`",
        f"- 选图列表：`{selected_path}`",
        "",
        "## 均值对比",
        "",
        "| 指标 | 原算法均值 | CACA 均值 | CACA 相对降低 |",
        "|---|---:|---:|---:|",
    ]

    for key, label in metrics:
        original = mean_metric(ok_rows, "original", key)
        improved = mean_metric(ok_rows, "caca", key)
        reduction = relative_reduction(original, improved)
        reduction_text = "nan" if not np.isfinite(reduction) else f"{reduction * 100:.2f}%"
        lines.append(
            f"| {label} | {format_float(original)} | {format_float(improved)} | {reduction_text} |"
        )

    lines.extend([
        "",
        "## 逐图关键结果",
        "",
        "| 图像 | 原算法 BER | CACA BER | 原算法失真 | CACA 失真 |",
        "|---|---:|---:|---:|---:|",
    ])

    for image in sorted(pairs):
        original = pairs[image]["original"]
        improved = pairs[image]["caca"]
        lines.append(
            f"| {image} | {format_float(safe_float(original['real_ber']))} | "
            f"{format_float(safe_float(improved['real_ber']))} | "
            f"{format_float(safe_float(original['uniward_distortion']))} | "
            f"{format_float(safe_float(improved['uniward_distortion']))} |"
        )

    lines.extend([
        "",
        "## 结论提示",
        "",
        "- `real_ber` 越低，说明经过真实 Telegram 上传下载后的信息提取越稳定。",
        "- J-UNIWARD 总失真、DCT 修改率、DCT 直方图偏移、空域残差能量偏移越低，通常表示抗隐写分析风险越低。",
        "- 如果 BER 和统计失真出现 trade-off，应优先在报告中分别讨论鲁棒性与不可检测性，而不是只用单一指标下结论。",
    ])

    if summary_rows:
        lines.extend(["", "## 原始汇总行", ""])
        for row in summary_rows:
            lines.append(f"- `{row['algorithm']}`: {row}")

    if errors:
        lines.extend(["", "## 失败样本", ""])
        for row in errors:
            lines.append(f"- `{row.get('image')}`: {row.get('error')}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_experiment(args):
    token, chat_id = resolve_telegram_config(args.env_file, args.telegram_token, args.telegram_chat_id)
    evaluator.telegram_preflight_check(token, chat_id, timeout=min(args.telegram_timeout, 30))
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_dir = evaluator.prepare_dataset(args.dataset, args.output_dir)
    images = evaluator.discover_images(dataset_dir, args.num_images)
    if len(images) < args.num_images:
        raise ValueError(f"need {args.num_images} images, found {len(images)} in {dataset_dir}")

    selected_path = os.path.join(args.output_dir, "selected_20_images.txt")
    write_selected_images(selected_path, images)

    upload_dir = os.path.join(args.output_dir, "upload_candidates")
    received_dir = os.path.join(args.output_dir, "telegram_received")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(received_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="real20_", dir=args.output_dir)
    eval_args = build_eval_args(args, token, chat_id)

    rows = []
    try:
        for image_path in tqdm(images, desc="Real-platform paired test"):
            try:
                ctx = evaluator.load_context(image_path, args.Qo, args.Qc, temp_dir)
                rows.extend(evaluator.evaluate_image(ctx, eval_args, upload_dir, None, received_dir))
            except Exception as exc:
                rows.append({
                    "image": os.path.basename(image_path),
                    "algorithm": "error",
                    "Qo": args.Qo,
                    "Qc": args.Qc,
                    "payload_bpnzac": args.payload,
                    "message_bytes": 0,
                    "message_bits_with_header": 0,
                    "usable_coefficients": 0,
                    "stc_changed_coefficients": 0,
                    "ideal_ber": np.nan,
                    "real_ber": np.nan,
                    "ideal_coeff_mismatch": np.nan,
                    "real_coeff_mismatch": np.nan,
                    "fallback_or_unmatched_coefficients": 0,
                    "upload_path": "",
                    "received_path": "",
                    "dct_change_rate": np.nan,
                    "dct_l1": np.nan,
                    "dct_l2": np.nan,
                    "dct_hist_l1": np.nan,
                    "residual_delta": np.nan,
                    "uniward_distortion": np.nan,
                    "error": str(exc),
                })
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)

    detail_csv = os.path.join(args.output_dir, "real_platform_20_detail.csv")
    summary_csv = os.path.join(args.output_dir, "real_platform_20_summary.csv")
    report_md = os.path.join(args.output_dir, "real_platform_20_analysis.md")
    ok_rows = [r for r in rows if r.get("algorithm") in evaluator.ALGORITHMS]
    summary_rows = evaluator.summarize(ok_rows)

    evaluator.write_csv(detail_csv, rows)
    evaluator.write_csv(summary_csv, summary_rows)
    build_analysis_report(report_md, args, images, rows, summary_rows, detail_csv, summary_csv, selected_path)

    evaluator.print_summary(summary_rows)
    print(f"\nSelected images: {selected_path}")
    print(f"Detail CSV:      {detail_csv}")
    print(f"Summary CSV:     {summary_csv}")
    print(f"Analysis report: {report_md}")
    print(f"Upload images:   {upload_dir}")
    print(f"Received images: {received_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the fixed 20-image real Telegram platform test and generate an analysis report."
    )
    parser.add_argument("--env-file", default=".env", help="path to .env with Telegram settings")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="BOSSbase directory or zip")
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES, help="must be 20 for the standard test")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="output directory")
    parser.add_argument("--Qo", type=int, default=caca.DEFAULT_QO)
    parser.add_argument("--Qc", type=int, default=caca.DEFAULT_QC)
    parser.add_argument("--payload", type=float, default=caca.DEFAULT_PAYLOAD_BPNZAC)
    parser.add_argument("--constraint-height", type=int, default=caca.DEFAULT_CONSTRAINT_HEIGHT)
    parser.add_argument("--key", type=int, default=1234)
    parser.add_argument("--search-min", type=int, default=caca.DEFAULT_SEARCH_MIN)
    parser.add_argument("--search-max", type=int, default=caca.DEFAULT_SEARCH_MAX)
    parser.add_argument("--original-search-radius", type=int, default=50)
    parser.add_argument("--max-message-bytes", type=int, default=64)
    parser.add_argument("--robust-repeat", type=int, default=63)
    parser.add_argument("--robust-min-abs", type=int, default=6)
    parser.add_argument("--telegram-token", help="override TELEGRAM_BOT_TOKEN")
    parser.add_argument("--telegram-chat-id", help="override TELEGRAM_CHAT_ID")
    parser.add_argument("--telegram-timeout", type=int, default=120)
    parser.add_argument("--telegram-sleep", type=float, default=0.5)
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_images != DEFAULT_NUM_IMAGES:
        raise ValueError("--num-images must stay 20 for this fixed experiment script")
    if args.search_min > args.search_max:
        raise ValueError("--search-min must be <= --search-max")
    run_experiment(args)


if __name__ == "__main__":
    main()
