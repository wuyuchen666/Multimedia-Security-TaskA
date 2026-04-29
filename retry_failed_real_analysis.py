import argparse
import csv
import os
import shutil
import tempfile

import numpy as np
from tqdm import tqdm

import evaluate_caca_ber as evaluator
import run_20_image_real_analysis as real20


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_selected_images(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def failed_images(rows):
    return sorted({row["image"] for row in rows if row.get("algorithm") == "error"})


def remove_rows_for_images(rows, image_names):
    image_names = set(image_names)
    return [row for row in rows if row.get("image") not in image_names]


def numeric_arg_from_rows(rows, key, default, cast):
    for row in rows:
        value = row.get(key)
        if value not in (None, "", "nan"):
            return cast(value)
    return default


def error_row(image_path, args, exc):
    return {
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
    }


def retry_failed(args):
    detail_csv = os.path.join(args.output_dir, "real_platform_20_detail.csv")
    summary_csv = os.path.join(args.output_dir, "real_platform_20_summary.csv")
    report_md = os.path.join(args.output_dir, "real_platform_20_analysis.md")
    selected_path = os.path.join(args.output_dir, "selected_20_images.txt")

    rows = read_csv(detail_csv)
    selected_images = read_selected_images(selected_path)
    failed = failed_images(rows)
    if not failed:
        print("No failed samples found. Rebuilding summary/report only.")

    args.Qo = numeric_arg_from_rows(rows, "Qo", args.Qo, int)
    args.Qc = numeric_arg_from_rows(rows, "Qc", args.Qc, int)
    args.payload = numeric_arg_from_rows(rows, "payload_bpnzac", args.payload, float)

    token, chat_id = real20.resolve_telegram_config(args.env_file, args.telegram_token, args.telegram_chat_id)
    evaluator.telegram_preflight_check(token, chat_id, timeout=min(args.telegram_timeout, 30))
    eval_args = real20.build_eval_args(args, token, chat_id)

    image_by_name = {os.path.basename(path): path for path in selected_images}
    missing = [name for name in failed if name not in image_by_name]
    if missing:
        raise ValueError(f"failed images are not in selected_20_images.txt: {missing}")

    upload_dir = os.path.join(args.output_dir, "upload_candidates")
    received_dir = os.path.join(args.output_dir, "telegram_received")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(received_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="retry_failed_", dir=args.output_dir)

    new_rows = []
    try:
        for image_name in tqdm(failed, desc="Retrying failed samples"):
            image_path = image_by_name[image_name]
            for attempt in range(1, args.retries + 1):
                try:
                    ctx = evaluator.load_context(image_path, args.Qo, args.Qc, temp_dir)
                    new_rows.extend(evaluator.evaluate_image(ctx, eval_args, upload_dir, None, received_dir))
                    break
                except Exception as exc:
                    if attempt == args.retries:
                        new_rows.append(error_row(image_path, args, exc))
                    elif args.retry_sleep > 0:
                        import time

                        time.sleep(args.retry_sleep)
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)

    merged_rows = remove_rows_for_images(rows, failed) + new_rows
    ok_rows = [row for row in merged_rows if row.get("algorithm") in evaluator.ALGORITHMS]
    summary_rows = evaluator.summarize(ok_rows)

    evaluator.write_csv(detail_csv, merged_rows)
    evaluator.write_csv(summary_csv, summary_rows)
    real20.build_analysis_report(report_md, args, selected_images, merged_rows, summary_rows, detail_csv, summary_csv, selected_path)

    evaluator.print_summary(summary_rows)
    print(f"\nRetried images: {', '.join(failed) if failed else '(none)'}")
    print(f"Detail CSV:      {detail_csv}")
    print(f"Summary CSV:     {summary_csv}")
    print(f"Analysis report: {report_md}")


def build_parser():
    parser = argparse.ArgumentParser(description="Retry failed samples in the 20-image real-platform analysis.")
    parser.add_argument("--output-dir", default=real20.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--dataset", default=real20.DEFAULT_DATASET)
    parser.add_argument("--Qo", type=int, default=95)
    parser.add_argument("--Qc", type=int, default=80)
    parser.add_argument("--payload", type=float, default=0.2)
    parser.add_argument("--constraint-height", type=int, default=10)
    parser.add_argument("--key", type=int, default=1234)
    parser.add_argument("--search-min", type=int, default=-2)
    parser.add_argument("--search-max", type=int, default=2)
    parser.add_argument("--original-search-radius", type=int, default=50)
    parser.add_argument("--max-message-bytes", type=int, default=64)
    parser.add_argument("--telegram-token")
    parser.add_argument("--telegram-chat-id")
    parser.add_argument("--telegram-timeout", type=int, default=120)
    parser.add_argument("--telegram-sleep", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.search_min > args.search_max:
        raise ValueError("--search-min must be <= --search-max")
    retry_failed(args)


if __name__ == "__main__":
    main()
