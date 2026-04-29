import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

import caca_algorithm as caca
import jpeg_backend as jio


DEFAULT_OUTPUT_DIR = "caca_eval_results"
DEFAULT_DATASET = "BOSSbase_1.01.zip"
ALGORITHMS = ("original", "caca")
ROBUST_AC_POSITIONS = (
    (0, 1), (1, 0), (1, 1),
    (0, 2), (2, 0), (1, 2), (2, 1),
    (2, 2), (0, 3), (3, 0), (1, 3), (3, 1),
)


def parse_env_line(line):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None, None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_env_file(path):
    if not path or not os.path.exists(path):
        return {}
    values = {}
    with open(path, "r") as f:
        for line in f:
            key, value = parse_env_line(line)
            if key:
                values[key] = value
    return values


def validate_telegram_config(token, chat_id):
    if not token or not chat_id:
        raise ValueError(
            "Telegram config is missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env, "
            "or pass --telegram-token and --telegram-chat-id."
        )
    if ":" not in token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN looks incomplete. It must include the numeric bot id and ':' prefix, "
            "for example 123456789:AA..."
        )
    bot_id, secret = token.split(":", 1)
    if not bot_id.isdigit() or not secret:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN format is invalid. Expected something like 123456789:AA..."
        )
    return token, chat_id


def telegram_preflight_check(token, chat_id, timeout=30):
    me = telegram_request_json(
        telegram_api_url(token, "getMe"),
        data={},
        timeout=timeout,
    )
    chat = telegram_request_json(
        telegram_api_url(token, "getChat"),
        data={"chat_id": chat_id},
        timeout=timeout,
    )

    bot_id = str(me.get("id", ""))
    target_id = str(chat.get("id", ""))
    chat_type = str(chat.get("type", ""))
    if bot_id and target_id and bot_id == target_id and chat_type == "private":
        raise ValueError(
            "TELEGRAM_CHAT_ID 当前指向 bot 自己。Telegram 不允许 bot 给 bot 发消息。"
            "请把 TELEGRAM_CHAT_ID 改成你自己的用户 chat id，或者改成已拉入并授权该 bot 的群/频道 id。"
        )

    telegram_request_json(
        telegram_api_url(token, "sendChatAction"),
        data={"chat_id": chat_id, "action": "typing"},
        timeout=timeout,
    )


@dataclass
class ImageContext:
    image_path: str
    base_name: str
    jpeg_qo: object
    jpeg_qc: object
    O: np.ndarray
    C: np.ndarray
    mo_full: np.ndarray
    mc_full: np.ndarray


def prepare_dataset(dataset, output_dir):
    if not dataset.lower().endswith(".zip"):
        return dataset
    if not os.path.exists(dataset):
        raise ValueError(f"dataset zip not found: {dataset}")

    name = os.path.splitext(os.path.basename(dataset))[0]
    extract_dir = os.path.join(output_dir, "datasets", name)
    marker = os.path.join(extract_dir, ".extracted")
    if os.path.exists(marker):
        return extract_dir

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(dataset, "r") as zf:
        zf.extractall(extract_dir)
    with open(marker, "w") as f:
        f.write(dataset + "\n")
    return extract_dir


def discover_images(dataset_dir, limit):
    patterns = ("*.pgm", "*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(dataset_dir, "**", pattern), recursive=True))
    images = sorted(set(images))
    if limit is not None and limit > 0:
        images = images[:limit]
    return images


def load_context(image_path, qo, qc, temp_dir):
    base = os.path.splitext(os.path.basename(image_path))[0]
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read image: {image_path}")

    qo_path = os.path.join(temp_dir, f"{base}_Qo.jpg")
    qc_path = os.path.join(temp_dir, f"{base}_Qc.jpg")
    jpeg_qo = caca.quantize_to_jpeg(gray, qo, qo_path)
    jpeg_qc = caca.quantize_to_jpeg(gray, qc, qc_path)

    O = jpeg_qo.coef_arrays[0].astype(np.int32)
    mo_full = caca.expand_quant_table(jpeg_qo.quant_tables[0], O.shape)
    mc_full = caca.expand_quant_table(jpeg_qc.quant_tables[0], O.shape)
    C = caca.paper_channel_model(O, mo_full, mc_full)
    return ImageContext(image_path, base, jpeg_qo, jpeg_qc, O, C, mo_full, mc_full)


def deterministic_message(base_name, payload_bpnzac, C, max_message_bytes, key):
    nzac = int(np.count_nonzero(caca.ac_nonzero_mask(C)))
    capacity_bits = int(np.floor(float(payload_bpnzac) * nzac))
    payload_bytes = max(0, (capacity_bits - caca.HEADER_BITS) // 8)
    if max_message_bytes is not None and max_message_bytes > 0:
        payload_bytes = min(payload_bytes, int(max_message_bytes))
    if payload_bytes <= 0:
        raise ValueError(f"payload capacity too small: {capacity_bits} bits over {nzac} nzAC coefficients")

    digest = hashlib.sha256(f"{base_name}:{int(key)}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big") or 1
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=payload_bytes, dtype=np.uint8).tobytes()


def original_inverse_search(O, S, mo_full, mc_full, search_radius):
    I = O.astype(np.int32).copy()
    found = caca.paper_channel_model(I, mo_full, mc_full) == S
    search_values = [0]
    for alpha in range(1, int(search_radius) + 1):
        search_values.extend([alpha, -alpha])

    for alpha in search_values:
        if found.all():
            break
        candidate = O + alpha
        mapped = caca.paper_channel_model(candidate, mo_full, mc_full)
        match = (mapped == S) & (~found)
        I[match] = candidate[match]
        found[match] = True

    if not found.all():
        lower, upper = caca.build_feasible_space(O, S, mo_full, mc_full)
        center = np.rint((lower.astype(np.float64) + upper.astype(np.float64)) / 2.0).astype(np.int32)
        I[~found] = (O + center)[~found]
    return I.astype(np.int32), (I - O).astype(np.int32), int(np.count_nonzero(~found))


def write_coefficients(template_jpeg, coeff, path):
    jpeg = template_jpeg
    jpeg.coef_arrays[0] = coeff.astype(jpeg.coef_arrays[0].dtype)
    jio.write(jpeg, path)


def simulate_upload(input_path, output_path, qc):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot decode upload candidate: {input_path}")
    if not cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, int(qc)]):
        raise ValueError(f"cannot write simulated upload image: {output_path}")


def telegram_api_url(token, method):
    return f"https://api.telegram.org/bot{token}/{method}"


def telegram_request_json(url, data=None, files=None, timeout=120):
    if files:
        boundary = f"----caca{int(time.time() * 1000000)}"
        body = bytearray()
        for key, value in data.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")
        for key, path in files.items():
            filename = os.path.basename(path)
            with open(path, "rb") as f:
                payload = f.read()
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode("utf-8")
            )
            body.extend(b"Content-Type: image/jpeg\r\n\r\n")
            body.extend(payload)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        request = urllib.request.Request(
            url,
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
    else:
        encoded = urllib.parse.urlencode(data or {}).encode("utf-8")
        request = urllib.request.Request(url, data=encoded, method="POST")

    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise ValueError(f"Telegram API error: {result}")
    return result["result"]


def telegram_send_photo_and_download(input_path, output_path, token, chat_id, caption="", timeout=120):
    sent = telegram_request_json(
        telegram_api_url(token, "sendPhoto"),
        data={"chat_id": chat_id, "caption": caption},
        files={"photo": input_path},
        timeout=timeout,
    )
    photos = sent.get("photo") or []
    if not photos:
        raise ValueError(f"Telegram did not return photo sizes for {input_path}")
    best = max(photos, key=lambda item: (item.get("file_size") or 0, item.get("width") or 0, item.get("height") or 0))
    file_info = telegram_request_json(
        telegram_api_url(token, "getFile"),
        data={"file_id": best["file_id"]},
        timeout=timeout,
    )
    file_path = file_info["file_path"]
    download_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    escaped_path = urllib.parse.quote(file_path)
    download_url_escaped = f"https://api.telegram.org/file/bot{token}/{escaped_path}"
    return telegram_download_file(
        [download_url, download_url_escaped],
        output_path,
        timeout=timeout,
    )


def telegram_download_file(urls, output_path, timeout=120, retries=2, sleep=1.0):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
        "Accept": "*/*",
    }
    last_exc = None
    for attempt in range(retries):
        for url in urls:
            try:
                request = urllib.request.Request(url, headers=headers, method="GET")
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                with open(output_path, "wb") as f:
                    f.write(payload)
                return output_path
            except urllib.error.HTTPError as exc:
                last_exc = exc
            except Exception as exc:
                last_exc = exc
        if attempt < retries - 1:
            time.sleep(sleep)
    if last_exc:
        raise last_exc
    raise ValueError("failed to download Telegram file")


def find_received_image(received_dir, candidate_name):
    stem = os.path.splitext(candidate_name)[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".pgm"):
        path = os.path.join(received_dir, stem + ext)
        if os.path.exists(path):
            return path
    path = os.path.join(received_dir, candidate_name)
    return path if os.path.exists(path) else None


def bit_error_rate(reference_bits, extracted_bits):
    n = min(len(reference_bits), len(extracted_bits))
    if n == 0:
        return 1.0
    mismatches = int(np.count_nonzero(reference_bits[:n] != extracted_bits[:n]))
    mismatches += abs(len(reference_bits) - len(extracted_bits))
    return mismatches / max(len(reference_bits), len(extracted_bits))


def extract_ber(received_dct, reference_bits, height, key):
    try:
        extracted = caca.stc_extract_coefficients(received_dct, len(reference_bits), height, key)
    except Exception:
        return 1.0
    return bit_error_rate(reference_bits, extracted)


def robust_positions(shape, num_bits, repeat, key):
    h, w = shape
    positions = []
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            for u, v in ROBUST_AC_POSITIONS:
                yy = y + u
                xx = x + v
                if yy < h and xx < w:
                    positions.append((yy, xx))

    needed = int(num_bits) * int(repeat)
    if len(positions) < needed:
        raise ValueError(f"robust capacity too small: need {needed} positions, got {len(positions)}")

    rng = np.random.default_rng(int(key) ^ 0xCACA2026)
    order = rng.permutation(len(positions))[:needed]
    return np.asarray([positions[i] for i in order], dtype=np.int64).reshape(int(num_bits), int(repeat), 2)


def nearest_coeff_with_parity(value, bit, min_abs):
    value = int(value)
    bit = int(bit) & 1
    min_abs = int(min_abs)
    sign = -1 if value < 0 else 1
    if value == 0:
        sign = 1 if bit else -1

    center = sign * max(abs(value), min_abs)
    candidates = []
    radius = max(8, min_abs + 4)
    for delta in range(-radius, radius + 1):
        candidate = center + delta
        if candidate == 0 or abs(candidate) < min_abs:
            continue
        if (candidate & 1) == bit:
            candidates.append(candidate)
    if not candidates:
        candidate = sign * min_abs
        if (candidate & 1) != bit:
            candidate += sign
        return int(candidate)
    return int(min(candidates, key=lambda c_: (abs(c_ - value), abs(c_))))


def robust_repetition_embed(channel_dct, reference_bits, repeat, key, min_abs):
    stego = channel_dct.astype(np.int32).copy()
    positions = robust_positions(stego.shape, len(reference_bits), repeat, key)
    changed = 0
    for bit_idx, bit in enumerate(np.asarray(reference_bits, dtype=np.uint8)):
        for r, c_ in positions[bit_idx]:
            old = int(stego[r, c_])
            new = nearest_coeff_with_parity(old, int(bit), min_abs)
            if new != old:
                changed += 1
                stego[r, c_] = new
    stats = caca.STCStats(
        message_bits=int(len(reference_bits)),
        usable_coefficients=int(len(reference_bits) * repeat),
        changed_coefficients=int(changed),
        total_cost=0.0,
    )
    return stego, stats


def robust_repetition_extract(received_dct, num_bits, repeat, key):
    positions = robust_positions(received_dct.shape, num_bits, repeat, key)
    bits = np.zeros(int(num_bits), dtype=np.uint8)
    for bit_idx in range(int(num_bits)):
        values = received_dct[positions[bit_idx, :, 0], positions[bit_idx, :, 1]].astype(np.int32)
        parities = values & 1
        bits[bit_idx] = 1 if int(np.sum(parities)) >= (int(repeat) + 1) // 2 else 0
    return bits


def robust_extract_ber(received_dct, reference_bits, repeat, key):
    try:
        extracted = robust_repetition_extract(received_dct, len(reference_bits), repeat, key)
    except Exception:
        return 1.0
    return bit_error_rate(reference_bits, extracted)


def dct_histogram_shift(cover_dct, stego_dct, clip_value=10):
    bins = np.arange(-clip_value, clip_value + 2)
    cover = np.clip(cover_dct.ravel(), -clip_value, clip_value + 1)
    stego = np.clip(stego_dct.ravel(), -clip_value, clip_value + 1)
    h_cover, _ = np.histogram(cover, bins=bins, density=False)
    h_stego, _ = np.histogram(stego, bins=bins, density=False)
    h_cover = h_cover / max(1, np.sum(h_cover))
    h_stego = h_stego / max(1, np.sum(h_stego))
    return float(np.sum(np.abs(h_cover - h_stego)))


def residual_energy_delta(cover_dct, stego_dct, quant_table):
    cover = caca.jpeg_spatial_from_coefficients(cover_dct, quant_table)
    stego = caca.jpeg_spatial_from_coefficients(stego_dct, quant_table)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    cover_res = cv2.filter2D(cover, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT)
    stego_res = cv2.filter2D(stego, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT)
    return float(abs(np.mean(np.abs(stego_res)) - np.mean(np.abs(cover_res))))


def steganalysis_proxy_metrics(O, I, quant_table, rho_plus, rho_minus):
    adjustment = (I - O).astype(np.int32)
    finite_cost = caca.cumulative_adjustment_cost(adjustment, rho_plus, rho_minus)
    finite_cost = np.where(np.isfinite(finite_cost), finite_cost, caca.WET_COST)
    return {
        "dct_change_rate": float(np.mean(adjustment != 0)),
        "dct_l1": float(np.mean(np.abs(adjustment))),
        "dct_l2": float(np.sqrt(np.mean(adjustment.astype(np.float64) ** 2))),
        "dct_hist_l1": dct_histogram_shift(O, I),
        "residual_delta": residual_energy_delta(O, I, quant_table),
        "uniward_distortion": float(np.sum(np.clip(finite_cost, 0, caca.WET_COST))),
    }


def evaluate_image(ctx, args, upload_dir, received_dir, temp_dir):
    message = deterministic_message(ctx.base_name, args.payload, ctx.C, args.max_message_bytes, args.key)
    reference_bits = caca.encode_payload(message)
    S_original, original_stats = caca.stc_embed_coefficients(
        ctx.C,
        message,
        ctx.jpeg_qc.quant_tables[0],
        args.constraint_height,
        args.key,
        args.payload,
    )
    S_caca, caca_stats = robust_repetition_embed(
        ctx.C,
        reference_bits,
        args.robust_repeat,
        args.key,
        args.robust_min_abs,
    )

    I_original, a_original, original_unmatched = original_inverse_search(
        ctx.O, S_original, ctx.mo_full, ctx.mc_full, args.original_search_radius
    )
    I_caca, a_caca, lower, upper, caca_fallbacks = caca.solve_global_caca(
        ctx.O,
        S_caca,
        ctx.mo_full,
        ctx.mc_full,
        ctx.jpeg_qo.quant_tables[0],
        args.search_min,
        args.search_max,
    )

    rho_plus_o, rho_minus_o = caca.jpeg_uniward_costs(ctx.O, ctx.jpeg_qo.quant_tables[0])
    rows = []
    algos = {
        "original": (I_original, a_original, original_unmatched, S_original, original_stats, "stc"),
        "caca": (I_caca, a_caca, caca_fallbacks, S_caca, caca_stats, "robust"),
    }

    for algorithm, (I, adjustment, fallback_count, target_s, embed_stats, extractor) in algos.items():
        candidate_name = f"{ctx.base_name}_{algorithm}.jpg"
        upload_path = os.path.join(upload_dir, candidate_name)
        received_path = os.path.join(temp_dir, f"{ctx.base_name}_{algorithm}_received.jpg")
        write_coefficients(ctx.jpeg_qo, I, upload_path)

        if args.channel == "received":
            matched = find_received_image(received_dir, candidate_name)
            if matched is None:
                raise ValueError(f"missing received image for {candidate_name} in {received_dir}")
            received_path = matched
        elif args.channel == "simulate":
            simulate_upload(upload_path, received_path, args.Qc)
        elif args.channel == "telegram":
            received_path = os.path.join(temp_dir, f"{ctx.base_name}_{algorithm}_telegram.jpg")
            telegram_send_photo_and_download(
                upload_path,
                received_path,
                args.telegram_token,
                args.telegram_chat_id,
                caption=f"{ctx.base_name}_{algorithm}",
                timeout=args.telegram_timeout,
            )
            if args.telegram_sleep > 0:
                time.sleep(args.telegram_sleep)
        elif args.channel == "write-only":
            received_path = ""

        if args.channel == "write-only":
            real_ber = np.nan
            coeff_mismatch = np.nan
        else:
            received_dct = jio.read(received_path, quality=args.Qc).coef_arrays[0].astype(np.int32)
            if extractor == "robust":
                real_ber = robust_extract_ber(received_dct, reference_bits, args.robust_repeat, args.key)
            else:
                real_ber = extract_ber(received_dct, reference_bits, args.constraint_height, args.key)
            coeff_mismatch = float(np.mean(received_dct != target_s))

        ideal_received = caca.paper_channel_model(I, ctx.mo_full, ctx.mc_full)
        if extractor == "robust":
            ideal_ber = robust_extract_ber(ideal_received, reference_bits, args.robust_repeat, args.key)
        else:
            ideal_ber = extract_ber(ideal_received, reference_bits, args.constraint_height, args.key)
        ideal_mismatch = float(np.mean(ideal_received != target_s))

        metrics = steganalysis_proxy_metrics(ctx.O, I, ctx.jpeg_qo.quant_tables[0], rho_plus_o, rho_minus_o)
        rows.append({
            "image": os.path.basename(ctx.image_path),
            "algorithm": algorithm,
            "Qo": args.Qo,
            "Qc": args.Qc,
            "payload_bpnzac": args.payload,
            "message_bytes": len(message),
            "message_bits_with_header": len(reference_bits),
            "usable_coefficients": embed_stats.usable_coefficients,
            "stc_changed_coefficients": embed_stats.changed_coefficients,
            "ideal_ber": ideal_ber,
            "real_ber": real_ber,
            "ideal_coeff_mismatch": ideal_mismatch,
            "real_coeff_mismatch": coeff_mismatch,
            "fallback_or_unmatched_coefficients": fallback_count,
            "upload_path": upload_path,
            "received_path": received_path,
            **metrics,
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for algorithm in ALGORITHMS:
        selected = [r for r in rows if r["algorithm"] == algorithm]
        if not selected:
            continue
        numeric_keys = [
            "ideal_ber", "real_ber", "ideal_coeff_mismatch", "real_coeff_mismatch",
            "dct_change_rate", "dct_l1", "dct_l2", "dct_hist_l1",
            "residual_delta", "uniward_distortion", "fallback_or_unmatched_coefficients",
        ]
        row = {"algorithm": algorithm, "num_images": len(selected)}
        for key in numeric_keys:
            values = np.array([float(r[key]) for r in selected], dtype=np.float64)
            row[f"mean_{key}"] = float(np.nanmean(values))
        summary.append(row)
    return summary


def print_summary(summary_rows):
    print("\nEvaluation summary")
    print("algorithm | images | real_BER | ideal_BER | coeff_mismatch | change_rate | uniward_distortion")
    print("-" * 100)
    for row in summary_rows:
        print(
            f"{row['algorithm']:<9} | {row['num_images']:>6} | "
            f"{row['mean_real_ber']:.6f} | {row['mean_ideal_ber']:.6f} | "
            f"{row['mean_real_coeff_mismatch']:.6f} | {row['mean_dct_change_rate']:.6f} | "
            f"{row['mean_uniward_distortion']:.6f}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate real-channel BER and steganalysis proxy metrics for original inverse search vs CACA."
    )
    parser.add_argument("--env-file", default=".env", help="path to .env file with Telegram settings")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="cover image directory or .zip archive")
    parser.add_argument("--limit", type=int, default=10, help="number of images to evaluate")
    parser.add_argument("--Qo", type=int, default=caca.DEFAULT_QO, help="original JPEG quality")
    parser.add_argument("--Qc", type=int, default=caca.DEFAULT_QC, help="channel JPEG quality")
    parser.add_argument("--payload", type=float, default=caca.DEFAULT_PAYLOAD_BPNZAC, help="payload in bpnzac")
    parser.add_argument("--constraint-height", type=int, default=caca.DEFAULT_CONSTRAINT_HEIGHT, help="STC height")
    parser.add_argument("--key", type=int, default=1234, help="STC parity-check key")
    parser.add_argument("--search-min", type=int, default=caca.DEFAULT_SEARCH_MIN, help="CACA search lower bound")
    parser.add_argument("--search-max", type=int, default=caca.DEFAULT_SEARCH_MAX, help="CACA search upper bound")
    parser.add_argument("--original-search-radius", type=int, default=50, help="original inverse alpha search radius")
    parser.add_argument("--max-message-bytes", type=int, default=64, help="cap generated message bytes for speed")
    parser.add_argument("--robust-repeat", type=int, default=63, help="repetitions per payload bit for CACA robust mode")
    parser.add_argument("--robust-min-abs", type=int, default=6, help="minimum absolute channel coefficient magnitude for CACA robust mode")
    parser.add_argument("--channel", choices=("telegram", "received", "write-only", "simulate"), default="telegram")
    parser.add_argument("--received-dir", help="directory with downloaded/received images named <base>_<algorithm>.*")
    parser.add_argument("--telegram-token", help="Telegram bot token")
    parser.add_argument("--telegram-chat-id", help="Telegram chat id")
    parser.add_argument("--telegram-timeout", type=int, default=120, help="Telegram API timeout in seconds")
    parser.add_argument("--telegram-sleep", type=float, default=0.5, help="sleep seconds after each Telegram upload")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for CSVs and upload candidates")
    parser.add_argument("--keep-temp", action="store_true", help="keep temporary simulated-channel files")
    return parser


def main():
    args = build_parser().parse_args()
    env_values = load_env_file(args.env_file)
    args.telegram_token = args.telegram_token or os.getenv("TELEGRAM_BOT_TOKEN") or env_values.get("TELEGRAM_BOT_TOKEN")
    args.telegram_chat_id = args.telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID") or env_values.get("TELEGRAM_CHAT_ID")

    if args.search_min > args.search_max:
        raise ValueError("--search-min must be <= --search-max")
    if args.channel == "received" and not args.received_dir:
        raise ValueError("--received-dir is required when --channel received")
    if args.channel == "telegram":
        validate_telegram_config(args.telegram_token, args.telegram_chat_id)
        telegram_preflight_check(args.telegram_token, args.telegram_chat_id, timeout=min(args.telegram_timeout, 30))

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dir = prepare_dataset(args.dataset, args.output_dir)
    images = discover_images(dataset_dir, args.limit)
    if not images:
        raise ValueError(f"no images found in {dataset_dir}")

    upload_dir = os.path.join(args.output_dir, "upload_candidates")
    os.makedirs(upload_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="caca_eval_", dir=args.output_dir if os.path.isdir(args.output_dir) else None)

    rows = []
    try:
        for image_path in tqdm(images, desc="Evaluating"):
            try:
                ctx = load_context(image_path, args.Qo, args.Qc, temp_dir)
                rows.extend(evaluate_image(ctx, args, upload_dir, args.received_dir, temp_dir))
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

    ok_rows = [r for r in rows if r["algorithm"] in ALGORITHMS]
    detail_csv = os.path.join(args.output_dir, "detail_results.csv")
    summary_csv = os.path.join(args.output_dir, "summary_results.csv")
    write_csv(detail_csv, rows)
    summary_rows = summarize(ok_rows)
    write_csv(summary_csv, summary_rows)
    print_summary(summary_rows)
    print(f"\nDetailed results: {detail_csv}")
    print(f"Summary results:  {summary_csv}")
    print(f"Upload candidates: {upload_dir}")


if __name__ == "__main__":
    main()
