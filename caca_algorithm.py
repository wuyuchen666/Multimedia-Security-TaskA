import argparse
import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

import jpeg_backend as jio


TEMP_DIR = "./temp_caca"
WET_COST = 1e13
SIGMA = 2.0**-50
HEADER_BITS = 32
DEFAULT_QO = 95
DEFAULT_QC = 80
DEFAULT_PAYLOAD_BPNZAC = 0.2
DEFAULT_CONSTRAINT_HEIGHT = 10
DEFAULT_SEARCH_MIN = -2
DEFAULT_SEARCH_MAX = 2


@dataclass
class STCStats:
    message_bits: int
    usable_coefficients: int
    changed_coefficients: int
    total_cost: float


@dataclass
class CACAResult:
    cover_dct: np.ndarray
    channel_dct: np.ndarray
    stego_target_dct: np.ndarray
    intermediate_dct: np.ndarray
    adjustment: np.ndarray
    feasible_lower: np.ndarray
    feasible_upper: np.ndarray
    recovered_message: bytes
    stc_stats: STCStats
    truncated_search_fallbacks: int
    paper_channel_ok: bool
    paper_mismatch_rate: float


def bytes_to_bits(data):
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big").astype(np.uint8)


def bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 8:
        bits = np.pad(bits, (0, 8 - bits.size % 8))
    return np.packbits(bits, bitorder="big").tobytes()


def encode_payload(message):
    if len(message) >= 2**32:
        raise ValueError("message is too large for a 32-bit payload header")
    return bytes_to_bits(len(message).to_bytes(4, "big") + message)


def decode_payload(bits, forced_length=None):
    if forced_length is None:
        payload_len = int.from_bytes(bits_to_bytes(bits[:HEADER_BITS]), "big")
        start = HEADER_BITS
    else:
        payload_len = int(forced_length)
        start = 0
    end = start + payload_len * 8
    if len(bits) < end:
        raise ValueError("not enough bits to decode payload")
    return bits_to_bytes(bits[start:end])


def read_message(args):
    if args.message is None and args.message_file is None:
        raise ValueError("provide --message or --message-file")
    if args.message is not None and args.message_file is not None:
        raise ValueError("use only one of --message and --message-file")
    if args.message_file:
        with open(args.message_file, "rb") as f:
            return f.read()
    return args.message.encode("utf-8")


def quantize_to_jpeg(gray_img, quality, path):
    if not cv2.imwrite(path, gray_img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
        raise ValueError(f"cannot write temporary JPEG: {path}")
    return jio.read(path, quality=quality)


def expand_quant_table(table, shape):
    h, w = shape
    if h % 8 or w % 8:
        raise ValueError("DCT coefficient shape must be divisible by 8")
    return np.tile(table, (h // 8, w // 8)).astype(np.float64)


def paper_channel_model(dct_coeff, mo_full, mc_full):
    return np.rint(dct_coeff * mo_full / mc_full).astype(np.int32)


def ac_nonzero_mask(dct_coeff):
    mask = np.ones_like(dct_coeff, dtype=bool)
    mask[0::8, 0::8] = False
    return mask & (dct_coeff != 0)


def jpeg_spatial_from_coefficients(dct_coeff, quant_table):
    h, w = dct_coeff.shape
    spatial = np.empty((h, w), dtype=np.float64)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = dct_coeff[y:y + 8, x:x + 8].astype(np.float64) * quant_table
            spatial[y:y + 8, x:x + 8] = cv2.idct(block) + 128.0
    return spatial


def db8_filters():
    low = np.array([
        -0.00011747678412476953, 0.0006754494059985568,
        -0.00039174037337694705, -0.004870352993451574,
        0.008746094047405777, 0.013981027917398282,
        -0.044088253930794755, -0.017369301001807547,
        0.12874742662047847, 0.0004724845739124,
        -0.2840155429615824, -0.015829105256349306,
        0.5853546836542067, 0.6756307362972898,
        0.31287159091429995, 0.0544158422431072,
    ], dtype=np.float64)
    high = low[::-1].copy()
    high[::2] *= -1
    return [np.outer(high, low), np.outer(low, high), np.outer(high, high)]


def idct_basis():
    out = np.empty((8, 8, 8, 8), dtype=np.float64)
    for u in range(8):
        for v in range(8):
            block = np.zeros((8, 8), dtype=np.float64)
            block[u, v] = 1.0
            out[u, v] = cv2.idct(block)
    return out


def local_wavelet_impact_cost(y, x, impact_patch, denominators, filters):
    h, w = denominators[0].shape
    radius = max(f.shape[0] for f in filters) - 1
    top = max(0, y - radius)
    left = max(0, x - radius)
    bottom = min(h, y + 8 + radius)
    right = min(w, x + 8 + radius)
    canvas = np.zeros((bottom - top, right - left), dtype=np.float64)
    canvas[y - top:y - top + 8, x - left:x - left + 8] = impact_patch

    cost = 0.0
    for denom, filt in zip(denominators, filters):
        impact = cv2.filter2D(canvas, cv2.CV_64F, filt, borderType=cv2.BORDER_CONSTANT)
        cost += float(np.sum(np.abs(impact) / denom[top:bottom, left:right]))
    return cost


def jpeg_uniward_costs(dct_coeff, quant_table):
    dct_coeff = dct_coeff.astype(np.int32)
    h, w = dct_coeff.shape
    spatial = jpeg_spatial_from_coefficients(dct_coeff, quant_table)
    filters = db8_filters()
    residuals = [cv2.filter2D(spatial, cv2.CV_64F, f, borderType=cv2.BORDER_REFLECT) for f in filters]
    denominators = [np.abs(r) + SIGMA for r in residuals]
    bases = idct_basis()

    rho = np.full((h, w), WET_COST, dtype=np.float64)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            for u in range(8):
                for v in range(8):
                    if u == 0 and v == 0:
                        continue
                    patch = bases[u, v] * float(quant_table[u, v])
                    rho[y + u, x + v] = local_wavelet_impact_cost(y, x, patch, denominators, filters)

    rho = np.clip(rho, 1e-6, WET_COST)
    rho_plus = rho.copy()
    rho_minus = rho.copy()
    mask = ac_nonzero_mask(dct_coeff)
    rho_plus[~mask] = WET_COST
    rho_minus[~mask] = WET_COST
    rho_minus[dct_coeff == 1] = WET_COST
    rho_plus[dct_coeff == -1] = WET_COST
    return rho_plus, rho_minus


def stc_column_masks(num_columns, height, key):
    rng = np.random.default_rng(int(key))
    masks = rng.integers(1, 1 << height, size=num_columns, dtype=np.uint32)
    masks |= 1
    return masks.astype(np.int64)


def stc_syndrome(parity, message_bits, height, key):
    n = int(message_bits) + height - 1
    if len(parity) < n:
        raise ValueError(f"need {n} coefficients, got {len(parity)}")
    masks = stc_column_masks(n, height, key)
    syndrome = np.zeros(int(message_bits), dtype=np.uint8)
    for j in range(n):
        if parity[j] & 1:
            mask = int(masks[j])
            for k in range(height):
                row = j + k
                if row >= message_bits:
                    break
                syndrome[row] ^= (mask >> k) & 1
    return syndrome


def stc_embed_parity(cover_parity, message_bits, flip_cost, height=DEFAULT_CONSTRAINT_HEIGHT, key=1234):
    message_bits = np.asarray(message_bits, dtype=np.uint8)
    m = len(message_bits)
    n = m + height - 1
    if len(cover_parity) < n:
        raise ValueError(f"capacity too small: need {n} usable coefficients, got {len(cover_parity)}")

    cover_parity = np.asarray(cover_parity[:n], dtype=np.uint8)
    flip_cost = np.asarray(flip_cost[:n], dtype=np.float64)
    if np.any(flip_cost >= WET_COST):
        raise ValueError("selected coefficients include wet positions")

    target = message_bits ^ stc_syndrome(cover_parity, m, height, key)
    masks = stc_column_masks(n, height, key)
    costs = {0: 0.0}
    parents = []

    for j in range(n):
        expected = int(target[j]) if j < m else 0
        new_costs = {}
        new_parent = {}
        for state, base_cost in costs.items():
            for flip in (0, 1):
                state2 = state ^ (int(masks[j]) if flip else 0)
                if (state2 & 1) != expected:
                    continue
                next_state = state2 >> 1
                candidate_cost = base_cost + (float(flip_cost[j]) if flip else 0.0)
                if candidate_cost < new_costs.get(next_state, float("inf")):
                    new_costs[next_state] = candidate_cost
                    new_parent[next_state] = (state, flip)
        if not new_costs:
            raise ValueError("STC embedding failed: no finite trellis path")
        costs = new_costs
        parents.append(new_parent)

    final_state = min(costs, key=costs.get)

    flips = np.zeros(n, dtype=np.uint8)
    state = final_state
    for j in range(n - 1, -1, -1):
        state, flip = parents[j][state]
        flips[j] = flip
    return flips, float(costs[final_state])


def stc_extract_parity(stego_parity, message_bits, height=DEFAULT_CONSTRAINT_HEIGHT, key=1234):
    return stc_syndrome(np.asarray(stego_parity, dtype=np.uint8), int(message_bits), height, key)


def choose_parity_flip_direction(coeff, rho_plus, rho_minus):
    plus_ok = rho_plus < WET_COST and coeff != -1
    minus_ok = rho_minus < WET_COST and coeff != 1
    if not plus_ok and not minus_ok:
        return 0, WET_COST
    if plus_ok and (not minus_ok or rho_plus <= rho_minus):
        return 1, float(rho_plus)
    return -1, float(rho_minus)


def stc_embed_coefficients(channel_dct, message, quant_table, height, key, payload_bpnzac):
    bits = encode_payload(message)
    rho_plus, rho_minus = jpeg_uniward_costs(channel_dct, quant_table)
    rows, cols = np.where(ac_nonzero_mask(channel_dct))
    payload_capacity = int(np.floor(float(payload_bpnzac) * len(rows)))
    if len(bits) > payload_capacity:
        raise ValueError(
            f"payload too large for {payload_bpnzac:.4f} bpnzac: "
            f"need {len(bits)} bits, capacity is {payload_capacity} bits over {len(rows)} nzAC coefficients"
        )
    use_rows, use_cols, directions, costs = [], [], [], []

    for r, c in zip(rows, cols):
        direction, cost = choose_parity_flip_direction(
            int(channel_dct[r, c]), float(rho_plus[r, c]), float(rho_minus[r, c])
        )
        if direction != 0 and cost < WET_COST:
            use_rows.append(r)
            use_cols.append(c)
            directions.append(direction)
            costs.append(cost)

    needed = len(bits) + height - 1
    if len(use_rows) < needed:
        raise ValueError(f"capacity too small: need {needed} usable coefficients, got {len(use_rows)}")

    use_rows = np.asarray(use_rows[:needed], dtype=np.int64)
    use_cols = np.asarray(use_cols[:needed], dtype=np.int64)
    directions = np.asarray(directions[:needed], dtype=np.int32)
    costs = np.asarray(costs[:needed], dtype=np.float64)
    cover_values = channel_dct[use_rows, use_cols].astype(np.int32)
    cover_parity = (cover_values & 1).astype(np.uint8)
    flips, total_cost = stc_embed_parity(cover_parity, bits, costs, height=height, key=key)

    stego = channel_dct.astype(np.int32).copy()
    changed = flips.astype(bool)
    stego[use_rows[changed], use_cols[changed]] += directions[changed]

    stats = STCStats(
        message_bits=int(len(bits)),
        usable_coefficients=int(len(use_rows)),
        changed_coefficients=int(np.count_nonzero(flips)),
        total_cost=float(total_cost),
    )
    return stego, stats


def stc_extract_coefficients(stego_dct, payload_bits, height, key):
    rows, cols = np.where(ac_nonzero_mask(stego_dct))
    needed = int(payload_bits) + height - 1
    if len(rows) < needed:
        raise ValueError(f"not enough usable coefficients to extract {payload_bits} bits")
    values = stego_dct[rows[:needed], cols[:needed]].astype(np.int32)
    parity = (values & 1).astype(np.uint8)
    return stc_extract_parity(parity, int(payload_bits), height=height, key=key)


def extract_message_from_stego(stego_dct, height, key, forced_length=None):
    if forced_length is None:
        header = stc_extract_coefficients(stego_dct, HEADER_BITS, height, key)
        payload_len = int.from_bytes(bits_to_bytes(header), "big")
        bits = stc_extract_coefficients(stego_dct, HEADER_BITS + payload_len * 8, height, key)
        return decode_payload(bits)
    bits = stc_extract_coefficients(stego_dct, int(forced_length) * 8, height, key)
    return decode_payload(bits, forced_length=forced_length)


def build_feasible_space(O, S, mo_full, mc_full):
    d = mc_full / mo_full
    lower = np.ceil(d * (S - 0.5) - O).astype(np.int32)
    upper = (np.ceil(d * (S + 0.5) - O) - 1).astype(np.int32)
    return lower, upper


def cumulative_adjustment_cost(adjustment, rho_plus, rho_minus):
    cost = np.zeros_like(adjustment, dtype=np.float64)
    pos = adjustment > 0
    neg = adjustment < 0
    cost[pos] = adjustment[pos] * rho_plus[pos]
    cost[neg] = (-adjustment[neg]) * rho_minus[neg]
    return cost


def solve_global_caca(O, S, mo_full, mc_full, quant_table, search_min, search_max):
    lower, upper = build_feasible_space(O, S, mo_full, mc_full)
    if np.any(lower > upper):
        bad = int(np.count_nonzero(lower > upper))
        raise ValueError(f"empty feasible solution space at {bad} coefficients")

    truncated_lower = np.maximum(lower, int(search_min))
    truncated_upper = np.minimum(upper, int(search_max))
    fallback = truncated_lower > truncated_upper
    truncated_lower[fallback] = lower[fallback]
    truncated_upper[fallback] = upper[fallback]

    rho_plus, rho_minus = jpeg_uniward_costs(O, quant_table)
    best_a = truncated_lower.copy()
    best_cost = cumulative_adjustment_cost(best_a, rho_plus, rho_minus)
    max_width = int(np.max(truncated_upper - truncated_lower))

    for offset in range(1, max_width + 1):
        candidate = truncated_lower + offset
        valid = candidate <= truncated_upper
        candidate_cost = cumulative_adjustment_cost(candidate, rho_plus, rho_minus)
        improve = valid & (candidate_cost < best_cost)
        best_a[improve] = candidate[improve]
        best_cost[improve] = candidate_cost[improve]

    return (O + best_a).astype(np.int32), best_a, lower, upper, int(np.count_nonzero(fallback))


def write_intermediate_jpeg(template_jpeg, intermediate_dct, output_path):
    template_jpeg.coef_arrays[0] = intermediate_dct.astype(template_jpeg.coef_arrays[0].dtype)
    jio.write(template_jpeg, output_path)


def load_cover_and_tables(image_path, qo, qc):
    os.makedirs(TEMP_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read image: {image_path}")
    qo_path = os.path.join(TEMP_DIR, f"{base}_Qo.jpg")
    qc_path = os.path.join(TEMP_DIR, f"{base}_Qc.jpg")
    jpeg_qo = quantize_to_jpeg(gray, qo, qo_path)
    jpeg_qc = quantize_to_jpeg(gray, qc, qc_path)
    for path in (qo_path, qc_path):
        if os.path.exists(path):
            os.remove(path)
    return jpeg_qo, jpeg_qc


def run_embed(image_path, qo, qc, message, output_path, key, height, payload_bpnzac, search_min, search_max):
    jpeg_qo, jpeg_qc = load_cover_and_tables(image_path, qo, qc)
    O = jpeg_qo.coef_arrays[0].astype(np.int32)
    mo_full = expand_quant_table(jpeg_qo.quant_tables[0], O.shape)
    mc_full = expand_quant_table(jpeg_qc.quant_tables[0], O.shape)

    C = paper_channel_model(O, mo_full, mc_full)
    S, stc_stats = stc_embed_coefficients(C, message, jpeg_qc.quant_tables[0], height, key, payload_bpnzac)
    I, adjustment, lower, upper, fallbacks = solve_global_caca(
        O, S, mo_full, mc_full, jpeg_qo.quant_tables[0], search_min, search_max
    )

    received = paper_channel_model(I, mo_full, mc_full)
    mismatch_rate = float(np.mean(received != S))
    if mismatch_rate != 0.0:
        raise ValueError(f"CACA constraint failed: mismatch rate {mismatch_rate:.8f}")

    recovered = extract_message_from_stego(received, height, key)
    write_intermediate_jpeg(jpeg_qo, I, output_path)
    return CACAResult(
        cover_dct=O,
        channel_dct=C,
        stego_target_dct=S,
        intermediate_dct=I,
        adjustment=adjustment,
        feasible_lower=lower,
        feasible_upper=upper,
        recovered_message=recovered,
        stc_stats=stc_stats,
        truncated_search_fallbacks=fallbacks,
        paper_channel_ok=True,
        paper_mismatch_rate=mismatch_rate,
    )


def load_qc_table_from_image(image_path, qc):
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"cannot read image: {image_path}")
    os.makedirs(TEMP_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".jpg", dir=TEMP_DIR, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        jpeg_qc = quantize_to_jpeg(gray, qc, tmp_path)
        return jpeg_qc.quant_tables[0]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_extract(image_path, qc, key, height, forced_length=None, output_path=None):
    jpeg_inter = jio.read(image_path)
    I = jpeg_inter.coef_arrays[0].astype(np.int32)
    mo_full = expand_quant_table(jpeg_inter.quant_tables[0], I.shape)
    mc_full = expand_quant_table(load_qc_table_from_image(image_path, qc), I.shape)
    received = paper_channel_model(I, mo_full, mc_full)
    message = extract_message_from_stego(received, height, key, forced_length=forced_length)
    if output_path:
        with open(output_path, "wb") as f:
            f.write(message)
    return message


def print_embed_report(result, output_path, original_message):
    changed_caca = int(np.count_nonzero(result.adjustment))
    total = int(result.adjustment.size)
    widths = result.feasible_upper - result.feasible_lower + 1
    print("CACA embed finished")
    print(f"output: {output_path}")
    print(f"payload bytes: {len(original_message)}")
    print(f"payload bits with header: {result.stc_stats.message_bits}")
    print(f"usable non-zero AC coefficients: {result.stc_stats.usable_coefficients}")
    print(f"STC changed coefficients: {result.stc_stats.changed_coefficients}")
    print(f"STC total cost: {result.stc_stats.total_cost:.6f}")
    print(f"CACA changed coefficients: {changed_caca}/{total} ({changed_caca / total:.4%})")
    print(f"feasible set width: min={int(np.min(widths))}, max={int(np.max(widths))}")
    print(f"truncated search fallbacks: {result.truncated_search_fallbacks}")
    print(f"paper channel exact: {result.paper_channel_ok}")
    print(f"paper mismatch rate: {result.paper_mismatch_rate:.8f}")
    print(f"extracted message matches: {result.recovered_message == original_message}")


def build_parser():
    parser = argparse.ArgumentParser(description="Independent real CACA implementation.")
    sub = parser.add_subparsers(dest="command", required=True)

    embed = sub.add_parser("embed", help="embed a message and write an intermediate JPEG")
    embed.add_argument("image", help="input cover image path")
    embed.add_argument("--Qo", type=int, default=DEFAULT_QO, help="original JPEG quality")
    embed.add_argument("--Qc", type=int, default=DEFAULT_QC, help="channel JPEG quality")
    embed.add_argument("--message", help="UTF-8 text message")
    embed.add_argument("--message-file", help="binary message file")
    embed.add_argument("--output", default="caca_intermediate.jpg", help="output intermediate JPEG")
    embed.add_argument("--payload", type=float, default=DEFAULT_PAYLOAD_BPNZAC, help="payload limit in bpnzac")
    embed.add_argument("--key", type=int, default=1234, help="STC parity-check key")
    embed.add_argument("--constraint-height", type=int, default=DEFAULT_CONSTRAINT_HEIGHT, help="STC height")
    embed.add_argument("--search-min", type=int, default=DEFAULT_SEARCH_MIN, help="minimum CACA adjustment to search")
    embed.add_argument("--search-max", type=int, default=DEFAULT_SEARCH_MAX, help="maximum CACA adjustment to search")

    extract = sub.add_parser("extract", help="extract a message from an intermediate JPEG")
    extract.add_argument("image", help="intermediate JPEG path")
    extract.add_argument("--Qo", type=int, default=DEFAULT_QO, help="accepted for CLI symmetry")
    extract.add_argument("--Qc", type=int, default=DEFAULT_QC, help="channel JPEG quality")
    extract.add_argument("--length", type=int, help="payload length in bytes; normally read from header")
    extract.add_argument("--output", help="write recovered bytes to this file")
    extract.add_argument("--key", type=int, default=1234, help="STC parity-check key")
    extract.add_argument("--constraint-height", type=int, default=DEFAULT_CONSTRAINT_HEIGHT, help="STC height")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "embed":
        message = read_message(args)
        if args.search_min > args.search_max:
            raise ValueError("--search-min must be <= --search-max")
        result = run_embed(
            args.image,
            args.Qo,
            args.Qc,
            message,
            args.output,
            args.key,
            args.constraint_height,
            args.payload,
            args.search_min,
            args.search_max,
        )
        print_embed_report(result, args.output, message)
    elif args.command == "extract":
        message = run_extract(args.image, args.Qc, args.key, args.constraint_height, args.length, args.output)
        if args.output:
            print(f"recovered bytes written: {args.output}")
        else:
            try:
                print(message.decode("utf-8"))
            except UnicodeDecodeError:
                print(message.hex())


if __name__ == "__main__":
    main()
