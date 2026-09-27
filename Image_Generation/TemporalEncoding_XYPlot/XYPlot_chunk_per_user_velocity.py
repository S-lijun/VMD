# -*- coding: utf-8 -*-
"""
Chunked XYPlot with per-user screen coords + speed-magnitude coloring.

Windowing: fixed-size event chunks (same as XYPlot_chunk_per_user.py, default 125).
Drawing: same per-user screen projection as XYPlot_chunk_per_user
  (shift by training min_x/min_y, canvas = max-min), not centered.

Stroke encoding:
  White background.
  R = 0 on stroke, G = B = v_norm (global CDF of |v|).

Usage:
  python XYPlot_chunk_per_user_velocity.py --dataset chaoshen \\
    --data_root Data/ChaoShen/training_files \\
    --velocity_dist ChaoShen_velocity_distribution_raw.npz \\
    --out_dir Images/ChaoShen/XYPlot_chunk_per_user_velocity
"""

import os
import re
import sys
import argparse

import cv2
import numpy as np
import pandas as pd
from scipy.stats import rankdata

_CHONG = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CHONG not in sys.path:
    sys.path.insert(0, _CHONG)

from XYPlot import (  # noqa: E402
    ROOT,
    TARGET_SIZE,
    INNER_PADDING,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
    clean_twos,
)

from XYPlot_per_user import (  # noqa: E402
    DEFAULT_TRAINING_ROOT,
    default_bounds_json,
    get_or_scan_user_max_xy,
    _norm_bounds,
)
from XYPlot_chunk_per_user import (  # noqa: E402
    N_FOLDS,
    scan_user_min_xy,
    _user_min,
    _shift_seq,
    is_skipped_user_dir,
    load_events,
    session_sequence_groups,
)

TENSOR_SUBDIR = "Chong_chunk_per_user_velocity"
GLOBAL_V_CDF = None


def natural_key(string):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", string)]


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


def load_raw_velocity_distribution(path):
    data = np.load(path)
    velocities = data["values"]
    print("\n[Velocity Distribution]")
    print("Samples:", len(velocities))
    print("Min:", velocities.min())
    print("Max:", velocities.max())
    return velocities


def build_runtime_cdf(raw_v, clip_pct):
    print("\nBuilding velocity runtime CDF")
    v_upper = np.percentile(raw_v, clip_pct)
    v_clipped = raw_v[raw_v <= v_upper]
    ranks = rankdata(v_clipped, method="average")
    cdf = (ranks - 1) / (len(v_clipped) - 1 + 1e-8)
    order = np.argsort(v_clipped)
    v_sorted = v_clipped[order]
    cdf_sorted = cdf[order]
    print("Runtime samples:", len(v_sorted))
    print("Runtime max:", v_sorted.max())
    return v_sorted, cdf_sorted


def compute_velocity(xs, ys, ts):
    dt = np.maximum(np.diff(ts), 1e-5)
    v = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / dt
    v = np.concatenate([[v[0]], v])
    return v


def _clean_df(dataset, df):
    if dataset == "balabit":
        return clean_balabit(df)
    if dataset == "chaoshen":
        return clean_chaoshen(df)
    if dataset == "dfl":
        return clean_dfl(df)
    if dataset == "twos":
        return clean_twos(df)
    raise ValueError(dataset)


def list_users(data_root):
    return sorted(
        [
            u for u in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, u)) and not is_skipped_user_dir(u)
        ],
        key=natural_key,
    )


def list_session_files(user_dir):
    return sorted(
        [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))],
        key=natural_key,
    )


# ============================================================
# Chunking
# ============================================================
# Chunking (same as XYPlot_chunk_per_user)
# ============================================================

def split_by_chunk_size(events, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]


def _session_sequences(dataset, path, chunk_size):
    return [seq for _, seqs in session_sequence_groups(load_events(dataset, path), chunk_size, False) for seq in seqs]


# ============================================================
# Draw: per-user screen coords + |v| stroke (R=0, G=B=v)
# ============================================================

def render_sequence_velocity(seq, norm_width, norm_height):
    """
    Per-user screen coords; white bg; polyline colored by CDF-normalized |v|.
    Stroke RGB = (0, v, v). Returns uint8 RGB (TARGET_SIZE, TARGET_SIZE, 3), or None.
    """
    if len(seq) < 2:
        return None

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)
    ts = np.array([float(e["time"]) for e in seq], dtype=np.float64)

    W = max(float(norm_width), 1.0)
    H = max(float(norm_height), 1.0)
    a = H / W
    canvas_w = int(W) + 1
    span = float(canvas_w - 1)
    canvas_h = int(np.ceil(a * span)) + 1

    x_pix = np.clip(np.rint(xs / W * span), 0, canvas_w - 1).astype(np.int32)
    y_pix = np.clip(np.rint(ys / W * span), 0, canvas_h - 1).astype(np.int32)

    v = compute_velocity(xs, ys, ts)
    v_norm = np.interp(v, GLOBAL_V_CDF[0], GLOBAL_V_CDF[1], left=0, right=1)

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    T = len(seq)

    for i in range(1, T):
        c = int(np.clip(round(float(v_norm[i]) * 255.0), 0, 255))
        # RGB (0, v, v) → BGR (v, v, 0)
        color_bgr = (c, c, 0)
        p0 = (int(x_pix[i - 1]), int(y_pix[i - 1]))
        p1 = (int(x_pix[i]), int(y_pix[i]))
        cv2.line(canvas, p0, p1, color_bgr, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 1, color_bgr, -1, lineType=cv2.LINE_AA)

    c0 = int(np.clip(round(float(v_norm[0]) * 255.0), 0, 255))
    cv2.circle(
        canvas,
        (int(x_pix[0]), int(y_pix[0])),
        1,
        (c0, c0, 0),
        -1,
        lineType=cv2.LINE_AA,
    )

    h, w = canvas.shape[:2]
    effective_size = max(1, TARGET_SIZE - 2 * INNER_PADDING)
    scale = effective_size / max(w, h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (TARGET_SIZE - new_h) // 2
    pad_bottom = TARGET_SIZE - new_h - pad_top
    pad_left = (TARGET_SIZE - new_w) // 2
    pad_right = TARGET_SIZE - new_w - pad_left

    final_bgr = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    return cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_tensor_chw(img_rgb):
    return np.transpose(img_rgb, (2, 0, 1))


def draw_sequence_velocity(seq, save_path, norm_width, norm_height):
    img_rgb = render_sequence_velocity(seq, norm_width, norm_height)
    if img_rgb is None:
        return
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)


def count_samples(dataset, data_root, chunk_size, five_fold=False):
    total = 0
    for user in list_users(data_root):
        user_dir = os.path.join(data_root, user)
        for file in list_session_files(user_dir):
            events = load_events(dataset, os.path.join(user_dir, file))
            for _, sequences in session_sequence_groups(events, chunk_size, five_fold):
                total += len(sequences)
    return total


# ============================================================
# Dataset Processing
# ============================================================

def process_dataset_tensors(
    dataset, data_root, out_dir, user_max_xy, user_min_xy, chunk_size, five_fold=False,
):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Chunk size:", chunk_size)
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Rendering: per-user screen (min-shift, max-min) + stroke R=0 G=B=|v| |", TARGET_SIZE, "x", TARGET_SIZE)
    if five_fold:
        print("\n[Phase] Generating chunk + per-user XYPlot velocity tensors, {} contiguous folds per session...".format(N_FOLDS))
    else:
        print("\n[Phase] Generating chunk + per-user XYPlot velocity tensors...")

    total_samples = count_samples(dataset, data_root, chunk_size, five_fold=five_fold)
    tensor_root = os.path.join(out_dir, TENSOR_SUBDIR)
    os.makedirs(tensor_root, exist_ok=True)

    H = W = int(TARGET_SIZE)
    print(f"\n[{TENSOR_SUBDIR}] Total samples: {total_samples} | Tensor size: {H}x{W}")

    images = np.memmap(
        os.path.join(tensor_root, "images.npy"),
        dtype=np.uint8,
        mode="w+",
        shape=(total_samples, 3, H, W),
    )
    labels = np.memmap(
        os.path.join(tensor_root, "labels.npy"),
        dtype=np.uint8,
        mode="w+",
        shape=(total_samples, num_users),
    )
    folds = (
        np.memmap(
            os.path.join(tensor_root, "folds.npy"),
            dtype=np.uint8,
            mode="w+",
            shape=(total_samples,),
        )
        if five_fold else None
    )

    sessions = []
    idx = 0

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)
        min_x, min_y = _user_min(user, user_min_xy)
        canvas_w = max(float(norm_x) - min_x, 1.0)
        canvas_h = max(float(norm_y) - min_y, 1.0)

        print("\n------------------------------")
        print("User:", user, "| min=({}, {}) max=({}, {})".format(
            min_x, min_y, norm_x, norm_y))

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            events = load_events(dataset, path)
            groups = session_sequence_groups(events, chunk_size, five_fold)
            n_chunks = sum(len(sequences) for _, sequences in groups)
            print(f"   Session: {session} -> {n_chunks} chunks")

            for fold, sequences in groups:
                for seq in sequences:
                    img_rgb = render_sequence_velocity(
                        _shift_seq(seq, min_x, min_y), canvas_w, canvas_h
                    )
                    if img_rgb is None:
                        continue

                    if img_rgb.shape[:2] != (H, W):
                        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

                    images[idx] = rgb_to_tensor_chw(img_rgb)
                    y = np.zeros(num_users, dtype=np.uint8)
                    y[user_to_idx[user]] = 1
                    labels[idx] = y
                    sessions.append(session)
                    if folds is not None:
                        folds[idx] = fold
                    idx += 1

    images.flush()
    labels.flush()
    if folds is not None:
        folds.flush()
    np.save(
        os.path.join(tensor_root, "sessions.npy"),
        np.array(sessions, dtype=object),
    )
    print(f"\nTensor dataset saved to: {tensor_root}")


def process_dataset(
    dataset, data_root, out_dir, user_max_xy, user_min_xy, chunk_size,
    tensors=False, five_fold=False,
):
    if tensors:
        process_dataset_tensors(
            dataset, data_root, out_dir, user_max_xy, user_min_xy, chunk_size,
            five_fold=five_fold,
        )
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Chunk size:", chunk_size)
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Rendering: per-user screen (min-shift, max-min) + stroke R=0 G=B=|v| |", TARGET_SIZE, "x", TARGET_SIZE)

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)
        min_x, min_y = _user_min(user, user_min_xy)
        canvas_w = max(float(norm_x) - min_x, 1.0)
        canvas_h = max(float(norm_y) - min_y, 1.0)

        print("\n------------------------------")
        print("User:", user, "| min=({}, {}) max=({}, {})".format(
            min_x, min_y, norm_x, norm_y))

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            print("   Session:", session)

            events = load_events(dataset, path)

            print("      Events:", len(events))
            groups = session_sequence_groups(events, chunk_size, five_fold)
            print("      Chunks:", sum(len(sequences) for _, sequences in groups))

            for fold, sequences in groups:
                for i, seq in enumerate(sequences):
                    parts = [out_dir]
                    if fold is not None:
                        parts.append("fold%d" % fold)
                    parts.extend([TENSOR_SUBDIR, user, f"{session}-{i}.png"])
                    draw_sequence_velocity(
                        _shift_seq(seq, min_x, min_y),
                        os.path.join(*parts),
                        canvas_w,
                        canvas_h,
                    )


# ============================================================
# CLI
# ============================================================

def main():
    global GLOBAL_V_CDF

    parser = argparse.ArgumentParser(
        description=(
            "XYPlot: fixed chunk windows (XYPlot_chunk_per_user) + "
            "per-user screen draw + stroke R=0 G=B=|v|."
        ),
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl", "twos"])
    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset training_files. "
             "Only used when scanning/resaving bounds JSON.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Sessions to render (train or test), relative to ROOT.",
    )
    parser.add_argument(
        "--velocity_dist",
        required=True,
        help="npz with 'values' array (e.g. ChaoShen_velocity_distribution_raw.npz).",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        default=125,
        help="Number of events per chunk.",
    )
    parser.add_argument(
        "--bounds_json",
        default=None,
        help="Per-user max_x/max_y cache; default ChongSOTA/bounds/<dataset>_xy_bounds.json "
             "(shared with XYPlot_per_user.py).",
    )
    parser.add_argument(
        "--rescan_bounds",
        action="store_true",
        default=False,
        help="Force rescan training_root and overwrite bounds JSON.",
    )
    parser.add_argument(
        "--v_percentile",
        type=float,
        default=100,
        help="Upper clip percentile for |v| CDF.",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="Output images.npy / labels.npy / sessions.npy instead of PNG.",
    )
    parser.add_argument(
        "--five-fold",
        action="store_true",
        default=False,
        help="每个 session 按事件顺序切成 5 段连续事件再切 chunk。tensors 时多写 folds.npy，取值 0–4。",
    )
    args = parser.parse_args()

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = resolve_path(training_rel)
    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    dist_path = resolve_path(args.velocity_dist)
    bounds_json = (
        resolve_path(args.bounds_json) if args.bounds_json
        else default_bounds_json(args.dataset)
    )

    print("[training_root]", training_root)
    print("[data_root]", data_root)
    print("[out_dir]", out_dir)
    print("[velocity_dist]", dist_path)
    print("Bounds JSON:", bounds_json)
    print("Chunk size:", args.sizes)
    skipped = sorted(
        [
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name)) and is_skipped_user_dir(name)
        ],
        key=natural_key,
    )
    if skipped:
        print("Excluded dirs:", ", ".join(skipped))

    raw_v = load_raw_velocity_distribution(dist_path)
    GLOBAL_V_CDF = build_runtime_cdf(raw_v, args.v_percentile)

    user_max_xy = get_or_scan_user_max_xy(
        dataset=args.dataset,
        training_root=training_root,
        bounds_json=bounds_json,
        rescan=args.rescan_bounds,
    )
    user_min_xy = scan_user_min_xy(args.dataset, training_root)

    print("\nUSER_MAX_XY:")
    for u in sorted(user_max_xy.keys(), key=natural_key):
        print("  ", u, "->", user_max_xy[u])

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        user_max_xy,
        user_min_xy,
        args.sizes,
        tensors=args.tensors,
        five_fold=args.five_fold,
    )
    print("\nChunk + per-user XYPlot velocity generation finished.")


if __name__ == "__main__":
    main()
