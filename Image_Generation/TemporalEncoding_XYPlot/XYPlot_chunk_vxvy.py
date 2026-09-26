# -*- coding: utf-8 -*-
"""
Chunked centered XYPlot with vx/vy coloring on the trajectory.

Per chunk (same windowing / centered draw as XYPlot_chunk.py):
  White background.
  Trajectory stroke colored by signed-CDF-normalized vx, vy:
    R = 0 on stroke (geometry), G = vx_norm, B = vy_norm
    (same channel layout as SRP_chunk_vxvy, but painted only on the polyline / points).
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
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
    clean_twos,
)

TENSOR_SUBDIR = "Chong"
N_FOLDS = 5
GLOBAL_VX_CDF = None
GLOBAL_VY_CDF = None


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


def load_raw_directional_velocity_distribution(path):
    data = np.load(path)
    vx = data["vx"]
    vy = data["vy"]

    print("\n[Directional Velocity Distribution]")
    print("\nvx")
    print("samples:", len(vx))
    print("min:", vx.min())
    print("max:", vx.max())
    print("\nvy")
    print("samples:", len(vy))
    print("min:", vy.min())
    print("max:", vy.max())

    return vx, vy


def build_runtime_cdf_signed(raw_values, clip_pct):
    print("\nBuilding signed runtime CDF")

    lower = np.percentile(raw_values, 100 - clip_pct)
    upper = np.percentile(raw_values, clip_pct)

    clipped = raw_values[(raw_values >= lower) & (raw_values <= upper)]

    ranks = rankdata(clipped, method="average")
    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)
    v_sorted = clipped[order]
    cdf_sorted = cdf[order]

    print("runtime samples:", len(v_sorted))
    print("runtime min:", v_sorted.min())
    print("runtime max:", v_sorted.max())

    return v_sorted, cdf_sorted


def compute_vx_vy(xs, ys, ts):
    dt = np.maximum(np.diff(ts), 1e-5)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    return vx, vy


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


def is_skipped_user_dir(name):
    """Data/TWOS 下面还有划分目录，不能当成用户。"""
    lower = name.lower()
    return lower == "training_files" or lower.startswith("testing_files")


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
# Chunking (same as XYPlot_chunk)
# ============================================================

def split_by_chunk_size(events, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]


def contiguous_fold_bounds(n_events, n_folds=N_FOLDS):
    """把一个 session 的事件按顺序切成 n_folds 段，段与段首尾相接、不重叠。"""
    return [
        (i * n_events // n_folds, (i + 1) * n_events // n_folds)
        for i in range(n_folds)
    ]


# ============================================================
# Centered coords (same transform as XYPlot_chunk)
# ============================================================

def _centered_pixel_coords(seq):
    """
    Per-sequence bbox fit + center → pixel (x, y) for each event.
    Returns (pts Nx2 int32, img_size) or (None, None).
    """
    if len(seq) < 2:
        return None, None

    img_size = int(TARGET_SIZE)
    effective_size = max(1, img_size - 2 * INNER_PADDING)

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    pad_x = range_x * 0.05
    pad_y = range_y * 0.05

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    range_x = max_x - min_x
    range_y = max_y - min_y

    scale = min(effective_size / range_x, effective_size / range_y)
    offset_x = (img_size - range_x * scale) / 2
    offset_y = (img_size - range_y * scale) / 2

    x_s = np.clip((xs - min_x) * scale + offset_x, 0, img_size - 1).astype(np.int32)
    y_s = np.clip((ys - min_y) * scale + offset_y, 0, img_size - 1).astype(np.int32)
    pts = np.stack([x_s, y_s], axis=1)
    return pts, img_size


# ============================================================
# Draw: white bg, color stroke/points by vx/vy (R=0, G=vx, B=vy)
# ============================================================

def render_xyplot_chunk_vxvy(seq):
    """
    White background; polyline / points colored by signed-CDF-normalized vx, vy.
    Stroke RGB = (0, vx, vy)  →  channel layout R=geometry(0), G=vx, B=vy.
    Returns uint8 RGB (H, W, 3), or None if seq too short.
    """
    pts, img_size = _centered_pixel_coords(seq)
    if pts is None:
        return None

    T = len(seq)
    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)
    ts = np.array([float(e["time"]) for e in seq], dtype=np.float64)

    vx, vy = compute_vx_vy(xs, ys, ts)

    vx_norm = np.interp(
        vx,
        GLOBAL_VX_CDF[0],
        GLOBAL_VX_CDF[1],
        left=0,
        right=1,
    )
    vy_norm = np.interp(
        vy,
        GLOBAL_VY_CDF[0],
        GLOBAL_VY_CDF[1],
        left=0,
        right=1,
    )

    # OpenCV canvas is BGR; we convert to RGB on return.
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    for i in range(1, T):
        gx = int(np.clip(round(float(vx_norm[i]) * 255.0), 0, 255))
        by = int(np.clip(round(float(vy_norm[i]) * 255.0), 0, 255))
        # RGB (0, vx, vy) → BGR (vy, vx, 0)
        color_bgr = (by, gx, 0)
        p0 = (int(pts[i - 1, 0]), int(pts[i - 1, 1]))
        p1 = (int(pts[i, 0]), int(pts[i, 1]))
        cv2.line(canvas, p0, p1, color_bgr, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 1, color_bgr, -1, lineType=cv2.LINE_AA)

    # First point
    gx0 = int(np.clip(round(float(vx_norm[0]) * 255.0), 0, 255))
    by0 = int(np.clip(round(float(vy_norm[0]) * 255.0), 0, 255))
    cv2.circle(
        canvas,
        (int(pts[0, 0]), int(pts[0, 1])),
        1,
        (by0, gx0, 0),
        -1,
        lineType=cv2.LINE_AA,
    )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def rgb_to_tensor_chw(img_rgb):
    """RGB H×W×3 uint8 -> (3, H, W) uint8."""
    return np.transpose(img_rgb, (2, 0, 1))


def draw_xyplot_chunk_vxvy(seq, save_path):
    img_rgb = render_xyplot_chunk_vxvy(seq)
    if img_rgb is None:
        return

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)


def load_events(dataset, path):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)
    return df.to_dict("records")


def sequences_from_events(events, chunk_size):
    if len(events) < 2:
        return []
    return [seq for seq in split_by_chunk_size(events, chunk_size) if len(seq) >= 2]


def iter_session_fold_sequences(events, chunk_size, n_folds=N_FOLDS):
    """每一折只在自己那段连续事件里切 chunk，chunk 不会跨到别的折。"""
    for fold, (start, end) in enumerate(contiguous_fold_bounds(len(events), n_folds)):
        yield fold, sequences_from_events(events[start:end], chunk_size)


def _session_sequences(dataset, path, chunk_size):
    return sequences_from_events(load_events(dataset, path), chunk_size)


def count_samples(dataset, data_root, chunk_size, five_fold=False):
    total = 0
    for user in list_users(data_root):
        user_dir = os.path.join(data_root, user)
        for file in list_session_files(user_dir):
            events = load_events(dataset, os.path.join(user_dir, file))
            if five_fold:
                for _, sequences in iter_session_fold_sequences(events, chunk_size):
                    total += len(sequences)
            else:
                total += len(sequences_from_events(events, chunk_size))
    return total


# ============================================================
# Dataset Processing
# ============================================================

def _write_chunk(images, labels, sessions, folds, idx, seq, H, W, user_idx, num_users, session, fold):
    img_rgb = render_xyplot_chunk_vxvy(seq)
    if img_rgb is None:
        return idx

    if img_rgb.shape[:2] != (H, W):
        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

    images[idx] = rgb_to_tensor_chw(img_rgb)
    y = np.zeros(num_users, dtype=np.uint8)
    y[user_idx] = 1
    labels[idx] = y
    sessions.append(session)
    if folds is not None:
        folds[idx] = fold
    return idx + 1


def process_dataset_tensors(dataset, data_root, out_dir, chunk_size, five_fold=False):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Chunk size:", chunk_size)
    print("Rendering: white bg, stroke colored by vx/vy (R=0, G=vx, B=vy) |", TARGET_SIZE, "x", TARGET_SIZE)
    if five_fold:
        print(f"\n[Phase] Generating XYPlot vx/vy-colored trajectory tensors, {N_FOLDS} contiguous folds per session...")
    else:
        print("\n[Phase] Generating XYPlot vx/vy-colored trajectory tensors...")

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
    folds = None
    if five_fold:
        folds = np.memmap(
            os.path.join(tensor_root, "folds.npy"),
            dtype=np.uint8,
            mode="w+",
            shape=(total_samples,),
        )

    sessions = []
    idx = 0

    for user in users:
        user_dir = os.path.join(data_root, user)

        print("\n------------------------------")
        print("User:", user)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            events = load_events(dataset, path)
            if five_fold:
                fold_sequences = list(iter_session_fold_sequences(events, chunk_size))
                n_chunks = sum(len(sequences) for _, sequences in fold_sequences)
                print(f"   Session: {session} -> {n_chunks} chunks")
                for fold, sequences in fold_sequences:
                    for seq in sequences:
                        idx = _write_chunk(
                            images, labels, sessions, folds, idx, seq,
                            H, W, user_to_idx[user], num_users, session, fold,
                        )
            else:
                sequences = sequences_from_events(events, chunk_size)
                print(f"   Session: {session} -> {len(sequences)} chunks")
                for seq in sequences:
                    idx = _write_chunk(
                        images, labels, sessions, folds, idx, seq,
                        H, W, user_to_idx[user], num_users, session, 0,
                    )

    images.flush()
    labels.flush()
    if folds is not None:
        folds.flush()
    np.save(
        os.path.join(tensor_root, "sessions.npy"),
        np.array(sessions, dtype=object),
    )
    print(f"\nTensor dataset saved to: {tensor_root}")


def process_dataset(dataset, data_root, out_dir, chunk_size, tensors=False, five_fold=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir, chunk_size, five_fold=five_fold)
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Chunk size:", chunk_size)
    print("Rendering: white bg, stroke colored by vx/vy (R=0, G=vx, B=vy) |", TARGET_SIZE, "x", TARGET_SIZE)

    for user in users:
        user_dir = os.path.join(data_root, user)

        print("\n------------------------------")
        print("User:", user)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            print("   Session:", session)

            events = load_events(dataset, path)

            print("      Events:", len(events))
            if five_fold:
                groups = list(iter_session_fold_sequences(events, chunk_size))
            else:
                groups = [(None, sequences_from_events(events, chunk_size))]
            print("      Chunks:", sum(len(sequences) for _, sequences in groups))

            for fold, sequences in groups:
                for i, seq in enumerate(sequences):
                    parts = [out_dir]
                    if fold is not None:
                        parts.append("fold%d" % fold)
                    parts.extend([TENSOR_SUBDIR, user, f"{session}-{i}.png"])
                    draw_xyplot_chunk_vxvy(seq, os.path.join(*parts))


# ============================================================
# CLI
# ============================================================

def main():
    global GLOBAL_VX_CDF
    global GLOBAL_VY_CDF

    parser = argparse.ArgumentParser(
        description="Chunk centered XYPlot: white bg, trajectory colored by vx/vy (R=0, G=vx, B=vy).",
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl", "twos"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--velocity_dist",
        required=True,
        help="npz with vx, vy arrays (e.g. vx_vy_distribution_raw.npz)",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        default=125,
        help="Number of events per chunk.",
    )
    parser.add_argument(
        "--v_percentile",
        type=float,
        default=100,
        help="Signed CDF clip percentile for vx/vy (same as SRP_chunk_vxvy).",
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

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    dist_path = resolve_path(args.velocity_dist)

    print("[data_root]", data_root)
    print("[out_dir]", out_dir)
    print("[velocity_dist]", dist_path)
    skipped = sorted(
        [
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name)) and is_skipped_user_dir(name)
        ],
        key=natural_key,
    )
    if skipped:
        print("Excluded dirs:", ", ".join(skipped))

    vx_raw, vy_raw = load_raw_directional_velocity_distribution(dist_path)
    GLOBAL_VX_CDF = build_runtime_cdf_signed(vx_raw, args.v_percentile)
    GLOBAL_VY_CDF = build_runtime_cdf_signed(vy_raw, args.v_percentile)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        tensors=args.tensors,
        five_fold=args.five_fold,
    )
    print("\nCentered chunk XYPlot vx/vy-colored trajectory finished.")


if __name__ == "__main__":
    main()
