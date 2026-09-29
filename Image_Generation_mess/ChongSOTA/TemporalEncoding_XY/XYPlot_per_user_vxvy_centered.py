# -*- coding: utf-8 -*-
"""
Per-user XYPlot with directional velocity coloring (vx/vy), centered.

Same windowing / per-user merge as XYPlot_per_user_uc_vxvy.py:
  max x / max y from training bounds JSON;
  split_by_time + merge_sequences with per-user norm_x.
  Bounds JSON cached under ChongSOTA/bounds/ (shared with XYPlot_per_user.py).

Drawing: centered bbox fit (same transform as XYPlot_chunk.py /
XYPlot_chunk_vxvy.py) — no screen-width / per-user projection.

Stroke encoding (same as XYPlot_chunk_vxvy.py):
  White background.
  R = 0 on stroke, G = vx_norm, B = vy_norm (signed CDFs).

Usage:
  python XYPlot_per_user_vxvy_centered.py --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files \\
    --velocity_dist Balabit_vxvy_distribution_raw.npz \\
    --out_dir Images/Balabit/XYPlot_per_user_vxvy_centered
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
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
)

from XYPlot_per_user import (  # noqa: E402
    default_bounds_json,
    get_or_scan_user_max_xy,
    _norm_bounds,
)

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
}

TENSOR_SUBDIR = "Chong_per_user_vxvy_centered"
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
    print("\nvx samples:", len(vx), "min:", vx.min(), "max:", vx.max())
    print("vy samples:", len(vy), "min:", vy.min(), "max:", vy.max())
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
    print("runtime min:", v_sorted.min(), "max:", v_sorted.max())
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
    raise ValueError(dataset)


def list_users(data_root):
    return sorted(
        [u for u in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, u))],
        key=natural_key,
    )


def list_session_files(user_dir):
    return sorted(
        [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))],
        key=natural_key,
    )



def _session_sequences(dataset, path, norm_x, norm_y):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)

    events = df.to_dict("records")
    if len(events) < 2:
        return []

    sequences = split_by_time(events)
    sequences = merge_sequences(sequences, norm_x)
    return [seq for seq in sequences if len(seq) >= 2]


# ============================================================
# Centered coords (same transform as XYPlot_chunk / XYPlot_chunk_vxvy)
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
# Draw: centered + white bg, stroke R=0 G=vx B=vy
# ============================================================

def render_sequence_vxvy(seq):
    """
    Centered bbox fit; white bg; polyline colored by signed-CDF vx/vy.
    Stroke RGB = (0, vx, vy). Returns uint8 RGB (TARGET_SIZE, TARGET_SIZE, 3), or None.
    """
    pts, img_size = _centered_pixel_coords(seq)
    if pts is None:
        return None

    T = len(seq)
    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)
    ts = np.array([float(e["time"]) for e in seq], dtype=np.float64)

    vx, vy = compute_vx_vy(xs, ys, ts)
    vx_norm = np.interp(vx, GLOBAL_VX_CDF[0], GLOBAL_VX_CDF[1], left=0, right=1)
    vy_norm = np.interp(vy, GLOBAL_VY_CDF[0], GLOBAL_VY_CDF[1], left=0, right=1)

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
    return np.transpose(img_rgb, (2, 0, 1))


def draw_sequence_vxvy(seq, save_path):
    img_rgb = render_sequence_vxvy(seq)
    if img_rgb is None:
        return
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)


def count_samples(dataset, data_root, user_max_xy):
    total = 0
    users = list_users(data_root)
    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)
        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            total += len(_session_sequences(dataset, path, norm_x, norm_y))
    return total, users


def process_dataset_tensors(dataset, data_root, out_dir, user_max_xy):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Per-user max bounds loaded for", len(user_max_xy), "users.")
    print("Rendering: centered bbox fit, stroke R=0 G=vx B=vy |", TARGET_SIZE, "x", TARGET_SIZE)
    print("\n[Phase] Generating per-user centered XYPlot vx/vy-colored tensors...")

    total_samples, _ = count_samples(dataset, data_root, user_max_xy)
    tensor_root = os.path.join(out_dir, TENSOR_SUBDIR)
    os.makedirs(tensor_root, exist_ok=True)

    H = W = int(TARGET_SIZE)
    print("\n[{}] Total samples: {} | Tensor size: {}x{}".format(
        TENSOR_SUBDIR, total_samples, H, W))

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

    sessions = []
    idx = 0

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)

        print("\n------------------------------")
        print("User:", user, "| merge min_length (norm_x from training):", norm_x)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            sequences = _session_sequences(dataset, path, norm_x, norm_y)
            print("   Session: {} -> {} sequences".format(session, len(sequences)))

            for seq in sequences:
                img_rgb = render_sequence_vxvy(seq)
                if img_rgb is None:
                    continue
                if img_rgb.shape[:2] != (H, W):
                    img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)

                images[idx] = rgb_to_tensor_chw(img_rgb)
                y = np.zeros(num_users, dtype=np.uint8)
                y[user_to_idx[user]] = 1
                labels[idx] = y
                sessions.append(session)
                idx += 1

    images.flush()
    labels.flush()
    np.save(os.path.join(tensor_root, "sessions.npy"), np.array(sessions, dtype=object))
    print("\nTensor dataset saved to:", tensor_root)


def process_dataset(dataset, data_root, out_dir, user_max_xy, tensors=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir, user_max_xy)
        return

    users = list_users(data_root)
    print("\nDataset:", dataset)
    print("Users in data_root:", len(users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users.")
    print("Rendering: centered bbox fit, stroke R=0 G=vx B=vy |", TARGET_SIZE, "x", TARGET_SIZE)

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)

        print("\n------------------------------")
        print("User:", user, "| merge min_length (norm_x from training):", norm_x)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            print("   Session:", session)

            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            events = df.to_dict("records")
            print("      Events:", len(events))
            if len(events) < 2:
                continue

            sequences = split_by_time(events)
            print("      After split:", len(sequences))
            sequences = merge_sequences(sequences, norm_x)
            print("      After merge:", len(sequences))

            for i, seq in enumerate(sequences):
                if len(seq) < 2:
                    continue
                save_path = os.path.join(
                    out_dir,
                    TENSOR_SUBDIR,
                    user,
                    "{}-{}.png".format(session, i),
                )
                draw_sequence_vxvy(seq, save_path)


def main():
    global GLOBAL_VX_CDF, GLOBAL_VY_CDF

    parser = argparse.ArgumentParser(
        description="Per-user XYPlot (centered): white bg, stroke R=0 G=vx B=vy.",
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument(
        "--training_root",
        default=None,
        help="Default follows --dataset training_files; used when scanning bounds.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Sessions to render (train or test).",
    )
    parser.add_argument(
        "--velocity_dist",
        required=True,
        help="npz with vx/vy arrays (e.g. Balabit_vxvy_distribution_raw.npz).",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--bounds_json",
        default=None,
        help="Per-user max_x/max_y cache; default ChongSOTA/bounds/<dataset>_xy_bounds.json.",
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
        help="Symmetric clip percentile for signed vx/vy CDF.",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="Output images.npy / labels.npy / sessions.npy.",
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
    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("[velocity_dist]", dist_path)
    print("Bounds JSON:", bounds_json)

    vx_raw, vy_raw = load_raw_directional_velocity_distribution(dist_path)
    print("\n[vx CDF]")
    GLOBAL_VX_CDF = build_runtime_cdf_signed(vx_raw, args.v_percentile)
    print("\n[vy CDF]")
    GLOBAL_VY_CDF = build_runtime_cdf_signed(vy_raw, args.v_percentile)

    user_max_xy = get_or_scan_user_max_xy(
        dataset=args.dataset,
        training_root=training_root,
        bounds_json=bounds_json,
        rescan=args.rescan_bounds,
    )

    print("\nUSER_MAX_XY:")
    for u in sorted(user_max_xy.keys(), key=natural_key):
        print("  ", u, "->", user_max_xy[u])

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        user_max_xy,
        tensors=args.tensors,
    )
    print("\nPer-user centered XYPlot vx/vy-colored generation finished.")


if __name__ == "__main__":
    main()
