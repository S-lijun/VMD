# -*- coding: utf-8 -*-
"""
Per-user XYPlot with speed-magnitude coloring on the trajectory (centered).

Same windowing / per-user merge as XYPlot_per_user_uc_velocity.py:
  max x / max y from all sessions of that user under training_files;
  split_by_time + merge_sequences with per-user norm_x.
  Bounds JSON cached under ChongSOTA/bounds/ (shared with XYPlot_per_user.py).

Drawing: centered bbox fit (same transform as XYPlot_chunk.py /
XYPlot_chunk_velocity.py) — no screen-width / per-user projection.

Velocity encoding (same as XYPlot_chunk_velocity.py):
  White background.
  Trajectory stroke colored by |v| (global CDF → [0, 1]):
    R = 0 on stroke (geometry), G = B = v_norm.
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

TENSOR_SUBDIR = "Chong_per_user_velocity_centered"
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
# Centered coords (same transform as XYPlot_chunk / XYPlot_chunk_velocity)
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
# Draw: centered + white bg, stroke/points colored by |v| (R=0, G=B=v)
# ============================================================

def render_sequence_velocity(seq):
    """
    Centered bbox fit; white bg; polyline colored by CDF-normalized |v|.
    Stroke RGB = (0, v, v). Returns uint8 RGB (TARGET_SIZE, TARGET_SIZE, 3), or None.
    """
    pts, img_size = _centered_pixel_coords(seq)
    if pts is None:
        return None

    T = len(seq)
    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)
    ts = np.array([float(e["time"]) for e in seq], dtype=np.float64)

    v = compute_velocity(xs, ys, ts)
    v_norm = np.interp(
        v,
        GLOBAL_V_CDF[0],
        GLOBAL_V_CDF[1],
        left=0,
        right=1,
    )

    # OpenCV canvas is BGR; convert to RGB on return.
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    for i in range(1, T):
        vn = float(v_norm[i])
        # RGB (0, v, v) → BGR (v, v, 0)
        c = int(np.clip(round(vn * 255.0), 0, 255))
        color_bgr = (c, c, 0)
        p0 = (int(pts[i - 1, 0]), int(pts[i - 1, 1]))
        p1 = (int(pts[i, 0]), int(pts[i, 1]))
        cv2.line(canvas, p0, p1, color_bgr, 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 1, color_bgr, -1, lineType=cv2.LINE_AA)

    c0 = int(np.clip(round(float(v_norm[0]) * 255.0), 0, 255))
    cv2.circle(
        canvas,
        (int(pts[0, 0]), int(pts[0, 1])),
        1,
        (c0, c0, 0),
        -1,
        lineType=cv2.LINE_AA,
    )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def rgb_to_tensor_chw(img_rgb):
    """RGB H×W×3 uint8 -> (3, H, W) uint8."""
    return np.transpose(img_rgb, (2, 0, 1))


def draw_sequence_velocity(seq, save_path):
    img_rgb = render_sequence_velocity(seq)
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
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Rendering: centered bbox fit, stroke colored by |v| (R=0, G=B=v) |", TARGET_SIZE, "x", TARGET_SIZE)
    print("\n[Phase] Generating per-user centered XYPlot velocity-colored tensors...")

    total_samples, _ = count_samples(dataset, data_root, user_max_xy)
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
            print(f"   Session: {session} -> {len(sequences)} sequences")

            for seq in sequences:
                img_rgb = render_sequence_velocity(seq)
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

    np.save(
        os.path.join(tensor_root, "sessions.npy"),
        np.array(sessions, dtype=object),
    )

    print(f"\nTensor dataset saved to: {tensor_root}")


def process_dataset(dataset, data_root, out_dir, user_max_xy, tensors=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir, user_max_xy)
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users in data_root:", len(users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Rendering: centered bbox fit, stroke colored by |v| (R=0, G=B=v) |", TARGET_SIZE, "x", TARGET_SIZE)

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
                    f"{session}-{i}.png",
                )
                draw_sequence_velocity(seq, save_path)


def main():
    global GLOBAL_V_CDF

    parser = argparse.ArgumentParser(
        description=(
            "Per-user XYPlot (centered): white bg, trajectory colored by |v| (R=0, G=B=v)."
        ),
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset (Balabit/ChaoShen/DFL training_files).",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Sessions to render (train or test), relative to ROOT.",
    )
    parser.add_argument(
        "--velocity_dist",
        required=True,
        help="npz with values array (e.g. velocity_distribution_raw.npz)",
    )
    parser.add_argument("--out_dir", required=True)
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
        help="Upper clip percentile for speed CDF (same as XYPlot_chunk_velocity).",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="直接输出 Images_convert 格式的 tensors（images.npy / labels.npy / sessions.npy），不写 PNG。",
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

    raw_v = load_raw_velocity_distribution(dist_path)
    GLOBAL_V_CDF = build_runtime_cdf(raw_v, args.v_percentile)

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
    print("\nPer-user centered XYPlot velocity-colored generation finished.")


if __name__ == "__main__":
    main()
