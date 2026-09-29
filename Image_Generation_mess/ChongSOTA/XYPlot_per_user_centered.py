# -*- coding: utf-8 -*-
"""
Per-user segmentation + centered XYPlot.

Segmentation / merge / bounds: identical to XYPlot_per_user.py
  (split_by_time + merge_sequences with per-user max_x from training bounds).

Drawing: identical to XYPlot_chunk.py
  (per-sequence bbox fit + center; no screen-width normalization).
"""

import os
import re
import argparse

import cv2
import numpy as np
import pandas as pd

from XYPlot import (
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
    DEFAULT_TRAINING_ROOT,
    default_bounds_json,
    get_or_scan_user_max_xy,
    _norm_bounds,
)

TENSOR_SUBDIR = "Chong_per_user_centered"


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
# Centered draw (identical to XYPlot_chunk.py)
# ============================================================

def render_sequence_centered(seq):
    """
    Fit trajectory bbox into TARGET_SIZE×TARGET_SIZE with uniform scale, centered.
    """
    if len(seq) < 2:
        return None

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

    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    prev = None
    for x, y in zip(xs, ys):
        x_s = int(np.clip((x - min_x) * scale + offset_x, 0, img_size - 1))
        y_s = int(np.clip((y - min_y) * scale + offset_y, 0, img_size - 1))

        if prev is not None:
            cv2.line(
                canvas,
                prev,
                (x_s, y_s),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

        prev = (x_s, y_s)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_sequence_centered(seq, save_path):
    final = render_sequence_centered(seq)
    if final is None:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final)


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


def bgr_to_tensor_chw(img):
    """Match Images_convert.py: BGR HWC -> (3, H, W) uint8."""
    return img.transpose(2, 0, 1)


def process_dataset_tensors(dataset, data_root, out_dir, user_max_xy):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Rendering: per-sequence bbox fit, centered in", TARGET_SIZE, "x", TARGET_SIZE)
    print("\n[Phase] Generating per-user centered XYPlot tensors...")

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
                img = render_sequence_centered(seq)
                if img is None:
                    continue

                images[idx] = bgr_to_tensor_chw(img)

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
    print("Rendering: per-sequence bbox fit, centered in", TARGET_SIZE, "x", TARGET_SIZE)

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
                save_path = os.path.join(
                    out_dir,
                    TENSOR_SUBDIR,
                    user,
                    f"{session}-{i}.png",
                )
                draw_sequence_centered(seq, save_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "XYPlot: per-user split/merge (same as XYPlot_per_user) + "
            "centered bbox draw (same as XYPlot_chunk)."
        ),
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset (Balabit/ChaoShen/DFL training_files)."
             " Only used when scanning/resaving bounds JSON.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Sessions to render (train or test), relative to ROOT.",
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
    bounds_json = (
        resolve_path(args.bounds_json) if args.bounds_json
        else default_bounds_json(args.dataset)
    )

    print("[training_root]", training_root)
    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("Bounds JSON:", bounds_json)

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
    print("\nPer-user centered XYPlot generation finished.")


if __name__ == "__main__":
    main()
