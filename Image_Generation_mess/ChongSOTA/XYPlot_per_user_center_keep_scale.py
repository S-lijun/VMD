# -*- coding: utf-8 -*-
"""
Per-user XYPlot with trajectory centering, keeping screen scale.

Same as XYPlot_per_user.py:
  - split_by_time + merge_sequences(min_length=user max_x)
  - canvas / pixel mapping from per-user training max_x, max_y
  - final uniform pack of the whole screen canvas into TARGET_SIZE
    (preserves trajectory-vs-screen scale)

Only change:
  - translate each trajectory so its bbox center sits at screen center
  - NO per-trajectory bbox fit / rescale (unlike XYPlot_per_user_centered.py)

Usage:
  python XYPlot_per_user_center_keep_scale.py --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files \\
    --out_dir Images/Balabit/XYPlot_per_user_center_keep_scale
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
from XYPlot_per_user import (
    DEFAULT_TRAINING_ROOT,
    default_bounds_json,
    get_or_scan_user_max_xy,
    _norm_bounds,
)

TENSOR_SUBDIR = "Chong_per_user_center_keep_scale"


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


def render_sequence_center_keep_scale(seq, norm_width, norm_height):
    """
    Same screen canvas / mapping as XYPlot.render_sequence, but translate
    the trajectory so its bbox center is at the screen center. No traj resize.
    """
    if len(seq) < 2:
        return None

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)

    W = max(float(norm_width), 1.0)
    H = max(float(norm_height), 1.0)

    # Center trajectory on screen (translation only).
    traj_cx = 0.5 * (xs.min() + xs.max())
    traj_cy = 0.5 * (ys.min() + ys.max())
    xs = xs - traj_cx + 0.5 * W
    ys = ys - traj_cy + 0.5 * H

    a = H / W
    canvas_w = int(W) + 1
    span = float(canvas_w - 1)
    canvas_h = int(np.ceil(a * span)) + 1

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    x_pix = np.clip(np.rint(xs / W * span), 0, canvas_w - 1).astype(np.int32)
    y_pix = np.clip(np.rint(ys / W * span), 0, canvas_h - 1).astype(np.int32)

    prev = None
    for x_i, y_i in zip(x_pix, y_pix):
        if prev is not None:
            cv2.line(
                canvas,
                prev,
                (int(x_i), int(y_i)),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )
        prev = (int(x_i), int(y_i))

    # Uniform pack of the whole screen canvas (same as XYPlot_per_user).
    # This keeps trajectory-vs-screen scale; it is not a per-traj bbox resize.
    h, w = canvas.shape[:2]
    effective_size = max(1, TARGET_SIZE - 2 * INNER_PADDING)
    scale = effective_size / max(w, h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    resized = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    pad_top = (TARGET_SIZE - new_h) // 2
    pad_bottom = TARGET_SIZE - new_h - pad_top
    pad_left = (TARGET_SIZE - new_w) // 2
    pad_right = TARGET_SIZE - new_w - pad_left

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def draw_sequence_center_keep_scale(seq, save_path, norm_width, norm_height):
    final = render_sequence_center_keep_scale(seq, norm_width, norm_height)
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
    return img.transpose(2, 0, 1)


def process_dataset_tensors(dataset, data_root, out_dir, user_max_xy):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Rendering: center traj on screen, keep screen scale (no bbox fit).")
    print("\n[Phase] Generating tensors...")

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
        print("User:", user, "| screen W×H (from training):", norm_x, norm_y)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            sequences = _session_sequences(dataset, path, norm_x, norm_y)
            print(f"   Session: {session} -> {len(sequences)} sequences")

            for seq in sequences:
                img = render_sequence_center_keep_scale(seq, norm_x, norm_y)
                if img is None:
                    continue

                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

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
    print("Rendering: center traj on screen, keep screen scale (no bbox fit).")

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)

        print("\n------------------------------")
        print("User:", user, "| screen W×H (from training):", norm_x, norm_y)

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
                draw_sequence_center_keep_scale(seq, save_path, norm_x, norm_y)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "XYPlot_per_user + center each traj on screen; "
            "keep screen scale (no per-traj bbox resize)."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset. Only for bounds scan.",
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
        help="Per-user max_x/max_y cache; default ChongSOTA/bounds/<dataset>_xy_bounds.json.",
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
        help="输出 images.npy / labels.npy / sessions.npy，不写 PNG。",
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
    print("\nPer-user center-keep-scale XYPlot generation finished.")


if __name__ == "__main__":
    main()
