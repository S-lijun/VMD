# -*- coding: utf-8 -*-
"""
Reproduce Chong et al. (2018) fused-curve XYPlot preprocessing:

  1) Time-difference split (dt > 1s)
  2) Fuse short segments until arc length >= fuse_fraction * screen_width
  3) Plot in per-user screen coordinates (estimated from training max coords)
  4) Resize: plot width -> 448, keep aspect ratio, pad to 448x448

Paper: "Mouse Authentication without the Temporal Aspect – What does a 2D-CNN learn?"
Default fuse_fraction=1.00 matches Table II best row (fused curve min length 1.00).
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
    TIME_THRESHOLD,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
)

TENSOR_SUBDIR = "Chong_paper"

# Common desktop resolutions (width, height). Paper maps max coords to closest match.
STANDARD_RESOLUTIONS = [
    (3840, 2160),
    (2560, 1440),
    (1920, 1200),
    (1920, 1080),
    (1680, 1050),
    (1600, 900),
    (1440, 900),
    (1366, 768),
    (1280, 1024),
    (1280, 800),
    (1280, 720),
    (1024, 768),
]

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
}


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


def map_to_standard_resolution(max_x, max_y):
    """
    Map observed max coordinates to the closest standard screen resolution.
    Returns (norm_width, norm_height) as max pixel indices (width-1, height-1).
    """
    max_x = float(max_x)
    max_y = float(max_y)

    fitting = [
        (w, h)
        for w, h in STANDARD_RESOLUTIONS
        if (w - 1) >= max_x and (h - 1) >= max_y
    ]
    if fitting:
        # Smallest screen that still contains all observed coordinates.
        w, h = min(fitting, key=lambda wh: wh[0] * wh[1])
        return float(w - 1), float(h - 1)

    # Fallback: closest resolution by max-coordinate distance.
    def dist(wh):
        w, h = wh
        return (max(max_x - (w - 1), 0)) ** 2 + (max(max_y - (h - 1), 0)) ** 2

    w, h = min(STANDARD_RESOLUTIONS, key=dist)
    return float(w - 1), float(h - 1)


def build_user_resolution_from_training(dataset, training_root):
    """Per-user screen bounds from training sessions (paper Section IV-A)."""
    user_res = {}

    for user in list_users(training_root):
        user_dir = os.path.join(training_root, user)
        max_x = 0.0
        max_y = 0.0
        saw_points = False

        for name in list_session_files(user_dir):
            path = os.path.join(user_dir, name)
            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            if len(df) == 0:
                continue
            max_x = max(max_x, float(df["x"].max()))
            max_y = max(max_y, float(df["y"].max()))
            saw_points = True

        if saw_points:
            user_res[user] = map_to_standard_resolution(max_x, max_y)
        else:
            user_res[user] = (GLOBAL_MAX_X, GLOBAL_MAX_Y)

    return user_res


def _resolution_bounds(user, user_res):
    if user in user_res:
        return user_res[user]
    print(
        "\n[WARN] User", user, "not in training scan; using GLOBAL_MAX_X/Y:",
        GLOBAL_MAX_X, GLOBAL_MAX_Y,
    )
    return GLOBAL_MAX_X, GLOBAL_MAX_Y


def render_sequence_paper(seq, norm_width, norm_height):
    """
    Draw on screen-resolution canvas, then resize plot width to 448 and pad.
    Matches paper Section III-D.3 and IV-E (no inner margin; width -> 448).
    """
    if len(seq) < 2:
        return None

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)

    W = max(float(norm_width), 1.0)
    H = max(float(norm_height), 1.0)
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

    h, w = canvas.shape[:2]
    target = int(TARGET_SIZE)

    # Paper IV-E: larger side (width) resized to 448, aspect ratio preserved.
    new_w = target
    new_h = max(1, int(round(h * target / max(w, 1))))

    resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left

    final = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return final


def draw_sequence_paper(seq, save_path, norm_width, norm_height):
    final = render_sequence_paper(seq, norm_width, norm_height)
    if final is None:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final)


def _session_sequences(dataset, path, screen_width, fuse_fraction):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)

    events = df.to_dict("records")
    if len(events) < 2:
        return []

    min_length = max(float(screen_width) * float(fuse_fraction), 1.0)
    sequences = split_by_time(events)
    sequences = merge_sequences(sequences, min_length)
    return [seq for seq in sequences if len(seq) >= 2]


def count_samples(dataset, data_root, user_res, fuse_fraction):
    total = 0
    for user in list_users(data_root):
        norm_w, _ = _resolution_bounds(user, user_res)
        user_dir = os.path.join(data_root, user)
        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            total += len(_session_sequences(dataset, path, norm_w, fuse_fraction))
    return total


def bgr_to_tensor_chw(img):
    return img.transpose(2, 0, 1)


def process_dataset_tensors(dataset, data_root, out_dir, user_res, fuse_fraction):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Fuse fraction (of screen width):", fuse_fraction)
    print("Time threshold (s):", TIME_THRESHOLD)
    print("Per-user resolutions loaded:", len(user_res))
    print("\n[Phase] Generating paper fused-curve tensors...")

    total_samples = count_samples(dataset, data_root, user_res, fuse_fraction)
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
        norm_w, norm_h = _resolution_bounds(user, user_res)

        print("\n------------------------------")
        print("User:", user, "| screen W×H (max idx):", norm_w, norm_h)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            sequences = _session_sequences(dataset, path, norm_w, fuse_fraction)
            print(f"   Session: {session} -> {len(sequences)} sequences")

            for seq in sequences:
                img = render_sequence_paper(seq, norm_w, norm_h)
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


def process_dataset(dataset, data_root, out_dir, user_res, fuse_fraction, tensors=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir, user_res, fuse_fraction)
        return

    users = list_users(data_root)
    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Fuse fraction:", fuse_fraction)

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_w, norm_h = _resolution_bounds(user, user_res)

        print("\n------------------------------")
        print("User:", user, "| screen W×H:", norm_w, norm_h)

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

            min_length = max(norm_w * fuse_fraction, 1.0)
            sequences = split_by_time(events)
            print("      After split:", len(sequences))
            sequences = merge_sequences(sequences, min_length)
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
                draw_sequence_paper(seq, save_path, norm_w, norm_h)


def main():
    parser = argparse.ArgumentParser(
        description="Chong et al. 2018 fused-curve XYPlot (time-diff split + fuse + screen plot).",
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument(
        "--training_root",
        default=None,
        help="Training data for per-user screen resolution estimation (paper IV-A).",
    )
    parser.add_argument("--data_root", required=True, help="Sessions to render.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--fuse_fraction",
        type=float,
        default=1.0,
        help="Fused curve min arc length as fraction of screen width: 0.33, 0.5, or 1.0 (paper Table II).",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="Output images.npy / labels.npy / sessions.npy instead of PNG.",
    )
    args = parser.parse_args()

    if args.fuse_fraction not in (0.33, 0.5, 1.0):
        parser.error("--fuse_fraction must be one of: 0.33, 0.5, 1.0")

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = resolve_path(training_rel)
    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)

    print("[training_root]", training_root)
    print("[data_root]", data_root)
    print("[out_dir]", out_dir)

    user_res = build_user_resolution_from_training(args.dataset, training_root)
    print("\nUSER_SCREEN_BOUNDS (from training_root):")
    for u in sorted(user_res.keys()):
        print("  ", u, "->", user_res[u])

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        user_res,
        args.fuse_fraction,
        tensors=args.tensors,
    )
    print("\nPaper fused-curve XYPlot generation finished.")


if __name__ == "__main__":
    main()
