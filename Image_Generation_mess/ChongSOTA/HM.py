# -*- coding: utf-8 -*-

"""
FAST Thin Trajectory Heatmap Representation

Changes from previous version:
- thinner trajectory
- smaller blur
- sharper heatmap
- still fast
"""

import os
import argparse

import pandas as pd
import numpy as np
import cv2

from XYPlot import (
    ROOT,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    TARGET_SIZE,
    INNER_PADDING,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
)

# ============================================================
# Training roots
# ============================================================

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
}

# ============================================================
# Clean wrapper
# ============================================================

def _clean_df(dataset, df):

    if dataset == "balabit":
        return clean_balabit(df)

    if dataset == "chaoshen":
        return clean_chaoshen(df)

    if dataset == "dfl":
        return clean_dfl(df)

    raise ValueError(dataset)

# ============================================================
# Build user max x/y
# ============================================================

def build_user_max_xy_from_training(dataset, training_root):

    user_max_xy = {}

    users = sorted(os.listdir(training_root))

    for user in users:

        user_dir = os.path.join(training_root, user)

        if not os.path.isdir(user_dir):
            continue

        max_x = 0.0
        max_y = 0.0

        saw_points = False

        for name in sorted(os.listdir(user_dir)):

            path = os.path.join(user_dir, name)

            if not os.path.isfile(path):
                continue

            df = pd.read_csv(path)

            df = _clean_df(dataset, df)

            if len(df) == 0:
                continue

            max_x = max(max_x, float(df["x"].max()))
            max_y = max(max_y, float(df["y"].max()))

            saw_points = True

        if saw_points:
            user_max_xy[user] = (max_x, max_y)

        else:
            user_max_xy[user] = (GLOBAL_MAX_X, GLOBAL_MAX_Y)

    return user_max_xy

# ============================================================
# Draw thin heatmap
# ============================================================

def draw_heatmap(seq, save_path, norm_width, norm_height):

    if len(seq) < 2:
        return

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float32)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float32)

    W = max(float(norm_width), 1.0)
    H = max(float(norm_height), 1.0)

    # ========================================================
    # Chong aspect ratio
    # ========================================================

    a = H / W

    effective_size = TARGET_SIZE - 2 * INNER_PADDING

    canvas_w = effective_size
    canvas_h = int(a * effective_size)

    canvas_h = max(canvas_h, 1)

    # ========================================================
    # Small heat canvas
    # ========================================================

    heat = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    x_pix = np.clip(
        np.rint(xs / W * (canvas_w - 1)),
        0,
        canvas_w - 1
    ).astype(np.int32)

    y_pix = np.clip(
        np.rint(ys / W * (canvas_w - 1)),
        0,
        canvas_h - 1
    ).astype(np.int32)

    # ========================================================
    # Thin trajectory accumulation
    # ========================================================

    prev = None

    for x_i, y_i in zip(x_pix, y_pix):

        if prev is not None:

            cv2.line(
                heat,
                prev,
                (int(x_i), int(y_i)),
                1.0,
                thickness=1,
                lineType=cv2.LINE_8
            )

        prev = (int(x_i), int(y_i))

    # ========================================================
    # Smaller blur
    # ========================================================

    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=1,
        sigmaY=1
    )

    # ========================================================
    # Log scaling
    # ========================================================

    heat = np.log1p(heat)

    if heat.max() > 0:
        heat = heat / heat.max()

    # ========================================================
    # Heatmap coloring
    # ========================================================

    heat_uint8 = (heat * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET
    )

    # ========================================================
    # White background: pad + replace JET low-end (blue) with white
    # ========================================================

    low_activity = heat < (5.0 / 255.0)
    heatmap[low_activity] = (255, 255, 255)

    final = np.full(
        (TARGET_SIZE, TARGET_SIZE, 3),
        (255, 255, 255),
        dtype=np.uint8
    )

    offset_y = (TARGET_SIZE - canvas_h) // 2
    offset_x = (TARGET_SIZE - canvas_w) // 2

    final[
        offset_y:offset_y + canvas_h,
        offset_x:offset_x + canvas_w
    ] = heatmap

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, final)

# ============================================================
# Dataset processing
# ============================================================

def process_dataset(dataset, data_root, out_dir, user_max_xy):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        if user in user_max_xy:
            norm_x, norm_y = user_max_xy[user]

        else:
            norm_x, norm_y = GLOBAL_MAX_X, GLOBAL_MAX_Y

        print("\n------------------------------")
        print("User:", user)
        print("Norm W×H:", norm_x, norm_y)

        session_files = sorted(os.listdir(user_dir))

        for file in session_files:

            path = os.path.join(user_dir, file)

            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]

            print("   Session:", session)

            df = pd.read_csv(path)

            df = _clean_df(dataset, df)

            events = df.to_dict("records")

            print("      Events:", len(events))

            if len(events) < 2:
                continue

            # ====================================================
            # Time split
            # ====================================================

            sequences = split_by_time(events)

            print("      After split:", len(sequences))

            # ====================================================
            # Merge short sequences
            # ====================================================

            min_length = norm_x

            sequences = merge_sequences(
                sequences,
                min_length
            )

            print("      After merge:", len(sequences))

            # ====================================================
            # Draw heatmaps
            # ====================================================

            for i, seq in enumerate(sequences):

                save_path = os.path.join(
                    out_dir,
                    "Heatmap_per_user",
                    user,
                    f"{session}-{i}.png"
                )

                draw_heatmap(
                    seq,
                    save_path,
                    norm_x,
                    norm_y
                )

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["balabit", "chaoshen", "dfl"]
    )

    parser.add_argument(
        "--training_root",
        default=None
    )

    parser.add_argument(
        "--data_root",
        required=True
    )

    parser.add_argument(
        "--out_dir",
        required=True
    )

    args = parser.parse_args()

    training_rel = (
        args.training_root
        or DEFAULT_TRAINING_ROOT[args.dataset]
    )

    training_root = os.path.join(ROOT, training_rel)

    data_root = os.path.join(ROOT, args.data_root)

    out_dir = os.path.join(ROOT, args.out_dir)

    user_max_xy = build_user_max_xy_from_training(
        args.dataset,
        training_root
    )

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        user_max_xy
    )

    print("\nFAST Thin Heatmap generation finished.")

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()