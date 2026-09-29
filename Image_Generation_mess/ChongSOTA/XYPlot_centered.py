# -*- coding: utf-8 -*-
"""
Per-sequence centered XYPlot: preserve each trajectory's shape (bbox fit + center),
no screen-width / per-user bandwidth normalization.

Segmentation matches XYPlot_global (split_by_time + merge with GLOBAL_MAX_X).
Drawing: per-sequence bbox fit + center (no screen-width normalization).
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
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
)

TENSOR_SUBDIR = "Chong_centered"


def natural_key(string):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", string)]


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


def render_sequence_centered(seq):
    """
    Fit trajectory bbox into TARGET_SIZE×TARGET_SIZE with uniform scale, centered.
    Same logic as Image_Generation/XYPlot/XYPlot.py draw_mouse_chunk.
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


def _session_sequences(dataset, path):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)

    events = df.to_dict("records")
    if len(events) < 2:
        return []

    sequences = split_by_time(events)
    sequences = merge_sequences(sequences, GLOBAL_MAX_X)
    return [seq for seq in sequences if len(seq) >= 2]


def count_samples(dataset, data_root):
    total = 0
    for user in list_users(data_root):
        user_dir = os.path.join(data_root, user)
        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            total += len(_session_sequences(dataset, path))
    return total


def bgr_to_tensor_chw(img):
    """Match Images_convert.py: BGR HWC -> (3, H, W) uint8."""
    return img.transpose(2, 0, 1)


def process_dataset_tensors(dataset, data_root, out_dir):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Rendering: per-sequence bbox fit, centered in", TARGET_SIZE, "x", TARGET_SIZE)
    print("\n[Phase] Generating centered XYPlot tensors (Images_convert format)...")

    total_samples = count_samples(dataset, data_root)
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

        print("\n------------------------------")
        print("User:", user)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            sequences = _session_sequences(dataset, path)
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


def process_dataset(dataset, data_root, out_dir, tensors=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir)
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Rendering: per-sequence bbox fit, centered in", TARGET_SIZE, "x", TARGET_SIZE)

    for user in users:
        user_dir = os.path.join(data_root, user)

        print("\n------------------------------")
        print("User:", user)

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

            sequences = merge_sequences(sequences, GLOBAL_MAX_X)
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
                draw_sequence_centered(seq, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="直接输出 Images_convert 格式的 tensors（images.npy / labels.npy / sessions.npy），不写 PNG。",
    )
    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    print("[data_root]", data_root)
    print("[out_dir]", out_dir)

    process_dataset(args.dataset, data_root, out_dir, tensors=args.tensors)
    print("\nCentered XYPlot generation finished.")


if __name__ == "__main__":
    main()
