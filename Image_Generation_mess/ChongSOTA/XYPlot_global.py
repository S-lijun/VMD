# -*- coding: utf-8 -*-
"""
Global normalization: all users share GLOBAL_MAX_X / GLOBAL_MAX_Y (1919 x 1079).
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
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
    draw_sequence,
    render_sequence,
)

TENSOR_SUBDIR = "Chong_global"


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
    print("Global norm W×H:", GLOBAL_MAX_X, GLOBAL_MAX_Y)
    print("\n[Phase] Generating global XYPlot tensors (Images_convert format)...")

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
        print("User:", user, "| norm W×H (global):", GLOBAL_MAX_X, GLOBAL_MAX_Y)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            sequences = _session_sequences(dataset, path)
            print(f"   Session: {session} -> {len(sequences)} sequences")

            for seq in sequences:
                img = render_sequence(seq, GLOBAL_MAX_X, GLOBAL_MAX_Y)
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


def process_dataset(dataset, data_root, out_dir, tensors=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir)
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Using global max x/y:", GLOBAL_MAX_X, GLOBAL_MAX_Y)

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
                draw_sequence(seq, save_path, GLOBAL_MAX_X, GLOBAL_MAX_Y)


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
    print("\nGlobal XYPlot generation finished.")


if __name__ == "__main__":
    main()
