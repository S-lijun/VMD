# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2
import math

# ============================================================
# ROOT
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("[ROOT]", ROOT)

# ============================================================
# Config
# ============================================================

BASE_CHUNK_SIZE = 600

# ============================================================
# ARP (SOTA VERSION - SAME BRIGHTNESS AS SRP)
# ============================================================

def compute_arp(seq, epsilon=0.3):

    coords = seq[:, :2]

    N = coords.shape[0]

    if N % 2 != 0:
        coords = coords[:-1]
        N -= 1

    half = N // 2

    # FULL distance
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    # normalize
    dist_norm = dist / (dist.max() + 1e-8)

    # global avg
    avg_dist = np.mean(dist_norm)

    # threshold（SRP逻辑）
    srp_matrix = np.where(dist_norm > avg_dist, epsilon, dist_norm)

    # build ARP
    arp = np.zeros((half, half), dtype=np.float32)

    for i in range(half):
        for j in range(half):

            if i < j:
                arp[i, j] = srp_matrix[i, j]
            elif i > j:
                arp[i, j] = srp_matrix[i + half, j + half]
            else:
                arp[i, j] = 0.0

    return arp


# ============================================================
# Draw
# ============================================================

def draw_arp(seq, save_path, epsilon):

    arp = compute_arp(seq, epsilon)

    img = (arp * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)


# ============================================================
# Sliding Window（关键）
# ============================================================

def generate_windows(events, chunk_size, stride):

    windows = []

    for start in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[start:start + chunk_size])

    return windows


# ============================================================
# Cleaning
# ============================================================

def clean_balabit(df):

    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state"
    })

    df = df[df["state"] == "Move"].copy()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["x", "y", "time"])

    df = df[(df["x"] < 1e4) & (df["y"] < 1e4)]

    return df


# ============================================================
# Process Dataset
# ============================================================

def process_dataset(dataset, data_root, out_dir, sizes, epsilon):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        print("\nUser:", user)

        for file in sorted(os.listdir(user_dir)):

            session = os.path.splitext(file)[0]
            path = os.path.join(user_dir, file)

            print("   Session:", session)

            df = pd.read_csv(path)
            df = clean_balabit(df)

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:

                # --------------------------------------------------
                # Sliding Strategy（核心）
                # --------------------------------------------------
                if "train" in data_root.lower():
                    stride = chunk_size // 4
                else:
                    stride = chunk_size

                windows = generate_windows(events, chunk_size, stride)

                print(f"      chunk={chunk_size}, stride={stride}, windows={len(windows)}")

                for i, seq in enumerate(windows):

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_arp(seq, save_path, epsilon)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[BASE_CHUNK_SIZE])
    parser.add_argument("--epsilon", type=float, default=0.3)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        args.epsilon
    )

    print("\nARP generation finished.")


if __name__ == "__main__":
    main()