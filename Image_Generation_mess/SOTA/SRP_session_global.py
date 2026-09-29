# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ============================================================
# CLEAN
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

    df = df[(df["x"] < 1e4) & (df["y"] < 1e4)]

    return df.dropna(subset=["x", "y", "time"])

# ============================================================
# SLIDING WINDOW（保持你原逻辑）
# ============================================================

def generate_windows(events, chunk_size, data_root):

    if len(events) < chunk_size:
        return []

    # 🔥 自动判断 train/test
    if "train" in data_root.lower():
        stride = chunk_size // 4
    else:
        stride = chunk_size

    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i+chunk_size])

    return windows

# ============================================================
# DISTANCE
# ============================================================

def compute_distance(seq):

    coords = seq[:, :2].astype(np.float32)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    return dist

# ============================================================
# GLOBAL MIN/MAX（🔥关键）
# ============================================================

def compute_global_min_max(data_root, chunk_size):

    global_min = float("inf")
    global_max = float("-inf")

    users = sorted(os.listdir(data_root))

    print("\n[Phase 1] Computing global min/max...")

    for user in users:

        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        for file in os.listdir(user_dir):

            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            df = pd.read_csv(path)
            df = clean_balabit(df)

            events = df[["x","y","time"]].values.astype(np.float32)

            windows = generate_windows(events, chunk_size, data_root)

            for seq in windows:

                dist = compute_distance(seq)

                mask = ~np.eye(dist.shape[0], dtype=bool)
                vals = dist[mask]

                global_min = min(global_min, vals.min())
                global_max = max(global_max, vals.max())

    print(f"[GLOBAL] min={global_min:.6f}, max={global_max:.6f}")

    return global_min, global_max

# ============================================================
# SRP（GLOBAL NORMALIZATION）
# ============================================================

def compute_srp(seq, epsilon, gmin, gmax):

    coords = seq[:, :2].astype(np.float32)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    # GLOBAL normalization
    dist = (dist - gmin) / (gmax - gmin + 1e-8)

    M = dist.shape[0]

    # avg（exclude self）
    sum_dist = np.sum(dist, axis=1) - np.diag(dist)
    avg = sum_dist / (M - 1)

    recurrent = avg <= epsilon

    dist_clipped = np.minimum(dist, epsilon)

    rp = np.where(
        recurrent[:, None] & recurrent[None, :],
        dist_clipped,
        epsilon
    ).astype(np.float32)

    return rp

def draw_srp(seq, save_path, epsilon, gmin, gmax):

    rp = compute_srp(seq, epsilon, gmin, gmax)

    img = (rp / epsilon * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

# ============================================================
# MAIN PROCESS
# ============================================================

def process_dataset(dataset, data_root, out_dir, sizes, epsilon):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))

    # Phase 1：global min/max
    gmin, gmax = compute_global_min_max(data_root, sizes[0])

    print("\n[Phase 2] Generating SRP...")

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue
    
        print("\n------------------------------")
        print("\nUser:", user)

        for file in os.listdir(user_dir):

            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:

                windows = generate_windows(events, chunk_size, data_root)

                print(f"  Session {session} | chunk={chunk_size} → {len(windows)} windows")

                for i, seq in enumerate(windows):

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_srp(seq, save_path, epsilon, gmin, gmax)

# ============================================================
# CLI（你的旧风格）
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300])
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

    print("\nDone.")

if __name__ == "__main__":
    main()