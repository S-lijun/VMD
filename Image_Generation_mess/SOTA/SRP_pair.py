# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


"""
Clean up Balabit dataset (no global normalization here).
"""
def clean_balabit(df):
    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state"
    })

    df = df[(df["x"] < 65536) & (df["y"] < 65536)]
    #df = df.drop_duplicates()
    df = df[df["state"] == "Move"].copy()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])


"""
Generate train/test windows.
"""
def generate_windows(events, chunk_size, data_root):
    if len(events) < chunk_size:
        return []

    if "train" in data_root.lower():
        #stride = max(1, chunk_size // 4)
        stride = chunk_size
    else:
        stride = chunk_size

    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i + chunk_size])
    return windows


"""
Per-sequence local normalization and pair-wise SRP.

Steps:
1) Use x,y only from one sequence.
2) Compute range_x/range_y in this sequence.
3) scale = max(range_x, range_y) (guarded by 1e-8).
4) Normalize x,y with same scale.
5) Pair-wise distance on normalized coordinates.
6) If distance < epsilon => keep; else clip to epsilon.
"""

'''
def compute_srp_pair(seq, epsilon):
    coords = seq[:, :2].astype(np.float32)

    min_xy = np.min(coords, axis=0, keepdims=True)
    max_xy = np.max(coords, axis=0, keepdims=True)
    ranges = max_xy - min_xy
    max_range = float(np.max(ranges))
    scale = max(max_range, 1e-8)

    coords_norm = (coords - min_xy) / scale

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    rp = np.minimum(dist, epsilon)
    return rp
'''

def compute_srp_pair(seq, epsilon):
    coords = seq[:, :2].astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    scale = max(x_max - x_min, y_max - y_min)
    if scale < 1e-8:
        scale = 1e-8

    x_norm = (x - x_min) / scale
    y_norm = (y - y_min) / scale

    coords_norm = np.stack([x_norm, y_norm], axis=1)

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    rp = np.minimum(dist, epsilon)
    return rp

"""
Save SRP to image (largest values to 255, smallest to 0).
"""
def draw_srp(seq, save_path, epsilon):
    rp = compute_srp_pair(seq, epsilon)

    rp_min = rp.min()
    rp_max = rp.max()
    print(rp_min, rp_max)
    denom = max(rp_max - rp_min, 1e-8)

    img = ((rp - rp_min) / denom * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


"""
Convert dataset into windows and SRP images.
"""
def process_dataset(dataset, data_root, out_dir, sizes, epsilon):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("\n[Phase] Generating pair-wise SRP (local normalization)...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("User:", user)

        for file in os.listdir(user_dir):
            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]
            df = pd.read_csv(path)

            if dataset.lower() == "balabit":
                df = clean_balabit(df)
            else:
                for c in ["x", "y", "time"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["x", "y", "time"])

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:
                windows = generate_windows(events, chunk_size, data_root)
                print(f"  Session {session} | chunk={chunk_size} -> {len(windows)} windows")

                for i, seq in enumerate(windows):
                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )
                    draw_srp(seq, save_path, epsilon)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[120])
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)

    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        epsilon=args.epsilon
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
