# -*- coding: utf-8 -*-
"""
SRP image generation: same pipeline as SRP_sota.py, but distance matrices are
normalized with one min/max per (session, chunk_size), computed from all sliding
windows in that session — not per-chunk and not dataset-global.
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2

# ============================================================
# ROOT
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("[ROOT]", ROOT)

# ============================================================
# SRP
# ============================================================


def compute_distance(seq):
    coords = seq[:, :2].astype(np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def compute_srp(seq, epsilon, gmin, gmax):
    dist = compute_distance(seq)
    dist = (dist - gmin) / (gmax - gmin + 1e-8)

    M = dist.shape[0]

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
# Sliding Window
# ============================================================


def generate_windows(events, chunk_size, stride):
    windows = []
    if len(events) < chunk_size:
        return windows
    for start in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[start:start + chunk_size])
    return windows


def stride_for_split(chunk_size, data_root):
    if "train" in data_root.lower():
        return chunk_size // 4
       
    return chunk_size


# ============================================================
# Per-session global min/max over all windows (off-diagonal distances)
# ============================================================


def compute_session_min_max(events, chunk_size, data_root):
    stride = stride_for_split(chunk_size, data_root)
    windows = generate_windows(events, chunk_size, stride)
    if not windows:
        return None, None

    session_min = float("inf")
    session_max = float("-inf")

    for seq in windows:
        dist = compute_distance(seq)
        mask = ~np.eye(dist.shape[0], dtype=bool)
        vals = dist[mask]
        session_min = min(session_min, float(vals.min()))
        session_max = max(session_max, float(vals.max()))

    return session_min, session_max


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

    df = df[(df["x"] < 1e4) & (df["y"] < 1e4)]

    return df.dropna(subset=["x", "y", "time"])

def clean_chaoshen(df):
    df = df.rename(columns={
        "X": "x",
        "Y": "y",
        "Timestamp": "time",
        "EventName": "event"
    })
    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y", "time"])


def clean_dfl(df):
    df.columns = [c.strip().lower() for c in df.columns]
    if "client timestamp" in df.columns:
        df = df.rename(columns={"client timestamp": "time"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    if "state" in df.columns:
        df = df[df["state"].str.lower() == "move"]
    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y", "time"])


# ============================================================
# Session Detection
# ============================================================


def get_session_files(dataset, user_dir):
    files = os.listdir(user_dir)
    sessions = []
    for f in files:
        path = os.path.join(user_dir, f)
        if os.path.isfile(path):
            sessions.append(f)
    return sorted(sessions)


# ============================================================
# Process Dataset
# ============================================================


def process_dataset(dataset, data_root, out_dir, sizes, epsilon):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("\nUser:", user)

        session_files = get_session_files(dataset, user_dir)
        print("Sessions found:", len(session_files))

        for file in session_files:
            session = os.path.splitext(file)[0]
            path = os.path.join(user_dir, file)

            print("   Session:", session)

            df = pd.read_csv(
                path,
                sep=",",
                engine="python",
                header=0
            )

            if dataset == "balabit":
                df = clean_balabit(df)
            elif dataset == "chaoshen":
                df = clean_chaoshen(df)
            elif dataset == "dfl":
                df = clean_dfl(df)

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:
                stride = stride_for_split(chunk_size, data_root)
                gmin, gmax = compute_session_min_max(events, chunk_size, data_root)

                windows = generate_windows(events, chunk_size, stride)
                print(
                    f"      chunk={chunk_size}, stride={stride}, windows={len(windows)} "
                    f"| session dist min/max=({gmin}, {gmax})"
                )

                if gmin is None or gmax is None:
                    continue

                for i, seq in enumerate(windows):
                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )
                    draw_srp(seq, save_path, epsilon, gmin, gmax)


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300])
    parser.add_argument("--epsilon", type=float, default=1)

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

    print("\nSRP (per-session global norm) generation finished.")


if __name__ == "__main__":
    main()
