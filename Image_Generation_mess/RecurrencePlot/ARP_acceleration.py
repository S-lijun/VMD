# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2
import math
from scipy.stats import rankdata

# ============================================================
# ROOT
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("[ROOT]", ROOT)

# ============================================================
# Config
# ============================================================

BASE_CHUNK_SIZE = 150
BASE_IMG_SIZE = 224

GLOBAL_A_CDF = None


# ============================================================
# Dynamic Image Size
# ============================================================

def get_dynamic_image_size(chunk_size):

    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)

    return int(round(BASE_IMG_SIZE * scale))


# ============================================================
# Acceleration Distribution
# ============================================================

def load_raw_acceleration_distribution(path):

    data = np.load(path)

    acc = data["values"]

    print("\n[Acceleration Distribution]")
    print("Samples:", len(acc))
    print("Min:", acc.min())
    print("Max:", acc.max())

    return acc


def build_runtime_cdf(raw_a, clip_pct):

    print("\nBuilding acceleration runtime CDF")

    raw_a = np.abs(raw_a)

    a_upper = np.percentile(raw_a, clip_pct)

    a_clipped = raw_a[raw_a <= a_upper]

    ranks = rankdata(a_clipped, method="average")

    cdf = (ranks - 1) / (len(a_clipped) - 1 + 1e-8)

    order = np.argsort(a_clipped)

    a_sorted = a_clipped[order]
    cdf_sorted = cdf[order]

    print("Runtime samples:", len(a_sorted))
    print("Runtime max:", a_sorted.max())

    return a_sorted, cdf_sorted


# ============================================================
# Velocity
# ============================================================

def compute_velocity(xs, ys, ts):

    dt = np.maximum(np.diff(ts), 1e-5)

    dx = np.diff(xs)
    dy = np.diff(ys)

    v = np.sqrt(dx**2 + dy**2) / dt

    v = np.concatenate([[v[0]], v])

    return v


# ============================================================
# Acceleration
# ============================================================

def compute_acceleration(xs, ys, ts):

    T = len(xs)

    if T < 3:
        return np.zeros(T)

    v = compute_velocity(xs, ys, ts)

    dt = np.maximum(np.diff(ts), 1e-5)

    a = np.zeros(T)

    a[1:] = (v[1:] - v[:-1]) / dt

    a[~np.isfinite(a)] = 0

    return a


# ============================================================
# ARP + Acceleration
# ============================================================

def compute_arp_acceleration(seq, percentile=95):

    coords = seq[:, :2]
    xs = seq[:,0]
    ys = seq[:,1]
    ts = seq[:,2]

    T = len(seq)
    half = T // 2

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    seq1 = coords[:half]
    seq2 = coords[half:]

    # --------------------------------------------------------
    # SRP for both halves（完全照抄）
    # --------------------------------------------------------

    def compute_srp(c):
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))

        eps = np.percentile(dist, percentile)
        rec = np.where(dist <= eps, dist, eps).astype(np.float32)

        if rec.max() > rec.min():
            rec = (rec - rec.min()) / (rec.max() - rec.min())

        return 1.0 - rec

    SRP1 = compute_srp(seq1)
    SRP2 = compute_srp(seq2)

    # --------------------------------------------------------
    # ARP structure（关键！！！）
    # --------------------------------------------------------

    U = np.triu(SRP2)
    L = np.tril(SRP1)

    arp = U + L


    # --------------------------------------------------------
    # Acceleration（唯一改动）
    # --------------------------------------------------------

    a = compute_acceleration(xs, ys, ts)

    a_mag = np.abs(a)

    a_norm = np.interp(
        a_mag,
        GLOBAL_A_CDF[0],
        GLOBAL_A_CDF[1],
        left=0,
        right=1
    )

    a1 = a_norm[:half]
    a2 = a_norm[half:]


    # --------------------------------------------------------
    # Triangle-aware stripe（完全照抄）
    # --------------------------------------------------------

    stripe = np.zeros((half, half), dtype=np.float32)

    for i in range(half):
        for j in range(half):

            if i < j:
                stripe[i, j] = a2[j]   # 上三角 → 后半段

            elif i > j:
                stripe[i, j] = a1[j]   # 下三角 → 前半段

            else:
                stripe[i, j] = a1[j]   # 对角线


    # --------------------------------------------------------
    # BGR（完全一样）
    # --------------------------------------------------------

    b_channel = stripe
    g_channel = stripe
    r_channel = arp

    img = np.stack([b_channel, g_channel, r_channel], axis=-1)

    return np.clip(img, 0, 1)


# ============================================================
# Draw
# ============================================================

def draw_arp_acceleration(seq, save_path, chunk_size):

    img = compute_arp_acceleration(seq)

    img_size = get_dynamic_image_size(chunk_size)

    img = (img * 255).astype(np.uint8)

    if img.shape[0] != img_size:

        img = cv2.resize(
            img,
            (img_size, img_size),
            interpolation=cv2.INTER_NEAREST
        )

    img = np.flipud(img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)


# ============================================================
# Cleaning（不变）
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

    return df.dropna(subset=["x", "y", "time"])


def clean_chaoshen(df):

    df = df.rename(columns={
        "X": "x",
        "Y": "y",
        "Timestamp": "time",
        "EventName": "event"
    })

    df = df[df["event"] == "Move"].copy()

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
# Dataset Processing（不变）
# ============================================================

def process_dataset(dataset, data_root, out_dir, sizes):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("User:", user)

        for file in sorted(os.listdir(user_dir)):

            session = os.path.splitext(file)[0]

            path = os.path.join(user_dir, file)

            print("   Session:", session)

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)
            elif dataset == "chaoshen":
                df = clean_chaoshen(df)
            elif dataset == "dfl":
                df = clean_dfl(df)

            events = df[["x", "y", "time"]].values.astype(np.float32)

            print("      Events:", len(events))

            for chunk_size in sizes:

                n_chunks = len(events) // chunk_size

                print("      chunk", chunk_size, "->", n_chunks)

                for i in range(n_chunks):

                    seq = events[i * chunk_size:(i + 1) * chunk_size]

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_arp_acceleration(seq, save_path, chunk_size)


# ============================================================
# CLI
# ============================================================

def main():

    global GLOBAL_A_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        required=True,
                        choices=["balabit", "chaoshen", "dfl"])

    parser.add_argument("--data_root",
                        required=True)

    parser.add_argument("--acceleration_dist",
                        required=True)

    parser.add_argument("--out_dir",
                        required=True)

    parser.add_argument("--sizes",
                        type=int,
                        nargs="+",
                        default=[150])

    parser.add_argument("--a_percentile",
                        type=float,
                        default=97.5)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)
    dist_path = os.path.join(ROOT, args.acceleration_dist)

    raw_a = load_raw_acceleration_distribution(dist_path)

    GLOBAL_A_CDF = build_runtime_cdf(raw_a, args.a_percentile)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes
    )

    print("\nARP Acceleration generation finished.")


if __name__ == "__main__":
    main()