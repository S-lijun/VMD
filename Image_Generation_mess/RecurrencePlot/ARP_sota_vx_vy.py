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

GLOBAL_VX_CDF = None
GLOBAL_VY_CDF = None

# ============================================================
# Dynamic Image Size
# ============================================================

def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# Load Distribution
# ============================================================

def load_raw_directional_velocity_distribution(path):

    data = np.load(path)

    vx = data["vx"]
    vy = data["vy"]

    print("\n[Directional Velocity Distribution]")
    print("vx:", len(vx), vx.min(), vx.max())
    print("vy:", len(vy), vy.min(), vy.max())

    return vx, vy

# ============================================================
# Signed CDF
# ============================================================

def build_runtime_cdf_signed(raw_values, clip_pct):

    lower = np.percentile(raw_values, 100 - clip_pct)
    upper = np.percentile(raw_values, clip_pct)

    clipped = raw_values[(raw_values >= lower) & (raw_values <= upper)]

    ranks = rankdata(clipped, method="average")
    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)

    v_sorted = clipped[order]
    cdf_sorted = cdf[order]

    return v_sorted, cdf_sorted

# ============================================================
# Compute vx vy
# ============================================================

def compute_vx_vy(xs, ys, ts):

    dt = np.maximum(np.diff(ts), 1e-5)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    return vx, vy

# ============================================================
# 🔥 ARP + vx vy（FINAL）
# ============================================================

def compute_arp_vxvy(seq):

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
    # 🔥 NEW recurrence（替换 SRP）
    # --------------------------------------------------------

    def compute_recurrence(c):

        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))

        dist_norm = dist / (dist.max() + 1e-8)
        avg_dist = np.mean(dist_norm)

        epsilon = 0.3

        rec = np.where(
            dist_norm > avg_dist,
            epsilon,
            dist_norm
        )

        return 1.0 - rec

    SRP1 = compute_recurrence(seq1)
    SRP2 = compute_recurrence(seq2)

    # --------------------------------------------------------
    # ARP structure
    # --------------------------------------------------------

    U = np.triu(SRP2)
    L = np.tril(SRP1)

    arp = U + L

    # --------------------------------------------------------
    # vx vy
    # --------------------------------------------------------

    vx, vy = compute_vx_vy(xs, ys, ts)

    vx_norm = np.interp(vx, GLOBAL_VX_CDF[0], GLOBAL_VX_CDF[1], left=0, right=1)
    vy_norm = np.interp(vy, GLOBAL_VY_CDF[0], GLOBAL_VY_CDF[1], left=0, right=1)

    vx1 = vx_norm[:half]
    vx2 = vx_norm[half:]

    vy1 = vy_norm[:half]
    vy2 = vy_norm[half:]

    # --------------------------------------------------------
    # Triangle-aware stripe
    # --------------------------------------------------------

    stripe_x = np.zeros((half, half), dtype=np.float32)
    stripe_y = np.zeros((half, half), dtype=np.float32)

    for i in range(half):
        for j in range(half):

            if i < j:
                stripe_x[i, j] = vx2[j]
                stripe_y[i, j] = vy2[j]

            else:
                stripe_x[i, j] = vx1[j]
                stripe_y[i, j] = vy1[j]

    # --------------------------------------------------------
    # OpenCV BGR
    # --------------------------------------------------------

    b_channel = stripe_y
    g_channel = stripe_x
    r_channel = arp

    img = np.stack([b_channel, g_channel, r_channel], axis=-1)

    return np.clip(img, 0, 1)

# ============================================================
# Draw
# ============================================================

def draw_arp_vxvy(seq, save_path, chunk_size):

    img = compute_arp_vxvy(seq)

    img_size = get_dynamic_image_size(chunk_size)

    img = (img * 255).astype(np.uint8)

    if img.shape[0] != img_size:
        img = cv2.resize(img, (img_size, img_size),
                         interpolation=cv2.INTER_NEAREST)

    img = np.flipud(img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

# ============================================================
# CLEAN（保持原样）
# ============================================================

def clean_balabit(df):

    df = df.rename(columns={
        "client timestamp":"time",
        "x":"x",
        "y":"y",
        "state":"state"
    })

    df = df[df["state"]=="Move"].copy()

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_chaoshen(df):

    df = df.rename(columns={
        "X":"x",
        "Y":"y",
        "Timestamp":"time",
        "EventName":"event"
    })

    df = df[df["event"]=="Move"].copy()

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_dfl(df):

    df.columns = [c.strip().lower() for c in df.columns]

    if "client timestamp" in df.columns:
        df = df.rename(columns={"client timestamp":"time"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp":"time"})

    if "state" in df.columns:
        df = df[df["state"].str.lower()=="move"]

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])

# ============================================================
# PROCESS
# ============================================================

def process_dataset(dataset, data_root, out_dir, sizes):

    users = sorted(os.listdir(data_root))

    for user in users:

        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        for file in sorted(os.listdir(user_dir)):

            session = os.path.splitext(file)[0]
            path = os.path.join(user_dir, file)

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)
            elif dataset == "chaoshen":
                df = clean_chaoshen(df)
            elif dataset == "dfl":
                df = clean_dfl(df)

            events = df[["x","y","time"]].values.astype(np.float32)

            for chunk_size in sizes:

                n_chunks = len(events) // chunk_size

                for i in range(n_chunks):

                    seq = events[i*chunk_size:(i+1)*chunk_size]

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_arp_vxvy(seq, save_path, chunk_size)

# ============================================================
# MAIN
# ============================================================

def main():

    global GLOBAL_VX_CDF, GLOBAL_VY_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True,
                        choices=["balabit","chaoshen","dfl"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--velocity_dist", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[150])
    parser.add_argument("--v_percentile", type=float, default=100)

    args = parser.parse_args()

    vx_raw, vy_raw = load_raw_directional_velocity_distribution(
        os.path.join(ROOT, args.velocity_dist)
    )

    GLOBAL_VX_CDF = build_runtime_cdf_signed(vx_raw, args.v_percentile)
    GLOBAL_VY_CDF = build_runtime_cdf_signed(vy_raw, args.v_percentile)

    process_dataset(
        args.dataset,
        os.path.join(ROOT, args.data_root),
        os.path.join(ROOT, args.out_dir),
        args.sizes
    )

    print("\nDONE.")

if __name__ == "__main__":
    main()