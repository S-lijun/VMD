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

GLOBAL_V_CDF = None

# ============================================================
# Dynamic Image Size
# ============================================================

def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# Velocity Distribution
# ============================================================

def load_raw_velocity_distribution(path):

    data = np.load(path)
    velocities = data["values"]

    print("\n[Velocity Distribution]")
    print("Samples:", len(velocities))
    print("Min:", velocities.min())
    print("Max:", velocities.max())

    return velocities


def build_runtime_cdf(raw_v, clip_pct):

    v_upper = np.percentile(raw_v, clip_pct)
    v_clipped = raw_v[raw_v <= v_upper]

    ranks = rankdata(v_clipped, method="average")
    cdf = (ranks - 1) / (len(v_clipped) - 1 + 1e-8)

    order = np.argsort(v_clipped)

    v_sorted = v_clipped[order]
    cdf_sorted = cdf[order]

    return v_sorted, cdf_sorted

# ============================================================
# Velocity Computation
# ============================================================

def compute_velocity(xs, ys, ts):

    dt = np.maximum(np.diff(ts), 1e-5)

    v = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
    v = np.concatenate([[v[0]], v])

    return v

# ============================================================
# 🔥 SRP + Velocity（FINAL）
# ============================================================

def compute_srp_velocity(seq):

    xs = seq[:,0]
    ys = seq[:,1]
    ts = seq[:,2]

    T = len(seq)

    # --------------------------------------------------------
    # 🔥 R channel（你要的 recurrence 逻辑）
    # --------------------------------------------------------

    coords = seq[:, :2]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    dist_norm = dist / (dist.max() + 1e-8)

    avg_dist = np.mean(dist_norm)

    epsilon = 0.3

    rp = np.where(
        dist_norm > avg_dist,
        epsilon,
        dist_norm
    )

    rp = 1.0 - rp

    # --------------------------------------------------------
    # Velocity Stripe
    # --------------------------------------------------------

    v = compute_velocity(xs, ys, ts)

    v_norm = np.interp(
        v,
        GLOBAL_V_CDF[0],
        GLOBAL_V_CDF[1],
        left=0,
        right=1
    )

    stripe = np.tile(v_norm[None, :], (T, 1))

    # --------------------------------------------------------
    # BGR（保持你原来的结构）
    # --------------------------------------------------------

    b_channel = stripe
    g_channel = rp
    r_channel = rp

    img = np.stack([b_channel, g_channel, r_channel], axis=-1)

    return np.clip(img, 0, 1)

# ============================================================
# Draw
# ============================================================

def draw_srp_velocity(seq, save_path, chunk_size):

    img = compute_srp_velocity(seq)

    img_size = get_dynamic_image_size(chunk_size)

    img = (img * 255).astype(np.uint8)

    if img.shape[0] != img_size:
        img = cv2.resize(img, (img_size, img_size),
                         interpolation=cv2.INTER_NEAREST)

    img = np.flipud(img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

# ============================================================
# CLEAN FUNCTIONS
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

    df = df.dropna(subset=["x","y","time"])

    # remove 65535
    df = df[(df["x"] < 1e4) & (df["y"] < 1e4)]

    return df


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

    print("\nDataset:", dataset)

    for user in users:

        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\nUser:", user)

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

                    draw_srp_velocity(seq, save_path, chunk_size)

# ============================================================
# MAIN
# ============================================================

def main():

    global GLOBAL_V_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True,
                        choices=["balabit","chaoshen","dfl"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--velocity_dist", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[150])
    parser.add_argument("--v_percentile", type=float, default=95)

    args = parser.parse_args()

    raw_v = load_raw_velocity_distribution(
        os.path.join(ROOT, args.velocity_dist)
    )

    GLOBAL_V_CDF = build_runtime_cdf(raw_v, args.v_percentile)

    process_dataset(
        args.dataset,
        os.path.join(ROOT, args.data_root),
        os.path.join(ROOT, args.out_dir),
        args.sizes
    )

    print("\nDONE.")

if __name__ == "__main__":
    main()