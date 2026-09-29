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

GLOBAL_AX_CDF = None
GLOBAL_AY_CDF = None


# ============================================================
# Dynamic Image Size
# ============================================================

def get_dynamic_image_size(chunk_size):

    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)

    return int(round(BASE_IMG_SIZE * scale))


# ============================================================
# Load RAW ax ay Distribution
# ============================================================

def load_raw_directional_acceleration_distribution(path):

    data = np.load(path)

    ax = data["vx"]
    ay = data["vy"]

    print("\n[Directional Acceleration Distribution]")

    print("\nax")
    print("samples:", len(ax))
    print("min:", ax.min())
    print("max:", ax.max())

    print("\nay")
    print("samples:", len(ay))
    print("min:", ay.min())
    print("max:", ay.max())

    return ax, ay


# ============================================================
# Build Signed Runtime CDF
# ============================================================

def build_runtime_cdf_signed(raw_values, clip_pct):

    print("\nBuilding signed runtime CDF")

    lower = np.percentile(raw_values, 100 - clip_pct)
    upper = np.percentile(raw_values, clip_pct)

    clipped = raw_values[
        (raw_values >= lower) & (raw_values <= upper)
    ]

    ranks = rankdata(clipped, method="average")

    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)

    v_sorted = clipped[order]
    cdf_sorted = cdf[order]

    print("runtime samples:", len(v_sorted))
    print("runtime min:", v_sorted.min())
    print("runtime max:", v_sorted.max())

    return v_sorted, cdf_sorted


# ============================================================
# Directional Velocity
# ============================================================

def compute_vx_vy(xs, ys, ts):

    dt = np.maximum(np.diff(ts), 1e-5)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    return vx, vy


# ============================================================
# Directional Acceleration（核心）
# ============================================================

def compute_ax_ay(xs, ys, ts):

    T = len(xs)

    if T < 3:
        return np.zeros(T), np.zeros(T)

    vx, vy = compute_vx_vy(xs, ys, ts)

    dt = np.maximum(np.diff(ts), 1e-5)

    ax = np.zeros(T)
    ay = np.zeros(T)

    ax[1:] = (vx[1:] - vx[:-1]) / dt
    ay[1:] = (vy[1:] - vy[:-1]) / dt

    ax[~np.isfinite(ax)] = 0
    ay[~np.isfinite(ay)] = 0

    return ax, ay


# ============================================================
# SRP + ax ay
# ============================================================

def compute_srp_axay(seq, percentile=95):

    xs = seq[:, 0]
    ys = seq[:, 1]
    ts = seq[:, 2]

    T = len(seq)

    # --------------------------------------------------------
    # Recurrence Plot
    # --------------------------------------------------------

    coords = seq[:, :2]

    diff = coords[:, None, :] - coords[None, :, :]

    dist = np.sqrt(np.sum(diff**2, axis=2))

    eps = np.percentile(dist, percentile)

    rec = np.where(dist <= eps, dist, eps).astype(np.float32)

    if rec.max() > rec.min():
        rec = (rec - rec.min()) / (rec.max() - rec.min())

    rp = 1.0 - rec


    # --------------------------------------------------------
    # Directional Acceleration
    # --------------------------------------------------------

    ax, ay = compute_ax_ay(xs, ys, ts)

    ax_norm = np.interp(
        ax,
        GLOBAL_AX_CDF[0],
        GLOBAL_AX_CDF[1],
        left=0,
        right=1
    )

    ay_norm = np.interp(
        ay,
        GLOBAL_AY_CDF[0],
        GLOBAL_AY_CDF[1],
        left=0,
        right=1
    )

    stripe_x = np.tile(ax_norm[None, :], (T, 1))
    stripe_y = np.tile(ay_norm[None, :], (T, 1))


    # --------------------------------------------------------
    # OpenCV BGR
    # --------------------------------------------------------

    b_channel = stripe_y   # ay
    g_channel = stripe_x   # ax
    r_channel = rp

    img = np.stack([b_channel, g_channel, r_channel], axis=-1)

    return np.clip(img, 0, 1)


# ============================================================
# Draw
# ============================================================

def draw_srp_axay(seq, save_path, percentile, chunk_size):

    img = compute_srp_axay(seq, percentile)

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
# Dataset Processing
# ============================================================

def process_dataset(dataset, data_root, out_dir, sizes, percentile):

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

            events = df[["x","y","time"]].values.astype(np.float32)

            print("      Events:", len(events))

            for chunk_size in sizes:

                n_chunks = len(events) // chunk_size

                print("      chunk", chunk_size, "->", n_chunks)

                for i in range(n_chunks):

                    seq = events[i*chunk_size:(i+1)*chunk_size]

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_srp_axay(seq, save_path, percentile, chunk_size)


# ============================================================
# CLI
# ============================================================

def main():

    global GLOBAL_AX_CDF
    global GLOBAL_AY_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        required=True,
                        choices=["balabit","chaoshen","dfl"])

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

    parser.add_argument("--percentile",
                        type=float,
                        default=95)

    parser.add_argument("--a_percentile",
                        type=float,
                        default=97.5)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)
    dist_path = os.path.join(ROOT, args.acceleration_dist)

    ax_raw, ay_raw = load_raw_directional_acceleration_distribution(dist_path)

    GLOBAL_AX_CDF = build_runtime_cdf_signed(ax_raw, args.a_percentile)
    GLOBAL_AY_CDF = build_runtime_cdf_signed(ay_raw, args.a_percentile)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        args.percentile
    )

    print("\nSRP ax ay generation finished.")


if __name__ == "__main__":
    main()