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
# CONFIG
# ============================================================

BASE_CHUNK_SIZE = 150
BASE_IMG_SIZE = 150

GLOBAL_VX_CDF = None
GLOBAL_VY_CDF = None

# ============================================================
# IMAGE SIZE
# ============================================================

def get_dynamic_image_size(chunk_size):
    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))

# ============================================================
# LOAD GLOBAL vx vy
# ============================================================

def load_raw_directional_velocity_distribution(path):

    data = np.load(path)

    vx = data["vx"]
    vy = data["vy"]

    print("\n[Directional Velocity Distribution]")
    print("vx samples:", len(vx))
    print("vy samples:", len(vy))

    return vx, vy


def build_runtime_cdf_signed(raw_values, clip_pct):

    lower = np.percentile(raw_values, 100 - clip_pct)
    upper = np.percentile(raw_values, clip_pct)

    clipped = raw_values[
        (raw_values >= lower) & (raw_values <= upper)
    ]

    ranks = rankdata(clipped, method="average")
    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)

    return clipped[order], cdf[order]

# ============================================================
# VELOCITY vx vy
# ============================================================

def compute_vx_vy(xs, ys, ts):

    dt = np.maximum(np.diff(ts), 1e-5)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    return vx, vy

# ============================================================
# SRP
# ============================================================

def compute_srp(seq, epsilon=0.3):

    coords = seq[:, :2].astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]

    min_x, max_x = x.min(), x.max()
    min_y, max_y = y.min(), y.max()

    scale = max(max_x - min_x, max_y - min_y)
    if scale < 1e-8:
        scale = 1e-8

    x_norm = (x - min_x) / scale
    y_norm = (y - min_y) / scale

    coords_norm = np.stack([x_norm, y_norm], axis=1)

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    dist = dist / np.sqrt(2)

    M = dist.shape[0]

    avg = np.sum(dist, axis=1) / (M - 1)
    recurrent = avg < epsilon

    dist_clipped = np.minimum(dist, epsilon)

    rp = np.where(
        recurrent[:, None] & recurrent[None, :],
        dist_clipped,
        epsilon
    ).astype(np.float32)

    return rp

# ============================================================
# SRP + vx vy
# ============================================================

def compute_srp_vxvy(seq, epsilon=0.3):

    xs = seq[:,0]
    ys = seq[:,1]
    ts = seq[:,2]

    T = len(seq)

    # ---------- SRP ----------
    rp = compute_srp(seq, epsilon)
    rp = rp / epsilon

    # ---------- vx vy ----------
    vx, vy = compute_vx_vy(xs, ys, ts)

    vx_norm = np.interp(
        vx,
        GLOBAL_VX_CDF[0],
        GLOBAL_VX_CDF[1],
        left=0,
        right=1
    )

    vy_norm = np.interp(
        vy,
        GLOBAL_VY_CDF[0],
        GLOBAL_VY_CDF[1],
        left=0,
        right=1
    )

    stripe_x = np.tile(vx_norm[None, :], (T, 1))
    stripe_y = np.tile(vy_norm[None, :], (T, 1))

    # BGR
    img = np.stack([stripe_y, stripe_x, rp], axis=-1)

    return np.clip(img, 0, 1)

# ============================================================
# DRAW
# ============================================================

def draw(seq, save_path, epsilon, chunk_size):

    img = compute_srp_vxvy(seq, epsilon)

    img_size = get_dynamic_image_size(chunk_size)

    img = (img * 255).astype(np.uint8)

    if img.shape[0] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

# ============================================================
# CLEANING（Move-only）
# ============================================================

def clean_balabit(df):

    df = df.rename(columns={
        "client timestamp":"time",
        "x":"x",
        "y":"y",
        "state":"state"
    })

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
# PROCESS（protocol1）
# ============================================================

def process_dataset(dataset, data_root, out_dir, sizes, epsilon):

    users = sorted(os.listdir(data_root))

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue

        print("\nUser:", user)

        for file in sorted(os.listdir(user_dir)):

            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)
            elif dataset == "chaoshen":
                df = clean_chaoshen(df)
            elif dataset == "dfl":
                df = clean_dfl(df)

            events = df[["x","y","time"]].values.astype(np.float32)

            for chunk_size in sizes:

                if "train" in data_root.lower():
                    stride = chunk_size // 4
                else:
                    stride = chunk_size

                windows = [
                    events[i:i+chunk_size]
                    for i in range(0, len(events)-chunk_size+1, stride)
                ]

                for i, seq in enumerate(windows):

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw(seq, save_path, epsilon, chunk_size)

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
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--v_percentile", type=float, default=97.5)

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
        args.sizes,
        args.epsilon
    )

    print("\nDONE.")

if __name__ == "__main__":
    main()