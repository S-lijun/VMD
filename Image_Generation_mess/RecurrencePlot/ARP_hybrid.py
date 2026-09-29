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
GLOBAL_TD_CDF = None


# ============================================================
# Dynamic Image Size
# ============================================================

def get_dynamic_image_size(chunk_size):

    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)

    return int(round(BASE_IMG_SIZE * scale))


# ============================================================
# Distribution Loading
# ============================================================

def load_velocity_distribution(path):

    data = np.load(path)

    vx = data["vx"]
    vy = data["vy"]

    print("\n[Load] vx")
    print("samples:", len(vx))
    print("min:", vx.min())
    print("max:", vx.max())

    print("\n[Load] vy")
    print("samples:", len(vy))
    print("min:", vy.min())
    print("max:", vy.max())

    return vx, vy


def load_raw_distribution(path, key):

    data = np.load(path)

    arr = data[key]

    print(f"\n[Load] {key}")
    print("samples:", len(arr))
    print("min:", arr.min())
    print("max:", arr.max())

    return arr


def build_runtime_cdf(raw_values, clip_pct, tag):

    print(f"\n[{tag}] Building runtime CDF")

    upper = np.percentile(raw_values, clip_pct)

    clipped = raw_values[raw_values <= upper]

    ranks = rankdata(clipped, method="average")

    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)

    sorted_val = clipped[order]
    sorted_cdf = cdf[order]

    print(f"[{tag}] runtime samples:", len(sorted_val))
    print(f"[{tag}] runtime max:", sorted_val.max())

    return sorted_val, sorted_cdf


# ============================================================
# Core ARP Hybrid
# ============================================================

def compute_arp_hybrid(seq, percentile=95):

    xs = seq[:,0]
    ys = seq[:,1]
    ts = seq[:,2]

    T = len(seq)

    # --------------------------------------------------------
    # Distance Recurrence
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
    # Time Difference Matrix
    # --------------------------------------------------------

    dt_matrix = np.abs(ts[:, None] - ts[None, :])

    td_norm = np.interp(
        dt_matrix,
        GLOBAL_TD_CDF[0],
        GLOBAL_TD_CDF[1],
        left=0,
        right=1
    )


    # --------------------------------------------------------
    # Triangular Hybrid R
    # --------------------------------------------------------

    r_channel = np.zeros((T, T), dtype=np.float32)

    upper = np.triu_indices(T, 1)
    lower = np.tril_indices(T, -1)

    r_channel[upper] = rp[upper]
    r_channel[lower] = 1.0 - td_norm[lower]

    np.fill_diagonal(r_channel, 0)


    # --------------------------------------------------------
    # Directional Velocity
    # --------------------------------------------------------

    dt = np.maximum(np.diff(ts), 1e-5)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

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

    vx_stripe = np.tile(vx_norm[None, :], (T, 1))
    vy_stripe = np.tile(vy_norm[None, :], (T, 1))


    # --------------------------------------------------------
    # OpenCV BGR
    # --------------------------------------------------------

    b_channel = vy_stripe
    g_channel = vx_stripe
    r_channel = r_channel

    img = np.stack([b_channel, g_channel, r_channel], axis=-1)

    return np.clip(img, 0, 1)


# ============================================================
# Drawing
# ============================================================

def draw_arp_image(seq, save_path, percentile, chunk_size):

    img = compute_arp_hybrid(seq, percentile)

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
# Cleaning
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

                    draw_arp_image(seq, save_path, percentile, chunk_size)


# ============================================================
# CLI
# ============================================================

def main():

    global GLOBAL_VX_CDF
    global GLOBAL_VY_CDF
    global GLOBAL_TD_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        required=True,
                        choices=["balabit","chaoshen","dfl"])

    parser.add_argument("--data_root",
                        required=True)

    parser.add_argument("--velocity_dist",
                        required=True)

    parser.add_argument("--td_dist",
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

    parser.add_argument("--v_percentile",
                        type=float,
                        default=95)

    parser.add_argument("--t_percentile",
                        type=float,
                        default=95)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    raw_vx, raw_vy = load_velocity_distribution(
        os.path.join(ROOT, args.velocity_dist)
    )

    GLOBAL_VX_CDF = build_runtime_cdf(raw_vx, args.v_percentile, "VX")
    GLOBAL_VY_CDF = build_runtime_cdf(raw_vy, args.v_percentile, "VY")

    raw_td = load_raw_distribution(
        os.path.join(ROOT, args.td_dist),
        "time_differences"
    )

    GLOBAL_TD_CDF = build_runtime_cdf(raw_td, args.t_percentile, "TimeDiff")

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        args.percentile
    )

    print("\nARP Hybrid generation finished.")


if __name__ == "__main__":
    main()