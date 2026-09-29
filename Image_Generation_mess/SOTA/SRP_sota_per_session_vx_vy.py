# -*- coding: utf-8 -*-
"""
SRP + vx/vy RGB encoding (same layout as SRP_vx_vy.py):
  R = recurrent plot distance channel
  G = vx horizontal stripe (tiled vertically)
  B = vy horizontal stripe (tiled vertically)

Distance channel uses per-(session, chunk_size) min/max over all sliding windows
in that session (same as SRP_sota_per_session_global.py), not per-chunk coord
normalization like SRP_vx_vy.py.

vx/vy are mapped with the same global velocity distribution + CDF as SRP_vx_vy.py.
"""

import os
import argparse
import math

import cv2
import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ============================================================
# ROOT
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("[ROOT]", ROOT)

# ============================================================
# CONFIG (match SRP_vx_vy.py)
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


def print_velocity_channel_stats(label, raw_values, clip_pct):
    """Print raw min/max/median/mean/std, then min/max after percentile clip."""
    arr = np.asarray(raw_values, dtype=np.float64).ravel()
    print(f"\n[{label}] raw (n={arr.size})")
    print(
        f"  min={arr.min():.8g}  max={arr.max():.8g}  median={np.median(arr):.8g}  "
        f"mean={arr.mean():.8g}  std={arr.std():.8g}"
    )
    lower = np.percentile(arr, 100 - clip_pct)
    upper = np.percentile(arr, clip_pct)
    clipped = arr[(arr >= lower) & (arr <= upper)]
    print(
        f"  clip_pct={clip_pct}  percentiles [{100 - clip_pct:g}, {clip_pct:g}]  "
        f"bounds [{lower:.8g}, {upper:.8g}]  kept n={clipped.size}"
    )
    if clipped.size == 0:
        print("  after clip: min/max = (empty)")
    else:
        print(
            f"  after clip: min={clipped.min():.8g}  max={clipped.max():.8g}"
        )


def build_runtime_cdf_signed(raw_values, clip_pct):
    lower = np.percentile(raw_values, 100 - clip_pct)
    upper = np.percentile(raw_values, clip_pct)
    clipped = raw_values[(raw_values >= lower) & (raw_values <= upper)]
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
# SRP (per-session distance min/max, same as SRP_sota_per_session_global)
# ============================================================


def compute_distance(seq):
    coords = seq[:, :2].astype(np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def compute_srp(seq, epsilon, gmin, gmax):
    dist = compute_distance(seq)
    dist = (dist - gmin) / (gmax - gmin + 1e-8)
    m = dist.shape[0]
    sum_dist = np.sum(dist, axis=1) - np.diag(dist)
    avg = sum_dist / (m - 1)
    recurrent = avg < epsilon
    dist_clipped = np.minimum(dist, epsilon)
    rp = np.where(
        recurrent[:, None] & recurrent[None, :],
        dist_clipped,
        epsilon,
    ).astype(np.float32)
    return rp


# ============================================================
# SRP + vx vy (BGR layout matches SRP_vx_vy.compute_srp_vxvy)
# ============================================================


def compute_srp_vxvy(seq, epsilon, gmin, gmax):
    xs = seq[:, 0]
    ys = seq[:, 1]
    ts = seq[:, 2]
    t = len(seq)

    rp = compute_srp(seq, epsilon, gmin, gmax)
    rp = rp / epsilon

    vx, vy = compute_vx_vy(xs, ys, ts)
    vx_norm = np.interp(
        vx, GLOBAL_VX_CDF[0], GLOBAL_VX_CDF[1], left=0, right=1
    )
    vy_norm = np.interp(
        vy, GLOBAL_VY_CDF[0], GLOBAL_VY_CDF[1], left=0, right=1
    )

    stripe_x = np.tile(vx_norm[None, :], (t, 1))
    stripe_y = np.tile(vy_norm[None, :], (t, 1))

    img = np.stack([stripe_y, stripe_x, rp], axis=-1)
    return np.clip(img, 0, 1)


def draw(seq, save_path, epsilon, chunk_size, gmin, gmax):
    img = compute_srp_vxvy(seq, epsilon, gmin, gmax)
    img_size = get_dynamic_image_size(chunk_size)
    img = (img * 255).astype(np.uint8)
    if img.shape[0] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


# ============================================================
# Sliding window (SRP_sota_per_session_global)
# ============================================================


def generate_windows(events, chunk_size, stride):
    windows = []
    if len(events) < chunk_size:
        return windows
    for start in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[start : start + chunk_size])
    return windows


def stride_for_split(chunk_size, data_root):
    if "train" in data_root.lower():
        return chunk_size // 4
    return chunk_size


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
# Cleaning (match SRP_sota_per_session_global)
# ============================================================


def clean_balabit(df):
    df = df.rename(
        columns={
            "client timestamp": "time",
            "x": "x",
            "y": "y",
            "state": "state",
        }
    )
    df = df[df["state"] == "Move"].copy()
    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["x"] < 1e4) & (df["y"] < 1e4)]
    return df.dropna(subset=["x", "y", "time"])


def clean_chaoshen(df):
    df = df.rename(
        columns={
            "X": "x",
            "Y": "y",
            "Timestamp": "time",
            "EventName": "event",
        }
    )
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


def get_session_files(user_dir):
    sessions = []
    for f in os.listdir(user_dir):
        path = os.path.join(user_dir, f)
        if os.path.isfile(path):
            sessions.append(f)
    return sorted(sessions)


# ============================================================
# Process dataset
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
        session_files = get_session_files(user_dir)
        print("Sessions found:", len(session_files))

        for file in session_files:
            session = os.path.splitext(file)[0]
            path = os.path.join(user_dir, file)
            print("   Session:", session)

            df = pd.read_csv(path, sep=",", engine="python", header=0)
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
                        f"{session}-{i}.png",
                    )
                    draw(seq, save_path, epsilon, chunk_size, gmin, gmax)


def main():
    global GLOBAL_VX_CDF, GLOBAL_VY_CDF

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["balabit", "chaoshen", "dfl"],
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--velocity_dist", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[150])
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--v_percentile", type=float, default=97.5)

    args = parser.parse_args()

    vx_raw, vy_raw = load_raw_directional_velocity_distribution(
        os.path.join(ROOT, args.velocity_dist)
    )
    clip_pct = args.v_percentile
    print("\n[Global velocity] stats (--v_percentile used as clip_pct for CDF)")
    print_velocity_channel_stats("vx", vx_raw, clip_pct)
    print_velocity_channel_stats("vy", vy_raw, clip_pct)

    GLOBAL_VX_CDF = build_runtime_cdf_signed(vx_raw, args.v_percentile)
    GLOBAL_VY_CDF = build_runtime_cdf_signed(vy_raw, args.v_percentile)

    process_dataset(
        args.dataset,
        os.path.join(ROOT, args.data_root),
        os.path.join(ROOT, args.out_dir),
        args.sizes,
        args.epsilon,
    )

    print("\nSRP (per-session dist norm) + vx/vy stripes — finished.")


if __name__ == "__main__":
    main()
