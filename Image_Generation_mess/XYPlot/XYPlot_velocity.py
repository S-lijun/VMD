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
# Base Config
# ============================================================

BASE_EVENT = 150
BASE_IMG_SIZE = 224
BASE_LINEWIDTH = 0.5
BASE_MARKERSIZE = 1.0

GLOBAL_V_RAW = None
GLOBAL_V_CDF = None


# ============================================================
# Scaling Utils
# ============================================================

def get_img_size(chunk_size):

    scale = chunk_size / BASE_EVENT
    side = math.sqrt(scale * BASE_IMG_SIZE * BASE_IMG_SIZE)

    return int(round(side))


def get_stroke_params(chunk_size):

    scale = math.sqrt(chunk_size / BASE_EVENT)

    lw = max(1, int(round(BASE_LINEWIDTH * scale)))
    ms = int(round(BASE_MARKERSIZE * scale))

    return lw, ms


def _scaled(val, min_val, scale, offset):

    return (val - min_val) * scale + offset


# ============================================================
# Load RAW Distribution
# ============================================================

def load_raw_velocity_distribution(path):

    data = np.load(path)

    velocities = data["values"] # This can be "values"

    print("\n[Velocity] Loaded RAW distribution")
    print("samples:", len(velocities))
    print("min:", velocities.min())
    print("max:", velocities.max())

    return velocities


# ============================================================
# Build Runtime CDF
# ============================================================

def build_runtime_cdf(raw_velocities, clip_pct):

    print(f"\n[Velocity] Building runtime CDF (clip={clip_pct}%)")

    v_upper = np.percentile(raw_velocities, clip_pct)

    v_clipped = raw_velocities[raw_velocities <= v_upper]

    ranks = rankdata(v_clipped, method="average")

    cdf = (ranks - 1) / (len(v_clipped) - 1 + 1e-8)

    order = np.argsort(v_clipped)

    v_sorted = v_clipped[order]
    cdf_sorted = cdf[order]

    print("[Velocity] runtime samples:", len(v_sorted))
    print("[Velocity] runtime max:", v_sorted.max())

    return v_sorted, cdf_sorted


# ============================================================
# Drawing
# ============================================================

def draw_mouse_chunk(chunk, save_path, chunk_size):

    if len(chunk) < 2:
        return

    IMG_SIZE = get_img_size(chunk_size)

    linewidth, markersize = get_stroke_params(chunk_size)

    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])
    ts = np.array([float(e["time"]) for e in chunk])

    dx = np.diff(xs)
    dy = np.diff(ys)
    dt = np.maximum(np.diff(ts), 1e-5)

    velocity = np.sqrt(dx**2 + dy**2) / dt

    # --------------------------------------------------------
    # Map velocity -> [0,1]
    # --------------------------------------------------------

    v_norm = np.interp(
        velocity,
        GLOBAL_V_CDF[0],
        GLOBAL_V_CDF[1],
        left=0.0,
        right=1.0
    )
    
    v_norm = v_norm * 0.8

    brightness = (v_norm * 255).astype(np.uint8)

    # --------------------------------------------------------
    # Spatial scaling
    # --------------------------------------------------------

    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    pad_x = range_x * 0.05
    pad_y = range_y * 0.05

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    range_x = max_x - min_x
    range_y = max_y - min_y

    scale = min(IMG_SIZE / range_x, IMG_SIZE / range_y)

    offset_x = (IMG_SIZE - range_x * scale) / 2
    offset_y = (IMG_SIZE - range_y * scale) / 2

    # --------------------------------------------------------
    # Draw trajectory
    # --------------------------------------------------------

    for i in range(len(xs) - 1):

        x1 = int(_scaled(xs[i], min_x, scale, offset_x))
        y1 = int(_scaled(ys[i], min_y, scale, offset_y))

        x2 = int(_scaled(xs[i+1], min_x, scale, offset_x))
        y2 = int(_scaled(ys[i+1], min_y, scale, offset_y))

        val = int(brightness[i])

        color = (val, val, val)

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            color,
            thickness=linewidth,
            lineType=cv2.LINE_AA
        )

        cv2.circle(
            img,
            (x2, y2),
            markersize,
            color,
            -1,
            lineType=cv2.LINE_AA
        )

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

    df = df[df["state"]=="Move"]

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

    df = df[df["event"]=="Move"]

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

        session_files = sorted(os.listdir(user_dir))

        for file in session_files:

            path = os.path.join(user_dir, file)

            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]

            print("   Session:", session)

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df)

            elif dataset == "chaoshen":
                df = clean_chaoshen(df)

            elif dataset == "dfl":
                df = clean_dfl(df)

            events = df.to_dict("records")

            print("      Events:", len(events))

            for chunk_size in sizes:

                n_chunks = len(events) // chunk_size

                print("      chunk", chunk_size, "->", n_chunks)

                for i in range(n_chunks):

                    chunk = events[i*chunk_size:(i+1)*chunk_size]

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_mouse_chunk(chunk, save_path, chunk_size)


# ============================================================
# CLI
# ============================================================

def main():

    global GLOBAL_V_RAW
    global GLOBAL_V_CDF

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        required=True,
                        choices=["balabit","chaoshen","dfl"])

    parser.add_argument("--data_root",
                        required=True)

    parser.add_argument("--velocity_dist",
                        required=True)

    parser.add_argument("--out_dir",
                        required=True)

    parser.add_argument("--sizes",
                        type=int,
                        nargs="+",
                        default=[150])

    parser.add_argument("--clip",
                        type=float,
                        default=95.0)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    dist_path = os.path.join(ROOT, args.velocity_dist)

    GLOBAL_V_RAW = load_raw_velocity_distribution(dist_path)

    GLOBAL_V_CDF = build_runtime_cdf(GLOBAL_V_RAW, args.clip)

    print("\n[Step] Generating XYPlot Velocity Images")

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes
    )

    print("\nXYPlot velocity generation finished.")


if __name__ == "__main__":
    main()