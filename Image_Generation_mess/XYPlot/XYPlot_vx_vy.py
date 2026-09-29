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
BASE_LINEWIDTH = 1
BASE_MARKERSIZE = 1

GLOBAL_VX_CDF = None
GLOBAL_VY_CDF = None


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
    ms = max(1, int(round(BASE_MARKERSIZE * scale)))

    return lw, ms


def _scaled(val, min_val, scale, offset):

    return (val - min_val) * scale + offset


# ============================================================
# Load RAW Distribution
# ============================================================

def load_raw_velocity_distribution(path):

    data = np.load(path)

    vx = data["vx"]
    vy = data["vy"]

    print("\n[Velocity] Loaded RAW vx/vy distribution")

    print("vx samples:", len(vx))
    print("vx min:", vx.min())
    print("vx max:", vx.max())

    print("vy samples:", len(vy))
    print("vy min:", vy.min())
    print("vy max:", vy.max())

    return vx, vy


# ============================================================
# Build Runtime CDF (SYMMETRIC — aligned with SRP)
# ============================================================

def build_runtime_cdf(raw_values, clip_pct, tag):

    print(f"\n[{tag}] Building runtime CDF (clip={clip_pct}%)")

    lower = np.percentile(raw_values, 100 - clip_pct)
    upper = np.percentile(raw_values, clip_pct)

    clipped = raw_values[
        (raw_values >= lower) &
        (raw_values <= upper)
    ]

    ranks = rankdata(clipped, method="average")

    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)

    sorted_val = clipped[order]
    sorted_cdf = cdf[order]

    print(f"[{tag}] runtime samples:", len(sorted_val))
    print(f"[{tag}] runtime min:", sorted_val.min())
    print(f"[{tag}] runtime max:", sorted_val.max())

    return sorted_val, sorted_cdf


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

    vx = dx / dt
    vy = dy / dt

    vx_norm = np.interp(
        vx,
        GLOBAL_VX_CDF[0],
        GLOBAL_VX_CDF[1],
        left=0,
        right=1
    )

    vx_norm = vx_norm 
    vy_norm = np.interp(
        vy,
        GLOBAL_VY_CDF[0],
        GLOBAL_VY_CDF[1],
        left=0,
        right=1
    )
    vy_norm = vy_norm 


    r_val = (vx_norm * 255).astype(np.uint8)
    g_val = (vy_norm * 255).astype(np.uint8)

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

        color = (
            0.0,           # B = constant
            int(g_val[i]), # G = vy
            int(r_val[i])  # R = vx
        )

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

    global GLOBAL_VX_CDF
    global GLOBAL_VY_CDF

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
                        default=97.5)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    vx_raw, vy_raw = load_raw_velocity_distribution(
        os.path.join(ROOT, args.velocity_dist)
    )

    GLOBAL_VX_CDF = build_runtime_cdf(vx_raw, args.clip, "VX")
    GLOBAL_VY_CDF = build_runtime_cdf(vy_raw, args.clip, "VY")

    print("\n[Step] Generating XYPlot VX/VY Images")

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes
    )

    print("\nXYPlot vx/vy generation finished.")


if __name__ == "__main__":
    main()