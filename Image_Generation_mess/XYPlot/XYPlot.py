# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np
import cv2
import math

# ============================================================
# ROOT
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("[ROOT]", ROOT)

# ============================================================
# Global Drawing Config
# ============================================================

BASE_EVENT = 60
BASE_IMG_SIZE = 224
BASE_LINEWIDTH = 1
BASE_MARKERSIZE = 1


# ============================================================
# Scaling Utilities
# ============================================================

def get_img_size(chunk_size):

    scale = chunk_size / BASE_EVENT
    side = math.sqrt(scale * BASE_IMG_SIZE * BASE_IMG_SIZE)

    return int(round(side))


def get_stroke_params(chunk_size):

    scale = math.sqrt(chunk_size / BASE_EVENT)

    linewidth = BASE_LINEWIDTH * scale
    markersize = BASE_MARKERSIZE * scale

    return linewidth, markersize


def _scaled(val, min_val, scale, offset):

    return (val - min_val) * scale + offset


# ============================================================
# XYPlot drawing (OpenCV)
# ============================================================

def draw_mouse_chunk(chunk, save_path, chunk_size):

    if len(chunk) < 2:
        return

    img_size = get_img_size(chunk_size)

    linewidth, markersize = get_stroke_params(chunk_size)

    thickness = max(1, int(round(linewidth * 2)))
    radius = max(1, int(round(markersize)))

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    xs = np.array([float(e["x"]) for e in chunk])
    ys = np.array([float(e["y"]) for e in chunk])

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

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

    scale = min(img_size / range_x, img_size / range_y)

    offset_x = (img_size - range_x * scale) / 2
    offset_y = (img_size - range_y * scale) / 2


    prev = None

    for x, y in zip(xs, ys):

        x_s = int((x - min_x) * scale + offset_x)
        y_s = int((y - min_y) * scale + offset_y)

        if prev is not None:

            cv2.line(
                img,
                prev,
                (x_s, y_s),
                (0,0,0),
                thickness,
                lineType=cv2.LINE_AA
            )

        cv2.circle(
            img,
            (x_s, y_s),
            radius,
            (0,0,0),
            -1
        )

        prev = (x_s, y_s)

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

    df = df[df["state"] == "Move"]

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

    df = df[df["event"] == "Move"]

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
        df = df[df["state"].str.lower() == "move"]

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


# ============================================================
# Dataset Processing (same as SRP)
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

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        required=True,
                        choices=["balabit","chaoshen","dfl"])

    parser.add_argument("--data_root",
                        required=True)

    parser.add_argument("--out_dir",
                        required=True)

    parser.add_argument("--sizes",
                        type=int,
                        nargs="+",
                        default=[60])

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes
    )

    print("\nXYPlot generation finished.")


if __name__ == "__main__":
    main()