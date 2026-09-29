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
# Config
# ============================================================

BASE_CHUNK_SIZE = 300
BASE_IMG_SIZE = 300

# ============================================================
# Dynamic Image Size
# ============================================================

def get_dynamic_image_size(chunk_size):

    scale = math.sqrt(chunk_size / BASE_CHUNK_SIZE)
    return int(round(BASE_IMG_SIZE * scale))


# ============================================================
# SRP Core
# ============================================================

def compute_rp(seq, percentile=95):

    coords = seq[:, :2]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    eps = np.percentile(dist, percentile)

    rec = np.where(dist <= eps, dist, eps).astype(np.float32)

    if rec.max() > rec.min():
        rec = (rec - rec.min()) / (rec.max() - rec.min())

    return rec


def draw_rp(seq, save_path, percentile, chunk_size):

    rp = compute_rp(seq, percentile)

    img_size = get_dynamic_image_size(chunk_size)

    img = (rp * 255).astype(np.uint8)

    if img.shape[0] != img_size:
        img = cv2.resize(img, (img_size, img_size),
                         interpolation=cv2.INTER_NEAREST)

    img = 255 - img
    img = np.flipud(img)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, img)


def generate_windows(events, chunk_size, stride):

    windows = []

    if len(events) < chunk_size:
        return windows

    for start in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[start:start + chunk_size])

    return windows


# ============================================================
# Dataset Cleaning
# ============================================================

def clean_balabit(df):

    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state"
    })

    df = df[df["state"] == "Move"].copy()

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

    df = df[df["event"] == "Move"].copy()

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
# Session Detection
# ============================================================

def get_session_files(dataset, user_dir):

    files = os.listdir(user_dir)

    sessions = []

    for f in files:

        path = os.path.join(user_dir, f)

        if os.path.isfile(path):
            sessions.append(f)

    return sorted(sessions)


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

        session_files = get_session_files(dataset, user_dir)

        print("Sessions found:", len(session_files))

        for file in session_files:

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

            events = df.to_dict("records")

            events_np = np.array(
                [[e["x"],e["y"],e["time"]] for e in events],
                dtype=np.float32
            )

            print("      Events:", len(events_np))

            for chunk_size in sizes:

                n_chunks = len(events_np) // chunk_size

                print("      chunk", chunk_size, "->", n_chunks)

                for i in range(n_chunks):

                    seq = events_np[i*chunk_size:(i+1)*chunk_size]

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )
                


                    draw_rp(seq, save_path, percentile, chunk_size)


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
                        default=[300])

    parser.add_argument("--percentile",
                        type=float,
                        default=100)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        args.percentile
    )

    print("\nSRP generation finished.")


if __name__ == "__main__":
    main()