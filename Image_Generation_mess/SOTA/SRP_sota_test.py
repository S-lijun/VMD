# -*- coding: utf-8 -*-

import os
import argparse
from matplotlib import rc_params_from_file
import pandas as pd
import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


'''
Clean up Balabit dataset
'''
def clean_balabit(df, gmin_x, gmax_x, gmin_y, gmax_y):
    
    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state"
    })

    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()
    
    x_range = gmax_x - gmin_x
    y_range = gmax_y - gmin_y

    df = df[df["state"] == "Move"].copy()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df['x'] = (df['x'] - gmin_x) / x_range
    df['y'] = (df['y'] - gmin_y) / y_range

    return df.dropna(subset=["x", "y", "time"])

'''
Generate train/test windows
'''
def generate_windows(events, chunk_size, data_root):

    if len(events) < chunk_size:
        return []

    # If the folder contains train, stride = M // 4
    if "train" in data_root.lower():
        stride = chunk_size // 4
        
    else:
        stride = chunk_size

    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i+chunk_size])

    return windows


'''
Compute Global Min/Max
'''
def compute_global_min_max(data_root, chunk_size):

    global_min_x = float("inf")
    global_max_x = float("-inf")
    global_min_y = float("inf")
    global_max_y = float("-inf")

    users = sorted(os.listdir(data_root))

    print("\n[Phase 1] Computing global min/max...")

    for user in users:

        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        for file in os.listdir(user_dir):

            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            df = pd.read_csv(path)

            # Disregard extreme outliers (logger bug, 16 bit INTMAX)
            df = df[(df["x"] < 65535) & (df["y"] < 65535)]

            sess_min_x = np.min(df['x'])
            sess_max_x = np.max(df['x'])

            sess_min_y = np.min(df['y'])
            sess_max_y = np.max(df['y'])

            if sess_min_x < global_min_x:
                global_min_x = sess_min_x
            if sess_max_x > global_max_x:
                global_max_x = sess_max_x

            if sess_min_y < global_min_y:
                global_min_y = sess_min_y
            if sess_max_y > global_max_y:
                global_max_y = sess_max_y

    
    return global_min_x, global_max_x, global_min_y, global_max_y

'''
Generate SRP
'''
def compute_srp(seq, epsilon):
    coords = seq[:, :2].astype(np.float32)

    # pairwise Euclidean distances
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    M = dist.shape[0]

    # average distance of each point to all other points
    avg_dist = (np.sum(dist, axis=1) - np.diag(dist)) / (M - 1)

    # recurrent points mask
    recurrent = avg_dist < epsilon
    #print(recurrent)
    
    # keep distances only if BOTH points are recurrent
    mask = recurrent[:, None] & recurrent[None, :]
    #print(mask)
    
    # recurrence matrix
    rp = np.where(mask, dist, epsilon)

    rp = np.minimum(rp, epsilon)
    
    return rp

'''
Save SRP (largest values to 255, smallest to 0)
'''
def draw_srp(seq, save_path, epsilon):
    rp = compute_srp(seq, epsilon)

    # normalize based on actual values
    rp_min = rp.min()
    rp_max = rp.max()
    denom = max(rp_max - rp_min, 1e-8)

    img = ((rp - rp_min) / denom * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


'''
Convert dataset into windows and SRP images
'''
def process_dataset(dataset, data_root, out_dir, sizes, epsilon):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))

    # Phase 1：global min/max
    gmin_x, gmax_x, gmin_y, gmax_y = compute_global_min_max(data_root, sizes[0])

    print("\n[Phase 2] Generating SRP...")

    for user in users:

        user_dir = os.path.join(data_root, user)

        if not os.path.isdir(user_dir):
            continue
    
        print("\n------------------------------")
        print("\nUser:", user)

        for file in os.listdir(user_dir):

            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]

            df = pd.read_csv(path)

            if dataset == "balabit":
                df = clean_balabit(df, gmin_x, gmax_x, gmin_y, gmax_y)

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:

                windows = generate_windows(events, chunk_size, data_root)

                print(f"  Session {session} | chunk={chunk_size} → {len(windows)} windows")

                for i, seq in enumerate(windows):

                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )

                    draw_srp(seq, save_path, epsilon)

'''Main'''
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300])
    parser.add_argument("--epsilon", type=float, default=0.3)

    args = parser.parse_args()

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        args.epsilon
    )

    print("\nDone.")

if __name__ == "__main__":
    main()