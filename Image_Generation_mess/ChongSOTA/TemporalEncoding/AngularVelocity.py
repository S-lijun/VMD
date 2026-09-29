# -*- coding: utf-8 -*-
"""
Chunk-wise angular-velocity temporal encoding (vertical stripes only).

Per chunk (same windowing as Velocity.py / Curvature.py):
  R = G = B = |omega| via global CDF + vertical stripe (np.tile along rows)

omega from build_global_distribution.compute_angular_velocity (theta = atan2(vy, vx), unwrap).
"""

import os
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from scipy.stats import rankdata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

GLOBAL_OMEGA_CDF = None


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


def load_raw_angular_velocity_distribution(path):
    data = np.load(path)
    omega = data["values"]

    print("\n[Angular Velocity Distribution]")
    print("Samples:", len(omega))
    print("Min:", omega.min())
    print("Max:", omega.max())

    return omega


def build_runtime_cdf(raw_omega, clip_pct):
    print("\nBuilding angular velocity runtime CDF")

    w_upper = np.percentile(raw_omega, clip_pct)
    w_clipped = raw_omega[raw_omega <= w_upper]

    ranks = rankdata(w_clipped, method="average")
    cdf = (ranks - 1) / (len(w_clipped) - 1 + 1e-8)

    order = np.argsort(w_clipped)
    w_sorted = w_clipped[order]
    cdf_sorted = cdf[order]

    print("Runtime samples:", len(w_sorted))
    print("Runtime max:", w_sorted.max())

    return w_sorted, cdf_sorted


def compute_vxvy(xs, ys, ts):
    dt = np.maximum(np.diff(ts), 1e-5)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    return vx, vy


def compute_angular_velocity(xs, ys, ts):
    """
    |omega| from unwrapped heading. Same logic as build_global_distribution.
    """
    T = len(xs)
    if T < 2:
        return np.array([])

    vx, vy = compute_vxvy(xs, ys, ts)
    eps = 1e-8
    speed_sq = vx * vx + vy * vy

    theta = np.arctan2(vy, vx)
    if speed_sq[1] >= eps:
        theta[0] = theta[1]
    else:
        theta[0] = 0.0

    theta = np.unwrap(theta)

    dt = np.maximum(np.diff(ts), eps)
    omega = np.zeros(T, dtype=np.float64)
    omega[:-1] = np.abs(np.diff(theta)) / dt
    omega[-1] = omega[-2]

    omega[speed_sq < eps] = 0.0

    return omega.astype(np.float32)


def clean_balabit(df):
    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state",
    })

    df = df[df["state"] == "Move"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])


def generate_windows(events, chunk_size, data_root):
    if len(events) < chunk_size:
        return []

    if "train" in data_root.lower():
        stride = chunk_size
    else:
        stride = chunk_size

    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i + chunk_size])
    return windows


def compute_angular_velocity_image(seq):
    """
    R = G = B = |omega| vertical stripes.
    Returns float32 H×W×3 in [0, 1], channels RGB (R, G, B).
    """
    T = len(seq)
    if T < 2:
        return None

    xs = seq[:, 0]
    ys = seq[:, 1]
    ts = seq[:, 2]

    omega = compute_angular_velocity(xs, ys, ts)
    if len(omega) == 0:
        return None

    w_norm = np.interp(
        omega,
        GLOBAL_OMEGA_CDF[0],
        GLOBAL_OMEGA_CDF[1],
        left=0,
        right=1,
    )

    stripe = np.tile(w_norm[None, :], (T, 1)).astype(np.float32)

    img = np.stack([stripe, stripe, stripe], axis=-1)
    return np.clip(img, 0, 1)


_resize_tfms = {}


def _resize_transform(side: int):
    s = int(side)
    if s not in _resize_tfms:
        _resize_tfms[s] = transforms.Resize((s, s))
    return _resize_tfms[s]


def draw_angular_velocity(seq, save_path, output_size=0):
    if len(seq) < 2:
        return

    img = compute_angular_velocity_image(seq)
    if img is None:
        return

    img_rgb = (img * 255).astype(np.uint8)

    if output_size and int(output_size) > 0:
        s = int(output_size)
        pil = Image.fromarray(img_rgb, mode="RGB")
        out_pil = _resize_transform(s)(pil)
        img_rgb = np.asarray(out_pil, dtype=np.uint8)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)


def process_dataset(dataset, data_root, out_dir, sizes, output_size=0):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("\n[Phase] Generating angular velocity vertical stripes (R=G=B)...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("User:", user)

        for file in os.listdir(user_dir):
            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]
            df = pd.read_csv(path)

            if dataset.lower() == "balabit":
                df = clean_balabit(df)
            else:
                for c in ["x", "y", "time"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["x", "y", "time"])

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:
                windows = generate_windows(events, chunk_size, data_root)
                print(f"  Session {session} | chunk={chunk_size} -> {len(windows)} windows")

                for i, seq in enumerate(windows):
                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png",
                    )
                    draw_angular_velocity(seq, save_path, output_size)


def main():
    global GLOBAL_OMEGA_CDF

    parser = argparse.ArgumentParser(
        description="Chunk angular velocity vertical stripes (R=G=B).",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--angular_velocity_dist",
        required=True,
        help="npz with values array (e.g. angular_velocity_distribution_raw.npz)",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[120])
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="若 > 0，用 transforms.Resize 将每张图存为 output_size×output_size PNG；0 表示保持 N×N。",
    )
    parser.add_argument(
        "--w_percentile",
        type=float,
        default=100,
        help="Upper clip percentile for angular velocity CDF (same as Velocity --v_percentile).",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    dist_path = resolve_path(args.angular_velocity_dist)

    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("Resolved angular_velocity_dist:", dist_path)

    raw_omega = load_raw_angular_velocity_distribution(dist_path)
    GLOBAL_OMEGA_CDF = build_runtime_cdf(raw_omega, args.w_percentile)

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        output_size=args.output_size,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
