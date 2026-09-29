# -*- coding: utf-8 -*-
"""
Chunk-wise SRP with node time-difference temporal encoding (vertical stripes).

Per chunk (same windowing as SRP_chunk_velocity.py):
  R = pair-wise distance on locally normalized x,y (compute_srp_pair), min-max -> [0, 1]
  G = B = node Δt via global CDF + vertical stripe (np.tile along rows)

Δt pipeline follows RecurrencePlot/SRP_time_node.py.
Global distribution: build_global_distribution.py --feature timediff_node (npz key: values).
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from scipy.stats import rankdata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

GLOBAL_TD_CDF = None


def natural_key(string):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", string)]


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


def load_raw_time_distribution(path):
    data = np.load(path)
    td = data["values"]

    print("\n[Time Difference Distribution]")
    print("Samples:", len(td))
    print("Min:", td.min())
    print("Max:", td.max())

    return td


def build_runtime_cdf(raw_td, clip_pct):
    print("\nBuilding runtime Δt CDF")

    td_upper = np.percentile(raw_td, clip_pct)
    td_clipped = raw_td[raw_td <= td_upper]

    ranks = rankdata(td_clipped, method="average")
    cdf = (ranks - 1) / (len(td_clipped) - 1 + 1e-8)

    order = np.argsort(td_clipped)
    td_sorted = td_clipped[order]
    cdf_sorted = cdf[order]

    print("Runtime samples:", len(td_sorted))
    print("Runtime max:", td_sorted.max())

    return td_sorted, cdf_sorted


def compute_timediff_node(ts):
    """
    Node Δt: dt[i] = t[i+1]-t[i], last node copies the previous interval.
    Same as RecurrencePlot/SRP_time_node.py (length T).
    """
    T = len(ts)
    dt_node = np.zeros(T, dtype=np.float32)
    if T < 2:
        return dt_node

    dt_node[:-1] = ts[1:] - ts[:-1]
    dt_node[-1] = dt_node[-2]
    return dt_node


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


def clean_chaoshen(df):
    df = df.rename(columns={
        "X": "x",
        "Y": "y",
        "Timestamp": "time",
        "EventName": "event",
    })

    df = df[df["event"] == "Move"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

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

    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])


def clean_twos(df):
    df = df.rename(columns={
        "timestamp": "time",
        "x": "x",
        "y": "y",
        "event": "event",
    })

    df = df[df["event"] == "Mouse Moved"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])


def _clean_df(dataset, df):
    if dataset == "balabit":
        return clean_balabit(df)
    if dataset == "chaoshen":
        return clean_chaoshen(df)
    if dataset == "dfl":
        return clean_dfl(df)
    if dataset == "twos":
        return clean_twos(df)
    raise ValueError(dataset)


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


def compute_srp_pair(seq, epsilon):
    """Pair-wise SRP with per-sequence local normalization (SRP_chunk)."""
    coords = seq[:, :2].astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    scale = max(x_max - x_min, y_max - y_min)
    if scale < 1e-8:
        scale = 1e-8

    x_norm = (x - x_min) / scale
    y_norm = (y - y_min) / scale

    coords_norm = np.stack([x_norm, y_norm], axis=1)

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    rp = np.minimum(dist, epsilon)
    return rp


def compute_srp_chunk_time_node(seq, epsilon):
    """
    R = normalized pair-wise distance matrix
    G = B = node Δt vertical stripes
    Returns float32 H×W×3 in [0, 1], channels RGB (R, G, B).
    """
    T = len(seq)
    if T < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)
    rp_min = rp.min()
    rp_max = rp.max()
    denom = max(rp_max - rp_min, 1e-8)
    r_channel = ((rp - rp_min) / denom).astype(np.float32)

    ts = seq[:, 2]
    dt_node = compute_timediff_node(ts)

    dt_norm = np.interp(
        dt_node,
        GLOBAL_TD_CDF[0],
        GLOBAL_TD_CDF[1],
        left=0,
        right=1,
    )

    stripe = np.tile(dt_norm[None, :], (T, 1)).astype(np.float32)

    g_channel = stripe
    b_channel = stripe

    img = np.stack([r_channel, g_channel, b_channel], axis=-1)
    return np.clip(img, 0, 1)


_resize_tfms = {}


def _resize_transform(side: int):
    s = int(side)
    if s not in _resize_tfms:
        _resize_tfms[s] = transforms.Resize((s, s))
    return _resize_tfms[s]


def render_srp_chunk_time_node(seq, epsilon, output_size=0):
    """Return uint8 RGB (H, W, 3), or None if seq too short."""
    if len(seq) < 2:
        return None

    img = compute_srp_chunk_time_node(seq, epsilon)
    if img is None:
        return None

    img_rgb = (img * 255).astype(np.uint8)

    if output_size and int(output_size) > 0:
        s = int(output_size)
        pil = Image.fromarray(img_rgb)
        out_pil = _resize_transform(s)(pil)
        img_rgb = np.asarray(out_pil, dtype=np.uint8)

    return img_rgb


def rgb_to_tensor_chw(img_rgb):
    """RGB H×W×3 uint8 -> (3, H, W) uint8 (Images_convert format)."""
    return np.transpose(img_rgb, (2, 0, 1))


def list_users(data_root):
    return sorted(
        [u for u in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, u))],
        key=natural_key,
    )


def list_session_csvs(user_dir):
    return sorted(
        [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))],
        key=natural_key,
    )


def load_events(dataset, path):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)
    return df[["x", "y", "time"]].values.astype(np.float32)


def count_windows(dataset, data_root, chunk_size):
    total = 0
    users = list_users(data_root)

    for user in users:
        user_dir = os.path.join(data_root, user)
        for file in list_session_csvs(user_dir):
            events = load_events(dataset, os.path.join(user_dir, file))
            total += len(generate_windows(events, chunk_size, data_root))

    return total, users


def draw_srp_chunk_time_node(seq, save_path, epsilon, output_size=0):
    img_rgb = render_srp_chunk_time_node(seq, epsilon, output_size)
    if img_rgb is None:
        return

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)


def process_dataset_tensors(dataset, data_root, out_dir, sizes, epsilon, output_size=0):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("\n[Phase] Generating pair-wise SRP + node Δt stripe tensors (Images_convert format)...")

    for chunk_size in sizes:
        H = int(output_size) if output_size and int(output_size) > 0 else chunk_size
        W = H

        total_samples, _ = count_windows(dataset, data_root, chunk_size)
        tensor_root = os.path.join(out_dir, f"event{chunk_size}")
        os.makedirs(tensor_root, exist_ok=True)

        print(f"\n[event{chunk_size}] Total samples: {total_samples} | Tensor size: {H}x{W}")

        images = np.memmap(
            os.path.join(tensor_root, "images.npy"),
            dtype=np.uint8,
            mode="w+",
            shape=(total_samples, 3, H, W),
        )
        labels = np.memmap(
            os.path.join(tensor_root, "labels.npy"),
            dtype=np.uint8,
            mode="w+",
            shape=(total_samples, num_users),
        )

        sessions = []
        idx = 0

        for user in users:
            user_dir = os.path.join(data_root, user)
            print("\n------------------------------")
            print("User:", user)

            for file in list_session_csvs(user_dir):
                path = os.path.join(user_dir, file)
                session = os.path.splitext(file)[0]
                events = load_events(dataset, path)
                windows = generate_windows(events, chunk_size, data_root)
                print(f"  Session {session} | chunk={chunk_size} -> {len(windows)} windows")

                for seq in windows:
                    img_rgb = render_srp_chunk_time_node(seq, epsilon, output_size)
                    if img_rgb is None:
                        continue

                    if img_rgb.shape[:2] != (H, W):
                        img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

                    images[idx] = rgb_to_tensor_chw(img_rgb)

                    y = np.zeros(num_users, dtype=np.uint8)
                    y[user_to_idx[user]] = 1
                    labels[idx] = y
                    sessions.append(session)
                    idx += 1

        images.flush()
        labels.flush()

        np.save(
            os.path.join(tensor_root, "sessions.npy"),
            np.array(sessions, dtype=object),
        )

        print(f"\nTensor dataset saved to: {tensor_root}")


def process_dataset(dataset, data_root, out_dir, sizes, epsilon, output_size=0, tensors=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir, sizes, epsilon, output_size)
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("\n[Phase] Generating pair-wise SRP + node Δt stripes...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("User:", user)

        for file in list_session_csvs(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            events = load_events(dataset, path)

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
                    draw_srp_chunk_time_node(seq, save_path, epsilon, output_size)


def main():
    global GLOBAL_TD_CDF

    parser = argparse.ArgumentParser(
        description="Chunk SRP (pair-wise R) + node time-difference vertical stripes (G=B).",
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl", "twos"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--td_dist",
        required=True,
        help="npz from build_global_distribution --feature timediff_node (key: values).",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[125])
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="若 > 0，用 transforms.Resize 将每张图存为 output_size×output_size PNG；0 表示保持 N×N。",
    )
    parser.add_argument(
        "--t_percentile",
        type=float,
        default=95,
        help="Upper clip percentile for Δt CDF (same as SRP_time_node).",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="直接输出 Images_convert 格式的 tensors（images.npy / labels.npy / sessions.npy），不写 PNG。",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    dist_path = resolve_path(args.td_dist)

    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("Resolved td_dist:", dist_path)

    raw_td = load_raw_time_distribution(dist_path)
    GLOBAL_TD_CDF = build_runtime_cdf(raw_td, args.t_percentile)

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        epsilon=args.epsilon,
        output_size=args.output_size,
        tensors=args.tensors,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
