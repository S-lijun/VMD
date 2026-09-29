import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))

def clean_balabit(df):
    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state",
    })
    df = df[(df["x"] < 65536) & (df["y"] < 65536)]
    df = df[df["state"] == "Move"].copy()
    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y", "time"])


def compute_srp_local(seq, epsilon):
    coords = seq[:, :2].astype(np.float32)
    x = coords[:, 0]
    y = coords[:, 1]

    x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-8)
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-8)
    coords_norm = np.stack([x_norm, y_norm], axis=1)

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    m = dist.shape[0]
    avg_dist = (np.sum(dist, axis=1) - np.diag(dist)) / max(m - 1, 1)
    recurrent = avg_dist < epsilon
    mask = recurrent[:, None] & recurrent[None, :]
    rp = np.where(mask, dist, epsilon)
    return rp, recurrent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root folder containing user/session CSVs")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--dataset", default="balabit")
    parser.add_argument("--out_png", required=True, help="Output path for distribution figure")
    parser.add_argument("--out_bar_png", required=True, help="Output path for recurrence True/False bar chart")
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_png = resolve_path(args.out_png)
    out_bar_png = resolve_path(args.out_bar_png)

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    all_values = []
    recurrent_true_total = 0
    recurrent_false_total = 0
    users = sorted(os.listdir(data_root))
    total_sessions = 0
    total_windows = 0

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        for file in os.listdir(user_dir):
            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue
            total_sessions += 1

            df = pd.read_csv(path)
            if args.dataset.lower() == "balabit":
                df = clean_balabit(df)
            else:
                for c in ["x", "y", "time"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["x", "y", "time"])

            events = df[["x", "y", "time"]].values.astype(np.float32)
            if len(events) < args.chunk_size:
                continue

            for i in range(0, len(events) - args.chunk_size + 1, args.chunk_size):
                seq = events[i:i + args.chunk_size]
                rp, recurrent = compute_srp_local(seq, args.epsilon)
                tri_i, tri_j = np.triu_indices(rp.shape[0], k=1)
                all_values.append(rp[tri_i, tri_j])
                recurrent_true_total += int(np.sum(recurrent))
                recurrent_false_total += int(recurrent.size - np.sum(recurrent))
                total_windows += 1

    if not all_values:
        raise ValueError("No valid windows found. Check data_root/chunk_size.")

    rp_values = np.concatenate(all_values, axis=0)

    below = int(np.sum(rp_values < args.epsilon))
    above_or_equal = int(np.sum(rp_values >= args.epsilon))
    total = int(rp_values.size)

    print(f"users scanned: {len([u for u in users if os.path.isdir(os.path.join(data_root, u))])}")
    print(f"sessions scanned: {total_sessions}")
    print(f"windows used: {total_windows}")
    print(f"threshold (epsilon): {args.epsilon}")
    print(f"total pair values: {total}")
    print(f"count < threshold: {below}")
    print(f"count >= threshold: {above_or_equal}")
    print(f"ratio < threshold: {below / max(total, 1):.6f}")
    print(f"ratio >= threshold: {above_or_equal / max(total, 1):.6f}")
    rec_total = recurrent_true_total + recurrent_false_total
    print(f"recurrent=True count: {recurrent_true_total}")
    print(f"recurrent=False count: {recurrent_false_total}")
    print(f"recurrent=True ratio: {recurrent_true_total / max(rec_total, 1):.6f}")
    print(f"recurrent=False ratio: {recurrent_false_total / max(rec_total, 1):.6f}")

    plt.figure(figsize=(10, 5))
    plt.hist(rp_values, bins=80, color="steelblue", alpha=0.85, edgecolor="none")
    plt.axvline(args.epsilon, color="crimson", linestyle="--", linewidth=2, label=f"threshold={args.epsilon}")
    plt.title("RP value distribution (upper triangle, diagonal excluded)")
    plt.xlabel("RP value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"saved histogram: {out_png}")

    plt.figure(figsize=(6, 5))
    labels = ["True (recurrence)", "False (not recurrence)"]
    counts = [recurrent_true_total, recurrent_false_total]
    colors = ["#2e7d32", "#c62828"]
    bars = plt.bar(labels, counts, color=colors)
    plt.title("Recurrence point count")
    plt.ylabel("Count")
    plt.tight_layout()
    for bar, value in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    os.makedirs(os.path.dirname(out_bar_png), exist_ok=True)
    plt.savefig(out_bar_png, dpi=150)
    print(f"saved recurrence bar chart: {out_bar_png}")


if __name__ == "__main__":
    main()
