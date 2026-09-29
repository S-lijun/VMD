# -*- coding: utf-8 -*-
"""
Plot raw feature distributions for a dataset split (training / testing).

Features match TemporalEncoding pipelines (Velocity.py / vxvy.py).
All plots use unified units: pixels / second.
ChaoShen & DFL timestamps are ms → feature values are scaled ×1000.
Percentile: velocity uses upper cap only; vx/vy use symmetric central band.

Example:
  python3 plot_feature_distribution.py \\
    --dataset balabit --split training --feature velocity --percentile 95

  python3 plot_feature_distribution.py \\
    --dataset balabit --split testing --feature vx --percentile 99
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

DEFAULT_DATA_ROOT = {
    "balabit": {
        "training": "Data/Balabit-dataset/training_files",
        "testing": "Data/Balabit-dataset/testing_files_protocol1",
    },
    "chaoshen": {
        "training": "Data/ChaoShen/training_files",
        "testing": "Data/ChaoShen/testing_files_protocol1",
    },
    "dfl": {
        "training": "Data/DFL/training_files",
        "testing": "Data/DFL/testing_files_protocol1",
    },
}

# Balabit timestamps are seconds; ChaoShen/DFL are milliseconds.
# Raw dx/dt is px/ms for the latter — multiply by 1000 for unified px/s plots.
FEATURE_SCALE_TO_PX_PER_SEC = {
    "balabit": 1.0,
    "chaoshen": 1000.0,
    "dfl": 1000.0,
}

FEATURE_ALIASES = {
    "velocity": "velocity",
    "speed": "velocity",
    "abs_velocity": "velocity",
    "absvelocity": "velocity",
    "vx": "vx",
    "vy": "vy",
}


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


def _clean_df(dataset, df):
    if dataset == "balabit":
        return clean_balabit(df)
    if dataset == "chaoshen":
        return clean_chaoshen(df)
    if dataset == "dfl":
        return clean_dfl(df)
    raise ValueError(dataset)


def compute_vx_vy(xs, ys, ts):
    """Same as TemporalEncoding/vxvy.py."""
    dt = np.maximum(np.diff(ts), 1e-5)
    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt
    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])
    return vx, vy


def compute_velocity(xs, ys, ts):
    """Same as TemporalEncoding/Velocity.py."""
    dt = np.maximum(np.diff(ts), 1e-5)
    v = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / dt
    v = np.concatenate([[v[0]], v])
    return v


def list_users(data_root):
    return sorted(
        [u for u in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, u))],
        key=natural_key,
    )


def list_session_files(user_dir):
    return sorted(
        [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))],
        key=natural_key,
    )


def collect_feature_values(dataset, data_root, feature):
    chunks = []
    n_sessions = 0
    n_events = 0

    for user in list_users(data_root):
        user_dir = os.path.join(data_root, user)
        for fname in list_session_files(user_dir):
            path = os.path.join(user_dir, fname)
            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            if len(df) < 2:
                continue

            xs = df["x"].to_numpy(dtype=np.float64)
            ys = df["y"].to_numpy(dtype=np.float64)
            ts = df["time"].to_numpy(dtype=np.float64)

            if feature == "velocity":
                vals = compute_velocity(xs, ys, ts)
            elif feature == "vx":
                vals, _ = compute_vx_vy(xs, ys, ts)
            elif feature == "vy":
                _, vals = compute_vx_vy(xs, ys, ts)
            else:
                raise ValueError(feature)

            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            scale = FEATURE_SCALE_TO_PX_PER_SEC[dataset]
            if scale != 1.0:
                vals = vals * scale

            chunks.append(vals)
            n_sessions += 1
            n_events += vals.size

    if not chunks:
        raise ValueError(f"No feature values collected from: {data_root}")

    return np.concatenate(chunks), n_sessions, n_events


def print_stats(name, values):
    print(f"\n{name} statistics")
    print("  samples :", len(values))
    print("  min     :", float(np.min(values)))
    print("  max     :", float(np.max(values)))
    print("  mean    :", float(np.mean(values)))
    print("  median  :", float(np.median(values)))
    print("  std     :", float(np.std(values)))


def apply_percentile_cap(values, percentile, feature="velocity"):
    """
    velocity: one-tailed upper cap — keep values <= p(percentile).
    vx / vy: symmetric — keep central `percentile`%% of mass; clip equal tails.
      e.g. percentile=90 → keep [p5, p95], clip 5%% on each side.
    Returns (kept, cap_low, cap_high, n_dropped). cap_low is None for velocity.
    """
    if percentile is None or percentile >= 100:
        return values, None, None, 0

    if feature in ("vx", "vy"):
        tail_pct = (100.0 - float(percentile)) / 2.0
        cap_low = float(np.percentile(values, tail_pct))
        cap_high = float(np.percentile(values, 100.0 - tail_pct))
        kept = values[(values >= cap_low) & (values <= cap_high)]
        n_dropped = int(values.size - kept.size)
        return kept, cap_low, cap_high, n_dropped

    cap_high = float(np.percentile(values, percentile))
    kept = values[values <= cap_high]
    n_dropped = int(values.size - kept.size)
    return kept, None, cap_high, n_dropped


def apply_iqr_cap(values, k=1.5, feature="velocity"):
    """
    Tukey fences per distribution: keep [Q1 - k*IQR, Q3 + k*IQR].
    Computed independently on each input array (e.g. per dataset/split).
    Returns (kept, cap_low, cap_high, n_dropped).
    """
    if k is None or values.size == 0:
        return values, None, None, 0

    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    if iqr <= 0:
        return values, q1, q3, 0

    cap_low = q1 - float(k) * iqr
    cap_high = q3 + float(k) * iqr
    if feature == "velocity":
        cap_low = max(0.0, cap_low)

    kept = values[(values >= cap_low) & (values <= cap_high)]
    n_dropped = int(values.size - kept.size)
    return kept, cap_low, cap_high, n_dropped


def plot_distribution(values, title, out_png, bins, percentile, feature, log_x, kde):
    fig, ax = plt.subplots(figsize=(11, 6))

    plot_vals, cap_low, cap_high, n_dropped = apply_percentile_cap(
        values, percentile, feature=feature
    )

    if plot_vals.size == 0:
        raise ValueError("No values left after percentile cap.")

    ax.hist(
        plot_vals,
        bins=bins,
        density=False,
        alpha=0.72,
        color="steelblue",
        edgecolor="none",
    )

    if kde and plot_vals.size > 1:
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(plot_vals.min(), plot_vals.max(), 400)
            kde_y = gaussian_kde(plot_vals)(xs)
            span = max(plot_vals.max() - plot_vals.min(), 1e-8)
            kde_y = kde_y * (len(plot_vals) * span / bins)
            ax.plot(xs, kde_y, color="darkorange", linewidth=1.8, label="KDE")
        except Exception:
            pass

    if cap_low is not None and cap_high is not None:
        tail_pct = (100.0 - float(percentile)) / 2.0
        ax.axvline(
            cap_low,
            color="crimson",
            linestyle="--",
            linewidth=1.8,
            label=f"p{tail_pct:g} = {cap_low:.4g}",
        )
        ax.axvline(
            cap_high,
            color="crimson",
            linestyle="--",
            linewidth=1.8,
            label=f"p{100.0 - tail_pct:g} = {cap_high:.4g}",
        )
        subtitle = (
            f" (plotted: p{tail_pct:g}–p{100.0 - tail_pct:g}, dropped {n_dropped:,})"
        )
        title = title + subtitle
    elif cap_high is not None:
        ax.axvline(
            cap_high,
            color="crimson",
            linestyle="--",
            linewidth=1.8,
            label=f"p{percentile:g} cap = {cap_high:.4g}",
        )
        subtitle = f" (plotted: ≤ p{percentile:g}, dropped {n_dropped:,})"
        title = title + subtitle

    if log_x:
        ax.set_xscale("symlog", linthresh=1.0)

    ax.set_title(title)
    ax.set_xlabel("Feature value (pixels / second)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    if cap_high is not None or kde:
        ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure: {out_png}")


def normalize_feature_name(name):
    key = name.strip().lower().replace("-", "_")
    if key not in FEATURE_ALIASES:
        raise argparse.ArgumentTypeError(
            f"Unknown feature '{name}'. Choose from: velocity, vx, vy"
        )
    return FEATURE_ALIASES[key]


def default_out_png(dataset, split, feature, percentile):
    suffix = "" if percentile is None or percentile >= 100 else f"_p{int(percentile)}"
    return os.path.join(
        ROOT,
        "Image_Generation",
        "ChongSOTA",
        "TemporalEncoding",
        "feature_distributions",
        f"{dataset}_{split}_{feature}{suffix}.png",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot TemporalEncoding feature distributions for train/test splits.",
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument(
        "--split",
        required=True,
        choices=["training", "testing"],
        help="Which default data split to scan.",
    )
    parser.add_argument(
        "--feature",
        required=True,
        type=normalize_feature_name,
        help="velocity (|v|), vx, or vy",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override dataset root (default from --dataset + --split).",
    )
    parser.add_argument(
        "--out_png",
        default=None,
        help="Output figure path (default: TemporalEncoding/feature_distributions/...).",
    )
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="velocity: keep <= pN. vx/vy: keep central N%% (symmetric tails). 100 = no cap.",
    )
    parser.add_argument("--log_x", action="store_true", help="Use symlog x-axis.")
    parser.add_argument("--kde", action="store_true", help="Overlay KDE curve.")
    args = parser.parse_args()

    if not (0 < args.percentile <= 100):
        parser.error("--percentile must be in (0, 100].")

    if args.data_root:
        data_root = resolve_path(args.data_root)
    else:
        rel = DEFAULT_DATA_ROOT[args.dataset][args.split]
        data_root = resolve_path(rel)

    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    out_png = resolve_path(args.out_png) if args.out_png else default_out_png(
        args.dataset, args.split, args.feature, args.percentile
    )

    print("[ROOT]", ROOT)
    print("[dataset]", args.dataset)
    print("[split]", args.split)
    print("[feature]", args.feature)
    print("[percentile]", args.percentile)
    print("[units]", "pixels/second (ChaoShen/DFL ×1000 from px/ms)")
    print("[data_root]", data_root)

    values, n_sessions, n_events = collect_feature_values(
        args.dataset, data_root, args.feature
    )

    label = {
        "velocity": "Absolute velocity |v|",
        "vx": "vx = dx/dt",
        "vy": "vy = dy/dt",
    }[args.feature]

    print_stats(f"{label} (all data)", values)
    print("  sessions:", n_sessions)
    print("  events  :", n_events)

    plot_vals, cap_low, cap_high, n_dropped = apply_percentile_cap(
        values, args.percentile, feature=args.feature
    )
    if cap_high is not None:
        if cap_low is not None:
            tail_pct = (100.0 - args.percentile) / 2.0
            print(f"\n[Symmetric percentile] central p{args.percentile:g} = [p{tail_pct:g}, p{100.0 - tail_pct:g}]")
            print(f"  low cap  : {cap_low:.6g}")
            print(f"  high cap : {cap_high:.6g}")
        else:
            print(f"\n[Percentile cap] p{args.percentile:g} = {cap_high:.6g}")
        print(f"  kept   : {plot_vals.size:,} / {values.size:,}")
        print(f"  dropped: {n_dropped:,} ({100.0 * n_dropped / values.size:.2f}%)")
        cap_label = (
            f"p{(100.0 - args.percentile) / 2.0:g}–p{100.0 - (100.0 - args.percentile) / 2.0:g}"
            if cap_low is not None
            else f"≤ p{args.percentile:g}"
        )
        print_stats(f"{label} ({cap_label})", plot_vals)

    title = f"{label} — {args.dataset} / {args.split}"
    plot_distribution(
        values,
        title=title,
        out_png=out_png,
        bins=args.bins,
        percentile=args.percentile,
        feature=args.feature,
        log_x=args.log_x,
        kde=args.kde,
    )


if __name__ == "__main__":
    main()
