# -*- coding: utf-8 -*-
"""
Sequence length stats + distribution plot matching XYPlot_per_user / paper:

  1) split_by_time (dt > 1s)
  2) merge short segments until arc length >= per-user screen width (max x)

Example:
  python sequence_length_stats.py \\
    --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files

  python sequence_length_stats.py \\
    --dataset balabit \\
    --data_root Data/Balabit-dataset/testing_files \\
    --out_png sequence_length_balabit_test.png
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from XYPlot import (
    ROOT,
    TIME_THRESHOLD,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
)

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
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


def _clean_df(dataset, df):
    if dataset == "balabit":
        return clean_balabit(df)
    if dataset == "chaoshen":
        return clean_chaoshen(df)
    if dataset == "dfl":
        return clean_dfl(df)
    raise ValueError(dataset)


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


def build_user_max_xy_from_training(dataset, training_root):
    """Same as XYPlot_per_user: per-user max x / max y from training sessions."""
    user_max_xy = {}

    for user in list_users(training_root):
        user_dir = os.path.join(training_root, user)
        max_x = 0.0
        max_y = 0.0
        saw_points = False

        for name in list_session_files(user_dir):
            path = os.path.join(user_dir, name)
            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            if len(df) == 0:
                continue
            max_x = max(max_x, float(df["x"].max()))
            max_y = max(max_y, float(df["y"].max()))
            saw_points = True

        if saw_points:
            user_max_xy[user] = (max_x, max_y)
        else:
            user_max_xy[user] = (GLOBAL_MAX_X, GLOBAL_MAX_Y)

    return user_max_xy


def _norm_x(user, user_max_xy):
    if user in user_max_xy:
        return float(user_max_xy[user][0])
    print(
        "\n[WARN] User", user, "not in training scan; using GLOBAL_MAX_X:",
        GLOBAL_MAX_X,
    )
    return float(GLOBAL_MAX_X)


def length_summary(lengths):
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.size == 0:
        return None
    q1, median, q3 = np.percentile(lengths, [25, 50, 75])
    return {
        "n": int(lengths.size),
        "min": float(np.min(lengths)),
        "q1": float(q1),
        "median": float(median),
        "average": float(np.mean(lengths)),
        "q3": float(q3),
        "max": float(np.max(lengths)),
    }


def print_length_stats(title, lengths):
    stats = length_summary(lengths)
    if stats is None:
        print(f"\n[{title}] No sequences.")
        return None

    print(f"\n========== {title} ==========")
    print(f"  n       : {stats['n']}")
    print(f"  min     : {int(stats['min'])}")
    print(f"  Q1      : {stats['q1']:.2f}")
    print(f"  median  : {stats['median']:.2f}")
    print(f"  average : {stats['average']:.2f}")
    print(f"  Q3      : {stats['q3']:.2f}")
    print(f"  max     : {int(stats['max'])}")
    return stats


def default_out_png(dataset, data_root):
    base = os.path.basename(os.path.normpath(data_root))
    parent = os.path.basename(os.path.dirname(os.path.normpath(data_root)))
    name = f"sequence_length_{dataset}_{parent}_{base}.png"
    name = re.sub(r"[^\w.\-]+", "_", name)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def plot_sequence_length_distribution(
    lengths,
    out_png,
    title,
    bins=80,
    clip_percentile=99.0,
    show_kde=False,
):
    """
    Histogram of sequence lengths (event counts) after split+merge.
    clip_percentile only affects the plot x-range / hist display, not printed stats.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    if lengths.size == 0:
        raise ValueError("No sequences to plot.")

    stats = length_summary(lengths)
    if clip_percentile is not None and clip_percentile < 100:
        x_max = float(np.percentile(lengths, clip_percentile))
        plot_vals = lengths[lengths <= x_max]
        n_dropped = int(lengths.size - plot_vals.size)
    else:
        x_max = float(stats["max"])
        plot_vals = lengths
        n_dropped = 0

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(
        plot_vals,
        bins=bins,
        range=(max(stats["min"], 0.0), x_max),
        density=False,
        alpha=0.75,
        color="steelblue",
        edgecolor="white",
        linewidth=0.4,
    )

    if show_kde and plot_vals.size > 1:
        try:
            from scipy.stats import gaussian_kde

            xs = np.linspace(plot_vals.min(), plot_vals.max(), 400)
            kde_y = gaussian_kde(plot_vals)(xs)
            span = max(plot_vals.max() - plot_vals.min(), 1e-8)
            kde_y = kde_y * (len(plot_vals) * span / bins)
            ax.plot(xs, kde_y, color="darkorange", linewidth=1.8, label="KDE")
        except Exception:
            pass

    line_specs = [
        ("Q1", stats["q1"], "#2ca02c"),
        ("median", stats["median"], "#d62728"),
        ("mean", stats["average"], "#ff7f0e"),
        ("Q3", stats["q3"], "#9467bd"),
    ]
    for label, val, color in line_specs:
        if val <= x_max:
            ax.axvline(
                val,
                color=color,
                linestyle="--",
                linewidth=1.6,
                label=f"{label}={val:.1f}",
            )

    stats_text = (
        f"n = {stats['n']:,}\n"
        f"min = {int(stats['min'])}\n"
        f"Q1 = {stats['q1']:.2f}\n"
        f"median = {stats['median']:.2f}\n"
        f"mean = {stats['average']:.2f}\n"
        f"Q3 = {stats['q3']:.2f}\n"
        f"max = {int(stats['max'])}"
    )
    if n_dropped > 0:
        stats_text += f"\nplot ≤ p{clip_percentile:g} (drop {n_dropped:,})"

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="#cccccc"),
    )

    ax.set_title(title)
    ax.set_xlabel("Sequence length (number of events)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"\n[plot] saved: {out_png}")


def collect_sequence_lengths(
    dataset,
    data_root,
    user_max_xy,
    min_points=2,
    verbose=False,
    also_split_only=False,
):
    """
    Default pipeline = split_by_time then merge_sequences(min_length=per-user max x).
    Length = number of events per sequence.
    """
    all_merged = []
    all_split = []
    per_user = {}
    n_sessions = 0
    n_events = 0
    n_after_split = 0
    n_after_merge = 0

    users = list_users(data_root)
    for user in users:
        user_dir = os.path.join(data_root, user)
        min_length = _norm_x(user, user_max_xy)
        user_lengths = []

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            events = df.to_dict("records")
            n_sessions += 1
            n_events += len(events)

            if len(events) < 2:
                if verbose:
                    print(f"  {user}/{file}: events={len(events)} -> skip")
                continue

            sequences = split_by_time(events)
            n_split = len(sequences)
            n_after_split += n_split

            if also_split_only:
                all_split.extend(
                    [len(seq) for seq in sequences if len(seq) >= min_points]
                )

            sequences = merge_sequences(sequences, min_length)
            lengths = [len(seq) for seq in sequences if len(seq) >= min_points]
            n_after_merge += len(lengths)
            user_lengths.extend(lengths)

            if verbose:
                print(
                    f"  {user}/{file}: events={len(events)} "
                    f"| split={n_split} | merge={len(lengths)} "
                    f"| min_length(max_x)={min_length:.1f}"
                )

        per_user[user] = user_lengths
        all_merged.extend(user_lengths)

    meta = {
        "users": len(users),
        "sessions": n_sessions,
        "events": n_events,
        "n_after_split": n_after_split,
        "n_after_merge": n_after_merge,
    }
    return all_merged, all_split, per_user, meta


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sequence length stats: split_by_time (dt > 1s) then merge until "
            "arc length >= per-user screen width (max x), same as XYPlot_per_user."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["balabit", "chaoshen", "dfl"],
        help="Cleaning pipeline for the raw CSVs.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help=(
            "Sessions to score, e.g. Data/Balabit-dataset/training_files "
            "or Data/Balabit-dataset/testing_files."
        ),
    )
    parser.add_argument(
        "--training_root",
        default=None,
        help=(
            "Used to estimate per-user screen width (max x) for merge. "
            "Default: that dataset's training_files."
        ),
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=2,
        help="Ignore sequences shorter than this (default: 2, same as draw pipeline).",
    )
    parser.add_argument(
        "--also_split_only",
        action="store_true",
        default=False,
        help="Also print stats before merge (for comparison).",
    )
    parser.add_argument(
        "--per_user",
        action="store_true",
        default=False,
        help="Also print per-user stats (after merge).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-session split/merge counts.",
    )
    parser.add_argument(
        "--out_png",
        default=None,
        help="Output histogram path (default: sequence_length_<dataset>_<split>.png next to this script).",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        default=False,
        help="Skip distribution plot.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bins (default: 80).",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Plot x-range upper percentile only (default: 99; does not change printed stats).",
    )
    parser.add_argument(
        "--kde",
        action="store_true",
        default=False,
        help="Overlay KDE on the histogram.",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = resolve_path(training_rel)
    if not os.path.isdir(training_root):
        raise FileNotFoundError(f"training_root not found: {training_root}")

    print("[ROOT]", ROOT)
    print("[dataset]", args.dataset)
    print("[data_root]", data_root)
    print("[training_root]", training_root)
    print("[time_threshold_s]", TIME_THRESHOLD)
    print("[min_points]", args.min_points)
    print("[merge]", "arc length >= per-user max x (screen width)")

    user_max_xy = build_user_max_xy_from_training(args.dataset, training_root)
    print("\nUSER_MAX_XY (merge min_length = max_x):")
    for u in sorted(user_max_xy.keys(), key=natural_key):
        print(f"  {u}: max_x={user_max_xy[u][0]:.1f}, max_y={user_max_xy[u][1]:.1f}")

    all_merged, all_split, per_user, meta = collect_sequence_lengths(
        args.dataset,
        data_root,
        user_max_xy,
        min_points=args.min_points,
        verbose=args.verbose,
        also_split_only=args.also_split_only,
    )

    print(f"\nUsers    : {meta['users']}")
    print(f"Sessions : {meta['sessions']}")
    print(f"Events   : {meta['events']} (after cleaning)")
    print(f"Sequences after split : {meta['n_after_split']}")
    print(f"Sequences after merge : {meta['n_after_merge']}")

    if args.also_split_only:
        print_length_stats(
            f"BEFORE merge | split_by_time dt>{TIME_THRESHOLD}s only",
            all_split,
        )

    print_length_stats(
        f"AFTER merge | dt>{TIME_THRESHOLD}s + fuse until arc length >= screen width",
        all_merged,
    )

    if args.per_user:
        print("\n========== Per-user summary (after merge) ==========")
        print(f"{'user':<12} {'n':>8} {'min':>8} {'Q1':>10} {'med':>10} "
              f"{'avg':>10} {'Q3':>10} {'max':>8} {'max_x':>10}")
        for user in sorted(per_user.keys(), key=natural_key):
            lengths = np.asarray(per_user[user], dtype=np.float64)
            mx = _norm_x(user, user_max_xy)
            if lengths.size == 0:
                print(f"{user:<12} {'0':>8} {'':>8} {'':>10} {'':>10} "
                      f"{'':>10} {'':>10} {'':>8} {mx:>10.1f}")
                continue
            q1, med, q3 = np.percentile(lengths, [25, 50, 75])
            print(
                f"{user:<12} {len(lengths):>8} {int(np.min(lengths)):>8} "
                f"{q1:>10.2f} {med:>10.2f} {float(np.mean(lengths)):>10.2f} "
                f"{q3:>10.2f} {int(np.max(lengths)):>8} {mx:>10.1f}"
            )

    if not args.no_plot:
        out_png = resolve_path(args.out_png) if args.out_png else default_out_png(
            args.dataset, data_root
        )
        split_name = os.path.basename(os.path.normpath(data_root))
        title = (
            f"{args.dataset} | {split_name}\n"
            f"Sequence length after dt>{TIME_THRESHOLD:g}s split + "
            f"merge (arc length ≥ screen width)"
        )
        plot_sequence_length_distribution(
            all_merged,
            out_png=out_png,
            title=title,
            bins=args.bins,
            clip_percentile=args.clip_percentile,
            show_kde=args.kde,
        )


if __name__ == "__main__":
    main()
