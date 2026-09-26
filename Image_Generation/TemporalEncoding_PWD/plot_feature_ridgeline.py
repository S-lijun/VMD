# -*- coding: utf-8 -*-
"""
Ridgeline grid: 3×3 subplots (datasets × features).
Row order: Balabit → ChaoShen → DFL. Each cell has train/test ridges + stats.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from plot_feature_distribution import (
    ROOT,
    DEFAULT_DATA_ROOT,
    collect_feature_values,
    apply_iqr_cap,
    resolve_path,
)

DATASETS = ["balabit", "chaoshen", "dfl"]
FEATURES = ["velocity", "vx", "vy"]
SPLITS = ["training", "testing"]

DATASET_LABELS = {
    "balabit": "Balabit",
    "chaoshen": "ChaoShen",
    "dfl": "DFL",
}

FEATURE_LABELS = {
    "velocity": "|v| (speed)",
    "vx": "vx",
    "vy": "vy",
}

SPLIT_COLORS = {
    "training": "#2c6faa",
    "testing": "#d95f02",
}


def subsample(values, max_n, seed):
    if values.size <= max_n:
        return values
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=max_n, replace=False)


def kde_shape(values, x_grid, max_kde_samples=None, seed=0):
    """Peak-normalized KDE shape (amplitude applied separately from sample count)."""
    from scipy.stats import gaussian_kde

    kde_vals = values
    if max_kde_samples is not None and values.size > max_kde_samples:
        kde_vals = subsample(values, max_kde_samples, seed)

    if kde_vals.size < 2:
        return np.zeros_like(x_grid)

    kde = gaussian_kde(kde_vals)
    y = kde(x_grid)
    peak = float(np.max(y))
    if peak <= 0:
        return np.zeros_like(x_grid)
    return y / peak


def collect_all(iqr_k, max_samples):
    """Return (clip_data, raw_data): clipped arrays for KDE, raw for statistics."""
    clip_data = {f: {d: {} for d in DATASETS} for f in FEATURES}
    raw_data = {f: {d: {} for d in DATASETS} for f in FEATURES}

    for feature in FEATURES:
        for dataset in DATASETS:
            for split in SPLITS:
                root = resolve_path(DEFAULT_DATA_ROOT[dataset][split])
                print(f"[collect] {dataset} / {split} / {feature} <- {root}")
                values, n_sess, n_evt = collect_feature_values(dataset, root, feature)
                raw_data[feature][dataset][split] = values
                plot_vals, cap_lo, cap_hi, n_drop = apply_iqr_cap(
                    values, k=iqr_k, feature=feature
                )
                clip_data[feature][dataset][split] = plot_vals
                cap_desc = (
                    f"[{cap_lo:.4g}, {cap_hi:.4g}]"
                    if cap_lo is not None and cap_hi is not None
                    else "no clip"
                )
                kde_note = (
                    f" (KDE subsample≤{max_samples:,})"
                    if plot_vals.size > max_samples
                    else ""
                )
                clip_label = (
                    f"IQR k={iqr_k:g}: {cap_desc}"
                    if iqr_k is not None
                    else "no clip"
                )
                print(
                    f"          sessions={n_sess:,} events={n_evt:,} "
                    f"raw_n={values.size:,} plot_n={plot_vals.size:,}{kde_note} "
                    f"({clip_label}, dropped={n_drop:,})"
                )

    return clip_data, raw_data


def global_xlim(feature, data, log_x):
    pooled = []
    for dataset in DATASETS:
        for split in SPLITS:
            pooled.append(data[feature][dataset][split])
    pooled = np.concatenate(pooled)
    if pooled.size == 0:
        return 0.0, 1.0

    lo = float(np.min(pooled))
    hi = float(np.max(pooled))
    if feature == "velocity" and log_x:
        lo = max(lo, 1e-3)
    pad = 0.03 * (hi - lo if hi > lo else abs(hi) + 1.0)
    return lo - pad, hi + pad


def format_stats(values):
    if values.size == 0:
        return "min: —\nmax: —\nmean: —\nmedian: —"
    return (
        f"min: {float(np.min(values)):.4g}\n"
        f"max: {float(np.max(values)):.4g}\n"
        f"mean: {float(np.mean(values)):.4g}\n"
        f"median: {float(np.median(values)):.4g}"
    )


def format_stats_labeled(split, values):
    tag = "train" if split == "training" else "test"
    n = f"{values.size:,}" if values.size else "—"
    return f"[{tag}]  n={n}\n{format_stats(values)}"


STAT_FONTSIZE = 6.5
STAT_PAD = 0.18


def draw_cell_ridgeline(
    ax, dataset, feature, clip_data, raw_data, x_grid, ridge_scale, max_kde_samples
):
    """KDE on IQR-clipped data; statistics on full (raw) distribution."""
    y_base = 0.0

    train_clip = clip_data[feature][dataset]["training"]
    test_clip = clip_data[feature][dataset]["testing"]
    train_raw = raw_data[feature][dataset]["training"]
    test_raw = raw_data[feature][dataset]["testing"]
    n_ref = max(train_clip.size, test_clip.size, 1)

    ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
    ax.set_ylim(-0.12, ridge_scale + 0.10)

    stat_kw = dict(
        fontsize=STAT_FONTSIZE,
        family="monospace",
        zorder=10,
        transform=ax.transAxes,
        va="top",
        linespacing=1.1,
    )
    bbox_kw = dict(boxstyle=f"round,pad={STAT_PAD}", facecolor="white", alpha=0.92)

    # Test behind, train in front — KDE on clipped values.
    for zi, split in enumerate(("testing", "training")):
        vals = clip_data[feature][dataset][split]
        seed = hash((dataset, split, feature)) % (2**32)
        shape = kde_shape(vals, x_grid, max_kde_samples=max_kde_samples, seed=seed)
        amp = vals.size / n_ref
        ridge = ridge_scale * shape * amp
        color = SPLIT_COLORS[split]
        fill_alpha = 0.42

        ax.fill_between(
            x_grid,
            y_base,
            y_base + ridge,
            color=color,
            alpha=fill_alpha,
            linewidth=0,
            zorder=zi + 1,
        )
        ax.plot(
            x_grid,
            y_base + ridge,
            color=color,
            alpha=0.95,
            linewidth=1.4,
            zorder=zi + 3,
        )

    ax.text(
        0.98,
        0.98,
        format_stats_labeled("training", train_raw),
        ha="right",
        color=SPLIT_COLORS["training"],
        bbox=dict(edgecolor=SPLIT_COLORS["training"], linewidth=0.8, **bbox_kw),
        **stat_kw,
    )
    ax.text(
        0.98,
        0.54,
        format_stats_labeled("testing", test_raw),
        ha="right",
        color=SPLIT_COLORS["testing"],
        bbox=dict(edgecolor=SPLIT_COLORS["testing"], linewidth=0.8, **bbox_kw),
        **stat_kw,
    )

    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)


def plot_ridgeline_figure(
    clip_data, raw_data, iqr_k, log_x, ridge_scale, x_grid_n, out_png, max_kde_samples
):
    n_rows = len(DATASETS)
    n_cols = len(FEATURES)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 10),
        sharex="col",
    )

    x_grids = {}
    for feature in FEATURES:
        x_min, x_max = global_xlim(feature, clip_data, log_x)
        x_grids[feature] = np.linspace(x_min, x_max, x_grid_n)

    for ri, dataset in enumerate(DATASETS):
        for ci, feature in enumerate(FEATURES):
            ax = axes[ri, ci]
            draw_cell_ridgeline(
                ax,
                dataset,
                feature,
                clip_data,
                raw_data,
                x_grids[feature],
                ridge_scale,
                max_kde_samples,
            )

            if log_x and feature == "velocity":
                ax.set_xscale("symlog", linthresh=1.0)

            if ri == 0:
                ax.set_title(FEATURE_LABELS[feature], fontsize=11, pad=3)
            if ci == 0:
                ax.set_ylabel(
                    DATASET_LABELS[dataset],
                    fontsize=11,
                    fontweight="bold",
                    labelpad=4,
                )
            if ri == n_rows - 1:
                ax.set_xlabel("pixels / second", fontsize=9)
            else:
                ax.tick_params(labelbottom=False)

    legend_elems = [
        Line2D([0], [0], color=SPLIT_COLORS["training"], lw=3, label="Training"),
        Line2D([0], [0], color=SPLIT_COLORS["testing"], lw=3, label="Testing"),
    ]
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.06, right=0.99, hspace=0.12, wspace=0.05)

    # Center title/legend on the subplot grid (not the full figure canvas).
    grid_x0 = min(axes[0, ci].get_position().x0 for ci in range(n_cols))
    grid_x1 = max(axes[0, ci].get_position().x1 for ci in range(n_cols))
    grid_center_x = (grid_x0 + grid_x1) / 2.0
    grid_top = max(axes[0, ci].get_position().y1 for ci in range(n_cols))

    fig.suptitle(
        "Temporal Feature Distributions",
        fontsize=14,
        fontweight="bold",
        x=grid_center_x,
        y=grid_top + 0.105,
        ha="center",
    )
    fig.legend(
        handles=legend_elems,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(grid_center_x, grid_top + 0.065),
    )

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"\nSaved ridgeline figure: {out_png}")


def default_out_png(iqr_k):
    if iqr_k is None:
        suffix = "_raw"
    else:
        k_str = str(iqr_k).replace(".", "p")
        suffix = f"_iqr{k_str}"
    return os.path.join(
        ROOT,
        "Image_Generation",
        "ChongSOTA",
        "TemporalEncoding",
        "feature_distributions",
        f"all_datasets_features_ridgeline{suffix}.png",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ridgeline plot: all datasets × features, train vs test per case.",
    )
    parser.add_argument(
        "--out_png",
        default=None,
        help="Output PNG path (default: feature_distributions/all_datasets_features_ridgeline_iqr1p5.png).",
    )
    parser.add_argument(
        "--iqr_k",
        type=float,
        default=1.5,
        help="Tukey fence multiplier: keep [Q1−k·IQR, Q3+k·IQR] per split (default 1.5).",
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="Disable IQR outlier clipping (plot full range).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=80000,
        help="Max points per split for KDE computation only; height uses full n (default 80000).",
    )
    parser.add_argument(
        "--ridge_scale",
        type=float,
        default=0.82,
        help="Vertical height of the largest ridge in each cell (smaller n → shorter).",
    )
    parser.add_argument(
        "--x_grid_n",
        type=int,
        default=400,
        help="Number of x points per KDE curve.",
    )
    parser.add_argument(
        "--log_x_velocity",
        action="store_true",
        help="Use symlog x-axis on the |v| panel only.",
    )
    args = parser.parse_args()

    iqr_k = None if args.no_clip else args.iqr_k
    if iqr_k is not None and iqr_k <= 0:
        parser.error("--iqr_k must be > 0 (or use --no_clip).")

    out_png = resolve_path(args.out_png) if args.out_png else default_out_png(iqr_k)

    print("[ROOT]", ROOT)
    print("[IQR k]", iqr_k if iqr_k is not None else "disabled")
    print("[max_samples for KDE]", args.max_samples)
    print("[units]", "pixels/second (ChaoShen/DFL ×1000)")

    clip_data, raw_data = collect_all(iqr_k, args.max_samples)

    plot_ridgeline_figure(
        clip_data,
        raw_data,
        iqr_k=iqr_k,
        log_x=args.log_x_velocity,
        ridge_scale=args.ridge_scale,
        x_grid_n=args.x_grid_n,
        out_png=out_png,
        max_kde_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
