# -*- coding: utf-8 -*-

"""
FAST FDS (Frequency Domain Spectrogram)

Single-channel grayscale version.

Representation:
velocity sequence spectrogram

After log1p(|Sxx|), intensities are scaled with global min/max from training_files
only; render pass (--data_root) uses the same bounds. Stats: fds_sxx_global_norm.json.

IMPORTANT:
- NO matplotlib
- NO fake RGB colormap
- single signal -> single channel
- suitable for future multi-signal stacking
"""

import argparse
import json
import os

import pandas as pd
import numpy as np
import cv2

from scipy.signal import spectrogram

from SRP import (
    ROOT,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
)

GLOBAL_MAX_X = 1919.0
GLOBAL_MAX_Y = 1079.0

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
}

# ============================================================
# Clean wrapper
# ============================================================

def _clean_df(dataset, df):

    if dataset == "balabit":
        return clean_balabit(df)

    if dataset == "chaoshen":
        return clean_chaoshen(df)

    if dataset == "dfl":
        return clean_dfl(df)

    raise ValueError(dataset)

# ============================================================
# Build user max x/y
# ============================================================

def build_user_max_xy_from_training(dataset, training_root):

    user_max_xy = {}

    users = sorted(os.listdir(training_root))

    for user in users:

        user_dir = os.path.join(training_root, user)

        if not os.path.isdir(user_dir):
            continue

        max_x = 0.0
        max_y = 0.0
        saw_points = False

        for name in sorted(os.listdir(user_dir)):

            path = os.path.join(user_dir, name)

            if not os.path.isfile(path):
                continue

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

# ============================================================
# Velocity sequence
# ============================================================

def compute_velocity(xs, ys, ts):

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    dt = ts[1:] - ts[:-1]

    dt = np.maximum(dt, 1e-5)

    v = np.sqrt(dx * dx + dy * dy) / dt

    return v

# ============================================================
# Spectrogram Sxx (log1p magnitude), global norm from training
# ============================================================

def compute_sxx_log1p(seq_array):
    """log1p spectrogram magnitude; None if too few points. Params match legacy draw_fds."""
    xs = seq_array[:, 0]
    ys = seq_array[:, 1]
    ts = seq_array[:, 2]

    if len(xs) < 8:
        return None

    v = compute_velocity(xs, ys, ts)
    v = np.log1p(v)

    _, _, Sxx = spectrogram(
        v,
        fs=1.0,
        nperseg=min(30, len(v)),
        noverlap=min(15, len(v) // 2),
        scaling="spectrum",
        mode="magnitude",
    )

    return np.log1p(Sxx)


def _sxx_to_uint8_image(Sxx, global_min, global_max, output_size):
    denom = float(global_max - global_min)
    if denom <= 0.0:
        Sxx_norm = np.zeros_like(Sxx, dtype=np.float64)
    else:
        Sxx_norm = np.clip((Sxx.astype(np.float64) - global_min) / denom, 0.0, 1.0)

    img = (Sxx_norm * 255.0).astype(np.uint8)
    img = np.flipud(img)
    img = cv2.resize(
        img,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )
    return img


def draw_fds_global(seq_array, save_path, global_min, global_max, output_size=448):
    Sxx = compute_sxx_log1p(seq_array)
    if Sxx is None:
        return

    img = _sxx_to_uint8_image(Sxx, global_min, global_max, output_size)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def default_norm_stats_path(out_dir):
    return os.path.join(out_dir, "fds_sxx_global_norm.json")


def save_norm_stats(path, dataset, training_root, global_min, global_max, n_spectrograms):
    payload = {
        "dataset": dataset,
        "training_root": training_root,
        "sxx_stats_source": "training_files",
        "pipeline": "FDS_split_by_time_merge",
        "spectrogram_nperseg_cap": 30,
        "spectrogram_noverlap_cap": 15,
        "global_min": float(global_min),
        "global_max": float(global_max),
        "n_spectrograms": int(n_spectrograms),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_norm_stats(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["global_min"]), float(payload["global_max"]), payload


def _iter_fds_sequence_jobs(dataset, sessions_root, out_dir, user_max_xy):
    """Yield (seq_array, save_path, n_events) for each sequence after time split + merge."""
    users = sorted(os.listdir(sessions_root))

    for user in users:

        user_dir = os.path.join(sessions_root, user)

        if not os.path.isdir(user_dir):
            continue

        if user in user_max_xy:
            norm_x, norm_y = user_max_xy[user]
        else:
            norm_x, norm_y = GLOBAL_MAX_X, GLOBAL_MAX_Y

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
            df = _clean_df(dataset, df)
            events = df.to_dict("records")

            if len(events) < 2:
                continue

            sequences = split_by_time(events)
            min_length = norm_x
            sequences = merge_sequences(sequences, min_length)

            for i, seq in enumerate(sequences):

                save_path = os.path.join(
                    out_dir,
                    "FDS",
                    user,
                    f"{session}-{i}.png",
                )

                seq_array = np.array(
                    [
                        [
                            float(e["x"]),
                            float(e["y"]),
                            float(e["time"]),
                        ]
                        for e in seq
                    ],
                    dtype=np.float32,
                )

                yield seq_array, save_path, len(seq)


def collect_global_sxx_bounds(dataset, sessions_root, out_dir, user_max_xy):
    global_min = np.inf
    global_max = -np.inf
    n_spectrograms = 0
    sequence_lengths = []
    skipped_short = 0

    for seq_array, _save_path, n_events in _iter_fds_sequence_jobs(
        dataset, sessions_root, out_dir, user_max_xy
    ):
        sequence_lengths.append(n_events)
        if n_events < 8:
            skipped_short += 1
            continue

        Sxx = compute_sxx_log1p(seq_array)
        if Sxx is None:
            skipped_short += 1
            continue

        global_min = min(global_min, float(Sxx.min()))
        global_max = max(global_max, float(Sxx.max()))
        n_spectrograms += 1

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError(
            "No valid spectrograms on training_root (need sequences with >=8 points). "
            "Cannot compute global Sxx bounds.",
        )

    return global_min, global_max, n_spectrograms, sequence_lengths, skipped_short


def write_all_fds_images(
    dataset,
    data_root,
    out_dir,
    user_max_xy,
    output_size,
    global_min,
    global_max,
):
    sequence_lengths = []
    skipped_short = 0
    written = 0

    for seq_array, save_path, n_events in _iter_fds_sequence_jobs(
        dataset, data_root, out_dir, user_max_xy
    ):
        sequence_lengths.append(n_events)
        if n_events < 8:
            skipped_short += 1
            continue

        draw_fds_global(seq_array, save_path, global_min, global_max, output_size)
        written += 1

    return sequence_lengths, skipped_short, written


# ============================================================
# Process dataset
# ============================================================

def process_dataset(
    dataset,
    training_root,
    data_root,
    out_dir,
    user_max_xy,
    output_size,
    norm_stats_path,
    norm_stats_in,
):

    render_users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("[Sxx stats] training_root:", training_root)
    print("[render]    data_root:", data_root)
    print("Users (render tree):", len(render_users))

    if norm_stats_in:
        global_min, global_max, meta = load_norm_stats(norm_stats_in)
        print("\n[global Sxx] Loaded from file (skips training scan):", norm_stats_in)
        print("[global Sxx] meta:", meta)
    else:
        print("\n[global Sxx] Pass 1/2: scanning training_root for Sxx min/max …")
        global_min, global_max, n_spec, sequence_lengths, skipped_short = collect_global_sxx_bounds(
            dataset, training_root, out_dir, user_max_xy
        )
        save_norm_stats(
            norm_stats_path,
            dataset,
            training_root,
            global_min,
            global_max,
            n_spec,
        )
        print("[global Sxx] Wrote", norm_stats_path)

        print("\n========== Sequence length (pass 1, training tree only) ==========")
        if len(sequence_lengths) == 0:
            print("No sequences.")
        else:
            lengths = np.array(sequence_lengths, dtype=np.float64)
            print("Total sequences:", len(lengths))
            print("min / median / mean / max:", int(lengths.min()), float(np.median(lengths)), float(np.mean(lengths)), int(lengths.max()))
            print("Sequences with <8 points (no PNG):", skipped_short)

    print("\n========== Sxx global range (log1p |S|), used for all PNGs ==========")
    print("global_min:", global_min)
    print("global_max:", global_max)
    if not norm_stats_in:
        print("n_spectrograms (training, >=8 points):", n_spec)
        print("norm JSON:", norm_stats_path)
    else:
        print("(bounds from --norm_stats_in)")

    print("\n[global Sxx] Writing PNGs (data_root) with training global min/max …")
    sequence_lengths, skipped_short, written = write_all_fds_images(
        dataset,
        data_root,
        out_dir,
        user_max_xy,
        output_size,
        global_min,
        global_max,
    )

    print("\n========== Sequence length (render tree / final) ==========")
    if len(sequence_lengths) == 0:
        print("No valid sequence generated.")
    else:
        lengths = np.array(sequence_lengths, dtype=np.float64)
        print("Total sequences:", len(lengths))
        print("min / median / mean / max:", int(lengths.min()), float(np.median(lengths)), float(np.mean(lengths)), int(lengths.max()))
        print("Sequences with <8 points (no PNG):", skipped_short)
        print("PNG files written this run (>=8 points):", written)

# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["balabit", "chaoshen", "dfl"]
    )

    parser.add_argument(
        "--training_root",
        default=None
    )

    parser.add_argument(
        "--data_root",
        required=True
    )

    parser.add_argument(
        "--out_dir",
        required=True
    )

    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--norm_stats_in",
        default=None,
        help="Optional JSON (global_min/global_max). Skips training_root Sxx scan; still renders --data_root.",
    )

    args = parser.parse_args()

    training_rel = (
        args.training_root
        or DEFAULT_TRAINING_ROOT[args.dataset]
    )

    training_root = os.path.join(ROOT, training_rel)
    print("[training_root]", training_rel)

    data_root = os.path.join(ROOT, args.data_root)

    out_dir = os.path.join(ROOT, args.out_dir)
    norm_stats_path = default_norm_stats_path(out_dir)

    user_max_xy = build_user_max_xy_from_training(
        args.dataset,
        training_root
    )

    norm_in = args.norm_stats_in
    if norm_in and not os.path.isabs(norm_in):
        norm_in = os.path.join(ROOT, norm_in)

    process_dataset(
        args.dataset,
        training_root,
        data_root,
        out_dir,
        user_max_xy,
        args.output_size,
        norm_stats_path,
        norm_in,
    )

    print("\nFAST single-channel FDS generation finished.")

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()