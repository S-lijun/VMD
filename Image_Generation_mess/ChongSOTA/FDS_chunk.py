# -*- coding: utf-8 -*-
"""
Single-channel FDS (same drawing as FDS.py / draw_fds), but sequences come from
pure fixed-size chunking (XYPlot_chunk.py): no time split, no merge.

Spectrogram magnitude Sxx (after log1p) uses global min/max over all chunks in
training_files only (training_root). Render pass (--data_root, e.g. testing_files)
reuses those bounds. Stats are saved to fds_sxx_global_norm.json under out_dir.

Note: chunks with fewer than 8 Move events are skipped for spectrogram (same gate as FDS.py).
"""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

# Drawing logic kept identical to FDS.draw_fds / FDS.compute_velocity (do not import
# FDS.py here: it pulls SRP.py which requires torchvision).


def compute_velocity(xs, ys, ts):
    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    dt = ts[1:] - ts[:-1]
    dt = np.maximum(dt, 1e-5)
    v = np.sqrt(dx * dx + dy * dy) / dt
    return v

def compute_vxvy(xs,ys,ts):

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    dt = ts[1:] - ts[:-1]

    dt = np.maximum(dt,1e-5)

    vx = np.zeros(len(xs))
    vy = np.zeros(len(xs))

    vx[1:] = dx/dt
    vy[1:] = dy/dt

    return vx,vy


def compute_sxx_log1p(seq_array):
    """Raw log1p spectrogram magnitude; None if too few points."""
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
        nperseg=min(10, len(v)),
        noverlap=min(5, len(v) // 2),
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


def save_norm_stats(path, dataset, chunk_size, training_root, global_min, global_max, n_spectrograms):
    payload = {
        "dataset": dataset,
        "chunk_size": chunk_size,
        "training_root": training_root,
        "sxx_stats_source": "training_files",
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


from XYPlot_chunk import (
    ROOT,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    DEFAULT_TRAINING_ROOT,
    split_by_chunk_size,
    build_user_max_xy_from_training,
    _clean_df,
)


def _iter_chunk_jobs(dataset, sessions_root, out_dir, chunk_size, user_max_xy):
    """Yield (seq_array, save_path, n_events) for every chunk under sessions_root."""
    users = sorted(os.listdir(sessions_root))

    for user in users:

        user_dir = os.path.join(sessions_root, user)

        if not os.path.isdir(user_dir):
            continue

        if user in user_max_xy:
            norm_x, norm_y = user_max_xy[user]
        else:
            norm_x, norm_y = GLOBAL_MAX_X, GLOBAL_MAX_Y
            print(
                "\n[WARN] User", user, "not in training scan; using GLOBAL_MAX_X/Y:",
                norm_x, norm_y,
            )

        print("\n------------------------------")
        print("User:", user, "| norm W×H (from training):", norm_x, norm_y)

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

            print("      Events:", len(events))
            if len(events) < 2:
                continue

            sequences = split_by_chunk_size(events, chunk_size)
            print("      Chunks:", len(sequences))

            for i, seq in enumerate(sequences):

                save_path = os.path.join(
                    out_dir,
                    "FDS",
                    user,
                    f"{session}-{i}.png",
                )

                seq_array = np.array(
                    [
                        [float(e["x"]), float(e["y"]), float(e["time"])]
                        for e in seq
                    ],
                    dtype=np.float32,
                )

                yield seq_array, save_path, len(seq)


def collect_global_sxx_bounds(dataset, sessions_root, out_dir, chunk_size, user_max_xy):
    global_min = np.inf
    global_max = -np.inf
    n_spectrograms = 0
    chunk_lengths = []
    skipped_short = 0

    for seq_array, _save_path, n_events in _iter_chunk_jobs(
        dataset, sessions_root, out_dir, chunk_size, user_max_xy
    ):
        chunk_lengths.append(n_events)
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
            "No valid spectrograms on training_root (need chunks with >=8 events). "
            "Cannot compute global Sxx bounds.",
        )

    return global_min, global_max, n_spectrograms, chunk_lengths, skipped_short


def write_all_fds_images(
    dataset,
    data_root,
    out_dir,
    chunk_size,
    user_max_xy,
    output_size,
    global_min,
    global_max,
):
    chunk_lengths = []
    skipped_short = 0
    written = 0

    for seq_array, save_path, n_events in _iter_chunk_jobs(
        dataset, data_root, out_dir, chunk_size, user_max_xy
    ):
        chunk_lengths.append(n_events)
        if n_events < 8:
            skipped_short += 1
            continue

        draw_fds_global(seq_array, save_path, global_min, global_max, output_size)
        written += 1

    return chunk_lengths, skipped_short, written


def process_dataset(
    dataset,
    training_root,
    data_root,
    out_dir,
    chunk_size,
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
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Chunk size:", chunk_size)
    print("FDS output_size:", output_size)

    if norm_stats_in:
        global_min, global_max, meta = load_norm_stats(norm_stats_in)
        print("\n[global Sxx] Loaded from file (skips training scan):", norm_stats_in)
        print("[global Sxx] meta:", meta)
    else:
        print("\n[global Sxx] Pass 1/2: scanning training_root chunks for Sxx min/max …")
        global_min, global_max, n_spec, chunk_lengths, skipped_short = collect_global_sxx_bounds(
            dataset, training_root, out_dir, chunk_size, user_max_xy
        )
        save_norm_stats(
            norm_stats_path,
            dataset,
            chunk_size,
            training_root,
            global_min,
            global_max,
            n_spec,
        )
        print("[global Sxx] Wrote", norm_stats_path)

        print("\n========== Events per chunk (pass 1, training tree only) ==========")
        if len(chunk_lengths) == 0:
            print("No chunks processed.")
        else:
            arr = np.array(chunk_lengths, dtype=np.float64)
            print("Total chunks:", len(arr))
            print(
                "event count min / median / mean / max:",
                int(arr.min()),
                float(np.median(arr)),
                float(np.mean(arr)),
                int(arr.max()),
            )
            print("Chunks with <8 events (no PNG):", skipped_short)

    print("\n========== Sxx global range (log1p |S|), used for all PNGs ==========")
    print("global_min:", global_min)
    print("global_max:", global_max)
    if not norm_stats_in:
        print("n_spectrograms (training, >=8 events):", n_spec)
        print("norm JSON:", norm_stats_path)
    else:
        print("(bounds from --norm_stats_in; see file for n_spectrograms if recorded)")

    print("\n[global Sxx] Writing PNGs (data_root) with training global min/max …")
    chunk_lengths, skipped_short, written = write_all_fds_images(
        dataset,
        data_root,
        out_dir,
        chunk_size,
        user_max_xy,
        output_size,
        global_min,
        global_max,
    )

    print("\n========== Events per chunk (render tree / final) ==========")
    if len(chunk_lengths) == 0:
        print("No chunks processed.")
    else:
        arr = np.array(chunk_lengths, dtype=np.float64)
        print("Total chunks:", len(arr))
        print(
            "event count min / median / mean / max:",
            int(arr.min()),
            float(np.median(arr)),
            float(np.mean(arr)),
            int(arr.max()),
        )
        print("Chunks with <8 events (no PNG):", skipped_short)
        print("PNG files written/updated this run (>=8 events):", written)


def main():

    parser = argparse.ArgumentParser(
        description="FDS velocity spectrogram from fixed-size event chunks (XYPlot_chunk logic).",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["balabit", "chaoshen", "dfl"],
    )

    parser.add_argument(
        "--data_root",
        required=True,
        help="Sessions to render (relative to project ROOT).",
    )

    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset training_files.",
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output root (relative to ROOT); images under FDS_velocity_single_channel_chunk/<user>/.",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        default=120,
        help="Number of Move events per chunk (same as XYPlot_chunk --sizes).",
    )

    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="Square PNG side length after resize (same as FDS.py).",
    )

    parser.add_argument(
        "--norm_stats_in",
        default=None,
        help="Optional JSON (global_min/global_max). Skips training_root Sxx scan; still renders --data_root.",
    )

    args = parser.parse_args()

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = os.path.join(ROOT, training_rel)
    print("[training_root]", training_rel)

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)
    norm_stats_path = default_norm_stats_path(out_dir)

    user_max_xy = build_user_max_xy_from_training(args.dataset, training_root)

    norm_in = args.norm_stats_in
    if norm_in and not os.path.isabs(norm_in):
        norm_in = os.path.join(ROOT, norm_in)

    process_dataset(
        args.dataset,
        training_root,
        data_root,
        out_dir,
        args.sizes,
        user_max_xy,
        args.output_size,
        norm_stats_path,
        norm_in,
    )

    print("\nFDS chunk generation finished.")


if __name__ == "__main__":
    main()
