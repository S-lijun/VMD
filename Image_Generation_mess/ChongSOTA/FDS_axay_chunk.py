# -*- coding: utf-8 -*-
"""
3-channel FDS chunk images (acceleration): same chunking / global Sxx norm as FDS_vxvy_chunk.

Per chunk (aligned length T-1 segments, same indexing as build_global_distribution.py):
  R = spectrogram( log1p(|a|) )   — scalar tangential acceleration from compute_acceleration logic
  G = spectrogram( compress(ax) ) — ax from compute_axay logic, then STFT
  B = spectrogram( compress(ay) ) — ay from compute_axay logic

compress(x) = sign(x)*log1p(|x|) for signed ax/ay (same as FDS_vxvy_chunk for vx/vy).

Global min/max per channel from training_files only; JSON: fds_axay_chunk_global_norm.json.
Images under FDS_axay/<user>/.
"""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

from XYPlot_chunk import (
    ROOT,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    DEFAULT_TRAINING_ROOT,
    split_by_chunk_size,
    build_user_max_xy_from_training,
    _clean_df,
)


def log1p_compress_signed(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def compute_a_ax_ay_segments(xs, ys, ts):
    """
    Same construction as build_global_distribution.compute_acceleration / compute_axay,
    but returns fixed-length per-interval series a[1:], ax[1:], ay[1:] (length T-1).
    """
    T = len(xs)
    if T < 3:
        return None, None, None

    dx = xs[1:] - xs[:-1]
    dy = ys[1:] - ys[:-1]
    dt = ts[1:] - ts[:-1]
    dt = np.maximum(dt, 1e-5)

    v_segment = np.sqrt(dx * dx + dy * dy) / dt

    v = np.zeros(T, dtype=np.float64)
    v[1:] = v_segment

    a = np.zeros(T, dtype=np.float64)
    a[1:] = (v[1:] - v[:-1]) / dt
    a_s = a[1:]

    vx = dx / dt
    vy = dy / dt
    vx_pad = np.concatenate([[vx[0]], vx])
    vy_pad = np.concatenate([[vy[0]], vy])

    ax = np.zeros(T, dtype=np.float64)
    ay = np.zeros(T, dtype=np.float64)
    ax[1:] = (vx_pad[1:] - vx_pad[:-1]) / dt
    ay[1:] = (vy_pad[1:] - vy_pad[:-1]) / dt
    ax_s = ax[1:]
    ay_s = ay[1:]

    a_s = np.nan_to_num(a_s, nan=0.0, posinf=0.0, neginf=0.0)
    ax_s = np.nan_to_num(ax_s, nan=0.0, posinf=0.0, neginf=0.0)
    ay_s = np.nan_to_num(ay_s, nan=0.0, posinf=0.0, neginf=0.0)

    return a_s, ax_s, ay_s


def _spectrogram_log1p(signal_1d):
    _, _, Sxx = spectrogram(
        signal_1d,
        fs=1.0,
        nperseg=min(10, len(signal_1d)),
        noverlap=min(5, len(signal_1d) // 2),
        scaling="spectrum",
        mode="magnitude",
    )
    return np.log1p(Sxx)


def compute_three_sxx_log1p(seq_array):
    xs = seq_array[:, 0]
    ys = seq_array[:, 1]
    ts = seq_array[:, 2]

    if len(xs) < 8:
        return None, None, None

    a_s, ax_s, ay_s = compute_a_ax_ay_segments(xs, ys, ts)
    if a_s is None or len(a_s) < 2:
        return None, None, None

    r_in = np.log1p(np.abs(a_s))
    g_in = log1p_compress_signed(ax_s)
    b_in = log1p_compress_signed(ay_s)

    Sxx_r = _spectrogram_log1p(r_in)
    Sxx_g = _spectrogram_log1p(g_in)
    Sxx_b = _spectrogram_log1p(b_in)

    return Sxx_r, Sxx_g, Sxx_b


def _sxx_plane_to_uint8(Sxx, global_min, global_max, output_size):
    denom = float(global_max - global_min)
    if denom <= 0.0:
        plane = np.zeros_like(Sxx, dtype=np.float64)
    else:
        plane = np.clip((Sxx.astype(np.float64) - global_min) / denom, 0.0, 1.0)

    img = (plane * 255.0).astype(np.uint8)
    img = np.flipud(img)
    img = cv2.resize(
        img,
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    )
    return img


def draw_fds_axay_rgb(seq_array, save_path, bounds_r, bounds_g, bounds_b, output_size=448):
    Sxx_r, Sxx_g, Sxx_b = compute_three_sxx_log1p(seq_array)
    if Sxx_r is None:
        return

    r = _sxx_plane_to_uint8(Sxx_r, bounds_r[0], bounds_r[1], output_size)
    g = _sxx_plane_to_uint8(Sxx_g, bounds_g[0], bounds_g[1], output_size)
    b = _sxx_plane_to_uint8(Sxx_b, bounds_b[0], bounds_b[1], output_size)

    rgb = np.stack([r, g, b], axis=-1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr)


def default_norm_stats_path(out_dir):
    return os.path.join(out_dir, "fds_axay_chunk_global_norm.json")


def save_norm_stats(
    path,
    dataset,
    chunk_size,
    training_root,
    gmin_r,
    gmax_r,
    gmin_g,
    gmax_g,
    gmin_b,
    gmax_b,
    n_spectrograms,
):
    payload = {
        "dataset": dataset,
        "chunk_size": chunk_size,
        "training_root": training_root,
        "sxx_stats_source": "training_files",
        "pipeline": "FDS_axay_chunk",
        "stats_are_on": "log1p(spectrogram_magnitude) per channel",
        "channels": {
            "R_a_mag": {
                "signal_stft": "log1p(|a|), a = tangential d|v|/dt (build_global_distribution.compute_acceleration logic)",
                "global_min": float(gmin_r),
                "global_max": float(gmax_r),
            },
            "G_ax": {
                "signal_stft": "sign(ax)*log1p(|ax|) then |STFT| then log1p(Sxx)",
                "global_min": float(gmin_g),
                "global_max": float(gmax_g),
            },
            "B_ay": {
                "signal_stft": "sign(ay)*log1p(|ay|) then |STFT| then log1p(Sxx)",
                "global_min": float(gmin_b),
                "global_max": float(gmax_b),
            },
        },
        "n_spectrograms": int(n_spectrograms),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_norm_stats(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    ch = payload["channels"]
    r = ch["R_a_mag"]
    g = ch["G_ax"]
    b = ch["B_ay"]
    bounds_r = (float(r["global_min"]), float(r["global_max"]))
    bounds_g = (float(g["global_min"]), float(g["global_max"]))
    bounds_b = (float(b["global_min"]), float(b["global_max"]))
    return bounds_r, bounds_g, bounds_b, payload


def _iter_chunk_jobs(dataset, sessions_root, out_dir, chunk_size, user_max_xy):
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
                    "FDS_axay",
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


def collect_global_bounds(dataset, sessions_root, out_dir, chunk_size, user_max_xy):
    gmin_r, gmax_r = np.inf, -np.inf
    gmin_g, gmax_g = np.inf, -np.inf
    gmin_b, gmax_b = np.inf, -np.inf
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

        Sxx_r, Sxx_g, Sxx_b = compute_three_sxx_log1p(seq_array)
        if Sxx_r is None:
            skipped_short += 1
            continue

        gmin_r = min(gmin_r, float(Sxx_r.min()))
        gmax_r = max(gmax_r, float(Sxx_r.max()))
        gmin_g = min(gmin_g, float(Sxx_g.min()))
        gmax_g = max(gmax_g, float(Sxx_g.max()))
        gmin_b = min(gmin_b, float(Sxx_b.min()))
        gmax_b = max(gmax_b, float(Sxx_b.max()))
        n_spectrograms += 1

    if not np.isfinite(gmin_r):
        raise RuntimeError(
            "No valid spectrograms on training_root (need chunks with >=8 events). "
            "Cannot compute global Sxx bounds.",
        )

    return (
        (gmin_r, gmax_r),
        (gmin_g, gmax_g),
        (gmin_b, gmax_b),
        n_spectrograms,
        chunk_lengths,
        skipped_short,
    )


def write_all_images(
    dataset,
    data_root,
    out_dir,
    chunk_size,
    user_max_xy,
    output_size,
    bounds_r,
    bounds_g,
    bounds_b,
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

        draw_fds_axay_rgb(
            seq_array,
            save_path,
            bounds_r,
            bounds_g,
            bounds_b,
            output_size,
        )
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
    print("FDS_axay output_size:", output_size)

    if norm_stats_in:
        bounds_r, bounds_g, bounds_b, meta = load_norm_stats(norm_stats_in)
        print("\n[global Sxx RGB] Loaded from file (skips training scan):", norm_stats_in)
        print("[global Sxx RGB] meta keys:", list(meta.keys()))
    else:
        print("\n[global Sxx RGB] Pass 1/2: scanning training_root …")
        bounds_r, bounds_g, bounds_b, n_spec, chunk_lengths, skipped_short = collect_global_bounds(
            dataset, training_root, out_dir, chunk_size, user_max_xy
        )
        save_norm_stats(
            norm_stats_path,
            dataset,
            chunk_size,
            training_root,
            bounds_r[0],
            bounds_r[1],
            bounds_g[0],
            bounds_g[1],
            bounds_b[0],
            bounds_b[1],
            n_spec,
        )
        print("[global Sxx RGB] Wrote", norm_stats_path)

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

    print("\n========== Sxx global range per channel (log1p |STFT|) ==========")
    print("R (log1p|a|):", bounds_r[0], bounds_r[1])
    print("G (ax):     ", bounds_g[0], bounds_g[1])
    print("B (ay):     ", bounds_b[0], bounds_b[1])
    if not norm_stats_in:
        print("n_spectrograms (training, >=8 events):", n_spec)
        print("norm JSON:", norm_stats_path)

    print("\n[global Sxx RGB] Writing PNGs (data_root) …")
    chunk_lengths, skipped_short, written = write_all_images(
        dataset,
        data_root,
        out_dir,
        chunk_size,
        user_max_xy,
        output_size,
        bounds_r,
        bounds_g,
        bounds_b,
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
        print("PNG files written this run (>=8 events):", written)


def main():

    parser = argparse.ArgumentParser(
        description="3-channel FDS (|a|, ax, ay spectrograms) from fixed-size chunks; global norm per channel from training.",
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
        help="Output root (relative to ROOT); images under FDS_axay/<user>/.",
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
        help="Square PNG side length after resize.",
    )

    parser.add_argument(
        "--norm_stats_in",
        default=None,
        help="Optional JSON (channels R_a_mag / G_ax / B_ay). Skips training scan.",
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

    print("\nFDS_axay chunk generation finished.")


if __name__ == "__main__":
    main()
