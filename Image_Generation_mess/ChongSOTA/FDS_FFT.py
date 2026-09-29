# -*- coding: utf-8 -*-
"""
2D spatial-frequency images from XYPlot_chunk-style trajectory rasters.

Per chunk (fixed-size chunking via XYPlot_chunk.split_by_chunk_size):
  1) Render 2D trajectory on a square canvas (same logic as XYPlot_chunk.draw_sequence)
  2) Build signal plane: trajectory strokes -> high, white background -> low
  3) 2D FFT magnitude, fftshift (DC at center), log1p(|F|)
  4) Global min/max from training_files; render pass reuses bounds

Writes fds_fft_global_norm.json under out_dir.
PNG output: out_dir/FDS_FFT/<user>/<session>-<i>.png
"""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

from XYPlot_chunk import (
    ROOT,
    INNER_PADDING,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    DEFAULT_TRAINING_ROOT,
    split_by_chunk_size,
    build_user_max_xy_from_training,
    _clean_df,
)


def render_trajectory_gray(events, norm_width, norm_height, output_size=448):
    """
    In-memory XYPlot_chunk trajectory raster (grayscale uint8, square output_size).
    Black trajectory on white background; returns None if len(events) < 2.
    """
    if len(events) < 2:
        return None

    xs = np.array([float(e["x"]) for e in events], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in events], dtype=np.float64)

    w = max(float(norm_width), 1.0)
    h = max(float(norm_height), 1.0)
    aspect = h / w
    canvas_w = int(w) + 1
    span = float(canvas_w - 1)
    canvas_h = int(np.ceil(aspect * span)) + 1

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    x_pix = np.clip(np.rint(xs / w * span), 0, canvas_w - 1).astype(np.int32)
    y_pix = np.clip(np.rint(ys / w * span), 0, canvas_h - 1).astype(np.int32)

    prev = None
    for x_i, y_i in zip(x_pix, y_pix):
        if prev is not None:
            cv2.line(
                canvas,
                prev,
                (int(x_i), int(y_i)),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )
        prev = (int(x_i), int(y_i))

    ch, cw = canvas.shape[:2]
    side = int(output_size)
    effective = max(1, side - 2 * INNER_PADDING)
    scale = effective / max(cw, ch)
    new_w = max(1, int(cw * scale))
    new_h = max(1, int(ch * scale))

    resized = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    pad_top = (side - new_h) // 2
    pad_bottom = side - new_h - pad_top
    pad_left = (side - new_w) // 2
    pad_right = side - new_w - pad_left

    final = cv2.copyMakeBorder(
        gray,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    return final


def compute_fft_log_magnitude(gray_u8, subtract_mean=True):
    """
    log1p(|fftshift(fft2(signal))|) on trajectory raster.
    signal: trajectory=1, background=0 (from inverted grayscale).
    """
    signal = (255.0 - gray_u8.astype(np.float64)) / 255.0
    if subtract_mean:
        signal = signal - float(signal.mean())

    spectrum = np.fft.fftshift(np.fft.fft2(signal))
    return np.log1p(np.abs(spectrum))


def _mag_to_uint8(mag, global_min, global_max, output_size):
    denom = float(global_max - global_min)
    if denom <= 0.0:
        plane = np.zeros_like(mag, dtype=np.float64)
    else:
        plane = np.clip((mag.astype(np.float64) - global_min) / denom, 0.0, 1.0)

    img = (plane * 255.0).astype(np.uint8)
    side = int(output_size)
    if img.shape[0] != side or img.shape[1] != side:
        img = cv2.resize(img, (side, side), interpolation=cv2.INTER_LINEAR)
    return img


def draw_fft_chunk(events, save_path, norm_x, norm_y, global_min, global_max, output_size=448):
    gray = render_trajectory_gray(events, norm_x, norm_y, output_size)
    if gray is None:
        return False

    mag = compute_fft_log_magnitude(gray)
    img = _mag_to_uint8(mag, global_min, global_max, output_size)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    return True


def default_norm_stats_path(out_dir):
    return os.path.join(out_dir, "fds_fft_global_norm.json")


def save_norm_stats(path, dataset, chunk_size, training_root, global_min, global_max, n_ffts):
    payload = {
        "dataset": dataset,
        "chunk_size": chunk_size,
        "training_root": training_root,
        "fft_stats_source": "training_files",
        "representation": "log1p(|fftshift(fft2(trajectory_raster))|)",
        "trajectory_raster": "XYPlot_chunk-style, per-user norm W/H from training",
        "global_min": float(global_min),
        "global_max": float(global_max),
        "n_ffts": int(n_ffts),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_norm_stats(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["global_min"]), float(payload["global_max"]), payload


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

        for file in sorted(os.listdir(user_dir)):
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
                    "FDS_FFT",
                    user,
                    f"{session}-{i}.png",
                )
                yield seq, save_path, norm_x, norm_y, len(seq)


def collect_global_fft_bounds(dataset, sessions_root, out_dir, chunk_size, user_max_xy, output_size):
    global_min = np.inf
    global_max = -np.inf
    n_ffts = 0
    chunk_lengths = []
    skipped_short = 0

    for seq, _save_path, norm_x, norm_y, n_events in _iter_chunk_jobs(
        dataset, sessions_root, out_dir, chunk_size, user_max_xy
    ):
        chunk_lengths.append(n_events)
        if n_events < 2:
            skipped_short += 1
            continue

        gray = render_trajectory_gray(seq, norm_x, norm_y, output_size)
        if gray is None:
            skipped_short += 1
            continue

        mag = compute_fft_log_magnitude(gray)
        global_min = min(global_min, float(mag.min()))
        global_max = max(global_max, float(mag.max()))
        n_ffts += 1

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError(
            "No valid FFT maps on training_root (need chunks with >=2 events). "
            "Cannot compute global bounds.",
        )

    return global_min, global_max, n_ffts, chunk_lengths, skipped_short


def write_all_fft_images(
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

    for seq, save_path, norm_x, norm_y, n_events in _iter_chunk_jobs(
        dataset, data_root, out_dir, chunk_size, user_max_xy
    ):
        chunk_lengths.append(n_events)
        if n_events < 2:
            skipped_short += 1
            continue

        if draw_fft_chunk(seq, save_path, norm_x, norm_y, global_min, global_max, output_size):
            written += 1
        else:
            skipped_short += 1

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
    print("[FFT stats] training_root:", training_root)
    print("[render]    data_root:", data_root)
    print("Users (render tree):", len(render_users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Chunk size:", chunk_size)
    print("FDS_FFT output_size:", output_size)

    if norm_stats_in:
        global_min, global_max, meta = load_norm_stats(norm_stats_in)
        print("\n[global FFT] Loaded from file (skips training scan):", norm_stats_in)
        print("[global FFT] meta:", meta)
        n_ffts = meta.get("n_ffts")
    else:
        print("\n[global FFT] Pass 1/2: scanning training_root for log1p|FFT| min/max …")
        global_min, global_max, n_ffts, chunk_lengths, skipped_short = collect_global_fft_bounds(
            dataset, training_root, out_dir, chunk_size, user_max_xy, output_size
        )
        save_norm_stats(
            norm_stats_path,
            dataset,
            chunk_size,
            training_root,
            global_min,
            global_max,
            n_ffts,
        )
        print("[global FFT] Wrote", norm_stats_path)

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
            print("Chunks with <2 events (no PNG):", skipped_short)

    print("\n========== FFT global range (log1p |F|), used for all PNGs ==========")
    print("global_min:", global_min)
    print("global_max:", global_max)
    if not norm_stats_in:
        print("n_ffts (training, >=2 events):", n_ffts)
        print("norm JSON:", norm_stats_path)

    print("\n[global FFT] Writing PNGs (data_root) …")
    chunk_lengths, skipped_short, written = write_all_fft_images(
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
        print("Chunks with <2 events (no PNG):", skipped_short)
        print("PNG files written this run (>=2 events):", written)


def main():
    parser = argparse.ArgumentParser(
        description="2D FFT spatial-frequency maps from XYPlot_chunk trajectory rasters.",
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
        help="Output root (relative to ROOT); images under FDS_FFT/<user>/.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        default=120,
        help="Number of Move events per chunk (same as XYPlot_chunk).",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="Square trajectory raster and FFT PNG side length.",
    )
    parser.add_argument(
        "--norm_stats_in",
        default=None,
        help="Optional JSON (global_min/global_max). Skips training_root FFT scan.",
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

    print("\nFDS_FFT generation finished.")


if __name__ == "__main__":
    main()
