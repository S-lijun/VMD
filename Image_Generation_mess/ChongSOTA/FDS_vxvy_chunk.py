# -*- coding: utf-8 -*-
"""
3-channel FDS chunk images: fixed-size chunking (XYPlot_chunk), no time split / merge.

Per chunk:
  R = spectrogram(log1p(|v|))   — same speed signal as FDS_chunk grayscale
  G = spectrogram( compress(vx) )   — vx = dx/dt, compress then magnitude STFT, then log1p(Sxx)
  B = spectrogram( compress(vy) ) — same for vy

compress(x) is sign(x)*log1p(|x|): for x>=0 this equals log1p(x); for x<-1, plain log1p(x)
is not real-valued, so this signed log form is used for all x (finite for any finite vx,vy).

Global bounds for R, G, B are from training_files only (render reuses them).

Post-STFT (on **log1p(Sxx)** planes, before global scaling):
  - Optional 2D Gaussian blur (--sxx_blur_sigma) to reduce speckle.
  - Global scale: min/max (default --norm_pct_low 0 --norm_pct_high 100) or training-set
    percentiles on all pixels (--norm_pct_low 1 --norm_pct_high 99) to clip outliers.

Writes fds_vxvy_chunk_global_norm.json under out_dir.
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


def compute_velocity(xs, ys, ts):

    dt = np.maximum(np.diff(ts), 1e-5)

    v = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
    v = np.concatenate([[v[0]], v])

    return v


def compute_vx_vy(xs, ys, ts):

    dt = np.maximum(np.diff(ts), 1e-5)

    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    return vx, vy


def log1p_compress_signed(x):
    """
    STFT input compression for signed velocity components.
    Equals log1p(x) for x >= 0. For x < -1, plain log1p(x) is not real; sign*log1p(|x|) is finite.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def _spectrogram_log1p(signal_1d):
    _, _, Sxx = spectrogram(
        signal_1d,
        fs=1.0,
        nperseg=min(10, len(signal_1d)),
        noverlap=min(5, len(signal_1d) // 2),
        scaling="density",
        mode="magnitude",
    )
    return np.log1p(Sxx)


def blur_sxx_plane(Sxx, sigma):
    """2D Gaussian blur on a log1p(Sxx) plane; sigma<=0 is no-op."""
    if sigma is None or float(sigma) <= 0.0:
        return Sxx
    s = float(sigma)
    x = np.asarray(Sxx, dtype=np.float32)
    if x.size == 0:
        return Sxx
    # ksize (0,0) lets OpenCV derive size from sigma
    return cv2.GaussianBlur(x, (0, 0), s)


def postprocess_sxx_planes(Sxx_r, Sxx_g, Sxx_b, blur_sigma):
    br = blur_sxx_plane(Sxx_r, blur_sigma)
    bg = blur_sxx_plane(Sxx_g, blur_sigma)
    bb = blur_sxx_plane(Sxx_b, blur_sigma)
    return br, bg, bb


def flatten_sxx(Sxx):
    return np.asarray(Sxx, dtype=np.float64).ravel()


def compute_three_sxx_log1p(seq_array):
    """Returns (Sxx_r, Sxx_g, Sxx_b) or (None, None, None) if too short."""
    xs = seq_array[:, 0]
    ys = seq_array[:, 1]
    ts = seq_array[:, 2]

    if len(xs) < 8:
        return None, None, None

    v = compute_velocity(xs, ys, ts)
    vx, vy = compute_vx_vy(xs, ys, ts)

    if len(v) < 2:
        return None, None, None

    v_in = np.log1p(v)
    vx_in = log1p_compress_signed(vx)
    vy_in = log1p_compress_signed(vy)
    Sxx_r = _spectrogram_log1p(v_in)
    Sxx_g = _spectrogram_log1p(vx_in)
    Sxx_b = _spectrogram_log1p(vy_in)

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


def draw_fds_vxvy_rgb(
    seq_array,
    save_path,
    bounds_r,
    bounds_g,
    bounds_b,
    output_size=448,
    blur_sigma=0.0,
):
    Sxx_r, Sxx_g, Sxx_b = compute_three_sxx_log1p(seq_array)
    if Sxx_r is None:
        return

    Sxx_r, Sxx_g, Sxx_b = postprocess_sxx_planes(Sxx_r, Sxx_g, Sxx_b, blur_sigma)

    r = _sxx_plane_to_uint8(Sxx_r, bounds_r[0], bounds_r[1], output_size)
    g = _sxx_plane_to_uint8(Sxx_g, bounds_g[0], bounds_g[1], output_size)
    b = _sxx_plane_to_uint8(Sxx_b, bounds_b[0], bounds_b[1], output_size)

    rgb = np.stack([r, g, b], axis=-1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, bgr)


def default_norm_stats_path(out_dir):
    return os.path.join(out_dir, "fds_vxvy_chunk_global_norm.json")


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
    sxx_blur_sigma,
    norm_pct_low,
    norm_pct_high,
    norm_mode,
):
    payload = {
        "dataset": dataset,
        "chunk_size": chunk_size,
        "training_root": training_root,
        "sxx_stats_source": "training_files",
        "sxx_blur_sigma": float(sxx_blur_sigma),
        "norm_percentile_low": float(norm_pct_low),
        "norm_percentile_high": float(norm_pct_high),
        "norm_mode": norm_mode,
        "stats_are_on": "log1p(spectrogram_magnitude) per channel; vx/vy use sign*log1p(|.|) before STFT",
        "channels": {
            "R_v_log1p": {
                "signal_stft": "log1p(|v|)",
                "global_min": float(gmin_r),
                "global_max": float(gmax_r),
            },
            "G_vx": {
                "signal_stft": "sign(vx)*log1p(|vx|) then |STFT| then log1p(Sxx)",
                "global_min": float(gmin_g),
                "global_max": float(gmax_g),
            },
            "B_vy": {
                "signal_stft": "sign(vy)*log1p(|vy|) then |STFT| then log1p(Sxx)",
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
    r = ch["R_v_log1p"]
    g = ch["G_vx"]
    b = ch["B_vy"]
    bounds_r = (float(r["global_min"]), float(r["global_max"]))
    bounds_g = (float(g["global_min"]), float(g["global_max"]))
    bounds_b = (float(b["global_min"]), float(b["global_max"]))
    return bounds_r, bounds_g, bounds_b, payload


def render_options_from_payload(payload, blur_sigma_cli, pct_lo_cli, pct_hi_cli):
    """When using --norm_stats_in, prefer JSON for blur/percentile if keys exist."""
    blur = float(payload.get("sxx_blur_sigma", blur_sigma_cli))
    lo = float(payload.get("norm_percentile_low", pct_lo_cli))
    hi = float(payload.get("norm_percentile_high", pct_hi_cli))
    return blur, lo, hi


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
                    "FDS_vxvy",
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


def collect_global_bounds(
    dataset,
    sessions_root,
    out_dir,
    chunk_size,
    user_max_xy,
    blur_sigma,
    norm_pct_low,
    norm_pct_high,
):
    use_percentile = not (
        abs(float(norm_pct_low) - 0.0) < 1e-12
        and abs(float(norm_pct_high) - 100.0) < 1e-12
    )
    norm_mode = "percentile" if use_percentile else "minmax"

    if use_percentile:
        buf_r, buf_g, buf_b = [], [], []
    else:
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

        Sxx_r, Sxx_g, Sxx_b = postprocess_sxx_planes(Sxx_r, Sxx_g, Sxx_b, blur_sigma)

        if use_percentile:
            buf_r.append(flatten_sxx(Sxx_r))
            buf_g.append(flatten_sxx(Sxx_g))
            buf_b.append(flatten_sxx(Sxx_b))
        else:
            gmin_r = min(gmin_r, float(Sxx_r.min()))
            gmax_r = max(gmax_r, float(Sxx_r.max()))
            gmin_g = min(gmin_g, float(Sxx_g.min()))
            gmax_g = max(gmax_g, float(Sxx_g.max()))
            gmin_b = min(gmin_b, float(Sxx_b.min()))
            gmax_b = max(gmax_b, float(Sxx_b.max()))

        n_spectrograms += 1

    if use_percentile:
        if n_spectrograms == 0:
            raise RuntimeError(
                "No valid spectrograms on training_root (need chunks with >=8 events). "
                "Cannot compute global Sxx bounds.",
            )
        ar = np.concatenate(buf_r) if buf_r else np.array([], dtype=np.float64)
        ag = np.concatenate(buf_g) if buf_g else np.array([], dtype=np.float64)
        ab = np.concatenate(buf_b) if buf_b else np.array([], dtype=np.float64)
        pl, ph = float(norm_pct_low), float(norm_pct_high)
        gmin_r, gmax_r = float(np.percentile(ar, pl)), float(np.percentile(ar, ph))
        gmin_g, gmax_g = float(np.percentile(ag, pl)), float(np.percentile(ag, ph))
        gmin_b, gmax_b = float(np.percentile(ab, pl)), float(np.percentile(ab, ph))
        if gmax_r <= gmin_r:
            gmax_r = gmin_r + 1e-8
        if gmax_g <= gmin_g:
            gmax_g = gmin_g + 1e-8
        if gmax_b <= gmin_b:
            gmax_b = gmin_b + 1e-8
    elif not np.isfinite(gmin_r):
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
        norm_mode,
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
    blur_sigma,
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

        draw_fds_vxvy_rgb(
            seq_array,
            save_path,
            bounds_r,
            bounds_g,
            bounds_b,
            output_size,
            blur_sigma,
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
    blur_sigma,
    norm_pct_low,
    norm_pct_high,
):

    render_users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("[Sxx stats] training_root:", training_root)
    print("[render]    data_root:", data_root)
    print("Users (render tree):", len(render_users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Chunk size:", chunk_size)
    print("FDS_vxvy output_size:", output_size)

    if norm_stats_in:
        bounds_r, bounds_g, bounds_b, meta = load_norm_stats(norm_stats_in)
        blur_sigma, norm_pct_low, norm_pct_high = render_options_from_payload(
            meta,
            blur_sigma,
            norm_pct_low,
            norm_pct_high,
        )
        norm_mode = meta.get("norm_mode", "minmax")
        print("\n[global Sxx RGB] Loaded from file (skips training scan):", norm_stats_in)
        print("[global Sxx RGB] meta keys:", list(meta.keys()))
    else:
        print("\n[global Sxx RGB] Pass 1/2: scanning training_root …")
        bounds_r, bounds_g, bounds_b, n_spec, chunk_lengths, skipped_short, norm_mode = collect_global_bounds(
            dataset,
            training_root,
            out_dir,
            chunk_size,
            user_max_xy,
            blur_sigma,
            norm_pct_low,
            norm_pct_high,
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
            blur_sigma,
            norm_pct_low,
            norm_pct_high,
            norm_mode,
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

    print(
        "\n[render] sxx_blur_sigma=%s norm_percentiles=(%s,%s)"
        % (blur_sigma, norm_pct_low, norm_pct_high),
    )

    print("\n========== Sxx global range per channel (after blur; scale = %s) ==========" % norm_mode)
    print("R (log1p|v|):", bounds_r[0], bounds_r[1])
    print("G (vx):     ", bounds_g[0], bounds_g[1])
    print("B (vy):     ", bounds_b[0], bounds_b[1])
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
        blur_sigma,
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
        description="3-channel FDS (v, vx, vy spectrograms) from fixed-size chunks; global norm per channel from training.",
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
        help="Output root (relative to ROOT); images under FDS_vxvy/<user>/.",
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
        help="Optional JSON from a prior run (channels.R/G/B). Skips training scan.",
    )

    parser.add_argument(
        "--sxx_blur_sigma",
        type=float,
        default=0.5,
        help="Gaussian sigma on each log1p(Sxx) plane before resize (0 = off). Saved in JSON when training scan runs.",
    )

    parser.add_argument(
        "--norm_pct_low",
        type=float,
        default=0.0,
        help="Training-set low percentile per channel on blurred log1p(Sxx) (subsampled). Use 0 for exact global min.",
    )

    parser.add_argument(
        "--norm_pct_high",
        type=float,
        default=95.0,
        help="Training-set high percentile per channel. Use 100 for exact global max.",
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
        args.sxx_blur_sigma,
        args.norm_pct_low,
        args.norm_pct_high,
    )

    print("\nFDS_vxvy chunk generation finished.")


if __name__ == "__main__":
    main()
