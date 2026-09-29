# -*- coding: utf-8 -*-
"""
AP vx/vy chunk: same windowing as AP_chunk, then 8×8 grid per window.

Per window:
  1) split into NUM_SUBSEQ equal subsequences
  2) build one RGB vx/vy patch per subsequence:
       R = pair-wise SRP distance on locally normalized x,y
           (mapped by full-window SRP min/max)
       G = vx via global signed CDF + vertical stripe
       B = vy via global signed CDF + vertical stripe
  3) compose GRID_SIZE×GRID_SIZE cells:
       lower triangle from row subseq patch,
       upper triangle from col subseq patch,
       diagonal filled only when row == col
"""

import os
import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata
from torchvision import transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

NUM_SUBSEQ = 4
GRID_SIZE = 4

GLOBAL_VX_CDF = None
GLOBAL_VY_CDF = None


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


def load_raw_directional_velocity_distribution(path):
    data = np.load(path)
    vx = data["vx"]
    vy = data["vy"]

    print("\n[Directional Velocity Distribution]")
    print("\nvx")
    print("samples:", len(vx))
    print("min:", vx.min())
    print("max:", vx.max())
    print("\nvy")
    print("samples:", len(vy))
    print("min:", vy.min())
    print("max:", vy.max())

    return vx, vy


def build_runtime_cdf_signed(raw_values, clip_pct):
    print("\nBuilding signed runtime CDF")

    lower = np.percentile(raw_values, 100 - clip_pct)
    upper = np.percentile(raw_values, clip_pct)
    clipped = raw_values[(raw_values >= lower) & (raw_values <= upper)]

    ranks = rankdata(clipped, method="average")
    cdf = (ranks - 1) / (len(clipped) - 1 + 1e-8)

    order = np.argsort(clipped)
    v_sorted = clipped[order]
    cdf_sorted = cdf[order]

    print("runtime samples:", len(v_sorted))
    print("runtime min:", v_sorted.min())
    print("runtime max:", v_sorted.max())

    return v_sorted, cdf_sorted


def compute_vx_vy(xs, ys, ts):
    dt = np.maximum(np.diff(ts), 1e-5)
    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt

    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])
    return vx, vy


def clean_balabit(df):
    df = df.rename(
        columns={
            "client timestamp": "time",
            "x": "x",
            "y": "y",
            "state": "state",
        }
    )
    df = df[df["state"] == "Move"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()
    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y", "time"])


def generate_windows(events, chunk_size, data_root):
    if len(events) < chunk_size:
        return []

    if "train" in data_root.lower():
        stride = chunk_size
    else:
        stride = chunk_size

    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i : i + chunk_size])
    return windows


def normalize_window_coords(window):
    coords = window[:, :2].astype(np.float32)
    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    scale = max(x_max - x_min, y_max - y_min)
    if scale < 1e-8:
        scale = 1e-8

    x_norm = (x - x_min) / scale
    y_norm = (y - y_min) / scale
    return np.stack([x_norm, y_norm], axis=1)


def pairwise_srp(coords_norm, epsilon):
    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.minimum(dist, epsilon)


def subseq_slices(n, k=NUM_SUBSEQ):
    q = n // k
    slices = []
    for i in range(k):
        start = i * q
        end = (i + 1) * q if i < k - 1 else n
        slices.append((start, end))
    return slices


def compute_vxvy_patch(seq, rp_min, rp_max, epsilon):
    """
    Build one RGB patch for a subsequence:
      R = normalized SRP distance matrix
      G = vx vertical stripes (signed CDF normalized)
      B = vy vertical stripes (signed CDF normalized)
    """
    t = len(seq)
    if t < 2:
        return None

    coords_norm = normalize_window_coords(seq)
    rp = pairwise_srp(coords_norm, epsilon)
    denom = max(float(rp_max) - float(rp_min), 1e-8)
    r_channel = ((rp - rp_min) / denom).astype(np.float32)

    xs = seq[:, 0]
    ys = seq[:, 1]
    ts = seq[:, 2]
    vx, vy = compute_vx_vy(xs, ys, ts)

    vx_norm = np.interp(vx, GLOBAL_VX_CDF[0], GLOBAL_VX_CDF[1], left=0, right=1)
    vy_norm = np.interp(vy, GLOBAL_VY_CDF[0], GLOBAL_VY_CDF[1], left=0, right=1)

    stripe_x = np.tile(vx_norm[None, :], (t, 1)).astype(np.float32)
    stripe_y = np.tile(vy_norm[None, :], (t, 1)).astype(np.float32)

    patch = np.stack([r_channel, stripe_x, stripe_y], axis=-1)
    return np.clip(patch, 0, 1)


_resize_tfms = {}


def _resize_transform(side):
    s = int(side)
    if s not in _resize_tfms:
        _resize_tfms[s] = transforms.Resize((s, s))
    return _resize_tfms[s]


def resize_patch_rgb01(img_rgb01, side):
    img_u8 = (np.clip(img_rgb01, 0, 1) * 255.0).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="RGB")
    out = _resize_transform(side)(pil)
    out_u8 = np.asarray(out, dtype=np.uint8)
    return out_u8.astype(np.float32) / 255.0


def build_ap_cell_rgb(patch_a, patch_b, cell_size, same_subseq=False):
    """
    Merge two RGB patches into one AP cell.
    Lower triangle from a, upper triangle from b, diagonal kept only if same_subseq.
    """
    cell_size = int(cell_size)
    pa = resize_patch_rgb01(patch_a, cell_size)
    pb = resize_patch_rgb01(patch_b, cell_size)

    cell = np.zeros((cell_size, cell_size, 3), dtype=np.float32)
    for i in range(cell_size):
        for j in range(cell_size):
            if i < j:
                cell[i, j, :] = pb[i, j, :]
            elif i > j:
                cell[i, j, :] = pa[i, j, :]
            else:
                if same_subseq:
                    cell[i, j, :] = pa[i, j, :]
                else:
                    cell[i, j, :] = 0.0
    return cell


def draw_ap_grid_vxvy(window, save_path, epsilon, output_size=448):
    n = len(window)
    if n < NUM_SUBSEQ * 2:
        return False

    full_coords_norm = normalize_window_coords(window)
    full_rp = pairwise_srp(full_coords_norm, epsilon)
    rp_min = float(full_rp.min())
    rp_max = float(full_rp.max())

    patches = []
    for start, end in subseq_slices(n, NUM_SUBSEQ):
        if end - start < 2:
            return False
        patch = compute_vxvy_patch(window[start:end], rp_min, rp_max, epsilon)
        if patch is None:
            return False
        patches.append(patch)

    if len(patches) != NUM_SUBSEQ:
        return False

    out = int(output_size)
    if out <= 0:
        return False

    cell_px = out // GRID_SIZE
    if cell_px < 2:
        return False

    canvas = np.zeros((out, out, 3), dtype=np.float32)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell = build_ap_cell_rgb(
                patches[row],
                patches[col],
                cell_px,
                same_subseq=(row == col),
            )
            y0, y1 = row * cell_px, (row + 1) * cell_px
            x0, x1 = col * cell_px, (col + 1) * cell_px
            canvas[y0:y1, x0:x1, :] = cell

    img_u8 = (np.clip(canvas, 0, 1) * 255.0).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="RGB")
    img_u8 = np.asarray(_resize_transform(out)(pil), dtype=np.uint8)

    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)
    return True


def process_dataset(dataset, data_root, out_dir, sizes, epsilon, output_size=448):
    users = sorted(os.listdir(data_root))
    skipped = 0

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print(
        f"[Phase] {GRID_SIZE}x{GRID_SIZE} AP_vxvy grid "
        f"(chunk windows, {NUM_SUBSEQ} equal subseqs per window)"
    )

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("User:", user)

        for file in os.listdir(user_dir):
            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]
            df = pd.read_csv(path)

            if dataset.lower() == "balabit":
                df = clean_balabit(df)
            else:
                for c in ["x", "y", "time"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["x", "y", "time"])

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:
                windows = generate_windows(events, chunk_size, data_root)
                print(f"  Session {session} | chunk={chunk_size} -> {len(windows)} windows")

                for i, window in enumerate(windows):
                    save_path = os.path.join(
                        out_dir,
                        f"AP_vxvy_event{chunk_size}",
                        user,
                        f"{session}-{i}.png",
                    )
                    if not draw_ap_grid_vxvy(window, save_path, epsilon, output_size):
                        skipped += 1

    print("Skipped windows:", skipped)


def main():
    global GLOBAL_VX_CDF
    global GLOBAL_VY_CDF

    parser = argparse.ArgumentParser(
        description="AP 8x8 grid from chunk windows with vx/vy temporal encoding.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument(
        "--velocity_dist",
        required=True,
        help="npz with vx, vy arrays (e.g. vx_vy_distribution_raw.npz)",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[120],
        help="Window length; each window split into 8 equal subseqs.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="Output PNG side; grid cell side is output_size/8.",
    )
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument(
        "--v_percentile",
        type=float,
        default=100,
        help="Signed CDF clip percentile for vx/vy.",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    dist_path = resolve_path(args.velocity_dist)

    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("Resolved velocity_dist:", dist_path)

    vx_raw, vy_raw = load_raw_directional_velocity_distribution(dist_path)
    GLOBAL_VX_CDF = build_runtime_cdf_signed(vx_raw, args.v_percentile)
    GLOBAL_VY_CDF = build_runtime_cdf_signed(vy_raw, args.v_percentile)

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        epsilon=args.epsilon,
        output_size=args.output_size,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
