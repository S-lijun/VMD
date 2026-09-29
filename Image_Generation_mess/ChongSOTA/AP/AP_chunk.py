# -*- coding: utf-8 -*-
"""
AP chunk: same windowing as SRP_chunk, then 4×4 grid per window.

Brightness matches SRP_chunk on the full window (e.g. 120 events):
  - coordinate normalization uses the whole window min/max
  - min/max greyscale mapping uses the full-window SRP matrix bounds
Grid layout only changes *where* each triangle is drawn, not the scale.
"""

import os
import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

# AP/ is one level deeper than ChongSOTA/SRP_chunk.py → three levels to repo root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

NUM_SUBSEQ = 8
GRID_SIZE = 8


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


def generate_windows(events, chunk_size, data_root):
    if len(events) < chunk_size:
        return []

    if "train" in data_root.lower():
        stride = chunk_size
    else:
        stride = chunk_size

    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i + chunk_size])
    return windows


def normalize_window_coords(window):
    """Same local normalization as SRP_chunk, on the full window."""
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


def srp_to_uint8_global(rp, rp_min, rp_max):
    """Map distances to greyscale using full-window SRP min/max."""
    denom = max(float(rp_max) - float(rp_min), 1e-8)
    return ((rp - rp_min) / denom * 255.0).astype(np.uint8)


def subseq_slices(n, k=NUM_SUBSEQ):
    q = n // k
    slices = []
    for i in range(k):
        start = i * q
        end = (i + 1) * q if i < k - 1 else n
        slices.append((start, end))
    return slices


_resize_tfms = {}


def _resize_transform(side):
    s = int(side)
    if s not in _resize_tfms:
        _resize_tfms[s] = transforms.Resize((s, s))
    return _resize_tfms[s]


def build_ap_cell(srp_a_u8, srp_b_u8, cell_size, same_subseq=False):
    """
    Merge two SRP patches (already on full-window greyscale) into one cell.
    Lower triangle from a, upper from b; diagonal filled only when a == b.
    """
    cell_size = int(cell_size)
    sa = cv2.resize(srp_a_u8, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)
    sb = cv2.resize(srp_b_u8, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR)

    cell = np.zeros((cell_size, cell_size), dtype=np.uint8)
    for i in range(cell_size):
        for j in range(cell_size):
            if i < j:
                cell[i, j] = sb[i, j]
            elif i > j:
                cell[i, j] = sa[i, j]
            else:
                cell[i, j] = sa[i, j] if same_subseq else 0
    return cell


def draw_ap_grid(window, save_path, epsilon, output_size=448):
    n = len(window)
    if n < NUM_SUBSEQ * 2:
        return False

    coords_norm = normalize_window_coords(window)
    full_rp = pairwise_srp(coords_norm, epsilon)
    rp_min = float(full_rp.min())
    rp_max = float(full_rp.max())

    srp_u8 = []
    for start, end in subseq_slices(n, NUM_SUBSEQ):
        if end - start < 2:
            return False
        sub_rp = pairwise_srp(coords_norm[start:end], epsilon)
        srp_u8.append(srp_to_uint8_global(sub_rp, rp_min, rp_max))

    if len(srp_u8) != NUM_SUBSEQ:
        return False

    out = int(output_size)
    if out <= 0:
        return False

    cell_px = out // GRID_SIZE
    if cell_px < 2:
        return False

    canvas = np.zeros((out, out), dtype=np.uint8)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            same = row == col
            cell_u8 = build_ap_cell(
                srp_u8[row], srp_u8[col], cell_px, same_subseq=same
            )

            y0, y1 = row * cell_px, (row + 1) * cell_px
            x0, x1 = col * cell_px, (col + 1) * cell_px
            canvas[y0:y1, x0:x1] = cell_u8

    pil = Image.fromarray(canvas, mode="L")
    canvas = np.asarray(_resize_transform(out)(pil), dtype=np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    return True


def process_dataset(dataset, data_root, out_dir, sizes, epsilon, output_size=448):
    users = sorted(os.listdir(data_root))
    skipped = 0

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print(f"[Phase] {GRID_SIZE}x{GRID_SIZE} AP grid (chunk windows, {NUM_SUBSEQ} equal subseqs per window)")

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
                        f"AP_event{chunk_size}",
                        user,
                        f"{session}-{i}.png",
                    )
                    if not draw_ap_grid(window, save_path, epsilon, output_size):
                        skipped += 1

    print("Skipped windows:", skipped)


def main():
    parser = argparse.ArgumentParser(
        description="AP 4×4 grid from chunk windows (same sliding windows as SRP_chunk)."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[120],
        help="Window length (same as SRP_chunk); split into 4 equal subseqs (e.g. 120 -> 4×30).",
    )
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="Output PNG side; each of 16 cells is output_size/4.",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)

    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)

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
