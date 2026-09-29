# -*- coding: utf-8 -*-
"""
Shared helpers for SRP + screen-location bias (RP_uc).

Pipeline:
  1) Build pair-wise SRP with chunk-local normalization (same as SRP_chunk).
  2) Min-max map SRP to [0, 255] / [0, 1].
  3) Per-user min/max from training scan.
  4) Location term (scalar α, scalar α+β, or pixel-wise α+β matrix) injected
     via bias method A/B/C with strength γ.
"""

import json
import os
import re
import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
BOUNDS_DIR = os.path.join(os.path.dirname(__file__), "bounds")

DEFAULT_SCAN_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL/training_files",
    "twos": "Data/TWOS/training_files",
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


def clean_twos(df):
    df = df.rename(columns={
        "timestamp": "time",
        "x": "x",
        "y": "y",
        "event": "event",
    })
    df = df[df["event"] == "Mouse Moved"]
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
    if dataset == "twos":
        return clean_twos(df)
    raise ValueError(dataset)


N_FOLDS = 5


def is_skipped_user_dir(name):
    """Data/TWOS 下面还有划分目录，不能当成用户。"""
    lower = name.lower()
    return lower == "training_files" or lower.startswith("testing_files")


def list_users(data_root):
    return sorted(
        [
            u for u in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, u)) and not is_skipped_user_dir(u)
        ],
        key=natural_key,
    )


def list_session_csvs(user_dir):
    return sorted(
        [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))],
        key=natural_key,
    )


def load_events(dataset, path):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)
    return df[["x", "y", "time"]].values.astype(np.float32)


def generate_windows(events, chunk_size, data_root):
    if len(events) < chunk_size:
        return []
    stride = chunk_size
    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i + chunk_size])
    return windows


def contiguous_fold_bounds(n_events, n_folds=N_FOLDS):
    """把一个 session 的事件按顺序切成 n_folds 段，段与段首尾相接、不重叠。"""
    return [
        (i * n_events // n_folds, (i + 1) * n_events // n_folds)
        for i in range(n_folds)
    ]


def iter_session_fold_windows(events, chunk_size, data_root, n_folds=N_FOLDS):
    """每一折只在自己那段连续事件里开窗，窗口不会跨到别的折。"""
    for fold, (start, end) in enumerate(contiguous_fold_bounds(len(events), n_folds)):
        yield fold, generate_windows(events[start:end], chunk_size, data_root)


def session_window_groups(events, chunk_size, data_root, five_fold):
    if five_fold:
        return list(iter_session_fold_windows(events, chunk_size, data_root))
    return [(None, generate_windows(events, chunk_size, data_root))]


# ChongSOTA SRP_per_user / XYPlot_per_user segmentation (not fixed-size chunking).
TIME_THRESHOLD = 1.0  # seconds


def compute_traj_length(seq_xy):
    """Path length of Nx2 (or Nx3) array of points."""
    if len(seq_xy) < 2:
        return 0.0
    d = np.diff(seq_xy[:, :2].astype(np.float64), axis=0)
    return float(np.sqrt(np.sum(d * d, axis=1)).sum())


def split_by_time(events, time_threshold=TIME_THRESHOLD):
    """
    Split Nx3 (x,y,time) array on gaps > time_threshold.
    Returns list of Nx3 float32 arrays.
    """
    if len(events) == 0:
        return []
    sequences = []
    current = [events[0]]
    for i in range(1, len(events)):
        dt = float(events[i, 2] - events[i - 1, 2])
        if dt > time_threshold:
            sequences.append(np.asarray(current, dtype=np.float32))
            current = []
        current.append(events[i])
    if current:
        sequences.append(np.asarray(current, dtype=np.float32))
    return sequences


def merge_sequences(sequences, min_length):
    """
    Merge consecutive sequences until path length >= min_length
    (same policy as ChongSOTA/SRP_per_user: min_length = user max_x).
    """
    if not sequences:
        return []
    merged = []
    i = 0
    while i < len(sequences):
        current = sequences[i]
        while compute_traj_length(current) < min_length and i + 1 < len(sequences):
            current = np.concatenate([current, sequences[i + 1]], axis=0)
            i += 1
        merged.append(current)
        i += 1
    return merged


def generate_per_user_sequences(events, min_length, time_threshold=TIME_THRESHOLD):
    """split_by_time -> merge_sequences(min_length); drop len < 2."""
    if len(events) < 2:
        return []
    sequences = split_by_time(events, time_threshold=time_threshold)
    sequences = merge_sequences(sequences, min_length)
    return [seq for seq in sequences if len(seq) >= 2]


def count_per_user_sequences(dataset, data_root, bounds):
    """Count samples under Chong-style per-user segmentation."""
    total = 0
    users = list_users(data_root)
    for user in users:
        user_dir = os.path.join(data_root, user)
        user_bounds = get_user_bounds(bounds, user)
        min_length = float(user_bounds["max_x"])
        for file in list_session_csvs(user_dir):
            events = load_events(dataset, os.path.join(user_dir, file))
            total += len(generate_per_user_sequences(events, min_length))
    return total, users


def compute_srp_pair(seq, epsilon):
    coords = seq[:, :2].astype(np.float32)
    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    scale = max(x_max - x_min, y_max - y_min)
    if scale < 1e-8:
        scale = 1e-8

    x_norm = (x - x_min) / scale
    y_norm = (y - y_min) / scale
    coords_norm = np.stack([x_norm, y_norm], axis=1)

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.minimum(dist, epsilon)


def compute_srp_pair_global_diag(seq, user_bounds, epsilon=1.0):
    """
    Pairwise distance after GLOBAL screen-diag normalization (not chunk-local):
      (x', y') = ((x-min_x)/diag, (y-min_y)/diag)
      dist = ||p'_i - p'_j||   # opposite corners of user bbox -> 1.0
      return min(dist, epsilon)
    """
    coords = seq[:, :2].astype(np.float32)
    min_x = float(user_bounds["min_x"])
    min_y = float(user_bounds["min_y"])
    diag = max(float(user_bounds["diag"]), 1e-8)
    coords_n = np.empty_like(coords)
    coords_n[:, 0] = (coords[:, 0] - min_x) / diag
    coords_n[:, 1] = (coords[:, 1] - min_y) / diag
    diff = coords_n[:, None, :] - coords_n[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.minimum(dist, float(epsilon))


_resize_tfms = {}


def _resize_transform(side):
    s = int(side)
    if s not in _resize_tfms:
        _resize_tfms[s] = transforms.Resize((s, s))
    return _resize_tfms[s]


def render_srp(seq, epsilon, output_size=0):
    """Return uint8 grayscale SRP after local min-max mapping, or None."""
    if len(seq) < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)
    rp_min = rp.min()
    rp_max = rp.max()
    denom = max(rp_max - rp_min, 1e-8)
    img = ((rp - rp_min) / denom * 255).astype(np.uint8)

    if output_size and int(output_size) > 0:
        s = int(output_size)
        pil = Image.fromarray(img, mode="L")
        out_pil = _resize_transform(s)(pil)
        img = np.asarray(out_pil, dtype=np.uint8)
    return img


def gray_to_tensor_chw(img):
    return np.stack([img, img, img], axis=0)


def _ab_event_loc(seq, user_bounds):
    """
    Per-event location score in [0, 1] from distances to BR/TR.

    s_k = dist(p_k, BR)/diag + dist(p_k, TR)/diag
    Map s from theoretical [s_min, s_max] into [0, 1] (no per-axis clip to 1).
    """
    coords = seq[:, :2].astype(np.float64)
    min_x = float(user_bounds["min_x"])
    max_x = float(user_bounds["max_x"])
    min_y = float(user_bounds["min_y"])
    max_y = float(user_bounds["max_y"])
    diag = max(float(user_bounds["diag"]), 1e-8)

    br = np.array([max_x, max_y], dtype=np.float64)
    tr = np.array([max_x, min_y], dtype=np.float64)
    d_br = np.sqrt(np.sum((coords - br) ** 2, axis=1)) / diag
    d_tr = np.sqrt(np.sum((coords - tr) ** 2, axis=1)) / diag
    s = d_br + d_tr

    # Inside bbox: min sum is on the right edge between TR/BR; max near left corners.
    s_min = abs(max_y - min_y) / diag
    s_max = 1.0 + abs(max_x - min_x) / diag
    loc = (s - s_min) / max(s_max - s_min, 1e-8)
    return np.clip(loc, 0.0, 1.0).astype(np.float32), d_br.astype(np.float32), d_tr.astype(np.float32)


def render_srp_tri_ab(seq, epsilon, output_size, user_bounds):
    """
    Hybrid triangle image (no gamma):
      upper triangle incl. diagonal: SRP local pairwise distance (min-max -> [0,1])
      lower triangle: vertical stripes by event index (left -> right = event 0 -> N-1)
        d_BR, d_TR in [0,1] (dist/diag, capped at 1)
        loc_k = d_BR(k) - d_TR(k) in [-1,1]
        stripe_k = (loc_k + 1) / 2   # linear map to [0,1]
        for i > j: out[i, j] = stripe[j]

    Built at native N×N, then optional resize.
    """
    if len(seq) < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)
    rp_min = float(rp.min())
    rp_max = float(rp.max())
    denom = max(rp_max - rp_min, 1e-8)
    srp = ((rp - rp_min) / denom).astype(np.float32)

    coords = seq[:, :2].astype(np.float64)
    diag = max(float(user_bounds["diag"]), 1e-8)
    br = np.array(
        [float(user_bounds["max_x"]), float(user_bounds["max_y"])],
        dtype=np.float64,
    )
    tr = np.array(
        [float(user_bounds["max_x"]), float(user_bounds["min_y"])],
        dtype=np.float64,
    )
    d_br = np.minimum(np.sqrt(np.sum((coords - br) ** 2, axis=1)) / diag, 1.0)
    d_tr = np.minimum(np.sqrt(np.sum((coords - tr) ** 2, axis=1)) / diag, 1.0)
    loc = (d_br - d_tr).astype(np.float32)          # [-1, 1]
    stripe = ((loc + 1.0) * 0.5).astype(np.float32)  # map [-1,1] -> [0,1]

    n = srp.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    upper = np.triu_indices(n, k=0)
    out[upper] = srp[upper]

    # Lower triangle: vertical stripes — column j uses event j's mapped loc.
    ii, jj = np.tril_indices(n, k=-1)
    out[ii, jj] = stripe[jj]

    img = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    return _maybe_resize_gray(img, output_size)


def render_br_tr_tri_stripe(seq, output_size, user_bounds):
    """
    Grayscale (no pairwise SRP); vertical stripes only:
      lower (i > j): out[i,j] = dist(event_j, BR) / diag   BR=(max_x, max_y)
      upper (i < j): out[i,j] = dist(event_j, TR) / diag   TR=(max_x, min_y)
      diagonal: 0.5 * (d_BR + d_TR) / diag
    Distances clipped to [0,1] after /diag.
    """
    if len(seq) < 2:
        return None

    coords = seq[:, :2].astype(np.float64)
    n = coords.shape[0]
    diag = max(float(user_bounds["diag"]), 1e-8)
    br = np.array(
        [float(user_bounds["max_x"]), float(user_bounds["max_y"])],
        dtype=np.float64,
    )
    tr = np.array(
        [float(user_bounds["max_x"]), float(user_bounds["min_y"])],
        dtype=np.float64,
    )
    d_br = np.minimum(
        np.sqrt(np.sum((coords - br) ** 2, axis=1)) / diag, 1.0
    ).astype(np.float32)
    d_tr = np.minimum(
        np.sqrt(np.sum((coords - tr) ** 2, axis=1)) / diag, 1.0
    ).astype(np.float32)

    out = np.zeros((n, n), dtype=np.float32)
    ii_l, jj_l = np.tril_indices(n, k=-1)
    out[ii_l, jj_l] = d_br[jj_l]
    ii_u, jj_u = np.triu_indices(n, k=1)
    out[ii_u, jj_u] = d_tr[jj_u]
    diag_idx = np.arange(n)
    out[diag_idx, diag_idx] = 0.5 * (d_br + d_tr)

    img = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    return _maybe_resize_gray(img, output_size)


def patch_grid_shape(max_x, max_y, short_side=10, min_x=0.0, min_y=0.0):
    """
    Screen aspect from per-user spans (handles negative coords, e.g. DFL):
      span_x = max_x - min_x, span_y = max_y - min_y
      n_y = short_side (default 10)
      n_x = round((span_x / span_y) * short_side)
    Returns (n_y, n_x), K = n_y * n_x.
    """
    mx = max(float(max_x) - float(min_x), 1e-8)
    my = max(float(max_y) - float(min_y), 1e-8)
    n_y = max(int(short_side), 1)
    n_x = max(int(round((mx / my) * n_y)), 1)
    return n_y, n_x


def _event_patch_ids(coords, max_x, max_y, n_y, n_x, min_x=0.0, min_y=0.0):
    """Map each (x,y) into patch id in [0, K), row-major: id = py * n_x + px."""
    span_x = max(float(max_x) - float(min_x), 1e-8)
    span_y = max(float(max_y) - float(min_y), 1e-8)
    px = np.floor((coords[:, 0] - float(min_x)) / span_x * n_x).astype(np.int32)
    py = np.floor((coords[:, 1] - float(min_y)) / span_y * n_y).astype(np.int32)
    px = np.clip(px, 0, n_x - 1)
    py = np.clip(py, 0, n_y - 1)
    return py * n_x + px


def _norm_xy_01(seq, user_bounds):
    """
    Per-user min-max normalize coords to [0, 1]:
      x' = (x - min_x) / (max_x - min_x)
      y' = (y - min_y) / (max_y - min_y)
    Needed for DFL (coords can be negative); for Balabit min≈0 this ≈ x/max_x.
    """
    min_x = float(user_bounds["min_x"])
    max_x = float(user_bounds["max_x"])
    min_y = float(user_bounds["min_y"])
    max_y = float(user_bounds["max_y"])
    span_x = max(max_x - min_x, 1e-8)
    span_y = max(max_y - min_y, 1e-8)
    x = np.clip((seq[:, 0].astype(np.float32) - min_x) / span_x, 0.0, 1.0)
    y = np.clip((seq[:, 1].astype(np.float32) - min_y) / span_y, 0.0, 1.0)
    return x, y


def _pair_patch_combo_index(a, b):
    """
    Unordered patch pair (with replacement) -> 0-based index in [0, C(K,2)+K).
    Uses sorted (lo, hi); lexicographic over hi then lo:
      index = hi*(hi+1)//2 + lo
    Covers same-cell (lo==hi) and distinct cells.
    """
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    return hi * (hi + 1) // 2 + lo


def render_srp_tri_patch(seq, epsilon, output_size, user_bounds, short_side=10):
    """
    Hybrid triangle image:
      upper (incl. diagonal): local SRP min-max -> [0,1]
      lower (i > j): patch-combination of events i and j
        screen [min_x,max_x] x [min_y,max_y] tiled n_y x n_x
        K = n_y * n_x; n_comb = K*(K+1)/2
        brightness = combo_index / (n_comb - 1) in [0,1]
    """
    if len(seq) < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)
    rp_min = float(rp.min())
    rp_max = float(rp.max())
    denom = max(rp_max - rp_min, 1e-8)
    srp = ((rp - rp_min) / denom).astype(np.float32)

    min_x = float(user_bounds["min_x"])
    max_x = float(user_bounds["max_x"])
    min_y = float(user_bounds["min_y"])
    max_y = float(user_bounds["max_y"])
    n_y, n_x = patch_grid_shape(
        max_x, max_y, short_side=short_side, min_x=min_x, min_y=min_y
    )
    k = n_y * n_x
    n_comb = k * (k + 1) // 2
    denom_c = float(max(n_comb - 1, 1))

    coords = seq[:, :2].astype(np.float64)
    patch_ids = _event_patch_ids(
        coords, max_x, max_y, n_y, n_x, min_x=min_x, min_y=min_y
    )

    n = srp.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    upper = np.triu_indices(n, k=0)
    out[upper] = srp[upper]

    ii, jj = np.tril_indices(n, k=-1)
    combo = _pair_patch_combo_index(patch_ids[ii], patch_ids[jj])
    out[ii, jj] = (combo.astype(np.float32) / denom_c)

    img = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    return _maybe_resize_gray(img, output_size)


def render_srp_rb_g_patch(seq, epsilon, output_size, user_bounds, short_side=10):
    """
    RGB image:
      R = B = full SRP (chunk-local pairwise, min-max)
      G = symmetric patch-pair matrix:
        same patch grid / combo index as render_srp_tri_patch
        G[i,j] = G[j,i] = combo(patch_i, patch_j) / (n_comb - 1)
    Returns HxWx3 uint8 RGB, or None.
    """
    if len(seq) < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)
    rp_min = float(rp.min())
    rp_max = float(rp.max())
    denom = max(rp_max - rp_min, 1e-8)
    srp = ((rp - rp_min) / denom).astype(np.float32)

    min_x = float(user_bounds["min_x"])
    max_x = float(user_bounds["max_x"])
    min_y = float(user_bounds["min_y"])
    max_y = float(user_bounds["max_y"])
    n_y, n_x = patch_grid_shape(
        max_x, max_y, short_side=short_side, min_x=min_x, min_y=min_y
    )
    k = n_y * n_x
    n_comb = k * (k + 1) // 2
    denom_c = float(max(n_comb - 1, 1))

    coords = seq[:, :2].astype(np.float64)
    patch_ids = _event_patch_ids(
        coords, max_x, max_y, n_y, n_x, min_x=min_x, min_y=min_y
    )
    pi = patch_ids[:, None]
    pj = patch_ids[None, :]
    G = _pair_patch_combo_index(pi, pj).astype(np.float32) / denom_c

    R = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R, G8, R], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def render_srp_rb_g_patch_stripe(seq, epsilon, output_size, user_bounds, short_side=10):
    """
    RGB image:
      R = B = full SRP (chunk-local pairwise, min-max)
      G = vertical stripes by event patch id (no pair combo):
        same n_y x n_x grid; K = n_y * n_x
        stripe[j] = patch_id(j) / (K - 1) in [0,1]
        G[i, j] = stripe[j]  for all i  (column j = event j)
    Returns HxWx3 uint8 RGB, or None.
    """
    if len(seq) < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)
    rp_min = float(rp.min())
    rp_max = float(rp.max())
    denom = max(rp_max - rp_min, 1e-8)
    srp = ((rp - rp_min) / denom).astype(np.float32)

    min_x = float(user_bounds["min_x"])
    max_x = float(user_bounds["max_x"])
    min_y = float(user_bounds["min_y"])
    max_y = float(user_bounds["max_y"])
    n_y, n_x = patch_grid_shape(
        max_x, max_y, short_side=short_side, min_x=min_x, min_y=min_y
    )
    k = n_y * n_x
    denom_k = float(max(k - 1, 1))

    coords = seq[:, :2].astype(np.float64)
    patch_ids = _event_patch_ids(
        coords, max_x, max_y, n_y, n_x, min_x=min_x, min_y=min_y
    )
    stripe = patch_ids.astype(np.float32) / denom_k

    n = srp.shape[0]
    G = np.broadcast_to(stripe[None, :], (n, n)).astype(np.float32).copy()

    R = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R, G8, R], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def _maybe_resize_rgb(img_hwc_u8, output_size):
    """img: HxWx3 uint8 RGB."""
    if not output_size or int(output_size) <= 0:
        return img_hwc_u8
    s = int(output_size)
    pil = Image.fromarray(img_hwc_u8, mode="RGB")
    return np.asarray(_resize_transform(s)(pil), dtype=np.uint8)


def render_srp_rb_g_xy(seq, epsilon, output_size, user_bounds):
    """
    RGB image:
      R = B = SRP (chunk-local pairwise, min-max), same as SRP_chunk
      G = position stripes (per-user min-max; supports negative coords e.g. DFL):
        lower triangle (i > j): G[i,j] = (x[j]-min_x)/(max_x-min_x)
        upper triangle (i < j): G[i,j] = (y[j]-min_y)/(max_y-min_y)
        diagonal: 0.5 * (x_norm[i] + y_norm[i])
      min/max from per-user training bounds.
    Returns HxWx3 uint8 RGB, or None.
    """
    if len(seq) < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)
    rp_min = float(rp.min())
    rp_max = float(rp.max())
    denom = max(rp_max - rp_min, 1e-8)
    srp = ((rp - rp_min) / denom).astype(np.float32)

    x, y = _norm_xy_01(seq, user_bounds)

    n = srp.shape[0]
    G = np.zeros((n, n), dtype=np.float32)
    ii_l, jj_l = np.tril_indices(n, k=-1)
    G[ii_l, jj_l] = x[jj_l]
    ii_u, jj_u = np.triu_indices(n, k=1)
    G[ii_u, jj_u] = y[jj_u]
    diag_idx = np.arange(n)
    G[diag_idx, diag_idx] = 0.5 * (x + y)

    R = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R, G8, R], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def render_srp_rb_g_xy_diag(seq, epsilon, output_size, user_bounds):
    """
    RGB image (global-diag SRP brightness):
      R = B:
        1) global normalize coords by user diag (NOT chunk-local scale)
             (x',y') = ((x-min_x)/diag, (y-min_y)/diag)
        2) dist = pairwise ||p'_i - p'_j||   # already in [0,1] (corner-to-corner=1)
        3) srp = min(dist, epsilon)          # NO per-trajectory dist min-max stretch
      G = same xy position stripes as render_srp_rb_g_xy
    Returns HxWx3 uint8 RGB, or None.
    """
    if len(seq) < 2:
        return None

    srp = compute_srp_pair_global_diag(seq, user_bounds, epsilon).astype(np.float32)

    x, y = _norm_xy_01(seq, user_bounds)

    n = srp.shape[0]
    G = np.zeros((n, n), dtype=np.float32)
    ii_l, jj_l = np.tril_indices(n, k=-1)
    G[ii_l, jj_l] = x[jj_l]
    ii_u, jj_u = np.triu_indices(n, k=1)
    G[ii_u, jj_u] = y[jj_u]
    diag_idx = np.arange(n)
    G[diag_idx, diag_idx] = 0.5 * (x + y)

    R = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R, G8, R], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def render_srp_rb_g_xy_diag_halves(seq, epsilon, output_size, user_bounds):
    """
    Same R/B as render_srp_rb_g_xy_diag (global-diag SRP, no local dist stretch).

    G = two horizontal rectangles, vertical stripes (column j = event j):
      top    (rows 0 .. mid-1): G[i,j] = (y[j]-min_y)/(max_y-min_y)
      bottom (rows mid .. n-1): G[i,j] = (x[j]-min_x)/(max_x-min_x)
    Each event gets the same number of x pixels and the same number of y pixels
    (bottom has one extra row when n is odd).
    Returns HxWx3 uint8 RGB, or None.
    """
    if len(seq) < 2:
        return None

    srp = compute_srp_pair_global_diag(seq, user_bounds, epsilon).astype(np.float32)
    x, y = _norm_xy_01(seq, user_bounds)

    n = srp.shape[0]
    mid = n // 2
    G = np.empty((n, n), dtype=np.float32)
    G[:mid, :] = y[None, :]
    G[mid:, :] = x[None, :]

    R = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R, G8, R], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def render_srp_r_gxy_b_vel(seq, epsilon, output_size, user_bounds, v_cdf):
    """
    RGB image:
      R = SRP (same as SRP_chunk)
      G = position vertical stripes ((x-min_x)/(max_x-min_x), (y-min_y)/(max_y-min_y))
      B = |v| vertical stripes via global CDF (same as SRP_chunk_velocity blue)

    v_cdf: (v_sorted, cdf_sorted) from build_runtime_cdf.
    Returns HxWx3 uint8 RGB, or None.
    """
    if len(seq) < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)
    rp_min = float(rp.min())
    rp_max = float(rp.max())
    denom = max(rp_max - rp_min, 1e-8)
    srp = ((rp - rp_min) / denom).astype(np.float32)

    x, y = _norm_xy_01(seq, user_bounds)

    n = srp.shape[0]
    G = np.zeros((n, n), dtype=np.float32)
    ii_l, jj_l = np.tril_indices(n, k=-1)
    G[ii_l, jj_l] = x[jj_l]
    ii_u, jj_u = np.triu_indices(n, k=1)
    G[ii_u, jj_u] = y[jj_u]
    diag_idx = np.arange(n)
    G[diag_idx, diag_idx] = 0.5 * (x + y)

    xs = seq[:, 0].astype(np.float64)
    ys = seq[:, 1].astype(np.float64)
    ts = seq[:, 2].astype(np.float64)
    dt = np.maximum(np.diff(ts), 1e-5)
    v = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / dt
    v = np.concatenate([[v[0]], v])
    v_norm = np.interp(v, v_cdf[0], v_cdf[1], left=0, right=1)
    B = np.tile(v_norm[None, :], (n, 1)).astype(np.float32)

    R8 = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    B8 = np.clip(np.rint(B * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R8, G8, B8], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def render_srp_r_gxy_b_vel_diag(seq, epsilon, output_size, user_bounds, v_cdf):
    """
    Same as render_srp_r_gxy_b_vel, but R uses global-diag SRP
    (compute_srp_pair_global_diag; no per-trajectory dist min-max).
    G / B unchanged.
    """
    if len(seq) < 2:
        return None

    srp = compute_srp_pair_global_diag(seq, user_bounds, epsilon).astype(np.float32)

    x, y = _norm_xy_01(seq, user_bounds)

    n = srp.shape[0]
    G = np.zeros((n, n), dtype=np.float32)
    ii_l, jj_l = np.tril_indices(n, k=-1)
    G[ii_l, jj_l] = x[jj_l]
    ii_u, jj_u = np.triu_indices(n, k=1)
    G[ii_u, jj_u] = y[jj_u]
    diag_idx = np.arange(n)
    G[diag_idx, diag_idx] = 0.5 * (x + y)

    xs = seq[:, 0].astype(np.float64)
    ys = seq[:, 1].astype(np.float64)
    ts = seq[:, 2].astype(np.float64)
    dt = np.maximum(np.diff(ts), 1e-5)
    v = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / dt
    v = np.concatenate([[v[0]], v])
    v_norm = np.interp(v, v_cdf[0], v_cdf[1], left=0, right=1)
    B = np.tile(v_norm[None, :], (n, 1)).astype(np.float32)

    R8 = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    B8 = np.clip(np.rint(B * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R8, G8, B8], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def render_srp_r_gxy_b_vel_diag_halves(seq, epsilon, output_size, user_bounds, v_cdf):
    """
    Same R/B as render_srp_r_gxy_b_vel_diag (global-diag SRP, |v| full-column stripes).

    G = two horizontal rectangles, vertical stripes (column j = event j):
      top    (rows 0 .. mid-1): G[i,j] = y_norm[j]
      bottom (rows mid .. n-1): G[i,j] = x_norm[j]
    Returns HxWx3 uint8 RGB, or None.
    """
    if len(seq) < 2:
        return None

    srp = compute_srp_pair_global_diag(seq, user_bounds, epsilon).astype(np.float32)
    x, y = _norm_xy_01(seq, user_bounds)

    n = srp.shape[0]
    mid = n // 2
    G = np.empty((n, n), dtype=np.float32)
    G[:mid, :] = y[None, :]
    G[mid:, :] = x[None, :]

    xs = seq[:, 0].astype(np.float64)
    ys = seq[:, 1].astype(np.float64)
    ts = seq[:, 2].astype(np.float64)
    dt = np.maximum(np.diff(ts), 1e-5)
    v = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) / dt
    v = np.concatenate([[v[0]], v])
    v_norm = np.interp(v, v_cdf[0], v_cdf[1], left=0, right=1)
    B = np.tile(v_norm[None, :], (n, 1)).astype(np.float32)

    R8 = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    B8 = np.clip(np.rint(B * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R8, G8, B8], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def render_srp_r_gxy_b_vxvy(seq, epsilon, output_size, user_bounds, vx_cdf, vy_cdf):
    """
    RGB image:
      R = SRP
      G = position vertical stripes ((x-min_x)/(max_x-min_x), (y-min_y)/(max_y-min_y))
      B = directional velocity in one channel:
           lower (i > j): vertical stripe vx[j] (signed CDF)
           upper (i < j): vertical stripe vy[j] (signed CDF)
           diagonal: 0.5 * (vx_norm[i] + vy_norm[i])

    vx_cdf / vy_cdf: (v_sorted, cdf_sorted) from build_runtime_cdf_signed.
    Returns HxWx3 uint8 RGB, or None.
    """
    if len(seq) < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)
    rp_min = float(rp.min())
    rp_max = float(rp.max())
    denom = max(rp_max - rp_min, 1e-8)
    srp = ((rp - rp_min) / denom).astype(np.float32)

    x, y = _norm_xy_01(seq, user_bounds)

    n = srp.shape[0]
    G = np.zeros((n, n), dtype=np.float32)
    ii_l, jj_l = np.tril_indices(n, k=-1)
    G[ii_l, jj_l] = x[jj_l]
    ii_u, jj_u = np.triu_indices(n, k=1)
    G[ii_u, jj_u] = y[jj_u]
    diag_idx = np.arange(n)
    G[diag_idx, diag_idx] = 0.5 * (x + y)

    xs = seq[:, 0].astype(np.float64)
    ys = seq[:, 1].astype(np.float64)
    ts = seq[:, 2].astype(np.float64)
    dt = np.maximum(np.diff(ts), 1e-5)
    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt
    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    vx_norm = np.interp(vx, vx_cdf[0], vx_cdf[1], left=0, right=1).astype(np.float32)
    vy_norm = np.interp(vy, vy_cdf[0], vy_cdf[1], left=0, right=1).astype(np.float32)

    B = np.zeros((n, n), dtype=np.float32)
    B[ii_l, jj_l] = vx_norm[jj_l]
    B[ii_u, jj_u] = vy_norm[jj_u]
    B[diag_idx, diag_idx] = 0.5 * (vx_norm + vy_norm)

    R8 = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    B8 = np.clip(np.rint(B * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R8, G8, B8], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def render_srp_r_gxy_b_vxvy_diag(
    seq, epsilon, output_size, user_bounds, vx_cdf, vy_cdf
):
    """
    Same as render_srp_r_gxy_b_vxvy, but R uses global-diag SRP
    (compute_srp_pair_global_diag; no per-trajectory dist min-max).
    G / B unchanged.
    """
    if len(seq) < 2:
        return None

    srp = compute_srp_pair_global_diag(seq, user_bounds, epsilon).astype(np.float32)

    x, y = _norm_xy_01(seq, user_bounds)

    n = srp.shape[0]
    G = np.zeros((n, n), dtype=np.float32)
    ii_l, jj_l = np.tril_indices(n, k=-1)
    G[ii_l, jj_l] = x[jj_l]
    ii_u, jj_u = np.triu_indices(n, k=1)
    G[ii_u, jj_u] = y[jj_u]
    diag_idx = np.arange(n)
    G[diag_idx, diag_idx] = 0.5 * (x + y)

    xs = seq[:, 0].astype(np.float64)
    ys = seq[:, 1].astype(np.float64)
    ts = seq[:, 2].astype(np.float64)
    dt = np.maximum(np.diff(ts), 1e-5)
    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt
    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    vx_norm = np.interp(vx, vx_cdf[0], vx_cdf[1], left=0, right=1).astype(np.float32)
    vy_norm = np.interp(vy, vy_cdf[0], vy_cdf[1], left=0, right=1).astype(np.float32)

    B = np.zeros((n, n), dtype=np.float32)
    B[ii_l, jj_l] = vx_norm[jj_l]
    B[ii_u, jj_u] = vy_norm[jj_u]
    B[diag_idx, diag_idx] = 0.5 * (vx_norm + vy_norm)

    R8 = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    B8 = np.clip(np.rint(B * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R8, G8, B8], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def render_srp_r_gxy_b_vxvy_diag_halves(
    seq, epsilon, output_size, user_bounds, vx_cdf, vy_cdf
):
    """
    Same R as render_srp_r_gxy_b_vxvy_diag (global-diag SRP).

    G and B use two horizontal rectangles, vertical stripes (column j = event j):
      G top=y_norm, G bottom=x_norm
      B top=vy_norm, B bottom=vx_norm
    (same halves cut as G so vx/vy are not triangle-biased by event index).
    Returns HxWx3 uint8 RGB, or None.
    """
    if len(seq) < 2:
        return None

    srp = compute_srp_pair_global_diag(seq, user_bounds, epsilon).astype(np.float32)
    x, y = _norm_xy_01(seq, user_bounds)

    n = srp.shape[0]
    mid = n // 2
    G = np.empty((n, n), dtype=np.float32)
    G[:mid, :] = y[None, :]
    G[mid:, :] = x[None, :]

    xs = seq[:, 0].astype(np.float64)
    ys = seq[:, 1].astype(np.float64)
    ts = seq[:, 2].astype(np.float64)
    dt = np.maximum(np.diff(ts), 1e-5)
    vx = np.diff(xs) / dt
    vy = np.diff(ys) / dt
    vx = np.concatenate([[vx[0]], vx])
    vy = np.concatenate([[vy[0]], vy])

    vx_norm = np.interp(vx, vx_cdf[0], vx_cdf[1], left=0, right=1).astype(np.float32)
    vy_norm = np.interp(vy, vy_cdf[0], vy_cdf[1], left=0, right=1).astype(np.float32)

    B = np.empty((n, n), dtype=np.float32)
    B[:mid, :] = vy_norm[None, :]
    B[mid:, :] = vx_norm[None, :]

    R8 = np.clip(np.rint(srp * 255.0), 0, 255).astype(np.uint8)
    G8 = np.clip(np.rint(G * 255.0), 0, 255).astype(np.uint8)
    B8 = np.clip(np.rint(B * 255.0), 0, 255).astype(np.uint8)
    rgb = np.stack([R8, G8, B8], axis=2)
    return _maybe_resize_rgb(rgb, output_size)


def rgb_to_tensor_chw(img_hwc):
    """HxWx3 RGB uint8 -> (3, H, W)."""
    return np.transpose(img_hwc, (2, 0, 1))


def default_bounds_json(dataset):
    return os.path.join(BOUNDS_DIR, "{}_xy_bounds.json".format(dataset))


def _make_user_bounds(min_x, max_x, min_y, max_y, n_points=0, n_files=0):
    diag = float(np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2))
    if diag < 1e-8:
        diag = 1e-8
    return {
        "min_x": float(min_x),
        "max_x": float(max_x),
        "min_y": float(min_y),
        "max_y": float(max_y),
        "corner_x": float(max_x),
        "corner_y": float(max_y),
        "diag": diag,
        "n_points": int(n_points),
        "n_files": int(n_files),
    }


def scan_dataset_xy_bounds(dataset, scan_root):
    """Scan each user under scan_root; record per-user min/max x,y."""
    users = list_users(scan_root)
    print("\n[bounds] Scanning {} users under: {}".format(len(users), scan_root))

    users_bounds = {}
    total_points = 0
    total_files = 0

    for user in users:
        user_dir = os.path.join(scan_root, user)
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        n_points = 0
        n_files = 0

        for name in list_session_csvs(user_dir):
            path = os.path.join(user_dir, name)
            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            if len(df) == 0:
                continue
            min_x = min(min_x, float(df["x"].min()))
            max_x = max(max_x, float(df["x"].max()))
            min_y = min(min_y, float(df["y"].min()))
            max_y = max(max_y, float(df["y"].max()))
            n_points += int(len(df))
            n_files += 1

        if n_points == 0:
            print("  [WARN] user {} has no points; skip".format(user))
            continue

        ub = _make_user_bounds(min_x, max_x, min_y, max_y, n_points, n_files)
        users_bounds[user] = ub
        total_points += n_points
        total_files += n_files
        print(
            "  scanned user: {} | min=({}, {}) max=({}, {}) diag={}".format(
                user, ub["min_x"], ub["min_y"], ub["max_x"], ub["max_y"], ub["diag"]
            )
        )

    if not users_bounds:
        raise RuntimeError("No valid points found under scan_root: {}".format(scan_root))

    # Fallback envelope = union of all users (for users missing in scan).
    g_min_x = min(u["min_x"] for u in users_bounds.values())
    g_max_x = max(u["max_x"] for u in users_bounds.values())
    g_min_y = min(u["min_y"] for u in users_bounds.values())
    g_max_y = max(u["max_y"] for u in users_bounds.values())
    fallback = _make_user_bounds(g_min_x, g_max_x, g_min_y, g_max_y)

    bounds = {
        "dataset": dataset,
        "scan_root": os.path.abspath(scan_root),
        "n_points": total_points,
        "n_files": total_files,
        "n_users": len(users_bounds),
        "users": users_bounds,
        "fallback": fallback,
    }
    return bounds


def save_bounds_json(bounds, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bounds, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("[bounds] Saved: {}".format(path))


def _normalize_loaded_bounds(bounds):
    """Ensure per-user format; reject old global-only JSON."""
    if "users" not in bounds or not isinstance(bounds["users"], dict) or not bounds["users"]:
        raise ValueError(
            "bounds json is not per-user format (missing non-empty 'users'). "
            "Re-run with --rescan_bounds."
        )
    for user, ub in bounds["users"].items():
        for key in ("min_x", "max_x", "min_y", "max_y"):
            if key not in ub:
                raise KeyError("user {} missing key: {}".format(user, key))
        if "corner_x" not in ub:
            ub["corner_x"] = ub["max_x"]
        if "corner_y" not in ub:
            ub["corner_y"] = ub["max_y"]
        if "diag" not in ub:
            ub["diag"] = _make_user_bounds(
                ub["min_x"], ub["max_x"], ub["min_y"], ub["max_y"]
            )["diag"]
    if "fallback" not in bounds:
        g_min_x = min(u["min_x"] for u in bounds["users"].values())
        g_max_x = max(u["max_x"] for u in bounds["users"].values())
        g_min_y = min(u["min_y"] for u in bounds["users"].values())
        g_max_y = max(u["max_y"] for u in bounds["users"].values())
        bounds["fallback"] = _make_user_bounds(g_min_x, g_max_x, g_min_y, g_max_y)
    return bounds


def load_bounds_json(path):
    with open(path, "r", encoding="utf-8") as f:
        bounds = json.load(f)
    bounds = _normalize_loaded_bounds(bounds)
    print("[bounds] Loaded: {} ({} users)".format(path, len(bounds["users"])))
    return bounds


def get_or_scan_bounds(dataset, scan_root, bounds_json, rescan=False):
    if (not rescan) and os.path.isfile(bounds_json):
        try:
            return load_bounds_json(bounds_json)
        except ValueError as e:
            print("[bounds] {}".format(e))
            print("[bounds] Re-scanning...")
    bounds = scan_dataset_xy_bounds(dataset, scan_root)
    save_bounds_json(bounds, bounds_json)
    return bounds


def get_user_bounds(bounds, user):
    """Return that user's min/max block; fallback if user absent from scan."""
    users = bounds.get("users", {})
    if user in users:
        return users[user]
    fb = bounds.get("fallback")
    print(
        "[WARN] user {} not in bounds scan; using fallback envelope {}".format(
            user, fb
        )
    )
    return fb


def compute_alpha(seq, user_bounds):
    """
    Mean distance to bottom-right corner (max_x, max_y),
    normalized by user bbox diagonal -> [0, 1].
    """
    coords = seq[:, :2].astype(np.float64)
    cx = float(user_bounds["max_x"])
    cy = float(user_bounds["max_y"])
    diag = max(float(user_bounds["diag"]), 1e-8)
    dist = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    return float(np.clip(float(dist.mean() / diag), 0.0, 1.0))


def compute_beta(seq, user_bounds):
    """
    Mean distance to top-right corner (max_x, min_y),
    normalized by user bbox diagonal -> [0, 1].
    """
    coords = seq[:, :2].astype(np.float64)
    cx = float(user_bounds["max_x"])
    cy = float(user_bounds["min_y"])
    diag = max(float(user_bounds["diag"]), 1e-8)
    dist = np.sqrt((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2)
    return float(np.clip(float(dist.mean() / diag), 0.0, 1.0))


def compute_loc_term(seq, user_bounds, use_beta=False):
    """
    Location scalar injected as γ * loc in bias formulas.
    use_beta=False: loc = α
    use_beta=True:  loc = α + β  (in [0, 2])
    Returns (loc, alpha, beta); beta is 0.0 when use_beta is False.
    """
    alpha = compute_alpha(seq, user_bounds)
    if use_beta:
        beta = compute_beta(seq, user_bounds)
        return alpha + beta, alpha, beta
    return alpha, alpha, 0.0


def compute_ab_pair_matrix(seq, user_bounds, return_parts=False):
    """
    Pixel-wise location matrix aligned with SRP (N×N), values in ~[0, 1].

    Per-event loc from BR/TR distances (see _ab_event_loc), then
      M_ij = 0.5 * (loc_i + loc_j)
    """
    loc, d_br, d_tr = _ab_event_loc(seq, user_bounds)
    alpha = 0.5 * (d_br[:, None] + d_br[None, :])
    beta = 0.5 * (d_tr[:, None] + d_tr[None, :])
    M = (0.5 * (loc[:, None] + loc[None, :])).astype(np.float32)
    if return_parts:
        return M, float(alpha.mean()), float(beta.mean())
    return M


def apply_bias_mix(img_u8, loc, gamma):
    """A: I' = (1-γ)I + γ·loc (scalar or HxW); clip to [0,1]."""
    I = img_u8.astype(np.float32) / 255.0
    g = float(gamma)
    loc_arr = np.asarray(loc, dtype=np.float32)
    out = (1.0 - g) * I + g * loc_arr
    out = np.clip(out, 0.0, 1.0)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def apply_bias_clip(img_u8, loc, gamma):
    """B: I' = clip(I + γ·loc, 0, 1) (scalar or HxW)."""
    I = img_u8.astype(np.float32) / 255.0
    loc_arr = np.asarray(loc, dtype=np.float32)
    out = I + float(gamma) * loc_arr
    out = np.clip(out, 0.0, 1.0)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def apply_bias_scale(img_u8, loc, gamma):
    """C: I' = (I + γ·loc) / (1+γ) (scalar or HxW); clip to [0,1]."""
    I = img_u8.astype(np.float32) / 255.0
    g = float(gamma)
    loc_arr = np.asarray(loc, dtype=np.float32)
    out = (I + g * loc_arr) / (1.0 + g)
    out = np.clip(out, 0.0, 1.0)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


BIAS_METHODS = {
    "mix": apply_bias_mix,
    "clip": apply_bias_clip,
    "scale": apply_bias_scale,
}


def _maybe_resize_gray(img_u8, output_size):
    if not output_size or int(output_size) <= 0:
        return img_u8
    s = int(output_size)
    pil = Image.fromarray(img_u8, mode="L")
    return np.asarray(_resize_transform(s)(pil), dtype=np.uint8)


def render_srp_with_bias(
    seq,
    epsilon,
    output_size,
    user_bounds,
    bias_fn,
    gamma,
    use_beta=False,
    use_matrix_ab=False,
):
    """
    Build native N×N SRP, inject location (scalar or pair matrix), then resize.
    Matrix mode must bias before resize so pixels stay aligned with event pairs.
    """
    if len(seq) < 2:
        return None

    img = render_srp(seq, epsilon, output_size=0)
    if img is None:
        return None

    if use_matrix_ab:
        loc = compute_ab_pair_matrix(seq, user_bounds)
    else:
        loc, _, _ = compute_loc_term(seq, user_bounds, use_beta=use_beta)

    img = bias_fn(img, loc, gamma)
    return _maybe_resize_gray(img, output_size)


def count_windows(dataset, data_root, chunk_size, five_fold=False):
    total = 0
    users = list_users(data_root)
    for user in users:
        user_dir = os.path.join(data_root, user)
        for file in list_session_csvs(user_dir):
            events = load_events(dataset, os.path.join(user_dir, file))
            for _, windows in session_window_groups(events, chunk_size, data_root, five_fold):
                total += len(windows)
    return total, users


def process_dataset_tensors(
    dataset,
    data_root,
    out_dir,
    sizes,
    epsilon,
    output_size,
    bounds,
    bias_fn,
    gamma,
    method_name,
    use_beta=False,
    use_matrix_ab=False,
):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print(
        "Bias method:", method_name,
        "| gamma:", gamma,
        "| use_beta:", use_beta,
        "| use_matrix_ab:", use_matrix_ab,
    )
    print("Per-user bounds loaded for", len(bounds.get("users", {})), "users.")
    print("\n[Phase] Generating SRP+uc bias tensors...")

    for chunk_size in sizes:
        H = int(output_size) if output_size and int(output_size) > 0 else chunk_size
        W = H
        total_samples, _ = count_windows(dataset, data_root, chunk_size)
        tensor_root = os.path.join(out_dir, "event{}".format(chunk_size))
        os.makedirs(tensor_root, exist_ok=True)

        print("\n[event{}] Total samples: {} | Tensor size: {}x{}".format(
            chunk_size, total_samples, H, W))

        images = np.memmap(
            os.path.join(tensor_root, "images.npy"),
            dtype=np.uint8,
            mode="w+",
            shape=(total_samples, 3, H, W),
        )
        labels = np.memmap(
            os.path.join(tensor_root, "labels.npy"),
            dtype=np.uint8,
            mode="w+",
            shape=(total_samples, num_users),
        )

        sessions = []
        alphas = []
        betas = []
        locs = []
        idx = 0

        for user in users:
            user_dir = os.path.join(data_root, user)
            user_bounds = get_user_bounds(bounds, user)
            print("\n------------------------------")
            print(
                "User:", user,
                "| BR=({}, {}) TR=({}, {}) diag={}".format(
                    user_bounds["max_x"], user_bounds["max_y"],
                    user_bounds["max_x"], user_bounds["min_y"],
                    user_bounds["diag"],
                ),
            )

            for file in list_session_csvs(user_dir):
                path = os.path.join(user_dir, file)
                session = os.path.splitext(file)[0]
                events = load_events(dataset, path)
                windows = generate_windows(events, chunk_size, data_root)
                print("  Session {} | chunk={} -> {} windows".format(
                    session, chunk_size, len(windows)))

                for seq in windows:
                    if len(seq) < 2:
                        continue
                    img = render_srp(seq, epsilon, output_size=0)
                    if img is None:
                        continue

                    if use_matrix_ab:
                        loc_m, alpha, beta = compute_ab_pair_matrix(
                            seq, user_bounds, return_parts=True
                        )
                        loc = float(loc_m.mean())
                        img = bias_fn(img, loc_m, gamma)
                    else:
                        loc, alpha, beta = compute_loc_term(
                            seq, user_bounds, use_beta=use_beta
                        )
                        img = bias_fn(img, loc, gamma)

                    img = _maybe_resize_gray(img, output_size)
                    if img.shape[:2] != (H, W):
                        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

                    images[idx] = gray_to_tensor_chw(img)
                    y = np.zeros(num_users, dtype=np.uint8)
                    y[user_to_idx[user]] = 1
                    labels[idx] = y
                    sessions.append(session)
                    alphas.append(alpha)
                    betas.append(beta)
                    locs.append(loc)
                    idx += 1

        images.flush()
        labels.flush()
        np.save(os.path.join(tensor_root, "sessions.npy"), np.array(sessions, dtype=object))
        np.save(os.path.join(tensor_root, "alphas.npy"), np.asarray(alphas, dtype=np.float32))
        np.save(os.path.join(tensor_root, "locs.npy"), np.asarray(locs, dtype=np.float32))
        if use_beta or use_matrix_ab:
            np.save(os.path.join(tensor_root, "betas.npy"), np.asarray(betas, dtype=np.float32))
        print("\nTensor dataset saved to: {} (wrote {} samples)".format(tensor_root, idx))


def process_dataset(
    dataset,
    data_root,
    out_dir,
    sizes,
    epsilon,
    output_size,
    bounds,
    bias_fn,
    gamma,
    method_name,
    tensors=False,
    use_beta=False,
    use_matrix_ab=False,
):
    if tensors:
        process_dataset_tensors(
            dataset, data_root, out_dir, sizes, epsilon, output_size,
            bounds, bias_fn, gamma, method_name,
            use_beta=use_beta, use_matrix_ab=use_matrix_ab,
        )
        return

    users = list_users(data_root)
    print("\nDataset:", dataset)
    print("Users:", len(users))
    print(
        "Bias method:", method_name,
        "| gamma:", gamma,
        "| use_beta:", use_beta,
        "| use_matrix_ab:", use_matrix_ab,
    )
    print("Per-user bounds loaded for", len(bounds.get("users", {})), "users.")
    print("\n[Phase] Generating SRP+uc bias PNGs...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        user_bounds = get_user_bounds(bounds, user)
        print("\n------------------------------")
        print(
            "User:", user,
            "| BR=({}, {}) TR=({}, {}) diag={}".format(
                user_bounds["max_x"], user_bounds["max_y"],
                user_bounds["max_x"], user_bounds["min_y"],
                user_bounds["diag"],
            ),
        )

        for file in list_session_csvs(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            events = load_events(dataset, path)

            for chunk_size in sizes:
                windows = generate_windows(events, chunk_size, data_root)
                print("  Session {} | chunk={} -> {} windows".format(
                    session, chunk_size, len(windows)))

                for i, seq in enumerate(windows):
                    img = render_srp_with_bias(
                        seq, epsilon, output_size, user_bounds, bias_fn, gamma,
                        use_beta=use_beta, use_matrix_ab=use_matrix_ab,
                    )
                    if img is None:
                        continue
                    save_path = os.path.join(
                        out_dir,
                        "event{}".format(chunk_size),
                        user,
                        "{}-{}.png".format(session, i),
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, img)


def build_argparser(method_name, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl", "twos"])
    parser.add_argument(
        "--data_root",
        required=True,
        help="要画图的数据根目录（training 或 testing 均可）；不影响 min/max 统计。",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[125])
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="若 > 0，Resize 为 output_size×output_size；0 表示保持 N×N。",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.2,
        help="位置偏置强度 γ（默认 0.2）。",
    )
    parser.add_argument(
        "--scan_root",
        default=None,
        help="仅用于扫描 per-user min/max；默认永远是该 dataset 的 training_files。"
             "生成 testing 时不要改成 testing。",
    )
    parser.add_argument(
        "--bounds_json",
        default=None,
        help="per-user min/max 缓存；默认 RP_uc/bounds/<dataset>_xy_bounds.json。"
             "train/test 生成都读同一份（由 training 扫出）。",
    )
    parser.add_argument(
        "--rescan_bounds",
        action="store_true",
        default=False,
        help="强制用 scan_root（默认 training）重扫并覆盖 JSON；生成 testing 时不要开。",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="输出 images.npy / labels.npy / sessions.npy / alphas.npy。",
    )
    parser.set_defaults(method_name=method_name)
    return parser


def run_main(method_name, description, use_beta=False, use_matrix_ab=False):
    if method_name not in BIAS_METHODS:
        raise ValueError(method_name)

    parser = build_argparser(method_name, description)
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    scan_rel = args.scan_root or DEFAULT_SCAN_ROOT[args.dataset]
    scan_root = resolve_path(scan_rel)
    bounds_json = resolve_path(args.bounds_json) if args.bounds_json else default_bounds_json(args.dataset)

    print(
        "Method:", method_name,
        "| use_beta:", use_beta,
        "| use_matrix_ab:", use_matrix_ab,
    )
    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("Resolved scan_root:", scan_root)
    print("Bounds JSON:", bounds_json)

    bounds = get_or_scan_bounds(
        dataset=args.dataset,
        scan_root=scan_root,
        bounds_json=bounds_json,
        rescan=args.rescan_bounds,
    )
    print("\nPer-user XY bounds:")
    for u in sorted(bounds["users"].keys(), key=natural_key):
        ub = bounds["users"][u]
        print(
            "  ", u, "->",
            "min=({}, {})".format(ub["min_x"], ub["min_y"]),
            "max=({}, {})".format(ub["max_x"], ub["max_y"]),
            "diag={}".format(ub["diag"]),
        )

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        epsilon=args.epsilon,
        output_size=args.output_size,
        bounds=bounds,
        bias_fn=BIAS_METHODS[method_name],
        gamma=args.gamma,
        method_name=method_name,
        tensors=args.tensors,
        use_beta=use_beta,
        use_matrix_ab=use_matrix_ab,
    )
    print("\nDone.")
