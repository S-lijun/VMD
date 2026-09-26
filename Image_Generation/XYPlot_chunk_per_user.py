# -*- coding: utf-8 -*-
"""
Chunked XYPlot with per-user screen-coordinate drawing (not centered).

Windowing: fixed-size event chunks (same as XYPlot_chunk.py, default 125).
Drawing: per-user screen projection (training max_x / max_y), after shifting
  each point by that user's training min_x / min_y. If min is 0 this matches
  the old x/max_x mapping.
Bounds JSON shared with XYPlot_per_user.py under ChongSOTA/bounds/ (max only).
"""

import os
import re
import argparse

import cv2
import numpy as np
import pandas as pd

from XYPlot import (
    ROOT,
    TARGET_SIZE,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
    clean_twos,
    draw_sequence,
    render_sequence,
)
from XYPlot_per_user import (  # noqa: E402
    DEFAULT_TRAINING_ROOT,
    _norm_bounds,
    default_bounds_json,
    get_or_scan_user_max_xy,
)

TENSOR_SUBDIR = "Chong_chunk_per_user"


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
    if dataset == "twos":
        return clean_twos(df)
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


# ============================================================
# Chunking (same as XYPlot_chunk)
# ============================================================

def split_by_chunk_size(events, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]


def scan_user_min_xy(dataset, training_root):
    """Per-user min x/y from training_root (same cleaning as drawing)."""
    user_min_xy = {}
    print("\n[min] Scanning users under:", training_root)
    for user in list_users(training_root):
        user_dir = os.path.join(training_root, user)
        min_x = float("inf")
        min_y = float("inf")
        saw = False
        for name in list_session_files(user_dir):
            path = os.path.join(user_dir, name)
            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            if len(df) == 0:
                continue
            min_x = min(min_x, float(df["x"].min()))
            min_y = min(min_y, float(df["y"].min()))
            saw = True
        if saw:
            user_min_xy[user] = (min_x, min_y)
        else:
            user_min_xy[user] = (0.0, 0.0)
        print("  scanned user:", user, "-> min", user_min_xy[user])
    return user_min_xy


def _user_min(user, user_min_xy):
    if user in user_min_xy:
        return user_min_xy[user]
    print("\n[WARN] User", user, "not in training min scan; using min=0")
    return 0.0, 0.0


def _shift_seq(seq, min_x, min_y):
    out = []
    for e in seq:
        d = dict(e)
        d["x"] = float(e["x"]) - min_x
        d["y"] = float(e["y"]) - min_y
        out.append(d)
    return out


def _session_sequences(dataset, path, chunk_size):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)

    events = df.to_dict("records")
    if len(events) < 2:
        return []

    sequences = split_by_chunk_size(events, chunk_size)
    return [seq for seq in sequences if len(seq) >= 2]


def count_samples(dataset, data_root, chunk_size):
    total = 0
    for user in list_users(data_root):
        user_dir = os.path.join(data_root, user)
        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            total += len(_session_sequences(dataset, path, chunk_size))
    return total


def bgr_to_tensor_chw(img):
    """Match Images_convert.py: BGR HWC -> (3, H, W) uint8."""
    return img.transpose(2, 0, 1)


# ============================================================
# Dataset Processing
# ============================================================

def _written_count(labels):
    row_sum = np.asarray(labels).sum(axis=1)
    nz = np.flatnonzero(row_sum)
    return int(nz[-1]) + 1 if len(nz) else 0


def process_dataset_tensors(
    dataset, data_root, out_dir, user_max_xy, user_min_xy, chunk_size, resume=False
):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Chunk size:", chunk_size)
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Rendering: per-user screen coords (same as XYPlot_per_user) |", TARGET_SIZE, "x", TARGET_SIZE)
    print("\n[Phase] Generating chunk + per-user XYPlot tensors...")

    total_samples = count_samples(dataset, data_root, chunk_size)
    tensor_root = os.path.join(out_dir, TENSOR_SUBDIR)
    os.makedirs(tensor_root, exist_ok=True)

    H = W = int(TARGET_SIZE)
    print(f"\n[{TENSOR_SUBDIR}] Total samples: {total_samples} | Tensor size: {H}x{W}")

    img_path = os.path.join(tensor_root, "images.npy")
    lab_path = os.path.join(tensor_root, "labels.npy")
    mmap_shape_img = (total_samples, 3, H, W)
    mmap_shape_lab = (total_samples, num_users)

    if resume:
        if not (os.path.isfile(img_path) and os.path.isfile(lab_path)):
            raise FileNotFoundError(
                "resume requested but images.npy/labels.npy missing under " + tensor_root
            )
        images = np.memmap(img_path, dtype=np.uint8, mode="r+", shape=mmap_shape_img)
        labels = np.memmap(lab_path, dtype=np.uint8, mode="r+", shape=mmap_shape_lab)
        start_idx = _written_count(labels)
        print("[resume] already written:", start_idx, "/", total_samples)
    else:
        if os.path.isfile(img_path):
            raise FileExistsError(
                img_path + " already exists. Re-run with --resume, do not overwrite."
            )
        images = np.memmap(img_path, dtype=np.uint8, mode="w+", shape=mmap_shape_img)
        labels = np.memmap(lab_path, dtype=np.uint8, mode="w+", shape=mmap_shape_lab)
        start_idx = 0

    sessions = []
    idx = 0
    catching_up = start_idx > 0

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)
        min_x, min_y = _user_min(user, user_min_xy)
        canvas_w = max(float(norm_x) - min_x, 1.0)
        canvas_h = max(float(norm_y) - min_y, 1.0)

        print("\n------------------------------")
        print("User:", user, "| min=({}, {}) max=({}, {})".format(
            min_x, min_y, norm_x, norm_y))

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            sequences = _session_sequences(dataset, path, chunk_size)
            print(f"   Session: {session} -> {len(sequences)} chunks")

            for seq in sequences:
                if idx < start_idx:
                    sessions.append(session)
                    idx += 1
                    continue

                if catching_up:
                    print("[resume] rendering from idx", idx, "user", user, "session", session)
                    catching_up = False

                img = render_sequence(_shift_seq(seq, min_x, min_y), canvas_w, canvas_h)
                if img is None:
                    continue

                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

                images[idx] = bgr_to_tensor_chw(img)
                y = np.zeros(num_users, dtype=np.uint8)
                y[user_to_idx[user]] = 1
                labels[idx] = y
                sessions.append(session)
                idx += 1

    images.flush()
    labels.flush()
    np.save(
        os.path.join(tensor_root, "sessions.npy"),
        np.array(sessions, dtype=object),
    )
    print(f"\nTensor dataset saved to: {tensor_root} (wrote {idx} samples)")


def process_dataset(
    dataset, data_root, out_dir, user_max_xy, user_min_xy, chunk_size,
    tensors=False, resume=False,
):
    if tensors:
        process_dataset_tensors(
            dataset, data_root, out_dir, user_max_xy, user_min_xy, chunk_size,
            resume=resume,
        )
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Chunk size:", chunk_size)
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Rendering: per-user screen coords (same as XYPlot_per_user) |", TARGET_SIZE, "x", TARGET_SIZE)

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)
        min_x, min_y = _user_min(user, user_min_xy)
        canvas_w = max(float(norm_x) - min_x, 1.0)
        canvas_h = max(float(norm_y) - min_y, 1.0)

        print("\n------------------------------")
        print("User:", user, "| min=({}, {}) max=({}, {})".format(
            min_x, min_y, norm_x, norm_y))

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
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
                if len(seq) < 2:
                    continue
                save_path = os.path.join(
                    out_dir,
                    TENSOR_SUBDIR,
                    user,
                    f"{session}-{i}.png",
                )
                draw_sequence(_shift_seq(seq, min_x, min_y), save_path, canvas_w, canvas_h)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "XYPlot: fixed chunk windows (same as XYPlot_chunk) + "
            "per-user screen draw (same as XYPlot_per_user)."
        ),
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl", "twos"])
    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset (Balabit/ChaoShen/DFL training_files)."
             " Only used when scanning/resaving bounds JSON.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Sessions to render (train or test), relative to ROOT.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        default=125,
        help="Number of events per chunk.",
    )
    parser.add_argument(
        "--bounds_json",
        default=None,
        help="Per-user max_x/max_y cache; default ChongSOTA/bounds/<dataset>_xy_bounds.json "
             "(shared with XYPlot_per_user.py).",
    )
    parser.add_argument(
        "--rescan_bounds",
        action="store_true",
        default=False,
        help="Force rescan training_root and overwrite bounds JSON.",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="Output images.npy / labels.npy / sessions.npy instead of PNG.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Continue an interrupted --tensors run without wiping images.npy.",
    )
    args = parser.parse_args()

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = resolve_path(training_rel)
    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    bounds_json = (
        resolve_path(args.bounds_json) if args.bounds_json
        else default_bounds_json(args.dataset)
    )

    print("[training_root]", training_root)
    print("[data_root]", data_root)
    print("[out_dir]", out_dir)
    print("Bounds JSON:", bounds_json)
    print("Chunk size:", args.sizes)

    user_max_xy = get_or_scan_user_max_xy(
        dataset=args.dataset,
        training_root=training_root,
        bounds_json=bounds_json,
        rescan=args.rescan_bounds,
    )
    user_min_xy = scan_user_min_xy(args.dataset, training_root)

    print("\nUSER_MAX_XY:")
    for u in sorted(user_max_xy.keys(), key=natural_key):
        print("  ", u, "->", user_max_xy[u])

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        user_max_xy,
        user_min_xy,
        args.sizes,
        tensors=args.tensors,
        resume=args.resume,
    )
    print("\nChunk + per-user XYPlot generation finished.")


if __name__ == "__main__":
    main()
