# -*- coding: utf-8 -*-
"""
Chunked centered XYPlot: fixed-size event windows + per-sequence bbox fit/center.

Drawing matches XYPlot_centered (no screen-width / per-user normalization).
Segmentation is fixed chunk_size (default 125), not time-diff split + merge.
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
    INNER_PADDING,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
    clean_twos,
)

TENSOR_SUBDIR = "Chong"
N_FOLDS = 5


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


def list_session_files(user_dir):
    return sorted(
        [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))],
        key=natural_key,
    )


# ============================================================
# Chunking
# ============================================================

def split_by_chunk_size(events, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return [events[i:i + chunk_size] for i in range(0, len(events), chunk_size)]


def contiguous_fold_bounds(n_events, n_folds=N_FOLDS):
    """把一个 session 的事件按顺序切成 n_folds 段，段与段首尾相接、不重叠。"""
    return [
        (i * n_events // n_folds, (i + 1) * n_events // n_folds)
        for i in range(n_folds)
    ]


# ============================================================
# Centered draw (same as XYPlot_centered)
# ============================================================

def render_sequence_centered(seq):
    """
    Fit trajectory bbox into TARGET_SIZE×TARGET_SIZE with uniform scale, centered.
    """
    if len(seq) < 2:
        return None

    img_size = int(TARGET_SIZE)
    effective_size = max(1, img_size - 2 * INNER_PADDING)

    xs = np.array([float(e["x"]) for e in seq], dtype=np.float64)
    ys = np.array([float(e["y"]) for e in seq], dtype=np.float64)

    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)

    pad_x = range_x * 0.05
    pad_y = range_y * 0.05

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    range_x = max_x - min_x
    range_y = max_y - min_y

    scale = min(effective_size / range_x, effective_size / range_y)
    offset_x = (img_size - range_x * scale) / 2
    offset_y = (img_size - range_y * scale) / 2

    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

    prev = None
    for x, y in zip(xs, ys):
        x_s = int(np.clip((x - min_x) * scale + offset_x, 0, img_size - 1))
        y_s = int(np.clip((y - min_y) * scale + offset_y, 0, img_size - 1))

        if prev is not None:
            cv2.line(
                canvas,
                prev,
                (x_s, y_s),
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

        prev = (x_s, y_s)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_sequence_centered(seq, save_path):
    final = render_sequence_centered(seq)
    if final is None:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, final)


def load_events(dataset, path):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)
    return df.to_dict("records")


def sequences_from_events(events, chunk_size):
    if len(events) < 2:
        return []
    return [seq for seq in split_by_chunk_size(events, chunk_size) if len(seq) >= 2]


def iter_session_fold_sequences(events, chunk_size, n_folds=N_FOLDS):
    """每一折只在自己那段连续事件里切 chunk，chunk 不会跨到别的折。"""
    for fold, (start, end) in enumerate(contiguous_fold_bounds(len(events), n_folds)):
        yield fold, sequences_from_events(events[start:end], chunk_size)


def _session_sequences(dataset, path, chunk_size):
    return sequences_from_events(load_events(dataset, path), chunk_size)


def count_samples(dataset, data_root, chunk_size, five_fold=False):
    total = 0
    for user in list_users(data_root):
        user_dir = os.path.join(data_root, user)
        for file in list_session_files(user_dir):
            events = load_events(dataset, os.path.join(user_dir, file))
            if five_fold:
                for _, sequences in iter_session_fold_sequences(events, chunk_size):
                    total += len(sequences)
            else:
                total += len(sequences_from_events(events, chunk_size))
    return total


def bgr_to_tensor_chw(img):
    return img.transpose(2, 0, 1)


# ============================================================
# Dataset Processing
# ============================================================

def _write_chunk(images, labels, sessions, folds, idx, seq, user_idx, num_users, session, fold):
    img = render_sequence_centered(seq)
    if img is None:
        return idx

    images[idx] = bgr_to_tensor_chw(img)
    y = np.zeros(num_users, dtype=np.uint8)
    y[user_idx] = 1
    labels[idx] = y
    sessions.append(session)
    if folds is not None:
        folds[idx] = fold
    return idx + 1


def process_dataset_tensors(dataset, data_root, out_dir, chunk_size, five_fold=False):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Chunk size:", chunk_size)
    print("Rendering: per-sequence bbox fit, centered in", TARGET_SIZE, "x", TARGET_SIZE)
    if five_fold:
        print(f"\n[Phase] Generating centered chunk XYPlot tensors, {N_FOLDS} contiguous folds per session...")
    else:
        print("\n[Phase] Generating centered chunk XYPlot tensors...")

    total_samples = count_samples(dataset, data_root, chunk_size, five_fold=five_fold)
    tensor_root = os.path.join(out_dir, TENSOR_SUBDIR)
    os.makedirs(tensor_root, exist_ok=True)

    H = W = int(TARGET_SIZE)
    print(f"\n[{TENSOR_SUBDIR}] Total samples: {total_samples} | Tensor size: {H}x{W}")

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
    folds = None
    if five_fold:
        folds = np.memmap(
            os.path.join(tensor_root, "folds.npy"),
            dtype=np.uint8,
            mode="w+",
            shape=(total_samples,),
        )

    sessions = []
    idx = 0

    for user in users:
        user_dir = os.path.join(data_root, user)

        print("\n------------------------------")
        print("User:", user)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            events = load_events(dataset, path)
            if five_fold:
                fold_sequences = list(iter_session_fold_sequences(events, chunk_size))
                n_chunks = sum(len(sequences) for _, sequences in fold_sequences)
                print(f"   Session: {session} -> {n_chunks} chunks")
                for fold, sequences in fold_sequences:
                    for seq in sequences:
                        idx = _write_chunk(
                            images, labels, sessions, folds, idx, seq,
                            user_to_idx[user], num_users, session, fold,
                        )
            else:
                sequences = sequences_from_events(events, chunk_size)
                print(f"   Session: {session} -> {len(sequences)} chunks")
                for seq in sequences:
                    idx = _write_chunk(
                        images, labels, sessions, folds, idx, seq,
                        user_to_idx[user], num_users, session, 0,
                    )

    images.flush()
    labels.flush()
    if folds is not None:
        folds.flush()
    np.save(
        os.path.join(tensor_root, "sessions.npy"),
        np.array(sessions, dtype=object),
    )
    print(f"\nTensor dataset saved to: {tensor_root}")


def process_dataset(dataset, data_root, out_dir, chunk_size, tensors=False, five_fold=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir, chunk_size, five_fold=five_fold)
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Chunk size:", chunk_size)
    print("Rendering: per-sequence bbox fit, centered in", TARGET_SIZE, "x", TARGET_SIZE)

    for user in users:
        user_dir = os.path.join(data_root, user)

        print("\n------------------------------")
        print("User:", user)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            print("   Session:", session)

            events = load_events(dataset, path)

            print("      Events:", len(events))
            if five_fold:
                groups = list(iter_session_fold_sequences(events, chunk_size))
            else:
                groups = [(None, sequences_from_events(events, chunk_size))]
            print("      Chunks:", sum(len(sequences) for _, sequences in groups))

            for fold, sequences in groups:
                for i, seq in enumerate(sequences):
                    parts = [out_dir]
                    if fold is not None:
                        parts.append("fold%d" % fold)
                    parts.extend([TENSOR_SUBDIR, user, f"{session}-{i}.png"])
                    draw_sequence_centered(seq, os.path.join(*parts))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl", "twos"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        default=125,
        help="Number of events per chunk.",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="Output images.npy / labels.npy / sessions.npy instead of PNG.",
    )
    parser.add_argument(
        "--five-fold",
        action="store_true",
        default=False,
        help="每个 session 按事件顺序切成 5 段连续事件再切 chunk。tensors 时多写 folds.npy，取值 0–4。",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)

    print("[data_root]", data_root)
    print("[out_dir]", out_dir)
    skipped = sorted(
        [
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name)) and is_skipped_user_dir(name)
        ],
        key=natural_key,
    )
    if skipped:
        print("Excluded dirs:", ", ".join(skipped))

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        tensors=args.tensors,
        five_fold=args.five_fold,
    )
    print("\nCentered chunk XYPlot generation finished.")


if __name__ == "__main__":
    main()
