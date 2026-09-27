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
# Chunking (same as XYPlot_chunk)
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


def session_sequence_groups(events, chunk_size, five_fold):
    if five_fold:
        return list(iter_session_fold_sequences(events, chunk_size))
    return [(None, sequences_from_events(events, chunk_size))]


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
    return sequences_from_events(load_events(dataset, path), chunk_size)


def count_samples(dataset, data_root, chunk_size, five_fold=False):
    total = 0
    for user in list_users(data_root):
        user_dir = os.path.join(data_root, user)
        for file in list_session_files(user_dir):
            events = load_events(dataset, os.path.join(user_dir, file))
            for _, sequences in session_sequence_groups(events, chunk_size, five_fold):
                total += len(sequences)
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
    dataset, data_root, out_dir, user_max_xy, user_min_xy, chunk_size,
    resume=False, five_fold=False,
):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Chunk size:", chunk_size)
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Rendering: per-user screen coords (same as XYPlot_per_user) |", TARGET_SIZE, "x", TARGET_SIZE)
    if five_fold:
        print("\n[Phase] Generating chunk + per-user XYPlot tensors, {} contiguous folds per session...".format(N_FOLDS))
    else:
        print("\n[Phase] Generating chunk + per-user XYPlot tensors...")

    total_samples = count_samples(dataset, data_root, chunk_size, five_fold=five_fold)
    tensor_root = os.path.join(out_dir, TENSOR_SUBDIR)
    os.makedirs(tensor_root, exist_ok=True)

    H = W = int(TARGET_SIZE)
    print(f"\n[{TENSOR_SUBDIR}] Total samples: {total_samples} | Tensor size: {H}x{W}")

    img_path = os.path.join(tensor_root, "images.npy")
    lab_path = os.path.join(tensor_root, "labels.npy")
    fold_path = os.path.join(tensor_root, "folds.npy")
    mmap_shape_img = (total_samples, 3, H, W)
    mmap_shape_lab = (total_samples, num_users)

    if resume:
        if not (os.path.isfile(img_path) and os.path.isfile(lab_path)):
            raise FileNotFoundError(
                "resume requested but images.npy/labels.npy missing under " + tensor_root
            )
        if five_fold and not os.path.isfile(fold_path):
            raise FileNotFoundError(
                "resume requested but folds.npy missing under " + tensor_root
            )
        images = np.memmap(img_path, dtype=np.uint8, mode="r+", shape=mmap_shape_img)
        labels = np.memmap(lab_path, dtype=np.uint8, mode="r+", shape=mmap_shape_lab)
        folds = (
            np.memmap(fold_path, dtype=np.uint8, mode="r+", shape=(total_samples,))
            if five_fold else None
        )
        start_idx = _written_count(labels)
        print("[resume] already written:", start_idx, "/", total_samples)
    else:
        if os.path.isfile(img_path):
            raise FileExistsError(
                img_path + " already exists. Re-run with --resume, do not overwrite."
            )
        images = np.memmap(img_path, dtype=np.uint8, mode="w+", shape=mmap_shape_img)
        labels = np.memmap(lab_path, dtype=np.uint8, mode="w+", shape=mmap_shape_lab)
        folds = (
            np.memmap(fold_path, dtype=np.uint8, mode="w+", shape=(total_samples,))
            if five_fold else None
        )
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
            events = load_events(dataset, path)
            groups = session_sequence_groups(events, chunk_size, five_fold)
            n_chunks = sum(len(sequences) for _, sequences in groups)
            print(f"   Session: {session} -> {n_chunks} chunks")

            for fold, sequences in groups:
                for seq in sequences:
                    if idx < start_idx:
                        sessions.append(session)
                        if folds is not None:
                            folds[idx] = fold
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
                    if folds is not None:
                        folds[idx] = fold
                    idx += 1

    images.flush()
    labels.flush()
    if folds is not None:
        folds.flush()
    np.save(
        os.path.join(tensor_root, "sessions.npy"),
        np.array(sessions, dtype=object),
    )
    print(f"\nTensor dataset saved to: {tensor_root} (wrote {idx} samples)")


def process_dataset(
    dataset, data_root, out_dir, user_max_xy, user_min_xy, chunk_size,
    tensors=False, resume=False, five_fold=False,
):
    if tensors:
        process_dataset_tensors(
            dataset, data_root, out_dir, user_max_xy, user_min_xy, chunk_size,
            resume=resume, five_fold=five_fold,
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

            events = load_events(dataset, path)

            print("      Events:", len(events))
            groups = session_sequence_groups(events, chunk_size, five_fold)
            print("      Chunks:", sum(len(sequences) for _, sequences in groups))

            for fold, sequences in groups:
                for i, seq in enumerate(sequences):
                    parts = [out_dir]
                    if fold is not None:
                        parts.append("fold%d" % fold)
                    parts.extend([TENSOR_SUBDIR, user, f"{session}-{i}.png"])
                    draw_sequence(
                        _shift_seq(seq, min_x, min_y),
                        os.path.join(*parts),
                        canvas_w,
                        canvas_h,
                    )


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
    parser.add_argument(
        "--five-fold",
        action="store_true",
        default=False,
        help="每个 session 按事件顺序切成 5 段连续事件再切 chunk。tensors 时多写 folds.npy，取值 0–4。",
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
    skipped = sorted(
        [
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name)) and is_skipped_user_dir(name)
        ],
        key=natural_key,
    )
    if skipped:
        print("Excluded dirs:", ", ".join(skipped))

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
        five_fold=args.five_fold,
    )
    print("\nChunk + per-user XYPlot generation finished.")


if __name__ == "__main__":
    main()
