# -*- coding: utf-8 -*-
"""
Per-user normalization: max x / max y from all sessions of that user under
training_files only (default path per --dataset). The same bounds are used
for any data_root (e.g. testing_files). Bounds are cached to
ChongSOTA/bounds/<dataset>_xy_bounds.json on first scan unless --rescan_bounds.
"""

import os
import re
import json
import argparse

import cv2
import numpy as np
import pandas as pd

from XYPlot import (
    ROOT,
    TARGET_SIZE,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
    clean_twos,
    draw_sequence,
    render_sequence,
)

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
    "twos": "Data/TWOS/training_files",
}

TENSOR_SUBDIR = "Chong_per_user"
BOUNDS_DIR = os.path.join(os.path.dirname(__file__), "bounds")


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


def _norm_bounds(user, user_max_xy):
    if user in user_max_xy:
        return user_max_xy[user]
    print(
        "\n[WARN] User", user, "not in training scan; using GLOBAL_MAX_X/Y:",
        GLOBAL_MAX_X, GLOBAL_MAX_Y,
    )
    return GLOBAL_MAX_X, GLOBAL_MAX_Y


def _session_sequences(dataset, path, norm_x, norm_y):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)

    events = df.to_dict("records")
    if len(events) < 2:
        return []

    sequences = split_by_time(events)
    sequences = merge_sequences(sequences, norm_x)
    return [seq for seq in sequences if len(seq) >= 2]


def default_bounds_json(dataset):
    return os.path.join(BOUNDS_DIR, "{}_xy_bounds.json".format(dataset))


def build_user_max_xy_from_training(dataset, training_root):
    """
    Scan every session under training_root/<user>/ and record each user's
    global max x and max y (after the same cleaning as drawing).
    Returns dict user -> (max_x, max_y).
    """
    user_max_xy = {}

    print("\n[bounds] Scanning users under:", training_root)
    for user in list_users(training_root):
        user_dir = os.path.join(training_root, user)

        max_x = 0.0
        max_y = 0.0
        saw_points = False

        for name in list_session_files(user_dir):
            path = os.path.join(user_dir, name)

            df = pd.read_csv(path)
            df = _clean_df(dataset, df)
            if len(df) == 0:
                continue

            max_x = max(max_x, float(df["x"].max()))
            max_y = max(max_y, float(df["y"].max()))
            saw_points = True

        if saw_points:
            user_max_xy[user] = (max_x, max_y)
        else:
            user_max_xy[user] = (GLOBAL_MAX_X, GLOBAL_MAX_Y)
        print("  scanned user:", user, "->", user_max_xy[user])

    return user_max_xy


def save_bounds_json(user_max_xy, dataset, scan_root, path):
    payload = {
        "dataset": dataset,
        "scan_root": os.path.abspath(scan_root),
        "n_users": len(user_max_xy),
        "users": {},
    }
    for user, (max_x, max_y) in user_max_xy.items():
        payload["users"][user] = {
            "max_x": float(max_x),
            "max_y": float(max_y),
        }
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("[bounds] Saved:", path)


def load_bounds_json(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "users" not in payload or not payload["users"]:
        raise ValueError(
            "bounds json missing non-empty 'users'; re-run with --rescan_bounds"
        )
    user_max_xy = {}
    for user, ub in payload["users"].items():
        if "max_x" not in ub or "max_y" not in ub:
            raise KeyError("user {} missing max_x/max_y".format(user))
        user_max_xy[user] = (float(ub["max_x"]), float(ub["max_y"]))
    print("[bounds] Loaded: {} ({} users)".format(path, len(user_max_xy)))
    return user_max_xy


def get_or_scan_user_max_xy(dataset, training_root, bounds_json, rescan=False):
    if (not rescan) and os.path.isfile(bounds_json):
        return load_bounds_json(bounds_json)
    user_max_xy = build_user_max_xy_from_training(dataset, training_root)
    save_bounds_json(user_max_xy, dataset, training_root, bounds_json)
    return user_max_xy


def count_samples(dataset, data_root, user_max_xy):
    total = 0
    users = list_users(data_root)

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            total += len(_session_sequences(dataset, path, norm_x, norm_y))

    return total, users


def bgr_to_tensor_chw(img):
    """Match Images_convert.py: BGR HWC -> (3, H, W) uint8."""
    return img.transpose(2, 0, 1)


def process_dataset_tensors(dataset, data_root, out_dir, user_max_xy):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("\n[Phase] Generating per-user XYPlot tensors (Images_convert format)...")

    total_samples, _ = count_samples(dataset, data_root, user_max_xy)
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

    sessions = []
    idx = 0

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)

        print("\n------------------------------")
        print("User:", user, "| norm W×H (from training):", norm_x, norm_y)

        for file in list_session_files(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            sequences = _session_sequences(dataset, path, norm_x, norm_y)
            print(f"   Session: {session} -> {len(sequences)} sequences")

            for seq in sequences:
                img = render_sequence(seq, norm_x, norm_y)
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

    print(f"\nTensor dataset saved to: {tensor_root}")


def process_dataset(dataset, data_root, out_dir, user_max_xy, tensors=False):
    if tensors:
        process_dataset_tensors(dataset, data_root, out_dir, user_max_xy)
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users in data_root:", len(users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")

    for user in users:
        user_dir = os.path.join(data_root, user)
        norm_x, norm_y = _norm_bounds(user, user_max_xy)

        print("\n------------------------------")
        print("User:", user, "| norm W×H (from training):", norm_x, norm_y)

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

            sequences = split_by_time(events)
            print("      After split:", len(sequences))

            sequences = merge_sequences(sequences, norm_x)
            print("      After merge:", len(sequences))

            for i, seq in enumerate(sequences):
                save_path = os.path.join(
                    out_dir,
                    TENSOR_SUBDIR,
                    user,
                    f"{session}-{i}.png",
                )
                draw_sequence(seq, save_path, norm_x, norm_y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
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
        "--bounds_json",
        default=None,
        help="Per-user max_x/max_y cache; default ChongSOTA/bounds/<dataset>_xy_bounds.json.",
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
        help="直接输出 Images_convert 格式的 tensors（images.npy / labels.npy / sessions.npy），不写 PNG。",
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
    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("Bounds JSON:", bounds_json)

    user_max_xy = get_or_scan_user_max_xy(
        dataset=args.dataset,
        training_root=training_root,
        bounds_json=bounds_json,
        rescan=args.rescan_bounds,
    )

    print("\nUSER_MAX_XY:")
    for u in sorted(user_max_xy.keys(), key=natural_key):
        print("  ", u, "->", user_max_xy[u])

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        user_max_xy,
        tensors=args.tensors,
    )
    print("\nPer-user XYPlot generation finished.")


if __name__ == "__main__":
    main()
