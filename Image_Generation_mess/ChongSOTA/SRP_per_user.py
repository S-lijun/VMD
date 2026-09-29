# -*- coding: utf-8 -*-
"""
Per-user merge threshold for SRP:
- Build each user's max x / max y from training_root only.
- Use per-user max x as min_length when merging split sequences.
- Drawing remains SRP (pair-wise recurrence plot) via draw_srp.
"""

import os
import argparse

import pandas as pd
import numpy as np

from SRP import (
    ROOT,
    split_by_time,
    merge_sequences,
    clean_balabit,
    clean_chaoshen,
    clean_dfl,
    draw_srp,
    DEFAULT_EPSILON,
)

GLOBAL_MAX_X = 1919.0
GLOBAL_MAX_Y = 1079.0

DEFAULT_TRAINING_ROOT = {
    "balabit": "Data/Balabit-dataset/training_files",
    "chaoshen": "Data/ChaoShen/training_files",
    "dfl": "Data/DFL-dataset_raw/training_files",
}


def _clean_df(dataset, df):
    if dataset == "balabit":
        return clean_balabit(df)
    if dataset == "chaoshen":
        return clean_chaoshen(df)
    if dataset == "dfl":
        return clean_dfl(df)
    raise ValueError(dataset)


def build_user_max_xy_from_training(dataset, training_root):
    user_max_xy = {}
    users = sorted(os.listdir(training_root))

    for user in users:
        user_dir = os.path.join(training_root, user)
        if not os.path.isdir(user_dir):
            continue

        max_x = 0.0
        max_y = 0.0
        saw_points = False

        for name in sorted(os.listdir(user_dir)):
            path = os.path.join(user_dir, name)
            if not os.path.isfile(path):
                continue

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

    return user_max_xy


def process_dataset(dataset, data_root, out_dir, epsilon, user_max_xy, output_size=0):
    users = sorted(os.listdir(data_root))
    sequence_lengths = []

    print("\nDataset:", dataset)
    print("Users in data_root:", len(users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")

    for user in users:
        user_dir = os.path.join(data_root, user)
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
        print("User:", user, "| merge bounds W×H (from training):", norm_x, norm_y)

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

            sequences = split_by_time(events)
            print("      After split:", len(sequences))

            # Keep same per-user merge policy as XYPlot_per_user.
            min_length = norm_x
            sequences = merge_sequences(sequences, min_length)
            print("      After merge:", len(sequences))
            sequence_lengths.extend([len(seq) for seq in sequences])

            for i, seq in enumerate(sequences):
                save_path = os.path.join(
                    out_dir,
                    "Chong_per_user",
                    user,
                    f"{session}-{i}.png",
                )
                seq_array = np.array(
                    [[float(e["x"]), float(e["y"]), float(e["time"])] for e in seq],
                    dtype=np.float32,
                )
                draw_srp(seq_array, save_path, epsilon, output_size)

    print("\n========== Sequence Length Stats (After Merge) ==========")
    if len(sequence_lengths) == 0:
        print("No valid sequence generated.")
    else:
        lengths = np.array(sequence_lengths, dtype=np.float64)
        print("Total sequences:", len(lengths))
        print("min:", int(np.min(lengths)))
        print("median:", float(np.median(lengths)))
        print("average:", float(np.mean(lengths)))
        print("max:", int(np.max(lengths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl"])
    parser.add_argument(
        "--training_root",
        default=None,
        help="Relative to ROOT; default follows --dataset (Balabit/ChaoShen/DFL training_files).",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Sessions to render (train or test), relative to ROOT.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="若 > 0，用 transforms.Resize 将每张 SRP 存为 output_size×output_size PNG；0 表示保持原始 N×N。",
    )
    args = parser.parse_args()

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = os.path.join(ROOT, training_rel)
    print("[training_root]", training_rel)
    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    user_max_xy = build_user_max_xy_from_training(args.dataset, training_root)

    print("\nUSER_MAX_XY (from training_root):")
    for u in sorted(user_max_xy.keys()):
        print("  ", u, "->", user_max_xy[u])

    process_dataset(
        args.dataset, data_root, out_dir, args.epsilon, user_max_xy, args.output_size
    )
    print("\nPer-user SRP image generation finished.")


if __name__ == "__main__":
    main()
