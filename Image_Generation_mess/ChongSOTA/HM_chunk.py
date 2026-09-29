# -*- coding: utf-8 -*-
"""
Thin trajectory heatmap (same drawing as HM.py), but sequences come from
pure fixed-size chunking (XYPlot_chunk.py): no time split, no merge.
"""

import os
import argparse

import pandas as pd

from HM import draw_heatmap
from XYPlot_chunk import (
    ROOT,
    GLOBAL_MAX_X,
    GLOBAL_MAX_Y,
    DEFAULT_TRAINING_ROOT,
    split_by_chunk_size,
    build_user_max_xy_from_training,
    _clean_df,
)


def process_dataset(dataset, data_root, out_dir, chunk_size, user_max_xy):

    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Per-user max bounds loaded for", len(user_max_xy), "users (from training_root).")
    print("Chunk size:", chunk_size)

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
                    "Heatmap_chunk",
                    user,
                    f"{session}-{i}.png",
                )

                draw_heatmap(seq, save_path, norm_x, norm_y)


def main():

    parser = argparse.ArgumentParser(
        description="HM-style heatmap from fixed-size event chunks (XYPlot_chunk logic).",
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
        help="Output root (relative to ROOT); images under Heatmap_chunk/<user>/.",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        default=120,
        help="Number of Move events per chunk (same as XYPlot_chunk --sizes).",
    )

    args = parser.parse_args()

    training_rel = args.training_root or DEFAULT_TRAINING_ROOT[args.dataset]
    training_root = os.path.join(ROOT, training_rel)
    print("[training_root]", training_rel)

    data_root = os.path.join(ROOT, args.data_root)
    out_dir = os.path.join(ROOT, args.out_dir)

    user_max_xy = build_user_max_xy_from_training(args.dataset, training_root)

    process_dataset(
        args.dataset,
        data_root,
        out_dir,
        args.sizes,
        user_max_xy,
    )

    print("\nHM chunk heatmap generation finished.")


if __name__ == "__main__":
    main()
