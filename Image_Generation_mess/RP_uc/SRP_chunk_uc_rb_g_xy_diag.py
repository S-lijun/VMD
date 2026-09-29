# -*- coding: utf-8 -*-
"""
R=B=SRP with global-diag brightness; G=position stripes.

  R/B (contrast with local SRP_chunk):
    OLD: chunk-local coord scale -> dist -> per-trajectory dist min-max -> brightness
    NEW:
      1) global coord norm by user diag:
           (x',y') = ((x-min_x)/diag, (y-min_y)/diag)
         diag = sqrt((max_x-min_x)^2+(max_y-min_y)^2) from training bounds
      2) dist = pairwise ||p'_i-p'_j||   # already diag-units, corner-to-corner = 1
      3) brightness = min(dist, epsilon)  # NO local dist min-max stretch

  G (same as SRP_chunk_uc_rb_g_xy):
    lower (i > j): (x[j]-min_x)/(max_x-min_x)
    upper (i < j): (y[j]-min_y)/(max_y-min_y)
    diagonal: 0.5 * (x_norm + y_norm)

Usage:
  python SRP_chunk_uc_rb_g_xy_diag.py --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files \\
    --out_dir Images/Balabit/SRP_uc_rb_g_xy_diag --sizes 125
"""

from __future__ import print_function

import argparse
import os

import cv2
import numpy as np

from srp_uc_common import (
    DEFAULT_SCAN_ROOT,
    N_FOLDS,
    count_windows,
    default_bounds_json,
    get_or_scan_bounds,
    get_user_bounds,
    is_skipped_user_dir,
    list_session_csvs,
    list_users,
    load_events,
    natural_key,
    render_srp_rb_g_xy_diag,
    resolve_path,
    rgb_to_tensor_chw,
    session_window_groups,
)


def process_dataset_tensors(
    dataset, data_root, out_dir, sizes, epsilon, output_size, bounds, five_fold=False
):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Mode: R=B=SRP global-diag coord norm (no local dist stretch), G=xy stripes")
    print("Per-user bounds loaded for", len(bounds.get("users", {})), "users.")
    if five_fold:
        print("\n[Phase] Generating SRP RGB (R=B=global-diag, G=xy) tensors, {} contiguous folds per session...".format(N_FOLDS))
    else:
        print("\n[Phase] Generating SRP RGB (R=B=global-diag, G=xy) tensors...")

    for chunk_size in sizes:
        H = int(output_size) if output_size and int(output_size) > 0 else chunk_size
        W = H
        total_samples, _ = count_windows(dataset, data_root, chunk_size, five_fold=five_fold)
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
            user_bounds = get_user_bounds(bounds, user)
            print("\n------------------------------")
            print(
                "User:", user,
                "| min=({}, {}) max=({}, {}) diag={}".format(
                    user_bounds["min_x"], user_bounds["min_y"],
                    user_bounds["max_x"], user_bounds["max_y"],
                    user_bounds["diag"],
                ),
            )

            for file in list_session_csvs(user_dir):
                path = os.path.join(user_dir, file)
                session = os.path.splitext(file)[0]
                events = load_events(dataset, path)
                groups = session_window_groups(events, chunk_size, data_root, five_fold)
                n_windows = sum(len(windows) for _, windows in groups)
                print("  Session {} | chunk={} -> {} windows".format(
                    session, chunk_size, n_windows))

                for fold, windows in groups:
                    for seq in windows:
                        img = render_srp_rb_g_xy_diag(
                            seq, epsilon, output_size, user_bounds
                        )
                        if img is None:
                            continue
                        if img.shape[:2] != (H, W):
                            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

                        images[idx] = rgb_to_tensor_chw(img)
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
        np.save(os.path.join(tensor_root, "sessions.npy"), np.array(sessions, dtype=object))
        print("\nTensor dataset saved to: {} (wrote {} samples)".format(tensor_root, idx))


def process_dataset(
    dataset, data_root, out_dir, sizes, epsilon, output_size, bounds,
    tensors=False, five_fold=False,
):
    if tensors:
        process_dataset_tensors(
            dataset, data_root, out_dir, sizes, epsilon, output_size, bounds,
            five_fold=five_fold,
        )
        return

    users = list_users(data_root)
    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Mode: R=B=SRP global-diag coord norm (no local dist stretch), G=xy stripes")
    print("Per-user bounds loaded for", len(bounds.get("users", {})), "users.")
    print("\n[Phase] Generating SRP RGB (R=B=global-diag, G=xy) PNGs...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        user_bounds = get_user_bounds(bounds, user)
        print("\n------------------------------")
        print(
            "User:", user,
            "| min=({}, {}) max=({}, {}) diag={}".format(
                user_bounds["min_x"], user_bounds["min_y"],
                user_bounds["max_x"], user_bounds["max_y"],
                user_bounds["diag"],
            ),
        )

        for file in list_session_csvs(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            events = load_events(dataset, path)

            for chunk_size in sizes:
                groups = session_window_groups(events, chunk_size, data_root, five_fold)
                n_windows = sum(len(windows) for _, windows in groups)
                print("  Session {} | chunk={} -> {} windows".format(
                    session, chunk_size, n_windows))

                for fold, windows in groups:
                    for i, seq in enumerate(windows):
                        img = render_srp_rb_g_xy_diag(
                            seq, epsilon, output_size, user_bounds
                        )
                        if img is None:
                            continue
                        parts = [out_dir]
                        if fold is not None:
                            parts.append("fold{}".format(fold))
                        parts.extend([
                            "event{}".format(chunk_size),
                            user,
                            "{}-{}.png".format(session, i),
                        ])
                        save_path = os.path.join(*parts)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "R=B=SRP with global-diag coord normalization "
            "(no per-trajectory dist min-max); G=xy position stripes."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl", "twos"])
    parser.add_argument(
        "--data_root",
        required=True,
        help="要画图的数据根目录（training 或 testing）；不影响 min/max 统计。",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[125])
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="亮度上限：先按 diag 归一化坐标再算 dist，srp=min(dist, epsilon)。默认 1.0。",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="若 > 0，Resize 为 output_size×output_size；0 表示保持 N×N。",
    )
    parser.add_argument(
        "--scan_root",
        default=None,
        help="扫描 per-user min/max；默认 training_files。生成 testing 时不要改。",
    )
    parser.add_argument(
        "--bounds_json",
        default=None,
        help="默认 RP_uc/bounds/<dataset>_xy_bounds.json。",
    )
    parser.add_argument(
        "--rescan_bounds",
        action="store_true",
        default=False,
        help="强制用 scan_root 重扫 bounds JSON。",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="输出 images.npy / labels.npy / sessions.npy。",
    )
    parser.add_argument(
        "--five-fold",
        action="store_true",
        default=False,
        help="每个 session 按事件顺序切成 5 段连续事件再开窗。tensors 时多写 folds.npy，取值 0–4。",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    scan_rel = args.scan_root or DEFAULT_SCAN_ROOT[args.dataset]
    scan_root = resolve_path(scan_rel)
    bounds_json = (
        resolve_path(args.bounds_json) if args.bounds_json
        else default_bounds_json(args.dataset)
    )

    print("Mode: R=B=SRP global-diag coord norm, G=xy stripes")
    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("Resolved scan_root:", scan_root)
    print("Bounds JSON:", bounds_json)
    skipped = sorted(
        [
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name)) and is_skipped_user_dir(name)
        ],
        key=natural_key,
    )
    if skipped:
        print("Excluded dirs:", ", ".join(skipped))

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
            "min=({}, {}) max=({}, {}) diag={}".format(
                ub["min_x"], ub["min_y"], ub["max_x"], ub["max_y"], ub["diag"],
            ),
        )

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        epsilon=args.epsilon,
        output_size=args.output_size,
        bounds=bounds,
        tensors=args.tensors,
        five_fold=args.five_fold,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
