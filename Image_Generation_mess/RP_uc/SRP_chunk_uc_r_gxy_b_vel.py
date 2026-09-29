# -*- coding: utf-8 -*-
"""
R = SRP, G = position stripes, B = velocity stripes.

  R: pair-wise SRP (same as ChongSOTA/SRP_chunk)
  G: position vertical stripes (per-user min-max; supports negative coords)
       lower (i > j): (x[j]-min_x)/(max_x-min_x)
       upper (i < j): (y[j]-min_y)/(max_y-min_y)
       diagonal: 0.5 * (x_norm[i] + y_norm[i])
  B: |v| vertical stripes via global CDF (same encoding as
     ChongSOTA/TemporalEncoding/SRP_chunk_velocity.py blue channel)

  min/max: per-user training bounds JSON (RP_uc/bounds/).

Usage:
  python SRP_chunk_uc_r_gxy_b_vel.py --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files \\
    --velocity_dist Balabit_velocity_distribution_raw.npz \\
    --out_dir Images/Balabit/SRP_uc_r_gxy_b_vel --sizes 125
"""

from __future__ import print_function

import argparse
import os

import cv2
import numpy as np
from scipy.stats import rankdata

from srp_uc_common import (
    DEFAULT_SCAN_ROOT,
    count_windows,
    default_bounds_json,
    generate_windows,
    get_or_scan_bounds,
    get_user_bounds,
    list_session_csvs,
    list_users,
    load_events,
    natural_key,
    render_srp_r_gxy_b_vel,
    resolve_path,
    rgb_to_tensor_chw,
)


def load_raw_velocity_distribution(path):
    data = np.load(path)
    velocities = data["values"]
    print("\n[Velocity Distribution]")
    print("Samples:", len(velocities))
    print("Min:", velocities.min())
    print("Max:", velocities.max())
    return velocities


def build_runtime_cdf(raw_v, clip_pct):
    print("\nBuilding velocity runtime CDF")
    v_upper = np.percentile(raw_v, clip_pct)
    v_clipped = raw_v[raw_v <= v_upper]
    ranks = rankdata(v_clipped, method="average")
    cdf = (ranks - 1) / (len(v_clipped) - 1 + 1e-8)
    order = np.argsort(v_clipped)
    v_sorted = v_clipped[order]
    cdf_sorted = cdf[order]
    print("Runtime samples:", len(v_sorted))
    print("Runtime max:", v_sorted.max())
    return v_sorted, cdf_sorted


def process_dataset_tensors(
    dataset, data_root, out_dir, sizes, epsilon, output_size, bounds, v_cdf
):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    print("Mode: R=SRP, G=xy stripes, B=|v| stripes")
    print("Per-user bounds loaded for", len(bounds.get("users", {})), "users.")
    print("\n[Phase] Generating SRP RGB (R/G_xy/B_vel) tensors...")

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
        idx = 0

        for user in users:
            user_dir = os.path.join(data_root, user)
            user_bounds = get_user_bounds(bounds, user)
            print("\n------------------------------")
            print(
                "User:", user,
                "| max_x={}, max_y={}".format(
                    user_bounds["max_x"], user_bounds["max_y"]
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
                    img = render_srp_r_gxy_b_vel(
                        seq, epsilon, output_size, user_bounds, v_cdf
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
                    idx += 1

        images.flush()
        labels.flush()
        np.save(os.path.join(tensor_root, "sessions.npy"), np.array(sessions, dtype=object))
        print("\nTensor dataset saved to: {} (wrote {} samples)".format(tensor_root, idx))


def process_dataset(
    dataset, data_root, out_dir, sizes, epsilon, output_size, bounds, v_cdf,
    tensors=False,
):
    if tensors:
        process_dataset_tensors(
            dataset, data_root, out_dir, sizes, epsilon, output_size, bounds, v_cdf
        )
        return

    users = list_users(data_root)
    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("Mode: R=SRP, G=xy stripes, B=|v| stripes")
    print("Per-user bounds loaded for", len(bounds.get("users", {})), "users.")
    print("\n[Phase] Generating SRP RGB (R/G_xy/B_vel) PNGs...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        user_bounds = get_user_bounds(bounds, user)
        print("\n------------------------------")
        print(
            "User:", user,
            "| max_x={}, max_y={}".format(
                user_bounds["max_x"], user_bounds["max_y"]
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
                    img = render_srp_r_gxy_b_vel(
                        seq, epsilon, output_size, user_bounds, v_cdf
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
                    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(
        description="R=SRP; G=xy position stripes; B=|v| velocity stripes."
    )
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl", "twos"])
    parser.add_argument(
        "--data_root",
        required=True,
        help="要画图的数据根目录（training 或 testing）；不影响 min/max 统计。",
    )
    parser.add_argument(
        "--velocity_dist",
        required=True,
        help="npz with values array (e.g. Balabit_velocity_distribution_raw.npz).",
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
        "--v_percentile",
        type=float,
        default=100,
        help="Upper clip percentile for speed CDF.",
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
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)
    dist_path = resolve_path(args.velocity_dist)
    scan_rel = args.scan_root or DEFAULT_SCAN_ROOT[args.dataset]
    scan_root = resolve_path(scan_rel)
    bounds_json = (
        resolve_path(args.bounds_json) if args.bounds_json
        else default_bounds_json(args.dataset)
    )

    print("Mode: R=SRP, G=xy stripes, B=|v| stripes")
    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    print("[velocity_dist]", dist_path)
    print("Resolved scan_root:", scan_root)
    print("Bounds JSON:", bounds_json)

    raw_v = load_raw_velocity_distribution(dist_path)
    v_cdf = build_runtime_cdf(raw_v, args.v_percentile)

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
            "max_x={}, max_y={}".format(ub["max_x"], ub["max_y"]),
        )

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        epsilon=args.epsilon,
        output_size=args.output_size,
        bounds=bounds,
        v_cdf=v_cdf,
        tensors=args.tensors,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
