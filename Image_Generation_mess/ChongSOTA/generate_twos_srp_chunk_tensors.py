# -*- coding: utf-8 -*-
"""
One-click TWOS tensor generation for the three RP_uc global-diag SRP
halves representations:

  SRP_chunk_uc_rb_g_xy_diag_halves        R=B=diag SRP, G=xy halves
  SRP_chunk_uc_r_gxy_b_vel_diag_halves    R=diag SRP, G=xy halves, B=|v|
  SRP_chunk_uc_r_gxy_b_vxvy_diag_halves   R=diag SRP, G=xy halves, B=vx/vy halves

Each representation writes training + protocol1 testing under ImagesTensors/TWOS/:

  ImagesTensors/TWOS/SRP_chunk_uc_rb_g_xy_diag_halves/event125/
  ImagesTensors/TWOS/SRP_chunk_uc_rb_g_xy_diag_halves_protocol1/event125/
  ImagesTensors/TWOS/SRP_chunk_uc_r_gxy_b_vel_diag_halves/event125/
  ImagesTensors/TWOS/SRP_chunk_uc_r_gxy_b_vel_diag_halves_protocol1/event125/
  ImagesTensors/TWOS/SRP_chunk_uc_r_gxy_b_vxvy_diag_halves/event125/
  ImagesTensors/TWOS/SRP_chunk_uc_r_gxy_b_vxvy_diag_halves_protocol1/event125/

Per-user min/max/diag always scanned from Data/TWOS/training_files.
Velocity / vxvy CDFs are rebuilt from training_files unless --skip-dist.

Usage:
  python Image_Generation/ChongSOTA/generate_twos_srp_chunk_tensors.py
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

TRAIN_ROOT = "Data/TWOS/training_files"
TEST_ROOT = "Data/TWOS/testing_files_protocol1"
OUT_BASE = "ImagesTensors/TWOS"

VEL_NPZ = "TWOS_velocity_distribution_raw.npz"
VXVY_NPZ = "TWOS_vxvy_distribution_raw.npz"

REPR_XY = "SRP_chunk_uc_rb_g_xy_diag_halves"
REPR_VEL = "SRP_chunk_uc_r_gxy_b_vel_diag_halves"
REPR_VXVY = "SRP_chunk_uc_r_gxy_b_vxvy_diag_halves"

SCRIPTS = {
    REPR_XY: os.path.join(
        "Image_Generation", "RP_uc", "SRP_chunk_uc_rb_g_xy_diag_halves.py"
    ),
    REPR_VEL: os.path.join(
        "Image_Generation", "RP_uc", "SRP_chunk_uc_r_gxy_b_vel_diag_halves.py"
    ),
    REPR_VXVY: os.path.join(
        "Image_Generation", "RP_uc", "SRP_chunk_uc_r_gxy_b_vxvy_diag_halves.py"
    ),
}

SPLITS = {
    "training": TRAIN_ROOT,
    "testing": TEST_ROOT,
}


def out_folder(repr_name, split):
    """training -> {repr}; testing -> {repr}_protocol1."""
    if split == "testing":
        return "%s_protocol1" % repr_name
    return repr_name


def run(cmd, dry_run):
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=ROOT)


def build_distributions(python, skip_dist, dry_run):
    dist_script = os.path.join("Image_Generation", "build_global_distribution.py")
    jobs = [
        ("velocity", VEL_NPZ),
        ("vxvy", VXVY_NPZ),
    ]
    for feature, out_name in jobs:
        out_path = os.path.join(ROOT, out_name)
        if skip_dist and os.path.isfile(out_path):
            print("[skip-dist] using existing", out_path)
            continue
        run(
            [
                python,
                dist_script,
                "--dataset", "twos",
                "--feature", feature,
                "--training_root", TRAIN_ROOT,
                "--out_dir", out_name,
            ],
            dry_run,
        )


def generate_split(python, repr_name, split, sizes, epsilon, output_size, dry_run):
    script = SCRIPTS[repr_name]
    data_root = SPLITS[split]
    out_dir = os.path.join(OUT_BASE, out_folder(repr_name, split))

    cmd = [
        python,
        script,
        "--dataset", "twos",
        "--data_root", data_root,
        "--scan_root", TRAIN_ROOT,
        "--out_dir", out_dir,
        "--sizes", *[str(s) for s in sizes],
        "--epsilon", str(epsilon),
        "--output_size", str(output_size),
        "--tensors",
    ]
    if repr_name == REPR_VEL:
        cmd.extend(["--velocity_dist", VEL_NPZ])
    elif repr_name == REPR_VXVY:
        cmd.extend(["--velocity_dist", VXVY_NPZ])

    run(cmd, dry_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TWOS RP_uc diag-halves SRP tensors (train+test).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[125],
        help="Event chunk sizes (default: 125).",
    )
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="Resize each SRP to output_size x output_size (default: 448).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[REPR_XY, REPR_VEL, REPR_VXVY],
        default=[REPR_XY, REPR_VEL, REPR_VXVY],
        help="Subset of representations to generate.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["training", "testing"],
        default=["training", "testing"],
    )
    parser.add_argument(
        "--skip-dist",
        action="store_true",
        help="Do not rebuild TWOS_velocity / TWOS_vxvy npz; reuse files at repo root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    python = sys.executable

    print("[ROOT]", ROOT)
    print("[train]", os.path.join(ROOT, TRAIN_ROOT))
    print("[test ]", os.path.join(ROOT, TEST_ROOT))
    print("[out  ]", os.path.join(ROOT, OUT_BASE))
    print("[repr ]", args.only)
    print("[split]", args.splits)

    need_dist = (REPR_VEL in args.only) or (REPR_VXVY in args.only)
    if need_dist:
        build_distributions(python, args.skip_dist, args.dry_run)

    for repr_name in args.only:
        for split in args.splits:
            print("\n" + "=" * 72)
            print("Generating", repr_name, "/", split)
            print("=" * 72)
            generate_split(
                python,
                repr_name,
                split,
                args.sizes,
                args.epsilon,
                args.output_size,
                args.dry_run,
            )

    print("\nDone. Tensor folders:")
    for repr_name in args.only:
        for split in args.splits:
            for chunk_size in args.sizes:
                path = os.path.join(
                    ROOT, OUT_BASE, out_folder(repr_name, split), "event%d" % chunk_size
                )
                print(" ", path)


if __name__ == "__main__":
    main()
